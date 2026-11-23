import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import imageio_ffmpeg
import yt_dlp
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from google import genai
from google.genai import types
from groq import Groq
from pydantic import BaseModel, Field
from yt_dlp.utils import download_range_func

ROOT = Path(__file__).resolve().parent
TMP = Path(tempfile.gettempdir())

MIN_LEN, MAX_LEN = 15.0, 60.0
MAX_VIDEO_SECONDS = 7200
CHUNK_SECONDS = 1200
VIDEO_FORMAT = "bv*[height<=1080]+ba/b[height<=1080]/b"
YT_URL = re.compile(r"^https?://((www|m|music)\.)?(youtube\.com|youtu\.be)/", re.I)


class ProcessRequest(BaseModel):
    url: str
    clip_count: int = Field(default=3, ge=1, le=10)


class Clip(BaseModel):
    start_time: float
    end_time: float
    title: str
    virality_score: int


# ---------- environment ----------

def ffmpeg_bin() -> str:
    dst = TMP / "ffmpeg"
    if not dst.exists():
        shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), dst)
        dst.chmod(dst.stat().st_mode | stat.S_IEXEC)
    return str(dst)


def cookie_file() -> str | None:
    dst = TMP / "cookies.txt"
    if os.getenv("YTDLP_COOKIES"):
        dst.write_text(os.environ["YTDLP_COOKIES"])
    elif (ROOT / "cookies.txt").exists():
        shutil.copy(ROOT / "cookies.txt", dst)
    else:
        return None
    return str(dst)


def ydl_opts(**extra) -> dict:
    opts = {
        "ffmpeg_location": ffmpeg_bin(),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cachedir": str(TMP / "yt-dlp-cache"),
    }
    cookies = cookie_file()
    if cookies:
        opts["cookiefile"] = cookies
    opts.update(extra)
    return opts


def run(cmd: list[str], label: str) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{label} failed: {p.stderr.strip()[-400:]}")


def require_keys() -> None:
    missing = [k for k in ("GROQ_API_KEY", "GEMINI_API_KEY") if not os.getenv(k)]
    if missing:
        raise HTTPException(500, f"Missing environment variables: {', '.join(missing)}")


def require_url(url: str) -> str:
    url = url.strip()
    if not YT_URL.match(url):
        raise HTTPException(400, "Enter a valid YouTube URL.")
    return url


# ---------- stage 1-4: audio, transcript, viral moments ----------

def too_long(info, *, incomplete=False):
    if (info.get("duration") or 0) > MAX_VIDEO_SECONDS:
        return "Video is too long."
    return None


def download_audio(url: str, work: Path) -> Path:
    opts = ydl_opts(format="bestaudio", outtmpl=str(work / "source.%(ext)s"), match_filter=too_long)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    src = next(work.glob("source.*"), None)
    if not src:
        raise RuntimeError("Video is longer than 2 hours or could not be downloaded.")
    return src


def split_audio(src: Path, work: Path) -> list[Path]:
    run(
        [
            ffmpeg_bin(), "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "libmp3lame", "-b:a", "32k", "-f", "segment",
            "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
            str(work / "chunk_%03d.mp3"),
        ],
        "ffmpeg audio split",
    )
    chunks = sorted(work.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("Audio split produced no chunks.")
    return chunks


def transcribe_chunk(path: Path, offset: float) -> list[dict]:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.audio.transcriptions.create(
        file=(path.name, path.read_bytes()),
        model="whisper-large-v3",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    data = resp if isinstance(resp, dict) else resp.model_dump()
    out = []
    for s in data.get("segments") or []:
        text = (s.get("text") or "").strip()
        if text:
            out.append({"start": s["start"] + offset, "end": s["end"] + offset, "text": text})
    return out


def transcribe(chunks: list[Path]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
        parts = list(pool.map(lambda a: transcribe_chunk(a[1], a[0] * CHUNK_SECONDS), enumerate(chunks)))
    return [seg for part in parts for seg in part]


def find_clips(segments: list[dict], count: int) -> list[Clip]:
    transcript = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    duration = segments[-1]["end"]
    prompt = (
        f"You are a short-form video editor. Below is a timestamped transcript (seconds) of a "
        f"{duration:.0f}s video. Pick the {count + 3} best self-contained segments for viral "
        "9:16 Shorts: strong opening hook, controversial or surprising claims, emotional peaks, "
        "or punchy insights that make sense without context. Each segment must be between 15 and "
        "60 seconds long, start at the beginning of a sentence and end at the end of one, and "
        "must not overlap other segments. Titles must be punchy, under 70 characters. "
        "virality_score is an integer 1-100.\n\nTRANSCRIPT:\n" + transcript
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Clip],
            temperature=0.3,
        ),
    )
    candidates = [Clip(**c) for c in json.loads(resp.text)]
    chosen: list[Clip] = []
    for c in sorted(candidates, key=lambda c: c.virality_score, reverse=True):
        start = max(0.0, c.start_time)
        end = min(c.end_time, start + MAX_LEN, duration)
        if end - start < MIN_LEN:
            continue
        if any(start < o.end_time and end > o.start_time for o in chosen):
            continue
        chosen.append(Clip(start_time=start, end_time=end, title=c.title.strip()[:90],
                           virality_score=max(1, min(100, c.virality_score))))
        if len(chosen) == count:
            break
    if not chosen:
        raise RuntimeError("No valid 15-60s segments were found in this video.")
    return chosen


def analyze(url: str, count: int) -> Iterator[str]:
    def emit(**kw) -> str:
        return json.dumps(kw) + "\n"

    work = Path(tempfile.mkdtemp(prefix="shorts_"))
    try:
        yield emit(stage="downloading")
        src = download_audio(url, work)
        chunks = split_audio(src, work)

        yield emit(stage="transcribing")
        segments = transcribe(chunks)
        if not segments:
            raise RuntimeError("No speech detected in this video.")

        yield emit(stage="analyzing")
        clips = find_clips(segments, count)
        yield emit(stage="done", url=url, clips=[c.model_dump() for c in clips])
    except Exception as exc:
        yield emit(stage="error", error=str(exc))
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- stage 5: cut + crop one clip ----------

def render_clip(url: str, start: float, end: float, work: Path) -> Path:
    opts = ydl_opts(
        format=VIDEO_FORMAT,
        outtmpl=str(work / "raw.%(ext)s"),
        merge_output_format="mp4",
        download_ranges=download_range_func(None, [(start, end)]),
        force_keyframes_at_cuts=True,
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    raw = next(work.glob("raw.*"), None)
    if not raw:
        raise RuntimeError("Segment download produced no file.")
    out = work / "clip.mp4"
    run(
        [
            ffmpeg_bin(), "-y", "-i", str(raw),
            "-vf", "crop='trunc(ih*9/16/2)*2':ih,scale=1080:1920,setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
        ],
        "ffmpeg crop",
    )
    raw.unlink(missing_ok=True)
    return out


# ---------- app ----------

app = FastAPI(title="Shorts Maker")
PAGE = ROOT / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))


@app.post("/api/process")
def process(req: ProcessRequest) -> StreamingResponse:
    url = require_url(req.url)
    require_keys()
    return StreamingResponse(
        analyze(url, req.clip_count),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/clip")
def clip(url: str, start: float = Query(ge=0), end: float = Query(gt=0)) -> StreamingResponse:
    url = require_url(url)
    if not (MIN_LEN - 0.5 <= end - start <= MAX_LEN + 0.5):
        raise HTTPException(400, "Clip length must be between 15 and 60 seconds.")
    work = Path(tempfile.mkdtemp(prefix="clip_"))
    try:
        out = render_clip(url, start, end, work)
    except Exception as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, str(exc))

    def stream() -> Iterator[bytes]:
        try:
            with open(out, "rb") as f:
                while chunk := f.read(1 << 20):
                    yield chunk
        finally:
            shutil.rmtree(work, ignore_errors=True)

    return StreamingResponse(stream(), media_type="video/mp4", headers={"Cache-Control": "no-store"})
