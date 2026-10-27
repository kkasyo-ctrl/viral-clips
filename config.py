"""Paths and tunables for the pipeline. Anything read with os.getenv can be overridden in .env."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return value.strip().lower() in {"1", "true", "yes", "on"} if value else default


def _ratio(name: str, default: tuple[int, int]) -> tuple[int, int]:
    """'9:16' or '9/16' -> (9, 16)."""
    value = os.getenv(name)
    if not value:
        return default
    width, _, height = value.replace("/", ":").partition(":")
    return int(width), int(height)


def _default_caption_font() -> str:
    """A font family that is installed by default on this OS, so libass never has to
    guess at a missing one and silently burn the captions in some fallback face."""
    if sys.platform == "win32":
        return "Arial"
    if sys.platform == "darwin":
        return "Helvetica"
    return "DejaVu Sans"


# Folders
DATA_DIR = ROOT / "Data"
INPUT_DIR = DATA_DIR / "Input_video"
TRANSCRIPTS_DIR = DATA_DIR / "Transcripts"
CLIPS_DIR = DATA_DIR / "clips"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

# External tools
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# 1. Audio extraction + silence detection
AUDIO_SAMPLE_RATE = 16_000
SILENCE_NOISE_DB = _float("SILENCE_NOISE_DB", -30.0)
SILENCE_MIN_SECONDS = 0.3

# 2. Transcription (OpenRouter by default, ElevenLabs Scribe still selectable)
TRANSCRIBE_PROVIDER = os.getenv("TRANSCRIBE_PROVIDER", "openrouter").strip().lower()
if TRANSCRIBE_PROVIDER not in {"openrouter", "elevenlabs"}:
    raise SystemExit(f"TRANSCRIBE_PROVIDER must be 'openrouter' or 'elevenlabs', not {TRANSCRIBE_PROVIDER!r}")
_OPENROUTER = TRANSCRIBE_PROVIDER == "openrouter"

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL") or (
    "microsoft/mai-transcribe-2" if _OPENROUTER else "scribe_v2"
)
# Reframing needs per-word speaker IDs, so diarization is on. It is part of the
# transcript cache key: an older transcript taken without it has no speakers to read.
TRANSCRIBE_DIARIZE = _bool("TRANSCRIBE_DIARIZE", True)
TRANSCRIBE_LANGUAGE = os.getenv("TRANSCRIBE_LANGUAGE") or None  # None = auto-detect

# How much audio goes up in one request. Scribe takes the lot (3 GB / 10 h), so it is
# only chunked for very long videos. OpenRouter caps a request at 25 MB, so it gets short
# chunks: the pipeline's 16 kHz mono WAV runs ~1.9 MB/minute, which puts an 8-minute chunk
# at ~15 MB and the longest plan_chunks will stretch to (1.25x) at ~19 MB. The audio is
# uploaded as WAV because mai-transcribe-2 rejects FLAC upstream.
OPENROUTER_MAX_REQUEST_BYTES = _int("OPENROUTER_MAX_REQUEST_BYTES", 25_000_000)
# Which upstream slug carries the diarization option (mai-transcribe-2 routes via Azure).
OPENROUTER_PROVIDER_SLUG = os.getenv("OPENROUTER_PROVIDER_SLUG", "azure")

TRANSCRIBE_CHUNK_SECONDS = _float("TRANSCRIBE_CHUNK_SECONDS", 8 * 60 if _OPENROUTER else 30 * 60)
TRANSCRIBE_MAX_SECONDS = _float("TRANSCRIBE_MAX_SECONDS", TRANSCRIBE_CHUNK_SECONDS if _OPENROUTER else 9.5 * 3600)
TRANSCRIBE_MAX_BYTES = _int(
    "TRANSCRIBE_MAX_BYTES",
    int(TRANSCRIBE_MAX_SECONDS * AUDIO_SAMPLE_RATE * 2) if _OPENROUTER else 2_800_000_000,
)
TRANSCRIBE_SPLIT_SEARCH_SECONDS = 90  # how far either side of a chunk mark to look for a silence

# 3. Moment extraction (OpenAI)
MOMENTS_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
MOMENTS_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT") or "medium"  # None = model default
MOMENTS_MAX_INPUT_TOKENS = _int("OPENAI_MAX_INPUT_TOKENS", 900_000)
MOMENTS_WINDOW_SECONDS = 8 * 60
MOMENTS_STRIDE_SECONDS = 6 * 60

# 4. Virality probabilities (Jev)
JEV_MODEL = os.getenv("JEV_MODEL") or None  # None = SDK default (jev-latest)
JEV_CONCURRENCY = _int("JEV_CONCURRENCY", 16)

# 5. Threshold
PROB_THRESHOLD = _float("PROB_THRESHOLD", 0.6)

# 6. Cut points
START_PAD_SECONDS = 0.15
END_PAD_SECONDS = _float("END_PAD_SECONDS", 1.0)  # tail after the last word; the audio fades out over it
MIN_CLIP_SECONDS = 15.0
MAX_CLIP_SECONDS = 90.0
PAUSE_MIN_SECONDS = 0.4
START_SEARCH_SECONDS = 4.0  # how far into a moment the start may move to land after a pause...
START_FRAGMENT_WORDS = 3  # ...stepping only over sentences this short
FILLER_OPENERS = {"and", "so", "but", "um", "umm", "uh", "uhh", "yeah", "right", "like", "okay", "ok"}

# 7. Captions
CAPTION_FONT = os.getenv("CAPTION_FONT") or _default_caption_font()
CAPTION_HIGHLIGHT = os.getenv("CAPTION_HIGHLIGHT", "#FFE600")
CAPTION_MIN_WORDS = 2
CAPTION_MAX_WORDS = 4
CAPTION_MAX_LINE_SECONDS = 1.8
CAPTION_BASELINE = 0.72  # fraction of frame height, landscape only
# Vertical frames instead keep the captions clear of the UI every short-form platform
# draws over the video: the caption block sits on top of the bottom safe margin, which
# puts it in the lower third, and never reaches into the side button columns.
CAPTION_SAFE_BOTTOM = _float("CAPTION_SAFE_BOTTOM", 0.18)  # fraction of frame height
CAPTION_SAFE_SIDE = _float("CAPTION_SAFE_SIDE", 0.08)  # fraction of frame width

# 8. Reframing to vertical (local, no API calls)
REFRAME = _bool("REFRAME", True)
REFRAME_ASPECT_RATIO = _ratio("REFRAME_ASPECT_RATIO", (9, 16))
REFRAME_WIDTH = _int("REFRAME_WIDTH", 1080)
REFRAME_HEIGHT = _int("REFRAME_HEIGHT", 1920)
REFRAME_MIN_SEGMENT_SECONDS = _float("REFRAME_MIN_SEGMENT_SECONDS", 1.5)
REFRAME_SAMPLES_PER_SEGMENT = _int("REFRAME_SAMPLES_PER_SEGMENT", 13)
REFRAME_SCENE_THRESHOLD = _float("REFRAME_SCENE_THRESHOLD", 0.25)  # shortest scene, seconds
REFRAME_SCENE_MERGE_THRESHOLD = _float("REFRAME_SCENE_MERGE_THRESHOLD", 0.25)
REFRAME_DEVICE = os.getenv("REFRAME_DEVICE") or None  # None = auto (cuda, mps, else cpu)
REFRAME_SEED = _int("REFRAME_SEED", 0)  # frame sampling is random; fix it so re-runs match
REFRAME_FACE_DETECT_WIDTH = _int("REFRAME_FACE_DETECT_WIDTH", 960)
# Within a shot the crop follows the face: sampled this often, held while the face stays
# within the dead zone (a fraction of the crop width) of centre, else panned over PAN seconds.
REFRAME_TRACK_INTERVAL = _float("REFRAME_TRACK_INTERVAL", 0.25)
REFRAME_TRACK_DEADZONE = _float("REFRAME_TRACK_DEADZONE", 0.10)
REFRAME_PAN_SECONDS = _float("REFRAME_PAN_SECONDS", 0.4)


# 9. Render
RENDER_CRF = _int("RENDER_CRF", 17)
RENDER_PRESET = os.getenv("RENDER_PRESET", "slow")  # libx264 only
# auto = NVIDIA's GPU encoder (h264_nvenc) when it works on this machine, else libx264 on the CPU
RENDER_ENCODER = os.getenv("RENDER_ENCODER", "auto")
RENDER_AUDIO_BITRATE = os.getenv("RENDER_AUDIO_BITRATE", "256k")
