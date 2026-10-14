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

# 2. Transcription (ElevenLabs Scribe)
SCRIBE_MODEL = os.getenv("SCRIBE_MODEL", "scribe_v1")
# Reframing needs per-word speaker IDs, so diarization is on. It is part of the
# transcript cache key: an older transcript taken without it has no speakers to read.
SCRIBE_DIARIZE = _bool("SCRIBE_DIARIZE", True)
SCRIBE_LANGUAGE = os.getenv("SCRIBE_LANGUAGE") or None  # None = auto-detect
# Scribe accepts up to 3 GB / 10 h per request; stay a little under that.
SCRIBE_MAX_BYTES = _int("SCRIBE_MAX_BYTES", 2_800_000_000)
SCRIBE_MAX_SECONDS = _float("SCRIBE_MAX_SECONDS", 9.5 * 3600)
SCRIBE_CHUNK_SECONDS = 30 * 60
SCRIBE_SPLIT_SEARCH_SECONDS = 90  # how far either side of a 30-min mark to look for a silence

# 3. Moment extraction (OpenAI)
MOMENTS_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
MOMENTS_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT") or None  # None = model default
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
END_PAD_SECONDS = 0.35
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


# 9. Render
RENDER_CRF = _int("RENDER_CRF", 17)
RENDER_PRESET = os.getenv("RENDER_PRESET", "slow")
RENDER_AUDIO_BITRATE = os.getenv("RENDER_AUDIO_BITRATE", "256k")
