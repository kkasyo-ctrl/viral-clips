"""Paths and tunables for the pipeline. Anything read with os.getenv can be overridden in .env."""
import os
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
CAPTION_FONT = os.getenv("CAPTION_FONT", "Arial")
CAPTION_HIGHLIGHT = os.getenv("CAPTION_HIGHLIGHT", "#FFE600")
CAPTION_MIN_WORDS = 2
CAPTION_MAX_WORDS = 4
CAPTION_MAX_LINE_SECONDS = 1.8
CAPTION_BASELINE = 0.72  # fraction of frame height

# 8. Render
RENDER_CRF = _int("RENDER_CRF", 17)
RENDER_PRESET = os.getenv("RENDER_PRESET", "slow")
RENDER_AUDIO_BITRATE = os.getenv("RENDER_AUDIO_BITRATE", "256k")
