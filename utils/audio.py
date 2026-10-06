"""Step 1: pull a 16 kHz mono WAV out of the video and detect silences in the same ffmpeg pass."""
import hashlib
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import config

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def require_ffmpeg() -> None:
    for binary in (config.FFMPEG, config.FFPROBE):
        if shutil.which(binary) is None and not Path(binary).is_file():
            raise SystemExit(
                f"'{binary}' not found. Install ffmpeg (e.g. `winget install Gyan.FFmpeg`) "
                "or point FFMPEG_BIN / FFPROBE_BIN at the executables in .env."
            )


def extract_audio(video: Path, wav_out: Path) -> list[Silence]:
    """Write the first audio track as 16 kHz mono PCM and return the silence spans ffmpeg found."""
    cmd = [
        config.FFMPEG, "-hide_banner", "-nostats", "-y",
        "-i", str(video),
        "-map", "0:a:0",
        "-af", f"silencedetect=noise={config.SILENCE_NOISE_DB:g}dB:d={config.SILENCE_MIN_SECONDS:g}",
        "-ac", "1", "-ar", str(config.AUDIO_SAMPLE_RATE), "-c:a", "pcm_s16le",
        # bitexact keeps the WAV byte-identical across runs, so its hash is a stable cache key
        "-fflags", "+bitexact", "-flags:a", "+bitexact",
        str(wav_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{proc.stderr[-2000:]}")
    return parse_silences(proc.stderr, wav_duration(wav_out))


def parse_silences(ffmpeg_log: str, duration: float) -> list[Silence]:
    silences, start = [], None
    for line in ffmpeg_log.splitlines():
        if m := _SILENCE_START.search(line):
            start = max(0.0, float(m.group(1)))
        elif (m := _SILENCE_END.search(line)) and start is not None:
            silences.append(Silence(start, float(m.group(1))))
            start = None
    if start is not None:  # audio ends while still silent
        silences.append(Silence(start, duration))
    return silences


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slice_wav(src: Path, start: float, end: float, dst: Path) -> None:
    """Sample-exact copy of [start, end) seconds of a WAV file."""
    with wave.open(str(src), "rb") as reader, wave.open(str(dst), "wb") as writer:
        rate = reader.getframerate()
        writer.setparams(reader.getparams())
        first = round(start * rate)
        remaining = min(reader.getnframes(), round(end * rate)) - first
        reader.setpos(first)
        while remaining > 0:
            n = min(remaining, rate * 60)
            writer.writeframes(reader.readframes(n))
            remaining -= n
