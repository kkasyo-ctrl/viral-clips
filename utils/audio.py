"""Step 1: pull a 16 kHz mono WAV out of the video and detect silences in the same ffmpeg pass.
Loudness spikes are read from the same WAV afterwards."""
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


@dataclass(frozen=True)
class Peak:
    """A stretch noticeably louder than the video's usual level: shouting, laughing, a scream."""
    start: float
    end: float
    gain_db: float  # the loudest window, in dB over the video's median


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


def loudness_peaks(
    wav: Path, window: float | None = None, threshold_db: float | None = None
) -> list[Peak]:
    """Stretches of the WAV at least threshold_db louder than its median window.

    Read a minute at a time, so a stream several hours long never sits in memory whole.
    Silent windows are left out of the median: a stream with long quiet gaps would
    otherwise make ordinary talking look like shouting.
    """
    import numpy as np

    window = window or config.LOUDNESS_WINDOW_SECONDS
    threshold_db = config.LOUDNESS_SPIKE_DB if threshold_db is None else threshold_db
    levels = []
    with wave.open(str(wav), "rb") as reader:
        size = max(1, round(window * reader.getframerate()))
        while block := reader.readframes(size * 120):
            samples = np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768
            usable = len(samples) // size * size
            if usable:
                rms = np.sqrt(np.mean(samples[:usable].reshape(-1, size) ** 2, axis=1))
                levels.append(20 * np.log10(np.maximum(rms, 1e-6)))
    if not levels:
        return []
    db = np.concatenate(levels)
    voiced = db[db > -50]
    if not len(voiced):
        return []
    gain = db - float(np.median(voiced))

    peaks, start = [], None
    for i, g in enumerate(np.append(gain, -np.inf)):  # the sentinel closes a peak at the end
        if g >= threshold_db and start is None:
            start = i
        elif g < threshold_db and start is not None:
            peaks.append(Peak(round(start * window, 2), round(i * window, 2), round(float(gain[start:i].max()), 1)))
            start = None
    return peaks


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
