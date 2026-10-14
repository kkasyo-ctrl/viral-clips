"""Step 9: cut the clip out of the source and burn the captions in, in one encode.

One ffmpeg pass does everything that touches pixels: the cut, the reframing crop and
scale when there is one, and the captions. Burning captions in is not optional -- most
short-form viewers watch muted, and several platforms will not take a sidecar subtitle
file at all -- so the caption filter is always the last link in the chain, drawn onto
the frame the viewer actually sees. The .srt written next to the clip is a convenience
for anything that can use it, not the delivery mechanism.
"""
import functools
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    width: int  # as displayed, i.e. after any rotation metadata
    height: int
    frame_rate: float
    duration: float


def probe_video(video: Path) -> VideoInfo:
    """Displayed size, frame rate and duration of the first video stream."""
    proc = subprocess.run(
        [
            config.FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_streams", "-show_format", "-of", "json", str(video),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video.name}:\n{proc.stderr[-1000:]}")
    probed = json.loads(proc.stdout)
    stream = probed["streams"][0]

    width, height = int(stream["width"]), int(stream["height"])
    rotation = int(float(stream.get("tags", {}).get("rotate", 0)))
    for side_data in stream.get("side_data_list", []):
        rotation = int(float(side_data.get("rotation", rotation)))
    if rotation % 180:
        width, height = height, width

    duration = stream.get("duration") or probed.get("format", {}).get("duration") or 0.0
    return VideoInfo(width, height, _frame_rate(stream), float(duration))


def _frame_rate(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        numerator, _, denominator = (stream.get(key) or "").partition("/")
        try:
            rate = float(numerator) / float(denominator or 1)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return 30.0


@functools.cache
def crop_eval_option() -> str:
    """":eval=frame"" when this ffmpeg's crop filter needs it, else an empty string.

    A crop that follows the speaker has x and y as expressions in `t`. ffmpeg 9 dropped
    the `eval` option and reads those expressions every frame; older builds default to
    reading them once, which would freeze the crop on the first segment, and ffmpeg 9
    rejects the option outright. So ask this ffmpeg which it is.
    """
    proc = subprocess.run(
        [config.FFMPEG, "-hide_banner", "-h", "filter=crop"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode == 0 and not any(
        line.strip().startswith("eval ") for line in proc.stdout.splitlines()
    ):
        return ""
    # Including it on a build that has it is correct; a build that does not will say so
    # loudly, which beats silently rendering a clip whose crop never moves.
    return ":eval=frame"


def render_clip(
    video: Path,
    start: float,
    end: float,
    ass_file: Path,
    out_file: Path,
    video_filter: str = "",
) -> None:
    """Encode one clip. `video_filter` runs before the captions, which always go last.

    Re-encodes only as much as the work requires: without a filter that means the source
    resolution and frame rate at a near-lossless CRF.
    """
    _require_captions(ass_file)
    # Relative path on purpose: ffmpeg runs inside the clip folder, which avoids escaping
    # a Windows drive letter and backslashes inside the filter argument.
    chain = ",".join(part for part in (video_filter, f"ass={ass_file.name}") if part)
    cmd = [
        config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video.resolve()), "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", chain,
        "-c:v", "libx264", "-preset", config.RENDER_PRESET, "-crf", str(config.RENDER_CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", config.RENDER_AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(out_file.resolve()),
    ]
    proc = subprocess.run(cmd, cwd=ass_file.parent, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg render failed for {out_file.name}:\n{proc.stderr[-2000:]}")


def _require_captions(ass_file: Path) -> None:
    """Refuse to render a clip whose captions would be missing from the picture."""
    if not ass_file.is_file():
        raise RuntimeError(
            f"{ass_file.name} is missing, so the captions could not be burned in. "
            "Captions are part of the clip, not an optional sidecar."
        )
    events = sum(
        line.startswith("Dialogue:")
        for line in ass_file.read_text(encoding="utf-8").splitlines()
    )
    if events == 0:
        log.warning("%s has no caption events; this clip will render without captions", ass_file.name)
