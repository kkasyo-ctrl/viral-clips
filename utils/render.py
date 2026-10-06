"""Step 8: cut the clip out of the source at full resolution and burn the captions in."""
import json
import subprocess
from pathlib import Path

import config


def video_size(video: Path) -> tuple[int, int]:
    """Displayed (width, height) of the first video stream, accounting for rotation metadata."""
    proc = subprocess.run(
        [config.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video.name}:\n{proc.stderr[-1000:]}")
    stream = json.loads(proc.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = int(float(stream.get("tags", {}).get("rotate", 0)))
    for side_data in stream.get("side_data_list", []):
        rotation = int(float(side_data.get("rotation", rotation)))
    return (height, width) if rotation % 180 else (width, height)


def render_clip(video: Path, start: float, end: float, ass_file: Path, out_file: Path) -> None:
    """Re-encode only as much as burning captions requires: same resolution and frame rate, near-lossless CRF."""
    cmd = [
        config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video.resolve()), "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        # Relative path on purpose: ffmpeg runs inside the clip folder, which avoids escaping a Windows
        # drive letter and backslashes inside the filter argument.
        "-vf", f"ass={ass_file.name}",
        "-c:v", "libx264", "-preset", config.RENDER_PRESET, "-crf", str(config.RENDER_CRF), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", config.RENDER_AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(out_file.resolve()),
    ]
    proc = subprocess.run(cmd, cwd=ass_file.parent, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg render failed for {out_file.name}:\n{proc.stderr[-2000:]}")
