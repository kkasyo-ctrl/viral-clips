"""Clips AI's 9:16 resize package, vendored so the pipeline keeps its own dependencies.

Copied from clipsai 0.2.1 (https://github.com/ClipsAI/clipsai), MIT licensed; the full
notice is in LICENSE next to this file. Everything that reached outside the resize
package has been replaced by local shims, so none of clipsai's WhisperX or Pyannote
dependencies are imported: speaker segments come from the cached Scribe transcript
(see utils/reframe.py) rather than from Pyannote diarization, which also means
clipsai's own `resize()` entry point is not part of the copy. Per-file changes are
listed at the top of each module.
"""
from .crops import Crops
from .rect import Rect
from .resizer import Resizer
from .segment import Segment
from .vid_proc import detect_scenes, extract_frames
from .video_file import VideoFile

__all__ = [
    "Crops",
    "Rect",
    "Resizer",
    "Segment",
    "VideoFile",
    "detect_scenes",
    "extract_frames",
]
