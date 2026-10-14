"""A value object standing in for clipsai.media.video_file.VideoFile.

Clips AI's resize package only ever asks a video for five things: its path, its pixel
width and height, its frame rate and its duration. The original class reads those
through a media-file hierarchy that imports the rest of clipsai (and with it WhisperX
and Pyannote), so this holds the same five values and nothing else. The caller probes
them once with ffprobe and hands them over.
"""


class VideoFile:
    """The handful of video properties the resize package reads."""

    def __init__(
        self,
        path: str,
        width_pixels: int,
        height_pixels: int,
        frame_rate: float,
        duration: float,
    ) -> None:
        self._path = str(path)
        self._width_pixels = int(width_pixels)
        self._height_pixels = int(height_pixels)
        self._frame_rate = float(frame_rate)
        self._duration = float(duration)

    @property
    def path(self) -> str:
        return self._path

    def get_filename(self) -> str:
        return self._path.replace("\\", "/").rsplit("/", 1)[-1]

    def get_width_pixels(self) -> int:
        return self._width_pixels

    def get_height_pixels(self) -> int:
        return self._height_pixels

    def get_frame_rate(self) -> float:
        return self._frame_rate

    def get_duration(self) -> float:
        return self._duration

    def __str__(self) -> str:
        return "VideoFile({}, {}x{}, {:g} fps, {:.3f}s)".format(
            self.get_filename(),
            self._width_pixels,
            self._height_pixels,
            self._frame_rate,
            self._duration,
        )
