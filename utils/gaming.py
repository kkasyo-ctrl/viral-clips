"""Step 8b: the gaming layout -- the streamer's facecam on top, the gameplay below -- locally.

Like reframing, everything here runs on this machine and ends in an ffmpeg filter that the
render step puts in front of the captions, so the cut, the layout and the captions are
still one encode.

1. `Planner.facecam` finds the cam once per video. Frames are sampled across the whole
   stream; the streamer's face holds one spot in them while game characters move about,
   so the face that keeps its place is the cam. The overlay around it is the stretch of
   frame that stays the same between samples minutes apart, while the gameplay around it
   keeps changing. The answer is kept in facecam.json in the clip folder.
2. `Planner.plan` lays out one clip: the cam cropped to the top panel around the face,
   the middle of the game below, and the captions on the seam between them. The streamer
   can switch scenes mid-stream, so a few frames of the clip are checked for their face in
   the cam first; a clip without it shows the whole game frame over a blurred copy.
3. `layout_filter` turns the layout into the filter chain.
"""
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import config
from utils.cutting import Cut
from utils.render import VideoInfo

log = logging.getLogger(__name__)

# Bumped when detection changes enough that a saved facecam.json should be redone.
FACECAM_VERSION = 1
ANALYSIS_WIDTH = 480  # frames are shrunk to this width to tell still areas from moving ones
STILL_SHARE = 0.5  # an overlay edge strip is still when at least this share of it is
OVERLAY_INSET = 0.03  # trimmed off each side of a found overlay, to drop its border
MAX_GAME_SHIFT = 0.12  # how far (of the frame width) the game window may move off centre to miss the cam
MAX_OVERLAY = 0.6  # an overlay wider or taller than this share of the frame is not a cam
BLUR_SIGMA = 8  # background blur for a clip without the cam, at quarter resolution


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_list(cls, values) -> "Rect":
        return cls(*(int(v) for v in values))


@dataclass(frozen=True)
class Facecam:
    overlay: Rect  # the cam overlay in the source frame
    face: Rect  # where the streamer's face usually is


@dataclass(frozen=True)
class Layout:
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    game: Rect  # the window into the source for the gameplay
    cam: Rect | None  # the window for the cam; None when the clip has no cam in it
    cam_height: int  # output rows the cam gets at the top; 0 without a cam

    @property
    def seam(self) -> int | None:
        """The output row where the cam meets the game, which is where the captions go."""
        return self.cam_height if self.cam else None

    def to_dict(self) -> dict:
        return {
            "format": "gaming",
            "output": [self.output_width, self.output_height],
            "cam": self.cam.to_list() if self.cam else None,
            "cam_height": self.cam_height,
            "game": self.game.to_list(),
        }


def _even(n: float) -> int:
    return max(2, int(n) // 2 * 2)


# --- geometry -----------------------------------------------------------------------

def fit(aspect: float, area: Rect, cx: float, cy: float, head_room: float = 0.5) -> Rect:
    """The largest even-sized window of this aspect (width / height) inside `area`, centred
    on (cx, cy) horizontally and with that point head_room of the way down, then moved
    back inside the area where it would run over an edge."""
    w, h = area.w, area.h
    if w / h > aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(_even(w), area.w // 2 * 2), min(_even(h), area.h // 2 * 2)
    x = min(max(cx - w / 2, area.x), area.x + area.w - w)
    y = min(max(cy - h * head_room, area.y), area.y + area.h - h)
    return Rect(int(x) // 2 * 2, int(y) // 2 * 2, w, h)


def game_window(width: int, height: int, aspect: float, avoid: Rect | None = None) -> Rect:
    """The middle of the frame, at the aspect of the panel it fills. When it would take in
    the cam overlay (`avoid`) it slides away from it, by at most MAX_GAME_SHIFT of the
    frame, so the streamer does not show up a second time in the game panel."""
    game = fit(aspect, Rect(0, 0, width, height), width / 2, height / 2)
    if avoid is None or avoid.y >= game.y + game.h or avoid.y + avoid.h <= game.y:
        return game
    reach = MAX_GAME_SHIFT * width
    pad = 2 * OVERLAY_INSET * avoid.w  # the found overlay has had its border trimmed off; miss that too
    if avoid.cx >= width / 2:  # cam on the right: move left until the window ends where it starts
        x = max(min(game.x, avoid.x - pad - game.w), game.x - reach, 0)
    else:
        x = min(max(game.x, avoid.x + avoid.w + pad), game.x + reach, width - game.w)
    return Rect(int(x) // 2 * 2, game.y, game.w, game.h)


def layout(info: VideoInfo, facecam: Facecam | None, cam_share: float) -> Layout:
    """Where the cam and the game come from, for a clip with the cam in it, or without."""
    out_w, out_h = config.REFRAME_WIDTH, config.REFRAME_HEIGHT
    if facecam is None:
        return Layout(info.width, info.height, out_w, out_h, Rect(0, 0, info.width, info.height), None, 0)
    lo, hi = config.GAMING_CAM_SHARE_RANGE
    cam_h = _even(out_h * min(max(cam_share, lo), hi))
    game_h = out_h - cam_h
    face = facecam.face
    # The face sits a little above the middle of the panel, the way a cam is framed.
    cam = fit(out_w / cam_h, facecam.overlay, face.cx, face.cy, head_room=0.42)
    game = game_window(info.width, info.height, out_w / game_h, avoid=facecam.overlay)
    return Layout(info.width, info.height, out_w, out_h, game, cam, cam_h)


# --- the ffmpeg side ----------------------------------------------------------------

def layout_filter(plan: Layout) -> str:
    """The filter chain for the layout, ready for the render step to add the captions to."""
    out_w, out_h = plan.output_width, plan.output_height
    if plan.cam is None:
        # The whole game frame, over a blurred copy of itself filling the rest of the frame.
        small_w, small_h = _even(out_w / 4), _even(out_h / 4)
        return (
            "split=2[bg][fg];"
            f"[bg]scale={small_w}:{small_h}:force_original_aspect_ratio=increase,crop={small_w}:{small_h},"
            f"gblur=sigma={BLUR_SIGMA},scale={out_w}:{out_h},setsar=1[back];"
            f"[fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,"
            "setsar=1[front];"
            "[back][front]overlay=(W-w)/2:(H-h)/2"
        )
    cam, game = plan.cam, plan.game
    game_h = out_h - plan.cam_height
    return (
        "split=2[cam][game];"
        f"[cam]crop={cam.w}:{cam.h}:{cam.x}:{cam.y},scale={out_w}:{plan.cam_height}:flags=lanczos,setsar=1[top];"
        f"[game]crop={game.w}:{game.h}:{game.x}:{game.y},scale={out_w}:{game_h}:flags=lanczos,setsar=1[bottom];"
        "[top][bottom]vstack"
    )


# --- finding the cam ----------------------------------------------------------------

def sample_times(duration: float, n: int) -> list[float]:
    """n times spread over the video, skipping the first and last 5% (intros, outros)."""
    start, end = duration * 0.05, duration * 0.95
    return [start + (k + 0.5) * (end - start) / n for k in range(n)]


def steady_face(detections: list, width: int, height: int) -> Rect | None:
    """The face that holds one spot across the sampled frames, or None.

    `detections` holds each frame's face boxes as (x1, y1, x2, y2) rows, or None. Every
    box is tried as the anchor of a spot; the spot found in the most frames wins, if it is
    in enough of them and the face is small enough to be an overlay rather than a
    full-screen camera.
    """
    boxes = [
        (i, [float(v) for v in box[:4]])
        for i, frame in enumerate(detections) if frame is not None for box in frame
    ]
    if not boxes or not detections:
        return None
    reach = config.GAMING_CAM_MAX_DRIFT * width

    def centre(box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    best: list = []
    for _, anchor in boxes:
        ax, ay = centre(anchor)
        near = [(i, b) for i, b in boxes if math.dist(centre(b), (ax, ay)) <= reach]
        if len({i for i, _ in near}) > len({i for i, _ in best}):
            best = near
    presence = len({i for i, _ in best}) / len(detections)
    if presence < config.GAMING_CAM_MIN_PRESENCE:
        log.info("No facecam: the steadiest face is in %.0f%% of the sampled frames", 100 * presence)
        return None
    x1, y1, x2, y2 = (sorted(b[k] for _, b in best)[len(best) // 2] for k in range(4))
    face = Rect(round(x1), round(y1), round(x2 - x1), round(y2 - y1))
    if face.h > config.GAMING_CAM_MAX_FACE * height:
        log.info("No facecam: the steady face fills the frame, so this is a camera shot, not an overlay")
        return None
    return face


def overlay_around(face: Rect, frames: list, width: int, height: int) -> Rect:
    """The cam overlay around the face: the still region the face sits in.

    `frames` are greyscale frames shrunk to ANALYSIS_WIDTH, sampled across the stream.
    The gameplay changes between them while the streamer's room barely does, so starting
    from the face, each side of a box is pushed outwards for as long as the strip it would
    take in is mostly still. Where that finds nothing sensible -- a green-screen cut-out
    has no still box around it -- a box around the face stands in.
    """
    import numpy as np

    scale = width / frames[0].shape[1]
    change = np.stack(frames).astype(np.float32).std(axis=0)
    still = change < float(np.clip(0.5 * np.median(change), 8, 30))
    rows, cols = still.shape

    left, top = int(face.x / scale), int(face.y / scale)
    right, bottom = int((face.x + face.w) / scale), int((face.y + face.h) / scale)
    left, top, right, bottom = max(left, 0), max(top, 0), min(right, cols - 1), min(bottom, rows - 1)
    grew = True
    while grew:
        grew = False
        if left > 0 and still[top : bottom + 1, left - 1].mean() >= STILL_SHARE:
            left, grew = left - 1, True
        if right < cols - 1 and still[top : bottom + 1, right + 1].mean() >= STILL_SHARE:
            right, grew = right + 1, True
        if top > 0 and still[top - 1, left : right + 1].mean() >= STILL_SHARE:
            top, grew = top - 1, True
        if bottom < rows - 1 and still[bottom + 1, left : right + 1].mean() >= STILL_SHARE:
            bottom, grew = bottom + 1, True

    w, h = (right - left + 1) * scale, (bottom - top + 1) * scale
    inset_x, inset_y = w * OVERLAY_INSET, h * OVERLAY_INSET
    found = Rect(
        round(left * scale + inset_x), round(top * scale + inset_y), round(w - 2 * inset_x), round(h - 2 * inset_y)
    )
    if (
        # A tight webcam can be little more than the face, so the floor is low.
        found.w >= 1.3 * face.w and found.h >= 1.1 * face.h
        and found.w <= MAX_OVERLAY * width and found.h <= MAX_OVERLAY * height
    ):
        return found
    log.info("Facecam overlay edges unclear; framing the face instead")
    return around_face(face, width, height)


def around_face(face: Rect, width: int, height: int) -> Rect:
    """A cam-shaped box around the face: head and shoulders, clamped to the frame."""
    h = min(2.5 * face.h, height)
    w = min(h * 16 / 9, width)
    x = min(max(face.cx - w / 2, 0), width - w)
    y = min(max(face.y - 0.6 * face.h, 0), height - h)
    return Rect(round(x), round(y), round(w), round(h))


# --- per video and per clip ---------------------------------------------------------

class Planner:
    """Lays out the clips of one video. The face detector is loaded once, and only if
    the facecam is not already on record or a clip needs checking."""

    def __init__(self, video: Path, info: VideoInfo, folder: Path, cam_share: float) -> None:
        self.video, self.info, self.cam_share = video, info, cam_share
        self.cache = folder / "facecam.json"
        self._data: dict = {}
        self._facecam: Facecam | None = None
        self._looked = False
        self._resizer = None

    @property
    def facecam(self) -> Facecam | None:
        if not self._looked:
            self._facecam = self._recorded() if self._on_record() else self._find()
            self._looked = True
        return self._facecam

    def plan(self, cut: Cut) -> Layout:
        cam = self.facecam
        if cam is not None and not self._cam_in(cut, cam):
            log.info("The streamer is not in the cam during this clip; using the full game frame")
            cam = None
        return layout(self.info, cam, self.cam_share)

    def close(self) -> None:
        if self._resizer is not None:
            self._resizer.cleanup()
            self._resizer = None

    # -- facecam.json

    def _on_record(self) -> bool:
        if not self.cache.is_file():
            return False
        try:
            data = json.loads(self.cache.read_text(encoding="utf-8"))
        except ValueError:
            return False
        self._data = data
        return (
            data.get("version") == FACECAM_VERSION
            and data.get("source") == [self.info.width, self.info.height, round(self.info.duration, 1)]
        )

    def _recorded(self) -> Facecam | None:
        data = self._data
        log.info("Facecam on record: %s", data["overlay"] or "none")
        if not data["overlay"]:
            return None
        return Facecam(Rect.from_list(data["overlay"]), Rect.from_list(data["face"]))

    def _find(self) -> Facecam | None:
        detections, small = [], []
        times = sample_times(self.info.duration, config.GAMING_DETECT_SAMPLES)
        for b in range(0, len(times), 8):  # a few at a time, so full-size frames never pile up
            frames = self._frames(times[b : b + 8])
            detections += self._detector()._detect_faces(frames, config.REFRAME_FACE_DETECT_WIDTH)
            small += [_shrink(f) for f in frames]
        face = steady_face(detections, self.info.width, self.info.height)
        cam = None
        if face is not None:
            cam = Facecam(overlay_around(face, small, self.info.width, self.info.height), face)
            log.info("Facecam: overlay %s, face %s", cam.overlay.to_list(), cam.face.to_list())
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text(json.dumps({
            "version": FACECAM_VERSION,
            "source": [self.info.width, self.info.height, round(self.info.duration, 1)],
            "overlay": cam.overlay.to_list() if cam else None,
            "face": cam.face.to_list() if cam else None,
        }, indent=2), encoding="utf-8")
        return cam

    # -- per clip

    def _cam_in(self, cut: Cut, cam: Facecam) -> bool:
        """Whether the streamer's face is in the cam for most of the clip."""
        n = config.GAMING_CLIP_CHECK_SAMPLES
        times = [cut.start + (k + 0.5) * cut.duration / n for k in range(n)]
        detections = self._detector()._detect_faces(self._frames(times), config.REFRAME_FACE_DETECT_WIDTH)
        hits = sum(
            frame is not None and any(cam.overlay.contains((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in frame)
            for frame in detections
        )
        return hits >= n / 2

    def _frames(self, times: list[float]) -> list:
        from utils.clipsai_resize import VideoFile, extract_frames

        info = self.info
        source = VideoFile(str(self.video.resolve()), info.width, info.height, info.frame_rate, info.duration)
        return extract_frames(source, [min(t, info.duration - 0.05) for t in times])

    def _detector(self):
        if self._resizer is None:
            from utils.clipsai_resize import Resizer

            self._resizer = Resizer(face_detect_margin=20, face_detect_post_process=False, device=config.REFRAME_DEVICE)
        return self._resizer


def _shrink(frame):
    """An RGB frame as small greyscale, for telling still areas from moving ones."""
    import cv2

    height = round(frame.shape[0] * ANALYSIS_WIDTH / frame.shape[1])
    return cv2.cvtColor(cv2.resize(frame, (ANALYSIS_WIDTH, height), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
