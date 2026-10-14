"""Step 8: reframe a clip to vertical, locally.

Everything here runs on this machine: no API calls, keys or tokens. Speaker turns come
from the speaker IDs already on the cached Scribe words, scene cuts from PySceneDetect
and the crop per segment from Clips AI's Resizer (MTCNN faces plus mouth movement,
vendored in utils/clipsai_resize). The result is a crop plan -- one window into the
source frame per segment -- which `crop_filter` turns into an ffmpeg filter chain. The
chain is handed to the render step, so cutting, cropping, scaling and burning the
captions in all happen in that one encode; a landscape clip is never rendered and then
re-encoded.

The plan is made in the clip's own timeline (0 = the first frame of the clip, pads
included) because that is what the ffmpeg filter and the .srt sidecar both use. Face
analysis reads the source video over the matching window.
"""
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import config
from utils import render
from utils.cutting import Cut
from utils.render import VideoInfo
from utils.transcript import Word

log = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """One speaker holding the floor, in seconds from the start of the clip."""
    start: float
    end: float
    speaker: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class CropSegment:
    """The window into the source frame to use for one stretch of the clip."""
    start: float  # seconds from the start of the clip
    end: float
    speaker: int
    x: int  # top-left corner of the crop in the source frame
    y: int
    fallback: bool  # True if no face was found and a held or centred crop was used

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class CropPlan:
    source_width: int
    source_height: int
    crop_width: int
    crop_height: int
    output_width: int
    output_height: int
    segments: list[CropSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "speaker": s.speaker,
                    "x": s.x,
                    "y": s.y,
                    "fallback": s.fallback,
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CropPlan":
        return cls(
            source_width=int(data["source_width"]),
            source_height=int(data["source_height"]),
            crop_width=int(data["crop_width"]),
            crop_height=int(data["crop_height"]),
            output_width=int(data["output_width"]),
            output_height=int(data["output_height"]),
            segments=[
                CropSegment(
                    start=float(s["start"]),
                    end=float(s["end"]),
                    speaker=int(s["speaker"]),
                    x=int(s["x"]),
                    y=int(s["y"]),
                    fallback=bool(s["fallback"]),
                )
                for s in data["segments"]
            ],
        )


# --- geometry -----------------------------------------------------------------------

def _even(n: float) -> int:
    """Round down to an even number of pixels, which every yuv420p filter is happy with."""
    return max(2, int(n) - int(n) % 2)


def output_size() -> tuple[int, int]:
    return config.REFRAME_WIDTH, config.REFRAME_HEIGHT


def crop_size(
    source_width: int, source_height: int, aspect_ratio: tuple[int, int] | None = None
) -> tuple[int, int]:
    """The largest window of `aspect_ratio` that fits inside the source frame."""
    ratio_width, ratio_height = aspect_ratio or config.REFRAME_ASPECT_RATIO
    wanted = ratio_width / ratio_height
    if source_width / source_height > wanted:  # source is wider: full height, crop the sides
        height, width = source_height, source_height * wanted
    else:  # source is as tall or taller: full width, crop top and bottom
        width, height = source_width, source_width / wanted
    return _even(min(width, source_width)), _even(min(height, source_height))


def clamp_crop(
    x: float, y: float, crop_width: int, crop_height: int, source_width: int, source_height: int
) -> tuple[int, int]:
    """Pull a crop's top-left corner inside the frame, and onto even pixels.

    A crop centred on a face near an edge would otherwise reach past the frame, which
    ffmpeg refuses outright.
    """
    x = min(max(int(x), 0), max(source_width - crop_width, 0))
    y = min(max(int(y), 0), max(source_height - crop_height, 0))
    return x - x % 2, y - y % 2


# --- speaker segments from the cached transcript ------------------------------------

def speaker_segments(
    cut: Cut, words: list[Word], min_seconds: float | None = None
) -> list[SpeakerSegment]:
    """Who holds the floor when, across the clip, from the Scribe words' speaker IDs.

    Consecutive words by one speaker become a turn; turns are moved into clip time,
    trimmed to the clip, and their speaker IDs numbered from 0 in order of appearance.
    Pauses go to the speaker who just finished, short turns are merged away, and the
    result covers the clip end to end -- the crop has to come from somewhere for every
    frame, so a gap is not an option.
    """
    if min_seconds is None:
        min_seconds = config.REFRAME_MIN_SEGMENT_SECONDS
    duration = cut.duration
    turns = _turns(cut, words)
    if len(turns) <= 1:  # one speaker, or a transcript with no speaker IDs at all
        return [SpeakerSegment(0.0, duration, turns[0].speaker if turns else 0)]
    return merge_short(fill_gaps(turns, duration), min_seconds)


def _turns(cut: Cut, words: list[Word]) -> list[SpeakerSegment]:
    """Consecutive words by one speaker, in clip time, trimmed to the clip window."""
    speakers: dict[str, int] = {}
    turns: list[SpeakerSegment] = []
    for word in words:
        if word.end <= cut.start or word.start >= cut.end:
            continue
        speaker = speakers.setdefault(word.speaker, len(speakers))
        start = max(0.0, word.start - cut.start)
        end = min(cut.duration, word.end - cut.start)
        if turns and turns[-1].speaker == speaker:
            turns[-1].end = max(turns[-1].end, end)
        else:
            turns.append(SpeakerSegment(max(start, turns[-1].end if turns else 0.0), end, speaker))
    return [t for t in turns if t.end > t.start]


def fill_gaps(segments: list[SpeakerSegment], duration: float) -> list[SpeakerSegment]:
    """Close every gap and overlap so the segments tile [0, duration] exactly.

    A silence between two turns is given to the speaker who just finished, which keeps
    the shot on them while they land the line instead of cutting away to a listener.
    """
    out = [replace(s) for s in segments]
    if not out:
        return []
    _close(out, duration)
    kept = [s for s in out if s.end > s.start]
    if not kept:  # nothing survived: one segment over the whole clip
        only = replace(out[0])
        only.start, only.end = 0.0, duration
        return [only]
    _close(kept, duration)  # dropping an empty segment reopens a gap
    return kept


def _close(segments: list, duration: float) -> None:
    """Make the segments abut, starting at 0 and ending at `duration`."""
    segments[0].start = 0.0
    for current, following in zip(segments, segments[1:]):
        current.end = max(following.start, current.start)
    segments[-1].end = duration


def merge_short(segments: list[SpeakerSegment], min_seconds: float) -> list[SpeakerSegment]:
    """Fold turns shorter than `min_seconds` into a neighbour, keeping coverage intact.

    A back-channel "yeah" is not worth a shot change, and a segment too short to sample
    frames from cannot be analysed anyway. Short turns go to the speaker before them, or
    to the one after if there is nobody before.
    """
    out = [replace(s) for s in segments]
    if min_seconds <= 0:
        return out
    i = 0
    while len(out) > 1 and i < len(out):
        if out[i].duration >= min_seconds:
            i += 1
        elif i > 0:
            out[i - 1].end = out[i].end
            del out[i]
        else:
            out[1].start = out[0].start
            del out[0]
    return out


# --- crop planning ------------------------------------------------------------------

def plan(
    video: Path, cut: Cut, words: list[Word], info: VideoInfo, recorded: dict | None = None
) -> CropPlan:
    """The crop plan for one clip, reusing a recorded one rather than redoing the faces."""
    reused = _reusable(recorded, cut, info)
    if reused is not None:
        log.info("Reusing %d recorded crop segment(s)", len(reused.segments))
        return reused
    segments = speaker_segments(cut, words)
    log.info(
        "Reframing: %d speaker segment(s), %d speaker(s)",
        len(segments),
        len({s.speaker for s in segments}),
    )
    return plan_crops(video, cut, segments, info)


def _reusable(recorded: dict | None, cut: Cut, info: VideoInfo) -> CropPlan | None:
    """A recorded plan, if it was made for this clip at these dimensions."""
    if not recorded:
        return None
    try:
        existing = CropPlan.from_dict(recorded)
    except (KeyError, TypeError, ValueError):
        log.warning("Recorded crop plan is unreadable; planning again")
        return None
    width, height = crop_size(info.width, info.height)
    out_width, out_height = output_size()
    current = (info.width, info.height, width, height, out_width, out_height)
    if (
        existing.source_width,
        existing.source_height,
        existing.crop_width,
        existing.crop_height,
        existing.output_width,
        existing.output_height,
    ) != current:
        log.info("Recorded crop plan was made for other dimensions; planning again")
        return None
    if not existing.segments or abs(existing.segments[-1].end - cut.duration) > 0.05:
        log.info("Recorded crop plan does not span this clip; planning again")
        return None
    return existing


def plan_crops(
    video: Path, cut: Cut, segments: list[SpeakerSegment], info: VideoInfo
) -> CropPlan:
    """Find one crop per segment with scene detection and face analysis on the source.

    The heavy dependencies are imported here rather than at module level, so nothing
    loads a model unless a clip is actually being reframed.
    """
    import numpy as np
    import torch

    from utils.clipsai_resize import Resizer, VideoFile, detect_scenes

    # Frame sampling inside the Resizer is random; a fixed seed makes a re-run of the
    # same clip produce the same crops.
    np.random.seed(config.REFRAME_SEED)
    torch.manual_seed(config.REFRAME_SEED)

    source = VideoFile(
        str(video.resolve()), info.width, info.height, info.frame_rate, info.duration
    )
    width, height = crop_size(info.width, info.height)
    out_width, out_height = output_size()

    # The Resizer reads frames out of the source video, so hand it source-time segments
    # over the clip's cut window and shift the answers back afterwards.
    scene_changes = detect_scenes(source, config.REFRAME_SCENE_THRESHOLD, cut.start, cut.end)
    first, last = cut.start + segments[0].start, cut.start + segments[-1].end
    # Clips AI walks the segment list once per scene change, so one landing outside the
    # covered span (or exactly on its edge) would run off the end of that list.
    inside = [t for t in scene_changes if first + 1e-3 < t < last - 1e-3]
    if len(inside) != len(scene_changes):
        log.debug("Dropped %d scene change(s) outside the clip", len(scene_changes) - len(inside))
    log.info("Reframing: %d scene change(s) in the clip", len(inside))

    resizer = Resizer(
        face_detect_margin=20,
        face_detect_post_process=False,
        device=config.REFRAME_DEVICE,
    )
    try:
        crops = resizer.resize(
            video_file=source,
            speaker_segments=[
                {
                    "speakers": [s.speaker],
                    "start_time": cut.start + s.start,
                    "end_time": cut.start + s.end,
                }
                for s in segments
            ],
            scene_changes=inside,
            aspect_ratio=tuple(config.REFRAME_ASPECT_RATIO),
            samples_per_segment=config.REFRAME_SAMPLES_PER_SEGMENT,
            face_detect_width=config.REFRAME_FACE_DETECT_WIDTH,
            scene_merge_threshold=config.REFRAME_SCENE_MERGE_THRESHOLD,
        )
    finally:
        resizer.cleanup()

    planned = to_crop_segments(crops.segments, cut, width, height, info)
    fallbacks = sum(s.fallback for s in planned)
    log.info(
        "Reframing: %d crop segment(s)%s",
        len(planned),
        f", {fallbacks} without a face" if fallbacks else "",
    )
    return CropPlan(info.width, info.height, width, height, out_width, out_height, planned)


def to_crop_segments(
    segments: list, cut: Cut, crop_width: int, crop_height: int, info: VideoInfo
) -> list[CropSegment]:
    """Clips AI segments in source time -> clamped crop segments in clip time.

    Clips AI centres a faceless segment on the middle of the frame. That is the right
    answer for the opening segment and the wrong one later on, where cutting to the
    middle of the room reads as a mistake; those hold the crop they already had.
    """
    out: list[CropSegment] = []
    held: tuple[int, int] | None = None
    for segment in segments:
        fallback = not getattr(segment, "found_face", True)
        x, y = held if fallback and held is not None else (segment.x, segment.y)
        x, y = clamp_crop(x, y, crop_width, crop_height, info.width, info.height)
        held = (x, y)
        out.append(
            CropSegment(
                start=round(segment.start_time - cut.start, 3),
                end=round(segment.end_time - cut.start, 3),
                speaker=segment.speakers[0] if segment.speakers else 0,
                x=x,
                y=y,
                fallback=fallback,
            )
        )
    # Scene merging can nudge the edges; the filter needs every frame covered.
    return fill_gaps(out, round(cut.duration, 3))


# --- the ffmpeg side ----------------------------------------------------------------

def crop_filter(plan: CropPlan, eval_option: str | None = None) -> str:
    """The filter chain that crops to the plan and scales to the output size.

    Returned for the render step to put in front of the caption filter, so one encode
    does the cut, the crop, the scale and the burned-in captions. `eval_option` is asked
    of ffmpeg when it is not given; see render.crop_eval_option.
    """
    chain = []
    if (plan.crop_width, plan.crop_height) != (plan.source_width, plan.source_height):
        ends = [s.end for s in plan.segments]
        x = _steps(ends, [s.x for s in plan.segments])
        y = _steps(ends, [s.y for s in plan.segments])
        where = f"x={_quote(x)}:y={_quote(y)}"
        if "(" in x or "(" in y:  # the crop moves, so ffmpeg has to re-read it per frame
            where += render.crop_eval_option() if eval_option is None else eval_option
        chain.append(f"crop=w={plan.crop_width}:h={plan.crop_height}:{where}")
    chain.append(f"scale={plan.output_width}:{plan.output_height}:flags=lanczos")
    chain.append("setsar=1")
    return ",".join(chain)


def _steps(ends: list[float], values: list[int]) -> str:
    """An ffmpeg expression holding each value until its segment ends, or one constant."""
    if len(set(values)) == 1:
        return str(values[0])
    expression = str(values[-1])
    for end, value in zip(reversed(ends[:-1]), reversed(values[:-1])):
        expression = f"if(lt(t,{end:.3f}),{value},{expression})"
    return expression


def _quote(expression: str) -> str:
    """Quote an expression, so the commas in it are not read as filter separators."""
    return f"'{expression}'" if "(" in expression else expression
