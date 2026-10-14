"""Logic tests for the reframing stage. Nothing here loads a model or reads a video."""
from dataclasses import dataclass
from pathlib import Path

import pytest

from utils import reframe
from utils.cutting import Cut
from utils.render import VideoInfo
from utils.transcript import Word, normalize_words

VENDORED = Path(__file__).resolve().parents[1] / "utils" / "clipsai_resize"


def cut(start: float, end: float) -> Cut:
    """A cut window; the sentence and word indices play no part in reframing."""
    return Cut(start=start, end=end, first_sentence=0, last_sentence=0, first_word=0, last_word=0)


def word(text: str, start: float, end: float, speaker: str) -> Word:
    return Word(text=text, start=start, end=end, speaker=speaker)


def spans(segments) -> list[tuple[float, float, int]]:
    return [(round(s.start, 3), round(s.end, 3), s.speaker) for s in segments]


@dataclass
class FakeSegment:
    """Stands in for a Clips AI Segment, so the model package is never imported."""
    start_time: float
    end_time: float
    speakers: list[int]
    x: int
    y: int
    found_face: bool = True


# --- the speaker-segment adapter ----------------------------------------------------

def test_turns_are_in_clip_time_with_integer_speakers():
    words = [
        word("hello", 10.0, 10.5, "speaker_3"),
        word("there", 10.5, 11.0, "speaker_3"),
        word("hi", 12.0, 12.5, "speaker_7"),
        word("back", 12.5, 13.0, "speaker_7"),
    ]
    segments = reframe.speaker_segments(cut(9.5, 13.5), words, min_seconds=0)
    # 0.5 s of lead-in pad, so the first word lands half a second into the clip; IDs are
    # numbered from 0 in order of appearance, not taken from Scribe's labels.
    assert spans(segments) == [(0.0, 2.5, 0), (2.5, 4.0, 1)]


def test_words_outside_the_cut_window_are_ignored():
    words = [
        word("before", 1.0, 2.0, "speaker_0"),
        word("inside", 11.0, 12.0, "speaker_1"),
        word("after", 30.0, 31.0, "speaker_2"),
    ]
    segments = reframe.speaker_segments(cut(10.0, 20.0), words, min_seconds=0)
    assert spans(segments) == [(0.0, 10.0, 0)]  # only speaker_1 was inside, so it is speaker 0


def test_a_word_straddling_the_edge_is_trimmed_to_the_window():
    words = [
        word("early", 9.0, 10.5, "speaker_0"),
        word("mid", 12.0, 13.0, "speaker_1"),
        word("late", 19.5, 21.0, "speaker_0"),
    ]
    segments = reframe.speaker_segments(cut(10.0, 20.0), words, min_seconds=0)
    assert segments[0].start == 0.0
    assert segments[-1].end == 10.0
    assert all(0.0 <= s.start and s.end <= 10.0 for s in segments)


def test_pauses_go_to_the_speaker_who_just_finished():
    words = [
        word("my", 0.0, 1.0, "speaker_0"),
        word("turn", 1.0, 2.0, "speaker_0"),
        # three seconds of nobody talking
        word("mine", 5.0, 6.0, "speaker_1"),
    ]
    segments = reframe.speaker_segments(cut(0.0, 6.0), words, min_seconds=0)
    assert spans(segments) == [(0.0, 5.0, 0), (5.0, 6.0, 1)]


def test_a_single_speaker_gets_one_segment_over_the_whole_clip():
    words = [word("all", 0.0, 1.0, "speaker_0"), word("mine", 4.0, 5.0, "speaker_0")]
    assert spans(reframe.speaker_segments(cut(0.0, 6.0), words)) == [(0.0, 6.0, 0)]


def test_a_transcript_without_speaker_ids_is_treated_as_one_speaker():
    # normalize_words falls back to "speaker" when Scribe returns no speaker_id.
    response = {"words": [
        {"text": "no", "start": 0.0, "end": 0.5, "type": "word"},
        {"text": "diarization", "start": 0.5, "end": 1.5, "type": "word"},
        {"text": "here", "start": 4.0, "end": 4.5, "type": "word"},
    ]}
    words = normalize_words(response)
    assert spans(reframe.speaker_segments(cut(0.0, 5.0), words)) == [(0.0, 5.0, 0)]


def test_no_words_in_the_window_still_yields_a_covering_segment():
    assert spans(reframe.speaker_segments(cut(0.0, 8.0), [])) == [(0.0, 8.0, 0)]


def test_adapter_reads_speaker_ids_straight_off_a_scribe_response():
    response = {"words": [
        {"text": "so", "start": 0.0, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing"},
        {"text": "anyway", "start": 0.5, "end": 1.0, "type": "word", "speaker_id": "speaker_0"},
        {"text": "(laughs)", "start": 1.0, "end": 1.2, "type": "audio_event"},
        {"text": "right", "start": 2.0, "end": 2.6, "type": "word", "speaker_id": "speaker_1"},
    ]}
    segments = reframe.speaker_segments(cut(0.0, 4.0), normalize_words(response), min_seconds=0)
    assert spans(segments) == [(0.0, 2.0, 0), (2.0, 4.0, 1)]


# --- gap filling --------------------------------------------------------------------

def test_fill_gaps_tiles_the_clip_end_to_end():
    segments = [
        reframe.SpeakerSegment(1.0, 2.0, 0),
        reframe.SpeakerSegment(4.0, 5.0, 1),
        reframe.SpeakerSegment(7.0, 8.0, 0),
    ]
    filled = reframe.fill_gaps(segments, 10.0)
    assert spans(filled) == [(0.0, 4.0, 0), (4.0, 7.0, 1), (7.0, 10.0, 0)]
    assert all(a.end == b.start for a, b in zip(filled, filled[1:]))


def test_fill_gaps_trims_overlaps():
    segments = [reframe.SpeakerSegment(0.0, 6.0, 0), reframe.SpeakerSegment(4.0, 9.0, 1)]
    assert spans(reframe.fill_gaps(segments, 9.0)) == [(0.0, 4.0, 0), (4.0, 9.0, 1)]


def test_fill_gaps_drops_empty_segments_without_reopening_a_gap():
    segments = [
        reframe.SpeakerSegment(0.0, 2.0, 0),
        reframe.SpeakerSegment(2.0, 2.0, 1),  # a word with identical start and end
        reframe.SpeakerSegment(2.0, 5.0, 2),
    ]
    filled = reframe.fill_gaps(segments, 5.0)
    assert spans(filled) == [(0.0, 2.0, 0), (2.0, 5.0, 2)]
    assert all(a.end == b.start for a, b in zip(filled, filled[1:]))


def test_fill_gaps_leaves_the_caller_list_alone():
    segments = [reframe.SpeakerSegment(1.0, 2.0, 0), reframe.SpeakerSegment(5.0, 6.0, 1)]
    reframe.fill_gaps(segments, 9.0)
    assert spans(segments) == [(1.0, 2.0, 0), (5.0, 6.0, 1)]


def test_fill_gaps_on_nothing_returns_nothing():
    assert reframe.fill_gaps([], 5.0) == []


# --- merging short turns ------------------------------------------------------------

def test_merge_short_folds_a_back_channel_into_the_previous_speaker():
    segments = [
        reframe.SpeakerSegment(0.0, 5.0, 0),
        reframe.SpeakerSegment(5.0, 5.4, 1),  # "yeah"
        reframe.SpeakerSegment(5.4, 10.0, 0),
    ]
    assert spans(reframe.merge_short(segments, 1.5)) == [(0.0, 5.4, 0), (5.4, 10.0, 0)]


def test_merge_short_folds_a_short_opening_turn_into_the_next():
    segments = [
        reframe.SpeakerSegment(0.0, 0.5, 0),
        reframe.SpeakerSegment(0.5, 6.0, 1),
    ]
    assert spans(reframe.merge_short(segments, 1.5)) == [(0.0, 6.0, 1)]


def test_merge_short_keeps_one_segment_when_every_turn_is_short():
    segments = [
        reframe.SpeakerSegment(0.0, 0.4, 0),
        reframe.SpeakerSegment(0.4, 0.8, 1),
        reframe.SpeakerSegment(0.8, 1.2, 0),
    ]
    assert spans(reframe.merge_short(segments, 1.5)) == [(0.0, 1.2, 0)]


def test_merge_short_keeps_coverage_and_order():
    segments = reframe.fill_gaps([
        reframe.SpeakerSegment(0.0, 0.3, 0),
        reframe.SpeakerSegment(0.3, 4.0, 1),
        reframe.SpeakerSegment(4.0, 4.2, 0),
        reframe.SpeakerSegment(4.2, 9.0, 1),
    ], 9.0)
    merged = reframe.merge_short(segments, 1.5)
    assert merged[0].start == 0.0 and merged[-1].end == 9.0
    assert all(a.end == b.start for a, b in zip(merged, merged[1:]))
    assert all(s.duration >= 1.5 for s in merged)


def test_merge_short_is_a_no_op_at_zero():
    segments = [reframe.SpeakerSegment(0.0, 0.1, 0), reframe.SpeakerSegment(0.1, 1.0, 1)]
    assert spans(reframe.merge_short(segments, 0)) == [(0.0, 0.1, 0), (0.1, 1.0, 1)]


def test_short_turns_are_gone_from_the_adapter_output():
    words = [
        word("a", 0.0, 3.0, "speaker_0"),
        word("mm", 3.0, 3.3, "speaker_1"),
        word("b", 3.3, 8.0, "speaker_0"),
    ]
    segments = reframe.speaker_segments(cut(0.0, 8.0), words, min_seconds=1.5)
    assert all(s.duration >= 1.5 for s in segments)
    assert segments[0].start == 0.0 and segments[-1].end == 8.0


# --- crop geometry ------------------------------------------------------------------

@pytest.mark.parametrize("source, expected", [
    ((1920, 1080), (606, 1080)),  # landscape: full height, sides cropped off
    ((1280, 720), (404, 720)),
    ((1080, 1920), (1080, 1920)),  # already 9:16: nothing to crop
    ((1080, 2160), (1080, 1920)),  # taller than 9:16: full width, top and bottom cropped
])
def test_crop_size_is_the_largest_window_that_fits(source, expected):
    assert reframe.crop_size(*source, (9, 16)) == expected


def test_crop_size_never_leaves_the_frame():
    for width, height in [(1920, 1080), (640, 480), (1080, 1920), (720, 1280), (900, 900)]:
        crop_width, crop_height = reframe.crop_size(width, height, (9, 16))
        assert crop_width <= width and crop_height <= height
        assert crop_width % 2 == 0 and crop_height % 2 == 0


def test_clamp_crop_pulls_a_crop_back_inside_the_frame():
    # A face at the right edge wants x=1700, but a 606-wide crop can only start at 1314.
    assert reframe.clamp_crop(1700, 0, 606, 1080, 1920, 1080) == (1314, 0)
    assert reframe.clamp_crop(-80, -20, 606, 1080, 1920, 1080) == (0, 0)
    assert reframe.clamp_crop(0, 500, 1080, 1920, 1080, 2160) == (0, 240)


def test_clamp_crop_snaps_to_even_pixels():
    x, y = reframe.clamp_crop(301, 455, 400, 600, 1920, 1080)
    assert (x, y) == (300, 454)


def test_clamp_crop_handles_a_crop_bigger_than_the_frame():
    assert reframe.clamp_crop(50, 50, 2000, 2000, 1920, 1080) == (0, 0)


def test_every_clamped_crop_stays_inside_the_frame():
    source_width, source_height = 1920, 1080
    crop_width, crop_height = reframe.crop_size(source_width, source_height, (9, 16))
    for x in (-500, 0, 700, 1900, 5000):
        for y in (-100, 0, 400, 2000):
            left, top = reframe.clamp_crop(x, y, crop_width, crop_height, source_width, source_height)
            assert 0 <= left and left + crop_width <= source_width
            assert 0 <= top and top + crop_height <= source_height


# --- Clips AI segments -> crop segments ---------------------------------------------

def info(width: int = 1920, height: int = 1080) -> VideoInfo:
    return VideoInfo(width=width, height=height, frame_rate=25.0, duration=600.0)


def test_a_faceless_segment_holds_the_previous_crop():
    segments = [
        FakeSegment(10.0, 13.0, [0], x=200, y=0, found_face=True),
        FakeSegment(13.0, 16.0, [1], x=657, y=0, found_face=False),  # Clips AI centres this one
        FakeSegment(16.0, 20.0, [0], x=900, y=0, found_face=True),
    ]
    planned = reframe.to_crop_segments(segments, cut(10.0, 20.0), 606, 1080, info())
    assert [(s.x, s.fallback) for s in planned] == [(200, False), (200, True), (900, False)]


def test_a_faceless_first_segment_keeps_the_centre_crop():
    segments = [
        FakeSegment(0.0, 4.0, [0], x=657, y=0, found_face=False),
        FakeSegment(4.0, 8.0, [0], x=300, y=0, found_face=True),
    ]
    planned = reframe.to_crop_segments(segments, cut(0.0, 8.0), 606, 1080, info())
    assert [(s.x, s.fallback) for s in planned] == [(656, True), (300, False)]


def test_crop_segments_are_in_clip_time_and_cover_the_clip():
    segments = [
        FakeSegment(100.5, 104.0, [0], x=10, y=0),
        FakeSegment(104.2, 109.0, [1], x=900, y=0),  # a small hole left by scene merging
    ]
    planned = reframe.to_crop_segments(segments, cut(100.0, 110.0), 606, 1080, info())
    assert [(s.start, s.end) for s in planned] == [(0.0, 4.2), (4.2, 10.0)]
    assert [s.speaker for s in planned] == [0, 1]


def test_crop_segments_are_clamped():
    segments = [FakeSegment(0.0, 5.0, [0], x=1900, y=-30)]
    planned = reframe.to_crop_segments(segments, cut(0.0, 5.0), 606, 1080, info())
    assert (planned[0].x, planned[0].y) == (1314, 0)


# --- the crop plan round trip -------------------------------------------------------

def plan(segments: list[reframe.CropSegment], source=(1920, 1080)) -> reframe.CropPlan:
    return reframe.CropPlan(source[0], source[1], 606, 1080, 1080, 1920, segments)


def test_crop_plan_survives_a_round_trip_through_json():
    original = plan([
        reframe.CropSegment(0.0, 3.0, 0, 100, 0, False),
        reframe.CropSegment(3.0, 9.0, 1, 900, 0, True),
    ])
    assert reframe.CropPlan.from_dict(original.to_dict()) == original


def test_recorded_segments_carry_times_speaker_position_and_fallback():
    recorded = plan([reframe.CropSegment(0.0, 9.0, 1, 900, 12, True)]).to_dict()
    assert recorded["segments"] == [
        {"start": 0.0, "end": 9.0, "speaker": 1, "x": 900, "y": 12, "fallback": True}
    ]


def test_a_recorded_plan_is_reused_for_the_same_clip():
    recorded = plan([reframe.CropSegment(0.0, 9.0, 0, 100, 0, False)]).to_dict()
    assert reframe._reusable(recorded, cut(5.0, 14.0), info()) is not None


def test_a_recorded_plan_is_rejected_when_it_no_longer_fits():
    recorded = plan([reframe.CropSegment(0.0, 9.0, 0, 100, 0, False)]).to_dict()
    assert reframe._reusable(recorded, cut(5.0, 25.0), info()) is None  # different clip length
    assert reframe._reusable(recorded, cut(5.0, 14.0), info(1280, 720)) is None  # other source
    assert reframe._reusable(None, cut(5.0, 14.0), info()) is None
    assert reframe._reusable({"segments": []}, cut(5.0, 14.0), info()) is None


# --- the ffmpeg filter --------------------------------------------------------------

def test_one_segment_crops_at_a_fixed_position():
    chain = reframe.crop_filter(plan([reframe.CropSegment(0.0, 9.0, 0, 320, 0, False)]))
    assert chain == "crop=w=606:h=1080:x=320:y=0,scale=1080:1920:flags=lanczos,setsar=1"


def test_several_segments_move_the_crop_over_time():
    chain = reframe.crop_filter(plan([
        reframe.CropSegment(0.0, 3.5, 0, 100, 10, False),
        reframe.CropSegment(3.5, 6.25, 1, 900, 20, False),
        reframe.CropSegment(6.25, 9.0, 0, 100, 10, False),
    ]), eval_option=":eval=frame")
    assert chain == (
        "crop=w=606:h=1080"
        ":x='if(lt(t,3.500),100,if(lt(t,6.250),900,100))'"
        ":y='if(lt(t,3.500),10,if(lt(t,6.250),20,10))'"
        ":eval=frame"
        ",scale=1080:1920:flags=lanczos,setsar=1"
    )


def test_a_coordinate_that_never_changes_stays_a_constant():
    # Landscape sources crop sideways only, so y is the same for every segment.
    chain = reframe.crop_filter(plan([
        reframe.CropSegment(0.0, 4.0, 0, 100, 0, False),
        reframe.CropSegment(4.0, 9.0, 1, 900, 0, False),
    ]), eval_option=":eval=frame")
    assert chain == (
        "crop=w=606:h=1080:x='if(lt(t,4.000),100,900)':y=0:eval=frame"
        ",scale=1080:1920:flags=lanczos,setsar=1"
    )


def test_a_crop_that_never_moves_needs_no_per_frame_evaluation():
    chain = reframe.crop_filter(plan([
        reframe.CropSegment(0.0, 4.0, 0, 320, 0, False),
        reframe.CropSegment(4.0, 9.0, 1, 320, 0, False),
    ]), eval_option=":eval=frame")
    assert chain == "crop=w=606:h=1080:x=320:y=0,scale=1080:1920:flags=lanczos,setsar=1"


def test_the_eval_option_is_left_out_when_ffmpeg_does_not_take_it():
    chain = reframe.crop_filter(plan([
        reframe.CropSegment(0.0, 4.0, 0, 100, 0, False),
        reframe.CropSegment(4.0, 9.0, 1, 900, 0, False),
    ]), eval_option="")
    assert ":eval=frame" not in chain
    assert "x='if(lt(t,4.000),100,900)'" in chain


def test_the_commas_in_the_expression_stay_inside_quotes():
    chain = reframe.crop_filter(plan([
        reframe.CropSegment(0.0, 1.0, 0, 1, 2, False),
        reframe.CropSegment(1.0, 2.0, 1, 3, 4, False),
    ]), eval_option=":eval=frame")
    # ffmpeg splits a filter chain on commas, so every comma outside the quoted
    # expressions has to be one that really does separate two filters.
    unquoted = "".join(part for i, part in enumerate(chain.split("'")) if i % 2 == 0)
    assert unquoted.count(",") == 2  # crop -> scale -> setsar


def test_a_source_already_at_the_target_ratio_is_only_scaled():
    chain = reframe.crop_filter(
        reframe.CropPlan(1080, 1920, 1080, 1920, 1080, 1920,
                         [reframe.CropSegment(0.0, 9.0, 0, 0, 0, False)])
    )
    assert chain == "scale=1080:1920:flags=lanczos,setsar=1"


def test_the_filter_never_ends_with_a_separator():
    # The render step appends the caption filter to this chain, so a trailing comma
    # would produce an empty filter and ffmpeg would refuse the whole graph.
    chain = reframe.crop_filter(plan([reframe.CropSegment(0.0, 9.0, 0, 320, 0, False)]))
    assert not chain.endswith(",") and ",," not in chain


# --- the vendored copy --------------------------------------------------------------

def test_the_vendored_resize_package_keeps_its_licence():
    assert "MIT License" in (VENDORED / "LICENSE").read_text(encoding="utf-8")
    assert "Clips AI" in (VENDORED / "LICENSE").read_text(encoding="utf-8")


def test_the_vendored_resize_package_imports_no_whisperx_pyannote_or_clipsai():
    for module in sorted(VENDORED.glob("*.py")):
        imports = " ".join(
            line for line in module.read_text(encoding="utf-8").splitlines()
            if line.startswith(("import ", "from "))
        ).lower()
        for banned in ("clipsai", "whisperx", "pyannote"):
            assert banned not in imports, f"{module.name} still imports {banned}"


def test_the_vendored_resize_package_has_no_pyannote_entry_point():
    # clipsai/resize/resize.py was the Pyannote-diarizing orchestrator; utils/reframe.py
    # replaces it, driving the Resizer from the cached Scribe transcript instead.
    assert not (VENDORED / "resize.py").exists()
