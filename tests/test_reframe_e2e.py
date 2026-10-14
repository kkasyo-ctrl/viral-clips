"""End-to-end: a two-speaker sample, a mocked Scribe transcript, and real ffmpeg.

Nothing here reaches the network. The transcript is a Scribe-shaped dict with speaker
IDs on the words, the sample video is generated locally, and the rest is the pipeline's
own code. The one encode has to come out 1080x1920, keep the audio, hold the planned
crops and have the captions drawn into the picture.

`test_plan_crops_on_the_sample` additionally runs the real face analysis, and skips
when PySceneDetect, facenet-pytorch, mediapipe and PyAV are not installed.
"""
import json
import subprocess
from pathlib import Path

import pytest

import config
from utils import captions, cutting, reframe, render, transcript

SOURCE_WIDTH, SOURCE_HEIGHT = 1280, 720
SAMPLE_SECONDS = 12
# A stand-in for each speaker's face, one per side of the frame, at two clearly
# different greys: which one is in the middle of the vertical frame says which side the
# crop moved to. Both stay darker than the white caption text.
LEFT_BOX = "x=180:y=220:w=200:h=200:color=0x606060"   # speaker 0
RIGHT_BOX = "x=900:y=220:w=200:h=200:color=0xB4B4B4"  # speaker 1
BRIGHT = 200  # a grey level only the caption text reaches
BOX_ROWS = (700, 1050)  # where the boxes land once the crop is scaled to 1080x1920
BOX_COLUMNS = (270, 810)  # the middle of the frame, where a crop centred on one puts it


def scribe_response() -> dict:
    """A Scribe response: two speakers trading turns, with the spacing and audio-event
    entries Scribe really sends mixed in."""
    turns = [
        ("speaker_0", 0.2, ["so", "here", "is", "the", "thing", "nobody", "tells", "you"]),
        ("speaker_1", 3.4, ["wait", "really", "that", "cannot", "be", "right"]),
        ("speaker_0", 6.2, ["it", "is", "and", "that", "is", "the", "whole", "point"]),
        ("speaker_1", 9.4, ["okay", "that", "actually", "changes", "everything"]),
    ]
    words = []
    for speaker, start, texts in turns:
        clock = start
        for i, text in enumerate(texts):
            end = clock + 0.3
            last = i + 1 == len(texts)
            words.append({
                "text": text + ("." if last else ""),
                "start": round(clock, 3),
                "end": round(end, 3),
                "type": "word",
                "speaker_id": speaker,
            })
            words.append({"text": " ", "start": round(end, 3), "end": round(end + 0.05, 3),
                          "type": "spacing", "speaker_id": speaker})
            clock = end + 0.05
    words.append({"text": "(laughter)", "start": 11.6, "end": 11.9, "type": "audio_event"})
    return {"language_code": "en", "text": " ".join(w["text"] for w in words), "words": words}


@pytest.fixture(scope="module")
def sample(tmp_path_factory, ffmpeg) -> Path:
    """A short two-speaker video, generated locally."""
    path = tmp_path_factory.mktemp("sample") / "two_speakers.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=0x101010:s={SOURCE_WIDTH}x{SOURCE_HEIGHT}:r=25:d={SAMPLE_SECONDS}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={SAMPLE_SECONDS}",
            "-vf", f"drawbox={LEFT_BOX}:t=fill,drawbox={RIGHT_BOX}:t=fill",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return path


@pytest.fixture(scope="module")
def clip(sample):
    """The pipeline's own view of the sample: words, sentences and a planned cut."""
    words = transcript.normalize_words(scribe_response())
    sentences = transcript.group_sentences(words)
    cut = cutting.plan_cut(0, len(sentences) - 1, sentences, words, [], float(SAMPLE_SECONDS))
    return words, sentences, cut


def crop_plan(clip, info) -> reframe.CropPlan:
    """A crop plan from the real speaker segments, with the face analysis stood in for.

    Each speaker gets the side of the frame their stand-in sits on, which is what face
    analysis would conclude, without needing the models to run.
    """
    words, _, cut = clip
    segments = reframe.speaker_segments(cut, words)
    width, height = reframe.crop_size(info.width, info.height)
    centres = {0: 280, 1: 1000}  # the middle of each drawn box
    planned = []
    for segment in segments:
        x, y = reframe.clamp_crop(
            centres[segment.speaker] - width // 2, 0, width, height, info.width, info.height
        )
        planned.append(reframe.CropSegment(segment.start, segment.end, segment.speaker, x, y, False))
    return reframe.CropPlan(info.width, info.height, width, height, *reframe.output_size(), planned)


def gray_frame(path: Path, at: float, width: int, height: int) -> list[bytes]:
    """One frame of a rendered clip as rows of 8-bit grey, straight out of ffmpeg."""
    proc = subprocess.run(
        [
            config.FFMPEG, "-hide_banner", "-loglevel", "error",
            "-ss", f"{at:.3f}", "-i", str(path), "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True, check=True,
    )
    assert len(proc.stdout) == width * height, "unexpected frame size"
    return [proc.stdout[row * width : (row + 1) * width] for row in range(height)]


def bright_pixels(rows: list[bytes], top: int, bottom: int) -> int:
    return sum(sum(value > BRIGHT for value in row) for row in rows[top:bottom])


def box_brightness(rows: list[bytes]) -> float:
    """Mean grey where a crop centred on a speaker's stand-in puts that stand-in."""
    values = [v for row in rows[slice(*BOX_ROWS)] for v in row[slice(*BOX_COLUMNS)]]
    return sum(values) / len(values)


def probe(path: Path) -> dict:
    proc = subprocess.run(
        [config.FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def rendered(clip, sample, tmp_path_factory):
    """The sample reframed, captioned and encoded once, exactly as main.py does it."""
    words, sentences, cut = clip
    out_dir = tmp_path_factory.mktemp("clips")
    info = render.probe_video(sample)
    plan = crop_plan(clip, info)

    lines = captions.build_lines(cut, sentences, words)
    captions.write_ass(lines, out_dir / "clip.ass", plan.output_width, plan.output_height)
    captions.write_srt(lines, out_dir / "clip.srt")
    render.render_clip(
        sample, cut.start, cut.end,
        out_dir / "clip.ass", out_dir / "clip.mp4", reframe.crop_filter(plan),
    )
    return out_dir / "clip.mp4", plan, lines, cut


def test_the_clip_is_vertical(rendered):
    path, plan, _, _ = rendered
    stream = next(s for s in probe(path)["streams"] if s["codec_type"] == "video")
    assert (stream["width"], stream["height"]) == (plan.output_width, plan.output_height)
    assert (stream["width"], stream["height"]) == (1080, 1920)
    assert stream.get("sample_aspect_ratio", "1:1") in ("1:1", "")


def test_the_clip_keeps_its_audio_and_length(rendered):
    path, _, _, cut = rendered
    probed = probe(path)
    assert any(s["codec_type"] == "audio" for s in probed["streams"])
    assert float(probed["format"]["duration"]) == pytest.approx(cut.duration, abs=0.3)


def test_the_captions_are_burned_into_the_picture(rendered):
    path, plan, lines, _ = rendered
    height, width = plan.output_height, plan.output_width
    line = lines[len(lines) // 2]
    rows = gray_frame(path, (line[0].start + line[-1].end) / 2, width, height)

    band_top = height - round(height * (config.CAPTION_SAFE_BOTTOM + 0.12))
    band_bottom = height - round(height * config.CAPTION_SAFE_BOTTOM)
    assert bright_pixels(rows, band_top, band_bottom) > 300, "no caption text in the lower third"
    # The background is far darker than the text, so bright pixels above the boxes or
    # below the band would mean a caption had been drawn outside the safe zone.
    assert bright_pixels(rows, 0, BOX_ROWS[0] - 200) == 0
    assert bright_pixels(rows, band_bottom, height) == 0


def test_the_crop_follows_the_speaker(rendered):
    """The crop should move to whichever speaker's side the segment belongs to."""
    path, plan, _, _ = rendered
    brightness = {}
    for segment in plan.segments:
        rows = gray_frame(
            path, (segment.start + segment.end) / 2, plan.output_width, plan.output_height
        )
        brightness.setdefault(segment.speaker, []).append(box_brightness(rows))

    assert set(brightness) == {0, 1}, "both speakers should get a segment"
    assert all(m > 50 for means in brightness.values() for m in means), "no stand-in in view"
    # Speaker 0's stand-in is the dark box on the left, speaker 1's the bright one on the
    # right. A crop frozen on one speaker would read the same grey in every segment.
    assert max(brightness[0]) + 40 < min(brightness[1])


def test_the_srt_timings_are_the_clip_timings(rendered):
    path, _, _, cut = rendered
    srt = (path.parent / "clip.srt").read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:00,")
    last_end = srt.strip().splitlines()[-2].split(" --> ")[1]
    hours, minutes, rest = last_end.split(":")
    seconds = float(rest.replace(",", "."))
    assert int(hours) * 3600 + int(minutes) * 60 + seconds <= cut.duration + 0.75


def test_rendering_refuses_to_drop_the_captions(rendered, sample, tmp_path):
    """A clip without its caption file is an error, not a silently bare render."""
    _, plan, _, cut = rendered
    with pytest.raises(RuntimeError, match="burned in"):
        render.render_clip(
            sample, cut.start, cut.end,
            tmp_path / "missing.ass", tmp_path / "out.mp4", reframe.crop_filter(plan),
        )


def test_plan_crops_on_the_sample(clip, sample, reframe_models):
    """The real crop planner: scene detection plus MTCNN over the clip's cut window."""
    words, _, cut = clip
    info = render.probe_video(sample)
    segments = reframe.speaker_segments(cut, words)
    plan = reframe.plan_crops(sample, cut, segments, info)

    assert (plan.crop_width, plan.crop_height) == reframe.crop_size(info.width, info.height)
    assert (plan.output_width, plan.output_height) == reframe.output_size()
    assert plan.segments[0].start == 0.0
    assert plan.segments[-1].end == pytest.approx(cut.duration, abs=0.05)
    assert all(a.end == b.start for a, b in zip(plan.segments, plan.segments[1:]))
    for segment in plan.segments:
        assert 0 <= segment.x and segment.x + plan.crop_width <= info.width
        assert 0 <= segment.y and segment.y + plan.crop_height <= info.height
    # Drawn boxes are not faces, so this sample exercises the fallback crops.
    assert all(isinstance(segment.fallback, bool) for segment in plan.segments)


def test_a_recorded_plan_is_reused_instead_of_reanalysed(clip, sample, monkeypatch):
    """Re-rendering reads the crops back out of moments.json rather than redoing faces."""
    words, _, cut = clip
    info = render.probe_video(sample)
    recorded = crop_plan(clip, info).to_dict()

    def fail(*args, **kwargs):
        raise AssertionError("face analysis ran again on a re-render")

    monkeypatch.setattr(reframe, "plan_crops", fail)
    reused = reframe.plan(sample, cut, words, info, recorded)
    assert reused.to_dict() == recorded
