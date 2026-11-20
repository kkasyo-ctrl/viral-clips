"""The gaming format: facecam detection, the stacked layout, captions on the seam and the
loudness hints. Detection is tested on synthetic detections and frames, so no face model
loads; the render tests generate a stream-like sample and run real ffmpeg."""
import subprocess
import wave

import numpy as np
import pytest

import config
import pipeline
from utils import audio, captions, cutting, gaming, moments, openai_client, render, transcript
from utils.audio import Peak
from utils.cutting import Cut
from utils.gaming import Facecam, Rect
from utils.render import VideoInfo
from utils.transcript import Sentence

HD = VideoInfo(1920, 1080, 30.0, 600.0)
CAM = Facecam(overlay=Rect(1440, 0, 480, 270), face=Rect(1620, 40, 100, 120))


# --- layout -------------------------------------------------------------------------

def test_cam_on_top_game_below():
    plan = gaming.layout(HD, CAM, 1 / 3)
    assert (plan.output_width, plan.output_height) == (1080, 1920)
    assert plan.cam_height == 640 and plan.seam == 640
    cam, overlay = plan.cam, CAM.overlay
    # The cam window stays inside the overlay, at the top panel's shape, around the face.
    assert overlay.x <= cam.x and cam.x + cam.w <= overlay.x + overlay.w
    assert overlay.y <= cam.y and cam.y + cam.h <= overlay.y + overlay.h
    assert cam.w / cam.h == pytest.approx(1080 / 640, rel=0.01)
    assert cam.x <= CAM.face.cx <= cam.x + cam.w
    # The game is the middle of the frame, at the bottom panel's shape.
    game = plan.game
    assert game.w / game.h == pytest.approx(1080 / 1280, rel=0.01)
    assert game.cx == pytest.approx(HD.width / 2, abs=0.05 * HD.width)  # may slide a little off the cam
    assert all(v % 2 == 0 for v in (*cam.to_list(), *game.to_list()))


def test_the_game_window_slides_off_the_cam():
    # At half height the game window is wide enough to take in the cam in the top right.
    plan = gaming.layout(HD, CAM, 0.5)
    assert plan.game.x + plan.game.w <= CAM.overlay.x
    # A cam on the left pushes it the other way.
    left = Facecam(Rect(0, 0, 480, 270), Rect(180, 40, 100, 120))
    assert gaming.layout(HD, left, 0.5).game.x >= 480
    # It moves no further than MAX_GAME_SHIFT, even if the cam is still partly in view.
    wide = Facecam(Rect(1000, 0, 920, 400), Rect(1400, 40, 100, 120))
    centred = gaming.game_window(1920, 1080, 1080 / 960)
    assert centred.x - gaming.layout(HD, wide, 0.5).game.x <= gaming.MAX_GAME_SHIFT * 1920 + 2


def test_cam_share_is_held_to_its_range():
    assert gaming.layout(HD, CAM, 0.9).cam_height == 960
    assert gaming.layout(HD, CAM, 0.1).cam_height == 480


def test_layout_without_a_cam():
    plan = gaming.layout(HD, None, 1 / 3)
    assert plan.cam is None and plan.seam is None
    chain = gaming.layout_filter(plan)
    assert "gblur" in chain and "overlay=(W-w)/2:(H-h)/2" in chain


def test_layout_filter_stacks_the_two_crops():
    plan = gaming.layout(HD, CAM, 1 / 3)
    chain = gaming.layout_filter(plan)
    cam, game = plan.cam, plan.game
    assert f"crop={cam.w}:{cam.h}:{cam.x}:{cam.y},scale=1080:640" in chain
    assert f"crop={game.w}:{game.h}:{game.x}:{game.y},scale=1080:1280" in chain
    assert chain.endswith("[top][bottom]vstack")


# --- finding the cam ----------------------------------------------------------------

def _detections(static_frames: int, total: int = 40, face=(1600, 60, 1720, 200)) -> list:
    """A steady face in the first static_frames frames, plus a game character crossing the screen."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(total):
        boxes = [[100 + 40 * i, 500, 180 + 40 * i, 600]]
        if i < static_frames:
            boxes.append([v + rng.integers(-3, 4) for v in face])
        out.append(np.array(boxes, dtype=np.int16))
    out[5] = None  # MTCNN returns None for a frame without faces
    return out


def test_the_steady_face_is_the_cam():
    face = gaming.steady_face(_detections(30), 1920, 1080)
    assert face is not None
    assert abs(face.x - 1600) <= 3 and abs(face.y - 60) <= 3
    assert abs(face.w - 120) <= 6 and abs(face.h - 140) <= 6


def test_a_face_seen_too_rarely_is_not_a_cam():
    assert gaming.steady_face(_detections(10), 1920, 1080) is None


def test_a_full_screen_camera_is_not_a_cam():
    assert gaming.steady_face(_detections(40, face=(700, 200, 1100, 800)), 1920, 1080) is None


def _stream_frames(cam_still: bool = True, n: int = 40) -> list:
    """Shrunk greyscale frames of a stream: noise for the gameplay, which changes every
    sample, and a cam overlay at rows 10-79, columns 350-469 that barely does."""
    rng = np.random.default_rng(1)
    room = rng.integers(0, 255, (70, 120))
    frames = []
    for _ in range(n):
        frame = rng.integers(0, 255, (270, 480)).astype(np.float32)
        if cam_still:
            frame[10:80, 350:470] = room + rng.normal(0, 2, room.shape)
            frame[25:50, 390:420] += rng.normal(0, 12, (25, 30))  # the streamer moves a little
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def test_the_overlay_is_the_still_box_around_the_face():
    face = Rect(390 * 4, 25 * 4, 30 * 4, 25 * 4)  # analysis frames are a quarter of 1920x1080
    overlay = gaming.overlay_around(face, _stream_frames(), 1920, 1080)
    expected = Rect(350 * 4, 10 * 4, 120 * 4, 70 * 4)
    for got, want, size in zip(overlay.to_list(), expected.to_list(), (480, 280, 480, 280)):
        assert abs(got - want) <= 0.08 * size, (overlay, expected)  # less the border trim


def test_no_still_box_frames_the_face_instead():
    face = Rect(390 * 4, 25 * 4, 30 * 4, 25 * 4)
    overlay = gaming.overlay_around(face, _stream_frames(cam_still=False), 1920, 1080)
    assert overlay == gaming.around_face(face, 1920, 1080)
    assert overlay.contains(face.cx, face.cy)


# --- loudness -----------------------------------------------------------------------

def _wav(path, seconds: float, loud: tuple[float, float]) -> None:
    rate = config.AUDIO_SAMPLE_RATE
    t = np.arange(int(seconds * rate)) / rate
    amplitude = np.where((t >= loud[0]) & (t < loud[1]), 0.5, 0.05)
    samples = (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def test_a_shout_is_a_peak(tmp_path):
    _wav(tmp_path / "a.wav", 30, (10, 12))
    peaks = audio.loudness_peaks(tmp_path / "a.wav")
    assert len(peaks) == 1
    assert peaks[0].start == pytest.approx(10, abs=0.5) and peaks[0].end == pytest.approx(12, abs=0.5)
    assert peaks[0].gain_db == pytest.approx(20, abs=1)


def test_an_even_voice_has_no_peaks(tmp_path):
    _wav(tmp_path / "a.wav", 30, (0, 0))
    assert audio.loudness_peaks(tmp_path / "a.wav") == []


def _sentences() -> list[Sentence]:
    return [
        Sentence(0, 0.0, 4.0, "speaker_0", "Okay we go in quiet.", 0, 4),
        Sentence(1, 4.5, 6.0, "speaker_0", "No no no!", 5, 7),
        Sentence(2, 9.0, 12.0, "speaker_0", "Anyway, back to the farm.", 8, 12),
    ]


def test_gaming_lines_carry_their_loudness(monkeypatch):
    seen = {}

    def extract(text, note=None, clip_format="podcast"):
        seen[clip_format] = text
        return []

    monkeypatch.setattr(openai_client, "extract_moments", extract)
    peaks = [Peak(5.0, 6.5, 14.2)]
    moments.find_moments(_sentences(), "gaming", peaks)
    moments.find_moments(_sentences(), "podcast", peaks)
    gaming_lines = seen["gaming"].splitlines()
    assert gaming_lines[1].endswith("[LOUD +14dB]")
    assert "LOUD" not in gaming_lines[0] and "LOUD" not in gaming_lines[2]
    assert "LOUD" not in seen["podcast"]


def test_a_reaction_just_after_a_sentence_counts_for_it():
    assert moments.loud_sentences(_sentences(), [Peak(4.3, 4.4, 9.0)]) == {0: 9.0}


def test_loud_moments_are_in_clip_time():
    cut = Cut(start=100.0, end=130.0, first_sentence=0, last_sentence=0, first_word=0, last_word=0)
    peaks = [Peak(50, 51, 20), Peak(104.2, 105, 11.4), Peak(170, 171, 9)]
    assert pipeline.loud_moments(cut, peaks) == "0:04 +11 dB"
    assert pipeline.loud_moments(cut, []) == "none"


def test_the_gaming_prompt_is_used_for_gaming():
    assert openai_client.PROMPTS["gaming"] is openai_client.GAMING_PROMPT
    assert "[LOUD" in openai_client.GAMING_PROMPT


# --- captions on the seam -------------------------------------------------------------

def test_captions_can_sit_on_the_seam(tmp_path):
    lines = [[captions.CaptionWord("no", 0.0, 0.4), captions.CaptionWord("way", 0.4, 0.8)]]
    captions.write_ass(lines, tmp_path / "seam.ass", 1080, 1920, center_y=640)
    text = (tmp_path / "seam.ass").read_text(encoding="utf-8")
    style = next(line for line in text.splitlines() if line.startswith("Style:"))
    assert style.split(",")[18] == "5"  # middle centre
    assert all("{\\pos(540,640)}" in line for line in text.splitlines() if line.startswith("Dialogue:"))


# --- rendering ----------------------------------------------------------------------

SOURCE = (1280, 720)
CAM_BOX = Rect(960, 0, 320, 180)  # a mid-grey "cam" in the top-right corner
SAMPLE_SECONDS = 8


@pytest.fixture(scope="module")
def stream(tmp_path_factory, ffmpeg):
    """A stream-like sample: a dark game with a box moving across it, and a still cam."""
    path = tmp_path_factory.mktemp("stream") / "stream.mp4"
    w, h = SOURCE
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=0x101010:s={w}x{h}:r=25:d={SAMPLE_SECONDS}",
            "-f", "lavfi", "-i", f"color=c=0x404040:s=160x160:r=25:d={SAMPLE_SECONDS}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={SAMPLE_SECONDS}",
            "-filter_complex",
            f"[0][1]overlay=x='mod(t*150,900)':y=400,"
            f"drawbox=x={CAM_BOX.x}:y={CAM_BOX.y}:w={CAM_BOX.w}:h={CAM_BOX.h}:color=0x808080:t=fill[v]",
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return path


def _words() -> list:
    texts = ["no", "way", "he", "actually", "jumped", "the", "bridge", "chat"]
    response = {"words": [
        {"text": t + ("." if i == len(texts) - 1 else ""), "start": 0.5 + 0.6 * i, "end": 0.9 + 0.6 * i,
         "type": "word", "speaker_id": "speaker_0"}
        for i, t in enumerate(texts)
    ]}
    return transcript.normalize_words(response)


def _render(stream, folder, facecam):
    words = _words()
    sentences = transcript.group_sentences(words)
    cut = cutting.plan_cut(0, len(sentences) - 1, sentences, words, [], float(SAMPLE_SECONDS))
    info = render.probe_video(stream)
    plan = gaming.layout(info, facecam, 1 / 3)
    lines = captions.build_lines(cut, sentences, words)
    captions.write_ass(lines, folder / "clip.ass", plan.output_width, plan.output_height, center_y=plan.seam)
    render.render_clip(stream, cut.start, cut.end, folder / "clip.ass", folder / "clip.mp4", gaming.layout_filter(plan))
    return folder / "clip.mp4", plan, lines


def _gray(path, at: float, width: int, height: int) -> np.ndarray:
    proc = subprocess.run(
        [config.FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", str(path),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(height, width)


def test_a_gaming_clip_has_the_cam_on_top_and_captions_on_the_seam(stream, tmp_path):
    facecam = Facecam(CAM_BOX, Rect(1080, 40, 80, 90))
    path, plan, lines = _render(stream, tmp_path, facecam)
    frame = _gray(path, (lines[0][0].start + lines[0][-1].end) / 2, 1080, 1920)
    seam = plan.seam

    assert 110 < frame[20 : seam - 200, 40:1040].mean() < 150, "the top panel should be the grey cam"
    assert frame[seam + 300 :, :].mean() < 70, "the bottom panel should be the dark game"
    assert (frame[seam - 150 : seam + 150] > 200).sum() > 300, "no caption text on the seam"
    assert (frame[seam + 300 :] > 200).sum() == 0, "a caption was drawn away from the seam"


def test_a_gaming_clip_without_a_cam_still_renders_vertical(stream, tmp_path):
    path, plan, _ = _render(stream, tmp_path, None)
    proc = subprocess.run(
        [config.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "1080,1920"
