"""Where the burned-in captions land, in a vertical frame and in a landscape one."""
import config
from utils import captions
from utils.captions import CaptionWord

VERTICAL = (1080, 1920)
LANDSCAPE = (1920, 1080)


def lines() -> list[list[CaptionWord]]:
    return [
        [CaptionWord("this", 0.0, 0.4), CaptionWord("is", 0.4, 0.6), CaptionWord("the", 0.6, 0.9)],
        [CaptionWord("part", 1.0, 1.4), CaptionWord("that", 1.4, 1.7), CaptionWord("lands", 1.7, 2.2)],
    ]


def style(path) -> dict[str, str]:
    """The Caption style row of an ASS file, as a dict."""
    text = path.read_text(encoding="utf-8")
    fields = [f.strip() for f in next(
        line for line in text.splitlines() if line.startswith("Format: Name, Fontname")
    ).removeprefix("Format:").split(",")]
    values = next(
        line for line in text.splitlines() if line.startswith("Style: Caption")
    ).removeprefix("Style:").split(",")
    return dict(zip(fields, [v.strip() for v in values]))


def test_vertical_captions_sit_in_the_lower_third(tmp_path):
    path = tmp_path / "vertical.ass"
    captions.write_ass(lines(), path, *VERTICAL)
    width, height = VERTICAL
    caption = style(path)

    # Alignment 2 is bottom-centre, so MarginV is the gap below the text.
    assert caption["Alignment"] == "2"
    baseline = height - int(caption["MarginV"])
    assert height * 2 / 3 < baseline < height, "captions should sit in the lower third"


def test_vertical_captions_clear_the_platform_safe_zone(tmp_path):
    path = tmp_path / "vertical.ass"
    captions.write_ass(lines(), path, *VERTICAL)
    width, height = VERTICAL
    caption = style(path)

    # Nothing may reach into the bottom band the platforms draw their own caption,
    # handle and buttons over, nor into the side columns.
    assert int(caption["MarginV"]) >= round(height * config.CAPTION_SAFE_BOTTOM)
    assert int(caption["MarginL"]) >= round(width * config.CAPTION_SAFE_SIDE)
    assert int(caption["MarginR"]) >= round(width * config.CAPTION_SAFE_SIDE)


def test_the_ass_is_written_for_the_frame_it_will_be_drawn_on(tmp_path):
    path = tmp_path / "vertical.ass"
    captions.write_ass(lines(), path, *VERTICAL)
    text = path.read_text(encoding="utf-8")
    # libass scales the script to the frame, so a landscape PlayRes on a vertical clip
    # would stretch and mis-place every line.
    assert "PlayResX: 1080" in text and "PlayResY: 1920" in text


def test_landscape_captions_keep_the_old_baseline(tmp_path):
    path = tmp_path / "landscape.ass"
    captions.write_ass(lines(), path, *LANDSCAPE)
    width, height = LANDSCAPE
    assert int(style(path)["MarginV"]) == round(height * (1 - config.CAPTION_BASELINE))


def test_every_word_gets_its_own_highlighted_event(tmp_path):
    path = tmp_path / "vertical.ass"
    captions.write_ass(lines(), path, *VERTICAL)
    events = [line for line in path.read_text(encoding="utf-8").splitlines()
              if line.startswith("Dialogue:")]
    # One event per word, each showing its whole line with that word recoloured.
    assert len(events) == 6
    highlight = config.CAPTION_HIGHLIGHT.lstrip("#")
    expected = f"&H{highlight[4:6]}{highlight[2:4]}{highlight[0:2]}&".upper()
    assert all(expected in event for event in events)


def test_the_srt_does_not_depend_on_the_frame_size(tmp_path):
    vertical, landscape = tmp_path / "a.srt", tmp_path / "b.srt"
    captions.write_srt(lines(), vertical)
    captions.write_srt(lines(), landscape)
    # Reframing changes the picture, never the timings.
    assert vertical.read_text(encoding="utf-8") == landscape.read_text(encoding="utf-8")
    assert "00:00:00,000 --> " in vertical.read_text(encoding="utf-8")
