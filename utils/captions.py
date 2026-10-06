"""Step 7: karaoke captions as ASS (burned in) plus a plain SRT sidecar, straight from Scribe word timings."""
from dataclasses import dataclass
from pathlib import Path

import config
from utils.cutting import Cut, sentence_word_ranges
from utils.transcript import Sentence, Word

HOLD_SECONDS = 0.6  # how long a line lingers after its last word when nobody starts talking


@dataclass
class CaptionWord:
    text: str
    start: float  # seconds from the start of the clip
    end: float


def build_lines(cut: Cut, sentences: list[Sentence], words: list[Word]) -> list[list[CaptionWord]]:
    """Lines of 2-4 words or ~1.8 s, never crossing the end of a sentence."""
    lines: list[list[CaptionWord]] = []
    for _, lo, hi in sentence_word_ranges(cut, sentences):
        sentence_lines: list[list[CaptionWord]] = []
        for w in words[lo : hi + 1]:
            word = CaptionWord(w.text, w.start - cut.start, w.end - cut.start)
            line = sentence_lines[-1] if sentence_lines else None
            if (
                line is None
                or len(line) >= config.CAPTION_MAX_WORDS
                or (len(line) >= config.CAPTION_MIN_WORDS and word.end - line[0].start > config.CAPTION_MAX_LINE_SECONDS)
            ):
                sentence_lines.append([word])
            else:
                line.append(word)
        # Don't leave a single word stranded at the end of a sentence.
        if len(sentence_lines) >= 2 and len(sentence_lines[-1]) == 1:
            previous = sentence_lines[-2]
            if len(previous) < config.CAPTION_MAX_WORDS:
                previous.extend(sentence_lines.pop())
            else:
                sentence_lines[-1].insert(0, previous.pop())
        lines.extend(sentence_lines)
    return lines


def _slots(lines: list[list[CaptionWord]]) -> list[list[tuple[int, int]]]:
    """Centisecond (start, end) per word: each word is lit until the next one starts; the last word of a line
    holds briefly. Slots touch but never overlap, because overlapping events make libass stack lines."""
    slots, cursor = [], 0
    for i, line in enumerate(lines):
        line_end = line[-1].end + HOLD_SECONDS
        if i + 1 < len(lines):
            line_end = min(line_end, lines[i + 1][0].start)
        bounds = []
        for t in [w.start for w in line] + [line_end]:
            cursor = max(cursor, round(t * 100))
            bounds.append(cursor)
        slots.append(list(zip(bounds, bounds[1:])))
    return slots


def write_ass(lines: list[list[CaptionWord]], path: Path, width: int, height: int) -> None:
    font_size = round(min(width, height) * 0.075)
    outline = max(2, round(font_size * 0.09))
    margin_v = round(height * (1 - config.CAPTION_BASELINE))  # bottom-anchored, so the baseline sits at ~72%
    margin_h = round(width * 0.06)
    r, g, b = (config.CAPTION_HIGHLIGHT.lstrip("#")[i : i + 2] for i in (0, 2, 4))
    highlight = f"&H{b}{g}{r}&".upper()

    out = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Caption,{config.CAPTION_FONT},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,"
        f"100,100,0,0,1,{outline},0,2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # One event per word, each showing the whole line with that word recoloured. Steadier across libass
    # versions than \k karaoke timing.
    for line, line_slots in zip(lines, _slots(lines)):
        for active, (start, end) in enumerate(line_slots):
            if end <= start:
                continue
            text = " ".join(
                f"{{\\c{highlight}}}{_ass_safe(w.text)}{{\\r}}" if i == active else _ass_safe(w.text)
                for i, w in enumerate(line)
            )
            out.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{text}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_srt(lines: list[list[CaptionWord]], path: Path) -> None:
    blocks = []
    for line, line_slots in zip(lines, _slots(lines)):
        start, end = line_slots[0][0], line_slots[-1][1]
        if end <= start:
            continue
        text = " ".join(w.text for w in line)
        blocks.append(f"{len(blocks) + 1}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def _ass_safe(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _ass_time(cs: int) -> str:
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _srt_time(cs: int) -> str:
    return f"{cs // 360000:02d}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d},{cs % 100 * 10:03d}"
