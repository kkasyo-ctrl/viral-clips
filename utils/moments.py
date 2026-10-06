"""Step 3: find candidate moments in the sentence-level transcript.

One pass over the whole transcript; 8-minute windows with a 6-minute stride only when the transcript is
too long for the model's input.
"""
import logging
from collections.abc import Iterator

import config
from utils import openai_client
from utils.openai_client import Moment
from utils.transcript import Sentence, format_line, format_timestamp

log = logging.getLogger(__name__)


def find_moments(sentences: list[Sentence]) -> list[Moment]:
    if not sentences:
        return []
    transcript = "\n".join(format_line(s) for s in sentences)
    if len(transcript) / 4 <= config.MOMENTS_MAX_INPUT_TOKENS:  # ~4 characters per token
        found = openai_client.extract_moments(transcript)
    else:
        windows = list(_windows(sentences))
        log.info("Transcript exceeds the input limit; scanning %d overlapping windows", len(windows))
        found = []
        for window in windows:
            note = (
                f"This is an excerpt ({format_timestamp(window[0].start)}-{format_timestamp(window[-1].end)}) "
                "of a longer video. Only return moments that lie fully inside it."
            )
            found += openai_client.extract_moments("\n".join(format_line(s) for s in window), note)
    return _merge_overlaps(_in_range(found, len(sentences)), sentences)


def _windows(sentences: list[Sentence]) -> Iterator[list[Sentence]]:
    start, last = 0.0, sentences[-1].start
    while True:
        window = [s for s in sentences if start <= s.start < start + config.MOMENTS_WINDOW_SECONDS]
        if window:
            yield window
        if start + config.MOMENTS_WINDOW_SECONDS > last:
            return
        start += config.MOMENTS_STRIDE_SECONDS


def _in_range(moments: list[Moment], n_sentences: int) -> list[Moment]:
    valid = []
    for m in moments:
        first, last = sorted((m.start_sentence, m.end_sentence))
        if last < 0 or first >= n_sentences:
            log.warning("Dropping moment with out-of-range sentence ids: %s", m.title)
            continue
        valid.append(m.model_copy(update={"start_sentence": max(first, 0), "end_sentence": min(last, n_sentences - 1)}))
    return valid


def _merge_overlaps(moments: list[Moment], sentences: list[Sentence]) -> list[Moment]:
    """Fold together moments covering mostly the same span (e.g. found twice by overlapping windows)."""

    def span(first: int, last: int) -> float:
        return sentences[last].end - sentences[first].start

    merged: list[Moment] = []
    for m in sorted(moments, key=lambda m: (m.start_sentence, m.end_sentence)):
        if merged:
            prev = merged[-1]
            lo, hi = max(prev.start_sentence, m.start_sentence), min(prev.end_sentence, m.end_sentence)
            prev_len, m_len = span(prev.start_sentence, prev.end_sentence), span(m.start_sentence, m.end_sentence)
            if lo <= hi and span(lo, hi) >= 0.5 * min(prev_len, m_len):
                keep = prev if prev_len >= m_len else m
                merged[-1] = keep.model_copy(update={
                    "start_sentence": prev.start_sentence,
                    "end_sentence": max(prev.end_sentence, m.end_sentence),
                })
                continue
        merged.append(m)
    return merged
