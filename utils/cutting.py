"""Step 6: turn a sentence range into cut points that sit on real word boundaries."""
import re
from collections.abc import Iterator
from dataclasses import dataclass

import config
from utils.audio import Silence
from utils.transcript import Sentence, Word


@dataclass(frozen=True)
class Cut:
    start: float  # seconds in the source video
    end: float
    first_sentence: int
    last_sentence: int
    first_word: int
    last_word: int  # inclusive; may stop short of last_sentence's end if one sentence exceeded the cap

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_cut(
    first_sentence: int,
    last_sentence: int,
    sentences: list[Sentence],
    words: list[Word],
    silences: list[Silence],
    media_end: float,
) -> Cut:
    s, e = first_sentence, last_sentence

    def start_at(sentence: Sentence) -> float:
        w = sentence.first_word
        t = words[w].start - config.START_PAD_SECONDS
        if w > 0:  # don't reach back into the previous word
            t = max(t, min(words[w - 1].end, words[w].start))
        return max(t, 0.0)

    def end_at(w: int) -> float:
        t = words[w].end + config.END_PAD_SECONDS
        if w + 1 < len(words):  # don't run into the next word
            t = min(t, max(words[w + 1].start, words[w].end))
        return min(t, media_end)

    # Never open on a filler or conjunction.
    while s < e and _opens_on_filler(sentences[s], words):
        s += 1
    # Prefer an opening right after a real pause, but only by stepping over short fragments ("Okay.",
    # "Really?") so the sentence the moment was chosen for is never dropped.
    if not _follows_pause(sentences[s], words, silences):
        horizon = sentences[s].start + config.START_SEARCH_SECONDS
        i = s
        while i < e and _is_fragment(sentences[i]) and sentences[i + 1].start <= horizon:
            i += 1
            if _follows_pause(sentences[i], words, silences) and not _opens_on_filler(sentences[i], words):
                s = i
                break
    start = start_at(sentences[s])

    # Too long: drop whole sentences from the end, then words if a single sentence is still too long.
    while e > s and end_at(sentences[e].last_word) - start > config.MAX_CLIP_SECONDS:
        e -= 1
    last_word = sentences[e].last_word
    while last_word > sentences[e].first_word and end_at(last_word) - start > config.MAX_CLIP_SECONDS:
        last_word -= 1

    # Too short: take in following sentences while they fit under the cap.
    while end_at(last_word) - start < config.MIN_CLIP_SECONDS and e + 1 < len(sentences):
        if end_at(sentences[e + 1].last_word) - start > config.MAX_CLIP_SECONDS:
            break
        e += 1
        last_word = sentences[e].last_word

    return Cut(start, end_at(last_word), s, e, sentences[s].first_word, last_word)


def _opens_on_filler(sentence: Sentence, words: list[Word]) -> bool:
    first = re.sub(r"[^\w']", "", words[sentence.first_word].text.lower())
    return first in config.FILLER_OPENERS


def _is_fragment(sentence: Sentence) -> bool:
    return sentence.last_word - sentence.first_word + 1 <= config.START_FRAGMENT_WORDS


def _follows_pause(sentence: Sentence, words: list[Word], silences: list[Silence]) -> bool:
    if sentence.first_word == 0:
        return True
    onset = words[sentence.first_word].start
    # silencedetect's end and Scribe's word start disagree by a little; allow some slack either way
    return any(
        s.duration >= config.PAUSE_MIN_SECONDS and onset - 0.35 <= s.end <= onset + 0.15
        for s in silences
    )


def sentence_word_ranges(cut: Cut, sentences: list[Sentence]) -> Iterator[tuple[Sentence, int, int]]:
    """Each sentence in the cut with the (inclusive) word index range of it that's inside the cut."""
    for sentence in sentences[cut.first_sentence : cut.last_sentence + 1]:
        lo, hi = max(sentence.first_word, cut.first_word), min(sentence.last_word, cut.last_word)
        if lo <= hi:
            yield sentence, lo, hi


def clip_transcript(cut: Cut, sentences: list[Sentence], words: list[Word]) -> str:
    """Speaker-labelled text of exactly the words that end up in the clip."""
    turns: list[list[str]] = []
    for sentence, lo, hi in sentence_word_ranges(cut, sentences):
        text = " ".join(w.text for w in words[lo : hi + 1])
        if turns and turns[-1][0] == sentence.speaker:
            turns[-1][1] += " " + text
        else:
            turns.append([sentence.speaker, text])
    return "\n".join(f"{speaker}: {text}" for speaker, text in turns)
