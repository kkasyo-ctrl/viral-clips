"""Step 2: Scribe transcription normalised into words and sentences, cached on a hash of the audio."""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config
from utils import audio, elevenlabs_client
from utils.audio import Silence

log = logging.getLogger(__name__)

_TERMINAL = (".", "?", "!", "…")
_CLOSERS = "\"'”’)]}»"
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "e.g.", "i.e."}


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Sentence:
    id: int
    start: float
    end: float
    speaker: str
    text: str
    first_word: int  # index into the word list
    last_word: int  # inclusive


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_or_transcribe(
    video: Path, wav: Path, audio_hash: str, duration: float, silences: list[Silence]
) -> tuple[list[Word], Path]:
    """Return the word list and the cache file it lives in. Scribe only runs on a cache miss."""
    cached = next(iter(sorted(config.TRANSCRIPTS_DIR.glob(f"*.{audio_hash[:16]}.json"))), None)
    if cached:
        log.info("Transcript cache hit: %s", cached.name)
        data = json.loads(cached.read_text(encoding="utf-8"))
    else:
        cached = config.TRANSCRIPTS_DIR / f"{video.stem}.{audio_hash[:16]}.json"
        data = _transcribe(video, wav, audio_hash, duration, silences)
        config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        partial = cached.with_name(cached.name + ".tmp")
        partial.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        partial.replace(cached)

    chunked = len(data["chunks"]) > 1
    words = []
    for i, chunk in enumerate(data["chunks"]):
        # Diarization is per request, so speaker_0 in one chunk need not be speaker_0 in the next.
        prefix = f"c{i}_" if chunked else ""
        words += normalize_words(chunk["response"], offset=chunk["start"], speaker_prefix=prefix)
    return words, cached


def _transcribe(video: Path, wav: Path, audio_hash: str, duration: float, silences: list[Silence]) -> dict:
    spans = plan_chunks(duration, wav.stat().st_size, silences)
    chunks = []
    for i, (start, end) in enumerate(spans):
        path = wav
        if len(spans) > 1:
            path = wav.with_name(f"chunk_{i:02d}.wav")
            audio.slice_wav(wav, start, end, path)
        log.info(
            "Transcribing %s-%s with %s (%d/%d)",
            format_timestamp(start), format_timestamp(end), config.SCRIBE_MODEL, i + 1, len(spans),
        )
        chunks.append({"start": start, "end": end, "response": elevenlabs_client.transcribe(path)})
        if path != wav:
            path.unlink()
    return {
        "audio_sha256": audio_hash,
        "source": video.name,
        "model": config.SCRIBE_MODEL,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration": duration,
        "chunks": chunks,
    }


def plan_chunks(duration: float, size_bytes: int, silences: list[Silence]) -> list[tuple[float, float]]:
    """One span if Scribe takes the whole file, else ~30-minute spans split in the middle of silences."""
    if duration <= config.SCRIBE_MAX_SECONDS and size_bytes <= config.SCRIBE_MAX_BYTES:
        return [(0.0, duration)]
    cuts = [0.0]
    while duration - cuts[-1] > config.SCRIBE_CHUNK_SECONDS * 1.25:
        mark = cuts[-1] + config.SCRIBE_CHUNK_SECONDS
        nearby = [
            s for s in silences
            if s.start > cuts[-1] and abs((s.start + s.end) / 2 - mark) <= config.SCRIBE_SPLIT_SEARCH_SECONDS
        ]
        if nearby:
            longest = max(nearby, key=lambda s: s.duration)
            cuts.append((longest.start + longest.end) / 2)
        else:
            log.warning("No silence near %s; splitting there anyway", format_timestamp(mark))
            cuts.append(mark)
    cuts.append(duration)
    return list(zip(cuts, cuts[1:]))


def normalize_words(response: dict, offset: float = 0.0, speaker_prefix: str = "") -> list[Word]:
    """Keep spoken words only (Scribe also returns spacing and audio-event entries), in absolute time."""
    words = []
    for item in response.get("words", []):
        text = (item.get("text") or "").strip()
        if item.get("type") != "word" or not text or item.get("start") is None:
            continue
        start = item["start"] + offset
        end = item["end"] + offset if item.get("end") is not None else start
        words.append(Word(text, start, end, speaker_prefix + (item.get("speaker_id") or "speaker")))
    return words


def _ends_sentence(text: str) -> bool:
    text = text.rstrip(_CLOSERS)
    return text.endswith(_TERMINAL) and text.lower() not in _ABBREVIATIONS


def group_sentences(words: list[Word]) -> list[Sentence]:
    """Split on terminal punctuation or a change of speaker."""
    sentences: list[Sentence] = []
    first = 0
    for i, word in enumerate(words):
        is_last = i + 1 == len(words)
        if is_last or _ends_sentence(word.text) or words[i + 1].speaker != word.speaker:
            sentences.append(Sentence(
                id=len(sentences),
                start=words[first].start,
                end=word.end,
                speaker=word.speaker,
                text=" ".join(w.text for w in words[first : i + 1]),
                first_word=first,
                last_word=i,
            ))
            first = i + 1
    return sentences


def format_line(sentence: Sentence) -> str:
    return f"[{sentence.id}] ({format_timestamp(sentence.start)}) {sentence.speaker}: {sentence.text}"
