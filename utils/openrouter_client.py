"""OpenRouter connector: one diarized, word-timestamped speech-to-text request per audio file.

Speaks OpenRouter's OpenAI-compatible /audio/transcriptions endpoint, which fronts
MAI-Transcribe-2 among others, and returns Scribe-shaped JSON so the rest of the
pipeline does not care which provider produced the transcript.
"""
import json
import logging
import os
import time
from pathlib import Path

import httpx

import config

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# The WAV is uploaded as multipart form data, so the chunk goes up in one write.
_TIMEOUT = httpx.Timeout(connect=30, read=900, write=900, pool=30)


def transcribe(audio_path: Path, attempts: int = 4) -> dict:
    """Return Scribe-shaped JSON: {language_code, text, words: [{text, start, end, type, speaker_id}]}."""
    api_key = os.getenv("OPEN_ROUTER")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER is not set (add it to .env)")

    raw = audio_path.read_bytes()
    if len(raw) > config.OPENROUTER_MAX_REQUEST_BYTES:
        raise RuntimeError(
            f"{audio_path.name} is {len(raw) / 1e6:.1f} MB, over OpenRouter's "
            f"{config.OPENROUTER_MAX_REQUEST_BYTES / 1e6:.0f} MB request cap. "
            "Lower TRANSCRIBE_CHUNK_SECONDS."
        )

    form = {
        "model": config.TRANSCRIBE_MODEL,
        # Word timestamps and per-word speakers both ride on the verbose response.
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["segment", "word"],
    }
    if config.TRANSCRIBE_LANGUAGE:
        form["language"] = config.TRANSCRIBE_LANGUAGE
    if config.TRANSCRIBE_DIARIZE:
        # Diarization is not a top-level field: it is passed through to the upstream
        # provider under its own slug, as a JSON string in the form body. OpenRouter
        # holds that provider relationship, so this needs no account of ours with them.
        # Without it the response comes back with no speakers at all, not merely one.
        form["provider"] = json.dumps(
            {"options": {config.OPENROUTER_PROVIDER_SLUG: {"diarization": {"enabled": True}}}}
        )

    for attempt in range(1, attempts + 1):
        try:
            response = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data=form,
                # Multipart sends the bytes as they are; base64 in a JSON body would
                # inflate every chunk by a third against the same 25 MB cap.
                files={"file": (audio_path.name, raw, "audio/wav")},
                timeout=_TIMEOUT,
            )
            if response.status_code == 200:
                return _to_scribe_shape(response.json())
            error = f"HTTP {response.status_code}: {response.text[:500]}"
            retryable = response.status_code in _RETRYABLE_STATUS
            if response.status_code == 402:
                error += "\nOpenRouter is out of credits; top up at https://openrouter.ai/credits"
        except httpx.TransportError as exc:
            error, retryable = f"{type(exc).__name__}: {exc}", True

        if not retryable or attempt == attempts:
            raise RuntimeError(f"OpenRouter transcription failed: {error}")
        wait = 5 * 2 ** (attempt - 1)
        log.warning("OpenRouter attempt %d failed (%s); retrying in %ds", attempt, error, wait)
        time.sleep(wait)
    raise AssertionError("unreachable")


def _to_scribe_shape(response: dict) -> dict:
    """Rewrite a verbose_json body into the Scribe response the pipeline already reads.

    Only the word list matters downstream. Speakers are labelled per word when the
    provider returns them there, and otherwise borrowed from the segment the word
    falls inside.
    """
    segments = [s for s in response.get("segments") or [] if s.get("start") is not None]
    usage = response.get("usage") or {}
    if usage.get("cost") is not None:
        log.info("OpenRouter billed $%.5f for %.0fs of audio", usage["cost"], usage.get("seconds") or 0)

    words = []
    for item in response.get("words") or []:
        # OpenAI's schema calls it "word"; some routes mirror Scribe and call it "text".
        text = (item.get("word") or item.get("text") or "").strip()
        if not text or item.get("start") is None:
            continue
        start = float(item["start"])
        end = float(item["end"]) if item.get("end") is not None else start
        speaker = item.get("speaker")
        if speaker is None:
            speaker = _speaker_at((start + end) / 2, segments)
        words.append({
            "text": text,
            "start": start,
            "end": end,
            "type": "word",
            # Speakers come back as bare integers, and the first of them is 0. Naming
            # them the way Scribe does keeps them truthy and printable, so neither the
            # `or` fallback downstream nor the speaker prefix on a chunked transcript
            # can quietly drop or trip over speaker zero.
            "speaker_id": None if speaker is None else f"speaker_{speaker}",
        })

    if not words:
        raise RuntimeError(
            f"{config.TRANSCRIBE_MODEL} returned no word timestamps; the pipeline cuts on word "
            "times, so it cannot use this transcript."
        )
    return {
        "language_code": response.get("language"),
        "text": response.get("text", ""),
        "words": words,
    }


def _speaker_at(when: float, segments: list[dict]) -> object | None:
    for segment in segments:
        end = segment.get("end")
        if segment["start"] <= when and (end is None or when <= end):
            return segment.get("speaker")
    return None
