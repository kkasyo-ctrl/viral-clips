"""ElevenLabs Scribe connector: one diarized, word-timestamped speech-to-text request per audio file."""
import logging
import os
import time
from pathlib import Path

import httpx

import config

log = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# Long audio is uploaded and transcribed in a single synchronous request.
_TIMEOUT = httpx.Timeout(connect=30, read=3600, write=3600, pool=30)


def transcribe(audio_path: Path, attempts: int = 4) -> dict:
    """Return Scribe's raw JSON response: {language_code, text, words: [{text, start, end, type, speaker_id}]}."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set (add it to .env)")

    form = {
        "model_id": config.SCRIBE_MODEL,
        "diarize": "true",
        "timestamps_granularity": "word",
        "tag_audio_events": "true",
    }
    if config.SCRIBE_LANGUAGE:
        form["language_code"] = config.SCRIBE_LANGUAGE

    for attempt in range(1, attempts + 1):
        try:
            with open(audio_path, "rb") as f:
                response = httpx.post(
                    API_URL,
                    headers={"xi-api-key": api_key},
                    data=form,
                    files={"file": (audio_path.name, f, "audio/wav")},
                    timeout=_TIMEOUT,
                )
            if response.status_code == 200:
                return response.json()
            error = f"HTTP {response.status_code}: {response.text[:500]}"
            retryable = response.status_code in _RETRYABLE_STATUS
        except httpx.TransportError as exc:
            error, retryable = f"{type(exc).__name__}: {exc}", True

        if not retryable or attempt == attempts:
            raise RuntimeError(f"Scribe transcription failed: {error}")
        wait = 5 * 2 ** (attempt - 1)
        log.warning("Scribe attempt %d failed (%s); retrying in %ds", attempt, error, wait)
        time.sleep(wait)
    raise AssertionError("unreachable")
