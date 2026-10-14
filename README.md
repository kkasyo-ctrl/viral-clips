# viral-clips
A repo that extracts viral moments from long videos

## Pipeline

1. **Audio**: ffmpeg pulls a 16 kHz mono WAV and runs `silencedetect` in the same pass.
2. **Transcription**: ElevenLabs Scribe v1 (diarized, word timestamps). Cached in `Data/Transcripts/` on a hash of the audio plus the diarization setting, so it never runs twice for the same audio and a transcript taken without speaker IDs is never reused by the reframing step. Audio past Scribe's request limit is split at ~30-minute marks snapped to silence.
3. **Moments**: `gpt-6-luna` reads the sentence-level transcript and returns sentence ranges with a title and reason.
4. **Virality**: Jev scores each clip with four calibrated yes/no probabilities: `p_viral`, `p_self_contained`, `p_has_payoff`, `p_opens_cold`.
5. **Threshold**: keep clips with `p_viral >= PROB_THRESHOLD` (default 0.6), optionally capped with `--top-n`.
6. **Cut points**: snapped to word boundaries (0.15 s before, 0.35 s after), skipping filler openers, preferring starts after a pause, clamped to 15–90 s. These are computed before step 4 so Jev scores exactly what ends up in the clip.
7. **Captions**: karaoke ASS plus an `.srt` sidecar, from the Scribe word timings. The ASS is **burned into the picture**, written for whatever frame the clip ships in; on a vertical clip the words sit in the lower third, inside the safe zone the platforms draw their own UI over. The `.srt` is a convenience, not the delivery mechanism -- most short-form viewers watch muted and several platforms will not take a sidecar at all.
8. **Reframing**: 9:16 at 1080x1920. Speaker turns come from the speaker IDs on the cached Scribe words, scene cuts from PySceneDetect, and the crop per segment from [Clips AI](https://github.com/ClipsAI/clipsai)'s Resizer (MTCNN faces plus mouth movement), vendored in `utils/clipsai_resize/`. All local: no API calls, keys or tokens. Only clips that survive step 5 are analysed. Crops are recorded in `moments.json` and reused on a re-render.
9. **Render**: one x264 CRF 17 encode per clip does the cut, the crop, the scale to 1080x1920 and the captions together -- a landscape clip is never rendered and then re-encoded. `--no-reframe` keeps the source aspect ratio.

## Setup

```
winget install Gyan.FFmpeg          # or set FFMPEG_BIN / FFPROBE_BIN in .env
pip install -r requirements.txt
copy .env.example .env              # then add the three API keys
```

## Usage

Put videos in `Data/Input_video/`, then:

```
python main.py                             # every video in Data/Input_video
python main.py episode.mp4 --top-n 10      # one video, at most 10 clips
python main.py --threshold 0 --no-render   # score everything, render nothing (for tuning)
python main.py episode.mp4 --no-reframe    # keep the source aspect ratio
```

Output for each video goes to `Data/clips/<video name>/`: `NN_title.mp4` (vertical, captions burned in), `.srt` and `.ass` files, plus `moments.json` with every candidate, its scores, whether it was kept, and for each rendered clip the crop segments it was reframed with (times, speaker, x/y, and whether a fallback crop was used). Re-running reuses those crops instead of redoing the face analysis; delete `moments.json` to force a fresh pass.

**Tuning the threshold**: run a few videos you know well with `--threshold 0 --no-render`. The ranked table shows where your own judgment cuts the list; set `PROB_THRESHOLD` in `.env` to that value.

**Reframing settings** live in `config.py` under step 9 and can be overridden in `.env`: `REFRAME` (on/off), `REFRAME_ASPECT_RATIO`, `REFRAME_WIDTH` / `REFRAME_HEIGHT`, `REFRAME_MIN_SEGMENT_SECONDS`, `REFRAME_SAMPLES_PER_SEGMENT`, `REFRAME_SCENE_THRESHOLD`, `REFRAME_SCENE_MERGE_THRESHOLD` and `REFRAME_DEVICE`. Frame sampling is seeded, so re-running a clip gives the same crops.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

Reframing is the one stage that needs no API keys, so it is tested offline end to end: the
logic tests cover the speaker-segment adapter, gap filling, merging and crop clamping
without loading a model, and the end-to-end test generates a two-speaker sample, feeds in
a mocked Scribe transcript with speaker IDs, and runs real ffmpeg -- checking the clip
comes out 1080x1920, keeps its audio, moves the crop to the speaking side, and has the
captions in the picture rather than only in the `.srt`. Tests needing ffmpeg or the face
models skip with a reason when those are missing.
