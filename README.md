# viral-clips
A repo that extracts viral moments from long videos

## Pipeline

1. **Audio**: ffmpeg pulls a 16 kHz mono WAV and runs `silencedetect` in the same pass.
2. **Transcription**: ElevenLabs Scribe v1 (diarized, word timestamps). Cached in `Data/Transcripts/` on a hash of the audio, so it never runs twice for the same audio. Audio past Scribe's request limit is split at ~30-minute marks snapped to silence.
3. **Moments**: `gpt-6-luna` reads the sentence-level transcript and returns sentence ranges with a title and reason.
4. **Virality**: Jev scores each clip with four calibrated yes/no probabilities: `p_viral`, `p_self_contained`, `p_has_payoff`, `p_opens_cold`.
5. **Threshold**: keep clips with `p_viral >= PROB_THRESHOLD` (default 0.6), optionally capped with `--top-n`.
6. **Cut points**: snapped to word boundaries (0.15 s before, 0.35 s after), skipping filler openers, preferring starts after a pause, clamped to 15–90 s. These are computed before step 4 so Jev scores exactly what ends up in the clip.
7. **Captions**: karaoke ASS (burned in) plus an `.srt` sidecar, from the Scribe word timings.
8. **Render**: original resolution and frame rate, no reframing, x264 CRF 17.

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
```

Output for each video goes to `Data/clips/<video name>/`: `NN_title.mp4`, `.srt` and `.ass` files, plus `moments.json` with every candidate, its scores and whether it was kept.

**Tuning the threshold**: run a few videos you know well with `--threshold 0 --no-render`. The ranked table shows where your own judgment cuts the list; set `PROB_THRESHOLD` in `.env` to that value.
