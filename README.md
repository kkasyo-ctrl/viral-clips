# viral-clips
A repo that extracts viral moments from long videos

## Pipeline

1. **Audio**: ffmpeg pulls a 16 kHz mono WAV and runs `silencedetect` in the same pass.
2. **Transcription**: `microsoft/mai-transcribe-2` through OpenRouter (diarized, word timestamps), billed per hour of audio. Cached in `Data/Transcripts/` on a hash of the audio plus the diarization setting, provider and model, so it never runs twice for the same audio and a transcript taken without speaker IDs -- or from a different model -- is never reused by the reframing step. Audio past the provider's request limit is split at chunk marks snapped to silence: ~10 minutes for OpenRouter, which caps a request at 25 MB, and ~30 for Scribe. Set `TRANSCRIBE_PROVIDER=elevenlabs` in `.env` to go back to Scribe direct.
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
winget install DenoLand.Deno        # only for YouTube links in the web page
pip install -r requirements.txt
copy .env.example .env              # then add the three API keys
```

## Web page

Double-click `start.bat`. It installs Flask and yt-dlp the first time (and updates yt-dlp on every launch), starts a local server and opens http://127.0.0.1:8000 in your browser (set `WEB_PORT` in `.env` to use another port). Close the console window to stop it.

Upload a video, pick the clip format (**Podcast** or **Gaming**), set the threshold and the maximum number of clips, then either the vertical reframing (podcast) or the facecam height (gaming), and press **Upload & process**. The page shows the upload, then a progress bar through transcription, moment finding, scoring and rendering (per transcription chunk and per clip). When it finishes, each clip has a preview player and a download button. The upload is saved to `Data/Input_video/`. Uploading the same video again reuses that file and its cached transcript. A different video with the same name is saved as `name (2).mp4`. One video is processed at a time, with the same stages as `python main.py run`.

**Link** instead of **Upload** takes a video link: YouTube, Twitch, Kick, X, TikTok, Vimeo, a direct video link, or anything else [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports. **Check link** shows the title and length before anything is downloaded, and gives a plain reason when a link can't be used (private, members-only, age-restricted, live, unsupported). The video is downloaded at up to 1080p into `Data/Input_video/` as `Title [id].mp4`, with its own progress stage; pasting the same link again reuses the file and its cached transcript. A video **4 hours or longer** asks for a From/To section of at most 4 hours (`DOWNLOAD_LONG_SECONDS` in `.env`), and only that part is downloaded, saved as `Title [id] 1h20m00s-2h45m00s.mp4`. Section downloads can be much slower than whole videos, because YouTube throttles them, so pick only the part you need. YouTube also needs [Deno](https://deno.com) installed (`winget install DenoLand.Deno`, then restart `start.bat`); without it, many YouTube videos fail, and the page warns about it when you check a YouTube link. Clip only what you have the rights to use.

## Usage

Put videos in `Data/Input_video/`, then:

Each stage runs on its own and reads what the previous one saved, so you only pay for the API calls you ask for:

```
python main.py transcribe [videos]   # OpenRouter -> Data/Transcripts/ (cached: never runs twice)
python main.py find [videos]         # GPT picks moments -> moments.json (overwrites it, unscored)
python main.py score [videos]        # Jev scores the moments in moments.json
python main.py render [videos]       # cut, caption, reframe, encode -- no API calls
python main.py run [videos]          # all four in order; `python main.py` alone does the same
```

With no videos named, each command works on everything in `Data/Input_video/`. `find`, `score` and `render` need the transcript and refuse to call the transcription API; `score` and `render` need `moments.json`. You can edit `moments.json` by hand between stages (delete moments, change a `sentences` range) and the later stages follow it. `render` with nothing scored renders every moment.

```
python main.py render episode.mp4 --top-n 10   # one video, at most 10 clips
python main.py render --threshold 0.3          # override PROB_THRESHOLD
python main.py render --no-reframe             # keep the source aspect ratio
python main.py run --no-render                 # stop after scoring
python main.py run --format gaming             # a stream: facecam on top, game below
python main.py render --format gaming --cam-share 0.4   # give the cam 40% of the height
```

Output for each video goes to `Data/clips/<video name>/`: `NN_title.mp4` (vertical, captions burned in), `.srt` and `.ass` files, plus `moments.json` with every candidate, its scores, whether it was kept, and for each rendered clip the crop segments it was reframed with (times, speaker, x/y, and whether a fallback crop was used). Re-rendering reuses those crops instead of redoing the face analysis; `find` replaces `moments.json`, so it starts fresh.

**Tuning the threshold**: run `score` on a few videos you know well. The ranked table shows where your own judgment cuts the list; set `PROB_THRESHOLD` in `.env` to that value.

**Gaming format** (`--format gaming`, or Gaming on the web page) is for streams with a facecam over the gameplay:

- **Moments**: each transcript line where the streamer's voice spikes well over its usual level is tagged `[LOUD +11dB]` for the moment finder, which gets a prompt written for streams. Jev is told the clip is from a stream and where in it the streamer got loud. The loudness is read locally from the audio, so it costs nothing extra.
- **Facecam**: found automatically, once per video. About 40 frames are sampled across the stream; the face that holds one spot through them is the streamer. Game characters move about, so they don't count. The overlay around the face is the area that stays still between samples while the gameplay changes. If no clean edge is found (a green-screen cut-out, for example), a box around the face is used. The result is saved in `Data/clips/<video>/facecam.json`; delete that file to detect again.
- **Layout**: 1080x1920 with the cam cropped around the face on top (`GAMING_CAM_SHARE` of the height, default 1/3, range 25-50%), the middle of the game below it, and the captions centred on the seam. The game window slides a little off centre so it doesn't show the cam overlay a second time.
- **No cam in a clip**: each clip is checked for the streamer's face in the cam, since streamers switch scenes. A clip without it, or a stream with no facecam at all, gets the whole game frame centred over a blurred copy of itself.

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
