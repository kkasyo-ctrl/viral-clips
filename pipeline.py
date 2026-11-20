"""The pipeline's stages, each runnable on its own. They hand off through files on disk:

    transcribe  video                       -> Data/Transcripts/<video>.<key>.json   (ElevenLabs)
    find        transcript                  -> Data/clips/<video>/moments.json       (GPT, no scores)
    score       transcript + moments.json   -> moments.json, with scores             (Jev)
    render      transcript + moments.json   -> Data/clips/<video>/*.mp4              (local only)

Only `transcribe` ever calls ElevenLabs; the later stages read the cached transcript and
fail if it is missing. Cut points are planned afresh from the transcript in every stage
that needs them, so they are always the same for the same moment.

`find`, `score` and `render` take the clip format (config.FORMATS): a podcast is judged on
its words and reframed onto whoever is speaking; a gaming stream is judged on the
streamer's words and how loud they get, and laid out with the facecam above the game.
"""
import json
import logging
import re
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import config
from utils import audio, captions, cutting, gaming, jev_client, moments, reframe, render, transcript
from utils.audio import Peak, Silence
from utils.cutting import Cut
from utils.openai_client import Moment
from utils.transcript import Sentence, Word, format_timestamp

log = logging.getLogger("viral_clips")

# progress(fraction of the stage done, what it is doing now): lets a caller such as the web UI
# follow a long stage. Every stage runs the same without one.
Progress = Callable[[float, str], None] | None


@dataclass
class Media:
    """A video's transcript and audio facts: everything the stages after transcription read."""
    video: Path
    words: list[Word]
    sentences: list[Sentence]
    silences: list[Silence]
    duration: float
    peaks: list[Peak] = field(default_factory=list)  # loudness spikes, read on gaming streams


@dataclass
class Candidate:
    moment: Moment
    cut: Cut
    text: str
    scores: dict[str, float] | None
    crop: dict | None = None  # the crop plan this clip was reframed with, if any
    layout: dict | None = None  # the gaming layout this clip was rendered with, if any


def out_dir(video: Path) -> Path:
    return config.CLIPS_DIR / video.stem


def report_path(video: Path) -> Path:
    return out_dir(video) / "moments.json"


# --- 1 + 2. Audio and transcription ---------------------------------------------------

def load_media(video: Path, allow_transcribe: bool = False, progress: Progress = None) -> Media:
    """Extract the audio (for silences and the cache key) and load the transcript.

    ElevenLabs is only called when allow_transcribe is set and no transcript is cached.
    """
    with tempfile.TemporaryDirectory(prefix="viral_clips_") as tmp:
        wav = Path(tmp) / "audio.wav"
        if progress:
            progress(0.0, "Extracting audio")
        silences = audio.extract_audio(video, wav)
        duration = audio.wav_duration(wav)
        audio_hash = audio.file_sha256(wav)
        peaks = audio.loudness_peaks(wav)
        log.info("Audio: %s long, %d silences, %d loud peaks", format_timestamp(duration), len(silences), len(peaks))
        words, transcript_file = transcript.load_or_transcribe(
            video, wav, audio_hash, duration, silences, allow_transcribe=allow_transcribe, progress=progress
        )
    sentences = transcript.group_sentences(words)
    transcript_file.with_suffix(".txt").write_text(
        "\n".join(transcript.format_line(s) for s in sentences) + "\n", encoding="utf-8"
    )
    log.info("Transcript: %d words, %d sentences", len(words), len(sentences))
    if words and not transcript.has_diarization(words):
        log.warning("Transcript has no speaker IDs; every clip will be reframed as single-speaker")
    return Media(video, words, sentences, silences, duration, peaks)


def transcribe(video: Path) -> Media:
    return load_media(video, allow_transcribe=True)


# --- 3. Moment extraction -------------------------------------------------------------

def find(media: Media, clip_format: str = "podcast") -> list[Candidate]:
    """Ask the LLM for moments and write them, unscored, to moments.json."""
    if not media.sentences:
        log.warning("No speech found; no moments to look for")
        return []
    found = moments.find_moments(media.sentences, clip_format, media.peaks)
    log.info("%s proposed %d moments", config.MOMENTS_MODEL, len(found))
    candidates = _candidates(media, found, [None] * len(found))
    write_report(media.video, candidates, {})
    return candidates


# --- 4. Virality scores ---------------------------------------------------------------

def score(media: Media, clip_format: str = "podcast") -> list[Candidate]:
    """Score the moments in moments.json with Jev and write the scores back."""
    candidates = load_candidates(media)
    if not candidates:
        return []
    states = [{"clip_transcript": c.text, "clip_duration_seconds": round(c.cut.duration, 1)} for c in candidates]
    if clip_format == "gaming":
        for state, c in zip(states, candidates):
            state["loud_moments"] = loud_moments(c.cut, media.peaks)
    scores = jev_client.score_clips(states, clip_format)
    for c, s in zip(candidates, scores):
        c.scores = s
    failed = sum(s is None for s in scores)
    if failed:
        log.warning("Jev could not score %d of %d moments", failed, len(candidates))
    candidates = rank(candidates)
    write_report(media.video, candidates, {})
    return candidates


def loud_moments(cut: Cut, peaks: list[Peak]) -> str:
    """Where in the clip the streamer got loud, in clip time: '0:04 +11 dB, 0:21 +9 dB'."""
    offsets = [(max(p.start - cut.start, 0), p.gain_db) for p in peaks if p.end > cut.start and p.start < cut.end]
    return ", ".join(f"{int(t) // 60}:{int(t) % 60:02d} +{gain:.0f} dB" for t, gain in offsets) or "none"


# --- 5-9. Threshold, captions, reframing, gaming layout, encode -----------------------

def select(candidates: list[Candidate], threshold: float, top_n: int | None) -> list[Candidate]:
    """The moments to render: scored ones at or over the threshold, best first. If nothing
    has been scored yet, every moment is kept, in the order they were found."""
    if not any(c.scores for c in candidates):
        log.warning("No moments are scored; rendering all of them (run `score` to rank them)")
        return candidates[:top_n]
    return [c for c in rank(candidates) if c.scores and c.scores["p_viral"] >= threshold][:top_n]


def render_clips(
    media: Media, threshold: float, top_n: int | None, reframe_clips: bool, progress: Progress = None,
    clip_format: str = "podcast", cam_share: float | None = None,
) -> list[Path]:
    """Cut, caption, reframe (or lay out, for gaming) and encode the selected moments in
    moments.json. No API calls. A gaming clip is always vertical, whatever reframe_clips says."""
    candidates = load_candidates(media)
    kept = select(candidates, threshold, top_n)
    print_table(candidates, kept, threshold)
    if not kept:
        log.info("Nothing to render")
        return []

    folder = out_dir(media.video)
    names = {id(c): f"{rank:02d}_{_slug(c.moment.title)}" for rank, c in enumerate(kept, 1)}
    info = render.probe_video(media.video)
    planner = None
    if clip_format == "gaming":
        planner = gaming.Planner(media.video, info, folder, config.GAMING_CAM_SHARE if cam_share is None else cam_share)
        if progress:
            progress(0.0, "Finding the facecam")
        planner.facecam  # once for the video, before the first clip
    written = []
    for i, c in enumerate(kept, 1):
        name = names[id(c)]
        video_filter = ""
        width, height = info.width, info.height
        seam = None
        if planner:
            if progress:
                progress((i - 1) / len(kept), f"Laying out clip {i} of {len(kept)}")
            layout = planner.plan(c.cut)
            c.layout = layout.to_dict()
            video_filter = gaming.layout_filter(layout)
            width, height, seam = layout.output_width, layout.output_height, layout.seam
        elif reframe_clips:
            if progress:
                progress((i - 1) / len(kept), f"Reframing clip {i} of {len(kept)}")
            plan = reframe.plan(media.video, c.cut, media.words, info, c.crop)
            c.crop = plan.to_dict()
            video_filter = reframe.crop_filter(plan)
            width, height = plan.output_width, plan.output_height
            # Record the crop straight away, so a failure on a later clip doesn't lose it.
            write_report(media.video, candidates, names)
        # Captions are written for the frame the clip is delivered in, and burned into it
        # by the same encode; the .srt alongside keeps the clip's own timings either way.
        lines = captions.build_lines(c.cut, media.sentences, media.words)
        captions.write_ass(lines, folder / f"{name}.ass", width, height, center_y=seam)
        captions.write_srt(lines, folder / f"{name}.srt")
        log.info("Rendering %d/%d: %s (%.0fs, %dx%d)", i, len(kept), name, c.cut.duration, width, height)
        if progress:
            progress((i - 1 + 0.5 * bool(planner or reframe_clips)) / len(kept), f"Encoding clip {i} of {len(kept)}")
        tail = c.cut.end - media.words[c.cut.last_word].end  # fade the audio out after the last word
        render.render_clip(
            media.video, c.cut.start, c.cut.end, folder / f"{name}.ass", folder / f"{name}.mp4", video_filter, tail
        )
        written.append(folder / f"{name}.mp4")
    if planner:
        planner.close()
    write_report(media.video, candidates, names)
    log.info("Clips written to %s", folder)
    return written


# --- moments.json ---------------------------------------------------------------------

def _candidates(media: Media, found: list[Moment], scores: list[dict | None], crops=None) -> list[Candidate]:
    # 6. Cut points, planned before scoring so Jev judges exactly the words the viewer will hear
    out = []
    for i, (m, s) in enumerate(zip(found, scores)):
        cut = cutting.plan_cut(m.start_sentence, m.end_sentence, media.sentences, media.words, media.silences, media.duration)
        crop = (crops or {}).get(_crop_key(cut))
        out.append(Candidate(m, cut, cutting.clip_transcript(cut, media.sentences, media.words), s, crop))
    return out


def load_candidates(media: Media) -> list[Candidate]:
    """The moments in moments.json, with their scores and recorded crops, re-cut against
    the transcript. Edit moments.json by hand (delete a moment, move its sentences) and
    the later stages follow it."""
    path = report_path(media.video)
    if not path.is_file():
        raise RuntimeError(f"No {path}; run `python main.py find` first")
    report = json.loads(path.read_text(encoding="utf-8"))
    found, scores, crops = [], [], {}
    for m in report.get("moments", []):
        first, last = m["sentences"]
        found.append(Moment(start_sentence=first, end_sentence=last, title=m["title"], reason=m.get("reason", "")))
        scores.append({k: v for k, v in m.items() if k.startswith("p_")} or None)
        if m.get("crop"):
            crops[(round(m["start"], 3), round(m["end"], 3))] = m["crop"]
    log.info("Loaded %d moments from %s", len(found), path)
    return _candidates(media, found, scores, crops)


def write_report(video: Path, candidates: list[Candidate], names: dict[int, str]) -> None:
    path = report_path(video)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": video.name,
        "moments": [
            {
                "clip": f"{names[id(c)]}.mp4" if id(c) in names else None,
                "title": c.moment.title,
                "reason": c.moment.reason,
                **{k: round(v, 4) for k, v in (c.scores or {}).items()},
                "start": round(c.cut.start, 3),
                "end": round(c.cut.end, 3),
                "start_hms": format_timestamp(c.cut.start),
                "duration": round(c.cut.duration, 2),
                "sentences": [c.cut.first_sentence, c.cut.last_sentence],
                "transcript": c.text,
                **({"crop": c.crop} if c.crop else {}),
                **({"layout": c.layout} if c.layout else {}),
            }
            for c in candidates
        ],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """Scored moments best first, then the unscored ones in their original order."""
    return sorted(candidates, key=lambda c: -c.scores["p_viral"] if c.scores else 1)


def print_table(candidates: list[Candidate], kept: list[Candidate], threshold: float) -> None:
    kept_ids = {id(c) for c in kept}
    print(f"\n{'':>1}{'#':>3}  {'viral':>5}  {'self':>5}  {'payoff':>6}  {'cold':>5}  {'start':>8}  {'len':>4}  title")
    for n, c in enumerate(rank(candidates), 1):
        s = c.scores
        cols = (
            f"{s['p_viral']:5.2f}  {s['p_self_contained']:5.2f}  {s['p_has_payoff']:6.2f}  {s['p_opens_cold']:5.2f}"
            if s else f"{'-':>5}  {'-':>5}  {'-':>6}  {'-':>5}"
        )
        mark = "*" if id(c) in kept_ids else " "
        print(f"{mark}{n:>3}  {cols}  {format_timestamp(c.cut.start):>8}  {c.cut.duration:3.0f}s  {c.moment.title}")
    print(f"\n* = rendered (threshold {threshold:.2f})\n" if kept else "")


def _crop_key(cut: Cut) -> tuple[float, float]:
    """A clip's cut window, which is what a crop plan was planned against."""
    return round(cut.start, 3), round(cut.end, 3)


def _slug(title: str, max_len: int = 48) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:max_len].rstrip("-") or "clip"
