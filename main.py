"""Find the viral moments in long videos and cut them into captioned clips.

    python main.py                              # every video in Data/Input_video
    python main.py episode.mp4                  # one video (a path, or a file name inside Data/Input_video)
    python main.py --threshold 0 --no-render    # tuning: score every moment, render nothing
"""
import argparse
import json
import logging
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import config
from utils import audio, captions, cutting, jev_client, moments, render, transcript
from utils.cutting import Cut
from utils.openai_client import Moment
from utils.transcript import format_timestamp

log = logging.getLogger("viral_clips")


@dataclass
class Candidate:
    moment: Moment
    cut: Cut
    text: str
    scores: dict[str, float] | None


def process(video: Path, threshold: float, top_n: int | None, render_clips: bool) -> None:
    log.info("=== %s ===", video.name)

    # 1. Audio extraction (+ silences, reused for chunk splitting and cut points)
    with tempfile.TemporaryDirectory(prefix="viral_clips_") as tmp:
        wav = Path(tmp) / "audio.wav"
        silences = audio.extract_audio(video, wav)
        duration = audio.wav_duration(wav)
        audio_hash = audio.file_sha256(wav)
        log.info("Audio: %s long, %d silences", format_timestamp(duration), len(silences))

        # 2. Transcription (cached on the audio hash)
        words, transcript_file = transcript.load_or_transcribe(video, wav, audio_hash, duration, silences)
    sentences = transcript.group_sentences(words)
    transcript_file.with_suffix(".txt").write_text(
        "\n".join(transcript.format_line(s) for s in sentences) + "\n", encoding="utf-8"
    )
    log.info("Transcript: %d words, %d sentences", len(words), len(sentences))
    if not sentences:
        log.warning("No speech found; skipping")
        return

    # 3. Moment extraction
    found = moments.find_moments(sentences)
    log.info("%s proposed %d moments", config.MOMENTS_MODEL, len(found))
    if not found:
        return

    # 6. Cut points, planned before scoring so Jev judges exactly the words the viewer will hear
    cuts = [cutting.plan_cut(m.start_sentence, m.end_sentence, sentences, words, silences, duration) for m in found]
    texts = [cutting.clip_transcript(c, sentences, words) for c in cuts]

    # 4. Virality probabilities
    scores = jev_client.score_clips(
        [{"clip_transcript": t, "clip_duration_seconds": round(c.duration, 1)} for c, t in zip(cuts, texts)]
    )
    candidates = [Candidate(m, c, t, s) for m, c, t, s in zip(found, cuts, texts, scores)]

    # 5. Threshold
    ranked = sorted((c for c in candidates if c.scores), key=lambda c: c.scores["p_viral"], reverse=True)
    kept = [c for c in ranked if c.scores["p_viral"] >= threshold][:top_n]
    _print_table(ranked, len(kept), threshold)

    out_dir = config.CLIPS_DIR / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {id(c): f"{rank:02d}_{_slug(c.moment.title)}" for rank, c in enumerate(kept, 1)}
    _write_report(out_dir / "moments.json", video, ranked, names, threshold)
    if not render_clips:
        return

    # 7 + 8. Captions and render
    width, height = render.video_size(video)
    for i, c in enumerate(kept, 1):
        name = names[id(c)]
        lines = captions.build_lines(c.cut, sentences, words)
        captions.write_ass(lines, out_dir / f"{name}.ass", width, height)
        captions.write_srt(lines, out_dir / f"{name}.srt")
        log.info("Rendering %d/%d: %s (%.0fs)", i, len(kept), name, c.cut.duration)
        render.render_clip(video, c.cut.start, c.cut.end, out_dir / f"{name}.ass", out_dir / f"{name}.mp4")
    log.info("Clips written to %s", out_dir)


def _print_table(ranked: list[Candidate], n_kept: int, threshold: float) -> None:
    print(f"\n{'#':>3}  {'viral':>5}  {'self':>5}  {'payoff':>6}  {'cold':>5}  {'start':>8}  {'len':>4}  title")
    for rank, c in enumerate(ranked, 1):
        if rank == n_kept + 1:
            print(f"{'':->3}  -- below threshold {threshold:.2f} or past --top-n --")
        s = c.scores
        print(
            f"{rank:>3}  {s['p_viral']:5.2f}  {s['p_self_contained']:5.2f}  {s['p_has_payoff']:6.2f}  "
            f"{s['p_opens_cold']:5.2f}  {format_timestamp(c.cut.start):>8}  {c.cut.duration:3.0f}s  {c.moment.title}"
        )
    print()


def _write_report(path: Path, video: Path, ranked: list[Candidate], names: dict[int, str], threshold: float) -> None:
    report = {
        "source": video.name,
        "threshold": threshold,
        "moments": [
            {
                "clip": f"{names[id(c)]}.mp4" if id(c) in names else None,
                "title": c.moment.title,
                "reason": c.moment.reason,
                **{k: round(v, 4) for k, v in c.scores.items()},
                "start": round(c.cut.start, 3),
                "end": round(c.cut.end, 3),
                "start_hms": format_timestamp(c.cut.start),
                "duration": round(c.cut.duration, 2),
                "sentences": [c.cut.first_sentence, c.cut.last_sentence],
                "transcript": c.text,
            }
            for c in ranked
        ],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _slug(title: str, max_len: int = 48) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")[:max_len].rstrip("-") or "clip"


def _resolve_videos(names: list[str]) -> list[Path]:
    if not names:
        if not config.INPUT_DIR.is_dir():
            return []
        return sorted(p for p in config.INPUT_DIR.iterdir() if p.suffix.lower() in config.VIDEO_EXTENSIONS)
    videos = []
    for name in names:
        path = Path(name) if Path(name).exists() else config.INPUT_DIR / name
        if not path.exists():
            raise SystemExit(f"Video not found: {name}")
        videos.append(path.resolve())
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut the viral moments out of long videos.")
    parser.add_argument("videos", nargs="*", help="video files (default: everything in Data/Input_video)")
    parser.add_argument("--threshold", type=float, default=config.PROB_THRESHOLD,
                        help="minimum p_viral for a moment to become a clip (default: %(default)s)")
    parser.add_argument("--top-n", type=int, help="render at most this many clips per video")
    parser.add_argument("--no-render", action="store_true", help="score and report moments without rendering")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):  # a title the console can't encode must not crash the run
        stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    audio.require_ffmpeg()
    videos = _resolve_videos(args.videos)
    if not videos:
        parser.error(f"no videos found in {config.INPUT_DIR}")

    failed = 0
    for video in videos:
        try:
            process(video, args.threshold, args.top_n, not args.no_render)
        except Exception:
            failed += 1
            log.exception("Failed on %s", video.name)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
