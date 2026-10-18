"""Find the viral moments in long videos and cut them into captioned clips.

Each stage runs on its own and picks up where the last one left off on disk:

    python main.py transcribe [videos]   # ElevenLabs Scribe -> Data/Transcripts (cached; never runs twice)
    python main.py find [videos]         # GPT picks moments  -> Data/clips/<video>/moments.json
    python main.py score [videos]        # Jev scores them    -> scores added to moments.json
    python main.py render [videos]       # cut, caption, reframe, encode (no API calls)
    python main.py run [videos]          # all four in a row (also what `python main.py` alone does)

Videos are paths or file names inside Data/Input_video; with none given, every video there.

    python main.py render --threshold 0.3 --top-n 5   # render the best five over 0.3
    python main.py render --no-reframe                # keep the source aspect ratio
    python main.py run --no-render                    # everything up to the scores
"""
import argparse
import logging
import sys
from pathlib import Path

import config
import pipeline
from utils import audio

log = logging.getLogger("viral_clips")

COMMANDS = ("transcribe", "find", "score", "render", "run")


def run_stage(command: str, video: Path, args: argparse.Namespace) -> None:
    log.info("=== %s: %s ===", command, video.name)
    media = pipeline.load_media(video, allow_transcribe=command in ("transcribe", "run"))
    if command == "transcribe":
        return
    if command in ("find", "run"):
        pipeline.find(media)
    if command in ("score", "run"):
        candidates = pipeline.score(media)
        if command == "score":
            pipeline.print_table(candidates, [], config.PROB_THRESHOLD)
    if command == "render" or (command == "run" and not args.no_render):
        pipeline.render_clips(media, args.threshold, args.top_n, config.REFRAME and not args.no_reframe)
    elif command == "run":
        pipeline.print_table(pipeline.load_candidates(media), [], config.PROB_THRESHOLD)


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cut the viral moments out of long videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    videos = argparse.ArgumentParser(add_help=False)
    videos.add_argument("videos", nargs="*", help="video files (default: everything in Data/Input_video)")
    rendering = argparse.ArgumentParser(add_help=False)
    rendering.add_argument("--threshold", type=float, default=config.PROB_THRESHOLD,
                           help="minimum p_viral for a moment to become a clip (default: %(default)s)")
    rendering.add_argument("--top-n", type=int, help="render at most this many clips per video")
    rendering.add_argument("--no-reframe", action="store_true",
                           help="render at the source aspect ratio instead of cropping to vertical")

    sub.add_parser("transcribe", parents=[videos], help="transcribe with ElevenLabs (skipped if cached)")
    sub.add_parser("find", parents=[videos], help="ask the LLM for moments; overwrites moments.json")
    sub.add_parser("score", parents=[videos], help="score the moments in moments.json with Jev")
    sub.add_parser("render", parents=[videos, rendering], help="cut and render clips from moments.json (no API calls)")
    run = sub.add_parser("run", parents=[videos, rendering], help="every stage in order")
    run.add_argument("--no-render", action="store_true", help="stop after scoring")
    return parser


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] not in COMMANDS + ("-h", "--help"):
        argv = ["run", *argv]  # `python main.py [videos]` keeps running the whole pipeline
    args = _parser().parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # a title the console can't encode must not crash the run
        stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    audio.require_ffmpeg()
    videos = _resolve_videos(args.videos)
    if not videos:
        raise SystemExit(f"no videos found in {config.INPUT_DIR}")

    failed = 0
    for video in videos:
        try:
            run_stage(args.command, video, args)
        except Exception:
            failed += 1
            log.exception("Failed on %s", video.name)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
