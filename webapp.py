"""A local web page for the pipeline: upload a video or paste a link to one, watch it go
through every stage, then preview and download the clips. Started by start.bat, or:

    python webapp.py          # http://127.0.0.1:8000 (WEB_PORT in .env to change it)

The video is kept in Data/Input_video, so the same upload or link again reuses it and its
cached transcript. A link is checked first (utils/download.py); one to a video four hours
or longer needs a section picked, and only that section is downloaded. One video is processed at a time, with the same stages as `python main.py run`,
in either clip format: podcast, or gaming (facecam above the game, cam height adjustable).
"""
import hashlib
import json
import logging
import re
import shutil
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.serving import make_server

import config
import pipeline
from utils import audio, download
from utils.download import LinkError, LinkInfo

log = logging.getLogger("viral_clips")

WEB_DIR = config.ROOT / "web"

# (key, label, share of the progress bar). Transcription and rendering are the slow ones;
# a video from a link starts with its download.
DOWNLOAD = ("download", "Download", 0.15)
STAGES = [
    ("transcribe", "Transcribe", 0.30),
    ("find", "Find moments", 0.15),
    ("score", "Score virality", 0.10),
    ("render", "Render clips", 0.45),
]


@dataclass
class Job:
    video: Path
    threshold: float
    top_n: int | None
    reframe: bool
    clip_format: str = "podcast"
    cam_share: float = config.GAMING_CAM_SHARE
    link: LinkInfo | None = None  # set for a video from a link, which is downloaded first
    section: tuple[float, float] | None = None  # the part of a long video to download
    state: str = "uploading"  # uploading -> running -> done | error
    stage: int = -1  # index into stages
    fraction: float = 0.0  # of the current stage
    message: str = ""
    detail: str = ""  # the pipeline's latest log line
    started: float = field(default_factory=time.time)
    finished: float | None = None
    clips: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def stages(self) -> list[tuple[str, str, float]]:
        return [DOWNLOAD, *STAGES] if self.link else STAGES

    def progress(self, fraction: float, message: str) -> None:
        self.fraction, self.message = min(max(fraction, 0.0), 1.0), message

    def enter(self, key: str, message: str) -> None:
        self.stage = [k for k, _, _ in self.stages].index(key)
        self.fraction, self.message = 0.0, message

    def percent(self) -> float:
        if self.state == "done":
            return 100.0
        weights = [weight for _, _, weight in self.stages]
        done = sum(weights[: max(self.stage, 0)])
        current = weights[self.stage] * self.fraction if self.stage >= 0 else 0.0
        return round(100 * (done + current) / sum(weights), 1)

    def to_dict(self) -> dict:
        stages = []
        for i, (key, label, _) in enumerate(self.stages):
            if self.state == "done" or i < self.stage:
                state = "done"
            elif i == self.stage:
                state = "error" if self.state == "error" else "active"
            else:
                state = "pending"
            stages.append({"key": key, "label": label, "state": state})
        return {
            "video": self.video.name,
            "source": "link" if self.link else "upload",
            "link": self.link.to_dict() if self.link else None,
            "settings": {
                "threshold": self.threshold, "top_n": self.top_n, "reframe": self.reframe,
                "format": self.clip_format, "cam_share": self.cam_share,
            },
            "state": self.state,
            "percent": self.percent(),
            "stages": stages,
            "message": self.message,
            "detail": self.detail,
            "elapsed": round((self.finished or time.time()) - self.started),
            "clips": self.clips,
            "error": self.error,
        }


app = Flask(__name__, static_folder=None)
_lock = threading.Lock()
_job: Job | None = None


class _LatestLine(logging.Handler):
    """Hands the pipeline's log lines to the running job, so the page shows what it is doing."""

    def emit(self, record: logging.LogRecord) -> None:
        if _job and _job.state == "running" and record.levelno >= logging.INFO:
            _job.detail = record.getMessage().splitlines()[0][:300]


# --- pages and API --------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "defaults": {
            "threshold": config.PROB_THRESHOLD, "top_n": None, "reframe": config.REFRAME,
            "format": config.CLIP_FORMAT, "cam_share": config.GAMING_CAM_SHARE,
        },
        "cam_share_range": config.GAMING_CAM_SHARE_RANGE,
        "extensions": sorted(config.VIDEO_EXTENSIONS),
        "job": _job.to_dict() if _job else None,
    })


@app.post("/api/upload")
def upload():
    """The video is the raw request body; its name and the settings are in the query string.
    Streaming the body straight to disk keeps a multi-GB upload out of memory and temp files."""
    name = _safe_name(request.args.get("filename", ""))
    if Path(name).suffix.lower() not in config.VIDEO_EXTENSIONS:
        return _error(f"Not a video file; expected one of {', '.join(sorted(config.VIDEO_EXTENSIONS))}")
    settings = _settings(request.args)
    if isinstance(settings, str):
        return _error(settings)
    job = _start(Job(config.INPUT_DIR / name, **settings))
    if job is None:
        return _busy()

    try:
        job.video = _receive(name, request.content_length)
    except Exception as exc:
        job.state, job.error, job.finished = "error", f"Upload failed: {exc}", time.time()
        log.warning("Upload of %s failed: %s", name, exc)
        return _error(job.error)
    job.state = "running"
    threading.Thread(target=_process, args=(job,), daemon=True).start()
    return jsonify({"job": job.to_dict()})


@app.post("/api/link/check")
def check_link():
    """What a link points at, before anything is downloaded: its title, length, and whether
    it is long enough to need a section picked."""
    url = str((request.get_json(silent=True) or {}).get("url", "")).strip()
    try:
        info = _probe(url)
    except LinkError as exc:
        return _error(str(exc))
    warning = None
    if info.site.lower().startswith("youtube") and not shutil.which("deno"):
        warning = "Deno isn't installed, and YouTube downloads usually fail without it. See the README."
    return jsonify({"link": info.to_dict(), "warning": warning})


@app.post("/api/link")
def from_link():
    """Download a checked link (or the picked section of a long one), then process it."""
    body = request.get_json(silent=True) or {}
    settings = _settings(body)
    if isinstance(settings, str):
        return _error(settings)
    try:
        info = _probe(str(body.get("url", "")).strip())
        section = None
        if body.get("start") is not None and body.get("end") is not None:
            section = (float(body["start"]), float(body["end"]))
        section = download.check_section(info, section)
    except (TypeError, ValueError):
        return _error("The section's start and end must be times")
    except LinkError as exc:
        return _error(str(exc))
    job = _start(Job(download.target(info, section), **settings, link=info, section=section))
    if job is None:
        return _busy()
    job.state = "running"
    threading.Thread(target=_process, args=(job,), daemon=True).start()
    return jsonify({"job": job.to_dict()})


@app.get("/clips/<path:filename>")
def clip(filename: str):
    """A clip of the current job; ?download=1 to save it rather than play it."""
    if not _job or _job.state != "done":
        abort(404)
    return send_from_directory(
        pipeline.out_dir(_job.video), filename, as_attachment=request.args.get("download") == "1"
    )


def _error(message: str, code: int = 400):
    return jsonify({"error": message}), code


def _busy():
    return _error(f"Already processing {_job.video.name}; wait for it to finish", 409)


def _start(job: Job) -> Job | None:
    """Make this the current job, unless one is still going."""
    global _job
    with _lock:
        if _job and _job.state in ("uploading", "running"):
            return None
        _job = job
    return job


def _settings(values) -> dict | str:
    """The pipeline settings from a query string or JSON body, or what is wrong with them."""
    try:
        threshold = float(values.get("threshold", config.PROB_THRESHOLD))
        top_n = int(values["top_n"]) if values.get("top_n") not in (None, "") else None
        cam_share = float(values.get("cam_share", config.GAMING_CAM_SHARE))
    except (TypeError, ValueError):
        return "Threshold and cam height must be numbers, and top-N a whole number"
    if not 0.0 <= threshold <= 1.0:
        return "Threshold must be between 0 and 1"
    if top_n is not None and top_n < 1:
        return "Top-N must be at least 1"
    clip_format = values.get("format", config.CLIP_FORMAT)
    if clip_format not in config.FORMATS:
        return f"Format must be one of {', '.join(config.FORMATS)}"
    lo, hi = config.GAMING_CAM_SHARE_RANGE
    if not lo <= cam_share <= hi:
        return f"Cam height must be between {lo:.0%} and {hi:.0%} of the frame"
    reframe_clips = str(values.get("reframe", "1")) in ("1", "true", "True")
    return {
        "threshold": threshold, "top_n": top_n, "reframe": reframe_clips,
        "clip_format": clip_format, "cam_share": cam_share,
    }


# Links already checked, so pressing Start right after Check doesn't ask the site twice.
_probed: dict[str, LinkInfo] = {}


def _probe(url: str) -> LinkInfo:
    if url not in _probed:
        info = download.probe(url)
        _probed[url] = _probed[info.url] = info
    return _probed[url]


# --- upload ---------------------------------------------------------------------------

def _safe_name(filename: str) -> str:
    """The file name alone, without anything Windows refuses in a path."""
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]  # not Path().name: Windows reads 'a:' as a drive
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", base).strip(" .")
    return name or "upload.mp4"


def _receive(name: str, length: int | None) -> Path:
    """Write the request body into Data/Input_video. Re-uploading a video already there
    reuses that file (and with it, the cached transcript); a different video with the same
    name is saved alongside it as 'name (2).mp4'."""
    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    part = config.INPUT_DIR / f".{name}.part"
    digest, received = hashlib.sha256(), 0
    try:
        with part.open("wb") as f:
            while block := request.stream.read(1 << 20):
                f.write(block)
                digest.update(block)
                received += len(block)
        if not received or (length is not None and received != length):
            raise RuntimeError(f"received {received} of {length} bytes")
        return _store(part, name, digest.hexdigest())
    finally:
        part.unlink(missing_ok=True)


def _store(part: Path, name: str, digest: str) -> Path:
    target, n = config.INPUT_DIR / name, 2
    while target.exists():
        if target.stat().st_size == part.stat().st_size and _sha256(target) == digest:
            log.info("%s is already in %s; using that copy", name, config.INPUT_DIR)
            return target
        target, n = target.with_name(f"{Path(name).stem} ({n}){Path(name).suffix}"), n + 1
    part.replace(target)
    log.info("Saved upload to %s", target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


# --- processing -----------------------------------------------------------------------

def _process(job: Job) -> None:
    """The same stages as `python main.py run`, reporting into the job as they go."""
    try:
        audio.require_ffmpeg()
        if job.link:
            job.enter("download", "Starting the download")
            job.video = download.download(job.link, job.section, progress=job.progress)
        job.enter("transcribe", "Extracting audio")
        media = pipeline.load_media(job.video, allow_transcribe=True, progress=job.progress)
        job.enter("find", "Finding moments")
        pipeline.find(media, job.clip_format)
        job.enter("score", "Scoring virality")
        pipeline.score(media, job.clip_format)
        job.enter("render", "Rendering clips")
        written = pipeline.render_clips(
            media, job.threshold, job.top_n, job.reframe, progress=job.progress,
            clip_format=job.clip_format, cam_share=job.cam_share,
        )
        job.clips = _clip_info(job.video, written)
        job.message = f"{len(written)} clip{'s' * (len(written) != 1)} ready" if written else "No moments cleared the threshold"
        job.state = "done"
    except (Exception, SystemExit) as exc:
        log.exception("Failed on %s", job.video.name)
        job.state, job.error = "error", str(exc) or type(exc).__name__
    finally:
        job.finished = time.time()


def _clip_info(video: Path, written: list[Path]) -> list[dict]:
    """What the page shows for each rendered clip, from moments.json, best first."""
    if not written:
        return []
    report = json.loads(pipeline.report_path(video).read_text(encoding="utf-8"))
    by_file = {m["clip"]: m for m in report["moments"] if m.get("clip")}
    clips = []
    for path in written:
        m = by_file.get(path.name, {})
        clips.append({
            "file": path.name,
            "title": m.get("title", path.stem),
            "reason": m.get("reason", ""),
            "p_viral": m.get("p_viral"),
            "duration": m.get("duration"),
            "start_hms": m.get("start_hms"),
            "size": path.stat().st_size,
        })
    return clips


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # a title the console can't encode must not crash the run
        stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # not a line per status poll
    logging.getLogger().addHandler(_LatestLine())

    url = f"http://{config.WEB_HOST}:{config.WEB_PORT}"
    try:
        server = make_server(config.WEB_HOST, config.WEB_PORT, app, threaded=True)
    except OSError:
        raise SystemExit(
            f"Port {config.WEB_PORT} is already in use. If Viral Clips is already running, open {url}; "
            "otherwise set WEB_PORT in .env to a free port."
        )
    print(f"\nViral Clips is running at {url}\nClose this window (or press Ctrl+C) to stop it.\n", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
