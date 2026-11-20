"""Videos from a link: YouTube, Twitch, Kick and the rest of what yt-dlp supports.

`probe` reads a link's details without downloading anything, so a private, live or
unsupported link fails in a second rather than halfway through a download. `download`
fetches the best video up to DOWNLOAD_MAX_HEIGHT plus the best audio, merged into one
MP4 in Data/Input_video. A video DOWNLOAD_LONG_SECONDS or longer is only fetched in part:
the section the caller picks, at most that long.

The file is named after the video's title and id (plus the section, if any), so the same
link twice reuses the download and with it the cached transcript. Intermediate files go to
a hidden folder, so a download cut short never leaves a half-file that looks like a video.

YouTube needs a JavaScript runtime (Deno) for yt-dlp to get past its download
protections; without one most other sites still work.
"""
import glob
import logging
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import config

log = logging.getLogger(__name__)

TEMP_DIR_NAME = ".downloading"
MAX_STEM = 120  # characters, keeping Windows paths well short of their limit


class LinkError(Exception):
    """A link that cannot be downloaded, with a reason fit to show the user."""


@dataclass(frozen=True)
class LinkInfo:
    url: str
    id: str
    title: str
    duration: float | None  # seconds; None when the site does not say
    site: str
    thumbnail: str | None
    requested: int  # how many streams the download takes: 2 for separate video and audio
    size: float | None = None  # bytes of the whole download, as near as the site says

    @property
    def long(self) -> bool:
        return self.duration is not None and self.duration >= config.DOWNLOAD_LONG_SECONDS

    def to_dict(self) -> dict:
        return {
            "url": self.url, "id": self.id, "title": self.title, "duration": self.duration, "site": self.site,
            "thumbnail": self.thumbnail, "long": self.long, "max_section": config.DOWNLOAD_LONG_SECONDS,
        }


def _options(**extra) -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,  # a watch link inside a playlist means that one video
        "logger": _Logger(),
        # Best video up to the cap plus best audio; H.264 and AAC ahead of other codecs at
        # the same height, since they decode fastest and merge into MP4 cleanly.
        "format": "bv*+ba/b",
        "format_sort": [f"res:{config.DOWNLOAD_MAX_HEIGHT}", "vcodec:h264", "acodec:m4a"],
        "merge_output_format": "mp4",
        **extra,
    }


def probe(url: str) -> LinkInfo:
    """The link's details, or LinkError saying why it cannot be used."""
    import yt_dlp

    if not re.match(r"https?://", url.strip(), re.I):
        raise LinkError("That doesn't look like a link. It should start with https://")
    try:
        with yt_dlp.YoutubeDL(_options()) as ydl:
            info = ydl.extract_info(url.strip(), download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise LinkError(explain(str(exc))) from None
    if info.get("_type") == "playlist":
        raise LinkError("That's a playlist or channel. Paste the link of a single video.")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        raise LinkError("That stream is live or still being processed. Try again once the VOD is up.")
    duration = info.get("duration")
    return LinkInfo(
        url=info.get("webpage_url") or url.strip(),
        id=str(info.get("id") or "video"),
        title=info.get("title") or info.get("id") or "video",
        duration=float(duration) if duration else None,
        site=info.get("extractor_key") or info.get("extractor") or "",
        thumbnail=info.get("thumbnail"),
        requested=len(info.get("requested_formats") or [None]),
        size=sum(
            f.get("filesize") or f.get("filesize_approx") or 0 for f in info.get("requested_formats") or [info]
        ) or None,
    )


def check_section(info: LinkInfo, section: tuple[float, float] | None) -> tuple[float, float] | None:
    """The section to download: required (and limited) for a long video, ignored otherwise."""
    if not info.long:
        return None
    if section is None:
        raise LinkError(
            f"This video is {hms(info.duration)} long. Pick a section of up to "
            f"{hms(config.DOWNLOAD_LONG_SECONDS)} to download."
        )
    start, end = section
    if not 0 <= start < end:
        raise LinkError("The section's end has to come after its start.")
    if end > info.duration + 1:
        raise LinkError(f"The section runs past the end of the video ({hms(info.duration)}).")
    if end - start > config.DOWNLOAD_LONG_SECONDS:
        raise LinkError(f"The section can be at most {hms(config.DOWNLOAD_LONG_SECONDS)} long.")
    return start, min(end, info.duration)


def target(info: LinkInfo, section: tuple[float, float] | None) -> Path:
    """Where the download lands: 'Title [id].mp4', or 'Title [id] 1h20m00s-2h45m00s.mp4'."""
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f%]', "_", info.title).strip(" .")
    suffix = f" [{re.sub(r'[^A-Za-z0-9_-]', '_', info.id)}]"
    if section:
        suffix += " " + "-".join(_compact(t) for t in section)
    return config.INPUT_DIR / f"{title[: MAX_STEM - len(suffix)].rstrip(' .') or 'video'}{suffix}.mp4"


def download(
    info: LinkInfo, section: tuple[float, float] | None, progress: Callable[[float, str], None] | None = None
) -> Path:
    """Download the link (or its section) into INPUT_DIR and return the file; a file already
    there from the same link and section is reused."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    path = target(info, section)
    if path.is_file():
        log.info("Already downloaded: %s", path.name)
        return path
    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp = config.INPUT_DIR / TEMP_DIR_NAME
    report = _Progress(info, section, progress)
    extra = {}
    if section:
        extra = {"download_ranges": download_range_func(None, [section]), "force_keyframes_at_cuts": False}
    options = _options(
        paths={"home": str(config.INPUT_DIR), "temp": str(temp)},
        outtmpl=path.stem.replace("%", "%%") + ".%(ext)s",
        progress_hooks=[report.hook],
        postprocessor_hooks=[report.postprocess],
        **extra,
    )
    log.info("Downloading %s%s", info.url, f" ({hms(section[0])}-{hms(section[1])})" if section else "")
    # A section is fetched by ffmpeg, which tells yt-dlp nothing on the way; watch the
    # files grow instead, against the site's size for the whole video scaled to the section.
    stop = threading.Event()
    if section and progress:
        expected = info.size * (section[1] - section[0]) / info.duration if info.size and info.duration else None
        threading.Thread(target=_watch, args=(temp, path.stem, expected, progress, stop), daemon=True).start()
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([info.url])
    except yt_dlp.utils.DownloadError as exc:
        raise LinkError(explain(str(exc))) from None
    finally:
        stop.set()
        shutil.rmtree(temp, ignore_errors=True)
    if not path.is_file():
        raise LinkError(f"The download finished but {path.name} is not there; see the console for details.")
    log.info("Downloaded %s (%.0f MB)", path.name, path.stat().st_size / 1e6)
    return path


def _watch(temp: Path, stem: str, expected: float | None, progress, stop: threading.Event) -> None:
    """Report a section download's progress from the size of its files, once a second."""
    started, last = time.monotonic(), 0
    while not stop.wait(1.0):
        files = [*temp.glob("*"), *config.INPUT_DIR.glob(f"{glob.escape(stem)}*")] if temp.exists() else []
        size = sum(f.stat().st_size for f in files if f.is_file())
        if size <= last:
            continue
        rate = size / (time.monotonic() - started)
        parts = [_mb(size) + (f" of ~{_mb(expected)}" if expected else ""), f"{rate / 1e6:.1f} MB/s"]
        if expected and size < expected:
            parts.append(f"~{hms((expected - size) / rate)} left")
        progress(min(size / expected, 0.98) if expected else 0.0, f"Downloading section · {' · '.join(parts)}")
        last = size


class _Progress:
    """yt-dlp's progress callbacks as one fraction for the whole download: video and audio
    come down one after the other, so each gets an equal share."""

    def __init__(self, info: LinkInfo, section, progress) -> None:
        self.progress = progress
        self.streams = 1 if section else info.requested  # a section comes down as one stream
        self.files: list[str] = []
        self.last = 0.0

    def hook(self, d: dict) -> None:
        if not self.progress or d.get("status") != "downloading":
            return
        name = d.get("filename") or ""
        started = name not in self.files
        if started:
            self.files.append(name)
        index = min(len(self.files), self.streams) - 1
        done, total = d.get("downloaded_bytes") or 0, d.get("total_bytes") or d.get("total_bytes_estimate")
        if not started and time.monotonic() - self.last < 0.5 and total and done < total:
            return  # the hook fires many times a second; the page polls once a second
        self.last = time.monotonic()
        parts = [_mb(done) + (f" of {_mb(total)}" if total else "")]
        if d.get("speed"):
            parts.append(f"{d['speed'] / 1e6:.1f} MB/s")
        if d.get("eta") is not None and total:
            parts.append(f"{hms(d['eta'])} left")
        what = "Downloading" if self.streams == 1 else ("Downloading video", "Downloading audio")[min(index, 1)]
        fraction = (index + (done / total if total else 0)) / self.streams
        self.progress(min(fraction, 0.99), f"{what} · {' · '.join(parts)}")

    def postprocess(self, d: dict) -> None:
        if self.progress and d.get("status") == "started" and d.get("postprocessor") == "Merger":
            self.progress(0.99, "Merging video and audio")


class _Logger:
    """yt-dlp's messages into this pipeline's log: its warnings and errors matter, its chatter does not."""

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        log.warning("yt-dlp: %s", message)

    def error(self, message: str) -> None:
        log.error("yt-dlp: %s", message)


# yt-dlp error text -> what to tell the user. The first match wins.
_EXPLANATIONS = [
    (r"Unsupported URL", "That site isn't supported. Try a YouTube, Twitch or Kick link, or a direct video link."),
    (r"Private video|video is private", "That video is private."),
    (r"members[- ]only|Join this channel", "That video is for channel members only."),
    (r"Sign in to confirm your age|age[- ]restricted|inappropriate", "That video is age-restricted, so it needs a sign-in."),
    (r"Sign in to confirm you.re not a bot", "YouTube is asking for a sign-in to prove this isn't a bot. Try again later."),
    (r"not available in your country|geo[- ]?restrict", "That video isn't available in your country."),
    (r"video (is )?unavailable|has been removed|does not exist|404", "That video is unavailable or has been removed."),
    (r"JavaScript runtime|js[- ]runtime|n challenge|nsig|deno", "YouTube needs Deno installed to download (see the README), then restart start.bat."),
    (r"Requested format is not available", "No downloadable video format was found for that link."),
    (r"HTTP Error 429|Too Many Requests", "The site is rate-limiting downloads. Wait a few minutes and try again."),
    (r"Unable to download|timed out|getaddrinfo|Connection", "Couldn't reach the site. Check your internet connection."),
]


def explain(error: str) -> str:
    error = re.sub(r"\x1b\[[0-9;]*m", "", error)  # yt-dlp colours its errors for a terminal
    message = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?([\w-]+:\s*)?", "", error.strip())
    for pattern, explanation in _EXPLANATIONS:
        if re.search(pattern, error, re.I):
            return explanation
    return f"Couldn't download that link: {message[:300]}"


def hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _mb(size: float) -> str:
    return f"{size / 1e6:,.1f} MB" if size < 1e8 else f"{size / 1e6:,.0f} MB"


def _compact(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h{s // 60 % 60:02d}m{s % 60:02d}s"
