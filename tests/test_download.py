"""Videos from a link: checking the link, the section rule for long videos, file naming,
error messages, and the web flow. yt-dlp is stubbed out, so nothing reaches the network."""
import time

import pytest

yt_dlp = pytest.importorskip("yt_dlp", reason="links need yt-dlp")

import config  # noqa: E402
import pipeline  # noqa: E402
import webapp  # noqa: E402
from utils import download  # noqa: E402
from utils.download import LinkError, LinkInfo  # noqa: E402

HOUR = 3600


def _info(duration=1200.0, **kw) -> LinkInfo:
    return LinkInfo(url="https://www.youtube.com/watch?v=abc123", id="abc123", title="Insane clutch: 1v5",
                    duration=duration, site="Youtube", thumbnail=None, requested=2, **kw)


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: returns `result` from extract_info, or raises it."""
    result: dict | Exception = {}

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def fake_ydl(monkeypatch):
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return FakeYDL


# --- checking a link ----------------------------------------------------------------

def test_probe_reads_the_details(fake_ydl):
    fake_ydl.result = {
        "id": "abc123", "title": "Insane clutch", "duration": 5 * HOUR, "extractor_key": "Youtube",
        "webpage_url": "https://www.youtube.com/watch?v=abc123", "thumbnail": "https://i.ytimg.com/x.jpg",
        "requested_formats": [{"filesize": 900}, {"filesize_approx": 100}],
    }
    info = download.probe("https://youtu.be/abc123")
    assert (info.id, info.title, info.duration, info.requested, info.size) == ("abc123", "Insane clutch", 5 * HOUR, 2, 1000)
    assert info.long and info.to_dict()["long"]


@pytest.mark.parametrize("result, message", [
    ({"_type": "playlist", "entries": []}, "playlist"),
    ({"id": "x", "title": "LIVE", "is_live": True}, "live"),
    ({"id": "x", "title": "Soon", "live_status": "is_upcoming"}, "live"),
])
def test_probe_refuses_what_cannot_be_clipped(fake_ydl, result, message):
    fake_ydl.result = result
    with pytest.raises(LinkError, match=message):
        download.probe("https://www.youtube.com/watch?v=x")


def test_probe_explains_a_download_error(fake_ydl):
    fake_ydl.result = yt_dlp.utils.DownloadError("ERROR: [youtube] x: Private video. Sign in if you've been granted access")
    with pytest.raises(LinkError, match="private"):
        download.probe("https://www.youtube.com/watch?v=x")


def test_probe_wants_a_link():
    with pytest.raises(LinkError, match="https://"):
        download.probe("my video please")


@pytest.mark.parametrize("error, expected", [
    ("ERROR: Unsupported URL: https://example.com/page", "isn't supported"),
    ("ERROR: [youtube] x: Sign in to confirm your age.", "age-restricted"),
    ("ERROR: [youtube] x: Join this channel to get access to members-only content", "members"),
    ("ERROR: [youtube] x: Video unavailable", "unavailable"),
    ("\x1b[0;31mERROR:\x1b[0m [youtube] x: This video is unavailable", "unavailable or has been removed"),
    ("\x1b[0;31mERROR:\x1b[0m [generic] odd failure", "Couldn't download that link: odd failure"),
    ("ERROR: [youtube] x: The uploader has not made this video available in your country", "country"),
    ("ERROR: unable to download video data: HTTP Error 429: Too Many Requests", "rate-limiting"),
    ("ERROR: something nobody has seen before", "something nobody has seen before"),
])
def test_errors_are_explained(error, expected):
    assert expected in download.explain(error)


# --- long videos and sections -------------------------------------------------------

def test_a_short_video_downloads_whole():
    assert download.check_section(_info(3 * HOUR), (60, 120)) is None


def test_a_long_video_needs_a_section():
    with pytest.raises(LinkError, match="Pick a section"):
        download.check_section(_info(6 * HOUR), None)


def test_the_four_hour_mark_counts_as_long():
    assert _info(config.DOWNLOAD_LONG_SECONDS).long
    assert not _info(config.DOWNLOAD_LONG_SECONDS - 1).long


@pytest.mark.parametrize("section, message", [
    ((2 * HOUR, HOUR), "after its start"),
    ((5 * HOUR, 7 * HOUR), "past the end"),
    ((0, 4 * HOUR + 60), "at most"),
])
def test_bad_sections_are_refused(section, message):
    with pytest.raises(LinkError, match=message):
        download.check_section(_info(6 * HOUR), section)


def test_a_good_section_is_kept():
    assert download.check_section(_info(6 * HOUR), (HOUR + 1200, 2 * HOUR + 2700)) == (HOUR + 1200, 2 * HOUR + 2700)


# --- naming -------------------------------------------------------------------------

def test_files_are_named_after_the_video_and_section(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
    info = LinkInfo("u", "abc123", 'GTA 6: "insane" 100%?', 600, "Youtube", None, 2)
    assert download.target(info, None).name == "GTA 6_ _insane_ 100__ [abc123].mp4"
    assert download.target(info, (4800, 9900)).name.endswith("[abc123] 1h20m00s-2h45m00s.mp4")
    long_title = LinkInfo("u", "abc123", "x" * 400, 600, "Youtube", None, 2)
    assert len(download.target(long_title, None).stem) <= download.MAX_STEM


def test_a_video_already_downloaded_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path)
    existing = download.target(_info(), None)
    existing.write_bytes(b"video")

    def explode(options):
        raise AssertionError("downloaded again")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", explode)
    assert download.download(_info(), None) == existing


def test_download_progress_covers_video_then_audio():
    seen = []
    report = download._Progress(_info(), None, lambda f, m: seen.append((round(f, 2), m)))
    report.hook({"status": "downloading", "filename": "v.f137.mp4", "downloaded_bytes": 50e6, "total_bytes": 100e6})
    report.hook({"status": "downloading", "filename": "a.f140.m4a", "downloaded_bytes": 5e6, "total_bytes": 10e6})
    assert [f for f, _ in seen] == [0.25, 0.75]
    assert seen[0][1].startswith("Downloading video") and seen[1][1].startswith("Downloading audio")


# --- the web flow -------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path / "Input_video")
    monkeypatch.setattr(config, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(webapp, "_job", None)
    monkeypatch.setattr(webapp, "_probed", {})
    monkeypatch.setattr(webapp.audio, "require_ffmpeg", lambda: None)
    calls = []
    infos = {"https://short": _info(1200), "https://long": _info(6 * HOUR)}

    def probe(url):
        calls.append(("probe", url))
        if url not in infos:
            raise LinkError("That site isn't supported.")
        return infos[url]

    def fetch(info, section, progress=None):
        progress(0.5, "Downloading video · 50 MB of 100 MB")
        calls.append(("download", section))
        path = download.target(info, section)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return path

    monkeypatch.setattr(download, "probe", probe)
    monkeypatch.setattr(download, "download", fetch)
    monkeypatch.setattr(pipeline, "load_media", lambda video, allow_transcribe=False, progress=None: video)
    monkeypatch.setattr(pipeline, "find", lambda media, clip_format: calls.append(("find", media)))
    monkeypatch.setattr(pipeline, "score", lambda media, clip_format: None)
    monkeypatch.setattr(pipeline, "render_clips", lambda *a, **kw: [])
    client = webapp.app.test_client()
    client.calls = calls
    return client


def _wait(client) -> dict:
    for _ in range(200):
        job = client.get("/api/status").get_json()["job"]
        if job["state"] not in ("uploading", "running"):
            return job
        time.sleep(0.01)
    raise AssertionError("job never finished")


def test_check_shows_what_the_link_is(client):
    res = client.post("/api/link/check", json={"url": "https://long"})
    assert res.status_code == 200
    assert res.get_json()["link"]["long"] is True
    assert client.post("/api/link/check", json={"url": "https://nope"}).get_json()["error"] == "That site isn't supported."


def test_a_link_is_downloaded_then_processed(client):
    client.post("/api/link/check", json={"url": "https://short"})
    res = client.post("/api/link", json={"url": "https://short", "format": "gaming", "threshold": 0.5})
    assert res.status_code == 200, res.get_json()
    job = _wait(client)
    assert job["state"] == "done", job["error"]
    assert job["source"] == "link"
    assert [s["key"] for s in job["stages"]] == ["download", "transcribe", "find", "score", "render"]
    assert ("download", None) in client.calls
    # The check and the start share one look at the site.
    assert [c for c in client.calls if c[0] == "probe"] == [("probe", "https://short")]
    downloaded = next(c[1] for c in client.calls if c[0] == "find")
    assert downloaded.name == "Insane clutch_ 1v5 [abc123].mp4"


def test_a_long_link_needs_its_section(client):
    res = client.post("/api/link", json={"url": "https://long"})
    assert res.status_code == 400 and "Pick a section" in res.get_json()["error"]
    res = client.post("/api/link", json={"url": "https://long", "start": 3600, "end": 7200})
    assert res.status_code == 200
    _wait(client)
    assert ("download", (3600.0, 7200.0)) in client.calls


def test_download_progress_shows_on_the_bar(client):
    job = webapp.Job(config.INPUT_DIR / "x.mp4", 0.6, None, True, link=_info())
    job.enter("download", "Starting the download")
    job.progress(0.5, "Downloading")
    assert job.percent() == pytest.approx(100 * 0.15 * 0.5 / 1.15, abs=0.1)
