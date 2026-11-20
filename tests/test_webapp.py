"""The web page's upload -> process -> download flow, with the pipeline stages stubbed out:
no API calls, no ffmpeg."""
import json
import time

import pytest

pytest.importorskip("flask", reason="the web UI needs flask")

import config  # noqa: E402
import pipeline  # noqa: E402
import webapp  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path / "Input_video")
    monkeypatch.setattr(config, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(webapp, "_job", None)
    monkeypatch.setattr(webapp.audio, "require_ffmpeg", lambda: None)
    calls = []

    def load_media(video, allow_transcribe=False, progress=None):
        progress(0.5, "Transcribing part 1 of 2")
        calls.append(("load", video, allow_transcribe))
        return video

    def render_clips(video, threshold, top_n, reframe_clips, progress=None):
        calls.append(("render", threshold, top_n, reframe_clips))
        folder = pipeline.out_dir(video)
        folder.mkdir(parents=True)
        (folder / "01_big-moment.mp4").write_bytes(b"fake mp4")
        pipeline.report_path(video).write_text(json.dumps({"moments": [
            {"clip": "01_big-moment.mp4", "title": "Big moment", "reason": "Because", "p_viral": 0.8,
             "duration": 31.2, "start_hms": "00:01:02"},
            {"clip": None, "title": "Dropped"},
        ]}), encoding="utf-8")
        return [folder / "01_big-moment.mp4"]

    monkeypatch.setattr(pipeline, "load_media", load_media)
    monkeypatch.setattr(pipeline, "find", lambda media: calls.append(("find",)))
    monkeypatch.setattr(pipeline, "score", lambda media: calls.append(("score",)))
    monkeypatch.setattr(pipeline, "render_clips", render_clips)
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


def test_upload_runs_every_stage_and_serves_the_clips(client):
    res = client.post("/api/upload?filename=My%20talk.mp4&threshold=0.4&top_n=3&reframe=0", data=b"video bytes")
    assert res.status_code == 200, res.get_json()
    job = _wait(client)

    assert job["state"] == "done", job["error"]
    assert job["percent"] == 100
    assert [s["state"] for s in job["stages"]] == ["done"] * 4
    assert (config.INPUT_DIR / "My talk.mp4").read_bytes() == b"video bytes"
    assert [c[0] for c in client.calls] == ["load", "find", "score", "render"]
    assert client.calls[0][2] is True  # transcribes on a cache miss
    assert client.calls[-1] == ("render", 0.4, 3, False)
    assert [c["title"] for c in job["clips"]] == ["Big moment"]

    assert client.get("/clips/01_big-moment.mp4").data == b"fake mp4"
    download = client.get("/clips/01_big-moment.mp4?download=1")
    assert "attachment" in download.headers["Content-Disposition"]


def test_same_video_is_reused_and_a_different_one_kept_alongside(client):
    client.post("/api/upload?filename=talk.mp4", data=b"one")
    _wait(client)
    client.post("/api/upload?filename=talk.mp4", data=b"one")
    _wait(client)
    client.post("/api/upload?filename=talk.mp4", data=b"two")
    assert _wait(client)["video"] == "talk (2).mp4"
    assert sorted(p.name for p in config.INPUT_DIR.iterdir()) == ["talk (2).mp4", "talk.mp4"]


@pytest.mark.parametrize("query, message", [
    ("filename=notes.txt", "Not a video"),
    ("filename=a.mp4&threshold=2", "between 0 and 1"),
    ("filename=a.mp4&top_n=0", "at least 1"),
    ("filename=a.mp4&top_n=many", "whole number"),
])
def test_bad_uploads_are_refused(client, query, message):
    res = client.post(f"/api/upload?{query}", data=b"x")
    assert res.status_code == 400
    assert message in res.get_json()["error"]


def test_a_second_upload_waits_for_the_first(client, monkeypatch):
    monkeypatch.setattr(webapp, "_job", webapp.Job(config.INPUT_DIR / "busy.mp4", 0.6, None, True, state="running"))
    res = client.post("/api/upload?filename=other.mp4", data=b"x")
    assert res.status_code == 409


def test_a_failing_stage_is_reported(client, monkeypatch):
    def boom(media):
        raise RuntimeError("OpenAI said no")
    monkeypatch.setattr(pipeline, "find", boom)
    client.post("/api/upload?filename=talk.mp4", data=b"x")
    job = _wait(client)
    assert job["state"] == "error"
    assert job["error"] == "OpenAI said no"
    assert [s["state"] for s in job["stages"]] == ["done", "error", "pending", "pending"]


def test_path_tricks_stay_inside_the_input_folder():
    assert webapp._safe_name("..\\..\\Windows\\evil.mp4") == "evil.mp4"
    assert webapp._safe_name('a:b*c?.mp4') == "a_b_c_.mp4"
