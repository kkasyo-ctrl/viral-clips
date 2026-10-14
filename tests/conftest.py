import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (needs the path above)


def _tool(binary: str) -> str | None:
    return binary if shutil.which(binary) or Path(binary).is_file() else None


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    """The ffmpeg binary, or skip. Honours FFMPEG_BIN the same way the pipeline does."""
    for binary in (config.FFMPEG, config.FFPROBE):
        if _tool(binary) is None:
            pytest.skip(f"{binary} not found; install ffmpeg or set FFMPEG_BIN / FFPROBE_BIN")
    return config.FFMPEG


@pytest.fixture(scope="session")
def reframe_models() -> None:
    """Skip unless the face-analysis dependencies are installed."""
    pytest.importorskip("torch", reason="reframing needs torch")
    pytest.importorskip("scenedetect", reason="reframing needs PySceneDetect")
    pytest.importorskip("facenet_pytorch", reason="reframing needs facenet-pytorch")
    pytest.importorskip("mediapipe", reason="reframing needs mediapipe")
    pytest.importorskip("av", reason="reframing needs PyAV")
