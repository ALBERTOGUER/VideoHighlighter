"""Synthetic videos made with ffmpeg: no real footage in tests, ever.

Kept apart from upstream's ``tests/`` on purpose: that suite's conftest
replaces cv2, torch and friends with mocks, and DroneReel's tests must run
against the real thing. Run with ``pytest dronereel/tests``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not installed")


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def videos(tmp_path_factory):
    """``{"h264_audio": path, "hevc10": path}`` in a read-only-by-convention
    source folder, plus ``"dir"``."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")
    src = tmp_path_factory.mktemp("originals")
    h264 = str(src / "DJI_0001.MP4")
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            # setparams, not -color_trc: the output options alone left the
            # transfer and primaries untagged on this ffmpeg (7.1).
            "-vf", "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-c:a", "aac", "-shortest", h264)
    hevc = str(src / "DJI_0002.MP4")
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30000/1001",
            "-t", "2", "-c:v", "libx265", "-preset", "ultrafast",
            "-x265-params", "log-level=error:frame-threads=1",
            "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", hevc)
    return {"dir": str(src), "h264_audio": h264, "hevc10": hevc}
