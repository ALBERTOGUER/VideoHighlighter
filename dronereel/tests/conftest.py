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


@pytest.fixture(scope="session")
def motion_clips(tmp_path_factory):
    """Clips of known camera movement over a still texture, 640x360 @ 30 fps.

    Expected values (see test_analyze): pan = camera right at 0.2 frame
    widths/s; roll = 0.35 rad/s (~20°/s); zoom = scale 1 + 0.25 t (log-rate
    0.25 falling to ~0.19); blur, blown highlights, one hard cut at 2.0 s, and
    a featureless frame where movement cannot be measured.
    """
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")
    d = tmp_path_factory.mktemp("motion")
    tex = str(d / "tex.png")
    _ffmpeg("-f", "lavfi", "-i",
            "mandelbrot=size=2400x1600:start_scale=0.4:end_scale=0.4:maxiter=200,format=rgb24",
            "-frames:v", "1", tex)
    still = ["-loop", "1", "-r", "30", "-i", tex, "-t", "2.5"]
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p"]
    specs = {
        "static": "crop=640:360:600:500",
        "pan": "crop=640:360:x='600+t*128':y=500",
        "roll": "crop=1000:1000:400:300,rotate=a='t*0.35':ow=640:oh=360",
        "zoom": ("crop=1280:720:400:400,zoompan=z='1+0.25*on/30':d=1:"
                 "x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':s=640x360:fps=30"),
        "blur": "crop=640:360:600:500,gblur=sigma=4",
        "blown": "crop=640:360:600:500,eq=brightness=0.6:contrast=2",
    }
    out = {}
    for name, vf in specs.items():
        out[name] = str(d / f"{name}.mp4")
        _ffmpeg(*still, "-vf", vf, *enc, out[name])
    out["cut"] = str(d / "cut.mp4")
    _ffmpeg("-loop", "1", "-r", "30", "-i", tex, "-loop", "1", "-r", "30", "-i", tex,
            "-filter_complex",
            "[0]crop=640:360:600:500,trim=duration=2[a];"
            "[1]crop=640:360:1500:200,trim=duration=2[b];[a][b]concat=n=2:v=1[v]",
            "-map", "[v]", *enc, out["cut"])
    out["flat"] = str(d / "flat.mp4")
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x3060c0:size=640x360:rate=30",
            "-t", "2.5", *enc, out["flat"])
    return out
