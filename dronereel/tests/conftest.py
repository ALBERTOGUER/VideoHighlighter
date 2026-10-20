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


# Frame-numbered clips: frame k shows k in binary as 12 white/black blocks,
# so an export can be checked frame by frame — which source frames it holds,
# in which order, with nothing skipped, repeated or spliced in.
NUM_W, NUM_H, NUM_BITS = 320, 240, 12


def _numbered_frame(k: int):
    import numpy as np
    img = np.zeros((NUM_H, NUM_W), np.uint8)
    for b in range(NUM_BITS):
        r, c = divmod(b, 4)
        if k >> b & 1:
            img[r * 80 + 10:r * 80 + 70, c * 80 + 10:c * 80 + 70] = 255
    return img


def make_numbered(path: str, frames: int, rate: str, video_args: list, audio: bool) -> str:
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "gray", "-s", f"{NUM_W}x{NUM_H}", "-r", rate, "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-shortest", "-c:a", "aac"]
    proc = subprocess.Popen([*cmd, *video_args, path], stdin=subprocess.PIPE)
    for k in range(frames):
        proc.stdin.write(_numbered_frame(k).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def read_frame_numbers(path: str) -> list:
    """The frame number shown in every decoded frame of ``path``, in order."""
    import numpy as np
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
                          "-fps_mode", "passthrough", "-vf", f"scale={NUM_W}:{NUM_H}",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(out, np.uint8).reshape(-1, NUM_H, NUM_W)
    numbers = []
    for img in frames:
        k = 0
        for b in range(NUM_BITS):
            r, c = divmod(b, 4)
            if img[r * 80 + 40, c * 80 + 40] > 128:
                k |= 1 << b
        numbers.append(k)
    return numbers


@pytest.fixture(scope="session")
def numbered(tmp_path_factory):
    """``{"h264": path, "hevc": path, "hevc_open": path, "dir": folder}``.

    h264: 300 frames at 30000/1001 with B-frames, a keyframe every 30 frames,
    AAC audio, BT.709 tags. hevc: 125 frames at 25 fps, 10-bit, no audio.
    hevc_open: 400 frames at 60 fps, 10-bit, open GOP, keyframe every 60."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")
    src = tmp_path_factory.mktemp("numbered_originals")
    h264 = make_numbered(str(src / "DJI_0010.MP4"), 300, "30000/1001",
                         ["-c:v", "libx264", "-g", "30", "-bf", "2", "-crf", "16",
                          "-pix_fmt", "yuv420p", "-vf",
                          "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"],
                         audio=True)
    hevc = make_numbered(str(src / "DJI_0011.MP4"), 125, "25",
                         ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "16",
                          "-x265-params", "log-level=error:frame-threads=1:keyint=25",
                          "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"], audio=False)
    # Open GOP with B-frames across each keyframe: pictures shown before a
    # keyframe are stored after it, which a stream copy must account for.
    hevc_open = make_numbered(str(src / "DJI_0012.MP4"), 400, "60",
                              ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "16",
                               "-x265-params", "log-level=error:frame-threads=1:keyint=60:"
                               "open-gop=1:bframes=3:b-adapt=0",
                               "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"], audio=False)
    return {"dir": str(src), "h264": h264, "hevc": hevc, "hevc_open": hevc_open}
