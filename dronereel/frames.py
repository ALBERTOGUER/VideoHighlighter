"""Small BGR frames at a fixed rate, decoded by ffmpeg.

Why not ``cv2.VideoCapture``
============================
Measured on the Raspberry Pi 5 with a 10 s 4K60 10-bit HEVC file, sampling at
5 fps down to 384 px wide:

* OpenCV / ffmpeg software decode: ~34-37 s — every 4K frame is decoded and
  converted before anything is dropped.
* ffmpeg with the Pi's HEVC decoder (``-hwaccel drm``, frames kept as
  ``drm_prime`` until after the ``fps`` filter): **~8 s**. Downloading every
  4K frame first (``-hwaccel drm`` alone) costs ~34 s again, which is why the
  filter order matters. Its frames matched the software ones to a mean of
  0.55/255 (max 4).
* ``-skip_frame bidir`` (~21 s) breaks HEVC references and was rejected.

So: hardware first for HEVC with no rotation, software otherwise or when the
hardware path fails before its first frame. Which one ran is reported, since
it changes analysis time by ~4x.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from dronereel.media import MediaInfo

HW_ARGS = ["-hwaccel", "drm", "-hwaccel_output_format", "drm_prime"]


class DecodeError(RuntimeError):
    pass


@dataclass
class FrameSource:
    media: MediaInfo
    fps: float
    width: int
    height: int
    decoder: Optional[str] = None       # "hardware" | "software", once running
    start: float = 0.0                  # seconds of the source
    duration: Optional[float] = None    # None = to the end

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3


def work_size(media: MediaInfo, width: int) -> tuple:
    """Even ``(w, h)`` no wider than the source, in display orientation."""
    src_w, src_h = media.width, media.height
    if media.rotation in (90, 270):
        src_w, src_h = src_h, src_w
    w = min(int(width), src_w) // 2 * 2
    h = max(2, int(round(w * src_h / max(1, src_w) / 2)) * 2)
    return w, h


def _command(src: FrameSource, hardware: bool) -> list:
    scale = f"scale={src.width}:{src.height}:flags=area"
    if hardware:
        vf = f"fps={src.fps},hwdownload,format=nv12,{scale}"
        pre = HW_ARGS
    else:
        vf = f"fps={src.fps},{scale}"
        pre = []
    span = []
    if src.start > 0:
        span += ["-ss", f"{src.start:.6f}"]
    if src.duration is not None:
        span += ["-t", f"{src.duration:.6f}"]
    return ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *pre, *span,
            "-i", src.media.path, "-map", "0:v:0", "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


def _run(src: FrameSource, hardware: bool) -> Iterator[np.ndarray]:
    proc = subprocess.Popen(_command(src, hardware), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    count = 0
    try:
        while True:
            buf = proc.stdout.read(src.frame_bytes)
            if len(buf) < src.frame_bytes:
                break
            count += 1
            yield np.frombuffer(buf, np.uint8).reshape(src.height, src.width, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        proc.stderr.close()
        code = proc.wait()
    if code != 0:
        raise DecodeError(f"ffmpeg ({'hardware' if hardware else 'software'}) "
                          f"failed after {count} frames: {err[-400:]}")


def iter_frames(media: MediaInfo, fps: float, width: int,
                prefer_hardware: bool = True, start: float = 0.0,
                duration: Optional[float] = None) -> tuple:
    """``(source, frames)``: frame ``k`` of ``frames`` is at ``start + k / fps``
    s; ``duration`` limits the stretch read (None = to the end)."""
    w, h = work_size(media, width)
    src = FrameSource(media, float(fps), w, h, start=max(0.0, float(start)),
                      duration=duration)

    def frames() -> Iterator[np.ndarray]:
        if prefer_hardware and media.video_codec == "hevc" and media.rotation == 0:
            produced = 0
            try:
                src.decoder = "hardware"
                for frame in _run(src, hardware=True):
                    produced += 1
                    yield frame
                return
            except DecodeError:
                if produced:
                    raise
        src.decoder = "software"
        yield from _run(src, hardware=False)

    return src, frames()
