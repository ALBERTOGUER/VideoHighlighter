"""What a video file is, as far as its metadata says.

One ffprobe call through upstream's ``modules.media.ffmpeg_tools.probe`` (stdlib
only), reduced to the fields the rest of DroneReel needs: frame rate as the
exact fraction (exports must keep it), bit depth (a log profile without 10
bits has little latitude to recover), the colour tags, and the container and
stream tags where a camera may name itself.

Nothing here decides anything about colour. Tags are *evidence* for
:mod:`dronereel.profiles`, which is where "is this D-Log M?" gets answered,
and which refuses to answer from evidence that has not been verified.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from modules.media.ffmpeg_tools import probe
from modules.media.video_probe import _fps, _rotation_from_stream

# DJI writes a subtitle file beside the video on some cameras; on others it is
# a setting. Its per-frame lines can carry the colour mode, e.g. "color_md :
# dlog_m" — seen on other DJI models, not yet confirmed on the Avata 2 or Osmo
# Nano, which is why profiles.py treats it as a hint until verified.
_SRT_COLOR_MD = re.compile(r"color_md\s*[:=]\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
_SRT_READ_LIMIT = 256 * 1024


@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    fps_fraction: str           # e.g. "60000/1001", kept exact for export
    video_codec: str
    pix_fmt: str
    bit_depth: int
    color_primaries: str
    color_transfer: str
    color_space: str
    color_range: str
    rotation: int
    has_audio: bool
    audio_codec: Optional[str]
    tags: dict = field(default_factory=dict)   # format tags, then video-stream tags
    srt_path: Optional[str] = None
    srt_color_modes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def bit_depth_of(stream: dict) -> int:
    """Bits per sample: ffprobe's own figure when it gives one, else the pixel
    format's name (``yuv420p10le`` -> 10). Plain formats are 8 bit."""
    raw = str(stream.get("bits_per_raw_sample") or "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    m = re.search(r"p(\d{2})(le|be)?$", str(stream.get("pix_fmt") or ""))
    return int(m.group(1)) if m else 8


def find_sidecar_srt(video_path: str) -> Optional[str]:
    """The ``.SRT`` next to a video with the same stem, any case."""
    folder = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0].lower()
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    for name in sorted(names):
        base, ext = os.path.splitext(name)
        if ext.lower() == ".srt" and base.lower() == stem:
            return os.path.join(folder, name)
    return None


def srt_color_modes(srt_path: str) -> list:
    """Distinct ``color_md`` values in the head of a DJI subtitle file."""
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(_SRT_READ_LIMIT)
    seen = []
    for value in _SRT_COLOR_MD.findall(head):
        value = value.lower()
        if value not in seen:
            seen.append(value)
    return seen


def probe_media(path: str) -> MediaInfo:
    """Read ``path``'s metadata. Opens the file for reading only."""
    data = probe(path)
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}

    tags = {str(k).lower(): v for k, v in (fmt.get("tags") or {}).items()}
    tags.update({str(k).lower(): v for k, v in (video.get("tags") or {}).items()})

    try:
        duration = float(fmt.get("duration") or video.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0

    srt = find_sidecar_srt(path)
    return MediaInfo(
        path=os.path.abspath(path),
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_fps(video),
        fps_fraction=str(video.get("r_frame_rate") or video.get("avg_frame_rate") or ""),
        video_codec=str(video.get("codec_name") or ""),
        pix_fmt=str(video.get("pix_fmt") or ""),
        bit_depth=bit_depth_of(video),
        color_primaries=str(video.get("color_primaries") or "unknown"),
        color_transfer=str(video.get("color_transfer") or "unknown"),
        color_space=str(video.get("color_space") or "unknown"),
        color_range=str(video.get("color_range") or "unknown"),
        rotation=_rotation_from_stream(video),
        has_audio=audio is not None,
        audio_codec=str(audio.get("codec_name")) if audio else None,
        tags=tags,
        srt_path=srt,
        srt_color_modes=srt_color_modes(srt) if srt else [],
    )
