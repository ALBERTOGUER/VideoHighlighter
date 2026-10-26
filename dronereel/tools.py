"""ffmpeg and ffprobe, wherever they are.

On the Pi both come from the system (``apt install ffmpeg``). On a Mac without
Homebrew there is no ffprobe and no ffmpeg on PATH, but the desktop app ships
ffmpeg through imageio-ffmpeg and PyAV reads containers. So: the system tools
when present, otherwise the bundled ffmpeg and PyAV in ffprobe's place, for
the handful of packet-level questions export asks.
"""

from __future__ import annotations

import shutil
import subprocess


def ffmpeg() -> str:
    """The ffmpeg executable: PATH first, then the one shipped with the app."""
    from modules.system.app_paths import ffmpeg_exe
    return ffmpeg_exe()


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def _pyav():
    try:
        import av
    except ImportError as e:  # pragma: no cover - only without both tools
        raise FileNotFoundError(
            "ffprobe is not installed and PyAV is not available to stand in for it") from e
    return av


def format_start(path: str) -> float:
    """The container's start time in seconds (ffprobe ``format=start_time``)."""
    if has_ffprobe():
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time",
                              "-of", "csv=p=0", path], capture_output=True, text=True)
        try:
            return float(out.stdout.strip())
        except ValueError:
            return 0.0
    av = _pyav()
    with av.open(path) as c:
        return (c.start_time or 0) / av.time_base


def video_packets(path: str, around=None) -> list:
    """``[(pts_time or None, flags)]`` of the first video stream in decode
    order, flags as ffprobe prints them (``K`` keyframe, ``D`` discard).
    ``around=(a, b)`` limits the read to that stretch plus 20 s."""
    if has_ffprobe():
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0"]
        if around:
            a, b = around
            cmd += ["-read_intervals", f"{max(0.0, a - 20):.3f}%{b + 20:.3f}"]
        out = subprocess.run([*cmd, path], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-300:])
        packets = []
        for line in out.stdout.split():
            parts = line.split(",")
            if len(parts) < 2:
                continue
            pts = float(parts[0]) if parts[0] not in ("", "N/A") else None
            packets.append((pts, parts[1]))
        return packets
    av = _pyav()
    packets = []
    with av.open(path) as c:
        stream = c.streams.video[0]
        tb = stream.time_base
        stop = None
        if around:
            a, b = around
            c.seek(int(max(0.0, a - 20) / tb), stream=stream, backward=True, any_frame=False)
            stop = b + 20
        for p in c.demux(stream):
            if p.size == 0 and p.pts is None:
                continue  # demuxer flush packet
            pts = float(p.pts * tb) if p.pts is not None else None
            if stop is not None and pts is not None and pts > stop:
                break
            flags = ("K" if p.is_keyframe else "_") + ("D" if p.is_discard else "_")
            packets.append((pts, flags))
    return packets


def video_stream_count(path: str) -> int:
    if has_ffprobe():
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                              "-show_entries", "stream=index", "-of", "csv=p=0", path],
                             capture_output=True, text=True)
        return len(out.stdout.split())
    av = _pyav()
    with av.open(path) as c:
        return len(c.streams.video)
