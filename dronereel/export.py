"""Write ONE continuous segment of a video to its own file. Never a montage.

Two ways, both measured frame by frame on a clip with its frame number drawn
in the pixels (see tests/test_export.py):

``copy``
    The original packets, untouched: no quality lost, seconds of work even at
    4K. Stream copy can only start on a keyframe, so the segment grows
    outwards to the keyframe at or before ``start`` and a keyframe after
    ``end``. The end is cut by ffmpeg's segment muxer, which splits exactly
    on a keyframe: a plain ``-t`` dropped one B-frame and kept the next one
    in testing (frames ...151, 153). With open GOPs the pictures shown just
    before that keyframe are stored after it and fall out of the copy, so
    the cut keyframe is chosen to leave the request covered anyway
    (:func:`keyframes`).
``reencode``
    Exactly the selected frames (input ``-ss`` half a frame early, then
    ``-frames:v``), re-encoded — by default H.265 10-bit (x265 with the
    memory-limited settings from the camera's YAML). Required for anything
    that changes the picture: colour (F5), 9:16 (F8). Slow on the Pi:
    ~1 fps at 4K.

``auto`` is ``copy`` unless the picture must change. Either way: one video
stream, the source's frame rate (never resampled; ``-fps_mode passthrough``),
the original audio copied when there is one, the colour tags of the source,
and no music, transitions or titles. The source is fingerprinted before and
checked after; the output goes through :func:`dronereel.safety.output_path`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from dronereel.media import MediaInfo
from dronereel.safety import assert_unchanged, fingerprint, output_path

EXPORT_VERSION = 1

ENCODERS = {"hevc": "libx265", "h264": "libx264"}
PIX_FMT = {8: "yuv420p", 10: "yuv420p10le"}
LEADING_SCAN = 32          # packets after a keyframe searched for leading pictures
COLOR_TAGS = (("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc"),
              ("color_space", "-colorspace"), ("color_range", "-color_range"))


class ExportError(RuntimeError):
    pass


@dataclass
class ExportConfig:
    mode: str = "auto"                      # auto | copy | reencode
    codec: str = "hevc"                     # hevc | h264 (reencode)
    bit_depth: int = 10
    crf: float = 18
    preset: str = "medium"
    x265_params: str = "frame-threads=1:rc-lookahead=10"
    audio: str = "copy"                     # copy | aac | none
    container: str = "mp4"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ExportConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Opciones de exportación desconocidas: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in ("auto", "copy", "reencode"):
            raise ValueError(f"Modo de exportación desconocido: {self.mode}")
        if self.codec not in ENCODERS:
            raise ValueError(f"Códec no soportado: {self.codec} (hevc o h264)")
        if int(self.bit_depth) not in PIX_FMT:
            raise ValueError(f"Profundidad no soportada: {self.bit_depth} (8 o 10)")
        if self.audio not in ("copy", "aac", "none"):
            raise ValueError(f"Audio desconocido: {self.audio} (copy, aac o none)")
        if self.container not in ("mp4", "mov"):
            raise ValueError(f"Contenedor no soportado: {self.container} (mp4 o mov)")


@dataclass
class ExportResult:
    path: str
    mode: str                               # what actually ran: copy | reencode
    requested: tuple                        # (start, end) asked for, seconds
    start: float                            # what the file holds, seconds of the source
    end: float
    first_frame: int                        # source frame indices, end exclusive
    end_frame: int
    frames: int                             # frames in the output (ffprobe count)
    fps: str
    seconds: float                          # wall time
    command: list
    warnings: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 4)
        return d


def _fps_fraction(media: MediaInfo) -> tuple:
    """``(num, den)`` of the source frame rate, exact when ffprobe gave one."""
    try:
        num, den = (int(x) for x in media.fps_fraction.split("/"))
        if num > 0 and den > 0:
            return num, den
    except (ValueError, AttributeError):
        pass
    if media.fps <= 0:
        raise ExportError("No se conoce la frecuencia de fotogramas del original")
    return int(round(media.fps * 1001)), 1001


def _format_start(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def keyframes(path: str, around: Optional[tuple] = None) -> list:
    """``[(time, leading)]`` for the keyframes of the first video stream:
    time in seconds from the file start, and how many pictures come after
    the keyframe in decode order but are shown before it (open GOP).

    Cutting a stream copy at keyframe ``k`` leaves those ``leading`` pictures
    out, so the copy really ends ``leading`` frames before ``k``: measured
    on x265 at 60 fps, 3 frames (cut at frame 240, last frame shown 236).

    Reads packet headers only (no decoding); ``around=(a, b)`` limits the
    read to that stretch plus a margin, which matters for long 4K files on
    an SD card."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0"]
    if around:
        a, b = around
        cmd += ["-read_intervals", f"{max(0.0, a - 20):.3f}%{b + 20:.3f}"]
    out = subprocess.run([*cmd, path], capture_output=True, text=True)
    if out.returncode != 0:
        raise ExportError(f"ffprobe no pudo leer los fotogramas clave: {out.stderr[-300:]}")
    t0 = _format_start(path)
    packets = []                                        # decode order
    for line in out.stdout.split():
        parts = line.split(",")
        if len(parts) >= 2 and parts[0] not in ("", "N/A"):
            packets.append((float(parts[0]) - t0, "K" in parts[1]))
    found = {}
    for i, (t, key) in enumerate(packets):
        if key:
            lead = sum(1 for u, _ in packets[i + 1:i + 1 + LEADING_SCAN] if u < t - 1e-6)
            found[round(t, 6)] = lead
    return sorted(found.items())


def count_frames(path: str) -> int:
    """Frames a decoder will show: video packets not flagged discardable.

    A stream copy of open-GOP HEVC (x265's default, likely DJI's too) keeps
    the leading pictures that reference the previous GOP; they are flagged
    ``D`` and dropped by decoders. Measured: 663 packets, 660 frames shown."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "packet=flags", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return -1
    return sum(1 for flags in out.stdout.split() if "D" not in flags)


def _video_streams(path: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                          "-show_entries", "stream=index", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    return len(out.stdout.split())


def _audio_args(media: MediaInfo, cfg: ExportConfig) -> list:
    if not media.has_audio or cfg.audio == "none":
        return ["-an"]
    if cfg.audio == "aac":
        return ["-map", "0:a:0", "-c:a", "aac", "-b:a", "320k"]
    return ["-map", "0:a:0", "-c:a", "copy"]


def _color_args(media: MediaInfo, tags: Optional[dict] = None) -> list:
    """The source's colour tags, or ``tags`` when the colour was converted."""
    args = []
    for attr, flag in COLOR_TAGS:
        value = (tags or {}).get(attr) if tags else getattr(media, attr, "unknown")
        if value and value not in ("unknown", "reserved"):
            args += [flag, value]
    return args


def _encoder_args(media: MediaInfo, cfg: ExportConfig, tags: Optional[dict] = None) -> list:
    enc = ENCODERS[cfg.codec]
    args = ["-c:v", enc, "-pix_fmt", PIX_FMT[int(cfg.bit_depth)],
            "-crf", f"{cfg.crf:g}", "-preset", cfg.preset]
    if enc == "libx265":
        params = "log-level=error"
        if cfg.x265_params:
            params += ":" + cfg.x265_params
        args += ["-x265-params", params, "-tag:v", "hvc1"]
    return args + _color_args(media, tags)


def _run(cmd: list, total: float, progress) -> None:
    """Run ffmpeg, reporting ``out_time`` as a fraction of ``total``."""
    proc = subprocess.Popen([*cmd[:1], "-progress", "pipe:1", "-nostats", *cmd[1:]],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if progress and line.startswith("out_time_us="):
            try:
                progress(min(1.0, int(line.split("=")[1]) / 1e6 / max(total, 1e-6)))
            except ValueError:
                pass
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise ExportError(f"ffmpeg falló: {err.strip()[-600:]}")


def plan_copy(media: MediaInfo, start: float, end: float, fps: float) -> tuple:
    """``(a, cut, b)`` for a stream copy covering ``[start, end)``: start at
    keyframe ``a`` (at or before ``start``), cut at keyframe ``cut``, and the
    copy then really ends at ``b`` = ``cut`` minus its leading pictures (or
    at the end of the file)."""
    kfs = keyframes(media.path, around=(start, end))
    if not kfs:
        raise ExportError("No se encontraron fotogramas clave")
    eps = 0.5 / fps
    a = max([k for k, _ in kfs if k <= start + eps], default=kfs[0][0])
    for k, lead in kfs:
        b = k - lead / fps
        if k > a and b >= end - eps:
            return a, k, b
    return a, None, media.duration


def export_segment(media: MediaInfo, start: float, end: float, output_dir: str,
                   cfg: Optional[ExportConfig] = None, needs_reencode: bool = False,
                   name: Optional[str] = None, progress=None,
                   video_filter: Optional[str] = None,
                   color_tags: Optional[dict] = None) -> ExportResult:
    """Write ``[start, end)`` of ``media`` (seconds) as one file in
    ``output_dir``. ``needs_reencode`` says the picture will change (colour,
    reframing), which rules out ``copy``; a ``video_filter`` implies it.
    ``color_tags`` replace the source's tags when the filter converts colour."""
    cfg = cfg or ExportConfig()
    cfg.validate()
    needs_reencode = needs_reencode or bool(video_filter)
    if not (0 <= start < end):
        raise ValueError(f"Segmento no válido: {start}–{end}")
    if media.duration and start >= media.duration:
        raise ValueError(f"El inicio ({start:.2f} s) está fuera del vídeo ({media.duration:.2f} s)")
    end = min(end, media.duration) if media.duration else end
    mode = cfg.mode
    if mode == "auto":
        mode = "reencode" if needs_reencode else "copy"
    if mode == "copy" and needs_reencode:
        raise ExportError("La copia directa no puede cambiar la imagen (color o encuadre): "
                          "usa el modo reencode.")

    num, den = _fps_fraction(media)
    fps = num / den
    original = fingerprint(media.path)
    stem = os.path.splitext(os.path.basename(media.path))[0]
    warnings = []

    if mode == "copy":
        a, cut, b = plan_copy(media, start, end, fps)
        first, last = int(round(a * fps)), int(round(b * fps))
        if a < start - 0.5 / fps or b > end + 0.5 / fps:
            warnings.append(
                f"Copia directa: el segmento se amplía a fotogramas clave "
                f"({start:.2f}–{end:.2f} s → {a:.2f}–{b:.2f} s). "
                f"Para el corte exacto usa --mode reencode (más lento, recomprime).")
    else:
        first = int(round(start * fps))
        last = max(first + 1, int(round(end * fps)))
        a, b = first / fps, last / fps

    filename = name or f"{stem}_{a:07.2f}-{b:07.2f}.{cfg.container}"
    final = output_path(output_dir, filename, [media.path])
    work = tempfile.mkdtemp(prefix=".dronereel-", dir=os.path.dirname(final))
    t0 = time.time()
    try:
        if mode == "copy":
            # The segment muxer splits exactly on keyframe ``cut``; reading a
            # margin past it lets the muxer see that keyframe.
            pattern = os.path.join(work, f"part%03d.{cfg.container}")
            span = (cut if cut is not None else media.duration) - a
            cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
                   "-ss", f"{a:.6f}", "-i", media.path, "-map", "0:v:0",
                   *_audio_args(media, cfg),
                   "-t", f"{span + 2.0:.6f}", "-c:v", "copy", "-map_metadata", "0",
                   "-f", "segment", "-segment_times", f"{span:.6f}",
                   "-segment_format", cfg.container, "-reset_timestamps", "1", pattern]
            _run(cmd, span + 2.0, progress)
            produced = os.path.join(work, f"part000.{cfg.container}")
        else:
            produced = os.path.join(work, f"out.{cfg.container}")
            n = last - first
            seek = max(0.0, (first - 0.5) / fps)
            cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
                   "-ss", f"{seek:.6f}", "-i", media.path, "-map", "0:v:0",
                   *_audio_args(media, cfg), "-frames:v", str(n),
                   "-t", f"{(n + 0.5) / fps:.6f}", "-fps_mode", "passthrough",
                   *(["-vf", video_filter] if video_filter else []),
                   *_encoder_args(media, cfg, color_tags), "-map_metadata", "0",
                   "-movflags", "+faststart", produced]
            _run(cmd, b - a, progress)
        if not os.path.exists(produced):
            raise ExportError("ffmpeg no produjo el archivo de salida")
        os.replace(produced, final)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    elapsed = time.time() - t0

    assert_unchanged(original)
    frames = count_frames(final)
    if _video_streams(final) != 1:
        raise ExportError(f"La salida no tiene exactamente una pista de vídeo: {final}")
    expected = last - first
    if frames >= 0 and abs(frames - expected) > 1:
        warnings.append(f"La salida tiene {frames} fotogramas; se esperaban {expected}.")
    return ExportResult(path=final, mode=mode, requested=(round(start, 4), round(end, 4)),
                        start=a, end=b, first_frame=first, end_frame=last, frames=frames,
                        fps=f"{num}/{den}", seconds=elapsed, command=cmd, warnings=warnings)


def write_record(result: ExportResult, media: MediaInfo, extra: dict) -> str:
    """``<video>.json`` beside the export: what was cut from what, and how —
    the raw material for ratings and training (F6)."""
    original = fingerprint(media.path)
    record = {"version": EXPORT_VERSION, "export": result.to_dict(),
              "source": {"path": media.path, "sha256": original.sha256,
                         "size": original.size, "duration": media.duration,
                         "fps": media.fps_fraction, "codec": media.video_codec,
                         "bit_depth": media.bit_depth},
              **extra}
    path = os.path.splitext(result.path)[0] + ".json"
    with open(path, "x", encoding="utf-8") as f:       # never overwrite
        json.dump(record, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


# Measured on the Pi 5 (8 GB): x265 10-bit, preset medium, the YAML's
# x265_params, 4K60 source -> 0.92 frames/s, peak 2.5 GB.
MEASURED_4K_FPS = 0.92
MEASURED_4K_PEAK_MB = 2500


def _mem_available_mb() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def reencode_notes(media: MediaInfo, start: float, end: float) -> list:
    """Time estimate and memory warning before a re-encode, from the Pi
    measurement scaled by pixel count (a rough guide, not a promise)."""
    pixels = max(1, media.width * media.height)
    scale = pixels / (3840 * 2160)
    frames = max(1, int(round((end - start) * media.fps)))
    minutes = frames / (MEASURED_4K_FPS / max(scale, 0.05)) / 60
    notes = [f"Recodificar {frames} fotogramas: unos {minutes:.0f} min estimados "
             f"(medido en la Pi 5: {MEASURED_4K_FPS} fps en 4K)."]
    need = int(MEASURED_4K_PEAK_MB * max(scale, 0.25))
    avail = _mem_available_mb()
    if avail is not None and avail < need + 300:
        notes.append(f"AVISO: hay {avail} MB de memoria disponible y la codificación necesita "
                     f"unos {need} MB. Cierra Firefox u otros programas, o el sistema puede "
                     f"detener ffmpeg por falta de memoria.")
    return notes


def print_progress(fraction: float) -> None:
    print(f"\r  exportando… {fraction * 100:5.1f} %", end="", file=sys.stderr, flush=True)

