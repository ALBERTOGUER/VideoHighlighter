"""Vertical 9:16 from a horizontal clip: one crop window that follows what
stands out in the picture, smoothly (F8).

Why not upstream's ``modules/crop``
===================================
Upstream's cropper follows *people*: a person detector plus poses, several
crop slots per frame, frames written through OpenCV. FPV footage is mostly
landscape with no person in it, the detector model is not installed here, and
writing through OpenCV would lose the 10-bit picture and the original audio.
So this module only decides *where the window is*; the clip is written by the
same ffmpeg path as every other export (``export.export_segment``), with the
crop as the first filter — before the colour LUT, which then has ~3x fewer
pixels to convert.

What "follows" means, honestly
==============================
There is no subject detection. Each sampled frame gets a **spectral residual
saliency** map (Hou & Zhang, 2007: what stands out from the image's own
statistics — a lone tree, a building, a bright boat; content-neutral, nothing
learned). Its column profile says how much of that saliency each horizontal
position of the 9:16 window would hold. Then:

1. **Viterbi path** over 64 window positions: the best-scoring path with a
   cost for every move, so the window does not hop between two equally
   interesting places, and a small pull towards the centre.
2. **Gaussian smoothing** (``smoothness`` seconds) — a calm virtual camera.
3. **Speed limit** (``max_speed`` frame-widths per second), forwards and
   backwards, so the window never whips: the drone already moves, and a
   fast virtual pan on top of a real one is what makes vertical FPV tiring.

``fixed`` picks the single best position for the whole clip (no virtual
movement at all); ``center`` is the plain centre crop.

The window keeps the source's full height when it can, at an exact 9:16 with
even sides (2160 → 1206x2144); it moves only horizontally. Output keeps that
native size unless ``height`` asks for a smaller one (e.g. 1920 → 1080x1920);
it is never upscaled.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from dronereel.frames import iter_frames, work_size
from dronereel.media import MediaInfo

VERTICAL_VERSION = 1
FOLLOW_MODES = ("smooth", "fixed", "center")
PLAN_WIDTH = 320            # decode width for planning
SAL_WIDTH = 128             # saliency map width
POSITIONS = 64              # window positions the path chooses among
KNOT_SECONDS = 0.25         # path resolution in the ffmpeg crop expression


class VerticalError(RuntimeError):
    pass


@dataclass
class VerticalConfig:
    follow: str = "smooth"          # smooth | fixed | center
    smoothness: float = 1.5         # s, Gaussian sigma of the path
    stickiness: float = 1.5         # reluctance to move (s of advantage per frame-width)
    max_speed: float = 0.12         # frame-widths per second
    center_bias: float = 0.15       # pull towards the centre, 0..1
    height: int = 0                 # output height; 0 = native crop
    sample_fps: float = 10.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "VerticalConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Opciones de vertical desconocidas: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.follow not in FOLLOW_MODES:
            raise ValueError(f"Seguimiento desconocido: {self.follow} (smooth, fixed o center)")
        checks = (("smoothness", 0, 10), ("stickiness", 0, 20), ("max_speed", 0.01, 2),
                  ("center_bias", 0, 1), ("sample_fps", 1, 30))
        for name, lo, hi in checks:
            v = float(getattr(self, name))
            if not lo <= v <= hi:
                raise ValueError(f"vertical.{name} = {v:g} fuera de rango ({lo}–{hi})")
        h = int(self.height)
        if h and (h < 64 or h % 32):
            raise ValueError(f"vertical.height = {h}: 0 (nativo) o múltiplo de 32 (p. ej. 1920)")


def crop_size(width: int, height: int) -> tuple:
    """Largest exact 9:16 window with even sides inside ``width x height``."""
    m = int(min(height / 16, width / 9)) // 2 * 2
    if m < 2:
        raise VerticalError(f"Imagen demasiado pequeña para 9:16: {width}x{height}")
    return 9 * m, 16 * m


def output_size(crop_w: int, crop_h: int, height: int) -> tuple:
    if not height or height >= crop_h:
        return crop_w, crop_h
    return height * 9 // 16, height


def display_size(media: MediaInfo) -> tuple:
    w, h = media.width, media.height
    return (h, w) if media.rotation in (90, 270) else (w, h)


@dataclass
class CropPath:
    """Where the window's left edge is (source pixels), over the clip."""
    follow: str
    src_w: int
    src_h: int
    crop_w: int
    crop_h: int
    out_w: int
    out_h: int
    y: int
    times: list                     # s from the clip start
    x: list                         # left edge, source px

    def at(self, t) -> np.ndarray:
        return np.interp(t, self.times, self.x)

    def expression(self, fps: float, duration: float) -> str:
        """Left edge as an ffmpeg expression of the frame number ``n``
        (0 = the clip's first frame): piecewise linear through knots every
        ``KNOT_SECONDS``, written as a flat sum of clipped ramps (no nesting,
        so no expression depth limit)."""
        n_knots = max(2, int(math.ceil(duration / KNOT_SECONDS)) + 1)
        kt = np.linspace(0.0, max(duration, 1e-3), n_knots)
        kx = self.at(kt)
        if np.ptp(kx) < 0.5:
            return f"{kx[0]:.1f}"
        kf = kt * fps
        terms = [f"{kx[0]:.2f}"]
        for i in range(len(kt) - 1):
            span = kf[i + 1] - kf[i]
            slope = (kx[i + 1] - kx[i]) / span
            if abs(slope * span) < 0.01:
                continue
            terms.append(f"{slope:+.5f}*clip(n-{kf[i]:.3f},0,{span:.3f})")
        return "".join(terms)

    def video_filter(self, fps: float, duration: float) -> str:
        x = self.expression(fps, duration)
        f = f"crop=w={self.crop_w}:h={self.crop_h}:x='{x}':y={self.y}"
        if (self.out_w, self.out_h) != (self.crop_w, self.crop_h):
            f += f",scale={self.out_w}:{self.out_h}:flags=lanczos"
        return f

    def stats(self) -> dict:
        x = np.asarray(self.x, float)
        t = np.asarray(self.times, float)
        speed = np.abs(np.diff(x)) / np.maximum(np.diff(t), 1e-6) / self.src_w if len(x) > 1 else [0]
        return {"travel": round(float(np.ptp(x)) / self.src_w, 4) if len(x) else 0.0,
                "max_speed": round(float(np.max(speed)), 4) if len(x) > 1 else 0.0}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["times"] = [round(float(v), 3) for v in self.times]
        d["x"] = [round(float(v), 1) for v in self.x]
        d["version"] = VERTICAL_VERSION
        d.update(self.stats())
        return d


def saliency(bgr: np.ndarray) -> np.ndarray:
    """Spectral residual saliency (Hou & Zhang 2007) on each Lab channel,
    at ``SAL_WIDTH`` px wide; returns a non-negative map summing to 1."""
    h, w = bgr.shape[:2]
    sh = max(8, int(round(SAL_WIDTH * h / w)))
    small = cv2.resize(bgr, (SAL_WIDTH, sh), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    total = np.zeros((sh, SAL_WIDTH), np.float32)
    for c in range(3):
        f = np.fft.fft2(lab[..., c])
        log_amp = np.log(np.abs(f) + 1e-6).astype(np.float32)
        residual = log_amp - cv2.blur(log_amp, (3, 3))
        s = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(f)))) ** 2
        total += cv2.GaussianBlur(s.astype(np.float32), (0, 0), SAL_WIDTH * 0.02)
    # The FFT wraps around: the outermost pixels light up on any image.
    b = 2
    total[:b, :] = total[-b:, :] = 0
    total[:, :b] = total[:, -b:] = 0
    s = float(total.sum())
    if not s > 0:
        return np.full_like(total, 1.0 / total.size)
    return total / s


def window_gain(columns: np.ndarray, frac: float, positions: np.ndarray) -> np.ndarray:
    """Saliency inside a window ``frac`` of the width wide starting at each of
    ``positions`` (fractions of the width), relative to an even spread (1)."""
    w = len(columns)
    cum = np.concatenate([[0.0], np.cumsum(columns)])
    grid = np.arange(w + 1, dtype=np.float64)
    a = np.interp(positions * w, grid, cum)
    b = np.interp((positions + frac) * w, grid, cum)
    return (b - a) / max(frac, 1e-6)


def viterbi(score: np.ndarray, positions: np.ndarray, move_cost: float) -> np.ndarray:
    """Index of the best position per sample: maximise the summed ``score``
    minus ``move_cost`` per frame-width moved."""
    n, p = score.shape
    trans = move_cost * np.abs(positions[:, None] - positions[None, :])   # [from, to]
    total = score[0].copy()
    back = np.zeros((n, p), np.int32)
    for t in range(1, n):
        cand = total[:, None] - trans
        back[t] = np.argmax(cand, axis=0)
        total = cand[back[t], np.arange(p)] + score[t]
    path = np.zeros(n, np.int32)
    path[-1] = int(np.argmax(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    # Odd reflection keeps a steady ramp straight up to the ends; "edge"
    # padding flattened it and left the window 150 px behind at both ends.
    padded = np.pad(x, r, mode="reflect", reflect_type="odd")
    return np.convolve(padded, k, mode="valid")


def limit_speed(x: np.ndarray, step: float) -> np.ndarray:
    """No sample more than ``step`` from its neighbours (forward then back)."""
    y = x.copy()
    for i in range(1, len(y)):
        y[i] = min(max(y[i], y[i - 1] - step), y[i - 1] + step)
    for i in range(len(y) - 2, -1, -1):
        y[i] = min(max(y[i], y[i + 1] - step), y[i + 1] + step)
    return y


def follow_path(columns: np.ndarray, frac: float, cfg: VerticalConfig) -> np.ndarray:
    """Left edge per sample, as a fraction of the width, from per-sample
    saliency ``columns`` (n x SAL_WIDTH)."""
    n = len(columns)
    room = 1.0 - frac
    if room <= 1e-6 or n == 0:
        return np.zeros(max(n, 1))
    positions = np.linspace(0.0, room, POSITIONS)
    off_centre = np.abs(positions - room / 2) / (room / 2)
    # Whole window plus its inner half: anything narrower than the window
    # scores the same at every position that contains it, so without the
    # inner half the tie went to the centre and the object sat at the edge.
    score = np.stack([0.5 * window_gain(c, frac, positions)
                      + 0.5 * window_gain(c, frac / 2, positions + frac / 4)
                      for c in columns])
    score -= cfg.center_bias * off_centre
    if cfg.follow == "center":
        return np.full(n, room / 2)
    if cfg.follow == "fixed":
        return np.full(n, positions[int(np.argmax(score.mean(axis=0)))])
    idx = viterbi(score, positions, cfg.stickiness * cfg.sample_fps)
    x = gaussian_smooth(positions[idx], cfg.smoothness * cfg.sample_fps)
    x = limit_speed(x, cfg.max_speed / cfg.sample_fps)
    return np.clip(x, 0.0, room)


def plan(media: MediaInfo, start: float, end: float, cfg: Optional[VerticalConfig] = None,
         prefer_hardware: bool = True) -> CropPath:
    """The crop path for ``[start, end)`` of ``media``. Reads only."""
    cfg = cfg or VerticalConfig()
    cfg.validate()
    src_w, src_h = display_size(media)
    crop_w, crop_h = crop_size(src_w, src_h)
    out_w, out_h = output_size(crop_w, crop_h, int(cfg.height))
    y = (src_h - crop_h) // 2 // 2 * 2
    frac = crop_w / src_w
    duration = max(1e-3, end - start)
    base = dict(src_w=src_w, src_h=src_h, crop_w=crop_w, crop_h=crop_h,
                out_w=out_w, out_h=out_h, y=y)
    if cfg.follow == "center" or crop_w >= src_w:
        x = (src_w - crop_w) / 2
        return CropPath(follow=cfg.follow, times=[0.0, duration], x=[x, x], **base)

    _, frames = iter_frames(media, cfg.sample_fps, min(PLAN_WIDTH, work_size(media, PLAN_WIDTH)[0]),
                            prefer_hardware, start=start, duration=duration)
    columns = [saliency(f).sum(axis=0) for f in frames]
    if not columns:
        raise VerticalError(f"No se pudieron leer fotogramas de {start:.2f}–{end:.2f} s")
    left = follow_path(np.asarray(columns), frac, cfg)
    times = np.arange(len(left)) / cfg.sample_fps
    return CropPath(follow=cfg.follow, times=times.tolist(),
                    x=(left * src_w).tolist(), **base)


def preview_image(media: MediaInfo, path: CropPath, start: float, end: float,
                  png: str, count: int = 5, width: int = 384) -> None:
    """One PNG: ``count`` frames across the clip with the 9:16 window drawn
    and the rest dimmed. Writes only ``png``."""
    w = width // 2 * 2
    h = max(2, int(round(w * path.src_h / path.src_w)) // 2 * 2)
    sx = w / path.src_w
    tiles = []
    for i in range(count):
        rel = (end - start) * (i + 0.5) / count
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{start + rel:.3f}",
             "-i", media.path, "-map", "0:v:0", "-frames:v", "1", "-vf", f"scale={w}:{h}",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
        if proc.returncode != 0 or len(proc.stdout) < w * h * 3:
            raise VerticalError(f"ffmpeg no pudo leer el fotograma a {start + rel:.2f} s")
        img = np.frombuffer(proc.stdout[:w * h * 3], np.uint8).reshape(h, w, 3).copy()
        x0 = int(round(float(path.at(rel)) * sx))
        x1 = x0 + int(round(path.crop_w * sx))
        dim = (img * 0.35).astype(np.uint8)
        dim[:, x0:x1] = img[:, x0:x1]
        cv2.rectangle(dim, (x0, 0), (max(x0, x1 - 1), h - 1), (0, 220, 255), 2)
        cv2.putText(dim, f"{start + rel:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(dim)
    if not cv2.imwrite(png, np.hstack(tiles)):
        raise VerticalError(f"No se pudo escribir {png}")
