"""Per-sample measurements of a whole video: what the selector (F3) scores.

Every sample is one small frame at ``t = k / fps``. Each measurement is a
*description*, not a verdict — "the image rotates at 40°/s" rather than "good
roll". Turning descriptions into a score is F3's job, and later a model's.

Camera movement: a similarity model, not upstream's phase correlation
=====================================================================
``modules/segments/shot_window.py`` measures movement as a rigid shift
(phase correlation). That fits a handheld camera; it does not fit FPV, where
the commonest smooth movements are a roll and flying forward, which to the
image are a rotation and a zoom. Measured on synthetic clips of perfectly
smooth movement (DroneReel F2 notes):

    movement     phase-correlation response    similarity-model inliers
    pan          0.98                          0.99
    roll         0.22                          0.99  (3.98°/step, 4.0 expected)
    zoom         0.64                          0.99

So upstream's coherence would call a clean roll chaotic. Here, corners are
tracked between consecutive samples (Lucas-Kanade) and a similarity transform
(shift, rotation, scale) is fitted with RANSAC. From it:

* ``pan_x``, ``pan_y`` — camera movement in frame-widths per second, measured
  at the image centre; positive = camera turning right / up (the image moving
  left / down).
* ``roll`` — image rotation, degrees per second, clockwise positive.
* ``zoom`` — log-scale change per second; positive = image expanding, which
  is what flying forward looks like.
* ``coherence`` — share of tracked points the model explains (0..1).
* ``parallax`` — median distance, in frame-widths per second, of tracked
  points from where the model puts them. Depth makes near things move more
  than far ones, so this is high when flying close to terrain or objects —
  but it is also high for chaos, so it only means something next to
  ``coherence``. It is a description; nothing here claims it measures distance.
* ``tracked`` — points tracked. Featureless frames (open sky, water, blur)
  track few; their movement is then *unknown* (NaN), not zero.

Picture: ``brightness`` (mean luma 0..1), ``clip_high`` (share of *bright*
pixels with a channel at the top of the range: a blown sky, a clipped sunset —
luma alone misses a tinted clip, which reads ~220 grey; a channel alone counts
a vivid red flower, which is saturated but dark),
``clip_low`` (share of pixels at black: crushed shadows),
``sharpness`` (variance of the Laplacian, upstream's ``clip_quality``),
``colorfulness`` (Hasler & Süsstrunk), ``edges`` (share of edge pixels: how
much there is to look at), ``diff`` (mean change from the previous sample).

Cuts use upstream's ``scene_cuts`` (median + robust deviations) with an
absolute floor: one DJI file is one recording and normally has none, and on a
continuous take the adaptive threshold alone would call every fast turn a cut.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from dronereel.frames import iter_frames
from dronereel.media import MediaInfo
from modules.segments import scene_cuts
from modules.segments.clip_quality import laplacian_sharpness

ANALYSIS_VERSION = 1
DEFAULT_FPS = 5.0
DEFAULT_WIDTH = 384

MAX_CORNERS = 400
MIN_TRACKED = 12                # fewer points than this: movement unknown
RANSAC_PX = 1.5
# Clipped highlight: brightest channel at HIGH_CHANNEL and grey at least
# CLIP_MIN_LUMA. On test frames this put a vividly coloured but properly
# exposed image at 0.2 % and an over-exposed one at 97.6 %.
HIGH_CHANNEL = 250
CLIP_MIN_LUMA = 200
LOW_LUMA = 5
# Mean absolute grey change (0-255) a cut must also clear. A hard cut between
# two shots measures 30+ on most footage; a fast turn over similar ground
# stays well under it at 5 samples/s.
CUT_FLOOR = 25.0

CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
    "dronereel", "analysis")

FIELDS = ("t", "brightness", "clip_high", "clip_low", "sharpness", "colorfulness",
          "edges", "diff", "pan_x", "pan_y", "roll", "zoom", "coherence",
          "parallax", "tracked", "jerk")


@dataclass
class Analysis:
    path: str
    fps: float
    width: int
    height: int
    decoder: str
    seconds: float                  # wall time the analysis took
    duration: float                 # video duration covered
    samples: dict                   # field -> np.ndarray, one entry per sample
    cuts: list = field(default_factory=list)   # cut times, seconds
    cached: bool = False

    @property
    def count(self) -> int:
        return len(self.samples["t"])

    def per_second(self, name: str, how: str = "mean") -> np.ndarray:
        """``name`` reduced to one value per whole second (NaN-aware)."""
        t, v = self.samples["t"], self.samples[name]
        n = int(math.ceil(self.duration)) if self.duration else int(t[-1]) + 1
        out = np.full(n, np.nan)
        reduce = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}[how]
        sec = np.floor(t).astype(int)
        for s in range(n):
            vals = v[sec == s]
            if vals.size and not np.all(np.isnan(vals)):
                out[s] = reduce(vals)
        return out


def content_key(path: str, fps: float, width: int) -> str:
    """Cache key from the file's size and its first and last MiB, plus the
    analysis settings. Survives a move or copy of the file; a hash of the
    whole multi-GB file would cost more than reading it once."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    h.update(f"{ANALYSIS_VERSION}|{fps}|{width}|{size}|".encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()[:24]


def colorfulness(bgr: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness, roughly 0 (grey) to 100+."""
    b, g, r = (bgr[..., i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))


UNKNOWN_MOTION = {"pan_x": np.nan, "pan_y": np.nan, "roll": np.nan, "zoom": np.nan,
                  "coherence": 0.0, "parallax": np.nan, "tracked": 0}


def similarity_motion(prev: np.ndarray, gray: np.ndarray, dt: float) -> dict:
    """Movement from ``prev`` to ``gray`` (see module docstring for units)."""
    unknown = UNKNOWN_MOTION
    pts = cv2.goodFeaturesToTrack(prev, MAX_CORNERS, 0.01, 8)
    if pts is None or len(pts) < MIN_TRACKED:
        return unknown
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None,
                                              winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    a, b = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
    if len(a) < MIN_TRACKED:
        return {**unknown, "tracked": int(len(a))}
    m, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                             ransacReprojThreshold=RANSAC_PX)
    if m is None:
        return {**unknown, "tracked": int(len(a))}
    h, w = gray.shape
    centre = np.array([w / 2.0, h / 2.0])
    moved = m[:, :2] @ centre + m[:, 2] - centre      # where the centre went
    scale = math.hypot(m[0, 0], m[1, 0])
    predicted = a @ m[:, :2].T + m[:, 2]
    residual = np.linalg.norm(b - predicted, axis=1)
    return {
        "pan_x": float(-moved[0] / w / dt),
        "pan_y": float(moved[1] / w / dt),
        "roll": float(math.degrees(math.atan2(m[1, 0], m[0, 0])) / dt),
        "zoom": float(math.log(max(scale, 1e-6)) / dt),
        "coherence": float(inliers.mean()) if inliers is not None else 0.0,
        "parallax": float(np.median(residual) / w / dt),
        "tracked": int(len(a)),
    }


def _jerk(s: dict, dt: float) -> np.ndarray:
    """Change of the movement vector per second. Roll is brought to a scale
    comparable with pans (90°/s ~ one frame-width/s); unknown stays unknown."""
    v = np.stack([s["pan_x"], s["pan_y"], s["roll"] / 90.0, s["zoom"]], axis=1)
    out = np.full(len(v), np.nan)
    if len(v) > 1:
        out[1:] = np.linalg.norm(np.diff(v, axis=0), axis=1) / dt
    return out


def detect_cuts(diff: np.ndarray, fps: float) -> list:
    d = np.nan_to_num(diff, nan=0.0)
    if len(d) < 3:
        return []
    threshold = max(CUT_FLOOR, scene_cuts.adaptive_threshold(d))
    idx = scene_cuts.cuts_from_diffs(d, threshold)
    idx = scene_cuts.suppress_adjacent(idx, d, min_gap=max(1, int(fps)))
    return [round(float(i) / fps, 3) for i in idx]


def measure(media: MediaInfo, fps: float = DEFAULT_FPS, width: int = DEFAULT_WIDTH,
            prefer_hardware: bool = True,
            progress: Optional[Callable[[float], None]] = None) -> Analysis:
    """Decode ``media`` at ``fps`` and measure every sample. Reads only."""
    start = time.monotonic()
    src, frames = iter_frames(media, fps, width, prefer_hardware)
    dt = 1.0 / src.fps
    rows = {k: [] for k in FIELDS if k != "jerk"}
    prev = None
    expected = max(1, int(media.duration * fps))
    for k, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n = gray.size
        rows["t"].append(k * dt)
        rows["brightness"].append(float(gray.mean()) / 255.0)
        clipped = (frame.max(axis=2) >= HIGH_CHANNEL) & (gray >= CLIP_MIN_LUMA)
        rows["clip_high"].append(float(np.count_nonzero(clipped)) / n)
        rows["clip_low"].append(float(np.count_nonzero(gray <= LOW_LUMA)) / n)
        rows["sharpness"].append(laplacian_sharpness(gray))
        rows["colorfulness"].append(colorfulness(frame))
        rows["edges"].append(float(np.count_nonzero(cv2.Canny(gray, 100, 200))) / n)
        if prev is None:
            rows["diff"].append(np.nan)
            motion = UNKNOWN_MOTION
        else:
            rows["diff"].append(float(cv2.absdiff(prev, gray).mean()))
            motion = similarity_motion(prev, gray, dt)
        for k2, v in motion.items():
            rows[k2].append(v)
        prev = gray
        if progress and k % max(1, int(fps * 10)) == 0:
            progress(min(1.0, k / expected))

    samples = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    # The first sample has no predecessor; borrow the second's movement rather
    # than scoring the opening as either still or chaotic (as shot_window does).
    if len(samples["t"]) > 1:
        for k in ("pan_x", "pan_y", "roll", "zoom", "coherence", "parallax", "tracked"):
            samples[k][0] = samples[k][1]
    samples["jerk"] = _jerk(samples, dt)
    count = len(samples["t"])
    return Analysis(
        path=media.path, fps=src.fps, width=src.width, height=src.height,
        decoder=src.decoder or "none", seconds=time.monotonic() - start,
        duration=media.duration or count * dt, samples=samples,
        cuts=detect_cuts(samples["diff"], src.fps))


def _cache_files(key: str, cache_dir: str) -> tuple:
    return (os.path.join(cache_dir, f"{key}.npz"), os.path.join(cache_dir, f"{key}.json"))


def save(analysis: Analysis, key: str, cache_dir: str = CACHE_DIR) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    npz, meta = _cache_files(key, cache_dir)
    np.savez_compressed(npz, **analysis.samples)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"version": ANALYSIS_VERSION, "path": analysis.path,
                   "fps": analysis.fps, "width": analysis.width,
                   "height": analysis.height, "decoder": analysis.decoder,
                   "seconds": analysis.seconds, "duration": analysis.duration,
                   "cuts": analysis.cuts}, f, indent=1)


def load(key: str, cache_dir: str = CACHE_DIR) -> Optional[Analysis]:
    npz, meta = _cache_files(key, cache_dir)
    try:
        with open(meta, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("version") != ANALYSIS_VERSION:
            return None
        with np.load(npz, allow_pickle=False) as z:
            samples = {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError):
        return None
    if set(samples) != set(FIELDS):
        return None
    return Analysis(path=m["path"], fps=m["fps"], width=m["width"], height=m["height"],
                    decoder=m["decoder"], seconds=m["seconds"], duration=m["duration"],
                    samples=samples, cuts=list(m["cuts"]), cached=True)


def analyze(media: MediaInfo, fps: float = DEFAULT_FPS, width: int = DEFAULT_WIDTH,
            use_cache: bool = True, cache_dir: str = CACHE_DIR,
            prefer_hardware: bool = True, progress=None) -> Analysis:
    """Cached :func:`measure`. The cache lives under ``~/.cache/dronereel``,
    never beside the video."""
    key = content_key(media.path, fps, width)
    if use_cache:
        hit = load(key, cache_dir)
        if hit is not None:
            hit.path = media.path
            return hit
    result = measure(media, fps, width, prefer_hardware, progress)
    if result.count:
        save(result, key, cache_dir)
    return result


def write_csv(analysis: Analysis, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(FIELDS) + "\n")
        for i in range(analysis.count):
            f.write(",".join(f"{analysis.samples[k][i]:.5g}" for k in FIELDS) + "\n")


def print_progress(fraction: float) -> None:
    print(f"\r  analizando… {fraction * 100:5.1f} %", end="", file=sys.stderr, flush=True)
