"""Pick ONE continuous segment of a video: the best window, never a montage.

Input is F2's :class:`~dronereel.analyze.Analysis`. Output is a single
``[start, end)`` inside one shot (no window crosses a cut), plus a log of the
runner-up candidates so a person can see what was weighed — and, from F6 on,
say which one they would have picked.

Why not upstream's ``highlight_select``: its windows have a fixed length and
are centred on the top-scoring second, and the pipeline then concatenates
several of them. Here the length varies between ``min_duration`` and
``max_duration``, the window is scored as a whole, and there is only ever one.

Scoring, in two levels — all heuristics, all weights configurable
===================================================================
Per sample, components in 0..1 (``COMPONENTS``):

* ``exposure``  — penalises blown highlights, crushed shadows and darkness.
* ``clarity``   — sharpness relative to this video's median (blur, mush).
* ``stability`` — coherence of the movement times a jerk penalty. Unknown
  movement (open sky, water) is 0.5: not known to be good or bad.
* ``dynamism``  — how much the camera moves, *saturating* and multiplied by
  coherence: a fast, clean line scores; fast chaos does not, and twice as fast
  is not twice as good. Intense movement is not quality.
* ``manoeuvre`` — rotation (roll, yaw/pitch turns) that is coherent and
  smooth. It does NOT recognise named FPV manoeuvres (flip, dive, power loop);
  it measures rotation rates. Recognising those would need labelled examples.
* ``detail``    — edges and colourfulness on an absolute scale: something to
  look at. A weak stand-in for "interesting landscape" until there are
  embeddings and your preferences (F7).
* ``proximity`` — parallax with coherence, saturating early and with a low
  default weight: close flying is not assumed to be better.

Per window: the weighted mean of the sample quality, blended with its worst
second (a clip is as good as its weakest moment), plus variety of movement,
minus the share of empty samples (little movement and little to look at),
minus missing context around the highlight, plus a slight preference for
length.

The highlight is the stretch of the window within ``HIGHLIGHT_TOL`` of its
best second, so a sustained manoeuvre counts as a whole, not as its first
instant. Context means ``pre_roll`` seconds before it and ``post_roll`` after
it inside the window, so the manoeuvre does not start or end abruptly. The
penalty scales with how much the highlight stands out from the rest of the
window: on uniform footage there is no event to frame, and chasing context
there would only drag the window into worse material.

The constants below were set on synthetic clips and on reasoning about the
units, not on real Avata 2 / Osmo Nano footage yet; expect to retune them.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import numpy as np

from dronereel.analyze import Analysis

SELECT_VERSION = 1

COMPONENTS = ("exposure", "clarity", "stability", "dynamism", "manoeuvre",
              "detail", "proximity")

# Scales, in F2's units. sat(x, x0) = 1 - exp(-x / x0): ~63 % at x0.
JERK_SCALE = 0.8            # jerk at which stability drops to 1/e
MOTION_SCALE = 0.3          # frame-widths/s (+ roll/90°, zoom log/s)
ROLL_SCALE = 30.0           # °/s
TURN_SCALE = 0.3            # frame-widths/s of yaw/pitch turn
EDGES_SCALE = 0.05          # share of edge pixels
COLOR_SCALE = 30.0          # Hasler-Süsstrunk ("moderately colourful" ~33)
PARALLAX_SCALE = 0.03       # frame-widths/s
VARIETY_SCALE = 0.15
HIGHLIGHT_TOL = 0.03       # quality within this of the best second
CONTRAST_SCALE = 0.1       # highlight-over-window-mean for the full context penalty
EMPTY_DYNAMISM = 0.1        # below both of these, a sample is "empty"
EMPTY_DETAIL = 0.25
MAX_LENGTHS = 60            # durations tried between min and max
CHUNK_VALUES = 1 << 21      # floats per window slice (~16 MB): RAM on the Pi


@dataclass
class SelectConfig:
    min_duration: float = 10.0
    max_duration: Optional[float] = 30.0      # None or 0: up to the whole shot
    duration_step: float = 1.0
    pre_roll: float = 2.0                     # context before the peak, s
    post_roll: float = 2.0                    # context after the peak, s
    edge_slack: float = 1.0                   # how far an edge may move to a calm moment
    weights: dict = field(default_factory=lambda: {
        "exposure": 1.0, "clarity": 1.0, "stability": 1.5, "dynamism": 1.0,
        "manoeuvre": 1.0, "detail": 1.0, "proximity": 0.3})
    worst: float = 0.25        # blend of the worst second into the window mean
    variety: float = 0.1
    empty: float = 0.3
    context: float = 0.05      # per second of missing pre/post roll
    length: float = 0.02       # bonus at max length, 0 at min
    candidates: int = 8        # non-overlapping alternatives kept in the log

    @classmethod
    def from_dict(cls, d: dict) -> "SelectConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Opciones de selección desconocidas: {sorted(unknown)}")
        cfg = cls(**{k: v for k, v in d.items() if k != "weights"})
        if "weights" in d:
            bad = set(d["weights"]) - set(COMPONENTS)
            if bad:
                raise ValueError(f"Pesos desconocidos: {sorted(bad)}")
            cfg.weights = {**cfg.weights, **d["weights"]}
        return cfg


@dataclass
class Window:
    start: float
    end: float
    score: float
    peak: float                 # time of the best moment inside
    parts: dict                 # mean of each component + window terms
    highlight: tuple = (0.0, 0.0)   # [start, end) of the stretch near the peak

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "duration": round(self.duration, 3), "score": round(self.score, 4),
                "peak": round(self.peak, 3),
                "highlight": [round(self.highlight[0], 3), round(self.highlight[1], 3)],
                "parts": {k: round(float(v), 4) for k, v in self.parts.items()}}


@dataclass
class Selection:
    best: Window                # the one segment to export
    scored: Window              # the best window before moving its edges
    candidates: list            # non-overlapping alternatives, best first (incl. scored)
    config: SelectConfig
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"best": self.best.to_dict(), "scored": self.scored.to_dict(),
                "candidates": [c.to_dict() for c in self.candidates],
                "config": asdict(self.config), "warnings": list(self.warnings)}


def _sat(x, x0):
    return 1.0 - np.exp(-np.asarray(x, dtype=np.float64) / x0)


def components(a: Analysis) -> dict:
    """Per-sample components, each an array in 0..1 (see module docstring)."""
    s = a.samples
    known = ~np.isnan(s["roll"])
    coh = np.where(known, np.nan_to_num(s["coherence"]), 0.0)
    jerk = s["jerk"]
    smooth = np.where(np.isnan(jerk), 1.0, np.exp(-np.nan_to_num(jerk) / JERK_SCALE))

    exposure = np.clip(1.0 - 4.0 * s["clip_high"] - 4.0 * s["clip_low"], 0.0, 1.0)
    exposure = exposure * np.clip(s["brightness"] / 0.12, 0.0, 1.0)

    sharp = s["sharpness"]
    med = float(np.nanmedian(sharp)) if np.any(sharp > 0) else 0.0
    clarity = np.clip(sharp / med, 0.0, 1.0) if med > 0 else np.ones_like(sharp)

    stability = np.where(known, coh * smooth, 0.5)

    pan = np.nan_to_num(np.hypot(s["pan_x"], s["pan_y"]))
    roll = np.nan_to_num(np.abs(s["roll"]))
    zoom = np.nan_to_num(np.abs(s["zoom"]))
    dynamism = _sat(pan + roll / 90.0 + zoom, MOTION_SCALE) * coh

    turn = roll / ROLL_SCALE + pan / TURN_SCALE
    manoeuvre = (1.0 - np.exp(-turn)) * coh * smooth

    detail = (0.6 * _sat(s["edges"], EDGES_SCALE)
              + 0.4 * _sat(s["colorfulness"], COLOR_SCALE))

    proximity = _sat(np.nan_to_num(s["parallax"]), PARALLAX_SCALE) * coh

    return {"exposure": exposure, "clarity": clarity, "stability": stability,
            "dynamism": dynamism, "manoeuvre": manoeuvre, "detail": detail,
            "proximity": proximity}


def quality(comp: dict, weights: dict) -> np.ndarray:
    total = sum(max(0.0, float(weights.get(k, 0.0))) for k in COMPONENTS)
    if total <= 0:
        raise ValueError("Todos los pesos son cero")
    return sum(max(0.0, float(weights.get(k, 0.0))) * comp[k] for k in COMPONENTS) / total


def shots(a: Analysis) -> list:
    """``[(first, end)]`` sample ranges between cuts; end is exclusive."""
    n = a.count
    bounds = sorted({min(n, max(0, int(round(c * a.fps)))) for c in a.cuts} - {0, n})
    edges = [0, *bounds, n]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
            if edges[i + 1] > edges[i]]


def _moving_mean(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(x)))
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty(len(x))
    half = k // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i - half + k)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


class _Scorer:
    """Every window of one shot, scored in vectorised passes per length."""

    def __init__(self, a: Analysis, cfg: SelectConfig):
        self.a, self.cfg, self.fps = a, cfg, a.fps
        self.comp = components(a)
        self.q = quality(self.comp, cfg.weights)
        one_s = max(1, int(round(a.fps)))
        self.q1 = _moving_mean(self.q, one_s)            # for peak and worst second
        self.empty = ((self.comp["dynamism"] < EMPTY_DYNAMISM)
                      & (self.comp["detail"] < EMPTY_DETAIL)).astype(np.float64)
        s = a.samples
        self.mv = np.stack([s["pan_x"], s["pan_y"], s["roll"] / 90.0, s["zoom"]], axis=1)
        self.cum = {k: np.concatenate([[0.0], np.cumsum(v)])
                    for k, v in {**self.comp, "q": self.q, "empty": self.empty}.items()}
        known = ~np.isnan(self.mv)
        x = np.nan_to_num(self.mv)
        self.mv_n = np.concatenate([np.zeros((1, 4)), np.cumsum(known, axis=0)])
        self.mv_s = np.concatenate([np.zeros((1, 4)), np.cumsum(x, axis=0)])
        self.mv_s2 = np.concatenate([np.zeros((1, 4)), np.cumsum(x * x, axis=0)])

    def lengths(self, size: int) -> list:
        cfg, fps = self.cfg, self.fps
        lo = max(1, int(round(cfg.min_duration * fps)))
        hi = size if not cfg.max_duration else int(round(cfg.max_duration * fps))
        hi = min(hi, size)
        if lo >= hi:
            return [min(lo, size)]
        step = max(1, int(round(cfg.duration_step * fps)), math.ceil((hi - lo) / MAX_LENGTHS))
        out = list(range(lo, hi + 1, step))
        if out[-1] != hi:
            out.append(hi)
        return out

    def score(self, s: np.ndarray, L: int, lo_len: int, hi_len: int) -> tuple:
        """Scores of windows ``[s, s+L)`` (``s`` an array), their parts, and
        the absolute sample indices of peak, highlight start and highlight end
        (exclusive)."""
        cfg, fps = self.cfg, self.fps
        e = s + L
        mean = {k: (c[e] - c[s]) / L for k, c in self.cum.items()}
        view = np.lib.stride_tricks.sliding_window_view(self.q1, L)
        worst = np.empty(len(s))
        contrast = np.empty(len(s))
        peak_off = np.empty(len(s), dtype=np.int64)
        hl_first = np.empty(len(s), dtype=np.int64)
        hl_last = np.empty(len(s), dtype=np.int64)
        chunk = max(1, CHUNK_VALUES // L)        # indexing the view copies it
        for i in range(0, len(s), chunk):
            sl = slice(i, i + chunk)
            win = view[s[sl]]
            top = win.max(axis=1)
            worst[sl] = win.min(axis=1)
            contrast[sl] = top - win.mean(axis=1)
            peak_off[sl] = win.argmax(axis=1)
            near = win >= (top - HIGHLIGHT_TOL)[:, None]
            hl_first[sl] = near.argmax(axis=1)
            hl_last[sl] = L - 1 - near[:, ::-1].argmax(axis=1)
        n = np.maximum(self.mv_n[e] - self.mv_n[s], 1)
        m1 = (self.mv_s[e] - self.mv_s[s]) / n
        var = np.clip((self.mv_s2[e] - self.mv_s2[s]) / n - m1 * m1, 0, None)
        variety = _sat(np.sqrt(var).sum(axis=1), VARIETY_SCALE)
        before = hl_first / fps
        after = (L - 1 - hl_last) / fps
        missing = (np.maximum(0.0, cfg.pre_roll - before)
                   + np.maximum(0.0, cfg.post_roll - after))
        missing = missing * np.clip(contrast / CONTRAST_SCALE, 0.0, 1.0)
        span = hi_len - lo_len
        length = (L - lo_len) / span if span > 0 else 1.0
        total = ((1.0 - cfg.worst) * mean["q"] + cfg.worst * worst
                 + cfg.variety * variety - cfg.empty * mean["empty"]
                 - cfg.context * missing + cfg.length * length)
        parts = {**{k: mean[k] for k in COMPONENTS}, "quality": mean["q"],
                 "worst_second": worst, "variety": variety, "empty": mean["empty"],
                 "missing_context": missing}
        return total, parts, (s + peak_off, s + hl_first, s + hl_last + 1)

    def window(self, s: int, L: int, lo_len: int, hi_len: int) -> Window:
        total, parts, (peak, hl0, hl1) = self.score(np.array([s]), L, lo_len, hi_len)
        return Window(start=s / self.fps, end=(s + L) / self.fps, score=float(total[0]),
                      peak=float(peak[0]) / self.fps,
                      parts={k: float(v[0]) for k, v in parts.items()},
                      highlight=(float(hl0[0]) / self.fps, float(hl1[0]) / self.fps))


def _calm_edge(jerk: np.ndarray, options: Iterable[int], prefer: int) -> int:
    """The option with the least jerk; ties go to the one nearest ``prefer``."""
    j = np.where(np.isnan(jerk), np.nanmedian(jerk) if np.any(~np.isnan(jerk)) else 0.0, jerk)
    return min(options, key=lambda i: (round(float(j[min(i, len(j) - 1)]), 6), abs(i - prefer)))


def select(a: Analysis, cfg: Optional[SelectConfig] = None) -> Selection:
    """The single best continuous window of ``a``. Deterministic."""
    cfg = cfg or SelectConfig()
    if cfg.min_duration <= 0:
        raise ValueError("min_duration debe ser > 0")
    if cfg.max_duration and cfg.max_duration < cfg.min_duration:
        raise ValueError("max_duration debe ser ≥ min_duration")
    if a.count < 2:
        raise ValueError("El análisis no tiene muestras suficientes")
    sc = _Scorer(a, cfg)
    warnings = []
    pool = []                      # (score, start, L, lo_len, hi_len)
    ranges = shots(a)
    longest = max(e - s for s, e in ranges)
    if longest < cfg.min_duration * a.fps:
        warnings.append(
            f"Ninguna toma llega a la duración mínima ({cfg.min_duration:g} s); "
            f"se usa la toma completa más larga ({longest / a.fps:.1f} s).")
    for first, end in ranges:
        size = end - first
        if size < cfg.min_duration * a.fps and size < longest:
            continue
        lens = sc.lengths(size)
        lo_len, hi_len = lens[0], lens[-1]
        for L in lens:
            starts = np.arange(first, end - L + 1)
            total, _, _ = sc.score(starts, L, lo_len, hi_len)
            pool.extend((float(t), int(st), L, lo_len, hi_len) for t, st in zip(total, starts))
    # Best first; ties: earlier, then longer.
    pool.sort(key=lambda p: (-round(p[0], 9), p[1], -p[2]))

    chosen = []
    for t, st, L, lo_len, hi_len in pool:
        if len(chosen) >= max(1, cfg.candidates):
            break
        if all(_overlap(st, st + L, c[1], c[1] + c[2]) < 0.3 for c in chosen):
            chosen.append((t, st, L, lo_len, hi_len))
    candidates = [sc.window(st, L, lo, hi) for _, st, L, lo, hi in chosen]
    scored = candidates[0]

    _, st, L, lo_len, hi_len = chosen[0]
    best = _refine(sc, a, cfg, st, L, lo_len, hi_len, scored)
    if scored.parts["missing_context"] > 0.05:
        warnings.append(
            f"El mejor momento ({scored.highlight[0]:.1f}–{scored.highlight[1]:.1f} s) no "
            f"cabe con todo su contexto: está cerca de un borde del vídeo o de un corte, "
            f"o la duración máxima es corta.")
    return Selection(best=best, scored=scored, candidates=candidates, config=cfg,
                     warnings=warnings)


def _overlap(s1, e1, s2, e2) -> float:
    inter = max(0, min(e1, e2) - max(s1, s2))
    return inter / max(1, min(e1 - s1, e2 - s2))


def _refine(sc, a, cfg, st, L, lo_len, hi_len, scored: Window) -> Window:
    """Move each edge up to ``edge_slack`` to the calmest sample, staying in the
    same shot, inside the allowed lengths and keeping the highlight's context."""
    slack = int(round(cfg.edge_slack * a.fps))
    if slack <= 0:
        return scored
    first, end = next((f, e) for f, e in shots(a) if f <= st < e)
    hl0 = int(round(scored.highlight[0] * a.fps))
    hl1 = int(round(scored.highlight[1] * a.fps))
    pre = int(round(cfg.pre_roll * a.fps))
    post = int(round(cfg.post_roll * a.fps))
    jerk = a.samples["jerk"]
    e0 = st + L

    def ok(s, e):
        return (first <= s and e <= end and lo_len <= e - s <= hi_len
                and s <= hl0 and e >= hl1
                and (s <= hl0 - pre or s <= st) and (e >= hl1 + post or e >= e0))

    starts = [s for s in range(st - slack, st + slack + 1) if ok(s, e0)]
    s1 = _calm_edge(jerk, starts, st) if starts else st
    ends = [e for e in range(e0 - slack, e0 + slack + 1) if ok(s1, e)]
    # The end edge is the last sample shown: e - 1.
    e1 = _calm_edge(jerk, [e - 1 for e in ends], e0 - 1) + 1 if ends else e0
    if (s1, e1) == (st, e0):
        return scored
    return sc.window(s1, e1 - s1, lo_len, hi_len)


def describe(a: Analysis, start: float, end: float,
             cfg: Optional[SelectConfig] = None) -> Window:
    """Score an arbitrary ``[start, end)`` the way :func:`select` scores its
    windows — e.g. one the user picked by hand — so it can be compared with
    the windows the selector preferred. It may cross a cut; that is the
    user's call."""
    cfg = cfg or SelectConfig()
    s = max(0, min(a.count - 1, int(round(start * a.fps))))
    e = max(s + 1, min(a.count, int(round(end * a.fps))))
    sc = _Scorer(a, cfg)
    lo = max(1, int(round(cfg.min_duration * a.fps)))
    hi = int(round(cfg.max_duration * a.fps)) if cfg.max_duration else a.count
    return sc.window(s, e - s, min(lo, e - s), max(hi, e - s))


def log_record(sel: Selection, a: Analysis, video: str, analysis_key: str) -> dict:
    """What gets written next to the result: enough to reproduce the choice
    and, later (F6), to learn from what the user accepts or picks instead."""
    return {"version": SELECT_VERSION, "video": os.path.abspath(video),
            "analysis_key": analysis_key,
            "analysis": {"fps": a.fps, "width": a.width, "duration": a.duration,
                         "cuts": a.cuts},
            **sel.to_dict()}


def write_log(record: dict, path: str) -> None:
    with open(path, "x", encoding="utf-8") as f:       # never overwrite
        json.dump(record, f, indent=1, ensure_ascii=False)
        f.write("\n")
