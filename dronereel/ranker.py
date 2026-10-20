"""Learn your taste from the ratings dataset (F7). Small, portable, honest.

Not from scratch
================
The hand-made selector (F3) is the starting point. The model learns a
*correction* to it::

    score(window) = selector score + adj · standardise(features)

``features`` are the window parts the selector already computes (exposure,
stability, dynamism… plus worst second, variety, empty share, missing
context and duration). Training is pairwise logistic regression — "A was
preferred to B" — with an L2 penalty on the correction, so with few ratings
the model stays close to the hand-made selector instead of fitting noise.
Pure numpy; the model is a few dozen numbers in a JSON file that runs
anywhere (Pi, Mac). No heavy training, no GPU.

Pairs from the dataset (:func:`pairs`), weight in brackets:

* "alternative N was better": N > the chosen clip [1], N > the others [0.5]
* accepted (or 4-5 stars) without a preference: chosen > alternatives [0.5]
* a stretch picked by hand: it > the chosen clip and every candidate [1]
  (skipping windows that overlap it by more than half)
* across clips: accepted > rejected, and more stars > fewer [0.5]

Before and after
================
:func:`evaluate` holds out whole videos (upstream's ``teach.linear._folds``)
and measures how often each scorer orders the held-out pairs the way you did:
the hand-made selector alone vs with the learned correction. A model is
marked ``better`` — and only then used by default — when it wins on held-out
videos with enough data behind it (``MIN_VIDEOS``, ``MIN_TEST_PAIRS``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from dronereel import selector as sel
from dronereel.dataset import DATA_DIR

RANKER_VERSION = 1
DEFAULT_MODEL = os.path.join(DATA_DIR, "model.json")
FEATURES = (*sel.COMPONENTS, "quality", "worst_second", "variety", "empty",
            "missing_context", "duration")
DURATION_UNIT = 30.0          # seconds; duration feature = seconds / 30
L2 = 1.0                      # on the correction: keeps it near zero with few pairs
STEPS = 600
# Plain gradient descent on a weighted-mean loss. The selector-score
# differences are ~100x smaller than standardised feature differences, so the
# scale k gets a proportionally larger step.
LR_ADJ = 1.0
LR_SCALE = 100.0
INIT_SCALE = 10.0             # selector-score differences are ~0.01-0.3
MIN_SCALE = 0.5
OVERLAP_SAME = 0.5            # windows overlapping more than this are "the same"
MIN_VIDEOS = 5             # 3 looked "better" held-out yet chose worse on new videos (simulated)
MIN_TEST_PAIRS = 30
MIN_GAIN = 0.02               # held-out pairwise accuracy points


# ---------------------------------------------------------------- features

def vector(window: dict) -> Optional[np.ndarray]:
    """Feature vector of a dataset window (``{"start","end","features"}``)."""
    f = window.get("features") or {}
    if not f:
        return None
    dur = float(window.get("end", 0) - window.get("start", 0)) / DURATION_UNIT
    vals = [dur if k == "duration" else f.get(k) for k in FEATURES]
    if any(v is None for v in vals):
        return None
    return np.array(vals, dtype=np.float64)


def base_score(x: np.ndarray, cfg: Optional[dict] = None) -> float:
    """An approximation of the selector's window score from a feature vector,
    without its small length bonus (which depends on the search range). Only
    a fallback: :func:`selector_score` prefers the score the selector itself
    stored, because measuring "before" against an approximation made the
    selector look worse than it is (91 % on its own choices in a test)."""
    c = {**sel.SelectConfig().__dict__, **(cfg or {})}
    f = dict(zip(FEATURES, x))
    return ((1.0 - c["worst"]) * f["quality"] + c["worst"] * f["worst_second"]
            + c["variety"] * f["variety"] - c["empty"] * f["empty"]
            - c["context"] * f["missing_context"])


# ---------------------------------------------------------------- pairs

@dataclass
class Pair:
    better: np.ndarray
    worse: np.ndarray
    weight: float
    groups: frozenset           # videos (source SHA-256) the pair comes from
    base_better: float = 0.0
    base_worse: float = 0.0


def selector_score(window: dict, x: np.ndarray, cfg: Optional[dict]) -> float:
    s = window.get("score")
    return float(s) if s is not None else base_score(x, cfg)


def _overlap(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / max(1e-9, min(a["end"] - a["start"], b["end"] - b["start"]))


def _verdict(e: dict) -> Optional[str]:
    if e.get("verdict"):
        return e["verdict"]
    if e.get("score"):
        return "accept" if e["score"] >= 4 else "reject" if e["score"] <= 2 else None
    return None


def pairs(entries: list) -> list:
    out = []

    def add(a: dict, b: dict, w: float, groups, cfg):
        xa, xb = vector(a), vector(b)
        if xa is None or xb is None or w <= 0:
            return
        out.append(Pair(xa, xb, w, frozenset(groups), selector_score(a, xa, cfg),
                        selector_score(b, xb, cfg)))

    clips = []                  # (entry, clip window, verdict) for cross pairs
    for e in entries:
        g = (e.get("source") or {}).get("sha256") or e.get("id")
        cfg = ((e.get("selector") or {}).get("config")) or None
        cands = e.get("candidates") or []
        if e.get("kind") == "rating":
            clip = {**(e.get("clip") or {})}
            v = _verdict(e)
            prefer = e.get("prefer")
            if prefer and prefer > 1 and prefer <= len(cands):
                win = cands[prefer - 1]
                add(win, clip, 1.0, [g], cfg)
                for i, c in enumerate(cands):
                    if i != prefer - 1 and _overlap(c, win) <= OVERLAP_SAME:
                        add(win, c, 0.5, [g], cfg)
            elif v == "accept":
                for c in cands[1:]:
                    if _overlap(c, clip) <= OVERLAP_SAME:
                        add(clip, c, 0.5, [g], cfg)
            if v or e.get("score"):
                clips.append((e, clip, v, g, cfg))
        elif e.get("kind") == "manual":
            seg = e.get("segment") or {}
            for c in cands:
                if _overlap(c, seg) <= OVERLAP_SAME:
                    add(seg, c, 1.0, [g], cfg)
    for i, (e1, c1, v1, g1, cfg1) in enumerate(clips):
        for e2, c2, v2, g2, _ in clips[i + 1:]:
            if v1 == "accept" and v2 == "reject":
                add(c1, c2, 0.5, [g1, g2], cfg1)
            elif v1 == "reject" and v2 == "accept":
                add(c2, c1, 0.5, [g1, g2], cfg1)
            elif e1.get("score") and e2.get("score") and e1["score"] != e2["score"]:
                hi, lo = (c1, c2) if e1["score"] > e2["score"] else (c2, c1)
                add(hi, lo, abs(e1["score"] - e2["score"]) / 8.0, [g1, g2], cfg1)
    return out


# ---------------------------------------------------------------- model

@dataclass
class Ranker:
    mean: np.ndarray
    std: np.ndarray
    adj: np.ndarray                     # correction per standardised feature
    trained: dict = field(default_factory=dict)   # counts, evaluation, date

    @property
    def better(self) -> bool:
        return bool(self.trained.get("better"))

    def correction(self, x: np.ndarray) -> np.ndarray:
        """Added to the selector's score; ``x`` is ``[..., features]``."""
        return ((np.asarray(x) - self.mean) / self.std) @ self.adj

    def correction_parts(self, parts: dict, duration) -> np.ndarray:
        """Vectorised over the selector's windows (``parts`` of arrays)."""
        cols = [np.asarray(duration, dtype=np.float64) / DURATION_UNIT if k == "duration"
                else np.asarray(parts[k], dtype=np.float64) for k in FEATURES]
        return self.correction(np.stack(np.broadcast_arrays(*cols), axis=-1))

    def to_dict(self) -> dict:
        return {"version": RANKER_VERSION, "features": list(FEATURES),
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "adj": self.adj.tolist(), "trained": self.trained}

    @classmethod
    def from_dict(cls, d: dict) -> "Ranker":
        if d.get("version") != RANKER_VERSION or list(d.get("features") or []) != list(FEATURES):
            raise ValueError("Modelo de otra versión: vuelve a entrenarlo con 'train'")
        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["adj"]),
                   dict(d.get("trained") or {}))


def save(model: Ranker, path: str = DEFAULT_MODEL) -> Optional[str]:
    """Write ``path`` (JSON, no pickle); the model it replaces is kept as
    ``model-<date>.json`` beside it. Returns that backup's path, if any."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    backup = None
    if os.path.exists(path):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        stem, ext = os.path.splitext(path)
        backup = f"{stem}-{stamp}{ext}"
        n = 1
        while os.path.exists(backup):
            backup = f"{stem}-{stamp}_{n}{ext}"
            n += 1
        os.replace(path, backup)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=1)
    os.replace(tmp, path)
    return backup


def load(path: str = DEFAULT_MODEL) -> Optional[Ranker]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return Ranker.from_dict(json.load(f))


def fit(ps: list, l2: float = L2, steps: int = STEPS) -> Ranker:
    """Pairwise logistic regression: P(better ≻ worse) = σ(k·Δbase + adj·Δz),
    L2 on ``adj`` only; ``k`` (how much to trust the selector) is learned and
    kept ≥ ``MIN_SCALE``. Deterministic."""
    xs = np.array([p.better for p in ps] + [p.worse for p in ps])
    mean = xs.mean(axis=0)
    std = xs.std(axis=0)
    std[std < 1e-6] = 1.0
    dz = np.array([(p.better - p.worse) / std for p in ps])
    db = np.array([p.base_better - p.base_worse for p in ps])
    w = np.array([p.weight for p in ps])
    w = w / w.sum()
    k, adj = INIT_SCALE, np.zeros(len(FEATURES))
    for _ in range(steps):
        margin = k * db + dz @ adj
        g = -w / (1.0 + np.exp(np.clip(margin, -50, 50)))      # d loss / d margin
        grad_adj = dz.T @ g + 2 * l2 * adj / len(ps)
        grad_k = float(db @ g)
        adj -= LR_ADJ * grad_adj
        k = max(MIN_SCALE, k - LR_SCALE * grad_k)
    return Ranker(mean, std, adj / k, {"scale": k})


def _accuracy(ps: list, model: Optional[Ranker]) -> float:
    right = total = 0.0
    for p in ps:
        diff = p.base_better - p.base_worse
        if model is not None:
            diff += float(model.correction(p.better) - model.correction(p.worse))
        total += p.weight
        right += p.weight * (1.0 if diff > 0 else 0.5 if diff == 0 else 0.0)
    return right / total if total else float("nan")


def evaluate(ps: list, folds: int = 5) -> dict:
    """Held-out pairwise accuracy, before (selector) and after (with the
    correction), folding by video; a pair is tested only in the fold of its
    video(s) and trained on only when none of its videos is held out."""
    from modules.teach.linear import _folds       # upstream's grouping by video

    videos = sorted({g for p in ps for g in p.groups})
    result = {"pairs": len(ps), "videos": len(videos)}
    if len(videos) < 2:
        result["reason"] = "hacen falta valoraciones de al menos 2 vídeos"
        return result
    fold_of = {}
    for f, idx in enumerate(_folds(videos, min(folds, len(videos)))):
        for i in idx:
            fold_of[videos[i]] = f
    tested, before, after = [], [], []
    for f in sorted(set(fold_of.values())):
        test = [p for p in ps if any(fold_of[g] == f for g in p.groups)]
        train = [p for p in ps if all(fold_of[g] != f for g in p.groups)]
        if not test or not train:
            continue
        model = fit(train)
        tested.extend(test)
        before.append((_accuracy(test, None), sum(p.weight for p in test)))
        after.append((_accuracy(test, model), sum(p.weight for p in test)))
    if not tested:
        result["reason"] = "ningún pliegue tuvo datos para entrenar y probar"
        return result

    def wmean(rows):
        return sum(a * w for a, w in rows) / sum(w for _, w in rows)

    result.update({"test_pairs": len(tested), "before": wmean(before), "after": wmean(after)})
    gain = result["after"] - result["before"]
    enough = len(videos) >= MIN_VIDEOS and len(tested) >= MIN_TEST_PAIRS
    result["better"] = bool(enough and gain >= MIN_GAIN)
    if not enough:
        result["reason"] = (f"pocos datos: {len(videos)} vídeos y {len(tested)} pares de prueba "
                            f"(mínimo {MIN_VIDEOS} y {MIN_TEST_PAIRS})")
    elif gain < MIN_GAIN:
        result["reason"] = "no mejora al selector en vídeos que no ha visto"
    return result


def train(entries: list) -> tuple:
    """``(model or None, evaluation)``: evaluated held-out, then fitted on all."""
    ps = pairs(entries)
    ev = evaluate(ps)
    if not ps:
        ev.setdefault("reason", "no hay pares de preferencia en el dataset")
        return None, ev
    model = fit(ps)
    model.trained.update({
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": len(entries), "pairs": len(ps), "videos": ev.get("videos"),
        "evaluation": {k: v for k, v in ev.items() if k != "better"},
        "better": ev.get("better", False)})
    return model, ev
