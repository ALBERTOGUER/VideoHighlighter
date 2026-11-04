"""Teaching DroneReel your taste from the app: rate the best segments a run
delivered, then train.

Each best segment gets a record beside it (``<segment>.json``) in the layout
DroneReel's own ``export`` writes, so the same ratings file and the same
trainer serve the app and the CLI: ``dronereel review`` sees the app's
segments, and the Taste tab sees the CLI's exports. Ratings go to DroneReel's
dataset (``~/.local/share/dronereel/ratings.jsonl``) and the model to
``model.json`` beside it; neither holds frames or clips, only numbers and
references.

Times in a record are the original video's. When a run worked on a
time-range trim, the trim's selection is shifted by its offset, and the trim
is noted as where the analysis was measured (``app.analysis_video``) so a
stretch picked by hand can be scored against the same measurements.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

PREVIEW_WIDTH = 960


def record_path(segment_path: str) -> str:
    return os.path.splitext(segment_path)[0] + ".json"


def _shift(window: dict, offset: float) -> dict:
    w = dict(window)
    for key in ("start", "end", "peak"):
        if key in w:
            w[key] = round(w[key] + offset, 3)
    if "highlight" in w:
        w["highlight"] = [round(t + offset, 3) for t in w["highlight"]]
    return w


def segment_record(selection: dict, segment_path: str, source: str, offset: float = 0.0,
                   analysis_video: Optional[str] = None, lut: Optional[str] = None) -> dict:
    """The record of a best segment cut to ``segment_path`` from ``source``.

    ``selection`` is DroneReel's ``log_record`` of the choice, in the times
    of ``analysis_video`` (``source`` itself unless the run trimmed it)."""
    from dronereel.media import probe_media
    from dronereel.safety import sha256_file

    media = probe_media(source)
    shifted = dict(selection)
    for key in ("best", "scored"):
        if shifted.get(key):
            shifted[key] = _shift(shifted[key], offset)
    shifted["candidates"] = [_shift(c, offset) for c in shifted.get("candidates") or []]
    best = shifted["best"]
    return {
        "source": {"path": os.path.abspath(source), "sha256": sha256_file(source),
                   "duration": media.duration, "fps": media.fps_fraction},
        "export": {"path": os.path.abspath(segment_path), "start": best["start"],
                   "end": best["end"], "requested": [best["start"], best["end"]],
                   "mode": "app"},
        "selection": shifted,
        "color": {"lut": lut, "applied": bool(lut)},
        "app": {"analysis_video": os.path.abspath(analysis_video or source),
                "offset": offset},
    }


def write_record(record: dict) -> str:
    path = record_path(record["export"]["path"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


# ── what there is to rate ───────────────────────────────────────────────────

def segments(folder: str, dataset: Optional[str] = None) -> list:
    """``[(record_path, record, rated)]`` of every DroneReel record in
    ``folder``, unrated first. A vertical twin is its horizontal record's
    segment, so it is left out."""
    from dronereel import dataset as ds
    done = ds.rated_clips(ds.read(dataset or ds.DEFAULT_PATH))
    found = [(p, r, p in done) for p, r in ds.find_records(folder) if not r.get("twin")]
    return sorted(found, key=lambda item: item[2])


def alternatives(record: dict) -> list:
    """``[(number, start, end)]``: the selector's other candidates, numbered
    the way a rating's ``prefer`` counts them (1 is the chosen clip)."""
    cands = (record.get("selection") or {}).get("candidates") or []
    return [(i, c["start"], c["end"]) for i, c in enumerate(cands[1:], start=2)]


# ── saving a judgement ──────────────────────────────────────────────────────

def rate(record_file: str, record: dict, score=None, verdict=None, prefer=None,
         note: str = "", dataset: Optional[str] = None) -> dict:
    """Append a rating of the segment ``record`` describes."""
    from dronereel import dataset as ds
    entry = ds.rating_entry(record, record_file, score=score, verdict=verdict,
                            prefer=prefer, note=note)
    ds.append([entry], dataset or ds.DEFAULT_PATH)
    return entry


def _analysis_video(record: dict) -> tuple:
    app = record.get("app") or {}
    video = app.get("analysis_video") or record["source"]["path"]
    offset = float(app.get("offset") or 0.0)
    if os.path.exists(video):
        return video, offset
    if os.path.exists(record["source"]["path"]):
        return record["source"]["path"], 0.0        # measured again, whole
    raise FileNotFoundError("the original video is not on this computer any more")


def rate_manual(record: dict, start: float, end: float, score=None, note: str = "",
                dataset: Optional[str] = None) -> dict:
    """Append "this is the stretch I would have picked": ``[start, end)`` in
    the original's seconds, scored on the analysis the selection used."""
    from dronereel import analyze as an
    from dronereel import dataset as ds
    from dronereel import selector as sel
    from dronereel.media import probe_media

    video, offset = _analysis_video(record)
    selection = record.get("selection") or {}
    cfg = sel.SelectConfig.from_dict(selection.get("config") or {})
    analysis = an.analyze(probe_media(video))           # cached by the run
    window = sel.describe(analysis, start - offset, end - offset, cfg)
    entry = ds.manual_entry(record["source"], start, end, window.to_dict()["parts"],
                            selection, score=score, note=note, selector_score=window.score)
    ds.append([entry], dataset or ds.DEFAULT_PATH)
    return entry


# ── watching an alternative ─────────────────────────────────────────────────

def preview_path(record: dict, start: float, end: float) -> str:
    from dronereel.analyze import CACHE_DIR
    sha = (record.get("source") or {}).get("sha256", "x")[:12]
    return os.path.join(CACHE_DIR, "previews", f"{sha}_{start:.1f}-{end:.1f}.mp4")


def make_preview(record: dict, start: float, end: float) -> str:
    """A small, quick H.264 of ``[start, end)`` of the original (with the
    run's LUT), cached, so an alternative can be watched before choosing it."""
    from modules.media.lut import lut_filter
    from modules.system.app_paths import ffmpeg_exe

    out = preview_path(record, start, end)
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vf = f"scale={PREVIEW_WIDTH}:-2,format=yuv420p"
    lut = (record.get("color") or {}).get("lut")
    if lut and os.path.exists(lut):
        vf = f"{lut_filter(lut)},{vf}"
    tmp = out + ".part.mp4"
    result = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-i",
         record["source"]["path"], "-t", f"{end - start:.3f}", "-vf", vf,
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
         "-c:a", "aac", "-b:a", "128k", tmp], capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"preview failed: {result.stderr.strip()[-300:]}")
    os.replace(tmp, out)
    return out


def open_file(path: str) -> None:
    import sys
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if sys.platform.startswith("win"):
        os.startfile(path)      # noqa: S606 — the user's own file
        return
    subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── the dataset and the model ───────────────────────────────────────────────

FEATURE_NAMES = {"exposure": "exposure", "clarity": "sharpness", "stability": "stability",
                 "dynamism": "dynamism", "manoeuvre": "turns/manoeuvres",
                 "detail": "detail", "proximity": "proximity", "quality": "average quality",
                 "worst_second": "worst second", "variety": "variety",
                 "empty": "empty stretches", "missing_context": "missing context",
                 "duration": "length"}


def status(dataset: Optional[str] = None, model_path: Optional[str] = None) -> dict:
    """What has been rated so far and the state of the learned model."""
    from dronereel import dataset as ds
    from dronereel import ranker as rk
    out = {"summary": ds.summary(ds.read(dataset or ds.DEFAULT_PATH)), "model": None,
           "needed": {"videos": rk.MIN_VIDEOS, "test_pairs": rk.MIN_TEST_PAIRS}}
    try:
        model = rk.load(model_path or rk.DEFAULT_MODEL)
    except (ValueError, OSError, KeyError) as e:
        out["model_error"] = str(e)
        return out
    if model is not None:
        out["model"] = {"better": model.better, **model.trained}
    return out


def learned(model, top: int = 4) -> list:
    """The few features the model leans on most, as ``+name`` / ``−name``."""
    import numpy as np
    from dronereel import ranker as rk
    order = np.argsort(-np.abs(model.adj))
    return [f"{'+' if model.adj[i] > 0 else '−'}{FEATURE_NAMES.get(rk.FEATURES[i], rk.FEATURES[i])}"
            for i in order[:top] if abs(model.adj[i]) > 0.005]


def train(dataset: Optional[str] = None, model_path: Optional[str] = None) -> dict:
    """Train on every rating, evaluated on held-out videos first; the model
    is saved (the previous one kept as a backup) and used by a run only when
    it beat the selector."""
    from dronereel import dataset as ds
    from dronereel import ranker as rk
    entries = ds.read(dataset or ds.DEFAULT_PATH)
    model, ev = rk.train(entries)
    out = {"entries": len(entries), **{k: ev.get(k) for k in
           ("pairs", "videos", "test_pairs", "before", "after", "reason")},
           "better": bool(ev.get("better")), "saved": None, "learned": []}
    if model is not None:
        out["learned"] = learned(model)
        rk.save(model, model_path or rk.DEFAULT_MODEL)
        out["saved"] = model_path or rk.DEFAULT_MODEL
    return out
