"""Your ratings, kept as training data (F6). Append-only JSONL.

One line per judgement, never rewritten:

* ``rating``      — about an exported clip: a score (1-5), accept/reject,
  and/or "alternative N was better", plus a note.
* ``manual``      — "this is the stretch I would have picked" for a video.

Each entry carries what learning needs (F7) without needing the video
again: the **features** of the judged window (the selector's per-window
parts), the features of the selector's other candidates for that video, the
selector settings in force, and where it came from — the source's SHA-256
(not just its path, which differs between the Pi and the Mac) and the clip.
Videos are never copied into the dataset: it holds numbers and references.

Portable: the file lives in ``~/.local/share/dronereel/ratings.jsonl``
(``$XDG_DATA_HOME`` if set) on any machine; every entry has a random id, so
two machines' files merge by concatenation without duplicates
(:func:`merge`).
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Iterable, Optional

DATASET_VERSION = 1
DATA_DIR = os.path.join(
    os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "dronereel")
DEFAULT_PATH = os.path.join(DATA_DIR, "ratings.jsonl")
VERDICTS = ("accept", "reject")


class DatasetError(ValueError):
    pass


def read(path: str = DEFAULT_PATH) -> list:
    """All entries; a missing file is an empty dataset. A damaged line (e.g.
    a write cut short) is skipped and reported, not fatal."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"AVISO: línea {n} de {path} dañada; se ignora")
                continue
            if isinstance(entry, dict) and entry.get("id"):
                entries.append(entry)
    return entries


def append(entries: Iterable[dict], path: str = DEFAULT_PATH) -> int:
    """Append ``entries`` (one JSON line each). Never rewrites the file."""
    entries = list(entries)
    if not entries:
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return len(entries)


def merge(other: str, path: str = DEFAULT_PATH) -> int:
    """Append the entries of ``other`` whose id is not in ``path`` yet."""
    have = {e["id"] for e in read(path)}
    new = [e for e in read(other) if e["id"] not in have]
    return append(new, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _window(w: dict) -> dict:
    """The fields of a selector window worth keeping."""
    return {"start": w["start"], "end": w["end"], "score": w.get("score"),
            "features": dict(w.get("parts") or {})}


def _base(kind: str, source: dict, selection: Optional[dict], note: str) -> dict:
    entry = {"id": uuid.uuid4().hex, "version": DATASET_VERSION, "kind": kind,
             "created": _now(), "source": source, "note": note or ""}
    if selection:
        entry["selector"] = {"version": selection.get("version"),
                             "config": selection.get("config"),
                             "analysis": selection.get("analysis")}
        entry["candidates"] = [_window(c) for c in selection.get("candidates") or []]
        entry["chosen"] = _window(selection["best"]) if selection.get("best") else None
    return entry


def rating_entry(record: dict, record_path: str, score: Optional[int] = None,
                 verdict: Optional[str] = None, prefer: Optional[int] = None,
                 note: str = "", features: Optional[dict] = None) -> dict:
    """An entry about the clip described by an export ``record`` (the JSON
    that ``export`` writes beside the clip). ``prefer`` is the 1-based
    position in the selector's list: 1 is the clip itself, 2 the first
    alternative. ``features``: the clip window's, when the record has none
    (a range given by hand)."""
    if score is None and verdict is None and prefer is None:
        raise DatasetError("Indica al menos una puntuación, aceptar/rechazar o una alternativa")
    if score is not None and not 1 <= int(score) <= 5:
        raise DatasetError("La puntuación va de 1 a 5")
    if verdict is not None and verdict not in VERDICTS:
        raise DatasetError(f"Veredicto desconocido: {verdict}")
    exp = record.get("export")
    if not exp or "source" not in record:
        raise DatasetError(f"{record_path} no es un registro de exportación de DroneReel")
    selection = record.get("selection") if not (record.get("selection") or {}).get("manual") else None
    if prefer is not None:
        n = len((selection or {}).get("candidates") or [])
        if not 1 <= int(prefer) <= n:
            raise DatasetError(f"Alternativa {prefer} no existe (hay {n} en la lista)")
    entry = _base("rating", {k: record["source"].get(k) for k in
                             ("path", "sha256", "duration", "fps")}, selection, note)
    clip_features = features
    if clip_features is None and selection and selection.get("best"):
        clip_features = dict(selection["best"].get("parts") or {})
    entry.update({
        "clip": {"path": exp.get("path"), "record": os.path.abspath(record_path),
                 "start": exp.get("start"), "end": exp.get("end"),
                 "requested": exp.get("requested"), "mode": exp.get("mode"),
                 "features": clip_features},
        "score": int(score) if score is not None else None,
        "verdict": verdict,
        "prefer": int(prefer) if prefer is not None else None,
        "color": record.get("color"),
    })
    return entry


def manual_entry(source: dict, start: float, end: float, features: dict,
                 selection: Optional[dict], score: Optional[int] = None,
                 note: str = "") -> dict:
    """"This is the stretch I would pick": ``[start, end)`` of the source,
    with its features and the selector's own candidates for comparison."""
    if not 0 <= start < end:
        raise DatasetError(f"Tramo no válido: {start}–{end}")
    if score is not None and not 1 <= int(score) <= 5:
        raise DatasetError("La puntuación va de 1 a 5")
    entry = _base("manual", source, selection, note)
    entry.update({"segment": {"start": round(start, 3), "end": round(end, 3),
                              "features": features},
                  "score": int(score) if score is not None else None})
    return entry


def rated_clips(entries: Iterable[dict]) -> set:
    """Export records already rated (by absolute path of the record)."""
    return {e["clip"]["record"] for e in entries
            if e.get("kind") == "rating" and e.get("clip", {}).get("record")}


def summary(entries: list) -> dict:
    out = {"total": len(entries), "ratings": 0, "manual": 0, "accept": 0, "reject": 0,
           "prefer_alternative": 0, "scores": {}, "sources": len({
               (e.get("source") or {}).get("sha256") for e in entries})}
    for e in entries:
        kind = e.get("kind")
        if kind == "rating":
            out["ratings"] += 1
            if e.get("verdict") in VERDICTS:
                out[e["verdict"]] += 1
            if e.get("prefer") and e["prefer"] > 1:
                out["prefer_alternative"] += 1
        elif kind == "manual":
            out["manual"] += 1
        if e.get("score"):
            out["scores"][e["score"]] = out["scores"].get(e["score"], 0) + 1
    return out


def find_records(folder: str) -> list:
    """Export records (``.json`` written by ``export``) in ``folder``, sorted."""
    found = []
    if not os.path.isdir(folder):
        return found
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rec, dict) and "export" in rec and "source" in rec:
            found.append((os.path.abspath(path), rec))
    return found
