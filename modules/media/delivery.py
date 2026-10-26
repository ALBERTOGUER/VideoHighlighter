"""What a run delivers: the highlight reel, DroneReel's best continuous
segment, or both.

The reel is the app's own result — many short moments joined. The best
segment is DroneReel's (``dronereel/``): one unbroken stretch, chosen by its
camera-motion, sharpness and exposure measures and, once it has been trained
and shown to beat the hand-made selector, by the user's learned taste. It is
cut through the same path as a reel clip, so stabilization and the colour LUT
apply to it too.
"""

from __future__ import annotations

import os

MODES = ("reel", "segment", "both")

DEFAULTS = {
    "mode": "reel",
    "ask": False,
    "min_duration": 10.0,
    "max_duration": 30.0,   # 0 = up to the whole shot
    "pre_roll": 2.0,
    "post_roll": 2.0,
    "use_model": True,
}


def resolve_settings(raw) -> dict:
    """``raw`` (config or GUI dict, possibly partial) over the defaults, clamped."""
    s = dict(DEFAULTS)
    if isinstance(raw, dict):
        s.update({k: v for k, v in raw.items() if k in DEFAULTS and v is not None})
    if s["mode"] not in MODES:
        s["mode"] = "reel"
    s["min_duration"] = min(600.0, max(1.0, float(s["min_duration"])))
    s["max_duration"] = max(0.0, float(s["max_duration"]))
    if s["max_duration"] and s["max_duration"] < s["min_duration"]:
        s["max_duration"] = s["min_duration"]
    for key in ("pre_roll", "post_roll"):
        s[key] = min(30.0, max(0.0, float(s[key])))
    s["ask"] = bool(s["ask"])
    s["use_model"] = bool(s["use_model"])
    return s


def wants_reel(settings) -> bool:
    return settings["mode"] in ("reel", "both")


def wants_segment(settings) -> bool:
    return settings["mode"] in ("segment", "both")


def _model(use_model):
    """``(ranker or None, why)``: the user's trained taste, only when its
    evaluation showed it beats the selector — the rule DroneReel's CLI uses."""
    if not use_model:
        return None, "selector only (learned taste off)"
    from dronereel import ranker as rk
    try:
        model = rk.load(rk.DEFAULT_MODEL)
    except (ValueError, OSError, KeyError) as e:
        return None, f"selector only (saved model unusable: {e})"
    if model is None:
        return None, "selector only (no learned taste yet)"
    if not model.better:
        return None, "selector only (learned taste does not beat it yet)"
    return model, "with your learned taste"


def best_segment(video_path, settings, log=print):
    """``(start, end, description)`` of the best continuous segment of
    ``video_path``, in its own seconds. Raises when nothing can be chosen."""
    from dronereel import analyze as an
    from dronereel import selector as sel
    from dronereel.media import probe_media

    media = probe_media(video_path)
    cfg = sel.SelectConfig.from_dict({
        "min_duration": settings["min_duration"],
        "max_duration": settings["max_duration"] or None,
        "pre_roll": settings["pre_roll"],
        "post_roll": settings["post_roll"],
    })
    step = {"next": 0.1}

    def progress(fraction):
        if fraction >= step["next"]:
            log(f"   DroneReel analysis {int(fraction * 100)}%")
            step["next"] = int(fraction * 10) / 10 + 0.1

    analysis = an.analyze(media, progress=progress)
    model, why = _model(settings["use_model"])
    selection = sel.select(analysis, cfg, model)
    for warning in selection.warnings:
        log(f"   ⚠️ {warning}")
    choice = selection.best
    return choice.start, choice.end, f"score {choice.score:.2f}, {why}"


def segment_path(output_file, start, end) -> str:
    """``<highlight>_best_01m06s-01m24s.mp4`` beside the highlight."""
    root, ext = os.path.splitext(output_file)
    if root.endswith("_highlight"):
        root = root[: -len("_highlight")]

    def stamp(t):
        t = int(round(t))
        return f"{t // 60:02d}m{t % 60:02d}s"

    return f"{root}_best_{stamp(start)}-{stamp(end)}{ext or '.mp4'}"
