"""Where the 9:16 window looks, steered inside Gyroflow.

DroneReel's vertical planner (``dronereel.vertical.plan``) decides, ten times
a second, where a 9:16 window should sit to hold what stands out (spectral
residual saliency, a Viterbi path that does not hop, Gaussian smoothing and a
speed limit). Rather than cropping that window afterwards with ffmpeg, its
path is handed to Gyroflow as ``ZoomingCenterX`` keyframes, so the one
render stabilizes, undistorts and frames.

Gyroflow's unit, measured on DJI Osmo Nano footage rendered at 9:16 (ORB
feature matching between renders, ~400 inliers each): an offset of 1.0
moves the view 1.44 frame widths, horizontally only; ±0.05 moved it 7–8 %
and ±0.1 17–18 %. Gyroflow zooms in to keep the shifted view free of
borders — ×1.02–1.07 at ±0.05, ×1.2–1.27 at ±0.1 — and does not clamp: ±0.5
left the frame black. Hence ``max_shift``, in frame widths, which is also
the user's trade between following and zoom.
"""

from __future__ import annotations

import numpy as np

MODES = ("center", "follow", "fixed")
OFFSET_PER_WIDTH = 1 / 1.44   # Gyroflow ZoomingCenterX per frame width of view shift

DEFAULTS = {
    "mode": "center",
    "ask": False,
    "max_shift": 0.10,       # frame widths either way
    "smoothness": 1.5,       # s, Gaussian sigma of the path
    "max_speed": 0.12,       # frame widths per second
    "stickiness": 1.5,       # reluctance to move between equally good places
    "center_bias": 0.15,     # pull towards the centre, 0..1
    "keyframe_step": 0.5,    # s between keyframes handed to Gyroflow
}

_RANGES = {"max_shift": (0.0, 0.25), "smoothness": (0.0, 10.0), "max_speed": (0.01, 2.0),
           "stickiness": (0.0, 20.0), "center_bias": (0.0, 1.0), "keyframe_step": (0.1, 2.0)}


def resolve_settings(raw) -> dict:
    s = dict(DEFAULTS)
    if isinstance(raw, dict):
        s.update({k: v for k, v in raw.items() if k in DEFAULTS and v is not None})
    if s["mode"] not in MODES:
        s["mode"] = "center"
    for key, (lo, hi) in _RANGES.items():
        s[key] = round(min(hi, max(lo, float(s[key]))), 3)
    s["ask"] = bool(s["ask"])
    return s


def active(s) -> bool:
    return s["mode"] != "center" and s["max_shift"] > 0


def to_offsets(times, centers_widths, s):
    """``[(t, ZoomingCenterX)]`` every ``keyframe_step`` s from a path of the
    window centre in frame widths off the picture's centre."""
    times = np.asarray(times, float)
    d = np.clip(np.asarray(centers_widths, float), -s["max_shift"], s["max_shift"])
    if len(times) == 0:
        return []
    knots = np.arange(times[0], times[-1] + 1e-9, s["keyframe_step"])
    if knots[-1] < times[-1] - 1e-6:
        knots = np.append(knots, times[-1])
    values = np.interp(knots, times, d) * OFFSET_PER_WIDTH
    return [(float(t), round(float(v), 4)) for t, v in zip(knots, values)]


def center_curve(video, start, end, s):
    """``[(seconds from start, ZoomingCenterX)]`` for ``video``'s [start, end):
    DroneReel's crop path, as Gyroflow keyframes. Reads only."""
    from dronereel import vertical
    from dronereel.media import probe_media
    cfg = vertical.VerticalConfig(
        follow="smooth" if s["mode"] == "follow" else "fixed",
        smoothness=s["smoothness"], stickiness=s["stickiness"],
        max_speed=s["max_speed"], center_bias=s["center_bias"])
    path = vertical.plan(probe_media(video), start, end, cfg)
    centre = (np.asarray(path.x, float) + path.crop_w / 2 - path.src_w / 2) / path.crop_w
    return to_offsets(path.times, centre, s)
