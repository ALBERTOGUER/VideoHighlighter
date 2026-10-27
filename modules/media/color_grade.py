"""Colour for the exported clips: the camera's LUT plus a grade, as one LUT.

The grade is DroneReel's (``dronereel.color``): exposure, contrast,
saturation, white balance (temperature/tint) and highlight protection, all
folded with the base LUT into a single 3D LUT that ``cut_video`` applies in
the same pass as before — identical on every frame, nothing generative.

``recommend`` measures a few frames of the video *after* the base LUT and
suggests values, each with the measurement behind it:

- white balance from near-neutral pixels only (low chroma, mid-tones), so a
  blue lake or a green forest is not "corrected" towards grey;
- exposure from the median luminance against a mid-grey target, held back
  so the brightest 1 % does not clip;
- contrast from the spread of luminance in stops;
- saturation from the mean chroma;
- highlight protection from how much already sits at white.

Suggestions are deliberately partial (they move part of the way) and
clamped: a starting point to adjust, not a verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np

from modules.media.lut import resolve_lut

# UI/config key -> dronereel.color.ColorSettings field, (min, max, neutral).
FIELDS = {
    "exposure": ("exposure_ev", -3.0, 3.0, 0.0),
    "contrast": ("contrast", 0.5, 2.0, 1.0),
    "saturation": ("saturation", 0.0, 2.0, 1.0),
    "temperature": ("wb_temperature", -100.0, 100.0, 0.0),
    "tint": ("wb_tint", -100.0, 100.0, 0.0),
    "highlights": ("highlight_protection", 0.0, 1.0, 0.0),
}

DEFAULTS = {"lut_path": "", "ask": False, **{k: v[3] for k, v in FIELDS.items()}}

TARGET_MEDIAN = 0.13      # linear luminance of a well-exposed median (≈ 0.43 display)
TARGET_SPREAD = 8.0       # stops between the 5th and 95th luminance percentiles
TARGET_CHROMA = 0.10      # mean |RGB − luma| in display space, lit areas
SHADOW_FLOOR = 2.0 ** -10  # linear: ~10 stops below white, where shadow detail ends
STRENGTH = 0.7            # how far a suggestion moves towards its target
WB_STRENGTH = 0.5         # white balance moves less: a cast can be the light itself
GAMMA = 2.4
LUMA = np.array([0.2126, 0.7152, 0.0722])


def resolve_settings(raw) -> dict:
    s = dict(DEFAULTS)
    if isinstance(raw, dict):
        s.update({k: v for k, v in raw.items() if k in DEFAULTS and v is not None})
    for key, (_, lo, hi, _) in FIELDS.items():
        s[key] = round(min(hi, max(lo, float(s[key]))), 3)
    s["lut_path"] = str(s["lut_path"] or "").strip()
    s["ask"] = bool(s["ask"])
    return s


def is_neutral(s) -> bool:
    return all(s[k] == v[3] for k, v in FIELDS.items())


def _dronereel_settings(s):
    from dronereel.color import ColorSettings
    return ColorSettings(**{field: s[key] for key, (field, *_rest) in FIELDS.items()})


def _cache_dir():
    from dronereel.color import LUT_CACHE_DIR
    return LUT_CACHE_DIR


def effective_lut(s, log=print):
    """The LUT ``cut_video`` should apply: the base LUT alone when the grade is
    neutral, otherwise base + grade composed into one cached ``.cube``."""
    base = resolve_lut(s["lut_path"], log=log) if s["lut_path"] else None
    if is_neutral(s):
        return base
    from dronereel.color import compose, read_cube, sha256_file, write_cube
    key = json.dumps({"base": sha256_file(base) if base else None,
                      **{k: s[k] for k in FIELDS}}, sort_keys=True)
    path = os.path.join(_cache_dir(), f"grade_{hashlib.sha256(key.encode()).hexdigest()[:16]}.cube")
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lut = compose(read_cube(base) if base else None, _dronereel_settings(s))
        write_cube(lut, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


# ── frames ──────────────────────────────────────────────────────────────────

def _duration(video):
    from modules.media.ffmpeg_tools import probe
    try:
        return float(probe(video)["format"]["duration"])
    except Exception:
        return 0.0


def grab_frame(video, at, width=480):
    """One frame as display RGB in 0..1, ``width`` wide, rotation applied."""
    from modules.system.app_paths import ffmpeg_exe
    out = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-ss", f"{max(0.0, at):.3f}", "-i", video,
         "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True)
    raw = out.stdout
    if out.returncode != 0 or not raw:
        raise RuntimeError(f"could not read a frame at {at:.1f}s")
    height = len(raw) // (width * 3)
    return np.frombuffer(raw[: width * height * 3], np.uint8).reshape(height, width, 3) / 255.0


def sample_times(video, count=6):
    d = _duration(video)
    if d <= 0:
        return [0.0]
    return [d * (i + 1) / (count + 1) for i in range(count)]


def _with_lut(rgb, lut_path):
    if not lut_path:
        return rgb
    from dronereel.color import apply_lut, read_cube
    return apply_lut(read_cube(lut_path), rgb)


# ── recommendation ──────────────────────────────────────────────────────────

def measure(rgb) -> dict:
    """Statistics of display RGB (``...x3``, 0..1) that the suggestions read."""
    px = rgb.reshape(-1, 3)
    lin = np.clip(px, 0, 1) ** GAMMA
    y = lin @ LUMA
    yd = px @ LUMA
    chroma = np.abs(px - yd[:, None]).mean(axis=1)
    neutral = (chroma < 0.06) & (yd > 0.15) & (yd < 0.85)
    if neutral.sum() >= 200:
        means = lin[neutral].mean(axis=0)
    else:
        means = None
    p5, p50, p95, p99 = np.percentile(np.maximum(y, 1e-5), [5, 50, 95, 99])
    # Deep shadow (dark water, a night sky) has neither measurable contrast nor
    # colour: the spread stops at the darkest tone a display separates, and
    # colour is judged where there is light to carry it.
    lit = yd > 0.1
    return {
        "median": float(p50),
        "p99": float(p99),
        "spread": float(np.log2(p95 / max(p5, SHADOW_FLOOR))),
        "chroma": float(chroma[lit].mean()) if lit.any() else float(chroma.mean()),
        "clipped": float((px.max(axis=1) >= 0.985).mean()),
        "neutral_rgb": None if means is None else [float(v) for v in means],
        "neutral_share": float(neutral.mean()),
    }


def suggest(m) -> tuple:
    """``(values, reasons)`` from :func:`measure`'s statistics."""
    from dronereel.color import TEMPERATURE_STOPS, TINT_STOPS
    v, why = {}, []

    ev = STRENGTH * np.log2(TARGET_MEDIAN / max(m["median"], 1e-4))
    headroom = np.log2(1.0 / max(m["p99"], 1e-4))       # stops before the top 1 % clips
    ev = float(np.clip(min(ev, headroom + 0.3), -1.5, 1.5))
    v["exposure"] = round(ev, 2)
    why.append(f"exposure {ev:+.2f} EV: median luminance {m['median']:.3f} "
               f"(target {TARGET_MEDIAN})")

    contrast = float(np.clip((TARGET_SPREAD / max(m["spread"], 0.5)) ** STRENGTH, 0.85, 1.3))
    v["contrast"] = round(contrast, 2)
    why.append(f"contrast {contrast:.2f}: {m['spread']:.1f} stops between shadows and "
               f"highlights (target {TARGET_SPREAD:g})")

    sat = float(np.clip((TARGET_CHROMA / max(m["chroma"], 1e-3)) ** STRENGTH, 0.85, 1.25))
    v["saturation"] = round(sat, 2)
    why.append(f"saturation {sat:.2f}: mean colour {m['chroma']:.3f} (target {TARGET_CHROMA})")

    if m["neutral_rgb"]:
        r, g, b = m["neutral_rgb"]
        temp = WB_STRENGTH * 100 * np.log2(b / r) / (2 * TEMPERATURE_STOPS)
        tint = -WB_STRENGTH * 100 * np.log2(np.sqrt(r * b) / g) / TINT_STOPS
        v["temperature"] = round(float(np.clip(temp, -40, 40)), 1)
        v["tint"] = round(float(np.clip(tint, -30, 30)), 1)
        why.append(f"white balance {v['temperature']:+.0f} / tint {v['tint']:+.0f}: "
                   f"near-neutral areas ({m['neutral_share'] * 100:.0f}% of the picture) "
                   f"read R {r:.3f} G {g:.3f} B {b:.3f}")
    else:
        v["temperature"], v["tint"] = 0.0, 0.0
        why.append("white balance unchanged: too few near-neutral areas to judge it "
                   "(sky, water and foliage are not grey)")

    hl = float(np.clip(m["clipped"] * 25, 0.0, 0.6))
    v["highlights"] = round(hl, 2)
    why.append(f"highlight protection {hl:.2f}: {m['clipped'] * 100:.1f}% of the picture at white")
    return v, why


def recommend(video, s, times=None) -> tuple:
    """``(settings, reasons)``: ``s`` with suggested grade values, measured on
    frames of ``video`` after its base LUT."""
    base = resolve_lut(s["lut_path"], log=lambda *_: None) if s["lut_path"] else None
    frames = [_with_lut(grab_frame(video, t), base) for t in (times or sample_times(video))]
    values, reasons = suggest(measure(np.concatenate([f.reshape(-1, 3) for f in frames])))
    return resolve_settings({**s, **values}), reasons


def preview(video, s, at=None, width=480) -> np.ndarray:
    """Side by side, uint8: the frame with the base LUT only | with the grade."""
    if at is None:
        times = sample_times(video, 1)
        at = times[0]
    base = resolve_lut(s["lut_path"], log=lambda *_: None) if s["lut_path"] else None
    frame = grab_frame(video, at, width)
    before = _with_lut(frame, base)
    graded = _with_lut(frame, effective_lut(s, log=lambda *_: None))
    gap = np.ones((before.shape[0], 8, 3))
    return (np.concatenate([before, gap, graded], axis=1) * 255).round().astype(np.uint8)


# ── which LUT the footage needs ─────────────────────────────────────────────

def _user_profiles_dir():
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
                        "dronereel", "profiles")


def detect(video) -> dict:
    """DroneReel's verdict on ``video``: ``{"camera", "profile", "lut", "message"}``.

    Strict as DroneReel is: a LUT only for a profile confirmed by a verified
    rule (e.g. the Osmo Nano's own colour-mode metadata), ``lut`` None for a
    non-log profile, and a message naming what is missing otherwise."""
    from dronereel import profiles
    from dronereel.media import probe_media
    cameras = profiles.load_cameras(_user_profiles_dir())
    decision = profiles.decide(probe_media(video), cameras)
    out = {"camera": decision.camera, "profile": decision.profile, "lut": None,
           "log": False, "message": ""}
    if not decision.confirmed:
        hint = f" ({'; '.join(decision.hints)})" if decision.hints else ""
        out["message"] = f"Camera or colour profile not confirmed — colours kept as recorded{hint}."
        return out
    cam = cameras[decision.camera]
    prof = cam.profile(decision.profile)
    if not prof.is_log:
        out["message"] = f"{cam.name}, {prof.name}: already Rec.709, no LUT needed."
        return out
    out["log"] = True
    source = prof.settings.get("lut")
    if source:
        source = os.path.expanduser(str(source))
        if not os.path.isabs(source):
            source = os.path.join(os.path.dirname(cam.source), source)
    if source and os.path.isfile(source):
        out["lut"] = os.path.abspath(source)
        out["message"] = f"{cam.name}, {prof.name} → {os.path.basename(source)}."
    else:
        out["message"] = (f"{cam.name}, {prof.name}: needs DJI's official LUT. Choose it "
                          f"with Browse, or set 'lut:' in {_user_profiles_dir()}/"
                          f"{cam.camera}.yaml so it is found next time.")
    return out
