"""Gyro stabilization of the kept clips through Gyroflow's command line.

Cameras that log their gyroscope (DJI, GoPro, Insta360, Sony…) can be
stabilized after the fact far better than any image-based method, and Gyroflow
(GPL-3.0, run as a separate program — nothing of it is linked here) does it.
This module drives it for the segments a run keeps, before ``cut_video`` grades
and encodes them.

Why it renders segments of the *original* file rather than cut clips: the gyro
lives in a data track ffmpeg cannot copy, so any cut made first loses it. Each
segment is rendered from the whole source with Gyroflow's own trim, which also
gives the smoothing the motion on both sides of the cut.

Three modes, all reading the same gyro the camera wrote:

- ``manual``     — the user's smoothness and low-pass filter for every clip.
- ``auto_clip``  — per clip: smoothness from how much it shakes, and a low-pass
                   filter only when there is high-frequency vibration that sits
                   clear of real camera motion.
- ``auto_frame`` — ``auto_clip``'s filter, plus smoothness keyframed every
                   ``keyframe_step`` seconds: up where the camera shakes, down
                   where it is still (less crop) or turning on purpose (so the
                   frame follows the turn instead of lagging behind it).

The low-pass filter cannot be keyframed in Gyroflow; it is one value per clip.

The Mac App Store build of Gyroflow is sandboxed: from the command line it can
only read and write inside its own Application Support folder. The source is
hard-linked in there (no copy, no extra space) and the output written there.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.parse

import numpy as np

MODES = ("manual", "auto_clip", "auto_frame")

# (value, label) — width x height of Gyroflow's output.
OUTPUT_SIZES = (
    ("2160x3840", "9:16 4K (2160×3840)"),
    ("1080x1920", "9:16 1080p (1080×1920)"),
    ("3840x2160", "16:9 4K (3840×2160)"),
    ("1920x1080", "16:9 1080p (1920×1080)"),
)
CODECS = ("H.265/HEVC", "H.264/AVC", "ProRes")

DEFAULTS = {
    "enabled": False,
    "ask": False,
    "mode": "manual",
    "output_size": "2160x3840",
    "codec": "H.265/HEVC",
    "bitrate": 100,          # Mbit/s — an intermediate: cut_video encodes the final clip
    "use_gpu": False,        # Gyroflow's VideoToolbox path rejects 10-bit on Intel Macs
    "smoothness": 0.30,
    "lpf_enabled": False,
    "lpf_hz": 10.0,
    "smoothness_min": 0.15,
    "smoothness_max": 0.60,
    "pan_response": 0.5,
    "keyframe_step": 0.5,
}

# Shake (oscillation around the camera's path, deg/s) mapped onto the
# smoothness range: at or below LOW the minimum, at or above HIGH the maximum.
# Fitted on handheld/FPV action-camera footage; calm shots sit near 10, a
# rough ride above 100.
SHAKE_LOW = 10.0
SHAKE_HIGH = 110.0
# A vibration peak below this is too close to real camera motion to filter
# without smearing the motion itself — smoothness takes it instead.
LPF_MIN_SOURCE_HZ = 20.0
LPF_RANGE_HZ = (8.0, 30.0)
# Gyroflow 1.6.x knows NoEasing/EaseIn/EaseOut/EaseInOut; an unknown easing
# makes it drop every keyframe without a word.
KEYFRAME_EASING = "EaseInOut"


class StabilizeError(RuntimeError):
    pass


def resolve_settings(raw) -> dict:
    """``raw`` (config or GUI dict, possibly partial) over the defaults, clamped."""
    s = dict(DEFAULTS)
    if isinstance(raw, dict):
        s.update({k: v for k, v in raw.items() if k in DEFAULTS and v is not None})
    if s["mode"] not in MODES:
        s["mode"] = "manual"
    if s["output_size"] not in dict(OUTPUT_SIZES):
        s["output_size"] = DEFAULTS["output_size"]
    if s["codec"] not in CODECS:
        s["codec"] = DEFAULTS["codec"]
    for key in ("smoothness", "smoothness_min", "smoothness_max", "pan_response"):
        s[key] = min(1.0, max(0.0, float(s[key])))
    if s["smoothness_min"] > s["smoothness_max"]:
        s["smoothness_min"], s["smoothness_max"] = s["smoothness_max"], s["smoothness_min"]
    s["lpf_hz"] = min(100.0, max(1.0, float(s["lpf_hz"])))
    s["keyframe_step"] = min(5.0, max(0.1, float(s["keyframe_step"])))
    s["bitrate"] = int(min(800, max(5, int(s["bitrate"]))))
    for key in ("enabled", "ask", "lpf_enabled", "use_gpu"):
        s[key] = bool(s[key])
    return s


# ── Motion analysis ─────────────────────────────────────────────────────────

def angular_velocity(quaternions: dict):
    """``(t, v)``: sample times in seconds and per-axis angular velocity in
    deg/s, from Gyroflow's parsed ``quaternions`` ({µs: [w, x, y, z]})."""
    ts = np.array(sorted(int(k) for k in quaternions), dtype=np.int64)
    if len(ts) < 3:
        raise StabilizeError("the file carries no usable gyro data")
    q = np.array([quaternions[str(t)] for t in ts], dtype=float)
    a, b = q[:-1] * np.array([1, -1, -1, -1]), q[1:]
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    rel = np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)
    rel[rel[:, 0] < 0] *= -1
    dt = np.diff(ts) / 1e6
    dt[dt <= 0] = np.median(dt)
    v = np.degrees(2 * rel[:, 1:] / dt[:, None])
    return ts[1:] / 1e6, v


def window_motion(t, v, start, end):
    """``(pan, shake)`` in deg/s over [start, end): the sustained turn (the
    mean rotation, which the frame should follow) and the oscillation around
    it (which stabilization should remove)."""
    m = (t >= start) & (t < end)
    if m.sum() < 2:
        return 0.0, 0.0
    x = v[m]
    return float(np.linalg.norm(x.mean(axis=0))), float(np.linalg.norm(x.std(axis=0)))


def clip_metrics(t, v, start, end) -> dict:
    """Shake and vibration of one clip: ``pan``/``shake`` (deg/s), the
    ``vibration`` above 10 Hz (deg/s RMS) and its peak frequency ``vib_hz``."""
    pan, shake = window_motion(t, v, start, end)
    m = (t >= start) & (t < end)
    x = v[m]
    vibration, vib_hz = 0.0, 0.0
    if len(x) > 32:
        fs = 1.0 / np.median(np.diff(t[m]))
        freqs = np.fft.rfftfreq(len(x), 1 / fs)
        power = (np.abs(np.fft.rfft(x - x.mean(axis=0), axis=0)) ** 2).sum(axis=1)
        band = (freqs >= 10) & (freqs < min(100, fs / 2))
        if band.any():
            vibration = float(np.sqrt(power[band].sum() / len(x) ** 2 * 2))
            vib_hz = float(freqs[band][np.argmax(power[band])])
    return {"pan": pan, "shake": shake, "vibration": vibration, "vib_hz": vib_hz}


def _shake_score(shake):
    return float(np.clip((shake - SHAKE_LOW) / (SHAKE_HIGH - SHAKE_LOW), 0.0, 1.0))


def auto_clip_values(metrics: dict, settings: dict):
    """``(smoothness, lpf_hz or None, reason)`` for one clip."""
    lo, hi = settings["smoothness_min"], settings["smoothness_max"]
    smooth = lo + (hi - lo) * _shake_score(metrics["shake"])
    lpf = None
    reason = f"shake {metrics['shake']:.0f}°/s"
    if settings["lpf_enabled"] and metrics["vibration"] > 1.0:
        if metrics["vib_hz"] >= LPF_MIN_SOURCE_HZ:
            lpf = float(np.clip(metrics["vib_hz"] / 2, *LPF_RANGE_HZ))
            reason += f", vibration at {metrics['vib_hz']:.0f} Hz → low-pass {lpf:.0f} Hz"
        else:
            smooth = min(hi, smooth + 0.1)
            reason += (f", vibration at {metrics['vib_hz']:.0f} Hz is too close to camera "
                       "motion to filter → more smoothness instead")
    return round(smooth, 3), lpf, reason


def frame_curve(t, v, start, end, settings):
    """``[(time_s, smoothness)]`` every ``keyframe_step`` seconds across the clip."""
    lo, hi = settings["smoothness_min"], settings["smoothness_max"]
    step = settings["keyframe_step"]
    times = list(np.arange(start, end, step)) + [end]
    scores = []
    for c in times:
        pan, shake = window_motion(t, v, c - step, c + step)
        score = _shake_score(shake)
        # A turn the operator meant: lower the smoothing so the frame follows it.
        score -= settings["pan_response"] * float(np.clip(pan / (shake + 1e-9) - 0.5, 0, 1))
        scores.append(float(np.clip(score, 0, 1)))
    if len(scores) >= 3:  # no jumps from one window to the next
        padded = [scores[0]] + scores + [scores[-1]]
        scores = [sum(padded[i:i + 3]) / 3 for i in range(len(scores))]
    return [(float(c), round(lo + (hi - lo) * s, 3)) for c, s in zip(times, scores)]


# ── Gyroflow presets and output parameters ──────────────────────────────────

def build_preset(smoothness, lpf_hz=None, curve=None, trim=None, centre=None) -> dict:
    """A Gyroflow preset. ``curve`` = [(source time s, smoothness)] keyframes,
    ``centre`` = [(source time s, ZoomingCenterX)] keyframes (where the
    output window looks, see ``modules.media.framing``), ``trim`` = (start,
    end) as fractions of the source duration."""
    preset = {
        "version": 2,
        "stabilization": {
            "method": "Default",
            "smoothing_params": [{"name": "smoothness", "value": float(smoothness)}],
        },
        "gyro_source": {"lpf": float(lpf_hz) if lpf_hz else 0.0},
    }
    keyframes = {}
    for name, points in (("SmoothingParamSmoothness", curve), ("ZoomingCenterX", centre)):
        if points:
            keyframes[name] = {
                str(int(round(ts * 1_000_000))): {"value": float(val), "easing": KEYFRAME_EASING}
                for ts, val in points}
    if keyframes:
        preset["keyframes"] = keyframes
    if trim:
        preset["trim_start"], preset["trim_end"] = float(trim[0]), float(trim[1])
    return preset


def output_params(settings, folder, filename) -> dict:
    width, height = (int(x) for x in settings["output_size"].split("x"))
    return {
        "codec": settings["codec"],
        "bitrate": settings["bitrate"],
        "use_gpu": settings["use_gpu"],
        "audio": True,
        "output_width": width,
        "output_height": height,
        "output_folder": "file://" + urllib.parse.quote(os.path.join(folder, "")),
        "output_filename": filename,
    }


# ── Running Gyroflow ────────────────────────────────────────────────────────

def find_gyroflow():
    """Path of the Gyroflow executable, or None. ``VH_GYROFLOW`` overrides."""
    env = os.environ.get("VH_GYROFLOW")
    candidates = [env] if env else []
    for root in ("/Applications", os.path.expanduser("~/Applications")):
        candidates.append(os.path.join(root, "Gyroflow.app", "Contents", "MacOS", "gyroflow"))
    candidates += [shutil.which("gyroflow"), shutil.which("Gyroflow")]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def is_sandboxed(exe) -> bool:
    """The Mac App Store build: it ships a receipt and runs sandboxed."""
    if not exe or ".app/" not in exe:
        return False
    bundle = exe.split(".app/")[0] + ".app"
    return os.path.exists(os.path.join(bundle, "Contents", "_MASReceipt"))


def _sandbox_folder():
    return os.path.expanduser("~/Library/Application Support/Gyroflow/VideoHighlighter")


class ClipStabilizer:
    """Stabilizes segments of one source video. ``prepare()`` once, then
    ``render()`` per segment; every failure raises ``StabilizeError`` so the
    caller can fall back to the unstabilized cut."""

    def __init__(self, settings, source, source_duration, offset=0.0, log=print, exe=None):
        self.s = resolve_settings(settings)
        self.source = source
        self.duration = float(source_duration)
        self.offset = float(offset)  # where the processed video starts in the source
        self.log = log
        self.exe = exe or find_gyroflow()
        self.workdir = None
        self.staged = None
        self._linked = False
        self.t = self.v = None

    def prepare(self):
        if not self.exe:
            raise StabilizeError("Gyroflow was not found (install it, or set VH_GYROFLOW)")
        if self.duration <= 0:
            raise StabilizeError("unknown source duration")
        if is_sandboxed(self.exe):
            self.workdir = _sandbox_folder()
            os.makedirs(self.workdir, exist_ok=True)
            self.staged = os.path.join(self.workdir, "source" + os.path.splitext(self.source)[1])
            if os.path.lexists(self.staged):
                os.remove(self.staged)
            try:
                os.link(self.source, self.staged)
                self._linked = True
            except OSError as e:
                raise StabilizeError(
                    "the App Store build of Gyroflow can only read files inside its own "
                    f"folder, and the video could not be linked there ({e}). Keep the "
                    "footage on the startup disk, or install Gyroflow from gyroflow.xyz") from e
        else:
            self.workdir = tempfile.mkdtemp(prefix="vh_gyroflow_")
            self.staged = self.source
        if self.s["mode"] != "manual":
            motion = os.path.join(self.workdir, "motion.json")
            self._run([self.staged, "-f", "--export-metadata", f"2:{motion}"], expect=motion)
            try:
                with open(motion, encoding="utf-8") as f:
                    quats = json.load(f).get("quaternions") or {}
            finally:
                _remove(motion)
            self.t, self.v = angular_velocity(quats)

    def plan(self, start, end):
        """``(smoothness, lpf_hz, curve, reason)`` for source seconds [start, end)."""
        s = self.s
        if s["mode"] == "manual":
            lpf = s["lpf_hz"] if s["lpf_enabled"] else None
            return s["smoothness"], lpf, None, "manual"
        metrics = clip_metrics(self.t, self.v, start, end)
        smooth, lpf, reason = auto_clip_values(metrics, s)
        curve = frame_curve(self.t, self.v, start, end, s) if s["mode"] == "auto_frame" else None
        return smooth, lpf, curve, reason

    def render(self, index, start, end, centre=None):
        """Stabilized file for processed-video seconds [start, end).
        ``centre`` = [(seconds from start, ZoomingCenterX)] steers the window."""
        a, b = start + self.offset, end + self.offset
        smooth, lpf, curve, reason = self.plan(a, b)
        detail = (f"smoothness {min(v for _, v in curve):.2f}–{max(v for _, v in curve):.2f} "
                  f"over {len(curve)} keyframes" if curve else f"smoothness {smooth:.2f}")
        self.log(f"  🎥 Stabilizing clip {index}: {detail}"
                 f"{f', low-pass {lpf:.0f} Hz' if lpf else ''} ({reason})")
        preset_path = os.path.join(self.workdir, f"clip_{index:02d}.gyroflow")
        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump(build_preset(smooth, lpf, curve,
                                   (a / self.duration, min(1.0, b / self.duration)),
                                   centre=[(a + t, v) for t, v in centre] if centre else None), f)
        ext = ".mov" if self.s["codec"] == "ProRes" else ".mp4"
        name = f"clip_{index:02d}_stabilized{ext}"
        out = os.path.join(self.workdir, name)
        _remove(out)
        try:
            self._run([self.staged, preset_path, "-f", "-p",
                       json.dumps(output_params(self.s, self.workdir, name))], expect=out)
        finally:
            _remove(preset_path)
        return out

    def _run(self, args, expect):
        try:
            r = subprocess.run([self.exe] + args, capture_output=True, text=True,
                               timeout=6 * 3600)
        except subprocess.TimeoutExpired as e:
            raise StabilizeError("Gyroflow timed out") from e
        # Gyroflow exits 0 even when a render fails; the file is the verdict.
        if os.path.isfile(expect) and os.path.getsize(expect) > 0:
            return
        errors = [ln.split("Error:", 1)[-1].strip() for ln in (r.stdout + r.stderr).splitlines()
                  if "[ERROR]" in ln and "constant_bit_rate" not in ln]
        raise StabilizeError(errors[-1] if errors else f"Gyroflow produced nothing (rc={r.returncode})")

    def cleanup(self):
        if self._linked and self.staged:
            _remove(self.staged)  # the link only; the source keeps its other name
        if self.workdir and self.workdir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.workdir, ignore_errors=True)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
