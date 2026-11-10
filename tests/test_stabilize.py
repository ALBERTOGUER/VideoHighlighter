"""
Gyroflow stabilization of the kept clips (`modules.media.stabilize`).

The motion analysis runs on synthetic gyro (a known turn, shake or vibration)
so each rule is checked against motion whose answer is known. Gyroflow itself
is replaced by a small script that behaves like its CLI: it exits 0 whether or
not it rendered, so the file it leaves is the only verdict.
"""

from __future__ import annotations

import json
import os
import stat
import sys

import numpy as np
import pytest

from modules.media import stabilize as st

FS = 1000.0


def _quats(rate_fn, seconds=10.0):
    """Gyroflow-style {µs: [w, x, y, z]} integrating a per-sample angular
    velocity (deg/s, per axis) given by ``rate_fn(t)``."""
    n = int(seconds * FS)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    out = {}
    for i in range(n):
        t = i / FS
        w = np.radians(rate_fn(t)) / FS
        half = 0.5 * w
        dq = np.array([1.0, half[0], half[1], half[2]])
        dq /= np.linalg.norm(dq)
        a, b = q, dq
        q = np.array([
            a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
        ])
        out[str(int(t * 1e6) + 1000)] = list(q)
    return out


def _motion(rate_fn, seconds=10.0):
    return st.angular_velocity(_quats(rate_fn, seconds))


def turn(t):
    return np.array([0.0, 0.0, 90.0])


def shake(t):
    return np.array([60.0 * np.sin(2 * np.pi * 3 * t), 40.0 * np.cos(2 * np.pi * 2 * t), 0.0])


def still(t):
    return np.zeros(3)


def vibration(hz):
    return lambda t: np.array([0.0, 15.0 * np.sin(2 * np.pi * hz * t), 0.0])


# ── settings ────────────────────────────────────────────────────────────────

def test_resolve_settings_fills_defaults_and_clamps():
    s = st.resolve_settings({"smoothness": 3, "lpf_hz": 0, "mode": "bogus",
                             "output_size": "1x1", "smoothness_min": 0.8,
                             "smoothness_max": 0.2, "unknown": 1})
    assert s["smoothness"] == 1.0
    assert s["lpf_hz"] == 1.0
    assert s["mode"] == "manual"
    assert s["output_size"] == st.DEFAULTS["output_size"]
    assert (s["smoothness_min"], s["smoothness_max"]) == (0.2, 0.8)
    assert "unknown" not in s


def test_defaults_are_the_users_gyroflow_settings():
    s = st.resolve_settings(None)
    assert (s["output_size"], s["codec"], s["smoothness"]) == ("2160x3840", "H.265/HEVC", 0.30)
    assert s["enabled"] is False


# ── motion analysis ─────────────────────────────────────────────────────────

def test_angular_velocity_recovers_a_constant_turn():
    t, v = _motion(turn, 2)
    assert np.allclose(np.median(v, axis=0), [0, 0, 90], atol=0.5)
    assert t[0] > 0


def test_window_motion_separates_turn_from_shake():
    t, v = _motion(turn)
    pan, sh = st.window_motion(t, v, 2, 4)
    assert pan == pytest.approx(90, abs=1) and sh < 1
    t, v = _motion(shake)
    pan, sh = st.window_motion(t, v, 2, 4)
    assert pan < 5 and sh > 40


def test_clip_metrics_finds_the_vibration_frequency():
    t, v = _motion(vibration(33))
    m = st.clip_metrics(t, v, 2, 8)
    assert m["vib_hz"] == pytest.approx(33, abs=1)
    assert m["vibration"] > 5


def test_auto_clip_filters_vibration_clear_of_camera_motion():
    s = st.resolve_settings({"lpf_enabled": True})
    smooth, lpf, reason = st.auto_clip_values(
        {"pan": 0, "shake": 30, "vibration": 9, "vib_hz": 34}, s)
    assert lpf == pytest.approx(17)
    assert "low-pass 17 Hz" in reason


def test_auto_clip_does_not_filter_slow_vibration_but_smooths_more():
    s = st.resolve_settings({"lpf_enabled": True})
    base, _, _ = st.auto_clip_values({"pan": 0, "shake": 30, "vibration": 0, "vib_hz": 0}, s)
    smooth, lpf, reason = st.auto_clip_values(
        {"pan": 0, "shake": 30, "vibration": 50, "vib_hz": 10}, s)
    assert lpf is None
    assert smooth == pytest.approx(base + 0.1)
    assert "too close to camera motion" in reason


def test_auto_clip_never_filters_when_the_user_turned_it_off():
    s = st.resolve_settings({"lpf_enabled": False})
    _, lpf, _ = st.auto_clip_values({"pan": 0, "shake": 30, "vibration": 9, "vib_hz": 34}, s)
    assert lpf is None


def test_auto_clip_smoothness_follows_shake_within_the_range():
    s = st.resolve_settings({"smoothness_min": 0.1, "smoothness_max": 0.7})
    calm, _, _ = st.auto_clip_values({"pan": 0, "shake": 0, "vibration": 0, "vib_hz": 0}, s)
    rough, _, _ = st.auto_clip_values({"pan": 0, "shake": 500, "vibration": 0, "vib_hz": 0}, s)
    assert (calm, rough) == (0.1, 0.7)


def test_frame_curve_tracks_shake_and_backs_off_for_turns():
    s = st.resolve_settings({"smoothness_min": 0.1, "smoothness_max": 0.7,
                             "pan_response": 1.0, "keyframe_step": 0.5})
    t, v = _motion(lambda x: shake(x) if x < 5 else still(x))
    curve = st.frame_curve(t, v, 1, 9, s)
    assert curve[0][0] == 1 and curve[-1][0] == 9
    assert len(curve) == 17
    rough = [val for ts, val in curve if ts <= 3.5]
    calm = [val for ts, val in curve if ts >= 6.5]
    assert min(rough) > max(calm)
    assert max(calm) == pytest.approx(0.1, abs=0.02)

    t, v = _motion(lambda x: turn(x) + 0.5 * shake(x))
    turning = st.frame_curve(t, v, 2, 8, s)
    t, v = _motion(lambda x: 0.5 * shake(x))
    shaking = st.frame_curve(t, v, 2, 8, s)
    assert np.mean([v for _, v in turning]) < np.mean([v for _, v in shaking])


# ── preset and output parameters ────────────────────────────────────────────

def test_build_preset_keyframes_trim_and_filter():
    p = st.build_preset(0.3, lpf_hz=12, curve=[(431.0, 0.2), (431.5, 0.4)], trim=(0.1, 0.2))
    assert p["stabilization"]["smoothing_params"] == [{"name": "smoothness", "value": 0.3}]
    assert p["gyro_source"]["lpf"] == 12.0
    assert (p["trim_start"], p["trim_end"]) == (0.1, 0.2)
    kfs = p["keyframes"]["SmoothingParamSmoothness"]
    assert kfs == {"431000000": {"value": 0.2, "easing": "EaseInOut"},
                   "431500000": {"value": 0.4, "easing": "EaseInOut"}}


def test_build_preset_without_filter_turns_it_off():
    p = st.build_preset(0.3)
    assert p["gyro_source"]["lpf"] == 0.0
    assert "keyframes" not in p and "trim_start" not in p


def test_output_params_size_and_folder_url():
    s = st.resolve_settings({"output_size": "2160x3840"})
    p = st.output_params(s, "/a b/c", "x.mp4")
    assert (p["output_width"], p["output_height"]) == (2160, 3840)
    assert p["output_folder"] == "file:///a%20b/c/"
    assert p["codec"] == "H.265/HEVC" and p["audio"] is True


# ── running Gyroflow ────────────────────────────────────────────────────────

FAKE_GYROFLOW = r'''import json, os, sys, urllib.parse
args = sys.argv[1:]
log = os.environ.get("FAKE_GF_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps(args) + "\n")
if "--export-metadata" in args:
    out = args[args.index("--export-metadata") + 1].split(":", 1)[1]
    with open(os.environ["FAKE_GF_QUATS"]) as f:
        q = json.load(f)
    json.dump({"quaternions": q}, open(out, "w"))
elif "-p" in args and not os.environ.get("FAKE_GF_FAIL"):
    p = json.loads(args[args.index("-p") + 1])
    folder = urllib.parse.unquote(p["output_folder"][len("file://"):])
    open(os.path.join(folder, p["output_filename"]), "wb").write(b"video")
else:
    print("12:00:00 [ERROR] [x] Error: Pixel format is not supported")
sys.exit(0)
'''


@pytest.fixture
def fake_gyroflow(tmp_path, monkeypatch):
    # A shell wrapper: the interpreter's path may hold spaces, which a #! line can't.
    script = tmp_path / "fake_gyroflow.py"
    script.write_text(FAKE_GYROFLOW)
    exe = tmp_path / "gyroflow"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    quats = tmp_path / "quats.json"
    quats.write_text(json.dumps(_quats(lambda x: shake(x) if x < 5 else still(x))))
    monkeypatch.setenv("FAKE_GF_QUATS", str(quats))
    monkeypatch.setenv("FAKE_GF_LOG", str(tmp_path / "calls.log"))
    src = tmp_path / "src.MP4"
    src.write_bytes(b"source")
    return exe, src, tmp_path / "calls.log"


def test_manual_render_passes_the_users_values(fake_gyroflow):
    exe, src, calls = fake_gyroflow
    stab = st.ClipStabilizer({"mode": "manual", "smoothness": 0.3, "lpf_enabled": True,
                              "lpf_hz": 10}, str(src), 10.0, exe=str(exe), log=lambda *_: None)
    stab.prepare()
    try:
        out = stab.render(1, 2, 4)
        assert open(out, "rb").read() == b"video"
        args = json.loads(calls.read_text().splitlines()[-1])
        assert "--export-metadata" not in calls.read_text()  # manual needs no gyro pass
        params = json.loads(args[args.index("-p") + 1])
        assert (params["output_width"], params["output_height"]) == (2160, 3840)
    finally:
        stab.cleanup()


def test_frame_mode_writes_keyframes_on_the_source_timeline(fake_gyroflow, monkeypatch):
    exe, src, calls = fake_gyroflow
    seen = {}
    real_dump = st.json.dump

    def capture(obj, f, *a, **k):
        if isinstance(obj, dict) and "stabilization" in obj:
            seen.update(obj)
        return real_dump(obj, f, *a, **k)

    monkeypatch.setattr(st.json, "dump", capture)
    stab = st.ClipStabilizer({"mode": "auto_frame", "keyframe_step": 0.5}, str(src), 10.0,
                             offset=1.0, exe=str(exe), log=lambda *_: None)
    stab.prepare()
    try:
        stab.render(3, 1, 3)  # processed 1–3 s = source 2–4 s
    finally:
        stab.cleanup()
    kfs = seen["keyframes"]["SmoothingParamSmoothness"]
    assert min(int(k) for k in kfs) == 2_000_000 and max(int(k) for k in kfs) == 4_000_000
    assert (seen["trim_start"], seen["trim_end"]) == (0.2, 0.4)


def test_a_render_that_leaves_no_file_raises_with_gyroflows_error(fake_gyroflow, monkeypatch):
    exe, src, _ = fake_gyroflow
    monkeypatch.setenv("FAKE_GF_FAIL", "1")
    stab = st.ClipStabilizer({"mode": "manual"}, str(src), 10.0, exe=str(exe), log=lambda *_: None)
    stab.prepare()
    try:
        with pytest.raises(st.StabilizeError, match="Pixel format"):
            stab.render(1, 0, 2)
    finally:
        stab.cleanup()


def test_missing_gyroflow_is_a_stabilize_error(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "find_gyroflow", lambda: None)
    with pytest.raises(st.StabilizeError, match="not found"):
        st.ClipStabilizer({}, str(tmp_path / "x.mp4"), 10.0).prepare()


def test_app_store_build_gets_a_hard_link_in_its_own_folder(fake_gyroflow, tmp_path, monkeypatch):
    exe, src, _ = fake_gyroflow
    bundle = tmp_path / "Gyroflow.app" / "Contents"
    (bundle / "MacOS").mkdir(parents=True)
    (bundle / "_MASReceipt").mkdir()
    app_exe = bundle / "MacOS" / "gyroflow"
    os.link(exe, app_exe)
    sandbox = tmp_path / "sandbox"
    monkeypatch.setattr(st, "_sandbox_folder", lambda: str(sandbox))
    assert st.is_sandboxed(str(app_exe))
    stab = st.ClipStabilizer({"mode": "manual"}, str(src), 10.0, exe=str(app_exe),
                             log=lambda *_: None)
    stab.prepare()
    staged = stab.staged
    assert os.path.samefile(staged, src) and staged.startswith(str(sandbox))
    out = stab.render(1, 0, 2)
    assert out.startswith(str(sandbox))
    stab.cleanup()
    assert not os.path.exists(staged) and src.exists()


# ── panel ───────────────────────────────────────────────────────────────────

@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_panel_round_trips_every_setting(qapp):
    from modules.ui.stabilization_panel import StabilizationPanel
    wanted = st.resolve_settings({
        "enabled": True, "ask": True, "mode": "auto_frame", "output_size": "1080x1920",
        "codec": "ProRes", "bitrate": 220, "smoothness": 0.42, "lpf_enabled": True,
        "lpf_hz": 12.5, "smoothness_min": 0.2, "smoothness_max": 0.55,
        "pan_response": 0.7, "keyframe_step": 0.8})
    panel = StabilizationPanel(wanted)
    assert panel.settings() == wanted


def test_panel_shows_the_controls_of_the_chosen_mode(qapp):
    from modules.ui.stabilization_panel import StabilizationPanel
    panel = StabilizationPanel({"enabled": True, "mode": "manual"})
    panel.show()
    assert panel.smooth.isVisible() and not panel.smooth_min.isVisible()
    panel.mode_buttons["auto_frame"].setChecked(True)
    assert not panel.smooth.isVisible() and panel.smooth_min.isVisible() and panel.pan.isVisible()
    panel.mode_buttons["auto_clip"].setChecked(True)
    assert panel.smooth_min.isVisible() and not panel.pan.isVisible()


def test_the_reel_of_hevc_clips_is_tagged_hvc1(monkeypatch):
    import pipeline
    from modules.media import ffmpeg_tools
    monkeypatch.setattr(ffmpeg_tools, "probe", lambda p: {"streams": [
        {"codec_type": "audio", "codec_name": "aac"},
        {"codec_type": "video", "codec_name": "hevc" if p.endswith("h.mp4") else "h264"}]})
    assert pipeline._hvc1_tag("/c/h.mp4") == ["-tag:v", "hvc1"]
    assert pipeline._hvc1_tag("/c/a.mp4") == []
