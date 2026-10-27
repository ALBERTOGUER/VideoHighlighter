"""
Where the 9:16 window looks (`modules.media.framing`): DroneReel's crop path
turned into Gyroflow ZoomingCenterX keyframes.

DroneReel's planner needs the real cv2, which this suite's conftest replaces;
dronereel/tests/test_vertical.py covers it on synthetic video. Here the plan
is stood in for, and the conversion — units, clamping, timing — is checked.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from modules.media import framing as fr
from modules.media import stabilize as st


def test_resolve_settings_defaults_and_clamps():
    assert fr.resolve_settings(None) == fr.DEFAULTS
    s = fr.resolve_settings({"mode": "bogus", "max_shift": 3, "max_speed": 0, "x": 1})
    assert s["mode"] == "center" and s["max_shift"] == 0.25 and s["max_speed"] == 0.01
    assert "x" not in s


@pytest.mark.parametrize("raw, on", [
    ({"mode": "center"}, False), ({"mode": "follow"}, True),
    ({"mode": "fixed"}, True), ({"mode": "follow", "max_shift": 0}, False)])
def test_framing_is_active_only_when_it_can_move(raw, on):
    assert fr.active(fr.resolve_settings(raw)) is on


def test_offsets_are_keyframed_clamped_and_in_gyroflows_unit():
    s = fr.resolve_settings({"mode": "follow", "max_shift": 0.1, "keyframe_step": 0.5})
    times = np.arange(0, 2.01, 0.1)
    centre = np.linspace(-0.3, 0.05, len(times))           # starts beyond the limit
    kf = fr.to_offsets(times, centre, s)

    assert [t for t, _ in kf] == pytest.approx([0, 0.5, 1.0, 1.5, 2.0])
    assert kf[0][1] == pytest.approx(-0.1 * fr.OFFSET_PER_WIDTH, abs=1e-4)
    assert kf[-1][1] == pytest.approx(0.05 * fr.OFFSET_PER_WIDTH, abs=1e-4)
    assert all(abs(v) <= 0.1 * fr.OFFSET_PER_WIDTH + 1e-6 for _, v in kf)


def test_the_last_keyframe_lands_on_the_end():
    s = fr.resolve_settings({"keyframe_step": 0.5})
    kf = fr.to_offsets([0.0, 1.2], [0.0, 0.0], s)
    assert [t for t, _ in kf] == pytest.approx([0.0, 0.5, 1.0, 1.2])


def test_one_unit_moves_the_view_1_44_widths():
    # The calibration this module rests on: ±0.05 moved the view 7–8 % of the width.
    assert 0.05 / fr.OFFSET_PER_WIDTH == pytest.approx(0.072, abs=0.005)


@pytest.mark.parametrize("mode, follow", [("follow", "smooth"), ("fixed", "fixed")])
def test_center_curve_converts_dronereels_crop_path(monkeypatch, mode, follow):
    from dronereel import media as dm
    from dronereel import vertical
    seen = {}

    def fake_plan(media, start, end, cfg):
        seen.update(start=start, end=end, follow=cfg.follow)
        # 2880-wide picture, 2160 window: centred at x=360, fully right at x=720.
        return types.SimpleNamespace(times=[0.0, 1.0, 2.0], x=[360.0, 540.0, 720.0],
                                     crop_w=2160, src_w=2880)

    monkeypatch.setattr(dm, "probe_media", lambda p: types.SimpleNamespace(path=p))
    monkeypatch.setattr(vertical, "plan", fake_plan)
    s = fr.resolve_settings({"mode": mode, "max_shift": 0.25, "keyframe_step": 1.0})
    kf = fr.center_curve("/v/a.mp4", 10, 12, s)

    assert (seen["start"], seen["end"], seen["follow"]) == (10, 12, follow)
    assert [t for t, _ in kf] == [0.0, 1.0, 2.0]
    widths = [v / fr.OFFSET_PER_WIDTH for _, v in kf]
    assert widths == pytest.approx([0.0, 180 / 2160, 360 / 2160], abs=1e-3)


def test_preset_carries_centre_and_smoothness_keyframes():
    p = st.build_preset(0.3, curve=[(5.0, 0.4)], centre=[(5.0, 0.02), (5.5, -0.01)])
    kfs = p["keyframes"]
    assert kfs["SmoothingParamSmoothness"] == {"5000000": {"value": 0.4, "easing": "EaseInOut"}}
    assert kfs["ZoomingCenterX"] == {"5000000": {"value": 0.02, "easing": "EaseInOut"},
                                     "5500000": {"value": -0.01, "easing": "EaseInOut"}}


def test_render_puts_centre_keyframes_on_the_source_timeline(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(st, "_remove", lambda path: None)

    def fake_run(self, args, expect):
        import json
        seen.update(json.load(open(args[1])))
        open(expect, "wb").write(b"video")

    monkeypatch.setattr(st.ClipStabilizer, "_run", fake_run)
    stab = st.ClipStabilizer({"mode": "manual"}, str(tmp_path / "src.mp4"), 100.0,
                             offset=20.0, exe="/bin/true", log=lambda *_: None)
    stab.workdir, stab.staged = str(tmp_path), str(tmp_path / "src.mp4")
    stab.render(1, 10, 12, centre=[(0.0, 0.01), (2.0, -0.02)])  # processed 10–12 = source 30–32
    assert set(seen["keyframes"]["ZoomingCenterX"]) == {"30000000", "32000000"}


# ── panel ───────────────────────────────────────────────────────────────────

@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_framing_panel_round_trips(qapp):
    from modules.ui.framing_panel import FramingPanel
    wanted = fr.resolve_settings({"mode": "follow", "ask": True, "max_shift": 0.12,
                                  "max_speed": 0.2, "smoothness": 2.0, "stickiness": 3.0,
                                  "center_bias": 0.3, "keyframe_step": 0.4})
    assert FramingPanel(wanted).settings() == wanted


def test_framing_panel_enables_what_the_mode_uses(qapp):
    from modules.ui.framing_panel import FramingPanel
    panel = FramingPanel({"mode": "center"})
    assert not panel.max_shift.isEnabled()
    panel.mode_buttons["fixed"].setChecked(True)
    assert panel.max_shift.isEnabled() and not panel.speed.isEnabled()
    panel.mode_buttons["follow"].setChecked(True)
    assert panel.speed.isEnabled()
