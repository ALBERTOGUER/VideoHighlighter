"""
What a run delivers (`modules.media.delivery`) and the before-run dialog.

DroneReel's analysis needs the real cv2, which this suite's conftest replaces,
so the segment choice is checked here with DroneReel's analyze/select stood
in for; dronereel/tests covers the selector itself on synthetic videos.
"""

from __future__ import annotations

import os
import types

import pytest

from modules.media import delivery as dl


def test_resolve_settings_defaults_and_clamps():
    assert dl.resolve_settings(None) == dl.DEFAULTS
    s = dl.resolve_settings({"mode": "bogus", "min_duration": 20, "max_duration": 5,
                             "pre_roll": -1, "post_roll": 99, "extra": 1})
    assert s["mode"] == "reel"
    assert s["max_duration"] == 20          # never shorter than the minimum
    assert (s["pre_roll"], s["post_roll"]) == (0.0, 30.0)
    assert "extra" not in s


def test_no_limit_is_kept():
    assert dl.resolve_settings({"max_duration": 0})["max_duration"] == 0.0


@pytest.mark.parametrize("mode, reel, segment", [
    ("reel", True, False), ("segment", False, True), ("both", True, True)])
def test_what_each_mode_delivers(mode, reel, segment):
    s = dl.resolve_settings({"mode": mode})
    assert (dl.wants_reel(s), dl.wants_segment(s)) == (reel, segment)


def test_segment_path_sits_beside_the_highlight():
    assert dl.segment_path("/v/DJI_0032_highlight.mp4", 66.2, 84.7) == \
        "/v/DJI_0032_best_01m06s-01m25s.mp4"
    assert dl.segment_path("/v/out.mov", 0, 600) == "/v/out_best_00m00s-10m00s.mov"


# ── the learned taste ───────────────────────────────────────────────────────

@pytest.fixture
def ranker(monkeypatch):
    from dronereel import ranker as rk
    state = {"model": None}
    monkeypatch.setattr(rk, "load", lambda path=None: state["model"])
    return state


def test_taste_off_means_selector_only(ranker):
    ranker["model"] = types.SimpleNamespace(better=True)
    assert dl._model(False) == (None, "selector only (learned taste off)")


def test_no_model_yet(ranker):
    model, why = dl._model(True)
    assert model is None and "no learned taste yet" in why


def test_a_model_that_does_not_beat_the_selector_is_not_used(ranker):
    ranker["model"] = types.SimpleNamespace(better=False)
    model, why = dl._model(True)
    assert model is None and "does not beat" in why


def test_a_model_that_beats_the_selector_is_used(ranker):
    ranker["model"] = types.SimpleNamespace(better=True)
    model, why = dl._model(True)
    assert model is ranker["model"] and why == "with your learned taste"


# ── choosing the segment ────────────────────────────────────────────────────

def test_best_segment_passes_the_settings_to_dronereel(monkeypatch, ranker):
    from dronereel import analyze as an
    from dronereel import media as dm
    from dronereel import selector as sel
    seen = {}
    monkeypatch.setattr(dm, "probe_media", lambda p: types.SimpleNamespace(path=p))

    def fake_analyze(media, progress=None):
        for f in (0.05, 0.5, 1.0):
            progress(f)
        return "analysis"

    def fake_select(analysis, cfg, model):
        seen.update(cfg=cfg, model=model)
        best = types.SimpleNamespace(start=12.0, end=30.5, score=0.81)
        return types.SimpleNamespace(best=best, warnings=["short shot"])

    monkeypatch.setattr(an, "analyze", fake_analyze)
    monkeypatch.setattr(sel, "select", fake_select)
    lines = []
    s = dl.resolve_settings({"mode": "segment", "min_duration": 12, "max_duration": 0,
                             "pre_roll": 1.5, "post_roll": 3})
    start, end, why = dl.best_segment("/v/a.mp4", s, log=lines.append)

    assert (start, end) == (12.0, 30.5)
    assert why.startswith("score 0.81")
    cfg = seen["cfg"]
    assert (cfg.min_duration, cfg.max_duration, cfg.pre_roll, cfg.post_roll) == (12, None, 1.5, 3)
    assert seen["model"] is None
    assert any("50%" in line for line in lines) and any("short shot" in line for line in lines)


# ── panels and the before-run dialog ────────────────────────────────────────

@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_delivery_panel_round_trips(qapp):
    from modules.ui.delivery_panel import DeliveryPanel
    wanted = dl.resolve_settings({"mode": "both", "ask": True, "min_duration": 15,
                                  "max_duration": 0, "pre_roll": 3, "post_roll": 1,
                                  "use_model": False})
    assert DeliveryPanel(wanted).settings() == wanted


def test_segment_options_only_matter_when_a_segment_is_delivered(qapp):
    from modules.ui.delivery_panel import DeliveryPanel
    panel = DeliveryPanel({"mode": "reel"})
    assert not panel.min_spin.isEnabled()
    panel.mode_buttons["segment"].setChecked(True)
    assert panel.min_spin.isEnabled() and panel.model_chk.isEnabled()


def test_before_run_dialog_one_section_per_asked_panel(qapp):
    from modules.media.stabilize import resolve_settings as stab_settings
    from modules.ui.before_run_dialog import BeforeRunDialog
    from modules.ui.delivery_panel import DeliveryPanel
    from modules.ui.stabilization_panel import StabilizationPanel
    delivery = dl.resolve_settings({"mode": "segment", "ask": True})
    stab = stab_settings({"enabled": True, "ask": True, "mode": "auto_clip"})
    dialog = BeforeRunDialog([("delivery", "Delivery", DeliveryPanel, delivery),
                              ("stabilization", "Stabilization", StabilizationPanel, stab)])

    assert set(dialog.panels) == {"delivery", "stabilization"}
    dialog.panels["delivery"].mode_buttons["both"].setChecked(True)
    results = dialog.results()
    assert results["delivery"] == {**delivery, "mode": "both"}
    assert results["stabilization"] == stab

    dialog.dont_ask.setChecked(True)
    assert not any(r["ask"] for r in dialog.results().values())
