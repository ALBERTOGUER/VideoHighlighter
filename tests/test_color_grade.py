"""
Colour for the exported clips (`modules.media.color_grade`): the LUT plus a
grade composed into one LUT, and the recommendation.

The recommendation is checked on synthetic pictures whose answer is known — an
underexposed grey, a blue cast on neutral surfaces, a scene with no neutrals at
all, a frame that is half deep water — and the white balance it suggests is
fed back through DroneReel's own adjustment to show the cast actually shrinks.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from modules.media import color_grade as cg

IDENTITY_CUBE = "LUT_3D_SIZE 2\n" + "".join(
    f"{r} {g} {b}\n" for b in (0, 1) for g in (0, 1) for r in (0, 1))


@pytest.fixture(autouse=True)
def lut_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "_cache_dir", lambda: str(tmp_path / "luts"))
    return tmp_path / "luts"


@pytest.fixture
def identity_lut(tmp_path):
    path = tmp_path / "base.cube"
    path.write_text(IDENTITY_CUBE)
    return str(path)


def _image(rgb, size=(60, 80)):
    return np.broadcast_to(np.asarray(rgb, float), (*size, 3)).copy()


# ── settings and the composed LUT ───────────────────────────────────────────

def test_resolve_settings_clamps_and_starts_neutral():
    assert cg.is_neutral(cg.resolve_settings(None))
    s = cg.resolve_settings({"exposure": 9, "saturation": -1, "temperature": 500, "x": 1})
    assert (s["exposure"], s["saturation"], s["temperature"]) == (3.0, 0.0, 100.0)
    assert "x" not in s


def test_a_neutral_grade_is_the_lut_itself(identity_lut):
    assert cg.effective_lut(cg.resolve_settings({"lut_path": identity_lut})) == identity_lut
    assert cg.effective_lut(cg.resolve_settings({})) is None


def test_a_grade_is_composed_once_and_reused(identity_lut, lut_cache):
    from dronereel.color import read_cube
    s = cg.resolve_settings({"lut_path": identity_lut, "exposure": 1.0})
    path = cg.effective_lut(s)
    assert path.startswith(str(lut_cache)) and path.endswith(".cube")
    assert cg.effective_lut(s) == path                       # cached, same file
    mid = read_cube(path).table[32, 32, 32]                  # 65-point grid: 0.5 grey
    assert mid[0] == pytest.approx(min(1.0, (0.5 ** 2.4 * 2) ** (1 / 2.4)), abs=0.02)
    other = cg.effective_lut(cg.resolve_settings({"lut_path": identity_lut, "exposure": 0.5}))
    assert other != path


def test_a_grade_without_a_camera_lut_still_applies(lut_cache):
    assert cg.effective_lut(cg.resolve_settings({"saturation": 1.2})).endswith(".cube")


# ── measuring and suggesting ────────────────────────────────────────────────

def test_an_underexposed_grey_gets_brighter():
    v, why = cg.suggest(cg.measure(_image([0.2, 0.2, 0.2])))
    assert v["exposure"] > 0.3
    assert "exposure" in why[0]


def test_a_bright_frame_is_not_pushed_into_clipping():
    img = _image([0.3, 0.3, 0.3])
    img[:6] = 0.97                                             # 10 % near white
    v, _ = cg.suggest(cg.measure(img))
    assert v["exposure"] <= np.log2(1 / 0.97 ** 2.4) + 0.3 + 0.005   # rounded to 0.01


def test_a_blue_cast_on_neutral_surfaces_is_warmed_and_shrinks():
    from dronereel.color import ColorSettings, adjust
    cast = _image([0.48, 0.50, 0.56])                          # grey card under blue light
    v, why = cg.suggest(cg.measure(cast))
    assert v["temperature"] > 5
    assert "near-neutral" in why[3]

    fixed = adjust(cast[:1, :1], ColorSettings(wb_temperature=v["temperature"],
                                               wb_tint=v["tint"]))[0, 0]
    before = abs(np.log2(0.56 / 0.48))
    after = abs(np.log2(fixed[2] / fixed[0]))
    assert after < before * 0.75                               # moves part of the way


def test_a_scene_without_neutrals_keeps_its_white_balance():
    sky_and_forest = np.concatenate([_image([0.2, 0.45, 0.85]), _image([0.15, 0.5, 0.12])])
    v, why = cg.suggest(cg.measure(sky_and_forest))
    assert (v["temperature"], v["tint"]) == (0.0, 0.0)
    assert "too few near-neutral" in why[3]


def test_deep_shadow_does_not_read_as_huge_contrast():
    water = np.concatenate([_image([0.0, 0.0, 0.002]), _image([0.6, 0.55, 0.5])])
    m = cg.measure(water)
    assert m["spread"] <= 10.5
    v, _ = cg.suggest(m)
    assert v["contrast"] >= 0.85


def test_dull_colour_gets_more_saturation_within_bounds():
    v, _ = cg.suggest(cg.measure(_image([0.5, 0.48, 0.46])))
    assert 1.0 < v["saturation"] <= 1.25


def test_clipped_highlights_ask_for_protection():
    img = _image([0.4, 0.4, 0.4])
    img[:12] = 1.0                                             # 20 % at white
    v, _ = cg.suggest(cg.measure(img))
    assert v["highlights"] == pytest.approx(0.6)


def test_recommend_measures_frames_after_the_lut(monkeypatch, identity_lut):
    times = []

    def frame(video, at, width=480):
        times.append(at)
        return _image([0.2, 0.2, 0.25])

    monkeypatch.setattr(cg, "grab_frame", frame)
    s, reasons = cg.recommend("/v/a.mp4", cg.resolve_settings({"lut_path": identity_lut}),
                              times=[1.0, 2.0, 3.0])
    assert times == [1.0, 2.0, 3.0]
    assert s["lut_path"] == identity_lut and not cg.is_neutral(s)
    assert len(reasons) == 5


def test_preview_is_before_and_after_side_by_side(monkeypatch, identity_lut):
    monkeypatch.setattr(cg, "grab_frame", lambda video, at, width=480: _image([0.4, 0.4, 0.4], (30, 40)))
    img = cg.preview("/v/a.mp4", cg.resolve_settings({"lut_path": identity_lut, "exposure": 1}), at=1)
    assert img.dtype == np.uint8 and img.shape == (30, 88, 3)
    assert img[0, 50:].mean() > img[0, :40].mean()            # the graded side is brighter


# ── panel ───────────────────────────────────────────────────────────────────

@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_color_panel_round_trips_and_resets(qapp, identity_lut):
    from modules.ui.color_panel import ColorPanel
    wanted = cg.resolve_settings({"lut_path": identity_lut, "ask": True, "exposure": 0.4,
                                  "contrast": 1.1, "saturation": 1.2, "temperature": -12,
                                  "tint": 6, "highlights": 0.3})
    panel = ColorPanel(wanted)
    assert panel.settings() == wanted
    panel.reset_grade()
    assert panel.settings() == cg.resolve_settings({"lut_path": identity_lut, "ask": True})


def test_color_panel_needs_a_video_to_recommend(qapp):
    from modules.ui.color_panel import ColorPanel
    panel = ColorPanel(video_provider=lambda: None)
    panel.recommend()
    assert "Add a video first" in panel.status.text()
    assert panel.dialog_kwargs["video_provider"]() is None
