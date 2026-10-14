import os

import numpy as np
import pytest

from dronereel import analyze as an
from dronereel.media import probe_media
from dronereel.safety import assert_unchanged, fingerprint


def measured(path, **kw):
    return an.measure(probe_media(path), **kw)


def middle(a: an.Analysis, name: str) -> float:
    """Median over samples, skipping the first (copied from the second)."""
    return float(np.nanmedian(a.samples[name][1:]))


def test_static_is_still_and_coherent(motion_clips):
    a = measured(motion_clips["static"])
    assert a.count == pytest.approx(2.5 * an.DEFAULT_FPS, abs=1)
    for name in ("pan_x", "pan_y", "zoom"):
        assert abs(middle(a, name)) < 0.02
    assert abs(middle(a, "roll")) < 1.0
    assert middle(a, "coherence") > 0.9


def test_pan_measured_in_frame_widths_with_sign(motion_clips):
    a = measured(motion_clips["pan"])
    assert middle(a, "pan_x") == pytest.approx(0.2, rel=0.15)
    assert abs(middle(a, "pan_y")) < 0.02
    assert abs(middle(a, "roll")) < 1.0
    assert middle(a, "coherence") > 0.9


def test_roll_is_rotation_not_chaos(motion_clips):
    a = measured(motion_clips["roll"])
    assert abs(middle(a, "roll")) == pytest.approx(np.degrees(0.35), rel=0.15)
    # Rotation about the centre is not a pan.
    assert abs(middle(a, "pan_x")) < 0.03
    # The point of the similarity model: a clean roll is coherent.
    assert middle(a, "coherence") > 0.85


def test_zoom_reads_as_forward_movement(motion_clips):
    a = measured(motion_clips["zoom"])
    assert 0.12 < middle(a, "zoom") < 0.3
    assert abs(middle(a, "roll")) < 1.0
    assert middle(a, "coherence") > 0.85


def test_smooth_movement_has_low_jerk(motion_clips):
    for name in ("pan", "roll", "zoom"):
        assert middle(measured(motion_clips[name]), "jerk") < 0.3, name


def test_blur_lowers_sharpness(motion_clips):
    sharp = middle(measured(motion_clips["static"]), "sharpness")
    soft = middle(measured(motion_clips["blur"]), "sharpness")
    assert soft < 0.3 * sharp


def test_blown_highlights_measured(motion_clips):
    assert middle(measured(motion_clips["static"]), "clip_high") < 0.05
    assert middle(measured(motion_clips["blown"]), "clip_high") > 0.5


def test_hard_cut_found_and_smooth_movement_is_not_a_cut(motion_clips):
    assert measured(motion_clips["cut"]).cuts == [pytest.approx(2.0, abs=0.25)]
    for name in ("static", "pan", "roll", "zoom"):
        assert measured(motion_clips[name]).cuts == [], name


def test_featureless_frames_give_unknown_movement_not_zero(motion_clips):
    a = measured(motion_clips["flat"])
    assert np.all(np.isnan(a.samples["roll"]))
    assert np.all(a.samples["coherence"] == 0)
    assert middle(a, "colorfulness") > 20


def test_analysis_leaves_original_untouched_and_writes_nothing_beside_it(motion_clips, tmp_path):
    path = motion_clips["pan"]
    before = sorted(os.listdir(os.path.dirname(path)))
    fp = fingerprint(path)
    an.analyze(probe_media(path), cache_dir=str(tmp_path / "cache"))
    assert_unchanged(fp)
    assert sorted(os.listdir(os.path.dirname(path))) == before


def test_cache_round_trip(motion_clips, tmp_path):
    media = probe_media(motion_clips["zoom"])
    first = an.analyze(media, cache_dir=str(tmp_path))
    second = an.analyze(media, cache_dir=str(tmp_path))
    assert not first.cached and second.cached
    for k in an.FIELDS:
        np.testing.assert_array_equal(first.samples[k], second.samples[k])
    assert second.cuts == first.cuts


def test_cache_key_depends_on_settings(motion_clips):
    p = motion_clips["static"]
    assert an.content_key(p, 5.0, 384) != an.content_key(p, 4.0, 384)
    assert an.content_key(p, 5.0, 384) != an.content_key(p, 5.0, 256)


def test_hardware_and_software_decoders_agree(videos):
    media = probe_media(videos["hevc10"])
    hw = an.measure(media, prefer_hardware=True)
    sw = an.measure(media, prefer_hardware=False)
    assert sw.decoder == "software"
    assert hw.count == sw.count
    assert np.max(np.abs(hw.samples["brightness"] - sw.samples["brightness"])) < 0.01


def test_per_second_reduction(motion_clips):
    a = measured(motion_clips["blown"])
    per = a.per_second("clip_high")
    assert len(per) == 3 and np.all(per > 0.5)
