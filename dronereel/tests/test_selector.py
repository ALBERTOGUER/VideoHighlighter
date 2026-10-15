import json
import os

import numpy as np
import pytest

from dronereel import analyze as an
from dronereel import cli
from dronereel import selector as sel
from dronereel.safety import assert_unchanged, fingerprint

FPS = 5.0


def synthetic(seconds: float, cuts=(), **spans) -> an.Analysis:
    """An Analysis of a decent, calm, slowly panning video. ``spans`` maps a
    preset name to ``[(start_s, end_s), ...]`` where the samples are replaced."""
    n = int(seconds * FPS)
    s = {k: np.zeros(n) for k in an.FIELDS}
    s["t"] = np.arange(n) / FPS
    base = {"brightness": 0.45, "clip_high": 0.0, "clip_low": 0.0, "sharpness": 100.0,
            "colorfulness": 25.0, "edges": 0.04, "diff": 3.0, "pan_x": 0.05,
            "pan_y": 0.0, "roll": 0.0, "zoom": 0.05, "coherence": 0.95,
            "parallax": 0.005, "tracked": 300, "jerk": 0.1}
    presets = {
        # A clean, sustained FPV turn: coherent rotation and forward motion.
        "smooth_turn": {"pan_x": 0.25, "roll": 45.0, "zoom": 0.2, "coherence": 0.95,
                        "jerk": 0.15, "parallax": 0.02},
        # Fast but chaotic: huge rates, nothing coherent, constant jerk.
        "chaos": {"pan_x": 1.5, "pan_y": 1.0, "roll": 200.0, "zoom": 0.8,
                  "coherence": 0.25, "jerk": 6.0, "parallax": 0.2, "sharpness": 40.0},
        # Open sky: nothing to track or look at.
        "empty": {"pan_x": np.nan, "pan_y": np.nan, "roll": np.nan, "zoom": np.nan,
                  "coherence": 0.0, "parallax": np.nan, "tracked": 0, "jerk": np.nan,
                  "edges": 0.002, "colorfulness": 5.0},
        "blown": {"clip_high": 0.6, "brightness": 0.9},
        "blur": {"sharpness": 10.0},
    }
    for k, v in base.items():
        s[k][:] = v
    for name, ranges in spans.items():
        for a, b in ranges:
            sl = slice(int(a * FPS), int(b * FPS))
            for k, v in presets[name].items():
                s[k][sl] = v
    return an.Analysis(path="/x/DJI_0001.MP4", fps=FPS, width=384, height=216,
                       decoder="test", seconds=0.0, duration=n / FPS, samples=s,
                       cuts=list(cuts))


def cfg(**kw) -> sel.SelectConfig:
    return sel.SelectConfig(**kw)


def assert_inside_one_shot(a, w):
    assert 0 <= w.start < w.end <= a.duration + 1e-9
    for c in a.cuts:
        assert not (w.start < c < w.end), f"window {w.start}-{w.end} crosses cut {c}"


def test_one_window_within_bounds():
    a = synthetic(60, smooth_turn=[(30, 36)])
    r = sel.select(a, cfg(min_duration=10, max_duration=20))
    assert isinstance(r.best, sel.Window)
    assert 10 - 1e-9 <= r.best.duration <= 20 + 1e-9
    assert_inside_one_shot(a, r.best)


def test_contains_the_moment_with_context():
    a = synthetic(60, smooth_turn=[(30, 36)])
    r = sel.select(a, cfg(min_duration=10, max_duration=20, pre_roll=2, post_roll=2))
    assert 30 <= r.best.peak < 36
    # The highlight is the whole turn, give or take the 1 s smoothing.
    hl0, hl1 = r.best.highlight
    assert 30 <= hl0 <= 30.5 and 35.5 <= hl1 <= 36
    assert r.best.start <= 30 and r.best.end >= 36          # the whole manoeuvre
    assert r.best.start <= hl0 - 2 + 1e-9 and r.best.end >= hl1 + 2 - 1e-9
    assert r.best.parts["missing_context"] == 0


def test_smooth_beats_intense_chaos():
    """Intense movement is not quality."""
    a = synthetic(80, smooth_turn=[(15, 25)], chaos=[(50, 62)])
    r = sel.select(a, cfg(min_duration=10, max_duration=15))
    assert r.best.end <= 50 or r.best.start >= 62
    assert r.best.start < 25 and r.best.end > 15
    comp = sel.components(a)
    smooth = slice(int(16 * FPS), int(24 * FPS))
    chaos = slice(int(51 * FPS), int(61 * FPS))
    assert comp["stability"][chaos].mean() < 0.1 < 0.7 < comp["stability"][smooth].mean()
    assert comp["dynamism"][chaos].mean() < comp["dynamism"][smooth].mean()


def test_dynamism_saturates():
    a = synthetic(20, smooth_turn=[(0, 10)])
    fast = synthetic(20, smooth_turn=[(0, 10)])
    fast.samples["pan_x"][:] *= 4
    d1 = sel.components(a)["dynamism"][:10].mean()
    d4 = sel.components(fast)["dynamism"][:10].mean()
    assert d4 < 1.0 and d4 - d1 < 0.25            # 4x faster is not 4x better


def test_never_crosses_a_cut():
    # The best moment sits right on a cut: the window must stay on one side.
    a = synthetic(60, cuts=[30.0], smooth_turn=[(26, 34)])
    for mx in (12, 20, 0):
        r = sel.select(a, cfg(min_duration=10, max_duration=mx or None))
        assert_inside_one_shot(a, r.best)
        for c in r.candidates:
            assert_inside_one_shot(a, c)


def test_avoids_empty_blown_and_blurred_stretches():
    a = synthetic(70, empty=[(0, 20)], blown=[(20, 35)], blur=[(55, 70)])
    r = sel.select(a, cfg(min_duration=12, max_duration=15))
    assert r.best.start >= 35 - 1e-9 and r.best.end <= 55 + 1e-9


def test_empty_component_and_penalty():
    a = synthetic(40, empty=[(0, 20)])
    r = sel.select(a, cfg(min_duration=10, max_duration=10, candidates=4))
    by_start = {round(c.start): c for c in r.candidates}
    worst_empty = max(c.parts["empty"] for c in r.candidates)
    assert r.best.parts["empty"] == 0 and worst_empty > 0.5
    assert len(by_start) == len(r.candidates)


def test_short_video_gives_whole_shot_with_warning():
    a = synthetic(6)
    r = sel.select(a, cfg(min_duration=10, max_duration=30))
    assert r.best.start == 0 and r.best.end == pytest.approx(6.0)
    assert any("duración mínima" in w for w in r.warnings)


def test_no_upper_limit_can_take_the_whole_shot():
    a = synthetic(40, smooth_turn=[(0, 40)])
    r = sel.select(a, cfg(min_duration=10, max_duration=None, pre_roll=0, post_roll=0))
    assert r.best.duration > 30


def test_candidates_do_not_overlap_and_best_first():
    a = synthetic(120, smooth_turn=[(20, 30), (70, 80)])
    r = sel.select(a, cfg(min_duration=10, max_duration=15, candidates=6))
    assert r.scored is r.candidates[0]
    scores = [c.score for c in r.candidates]
    assert scores == sorted(scores, reverse=True)
    for i, x in enumerate(r.candidates):
        for y in r.candidates[i + 1:]:
            inter = max(0.0, min(x.end, y.end) - max(x.start, y.start))
            assert inter < 0.3 * min(x.duration, y.duration) + 1e-9


def test_edges_move_to_calm_moments():
    a = synthetic(60, smooth_turn=[(20, 32)])
    jerk = a.samples["jerk"]
    jerk[:] = 1.0
    jerk[int(15.4 * FPS)] = 0.0          # one calm sample near each edge
    jerk[int(36.6 * FPS)] = 0.0
    r = sel.select(a, cfg(min_duration=10, max_duration=30, edge_slack=1.5))
    assert r.best.start == pytest.approx(15.4, abs=1e-6) or r.best.start == r.scored.start
    assert r.best.duration >= 10
    assert_inside_one_shot(a, r.best)


def test_deterministic():
    a = synthetic(90, smooth_turn=[(20, 30), (60, 70)])
    r1 = sel.select(a, cfg()).to_dict()
    r2 = sel.select(a, cfg()).to_dict()
    assert r1 == r2


def test_config_rejects_unknown_options():
    with pytest.raises(ValueError):
        sel.SelectConfig.from_dict({"min_duratoin": 5})
    with pytest.raises(ValueError):
        sel.SelectConfig.from_dict({"weights": {"beauty": 1}})
    c = sel.SelectConfig.from_dict({"min_duration": 5, "weights": {"proximity": 0}})
    assert c.min_duration == 5 and c.weights["proximity"] == 0 and c.weights["stability"] == 1.5
    with pytest.raises(ValueError):
        sel.select(synthetic(30), cfg(min_duration=20, max_duration=10))


def test_cli_select_writes_one_log_and_leaves_original(motion_clips, tmp_path, monkeypatch):
    monkeypatch.setattr(an, "CACHE_DIR", str(tmp_path / "cache" / "dronereel" / "analysis"))
    video = motion_clips["cut"]
    src_dir = os.path.dirname(video)
    before = sorted(os.listdir(src_dir))
    fp = fingerprint(video)
    out = tmp_path / "out"
    argv = ["select", video, "--min", "1", "--max", "3", "--pre", "0.4", "--post", "0.4",
            "--output", str(out)]
    assert cli.main(argv) == 0
    assert cli.main(argv) == 0                      # second run: new file, no overwrite
    logs = sorted(os.listdir(out))
    assert logs == ["cut_select.json", "cut_select_1.json"]
    rec = json.loads((out / logs[0]).read_text())
    best = rec["best"]
    assert 1.0 - 1e-6 <= best["duration"] <= 3.0 + 1e-6
    for c in rec["analysis"]["cuts"]:
        assert not (best["start"] < c < best["end"])
    assert rec["analysis"]["cuts"], "the synthetic clip has a hard cut"
    assert_unchanged(fp)
    assert sorted(os.listdir(src_dir)) == before


def test_cli_refuses_output_next_to_original(motion_clips, tmp_path, monkeypatch):
    monkeypatch.setattr(an, "CACHE_DIR", str(tmp_path / "cache"))
    video = motion_clips["static"]
    from dronereel.safety import UnsafeOutputError
    with pytest.raises(UnsafeOutputError):
        cli.main(["select", video, "--min", "1", "--output", os.path.dirname(video)])
