"""F7: learning a taste from ratings. A simulated user with a hidden
preference rates the selector's choices; the model must find that
preference, choose better on videos it never saw, and claim nothing when
there is nothing to learn or too little data."""
import json
import os

import numpy as np
import pytest

from dronereel import analyze as an
from dronereel import cli
from dronereel import dataset as ds
from dronereel import ranker as rk
from dronereel import selector as sel
from dronereel.tests.test_selector import FPS

CFG = sel.SelectConfig(min_duration=8, max_duration=20)


def video(seed: int, seconds: float = 120) -> an.Analysis:
    """Random stretches of turns, close flying, calm, chaos and blur."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    s = {k: np.zeros(n) for k in an.FIELDS}
    s["t"] = np.arange(n) / FPS
    for k, v in {"brightness": 0.45, "sharpness": 100.0, "colorfulness": 25.0, "edges": 0.04,
                 "diff": 3.0, "pan_x": 0.05, "zoom": 0.05, "coherence": 0.95,
                 "parallax": 0.005, "tracked": 300, "jerk": 0.1}.items():
        s[k][:] = v
    t = 0.0
    while t < seconds:
        span = rng.uniform(4, 15)
        sl = slice(int(t * FPS), int(min(seconds, t + span) * FPS))
        kind = rng.choice(["turn", "close", "calm", "chaos", "blur"])
        vals = {"turn": {"pan_x": rng.uniform(.15, .35), "roll": rng.uniform(20, 60),
                         "zoom": rng.uniform(.1, .3), "jerk": .15, "parallax": .01},
                "close": {"pan_x": rng.uniform(.03, .1), "zoom": rng.uniform(.2, .4),
                          "parallax": rng.uniform(.05, .12), "jerk": .2, "edges": .07},
                "calm": {"pan_x": .03, "zoom": .02},
                "chaos": {"pan_x": 1.2, "roll": 150, "coherence": .3, "jerk": 5, "sharpness": 40},
                "blur": {"sharpness": 15}}[kind]
        for k, v in vals.items():
            s[k][sl] = v
        t += span
    for k in ("pan_x", "roll", "zoom", "parallax", "edges"):
        s[k] *= 1 + 0.15 * rng.standard_normal(n)
    return an.Analysis(path=f"/v/{seed}.mp4", fps=FPS, width=384, height=216, decoder="t",
                       seconds=0, duration=n / FPS, samples=s)


def loves_proximity(w: dict) -> float:
    x = rk.vector({"start": w["start"], "end": w["end"], "features": w["parts"]})
    return rk.base_score(x) + 0.6 * w["parts"]["proximity"]


def same_as_selector(w: dict) -> float:
    """Exactly the selector's own window score (base_score alone leaves out
    its small length bonus — a slightly different taste, which the model
    rightly picked up: 91 % -> 98 %)."""
    return w["score"]


def rate(seeds, utility, ranker=None) -> list:
    """The simulated user: sees the clip and the alternatives, says which is best."""
    out = []
    for sd in seeds:
        a = video(sd)
        r = sel.select(a, CFG, ranker)
        rec = sel.log_record(r, a, a.path, "k")
        best = int(np.argmax([utility(c) for c in rec["candidates"]]))
        record = {"export": {"path": f"/o/{sd}.mp4", "start": r.best.start, "end": r.best.end,
                             "requested": [0, 0], "mode": "copy"},
                  "source": {"path": a.path, "sha256": f"sha{sd}", "duration": a.duration,
                             "fps": "5/1"},
                  "selection": rec}
        out.append(ds.rating_entry(record, f"/o/{sd}.json",
                                   prefer=best + 1 if best else None,
                                   verdict=None if best else "accept"))
    return out


def mean_utility(seeds, utility, model) -> float:
    return float(np.mean([utility(sel.select(video(s), CFG, model).best.to_dict())
                          for s in seeds]))


def test_learns_a_hidden_taste_and_chooses_better_on_new_videos():
    model, ev = rk.train(rate(range(8), loves_proximity))
    assert ev["better"] and ev["after"] > ev["before"] + 0.2
    adj = dict(zip(rk.FEATURES, model.adj))
    assert adj["proximity"] > 0 and max(adj, key=lambda k: abs(adj[k])) == "proximity"
    unseen = range(1000, 1010)
    before = mean_utility(unseen, loves_proximity, None)
    after = mean_utility(unseen, loves_proximity, model)
    assert after > before + 0.05


def test_nothing_to_learn_claims_nothing():
    model, ev = rk.train(rate(range(8), same_as_selector))
    assert not ev["better"] and "no mejora" in ev["reason"]
    assert np.abs(model.adj).max() < 0.05


def test_too_little_data_is_not_trusted():
    model, ev = rk.train(rate(range(3), loves_proximity))
    assert not ev["better"] and "pocos datos" in ev["reason"]
    assert rk.train([])[0] is None


def test_pairs_from_each_kind_of_judgement():
    def win(start, end, q):
        return {"start": start, "end": end, "score": q,
                "parts": {**{k: q for k in rk.FEATURES if k != "duration"}}}
    cands = [win(0, 10, .5), win(20, 30, .4), win(40, 50, .3)]
    selection = {"best": cands[0], "candidates": cands, "config": None}
    record = {"export": {"path": "/o/c.mp4", "start": 0, "end": 10}, "source": {"sha256": "v1"},
              "selection": selection}
    pref = ds.rating_entry(record, "/o/c.json", prefer=3)
    ps = rk.pairs([pref])
    assert [(p.weight, round(p.better[0], 2), round(p.worse[0], 2)) for p in ps] == \
        [(1.0, .3, .5), (0.5, .3, .5), (0.5, .3, .4)]
    acc = ds.rating_entry(record, "/o/c.json", verdict="accept")
    assert [(p.weight, round(p.worse[0], 2)) for p in rk.pairs([acc])] == [(0.5, .4), (0.5, .3)]
    man = ds.manual_entry({"sha256": "v1"}, 22, 31, {k: .9 for k in rk.FEATURES}, selection)
    assert len(rk.pairs([man])) == 2                       # the overlapping window is skipped
    rej = ds.rating_entry({**record, "source": {"sha256": "v2"}}, "/o/d.json", verdict="reject")
    cross = [p for p in rk.pairs([acc, rej]) if len(p.groups) == 2]
    assert len(cross) == 1 and cross[0].weight == 0.5


def test_zero_correction_changes_nothing():
    a = video(7)
    zero = rk.Ranker(np.zeros(len(rk.FEATURES)), np.ones(len(rk.FEATURES)),
                     np.zeros(len(rk.FEATURES)))
    plain, with_zero = sel.select(a, CFG), sel.select(a, CFG, zero)
    assert (plain.best.start, plain.best.end) == (with_zero.best.start, with_zero.best.end)


def test_save_load_keeps_previous(tmp_path):
    model, _ = rk.train(rate(range(5), loves_proximity))
    path = str(tmp_path / "m" / "model.json")
    assert rk.save(model, path) is None
    backup = rk.save(model, path)
    assert backup and os.path.exists(backup) and os.path.exists(path)
    back = rk.load(path)
    assert np.allclose(back.adj, model.adj) and back.trained == json.loads(json.dumps(model.trained))
    raw = json.load(open(path))
    raw["version"] = 99
    json.dump(raw, open(path, "w"))
    with pytest.raises(ValueError):
        rk.load(path)


def test_cli_train_then_select_with_and_without(numbered, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(an, "CACHE_DIR", str(tmp_path / "cache"))
    data = str(tmp_path / "ratings.jsonl")
    model_path = str(tmp_path / "model.json")
    monkeypatch.setattr(rk, "DEFAULT_MODEL", model_path)
    ds.append(rate(range(8), loves_proximity), data)
    assert cli.main(["train", "--dataset", data, "--model", model_path]) == 0
    out = capsys.readouterr().out
    assert "Antes:" in out and "Después:" in out and "+proximidad" in out
    assert "Se usará por defecto" in out
    args = ["select", numbered["h264"], "--min", "2", "--max", "4",
            "--output", str(tmp_path / "logs")]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "Modelo:     el tuyo" in out and "sin tu modelo habría elegido" in out
    assert cli.main([*args, "--no-model"]) == 0
    assert "desactivado" in capsys.readouterr().out
    logs = sorted(os.listdir(tmp_path / "logs"))
    first = json.load(open(tmp_path / "logs" / logs[0]))
    assert first["model"]["better"] is True
    second = json.load(open(tmp_path / "logs" / logs[1]))
    assert second["model"] is None


def test_cli_model_not_better_is_not_used_by_default(numbered, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(an, "CACHE_DIR", str(tmp_path / "cache"))
    model_path = str(tmp_path / "model.json")
    monkeypatch.setattr(rk, "DEFAULT_MODEL", model_path)
    data = str(tmp_path / "r.jsonl")
    ds.append(rate(range(3), loves_proximity), data)
    assert cli.main(["train", "--dataset", data, "--model", model_path]) == 0
    assert "No se usará por defecto" in capsys.readouterr().out
    assert cli.main(["select", numbered["h264"], "--min", "2", "--max", "4",
                     "--output", str(tmp_path / "l")]) == 0
    assert "no mejora al selector todavía" in capsys.readouterr().out
