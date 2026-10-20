"""F6: ratings dataset. Append-only, portable between machines, and every
entry carries the features learning will need — without copying videos."""
import json
import os

import pytest

from dronereel import analyze as an
from dronereel import cli
from dronereel import dataset as ds
from dronereel import selector as sel
from dronereel.safety import assert_unchanged, fingerprint
from dronereel.tests.test_selector import cfg, synthetic


@pytest.fixture
def exported(numbered, tmp_path, monkeypatch):
    """Two clips exported by the CLI from the numbered H.264: one chosen by
    the selector, one by hand. Returns (output dir, dataset path)."""
    monkeypatch.setattr(an, "CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "out"
    assert cli.main(["export", numbered["h264"], "--min", "2", "--max", "3", "--pre", "0.2",
                     "--post", "0.2", "--output", str(out)]) == 0
    assert cli.main(["export", numbered["h264"], "--start", "6", "--end", "8",
                     "--output", str(out)]) == 0
    return out, str(tmp_path / "data" / "ratings.jsonl")


def records(out):
    return ds.find_records(str(out))


def test_describe_matches_the_selector():
    a = synthetic(60, smooth_turn=[(30, 36)])
    c = cfg(min_duration=10, max_duration=20)
    r = sel.select(a, c)
    d = sel.describe(a, r.best.start, r.best.end, c)
    assert d.score == pytest.approx(r.best.score)
    assert d.parts == pytest.approx(r.best.parts)


def test_rate_a_selected_clip(exported, numbered):
    out, data = exported
    (path, rec), _ = sorted(records(out), key=lambda pr: "selection" in pr[1]
                            and pr[1]["selection"].get("manual", False))
    assert cli.main(["rate", path, "--score", "4", "--accept", "--note", "buen giro",
                     "--dataset", data]) == 0
    (e,) = ds.read(data)
    assert e["kind"] == "rating" and e["score"] == 4 and e["verdict"] == "accept"
    assert e["source"]["sha256"] == fingerprint(numbered["h264"]).sha256
    assert e["clip"]["features"]["stability"] >= 0
    assert e["chosen"]["features"] == e["clip"]["features"]
    assert e["candidates"] and all("features" in c for c in e["candidates"])
    assert e["selector"]["config"]["weights"]["stability"] == 1.5
    assert e["note"] == "buen giro"


def test_rate_by_clip_path_and_hand_range_gets_features(exported):
    out, data = exported
    manual = [(p, r) for p, r in records(out) if r["selection"].get("manual")]
    (path, rec), = manual
    assert cli.main(["rate", rec["export"]["path"], "--reject", "--dataset", data]) == 0
    (e,) = ds.read(data)
    assert e["verdict"] == "reject" and e["clip"]["record"] == path
    assert e["clip"]["features"] and "quality" in e["clip"]["features"]
    assert "candidates" not in e                       # nothing was selected


def test_rate_rejects_empty_or_impossible(exported):
    out, data = exported
    path = records(out)[0][0]
    assert cli.main(["rate", path, "--dataset", data]) == 2          # nothing said
    assert cli.main(["rate", path, "--prefer", "99", "--dataset", data]) == 2
    assert not os.path.exists(data)


def test_manual_pick_on_a_video(numbered, tmp_path, monkeypatch):
    monkeypatch.setattr(an, "CACHE_DIR", str(tmp_path / "cache"))
    data = str(tmp_path / "r.jsonl")
    fp = fingerprint(numbered["h264"])
    assert cli.main(["rate", numbered["h264"], "--manual", "1.5", "4.5", "--score", "5",
                     "--dataset", data]) == 0
    (e,) = ds.read(data)
    assert e["kind"] == "manual" and e["segment"]["start"] == 1.5
    assert e["segment"]["features"]["quality"] > 0
    assert e["candidates"] and e["chosen"]
    assert e["source"]["sha256"] == fp.sha256
    assert_unchanged(fp)


def test_review_interactive(exported, monkeypatch, capsys):
    out, data = exported
    # Records come sorted by name: first the hand-range clip (no alternatives,
    # so no "prefer" question), then the selected one (two alternatives).
    answers = iter(["5", "s", "precioso",          # first clip: rated
                    "", "", "", ""])               # second: skipped (all Enter)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main(["review", "--output", str(out), "--dataset", data]) == 0
    (e,) = ds.read(data)
    assert e["score"] == 5 and e["verdict"] == "accept" and e["note"] == "precioso"
    assert e["clip"]["start"] == pytest.approx(6.0, abs=0.05)
    # Next time only the skipped one is pending; 'q' quits without saving.
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    assert cli.main(["review", "--output", str(out), "--dataset", data]) == 0
    assert "1 clip(s) sin valorar" in capsys.readouterr().out
    assert len(ds.read(data)) == 1


def test_append_only_merge_and_damaged_lines(tmp_path):
    pi, mac = str(tmp_path / "pi.jsonl"), str(tmp_path / "mac.jsonl")
    src = {"path": "/x.mp4", "sha256": "ab", "duration": 10, "fps": "30/1"}
    ds.append([ds.manual_entry(src, 1, 3, {"quality": 0.5}, None)], pi)
    before = open(pi, encoding="utf-8").read()
    ds.append([ds.manual_entry(src, 4, 6, {"quality": 0.6}, None)], pi)
    assert open(pi, encoding="utf-8").read().startswith(before)    # never rewritten
    ds.append([ds.manual_entry(src, 7, 9, {"quality": 0.7}, None)], mac)
    assert ds.merge(mac, pi) == 1
    assert ds.merge(mac, pi) == 0                                   # no duplicates
    assert ds.merge(pi, mac) == 2
    assert {e["id"] for e in ds.read(pi)} == {e["id"] for e in ds.read(mac)}
    with open(pi, "a", encoding="utf-8") as f:
        f.write('{"id": "half-writ')
    assert len(ds.read(pi)) == 3
    sm = ds.summary(ds.read(pi))
    assert sm["manual"] == 3 and sm["sources"] == 1


def test_dataset_command(tmp_path, capsys):
    data = str(tmp_path / "r.jsonl")
    other = str(tmp_path / "o.jsonl")
    ds.append([ds.manual_entry({"sha256": "x"}, 0, 1, {}, None, score=3)], other)
    assert cli.main(["dataset", "--dataset", data, "--merge", other]) == 0
    text = capsys.readouterr().out
    assert "Añadidas 1" in text and "3★ ×1" in text
