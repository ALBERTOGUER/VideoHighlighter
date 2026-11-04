"""
Rating best segments and training the learned taste from the app
(`modules.media.taste`, the Taste tab).

DroneReel's analysis needs the real cv2, which this suite's conftest replaces,
so selections here are written by hand in the layout DroneReel logs;
dronereel/tests covers the selector and the trainer on synthetic videos.
"""

from __future__ import annotations

import json
import os
import types

import pytest

from dronereel import dataset as ds
from modules.media import taste


def window(start, end, q=0.5):
    parts = {k: q for k in ("exposure", "clarity", "stability", "dynamism", "manoeuvre",
                            "detail", "proximity", "quality", "worst_second", "variety")}
    parts.update(empty=0.0, missing_context=0.0)
    return {"start": start, "end": end, "duration": end - start, "score": q, "peak": start + 1,
            "highlight": [start + 1, start + 2], "parts": parts}


def selection():
    return {"version": 1, "video": "/v/trim.mp4", "analysis_key": "k",
            "analysis": {"fps": 5.0, "width": 384, "duration": 60, "cuts": []},
            "best": window(10, 25, 0.8), "scored": window(10, 25, 0.8),
            "candidates": [window(10, 25, 0.8), window(30, 45, 0.6), window(0, 8, 0.4)],
            "config": {"min_duration": 8, "max_duration": 20}, "warnings": [], "model": None}


@pytest.fixture
def source(tmp_path, monkeypatch):
    from dronereel import media as dm
    video = tmp_path / "DJI_0032_D.MP4"
    video.write_bytes(b"not really a video")
    monkeypatch.setattr(dm, "probe_media", lambda p: types.SimpleNamespace(
        path=p, duration=120.0, fps_fraction="30000/1001"))
    return str(video)


@pytest.fixture
def record(tmp_path, source):
    seg = tmp_path / "DJI_0032_D_best_02m10s-02m25s.mp4"
    seg.write_bytes(b"clip")
    rec = taste.segment_record(selection(), str(seg), source, offset=120.0,
                               analysis_video=str(tmp_path / "trim.mp4"), lut="/l/nano.cube")
    return taste.write_record(rec), rec


def test_the_record_is_in_the_original_s_seconds(record, source):
    path, rec = record
    assert path.endswith("_best_02m10s-02m25s.json")
    assert (rec["export"]["start"], rec["export"]["end"]) == (130, 145)
    assert [(c["start"], c["end"]) for c in rec["selection"]["candidates"]] == \
        [(130, 145), (150, 165), (120, 128)]
    assert rec["selection"]["best"]["highlight"] == [131, 132]
    assert rec["source"]["path"] == source and len(rec["source"]["sha256"]) == 64
    assert rec["app"]["offset"] == 120.0 and rec["color"] == {"lut": "/l/nano.cube",
                                                              "applied": True}


def test_dronereel_reads_the_app_s_record_as_its_own(record, tmp_path):
    path, rec = record
    assert [p for p, _ in ds.find_records(str(tmp_path))] == [path]
    entry = ds.rating_entry(rec, path, score=4)
    assert entry["chosen"]["features"] == rec["selection"]["best"]["parts"]


def test_unrated_segments_come_first_and_twins_are_left_out(record, tmp_path):
    path, rec = record
    other = dict(rec, export={**rec["export"], "path": str(tmp_path / "b.mp4")})
    (tmp_path / "b.json").write_text(json.dumps(other))
    (tmp_path / "b_vertical.json").write_text(json.dumps({**other, "twin": "b.json"}))
    dataset = str(tmp_path / "ratings.jsonl")
    taste.rate(path, rec, verdict="accept", dataset=dataset)
    found = taste.segments(str(tmp_path), dataset)
    assert [(os.path.basename(p), rated) for p, _, rated in found] == \
        [("b.json", False), (os.path.basename(path), True)]


def test_alternatives_are_numbered_like_prefer(record):
    assert taste.alternatives(record[1]) == [(2, 150, 165), (3, 120, 128)]


def test_a_rating_lands_in_the_dataset(record, tmp_path):
    path, rec = record
    dataset = str(tmp_path / "ratings.jsonl")
    taste.rate(path, rec, score=2, verdict="reject", prefer=2, note="too calm",
               dataset=dataset)
    (entry,) = ds.read(dataset)
    assert (entry["kind"], entry["score"], entry["verdict"], entry["prefer"]) == \
        ("rating", 2, "reject", 2)
    assert entry["note"] == "too calm" and len(entry["candidates"]) == 3


def test_an_own_pick_is_scored_on_the_trim_s_analysis(record, tmp_path, monkeypatch):
    from dronereel import analyze as an
    from dronereel import selector as sel
    path, rec = record
    (tmp_path / "trim.mp4").write_bytes(b"trim")
    seen = {}
    monkeypatch.setattr(an, "analyze", lambda media: seen.setdefault("video", media.path))

    def describe(a, start, end, cfg):
        seen.update(span=(start, end), cfg=cfg)
        return types.SimpleNamespace(score=0.7, to_dict=lambda: {"parts": {"exposure": 1}})

    monkeypatch.setattr(sel, "describe", describe)
    dataset = str(tmp_path / "ratings.jsonl")
    taste.rate_manual(rec, 140.0, 155.0, dataset=dataset)
    assert seen["video"].endswith("trim.mp4") and seen["span"] == (20.0, 35.0)
    assert (seen["cfg"].min_duration, seen["cfg"].max_duration) == (8, 20)
    (entry,) = ds.read(dataset)
    assert entry["kind"] == "manual" and entry["segment"]["start"] == 140.0


def test_an_own_pick_without_the_trim_measures_the_original(record, monkeypatch, source):
    _, rec = record
    assert taste._analysis_video(rec) == (source, 0.0)
    os.remove(source)
    with pytest.raises(FileNotFoundError):
        taste._analysis_video(rec)


def test_status_and_training_with_nothing_rated(tmp_path):
    dataset, model = str(tmp_path / "r.jsonl"), str(tmp_path / "m.json")
    st = taste.status(dataset, model)
    assert st["summary"]["total"] == 0 and st["model"] is None
    r = taste.train(dataset, model)
    assert r["saved"] is None and not r["better"] and r["reason"]
    assert not os.path.exists(model)


def test_training_saves_a_model_that_runs_do_not_use_until_it_is_better(record, tmp_path):
    path, rec = record
    dataset, model = str(tmp_path / "r.jsonl"), str(tmp_path / "m.json")
    taste.rate(path, rec, prefer=2, dataset=dataset)
    r = taste.train(dataset, model)
    assert r["saved"] == model and not r["better"]
    st = taste.status(dataset, model)
    assert st["model"] is not None and not st["model"]["better"]


# ── the tab ─────────────────────────────────────────────────────────────────

@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(panel, qapp):
    if panel._job is not None:
        panel._job.wait(10000)
    for _ in range(5):
        qapp.processEvents()


def test_the_tab_rates_the_segment_and_moves_on(qapp, record, tmp_path):
    from modules.ui.taste_panel import TastePanel
    path, rec = record
    dataset = str(tmp_path / "ratings.jsonl")
    panel = TastePanel(folder_provider=lambda: str(tmp_path), dataset=dataset,
                       model_path=str(tmp_path / "m.json"))
    panel.folder_edit.setText(str(tmp_path))
    panel.refresh()
    assert panel.list.count() == 1 and panel.prefer.count() == 3
    assert panel.manual_from.value() == 130

    panel._save()                                    # nothing chosen: refused
    assert ds.read(dataset) == [] and "Give a score" in panel.message.text()

    panel.score.setCurrentIndex(5)
    panel.verdict_buttons["accept"].setChecked(True)
    panel._save()
    _wait(panel, qapp)
    (entry,) = ds.read(dataset)
    assert (entry["score"], entry["verdict"], entry["prefer"]) == (5, "accept", None)
    assert panel.list.item(0).text().startswith("✓")
    assert "1 rating(s)" in panel.status_label.text()


def test_the_tab_says_what_to_do_when_there_is_nothing_to_rate(qapp, tmp_path):
    from modules.ui.taste_panel import TastePanel
    panel = TastePanel(dataset=str(tmp_path / "r.jsonl"), model_path=str(tmp_path / "m.json"))
    panel.folder_edit.setText(str(tmp_path))
    panel.refresh()
    assert "No best segments" in panel.list.item(0).text()
    assert not panel.save_btn.isEnabled()
    assert "No model yet" in panel.status_label.text()
