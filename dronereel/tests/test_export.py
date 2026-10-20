"""F4: one continuous segment, frame-exact, originals untouched.

Every check decodes the export and reads the frame number drawn in each
frame (see conftest.numbered), so "continuous" means: consecutive source
frames, in order, nothing skipped, repeated or spliced in.
"""
import json
import os
import subprocess

import pytest

from dronereel import analyze as an
from dronereel import cli
from dronereel import export as ex
from dronereel.media import probe_media
from dronereel.safety import UnsafeOutputError, assert_unchanged, fingerprint
from dronereel.tests.conftest import read_frame_numbers

NTSC = 30000 / 1001


def streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,pix_fmt,r_frame_rate,color_primaries",
                          "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"]


def assert_consecutive(numbers, first, count):
    assert numbers == list(range(first, first + count)), (
        f"expected frames {first}..{first + count - 1}, got {numbers[:3]}...{numbers[-3:]} "
        f"({len(numbers)} frames)")


@pytest.fixture
def guard(numbered):
    """Fingerprints and folder listing of the originals, checked after the test."""
    fps = {k: fingerprint(v) for k, v in numbered.items() if k != "dir"}
    listing = sorted(os.listdir(numbered["dir"]))
    yield
    for fp in fps.values():
        assert_unchanged(fp)
    assert sorted(os.listdir(numbered["dir"])) == listing


def test_reencode_is_frame_exact(numbered, guard, tmp_path):
    media = probe_media(numbered["h264"])
    r = ex.export_segment(media, 70 / NTSC, 160 / NTSC, str(tmp_path),
                          ex.ExportConfig(mode="reencode", preset="ultrafast"))
    assert r.mode == "reencode" and (r.first_frame, r.end_frame) == (70, 160)
    assert_consecutive(read_frame_numbers(r.path), 70, 90)
    assert r.frames == 90 and not r.warnings
    v, a = streams(r.path)[0], streams(r.path)[1]
    assert v["codec_name"] == "hevc" and v["pix_fmt"] == "yuv420p10le"
    assert v["r_frame_rate"] == "30000/1001"
    assert v.get("color_primaries") == "bt709"
    assert a["codec_type"] == "audio" and a["codec_name"] == "aac"     # copied


def test_copy_snaps_to_keyframes_and_covers_the_request(numbered, guard, tmp_path):
    media = probe_media(numbered["h264"])
    r = ex.export_segment(media, 2.3, 4.5, str(tmp_path), ex.ExportConfig(mode="copy"))
    assert r.mode == "copy"
    # Keyframes every 30 frames: 2.3 s -> frame 60, 4.5 s -> frame 150.
    assert (r.first_frame, r.end_frame) == (60, 150)
    assert r.start <= 2.3 and r.end >= 4.5
    assert_consecutive(read_frame_numbers(r.path), 60, 90)
    assert any("fotogramas clave" in w for w in r.warnings)
    kinds = [s["codec_type"] for s in streams(r.path)]
    assert kinds == ["video", "audio"]
    assert streams(r.path)[0]["codec_name"] == "h264"                  # not re-encoded


def test_copy_of_open_gop_ends_after_the_request(numbered, guard, tmp_path):
    """Cut at keyframe 240, an open-GOP copy shows frames up to 236 (measured):
    the plan must move the cut so the request is still covered."""
    media = probe_media(numbered["hevc_open"])
    leads = dict(ex.keyframes(media.path))
    assert leads[4.0] > 0, "fixture should have leading pictures"
    for start, end in ((1.2, 3.5), (1.0, 3.98), (0.5, 4.0)):
        r = ex.export_segment(media, start, end, str(tmp_path / f"{start}-{end}"),
                              ex.ExportConfig(mode="copy"))
        numbers = read_frame_numbers(r.path)
        assert_consecutive(numbers, r.first_frame, r.end_frame - r.first_frame)
        assert numbers[0] <= round(start * 60) and numbers[-1] >= round(end * 60) - 1
        assert r.frames == len(numbers)
        assert not [w for w in r.warnings if "fotogramas;" in w]


def test_auto_copies_unless_the_picture_changes(numbered, guard, tmp_path):
    media = probe_media(numbered["hevc"])
    r = ex.export_segment(media, 1.0, 3.0, str(tmp_path / "a"))
    assert r.mode == "copy"
    assert_consecutive(read_frame_numbers(r.path), 25, 50)
    assert r.frames == 50 and not [w for w in r.warnings if "fotogramas;" in w]
    r2 = ex.export_segment(media, 1.2, 2.2, str(tmp_path / "b"),
                           ex.ExportConfig(preset="ultrafast"), needs_reencode=True)
    assert r2.mode == "reencode"
    assert_consecutive(read_frame_numbers(r2.path), 30, 25)
    v = streams(r2.path)
    assert len(v) == 1 and v[0]["r_frame_rate"] == "25/1"              # no audio invented
    with pytest.raises(ex.ExportError):
        ex.export_segment(media, 1.0, 2.0, str(tmp_path / "c"),
                          ex.ExportConfig(mode="copy"), needs_reencode=True)


def test_h264_8bit_option(numbered, guard, tmp_path):
    media = probe_media(numbered["hevc"])
    r = ex.export_segment(media, 0.0, 1.0, str(tmp_path),
                          ex.ExportConfig(mode="reencode", codec="h264", bit_depth=8,
                                          preset="ultrafast"))
    v = streams(r.path)[0]
    assert v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"
    assert_consecutive(read_frame_numbers(r.path), 0, 25)


def test_segment_at_the_end_of_the_video(numbered, guard, tmp_path):
    media = probe_media(numbered["hevc"])
    for mode in ("copy", "reencode"):
        r = ex.export_segment(media, 4.0, 99.0, str(tmp_path / mode),
                              ex.ExportConfig(mode=mode, preset="ultrafast"))
        assert_consecutive(read_frame_numbers(r.path), 100, 25)


def test_never_overwrites_and_refuses_the_originals_folder(numbered, guard, tmp_path):
    media = probe_media(numbered["hevc"])
    cfg = ex.ExportConfig(mode="copy")
    r1 = ex.export_segment(media, 1.0, 2.0, str(tmp_path), cfg)
    r2 = ex.export_segment(media, 1.0, 2.0, str(tmp_path), cfg)
    assert r1.path != r2.path and os.path.exists(r1.path) and os.path.exists(r2.path)
    with pytest.raises(UnsafeOutputError):
        ex.export_segment(media, 1.0, 2.0, numbered["dir"], cfg)
    assert not [n for n in os.listdir(tmp_path) if n.startswith(".dronereel-")]


def test_rejects_bad_segments_and_options(numbered):
    media = probe_media(numbered["hevc"])
    for a, b in ((3.0, 2.0), (-1.0, 2.0), (10.0, 12.0)):
        with pytest.raises(ValueError):
            ex.export_segment(media, a, b, "/nonexistent")
    for bad in ({"mode": "concat"}, {"codec": "vp9"}, {"bit_depth": 12}, {"music": True}):
        with pytest.raises(ValueError):
            ex.ExportConfig.from_dict(bad)


def test_cli_export_one_file_one_record(numbered, guard, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(an, "CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "out"
    assert cli.main(["export", numbered["h264"], "--min", "2", "--max", "4",
                     "--pre", "0.2", "--post", "0.2", "--mode", "reencode",
                     "--preset", "ultrafast", "--output", str(out)]) == 0
    files = sorted(os.listdir(out))
    videos = [f for f in files if f.endswith(".mp4")]
    assert len(videos) == 1 and len(files) == 2, files                 # one clip + its record
    rec = json.loads((out / videos[0].replace(".mp4", ".json")).read_text())
    e = rec["export"]
    assert 2.0 - 1e-6 <= e["duration"] <= 4.0 + 1e-6
    numbers = read_frame_numbers(str(out / videos[0]))
    assert_consecutive(numbers, e["first_frame"], e["end_frame"] - e["first_frame"])
    assert rec["source"]["sha256"] == fingerprint(numbered["h264"]).sha256
    assert rec["selection"]["best"]["start"] == pytest.approx(e["requested"][0])
    assert rec["color"]["applied"] is False
    assert "sin cambios" in capsys.readouterr().out


def test_cli_export_manual_range_copy(numbered, guard, tmp_path):
    out = tmp_path / "out"
    assert cli.main(["export", numbered["hevc"], "--start", "1", "--end", "2",
                     "--output", str(out)]) == 0
    (video,) = [f for f in os.listdir(out) if f.endswith(".mp4")]
    assert_consecutive(read_frame_numbers(str(out / video)), 25, 25)
    assert cli.main(["export", numbered["hevc"], "--start", "1",
                     "--output", str(out)]) == 2                       # --end missing
