"""
Color LUT applied while clips are cut (`modules.media.lut`, `cut_video`).

Log footage (D-Log, S-Log…) is flat until the camera maker's 3D LUT maps it to
Rec.709. The path goes into an ffmpeg filter graph, which is parsed twice, so a
path with a space, an apostrophe, brackets or a drive colon has to survive both
levels — the real-ffmpeg test below is the one that proves the escaping.
"""

from __future__ import annotations

import subprocess

import pytest

from modules.media import lut, video_cutter

IDENTITY_CUBE = "LUT_3D_SIZE 2\n" + "".join(
    f"{r} {g} {b}\n" for b in (0, 1) for g in (0, 1) for r in (0, 1))


def _write_cube(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(IDENTITY_CUBE)
    return path


def test_resolve_lut_empty_means_none():
    assert lut.resolve_lut(None) is None
    assert lut.resolve_lut("") is None
    assert lut.resolve_lut("   ") is None


def test_resolve_lut_missing_file_is_skipped_with_a_log(tmp_path):
    lines = []
    assert lut.resolve_lut(str(tmp_path / "gone.cube"), log=lines.append) is None
    assert "not found" in lines[0]


def test_resolve_lut_rejects_non_lut_files(tmp_path):
    other = tmp_path / "clip.mp4"
    other.write_text("x")
    lines = []
    assert lut.resolve_lut(str(other), log=lines.append) is None
    assert "not a LUT" in lines[0]


def test_resolve_lut_accepts_any_case_suffix(tmp_path):
    cube = _write_cube(tmp_path / "Rec709.CUBE")
    assert lut.resolve_lut(str(cube)) == str(cube)


def test_lut_filter_escapes_both_levels():
    vf = lut.lut_filter("/a b/it's [x],y;z:w.cube")
    assert vf.startswith("lut3d=file=/a b/it\\\\\\'s \\[x\\]\\,y\\;z\\\\:w.cube")
    assert vf.endswith(":interp=tetrahedral")


def test_lut_filter_loads_in_real_ffmpeg(tmp_path):
    from modules.system.app_paths import ffmpeg_exe
    try:
        exe = ffmpeg_exe()
    except Exception:
        pytest.skip("no ffmpeg available")
    cube = _write_cube(tmp_path / "it's [a],b;c" / "look.cube")
    vf = lut.lut_filter(str(cube)) + ",format=yuv420p"
    result = subprocess.run(
        [exe, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x32:d=0.1",
         "-vf", vf, "-f", "null", "-"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _capture_cut(monkeypatch, tmp_path, **kwargs):
    calls = []
    out = tmp_path / "clip.mp4"

    def fake_run(cmd, **_kw):
        calls.append(cmd)
        out.write_text("ok")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(video_cutter, "encoder_chain",
                        lambda *_a, **_k: [("libx264", ["-c:v", "libx264"])])
    monkeypatch.setattr(video_cutter, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(video_cutter.subprocess, "run", fake_run)
    video_cutter.cut_video("in.mp4", 1, 3, str(out), mode="cpu", **kwargs)
    cmd = calls[0]
    return cmd[cmd.index("-vf") + 1]


def test_cut_video_without_lut_keeps_plain_format(monkeypatch, tmp_path):
    assert _capture_cut(monkeypatch, tmp_path) == "format=yuv420p,setpts=PTS-STARTPTS"


def test_cut_video_applies_lut_before_format(monkeypatch, tmp_path):
    vf = _capture_cut(monkeypatch, tmp_path, lut_path="/luts/dlog.cube")
    assert vf.startswith("lut3d=file=/luts/dlog.cube")
    assert vf.endswith(",format=yuv420p,setpts=PTS-STARTPTS")
