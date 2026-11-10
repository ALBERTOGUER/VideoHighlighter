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


# ── the codec of a stabilized clip ──────────────────────────────────────────

def _capture_cmd(monkeypatch, tmp_path, **kwargs):
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
    return calls[0]


@pytest.mark.parametrize("codec", ["H.265/HEVC", "ProRes"])
def test_a_stabilized_clip_keeps_hevc_10_bit_playable_on_apple(monkeypatch, tmp_path, codec):
    cmd = _capture_cmd(monkeypatch, tmp_path, codec=codec, lut_path="/luts/dlog.cube")
    assert cmd[cmd.index("-c:v") + 1] == "libx265"
    assert cmd[cmd.index("-tag:v") + 1] == "hvc1"
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("lut3d=") and vf.endswith(",format=yuv420p10le,setpts=PTS-STARTPTS")


def test_h264_chosen_means_h264(monkeypatch, tmp_path):
    cmd = _capture_cmd(monkeypatch, tmp_path, codec="H.264/AVC")
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-vf") + 1] == "format=yuv420p,setpts=PTS-STARTPTS"


def test_hevc_falls_back_to_h264_when_x265_fails(monkeypatch, tmp_path):
    calls = []
    out = tmp_path / "clip.mp4"

    def fake_run(cmd, **_kw):
        calls.append(cmd)
        if "libx265" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "x265 missing")
        out.write_text("ok")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(video_cutter, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(video_cutter.subprocess, "run", fake_run)
    video_cutter.cut_video("in.mp4", 1, 3, str(out), codec="H.265/HEVC")
    assert [c[c.index("-c:v") + 1] for c in calls] == ["libx265", "libx264"]


def test_a_real_hevc_cut_is_10_bit_and_tagged_hvc1(tmp_path):
    from modules.system.app_paths import ffmpeg_exe
    try:
        exe = ffmpeg_exe()
    except Exception:
        pytest.skip("no ffmpeg available")
    src = tmp_path / "src.mp4"
    made = subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i",
                           "testsrc=size=64x64:rate=10:d=2", "-pix_fmt", "yuv420p",
                           "-c:v", "libx264", str(src)], capture_output=True)
    if made.returncode != 0:
        pytest.skip("this ffmpeg cannot make a test clip")
    out = tmp_path / "clip.mp4"
    video_cutter.cut_video(str(src), 0.5, 1.5, str(out), mode="cpu", codec="H.265/HEVC")
    import av
    with av.open(str(out)) as c:
        v = c.streams.video[0]
        assert v.codec_context.name == "hevc" and v.codec_context.format.name == "yuv420p10le"
    assert b"hvc1" in out.read_bytes()
