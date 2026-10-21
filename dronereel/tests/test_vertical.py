"""F8: the vertical 9:16 export — still ONE continuous segment, frame-exact,
with a crop window that follows what stands out, smoothly; originals intact.

Frame numbers are drawn as full-width horizontal stripes here (the numbered
clips of test_export put them in a 4x3 grid, which a 9:16 crop cuts off).
"""
import json
import os
import subprocess

import numpy as np
import pytest

from dronereel import cli
from dronereel import export as ex
from dronereel import vertical as vt
from dronereel.media import probe_media
from dronereel.safety import assert_unchanged, fingerprint

W, H, BITS, STRIPE = 640, 360, 12, 30


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", *args], check=True)


def _stripes(k):
    img = np.zeros((H, W), np.uint8)
    for b in range(BITS):
        if k >> b & 1:
            img[b * STRIPE + 4:(b + 1) * STRIPE - 4, :] = 255
    return img


def read_stripe_numbers(path, y_offset):
    info = probe_media(path)
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(out, np.uint8).reshape(-1, info.height, info.width)
    nums = []
    for img in frames:
        k = 0
        for b in range(BITS):
            if img[b * STRIPE + STRIPE // 2 - y_offset, info.width // 2] > 128:
                k |= 1 << b
        nums.append(k)
    return nums


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    """``striped``: 150 frames at 30000/1001, frame number in stripes, B-frames,
    audio. ``moving``: a textured square crossing a plain background at
    60 px/s. ``right``: the same square standing still on the right.
    ``two``: two equal squares at both sides. ``flat``: nothing at all.
    ``gradient``: grey level = column, to read where a crop sat."""
    d = tmp_path_factory.mktemp("vertical_originals")
    out = {"dir": str(d)}
    striped = str(d / "DJI_0020.MP4")
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{W}x{H}", "-r", "30000/1001", "-i", "-",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-shortest",
         "-c:a", "aac", "-c:v", "libx264", "-g", "30", "-bf", "2", "-crf", "16",
         "-pix_fmt", "yuv420p", striped], stdin=subprocess.PIPE)
    for k in range(150):
        proc.stdin.write(_stripes(k).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    out["striped"] = striped

    tex = str(d.parent / "square.png")
    _ffmpeg("-f", "lavfi", "-i",
            "mandelbrot=size=100x100:start_scale=0.3:end_scale=0.3,format=rgb24",
            "-frames:v", "1", tex)
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "14", "-pix_fmt", "yuv420p"]
    bg = ["-f", "lavfi", "-i", f"color=c=0x4a6a8a:size={W}x{H}:rate=30"]
    for name, graph in (
            ("moving", "[0][1]overlay=x='40+t*60':y=130:shortest=1"),
            ("right", "[0][1]overlay=x=500:y=130:shortest=1"),
            ("two", "[0][1]overlay=x=30:y=130[a];[a][2]overlay=x=510:y=130:shortest=1")):
        extra = ["-loop", "1", "-i", tex] * (2 if name == "two" else 1)
        out[name] = str(d / f"{name}.mp4")
        _ffmpeg(*bg, *extra, "-filter_complex", graph, "-t", "4", *enc, out[name])
    out["flat"] = str(d / "flat.mp4")
    _ffmpeg(*bg, "-t", "3", *enc, out["flat"])
    out["gradient"] = str(d / "gradient.mp4")
    _ffmpeg("-f", "lavfi", "-i", f"color=c=black:size={W}x{H}:rate=30", "-t", "3",
            "-vf", "geq=lum='X*255/W':cb=128:cr=128", *enc, out["gradient"])
    return out


@pytest.fixture
def guard(clips):
    fps = [fingerprint(v) for k, v in clips.items() if k != "dir"]
    listing = sorted(os.listdir(clips["dir"]))
    yield
    for fp in fps:
        assert_unchanged(fp)
    assert sorted(os.listdir(clips["dir"])) == listing


def test_geometry():
    assert vt.crop_size(3840, 2160) == (1206, 2144)
    assert vt.crop_size(1920, 1080) == (594, 1056)
    for w, h in ((3840, 2160), (2688, 1512), (1920, 1080), (640, 360)):
        cw, ch = vt.crop_size(w, h)
        assert cw * 16 == ch * 9 and cw % 2 == 0 and ch % 2 == 0 and ch <= h
    assert vt.output_size(1206, 2144, 1920) == (1080, 1920)
    assert vt.output_size(594, 1056, 1920) == (594, 1056)        # never upscaled
    assert vt.output_size(1206, 2144, 0) == (1206, 2144)


def test_config_validation():
    vt.VerticalConfig.from_dict({"follow": "fixed", "height": 1920})
    for bad in ({"follow": "track"}, {"height": 1000}, {"max_speed": 0}, {"zoom": 2}):
        with pytest.raises(ValueError):
            vt.VerticalConfig.from_dict(bad)


def _window_centres(path, t):
    return path.at(t) + path.crop_w / 2


def test_follows_a_moving_object_within_the_speed_limit(clips):
    m = probe_media(clips["moving"])
    cfg = vt.VerticalConfig()
    p = vt.plan(m, 0, 4, cfg)
    t = np.arange(0, 4, 0.1)
    err = np.abs(40 + t * 60 + 50 - _window_centres(p, t))
    inside = err < p.crop_w / 2 - 50                 # the whole square in the window
    assert inside.mean() >= 0.95, err.round()
    assert np.median(err) < 30, err.round()
    assert p.stats()["max_speed"] <= cfg.max_speed + 1e-6
    # The plain centre crop loses it.
    centre = vt.plan(m, 0, 4, vt.VerticalConfig(follow="center"))
    assert (np.abs(40 + t * 60 + 50 - _window_centres(centre, t)) < centre.crop_w / 2 - 50).mean() < 0.6


def test_fixed_picks_the_object_and_flat_stays_centred(clips):
    p = vt.plan(probe_media(clips["right"]), 0, 4, vt.VerticalConfig(follow="fixed"))
    assert p.stats()["travel"] == 0
    assert abs(_window_centres(p, 0.0) - 550) < p.crop_w / 2 - 50
    flat = vt.plan(probe_media(clips["flat"]), 0, 3)
    assert abs(_window_centres(flat, np.arange(0, 3, 0.1)) - W / 2).max() < 0.05 * W


def test_does_not_hop_between_two_equal_places(clips):
    p = vt.plan(probe_media(clips["two"]), 0, 4)
    assert p.stats()["travel"] < 0.05, p.stats()


def test_crop_expression_moves_frame_by_frame(clips, guard, tmp_path):
    """A hand-made path from x=40 to x=340 over 2 s, exported from 1.0 s: the
    grey level at the centre of each output frame says where the window was,
    and frame n of the clip must sit at x(n)."""
    m = probe_media(clips["gradient"])
    cw, ch = vt.crop_size(W, H)
    p = vt.CropPath(follow="smooth", src_w=W, src_h=H, crop_w=cw, crop_h=ch, out_w=cw,
                    out_h=ch, y=4, times=[0.0, 2.0], x=[40.0, 340.0])
    res = ex.export_segment(m, 1.0, 3.0, str(tmp_path), ex.ExportConfig(preset="ultrafast"),
                            video_filter=p.video_filter(30.0, 2.0), suffix="_vertical")
    assert res.frames == 60 and res.path.endswith("_vertical.mp4")
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", res.path, "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(out, np.uint8).reshape(-1, ch, cw)
    # Grey level -> source column, calibrated on the source itself (its
    # limited range clips the darkest columns, so start the window at x>=40).
    src = subprocess.run(["ffmpeg", "-v", "error", "-i", m.path, "-frames:v", "1", "-f",
                          "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True,
                         check=True).stdout
    profile = np.frombuffer(src, np.uint8).reshape(H, W)[H // 2].astype(float)
    profile = np.maximum.accumulate(profile)
    level = frames[:, ch // 2, cw // 2].astype(float)
    seen = np.interp(level, profile, np.arange(W)) - cw // 2   # left edge
    expected = 40 + 300.0 * np.arange(60) / 60
    assert np.abs(seen - expected).max() < 6, np.round(seen - expected)


def test_cli_both_formats_same_segment(clips, guard, tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(["export", clips["striped"], "--start", "1", "--end", "4",
                     "--format", "both", "--preset", "ultrafast", "--output", str(out)]) == 0
    files = sorted(os.listdir(out))
    videos = [f for f in files if f.endswith(".mp4")]
    assert len(videos) == 2, files
    horizontal = [v for v in videos if "_vertical" not in v][0]
    vertical = [v for v in videos if "_vertical" in v][0]
    assert vertical.replace("_vertical", "") == horizontal
    assert any(f.endswith("_encuadre.png") for f in files)

    rec_h = json.loads((out / horizontal.replace(".mp4", ".json")).read_text())
    rec_v = json.loads((out / vertical.replace(".mp4", ".json")).read_text())
    assert rec_h["export"]["mode"] == "copy" and rec_v["export"]["mode"] == "reencode"
    assert rec_v["twin"] == str(out / horizontal.replace(".mp4", ".json"))
    assert rec_v["vertical"]["crop_w"] == 198 and rec_v["vertical"]["crop_h"] == 352

    info = probe_media(str(out / vertical))
    assert (info.width, info.height) == (198, 352) and info.has_audio
    assert ex._video_streams(str(out / vertical)) == 1
    e = rec_v["export"]
    nums = read_stripe_numbers(str(out / vertical), rec_v["vertical"]["y"])
    assert nums == list(range(e["first_frame"], e["end_frame"])), (nums[:3], nums[-3:])
    assert "sin cambios" in capsys.readouterr().out


def test_cli_vertical_refuses_copy_and_preview_writes_only_a_png(clips, guard, tmp_path):
    out = tmp_path / "out"
    assert cli.main(["export", clips["striped"], "--start", "1", "--end", "2",
                     "--format", "vertical", "--mode", "copy", "--output", str(out)]) == 2
    assert not out.exists() or not os.listdir(out)
    assert cli.main(["vertical", clips["moving"], "--start", "0", "--end", "4",
                     "--output", str(out)]) == 0
    files = os.listdir(out)
    assert len(files) == 1 and files[0].endswith("_encuadre.png"), files


def test_10bit_reencode_keeps_every_picture(clips, guard, tmp_path):
    """x265 4.1's NEON code at 10 bits dropped a whole stripe from 30 frames of
    exactly this export (640x352) on the Pi 5; ``x265_asm: auto`` must not."""
    m = probe_media(clips["striped"])
    res = ex.export_segment(m, 1.0, 4.0, str(tmp_path), ex.ExportConfig(),
                            video_filter="crop=640:352:0:4")
    nums = read_stripe_numbers(res.path, 4)
    assert nums == list(range(res.first_frame, res.end_frame)), (nums[30:40])
    if ex.platform.machine().lower() in ("aarch64", "arm64"):
        assert "asm=none" in " ".join(res.command)
