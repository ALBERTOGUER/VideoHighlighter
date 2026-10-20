"""F5: colour. LUT semantics checked against ffmpeg itself, adjustments by
their defining properties, and the safety rules: no LUT without a confirmed
log profile, never on the original."""
import json
import os
import subprocess

import numpy as np
import pytest

from dronereel import analyze as an
from dronereel import cli
from dronereel import color as C
from dronereel import profiles
from dronereel.media import probe_media
from dronereel.safety import assert_unchanged, fingerprint

P, G = 48, 6                                     # patch size, patches per side


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", *args], check=True)


def asym_lut(size=17) -> C.Lut:
    """out = (b, sqrt(r), g): any axis mix-up in reading or applying shows."""
    t = C.identity_lut(size).table
    return C.Lut(np.stack([t[..., 2], np.sqrt(t[..., 0]), t[..., 1]], -1))


@pytest.fixture(scope="session")
def patches(tmp_path_factory):
    """A 10-bit HEVC of flat colour patches (BT.709, limited range) and the
    patch colours as decoded."""
    d = tmp_path_factory.mktemp("color_originals")
    rng = np.random.default_rng(1)
    cols = rng.uniform(0.05, 0.95, (G * G, 3))
    img = np.zeros((G * P, G * P, 3), np.uint16)
    for i, c in enumerate(cols):
        r, cc = divmod(i, G)
        img[r * P:(r + 1) * P, cc * P:(cc + 1) * P] = np.round(c * 65535)
    raw = d / "patches.rgb"
    raw.write_bytes(np.tile(img.astype("<u2"), (25, 1, 1)).tobytes())
    video = str(d / "DJI_0020.MP4")
    _ffmpeg("-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{G * P}x{G * P}", "-r", "25",
            "-i", str(raw), "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le",
            "-c:v", "libx265", "-x265-params", "log-level=error:lossless=1",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-color_range", "tv", "-tag:v", "hvc1", video)
    raw.unlink()
    return {"dir": str(d), "video": video, "colors": decode_patches(video)}


def decode_patches(path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-vf",
                          "scale=in_color_matrix=bt709:in_range=tv:"
                          "flags=accurate_rnd+full_chroma_int,format=rgb48le",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    img = np.frombuffer(out, "<u2").reshape(G * P, G * P, 3) / 65535
    return np.array([img[r * P + P // 2, c * P + P // 2] for r in range(G) for c in range(G)])


def cameras_with(tmp_path, lut_path=None) -> dict:
    """The built-in cameras, with the Avata 2's D-Log M LUT pointing at
    ``lut_path`` through a user YAML (as a user would configure it)."""
    import yaml
    src = os.path.join(profiles.BUILTIN_DIR, "avata2.yaml")
    data = yaml.safe_load(open(src, encoding="utf-8"))
    data["color_profiles"]["dlog_m"]["lut"] = lut_path
    user = tmp_path / "profiles"
    user.mkdir(exist_ok=True)
    (user / "avata2.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(user)


# ---------------------------------------------------------------- the LUT itself

def test_cube_round_trip_and_axis_order(tmp_path):
    lut = asym_lut()
    path = str(tmp_path / "a.cube")
    C.write_cube(lut, path)
    back = C.read_cube(path)
    assert back.size == 17 and np.allclose(back.table, lut.table, atol=1e-6)
    x = np.array([[0.25, 0.5, 1.0]])
    assert np.allclose(C.apply_lut(back, x), [[1.0, 0.5, 0.5]], atol=0.02)
    with pytest.raises(FileExistsError):
        C.write_cube(lut, path)                        # never overwrites


def test_cube_rejects_broken_files(tmp_path):
    bad = tmp_path / "bad.cube"
    bad.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n")
    with pytest.raises(C.ColorError):
        C.read_cube(str(bad))
    one_d = tmp_path / "1d.cube"
    one_d.write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n")
    with pytest.raises(C.ColorError):
        C.read_cube(str(one_d))


def test_cube_domain(tmp_path):
    lut = C.identity_lut(5)
    lut.domain_max = np.array([2.0, 2.0, 2.0])
    lut.table = lut.table * 2
    path = str(tmp_path / "d.cube")
    C.write_cube(lut, path)
    back = C.read_cube(path)
    assert np.allclose(C.apply_lut(back, np.array([1.0, 0.5, 1.5])), [1.0, 0.5, 1.5], atol=1e-6)


def test_ffmpeg_applies_the_lut_like_numpy(patches, tmp_path):
    """Same LUT, same patches: ffmpeg's lut3d (tetrahedral, via the source
    matrix and range) agrees with the numpy reference to 10-bit rounding."""
    lut_path = str(tmp_path / "asym.cube")
    C.write_cube(asym_lut(33), lut_path)
    plan = C.ColorPlan(lut_path=lut_path, lut_sha256="", source_lut=None,
                       source_lut_sha256=None, profile="t", camera="t",
                       settings=C.ColorSettings(), in_matrix="bt709", in_range="tv")
    out = str(tmp_path / "out.mp4")
    _ffmpeg("-i", patches["video"], "-vf", plan.video_filter(10), "-c:v", "libx265",
            "-x265-params", "log-level=error:lossless=1", out)
    got = decode_patches(out)
    want = C.apply_lut(C.read_cube(lut_path), patches["colors"])
    assert np.abs(got - want).max() < 0.004 and np.abs(got - want).mean() < 0.001


# ---------------------------------------------------------------- adjustments

RAMP = np.linspace(0, 1, 101)[:, None] * np.ones(3)


def test_neutral_is_identity_and_compose_keeps_the_official_lut():
    assert np.array_equal(C.adjust(RAMP, C.ColorSettings()), RAMP)
    base = asym_lut(17)
    composed = C.compose(base, C.ColorSettings(), size=17)
    assert np.allclose(composed.table, base.table, atol=1e-9)


def test_exposure_is_a_gain_in_linear_light():
    out = C.adjust(np.array([[0.3, 0.3, 0.3]]), C.ColorSettings(exposure_ev=1))
    assert out[0, 0] ** C.GAMMA == pytest.approx(2 * 0.3 ** C.GAMMA, rel=1e-9)


def test_contrast_keeps_grey_and_black():
    grey = C.GREY ** (1 / C.GAMMA)
    s = C.ColorSettings(contrast=1.5)
    out = C.adjust(np.array([[grey] * 3, [0, 0, 0], [0.6] * 3, [0.2] * 3]), s)
    assert out[0, 0] == pytest.approx(grey) and out[1, 0] == 0
    assert out[2, 0] > 0.6 and out[3, 0] < 0.2


def test_saturation_zero_is_grey_and_keeps_luma():
    c = np.array([[0.8, 0.3, 0.2]])
    out = C.adjust(c, C.ColorSettings(saturation=0))
    assert np.ptp(out) < 1e-12
    assert (out[0] ** C.GAMMA) @ C.LUMA == pytest.approx((c[0] ** C.GAMMA) @ C.LUMA)


def test_temperature_warms_and_tint_moves_green():
    grey = np.array([[0.5, 0.5, 0.5]])
    warm = C.adjust(grey, C.ColorSettings(wb_temperature=50))[0]
    cool = C.adjust(grey, C.ColorSettings(wb_temperature=-50))[0]
    assert warm[0] > warm[2] and cool[0] < cool[2]
    magenta = C.adjust(grey, C.ColorSettings(wb_tint=50))[0]
    assert magenta[1] < magenta[0] and magenta[1] < magenta[2]


def test_highlight_protection_keeps_gradation_past_white():
    """+1 EV pushes the top of a ramp past white: without protection it clips
    flat; with it, it stays monotonic and distinct, and the shadows are
    unchanged."""
    hard = C.adjust(RAMP, C.ColorSettings(exposure_ev=1))[:, 0]
    soft = C.adjust(RAMP, C.ColorSettings(exposure_ev=1, highlight_protection=1))[:, 0]
    assert np.sum(hard >= 1 - 1e-9) > 20
    assert np.sum(soft >= 1 - 1e-6) <= 1
    assert np.all(np.diff(soft) > 0) and soft.max() <= 1
    assert np.allclose(soft[:40], hard[:40])


def test_settings_validation():
    with pytest.raises(C.ColorError):
        C.ColorSettings.from_dict({"exposure_ev": 5})
    with pytest.raises(C.ColorError):
        C.ColorSettings.from_dict({"vibrance": 1})
    assert C.ColorSettings.from_dict({"contrast": "1.2"}).contrast == 1.2


# ---------------------------------------------------------------- safety

def test_no_lut_unless_the_log_profile_is_confirmed(patches, tmp_path):
    lut_path = str(tmp_path / "x.cube")
    C.write_cube(asym_lut(), lut_path)
    media = probe_media(patches["video"])
    cams = profiles.load_cameras(cameras_with(tmp_path, lut_path))
    cache = str(tmp_path / "luts")
    unknown = profiles.decide(media, cams)
    assert C.plan(media, unknown, None, C.ColorSettings(), cache_dir=cache) is None
    assert C.plan(media, unknown, None, C.ColorSettings(exposure_ev=1), cache_dir=cache) is None
    with pytest.raises(C.ColorError):
        C.plan(media, unknown, None, C.ColorSettings(), lut_override=lut_path, cache_dir=cache)
    normal = profiles.decide(media, cams, camera="avata2", profile="normal")
    assert C.plan(media, normal, cams["avata2"], C.ColorSettings(), cache_dir=cache) is None
    with pytest.raises(C.ColorError):                 # a log LUT on a non-log profile
        C.plan(media, normal, cams["avata2"], C.ColorSettings(), lut_override=lut_path,
               cache_dir=cache)
    adj = C.plan(media, normal, cams["avata2"], C.ColorSettings(saturation=1.2), cache_dir=cache)
    assert adj is not None and adj.source_lut is None
    dlog = profiles.decide(media, cams, camera="avata2", profile="dlog_m")
    p = C.plan(media, dlog, cams["avata2"], C.ColorSettings(), cache_dir=cache)
    assert p.source_lut == lut_path and p.lut_sha256 == C.sha256_file(lut_path)
    assert os.path.dirname(p.lut_path) == cache       # safe copy, same content


def test_log_profile_without_lut_is_an_error(patches, tmp_path):
    media = probe_media(patches["video"])
    cams = profiles.load_cameras(cameras_with(tmp_path, None))
    dlog = profiles.decide(media, cams, camera="avata2", profile="dlog_m")
    with pytest.raises(C.ColorError, match="LUT oficial"):
        C.plan(media, dlog, cams["avata2"], C.ColorSettings(), cache_dir=str(tmp_path / "l"))


def test_osmo_nano_never_gets_the_avata_lut(patches, tmp_path):
    media = probe_media(patches["video"])
    cams = profiles.load_cameras(cameras_with(tmp_path, str(tmp_path / "x.cube")))
    with pytest.raises(profiles.ProfileError):
        profiles.decide(media, cams, camera="osmo_nano", profile="dlog_m")


# ---------------------------------------------------------------- end to end

def test_cli_export_with_color(patches, tmp_path, monkeypatch):
    monkeypatch.setattr(an, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(C, "LUT_CACHE_DIR", str(tmp_path / "luts"))
    lut_path = str(tmp_path / "official.cube")
    C.write_cube(asym_lut(33), lut_path)
    user = cameras_with(tmp_path, lut_path)
    fp = fingerprint(patches["video"])
    listing = sorted(os.listdir(patches["dir"]))
    out = tmp_path / "out"
    assert cli.main(["--profiles-dir", user, "export", patches["video"], "--start", "0.2",
                     "--end", "0.6", "--camera", "avata2", "--profile", "dlog_m",
                     "--preset", "ultrafast", "--output", str(out)]) == 0
    files = sorted(os.listdir(out))
    (video,) = [f for f in files if f.endswith(".mp4")]
    assert any(f.endswith("_comparacion.png") for f in files)
    rec = json.loads((out / video.replace(".mp4", ".json")).read_text())
    assert rec["color"]["applied"] is True and rec["export"]["mode"] == "reencode"
    assert rec["color"]["source_lut_sha256"] == C.sha256_file(lut_path)
    got = decode_patches(str(out / video))
    want = C.apply_lut(C.read_cube(lut_path), patches["colors"])
    assert np.abs(got - want).mean() < 0.01           # lossy x265 this time
    tags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                           "-show_entries", "stream=color_transfer,color_primaries,pix_fmt",
                           "-of", "csv=p=0", str(out / video)],
                          capture_output=True, text=True).stdout
    assert "bt709" in tags and "yuv420p10le" in tags
    assert_unchanged(fp)
    assert sorted(os.listdir(patches["dir"])) == listing


def test_cli_refuses_color_with_copy_and_lut_without_profile(patches, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "LUT_CACHE_DIR", str(tmp_path / "luts"))
    lut_path = str(tmp_path / "official.cube")
    C.write_cube(asym_lut(), lut_path)
    user = cameras_with(tmp_path, lut_path)
    base = ["--profiles-dir", user, "export", patches["video"], "--start", "0", "--end", "0.4",
            "--output", str(tmp_path / "out")]
    assert cli.main([*base, "--camera", "avata2", "--profile", "dlog_m", "--mode", "copy"]) == 2
    assert cli.main([*base, "--lut", lut_path]) == 2
    assert not (tmp_path / "out").exists() or not os.listdir(tmp_path / "out")
    # --no-color: a plain copy, original colour.
    assert cli.main([*base, "--camera", "avata2", "--profile", "dlog_m", "--no-color"]) == 0


def test_cli_color_writes_comparisons_only(patches, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "LUT_CACHE_DIR", str(tmp_path / "luts"))
    lut_path = str(tmp_path / "official.cube")
    C.write_cube(asym_lut(), lut_path)
    user = cameras_with(tmp_path, lut_path)
    out = tmp_path / "cmp"
    assert cli.main(["--profiles-dir", user, "color", patches["video"], "--camera", "avata2",
                     "--profile", "dlog_m", "--at", "0.5", "--exposure", "0.5",
                     "--output", str(out)]) == 0
    (png,) = os.listdir(out)
    assert png.endswith(".png")
    w = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(out / png)], capture_output=True, text=True).stdout
    assert w.strip().startswith("1920,")
