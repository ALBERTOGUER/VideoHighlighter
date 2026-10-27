import pytest

from dronereel import profiles
from dronereel.media import MediaInfo
from dronereel.profiles import ProfileError, UNKNOWN, decide, load_cameras


def media(bit_depth=10, srt=(), tags=None):
    return MediaInfo(
        path="/x/DJI_0001.MP4", duration=10.0, width=3840, height=2160, fps=60.0,
        fps_fraction="60/1", video_codec="hevc",
        pix_fmt="yuv420p10le" if bit_depth == 10 else "yuv420p", bit_depth=bit_depth,
        color_primaries="bt709", color_transfer="bt709", color_space="bt709",
        color_range="tv", rotation=0, has_audio=False, audio_codec=None,
        tags=dict(tags or {}), srt_path="/x/DJI_0001.SRT" if srt else None,
        srt_color_modes=list(srt))


VERIFIED_CAM = """
camera: testcam
name: Test Cam
identify: {verified: true, tags: {encoder: "^TESTCAM"}}
color_profiles:
  normal: {transform: none}
  dlog_m:
    transform: lut
    lut: null
    identify: {verified: true, srt_color_md: [dlog_m]}
"""


@pytest.fixture
def builtin():
    return load_cameras()


def test_builtin_cameras_load_with_their_own_profiles(builtin):
    assert set(builtin) >= {"avata2", "osmo_nano"}
    assert set(builtin["avata2"].color_profiles) == {"normal", "dlog_m"}
    # The Osmo Nano has its own D-Log M, never the Avata 2's: its own rule, and
    # no LUT shipped (each camera's official LUT is configured per camera).
    nano = builtin["osmo_nano"].color_profiles
    assert set(nano) == {"normal", "dlog_m"}
    assert nano["dlog_m"].identify["dji_color_mode"] == [19]
    assert nano["dlog_m"].settings.get("lut") is None


def test_only_rules_checked_on_real_clips_are_verified(builtin):
    # Avata 2: no real clips yet, so nothing it says is acted on.
    avata = builtin["avata2"]
    assert not avata.identify.get("verified")
    assert not any(p.identify.get("verified") for p in avata.color_profiles.values())
    # Osmo Nano: checked on three real clips (encoder tag, colour mode 19).
    nano = builtin["osmo_nano"]
    assert nano.identify.get("verified") and nano.identify["tags"]["encoder"]
    assert nano.color_profiles["dlog_m"].identify.get("verified")
    assert not nano.color_profiles["normal"].identify


def test_no_evidence_means_unknown_and_no_colour(builtin):
    d = decide(media(), builtin)
    assert d.camera is None and d.profile == UNKNOWN
    assert not d.confirmed and not d.apply_color


def test_unverified_srt_hint_never_confirms(builtin):
    d = decide(media(srt=["dlog_m"]), builtin, camera="avata2")
    assert d.profile == UNKNOWN and not d.apply_color
    assert any("dlog_m" in h for h in d.hints)


def test_user_confirmation_enables_transform(builtin):
    d = decide(media(), builtin, camera="avata2", profile="dlog_m")
    assert d.confirmed and d.apply_color and d.transform == "lut"
    assert d.profile_source == "user"


def test_user_confirmed_normal_keeps_colour(builtin):
    d = decide(media(), builtin, camera="avata2", profile="normal")
    assert d.confirmed and not d.apply_color


def test_profile_without_camera_is_refused(builtin):
    with pytest.raises(ProfileError):
        decide(media(), builtin, profile="dlog_m")


def test_profile_is_never_borrowed_from_another_camera(tmp_path):
    (tmp_path / "logcam.yaml").write_text(
        "camera: logcam\ncolor_profiles:\n  dlog:\n    transform: lut\n")
    cams = load_cameras(str(tmp_path))
    with pytest.raises(ProfileError):                 # D-Log exists, but not on the Nano
        decide(media(), cams, camera="osmo_nano", profile="dlog")


def test_unknown_camera_is_refused(builtin):
    with pytest.raises(ProfileError):
        decide(media(), builtin, camera="mavic")


def test_log_profile_on_8_bit_file_warns(builtin):
    d = decide(media(bit_depth=8), builtin, camera="avata2", profile="dlog_m")
    assert any("8 bits" in w for w in d.warnings)


def test_verified_rules_confirm_camera_and_profile(tmp_path):
    (tmp_path / "testcam.yaml").write_text(VERIFIED_CAM)
    cams = load_cameras(str(tmp_path))
    d = decide(media(srt=["dlog_m"], tags={"encoder": "TESTCAM v1"}), cams)
    assert (d.camera, d.camera_source) == ("testcam", "verified_rule")
    assert (d.profile, d.profile_source) == ("dlog_m", "verified_rule")
    assert d.apply_color


def test_verified_profile_rule_needs_its_evidence(tmp_path):
    (tmp_path / "testcam.yaml").write_text(VERIFIED_CAM)
    cams = load_cameras(str(tmp_path))
    d = decide(media(srt=["default"], tags={"encoder": "TESTCAM v1"}), cams)
    assert d.camera == "testcam" and d.profile == UNKNOWN and not d.apply_color


def test_empty_rule_matches_nothing():
    assert not profiles.rule_matches({"verified": True}, media())


def test_user_folder_overrides_builtin(tmp_path):
    (tmp_path / "avata2.yaml").write_text(
        "camera: avata2\nname: Mine\ncolor_profiles:\n  normal: {transform: none}\n")
    cams = load_cameras(str(tmp_path))
    assert cams["avata2"].name == "Mine"
    assert set(cams["avata2"].color_profiles) == {"normal"}


def test_unknown_transform_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "camera: bad\ncolor_profiles:\n  x: {transform: magic}\n")
    with pytest.raises(ProfileError):
        load_cameras(str(tmp_path))
