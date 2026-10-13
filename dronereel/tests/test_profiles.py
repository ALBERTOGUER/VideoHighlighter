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
    # The Osmo Nano does not inherit the Avata 2's log profile.
    assert "dlog_m" not in builtin["osmo_nano"].color_profiles


def test_builtin_rules_start_unverified(builtin):
    for cam in builtin.values():
        assert not cam.identify.get("verified")
        for prof in cam.color_profiles.values():
            assert not prof.identify.get("verified")


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


def test_profile_is_never_borrowed_from_another_camera(builtin):
    with pytest.raises(ProfileError):
        decide(media(), builtin, camera="osmo_nano", profile="dlog_m")


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
