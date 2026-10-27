"""The colour mode DJI cameras write in their djmd metadata, and the Osmo
Nano's verified D-Log M rule built on it.

The packets here are built by hand in protobuf's wire format with the same
layout the Osmo Nano writes (ProductMeta → StreamMeta → ColorMode); the real
clips this was checked against are not in the repository.
"""
import dataclasses

import pytest

from dronereel import dji_meta
from dronereel.profiles import UNKNOWN, decide, load_cameras, rule_matches
from dronereel.tests.test_profiles import media


def _varint(n):
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num, value):
    if isinstance(value, int):
        return _varint(num << 3) + _varint(value)
    return _varint(num << 3 | 2) + _varint(len(value)) + value


def product_meta(mode=None, with_stream=True):
    header = _field(1, _field(1, b"dvtm_ow001.proto") + _field(9, b"DJI Osmo Nano"))
    stream = _field(1, b"\x08\x01") + _field(3, b"\x08\x0a")
    if mode is not None:
        stream += _field(4, _field(1, mode) if mode else b"")    # proto3 omits 0
    packet = _field(1, header)
    if with_stream:
        packet += _field(2, stream)
    return packet + _field(3, b"\x08\x01")


@pytest.mark.parametrize("mode", [19, 2, 9])
def test_the_colour_mode_is_read_from_stream_meta(mode):
    assert dji_meta.color_mode_from_packet(product_meta(mode)) == mode


def test_an_omitted_value_is_the_default_mode():
    assert dji_meta.color_mode_from_packet(product_meta(0)) == 0


def test_no_colour_mode_means_none():
    assert dji_meta.color_mode_from_packet(product_meta(None)) is None
    assert dji_meta.color_mode_from_packet(product_meta(19, with_stream=False)) is None
    assert dji_meta.color_mode_from_packet(b"\xff\xff\xff") is None
    assert dji_meta.color_mode_from_packet(b"") is None


def test_the_enum_names_dlog_m():
    assert dji_meta.COLOR_MODES[19] == "D-Log M" and dji_meta.COLOR_MODES[2] == "D-Log"


def test_a_file_without_a_data_track_says_nothing(videos):
    assert dji_meta.color_mode(videos["h264_audio"]) is None


# ── the Osmo Nano rule ──────────────────────────────────────────────────────

NANO_TAGS = {"encoder": "DJI Osmo Nano"}


def nano(mode, tags=NANO_TAGS):
    return dataclasses.replace(media(tags=tags), dji_color_mode=mode)


def test_rule_matches_the_stated_colour_mode():
    assert rule_matches({"dji_color_mode": [19]}, nano(19))
    assert not rule_matches({"dji_color_mode": [19]}, nano(0))
    assert not rule_matches({"dji_color_mode": [19]}, nano(None))
    assert rule_matches({"dji_color_mode": 19}, nano(19))      # a bare value works too


def test_nano_dlog_m_footage_is_confirmed_by_its_own_metadata():
    d = decide(nano(19), load_cameras())
    assert (d.camera, d.camera_source) == ("osmo_nano", "verified_rule")
    assert (d.profile, d.profile_source) == ("dlog_m", "verified_rule")
    assert d.apply_color and d.transform == "lut"
    assert any("color_mode=19 (D-Log M)" in e for e in d.evidence)


@pytest.mark.parametrize("mode", [0, 2, 12, None])
def test_nano_footage_in_any_other_mode_keeps_its_colour(mode):
    d = decide(nano(mode), load_cameras())
    assert d.camera == "osmo_nano"
    assert d.profile == UNKNOWN and not d.apply_color


def test_mode_19_from_another_camera_does_not_make_it_a_nano():
    d = decide(nano(19, tags={"encoder": "DJI Avata 2"}), load_cameras())
    assert d.camera is None and not d.apply_color
