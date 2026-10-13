import os
import shutil

from dronereel.media import bit_depth_of, find_sidecar_srt, probe_media, srt_color_modes
from dronereel.safety import fingerprint, assert_unchanged


def test_probe_h264_with_audio(videos):
    m = probe_media(videos["h264_audio"])
    assert (m.width, m.height) == (320, 240)
    assert m.fps_fraction == "25/1"
    assert abs(m.duration - 2.0) < 0.2
    assert m.video_codec == "h264"
    assert m.bit_depth == 8
    assert m.color_transfer == "bt709"
    assert m.has_audio and m.audio_codec == "aac"
    assert m.srt_path is None and m.srt_color_modes == []


def test_probe_hevc_10bit_keeps_exact_fraction(videos):
    m = probe_media(videos["hevc10"])
    assert m.video_codec == "hevc"
    assert m.bit_depth == 10
    assert m.fps_fraction == "30000/1001"
    assert abs(m.fps - 29.97) < 0.01
    assert not m.has_audio


def test_probe_leaves_original_untouched(videos):
    fp = fingerprint(videos["hevc10"])
    probe_media(videos["hevc10"])
    assert_unchanged(fp)


def test_bit_depth_from_pix_fmt_when_ffprobe_is_silent():
    assert bit_depth_of({"pix_fmt": "yuv420p10le"}) == 10
    assert bit_depth_of({"pix_fmt": "yuv422p12be"}) == 12
    assert bit_depth_of({"pix_fmt": "yuv420p"}) == 8
    assert bit_depth_of({"bits_per_raw_sample": "10", "pix_fmt": "yuv420p"}) == 10


def test_sidecar_srt_found_case_insensitively(tmp_path):
    video = tmp_path / "DJI_0007.MP4"
    video.write_bytes(b"")
    srt = tmp_path / "dji_0007.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,033\n[iso : 100] [color_md : dlog_m]\n"
                   "2\n00:00:00,033 --> 00:00:00,066\n[iso : 100] [color_md : dlog_m]\n")
    assert find_sidecar_srt(str(video)) == str(srt)
    assert srt_color_modes(str(srt)) == ["dlog_m"]


def test_probe_reads_sidecar_srt(videos, tmp_path):
    video = tmp_path / "DJI_0003.MP4"
    shutil.copy(videos["hevc10"], video)
    (tmp_path / "DJI_0003.SRT").write_text("[color_md : default]\n")
    m = probe_media(str(video))
    assert os.path.basename(m.srt_path) == "DJI_0003.SRT"
    assert m.srt_color_modes == ["default"]
