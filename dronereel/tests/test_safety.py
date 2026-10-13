import os

import pytest

from dronereel.safety import (OriginalModifiedError, UnsafeOutputError, assert_unchanged,
                              fingerprint, output_path)


def test_fingerprint_detects_content_change_even_with_same_size_and_mtime(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"0123456789")
    fp = fingerprint(str(f))
    st = os.stat(f)
    f.write_bytes(b"9876543210")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(OriginalModifiedError):
        assert_unchanged(fp)


def test_fingerprint_detects_deletion(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    fp = fingerprint(str(f))
    f.unlink()
    with pytest.raises(OriginalModifiedError):
        assert_unchanged(fp)


def test_unchanged_file_passes(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x" * 10_000)
    assert_unchanged(fingerprint(str(f)))


def test_output_never_in_the_originals_folder(tmp_path):
    src = tmp_path / "DJI_0001.MP4"
    src.write_bytes(b"x")
    with pytest.raises(UnsafeOutputError):
        output_path(str(tmp_path), "clip.mp4", [str(src)])


def test_output_never_overwrites(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "DJI_0001.MP4"
    src.write_bytes(b"x")
    out = tmp_path / "out"
    first = output_path(str(out), "clip.mp4", [str(src)])
    open(first, "wb").close()
    second = output_path(str(out), "clip.mp4", [str(src)])
    assert first != second and second.endswith("clip_1.mp4")
    assert not os.path.exists(second)


def test_output_rejects_paths_in_the_name(tmp_path):
    for bad in ("../DJI_0001.MP4", "sub/clip.mp4", "", ".."):
        with pytest.raises(UnsafeOutputError):
            output_path(str(tmp_path / "out"), bad, [])


def test_output_folder_reached_through_a_symlink_is_still_refused(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "DJI_0001.MP4"
    src.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(src_dir)
    with pytest.raises(UnsafeOutputError):
        output_path(str(link), "clip.mp4", [str(src)])
