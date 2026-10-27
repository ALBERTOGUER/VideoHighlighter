"""The colour mode a DJI camera says it recorded in.

Recent DJI cameras (dvtm protobuf metadata: Osmo Nano ``dvtm_ow001``, Osmo
360, Action 4/5, Avata 2…) write a ``djmd`` data track whose first packet is a
``ProductMeta`` message. Its ``StreamMeta`` (field 2) carries ``ColorMode``
(field 4), whose ``color_mode`` (field 1) is DJI's own enum from
``dvtm_library.proto`` — 19 is D-Log M, 2 is D-Log, 0 the default colour.
That is the camera stating its profile, which DroneReel can treat as
evidence, unlike file tags that never name it.

Read with the ffmpeg already required (one packet, ~0.05 s) and a minimal
protobuf wire-format reader, so the Pi needs nothing new.
"""

from __future__ import annotations

import subprocess
from typing import Optional

from dronereel import tools

# dvtm_library.proto, ColorMode.ColorModeType — the values cameras use.
COLOR_MODES = {
    0: "default", 1: "D-Cinelike", 2: "D-Log", 3: "film A", 4: "film B", 5: "film C",
    6: "film D", 7: "film E", 8: "film F", 9: "HLG", 10: "art", 11: "black & white",
    12: "vivid", 13: "beach", 14: "dream", 15: "sRGB", 16: "Adobe RGB", 17: "IR cut",
    18: "racing", 19: "D-Log M", 20: "laser fill light",
}

_PRODUCT_STREAM_META = 2
_STREAM_COLOR_MODE = 4
_COLOR_MODE_VALUE = 1


def _varint(buf: bytes, i: int) -> tuple:
    value = shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated varint")
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            return value, i


def fields(buf: bytes) -> list:
    """``[(field, wire_type, value)]`` of one protobuf message, schema-less:
    varints as ints, length-delimited as bytes."""
    i, out = 0, []
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            value, i = buf[i:i + n], i + n
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.append((field, wire, value))
    return out


def _first(buf: bytes, field: int, wire: int):
    return next((v for f, w, v in fields(buf) if f == field and w == wire), None)


def color_mode_from_packet(packet: bytes) -> Optional[int]:
    """``color_mode`` from a ProductMeta packet, or None when it has none."""
    try:
        stream = _first(packet, _PRODUCT_STREAM_META, 2)
        mode = stream and _first(stream, _STREAM_COLOR_MODE, 2)
        return None if mode is None else _first(mode, _COLOR_MODE_VALUE, 0) or 0
    except (ValueError, IndexError):
        return None


def first_meta_packet(path: str) -> bytes:
    """The first packet of the file's first data track (b"" when none)."""
    out = subprocess.run([tools.ffmpeg(), "-v", "error", "-nostdin", "-i", path,
                          "-map", "0:d:0", "-frames:d", "1", "-c", "copy", "-f", "data", "-"],
                         capture_output=True, timeout=60)
    return out.stdout if out.returncode == 0 else b""


def color_mode(path: str) -> Optional[int]:
    """The DJI colour mode ``path`` was recorded in, or None if it does not say."""
    try:
        packet = first_meta_packet(path)
    except (OSError, subprocess.SubprocessError):
        return None
    return color_mode_from_packet(packet) if packet else None
