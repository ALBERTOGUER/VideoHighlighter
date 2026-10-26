"""Color LUTs applied when clips are cut.

Cameras and drones record in a log profile (D-Log, S-Log, V-Log…) so the grade
has room later; straight out of the camera that footage is flat and grey. The
maker ships a 3D LUT that maps it to Rec.709, and ffmpeg's ``lut3d`` filter
applies it. ``cut_video`` already re-encodes every clip, so applying it there
costs nothing extra and the reel — a stream-copy concat of those clips —
inherits it.

The LUT is the user's file, picked at runtime; nothing here knows about any
particular camera.
"""

from __future__ import annotations

import os

# The 3D LUT formats ffmpeg's lut3d filter reads.
LUT_SUFFIXES = (".cube", ".3dl", ".dat", ".m3d", ".csp")


def _escape(text: str, chars: str) -> str:
    return "".join("\\" + c if c in chars else c for c in text)


def lut_filter(path: str) -> str:
    """The ``lut3d`` filter for ``path``, escaped for a ``-vf`` chain.

    A filter graph is parsed twice — once for the option value, once for the
    graph — so a path carrying ``:`` (a Windows drive), ``'``, ``,`` or brackets
    is escaped for both levels, as ffmpeg's filter documentation prescribes."""
    value = _escape(os.path.abspath(path), "\\':")
    return f"lut3d=file={_escape(value, chr(92) + chr(39) + '[],;')}:interp=tetrahedral"


def resolve_lut(path, log=print):
    """The LUT path to use, or None — with a log line saying why not.

    A missing or unreadable LUT must not cost a run its highlight: the clips are
    written ungraded and the log says so, rather than every cut failing."""
    if not path:
        return None
    path = os.path.expanduser(str(path).strip())
    if not path:
        return None
    if not path.lower().endswith(LUT_SUFFIXES):
        log(f"⚠️ Color LUT ignored — not a LUT file ({', '.join(LUT_SUFFIXES)}): {path}")
        return None
    if not os.path.isfile(path):
        log(f"⚠️ Color LUT ignored — file not found: {path}")
        return None
    return path
