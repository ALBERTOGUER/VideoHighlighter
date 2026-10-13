"""Originals are never written, and results never land on top of anything.

Every command that touches a source video takes a :class:`Fingerprint` first
(size, mtime and SHA-256, read in chunks) and calls :func:`assert_unchanged`
when it is done, so a bug that writes to an original fails loudly instead of
passing quietly. Sources are only ever opened for reading.

Results go through :func:`output_path`: a folder that is not the source's own
folder, and a file name that does not exist yet — an existing file gets a
numbered sibling, never an overwrite.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable

_CHUNK = 4 * 1024 * 1024


class OriginalModifiedError(RuntimeError):
    """A source video changed while DroneReel was working on it."""


class UnsafeOutputError(ValueError):
    """An output location that could overwrite or mix with the originals."""


@dataclass(frozen=True)
class Fingerprint:
    path: str
    size: int
    mtime_ns: int
    sha256: str


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: str) -> Fingerprint:
    path = os.path.realpath(path)
    st = os.stat(path)
    return Fingerprint(path, st.st_size, st.st_mtime_ns, sha256_file(path))


def assert_unchanged(fp: Fingerprint) -> None:
    """Raise unless the file still has the size, mtime and content it had."""
    try:
        st = os.stat(fp.path)
    except FileNotFoundError:
        raise OriginalModifiedError(f"El original ha desaparecido: {fp.path}") from None
    if st.st_size != fp.size or st.st_mtime_ns != fp.mtime_ns:
        raise OriginalModifiedError(f"El original ha cambiado: {fp.path}")
    if sha256_file(fp.path) != fp.sha256:
        raise OriginalModifiedError(f"El contenido del original ha cambiado: {fp.path}")


def output_path(output_dir: str, filename: str, originals: Iterable[str]) -> str:
    """A fresh path for ``filename`` in ``output_dir``; creates the folder.

    Refuses an output folder that is the folder of any original (results
    belong in a separate folder) and a file name with a directory part.
    """
    if os.path.basename(filename) != filename or filename in ("", ".", ".."):
        raise UnsafeOutputError(f"Nombre de archivo no válido: {filename!r}")
    out_dir = os.path.realpath(os.path.expanduser(output_dir))
    originals = [os.path.realpath(p) for p in originals]
    for src in originals:
        if os.path.dirname(src) == out_dir:
            raise UnsafeOutputError(
                f"La carpeta de salida es la misma que la del original ({out_dir}); "
                "usa una carpeta independiente.")
    os.makedirs(out_dir, exist_ok=True)

    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(out_dir, filename)
    n = 1
    while os.path.lexists(candidate) or os.path.realpath(candidate) in originals:
        candidate = os.path.join(out_dir, f"{stem}_{n}{ext}")
        n += 1
    return candidate
