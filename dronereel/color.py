"""Colour: log to Rec.709 with the camera's official LUT, plus a few
deterministic adjustments. Applied to the exported segment, never the source.

How
===
Everything becomes ONE 3D LUT, applied by ffmpeg's ``lut3d`` (tetrahedral)
on 16-bit RGB in a single pass:

    source YUV --(source matrix and range)--> RGB, still log-encoded
      --> official LUT (log -> Rec.709)          [only for a log profile]
      --> adjustments in linear light            [only when not neutral]
      --> Rec.709 YUV, tagged bt709

With neutral adjustments the official ``.cube`` is used byte for byte. With
adjustments, a 65-point LUT is composed here in numpy (the official LUT
sampled with trilinear interpolation, then the adjustments) and cached under
``~/.cache/dronereel/luts/`` by content hash. A fixed LUT is the same function
on every frame: nothing adapts over time, so nothing can flicker. No
generative model is involved.

The adjustments work on display-referred Rec.709 (BT.1886, gamma 2.4) made
linear, in this order: white balance (channel gains, luma kept), exposure
(``2**ev``), contrast (a power around 18 % grey, so grey and black stay put),
saturation (around Rec.709 luma), highlight protection (a soft shoulder from
a knee to white instead of a hard clip).

Limits, said plainly
====================
* The official LUT already decides how the top of the log range maps to
  white; highlight protection can only soften what is at or pushed past
  white after it, not recover detail the LUT clipped.
* White balance is a channel-gain approximation, not a chromatic-adaptation
  transform; fine for small corrections.
* Whether DJI's LUT expects full-range RGB from limited-range video (the
  usual convention, assumed here) must be checked against a real D-Log M
  clip and the LUT's look in DJI's own software.

Safety: :func:`plan` returns nothing to do unless the profile is confirmed
(see :mod:`dronereel.profiles`); a log profile without its LUT is an error,
never a silent guess.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from dronereel.media import MediaInfo
from dronereel.profiles import NO_TRANSFORM, Camera, ColorDecision

LUT_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "dronereel", "luts")
COMPOSED_SIZE = 65
GAMMA = 2.4                          # BT.1886 with a black level of 0
GREY = 0.18
LUMA = np.array([0.2126, 0.7152, 0.0722])
TEMPERATURE_STOPS = 0.3              # R/B gain at ±100, in stops
TINT_STOPS = 0.2                     # G gain at ±100, in stops
MAX_KNEE_DROP = 0.5                  # knee = 1 - 0.5 * highlight_protection

# Source matrix names (ffprobe) -> swscale's in_color_matrix.
_MATRIX = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601",
           "bt2020nc": "bt2020", "bt2020c": "bt2020", "smpte240m": "smpte240m",
           "fcc": "fcc"}


class ColorError(ValueError):
    pass


# ---------------------------------------------------------------- settings

_RANGES = {"exposure_ev": (-3.0, 3.0, 0.0), "contrast": (0.5, 2.0, 1.0),
           "saturation": (0.0, 2.0, 1.0), "wb_temperature": (-100.0, 100.0, 0.0),
           "wb_tint": (-100.0, 100.0, 0.0), "highlight_protection": (0.0, 1.0, 0.0)}


@dataclass
class ColorSettings:
    exposure_ev: float = 0.0             # stops
    contrast: float = 1.0                # 1 = unchanged
    saturation: float = 1.0              # 0 = grey, 1 = unchanged
    wb_temperature: float = 0.0          # -100 cooler .. +100 warmer
    wb_tint: float = 0.0                 # -100 green .. +100 magenta
    highlight_protection: float = 0.0    # 0 = hard clip at white .. 1 = longest shoulder

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ColorSettings":
        d = dict(d or {})
        unknown = set(d) - set(_RANGES)
        if unknown:
            raise ColorError(f"Ajustes de color desconocidos: {sorted(unknown)}")
        values = {}
        for k, v in d.items():
            lo, hi, _ = _RANGES[k]
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ColorError(f"{k} debe ser un número") from None
            if not lo <= v <= hi:
                raise ColorError(f"{k}={v:g} fuera de rango ({lo:g} a {hi:g})")
            values[k] = v
        return cls(**values)

    @property
    def neutral(self) -> bool:
        return all(getattr(self, k) == default for k, (_, _, default) in _RANGES.items())


def adjust(rgb: np.ndarray, s: ColorSettings) -> np.ndarray:
    """Apply ``s`` to Rec.709 display-referred RGB in 0..1 (any shape ``...x3``)."""
    v = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    if s.neutral:
        return v
    lin = v ** GAMMA

    gains = np.array([2.0 ** (s.wb_temperature / 100 * TEMPERATURE_STOPS),
                      2.0 ** (-s.wb_tint / 100 * TINT_STOPS),
                      2.0 ** (-s.wb_temperature / 100 * TEMPERATURE_STOPS)])
    lin = lin * (gains / float(LUMA @ gains))
    lin = lin * 2.0 ** s.exposure_ev
    if s.contrast != 1.0:
        lin = GREY * (lin / GREY) ** s.contrast
    if s.saturation != 1.0:
        y = (lin @ LUMA)[..., None]
        lin = np.maximum(y + s.saturation * (lin - y), 0.0)
    lin = shoulder(lin, s.highlight_protection)
    return np.clip(lin, 0.0, 1.0) ** (1.0 / GAMMA)


def shoulder(lin: np.ndarray, protection: float) -> np.ndarray:
    """Identity below the knee, then an exponential roll-off that meets white
    asymptotically with slope 1 at the knee. ``protection=0``: no shoulder."""
    if protection <= 0:
        return lin
    k = 1.0 - MAX_KNEE_DROP * protection
    over = np.maximum(lin - k, 0.0)
    return np.where(lin > k, k + (1.0 - k) * (1.0 - np.exp(-over / (1.0 - k))), lin)


# ---------------------------------------------------------------- .cube files

@dataclass
class Lut:
    table: np.ndarray                    # (N, N, N, 3) indexed [b, g, r]
    domain_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    domain_max: np.ndarray = field(default_factory=lambda: np.ones(3))
    title: str = ""

    @property
    def size(self) -> int:
        return self.table.shape[0]


def identity_lut(size: int) -> Lut:
    g = np.linspace(0.0, 1.0, size)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    return Lut(np.stack([r, gg, b], axis=-1))


def read_cube(path: str) -> Lut:
    """A 3D ``.cube`` (Adobe/Resolve). Red varies fastest in the data."""
    size, title = None, ""
    dmin, dmax = np.zeros(3), np.ones(3)
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key = line.split()[0].upper()
            if key == "TITLE":
                title = line[5:].strip().strip('"')
            elif key == "LUT_3D_SIZE":
                size = int(line.split()[1])
            elif key == "LUT_1D_SIZE":
                raise ColorError(f"{path}: es una LUT 1D; se necesita una LUT 3D")
            elif key == "DOMAIN_MIN":
                dmin = np.array([float(x) for x in line.split()[1:4]])
            elif key == "DOMAIN_MAX":
                dmax = np.array([float(x) for x in line.split()[1:4]])
            elif re.match(r"^[-+0-9.eE]", key):
                rows.append([float(x) for x in line.split()[:3]])
            # LUT_3D_INPUT_RANGE and other keywords: ignored.
    if not size:
        raise ColorError(f"{path}: falta LUT_3D_SIZE")
    if len(rows) != size ** 3:
        raise ColorError(f"{path}: {len(rows)} filas; se esperaban {size ** 3}")
    if np.any(dmax <= dmin):
        raise ColorError(f"{path}: DOMAIN_MIN/MAX no válidos")
    table = np.array(rows, dtype=np.float64).reshape(size, size, size, 3)
    return Lut(table, dmin, dmax, title)


def write_cube(lut: Lut, path: str) -> None:
    n = lut.size
    with open(path, "x", encoding="ascii") as f:
        if lut.title:
            f.write(f'TITLE "{lut.title}"\n')
        f.write(f"LUT_3D_SIZE {n}\n")
        f.write("DOMAIN_MIN {:.6f} {:.6f} {:.6f}\n".format(*lut.domain_min))
        f.write("DOMAIN_MAX {:.6f} {:.6f} {:.6f}\n".format(*lut.domain_max))
        np.savetxt(f, lut.table.reshape(-1, 3), fmt="%.6f")


def apply_lut(lut: Lut, rgb: np.ndarray) -> np.ndarray:
    """Trilinear lookup of ``rgb`` (``...x3``) — for composing and testing."""
    x = (np.asarray(rgb, dtype=np.float64) - lut.domain_min) / (lut.domain_max - lut.domain_min)
    n = lut.size
    p = np.clip(x, 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(p).astype(np.int64), n - 2)
    f = p - i0
    t = lut.table
    out = np.zeros(p.shape, dtype=np.float64)
    for db in (0, 1):
        wb = f[..., 2] if db else 1 - f[..., 2]
        for dg in (0, 1):
            wg = f[..., 1] if dg else 1 - f[..., 1]
            for dr in (0, 1):
                wr = f[..., 0] if dr else 1 - f[..., 0]
                w = (wb * wg * wr)[..., None]
                out += w * t[i0[..., 2] + db, i0[..., 1] + dg, i0[..., 0] + dr]
    return out


def compose(base: Optional[Lut], settings: ColorSettings,
            size: int = COMPOSED_SIZE) -> Lut:
    """One LUT: ``base`` (or nothing) followed by ``settings``."""
    grid = identity_lut(size).table
    out = apply_lut(base, grid) if base is not None else grid
    return Lut(adjust(out, settings), title="DroneReel composed LUT")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- planning

@dataclass
class ColorPlan:
    lut_path: str                       # what ffmpeg reads (cache copy, safe name)
    lut_sha256: str
    source_lut: Optional[str]           # the official LUT, when there is one
    source_lut_sha256: Optional[str]
    profile: str
    camera: str
    settings: ColorSettings
    in_matrix: str
    in_range: str
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["applied"] = True
        return d

    def rgb_filter(self) -> str:
        """Source YUV -> 16-bit RGB with the source's own matrix and range."""
        return (f"scale=in_color_matrix={self.in_matrix}:in_range={self.in_range}"
                f":flags=accurate_rnd+full_chroma_int,format=gbrp16le")

    def lut_filter(self) -> str:
        return f"lut3d=file={self.lut_path}:interp=tetrahedral"

    def video_filter(self, bit_depth: int) -> str:
        """Full chain to Rec.709 YUV at ``bit_depth``."""
        fmt = "yuv420p10le" if int(bit_depth) == 10 else "yuv420p"
        # setparams: the encoder's -color_* options alone left primaries and
        # transfer untagged on ffmpeg 7.1; tagging the frames does not.
        return (f"{self.rgb_filter()},{self.lut_filter()},"
                f"scale=out_color_matrix=bt709:out_range=tv"
                f":flags=accurate_rnd+full_chroma_int,format={fmt},"
                f"setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv")

    @staticmethod
    def output_tags() -> dict:
        return {"color_primaries": "bt709", "color_transfer": "bt709",
                "color_space": "bt709", "color_range": "tv"}


def _cache_copy(src: str, cache_dir: str) -> tuple:
    """Copy ``src`` into the cache under its hash: a path with no characters
    that need escaping in an ffmpeg filter graph."""
    digest = sha256_file(src)
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, f"{digest}.cube")
    if not os.path.exists(dst):
        tmp = dst + f".{os.getpid()}.tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    return dst, digest


def _source_matrix(media: MediaInfo, warnings: list) -> tuple:
    matrix = _MATRIX.get(media.color_space)
    if matrix is None:
        matrix = "bt709"
        warnings.append(f"Matriz de color del original desconocida ({media.color_space}); "
                        "se asume bt709.")
    rng = "pc" if media.color_range in ("pc", "jpeg") else "tv"
    return matrix, rng


def plan(media: MediaInfo, decision: ColorDecision, camera: Optional[Camera],
         settings: ColorSettings, lut_override: Optional[str] = None,
         cache_dir: str = LUT_CACHE_DIR) -> Optional[ColorPlan]:
    """What to do to the colour, or ``None`` to keep the original colour.

    ``None`` when the profile is not confirmed (adjustments included: they
    assume Rec.709 input), or when it is a non-log profile and the
    adjustments are neutral. An explicit LUT for an unconfirmed profile is
    an error rather than a silent no-op."""
    if not decision.confirmed or camera is None:
        if lut_override:
            raise ColorError(
                "El perfil de color no está confirmado: no se aplica ninguna LUT. "
                "Confírmalo con --camera y --profile.")
        return None
    prof = camera.profile(decision.profile)
    warnings = list(decision.warnings)
    source = None
    if prof.transform == NO_TRANSFORM:
        if lut_override:
            raise ColorError(
                f"El perfil '{prof.name}' de {camera.name} no es logarítmico: no se le "
                "aplica una LUT de conversión.")
        if settings.neutral:
            return None
    else:
        source = lut_override or prof.settings.get("lut")
        if not source:
            raise ColorError(
                f"El perfil '{prof.name}' de {camera.name} necesita la LUT oficial de DJI "
                f"(.cube), y no está configurada. Indícala con --lut o en el campo 'lut' "
                f"del YAML de la cámara. Para exportar sin corregir el color: --no-color.")
        source = os.path.expanduser(source)
        if not lut_override and not os.path.isabs(source):
            # A relative path in a camera YAML is relative to that YAML.
            source = os.path.join(os.path.dirname(camera.source), source)
        source = os.path.abspath(source)
        if not os.path.isfile(source):
            raise ColorError(f"No existe la LUT: {source}")
    matrix, rng = _source_matrix(media, warnings)

    base = read_cube(source) if source else None      # validates the official file too
    if settings.neutral and source:
        path, digest = _cache_copy(source, cache_dir)
    else:
        lut = compose(base, settings)
        os.makedirs(cache_dir, exist_ok=True)
        tmp = os.path.join(cache_dir, f".composed-{os.getpid()}.cube")
        if os.path.exists(tmp):
            os.remove(tmp)
        write_cube(lut, tmp)
        digest = sha256_file(tmp)
        path = os.path.join(cache_dir, f"{digest}.cube")
        os.replace(tmp, path)
    return ColorPlan(lut_path=path, lut_sha256=digest, source_lut=source,
                     source_lut_sha256=sha256_file(source) if source else None,
                     profile=prof.name, camera=camera.camera, settings=settings,
                     in_matrix=matrix, in_range=rng, warnings=warnings)


# ---------------------------------------------------------------- comparison

def compare_image(media: MediaInfo, cplan: ColorPlan, at: float, path: str,
                  width: int = 1920) -> None:
    """One PNG: the original frame at ``at`` (left) and the corrected one
    (right), each ``width / 2`` wide. Writes only ``path``."""
    half = max(2, width // 2 // 2 * 2)
    graph = (f"[0:v]{cplan.rgb_filter()},split[o][c];"
             f"[c]{cplan.lut_filter()}[g];"
             f"[o]scale={half}:-2[o2];[g]scale={half}:-2[g2];"
             f"[o2][g2]hstack=inputs=2,format=rgb24[out]")
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
                           "-ss", f"{max(0.0, at):.3f}", "-i", media.path,
                           "-filter_complex", graph, "-map", "[out]", "-frames:v", "1",
                           "-f", "image2", "-c:v", "png", path],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(path):
        raise ColorError(f"ffmpeg no pudo generar la comparación: {proc.stderr.strip()[-400:]}")
