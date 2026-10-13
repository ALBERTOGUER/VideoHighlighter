"""Which camera shot a video, in which colour profile, and whether that is known.

Why this is strict
==================
A log-to-Rec.709 LUT on footage that is not log crushes and oversaturates it;
the Avata 2's LUT on the Osmo Nano's log is wrong in subtler ways; D-Log and
D-Log M are different curves. And DJI's files do not reliably say which they
are. So the rule is: **colour is transformed only when the profile is
confirmed**, and there are exactly two ways to confirm it:

1. **The user says so** (``camera=`` and ``profile=``), and the profile exists
   for *that* camera's YAML — a profile is never looked up across cameras.
2. **A rule marked ``verified: true``** in the camera's YAML matches. Rules
   start unverified and become verified only after being checked against real
   clips from that camera. An unverified rule that matches is reported as a
   hint, never acted on.

Anything else is ``unknown``, and unknown keeps the original colour.

Camera files live in ``dronereel/profiles/*.yaml``; a user folder with files
of the same name takes precedence (see :func:`load_cameras`).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

from dronereel.media import MediaInfo

BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
UNKNOWN = "unknown"
NO_TRANSFORM = "none"

# Tags worth showing as evidence: where cameras usually name themselves.
_EVIDENCE_TAG = re.compile(r"make|model|encoder|handler|vendor|^com\.|creation_time", re.IGNORECASE)


class ProfileError(ValueError):
    """A request that would risk the wrong colour transform."""


@dataclass
class ColorProfile:
    name: str
    transform: str                      # "none" | "lut"
    settings: dict
    identify: dict

    @property
    def is_log(self) -> bool:
        return self.transform != NO_TRANSFORM


@dataclass
class Camera:
    camera: str
    name: str
    identify: dict
    color_profiles: dict                # name -> ColorProfile
    color: dict
    export: dict
    source: str

    def profile(self, name: str) -> ColorProfile:
        if name not in self.color_profiles:
            known = ", ".join(sorted(self.color_profiles)) or "(ninguno)"
            raise ProfileError(
                f"El perfil '{name}' no está definido para {self.name}. "
                f"Perfiles de esta cámara: {known}.")
        return self.color_profiles[name]


@dataclass
class ColorDecision:
    camera: Optional[str]
    camera_source: str                  # "user" | "verified_rule" | "none"
    profile: str                        # a profile name or "unknown"
    profile_source: str                 # "user" | "verified_rule" | "none"
    transform: str                      # what to apply: "none" unless confirmed
    evidence: list = field(default_factory=list)
    hints: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.profile != UNKNOWN and self.profile_source != "none"

    @property
    def apply_color(self) -> bool:
        return self.confirmed and self.transform != NO_TRANSFORM


def _load_camera(path: str) -> Camera:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cam_id = str(data.get("camera") or "").strip()
    if not cam_id:
        raise ProfileError(f"{path}: falta 'camera'")
    profiles = {}
    for name, spec in (data.get("color_profiles") or {}).items():
        spec = dict(spec or {})
        transform = str(spec.pop("transform", NO_TRANSFORM))
        if transform not in (NO_TRANSFORM, "lut"):
            raise ProfileError(f"{path}: transformación desconocida '{transform}' en '{name}'")
        identify = dict(spec.pop("identify", None) or {})
        profiles[str(name)] = ColorProfile(str(name), transform, spec, identify)
    return Camera(
        camera=cam_id,
        name=str(data.get("name") or cam_id),
        identify=dict(data.get("identify") or {}),
        color_profiles=profiles,
        color=dict(data.get("color") or {}),
        export=dict(data.get("export") or {}),
        source=path,
    )


def load_cameras(*dirs: str) -> dict:
    """``{camera_id: Camera}`` from the built-in folder, then each of ``dirs``;
    a later folder's camera replaces an earlier one with the same id."""
    cameras = {}
    for folder in (BUILTIN_DIR, *dirs):
        if not folder or not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".yaml", ".yml")):
                cam = _load_camera(os.path.join(folder, name))
                cameras[cam.camera] = cam
    return cameras


def rule_matches(rule: dict, media: MediaInfo) -> bool:
    """True when every condition the rule states holds. A rule that states
    nothing matches nothing — an empty rule must not confirm a profile."""
    stated = False
    if rule.get("srt_color_md"):
        stated = True
        wanted = {str(v).lower() for v in rule["srt_color_md"]}
        if not wanted.intersection(media.srt_color_modes):
            return False
    if rule.get("tags"):
        stated = True
        for tag, pattern in rule["tags"].items():
            value = media.tags.get(str(tag).lower())
            if value is None or not re.search(str(pattern), str(value)):
                return False
    if rule.get("min_bit_depth"):
        stated = True
        if media.bit_depth < int(rule["min_bit_depth"]):
            return False
    return stated


def _evidence(media: MediaInfo) -> list:
    out = [
        f"bits: {media.bit_depth} ({media.pix_fmt or '?'})",
        f"color_transfer={media.color_transfer} color_primaries={media.color_primaries} "
        f"color_space={media.color_space}",
    ]
    if media.srt_path:
        modes = ", ".join(media.srt_color_modes) or "(sin color_md)"
        out.append(f"SRT {os.path.basename(media.srt_path)}: color_md {modes}")
    else:
        out.append("sin archivo .SRT")
    for k, v in sorted(media.tags.items()):
        if _EVIDENCE_TAG.search(k):
            out.append(f"tag {k}={v}")
    return out


def decide(media: MediaInfo, cameras: dict, camera: Optional[str] = None,
           profile: Optional[str] = None) -> ColorDecision:
    """Settle camera and colour profile for ``media``. Raises ProfileError for
    a request that names a profile without a camera, or a camera/profile the
    YAML does not define; everything uncertain comes back as ``unknown``."""
    if profile and not camera:
        raise ProfileError(
            "Para fijar un perfil de color indica también la cámara: un perfil "
            "nunca se busca en otra cámara.")

    decision = ColorDecision(None, "none", UNKNOWN, "none", NO_TRANSFORM,
                             evidence=_evidence(media))

    # Camera.
    if camera:
        if camera not in cameras:
            known = ", ".join(sorted(cameras)) or "(ninguna)"
            raise ProfileError(f"Cámara '{camera}' desconocida. Definidas: {known}.")
        decision.camera, decision.camera_source = camera, "user"
    else:
        verified = []
        for cam in cameras.values():
            if rule_matches(cam.identify, media):
                if cam.identify.get("verified"):
                    verified.append(cam.camera)
                else:
                    decision.hints.append(
                        f"parece {cam.name} (regla sin verificar; no se usa)")
        if len(verified) == 1:
            decision.camera, decision.camera_source = verified[0], "verified_rule"
        elif len(verified) > 1:
            decision.warnings.append(
                "varias cámaras coinciden con reglas verificadas: " + ", ".join(verified))

    if decision.camera is None:
        return decision
    cam = cameras[decision.camera]

    # Profile — only ever among this camera's own profiles.
    if profile:
        chosen = cam.profile(profile)
        decision.profile, decision.profile_source = chosen.name, "user"
    else:
        verified = []
        for prof in cam.color_profiles.values():
            if prof.identify and rule_matches(prof.identify, media):
                if prof.identify.get("verified"):
                    verified.append(prof.name)
                else:
                    decision.hints.append(
                        f"parece perfil '{prof.name}' (regla sin verificar; no se usa)")
        if len(verified) == 1:
            decision.profile, decision.profile_source = verified[0], "verified_rule"
        elif len(verified) > 1:
            decision.warnings.append(
                "varios perfiles coinciden con reglas verificadas: " + ", ".join(verified))

    if decision.profile != UNKNOWN:
        chosen = cam.profile(decision.profile)
        decision.transform = chosen.transform
        if chosen.is_log and media.bit_depth < 10:
            decision.warnings.append(
                f"'{chosen.name}' es un perfil logarítmico pero el archivo es de "
                f"{media.bit_depth} bits: comprueba que el perfil es correcto")
    return decision
