"""Command line: ``python -m dronereel <command>``.

    cameras                 list the camera files and their colour profiles
    probe VIDEO [...]       metadata, evidence, and the colour decision

``probe`` only reads. ``--camera`` and ``--profile`` are how a profile is
confirmed by hand until a camera's identification rules are verified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from dronereel import profiles
from dronereel.media import probe_media

USER_PROFILES_DIR = os.path.join(
    os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
    "dronereel", "profiles")


def _cameras(args) -> dict:
    return profiles.load_cameras(*(args.profiles_dir or [USER_PROFILES_DIR]))


def cmd_cameras(args) -> int:
    for cam in _cameras(args).values():
        print(f"{cam.camera}: {cam.name}  ({cam.source})")
        verified = "verificada" if cam.identify.get("verified") else "sin verificar"
        print(f"  identificación automática: {verified}")
        for prof in cam.color_profiles.values():
            extra = ""
            if prof.is_log:
                lut = prof.settings.get("lut")
                extra = f"  LUT: {lut or 'sin definir'}"
            print(f"  perfil {prof.name}: transformación {prof.transform}{extra}")
    return 0


def cmd_probe(args) -> int:
    media = probe_media(args.video)
    try:
        decision = profiles.decide(media, _cameras(args),
                                   camera=args.camera, profile=args.profile)
    except profiles.ProfileError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.json:
        out = {"media": media.to_dict(),
               "decision": {**decision.__dict__,
                            "confirmed": decision.confirmed,
                            "apply_color": decision.apply_color}}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print(f"Archivo:    {media.path}")
    print(f"Vídeo:      {media.video_codec} {media.width}x{media.height} "
          f"@ {media.fps:.3f} fps ({media.fps_fraction}), {media.duration:.2f} s, "
          f"rotación {media.rotation}°")
    print(f"Audio:      {media.audio_codec if media.has_audio else 'no'}")
    print("Evidencia:")
    for line in decision.evidence:
        print(f"  - {line}")
    print(f"Cámara:     {decision.camera or 'desconocida'} ({decision.camera_source})")
    print(f"Perfil:     {decision.profile} ({decision.profile_source})")
    for h in decision.hints:
        print(f"  pista: {h}")
    for w in decision.warnings:
        print(f"  AVISO: {w}")
    if decision.apply_color:
        print(f"Color:      se aplicará '{decision.transform}' (perfil confirmado)")
    elif decision.confirmed:
        print("Color:      perfil confirmado sin transformación; se conserva el color")
    else:
        print("Color:      perfil NO confirmado; se conserva el color original.\n"
              "            Para confirmarlo: --camera <cámara> --profile <perfil>")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dronereel", description="DroneReel-AI")
    p.add_argument("--profiles-dir", action="append",
                   help=f"carpeta con YAML de cámaras (por defecto {USER_PROFILES_DIR})")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("cameras", help="cámaras y perfiles definidos").set_defaults(func=cmd_cameras)

    pr = sub.add_parser("probe", help="metadatos y decisión de color (solo lectura)")
    pr.add_argument("video")
    pr.add_argument("--camera", help="cámara de origen (p. ej. avata2)")
    pr.add_argument("--profile", help="perfil de color confirmado (p. ej. dlog_m)")
    pr.add_argument("--json", action="store_true", help="salida JSON")
    pr.set_defaults(func=cmd_probe)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
