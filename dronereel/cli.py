"""Command line: ``python -m dronereel <command>``.

    cameras                 list the camera files and their colour profiles
    probe VIDEO [...]       metadata, evidence, and the colour decision
    analyze VIDEO [...]     per-sample measurements, cached (F2)

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


def cmd_analyze(args) -> int:
    import numpy as np

    from dronereel import analyze as an     # cv2 only when analysing
    from dronereel.safety import output_path

    media = probe_media(args.video)
    result = an.analyze(media, fps=args.fps, width=args.width,
                        use_cache=not args.no_cache,
                        prefer_hardware=not args.software,
                        progress=an.print_progress)
    print(file=sys.stderr)
    s = result.samples
    speed = result.seconds / max(result.duration, 1e-6)
    src = "caché" if result.cached else f"decodificación {result.decoder}"
    print(f"Archivo:    {media.path}")
    print(f"Muestras:   {result.count} a {result.fps:g}/s, {result.width}x{result.height} ({src})")
    when = " (cuando se midió)" if result.cached else ""
    print(f"Tiempo:     {result.seconds:.1f} s para {result.duration:.1f} s de vídeo "
          f"= {speed:.2f} s de análisis por segundo de vídeo{when}")

    def pct(mask) -> str:
        return f"{100.0 * np.count_nonzero(mask) / max(1, result.count):.0f} %"

    known = ~np.isnan(s["roll"])
    print(f"Movimiento: medible en {pct(known)} de las muestras; "
          f"coherencia mediana {np.nanmedian(s['coherence']):.2f}")
    if known.any():
        print(f"            alabeo máx {np.nanmax(np.abs(s['roll'])):.0f}°/s, "
              f"giro máx {np.nanmax(np.abs(s['pan_x'])):.2f} anchos/s, "
              f"avance máx {np.nanmax(s['zoom']):.2f}/s")
    print(f"Imagen:     altas luces quemadas (>5 %) en {pct(s['clip_high'] > 0.05)}; "
          f"sombras negras (>5 %) en {pct(s['clip_low'] > 0.05)}")
    print(f"Cortes:     {', '.join(f'{c:.1f} s' for c in result.cuts) or 'ninguno'}")
    if args.csv:
        csv = output_path(os.path.dirname(os.path.abspath(args.csv)),
                          os.path.basename(args.csv), [media.path])
        an.write_csv(result, csv)
        print(f"CSV:        {csv}")
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

    an = sub.add_parser("analyze", help="mide el vídeo completo (solo lectura; caché en ~/.cache/dronereel)")
    an.add_argument("video")
    an.add_argument("--fps", type=float, default=5.0, help="muestras por segundo (5)")
    an.add_argument("--width", type=int, default=384, help="ancho de análisis en px (384)")
    an.add_argument("--software", action="store_true", help="no usar el decodificador por hardware")
    an.add_argument("--no-cache", action="store_true", help="medir de nuevo aunque haya caché")
    an.add_argument("--csv", help="guardar las medidas por muestra en este CSV")
    an.set_defaults(func=cmd_analyze)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
