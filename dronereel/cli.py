"""Command line: ``python -m dronereel <command>``.

    cameras                 list the camera files and their colour profiles
    probe VIDEO [...]       metadata, evidence, and the colour decision
    analyze VIDEO [...]     per-sample measurements, cached (F2)
    select VIDEO [...]      the one continuous segment to export, and why (F3)

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
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/dronereel-output")


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


def cmd_select(args) -> int:
    import yaml

    from dronereel import analyze as an
    from dronereel import selector as sel
    from dronereel.safety import output_path

    try:
        raw = {}
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            raw = raw.get("select", raw)
        for key, value in (("min_duration", args.min), ("max_duration", args.max),
                           ("pre_roll", args.pre), ("post_roll", args.post)):
            if value is not None:
                raw[key] = value
        cfg = sel.SelectConfig.from_dict(raw)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        print(f"Error en la configuración: {e}", file=sys.stderr)
        return 2

    media = probe_media(args.video)
    result = an.analyze(media, use_cache=not args.no_cache, cache_dir=an.CACHE_DIR,
                        prefer_hardware=not args.software, progress=an.print_progress)
    print(file=sys.stderr)
    try:
        choice = sel.select(result, cfg)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    best = choice.best
    limit = f"{cfg.max_duration:g} s" if cfg.max_duration else "sin límite"
    print(f"Archivo:    {media.path} ({result.duration:.1f} s)")
    print(f"Duración:   entre {cfg.min_duration:g} s y {limit}; "
          f"contexto {cfg.pre_roll:g} s antes y {cfg.post_roll:g} s después del mejor momento")
    print(f"Segmento:   {best.start:.2f} s → {best.end:.2f} s ({best.duration:.1f} s), "
          f"puntuación {best.score:.3f}, "
          f"mejor momento {best.highlight[0]:.1f}–{best.highlight[1]:.1f} s")
    print("            " + ", ".join(f"{k} {best.parts[k]:.2f}" for k in sel.COMPONENTS))
    for w in choice.warnings:
        print(f"  AVISO: {w}")
    print("Alternativas (sin solaparse; no se exportan):")
    for i, c in enumerate(choice.candidates[1:], start=2):
        print(f"  {i}. {c.start:7.2f} → {c.end:7.2f} s  ({c.duration:4.1f} s)  {c.score:.3f}")
    if len(choice.candidates) < 2:
        print("  ninguna")

    stem = os.path.splitext(os.path.basename(media.path))[0]
    log = output_path(args.output, f"{stem}_select.json", [media.path])
    record = sel.log_record(choice, result, media.path,
                            an.content_key(media.path, result.fps, result.width))
    sel.write_log(record, log)
    print(f"Registro:   {log}")
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

    se = sub.add_parser("select", help="elige UN segmento continuo (solo lee el vídeo)")
    se.add_argument("video")
    se.add_argument("--min", type=float, help="duración mínima en s (10)")
    se.add_argument("--max", type=float, help="duración máxima en s (30; 0 = sin límite)")
    se.add_argument("--pre", type=float, help="contexto antes del mejor momento, s (2)")
    se.add_argument("--post", type=float, help="contexto después del mejor momento, s (2)")
    se.add_argument("--config", help="YAML con opciones y pesos (sección 'select:')")
    se.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                    help=f"carpeta para el registro (por defecto {DEFAULT_OUTPUT_DIR})")
    se.add_argument("--software", action="store_true", help="no usar el decodificador por hardware")
    se.add_argument("--no-cache", action="store_true", help="medir de nuevo aunque haya caché")
    se.set_defaults(func=cmd_select)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
