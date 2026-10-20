"""Command line: ``python -m dronereel <command>``.

    cameras                 list the camera files and their colour profiles
    probe VIDEO [...]       metadata, evidence, and the colour decision
    analyze VIDEO [...]     per-sample measurements, cached (F2)
    select VIDEO [...]      the one continuous segment to export, and why (F3)
    export VIDEO [...]      write that one segment to its own file (F4, colour F5)
    color VIDEO [...]       original vs corrected stills, to tune the colour (F5)

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


class _ConfigError(Exception):
    pass


def _read_config(path) -> dict:
    import yaml
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        raise _ConfigError(e) from None


def _select_config(args, raw: dict):
    from dronereel import selector as sel
    # A file with sections uses 'select:'; a flat file is all selection options.
    raw = dict((raw.get("select") or {}) if {"select", "export"} & set(raw) else raw)
    for key, value in (("min_duration", args.min), ("max_duration", args.max),
                       ("pre_roll", args.pre), ("post_roll", args.post)):
        if value is not None:
            raw[key] = value
    try:
        return sel.SelectConfig.from_dict(raw)
    except (ValueError, TypeError) as e:
        raise _ConfigError(e) from None


def _run_selection(args, media, cfg):
    """``(analysis, selection)``; raises ValueError when nothing can be chosen."""
    from dronereel import analyze as an
    from dronereel import selector as sel
    result = an.analyze(media, use_cache=not args.no_cache, cache_dir=an.CACHE_DIR,
                        prefer_hardware=not args.software, progress=an.print_progress)
    print(file=sys.stderr)
    return result, sel.select(result, cfg)


def _print_selection(media, result, choice, cfg) -> None:
    from dronereel import selector as sel
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


def cmd_select(args) -> int:
    from dronereel import analyze as an
    from dronereel import selector as sel
    from dronereel.safety import output_path

    try:
        cfg = _select_config(args, _read_config(args.config))
    except _ConfigError as e:
        print(f"Error en la configuración: {e}", file=sys.stderr)
        return 2
    media = probe_media(args.video)
    try:
        result, choice = _run_selection(args, media, cfg)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    _print_selection(media, result, choice, cfg)

    stem = os.path.splitext(os.path.basename(media.path))[0]
    log = output_path(args.output, f"{stem}_select.json", [media.path])
    record = sel.log_record(choice, result, media.path,
                            an.content_key(media.path, result.fps, result.width))
    sel.write_log(record, log)
    print(f"Registro:   {log}")
    return 0


_COLOR_FLAGS = (("exposure", "exposure_ev"), ("contrast", "contrast"),
                ("saturation", "saturation"), ("temperature", "wb_temperature"),
                ("tint", "wb_tint"), ("highlights", "highlight_protection"))


def _color_plan(args, media, decision, cameras, raw):
    """``ColorPlan`` or ``None`` (keep the original colour). Raises
    ``color.ColorError`` for a request that cannot be honoured safely."""
    from dronereel import color

    flags = {key: getattr(args, name) for name, key in _COLOR_FLAGS
             if getattr(args, name, None) is not None}
    if getattr(args, "no_color", False):
        return None
    camera = cameras.get(decision.camera) if decision.camera else None
    values = {}
    if decision.confirmed and camera is not None:
        values.update(camera.color)
    values.update(raw.get("color") or {})
    values.update(flags)
    settings = color.ColorSettings.from_dict(values)
    cplan = color.plan(media, decision, camera, settings, lut_override=args.lut)
    if cplan is None and not decision.confirmed and (flags or raw.get("color")):
        print("AVISO: perfil de color no confirmado; los ajustes de color se ignoran y se "
              "conserva el color original. Confírmalo con --camera y --profile.")
    return cplan


def _describe_color(cplan) -> str:
    if cplan is None:
        return "se conserva el color original"
    what = (f"LUT oficial {os.path.basename(cplan.source_lut)}" if cplan.source_lut
            else "sin LUT de conversión")
    s = cplan.settings
    adj = "ajustes neutros" if s.neutral else (
        f"exposición {s.exposure_ev:+g} EV, contraste {s.contrast:g}, saturación "
        f"{s.saturation:g}, temperatura {s.wb_temperature:+g}, tinte {s.wb_tint:+g}, "
        f"protección de altas luces {s.highlight_protection:g}")
    return f"{cplan.camera}/{cplan.profile} → Rec.709: {what}, {adj}"


def cmd_color(args) -> int:
    from dronereel import color
    from dronereel.safety import assert_unchanged, fingerprint, output_path

    media = probe_media(args.video)
    try:
        raw = _read_config(args.config)
        cameras = _cameras(args)
        decision = profiles.decide(media, cameras, camera=args.camera, profile=args.profile)
        cplan = _color_plan(args, media, decision, cameras, raw)
    except (_ConfigError, profiles.ProfileError, color.ColorError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if cplan is None:
        print("Nada que comparar: " + ("perfil no confirmado; " if not decision.confirmed
                                       else "perfil sin transformación y ajustes neutros; ")
              + "se conserva el color original.")
        return 0
    original = fingerprint(media.path)
    times = args.at or [media.duration * f for f in (0.25, 0.5, 0.75)]
    stem = os.path.splitext(os.path.basename(media.path))[0]
    print(f"Color:      {_describe_color(cplan)}")
    for w in cplan.warnings:
        print(f"  AVISO: {w}")
    for t in times:
        path = output_path(args.output, f"{stem}_color_{t:07.2f}.png", [media.path])
        color.compare_image(media, cplan, t, path)
        print(f"Comparación {t:.2f} s: {path}  (izquierda original, derecha corregido)")
    assert_unchanged(original)
    return 0


def cmd_export(args) -> int:
    from dronereel import analyze as an
    from dronereel import export as ex
    from dronereel import selector as sel

    media = probe_media(args.video)
    try:
        raw = _read_config(args.config)
        cameras = _cameras(args)
        decision = profiles.decide(media, cameras, camera=args.camera, profile=args.profile)
        export_raw = {}
        if decision.camera and decision.camera in cameras:
            export_raw.update(cameras[decision.camera].export)
        export_raw.update(raw.get("export") or {})
        for key, value in (("mode", args.mode), ("codec", args.codec), ("crf", args.crf),
                           ("preset", args.preset), ("bit_depth", args.bit_depth)):
            if value is not None:
                export_raw[key] = value
        ecfg = ex.ExportConfig.from_dict(export_raw)
        scfg = _select_config(args, raw)
    except (_ConfigError, ValueError, TypeError) as e:
        print(f"Error en la configuración: {e}", file=sys.stderr)
        return 2

    from dronereel import color
    try:
        cplan = _color_plan(args, media, decision, cameras, raw)
    except (color.ColorError, profiles.ProfileError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if cplan is not None and ecfg.mode == "copy":
        print("Error: corregir el color exige recodificar; quita --mode copy o usa --no-color.",
              file=sys.stderr)
        return 2
    extra = {"color": cplan.to_dict() if cplan else
             {"camera": decision.camera, "profile": decision.profile,
              "profile_source": decision.profile_source, "applied": False}}
    if (args.start is None) != (args.end is None):
        print("Error: --start y --end van juntos", file=sys.stderr)
        return 2
    if args.start is not None:
        start, end = args.start, args.end
        print(f"Segmento:   {start:.2f} s → {end:.2f} s (indicado a mano)")
        extra["selection"] = {"manual": True, "start": start, "end": end}
    else:
        try:
            result, choice = _run_selection(args, media, scfg)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        _print_selection(media, result, choice, scfg)
        start, end = choice.best.start, choice.best.end
        extra["selection"] = sel.log_record(
            choice, result, media.path, an.content_key(media.path, result.fps, result.width))

    print(f"Color:      {_describe_color(cplan)}")
    if cplan:
        for w in cplan.warnings:
            print(f"  AVISO: {w}")
    if ecfg.mode == "reencode" or cplan is not None:
        for note in ex.reencode_notes(media, start, end):
            print(note)
    try:
        res = ex.export_segment(
            media, start, end, args.output, ecfg, progress=ex.print_progress,
            video_filter=cplan.video_filter(ecfg.bit_depth) if cplan else None,
            color_tags=cplan.output_tags() if cplan else None)
    except (ex.ExportError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 2
    print(file=sys.stderr)
    record = ex.write_record(res, media, extra)
    how = ("copia directa, sin recomprimir" if res.mode == "copy"
           else f"recodificado {ecfg.codec} {ecfg.bit_depth} bits, crf {ecfg.crf:g}")
    print(f"Exportado:  {res.path}")
    print(f"            {res.start:.2f} s → {res.end:.2f} s ({res.duration:.2f} s, "
          f"{res.frames} fotogramas a {res.fps} fps), {how}, en {res.seconds:.1f} s")
    for w in res.warnings:
        print(f"  AVISO: {w}")
    print(f"Registro:   {record}")
    if cplan:
        try:
            cmp_path = os.path.splitext(res.path)[0] + "_comparacion.png"
            color.compare_image(media, cplan, (res.start + res.end) / 2, cmp_path)
            print(f"Comparación: {cmp_path}  (izquierda original, derecha corregido)")
        except color.ColorError as e:
            print(f"  AVISO: no se pudo generar la comparación: {e}")
    print("Original:   sin cambios (huella SHA-256 verificada)")
    return 0


def _add_color_args(p) -> None:
    p.add_argument("--lut", help="LUT oficial .cube (sustituye a la del YAML de la cámara)")
    p.add_argument("--exposure", type=float, help="exposición en EV (-3 a 3)")
    p.add_argument("--contrast", type=float, help="contraste (0.5 a 2; 1 = igual)")
    p.add_argument("--saturation", type=float, help="saturación (0 a 2; 1 = igual)")
    p.add_argument("--temperature", type=float, help="balance de blancos: -100 frío a 100 cálido")
    p.add_argument("--tint", type=float, help="tinte: -100 verde a 100 magenta")
    p.add_argument("--highlights", type=float, help="protección de altas luces (0 a 1)")


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

    ex = sub.add_parser("export", help="exporta UN segmento continuo a un archivo nuevo")
    ex.add_argument("video")
    ex.add_argument("--start", type=float, help="inicio en s (sin él, lo elige 'select')")
    ex.add_argument("--end", type=float, help="fin en s (con --start)")
    ex.add_argument("--min", type=float, help="duración mínima en s (10)")
    ex.add_argument("--max", type=float, help="duración máxima en s (30; 0 = sin límite)")
    ex.add_argument("--pre", type=float, help="contexto antes del mejor momento, s (2)")
    ex.add_argument("--post", type=float, help="contexto después del mejor momento, s (2)")
    ex.add_argument("--mode", choices=["auto", "copy", "reencode"],
                    help="auto (copia si no cambia la imagen), copy o reencode")
    ex.add_argument("--codec", choices=["hevc", "h264"], help="códec al recodificar (hevc)")
    ex.add_argument("--bit-depth", type=int, choices=[8, 10], help="bits al recodificar (10)")
    ex.add_argument("--crf", type=float, help="calidad al recodificar (18; menor = mejor)")
    ex.add_argument("--preset", help="preset del codificador (medium)")
    ex.add_argument("--camera", help="cámara de origen (p. ej. avata2)")
    ex.add_argument("--profile", help="perfil de color confirmado (p. ej. dlog_m)")
    ex.add_argument("--config", help="YAML con secciones 'select:' y 'export:'")
    ex.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                    help=f"carpeta de salida (por defecto {DEFAULT_OUTPUT_DIR})")
    ex.add_argument("--software", action="store_true", help="no usar el decodificador por hardware")
    ex.add_argument("--no-cache", action="store_true", help="medir de nuevo aunque haya caché")
    _add_color_args(ex)
    ex.add_argument("--no-color", action="store_true", help="no corregir el color")
    ex.set_defaults(func=cmd_export)

    co = sub.add_parser("color", help="compara original y corregido en fotogramas sueltos")
    co.add_argument("video")
    co.add_argument("--camera", help="cámara de origen (p. ej. avata2)")
    co.add_argument("--profile", help="perfil de color confirmado (p. ej. dlog_m)")
    co.add_argument("--at", type=float, action="append", help="instante en s (repetible)")
    co.add_argument("--config", help="YAML con sección 'color:'")
    co.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                    help=f"carpeta de salida (por defecto {DEFAULT_OUTPUT_DIR})")
    _add_color_args(co)
    co.set_defaults(func=cmd_color)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
