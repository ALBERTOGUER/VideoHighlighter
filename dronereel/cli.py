"""Command line: ``python -m dronereel <command>``.

    cameras                 list the camera files and their colour profiles
    probe VIDEO [...]       metadata, evidence, and the colour decision
    analyze VIDEO [...]     per-sample measurements, cached (F2)
    select VIDEO [...]      the one continuous segment to export, and why (F3)
    export VIDEO [...]      write that one segment to its own file (F4, colour F5)
    color VIDEO [...]       original vs corrected stills, to tune the colour (F5)
    rate RECORD|VIDEO [...] add a judgement to your ratings dataset (F6)
    review [...]            rate the exported clips not rated yet, one by one (F6)
    dataset [...]           what the ratings dataset holds; merge another machine's
    train [...]             learn your taste from the ratings, before vs after (F7)

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


_FEATURE_NAMES = {"exposure": "exposición", "clarity": "nitidez", "stability": "estabilidad",
                  "dynamism": "dinamismo", "manoeuvre": "giros/maniobras", "detail": "detalle",
                  "proximity": "proximidad", "quality": "calidad media",
                  "worst_second": "peor segundo", "variety": "variedad",
                  "empty": "tramos vacíos", "missing_context": "falta de contexto",
                  "duration": "duración"}


def _ranker(args):
    """``(model or None, why)`` for selection: an explicit ``--model``, else
    the default model only when its evaluation showed it beats the selector."""
    from dronereel import ranker as rk
    if getattr(args, "no_model", False):
        return None, "desactivado (--no-model)"
    path = getattr(args, "model", None)
    try:
        model = rk.load(path or rk.DEFAULT_MODEL)
    except (ValueError, OSError, KeyError) as e:
        if path:
            raise _ConfigError(f"modelo {path}: {e}") from None
        return None, f"el modelo guardado no se puede usar ({e})"
    if model is None:
        if path:
            raise _ConfigError(f"no existe el modelo {path}")
        return None, "sin modelo todavía (valora clips y ejecuta 'train')"
    ev = model.trained.get("evaluation") or {}
    acc = (f"; acierta {ev['after'] * 100:.0f} % frente a {ev['before'] * 100:.0f} % "
           f"en vídeos no vistos" if "after" in ev else "")
    desc = f"{model.trained.get('entries', '?')} valoraciones de {model.trained.get('videos', '?')} vídeos{acc}"
    if path:
        return model, f"{path} ({desc})"
    if not model.better:
        return None, f"el modelo guardado no mejora al selector todavía ({desc})"
    return model, f"el tuyo ({desc})"


def _run_selection(args, media, cfg, ranker=None):
    """``(analysis, selection)``; raises ValueError when nothing can be chosen."""
    from dronereel import analyze as an
    from dronereel import selector as sel
    result = an.analyze(media, use_cache=not args.no_cache, cache_dir=an.CACHE_DIR,
                        prefer_hardware=not args.software, progress=an.print_progress)
    print(file=sys.stderr)
    return result, sel.select(result, cfg, ranker)


def _print_model(result, choice, cfg, model, why) -> None:
    """Before/after for this video: what the hand-made selector alone picks."""
    from dronereel import selector as sel
    print(f"Modelo:     {why}")
    if model is None:
        return
    plain = sel.select(result, cfg).best
    b = choice.best
    if abs(plain.start - b.start) < 0.05 and abs(plain.end - b.end) < 0.05:
        print("            sin tu modelo habría elegido el mismo tramo")
    else:
        print(f"            sin tu modelo habría elegido {plain.start:.2f} → {plain.end:.2f} s")


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
        model, why = _ranker(args)
        result, choice = _run_selection(args, media, cfg, model)
    except (ValueError, _ConfigError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    _print_selection(media, result, choice, cfg)
    _print_model(result, choice, cfg, model, why)

    stem = os.path.splitext(os.path.basename(media.path))[0]
    log = output_path(args.output, f"{stem}_select.json", [media.path])
    record = sel.log_record(choice, result, media.path,
                            an.content_key(media.path, result.fps, result.width))
    record["model"] = model.trained if model is not None else None
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
        model, why = _ranker(args)
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
            result, choice = _run_selection(args, media, scfg, model)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        _print_selection(media, result, choice, scfg)
        _print_model(result, choice, scfg, model, why)
        start, end = choice.best.start, choice.best.end
        extra["selection"] = sel.log_record(
            choice, result, media.path, an.content_key(media.path, result.fps, result.width))
        extra["selection"]["model"] = model.trained if model is not None else None

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


def _clip_features(record: dict, args):
    """``(features, selector score)`` of a clip exported with a hand-given
    range (its record has none): measured now if the source is here."""
    from dronereel import analyze as an
    from dronereel import selector as sel
    src = record["source"]["path"]
    if not os.path.exists(src):
        print("AVISO: el original no está en este equipo; la valoración se guarda sin "
              "características del tramo (servirá menos para aprender).")
        return None, None
    media = probe_media(src)
    a = an.analyze(media, cache_dir=an.CACHE_DIR, progress=an.print_progress)
    print(file=sys.stderr)
    e = record["export"]
    w = sel.describe(a, e["start"], e["end"])
    return w.to_dict()["parts"], w.score


def _load_record(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_rate(args) -> int:
    from dronereel import analyze as an
    from dronereel import dataset as ds
    from dronereel import selector as sel
    from dronereel.safety import fingerprint

    verdict = "accept" if args.accept else "reject" if args.reject else None
    try:
        if args.manual:
            start, end = args.manual
            media = probe_media(args.target)
            cfg = _select_config(args, _read_config(args.config))
            a, choice = _run_selection(args, media, cfg)
            fp = fingerprint(media.path)
            source = {"path": media.path, "sha256": fp.sha256, "duration": media.duration,
                      "fps": media.fps_fraction}
            selection = sel.log_record(choice, a, media.path,
                                       an.content_key(media.path, a.fps, a.width))
            w = sel.describe(a, start, end, cfg)
            entry = ds.manual_entry(source, start, end, w.to_dict()["parts"], selection,
                                    score=args.score, note=args.note or "",
                                    selector_score=w.score)
        else:
            if args.target.lower().endswith((".mp4", ".mov")):
                guess = os.path.splitext(args.target)[0] + ".json"
                if not os.path.exists(guess):
                    raise ds.DatasetError("Pasa el .json del clip (o un vídeo con --manual)")
                args.target = guess
            record = _load_record(args.target)
            features = base = None
            if not ((record.get("selection") or {}).get("best")):
                features, base = _clip_features(record, args)
            entry = ds.rating_entry(record, args.target, score=args.score, verdict=verdict,
                                    prefer=args.prefer, note=args.note or "",
                                    features=features, selector_score=base)
    except (ds.DatasetError, _ConfigError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    ds.append([entry], args.dataset)
    print(f"Guardado en {args.dataset} ({len(ds.read(args.dataset))} valoraciones en total)")
    return 0


def _open_player(path: str) -> None:
    import shutil
    import subprocess
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener):
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ask(prompt: str, valid=None):
    """Input until valid; '' (Enter) means skip, 'q' quits."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("", "q") or valid is None or answer in valid:
            return answer
        print("  Respuesta no válida.")


def cmd_review(args) -> int:
    from dronereel import dataset as ds

    done = ds.rated_clips(ds.read(args.dataset))
    pending = [(p, r) for p, r in ds.find_records(args.output) if p not in done]
    if not pending:
        print(f"No hay clips sin valorar en {args.output}.")
        return 0
    print(f"{len(pending)} clip(s) sin valorar. Enter = saltar, q = salir.\n")
    saved = 0
    for path, rec in pending:
        e = rec["export"]
        print(f"Clip:   {e['path']}")
        print(f"        {e['start']:.2f} → {e['end']:.2f} s de "
              f"{os.path.basename(rec['source']['path'])}")
        cands = (rec.get("selection") or {}).get("candidates") or []
        for i, c in enumerate(cands[1:], start=2):
            print(f"        alternativa {i}: {c['start']:.1f} → {c['end']:.1f} s")
        if args.open and os.path.exists(e["path"]):
            _open_player(e["path"])
        score = _ask("  Puntuación 1-5: ", {"1", "2", "3", "4", "5"})
        if score == "q":
            break
        verdict = _ask("  ¿Lo aceptas? (s/n): ", {"s", "n", "si", "sí", "no"})
        if verdict == "q":
            break
        prefer = ""
        if len(cands) > 1:
            prefer = _ask("  ¿Preferías una alternativa? (número): ",
                          {str(i) for i in range(2, len(cands) + 1)})
            if prefer == "q":
                break
        note = input("  Nota (opcional): ").strip()
        if not (score or verdict or prefer):
            print("  Saltado.\n")
            continue
        features = base = None
        if not (rec.get("selection") or {}).get("best"):
            features, base = _clip_features(rec, args)
        entry = ds.rating_entry(
            rec, path, score=int(score) if score else None,
            verdict=None if not verdict else ("reject" if verdict in ("n", "no") else "accept"),
            prefer=int(prefer) if prefer else None, note=note, features=features,
            selector_score=base)
        ds.append([entry], args.dataset)
        saved += 1
        print("  Guardado.\n")
    print(f"{saved} valoración(es) guardada(s) en {args.dataset}")
    return 0


def cmd_train(args) -> int:
    import numpy as np

    from dronereel import dataset as ds
    from dronereel import ranker as rk

    entries = ds.read(args.dataset)
    print(f"Dataset:    {args.dataset} ({len(entries)} entradas)")
    model, ev = rk.train(entries)
    print(f"Pares:      {ev.get('pairs', 0)} preferencias de {ev.get('videos', 0)} vídeo(s)")
    if "after" in ev:
        print(f"Antes:      el selector acierta el {ev['before'] * 100:.0f} % de tus "
              f"preferencias en vídeos que el modelo no ha visto")
        print(f"Después:    con tu modelo, el {ev['after'] * 100:.0f} % "
              f"({ev['test_pairs']} pares de prueba)")
    if ev.get("reason"):
        print(f"Nota:       {ev['reason']}")
    if model is None:
        print("No se ha entrenado ningún modelo.")
        return 1
    order = np.argsort(-np.abs(model.adj))
    learned = [f"{'+' if model.adj[i] > 0 else '−'}{_FEATURE_NAMES.get(rk.FEATURES[i], rk.FEATURES[i])}"
               for i in order[:4] if abs(model.adj[i]) > 0.005]
    print("Aprendido:  " + (", ".join(learned) if learned else "casi nada (se queda como el selector)"))
    backup = rk.save(model, args.model)
    print(f"Modelo:     {args.model}" + (f" (el anterior queda en {backup})" if backup else ""))
    if model.better:
        print("Se usará por defecto en select y export (desactívalo con --no-model).")
    else:
        print("No se usará por defecto hasta que mejore al selector; pruébalo con "
              f"--model {args.model}")
    return 0


def cmd_dataset(args) -> int:
    from dronereel import dataset as ds
    if args.merge:
        n = ds.merge(args.merge, args.dataset)
        print(f"Añadidas {n} valoraciones nuevas de {args.merge}")
    sm = ds.summary(ds.read(args.dataset))
    print(f"Dataset:    {args.dataset}")
    print(f"Entradas:   {sm['total']} de {sm['sources']} vídeo(s): {sm['ratings']} valoraciones "
          f"de clips, {sm['manual']} tramos elegidos a mano")
    print(f"Veredictos: {sm['accept']} aceptados, {sm['reject']} rechazados, "
          f"{sm['prefer_alternative']} con una alternativa preferida")
    if sm["scores"]:
        print("Puntuación: " + ", ".join(f"{k}★ ×{v}" for k, v in sorted(sm["scores"].items())))
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

    from dronereel.dataset import DEFAULT_PATH as DATASET
    ra = sub.add_parser("rate", help="guarda una valoración en tu dataset")
    ra.add_argument("target", help="el .json (o .mp4) de un clip exportado, o un vídeo con --manual")
    ra.add_argument("--score", type=int, choices=range(1, 6), metavar="1-5", help="puntuación")
    verdict = ra.add_mutually_exclusive_group()
    verdict.add_argument("--accept", action="store_true", help="el clip vale")
    verdict.add_argument("--reject", action="store_true", help="el clip no vale")
    ra.add_argument("--prefer", type=int, help="esta alternativa era mejor (2, 3...)")
    ra.add_argument("--manual", type=float, nargs=2, metavar=("INICIO", "FIN"),
                    help="el tramo que tú elegirías, en s (con un vídeo)")
    ra.add_argument("--note", help="comentario libre")
    ra.add_argument("--min", type=float, help=argparse.SUPPRESS)
    ra.add_argument("--max", type=float, help=argparse.SUPPRESS)
    ra.add_argument("--pre", type=float, help=argparse.SUPPRESS)
    ra.add_argument("--post", type=float, help=argparse.SUPPRESS)
    ra.add_argument("--config", help="YAML con la sección 'select:' (con --manual)")
    ra.add_argument("--software", action="store_true", help=argparse.SUPPRESS)
    ra.add_argument("--no-cache", action="store_true", help=argparse.SUPPRESS)
    ra.add_argument("--dataset", default=DATASET, help=f"archivo del dataset ({DATASET})")
    ra.set_defaults(func=cmd_rate)

    rv = sub.add_parser("review", help="valora uno a uno los clips exportados sin valorar")
    rv.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                    help=f"carpeta de los clips (por defecto {DEFAULT_OUTPUT_DIR})")
    rv.add_argument("--open", action="store_true", help="abrir cada clip en el reproductor")
    rv.add_argument("--dataset", default=DATASET, help=f"archivo del dataset ({DATASET})")
    rv.set_defaults(func=cmd_review)

    dt = sub.add_parser("dataset", help="resumen del dataset; unir el de otro equipo")
    dt.add_argument("--merge", help="otro ratings.jsonl (p. ej. del Mac) para añadir")
    dt.add_argument("--dataset", default=DATASET, help=f"archivo del dataset ({DATASET})")
    dt.set_defaults(func=cmd_dataset)

    from dronereel.ranker import DEFAULT_MODEL
    tr = sub.add_parser("train", help="aprende tu gusto de las valoraciones (ligero, en la Pi)")
    tr.add_argument("--dataset", default=DATASET, help=f"archivo del dataset ({DATASET})")
    tr.add_argument("--model", default=DEFAULT_MODEL, help=f"dónde guardar el modelo ({DEFAULT_MODEL})")
    tr.set_defaults(func=cmd_train)
    for parser in (se, ex):
        parser.add_argument("--model", help="usar este modelo (aunque no esté marcado como mejor)")
        parser.add_argument("--no-model", action="store_true", help="solo el selector hecho a mano")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
