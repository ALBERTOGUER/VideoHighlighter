# DroneReel-AI

Un único clip continuo, el mejor momento, a partir de un vídeo del DJI Avata 2
o del DJI Osmo Nano. Construido sobre VideoHighlighter, pero sin su interfaz ni
sus dependencias pesadas: funciona en una Raspberry Pi.

## Instalación (Raspberry Pi, Debian 13)

```bash
cd ~/dronereel-ai
python3 -m venv .venv
.venv/bin/pip install -r dronereel/requirements.txt
```

Requiere `ffmpeg` y `ffprobe` del sistema (`sudo apt install ffmpeg`).

## Uso

```bash
.venv/bin/python -m dronereel cameras                 # cámaras y perfiles
.venv/bin/python -m dronereel probe VIDEO.MP4          # metadatos y decisión de color
.venv/bin/python -m dronereel probe VIDEO.MP4 --camera avata2 --profile dlog_m
.venv/bin/python -m dronereel analyze VIDEO.MP4        # medidas del vídeo completo
.venv/bin/python -m dronereel analyze VIDEO.MP4 --csv ~/dronereel-output/medidas.csv
```

`analyze` mide 5 muestras por segundo a 384 px: movimiento de cámara (giro,
alabeo, avance, coherencia, paralaje, brusquedad), nitidez, exposición, luces
quemadas, colorido, detalle y cortes. Solo lee el vídeo; guarda la caché en
`~/.cache/dronereel/analysis/`, nunca junto al original. En la Raspberry Pi 5,
con HEVC usa el decodificador por hardware (~0,9 s por segundo de 4K60);
H.264 va por software (~6 s por segundo de 4K60). Cierra Firefox para
liberar memoria y CPU en vídeos largos.

```bash
.venv/bin/python -m dronereel select VIDEO.MP4 --min 10 --max 30
.venv/bin/python -m dronereel select VIDEO.MP4 --min 15 --max 0 --pre 3 --post 2
.venv/bin/python -m dronereel select VIDEO.MP4 --config mis_pesos.yaml
```

`select` elige **un único segmento continuo**: prueba todas las ventanas entre
`--min` y `--max` segundos (`--max 0` = sin límite) dentro de una misma toma
(nunca cruza un corte), las puntúa y se queda con la mejor. No exporta nada
todavía (F4); escribe un registro JSON en `~/dronereel-output/` con el
segmento, el desglose de su puntuación y las alternativas que no ganaron.

La puntuación combina exposición, nitidez, estabilidad, dinamismo (satura y se
multiplica por la coherencia: rápido y caótico no puntúa), giros/alabeo
limpios, detalle y paralaje (peso bajo: más cerca no es mejor). El movimiento
desconocido (cielo abierto, agua) cuenta como neutro. Penaliza los tramos
vacíos y el peor segundo de la ventana, y deja `--pre`/`--post` segundos de
contexto alrededor del mejor momento cuando este destaca. Los bordes se
mueven hasta `edge_slack` s hacia el instante más tranquilo.

Pesos y opciones en un YAML (sección `select:`), por ejemplo:

```yaml
select:
  min_duration: 12
  max_duration: 25
  weights: {stability: 2.0, proximity: 0.0}
```

**Limitación:** los pesos están puestos a mano y probados con datos
sintéticos; no reconoce maniobras con nombre (flip, dive), solo mide
velocidades de rotación y avance. Se ajustarán con tus valoraciones (F6–F7).

```bash
.venv/bin/python -m dronereel export VIDEO.MP4 --min 10 --max 30     # elige y exporta
.venv/bin/python -m dronereel export VIDEO.MP4 --start 62 --end 84    # tramo a mano
.venv/bin/python -m dronereel export VIDEO.MP4 --mode reencode        # corte exacto
```

`export` escribe **un solo archivo** con un único tramo continuo, en
`~/dronereel-output/` (nunca en la carpeta del original, nunca encima de otro
archivo), y a su lado un `.json` con el origen (y su SHA-256), los tiempos, la
selección y el comando de ffmpeg. Al terminar comprueba que el original no ha
cambiado.

* `--mode auto` (por defecto): copia directa si la imagen no cambia. Sin
  recomprimir, máxima calidad, menos de un segundo incluso en 4K. Solo puede
  cortar en fotogramas clave, así que el tramo se amplía un poco (lo avisa).
* `--mode reencode`: el tramo exacto al fotograma, en H.265 10 bits
  (`--codec h264` y `--bit-depth 8` como opción), `crf 18`. Lento en la Pi:
  ~0,9 fotogramas/s en 4K (un clip de 20 s a 4K60 ≈ 22 min) y ~2,5 GB de
  memoria: cierra Firefox antes. Lo avisa con una estimación.

Siempre: mismos fps que el original (sin remuestrear), audio original copiado
si existe, etiquetas de color del original, sin música, transiciones ni
títulos. El color (F5) todavía no se aplica.

`probe` solo lee el archivo. El color se transforma **únicamente** con un
perfil confirmado: por ti (`--camera` + `--profile`) o por una regla marcada
`verified: true` en el YAML de la cámara. Si no, se conserva el color original.

## Cámaras

`dronereel/profiles/*.yaml`, uno por cámara. Para cambiar ajustes sin tocar el
repositorio, copia el YAML a `~/.config/dronereel/profiles/` y edítalo allí:
tiene prioridad sobre el incluido.

## Tests

```bash
.venv/bin/python -m pytest dronereel/tests
```

Usan vídeos sintéticos generados con ffmpeg; nunca vídeos reales.
