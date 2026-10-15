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
