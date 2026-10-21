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
  ~0,45 fotogramas/s en 4K a 10 bits (un clip de 20 s a 4K60 ≈ 45 min; ver
  "x265 a 10 bits en ARM", con NEON era el doble de rápido) y ~2,5 GB de
  memoria: cierra Firefox antes. Lo avisa con una estimación.

Siempre: mismos fps que el original (sin remuestrear), audio original copiado
si existe, etiquetas de color del original, sin música, transiciones ni
títulos.

### Vertical 9:16

```bash
.venv/bin/python -m dronereel vertical VIDEO.MP4                  # solo vista previa (PNG)
.venv/bin/python -m dronereel export VIDEO.MP4 --format vertical  # solo el 9:16
.venv/bin/python -m dronereel export VIDEO.MP4 --format both      # horizontal + 9:16
```

El 9:16 es **el mismo tramo continuo** recortado: con `both` son dos
archivos con exactamente los mismos fotogramas (`…_vertical.mp4` al lado del
horizontal). Mantiene toda la altura del original con proporción 9:16 exacta
(4K → 1206x2144, sin reescalar; `--vertical-height 1920` lo reduce a
1080x1920; nunca se amplía). Deja un `…_encuadre.png` con la ventana dibujada
sobre el original en cinco momentos.

`--follow`:

* `smooth` (por defecto): la ventana sigue **lo que destaca** en la imagen
  (saliencia por residuo espectral, Hou & Zhang 2007), con un camino que no
  salta entre dos zonas igual de interesantes, suavizado (σ 1,5 s) y con
  velocidad máxima de 0,12 anchos/s, para no sumar un barrido rápido al
  movimiento real del dron.
* `fixed`: la mejor posición para todo el clip, sin movimiento virtual.
* `center`: recorte central.

Ajustes en la sección `vertical:` de `--config` (`smoothness`, `stickiness`,
`max_speed`, `center_bias`, `height`). Siempre recodifica (con color, el
recorte va antes de la LUT).

**Limitaciones:** no detecta sujetos (personas, coches): sigue contraste,
color y detalle que destacan del resto. En un paisaje uniforme se queda en el
centro. Solo se mueve en horizontal y no corrige el horizonte con el roll.
Probado con vídeos sintéticos (un objeto que cruza el plano queda centrado a
±40 px en una ventana de 396 px); falta verlo con tus vuelos.

### x265 a 10 bits en ARM (Pi y Mac con Apple silicon)

Al probar la F8 apareció un fallo de **x265 4.1** (el de Debian) en la Pi: con
su código NEON a 10 bits escribe un vídeo válido pero con imágenes
incorrectas (en una prueba desapareció una franja de 22 px durante 30
fotogramas). Sin NEON (`asm=none`) y a 8 bits el resultado es correcto.
Por eso `x265_asm: auto` (por defecto) desactiva NEON al recodificar a 10 bits
en ARM. Cuesta velocidad: 4K completo 1,66 → 0,80 fps; 9:16 desde 4K
3,58 → 2,78 fps (medido en esta Pi con un clip sintético 4K60). Si confías en tu x265 (p. ej. una versión que lo corrija),
`x265_asm: default` en `export:` vuelve a activarlo; el test
`test_10bit_reencode_keeps_every_picture` dice si tu equipo lo tiene.

### Color (D-Log M → Rec.709)

Solo con el perfil **confirmado** (`--camera avata2 --profile dlog_m`) y la
**LUT oficial de DJI** para esa cámara (`.cube`, descárgala de dji.com). Ponla
en el campo `lut:` del YAML de la cámara (una ruta relativa se lee desde el
YAML) o pásala con `--lut`.

```bash
# Probar ajustes en fotogramas sueltos (rápido): PNG original | corregido
.venv/bin/python -m dronereel color VIDEO.MP4 --camera avata2 --profile dlog_m \
    --at 42 --exposure 0.3 --highlights 0.5
# Exportar con color (recodifica; deja también un PNG de comparación)
.venv/bin/python -m dronereel export VIDEO.MP4 --camera avata2 --profile dlog_m
```

Ajustes (YAML `color:` o en la línea de comandos): `--exposure` (EV),
`--contrast`, `--saturation`, `--temperature` (−100 frío a 100 cálido),
`--tint` (−100 verde a 100 magenta), `--highlights` (0 a 1: transición suave
a blanco en vez de recorte). Todo se reduce a **una LUT 3D** aplicada por
ffmpeg en una pasada: reproducible e idéntica en todos los fotogramas, sin
modelos generativos. Con ajustes neutros se usa la LUT oficial tal cual.

Reglas: sin perfil confirmado no se aplica nada (ni LUT ni ajustes); una LUT
de conversión nunca se aplica a un perfil `normal`; un perfil logarítmico sin
LUT configurada es un error (para exportar sin color: `--no-color`). En la Pi,
el color añade ~7 % al tiempo de recodificación (0,86 fps en 4K).

Limitaciones: la protección de altas luces no recupera lo que la LUT oficial
ya recorta; el balance de blancos es una aproximación por ganancias de canal.

### Valorar los clips (dataset para aprender tu gusto)

```bash
.venv/bin/python -m dronereel review --open          # uno a uno, abre cada clip
.venv/bin/python -m dronereel rate ~/dronereel-output/CLIP.json --score 4 --accept
.venv/bin/python -m dronereel rate ~/dronereel-output/CLIP.json --prefer 2   # la alternativa 2 era mejor
.venv/bin/python -m dronereel rate VIDEO.MP4 --manual 62 84 --score 5     # el tramo que tú elegirías
.venv/bin/python -m dronereel dataset                 # resumen
```

Se guarda en `~/.local/share/dronereel/ratings.jsonl`: una línea por
valoración, solo se añaden líneas (nunca se reescribe). Cada línea lleva lo
necesario para aprender sin volver a abrir el vídeo: las características del
tramo valorado y de las alternativas del selector, los ajustes usados y la
huella SHA-256 del original y su ruta. **Los vídeos no se copian**: el
dataset son números y referencias; los clips siguen en `~/dronereel-output/`.

Si valoras en dos equipos (Pi y Mac), une los archivos sin duplicados:
`dataset --merge /ruta/al/otro/ratings.jsonl`.

### Aprender tu gusto

```bash
.venv/bin/python -m dronereel train            # entrena y compara antes/después
.venv/bin/python -m dronereel select VIDEO.MP4  # usa tu modelo si mejora; muestra qué
                                                # habría elegido sin él
.venv/bin/python -m dronereel export VIDEO.MP4 --no-model   # solo el selector a mano
```

No entrena desde cero ni nada pesado: parte del selector hecho a mano y
aprende una **corrección** (unos pocos números por característica) a partir de
pares de preferencia: "la alternativa 2 era mejor", "acepto este clip",
"yo habría elegido este tramo", aceptados frente a rechazados. Con pocos datos
la corrección se queda cerca de cero. Segundos en la Pi; el modelo es un JSON
de ~1 KB que funciona igual en el Mac (`~/.local/share/dronereel/model.json`;
el anterior se guarda con fecha al reentrenar).

**Antes y después**: `train` deja fuera vídeos enteros, entrena con el resto y
mide qué porcentaje de tus preferencias en esos vídeos acierta el selector
solo y con tu modelo. El modelo se usa por defecto únicamente si mejora con al
menos 5 vídeos y 30 pares de prueba; si no, solo con `--model`.

Comprobado con un usuario simulado al que le gusta volar cerca: con 8 vídeos
valorados pasa del 46 % al 99 % de aciertos en vídeos no vistos, y elige
tramos mejores en vídeos nuevos. Con un usuario cuyo gusto coincide con el
selector, no aprende nada y no se marca como mejor. Con valoraciones reales
aún no se ha probado.

## En un Mac

```bash
brew install ffmpeg git python
git clone -b dronereel https://github.com/ALBERTOGUER/VideoHighlighter.git
cd VideoHighlighter
python3 -m venv .venv
.venv/bin/pip install -r dronereel/requirements.txt
.venv/bin/python -m pytest dronereel/tests
```

Todo igual que en la Pi; las rutas `~/…` son las mismas. Diferencias: el
análisis decodifica por software (el atajo por hardware es de la Pi), y la
recodificación es mucho más rápida, así que conviene quitar el límite de
memoria de x265: copia el YAML de la cámara a `~/.config/dronereel/profiles/`
y pon `x265_params: ""` en `export:`. Probado solo en la Pi.

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
