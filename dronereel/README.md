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
```

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
