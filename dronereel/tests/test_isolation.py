"""DroneReel must stay light enough for a Raspberry Pi: importing it must not
pull in upstream's pipeline, its GUI, or the heavy ML stack."""
import json
import os
import subprocess
import sys

from dronereel.tests.conftest import REPO

HEAVY = ["torch", "cv2", "openvino", "PySide6", "whisper", "transformers",
         "selenium", "yt_dlp", "pipeline", "main", "fastapi"]


def test_importing_dronereel_stays_light():
    code = (
        "import sys, json\n"
        "import dronereel.cli, dronereel.media, dronereel.profiles, dronereel.safety\n"
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True,
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": REPO})
    assert json.loads(out.stdout) == []
