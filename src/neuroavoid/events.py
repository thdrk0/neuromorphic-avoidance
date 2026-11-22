"""Frame-to-event bridge, preserving the simulator's Python environment."""

import base64
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def event_image(events, width=128, height=128):
    """ON in red, OFF in cyan; white if both occur in the window."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if len(events):
        x, y = events[:, 1].astype(int), events[:, 2].astype(int)
        on = events[:, 3] > 0
        image[y[on], x[on], 0] = 255
        image[y[~on], x[~on], 1:] = 255
    return image


class V2EBridge:
    def __init__(self, width=128, height=128, threshold=0.2):
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Le seuil doit etre positif et fini")
        root = Path(__file__).resolve().parents[2]
        python = Path(os.environ.get("V2E_PYTHON", root / ".venv-v2e/bin/python"))
        if not python.exists():
            raise RuntimeError("Environnement v2e absent : lancer bash scripts/setup_v2e.sh")
        self.width, self.height = width, height
        self.previous = None
        self.proc = subprocess.Popen([str(python), "-u", str(root / "scripts/v2e_worker.py"),
                                      "--width", str(width), "--height", str(height),
                                      "--threshold", str(threshold)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"close": true}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()

    def convert(self, rgb, timestamp):
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Image attendue : RGB uint8 a la resolution configuree")
        if not math.isfinite(timestamp) or timestamp < 0 or (self.previous is not None and timestamp <= self.previous):
            raise ValueError("Les temps simules doivent etre finis, positifs ou nuls et strictement croissants")
        # Conventional luma on display-encoded RGB, not a calibrated irradiance model.
        gray = np.rint(rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        request = {"time": float(timestamp), "gray": base64.b64encode(gray.tobytes()).decode()}
        started = time.perf_counter()
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except BrokenPipeError as exc:
            raise RuntimeError("Le processus v2e s'est arrete ; voir l'erreur dans le terminal") from exc
        if not line:
            raise RuntimeError("v2e n'a pas repondu ; relancer bash scripts/setup_v2e.sh")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        elapsed = time.perf_counter() - started
        events = np.frombuffer(base64.b64decode(reply["events"]), dtype="<f8").reshape(-1, 4).copy()
        if len(events):
            if not np.isfinite(events).all() or np.any(np.diff(events[:, 0]) < 0):
                raise RuntimeError("Evenements invalides ou non ordonnes")
            lower = self.previous if self.previous is not None else 0
            tolerance = max(1e-6, 2 * abs(float(np.spacing(np.float32(timestamp)))))
            if np.any(events[:, 0] < lower - tolerance) or np.any(events[:, 0] > timestamp + tolerance):
                raise RuntimeError("Evenements hors de la fenetre temporelle")
            if (np.any(events[:, 1:3] != np.floor(events[:, 1:3])) or
                np.any(events[:, 1] < 0) or np.any(events[:, 1] >= self.width) or
                np.any(events[:, 2] < 0) or np.any(events[:, 2] >= self.height) or
                not np.isin(events[:, 3], [-1, 1]).all()):
                raise RuntimeError("Coordonnees ou polarites invalides")
        self.previous = timestamp
        return events, float(reply["emulator_seconds"]), elapsed
