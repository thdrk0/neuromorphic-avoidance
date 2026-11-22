"""Connect the MuJoCo environment to the existing PyTorch environment."""

import base64
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


class SNNBridge:
    def __init__(self, checkpoint):
        root = Path(__file__).resolve().parents[2]
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint absent : {checkpoint}")
        python = os.environ.get("V2E_PYTHON", str(root / ".venv-v2e/bin/python"))
        self.proc = subprocess.Popen([python, "-u", str(root / "scripts/snn_worker.py"), str(checkpoint)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.metadata = self.receive()
            if not self.metadata.get("ready"):
                raise RuntimeError("Le modele n'est pas pret")
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def receive(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Le processus SNN s'est arrete ; consulter l'erreur du terminal")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def predict(self, counts):
        if counts.shape != (5, 2, 128, 128) or counts.dtype != np.uint16:
            raise ValueError("Fenetre attendue : uint16 [5,2,128,128]")
        message = {"counts": base64.b64encode(counts.astype("<u2").tobytes()).decode()}
        started = time.perf_counter()
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        reply = self.receive()
        if reply["command"] not in ("LEFT", "STRAIGHT", "RIGHT"):
            raise RuntimeError("Commande SNN invalide")
        reply["bridge_seconds"] = time.perf_counter() - started
        return reply

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
