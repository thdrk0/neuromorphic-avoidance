"""Persistent JSON-lines bridge to the pinned, upstream v2e emulator."""

import argparse
import base64
from contextlib import redirect_stdout
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/v2e"))


def emulator(width, height, threshold):
    class QuietStaticFrames(logging.Filter):
        def filter(self, record):
            return not record.getMessage().startswith("no signal events generated")

    logging.getLogger("v2ecore.emulator").addFilter(QuietStaticFrames())
    with redirect_stdout(sys.stderr):
        import torch
        from v2ecore.emulator import EventEmulator
        torch.set_num_threads(1)
        return EventEmulator(pos_thres=threshold, neg_thres=threshold,
                             sigma_thres=0, cutoff_hz=0, leak_rate_hz=0,
                             shot_noise_rate_hz=0, refractory_period_s=0,
                             seed=1, device="cpu", output_width=width, output_height=height)


def self_test():
    import numpy as np
    emu = emulator(8, 8, 0.2)
    with redirect_stdout(sys.stderr):
        first = emu.generate_events(np.full((8, 8), 100, np.uint8), 0)
        static = emu.generate_events(np.full((8, 8), 100, np.uint8), 0.02)
        on = emu.generate_events(np.full((8, 8), 200, np.uint8), 0.04)
        off = emu.generate_events(np.full((8, 8), 100, np.uint8), 0.06)
    assert first is None or len(first) == 0
    assert static is None or len(static) == 0
    assert len(on) > 0 and np.all(on[:, 3] == 1)
    assert len(off) > 0 and np.all(off[:, 3] == -1)
    assert np.all(np.diff(on[:, 0]) >= 0)
    assert np.all(np.diff(off[:, 0]) >= 0)
    assert np.all((on[:, 0] >= 0.02) & (on[:, 0] <= 0.04 + 1e-7))
    assert np.all((off[:, 0] >= 0.04) & (off[:, 0] <= 0.06 + 1e-7))
    emu.cleanup()
    return {"static_events": 0, "on_events": len(on), "off_events": len(off), "status": "OK"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    if args.check or args.self_test:
        result = self_test()
        result["versions"] = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "numba", "screeninfo")}
        import cv2
        result["versions"]["opencv"] = cv2.__version__
        print(json.dumps(result), flush=True)
        return
    import numpy as np
    emu = emulator(args.width, args.height, args.threshold)
    previous = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("close"):
                    break
                timestamp = float(request["time"])
                if not math.isfinite(timestamp) or timestamp < 0 or (previous is not None and timestamp <= previous):
                    raise ValueError("Timestamps must be finite, non-negative and strictly increasing")
                frame = np.frombuffer(base64.b64decode(request["gray"], validate=True), dtype=np.uint8).reshape(args.height, args.width)
                started = time.perf_counter()
                with redirect_stdout(sys.stderr):
                    events = emu.generate_events(frame, timestamp)
                elapsed = time.perf_counter() - started
                previous = timestamp
                if events is None:
                    events = np.empty((0, 4))
                # v2e uses float32 internally; serialize explicitly as little-endian float64.
                events = np.asarray(events, dtype="<f8").reshape(-1, 4)
                reply = {"events": base64.b64encode(events.tobytes()).decode(), "emulator_seconds": elapsed}
            except Exception as exc:
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(reply), flush=True)
    finally:
        with redirect_stdout(sys.stderr):
            emu.cleanup()


if __name__ == "__main__":
    main()
