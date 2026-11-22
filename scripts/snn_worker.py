"""Persistent CPU inference process; receives event counts, never robot state."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch
    from neuroavoid.snn import encode_events
    from neuroavoid.cnn import model_from_config
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = model_from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(json.dumps({"ready": True, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      "config": checkpoint["config"]}), flush=True)
    with torch.inference_mode():
        for line in sys.stdin:
            try:
                message = json.loads(line)
                if message.get("close"):
                    break
                counts = np.frombuffer(base64.b64decode(message["counts"], validate=True), dtype="<u2").reshape(1, 5, 2, 128, 128)
                tick = time.perf_counter()
                encoded = encode_events(torch.from_numpy(counts.astype(np.float32)))
                encoding = time.perf_counter() - tick
                tick = time.perf_counter()
                logits, stats = model(encoded)
                elapsed = time.perf_counter() - tick
                reply = {"command": checkpoint["config"]["labels"][int(logits.argmax(1))],
                         "scores": logits.softmax(1)[0].tolist(),
                         "hidden_spikes": int(stats["hidden_spikes"][0]) if stats["hidden_spikes"] is not None else None,
                         "input_spikes": int(stats["input_spikes"][0]), "model_seconds": elapsed, "encoding_seconds": encoding}
            except Exception as exc:
                reply = {"error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
