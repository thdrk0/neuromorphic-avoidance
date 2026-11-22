"""Final held-out classification and warm CPU inference comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from neuroavoid.cnn import model_from_config
from neuroavoid.snn_cli import classification_metrics, evaluate, load_split, save_confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/dataset_corrections"))
    parser.add_argument("--snn", type=Path, default=Path("models/snn.pt"))
    parser.add_argument("--cnn", type=Path, default=Path("models/cnn.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark_offline"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    digest = hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
    checkpoints = {name: torch.load(path, map_location="cpu", weights_only=True)
                   for name, path in (("snn", args.snn), ("cnn", args.cnn))}
    if any(c["config"]["dataset_manifest_sha256"] != digest for c in checkpoints.values()):
        raise ValueError("Les deux modeles doivent provenir du meme dataset")
    inputs, labels, scenes = load_split(args.dataset, "test")
    report = {"split": "test", "samples": len(labels), "scenes": sorted(set(scenes)),
              "input": "Identical 5x2x32x32 binary event occupancy windows, 100 ms",
              "cpu_threads": 2, "warmup_passes": 10, "latency_scope": "model only, batch 1, encoding/camera/v2e excluded",
              "dataset_manifest_sha256": digest, "models": {}}
    for name, checkpoint in checkpoints.items():
        model = model_from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        metrics, predictions = evaluate(model, inputs, labels)
        timings = []
        with torch.inference_mode():
            for _ in range(10):
                model(inputs[:1])
            for i in range(min(200, len(inputs))):
                started = time.perf_counter()
                model(inputs[i:i + 1])
                timings.append((time.perf_counter() - started) * 1000)
        majority = int(np.argmax(checkpoint["config"]["train_counts"]))
        metrics.update({"parameters": sum(p.numel() for p in model.parameters()),
                        "selected_validation_epoch": checkpoint["epoch"],
                        "latency_ms": {"mean": float(np.mean(timings)), "median": float(np.median(timings)),
                                       "p95": float(np.percentile(timings, 95)), "samples": len(timings)},
                        "majority_baseline": classification_metrics(labels.tolist(), [majority] * len(labels))})
        report["models"][name] = metrics
        save_confusion(args.output / f"confusion_{name}_test.png", metrics["confusion_matrix"], title=f"{name.upper()} - test")
        (args.output / f"predictions_{name}.json").write_text(json.dumps([
            {"index": i, "scene": scenes[i], "reference": int(labels[i]), "prediction": predictions[i]}
            for i in range(len(labels))], indent=2) + "\n")
    (args.output / "classification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
