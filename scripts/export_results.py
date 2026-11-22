"""Export small public reports and figures, excluding local datasets and paths."""

import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    import torch
    from neuroavoid.snn_cli import save_confusion
    import matplotlib
    matplotlib.use("Agg")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
        from matplotlib import pyplot as plt
    target = ROOT / "reports"
    target.mkdir(exist_ok=True)
    offline = json.loads((ROOT / "artifacts/benchmark_offline/classification.json").read_text())
    navigation = json.loads((ROOT / "artifacts/benchmark_navigation/summary.json").read_text())
    for source, name in (("artifacts/benchmark_offline/classification.json", "classification_test.json"),
                         ("artifacts/benchmark_navigation/summary.json", "navigation_test.json"),
                         ("artifacts/benchmark_navigation/protocol.json", "frozen_protocol.json"),
                         ("artifacts/benchmark_navigation/episodes.csv", "navigation_episodes.csv"),
                         ("artifacts/snn_comparison.json", "correction_comparison.json")):
        shutil.copyfile(ROOT / source, target / name)
    dataset = json.loads((ROOT / "artifacts/dataset_corrections/manifest.json").read_text())
    summary = {key: dataset[key] for key in ("labels", "fps", "window_ms", "bins", "resolution", "channels", "split_counts", "class_counts")}
    summary["maps"] = [{key: scene[key] for key in ("scene_id", "split", "seed", "geometry_sha256")} for scene in dataset["scene_catalog"]]
    (target / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    version = {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__),
               "numpy_ml": np.__version__, "inference_device": "cpu", "torch_threads": 2,
               "timing_protocol": "batch 1, 10 warmups, 200 test windows, encoding and transport excluded"}
    (target / "environment.json").write_text(json.dumps(version, indent=2) + "\n")
    hashes = {f"{name}.pt": hashlib.sha256((ROOT / "models" / f"{name}.pt").read_bytes()).hexdigest() for name in ("snn", "cnn")}
    (ROOT / "models/SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in hashes.items()))
    for name in ("snn", "cnn"):
        save_confusion(target / f"confusion_{name}_test.png", offline["models"][name]["confusion_matrix"], title=f"{name.upper()} - test")
    with (target / "classification_summary.csv").open("w", newline="") as file:
        fields = ["model", "parameters", "accuracy", "macro_f1", "balanced_accuracy", "mean_latency_ms", "p95_latency_ms", "mean_hidden_spikes"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, result in offline["models"].items():
            writer.writerow({"model": name, "parameters": result["parameters"], "accuracy": result["accuracy"],
                             "macro_f1": result["macro_f1"], "balanced_accuracy": result["balanced_accuracy"],
                             "mean_latency_ms": result["latency_ms"]["mean"], "p95_latency_ms": result["latency_ms"]["p95"],
                             "mean_hidden_spikes": result["hidden_spikes_mean"]})
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = ["snn", "cnn"]
    x = np.arange(2)
    axes[0].bar(x - 0.18, [offline["models"][n]["accuracy"] for n in names], 0.36, label="Accuracy")
    axes[0].bar(x + 0.18, [offline["models"][n]["macro_f1"] for n in names], 0.36, label="Macro-F1")
    axes[0].set(xticks=x, xticklabels=[n.upper() for n in names], ylim=(0, 1), title="Classification: 288 held-out windows")
    axes[0].legend()
    axes[1].bar([n.upper() for n in names], [offline["models"][n]["latency_ms"]["mean"] for n in names])
    axes[1].set(ylabel="ms", title="Model only: warm CPU, batch 1")
    for i, name in enumerate(names):
        subset = [g for g in navigation["groups"] if g["model"] == name]
        positions = np.arange(len(subset)) + (i - 0.5) * 0.35
        axes[2].bar(positions, [g["success_rate"] for g in subset], 0.35, label=name.upper())
        for pos, group in zip(positions, subset):
            axes[2].text(pos, group["success_rate"] + 0.03, f"{group['successes']}/{group['episodes']}", ha="center")
    axes[2].set(xticks=[0, 1], xticklabels=["x1", "x1.75"], ylim=(0, 1.2), title="Closed-loop success: only 2 maps")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(target / "comparison.png", dpi=160)
    plt.close(figure)
    source = ROOT / "artifacts/events_moving/preview.gif"
    if source.exists():
        shutil.copyfile(source, target / "events_demo.gif")
    print(f"Public reports exported to {target}")


if __name__ == "__main__":
    main()
