"""Train, validate and inspect a first SNN without touching held-out test data."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch
from torch import nn

from neuroavoid.dataset import EventDataset, LABELS
from neuroavoid.snn import SmallSNN, encode_events
from neuroavoid.cnn import SmallCNN, model_from_config


def load_split(root, split):
    data = EventDataset(root, split)
    if not data.manifest["complete"] or not len(data):
        raise ValueError("Dataset incomplet ou partition vide")
    if data.manifest["bins"] != 5 or data.manifest["window_ms"] != 100 or data.manifest["fps"] != 50:
        raise ValueError("Ce premier SNN attend 5 bins de 20 ms (fenetre 100 ms)")
    inputs, labels, scenes = [], [], []
    # Load and encode one episode at a time; never shuffle compressed NPZ reads.
    for episode in data.episodes:
        with np.load(Path(root) / episode["samples_file"], allow_pickle=False) as archive:
            counts = archive["voxels"]
            for start in range(0, len(counts), 32):
                inputs.append(encode_events(torch.from_numpy(counts[start:start + 32].astype(np.float32))))
            labels.append(torch.from_numpy(archive["labels"].astype(np.int64)))
            scenes.extend([episode["scene_id"]] * len(counts))
    return torch.cat(inputs), torch.cat(labels), scenes


def classification_metrics(labels, predictions):
    matrix = np.zeros((3, 3), np.int64)
    np.add.at(matrix, (np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    diagonal = matrix.diagonal()
    recall = np.divide(diagonal, support, out=np.zeros(3, float), where=support > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros(3, float), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3, float), where=precision + recall > 0)
    return {"accuracy": float(diagonal.sum() / max(1, matrix.sum())),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "confusion_matrix": matrix.tolist(), "confusion_rows": "reference", "confusion_columns": "prediction",
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(LABELS)}}


def evaluate(model, inputs, labels, batch_size=32):
    model.eval()
    predictions, hidden, incoming = [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            x, y = inputs[start:start + batch_size], labels[start:start + batch_size]
            logits, stats = model(x)
            predictions.extend(logits.argmax(1).tolist())
            if stats["hidden_spikes"] is not None:
                hidden.extend(stats["hidden_spikes"].tolist())
            incoming.extend(stats["input_spikes"].tolist())
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
    result = classification_metrics(labels.tolist(), predictions)
    result.update({"cross_entropy": total_loss / len(inputs), "samples": len(inputs),
                   "hidden_spikes_mean": float(np.mean(hidden)) if hidden else None,
                   "hidden_spikes_total": int(sum(hidden)) if hidden else None,
                   "input_spikes_mean": float(np.mean(incoming))})
    return result, predictions


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choisir un dossier de sortie vide pour conserver les experiences")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    print("Chargement train / validation uniquement...", flush=True)
    train_x, train_y, _ = load_split(args.dataset, "train")
    val_x, val_y, scenes = load_split(args.dataset, "validation")
    counts = torch.bincount(train_y, minlength=3)
    if (counts == 0).any():
        raise ValueError("Chaque classe doit etre representee dans train")
    weights = len(train_y) / (3 * counts.float())
    model = SmallCNN() if args.model == "cnn" else SmallSNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    config = {"model_type": args.model,
              "architecture": "conv12-LIF-conv24-LIF-linear3" if args.model == "snn" else "stack10-conv12-ReLU-conv24-ReLU-linear3",
              "beta": 0.8, "threshold": 0.5,
              "encoding": "4x4 spatial occupancy -> binary spikes, 5 x 20 ms bins",
              "readout": "continuous logits averaged over time; hidden layers emit binary spikes" if args.model == "snn" else "continuous CNN logits, no hidden spikes",
              "state_reset": "every 100 ms window", "labels": list(LABELS),
              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "class_weights": weights.tolist(), "train_counts": counts.tolist(),
              "dataset_manifest_sha256": hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest(),
              "torch_version": str(torch.__version__), "numpy_version": np.__version__,
              "threads": args.threads, "device": "cpu", "test_used": False,
              "parameters": sum(p.numel() for p in model.parameters())}
    save_json(args.output / "config.json", config)
    baseline = classification_metrics(val_y.tolist(), [int(counts.argmax())] * len(val_y))
    best_score, history = (-1.0, -1.0, -float("inf")), []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        losses = []
        for indices in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(train_x[indices])
            loss = criterion(logits, train_y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Perte non finie")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, _ = evaluate(model, val_x, val_y, args.batch_size)
        row = {"epoch": epoch, "weighted_train_loss": float(np.mean(losses)), "validation": metrics}
        history.append(row)
        score = (metrics["macro_f1"], metrics["balanced_accuracy"], -metrics["cross_entropy"])
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch}, args.output / "best.pt")
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | perte {row['weighted_train_loss']:.4f} | validation accuracy {metrics['accuracy']:.3f} | macro-F1 {metrics['macro_f1']:.3f}", flush=True)
        save_json(args.output / "history.json", history)
    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, predictions = evaluate(model, val_x, val_y, args.batch_size)
    model.eval()
    timings = []
    with torch.inference_mode():
        for _ in range(10):
            model(val_x[:1])
        for i in range(min(100, len(val_x))):
            tick = time.perf_counter()
            model(val_x[i:i + 1])
            timings.append(time.perf_counter() - tick)
    metrics.update({"split": "validation", "selected_epoch": checkpoint["epoch"], "majority_baseline": baseline,
                    "model_only_latency_ms": {"mean": float(np.mean(timings) * 1000), "p95": float(np.percentile(timings, 95) * 1000)},
                    "training_wall_seconds": time.perf_counter() - started, "test_used": False,
                    "note": "Validation d'imitation sur une seule carte. Pas de navigation autonome, ni de mesure energetique."})
    save_json(args.output / "validation.json", metrics)
    save_json(args.output / "predictions_validation.json", [{"index": i, "scene_id": scenes[i],
              "reference": LABELS[int(val_y[i])], "prediction": LABELS[predictions[i]]} for i in range(len(val_y))])
    save_confusion(args.output / "confusion_validation.png", metrics["confusion_matrix"])
    print(json.dumps(metrics, indent=2))


def save_confusion(path, matrix, title="SNN - validation"):
    import matplotlib
    matplotlib.use("Agg")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
        from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
             xlabel="Prediction", ylabel="Reference", title=title)
    for i in range(3):
        for j in range(3):
            axis.text(j, i, str(matrix[i][j]), ha="center", va="center",
                      color="white" if matrix[i][j] > np.max(matrix) / 2 else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def load_checkpoint(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected = checkpoint["config"]["dataset_manifest_sha256"]
    actual = hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("Ce checkpoint ne correspond pas au manifeste du dataset")
    model = model_from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def validate(args):
    model, checkpoint = load_checkpoint(args)
    inputs, labels, _ = load_split(args.dataset, "validation")
    metrics, _ = evaluate(model, inputs, labels)
    metrics.update({"split": "validation", "selected_epoch": checkpoint["epoch"], "test_used": False})
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "validation.json", metrics)
    save_confusion(args.output / "confusion_validation.png", metrics["confusion_matrix"])
    print(json.dumps(metrics, indent=2))


def predict(args):
    data = EventDataset(args.dataset, args.split)
    sample = data[args.index]
    model, _ = load_checkpoint(args)
    inputs = encode_events(torch.from_numpy(sample["events"].astype(np.float32))[None])
    with torch.inference_mode():
        logits, stats = model(inputs)
    print(json.dumps({"scene_id": sample["scene_id"], "time": sample["time"], "reference": LABELS[sample["label"]],
                      "prediction": LABELS[int(logits.argmax(1))], "scores_softmax": logits.softmax(1)[0].tolist(),
                      "hidden_spikes": int(stats["hidden_spikes"][0]) if stats["hidden_spikes"] is not None else None,
                      "input_spikes": int(stats["input_spikes"][0])}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", type=Path, default=Path("artifacts/dataset"))
    training.add_argument("--output", type=Path, default=Path("artifacts/snn"))
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--model", choices=["snn", "cnn"], default="snn")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--dataset", type=Path, default=Path("artifacts/dataset"))
    prediction.add_argument("--checkpoint", type=Path, default=Path("artifacts/snn/best.pt"))
    prediction.add_argument("--split", choices=["train", "validation"], default="validation")
    prediction.add_argument("--index", type=int, default=0)
    validation = commands.add_parser("validate")
    validation.add_argument("--dataset", type=Path, default=Path("artifacts/dataset"))
    validation.add_argument("--checkpoint", type=Path, default=Path("artifacts/snn/best.pt"))
    validation.add_argument("--output", type=Path, default=Path("artifacts/snn_recheck"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads doit etre positif")
    if args.command == "train" and (args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("epochs, batch-size et lr doivent etre positifs")
    torch.set_num_threads(args.threads)
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
