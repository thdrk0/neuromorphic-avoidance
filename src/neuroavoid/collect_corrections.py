"""Aggregate expert labels on student-visited training states, preserving splits."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from neuroavoid.dataset import LABELS
from neuroavoid.record_dataset import record_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("artifacts/dataset"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/snn/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_corrections"))
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error("seconds doit etre dans ]0,60]")
    if not args.probabilities or any(not math.isfinite(p) or not 0 <= p <= 1 for p in args.probabilities):
        parser.error("Les probabilites doivent etre dans [0,1]")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choisir un dossier vide")
    manifest = json.loads((args.base / "manifest.json").read_text())
    if not manifest["complete"] or manifest["fps"] != 50 or manifest["bins"] != 5:
        raise ValueError("Dataset complet 50 Hz / 5 bins requis")
    os.environ["MUJOCO_GL"] = "glfw"
    if os.environ.get("WSL_DISTRO_NAME"):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(manifest)
    manifest["complete"] = False
    manifest["parent_manifest_sha256"] = hashlib.sha256((args.base / "manifest.json").read_bytes()).hexdigest()
    manifest["collection_checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    # Paths refer to unchanged original recordings. No test/validation arrays are loaded.
    for episode in manifest["episodes"]:
        for name in ("raw_file", "samples_file"):
            episode[name] = str((args.base / episode[name]).resolve())
    scenes = sorted({e["scene_id"] for e in manifest["episodes"] if e["split"] == "train"})

    def save():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    for scene_index, scene in enumerate(scenes):
        for run, probability in enumerate(args.probabilities):
            directory = args.output / f"correction_{scene}_{run}"
            directory.mkdir()
            print(f"Collecte {scene} | probabilite expert {probability}", flush=True)
            result = record_episode(scene, directory, args.seconds, 50, 5,
                                    checkpoint=args.checkpoint, teacher_probability=probability,
                                    seed=100 + scene_index * 10 + run)
            if result["split"] != "train":
                raise RuntimeError("La collecte doit rester dans train")
            result["raw_file"] = str((directory / "raw.npz").resolve())
            result["samples_file"] = str((directory / "samples.npz").resolve())
            manifest["episodes"].append(result)
            save()
            print(f"  {result['status']} | {result['samples']} fenetres", flush=True)
    manifest["split_counts"] = {s: sum(e["samples"] for e in manifest["episodes"] if e["split"] == s)
                                for s in ("train", "validation", "test")}
    manifest["class_counts"] = {s: {label: sum(e["class_counts"][label] for e in manifest["episodes"] if e["split"] == s)
                                   for label in LABELS} for s in manifest["split_counts"]}
    manifest["complete"] = True
    save()
    print(json.dumps(manifest["split_counts"], indent=2))


if __name__ == "__main__":
    main()
