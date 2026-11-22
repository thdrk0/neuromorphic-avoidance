"""Record privileged teacher labels with causal MuJoCo/v2e observations."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from neuroavoid.dataset import LABELS, count_frame, make_windows
from neuroavoid.scenes import SCENES, make_scene, scene_manifest


def choose_collection_command(student_command, teacher_command, probability, rng):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probabilite expert invalide")
    use_teacher = bool(rng.random() < probability)
    return (teacher_command if use_teacher else student_command), use_teacher


def record_episode(scene_id, directory, seconds, fps, bins, checkpoint=None, teacher_probability=1.0, seed=0):
    import mujoco
    from neuroavoid.events import V2EBridge
    from neuroavoid.navigation import NavigationTask, ReferenceController
    from neuroavoid.robot import RobotSimulation

    xml, metadata = make_scene(scene_id)
    sim = RobotSimulation(scene_xml=xml)
    task = NavigationTask(timeout=seconds + 0.5)
    for _ in range(100):
        sim.step()
    origin = float(sim.data.time)
    controller = ReferenceController(sim, task)
    (directory / "scene.xml").write_text(xml, encoding="utf-8")
    frames, times, poses, commands, counts, batches, offsets = [], [], [], [], [], [], [0]
    stride = round(200 / fps)
    started = time.perf_counter()
    status = "truncated"
    applied, pending, student_predictions = [], [], []
    rng = np.random.default_rng(seed)
    with ExitStack() as stack:
        student = None
        if checkpoint is not None:
            from neuroavoid.snn_bridge import SNNBridge
            if fps != 50 or bins != 5:
                raise ValueError("La collecte SNN attend 5 bins a 50 Hz")
            student = stack.enter_context(SNNBridge(checkpoint))
        bridge = stack.enter_context(V2EBridge())
        renderer = stack.enter_context(mujoco.Renderer(sim.model, height=128, width=128))
        for index in range(math.floor(seconds * fps + 1e-9) + 1):
            if index:
                for _ in range(stride):
                    sim.step()
                    if sim.collided:
                        status = "collision"
                        break
                if sim.collided:
                    break  # No expert target attached to an impact frame.
            timestamp = index / fps
            if task.outcome(sim) == "success":
                status = "success"
                break  # STOP is terminal behavior, outside the three-class dataset.
            renderer.update_scene(sim.data, camera="onboard")
            rgb = renderer.render().copy()
            events, _, _ = bridge.convert(rgb, timestamp)
            # The target is chosen after observing t, and held during the NEXT interval.
            command = controller.decide(sim)
            frames.append(rgb)
            times.append(timestamp)
            poses.append(sim.pose)
            commands.append(LABELS.index(command))
            frame_counts = count_frame(events)
            counts.append(frame_counts)
            batches.append(events)
            offsets.append(offsets[-1] + len(events))
            if student:
                if index:
                    pending.append(frame_counts)
                if len(pending) == bins:
                    prediction = student.predict(np.stack(pending))
                    pending.clear()
                    selected, use_teacher = choose_collection_command(prediction["command"], command, teacher_probability, rng)
                    sim.set_command(selected)
                    student_predictions.append({"time": timestamp, "student": prediction["command"],
                                                "teacher": command, "teacher_applied": use_teacher})
            else:
                sim.set_command(command)
            applied.append(sim.command)
            if index and index % (fps * 5) == 0:
                print(f"  {scene_id} : {timestamp:.1f} s, {offsets[-1]} evenements", flush=True)
    if not frames:
        raise RuntimeError("Episode sans observation")
    voxels, labels, indices = make_windows(np.stack(counts), commands, bins)
    if not len(labels):
        raise RuntimeError("Episode trop court pour constituer une fenetre")
    np.savez_compressed(directory / "raw.npz", events=np.concatenate(batches), rgb=np.stack(frames),
                        frame_times=np.asarray(times), event_offsets=np.asarray(offsets, dtype=np.int64),
                        labels=np.asarray(commands, dtype=np.int64), poses=np.asarray(poses),
                        applied_commands=np.asarray(applied))
    np.savez_compressed(directory / "samples.npz", voxels=voxels, labels=labels,
                        times=np.asarray(times)[indices], frame_indices=indices)
    result = {**metadata, "status": status, "samples": len(labels), "frames": len(frames),
              "event_count": offsets[-1], "simulated_seconds": float(sim.data.time) - origin,
              "wall_seconds": time.perf_counter() - started,
              "class_counts": {name: int(np.sum(labels == i)) for i, name in enumerate(LABELS)},
              "path": controller.path, "collided": sim.collided,
              "collection": "mixed student/expert" if checkpoint else "expert",
              "teacher_probability": teacher_probability, "collection_seed": seed,
              "student_predictions": student_predictions}
    (directory / "episode.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset"))
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=int, choices=[25, 50, 100, 200], default=50)
    parser.add_argument("--window-ms", type=int, default=100)
    parser.add_argument("--scenes", nargs="+", choices=[s[0] for s in SCENES], default=[s[0] for s in SCENES])
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error("--seconds doit etre dans ]0, 60]")
    bins_exact = args.window_ms * args.fps / 1000
    if bins_exact < 1 or not math.isclose(bins_exact, round(bins_exact)):
        parser.error("La fenetre doit contenir un nombre entier positif de periodes camera")
    if args.seconds * args.fps < round(bins_exact):
        parser.error("La duree ne couvre pas une fenetre complete")
    if len(set(args.scenes)) != len(args.scenes):
        parser.error("Ne pas repeter une scene")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Le dossier de sortie doit etre vide pour ne pas melanger les experiences")
    os.environ.setdefault("MUJOCO_GL", "glfw")
    if os.environ.get("WSL_DISTRO_NAME"):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "labels": list(LABELS), "fps": args.fps,
                "window_ms": args.window_ms, "bins": round(bins_exact), "resolution": [128, 128],
                "channels": ["ON", "OFF"], "representation": "uint16 event counts [sample,time_bin,polarity,y,x]",
                "target_alignment": "Teacher command at observation-window end, applied to the following interval",
                "threshold": 0.2, "noise_enabled": False, "v2e_commit": "cac638444557a224ba6ee5c29260916f152c00bd",
                "scene_catalog": scene_manifest(), "episodes": [], "complete": False}

    def save_manifest():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    for scene_id in args.scenes:
        directory = args.output / scene_id
        directory.mkdir()
        print(f"Enregistrement {scene_id}...", flush=True)
        result = record_episode(scene_id, directory, args.seconds, args.fps, round(bins_exact))
        result["samples_file"] = f"{scene_id}/samples.npz"
        result["raw_file"] = f"{scene_id}/raw.npz"
        manifest["episodes"].append(result)
        save_manifest()
        print(f"  {result['status']} | {result['samples']} fenetres | {result['class_counts']}", flush=True)
    manifest["complete"] = True
    manifest["split_counts"] = {split: sum(e["samples"] for e in manifest["episodes"] if e["split"] == split)
                                for split in ("train", "validation", "test")}
    manifest["class_counts"] = {split: {name: sum(e["class_counts"][name] for e in manifest["episodes"] if e["split"] == split)
                                        for name in LABELS} for split in manifest["split_counts"]}
    save_manifest()
    print(json.dumps({"samples": manifest["split_counts"], "classes": manifest["class_counts"]}, indent=2))
    print(f"Dataset : {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
