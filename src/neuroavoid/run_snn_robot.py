"""Closed-loop event-camera/SNN driving, with privileged state only for scoring."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from neuroavoid.dataset import count_frame
from neuroavoid.events import V2EBridge, event_image
from neuroavoid.scenes import SCENES, make_scene
from neuroavoid.snn_bridge import SNNBridge


def run_episode(scene_id, checkpoint, seconds, directory, show=False, speed_factor=1.0, allow_test=False):
    import mujoco
    from neuroavoid.navigation import NavigationTask
    from neuroavoid.robot import RobotSimulation
    xml, metadata = make_scene(scene_id)
    if metadata["split"] == "test" and not allow_test:
        raise ValueError("Les cartes test sont reservees au benchmark final")
    sim = RobotSimulation(scene_xml=xml)
    for _ in range(100):
        sim.step()
    origin = float(sim.data.time)
    task = NavigationTask(timeout=origin + seconds)
    trajectory, decisions, pending = [], [], []
    event_count, hidden_spikes, distance = 0, 0, 0.0
    render_times, emulation_times, cycle_times = [], [], []
    previous = sim.pose[:2]
    quitting = False

    def on_key(key):
        nonlocal quitting
        if key == 88:
            quitting = True

    started = time.perf_counter()
    viewer = None
    status = "running"
    with ExitStack() as stack:
        snn = stack.enter_context(SNNBridge(checkpoint))
        model_type = snn.metadata["config"].get("model_type", "snn")
        if model_type == "cnn":
            hidden_spikes = None
        bridge = stack.enter_context(V2EBridge())
        renderer = stack.enter_context(mujoco.Renderer(sim.model, height=128, width=128))
        if show:
            from neuroavoid.viewer import RobotViewer
            viewer = stack.enter_context(RobotViewer(sim.model, sim.data, on_key))
            viewer.cam.lookat[:] = [0, 0, 0]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7, 135, -55
            viewer.help_text = "Pilotage SNN | fenetre 100 ms | X ou fermer : quitter"
            viewer.sync()
        frame_index = 0
        rgb = None
        last_events = np.empty((0, 4))
        while status == "running":
            tick = time.perf_counter()
            if viewer and (quitting or not viewer.is_running()):
                status = "aborted"
                break
            if frame_index:
                for _ in range(4):
                    sim.step()
                    distance += math.dist(previous, sim.pose[:2])
                    previous = sim.pose[:2]
                    status = task.outcome(sim)
                    if status != "running":
                        break
                if status != "running":
                    break
            timestamp = frame_index / 50
            render_started = time.perf_counter()
            renderer.update_scene(sim.data, camera="onboard")
            rgb = renderer.render().copy()
            render_times.append(time.perf_counter() - render_started)
            events, emulation, _ = bridge.convert(rgb, timestamp)
            emulation_times.append(emulation)
            event_count += len(events)
            last_events = events
            if frame_index:  # frame zero initializes v2e, never a model input.
                pending.append(count_frame(events))
            if len(pending) == 5:
                reply = snn.predict(np.stack(pending))
                pending.clear()
                sim.set_command(reply["command"], speed_factor=speed_factor)
                if reply["hidden_spikes"] is not None:
                    hidden_spikes += reply["hidden_spikes"]
                decisions.append({"time": timestamp, **reply})
                if viewer:
                    viewer.help_text = f"Pilotage SNN : {reply['command']} | t={timestamp:.1f}s\nX ou fermer : quitter"
            trajectory.append({"time": timestamp, "pose": sim.pose, "command": sim.command})
            if viewer:
                viewer.sync()
                time.sleep(max(0, 0.02 - (time.perf_counter() - tick)))
            cycle_times.append(time.perf_counter() - tick)
            if frame_index and frame_index % 250 == 0:
                print(f"  {timestamp:.1f}s : {sim.command} | {len(decisions)} decisions", flush=True)
            frame_index += 1
        sim.set_command("STOP")
        if rgb is not None:
            Image.fromarray(np.concatenate([rgb, event_image(last_events)], axis=1)).resize(
                (768, 384), Image.Resampling.NEAREST).save(directory / "camera_events.png")
        if viewer:
            viewer.sync()
            Image.fromarray(viewer.screenshot()).save(directory / "final_view.png")
        worker_metadata = snn.metadata
    def mean_ms(values):
        return float(np.mean(values) * 1000) if values else None

    report = {**metadata, "status": status, "collided": sim.collided, "contacts": sorted(sim.contacts),
              "simulation_seconds": float(sim.data.time) - origin, "wall_seconds": time.perf_counter() - started,
              "goal_distance_m": math.dist(sim.pose[:2], task.goal), "distance_m": distance,
              "dvs_events": event_count, "hidden_spikes": hidden_spikes, "decisions": decisions,
              "trajectory": trajectory, "checkpoint_sha256": worker_metadata["checkpoint_sha256"],
              "camera_fps_simulated": 50, "decision_hz_simulated": 10, "window_ms": 100,
              "timeout_simulation_seconds": seconds,
              "startup": "STOP during first 100 ms, until one full event window exists",
              "teacher_used": False, "test_used": metadata["split"] == "test",
              "model_type": model_type, "speed_factor": speed_factor,
              "straight_speed_m_s": 0.35 * speed_factor,
              "mean_camera_render_ms": mean_ms(render_times),
              "mean_v2e_ms": mean_ms(emulation_times),
              "mean_frame_cycle_ms": mean_ms(cycle_times),
              "note": "Pose et carte reservees a la simulation, au journal et au score, jamais envoyees au SNN."}
    (directory / "episode.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=[s[0] for s in SCENES if s[1] != "test"], default=["validation_41"])
    parser.add_argument("--checkpoint", type=Path, default=Path("models/snn.pt"))
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/snn_robot"))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        parser.error("seconds doit etre dans ]0,60]")
    if not math.isfinite(args.speed_factor) or not 0 < args.speed_factor <= 1.9:
        parser.error("speed-factor doit etre dans ]0,1.9]")
    if len(set(args.scenes)) != len(args.scenes):
        parser.error("Ne pas repeter une scene")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choisir un dossier de sortie vide")
    os.environ["MUJOCO_GL"] = "glfw"
    if os.environ.get("WSL_DISTRO_NAME"):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for scene in args.scenes:
        print(f"Camera -> v2e -> SNN -> moteurs : {scene}", flush=True)
        directory = args.output / scene
        directory.mkdir()
        report = run_episode(scene, args.checkpoint, args.seconds, directory, args.viewer, args.speed_factor)
        reports.append(report)
        print(f"{scene} : {report['status']} | {report['simulation_seconds']:.2f}s simulees", flush=True)
        if report["status"] == "aborted":
            break
    counts = {state: sum(r["status"] == state for r in reports) for state in ("success", "collision", "timeout", "aborted")}
    summary = {"episodes": len(reports), "counts": counts, "success_rate": counts["success"] / len(reports),
               "teacher_used": False, "test_used": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
