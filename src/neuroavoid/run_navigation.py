"""Run reference episodes, optionally showing the robot in a window."""

import argparse
import json
import math
import os
from pathlib import Path
import time


def run_episode(task, seed=0, policy="reference", viewer_enabled=False):
    import mujoco
    import numpy as np
    from neuroavoid.config import PHYSICS_TIMESTEP
    from neuroavoid.navigation import ReferenceController
    from neuroavoid.robot import RobotSimulation

    sim = RobotSimulation()
    rng = np.random.default_rng(seed)
    sim.data.qpos[1] = float(rng.uniform(-0.15, 0.15))
    sim.model.site_pos[sim.model.site("start_zone").id, 1] = sim.data.qpos[1]
    sim.model.site_pos[sim.model.site("goal_zone").id, :2] = task.goal
    mujoco.mj_forward(sim.model, sim.data)
    controller = ReferenceController(sim, task) if policy == "reference" else None
    viewer = None
    quitting = False

    def on_key(key):
        nonlocal quitting
        if key == 88:
            quitting = True

    if viewer_enabled:
        from neuroavoid.viewer import RobotViewer
        viewer = RobotViewer(sim.model, sim.data, on_key)
        viewer.help_text = "Controleur de reference automatique\nZone verte : arrivee  |  X ou fermer : quitter"
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7, 135, -55
        viewer.sync()
    trajectory, decisions = [], {name: 0 for name in ("LEFT", "STRAIGHT", "RIGHT", "STOP")}
    distance = 0.0
    previous = sim.pose[:2]
    outcome = task.outcome(sim)
    started = time.monotonic()
    try:
        while outcome == "running":
            tick = time.monotonic()
            if viewer and (quitting or not viewer.is_running()):
                outcome = "aborted"
                break
            if sim.steps % sim.camera_stride == 0:
                command = controller.decide(sim) if controller else "STRAIGHT" if policy == "straight" else "STOP"
                sim.set_command(command)
                decisions[command] += 1
                trajectory.append({"time": float(sim.data.time), "pose": sim.pose, "command": command})
            frame_due = sim.step()
            distance += math.dist(previous, sim.pose[:2])
            previous = sim.pose[:2]
            outcome = task.outcome(sim)
            if viewer:
                if frame_due:
                    viewer.sync()
                time.sleep(max(0, PHYSICS_TIMESTEP - (time.monotonic() - tick)))
        sim.set_command("STOP")
        if viewer:
            viewer.sync()
    finally:
        if viewer:
            viewer.__exit__(None, None, None)
    trajectory.append({"time": float(sim.data.time), "pose": sim.pose, "command": "STOP"})
    report = {"seed": seed, "policy": policy, "status": outcome,
              "simulation_seconds": float(sim.data.time), "wall_seconds": time.monotonic() - started,
              "distance_m": distance, "goal_distance_m": math.dist(sim.pose[:2], task.goal),
              "collided": sim.collided, "contacts": sorted(sim.contacts), "decisions": decisions,
              "path": controller.path if controller else [], "trajectory": trajectory}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["reference", "straight", "stop"], default="reference")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/navigation"))
    args = parser.parse_args()
    if args.episodes < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("episodes et timeout doivent etre positifs et finis")
    if args.viewer:
        print("Navigation automatique : zone verte = arrivee ; X ou fermer = interrompre.", flush=True)
        os.environ["MUJOCO_GL"] = "glfw"
        if os.environ.get("WSL_DISTRO_NAME"):
            os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    from neuroavoid.navigation import NavigationTask
    task = NavigationTask(timeout=args.timeout)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index in range(args.episodes):
        report = run_episode(task, args.seed + index, args.policy, args.viewer)
        reports.append(report)
        (args.output / f"episode_{index:03d}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Episode {index + 1} : {report['status']} | {report['simulation_seconds']:.2f} s simulees | {report['distance_m']:.2f} m", flush=True)
        if report["status"] == "aborted":
            break
    counts = {state: sum(r["status"] == state for r in reports) for state in ("success", "collision", "timeout", "aborted")}
    summary = {"episodes": len(reports), "policy": args.policy, "goal": list(task.goal),
               "goal_radius_m": task.goal_radius, "timeout_simulation_seconds": task.timeout,
               "counts": counts, "success_rate": counts["success"] / len(reports),
               "note": "Meme carte, departs legerement varies ; controleur avec carte et pose privilegiees."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
