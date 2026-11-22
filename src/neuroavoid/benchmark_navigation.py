"""Frozen final protocol: 2 held-out maps x 2 speeds x 2 models, no teacher."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from neuroavoid.run_snn_robot import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark_navigation"))
    parser.add_argument("--snn", type=Path, default=Path("models/snn.pt"))
    parser.add_argument("--cnn", type=Path, default=Path("models/cnn.pt"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choisir un dossier de sortie vide")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MUJOCO_GL"] = "glfw"
    if os.environ.get("WSL_DISTRO_NAME"):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    models = {"snn": args.snn, "cnn": args.cnn}
    plan = {"scenes": ["test_53", "test_67"], "speed_factors": [1.0, 1.75], "seconds_limit": 30,
            "camera_fps_simulated": 50, "decision_hz_simulated": 10, "window_ms": 100,
            "speed_scaling": "both linear and angular targets, same path curvature, no actuator saturation",
            "checkpoint_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in models.items()},
            "teacher_used": False, "policy_fallback": False,
            "note": "Frozen before held-out evaluation. One deterministic start per map. No retuning on test."}
    (args.output / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for name, checkpoint in models.items():
        for factor in plan["speed_factors"]:
            for scene in plan["scenes"]:
                directory = args.output / f"{name}_{scene}_speed_{factor}"
                directory.mkdir()
                print(f"Benchmark {name} | {scene} | vitesse x{factor}", flush=True)
                report = run_episode(scene, checkpoint, plan["seconds_limit"], directory,
                                     speed_factor=factor, allow_test=True)
                row = {"model": name, "scene": scene, "speed_factor": factor, "status": report["status"],
                       "collided": report["collided"], "simulation_seconds": report["simulation_seconds"],
                       "wall_seconds": report["wall_seconds"], "distance_m": report["distance_m"],
                       "dvs_events": report["dvs_events"], "hidden_spikes": report["hidden_spikes"],
                       "decisions": len(report["decisions"])}
                rows.append(row)
                print(f"  {report['status']} | {report['simulation_seconds']:.2f}s simulees", flush=True)
                (args.output / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = []
    for name in models:
        for factor in plan["speed_factors"]:
            subset = [r for r in rows if r["model"] == name and r["speed_factor"] == factor]
            summary.append({"model": name, "speed_factor": factor, "episodes": len(subset),
                            "successes": sum(r["status"] == "success" for r in subset),
                            "collisions": sum(r["collided"] for r in subset),
                            "timeouts": sum(r["status"] == "timeout" for r in subset),
                            "success_rate": sum(r["status"] == "success" for r in subset) / len(subset)})
    result = {"protocol": plan, "groups": summary, "episodes": rows,
              "limitations": "Two nearby map variants, deterministic starts, no statistical confidence or energy claim."}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
