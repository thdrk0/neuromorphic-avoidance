"""Compare matched closed-loop episodes; assisted runs are rejected."""

import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    def load(root):
        reports = [json.loads(p.read_text()) for p in Path(root).glob("*/episode.json")]
        result = {r["scene_id"]: r for r in reports}
        if not result or len(result) != len(reports):
            raise ValueError("Episodes manquants ou scenes repetees")
        return result

    old, new = load(baseline), load(candidate)
    if set(old) != set(new):
        raise ValueError("Comparer les memes scenes")
    rows = []
    for scene in sorted(old):
        a, b = old[scene], new[scene]
        for report in (a, b):
            if report["teacher_used"] or report["test_used"] or report["split"] == "test":
                raise ValueError("Cette comparaison exige des pilotes autonomes et exclut les cartes test")
        if a["geometry_sha256"] != b["geometry_sha256"]:
            raise ValueError("Geometries differentes")
        for field in ("camera_fps_simulated", "decision_hz_simulated", "window_ms"):
            if a[field] != b[field]:
                raise ValueError(f"Cadences incompatibles : {field}")
        rows.append({"scene": scene, "split": a["split"], "before": a["status"], "after": b["status"],
                     "before_simulation_seconds": a["simulation_seconds"], "after_simulation_seconds": b["simulation_seconds"]})
    return {"episodes": len(rows), "before_successes": sum(r["before"] == "success" for r in rows),
            "after_successes": sum(r["after"] == "success" for r in rows), "episodes_detail": rows,
            "baseline_checkpoint_sha256": next(iter(old.values()))["checkpoint_sha256"],
            "candidate_checkpoint_sha256": next(iter(new.values()))["checkpoint_sha256"],
            "test_used": False, "note": "Cartes de mise au point, pas un benchmark de generalisation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/snn_robot"))
    parser.add_argument("--candidate", type=Path, default=Path("artifacts/snn_corrections_robot"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/snn_comparison.json"))
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
