"""Check the Git candidate inventory and smoke-test it without local artifacts."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    names = sorted(set(n for n in names if n))
    for name in names:
        parts = Path(name).parts
        assert not any(p in (".venv", ".venv-v2e", "__pycache__", "artifacts") for p in parts), name
        assert not name.startswith(("ros_ws/build/", "ros_ws/install/", "ros_ws/log/")), name
        assert not name.startswith(("reports/", "third_party/wheels/")), name
        assert name != "docs/DEVELOPMENT.md", name
        assert not name.endswith((".whl", ".pyc", ".npz")), name
    total = sum((ROOT / name).stat().st_size for name in names)
    with tempfile.TemporaryDirectory(prefix="neuroavoid-export-") as directory:
        target = Path(directory)
        for name in names:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        env = dict(os.environ, PYTHONPATH=str(target / "src"), V2E_PYTHON=str(ROOT / ".venv-v2e/bin/python"), LIBGL_ALWAYS_SOFTWARE="1")
        subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "neuroavoid.run_snn_robot", "--seconds", "0.24",
                        "--output", "artifacts/smoke"], cwd=target, env=env, check=True)
        report = json.loads((target / "artifacts/smoke/validation_41/episode.json").read_text())
        assert len(report["decisions"]) == 2
        assert report["status"] == "timeout" and not report["collided"]
    print(json.dumps({"status": "OK", "git_candidate_files": len(names), "total_bytes": total,
                      "export_smoke": "source+models only, existing dependency environments reused; not a fresh dependency installation"}, indent=2))


if __name__ == "__main__":
    main()
