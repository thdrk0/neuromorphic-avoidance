"""Exercise physics and camera rendering without opening a window."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys

from neuroavoid.config import CAMERA_HEIGHT, CAMERA_WIDTH, PHYSICS_TIMESTEP

SCENE = """
<mujoco>
  <visual><global offwidth="128" offheight="128"/></visual>
  <worldbody>
    <light pos="0 -1 3"/>
    <camera name="check" pos="0 -2 1.5" xyaxes="1 0 0 0 0.6 0.8"/>
    <geom type="plane" size="3 3 0.1" rgba="0.3 0.4 0.5 1"/>
    <body pos="0 0 0.6">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1" rgba="1 0.25 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["glfw", "egl", "osmesa"],
                        default=os.environ.get("MUJOCO_GL", "glfw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    # Select OpenGL before importing MuJoCo.
    os.environ["MUJOCO_GL"] = args.backend
    try:
        import mujoco
        import numpy as np
        from PIL import Image

        model = mujoco.MjModel.from_xml_string(SCENE)
        model.opt.timestep = PHYSICS_TIMESTEP
        data = mujoco.MjData(model)
        for _ in range(200):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isclose(data.time, 1.0):
            raise RuntimeError("Simulation physique invalide.")
        if not 0.08 < data.qpos[2] < 0.12 or data.ncon == 0:
            raise RuntimeError("La sphere ne repose pas sur le sol.")
        print("OK : physique, 1 seconde simulee, contact avec le sol.")
        with mujoco.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH) as renderer:
            renderer.update_scene(data, camera="check")
            frame = renderer.render().copy()
        if frame.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3) or frame.std() < 1:
            raise RuntimeError("Image camera invalide ou uniforme.")
        args.output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(args.output / "camera_check.png")
        report = {"python": platform.python_version(), "mujoco": mujoco.__version__,
                  "numpy": np.__version__, "backend": args.backend,
                  "simulation_seconds": float(data.time), "contacts": int(data.ncon),
                  "image_shape": list(frame.shape)}
        (args.output / "environment.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"OK : camera {CAMERA_WIDTH}x{CAMERA_HEIGHT}, backend {args.backend}.")
        print(f"Resultats : {args.output.resolve()}")
        return 0
    except Exception as exc:
        print(f"ECHEC : {exc}", file=sys.stderr)
        if isinstance(exc, ModuleNotFoundError):
            print("Installer les dependances avec uv sync, puis relancer avec uv run.", file=sys.stderr)
        else:
            print("Si le rendu echoue, essayer --backend egl ou --backend osmesa.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
