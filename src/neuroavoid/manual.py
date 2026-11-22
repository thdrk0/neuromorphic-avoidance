"""Manual driving and deterministic headless verification."""

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--command", choices=["STRAIGHT", "LEFT", "RIGHT", "STOP"], default="STOP")
    parser.add_argument("--view", choices=["overview", "onboard"], default="overview")
    parser.add_argument("--backend", choices=["glfw", "egl", "osmesa"], default=os.environ.get("MUJOCO_GL", "glfw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/manual"))
    parser.add_argument("--debug-keys", action="store_true", help="Afficher chaque code de touche recu")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps doit etre positif")
    if not args.headless and args.backend != "glfw":
        parser.error("La fenetre interactive utilise --backend glfw")
    os.environ["MUJOCO_GL"] = args.backend
    # On this WSLg/D3D12 setup, the window framebuffer is black with the GPU
    # driver, although offscreen rendering works. Select Mesa software rendering
    # before importing OpenGL; an explicit environment value remains respected.
    if not args.headless and os.environ.get("WSL_DISTRO_NAME"):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    import mujoco
    from PIL import Image
    from neuroavoid.config import CAMERA_HEIGHT, CAMERA_WIDTH, PHYSICS_TIMESTEP
    from neuroavoid.robot import RobotSimulation
    from neuroavoid.keyboard import command_for_key

    sim = RobotSimulation()
    sim.set_command(args.command)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = 0
    viewer_report = None
    pending = []

    def key_callback(key):
        pending.append(key)

    def capture(renderer):
        renderer.update_scene(sim.data, camera="onboard")
        return renderer.render().copy()

    print("Demarrage MuJoCo : initialisation du rendu...", flush=True)
    if not args.headless and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1":
        print("Rendu logiciel Mesa actif (compatibilite WSLg).", flush=True)
    with mujoco.Renderer(sim.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH) as renderer:
        if args.headless:
            frame = capture(renderer)
            for _ in range(args.steps):
                if sim.step():
                    frame = capture(renderer)
                    frames += 1
        else:
            from neuroavoid.viewer import RobotViewer
            print("Z/W : avancer | Q/A : gauche | D : droite | S/espace : arret", flush=True)
            print("R : recommencer | C : photo camera | X : quitter")
            print("Commandes persistantes : appuyer sur S pour arreter.")
            with RobotViewer(sim.model, sim.data, key_callback) as viewer:
                if args.view == "onboard":
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = sim.model.camera("onboard").id
                else:
                    viewer.cam.lookat[:] = [0, 0, 0]
                    viewer.cam.distance = 7
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -55
                viewer.sync()
                print("Fenetre ouverte : MuJoCo - Pilotage robot", flush=True)
                frame = capture(renderer)
                running = True
                last_contacts = set()
                while viewer.is_running() and running:
                    start = time.monotonic()
                    keys, pending[:] = pending[:], []
                    for key in keys:
                        if args.debug_keys:
                            print(f"Touche recue : {key} ({chr(key) if 32 <= key < 127 else 'speciale'})", flush=True)
                        command = command_for_key(key)
                        if command:
                            sim.set_command(command)
                            print(f"Commande : {command} | moteurs : {sim.data.ctrl.round(2).tolist()}", flush=True)
                        elif key == 82:
                            sim.reset()
                            frames = 0
                            last_contacts = set()
                            frame = capture(renderer)
                            print("Simulation reinitialisee.")
                        elif key == 67:
                            Image.fromarray(frame).save(args.output / "camera_snapshot.png")
                            Image.fromarray(viewer.screenshot()).save(args.output / "viewer_snapshot.png")
                            print(f"Photo : {args.output / 'camera_snapshot.png'}")
                        elif key == 88:
                            running = False
                    if not running:
                        break
                    if sim.step():
                        frame = capture(renderer)
                        frames += 1
                        viewer.sync()
                    if sim.contacts and sim.contacts != last_contacts:
                        print(f"COLLISION : {', '.join(sorted(sim.contacts))}")
                    if args.debug_keys and sim.steps % 100 == 0:
                        print("Etat : " + json.dumps({"time": float(sim.data.time), "pose": sim.pose}), flush=True)
                    last_contacts = sim.contacts.copy()
                    time.sleep(max(0, PHYSICS_TIMESTEP - (time.monotonic() - start)))
                viewer_report = viewer.diagnostics()
        Image.fromarray(frame).save(args.output / "camera_latest.png")
    report = {"simulation_seconds": float(sim.data.time), "pose_xy_yaw": sim.pose,
              "command": sim.command, "collided": sim.collided,
              "collision_steps": sim.collision_steps, "camera_frames": frames,
              "viewer": viewer_report}
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
