"""MuJoCo camera -> pinned v2e core -> timestamped DVS events."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from neuroavoid.events import V2EBridge, event_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--fps", type=int, choices=[50, 100, 200], default=50)
    parser.add_argument("--motion", choices=["static", "straight", "left", "right", "reference"], default="straight")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--backend", choices=["glfw", "egl", "osmesa"], default=os.environ.get("MUJOCO_GL", "glfw"))
    parser.add_argument("--show", action="store_true", help="Afficher les images et les evenements cote a cote")
    parser.add_argument("--output", type=Path, default=Path("artifacts/events"))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.seconds > 60:
        parser.error("--seconds doit etre dans ]0, 60]")
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold doit etre positif et fini")
    if args.seconds * args.fps < 1:
        parser.error("La duree doit couvrir au moins une periode camera")
    os.environ["MUJOCO_GL"] = args.backend
    import mujoco
    from neuroavoid.config import CAMERA_HEIGHT, CAMERA_WIDTH, PHYSICS_TIMESTEP
    from neuroavoid.navigation import NavigationTask, ReferenceController
    from neuroavoid.robot import RobotSimulation

    args.output.mkdir(parents=True, exist_ok=True)
    sim = RobotSimulation()
    # Settle the chassis before starting the event clock.
    for _ in range(100):
        sim.step()
    origin = float(sim.data.time)
    stride = round(1 / (args.fps * PHYSICS_TIMESTEP))
    samples = math.floor(args.seconds * args.fps + 1e-9)
    controller = ReferenceController(sim, NavigationTask()) if args.motion == "reference" else None
    window = label = None
    closed = False
    if args.show:
        import tkinter as tk
        from PIL import ImageTk
        window = tk.Tk()
        window.title("Camera RGB (gauche) | DVS ON rouge / OFF cyan (droite)")
        label = tk.Label(window)
        label.pack()

        def on_close():
            nonlocal closed
            closed = True

        window.protocol("WM_DELETE_WINDOW", on_close)

    batches, rgb_frames, times, offsets = [], [], [], [0]
    emulator_times, bridge_times, render_times = [], [], []
    first_call_seconds = 0
    started = time.perf_counter()
    print(f"Camera {args.fps} Hz -> v2e CPU ; mouvement : {args.motion}", flush=True)
    try:
        with V2EBridge(CAMERA_WIDTH, CAMERA_HEIGHT, args.threshold) as bridge:
            with mujoco.Renderer(sim.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH) as renderer:
                frozen = None
                for index in range(samples + 1):
                    tick = time.perf_counter()
                    if index:
                        for _ in range(stride):
                            if args.motion != "static":
                                command = controller.decide(sim) if controller else args.motion.upper()
                                sim.set_command(command)
                                sim.step()
                    timestamp = index / args.fps
                    rendering_started = time.perf_counter()
                    if args.motion == "static" and frozen is not None:
                        rgb = frozen.copy()
                    else:
                        renderer.update_scene(sim.data, camera="onboard")
                        rgb = renderer.render().copy()
                        frozen = rgb.copy()
                    render_seconds = time.perf_counter() - rendering_started
                    events, emulation_seconds, bridge_seconds = bridge.convert(rgb, timestamp)
                    if index == 0:
                        first_call_seconds = bridge_seconds
                    else:
                        emulator_times.append(emulation_seconds)
                        bridge_times.append(bridge_seconds)
                        render_times.append(render_seconds)
                    batches.append(events)
                    rgb_frames.append(rgb)
                    times.append(timestamp)
                    offsets.append(offsets[-1] + len(events))
                    preview = Image.fromarray(np.concatenate([rgb, event_image(events)], axis=1))
                    if window:
                        photo = ImageTk.PhotoImage(preview.resize((768, 384), Image.Resampling.NEAREST))
                        label.configure(image=photo)
                        label.image = photo
                        window.update_idletasks()
                        window.update()
                        if closed:
                            break
                        time.sleep(max(0, 1 / args.fps - (time.perf_counter() - tick)))
                    if index and index % args.fps == 0:
                        print(f"{timestamp:.2f} s simulees : {offsets[-1]} evenements", flush=True)
                    if args.motion != "static" and sim.collided:
                        break
    finally:
        if window:
            window.destroy()
    if not batches:
        raise RuntimeError("Aucune image convertie")
    events = np.concatenate(batches, axis=0)
    np.savez_compressed(args.output / "events.npz", events=events,
                        frame_times=np.asarray(times), event_offsets=np.asarray(offsets, dtype=np.int64))
    np.savez_compressed(args.output / "frames.npz", rgb=np.stack(rgb_frames), frame_times=np.asarray(times))
    # Accumulate all events only for the still image; animation keeps individual windows.
    Image.fromarray(np.concatenate([rgb_frames[-1], event_image(events)], axis=1)).resize(
        (768, 384), Image.Resampling.NEAREST).save(args.output / "preview.png")
    # GIF has 10 ms time units: accumulate high-rate input into 20 ms display bins.
    gif_stride = max(1, args.fps // 50)
    gif_frames, pending_events = [], []
    for index, batch in enumerate(batches):
        pending_events.append(batch)
        if index % gif_stride == 0 or index == len(batches) - 1:
            grouped = np.concatenate(pending_events, axis=0)
            gif_frames.append(Image.fromarray(np.concatenate([rgb_frames[index], event_image(grouped)], axis=1)))
            pending_events = []
    gif_frames[0].save(args.output / "preview.gif", save_all=True, append_images=gif_frames[1:],
                       duration=20, loop=0)

    def stats(values):
        return {"mean_ms": float(np.mean(values) * 1000) if values else None,
                "p95_ms": float(np.percentile(values, 95) * 1000) if values else None}

    total = time.perf_counter() - started
    report = {"engine": "SensorsINI/v2e EventEmulator", "commit": "cac638444557a224ba6ee5c29260916f152c00bd",
              "motion": args.motion, "camera_fps": args.fps, "resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
              "threshold": args.threshold, "noise_enabled": False, "device": "cpu",
              "frame_count": len(times), "simulated_seconds": times[-1], "wall_seconds": total,
              "event_count": len(events), "on_count": int(np.sum(events[:, 3] == 1)),
              "off_count": int(np.sum(events[:, 3] == -1)), "collided": sim.collided,
              "aborted": closed, "emulator": stats(emulator_times), "bridge_including_emulator": stats(bridge_times),
              "camera_render": stats(render_times), "first_bridge_call_seconds": first_call_seconds,
              "event_columns": ["t_seconds", "x", "y", "polarity_plus_or_minus_1"],
              "warmup_simulation_seconds": origin,
              "note": "Pas de Super SloMo. Temps evenements estimes par v2e entre images, pas mesures par une vraie DVS."
              }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Resultats : {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
