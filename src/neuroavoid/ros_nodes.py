"""ROS2 policy and simulator nodes; reliable, correlated window/decision exchange."""

import argparse
from array import array
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from neuroavoid.ros_protocol import DecisionGate, VELOCITIES, stamp_parts, validate_velocity


def policy(args):
    import rclpy
    from geometry_msgs.msg import Twist
    from neuroavoid_interfaces.msg import ControlDecision, EventWindow
    from neuroavoid.snn_bridge import SNNBridge
    rclpy.init()
    node = rclpy.create_node("snn_policy", namespace="neuroavoid")
    try:
        with SNNBridge(args.checkpoint) as model:
            if model.metadata["config"].get("model_type", "snn") != "snn":
                raise ValueError("Le noeud ROS2 utilise le SNN ; lancer le CNN avec run_snn_robot ou le benchmark direct")
            publisher = node.create_publisher(ControlDecision, "control/decision", 1)
            velocities = node.create_publisher(Twist, "cmd_vel", 1)

            def receive(message):
                counts = np.asarray(message.counts, dtype=np.uint16).reshape(5, 2, 128, 128)
                result = model.predict(counts)
                reply = ControlDecision()
                reply.run_id, reply.window_id = message.run_id, message.window_id
                reply.observation_stamp = message.observation_stamp
                reply.label = result["command"]
                reply.velocity.linear.x, reply.velocity.angular.z = VELOCITIES[reply.label]
                reply.input_spikes, reply.hidden_spikes = result["input_spikes"], result["hidden_spikes"]
                reply.model_seconds = result["model_seconds"]
                publisher.publish(reply)
                velocities.publish(reply.velocity)

            node.create_subscription(EventWindow, "events/window", receive, 1)
            node.get_logger().info(f"SNN pret, checkpoint {model.metadata['checkpoint_sha256'][:12]}")
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def simulate(args):
    os.environ["MUJOCO_GL"] = "glfw"
    if os.environ.get("WSL_DISTRO_NAME"):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    import mujoco
    import rclpy
    from geometry_msgs.msg import Twist
    from neuroavoid_interfaces.msg import ControlDecision, EventWindow
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from neuroavoid.dataset import count_frame
    from neuroavoid.events import V2EBridge
    from neuroavoid.navigation import NavigationTask
    from neuroavoid.robot import RobotSimulation
    from neuroavoid.scenes import make_scene

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choisir un dossier de sortie vide")
    args.output.mkdir(parents=True, exist_ok=True)
    xml, metadata = make_scene(args.scene)
    sim = RobotSimulation(scene_xml=xml)
    for _ in range(100):
        sim.step()
    origin = float(sim.data.time)
    task = NavigationTask(timeout=origin + args.seconds)
    gate = DecisionGate(uuid.uuid4().hex)
    result = None
    rejected = 0
    quitting = False
    node = None
    decisions, trajectory, latencies = [], [], []
    status = "running"
    error_detail = None
    event_count = hidden_spikes = 0
    started = time.perf_counter()
    rclpy.init()
    node = rclpy.create_node("mujoco_robot", namespace="neuroavoid")
    windows = node.create_publisher(EventWindow, "events/window", 1)
    clocks = node.create_publisher(Clock, "/clock", 1)
    states = node.create_publisher(String, "status", 1)
    stop_publisher = node.create_publisher(Twist, "cmd_vel", 1)

    def receive(message):
        nonlocal result, rejected
        try:
            validate_velocity(message.label, message.velocity.linear.x, message.velocity.angular.z)
            if not math.isfinite(message.model_seconds) or message.model_seconds < 0:
                raise ValueError("Temps modele invalide")
            stamp = (message.observation_stamp.sec, message.observation_stamp.nanosec)
            if not gate.accept(message.run_id, message.window_id, stamp):
                rejected += 1
                return
            result = message
        except ValueError:
            rejected += 1

    def on_key(key):
        nonlocal quitting
        if key == 88:
            quitting = True

    subscriber = node.create_subscription(ControlDecision, "control/decision", receive, 1)
    try:
        deadline = time.monotonic() + args.discovery_timeout
        while windows.get_subscription_count() < 1 or node.count_publishers("/neuroavoid/control/decision") < 1:
            if time.monotonic() >= deadline:
                status = "policy_unavailable"
                break
            rclpy.spin_once(node, timeout_sec=0.05)
        if status == "running":
            node.get_logger().info("Politique connectee ; simulation synchronisee aux reponses ROS2")
            with ExitStack() as stack:
                bridge = stack.enter_context(V2EBridge())
                renderer = stack.enter_context(mujoco.Renderer(sim.model, height=128, width=128))
                viewer = None
                if args.viewer:
                    from neuroavoid.viewer import RobotViewer
                    viewer = stack.enter_context(RobotViewer(sim.model, sim.data, on_key))
                    viewer.cam.lookat[:] = [0, 0, 0]
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 7, 135, -55
                    viewer.help_text = "Pilotage SNN via ROS2 | X ou fermer : quitter"
                    viewer.sync()
                frame = 0
                pending = []
                while status == "running":
                    tick = time.perf_counter()
                    rclpy.spin_once(node, timeout_sec=0)
                    if quitting or (viewer and not viewer.is_running()):
                        status = "aborted"
                        break
                    if frame:
                        for _ in range(4):
                            sim.step()
                            status = task.outcome(sim)
                            if status != "running":
                                break
                        if status != "running":
                            break
                    timestamp = frame / 50
                    clock = Clock()
                    clock.clock.sec, clock.clock.nanosec = stamp_parts(timestamp)
                    clocks.publish(clock)
                    renderer.update_scene(sim.data, camera="onboard")
                    events, _, _ = bridge.convert(renderer.render().copy(), timestamp)
                    event_count += len(events)
                    if frame:
                        pending.append(count_frame(events))
                    if len(pending) == 5:
                        message = EventWindow()
                        message.run_id, message.window_id = gate.run_id, frame // 5
                        message.observation_stamp.sec, message.observation_stamp.nanosec = stamp_parts(timestamp)
                        payload = array("H")
                        payload.frombytes(np.stack(pending).astype(np.uint16).tobytes())
                        message.counts = payload
                        pending.clear()
                        result = None
                        gate.begin(message.window_id, stamp_parts(timestamp))
                        sent = time.perf_counter()
                        windows.publish(message)
                        deadline = time.monotonic() + args.command_timeout
                        while result is None:
                            rclpy.spin_once(node, timeout_sec=0.02)
                            if time.monotonic() >= deadline:
                                sim.set_command("STOP")
                                status = "command_timeout"
                                break
                            if viewer and (quitting or not viewer.is_running()):
                                status = "aborted"
                                break
                        if status != "running":
                            break
                        elapsed = time.perf_counter() - sent
                        latencies.append(elapsed)
                        sim.set_command(result.label)
                        hidden_spikes += result.hidden_spikes
                        decisions.append({"time": timestamp, "window_id": result.window_id, "command": result.label,
                                          "linear_x": result.velocity.linear.x, "angular_z": result.velocity.angular.z,
                                          "hidden_spikes": result.hidden_spikes, "input_spikes": result.input_spikes,
                                          "model_seconds": result.model_seconds, "ros_roundtrip_seconds": elapsed})
                        if viewer:
                            viewer.help_text = f"SNN via ROS2 : {result.label} | t={timestamp:.1f}s\nX ou fermer : quitter"
                    trajectory.append({"time": timestamp, "pose": sim.pose, "command": sim.command})
                    if viewer:
                        viewer.sync()
                        time.sleep(max(0, 0.02 - (time.perf_counter() - tick)))
                    if frame and frame % 250 == 0:
                        node.get_logger().info(f"{timestamp:.1f}s simulees : {sim.command}")
                    frame += 1
                if viewer:
                    from PIL import Image
                    viewer.sync()
                    Image.fromarray(viewer.screenshot()).save(args.output / "final_view.png")
    except KeyboardInterrupt:
        status = "aborted"
    except Exception as exc:
        status = "error"
        error_detail = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sim.set_command("STOP")
        if rclpy.ok():
            stop_publisher.publish(Twist())
            states.publish(String(data=json.dumps({"status": status, "run_id": gate.run_id})))
        report = {**metadata, "status": status, "run_id": gate.run_id,
                  "simulation_seconds": float(sim.data.time) - origin, "wall_seconds": time.perf_counter() - started,
                  "collided": sim.collided, "contacts": sorted(sim.contacts), "final_motor_controls": sim.data.ctrl.tolist(),
                  "decisions": decisions, "trajectory": trajectory, "rejected_decisions": rejected,
                  "dvs_events": event_count, "hidden_spikes": hidden_spikes, "teacher_used": False, "test_used": False,
                  "transport": "ROS2 Humble reliable DDS, depth 1, lockstep",
                  "error": error_detail,
                  "ros_roundtrip_mean_ms": float(np.mean(latencies) * 1000) if latencies else None,
                  "ros_roundtrip_p95_ms": float(np.percentile(latencies, 95) * 1000) if latencies else None,
                  "note": "Roundtrip = transport + encodage + inference + ordonnancement, pas latence du modele seul."}
        (args.output / "episode.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("status", "simulation_seconds", "collided", "ros_roundtrip_mean_ms")}), flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if status in ("success", "collision", "timeout", "aborted") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("policy")
    p.add_argument("--checkpoint", type=Path, default=Path("models/snn.pt"))
    s = sub.add_parser("sim")
    s.add_argument("--scene", choices=["train_11", "train_23", "train_37", "validation_41"], default="validation_41")
    s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--viewer", action="store_true")
    s.add_argument("--output", type=Path, default=Path("artifacts/ros2"))
    s.add_argument("--discovery-timeout", type=float, default=15)
    s.add_argument("--command-timeout", type=float, default=15)
    args = parser.parse_args()
    if args.mode == "sim" and any(not math.isfinite(v) or v <= 0 for v in (args.seconds, args.discovery_timeout, args.command_timeout)):
        parser.error("Les durees doivent etre positives et finies")
    if args.mode == "policy":
        policy(args)
        return 0
    return simulate(args)


if __name__ == "__main__":
    raise SystemExit(main())
