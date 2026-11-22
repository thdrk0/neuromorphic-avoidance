"""Live DDS test: stop the policy after a decision and verify motor shutdown."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import rclpy
from neuroavoid_interfaces.msg import ControlDecision


def main():
    rclpy.init()
    node = rclpy.create_node("fault_test_observer")
    seen = []
    node.create_subscription(ControlDecision, "/neuroavoid/control/decision", lambda msg: seen.append(msg.window_id), 1)
    policy = sim = None
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "processes.log").open("w") as logs:
                policy = subprocess.Popen([sys.executable, "-u", "-m", "neuroavoid.ros_nodes", "policy"], stdout=logs, stderr=subprocess.STDOUT)
                sim = subprocess.Popen([sys.executable, "-u", "-m", "neuroavoid.ros_nodes", "sim", "--seconds", "1",
                                        "--command-timeout", "1", "--output", str(root / "result")], stdout=logs, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 30
                while not seen and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                assert seen, (root / "processes.log").read_text()
                policy.send_signal(signal.SIGINT)
                policy.wait(timeout=10)
                sim.wait(timeout=15)
                report = json.loads((root / "result/episode.json").read_text())
                assert report["status"] == "command_timeout", report
                assert report["final_motor_controls"] == [0.0, 0.0], report
                print("OK : perte du noeud SNN -> command_timeout et moteurs a zero")
    finally:
        for process in (sim, policy):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
