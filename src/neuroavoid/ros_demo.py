"""Start the two ROS processes and always clean up the policy on exit."""

import argparse
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/snn.pt"))
    args, sim_args = parser.parse_known_args()
    policy = subprocess.Popen([sys.executable, "-u", "-m", "neuroavoid.ros_nodes", "policy", "--checkpoint", str(args.checkpoint)])
    sim = None
    try:
        sim = subprocess.Popen([sys.executable, "-u", "-m", "neuroavoid.ros_nodes", "sim", *sim_args])
        return sim.wait()
    except KeyboardInterrupt:
        if sim and sim.poll() is None:
            sim.send_signal(signal.SIGINT)
            try:
                sim.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sim.kill()
                sim.wait()
        return 130
    finally:
        if policy.poll() is None:
            policy.send_signal(signal.SIGINT)
            try:
                policy.wait(timeout=10)
            except subprocess.TimeoutExpired:
                policy.kill()
                policy.wait()


if __name__ == "__main__":
    raise SystemExit(main())
