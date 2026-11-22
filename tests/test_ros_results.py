import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "artifacts"


@unittest.skipUnless((ROOT / "ros2_validation/episode.json").exists(), "Executer le parcours ROS2 complet")
class ROSResultChecks(unittest.TestCase):
    def test_ros_and_direct_have_identical_actions_and_trajectory(self):
        ros = json.loads((ROOT / "ros2_validation/episode.json").read_text())
        direct = json.loads((ROOT / "snn_corrections_robot/validation_41/episode.json").read_text())
        self.assertEqual(ros["status"], "success")
        self.assertFalse(ros["teacher_used"])
        self.assertEqual(ros["final_motor_controls"], [0, 0])
        self.assertEqual([d["command"] for d in ros["decisions"]], [d["command"] for d in direct["decisions"]])
        self.assertEqual([d["time"] for d in ros["decisions"]], [d["time"] for d in direct["decisions"]])
        np.testing.assert_allclose([t["pose"] for t in ros["trajectory"]],
                                   [t["pose"] for t in direct["trajectory"]], atol=1e-9)
        self.assertAlmostEqual(ros["simulation_seconds"], direct["simulation_seconds"])

    @unittest.skipUnless((ROOT / "ros2_missing_policy/episode.json").exists(), "Executer le test sans politique")
    def test_missing_policy_keeps_robot_stopped(self):
        report = json.loads((ROOT / "ros2_missing_policy/episode.json").read_text())
        self.assertEqual(report["status"], "policy_unavailable")
        self.assertEqual(report["final_motor_controls"], [0, 0])
        self.assertEqual(report["simulation_seconds"], 0)
