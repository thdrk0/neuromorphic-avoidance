import unittest

from neuroavoid.navigation import NavigationTask, ReferenceController
from neuroavoid.robot import RobotSimulation
from neuroavoid.run_navigation import run_episode


class NavigationChecks(unittest.TestCase):
    def test_reference_reaches_goal_from_varied_starts(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                report = run_episode(NavigationTask(), seed=seed)
                self.assertEqual(report["status"], "success")
                self.assertFalse(report["collided"])
                self.assertLessEqual(report["goal_distance_m"], 0.28)
                self.assertTrue(any(r["command"] in ("LEFT", "RIGHT") for r in report["trajectory"]))

    def test_straight_hits_obstacle(self):
        report = run_episode(NavigationTask(), policy="straight")
        self.assertEqual(report["status"], "collision")
        self.assertIn("obstacle_front", report["contacts"])

    def test_stopped_robot_times_out(self):
        report = run_episode(NavigationTask(timeout=0.5), policy="stop")
        self.assertEqual(report["status"], "timeout")
        self.assertFalse(report["collided"])
        self.assertAlmostEqual(report["simulation_seconds"], 0.5)

    def test_collision_has_priority_over_goal(self):
        sim = RobotSimulation()
        task = NavigationTask(goal=tuple(sim.pose[:2]))
        self.assertEqual(task.outcome(sim), "success")
        sim.collided = True
        self.assertEqual(task.outcome(sim), "collision")

    def test_planner_rejects_goal_inside_obstacle(self):
        with self.assertRaises(ValueError):
            ReferenceController(RobotSimulation(), NavigationTask(goal=(0, 0)))
