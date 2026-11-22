"""Physical checks independent of graphics and wall-clock speed."""

import unittest

from neuroavoid.robot import RobotSimulation


class RobotChecks(unittest.TestCase):
    def test_fast_mode_does_not_saturate_actuators(self):
        import numpy as np
        sim = RobotSimulation()
        for command in ("LEFT", "STRAIGHT", "RIGHT"):
            sim.set_command(command, speed_factor=1.75)
            self.assertTrue(np.all(np.abs(sim.data.ctrl) < 8))
        with self.assertRaises(ValueError):
            sim.set_command("STRAIGHT", speed_factor=2)

    def run_command(self, command, steps):
        sim = RobotSimulation()
        sim.set_command(command)
        frames = sum(sim.step() for _ in range(steps))
        self.assertEqual(frames, steps // sim.camera_stride)
        return sim

    def test_stop_and_floor_are_not_collision(self):
        sim = self.run_command("STOP", 400)
        self.assertLess(abs(sim.pose[0] + 2), 0.03)
        self.assertFalse(sim.collided)

    def test_straight_moves_forward(self):
        sim = self.run_command("STRAIGHT", 400)
        self.assertGreater(sim.pose[0], -1.5)
        self.assertLess(abs(sim.pose[1]), 0.05)
        self.assertFalse(sim.collided)

    def test_turn_directions(self):
        left = self.run_command("LEFT", 200)
        right = self.run_command("RIGHT", 200)
        self.assertGreater(left.pose[2], 0.5)
        self.assertLess(right.pose[2], -0.5)

    def test_collision_and_reset(self):
        sim = self.run_command("STRAIGHT", 1600)
        self.assertTrue(sim.collided)
        self.assertGreater(sim.collision_steps, 0)
        self.assertIn("obstacle_front", sim.contacts)
        sim.reset()
        self.assertFalse(sim.collided)
        self.assertEqual(sim.collision_steps, 0)
        self.assertEqual(sim.command, "STOP")
        self.assertEqual(list(sim.data.ctrl), [0, 0])


if __name__ == "__main__":
    unittest.main()
