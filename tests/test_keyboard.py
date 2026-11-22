import unittest

from neuroavoid.keyboard import command_for_key
from neuroavoid.robot import RobotSimulation


class KeyboardChecks(unittest.TestCase):
    def test_layout_aliases_move_robot(self):
        for key, expected in [("W", "STRAIGHT"), ("Z", "STRAIGHT"),
                              ("A", "LEFT"), ("Q", "LEFT"), ("D", "RIGHT")]:
            with self.subTest(key=key):
                sim = RobotSimulation()
                command = command_for_key(ord(key))
                self.assertEqual(command, expected)
                sim.set_command(command)
                for _ in range(200):
                    sim.step()
                if expected == "STRAIGHT":
                    self.assertGreater(sim.pose[0], -1.8)
                elif expected == "LEFT":
                    self.assertGreater(sim.pose[2], 0.5)
                else:
                    self.assertLess(sim.pose[2], -0.5)

    def test_stop_and_unmapped_keys(self):
        self.assertEqual(command_for_key(ord("S")), "STOP")
        self.assertEqual(command_for_key(32), "STOP")
        self.assertIsNone(command_for_key(ord("X")))
        self.assertIsNone(command_for_key(999))
