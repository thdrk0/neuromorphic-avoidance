import unittest

from neuroavoid.ros_protocol import DecisionGate, VELOCITIES, stamp_parts, validate_velocity


class ROSProtocolChecks(unittest.TestCase):
    def test_window_correlation_and_duplicate_rejection(self):
        gate = DecisionGate("run-a")
        gate.begin(1, stamp_parts(0.1))
        self.assertFalse(gate.accept("other-run", 1, stamp_parts(0.1)))
        self.assertFalse(gate.accept("run-a", 2, stamp_parts(0.1)))
        self.assertFalse(gate.accept("run-a", 1, stamp_parts(0.2)))
        self.assertTrue(gate.accept("run-a", 1, stamp_parts(0.1)))
        self.assertFalse(gate.accept("run-a", 1, stamp_parts(0.1)))
        with self.assertRaises(ValueError):
            gate.begin(1, stamp_parts(0.1))
        gate.begin(2, stamp_parts(0.2))
        with self.assertRaises(ValueError):
            gate.begin(3, stamp_parts(0.3))

    def test_velocity_matches_the_simulator(self):
        from neuroavoid.robot import COMMANDS
        for label, velocity in VELOCITIES.items():
            self.assertEqual(velocity, COMMANDS[label])
            validate_velocity(label, *velocity)
        with self.assertRaises(ValueError):
            validate_velocity("LEFT", float("nan"), 1.5)
        with self.assertRaises(ValueError):
            validate_velocity("LEFT", 5, 0)
        with self.assertRaises(ValueError):
            validate_velocity("UNKNOWN", 0, 0)

    def test_clock_has_exact_nanosecond_carry(self):
        self.assertEqual(stamp_parts(0.1), (0, 100_000_000))
        self.assertEqual(stamp_parts(1.9999999996), (2, 0))
