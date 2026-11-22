import unittest
from pathlib import Path

import numpy as np

from neuroavoid.events import V2EBridge, event_image


class EventImages(unittest.TestCase):
    def test_polarity_and_xy_convention(self):
        pixels = event_image(np.array([[0, 2, 3, 1], [0, 4, 5, -1]]), 8, 8)
        self.assertEqual(pixels[3, 2].tolist(), [255, 0, 0])
        self.assertEqual(pixels[5, 4].tolist(), [0, 255, 255])
        self.assertEqual(int(pixels[2, 3].sum()), 0)


@unittest.skipUnless((Path(__file__).resolve().parents[1] / ".venv-v2e/bin/python").exists(), "Installer v2e avec scripts/setup_v2e.sh")
class V2EChecks(unittest.TestCase):
    def test_static_on_off_and_timestamps(self):
        with V2EBridge(8, 8) as bridge:
            rgb = np.full((8, 8, 3), 100, np.uint8)
            first, _, _ = bridge.convert(rgb, 0)
            static, _, _ = bridge.convert(rgb, 0.02)
            on, _, _ = bridge.convert(rgb * 2, 0.04)
            off, _, _ = bridge.convert(rgb, 0.06)
            self.assertEqual(len(first) + len(static), 0)
            self.assertGreater(len(on), 0)
            self.assertGreater(len(off), 0)
            self.assertTrue(np.all(on[:, 3] == 1))
            self.assertTrue(np.all(off[:, 3] == -1))
            self.assertTrue(np.all(np.diff(np.concatenate([on[:, 0], off[:, 0]])) >= 0))
            with self.assertRaises(ValueError):
                bridge.convert(rgb, 0.06)
            with self.assertRaises(ValueError):
                bridge.convert(rgb.astype(float), 0.08)
