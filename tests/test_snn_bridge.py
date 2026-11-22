from pathlib import Path
import unittest

import numpy as np

from neuroavoid.snn_bridge import SNNBridge

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "artifacts/snn/best.pt").exists() and (ROOT / ".venv-v2e/bin/python").exists(), "Checkpoint SNN requis")
class SNNBridgeChecks(unittest.TestCase):
    def test_saved_prediction_and_independent_window_state(self):
        with np.load(ROOT / "artifacts/dataset/validation_41/samples.npz", allow_pickle=False) as data:
            counts = data["voxels"][0]
        with SNNBridge(ROOT / "artifacts/snn/best.pt") as model:
            a = model.predict(counts)
            model.predict(np.zeros_like(counts))
            b = model.predict(counts)
            self.assertEqual(a["command"], "RIGHT")
            self.assertEqual(a["command"], b["command"])
            self.assertEqual(a["scores"], b["scores"])
            self.assertEqual(a["hidden_spikes"], b["hidden_spikes"])
            self.assertGreater(a["hidden_spikes"], 0)
            self.assertGreater(a["model_seconds"], 0)
            with self.assertRaises(ValueError):
                model.predict(counts.astype(float))
        self.assertIsNotNone(model.proc.returncode)
