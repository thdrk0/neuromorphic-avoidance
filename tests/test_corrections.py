import unittest
import json
from pathlib import Path

import numpy as np

from neuroavoid.record_dataset import choose_collection_command

ROOT = Path(__file__).resolve().parents[1]


class CorrectionChecks(unittest.TestCase):
    def test_pure_student_does_not_apply_teacher(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            self.assertEqual(choose_collection_command("LEFT", "RIGHT", 0, rng), ("LEFT", False))

    def test_pure_teacher_applies_correction(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            self.assertEqual(choose_collection_command("LEFT", "RIGHT", 1, rng), ("RIGHT", True))

    def test_mixture_is_reproducible_and_rejects_invalid_probability(self):
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        seq_a = [choose_collection_command("LEFT", "RIGHT", 0.5, a) for _ in range(20)]
        seq_b = [choose_collection_command("LEFT", "RIGHT", 0.5, b) for _ in range(20)]
        self.assertEqual(seq_a, seq_b)
        self.assertEqual({used_teacher for _, used_teacher in seq_a}, {True, False})
        with self.assertRaises(ValueError):
            choose_collection_command("LEFT", "RIGHT", -0.1, a)

    @unittest.skipUnless((ROOT / "artifacts/dataset_corrections/manifest.json").exists(), "Collecte requise")
    def test_recorded_controls_are_separate_from_expert_targets(self):
        from neuroavoid.dataset import LABELS
        manifest = json.loads((ROOT / "artifacts/dataset_corrections/manifest.json").read_text())
        if not manifest["complete"]:
            self.skipTest("Collecte en cours")
        corrected = [e for e in manifest["episodes"] if e.get("collection") == "mixed student/expert"]
        self.assertEqual(len(corrected), 6)
        for episode in corrected:
            self.assertEqual(episode["split"], "train")
            with np.load(episode["raw_file"], allow_pickle=False) as raw:
                labels, applied = raw["labels"], raw["applied_commands"]
                self.assertTrue(np.all(applied[:5] == "STOP"))
                for prediction in episode["student_predictions"]:
                    index = round(prediction["time"] * 50)
                    self.assertEqual(LABELS[int(labels[index])], prediction["teacher"])
                    expected = prediction["teacher"] if prediction["teacher_applied"] else prediction["student"]
                    self.assertEqual(applied[index], expected)
                    if episode["teacher_probability"] == 0:
                        self.assertFalse(prediction["teacher_applied"])
