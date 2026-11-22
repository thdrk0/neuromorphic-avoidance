from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FinalBenchmarkChecks(unittest.TestCase):
    @unittest.skipUnless((ROOT / "artifacts/benchmark_offline/classification.json").exists(), "Executer le benchmark final")
    def test_offline_uses_same_test_inputs_and_no_fake_cnn_spikes(self):
        report = json.loads((ROOT / "artifacts/benchmark_offline/classification.json").read_text())
        self.assertEqual(report["split"], "test")
        self.assertEqual(report["samples"], 288)
        self.assertEqual(set(report["models"]), {"snn", "cnn"})
        self.assertEqual(report["models"]["snn"]["samples"], report["models"]["cnn"]["samples"])
        self.assertIsNone(report["models"]["cnn"]["hidden_spikes_mean"])
        self.assertGreater(report["models"]["snn"]["hidden_spikes_mean"], 0)

    @unittest.skipUnless((ROOT / "artifacts/benchmark_navigation/summary.json").exists(), "Terminer les huit parcours")
    def test_protocol_covers_all_frozen_cases_without_teacher(self):
        report = json.loads((ROOT / "artifacts/benchmark_navigation/summary.json").read_text())
        expected = {(model, scene, speed) for model in ("snn", "cnn") for scene in ("test_53", "test_67") for speed in (1.0, 1.75)}
        actual = {(r["model"], r["scene"], r["speed_factor"]) for r in report["episodes"]}
        self.assertEqual(expected, actual)
        self.assertFalse(report["protocol"]["teacher_used"])
        self.assertEqual(len(report["episodes"]), 8)
