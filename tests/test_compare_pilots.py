import json
from pathlib import Path
import tempfile
import unittest

from neuroavoid.compare_pilots import compare


class PilotComparisonChecks(unittest.TestCase):
    def test_matched_results_and_rejection_of_assistance_or_geometry_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "old/scene", root / "new/scene"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            old = {"scene_id": "scene", "split": "validation", "teacher_used": False, "test_used": False,
                   "geometry_sha256": "map", "checkpoint_sha256": "weights", "camera_fps_simulated": 50,
                   "decision_hz_simulated": 10, "window_ms": 100, "status": "collision", "simulation_seconds": 10}
            new = {**old, "status": "success"}
            (a / "episode.json").write_text(json.dumps(old))
            (b / "episode.json").write_text(json.dumps(new))
            result = compare(a.parent, b.parent)
            self.assertEqual(result["before_successes"], 0)
            self.assertEqual(result["after_successes"], 1)
            for change in ({"teacher_used": True}, {"geometry_sha256": "other"}, {"split": "test"}):
                (b / "episode.json").write_text(json.dumps({**new, **change}))
                with self.assertRaises(ValueError):
                    compare(a.parent, b.parent)
