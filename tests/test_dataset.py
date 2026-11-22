import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from neuroavoid.dataset import EventDataset, count_frame, make_windows
from neuroavoid.scenes import SCENES, make_scene, scene_manifest


class DatasetChecks(unittest.TestCase):
    def test_geometry_splits_are_distinct_and_reproducible(self):
        scenes = scene_manifest()
        self.assertEqual(len({s["geometry_sha256"] for s in scenes}), len(scenes))
        self.assertEqual({s["split"] for s in scenes}, {"train", "validation", "test"})
        for scene_id, _, _ in SCENES:
            self.assertEqual(make_scene(scene_id), make_scene(scene_id))

    def test_counts_preserve_repeated_events_and_polarity(self):
        events = np.array([[0.01, 2, 3, 1], [0.02, 2, 3, 1], [0.03, 2, 3, -1]])
        counts = count_frame(events, 8, 8)
        self.assertEqual(counts[0, 3, 2], 2)
        self.assertEqual(counts[1, 3, 2], 1)
        self.assertEqual(int(counts.sum()), len(events))

    def test_window_excludes_future_and_initialization(self):
        counts = np.arange(13, dtype=np.uint16).reshape(13, 1, 1, 1)
        labels = np.arange(13) % 3
        voxels, targets, indices = make_windows(counts, labels, bins=5)
        self.assertEqual(indices.tolist(), [5, 10])
        self.assertEqual(voxels[0, :, 0, 0, 0].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(voxels[1, :, 0, 0, 0].tolist(), [6, 7, 8, 9, 10])
        self.assertEqual(targets.tolist(), labels[[5, 10]].tolist())
        changed = counts.copy()
        changed[6:] = 999
        self.assertTrue(np.array_equal(make_windows(changed, labels, 5)[0][0], voxels[0]))

    def test_loader_rejects_geometry_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"episodes": [{"geometry_sha256": "same-map", "split": "train", "samples": 1},
                                      {"geometry_sha256": "same-map", "split": "test", "samples": 1}]}
            (Path(directory) / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Fuite"):
                EventDataset(directory, "train")
