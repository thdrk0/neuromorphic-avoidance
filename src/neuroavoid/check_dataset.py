"""Validate partitions, causal labels and exact reconstruction of event windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from neuroavoid.dataset import EventDataset, LABELS, count_frame
from neuroavoid.scenes import make_scene


def validate(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Enregistrement incomplet")
    fingerprints = {}
    total = 0
    for split in ("train", "validation", "test"):
        data = EventDataset(root, split)
        if not len(data):
            raise ValueError(f"Partition vide : {split}")
    for episode in manifest["episodes"]:
        expected_xml, expected = make_scene(episode["scene_id"])
        assert (root / episode["raw_file"]).parent.joinpath("scene.xml").read_text() == expected_xml
        if expected["split"] != episode["split"] or expected["geometry_sha256"] != episode["geometry_sha256"]:
            raise ValueError("Scene ou partition incoherente")
        fingerprint = episode["geometry_sha256"]
        if fingerprint in fingerprints and fingerprints[fingerprint] != episode["split"]:
            raise ValueError("Geometrie partagee entre partitions")
        fingerprints[fingerprint] = episode["split"]
        with np.load(root / episode["raw_file"], allow_pickle=False) as raw, np.load(root / episode["samples_file"], allow_pickle=False) as samples:
            events, offsets, times = raw["events"], raw["event_offsets"], raw["frame_times"]
            labels, raw_labels, voxels, indices = samples["labels"], raw["labels"], samples["voxels"], samples["frame_indices"]
            bins = manifest["bins"]
            assert len(offsets) == len(times) + 1 and offsets[0] == 0 and offsets[-1] == len(events)
            assert np.all(np.diff(offsets) >= 0) and np.all(np.diff(times) > 0)
            assert len(raw["rgb"]) == len(times) == len(raw_labels)
            assert np.array_equal(indices, np.arange(bins, len(times), bins))
            assert np.array_equal(labels, raw_labels[indices])
            assert np.array_equal(samples["times"], times[indices])
            assert np.isin(labels, np.arange(len(LABELS))).all()
            assert voxels.dtype == np.uint16 and voxels.shape == (len(indices), bins, 2, 128, 128)
            assert len(indices) == episode["samples"]
            if len(events):
                assert np.isfinite(events).all() and np.all(np.diff(events[:, 0]) >= -1e-5)
                assert np.isin(events[:, 3], [-1, 1]).all()
                assert np.all(events[:, 1:3] == np.floor(events[:, 1:3]))
                assert np.all((events[:, 1:3] >= 0) & (events[:, 1:3] < 128))
            assert episode["class_counts"] == {name: int(np.sum(labels == i)) for i, name in enumerate(LABELS)}
            for sample_index, end in enumerate(indices):
                for slot, frame in enumerate(range(end - bins + 1, end + 1)):
                    batch = events[offsets[frame]:offsets[frame + 1]]
                    if len(batch):
                        assert np.all(batch[:, 0] >= times[frame - 1] - 1e-5)
                        assert np.all(batch[:, 0] <= times[frame] + 1e-5)
                    assert np.array_equal(voxels[sample_index, slot], count_frame(batch))
            total += len(indices)
    return {"status": "OK", "scenes": len(fingerprints), "samples": total,
            "split_counts": manifest["split_counts"], "class_counts": manifest["class_counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/dataset"))
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2))


if __name__ == "__main__":
    main()
