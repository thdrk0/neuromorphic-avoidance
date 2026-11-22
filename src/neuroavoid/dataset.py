"""Causal event windows and a recording format shared by future SNN/CNN models."""

import json
from pathlib import Path

import numpy as np

LABELS = ("LEFT", "STRAIGHT", "RIGHT")


def count_frame(events, height=128, width=128):
    """Two integer channels, ON then OFF, preserving event multiplicity."""
    counts = np.zeros((2, height, width), np.uint32)
    if len(events):
        channel = (events[:, 3] < 0).astype(int)
        np.add.at(counts, (channel, events[:, 2].astype(int), events[:, 1].astype(int)), 1)
    if counts.max() > np.iinfo(np.uint16).max:
        raise OverflowError("Trop d'evenements par pixel pour uint16")
    return counts.astype(np.uint16)


def make_windows(frame_counts, frame_labels, bins=5):
    """Non-overlapping causal windows; label at their final observation time.

    Frame zero only initializes v2e. A window ending at frame k contains batches
    k-bins+1..k and predicts the teacher command applied AFTER frame k.
    """
    if bins < 1 or len(frame_counts) != len(frame_labels):
        raise ValueError("Nombre de bins ou alignement invalide")
    indices = np.arange(bins, len(frame_counts), bins, dtype=np.int64)
    shape = (0, bins) + tuple(frame_counts.shape[1:])
    voxels = np.stack([frame_counts[k - bins + 1:k + 1] for k in indices]) if len(indices) else np.empty(shape, np.uint16)
    labels = np.asarray(frame_labels, dtype=np.int64)[indices]
    return voxels, labels, indices


class EventDataset:
    """Episode-wise access; privileged poses/maps are not returned as inputs."""
    def __init__(self, root, split):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if split not in ("train", "validation", "test"):
            raise ValueError("Split inconnu")
        all_episodes = self.manifest["episodes"]
        owners = {}
        for episode in all_episodes:
            fingerprint = episode["geometry_sha256"]
            if fingerprint in owners and owners[fingerprint] != episode["split"]:
                raise ValueError("Fuite de geometrie entre partitions")
            owners[fingerprint] = episode["split"]
        self.episodes = [e for e in all_episodes if e["split"] == split]
        self.ends = np.cumsum([e["samples"] for e in self.episodes])
        self._cached_episode = None
        self._cached_arrays = None

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        local_index = index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        episode = self.episodes[episode_index]
        if self._cached_episode != episode_index:
            with np.load(self.root / episode["samples_file"], allow_pickle=False) as data:
                self._cached_arrays = (data["voxels"], data["labels"], data["times"])
            self._cached_episode = episode_index
        voxels, labels, times = self._cached_arrays
        return {"events": voxels[local_index].copy(), "label": int(labels[local_index]),
                "time": float(times[local_index]), "scene_id": episode["scene_id"]}
