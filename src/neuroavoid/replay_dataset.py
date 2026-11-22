"""Display a recorded camera/event window with its reference command."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from neuroavoid.dataset import EventDataset, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/dataset"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_replay"))
    args = parser.parse_args()
    dataset = EventDataset(args.root, args.split)
    sample = dataset[args.index]
    episode_index = int(np.searchsorted(dataset.ends, args.index, side="right"))
    episode = dataset.episodes[episode_index]
    local = args.index - (int(dataset.ends[episode_index - 1]) if episode_index else 0)
    with np.load(args.root / episode["samples_file"], allow_pickle=False) as prepared, np.load(args.root / episode["raw_file"], allow_pickle=False) as raw:
        frame_index = int(prepared["frame_indices"][local])
        rgb = raw["rgb"][frame_index]
    counts = sample["events"].astype(np.uint32).sum(axis=0)
    pixels = np.zeros((128, 128, 3), np.uint8)
    pixels[counts[0] > 0, 0] = 255
    pixels[counts[1] > 0, 1:] = 255
    image = Image.fromarray(np.concatenate([rgb, pixels], axis=1)).resize((768, 384), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (768, 420), "#202020")
    canvas.paste(image, (0, 36))
    caption = f"{sample['scene_id']} | {sample['time']:.2f}s | {LABELS[sample['label']]} | RGB / events ({dataset.manifest['window_ms']} ms)"
    ImageDraw.Draw(canvas).text((8, 10), caption, fill="white")
    args.output.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output / "sample.png")
    metadata = {"scene_id": sample["scene_id"], "split": args.split, "index": args.index,
                "time": sample["time"], "command": LABELS[sample["label"]],
                "shape": list(sample["events"].shape), "event_count": int(counts.sum())}
    (args.output / "sample.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if args.show:
        import tkinter as tk
        from PIL import ImageTk
        window = tk.Tk()
        window.title("Replay dataset : image, evenements et commande cible")
        photo = ImageTk.PhotoImage(canvas)
        label = tk.Label(window, image=photo)
        label.pack()
        window.mainloop()


if __name__ == "__main__":
    main()
