"""Fixed scene-level splits: no geometry shared between dataset partitions."""

import hashlib
from importlib.resources import files
import json
import random
import xml.etree.ElementTree as ET

SCENES = [("train_11", "train", 11), ("train_23", "train", 23),
          ("train_37", "train", 37), ("validation_41", "validation", 41),
          ("test_53", "test", 53), ("test_67", "test", 67)]


def make_scene(scene_id):
    entry = next((s for s in SCENES if s[0] == scene_id), None)
    if entry is None:
        raise ValueError(f"Scene inconnue : {scene_id}")
    _, split, seed = entry
    rng = random.Random(seed)
    root = ET.fromstring((files("neuroavoid") / "assets/arena.xml").read_text())
    geometry = []
    for geom in root.findall("./worldbody/geom"):
        name = geom.get("name", "")
        if name.startswith("obstacle_"):
            pos = [float(v) for v in geom.get("pos").split()]
            pos[0] = round(pos[0] + rng.uniform(-0.20, 0.20), 5)
            pos[1] = round(pos[1] + rng.uniform(-0.15, 0.15), 5)
            geom.set("pos", " ".join(map(str, pos)))
        if name.startswith(("wall_", "obstacle_")):
            geometry.append({"name": name, "type": geom.get("type"),
                             "pos": geom.get("pos"), "size": geom.get("size")})
    fingerprint = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    return ET.tostring(root, encoding="unicode"), {
        "scene_id": scene_id, "split": split, "seed": seed, "geometry_sha256": fingerprint,
        "geometry": geometry}


def scene_manifest():
    return [make_scene(scene_id)[1] for scene_id, _, _ in SCENES]
