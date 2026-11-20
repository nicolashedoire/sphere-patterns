"""Prepare measured camera support masks for the returning-reference probe."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 25


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_mask(entry, directory):
    x, y, w, h = entry["box"]
    side = math.ceil(max(w, h) * 1.25)
    xs = x + (w - side) / 2 + side * (np.arange(SIZE) + .5) / SIZE
    ys = y + (h - side) / 2 + side * (np.arange(SIZE) + .5) / SIZE
    with Image.open(directory / entry["mask"]) as image:
        data = np.asarray(image.convert("L"))
        xx, yy = np.meshgrid(xs.astype(int), ys.astype(int))
        height, width = data.shape
        legal = (xx >= 0) & (yy >= 0) & (xx < width) & (yy < height)
        return (legal & (data[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)] > 0)).astype(int).ravel().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = {item["id"]: item for item in read(args.training)}
    predictions = read(args.predictions)["frames"]
    templates, candidates = {}, []
    for frame in predictions:
        for item in frame["detections"]:
            if not item["accepted"]:
                continue
            key = item["reference"]
            if key not in templates:
                templates[key] = reference_mask(entries[key], args.training.parent)
            candidates.append({
                "id": frame["id"], "box": item["box"], "reference": key,
                "detectorScore": item["score"], "supportMask": np.ravel(item["supportMask"]).astype(int).tolist(),
            })
    args.output.write_text(json.dumps({"size": SIZE, "candidates": candidates, "templates": templates},
                                      indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
