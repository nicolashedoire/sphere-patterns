"""Prepare a small photo gallery from the measured depth-component trial."""
import argparse
import json
from pathlib import Path

from PIL import Image

METHODS = {
    "wave": "Returning wave reference",
    "rgbd": "Single-view RGB-D",
    "components": "Depth components + reference check",
}
# Show both successful setups, the previous detector's false alarm, and misses.
EXAMPLES = ("07-000130", "20-000288", "19-000291", "18-000236",
            "07-000054", "07-000185", "07-000285", "20-000339")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(report, manifest, output):
    metrics = read(report)
    source = {row["id"]: row for row in read(manifest)}
    measurements = {key: {frame["id"]: frame for frame in metrics["methods"][name]["visible"]["frames"]}
                    for key, name in METHODS.items()}
    visibility = metrics["protocol"]["visibleFractionAtLeast"]
    scenes = []
    for label in EXAMPLES:
        frame = measurements["components"][label]
        image_name = f"components-{label}.jpg"
        with Image.open(manifest.parent / source[label]["rgb"]) as opened:
            photo = opened.convert("RGB")
            photo.save(output / image_name, quality=89, optimize=True)
        detections = {key: [{**row, "overlap": row["nearestTargetIoU"]} for row in values[label]["accepted"]]
                      for key, values in measurements.items()}
        scenes.append({"scene": frame["scene"], "frame": frame["frame"], "image": image_name,
                       "width": photo.width, "height": photo.height, "detections": detections,
                       "targets": [{"box": target["visibleBox"]} for target in frame["targets"]
                                   if target["visibility"] >= visibility]})
    summary = {key: metrics["methods"][name]["visible"]["summary"] for key, name in METHODS.items()}
    data = {"overlapThreshold": metrics["protocol"]["boxIoUAtLeast"], "visibilityThreshold": visibility,
            "photoCount": len(source), "totals": summary, "scenes": scenes}
    (output / "benchmark-components.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.report, args.manifest, args.destination)


if __name__ == "__main__":
    main()
