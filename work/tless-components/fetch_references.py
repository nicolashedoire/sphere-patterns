"""Fetch isolated T-LESS target and competitor examples from the training archive."""
import argparse
from pathlib import Path
import runpy

PREVIOUS = runpy.run_path(str(Path(__file__).resolve().parent.parent / "sphere-tless-photos-fetch.py"))
PARTS = (2, 3, 4, 7)
VIEWS = (108, 430, 865, 1184)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    for folder in ("rgb", "depth", "masks"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    PREVIOUS["training"](root)
    records = []
    with PREVIOUS["ARCHIVE"]["load_archive"]("train") as archive:
        for part in PARTS:
            scene = f"train_primesense/{part:06d}"
            camera, labels, objects = [PREVIOUS["annotation"](archive, scene, key)
                                       for key in ("camera", "gt_info", "gt")]
            for view in VIEWS:
                key = str(view)
                label = f"competitor-{part:02d}-{view:06d}"
                index = next(i for i, row in enumerate(objects[key]) if row["obj_id"] == part)
                record = PREVIOUS["copy_frame"](archive, root, scene, view, label, camera[key], index,
                                                labels[key][index]["bbox_obj"])
                records.append({**record, "object": part})
                print(label, flush=True)
    PREVIOUS["dump"](root / "competitors.json", records)


if __name__ == "__main__":
    main()
