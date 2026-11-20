"""Prepare disjoint development and held-out groups of calibrated T-LESS views."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
PREVIOUS = runpy.run_path(str(HERE / "sphere-tless-photos-fetch.py"))
ARCHIVE = PREVIOUS["ARCHIVE"]

PART = 1
SPLITS = {
    "development": {"positive": (5, 9, 17), "absent": (4, 8, 11, 15)},
    "heldout": {"positive": (7, 18, 20), "absent": (1, 2, 3, 6, 12, 16)},
}
CONFIRMATION = {"positive": (7, 18, 20), "absent": (1, 2, 6, 10, 13, 19)}
POSITIVE_LEVELS = (("high", .93, .80, 1.01), ("high", .87, .80, 1.01),
                   ("partial", .55, .25, .78), ("partial", .70, .25, .78))


def select_frames(scene, available, gt, info, positive, old, allow_band_fallback=False, minimum_gap=30):
    if not positive:
        assert all(item["obj_id"] != PART for frame in available for item in gt[str(frame)])
        frames = [frame for frame in available if all(abs(frame - seen) >= minimum_gap for seen in old)]
        return [(frames[round(p * (len(frames) - 1))], "absent") for p in (.22, .54, .81)]

    candidates = {}
    for frame in available:
        entries = [(row["visib_fract"], row["bbox_visib"])
                   for item, row in zip(gt[str(frame)], info[str(frame)]) if item["obj_id"] == PART
                   and min(row["bbox_visib"][2:]) >= 20]
        if entries and all(abs(frame - seen) >= minimum_gap for seen in old):
            candidates[frame] = max(item[0] for item in entries)

    selected = []
    for band, wanted, lower, upper in POSITIVE_LEVELS:
        allowed = [(abs(vis - wanted), frame) for frame, vis in candidates.items()
                   if lower <= vis < upper and all(abs(frame - saved) >= 20 for saved, _ in selected)]
        if not allowed:
            # Keep the declared visibility band and use a closer alternate camera
            # only when the requested spacing cannot be met.
            allowed = [(abs(vis - wanted), frame) for frame, vis in candidates.items()
                       if lower <= vis < upper and frame not in {saved for saved, _ in selected}]
        if not allowed and allow_band_fallback:
            allowed = [(abs(vis - wanted), frame) for frame, vis in candidates.items()
                       if vis >= .25 and frame not in {saved for saved, _ in selected}]
        if not allowed:
            raise ValueError(f"No fresh {band} frame in scene {scene}")
        frame = min(allowed)[1]
        selected.append((frame, "high" if candidates[frame] >= .8 else "partial"))
    return sorted(selected)


def prepare(split, groups, root, previous, source):
    views, truth = [], []
    with ARCHIVE["load_archive"]("test") as archive:
        names = archive.namelist()
        for scene_id in sorted(groups["positive"] + groups["absent"]):
            key = f"test_primesense/{scene_id:06d}"
            gt, info, cameras = [PREVIOUS["annotation"](archive, key, kind) for kind in ("gt", "gt_info", "camera")]
            available = sorted(int(name.rsplit("/", 1)[-1][:-4]) for name in names if name.startswith(key + "/rgb/"))
            selected = select_frames(scene_id, available, gt, info, scene_id in groups["positive"],
                                     previous.get(scene_id, ()) if split != "development" else (),
                                     allow_band_fallback=split != "heldout",
                                     minimum_gap=10 if split == "confirmation" else 30)
            for frame, band in selected:
                label = f"{scene_id:02d}-{frame:06d}"
                camera = cameras[str(frame)]
                item = PREVIOUS["copy_frame"](archive, root, key, frame, label, camera)
                item["scene"] = scene_id
                item["camera"]["worldToCameraR"] = camera["cam_R_w2c"]
                item["camera"]["worldToCameraT"] = camera["cam_t_w2c"]
                views.append(item)
                targets = [
                    {"index": idx, "visibleBox": row["bbox_visib"], "amodalBox": row["bbox_obj"],
                     "visibility": row["visib_fract"]}
                    for idx, (obj, row) in enumerate(zip(gt[str(frame)], info[str(frame)])) if obj["obj_id"] == PART
                ]
                truth.append({"id": label, "scene": scene_id, "frame": frame, "band": band, "targets": targets})
            print(split, scene_id, [frame for frame, _ in selected], flush=True)
    PREVIOUS["dump"](root / f"{split}.json", views)
    PREVIOUS["dump"](root / f"{split}-truth.json", {"source": source, "target": PART, "frames": truth, "split": groups})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--confirmation-after", type=Path, action="append",
                        help="Exclude photos from an earlier held-out manifest. Repeat for several manifests.")
    args = parser.parse_args()
    for folder in ("rgb", "depth"):
        (args.directory / folder).mkdir(parents=True, exist_ok=True)
    older = json.loads((HERE.parent / "outputs/sphere-tless-fixtures.json").read_text(encoding="utf-8"))
    previous = {}
    for sample in older["cases"]:
        previous.setdefault(sample["scene"], set()).add(sample["frame"])
    for scene, frame in PREVIOUS["EMPTY"]:
        previous.setdefault(scene, set()).add(frame)
    if args.confirmation_after:
        for manifest in args.confirmation_after:
            for view in json.loads(manifest.read_text(encoding="utf-8")):
                scene, frame = map(int, view["id"].split("-"))
                previous.setdefault(scene, set()).add(frame)
        prepare("confirmation", CONFIRMATION, args.directory, previous, older["source"])
        return
    with ThreadPoolExecutor(max_workers=len(SPLITS)) as pool:
        tasks = [pool.submit(prepare, split, groups, args.directory, previous, older["source"])
                 for split, groups in SPLITS.items()]
        for task in tasks:
            task.result()


if __name__ == "__main__":
    main()
