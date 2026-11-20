"""Fetch a fixed full-frame T-LESS panel from its official pinned ZIPs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
ARCHIVE = runpy.run_path(str(HERE / "sphere-tless-fetch.py"))
TARGET = 1
# Frozen from a farthest-view-direction sample of the official training split.
REFERENCES = (0, 1286, 688, 663, 621, 720, 507, 414, 325, 963, 914, 311,
              945, 928, 299, 354, 975, 742, 752, 600, 614, 462, 695, 262)
CALIBRATION = (1, 430, 865, 1295)
EMPTY = ((1, 1), (2, 3), (12, 2), (16, 8))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def annotation(archive, scene, name):
    return json.loads(archive.read(f"{scene}/scene_{name}.json"))


def copy_frame(archive, root, scene, frame, label, camera, mask=None, box=None):
    def copy(member, destination):
        target = root / destination
        if not target.is_file():
            target.write_bytes(archive.read(member))
        return destination

    item = {"id": label}
    for mode in ("rgb", "depth"):
        item[mode] = copy(f"{scene}/{mode}/{frame:06d}.png", f"{mode}/{label}.png")
    item["camera"] = {"K": camera["cam_K"], "depthScale": camera["depth_scale"]}
    if mask is not None:
        item["mask"] = copy(f"{scene}/mask_visib/{frame:06d}_{mask:06d}.png", f"masks/{label}.png")
        item["box"] = box
    return item


def training(root):
    references, calibration = [], []
    with ARCHIVE["load_archive"]("train") as archive:
        scene = f"train_primesense/{TARGET:06d}"
        info = annotation(archive, scene, "gt_info")
        cams = annotation(archive, scene, "camera")
        for frame in REFERENCES:
            key = str(frame)
            references.append(copy_frame(archive, root, scene, frame, f"train-{TARGET:02d}-{frame:06d}",
                                         cams[key], 0, info[key][0]["bbox_visib"]))
        for frame in CALIBRATION:
            key = str(frame)
            item = copy_frame(archive, root, scene, frame, f"cal-{TARGET:02d}-{frame:06d}", cams[key])
            calibration.append({**item, "boxes": [info[key][0]["bbox_visib"]]})
        for part in (2, 3, 4, 7):
            scene = f"train_primesense/{part:06d}"
            frame = 0
            item = copy_frame(archive, root, scene, frame, f"cal-{part:02d}-{frame:06d}",
                              annotation(archive, scene, "camera")["0"])
            calibration.append({**item, "boxes": []})
    dump(root / "train.json", references)
    dump(root / "calibration.json", calibration)
    print(f"{len(references)} training views and {len(calibration)} calibration frames", flush=True)


def test_panel(root):
    prior = json.loads((HERE.parent / "outputs/sphere-tless-fixtures.json").read_text(encoding="utf-8"))
    frames = {(item["scene"], item["frame"]): item["band"] for item in prior["cases"] if item["part"] == TARGET}
    frames.update({sample: "absent" for sample in EMPTY})
    scenes, truth = [], []
    with ARCHIVE["load_archive"]("test") as archive:
        cached = {}
        for (scene_id, frame), band in sorted(frames.items()):
            scene = f"test_primesense/{scene_id:06d}"
            if scene not in cached:
                cached[scene] = tuple(annotation(archive, scene, kind) for kind in ("gt", "gt_info", "camera"))
            gt, info, cams = cached[scene]
            key = str(frame)
            label = f"{scene_id:02d}-{frame:06d}"
            frame_truth = [
                {"visibleBox": pose_info["bbox_visib"], "amodalBox": pose_info["bbox_obj"],
                 "visibility": pose_info["visib_fract"], "index": i}
                for i, (pose, pose_info) in enumerate(zip(gt[key], info[key])) if pose["obj_id"] == TARGET
            ]
            assert bool(frame_truth) == (band != "absent")
            scenes.append(copy_frame(archive, root, scene, frame, label, cams[key]))
            truth.append({"id": label, "scene": scene_id, "frame": frame, "band": band, "targets": frame_truth})
            print(f"Test frame {label}", flush=True)
    dump(root / "scenes.json", scenes)
    dump(root / "truth.json", {"target": TARGET, "source": prior["source"], "frames": truth})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for folder in ("rgb", "depth", "masks"):
        (args.directory / folder).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(function, args.directory) for function in (training, test_panel)]
        for task in tasks:
            task.result()


if __name__ == "__main__":
    main()
