"""Select calibrated T-LESS views without looking at the objects in the images."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np

PHOTOS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "sphere-tless-photos-fetch.py"))
DEVELOPMENT_SCENES = {4, 5, 8, 9, 11, 15, 17}
VIEWS_PER_SCENE = 6


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def camera_direction(info):
    rotation = np.asarray(info["cam_R_w2c"]).reshape(3, 3)
    center = -rotation.T @ np.asarray(info["cam_t_w2c"])
    return center / np.linalg.norm(center)


def choose_views(frames, cameras):
    directions = {frame: camera_direction(cameras[str(frame)]) for frame in frames}
    chosen = [max(frames, key=lambda frame: (directions[frame][2], -frame))]
    for _ in range(VIEWS_PER_SCENE - 1):
        selected = max((frame for frame in frames if frame not in chosen),
                       key=lambda frame: (1 - max(directions[frame] @ directions[other] for other in chosen), -frame))
        chosen.append(selected)
    return chosen


def fetch_views(archive, root, excluded, selection):
    prior = read(Path(__file__).resolve().parents[2] / "outputs/sphere-tless-fixtures.json")
    skip = {f"{item['scene']:02d}-{item['frame']:06d}" for item in prior["cases"]}
    skip.update(f"{scene:02d}-{frame:06d}" for scene, frame in PHOTOS["EMPTY"])
    skip.update(row["id"] for path in excluded for row in read(path))
    for name in ("depth", "rgb"):
        (root / name).mkdir(parents=True, exist_ok=True)
    files = archive.namelist()
    views = []
    origins = []
    for scene in sorted(set(range(1, 21)) - DEVELOPMENT_SCENES):
        prefix = f"test_primesense/{scene:06d}"
        cameras = PHOTOS["annotation"](archive, prefix, "camera")
        available = sorted(int(Path(name).stem) for name in files if name.startswith(prefix + "/depth/")
                           and name.endswith(".png") and f"{scene:02d}-{Path(name).stem}" not in skip)
        chosen = selection[str(scene)] if selection is not None else choose_views(available, cameras)
        if len(set(chosen)) != VIEWS_PER_SCENE or any(frame not in available for frame in chosen):
            raise ValueError(f"Invalid recorded camera selection for scene {scene}")
        for rank, frame in enumerate(chosen):
            label = f"{scene:02d}-{frame:06d}"
            camera = cameras[str(frame)]
            record = PHOTOS["copy_frame"](archive, root, prefix, frame, label, camera)
            record["scene"] = scene
            record["camera"].update(worldToCameraR=camera["cam_R_w2c"], worldToCameraT=camera["cam_t_w2c"])
            views.append(record)
            if rank % 2 == 0:
                origins.append(record)
        print(f"Scene {scene} has {VIEWS_PER_SCENE} selected camera positions", flush=True)
    PHOTOS["dump"](root / "views.json", views)
    PHOTOS["dump"](root / "origins.json", origins)


def fetch_labels(archive, manifest, destination):
    cached = {}
    rows = {}
    for item in read(manifest):
        scene, frame = map(int, item["id"].split("-"))
        if scene not in cached:
            prefix = f"test_primesense/{scene:06d}"
            cached[scene] = [PHOTOS["annotation"](archive, prefix, kind) for kind in ("gt", "gt_info")]
        objects, info = (part[str(frame)] for part in cached[scene])
        rows[item["id"]] = [{"part": a["obj_id"], "box": b["bbox_visib"], "visibility": b["visib_fract"]}
                            for a, b in zip(objects, info, strict=True)]
    PHOTOS["dump"](destination, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    panel = commands.add_parser("views")
    panel.add_argument("directory", type=Path)
    selection = panel.add_mutually_exclusive_group()
    selection.add_argument("--exclude", action="append", type=Path, default=[])
    selection.add_argument("--selection", type=Path, help="Replay the previously recorded camera frame IDs")
    labels = commands.add_parser("labels")
    labels.add_argument("manifest", type=Path)
    labels.add_argument("destination", type=Path)
    args = parser.parse_args()
    with PHOTOS["ARCHIVE"]["load_archive"]("test") as archive:
        if args.command == "views":
            fetch_views(archive, args.directory, args.exclude, read(args.selection) if args.selection else None)
        else:
            fetch_labels(archive, args.manifest, args.destination)


if __name__ == "__main__":
    main()
