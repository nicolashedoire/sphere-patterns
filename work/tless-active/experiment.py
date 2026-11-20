"""Plan one more RGB-D observation, then compare it with two frozen CAD poses."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from zipfile import ZipFile

import numpy as np

CAD_DIR = Path(__file__).resolve().parents[1] / "tless-cad"
sys.path.insert(0, str(CAD_DIR))
from cad_consensus import transform, world_pose
from cad_verify import MODEL_IDS, box_iou, load_depth, load_mesh, render_function, rendered_views
from decision_stream import decide
from view_projection import point_cloud, project_pair

RAY_LIMIT = 128
BOXES_PER_VIEW = 3
POLICY = {"first_threshold": .06, "final_threshold": .06, "second_weight": .45}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def models(path):
    with ZipFile(path) as archive:
        meshes = {part: load_mesh(archive, part) for part in MODEL_IDS}
    return meshes, render_function(CAD_DIR)


def camera_position(camera):
    rotation, translation = transform(camera)
    return -rotation.T @ translation


def sampled_cloud(depth, camera):
    sampled_camera = {**camera, "K": [value / 2 if i < 6 else value for i, value in enumerate(camera["K"])]}
    return point_cloud(depth[::2, ::2], sampled_camera)


def choose_candidates(rows):
    ranked = sorted(rows, key=lambda row: -row["cad"]["competition"][0]["metricFit"])
    selected = []
    for row in ranked:
        a, b = row["cad"]["competition"][:2]
        if a["metricFit"] < .30 or b["metricFit"] < .20:
            continue
        if any(box_iou(row["box"], chosen["box"]) >= .35 for chosen in selected):
            continue
        selected.append(row)
        if len(selected) == BOXES_PER_VIEW:
            break
    return selected


def get_projection(hypotheses, view, points, shape, meshes, render):
    first, second = hypotheses
    poses = [(np.asarray(item["rotation"]), np.asarray(item["position"])) for item in hypotheses]
    return project_pair(meshes[first["part"]], poses[0], meshes[second["part"]], poses[1],
                        view["camera"], points, render, shape)


def probe_indices(projection):
    indices = np.flatnonzero(projection.selected)
    if len(indices) > RAY_LIMIT:
        indices = indices[np.linspace(0, len(indices) - 1, RAY_LIMIT, dtype=int)]
    return np.unravel_index(indices, projection.first.shape)


def make_plans(origins, manifest, verified, model_file):
    seeds = {frame["id"]: frame["detections"] for frame in read(verified)["frames"]}
    views = read(manifest)
    meshes, render = models(model_file)
    banks = {view.part: view for view in rendered_views(model_file)}
    plans = []
    for origin in read(origins):
        frame_id, camera = origin["id"], origin["camera"]
        depth = load_depth(origin, origins.parent)
        points = sampled_cloud(depth, camera)
        peers = sorted((view for view in views if view["scene"] == origin["scene"] and view["id"] != frame_id),
                       key=lambda view: view["id"])
        for n, row in enumerate(choose_candidates(seeds[frame_id]), 1):
            hypotheses = []
            for fit in row["cad"]["competition"][:2]:
                part = fit["part"]
                rotation, center = world_pose(row, fit, camera, banks[part])
                hypotheses.append({"part": part, "rotation": rotation.tolist(), "position": center.tolist(),
                                   "initialFit": fit["metricFit"]})
            center = np.mean([h["position"] for h in hypotheses], axis=0)
            direction = camera_position(camera) - center
            direction /= np.linalg.norm(direction)
            options = []
            for view in peers:
                projection = get_projection(hypotheses, view, points, depth.shape, meshes, render)
                if projection is None:
                    priority = 0.
                else:
                    priority = projection.priority
                relative = camera_position(view["camera"]) - center
                angle = np.degrees(np.arccos(np.clip(direction @ relative / np.linalg.norm(relative), -1, 1)))
                options.append({"view": view["id"], "priority": round(priority, 6), "angleDeg": round(float(angle), 3)})
            active = max(options, key=lambda option: option["priority"])["view"]
            fixed = min(options, key=lambda option: abs(option["angleDeg"] - 45))["view"]
            random_view = random.Random(f"8819:{frame_id}:{n}").choice(options)["view"]
            plans.append({"origin": frame_id, "scene": origin["scene"], "query": n, "box": row["box"],
                          "hypotheses": hypotheses, "options": options,
                          "choices": {"active": active, "fixed": fixed, "random": random_view}})
        print(f"Planned {frame_id}", flush=True)
    return {"settings": {**POLICY, "rayLimit": RAY_LIMIT, "boxesPerFrame": BOXES_PER_VIEW}, "plans": plans}


def measured_evidence(first, second, observed):
    nearest = np.minimum(first, second)
    usable = (observed > 0) & (observed >= nearest - 10)
    if usable.sum() < 12:
        return {"margin": None, "reads": len(observed), "usable": int(usable.sum())}
    a, b, z = first[usable], second[usable], observed[usable]
    # An observed surface at the expected depth supports that model. Where a
    # model leaves a clear ray, a sufficiently distant measured return supports it.
    def support(shape, rival):
        return np.where(np.isfinite(shape), np.exp(-np.abs(z - shape) / 6), .75 * np.clip((z - rival - 8) / 18, 0, 1))
    votes = support(a, b) - support(b, a)
    return {"margin": round(float(votes.mean()), 6), "reads": len(observed), "usable": int(usable.sum())}


def compare(depth, projection):
    if projection is None:
        empty = {"margin": None, "reads": 0, "usable": 0}
        return {"probe": empty, "dense": empty}
    def sample(y, x):
        return measured_evidence(projection.first[y, x], projection.second[y, x],
                                 depth[projection.rows[y], projection.cols[x]])
    return {"probe": sample(*probe_indices(projection)), "dense": sample(*np.nonzero(projection.visibility))}


def reveal(plans, manifest, model_file):
    records = {view["id"]: view for view in read(manifest)}
    meshes, render = models(model_file)
    result = []
    origins = {}
    for trial in plans["plans"]:
        frame = trial["origin"]
        if frame not in origins:
            origin = records[frame]
            depth = load_depth(origin, manifest.parent)
            origins[frame] = (sampled_cloud(depth, origin["camera"]), depth.shape)
        points, shape = origins[frame]
        opened = {}
        for frame in dict.fromkeys(trial["choices"].values()):
            view = records[frame]
            projection = get_projection(trial["hypotheses"], view, points, shape, meshes, render)
            opened[frame] = compare(load_depth(view, manifest.parent), projection)
        a, b = trial["hypotheses"]
        initial = a["initialFit"] - b["initialFit"]
        policy = {key: plans["settings"][key] for key in POLICY}
        early = decide(initial, **policy)
        verdicts = {"one_view": {"part": a["part"] if early.early == 1 else b["part"] if early.early == -1 else None}}
        for method, mode in (("active", "probe"), ("fixed", "probe"), ("random", "probe"), ("active", "dense")):
            observation = opened[trial["choices"][method]][mode]
            withheld = decide(initial, observation["margin"], second_ready=False, **policy)
            if withheld.final != early.early:
                raise RuntimeError("A withheld observation changed the early decision")
            update = decide(initial, observation["margin"], **policy)
            verdicts[f"{method}_{mode}"] = {"part": a["part"] if update.final == 1 else b["part"] if update.final == -1 else None,
                                             "revised": update.revised, **observation,
                                             "decisionMargin": update.final_margin}
        result.append({**trial, "observations": opened, "verdicts": verdicts})
    return {**plans, "plans": result}


def truth_part(trial, labels):
    matches = [(float(box_iou(trial["box"], row["box"])), row["part"], row["visibility"])
               for row in labels[trial["origin"]] if row["visibility"] >= .20 and min(row["box"][2:]) > 0]
    iou, identity, visibility = max(matches, default=(0., None, None))
    return (identity, visibility) if iou >= .50 else (None, None)


def evaluate(results, annotations):
    methods = {}
    panel = []
    for row in results["plans"]:
        actual, visibility = truth_part(row, annotations)
        possible = actual in [part["part"] for part in row["hypotheses"]]
        early = None
        for method, verdict in row["verdicts"].items():
            predicted = verdict["part"]
            score = "abstained" if predicted is None else "correct" if predicted == actual else "falseBox" if actual is None else "wrongPart"
            verdict["result"] = score
            if method == "one_view":
                early = score
            total = methods.setdefault(method, Counter())
            total[score] += 1
            if early != score:
                total[f"{early}_to_{score}"] += 1
            total["coveredPair"] += possible
            if possible:
                total[f"inPair_{score}"] += 1
            if visibility is not None and visibility < .75:
                total[f"partial_{score}"] += 1
            if verdict.get("revised", False):
                total[f"revisedTo_{score}"] += 1
            total["reads"] += verdict.get("reads", 0)
            total["usable"] += verdict.get("usable", 0)
        panel.append({**row, "actual": actual, "visibility": visibility, "coveredPair": possible})
    return {"settings": results["settings"], "queries": len(panel), "physicalScenes": len({row["scene"] for row in panel}),
            "results": methods, "cases": panel}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    for name, paths in (("plan", ("origins", "manifest", "verified", "models")),
                        ("reveal", ("plans", "manifest", "models")), ("evaluate", ("predictions", "labels"))):
        command = sub.add_parser(name)
        for path in (*paths, "output"):
            command.add_argument("--" + path, type=Path, required=True)
    args = parser.parse_args()
    if args.step == "plan":
        result = make_plans(args.origins, args.manifest, args.verified, args.models)
    elif args.step == "reveal":
        result = reveal(read(args.plans), args.manifest, args.models)
    else:
        result = evaluate(read(args.predictions), read(args.labels))
        print(json.dumps(result["results"], indent=2))
    write(args.output, result)


if __name__ == "__main__":
    main()
