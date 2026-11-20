#!/usr/bin/env python3
"""Verify metric CAD hypotheses through independently measured camera rays."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi

from cad_verify import CAD_URL, MODEL_IDS, load_depth, load_mesh, rendered_views, render_function, rotations


def transform(camera):
    return np.asarray(camera["worldToCameraR"]).reshape(3, 3), np.asarray(camera["worldToCameraT"])


def world_pose(proposal, fit, camera, views):
    base = proposal.get("scanCenter")
    if base is None:
        x, y, w, h = proposal["box"]
        base = [x + (w - 1) / 2, y + (h - 1) / 2]
    z = proposal["centerDepthMm"]
    fx, _, cx, _, fy, cy, _, _, _ = camera["K"]
    shift = np.asarray(fit["centerShiftMm"]) * np.array([fx, fy]) / z
    u, v = np.asarray(base) + shift
    front_depth = fit["surfaceDepthMm"]
    model_z = front_depth - views.medians[fit["orientationIndex"]]
    model_t = model_z * np.array([(u - cx) / fx, (v - cy) / fy, 1.])
    camera_r, camera_t = transform(camera)
    return camera_r.T @ rotations(fit["tilt"], fit["roll"]), camera_r.T @ (model_t - camera_t)


def measured_projection(mesh, pose, camera, depth, render):
    world_r, world_t = pose
    camera_r, camera_t = transform(camera)
    vertices, faces = mesh
    local = vertices @ (camera_r @ world_r).T + (camera_r @ world_t + camera_t)
    if local[:, 2].min() < 100:
        return None
    fx, _, cx, _, fy, cy, _, _, _ = camera["K"]
    local[:, :2] = local[:, :2] / local[:, 2, None] * np.array([fx, fy]) + np.array([cx, cy])
    position = np.rint((local[:, :2].min(axis=0) + local[:, :2].max(axis=0)) / 2).astype(int)
    local[:, :2] -= position
    radius = int(np.ceil(np.abs(local[:, :2]).max())) + 12
    if radius > 165:
        return None
    width = radius * 2 + 1
    model = np.full((width, width), np.inf, dtype=np.float32)
    render(np.ascontiguousarray(local, dtype=np.float32), faces, len(faces), width, 1., model)
    target = np.isfinite(model)
    rows = np.arange(position[1] - radius, position[1] + radius + 1)
    cols = np.arange(position[0] - radius, position[0] + radius + 1)
    inside = ((rows[:, None] >= 0) & (rows[:, None] < depth.shape[0])
              & (cols[None, :] >= 0) & (cols[None, :] < depth.shape[1]))
    if (target & inside).sum() < max(20, target.sum() * .35):
        return None
    observed = depth[rows[:, None].clip(0, depth.shape[0] - 1), cols[None, :].clip(0, depth.shape[1] - 1)]
    delta = (observed - model)[target]
    present = (observed[target] > 0) & inside[target]
    match = np.exp(-np.abs(delta) / 6) * present
    contradiction = ((delta > 10) & present).mean()
    outside = ndi.binary_dilation(target, iterations=5) & ~target & inside
    nearest = ndi.distance_transform_edt(~target, return_distances=False, return_indices=True)
    continuation = ((np.abs(observed[outside] - model[tuple(nearest[:, outside])]) < 8)).mean() if outside.any() else 0.
    supported = target.copy()
    supported[target] = (np.abs(delta) <= 9) & present
    # Keep connected measured surface fragments, removing isolated noise.
    support = ndi.convolve(supported.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) >= 9
    supported &= support
    fit = match.mean() - .30 * contradiction - .28 * continuation

    def bounds(mask):
        y, x = np.nonzero(mask)
        return [int(cols[x.min()]), int(rows[y.min()]), int(np.ptp(x) + 1), int(np.ptp(y) + 1)] if len(y) else None

    return {"score": round(float(fit), 6), "support": round(float(supported.sum() / target.sum()), 6),
            "contradiction": round(float(contradiction), 6), "continuation": round(float(continuation), 6),
            "visibleBox": bounds(supported), "projectedBox": bounds(target & inside)}


def build_consensus(manifest, raw, directory, meshes, views, render):
    proposals = {frame["id"]: frame["detections"] for frame in raw["frames"]}
    depth = {frame["id"]: load_depth(frame, directory) for frame in manifest}
    output = []
    for frame in manifest:
        frame_id = frame["id"]
        related = [peer for peer in manifest if peer["scene"] == frame["scene"]]
        for index, row in enumerate(proposals[frame_id]):
            fit = row["cad"]["target"]
            if fit["metricFit"] < .31 or row["cad"]["metricMargin"] < -.15:
                continue
            rivals = sorted((x for x in row["cad"]["competition"] if x["part"] != 1),
                            key=lambda x: -x["metricFit"])[:2]
            pose = world_pose(row, fit, frame["camera"], views[1])
            projected = {}
            for peer in related:
                peer_id = peer["id"]
                predicted = measured_projection(meshes[1], pose, peer["camera"], depth[peer_id], render)
                if predicted is not None:
                    projected[peer_id] = predicted
            if len(projected) <= 1:
                continue
            competitors = []
            for rival in rivals:
                part = rival["part"]
                other = world_pose(row, rival, frame["camera"], views[part])
                projections = {}
                for peer in related:
                    peer_id = peer["id"]
                    value = measured_projection(meshes[part], other, peer["camera"], depth[peer_id], render)
                    if value is not None:
                        projections[peer_id] = value["score"]
                competitors.append({"part": part, "scores": projections})
            output.append({"id": frame_id, "proposalRank": index + 1, "box": row["box"],
                           "reference": row.get("reference"), "positionMm": pose[1].tolist(),
                           "fit": fit["metricFit"], "margin": row["cad"]["metricMargin"],
                           "projections": projected, "competitors": competitors})
        print(frame_id, len(output), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path(__file__).resolve().parent / "models.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw = json.loads(args.verified.read_text(encoding="utf-8"))
    views = {view.part: view for view in rendered_views(args.models)}
    with ZipFile(args.models) as archive:
        meshes = {part: load_mesh(archive, part) for part in MODEL_IDS}
    render = render_function(Path(__file__).resolve().parent)
    hypotheses = build_consensus(manifest, raw, args.manifest.parent, meshes, views, render)
    args.output.write_text(json.dumps({"modelSource": CAD_URL, "hypotheses": hypotheses}, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
