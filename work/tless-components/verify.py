"""Verify RGB-D proposals with dense depth agreement and physical extent."""
import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

OVERLAP = 0.5
DEPTH_SCALE = 8.0
COARSE_ANGLES = tuple(range(0, 360, 45))
COARSE_XY = (-3., 0., 3.)
FINE_XY = (-2., 0., 2.)
FINE_DEPTH = (-6., 0., 6.)
COARSE_DEPTH = (-10., 0., 10.)
REFINE_PER_BANK = 5


@dataclass(frozen=True)
class DepthTemplate:
    name: str
    points: np.ndarray
    depths: np.ndarray
    background_points: np.ndarray
    background_depths: np.ndarray
    corners: np.ndarray
    coarse_indices: np.ndarray
    background_indices: np.ndarray
    detail_points: np.ndarray
    detail_directions: np.ndarray
    detail_weights: np.ndarray
    detail_indices: np.ndarray
    rim_inside: np.ndarray
    rim_outside: np.ndarray
    rim_depths: np.ndarray
    rim_indices: np.ndarray


def read_baseline(file):
    spec = importlib.util.spec_from_file_location("physical_baseline", file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_manifest(filename):
    path = Path(filename).resolve()
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def read_depth(record, directory):
    with Image.open(directory / record["depth"]) as image:
        depth = np.asarray(image, dtype=np.float32)
    camera = record["camera"]
    return depth * camera["depthScale"], np.array([camera["K"][0], camera["K"][4]], dtype=np.float32)


def color_gradients(record, directory):
    with Image.open(directory / record["rgb"]) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    yy, xx = np.gradient(ndi.gaussian_filter(gray, 1.0))
    magnitude = np.hypot(xx, yy)
    magnitude_safe = np.maximum(magnitude, 0.0001)
    return xx / magnitude_safe, yy / magnitude_safe, magnitude


def subsample(mask, maximum):
    y, x = np.nonzero(mask)
    if len(x) > maximum:
        indices = np.linspace(0, len(x) - 1, maximum, dtype=int)
        x, y = x[indices], y[indices]
    return np.column_stack([x, y]).astype(np.float32)


def make_template(record, directory):
    depth, focal = read_depth(record, directory)
    with Image.open(directory / record["mask"]) as image:
        raw_mask = np.asarray(image.convert("L")) > 0

    mask = ndi.binary_fill_holes(raw_mask)
    inside = mask & (depth > 0)
    if np.count_nonzero(inside) < 40:
        return None
    z = float(np.median(depth[inside]))
    # A small fraction of edge pixels can return the platform behind the part.
    # Keep complete foreground geometry but avoid these depth outliers.
    dl, dh = np.quantile(depth[inside], (0.025, 0.95))
    good = inside & (depth >= dl - 5) & (depth <= dh + 5)
    metric_scale = z / focal
    x, y, w, h = record["box"]
    center = np.array([x + (w - 1) / 2, y + (h - 1) / 2], dtype=np.float32)
    surface = subsample(good, 512)
    sx, sy = surface.T.astype(int)
    offsets = depth[sy, sx] - z

    # The outer band is several physical millimetres wide. Give each exterior
    # sample the depth of its nearest supported foreground surface.
    edge_gap, indices = ndi.distance_transform_edt(~good, return_indices=True)
    outside = ((edge_gap >= 3 / metric_scale.mean()) & (edge_gap <= 11 / metric_scale.mean()) & ~mask)
    background = subsample(outside, 384)
    ox, oy = background.T.astype(int)
    nearest_y, nearest_x = indices[:, oy, ox]
    background_offsets = depth[nearest_y, nearest_x] - z
    corners = np.array([[x, y], [x + w - 1, y], [x, y + h - 1], [x + w - 1, y + h - 1]], dtype=np.float32)
    dx, dy, strength = color_gradients(record, directory)
    interior = ndi.binary_erosion(raw_mask, iterations=3)
    sharp = strength >= max(4., np.quantile(strength[interior], 0.78)) if interior.any() else interior
    details = subsample(interior & sharp, 132)
    gx, gy = details.T.astype(int)
    rim = subsample(mask & ~ndi.binary_erosion(mask), 160)
    rx, ry = rim.T.astype(int)
    signed_distance = ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
    rim_y, rim_x = np.gradient(ndi.gaussian_filter(signed_distance, 1.2))
    normals = -np.column_stack((rim_x[ry, rx], rim_y[ry, rx]))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    rim_metric = (rim - center) * metric_scale
    rim_inside = rim_metric - 2.5 * normals
    rim_outside = rim_metric + 3.5 * normals
    sample_inside = np.rint(rim_inside / metric_scale + center).astype(int)
    probe_x = np.clip(sample_inside[:, 0], 0, depth.shape[1] - 1)
    probe_y = np.clip(sample_inside[:, 1], 0, depth.shape[0] - 1)
    rim_a, rim_b = indices[:, probe_y, probe_x]
    rim_depth = depth[rim_a, rim_b] - z
    return DepthTemplate(
        record["id"], (surface - center) * metric_scale, offsets,
        (background - center) * metric_scale, background_offsets,
        (corners - center) * metric_scale,
        np.linspace(0, len(surface) - 1, min(56, len(surface)), dtype=int),
        np.linspace(0, len(background) - 1, min(56, len(background)), dtype=int),
        (details - center) * metric_scale, np.column_stack((dx[gy, gx], dy[gy, gx])),
        np.sqrt(strength[gy, gx]),
        np.linspace(0, len(details) - 1, min(40, len(details)), dtype=int),
        rim_inside, rim_outside, rim_depth,
        np.linspace(0, len(rim) - 1, min(48, len(rim)), dtype=int),
    )


def make_bank(filename):
    rows, directory = read_manifest(filename)
    templates = [template for entry in rows if (template := make_template(entry, directory)) is not None]
    if not templates:
        raise ValueError(f"No usable references in {filename}")
    return templates


def project(points, angle, focal):
    sine, cosine = np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float32)
    return (points @ rotation) * focal


def sample(depth, points, centers, anchors):
    coords = np.rint(centers[:, None, :] + points[None, :, :] / anchors[:, None, None]).astype(int)
    xx, yy = coords[:, :, 0], coords[:, :, 1]
    in_frame = (xx >= 0) & (xx < depth.shape[1]) & (yy >= 0) & (yy < depth.shape[0])
    return np.where(in_frame, depth[np.clip(yy, 0, depth.shape[0] - 1), np.clip(xx, 0, depth.shape[1] - 1)], 0)


def box_iou(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    overlap = max(0., min(lx + lw, rx + rw) - max(lx, rx)) * max(0., min(ly + lh, ry + rh) - max(ly, ry))
    return overlap / max(1., lw * lh + rw * rh - overlap)


def box_at(corners, center, anchor):
    bounds = corners / anchor + center
    top_left = bounds.min(axis=0)
    size = bounds.max(axis=0) - top_left + 1
    return [*top_left.tolist(), *size.tolist()]


def sample_poses(center, anchor, offsets, depths):
    centers = np.array([center + np.array((x, y), dtype=np.float32) for x in offsets for y in offsets for _ in depths])
    anchors = np.tile(np.array([anchor + d for d in depths], dtype=np.float32), len(offsets) ** 2)
    return centers, anchors


def physical_scores(depth, gradients, focal, ref, center, anchor, angle, dense):
    ii = slice(None) if dense else ref.coarse_indices
    jj = slice(None) if dense else ref.background_indices
    seen = sample(depth, project(ref.points[ii], angle, focal), center, anchor)
    expected = anchor[:, None] + ref.depths[ii][None, :]
    delta = seen - expected
    surface = ((seen > 0) * np.exp(-np.abs(delta) / DEPTH_SCALE)).mean(axis=1)
    support = ((seen > 0) & (np.abs(delta) <= 10.)).mean(axis=1)
    observed_outer = sample(depth, project(ref.background_points[jj], angle, focal), center, anchor)
    outer_depth = anchor[:, None] + ref.background_depths[jj][None, :]
    # Paired outside samples distinguish a true contour from a cut-out patch
    # within a geometrically larger object.
    spill = ((observed_outer > 0) & (np.abs(observed_outer - outer_depth) <= 9.)).mean(axis=1)
    kk = slice(None) if dense else ref.detail_indices
    details = ref.detail_points[kk]
    if len(details) > 5:
        projected = project(details, angle, focal)
        dx, dy, strengths = gradients
        directions = project(ref.detail_directions[kk], angle, np.ones(2))
        gx = sample(dx, projected, center, anchor)
        gy = sample(dy, projected, center, anchor)
        strength = sample(strengths, projected, center, anchor)
        alignment = gx * directions[None, :, 0] + gy * directions[None, :, 1]
        weights = ref.detail_weights[kk]
        texture = np.average(alignment ** 4 * strength / (strength + 5.), axis=1, weights=weights)
    else:
        texture = np.zeros_like(surface)
    rr = slice(None) if dense else ref.rim_indices
    rim_in = sample(depth, project(ref.rim_inside[rr], angle, focal), center, anchor)
    rim_out = sample(depth, project(ref.rim_outside[rr], angle, focal), center, anchor)
    rim_depth = anchor[:, None] + ref.rim_depths[rr][None, :]
    good_inside = (rim_in > 0) * np.exp(-np.abs(rim_in - rim_depth) / 11.)
    # A raised object's outline has nearer material on the inside.
    # A concave recess flips this sign. Missing returns contribute weaker evidence.
    outward_step = np.where(rim_out > 0, np.clip((rim_out - rim_in - 4.) / 14., -1., 1.), 0.65)
    boundary = np.mean(good_inside * outward_step, axis=1)
    quality = 0.34 * surface + 0.13 * support + 0.11 * (1 - spill) + 0.17 * texture + 0.25 * boundary
    return quality, surface, support, spill, texture, boundary


def best_fit(depth, gradients, focal, bank, detection):
    x, y, width, height = detection["box"]
    initial_center = np.array([x + (width - 1) / 2, y + (height - 1) / 2], dtype=np.float32)
    initial_depth = detection["centerDepthMm"]
    centers, depths = sample_poses(initial_center, initial_depth, COARSE_XY, COARSE_DEPTH)
    coarse = []
    for ref in bank:
        angles = list(COARSE_ANGLES)
        if ref.name == detection.get("reference") and detection.get("angle") is not None:
            angles.append(detection["angle"])
        for angle in dict.fromkeys(angles):
            corners = project(ref.corners, angle, focal)
            # Require a plausible view at the same location, while permitting a
            # larger competing shape to explain foreground beyond the proposal.
            if box_iou(detection["box"], box_at(corners, initial_center, initial_depth)) < .22:
                continue
            scores = physical_scores(depth, gradients, focal, ref, centers, depths, angle, False)[0]
            k = int(np.argmax(scores))
            coarse.append((float(scores[k]), ref, angle, centers[k], float(depths[k])))

    best = None
    for _, ref, angle, center, anchor in sorted(coarse, key=lambda row: -row[0])[:REFINE_PER_BANK]:
        around_center, around_depth = sample_poses(center, anchor, FINE_XY, FINE_DEPTH)
        for fine_angle in (angle - 10, angle, angle + 10):
            q, surface, support, spill, texture, boundary = physical_scores(
                depth, gradients, focal, ref, around_center, around_depth, fine_angle, True
            )
            ix = int(np.argmax(q))
            candidate = {
                "fit": float(q[ix]), "surface": float(surface[ix]),
                "support": float(support[ix]), "spill": float(spill[ix]),
                "texture": float(texture[ix]),
                "boundary": float(boundary[ix]),
                "reference": ref.name, "angle": int(fine_angle) % 360,
                "center": around_center[ix].tolist(), "centerDepthMm": float(around_depth[ix]),
            }
            if best is None or candidate["fit"] > best["fit"]:
                best = candidate
    return best


def evidence(depth, gradients, focal, positive, negative, detection):
    fit = best_fit(depth, gradients, focal, positive, detection)
    competitor = best_fit(depth, gradients, focal, negative, detection)
    if fit is None:
        return {"score": -1., "targetFit": None, "competitorFit": competitor}
    competitor_score = competitor["fit"] if competitor else 0.
    score = fit["fit"] - 0.5 * max(0., competitor_score - fit["fit"])
    return {"score": round(score, 6), "targetFit": fit, "competitorFit": competitor}


def ranked_evidence(record, directory, proposals, positive, negative):
    depth, focal = read_depth(record, directory)
    gradients = color_gradients(record, directory)
    rows = []
    for proposal in proposals:
        verdict = evidence(depth, gradients, focal, positive, negative, proposal)
        rows.append({
            **proposal, "proposalScore": proposal["score"],
            "score": verdict.pop("score"), "verification": verdict,
        })
    return sorted(rows, key=lambda item: (-item["score"], *item["box"]))


def calibration_score(rows, gate):
    hits = false = 0
    for frame, detections in rows:
        unmatched = list(frame["boxes"])
        for row in detections:
            if row["score"] < gate:
                continue
            overlaps = [box_iou(row["box"], box) for box in unmatched]
            if overlaps and max(overlaps) >= OVERLAP:
                del unmatched[int(np.argmax(overlaps))]
                hits += 1
            else:
                false += 1
    return hits - false, -false, hits, gate


def calibrate(rows):
    levels = sorted({row["score"] for _, detections in rows for row in detections} | {1.01})
    gate = max(levels, key=lambda value: calibration_score(rows, value))
    utility, negative_false, hits, _ = calibration_score(rows, gate)
    return gate, {"hits": hits, "falseAlarms": -negative_false, "utility": utility}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "competitors", "calibration", "scenes", "proposals", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--calibration-proposals", type=Path)
    parser.add_argument("--threshold", type=float, help="Use a frozen development cutoff.")
    parser.add_argument("--localizer", type=Path,
                        default=Path(__file__).resolve().parent.parent / "sphere-tless-rgbd.py")
    args = parser.parse_args()
    positive, negative = make_bank(args.training), make_bank(args.competitors)
    references, training_dir = read_manifest(args.training)
    calibration, calibration_dir = read_manifest(args.calibration)
    if args.calibration_proposals:
        candidates = {f["id"]: f["detections"] for f in json.loads(args.calibration_proposals.read_text())["frames"]}
    else:
        rgbd = read_baseline(args.localizer)
        bank = [ref for row in references if (ref := rgbd.build_template(row, training_dir)) is not None]
        candidates = {record["id"]: rgbd.predict(rgbd.load_frame(record, calibration_dir), bank)
                      for record in calibration}
    calibration_rows = [(row, ranked_evidence(row, calibration_dir, candidates[row["id"]], positive, negative))
                        for row in calibration]
    isolated_threshold, outcome = calibrate(calibration_rows)
    threshold = isolated_threshold if args.threshold is None else args.threshold
    scenes, scene_dir = read_manifest(args.scenes)
    proposals = {f["id"]: f["detections"] for f in json.loads(args.proposals.read_text())["frames"]}
    output = {
        "detector": "dense-depth-and-edge-polarity", "threshold": threshold,
        "calibration": {"threshold": isolated_threshold, **outcome},
        "calibrationFrames": [{"id": row["id"], "boxes": row["boxes"], "detections": detections}
                              for row, detections in calibration_rows],
        "frames": [],
    }
    for row in scenes:
        found = ranked_evidence(row, scene_dir, proposals[row["id"]], positive, negative)
        for candidate in found:
            candidate["accepted"] = candidate["score"] >= threshold
        output["frames"].append({"id": row["id"], "detections": found})
        print(row["id"], "accepted", [p["score"] for p in found if p["accepted"]], flush=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Calibration", threshold, outcome, flush=True)


if __name__ == "__main__":
    main()
