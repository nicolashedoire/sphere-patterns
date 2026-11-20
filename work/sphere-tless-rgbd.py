"""Localize a reference part in complete RGB-D images with a calibrated template bank."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


ANGLE_STEP = 45
GRID_STEP = 8
POSE_POOL = 8
DEPTH_TOLERANCE_MM = 12.0


@dataclass
class Frame:
    gray: np.ndarray
    depth: np.ndarray
    fx: float
    fy: float


@dataclass
class Template:
    name: str
    interior: np.ndarray
    relative_depth: np.ndarray
    contour: np.ndarray
    surface_edges: np.ndarray
    exterior: np.ndarray
    box: np.ndarray
    rgb_weight: float


def load_manifest(filename):
    path = Path(filename).resolve()
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def load_frame(record, base):
    gray = np.asarray(Image.open(base / record["rgb"]).convert("L"), dtype=np.float32)
    depth = np.asarray(Image.open(base / record["depth"]), dtype=np.float32)
    calibrate = record["camera"]
    return Frame(gray, depth * float(calibrate["depthScale"]), calibrate["K"][0], calibrate["K"][4])


def sample_pixels(mask, count):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return np.zeros((0, 2), dtype=np.float32)
    indices = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    return np.column_stack((cols[indices], rows[indices])).astype(np.float32)


def build_template(record, base):
    frame = load_frame(record, base)
    mask = np.asarray(Image.open(base / record["mask"]).convert("L")) > 0
    mask = ndi.binary_fill_holes(mask)
    valid = frame.depth > 0
    depths = frame.depth[mask & valid]
    if len(depths) < 40:
        return None
    z = np.median(depths)

    x, y, width, height = record["box"]
    center = np.array([x + (width - 1) / 2, y + (height - 1) / 2], dtype=np.float32)
    pixel_scale = np.array([z / frame.fx, z / frame.fy], dtype=np.float32)
    to_mm = lambda points: (points - center) * pixel_scale

    inside = sample_pixels(mask & valid & (np.abs(frame.depth - z) < 65), 40)
    if len(inside) < 18:
        return None
    rr, cc = inside[:, 1].astype(int), inside[:, 0].astype(int)
    contour = sample_pixels(mask & ~ndi.binary_erosion(mask), 36)
    exterior = sample_pixels(ndi.binary_dilation(mask, iterations=5) &
                             ~ndi.binary_dilation(mask, iterations=2), 24)
    interior_gray = ndi.gaussian_filter(frame.gray, 0.8)
    magnitude = np.hypot(*np.gradient(interior_gray))
    inner = ndi.binary_erosion(mask, iterations=2)
    strong = inner & (magnitude >= np.quantile(magnitude[inner], .91)) if inner.any() else inner
    edges = sample_pixels(strong & (magnitude > 3.0), 20)

    return Template(
        record["id"], to_mm(inside), frame.depth[rr, cc] - z, to_mm(contour),
        to_mm(edges), to_mm(exterior),
        to_mm(np.array([[x, y], [x + width - 1, y], [x, y + height - 1],
                        [x + width - 1, y + height - 1]], dtype=np.float32)),
        0.10 if len(edges) >= 6 else 0.0
    )


def scene_features(frame):
    depth = frame.depth
    valid = depth > 0
    edge = np.zeros(depth.shape, dtype=bool)
    edge[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (np.abs(depth[:, 1:] - depth[:, :-1]) >= 16)
    edge[1:, :] |= valid[1:, :] & valid[:-1, :] & (np.abs(depth[1:, :] - depth[:-1, :]) >= 16)
    gray = ndi.gaussian_filter(frame.gray, .8)
    magnitude = np.hypot(*np.gradient(gray))
    rgb_edge = magnitude >= max(4, np.quantile(magnitude, .86))
    depth_dist = ndi.distance_transform_edt(~edge)
    rgb_dist = ndi.distance_transform_edt(~rgb_edge)
    return depth.ravel(), np.exp(-depth_dist / 2.4).astype(np.float32).ravel(), (
        np.exp(-rgb_dist / 2.0).astype(np.float32).ravel()
    )


def rotated(points, angle, focal):
    radians = np.radians(angle)
    c, s = np.cos(radians), np.sin(radians)
    return ((points @ np.array([[c, s], [-s, c]], dtype=np.float32)) * focal).astype(np.float32)


def gather(array, points, centers, scales, width, height):
    cols = np.rint(centers[:, 0, None] + scales[:, None] * points[None, :, 0]).astype(np.int32)
    rows = np.rint(centers[:, 1, None] + scales[:, None] * points[None, :, 1]).astype(np.int32)
    legal = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    indices = np.clip(rows, 0, height - 1) * width + np.clip(cols, 0, width - 1)
    return np.where(legal, array[indices], 0)


def measurements(frame, feature, template, angle, centers, depths):
    height, width = frame.depth.shape
    flat_depth, depth_near, rgb_near = feature
    inverse_depth = 1 / depths
    focal = np.array([frame.fx, frame.fy], dtype=np.float32)

    def projected(array, points):
        return gather(array, rotated(points, angle, focal), centers, inverse_depth, width, height)

    observed = projected(flat_depth, template.interior)
    delta = observed - depths[:, None] - template.relative_depth[None, :]
    match = np.exp(-np.abs(delta) / DEPTH_TOLERANCE_MM) * (observed > 0)
    surface = match.mean(axis=1)
    boundary = projected(depth_near, template.contour).mean(axis=1)
    nearby = projected(flat_depth, template.exterior)
    # The surrounding samples penalize a larger continuous surface at the
    # estimated part depth. Unknown background depth contributes no penalty.
    outside = ((nearby > 0) & (np.abs(nearby - depths[:, None]) < 19)).mean(axis=1)
    rgb = (projected(rgb_near, template.surface_edges).mean(axis=1)
           if template.rgb_weight else np.zeros(len(centers), dtype=np.float32))
    return (0.64 - template.rgb_weight) * surface + 0.28 * boundary + .08 * (
        1 - outside
    ) + template.rgb_weight * rgb


def projected_box(frame, template, angle, center, depth):
    height, width = frame.depth.shape
    focal = np.array([frame.fx, frame.fy], dtype=np.float32)
    corners = rotated(template.box, angle, focal) / depth + center
    x, y = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    right, bottom = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 1, [width, height])
    return [int(x), int(y), int(right - x), int(bottom - y)]


def box_iou(a, b):
    x, y, w, h = a
    q, r, u, v = b
    inter = max(0, min(x + w, q + u) - max(x, q)) * max(0, min(y + h, r + v) - max(y, r))
    return inter / max(w * h + u * v - inter, 1)


def measured_support(frame, box, depth):
    x, y, w, h = box
    side = np.ceil(max(w, h) * 1.25)
    xs = x + (w - side) / 2 + side * (np.arange(25) + .5) / 25
    ys = y + (h - side) / 2 + side * (np.arange(25) + .5) / 25
    xx, yy = np.meshgrid(xs.astype(int), ys.astype(int))
    height, width = frame.depth.shape
    legal = (xx >= 0) & (yy >= 0) & (xx < width) & (yy < height)
    sampled = frame.depth[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)]
    return (legal & (sampled > 0) & (np.abs(sampled - depth) <= 25)).astype(int).tolist()


def predict(frame, templates):
    h, w = frame.depth.shape
    features = scene_features(frame)
    grid_x, grid_y = np.meshgrid(np.arange(12, w - 12, GRID_STEP), np.arange(12, h - 12, GRID_STEP))
    centers = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    candidate_depths = ndi.median_filter(frame.depth, size=5)
    depth = candidate_depths[grid_y, grid_x].ravel()
    valid = (depth >= 250) & (depth <= 1800)
    centers, depth = centers[valid], depth[valid]
    if not len(centers):
        return []

    best_coarse = []
    for template in templates:
        for angle in range(0, 360, ANGLE_STEP):
            scores = measurements(frame, features, template, angle, centers, depth)
            shortlist = np.argpartition(scores, -min(POSE_POOL, len(scores)))[-POSE_POOL:]
            for idx in shortlist:
                best_coarse.append((float(scores[idx]), template, angle, centers[idx], float(depth[idx])))

    best_coarse.sort(key=lambda item: (-item[0], item[1].name, item[2], *item[3]))
    proposals = []
    for _, template, angle, center, initial_depth in best_coarse[:50]:
        # Refine shortlisted positions and in-plane angle around the same
        # measured surface. The test frame supplies every refinement input.
        offsets = np.array([(x, y) for y in (-4, -2, 0, 2, 4) for x in (-4, -2, 0, 2, 4)],
                           dtype=np.float32)
        locations = offsets + center
        depths = candidate_depths[locations[:, 1].astype(int), locations[:, 0].astype(int)]
        depths = np.where((depths > 0) & (np.abs(depths - initial_depth) <= 35), depths, initial_depth)
        for trial in (angle - 15, angle, angle + 15):
            ranked = measurements(frame, features, template, trial, locations, depths)
            idx = np.argmax(ranked)
            box = projected_box(frame, template, trial, locations[idx], depths[idx])
            if min(box[2:]) <= 0:
                continue
            proposals.append({
                "box": box,
                "score": round(float(ranked[idx]), 6),
                "reference": template.name,
                "angle": trial % 360,
                "centerDepthMm": round(float(depths[idx]), 3),
            })

    result = []
    for item in sorted(proposals, key=lambda x: (-x["score"], *x["box"])):
        if all(box_iou(item["box"], previous["box"]) < .35 for previous in result):
            result.append(item)
        if len(result) == 5:
            break
    return result


def calibrate(records, base, templates):
    observations = []
    for record in records:
        detections = predict(load_frame(record, base), templates)
        top = detections[0] if detections else None
        overlap = max((box_iou(top["box"], box) for box in record["boxes"]), default=0) if top else 0
        row = {"id": record["id"], "positive": bool(record["boxes"]), "iou": round(overlap, 6),
               "score": top["score"] if top else 0}
        observations.append(row)
        print(row, flush=True)
    # Tune acceptance on these separate training views only. A wrong box
    # counts as an error even if the frame contains the target.
    def rank(threshold):
        hit = sum(row["iou"] >= .50 and row["score"] >= threshold for row in observations)
        false = sum(row["iou"] < .50 and row["score"] >= threshold for row in observations)
        return hit - false, -false, hit, threshold
    levels = [.35, .90, *(row["score"] for row in observations)]
    return max(levels, key=rank), observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training, train_dir = load_manifest(args.training)
    records, scene_dir = load_manifest(args.scenes)
    templates = [template for item in training if (template := build_template(item, train_dir)) is not None]
    if not templates:
        raise ValueError("No reference templates with sufficient depth support")
    threshold, calibration = .56, None
    if args.calibration:
        examples, calibration_dir = load_manifest(args.calibration)
        threshold, calibration = calibrate(examples, calibration_dir, templates)
    result = []
    for item in records:
        frame = load_frame(item, scene_dir)
        detections = predict(frame, templates)
        for detection in detections:
            detection["supportMask"] = measured_support(frame, detection["box"], detection["centerDepthMm"])
            detection["accepted"] = detection["score"] >= threshold
        result.append({"id": item["id"], "detections": detections})
        print(item["id"], detections[0]["score"] if detections else None, flush=True)
    args.output.write_text(json.dumps({"detector": "depth-profile-templates", "templates": len(templates),
                                       "threshold": threshold, "calibration": calibration,
                                       "frames": result}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
