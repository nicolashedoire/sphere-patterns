#!/usr/bin/env python3
"""Verify RGB-D proposals with competing metric CAD depth maps."""
import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import numpy as np
from numpy.ctypeslib import ndpointer
from PIL import Image
from scipy import ndimage as ndi

CAD_URL = "https://huggingface.co/datasets/bop-benchmark/tless/resolve/5fd309a04476a842d93abfb584fba9ee7caecdf1/tless_models.zip"
MODEL_IDS = (1, 2, 3, 4, 13, 14, 15, 16, 24)
TILTS = np.arange(0, 181, 15)
ROLLS = np.arange(0, 360, 20)
GRID = 61
SPACING = 1.8
OFFSETS = np.array(list(itertools.product((-5.4, 0., 5.4), repeat=2)))
DENSE_OFFSETS = np.array(list(itertools.product((-2.7, 0., 2.7), repeat=2)))
DEPTH_OFFSETS = np.arange(-16., 17., 4., dtype=np.float32)
SCAN_STEP = 10


@dataclass(frozen=True)
class Views:
    part: int
    depth: np.ndarray
    inner: np.ndarray
    outer: np.ndarray
    values: np.ndarray
    neighbor: np.ndarray
    angles: np.ndarray
    medians: np.ndarray


def load_mesh(archive, part):
    stream = io.BytesIO(archive.read(f"models_cad/obj_{part:06d}.ply"))
    fields = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Incomplete PLY header")
        items = line.decode("ascii").strip().split()
        if items == ["end_header"]:
            break
        if items and items[0] == "format" and items[1] != "ascii":
            raise ValueError("Expected ASCII PLY")
        if items and items[0] == "element":
            fields[items[1]] = int(items[2])
    vertices = np.loadtxt(stream, max_rows=fields["vertex"], usecols=(0, 1, 2), dtype=np.float32)
    faces = np.loadtxt(stream, max_rows=fields["face"], dtype=np.int32)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected triangular faces")
    return vertices, np.ascontiguousarray(faces[:, 1:])


def rotations(tilt, roll):
    x, z = np.radians((tilt, roll))
    c, s, h, j = np.cos(x), np.sin(x), np.cos(z), np.sin(z)
    return np.array([[h, -j, 0.], [j, h, 0.], [0., 0., 1.]]) @ np.array(
        [[1., 0., 0.], [0., c, -s], [0., s, c]]
    )


def rasterizer(location):
    library = location / "raster.so"
    source = location / "raster.c"
    if not library.exists() or library.stat().st_mtime_ns < source.stat().st_mtime_ns:
        subprocess.run(["cc", "-O3", "-fno-math-errno", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    return ctypes.CDLL(str(library))


def render_function(location):
    render = rasterizer(location).rasterize
    render.argtypes = (
        ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"), ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    )
    render.restype = None
    return render


def rendered_views(path, model_ids=MODEL_IDS):
    location = Path(__file__).resolve().parent
    mesh_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    source_hash = hashlib.sha256((location / "raster.c").read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
    cache_path = location / f"bank-{mesh_hash}-{source_hash}-{'-'.join(map(str, model_ids))}.npz"
    keys = ("depth", "inner", "outer", "values", "neighbor", "angles", "medians")
    if cache_path.exists():
        with np.load(cache_path) as data:
            return [Views(part, *(data[f"{part}-{key}"] for key in keys)) for part in model_ids]
    render = render_function(location)
    bank = []
    with ZipFile(path) as archive:
        for part in model_ids:
            vertices, faces = load_mesh(archive, part)
            views, inside, outside, expected, edge_depth, angles, medians = [], [], [], [], [], [], []
            for tilt in TILTS:
                for roll in ROLLS if 0 < tilt < 180 else (0,):
                    cloud = np.ascontiguousarray(vertices @ rotations(tilt, roll).T, dtype=np.float32)
                    z = np.full((GRID, GRID), np.inf, dtype=np.float32)
                    render(cloud, faces, len(faces), GRID, SPACING, z)
                    valid = np.isfinite(z)
                    median = np.median(z[valid])
                    z -= median
                    compact = np.flatnonzero(valid)
                    compact = compact[np.linspace(0, len(compact) - 1, 96, dtype=int)]
                    band = ndi.binary_dilation(valid, iterations=3) & ~valid
                    ring = np.flatnonzero(band)
                    ring = ring[np.linspace(0, len(ring) - 1, 48, dtype=int)]
                    nearest = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
                    relative = z[tuple(nearest[:, band])]
                    relative = relative[np.linspace(0, len(relative) - 1, 48, dtype=int)]
                    views.append(z)
                    inside.append(compact)
                    outside.append(ring)
                    expected.append(z.ravel()[compact])
                    edge_depth.append(relative)
                    angles.append((float(tilt), float(roll)))
                    medians.append(median)
            bank.append(Views(part, *(np.asarray(value) for value in (
                views, inside, outside, expected, edge_depth, angles, medians,
            ))))
            print(f"Rendered CAD part {part}: {len(views)} orientations", flush=True)
    np.savez_compressed(cache_path, **{
        f"{view.part}-{key}": getattr(view, key) for view in bank for key in keys
    })
    return bank


def load_depth(item, directory):
    image = np.asarray(Image.open(directory / item["depth"]), dtype=np.float32)
    return image * item["camera"]["depthScale"]


def read_manifest(directory, filename):
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def known_camera(item):
    k = item["camera"]["K"]
    return float(k[0]), float(k[4])


def projected_patches(depth, box, fx, fy, z, translations=OFFSETS):
    x, y, w, h = box
    physical = (np.arange(GRID, dtype=np.float32) - (GRID - 1) / 2) * SPACING
    u = np.rint(x + (w - 1) / 2 + (translations[:, 0, None] + physical) * fx / z).astype(int)
    v = np.rint(y + (h - 1) / 2 + (translations[:, 1, None] + physical) * fy / z).astype(int)
    legal = (u[:, None, :] >= 0) & (u[:, None, :] < depth.shape[1]) & (v[:, :, None] >= 0) & (v[:, :, None] < depth.shape[0])
    sampled = depth[v[:, :, None].clip(0, depth.shape[0] - 1), u[:, None, :].clip(0, depth.shape[1] - 1)]
    return np.where(legal, sampled, 0).reshape(len(translations), GRID * GRID)


def score(difference, outside, missing):
    match = np.exp(-np.abs(difference) / 4) * ~missing
    contradiction = ((difference > 9) & ~missing).mean(axis=-1)
    continuation = ((np.abs(outside) < 7)).mean(axis=-1)
    # Fewer support points and a flat surface beyond the silhouette hurt the fit.
    return match.mean(axis=-1) - .27 * contradiction - .32 * continuation


def prior_depth(depth, detection):
    if "centerDepthMm" in detection:
        return float(detection["centerDepthMm"])
    box = detection.get("box", detection.get("bbox"))
    x, y, w, h = box
    px, py = round(x + w / 2), round(y + h / 2)
    center = depth[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
    valid = center[center > 0]
    return float(np.median(valid)) if len(valid) else 0.0


def box_from_mask(mask, center, focal, z, shift):
    rows, columns = np.nonzero(mask.reshape(GRID, GRID))
    samples = np.column_stack((columns, rows))
    projections = center + ((samples - (GRID - 1) / 2) * SPACING + shift) * focal / z
    padding = 0.5 * SPACING * focal / z
    origin = np.floor(projections.min(axis=0) - padding).astype(int)
    end = np.ceil(projections.max(axis=0) + padding).astype(int)
    return [*origin.tolist(), *(end - origin).tolist()]


def box_iou(left, right):
    start_l = np.asarray(left[:2])
    start_r = np.asarray(right[:2])
    end_l = start_l + left[2:]
    end_r = start_r + right[2:]
    shared = np.maximum(np.minimum(end_l, end_r) - np.maximum(start_l, start_r), 0).prod()
    return shared / (np.prod(left[2:]) + np.prod(right[2:]) - shared)


def scan_frame(depth, focal, view, max_proposals):
    """Generate independent hypotheses from depth over the whole camera frame."""
    scan = rasterizer(Path(__file__).resolve().parent).scan_depth
    array = ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    scan.argtypes = (array, array, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
                     array, array, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, array)
    scan.restype = None

    def points(indices, relative):
        rows, cols = np.divmod(indices, GRID)
        xy = (np.stack((cols, rows), axis=-1) - (GRID - 1) / 2) * SPACING
        return np.ascontiguousarray(np.concatenate((xy, relative[..., None]), axis=-1), dtype=np.float32)

    orientations = np.arange(len(view.depth))
    inner = points(view.inner[orientations, ::3], view.values[orientations, ::3])
    outer = points(view.outer[orientations, ::4], view.neighbor[orientations, ::4])
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    h, w = depth.shape
    grid_y = np.arange(SCAN_STEP // 2, h, SCAN_STEP, dtype=np.int32)
    grid_x = np.arange(SCAN_STEP // 2, w, SCAN_STEP, dtype=np.int32)
    coarse = np.zeros((len(grid_y), len(grid_x), 4), dtype=np.float32)
    scan(depth, ndi.median_filter(depth, size=5), w, h, *focal, inner, outer,
         len(orientations), inner.shape[1], outer.shape[1], SCAN_STEP, coarse)

    peaks = ndi.maximum_filter(coarse[:, :, 0], size=3) == coarse[:, :, 0]
    scores = np.where(peaks, coarse[:, :, 0], 0)
    candidates = []
    for index in np.argsort(-scores, axis=None):
        y, x = np.unravel_index(index, scores.shape)
        fit, orientation, z, _ = coarse[y, x]
        if fit < .18:
            break
        mask = np.isfinite(view.depth[orientations[int(orientation)]])
        seed = np.array([grid_x[x], grid_y[y]], dtype=np.float32)
        box = box_from_mask(mask, seed, np.asarray(focal), z, np.zeros(2))
        if any(box_iou(box, row["box"]) >= .35 for row in candidates):
            continue
        candidates.append({"box": box, "score": round(float(fit), 6), "centerDepthMm": float(z),
                           "scanCenter": seed.tolist(), "reference": "cad-depth-scan"})
        if len(candidates) == max_proposals:
            break
    return candidates


def fit_view(view, patches, z):
    interior = np.take(patches, view.inner, axis=1)
    exterior = np.take(patches, view.outer, axis=1)
    best = None
    for shift in DEPTH_OFFSETS:
        delta = interior - view.values - (z + shift)
        outside = exterior - view.neighbor - (z + shift)
        scores = score(delta, outside, interior <= 0)
        index = np.unravel_index(scores.argmax(), scores.shape)
        if best is None or scores[index] > best[0]:
            best = float(scores[index]), index, float(shift)
    coarse, (offset, rotation), shift = best
    return coarse, offset, rotation, shift


def verify(depth, camera, detection, bank):
    z = prior_depth(depth, detection)
    box = detection.get("box", detection.get("bbox"))
    if not z or not box:
        return None
    if "scanCenter" in detection:
        x, y = detection["scanCenter"]
        box = [x, y, 1, 1]
    else:
        x, y, width, height = box
        x, y = x + (width - 1) / 2, y + (height - 1) / 2
    center = np.array([x, y], dtype=np.float32)
    focal = np.asarray(camera, dtype=np.float32)
    fx, fy = camera
    patches = projected_patches(depth, box, fx, fy, z)
    ranked = []
    for view in bank:
        coarse, offset, orientation, shift = fit_view(view, patches, z)
        fit_depth = view.depth[orientation].ravel()
        inner = np.isfinite(fit_depth)
        exterior = ndi.binary_dilation(inner.reshape(GRID, GRID), iterations=3) & ~inner.reshape(GRID, GRID)
        nearest = ndi.distance_transform_edt(~inner.reshape(GRID, GRID), return_distances=False, return_indices=True)
        ring_depth = view.depth[orientation][tuple(nearest[:, exterior])]
        translations = DENSE_OFFSETS + OFFSETS[offset]
        local_patches = projected_patches(depth, box, fx, fy, z, translations)
        metrics = []
        for refinement in np.arange(shift - 3, shift + 4, 1):
            delta = local_patches[:, inner] - fit_depth[inner] - (z + refinement)
            outside = local_patches[:, exterior.ravel()] - ring_depth - (z + refinement)
            fits = score(delta, outside, local_patches[:, inner] <= 0)
            peak = int(fits.argmax())
            metrics.append((float(fits[peak]), peak, float(refinement)))
        value, peak, relief = max(metrics, key=lambda metric: metric[0])
        observed = local_patches[peak]
        residual = observed[inner] - fit_depth[inner] - (z + relief)
        outside = observed[exterior.ravel()] - ring_depth - (z + relief)
        visible = inner.copy()
        visible[inner] = (np.abs(residual) <= 7) & (observed[inner] > 0)
        # Suppress isolated depth samples before taking the visible boundary.
        visible = visible.reshape(GRID, GRID)
        support = ndi.convolve(visible.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) >= 3
        visible &= support
        amodal = box_from_mask(inner, center, focal, z, translations[peak])
        supported = box_from_mask(visible, center, focal, z, translations[peak]) if visible.sum() >= 15 else amodal
        ranked.append({
            "part": view.part, "metricFit": round(value, 6), "coarseFit": round(coarse, 6),
            "consistentAt4Mm": round(float((np.abs(residual) <= 4).mean()), 6),
            "contradictedAt9Mm": round(float((residual > 9).mean()), 6),
            "occludedOrMissingAt9Mm": round(float((residual < -9).mean()), 6),
            "continuationAt7Mm": round(float((np.abs(outside) < 7).mean()), 6),
            "tilt": float(view.angles[orientation, 0]), "roll": float(view.angles[orientation, 1]),
            "centerShiftMm": list(translations[peak]), "surfaceDepthMm": round(z + relief, 3),
            "orientationIndex": int(orientation),
            "projectedBox": amodal, "visibleBox": supported,
        })
    target = ranked[0]
    rival = max(ranked[1:], key=lambda item: item["metricFit"])
    return {"target": target, "competitor": rival, "metricMargin": round(target["metricFit"] - rival["metricFit"], 6),
            "competition": sorted(ranked, key=lambda item: -item["metricFit"])}


def verify_frames(manifest, candidates, directory, bank, scan_proposals):
    records = {item["id"]: item for item in manifest}
    output = []
    for item in candidates:
        frame_id = item["id"]
        if frame_id not in records:
            continue
        source = records[frame_id]
        depth = load_depth(source, directory)
        camera = known_camera(source)
        detections = []
        proposals = list(item.get("detections", []))
        if scan_proposals:
            proposals += scan_frame(depth, camera, bank[0], scan_proposals)
        for row in proposals:
            result = verify(depth, camera, row, bank)
            if result is None:
                continue
            cleaned = {key: value for key, value in row.items() if key != "supportMask"}
            detections.append({**cleaned, "cad": result})
        output.append({"id": frame_id, "detections": detections})
        best = max(detections, default=None, key=lambda x: x["cad"]["metricMargin"])
        print(frame_id, "fit/margin", (best["cad"]["target"]["metricFit"], best["cad"]["metricMargin"]) if best else None, flush=True)
    return output


def main():
    location = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--manifest", default="scenes.json")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--models", type=Path, default=location / "models.zip")
    parser.add_argument("--model-ids", nargs="+", type=int, default=MODEL_IDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-calibration-box", action="store_true")
    parser.add_argument("--scan-proposals", type=int, default=0)
    args = parser.parse_args()
    manifest = read_manifest(args.frames, args.manifest)
    if args.use_calibration_box:
        if args.manifest != "calibration.json":
            parser.error("Explicit calibration boxes are allowed only for calibration.json.")
        candidates = [
            {"id": item["id"], "detections": [{"box": box, "score": 1.0}]}
            for item in manifest for box in item.get("boxes", [])
        ]
    elif args.predictions:
        raw = json.loads(args.predictions.read_text(encoding="utf-8"))
        candidates = raw["frames"] if isinstance(raw, dict) else raw
    elif args.scan_proposals:
        candidates = [{"id": item["id"]} for item in manifest]
    else:
        parser.error("Supply proposals with --predictions or generate them with --scan-proposals.")
    bank = rendered_views(args.models, tuple(args.model_ids))
    frames = verify_frames(manifest, candidates, args.frames, bank, args.scan_proposals)
    args.output.write_text(json.dumps({"modelSource": CAD_URL, "frames": frames}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
