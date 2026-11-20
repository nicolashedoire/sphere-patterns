"""Propose complete parts by cutting a calibrated depth image at physical heights."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from verify import box_iou, read_manifest


def geometry(rec, folder):
    with Image.open(folder / rec["depth"]) as image:
        depth = np.asarray(image, dtype=np.float32) * rec["camera"]["depthScale"]
    fx, _, cx, _, fy, cy, *_ = rec["camera"]["K"]
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    xyz = np.stack([(xx - cx) * depth / fx, (yy - cy) * depth / fy, depth], axis=-1)
    t = np.asarray(rec["camera"]["worldToCameraT"], dtype=np.float32)
    rot = np.asarray(rec["camera"]["worldToCameraR"], dtype=np.float32).reshape(3, 3)
    world = (xyz - t) @ rot
    return depth, xyz, world, rot, np.array([fx, fy], dtype=np.float32)


def training_sizes(path):
    rows, folder = read_manifest(path)
    dimensions = []
    for row in rows:
        with Image.open(folder / row["depth"]) as image:
            z = np.asarray(image, dtype=np.float32) * row["camera"]["depthScale"]
        with Image.open(folder / row["mask"]) as source:
            y, x = np.nonzero(np.asarray(source) > 0)
        center_depth = np.median(z[y, x][z[y, x] > 0])
        _, _, width, height = row["box"]
        focal = np.array([row["camera"]["K"][0], row["camera"]["K"][4]])
        p = np.column_stack((x, y)) * center_depth / focal
        _, _, axis = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)
        minor, major = sorted(np.ptp(p @ axis.T, axis=0))
        dimensions.append([minor, major, len(x) * center_depth ** 2 / np.prod(focal)])
    return np.array(dimensions)


def eligible_points(depth, world):
    return ((depth > 260) & (depth < 1400) &
            (np.max(np.abs(world[:, :, :2]), axis=-1) <= 185) &
            (world[:, :, 2] >= 6) & (world[:, :, 2] <= 165))


def repaired_surface(depth, allowed):
    small = ndi.binary_closing(allowed, iterations=2)
    missing = small & ~allowed & (depth == 0)
    return allowed | missing


def get_regions(mask, origin, recipe):
    labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    for label, span in enumerate(ndi.find_objects(labels), 1):
        if span is None:
            continue
        sy, sx = span
        height, width = sy.stop - sy.start, sx.stop - sx.start
        if not 15 <= min(width, height) or max(width, height) > 158:
            continue
        region = labels[span] == label
        if region.sum() < 110:
            continue
        yield region, (sx.start + origin[0], sy.start + origin[1], width, height), recipe


def geometric_candidates(depth, world, focal, sizes):
    foreground = eligible_points(depth, world)
    yy, xx = np.nonzero(foreground)
    if not len(xx):
        return []
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    inside = np.s_[y0:y1, x0:x1]
    d = depth[inside]
    h = world[inside][:, :, 2]
    fg = foreground[inside]
    repaired = repaired_surface(d, fg)
    origins = (x0, y0)
    cuts = []
    for height in range(8, 151, 5):
        mask = repaired & ((h >= height) | (d == 0))
        cuts.append((ndi.binary_closing(mask, iterations=2), "world-height"))
    for z in range(int(np.min(d[fg]) // 10) * 10, int(np.max(d[fg])) + 1, 9):
        for span in (18, 29):
            mask = repaired & ((np.abs(d - z) < span) | (d == 0))
            cuts.append((ndi.binary_closing(mask, iterations=2), "camera-depth"))

    # The sign and location of the rim are retained by removing only the
    # farther sample adjacent to each valid discontinuity.
    jumps = np.zeros_like(d)
    for along in (0, 1):
        near_side = [slice(None)] * 2
        far_side = near_side.copy()
        near_side[along] = slice(None, -2)
        far_side[along] = slice(2, None)
        near_side, far_side = tuple(near_side), tuple(far_side)
        valid = fg[near_side] & fg[far_side]
        gap = np.where(valid, d[far_side] - d[near_side], 0)
        jumps[far_side] = np.maximum(jumps[far_side], gap)
        jumps[near_side] = np.maximum(jumps[near_side], -gap)
    for mm in (7, 11, 16):
        cuts.append((ndi.binary_opening(repaired & (jumps < mm), iterations=2), "depth-rim"))

    shortlist = []
    for mask, recipe in cuts:
        for region, box, source in get_regions(mask, origins, recipe):
            x, y, w, height = box
            ddy, ddx = np.nonzero(region)
            pixels = depth[ddy + y, ddx + x]
            pixels = pixels[pixels > 0]
            if len(pixels) < 85:
                continue
            z = float(np.median(pixels))
            scale = z / focal
            dims = np.array([w, height]) * scale
            if dims.max() > 90 or dims.min() < 15 or dims.min() > 60:
                continue
            samples = np.column_stack((ddx, ddy))[::4] * scale
            _, axes = np.linalg.eigh(np.cov(samples.T))
            minor, major = sorted(np.ptp(samples @ axes, axis=0))
            area = len(samples) * 4 * scale.prod()
            delta = (np.array([minor, major, area]) - sizes) / np.array([7., 14., 470.])
            closeness = np.exp(-.5 * np.min(np.square(delta).sum(axis=1)))
            if closeness < .15:
                continue
            outlined = ndi.binary_erosion(region)
            contour_y, contour_x = np.nonzero(region & ~outlined)
            jj = jumps[contour_y + y - y0, contour_x + x - x0]
            rim = np.average(np.minimum(jj, 22.) / 22.)
            quality = .87 * closeness + .13 * rim
            shortlist.append({
                "box": [int(n) for n in box], "score": float(quality), "centerDepthMm": z,
                "reference": source, "angle": None, "minorMm": float(minor), "majorMm": float(major),
                "areaMm2": float(area), "sizeFit": float(closeness), "positiveRim": float(rim),
            })
    return shortlist


def dedupe(candidates, limit, iou=.42):
    chosen = []
    for item in sorted(candidates, key=lambda row: (-row["score"], *row["box"])):
        if any(box_iou(item["box"], other["box"]) >= iou for other in chosen):
            continue
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "scenes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", default=15, type=int)
    args = parser.parse_args()
    scenes, folder = read_manifest(args.scenes)
    sizes = training_sizes(args.training)
    frames = []
    for rec in scenes:
        depth, _, world, _, focal = geometry(rec, folder)
        pool = geometric_candidates(depth, world, focal, sizes)
        selected = dedupe(pool, args.limit)
        frames.append({"id": rec["id"], "detections": selected})
        print(rec["id"], len(pool), len(selected), flush=True)
    args.output.write_text(json.dumps({"detector": "depth-components", "frames": frames}, indent=2) + "\n")


if __name__ == "__main__":
    main()
