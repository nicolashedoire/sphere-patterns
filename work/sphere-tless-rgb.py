"""Locate a referenced part in a full image using multiview gradient templates."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


VIEW_SIZE = 56
NORMALIZED_SIDE = 44
IMAGE_RATIO = 2
ORIENTATIONS = 12
SIZES = (40, 48, 58, 70, 84, 101, 121, 145)
ANGLES = tuple(range(0, 360, 30))
COARSE_STRIDE = 4
COARSE_PEAKS = 48
FEATURES = (26, 14)
ACCEPT_SCORE = 0.66
GRADIENT_SATURATION = 0.053


@dataclass(frozen=True)
class Template:
    identity: str
    x: np.ndarray
    y: np.ndarray
    angles: np.ndarray
    weights: np.ndarray
    hull_x: np.ndarray
    hull_y: np.ndarray


@dataclass(frozen=True)
class Variant:
    template: Template
    side: float
    rotation: int
    x: np.ndarray
    y: np.ndarray
    orientation: np.ndarray
    bounds: tuple


def luminance(path, factor=1):
    with Image.open(path) as source:
        image = source.convert("L")
        if factor != 1:
            image = image.resize((image.width // factor, image.height // factor), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255


def gradients(image, sigma):
    blurred = ndimage.gaussian_filter(image, sigma)
    dy, dx = np.gradient(blurred)
    return np.hypot(dx, dy), np.mod(np.arctan2(dy, dx), np.pi)


def spread_features(selector, strength, limit, spacing):
    y, x = np.nonzero(selector)
    if not len(x):
        return y, x
    priority = strength[y, x].astype(np.float64)
    priority /= max(float(priority.max()), 1e-6)
    priority = 0.5 + 0.5 * priority
    distances = np.full(len(x), np.inf)
    indexes = [int(np.argmax(priority))]
    while len(indexes) < limit:
        i = indexes[-1]
        distances = np.minimum(distances, (y - y[i]) ** 2 + (x - x[i]) ** 2)
        score = distances * priority
        score[distances < spacing ** 2] = -1
        i = int(np.argmax(score))
        if score[i] < 0:
            break
        indexes.append(i)
    return y[indexes], x[indexes]


def load_template(entry, directory):
    image = luminance(directory / entry["rgb"])
    with Image.open(directory / entry["mask"]) as source:
        mask = np.asarray(source.convert("L")) > 0
    if mask.shape != image.shape or not mask.any():
        raise ValueError(f"Training image and nonempty mask must agree for {entry['id']}")
    y, x = np.nonzero(mask)
    middle = (VIEW_SIZE - 1) / 2
    camera_center = np.array([(y.min() + y.max()) / 2, (x.min() + x.max()) / 2])
    scaling = (max(np.ptp(x), np.ptp(y)) + 1) / NORMALIZED_SIDE
    into_view = dict(matrix=np.eye(2) * scaling, offset=camera_center - scaling * middle,
                     output_shape=(VIEW_SIZE, VIEW_SIZE))
    thumbnail = ndimage.affine_transform(image, order=1, **into_view)
    binary = ndimage.affine_transform(mask, order=0, **into_view)
    rim = binary & ~ndimage.binary_erosion(binary)
    rim_strength, rim_angles = gradients(binary.astype(np.float32), 0.9)
    rim_y, rim_x = spread_features(rim, rim_strength, FEATURES[0], 2.2)
    strength, angles = gradients(thumbnail, 0.7)
    inner = ndimage.binary_erosion(binary, iterations=3)
    inner &= strength == ndimage.maximum_filter(strength, size=3)
    inner &= strength > 0.019
    inner_y, inner_x = spread_features(inner, strength, FEATURES[1], 3.0)
    weights = np.ones(len(rim_y) + len(inner_y), dtype=np.float32)
    if len(inner_y):
        weights[:len(rim_y)] *= 0.67 / len(rim_y)
        weights[len(rim_y):] *= 0.33 / len(inner_y)
    else:
        weights /= len(rim_y)
    return Template(str(entry["id"]), np.r_[rim_x, inner_x] - middle,
                    np.r_[rim_y, inner_y] - middle,
                    np.r_[rim_angles[rim_y, rim_x], angles[inner_y, inner_x]],
                    weights, rim_x - middle, rim_y - middle)


def make_variant(template, side, rotation):
    angle = np.deg2rad(rotation)
    sin, cos = np.sin(angle), np.cos(angle)
    scale = side / (NORMALIZED_SIDE * IMAGE_RATIO)
    dx = np.rint((template.x * cos - template.y * sin) * scale).astype(np.int32)
    dy = np.rint((template.x * sin + template.y * cos) * scale).astype(np.int32)
    ox = (template.hull_x * cos - template.hull_y * sin) * scale
    oy = (template.hull_x * sin + template.hull_y * cos) * scale
    orientations = np.rint((template.angles + angle) * ORIENTATIONS / np.pi).astype(np.int32) % ORIENTATIONS
    return Variant(template, side, rotation, dx, dy, orientations,
                   (float(ox.min() - 0.5), float(oy.min() - 0.5), float(ox.max() + 0.5), float(oy.max() + 0.5)))


def response_maps(image):
    strength, direction = gradients(image, 0.7)
    amplitude = strength / (strength + GRADIENT_SATURATION)
    coarse, fine = [], []
    for bin_index in range(ORIENTATIONS):
        expected = bin_index * np.pi / ORIENTATIONS
        response = (amplitude * np.cos(direction - expected) ** 4).astype(np.float32)
        close = ndimage.maximum_filter(response, size=3)
        fine.append(0.2 * response + 0.8 * close)
        coarse.append(ndimage.maximum_filter(close, size=3))
    return np.asarray(coarse), np.asarray(fine)


def coarse_hypotheses(responses, bank):
    _, height, width = responses.shape
    margin = max(max(np.abs(item.x).max(), np.abs(item.y).max()) for item in bank)
    padded = np.pad(responses, ((0, 0), (margin, margin), (margin, margin)))
    rows = (height + COARSE_STRIDE - 1) // COARSE_STRIDE
    columns = (width + COARSE_STRIDE - 1) // COARSE_STRIDE
    best = np.full((rows, columns), -1, dtype=np.float32)
    best_variant = np.zeros(best.shape, dtype=np.int32)
    for index, item in enumerate(bank):
        scores = np.zeros(best.shape, dtype=np.float32)
        for dx, dy, channel, weight in zip(item.x, item.y, item.orientation, item.template.weights):
            y, x = margin + dy, margin + dx
            scores += weight * padded[channel, y:y + height:COARSE_STRIDE, x:x + width:COARSE_STRIDE]
        better = scores > best
        best[better] = scores[better]
        best_variant[better] = index
    peaks = []
    for _ in range(COARSE_PEAKS):
        r, c = np.unravel_index(best.argmax(), best.shape)
        if best[r, c] < 0:
            break
        peaks.append((c * COARSE_STRIDE, r * COARSE_STRIDE, bank[best_variant[r, c]]))
        best[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] = -1
    return peaks


def box_at(x, y, bounds, image_width, image_height):
    left, top, right, bottom = bounds
    left = max(0, int(np.floor((x + left) * IMAGE_RATIO)))
    top = max(0, int(np.floor((y + top) * IMAGE_RATIO)))
    right = min(image_width, int(np.ceil((x + right) * IMAGE_RATIO)))
    bottom = min(image_height, int(np.ceil((y + bottom) * IMAGE_RATIO)))
    return [left, top, right - left, bottom - top]


def refine(responses, peak, image_width, image_height):
    x, y, variant = peak
    _, height, width = responses.shape
    delta = np.arange(-3, 4, dtype=np.int32)
    dx, dy = np.meshgrid(delta, delta)
    x_centers, y_centers = x + dx.reshape(-1), y + dy.reshape(-1)
    best = None
    for side_factor in (0.93, 1, 1.07):
        for rotation_delta in (-12, 0, 12):
            item = make_variant(variant.template, variant.side * side_factor, (variant.rotation + rotation_delta) % 360)
            xx, yy = x_centers[:, None] + item.x, y_centers[:, None] + item.y
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            samples = responses[item.orientation, yy.clip(0, height - 1), xx.clip(0, width - 1)]
            scores = (valid * samples * item.template.weights).sum(axis=1)
            ix = int(scores.argmax())
            value = float(scores[ix])
            if best is None or value > best["score"]:
                best = {
                    "bbox": box_at(x_centers[ix], y_centers[ix], item.bounds, image_width, image_height),
                    "score": value,
                    "reference": item.template.identity,
                    "rotationDegrees": item.rotation,
                    "nominalSide": round(item.side, 3),
                }
    return best


def iou(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    overlap = max(0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0, min(ly + lh, ry + rh) - max(ly, ry))
    return overlap / (lw * lh + rw * rh - overlap) if overlap else 0.0


def locate(entry, directory, bank):
    photo = directory / entry["rgb"]
    with Image.open(photo) as source:
        width, height = source.size
    image = luminance(photo, IMAGE_RATIO)
    coarse, fine = response_maps(image)
    peaks = coarse_hypotheses(coarse, bank)
    candidates = sorted((refine(fine, peak, width, height) for peak in peaks), key=lambda row: row["score"], reverse=True)
    kept = []
    for item in candidates:
        if not any(iou(item["bbox"], previous["bbox"]) > 0.32 for previous in kept):
            item["accepted"] = item["score"] >= ACCEPT_SCORE
            item["score"] = round(item["score"], 6)
            kept.append(item)
            if len(kept) == 5:
                break
    return {"id": entry["id"], **kept[0], "alternatives": kept[1:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-limit", type=int)
    args = parser.parse_args()
    training = json.loads(args.training.read_text())[:args.training_limit]
    templates = [load_template(entry, args.training.parent) for entry in training]
    if not templates:
        raise ValueError("At least one training view is required")
    bank = [make_variant(template, side, angle) for template in templates for side in SIZES for angle in ANGLES]
    rows = []
    for entry in json.loads(args.scenes.read_text()):
        row = locate(entry, args.scenes.parent, bank)
        rows.append(row)
        print(f"{entry['id']}: {row['score']:.3f}  {row['bbox']}  {row['reference']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
