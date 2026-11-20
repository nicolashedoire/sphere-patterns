"""Prepare unlabeled full-frame RootSIFT points and two-example reference C.

Scene x/y are image pixels. Reference x/y and bounds are pixels relative to
the center of the foreground box in its first training view. Angles are
clockwise degrees in image coordinates, and size is the SIFT keypoint diameter.
Descriptors are nonnegative 128-component vectors normalized to unit L2 norm.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SIZE = (720, 540)
PART = 1
TEMPLATE = 101
TRAIN_PAIRS = ((0, 1), (688, 689), (621, 622), (507, 508), (325, 326), (914, 915),
               (945, 946), (299, 300), (975, 976), (752, 753), (614, 615), (695, 696))
CONTROL_GROUPS = ('Bore', 'Groove', 'Shoulder')
CONTROL_CENTER = np.array([140., 87.])
CONTROL_OUTLINE = np.array([[28, 35], [197, 35], [197, 55], [252, 55],
                            [252, 137], [28, 137]], dtype=np.int32)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def image_at(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


def sift_points(sift, image, mask=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is None:
        return []
    roots = np.sqrt(descriptors / descriptors.sum(axis=1, keepdims=True))
    return [(np.array(p.pt), p.size, p.angle, d) for p, d in zip(keypoints, roots)]


def foreground(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((gray >= 24).astype(np.uint8))
    assert len(stats) > 1, 'The isolated reference must contain a visible object.'
    part = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    box = stats[part, :4].astype(float)
    mask = cv2.dilate((labels == part).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    return box, mask


def point_json(xy, size, angle, descriptor):
    return {'x': round(float(xy[0]), 4), 'y': round(float(xy[1]), 4), 'size': round(float(size), 4),
            'angle': round(float(angle % 360), 4) % 360, 'd': np.round(descriptor.astype(float), 6).tolist()}


def angle_gap(a, b):
    return (b - a + 180) % 360 - 180


def control_group(xy):
    x, y = xy + CONTROL_CENTER
    if np.hypot(x - 65, y - 65) < 25:
        return 0
    if 68 < x < 217 and 90 < y < 120:
        return 1
    if x > 176 and y < 94:
        return 2
    return None


def reference_c(sift, image_a, image_b, part, key, domain, sources):
    box_a, mask_a = foreground(image_a)
    box_b, mask_b = foreground(image_b)
    center_a, center_b = (box[:2] + box[2:] / 2 for box in (box_a, box_b))
    scale = np.sqrt(np.prod(box_a[2:]) / np.prod(box_b[2:]))
    if domain == 'control':
        # The tooling sits on a neutral surface. Keep that background in the
        # local gradient descriptors, while using the silhouette for alignment.
        image_a = np.where(image_a > 0, image_a, 104).astype(np.uint8)
        image_b = np.where(image_b > 0, image_b, 104).astype(np.uint8)
    candidates_a = sift_points(sift, image_a, mask_a)
    candidates_b = sift_points(sift, image_b, mask_b)
    pairs = []
    for i, (xy_a, size_a, angle_a, d_a) in enumerate(candidates_a):
        local_a = xy_a - center_a
        for j, (xy_b, size_b, angle_b, d_b) in enumerate(candidates_b):
            local_b = (xy_b - center_b) * scale
            gap = angle_gap(angle_a, angle_b)
            if abs(gap) > 45 or not .62 <= size_b * scale / size_a <= 1.62:
                continue
            offset = float(np.linalg.norm(local_a - local_b))
            if offset > max(box_a[2:]) * .13:
                continue
            distance = float(np.linalg.norm(d_a - d_b))
            if distance <= .50:
                score = distance + offset / max(box_a[2:])
                pairs.append((score, i, j, (local_a + local_b) / 2, (size_a + size_b * scale) / 2,
                              angle_a + gap / 2, d_a + d_b))

    groups = CONTROL_GROUPS if domain == 'control' else ('Region 1', 'Region 2', 'Region 3')
    primary_axis = int(box_a[3] > box_a[2])
    axis_length = box_a[2 + primary_axis]
    slots = [[], [], []]
    used_a, used_b = set(), set()
    for _, i, j, xy, size, angle, desc in sorted(pairs, key=lambda pair: pair[:3]):
        group = (control_group(xy) if domain == 'control'
                 else min(2, max(0, int((xy[primary_axis] / axis_length + .5) * 3))))
        if group is None or i in used_a or j in used_b or len(slots[group]) >= 18:
            continue
        # Distinct orientations from the same pixels are still one detail.
        if any(np.linalg.norm(xy - existing) < max(1.5, size * .18) for existing, _ in slots[group]):
            continue
        used_a.add(i)
        used_b.add(j)
        point = point_json(xy, size, angle, desc / np.linalg.norm(desc))
        slots[group].append((xy, {**point, 'group': group}))

    points = [point for slot in slots for _, point in slot]
    bounds = [-box_a[2] / 2, -box_a[3] / 2, *box_a[2:]]
    return {'part': part, 'key': key, 'domain': domain, 'points': points, 'bounds': bounds,
            'groups': groups, 'sourcePairs': sources}


def draw_fixture(seed, shifted_hole=False):
    """A milled plate with three spatially separated details, rendered in grayscale."""
    rng = np.random.default_rng(seed)
    image = np.zeros((174, 280), dtype=np.uint8)
    mask = cv2.fillPoly(np.zeros_like(image), [CONTROL_OUTLINE], 255, cv2.LINE_AA)
    y, x = np.mgrid[:174, :280]
    lighting = 163 + rng.integers(-12, 13) + .10 * (x - 140) + .12 * (y - 87)
    image[mask > 0] = np.clip(lighting[mask > 0], 0, 255).astype(np.uint8)
    cv2.polylines(image, [CONTROL_OUTLINE], True, 193, 2, cv2.LINE_AA)
    bore = (112 if shifted_hole else 65, 65)
    cv2.circle(image, bore, 18, 105, -1, cv2.LINE_AA)
    cv2.circle(image, bore, 15, 232, 2, cv2.LINE_AA)
    cv2.circle(image, bore, 12, 53, -1, cv2.LINE_AA)
    cv2.ellipse(image, (bore[0] + 2, bore[1] - 2), (11, 10), 0, 340, 161, 88, 2, cv2.LINE_AA)
    cv2.line(image, (87, 106), (202, 106), 222, 15, cv2.LINE_AA)
    cv2.line(image, (87, 104), (202, 104), 74, 11, cv2.LINE_AA)
    cv2.line(image, (90, 107), (199, 107), 94, 2, cv2.LINE_AA)
    cv2.line(image, (195, 38), (195, 56), 122, 2, cv2.LINE_AA)
    cv2.line(image, (195, 56), (249, 56), 229, 2, cv2.LINE_AA)
    chamfer = np.array([[196, 64], [234, 64], [234, 76], [224, 84], [196, 84]], dtype=np.int32)
    cv2.fillPoly(image, [chamfer], 138, cv2.LINE_AA)
    cv2.polylines(image, [chamfer], True, 213, 2, cv2.LINE_AA)
    cv2.polylines(image, [np.array([[201, 68], [228, 68], [228, 75], [222, 79]], dtype=np.int32)],
                  False, 104, 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), mask


def place_fixture(scene, seed, center, angle, scale, shifted_hole=False):
    part, mask = draw_fixture(seed, shifted_hole)
    matrix = cv2.getRotationMatrix2D(tuple(CONTROL_CENTER), -angle, scale)
    matrix[:, 2] += np.array(center) - CONTROL_CENTER
    dimensions = (scene.shape[1], scene.shape[0])
    warped = cv2.warpAffine(part, matrix, dimensions)
    alpha = cv2.warpAffine(mask, matrix, dimensions).astype(float)[..., None] / 255
    scene[:] = (warped * alpha + scene * (1 - alpha)).astype(np.uint8)
    outline = CONTROL_OUTLINE @ matrix[:, :2].T + matrix[:, 2]
    low, high = outline.min(axis=0), outline.max(axis=0)
    return {'part': TEMPLATE, 'box': [*low.tolist(), *(high - low).tolist()], 'visibility': 1.,
            'center': center, 'angle': angle, 'scale': scale, 'index': 0}


def make_control(seed, target, lures, cover):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:SIZE[1], :SIZE[0]]
    texture = 105 + .028 * (x - 360) + .021 * (y - 270) + rng.normal(0, 1.5, x.shape)
    scene = cv2.cvtColor(np.clip(texture, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for center, radius in (((96, 442), 30), ((520, 466), 22), ((331, 101), 21)):
        cv2.circle(scene, center, radius, (79, 79, 79), -1, cv2.LINE_AA)
        cv2.circle(scene, center, radius - 1, (174, 174, 174), 3, cv2.LINE_AA)
        cv2.circle(scene, center, radius - 10, (102, 102, 102), -1, cv2.LINE_AA)
    decoys = [place_fixture(scene, seed + i + 3, *lure, shifted_hole=True) for i, lure in enumerate(lures)]
    truth = [place_fixture(scene, seed + 2, *target)] if target else []
    if cover:
        center, angle, scale = target
        cover_at = cv2.getRotationMatrix2D(tuple(CONTROL_CENTER), -angle, scale)
        cover_at[:, 2] += np.array(center) - CONTROL_CENTER
        xx, yy = np.rint(cover_at @ [65, 65, 1]).astype(int)
        cv2.rectangle(scene, (xx - 29, yy - 29), (xx + 29, yy + 29), (126, 126, 126), -1, cv2.LINE_AA)
        cv2.rectangle(scene, (xx - 29, yy - 29), (xx + 29, yy + 29), (112, 112, 112), 1, cv2.LINE_AA)
        truth[0]['visibility'] = .84
    return scene, truth, [{key: decoy[key] for key in ('box', 'center', 'angle', 'scale')} for decoy in decoys]


def save_preview(image, directory, name):
    if directory is None:
        return None
    filename = f'{name}.jpg'
    height, width = image.shape[:2]
    fraction = min(1., 540 / width)
    thumb = cv2.resize(image, (round(width * fraction), round(height * fraction)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(directory / filename), thumb, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f'Cannot write preview {directory / filename}')
    return filename


def scene_frame(sift, name, split, domain, image, previews=None):
    frame = {'id': name, 'split': split, 'domain': domain, 'width': image.shape[1], 'height': image.shape[0],
             'points': [point_json(*p) for p in sift_points(sift, image)]}
    preview = save_preview(image, previews, name)
    if preview:
        frame['preview'] = preview
    return frame


def control_inputs(sift, previews):
    a, b = (draw_fixture(seed)[0] for seed in (11, 22))
    reference = reference_c(sift, a, b, TEMPLATE, 'mechanical-plate', 'control', ['control-a', 'control-b'])
    reference['preview'] = save_preview(a, previews, 'control-reference')
    cases = (
        ('control-dev-target', ((233, 235), 29, .85), ((512, 292), -18, .85), False),
        ('control-dev-lure', None, ((418, 298), 17, .86), False),
        ('control-test-target-lure', ((235, 285), 58, .83), ((483, 284), -32, .95), False),
        ('control-test-angle-scale', ((464, 260), 113, .70), ((212, 316), -51, .97), False),
        ('control-test-same-angle', ((220, 248), -27, 1.08), ((518, 287), -27, 1.08), False),
        ('control-test-lure-only', None, ((359, 275), 64, .92), False),
        ('control-test-two-lures', None, ((408, 273), -37, 1.05), False),
        ('control-test-occluded-bore', ((249, 259), 24, .92), ((530, 287), 51, .80), True),
    )
    frames = []
    for seed, (name, target, lure, cover) in enumerate(cases, 100):
        lures = [lure] if lure else []
        if name == 'control-test-two-lures':
            lures.append(((147, 311), 61, .75))
        image, truth, decoys = make_control(seed, target, lures, cover)
        split = 'control-dev' if name.startswith('control-dev-') else 'control-test'
        frames.append({**scene_frame(sift, name, split, 'control', image, previews), 'truth': truth, 'decoys': decoys})
    return reference, frames


def validation_controls(sift, previews):
    """Generate a fixed sample of unseen poses without consulting recognition scores."""
    rng = np.random.default_rng(740216)
    kinds = rng.permutation(['intact'] * 12 + ['covered'] * 4 + ['lure'] * 4 + ['two-lures'] * 4)
    seen = set()
    frames = []

    def pose(side):
        x = int(rng.integers(174, 205) if side == 0 else rng.integers(516, 547))
        y = int(rng.integers(163, 378))
        return (x, y), int(rng.integers(-179, 181)), round(float(rng.uniform(.65, 1.19)), 2)

    for index, kind in enumerate(kinds, 1):
        side = int(rng.integers(0, 2))
        target = pose(side) if kind in ('intact', 'covered') else None
        lures = [pose(1 - side)]
        if kind == 'two-lures':
            lures.append(pose(side))
        image, truth, decoys = make_control(740216 + index, target, lures, kind == 'covered')
        show_example = (kind not in seen and kind != 'two-lures') or index == 9
        out = previews if show_example else None
        seen.add(kind)
        frame = scene_frame(sift, f'control-validation-{index:02d}', 'control-validation', 'control', image, out)
        frames.append({**frame, 'truth': truth, 'decoys': decoys})
    return frames


def real_inputs(sift, panel, references, pairs, previews):
    learned = []
    for a_id, b_id in pairs:
        a, b = (image_at(references / f'{frame:06d}.png') for frame in (a_id, b_id))
        learned.append(reference_c(sift, a, b, PART, f'tless-{a_id:06d}', 'real', [a_id, b_id]))
    frames = []
    for split in ('development', 'heldout'):
        annotations = read_json(panel / f'{split}-truth.json')['frames']
        truth = {row['id']: row for row in annotations}
        show = {next(row['id'] for row in annotations if row['band'] == band) for band in ('high', 'absent')}
        for row in read_json(panel / f'{split}.json'):
            label = row['id']
            image = image_at(panel / row['rgb'])
            frame = scene_frame(sift, label, split, 'real', image, previews if label in show else None)
            reference = truth[label]
            targets = [dict(t, part=PART, box=t['visibleBox']) for t in reference['targets']]
            frames.append({**frame, 'scene': row['scene'], 'band': reference['band'], 'truth': targets})
    dev = {frame['scene'] for frame in frames if frame['split'] == 'development'}
    heldout = {frame['scene'] for frame in frames if frame['split'] == 'heldout'}
    assert dev.isdisjoint(heldout), 'Development and evaluation must use different physical setups.'
    return learned, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--full-wave', type=Path, help='Optional previous full-wave input manifest with pair IDs.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previews', type=Path)
    parser.add_argument('--dense-real', action='store_true', help='Use the higher-coverage encoder for T-LESS only.')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=600, contrastThreshold=.012, edgeThreshold=12)
    real_budget, real_contrast = (1500, .004) if args.dense_real else (600, .012)
    real_sift = cv2.SIFT_create(nfeatures=real_budget, contrastThreshold=real_contrast, edgeThreshold=12)
    if args.previews:
        args.previews.mkdir(parents=True, exist_ok=True)
    pairs = ([entry['frames'] for entry in read_json(args.full_wave)['references']]
             if args.full_wave else TRAIN_PAIRS)
    control, controls = control_inputs(sift, args.previews)
    references, frames = real_inputs(real_sift, args.panel, args.references, pairs, args.previews)
    validation = validation_controls(sift, args.previews)
    data = {'width': SIZE[0], 'height': SIZE[1], 'references': [control, *references],
            'frames': [*controls, *validation, *frames],
            'metadata': {'descriptor': 'RootSIFT', 'descriptorSize': 128, 'realKeypointBudget': real_budget,
                         'realContrastThreshold': real_contrast, 'controlKeypointBudget': 600,
                         'sizeConvention': 'keypoint diameter in pixels', 'angleConvention': 'clockwise degrees',
                         'realDataset': 'T-LESS object 1, published full-frame replay panel (CC BY 4.0)'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'Wrote {len(data["frames"])} scenes and {len(data["references"])} C templates to {args.output}')
    for template in data['references']:
        counts = [sum(p['group'] == group for p in template['points']) for group in range(3)]
        print(template['key'], counts)


if __name__ == '__main__':
    main()
