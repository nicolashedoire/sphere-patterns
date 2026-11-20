"""Generate full-frame coarse poses from frozen DINOv2 patch descriptors.

Angles are clockwise in image coordinates. A candidate's x/y locates the
center of its isolated reference box, and scale is relative to that box.
The scene images are tiled uniformly. Test boxes are evaluation metadata.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
MODEL = 'facebook/dinov2-small'
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
WEIGHT_SHA256 = 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'
PATCH = 14
ANGLES = tuple(range(0, 360, 30))
SCALES = {'real': (.60, .78, 1., 1.25, 1.55), 'control': (.70, .85, 1., 1.15)}
STRIDE = {'real': 9, 'control': 14}
IMAGE_FIELDS = ('id', 'split', 'domain', 'width', 'height', 'preview', 'truth', 'decoys', 'scene', 'band')
REPLAY_EXAMPLES = {'12-000260', '20-000407'}


class DenseEncoder:
    """Extract the normalized patch grid, without a detector or a learned head."""

    def __init__(self, weights=None, device='auto'):
        from transformers import Dinov2Model

        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cpu':
            torch.set_num_threads(min(torch.get_num_threads(), 4))
        if weights is None:
            self.model = Dinov2Model.from_pretrained(MODEL, revision=REVISION, use_safetensors=True)
        else:
            weights = Path(weights)
            with (weights / 'model.safetensors').open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != WEIGHT_SHA256:
                    raise ValueError('Local DINOv2 weight digest does not match the pinned checkpoint.')
            self.model = Dinov2Model.from_pretrained(weights, local_files_only=True, use_safetensors=True)
        self.model = self.model.eval().to(self.device)
        self.dimension = self.model.config.hidden_size
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def batch_features(self, images, size=None):
        if size is None:
            height, width = images[0].shape[:2]
            size = tuple(max(PATCH, round(length / PATCH) * PATCH) for length in (width, height))
        if any(length < PATCH or length % PATCH for length in size):
            raise ValueError('DINOv2 input dimensions must be positive multiples of 14.')
        resized = np.stack([cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)[..., ::-1] for image in images])
        pixels = torch.from_numpy(resized).to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
        tokens = self.model(pixel_values=(pixels / 255 - self.mean) / self.std).last_hidden_state[:, 1:]
        normalized = torch.nn.functional.normalize(tokens, dim=-1).cpu().numpy()
        return normalized.reshape(len(images), size[1] // PATCH, size[0] // PATCH, -1)

    def features(self, image, size=None):
        return self.batch_features([image], size)[0]


class GradientEncoder:
    """Dense unsigned gradients on the same patch grid as the neural encoder."""

    device = torch.device('cpu')
    dimension = 32

    def __init__(self):
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    def batch_features(self, images, size):
        rows, columns = size[1] // PATCH, size[0] // PATCH
        bin_count = 8
        bin_numbers = np.arange(bin_count, dtype=np.int32)
        grid = []
        for image in images:
            gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size).astype(np.float32) / 255
            dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=.125)
            dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=.125)
            magnitude = np.hypot(dx, dy)[..., None]
            orientation = (np.arctan2(dy, dx) % np.pi) * bin_count / np.pi
            first = np.floor(orientation).astype(np.int32)
            fraction = (orientation - first).astype(np.float32)[..., None]
            lower = first[..., None] % bin_count == bin_numbers
            upper = (first[..., None] + 1) % bin_count == bin_numbers
            votes = magnitude * ((1 - fraction) * lower + fraction * upper)
            cells = votes.reshape(rows, 2, PATCH // 2, columns, 2, PATCH // 2, bin_count)
            descriptor = cells.sum(axis=(2, 5)).transpose(0, 2, 1, 3, 4).reshape(rows, columns, self.dimension)
            descriptor = np.sqrt(descriptor)
            norms = np.linalg.norm(descriptor, axis=-1, keepdims=True)
            grid.append(descriptor / np.maximum(norms, 1e-8))
        return np.stack(grid)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write {path}')


def tiled_features(encoder, image, stride):
    """Uniform overlapping tiles give all pixels a path into the descriptor map."""
    height, width = image.shape[:2]
    rows, cols = ((length + stride - 1) // stride for length in (height, width))
    tile = min(20, rows, cols)

    def starts(length):
        return sorted(set([*range(0, length - tile, max(1, tile - 4)), length - tile]))

    padded = cv2.copyMakeBorder(image, 0, rows * stride - height, 0, cols * stride - width, cv2.BORDER_REPLICATE)
    centers = [(y, x) for y in starts(rows) for x in starts(cols)]
    weights = np.maximum(.15, np.hanning(tile)).astype(np.float32)
    weight = weights[:, None] * weights[None, :]
    result = np.zeros((rows, cols, encoder.dimension), dtype=np.float32)
    counts = np.zeros((rows, cols), dtype=np.float32)
    for begin in range(0, len(centers), 10):
        batch = centers[begin:begin + 10]
        images = [padded[y * stride:(y + tile) * stride, x * stride:(x + tile) * stride] for y, x in batch]
        descriptors = encoder.batch_features(images, (tile * PATCH, tile * PATCH))
        for (y, x), grid in zip(batch, descriptors):
            result[y:y + tile, x:x + tile] += grid * weight[..., None]
            counts[y:y + tile, x:x + tile] += weight
    if not np.all(counts > 0):
        raise AssertionError('The tile grid did not cover the full frame.')
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8)


def foreground(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    count, labels, regions, _ = cv2.connectedComponentsWithStats((gray >= 24).astype(np.uint8))
    if count < 2:
        raise ValueError('An isolated reference must have a foreground component.')
    index = int(np.argmax(regions[1:, cv2.CC_STAT_AREA])) + 1
    return regions[index, :4].tolist(), labels == index


def rotation(angle):
    radians = np.deg2rad(angle)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def controls_from_baseline(fixtures, baseline, output):
    frames = []
    for index, frame in enumerate(baseline):
        if frame['domain'] != 'control':
            continue
        target = next(iter(frame['truth']), None)

        def pose(record):
            return record['center'], record['angle'], record['scale']

        case = frame['id']
        seed = 740216 + int(case.rsplit('-', 1)[1]) if case.startswith('control-validation-') else 100 + index
        image, _, _ = fixtures['make_control'](seed, pose(target) if target else None,
                                               [pose(decoy) for decoy in frame['decoys']],
                                               bool(target and target['visibility'] < 1))
        path = output / f'{case}.png'
        write_image(path, image)
        frames.append({**{key: frame[key] for key in IMAGE_FIELDS if key in frame}, 'image': str(path)})
    return frames


def real_frames(panel, baseline, previews):
    by_id = {frame['id']: frame for frame in baseline if frame['domain'] == 'real'}
    for split in ('development', 'heldout'):
        for record in read_json(panel / f'{split}.json'):
            frame = by_id[record['id']]
            image = panel / record['rgb']
            row = {**{key: frame[key] for key in IMAGE_FIELDS if key in frame}, 'image': str(image)}
            if frame['id'] in REPLAY_EXAMPLES:
                name = f"neural-{frame['id']}.jpg"
                thumbnail = cv2.resize(read_image(image), (540, 405), interpolation=cv2.INTER_AREA)
                write_image(previews / name, thumbnail)
                row['preview'] = name
            yield row


def render_reference(image, mask, box, linear, side, stride):
    """Align the isolated training view to the paired reference's box center."""
    center = np.array(box[:2]) + np.array(box[2:]) / 2
    transform = np.column_stack([linear, side / 2 - linear @ center])
    backdrop = (103, 103, 103)
    clean = np.where(mask[..., None], image, np.uint8(backdrop))
    rotated = cv2.warpAffine(clean, transform, (side, side), borderValue=backdrop)
    cropped_mask = cv2.warpAffine(mask.astype(np.uint8), transform, (side, side), flags=cv2.INTER_NEAREST)
    mask_grid = cv2.resize(cropped_mask.astype(np.float32), (side // stride, side // stride), interpolation=cv2.INTER_AREA)
    return rotated, mask_grid


def anchors(mask, box, domain, forced):
    """Use separated visible reference patches and three repeatable cue sites."""
    x, y, width, height = box
    center = np.array([x + width / 2, y + height / 2])
    fractions = (-.30, 0., .30)
    if domain == 'control':
        points = np.array([(65, 65), (145, 106), (210, 71)])
        cues = [{'id': name, 'xy': point.tolist()} for name, point in zip(('bore', 'groove', 'shoulder'), points)]
    else:
        offsets = [(f * width, 0) if width > height else (0, f * height) for f in fractions]
        points = center + offsets
        cues = [{'id': f'region-{index}', 'xy': point.tolist()} for index, point in enumerate(points, 1)]
    offsets = np.array([(fx * width, fy * height) for fx in fractions for fy in fractions])
    nominal = center + offsets
    valid_y, valid_x = np.where(mask)
    valid = np.column_stack([valid_x, valid_y])
    selected = []
    for sample in nominal:
        near = np.argmin(np.sum((valid - sample) ** 2, axis=1))
        nearest = valid[near]
        if not selected or min(np.linalg.norm(nearest - other) for other in selected) > forced:
            selected.append(nearest)
    if domain == 'real':
        for cue in cues:
            nearest = valid[np.argmin(np.sum((valid - cue['xy']) ** 2, axis=1))]
            cue['xy'] = nearest.tolist()
    return np.array(selected) - center, cues


def build_reference(encoder, part, key, paths, domain):
    images = [read_image(path) for path in paths]
    masks = [foreground(image) for image in images]
    box, first_mask = masks[0]
    stride = STRIDE[domain]
    offsets, cues = anchors(first_mask, box, domain, stride * .65)
    side = (int(np.ceil(np.hypot(*box[2:]) / stride)) + 4) * stride
    size = (side // stride * PATCH,) * 2
    variants = []
    for angle in ANGLES:
        rotated = rotation(angle)
        pairs = [render_reference(image, mask, dimensions, rotated @ np.diag(np.array(box[2:]) / dimensions[2:]),
                                  side, stride) for image, (dimensions, mask) in zip(images, masks)]
        descriptors = encoder.batch_features([img for img, _ in pairs], size)
        coords = offsets @ rotated.T
        index = np.floor((coords + side / 2) / stride).astype(np.int32)
        x, y = index.T
        inside = np.minimum(pairs[0][1][y, x], pairs[1][1][y, x]) > .25
        if not np.any(inside):
            continue
        a, b = descriptors[:, y[inside], x[inside]]
        c = a + b
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
        site = (index[inside] + .5) * stride - side / 2
        quality = np.clip((np.sum(a * b, axis=1) + 1) / 2, .15, 1)
        center = np.array(box[:2]) + np.array(box[2:]) / 2
        cue_offsets = (np.array([cue['xy'] for cue in cues]) - center) @ rotated.T
        cue_indices = np.floor((cue_offsets + side / 2) / stride).astype(np.int32)
        cue_a, cue_b = descriptors[:, cue_indices[:, 1], cue_indices[:, 0]]
        cue_bank = cue_a + cue_b
        cue_bank /= np.maximum(np.linalg.norm(cue_bank, axis=1, keepdims=True), 1e-8)
        variants.append({'angle': angle, 'offsets': site, 'features': c, 'quality': quality,
                         'cueOffsets': cue_offsets, 'cueBank': cue_bank})
    return {'part': part, 'key': key, 'image': str(paths[0]), 'pairedImage': str(paths[1]),
            'width': images[0].shape[1], 'height': images[0].shape[0], 'box': box, 'cues': cues,
            'domain': domain, 'variants': variants}


def offset_map(image, x, y):
    """At each candidate center, return the response from its shifted cue."""
    height, width = image.shape
    target = np.full_like(image, -1.)
    if abs(x) >= width or abs(y) >= height:
        return target
    x0, x1, y0, y1 = max(0, -x), min(width, width - x), max(0, -y), min(height, height - y)
    target[y0:y1, x0:x1] = image[y0 + y:y1 + y, x0 + x:x1 + x]
    return target


def iou(a, b):
    x, y, w, h = a
    u, v, s, t = b
    overlap = max(0, min(x + w, u + s) - max(x, u)) * max(0, min(y + h, v + t) - max(y, v))
    return overlap / max(1e-6, w * h + s * t - overlap)


def proposals(encoder, grid, references, domain, limit):
    rows, cols, size = grid.shape
    tensors = [variant['features'] for ref in references for variant in ref['variants']]
    bank = torch.from_numpy(np.concatenate(tensors)).to(encoder.device)
    query = torch.from_numpy(grid.reshape(-1, size)).to(encoder.device)
    responses = (bank @ query.T).cpu().numpy().reshape(-1, rows, cols)
    stride = STRIDE[domain]
    candidates = []
    cursor = 0
    for reference in references:
        _, _, width, height = reference['box']
        for variant in reference['variants']:
            count = len(variant['features'])
            local = responses[cursor:cursor + count]
            cursor += count
            weights = variant['quality'] / np.sum(variant['quality'])
            angle = variant['angle']
            matrix = rotation(angle)
            for scale in SCALES[domain]:
                offsets = np.rint(variant['offsets'] * scale / stride).astype(np.int32)
                shifted = np.stack([offset_map(response, x, y) for response, (x, y) in zip(local, offsets)])
                score = np.sum(shifted * weights[:, None, None], axis=0) - .10 * shifted.std(axis=0)
                peaks = (score == cv2.dilate(score, np.ones((5, 5), dtype=np.uint8))).ravel()
                indices = np.flatnonzero(peaks)
                best = indices[np.argsort(-score.ravel()[indices], kind='stable')[:limit]]
                box_w, box_h = scale * (np.abs(matrix) @ np.array([width, height]))
                for flat_index in best:
                    row, col = divmod(flat_index, cols)
                    x, y = (col + .5) * stride, (row + .5) * stride
                    box = [x - box_w / 2, y - box_h / 2, box_w, box_h]
                    candidates.append({'part': reference['part'], 'key': reference['key'], 'box': box,
                                       'x': x, 'y': y, 'angle': angle, 'scale': scale,
                                       'score': float(np.clip((score[row, col] + 1) / 2, 0, 1)),
                                       'similarity': float(score[row, col])})
    finalists = []
    for candidate in sorted(candidates, key=lambda entry: -entry['score']):
        if any(iou(candidate['box'], other['box']) > .38 for other in finalists):
            continue
        finalists.append(candidate)
        if len(finalists) == limit:
            break
    by_key = {reference['key']: reference for reference in references}
    for candidate in finalists:
        reference = by_key[candidate['key']]
        variant = next(variant for variant in reference['variants'] if variant['angle'] == candidate['angle'])
        points = np.array([candidate['x'], candidate['y']]) + candidate['scale'] * variant['cueOffsets']
        checks = []
        for cue, xy, description in zip(reference['cues'], points, variant['cueBank']):
            x, y = np.floor(xy / stride).astype(int)
            strength = float(np.clip((np.dot(grid[y, x], description) + 1) / 2, 0, 1)) if 0 <= y < rows and 0 <= x < cols else 0.
            checks.append({'id': cue['id'], 'xy': xy.tolist(), 'strength': strength})
        candidate['cues'] = checks
    return finalists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--weights', type=Path, help='Directory containing pinned Hugging Face weights and config.')
    parser.add_argument('--descriptor', choices=('dinov2', 'gradient'), default='dinov2')
    parser.add_argument('--baseline', type=Path, default=HERE.parent / 'outputs/sphere-pose-consensus-measurements.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True, help='Write generated frames and reusable descriptor grids here.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--split', action='append', help='Optional split name. Repeat to select multiple splits.')
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be at least one')
    if args.descriptor == 'dinov2' and not args.weights:
        parser.error('--weights is required for the DINOv2 descriptor')
    cv2.setNumThreads(1)
    neural = args.descriptor == 'dinov2'
    encoder = DenseEncoder(args.weights, args.device) if neural else GradientEncoder()
    fixtures = runpy.run_path(str(HERE / 'sphere-pose-inputs.py'))
    baseline = read_json(args.baseline)['frames']
    args.cache.mkdir(parents=True, exist_ok=True)
    frame_dir = args.cache / 'frames'
    previews = args.cache / 'previews'
    descriptor_id = f'dinov2-{REVISION[:8]}' if neural else 'gradient-v1'
    map_dir = args.cache / f'{descriptor_id}-{encoder.device.type}'
    frame_dir.mkdir(exist_ok=True)
    previews.mkdir(exist_ok=True)
    map_dir.mkdir(exist_ok=True)
    references = []
    if not args.split or any(split.startswith('control-') for split in args.split):
        paths = [frame_dir / f'control-training-{index}.png' for index in (11, 22)]
        for path, index in zip(paths, (11, 22)):
            write_image(path, fixtures['draw_fixture'](index)[0])
        references.append(build_reference(encoder, 101, 'mechanical-plate', paths, 'control'))
    if not args.split or any(not split.startswith('control-') for split in args.split):
        for a, b in fixtures['TRAIN_PAIRS']:
            paths = [args.references / f'{index:06d}.png' for index in (a, b)]
            references.append(build_reference(encoder, 1, f'tless-{a:06d}', paths, 'real'))
    by_domain = {domain: [reference for reference in references if reference['domain'] == domain] for domain in STRIDE}
    frames = []
    if by_domain['control']:
        frames.extend(controls_from_baseline(fixtures, baseline, frame_dir))
    if by_domain['real']:
        frames.extend(real_frames(args.panel, baseline, previews))
    rows = []
    for frame in frames:
        if args.split and frame['split'] not in args.split:
            continue
        domain = frame['domain']
        image_path = Path(frame['image'])
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]
        cache = map_dir / f"{frame['id']}-{STRIDE[domain]}-{image_hash}.npy"
        if cache.is_file():
            grid = np.load(cache, allow_pickle=False)
        else:
            grid = tiled_features(encoder, read_image(image_path), STRIDE[domain])
            np.save(cache, grid, allow_pickle=False)
        candidates = proposals(encoder, grid, by_domain[domain], domain, args.limit)
        rows.append({**frame, 'candidates': candidates})
        print(f"{frame['split']} {frame['id']}: {len(candidates)} proposals", flush=True)
    report = {'model': MODEL if neural else 'dense-gradient', 'revision': REVISION if neural else None,
              'safetensorsSha256': WEIGHT_SHA256 if neural else None, 'device': encoder.device.type,
              'descriptorStride': STRIDE, 'angles': ANGLES, 'scales': SCALES,
              'references': [{key: value for key, value in ref.items() if key != 'variants'} for ref in references],
              'frames': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
