"""Read the frozen native-pixel requests and compare them with posed reference patches.

Each response describes only the requested patch. Reference and scene boxes
use image coordinates, and positive angles rotate clockwise. Full PNGs are
decoded once into memory. The pixel budget measures the regions exposed to
the verifier, independently of PNG decoding and model input resizing.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


INPUT_SIZE = (84, 84)
NEUTRAL_COSINE = .62
COSINE_SPAN = .30


def load_encoder():
    source = Path(__file__).with_name('sphere-neural-pose-inputs.py')
    spec = importlib.util.spec_from_file_location('sphere_neural_pose_inputs', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DenseEncoder


def native_patch(image, box):
    if len(box) != 4 or any(not isinstance(value, int) for value in box):
        raise ValueError(f'Request must use four integer native pixel coordinates: {box}')
    x, y, width, height = box
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError(f'Request outside the native image: {box}')
    return image[y:y + height, x:x + width].copy()


def posed_reference(image, box, angle, width, height):
    x, y, ref_width, ref_height = box
    if not all(math.isfinite(value) for value in (*box, angle)) or min(ref_width, ref_height) <= 0:
        raise ValueError(f'Invalid reference patch: {box}, angle {angle}')
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    scale_x, scale_y = ref_width / width, ref_height / height
    # Map the requested output pixel back into the unrotated reference image.
    matrix = np.array([[cosine * scale_x, sine * scale_y, x + (ref_width - 1) / 2],
                       [-sine * scale_x, cosine * scale_y, y + (ref_height - 1) / 2]])
    matrix[:, 2] -= matrix[:, :2] @ [(width - 1) / 2, (height - 1) / 2]
    patch = cv2.warpAffine(image, matrix, (width, height),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.ogrid[:height, :width]
    original_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
    original_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
    valid = ((original_x >= 0) & (original_x <= image.shape[1] - 1)
             & (original_y >= 0) & (original_y <= image.shape[0] - 1))
    return patch, valid.astype(np.float32)


def texture(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=.125)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=.125)
    return np.hypot(dx, dy)


def describe(patch, valid, descriptors):
    height, width = descriptors.shape[:2]
    gradients = texture(patch)
    coverage = cv2.resize(valid, (width, height), interpolation=cv2.INTER_AREA)
    informative = gradients[valid > 0]
    strength = float(np.clip((np.percentile(informative, 90) - .006) / .045, 0, 1)) if informative.size else 0.
    detail = cv2.resize(gradients, (width, height), interpolation=cv2.INTER_AREA)
    return descriptors, coverage, detail, strength


def compare(query, reference, neutral):
    query_tokens, query_valid, _, query_texture = query
    reference_tokens, ref_valid, ref_detail, reference_texture = reference
    height, width = ref_valid.shape
    yy, xx = np.mgrid[:height, :width]
    # The request is centered on the diagnostic detail. Weight it more than
    # context and padding, with extra weight on edges in the training reference.
    radius = ((yy + .5 - height / 2) / height) ** 2 + ((xx + .5 - width / 2) / width) ** 2
    weights = np.exp(-6 * radius) * np.clip(.25 + ref_detail / .03, .25, 2.) * ref_valid * query_valid
    if weights.sum() <= 0:
        return {'match': 0., 'visibility': 0., 'delta': 0.}
    cosine = np.einsum('ijk,ijk->ij', query_tokens, reference_tokens)
    match = float(np.clip(np.average(cosine, weights=weights), -1, 1))
    reliability = float(ref_valid.mean() * math.sqrt(query_texture * reference_texture))
    # This is a texture/coverage proxy. A visually ambiguous patch can give
    # little evidence in either direction; no ground-truth visibility is used.
    delta = reliability * float(np.clip((match - neutral) / COSINE_SPAN, -1, 1))
    return {'match': round(match, 8), 'visibility': round(reliability, 8), 'delta': round(delta, 8)}


class PatchReader:
    def __init__(self, encoder, reference_root, scene_root, batch_size, neutral):
        self.encoder = encoder
        self.reference_root = reference_root
        self.scene_root = scene_root
        self.batch_size = batch_size
        self.neutral = neutral
        self.reference_images = {}

    @staticmethod
    def image(root, name):
        image = cv2.imread(str(root / name), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode image: {root / name}')
        return image

    def reference(self, row):
        name = row['image']
        if name not in self.reference_images:
            self.reference_images[name] = self.image(self.reference_root, name)
        return self.reference_images[name]

    def encode(self, patches):
        descriptions = {}
        items = list(patches.items())
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            vectors = self.encoder.batch_features([patch for _, (patch, _) in batch], INPUT_SIZE)
            for (key, (patch, valid)), descriptor in zip(batch, vectors, strict=True):
                descriptions[key] = describe(patch, valid, descriptor)
        return descriptions

    def read(self, frame, references):
        candidates = {row['id']: row for row in frame['candidates']}
        if not any(frame['plans'].values()):
            return {'id': frame['id'], 'responses': {name: [] for name in frame['plans']}}
        image = self.image(self.scene_root, frame['image'])
        if (frame['width'], frame['height']) != image.shape[1::-1]:
            raise ValueError(f'Unexpected native image size: {frame["id"]}')
        patches = {}
        comparisons = {}
        for policy, plan in frame['plans'].items():
            comparisons[policy] = []
            for request in plan:
                candidate = candidates[request['candidateId']]
                square = tuple(request['box'])
                reference_key = ('reference', candidate['key'], tuple(request['refBox']), request['angle'], square[2:])
                query_key = ('query', square)
                if query_key not in patches:
                    patch = native_patch(image, square)
                    patches[query_key] = patch, np.ones(patch.shape[:2], dtype=np.float32)
                if reference_key not in patches:
                    patches[reference_key] = posed_reference(self.reference(references[candidate['key']]),
                                                              request['refBox'], request['angle'], square[2], square[3])
                comparisons[policy].append((request['id'], query_key, reference_key))
        descriptors = self.encode(patches)
        responses = {policy: [{'id': identifier, **compare(descriptors[query], descriptors[reference], self.neutral)}
                              for identifier, query, reference in requests]
                     for policy, requests in comparisons.items()}
        return {'id': frame['id'], 'responses': responses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--weights', type=Path, help='Directory with official frozen neural weights.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--scene-root', type=Path, default=Path.cwd())
    parser.add_argument('--reference-root', type=Path, default=Path.cwd())
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--neutral-cosine', type=float, default=NEUTRAL_COSINE)
    arguments = parser.parse_args()
    if arguments.batch_size < 1 or not -1 <= arguments.neutral_cosine <= 1:
        parser.error('Batch size must be positive and --neutral-cosine must be in [-1, 1].')
    cv2.setNumThreads(1)
    plan = json.loads(arguments.plan.read_text(encoding='utf-8'))
    references = {row['key']: row for row in plan['references']}
    encoder = load_encoder()(weights=arguments.weights, device=arguments.device)
    reader = PatchReader(encoder, arguments.reference_root, arguments.scene_root, arguments.batch_size, arguments.neutral_cosine)
    rows = []
    for index, frame in enumerate(plan['frames'], 1):
        rows.append(reader.read(frame, references))
        if index % 20 == 0:
            print(f'Compared native patches in {index}/{len(plan["frames"])} frames', file=sys.stderr, flush=True)
    report = {'metadata': {**plan.get('metadata', {}), 'verifier': 'position-weighted dense cosine',
                           'modelPatchSize': list(INPUT_SIZE), 'neutralCosine': arguments.neutral_cosine,
                           'visibility': 'reference coverage times geometric mean of crop texture estimates',
                           'delta': 'visibility times clipped cosine residual'}, 'frames': rows}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
