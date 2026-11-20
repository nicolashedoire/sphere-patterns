"""Read one equal-resolution silhouette grid per photo in the contact sheet."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--photo', type=Path, default=root / 'outputs/sphere-part-photos.png')
parser.add_argument('--scene', type=Path, help='Replace the third panel with a separate square photo.')
parser.add_argument('--output', type=Path, default=root / 'work/sphere-mechanical-part-masks.json')
parser.add_argument('--size', type=int, default=25)
parser.add_argument('--threshold', type=int, default=190)
args = parser.parse_args()
photo, size = args.photo, args.size
threshold, margin = args.threshold, 5
with Image.open(photo) as source:
    assert source.width == 3 * source.height
    side = source.height
    masks = []
    for panel in range(3):
        box = (panel * side + margin, margin, (panel + 1) * side - margin, side - margin)
        reduced = source.crop(box).convert('L').resize((size, size), Image.Resampling.BOX)
        masks.append([int(value < threshold) for value in reduced.get_flattened_data()])

parameters = {'size': size, 'threshold': threshold, 'cropMargin': margin, 'sourceFile': photo.name,
              'sourceSHA256': hashlib.sha256(photo.read_bytes()).hexdigest(), 'masks': masks}
if args.scene:
    with Image.open(args.scene) as image:
        assert image.width == image.height, 'The separate photo must be square.'
        scene = image.crop((margin, margin, image.width - margin, image.height - margin))
        scene = scene.convert('L').resize((size, size), Image.Resampling.BOX)
        masks[2] = [int(value < threshold) for value in scene.get_flattened_data()]
    parameters.update(sceneFile=args.scene.name, sceneSHA256=hashlib.sha256(args.scene.read_bytes()).hexdigest())
args.output.write_text(json.dumps(parameters) + '\n')
