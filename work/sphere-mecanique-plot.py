"""Plot measured localization over the synthetic photograph used in the test."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from PIL import Image


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
BACKGROUND = "#f8f7f3"
INK = "#1d2830"
SECONDARY = "#53616b"
GOLD = "#e4a11b"


def show_photo(axis, photo):
    width, height = photo.size
    axis.imshow(photo, extent=(0, width, height, 0), origin="upper")
    axis.axis("off")


def highlight_match(axis, photo, matched, margin):
    rows, columns = matched.shape
    width, height = photo.size
    x_edges = np.linspace(margin, width - margin, columns + 1)
    y_edges = np.linspace(margin, height - margin, rows + 1)

    fill = np.zeros((rows, columns, 4))
    fill[matched, :3] = to_rgb(GOLD)
    fill[matched, 3] = 0.42
    axis.imshow(fill, extent=(x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]),
                origin="upper", interpolation="nearest")

    border = []
    for row, column in np.argwhere(matched):
        x0, x1 = x_edges[column:column + 2]
        y0, y1 = y_edges[row:row + 2]
        if row == 0 or not matched[row - 1, column]:
            border.append([(x0, y0), (x1, y0)])
        if row == rows - 1 or not matched[row + 1, column]:
            border.append([(x0, y1), (x1, y1)])
        if column == 0 or not matched[row, column - 1]:
            border.append([(x0, y0), (x0, y1)])
        if column == columns - 1 or not matched[row, column + 1]:
            border.append([(x1, y0), (x1, y1)])
    axis.add_collection(LineCollection(border, colors="#a35a17", linewidths=1.2))
    axis.set(xlim=(0, width), ylim=(height, 0))


def render(data):
    size = data["size"]
    pixels = data["patterns"]
    reference = np.asarray(pixels["reference"]).reshape(size, size)
    matched = np.asarray(pixels["matchedInScene"]).reshape(size, size) > 0

    with Image.open(OUTPUTS / "sphere-photos-pieces.png") as triptych:
        width, height = triptych.size
        first, second, scene = [triptych.crop((i * width // 3, 0, (i + 1) * width // 3, height))
                                for i in range(3)]

    figure = plt.figure(figsize=(11.7, 4.7), facecolor=BACKGROUND)
    figure.text(0.043, 0.92, "Retrouver une pièce dans une scène", fontsize=22,
                color=INK, fontweight="semibold")
    figure.text(0.043, 0.858, "Essai sur trois photos synthétiques", fontsize=11, color=SECONDARY)
    figure.text(0.043, 0.751, "1. Former la référence", fontsize=14, color=INK, fontweight="semibold")
    figure.text(0.658, 0.751, "2. Localiser la pièce", fontsize=14, color=INK, fontweight="semibold")

    for x_position, photo, label in ((0.043, first, "Photo 1"), (0.216, second, "Photo 2")):
        photo_axis = figure.add_axes([x_position, 0.287, 0.152, 0.3784])
        show_photo(photo_axis, photo)
        figure.text(x_position, 0.239, label, fontsize=10.5, color=SECONDARY)

    figure.text(0.392, 0.461, "→", fontsize=16, color=SECONDARY)
    reference_axis = figure.add_axes([0.432, 0.287, 0.152, 0.3784])
    reference_axis.imshow(reference, cmap=ListedColormap(["#eceae2", "#31556d"]),
                          vmin=0, vmax=1, interpolation="nearest")
    reference_axis.axis("off")
    figure.text(0.432, 0.239, f"Référence · {size} × {size}", fontsize=10.5, color=SECONDARY)

    scene_axis = figure.add_axes([0.658, 0.082, 0.234, 0.5825])
    show_photo(scene_axis, scene)
    if matched.any():
        highlight_match(scene_axis, scene, matched, data["cropMargin"])
        label = "Cellules correspondantes en doré"
    else:
        label = "Aucune correspondance détectée"
    figure.text(0.658, 0.042, label, fontsize=10.5, color=SECONDARY)

    figure.savefig(OUTPUTS / "sphere-piece-localisee.png", dpi=160, facecolor=BACKGROUND)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-mecanique-mesures.json"
    render(json.loads(source.read_text(encoding="utf-8")))
