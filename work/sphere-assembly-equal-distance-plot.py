"""Compare the measured bolt-to-nut paths over the generated photo."""

import json
from math import hypot
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from PIL import Image


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
BACKGROUND = "#f8f7f3"
INK = "#1d2830"
SECONDARY = "#53616b"
TEAL = "#0f675b"
MUTED = "#677771"


def direction_label(dx, dy):
    if abs(dy) > abs(dx):
        return "Nut above" if dy < 0 else "Nut below"
    return "Nut on the left" if dx < 0 else "Nut on the right"


def render(data):
    with Image.open(OUTPUTS / "sphere-assembly-equal-distance.png") as opened:
        photo = opened.convert("RGB")
    width, height = photo.size
    margin = data["cropMargin"]
    size = data["size"]
    x_scale, y_scale = (width - 2 * margin) / size, (height - 2 * margin) / size

    def point(x, y):
        return margin + x * x_scale, margin + y * y_scale

    def center(item):
        return point(item["x"] + .5, item["y"] + .5)

    def box(item, color):
        bounds = item["bounds"]
        border = Rectangle(point(bounds["x0"], bounds["y0"]),
                           (bounds["x1"] + 1 - bounds["x0"]) * x_scale,
                           (bounds["y1"] + 1 - bounds["y0"]) * y_scale,
                           facecolor="none", edgecolor=color, linewidth=2.05, zorder=4)
        border.set_path_effects([patheffects.Stroke(linewidth=3.4, foreground="white"), patheffects.Normal()])
        axis.add_patch(border)

    bolt, = data["scene"]["bolts"]
    nuts = data["scene"]["nuts"]
    matches = {pair["nutIndex"] for pair in data["matches"] if pair["boltIndex"] == 0}

    figure = plt.figure(figsize=(10.1, 5.3), facecolor=BACKGROUND)
    figure.text(.047, .923, "Compare the two nuts", fontsize=22,
                color=INK, fontweight="semibold")
    axis = figure.add_axes([.047, .089, .394, .7508])
    axis.imshow(photo, extent=(0, width, height, 0), origin="upper")
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.axis("off")
    figure.text(.047, .04, "Measurements from a synthetic image", color=SECONDARY, fontsize=10)

    measurements = []
    bx, by = center(bolt)
    box(bolt, TEAL if matches else MUTED)
    for index, nut in enumerate(nuts):
        color = TEAL if index in matches else MUTED
        nx, ny = center(nut)
        line, = axis.plot([bx, nx], [by, ny], linestyle=(0, (3, 3)), linewidth=2,
                          color=color, zorder=3)
        line.set_path_effects([patheffects.Stroke(linewidth=3.5, foreground="white"), patheffects.Normal()])
        axis.scatter([nx], [ny], s=22, color=color, edgecolors="white", linewidths=1, zorder=5)
        box(nut, color)
        dx, dy = nut["x"] - bolt["x"], nut["y"] - bolt["y"]
        measurements.append((direction_label(dx, dy), hypot(dx, dy), index in matches))
    axis.scatter([bx], [by], s=22, color=INK, edgecolors="white", linewidths=1, zorder=5)

    figure.text(.497, .785, "Measured distances", color=INK, fontsize=16, fontweight="semibold")
    figure.text(.497, .735, "Center to center on the grid", color=SECONDARY, fontsize=11)
    for row, (label, distance, matched) in enumerate(measurements):
        y = .605 - row * .193
        color = TEAL if matched else MUTED
        figure.text(.497, y, label, color=color, fontsize=13, fontweight="semibold")
        figure.text(.497, y - .067, f"{distance:.1f} cells",
                    color=INK, fontsize=16)
        if matched:
            figure.text(.497, y - .114, "Match found", color=TEAL, fontsize=10.5)
    if not matches:
        figure.text(.497, .19, "No match found", color=SECONDARY, fontsize=11)

    figure.savefig(OUTPUTS / "sphere-assembly-equal-distance-result.png", dpi=160, facecolor=BACKGROUND)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-assembly-equal-distance-measurements.json"
    render(json.loads(source.read_text(encoding="utf-8")))
