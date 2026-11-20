"""Show measured bolt–nut matches over the synthetic test photos."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
BACKGROUND = "#f8f7f3"
INK = "#1d2830"
SECONDARY = "#53616b"
TEAL = "#0f675b"


def show_photo(axis, photo):
    width, height = photo.size
    axis.imshow(photo, extent=(0, width, height, 0), origin="upper")
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.axis("off")


def annotate_match(axis, photo, size, margin, bolt, nut):
    width, height = photo.size
    x_scale = (width - 2 * margin) / size
    y_scale = (height - 2 * margin) / size

    for item in (bolt, nut):
        bounds = item["bounds"]
        highlight = Rectangle(
            (margin + bounds["x0"] * x_scale, margin + bounds["y0"] * y_scale),
            (bounds["x1"] + 1 - bounds["x0"]) * x_scale,
            (bounds["y1"] + 1 - bounds["y0"]) * y_scale,
            facecolor="none", edgecolor=TEAL, linewidth=2.1, zorder=3,
        )
        highlight.set_path_effects([patheffects.Stroke(linewidth=3.8, foreground="#f4fff9"), patheffects.Normal()])
        axis.add_patch(highlight)

    ux, uy = bolt["axis"]

    def point(offset):
        return (margin + (bolt["x"] + 0.5 + ux * offset) * x_scale,
                margin + (bolt["y"] + 0.5 + uy * offset) * y_scale)

    length = bolt["length"]
    arrow = FancyArrowPatch(point(length * 0.20), point(length * 0.66), arrowstyle="-|>",
                            mutation_scale=12, linewidth=2, color=TEAL, zorder=4)
    arrow.set_path_effects([patheffects.Stroke(linewidth=3.6, foreground="#f4fff9"), patheffects.Normal()])
    axis.add_patch(arrow)


def render(data):
    with Image.open(OUTPUTS / data["sourceFile"]) as triptych:
        width, height = triptych.size
        first, second, scene = [triptych.crop((i * width // 3, 0, (i + 1) * width // 3, height))
                                for i in range(3)]

    figure = plt.figure(figsize=(11.3, 5.4), facecolor=BACKGROUND)
    figure.text(.043, .916, "Retrouver un assemblage sur des photos", fontsize=22,
                color=INK, fontweight="semibold")
    figure.text(.043, .862, "Essai sur trois images synthétiques", fontsize=11, color=SECONDARY)

    for x_position, photo, title in ((.043, first, "Exemple 1"), (.310, second, "Exemple 2")):
        photo_axis = figure.add_axes([x_position, .264, .224, .4687])
        show_photo(photo_axis, photo)
        figure.text(x_position, .765, title, fontsize=14, color=INK, fontweight="semibold")

    scene_x = .610
    scene_axis = figure.add_axes([scene_x, .065, .319, .6676])
    show_photo(scene_axis, scene)
    figure.text(scene_x, .765, "Nouvelle scène", fontsize=14, color=INK, fontweight="semibold")
    for match in data["matches"]:
        bolt = data["scene"]["bolts"][match["boltIndex"]]
        nut = data["scene"]["nuts"][match["nutIndex"]]
        annotate_match(scene_axis, scene, data["size"], data["cropMargin"], bolt, nut)

    label = "Paire repérée en vert" if data["matches"] else "Aucune paire repérée"
    figure.text(scene_x, .027, label, fontsize=10.5, color=SECONDARY)
    figure.savefig(OUTPUTS / "sphere-assemblage-photo-resultat.png", dpi=160, facecolor=BACKGROUND)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-assemblage-photo-mesures.json"
    render(json.loads(source.read_text(encoding="utf-8")))
