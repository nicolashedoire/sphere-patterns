"""Draw the measured bolt and nut arrangements as a schematic."""

import json
from math import ceil, cos, floor, pi, sin
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, RegularPolygon


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
BACKGROUND = "#f8f7f3"
PANEL = "#f1f1ea"
INK = "#1d2830"
SECONDARY = "#53616b"
BLUE = ("#819dab", "#365368")
GREEN = ("#7fc5b1", "#0f6155")
MUTED = ("#c4cbca", "#687878")


def draw_bolt(axis, bolt, palette):
    x, y = bolt["x"], bolt["y"]
    angle = bolt["orientation"] * pi / 2
    ca, sa = cos(angle), sin(angle)

    def rotate(point):
        x_local, y_local = point
        return (x + x_local * ca - y_local * sa, y + x_local * sa + y_local * ca)

    edges = [(-.70, -.27), (-.61, -.36), (-.30, -.36), (-.22, -.28),
             (-.22, -.17), (.71, -.17), (.82, -.09), (.82, .09),
             (.71, .17), (-.22, .17), (-.22, .28), (-.30, .36),
             (-.61, .36), (-.70, .27)]
    fill, outline = palette
    axis.add_patch(Polygon([rotate(point) for point in edges], closed=True, facecolor=fill,
                           edgecolor=outline, linewidth=1.8, joinstyle="round", zorder=3))
    for thread in (.23, .39, .55, .71):
        start = rotate((thread, -.17))
        end = rotate((thread - .12, .17))
        axis.plot([start[0], end[0]], [start[1], end[1]], color=outline, linewidth=1.1, zorder=4)


def draw_nut(axis, nut, palette):
    center = (nut["x"], nut["y"])
    fill, outline = palette
    axis.add_patch(RegularPolygon(center, numVertices=6, radius=.48, orientation=pi / 6,
                                  facecolor=fill, edgecolor=outline, linewidth=1.8, zorder=3))
    axis.add_patch(Circle(center, radius=.22, facecolor=PANEL, edgecolor=outline,
                          linewidth=1.1, zorder=4))


def draw_scene(axis, scene, pairs=None):
    bolts = scene["bolts"]
    nuts = scene["nuts"]
    objects = [*bolts, *nuts]
    xs = [o["x"] for o in objects]
    ys = [o["y"] for o in objects]
    axis.set_facecolor(PANEL)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    axis.set_anchor("W")
    x_span, y_span = max(xs) - min(xs), max(ys) - min(ys)
    if pairs is None:
        x_span = y_span = max(x_span, y_span)
    x_mean = (min(xs) + max(xs)) / 2
    y_mean = (min(ys) + max(ys)) / 2
    x_min, x_max = x_mean - x_span / 2 - 1.15, x_mean + x_span / 2 + 1.15
    y_min, y_max = y_mean - y_span / 2 - 1.15, y_mean + y_span / 2 + 1.15
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_max, y_min)

    x_grid = list(range(ceil(x_min), floor(x_max) + 1))
    for y in range(ceil(y_min), floor(y_max) + 1):
        axis.scatter(x_grid, [y] * len(x_grid), s=4, color="#d8dbd1", zorder=1)

    matched_bolts = set()
    matched_nuts = set()
    for pair in pairs or []:
        i, j = pair["boltIndex"], pair["nutIndex"]
        matched_bolts.add(i)
        matched_nuts.add(j)
        bolt, nut = bolts[i], nuts[j]
        axis.plot([bolt["x"], nut["x"]], [bolt["y"], nut["y"]],
                  color=GREEN[1], alpha=.65, linewidth=1.6, linestyle=(0, (3, 3)), zorder=2)

    for i, bolt in enumerate(bolts):
        palette = BLUE if pairs is None else GREEN if i in matched_bolts else MUTED
        draw_bolt(axis, bolt, palette)
    for i, nut in enumerate(nuts):
        palette = BLUE if pairs is None else GREEN if i in matched_nuts else MUTED
        draw_nut(axis, nut, palette)


def render(data):
    figure = plt.figure(figsize=(11.6, 4.65), facecolor=BACKGROUND)
    figure.text(.043, .91, "Retrouver un assemblage boulon–écrou", fontsize=22,
                color=INK, fontweight="semibold")
    figure.text(.043, .771, "1. Observer deux exemples", fontsize=14,
                color=INK, fontweight="semibold")
    figure.text(.562, .771, "2. Présenter une nouvelle scène", fontsize=14,
                color=INK, fontweight="semibold")
    for x, example, label in zip((.043, .281), data["examples"], ("Exemple 1", "Exemple 2")):
        axis = figure.add_axes([x, .254, .213, .447])
        draw_scene(axis, example)
        figure.text(x, .219, label, fontsize=10.5, color=SECONDARY)

    learned = data["learned"]
    figure.text(.043, .125, f"Écart appris · {learned['along']} cases dans l’axe, {learned['lateral']} sur le côté",
                fontsize=10.5, color=SECONDARY)

    scene_axis = figure.add_axes([.562, .155, .393, .551])
    draw_scene(scene_axis, data["scene"], data["matches"])
    caption = "Paire repérée en vert" if data["matches"] else "Aucune paire repérée"
    figure.text(.562, .107, caption, fontsize=10.5, color=SECONDARY)

    figure.savefig(OUTPUTS / "sphere-assemblage-boulon-ecrou.png", dpi=160, facecolor=BACKGROUND)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-assemblage-mesures.json"
    render(json.loads(source.read_text(encoding="utf-8")))
