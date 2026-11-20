"""Show the two learned shapes and their matches in the measured scene."""

import json
from pathlib import Path
import runpy


WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"


def render(data):
    style = runpy.run_path(str(WORK / "sphere-pattern-compose-plot.py"))
    plt = style["plt"]
    np = style["np"]
    colors = style["ListedColormap"]
    grid_data = style["grid_data"]
    draw_grid = style["draw_grid"]
    size = data["size"]
    patterns = data["patterns"]
    empty = style["EMPTY"]
    orange = style["ORANGE"]
    green = "#1d6b63"
    gray = "#8d9295"
    background = style["BACKGROUND"]

    reference_l, reference_t = (grid_data(mask, size) for mask in patterns["references"])
    scene = grid_data(patterns["scene"], size)
    match_l, match_t = (grid_data(mask, size) > 0 for mask in patterns["matchedInScene"])
    highlights = np.where(match_l, 0.6, np.where(match_t, 1, scene * 0.34))

    figure = plt.figure(figsize=(12.4, 4.2), facecolor=background)
    ink = style["INK"]
    secondary = "#53616b"
    figure.text(0.043, 0.907, "How the sphere finds two shapes", fontsize=22,
                color=ink, fontweight="semibold")

    stages = (
        (0.043, "1. Form a reference", "Two examples share a pattern."),
        (0.493, "2. Show a scene", "Two shapes and a decoy."),
        (0.744, "3. Find L and T", "The sphere locates their pixels."),
    )
    for x_position, title, description in stages:
        figure.text(x_position, 0.773, title, fontsize=14, color=ink, fontweight="semibold")
        figure.text(x_position, 0.713, description, fontsize=10.5, color=secondary)

    blue = "#31556d"
    first_t, second_t = (grid_data(mask, size) for mask in patterns["trainingPairs"][1])
    miniatures = (
        (0.043, first_t, "Example 1", blue),
        (0.167, second_t, "Example 2", blue),
        (0.317, reference_t, "Reference T", green),
    )
    for x_position, grid, label, color in miniatures:
        axis = figure.add_axes([x_position, 0.388, 0.093, 0.2746])
        draw_grid(axis, grid, "", "", colors([empty, color]))
        figure.text(x_position, 0.348, label, fontsize=9.5, color=secondary)
    figure.text(0.145, 0.506, "+", fontsize=13, color=secondary)
    figure.text(0.275, 0.506, "→", fontsize=17, color=secondary)

    l_axis = figure.add_axes([0.043, 0.167, 0.047, 0.1388])
    draw_grid(l_axis, reference_l, "", "", colors([empty, orange]))
    figure.text(0.106, 0.226, "The reference for L stays in memory.", fontsize=10.5, color=secondary)

    scene_axis = figure.add_axes([0.493, 0.148, 0.174, 0.5137])
    found_axis = figure.add_axes([0.744, 0.148, 0.174, 0.5137])
    draw_grid(scene_axis, scene, "", "", colors([empty, blue]))
    draw_grid(found_axis, highlights, "", "", colors([empty, gray, orange, green]))
    figure.text(0.744, 0.081, "L", fontsize=11, fontweight="semibold", color=orange)
    figure.text(0.785, 0.081, "T", fontsize=11, fontweight="semibold", color=green)
    figure.text(0.825, 0.081, "Decoy", fontsize=11, color=secondary)

    figure.savefig(OUTPUTS / "sphere-two-shapes-found.png", dpi=160, facecolor=background)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-two-references-measurements.json"
    render(json.loads(source.read_text(encoding="utf-8")))
