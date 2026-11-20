"""Show the stored reference and measured matching pixels in the scene."""

import json
from pathlib import Path
import runpy


WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"


def render(data):
    style = runpy.run_path(str(WORK / "sphere-motif-compose-plot.py"))
    plt = style["plt"]
    np = style["np"]
    colors = style["ListedColormap"]
    grid_data = style["grid_data"]
    draw_grid = style["draw_grid"]
    size = data["size"]
    patterns = data["patterns"]
    empty = style["EMPTY"]
    orange = style["ORANGE"]
    blue = "#31556d"
    background = style["BACKGROUND"]

    reference = grid_data(patterns["reference"], size)
    scene = grid_data(patterns["scene"], size)
    matched = grid_data(patterns["matchedInScene"], size) > 0
    highlight = np.where(matched, 1, scene * 0.5)

    figure, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), facecolor=background)
    figure.subplots_adjust(left=0.048, right=0.952, bottom=0.1, top=0.74, wspace=0.24)
    draw_grid(axes[0], reference, "Référence C", "Motif conservé", colors([empty, orange]))
    draw_grid(axes[1], scene, "Nouvelle scène", "Motifs à l’entrée", colors([empty, blue]))
    draw_grid(axes[2], highlight, "Motif retrouvé", "Correspondance en orange", colors([empty, blue, orange]))

    figure.savefig(OUTPUTS / "sphere-reference-apprise.png", dpi=160, facecolor=background)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-reference-apprise-mesures.json"
    render(json.loads(source.read_text(encoding="utf-8")))
