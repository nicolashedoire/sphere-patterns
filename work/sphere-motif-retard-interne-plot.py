"""Render the two input motifs and the decoded result selected by the model."""

import json
from pathlib import Path
import runpy


WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"


def render(data):
    style = runpy.run_path(str(WORK / "sphere-motif-compose-plot.py"))
    plt = style["plt"]
    colors = style["ListedColormap"]
    size = data["size"]
    patterns = data["patterns"]
    background = style["BACKGROUND"]
    input_colors = colors([style["EMPTY"], "#31556d"])
    decoded_colors = colors([style["EMPTY"], style["ORANGE"]])

    delay = data["selection"]["delayTicks"]
    alignment_label = "Alignement retenu"
    if delay:
        motion = "avancé" if delay < 0 else "retardé"
        alignment_label = f"B {motion} de {abs(delay)} pas"

    figure, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), facecolor=background)
    figure.subplots_adjust(left=0.048, right=0.952, bottom=0.1, top=0.74, wspace=0.24)
    panels = (
        ("a", "Motif A", "Position à l’entrée", input_colors),
        ("b", "Motif B", "Position à l’entrée", input_colors),
        ("c", "Signal C décodé", alignment_label, decoded_colors),
    )
    for axis, (key, title, subtitle, palette) in zip(axes, panels):
        grid = style["grid_data"](patterns[key], size)
        style["draw_grid"](axis, grid, title, subtitle, palette)

    figure.savefig(OUTPUTS / "sphere-decalage-signal-autonome.png", dpi=160, facecolor=background)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-decalage-interne-mesures.json"
    render(json.loads(source.read_text(encoding="utf-8")))
