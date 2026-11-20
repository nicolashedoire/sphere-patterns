"""Display the common motif measured before and after compensating for delay."""

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
    delay = data["bestShift"]["delayTicks"]
    if delay:
        motion = "avancé" if delay < 0 else "retardé"
        alignment_label = f"B {motion} de {abs(delay)} pas"
    else:
        alignment_label = "Au meilleur alignement"

    figure, axes = plt.subplots(1, 4, figsize=(10.8, 3.2), facecolor=background)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.1, top=0.74, wspace=0.24)
    panels = (
        ("a", "Motif A", "Position à l’entrée", input_colors),
        ("b", "Motif B", "Position à l’entrée", input_colors),
        ("cDirect", "C direct", "Sans alignement", decoded_colors),
        ("cAligned", "C aligné", alignment_label, decoded_colors),
    )
    for axis, (key, title, subtitle, palette) in zip(axes, panels):
        grid = style["grid_data"](patterns[key], size)
        style["draw_grid"](axis, grid, title, subtitle, palette)

    figure.savefig(OUTPUTS / "sphere-motif-commun-decale.png", dpi=160, facecolor=background)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-motifs-decales-mesures.json"
    render(json.loads(source.read_text(encoding="utf-8")))
