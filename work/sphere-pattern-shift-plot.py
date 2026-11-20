"""Display the common motif measured before and after compensating for delay."""

import json
from pathlib import Path
import runpy


WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"


def render(data):
    style = runpy.run_path(str(WORK / "sphere-pattern-compose-plot.py"))
    plt = style["plt"]
    colors = style["ListedColormap"]
    size = data["size"]
    patterns = data["patterns"]
    background = style["BACKGROUND"]
    input_colors = colors([style["EMPTY"], "#31556d"])
    decoded_colors = colors([style["EMPTY"], style["ORANGE"]])
    delay = data["bestShift"]["delayTicks"]
    if delay:
        motion = "advanced" if delay < 0 else "delayed"
        alignment_label = f"B {motion} by {abs(delay)} ticks"
    else:
        alignment_label = "Best alignment"

    figure, axes = plt.subplots(1, 4, figsize=(10.8, 3.2), facecolor=background)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.1, top=0.74, wspace=0.24)
    panels = (
        ("a", "Pattern A", "Input position", input_colors),
        ("b", "Pattern B", "Input position", input_colors),
        ("cDirect", "Direct C", "Without alignment", decoded_colors),
        ("cAligned", "Aligned C", alignment_label, decoded_colors),
    )
    for axis, (key, title, subtitle, palette) in zip(axes, panels):
        grid = style["grid_data"](patterns[key], size)
        style["draw_grid"](axis, grid, title, subtitle, palette)

    figure.savefig(OUTPUTS / "sphere-shifted-common-pattern.png", dpi=160, facecolor=background)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-shifted-patterns-measurements.json"
    render(json.loads(source.read_text(encoding="utf-8")))
