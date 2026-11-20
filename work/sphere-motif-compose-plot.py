"""Render the measured motifs and the fraction of probe energy rerouted."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
BACKGROUND = "#f8f7f3"
EMPTY = "#eceae2"
INK = "#1d2830"
ORANGE = "#ac4925"


def grid_data(values, size):
    grid = np.asarray(values, dtype=float).reshape(size, size)
    if not np.isfinite(grid).all():
        raise ValueError("Un motif contient une amplitude non finie.")
    return grid


def draw_grid(axis, grid, title, subtitle, colormap):
    size = grid.shape[0]
    image = axis.imshow(grid, cmap=colormap, vmin=0, vmax=1, interpolation="nearest")
    borders = np.arange(size + 1) - 0.5
    axis.set_xticks(borders, minor=True)
    axis.set_yticks(borders, minor=True)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(which="minor", color=BACKGROUND, linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, loc="left", fontsize=14, fontweight="semibold", color=INK, pad=26)
    axis.text(0, 1.025, subtitle, transform=axis.transAxes, color="#53616b", fontsize=10, va="bottom")
    return image


def render(data):
    size = data["size"]
    patterns = data["patterns"]
    probe = data["probe"]["withC"]
    original = grid_data(probe["original"], size) ** 2
    diverted = grid_data(probe["diverted"], size) ** 2
    total = original + diverted
    fraction = np.divide(diverted, total, out=np.zeros_like(diverted), where=total > 0)

    input_colors = ListedColormap([EMPTY, "#31556d"])
    decoded_colors = ListedColormap([EMPTY, ORANGE])
    energy_colors = LinearSegmentedColormap.from_list("diverted_energy", [EMPTY, ORANGE])

    figure, axes = plt.subplots(1, 4, figsize=(10.8, 3.5), facecolor=BACKGROUND)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.20, top=0.73, wspace=0.24)
    draw_grid(axes[0], grid_data(patterns["a"], size), "Motif A", "Bande verticale", input_colors)
    draw_grid(axes[1], grid_data(patterns["b"], size), "Motif B", "Bande horizontale", input_colors)
    draw_grid(axes[2], grid_data(patterns["c"], size), "Signal C décodé", "Zone de croisement", decoded_colors)
    color_image = draw_grid(axes[3], fraction, "Onde D", "Part d’énergie déviée", energy_colors)

    position = axes[3].get_position()
    legend = figure.add_axes([position.x0, 0.16, position.width, 0.022])
    scale = figure.colorbar(color_image, cax=legend, orientation="horizontal", ticks=[0, 0.5, 1])
    scale.ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    scale.ax.tick_params(labelsize=9, length=0, pad=3, colors="#53616b")
    scale.outline.set_visible(False)

    figure.savefig(OUTPUTS / "sphere-troisieme-signal.png", dpi=160, facecolor=BACKGROUND)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-croisement-motifs-mesures.json"
    render(json.loads(source.read_text(encoding="utf-8")))
