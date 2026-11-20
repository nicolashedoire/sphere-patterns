"""Show the measured pairings and compared distances over the generated scene."""

import json
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
MUTED = "#7c8580"
PAIR_COLORS = ("#0f675b", "#a34b20", "#5e4d9b")


def render(data, destination=None, title="Associer chaque boulon à son écrou"):
    with Image.open(OUTPUTS / data["sceneFile"]) as opened:
        photo = opened.convert("RGB")
    width, height = photo.size
    margin = data["cropMargin"]
    size = data["size"]
    x_scale, y_scale = (width - 2 * margin) / size, (height - 2 * margin) / size

    def position(x, y):
        return margin + x * x_scale, margin + y * y_scale

    def center(item):
        return position(item["x"] + .5, item["y"] + .5)

    def box(item, color):
        bounds = item["bounds"]
        border = Rectangle(position(bounds["x0"], bounds["y0"]),
                           (bounds["x1"] + 1 - bounds["x0"]) * x_scale,
                           (bounds["y1"] + 1 - bounds["y0"]) * y_scale,
                           facecolor="none", edgecolor=color, linewidth=1.7, zorder=4)
        border.set_path_effects([patheffects.Stroke(linewidth=3.1, foreground="white"), patheffects.Normal()])
        axis.add_patch(border)

    def path(bolt, nut, color, selected):
        bx, by = center(bolt)
        nx, ny = center(nut)
        line, = axis.plot([bx, nx], [by, ny], color=color, linewidth=1.6 if selected else 1,
                          linestyle=(0, (3, 3)), alpha=1 if selected else .7, zorder=3 if selected else 2)
        line.set_path_effects([patheffects.Stroke(linewidth=2.7, foreground="white"), patheffects.Normal()])

    bolts, nuts = data["scene"]["bolts"], data["scene"]["nuts"]
    controls = data["distanceControls"]
    pairs = {(match["boltIndex"], match["nutIndex"]) for match in data["matches"]}
    color_by_nut = {j: PAIR_COLORS[i] for i, j in sorted(pairs)}
    selected_bolts = {i for i, _ in pairs}
    intended_pairs = {(row["boltIndex"], row["matchedNutIndex"]) for row in controls}

    figure = plt.figure(figsize=(11, 5.6), facecolor=BACKGROUND)
    figure.text(.046, .925, title, fontsize=22,
                color=INK, fontweight="semibold")
    axis = figure.add_axes([.046, .084, .394, .7739])
    axis.imshow(photo, extent=(0, width, height, 0), origin="upper")
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.axis("off")
    figure.text(.046, .041, "Mesures sur une photo générée", color=SECONDARY, fontsize=10)

    for row in controls:
        i, j = row["boltIndex"], row["decoyNutIndex"]
        if (i, j) not in pairs:
            path(bolts[i], nuts[j], MUTED, False)
    for i, j in sorted(pairs):
        path(bolts[i], nuts[j], PAIR_COLORS[i], True)

    for i, bolt in enumerate(bolts):
        color = PAIR_COLORS[i] if i in selected_bolts else MUTED
        box(bolt, color)
        x, y = position(bolt["bounds"]["x0"], bolt["bounds"]["y0"])
        axis.annotate(str(i + 1), xy=(x, y), xytext=(0, 3), textcoords="offset points", va="bottom",
                      weight="semibold", fontsize=10, color="white", zorder=5,
                      bbox={"boxstyle": "round,pad=.22", "fc": color, "ec": "white", "lw": .8})
    for j, nut in enumerate(nuts):
        box(nut, color_by_nut.get(j, MUTED))

    figure.text(.503, .804, "Distances mesurées", color=INK, fontsize=16, fontweight="semibold")
    figure.text(.503, .747, "De centre à centre, en cases", color=SECONDARY, fontsize=11)
    figure.text(.503, .641, "Boulon", color=SECONDARY, fontsize=10.5)
    targets_found = pairs == intended_pairs
    figure.text(.620, .641, "Écrou retenu" if targets_found else "Écrou attendu", color=SECONDARY, fontsize=10.5)
    figure.text(.807, .641, "Leurre", color=SECONDARY, fontsize=10.5)

    def french(value):
        return f"{value:.2f}".replace(".", ",")

    for rank, row in enumerate(sorted(controls, key=lambda item: item["boltIndex"])):
        i = row["boltIndex"]
        y = .539 - rank * .132
        color = PAIR_COLORS[i] if i in selected_bolts else MUTED
        figure.text(.503, y, str(i + 1), fontsize=15, fontweight="semibold", color=color)
        figure.text(.620, y, french(row["targetDistance"]), fontsize=15, color=INK)
        figure.text(.807, y, french(row["decoyDistance"]), fontsize=15, color=INK)

    detected_label = "Paire repérée" if len(pairs) == 1 else "Paires repérées"
    figure.text(.503, .089, f"{len(pairs)} {detected_label.lower()}" if pairs else "Aucune paire repérée",
                color=SECONDARY, fontsize=11)
    figure.savefig(destination or OUTPUTS / "sphere-assemblages-multiples-resultat.png",
                   dpi=160, facecolor=BACKGROUND)
    plt.close(figure)


if __name__ == "__main__":
    source = OUTPUTS / "sphere-assemblages-multiples-mesures.json"
    render(json.loads(source.read_text(encoding="utf-8")))
