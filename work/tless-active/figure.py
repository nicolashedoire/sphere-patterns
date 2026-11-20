"""Show a successful resolution and a false alarm from the active-view trial."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiment import get_projection, load_depth, models, probe_indices, read, sampled_cloud

BLUE = "#55c7fb"
YELLOW = "#ffd077"


def build(report, manifest, model_file, output):
    trials = read(report)["cases"]
    success = next(row for row in trials if row["verdicts"]["one_view"]["result"] == "abstained"
                   and row["verdicts"]["active_probe"]["result"] == "correct")
    failure = max((row for row in trials if row["verdicts"]["one_view"]["result"] == "abstained"
                   and row["verdicts"]["active_probe"]["result"] == "falseBox"),
                  key=lambda row: abs(row["verdicts"]["active_probe"]["decisionMargin"]))
    cameras = {row["id"]: row for row in read(manifest)}
    meshes, render = models(model_file)
    canvas = Image.new("RGB", (1240, 1050), "#111a2b")
    draw = ImageDraw.Draw(canvas)
    main = ImageFont.load_default(size=34)
    label = ImageFont.load_default(size=22)
    caption = ImageFont.load_default(size=18)
    draw.text((28, 18), "Choosing a second view of a mechanical part", font=main, fill="#f1f5ff")
    draw.text((28, 65), "Yellow marks the initial candidate. Blue dots mark planned depth samples.", font=label, fill="#b9c9e4")

    for index, row in enumerate((success, failure)):
        y = 111 + index * 461
        origin = cameras[row["origin"]]
        peer = cameras[row["choices"]["active"]]
        verdict = row["verdicts"]["active_probe"]
        parts = [item["part"] for item in row["hypotheses"]]
        status = (f"Resolved part {verdict['part']} correctly" if row["actual"] is not None
                  else f"False alarm for part {verdict['part']}")
        draw.text((28, y), f"{status}  ·  Initial hypotheses {parts[0]} and {parts[1]}", font=label, fill="#f1f5ff")
        h, w = 390, 520
        for x, view, title in ((28, origin, "First real view"), (632, peer, "Selected real view")):
            with Image.open(manifest.parent / view["rgb"]) as photo:
                source_w, source_h = photo.size
                canvas.paste(photo.convert("RGB").resize((w, h), Image.Resampling.LANCZOS), (x, y + 32))
            sx, sy = w / source_w, h / source_h
            if view is origin:
                bx, by, bw, bh = row["box"]
                draw.rectangle((x + bx * sx, y + 32 + by * sy, x + (bx + bw) * sx, y + 32 + (by + bh) * sy),
                               outline=YELLOW, width=3)
            else:
                depth = load_depth(origin, manifest.parent)
                points = sampled_cloud(depth, origin["camera"])
                projection = get_projection(row["hypotheses"], peer, points, depth.shape, meshes, render)
                ys, xs = probe_indices(projection)
                for py, px in zip(projection.rows[ys], projection.cols[xs], strict=True):
                    cx, cy = x + px * sx, y + 32 + py * sy
                    draw.ellipse((cx - 1.6, cy - 1.6, cx + 1.6, cy + 1.6), fill=BLUE)
            draw.text((x, y + 429), f"{title} · {view['id']}", font=caption, fill="#b9c9e4")
    draw.text((28, 1023), "T-LESS RGB-D photos by Tomas Hodan et al. · CC BY 4.0", font=caption, fill="#b9c9e4")
    canvas.save(output, quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "manifest", "models", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    build(args.report, args.manifest, args.models, args.output)


if __name__ == "__main__":
    main()
