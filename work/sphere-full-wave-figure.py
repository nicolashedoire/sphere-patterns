"""Draw an actual correct proposal and a false alarm from the raw wave ranking."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(report, manifest, destination):
    experiment = read(report)
    frames = experiment["heldout"]["topFive"]["frames"]
    best_match = max(((box["score"], frame, box) for frame in frames for box in frame["accepted"]
                      if box["result"] == "hit"), key=lambda row: row[0])
    best_false = max(((box["score"], frame, box) for frame in frames for box in frame["accepted"]
                      if box["result"] == "false-alarm" and frame["band"] == "absent"), key=lambda row: row[0])
    images = {frame["id"]: manifest.parent / frame["rgb"] for frame in read(manifest)}
    canvas = Image.new("RGB", (1168, 566), "#111a2b")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.load_default(size=31)
    label = ImageFont.load_default(size=21)
    secondary = ImageFont.load_default(size=17)
    draw.text((26, 21), "Proposals from the distant wave probe", font=heading, fill="#eef2ff")
    draw.text((26, 64), "Raw ranking before the cutoff fitted on development photos", font=label, fill="#b5c5df")
    for x, case, title, color in ((26, best_match, "Correct proposal", "#61e4ba"),
                                   (598, best_false, "False proposal in a target-absent photo", "#ffc477")):
        score, frame, proposal = case
        with Image.open(images[frame["id"]]) as photo:
            sx, sy = 544 / photo.width, 408 / photo.height
            canvas.paste(photo.convert("RGB").resize((544, 408), Image.Resampling.LANCZOS), (x, 100))
        bx, by, w, h = proposal["box"]
        draw.rectangle((x + sx * bx, 100 + sy * by, x + sx * (bx + w), 100 + sy * (by + h)),
                       outline=color, width=3)
        rank = frame["accepted"].index(proposal) + 1
        draw.text((x, 517), title, font=label, fill=color)
        draw.text((x, 542), f"{frame['id']} · Rank {rank} · Score {score:.3f}", font=secondary, fill="#b5c5df")
    canvas.save(destination, quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    options = parser.parse_args()
    build(options.report, options.manifest, options.output)


if __name__ == "__main__":
    main()
