"""Evaluate full-frame predictions against annotations in a separate step."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VISIBILITY = 0.20
MIN_IOU = 0.50
COLORS = {"truth": "#37e4ad", "hit": "#74bfff", "error": "#ffbd70"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = w * h
    return intersection / (aw * ah + bw * bh - intersection) if intersection else 0


def predictions(data):
    frames = data if isinstance(data, list) else data["frames"]
    threshold = None if isinstance(data, list) else data.get("threshold")
    output = {}
    for row in frames:
        candidates = row.get("detections")
        if candidates is None:
            candidates = ([row] if row.get("bbox") else []) + row.get("alternatives", [])
        clean = []
        for item in candidates:
            box = item.get("bbox", item.get("box"))
            score = float(item["score"])
            assert len(box) == 4 and all(math.isfinite(x) for x in box) and min(box[2:]) > 0
            assert math.isfinite(score)
            accepted = item.get("accepted", threshold is None or score >= threshold)
            clean.append({"box": box, "score": score, "accepted": bool(accepted),
                          **{k: item[k] for k in ("reference", "rotationDegrees", "nominalSide") if k in item}})
        assert row["id"] not in output
        output[row["id"]] = sorted(clean, key=lambda item: -item["score"])
    return output


def match(rows, truth, box_key):
    relevant = [target for target in truth if target["visibility"] >= VISIBILITY]
    obscured = [target for target in truth if target["visibility"] < VISIBILITY]
    available = {target["index"]: target[box_key] for target in relevant}
    rated = []
    for row in rows:
        best = max(((iou(row["box"], box), index) for index, box in available.items()), default=(0, -1))
        nearest = max((iou(row["box"], item[box_key]) for item in relevant), default=0)
        if best[0] >= MIN_IOU:
            del available[best[1]]
            result = "hit"
        elif nearest < MIN_IOU and any(iou(row["box"], item[box_key]) >= MIN_IOU for item in obscured):
            result = "ignored"
        else:
            result = "false-alarm"
        rated.append({**row, "result": result, "nearestTargetIoU": round(nearest, 6),
                      "targetIndex": best[1] if result == "hit" else None})
    return rated


def summarize(frames):
    def entries(label):
        return [row for frame in frames for row in frame["accepted"] if row["result"] == label]
    matched = entries("hit")
    alarms = entries("false-alarm")
    present = [frame for frame in frames if frame["eligible"]]
    absent = [frame for frame in frames if frame["band"] == "absent"]
    bands = {}
    for band in ("high", "partial"):
        rows = [frame for frame in present if frame["band"] == band]
        bands[band] = {"photos": len(rows), "rankOneHits": sum(frame["rankOneHit"] for frame in rows),
                       "eligible": sum(frame["eligible"] for frame in rows),
                       "hits": sum(row["result"] == "hit" for frame in rows for row in frame["accepted"])}
    return {
        "positivePhotos": len(present), "rankOneHits": sum(frame["rankOneHit"] for frame in present),
        "photosWithAcceptedHit": sum(any(row["result"] == "hit" for row in f["accepted"]) for f in present),
        "eligibleInstances": sum(frame["eligible"] for frame in frames),
        "hits": len(matched), "falseAlarms": len(alarms), "ignored": len(entries("ignored")),
        "absentPhotos": len(absent),
        "absentPhotosWithAlarm": sum(any(row["result"] == "false-alarm" for row in f["accepted"]) for f in absent),
        "bands": bands,
    }


def evaluate(rows, truth, box_key):
    assert set(rows) == {frame["id"] for frame in truth}
    frames = []
    for frame in truth:
        ranked = rows[frame["id"]]
        targets = frame["targets"]
        first = match(ranked[:1], targets, box_key)
        accepted = match([item for item in ranked if item["accepted"]], targets, box_key)
        frames.append({**frame, "eligible": sum(t["visibility"] >= VISIBILITY for t in targets),
                       "rankOneHit": bool(first and first[0]["result"] == "hit"),
                       "rankOne": first[0] if first else None, "accepted": accepted})
    return {"summary": summarize(frames), "frames": frames}


def wave_predictions(data, frame_ids, policy):
    predictions_by_frame = {label: [] for label in frame_ids}
    for item in data["results"]:
        verdict = item[policy]["verdict"]
        predictions_by_frame[item["id"]].append({
            "box": item["box"], "score": verdict["score"], "accepted": verdict["accepted"],
            "reference": item["reference"],
        })
    return {key: sorted(rows, key=lambda item: -item["score"]) for key, rows in predictions_by_frame.items()}


def plot(data, scenes, destination, preferred=None):
    methods = data["methods"]
    title = ImageFont.load_default(size=29)
    label = ImageFont.load_default(size=21)
    note = ImageFont.load_default(size=17)
    base = Image.new("RGB", (1240, 1040), "#f0f4f4")
    draw = ImageDraw.Draw(base)
    draw.text((36, 24), "Finding a real part in a full photo", fill="#132e3b", font=title)
    draw.text((36, 65), "T-LESS object 1 · The detector receives the full camera frame", fill="#395660", font=label)
    frames = {f["id"]: scenes.parent / f["rgb"] for f in read(scenes)}
    preferred = preferred or max(methods, key=lambda method: methods[method]["visible"]["summary"]["hits"])
    display = methods[preferred]["visible"]["frames"]
    cases = []
    for band in ("high", "partial"):
        cohort = [frame for frame in display if frame["band"] == band]
        best = max(cohort, key=lambda frame: (sum(r["result"] == "hit" for r in frame["accepted"]),
                                             bool(frame["rankOneHit"]),
                                             (frame["rankOne"] or {}).get("nearestTargetIoU", 0)))
        cases.append(best)
    cohort = [frame for frame in display if frame["band"] != "absent" and frame["id"] not in {x["id"] for x in cases}]
    cases.append(min(cohort, key=lambda frame: (sum(r["result"] == "hit" for r in frame["accepted"]),
                                               (frame["rankOne"] or {}).get("nearestTargetIoU", 0))))
    absent = [frame for frame in display if frame["band"] == "absent"]
    cases.append(max(absent, key=lambda frame: (len(frame["accepted"]), (frame["rankOne"] or {}).get("score", 0))))

    for index, frame in enumerate(cases):
        x = 36 + (index % 2) * 608
        y = 115 + (index // 2) * 440
        with Image.open(frames[frame["id"]]) as opened:
            photo = opened.convert("RGB")
        thumb = photo.resize((556, 417), Image.Resampling.LANCZOS)
        thumb.thumbnail((556, 367))
        base.paste(thumb, (x, y))
        sx, sy = thumb.width / photo.width, thumb.height / photo.height

        def rectangle(box, color, width):
            bx, by, bw, bh = box
            draw.rectangle((x + sx * bx, y + sy * by, x + sx * (bx + bw), y + sy * (by + bh)),
                           outline=color, width=width)

        for target in frame["targets"]:
            if target["visibility"] >= VISIBILITY:
                rectangle(target["visibleBox"], COLORS["truth"], 1)
        proposals = frame["accepted"][:5]
        if not proposals and frame["rankOne"]:
            rectangle(frame["rankOne"]["box"], COLORS["error"], 2)
        for prediction in proposals:
            if prediction["result"] != "ignored":
                rectangle(prediction["box"], COLORS["hit" if prediction["result"] == "hit" else "error"], 3)
        hits = sum(row["result"] == "hit" for row in frame["accepted"])
        false = sum(row["result"] == "false-alarm" for row in frame["accepted"])
        draw.text((x, y + 375), f"{frame['id']} · {frame['band']} · {hits} hit / {false} false alarms",
                  fill="#132e3b", font=label)
        if not proposals and frame["rankOne"]:
            draw.text((x, y + 402), "Outline shows the top proposal below the acceptance cutoff",
                      fill="#395660", font=note)
    draw.text((36, 1000), f"{preferred} · Mint = annotation · Blue = match · Amber = false alarm or rejected proposal",
              fill="#395660", font=note)
    base.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--method", nargs=2, action="append", required=True, metavar=("NAME", "PREDICTIONS"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--wave", type=Path)
    parser.add_argument("--display", help="Select the method shown in the figure.")
    args = parser.parse_args()
    annotations = read(args.truth)
    methods = {}
    for name, filename in args.method:
        value = predictions(read(Path(filename)))
        methods[name] = {"visible": evaluate(value, annotations["frames"], "visibleBox"),
                         "amodal": evaluate(value, annotations["frames"], "amodalBox")}
    if args.wave:
        wave = read(args.wave)
        ids = [frame["id"] for frame in annotations["frames"]]
        for policy, label in (("strictPose", "RGB-D + wave, strict pose"), ("locationOnly", "RGB-D + wave, location")):
            value = wave_predictions(wave, ids, policy)
            methods[label] = {"visible": evaluate(value, annotations["frames"], "visibleBox"),
                              "amodal": evaluate(value, annotations["frames"], "amodalBox")}

    dump = {
        "source": annotations["source"], "target": annotations["target"],
        "protocol": {"visibleFractionAtLeast": VISIBILITY, "boxIoUAtLeast": MIN_IOU,
                     "primary": "visible boxes", "secondary": "amodal boxes",
                     "negatives": ("full frames without the target in the declared scene groups"
                                   if "split" in annotations else "four full frames without the target")},
        "methods": methods,
    }
    args.output.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    if args.figure:
        plot(dump, args.scenes, args.figure, args.display)
    for name, values in methods.items():
        print(name, json.dumps({key: result["summary"] for key, result in values.items()}))


if __name__ == "__main__":
    main()
