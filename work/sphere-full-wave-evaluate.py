"""Calibrate and evaluate the distant probe on complete photographs."""
import argparse
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
BENCHMARK = runpy.run_path(str(HERE / "sphere-tless-photos-evaluate.py"))
ISOLATED_INPUTS = runpy.run_path(str(HERE / "sphere-full-wave-inputs.py"))
read = BENCHMARK["read"]


def scored(data, annotations, threshold):
    values = BENCHMARK["predictions"]({"threshold": threshold, "frames": data["frames"]})
    return BENCHMARK["evaluate"](values, annotations["frames"], "visibleBox")


def calibrate(data, annotations):
    levels = {1.01, *(hit["score"] for frame in data["frames"] for hit in frame["detections"])}
    candidates = []
    for level in sorted(levels):
        summary = scored(data, annotations, level)["summary"]
        if summary["falseAlarms"] <= 2 and summary["absentPhotosWithAlarm"] <= 1:
            candidates.append((summary["hits"], -summary["falseAlarms"], -summary["absentPhotosWithAlarm"], level))
    return max(candidates)[-1]


def scope(data):
    frames = data["frames"]
    return {"frameCount": len(frames), "raster": data["frameSize"], "referencePairs": data["referencePairs"],
            "delayFactor": data["delayFactor"], "loopTicks": data["loopTicks"],
            "probe": {"referenceSHA256": sorted({row["probe"]["cSHA256"] for row in frames}),
                      "photoSamplesPerFrame": sorted({row["probe"]["sceneSamples"] for row in frames}),
                      "maxEnergyError": max(row["probe"]["energyError"] for row in frames),
                      "framesWithOutput": sum(row["probe"]["resultSamples"] > 0 for row in frames)}}


def validate_cut(connected, detached):
    plugged = {row["id"]: row for row in connected["frames"]}
    assert set(plugged) == {row["id"] for row in detached["frames"]}
    for frame in detached["frames"]:
        before = plugged[frame["id"]]["probe"]
        after = frame["probe"]
        assert before["cSHA256"] == after["cSHA256"] and before["sceneSamples"] == after["sceneSamples"]
        assert not frame["detections"] and not after["resultSamples"]
    return scope(detached)


def isolated_truth(calibration, data):
    left, top, _, _ = ISOLATED_INPUTS["PADDING"]
    allowed = ISOLATED_INPUTS["ISOLATED_IDS"]
    training = {frame for pair in data["referencePairs"] for frame in pair}
    frames = []
    for row in read(calibration):
        if row["id"] not in allowed:
            continue
        part, frame = map(int, row["id"].split("-")[1:])
        assert part != 1 or frame not in training
        targets = [{"index": i, "visibility": 1., "visibleBox": [left + x, top + y, w, h]}
                   for i, (x, y, w, h) in enumerate(row["boxes"])]
        frames.append({"id": row["id"], "band": "high" if targets else "absent", "targets": targets})
    return {"frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "development-truth", "heldout", "heldout-truth", "cut", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--synthetic-cut", type=Path, required=True)
    parser.add_argument("--isolated", type=Path, required=True)
    parser.add_argument("--isolated-truth", type=Path, required=True)
    args = parser.parse_args()
    dev, dev_truth = read(args.development), read(args.development_truth)
    hold, hold_truth = read(args.heldout), read(args.heldout_truth)
    threshold = calibrate(dev, dev_truth)
    control, control_cut = read(args.synthetic), read(args.synthetic_cut)
    assert BENCHMARK["iou"](control["frames"][0]["detections"][0]["box"], [94, 29, 25, 25]) >= .50
    isolated = read(args.isolated)
    truth = isolated_truth(args.isolated_truth, isolated)
    rank = scored(isolated, truth, 0.)
    out = {"source": hold_truth["source"], "part": hold_truth["target"], "localizationIoU": .50,
           "minimumVisibility": .20,
           "threshold": threshold, "development": {"fixed": scored(dev, dev_truth, threshold),
                                                    "topFive": scored(dev, dev_truth, 0.)["summary"]},
           "heldout": {"fixed": scored(hold, hold_truth, threshold), "topFive": scored(hold, hold_truth, 0.)},
           "isolated": {"topFive": rank, "transport": scope(isolated)},
           "transport": scope(hold), "cut": validate_cut(hold, read(args.cut)),
           "synthetic": {"connected": control["frames"][0], "disconnected": validate_cut(control, control_cut)}}
    args.output.write_text(json.dumps(out, indent=2) + '\n', encoding="utf-8")
    print(f"Decision threshold {threshold:.6f}")
    for panel in ("development", "heldout"):
        fixed = out[panel]["fixed"]["summary"]
        ranked = out[panel]["topFive"]
        if panel == "heldout":
            ranked = ranked["summary"]
        print(f"{panel}: accepted {fixed['hits']}/{fixed['eligibleInstances']}, {fixed['falseAlarms']} false boxes. "
              f"Top one {ranked['rankOneHits']}, top five {ranked['hits']} with {ranked['falseAlarms']} false boxes.")
    summary = rank["summary"]
    print(f"Isolated: rank one {summary['rankOneHits']}/{summary['eligibleInstances']}, top five {summary['hits']}.")


if __name__ == "__main__":
    main()
