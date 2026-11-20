#!/usr/bin/env python3
"""Apply the CAD acceptance policy frozen on the disjoint development panel."""
import argparse
import json
from pathlib import Path

from cad_verify import box_iou


POLICY = {
    "minimumFit": 0.40,
    "minimumMargin": 0.04,
    "maximumContinuation": 0.19,
    "maximumContradiction": 0.05,
    "suppressionIoU": 0.35,
    "marginWeight": 0.5,
}


def classify(proposals):
    rows = []
    for row in proposals:
        if row.get("reference") == "cad-depth-scan":
            continue
        evidence = row["cad"]
        target = evidence["target"]
        fit, margin = target["metricFit"], evidence["metricMargin"]
        plausible = (fit >= POLICY["minimumFit"] and margin >= POLICY["minimumMargin"]
                     and target["continuationAt7Mm"] <= POLICY["maximumContinuation"]
                     and target["contradictedAt9Mm"] <= POLICY["maximumContradiction"])
        rows.append({
            **row,
            "baselineScore": row["score"],
            "baselineBox": row["box"],
            "baselineAccepted": row.get("accepted"),
            "box": target["visibleBox"],
            "score": round(fit + POLICY["marginWeight"] * margin, 6),
            "accepted": plausible,
        })
    detections = sorted(rows, key=lambda candidate: -candidate["score"])
    accepted = []
    for row in detections:
        if row["accepted"] and all(box_iou(row["box"], previous) < POLICY["suppressionIoU"] for previous in accepted):
            accepted.append(row["box"])
        else:
            row["accepted"] = False
    return detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.predictions.read_text(encoding="utf-8"))
    results = {
        "detector": "RGB-D proposal + competing CAD depth fits",
        "modelSource": source["modelSource"],
        "policy": POLICY,
        "frames": [{"id": frame["id"], "detections": classify(frame["detections"])} for frame in source["frames"]],
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
