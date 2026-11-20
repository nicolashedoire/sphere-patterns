#!/usr/bin/env python3
"""Accept CAD hypotheses when two calibrated camera views support the same pose."""
import argparse
import json
from pathlib import Path

from cad_verify import box_iou


POLICY = {
    "meanBestTwoCameraFits": .46,
    "minimumGapFromRival": .13,
    "minimumViewFit": .28,
    "minimumViewSupport": .40,
    "maximumViewContradiction": .15,
    "suppressionIoU": .35,
}


def strongest_pair(values):
    return sum(sorted(values, reverse=True)[:2]) / 2


def classify(manifest, hypotheses):
    candidates = {frame["id"]: [] for frame in manifest}
    for hypothesis in hypotheses:
        projections = hypothesis["projections"]
        if len(projections) < 2:
            continue
        views = sorted(projections, key=lambda key: -projections[key]["score"])
        consensus = strongest_pair(row["score"] for row in projections.values())
        rival = max(strongest_pair(row["scores"].values()) for row in hypothesis["competitors"])
        margin = consensus - rival
        for frame_id, view in projections.items():
            if view["visibleBox"] is None:
                continue
            accepted = (consensus >= POLICY["meanBestTwoCameraFits"] and margin >= POLICY["minimumGapFromRival"]
                        and view["score"] >= POLICY["minimumViewFit"] and view["support"] >= POLICY["minimumViewSupport"]
                        and view["contradiction"] <= POLICY["maximumViewContradiction"])
            candidates[frame_id].append({
                "box": view["visibleBox"], "score": round(consensus + .5 * margin + .4 * view["score"], 6),
                "accepted": accepted, "originFrame": hypothesis["id"], "originProposal": hypothesis["proposalRank"],
                "originReference": hypothesis["reference"], "worldPositionMm": hypothesis["positionMm"],
                "supportingViews": views[:2], "meanFit": round(consensus, 6), "rivalGap": round(margin, 6),
                "view": view,
            })
    for rows in candidates.values():
        rows.sort(key=lambda row: -row["score"])
        kept = []
        for row in rows:
            row["accepted"] = row["accepted"] and all(box_iou(row["box"], x) < POLICY["suppressionIoU"] for x in kept)
            if row["accepted"]:
                kept.append(row["box"])
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.hypotheses.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = classify(manifest, source["hypotheses"])
    report = {
        "detector": "multiview CAD depth reprojection", "modelSource": source["modelSource"], "policy": POLICY,
        "frames": [{"id": label, "detections": rows} for label, rows in candidates.items()],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
