"""Accept depth components after a target fit and a competing-part fit."""
import argparse
import json
from pathlib import Path

THRESHOLD = .315316
COMPETITOR_WEIGHT = .50


def apply_policy(data):
    frames = []
    for frame in data["frames"]:
        detections = []
        for proposal in frame["detections"]:
            fit = proposal["verification"]["targetFit"]
            competitor = proposal["verification"]["competitorFit"]
            score = round(fit["fit"] - COMPETITOR_WEIGHT * competitor["fit"], 6) if fit and competitor else -1.
            detections.append({**proposal, "denseScore": proposal["score"], "score": score,
                               "accepted": score >= THRESHOLD})
        frames.append({**frame, "detections": sorted(detections, key=lambda row: (-row["score"], *row["box"]))})
    return {"detector": "depth-components-and-competitor-adjusted-dense-fit", "threshold": THRESHOLD,
            "competitorWeight": COMPETITOR_WEIGHT, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    findings = apply_policy(json.loads(args.verification.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
