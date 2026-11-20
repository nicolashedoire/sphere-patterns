#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'Usage: %s REFERENCE_DIR SCENE_MANIFEST TRUTH_JSON OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${SPHERE_PHOTO_PYTHON:-python3}"
refs="$1"
scenes="$2"
truth="$3"
trial="$4"
mkdir -p -- "$trial"

"$python" "$repo/work/sphere-tless-rgbd.py" --training "$refs/train.json" --calibration "$refs/calibration.json" \
  --scenes "$scenes" --output "$trial/rgbd.json"
"$python" "$repo/work/sphere-tless-probe-fixtures.py" --training "$refs/train.json" \
  --predictions "$trial/rgbd.json" --output "$trial/probes.json"
node "$repo/work/sphere-tless-legacy-gate.cjs" "$trial/probes.json" "$trial/rgbd.json" "$trial/legacy.json"
"$python" "$repo/work/tless-components/depth_components.py" --training "$refs/train.json" --scenes "$scenes" \
  --output "$trial/components.json"
"$python" "$repo/work/tless-components/verify.py" --training "$refs/train.json" --competitors "$refs/competitors.json" \
  --calibration "$refs/calibration.json" --scenes "$scenes" --proposals "$trial/components.json" \
  --threshold 0.60 --output "$trial/verified.json"
"$python" "$repo/work/tless-components/verify.py" --training "$refs/train.json" --competitors "$refs/competitors.json" \
  --calibration "$refs/calibration.json" --scenes "$scenes" --proposals "$trial/rgbd.json" \
  --threshold 0.60 --output "$trial/old-proposal-check.json"
"$python" "$repo/work/sphere-tless-photos-evaluate.py" --truth "$truth" --scenes "$scenes" \
  --method 'Returning wave reference' "$trial/legacy.json" --method 'Single-view RGB-D' "$trial/rgbd.json" \
  --method 'Original proposals + reference check' "$trial/old-proposal-check.json" \
  --method 'Depth components + reference check' "$trial/verified.json" --output "$trial/evaluation.json" \
  --figure "$trial/figure.png" --display 'Depth components + reference check'
