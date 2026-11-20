#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'Usage: %s MULTIVIEW_PANEL_DIR PHOTO_TRAINING_DIR OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${SPHERE_PHOTO_PYTHON:-python3}"
panel="$1"
training="$2"
trial="$3"
mkdir -p -- "$trial"

encode="$repo/work/sphere-full-wave-inputs.py"
run="$repo/work/sphere-full-wave-run.cjs"
"$python" "$encode" --scenes "$panel/development.json" --reference-cache "$trial/reference-photos" \
  --output "$trial/development-inputs.json"
"$python" "$encode" --scenes "$panel/heldout.json" --reference-cache "$trial/reference-photos" \
  --references "$trial/development-inputs.json" --output "$trial/heldout-inputs.json"
"$python" "$encode" --isolated "$training/calibration.json" --reference-cache "$trial/reference-photos" \
  --references "$trial/development-inputs.json" --output "$trial/isolated-inputs.json"
"$python" "$encode" --synthetic --reference-cache "$trial/reference-photos" \
  --references "$trial/development-inputs.json" --output "$trial/synthetic-inputs.json"

node "$run" "$trial/development-inputs.json" "$trial/development-probes.json"
node "$run" "$trial/heldout-inputs.json" "$trial/heldout-probes.json"
node "$run" "$trial/heldout-inputs.json" "$trial/heldout-cut-probes.json" cut
node "$run" "$trial/isolated-inputs.json" "$trial/isolated-probes.json"
node "$run" "$trial/synthetic-inputs.json" "$trial/synthetic-probes.json"
node "$run" "$trial/synthetic-inputs.json" "$trial/synthetic-cut-probes.json" cut

"$python" "$repo/work/sphere-full-wave-evaluate.py" --development "$trial/development-probes.json" \
  --development-truth "$panel/development-truth.json" --heldout "$trial/heldout-probes.json" \
  --heldout-truth "$panel/heldout-truth.json" --cut "$trial/heldout-cut-probes.json" \
  --isolated "$trial/isolated-probes.json" --isolated-truth "$training/calibration.json" \
  --synthetic "$trial/synthetic-probes.json" --synthetic-cut "$trial/synthetic-cut-probes.json" \
  --output "$trial/measurements.json"
