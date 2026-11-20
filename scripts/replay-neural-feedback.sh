#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'Usage: %s MULTIVIEW_PANEL_DIR ISOLATED_REFERENCE_DIR DINO_WEIGHTS_DIR OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${SPHERE_PHOTO_PYTHON:-python3}"
device="${SPHERE_NEURAL_DEVICE:-auto}"
panel="$1"
references="$2"
weights="$3"
trial="$4"
baseline="$repo/outputs/sphere-pose-consensus-measurements.json"

"$python" "$repo/work/sphere-neural-pose-inputs.py" --panel "$panel" --references "$references" \
  --weights "$weights" --device "$device" --cache "$trial/cache" --output "$trial/coarse.json"
"$python" "$repo/work/sphere-neural-pose-inputs.py" --panel "$panel" --references "$references" \
  --descriptor gradient --device cpu --cache "$trial/cache" --output "$trial/gradient.json"
node "$repo/work/sphere-neural-feedback.cjs" plan "$trial/coarse.json" "$trial/plan.json"
"$python" "$repo/work/sphere-neural-patch-read.py" "$trial/plan.json" "$trial/reads.json" --weights "$weights" --device "$device"
node "$repo/work/sphere-neural-feedback.cjs" score "$trial/plan.json" "$trial/reads.json" \
  "$trial/measurements.json" "$trial/neural-feedback.json" "$baseline" "$trial/gradient.json"
