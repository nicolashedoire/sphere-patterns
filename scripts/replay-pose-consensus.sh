#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ( $# -eq 4 && "$4" != --dense-real ) ]]; then
  printf 'Usage: %s MULTIVIEW_PANEL_DIR ISOLATED_REFERENCE_DIR OUTPUT_DIR [--dense-real]\n' "$0" >&2
  exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${SPHERE_PHOTO_PYTHON:-python3}"
panel="$1"
references="$2"
trial="$3"
shift 3

"$python" "$repo/work/sphere-pose-inputs.py" --panel "$panel" --references "$references" \
  --output "$trial/inputs.json" --previews "$trial/previews" "$@"
node "$repo/work/sphere-pose-benchmark.cjs" "$trial/inputs.json" "$trial/measurements.json" "$trial/pose-consensus.json"
