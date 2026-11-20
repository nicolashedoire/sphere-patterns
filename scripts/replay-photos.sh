#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [downloaded-panel-directory]\n' "$0" >&2
  exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
trial="$(mktemp -d "${TMPDIR:-/tmp}/sphere-photos.XXXXXX")"
trap 'rm -rf -- "$trial"' EXIT
python="${SPHERE_PHOTO_PYTHON:-python3}"
if [[ $# -eq 1 ]]; then
  panel="$(cd -- "$1" && pwd)"
else
  panel="$trial/panel"
  "$python" "$repo/work/sphere-tless-photos-fetch.py" "$panel"
fi

"$python" "$repo/work/sphere-tless-rgb.py" --training "$panel/train.json" --scenes "$panel/scenes.json" \
  --output "$trial/rgb.json"
"$python" "$repo/work/sphere-tless-rgb.py" --training "$panel/train.json" --scenes "$panel/scenes.json" \
  --training-limit 1 --output "$trial/rgb-one.json"
"$python" "$repo/work/sphere-tless-rgbd.py" --training "$panel/train.json" --scenes "$panel/scenes.json" \
  --calibration "$panel/calibration.json" --output "$trial/rgbd.json"

"$python" "$repo/work/sphere-tless-probe-fixtures.py" --training "$panel/train.json" \
  --predictions "$trial/rgbd.json" --output "$trial/fixtures.json"
cmp "$trial/fixtures.json" "$repo/outputs/sphere-tless-wave-fixtures.json"
node "$repo/work/sphere-tless-wave-run.cjs" "$trial/wave.json"
cmp "$trial/wave.json" "$repo/outputs/sphere-tless-wave-measurements.json"

"$python" "$repo/work/sphere-tless-photos-evaluate.py" --truth "$panel/truth.json" --scenes "$panel/scenes.json" \
  --method "RGB, 24 views" "$trial/rgb.json" --method "RGB, one view" "$trial/rgb-one.json" \
  --method RGB-D "$trial/rgbd.json" --wave "$trial/wave.json" --display "RGB-D + wave, location" \
  --output "$trial/report.json" --figure "$trial/result.png"
cmp "$trial/report.json" "$repo/outputs/sphere-tless-full-photo-measurements.json"
cmp "$trial/result.png" "$repo/outputs/sphere-tless-full-photo-result.png"

printf '%s\n' 'Full-frame photo measurements, wave fixtures and figure reproduced.'
