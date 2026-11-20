#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
if [[ $# -gt 1 || ( -n "$mode" && "$mode" != --refresh ) ]]; then
  printf 'Usage: %s [--refresh]\n' "$0" >&2
  exit 2
fi
trial="$(mktemp -d "${TMPDIR:-/tmp}/sphere-patterns.XXXXXX")"
trap 'rm -rf -- "$trial"' EXIT

cp -R "$repo/work" "$trial/work"
cp -R "$repo/outputs" "$trial/outputs"
cd -- "$trial"

for test in test-sphere-kernel test-sphere-rules test-sphere-separated-lanes; do
  node "work/$test.cjs" > /dev/null
done
printf '%s\n' 'Network and separated-lane checks passed.'
reports=()

replay() {
  local runner="$1"
  local report="$2"
  shift 2
  rm -f -- "outputs/$report"
  node "work/$runner.cjs" "$@" > /dev/null
  test -s "outputs/$report"
  reports+=("$report")
  if [[ "$mode" != --refresh ]] && ! cmp -s "outputs/$report" "$repo/outputs/$report"; then
    printf 'Report differs: %s\n' "$report" >&2
    return 1
  fi
  printf 'Reproduced %s\n' "$report"
}

replay sphere-pattern-compose-run sphere-pattern-crossing-measurements.json
replay sphere-pattern-shift-run sphere-shifted-patterns-measurements.json
replay sphere-pattern-internal-delay-run sphere-internal-shift-measurements.json
replay sphere-pattern-internal-rotation-run sphere-internal-rotation-measurements.json
replay sphere-learned-reference-run sphere-learned-reference-measurements.json
replay sphere-two-references-run sphere-two-references-measurements.json
replay sphere-mechanical-part-run sphere-mechanical-part-measurements.json
replay sphere-word-reference-run sphere-word-reference-measurements.json
replay sphere-assembly-run sphere-assembly-measurements.json
replay sphere-assembly-photo-run sphere-assembly-photo-measurements.json
replay sphere-assembly-photo-run sphere-assembly-equal-distance-measurements.json \
  sphere-assembly-equal-distance-masks.json sphere-assembly-equal-distance-measurements.json \
  sphere-assembly-equal-distance-prompt.txt
for name in sphere-multiple-assemblies sphere-diagonal-assemblies; do
  replay sphere-assembly-photo-run "$name-measurements.json" \
    "$name-masks.json" "$name-measurements.json" "$name-prompt.txt" multiple
done

replay sphere-pattern-oblique-angle-run sphere-pattern-oblique-angle-measurements.json
rm -f -- outputs/sphere-angle-benchmark-figure.svg
replay sphere-angle-benchmark-run sphere-angle-benchmark-measurements.json
replay sphere-occlusion-run sphere-occlusion-measurements.json
rm -f -- outputs/sphere-occlusion-figure.svg
node work/sphere-occlusion-plot.cjs
replay sphere-stress-run sphere-stress-measurements.json
rm -f -- outputs/sphere-stress-figure.svg
node work/sphere-stress-plot.cjs
replay sphere-tless-pilot sphere-tless-measurements.json
figures=(sphere-angle-benchmark-figure.svg sphere-occlusion-figure.svg sphere-stress-figure.svg)
for figure in "${figures[@]}"; do
  test -s "outputs/$figure"
done
if [[ "$mode" == --refresh ]]; then
  for file in "${reports[@]}" "${figures[@]}"; do
    cp -- "outputs/$file" "$repo/outputs/$file"
  done
  printf '%s\n' 'Updated the tracked reports and benchmark figures.'
else
  for figure in "${figures[@]}"; do
    cmp "outputs/$figure" "$repo/outputs/$figure"
  done
  printf '%s\n' 'All tracked reports and benchmark figures match byte for byte.'
fi
