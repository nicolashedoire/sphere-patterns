#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
trial="$(mktemp -d "${TMPDIR:-/tmp}/sphere-motifs.XXXXXX")"
trap 'rm -rf -- "$trial"' EXIT

cp -R "$repo/work" "$trial/work"
cp -R "$repo/outputs" "$trial/outputs"
cd -- "$trial"

for test in test-sphere-kernel test-sphere-rules test-sphere-separated-lanes; do
  node "work/$test.cjs" > /dev/null
done
printf '%s\n' 'Contrôles du réseau et des voies séparées validés.'

replay() {
  local runner="$1"
  local report="$2"
  shift 2
  rm -f -- "outputs/$report"
  node "work/$runner.cjs" "$@" > /dev/null
  if ! cmp -s "outputs/$report" "$repo/outputs/$report"; then
    printf 'Rapport différent : %s\n' "$report" >&2
    return 1
  fi
  printf 'Rapport identique : %s\n' "$report"
}

replay sphere-motif-compose-run sphere-croisement-motifs-mesures.json
replay sphere-motif-decalage-run sphere-motifs-decales-mesures.json
replay sphere-motif-retard-interne-run sphere-decalage-interne-mesures.json
replay sphere-motif-rotation-interne-run sphere-rotation-interne-mesures.json
replay sphere-reference-apprise-run sphere-reference-apprise-mesures.json
replay sphere-deux-references-run sphere-deux-references-mesures.json
replay sphere-mecanique-run sphere-mecanique-mesures.json
replay sphere-mots-reference-run sphere-mots-reference-mesures.json
replay sphere-assemblage-run sphere-assemblage-mesures.json
replay sphere-assemblage-photo-run sphere-assemblage-photo-mesures.json
replay sphere-assemblage-photo-run sphere-assemblage-equidistance-mesures.json \
  sphere-assemblage-equidistance-masques.json sphere-assemblage-equidistance-mesures.json \
  sphere-assemblage-equidistance-prompt.txt
for mode in multiples diagonaux; do
  name="sphere-assemblages-$mode"
  replay sphere-assemblage-photo-run "$name-mesures.json" \
    "$name-masques.json" "$name-mesures.json" "$name-prompt.txt" multiple
done

replay sphere-motif-angle-oblique-run sphere-motif-angle-oblique-mesures.json
rm -f -- outputs/sphere-banc-angles-figure.svg
replay sphere-banc-angles-run sphere-banc-angles-mesures.json
cmp outputs/sphere-banc-angles-figure.svg "$repo/outputs/sphere-banc-angles-figure.svg"
printf '%s\n' 'Figure du banc des angles identique.'
