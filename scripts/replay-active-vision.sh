#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'Usage: %s TRAIN_JSON PANEL_DIR MODELS_ZIP OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${SPHERE_PHOTO_PYTHON:-python3}"
training="$1"
panel="$2"
meshes="$3"
trial="$4"
active="$repo/work/tless-active"
mkdir -p -- "$trial"

"$python" "$repo/work/sphere-tless-rgbd.py" --training "$training" --scenes "$panel/origins.json" --output "$trial/rgbd.json"
"$python" "$repo/work/tless-components/depth_components.py" --training "$training" --scenes "$panel/origins.json" \
  --output "$trial/components.json"
"$python" - "$trial" <<'PY'
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1])
original = json.loads((folder / 'rgbd.json').read_text(encoding='utf-8'))['frames']
components = json.loads((folder / 'components.json').read_text(encoding='utf-8'))['frames']
frames = []
for first, second in zip(original, components, strict=True):
    if first['id'] != second['id']:
        raise ValueError('Proposal lists disagree on the source view')
    frames.append({'id': first['id'], 'detections': first['detections'] + second['detections']})
(folder / 'seeds.json').write_text(json.dumps({'frames': frames}), encoding='utf-8')
PY
"$python" "$repo/work/tless-cad/cad_verify.py" --frames "$panel" --manifest origins.json --models "$meshes" \
  --scan-proposals 18 --predictions "$trial/seeds.json" --output "$trial/verified.json"

# The planner receives camera metadata for all views and only source depth files.
planning="$trial/planning-inputs"
mkdir -p -- "$planning/depth"
cp -- "$panel/views.json" "$panel/origins.json" "$planning/"
"$python" - "$panel" "$planning" <<'PY'
import json
from pathlib import Path
import sys

panel, planning = map(Path, sys.argv[1:])
for frame in json.loads((planning / 'origins.json').read_text(encoding='utf-8')):
    relative = frame['depth']
    link = planning / relative
    source = (panel / relative).resolve()
    if link.is_symlink() and link.resolve() == source:
        continue
    link.symlink_to(source)
PY
"$python" "$active/experiment.py" plan --origins "$planning/origins.json" --manifest "$planning/views.json" \
  --verified "$trial/verified.json" --models "$meshes" --output "$trial/plans.json"
"$python" "$active/experiment.py" reveal --plans "$trial/plans.json" --manifest "$panel/views.json" \
  --models "$meshes" --output "$trial/predictions.json"
"$python" "$active/fetch.py" labels "$panel/origins.json" "$trial/labels.json"
"$python" "$active/experiment.py" evaluate --predictions "$trial/predictions.json" --labels "$trial/labels.json" \
  --output "$trial/measurements.json"
node "$active/wave-readout.cjs" "$trial/predictions.json" "$trial/wave.json"
