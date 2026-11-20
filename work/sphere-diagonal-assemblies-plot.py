"""Show measured diagonal pairings using the same scales as the previous figure."""

import json
from pathlib import Path
import runpy


WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"


if __name__ == "__main__":
    shared = runpy.run_path(str(WORK / "sphere-multiple-assemblies-plot.py"))
    source = OUTPUTS / "sphere-diagonal-assemblies-measurements.json"
    destination = OUTPUTS / "sphere-diagonal-assemblies-result.png"
    shared["render"](json.loads(source.read_text(encoding="utf-8")), destination,
                     "Match bolts at oblique angles")
