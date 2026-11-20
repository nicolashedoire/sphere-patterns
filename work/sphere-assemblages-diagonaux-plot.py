"""Show measured diagonal pairings using the same scales as the previous figure."""

import json
from pathlib import Path
import runpy


WORK = Path(__file__).resolve().parent
OUTPUTS = WORK.parent / "outputs"


if __name__ == "__main__":
    shared = runpy.run_path(str(WORK / "sphere-assemblages-multiples-plot.py"))
    source = OUTPUTS / "sphere-assemblages-diagonaux-mesures.json"
    destination = OUTPUTS / "sphere-assemblages-diagonaux-resultat.png"
    shared["render"](json.loads(source.read_text(encoding="utf-8")), destination,
                     "Associer les boulons en diagonale")
