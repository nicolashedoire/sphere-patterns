# CAD depth checks

These two experimental verifiers use T-LESS CAD meshes and calibrated depth in millimeters. The archive comes from Tomáš Hodaň and colleagues under CC BY 4.0. [Pinned mesh archive](https://huggingface.co/datasets/bop-benchmark/tless/resolve/5fd309a04476a842d93abfb584fba9ee7caecdf1/tless_models.zip)

The archive SHA-256 is `6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928`. Dependencies are Python with NumPy, SciPy, and Pillow, and a C compiler named `cc`. The scripts generate a render cache and a shared library adjacent to the source.

For the single-view trial, run `cad_verify.py` with a panel directory, manifest filename, original RGB-D proposal JSON, and the model archive. Run `cad_gate.py --predictions VERIFIED_JSON --output ACCEPTED_JSON` to apply the frozen thresholds.

For the multi-view trial, add `--scan-proposals 18` to `cad_verify.py` to add independent CAD seeds. Run `cad_consensus.py` with the full manifest path, verified predictions and model archive. Run `cad_consensus_gate.py` with the same manifest and the resulting hypotheses. The multi-view method leaves the world pose fixed when projecting into each camera and uses the measured depth at the projected location. [Trial results](../../RESEARCH.md#earlier-trials-and-selection-of-the-cutoff)
