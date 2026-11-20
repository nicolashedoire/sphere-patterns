# Research protocol

## What the study can establish

Sphere Patterns tests example-based recognition of a known mechanical part under controlled changes in its image. The network carries the reference formed from two input patterns. A separate comparator searches pose and decides whether the evidence supports a match. Each measurement identifies which information the comparator received.

The wave network and the recognition rule have separate controls. For each new synthetic fixture, run the same comparator with the returned reference, with the identical reference read directly from memory, and with the return disconnected. Return and memory parity establishes that the propagation preserved the reference. To claim a benefit from propagation, measure a difference against the same direct comparator under a defined condition, such as interference between references, memory cost, noise or latency.

The recognition baseline is transformed template matching. Optical matched filtering and gradient-based templates provide relevant precedents. The research contribution to test is the recirculating reference and what several such references can do under interference. [VanderLugt's account of matched filtering](https://garfield.library.upenn.edu/classics1979/A1979HZ20500001.pdf) · [Hinterstoisser et al. on gradient templates](https://doi.org/10.1109/TPAMI.2011.206)

## Difficulty ladder

| Level | Change to the scene | Primary check | Status |
| --- | --- | --- | --- |
| 1 | In-plane translation and rotation | Correct position and angle, with a wrong object and an absent target as controls | Measured with synthetic silhouettes |
| 2 | Scale from 0.75× to 1.25×, surrounding objects, intruding foreground and shape lookalikes | Correct position, angle and scale. Count accepted wrong poses and false alarms separately. | Measured in the wave network |
| 3 | Partial occlusion, including a similar distractor and a target with too little visible evidence | Compare an unknown-visibility run with a separately labeled known-mask control. Report ambiguity as an abstention. | Measured. The three occluded cases in the new stress test remain unresolved without an occlusion mask. |
| 4 | Actual camera views of textureless parts, known candidate boxes | Compare a grayscale mask, a depth mask and the annotated visible mask with the same geometric matcher. | Measured on 24 candidates across six T-LESS scenes |
| 5 | Full camera image, no box or mask supplied | Generate candidates from the image and score both localization and false alarms against hidden annotations. | Measured on 16 real T-LESS camera frames |
| 6 | Out-of-plane rotation, specular reflections, severe overlap and different part families | Compare multiple reference views and local contours or holes, including sets of nearly identical parts and target-absent scenes. | Research target |

Increase one difficulty at a time before combining them. Keep a fixed calibration set for score thresholds, scale and angle search. Reserve whole scenes and capture setups for evaluation. Treat repeated views of one physical scene as related samples. Publish positives, negatives, wrong poses and abstentions by difficulty with their denominators. For real scenes, use annotated boxes and masks as evaluation labels or explicit oracle controls.

## In-sphere stress test

The same returning reference C is used for all arms. Local intersection over union (IoU) penalizes extra foreground inside the proposed template footprint. Expanding the scale bank alone retains the failure mode of foreground-only overlap.

| Comparator | Correct poses out of 8 | Accepted wrong poses | Misses | False alarms out of 4 |
| --- | ---: | ---: | ---: | ---: |
| Fixed scale, foreground overlap | 1 | 3 | 4 | 1 |
| Multiple scales, foreground overlap | 1 | 0 | 7 | 1 |
| Multiple scales, local IoU | 5 | 0 | 3 | 0 |

The local-IoU arm recovers all five intact targets in this set. All three misses are occluded targets. The return travels 8,597 simulation steps. For the scene with the intruding stripe, the distant probe receives 51 reference-cell pulses with the return connected and zero after disconnection. Applying the same comparator directly to the cached probe reference reproduces its scores, selections and output drive in all 12 connected trials. [Stress figure](outputs/sphere-stress-figure.svg) · [Measurements and fixture masks](outputs/sphere-stress-measurements.json)

The separate known-mask diagnostic localizes one target and accepts one wrong pose in this harder panel. The earlier fixed-scale result with known visibility came from a narrower set of scenes. [Fixed-scale occlusion measurements](outputs/sphere-occlusion-measurements.json)

## First real-image pilot

The panel uses T-LESS object 1, two training views and 12 test photos from six different scenes. Each test photo supplies one target crop and one similarly sized distractor crop. The selection uses visibility bands and box dimensions alone, before matching. The visible target fractions range from approximately 90% to 93% in the high band and 50% to 54% in the partial band. [T-LESS](https://cmp.felk.cvut.cz/t-less/) · [BOP annotation format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)

| Proposed candidate input | Correct targets out of 12 | False accepts out of 12 |
| --- | ---: | ---: |
| Grayscale crop, border-contrast silhouette | 0 | 1 |
| Depth crop, 25 mm band around the center depth | 0 | 1 |
| Dataset's annotated visible silhouette | 2 | 3 |

The pilot runs outside the wave network to isolate the image representation. Dataset boxes are supplied to all three arms. The grayscale cutoff is fitted on the two training views. The matcher uses a common IoU cutoff of 0.86. The annotated-mask arm shows that improved segmentation alone does not resolve the lookalikes and changing viewpoint for this reference. [Camera scenes](outputs/sphere-tless-scenes.png) · [Measurements](outputs/sphere-tless-measurements.json)

### Dataset credit and reproduction

The scene figure and fixture crops are derived from *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*, by Tomáš Hodaň and colleagues, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The derivatives include downsampling, crops, luminance and depth grids, visible masks, and box overlays. The source archives are pinned to revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`. [Official dataset archives](https://huggingface.co/datasets/bop-benchmark/tless/tree/5fd309a04476a842d93abfb584fba9ee7caecdf1)

Run `bash scripts/replay.sh` to reproduce measurements and synthetic figures from the checked-in inputs. The optional `python3 work/sphere-tless-fetch.py` command uses Pillow 12 and curl to rebuild the real-image input panel from selected archive entries.

## Full-frame real photos

The detector now receives the full **720 × 540** camera frame. We searched for T-LESS object 1 using 24 real reference views from its official training split. The test panel includes the 12 positive frames from the crop pilot and four additional frames without this object. The positive frames come from six physical scenes. Test boxes and masks are read only by the evaluator. It matches predictions one to one at visible-box IoU ≥ 0.50 for the 35 target instances at least 20% visible. The report also includes amodal-box results. [BOP dataset and annotations](https://bop.felk.cvut.cz/datasets/#t-less)

| Pipeline | Accepted target instances / 35 | False boxes | Target-absent frames with an alarm / 4 |
| --- | ---: | ---: | ---: |
| RGB gradient templates, 24 views | 0 | 80 | 4 |
| RGB gradient templates, one view | 1 | 54 | 4 |
| Full-frame RGB-D proposals | 3 | 7 | 3 |
| RGB-D proposals with returning-reference verification | 2 | 2 | 1 |

The RGB-D threshold was chosen using eight separate official training frames. Its first-ranked candidate intersects a true target in four of the twelve positive photos. After thresholding, all three accepted true instances occur in **one** photo. The RGB and RGB-D algorithms also differ in their proposal search and scoring, so the table compares complete pipelines.

For the wave check, the front end supplies a 25 × 25 measured depth-support raster from each accepted proposal and the selected training-view reference. We inject two aligned copies of that one reference, which lets us test verification independently of viewpoint learning. The score uses the existing local IoU threshold of 0.86. With a unique-pose requirement, the wave comparator rejects both correct candidates with tied fits. In the follow-up on the same panel, it accepts tied fits whose centers are within a two-cell diagonal, leaving the angle undetermined. This retains two correct positions in frame **09-000057** and two false alarms in frame **02-000003**. The full-frame search happens in the RGB-D front end.

The highest-scoring correct window produces **193 pulses at the probe** with the return connected and zero after disconnection. Directly providing the cached reference reproduces both comparator policies. This isolates how the recognition rule and the returning signal contribute to the result. [Interactive photo inspection](https://nicolashedoire.github.io/sphere-patterns/real-photos.html) · [Photo panel](outputs/sphere-tless-full-photo-result.png) · [Detection measurements](outputs/sphere-tless-full-photo-measurements.json) · [Wave comparison](outputs/sphere-tless-wave-measurements.json)

Run `bash scripts/replay-photos.sh` with Node.js 24, Python with Pillow 12, NumPy and SciPy, and curl. It downloads the specific frames and regenerates the full evaluation. Pass a prepared panel directory as the first argument to reuse it. The photos and probe fixtures are derived from the same pinned T-LESS archives and CC BY 4.0 source credited above.

## Depth components on held-out views

We added a geometric front end for a calibrated RGB-D camera. It makes several cuts through the depth image in millimeters, groups connected surfaces, and ranks regions by the measured dimensions of the target. The verifier checks each region against 24 isolated views of the target and 16 isolated views of four similar parts. It compares surface relief, the depth step across the outline, and the direction of interior image edges. Both stages use the full camera frame. They run outside the wave simulator.

The confirmation panel contains **30 previously unused Primesense views**. Twelve views come from three physical setups containing object 1 (scenes 07, 18, and 20), and 18 views come from six setups without it. The photos show new viewpoints of setups used in earlier tests. The detector receives color, depth, intrinsics, and the camera's pose relative to the work surface. Frame selection uses annotation presence and visibility for stratification. The evaluator reads the target boxes after inference and applies one-to-one visible-box IoU ≥ 0.50 with visibility ≥ 0.20. The 33 eligible instances count repeated sightings across views.

| Pipeline on the same 30 views | Correct / 33 | False boxes | Target-absent views with an alarm / 18 |
| --- | ---: | ---: | ---: |
| Original RGB-D + returning-reference check | 1 | 1 | 1 |
| Original RGB-D | 4 | 2 | 1 |
| **Depth components + reference check** | **5** | **0** | **0** |

The five matches are spread across five views. Four are in scene 07, and one is in scene 20. Zero of the three eligible instances in the partial-visibility band were detected. The amodal-box evaluation gives the same counts. The predeclared comparison required at least two additional matches over the published wave cascade, matches in at least two physical setups, and no increase in false boxes or target-absent views with an alarm. This result meets that comparison for this part and camera. [Interactive photo comparison](https://nicolashedoire.github.io/sphere-patterns/real-photos.html?trial=components) · [All predictions and matches](outputs/sphere-tless-component-measurements.json)

A diagnostic replay gave the same reference check and cutoff to the original RGB-D proposals. It produced 8 matches and 3 false boxes. On the final panel, the top 15 component proposals cover 14 eligible instances before verification, while the original top five cover 12. At an equal five-proposal budget the component shortlist covers 9. The measured gain at the chosen operating point is jointly attributable to proposal generation and verification. Most eligible targets are still missed, especially in clutter. The setup also assumes a measured depth map and a calibrated work surface.

### Earlier trials and selection of the cutoff

We used 24 development views across seven setups for iteration. The first held-out trial compared a single-view CAD filter with the published wave check. The CAD filter gave 2 matches and 1 false box versus 1 match and 0 false boxes for the wave check, both CAD matches in one setup. The second trial projected a CAD hypothesis through multiple calibrated views. It yielded 6 matches and 0 false boxes in development, then accepted no boxes on 30 held-out views. Reviewing the failed predictions showed genuine target overlaps with weak margins over competing geometry. Some competing false hypotheses overlapped fragments of larger assemblies.

On the third 30-photo trial, the component detector at its development threshold of 0.587722 gave 7 matches and 2 false boxes. The wave check gave 2 matches and 0 false boxes there. An alternate competitor penalty gave 6 matches and 4 false boxes. For the fourth trial we froze the original reference scoring rule at 0.60, the next hundredth above the third panel's highest false score (0.594272), before running it on the new photos. Each holdout used distinct frame IDs. These sequential trials use the same sensor and reuse physical setups, so scene-level transfer remains to be measured. [CAD trial code](work/tless-cad/) · [Depth-component detector](work/tless-components/)

### Reproduce the confirmation

Run the following from the repository root with Python, NumPy, SciPy, Pillow 12, Node.js 24, Bash, and curl. The fetcher reads only the required entries from the pinned archive. The final replay writes per-method results plus a figure. It uses the cached wave comparator for the original cascade, verified against the actual returning-wave run on all ten candidates from the earlier published experiment.

```sh
data=$(mktemp -d)
python3 work/tless-components/fetch_references.py "$data/refs"
python3 work/sphere-tless-multiview-fetch.py "$data/panel-1"
python3 work/sphere-tless-multiview-fetch.py "$data/panel-2" \
  --confirmation-after "$data/panel-1/heldout.json"
python3 work/sphere-tless-multiview-fetch.py "$data/panel-3" \
  --confirmation-after "$data/panel-1/heldout.json" \
  --confirmation-after "$data/panel-2/confirmation.json"
python3 work/sphere-tless-multiview-fetch.py "$data/panel-4" \
  --confirmation-after "$data/panel-1/heldout.json" \
  --confirmation-after "$data/panel-2/confirmation.json" \
  --confirmation-after "$data/panel-3/confirmation.json"
bash scripts/replay-components.sh "$data/refs" "$data/panel-4/confirmation.json" \
  "$data/panel-4/confirmation-truth.json" "$data/result"
cmp "$data/result/evaluation.json" outputs/sphere-tless-component-measurements.json
```

Photos and CAD data are from the T-LESS authors under CC BY 4.0. Gallery photos were converted to JPEG. The figure also resizes photos and adds overlays. [T-LESS project and attribution](https://cmp.felk.cvut.cz/t-less/)
