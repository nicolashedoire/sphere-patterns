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

## Active camera selection

We tested three ideas on T-LESS RGB-D views. A virtual camera predicts which rays would distinguish two CAD poses. A targeted readout requests a subset of those pixels in the next real depth frame. A streaming decision combines the first score with new evidence when the second measurement arrives. Choosing a view for object recognition is an established active-perception problem. [Active object detection and pose estimation](https://erl.ucsd.edu/ref/Atanasov_ActiveObjectRecognition_TRO14.pdf)

The panel contains 78 new photos, six poses in each of 13 T-LESS physical setups disjoint from the seven development setups. The camera poses were selected by calibration alone, starting at the most overhead view and spreading the remaining camera positions across the viewing hemisphere. Three starting views per setup produced three candidate regions each. The CAD catalogue has nine part IDs. Each pair of hypotheses uses the two strongest independently fitted part types at that location, with both poses estimated from the first depth image. The planner can read the alternative camera calibrations. It saves the camera choices and poses before measuring any second observation for that query.

The active rule ranks cameras by projected depth and silhouette differences, accounting for foreground already measured in the first view. The fixed rule selects the angle closest to 45° relative to the first camera. The random rule uses one seeded choice per query. Each method gets one actual depth view. The sparse readout samples up to 128 projected differences. The dense control samples the visible projected footprint on the same actively selected camera. The decision rule and thresholds are shared across the view methods and were fixed before evaluating the panel. [Full per-query results](outputs/sphere-tless-active-vision-measurements.json)

| Method | Correct | Wrong identity | Incorrect region | Abstained |
| --- | ---: | ---: | ---: | ---: |
| First view only | 4 | 7 | 35 | 71 |
| Fixed second view, targeted probe | 6 | 11 | 59 | 41 |
| Seeded random second view, targeted probe | 7 | 9 | 57 | 44 |
| Actively selected second view, targeted probe | 5 | 13 | 57 | 42 |
| Actively selected second view, dense probe | 6 | 11 | 50 | 50 |

The evaluator matches the proposal from the first frame to a visible annotation with IoU ≥ 0.50 and visibility ≥ 0.20. A region below this matching threshold is scored as incorrect if the system commits to a part. Only 24 of the 117 proposals match eligible boxes, and only 12 include the true identity among their two hypotheses. These are per-proposal classification results. The active policy gains one correct identification over the first view and adds 28 errors. It falls short of both tested second-view controls in correct identifications. The real example in scene 07 resolves part 15 with any of the three selected camera policies. The lower example shows a spurious response on a fragment of a larger assembly. [See both real photo examples](https://nicolashedoire.github.io/sphere-patterns/active-vision.html)

The targeted probe checks 12,650 depth pixels on the selected cameras. The dense probe checks 762,752. This measures requests to the depth array after decoding, and each method still loads a full depth frame and projects CAD models. The sparse readout gives one fewer correct result and nine additional errors on those same selected views.

We injected the first score and the weighted second observation as separate streams in the existing 1,000-junction sphere. The far probe recovered the sum and reproduced the numerical verdict in all 117 cases, including 47 changes between initial and updated verdicts. A control omitting the second stream recovered the original score each time. The largest absolute margin error was 5.56 × 10⁻¹⁷. This test covers transport and summation. Candidate generation, pose fitting and angle choice are geometric computations outside the sphere. [Wave measurements](outputs/sphere-tless-active-vision-wave.json)

### Reproduce the active-view trial

Use the pinned T-LESS RGB-D archive, the CAD archive described in the [CAD notes](work/tless-cad/README.md), Python with NumPy, SciPy and Pillow 12, a C compiler, Node.js 24, Bash and curl. `TRAIN_JSON` points to the isolated target references prepared in the [preceding trial](#reproduce-the-confirmation). The recorded camera IDs restore the exact panel. The replay generates first-view hypotheses, saves plans, opens real second observations, fetches the evaluation annotations, and verifies the wave probe.

```sh
python3 work/tless-active/fetch.py views "$panel" --selection outputs/sphere-tless-active-vision-views.json
bash scripts/replay-active-vision.sh "$train_json" "$panel" "$models_zip" "$run"
cmp "$run/measurements.json" outputs/sphere-tless-active-vision-measurements.json
cmp "$run/wave.json" outputs/sphere-tless-active-vision-wave.json
```

## Full-frame wave search

The full scene is converted to grayscale and uniformly downsampled from 720 × 540 to 144 × 108. An amplitude therefore carries luminance and its time slot fixes its row and column. Twelve pairs of distinct isolated T-LESS training photos are centered from their training boxes. Each pair overlaps at the simulation's coincidence junction, producing one luminance reference in C. The complete bank travels around the return circuit. The original 1,000-junction topology is preserved and the link delays are tripled so that the larger input fits between its earliest echoes. [Signal diagram and photos](https://nicolashedoire.github.io/sphere-patterns/full-wave.html)

A programmed comparator at the junction buffers the received scene and the physically returned C. It scans the scene on a two-pixel grid, with local one-pixel refinement, seven scales and 24 in-plane angles per reference. Its score combines normalized luminance correlation (70%) and signed image gradient correlation (30%). Five locations, scores and reference indices are emitted on a carrier and decoded at the remote probe. The prediction stage gets no region proposals or test annotations. The rotation bank handles in-plane rotations. The distinct training pairs supply several appearances.

The separate evaluator matches visible boxes one to one at IoU ≥ 0.50 with visibility ≥ 0.20. The score cutoff maximizes the number of development hits with at most two false boxes and at most one target-absent frame with an alarm. The 24 development photos and 30 held-out photos use disjoint setups and come from existing study panels. Every cutoff that kept a development hit exceeded the false-alarm budget, so the calibration selected abstention. The top-five rows show what was available before that cutoff.

| Raw ranking and final decisions | Eligible instances | Rank-one matches | Top-five matches | False top-five boxes | Accepted matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seven isolated real photos | 3 | 2 | 3 | 32 | — |
| 24 development photos | 24 | 0 | 1 | 119 | 0 |
| 30 held-out cluttered photos | 29 | 1 | 2 | 148 | 0 |

The isolated control includes three target photos outside the C pairs and four photos of other isolated parts. All photos in this control receive the same black padding before resizing to preserve aspect ratio. The synthetic control places a copy of the paired reference away from the center and includes a second shape. The remote probe localizes it first at score 0.999988. The two held-out matches occur in two views of a single physical setup. A target-absent photo has a higher-scoring region than either match.

An exact transport check verifies all 15,552 scene amplitudes at the junction in each run. The physical C trace is identical in the connected and cut runs. The cut suppresses the comparator's access to C while the reference wave continues to circulate. It reduces raw probe output from 30 of 30 cluttered photos to zero. The recognition rule is programmed at the junction. The result tests signal-dependent localization and its accuracy under clutter. [All measurements](outputs/sphere-full-wave-measurements.json)

Run `bash scripts/replay-full-wave.sh MULTIVIEW_PANEL_DIR PHOTO_TRAINING_DIR OUTPUT_DIR` with Node.js 24, Python and Pillow 12. Prepare the two dataset directories using `work/sphere-tless-multiview-fetch.py` and `work/sphere-tless-photos-fetch.py`. Inspect `measurements.json` in the output directory. T-LESS photos are credited under CC BY 4.0. [Dataset and licence](https://cmp.felk.cvut.cz/t-less/)

## Pose agreement from local details

Encode the entire camera frame as RootSIFT points without candidate boxes, masks or depth. Two isolated training views contribute to each C by retaining aligned local descriptors found in both. Twelve C templates represent the different views of T-LESS object 1. On the synthetic plate, the three C groups correspond to the bore, groove and shoulder. On T-LESS, they are three spatial strips in the reference. The synthetic part rotates in the image plane. T-LESS also varies in viewpoint, occlusion and illumination.

The same three local matchers propose pose bins in all arms. Position spacing is 11 pixels at 720 × 540, angles are 15° apart and scale steps are quarter octaves. Each group emits at most one amplitude per bin. The sphere routes the three streams along distinct paths in the 1,000-junction topology. One stream completes a loop before mixing. An agreed bin means pulses arrive in the same time slot. A probe farther along the network reads the combined amplitudes. The event-driven simulator skips empty ticks and models dedicated, lossless paths with launch times compensated by known link delays. RootSIFT encoding and construction of C run offline. The fixed local matchers supply votes to the network, whose measured computation is their timed summation. The comparator requires at least two groups and uses the average of the three amplitudes, counting absent groups as zero. [3D replay](https://nicolashedoire.github.io/sphere-patterns/pose-consensus.html)

Compare the complete vote vectors before thresholding or scoring. Direct arithmetic, the returning circuit and the compensated feed-forward circuit agree exactly in 86 of 86 scenes. Removing the return link zeroes only that branch. Shifting its launch by one tick also removes it from the clocked readout, with the pulses still present outside their sampling slots. Both interventions change the raw vector in all 86 scenes. The connected return can carry a required signal. Its additional circuit does not improve recognition over a compensated feed-forward route with this model.

The cutoff maximizes accepted matches on each domain's development panel under its false-alarm budget, then uses the midpoint of the adjacent observed scores with the same development verdict. The synthetic budget is zero. The real budget is two false boxes and one target-absent frame with an alarm. An accepted localization must have visible-box IoU of at least 0.50. Targets must be at least 20% visible. The two real panels use disjoint physical setups that have appeared in prior project trials. We checked the initial synthetic panel while settling the cutoff procedure. The additional evaluation panel was generated at a fixed seed before scoring and uses the same rendering family as the training and development examples.

| Evaluation panel | Targets | Connected hits | Connected false alarms | Cut hits |
| --- | ---: | ---: | ---: | ---: |
| Six synthetic controls | 4 | 3 | 0 | 0 |
| 24 additional synthetic controls | 16 | 7 | 0 | 0 |
| 30 real T-LESS photos | 29 | 0 | 0 | 0 |

In the additional synthetic controls, 12 targets are intact, four have a covered bore and eight scenes contain only lures. The three intact targets in the first synthetic panel rank ahead of their lures. In the 58° example, the target scores 0.714 and the lure scores 0.506. Cutting the bore path lowers the first score to 0.476 and leaves the lure at 0.506. Both fall below the calibrated 0.630 cutoff. On the real evaluation panel, none of the 29 targets reaches even the unthresholded top five and 24 have no encoded point centered inside their visible bounding box. The zero accepted real alarms result from calibration abstaining on all proposals. In a second run, a higher point budget and lower contrast threshold reduce the number of targets without points from 24 to two. Top-five matches stay at zero and false top-five proposals rise from 25 to 128. The added points rarely support the correct pose. [Baseline measurements](outputs/sphere-pose-consensus-measurements.json)

To regenerate the results, use Node.js 24 and Python with NumPy and OpenCV. The reference directory contains the 24 isolated PNGs cached by the [full-frame wave replay](#full-frame-wave-search). The camera panel is the same as in that replay. The measured run used OpenCV 5.1.0. T-LESS previews are resized JPEGs under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), credited to Tomáš Hodaň and colleagues. [T-LESS](https://cmp.felk.cvut.cz/t-less/)

```sh
bash scripts/replay-pose-consensus.sh "$panel" "$full_wave/reference-photos" "$run"
cmp "$run/measurements.json" outputs/sphere-pose-consensus-measurements.json
bash scripts/replay-pose-consensus.sh "$panel" "$full_wave/reference-photos" "$dense_run" --dense-real
```

## Neural features and same-frame feedback

Frozen DINOv2 ViT-S/14 encodes uniform overlapping tiles of each full photo. The scene feature grid uses a nine-pixel stride on T-LESS and a 14-pixel stride on the synthetic plate. For each rotation, C combines normalized features from two aligned isolated training views. The same geometric search shifts the response maps by the predicted locations of reference details, checks 12 in-plane angles and the recorded scales, and suppresses overlapping proposals. The neural and dense gradient encoders use the same search, views, tile positions and evaluation limits. The pinned DINOv3 checkpoint requires gated access, so this trial uses the publicly available DINOv2 checkpoint. The full-frame search reads only RGB, and scene annotations enter only during evaluation. [DINOv2 checkpoint](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)

Each of the five leading candidates gets one native 48 × 48 scene crop. The fixed policy centers the crop on the candidate. The targeted policy chooses one of three reference sites using their projected separation under competing poses and their displacement from the center. It selects the addresses before the local verifier runs. Both policies expose 11,520 additional native pixels per scene to the verifier, counting pixels again when queries overlap. The neural verifier processes each crop at 84 × 84 and compares it to the matching crop of the posed reference. Agreement and disagreement generate signed evidence weighted by a texture and reference-coverage proxy. The score increases by 0.16 times that evidence. The previous full-frame encoding has already processed a coarse grid covering the scene. The charged budget measures extra crop access independently of full-frame encoding cost.

For the wave arm, the chosen address travels as a word of discrete amplitude samples through the reverse route in the 1,000-junction sphere topology. The reader receives its crop coordinates from the decoded word. The signed response returns on the corresponding forward route. The junction updates the candidate score and the far probe reads it. The sparse simulator preserves amplitudes on dedicated delay lines and rounds scores and evidence to 10⁻⁴. The direct and wave arms match requests, local evidence and scores in all 86 scenes. Cutting the request route before the decoder issues zero crop reads across the same 86 scenes. The cut keeps the coarse result. The tight parity control measures the effect of routing with an otherwise identical feedback policy. [Interactive replay](https://nicolashedoire.github.io/sphere-patterns/neural-feedback.html)

The two real panels use different physical setups from one another, with the same 24 development photos and 30 evaluation photos as the previous experiment. Both panels have appeared in earlier project trials. The 29 eligible evaluation instances have at least 20% visibility. Matches use visible-box IoU of at least 0.50 and one-to-one assignment. Each method is calibrated on development to maximize matches under the same budget of two false boxes and one target-absent photo with an alarm. The synthetic development budget is zero false alarms. The five-proposal rankings are reported separately from accepted detections.

| T-LESS replay arm | Targets ranked first | Targets in top five | Accepted targets | False alarms at cutoff |
| --- | ---: | ---: | ---: | ---: |
| Previous RootSIFT | — | 0 | 0 | 0 |
| Dense gradients with the same search | 0 | 0 | 0 | 0 |
| Frozen DINOv2 proposals | 4 | 8 | 1 | 4 |
| Fixed crops | 5 | 8 | 1 | 3 |
| Targeted crops, direct or via waves | 6 | 8 | 1 | 1 |
| Request route cut | 4 | 8 | 1 | 4 |

The DINOv2 top five also contain 142 unmatched boxes before thresholding. In the additional 24-scene synthetic panel, targeted crops place 12 of 16 targets first, versus nine with fixed crops and eight with the first pass alone. The targeted arm abstains at its synthetic cutoff because its calibration panel supplies no safe nonzero cutoff. The reread only reranks the original five hypotheses. On the real scene `20-000407`, it moves a correctly localized object from third to first while leaving its score below the cutoff. This experiment supports neural proposals and tests feedback mechanics. The gain in accepted real detections over the gradient and old SIFT controls is one instance. [Measurements and scene records](outputs/sphere-neural-feedback-measurements.json)

Reproduce with Node.js 24 and Python 3.12. The measured runtime used PyTorch 2.12.0, Transformers 4.57.6, NumPy 2.5.5, OpenCV 5.1.0 and Apple MPS. Set `SPHERE_NEURAL_DEVICE=cpu` to use a CPU, allowing for backend-level numerical differences. The script checks the SHA-256 of the official Safetensors checkpoint. The weights are downloaded separately from the public model repository. The inputs are the same camera panel and isolated PNGs as in the [pose agreement trial](#pose-agreement-from-local-details). T-LESS photo previews are resized JPEGs credited to Tomáš Hodaň and colleagues under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Dataset](https://cmp.felk.cvut.cz/t-less/)

```sh
hf download facebook/dinov2-small config.json model.safetensors \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 --local-dir "$weights"
bash scripts/replay-neural-feedback.sh "$panel" "$full_wave/reference-photos" "$weights" "$run"
cmp "$run/measurements.json" outputs/sphere-neural-feedback-measurements.json
```
