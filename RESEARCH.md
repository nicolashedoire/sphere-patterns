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
| 5 | Full camera image, no box or mask supplied | Generate candidates from the image and score both localization and false alarms against hidden annotations. | Pending a held-out scene benchmark |
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

The pilot runs outside the wave network to isolate the image representation. Dataset boxes are supplied to all three arms. The grayscale cutoff is fitted on the two training views. The matcher uses a common IoU cutoff of 0.86. The annotated-mask arm shows that improved segmentation alone does not resolve the lookalikes and changing viewpoint for this reference. The next study compares local features and additional reference views on separately held-out scenes before evaluating full-image detection. [Camera scenes](outputs/sphere-tless-scenes.png) · [Measurements](outputs/sphere-tless-measurements.json)

### Dataset credit and reproduction

The scene figure and fixture crops are derived from *T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects*, by Tomáš Hodaň and colleagues, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The derivatives include downsampling, crops, luminance and depth grids, visible masks, and box overlays. The source archives are pinned to revision `5fd309a04476a842d93abfb584fba9ee7caecdf1`. [Official dataset archives](https://huggingface.co/datasets/bop-benchmark/tless/tree/5fd309a04476a842d93abfb584fba9ee7caecdf1)

Run `bash scripts/replay.sh` to reproduce measurements and synthetic figures from the checked-in inputs. The optional `python3 work/sphere-tless-fetch.py` command uses Pillow 12 and curl to rebuild the real-image input panel from selected archive entries.
