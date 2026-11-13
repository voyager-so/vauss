# Demo geometry and per-view QA

Audit baseline: private main `fe458ca0bcd76a2ca94f3c0dd785fb5ffb4f57a7`.
Findings 1 and 4 were **partly implemented**: camera-position collapse/outlier heuristics,
component inventory, and worst-view metrics existed. Observed-depth checks, high-registration
geometry rescue, and explicit per-view quality triage were missing. Position-only heuristics
also confused legitimate stationary sections with collapse.

## Geometry contract

`sfm-gates-v4` adds `geometry_coherent` for photo and video solves. It checks:

- Per-camera median positive observed depth: p90/p10 above 1000 is a scale discontinuity,
  with at least ten cameras having at least sixteen observations each.
- More than 10% nonpositive/nonfinite observed depths in a sufficiently observed camera,
  or any nonfinite camera centre.
- Two observation-connected components containing at least three sufficiently observed
  cameras each. The mapper's separate-component inventory remains available too.
- For ordered video, an isolated out-and-back translation over 20 times typical frame-normalized
  motion and 10 times the neighboring chord, also over half observed scene depth. Gaps larger
  than four times the normal frame gap and different filename sequences are not compared.

All length thresholds are relative; no world-unit cutoff is used. Camera stillness and distant
camera-position groups are reported but no longer cause rejection without independent geometry
evidence. Insufficient evidence is labeled, not asserted to be sound. These conservative checks
do not prove global geometry or detect every drift, sustained jump, or weak graph bridge.

Failed geometry is a hard error, including high-registration photo-folder solves and previews.
Eligible failures take the existing learned-pose replacement path; disabled/missing rescue or
an invalid rescue cannot return the failed model. Existing frame-count limits remain in force.
Training rechecks the actual model before acquiring a GPU lock, including old/imported artifacts.

The original failed179 fixture fails the new gate while every old gate passes; good241 passes
all gates. Tests repeat the verdict across similarity transforms and scales 1e-9 through 1e9,
including the demo export's 3173 scale. See the fixture provenance under
`crates/splat-eval/tests/fixtures/README.md`.

## Per-view triage and exact-camera repro

Training, evaluation and requested native view renders report `view_quality`: per-view PSNR
below 10 dB, missing/nonfinite PSNR, or a view over 8 dB below the median of at least eight valid
views requires review. Operation success still means the operation completed, not that appearance
passed QA. The quality status is `review_required`, `insufficient`, or
`no_outliers_in_evaluated_views`; it never certifies unsampled training/novel views.

`render_views.image_names` requests exact registered names across training/held-out splits,
without uniform sampling skipping the suspect interval. Native render workers also save camera
provenance and per-view QA. Their poses come from the supplied dataset; training pose corrections
are not silently applied. When pose refinement was enabled, compare against the corresponding
candidate's refined cameras before attributing a discrepancy to the splats.

```json
{"candidate_id":"candidate_944c3e5fc9af","max_views":7,"image_names":["frame_000100.jpg","frame_000101.jpg","frame_000103.jpg","frame_000104.jpg","frame_000130.jpg","frame_000149.jpg","frame_000169.jpg"]}
```

The CPU-only `eval/demo_geometry_qa.py` generates exact-name native renderer jobs and an optional
separate diagnostic PLY with selected foreground candidates' opacity suppressed. It preserves
the PLY header, mip/SH metadata and all other attributes. This is a falsification experiment,
not a product cleanup filter:

```sh
OPENBLAS_NUM_THREADS=1 python3 eval/demo_geometry_qa.py \
  --dataset /path/to/strict241-sfm --ply /path/to/final-model/splat.ply \
  --out /path/to/new-diagnosis --ablate
# On an approved GPU with lease cleanup, run each generated job:
vauss __brush render /path/to/new-diagnosis/native-original-job.json
vauss __brush render /path/to/new-diagnosis/native-ablated-job.json
```

The jobs reference the actual dataset's `images/` and `sparse/0`. Transfer the dataset and rebase
the JSON paths when changing machines. Compare `scores.json`, `view_quality.json`, camera
provenance, native/source images and depth/alpha outputs, including the clean controls.
The generator sets `max_resolution` to the largest requested source edge (1180 for good241);
zero is **not** unlimited in the image loader. Check output dimensions against the diagnosis's
`expected_render_dimensions` (1180×784 here). Public0.1.21 produces the existing scores/images;
the new QA/provenance JSON files require the integrated build. Regression:
`python3 -m unittest discover -s eval -p 'test_demo_geometry_qa.py'`.

## Diagnosis status (cause remains open)

The supplied native comparison sheet shows heavy foreground occlusions near source captures
100/101/103/104 and clean rear controls 130/149/169. The left images are **film cameras** N94,
N97, N101, N104, while the right images are nearest source captures. `camgen.py` interpolates
calibrated poses (and varies framing elsewhere); the sheet is not exact-training-camera parity.
It does establish a native film-path failure, independent of the custom renderer mismatch.

At the exact original source poses, conservative CPU bounding-sphere projection found these
large, near-camera Gaussian candidates (positive depth under 15% of median sparse depth,
opacity over .3, 3-sigma footprint intersecting the image and over .1 image widths/heights):

| Capture | Candidates | Median observed depth |
|---|---:|---:|
| 100 | 29 | 3.973 |
| 101 | 287 | 3.959 |
| 103 | 340 | 3.989 |
| 104 | 335 | 4.002 |
| 130 | 0 | 2.884 |
| 149 | 1 | 3.577 |
| 169 | 0 | 4.074 |

The union contains 701 Gaussians of 967701. At capture104, candidate61045 has centre depth
0.01924 and opacity0.459; its conservative radius is 12.36 image widths. This proxy ignores
orientation, native mip filtering, occlusion and compositing: it motivates the controlled ablation,
but cannot prove those candidates cause the visual failure or can be safely removed globally.

Training can refine poses, while native requested renders load original cameras. The supplied
good241 evidence lacks `pose_refine.json`, `train_summary.json`, `train_job.json` and
`training.jsonl`. `couple-diagnosis/remote-params.txt` contains **failed179** corrections (150
training views for sfm_6d6511b5cd63), which must not be applied to good241. Original/refined
camera comparison and original/ablated native rendering remain required before a causal fix is
claimed. No evidence here justifies blaming anisotropic intrinsics, rear geometry or AI input.

The original demo lead also reports that a full native capture-orbit showed the same four bad
captures, beyond the interpolated N sheet. Exact-source-camera baseline/ablation remains the
reproducible next test. The lead confirmed the good241 refinement records were never downloaded;
the source machine was released and its only snapshot predates the solve, so those records cannot
currently be recovered. The checked-in raw model contains67691 sparse points.

Restaurant222 has recorded held-out PSNR4.29063 and an independently confirmed exported position
impulse: 16.007 units in, 15.995 out, while the neighboring camera chord is0.01265. The exported
scene contains27 sampled cameras; raw restaurant SfM observations are absent from the provided
bundle. Whether the outlier originated in SfM, training refinement or export therefore remains
open. Its recorded per-view score is a regression case for explicit QA review.

No retraining or local GPU rendering was used for this diagnosis. Parent coordinates independent
review, green CI/merge queue, GPU confirmation and release inclusion; no direct merge is permitted.
