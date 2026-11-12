# DRAFT registration: pubs:p1:frozen-posefit-v1

Status: DRAFT for publications, AMENDED (v1 -> v1-amended, 2026-10-09, from R-LIT 13d). Nothing launched.
Class: DESCRIPTIVE (no superiority claim).

## AMENDMENT (v1 -> amended): applies to EVERY fitted row (CoMo, B1c, BAD-GS)

CoMo's `optimize_test_pose` keeps the best 10-epoch check by mean PSNR on the test views
(pose_optimize_utils.py:183-191). That is selection on the evaluation set. From this amendment on:

- **Fitted PRIMARY = final epoch.** Score the poses left after all 200 epochs, with no selection.
- **Fitted SECONDARY = "author protocol, best epoch on test views".** CoMo's rule, reported from the
  SAME fit run as the final-epoch score.
- Both get per-view schema-v1 rows. The per-epoch mean (and per-view) pre-update PSNR history is saved for all 200 epochs.
- No-fit remains the overall PRIMARY. Both fitted numbers are secondary to it.
- Existing CoMo s0/s1/s2 and B1c fitted numbers are best-epoch only. Their final-epoch numbers need the
  re-score described in the BAD-GS harness report (load saved checkpoints, rerun the fit, record both).
  Until then they may only be cited as "author protocol, best epoch on test views".

## Question

How much of the released BAD-Gaussians held-out gap on Deblur-NeRF real is held-out camera
misregistration? The stage-1 no-fit score is 6-8 dB below the paper. The renders are sharp but show
a uniform per-scene zoom/shift against the targets (BADGS.md, diagnosis 2026-10-09). Answer it with
the same frozen-scene test-pose fit that every other fitted row uses (CoMo, B1c).

## Final-epoch numbers for existing CoMo / B1c fits (amendment item 2)

**They can't be derived from saved output; a GPU re-score is needed.**

- **CoMo** `como-*/run` (s0 = full-suite plus reference-r4, s1, s2):
  - `pose-fitted/` holds only the best-epoch render scores.
  - The client logs hold only CoMo's tqdm description at each 10-epoch check: the 2-decimal mean
    pre-update PSNR of the check (e.g. coffee s0 ... 35.36, 35.37, 35.38). That is the selection
    metric, not a score of any pose, and the final post-epoch-199 poses were never saved.
- **B1c and A1** `secondary-pose-alignment/result.json`:
  - holds the 10-epoch selection history (`selected_epoch_index` is anywhere from 40 to 190, often < 190),
    the best-epoch scores and the selected poses;
  - has no final-epoch poses or scores.
- Late histories are flat to about 0.001-0.03 dB, so final and best probably differ little. That is an inference, not a number.
- **Cheapest path, no retraining.** The checkpoints are still on fal /data:
  - CoMo PLYs: `run/trained.json` `checkpoint`;
  - B1c: `run/b1c/remote-models.json` `checkpoint`.
  One short GPU exec per checkpoint loads it, reruns the identical 200-epoch fit once, and scores BOTH the
  final-epoch and the best-epoch poses from that same run (re-scoring only final poses would mix two stochastic fits).
- **Code changes:**
  - (a) `evaluate_test_alignment.py align()`: also keep `cameras()` after the last epoch and score it,
    plus the per-epoch history (a small B1a-owned edit; B1c's runner calls it unchanged);
  - (b) a CoMo `--score-only` wrapper that loads `Scene(load_iteration=40000)` from the PLY root,
    without training. CoMo's function returns only the best poses, so the wrapper runs the same loop (MIT source) recording both.
- **Cost:** each job is mostly setup (CoMo CUDA build about 1.5-3 min) plus a 1-2 min fit, so about $0.20-0.30 each.
  That is about 30 CoMo runs (s0 10, s1 10, s2 10) for about $6-9, and 20 B1c confirm runs (2 seeds x 10; each also
  re-fits its A1 arm) for about $4-6.

## Units

- 10 checkpoints: the stage-1 seed-42 BAD-Gaussians 30k models from claim group
  pubs:p1:released-deblur-protocol-v1 (state `badgs-protocol-state.json`). Each fit job loads one
  checkpoint from the `trained.json` path on fal `/data` and verifies its sha256 (step-000030000.ckpt plus
  config.yml). Readability on fal /data was confirmed by the coordinator with `fal files ls` and the
  admin key. Each job also fails fast (exit 3, before any install) if either file is unreadable on the runner.
- Scenes: blurcoffee, blurdecoration, blurstair, blurbasket, blurpuppet, blurbuick, blurheron,
  blurball, blurparterre, blurgirl. The held-out views are exactly each `splits/deblur-real-<scene>.json`
  paper.test list; the job asserts this.
- Seed 42 (the training seed, also used for the fit's view-order RNG). There is no retraining and no
  new checkpoint. The frozen model's parameter hash and the checkpoint sha256 are asserted unchanged after the fit.

## Fit procedure (identical across methods)

Source: the CoMoGaussian release cd6d536, `utils/pose_optimize_utils.py` `optimize_test_pose(num_iter_per_view=200)`,
as called by our released-CoMo run (`run_como_reference.py`). The B1c secondary fit is
`research/b1a/evaluate_test_alignment.py` `align()`, i.e. B1c protocol.json evaluator_source
historical-build-036, which is unchanged at HEAD. B1c mirrors CoMo on every item below. BAD-GS uses the same
items (`research/b1a/badgs_posefit.py`).

| item | CoMo release | B1c | BAD-GS (this) |
|---|---|---|---|
| parameters per held-out view | c2w rotation quaternion (roma, xyzw; renormalised after +1e-8) + w2c translation, original COLMAP frame and units | same | same; mapped to the nerfstudio frame only for rendering, and asserted equal to COLMAP sparse/0 images.bin |
| optimiser | Adam eps 1e-15; lr 5e-5 (quat), 5e-4 (trans) | same | same |
| schedule | StepLR step 200//20 = 10 epochs, gamma 0.9 | same | same |
| budget | 200 epochs; each epoch shuffles all test views and takes one Adam step per view (200 steps/view) | same | same |
| loss | L1(clamped render, sharp target) | same | same |
| rendered | one sharp image at the candidate pose | same | one sharp image, BAD `mode='mid'` with no learned trajectory delta (no blur synthesis) |
| reported (amended) | release: best check only | best check only (to be re-scored) | PRIMARY final-epoch poses; SECONDARY best check (CoMo rule), same run |
| shuffle RNG | global python `random` | `random.Random(seed)` | `random.Random(seed)` |
| renderer / pose gradient | CoMo pose-backprop 3DGS rasteriser | gsplat (our A1 runner) | gsplat 0.1.11 `project_gaussians` view-matrix gradient (the release's own training path; its rotation gradient is the documented small-angle approximation) |

Differences are limited to the shuffle RNG source and each method's own renderer. The optimisation
procedure is the same.

## Outputs per scene

- `run/result.json` holds a `test_pose_fit` block:
  - fit settings, the 10-epoch check history, the selected epoch, `fit_s`;
  - `per_epoch`: the mean and per-view pre-update PSNR for all 200 epochs;
  - `scores_final` (PRIMARY fitted) and `scores_best_epoch` ("author protocol, best epoch on test views"),
    each float + PNG (`como_secondary`) from the frozen `evaluate_sharp.py`
    (sha256 0f3b0b352448329799bd6db36eb622ba251ad49932390bba3d7ededd7a21e10b);
  - a reload no-fit score, as a check that the reloaded checkpoint reproduces the stage-1 no-fit PSNR (|diff| < 0.01 dB expected);
  - per-view pose deltas: rotation (deg); camera-centre shift in COLMAP units, along the view axis, and as a
    fraction of the median train-camera spacing. These are the image-alignment-free sanity check.
- Schema-v1 jsonl: `reload-no-fit/`, `test-pose-fit-final/` and `test-pose-fit-best/`, each `{float,png}.jsonl`.
  Pose deltas are given for both the final and the best-epoch poses.
- The collector (`badgs_launch.py collect-rescore`) merges `test_pose_fit` into the stage-1
  `run/result.json` and records the pre-merge sha256. `no_fit` and the release-eval-path block stay unchanged.

## Reporting

- PRIMARY remains no-fit (unadapted). `test_pose_fit` is SECONDARY: sharp-test-target pose fitting, not
  unadapted generalisation. Within it, final-epoch is primary and best-epoch is labelled "author protocol,
  best epoch on test views". It is the matched row for CoMo/B1c fitted numbers once those are re-scored the same way.
- Report per scene, plus the all-10 mean and the scene-bootstrap CI, for no-fit, release eval path and test-pose fit;
  also the pose-delta table.
- No paper-parity claim. BAD-GS's paper does not state a test-pose procedure, so 28.11 etc. are context only.

## Cost / time

- One `drive.py exec` per scene on RTX PRO 6000 (py v3), timeout 1800 s.
- Expected about 4-6 min run each:
  - setup about 85 s (venv + gsplat JIT);
  - checkpoint load and a reload score under 30 s;
  - fit 200 x (4-7 views) render+backward steps, about 1-2 min;
  - scoring (reload, final, best) under 1 min.
- About $0.20-0.30 per scene at $0.0008/s, so about $2-3 for all 10, and under $5 including one retry.

## Launch (after registration)

```
cd $HOME/code/nf/splat-agent-worktrees/research-robust
python3 research/b1a/badgs_launch.py rescore --registration <msg id> --scripts posefit-v2
python3 research/b1a/historical-record.py $SPLAT_DATA/research/b1a/badgs-posefit-state.json --max-inflight 10
cd research/b1a && python3 badgs_launch.py collect-rescore
```

`rescore` only registers experiments and writes the state file. It skips scenes already prepared, so
re-running it later adds scenes whose stage-1 output has arrived. The scripts tar is
`inputs/badgs-scripts-posefit-v2.tar` (sha256 b662cb89...42e4).
