# X25 registration draft: PPISP controller-only re-run on the kept x5v2 checkpoints

Status: DRAFT for publications (historical-record). Written 2026-10-09 ~08:55 PT, before any GPU run.
Label: **post-hoc on v2 checkpoints, not a v2 re-verdict.** The v2 verdict (8-scene fallback PARITY, Δ +0.131 dB,
seed-mean SD 0.115) stands as recorded; nothing here changes it.

## Why (desk evidence, research/x25/diagnosis.md)
- The bonsai-s45 and kitchen-s46 STACK controllers are dead. Their exposure output is constant over every image
  (SD ~1e-9); 127/128 and 128/128 first-layer trunk ReLUs never fire. The good seeds' controllers track the fitted
  per-frame latents (r 0.87–0.95).
- In those two runs the test-view exposure error equals the neighbouring training latent (slope 1.09 / 1.02,
  r 0.98 / 0.88). This is the v2 seed outlier.

## What runs
- Inputs: all 27 kept v2 STACK checkpoints (`/data/splat-fal/research/r-oracle/stack-models/<attempt>/FasterGSVDA/*/checkpoints/final.pt`;
  9 scenes × seeds 45/46/47), the same scene tars and job.json as v2 (views_hash and split asserted).
- Frozen: every Gaussian tensor, the appearance network, and PPISP's per-frame exposure/colour, vignetting and
  CRF. Only the controller is re-initialised and re-trained, with the same 5,000-step photometric distillation
  loop as EGA 4efbe67 `Trainer.finalize` (Trainer.py:254-275) on training views only.
- **Geometry check (hard assert, per job):** sha256 of all model tensors except `ppisp.model.controllers.*`,
  before and after, must be bitwise equal; otherwise the job fails.

## Arms (per checkpoint)
- **R, recorded:** the controller saved in final.pt. Harness check: per-view test PSNR must reproduce v2's recorded
  STACK per-view PSNR (stack-per-view.jsonl) within 0.05 dB mean |Δ|. If not, the job is flagged HARNESS-MISMATCH
  and is not used.
- **S, stock re-run (diagnostic):** ppisp v1.3.0 controller and optimiser as shipped (ReLU, Adam lr 1e-3 constant),
  3 candidates. Used for the stock collapse rate and a "selection-only" ablation.
- **F, fixed = `ctrlfix-v1` (the registered fix):**
  1. Every ReLU in the controller (CNN encoder and MLP trunk) becomes LeakyReLU(negative_slope=0.01). The
     parameter shapes and state dict are unchanged.
  2. Controller Adam (betas 0.9/0.999, eps 1e-15, as shipped). LR over the 5,000 controller steps k = 0..4999:
     linear warmup 1e-3·(k+1)/500 for k < 500, then cosine from 1e-3 down to 1e-5 at k = 4999.
  3. K = 3 candidates. Candidate c ∈ {0,1,2} uses torch seed and view-order seed 1000·(c+1) + run seed.
  4. **Selection rule (GT-free for test views; uses training views only):** after training, one no-grad pass
     over all training views computes, per candidate:
     - L_c, the mean distillation loss (the same loss the controller is trained on);
     - ρ_c, the SD over training views of the predicted exposure divided by the SD of the fitted per-frame
       exposure params.

     Candidates with ρ_c ≥ 0.25 are eligible; pick the eligible candidate with the lowest L_c. If none is
     eligible, pick the lowest L_c and flag the job ALL-COLLAPSED.
- View order: uniform sampling without replacement over training views, reshuffled every epoch.
- Diagnostics logged but not used for selection: every candidate's test PSNR; per-candidate dead-unit counts.

## Metrics and report
- Test PSNR as `paired_score.py` computes it (per-view float MSE, clamp [0,1], range 1, views weighted equally),
  on the same test views. NHT values are the recorded v2 per-pair NHT scores (unchanged).
- **Primary (pre-declared):** on the v2 8-scene fallback set (room excluded as in v2), with R vs F:
  1. the sample SD over seeds of the per-seed 8-scene mean Δ(STACK − NHT) PSNR;
  2. the mean Δ;
  3. the mean STACK absolute PSNR.

  Room is reported descriptively.
- **CLEAN = all of:**
  - (i) SD_F < SD_R;
  - (ii) mean Δ_F ≥ mean Δ_R − 0.02 dB;
  - (iii) no scene's 3-seed mean STACK PSNR drops by more than 0.10 dB (F vs R);
  - (iv) 0 of 27 F selections ALL-COLLAPSED;
  - (v) geometry assert passes on 27/27 jobs.
- **Health metric (GT-free):** ρ and the first-layer dead-unit count for R, S and F. A controller with ρ < 0.25 is
  "collapsed". Report the collapse rate of S (stock, 81 trainings) vs F candidates (81 trainings).
- Named target views: bonsai DSCF5565, DSCF5757, DSCF5765, DSCF5853 (controller collapse), DSCF5837 (X18:
  shading plus exposure, not expected to be fixed by this); kitchen DSCF0728 (small colour offset).
- Seed-noise rule: per-scene deltas under 0.3 dB on a single seed are not claimed. This study removes controller
  noise only; render nondeterminism shows up in R vs recorded.

## Cost and wall (estimate)
- 27 fal RTX PRO 6000 jobs, one per checkpoint. Each job: unpack the scene and load the checkpoint (~2 min), then
  1 + 3 + 3 controller trainings at ~35–40 s each, plus 7 test-render passes and 6 train-view selection passes
  (~2 min). About 9 min per job.
- About 4 GPU-h, roughly $12–15. Wall about 20–30 min if run in parallel.

## Fresh-seed confirmation
- In the next full STACK run (X21 2×2, if this lands CLEAN by ~10:20), every arm uses `ctrlfix-v1` via the env flag
  `X25_CTRLFIX=1` (module `ctrlfix.py`, sha-pinned). It is declared before launch.
- That run's per-run ρ, dead-unit count and the selected candidate are logged as health columns.
