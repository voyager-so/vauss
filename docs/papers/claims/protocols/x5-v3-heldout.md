# DRAFT registration: x5-stack-vs-nht-heldout-v3

Status: DRAFT 2026-10-09 07:08 PT (R-oracle); scene split approved by the SOTA sub-lead 06:46. To be bound by publications before 07:35 PT; launch about 07:35 PT
(after v2 reports, whatever its result), into the py RTX PRO 6000 slots left after X20 Step 0.
If v2 (x5-stack-vs-nht-m360-3seed-b16-v2) is AHEAD, that claim stands only if v3 agrees.

## Question
Does the frozen X5 STACK recipe keep its standing against NHT on scenes the recipe never saw, with both methods at
their frozen recipes and paper defaults, on one pinned harness?

## Arms (fairness rules, 06:44)
1. **One pinned sha per comparison.** STACK = v2's `bundle-trim`, byte-identical (paired_adapter a462ce11,
   paired_run 2e194e72, paired_score 6fac406b, stack_run 02ab93e2, color_tail 9182be88, runtime.sh 041ec20a,
   stack_ballast 28c82a62, ballast_trace 6eadf969), `ORACLE_STACK_MEMTRIM=full`. One disclosed addition: paired_adapter.py
   394cbff8 (main historical-build-078) = a462ce11 + an opt-in `ORACLE_STACK_TEMPLATE` block. With it unset, the patched STACK source is
   byte-identical to v2's. With it set, STACK loads the frozen M360 config by that template name (STACK ships per-scene configs
   only for M360 names, and all 9 frozen M360 configs are identical apart from PATH, IMAGE_SCALE_FACTOR and RANDOM_SEED:
   verified), and scale 1.0 maps to images/. No training, rendering or metric change. NHT = nv-tlabs/neural-harmonic-textures 2ae7ab0a (gsplat
   submodule 068dc659), simple_trainer_nht.py 5bd3155a, exactly as v1/v2.
2. **Equal footing: both at their frozen recipes/defaults.** STACK = the frozen X5 config (seed-42 normalized config of the
   matching scene type, only DATASET.PATH, IMAGE_SCALE_FACTOR and GLOBAL.RANDOM_SEED set per run; cap 11M, tail 3000,
   30k steps). NHT = paper recipe by NHT's own scene-type rule: outdoor = F64 unfused, 5M cap, 25k steps, per-ray encoding;
   indoor = 2M cap, 45k steps, center-ray encoding. Neither arm's cap is matched to the other; this is stated as
   "both at defaults".
3. **Nothing is chosen on test views.** v3 tunes and selects nothing. Every setting is frozen from v2 or the NHT paper.
   If any threshold were ever needed it would come from a dev hold-out of TRAINING views (GSEQ-SWEEP method); none is used.

## Scenes (corpus eval/corpus/manifest.yaml on main; images and poses as in the corpus tars)
Approved split (sub-lead 06:46; Q1 informed): VERDICT = the 4 manifest test / never-tuned scenes; DESCRIPTIVE "tuned-by-perf"
column = the 3 manifest dev/tuning scenes, never pooled, not part of the verdict.

| scene | manifest role / tuning | category → NHT recipe | images | poses | tar sha256 |
|---|---|---|---|---|---|
| tnt-family | test / false | outdoor → outdoor | 152, 960×540 (1920×1080 halved, as 3DGS tandt) | our SfM (sfm-gates-v3) | 521ed4bf… |
| tnt-horse | test / false | outdoor → outdoor | 151, 960×540 | our SfM | 5521c616… |
| tnt-m60 | test / false | outdoor → outdoor | 313, 1024×540 | our SfM | 000f6a51… |
| zipnerf-london | test / false | outdoor → outdoor | 1874, 696×396 (images_4) | release COLMAP | 4161e88a… |
| tnt-francis | **dev / true** | outdoor → outdoor | 302, 960×540 | our SfM | 3e173ab3… |
| tnt-panther | **dev / true** | outdoor → outdoor | 314, 1024×540 | our SfM | cac2d2f1… |
| zipnerf-nyc | **dev / true** | indoor → indoor | 990, 696×396 (images_4) | release COLMAP | 8cd99ae1… |

Verdict set: tnt-family, tnt-horse, tnt-m60, zipnerf-london (12 pairs, launched first). Descriptive: tnt-francis, tnt-panther,
zipnerf-nyc (9 pairs, after). Truck, train and Deep Blending are excluded (perf-tracked); an optional "tracked" column only if
budget remains. Poses: T&T uses our corpus SfM (native incremental, sfm-gates-v3 pass), shared by both arms, so rule:sfm-noise
does not apply; Zip-NeRF uses the release COLMAP.

Split: every-8th of sorted image names, index 0 (`every8_idx0_sorted`), same as v1/v2. Both arms assert the same view list and
decoded-target hashes (paired_score). Seeds: 51, 52, 53 per scene (fresh).

Scene-type and scale per arm:
- STACK IMAGE_SCALE_FACTOR: Zip-NeRF 0.25 (images_4 with full-resolution release intrinsics, as M360). T&T 1.0: the corpus
  SfM was run on the 960×540 images, so the intrinsics already match. The frozen stack_run maps only {0.25, 0.5, 0.125}
  to an image directory, so the bundle adds `1.0 → images` (an adapter replacement; no training or metric change).
  STACK config = the frozen M360 template (ORACLE_STACK_TEMPLATE=bicycle for every scene, since all frozen M360 configs are
  identical) with only PATH, IMAGE_SCALE_FACTOR and RANDOM_SEED set. Normalized config sha256s are listed at binding.
- NHT: data_factor 1 on the corpus images (native_images_factor), test_every 8, no app_opt, pose_opt, post_processing or
  colour-corrected metric (see R-oracle 06:45 evidence: test views rendered with train-derived parameters only).

## Harness, guards and evidence (as v2)
- Both arms of a pair run in one fal exec on one runner (RTX PRO 6000 Blackwell, py image v3), under the same B4 16 GiB holder.
- Scorer paired_score 6fac406b, float32 renders, one base-image process: PSNR, torchmetrics Gaussian SSIM, VGG and Alex LPIPS.
- Memory guard, log-only during the run, judged post hoc: torch training peak (through finalize) ≤ 15 GiB; NVML peak minus
  the pair's own setup peak ≤ 15 GiB (pubs 05:30); whole-process torch ≤ 16 GiB; physical OOM in the holder = guard
  failure. Also logged: raw NVML increment, NVML gaps (> 1.0 s = disclosed flag; within 1 GiB of 15 = guard (b) unverified).
- A pair that fails the guard is reported as failed; no swapping, no reruns except for infra failures (fresh SPLAT_ATTEMPT,
  original kept on record, cancelled.jsonl).
- keep_renders on (both arms' float test renders to /data/splat-fal/research/r-oracle/x5v3-renders/).
- 0%-GPU watchdog at 25 min (CPU LPIPS phase is 0% GPU by design).

## Pre-launch smokes (non-citable, inside the cap)
One full-pair smoke (both arms, seed 0, separate non-citable root) per dataset family: tnt-family (new scale-1.0 / template
path and T&T loader for both arms) and zipnerf-london (1874 views: memory, NHT and STACK loaders, CPU-LPIPS wall). Smoke
quality is not opened or reported. The loader/memory outcome is reported before verdict pairs launch. If a smoke fails, v3
waits for a fix plus an amendment.

## Decision rule (copied from v2; scene count adjusted; publications to fix the wording)
On the verdict set (4 scenes × 3 seeds = 12 valid pairs required): Δ = STACK − NHT PSNR.
- AHEAD iff mean of the 3 per-seed means of the 4-scene mean Δ > 2 × SD (ddof=1) of those per-seed means, AND ≥ 3/4 scenes
  have a positive 3-seed mean Δ, AND mean ΔLPIPS-VGG ≤ 0.
- PARITY iff |Δ| ≤ 2 × SD; otherwise BEHIND.
- Reported, not decisive: no scene's seed-mean Δ < −0.05 dB (seed-gate guard); SSIM and Alex deltas; per-scene table;
  tuned-by-perf column; both memory readings per arm.
- Incomplete (fewer than 12 valid verdict-set pairs, or more than one runtime) → INCOMPLETE.

## Budget and timing
Cap $130 (research lead). Estimate: T&T pairs ~$1.5–2.5, london ~$4–6 (1874 images, 235 test views, CPU LPIPS), nyc indoor
~$4; 21 pairs ≈ $50–65 plus smokes ≈ $5. Launcher gate: actual + per-type reservations ≤ cap; watchdog 25 min.
Results ~75–100 min after launch.

## Coordination
Scene list, split and pinned-bundle approach are shared with perf lead historical-record (Q1 generalisation
column) and Q1 historical-record so both use one protocol.
