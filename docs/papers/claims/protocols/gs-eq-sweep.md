# pubs:p0:GS-EQ-SWEEP — registration draft (GSEQ-SWEEP lane, 2026-10-09 06:50 PT)

Purpose: answer the FATAL review of pubs:p0:GS-EQ (ours tuned over 4 budgets vs gsplat-MCMC at its untuned 1M default;
budgets picked on the scored views). Both arms get a cap ladder; every selection is made on DEV views; every reported
number is on disjoint TEST views.

## Design
- Scenes: bicycle, garden, counter, room, kitchen, truck-full (GS-EQ's 6; same scene dirs runs/perf/scenes/<S>, M360
  leaderboard res, data_factor 1). Seeds 1/2/3. 30k steps. No in-run eval; held-out renders at 30k only.
- GPU: L40S, fal py image v2 (`--gpu l40 --no-spill --backend fal`), as GS-EQ. Hardware recorded per runner.
- Arms: gsplat 1.5.3 simple_trainer MCMC preset (GS-EQ's gs_run.py; only change: `strategy.cap_max` from env) at
  cap_max 300k / 600k / 1M / 3M. Ours = brush-cli main historical-build-096 (sha256 bound before first submit) at --max-splats
  300k / 500k / 800k / 1.2M / 2M.
- Runners: TWO per (scene, seed) = 36 runners. Part A = gsplat 3M + the full ours ladder (5 runs). Part B = gsplat
  300k, 600k, 1M + the full ours ladder (5 runs). Every gsplat arm is compared ONLY against the ours ladder on its own
  runner. The duplicated ours ladder (A vs B) is reported as a runner-variance measurement, never pooled into a ratio.
- Warm-ups (as GS-EQ, disclosed): ours one full 30k run at 1.2M (discarded; fills autotune cache); gsplat 2k steps.
  gs_preload=0. Order: odd seeds = gsplat caps ascending then ours ascending; even seeds = exact reverse.
- Clocks: primary train_s = first -> last optimizer step (ours `train_only_s`, GPU-synced; gsplat cuda-synced
  perf_counter bracket). Secondary whole-process wall_s.
- Memory: NVML board memory.used peak minus the idle baseline taken right before launch, 100 ms sampler, same sampler
  for both arms, MiB. gsplat torch max_memory_allocated also recorded (context only).
- Scoring: splat-perf (sha256 4faae9d2..., unchanged from GS-EQ), full-res GPU LPIPS-VGG canonical (lpips_scale=full),
  per-view rows, n_scored == n_expected or the runner fails.
- Harness: research/gseq-sweep/run.py + gs_run.py, a copy of o1-6e research/gseq @historical-build-051 (run.py 2a4fee64, gs_run.py
  d166e5aa) changing only the arm list, a GS_CAP_MAX env for gsplat, and labels.

## View split (fixed here, before any run)
H = every8_idx0_sorted held-out list (image names sorted bytewise, index % 8 == 0), in that sorted order.
DEV = H[1::2] (odd positions in H), TEST = H[0::2] (even positions in H). Training views are unchanged (all non-H
views, identical to GS-EQ). All selection uses DEV only; all reported quality and verdicts use TEST only.

## Selection and readouts (all LPIPS = canonical LPIPS-VGG, per-run mean over the view subset)
(a) Equal-LPIPS at EACH gsplat cap C: target = 3-seed mean DEV LPIPS of gsplat-C. b(C) = smallest ours cap whose
    3-seed mean DEV LPIPS (same part's ladder) <= target (no tolerance). Ratio = mean over 3 paired seeds of
    ours_b(C) / gsplat-C train_s (same runner). TEST hold = #seeds with ours_b(C) TEST LPIPS <= gsplat-C TEST LPIPS.
    Per (scene, C): FASTER if ratio < 0.95 and TEST hold >= 2/3; SLOWER if ratio > 1.05 or no ours cap reaches the
    target on DEV; else TIE.
(b) Best-tuned vs best-tuned: gsplat* = gsplat cap with the lowest 3-seed mean DEV LPIPS (its quality-tuned config).
    ours* = b(gsplat*) (ours' cheapest config reaching it on DEV). Verdict per scene = (a) at C = gsplat*.
    Also reported: each arm's DEV-best-quality config (min DEV LPIPS) compared on TEST: LPIPS, PSNR, SSIM, train_s ratio.
(c) NVML peak-minus-idle for every run, both arms, and the ratio at the (b) pair.
Also reported: whole-wall ratios for (a)/(b); our best TEST LPIPS among ours caps with train_s <= gsplat-C train_s
(same runner) for each C; full LPIPS-vs-train_s curves per scene (both arms, TEST).

## Decision rule (pre-registered)
PRIMARY (b): row AHEAD if FASTER on >= 4/6 scenes and SLOWER on 0; PARITY if FASTER >= SLOWER; else BEHIND.
ROBUSTNESS (worst case over gsplat caps): per scene take the worst verdict over the 4 caps (SLOWER < TIE < FASTER);
same AHEAD / PARITY / BEHIND thresholds. Reported alongside the primary, with the ratio range over caps.
Headline wording permitted only if PRIMARY is AHEAD; the quoted ratio range is the (b) TEST range; the robustness row
and every per-cap ratio are shown next to it.
Quality (each arm's DEV-best vs DEV-best, TEST): "better quality" claimed per scene only if rule:seed-gate passes on
paired PSNR delta (mean > 2x sample SD over 3 seeds, no guard violated); otherwise reported descriptively.
Infra: a runner that fails on infra (non-zero exit, missing arm, n_scored != n_expected) is rerun whole and logged;
pass/fail judged without reference to its quality or speed. GS-EQ's headline row is unchanged by this group; P2 text
follows whatever this group's rule returns.
Optional add-on (registered separately if run): Faster-GS (Apache) at its shipped config, same 6 scenes, seed 1.
