# pubs:p0:GS-EQ-SWEEP-ext — POST-RESULT extension (disclosed), draft 07:40 PT 2026-10-09

Trigger (disclosed): after the first registered GS-EQ-SWEEP runner (bicycle s3 part A) returned, all-views LPIPS showed
ours' top registered rung (2M, 0.211) short of gsplat-MCMC 3M (0.206). Under the registered rule such a scene is
SLOWER (not reached). This extension measures whether a larger ours budget reaches gsplat-3M and at what time.
It does NOT change or replace the registered GS-EQ-SWEEP verdict; it is reported next to it, never pooled.

Design: identical to pubs:p0:GS-EQ-SWEEP (scenes, seeds, RTX PRO 6000 py-v2, pinned historical-build-096 933d67da, harness, warm-ups,
clocks, NVML, scorer 4faae9d2, DEV = H[1::2] / TEST = H[0::2]) except: part C runner per (scene, seed) = gsplat 3M + ours
2M / 3M / 4M (the 2M arm repeats the registered rung on this runner).
Rule: target = gsplat-3M 3-seed mean DEV LPIPS (part C runner); ours = smallest of 2M/3M/4M reaching it on DEV; ratio =
mean over 3 seeds of train_s (same runner); FASTER < 0.95 with TEST hold >= 2/3; SLOWER > 1.05 or not reached; else TIE.
Row: FASTER count over 6 scenes reported; labelled "post-result extension". Also reported: ours best TEST LPIPS within
gsplat-3M's train_s; NVML peak-minus-idle; part A vs part C gsplat-3M and ours-2M runner variance.

## Recipe source (rule:competitor-recipe-source) — applies to GS-EQ-SWEEP and this extension
gsplat v1.5.3 @937e29912570 (local copy $SPLAT_DATA/research/refs/gsplat-v1.5.3):
- examples/benchmarks/mcmc.sh l.6 `CAP_MAX=1000000` (the only upstream MCMC benchmark recipe; M360 scenes, data_factor
  4 outdoor / 2 indoor, l.11-15; our scene dirs are those resolutions pre-downsampled, run at data_factor 1 as in GS-EQ).
- examples/simple_trainer.py l.1213-1222 MCMC preset (init_opa 0.5, init_scale 0.1, opacity_reg 0.01, scale_reg 0.01).
- gsplat/strategy/mcmc.py l.49 `cap_max: int = 1_000_000` (documented default; used for truck-full, which has no
  upstream per-dataset recipe).
The 300k / 600k / 3M rungs are NOT upstream recipes: they are a cap sweep given to the competitor for fairness (only
cap_max changes). The extension's trigger was an all-views peek at one registered runner (bicycle s3 part A).

## Training-health metric (symmetric, from the kept logs, computed at verdict time)
gsplat: "Relocated N GSs" / "Added N GSs" every 100 steps (MCMCStrategy verbose) -> relocated fraction of current count
per 100 steps; flag if > 0.5. Ours (historical-build-096 logs no relocation counts): splat-count trajectory every 200 steps
("Refine iter") and the count of "dead slots without a replacement left empty" warnings; flag a run whose count falls
> 50% between refines or ends < 50% of its cap. Flagged arms are disclosed, not excluded.
