# pubs:p0:SP-EQ-fair — registration draft (GSEQ-SWEEP lane, 2026-10-09 06:58 PT)

Purpose: fairness re-check of SP-EQ r1/r2/r3 (ours' budget tuned on scored views; Spirula at its untuned default).
Both arms get a cap ladder; selection on DEV views; all numbers on disjoint TEST views; one pinned ours binary.

## Design
- Scenes: SP-EQ's 6 (bicycle, counter, garden, playroom, drjohnson, truck-full), the scenes of the claims being
  re-checked. Seeds/runs 1/2/3. 30k steps. L40S engine image, `drive.py queue --gpu l40` (paired groups stay on L40S).
- One runner per (scene, seed) group, all 11 arms back to back:
  Spirula v2026.10.6 (b9c8042, $SPLAT_DATA/build/release/spirula sha256 d68620c1...) `--cap-max`
  300k / 500k / 1M (its default) / 3M; Spirula has no seed flag (runs 1-3 are repeat runs, as in SP-EQ).
  Ours brush-cli main historical-build-096 (sha256 933d67da...e66f) `--max-splats` 300k / 500k / 800k / 1M / 1.2M / 2M, plus
  ours DEFAULT (`--max-splats 0`, automatic budget), `--seed k`.
- Harness: SP-EQ r3's exactly (drive.py queue paired group -> splat-perf bench; canonical Spirula full-checkpoint
  zero-update eval N/N; every arm warmed untimed for min(30k, 7k) steps before the timed arms, drive.py default as in
  r3); scorer splat-perf at historical-build-096 (`--splat-perf git:historical-build-096`), full-res GPU LPIPS-VGG canonical, per-view rows.
- Clocks: primary train_s (ours train_only_s; Spirula final-step elapsed), secondary whole wall_s.
  Memory: splat-perf NVML board peak minus pre-launch idle (peak_vram_mib), same sampler both arms.

## View split (fixed before any run; identical to pubs:p0:GS-EQ-SWEEP)
H = every8_idx0_sorted held-out list in sorted order; DEV = H[1::2], TEST = H[0::2]. Training views unchanged.

## Readouts and decision rule
(1) PRIMARY, best-tuned vs best-tuned at equal LPIPS: spirula* = Spirula cap with the lowest 3-run mean DEV LPIPS-VGG.
    ours* = smallest ours ladder cap (300k..2M) whose 3-seed mean DEV LPIPS <= spirula*'s (no tolerance). Ratio = mean
    over 3 paired groups of ours*/spirula* train_s. FASTER if ratio < 0.95 and ours* TEST LPIPS <= spirula* TEST LPIPS on
    >= 2/3 groups; SLOWER if ratio > 1.05 or no ours cap reaches on DEV; else TIE. Row AHEAD if FASTER >= 4/6 and
    SLOWER 0; PARITY if FASTER >= SLOWER; else BEHIND.
(2) ROBUSTNESS: the same per-scene verdict at EACH Spirula cap; per scene the worst verdict over the 4 caps; same row
    thresholds. Shown next to the primary with every per-cap ratio.
(3) EQUAL CAP (no selection): at 500k and 1M, ours vs Spirula at the same cap: train_s ratio (FASTER/TIE/SLOWER by
    0.95/1.05), TEST PSNR/LPIPS deltas; 'better quality at equal cap' per scene only if rule:seed-gate passes on paired
    TEST PSNR (mean > 2x SD over 3 groups, LPIPS within +0.002).
(4) DEFAULTS vs DEFAULTS: ours --max-splats 0 vs Spirula 1M: train_s ratio, TEST PSNR/LPIPS, splat counts, memory.
Reported for all: NVML peak-minus-idle ratios, whole-wall ratios, both-arm LPIPS-vs-train_s curves (TEST).
Headline wording only if (1) is AHEAD; quote the (1) TEST range together with (2), (3) and (4).
Infra: a group failing on infra (non-zero exit, missing arm, n_scored != n_expected) is rerun whole, logged; judged
without reference to its quality or speed. Never pooled with SP-EQ r1/r2/r3.
