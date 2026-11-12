# Gated transient mask v2 — design + registration (2026-10-09 07:45 PT; GPU after 10:30 PT only)

Binary: b13-hpcmask-main 924ce938 (historical source archive historical-build-011 + mask port + gate; fixed-shape stores from devvel historical-build-097; gate OFF/Pending = exact default step incl. robust rejection; B13_NO_ROBUST only while ON). sha256 2626f8bc8ab6473b4f40ad037872c40277b2dd0258ad67458f827e1e87787fc5, durable copy research/b13/bin/brush-cli-hpcmask-main-gate-924ce938.

Arms (same binary, same-runner groups, 3 seeds 42/43/44, 30k, canonical scorer, lpips_scale full): DEFAULT (flags off) | GATED (B13_HPCMASK=1 B13_NO_ROBUST=1 GATE_STEP=3000 WINDOW=1000 FRAC=τ) | FORCED-ON (B13_HPCMASK=1 B13_NO_ROBUST=1, no gate). Every report carries wall/step_ms and a training-health column = dead/relocated splat fraction per 100 steps where the engine exposes it (flag > 50%); cap asymmetries disclosed separately.

## Stage A — criterion on DEV scenes only (all already seen: spot, garden, room, counter, mountain, patio; RobustSplat corpus)
Run DEFAULT | FORCED-ON on mountain + patio (the two scenes whose benefit is unknown). Benefit b(scene) = seed-mean ΔPSNR(FORCED-ON − DEFAULT). Known from v1/nohard: spot b ≈ +3.8 (t≈0.27); garden/room b < 0 (t≈0.02–0.03); counter t≈0.027.
Criterion fixed now: τ = the largest value in {0.04, 0.05, 0.075, 0.10, 0.15} such that every dev scene with b > +0.10 dB has gate-window transient fraction t > τ, and every dev scene with b ≤ 0 has t ≤ τ. If no value separates them, v2 stops (no Stage B) and the mask stays opt-in.
No hand tuning beyond picking τ from this list by this rule.

## Stage B — validation ONCE on never-seen scenes
Transient: onthego-corner, onthego-fountain, onthego-patio-high (RobustSplat mixed splits, never run with the gate). Static: moskvitch-walk-around-D, kindel-house-walkthrough-D (the casual scenes v1 failed to run — v1 used non-existent names). DEFAULT | GATED(τ) | FORCED-ON, 3 seeds.
PASS iff all hold:
1. Static scenes: gate OFF in ≥ 2/3 seeds; GATED − DEFAULT seed-mean ΔPSNR ≥ −0.05 dB; Alex/VGG within +0.002; wall/step_ms ≤ +5% (OFF runs).
2. Transient scenes: the gate decision agrees with benefit: if FORCED-ON helps (seed-mean Δ > 2×SD and > +0.10) the gate is ON in ≥ 2/3 seeds and GATED − DEFAULT > 0 with mean > 2×SD across seeds (scene-mean per seed); if FORCED-ON does not help, the gate is OFF in ≥ 2/3 seeds and GATED − DEFAULT ≥ −0.05.
3. Wall where ON reported honestly (no bar).
Mountain/patio forced-ON arms are a DIAGNOSTIC (Stage A), not part of the verdict. Before Stage B: a smoke showing gate-OFF reproduces the default recipe exactly (same step count, same densification/splat counts, robust loss unchanged). Plus a wall-only equivalence check on 924ce938: counter + spot, DEFAULT | GATED, seed 42 (OFF and ON cases; quality must match v1 within run noise).

Cost estimate: Stage A 12 runs, Stage B 36 runs, wall 4 runs ≈ $12. EXPLORATORY until Stage B passes; product default only after Stage B PASS + perf-lead sign-off.
