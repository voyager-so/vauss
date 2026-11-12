# q1-heldout-historical-build-096: registration draft (Q1, 2026-10-09 07:00 PT)

Purpose: answer the contamination objection to the quality scoreboard, GS-EQ and SP-EQ. The comparison runs on scenes
our engine was never tuned on. Both arms are tuned only through cap ladders, every selection is made on DEV views, and
every reported number comes from disjoint TEST views. This is one claim group, separate from X5 v3 (STACK vs NHT). The
two groups share the scene roles and split of `docs/papers/claims/protocols/x5-v3-heldout.md` (main historical-build-050).

## Scenes (frozen here, before any run, including smoke runs)
- VERDICT: tnt-family, tnt-horse, tnt-m60, zipnerf-london. All four are `role: test, tuning: false` in
  eval/corpus/manifest.yaml.
- Descriptive "tuned" column: tnt-francis, tnt-panther, zipnerf-nyc (manifest dev/tuning). It is NOT part of this group
  and is run only if added later as a separate group.
- Data: the corpus tars, sha256-checked at pull (`runs/q1/heldout/scenes/urls.txt`), with images and poses as shipped.
  T&T is 960×540 (1024×540 for m60) with our native SfM poses (sfm-gates-v3). Zip-NeRF uses images_4 with the release
  COLMAP. Both arms use identical poses. All 13 tracker scenes are excluded.
- Licenses: T&T intermediate is "CC BY 4.0 header, but the License Grant limits use to non-commercial scientific
  research", so results are for paper and internal use only, never marketing. Zip-NeRF is CC BY 4.0 (Barron et al.,
  SMERF/Zip-NeRF release).

## Split (identical to GSEQ-SWEEP)
H = every8_idx0_sorted held-out list in sorted order. DEV = H[1::2], TEST = H[0::2]. Training views = all views not in
H, so neither arm trains on DEV or TEST. All selection uses DEV; every reported number and verdict uses TEST.

## Part A/B: ours vs gsplat-MCMC, cap ladders (GSEQ-SWEEP harness, verbatim)
- Harness: research/gseq-sweep/{run.py, gs_run.py, launch.sh} (GSEQ-SWEEP lane, pubs:p0:GS-EQ-SWEEP), unchanged. Only
  the scene list and data dir differ, plus labels (claim id).
- Runners: two L40S runners per (scene, seed), fal py image v2, `--backend fal`.
  - Part A: gsplat 3M plus the full ours ladder.
  - Part B: gsplat 300k/600k/1M plus the full ours ladder.
- Arms:
  - gsplat 1.5.3 simple_trainer MCMC preset, data_factor 1, caps 300k/600k/1M/3M. Stock 1M is one of these.
  - Ours: brush-cli main historical-build-096 at --max-splats 300k/500k/800k/1.2M/2M. This is the same binary file as GSEQ-SWEEP, with
    its sha256 bound before the first submit.
- Seeds 1/2/3. Odd seeds run gsplat ascending then ours ascending; even seeds run the exact reverse. Warm-ups as in
  GS-EQ: ours one 30k at 1.2M, gsplat 2k steps.
- Clocks, memory and scoring follow GSEQ-SWEEP:
  - train_s runs from the first to the last optimizer step; wall_s is secondary.
  - Memory is the NVML board peak minus idle (100 ms sampler, MiB, same sampler for both arms).
  - Scoring uses splat-perf full-res GPU LPIPS-VGG (lpips_scale=full). A run fails if n_scored != n_expected.
- Readouts and decision rule: GSEQ-SWEEP's (a)/(b)/(c) and its PRIMARY/ROBUSTNESS rule verbatim, applied to the 4
  verdict scenes:
  - Row AHEAD if FASTER on at least 3/4 scenes and SLOWER on 0. This is the 4-scene analogue of GSEQ-SWEEP's ≥4/6.
  - Row PARITY if FASTER ≥ SLOWER.
  - Otherwise BEHIND.
  - Quality claims require rule:seed-gate.

## Part C: ours at frozen defaults vs gsplat at our count
- One extra runner per (scene, seed), same harness, with this arm list: ours with no --max-splats (default auto budget),
  then gsplat-MCMC with cap_max = ours' achieved count on that runner.
- The order is fixed (ours first, since the cap comes from ours' output), and this is disclosed.
- Reported on TEST: PSNR/SSIM/LPIPS, train_s ratio and NVML ratio. This part is descriptive only and is not in the row
  verdict, which comes from the ladders.
- Stock 1M is covered by part B.

## Part D: ours vs Spirula (SP-EQ design on held-out scenes)
- SP-EQ-r3 design: one paired drive.py queue group per (scene, seed) on one L40S engine-image runner.
  - Spirula v2026.10.6 (b9c8042, stock defaults; binary only, never its source).
  - Ours brush-cli git:historical-build-096 at 300k/500k/800k/1.2M.
- 4 verdict scenes × seeds 1/2/3.
- Budget pick on DEV (smallest ours cap with 3-seed mean DEV LPIPS ≤ Spirula's). Ratio = train_s, both arms. Verdict
  uses TEST, with SP-EQ-r3's FASTER/TIE/SLOWER thresholds.
- Row AHEAD if FASTER on at least 3/4 scenes and SLOWER on 0.

## Infra
- A runner that fails on infra is rerun whole and logged. Pass/fail is judged without reference to quality or speed.
- No scene is swapped. A scene that fails twice is reported as failed.
- Per-view v1 rows with seed echo. Fal only (`--backend fal`).

Estimated cost: A+B ≈ $30, C ≈ $6, D ≈ $8; ≈ $44 total.
