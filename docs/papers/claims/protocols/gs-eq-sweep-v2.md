# pubs:p0:GS-EQ-SWEEP-v2 — pre-registered CONFIRMATION of GS-EQ-SWEEP, draft 08:20 PT 2026-10-09

Context (disclosed): registered after GS-EQ-SWEEP (PARITY 5/0/1, bicycle not reached at ours 2M; pubs historical-build-070) and its
descriptive extension were seen. This is a fresh confirmation with new seeds and an extended ours ladder; it never
pools with GS-EQ-SWEEP, which stands as recorded.

Design = pubs:p0:GS-EQ-SWEEP exactly, except:
- Seeds 11 / 12 / 13 (not used by any GS-EQ / GS-EQ-SWEEP / -ext / -brush group). Order: odd seeds gsplat caps ascending
  then ours ascending; even seeds exact reverse.
- Ours ladder on EVERY runner: --max-splats 300k / 500k / 800k / 1.2M / 2M / 3M / 4M (historical-build-096, brush 933d67da).
- gsplat-MCMC 1.5.3 caps 300k / 600k / 1M / 3M PLUS 5M and 8M (pubs condition 08:12: no rung above 3M would leave
  gsplat's ladder truncated at its best). Runners per (scene, seed): A2 = gsplat 3M + 5M, B2 = gsplat 300k / 600k / 1M,
  C2 = gsplat 8M; each runner carries the full ours ladder. 3 runners x 18 = 54.
- Hardware RTX PRO 6000, fal py image v2 (`--gpu rtxpro6000 --no-spill --py-version v2`), as GS-EQ-SWEEP.
Unchanged: 6 scenes (bicycle, garden, counter, room, kitchen, truck-full), 30k, warm-ups (ours one 30k at 1.2M; gsplat
2k), clocks (train_s primary, wall_s secondary), NVML peak-minus-idle, scorer splat-perf 4faae9d2 full-res LPIPS-VGG,
per-view rows, DEV = H[1::2] / TEST = H[0::2], recipe citations and health metric as in protocols/gs-eq-sweep-ext.md.
Rule: identical selection and per-scene verdicts to GS-EQ-SWEEP (primary = gsplat's DEV-best cap vs ours' smallest DEV
match; robustness = worst over the 4 gsplat caps; FASTER < 0.95 with TEST hold >= 2/3; SLOWER > 1.05 or not reached).
Truncation: if gsplat* (DEV-best) is its top rung (8M) on a scene, that scene is labelled "gsplat ladder may be
truncated" and does not count as FASTER toward AHEAD (counted as TIE in the tally).
Row: AHEAD only if FASTER on 6/6 scenes (every scene reached and faster); PARITY if FASTER >= SLOWER; else BEHIND.
Same thresholds for the robustness row.
Upstream Brush v0.3.0 is NOT folded in: its separate descriptive group (pubs:p0:GS-EQ-SWEEP-brush, historical-build-070) was
already running when this was ordered and is not duplicated.
Budget: ~$120 (54 runners: A2 ~45 min, B2 ~33 min, C2 ~55 min at ~$2.9/h RTX PRO); <= 54 in flight = <= $160/h, under
the $280/h cap. Launch 09:15 PT; results ~10:10-10:20 PT. Infra rerun = whole runner.
