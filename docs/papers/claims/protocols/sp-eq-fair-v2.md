# pubs:p0:SP-EQ-fair-v2 — pre-registered confirmation of SP-EQ-fair (draft 08:25 PT 2026-10-09; pending lead approval)

Context (disclosed): registered after SP-EQ-fair (AHEAD 5/1/0, playroom Spirula collapse in 2/3 groups) was seen.
Trigger: Spirula's DEV-best cap was its top rung (3M) on 5/6 scenes, so its ladder may have been truncated at its best
(the same weakness pubs flagged for gsplat). Never pooled with SP-EQ-fair.
Design = pubs:p0:SP-EQ-fair exactly (RTX PRO 6000 engine image, drive.py queue paired group, SP-EQ r3 harness, splat-perf
historical-build-096 e1358e26, DEV = H[1::2] / TEST = H[0::2], same 6 scenes) except:
- Seeds/runs 11 / 12 / 13 (ours --seed; Spirula has no seed flag).
- Spirula v2026.10.6 --cap-max 300k / 500k / 1M / 3M / 5M / 8M (Spirula's own presets include 3M/50k and 10M/80k; 30k
  steps kept for all arms; disclosed).
- Ours historical-build-096 (933d67da) --max-splats 300k / 500k / 800k / 1M / 1.2M / 2M / 3M / 4M + default (0).
One runner per (scene, run) with all 15 arms; 18 runners.
Rule: primary / robustness / equal-cap (500k, 1M) / defaults exactly as SP-EQ-fair, except AHEAD only if FASTER on 6/6;
truncation clause: if Spirula* is its top rung (8M) on a scene, that scene is "Spirula ladder may be truncated" and
counts TIE, never FASTER. Health: Spirula dead/relocated counts per 100 steps from its train.log (collapse = train PSNR
falling > 5 dB after 8k steps or dead > 50% of cap); a collapsed Spirula arm is disclosed, not excluded, and every
headline must also hold with that scene excluded from both arms.
Budget ~$55 (18 runners x ~60 min at ~$2.9/h); launch 09:15 PT; results ~10:20 PT.
