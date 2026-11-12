# pubs:p0:GS-EQ-SWEEP-brush — DESCRIPTIVE group (requested), draft 08:12 PT 2026-10-09

Question: time for ours to reach upstream Brush's quality at Brush's own defaults (Brush is our fork parent).
Never pooled with GS-EQ-SWEEP; cannot change any registered verdict. Requested after GS-EQ-SWEEP results were seen
(disclosed); Brush has not been run in this protocol before, so no Brush result has been seen.

Competitor: upstream Brush release v0.3.0, official Linux binary brush-app-x86_64-unknown-linux-gnu (brush_app,
sha256 13d28ee06a388bc4e987774e890b594d60a75bba26064e82b4ee338a78f158a4), Apache-2.0. Recipe source
(rule:competitor-recipe-source): Brush has no per-dataset recipe; its documented CLI defaults are used (brush_app --help
@v0.3.0: --total-steps 30000, --max-splats 10000000 default, all LRs/loss weights default). Only data path, --seed k,
--eval-split-every 8 and eval/export flags are passed (export once at 30k for the splat count).
Ours: brush-cli historical-build-096 (933d67da) at --max-splats 300k/500k/800k/1.2M/2M/3M.
Design: as pubs:p0:GS-EQ-SWEEP (6 scenes x seeds 1/2/3, RTX PRO 6000 py-v2, one runner per (scene, seed) = part D,
scorer 4faae9d2 full-res LPIPS-VGG, DEV = H[1::2] / TEST = H[0::2], NVML peak-minus-idle). Warm-up: BOTH engines get one
untimed full 30k run (ours at 1.2M, Brush at its default) to fill their autotune caches.
Clock: upstream Brush logs only whole seconds, so this group uses whole-process wall_s for BOTH arms (ours' wall_s
includes its ~2-4 s startup + final eval; same for Brush).
Rule (descriptive): target = Brush-default 3-seed DEV LPIPS; ours = smallest cap reaching it on DEV; wall ratio = 3-seed
mean ours/Brush; FASTER < 0.95 with TEST hold >= 2/3, SLOWER > 1.05 or not reached, else TIE; tally over 6 scenes.
Also: ours at the cap nearest Brush's final splat count vs Brush (TEST quality, wall, memory).
Health: Brush logs refine splat counts like ours (count trajectory, flag a > 50% drop or final < 50% of its cap).
