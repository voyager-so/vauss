# pubs:p0:GS-EQ-SWEEP-fastergs — DESCRIPTIVE group (lead-requested), draft 08:58 PT 2026-10-09

Question: time for ours to reach Faster-GS MCMC's quality (CVPR'26, Apache-2.0; reports 0.23x 3DGS-MCMC time; the
strongest published speed competitor). Never pooled; cannot change any registered verdict. Requested after
GS-EQ-SWEEP / SP-EQ-fair results were seen (disclosed); Faster-GS has not been timed in this protocol before.

Competitor: Faster-GS nerficg-project/faster-gaussian-splatting @ d7fae871c45384b67b4c348f804762c923949018 on NeRFICG
@ e458ac6e41810ad6d9405687c1e698fadd28dac5 (assembled exactly as ledger row FG1: sklearn kNN, Inria-licensed files left
out). Recipe source (rule:competitor-recipe-source): its shipped config fastergs_garden.yaml with the code's own MCMC
switches: USE_MCMC True (yaml:79, Trainer.py:31), DENSIFICATION_END_ITERATION 24900 (yaml:68, Trainer.py:19),
MORTON_ORDERING_END_ITERATION 25000 (yaml:84, Trainer.py:36), opacity/scale reg 0.01 (yaml:100-101, Trainer.py:54-55),
opacity LR 0.05 (yaml:108, Trainer.py:63), MAX_PRIMITIVES 1,000,000 default (yaml:80, Trainer.py:32). Full per-line
citations in research/gseq-sweep/fastergs_run.py's docstring. Two Faster-GS arms: shipped default cap (1M) and 3M (the
only change is MAX_PRIMITIVES). Changes vs recipe: seed, 30k steps (= shipped), eval/ckpt/GUI off, data path, split
asserted = every8_idx0_sorted, outputs to /tmp. Images at their stored benchmark resolution (as FG1).
Ours: historical-build-096 (933d67da) at 300k / 500k / 800k / 1.2M / 2M / 3M / 4M.
Design: as GS-EQ-SWEEP-v2: 6 scenes x seeds 11/12/13, RTX PRO 6000 py-v2, one runner per (scene, seed) = part F, scorer
splat-perf 4faae9d2 full-res LPIPS-VGG, DEV = H[1::2] / TEST = H[0::2], NVML peak-minus-idle. Warm-up: ours one 30k run
at 1.2M; Faster-GS one 2k-step run (absorbs its CUDA build and JIT). Clock: train_s for both (ours train_only_s;
Faster-GS cuda-synced perf_counter around NeRFICG's training loop, Trainer.py:236-251; setup/preload/finalize printed
separately and excluded) plus whole wall_s.
Rule (descriptive): target = Faster-GS DEV-best arm (of 1M default / 3M), 3-seed mean DEV LPIPS; ours = smallest cap
reaching it on DEV; train_s ratio; FASTER < 0.95 with TEST hold >= 2/3, SLOWER > 1.05 or not reached, else TIE; also the
same vs the 1M default alone; tally only. If Faster-GS's DEV-best is its 3M arm, label that scene "ladder may be
truncated".
Disclosures: Faster-GS sets PyTorch expandable_segments and preloads training images into VRAM (Trainer.py:72-76,
yaml:28), so memory is NVML board peak on both arms with that noted; Faster-GS on RTX PRO 6000 is a first run of its
CUDA extensions on that GPU (FG1 was L40S).
Budget ~$45 (18 runners x ~50 min at ~$2.9/h). Launch on binding (~09:05); results ~10:00.
