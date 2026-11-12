# X24 registration draft: 2609.05255's released nerficg config × our recipe (cap × tail), M360-9

Status: DRAFT for publications. Bind before launch. Launch moved to **10:30 PT**, together with X21, under the ≤ $150/h cap (sub-lead 08:00).
Owner: X24 lane, reporting to the SOTA quality sub-lead. Claim group `x24-nerficg-paper-cap-x-tail-m360` (T&T: `x24-nerficg-paper-cap-x-tail-tnt-descriptive`).
Proposed class: PROVISIONAL at most. It attributes a P7 claim; publications sets the class.

## Question and the claim it attributes

P7 (sub-lead): "Spending the 16 GB budget on primitives (plus a colour-only tail) lifts the released nerficg config to NHT parity, with lower LPIPS."

STACK runs the same code as arXiv 2609.05255 (`nerficg-project/efficient-gaussian-appearance@4efbe677` + NeRFICG `e458ac6e`) and starts from the same per-scene YAML. The desk diff (`research/x24/config-diff.md`, main historical-build-018) shows the two differ in exactly two method choices:
- **cap**: `TRAINING.MAX_PRIMITIVES`, upstream per scene vs our flat 11M;
- **tail**: our final 3 000 of 30 000 steps update colour and PPISP only, with geometry frozen and geometry regularisers at 0 (`color_tail.py` 9182be88).

This registration is the full 2×2 of those two factors. Each cell runs the released code, scored through the same harness and scorer as STACK.

## Arms

Every arm loads `paper_configs/default/neural/<scene>.yaml @4efbe677` and sets only GLOBAL.RANDOM_SEED, the backup-render flags (off), MODEL_NAME and DATASET.PATH. `x24_prepare.py` asserts this against the upstream YAML. The cap is applied through stack_run's `--cap`, and the adapter asserts each arm's registered config sha.

| cell | cap | tail | runs in |
|---|---|---|---|
| **paper** (their released config) | upstream per scene, `MAX_PRIMITIVES` at line 102: bicycle 6.0M, garden 5.0M, stump 4.5M, treehill 3.5M, flowers 3.0M, room/counter/kitchen/bonsai 1.5M | none | X24 exec |
| **paper_tail** | upstream | 3 000 | X24 exec |
| **paper_11m** | 11M | none | X24 exec |
| **STACK** (control) | 11M | 3 000 | **X21's control arm**, same scene and seed (X21 2×2, `x21-stack-regs-x-filter3d-m360`) |

The paper_11m config sha equals X21's control config sha (outdoor `4554c537edd5…`, indoor `717619ad5813…` at s101): it is STACK without the tail.

Paper-arm config shas (s101; only RANDOM_SEED changes across seeds):
- bicycle `f39812294d46`, flowers `e8fa66cbad36`, garden `2dd33759d257`, stump `632df5045eff`, treehill `f070af9fe05a`;
- room/counter/kitchen/bonsai `92d07ec01ac7`.

The full list will be in `$SPLAT_DATA/research/x24/m360/stack-configs.json` at binding.

Within each exec, the arm order rotates by seed to spread position effects:
- s101: paper, paper_tail, paper_11m;
- s102: paper_tail, paper_11m, paper;
- s103: paper_11m, paper, paper_tail.

## Sharing the control with X21: two options (publications decides)

**(A) Ride inside X21's execs (7 arms per exec).** Not feasible on the 7 200 s timeout.
- X21's exec already runs 3 000–6 200 s.
- Our three arms add about 1 900 s (outdoor) to 3 400 s (indoor).
- Indoor execs would reach about 9 000–9 600 s, past the timeout.
- A hybrid (outdoor inside X21, indoor separate) splits the pairing rule by scene, so I don't recommend it.

**(B) Separate X24 execs in the same 10:30 window, pointing at X21's control (recommended).**
- The cap effect (paper_11m − paper) and the tail effect (paper_tail − paper) are **same-exec** contrasts.
- Only the contrasts that involve the STACK cell are **cross-runner**: the headline STACK − paper and the interaction. They pair X24 and X21 execs by scene and seed, on the same runner class, bundle files, seeds and window. They are not the same allocation.
- Batch-to-batch moves of ±0.14–0.42 dB have been seen on this fleet. We state this pairing as cross-runner, and publications may require the headline to be called PROVISIONAL-cross-runner.

**(B+) (B) plus our own same-exec STACK arm.** This makes the headline same-exec. It adds about $21 (≈ 8 900 s per seed), for ≈ $81 total with T&T or ≈ $71 without, which is over the ≤ $80 cap with T&T. Offered only if publications rules that the headline must be same-exec. Then T&T would be dropped.

## Scenes, seeds

- **M360-9:** bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai. Corpus tars, scales (images_4 outdoor, images_2 indoor), test views and views_hash are identical to x5v2 and X21.
- **Seeds 101, 102, 103**, the same as X21 (sub-lead 08:00), so the STACK cell pairs by seed. Checked: no other claim group and no registry row uses 101–103 apart from X21's 2×2. The brief's 111–113 are superseded.
- **T&T (DESCRIPTIVE ONLY, sub-lead 08:00 (b)):** tnt-family, tnt-horse, tnt-m60. Upstream has no recipe for these scenes. Upstream T&T ships only `truck.yaml` and `train.yaml`, whose `TanksAndTemples_3DGS` loader needs a single PINHOLE camera with a centred principal point (NeRFICG `src/Datasets/TanksAndTemples_3DGS.py:42-58`). So:
  - **paper** = truck.yaml's cap of 2.5M (`truck.yaml:102`) on the MipNeRF360 loader at scale 1.0. This is the same path as STACK v3: template config, `ORACLE_STACK_TEMPLATE=bicycle`, v3 adapter 394cbff8.
  - **stack** = 11M + tail, as a same-exec arm. X21 does not run T&T.
  - This is 9 execs with 2 arms each, and no gate. Dropped first if the budget is tight.

## Harness (pinned; final sha bound at registration)

- `research/x24/x24_prepare.py`, `x24_pair_run.py`, `x24_u8.py` on research-x24 (this commit; merged to main before binding).
- **Bundle = X21's bundle**, built by the same code:
  - the x5v2 bundle-trim: paired_adapter a462ce11, paired_run 2e194e72, paired_score 6fac406b, stack_run 02ab93e2, run 98d50885, color_tail 9182be88, runtime.sh 041ec20a, ballast 6eadf969;
  - `x21_stack_run.py` 4d8427a1 and `x21_color_tail.py` 4f3ce0fd, generated by importing x21_prepare.py's own replacement tables;
  - `x21_{pair_run,score,health}.py` a4034bfb / eb659b49 / 31c8b5a5.
- **X24 additions:**
  - `x24_pair_run.py` 40d0757e = x21_pair_run plus three lines: (1) per-arm `--cap`/`--tail` from the job (paired_run hard-codes 11M/3000); (2) run x24_u8 after the common scorer; (3) record cap/tail.
  - `x24_u8.py` 39b3ce6a: descriptive only.
- `ORACLE_STACK_MEMTRIM=full` in every arm. It is eval-only and leaves training unchanged, as in x5v2 and X21.
- The T&T bundle differs only in paired_adapter 394cbff8 (v3).
- Runner class: fal py-v3 RTX PRO 6000. 16 GiB ballast certificate per exec (`stack_ballast.py`, leave_gib 16).

## Scorer

- `paired_score.py` 6fac406b, run through `x21_score.py` (arms = job['arms'], maths unchanged).
- It scores float32 renders against the decoded corpus JPEG targets: full-scale torchmetrics VGG and Alex LPIPS with `normalize=True`, SSIM, and per-view PSNR, all views weighted equally.
- The same views_hash is asserted in every arm and matches X21.
- **8-bit line (sub-lead (c), descriptive):** `x24_u8.py` rescores the same exports with the prediction quantised to 8 bit (round(x·255)/255), as the PNG save in upstream NeRFICG does (`Renderer.py:261`, scored at `:105-140`). It is reported for every arm as float-minus-u8. It never gates anything.

## Contrasts and decision rules (rule:seed-gate)

All contrasts use per-seed values of the 9-scene mean (n = 3 seeds) and the sample SD (ddof = 1).

| contrast | definition | pairing |
|---|---|---|
| **Headline R (recipe)** | STACK − paper | cross-runner under (B) |
| Cap effect C | paper_11m − paper | same exec |
| Tail effect T | paper_tail − paper | same exec |
| Tail at 11M | STACK − paper_11m | cross-runner under (B) |
| Interaction I | (STACK − paper_11m) − (paper_tail − paper) | cross-runner under (B) |

**PSNR gate, for each of R, C, T:**
- PASS iff mean Δ > 2·SD(Δ);
- AND no scene's 3-seed mean Δ < −0.05 dB.

**LPIPS-VGG gate, reported separately because LPIPS is the claim of interest, for each of R, C, T:**
- "lower LPIPS" iff −mean ΔLPIPS > 2·SD(ΔLPIPS);
- AND no scene's 3-seed mean ΔLPIPS > +0.002.

**Interaction:** called non-zero only if |I| > 2·SD(I), for PSNR and for LPIPS separately.

**Wording:**
- A PASS on R supports the P7 attribution sentence.
- C and T apportion it.
- If R passes but neither C nor T passes alone, P7 says "the combination", not "primitives".

**Reference rows only, never gated:** NHT (x5v2 group, different seeds and batch) and the 2609.05255 Table 1 numbers (28.55 / .831 / .227). Their table is on an RTX 4090, scores 8-bit PNGs and averages 5 runs, so it is **never** called same-run.

## Memory and health (reported per arm, every exec)

- **Peak memory column:**
  - NVML increment above the ballast baseline, plus the setup-subtracted NVML (phase peak − setup-phase peak; this removes the ~0.58 GiB monitor offset), from `pair-nvml.json`;
  - torch whole-process peak (`whole_process_torch_peak_gib`).
- `resource_pass` = NVML increment ≤ 15 GiB AND torch ≤ 15 GiB (paired_run).
- **Health:** `x21_health` MCMC relocated fraction per 100-step call. Reported as mean and max of dead/n, and also as dead/cap, **flagged if > 0.5 of cap**, with final count vs cap. Symmetric in every arm.

## Failures (declared up front, mirroring X21)

- Room: if any room cell fails `resource_pass` or OOMs on any seed (including X21's control), room is excluded from every X24 contrast. The verdict then becomes an 8-scene PROVISIONAL, with room shown as failed.
- Any other failed exec is reported with no retry. A missing cell means no verdict for that seed's contrasts that need it.
- If X21's control is missing for a scene/seed, R, the tail-at-11M contrast and I have no verdict for that seed. C and T stand.

## Cost and wall

Per-arm wall estimates come from x5v2 STACK arm walls in the same harness and allocation type: the 9-scene sum is 8 859 s per seed (bicycle 775 … kitchen 1 619). Paper-cap trainings run in 250–365 s (B11 final30k, unballasted). The CPU-LPIPS eval is added on top; paper arms are estimated at about 0.55× the STACK wall. Rate $0.00080/s.

| option | per seed | total (3 seeds) | T&T descriptive | all-in | longest exec | wall at ≤ 20 concurrent X24 runners |
|---|---|---|---|---|---|---|
| **(B) recommended** | ≈ 21 000 s | **≈ $50 (range $42–60)** | ≈ $10 (9 × ~1 350 s) | **≈ $60 (≤ $72)** | ≈ 3 700 s (kitchen) | ≈ 1.6–1.8 h (2 waves; ≈ $58/h, which leaves ≈ $90/h for X21's ≈ $78/h within $150/h) |
| (B+) own STACK arm | ≈ 30 000 s | ≈ $71 | dropped | ≈ $71 | ≈ 5 300 s | ≈ 2.2 h |
| (A) inside X21 | n/a | — | — | — | indoor ≈ 9 000–9 600 s > 7 200 s timeout | infeasible |

Reserve cap: **$80** (lead approval). The launcher is a copy of x21_gate/x21_launch with owner `x24` and the same free-slot logic. It reserves $3.0 per exec in flight and stops submitting when the reservation plus actual spend would pass $80. T&T is submitted last.

## Disclosures

- **KNN:** the upstream default `_ALLOW_PROPRIETARY_LICENSED_CODE = True` (NeRFICG `src/Optim/knn_utils.py:8`) uses the NC SimpleKNN extension when it is built. Our runtime sets it to False and asserts that SimpleKNN is absent (`stack_run.py:86-89`), so every arm uses sklearn exact KNN: the same initial-scale maths, a different implementation. This applies to all arms, including the "paper" arm.
- **Hardware and eval differ from the paper** (4090, 8-bit PNG scoring, 5 runs). We never compare same-run with their Table 1; the u8 line shows the size of the PNG effect.
- **The STACK recipe (11M cap, tail) was developed on M360 test views** (P7 caveat). The released config was not. This favours the STACK cell on M360, so a held-out T&T confirmation would be needed before any non-provisional claim; the T&T rows here are descriptive only.
- The T&T paper arm uses truck.yaml's cap on scenes that have no upstream recipe, via the MipNeRF360 loader. This is an adaptation and is labelled as such.
- **Licence:** efficient-gaussian-appearance is Apache-2.0 at the root, and research use is fine. Anything ported into our engine needs Apache attribution and a NOTICE entry. Spirula is unrelated and was not opened.
- Diagnostics stay blind: X24 reads no outputs of v3, v2.1, X20, X21 or GSEQ until their verdicts are recorded. X21's control rows are read only after X21's own verdict is recorded, or with publications' release.
