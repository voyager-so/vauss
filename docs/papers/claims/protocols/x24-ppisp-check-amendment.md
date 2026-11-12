# Amendment to x24-nerficg-arm (historical-build-095): ppisp version check before the 10:30 launch

X24, 2026-10-09 ~08:30 PT, for publications. Source: R-LIT 6b (main historical-build-003), via the SOTA sub-lead at 08:27.

**Issue.** NeRFICG installs ppisp unpinned. Our runtime runs ppisp v1.3.0 (`a2473680`), which changed the CRF backward pass. The paper (2609.05255) can only have used v1.2.x. The paper-era commit is `df33809f7` (`ppisp/_version.py` = 1.2.1). The two versions differ in 14 files, including `ppisp_impl.cu` and `ppisp_math_bwd.cuh`.

**Check.** One exec on bonsai (indoor, PPISP-sensitive), seed 131 (fresh, never reused), using the paper config: `paper_configs/default/neural/bonsai.yaml @4efbe677`, cap 1.5M, no tail, 30k steps plus the 5k PPISP controller. It runs two arms that are identical except for ppisp:
- `ppisp_a2473680` uses the runtime's site, as in every STACK run so far;
- `ppisp_df33809f` uses ppisp @df33809f7, built inside the exec from `git archive df33809f7` (no deps, isolated `--target`, prepended to that arm's PYTHONPATH only).

Each arm records the ppisp version it loaded. The harness, bundle and scorer are X24's (`x24_pair_run.py` → `x21_score` → `paired_score` 6fac406b), plus `research/x24/x24_ppisp_run.py` (in-exec build, per-arm PYTHONPATH) and `ppisp_check_prepare.py`. Cost is ~$1–2, in the approved probe class.

**Material (pre-declared; also written into job.json).** The difference is MATERIAL if, on the bonsai scene mean, any of these holds:
- |ΔPSNR| > 0.05 dB;
- |ΔLPIPS-VGG| > 0.002;
- any per-view |ΔPSNR| > 0.5 dB.

- **If material:** the X24 paper arms (paper, paper_tail, paper_11m) run with ppisp df33809f7, and STACK's ppisp version (a2473680) is disclosed.
- **If not material:** every X24 arm keeps a2473680, and this check is disclosed.
- This is a single seed on one scene. No claim beyond this version choice.

**Failure.** If the df33809f7 build or arm fails, all X24 arms keep a2473680, and the failure is disclosed as "paper-era ppisp could not be run in our runtime".
