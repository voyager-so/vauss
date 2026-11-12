# SfM measurements

## Incremental default: pose correctness (2026-10-08, Pacific)

The global truck model has five adjacent camera-rotation jumps above 90 degrees.
O1's training runs on those poses scored approximately 5 dB below the historical
Spirula-pose model. Low observation reprojection alone did not reveal this failure.
The default is now incremental; global mapping requires explicit selection.

A fresh CPU-only PINHOLE reconstruction with default exhaustive matching and
incremental mapping registered 63/63, produced 11,439 points (10,814 multiview),
and had 0.446072 px mean observation reprojection. Its maximum adjacent rotation
was 49.412 degrees, with no jumps above 90 degrees. CPU wall was 31.155 s
(extraction 6.798, matching 18.616, mapping 5.603); this is a correctness run,
not a GPU speed comparison. Build passed; no lint or test suite was run.

The historical Spirula reference itself has a suspect camera at `000165.jpg`:
its rotations to `000161.jpg` and `000169.jpg` are 170.419 and 155.051 degrees.
Native incremental gives 14.79 and 23.26 degrees. Visually, `000165.jpg` and
`000241.jpg` show nearby viewpoints of the same truck side; native relative
rotation is 15.59 degrees versus the reference's 162.28 degrees. Do not require
blind agreement with this reference camera. Excluding it from the Sim3 fit,
the other 62 native cameras agree within 0.224 degrees and center RMS 0.000531
(reference extent 2.493); held-out cameras agree within 0.219 degrees.

Evidence under `runs/perf-sfm/`: `truck-temporal-audit.json`,
`truck-incremental-fresh-pinhole/{reconstruction.json,pose-audit.json,
pose-audit-reference-outlier-excluded.json}`, and `align-cameras.py`.
The dense two-view default remains held pending new training A/B on correct poses.

## Historical exploratory runs

RTX 5080 16 GB / Ryzen 9 9950X, 2026-10-08. Same images as the historical
Spirula release v2026.10.6 runs. Truck has 63 images at 960×535; Pegasus has
69 frames at 960 pixels wide. SfM sees all frames; downstream holdout is sorted
filename index modulo 8. Reprojection is always recomputed per observation with
`splat-eval`. Times include subprocess startup, calibration and health reporting,
exclude GPU-lock wait. Every GPU invocation held one shared GPU lock file.

These are **exploratory**, not controlled speedup claims: concurrent swarm builds
and evaluation caused CPU load averages over 60 on 32 logical cores. This COLMAP
build falls back to CPU for sparse Ceres solves (no cuDSS).

| Scene | Configuration | Registered | Mean reproj px | Points | Median track | Health | Wall s |
|---|---|---:|---:|---:|---:|---|---:|
| truck-63 | Spirula historical | 63/63 | 0.36 | 21,188 | 2 | 6/7 | 8.825 |
| truck-63 | Global, exhaustive, 8192 features, 8 threads | 63/63 | 0.430 | 10,254 | 4 | pass | 22.596 |
| truck-63 | Global, sequential overlap 8, 4096 features, BA 1×50 | 63/63 | 0.311 | 3,971 | 3 | fail | 51.903 |
| truck-63 | Global, exhaustive, 2048 features, 16 threads, BA 2×50 | 63/63 | 0.487 | 5,690 | 4 | fail | 12.807 |
| truck-63 | Global, exhaustive, 8192 features, 16 threads, BA 2×100, BLAS/OMP 1 | 63/63 | 0.436 | 9,141 | 4 | pass | 29.892 |
| pegasus-69 | Spirula historical | 69/69 | 0.363 | 19,816 | — | — | 8.325 |
| pegasus-69 | Global, sequential overlap 10, 4096 features, OPENCV, BA 2×50 | 69/69 | 0.336 | 11,426 | 4 | pass | 58.618 |

No configuration yet beats the reference on speed and quality together. The
truck sequential candidate's lower residual does not compensate for poor feature
coverage. Downstream PSNR/SSIM/LPIPS remains pending O1's pose A/B runs.

Full options, argv, stage logs and `reconstruction.json` were retained per run
(`runs/perf-sfm/{truck-global-default,truck-seq4096,truck-exhaustive2048,pegasus-seq4096}`),
with leaderboard rows in `runs/perf/sfm.jsonl`, on the benchmark host (not in the repo).

The initial ALIKED-N16ROT GPU probe aborted in ONNX Runtime 1.30 because its dynamic
loader could not find unversioned `libcudnn.so`. This is an environment dependency,
not evidence about feature quality. No learned-feature performance claim is made
from that failed probe. Models and their SHA-256 are recorded in README.md.

The cuDNN probe was fixed with an O5-local `libcudnn.so` symlink to the existing
`stackA/venv153/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9`, and
`LD_LIBRARY_PATH` containing that shim directory and the cuDNN directory. ALIKED
then extracted 512 features on each of three truck images in 1.41 s total. This
is an extraction correctness probe, not a full SfM or Rust `ort` implementation
benchmark. Full ALIKED/LightGlue reconstruction is still unmeasured. The working
native ORT integration avoids duplicating COLMAP's feature/database format.

A CPU-only diagnostic on the existing truck match database took 46.24 s for global
mapping with one Ceres thread. That excludes extraction/matching and is not a
leaderboard row or speedup. A small SIFT vocabulary built from 16,384 existing
truck descriptors (256 words, 10 iterations, 4 threads) took 1.05 s and 168 KiB;
visual loop-retrieval reconstruction remains untested. Training that vocabulary
must be included in a fresh-run timing if used for a comparison.

## Deferred experiments

The coordinator reserved the GPU for O1's 19-run baseline leaderboard at 18:34
UTC. Three queued experiments were cancelled before starting; there are no O5
GPU processes or locks left behind. Resume after O1 releases that priority:

1. Truck: exhaustive SIFT 8192, 16 threads, BA 2×100, `--ransac-max-trials 1000`,
   with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Compare to `truck-threads1`
   while controlling CPU contention, then repeat a paired Spirula baseline.
2. Pegasus: `--matching sequential --overlap 8 --max-features 1024 --video
   --camera-model OPENCV`, ALIKED/LightGlue local models and cuDNN loader fix.
3. Goresbridge: exact 115-frame `runs/bench-lane/frames/gor_dedup` input,
   incremental mapping, sequential overlap 8, 4096 SIFT features, OPENCV.
4. Copped Hall: exact `runs/bench-lane/frames/cop_ok` input (150 frames), paired
   Spirula/native SfM. Additional prepared inputs are `runs/perf-sfm/frames/groei`
   (81 at 15 fps) and `frames/yagan` (73 at 3 fps), ffmpeg scaled to 960 px. Use
   these identical frames for both engines; Yagan is a low-parallax failure case.
5. O1 downstream pose A/B: `pegasus-seq4096/sparse/0` versus
   `build/pegasus/sparse/0`, both with `build/pegasus/images`, index-modulo-8 holdout.

No defaults were promoted based on the exploratory times. Library validation:
4 tests pass, including actual worker death while a native child retains the GPU
lock; original input files and previous run evidence are preserved.

After these exploratory runs, the coordinator introduced `splatcap` (20/32 CPU
cores; memory 60 GB soft / 72 GB hard). Prefix every resumed heavy command with
`splatcap`, including COLMAP, Spirula, training/evaluation, ffmpeg and cargo;
`CARGO_BUILD_JOBS` must be at most 4 and use `RUSTC_WRAPPER=sccache`. Re-run both
engines under that same cap before making any speed comparison. The rows above
precede the cap and cannot serve as the controlled baseline for capped runs.

## Incremental mapping with default dense initialization (2026-10-08)

Rebased the pose-fixed two-view pass onto incremental mapping and canonical export.
The selected mapped component is archived at `mapped/0`; the initialized model is
exported at `sparse/0`. No density flag is exposed. The initialization stage uses
CPU triangulation/BA and rejects any change to camera intrinsics or poses.

One release build and one fresh truck-63 CPU reconstruction under `splatcap`
(PINHOLE, exhaustive, 8,192 SIFT features, 960px, 8 threads, BLAS/OMP 1) produced
63/63 cameras, 25,489 points (11,606 multiview), 0.393449 px mean recomputed
reprojection, and zero continuity flips. All health gates pass. Intrinsics and
poses are exactly identical before/after initialization.

| Stage | Seconds |
|---|---:|
| CPU feature extraction | 10.800 |
| CPU exhaustive matching | 38.100 |
| Incremental mapping | 25.150 |
| Pose-fixed initialization | **3.332** |
| Full wall including health/export | 78.671 |

This is a CPU correctness run on a shared host, not a paired L40S speed result.
The 3.332 seconds measures only the added native initialization process;
health/audit/export overhead is included in full wall. Evidence and exact argv:
`runs/perf-sfm/truck-dense-default-fresh/{reconstruction.json,audit.json}`.
The same-worker three-seed training comparison is pending; interim unpaired
7k rows favor dense over sparse by 0.318 dB on truck and 0.553 dB on Pegasus.
Do not treat those interim rows as a qualified paired win.
