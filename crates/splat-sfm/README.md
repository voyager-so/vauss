# splat-sfm

Rust orchestration of a supplied **COLMAP 4.2.1** binary, using CUDA SIFT and the
incremental mapper by default. `--mapper global` selects the experimental
integrated global mapper (GLOMAP). No COLMAP, GLOMAP or
Spirula source is bundled. Health and per-observation reprojection are computed by
`splat-eval`; stored point errors are not compared against observation averages.

```rust,ignore
let mut opts = splat_sfm::ReconstructOptions::new("/runs/fresh-sfm");
opts.colmap = "/tools/colmap/bin/colmap".into();
opts.gpu_lock = Some("/shared/gpu.lock".into());
let result = splat_sfm::reconstruct(frames_dir, &opts)?;
// result.model_dir: largest component; result.health includes registration/gates/tracks
```

`output_dir` must not exist: the pipeline never reuses cached features or an old
model when reporting wall time. It writes `options.json`, per-stage logs and
`reconstruction.json` with exact argv, stage times, total wall time and separately
measured lock wait. The largest registered component is exported as `sparse/0`;
other components are preserved in sibling `components/`. The default final CPU
triangulation retains two-view points with fixed poses and intrinsics; the original
selected model is preserved in `mapped/0`. A camera audit rejects any movement. Input images stay in place. For Brush, the caller assembles a
dataset with `images/` linked to those images and `sparse/0/` pointing to
`result.model_dir`, keeping PLYs out of that dataset root.

`commands(frames_dir, opts)` exposes shell-free stage commands for a caller that
needs its own process runner/cancellation. Call `finalize_model` after those stages
to select the largest component and export it as `sparse/0`. Then run
`triangulation_command` into a fresh separate directory, validate with
`ensure_fixed_cameras`, archive the mapped model, and put the initialized model
at `sparse/0`. `best_model` only inspects components without changing files. Synchronous `reconstruct` blocks until each subprocess finishes;
job runtimes should supervise its process group as they do other backends.

```sh
RUSTC_WRAPPER=sccache CARGO_BUILD_JOBS=4 cargo build -p splat-sfm --release
flock /tmp/splat-gpu.lock target/release/splat-sfm \
  data/truck-small/images \
  --colmap /path/to/colmap-4.2.1/bin/colmap \
  --output runs/perf-sfm/truck-fresh
```

Use **either** external `flock` **or** `--gpu-lock PATH`, never both for the same
lock. The lock covers extraction, matching and mapping because GPU BA/global
positioning can use the device. Final triangulation and health use CPU after releasing the lock. On Unix, the native child inherits the
lock so a worker crash cannot release it while COLMAP is still running.
`--mask-dir PATH` passes COLMAP feature masks (black excludes features; mask
relative paths append `.png` to the full image name, e.g. `frame.jpg.png`).

For video use `--video --matching sequential`; filenames must sort in temporal
order. Sequential overlap and quadratic long-range candidates use temporal order
as a pair-selection prior, not fabricated pose coordinates. `--vocab-tree PATH`
enables COLMAP visual loop retrieval using an explicitly supplied SIFT vocabulary
tree. Without that artifact there is no claim of visual loop detection.

`--aliked-model PATH --lightglue-model PATH` selects ALIKED-N16ROT/LightGlue through
COLMAP's ONNX Runtime; weights must be explicit local files. The default uses no
learned weights. The official COLMAP 3.13.0 artifacts used for experiments are:

| Model | SHA-256 |
|---|---|
| aliked-n16rot.onnx | `39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547` |
| aliked-lightglue.onnx | `b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d` |

Sources: [COLMAP CLI](https://colmap.github.io/cli.html),
[GLOMAP](https://github.com/colmap/glomap),
[ALIKED BSD-3](https://github.com/Shiaoming/ALIKED/blob/main/LICENSE),
[LightGlue Apache-2.0](https://github.com/cvg/LightGlue/blob/main/LICENSE),
[ONNX artifacts](https://github.com/colmap/colmap/releases/tag/3.13.0).
No binary or model is redistributed by this crate. Distribution pins and bundled
notices belong to `splat-fetch` (R7).

The current Linux development binary reports CUDA but its Ceres lacks cuDSS:
GPU sparse BA/global positioning silently fall back to CPU (visible in stage
logs). Do not label it GPU BA or assume GPU flock removes CPU contention from
benchmarks. Results and limitations live in `BENCHMARKS.md`.
