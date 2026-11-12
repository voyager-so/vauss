# Memory lane: baseline and first experiment

Source baseline: Brush `1388f74c6fe0236f68ee4915564bf00e9d2e3747`, Burn `faec398`.
This fork inherits Brush's Apache-2.0 attribution. No Spirula implementation code was used.

## What upstream already does

`Splats` packs position, quaternion and log-scale into `[N,10]`, with opacity
`[N]` and RGB SH coefficients `[N,(degree+1)^2,3]`. At SH3 these total 59 floats
(236 bytes) per splat. `AdamState` retains full first moments, but SH second
moments are already reduced to `[N,1,1]`. Thus Adam costs **284 bytes**, not
472 bytes per splat. Three refinement accumulators add 12 bytes, with an
optional mip scale floor adding 4. A full parameter gradient adds 236 bytes.

The upstream `SceneLoader::BatchCache` holds packed CPU image bytes and has a
byte budget; a zero budget streams from disk. It does not keep the dataset in
VRAM. Adding a GPU cache trades GPU capacity for throughput and needs separate
measurement. Canonical half-precision SH needs renderer/backward/export changes;
casting a temporary while retaining the f32 tensor cannot save persistent memory.

## Analytical bytes, not measured GPU capacity

| SH3, 10,000,000 splats | Parameters | Adam | Refinement + mip | Persistent total | With fp32 parameter gradients |
|---|---:|---:|---:|---:|---:|
| Upstream f32 | 2.360 GB | 2.840 GB | 0.160 GB | 5.360 GB | 7.720 GB |
| 8-bit SH first moment | 2.360 GB | 1.415 GB | 0.160 GB | 3.935 GB | 6.295 GB |
| Above + canonical half SH (experimental) | 1.400 GB | 1.415 GB | 0.160 GB | 2.975 GB | 5.335 GB |

Decimal GB. Block size 128, f32 scales; tensor tail rounded up to a complete
block. Geometry moments and reduced SH variance stay f32. The helper
`tensor_memory` checks overflow and makes the terms explicit.

These numbers exclude projected splats, tile intersections, sorting, rendered
images, backward state, reduction scratch, cached allocator pages and simultaneous
old/new parameters during refinement. None establishes that 10M splats train in
16 GB. The raster workload and allocation lifetimes must also be measured.

## Default SH moment storage

SH first moments use block-scaled signed 8-bit storage by default.
`quant_kernels.rs` implements a fused f32 parameter
update and block repack, with four biased signed codes per 32-bit word. One
128-lane workgroup owns each block. Shared-memory reductions choose its f32
scale. The update never allocates a full decoded first-moment tensor. Uniquely
owned parameter, code and scale buffers are reused; aliases receive copies.
The reduced variance
remains f32: it is just four bytes per splat, and quantizing the denominator is
unnecessary for this first experiment. Split/prune decodes, maps rows, then
re-encodes; newly initialized zero moment rows must remain exact zeros.

This is uniform symmetric block quantization inspired by the storage strategy
in [Dettmers et al., 8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861).
It does not implement that paper's nonlinear codebook. The fused kernel uses
six tensor bindings and workgroup barriers, with no subgroup-width assumption.
A storage barrier separates all old-state reads from in-place writes within
each block; blocks own disjoint words and scales.
Refinement still adds decode/encode launches and transient allocations; benchmark
both wall time and peak memory. GPU regression tests cover partial blocks,
all-zero blocks, isolated outliers, 200 SH3 Adam steps across block boundaries,
per-band rates, supplied and fallback variance, parameter/state snapshot aliases,
and refinement shape changes.

## Default packed geometry moments

Transform `[N,10]` and opacity `[N]` Adam moments share one 32-bit word per
value (4 instead of 8 bytes; 44 bytes per splat) on the fused 3DGS path. With
Adam's epsilon `e`, the low half stores `u = m/(sqrt(v)+e)` sign-magnitude with
a 15-bit magnitude over [0, 8] (|u| is at most ~7.3 for beta1 0.9, beta2 0.999;
larger values clamp); the high half stores `l = ln(1 + sqrt(v)/e)` as a 16-bit
code over [0, 64], about 0.1% relative resolution in `sqrt(v)` at every scale.
Decode is `sqrt(v) = e(exp(l)-1)`, `m = u e exp(l)`; word 0 is exactly
`m = v = 0`, so refinement resets rows by zeroing words and appends zero words,
without decoding. `u` is normalized by the stored (quantized) scale, so `m`
carries only `u`'s error. The fused projection-backward + Adam kernel
(`splat-train-render` `bwd/kernels/project_update.rs`, codec in
`bwd/kernels/geo_moments.rs`) decodes, runs f32 Adam and re-encodes with
stochastic rounding from a per-element, per-step hash: one beta2 decay moves
`l` by about 5e-4, under half a code step, so round-to-nearest would freeze
decaying moments and let invisible splats keep a residual step. Morton sorting
permutes words directly; other row maps decode, map and re-encode with
round-to-nearest, which is idempotent for unchanged rows. The unfused surfel
path keeps f32 moments. The normalized-moment/log-scale encoding follows the
idea described in the Spirula reader notes; fixed global code ranges replace
per-block bounds.

## Default canonical half SH storage

The actual SH parameter tensor is stored in f16 after its first optimizer update. There is no f32 master
copy. Projection and view-direction backward load f16 coefficients into f32
arithmetic; SH gradients and all geometry remain f32. The fused optimizer rounds
its f32 result stochastically to adjacent half values using a reproducible hash
of coefficient index and update step. This avoids systematically discarding
updates smaller than half an f16 spacing. Normal/subnormal boundaries are handled
explicitly; non-finite values still reach the existing pruning path.

Degree padding and refinement preserve parameter dtype. Export converts the
readback data to f32 on the CPU, preserving the existing PLY format without a
full f32 GPU tensor. Both storage choices are unconditional; the experimental
environment switches have been removed.
BF16 is not implemented; f16's mantissa offers better coefficient precision at
the same storage cost for the expected SH range.

The initial L40S truck-63 7k comparison completed training, refinement,
evaluation and export with the frozen every8 split (55 train / 8 held out).
O1's `results.jsonl` records:

| Source / storage | PSNR | SSIM | LPIPS-VGG | Wall seconds | Sampled peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 62cac9c / f32 | 21.246045 | 0.792231 | 0.253218 | 86.38 | 2870 |
| 62cac9c / q8 moments | 21.230046 | 0.791576 | 0.254506 | 87.60 | 2838 |
| d1698ca / q8 + f16 SH | 21.304116 | 0.792100 | 0.253239 | 73.64 | 2241 |

Both compressed variants pass the PSNR loss <=0.05 dB and LPIPS increase
<=0.003 gate. Combined storage reduces sampled peak VRAM by 21.9% and wall
time by 14.7% versus f32 in this single pair. Repeat truck, garden and 30k
validation are pending; no cross-scene or 16 GB capacity claim follows from
this result. No test/lint suite was run, per coordinator.
The local 5080 protocol is recorded in `memory-capacity.md`.

## Diagnostic baseline

| Engine | Scene / split | Steps | Final splats | Worker wall | Training wall | Worker peak VRAM | Board delta peak | Quality |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Brush 1388f74 via R3 worker | truck 63 / R3 SfM, every-8 index 0 | 7000 | 304399 | 70.121 s | 67.994 s | 2299 MiB | 2320 MiB | 21.07194 PSNR / 0.77473 SSIM / 0.30375 LPIPS-VGG (splat-eval) |

Output: `runs/o4-memory/brush-truck-7k-baseline/` on the benchmark host.
Exact job, training log, summary and `vram.jsonl` retained. Sampling requested
200 ms intervals, actual 230 samples over 70 seconds due to `nvidia-smi` overhead.
Board baseline 581 MiB, peak 2901 MiB; process allocation is from `nvidia-smi -q -x`.
The whole worker and sampler held the shared exclusive GPU flock.

**Diagnostic only:** this R3 reconstruction differs from O1's frozen truck-63
scene. It is not a cross-engine leaderboard comparison, nor an optimization
improvement. Compression must be scored on the frozen scene with splat-eval
before becoming a default. A final full-scene training capacity test remains
required; a synthetic optimizer allocation test alone cannot prove capacity.

The sampled peak occurred around worker time 23.5 s, while sampled process
memory near the end was 635 MiB. Training logs place the peak near the early
1900–2000-step region after allowing for scene loading (not an exact event
alignment). The final persistent tensor model is only about 163 MB at 304399
splats. This indicates that transient allocations matter substantially on this
small scene; the trace does not identify which operation owns the peak.

## GPU image-cache helper

`memory_cache::GpuImageCache` takes a device and byte budget. It uploads packed
RGBA8 images (`i32` storage) and keys entries by the stable per-scene view index.
Resolution changes replace the prior entry. Least-recently-used views are evicted
before an admitted upload; images larger than the entire budget bypass caching.
A zero budget takes the existing upload-every-visit path. The cache bounds the
logical bytes it owns, not in-flight tensor clones, staging or allocator pages.
Construct a new cache when scene, preprocessing or device changes.

The helper is exported for the trainer lane; wiring and a nonzero configuration
are a separate experiment. It does not enable caching or change the image loader
by itself. GPU tests cover exact-budget admission, LRU eviction, pixel integrity,
resolution replacement, oversized bypass, clear and zero-budget streaming.

## Synthetic optimizer diagnostics

With the earlier Burn-native reference codec at `c6b073f`, three SH3 optimizer
steps at 1M splats completed in both modes. Sampled process
peaks were 2949 MiB (f32) and 1182 MiB (8-bit first moment). **These are not
leaderboard results or a validated memory improvement:** the probe uses fallback
SH variance reduction rather than the renderer-supplied variance, initialization
and shader cache warm-up differ, and it does not render or refine. The full
scene A/B is required. Raw runs are `synthetic-1000000-{32,8}bit` alongside the
training baseline. The final reduction checksum matched in both modes.

## Validation

`cargo test -p splat-train --lib --no-run -j4` passed. Three CPU memory
accounting tests and three ignored-by-default GPU tests passed; GPU tests ran
under a shared GPU lock (`flock`) on RTX 5080 in 1.92 s
for the reference implementation. The fused update passed the same three targeted
GPU tests in 4.13 s (`fused-crate-tests.log`) before the final in-place packed-state
reuse change. The final version compiles; its queued test run was canceled to
follow the coordinator's compile-plus-real-run policy. These are correctness
tests, not training A/B rows.
O1 stored the diagnostic training row in the benchmark host's `runs/perf/results.jsonl`
with `qualified: false` because it uses R3's reconstruction. Matched frozen-scene
A/B and 10M full-training capacity remain pending the baseline GPU schedule.
