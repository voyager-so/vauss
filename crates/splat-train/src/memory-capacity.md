# RTX 5080 capacity protocol

Decision from performance lead, 2026-10-08: use the registered Mip-NeRF 360
images_4 inputs at divisor 1: garden 1297×840 and bicycle 1237×822. Original
~5K images are not part of this comparison. Preserve O1's frozen cameras,
image bytes, seed and every8_idx0_sorted split for every engine/memory mode.

Capacity means splats actually present during training, not a configured cap
or an optimizer-only allocation. Require SH3 forward, backward, optimizer and
refinement on the local RTX 5080 16 GB. A cloud GPU with a 16 GB allocation
limit does not establish this device's capacity.

1. O1 schedules a quiet local window. The complete GPU child lifetime,
   including evaluation/export and VRAM sampler, holds gpu.lock. Release the
   lock between candidate runs. Use splatcap for CPU work; no test/lint suites.
2. Start from the same source reconstruction/configuration for both optimizer
   variants. Vary only moment/parameter storage in each paired comparison.
   GPU image cache budget stays zero. Record exact binary hash and environment.
3. Probe caps geometrically (2M, 4M, 8M, 10M, 12M), stopping at the first actual
   OOM. A run which never reaches its cap supplies only an observed lower bound;
   it is not a success at the requested count. Separate crash, timeout, driver
   error and OOM. Do not infer OOM from an arbitrary nonzero exit status.
4. Once a candidate count is reached, require at least one complete training
   image epoch crossing a refinement event. Record the minimum actual count
   throughout that interval, not just the maximum or final checkpoint count.
   This is the sustained capacity for the run. Apply this definition to Spirula
   as well. If growth/pruning prevents the condition, report unproven capacity.
5. Sample board and training-process VRAM during the entire run, preserve the
   time series and distinguish sampled peaks from exact allocator peaks. Log
   actual count and phase/step around the largest observed allocation. Refine
   the successful/OOM interval to 0.5M only if O1's schedule permits.
6. Send held-out images through splat-eval, retaining PSNR/SSIM/LPIPS and wall
   time in O1's leaderboard row. Capacity probes with different step counts or
   growth schedules are a separate section from the default 7k/30k comparison.
   Promotion still requires the matched default-quality/time comparison.

Keep one checkpoint per scene/candidate only while needed. Delete intermediate
PLY files after extracting count/evidence; retain logs, sampled memory, hashes
and scored rows. O4 total worktree outputs/builds must stay under 10 GB.

## Pending evidence

No full-scene capacity result has been measured yet. The analytical model in
memory-design.md excludes raster/sort scratch, gradients and refinement overlap.
It must not be substituted for a capacity row. O1 owns scheduling and the final
engine-specific probe commands after the baseline queue.
