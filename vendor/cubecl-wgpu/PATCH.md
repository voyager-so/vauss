# CubeCL wgpu bind group cache patch

Upstream: `tracel-ai/cubecl` commit
`467522442c7b68f02c7ff0c84b2e249e36d8da91`, crate `cubecl-wgpu`
version `0.11.0-pre.4`. Original MIT/Apache licenses and NOTICE are included.

## Why

On the CubeCL GPU-server thread, wgpu `Device::create_bind_group` measured
about 11% of CPU per training step: a step issues ~100 dispatches and
`WgpuStream::register_pipeline` built a fresh bind group (plus a
`get_bind_group_layout` call) for every one, although the memory pools keep
handing out the same buffer ranges step after step.

## Rust source change

- New `src/compute/bind_group_cache.rs`: `BindGroupCache`, a shared map from
  (compute pipeline, and per binding the `wgpu::Buffer`, offset and size) to
  the `wgpu::BindGroup` built from exactly those ranges. The key holds clones
  of the pipeline and buffers, so their wgpu identities cannot be reused while
  cached; a hit therefore binds exactly what a fresh group would. Lookups use
  a borrowed key (`hashbrown::Equivalent`), so a hit allocates nothing.
- `src/compute/stream.rs` `register_pipeline`: get the bind group from the
  cache instead of creating it every launch. `record_pipeline` (graph capture)
  and replay are unchanged.
- `src/compute/storage.rs`: `WgpuStorage` optionally holds the cache and, on
  every `dealloc`, evicts the entries that bind the freed buffer before the
  storage drops it, so a cached group never keeps a freed buffer (and its
  device memory) alive. This covers every free path, including the pools'
  reclaim-before-allocate, not just periodic cleanup. Evicting per buffer
  rather than clearing everything matters: the pools free pages often enough
  in training that full clears would throw away most of the cache.
- `src/compute/mem_manager.rs`: create one cache per stream and attach it to
  the main and uniforms pools' storages (staging buffers are never bound).
- The cache is also cleared whenever it would exceed 4096 entries, which
  bounds how long a dropped pipeline can be kept alive.

Always on; no feature flag or environment variable.

## Manifest

`Cargo.toml` expands upstream workspace inheritance without changing
dependency versions or features. Sibling CubeCL crates stay on the pinned
upstream Git revision, so the rest of CubeCL shares the same crate identities.
The root Git-source patch replaces only `cubecl-wgpu`; `Cargo.lock` changes
only by dropping `cubecl-wgpu`'s Git source line.

When updating CubeCL, re-evaluate this patch against upstream and remove the
vendor override if upstream caches bind groups itself.

# Persisted pipeline cache patch

## Why

Every process paid the driver's SPIR-V to machine-code compile for each
compute pipeline: an nsys trace of a cold 7k-step run on an L40S showed 296
`vkCreateComputePipelines` calls costing 3.88 s (~17% of the run). Hosted jobs
run in fresh containers and pay it every time.

## Rust source change

- New `src/compute/pipeline_cache.rs`: `PersistentPipelineCache`. When the
  device has `wgpu::Features::PIPELINE_CACHE` (Vulkan; the Vulkan and WGSL
  device requests already enable every adapter feature), it loads
  `<dir>/<wgpu::util::pipeline_cache_key>.bin` into a `wgpu::PipelineCache`
  (`fallback: true`, so stale/corrupt data means an empty cache). `<dir>` is
  `$SPLAT_PIPELINE_CACHE_DIR`, else `$XDG_CACHE_HOME` / `~/.cache`
  (Linux), `~/Library/Caches` (macOS) or `%LOCALAPPDATA%` (Windows) plus
  `splat/wgpu-pipeline-cache`.
- A saver thread writes the cache back (temp file + rename) after pipeline
  creation has been quiet for 250 ms, at most every 2 s during a burst, and
  on drop; it skips the write when the data is unchanged. The short debounce
  catches the eval/export burst that comes right before exit (the process
  does not drop the server). I/O errors are logged once at debug level.
- `src/compute/server.rs`: `WgpuServer` owns the cache (`None` without the
  feature). `src/backend/base.rs` `create_pipeline`: passes it as
  `ComputePipelineDescriptor::cache` and notifies the saver.
- Not compiled on wasm; other backends lack the feature and keep `cache: None`.

Always on; `SPLAT_PIPELINE_CACHE_DIR` only relocates the file (e.g. to
pre-seed it in a container image).

# Bounded host-visible transfers patch

## Why

`splat run --max` let a room scene grow to 5.87M splats, and held-out evaluation
then crashed: the `__brush align-eval` worker panicked with `wgpu error: Out of
Memory` while about 2 GB of the RTX PRO 6000's 96 GB was in use. Loading the
PLY uploads its SH coefficients (`[n, 16, 3]` f32, 1.13 GB) as one queue
write, which wgpu stages in one host-visible buffer of the same size. On that
GPU (driver 610.57.04) the driver refuses any single host-visible allocation
over roughly 1 GiB: a size sweep passed at 900 MB and failed at 1100 MB, for
uploads and for readbacks alike, while BAR1 (128 GiB) and the host heap (36 GiB)
had plenty of room. Readbacks hit the same limit: one staging buffer per tensor.

## Rust source change

- `src/compute/stream.rs` `write_to_buffer`: a write staging more than
  `MAX_WRITE_CHUNK` (64 MiB) is split into 64 MiB queue writes, each flushed
  (`flush_pending_writes`, the existing empty submit + wait, now its own
  function), so at most one bounded staging buffer is alive at a time.
- `read_resources`: a read larger than `MAX_WRITE_CHUNK` is copied into
  several 64 MiB staging buffers and assembled into one host `Vec`; a read
  that fits one chunk keeps the zero-copy mapped path.
- `src/compute/controller.rs`: `WgpuAllocController::mapped` exposes the
  mapped bytes for that assembly.

Always on. Covered by the GPU test `engine::tests::large_uploads_are_exact`
in `splat-brush` (chunk boundaries plus an 8M-splat SH tensor round trip).

The split covers the destination extent a write stages (the resource, padded
to 4 bytes), not the payload length: a payload shorter than its resource would
otherwise still stage the whole resource. `transfer_chunks` plans both
directions; its unit tests (`transfer_chunk_tests`, run with
`cargo test --lib transfer_chunk` in this crate) cover unaligned tails and
short payloads into a 1.5 GiB resource.
