# Startup tuning bundles

These are CubeCL's native flat bundles, exported with `tools/export_autotune.rs`
from measured caches. They contain only `autotune` and `throughput` namespaces:
tuning winners and the calibration values used to select them. No shaders,
scene data, or custom serialization are included.

Startup selects an exact adapter name, PCI vendor/device and graphics backend,
and checks that the runtime uses WGSL. Unknown adapters, custom logical device
IDs, and other compilers keep ordinary tuning. `CUBECL_AUTOTUNE_CACHE=0` also
keeps its existing meaning: no persistent tuning results, including these seeds.

Import is insert-only; existing locally measured values win. CubeCL verifies
candidate-set checksums before using an imported autotune result. Unseen shapes
and stale candidate sets tune normally. Throughput entries additionally key by
memory capacity, hardware fingerprint and probe version. Bundle metadata is
informational; adapter matching in `src/autotune.rs` is the selection guard.

## Sources

All captures use CubeCL `467522442c7b68f02c7ff0c84b2e249e36d8da91`
(`0.11.0-pre.4`), Burn `faec398324e203fdf7998318e4889b987fc802fc`, and WGSL.

- NVIDIA L40S, RTX PRO 6000 Blackwell Server Edition and H100 80GB HBM3:
  O2 exports, Oct 9 2026, from main `426f30f` (after tonight's depth24 sort,
  SH fold, Morton re-sort, Mip render default and the O10 SH step `d2f0156`).
  Each GPU's cache came from one fal runner training the full 30k default
  schedule on garden, truck-full, kitchen and onthego-patio-high-clean in turn,
  starting from an empty cache for that binary (`/data/splat-fal/autotune/
  <gpu>-c8fa6dccb9318373.tar`). Vendor `0x10de`, Vulkan; devices `0x26b9`
  (L40S), `0x2bb5` (RTX PRO 6000), `0x2330` (H100); drivers 590–610.
  - L40S: 45 autotune results + 15 throughput probes (22 measured that run,
    38 carried over from the Oct 8 seed, which CubeCL re-validates by checksum).
  - RTX PRO 6000: 33 autotune results + 11 throughput probes.
  - H100: 33 autotune results + 11 throughput probes.
  The throughput probes are what CubeCL's roofline bounds measure on a cold
  device (the ~2 GiB transient reservation), so seeded GPUs skip that probe too.
- STALE: the RTX 5080 and Apple M4 bundles below predate tonight's kernel
  changes (listed above); unseen or re-checksummed keys tune at runtime there.
  Regenerating the 5080 one needs a separate decision (no local training runs).
- RTX 5080 (stale): consistent SQLite backup of a workstation's `~/.cache/cubecl/default.db`,
  Oct 8 2026. NVIDIA GeForce RTX 5080, vendor 4318, device 11266, Vulkan.
  39 autotune results and 15 throughput probes.
- Apple M4 (stale): R7's consistent SQLite backup of macmini `~/.cache/cubecl/default.db`,
  Oct 8 2026. Apple M4 10-core GPU / 16 GB, macOS 26, Metal, vendor/device 0/0.
  29 autotune results and 15 throughput probes. Capture database SHA256:
  `71a72bfd4197dfc03b52a9c404641089c83f7d316318d8e3d194d3da210251f4`.
  Populated by truck16 native smokes and a pegasus 7k run.

The export tool takes `CACHE_DB OUTPUT DEVICE_LABEL OS ARCH` and uses CubeCL's
`bundle::export` with `BundleFormat::Flat`. Regenerate after changing the pinned
CubeCL/Burn versions; never relabel another GPU's measurements as a match.

## Measurement

Compare identical hardware/settings and fresh empty CubeCL environments,
without the bundle (baseline binary) versus startup import (candidate binary).
Use both run orders. Shader/driver cache state must match between arms.
Do not use `CUBECL_AUTOTUNE_CACHE=0` to measure bundle import: that setting
intentionally bypasses the imported results too.

The prior ~10.5s warm-cache gain also included NVIDIA GLCache shader artifacts.
It is not an autotune-bundle speedup. The cache-off diagnostic with equally warm
driver caches was neutral in two run orders (5.00/5.20s and 5.41/5.61s).

Oct 8 2026 L40S measurements compare baseline `35bd7ff` with bundle binary
`261eb55` (SHA256 `d00945e86b2acb47e491201d21628b3f0c379f899516444910b97e8ce18ae79a`).
Each pair used one runner, Truck-63-clean, seed 42, and a fresh CubeCL database
per timed arm. The controlled runs warmed each binary's driver cache over the
full schedule in separate throwaway CubeCL environments first.

| Schedule / driver cache | Baseline wall | Bundle wall |
| --- | ---: | ---: |
| 2201 steps / warm | 18.21 s | 9.20 s |
| 7000 steps / warm | 49.02 s | 36.22 s |
| 7000 steps / fresh | 53.23 s | 40.82 s |

The cold 7k pair used separate empty `XDG_CACHE_HOME` roots, including NVIDIA
GLCache. Peak measured VRAM was 2438 versus 742 MiB; this is a process peak,
not a steady training-memory comparison. The candidate tuned 7 keys versus
28 in the baseline, avoiding all six integer-sum tuning buckets encountered.
These measurements cover the combined autotune and throughput seeds; they do
not isolate either namespace's contribution. The cold pair's PSNR was
20.816 versus 20.926, and LPIPS 0.312 versus 0.314; broader quality claims are
not established by one pair.

Raw evidence is under `runs/fal/jobs/truck-63-clean-*-o3-bw*` and
`runs/fal/jobs/truck-63-clean-7000-o3-bcold7k-*`; logs are
`runs/fal/logs/o3-bundle-v2.log` and `o3-bundle-cold7k.log`.
CLI startup imports occur before its logger is initialized, so the absence of
the import info line is expected; cache-hit and tuning-key logs confirm use.

CubeCL already persists calibration by device identity for later launches on
all adapters. These seeds target first-use latency. Its memory-copy throughput
probe can allocate two 1 GiB pools, which is consistent with the cold peak
reduction, but that attribution has not been measured independently.
