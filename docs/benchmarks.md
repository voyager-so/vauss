# Benchmarks

Quality and training-speed numbers for the splat trainer on the standard Gaussian-splatting benchmarks. They
measure the trainer alone, on the datasets' own camera poses. They are not timings of a phone-video run; for
those, see the [examples](../examples/README.md) and [performance.md](performance.md).

## Quality on standard scenes

Protocol: the dataset's COLMAP poses; Mip-NeRF 360 at `images_4` (outdoor) / `images_2` (indoor) using the
provided JPEGs; every 8th image held out; 30,000 steps; final checkpoint; every test view scored. LPIPS is the
VGG network with inputs in [-1, 1]. Three seeds per scene on one NVIDIA L40S. Trainer at commit `37c4a56`
(2026-10-09), scene-adaptive splat budget (about 1.2M splats on indoor scenes, up to 2M outdoors).

| Benchmark | PSNR | SSIM | LPIPS (VGG) | Minutes per scene |
|---|---|---|---|---|
| Mip-NeRF 360, 9 scenes | 28.13 | 0.831 | 0.223 | 2.8–4.1 |
| Tanks and Temples (truck, train) | 24.37 | 0.869 | 0.156 | 2.8–3.6 |
| Deep Blending (playroom, drjohnson) | 30.35 | 0.913 | 0.291 | 2.1 |

A repeat batch with the same commit and seeds gave 28.125 / 24.35 / 30.36 PSNR. Per-scene means moved by at
most ±0.08 dB between the two batches, so differences smaller than that are noise.

For context, published numbers on the same JPEG protocol are 28.21 for 3DGS-MCMC with spherical harmonics and
28.74 for NHT (Mip-NeRF 360). Many papers report 29+ using images resized from the full-resolution originals,
which is worth about +0.27 dB; those numbers are not comparable with the table above.

## Training speed

- The backward rasterisation kernel was made 2.03× faster (25.80 → 12.71 ms median on a 1.06M-splat truck
  checkpoint, L40S).
- That made 7,000-step training 14–24% faster end to end with unchanged quality (truck 34.6 → 26.9 s, garden
  62.4 → 53.6 s; 3 seeds, paired runs on the same L40S).
- On the same L40S, our portable (wgpu/Vulkan) backward kernel takes 1.296 ms against 1.554 ms for the gsplat
  1.5.3 CUDA kernel on one view of a 279,665-splat truck snapshot. Summed kernel time per step is now level with
  gsplat (2.892 vs 2.890 ms). The remaining 1.28× wall-clock gap to gsplat is in the forward pass and host
  overhead.

## Where these numbers have not been measured

- Only NVIDIA GPUs have performance numbers. Apple M4 (Metal) passes a short training, evaluation and
  rendering smoke test; AMD, Intel and DX12 have not been benchmarked.
- Peak GPU memory for the default settings was at most 4.5 GiB on these scenes, sampled on 48 and 96 GB cards.
  It has not been measured on a physical 16 GB card.
- The benchmark datasets are used for internal measurement only. Their licenses (several are non-commercial or
  unstated) do not allow redistribution, so no imagery or model derived from them is in this repository.
