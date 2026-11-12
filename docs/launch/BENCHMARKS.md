# Launch measurements

These results describe measured configurations from October 2026. Training, quality and creative-demo timings are separate experiments. They are not all measurements of one default `vauss run` command.

## Equal-cap training comparison

Measured October 9, 2026, on seven Mip-NeRF 360 scenes: **bicycle, bonsai, counter, garden, kitchen, room and stump**. Each scene ran three times per trainer. Each paired comparison ran on the same L40S machine, at 30,000 steps and a one-million-splat cap. Final counts were approximately 0.978–1.000 million.

| Trainer | Mean training seconds |
|---|---:|
| Vauss | 126.86 |
| VkSplat | 194.96 |
| gsplat-MCMC 1.5.3 | 679.37 |

The **mean paired training-time ratio** between Vauss and gsplat-MCMC is **0.185**. Its reciprocal is **5.405**, rounded to 5.4×. This differs slightly from dividing the two arithmetic means. GPU memory was **0.43–0.56×** gsplat-MCMC's across the comparison. Vauss had lower full-resolution VGG LPIPS on all seven scenes, higher SSIM on all seven, and higher PSNR on six; stump showed no detectable PSNR difference.

VkSplat used revision `b3ad2b0`. Its three runs are repeated runs, since it exposes no seed setting. These results establish performance within this comparison, not a universal ranking of all splat trainers or an equal-quality speedup at arbitrary budgets.

## Nine-scene quality

Mip-NeRF 360 covers **bicycle, flowers, garden, stump, treehill, room, counter, kitchen and bonsai**. Inputs use the dataset's original JPEGs: `images_4` outdoors and `images_2` indoors, without re-encoding. Every eighth image in sorted order is withheld; scores use the held-out views. LPIPS uses the VGG backbone at the recorded image resolution.

| Recipe | GPU | Runs per scene | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| Tuned native | RTX PRO 6000 | 1 | 28.521 | 0.8375 | 0.2079 |
| Three-run reference | L40S | 3 | 28.127 | 0.8311 | 0.2226 |

The tuned native recipe uses up to 45,000 steps and SSIM weight 0.1. Bonsai, counter and kitchen use neural colours at a 2.5M cap; the other six use ordinary splat colours and adaptive stopping. Training took **195–342 seconds per scene**. Total training-command times were **291–485 seconds**. This is a tuned, single-seed result, not a promise that the default command produces the same scores. Neural-colour models are not the ordinary SPZ files used by the Blender demo.

The three-run reference uses 30,000 steps. The same reference protocol measured **24.365 / 0.8693 / 0.1562** PSNR/SSIM/LPIPS on Tanks & Temples (truck and train), and **30.346 / 0.9126 / 0.2915** on Deep Blending (playroom and drjohnson).

## Picture 2 World

**Library:** one AI image became a 30-second 1080p orbit video, then a reconstruction containing 1,963,332 splats. On one RTX PRO 6000, camera solving took **143 seconds**, training **303 seconds** (45,000 steps), and the full video-to-exports stage **712 seconds**. The approximately **35-minute** finished production also includes AI image/video generation and final Blender rendering. The demo uses cleanup, a custom diffuse/emission material and artistic lighting.

**Cabin:** a Seedream 5 Pro image and Kling 4 camera moves produced a 60-second 1080p input. The scene contains **1,948,256 splats**, trained for **322 seconds** at 60,000 steps on one RTX PRO 6000. The full video-to-exports stage took **2,215 seconds**, including two camera-solve attempts, evaluation, mesh and exports. This is a separate workflow from the library's 35-minute production.

## Data and attribution

- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/): Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan and Peter Hedman, CVPR 2022.
- [Tanks and Temples](https://www.tanksandtemples.org/): Arno Knapitsch, Jaesik Park, Qian-Yi Zhou and Vladlen Koltun, SIGGRAPH 2017.
- [Deep Blending](https://repo-sam.inria.fr/fungraph/deep-blending/): Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis and Gabriel Brostow, SIGGRAPH Asia 2018.

Dataset terms are set by their authors. This directory publishes measurements and small rendered illustrations; it does not bundle the input datasets. [Machine-readable numeric summary](measurements.json) · [Media credits](CREDITS.md).
