# Performance: where the time goes and what to turn

This page reports measurements, and says so when a number is a projection or an estimate. Each section names the hardware, the clip,
the frame count, the splat release and where the timer comes from. The conclusions hold for these clips and presets; other captures may
behave differently.

All runs use public footage. The "Austin room" clip is *iPhone 14 Pro Max | Dolby HDR 4K | Austin Texas | Luxury Real Estate* by Showing
Austin (<https://www.youtube.com/watch?v=cpl03Le0uCA>, CC BY 4.0). We cut two pieces from it: source seconds 16–101 (85 s; the capture
check reports 86 s) and 16–66 (the "50 s cut"). Audio was removed and the video re-encoded at 1080p.

**Terms.**
- **Camera solve (SfM):** finding each frame's camera pose with COLMAP.
- **Rung:** one attempt of the solve's retry ladder (for example `hybrid`, `exhaustive`, `flip-retry/hybrid`). A rung that fails a health gate hands over to the next one.
- **Held-out PSNR:** how closely renders match real frames left out of training. It is measured in dB; higher is closer.
- **SH:** spherical-harmonic colour coefficients.
- **Loaded vs wall time:** loaded time excludes start-up (model loading); wall time is everything.

## Summary

| If you want... | Turn this | Measured effect (details below) |
|---|---|---|
| A somewhat faster camera solve on Linux with an NVIDIA GPU | Point `SPLAT_COLMAP` at a CUDA build of COLMAP 4.x. The Linux COLMAP that splat 0.1.19-rc fetches is CPU-only | One run on the same machine: 1,207 s -> 1,053 s (-12.8%). Two retry-ladder rungs were 1.24x and 1.30x faster; one was slower |
| To know where a default run's time goes | – | The camera solve took 88–93% of each run's journal span in all three profiled runs; training (7,000 steps) took 47–49 s |
| To move a trained model between machines | Copy it machine to machine, or compress it losslessly with `zstd -T0 -3 splat.ply` | 1.06 GB -> 284 MB. The native trainer and exporter load `.ply.zst` directly (since #16; scope below) |
| A much smaller model file | `vauss export-web <ply> --raw --spz-version 4 --out scene.spz` | 7.9x smaller. On this model, renders match the original's at 71.7 dB mean PSNR and mean held-out PSNR moves < 0.001 dB |
| To use several GPUs | Run independent jobs, one per GPU: `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' splat ...` | 4 independent `vauss refine` jobs on 4 GPUs: 2.94x the throughput of one |

## One full run, stage by stage

The command was `vauss run <clip> --goal "a walkable 3D model of this room" --driver fixed --room` with the default preset (7,000 steps,
up to 1M splats). It ran release 0.1.19-rc on the 85 s Austin cut (1080p, 57 MB), on one RTX PRO 6000 (96 GB) with 16 vCPU, Linux.
Stage times come from the run journal (`journal.jsonl`: tool call to tool result, job start to job end). Uploads and downloads come from the transfer client's timer.

| stage | wall | notes |
|---|---|---|
| upload the video to the GPU machine | 60.8 s (43.8 s for a second machine) | 57 MB over a home uplink shared with other traffic (see [Moving data](#moving-data)) |
| capture check | 4.9 s | |
| frame extraction | 16.4 s, then 8.2 s | 425 frames; then 170 frames re-extracted for the second solve attempt |
| **camera solve, COLMAP on CPU** | **1,220.5 s + 362.5 s** | attempt 1 placed 121/425 frames (28%) and failed the health gates; the retry at 170 frames placed 61 (36%); the run kept attempt 1 (journal `sfm_placement_retry`) |
| training | 48.8 s | 7,000 steps |
| held-out evaluation | 25.6 s | LPIPS ran on the CPU: the bundled ONNX Runtime has no CUDA provider |
| mesh | 17.6 s | |
| web package | 13.5 s (its own timer, `scene.json` `stats.total_s`) | written while the run finishes, so don't add it to the training time; `scene.spz` + `.rad` level of detail + collision, 95 MB |
| download the web package | 35.0 s | 95 MB |

The journal spans 1,704.9 s from run creation to finish. The process ran 1,821 s; we did not time the extra 116 s (start-up and the
exports written after the journal closes). **The camera solve took 1,583 s: 93% of the journal span, 87% of the process wall.**
Training took 2.7% of the process wall. The two runs on the 50 s cut below split the same way: the solve was 90% (CPU COLMAP) and 88%
(CUDA COLMAP) of the journal span.

Each rung's wall time covers its matching, geometric verification and mapping together; these logs don't split them further. The
largest single rung here was exhaustive matching plus mapping: 559.5 s of wall time and 6,360 s of COLMAP CPU time. A spot check
with `nvidia-smi` during the solve showed 0% GPU use. The Linux COLMAP that splat fetches (`colmap-4.2.1-splat.1-linux-x64-cpu`)
reports "without GPU support".

### CPU vs CUDA COLMAP

Both arms ran on the same machine (RTX PRO 6000, 16 vCPU) and the same 50 s cut, with release 0.1.19-rc, one after the other.
The CUDA build was conda-forge `colmap 4.2.1 cuda_129`, passed in with `SPLAT_COLMAP`. Each arm is a single run. Times come from the run journals.

| stage | CPU COLMAP (bundled) | CUDA COLMAP |
|---|---|---|
| frame extraction | 9.5 s + 5.8 s | 9.2 s + 6.6 s |
| solve attempt 1, 250 frames (3 rungs: hybrid, exhaustive, flip-retry/hybrid) | 666.5 s | 569.8 s |
| solve attempt 2, 150 frames (5 rungs: hybrid, exhaustive, flip-retry/hybrid, sfm, sfm) | 425.2 s | 356.5 s |
| training (7,000 steps) | 47.3 s | 48.6 s |
| evaluation / mesh | 27.8 s / 20.9 s | 32.9 s / 25.1 s |
| **total (journal)** | **1,207.3 s** | **1,052.9 s** (-12.8%) |

Rung by rung (from the solver's own `[sfm] rung … wall` lines):

| rung (attempt 1) | CPU | CUDA |
|---|---|---|
| hybrid | 160.2 s | 129.4 s (1.24x) |
| exhaustive | 357.2 s | 274.6 s (1.30x) |
| flip-retry/hybrid | 148.4 s | 165.3 s (0.90x) |

How we read this: CUDA COLMAP helped, but much less than "GPU vs CPU" suggests. In a separate test on 4x L40S with 482 frames, COLMAP's
GPU use averaged 52% during SIFT extraction on one GPU and 1.4% during sequential matching. COLMAP's own CPU work (verification,
mapping) is a plausible limit, but we did not isolate it. Every run made two solve attempts, and in attempt 1 the exhaustive rung was the
largest cost (357 s CPU) without producing the kept model. Making the solve succeed sooner looks like the bigger lever than a faster matcher.

The two arms also kept different solves and scored differently. CPU placed 137/250, then 145/150 on the retry, and kept the retry;
held-out PSNR was 17.6 dB. CUDA placed 181/250, then 109/150, and kept attempt 1; held-out PSNR was 19.9 dB. The evaluation views
differ with the kept solve (19 vs 23). With one run each, we did not establish whether CUDA caused the difference.

### Inside the camera solve

One `reconstruct_sfm` call (a single solve attempt, the step that a full run repeats when the first attempt fails its health gates) on
the 50 s cut's 250 extracted frames (5 frames/s, sharpness-selected, at most 1600 px, SIFT 8,192 features). It ran on the workstation
CPU with COLMAP 4.2.1 (CPU build) and 8 threads, using splat built from PR #7's head (3d16f047) in its own target directory. Arms:
`SPLAT_SFM_MAPPER=incremental` (today's default) vs `global` (PR #7's default for video). Times come from the solver's `[sfm] rung`
lines and `reconstruction.json` stages.

| rung | what it does | incremental | global (PR #7) |
|---|---|---|---|
| hybrid | SIFT extraction + sequential matching with loop detection + geometric verification + mapping | 143.9 s; model failed the health gates | 298.6 s; no model kept |
| exhaustive | matches the remaining image pairs, then maps again | 255.6 s; no model kept | 316.3 s; no model kept |
| flip-retry/hybrid | maps the same matches again with a SIMPLE_RADIAL camera and an 8° initial pair (incremental mapper in both arms) | 206.6 s; **kept: 137/250 frames** | 145.7 s; **kept: 137/250 frames** |
| **attempt total** | | **607 s** | **761 s** |

Both arms kept the same 137-frame model, and it still failed two gates: placed fraction ≥ 0.95 (0.548) and no unplaced run
of more than 5 frames. In a full `vauss run`, that failure triggers the second attempt on fewer frames. On this clip and frame set,
PR #7's global mapper didn't place more frames and made the attempt 25% slower. The measure lane's own global solve of the same
50 s placed 488 of 505 frames (97%), but from 505 frames at 10 frames/s with different matching options. We did not test which
difference matters.

The same COLMAP steps run alone on the same 250 frames, with the solver's options:

| step | wall (8 threads, CPU) |
|---|---|
| SIFT extraction (8,192 features) | 32.9 s |
| sequential matching, overlap 40, quadratic, vocabulary-tree loop detection, with verification | 63.7 s |
| exhaustive matching of all 31,125 pairs from scratch, with verification | 554.1 s |

Inside the ladder, the exhaustive rung reuses the hybrid rung's matches and only matches the remaining pairs, which is why it is cheaper
there (256 s including mapping) than from scratch. The mapping in the kept rung breaks down as follows (incremental arm):

| mapping step | wall |
|---|---|
| keyframe mapper | 6.7 s |
| mapper on every 2nd frame | 39.6 s |
| continuation mapper (all frames) | 85.0 s |
| leftover-frame mapper | 5.3 s |
| point triangulation | 8.8 s |
| lens bundle adjustment | 60.3 s (465 CPU-s) |

**Skipping the exhaustive rung on video: tested, rejected.** The exhaustive rung's own model was never the kept one in the five
first-attempt ladders above, and it cost 256–560 s each time. So we tried keeping the hybrid attempt for video instead (when it
placed at least half the frames). Same build, on an 8-thread workstation CPU; `before` = the existing ladder, `after` = no exhaustive
fallback; one solve attempt each:

| clip (public) | before: wall, kept rung, frames placed | after: wall, kept rung, frames placed |
|---|---|---|
| Austin room, 50 s cut (250 frames) | 435.7 s, flip-retry/hybrid, **188**/250 | 156.1 s, flip-retry/hybrid, **141**/250 |
| moskvitch-car, 41 s (206 frames) | 355.4 s, hybrid, 206/206 (no exhaustive needed) | 223.5 s, hybrid, 206/206 |
| la-apartment, 42 s (209 frames) | 47.3 s, exhaustive, 109/209 | 25.3 s, hybrid, 105/209 |

On the Austin cut, the skip placed 47 fewer frames. The exhaustive rung's extra matches feed the flip-retry rung that follows it, so
removing the rung costs placement even when its own model isn't kept. We didn't ship it, and didn't spend GPU time on the held-out
check. The moskvitch-car arms ran the identical ladder yet took 355 s and 224 s, so solve wall times on a shared workstation vary by
about ±40% between runs. Compare solve times only within one quiet machine, and repeat them.

Other levers:
- **CUDA COLMAP:** -12.8% on one full run (above).
- **The global mapper (PR #7):** not a speed win on the frame set above (607 -> 761 s, same kept model).
- **Thread counts and frame density:** not measured here.

### The fill-in loop

The fill-in loop renders novel views near the camera path, repairs them with a fixer model and trains them in at low weight. These
timings come from one round of the research implementation (gsplat + Difix-ref, RTX PRO 6000). They are taken from that run's log, not
re-run here.

| step | wall |
|---|---|
| generate: render + repair, 1,846 views | 2,242 s (1.21 s/view) |
| train in, 6,000 steps | 279 s |

Generation was 89% of the round, and each view is independent. NVIDIA Fixer (Cosmos-Predict2 0.6B, bf16, one step) took 65.7 ms of
model time per 1024x576 view on an L40S in the multi-GPU test below. Two other L40S containers measured 85 ms and 180 ms (the 180 ms
was a short smoke run), so per-view speed varies between machines.

### `vauss fill` (the Rust command, #21), one round

Release build of `fill-loop` at 0c1780f5, on the public Austin room (the measure lane's 4.5M-splat model). It ran 1,200 novel views
within 1 m of the path (`--limit 1200`, from 11,625 planned) with stock NVIDIA Fixer, on 4x RTX PRO 6000 with 32 vCPU. Stage times
come from the command's own summary.

| stage | 1 fixer GPU (`--fixer-gpus 0`) | 4 fixer GPUs (`--fixer-gpus 0,1,2,3`) |
|---|---|---|
| plan cameras | 131.7 s | 125.2 s |
| render 1,200 views | 30.5 s | 29.0 s |
| repair with Fixer | 322.1 s (about 258 ms/view including a 12 s model load) | **89.3 s** (300 views per GPU at 219–220 ms/view) |
| confidence | 123.1 s | **33.1 s** |
| train in | 189.1 s (96 s of training steps) | 169.1 s (85 s of training steps) |
| **total** | **796 s**; held-out PSNR 27.71 dB | **446 s (1.79x)**; held-out PSNR 27.75 dB |

The 4-GPU run used a fixed build (0f34218d; the first 4-GPU attempt at 0c1780f5 crashed on a temp file shared by the fixer
processes). It ran on a second machine of the same type with the same settings. It re-planned the same 11,625 cameras; `--limit 1200` takes a random subset, so the two runs may have repaired different views. Its held-out PSNR matches
the 1-GPU run within noise (+0.04 dB). Repair and confidence, the two stages that shard across GPUs, ran 3.6x and 3.7x faster. Part of
that is the machine: Fixer ran at 220 ms/view per GPU there against about 258 ms/view on the first machine. At equal per-view speed,
repair would be about 3.1x faster. Camera planning, rendering and training stay on one GPU, so they now dominate the round
(323 of 446 s).

## Moving data

Big files are trained PLYs, datasets and renders. This writer's SH-3 PLY is 236 bytes per splat: 4,499,579 splats is 1.06 GB.

**Why one upload took 19 minutes.** On Oct 23, a 957 MB PLY took 1,139 s (0.84 MB/s) to push from a home workstation to a cloud GPU
machine. The transfer logs show three more big pushes from the same workstation in the same 19 minutes: the same 957 MB PLY to a
second machine, a 262 MB dataset and another dataset, at least 2.2 GB in all. Later the same evening (22:30–22:56), we summed the
workstation's internet sockets per process. Its upload was full at about 4.7 MB/s in total, shared among several programs. We did not
sample the link during the slow push itself, so contention is the explanation the logs support, not a decomposition measured at that moment.

A controlled test on one machine (200 MB random files; the link was busy with other traffic throughout):

| client | pushes at once | per push | combined |
|---|---|---|---|
| one signed URL per file | 1 | 0.47 MB/s (425 s) | 0.47 MB/s |
| one signed URL per file | 3 | 0.62 MB/s (320–326 s) | 1.86 MB/s |
| 8 parallel signed parts | 1 | 0.84 MB/s (238 s) | 0.84 MB/s |
| 8 parallel signed parts | 3 | 0.68 MB/s (291–293 s) | 2.05 MB/s |

On this busy link, three simultaneous uploads took a larger combined share than one, for both clients (1.86 and 2.05 MB/s vs 0.47 and
0.84 MB/s). Per upload the clients differed. With one signed URL per file, each of three ran faster than one alone did (0.61–0.62 vs 0.47 MB/s).
With 8 parallel parts, each of three ran slower (0.68 vs 0.84 MB/s). The single-upload runs and the three-upload runs happened at
different minutes of a link shared with other traffic, and the test doesn't separate the client's part count from its other differences.

**What to do instead:**
1. **Run the next stage where the data already is.** `vauss run --cloud` keeps full PLYs and COLMAP databases on the cloud volume and
   brings back the web package and metrics; `splat pull <run>` fetches the rest.
2. **Copy machine to machine, not through your workstation.** Between two cloud GPU machines, over an authenticated HTTPS link
   served by one and fetched by the other, 1 GiB took 56.1 s and 55.8 s on a single stream (19 MB/s) and 46.3 s in 8 parallel parts
   (23 MB/s), sha256-checked. Between two machines that sat closer together, a 926 MB run directory took 1.6 s. Either way your uplink isn't used.
3. **Compress what you must send.** The Austin room model (4,499,579 splats, SH 3, trained by splat with its quantisation-aware tail):

| encoding | bytes | ratio | encode (8 threads) | render PSNR vs original (mean of 152 held-out views) | mean held-out PSNR change |
|---|---|---|---|---|---|
| PLY, fp32 | 1,061,902,255 | 1.0x | – | – | – |
| `zstd -T0 -3` | 284,294,942 | 3.7x | 0.4 s with the file in page cache; decode 0.9 s | lossless | 0 |
| `zstd -T0 -19` | 215,456,508 | 4.9x | 77 s | lossless | 0 |
| fp16 attributes, fp32 positions | ~558 MB, an estimate (124 B/splat); not written as a file | ~1.9x | – | 79.9 dB | < 0.001 dB |
| SPZ v3 (`vauss export-web --raw`, default) | 137,023,023 | 7.7x | – | not measured | – |
| SPZ v4 (`--spz-version 4`) | 133,724,937 | 7.9x | 12.3 s | 71.7 dB | < 0.001 dB |

The fp16 row converts attributes to fp16 and back in memory and renders that result; its size is calculated, not measured.

**Where `.ply.zst` works (since #16):**
- Native trainer loaders: init, warm start, warm cap, inpaint host.
- `vauss export-web` and the other exports that resolve their input through the exporter: an explicit `*.ply.zst`, or a directory holding `splat.ply.zst`.
- The exporter's PLY header reader.

Detection uses the zstd magic bytes, but input resolution also needs a `.ply` or `.ply.zst` name; a run's candidate may hold either
`splat.ply` or `splat.ply.zst`. Not covered, so decompress first for these:
- dataset formats (COLMAP/nerfstudio init PLYs);
- the training process's streamed PLY input;
- web-package and asset ingestion, and the web viewer;
- the Blender pack;
- mesh PLYs.

This model's SH coefficients already sit on the 8-bit grid that SPZ stores, so SPZ's SH quantisation costs nothing for it:
`--sh-rest-bits 6` and `8` gave byte-identical files. A model trained without that tail would lose more; we didn't measure one.

## Multi-GPU

The question: what does a machine with several GPUs speed up? Each comparison below ran 1 GPU and 4 GPUs inside the same container or
machine. Public data only.

| stage | setup | 1 GPU | 4 GPUs | result |
|---|---|---|---|---|
| independent `vauss refine` jobs (our trainer: 7k -> 15k steps, then evaluation, mesh and export) | 50 s Austin cut, 4x RTX PRO 6000, 32 vCPU; each job pinned with `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(i)'` | 174 s for one | 237 s for four | **2.94x throughput, 73% efficiency.** Each GPU reached ~100% during the four-job phase |
| independent trainings (gsplat 1.5.3 `simple_trainer` as a stand-in, 3,000 steps) | 482 frames (10 fps, 1600x900) of the same 50 s, 4x L40S, 16 vCPU | 52.9 s for one | 58.3 s for four | **3.63x throughput, 91% efficiency** |
| fill-in generation: gsplat render + stock NVIDIA Fixer + JPEG write | 1,216 views (152 held-out cameras x 8 jitters), 1024x576, 4x L40S, one worker process per GPU | 148.4 s wall (9.71 views/s loaded) | 58.5 s wall (36.72 views/s loaded) | **3.78x loaded (95%), 2.54x wall (63%).** Each worker spends 22–25 s loading the PLY and Fixer, which caps the wall-time gain on a job this short. Peak container CPU was 24% |
| one scene's training on 4 GPUs | gsplat's distributed mode (Gaussians sharded across ranks), same container as its 1-GPU baseline | train timer 28.4 s; wall 56.8 s | slowest rank 22.3 s; wall 101.9 s (process-group start-up, per-rank evaluation) | 1.28x on the training timer, slower end to end. Our trainer (Brush on wgpu) uses one device and has no multi-GPU mode |
| COLMAP SIFT extraction | the same 482 frames, COLMAP 4.2.1 CUDA, 4x L40S | 8.50 s | 7.25 s | 1.17x; GPU use 52% on one GPU, 12–13% each on four |
| COLMAP sequential matching | same | 133.4 s | 132.6 s | 1.01x; GPU use 1.4% on one GPU, 0.3% each on four |

What this means for these workloads:
- **Independent runs** (several clips, seeds or presets) use several GPUs well. Start one `splat` per GPU, each with its own
  `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(i)'` (0-based). `CUDA_VISIBLE_DEVICES` doesn't select the device here, because splat renders
  through Vulkan, Metal or DX12. The camera solve and evaluation also use CPU. We didn't measure how many vCPUs each job needs; the 4-job
  run above had 8 per job.
- **Fill-in generation** is the part of the fill loop that spreads across GPUs: `vauss fill --fixer-gpus 0,1,2,3` (#21) runs one fixer
  worker per GPU: measured 1.79x on a whole round with 4 GPUs (see the `vauss fill` section above). The projection below was made before that measurement. *Projection, not a measurement.* Take the Difix-ref round above (RTX PRO 6000): 2,242 s of generation plus 279 s of
  training, 2,521 s in all. Assume generation scales like the Fixer test on L40S: 95% efficiency on 4 GPUs, plus about 25 s of worker
  start-up, with training on one GPU. Then 4 GPUs take 2,242 / (4 x 0.95) + 25 + 279 ≈ 894 s, 2.82x faster. Assuming 91% efficiency on
  8 GPUs (not measured) gives 2,242 / (8 x 0.91) + 25 + 279 ≈ 612 s, 1.46x faster than four. Neither round was run; the 1-vs-4
  `vauss fill --fixer-gpus` measurement is pending.
- **One scene's training and COLMAP matching** gained little from extra GPUs in these tests.

Prices: these runs used Modal L40S at about $1.95/h per GPU and Voyager RTX PRO 6000 machines. Those were the rates at the time, not
current prices.

## Knobs

| knob | where | effect |
|---|---|---|
| `SPLAT_COLMAP=/path/to/colmap` | env or config `colmap` | Use your own COLMAP 4.x. A CUDA build gave -12.8% on one whole run (A/B above) |
| `--fast` | `vauss run` | Quick preview: 7,000 steps, up to 250,000 splats, 800 px frames, no mesh |
| `--steps N`, `--cap-max N` | `vauss run` | Training length and splat cap. Training was a small share of the default runs above; the cap sets model size (236 B/splat as an SH-3 PLY) |
| `--max-edge`, `--resolution-divisor` | `vauss run` | Smaller frames, less detail. We didn't measure their effect on time |
| `--cloud` / `--fetch-all` | `vauss run` | Run GPU stages remotely; big outputs stay remote unless you ask for them |
| `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)'` | env | Which GPU a `splat` process uses (0-based). Quote it in the shell |
| `SPLAT_FAL_OWNER` | env | Point `--cloud` at a fal account that hosts your own deployment of the runner apps (since #16). Deploying them is not covered here |
| clip length and cuts | how you film | The 85 s multi-shot cut spent 1,583 s solving; its kept solve placed 121 of 425 frames (28%). The 50 s cut, on a machine of the same type, spent 1,092 s (CPU COLMAP; kept solve 145/150) and 926 s (CUDA COLMAP; kept solve 181/250) |
