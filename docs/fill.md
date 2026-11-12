# `vauss fill`: generative fill-in for free roaming

A splat trained from one walk through a room is sharp along the walked path. A metre to the side it shows haze,
holes and floaters, because no photo saw the room from there. `vauss fill` fills in those views. It renders
novel views next to the path, repairs them with a diffusion "fixer" model, and trains the repaired views back
into the splat at low weight, beside the real frames.

```sh
# A finished run: uses its result splat, its SfM and its frozen held-out split. Writes <run>/fill/.
vauss fill runs/living-room --fixer-python ~/.local/share/splat-fixer/venv/bin/python

# Any COLMAP dataset (images/ + sparse/0) plus a splat trained on it:
vauss fill data/room --ply room.ply --out room-fill \
  --train-list train.txt --heldout-list heldout.txt --fixer-python ~/.local/share/splat-fixer/venv/bin/python
```

The new splat is `<out>/model/splat.ply`. `<out>/fill.json` records the inputs, the view counts, the time per
stage and the held-out PSNR/SSIM of the result (the `train` block).

## What it does

| stage | where | what |
|---|---|---|
| cams | Rust (GPU) | Builds a floor plan in the metric frame (floor at y = 0, metres; fitted like `export-web`, or `--frame`). Opaque splats 0.2–0.9 m above the floor are obstacles. The free floor is flood-filled from the walked path, keeping a `--clearance` of 0.25 m and staying within `--hull-margin` of the path's convex hull. Then it places cameras on a 0.4 m grid × heights 1.1/1.5/1.8 m × 8 yaws × a random pitch, and drops views facing a wall at arm's length or looking out of the model. → `cams.json`, `floor_map.png` |
| render | Rust (GPU) | Renders every view within `--max-dist` (1 m) of the path at 1024×576, with alpha and depth. → `fix/in`, `fix/aux` |
| fix | Python (GPU), external | Runs the fixer on `fix/in` and writes `fix/out`. Default: NVIDIA Fixer. |
| confidence | Rust + Python | Fits the fixer's colours to the render with a 3×3 + bias least-squares map. Per-pixel trust = 1 − LPIPS(fixed, render)/0.8, clamped to [0.3, 1], so a big rewrite of a well-seen area is trusted less. Holes the fixer filled keep full trust. Pixels nearer than 0.3 m get none. The trusted area is then eroded by 2 px. → `pseudo/*.png` (RGBA, alpha = trust), `pseudo/views.json` |
| train | Rust (GPU) | Warm start from the splat, 6,000 steps. 30% of steps train on a novel view, at weight 0.3 × (1 within 1 m of the path, fading to 0.5 at 3 m). That weight ramps in on a sine over the first 30% of steps, and the last 30% train on real frames only (`--real-finish`). On those steps the splat positions move at 0.2× the learning rate and the loss is the reference's: L1 normalised by the confidence sum plus SSIM of confidence-weighted images. The real frames train at full weight. The learning rates follow the reference fine-tune (see `fill_recipe` in crates/splat-fill/src/lib.rs); no splat growth (relocation every 100 steps only), no quantisation tail, full resolution from the first step. → `model/` |

Each stage records the inputs and settings it ran with (`stamps.json`) and is skipped when they are unchanged; a change
redoes that stage and every stage after it, removing their old outputs first. So `--until cams|render|fix|train` stops
early, and a later run picks up from there. `--force` redoes everything. Images are written atomically, so an
interrupted run never leaves a truncated file that a later run would trust.

## Training in views from another generator (`--views`)

Views made outside the loop (e.g. frames of a video-model clip along a planned camera path) train in the same way:

```sh
vauss fill runs/living-room --views gen/clip-a,gen/clip-b      # comma-separated dirs are merged
```

Each dir is a pseudo-view dir in the trainer's own format, the one the confidence stage writes to `<out>/pseudo`:

- `views.json`: `{"views": [{"image", "c2w", "fx", "fy", "cx", "cy", "width", "height", "weight", "conf_mean"}, ...]}`.
  `image` is a path relative to the dir; `c2w` is the camera-to-world matrix, 4×4 row-major, OpenCV axes (+x right,
  +y down, +z forward), in the dataset's SfM units and frame; `fx fy cx cy` are pinhole intrinsics in pixels of the
  image; `width`/`height` its size; `weight` a per-view factor on `--pseudo-weight`; `conf_mean` the mean confidence
  (informational).
- one PNG per view, RGBA: RGB is the view, alpha the per-pixel confidence (0 = ignore, 255 = full trust).

With `--views`, cams, render, fix and confidence are skipped (no fixer or `--fixer-python` needed). The dirs are
checked first (views.json parses, at least one view, every image there with alpha at its listed size); `--until`
cams/render/fix only checks them. A change to a dir's views.json or images retrains. Several dirs are merged through
`<out>/views` (links to each dir, image paths prefixed `<i>/`). `fill.json` records `views_dir` and `n_views`.

## Fixers and licences

| `--fixer` | model | licence | needs |
|---|---|---|---|
| `fixer` (default) | [NVIDIA Fixer](https://github.com/nv-tlabs/Fixer) (Cosmos-Predict2 0.6B, one step). `--fixer-ckpt stock` or a fine-tuned `.pkl` | code Apache-2.0; weights [nvidia/Fixer](https://huggingface.co/nvidia/Fixer) under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/): commercial use allowed, **"Built on NVIDIA Cosmos"** attribution required where the licence asks for it (e.g. models improved with its outputs) | `FIXER_SRC`, `FIXER_MODELS` |
| `difix-ref` | [Difix3D+](https://github.com/nv-tlabs/Difix3D) with a reference photo (`nvidia/difix_ref`) | **NON-COMMERCIAL. Research use only.** | `DIFIX_SRC` |

Every run records what repaired its views in `<out>/fixer.json` (model, checkpoint path and SHA-256, the weights'
Hugging Face revision from `FIXER_MODELS/MANIFEST.json`, licence), copied into `fill.json`.

Set up the Python side once per machine (Linux + NVIDIA GPU, no root needed, ~5 min plus a 5.5 GB download):

```sh
bash crates/splat-fill/fixer/setup_fixer.sh          # fixer;  `difix` or `all` for the non-commercial Difix
# prints: export FIXER_SRC=... FIXER_MODELS=... and the --fixer-python to pass
```

### Plugging in another fixer

The fixer is a subprocess behind a directory protocol, so any model can be dropped in:

- input: a directory of PNG renders, plus an optional directory of reference photos with the same names (the
  training photo that sees most of each view);
- output: one repaired PNG per render, same name, same size.

```sh
vauss fill runs/room --fixer-cmd 'python my_fixer.py --in {in} --out {out} --ref {ref}'
```

The command is split on whitespace and run directly, never through a shell; the exact tokens `{in}`, `{out}` and
`{ref}` become the paths. Wrap anything that needs shell syntax in your own script. A fixer that stops part-way is
re-run for the views it did not finish (or left unreadable), up to twice.

The confidence step still runs through `--fixer-python` (it needs `torch` and `lpips`).

## Knobs

`--max-dist` (how far off the path), `--steps`, `--pseudo-frac`, `--pseudo-weight`, `--ramp`, `--pos-lr`,
`--heights`, `--yaw`, `--pitches`, `--vfov`, `--grid`, `--clearance`, `--hull-margin`, `--lp-tau`/`--lp-floor`.
`--real-finish F` (default 0.3) trains the last F of the steps on real frames only (0 = novel views to the end); the
research reference measured −0.4 dB on the walked path without such a finish.
`--exclude grid.json` keeps novel cameras at least 10 cm from an evaluation grid's eyes, so off-path scores stay
honest. `--limit N` uses a random N views (smoke tests).

## Results (public room, measured)

The CC BY iPhone room clip used for the examples (see examples/DATA.md): 330 training frames, 47 near held-out frames, 105
far real frames 1–4.5 m off the path, 480 fixed grid views on the free floor. Scored with one gsplat-based scorer for every
arm: held-out PSNR/SSIM/LPIPS, MUSIQ and NIQE on the grid, and free-space floater count. One seed per arm; on-path PSNR on
this clip moves up to ~1.3 dB from run to run.

| arm | repaired views | near PSNR / LPIPS | far 1–2 m PSNR / LPIPS | grid MUSIQ all / 0.5–1 m | free-space floaters |
|---|---|---|---|---|---|
| starting splat (4.5M splats) | – | 26.73 / 0.211 | 11.60 / 0.623 | 29.66 / 33.61 | 109k |
| research reference (gsplat fill.py), Difix-ref | 4,327 | 27.92 / 0.214 | 12.08 / 0.604 | 30.11 / 35.62 | 54k |
| `vauss fill --fixer difix-ref` | 4,356 | 26.74 / 0.216 | 12.15 / 0.601 | 30.20 / 35.41 | 67k |
| `vauss fill` warm start, no novel views (control) | – | 26.99 / 0.212 | – | 29.87 / 33.69 | 108k |
| `vauss fill` (default: stock NVIDIA Fixer) | 4,357 | 25.93 / 0.220 | 12.67 / 0.598 | 29.09 / 34.42 | 62k |

What this says:
- With the same fixer as the reference, `vauss fill` matches or beats it off the path (MUSIQ, far PSNR and LPIPS) and keeps
  the walked path within 0.25 dB of its own no-fill control. It does **not** reproduce the reference's +1.2 dB on the walked
  path: that gain comes from the reference's gsplat retrain of the real frames, not from the novel views. The reference
  also removes more floaters (54k vs 67k).
- The stock (not fine-tuned) NVIDIA Fixer rewrites more of each view than Difix-ref and costs about 1 dB on the walked path
  at the default weight, with MUSIQ below the starting splat.

### Follow-up round: the reference's regularisers and lower Fixer weights (not on `main`)

We ported the reference's per-step regularisers (0.01 × mean opacity and 0.01 × mean scale on every fine-tune step) into
the fill fine-tune on an experimental branch (`fill-regs`; plain `vauss train` unchanged) and re-ran the arms above with
three training seeds where noted. Same clip, scorer and views; the Difix-ref arms of this round were trained but their
scoring did not finish, so they are not reported.

| arm (with the regularisers) | seeds | near PSNR / LPIPS | far 1–2 m PSNR / LPIPS | grid MUSIQ all | free-space floaters |
|---|---|---|---|---|---|
| warm start, no novel views (control) | 3 | **27.15** (27.09–27.21) / 0.214 | 11.76 / 0.619 | 29.11 | **65k** |
| stock NVIDIA Fixer, `--pseudo-weight 0.3` (default) | 1 | 26.82 / 0.221 | 12.70 / 0.604 | 28.81 | 59k |
| stock NVIDIA Fixer, `--pseudo-weight 0.15` | 2 | 26.52 (26.44–26.60) / 0.222 | 12.65 / 0.603 | 28.86 | 59k |
| stock NVIDIA Fixer, `--pseudo-weight 0.1` | 1 | 26.75 / 0.220 | 12.53 / 0.608 | 28.71 | 60k |
| stock NVIDIA Fixer, 0.15, `--lp-floor 0` | 1 | 26.72 / 0.219 | 12.52 / 0.606 | 29.12 | 59k |
| stock NVIDIA Fixer, 0.3, `--pos-lr 0` | 1 | 26.26 / 0.224 | 12.71 / 0.600 | 29.11 | 57k |

What this says:
- The regularisers do what they do in the reference to floaters (109k → 65k without any novel views; the reference reaches
  54k) and add about +0.16 dB on the walked path (26.99 → 27.15), at a cost in grid MUSIQ (29.87 → 29.11). They do not close
  the gap: the control stays 0.77 dB below the reference's 27.92, and novel views do not add on-path PSNR in any arm, so a
  `vauss fill` on-path score within 0.3 dB of the reference is not reachable with this change. Seed spread of the Rust
  control is small (0.13 dB over three seeds), so the gap is not noise.
- The stock NVIDIA Fixer still costs 0.3–0.9 dB on the walked path against that control at every weight tried (0.1–0.3),
  with or without the confidence floor and position updates, while it keeps the off-path gain (+1 dB far 1–2 m PSNR over
  the starting splat, lower far LPIPS, ~59k floaters). Its grid MUSIQ stays below the starting splat (28.7–29.1 vs 29.66).
- It is not the steps the novel views take away from the real frames: the trainer's own held-out PSNR (brush renderer,
  same 47 frames) is 27.80 for the control at 6,000 steps (3 seeds), 27.79 at 4,200 steps, and 27.55 for the default
  Fixer arm at 6,000 steps vs 27.50 at 8,572 steps (6,000 of them on real frames). Fewer novel-view steps
  (`--pseudo-frac 0.15`) lose less: 27.66.

Time on one RTX PRO 6000 (4,356 views): cams 107 s, render 203 s, Difix-ref 52 min (~83 views/min; stock NVIDIA Fixer ran ~224 views/min on the same GPU type),
confidence 453 s, training 167 s (6,000 steps). `--fixer-gpus` splits the fixer and confidence steps across GPUs.

## Next steps

- Find where the reference's retrain gains its extra on-path PSNR beyond the regularisers (+0.77 dB over the Rust control
  with them): its trainer and renderer (gsplat, antialiased, the scorer's own renderer) differ from the Rust fine-tune.
- Default fixer: the stock NVIDIA Fixer's on-path loss is in its views, not in their weight; a scene-fine-tuned
  `--fixer-ckpt`, or keeping novel-view losses off pixels the real frames see well, are the next things to try.
- Score the Difix-ref arm with the regularisers (three seeds) before deciding whether they go into the default recipe.

