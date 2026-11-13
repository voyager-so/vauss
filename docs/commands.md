# Commands

`vauss --help` lists every command and `splat <command> --help` its flags. Put `--json` before the command
(`vauss --json status runs/room`) for machine-readable output. This page covers the ones you will use.

## Check a video before you spend GPU time

```sh
vauss check room.mp4 --room     # --room for interiors (adds a coverage check); omit it for objects and places
vauss check --guide             # print the filming guide
```

A few seconds of local analysis, no GPU. It reports length, light, sharpness, movement, detail, speed and
(for rooms) coverage as pass / warn / fail, with a fix for each problem, and exits with code 6 when the video
won't work. `vauss run` runs the same check first and stops on a fail (`--ignore-check` overrides it). What each
line means: [filming-guide.md](filming-guide.md).

## Reconstruct: `vauss run`

```sh
vauss run SOURCE --goal "what you want" --driver fixed [--run-dir runs/name]
```

`SOURCE` is a video, a folder of images, or a phone capture export (Record3D, Polycam, Nerfstudio
`transforms.json`, ARKitScenes; see [phone-capture-import.md](phone-capture-import.md)).

`--driver` picks who runs the pipeline:

| Driver | What happens |
|---|---|
| `fixed` | A fixed, deterministic pipeline. No AI, no account needed. Recommended to start. |
| `claude` (the default) | Claude Code runs the pipeline through splat's tools, reads the metrics and images, and retries failed steps. Needs the `claude` CLI installed and logged in. |
| `codex` | The same with OpenAI's `codex` CLI. |

The goal is free text. The AI drivers read it as instructions. The fixed driver looks for three things in it:

- a room word ("living room", "kitchen"…) turns on the room coverage check (same as `--room`);
- "mesh", "glb" or "surface" trains flat surfels (2D Gaussian splatting) instead of 3D Gaussians, which gives
  a cleaner mesh at some cost in image quality;
- "remove the <object>" selects that object after training, deletes it, and makes the edited model the result.
  This needs the optional Spirula segmenter (GPL-3.0, a separate program): opt in with
  `vauss fetch spirula --allow-gpl` or `SPLAT_ALLOW_GPL=1`; see [agents.md](agents.md#the-tools).

What the fixed pipeline does: check the video → pick sharp frames → solve camera poses (COLMAP) → freeze a
held-out split (every 8th registered frame is never trained on) → a quick preview → full training →
web and viewer export → held-out evaluation (PSNR, SSIM, LPIPS) → mesh → report.

Quality presets:

| Flags | Training | Splat budget | Image size | Extras |
|---|---|---|---|---|
| `--fast` | 7,000 steps | 250,000 | 800 px long edge | No mesh, no evaluation (marked partial) |
| (default) | 7,000 steps | up to 1,000,000 | 1,600 px long edge | Mesh, evaluation |
| `--max` | 45,000 steps | scene-adaptive, up to 5,000,000 | full resolution | Mesh, evaluation |

`--steps`, `--cap-max` (splat budget), `--max-edge` and `--no-mesh` override single settings.
`--time-budget` caps an AI driver's wall time.

## Long runs: status, logs, stop, resume

A run keeps going in the background if your terminal closes. Every command below takes the run directory:

```sh
vauss status runs/room            # stage, progress, ETA, result when done
vauss logs runs/room --follow     # stream the pipeline log
vauss stop runs/room              # stop gracefully (training saves a checkpoint)
vauss resume runs/room            # continue a stopped or failed run where it left off
```

Progress lines also go to stderr while `vauss run` is attached. `vauss --json status` includes a `progress`
object (`stage`, `fraction`, `eta_s`). ETAs are estimates of remaining compute for the current stage.

## Improve a finished run: `vauss refine`

```sh
vauss refine runs/room --steps 30000              # train the result further, in place
vauss refine runs/room --max --run-dir runs/room-max   # into a copy; the original is untouched
```

Reuses the run's cameras and frozen held-out split (no new camera solve), trains on from the final
checkpoint, re-evaluates, re-meshes if the run had a mesh, re-exports, and makes the refined model the run's
result.

## Look at the result

```sh
vauss view runs/room                 # also takes .ply / .spz files or scene folders; no argument opens a picker
```

Starts a local web server on `127.0.0.1` (port 8787, or the next free one) and opens your browser. Every
finished run also has double-click viewers that need no server; see [outputs.md](outputs.md).

## Exports

| Command | Writes |
|---|---|
| `vauss export-viewer RUN` | `RUN/exports/viewer/`: standalone double-click viewer (also written by every run) |
| `vauss export-web RUN --out DIR` | Portable web scene: `scene.spz` + `scene.json` (+Y up, metres, floor at y = 0) |
| `vauss export-collision DIR [--unity OUT]` | A closed collision mesh, convex pieces and `floor.json` for game engines |
| `vauss export-blender RUN` | `RUN/exports/blender/`: textured mesh, full splat PLY, web scene.spz, voxels, Blender import steps |
| `vauss export-game RUN` | `RUN/exports/game/`: a first-person voxel game of the scene |

Each runs on the CPU in seconds to a minute. `SPLAT_EXPORT_VIEWER=0`, `SPLAT_EXPORT_BLENDER=0` and
`SPLAT_EXPORT_GAME=0` skip the automatic ones during `vauss run`.

## Camera moves and videos

```sh
splat scene --ply runs/room/<candidate>/splat.ply --data runs/room/<sfm>/dataset   # scene frame used for planning
vauss shot  --ply ... --data ... --out shots/push --prompt "slow push-in, 6s"
vauss stabilize --ply ... --data ... --out shots/steady --video room.mp4
splat review shots/push shots/steady                                               # side-by-side HTML
```

`shot` plans a camera path from a plain-English prompt (orbit, push-in, crane up, dolly…), checks how far it
strays from the filmed viewpoints, and renders it to MP4. `stabilize` re-renders the capture's own path,
smoothed and level. Both run locally on your GPU. With `--model` they can also send the render to hosted video
models (needs a fal.ai key in `FAL_KEY`; costs money). See [examples/camera-moves](../examples/camera-moves/README.md).

## Individual steps and agents

```sh
vauss tools                                            # the typed tools behind every pipeline step
vauss init room.mp4 --goal "my room" --run-dir runs/r  # make an empty run
vauss tool extract_frames --run-dir runs/r --input '{}'
vauss serve --run-dir runs/r                           # MCP server over stdio, for your own agent
```

The CLI, the MCP server and the AI drivers all call the same tools, so an agent run and a scripted run are the
same code path. See [agents.md](agents.md).

## Setup and helpers

```sh
vauss doctor             # GPU adapter, each helper and where it was found (bundled, cache, download)
vauss fetch colmap       # download a pinned helper ahead of time
```

## Frame inclusion, provenance, and matching

`--target-count` is a selection budget, not a promise that every file is included. Sampling,
black-frame filtering, duplicate filtering, and camera registration can all reduce coverage.
Use the Brush backend's explicit preserve-all contract when every input must be retained:

```sh
vauss run ./extracted-frames --preserve-all --input-kind video --goal "reconstruct this capture"
vauss run ./photos --preserve-all --goal "reconstruct these photos"
```

`--preserve-all` keeps every readable image or decoded source-video frame, including black frames
and duplicates, regardless of `--target-count`. It disables sampled previews, resampling retries,
SfM keyframe thinning, and training view caps (including `SPLAT_TRAIN_MAX_FRAMES`). A decode failure
or suspected decoder corruption fails extraction instead of silently dropping frames. It cannot
be combined with a trimmed time range or preview extraction. Resizing and image calibration still
apply. It does not guarantee camera registration: unregistered views are reported separately, and
held-out views remain reserved for evaluation. Large all-frame videos may cost substantially more.

`--input-kind video` declares that an image folder contains chronological frames from one video.
Use filenames that express that order (`frame_2` sorts before `frame_10`); the selection manifest
records each original filename and index. Video files carry video provenance automatically.
Ordinary folders default to `images`; filenames alone never change their calibration assumptions.

Video auto-calibration uses the existing shared square-pixel RADIAL model and available focal
prior. Photo auto-calibration retains PINHOLE with independent fx/fy. For known anisotropic frames,
use `--camera-model pinhole` or `--camera-model opencv`, even with video provenance. When metadata is unavailable,
the existing video recovery policy can use a labeled 65-degree default lens prior; explicit
PINHOLE/OPENCV models retain independent fx/fy through video calibration retries.

Video auto matching starts sequentially with vocabulary-tree loop detection, retaining the
existing hybrid coverage recovery. Photo auto matching is exhaustive. `--pairs sequential` and
`--pairs exhaustive` explicitly choose the initial matcher; sequential retains loop detection,
while exhaustive is never silently replaced with hybrid. Without a configured vocabulary tree,
the API uses sequential quadratic long-range pairs and warns that visual loop detection is absent.

For tool/API callers, `extract_frames` accepts `preserve_all: true` and `input_kind: "video"`.
`reconstruct_sfm` accepts `pairs` and `camera_model`; `run_default_pipeline` accepts all four
options and otherwise inherits the CLI settings. Preserve-all follows the frame artifact into
training. Spirula rejects these new extraction options explicitly; use Brush for this contract.

`selection.json` and extraction metrics report `input_count`, `selected_count`, `dropped_count`,
`drop_reasons`, and `input_count_basis`. A normal sampled video counts its decoded sampling timeline;
preserve-all counts decoded source frames, avoiding constant-rate resampling of variable-rate video.
`report.json.frame_counts` binds input/selected/dropped/registered/train/heldout counts to the selected
candidate, with separate unregistered and training-cap exclusions. Planned training counts are not
reported as completed training. Missing or legacy-stage counts remain null with an explanation.
