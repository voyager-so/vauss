# splat documentation

New here? Start with the [README](../README.md) quickstart, then a worked [example](../examples/README.md).

## Using splat

| Page | What's in it |
|---|---|
| [commands.md](commands.md) | Every command you will use: check, run (drivers, presets), status/logs/stop/resume, refine, view, exports, shots, tools |
| [outputs.md](outputs.md) | What a run writes: run directory, metrics and report, splat files, viewers, web scene, collision, Blender and game exports |
| [filming-guide.md](filming-guide.md) | How to film a room or an object, and what each line of `vauss check` means (also `vauss check --guide`) |
<<<<<<< HEAD
| [phone-capture-import.md](phone-capture-import.md) | Using Record3D, Polycam, Nerfstudio, ARKitScenes and COLMAP-folder captures instead of a plain video |
| [fisheye.md](fisheye.md) | Fisheye and action-camera footage: the `--camera-model opencv-fisheye` flag, image-circle masks, calibrated fisheye captures |
=======
| [phone-capture-import.md](phone-capture-import.md) | Using Record3D, Polycam, Nerfstudio and ARKitScenes captures instead of a plain video |
| [360.md](360.md) | Stitched 360 (equirectangular) video and photos: detection, the flags, what to expect |
>>>>>>> 7bdaee57 (feat(360): --camera-model equirect through vauss run, exports, acceptance and docs (PR 3 of 3))
| [agents.md](agents.md) | The typed tools, the MCP server (`vauss serve`) and the Claude Code / Codex drivers |
| [fill.md](fill.md) | `vauss fill`: generative fill-in so a room holds up away from the filmed path (NVIDIA Fixer by default) |

## Speed and quality

| Page | What's in it |
|---|---|
| [performance.md](performance.md) | Where the time goes in a run, which settings change it, measured numbers |
| [benchmarks.md](benchmarks.md) | Trainer quality and speed on Mip-NeRF 360, Tanks and Temples and Deep Blending |

## Building and packaging

| Page | What's in it |
|---|---|
| [distribution.md](distribution.md) | Release archives, installers, bundled helpers, the download cache, licenses, building archives |
| [distribution-colmap.md](distribution-colmap.md) | How the bundled Linux COLMAP is built and audited |
| [provenance/](provenance/) | Clean-room statements for code written from descriptions of other projects |
