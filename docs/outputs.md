# What a run produces

`vauss run SOURCE --run-dir runs/room` writes everything into `runs/room/`. This is the layout from the
[room-walk example](../examples/room-walk/README.md):

```
runs/room/
  report.md, report.json      verdict, metrics, deliverables, held-out comparison images
  journal.jsonl               every tool call, result, decision and progress event, in order
  run.json                    source, goal, driver
  capture_check.json          the result of `vauss check`
  artifacts/
    frames_*/                 the selected frames
    sfm_*/                    camera solve: COLMAP database, sparse model, health metrics
    split_*/                  the frozen held-out split
    candidate_*/              a trained model: splat.ply, checkpoint, training log, metrics
    eval_*/                   held-out renders, metrics and gt|render comparison images
    mesh_*/                   mesh.glb (textured) and previews
  exports/
    viewer/                   double-click viewer (index.html)
    web/                      portable web scene (scene.spz + scene.json)
    blender/                  mesh, splat and voxels for Blender, with a README
    game/                     double-click first-person voxel game
```

Ids such as `candidate_66f47d0744ec` are content-addressed, so a retry never overwrites earlier work.
`report.json` → `deliverables` names the files that count (`splat`, `mesh`). The other candidates and solves
are kept so you can see what was tried.

## Metrics

`report.md` and `report.json` give scores on held-out frames: every 8th registered frame, frozen before training
and never trained on.

| Metric | Meaning | Better |
|---|---|---|
| `psnr` | Pixel accuracy in dB, after a short pose alignment of the held-out cameras (`psnr_unaligned` without it) | higher |
| `ssim` | Structural similarity, 0–1 | higher |
| `lpips_vgg`, `lpips_alex` | Perceptual distance, 0–1 | lower |
| `cc_psnr`, `cc_ssim` | The same after a per-image colour correction, so exposure changes in the video count less | higher |
| `worst_views` | The held-out frames with the lowest PSNR, worth looking at first | |

On a GPU without the ONNX Runtime GPU provider, LPIPS runs on the CPU on images downscaled to 512 px
(`lpips_scale: "cpu<=512"`). Don't compare those numbers with full-resolution LPIPS.

These are on-path scores: held-out frames sit between training frames. They say nothing about how the scene
looks far from where the camera went.

## The splat

`artifacts/candidate_*/splat.ply` is a standard 3D Gaussian splatting PLY (positions, scales, rotations,
opacity, spherical harmonics). Most splat viewers and editors open it, as does `vauss view`.
`exports/web/scene.spz` is the same model in the compressed SPZ format.

## Viewers

- `vauss view runs/room` serves the run on `127.0.0.1` and opens the browser viewer.
- `exports/viewer/index.html` opens by double-click (file://, no server; three.js and the Spark renderer load
  from jsDelivr). It shows a small first picture at once, then the interactive model (up to 1M splats); "Load full
  quality" swaps in the full model. It opens in **walk** mode: you stand at eye height on the reconstructed floor,
  drag to look, and move with WASD/arrows, the scroll wheel or right-drag. A collision body keeps you out of walls
  and furniture and steps over obstacles up to 20 cm. Nothing ties you to the filmed path, so you can walk
  anywhere, but quality drops away from where the camera went. O switches to orbit/fly (Q/E up and down), R
  resets, H hides the help, and the view picker jumps to named views at well-spread capture poses.
  `vauss export-viewer RUN [--views views.json]` rebuilds it.

## Web scene

`exports/web/` (or `vauss export-web RUN --out DIR`) is a portable package: `scene.spz` and `scene.json`, with +Y up,
metres and the floor at y = 0. `vauss export-collision DIR` adds `collision/`: a closed collision mesh (GLB and
OBJ) fused from the splats' depth, a floor slab, convex pieces, and `floor.json` with the floor and scale estimate.
`--unity OUT` also writes a Unity folder. Import notes for Unity, Godot and Unreal are in the generated
`collision/README.md`.

Scale is an estimate from the camera height: check `floor.json` (`implied_units_per_meter`) before relying on
metres.

## Mesh

`artifacts/mesh_*/mesh.glb` is a textured mesh made by fusing rendered depth from the training cameras (TSDF),
then marching cubes and cleanup. A goal that names a mesh trains flat surfels, which gives cleaner surfaces; see
the [car-mesh example](../examples/car-mesh/README.md).

## Blender and game

`exports/blender/` holds `mesh.glb`, the full `splat.ply`, `scene.spz` (the web export, upright and in metres;
Blender 5.3+ imports both splat files natively, older Blender needs the KIRI 3DGS Render add-on for the PLY),
`voxels-5cm.glb` (coloured surface cubes) and a README with import steps. `exports/game/index.html` is a
first-person voxel version of the scene (WASD and mouse, or a gamepad). `vauss export-blender` and `vauss export-game`
rebuild them.
