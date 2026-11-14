# Fisheye input

Vauss trains on fisheye footage directly. It doesn't undistort the frames into narrower crops. The
rasterizer projects through the Kannala-Brandt (KB4) fisheye model, COLMAP's `OPENCV_FISHEYE`, so
the whole field of view trains, including rays more than 90° off the axis.

## What works

| Input | How to run it | What happens |
| --- | --- | --- |
| Fisheye or action-cam video or photos | `vauss run clip.mp4 --camera-model opencv-fisheye ...` | COLMAP solves `OPENCV_FISHEYE` cameras, and training uses those cameras. |
| Circular fisheye (black outside the image circle) | `--camera-model auto` (the default) is enough | The image circle is detected and `OPENCV_FISHEYE` is chosen. The black area outside the circle is masked out of SfM features and the training loss. With a multi-camera rig whose circles sit in slightly different places, each frame gets its own mask. |
| Calibrated capture: Nerfstudio `transforms.json` with `camera_model: OPENCV_FISHEYE` | `vauss run ./capture ...` | SfM runs first with `OPENCV_FISHEYE`. If it places fewer than 90% of the frames, the capture's own fisheye cameras are used. |
| COLMAP folder: `images/` + `sparse/0` with `OPENCV_FISHEYE`, `RADIAL_FISHEYE`, `SIMPLE_RADIAL_FISHEYE` or `THIN_PRISM_FISHEYE` cameras | `vauss run ./colmap-folder ...` | Same as the row above. Every registered image keeps its own camera, so multi-camera rigs work too. |
| Either calibrated capture, with your poses as ground truth | add `--trust-poses` | No SfM runs. Seed points come from sensor depth, else from the COLMAP model's own points, else from COLMAP triangulation under your fixed fisheye cameras. Training never starts from the random initializer because of the lens. |

`--camera-model auto` doesn't switch to fisheye on full-frame fisheye footage, which has no black
circle: it can't tell such a lens from a wide perspective lens before solving. It solves a
perspective model and warns when the result is strongly barrel-distorted. The warning tells you to
re-run with `--camera-model opencv-fisheye`. If you know the lens is a fisheye, pass the flag from
the start.

## Outputs

- **Splat, web viewer, held-out evaluation:** these use the native fisheye cameras. Held-out PSNR
  is measured against the original fisheye frames.
- **Mesh goals (surfels):** surfels render pinhole cameras only. Fisheye frames are undistorted to
  a pinhole crop of their central view, at most 120° across, for surfel training and evaluation.
  The source frames stay untouched. Rays at or beyond 90° have no pinhole image, so the outer rim
  of a very wide lens doesn't contribute to the mesh.
- **Blender pack:** the splat, mesh and voxels import as usual. A Blender camera can't reproduce a
  fisheye lens's framing exactly, so `import_cameras.py` refuses fisheye capture cameras with an
  explanation instead of placing a wrongly framed camera. The pack's README says so too.

## Not supported yet

- 360° equirectangular video (stitched panoramas) and raw dual-fisheye `.insv` files.
- The Spirula backend passes `--camera-model opencv-fisheye` through, but that path is untested.
- The learned-pose rescue (DA3) solves perspective cameras, so it is skipped for fisheye solves.

## Focal length

COLMAP can't recover a fisheye focal length from image pairs alone, and it marks such pairs
degenerate. Fisheye solves therefore start from a focal estimate that COLMAP treats as a prior and
still refines. The estimate comes from the EXIF focal when the photos have one. Otherwise it comes
from the image circle (its rim at about 95°), or, without a circle, from a typical 150° full-frame
fisheye.

## Tips

- Keep the camera's lens correction ("Linear", "SuperView off" and similar) consistent through a
  clip. A clip that mixes corrected and uncorrected frames can't be solved as one lens.
- A fixed operator or mount visible in every frame (a helmet edge, a selfie stick) is static in
  the image. Mask it with `mask_distractors`, or crop it out before importing.
