# Import phone captures

Pass a capture directory, `.zip`, or Record3D `.r3d` export to `vauss run` in place of a video:

```sh
vauss run /path/to/capture --goal "my kitchen" --driver fixed
vauss run /path/to/capture.r3d --goal "my kitchen" --driver fixed
```

The pipeline extracts calibrated frames and runs its usual SfM first. If SfM registers at least 90% of the extracted frames, its cameras and sparse points are retained. Below 90%, or if SfM cannot reconstruct a model, the pipeline falls back to the capture's full calibrated trajectory in metres and enables training pose refinement. Exactly 90% keeps SfM. The usual preview, training, held-out evaluation, and viewer export follow. Input detection is automatic; an archive must contain a single capture.

Supported layouts:

| Export | Required files | Optional depth |
| --- | --- | --- |
| Record3D | `metadata` or `metadata.json` with `K`, `poses`, `w`, `h`; numeric RGB filenames in `rgb`, `rgbd`, or `images` | LZFSE `.depth` with `dw`/`dh`, EXR, or uint16 PNG; LZFSE `.conf` |
| Polycam raw | `keyframes/cameras` + `images`, or `corrected_cameras` + `corrected_images`; per-frame JSON or `cameras.json` | `depth/<stem>.png` or `depths/<stem>.png` |
| Nerfstudio / Spectacular AI | `transforms.json` with `frames`, image paths, camera-to-world matrices, and focal lengths or camera angles | Per-frame `depth_file_path`; `depth_unit_scale_factor` for integer depth (default 0.001 metres) |
| ARKitScenes | `lowres_wide.traj`, `lowres_wide` + `lowres_wide_intrinsics`, or matching `vga_wide` directories | `lowres_depth/<stem>.png` |

Spectacular Rec's raw video/IMU recordings require the capture application's pose export first. Use its Nerfstudio export; the importer does not run a VIO SDK or accept raw IMU streams.

Record3D, Polycam, and Nerfstudio poses use OpenGL camera axes. Import changes the camera's Y and Z axes to OpenCV's right/down/forward convention. It preserves world axes and translations. ARKitScenes trajectories already describe OpenCV world-to-camera poses; import inverts them without another axis flip.

Frames may be uniformly resized, with corresponding intrinsics scaled. Stored pixels are never rotated according to EXIF or cropped. Rotation and time-range overrides are rejected for calibrated inputs. Corrected Polycam cameras require corrected images. Registered depth must match the RGB aspect ratio.

The generated dataset contains ordinary COLMAP `cameras.bin`, `images.bin`, and `points3D.bin`, plus a `capture.json` provenance manifest. `capture_selection.json` and `health.json` record the attempted SfM coverage and the winning camera source (`sfm` or `capture_fallback`). Sensor depth stays available with either camera source as `depths/<frame stem>.png`: uint16 optical-axis Z, zero invalid. Phone-pose fallback uses millimetres and a bounded sensor-depth point seed. With SfM cameras, depth is scaled to SfM world units using sparse observations from training views; training camera baselines provide a secondary scale estimate when depth observations are insufficient. The encoded value divided by 1000 gives Z in those world units. `depths.json` keeps `kind: "metric"` and records `units_per_meter`, fitting method, and any pixels discarded outside uint16 range. Confidence becomes `depths_conf/<stem>.png` with values 0, 128, 255. The depth-loss integration reads these sidecars separately. SfM retains its own sparse point seed.

Nerfstudio pinhole, OpenCV, full OpenCV, and OpenCV fisheye camera models are accepted. Perspective radial `k3` is preserved with the full OpenCV model; Nerfstudio's eighth-order `k4` requires an undistorted export. Unsupported models, non-rigid poses, missing calibration, unsafe archive paths, and multiple captures fail with a diagnostic instead of falling back to SfM. Existing JPEG pixels are retained when no resize is needed; other frames are re-encoded as quality-90 JPEG. Canonical `frame_000123.jpg` filenames are preserved when every input frame follows that pattern, so paired video evaluations retain their source identities.

Nerfstudio `test_filenames` is supported as an explicit frozen evaluation split. Its paths must name frames present in the capture and retained by extraction. Those frames are excluded from training and sensor-depth point initialization. Without an explicit split the normal every-eighth split applies. Keep enough training frames, and pass a sufficient `--target-count` count when importing a large capture with explicit test views.

## Authoritative imported cameras

```sh
vauss run ./calibrated-capture --trust-poses --driver fixed --goal "reconstruct the authored world"
```

`--trust-poses` explicitly selects `camera_source: "imported"` and
`sfm_skipped: true`. It requires the Brush backend and a supported calibrated
layout from the table above. Plain photos, videos, raw IMU streams, and standalone
COLMAP folders are not supported by this flag. No COLMAP solve, learned-camera
rescue, matching, or quick SfM preview runs. Without this flag the SfM-first
policy described above remains the default; `--preserve-all` and `--driver fixed`
do not select imported cameras.

Trusted imports retain every source view, including explicit held-out views;
target-count, input-kind, matcher and camera-model overrides are incompatible. Rotation, partial time ranges, motion windows and
preview extraction are rejected. A smaller `--max-edge` may resize RGB: width,
height, fx/cx and fy/cy scale by the corresponding image-axis ratio. Set it at
least as large as the source edge to retain native K. Unequal fx/fy and all
supported lens coefficients are preserved. JPEG conversion rules above still
apply.

Nerfstudio `test_filenames` names the frozen held-out set. Optional
`train_filenames` must be nonempty; when both lists exist they must be disjoint
and together cover every frame. With only train names, the other frames are
held out. With neither list, or with an empty test list, every frame trains and
no evaluation views are invented. Such a run reports quality as unmeasured and
finishes partial after export. Duplicate, unknown, overlapping names and an
empty training set fail clearly. Held-out RGB/depth never seeds training.

The health policy is **structural validation**, not a claim of reconstruction
quality. Trusted cameras require finite rigid homogeneous transforms (rotation
orthonormality and determinant within 1e-6), positive focal lengths, finite principal
points (including off-image principal points), calibration dimensions matching stored RGB, unique
source images, and a baseline greater than 1e-9 source units. Malformed or
unsupported imports fail; there is no reconstruction fallback or pose repair.
Straight camera paths are supported. This cannot establish scene rigidity,
image overlap, accurate lens calibration, or photometric quality.

Pose refinement, focal refinement and held-out camera alignment are disabled,
including automatic enablement, `SPLAT_POSE_REFINE=1`, and serialized config
overrides. There is currently no opt-in camera refinement for trusted imports.
Source world axes, scale and translations survive import and training-camera
loading. Only the documented OpenGL/OpenCV camera-axis conversion applies.
Web export intentionally applies the similarity in `scene.json`
`transform_from_source`; apply that same matrix to authored world landmarks.
Exported `source_c2w` retains the source frame and `c2w` describes the export
frame with a rigid, unit-length camera basis. Reports and `capture_selection.json`
identify the authority; imported view counts are not measured SfM registration.

## Extreme solved-calibration drift

SfM-first captures with supplied calibration, and video without calibration,
now reject a selected solve whose normalized focal aspect
`(fx / width) / (fy / height)` differs by more than **2×** from its prior in
either direction. Supplied per-view K takes precedence and is compared with
its own dimensions, so explicit unequal K and image resizing remain valid.
Only video lacking supplied K uses a square-pixel prior. Unknown photo optics
receive no guessed aspect prior. `health.json.calibration_health` identifies each
camera/image, solved/prior calibration and drift factor; the worker rechecks
before training. An unhealthy high-coverage solve fails instead of silently
rewriting K or selecting imported cameras. The existing below-90% capture
fallback is unchanged. This conservative gate catches extreme aspect drift;
it does not prove either focal length or all distortion coefficients are correct.
