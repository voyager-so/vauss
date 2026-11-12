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

Nerfstudio `test_filenames` is supported as an explicit frozen evaluation split. Its paths must name frames present in the capture and retained by extraction. Those frames are excluded from training and sensor-depth point initialization. Without an explicit split the normal every-eighth split applies. Keep enough training frames, and pass a sufficient `--frames` count when importing a large capture with explicit test views.
