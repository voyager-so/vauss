# Export camera acceptance

CPU regressions:

```
cargo test --locked -p splat-webexport
cargo test --locked -p splat-cli --bin vauss blender::tests
python3 -m unittest discover -s dev/camera-fidelity -v
blender -b --python dev/camera-fidelity/test_camera_import.py
```

The Blender command only evaluates its camera projection matrix, not a render. Without
Blender the Python test suite reports that check as skipped, never as verified.

Hardware GPU acceptance (cloud/CI only, with parent-coordinated bounded lease/cleanup):

```
cargo build --locked -p splat-cli --example export_camera_parity
./target/debug/export_camera_parity \
  --source /capture/splat.ply --data /capture/cameras/0 \
  --package /capture/export --image frame_000038.jpg --out /capture/parity-38
```

Regenerate `/capture/export` with this version's `vauss export-web` first, retaining
all splats (no budget), so its camera records contain exact K and matrices. `--data`
uses the same COLMAP model as training; any training `scene_transform.json` is found
next to the source PLY by the usual product resolver. Run more than one camera
(e.g. strict241 capture 38 and 84) to detect pose errors hidden in a single view.

The harness renders through the **native** Gaussian engine in all three cases:

1. Original source PLY and source COLMAP camera, with separate fx/fy/cx/cy.
2. Uncropped PLY transformed through the export similarity, including covariance
   and SH rotation, at the exported exact camera (default minimum PSNR 50 dB).
3. The file an ordinary importer receives: SPZ alone if complete, otherwise the
   complete PLY fallback. No overflow decoder is used here. This includes intended
   visibility/opacity cropping and SPZ quantization (minimum PSNR 30 dB).

All three PNGs and `metrics.json` include RGB PSNR/MAE/max-error, alpha MAE,
thresholds, splat counts, source/package paths and camera K. A failed threshold
returns a nonzero exit code after writing evidence. A mismatch is not grounds to
relax thresholds without diagnosing it. Exact PLY isolates transform/K errors from
SPZ quantization and crop losses. Pixel PSNR is measured on native RGB8 output;
alpha is compared as float. Neural and surfel models and distorted lenses are
explicitly rejected instead of approximated.

This is native-versus-export **representation** parity. Blender and web renderers
can still differ in SH support, filtering, compositing and colour management. The
camera adapter test verifies Blender framing, not rasterizer equivalence. The
separate visual-lab film renderer and launch assets are outside this work.

Compatibility: the web package retains fixed 12-bit SPZ precision and lossless
sidecars. When sidecars are needed it also writes `scene.ply` with the complete
transformed SH Gaussian data; `splat_overflow.standalone_splat` names the ordinary
import file. This costs extra disk space only for affected packages. Blender packs
never deliver a knowingly incomplete SPZ, can recover old packages' sidecars into
a PLY, and record warnings when exact camera metadata or neural appearance is
unavailable. Inferred scale remains approximate (not a metric measurement).
