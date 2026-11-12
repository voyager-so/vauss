# Splat collision package

`collision.glb` and `collision.obj` contain the same static, closed triangle mesh, in the exact
Y-up frame of `scene.spz`. It is built on CPU by depth fusion: the splats' depth is rendered at
every capture camera, seen-free space carves away floaters, small gaps are closed, and a solid
floor slab is always added under the occupied footprint.

Units are scene units: metres only as far as the export's scale estimate is right.
`floor.json` records `scale_estimate` (with its confidence), the export floor, and
`collision_floor`: the slab top `y`, how it was found, and `implied_units_per_meter`
(camera height above that floor over a handheld 1.5 m). Rescale by `1 / implied_units_per_meter`
when the scale confidence is low. `manifest.json` records the grid, triangle counts,
boundary/nonmanifold edges and timings. `convex/` holds V-HACD pieces.

The package is published first; collision/ is added afterwards and scene.json gains
`collision` (and `collider` when the run has no TSDF mesh). Quick previews have none.

## Unity — four steps

1. Install [aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)
   (MIT) following that project's render-pipeline setup. It is referenced, not bundled.
2. Make the Unity folder (it holds a full-size splat PLY copy, so packages do not carry it):
   `vauss export-collision <scene package or run> --unity <dir>`, then copy
   `<dir>/Assets/SplatScene` into your Unity project's `Assets/`. The OBJ importer
   enables colliders and uses scale 1. Import `scene.ply` using the plugin's Gaussian
   Splatting asset creator, then place its renderer at the origin with identity rotation
   and scale 1 (this PLY is already Y-up).
3. Choose **Tools > Splat > Create Static Collision**. This creates a static object with
   non-convex MeshCollider and a hidden mesh renderer. The script reads the OBJ coordinates
   directly, bypassing model-import axis changes to match the plugin's PLY reader.
   Use a capsule/player Rigidbody
   to collide with it; keep this environment static.
4. Check alignment with the collision renderers temporarily enabled. Match the plugin's
   identity transform on both objects, then parent splat and collision together for
   any scene rotation or scale correction. Do not apply `transform_from_source` again:
   it is already baked into both files.

## Godot

Import `collision.glb`, open the imported scene, and create a Trimesh Static Body from its mesh.
Load the splat with your chosen splat addon in the same Y-up frame; parent both for scale changes.

## Unreal

Import `collision.obj` (or GLB via Interchange), then set Collision Complexity to **Use Complex Collision As Simple** for a static environment.
Set meters-to-centimeters scale to 100 and convert Y-up to Z-up consistently for both splat and mesh.

## Dynamic bodies

`convex/hull-*.glb` and matching `.obj` files are V-HACD convex pieces in the same scene
frame. Add the pieces to one compound Rigidbody/physics body (convex MeshColliders in
Unity, convex collision shapes in Godot, simple convex collision in Unreal). These are
approximate pieces, not semantic object segmentation. Do not attach the entire static
room's triangle mesh to a moving Rigidbody. Each convex piece is limited to 128 vertices / 252 triangular faces for PhysX import.
Large hulls are reduced by support sampling and rehulling, which can slightly shrink them.

Depth fusion reconstructs observed opaque surfaces; it cannot infer hidden physical
material. Transparent surfaces, vegetation, mirrors and weakly observed areas may differ from reality.

Neural appearance is not supported by the referenced Unity plugin: the supplied PLY
retains its base color/SH appearance. No GPU is used to generate this package.
