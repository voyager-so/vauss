"""Build a .blend from a shot bundle. Runs inside Blender (bundled Python, bpy):

    blender -b --factory-startup --python blender_build.py -- BUNDLE_DIR [--render-frames 0,60,119]

Reads BUNDLE_DIR/bundle.json, written by splat_shots::blender::export_bundle (crates/splat-shots/src/blender.rs). Everything is
placed in a Z-up frame (scene up -> +Z, focus point on the ground at the origin), so the scene opens
right-side up. Writes scene.blend, scene.usdc (camera + mesh for other DCCs) and optional
workbench stills under renders/ to check the camera against the vauss render.
"""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

# OpenCV camera (x right, y down, z forward) -> Blender camera (x right, y up, -z forward)
CV_TO_BLENDER_CAM = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
# glTF importer converts Y-up to Z-up; undo it so the mesh is back in training coordinates
GLTF_TO_TRAIN = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def main(bundle: Path, render_frames: list[int]) -> None:
    b = json.loads((bundle / "bundle.json").read_text())
    M = Matrix(b["blender_from_train"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = b["fps"]
    scene.render.resolution_x, scene.render.resolution_y = b["width"], b["height"]
    scene.frame_start, scene.frame_end = 0, len(b["c2w"]) - 1

    if b.get("mesh"):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(bundle / b["mesh"]))
        for ob in set(bpy.data.objects) - before:
            if ob.parent is None:
                ob.matrix_world = M @ GLTF_TO_TRAIN @ ob.matrix_world
    if b.get("points"):
        bpy.ops.wm.ply_import(filepath=str(bundle / b["points"]))
        pts = bpy.context.selected_objects[0]
        pts.name = "splat_points"
        pts.matrix_world = M @ pts.matrix_world
        # show the point cloud as small spheres in viewport/workbench renders
        mod = pts.modifiers.new("points", "NODES")
        tree = bpy.data.node_groups.new("splat_points", "GeometryNodeTree")
        tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        n_in, n_out = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
        to_pts = tree.nodes.new("GeometryNodeMeshToPoints")
        to_pts.inputs["Radius"].default_value = b["point_radius"]
        tree.links.new(n_in.outputs[0], to_pts.inputs["Mesh"])
        tree.links.new(to_pts.outputs[0], n_out.inputs[0])
        mod.node_group = tree

    cam_data = bpy.data.cameras.new("shot_cam")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.lens_unit = "FOV"
    cam = bpy.data.objects.new("shot_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    for f, (c2w, vfov) in enumerate(zip(b["c2w"], b["vfov_rad"])):
        cam.matrix_world = M @ Matrix(c2w) @ CV_TO_BLENDER_CAM
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        cam_data.angle = vfov
        cam_data.keyframe_insert("lens", frame=f)
    # avoid euler flips between frames (rotation_euler keyed per frame is enough; just be linear)
    if cam.animation_data and cam.animation_data.action:
        for fc in getattr(cam.animation_data.action, "fcurves", []):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    world = bpy.data.worlds.new("world")
    world.color = (0.05, 0.05, 0.06)
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.6, 0.2, 0.8)

    bpy.ops.wm.save_as_mainfile(filepath=str(bundle / "scene.blend"))
    bpy.ops.wm.usd_export(filepath=str(bundle / "scene.usdc"), selected_objects_only=False)

    if render_frames:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.color_type = "TEXTURE" if b.get("mesh") else "ATTRIBUTE"
        scene.render.image_settings.file_format = "PNG"
        (bundle / "renders").mkdir(exist_ok=True)
        for f in render_frames:
            scene.frame_set(f)
            scene.render.filepath = str(bundle / "renders" / f"{f:05d}.png")
            bpy.ops.render.render(write_still=True)
    print("BUILD_OK", json.dumps({"objects": [o.name for o in bpy.data.objects], "frames": scene.frame_end + 1}))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :]
    frames: list[int] = []
    if "--render-frames" in argv:
        frames = [int(x) for x in argv[argv.index("--render-frames") + 1].split(",")]
    main(Path(argv[0]).resolve(), frames)
