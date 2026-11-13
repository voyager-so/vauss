"""Run in Blender: blender scene.blend --python import_cameras.py -- --image frame_000038.jpg

Import splat.ply or scene.ply with its numeric XYZ unchanged, then select --frame source
or scene respectively. For Blender's native SPZ importer use --frame spz (Y-up to Z-up).
Does not import or transform scene objects. Distorted lenses need undistortion first.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def camera_settings(c):
    """Blender horizontal sensor fit; preserve unequal focal axes and principal point."""
    if c["model"] not in ("PINHOLE", "SIMPLE_PINHOLE"):
        raise ValueError("Exact Blender camera requires undistorted PINHOLE/SIMPLE_PINHOLE input; got " + c["model"])
    w, h, fx, fy = (c[k] for k in ("width", "height", "fx", "fy"))
    if not all(math.isfinite(v) for v in (w, h, fx, fy, c["cx"], c["cy"])) or min(w, h, fx, fy) <= 0:
        raise ValueError("Camera dimensions and focal lengths must be positive")
    aspect = fx / fy
    if not math.isfinite(aspect) or not 1.0 / 200.0 <= aspect <= 200.0:
        raise ValueError("Focal ratio cannot be represented by Blender pixel aspect [1, 200]")
    aspect_x, aspect_y = (1.0, aspect) if aspect >= 1.0 else (1.0 / aspect, 1.0)
    return dict(lens=fx * 36.0 / w, sensor_width=36.0,
                pixel_aspect_x=aspect_x, pixel_aspect_y=aspect_y,
                shift_x=(w / 2.0 - c["cx"]) / w,
                shift_y=(c["cy"] - h / 2.0) * aspect / w)


def import_camera(c, data, frame="source"):
    """Install a calibrated camera and the actual scene render settings used for its pixels."""
    import bpy
    from mathutils import Matrix
    if frame not in ("source", "scene", "spz"):
        raise ValueError("Unknown camera frame: " + frame)
    settings = camera_settings(c)  # fail before changing the Blender scene
    values = c["source_c2w" if frame == "source" else "c2w"]
    pose = Matrix([[values[4 * col + row] for col in range(4)] for row in range(4)])
    basis = Matrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    if frame == "spz":
        basis = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    cv_to_blender = Matrix.Diagonal((1, -1, -1, 1))
    cam = bpy.data.cameras.new(c["image"])
    cam.type = "PERSP"
    cam.sensor_fit = "HORIZONTAL"
    for key in ("lens", "sensor_width", "shift_x", "shift_y"):
        setattr(cam, key, settings[key])
    if data.get("bounds") and frame != "source":
        # Export units can be large; Blender's default 1000-unit far plane may hide the scene.
        bounds = data["bounds"]
        center = [values[12 + i] for i in range(3)]
        far = math.sqrt(sum(max(abs(bounds["min"][i] - center[i]), abs(bounds["max"][i] - center[i])) ** 2 for i in range(3)))
        cam.clip_end = max(cam.clip_end, 2.0 * far)
    obj = bpy.data.objects.new(c["image"], cam)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = basis @ pose @ cv_to_blender
    scene = bpy.context.scene
    scene.camera = obj
    scene.render.resolution_x, scene.render.resolution_y = c["width"], c["height"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = settings["pixel_aspect_x"], settings["pixel_aspect_y"]
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=Path, default=Path(__file__).with_name("cameras.json"))
    parser.add_argument("--frame", choices=("source", "scene", "spz"), default="source")
    parser.add_argument("--image", required=True, help="Exact capture image name")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    data = json.loads(args.cameras.read_text())
    c = next((c for c in data["capture_cameras"] if c["image"] == args.image), None)
    if c is None:
        raise ValueError("No capture camera for " + args.image)
    import_camera(c, data, args.frame)
    print("Exact capture camera:", args.image, "frame:", args.frame)


if __name__ == "__main__":
    main()
