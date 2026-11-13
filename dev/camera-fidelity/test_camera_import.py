"""CPU adapter checks; optional Blender run exercises its actual projection matrix.

python3 -m unittest discover -s dev/camera-fidelity -v
blender -b --python dev/camera-fidelity/test_camera_import.py
"""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("camera_import", ROOT / "crates/splat-webexport/src/import_cameras.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None

CAMERA = dict(model="PINHOLE", width=1180, height=784, fx=1658.415340332738,
              fy=1205.6835281040944, cx=571.25, cy=406.75)


class CameraImport(unittest.TestCase):
    def test_anisotropic_settings(self):
        for c in (CAMERA, dict(CAMERA, fx=800.0, fy=1400.0)):
            with self.subTest(fx=c["fx"], fy=c["fy"]):
                s = module.camera_settings(c)
                self.assertAlmostEqual(s["lens"] * c["width"] / s["sensor_width"], c["fx"])
                self.assertAlmostEqual(s["pixel_aspect_y"] / s["pixel_aspect_x"], c["fx"] / c["fy"])
                self.assertTrue(1 <= s["pixel_aspect_x"] <= 200)
                self.assertTrue(1 <= s["pixel_aspect_y"] <= 200)
                self.assertGreater(s["shift_x"], 0)
                self.assertGreater(s["shift_y"], 0)
        for fx, fy in ((1, 201), (201, 1)):
            with self.assertRaisesRegex(ValueError, "pixel aspect"):
                module.camera_settings(dict(CAMERA, fx=fx, fy=fy))

    def test_distortion_is_not_silently_discarded(self):
        with self.assertRaisesRegex(ValueError, "undistorted"):
            module.camera_settings(dict(CAMERA, model="OPENCV"))
        with self.assertRaises(ValueError):
            module.camera_settings(dict(CAMERA, fx=float("nan")))

    @unittest.skipUnless(bpy, "requires Blender; no GPU rendering needed")
    def test_blender_projection_matches_native_K(self):
        # Exercise the real adapter and read Blender's stored properties (RNA clamps them).
        # Passing raw settings directly to calc_matrix_camera would hide that clamping.
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        for intrinsics in (CAMERA, dict(CAMERA, fx=800.0, fy=1400.0)):
            with self.subTest(fx=intrinsics["fx"], fy=intrinsics["fy"]):
                c = dict(intrinsics, image="calibrated", source_c2w=identity, c2w=identity)
                o = module.import_camera(c, {}, "source")
                scene = bpy.context.scene
                render = scene.render
                self.assertAlmostEqual(render.pixel_aspect_y / render.pixel_aspect_x, c["fx"] / c["fy"], delta=1e-6)
                projection = o.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),
                    x=render.resolution_x, y=render.resolution_y,
                    scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y)
                for x, y, z in ((-.6, .3, 4), (.4, -.5, 3), (.7, .8, 5)):
                    q = projection @ Vector((x, -y, -z, 1))
                    pixel = ((q.x / q.w + 1) * render.resolution_x / 2,
                             (1 - q.y / q.w) * render.resolution_y / 2)
                    self.assertAlmostEqual(pixel[0], c["fx"] * x / z + c["cx"], delta=.001)
                    self.assertAlmostEqual(pixel[1], c["fy"] * y / z + c["cy"], delta=.001)
                camera_data = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                bpy.data.cameras.remove(camera_data)



if __name__ == "__main__":
    unittest.main(argv=[__file__])
