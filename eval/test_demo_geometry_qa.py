import unittest
from pathlib import Path

from demo_geometry_qa import native_job


class NativeResolutionTests(unittest.TestCase):
    def test_both_arms_keep_full_calibrated_source_size_and_same_cameras(self):
        names = [f'frame_{i:06}.jpg' for i in [100, 101, 103, 104, 130, 149, 169]]
        sizes = {name: [1180, 784] for name in names}
        original = native_job(Path('original.ply'), Path('dataset'), Path('original'), names, sizes)
        ablated = native_job(Path('ablated.ply'), Path('dataset'), Path('ablated'), names, sizes)
        self.assertEqual(original['max_resolution'], 1180)
        for key in ['dataset', 'max_resolution', 'names', 'orbit', 'depth']:
            self.assertEqual(original[key], ablated[key])
        # Match LoadImage's output_scale contract; a zero cap would erase the source dimensions.
        for width, height in sizes.values():
            cap = min(1, original['max_resolution'] / max(width, height))
            self.assertEqual((int(width * cap), int(height * cap)), (1180, 784))

    def test_mixed_camera_sizes_do_not_downscale_and_invalid_sizes_fail(self):
        names = ['landscape.jpg', 'portrait.jpg']
        sizes = {'landscape.jpg': [1180, 784], 'portrait.jpg': [1536, 2048]}
        job = native_job(Path('splat.ply'), Path('dataset'), Path('out'), names, sizes)
        self.assertEqual(job['max_resolution'], 2048)
        sizes['portrait.jpg'] = [0, 2048]
        with self.assertRaises(ValueError):
            native_job(Path('splat.ply'), Path('dataset'), Path('out'), names, sizes)
        with self.assertRaises(ValueError):
            native_job(Path('splat.ply'), Path('dataset'), Path('out'), [], {})


if __name__ == '__main__':
    unittest.main()
