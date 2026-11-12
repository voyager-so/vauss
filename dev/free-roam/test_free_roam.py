"""Free-walk packaging regressions (stdlib + NumPy; Node for the navigation check).

python3 -m unittest discover -s dev/free-roam -p 'test_*.py'
Set THREE_MODULE=/path/to/three.module.js (or install npm three) to include the walk checks.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = ROOT / 'crates/splat-webexport/src/viewer/index.html'
INJECTION = '</script><script>globalThis.pr8Injected=true</script><!--'
FIELDS = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2']


def write_ply(path, points, comments):
    """Binary little-endian PLY, header as splat-train-serde's exporter writes it."""
    header = ['ply', 'format binary_little_endian 1.0', *[f'comment {c}' for c in comments],
              f'element vertex {len(points)}', *[f'property float {f}' for f in FIELDS], 'end_header']
    data = np.zeros(len(points), dtype=[(f, '<f4') for f in FIELDS])
    data['x'], data['y'], data['z'] = points.T
    data['opacity'] = 4.0  # sigmoid ~0.98: opaque
    for k in ('scale_0', 'scale_1', 'scale_2'):
        data[k] = np.log(0.02)
    path.write_bytes(('\n'.join(header) + '\n').encode() + data.tobytes())


def room_points(up=(0, 1, 0), seed=0):
    """A floor slab and two walls in a frame whose vertical is `up`."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(-2, 2, 4000), rng.uniform(-0.05, 0, 4000), rng.uniform(-2, 2, 4000)])
    wall = np.column_stack([np.full(2000, 2.0), rng.uniform(0, 2.5, 2000), rng.uniform(-2, 2, 2000)])
    pts = np.vstack([floor, wall])
    u = np.asarray(up, float) / np.linalg.norm(up)
    if abs(u[1]) > 0.999:
        return pts * [1, np.sign(u[1]), 1]
    # Rotate +Y onto u (Rodrigues).
    k = np.cross([0, 1, 0], u); s = np.linalg.norm(k); c = u[1]; k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return pts @ (np.eye(3) + s * K + (1 - c) * K @ K).T


def run(*cmd, check=True):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=check)


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(); self.scripts = []; self._in = False

    def handle_starttag(self, tag, attrs):
        if tag == 'script': self.scripts.append(''); self._in = True

    def handle_endtag(self, tag):
        if tag == 'script': self._in = False

    def handle_data(self, data):
        if self._in: self.scripts[-1] += data


def scripts(text):
    p = Scripts(); p.feed(text); return p.scripts


def page_cfg(text):
    return json.loads(re.search(r'^const CFG = (.*);$', text, re.M)[1])


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='free-roam-test-'))
        self.addCleanup(shutil.rmtree, self.tmp)


class RawScene(Tmp):
    def scene(self, axis_comment, up=(0, 1, 0)):
        ply = self.tmp / 'model.ply'
        comments = ['Exported from Brush'] + ([f'Vertical axis: {axis_comment}'] if axis_comment is not None else []) + ['SplatRenderMode: default']
        write_ply(ply, room_points(up), comments)
        out = self.tmp / 'scene.json'
        run(sys.executable, HERE / 'raw-scene.py', ply, out)
        return json.loads(out.read_text())

    def assert_level_start(self, s):
        u = np.array(s['up']); cam = s['start_camera']
        f = np.array(cam['look_at']) - np.array(cam['position'])
        self.assertAlmostEqual(np.linalg.norm(u), 1, places=9)
        self.assertAlmostEqual(np.linalg.norm(f), 1, places=6)
        self.assertAlmostEqual(float(f @ u), 0, places=6)

    def test_default_exporter_header(self):
        # splat_to_ply(..., None) writes "Vertical axis: y"; the importer reads it as -Y.
        s = self.scene('y', up=(0, -1, 0))
        self.assertEqual(s['up'], [0.0, -1.0, 0.0])
        self.assert_level_start(s)

    def test_symbolic_and_numeric_axes(self):
        for comment, want in [('x', [1, 0, 0]), ('z', [0, 0, -1]), ('Y', [0, -1, 0]), ('0 0 1', [0, 0, 1]),
                              ('[0, 2, 0]', [0, 1, 0]), ('-1 0 0', [-1, 0, 0]), ('0.1 0.98 0.17', None), (None, [0, 1, 0])]:
            with self.subTest(comment=comment):
                v = want or [0.1, 0.98, 0.17]
                s = self.scene(comment, up=v)
                np.testing.assert_allclose(s['up'], np.array(v) / np.linalg.norm(v), atol=1e-9)
                self.assert_level_start(s)
                self.assertNotIn(str(self.tmp), json.dumps(s))

    def test_rejects_degenerate_axes(self):
        for comment in ('0 0 0', 'nan 1 0', 'inf 0 0'):
            with self.subTest(comment=comment), self.assertRaises(subprocess.CalledProcessError) as e:
                self.scene(comment)
            self.assertIn('finite nonzero', e.exception.stderr)


class InlineScripts(Tmp):
    """collider.py and patch-viewer-roam.py must keep every generated page's scripts closed."""

    def package(self):
        pkg = self.tmp / 'viewer'; (pkg / 'data').mkdir(parents=True)
        cfg = {'title': INJECTION, 'up': [0, 1, 0], 'radius': 2, 'metersPerUnit': 1, 'variants': [], 'budget': 1,
               'cameras': [{'label': 'Start view', 'position': [0, 1.5, 0], 'look_at': [0, 1.5, -1]}]}
        # Exactly what the Rust exporter writes (viewer.rs script_json / html_escape).
        page = (TEMPLATE.read_text().replace('__TITLE__', '&lt;/script&gt;')
                .replace('__CFG__', json.dumps(cfg, separators=(',', ':')).replace('<', '\\u003c'))
                .replace('__THREE_ADDONS__', 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/')
                .replace('__THREE__', 'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js'))
        (pkg / 'index.html').write_text(page)
        (pkg / 'build.json').write_text('{}\n')
        ply = self.tmp / 'model.ply'; write_ply(ply, room_points(), ['Vertical axis: 0 1 0'])
        scene = self.tmp / 'scene.json'
        scene.write_text(json.dumps({'meters_per_unit': 1, 'up': [0, 1, 0], 'floor': {'y': 0.0, 'detected': True,
                                     'extent': {'min': [-2, -2], 'max': [2, 2]}}}))
        return pkg, ply, scene

    def assert_safe(self, path):
        text = path.read_text()
        n = len(re.findall(r'<script\b', TEMPLATE.read_text()))
        self.assertEqual(len(re.findall(r'<script\b', text)), n, f'{path.name}: extra script element')
        self.assertEqual(len(scripts(text)), n)
        cfg = re.search(r'^const CFG = (.*);$', text, re.M)[1]
        self.assertNotIn('<', cfg)
        self.assertEqual(page_cfg(text)['title'], INJECTION)
        importmap = re.search(r'<script type="importmap">(.*?)</script>', text, re.S)[1]
        self.assertNotIn('<', importmap)

    def test_collider_then_patch(self):
        pkg, ply, scene = self.package()
        run(sys.executable, HERE / 'collider.py', ply, scene, pkg)
        self.assert_safe(pkg / 'index.html')
        self.assertEqual(page_cfg((pkg / 'index.html').read_text())['floorHeight'], 0.0)
        run(sys.executable, ROOT / 'dev/patch-viewer-roam.py', pkg, '--no-backup')
        self.assertEqual(sorted(p.name for p in pkg.glob('*.html')), ['index.html'], 'fresh packages serve no backup page')
        self.assert_safe(pkg / 'index.html')

    def test_backups_stay_outside_the_served_folder(self):
        pkg, ply, scene = self.package()
        run(sys.executable, HERE / 'collider.py', ply, scene, pkg)
        # An earlier version of the patch script left its backup inside the package.
        (pkg / 'index.before-roam.html').write_text('<script>globalThis.pr8Injected=true</script>')
        run(sys.executable, ROOT / 'dev/patch-viewer-roam.py', pkg)
        self.assertEqual(sorted(p.name for p in pkg.glob('*.html')), ['index.html'])
        self.assertTrue((self.tmp / 'viewer.before-roam.html').is_file())
        inside = run(sys.executable, ROOT / 'dev/patch-viewer-roam.py', pkg, '--backup', pkg / 'old.html', check=False)
        self.assertNotEqual(inside.returncode, 0)
        self.assertFalse((pkg / 'old.html').exists())


class Provenance(Tmp):
    def test_no_local_paths_served(self):
        home = '/home/someone/private-room'
        pkg = self.tmp / 'viewer'; pkg.mkdir()
        cfg = {'title': INJECTION, 'collide': {'source': f'game export {home}/exports/game (5 cm)'}, 'variants': [{'src': 'data/first.js'}],
               'cameras': [{'label': 'Capture 2 (frame_000151.jpg)'}]}
        (pkg / 'index.html').write_text('<script type="importmap">{"imports": {"three": "./vendor/three.module.js"}}</script>\n'
                                        f'const CFG = {json.dumps(cfg)};\n')
        (pkg / 'build.json').write_text(json.dumps({'source': f'{home}/exports/web', 'nested': [{'p': 'C:\\Users\\a\\b.ply'}],
                                                    'url': 'https://cdn.jsdelivr.net/npm/three@0.180.0/x.js'}))
        ply = self.tmp / 'model.ply'; write_ply(ply, room_points(), [])
        private = self.tmp / 'private' / 'provenance.json'
        run(sys.executable, HERE / 'provenance.py', pkg, ply, '--web', self.tmp / 'web', '--private', private)
        served = ''.join(p.read_text() for p in pkg.rglob('*') if p.is_file())
        for leak in ('/home/', 'someone', 'private-room', 'Users', str(self.tmp)):
            self.assertNotIn(leak, served)
        self.assertEqual(sorted(json.loads((pkg / 'package-source.json').read_text())), ['coordinate_note', 'ply_bytes', 'ply_sha256'])
        cfg = page_cfg((pkg / 'index.html').read_text())
        self.assertEqual(cfg['title'], INJECTION, 'markup is not a path')
        self.assertNotIn('<', re.search(r'^const CFG = (.*);$', (pkg / 'index.html').read_text(), re.M)[1])
        self.assertEqual(cfg['variants'], [{'src': 'data/first.js'}])
        self.assertEqual(cfg['cameras'], [{'label': 'Capture 2 (frame_000151.jpg)'}])
        self.assertIn('https://cdn.jsdelivr.net', (pkg / 'build.json').read_text())
        self.assertEqual(json.loads(private.read_text())['ply'], str(ply))
        inside = run(sys.executable, HERE / 'provenance.py', pkg, ply, '--private', pkg / 'p.json', check=False)
        self.assertNotEqual(inside.returncode, 0)


class Navigation(unittest.TestCase):
    def test_missing_page_fails(self):
        r = run('node', HERE / 'check-navigation.mjs', '/definitely/missing/viewer.html', check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('viewer page not found', r.stderr)

    def test_template_walk(self):
        if not os.environ.get('THREE_MODULE') and run('node', '-e', "import('three')", check=False).returncode:
            self.skipTest('THREE_MODULE not set and npm three not installed')
        r = run('node', ROOT / 'scripts/accept/viewer-roam.js', check=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count('PASS'), 5, r.stdout)


if shutil.which('node') is None:
    del Navigation

if __name__ == '__main__':
    unittest.main()
