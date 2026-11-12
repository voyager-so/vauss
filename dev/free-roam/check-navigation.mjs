// Walk / collision acceptance on the viewer's own JavaScript (viewer-harness.mjs runs it verbatim).
// node check-navigation.mjs [PAGE.html]: the template by default, else that exported / patched page.
// Requires THREE_MODULE=/path/to/exported/viewer/vendor/three.module.js or npm three.
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readPage, extract, loadViewer, setGrid, spawn, move } from './viewer-harness.mjs';
const file = process.argv[2];
const html = readPage(file);  // a named page that does not exist fails here, never falls back to the template
const THREE = await import(process.env.THREE_MODULE || 'three');
console.log('page:', file ?? 'template');
const V = (...a) => new THREE.Vector3(...a);
const room = (extra = {}, query = '') => loadViewer(html, THREE, { up: [0, 1, 0], metersPerUnit: 1, radius: 4,
  collide: { origin: [-1, -0.5, -1], edge: 0.05, dims: [160, 90, 160], radiusM: 0.25, stopM: 0.25 }, ...extra }, query);
const fill = (c, test) => setGrid(c, [-1, -0.5, -1], 0.05, [160, 90, 160], test);
const pos = (c) => c.camera.position;
const pen = (c) => c.run(`penetration(${pos(c).toArray().join(',')})`);

// Capsule against furniture, swept walls, sliding, no synthetic bounds, steps (exported floor metadata).
let c = room({ floorHeight: 0 });
fill(c, (x, y, z) => y < 0 || x > 5.9 || (x > 2 && x < 3 && z > 2 && z < 3 && y < 0.75));
spawn(c, [1, 1.5, 1]); for (let i = 0; i < 200; i++) move(c, [0.03, 0, 0]); assert(pos(c).x < 5.66); assert.equal(pen(c), 0);
const wallY = pos(c).y; for (let i = 0; i < 100; i++) move(c, [0.02, 0, 0.02]); assert(pos(c).z > 2.5); assert(Math.abs(pos(c).y - wallY) < 0.002, 'wall cannot be climbed');
spawn(c, [1, 1.5, 2.5]); for (let i = 0; i < 120; i++) move(c, [0.03, 0, 0]); assert(pos(c).x < 1.8, 'body catches table below eyes');
spawn(c, [0, 1.5, 1]); for (let i = 0; i < 120; i++) move(c, [-0.03, 0, 0]); assert(pos(c).x < -3, 'free space beyond the stored grid stays traversable');
fill(c, (x, y) => y < 0 || (x > 2 && x < 3 && y < 0.15)); spawn(c, [1, 1.5, 1]); for (let i = 0; i < 100; i++) move(c, [0.025, 0, 0]);
assert(pos(c).x > 3.2, '15 cm step is passable'); assert(pos(c).y < 1.52, 'descend after step');
fill(c, (x, y) => y < 0 || (x > 2 && x < 3 && y < 0.3)); spawn(c, [1, 1.5, 1]); for (let i = 0; i < 100; i++) move(c, [0.025, 0, 0]);
assert(pos(c).x < 2, '30 cm obstacle stops walk');
fill(c, (x, y) => y < 0 || x > 5.9); spawn(c, [6.02, 1.5, 1]); assert.equal(pen(c), 0, 'occupied spawn finds clear point');
const CO = c.run('CO'); CO.grid.fill(0); CO.grid[12345] = 1; assert.equal(c.run('cleanCollision(CO.grid)'), 1); assert.equal(CO.grid[12345], 0);
console.log('PASS: capsule/table, swept wall, wall sliding, no grid boundary, 15 cm step, 30 cm obstacle, spawn escape, isolated voxel removal');
fill(c, (x, y) => y < 0 || x > 5.9); spawn(c, [1, 1.5, 1]); move(c, [10, 0, 0]); assert(pos(c).x < 5.71, 'a delayed frame cannot tunnel'); assert.equal(pen(c), 0);

// Levelled axes: pushing into a tilted wall cannot turn horizontal input into ascent.
const t = 0.096, si = Math.sin(t), co = Math.cos(t), local = (x, y, z) => [co * x - si * y, si * x + co * y, z];
c = room({ floorHeight: 0, up: [si, co, 0] }); const up = V(si, co, 0);
fill(c, (x, y, z) => { const p = local(x, y, z); return p[1] < 0 || p[0] > 5.9; });
spawn(c, [co * 5.6 + si * 1.5, -si * 5.6 + co * 1.5, 1]); const startHeight = pos(c).dot(up);
for (let i = 0; i < 100; i++) move(c, [co * 0.02, -si * 0.02, 0.02]);
assert(pos(c).z > 2.5); assert(Math.abs(pos(c).dot(up) - startHeight) < 0.06, 'no climbing tilted wall');
c.run('walk.yaw = 0; walk.pitch = Math.PI / 2'); assert(c.run('walkDir(new THREE.Vector3())').dot(up) > 0.999999);
c.run('walk.pitch = -Math.PI / 2'); assert(c.run('walkDir(new THREE.Vector3())').dot(up) < -0.999999);
console.log('PASS: long-frame sweep, tilted wall, full upward/downward look');

// Floor without exported metadata (plain export-viewer CFG): high and low capture cameras stand on the floor.
for (const start of [2, 0.9, 4.2]) {
  c = room(); fill(c, (x, y) => y < 0);
  spawn(c, [1, start, 1]);
  assert(Math.abs(pos(c).y - 1.5) < 0.002, `camera at y=${start} stands at eye height (got ${pos(c).y})`); assert.equal(pen(c), 0);
  for (let i = 0; i < 60; i++) move(c, [0.02, 0, 0]);
  assert(Math.abs(pos(c).y - 1.5) < 0.002 && pos(c).x > 2, `camera from y=${start} keeps walking on the floor`); assert.equal(pen(c), 0);
}
// A wrong exported floor falls back to the support found below the eye.
c = room({ floorHeight: -0.6 }); fill(c, (x, y) => y < 0); spawn(c, [1, 0.9, 1]);
assert(Math.abs(pos(c).y - 1.5) < 0.002, 'misplaced floor metadata'); assert.equal(pen(c), 0);
console.log('PASS: floor found without metadata from high/low/very high cameras, wrong metadata recovered');

// Mode changes: startup keeps the named view's pivot; walk -> orbit retargets near; orbit -> walk lands on the floor.
const modeFns = ['clearNavigation', 'setMode'].map((n) => extract(html, n)).join('\n');
const withModes = (cfg, collision) => {
  const m = collision ? room(cfg) : loadViewer(html, THREE, { up: [0, 1, 0], metersPerUnit: 1, radius: 4, ...cfg });
  const controls = { enabled: false, target: m.camera.target, _sphericalDelta: V(), _panOffset: V(), _scale: 1 };
  Object.assign(m, { controls, velocity: V(), modeBtn: { classList: { toggle() {} } }, helpEl: {}, HELP: {}, touch() {}, WALK_OK: collision });
  vm.runInContext(modeFns, m);
  return m;
};
let o = withModes({}, false);
o.camera.position.set(0, 0, 10); o.camera.target.set(0, 0, 0);
o.setMode('orbit', true);
assert.deepEqual(o.camera.target.toArray(), [0, 0, 0], 'no-collision export keeps its look_at pivot at startup');
o = withModes({ floorHeight: 0 }, true); fill(o, (x, y) => y < 0);
o.run('mode = "orbit"'); o.camera.position.set(1, 0.4, 1); o.camera.target.set(1, 0.4, -9); o.setMode('orbit', true);
assert.deepEqual(o.camera.target.toArray(), [1, 0.4, -9], '?mode=orbit keeps its look_at pivot at startup');
o.run('mode = "walk"'); o.camera.position.set(1, 1.5, 1); o.camera.target.set(1, 1.5, -9); o.setMode('orbit');
assert(Math.abs(o.camera.target.distanceTo(o.camera.position) - 1.5) < 1e-9, 'leaving walk pivots 1.5 m ahead');
o.camera.position.set(1, 3, 1); o.setMode('walk'); o.run('walkCollide(.016)');
assert(Math.abs(pos(o).y - 1.5) < 0.002, `orbit -> walk stands on the floor (got ${pos(o).y})`);
console.log('PASS: startup pivot kept (no collision, ?mode=orbit), walk -> orbit pivot, orbit -> walk floor');

// Horizontal basis for every cardinal up and a tilted one; Z-up rooms walk horizontally.
for (const u of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0.3, 0.2, 0.93]]) {
  const b = loadViewer(html, THREE, { up: u, metersPerUnit: 1, radius: 4 });
  const [U, F, R] = ['LVL_U', 'LVL_F', 'LVL_R'].map((n) => b.run(n));
  for (const v of [F, R]) { assert(Math.abs(v.length() - 1) < 1e-9, `unit basis for up ${u}`); assert(Math.abs(v.dot(U)) < 1e-9, `horizontal basis for up ${u}`); }
  assert(Math.abs(F.dot(R)) < 1e-9);
}
c = room({ up: [0, 1, 0] }); assert.deepEqual([c.run('LVL_F').toArray(), c.run('LVL_R').toArray()].map((v) => v.map((x) => x + 0)), [[0, 0, -1], [1, 0, 0]], 'y-up rooms keep -Z ahead');
c = loadViewer(html, THREE, { up: [0, 0, 1], metersPerUnit: 1, radius: 4,
  collide: { origin: [-1, -1, -0.5], edge: 0.05, dims: [160, 160, 90], radiusM: 0.25, stopM: 0.25 } });
setGrid(c, [-1, -1, -0.5], 0.05, [160, 160, 90], (x, y, z) => z < 0);
spawn(c, [1, 1, 1.5]); const zStart = pos(c).clone(); const f = c.run('LVL_F');
for (let i = 0; i < 50; i++) move(c, f.clone().multiplyScalar(0.02).toArray());
assert(pos(c).clone().sub(zStart).length() > 0.9 && Math.abs(pos(c).z - 1.5) < 0.002, 'Z-up room walks horizontally at eye height');
console.log('PASS: orthonormal floor basis for +-X/Y/Z and tilted up; Z-up walking');
