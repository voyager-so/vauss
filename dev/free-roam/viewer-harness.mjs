// Runs the walk / collision code of an exported viewer page (or the template) verbatim in Node.
// The block from `const _a = new THREE.Vector3()` up to `function applyCamera(` is evaluated in a VM
// context with the page's own CFG; run(code) evaluates code inside that scope (CO, walk, mode, ...).
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

export const TEMPLATE = new URL('../../crates/splat-webexport/src/viewer/index.html', import.meta.url);

export function readPage(file) {
  const path = file ?? TEMPLATE;
  assert(fs.existsSync(path), `viewer page not found: ${file}`);
  return fs.readFileSync(path, 'utf8');
}

// The exported CFG object literal (a JSON line in every package; the template has the __CFG__ placeholder).
export function pageConfig(html) {
  const m = html.match(/^const CFG = (.*);$/m);
  assert(m, 'viewer CFG missing');
  return m[1] === '__CFG__' ? null : JSON.parse(m[1]);
}

export function extract(html, name) {
  const a = html.indexOf(`function ${name}(`);
  assert(a >= 0, `function ${name} missing`);
  let b = html.indexOf('{', a), n = 1;
  while (n && ++b < html.length) { if (html[b] === '{') n++; if (html[b] === '}') n--; }
  return html.slice(a, b + 1);
}

function walkBlock(html) {
  const a = html.indexOf('const _a = new THREE.Vector3()'), b = html.indexOf('function applyCamera(');
  assert(a >= 0 && b > a, 'walk / collision block missing');
  return html.slice(a, b);
}

// A page instance: cfg is the CFG object (cfg.collide must describe the grid for collision); query the URL flags.
export function loadViewer(html, THREE, cfg, query = '') {
  const camera = {
    position: new THREE.Vector3(), up: new THREE.Vector3(...(cfg.up ?? [0, 1, 0])).normalize(), target: new THREE.Vector3(),
    getWorldDirection(v) { return v.copy(this.target).sub(this.position).normalize(); },
  };
  const ctx = vm.createContext({ THREE, CFG: cfg, Q: new URLSearchParams(query), BENCH: false, MPU: cfg.metersPerUnit || 1,
    camera, state: {}, console, dataBytes: () => new Promise(() => {}) });
  vm.runInContext(walkBlock(html) + '\nglobalThis.run = (code) => eval(code);', ctx);
  return ctx;
}

// Grid helpers: fill the context's CO grid from a predicate over cell centres; spawn / move like the page loop.
export function setGrid(ctx, o, e, n, test) {
  const CO = ctx.run('CO');
  Object.assign(CO, { o, e, n, grid: new Uint8Array(n[0] * n[1] * n[2]) });
  for (let z = 0; z < n[2]; z++) for (let y = 0; y < n[1]; y++) for (let x = 0; x < n[0]; x++) {
    const p = [x, y, z].map((v, i) => o[i] + (v + 0.5) * e);
    CO.grid[(z * n[1] + y) * n[0] + x] = test(...p) ? 1 : 0;
  }
  return CO;
}
export function spawn(ctx, p) { ctx.camera.position.set(...p); ctx.run('CO').jump = true; ctx.run('walkCollide(.016)'); }
export function move(ctx, d) { ctx.camera.position.add(new ctx.THREE.Vector3(...d)); ctx.run('walkCollide(.016)'); }
