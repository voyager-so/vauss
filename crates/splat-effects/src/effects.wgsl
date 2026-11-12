// Effect graph v1 evaluator: a 1:1 port of src/eval.rs (the CPU reference) over brush's buffers.
// One invocation per splat. Rest buffers are read-only in practice; outputs are transient.
//
// prog (f32 words, u32 fields bitcast), built by gpu.rs each frame:
//   [0] n_nodes u32  [1] seed u32  [2] h  [3] t (effective)  [4] cov mode u32 (0 exact, 1 fast)
//   [5] n_splats u32 [6] sh coeffs per channel u32  [7] duration
//   [8] floor normal xyz  [11] floor offset  [12] gravity  [13] aniso_max  [14] 0  [15] 0
//   nodes from 16, stride 64: [type u32, curve weight, mask offset u32, 0, params x60]
//     physical nodes keep their region at params[14..28]: lo, hi, centre, foot, hmin, hmax
//     crumble continues at params[28..51] (CRUMBLE_EXT): front dir xyz, front lo, front hi,
//       size_min, size_max, wind xyz, turbulence, turb_freq, haze, haze_size, haze_alpha, haze_s,
//       haze_rise, floor basis e1 xyz, e2 xyz, gather, dust_desat, heap_ao, grain_opacity,
//       front_noise, front_freq, chunk_frac, palette rgb, palette_tol, grain_jitter, land_shrink
//   mask programs (postfix), 9 words per instruction: [opcode u32, a0..a7]
//   shatter chunk tables, 12 words per chunk: [seed point xyz, ca.r, ca.axis xyz, cb.r, spin axis xyz, 0]

@group(0) @binding(0) var<storage, read_write> rest_tr: array<f32>;   // [N,10] means, quat wxyz, log scales
@group(0) @binding(1) var<storage, read_write> rest_op: array<f32>;   // [N] opacity logits
@group(0) @binding(2) var<storage, read_write> rest_sh: array<f32>;   // [N,C,3]
@group(0) @binding(3) var<storage, read_write> objects: array<f32>;   // [K,N] object weights
@group(0) @binding(4) var<storage, read_write> prog: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_tr: array<f32>;    // [N,10]
@group(0) @binding(6) var<storage, read_write> out_op: array<f32>;    // [N]
@group(0) @binding(7) var<storage, read_write> out_x: array<f32>;     // [N,6] dc rgb, weight, invalid, sh drop

const SH_C0: f32 = 0.2820948;
const MIN_SCALE: f32 = 1e-7;
const MAX_SCALE_FACTOR: f32 = 10.0;
const DROPPED_LOGIT: f32 = -30.0;
const TAU: f32 = 6.2831855;
const PI: f32 = 3.1415927;
const HEADER: u32 = 16u;
const NODE_STRIDE: u32 = 80u;
const REGION_AT: u32 = 14u;
const CRUMBLE_EXT: u32 = 28u;
const AUX: u32 = 6u;
const FLOAT_SHADOW_LIFT: f32 = 0.08;
const CHUNK_STRIDE: u32 = 12u;
const GRAIN_S: f32 = 0.2;
const DUST_S: f32 = 0.3;
const TURB_RAMP_S: f32 = 0.3;
const HAZE_DRIFT: f32 = 0.35;
const SHADOW_BELOW: f32 = 0.25;
const SHADOW_ABOVE: f32 = 0.1;
const SHADOW_LAG_S: f32 = 1.5;
const MOUND_SCATTER: f32 = 0.25;
const RIM_START: f32 = 0.8;
const RIM_WIDTH: f32 = 0.6;
const RIM_FADE: f32 = 0.7;
const HEAP_FILL_POW: f32 = 0.35;
const SPIN_DAMP_S: f32 = 0.25;
const RIPPLE_CUT: f32 = 16.0;

fn pf(k: u32) -> f32 { return prog[k]; }
fn pu(k: u32) -> u32 { return bitcast<u32>(prog[k]); }
fn pv(k: u32) -> vec3<f32> { return vec3<f32>(prog[k], prog[k + 1u], prog[k + 2u]); }

fn finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
fn finite3(v: vec3<f32>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z); }

fn pcg(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn unit(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }

struct Rnd { r: f32, r3: vec3<f32>, axis: vec3<f32> }

fn splat_randoms(i: u32, seed: u32) -> Rnd {
    var h = pcg(i ^ (seed * 0x9E3779B9u));
    let r = unit(h);
    h = pcg(h); let a = unit(h);
    h = pcg(h); let b = unit(h);
    h = pcg(h); let c = unit(h);
    h = pcg(h); let d = unit(h) * 2.0 - 1.0;
    h = pcg(h); let e = unit(h) * 2.0 - 1.0;
    h = pcg(h); let f = unit(h) * 2.0 - 1.0;
    return Rnd(r, vec3<f32>(a, b, c), vec3<f32>(d, e, f));
}

fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l > 0.0) {
        let r = 1.0 / l;
        if (finite(r)) { return v * r; }
    }
    return vec3<f32>(0.0);
}

fn ramp(d: f32, falloff: f32) -> f32 {
    if (falloff <= 0.0) { return select(0.0, 1.0, d <= 0.0); }
    return 1.0 - clamp(d / falloff, 0.0, 1.0);
}

fn smooth01(x0: f32) -> f32 {
    let x = clamp(x0, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

fn dissolve_progress(r: f32, softness: f32, w: f32) -> f32 {
    let c = softness + r * (1.0 - 2.0 * softness);
    if (softness <= 0.0) { return select(0.0, 1.0, w > c); }
    return smooth01((w - (c - softness)) / (2.0 * softness));
}

fn rotate(v: vec3<f32>, axis: vec3<f32>, theta: f32) -> vec3<f32> {
    let s = sin(theta);
    let c = cos(theta);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

fn axis_angle(a: vec3<f32>, angle: f32) -> mat3x3<f32> {
    let s = sin(angle);
    let c = cos(angle);
    let k = 1.0 - c;
    return mat3x3<f32>(
        vec3<f32>(a.x * a.x * k + c, a.x * a.y * k + a.z * s, a.x * a.z * k - a.y * s),
        vec3<f32>(a.x * a.y * k - a.z * s, a.y * a.y * k + c, a.y * a.z * k + a.x * s),
        vec3<f32>(a.x * a.z * k + a.y * s, a.y * a.z * k - a.x * s, a.z * a.z * k + c),
    );
}

// Unit quaternion (x, y, z, w) to a rotation matrix (glam Mat3::from_quat).
fn quat_to_mat(q: vec4<f32>) -> mat3x3<f32> {
    let x2 = q.x + q.x; let y2 = q.y + q.y; let z2 = q.z + q.z;
    let xx = q.x * x2; let xy = q.x * y2; let xz = q.x * z2;
    let yy = q.y * y2; let yz = q.y * z2; let zz = q.z * z2;
    let wx = q.w * x2; let wy = q.w * y2; let wz = q.w * z2;
    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

// Rotation matrix to quaternion (x, y, z, w) (glam Quat::from_mat3).
fn mat_to_quat(m: mat3x3<f32>) -> vec4<f32> {
    let m00 = m[0].x; let m01 = m[0].y; let m02 = m[0].z;
    let m10 = m[1].x; let m11 = m[1].y; let m12 = m[1].z;
    let m20 = m[2].x; let m21 = m[2].y; let m22 = m[2].z;
    if (m22 <= 0.0) {
        let dif10 = m11 - m00;
        let omm22 = 1.0 - m22;
        if (dif10 <= 0.0) {
            let f = omm22 - dif10; let inv = 0.5 / sqrt(f);
            return vec4<f32>(f * inv, (m01 + m10) * inv, (m02 + m20) * inv, (m12 - m21) * inv);
        }
        let f = omm22 + dif10; let inv = 0.5 / sqrt(f);
        return vec4<f32>((m01 + m10) * inv, f * inv, (m12 + m21) * inv, (m20 - m02) * inv);
    }
    let sum10 = m11 + m00;
    let opm22 = 1.0 + m22;
    if (sum10 <= 0.0) {
        let f = opm22 - sum10; let inv = 0.5 / sqrt(f);
        return vec4<f32>((m02 + m20) * inv, (m12 + m21) * inv, f * inv, (m01 - m10) * inv);
    }
    let f = opm22 + sum10; let inv = 0.5 / sqrt(f);
    return vec4<f32>((m12 - m21) * inv, (m20 - m02) * inv, (m01 - m10) * inv, f * inv);
}

// Hamilton product a * b, both (x, y, z, w).
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

// Quaternion (x, y, z, w) about the unit `axis` (glam Quat::from_axis_angle).
fn quat_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let h = angle * 0.5;
    return vec4<f32>(axis * sin(h), cos(h));
}

// q * v for a unit quaternion (glam Quat::mul_vec3).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    let b2 = dot(b, b);
    return v * (q.w * q.w - b2) + b * (dot(v, b) * 2.0) + cross(b, v) * (q.w * 2.0);
}

fn horiz(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> { return v - dot(v, n) * n; }

// Rotation taking unit a onto unit b (eval.rs rotation_arc).
fn rotation_arc(a: vec3<f32>, b: vec3<f32>) -> vec4<f32> {
    let q = vec4<f32>(cross(a, b), 1.0 + dot(a, b));
    if (dot(q, q) < 1e-20) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    return normalize(q);
}

struct Ballistic { pos: vec3<f32>, t_contact: f32, t_land: f32 }

// Fall to the floor (n.p = d, raised by rc), one bounce, slide to rest (eval.rs ballistic).
fn ballistic(p0: vec3<f32>, v0: vec3<f32>, tau: f32, n: vec3<f32>, d: f32, g: f32, rc: f32, e: f32, friction: f32) -> Ballistic {
    let y_raw = dot(p0, n) - d - rc;
    let y0 = max(y_raw, 0.0);
    let base = p0 - y_raw * n;
    let vy = dot(v0, n);
    let vh = v0 - vy * n;
    let tc = (vy + sqrt(vy * vy + 2.0 * g * y0)) / g;
    let vb = e * (g * tc - vy);
    let tb = 2.0 * vb / g;
    let t_land = tc + tb;
    var y: f32;
    var xh: vec3<f32>;
    if (tau <= tc) {
        y = y0 + vy * tau - 0.5 * g * tau * tau;
        xh = vh * tau;
    } else if (tau <= t_land) {
        let s = tau - tc;
        y = vb * s - 0.5 * g * s * s;
        xh = vh * tau;
    } else {
        let s0 = tau - t_land;
        let spd = length(vh);
        var slide = vec3<f32>(0.0);
        if (spd > 1e-9) {
            let stop = spd / (friction * g);
            let s = min(s0, stop);
            slide = vh / spd * (spd * s - 0.5 * friction * g * s * s);
        }
        y = 0.0;
        xh = vh * t_land + slide;
    }
    return Ballistic(base + y * n + xh, tc, t_land);
}

fn polar(j: mat3x3<f32>) -> mat3x3<f32> {
    if (abs(determinant(j)) < 1e-6) { return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)); }
    var r = j;
    for (var k = 0; k < 3; k++) {
        // R^-T = [b x c, c x a, a x b] / det for columns a, b, c.
        let inv_t = mat3x3<f32>(cross(r[1], r[2]), cross(r[2], r[0]), cross(r[0], r[1])) * (1.0 / determinant(r));
        r = (r + inv_t) * 0.5;
    }
    return r;
}

struct Eigen { lam: vec3<f32>, v: mat3x3<f32> }

// Cyclic Jacobi, 6 sweeps (eval.rs sym_eigen). a[col][row].
fn sym_eigen(m: mat3x3<f32>) -> Eigen {
    var a: array<array<f32, 3>, 3>;
    var v: array<array<f32, 3>, 3>; // row-major: v[row][col]
    for (var c = 0u; c < 3u; c++) {
        for (var r = 0u; r < 3u; r++) {
            a[c][r] = m[c][r];
            v[r][c] = select(0.0, 1.0, r == c);
        }
    }
    for (var sweep = 0; sweep < 6; sweep++) {
        for (var pair = 0u; pair < 3u; pair++) {
            let p = select(select(1u, 0u, pair < 2u), 0u, pair == 0u);
            let q = select(select(2u, 2u, pair == 1u), 1u, pair == 0u);
            let apq = a[q][p];
            if (abs(apq) < 1e-30) { continue; }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            var t: f32;
            if (abs(theta) > 1e15) {
                t = 0.5 / theta;
            } else {
                t = select(-1.0, 1.0, theta >= 0.0) / (abs(theta) + sqrt(theta * theta + 1.0));
            }
            let c = 1.0 / sqrt(t * t + 1.0);
            let s = t * c;
            for (var k = 0u; k < 3u; k++) {
                let akp = a[p][k];
                let akq = a[q][k];
                a[p][k] = c * akp - s * akq;
                a[q][k] = s * akp + c * akq;
            }
            for (var k = 0u; k < 3u; k++) {
                let apk = a[k][p];
                let aqk = a[k][q];
                a[k][p] = c * apk - s * aqk;
                a[k][q] = s * apk + c * aqk;
            }
            for (var k = 0u; k < 3u; k++) {
                let vp = v[k][p];
                let vq = v[k][q];
                v[k][p] = c * vp - s * vq;
                v[k][q] = s * vp + c * vq;
            }
        }
    }
    var vm = mat3x3<f32>(
        vec3<f32>(v[0][0], v[1][0], v[2][0]),
        vec3<f32>(v[0][1], v[1][1], v[2][1]),
        vec3<f32>(v[0][2], v[1][2], v[2][2]),
    );
    if (determinant(vm) < 0.0) { vm[2] = -vm[2]; }
    return Eigen(vec3<f32>(a[0][0], a[1][1], a[2][2]), vm);
}

fn mask_weight(off: u32, p: vec3<f32>, i: u32, n: u32) -> f32 {
    var stack: array<f32, 8>;
    var sp = 0u;
    var pc = off;
    loop {
        let op = pu(pc);
        if (op == 0u) { break; }
        if (op == 1u) {
            stack[sp] = 1.0; sp++;
        } else if (op == 2u) {
            let q = max(pv(pc + 1u) - p, vec3<f32>(0.0)) + max(p - pv(pc + 4u), vec3<f32>(0.0));
            stack[sp] = ramp(length(q), pf(pc + 7u)); sp++;
        } else if (op == 3u) {
            stack[sp] = ramp(max(length(p - pv(pc + 1u)) - pf(pc + 4u), 0.0), pf(pc + 5u)); sp++;
        } else if (op == 4u) {
            let h = dot(p, pv(pc + 1u));
            stack[sp] = ramp(max(max(pf(pc + 4u) - h, h - pf(pc + 5u)), 0.0), pf(pc + 6u)); sp++;
        } else if (op == 5u) {
            stack[sp] = clamp(objects[pu(pc + 1u) * n + i], 0.0, 1.0); sp++;
        } else if (op == 6u) {
            stack[sp - 1u] = 1.0 - stack[sp - 1u];
        } else if (op == 7u) {
            let k = pu(pc + 1u);
            var prod = 1.0;
            for (var j = 0u; j < k; j++) { sp--; prod *= stack[sp]; }
            stack[sp] = prod; sp++;
        }
        pc += 9u;
    }
    return stack[0];
}

// deform: a position-deforming classic node (or melt) is active, covariance from the FD
// Jacobian; else the exact rigid path. rigid / self_rot: accumulated physical rotations (x, y, z, w),
// self_rot being the part about the splat's own centre. Grain: s -> lerp(s, min(s, grain), grain_k),
// or lerp(s, grain, grain_k) when grain_iso (haze puffs grow). sh_drop: render the DC colour only.
struct Side {
    weight: f32,
    alpha: f32,
    shrink: f32,
    spin: mat3x3<f32>,
    color: vec3<f32>,
    deform: bool,
    rigid: vec4<f32>,
    self_rot: vec4<f32>,
    has_grain: bool,
    grain: f32,
    grain_k: f32,
    grain_iso: bool,
    sh_drop: bool,
}

// Nearest Voronoi seed of a shatter chunk table (ties -> lowest k). Starts from chunk 0 rather
// than +inf (a constant infinity is a WGSL creation error); same result for finite p.
fn nearest_seed(seeds: u32, table: u32, p: vec3<f32>) -> u32 {
    let d0 = pv(table) - p;
    var best_d = dot(d0, d0);
    var best = 0u;
    for (var k = 1u; k < seeds; k++) {
        let dv = pv(table + k * CHUNK_STRIDE) - p;
        let d = dot(dv, dv);
        if (d < best_d) {
            best_d = d;
            best = k;
        }
    }
    return best;
}

// Crumble (eval.rs physical, COp::Crumble). Params: start, sweep, burst, up_kick, grain,
// restitution, friction, settle, pile_h, pile_r, dust, reverse, jitter, shadow, region, then
// CRUMBLE_EXT. Members release along the front; non-members near the floor under the pile get
// the contact shadow.
fn crumble(b: u32, mu0: vec3<f32>, p: vec3<f32>, i: u32, rnd: Rnd, side: ptr<function, Side>) -> vec3<f32> {
    let a = b + 4u;
    let fnrm = pv(8u);
    let fd = pf(11u);
    let g = pf(12u);
    let foot = pv(a + REGION_AT + 9u);
    let start = pf(a);
    let sweep = pf(a + 1u);
    let pile_r = pf(a + 9u);
    var t = pf(3u);
    if (pu(a + 11u) != 0u) { t = pf(7u) - t; }
    if (mask_weight(pu(b + 2u), mu0, i, pu(5u)) < 0.5) {
        // Contact shadow: floor splats under the pile darken as it lands (fake AO).
        let shadow = pf(a + 13u);
        let y = dot(mu0, fnrm) - fd;
        let rho = length(horiz(mu0 - foot, fnrm));
        if (shadow <= 0.0 || !(y >= -SHADOW_BELOW && y <= SHADOW_ABOVE) || rho > 2.0 * pile_r) { return p; }
        let grow = smooth01((t - start) / (sweep + SHADOW_LAG_S));
        if (grow <= 0.0) { return p; }
        (*side).weight = max((*side).weight, grow);
        (*side).color *= 1.0 - shadow * grow * exp(-(rho * rho) / (pile_r * pile_r));
        return p;
    }
    let e = a + CRUMBLE_EXT;
    let lo = pf(e + 3u);
    let hi = pf(e + 4u);
    // A noisy, irregular front: release height perturbed by a smooth positional noise.
    let fq = pf(e + 28u);
    let nz = 0.6 * sin(fq * mu0.x + 1.7) * sin(fq * mu0.z + 0.4)
        + 0.4 * sin(2.7 * fq * mu0.y + 0.9) * sin(2.3 * fq * mu0.x + 2.2);
    let h = clamp((dot(mu0, pv(e)) - lo) / max(hi - lo, 1e-6) + pf(e + 27u) * nz, 0.0, 1.0);
    let ti = start + h * sweep + pf(a + 12u) * sweep * rnd.r;
    let tau = t - ti;
    if (tau <= 0.0) { return p; }
    // Released grains stay in the material's palette: within ±palette_tol of its mean colour.
    let pal = pv(e + 30u);
    let tol = pf(e + 33u);
    (*side).color = pal + clamp((*side).color - pal, vec3<f32>(-tol), vec3<f32>(tol));
    // Per-grain brightness jitter so a heap reads as granular stone, not clay.
    (*side).color *= 1.0 + pf(e + 34u) * (2.0 * rnd.r3.y - 1.0);
    let outward = normalize_or_zero(horiz(mu0 - foot, fnrm));
    let jit = horiz(rnd.r3 * 2.0 - vec3<f32>(1.0), fnrm);
    let v0 = pf(a + 2u) * (outward + 0.3 * jit) + pf(a + 3u) * (0.5 + rnd.r) * fnrm;
    let ua = 0.5 * rnd.axis.x + 0.5;
    let haze = 0.5 * rnd.axis.z + 0.5 < pf(e + 12u);
    // Horizontal curl-like turbulence: a deterministic sum of sines of rest position and time.
    let tf = pf(e + 11u);
    let turb = pf(e + 10u)
        * horiz(
            vec3<f32>(
                sin(tf * mu0.y + 1.3 * tau + TAU * rnd.r),
                sin(tf * mu0.z + 1.7 * tau + TAU * rnd.r3.x),
                sin(tf * mu0.x + 1.1 * tau + TAU * rnd.r3.y),
            ),
            fnrm,
        )
        * smooth01(tau / TURB_RAMP_S);
    let wind = pv(e + 7u);
    var rim = 0.0;
    var settled = 0.0;
    var heap_shade = 1.0;
    var pos: vec3<f32>;
    if (haze) {
        pos = mu0 + HAZE_DRIFT * v0 * tau + pf(e + 16u) * tau * fnrm + wind * tau + turb;
    } else {
        let bl = ballistic(mu0, v0, tau, fnrm, fd, g, 0.0, pf(a + 5u), pf(a + 6u));
        let air = 1.0 - smooth01((tau - bl.t_land) / TURB_RAMP_S);
        // After landing, grains settle into a low, irregular mound: the landing spot (which keeps
        // the fall footprint and wind drift) pulled toward the footprint centre by `gather`, plus
        // a small scatter; height is a Gaussian profile with positional ripples, larger flakes
        // resting higher; the rim thins out.
        let settle = smooth01((tau - bl.t_land) / pf(a + 7u));
        let ang = TAU * (0.5 * rnd.axis.y + 0.5);
        let e1 = pv(e + 17u);
        let e2 = pv(e + 20u);
        let floor_foot = foot + (fd - dot(foot, fnrm)) * fnrm;
        let at_land = ballistic(mu0, v0, bl.t_land, fnrm, fd, g, 0.0, pf(a + 5u), pf(a + 6u));
        let land = horiz(at_land.pos + wind * bl.t_land - floor_foot, fnrm);
        let off = land * (1.0 - pf(e + 23u)) + MOUND_SCATTER * pile_r * sqrt(rnd.r) * (cos(ang) * e1 + sin(ang) * e2);
        let rho = length(off);
        let irr = 0.6 + 0.4 * sin(7.3 * dot(off, e1) + 2.1) * sin(6.1 * dot(off, e2) + 0.7);
        // Height fraction r3.z^HEAP_FILL_POW packs grains toward the mound's surface (WGSL pow is
        // undefined at 0; Rust powf gives 0).
        let fill = select(pow(rnd.r3.z, HEAP_FILL_POW), 0.0, rnd.r3.z <= 0.0);
        let heap = pf(a + 8u) * exp(-(rho * rho) / (pile_r * pile_r)) * irr * (0.4 + 0.6 * ua) * fill;
        let heap_at = floor_foot + off + heap * fnrm;
        // Horizontally, grains curve toward their heap spot already while falling (so an overhang
        // lands on the base, not beside it); vertically they fall, then settle.
        let pull = max(settle, smooth01(tau / max(bl.t_land, 1e-3)));
        let fall = bl.pos + wind * min(tau, bl.t_land);
        let hf = horiz(fall, fnrm);
        let ht = horiz(heap_at, fnrm);
        pos = hf + (ht - hf) * pull + (dot(fall, fnrm) + (dot(heap_at, fnrm) - dot(fall, fnrm)) * settle) * fnrm + air * turb;
        rim = settle * smooth01((rho / pile_r - RIM_START) / RIM_WIDTH);
        settled = settle;
        // Self-shadowing: a grain's colour scales with its height fraction in the mound (fill),
        // darker underneath, full brightness on top.
        heap_shade = 1.0 - pf(e + 25u) * (1.0 - fill) * settle;
        // Landed grains turn to dust: desaturate toward their luminance, then darken.
        let landed = smooth01((tau - bl.t_land) / DUST_S);
        let luma = dot((*side).color, vec3<f32>(0.299, 0.587, 0.114));
        (*side).color = (*side).color + (vec3<f32>(luma) - (*side).color) * (pf(e + 24u) * landed);
        (*side).color *= 1.0 - (1.0 - pf(a + 10u)) * landed;
    }
    (*side).weight = 1.0;
    (*side).sh_drop = true;
    (*side).has_grain = true;
    // Flakes snap to grain size at release (no ease: an easing splat streaks); haze puffs grow
    // over GRAIN_S.
    if (haze) {
        (*side).alpha *= pf(e + 14u) * (1.0 - smooth01(tau / pf(e + 15u)));
        (*side).grain = pf(e + 13u);
        (*side).grain_k = smooth01(tau / GRAIN_S);
        (*side).grain_iso = true;
    } else {
        // Flakes are opaque stone: lift the rest opacity to at least grain_opacity.
        let a0 = 1.0 / (1.0 + exp(-rest_op[i]));
        (*side).alpha *= max(pf(e + 26u) / max(a0, 1e-6), 1.0);
        // The mound's rim thins out: smaller, fainter grains blend into the floor.
        (*side).alpha *= 1.0 - RIM_FADE * rim;
        (*side).color *= heap_shade;
        // A few chunky fragments among the grains (eval.rs CHUNK_SCALE = 3).
        let chunk = select(1.0, 3.0, 0.5 * rnd.axis.y + 0.5 < pf(e + 29u));
        (*side).grain = pf(a + 4u) * pf(e + 5u) * pow(pf(e + 6u) / pf(e + 5u), ua) * chunk * (1.0 - 0.5 * rim) * (1.0 - pf(e + 35u) * settled);
        (*side).grain_k = 1.0;
        (*side).grain_iso = false;
    }
    return p + (pos - mu0);
}

// A physical node (type 8..12): mask, release time and chunk from the rest mu0, a rigid motion
// (or melt's squash) of the current p. Not a member or not released: p unchanged, side untouched.
// A float's rigid motion at t (eval.rs float_state). Params: [a] start, rise, rise_s, spin, bob,
// bob_hz, has down, down_s, tilt, tilt axis (9..11); region (14..27); turn, its start and span,
// pivot offset (28..33); land_hard, shadow, shadow_soft, e1, e2, half extents (34..44).
struct FloatState { on: bool, lift: f32, lean: vec4<f32>, yaw: vec4<f32>, piv: vec3<f32> }

fn float_state(a: u32, t: f32, fnrm: vec3<f32>, centre: vec3<f32>) -> FloatState {
    let qid = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let tau = t - pf(a);
    if (tau <= 0.0) { return FloatState(false, 0.0, qid, qid, centre); }
    let rise_s = pf(a + 2u);
    // Up over rise_s, optionally back down from down_s (eased, or hard: x²); spin and bob stop once down.
    var down = 0.0;
    var ts = tau;
    if (pu(a + 6u) != 0u) {
        let d = pf(a + 7u);
        let x = clamp((t - d) / rise_s, 0.0, 1.0);
        down = smooth01(x) + pf(a + 34u) * (x * x - smooth01(x));
        ts = min(tau, d + rise_s - pf(a));
    }
    let k = smooth01(tau / rise_s) * (1.0 - down);
    let lift = k * (pf(a + 1u) + pf(a + 4u) * sin(TAU * pf(a + 5u) * tau));
    var lean = qid;
    let tilt = pf(a + 8u);
    if (tilt != 0.0) { lean = quat_axis_angle(pv(a + 9u), tilt * k); }
    let turn = pf(a + 28u) * smooth01((t - pf(a + 29u)) / pf(a + 30u));
    let yaw = quat_axis_angle(fnrm, pf(a + 3u) * ts * smooth01(ts / rise_s) + turn);
    return FloatState(true, lift, lean, yaw, centre + pv(a + 31u));
}

// The darkening a float casts on a non-member splat at mu0 (eval.rs float_shadow).
fn float_shadow(a: u32, fs: FloatState, mu0: vec3<f32>, fnrm: vec3<f32>, centre: vec3<f32>, hmin: f32) -> f32 {
    let h0 = hmin + fs.lift - dot(mu0, fnrm);
    if (h0 < -0.02) { return 0.0; }
    let h = max(h0, 0.0);
    let c = fs.piv + quat_rotate(fs.yaw, centre - fs.piv);
    let inv = vec4<f32>(-fs.yaw.xyz, fs.yaw.w);
    let local = quat_rotate(inv, horiz(mu0 - c, fnrm));
    let ex = pf(a + 43u);
    let ez = pf(a + 44u);
    let bl = 0.05 + pf(a + 36u) * h;
    let u = dot(local, pv(a + 37u)) / (ex + bl);
    let v = dot(local, pv(a + 40u)) / (ez + bl);
    let r2 = u * u + v * v;
    if (r2 > 9.0) { return 0.0; }
    let dark = pf(a + 35u) * (ex * ez / ((ex + bl) * (ez + bl))) * exp(-2.0 * r2) * smooth01(fs.lift / FLOAT_SHADOW_LIFT);
    return select(dark, 0.0, dark < 1e-4);
}

fn physical(ty: u32, b: u32, mu0: vec3<f32>, p: vec3<f32>, i: u32, rnd: Rnd, side: ptr<function, Side>) -> vec3<f32> {
    let t = pf(3u);
    let n = pu(5u);
    let fnrm = pv(8u);
    let fd = pf(11u);
    let g = pf(12u);
    let a = b + 4u;
    let centre = pv(a + REGION_AT + 6u);
    let foot = pv(a + REGION_AT + 9u);
    let hmin = pf(a + REGION_AT + 12u);
    let hmax = pf(a + REGION_AT + 13u);
    if (ty == 10u) { // ripple: source, start, speed, wavelength, amplitude, decay, packet_width, light, shade, rigid, has pivot, pivot
        let tau = t - pf(a + 3u);
        if (tau <= 0.0) { return p; }
        let w0 = mask_weight(pu(b + 2u), mu0, i, n);
        if (w0 == 0.0 || (pu(a + 13u) != 0u && w0 < 0.5)) { return p; }
        // Rigid objects sample the wave once, at their footprint centre (the pivot).
        let has_pivot = pu(a + 14u) != 0u;
        let pivot = pv(a + 15u);
        let w = select(w0, 1.0, has_pivot);
        let wavelength = pf(a + 5u);
        let decay = pf(a + 7u);
        let packet_width = pf(a + 8u);
        let xh = horiz(select(mu0, pivot, has_pivot) - pv(a), fnrm);
        let rho = length(xh);
        let front = rho - pf(a + 4u) * tau;
        let qd = front / packet_width;
        if (qd * qd > RIPPLE_CUT) { return p; }
        let env = w * pf(a + 6u) * exp(-decay * rho) * exp(-qd * qd);
        let ph = TAU * front / wavelength;
        let sn = sin(ph);
        let cs = cos(ph);
        let u = env * sn;
        let du = env * (sn * (-decay - 2.0 * qd / packet_width) + cs * TAU / wavelength);
        var rhat = vec3<f32>(0.0);
        if (rho > 1e-6) { rhat = xh / rho; }
        let n2 = normalize(fnrm - du * rhat);
        let q = rotation_arc(fnrm, n2);
        if (has_pivot) {
            (*side).weight = 1.0;
            (*side).rigid = quat_mul(q, (*side).rigid);
            return pivot + quat_rotate(q, p - pivot) + u * fnrm;
        }
        (*side).weight = max((*side).weight, w);
        (*side).rigid = quat_mul(q, (*side).rigid);
        (*side).self_rot = quat_mul(q, (*side).self_rot);
        (*side).color *= max(1.0 + pf(a + 12u) * dot(n2 - fnrm, pv(a + 9u)), 0.0);
        return p + u * fnrm;
    }
    if (ty == 8u) { return crumble(b, mu0, p, i, rnd, side); }
    if (mask_weight(pu(b + 2u), mu0, i, n) < 0.5) {
        if (ty == 11u && pf(a + 35u) > 0.0) {
            let fs = float_state(a, t, fnrm, centre);
            if (fs.on) {
                let dark = float_shadow(a, fs, mu0, fnrm, centre, hmin);
                if (dark > 0.0) {
                    (*side).weight = max((*side).weight, dark);
                    (*side).color *= 1.0 - dark;
                }
            }
        }
        return p;
    }
    if (ty == 9u) { // shatter: seeds, start, burst, up_kick, spin, restitution, friction, chunk radius, table
        let table = pu(a + 8u);
        let c = table + nearest_seed(pu(a), table, mu0) * CHUNK_STRIDE;
        let seed_k = pv(c);
        let tau = t - pf(a + 1u) - 0.1 * pf(c + 3u);
        if (tau <= 0.0) { return p; }
        let dir = normalize_or_zero(horiz(seed_k - foot, fnrm));
        let jit = horiz(pv(c + 4u), fnrm);
        let cb_r = pf(c + 7u);
        let v0 = pf(a + 2u) * (dir + 0.3 * jit) + pf(a + 3u) * (0.5 + cb_r) * fnrm;
        let bl = ballistic(seed_k, v0, tau, fnrm, fd, g, pf(a + 7u), pf(a + 5u), pf(a + 6u));
        let rate = pf(a + 4u) * (0.5 + cb_r);
        var theta: f32;
        if (tau <= bl.t_contact) {
            theta = rate * tau;
        } else {
            theta = rate * (bl.t_contact + SPIN_DAMP_S * (1.0 - exp(-(tau - bl.t_contact) / SPIN_DAMP_S)));
        }
        let q = quat_axis_angle(pv(c + 8u), theta);
        (*side).weight = 1.0;
        (*side).rigid = quat_mul(q, (*side).rigid);
        // No splat of a turned chunk sinks below the floor.
        let p1 = bl.pos + quat_rotate(q, p - seed_k);
        return p1 + max(fd - dot(p1, fnrm), 0.0) * fnrm;
    } else if (ty == 11u) { // float (members; the shadow on non-members is float_shadow)
        let fs = float_state(a, t, fnrm, centre);
        if (!fs.on) { return p; }
        (*side).weight = 1.0;
        (*side).rigid = quat_mul(fs.yaw, quat_mul(fs.lean, (*side).rigid));
        return fs.piv + quat_rotate(fs.yaw, centre + quat_rotate(fs.lean, p - centre) - fs.piv) + fs.lift * fnrm;
    }
    // melt: start, duration, spread, sheen
    let dur = pf(a + 1u);
    let h = clamp((dot(mu0, fnrm) - hmin) / max(hmax - hmin, 1e-6), 0.0, 1.0);
    let mm = smooth01((t - pf(a) - (1.0 - h) * 0.4 * dur) / (0.6 * dur));
    if (mm <= 0.0) { return p; }
    (*side).weight = 1.0;
    (*side).deform = true;
    (*side).color *= 1.0 + pf(a + 3u) * mm;
    let y = dot(p, fnrm) - fd;
    let keep = (1.0 - mm) * (1.0 - mm);
    let p1 = p + (keep - 1.0) * y * fnrm;
    return p1 + pf(a + 2u) * mm * horiz(p1 - foot, fnrm);
}

// F(mu): classic nodes mask at mu, physical nodes read the unperturbed rest mu0.
fn map_pos(mu: vec3<f32>, mu0: vec3<f32>, i: u32, rnd: Rnd, side: ptr<function, Side>) -> vec3<f32> {
    let n_nodes = pu(0u);
    let t = pf(3u);
    let n = pu(5u);
    var p = mu;
    for (var k = 0u; k < n_nodes; k++) {
        let b = HEADER + k * NODE_STRIDE;
        let cw = pf(b + 1u);
        if (cw == 0.0) { continue; }
        let ty = pu(b);
        if (ty >= 8u) {
            p = physical(ty, b, mu0, p, i, rnd, side);
            continue;
        }
        let w = mask_weight(pu(b + 2u), mu, i, n) * cw;
        if (w == 0.0) { continue; }
        (*side).weight = max((*side).weight, w);
        if (ty <= 5u) { (*side).deform = true; }
        let a = b + 4u;
        if (ty == 1u) { // vortex: center, axis, angle, angle_per_unit, lift, expand
            let center = pv(a);
            let axis = pv(a + 3u);
            var d = p - center;
            let theta = w * (pf(a + 6u) + pf(a + 7u) * dot(d, axis));
            d = rotate(d, axis, theta);
            let perp = d - dot(d, axis) * axis;
            let len = length(perp);
            var radial = vec3<f32>(0.0);
            if (len > 1e-9) { radial = perp / len; }
            p = center + d + w * pf(a + 8u) * axis + w * pf(a + 9u) * radial;
        } else if (ty == 2u) { // wave: direction, wavelength, speed, amplitude, axis
            let phase = TAU * (dot(pv(a), p) - pf(a + 4u) * t) / pf(a + 3u);
            p += w * pf(a + 5u) * sin(phase) * pv(a + 6u);
        } else if (ty == 3u) { // explode: center, distance, spin, jitter
            let r3 = rnd.r3 - vec3<f32>(0.5);
            let dir = normalize_or_zero(p - pv(a) + pf(a + 5u) * r3);
            p += w * pf(a + 3u) * (0.5 + rnd.r) * dir;
            var ax = rnd.axis;
            if (dot(ax, ax) > 1e-12) { ax = normalize(ax); } else { ax = vec3<f32>(0.0, 1.0, 0.0); }
            (*side).spin = axis_angle(ax, w * pf(a + 4u) * (rnd.r - 0.5)) * (*side).spin;
        } else if (ty == 4u) { // dissolve: direction, drift, softness, shrink
            let e = dissolve_progress(rnd.r, pf(a + 4u), w);
            p += e * pf(a + 3u) * pv(a);
            (*side).alpha *= 1.0 - e;
            (*side).shrink *= 1.0 - pf(a + 5u) * e;
        } else if (ty == 5u) { // morph: target, center, radius, axis
            let tgt = pu(a);
            let center = pv(a + 1u);
            let radius = pf(a + 4u);
            let axis = pv(a + 5u);
            let d = p - center;
            var proj: vec3<f32>;
            if (tgt == 0u) {
                proj = center + radius * normalize_or_zero(d);
            } else if (tgt == 1u) {
                let h = dot(d, axis);
                proj = center + h * axis + radius * normalize_or_zero(d - h * axis);
            } else {
                proj = p - dot(d, axis) * axis;
            }
            p = p + (proj - p) * w;
        } else if (ty == 6u) { // opacity: to
            (*side).alpha *= 1.0 - w + w * pf(a);
        } else if (ty == 7u) { // tint: color, amount
            (*side).color *= vec3<f32>(1.0) + w * pf(a + 3u) * (pv(a) - vec3<f32>(1.0));
        }
    }
    return p;
}

fn scale_alpha(logit: f32, m: f32) -> f32 {
    if (m == 1.0) { return logit; }
    let a = clamp(m / (1.0 + exp(-logit)), 1e-12, 1.0 - 1e-7);
    return log(a / (1.0 - a));
}

fn copy_rest(i: u32, weight: f32, invalid: bool, nc: u32) {
    for (var k = 0u; k < 10u; k++) { out_tr[i * 10u + k] = rest_tr[i * 10u + k]; }
    out_op[i] = select(rest_op[i], DROPPED_LOGIT, invalid);
    let sh = i * nc * 3u;
    out_x[i * AUX] = rest_sh[sh];
    out_x[i * AUX + 1u] = rest_sh[sh + 1u];
    out_x[i * AUX + 2u] = rest_sh[sh + 2u];
    out_x[i * AUX + 3u] = weight;
    out_x[i * AUX + 4u] = select(0.0, 1.0, invalid);
    out_x[i * AUX + 5u] = 0.0;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    let n = pu(5u);
    if (i >= n) { return; }
    let nc = pu(6u);
    let rnd = splat_randoms(i, pu(1u));
    let mu = vec3<f32>(rest_tr[i * 10u], rest_tr[i * 10u + 1u], rest_tr[i * 10u + 2u]);
    let dc = vec3<f32>(rest_sh[i * nc * 3u], rest_sh[i * nc * 3u + 1u], rest_sh[i * nc * 3u + 2u]);
    let ident = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    let qid = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var side = Side(0.0, 1.0, 1.0, ident, vec3<f32>(0.5) + SH_C0 * dc, false, qid, qid, false, 0.0, 0.0, false, false);
    let p = map_pos(mu, mu, i, rnd, &side);
    if (side.weight == 0.0) {
        copy_rest(i, 0.0, false, nc);
        return;
    }

    // Rest quaternion is stored w, x, y, z; work in (x, y, z, w).
    let q = normalize(vec4<f32>(rest_tr[i * 10u + 4u], rest_tr[i * 10u + 5u], rest_tr[i * 10u + 6u], rest_tr[i * 10u + 3u]));
    let s = exp(vec3<f32>(rest_tr[i * 10u + 7u], rest_tr[i * 10u + 8u], rest_tr[i * 10u + 9u]));
    let s_max = MAX_SCALE_FACTOR * max(s.x, max(s.y, s.z));
    var q2: vec4<f32>;
    var s2: vec3<f32>;
    if (!side.deform) {
        // Rigid only (physical nodes, opacity, tint): exact rotation, no Jacobian.
        q2 = normalize(quat_mul(side.rigid, q));
        s2 = s;
    } else {
        let h = pf(2u);
        var scratch = side;
        let cx = (map_pos(mu + vec3<f32>(h, 0.0, 0.0), mu, i, rnd, &scratch) - map_pos(mu - vec3<f32>(h, 0.0, 0.0), mu, i, rnd, &scratch)) / (2.0 * h);
        let cy = (map_pos(mu + vec3<f32>(0.0, h, 0.0), mu, i, rnd, &scratch) - map_pos(mu - vec3<f32>(0.0, h, 0.0), mu, i, rnd, &scratch)) / (2.0 * h);
        let cz = (map_pos(mu + vec3<f32>(0.0, 0.0, h), mu, i, rnd, &scratch) - map_pos(mu - vec3<f32>(0.0, 0.0, h), mu, i, rnd, &scratch)) / (2.0 * h);
        let jac = quat_to_mat(side.self_rot) * side.spin * mat3x3<f32>(cx, cy, cz);
        let r = quat_to_mat(q);
        if (pu(4u) == 0u) {
            let m = jac * r * mat3x3<f32>(vec3<f32>(s.x, 0.0, 0.0), vec3<f32>(0.0, s.y, 0.0), vec3<f32>(0.0, 0.0, s.z));
            let e = sym_eigen(m * transpose(m));
            q2 = normalize(mat_to_quat(e.v));
            s2 = sqrt(max(e.lam, vec3<f32>(0.0)));
        } else {
            let jr = jac * r;
            s2 = s * vec3<f32>(length(jr[0]), length(jr[1]), length(jr[2]));
            q2 = normalize(quat_mul(mat_to_quat(polar(jac)), q));
        }
    }
    s2 = s2 * side.shrink;
    if (side.has_grain) {
        if (side.grain_iso) {
            s2 = s2 + (vec3<f32>(side.grain) - s2) * side.grain_k;
        } else {
            // Flakes: capped to the grain size, then rounded (eval.rs FLAKE_ANISO = 1.6).
            var f = min(s2, vec3<f32>(side.grain));
            f = max(f, vec3<f32>(max(f.x, max(f.y, f.z)) / 1.6));
            s2 = s2 + (f - s2) * side.grain_k;
        }
    }
    // Anisotropy clamp: long axes <= max(rest ratio, 1) x aniso_max times the short one.
    let limit = max(max(s.x, max(s.y, s.z)) / max(min(s.x, min(s.y, s.z)), MIN_SCALE), 1.0) * pf(13u);
    // The short axis is floored at 0.1 x the rest short axis (eval.rs ANISO_BASE_FLOOR).
    let base = max(max(min(s2.x, min(s2.y, s2.z)), 0.1 * min(s.x, min(s.y, s.z))), MIN_SCALE);
    s2 = min(s2, vec3<f32>(base * limit));
    s2 = clamp(s2, vec3<f32>(MIN_SCALE), vec3<f32>(max(s_max, MIN_SCALE)));
    if (q2.w < 0.0) { q2 = -q2; }
    let dc2 = (side.color - vec3<f32>(0.5)) / SH_C0;
    let opac = scale_alpha(rest_op[i], side.alpha);
    let ok = finite3(p) && finite3(q2.xyz) && finite(q2.w) && finite3(s2) && finite3(dc2) && finite(opac);
    if (!ok) {
        copy_rest(i, side.weight, true, nc);
        return;
    }
    let ls = log(s2);
    out_tr[i * 10u] = p.x; out_tr[i * 10u + 1u] = p.y; out_tr[i * 10u + 2u] = p.z;
    out_tr[i * 10u + 3u] = q2.w; out_tr[i * 10u + 4u] = q2.x; out_tr[i * 10u + 5u] = q2.y; out_tr[i * 10u + 6u] = q2.z;
    out_tr[i * 10u + 7u] = ls.x; out_tr[i * 10u + 8u] = ls.y; out_tr[i * 10u + 9u] = ls.z;
    out_op[i] = opac;
    out_x[i * AUX] = dc2.x; out_x[i * AUX + 1u] = dc2.y; out_x[i * AUX + 2u] = dc2.z;
    out_x[i * AUX + 3u] = side.weight;
    out_x[i * AUX + 4u] = 0.0;
    out_x[i * AUX + 5u] = select(0.0, 1.0, side.sh_drop);
}
