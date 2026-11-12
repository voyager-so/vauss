//! Per-photo continuous exposure trajectories (B1c, after CoMoGaussian).
//!
//! Each training photo owns a small network that maps a learned embedding and
//! its original camera pose to two 64-wide latents, integrates each through a
//! time-conditioned ODE (classical RK4) to nine exposure times, and decodes
//! every state into a rigid screw motion composed with a near-identity affine
//! refinement. A blurred photo is rendered from `c2w @ transform_k`.
//!
//! The networks are tiny (~23k weights per photo) and only one photo is
//! touched per step, so they live on the host with hand-written reverse-mode
//! gradients (computed in f64) and a lazy per-view Adam, which is exactly
//! torch's per-parameter Adam when unvisited views receive no gradient.

use glam::Mat4;

const LATENT: usize = 64;
const HALF: usize = LATENT / 2;
const CODE: usize = 8;
/// Exposure sub-poses per photo (times 0..=8).
pub const SUB_POSES: usize = 9;
/// Index of the unshifted (encoder output) state among the nine.
const MIDDLE: usize = SUB_POSES / 2;
const FEATURE: usize = 2 * LATENT;

const BETA_1: f32 = 0.9;
const BETA_2: f32 = 0.999;
const EPSILON: f32 = 1e-8;

type Latent = [f64; LATENT];
type M3 = [[f64; 3]; 3];

/// A dense layer whose row-major `out x inp` weight, then bias, sit at `off`
/// in a view's flat parameter vector.
#[derive(Clone, Copy)]
struct Linear {
    off: usize,
    out: usize,
    inp: usize,
}

impl Linear {
    const fn new(off: usize, inp: usize, out: usize) -> Self {
        Self { off, out, inp }
    }

    const fn end(self) -> usize {
        self.off + self.out * (self.inp + 1)
    }

    fn forward(self, p: &[f32], x: &[f64], y: &mut [f64]) {
        let (w, b) = p[self.off..self.end()].split_at(self.out * self.inp);
        for (o, y) in y[..self.out].iter_mut().enumerate() {
            let row = &w[o * self.inp..][..self.inp];
            *y = f64::from(b[o]) + row.iter().zip(x).map(|(w, x)| f64::from(*w) * x).sum::<f64>();
        }
    }

    /// Accumulates the weight and bias gradients into `g` and `dx`.
    fn backward(self, p: &[f32], g: &mut [f64], x: &[f64], dy: &[f64], dx: &mut [f64]) {
        let w = &p[self.off..self.off + self.out * self.inp];
        let (gw, gb) = g[self.off..self.end()].split_at_mut(self.out * self.inp);
        for (o, &dy) in dy[..self.out].iter().enumerate() {
            gb[o] += dy;
            let row = &w[o * self.inp..][..self.inp];
            let grow = &mut gw[o * self.inp..][..self.inp];
            for i in 0..self.inp {
                grow[i] += dy * x[i];
                dx[i] += dy * f64::from(row[i]);
            }
        }
    }
}

/// `Dynamics`: a `[9, 8]` time-code table, then one layer per latent half.
#[derive(Clone, Copy)]
struct Flow {
    time: usize,
    rotation: Linear,
    translation: Linear,
}

/// Stage inputs of one RK4 step, kept for the adjoint.
struct Advance {
    stages: [Latent; 4],
    time: f64,
    dt: f64,
}

/// Python `int(time)` (truncation; times are non-negative), clamped to the table.
fn code(time: f64) -> usize {
    (time as i64).clamp(0, SUB_POSES as i64 - 1) as usize
}

fn axpy(a: &Latent, s: f64, b: &Latent) -> Latent {
    std::array::from_fn(|i| a[i] + s * b[i])
}

impl Flow {
    const fn new(off: usize) -> Self {
        let rotation = Linear::new(off + SUB_POSES * CODE, HALF + CODE, HALF);
        Self { time: off, rotation, translation: Linear::new(rotation.end(), HALF + CODE, HALF) }
    }

    const fn end(self) -> usize {
        self.translation.end()
    }

    /// Input of half `h`: `cat(relu(value[h]), time code)`.
    fn input(self, p: &[f32], value: &Latent, time: f64, h: usize) -> [f64; HALF + CODE] {
        let c = self.time + code(time) * CODE;
        std::array::from_fn(|i| {
            if i < HALF { value[h * HALF + i].max(0.0) } else { f64::from(p[c + i - HALF]) }
        })
    }

    fn eval(self, p: &[f32], value: &Latent, time: f64) -> Latent {
        let mut out = [0.0; LATENT];
        for (h, layer) in [self.rotation, self.translation].into_iter().enumerate() {
            layer.forward(p, &self.input(p, value, time, h), &mut out[h * HALF..]);
        }
        out
    }

    fn eval_backward(self, p: &[f32], g: &mut [f64], value: &Latent, time: f64, dout: &Latent) -> Latent {
        let c = self.time + code(time) * CODE;
        let mut dvalue = [0.0; LATENT];
        for (h, layer) in [self.rotation, self.translation].into_iter().enumerate() {
            let mut dx = [0.0; HALF + CODE];
            layer.backward(p, g, &self.input(p, value, time, h), &dout[h * HALF..], &mut dx);
            for i in 0..HALF {
                if value[h * HALF + i] > 0.0 {
                    dvalue[h * HALF + i] = dx[i];
                }
            }
            for i in 0..CODE {
                g[c + i] += dx[HALF + i];
            }
        }
        dvalue
    }

    fn advance(self, p: &[f32], value: &Latent, time: f64, dt: f64) -> (Latent, Advance) {
        let k1 = self.eval(p, value, time);
        let v2 = axpy(value, dt / 2.0, &k1);
        let k2 = self.eval(p, &v2, time + dt / 2.0);
        let v3 = axpy(value, dt / 2.0, &k2);
        let k3 = self.eval(p, &v3, time + dt / 2.0);
        let v4 = axpy(value, dt, &k3);
        let k4 = self.eval(p, &v4, time + dt);
        let next = std::array::from_fn(|i| value[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0);
        (next, Advance { stages: [*value, v2, v3, v4], time, dt })
    }

    /// RK4 adjoint: walks the stages back, each stage input feeding the
    /// previous stage's slope.
    fn advance_backward(self, p: &[f32], g: &mut [f64], a: &Advance, dnext: &Latent) -> Latent {
        let (time, dt) = (a.time, a.dt);
        let mut dvalue = *dnext;
        let mut dk = [dt / 6.0, dt / 3.0, dt / 3.0, dt / 6.0].map(|c| dnext.map(|d| c * d));
        for s in (0..4).rev() {
            let (t, coefficient) = match s {
                0 => (time, 0.0),
                3 => (time + dt, dt),
                _ => (time + dt / 2.0, dt / 2.0),
            };
            let dstage = self.eval_backward(p, g, &a.stages[s], t, &dk[s]);
            for i in 0..LATENT {
                dvalue[i] += dstage[i];
            }
            if s > 0 {
                dk[s - 1] = axpy(&dk[s - 1], coefficient, &dstage);
            }
        }
        dvalue
    }

    /// Nine states at times 0..=8: four steps back and four forward from the
    /// encoder output at time 4.
    fn states(self, p: &[f32], initial: &Latent) -> ([Latent; SUB_POSES], Vec<Advance>) {
        let mut states = [*initial; SUB_POSES];
        let mut tapes = Vec::with_capacity(SUB_POSES - 1);
        for dt in [-1.0, 1.0] {
            let mut value = *initial;
            for i in 1..=MIDDLE {
                let (next, tape) = self.advance(p, &value, MIDDLE as f64 + dt * (i - 1) as f64, dt);
                value = next;
                states[(MIDDLE as f64 + dt * i as f64) as usize] = value;
                tapes.push(tape);
            }
        }
        (states, tapes)
    }

    fn states_backward(self, p: &[f32], g: &mut [f64], tapes: &[Advance], dstates: &[Latent; SUB_POSES]) -> Latent {
        let mut dinitial = dstates[MIDDLE];
        for (side, tapes) in tapes.chunks(MIDDLE).enumerate() {
            let mut carry = [0.0; LATENT];
            for i in (1..=MIDDLE).rev() {
                let state = if side == 0 { MIDDLE - i } else { MIDDLE + i };
                carry = axpy(&carry, 1.0, &dstates[state]);
                carry = self.advance_backward(p, g, &tapes[i - 1], &carry);
            }
            dinitial = axpy(&dinitial, 1.0, &carry);
        }
        dinitial
    }
}

/// Offsets of every parameter in a view's flat vector; the embedding is at 0.
struct Layout {
    pose_features: Linear,
    /// Rigid, then refinement ("cmr"), branch.
    encoders: [Linear; 2],
    flows: [Flow; 2],
    axis: Linear,
    angle: Linear,
    translation: Linear,
    cmr_matrix: Linear,
    cmr_translation: Linear,
    len: usize,
}

const fn layout() -> Layout {
    let pose_features = Linear::new(LATENT, 12, LATENT);
    let rigid_encoder = Linear::new(pose_features.end(), FEATURE, LATENT);
    let cmr_encoder = Linear::new(rigid_encoder.end(), FEATURE, LATENT);
    let rigid_flow = Flow::new(cmr_encoder.end());
    let cmr_flow = Flow::new(rigid_flow.end());
    let axis = Linear::new(cmr_flow.end(), HALF, 3);
    let angle = Linear::new(axis.end(), HALF, 1);
    let translation = Linear::new(angle.end(), HALF, 3);
    let cmr_matrix = Linear::new(translation.end(), HALF, 9);
    let cmr_translation = Linear::new(cmr_matrix.end(), HALF, 3);
    Layout {
        pose_features,
        encoders: [rigid_encoder, cmr_encoder],
        flows: [rigid_flow, cmr_flow],
        axis,
        angle,
        translation,
        cmr_matrix,
        cmr_translation,
        len: cmr_translation.end(),
    }
}

const LAYOUT: Layout = layout();

/// splitmix64, so init needs no RNG dependency and is reproducible per view.
struct SplitMix(u64);

impl SplitMix {
    fn uniform(&mut self, bound: f32) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u32 << 24) as f32).mul_add(2.0, -1.0) * bound
    }
}

fn mul(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn transpose(a: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

fn apply(a: &M3, v: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
}

fn combine(terms: &[(f64, &M3)]) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| terms.iter().map(|(s, m)| s * m[i][j]).sum()))
}

fn dot(a: &M3, b: &M3) -> f64 {
    (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| a[i][j] * b[i][j]).sum()
}

const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Screw-motion pieces (paper eq. 9/10): the unit axis's skew `W`, `W^2`,
/// rotation `R` and Jacobian `J`. The axis is normalised with `+1e-10` as in
/// the Python, which matters because heads start at ~1e-5 scale.
struct Screw {
    norm: f64,
    w: M3,
    w2: M3,
    r: M3,
    j: M3,
}

impl Screw {
    fn new(axis: &[f64; 3], theta: f64) -> Self {
        let norm = axis.iter().map(|a| a * a).sum::<f64>().sqrt();
        let [x, y, z] = axis.map(|a| a / (norm + 1e-10));
        let w = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]];
        let w2 = mul(&w, &w);
        let (s, c) = theta.sin_cos();
        let r = combine(&[(1.0, &IDENTITY), (s, &w), (1.0 - c, &w2)]);
        let j = combine(&[(theta, &IDENTITY), (1.0 - c, &w), (theta - s, &w2)]);
        Self { norm, w, w2, r, j }
    }

    /// Gradients wrt the raw axis and the angle given `dr` and `dj`.
    fn backward(&self, axis: &[f64; 3], theta: f64, dr: &M3, dj: &M3) -> ([f64; 3], f64) {
        let (s, c) = theta.sin_cos();
        let dtheta = dot(dr, &combine(&[(c, &self.w), (s, &self.w2)]))
            + dot(dj, &combine(&[(1.0, &IDENTITY), (s, &self.w), (1.0 - c, &self.w2)]));
        let dw2 = combine(&[(1.0 - c, dr), (theta - s, dj)]);
        let wt = transpose(&self.w);
        let dw = combine(&[(s, dr), (1.0 - c, dj), (1.0, &mul(&dw2, &wt)), (1.0, &mul(&wt, &dw2))]);
        let du = [dw[2][1] - dw[1][2], dw[0][2] - dw[2][0], dw[1][0] - dw[0][1]];
        // d(a / (|a| + eps)); torch's norm has zero gradient at the origin.
        let scale = self.norm + 1e-10;
        let radial = if self.norm > 0.0 {
            axis.iter().zip(&du).map(|(a, d)| a * d).sum::<f64>() / (self.norm * scale * scale)
        } else {
            0.0
        };
        (std::array::from_fn(|i| du[i] / scale - axis[i] * radial), dtheta)
    }
}

/// Decoded head outputs of one exposure pose.
struct Pose {
    axis: [f64; 3],
    angle: f64,
    /// Screw translation `v`, already times `scene_scale`.
    screw: [f64; 3],
    /// Refinement matrix including `+I`.
    rc: M3,
    /// Refinement translation, already times `scene_scale`.
    pc: [f64; 3],
}

impl Pose {
    /// Top three rows, row-major, of `T_rigid @ refinement`.
    fn transform(&self) -> [f64; 12] {
        let screw = Screw::new(&self.axis, self.angle);
        let a = mul(&screw.r, &self.rc);
        let rp = apply(&screw.r, &self.pc);
        let p = apply(&screw.j, &self.screw);
        std::array::from_fn(|e| {
            let (i, j) = (e / 4, e % 4);
            if j < 3 { a[i][j] } else { rp[i] + p[i] }
        })
    }

    /// `Rc Rc^T - I`.
    fn deviation(&self) -> M3 {
        combine(&[(1.0, &mul(&self.rc, &transpose(&self.rc))), (-1.0, &IDENTITY)])
    }
}

/// Activations of one forward pass, kept for the backward.
struct Forward {
    feature: [f64; FEATURE],
    /// Rigid, then refinement, latent states.
    states: [[Latent; SUB_POSES]; 2],
    tapes: [Vec<Advance>; 2],
    poses: [Pose; SUB_POSES],
}

struct View {
    pose: [f64; 12],
    params: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl View {
    /// torch defaults: `Linear` weights and biases `U(±1/sqrt(fan_in))`;
    /// heads xavier-uniform with the Python's tiny gain and zero bias;
    /// embedding and time codes zero.
    fn new(c2w: [f32; 12], seed: u64) -> Self {
        let l = &LAYOUT;
        let mut rng = SplitMix(seed);
        let mut params = vec![0.0; l.len];
        let flows = l.flows.iter().flat_map(|f| [f.rotation, f.translation]);
        for layer in [l.pose_features, l.encoders[0], l.encoders[1]].into_iter().chain(flows) {
            let bound = 1.0 / (layer.inp as f32).sqrt();
            for x in &mut params[layer.off..layer.end()] {
                *x = rng.uniform(bound);
            }
        }
        let gain = 1e-5 / ((HALF + 3) as f32 / 6.0).sqrt();
        for layer in [l.axis, l.angle, l.translation, l.cmr_matrix, l.cmr_translation] {
            let bound = gain * (6.0 / (layer.inp + layer.out) as f32).sqrt();
            for x in &mut params[layer.off..layer.off + layer.out * layer.inp] {
                *x = rng.uniform(bound);
            }
        }
        Self { pose: c2w.map(f64::from), m: vec![0.0; l.len], v: vec![0.0; l.len], params, t: 0 }
    }

    fn forward(&self, scene_scale: f64) -> Forward {
        let (l, p) = (&LAYOUT, &self.params[..]);
        let mut feature = [0.0; FEATURE];
        for (f, &e) in feature.iter_mut().zip(&p[..LATENT]) {
            *f = f64::from(e);
        }
        l.pose_features.forward(p, &self.pose, &mut feature[LATENT..]);
        let mut initial = [[0.0; LATENT]; 2];
        let mut tapes = [Vec::new(), Vec::new()];
        let mut states = [[[0.0; LATENT]; SUB_POSES]; 2];
        for b in 0..2 {
            l.encoders[b].forward(p, &feature, &mut initial[b]);
            (states[b], tapes[b]) = l.flows[b].states(p, &initial[b]);
        }
        let poses = std::array::from_fn(|s| {
            let (rigid, cmr) = (&states[0][s], &states[1][s]);
            let (mut axis, mut angle, mut screw) = ([0.0; 3], [0.0], [0.0; 3]);
            l.axis.forward(p, &rigid[..HALF], &mut axis);
            l.angle.forward(p, &rigid[..HALF], &mut angle);
            l.translation.forward(p, &rigid[HALF..], &mut screw);
            let (mut m, mut pc) = ([0.0; 9], [0.0; 3]);
            l.cmr_matrix.forward(p, &cmr[..HALF], &mut m);
            l.cmr_translation.forward(p, &cmr[HALF..], &mut pc);
            Pose {
                axis,
                angle: angle[0],
                screw: screw.map(|v| v * scene_scale),
                rc: std::array::from_fn(|i| std::array::from_fn(|j| m[3 * i + j] + IDENTITY[i][j])),
                pc: pc.map(|v| v * scene_scale),
            }
        });
        Forward { feature, states, tapes, poses }
    }

    /// Gradient of `<grad_k, transform_k> + penalty_weight * orthogonal`
    /// wrt every parameter of this view.
    fn gradient(&self, scene_scale: f64, k: usize, grad_k: &[f64; 12], penalty_weight: f64) -> Vec<f64> {
        let (l, p) = (&LAYOUT, &self.params[..]);
        let fw = self.forward(scene_scale);
        let mut g = vec![0.0; l.len];
        let mut dstates = [[[0.0; LATENT]; SUB_POSES]; 2];
        let unit = penalty_weight / (SUB_POSES * 9) as f64;
        for (s, pose) in fw.poses.iter().enumerate() {
            // |E| backward (sign, zero at zero), then E = Rc Rc^T - I.
            let de = pose.deviation().map(|row| row.map(|e| if e == 0.0 { 0.0 } else { unit * e.signum() }));
            let mut drc = mul(&combine(&[(1.0, &de), (1.0, &transpose(&de))]), &pose.rc);
            let mut dpc = [0.0; 3];
            if s == k {
                // transform = [R Rc | R pc + J v].
                let ga: M3 = std::array::from_fn(|i| std::array::from_fn(|j| grad_k[4 * i + j]));
                let gt = [grad_k[3], grad_k[7], grad_k[11]];
                let screw = Screw::new(&pose.axis, pose.angle);
                let rt = transpose(&screw.r);
                let dr: M3 = std::array::from_fn(|i| {
                    std::array::from_fn(|j| (0..3).map(|c| ga[i][c] * pose.rc[j][c]).sum::<f64>() + gt[i] * pose.pc[j])
                });
                drc = combine(&[(1.0, &drc), (1.0, &mul(&rt, &ga))]);
                dpc = apply(&rt, &gt);
                let dj: M3 = std::array::from_fn(|i| std::array::from_fn(|j| gt[i] * pose.screw[j]));
                let dscrew = apply(&transpose(&screw.j), &gt).map(|d| d * scene_scale);
                let (daxis, dangle) = screw.backward(&pose.axis, pose.angle, &dr, &dj);
                let rigid = &fw.states[0][s];
                let (front, back) = dstates[0][s].split_at_mut(HALF);
                l.axis.backward(p, &mut g, &rigid[..HALF], &daxis, front);
                l.angle.backward(p, &mut g, &rigid[..HALF], &[dangle], front);
                l.translation.backward(p, &mut g, &rigid[HALF..], &dscrew, back);
            }
            let cmr = &fw.states[1][s];
            let (front, back) = dstates[1][s].split_at_mut(HALF);
            l.cmr_matrix.backward(p, &mut g, &cmr[..HALF], drc.as_flattened(), front);
            l.cmr_translation.backward(p, &mut g, &cmr[HALF..], &dpc.map(|d| d * scene_scale), back);
        }
        let mut dfeature = [0.0; FEATURE];
        for b in 0..2 {
            let dinitial = l.flows[b].states_backward(p, &mut g, &fw.tapes[b], &dstates[b]);
            l.encoders[b].backward(p, &mut g, &fw.feature, &dinitial, &mut dfeature);
        }
        for i in 0..LATENT {
            g[i] += dfeature[i];
        }
        l.pose_features.backward(p, &mut g, &self.pose, &dfeature[LATENT..], &mut [0.0; 12]);
        g
    }

    fn adam(&mut self, g: &[f64], lr: f32) {
        self.t += 1;
        let bc1 = 1.0 - BETA_1.powi(self.t);
        let bc2 = 1.0 - BETA_2.powi(self.t);
        for (((x, m), v), &g) in self.params.iter_mut().zip(&mut self.m).zip(&mut self.v).zip(g) {
            let g = g as f32;
            *m = BETA_1 * *m + (1.0 - BETA_1) * g;
            *v = BETA_2 * *v + (1.0 - BETA_2) * g * g;
            *x -= lr * (*m / bc1) / ((*v / bc2).sqrt() + EPSILON);
        }
    }
}

pub struct Trajectories {
    scene_scale: f32,
    seed: u64,
    views: Vec<Option<View>>,
}

impl Trajectories {
    pub fn new(scene_scale: f32, seed: u64) -> Self {
        Self { scene_scale, seed, views: Vec::new() }
    }

    /// The nine sub-pose transforms of `view` (render with `c2w * transform`),
    /// creating its model from `c2w` (3x4 row-major) on first use. Each view's
    /// init depends only on `seed` and its index.
    pub fn transforms(&mut self, view: usize, c2w: [f32; 12]) -> [Mat4; SUB_POSES] {
        if self.views.len() <= view {
            self.views.resize_with(view + 1, || None);
        }
        let seed = self.seed ^ (view as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        let model = self.views[view].get_or_insert_with(|| View::new(c2w, seed));
        let fw = model.forward(f64::from(self.scene_scale));
        fw.poses.map(|pose| {
            let t = pose.transform().map(|x| x as f32);
            Mat4::from_cols_array(&[
                t[0], t[4], t[8], 0.0, t[1], t[5], t[9], 0.0, t[2], t[6], t[10], 0.0, t[3], t[7], t[11], 1.0,
            ])
        })
    }

    /// One Adam step for `view` (after `transforms` created it): `grad_k` is
    /// `dLoss/dTransform_k` for sub-pose `k` (top three rows, row-major), and
    /// the gradient of `penalty_weight * orthogonal` is added.
    pub fn step(&mut self, view: usize, k: usize, grad_k: [f32; 12], penalty_weight: f32, lr: f32) {
        let model = self.views[view].as_mut().expect("transforms() creates a view before its step");
        let g = model.gradient(
            f64::from(self.scene_scale),
            k,
            &grad_k.map(f64::from),
            f64::from(penalty_weight),
        );
        model.adam(&g, lr);
    }
}
