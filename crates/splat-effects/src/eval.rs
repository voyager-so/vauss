//! CPU reference evaluator (rayon). Every other evaluator (WGSL here, GLSL in the viewer) ports
//! this math and is checked against `crates/splat-effects/golden.json`, which this file generates.
//!
//! Per splat `i`, from the immutable rest state only (pure in rest, graph, t):
//! 1. `p = F(μ)`: the nodes in order, each with weight `w_k = mask_k(μ) · curve_k(t)`.
//! 2. `J`: central difference of the whole position map over the rest μ (masks re-evaluated at
//!    the perturbed point), step `h = jacobian_step` or `1e-3 · scene radius`; then
//!    `J ← R_spin · J` for explode's rigid per-splat spin.
//! 3. Covariance: `Σ' = J Σ Jᵀ`, refactored by a symmetric eigendecomposition (`CovMode::Exact`)
//!    or by `R = polar(J)`, `s'_k = s_k |J R(q) e_k|` (`CovMode::Fast`, the viewers' mode).
//!    Dissolve's shrink multiplies `s'`; `s'` is clamped to `[1e-7, 10 · max rest scale]`.
//! 4. Opacity `α' = α · Π factors`, DC colour `c = 0.5 + C0·dc` tinted, SH view direction unchanged.
//!
//! Splats whose weight is zero on every node keep their rest state bit for bit. Splats that come
//! out non-finite are counted in `Frame::invalid` and reset to rest with opacity logit
//! [`DROPPED_LOGIT`], so the splat count and package order never change.

use std::collections::HashMap;

use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;

use crate::Splats;
use crate::graph::{Bbox, Crumble, Curve, FrontKind, DEFAULT_ANISO_MAX, Float, Floor, Graph, Mask, Melt, MorphTarget, Op, Ripple, Shatter, V3, Window};
use crate::hash::{SplatRandoms, splat_randoms};

pub const SH_C0: f32 = 0.282_094_8;
/// Opacity logit given to dropped (invalid) splats: alpha ~1e-13.
pub const DROPPED_LOGIT: f32 = -30.0;
pub const MIN_SCALE: f32 = 1e-7;
pub const MAX_SCALE_FACTOR: f32 = 10.0;
/// The engine never rotates the SH view direction into the deformed frame.
pub const SH_VIEW_ROTATION: &str = "unchanged";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovMode {
    /// Full `J Σ Jᵀ` refactored by symmetric eigendecomposition (engine high-quality mode).
    #[default]
    Exact,
    /// Polar rotation of `J` composed with `q`, `s' = s ⊙ |J R(q) columns|` (viewer fast mode).
    Fast,
}

#[derive(Debug, Clone)]
pub struct EvalOpts {
    pub cov: CovMode,
    /// The package's up axis (scene.json `up`); default axes resolve to it.
    pub up: V3,
    /// Scene radius for the Jacobian step when the graph has no `jacobian_step`.
    /// None: half the diagonal of the rest positions' bounding box.
    pub scene_radius: Option<f32>,
    /// Per-splat weights in [0, 1] for each object mask, keyed by `ObjectMask::key()`.
    pub objects: HashMap<String, Vec<f32>>,
    /// The package's floor (collision/floor.json); a graph `floor` overrides it.
    pub floor: Option<Floor>,
    /// Scene units per metre (1 / scene.json `meters_per_unit`); gravity = 9.81 × this.
    pub units_per_meter: f32,
}

impl Default for EvalOpts {
    fn default() -> Self {
        Self { cov: CovMode::Exact, up: [0.0, 1.0, 0.0], scene_radius: None, objects: HashMap::new(), floor: None, units_per_meter: 1.0 }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("object mask {key:?}: no per-splat weights in EvalOpts::objects and no `box` fallback")]
    MissingObject { key: String },
    #[error("object mask {key:?}: {got} weights for {want} splats")]
    ObjectLen { key: String, got: usize, want: usize },
    #[error("mask nesting needs a stack of {depth}; the GPU evaluator supports {max}")]
    MaskTooDeep { depth: usize, max: usize },
}

/// One evaluated frame, in the rest's layout and parameterisation.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub pos: Vec<[f32; 3]>,
    /// Unit quaternion w, x, y, z with w >= 0.
    pub rot: Vec<[f32; 4]>,
    /// Log scales.
    pub scale: Vec<[f32; 3]>,
    /// Opacity logit.
    pub opacity: Vec<f32>,
    /// SH DC coefficient.
    pub dc: Vec<[f32; 3]>,
    /// Max node weight per splat (mask x curve), for mask passes.
    pub weight: Vec<f32>,
    /// Splats dropped for non-finite output (reset to rest, opacity [`DROPPED_LOGIT`]).
    pub invalid: usize,
    /// Splats with a non-zero weight on some node.
    pub selected: usize,
    /// Released particles: render with the DC colour only (higher SH bands zeroed).
    pub sh_drop: Vec<bool>,
}

impl Frame {
    /// The rest splats with this frame's attributes (higher-order SH, render mode copied).
    pub fn to_splats(&self, rest: &Splats) -> Splats {
        let mut g = rest.clone();
        g.pos.clone_from(&self.pos);
        g.rot.clone_from(&self.rot);
        g.scale.clone_from(&self.scale);
        g.opacity.clone_from(&self.opacity);
        g.dc.clone_from(&self.dc);
        let k = g.sh_coeffs() * 3;
        if k > 0 {
            for (i, _) in self.sh_drop.iter().enumerate().filter(|(_, d)| **d) {
                g.rest[i * k..(i + 1) * k].iter_mut().for_each(|v| *v = 0.0);
            }
        }
        g
    }
}

/// `|μ' - μ|` per splat.
pub fn displacement(rest: &Splats, frame: &Frame) -> Vec<f32> {
    rest.pos.par_iter().zip(&frame.pos).map(|(a, b)| (Vec3::from(*b) - Vec3::from(*a)).length()).collect()
}

/// Half the diagonal of the positions' bounding box (1 when empty or degenerate).
pub fn scene_radius(pos: &[[f32; 3]]) -> f32 {
    let (lo, hi) = pos.par_iter().fold(
        || (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), p| {
            let p = Vec3::from(*p);
            if p.is_finite() { (lo.min(p), hi.max(p)) } else { (lo, hi) }
        },
    ).reduce(
        || (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |a, b| (a.0.min(b.0), a.1.max(b.1)),
    );
    let r = 0.5 * (hi - lo).length();
    if r.is_finite() && r > 0.0 { r } else { 1.0 }
}

/// Evaluate `graph` at time `t` (seconds). For many frames build a [`Prepared`] once.
pub fn evaluate(rest: &Splats, graph: &Graph, t: f32, opts: &EvalOpts) -> Result<Frame, EvalError> {
    Ok(Prepared::new(rest, graph, opts)?.frame(t))
}

// ---- compiled graph: axes resolved and normalised, object masks bound to weight slices ----

#[derive(Debug, Clone)]
pub(crate) enum CMask {
    All,
    Box { min: Vec3, max: Vec3, falloff: f32 },
    Sphere { center: Vec3, radius: f32, falloff: f32 },
    Height { axis: Vec3, min: f32, max: f32, falloff: f32 },
    Object { slot: usize },
    Invert(Box<CMask>),
    Mul(Vec<CMask>),
}

#[derive(Debug, Clone)]
pub(crate) enum COp {
    Vortex { center: Vec3, axis: Vec3, angle: f32, angle_per_unit: f32, lift: f32, expand: f32 },
    Wave { direction: Vec3, wavelength: f32, speed: f32, amplitude: f32, axis: Vec3 },
    Explode { center: Vec3, distance: f32, spin: f32, jitter: f32 },
    Dissolve { direction: Vec3, drift: f32, softness: f32, shrink: f32 },
    Morph { target: MorphTarget, center: Vec3, radius: f32, axis: Vec3 },
    Opacity { to: f32 },
    Tint { color: Vec3, amount: f32 },
    Crumble(Crumble, Region),
    Shatter(Shatter, Region),
    Ripple(Ripple, Vec3),
    Float(Float, Region),
    Melt(Melt, Region),
}

/// A physical node's resolved bbox: `lo`/`hi` corners, `centre` (bbox centre), `foot` (centre
/// projected onto the floor), `hmin`/`hmax` (height range along the floor normal).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Region {
    pub lo: Vec3,
    pub hi: Vec3,
    pub centre: Vec3,
    pub foot: Vec3,
    pub hmin: f32,
    pub hmax: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct CNode {
    pub mask: CMask,
    pub window: Option<Window>,
    pub op: COp,
}

fn unit3(v: V3) -> Vec3 {
    Vec3::from(v).normalize_or_zero()
}

/// A graph bound to one rest scene: compiled nodes, object weights and the Jacobian step.
pub struct Prepared<'a> {
    pub(crate) rest: &'a Splats,
    pub(crate) nodes: Vec<CNode>,
    pub(crate) objects: Vec<&'a [f32]>,
    pub(crate) seed: u32,
    pub(crate) duration: f32,
    pub(crate) looping: bool,
    pub(crate) h: f32,
    pub(crate) cov: CovMode,
    /// Floor normal (unit), floor offset (`n·p = d`), gravity, anisotropy limit.
    pub(crate) floor_n: Vec3,
    pub(crate) floor_d: f32,
    pub(crate) gravity: f32,
    pub(crate) aniso_max: f32,
}

impl<'a> Prepared<'a> {
    pub fn new(rest: &'a Splats, graph: &Graph, opts: &'a EvalOpts) -> Result<Self, EvalError> {
        let graph = resolve(rest, graph, opts)?;
        let (nodes, objects) = compile(&graph, opts, rest.len())?;
        let h = graph.jacobian_step.unwrap_or_else(|| 1e-3 * opts.scene_radius.unwrap_or_else(|| scene_radius(&rest.pos)));
        let floor = graph.floor.expect("resolved");
        Ok(Self {
            rest,
            nodes,
            objects,
            seed: graph.seed,
            duration: graph.duration_s,
            looping: graph.looping,
            h,
            cov: opts.cov,
            floor_n: unit3(floor.normal),
            floor_d: floor.offset,
            gravity: graph.gravity.expect("resolved"),
            aniso_max: graph.aniso_max.expect("resolved"),
        })
    }

    /// The Jacobian finite-difference step in scene units.
    pub fn jacobian_step(&self) -> f32 {
        self.h
    }

    /// Effective time: wrapped into [0, duration) for looping graphs.
    pub fn effective_t(&self, t: f32) -> f32 {
        if self.looping { t.rem_euclid(self.duration) } else { t }
    }

    /// Per-node curve weights at time `t`.
    pub fn curve_weights(&self, t: f32) -> Vec<f32> {
        let t = self.effective_t(t);
        self.nodes.iter().map(|n| n.window.map_or(1.0, |w| curve_weight(&w, t))).collect()
    }

    pub fn frame(&self, t: f32) -> Frame {
        let te = self.effective_t(t);
        let cw = self.curve_weights(t);
        let n = self.rest.len();
        let mut f = Frame {
            pos: vec![[0.0; 3]; n],
            rot: vec![[0.0; 4]; n],
            scale: vec![[0.0; 3]; n],
            opacity: vec![0.0; n],
            dc: vec![[0.0; 3]; n],
            weight: vec![0.0; n],
            invalid: 0,
            selected: 0,
            sh_drop: vec![false; n],
        };
        let (invalid, selected) = (f.pos.par_iter_mut(), f.rot.par_iter_mut(), f.scale.par_iter_mut(), f.opacity.par_iter_mut(), f.dc.par_iter_mut(), f.weight.par_iter_mut(), f.sh_drop.par_iter_mut())
            .into_par_iter()
            .enumerate()
            .map(|(i, (pos, rot, scale, opac, dc, weight, drop))| {
                let o = self.splat(i, te, &cw);
                *drop = o.sh_drop;
                *pos = o.pos;
                *rot = o.rot;
                *scale = o.scale;
                *opac = o.opacity;
                *dc = o.dc;
                *weight = o.weight;
                (usize::from(o.invalid), usize::from(o.weight > 0.0))
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        f.invalid = invalid;
        f.selected = selected;
        f
    }

    fn mask_weight(&self, m: &CMask, p: Vec3, i: usize) -> f32 {
        mask_weight(&self.objects, m, p, i)
    }

    /// The position map `F(μ)` of splat `i`, plus its side effects when `side` is given.
    /// Physical nodes read the unperturbed rest `mu0` (mask, release time, chunk) and act on the
    /// current `p` as a rigid motion, so the finite-difference Jacobian sees their rotation.
    #[allow(clippy::too_many_arguments)]
    fn map(&self, mu: Vec3, mu0: Vec3, i: usize, rnd: &SplatRandoms, t: f32, cw: &[f32], mut side: Option<&mut Side>) -> Vec3 {
        let mut p = mu;
        for (k, n) in self.nodes.iter().enumerate() {
            if cw[k] == 0.0 {
                continue;
            }
            if let Some(p2) = self.physical(n, p, mu0, i, rnd, t, side.as_deref_mut()) {
                p = p2;
                continue;
            }
            let w = self.mask_weight(&n.mask, mu, i) * cw[k];
            if w == 0.0 {
                continue;
            }
            if let Some(s) = side.as_deref_mut() {
                s.weight = s.weight.max(w);
                if !matches!(n.op, COp::Opacity { .. } | COp::Tint { .. }) {
                    s.deform = true;
                }
            }
            p = self.classic(&n.op, p, w, rnd, t, side.as_deref_mut());
        }
        p
    }
}

pub(crate) fn mask_weight(objects: &[&[f32]], m: &CMask, p: Vec3, i: usize) -> f32 {
    {
        match m {
            CMask::All => 1.0,
            CMask::Box { min, max, falloff } => {
                let q = (*min - p).max(Vec3::ZERO) + (p - *max).max(Vec3::ZERO);
                ramp(q.length(), *falloff)
            }
            CMask::Sphere { center, radius, falloff } => ramp(((p - *center).length() - radius).max(0.0), *falloff),
            CMask::Height { axis, min, max, falloff } => {
                let h = p.dot(*axis);
                ramp((min - h).max(h - max).max(0.0), *falloff)
            }
            CMask::Object { slot } => objects[*slot][i].clamp(0.0, 1.0),
            CMask::Invert(m) => 1.0 - mask_weight(objects, m, p, i),
            CMask::Mul(ms) => ms.iter().map(|m| mask_weight(objects, m, p, i)).product(),
        }
    }
}

impl Prepared<'_> {
    /// One classic (effect graph v1) node at weight `w`.
    fn classic(&self, op: &COp, mut p: Vec3, w: f32, rnd: &SplatRandoms, t: f32, mut side: Option<&mut Side>) -> Vec3 {
        {
            match op {
                COp::Vortex { center, axis, angle, angle_per_unit, lift, expand } => {
                    let d = p - *center;
                    let theta = w * (angle + angle_per_unit * d.dot(*axis));
                    let d = rotate(d, *axis, theta);
                    let perp = d - d.dot(*axis) * *axis;
                    let len = perp.length();
                    let radial = if len > 1e-9 { perp / len } else { Vec3::ZERO };
                    p = *center + d + w * lift * *axis + w * expand * radial;
                }
                COp::Wave { direction, wavelength, speed, amplitude, axis } => {
                    let phase = std::f32::consts::TAU * (direction.dot(p) - speed * t) / wavelength;
                    p += w * amplitude * phase.sin() * *axis;
                }
                COp::Explode { center, distance, spin, jitter } => {
                    let r3 = Vec3::from(rnd.r3) - Vec3::splat(0.5);
                    let dir = (p - *center + *jitter * r3).normalize_or_zero();
                    p += w * distance * (0.5 + rnd.r) * dir;
                    if let Some(s) = side.as_deref_mut() {
                        let a = Vec3::from(rnd.axis);
                        let a = if a.length_squared() > 1e-12 { a.normalize() } else { Vec3::Y };
                        s.spin = Mat3::from_axis_angle(a, w * spin * (rnd.r - 0.5)) * s.spin;
                    }
                }
                COp::Dissolve { direction, drift, softness, shrink } => {
                    let e = dissolve_progress(rnd.r, *softness, w);
                    p += e * drift * *direction;
                    if let Some(s) = side.as_deref_mut() {
                        s.alpha *= 1.0 - e;
                        s.shrink *= 1.0 - shrink * e;
                    }
                }
                COp::Morph { target, center, radius, axis } => {
                    let d = p - *center;
                    let proj = match target {
                        MorphTarget::Sphere => *center + *radius * d.normalize_or_zero(),
                        MorphTarget::Cylinder => {
                            let h = d.dot(*axis);
                            *center + h * *axis + *radius * (d - h * *axis).normalize_or_zero()
                        }
                        MorphTarget::Plane => p - d.dot(*axis) * *axis,
                    };
                    p = p.lerp(proj, w);
                }
                COp::Opacity { to } => {
                    if let Some(s) = side.as_deref_mut() {
                        s.alpha *= 1.0 - w + w * to;
                    }
                }
                COp::Tint { color, amount } => {
                    if let Some(s) = side.as_deref_mut() {
                        s.color *= Vec3::ONE + w * amount * (*color - Vec3::ONE);
                    }
                }
                _ => unreachable!("physical node in classic()"),
            }
        }
        p
    }

    /// A physical node: `None` when `n` is classic; otherwise the new position (unchanged when
    /// the splat is not a member or not yet released, which also leaves `side` untouched).
    #[allow(clippy::too_many_arguments)]
    fn physical(&self, n: &CNode, p: Vec3, mu0: Vec3, i: usize, rnd: &SplatRandoms, t: f32, mut side: Option<&mut Side>) -> Option<Vec3> {
        let (fnrm, fd, g) = (self.floor_n, self.floor_d, self.gravity);
        let member = || self.mask_weight(&n.mask, mu0, i) >= 0.5;
        Some(match &n.op {
            COp::Crumble(c, reg) => {
                let t = if c.reverse { self.duration - t } else { t };
                let pile_r = c.pile_r.expect("resolved");
                if !member() {
                    // Contact shadow: floor splats under the pile darken as it lands (fake AO).
                    let y = mu0.dot(fnrm) - fd;
                    let rho = horiz(mu0 - reg.foot, fnrm).length();
                    if c.shadow <= 0.0 || !(-SHADOW_BELOW..=SHADOW_ABOVE).contains(&y) || rho > 2.0 * pile_r {
                        return Some(p);
                    }
                    let grow = smoothstep01((t - c.start_s) / (c.sweep_s + SHADOW_LAG_S));
                    if grow <= 0.0 {
                        return Some(p);
                    }
                    if let Some(s) = side {
                        s.weight = s.weight.max(grow);
                        s.color *= 1.0 - c.shadow * grow * (-(rho * rho) / (pile_r * pile_r)).exp();
                    }
                    return Some(p);
                }
                let d = front_dir(c, fnrm);
                let reach = (0.5 * (reg.hi - reg.lo)).dot(d.abs());
                let (lo, hi) = (reg.centre.dot(d) - reach, reg.centre.dot(d) + reach);
                // A noisy, irregular front: release height perturbed by a smooth positional noise.
                let fq = c.front_freq;
                let nz = 0.6 * (fq * mu0.x + 1.7).sin() * (fq * mu0.z + 0.4).sin()
                    + 0.4 * (2.7 * fq * mu0.y + 0.9).sin() * (2.3 * fq * mu0.x + 2.2).sin();
                let h = ((mu0.dot(d) - lo) / (hi - lo).max(1e-6) + c.front_noise * nz).clamp(0.0, 1.0);
                let ti = c.start_s + h * c.sweep_s + c.jitter * c.sweep_s * rnd.r;
                let tau = t - ti;
                if tau <= 0.0 {
                    return Some(p);
                }
                // Released grains stay in the material's palette: within ±palette_tol of its mean colour.
                if let (Some(s), Some(pal)) = (side.as_deref_mut(), c.palette) {
                    let pal = Vec3::from(pal);
                    s.color = pal + (s.color - pal).clamp(Vec3::splat(-c.palette_tol), Vec3::splat(c.palette_tol));
                }
                // Per-grain brightness jitter so a heap reads as granular stone, not clay.
                if let Some(s) = side.as_deref_mut() {
                    s.color *= 1.0 + c.grain_jitter * (2.0 * rnd.r3[1] - 1.0);
                }
                let out = horiz(mu0 - reg.foot, fnrm).normalize_or_zero();
                let jit = horiz(Vec3::from(rnd.r3) * 2.0 - Vec3::ONE, fnrm);
                let v0 = c.burst * (out + 0.3 * jit) + c.up_kick * (0.5 + rnd.r) * fnrm;
                let ua = 0.5 * rnd.axis[0] + 0.5;
                let haze = 0.5 * rnd.axis[2] + 0.5 < c.haze;
                // Horizontal curl-like turbulence: a deterministic sum of sines of rest position and time.
                let tf = c.turb_freq;
                let ph = std::f32::consts::TAU;
                let turb = c.turbulence
                    * horiz(
                        Vec3::new(
                            (tf * mu0.y + 1.3 * tau + ph * rnd.r).sin(),
                            (tf * mu0.z + 1.7 * tau + ph * rnd.r3[0]).sin(),
                            (tf * mu0.x + 1.1 * tau + ph * rnd.r3[1]).sin(),
                        ),
                        fnrm,
                    )
                    * smoothstep01(tau / TURB_RAMP_S);
                let wind = Vec3::from(c.wind);
                let mut rim = 0.0;
                let mut settled = 0.0;
                let mut heap_shade = 1.0;
                let pos = if haze {
                    mu0 + HAZE_DRIFT * v0 * tau + c.haze_rise * tau * fnrm + wind * tau + turb
                } else {
                    let b = ballistic(mu0, v0, tau, fnrm, fd, g, 0.0, c.restitution, c.friction);
                    let air = 1.0 - smoothstep01((tau - b.t_land) / TURB_RAMP_S);
                    // After landing, grains settle into a low, irregular mound: the landing spot
                    // (which keeps the fall footprint and wind drift) pulled toward the footprint
                    // centre by `gather`, plus a small scatter; height is a Gaussian profile with
                    // positional ripples, larger flakes resting higher; the rim thins out.
                    let settle = smoothstep01((tau - b.t_land) / c.settle_s);
                    let (sa, ca) = (ph * (0.5 * rnd.axis[1] + 0.5)).sin_cos();
                    let (e1, e2) = floor_basis(fnrm);
                    let foot = reg.foot + (fd - reg.foot.dot(fnrm)) * fnrm;
                    let bl = ballistic(mu0, v0, b.t_land, fnrm, fd, g, 0.0, c.restitution, c.friction);
                    let land = horiz(bl.pos + wind * b.t_land - foot, fnrm);
                    let off = land * (1.0 - c.gather) + MOUND_SCATTER * pile_r * rnd.r.sqrt() * (ca * e1 + sa * e2);
                    let rho = off.length();
                    let irr = 0.6 + 0.4 * (7.3 * off.dot(e1) + 2.1).sin() * (6.1 * off.dot(e2) + 0.7).sin();
                    let heap = c.pile_h * (-(rho * rho) / (pile_r * pile_r)).exp() * irr * (0.4 + 0.6 * ua) * rnd.r3[2].powf(HEAP_FILL_POW);
                    let target = foot + off + heap * fnrm;
                    // Horizontally, grains curve toward their heap spot already while falling (so an
                    // overhang lands on the base, not beside it); vertically they fall, then settle.
                    let pull = settle.max(smoothstep01(tau / b.t_land.max(1e-3)));
                    let fall = b.pos + wind * tau.min(b.t_land);
                    let (hf, ht) = (horiz(fall, fnrm), horiz(target, fnrm));
                    let pos = hf.lerp(ht, pull) + (fall.dot(fnrm) + (target.dot(fnrm) - fall.dot(fnrm)) * settle) * fnrm + air * turb;
                    rim = settle * smoothstep01((rho / pile_r - RIM_START) / RIM_WIDTH);
                    settled = settle;
                    // Self-shadowing: a grain's colour scales with its height fraction in the mound
                    // (r3.z), darker underneath, full brightness on top.
                    heap_shade = 1.0 - c.heap_ao * (1.0 - rnd.r3[2].powf(HEAP_FILL_POW)) * settle;
                    if let Some(s) = side.as_deref_mut() {
                        // Landed grains turn to dust: desaturate toward their luminance, then darken.
                        let landed = smoothstep01((tau - b.t_land) / DUST_S);
                        let luma = s.color.dot(Vec3::new(0.299, 0.587, 0.114));
                        s.color = s.color.lerp(Vec3::splat(luma), c.dust_desat * landed);
                        s.color *= 1.0 - (1.0 - c.dust) * landed;
                    }
                    pos
                };
                if let Some(s) = side {
                    s.weight = 1.0;
                    s.sh_drop = true;
                    // Flakes snap to grain size at release (no ease: an easing splat streaks);
                    // haze puffs grow over GRAIN_S.
                    s.grain = Some(if haze {
                        s.alpha *= c.haze_alpha * (1.0 - smoothstep01(tau / c.haze_s));
                        (c.haze_size, smoothstep01(tau / GRAIN_S), true)
                    } else {
                        // The mound's rim thins out: smaller, fainter grains blend into the floor.
                        // Flakes are opaque stone: lift the rest opacity to at least grain_opacity
                        // (scans are full of faint interior splats that would read as haze).
                        let a0 = 1.0 / (1.0 + (-self.rest.opacity[i]).exp());
                        s.alpha *= (c.grain_opacity / a0.max(1e-6)).max(1.0);
                        s.alpha *= 1.0 - RIM_FADE * rim;
                        s.color *= heap_shade;
                        // A few chunky fragments among the grains.
                        let chunk = if 0.5 * rnd.axis[1] + 0.5 < c.chunk_frac { CHUNK_SCALE } else { 1.0 };
                        (c.grain * c.size_min * (c.size_max / c.size_min).powf(ua) * chunk * (1.0 - 0.5 * rim) * (1.0 - c.land_shrink * settled), 1.0, false)
                    });
                }
                p + (pos - mu0)
            }
            COp::Shatter(sh, reg) => {
                if !member() {
                    return Some(p);
                }
                let k = nearest_seed(sh.seeds, reg, mu0, self.seed);
                let (ca, cb) = chunk_randoms(k, self.seed);
                let seed_k = chunk_seed(reg, &ca);
                let tau = t - sh.start_s - 0.1 * ca.r;
                if tau <= 0.0 {
                    return Some(p);
                }
                let rc = chunk_radius(sh.seeds, reg);
                let dir = horiz(seed_k - reg.foot, fnrm).normalize_or_zero();
                let jit = horiz(Vec3::from(ca.axis), fnrm);
                let v0 = sh.burst * (dir + 0.3 * jit) + sh.up_kick * (0.5 + cb.r) * fnrm;
                let b = ballistic(seed_k, v0, tau, fnrm, fd, g, rc, sh.restitution, sh.friction);
                let a = Vec3::from(cb.axis);
                let axis = if a.length_squared() > 1e-12 { a.normalize() } else { fnrm };
                let rate = sh.spin * (0.5 + cb.r);
                let theta = if tau <= b.t_contact {
                    rate * tau
                } else {
                    rate * (b.t_contact + SPIN_DAMP_S * (1.0 - (-(tau - b.t_contact) / SPIN_DAMP_S).exp()))
                };
                let q = Quat::from_axis_angle(axis, theta);
                if let Some(s) = side {
                    s.weight = 1.0;
                    s.rigid = q * s.rigid;
                }
                // Chunks are approximated by a contact radius; no splat of a turned chunk sinks below the floor.
                let p = b.pos + q * (p - seed_k);
                p + (fd - p.dot(fnrm)).max(0.0) * fnrm
            }
            COp::Ripple(r, light) => {
                let tau = t - r.start_s;
                if tau <= 0.0 {
                    return Some(p);
                }
                let w = self.mask_weight(&n.mask, mu0, i);
                if w == 0.0 || (r.rigid && w < 0.5) {
                    return Some(p);
                }
                // Rigid objects sample the wave once, at their footprint centre (the pivot).
                let pivot = r.bbox.filter(|_| r.rigid).map(|b| {
                    let c = 0.5 * (Vec3::from(b.min) + Vec3::from(b.max));
                    c - (c.dot(fnrm) - fd) * fnrm
                });
                let w = if pivot.is_some() { 1.0 } else { w };
                let xh = horiz(pivot.unwrap_or(mu0) - Vec3::from(r.source), fnrm);
                let rho = xh.length();
                let front = rho - r.speed * tau;
                let qd = front / r.packet_width;
                if qd * qd > RIPPLE_CUT {
                    return Some(p);
                }
                let env = w * r.amplitude * (-r.decay * rho).exp() * (-qd * qd).exp();
                let ph = std::f32::consts::TAU * front / r.wavelength;
                let (sn, cs) = ph.sin_cos();
                let u = env * sn;
                let du = env * (sn * (-r.decay - 2.0 * qd / r.packet_width) + cs * std::f32::consts::TAU / r.wavelength);
                let rhat = if rho > 1e-6 { xh / rho } else { Vec3::ZERO };
                let n2 = (fnrm - du * rhat).normalize();
                let q = rotation_arc(fnrm, n2);
                if let Some(pv) = pivot {
                    if let Some(s) = side {
                        s.weight = 1.0;
                        s.rigid = q * s.rigid;
                    }
                    return Some(pv + q * (p - pv) + u * fnrm);
                }
                if let Some(s) = side {
                    s.weight = s.weight.max(w);
                    s.rigid = q * s.rigid;
                    s.self_rot = q * s.self_rot;
                    s.color *= (1.0 + r.shade * (n2 - fnrm).dot(*light)).max(0.0);
                }
                p + u * fnrm
            }
            COp::Float(f, reg) => {
                let Some(fs) = float_state(f, reg, t, fnrm, fd) else { return Some(p) };
                if !member() {
                    // Ground shadow on the splats under the object.
                    if let Some(s) = side {
                        let dark = float_shadow(f, reg, &fs, mu0, fnrm);
                        if dark > 0.0 {
                            s.weight = s.weight.max(dark);
                            s.color *= 1.0 - dark;
                        }
                    }
                    return Some(p);
                }
                if let Some(s) = side {
                    s.weight = 1.0;
                    s.rigid = fs.yaw * fs.lean * s.rigid;
                }
                fs.pv + fs.yaw * (reg.centre + fs.lean * (p - reg.centre) - fs.pv) + fs.lift * fnrm
            }
            COp::Melt(m, reg) => {
                if !member() {
                    return Some(p);
                }
                let h = ((mu0.dot(fnrm) - reg.hmin) / (reg.hmax - reg.hmin).max(1e-6)).clamp(0.0, 1.0);
                let mm = smoothstep01((t - m.start_s - (1.0 - h) * 0.4 * m.duration_s) / (0.6 * m.duration_s));
                if mm <= 0.0 {
                    return Some(p);
                }
                if let Some(s) = side {
                    s.weight = 1.0;
                    s.deform = true;
                    s.color *= 1.0 + m.sheen * mm;
                }
                let y = p.dot(fnrm) - fd;
                let keep = (1.0 - mm) * (1.0 - mm);
                let p = p + (keep - 1.0) * y * fnrm;
                p + m.spread * mm * horiz(p - reg.foot, fnrm)
            }
            _ => return None,
        })
    }

    fn splat(&self, i: usize, t: f32, cw: &[f32]) -> SplatOut {
        let g = self.rest;
        let rest = SplatOut { pos: g.pos[i], rot: g.rot[i], scale: g.scale[i], opacity: g.opacity[i], dc: g.dc[i], weight: 0.0, invalid: false, sh_drop: false };
        let rnd = splat_randoms(i as u32, self.seed);
        let mu = Vec3::from(g.pos[i]);
        let dc = Vec3::from(g.dc[i]);
        let mut side = Side {
            weight: 0.0,
            alpha: 1.0,
            shrink: 1.0,
            spin: Mat3::IDENTITY,
            color: Vec3::splat(0.5) + SH_C0 * dc,
            deform: false,
            rigid: Quat::IDENTITY,
            self_rot: Quat::IDENTITY,
            grain: None,
            sh_drop: false,
        };
        let p = self.map(mu, mu, i, &rnd, t, cw, Some(&mut side));
        if side.weight == 0.0 {
            return rest;
        }
        let [w, x, y, z] = g.rot[i];
        let q = Quat::from_xyzw(x, y, z, w).normalize();
        let s = Vec3::from(g.scale[i]).exp();
        let s_max = MAX_SCALE_FACTOR * s.max_element();
        let (q2, s2) = if !side.deform {
            // Rigid only (physical nodes, opacity, tint): Σ' = R Σ Rᵀ exactly, no Jacobian.
            ((side.rigid * q).normalize(), s)
        } else {
            let h = self.h;
            let col = |e: Vec3| (self.map(mu + h * e, mu, i, &rnd, t, cw, None) - self.map(mu - h * e, mu, i, &rnd, t, cw, None)) / (2.0 * h);
            let jac = Mat3::from_quat(side.self_rot) * side.spin * Mat3::from_cols(col(Vec3::X), col(Vec3::Y), col(Vec3::Z));
            self.deformed_cov(jac, q, s)
        };
        let s2 = s2 * side.shrink;
        let s2 = match side.grain {
            Some((size, k, true)) => s2.lerp(Vec3::splat(size), k),
            // Flakes: capped to the grain size, then rounded (short axes grow to at least
            // max / FLAKE_ANISO) so thin rest splats don't read as hairs or needles.
            Some((size, k, false)) => {
                let f = s2.min(Vec3::splat(size));
                let f = f.max(Vec3::splat(f.max_element() / FLAKE_ANISO));
                s2.lerp(f, k)
            }
            None => s2,
        };
        // Anisotropy clamp: shrink long axes to max(rest ratio, 1) × aniso_max times the short one.
        let limit = (s.max_element() / s.min_element().max(MIN_SCALE)).max(1.0) * self.aniso_max;
        // The short axis is floored at ANISO_BASE_FLOOR x the rest short axis: a collapsed axis
        // (planar morph, melt) is numerically noisy and must not drive the long axes to zero.
        let base = s2.min_element().max(ANISO_BASE_FLOOR * s.min_element()).max(MIN_SCALE);
        let s2 = s2.min(Vec3::splat(base * limit));
        let s2 = s2.clamp(Vec3::splat(MIN_SCALE), Vec3::splat(s_max.max(MIN_SCALE)));
        let q2 = if q2.w < 0.0 { -q2 } else { q2 };
        let color = side.color;
        let dc2 = (color - Vec3::splat(0.5)) / SH_C0;
        let opac = scale_alpha(g.opacity[i], side.alpha);
        let ok = p.is_finite() && q2.is_finite() && s2.is_finite() && dc2.is_finite() && opac.is_finite();
        if !ok {
            return SplatOut { opacity: DROPPED_LOGIT, weight: side.weight, invalid: true, ..rest };
        }
        SplatOut {
            pos: p.into(),
            rot: [q2.w, q2.x, q2.y, q2.z],
            scale: s2.ln().into(),
            opacity: opac,
            dc: dc2.into(),
            weight: side.weight,
            invalid: false,
            sh_drop: side.sh_drop,
        }
    }

    /// `J Σ Jᵀ` refactored per [`CovMode`] into a rotation and linear scales.
    fn deformed_cov(&self, jac: Mat3, q: Quat, s: Vec3) -> (Quat, Vec3) {
        {
            match self.cov {
            CovMode::Exact => {
                let r = Mat3::from_quat(q);
                let m = jac * r * Mat3::from_diagonal(s);
                let (lam, v) = sym_eigen(m * m.transpose());
                (Quat::from_mat3(&v).normalize(), Vec3::from(lam).max(Vec3::ZERO).map(f32::sqrt))
            }
            CovMode::Fast => {
                let r = Mat3::from_quat(q);
                let jr = jac * r;
                let s2 = s * Vec3::new(jr.x_axis.length(), jr.y_axis.length(), jr.z_axis.length());
                (Quat::from_mat3(&polar(jac)).mul_quat(q).normalize(), s2)
            }
            }
        }
    }
}

struct Side {
    weight: f32,
    alpha: f32,
    shrink: f32,
    spin: Mat3,
    color: Vec3,
    /// A position-deforming classic node (or melt) is active: covariance from the FD Jacobian.
    deform: bool,
    /// Accumulated rigid rotation of the physical nodes (exact path when `!deform`).
    rigid: Quat,
    /// The part of `rigid` about the splat's own centre (ripple tilt), which the FD Jacobian
    /// cannot see; it premultiplies J on the deform path.
    self_rot: Quat,
    /// Grain size, blend and mode: s → lerp(s, min(s, size), k), or lerp(s, size, k) when iso
    /// (haze puffs grow).
    grain: Option<(f32, f32, bool)>,
    /// Released particles render with the DC colour only (higher SH bands dropped).
    sh_drop: bool,
}

struct SplatOut {
    pos: [f32; 3],
    rot: [f32; 4],
    scale: [f32; 3],
    opacity: f32,
    dc: [f32; 3],
    weight: f32,
    invalid: bool,
    sh_drop: bool,
}

/// Compile the graph's nodes for `n` splats; object masks become slots into the returned slices.
pub(crate) fn compile<'a>(graph: &Graph, opts: &'a EvalOpts, n: usize) -> Result<(Vec<CNode>, Vec<&'a [f32]>), EvalError> {
    let up = unit3(opts.up);
    let mut objects: Vec<&'a [f32]> = Vec::new();
    let mut slots: HashMap<String, usize> = HashMap::new();
    let mut nodes = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let mask = compile_mask(&node.mask, up, n, opts, &mut objects, &mut slots)?;
        nodes.push(CNode { mask, window: node.window, op: compile_op(&node.op, up, graph.floor.unwrap_or(Floor { normal: up.into(), offset: 0.0 })) });
    }
    Ok((nodes, objects))
}

fn compile_mask<'a>(
    m: &Mask,
    up: Vec3,
    n: usize,
    opts: &'a EvalOpts,
    objects: &mut Vec<&'a [f32]>,
    slots: &mut HashMap<String, usize>,
) -> Result<CMask, EvalError> {
    Ok(match m {
        Mask::All => CMask::All,
        Mask::Box { min, max, falloff } => CMask::Box { min: (*min).into(), max: (*max).into(), falloff: *falloff },
        Mask::Sphere { center, radius, falloff } => CMask::Sphere { center: (*center).into(), radius: *radius, falloff: *falloff },
        Mask::Height { axis, min, max, falloff } => {
            CMask::Height { axis: axis.map_or(up, unit3), min: *min, max: *max, falloff: *falloff }
        }
        Mask::Object(o) => {
            let key = o.key();
            match opts.objects.get(&key) {
                Some(wts) => {
                    if wts.len() != n {
                        return Err(EvalError::ObjectLen { key, got: wts.len(), want: n });
                    }
                    let next = objects.len();
                    let slot = *slots.entry(key).or_insert(next);
                    if slot == next {
                        objects.push(wts);
                    }
                    CMask::Object { slot }
                }
                None => match &o.bbox {
                    Some(b) => CMask::Box { min: b.min.into(), max: b.max.into(), falloff: b.falloff },
                    None => return Err(EvalError::MissingObject { key }),
                },
            }
        }
        Mask::Invert { mask } => CMask::Invert(Box::new(compile_mask(mask, up, n, opts, objects, slots)?)),
        Mask::Mul { masks } => CMask::Mul(masks.iter().map(|m| compile_mask(m, up, n, opts, objects, slots)).collect::<Result<_, _>>()?),
    })
}

fn compile_op(op: &Op, up: Vec3, floor: Floor) -> COp {
    let axis = |a: &Option<V3>| a.map_or(up, unit3);
    match op {
        Op::Vortex(v) => COp::Vortex {
            center: v.center.into(),
            axis: axis(&v.axis),
            angle: v.angle,
            angle_per_unit: v.angle_per_unit,
            lift: v.lift,
            expand: v.expand,
        },
        Op::Wave(w) => COp::Wave {
            direction: w.direction.into(),
            wavelength: w.wavelength,
            speed: w.speed,
            amplitude: w.amplitude,
            axis: w.displace_axis.map_or(up, Vec3::from),
        },
        Op::Explode(e) => COp::Explode { center: e.center.into(), distance: e.distance, spin: e.spin, jitter: e.jitter },
        Op::Dissolve(d) => COp::Dissolve { direction: axis(&d.direction), drift: d.drift, softness: d.softness, shrink: d.shrink },
        Op::Morph(m) => COp::Morph { target: m.target, center: m.center.into(), radius: m.radius, axis: axis(&m.axis) },
        Op::Opacity(o) => COp::Opacity { to: o.to },
        Op::Tint(t) => COp::Tint { color: t.color.into(), amount: t.amount },
        Op::Crumble(c) => COp::Crumble(c.clone(), region(c.bbox, floor)),
        Op::Shatter(s) => COp::Shatter(s.clone(), region(s.bbox, floor)),
        Op::Ripple(r) => COp::Ripple(r.clone(), unit3(r.light_dir)),
        Op::Float(f) => COp::Float(f.clone(), region(f.bbox, floor)),
        Op::Melt(m) => COp::Melt(m.clone(), region(m.bbox, floor)),
    }
}

fn region(b: Option<Bbox>, floor: Floor) -> Region {
    let b = b.expect("resolved bbox");
    let (lo, hi) = (Vec3::from(b.min), Vec3::from(b.max));
    let n = unit3(floor.normal);
    let centre = 0.5 * (lo + hi);
    let half = 0.5 * (hi - lo);
    let reach = half.dot(n.abs());
    let hc = centre.dot(n);
    Region { lo, hi, centre, foot: centre - (hc - floor.offset) * n, hmin: hc - reach, hmax: hc + reach }
}

/// The graph with every derived value filled in: `floor` (graph, else `opts.floor`, else the up
/// axis through the origin), `gravity`, `aniso_max`, each physical node's `bbox` (rest positions
/// with mask weight >= 0.5; an empty mask gives a zero box at the origin) and crumble `pile_r`.
/// Viewers read these; `vauss effect export` writes them.
pub fn resolve(rest: &Splats, graph: &Graph, opts: &EvalOpts) -> Result<Graph, EvalError> {
    let mut g = graph.clone();
    let floor = g.floor.or(opts.floor).unwrap_or(Floor { normal: opts.up, offset: 0.0 });
    g.floor = Some(Floor { normal: unit3(floor.normal).into(), offset: floor.offset });
    g.gravity = Some(g.gravity.unwrap_or(9.81 * opts.units_per_meter));
    g.aniso_max = Some(g.aniso_max.unwrap_or(DEFAULT_ANISO_MAX));
    let up = unit3(opts.up);
    let mut objects: Vec<&[f32]> = Vec::new();
    let mut slots: HashMap<String, usize> = HashMap::new();
    for node in &mut g.nodes {
        let need = match &node.op {
            Op::Crumble(c) => c.bbox.is_none(),
            Op::Shatter(s) => s.bbox.is_none(),
            Op::Float(f) => f.bbox.is_none(),
            Op::Melt(m) => m.bbox.is_none(),
            Op::Ripple(r) => r.rigid && r.bbox.is_none(),
            _ => false,
        };
        let bbox = if need {
            let mask = compile_mask(&node.mask, up, rest.len(), opts, &mut objects, &mut slots)?;
            let (lo, hi) = rest
                .pos
                .par_iter()
                .enumerate()
                .filter(|(i, p)| mask_weight(&objects, &mask, Vec3::from(**p), *i) >= 0.5)
                .fold(
                    || (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                    |(lo, hi), (_, p)| (lo.min(Vec3::from(*p)), hi.max(Vec3::from(*p))),
                )
                .reduce(|| (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |a, b| (a.0.min(b.0), a.1.max(b.1)));
            let b = if lo.is_finite() && hi.is_finite() { Bbox { min: lo.into(), max: hi.into() } } else { Bbox { min: [0.0; 3], max: [0.0; 3] } };
            Some(b)
        } else {
            None
        };
        match &mut node.op {
            Op::Crumble(c) => {
                if bbox.is_some() {
                    c.bbox = bbox;
                }
                if c.palette.is_none() {
                    let mask = compile_mask(&node.mask, up, rest.len(), opts, &mut objects, &mut slots)?;
                    let (sum, n) = rest
                        .pos
                        .par_iter()
                        .zip(&rest.dc)
                        .enumerate()
                        .filter(|(i, (p, _))| mask_weight(&objects, &mask, Vec3::from(**p), *i) >= 0.5)
                        .map(|(_, (_, dc))| (Vec3::splat(0.5) + SH_C0 * Vec3::from(*dc), 1usize))
                        .reduce(|| (Vec3::ZERO, 0), |a, b| (a.0 + b.0, a.1 + b.1));
                    c.palette = Some(if n > 0 { (sum / n as f32).into() } else { [0.5; 3] });
                }
                if c.pile_r.is_none() {
                    let b = c.bbox.expect("bbox");
                    let n = unit3(floor.normal);
                    let half = horiz(0.5 * (Vec3::from(b.max) - Vec3::from(b.min)), n).length();
                    c.pile_r = Some((0.75 * half).max(1e-3));
                }
            }
            Op::Shatter(s) if bbox.is_some() => s.bbox = bbox,
            Op::Float(f) if bbox.is_some() => f.bbox = bbox,
            Op::Melt(m) if bbox.is_some() => m.bbox = bbox,
            Op::Ripple(r) if bbox.is_some() => r.bbox = bbox,
            _ => {}
        }
    }
    Ok(g)
}

/// Released crumble flakes: max/min axis ratio at most this (rounded chips, not hairs).
pub const FLAKE_ANISO: f32 = 1.6;
/// Size factor of crumble's chunky fragments (`chunk_frac` of the grains).
pub const CHUNK_SCALE: f32 = 3.0;
/// Anisotropy clamp: the short axis it measures from is at least this x the rest short axis.
pub const ANISO_BASE_FLOOR: f32 = 0.1;
/// Seconds over which a released grain shrinks to `grain`.
pub const GRAIN_S: f32 = 0.2;
/// Seconds after landing over which dust darkens to `dust`.
pub const DUST_S: f32 = 0.3;
/// Turbulence fades in over this after release (and out after landing).
pub const TURB_RAMP_S: f32 = 0.3;
/// Haze puffs drift at this fraction of the launch velocity, without gravity.
pub const HAZE_DRIFT: f32 = 0.35;
/// Contact shadow: floor splats within [−below, +above] of the floor plane, growing over
/// `sweep_s + SHADOW_LAG_S` from `start_s`.
pub const SHADOW_BELOW: f32 = 0.25;
pub const SHADOW_ABOVE: f32 = 0.1;
pub const SHADOW_LAG_S: f32 = 1.5;
/// Mound: scatter radius (× pile_r) added to the gathered landing spot; the rim starts at
/// RIM_START × pile_r and is fully thinned RIM_WIDTH × pile_r later (alpha × (1 − RIM_FADE),
/// size × 0.5).
pub const MOUND_SCATTER: f32 = 0.25;
pub const RIM_START: f32 = 0.8;
pub const RIM_WIDTH: f32 = 0.6;
pub const RIM_FADE: f32 = 0.7;
/// Height fraction of a grain in the mound is r3.z^HEAP_FILL_POW: < 1 packs grains toward the
/// mound's surface so it reads as a solid volume, not a sparse spike.
pub const HEAP_FILL_POW: f32 = 0.35;

/// Crumble front travel direction (unit): `front_dir`, else top → −n, bottom → +n, side → the
/// horizontal of +x (+z when n is along x).
pub fn front_dir(c: &Crumble, n: Vec3) -> Vec3 {
    if let Some(d) = c.front_dir {
        return Vec3::from(d).normalize_or_zero();
    }
    match c.front {
        FrontKind::Top => -n,
        FrontKind::Bottom => n,
        FrontKind::Side => {
            let x = horiz(Vec3::X, n);
            if x.length_squared() > 1e-6 { x.normalize() } else { horiz(Vec3::Z, n).normalize() }
        }
    }
}
/// Shatter spin decays with this time constant after first floor contact.
pub const SPIN_DAMP_S: f32 = 0.25;
/// Ripple packet cut-off: splats with ((ρ − speed·τ)/packet_width)² above this are untouched.
pub const RIPPLE_CUT: f32 = 16.0;

/// An orthonormal basis (e1, e2) of the floor plane: e1 = horiz(+x) (or horiz(+z)), e2 = n × e1.
pub fn floor_basis(n: Vec3) -> (Vec3, Vec3) {
    let x = horiz(Vec3::X, n);
    let e1 = if x.length_squared() > 1e-6 { x.normalize() } else { horiz(Vec3::Z, n).normalize() };
    (e1, n.cross(e1))
}

/// A float's rigid motion at `t`: lift along n, lean (object frame, about the centre), yaw about
/// the vertical through `pv`. `None` before `start_s`.
pub(crate) struct FloatState {
    pub lift: f32,
    pub lean: Quat,
    pub yaw: Quat,
    pub pv: Vec3,
}

pub(crate) fn float_state(f: &Float, reg: &Region, t: f32, n: Vec3, _fd: f32) -> Option<FloatState> {
    let tau = t - f.start_s;
    if tau <= 0.0 {
        return None;
    }
    // Up over rise_s, optionally back down from down_s (eased, or hard: x²); spin and bob stop once down.
    let down = f.down_s.map_or(0.0, |d| {
        let x = ((t - d) / f.rise_s).clamp(0.0, 1.0);
        smoothstep01(x) + f.land_hard * (x * x - smoothstep01(x))
    });
    let k = smoothstep01(tau / f.rise_s) * (1.0 - down);
    let ts = f.down_s.map_or(tau, |d| tau.min(d + f.rise_s - f.start_s));
    let lift = k * (f.rise + f.bob * (std::f32::consts::TAU * f.bob_hz * tau).sin());
    let lean = float_tilt(f, n).map_or(Quat::IDENTITY, |axis| Quat::from_axis_angle(axis, f.tilt * k));
    let yaw = Quat::from_axis_angle(n, f.spin * ts * smoothstep01(ts / f.rise_s) + f.turn * float_turn(f, t));
    let pv = reg.centre + f.pivot.map_or(Vec3::ZERO, |v| horiz(Vec3::from(v), n));
    Some(FloatState { lift, lean, yaw, pv })
}

/// Seconds-free lift (units) over which a float's ground shadow fades in after take-off.
pub const FLOAT_SHADOW_LIFT: f32 = 0.08;

/// The darkening (0..shadow) a float casts on a non-member splat at rest position `mu0`: a soft
/// ellipse with the bbox's horizontal half extents (in the object's yawed frame) under its
/// current centre, blurred by 0.05 + shadow_soft·h (h = height of the object's bottom above
/// the splat) and dimmed by the blur's area growth; zero when the splat is above the bottom.
pub(crate) fn float_shadow(f: &Float, reg: &Region, fs: &FloatState, mu0: Vec3, n: Vec3) -> f32 {
    if f.shadow <= 0.0 {
        return 0.0;
    }
    let h = reg.hmin + fs.lift - mu0.dot(n);
    if h < -0.02 {
        return 0.0;
    }
    let h = h.max(0.0);
    let c = fs.pv + fs.yaw * (reg.centre - fs.pv);
    let local = fs.yaw.inverse() * horiz(mu0 - c, n);
    let (e1, e2) = floor_basis(n);
    let half = 0.5 * (reg.hi - reg.lo);
    let (ex, ez) = (half.dot(e1).abs(), half.dot(e2).abs());
    let b = 0.05 + f.shadow_soft * h;
    let (u, v) = (local.dot(e1) / (ex + b), local.dot(e2) / (ez + b));
    let r2 = u * u + v * v;
    if r2 > 9.0 {
        return 0.0;
    }
    let dark = f.shadow * (ex * ez / ((ex + b) * (ez + b))) * (-2.0 * r2).exp() * smoothstep01(fs.lift / FLOAT_SHADOW_LIFT);
    if dark < 1e-4 { 0.0 } else { dark }
}

/// A float's lean axis, `n × horiz(tilt_dir)` (unit), when it leans at all.
pub(crate) fn float_tilt(f: &Float, n: Vec3) -> Option<Vec3> {
    let d = horiz(Vec3::from(f.tilt_dir?), n);
    (f.tilt != 0.0 && d.length_squared() > 1e-12).then(|| n.cross(d.normalize()).normalize())
}

/// A float's flight-turn progress in [0, 1]: smoothstep from mid-rise to mid-descent.
pub(crate) fn float_turn(f: &Float, t: f32) -> f32 {
    let a = f.start_s + 0.5 * f.rise_s;
    let b = f.down_s.map_or(f.start_s + 2.5 * f.rise_s, |d| d + 0.5 * f.rise_s);
    smoothstep01((t - a) / (b - a).max(1e-6))
}

/// `v` with its component along the unit `n` removed.
pub fn horiz(v: Vec3, n: Vec3) -> Vec3 {
    v - v.dot(n) * n
}

/// Rotation taking unit `a` onto unit `b`: normalize(w = 1 + a·b, xyz = a × b).
pub fn rotation_arc(a: Vec3, b: Vec3) -> Quat {
    let c = a.cross(b);
    let q = Quat::from_xyzw(c.x, c.y, c.z, 1.0 + a.dot(b));
    if q.length_squared() < 1e-20 { Quat::IDENTITY } else { q.normalize() }
}

/// Randoms of shatter chunk `k`: the shared pcg chain at indices 0x4000_0000 + 2k and + 2k + 1.
pub fn chunk_randoms(k: u32, seed: u32) -> (SplatRandoms, SplatRandoms) {
    (splat_randoms(0x4000_0000 + 2 * k, seed), splat_randoms(0x4000_0001 + 2 * k, seed))
}

pub(crate) fn chunk_seed(reg: &Region, ca: &SplatRandoms) -> Vec3 {
    reg.lo + (reg.hi - reg.lo) * Vec3::from(ca.r3)
}

/// Chunk contact radius: 0.25 × |bbox diagonal| / K^(1/3).
pub(crate) fn chunk_radius(k: u32, reg: &Region) -> f32 {
    0.25 * (reg.hi - reg.lo).length() / (k as f32).cbrt()
}

/// Nearest Voronoi seed to `p` (ties → lowest k).
pub(crate) fn nearest_seed(seeds: u32, reg: &Region, p: Vec3, seed: u32) -> u32 {
    let mut best = (f32::INFINITY, 0);
    for k in 0..seeds {
        let d = (chunk_seed(reg, &chunk_randoms(k, seed).0) - p).length_squared();
        if d < best.0 {
            best = (d, k);
        }
    }
    best.1
}

pub struct Ballistic {
    pub pos: Vec3,
    /// First floor contact and end of the bounce (start of the slide), in τ.
    pub t_contact: f32,
    pub t_land: f32,
}

/// Closed-form flight of a point launched from `p0` with `v0` under gravity `−g·n`, floor
/// `n·p = d` raised by `rc`: fall to contact, one bounce with restitution `e` (horizontal speed
/// kept), then slide decelerating at `friction·g` to rest. A start below the floor starts on it.
#[allow(clippy::too_many_arguments)]
pub fn ballistic(p0: Vec3, v0: Vec3, tau: f32, n: Vec3, d: f32, g: f32, rc: f32, e: f32, friction: f32) -> Ballistic {
    let y_raw = p0.dot(n) - d - rc;
    let y0 = y_raw.max(0.0);
    let base = p0 - y_raw * n; // the launch point dropped to contact height
    let vy = v0.dot(n);
    let vh = v0 - vy * n;
    let tc = (vy + (vy * vy + 2.0 * g * y0).sqrt()) / g;
    let vb = e * (g * tc - vy);
    let tb = 2.0 * vb / g;
    let t_land = tc + tb;
    let (y, xh) = if tau <= tc {
        (y0 + vy * tau - 0.5 * g * tau * tau, vh * tau)
    } else if tau <= t_land {
        let s = tau - tc;
        (vb * s - 0.5 * g * s * s, vh * tau)
    } else {
        let s = tau - t_land;
        let spd = vh.length();
        let slide = if spd > 1e-9 {
            let stop = spd / (friction * g);
            let s = s.min(stop);
            vh / spd * (spd * s - 0.5 * friction * g * s * s)
        } else {
            Vec3::ZERO
        };
        (0.0, vh * t_land + slide)
    };
    Ballistic { pos: base + y * n + xh, t_contact: tc, t_land }
}

// ---- scalar helpers (ported 1:1 to WGSL/GLSL) ----

/// Linear ramp over `falloff` units outside the region; `falloff <= 0` is a hard step.
pub fn ramp(d: f32, falloff: f32) -> f32 {
    if falloff <= 0.0 {
        if d <= 0.0 { 1.0 } else { 0.0 }
    } else {
        1.0 - (d / falloff).clamp(0.0, 1.0)
    }
}

pub fn smoothstep01(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

pub fn curve_weight(w: &Window, t: f32) -> f32 {
    let span = w.end_s - w.start_s;
    let u = if span > 0.0 {
        ((t - w.start_s) / span).clamp(0.0, 1.0)
    } else if t >= w.start_s {
        1.0
    } else {
        0.0
    };
    match w.curve {
        Curve::Linear => u,
        Curve::Smooth => smoothstep01(u),
        // Exactly 0 at both ends (sin(π) in f32 is ~-8.7e-8, not 0), so the last frame is rest.
        Curve::Pulse if u <= 0.0 || u >= 1.0 => 0.0,
        Curve::Pulse => {
            let s = (std::f32::consts::PI * u).sin();
            s * s
        }
        Curve::Hold => smoothstep01(u / 0.25),
        Curve::Out => 1.0 - smoothstep01(u),
    }
}

/// Dissolve progress of a splat with random `r` at weight `w`: 0 at w = 0, 1 at w = 1, staggered
/// by `r` over a `2·softness` band (the band centre is remapped into [s, 1 - s]).
pub fn dissolve_progress(r: f32, softness: f32, w: f32) -> f32 {
    let c = softness + r * (1.0 - 2.0 * softness);
    if softness <= 0.0 {
        if w > c { 1.0 } else { 0.0 }
    } else {
        smoothstep01((w - (c - softness)) / (2.0 * softness))
    }
}

/// Rodrigues rotation of `v` about the unit `axis` by `theta` (right-handed).
pub fn rotate(v: Vec3, axis: Vec3, theta: f32) -> Vec3 {
    let (s, c) = theta.sin_cos();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

/// New opacity logit after multiplying alpha by `m`. `m == 1` returns `logit` unchanged.
pub fn scale_alpha(logit: f32, m: f32) -> f32 {
    if m == 1.0 {
        return logit;
    }
    let a = (m / (1.0 + (-logit).exp())).clamp(1e-12, 1.0 - 1e-7);
    (a / (1.0 - a)).ln()
}

/// Polar rotation of `j` by 3 Newton steps `R ← (R + R⁻ᵀ)/2`; identity when `|det j| < 1e-6`.
pub fn polar(j: Mat3) -> Mat3 {
    if j.determinant().abs() < 1e-6 {
        return Mat3::IDENTITY;
    }
    let mut r = j;
    for _ in 0..3 {
        r = (r + r.inverse().transpose()) * 0.5;
    }
    r
}

/// Symmetric 3x3 eigendecomposition by cyclic Jacobi (6 sweeps). Returns eigenvalues and the
/// eigenvectors as the columns of a proper rotation (det +1).
pub fn sym_eigen(m: Mat3) -> ([f32; 3], Mat3) {
    let mut a = m.to_cols_array_2d();
    let mut v = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..6 {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[q][p];
            if apq.abs() < 1e-30 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = if theta.abs() > 1e15 {
                0.5 / theta
            } else {
                theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
            };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // A ← Gᵀ A G with the Givens rotation G in the (p, q) plane; a[col][row].
            for k in 0..3 {
                let akp = a[p][k];
                let akq = a[q][k];
                a[p][k] = c * akp - s * akq;
                a[q][k] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[k][p];
                let aqk = a[k][q];
                a[k][p] = c * apk - s * aqk;
                a[k][q] = s * apk + c * aqk;
            }
            for row in &mut v {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    // `v` is row-major here: v[row][col], columns are eigenvectors.
    let mut vm = Mat3::from_cols(
        Vec3::new(v[0][0], v[1][0], v[2][0]),
        Vec3::new(v[0][1], v[1][1], v[2][1]),
        Vec3::new(v[0][2], v[1][2], v[2][2]),
    );
    if vm.determinant() < 0.0 {
        vm.z_axis = -vm.z_axis;
    }
    ([a[0][0], a[1][1], a[2][2]], vm)
}

