//! wgpu evaluator: `effects.wgsl`, the same math as [`crate::eval`], run on brush's own wgpu
//! device over brush's buffers.
//!
//! [`GpuEffect::new`] binds a graph to rest splats already on the device (uploading object
//! weights once). [`GpuEffect::frame`] then dispatches one kernel per frame and returns a
//! [`GpuFrame`] whose `splats` is a `brush_render::Splats` (transforms `[N,10]` = means, quat
//! w,x,y,z, log scales; `raw_opacities` logits; `sh_coeffs` with the tinted DC and, for released
//! crumble particles, the higher bands zeroed) that feeds
//! `brush_render::render_splats` directly. Nothing per-splat crosses to the host per frame; only
//! the ~1 KB program (curve weights, params) goes up.

use brush_cube::{MainBackendBase, create_tensor};
use brush_render::gaussian_splats::SplatRenderMode;
pub use brush_render::gaussian_splats::Splats as GpuSplats;
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{Backend, Dispatch, TensorMetadata, backend_extension, tensor::FloatTensor};
use burn::cubecl::prelude::KernelId;
use burn::cubecl::server::KernelArguments;
use burn::cubecl::{CubeCount, CubeDim};
use burn::module::{Param, ParamId};
use burn::tensor::{DType, Device, Shape, Tensor, TensorData, s};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::template::{KernelSource, SourceKernel, SourceTemplate};
use glam::Vec3;

use crate::eval::{CMask, CNode, COp, CovMode, EvalError, EvalOpts, Frame, Region, chunk_radius, chunk_randoms, chunk_seed, compile, curve_weight, float_tilt, floor_basis, horiz, front_dir, resolve, scene_radius};
use crate::graph::{Graph, MorphTarget, Op};
use crate::Splats;

const WGSL: &str = include_str!("effects.wgsl");
const WORKGROUP: u32 = 256;
const HEADER: usize = 16;
const NODE_STRIDE: usize = 80;
/// Physical nodes' [`Region`] sits at this param index (14 words: lo, hi, centre, foot, hmin, hmax).
const REGION_AT: usize = 14;
/// Crumble's params continue after its region here (front, grain range, wind, turbulence, haze, floor basis).
const CRUMBLE_EXT: usize = REGION_AT + 14;
/// Per-splat aux words: dc rgb, weight, invalid, SH drop.
const AUX: usize = 6;
const MASK_STRIDE: usize = 9;
const CHUNK_STRIDE: usize = 12;
const MAX_MASK_STACK: usize = 8;

struct EffectKernel;

impl KernelSource for EffectKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(WGSL)
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
    fn lang(&self) -> &'static str {
        "wgsl"
    }
}

#[backend_extension(Cube, Fusion)]
pub(crate) trait EffectOps: Backend {
    /// `(transforms [N*10], raw_opacities [N], aux [N*6] = dc rgb, weight, invalid, sh drop)`.
    #[fusion(meta = |tr, op, _sh, _obj, _prog| (
        TensorSpec::new(tr.shape.clone(), DType::F32),
        TensorSpec::new(op.shape.clone(), DType::F32),
        TensorSpec::new(Shape::new([op.shape.num_elements() * AUX]), DType::F32),
    ))]
    fn effect_frame(
        rest_tr: FloatTensor<Self>,
        rest_op: FloatTensor<Self>,
        rest_sh: FloatTensor<Self>,
        objects: FloatTensor<Self>,
        prog: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}

impl EffectOps for MainBackendBase {
    fn effect_frame(
        rest_tr: FloatTensor<Self>,
        rest_op: FloatTensor<Self>,
        rest_sh: FloatTensor<Self>,
        objects: FloatTensor<Self>,
        prog: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let [rest_tr, rest_op, rest_sh, objects, prog] = [rest_tr, rest_op, rest_sh, objects, prog].map(into_contiguous);
        let n = rest_op.shape().num_elements();
        assert_eq!(rest_tr.shape().num_elements(), n * 10, "transforms must be [N, 10]");
        let device = rest_tr.device.clone();
        let out_tr = create_tensor([n, 10], &device, DType::F32);
        let out_op = create_tensor([n], &device, DType::F32);
        let out_x = create_tensor([n * AUX], &device, DType::F32);
        let groups = (n as u32).div_ceil(WORKGROUP).max(1);
        let (gx, gy) = if groups <= 65_535 { (groups, 1) } else { (65_535, groups.div_ceil(65_535)) };
        rest_tr.client.launch(
            Box::new(SourceKernel::new(EffectKernel, CubeDim::new_1d(WORKGROUP))),
            CubeCount::Static(gx, gy, 1),
            KernelArguments::new().with_buffers(vec![
                rest_tr.handle.clone().binding(),
                rest_op.handle.clone().binding(),
                rest_sh.handle.clone().binding(),
                objects.handle.clone().binding(),
                prog.handle.clone().binding(),
                out_tr.handle.clone().binding(),
                out_op.handle.clone().binding(),
                out_x.handle.clone().binding(),
            ]),
        );
        (out_tr, out_op, out_x)
    }
}

/// A graph bound to rest splats on the GPU.
pub struct GpuEffect {
    nodes: Vec<CNode>,
    /// Static tail of the program: mask programs, then shatter chunk tables.
    statics: Vec<f32>,
    mask_offsets: Vec<u32>,
    chunk_offsets: Vec<u32>,
    objects: Tensor<1>,
    seed: u32,
    duration: f32,
    looping: bool,
    h: f32,
    cov: CovMode,
    n: usize,
    has_tint: bool,
    /// A crumble node is present: released particles drop their higher SH bands.
    has_crumble: bool,
    floor_n: Vec3,
    floor_d: f32,
    gravity: f32,
    aniso_max: f32,
}

/// One frame on the GPU.
pub struct GpuFrame {
    /// Render-ready splats (same render mode, SH and min-scale as the rest).
    pub splats: GpuSplats,
    /// `[N, 6]`: DC rgb coefficient, max node weight, invalid flag (0/1), SH drop flag (0/1).
    pub aux: Tensor<2>,
}

impl GpuFrame {
    pub fn weight(&self) -> Tensor<1> {
        let n = self.aux.dims()[0];
        self.aux.clone().slice(s![.., 3..4]).reshape([n])
    }
    /// `(invalid, selected)` counts. Reads back two numbers.
    pub fn counts(&self) -> (usize, usize) {
        let inv = self.aux.clone().slice(s![.., 4..5]).sum();
        let sel = self.weight().greater_elem(0.0).float().sum();
        let v: Vec<f32> = Tensor::cat(vec![inv, sel], 0).into_data().try_to_vec::<f32>().expect("f32 counts");
        (v[0] as usize, v[1] as usize)
    }
}

/// Upload CPU splats to `device` in brush's layout.
pub fn upload(rest: &Splats, device: &Device) -> GpuSplats {
    let n = rest.len();
    let k = rest.sh_coeffs();
    let mut coeffs = Vec::with_capacity(n * (k + 1) * 3);
    for i in 0..n {
        coeffs.extend_from_slice(&rest.dc[i]);
        coeffs.extend_from_slice(&rest.rest[i * k * 3..(i + 1) * k * 3]);
    }
    let mode = if rest.antialiased { SplatRenderMode::Mip } else { SplatRenderMode::Default };
    GpuSplats::from_raw(
        rest.pos.iter().flatten().copied().collect(),
        rest.rot.iter().flatten().copied().collect(),
        rest.scale.iter().flatten().copied().collect(),
        coeffs,
        rest.opacity.clone(),
        mode,
        device,
    )
}

impl GpuEffect {
    /// Bind `graph` to `rest`. Object weights (`opts.objects`) are uploaded here, once.
    /// A graph not yet [`resolve`]d (physical bboxes, pile radius, floor, gravity) reads the
    /// means back once to resolve it; otherwise, without `opts.scene_radius` or a graph
    /// `jacobian_step`, only the means' bounds are read back.
    pub fn new(rest: &GpuSplats, graph: &Graph, opts: &EvalOpts) -> Result<Self, EvalError> {
        let n = rest.num_splats() as usize;
        let device = rest.device();
        let host = if is_resolved(graph) {
            Splats::default()
        } else {
            let flat: Vec<f32> = rest.means().into_data().try_to_vec::<f32>().expect("f32 means");
            // The DC term too: crumble resolves its palette from the members' base colour.
            let dc: Vec<f32> = rest.sh_coeffs.val().slice(s![.., 0..1, ..]).into_data().try_to_vec::<f32>().expect("f32 dc");
            Splats {
                pos: flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
                dc: dc.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
                ..Splats::default()
            }
        };
        let graph = &resolve(&host, graph, opts)?;
        let (nodes, objects) = compile(graph, opts, n)?;
        let objects = if objects.is_empty() {
            Tensor::<1>::zeros([1], &device)
        } else {
            let flat: Vec<f32> = objects.iter().flat_map(|w| w.iter().copied()).collect();
            let len = flat.len();
            Tensor::<1>::from_data(TensorData::new(flat, [len]), &device)
        };
        let h = match (graph.jacobian_step, opts.scene_radius) {
            (Some(h), _) => h,
            (None, Some(r)) => 1e-3 * r,
            (None, None) if !host.is_empty() => 1e-3 * scene_radius(&host.pos),
            (None, None) => 1e-3 * device_scene_radius(rest),
        };
        let mut statics = Vec::new();
        let mut mask_offsets = Vec::new();
        let base = HEADER + NODE_STRIDE * nodes.len();
        for node in &nodes {
            mask_offsets.push((base + statics.len()) as u32);
            let mut depth = 0;
            let mut max_depth = 0;
            emit_mask(&node.mask, &mut statics, &mut depth, &mut max_depth);
            statics.push(f32::from_bits(0));
            statics.extend([0.0; MASK_STRIDE - 1]);
            if max_depth > MAX_MASK_STACK {
                return Err(EvalError::MaskTooDeep { depth: max_depth, max: MAX_MASK_STACK });
            }
        }
        let floor = graph.floor.expect("resolved");
        let floor_n = Vec3::from(floor.normal).normalize_or_zero();
        // Shatter chunk tables (per chunk k): seed point, ca.r, ca.axis, cb.r, unit spin axis.
        let mut chunk_offsets = Vec::new();
        for node in &nodes {
            chunk_offsets.push((base + statics.len()) as u32);
            if let COp::Shatter(sh, reg) = &node.op {
                for k in 0..sh.seeds.max(1) {
                    let start = statics.len();
                    let (ca, cb) = chunk_randoms(k, graph.seed);
                    let a = Vec3::from(cb.axis);
                    let axis = if a.length_squared() > 1e-12 { a.normalize() } else { floor_n };
                    push3(&mut statics, chunk_seed(reg, &ca));
                    statics.push(ca.r);
                    statics.extend(ca.axis);
                    statics.push(cb.r);
                    push3(&mut statics, axis);
                    statics.push(0.0);
                    debug_assert_eq!(statics.len() - start, CHUNK_STRIDE);
                }
            }
        }
        // Any node that changes colour: tint, ripple slope shading, crumble dust, melt sheen, float shadow.
        let has_tint = nodes.iter().any(|n| matches!(n.op, COp::Tint { .. } | COp::Ripple(..) | COp::Crumble(..) | COp::Melt(..)) || matches!(&n.op, COp::Float(f, _) if f.shadow > 0.0));
        let has_crumble = nodes.iter().any(|n| matches!(n.op, COp::Crumble(..)));
        Ok(Self {
            nodes,
            statics,
            mask_offsets,
            chunk_offsets,
            objects,
            seed: graph.seed,
            duration: graph.duration_s,
            looping: graph.looping,
            h,
            cov: opts.cov,
            n,
            has_tint,
            has_crumble,
            floor_n,
            floor_d: floor.offset,
            gravity: graph.gravity.expect("resolved"),
            aniso_max: graph.aniso_max.expect("resolved"),
        })
    }

    pub fn jacobian_step(&self) -> f32 {
        self.h
    }

    /// The per-frame program buffer (header, node table, mask programs).
    fn program(&self, t: f32, sh_coeffs: usize) -> Vec<f32> {
        let t = if self.looping { t.rem_euclid(self.duration) } else { t };
        let u = f32::from_bits;
        let mut p = vec![
            u(self.nodes.len() as u32),
            u(self.seed),
            self.h,
            t,
            u(u32::from(self.cov == CovMode::Fast)),
            u(self.n as u32),
            u(sh_coeffs as u32),
            self.duration,
            self.floor_n.x,
            self.floor_n.y,
            self.floor_n.z,
            self.floor_d,
            self.gravity,
            self.aniso_max,
            0.0,
            0.0,
        ];
        debug_assert_eq!(p.len(), HEADER);
        for ((node, off), chunks) in self.nodes.iter().zip(&self.mask_offsets).zip(&self.chunk_offsets) {
            let cw = node.window.map_or(1.0, |w| curve_weight(&w, t));
            let mut rec = [0.0f32; NODE_STRIDE];
            let (ty, params) = op_params(&node.op, *chunks, self.floor_n, self.floor_d);
            rec[0] = u(ty);
            rec[1] = cw;
            rec[2] = u(*off);
            rec[4..4 + params.len()].copy_from_slice(&params);
            p.extend_from_slice(&rec);
        }
        p.extend_from_slice(&self.statics);
        p
    }

    /// Evaluate one frame at `t` seconds from `rest` (which must be the splats given to `new`).
    pub fn frame(&self, rest: &GpuSplats, t: f32) -> GpuFrame {
        let device = rest.device();
        let sh = rest.sh_coeffs.val();
        let [n, c, _] = sh.dims();
        assert_eq!(n, self.n, "rest splat count changed since GpuEffect::new");
        let prog = self.program(t, c);
        let len = prog.len();
        let prog = Tensor::<1>::from_data(TensorData::new(prog, [len]), &device);
        let (tr, op, x) = Dispatch::effect_frame(
            rest.transforms.val().into_dispatch(),
            rest.raw_opacities.val().into_dispatch(),
            sh.clone().into_dispatch(),
            self.objects.clone().into_dispatch(),
            prog.into_dispatch(),
        );
        let tr = Tensor::<2>::from_dispatch(tr);
        let op = Tensor::<1>::from_dispatch(op);
        let aux = Tensor::<1>::from_dispatch(x).reshape([n, AUX]);
        let mut splats = rest.clone();
        splats.transforms = Param::initialized(ParamId::new(), tr);
        splats.raw_opacities = Param::initialized(ParamId::new(), op);
        let tint = self.has_tint && rest.decoder.is_none();
        let drop = self.has_crumble && c > 1 && rest.decoder.is_none();
        if tint || drop {
            let mut sh = sh;
            if tint {
                let dc = aux.clone().slice(s![.., 0..3]).reshape([n, 1, 3]);
                sh = sh.slice_assign(s![.., 0..1, ..], dc);
            }
            if drop {
                // Released particles render with the DC colour only (eval.rs Frame::sh_drop).
                let flag = aux.clone().slice(s![.., 5..6]).greater_elem(0.5).reshape([n, 1, 1]).expand([n, c - 1, 3]);
                let bands = sh.clone().slice(s![.., 1..c, ..]).mask_fill(flag, 0.0);
                sh = sh.slice_assign(s![.., 1..c, ..], bands);
            }
            splats.sh_coeffs = Param::initialized(ParamId::new(), sh);
        }
        GpuFrame { splats, aux }
    }
}

/// Read a GPU frame back as a CPU [`Frame`] (tests, apply/export paths).
pub fn download(frame: &GpuFrame) -> Frame {
    let tr: Vec<f32> = frame.splats.transforms.val().into_data().try_to_vec::<f32>().expect("f32");
    let op: Vec<f32> = frame.splats.raw_opacities.val().into_data().try_to_vec::<f32>().expect("f32");
    let x: Vec<f32> = frame.aux.clone().into_data().try_to_vec::<f32>().expect("f32");
    let n = op.len();
    let mut f = Frame::default();
    for i in 0..n {
        let t = &tr[i * 10..i * 10 + 10];
        f.pos.push([t[0], t[1], t[2]]);
        f.rot.push([t[3], t[4], t[5], t[6]]);
        f.scale.push([t[7], t[8], t[9]]);
        let a = &x[i * AUX..i * AUX + AUX];
        f.dc.push([a[0], a[1], a[2]]);
        f.weight.push(a[3]);
        f.invalid += usize::from(a[4] > 0.5);
        f.selected += usize::from(a[3] > 0.0);
        f.sh_drop.push(a[5] > 0.5);
    }
    f.opacity = op;
    f
}

fn device_scene_radius(rest: &GpuSplats) -> f32 {
    let means = rest.means();
    let lo = means.clone().min_dim(0);
    let hi = means.max_dim(0);
    let v: Vec<f32> = Tensor::cat(vec![lo, hi], 1).into_data().try_to_vec::<f32>().expect("f32 bounds");
    let r = 0.5 * (Vec3::new(v[3], v[4], v[5]) - Vec3::new(v[0], v[1], v[2])).length();
    if r.is_finite() && r > 0.0 { r } else { 1.0 }
}

fn push3(out: &mut Vec<f32>, v: Vec3) {
    out.extend([v.x, v.y, v.z]);
}

fn emit_mask(m: &CMask, out: &mut Vec<f32>, depth: &mut usize, max_depth: &mut usize) {
    let u = f32::from_bits;
    let start = out.len();
    let push = |depth: &mut usize, max_depth: &mut usize| {
        *depth += 1;
        *max_depth = (*max_depth).max(*depth);
    };
    match m {
        CMask::All => {
            out.push(u(1));
            push(depth, max_depth);
        }
        CMask::Box { min, max, falloff } => {
            out.push(u(2));
            push3(out, *min);
            push3(out, *max);
            out.push(*falloff);
            push(depth, max_depth);
        }
        CMask::Sphere { center, radius, falloff } => {
            out.push(u(3));
            push3(out, *center);
            out.extend([*radius, *falloff]);
            push(depth, max_depth);
        }
        CMask::Height { axis, min, max, falloff } => {
            out.push(u(4));
            push3(out, *axis);
            out.extend([*min, *max, *falloff]);
            push(depth, max_depth);
        }
        CMask::Object { slot } => {
            out.push(u(5));
            out.push(u(*slot as u32));
            push(depth, max_depth);
        }
        CMask::Invert(inner) => {
            emit_mask(inner, out, depth, max_depth);
            out.push(u(6));
        }
        CMask::Mul(ms) => {
            for m in ms {
                emit_mask(m, out, depth, max_depth);
            }
            out.push(u(7));
            out.push(u(ms.len() as u32));
            *depth = *depth + 1 - ms.len();
        }
    }
    // Pad this instruction to MASK_STRIDE words (children already are).
    while (out.len() - start) % MASK_STRIDE != 0 {
        out.push(0.0);
    }
}

/// Whether every derived value [`resolve`] fills is already present (no means readback needed).
fn is_resolved(g: &Graph) -> bool {
    g.floor.is_some()
        && g.gravity.is_some()
        && g.aniso_max.is_some()
        && g.nodes.iter().all(|n| match &n.op {
            Op::Crumble(c) => c.bbox.is_some() && c.pile_r.is_some() && c.palette.is_some(),
            Op::Shatter(s) => s.bbox.is_some(),
            Op::Float(f) => f.bbox.is_some(),
            Op::Melt(m) => m.bbox.is_some(),
            Op::Ripple(r) => !r.rigid || r.bbox.is_some(),
            _ => true,
        })
}

/// Pad `p` to [`REGION_AT`] and append the region.
fn push_region(p: &mut Vec<f32>, reg: &Region) {
    p.resize(REGION_AT, 0.0);
    push3(p, reg.lo);
    push3(p, reg.hi);
    push3(p, reg.centre);
    push3(p, reg.foot);
    p.extend([reg.hmin, reg.hmax]);
}

/// A node's type code and params; `chunks` is its shatter chunk table offset, `floor_n` and
/// `floor_d` the floor plane (crumble's front direction and heap basis and a rigid ripple's
/// pivot are precomputed from them).
fn op_params(op: &COp, chunks: u32, floor_n: Vec3, floor_d: f32) -> (u32, Vec<f32>) {
    let u = f32::from_bits;
    let mut p = Vec::with_capacity(NODE_STRIDE - 4);
    let ty = match op {
        COp::Vortex { center, axis, angle, angle_per_unit, lift, expand } => {
            push3(&mut p, *center);
            push3(&mut p, *axis);
            p.extend([*angle, *angle_per_unit, *lift, *expand]);
            1
        }
        COp::Wave { direction, wavelength, speed, amplitude, axis } => {
            push3(&mut p, *direction);
            p.extend([*wavelength, *speed, *amplitude]);
            push3(&mut p, *axis);
            2
        }
        COp::Explode { center, distance, spin, jitter } => {
            push3(&mut p, *center);
            p.extend([*distance, *spin, *jitter]);
            3
        }
        COp::Dissolve { direction, drift, softness, shrink } => {
            push3(&mut p, *direction);
            p.extend([*drift, *softness, *shrink]);
            4
        }
        COp::Morph { target, center, radius, axis } => {
            p.push(f32::from_bits(match target {
                MorphTarget::Sphere => 0,
                MorphTarget::Cylinder => 1,
                MorphTarget::Plane => 2,
            }));
            push3(&mut p, *center);
            p.push(*radius);
            push3(&mut p, *axis);
            5
        }
        COp::Opacity { to } => {
            p.push(*to);
            6
        }
        COp::Tint { color, amount } => {
            push3(&mut p, *color);
            p.push(*amount);
            7
        }
        COp::Crumble(c, reg) => {
            p.extend([c.start_s, c.sweep_s, c.burst, c.up_kick, c.grain, c.restitution, c.friction, c.settle_s, c.pile_h]);
            p.extend([c.pile_r.expect("resolved"), c.dust, u(u32::from(c.reverse)), c.jitter, c.shadow]);
            push_region(&mut p, reg);
            debug_assert_eq!(p.len(), CRUMBLE_EXT);
            // Front: unit travel direction and the region's projected range along it.
            let d = front_dir(c, floor_n);
            let reach = (0.5 * (reg.hi - reg.lo)).dot(d.abs());
            push3(&mut p, d);
            p.extend([reg.centre.dot(d) - reach, reg.centre.dot(d) + reach, c.size_min, c.size_max]);
            p.extend(c.wind);
            p.extend([c.turbulence, c.turb_freq, c.haze, c.haze_size, c.haze_alpha, c.haze_s, c.haze_rise]);
            let (e1, e2) = floor_basis(floor_n);
            push3(&mut p, e1);
            push3(&mut p, e2);
            p.extend([c.gather, c.dust_desat, c.heap_ao, c.grain_opacity, c.front_noise, c.front_freq, c.chunk_frac]);
            push3(&mut p, Vec3::from(c.palette.expect("resolved")));
            p.extend([c.palette_tol, c.grain_jitter, c.land_shrink]);
            8
        }
        COp::Shatter(sh, reg) => {
            p.extend([u(sh.seeds), sh.start_s, sh.burst, sh.up_kick, sh.spin, sh.restitution, sh.friction]);
            p.extend([chunk_radius(sh.seeds, reg), u(chunks)]);
            push_region(&mut p, reg);
            9
        }
        COp::Ripple(r, light) => {
            p.extend(r.source);
            p.extend([r.start_s, r.speed, r.wavelength, r.amplitude, r.decay, r.packet_width]);
            push3(&mut p, *light);
            p.push(r.shade);
            // Rigid object mode: the bbox centre projected onto the floor (eval.rs physical).
            let pivot = r.bbox.filter(|_| r.rigid).map(|b| {
                let c = 0.5 * (Vec3::from(b.min) + Vec3::from(b.max));
                c - (c.dot(floor_n) - floor_d) * floor_n
            });
            p.extend([u(u32::from(r.rigid)), u(u32::from(pivot.is_some()))]);
            push3(&mut p, pivot.unwrap_or(Vec3::ZERO));
            10
        }
        COp::Float(f, reg) => {
            p.extend([f.start_s, f.rise, f.rise_s, f.spin, f.bob, f.bob_hz, u(u32::from(f.down_s.is_some())), f.down_s.unwrap_or(0.0)]);
            // Lean: angle (0 = none) and the precomputed unit axis n × horiz(tilt_dir).
            let axis = float_tilt(f, floor_n);
            p.push(if axis.is_some() { f.tilt } else { 0.0 });
            push3(&mut p, axis.unwrap_or(Vec3::ZERO));
            push_region(&mut p, reg);
            // Flight (after the region, at FLOAT_FLIGHT_AT): turn, its window, the pivot offset.
            let a = f.start_s + 0.5 * f.rise_s;
            let b = f.down_s.map_or(f.start_s + 2.5 * f.rise_s, |d| d + 0.5 * f.rise_s);
            p.extend([f.turn, a, (b - a).max(1e-6)]);
            push3(&mut p, f.pivot.map_or(Vec3::ZERO, |v| horiz(Vec3::from(v), floor_n)));
            // Hard landing, ground shadow (strength, blur growth, floor basis, footprint half extents).
            p.extend([f.land_hard, f.shadow, f.shadow_soft]);
            let (e1, e2) = floor_basis(floor_n);
            push3(&mut p, e1);
            push3(&mut p, e2);
            let half = 0.5 * (reg.hi - reg.lo);
            p.extend([half.dot(e1).abs(), half.dot(e2).abs()]);
            11
        }
        COp::Melt(m, reg) => {
            p.extend([m.start_s, m.duration_s, m.spread, m.sheen]);
            push_region(&mut p, reg);
            12
        }
    };
    debug_assert!(p.len() <= NODE_STRIDE - 4);
    (ty, p)
}

/// The wgpu device for `kind`: "cpu" is the software adapter (lavapipe/llvmpipe, works without a
/// GPU), anything else the default adapter.
pub fn device(kind: &str) -> Device {
    use burn::cubecl::wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind};
    // Pinned to Vulkan: `Auto` skips a Vulkan that offers only lavapipe and moves on to GL.
    let d = if kind == "cpu" { WgpuDevice::new(WgpuDeviceKind::Cpu).on(WgpuBackend::Vulkan) } else { WgpuDevice::default() };
    d.into()
}

/// One CPU-API frame through the GPU evaluator: upload, dispatch, download.
pub fn evaluate_via_gpu(rest: &Splats, graph: &Graph, t: f32, opts: &EvalOpts, device: &Device) -> Result<Frame, EvalError> {
    let g = upload(rest, device);
    let fx = GpuEffect::new(&g, graph, opts)?;
    Ok(download(&fx.frame(&g, t)))
}
