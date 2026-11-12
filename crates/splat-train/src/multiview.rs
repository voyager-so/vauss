//! Multi-view geometric and photometric consistency for surfel (mesh-goal)
//! training, after PGSR (clean-room note docs/research/lit/pgsr-clean-room.md §5).
//!
//! Each step after the start, one neighbor training camera (picked
//! pseudo-randomly among up to 8 within 30 degrees) is rendered too.
//! - geometric, on a random pixel sample (`surfel_mv_ncc_samples`): primary surface point -> neighbor pixel ->
//!   neighbor depth -> back to the primary image; the pixel round-trip error
//!   `e` is weighted by detached `exp(-e)`, valid where the depths are > .01,
//!   the neighbor point is in front (z > .1) and in frame, and `e < 1`;
//! - photometric, on the same samples (PGSR: 102,400): the 7x7 patch around the
//!   pixel is lifted onto the primary's rendered plane (normal, surface depth),
//!   projected into the neighbor and compared to the primary's patch on the
//!   ground-truth grayscale; score `1 - cov^2 / (var var)`, patches scoring
//!   >= .9 dropped, weighted by the geometric `exp(-e)`.
//! All of it is plain differentiable tensor ops (gather-based bilinear
//! sampling, so neighbor depth and sample coordinates both get gradients).

use brush_render::camera::Camera;
use burn::tensor::{Int, Tensor, TensorData};
use glam::{Mat3, UVec2, Vec3};

/// Patch half-size (7x7).
const PATCH_R: i32 = 3;
/// Neighbors per view: the nearest camera centers with view directions
/// within 30 degrees, at most 8.
const NEIGHBORS: usize = 8;
const MIN_VIEW_DOT: f32 = 0.866_025_4;

/// Training views for the consistency losses: cameras plus a grayscale
/// (luma, [0, 1]) copy of each training image at its loaded size.
pub struct MultiView {
    cams: Vec<Camera>,
    grays: Vec<(Tensor<2>, UVec2)>,
    neighbors: Vec<Vec<usize>>,
}

impl MultiView {
    pub fn new(cams: Vec<Camera>, grays: Vec<(Tensor<2>, UVec2)>) -> Self {
        let fwd = |c: &Camera| c.rotation * Vec3::Z;
        let neighbors = cams
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let mut cand: Vec<(f32, f32, usize)> = cams
                    .iter()
                    .enumerate()
                    .filter(|&(j, b)| j != i && fwd(a).dot(fwd(b)) >= MIN_VIEW_DOT)
                    .map(|(j, b)| (a.position.distance(b.position), -fwd(a).dot(fwd(b)), j))
                    .collect();
                cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)));
                cand.into_iter().take(NEIGHBORS).map(|(_, _, j)| j).collect()
            })
            .collect();
        Self { cams, grays, neighbors }
    }

    /// Index of the training view rendered from `cam` (exact pose match).
    pub fn view_of(&self, cam: &Camera) -> Option<usize> {
        self.cams
            .iter()
            .position(|c| c.position.distance(cam.position) < 1e-5 && c.rotation.dot(cam.rotation).abs() > 0.99999)
    }

    /// The neighbor to render for `view` at `step` (pseudo-random pick).
    pub fn neighbor(&self, view: usize, step: u32) -> Option<&Camera> {
        (!self.neighbors[view].is_empty()).then(|| &self.cams[self.neighbor_index(view, step)])
    }

    fn neighbor_index(&self, view: usize, step: u32) -> usize {
        let n = &self.neighbors[view];
        n[(hash(step as u64 ^ ((view as u64) << 32)) % n.len() as u64) as usize]
    }
}

/// SplitMix64: a cheap deterministic per-step random stream.
fn hash(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Intrinsics at `size`: (fx, fy, cx, cy) in pixels.
fn intrinsics(cam: &Camera, size: UVec2) -> [f32; 4] {
    let f = cam.focal(size);
    let c = cam.center(size);
    [f.x, f.y, c.x, c.y]
}

/// Bilinear sample of a `[H, W]` map at continuous pixel coordinates
/// (pixel `i` has its center at `i + 0.5`), clamped to the border. Gradients
/// reach both the map values and the coordinates.
fn bilinear(map: Tensor<2>, x: Tensor<1>, y: Tensor<1>) -> Tensor<1> {
    let [h, w] = map.dims();
    let flat = map.reshape([h * w]);
    let gx = x - 0.5;
    let gy = y - 0.5;
    let x0 = gx.clone().detach().floor().clamp(0.0, (w - 1) as f32);
    let y0 = gy.clone().detach().floor().clamp(0.0, (h - 1) as f32);
    let x1 = (x0.clone() + 1.0).clamp_max((w - 1) as f32);
    let y1 = (y0.clone() + 1.0).clamp_max((h - 1) as f32);
    let fx = (gx - x0.clone()).clamp(0.0, 1.0);
    let fy = (gy - y0.clone()).clamp(0.0, 1.0);
    let idx = |xx: Tensor<1>, yy: Tensor<1>| (yy * w as f32 + xx).int();
    // gather, not select: select's backward runs one GPU thread per
    // non-indexed row, i.e. serially for a 1-D map.
    let at = |i: Tensor<1, Int>| flat.clone().gather(0, i);
    let v00 = at(idx(x0.clone(), y0.clone()));
    let v10 = at(idx(x1.clone(), y0));
    let v01 = at(idx(x0, y1.clone()));
    let v11 = at(idx(x1, y1));
    let one_fx = fx.clone().neg() + 1.0;
    let top = v00 * one_fx.clone() + v10 * fx.clone();
    let bot = v01 * one_fx + v11 * fx;
    top * (fy.clone().neg() + 1.0) + bot * fy
}

/// `[x, y, z]` camera-space components.
type V3 = [Tensor<1>; 3];

fn mat_apply(m: Mat3, v: &V3, t: Vec3) -> V3 {
    let row = |r: Vec3, o: f32| v[0].clone() * r.x + v[1].clone() * r.y + v[2].clone() * r.z + o;
    let mt = m.transpose(); // rows of m
    [row(mt.x_axis, t.x), row(mt.y_axis, t.y), row(mt.z_axis, t.z)]
}

/// Weighted consistency losses `(geometric, photometric)` for one step.
///
/// `primary_geom` / `neighbor_geom` are the surfel geometry channels
/// (`GEOM_OUT_CHANNELS`).
#[allow(clippy::too_many_arguments)]
pub fn consistency_losses(
    mv: &MultiView,
    view: usize,
    step: u32,
    size: UVec2,
    primary_geom: Tensor<3>,
    neighbor_geom: Tensor<3>,
    ncc_samples: usize,
) -> (Tensor<1>, Tensor<1>) {
    let device = primary_geom.device();
    let (w, h) = (size.x as usize, size.y as usize);
    let n = w * h;
    let cam1 = &mv.cams[view];
    let nidx = mv.neighbor_index(view, step);
    let cam2 = &mv.cams[nidx];
    let [fx1, fy1, cx1, cy1] = intrinsics(cam1, size);
    let [fx2, fy2, cx2, cy2] = intrinsics(cam2, size);

    // A random subset of primary pixels (without replacement); PGSR takes
    // the geometric mean over every pixel, this is its unbiased estimate.
    let m = ncc_samples.clamp(1, n);
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut rng = hash(0x5eed ^ step as u64 ^ ((view as u64) << 40));
    for i in 0..m {
        rng = hash(rng);
        let j = i + (rng % (n - i) as u64) as usize;
        perm.swap(i, j);
    }
    let (mut px, mut py, mut flat_idx) = (Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m));
    for &i in &perm[..m] {
        px.push((i as usize % w) as f32 + 0.5);
        py.push((i as usize / w) as f32 + 0.5);
        flat_idx.push(i as i64);
    }
    let f1 = |v: Vec<f32>| Tensor::<1>::from_data(TensorData::new(v, [m]), &device);
    let u1 = f1(px);
    let v1 = f1(py);
    let sel = Tensor::<1, Int>::from_data(TensorData::new(flat_idx, [m]), &device);
    let gather = |t: Tensor<3>, c: usize| t.slice([0..h, 0..w, c..c + 1]).reshape([n]).gather(0, sel.clone());

    let d1 = gather(primary_geom.clone(), 1);
    let nrm: V3 = [gather(primary_geom.clone(), 2), gather(primary_geom.clone(), 3), gather(primary_geom, 4)];
    let nlen = (nrm[0].clone().powi_scalar(2) + nrm[1].clone().powi_scalar(2) + nrm[2].clone().powi_scalar(2) + 1e-12)
        .sqrt()
        .clamp_min(1e-6);
    let nlen_ok = nlen.clone().detach().greater_elem(1e-3).float();
    let nrm: V3 = [nrm[0].clone() / nlen.clone(), nrm[1].clone() / nlen.clone(), nrm[2].clone() / nlen];

    // Primary camera ray and point.
    let r1: V3 = [(u1.clone() - cx1) / fx1, (v1.clone() - cy1) / fy1, Tensor::ones([m], &device)];
    let x1: V3 = [r1[0].clone() * d1.clone(), r1[1].clone() * d1.clone(), d1.clone()];

    // cam1 -> cam2: X2 = M X1 + t with M = R2^T R1, t = R2^T (p1 - p2).
    let r1m = Mat3::from_quat(cam1.rotation);
    let r2m = Mat3::from_quat(cam2.rotation);
    let m12 = r2m.transpose() * r1m;
    let t12 = r2m.transpose() * (cam1.position - cam2.position);
    let m21 = m12.transpose();
    let t21 = -(m21 * t12);

    let x2 = mat_apply(m12, &x1, t12);
    let z2 = x2[2].clone();
    let z2s = z2.clone().clamp_min(1e-6);
    let u2 = x2[0].clone() / z2s.clone() * fx2 + cx2;
    let v2 = x2[1].clone() / z2s * fy2 + cy2;

    let d2 = bilinear(neighbor_geom.slice([0..h, 0..w, 1..2]).reshape([h, w]), u2.clone(), v2.clone());

    // Back from the neighbor's surface to the primary image.
    let r2: V3 = [(u2.clone() - cx2) / fx2, (v2.clone() - cy2) / fy2, Tensor::ones([m], &device)];
    let x2b: V3 = [r2[0].clone() * d2.clone(), r2[1].clone() * d2.clone(), d2.clone()];
    let x1b = mat_apply(m21, &x2b, t21);
    let z1b = x1b[2].clone().clamp_min(1e-6);
    let u1b = x1b[0].clone() / z1b.clone() * fx1 + cx1;
    let v1b = x1b[1].clone() / z1b * fy1 + cy1;
    let err = ((u1b - u1.clone()).powi_scalar(2) + (v1b - v1.clone()).powi_scalar(2) + 1e-12).sqrt();

    let errd = err.clone().detach();
    let in_frame = u2.clone().detach().greater_equal_elem(0.0).float()
        * u2.clone().detach().lower_elem(w as f32).float()
        * v2.clone().detach().greater_equal_elem(0.0).float()
        * v2.clone().detach().lower_elem(h as f32).float();
    let mask = nlen_ok
        * d1.clone().detach().greater_elem(0.01).float()
        * z2.detach().greater_elem(0.1).float()
        * d2.clone().detach().greater_elem(0.01).float()
        * in_frame
        * errd.clone().lower_elem(1.0).float();
    let weight = mask.clone() * errd.neg().exp();
    let geometric = (weight.clone() * err).sum() / mask.clone().sum().clamp_min(1.0);

    // Photometric on the same samples.
    let (u1s, v1s, nrs, x1s) = (u1, v1, nrm, x1);
    let ws = weight.clone().detach();
    let ms = mask.clone().detach();

    let (g1, s1) = &mv.grays[view];
    let (g2, s2) = &mv.grays[nidx];
    // Grayscale observations live on the inner device; lift them (no grad).
    let g1 = g1.clone().autodiff();
    let g2 = g2.clone().autodiff();
    let sc1 = (s1.x as f32 / size.x as f32, s1.y as f32 / size.y as f32);
    let sc2 = (s2.x as f32 / size.x as f32, s2.y as f32 / size.y as f32);
    let dist = nrs[0].clone() * x1s[0].clone() + nrs[1].clone() * x1s[1].clone() + nrs[2].clone() * x1s[2].clone();
    let side = (2 * PATCH_R + 1) as usize;
    let k = side * side;
    // All 49 offsets at once: [m, k] tensors, flattened for sampling.
    let (mut ox, mut oy) = (Vec::with_capacity(k), Vec::with_capacity(k));
    for dy in -PATCH_R..=PATCH_R {
        for dx in -PATCH_R..=PATCH_R {
            ox.push(dx as f32);
            oy.push(dy as f32);
        }
    }
    let off = |v: Vec<f32>| Tensor::<2>::from_data(TensorData::new(v, [1, k]), &device);
    let col = |t: Tensor<1>| t.reshape([m, 1]);
    let pu = col(u1s) + off(ox); // [m, k]
    let pv = col(v1s) + off(oy);
    let flat = |t: Tensor<2>| t.reshape([m * k]);
    let a = bilinear(g1, flat(pu.clone()) * sc1.0, flat(pv.clone()) * sc1.1)
        .detach()
        .reshape([m, k]);
    let rx = (pu - cx1) / fx1;
    let ry = (pv - cy1) / fy1;
    let denom = col(nrs[0].clone()) * rx.clone() + col(nrs[1].clone()) * ry.clone() + col(nrs[2].clone());
    // Keep every lane finite (masked lanes still flow through the graph):
    // grazing or empty-normal rays get a floored denominator and are masked.
    let sign = denom.clone().detach().greater_equal_elem(0.0).float() * 2.0 - 1.0;
    let denom_safe = denom.clone().abs().clamp_min(1e-4) * sign;
    let z = col(dist) / denom_safe;
    let xp: V3 = [flat(rx * z.clone()), flat(ry * z.clone()), flat(z.clone())];
    let xq = mat_apply(m12, &xp, t12);
    let zq = xq[2].clone();
    let zqs = zq.clone().clamp_min(1e-6);
    let uq = xq[0].clone() / zqs.clone() * fx2 + cx2;
    let vq = xq[1].clone() / zqs * fy2 + cy2;
    let lane_ok = z.detach().greater_elem(0.01).float()
        * zq.detach().greater_elem(0.1).float().reshape([m, k])
        * denom.detach().abs().greater_elem(1e-4).float()
        * (uq.clone().detach().greater_equal_elem(0.0).float()
            * uq.clone().detach().lower_elem(w as f32).float()
            * vq.clone().detach().greater_equal_elem(0.0).float()
            * vq.clone().detach().lower_elem(h as f32).float())
        .reshape([m, k]);
    // A patch counts only if every one of its samples is valid.
    let valid = ms * lane_ok.min_dim(1).squeeze_dim::<1>(1);
    let b = bilinear(g2, uq * sc2.0, vq * sc2.1).reshape([m, k]);
    let a = a.clone() - a.mean_dim(1);
    let b = b.clone() - b.mean_dim(1);
    let va = a.clone().powi_scalar(2).sum_dim(1).squeeze_dim::<1>(1);
    let vb = b.clone().powi_scalar(2).sum_dim(1).squeeze_dim::<1>(1);
    let cov = (a * b).sum_dim(1).squeeze_dim::<1>(1);
    // PGSR: squared correlation, score 1 - cc in [0, 2]; textureless or
    // mismatched patches (score >= .9) are dropped.
    let cc = cov.powi_scalar(2) / (va * vb + 1e-8);
    let score = (cc.neg() + 1.0).clamp(0.0, 2.0);
    let keep = valid * score.clone().detach().lower_elem(0.9).float();
    let pw = keep.clone() * ws;
    let photometric = (pw * score).sum() / keep.sum().clamp_min(1.0);
    (geometric, photometric)
}
