//! Sparse voxel-hash TSDF: 8³ voxel blocks allocated on demand inside the truncation band
//! around observed surfaces, integrated in parallel per view.

use glam::{IVec3, Vec3};
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::{Aabb, MeshOpts, View};

pub(crate) const B: i32 = 8;
pub(crate) const BV: usize = (B * B * B) as usize;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Voxel {
    /// Truncated signed distance in units of the truncation distance, in [-1, 1]; positive in
    /// front of the surface.
    pub tsdf: f32,
    pub weight: f32,
    pub color_weight: f32,
    /// Running mean of sRGB in [0, 255].
    pub rgb: [f32; 3],
}

pub(crate) type Block = Box<[Voxel; BV]>;

#[inline]
pub(crate) fn local_index(l: IVec3) -> usize {
    (l.x + B * (l.y + B * l.z)) as usize
}

/// Depth-map pixel classes after filtering.
const UNKNOWN: f32 = 0.0;
const FREE: f32 = f32::INFINITY;

pub struct TsdfVolume {
    pub(crate) voxel: f32,
    pub(crate) trunc: f32,
    bounds: Aabb,
    pub(crate) index: HashMap<IVec3, u32>,
    pub(crate) coords: Vec<IVec3>,
    pub(crate) blocks: Vec<Block>,
}

impl TsdfVolume {
    /// `voxel` and `trunc` in world units; only surfaces inside `bounds` are integrated.
    pub fn new(voxel: f32, trunc: f32, bounds: Aabb) -> Self {
        Self { voxel, trunc, bounds: bounds.expanded(trunc), index: HashMap::new(), coords: vec![], blocks: vec![] }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<[Voxel; BV]>()
    }

    pub fn integrate(&mut self, view: &View, opts: &MeshOpts) {
        let (depth, conf) = filter_depth(view, opts);
        let k = view.camera.intrinsics;
        let (w, h) = (k.width as usize, k.height as usize);
        let c2w = view.camera.cam_to_world();
        let rot = glam::Mat3::from_mat4(c2w);
        let origin = c2w.w_axis.truncate();
        let (voxel, trunc, bounds) = (self.voxel, self.trunc, self.bounds);

        // 1. Blocks touched by the truncation band of every surface pixel.
        let touched: HashSet<IVec3> = (0..h)
            .into_par_iter()
            .fold(HashSet::new, |mut set, y| {
                for x in 0..w {
                    let d = depth[y * w + x];
                    if !(d > 0.0 && d.is_finite()) {
                        continue;
                    }
                    let dir = rot * Vec3::new((x as f32 + 0.5 - k.cx) / k.fx, (y as f32 + 0.5 - k.cy) / k.fy, 1.0);
                    let step = 0.5 * B as f32 * voxel / dir.length();
                    let n = ((2.0 * trunc / step).ceil() as i32).max(1);
                    for s in 0..=n {
                        let z = d - trunc + 2.0 * trunc * s as f32 / n as f32;
                        if z <= 0.0 {
                            continue;
                        }
                        let p = origin + dir * z;
                        if bounds.contains(p) {
                            set.insert(block_of(p, voxel));
                        }
                    }
                }
                set
            })
            .reduce(HashSet::new, |mut a, b| {
                if a.len() < b.len() {
                    return b.into_iter().fold(a, |mut acc, c| {
                        acc.insert(c);
                        acc
                    });
                }
                a.extend(b);
                a
            });
        let mut sorted: Vec<IVec3> = touched.iter().copied().collect();
        sorted.sort_by_key(|c| (c.z, c.y, c.x));
        for c in sorted {
            if !self.index.contains_key(&c) {
                self.index.insert(c, self.blocks.len() as u32);
                self.coords.push(c);
                self.blocks.push(Box::new([Voxel::default(); BV]));
            }
        }

        // 2. Update voxels: band blocks always; with carving, every allocated block in the frustum.
        let w2c = view.camera.world_to_cam;
        let carve = opts.carve;
        let block_radius = 0.5 * 3f32.sqrt() * B as f32 * voxel;
        let rgb = &view.rgb;
        self.blocks.par_iter_mut().zip(self.coords.par_iter()).for_each(|(block, &coord)| {
            if carve {
                let center = (coord.as_vec3() * B as f32 + 0.5 * (B - 1) as f32) * voxel;
                let pc = w2c.transform_point3(center);
                if pc.z < -block_radius {
                    return;
                }
                if pc.z > block_radius {
                    let r = block_radius / (pc.z - block_radius);
                    let (u, v) = (k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy);
                    if u < -r * k.fx || v < -r * k.fy || u > w as f32 + r * k.fx || v > h as f32 + r * k.fy {
                        return;
                    }
                }
            } else if !touched.contains(&coord) {
                return;
            }
            let base = coord * B;
            for lz in 0..B {
                for ly in 0..B {
                    for lx in 0..B {
                        let g = base + IVec3::new(lx, ly, lz);
                        let pc = w2c.transform_point3(g.as_vec3() * voxel);
                        if pc.z <= 1e-6 {
                            continue;
                        }
                        let u = k.fx * pc.x / pc.z + k.cx;
                        let v = k.fy * pc.y / pc.z + k.cy;
                        if !(u >= 0.0 && v >= 0.0 && u < w as f32 && v < h as f32) {
                            continue;
                        }
                        let i = v as usize * w + u as usize;
                        let d = depth[i];
                        if d == UNKNOWN {
                            continue;
                        }
                        let sdf = d - pc.z;
                        if sdf < -trunc {
                            continue;
                        }
                        let vx = &mut block[local_index(IVec3::new(lx, ly, lz))];
                        let t = (sdf / trunc).min(1.0);
                        let ow = conf[i];
                        let nw = vx.weight + ow;
                        vx.tsdf = (vx.tsdf * vx.weight + t * ow) / nw;
                        vx.weight = nw;
                        if sdf < trunc && !rgb.is_empty() {
                            let c = &rgb[3 * i..3 * i + 3];
                            let cw = vx.color_weight + ow;
                            for ch in 0..3 {
                                vx.rgb[ch] = (vx.rgb[ch] * vx.color_weight + c[ch] as f32 * ow) / cw;
                            }
                            vx.color_weight = cw;
                        }
                    }
                }
            }
        });
    }
}

#[inline]
pub(crate) fn block_of(p: Vec3, voxel: f32) -> IVec3 {
    let g = (p / voxel).round().as_ivec3();
    IVec3::new(g.x.div_euclid(B), g.y.div_euclid(B), g.z.div_euclid(B))
}

/// Per pixel: a depth to integrate, `UNKNOWN` (skip), or `FREE` (nothing rendered there, so
/// everything along the ray is empty; only when alpha is given); and the observation weight:
/// alpha times the cosine between the depth map's normal and the viewing ray, so grazing
/// (noisy) depth counts less. Free-space pixels get weight 1.
fn filter_depth(view: &View, opts: &MeshOpts) -> (Vec<f32>, Vec<f32>) {
    let k = view.camera.intrinsics;
    let (w, h) = (k.width as usize, k.height as usize);
    let raw = &view.depth;
    let alpha = &view.alpha;
    let ok = |i: usize| raw[i] > 0.0 && raw[i].is_finite() && alpha.get(i).is_none_or(|&a| a >= opts.min_alpha);
    let point = |x: usize, y: usize| {
        let d = raw[y * w + x];
        Vec3::new((x as f32 + 0.5 - k.cx) / k.fx * d, (y as f32 + 0.5 - k.cy) / k.fy * d, d)
    };
    let mut out = vec![UNKNOWN; w * h];
    let mut conf = vec![0f32; w * h];
    out.par_chunks_mut(w).zip(conf.par_chunks_mut(w)).enumerate().for_each(|(y, (row, crow))| {
        for x in 0..w {
            let i = y * w + x;
            if !ok(i) {
                if alpha.get(i).is_some_and(|&a| a < 0.02) {
                    row[x] = FREE;
                    crow[x] = 1.0;
                }
                continue;
            }
            let d = raw[i];
            let (mut lo, mut hi) = (d, d);
            for ny in y.saturating_sub(1)..(y + 2).min(h) {
                for nx in x.saturating_sub(1)..(x + 2).min(w) {
                    let j = ny * w + nx;
                    if ok(j) {
                        lo = lo.min(raw[j]);
                        hi = hi.max(raw[j]);
                    }
                }
            }
            if hi - lo > opts.depth_edge_tol * d {
                continue;
            }
            // Central differences where both neighbours are valid, else one-sided.
            let diff = |a: Option<(usize, usize)>, b: Option<(usize, usize)>| -> Option<Vec3> {
                let va = a.filter(|&(ax, ay)| ok(ay * w + ax));
                let vb = b.filter(|&(bx, by)| ok(by * w + bx));
                match (va, vb) {
                    (Some(p), Some(q)) => Some(point(q.0, q.1) - point(p.0, p.1)),
                    (Some(p), None) => Some(point(x, y) - point(p.0, p.1)),
                    (None, Some(q)) => Some(point(q.0, q.1) - point(x, y)),
                    (None, None) => None,
                }
            };
            let dx = diff(x.checked_sub(1).map(|a| (a, y)), (x + 1 < w).then_some((x + 1, y)));
            let dy = diff(y.checked_sub(1).map(|b| (x, b)), (y + 1 < h).then_some((x, y + 1)));
            let cos = match (dx, dy) {
                (Some(a), Some(b)) => a.cross(b).normalize_or_zero().dot(point(x, y).normalize()).abs(),
                _ => 0.5,
            };
            let a = alpha.get(i).copied().unwrap_or(1.0).min(1.0);
            row[x] = d;
            crow[x] = (a * cos).max(0.05);
        }
    });
    (out, conf)
}
