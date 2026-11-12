//! Mesh extraction from rendered views of a trained splat.
//!
//! Pipeline: rendered views (RGB, alpha, metric z-depth as `f32`, pinhole intrinsics,
//! world-to-camera pose) → sparse voxel-hash TSDF fusion with colour → marching cubes →
//! drop small floating components → optional quadric decimation → GLB / PLY.
//!
//! Conventions: cameras use OpenCV axes (+x right, +y down, +z forward). Depth is view-space z
//! (distance along the optical axis), not ray length; `0`, negative or non-finite means "no
//! surface". Pixel `(i, j)` covers `[i, i + 1) × [j, j + 1)`, so its centre is `i + 0.5`.
//! The mesh stays in the world frame of the input poses (the splat's training frame).
//!
//! Two entry points:
//! - [`fuse`]: you already have every view and know (or want auto) bounds.
//! - [`mesh_from_renders`]: you have cameras and a render callback (Brush); it first renders a
//!   low-resolution pass to find the scene bounds, then streams full-resolution views.

use anyhow::{Result, bail};
use glam::{Mat3, Mat4, Vec3};

mod clean;
mod decimate;
mod io;
mod load;
mod mc;
pub mod preview;
mod texture;
mod tsdf;
pub mod voxel;

pub use glam;
pub use clean::{remove_small_components, smooth_taubin, vertex_normals};
pub use decimate::decimate;
pub use mc::mesh_occupancy;
pub use io::{glb_bytes, glb_bytes_textured, write_glb, write_glb_textured, write_ply};
pub use texture::{TextureOpts, TexturedMesh, bake_texture};
pub use load::{Loaded, load_mesh};
pub use tsdf::TsdfVolume;

/// Pinhole intrinsics in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl Intrinsics {
    /// The same camera at `1 / factor` of the resolution.
    pub fn downscaled(&self, factor: u32) -> Self {
        let f = factor.max(1) as f32;
        let width = (self.width as f32 / f).round().max(1.0) as u32;
        let height = (self.height as f32 / f).round().max(1.0) as u32;
        let (sx, sy) = (width as f32 / self.width as f32, height as f32 / self.height as f32);
        Self { fx: self.fx * sx, fy: self.fy * sy, cx: self.cx * sx, cy: self.cy * sy, width, height }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub intrinsics: Intrinsics,
    /// World-to-camera transform (OpenCV axes).
    pub world_to_cam: Mat4,
}

impl Camera {
    /// From a camera-to-world rotation and position (what Brush's `RenderCamera` stores).
    pub fn from_c2w(rotation: glam::Quat, position: Vec3, intrinsics: Intrinsics) -> Self {
        Self { intrinsics, world_to_cam: Mat4::from_rotation_translation(rotation, position).inverse() }
    }

    pub fn cam_to_world(&self) -> Mat4 {
        self.world_to_cam.inverse()
    }

    /// Camera-to-world rotation (OpenCV axes), as Brush's `RenderCamera::rotation`.
    pub fn rotation(&self) -> glam::Quat {
        glam::Quat::from_mat4(&self.cam_to_world()).normalize()
    }

    pub fn position(&self) -> Vec3 {
        self.cam_to_world().w_axis.truncate()
    }

    pub fn forward(&self) -> Vec3 {
        self.cam_to_world().z_axis.truncate().normalize()
    }
}

/// One rendered view. Buffers are row-major, `width * height` pixels.
#[derive(Debug, Clone)]
pub struct View {
    pub camera: Camera,
    /// Straight (not premultiplied) RGB8, `3 * width * height` bytes, sRGB as displayed. If your
    /// renderer composited on black, call [`View::unpremultiply`].
    pub rgb: Vec<u8>,
    /// Accumulated opacity in [0, 1]. Empty means fully opaque everywhere.
    pub alpha: Vec<f32>,
    /// View-space z-depth in world units; `<= 0` or non-finite = no surface.
    pub depth: Vec<f32>,
}

impl View {
    pub fn validate(&self) -> Result<()> {
        let n = (self.camera.intrinsics.width * self.camera.intrinsics.height) as usize;
        if self.depth.len() != n {
            bail!("depth has {} pixels, camera has {n}", self.depth.len());
        }
        if !self.rgb.is_empty() && self.rgb.len() != 3 * n {
            bail!("rgb has {} bytes, expected {}", self.rgb.len(), 3 * n);
        }
        if !self.alpha.is_empty() && self.alpha.len() != n {
            bail!("alpha has {} pixels, camera has {n}", self.alpha.len());
        }
        Ok(())
    }

    /// Undo compositing on black: rgb /= alpha where alpha > 0.
    pub fn unpremultiply(&mut self) {
        if self.alpha.is_empty() {
            return;
        }
        for (px, &a) in self.rgb.chunks_exact_mut(3).zip(&self.alpha) {
            if a > 1e-3 && a < 1.0 {
                for c in px {
                    *c = (*c as f32 / a).round().min(255.0) as u8;
                }
            }
        }
    }
}

/// Axis-aligned box in world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    pub fn expanded(&self, by: f32) -> Self {
        Self { min: self.min - Vec3::splat(by), max: self.max + Vec3::splat(by) }
    }
}

#[derive(Debug, Clone)]
pub struct MeshOpts {
    /// Voxel edge length in world units. `None`: longest bounds axis / `resolution`.
    pub voxel_size: Option<f32>,
    /// Voxels along the longest axis of the bounds when `voxel_size` is auto.
    pub resolution: u32,
    /// Region to reconstruct. `None`: estimated from the depth maps and cameras.
    pub bounds: Option<Aabb>,
    /// Scale on the auto bounds' camera sphere (median camera distance to the look target).
    pub bounds_radius_scale: f32,
    /// Limit automatic bounds to the converging camera target (object captures). Disable for
    /// whole-scene reconstruction so room walls are not cropped to an object-sized sphere.
    pub crop_to_target: bool,
    /// Conservative cap on TSDF voxel storage. Auto voxels grow to fit the entire bounds;
    /// surfaces are never discarded to meet the budget. Explicit voxel sizes must fit too.
    pub max_voxel_memory_bytes: Option<usize>,
    /// TSDF truncation distance in voxels.
    pub trunc_voxels: f32,
    /// Pixels below this opacity are ignored.
    pub min_alpha: f32,
    /// Drop depth pixels whose 3×3 neighbourhood spans more than this fraction of their depth
    /// (silhouette pixels where expected depth mixes foreground and background).
    pub depth_edge_tol: f32,
    /// Also integrate free space ahead of the surface into every allocated voxel in the
    /// frustum, which erodes floaters that other views see through.
    pub carve: bool,
    /// Voxels need at least this accumulated weight to be meshed.
    pub min_weight: f32,
    /// Drop connected components with fewer triangles than this fraction of the largest one.
    pub min_component_ratio: f32,
    /// Taubin smoothing passes after cleanup (0 = off); evens out splat depth noise.
    pub smooth_iterations: u32,
    /// Quadric-decimate to at most this many triangles.
    pub max_triangles: Option<usize>,
    /// Downscale factor for the bounds pre-pass in [`mesh_from_renders`].
    pub prepass_downscale: u32,
}

impl Default for MeshOpts {
    fn default() -> Self {
        Self {
            voxel_size: None,
            resolution: 512,
            bounds: None,
            bounds_radius_scale: 1.0,
            crop_to_target: true,
            max_voxel_memory_bytes: None,
            trunc_voxels: 10.0,
            min_alpha: 0.5,
            depth_edge_tol: 0.04,
            carve: true,
            min_weight: 0.5,
            min_component_ratio: 0.02,
            smooth_iterations: 5,
            max_triangles: Some(1_500_000),
            prepass_downscale: 8,
        }
    }
}

/// Indexed triangle mesh with per-vertex sRGB colour.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub colors: Vec<[u8; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn bounds(&self) -> Option<Aabb> {
        let first = *self.positions.first()?;
        let mut b = Aabb { min: first, max: first };
        for &p in &self.positions {
            b.min = b.min.min(p);
            b.max = b.max.max(p);
        }
        Some(b)
    }

    pub fn transform(&mut self, m: Mat4) {
        for p in &mut self.positions {
            *p = m.transform_point3(*p);
        }
        if m.determinant() < 0.0 {
            for t in &mut self.triangles {
                t.swap(1, 2);
            }
        }
    }

    /// Drop vertices no triangle uses (indices are remapped).
    pub fn compact(&mut self) {
        let mut remap = vec![u32::MAX; self.positions.len()];
        let mut next = 0u32;
        for t in &mut self.triangles {
            for i in t.iter_mut() {
                let r = &mut remap[*i as usize];
                if *r == u32::MAX {
                    *r = next;
                    next += 1;
                }
                *i = *r;
            }
        }
        let mut positions = vec![Vec3::ZERO; next as usize];
        let mut colors = vec![[0u8; 3]; next as usize];
        for (old, &new) in remap.iter().enumerate() {
            if new != u32::MAX {
                positions[new as usize] = self.positions[old];
                colors[new as usize] = self.colors[old];
            }
        }
        self.positions = positions;
        self.colors = colors;
    }
}

/// What happened during a fusion, for logs and tool results.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub views: usize,
    pub bounds: Option<Aabb>,
    pub voxel_size: f32,
    pub blocks: usize,
    pub voxel_memory_bytes: usize,
    pub raw_triangles: usize,
    pub components_removed: usize,
    pub triangles: usize,
    pub vertices: usize,
    pub prepass_s: f64,
    pub integrate_s: f64,
    pub extract_s: f64,
    pub cleanup_s: f64,
}

impl Stats {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "views": self.views,
            "bounds_min": self.bounds.map(|b| b.min.to_array()),
            "bounds_max": self.bounds.map(|b| b.max.to_array()),
            "voxel_size": self.voxel_size,
            "blocks": self.blocks,
            "voxel_memory_mb": self.voxel_memory_bytes as f64 / 1e6,
            "raw_triangles": self.raw_triangles,
            "components_removed": self.components_removed,
            "triangles": self.triangles,
            "vertices": self.vertices,
            "prepass_s": self.prepass_s,
            "integrate_s": self.integrate_s,
            "extract_s": self.extract_s,
            "cleanup_s": self.cleanup_s,
        })
    }
}

/// Fuse views into a mesh. With `opts.bounds == None` the bounds come from the views' own
/// depth, which requires holding them all; prefer [`mesh_from_renders`] for large inputs.
pub fn fuse(views: impl IntoIterator<Item = View>, opts: &MeshOpts) -> Result<(Mesh, Stats)> {
    let mut stats = Stats::default();
    let views: Vec<View> = views.into_iter().collect();
    let bounds = match opts.bounds {
        Some(b) => b,
        None => {
            let t = std::time::Instant::now();
            let b = estimate_bounds(views.iter(), opts)?;
            stats.prepass_s = t.elapsed().as_secs_f64();
            b
        }
    };
    fuse_streaming(bounds, views.into_iter().map(Ok), opts, stats)
}

/// The integration call for a renderer: give it every camera (training views plus any extra
/// orbit views) and a callback that renders a camera to a [`View`]. The callback is first
/// called with downscaled copies of every camera (`opts.prepass_downscale`) to estimate the
/// bounds (skipped when `opts.bounds` is set), then once per camera at full resolution; views
/// are integrated as they arrive, so memory does not grow with the number of views.
pub fn mesh_from_renders(cameras: &[Camera], render: impl FnMut(&Camera) -> Result<View>, opts: &MeshOpts) -> Result<(Mesh, Stats)> {
    Ok(meshes_from_renders(cameras, render, std::slice::from_ref(opts))?.remove(0))
}

/// [`mesh_from_renders`] for several regions/settings at once (e.g. a cropped subject and the
/// whole scene): every camera is rendered once and integrated into each volume.
pub fn meshes_from_renders(cameras: &[Camera], mut render: impl FnMut(&Camera) -> Result<View>, opts: &[MeshOpts]) -> Result<Vec<(Mesh, Stats)>> {
    if cameras.is_empty() {
        bail!("no cameras to mesh from");
    }
    let mut prepass: Option<(Vec<View>, f64)> = None;
    let mut volumes = Vec::with_capacity(opts.len());
    for o in opts {
        let mut stats = Stats::default();
        let bounds = match o.bounds {
            Some(b) => b,
            None => {
                if prepass.is_none() {
                    let t = std::time::Instant::now();
                    let small: Vec<View> = cameras
                        .iter()
                        .map(|c| render(&Camera { intrinsics: c.intrinsics.downscaled(o.prepass_downscale), ..*c }))
                        .collect::<Result<_>>()?;
                    prepass = Some((small, t.elapsed().as_secs_f64()));
                }
                let (small, secs) = prepass.as_ref().expect("set above");
                stats.prepass_s = *secs;
                estimate_bounds(small.iter(), o)?
            }
        };
        let voxel = voxel_size(bounds, o)?;
        stats.bounds = Some(bounds);
        stats.voxel_size = voxel;
        volumes.push((TsdfVolume::new(voxel, o.trunc_voxels * voxel, bounds), stats));
    }
    drop(prepass);
    let t = std::time::Instant::now();
    for cam in cameras {
        let view = render(cam)?;
        view.validate()?;
        for ((vol, stats), o) in volumes.iter_mut().zip(opts) {
            vol.integrate(&view, o);
            stats.views += 1;
        }
    }
    let integrate_s = t.elapsed().as_secs_f64();
    volumes
        .into_iter()
        .zip(opts)
        .map(|((vol, mut stats), o)| {
            stats.integrate_s = integrate_s;
            finish(vol, o, stats)
        })
        .collect()
}

fn fuse_streaming(bounds: Aabb, views: impl Iterator<Item = Result<View>>, opts: &MeshOpts, mut stats: Stats) -> Result<(Mesh, Stats)> {
    let voxel = voxel_size(bounds, opts)?;
    stats.bounds = Some(bounds);
    stats.voxel_size = voxel;
    let mut vol = TsdfVolume::new(voxel, opts.trunc_voxels * voxel, bounds);
    let t = std::time::Instant::now();
    for view in views {
        let view = view?;
        view.validate()?;
        vol.integrate(&view, opts);
        stats.views += 1;
    }
    stats.integrate_s = t.elapsed().as_secs_f64();
    finish(vol, opts, stats)
}

fn finish(vol: TsdfVolume, opts: &MeshOpts, mut stats: Stats) -> Result<(Mesh, Stats)> {
    stats.blocks = vol.num_blocks();
    stats.voxel_memory_bytes = vol.memory_bytes();
    let t = std::time::Instant::now();
    let mut mesh = vol.extract_mesh(opts.min_weight);
    stats.extract_s = t.elapsed().as_secs_f64();
    drop(vol);
    stats.raw_triangles = mesh.triangles.len();
    let t = std::time::Instant::now();
    stats.components_removed = remove_small_components(&mut mesh, opts.min_component_ratio);
    smooth_taubin(&mut mesh, opts.smooth_iterations);
    if let Some(max) = opts.max_triangles {
        decimate(&mut mesh, max);
    }
    mesh.compact();
    stats.cleanup_s = t.elapsed().as_secs_f64();
    stats.triangles = mesh.triangles.len();
    stats.vertices = mesh.positions.len();
    Ok((mesh, stats))
}

/// Bound the *entire* sparse grid, including truncation padding and block rounding. This is
/// deliberately conservative: even a fully occupied room fits, without clipping allocations
/// when a later view reveals a new wall. Ordinary object callers can retain uncapped voxels.
fn voxel_size(bounds: Aabb, opts: &MeshOpts) -> Result<f32> {
    let extent = bounds.extent();
    if !extent.is_finite() || extent.min_element() < 0.0 || !(opts.trunc_voxels >= 0.0 && opts.trunc_voxels.is_finite()) {
        bail!("invalid mesh bounds or truncation: {bounds:?}");
    }
    let mut voxel = opts.voxel_size.unwrap_or_else(|| extent.max_element() / opts.resolution.max(8) as f32);
    if !(voxel > 0.0 && voxel.is_finite()) {
        bail!("invalid voxel size {voxel} (bounds {bounds:?})");
    }
    if let Some(budget) = opts.max_voxel_memory_bytes {
        // Two extra blocks cover endpoint alignment of a grid anchored at the world origin.
        let bytes = |v: f32| -> f64 {
            let blocks = extent / (tsdf::B as f32 * v) + Vec3::splat(2.0 * opts.trunc_voxels / tsdf::B as f32);
            blocks.to_array().into_iter().map(|n| (n.ceil() + 2.0) as f64).product::<f64>()
                * std::mem::size_of::<[tsdf::Voxel; tsdf::BV]>() as f64
        };
        for _ in 0..256 {
            if bytes(voxel) <= budget as f64 {
                return Ok(voxel);
            }
            if opts.voxel_size.is_some() {
                bail!("explicit voxel size {voxel} exceeds the {budget}-byte TSDF budget");
            }
            voxel *= 1.05;
        }
        bail!("TSDF budget {budget} is too small for the truncation band");
    }
    Ok(voxel)
}

/// Depth cut for the bounds of captures without a converging target, as a multiple of the
/// median rendered depth.
const NEAR_DEPTH_SCALE: f32 = 3.0;

/// Reconstruction region: the box (1st–99th percentile per axis, padded 5%) of the depth points
/// that fall inside a sphere around the cameras' look target, of radius median camera distance
/// times `opts.bounds_radius_scale`, or (no converging target) of the depth points up to
/// `NEAR_DEPTH_SCALE` x the median depth. Far background (sky, distant buildings) is excluded.
pub fn estimate_bounds<'a>(views: impl Iterator<Item = &'a View>, opts: &MeshOpts) -> Result<Aabb> {
    let views: Vec<&View> = views.collect();
    if views.is_empty() {
        bail!("no views to estimate bounds from");
    }
    let cams: Vec<Camera> = views.iter().map(|v| v.camera).collect();
    // Without a converging target (walks, slides) keep every depth point.
    let sphere = opts.crop_to_target.then(|| converging_target(&cams)).flatten().map(|target| {
        let mut dists: Vec<f32> = cams.iter().map(|c| (c.position() - target).length()).collect();
        dists.sort_by(f32::total_cmp);
        (target, dists[dists.len() / 2].max(1e-6) * opts.bounds_radius_scale)
    });
    // Subsample so the point count stays near ~2M regardless of input resolution.
    let total: usize = views.iter().map(|v| v.depth.len()).sum();
    let stride = (total / 2_000_000).max(1);
    let mut pts = Vec::new();
    let mut all = Vec::new();
    for v in &views {
        let c2w = v.camera.cam_to_world();
        let k = v.camera.intrinsics;
        for (i, &d) in v.depth.iter().enumerate().step_by(stride) {
            if !(d > 0.0 && d.is_finite()) || v.alpha.get(i).is_some_and(|&a| a < opts.min_alpha) {
                continue;
            }
            let (x, y) = ((i as u32 % k.width) as f32 + 0.5, (i as u32 / k.width) as f32 + 0.5);
            let p = c2w.transform_point3(Vec3::new((x - k.cx) / k.fx * d, (y - k.cy) / k.fy * d, d));
            all.push((p, d));
            if sphere.is_some_and(|(target, radius)| (p - target).length() <= radius) {
                pts.push(p);
            }
        }
    }
    if sphere.is_none() && !all.is_empty() {
        // No converging target (walks, slides): keep the near content, depths up to
        // NEAR_DEPTH_SCALE x the median, so sky and far background don't set the box
        // (a driveway walk: ~2300 -> ~130 units, voxel 4.6 -> ~0.27).
        let mut depths: Vec<f32> = all.iter().map(|&(_, d)| d).collect();
        let mid = depths.len() / 2;
        let median = *depths.select_nth_unstable_by(mid, f32::total_cmp).1;
        pts = all.iter().filter(|&&(_, d)| d <= NEAR_DEPTH_SCALE * median).map(|&(p, _)| p).collect();
    }
    let use_pts = if pts.len() >= 100 { pts } else { all.into_iter().map(|(p, _)| p).collect() };
    if use_pts.len() < 8 {
        bail!("views have no usable depth (need alpha >= {} and depth > 0)", opts.min_alpha);
    }
    let mut min = Vec3::ZERO;
    let mut max = Vec3::ZERO;
    for axis in 0..3 {
        let mut vals: Vec<f32> = use_pts.iter().map(|p| p[axis]).collect();
        vals.sort_by(f32::total_cmp);
        let n = vals.len();
        min[axis] = vals[n / 100];
        max[axis] = vals[(n * 99 / 100).min(n - 1)];
    }
    let pad = (max - min).max_element() * 0.05;
    Ok(Aabb { min, max }.expanded(pad))
}

/// Least-squares point closest to all optical axes (where the capture is aimed).
pub fn look_target(cams: &[Camera]) -> Vec3 {
    converging_target(cams).unwrap_or_else(|| {
        let sum: Vec3 = cams.iter().map(|c| c.position() + c.forward()).sum();
        sum / cams.len().max(1) as f32
    })
}

/// The point the optical axes converge on, when they do: the least-squares
/// solution must be well conditioned and in front of at least half the
/// cameras. Near-parallel axes (a walk, a sideways slide) put it far behind
/// the cameras (a driveway walk: 1186 units behind all 150), so they get `None`.
pub fn converging_target(cams: &[Camera]) -> Option<Vec3> {
    let mut a = Mat3::ZERO;
    let mut b = Vec3::ZERO;
    for c in cams {
        let d = c.forward();
        let m = Mat3::IDENTITY - Mat3::from_cols(d * d.x, d * d.y, d * d.z);
        a += m;
        b += m * c.position();
    }
    // det / (trace / 3)^3 is 1 for evenly spread axes and ~0 when one
    // direction is unconstrained (a driveway walk 7e-6, Mip-NeRF 360 garden 0.98).
    let scale = (a.x_axis.x + a.y_axis.y + a.z_axis.z) / 3.0;
    if !(scale > 0.0) || a.determinant() < 1e-3 * scale.powi(3) {
        return None;
    }
    let target = a.inverse() * b;
    let ahead = cams.iter().filter(|c| (target - c.position()).dot(c.forward()) > 0.0).count();
    (2 * ahead >= cams.len()).then_some(target)
}
