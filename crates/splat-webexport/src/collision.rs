//! CPU collision packages in the exact exported splat frame, built after the package is published.
//!
//! Depth fusion: the depth where the splats' front-to-back opacity reaches 70% is rasterised (CPU) at
//! every capture camera. Surface hits and seen-free space vote per view on a dense voxel grid: a voxel
//! is solid only when at least twice as many views put a surface in it as see through it, so faint
//! haze that stops some views' depth (invisible in colour) is carved away rather than kept as a shell.
//! A morphological closing bridges small gaps, components under 0.5% of the volume are dropped, and
//! a solid floor slab (export floor, else the dominant upward surface below the cameras) is always
//! added under the occupied footprint. Marching cubes closes the surface; meshoptimizer reduces it;
//! Parry's pinned Apache-2.0 V-HACD supplies approximate convex pieces for compound dynamic bodies.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat3, DMat4, IVec3, Mat3, Quat, Vec3};
use hashbrown::{HashMap, HashSet};
use parry3d::na::Point3;
use parry3d::transformation::vhacd::{VHACD, VHACDParameters};
use parry3d::transformation::voxelization::FillMode;
use rayon::prelude::*;
use serde_json::{Value, json};
use splat_mesh::Mesh;

use crate::ply::{self, Gaussians};

/// Fusion voxel (scene units); coarsened when the grid would exceed [`MAX_CELLS`]. Marching cubes
/// puts the surface half a voxel outside the solid samples, so this bounds the collision's bias.
const VOXEL: f32 = 0.025;
const MAX_CELLS: usize = 120_000_000;
/// Depth maps are rendered this many pixels tall (width from the camera's aspect). Coarser maps widen
/// every silhouette (an edge pixel takes the nearer surface's depth): 180 px put table edges ~10 cm out.
const DEPTH_HEIGHT: usize = 360;
/// Collision depth is taken where transmittance falls to this (70% opacity), past faint haze.
const DEPTH_TRANSMITTANCE: f32 = 0.3;
/// Splats below this opacity do not contribute to depth.
const MIN_ALPHA: f32 = 0.05;
/// Morphological closing radius (voxels, cube element). Larger closings fuse tables into chairs.
const CLOSE: usize = 1;
/// Components smaller than this fraction of the occupied volume are dropped.
const MIN_COMPONENT: f64 = 0.005;
/// Floor slab thickness (scene units).
const SLAB: f32 = 0.10;
const TARGET_TRIANGLES: usize = 80_000;

#[derive(clap::Args, Debug, Clone)]
pub struct ExportCollisionArgs {
    /// Scene package (adds collision/ to it), trained run, or Gaussian PLY with --data.
    pub input: PathBuf,
    /// Package directory for a run/PLY (scene.spz, scene.json, collision/). Default: <input>/collision-export.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// COLMAP capture: for a bare PLY, or replacing a package's source.data (its capture cameras).
    #[arg(long)]
    pub data: Option<PathBuf>,
    /// Splat PLY replacing a package's source.ply (e.g. a cloud package's local copy).
    #[arg(long)]
    pub ply: Option<PathBuf>,
    /// Also write a Unity project folder (README, .meta, editor setup script, aligned splat PLY copy).
    #[arg(long)]
    pub unity: Option<PathBuf>,
}

/// The package `input` names: itself when it holds scene.json, else where [`run`] exports it.
pub fn package_dir(a: &ExportCollisionArgs) -> PathBuf {
    if a.input.join("scene.json").is_file() {
        return a.input.clone();
    }
    let base = if a.input.is_dir() { a.input.as_path() } else { a.input.parent().unwrap_or(Path::new(".")) };
    a.out.clone().unwrap_or_else(|| base.join("collision-export"))
}

pub fn run(a: &ExportCollisionArgs) -> Result<Value> {
    let out = package_dir(a);
    if out != a.input {
        // export-web itself, so collision and splat can never acquire different frames.
        crate::export_web_with_options(&a.input, a.data.as_deref(), &out,
            &crate::SceneOptions { lod: crate::LodMode::Skip, ..Default::default() })?;
    }
    let sources = if out == a.input { (a.ply.as_deref(), a.data.as_deref()) } else { (None, None) };
    let manifest = add_to_package_from(&out, sources)?;
    if let Some(dir) = &a.unity {
        let scene = read_scene(&out)?;
        let (g, _) = scene_gaussians(&scene, sources)?;
        write_unity(&g, &out.join("collision"), dir)?;
        eprintln!("Unity folder: {}", dir.display());
    }
    Ok(manifest)
}

fn read_scene(pkg: &Path) -> Result<Value> {
    let path = pkg.join("scene.json");
    serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("{}", path.display()))?)
        .with_context(|| format!("{}", path.display()))
}

/// The package's splats in its scene frame (from the source PLY) and its capture cameras;
/// `sources` replaces scene.json's source PLY / capture paths.
fn scene_gaussians(scene: &Value, sources: (Option<&Path>, Option<&Path>)) -> Result<(Gaussians, Vec<View>)> {
    let ply_path = match sources.0 { Some(p) => p.to_path_buf(),
        None => PathBuf::from(scene["source"]["ply"].as_str().context("scene.json: no source.ply")?) };
    let m = scene_transform(scene)?;
    let mut g = ply::read_ply(&ply_path)?;
    ensure!(!g.is_empty(), "{}: no splats", ply_path.display());
    crate::to_scene_frame(&mut g, &m);
    let views = scene_views(scene, sources.1, Some(&ply_path), DEPTH_HEIGHT)?;
    Ok((g, views))
}

fn scene_transform(scene: &Value) -> Result<DMat4> {
    let m: Vec<f64> = scene["transform_from_source"].as_array().context("scene.json: no transform_from_source")?
        .iter().filter_map(Value::as_f64).collect();
    ensure!(m.len() == 16, "scene.json: transform_from_source is not a 4x4");
    Ok(DMat4::from_cols_slice(&m))
}

/// The package's capture cameras (scene.json's source.data unless `data` replaces it) in its scene frame,
/// at depth maps `depth_height` pixels tall; `ply` locates the source's scene_transform.json.
pub(crate) fn scene_views(scene: &Value, data: Option<&Path>, ply: Option<&Path>, depth_height: usize) -> Result<Vec<View>> {
    let data = match data { Some(p) => p.to_path_buf(),
        None => PathBuf::from(scene["source"]["data"].as_str().context("scene.json: no source.data (capture cameras)")?) };
    let m = scene_transform(scene)?;
    let frame = splat_shots::scene::load_scene(&data, ply.and_then(splat_shots::scene_transform_near).as_deref())?;
    let scale = m.x_axis.truncate().length();
    let r = DMat3::from_mat4(m) * (1.0 / scale);
    Ok(frame.cameras.iter().map(|c| {
        let h = depth_height;
        let w = ((h as f64 * c.width as f64 / c.height as f64).round() as usize).clamp(16, 4 * h);
        View {
            name: c.name.clone(),
            c: m.transform_point3(c.center()).as_vec3(),
            r: (r * DMat3::from_mat4(c.c2w)).as_mat3(),
            f: (c.fy * h as f64 / c.height as f64) as f32,
            w, h,
        }
    }).collect())
}

/// The package's splats in its scene frame for the game export: scene.json's source PLY when it is on
/// disk and readable, else the package's own splats (already in the scene frame; a cloud run's PLY is
/// often a fal stub). Returns the splats and where they came from.
pub(crate) fn package_splats(pkg: &Path, scene: &Value) -> Result<(Gaussians, String)> {
    if let Some(p) = scene["source"]["ply"].as_str().map(PathBuf::from).filter(|p| p.is_file()) {
        match ply::read_ply(&p) {
            Ok(mut g) if !g.is_empty() => {
                crate::to_scene_frame(&mut g, &scene_transform(scene)?);
                return Ok((g, p.display().to_string()));
            }
            Ok(_) => eprintln!("{}: no splats; using the package's splats", p.display()),
            Err(e) => eprintln!("{}: {e:#}; using the package's splats", p.display()),
        }
    }
    let g = crate::spz::read_package_splats(pkg)?;
    ensure!(!g.is_empty(), "{}: no splats", pkg.display());
    Ok((g, pkg.join(scene["splat"].as_str().unwrap_or("scene.spz")).display().to_string()))
}

/// Build collision/ for a published package and add it to scene.json (atomically; the viewer
/// hot-reloads scene.json). A package re-exported meanwhile is left alone.
pub fn add_to_package(pkg: &Path) -> Result<Value> {
    add_to_package_from(pkg, (None, None))
}

/// [`add_to_package`] reading the splats / capture from `sources` instead of scene.json's paths.
pub fn add_to_package_from(pkg: &Path, sources: (Option<&Path>, Option<&Path>)) -> Result<Value> {
    let t0 = Instant::now();
    let scene = read_scene(pkg)?;
    let (g, views) = scene_gaussians(&scene, sources)?;
    ensure!(!views.is_empty(), "no capture cameras");
    let t_read = t0.elapsed().as_secs_f64();
    let temp = tempfile::Builder::new().prefix(".collision-").tempdir_in(pkg)?;
    let dir = temp.path().join("collision");
    let mut manifest = build(&g, &views, &scene, &dir)?;
    manifest["stats"]["read_s"] = json!(t_read);
    manifest["stats"]["elapsed_s"] = json!(t0.elapsed().as_secs_f64());
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    let mut current = read_scene(pkg)?;
    let same = |k: &str| current["source"] == scene["source"] && current[k] == scene[k];
    if !(same("transform_from_source") && current["stats"]["splat_bytes"] == scene["stats"]["splat_bytes"]) {
        bail!("{} was re-exported while collision was built; discarded", pkg.display());
    }
    // Replace this generated subtree as a unit, so old convex pieces cannot survive a re-export.
    let dest = pkg.join("collision");
    let old = temp.path().join("previous");
    if dest.exists() { std::fs::rename(&dest, &old)?; }
    if let Err(e) = std::fs::rename(&dir, &dest) {
        if old.exists() { let _ = std::fs::rename(&old, &dest); }
        return Err(e.into());
    }
    current["collision"] = manifest.clone();
    if manifest["status"] == "ready" && current["collider"].is_null() {
        current["collider"] = json!("collision/collision.glb");
    }
    crate::write_atomic(&pkg.join("scene.json"), &serde_json::to_vec_pretty(&current)?)?;
    eprintln!("collision: {} triangles, {} convex pieces, {:.1}s; added to {}",
        manifest["stats"]["triangles"], manifest["stats"]["convex_parts"], t0.elapsed().as_secs_f64(), pkg.join("scene.json").display());
    Ok(manifest)
}

/// Capture camera at depth-map resolution: OpenCV axes (x right, y down, z forward) in the scene frame.
pub struct View {
    /// Image name (COLMAP).
    pub(crate) name: String,
    pub(crate) c: Vec3,
    /// Camera-to-world rotation.
    pub(crate) r: Mat3,
    pub(crate) f: f32,
    pub(crate) w: usize,
    pub(crate) h: usize,
}

impl View {
    /// Move the camera by the rigid transform `m`.
    pub(crate) fn transform(&mut self, m: &DMat4) {
        self.c = m.transform_point3(self.c.as_dvec3()).as_vec3();
        self.r = (DMat3::from_mat4(*m) * self.r.as_dmat3()).as_mat3();
    }
    /// The camera's right axis (world).
    pub(crate) fn right(&self) -> Vec3 { self.r.x_axis }
    /// Unnormalised world ray of pixel centre (x, y); a point at camera depth z is `c + ray * z`.
    fn ray(&self, x: usize, y: usize) -> Vec3 {
        self.r * Vec3::new((x as f32 + 0.5 - self.w as f32 * 0.5) / self.f, (y as f32 + 0.5 - self.h as f32 * 0.5) / self.f, 1.0)
    }
}

/// Splat geometry for depth rendering: centre, 3D covariance (xx xy xz yy yz zz), opacity.
pub(crate) struct Prim {
    p: Vec3,
    cov: [f32; 6],
    alpha: f32,
}

pub(crate) fn prims(g: &Gaussians) -> Vec<Prim> {
    (0..g.len()).into_par_iter().filter_map(|i| {
        let alpha = g.alpha(i);
        let p = Vec3::from(g.pos[i]);
        let s = Vec3::from(g.scale[i]).exp();
        let r = g.rot[i];
        let q = Quat::from_xyzw(r[1], r[2], r[3], r[0]);
        if !(alpha >= MIN_ALPHA) || !p.is_finite() || !s.is_finite() || !q.is_finite() || q.length_squared() < 1e-12 {
            return None;
        }
        let m = Mat3::from_quat(q.normalize()) * Mat3::from_diagonal(s);
        let c = m * m.transpose();
        Some(Prim { p, alpha, cov: [c.x_axis.x, c.y_axis.x, c.z_axis.x, c.y_axis.y, c.z_axis.y, c.z_axis.z] })
    }).collect()
}

/// Median depth (camera z where transmittance first falls to 1/2) per pixel; NaN where it never does.
pub(crate) fn render_depth(prims: &[Prim], v: &View) -> Vec<f32> { render_depth_at(prims, v, 0.5) }

/// Camera z where transmittance first falls to `t_stop` per pixel; NaN where it never does.
pub(crate) fn render_depth_at(prims: &[Prim], v: &View, t_stop: f32) -> Vec<f32> {
    let (w, h) = (v.w as f32, v.h as f32);
    let rt = v.r.transpose();
    // (z, conic a b c, centre, bbox, alpha)
    let mut items: Vec<(f32, [f32; 3], [f32; 2], [i32; 4], f32)> = Vec::with_capacity(prims.len() / 2);
    for s in prims {
        let q = rt * (s.p - v.c);
        if q.z < 0.05 { continue; }
        let u = v.f * q.x / q.z + w * 0.5;
        let y = v.f * q.y / q.z + h * 0.5;
        let c = s.cov;
        let sigma = Mat3::from_cols(Vec3::new(c[0], c[1], c[2]), Vec3::new(c[1], c[3], c[4]), Vec3::new(c[2], c[4], c[5]));
        let cam = rt * sigma * v.r;
        let (fz, fz2) = (v.f / q.z, v.f / (q.z * q.z));
        let j0 = Vec3::new(fz, 0.0, -q.x * fz2);
        let j1 = Vec3::new(0.0, fz, -q.y * fz2);
        let a = j0.dot(cam * j0) + 0.3;
        let b = j0.dot(cam * j1);
        let d = j1.dot(cam * j1) + 0.3;
        let det = a * d - b * b;
        if !(det > 0.0) { continue; }
        let mid = 0.5 * (a + d);
        let radius = 3.0 * (mid + (mid * mid - det).max(0.0).sqrt()).sqrt();
        // Screen-filling splats right in front of a camera are floaters, not surfaces.
        if !(radius < 0.5 * w.max(h)) { continue; }
        let bb = [((u - radius).floor() as i32).max(0), ((y - radius).floor() as i32).max(0),
            ((u + radius).ceil() as i32).min(v.w as i32 - 1), ((y + radius).ceil() as i32).min(v.h as i32 - 1)];
        if bb[0] > bb[2] || bb[1] > bb[3] { continue; }
        items.push((q.z, [d / det, -b / det, a / det], [u, y], bb, s.alpha));
    }
    items.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut t = vec![1.0f32; v.w * v.h];
    let mut depth = vec![f32::NAN; v.w * v.h];
    let mut open = v.w * v.h;
    for (z, con, [u, y], bb, alpha) in items {
        for py in bb[1]..=bb[3] {
            let dy = py as f32 + 0.5 - y;
            let row = py as usize * v.w;
            for px in bb[0]..=bb[2] {
                let k = row + px as usize;
                if t[k] <= t_stop { continue; }
                let dx = px as f32 + 0.5 - u;
                let power = -0.5 * (con[0] * dx * dx + con[2] * dy * dy) - con[1] * dx * dy;
                if power > 0.0 { continue; }
                let a = (alpha * power.exp()).min(0.99);
                if a < 1.0 / 255.0 { continue; }
                t[k] *= 1.0 - a;
                if t[k] <= t_stop {
                    depth[k] = z;
                    open -= 1;
                }
            }
        }
        if open == 0 { break; }
    }
    depth
}

/// Dense voxel grid over the fused region; cell (x, y, z) has its sample at `lo + cell * voxel`.
pub(crate) struct Grid {
    pub(crate) lo: Vec3,
    pub(crate) voxel: f32,
    pub(crate) n: [usize; 3],
}

impl Grid {
    pub(crate) fn index(&self, p: Vec3) -> Option<u32> {
        let c = ((p - self.lo) / self.voxel).round();
        if c.x < 0.0 || c.y < 0.0 || c.z < 0.0 { return None; }
        let (x, y, z) = (c.x as usize, c.y as usize, c.z as usize);
        (x < self.n[0] && y < self.n[1] && z < self.n[2]).then(|| ((z * self.n[1] + y) * self.n[0] + x) as u32)
    }
    pub(crate) fn len(&self) -> usize { self.n[0] * self.n[1] * self.n[2] }
    pub(crate) fn at(&self, x: usize, y: usize, z: usize) -> usize { (z * self.n[1] + y) * self.n[0] + x }
}

fn quantile(v: &mut [f32], q: f64) -> f32 {
    if v.is_empty() { return 0.0; }
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    *v.select_nth_unstable_by(k, f32::total_cmp).1
}

/// Depth-fusion occupancy: the solid cells, how they were made, and the floor slab.
pub(crate) struct Fused {
    pub(crate) grid: Grid,
    pub(crate) solid: Vec<bool>,
    pub(crate) floor: Value,
    pub(crate) max_depth: f32,
    fused: usize,
    closed: usize,
    kept: usize,
    components: usize,
    cavities: usize,
    pub(crate) t_depth: f64,
    pub(crate) t_fuse: f64,
    pub(crate) t_vox: f64,
    /// Mean surface hit (sum, count) per fused-solid voxel: where the views actually saw the surface.
    pub(crate) hit_mean: HashMap<u32, HitStat>,
}

/// Surface hits in one voxel: sum of hit points, sum of unit directions from the hit back to its camera, count.
pub(crate) type HitStat = (Vec3, Vec3, u32);

/// How [`fuse`] samples the scene.
pub(crate) struct FuseOpts {
    /// Voxel edge (units; coarsened when the grid would exceed [`MAX_CELLS`]).
    pub(crate) voxel: f32,
    /// Closing radius (voxels).
    pub(crate) close: usize,
    /// Depths beyond `max(depth_scale * median depth, min_depth)` are background.
    pub(crate) depth_scale: f32,
    pub(crate) min_depth: f32,
    /// Grid bounds: the `bound_q` .. 1 - `bound_q` quantiles of the surface points (and the cameras).
    pub(crate) bound_q: f64,
    /// Grid bounds also cover this box, then are clipped to `clip`.
    pub(crate) include: Option<(Vec3, Vec3)>,
    pub(crate) clip: Option<(Vec3, Vec3)>,
    /// Depth is taken where transmittance falls to this.
    pub(crate) transmittance: f32,
    /// A voxel is solid when `hits >= 2 && hits * solid_ratio >= free` (votes are per view).
    pub(crate) solid_ratio: f64,
    /// Free-space carving stops this many voxels short of each surface hit.
    pub(crate) carve_gap: f32,
    /// Keep the mean surface hit per solid voxel ([`Fused::hit_mean`]) for sub-voxel meshing.
    pub(crate) surface_hits: bool,
}

impl FuseOpts {
    /// The collision export's sampling.
    fn collision() -> Self {
        Self { voxel: VOXEL, close: CLOSE, depth_scale: 3.0, min_depth: 0.0, bound_q: 0.01, include: None, clip: None,
            transmittance: DEPTH_TRANSMITTANCE, solid_ratio: 0.5, carve_gap: 0.6, surface_hits: true }
    }
}

/// Fuse `views`' depth maps of `g` into a solid grid of (at least) `o.voxel` units, closed with a
/// `o.close`-voxel cube, small components dropped, enclosed pockets filled, floor slab added.
pub(crate) fn fuse(g: &Gaussians, views: &[View], scene: &Value, o: &FuseOpts) -> Result<Fused> {
    let close_r = o.close;
    let t0 = Instant::now();
    let prims = prims(g);
    let depths: Vec<Vec<f32>> = views.par_iter().map(|v| render_depth_at(&prims, v, o.transmittance)).collect();
    drop(prims);
    let t_depth = t0.elapsed().as_secs_f64();

    // Depths beyond 3x the median are background (sky, distant trees): no physics there.
    let mut all: Vec<f32> = depths.iter().flat_map(|d| d.iter().step_by(7).copied().filter(|z| z.is_finite())).collect();
    ensure!(!all.is_empty(), "no opaque surface in any capture view");
    let max_depth = (o.depth_scale * quantile(&mut all, 0.5)).max(o.min_depth);
    drop(all);
    // Grid bounds: robust (1-99%) extent of the surface points plus the cameras.
    let mut pts: [Vec<f32>; 3] = Default::default();
    for (v, d) in views.iter().zip(&depths) {
        for y in (0..v.h).step_by(3) { for x in (0..v.w).step_by(3) {
            let z = d[y * v.w + x];
            if z.is_finite() && z < max_depth {
                let p = v.c + v.ray(x, y) * z;
                for a in 0..3 { pts[a].push(p[a]); }
            }
        }}
    }
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for a in 0..3 {
        lo[a] = quantile(&mut pts[a], o.bound_q);
        hi[a] = quantile(&mut pts[a], 1.0 - o.bound_q);
    }
    drop(pts);
    for v in views { lo = lo.min(v.c); hi = hi.max(v.c); }
    if let Some((a, b)) = o.include { lo = lo.min(a); hi = hi.max(b); }
    if let Some((a, b)) = o.clip { lo = lo.max(a); hi = hi.min(b).max(lo); }
    // The finest voxel within the cell budget (padding included), never below VOXEL.
    let pad = (close_r + 2) as f32;
    let e = hi - lo + Vec3::new(1.2, 0.0, 1.2);
    let mut voxel = o.voxel.max(1.01 * (e.x * e.y * e.z / MAX_CELLS as f32).cbrt());
    let grid = loop {
        // XZ room for the floor slab's rim (see floor_slab).
        let rim = Vec3::new(0.6, 0.0, 0.6);
        let lo = lo - pad * voxel - rim;
        let n = ((hi + rim - lo) / voxel).ceil().as_uvec3() + glam::UVec3::splat(2 * close_r as u32 + 4);
        let n = [n.x as usize, n.y as usize, n.z as usize];
        if n[0] * n[1] * n[2] <= MAX_CELLS { break Grid { lo, voxel, n }; }
        voxel *= 1.05;
    };
    eprintln!("collision: {} views, max depth {max_depth:.2}, {}x{}x{} grid at {:.3} units from {lo} to {hi}",
        views.len(), grid.n[0], grid.n[1], grid.n[2], grid.voxel);

    // Per-view votes: a voxel holding a surface point, or crossed by a ray in front of its surface.
    let hits: Vec<AtomicU16> = (0..grid.len()).into_par_iter().map(|_| AtomicU16::new(0)).collect();
    let free: Vec<AtomicU16> = (0..grid.len()).into_par_iter().map(|_| AtomicU16::new(0)).collect();
    views.par_iter().zip(&depths).for_each(|(v, d)| {
        let mut hit = Vec::new();
        let mut empty = Vec::new();
        for y in 0..v.h { for x in 0..v.w {
            let z = d[y * v.w + x];
            if !(z.is_finite() && z < max_depth) { continue; }
            let ray = v.ray(x, y);
            if let Some(i) = grid.index(v.c + ray * z) { hit.push(i); }
            if x % 2 != 0 || y % 2 != 0 { continue; }
            let len = ray.length();
            let end = (z - o.carve_gap * grid.voxel / len).min(0.97 * z);
            let step = 0.7 * grid.voxel / len;
            let mut t = 0.1;
            while t < end {
                if let Some(i) = grid.index(v.c + ray * t) { empty.push(i); }
                t += step;
            }
            // bound scratch memory at fine voxels / large depth maps: dedupe as the list grows
            if empty.len() > (1 << 22) { empty.sort_unstable(); empty.dedup(); }
        }}
        for (list, votes) in [(&mut hit, &hits), (&mut empty, &free)] {
            list.sort_unstable();
            list.dedup();
            for &i in list.iter() { let _ = votes[i as usize].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| c.checked_add(1)); }
        }
    });
    let mut solid: Vec<bool> = hits.par_iter().zip(&free).map(|(h, f)| {
        let (h, f) = (h.load(Ordering::Relaxed) as u32, f.load(Ordering::Relaxed) as u32);
        h >= 2 && h as f64 * o.solid_ratio >= f as f64
    }).collect();
    drop((hits, free));
    // Where in each solid voxel the surface was seen (sub-voxel), for snapping the mesh onto it.
    let hit_mean = if !o.surface_hits { HashMap::new() } else { views.par_iter().zip(&depths).map(|(v, d)| {
        let mut m: HashMap<u32, HitStat> = HashMap::new();
        for y in 0..v.h { for x in 0..v.w {
            let z = d[y * v.w + x];
            if !(z.is_finite() && z < max_depth) { continue; }
            let p = v.c + v.ray(x, y) * z;
            if let Some(i) = grid.index(p) {
                if solid[i as usize] {
                    let e = m.entry(i).or_insert((Vec3::ZERO, Vec3::ZERO, 0));
                    e.0 += p; e.1 += (v.c - p).normalize_or_zero(); e.2 += 1;
                }
            }
        }}
        m
    }).reduce(HashMap::new, |mut a, b| {
        for (k, (p, d, n)) in b { let e = a.entry(k).or_insert((Vec3::ZERO, Vec3::ZERO, 0)); e.0 += p; e.1 += d; e.2 += n; }
        a
    }) };
    drop(depths);
    let fused = solid.iter().filter(|&&s| s).count();
    let t_fuse = t0.elapsed().as_secs_f64() - t_depth;

    close(&mut solid, &grid, close_r);
    let closed = solid.iter().filter(|&&s| s).count();
    let (kept, components) = drop_small_components(&mut solid, &grid);
    let cavities = fill_cavities(&mut solid, &grid, (1.0 / grid.voxel).powi(3) as usize);
    let floor = floor_slab(&mut solid, &grid, views, scene);
    let t_vox = t0.elapsed().as_secs_f64() - t_depth - t_fuse;
    Ok(Fused { grid, solid, floor, max_depth, fused, closed, kept, components, cavities, t_depth, t_fuse, t_vox, hit_mean })
}

/// Occupancy marching cubes puts every surface half a voxel outside the solid samples, and samples
/// are rounded to the nearest voxel: up to a voxel of bias on every face (a table top a voxel high).
/// Move each vertex along its normal onto the mean surface hit of the nearby solid voxels (where the
/// capture views saw the surface), at most one voxel. Returns how many vertices moved.
fn snap_to_hits(mesh: &mut Mesh, grid: &Grid, solid: &[bool], hit_mean: &HashMap<u32, HitStat>) -> usize {
    let mut nrm = vec![Vec3::ZERO; mesh.positions.len()];
    for t in &mesh.triangles {
        let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
        let n = (b - a).cross(c - a);
        for &i in t { nrm[i as usize] += n; }
    }
    let vox = grid.voxel;
    let cell = |p: Vec3| -> Option<IVec3> {
        let c = ((p - grid.lo) / vox).round();
        (c.x >= 0.0 && c.y >= 0.0 && c.z >= 0.0 && (c.x as usize) < grid.n[0] && (c.y as usize) < grid.n[1] && (c.z as usize) < grid.n[2])
            .then(|| c.as_ivec3())
    };
    let is_solid = |p: Vec3| cell(p).is_some_and(|c| solid[grid.at(c.x as usize, c.y as usize, c.z as usize)]);
    // outward normals: the mesh is closed and consistently wound, so its signed volume gives the winding's sign
    let vol: f64 = mesh.triangles.iter().map(|t| {
        let [a, b, c] = t.map(|i| mesh.positions[i as usize].as_dvec3());
        a.dot(b.cross(c))
    }).sum();
    let sign = if vol < 0.0 { -1.0f32 } else { 1.0 };
    let moved: Vec<Option<(f32, Vec3)>> = mesh.positions.par_iter().zip(&nrm).map(|(&v, &n)| {
        let mut n = n.normalize_or_zero();
        if n == Vec3::ZERO { return None; }
        // outward: solid behind, empty in front
        n *= sign;
        // solid in front and empty behind: a locally inverted normal; leave the vertex where it is
        if is_solid(v + 0.6 * vox * n) && !is_solid(v - 0.6 * vox * n) { return None; }
        let c = cell(v)?;
        let (mut sum, mut w) = (0.0f32, 0.0f32);
        for dz in -SNAP_R..=SNAP_R { for dy in -SNAP_R..=SNAP_R { for dx in -SNAP_R..=SNAP_R {
            let q = c + IVec3::new(dx, dy, dz);
            if q.min_element() < 0 || q.x as usize >= grid.n[0] || q.y as usize >= grid.n[1] || q.z as usize >= grid.n[2] { continue; }
            let Some(&(ps, ds, k)) = hit_mean.get(&(grid.at(q.x as usize, q.y as usize, q.z as usize) as u32)) else { continue };
            // only hits seen from this face's side: the back face of a thin sheet must not snap onto the front's hits
            if ds.dot(n) <= 0.2 * k as f32 { continue; }
            let m = ps / k as f32;
            let s = (m - v).dot(n);
            let lateral = (m - v - s * n).length();
            if lateral < (SNAP_R as f32 + 0.5) * vox && s.abs() < 1.5 * vox { sum += s * k as f32; w += k as f32; }
        }}}
        (w > 0.0).then(|| ((sum / w).clamp(-vox, vox), n))
    }).collect();
    let mut count = 0;
    for (p, m) in mesh.positions.iter_mut().zip(&moved) {
        if let Some((d, n)) = m { *p += *d * *n; count += 1; }
    }
    count
}

/// Flat horizontal surfaces (floors, table tops, seats) come out of voxels with a 1-2 cm staircase
/// ripple. Connected near-horizontal upward regions (normal within ~14 deg of up, neighbours within
/// 0.6 voxel of each other) that are already within ~3 cm of flat and at least 0.04 square units are
/// set to their area-weighted median height. Returns the number of regions flattened.
fn flatten_planes(mesh: &mut Mesh, voxel: f32) -> usize {
    let tri = &mesh.triangles;
    let pos = &mesh.positions;
    let n = tri.len();
    let info: Vec<(Vec3, f32, f32)> = tri.iter().map(|t| {
        let [a, b, c] = t.map(|i| pos[i as usize]);
        let cr = (b - a).cross(c - a);
        (cr.normalize_or_zero(), 0.5 * cr.length(), (a.y + b.y + c.y) / 3.0)
    }).collect();
    let up: Vec<bool> = info.iter().map(|(nr, area, _)| nr.y > 0.97 && *area > 0.0).collect();
    let mut by_edge: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (f, t) in tri.iter().enumerate() {
        if !up[f] { continue; }
        for k in 0..3 { let (a, b) = (t[k], t[(k + 1) % 3]); by_edge.entry((a.min(b), a.max(b))).or_default().push(f as u32); }
    }
    let mut comp = vec![u32::MAX; n];
    let mut regions = 0;
    let mut newy: Vec<Option<f32>> = vec![None; pos.len()];
    for seed in 0..n {
        if !up[seed] || comp[seed] != u32::MAX { continue; }
        let id = seed as u32;
        let mut stack = vec![seed];
        let mut faces = Vec::new();
        comp[seed] = id;
        while let Some(f) = stack.pop() {
            faces.push(f);
            let t = tri[f];
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                for &g in &by_edge[&(a.min(b), a.max(b))] {
                    let g = g as usize;
                    if comp[g] == u32::MAX && (info[g].2 - info[f].2).abs() < 0.6 * voxel { comp[g] = id; stack.push(g); }
                }
            }
        }
        let area: f32 = faces.iter().map(|&f| info[f].1).sum();
        if area < 0.04 { continue; }
        let mut hs: Vec<(f32, f32)> = faces.iter().map(|&f| (info[f].2, info[f].1)).collect();
        hs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let quant = |q: f32| { let mut acc = 0.0; for &(h, w) in &hs { acc += w; if acc >= q * area { return h; } } hs[hs.len() - 1].0 };
        let (med, lo, hi) = (quant(0.5), quant(0.05), quant(0.95));
        if hi - lo > 1.2 * voxel { continue; }
        regions += 1;
        // move only vertices already within ~3 cm of the plane: a coarse ramp of a few big triangles passes the
        // centroid test but its corners are far from the median, so they stay; a vertex shared with an earlier
        // levelled region keeps that height
        for &f in &faces { for &v in &tri[f] {
            if newy[v as usize].is_none() && (pos[v as usize].y - med).abs() <= 1.2 * voxel { newy[v as usize] = Some(med); }
        } }
    }
    for (p, y) in mesh.positions.iter_mut().zip(newy) { if let Some(y) = y { p.y = y; } }
    regions
}

/// Snap support: voxels within this many cells of a vertex (laterally along its surface).
const SNAP_R: i32 = 2;

/// Write collision/ into `dir` and return its manifest (paths relative to the package).
fn build(g: &Gaussians, views: &[View], scene: &Value, dir: &Path) -> Result<Value> {
    let t0 = Instant::now();
    std::fs::create_dir_all(dir.join("convex"))?;
    let Fused { grid, solid, floor, max_depth, fused, closed, kept, components, cavities, t_depth, t_fuse, t_vox, hit_mean } =
        fuse(g, views, scene, &FuseOpts::collision())?;
    let cells: Vec<IVec3> = (0..grid.n[2]).flat_map(|z| (0..grid.n[1]).flat_map(move |y| (0..grid.n[0]).map(move |x| (x, y, z))))
        .filter(|&(x, y, z)| solid[grid.at(x, y, z)]).map(|(x, y, z)| IVec3::new(x as i32, y as i32, z as i32)).collect();
    eprintln!("collision: {fused} fused -> {closed} closed -> {kept} kept ({components} components) -> {} with floor; marching cubes", cells.len());
    let mut mesh = splat_mesh::mesh_occupancy(&cells, grid.lo, grid.voxel);
    let raw_triangles = mesh.triangles.len();
    let before = topology(&mesh);
    simplify(&mut mesh);
    let after = topology(&mesh);
    ensure!(after.0 <= before.0 && after.1 <= before.1, "collision simplification damaged mesh topology");
    // Positions only (topology unchanged): put the faces where the views saw the surface.
    let snapped = snap_to_hits(&mut mesh, &grid, &solid, &hit_mean);
    drop(solid);
    drop(hit_mean);
    let planes = flatten_planes(&mut mesh, grid.voxel);
    let t_mesh = t0.elapsed().as_secs_f64() - t_depth - t_fuse - t_vox;
    eprintln!("collision: {snapped} vertices snapped onto surface hits, {planes} flat regions levelled; {} -> {} triangles; {} boundary / {} nonmanifold edges", raw_triangles, mesh.triangles.len(), after.0, after.1);
    ensure!(!mesh.triangles.is_empty(), "collision mesh is empty");
    splat_mesh::write_glb(&mesh, &dir.join("collision.glb"))?;
    write_obj(&mesh, &dir.join("collision.obj"))?;
    let floor_json = json!({
        "schema": 1, "coords": scene["coords"], "meters_per_unit": scene["meters_per_unit"],
        "scale_estimate": scene["scale_estimate"], "export_floor": scene["floor"], "collision_floor": floor,
        "transform_from_source": scene["transform_from_source"],
        "note": "Scene units are metres only as far as scale_estimate is right (confidence recorded). collision_floor.y is the top of the collision floor slab; its implied_units_per_meter assumes handheld cameras ~1.5 m above the floor."
    });
    std::fs::write(dir.join("floor.json"), serde_json::to_vec_pretty(&floor_json)?)?;
    let t1 = Instant::now();
    let convex = write_convex(&mesh, dir)?;
    let t_convex = t1.elapsed().as_secs_f64();
    std::fs::write(dir.join("README.md"), README)?;
    std::fs::write(dir.join("THIRD_PARTY_NOTICES.md"), NOTICES)?;
    let stats = json!({
        "vertices": mesh.positions.len(), "triangles": mesh.triangles.len(), "triangles_before_simplification": raw_triangles,
        "boundary_edges": after.0, "nonmanifold_edges": after.1, "watertight": after == (0, 0),
        "convex_parts": convex.len(), "depth_s": t_depth, "fuse_s": t_fuse, "voxel_s": t_vox, "mesh_s": t_mesh, "convex_s": t_convex,
        "build_s": t0.elapsed().as_secs_f64(),
        "glb_bytes": std::fs::metadata(dir.join("collision.glb"))?.len(),
        "obj_bytes": std::fs::metadata(dir.join("collision.obj"))?.len(),
    });
    Ok(json!({
        "schema": 2, "mesh": "collision/collision.glb", "obj": "collision/collision.obj", "floor": "collision/floor.json",
        "convex": convex, "readme": "collision/README.md", "method": "depth_fusion", "status": "ready",
        "grid": {"voxel_units": grid.voxel, "dims": grid.n, "views": views.len(), "depth_height_px": DEPTH_HEIGHT,
            "min_alpha": MIN_ALPHA, "max_depth_units": max_depth, "occupied_rule": "hit_views>=2 && hit_views>=2*free_views", "depth_transmittance": DEPTH_TRANSMITTANCE,
            "closing_voxels": CLOSE, "min_component_frac": MIN_COMPONENT, "fused_cells": fused, "closed_cells": closed,
            "kept_cells": kept, "components": components, "cavity_cells_filled": cavities, "cells_with_floor": cells.len(), "vertices_snapped_to_hits": snapped, "flat_regions_levelled": planes},
        "floor_estimate": floor, "stats": stats,
        "convex_method": {"name": "vhacd", "implementation": "parry3d-f64", "version": "0.25.3", "license": "Apache-2.0",
            "resolution": 64, "max_convex_hulls": 64, "fill_mode": "surface_only", "max_vertices_per_hull": 128},
        "coords": scene["coords"], "meters_per_unit": scene["meters_per_unit"],
        "transform_from_source": scene["transform_from_source"], "scale_estimate": scene["scale_estimate"],
    }))
}

/// Dilate along one axis by `r` cells (running window); `invert` dilates the complement (erosion).
pub(crate) fn dilate_axis(solid: &mut [bool], grid: &Grid, axis: usize, r: usize, invert: bool) {
    let n = grid.n;
    let stride = [1, n[0], n[0] * n[1]][axis];
    let len = n[axis];
    let lines: Vec<usize> = (0..grid.len()).filter(|&i| (i / stride) % len == 0).collect();
    let src = solid.to_vec();
    // Each line is independent; write through a raw pointer split by line.
    let ptr = solid.as_mut_ptr() as usize;
    lines.par_iter().for_each(|&start| {
        let out = ptr as *mut bool;
        let mut last: isize = -(r as isize) - 1; // last set position seen
        let mut buf = vec![false; len];
        for k in 0..len { if src[start + k * stride] != invert { last = k as isize; } buf[k] = k as isize - last <= r as isize; }
        let mut next: isize = (len + r + 1) as isize;
        for k in (0..len).rev() {
            if src[start + k * stride] != invert { next = k as isize; }
            let v = buf[k] || next - k as isize <= r as isize;
            // SAFETY: lines are disjoint sets of indices.
            unsafe { *out.add(start + k * stride) = v != invert; }
        }
    });
}

fn close(solid: &mut [bool], grid: &Grid, r: usize) {
    for axis in 0..3 { dilate_axis(solid, grid, axis, r, false); }
    for axis in 0..3 { dilate_axis(solid, grid, axis, r, true); }
}

/// Drop 6-connected components under [`MIN_COMPONENT`] of the occupied volume.
fn drop_small_components(solid: &mut [bool], grid: &Grid) -> (usize, usize) {
    let total = solid.iter().filter(|&&s| s).count();
    let mut label = vec![0u32; grid.len()];
    let mut sizes = vec![0usize];
    let n = grid.n;
    let mut stack = Vec::new();
    for start in 0..grid.len() {
        if !solid[start] || label[start] != 0 { continue; }
        let id = sizes.len() as u32;
        let mut size = 0;
        label[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
            let mut visit = |j: usize| if solid[j] && label[j] == 0 { label[j] = id; stack.push(j); };
            if x > 0 { visit(i - 1); } if x + 1 < n[0] { visit(i + 1); }
            if y > 0 { visit(i - n[0]); } if y + 1 < n[1] { visit(i + n[0]); }
            if z > 0 { visit(i - n[0] * n[1]); } if z + 1 < n[2] { visit(i + n[0] * n[1]); }
        }
        sizes.push(size);
    }
    let min = (MIN_COMPONENT * total as f64).ceil() as usize;
    let mut kept = 0;
    for i in 0..grid.len() {
        if solid[i] {
            if sizes[label[i] as usize] < min { solid[i] = false; } else { kept += 1; }
        }
    }
    (kept, sizes.iter().skip(1).filter(|&&s| s >= min).count())
}

/// Fill enclosed empty pockets under `max` cells (voids inside furniture and walls: nothing can
/// enter them, and a body spawned there would be trapped). Rooms are far larger and stay open.
fn fill_cavities(solid: &mut [bool], grid: &Grid, max: usize) -> usize {
    let n = grid.n;
    let mut seen = vec![false; grid.len()];
    let mut filled = 0;
    let mut stack = Vec::new();
    let mut pocket = Vec::new();
    for start in 0..grid.len() {
        if solid[start] || seen[start] { continue; }
        seen[start] = true;
        stack.push(start);
        pocket.clear();
        let mut open = false;
        while let Some(i) = stack.pop() {
            let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
            open |= x == 0 || y == 0 || z == 0 || x + 1 == n[0] || y + 1 == n[1] || z + 1 == n[2];
            if pocket.len() <= max { pocket.push(i); }
            let mut visit = |j: usize| if !solid[j] && !seen[j] { seen[j] = true; stack.push(j); };
            if x > 0 { visit(i - 1); } if x + 1 < n[0] { visit(i + 1); }
            if y > 0 { visit(i - n[0]); } if y + 1 < n[1] { visit(i + n[0]); }
            if z > 0 { visit(i - n[0] * n[1]); } if z + 1 < n[2] { visit(i + n[0] * n[1]); }
        }
        if !open && pocket.len() <= max {
            for &i in &pocket { solid[i] = true; }
            filled += pocket.len();
        }
    }
    filled
}

/// Always add a solid floor slab (top at the floor) under the convex hull of the occupied XZ
/// footprint and the camera path, plus a 0.5-unit rim. The floor is
/// the export's detected floor (y = 0), else the dominant upward-facing surface below the cameras.
fn floor_slab(solid: &mut [bool], grid: &Grid, views: &[View], scene: &Value) -> Value {
    let n = grid.n;
    let mut cam_y: Vec<f32> = views.iter().map(|v| v.c.y).collect();
    let cam_med = quantile(&mut cam_y, 0.5);
    let cam_low = quantile(&mut cam_y, 0.1);
    let cam_high = quantile(&mut cam_y, 0.9);
    let layer_of = |y: f32| ((y - grid.lo.y) / grid.voxel - 0.5).round() as isize;
    // Upward-facing surface cells per layer.
    let mut tops = vec![0usize; n[1]];
    for z in 0..n[2] { for y in 0..n[1] - 1 { for x in 0..n[0] {
        if solid[grid.at(x, y, z)] && !solid[grid.at(x, y + 1, z)] { tops[y] += 1; }
    }}}
    let detected = scene["floor"]["detected"].as_bool() == Some(true);
    let (layer, method) = if detected {
        (layer_of(0.0), "export_floor")
    } else {
        // Handheld cameras sit ~1.4-1.7 m above the floor; with an unreliable scale search a wide
        // band below them and take the strongest horizontal plane.
        let lo = layer_of(cam_low - 4.0 * (cam_high - cam_low).max(1.0)).max(0);
        let hi = layer_of(cam_low - 0.3).min(n[1] as isize - 2);
        match (lo..=hi).max_by_key(|&l| tops[l as usize]).filter(|&l| tops[l as usize] > 0) {
            Some(l) => (l, "dominant_upward_plane_below_cameras"),
            None => (layer_of(0.0), "export_frame_y0"),
        }
    };
    let layer = layer.clamp(1, n[1] as isize - 2) as usize;
    let floor_y = grid.lo.y + (layer as f32 + 0.5) * grid.voxel;
    let support = tops[layer] as f64 / tops.iter().copied().max().unwrap_or(1).max(1) as f64;
    // Footprint: columns with anything between the floor and above the cameras, closed and hole-filled.
    let top_layer = (layer_of(cam_high + 1.0).max(layer as isize) as usize).min(n[1] - 1);
    let mut foot = vec![false; n[0] * n[2]];
    for z in 0..n[2] { for x in 0..n[0] {
        foot[z * n[0] + x] = (layer.saturating_sub(2)..=top_layer).any(|y| solid[grid.at(x, y, z)]);
    }}
    // The operator walked where the cameras are: floor there, though no camera looks straight down.
    let disc = (0.8 / grid.voxel).ceil() as isize;
    for v in views {
        let (cx, cz) = (((v.c.x - grid.lo.x) / grid.voxel).round() as isize, ((v.c.z - grid.lo.z) / grid.voxel).round() as isize);
        for dz in -disc..=disc { for dx in -disc..=disc {
            let (x, z) = (cx + dx, cz + dz);
            if dx * dx + dz * dz <= disc * disc && x >= 0 && z >= 0 && (x as usize) < n[0] && (z as usize) < n[2] {
                foot[z as usize * n[0] + x as usize] = true;
            }
        }}
    }
    let flat = Grid { lo: Vec3::ZERO, voxel: 1.0, n: [n[0], n[2], 1] };
    // Bridge ~0.6 units of missing floor between observed columns.
    let r = ((0.6 / grid.voxel).ceil() as usize).max(3);
    for axis in 0..2 { dilate_axis(&mut foot, &flat, axis, r, false); }
    for axis in 0..2 { dilate_axis(&mut foot, &flat, axis, r, true); }
    // Fill holes: empty columns not reachable from the border.
    let mut outside = vec![false; foot.len()];
    let mut stack: Vec<usize> = (0..foot.len()).filter(|&i| {
        let (x, z) = (i % n[0], i / n[0]);
        (x == 0 || z == 0 || x + 1 == n[0] || z + 1 == n[2]) && !foot[i]
    }).collect();
    for &i in &stack { outside[i] = true; }
    while let Some(i) = stack.pop() {
        let (x, z) = (i % n[0], i / n[0]);
        let mut visit = |j: usize| if !foot[j] && !outside[j] { outside[j] = true; stack.push(j); };
        if x > 0 { visit(i - 1); } if x + 1 < n[0] { visit(i + 1); }
        if z > 0 { visit(i - n[0]); } if z + 1 < n[2] { visit(i + n[0]); }
    }
    // Rooms seen only through a doorway have no observed floor: extend the slab to the convex hull of
    // the footprint plus a rim, so anything rolling off the scanned area still lands.
    let rim = (0.5 / grid.voxel).ceil() as i64;
    let mut corners: Vec<(i64, i64)> = (0..foot.len()).filter(|&i| !outside[i])
        .flat_map(|i| { let (x, z) = ((i % n[0]) as i64, (i / n[0]) as i64);
            [(x - rim, z - rim), (x + rim, z - rim), (x - rim, z + rim), (x + rim, z + rim)] })
        .collect();
    let hull = hull_2d(&mut corners);
    for (i, o) in outside.iter_mut().enumerate() {
        let (x, z) = ((i % n[0]) as i64, (i / n[0]) as i64);
        *o = !(hull.len() >= 3 && (0..hull.len()).all(|k| {
            let (a, b) = (hull[k], hull[(k + 1) % hull.len()]);
            (b.0 - a.0) * (z - a.1) - (b.1 - a.1) * (x - a.0) >= 0
        }));
    }
    let thick = ((SLAB / grid.voxel).ceil() as usize).max(2);
    let mut slab_cells = 0;
    let mut min = [f32::MAX; 2];
    let mut max = [f32::MIN; 2];
    for z in 0..n[2] { for x in 0..n[0] {
        if outside[z * n[0] + x] { continue; }
        slab_cells += 1;
        let (wx, wz) = (grid.lo.x + x as f32 * grid.voxel, grid.lo.z + z as f32 * grid.voxel);
        min = [min[0].min(wx), min[1].min(wz)];
        max = [max[0].max(wx), max[1].max(wz)];
        for y in (layer + 1).saturating_sub(thick)..=layer { solid[grid.at(x, y, z)] = true; }
    }}
    let cam_above = cam_med - floor_y;
    json!({"y": floor_y, "normal": [0.0, 1.0, 0.0], "method": method, "slab_thickness_units": thick as f32 * grid.voxel,
        "slab_columns": slab_cells, "extent_xz": {"min": min, "max": max}, "plane_support": support,
        "camera_height_above_floor_units": cam_above, "implied_units_per_meter": cam_above / 1.5})
}

/// Counter-clockwise convex hull (Andrew's monotone chain), no collinear points.
fn hull_2d(p: &mut Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    p.sort_unstable();
    p.dedup();
    if p.len() < 3 { return p.clone(); }
    let cross = |o: (i64, i64), a: (i64, i64), b: (i64, i64)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut h: Vec<(i64, i64)> = Vec::with_capacity(2 * p.len());
    for pass in 0..2 {
        let start = h.len();
        let pts: Box<dyn Iterator<Item = &(i64, i64)>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in pts {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0 { h.pop(); }
            h.push(q);
        }
        h.pop();
    }
    h
}

fn topology(mesh: &Mesh) -> (usize, usize) {
    let mut edges = HashMap::<(u32,u32),u32>::new();
    for t in &mesh.triangles { for j in 0..3 {
        let (a,b) = (t[j],t[(j+1)%3]);
        *edges.entry((a.min(b),a.max(b))).or_default() += 1;
    }}
    (edges.values().filter(|&&n| n == 1).count(), edges.values().filter(|&&n| n > 2).count())
}

fn simplify(mesh: &mut Mesh) {
    if mesh.triangles.len() <= TARGET_TRIANGLES { return; }
    let positions: Vec<_> = mesh.positions.iter().map(|p| p.to_array()).collect();
    let indices: Vec<_> = mesh.triangles.iter().flatten().copied().collect();
    // The default edge-collapse simplifier respects topology (no sloppy/prune mode).
    let result = meshopt::simplify_decoder(&indices, &positions, TARGET_TRIANGLES*3, 1.0,
        meshopt::SimplifyOptions::LockBorder, None);
    mesh.triangles = result.chunks_exact(3).map(|t| [t[0],t[1],t[2]]).collect();
    repair_collapses(mesh);
    mesh.compact();
}

/// Aggressive edge collapse can leave zero-volume, opposite-facing triangle pairs and
/// closed shells sharing an edge. Cancel only opposite pairs (preserves the oriented
/// boundary), then give each paired surface its own edge midpoint. This repairs the
/// indexed topology without moving any surviving surface or filling physical holes.
fn repair_collapses(mesh: &mut Mesh) {
    let mut faces = HashMap::<[u32;3], [Vec<usize>;2]>::new();
    for (i,&t) in mesh.triangles.iter().enumerate() {
        let parity = ((t[0]>t[1]) as usize + (t[0]>t[2]) as usize + (t[1]>t[2]) as usize)%2;
        let mut key = t; key.sort_unstable();
        faces.entry(key).or_default()[parity].push(i);
    }
    let mut keep = vec![true;mesh.triangles.len()];
    for pair in faces.values() {
        for (&a,&b) in pair[0].iter().zip(&pair[1]) { keep[a]=false; keep[b]=false; }
    }
    mesh.triangles = mesh.triangles.iter().enumerate().filter_map(|(i,&t)| keep[i].then_some(t)).collect();
    drop(faces);
    let mut edges = HashMap::<(u32,u32),Vec<(usize,usize)>>::new();
    for (i,t) in mesh.triangles.iter().enumerate() { for k in 0..3 {
        let (a,b)=(t[k],t[(k+1)%3]); edges.entry((a.min(b),a.max(b))).or_default().push((i,k));
    }}
    let normals: Vec<_> = mesh.triangles.iter().map(|t| {
        let [p,q,r]=t.map(|v| mesh.positions[v as usize]); (q-p).cross(r-p).normalize_or_zero()
    }).collect();
    let mut splits = HashMap::<(usize,usize),u32>::new();
    // Sort: hash iteration order must not change exported geometry/indices.
    let mut bad: Vec<_> = edges.into_iter().filter(|(_,v)| v.len()>2).collect();
    bad.sort_unstable_by_key(|(key,_)| *key);
    for ((a,b), incident) in bad {
        let (mut forward, mut backward): (Vec<_>,Vec<_>) = incident.into_iter()
            .partition(|&(i,k)| mesh.triangles[i][k] == a);
        if forward.len()!=backward.len() { continue; } // Pre-existing open/non-oriented edge.
        forward.sort_unstable(); backward.sort_unstable();
        for f in forward {
            let n=normals[f.0];
            let idx=(0..backward.len()).max_by(|&i,&j| n.dot(normals[backward[i].0]).total_cmp(&n.dot(normals[backward[j].0]))).unwrap();
            let back=backward.remove(idx);
            let id=mesh.positions.len() as u32;
            mesh.positions.push((mesh.positions[a as usize]+mesh.positions[b as usize])*0.5);
            mesh.colors.push([88,196,210]);
            splits.insert(f,id); splits.insert(back,id);
        }
    }
    let mut triangles=Vec::with_capacity(mesh.triangles.len()+splits.len()*3);
    for (i,&t) in mesh.triangles.iter().enumerate() {
        if !(0..3).any(|k| splits.contains_key(&(i,k))) { triangles.push(t); continue; }
        let center=t.iter().map(|&v| mesh.positions[v as usize]).sum::<Vec3>()/3.0;
        let id=mesh.positions.len() as u32; mesh.positions.push(center); mesh.colors.push([88,196,210]);
        let mut polygon=Vec::with_capacity(6);
        for k in 0..3 { polygon.push(t[k]); if let Some(&mid)=splits.get(&(i,k)) { polygon.push(mid); } }
        for j in 0..polygon.len() { triangles.push([id,polygon[j],polygon[(j+1)%polygon.len()]]); }
    }
    mesh.triangles=triangles;
}

fn write_obj(mesh: &Mesh, path: &Path) -> Result<()> {
    let mut out = BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "# splat collision: same Y-up frame and units as scene.spz\no SplatCollision")?;
    for p in &mesh.positions { writeln!(out, "v {} {} {}", p.x,p.y,p.z)?; }
    for t in &mesh.triangles { writeln!(out, "f {} {} {}", t[0]+1,t[1]+1,t[2]+1)?; }
    out.flush()?;
    Ok(())
}

fn write_convex(mesh: &Mesh, dir: &Path) -> Result<Vec<String>> {
    let points: Vec<_> = mesh.positions.iter().map(|p| Point3::new(p.x as f64,p.y as f64,p.z as f64)).collect();
    let params = VHACDParameters { resolution: 64, max_convex_hulls: 64, concavity: 0.005,
        // Whole scenes are shells: do not turn the room interior into a solid dynamic hull.
        fill_mode: FillMode::SurfaceOnly, ..Default::default() };
    let decomposition = VHACD::decompose(&params, &points, &mesh.triangles, true);
    let mut paths = vec![];
    for (mut positions, mut triangles) in decomposition.compute_exact_convex_hulls(&points, &mesh.triangles) {
        if triangles.is_empty() { continue; }
        // PhysX convex MeshColliders allow at most 255 faces. A hull of <=128 support
        // vertices has <=252 triangular faces; recomputing its hull keeps it convex.
        if positions.len() > 128 {
            let mut chosen = HashSet::new();
            for i in 0..128 {
                let y = 1.0 - 2.0 * (i as f32 + 0.5) / 128.0;
                let angle = i as f32 * 2.3999631;
                let radius = (1.0-y*y).sqrt();
                let direction = Vec3::new(radius*angle.cos(), y, radius*angle.sin());
                if let Some((idx,_)) = positions.iter().enumerate().max_by(|(_,a),(_,b)| {
                    direction.dot(Vec3::new(a.x as f32,a.y as f32,a.z as f32)).total_cmp(&direction.dot(Vec3::new(b.x as f32,b.y as f32,b.z as f32)))
                }) { chosen.insert(idx); }
            }
            let mut chosen: Vec<_> = chosen.into_iter().collect();
            chosen.sort_unstable();
            let support: Vec<_> = chosen.into_iter().map(|i| positions[i]).collect();
            positions = parry3d::transformation::convex_hull(&support).0;
        }
        // GLB/OBJ store f32. Quantizing a double-precision hull can make its very thin
        // faces collinear or slightly concave; re-hull those exact output coordinates.
        let quantized: Vec<_> = positions.iter().map(|p| Point3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64)).collect();
        (positions, triangles) = parry3d::transformation::try_convex_hull(&quantized)
            .map_err(|e| anyhow::anyhow!("collision convex hull: {e:?}"))?;
        let hull = Mesh { colors: vec![[88,196,210];positions.len()],
            positions: positions.iter().map(|p| Vec3::new(p.x as f32,p.y as f32,p.z as f32)).collect(), triangles };
        ensure!(convex(&hull), "V-HACD returned a non-convex hull");
        let name = format!("hull-{:03}", paths.len());
        splat_mesh::write_glb(&hull, &dir.join(format!("convex/{name}.glb")))?;
        write_obj(&hull, &dir.join(format!("convex/{name}.obj")))?;
        paths.push(format!("collision/convex/{name}.glb"));
    }
    ensure!(!paths.is_empty(), "V-HACD produced no convex pieces");
    Ok(paths)
}

/// Check actual f32 output geometry, using double precision for the half-space test.
fn convex(mesh: &Mesh) -> bool {
    mesh.triangles.iter().all(|t| {
        let [a,b,c]=t.map(|i| mesh.positions[i as usize].as_dvec3());
        let n=(b-a).cross(c-a).normalize_or_zero();
        n.length_squared()>0.0 && mesh.positions.iter().all(|p| n.dot(p.as_dvec3()-a)<=0.001)
    })
}

/// UnityGaussianSplatting's PLY reader always interprets f_rest as three 15-coefficient
/// channel blocks. Pad lower-degree SH instead of letting G/B coefficients land in R.
fn write_unity_ply(g: &Gaussians, path: &Path) -> Result<()> {
    let mut out = BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "ply\nformat binary_little_endian 1.0\ncomment Splat collision export: Y-up; SH padded to degree 3\nelement vertex {}", g.len())?;
    for name in ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"] {
        writeln!(out, "property float {name}")?;
    }
    for i in 0..45 { writeln!(out, "property float f_rest_{i}")?; }
    writeln!(out, "end_header")?;
    let k = g.sh_coeffs();
    for i in 0..g.len() {
        for v in g.pos[i].iter().chain(g.scale[i].iter()).chain(g.rot[i].iter()).chain(std::iter::once(&g.opacity[i])).chain(g.dc[i].iter()) {
            out.write_all(&v.to_le_bytes())?;
        }
        for c in 0..3 { for j in 0..15 {
            let v = if j < k { g.rest[i*k*3+j*3+c] } else { 0.0 };
            out.write_all(&v.to_le_bytes())?;
        }}
    }
    out.flush()?;
    Ok(())
}

/// Unity project folder (explicit `--unity` only: it holds a full-size PLY copy of the splats).
fn write_unity(g: &Gaussians, collision: &Path, dir: &Path) -> Result<()> {
    let assets = dir.join("Assets/SplatScene");
    std::fs::create_dir_all(assets.join("Editor"))?;
    std::fs::copy(collision.join("collision.obj"), assets.join("collision.obj"))?;
    std::fs::copy(collision.join("floor.json"), assets.join("floor.json"))?;
    write_unity_ply(g, &assets.join("scene.ply"))?;
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(std::fs::read(collision.join("collision.obj"))?);
    let guid: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
    std::fs::write(assets.join("collision.obj.meta"), UNITY_META.replace("GUID", &guid))?;
    std::fs::write(assets.join("Editor/SplatCollisionSetup.cs"), UNITY_EDITOR)?;
    std::fs::write(dir.join("README.md"), README)?;
    Ok(())
}

const UNITY_META: &str = "fileFormatVersion: 2\nguid: GUID\nModelImporter:\n  serializedVersion: 22200\n  meshes:\n    globalScale: 1\n    meshCompression: 0\n    addColliders: 1\n    useFileScale: 0\n    isReadable: 1\n  importAnimation: 0\n";
const UNITY_EDITOR: &str = include_str!("collision/UnitySetup.cs");
const README: &str = include_str!("collision/README.md");
const NOTICES: &str = include_str!("collision/THIRD_PARTY_NOTICES.md");

#[cfg(test)]
mod tests {
    use super::*;

    /// A round table top (opaque, flat splats) under a layer of faint haze, a floor, and a ring of
    /// handheld-height cameras looking at it. Returns the splats and views in a Y-up scene frame.
    fn table_with_haze() -> (Gaussians, Vec<View>) {
        let mut g = Gaussians::default();
        let logit = |a: f32| (a / (1.0 - a)).ln();
        let mut add = |p: [f32; 3], s: [f32; 3], a: f32| {
            g.pos.push(p); g.scale.push(s.map(f32::ln)); g.rot.push([1.0, 0.0, 0.0, 0.0]); g.opacity.push(logit(a)); g.dc.push([0.0; 3]);
        };
        let mut seed = 7u32;
        let mut rnd = move || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; seed as f32 / u32::MAX as f32 };
        let (top, rad) = (0.42f32, 0.38f32);
        // table top: flat discs every 1.5 cm
        let mut x = -rad;
        while x <= rad { let mut z = -rad;
            while z <= rad { if x * x + z * z <= rad * rad { add([x, top, z], [0.012, 0.002, 0.012], 0.95); } z += 0.015; }
            x += 0.015; }
        // floor out to 2.5 units
        let mut x = -2.5f32;
        while x <= 2.5 { let mut z = -2.5f32;
            while z <= 2.5 { if (x * x + z * z).sqrt() > rad { add([x, 0.0, z], [0.03, 0.003, 0.03], 0.95); } z += 0.04; }
            x += 0.04; }
        // haze: faint splats 2..25 cm above the table, invisible in colour but enough to stop some views' median depth
        // (optical depth ~1 along a grazing ray: some views stop in it, most see through)
        for _ in 0..1000 {
            let (r, a) = (rad * rnd().sqrt(), std::f32::consts::TAU * rnd());
            add([r * a.cos(), top + 0.02 + 0.23 * rnd(), r * a.sin()], [0.015, 0.015, 0.015], 0.06);
        }
        let views = (0..48).map(|k| {
            let a = std::f32::consts::TAU * k as f32 / 48.0;
            let c = Vec3::new(2.0 * a.cos(), 1.45, 2.0 * a.sin());
            let fwd = (Vec3::new(0.0, top, 0.0) - c).normalize();
            let right = fwd.cross(Vec3::Y).normalize();
            let down = fwd.cross(right);
            View { name: format!("v{k}"), c, r: Mat3::from_cols(right, down, fwd), f: 300.0, w: 480, h: DEPTH_HEIGHT }
        }).collect();
        (g, views)
    }

    /// Height of the top solid sample in the column over the table centre.
    fn top_over_centre(f: &Fused) -> f32 {
        let gr = &f.grid;
        let ix = ((0.0 - gr.lo.x) / gr.voxel).round() as usize;
        let iz = ((0.0 - gr.lo.z) / gr.voxel).round() as usize;
        (0..gr.n[1]).rev().find(|&y| f.solid[gr.at(ix, y, iz)] && (gr.lo.y + y as f32 * gr.voxel) < 1.0)
            .map(|y| gr.lo.y + y as f32 * gr.voxel).unwrap_or(f32::NAN)
    }

    #[test]
    fn haze_over_a_table_is_carved_not_fused() {
        let (g, views) = table_with_haze();
        let scene = json!({"floor": {"detected": true}});
        let new = fuse(&g, &views, &scene, &FuseOpts::collision()).unwrap();
        let top = top_over_centre(&new);
        // collision surface = top sample + half a voxel; must sit within one voxel of the real table top
        assert!((top + 0.5 * new.grid.voxel - 0.42).abs() <= new.grid.voxel + 0.005, "table top sample {top}");
        // the previous rule (median depth, a third of views suffice, 5 cm voxels, wide closing) kept the haze as a shell
        let old = FuseOpts { voxel: 0.05, close: 2, transmittance: 0.5, solid_ratio: 2.0, carve_gap: 2.5, ..FuseOpts::collision() };
        let top_old = top_over_centre(&fuse(&g, &views, &scene, &old).unwrap());
        let (err_new, err_old) = (top + 0.5 * new.grid.voxel - 0.42, top_old + 0.5 * old.voxel - 0.42);
        eprintln!("collision table top error: new {err_new:.3}, old rule {err_old:.3}");
        assert!(err_old > 0.04 && err_old > 2.0 * err_new.abs(), "old rule {err_old} vs new {err_new}");
    }
}
