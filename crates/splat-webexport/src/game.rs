//! `vauss export-game`: a double-click (file://) first-person voxel game folder for a web package.
//! `vauss run` writes one at `<run>/exports/game/` by default (SPLAT_EXPORT_GAME=0: skip).
//!
//! The world is first levelled (true vertical from the capture cameras' right axes or a RANSAC floor
//! plane). The voxel world is the collision export's depth fusion ([`collision::fuse`]: the splats' median
//! depth per capture camera, hit/free voting, closing, small components dropped, pockets filled, floor slab)
//! at the requested edge (default: the smallest of 3/4/5 cm within the voxel budget), cropped to the capture
//! region, thin surfaces thickened away from the cameras, a 3-cell shell behind walls and ceiling, the floor
//! slab kept under the observed footprint (+0.3 m) and the capture path carved free (the operator walked
//! there). Cells take a denoised splat colour (opaque, luminance-trimmed), unseen floor the nearest observed
//! floor colour, interior cells the nearest coloured cell's; surface colours are lightly smoothed.
//! The page ([`PAGE`], game/index.html) gets the world as one binary (CONTRACT: "VXG1", 32^3 chunks of bit
//! masks + sRGB) in data/voxels-NN.js parts, the viewer's splat model (its full model when the run has one;
//! floaters on the capture path and near the cameras that occupancy does not support dropped, far background
//! without view-dependent colour) in data/splat-NN.js parts (Spark, as in export-viewer), the levelling
//! transform and the spawn: level, on the floor at the first capture camera's column, facing open space.
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use glam::{DMat3, DMat4, DQuat, DVec3, Vec3};
use hashbrown::HashMap;
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::collision::{self, Fused, Grid, dilate_axis};
use crate::viewer;

const PAGE: &str = include_str!("game/index.html");
/// Chunk edge (cells) of the voxel binary.
const CHUNK: usize = 32;
/// Standing eye height (m).
const EYE_M: f64 = 1.6;
/// Splats below this opacity do not colour cells.
const MIN_ALPHA: f32 = 0.05;
/// Minimum wall / floor thickness (m), at least 2 cells.
const THICK_M: f64 = 0.08;
/// Crop: cells farther than this (m) outside the capture cameras' XZ box, or above them, are dropped.
const CROP_XZ_M: f64 = 5.0;
const CROP_UP_M: f64 = 2.0;
/// Below the floor slab top (m).
const CROP_DOWN_M: f64 = 0.5;
/// Depth-fusion range (m): at least this far from each camera counts as scene, not background.
const MIN_DEPTH_M: f64 = 7.0;
/// Floor-plane tolerance (m) for the spawn column.
const SPAWN_FLOOR_TOL_M: f64 = 0.15;
/// The operator walked the capture path: cells within this horizontal radius (m, plus one cell) of each capture
/// camera and of the segments between consecutive ones are free from the floor up to the camera + CARVE_UP_M
/// (at least head height). Segments longer than CARVE_MAX_SEG_M are tracking jumps.
const CARVE_R_M: f64 = 0.35;
const CARVE_UP_M: f64 = 0.25;
const CARVE_MAX_SEG_M: f64 = 1.5;
/// The player's body box (m): half width and height.
const BODY_HALF_M: f64 = 0.25;
const BODY_H_M: f64 = 1.75;
/// Splat floaters (the game's splat copy only): opacity below FLOATER_ALPHA or largest axis above
/// FLOATER_SCALE_M, centre inside the carved capture path or within FLOATER_NEAR_M of a capture camera, and no
/// solid voxel within SUPPORT_CELLS (in front of the observed depth). Real surfaces near the path stay. Splats outside the crop box (beyond the room) keep only their base
/// colour, at most FAR_ALPHA opaque.
const FLOATER_ALPHA: f32 = 0.5;
const FLOATER_SCALE_M: f64 = 0.10;
/// A splat is supported by occupancy with a solid voxel within this many cells.
const SUPPORT_CELLS: i64 = 2;
const FLOATER_NEAR_M: f64 = 1.0;
const FAR_ALPHA: f32 = 0.8;
/// Cells added behind the room's outer surfaces (walls, ceiling: nothing solid behind them), so a crater shows
/// wall material, not the void; the floor slab is at least this many cells thick too.
const SHELL_CELLS: usize = 3;
/// Surface colour smoothing: blend toward the mean of the similar 6-neighbours (sRGB distance below).
const SMOOTH: f32 = 0.3;
const SMOOTH_MAX_DIST: f32 = 40.0;
/// Voxel world budget (the splat model is the viewer's, unbudgeted): the default edge is the smallest of
/// AUTO_CM whose world has at most AUTO_MAX_VOXELS cells and AUTO_MAX_VOXEL_BYTES of data/voxels-NN.js.
const AUTO_CM: [f64; 3] = [3.0, 4.0, 5.0];
const AUTO_MAX_VOXELS: usize = 2_000_000;
const AUTO_MAX_VOXEL_BYTES: u64 = 25_000_000;
const GREY: [u8; 3] = [128, 128, 128];
/// Interior / unseen cells take the nearest coloured cell's colour within this distance (m), else grey.
const SPREAD_M: f64 = 0.3;
/// The floor slab is kept under observed columns (any splat-coloured cell from the floor up to the cameras,
/// or within 0.8 m of a camera) dilated by this margin (m), holes filled.
const SLAB_MARGIN_M: f64 = 0.3;
/// Unseen floor takes the 2D-nearest observed floor colour within this distance (m), else their median.
const FLOOR_SPREAD_M: f64 = 0.15;
const SH_C0: f32 = 0.282_094_8;

fn voxel_cm(s: &str) -> std::result::Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if (2.0..=5.0).contains(&v) { Ok(v) } else { Err("voxel size must be 2..5 cm".into()) }
}

#[derive(clap::Args, Debug, Clone)]
pub struct ExportGameArgs {
    /// A `vauss run` dir (uses exports/web) or a web package dir.
    pub source: PathBuf,
    /// Output folder (default: <run>/exports/game, or <package>/../game).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Voxel edge in centimetres (2..5), by the package's metric scale (default: the smallest of 3/4/5 cm
    /// whose world fits the voxel budget).
    #[arg(long, value_parser = voxel_cm)]
    pub voxel_cm: Option<f64>,
    /// Splat model shown in splat mode: web dir, scene.spz or candidate dir (default: the viewer's full model,
    /// a cloud run's uncapped candidate, else the package's).
    #[arg(long)]
    pub full: Option<PathBuf>,
    /// Page title (default: the run's source file name).
    #[arg(long)]
    pub title: Option<String>,
    /// Raw MB per data/<key>-NN.js part.
    #[arg(long, default_value_t = 4.0)]
    pub part_mb: f64,
}

impl ExportGameArgs {
    /// The defaults `vauss run` uses for `<run>`.
    pub fn for_run(run: &Path) -> Self {
        Self { source: run.to_owned(), out: None, voxel_cm: None, full: None, title: None, part_mb: 4.0 }
    }
}

/// The final voxel world: `dims` cells from `origin` (min corner of cell 0), solid flags and sRGB colours.
struct World {
    origin: Vec3,
    edge: f32,
    dims: [usize; 3],
    solid: Vec<bool>,
    rgb: Vec<[u8; 3]>,
    /// Surface cells some capture frame saw (facing it, depth agreeing); all true when frames are unavailable.
    seen: Vec<bool>,
}

impl World {
    fn at(&self, x: usize, y: usize, z: usize) -> usize { (z * self.dims[1] + y) * self.dims[0] + x }
}

/// Lengthen every solid run shorter than `t` cells along each axis to `t`, growing it into empty cells on
/// the side away from `centre` (cell coordinates): behind walls and under floors, not into the room.
fn thicken(solid: &mut [bool], n: [usize; 3], t: usize, centre: Vec3) {
    for axis in 0..3 {
        let stride = [1, n[0], n[0] * n[1]][axis];
        let len = n[axis];
        let (o1, o2) = ((axis + 1) % 3, (axis + 2) % 3);
        for b in 0..n[o2] { for a in 0..n[o1] {
            let mut p = [0usize; 3];
            p[o1] = a;
            p[o2] = b;
            let start = (p[2] * n[1] + p[1]) * n[0] + p[0];
            let mut k = 0;
            while k < len {
                if !solid[start + k * stride] { k += 1; continue; }
                let s = k;
                while k < len && solid[start + k * stride] { k += 1; }
                let run = k - s;
                if run >= t { continue; }
                let mid = (s + k) as f32 * 0.5;
                if mid >= centre[axis] {
                    // Grow towards higher indices.
                    let mut e = k;
                    while e < len && e - s < t && !solid[start + e * stride] { solid[start + e * stride] = true; e += 1; }
                    k = e;
                } else {
                    let mut e = s;
                    while e > 0 && k - e < t && !solid[start + (e - 1) * stride] { e -= 1; solid[start + e * stride] = true; }
                }
            }
        }}
    }
}

/// Denoised colour of the splats centred in each cell (sparse: cell index -> rgb sums and weight): only the
/// opaque ones (alpha >= 0.5) when there are at least 3, the top and bottom 20% by luminance dropped (5 or
/// more), the rest averaged weighted by opacity.
fn splat_colours(g: &crate::ply::Gaussians, grid: &Grid) -> HashMap<u32, [f32; 4]> {
    let mut cells: HashMap<u32, Vec<([f32; 3], f32)>> = HashMap::new();
    for i in 0..g.len() {
        let a = g.alpha(i);
        if !(a >= MIN_ALPHA) { continue; }
        let p = Vec3::from(g.pos[i]);
        if !p.is_finite() { continue; }
        let Some(k) = grid.index(p) else { continue };
        cells.entry(k).or_default().push((g.dc[i].map(|d| (0.5 + SH_C0 * d).clamp(0.0, 1.0)), a));
    }
    cells.into_iter().map(|(k, mut v)| {
        if v.iter().filter(|s| s.1 >= 0.5).count() >= 3 { v.retain(|s| s.1 >= 0.5); }
        let luma = |c: &[f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let kept: &[([f32; 3], f32)] = if v.len() >= 5 {
            v.sort_by(|a, b| luma(&a.0).total_cmp(&luma(&b.0)));
            let cut = v.len() / 5;
            &v[cut..v.len() - cut]
        } else { &v };
        let mut e = [0.0f32; 4];
        for (c, a) in kept { for j in 0..3 { e[j] += c[j] * a; } e[3] += a; }
        (k, e)
    }).collect()
}

fn to_srgb8(c: [f32; 4]) -> [u8; 3] {
    [0, 1, 2].map(|j| (c[j] / c[3].max(1e-9) * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// Crop, thicken and colour the fused grid into the game world.
/// `cams`: every capture camera (fusion views); `path`: the capture cameras in capture order.
/// The carved capture-path capsules as fusion-grid cells (the splats centred in them are dropped too).
struct Carve {
    grid: Grid,
    mask: Vec<bool>,
}

impl Carve {
    fn contains(&self, p: Vec3) -> bool { self.grid.index(p).is_some_and(|i| self.mask[i as usize]) }
}

fn build_world(fused: Fused, g: &crate::ply::Gaussians, cams: &[Vec3], path: &[Vec3], mpu: f64) -> Result<(World, Value, Carve)> {
    let Fused { grid, mut solid, floor, .. } = fused;
    let n = grid.n;
    let v = grid.voxel;
    let cell_of = |p: Vec3| (p - grid.lo) / v; // continuous cell coordinates (cell centres at integers)
    let floor_y = floor["y"].as_f64().unwrap_or(0.0) as f32;
    // Crop to the capture region.
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for c in cams { lo = lo.min(*c); hi = hi.max(*c); }
    let xz = (CROP_XZ_M / mpu) as f32;
    let lo = Vec3::new(lo.x - xz, floor_y - (CROP_DOWN_M / mpu) as f32, lo.z - xz);
    let hi = Vec3::new(hi.x + xz, hi.y + (CROP_UP_M / mpu) as f32, hi.z + xz);
    let (clo, chi) = (cell_of(lo).floor(), cell_of(hi).ceil());
    let before = solid.iter().filter(|&&s| s).count();
    for z in 0..n[2] { for y in 0..n[1] { for x in 0..n[0] {
        let p = Vec3::new(x as f32, y as f32, z as f32);
        if p.cmplt(clo).any() || p.cmpgt(chi).any() { solid[grid.at(x, y, z)] = false; }
    }}}
    let cropped = solid.iter().filter(|&&s| s).count();
    // Thin surfaces: at least THICK_M (>= 2 cells) along every axis, grown away from the cameras.
    let t = ((THICK_M / mpu / v as f64).ceil() as usize).max(2);
    let centre = cams.iter().copied().sum::<Vec3>() / cams.len().max(1) as f32;
    thicken(&mut solid, n, t, cell_of(centre));
    let thick = solid.iter().filter(|&&s| s).count();
    let acc = splat_colours(g, &grid);

    // Floor slab layers [slab_lo, fl] (fl: the layer whose top is the floor).
    let fl = ((floor_y - grid.lo.y) / v - 0.5).round().clamp(0.0, n[1] as f32 - 1.0) as usize;
    let slab_cells = ((floor["slab_thickness_units"].as_f64().unwrap_or(0.1) as f32 / v).ceil() as usize).max(1);
    let slab_lo = (fl + 1).saturating_sub(slab_cells);
    // Observed footprint: columns with splat-coloured cells between the floor and above the cameras, or
    // near a camera; dilated by the margin; holes not reachable from the border filled.
    let cols = n[0] * n[2];
    let top_obs = ((cell_of(Vec3::new(0.0, hi.y, 0.0)).y.ceil()).max(0.0) as usize).min(n[1] - 1);
    let mut seen = vec![false; cols];
    for &k in acc.keys() {
        let k = k as usize;
        let (x, y, z) = (k % n[0], (k / n[0]) % n[1], k / (n[0] * n[1]));
        if y + 2 >= fl && y <= top_obs { seen[z * n[0] + x] = true; }
    }
    let disc = (0.8 / mpu / v as f64).ceil() as i64;
    for c in cams {
        let cc = cell_of(*c);
        let (cx, cz) = (cc.x.round() as i64, cc.z.round() as i64);
        for dz in -disc..=disc { for dx in -disc..=disc {
            let (x, z) = (cx + dx, cz + dz);
            if dx * dx + dz * dz <= disc * disc && x >= 0 && z >= 0 && (x as usize) < n[0] && (z as usize) < n[2] {
                seen[z as usize * n[0] + x as usize] = true;
            }
        }}
    }
    let flat = Grid { lo: Vec3::ZERO, voxel: 1.0, n: [n[0], n[2], 1] };
    let margin = ((SLAB_MARGIN_M / mpu / v as f64).ceil() as usize).max(1);
    for axis in 0..2 { dilate_axis(&mut seen, &flat, axis, margin, false); }
    let mut outside = vec![false; cols];
    let mut stack: Vec<usize> = (0..cols).filter(|&i| {
        let (x, z) = (i % n[0], i / n[0]);
        (x == 0 || z == 0 || x + 1 == n[0] || z + 1 == n[2]) && !seen[i]
    }).collect();
    for &i in &stack { outside[i] = true; }
    while let Some(i) = stack.pop() {
        let (x, z) = (i % n[0], i / n[0]);
        let mut visit = |j: usize| if !seen[j] && !outside[j] { outside[j] = true; stack.push(j); };
        if x > 0 { visit(i - 1); } if x + 1 < n[0] { visit(i + 1); }
        if z > 0 { visit(i - n[0]); } if z + 1 < n[2] { visit(i + n[0]); }
    }
    let mut slab_dropped = 0;
    for z in 0..n[2] { for x in 0..n[0] {
        if !outside[z * n[0] + x] { continue; }
        for y in slab_lo..=fl {
            let i = grid.at(x, y, z);
            if solid[i] { solid[i] = false; slab_dropped += 1; }
        }
    }}
    drop((seen, outside));

    // The floor slab at least SHELL_CELLS thick.
    let slab_lo = slab_lo.min((fl + 1).saturating_sub(SHELL_CELLS));
    let mut slab_added = 0;
    for z in 0..n[2] { for x in 0..n[0] {
        if !solid[grid.at(x, fl, z)] { continue; }
        for y in slab_lo..fl {
            let i = grid.at(x, y, z);
            if !solid[i] { solid[i] = true; slab_added += 1; }
        }
    }}

    // Carve the capture path: floaters under and around the cameras (they look out and down, never through
    // the space under themselves) would block the walk where the operator walked. The slab stays.
    let r = (CARVE_R_M / mpu) as f32 + v;
    let mut samples: Vec<Vec3> = path.to_vec();
    let mut segments = 0;
    for w in path.windows(2) {
        let d = w[1] - w[0];
        let len = Vec3::new(d.x, 0.0, d.z).length();
        if len as f64 * mpu > CARVE_MAX_SEG_M { continue; }
        segments += 1;
        let k = (len / (0.5 * v)).ceil() as usize;
        samples.extend((1..k).map(|i| w[0] + d * (i as f32 / k as f32)));
    }
    let head = floor_y + ((EYE_M + CARVE_UP_M) / mpu) as f32;
    let mut carved = 0;
    let mut carve_mask = vec![false; grid.len()];
    for c in &samples {
        let top = (c.y + (CARVE_UP_M / mpu) as f32).max(head);
        let (lo3, hi3) = (cell_of(Vec3::new(c.x - r, 0.0, c.z - r)), cell_of(Vec3::new(c.x + r, top, c.z + r)));
        let yhi = (hi3.y.round().max(0.0) as usize).min(n[1] - 1);
        for z in lo3.z.round().max(0.0) as usize..=(hi3.z.round().max(0.0) as usize).min(n[2] - 1) {
            for x in lo3.x.round().max(0.0) as usize..=(hi3.x.round().max(0.0) as usize).min(n[0] - 1) {
                let p = grid.lo + Vec3::new(x as f32, 0.0, z as f32) * v;
                if (p.x - c.x).powi(2) + (p.z - c.z).powi(2) > r * r { continue; }
                for y in fl + 1..=yhi {
                    let i = grid.at(x, y, z);
                    carve_mask[i] = true;
                    if solid[i] { solid[i] = false; carved += 1; }
                }
            }
        }
    }

    // Colours: own splats, else the 26-neighbourhood's splats; unseen floor from the 2D-nearest observed
    // floor colour; the rest from the nearest coloured solid cell (BFS), else grey.
    let mut rgb: Vec<Option<[u8; 3]>> = vec![None; grid.len()];
    let mut queue: VecDeque<(usize, u32)> = VecDeque::new();
    let rings = ((SPREAD_M / mpu / v as f64).ceil() as u32).max(3);
    let (mut own, mut near, mut spread) = (0usize, 0usize, 0usize);
    for z in 0..n[2] { for y in 0..n[1] { for x in 0..n[0] {
        let i = grid.at(x, y, z);
        if !solid[i] { continue; }
        if let Some(c) = acc.get(&(i as u32)) {
            rgb[i] = Some(to_srgb8(*c));
            own += 1;
        } else {
            let mut sum = [0.0f32; 4];
            for dz in -1i64..=1 { for dy in -1i64..=1 { for dx in -1i64..=1 {
                let (a, b, c) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                if a < 0 || b < 0 || c < 0 || a >= n[0] as i64 || b >= n[1] as i64 || c >= n[2] as i64 { continue; }
                if let Some(s) = acc.get(&(grid.at(a as usize, b as usize, c as usize) as u32)) {
                    for j in 0..4 { sum[j] += s[j]; }
                }
            }}}
            if sum[3] > 0.0 { rgb[i] = Some(to_srgb8(sum)); near += 1; }
        }
        if rgb[i].is_some() { queue.push_back((i, 0)); }
    }}}
    // Floor: 2D BFS over the floor layer from its observed colours, within FLOOR_SPREAD_M; else their median.
    // Seeds: exposed floor-top cells (nothing on them) with their own splat colour.
    let mut fcol: Vec<Option<[u8; 3]>> = (0..cols).map(|c| {
        let (x, z) = (c % n[0], c / n[0]);
        let i = grid.at(x, fl, z);
        let open = fl + 1 >= n[1] || !solid[grid.at(x, fl + 1, z)];
        if open && acc.contains_key(&(i as u32)) { rgb[i] } else { None }
    }).collect();
    let mut observed: [Vec<u8>; 3] = Default::default();
    let mut fq: VecDeque<(usize, u32)> = VecDeque::new();
    for (c, col) in fcol.iter().enumerate() {
        if let Some(col) = col {
            for j in 0..3 { observed[j].push(col[j]); }
            fq.push_back((c, 0));
        }
    }
    let median = if observed[0].is_empty() { GREY } else {
        [0, 1, 2].map(|j| { let o = &mut observed[j]; let k = o.len() / 2; *o.select_nth_unstable(k).1 })
    };
    let frings = ((FLOOR_SPREAD_M / mpu / v as f64).ceil() as u32).max(1);
    while let Some((c, d)) = fq.pop_front() {
        if d >= frings { continue; }
        let col = fcol[c];
        let (x, z) = (c % n[0], c / n[0]);
        let mut visit = |j: usize| if fcol[j].is_none() { fcol[j] = col; fq.push_back((j, d + 1)); };
        if x > 0 { visit(c - 1); } if x + 1 < n[0] { visit(c + 1); }
        if z > 0 { visit(c - n[0]); } if z + 1 < n[2] { visit(c + n[0]); }
    }
    let mut floor_coloured = 0;
    for z in 0..n[2] { for x in 0..n[0] {
        let col = fcol[z * n[0] + x].unwrap_or(median);
        for y in slab_lo..=fl {
            let i = grid.at(x, y, z);
            if solid[i] && rgb[i].is_none() { rgb[i] = Some(col); floor_coloured += 1; queue.push_back((i, 0)); }
        }
    }}
    drop((fcol, acc));
    while let Some((i, d)) = queue.pop_front() {
        if d >= rings { continue; }
        let c = rgb[i];
        let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
        let mut visit = |j: usize| if solid[j] && rgb[j].is_none() { rgb[j] = c; spread += 1; queue.push_back((j, d + 1)); };
        if x > 0 { visit(i - 1); } if x + 1 < n[0] { visit(i + 1); }
        if y > 0 { visit(i - n[0]); } if y + 1 < n[1] { visit(i + n[0]); }
        if z > 0 { visit(i - n[0] * n[1]); } if z + 1 < n[2] { visit(i + n[0] * n[1]); }
    }

    // Light smoothing of surface cells toward their similar 6-neighbours (edges stay crisp).
    let is_surface = |i: usize| {
        let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
        x == 0 || y == 0 || z == 0 || x + 1 == n[0] || y + 1 == n[1] || z + 1 == n[2]
            || [i - 1, i + 1, i - n[0], i + n[0], i - n[0] * n[1], i + n[0] * n[1]].iter().any(|&j| !solid[j])
    };
    let snapshot = rgb.clone();
    let mut smoothed = 0;
    for i in 0..grid.len() {
        let Some(c) = snapshot[i] else { continue };
        if !solid[i] || !is_surface(i) { continue; }
        let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
        let mut sum = [0.0f32; 3];
        let mut k = 0.0;
        let mut add = |j: usize| if let Some(o) = snapshot[j] {
            let d2: f32 = (0..3).map(|q| (o[q] as f32 - c[q] as f32).powi(2)).sum();
            if solid[j] && d2 < SMOOTH_MAX_DIST * SMOOTH_MAX_DIST { for q in 0..3 { sum[q] += o[q] as f32; } k += 1.0; }
        };
        if x > 0 { add(i - 1); } if x + 1 < n[0] { add(i + 1); }
        if y > 0 { add(i - n[0]); } if y + 1 < n[1] { add(i + n[0]); }
        if z > 0 { add(i - n[0] * n[1]); } if z + 1 < n[2] { add(i + n[0] * n[1]); }
        if k > 0.0 {
            rgb[i] = Some([0, 1, 2].map(|q| ((1.0 - SMOOTH) * c[q] as f32 + SMOOTH * sum[q] / k).round().clamp(0.0, 255.0) as u8));
            smoothed += 1;
        }
    }
    drop(snapshot);

    // Outer shell: from each surface cell, along the ray from its nearest camera, past its own solid run; when
    // nothing solid lies further out (a wall or ceiling with the void behind), add SHELL_CELLS cells there in
    // its colour.
    let mut shell: Vec<(usize, [u8; 3])> = Vec::new();
    for i in 0..grid.len() {
        if !solid[i] || !is_surface(i) { continue; }
        let Some(col) = rgb[i] else { continue };
        let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
        let p = grid.lo + Vec3::new(x as f32, y as f32, z as f32) * v;
        let Some(cam) = cams.iter().min_by(|a, b| a.distance_squared(p).total_cmp(&b.distance_squared(p))) else { break };
        let dir = (p - *cam).normalize_or_zero();
        if dir == Vec3::ZERO { continue; }
        let (mut t, mut left_run, mut behind) = (0.5 * v, false, Vec::new());
        let mut outer = true;
        loop {
            let Some(j) = grid.index(p + dir * t) else { break };
            let j = j as usize;
            if solid[j] {
                if left_run { outer = false; break; }
            } else {
                left_run = true;
                if behind.len() < SHELL_CELLS && behind.last() != Some(&j) { behind.push(j); }
            }
            t += 0.5 * v;
        }
        if outer { shell.extend(behind.into_iter().map(|j| (j, col))); }
    }
    let mut shell_added = 0;
    for (j, col) in shell {
        if !solid[j] { solid[j] = true; rgb[j] = Some(col); shell_added += 1; }
    }
    let clean = cleanup(&mut solid, &mut rgb, n, v, fl, &carve_mask, cell_of(centre), mpu);

    // Tight bounds of the remaining cells.
    let (mut bmin, mut bmax) = ([usize::MAX; 3], [0usize; 3]);
    for z in 0..n[2] { for y in 0..n[1] { for x in 0..n[0] {
        if solid[grid.at(x, y, z)] {
            for (a, c) in [x, y, z].into_iter().enumerate() { bmin[a] = bmin[a].min(c); bmax[a] = bmax[a].max(c); }
        }
    }}}
    ensure!(bmin[0] != usize::MAX, "voxel world is empty");
    let dims = [0, 1, 2].map(|a| bmax[a] - bmin[a] + 1);
    let origin = grid.lo + (Vec3::new(bmin[0] as f32, bmin[1] as f32, bmin[2] as f32) - 0.5) * v;
    let mut w = World { origin, edge: v, dims, solid: vec![false; dims[0] * dims[1] * dims[2]], rgb: vec![GREY; dims[0] * dims[1] * dims[2]],
        seen: vec![true; dims[0] * dims[1] * dims[2]] };
    let mut grey = 0;
    for z in 0..dims[2] { for y in 0..dims[1] { for x in 0..dims[0] {
        let i = grid.at(x + bmin[0], y + bmin[1], z + bmin[2]);
        if !solid[i] { continue; }
        let k = w.at(x, y, z);
        w.solid[k] = true;
        match rgb[i] { Some(c) => w.rgb[k] = c, None => grey += 1 }
    }}}
    let stats = json!({"fusion_voxel_units": v, "fusion_dims": n, "fused_with_floor": before, "after_crop": cropped,
        "after_thicken": thick, "min_thickness_cells": t, "crop": {"min": lo.to_array(), "max": hi.to_array()},
        "slab": {"layers": [slab_lo, fl], "dropped_unobserved": slab_dropped, "margin_cells": margin},
        "cleanup": clean,
        "shell": {"cells": shell_added, "slab_cells_added": slab_added, "smoothed_surface_cells": smoothed},
        "carve": {"cells": carved, "cameras": path.len(), "segments": segments, "radius_units": r},
        "colour": {"own_splats": own, "neighbour_splats": near, "floor_2d": floor_coloured, "floor_median": median, "propagated": spread, "propagate_rings": rings, "grey": grey},
        "collision_floor": floor});
    Ok((w, stats, Carve { grid: Grid { lo: grid.lo, voxel: grid.voxel, n: grid.n }, mask: carve_mask }))
}

/// Readability cleanup of the solid grid (cells above the floor layer `fl`, outside the carved path):
/// 1. specks (<= 1 solid face-neighbour) removed, single-cell pits (>= 5) filled;
/// 2. MAJORITY_PASSES of a 3x3x3 majority filter (>= 14 of 27 solid: solid, <= 8: empty);
/// 3. big planes snapped flat: the ceiling (strongest downward-facing layer above the cameras) and up to
///    MAX_WALLS axis-aligned walls (the export frame aligns walls with the grid; peaks of room-facing cells). Within each plane's observed
///    footprint, the cells up to SNAP_BAND cells on the room side are cleared when they are a thin bump
///    (empty 3 cells further in, so furniture against the wall stays) and SNAP_THICK cells behind are filled.
/// New cells take the colour of the nearest coloured cell.
fn cleanup(solid: &mut [bool], rgb: &mut [Option<[u8; 3]>], n: [usize; 3], v: f32, fl: usize, carved: &[bool], centre: Vec3,
    mpu: f64) -> Value {
    const MAJORITY_PASSES: usize = 2;
    const SNAP_BAND: f32 = 2.0;
    const SNAP_THICK: f32 = 3.0;
    const MAX_WALLS: usize = 8;
    let len = n[0] * n[1] * n[2];
    let at = |x: usize, y: usize, z: usize| (z * n[1] + y) * n[0] + x;
    let xyz = |i: usize| (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
    let editable = |i: usize, carved: &[bool]| xyz(i).1 > fl && !carved[i];
    let faces = |solid: &[bool], i: usize| -> usize {
        let (x, y, z) = xyz(i);
        [(x > 0).then(|| i - 1), (x + 1 < n[0]).then(|| i + 1), (y > 0).then(|| i - n[0]), (y + 1 < n[1]).then(|| i + n[0]),
            (z > 0).then(|| i - n[0] * n[1]), (z + 1 < n[2]).then(|| i + n[0] * n[1])].into_iter().flatten().filter(|&j| solid[j]).count()
    };
    let (mut removed, mut filled) = (0usize, 0usize);
    // 1. Specks and pits.
    let snap: Vec<bool> = solid.to_vec();
    for i in 0..len {
        if !editable(i, carved) { continue; }
        let f = faces(&snap, i);
        if snap[i] && f <= 1 { solid[i] = false; removed += 1; }
        else if !snap[i] && f >= 5 { solid[i] = true; filled += 1; }
    }
    // 2. Majority filter: 3x3x3 box counts via separable running sums.
    let mut maj_removed = 0usize;
    let mut maj_filled = 0usize;
    for _ in 0..MAJORITY_PASSES {
        let mut cnt: Vec<u8> = solid.iter().map(|&b| b as u8).collect();
        for (axis, stride) in [(0, 1), (1, n[0]), (2, n[0] * n[1])] {
            let src = cnt.clone();
            for i in 0..len {
                let c = [xyz(i).0, xyz(i).1, xyz(i).2][axis];
                let mut s = src[i];
                if c > 0 { s += src[i - stride]; }
                if c + 1 < n[axis] { s += src[i + stride]; }
                cnt[i] = s;
            }
        }
        for i in 0..len {
            if !editable(i, carved) { continue; }
            if solid[i] && cnt[i] <= 8 { solid[i] = false; maj_removed += 1; }
            else if !solid[i] && cnt[i] >= 14 { solid[i] = true; maj_filled += 1; }
        }
    }
    // 3. Planes. Surface cells with a horizontal or downward open face.
    let mut snapped_cleared = 0usize;
    let mut snapped_filled = 0usize;
    let mut planes = Vec::new();
    let cam_layer = centre.y.max(fl as f32 + 1.0);
    // Ceiling: the strongest layer of downward-facing surface cells above the cameras.
    let mut down = vec![0usize; n[1]];
    for i in 0..len {
        let (_, y, _) = xyz(i);
        if solid[i] && y > 0 && (y as f32) > cam_layer && !solid[i - n[0]] { down[y] += 1; }
    }
    let mut snap_plane = |solid: &mut [bool], nrm: Vec3, d: f32, foot: &dyn Fn(usize) -> bool| {
        // Signed distance (cells) of cell centre along nrm (pointing out of the room) from the plane.
        for i in 0..len {
            if !foot(i) || !editable(i, carved) { continue; }
            let (x, y, z) = xyz(i);
            let p = Vec3::new(x as f32, y as f32, z as f32);
            let s = nrm.dot(p) - d;
            if s < -SNAP_BAND - 0.5 || s >= SNAP_THICK - 0.5 { continue; }
            if s < -0.5 {
                if !solid[i] { continue; }
                // A thin bump only: nothing solid 3 cells further into the room.
                let q = p - nrm * 3.0;
                let inside = q.cmpge(Vec3::ZERO).all() && (q.x as usize) < n[0] && (q.y as usize) < n[1] && (q.z as usize) < n[2]
                    && solid[at(q.x.round() as usize, q.y.round() as usize, q.z.round() as usize)];
                if !inside { solid[i] = false; snapped_cleared += 1; }
            } else if !solid[i] {
                solid[i] = true;
                snapped_filled += 1;
            }
        }
    };
    if let Some((cy, &support)) = down.iter().enumerate().max_by_key(|(_, c)| **c) {
        let footprint = n[0] * n[2];
        if support as f64 >= 0.05 * footprint as f64 {
            let mut foot = vec![false; footprint];
            for i in 0..len {
                let (x, y, z) = xyz(i);
                if solid[i] && (y as isize - cy as isize).abs() <= 1 { foot[z * n[0] + x] = true; }
            }
            let f = |i: usize| { let (x, _, z) = xyz(i); foot[z * n[0] + x] };
            // Ceiling normal points up (out of the room); the plane is the ceiling's underside.
            snap_plane(solid, Vec3::Y, cy as f32, &f);
            planes.push(json!({"kind": "ceiling", "layer": cy, "support": support}));
        }
    }
    // Walls: RANSAC on horizontally facing surface cells between the floor and the ceiling.
    let mut pts: Vec<Vec3> = Vec::new();
    for i in 0..len {
        if !solid[i] { continue; }
        let (x, y, z) = xyz(i);
        if y <= fl + 2 { continue; }
        let open = (x > 0 && !solid[i - 1]) || (x + 1 < n[0] && !solid[i + 1]) || (z > 0 && !solid[i - n[0] * n[1]])
            || (z + 1 < n[2] && !solid[i + n[0] * n[1]]);
        if open { pts.push(Vec3::new(x as f32, y as f32, z as f32)); }
    }
    let min_support = ((1.0 / mpu) as f32 / v).powi(2) as usize; // 1 m^2 of wall facing the room
    const TOL: f32 = 1.5;
    // The export frame aligns the room's walls with the grid axes: per axis direction, the slices holding the
    // most room-facing cells (counted over +-1 slice) are walls.
    let cell_solid = |solid: &[bool], q: Vec3| -> bool {
        let r = q.round();
        r.cmpge(Vec3::ZERO).all() && (r.x as usize) < n[0] && (r.y as usize) < n[1] && (r.z as usize) < n[2]
            && solid[at(r.x as usize, r.y as usize, r.z as usize)]
    };
    let faces_room = |solid: &[bool], p: Vec3, nrm: Vec3| !cell_solid(solid, p - nrm) && cell_solid(solid, p + nrm);
    let mut walls: Vec<(Vec3, f32, usize)> = Vec::new();
    for nrm in [Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z] {
        // Only the side of the room this normal points out of.
        let mut hist: HashMap<i32, usize> = HashMap::new();
        for p in &pts {
            if nrm.dot(*p - centre) > 0.0 && faces_room(solid, *p, nrm) { *hist.entry(nrm.dot(*p).round() as i32).or_default() += 1; }
        }
        let band = |c: i32| (c - 1..=c + 1).map(|k| hist.get(&k).copied().unwrap_or(0)).sum::<usize>();
        let mut peaks: Vec<(i32, usize)> = hist.keys().map(|&c| (c, band(c))).collect();
        peaks.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut taken: Vec<i32> = Vec::new();
        for (c, k) in peaks {
            if k < min_support || taken.len() >= MAX_WALLS / 4 { break; }
            if taken.iter().any(|t| (t - c).abs() <= 3) { continue; }
            taken.push(c);
            walls.push((nrm, c as f32, k));
        }
    }
    for (nrm, d, k) in walls {
        let inl: Vec<Vec3> = pts.iter().copied().filter(|p| (nrm.dot(*p) - d).abs() <= TOL && faces_room(solid, *p, nrm)).collect();
        let along = Vec3::new(-nrm.z, 0.0, nrm.x);
        let mut foot: HashMap<(i32, i32), ()> = HashMap::new();
        for p in &inl {
            let (u, y) = (along.dot(*p).round() as i32, p.y as i32);
            for du in -2..=2 { for dy in -2..=2 { foot.insert((u + du, y + dy), ()); } }
        }
        let f = |i: usize| { let (x, y, z) = xyz(i);
            foot.contains_key(&(along.dot(Vec3::new(x as f32, 0.0, z as f32)).round() as i32, y as i32)) };
        // The plane sits at the room-side face of the wall cells.
        snap_plane(solid, nrm, d, &f);
        planes.push(json!({"kind": "wall", "normal": [nrm.x, nrm.z], "slice": d, "support": k}));
    }
    // Colours for new cells: nearest coloured cell (6-connected BFS), else grey.
    let mut q: VecDeque<usize> = (0..len).filter(|&i| solid[i] && rgb[i].is_some()).collect();
    let mut recoloured = 0usize;
    while let Some(i) = q.pop_front() {
        let c = rgb[i];
        let (x, y, z) = xyz(i);
        for j in [(x > 0).then(|| i - 1), (x + 1 < n[0]).then(|| i + 1), (y > 0).then(|| i - n[0]), (y + 1 < n[1]).then(|| i + n[0]),
            (z > 0).then(|| i - n[0] * n[1]), (z + 1 < n[2]).then(|| i + n[0] * n[1])].into_iter().flatten() {
            if solid[j] && rgb[j].is_none() { rgb[j] = c; recoloured += 1; q.push_back(j); }
        }
    }
    json!({"specks_removed": removed, "pits_filled": filled, "majority_removed": maj_removed, "majority_filled": maj_filled,
        "majority_passes": MAJORITY_PASSES, "snap_cleared": snapped_cleared, "snap_filled": snapped_filled, "planes": planes,
        "recoloured": recoloured})
}

/// Capture frames are sampled at this height (px) for colours.
const IMG_H: usize = 480;
/// At most this many capture frames colour the voxels (evenly spaced in capture order).
const MAX_FRAMES: usize = 200;
/// A frame sees a cell when the cell is within one cell of the frame's voxel z-buffer and the frame's rendered
/// splat depth is not more than SPLAT_OCCLUDER_M nearer (an occluder the grid misses).
const SPLAT_OCCLUDER_M: f64 = 0.30;

/// The folder holding the capture frame `name`: `<data>/images`, `<data>`, or a sibling artifact's `images`
/// (a run keeps frames in artifacts/frames_*/images next to artifacts/sfm_*).
fn image_dir(data: &Path, name: &str) -> Option<PathBuf> {
    let mut cands = vec![data.join("images"), data.to_path_buf()];
    if let Some(parent) = data.parent() {
        if let Ok(rd) = fs::read_dir(parent) {
            let mut sib: Vec<PathBuf> = rd.flatten().map(|e| e.path().join("images")).collect();
            sib.sort();
            cands.extend(sib);
        }
    }
    cands.into_iter().find(|d| d.join(name).is_file())
}

/// Colour the world's visible surface cells from the capture frames: each cell centre is projected into up to
/// MAX_FRAMES frames (pinhole, principal point at the centre, as the depth fusion), kept where it faces the camera,
/// is within 1.5 cells of the frame's z-buffer of the visible shell (cells facing the room's free space) itself, and the frame's splat depth is not
/// more than SPLAT_OCCLUDER_M nearer; the pixel footprint (~ the
/// cell's projected size) is averaged, and the frames' colours are combined luminance-trimmed (20%). Cells no
/// frame sees keep their splat colour. Interior cells then take the nearest surface cell's colour, and surface
/// colours get one edge-preserving smoothing pass.
fn colour_from_images(w: &mut World, scene: &Value, g: &crate::ply::Gaussians, level_m: Option<&DMat4>, ply_hint: Option<&Path>, mpu: f64)
    -> Result<Value> {
    let t0 = Instant::now();
    let data = PathBuf::from(scene["source"]["data"].as_str().context("scene.json: no source.data")?);
    let mut views = collision::scene_views(scene, None, ply_hint, IMG_H)?;
    if let Some(m) = level_m { for v in &mut views { v.transform(m); } }
    let first = views.first().context("no capture cameras")?.name.clone();
    let dir = image_dir(&data, &first).with_context(|| format!("capture frame {first} not found near {}", data.display()))?;
    let step = views.len().div_ceil(MAX_FRAMES).max(1);
    let views: Vec<collision::View> = views.into_iter().step_by(step).collect();
    let prims = collision::prims(g);
    let frames: Vec<Option<(Vec<[u8; 3]>, Vec<f32>)>> = views.par_iter().map(|v| {
        let img = image::open(dir.join(&v.name)).ok()?.to_rgb8();
        let img = image::imageops::resize(&img, v.w as u32, v.h as u32, image::imageops::FilterType::Triangle);
        Some((img.pixels().map(|p| p.0).collect(), collision::render_depth(&prims, v)))
    }).collect();
    drop(prims);
    let loaded = frames.iter().filter(|f| f.is_some()).count();
    let t_frames = t0.elapsed().as_secs_f64();
    let n = w.dims;
    let len = n[0] * n[1] * n[2];
    let xyz = |i: usize| (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
    let nbrs = |i: usize| -> [Option<usize>; 6] {
        let (x, y, z) = xyz(i);
        [(x > 0).then(|| i - 1), (x + 1 < n[0]).then(|| i + 1), (y > 0).then(|| i - n[0]), (y + 1 < n[1]).then(|| i + n[0]),
            (z > 0).then(|| i - n[0] * n[1]), (z + 1 < n[2]).then(|| i + n[0] * n[1])]
    };
    const DIRS: [Vec3; 6] = [Vec3::NEG_X, Vec3::X, Vec3::NEG_Y, Vec3::Y, Vec3::NEG_Z, Vec3::Z];
    // The room's free space: empty cells reachable from the capture cameras.
    let mut room = vec![false; len];
    let mut q: VecDeque<usize> = VecDeque::new();
    for v in &views {
        let c = ((v.c - w.origin) / w.edge).floor();
        if c.cmpge(Vec3::ZERO).all() && (c.x as usize) < n[0] && (c.y as usize) < n[1] && (c.z as usize) < n[2] {
            let i = (c.z as usize * n[1] + c.y as usize) * n[0] + c.x as usize;
            if !w.solid[i] && !room[i] { room[i] = true; q.push_back(i); }
        }
    }
    while let Some(i) = q.pop_front() {
        for j in nbrs(i).into_iter().flatten() {
            if !w.solid[j] && !room[j] { room[j] = true; q.push_back(j); }
        }
    }
    // The visible shell: solid cells with a face on the room's free space, and their normals (toward it).
    let surface: Vec<(usize, Vec3)> = (0..len).into_par_iter().filter_map(|i| {
        if !w.solid[i] { return None; }
        let nb = nbrs(i);
        let nrm: Vec3 = (0..6).filter(|&k| nb[k].is_some_and(|j| room[j])).map(|k| DIRS[k]).sum();
        nb.iter().any(|j| j.is_some_and(|j| room[j])).then(|| (i, nrm.normalize_or_zero()))
    }).collect();
    drop(room);
    let rts: Vec<glam::Mat3> = views.iter().map(|v| v.r.transpose()).collect();
    let e = w.edge;
    // Per frame, the voxel world's own z-buffer: every surface cell facing the camera splatted as a square of its
    // projected size, nearest depth kept.
    let zbufs: Vec<Vec<f32>> = views.par_iter().enumerate().map(|(k, v)| {
        let mut zb = vec![f32::INFINITY; v.w * v.h];
        if frames[k].is_none() { return zb; }
        for &(i, nrm) in &surface {
            let (x, y, z) = xyz(i);
            let p = w.origin + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * e;
            if nrm != Vec3::ZERO && nrm.dot(v.c - p) <= 0.0 { continue; }
            let q = rts[k] * (p - v.c);
            if q.z < 0.1 { continue; }
            let (u, vv) = (v.f * q.x / q.z + v.w as f32 * 0.5, v.f * q.y / q.z + v.h as f32 * 0.5);
            let r = 0.75 * v.f * e / q.z;
            let (x0, x1) = ((u - r).floor().max(0.0) as i64, (u + r).ceil().min(v.w as f32 - 1.0) as i64);
            let (y0, y1) = ((vv - r).floor().max(0.0) as i64, (vv + r).ceil().min(v.h as f32 - 1.0) as i64);
            for py in y0..=y1 { for px in x0..=x1 {
                let o = py as usize * v.w + px as usize;
                if q.z < zb[o] { zb[o] = q.z; }
            }}
        }
        zb
    }).collect();
    let (ztol, stol) = (1.5 * e, (SPLAT_OCCLUDER_M / mpu) as f32);
    // Rejections per reason (for build.json): [behind the voxel z-buffer, behind a nearer splat surface, no depth].
    let rejects: [std::sync::atomic::AtomicUsize; 3] = Default::default();
    // Surface cells some frame sees (front of its voxel z-buffer), splat-occluded or not.
    let in_view = std::sync::atomic::AtomicUsize::new(0);
    let samples: Vec<(usize, Option<[u8; 3]>, u16)> = surface.par_iter().map(|&(i, nrm)| {
        let mut rej = [0usize; 3];
        let mut looked = false;
        let (x, y, z) = xyz(i);
        let p = w.origin + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * e;
        let mut cols: Vec<[f32; 3]> = Vec::new();
        for (k, v) in views.iter().enumerate() {
            let Some((rgb, depth)) = &frames[k] else { continue };
            if nrm != Vec3::ZERO && nrm.dot(v.c - p) <= 0.0 { continue; }
            let q = rts[k] * (p - v.c);
            if q.z < 0.1 { continue; }
            let (u, vv) = (v.f * q.x / q.z + v.w as f32 * 0.5, v.f * q.y / q.z + v.h as f32 * 0.5);
            if !(u >= 0.0 && vv >= 0.0 && u < v.w as f32 && vv < v.h as f32) { continue; }
            let o = vv as usize * v.w + u as usize;
            if q.z > zbufs[k][o] + ztol { rej[0] += 1; continue; }
            looked = true;
            // An occluder the voxel grid misses: the frame's splat surface well in front of the cell.
            let d = depth[o];
            if d.is_finite() && d < q.z - stol { rej[1] += 1; continue; }
            let r = (0.5 * v.f * e / q.z).clamp(0.0, 3.0) as i64;
            let (mut acc, mut m) = ([0.0f32; 3], 0.0);
            for dy in -r..=r { for dx in -r..=r {
                let (px, py) = (u as i64 + dx, vv as i64 + dy);
                if px < 0 || py < 0 || px >= v.w as i64 || py >= v.h as i64 { continue; }
                let c = rgb[py as usize * v.w + px as usize];
                for j in 0..3 { acc[j] += c[j] as f32; }
                m += 1.0;
            }}
            if m > 0.0 { cols.push(acc.map(|a| a / m)); }
        }
        for k in 0..3 { rejects[k].fetch_add(rej[k], std::sync::atomic::Ordering::Relaxed); }
        if looked { in_view.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        let used = cols.len() as u16;
        if cols.is_empty() { return (i, None, 0); }
        let luma = |c: &[f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        cols.sort_by(|a, b| luma(a).total_cmp(&luma(b)));
        let cut = if cols.len() >= 5 { cols.len() / 5 } else { 0 };
        let kept = &cols[cut..cols.len() - cut];
        let mut acc = [0.0f32; 3];
        for c in kept { for j in 0..3 { acc[j] += c[j]; } }
        (i, Some(acc.map(|a| (a / kept.len() as f32).round().clamp(0.0, 255.0) as u8)), used)
    }).collect();
    let mut coloured = 0usize;
    let mut per: Vec<u16> = Vec::with_capacity(samples.len());
    let mut is_surface = vec![false; len];
    for &(i, c, used) in &samples {
        is_surface[i] = true;
        w.seen[i] = c.is_some();
        if let Some(c) = c { w.rgb[i] = c; coloured += 1; per.push(used); }
    }
    // Interior cells: the nearest surface cell's colour.
    let mut seen = is_surface.clone();
    let mut q: VecDeque<usize> = samples.iter().map(|s| s.0).collect();
    while let Some(i) = q.pop_front() {
        for j in nbrs(i).into_iter().flatten() {
            if w.solid[j] && !seen[j] { seen[j] = true; w.rgb[j] = w.rgb[i]; q.push_back(j); }
        }
    }
    // One edge-preserving smoothing pass over the surface (30% toward similar face-neighbours on the surface).
    let snap = w.rgb.clone();
    for &(i, _, _) in &samples {
        let c = snap[i];
        let (mut acc, mut m) = ([0.0f32; 3], 0.0);
        for j in nbrs(i).into_iter().flatten() {
            if !is_surface[j] { continue; }
            let o = snap[j];
            let d2: f32 = (0..3).map(|k| (o[k] as f32 - c[k] as f32).powi(2)).sum();
            if d2 < 30.0 * 30.0 { for k in 0..3 { acc[k] += o[k] as f32; } m += 1.0; }
        }
        if m > 0.0 { w.rgb[i] = [0, 1, 2].map(|k| (0.7 * c[k] as f32 + 0.3 * acc[k] / m).round() as u8); }
    }
    per.sort_unstable();
    let mean = if per.is_empty() { 0.0 } else { per.iter().map(|&p| p as f64).sum::<f64>() / per.len() as f64 };
    Ok(json!({"frames_dir": dir, "frames_used": loaded, "frames_available": views.len() * step, "frame_height_px": IMG_H,
        "surface_cells": samples.len(), "frame_visible_cells": in_view.load(std::sync::atomic::Ordering::Relaxed),
        "coloured_from_images": coloured,
        "coloured_pct_of_frame_visible": (1000.0 * coloured as f64 / in_view.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64).round() / 10.0,
        "coloured_pct": (1000.0 * coloured as f64 / samples.len().max(1) as f64).round() / 10.0,
        "frames_per_voxel": {"mean": (mean * 10.0).round() / 10.0, "median": per.get(per.len() / 2), "p10": per.get(per.len() / 10)},
        "zbuffer_tolerance_units": ztol, "splat_occluder_units": stol,
        "rejected": {"voxel_occluded": rejects[0].load(std::sync::atomic::Ordering::Relaxed),
            "splat_occluded": rejects[1].load(std::sync::atomic::Ordering::Relaxed)},
        "frames_s": t_frames, "total_s": t0.elapsed().as_secs_f64()}))
}

/// Flatten the room's walls and ceiling as seen from the room centre (the floor layer's centroid). For each
/// horizontal axis direction (and up, for the ceiling), the farthest empty -> solid transition outward (the wall,
/// behind any furniture) forms a depth image over the other two axes (walls from 0.3 m above the floor); RANSAC fits a plane depth = a*u + b*v + c with
/// a WALL_INLIER-cell band, up to 2 per direction, kept at >= WALL_MIN_M2 / CEILING_MIN_M2. Planes within
/// WALL_AXIS_DEG of the axis are snapped to one grid layer (the inliers' median). Along each column within the bulge
/// band of the plane (WALL_GROW_DIST cells, at least WALL_BULGE_M): a bulge of the wall's own surface toward the room
/// is cleared down to the plane (up to WALL_CLEAR_M; never earlier runs of solid, i.e. furniture), and empty cells of
/// the plane plus 3 behind are filled with the wall's colour (the ceiling is only filled).
fn flatten_walls(w: &mut World, mpu: f64, floor_y: f64) -> Value {
    let n = w.dims;
    let e = w.edge as f64;
    let at = |x: i64, y: i64, z: i64| -> Option<usize> {
        (x >= 0 && y >= 0 && z >= 0 && (x as usize) < n[0] && (y as usize) < n[1] && (z as usize) < n[2])
            .then(|| (z as usize * n[1] + y as usize) * n[0] + x as usize)
    };
    let fl = ((floor_y - w.origin.y as f64) / e).floor() as i64;
    let (mut sx, mut sz, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
    if fl >= 0 && (fl as usize) < n[1] {
        for z in 0..n[2] { for x in 0..n[0] { if w.solid[(z * n[1] + fl as usize) * n[0] + x] { sx += x as f64; sz += z as f64; cnt += 1.0; } } }
    }
    if cnt == 0.0 { return json!({"summary": {"walls": 0}, "planes": []}); }
    let centre = [(sx / cnt).round() as i64, (sz / cnt).round() as i64];
    let v_from = fl + 1 + (0.3 / mpu / e).ceil() as i64;
    let ceil_from = fl + (1.8 / mpu / e).ceil() as i64;
    let mut rng = 0xD1B54A32D192ED03u64;
    let mut planes = Vec::new();
    let (mut cleared, mut filled) = (0usize, 0usize);
    // (march axis, sign, label): cell = f(u, v, t) with t the march coordinate.
    for (axis, sign, label) in [(0usize, 1i64, "+x"), (0, -1, "-x"), (2, 1, "+z"), (2, -1, "-z"), (1, 1, "ceiling")] {
        let cell = |u: i64, v: i64, t: i64| -> Option<usize> {
            match axis { 0 => at(t, v, u), 2 => at(u, v, t), _ => at(u, t, v) }
        };
        let (nu, nv) = match axis { 0 => (n[2], n[1]), 2 => (n[0], n[1]), _ => (n[0], n[2]) };
        let (t0, tn) = match axis { 0 => (centre[0], n[0] as i64), 2 => (centre[1], n[2] as i64), _ => (ceil_from, n[1] as i64) };
        // Depth image of the WALL: per column, the farthest empty -> solid transition outward from the centre
        // (furniture in front gives earlier transitions), with that cell's colour.
        let mut depth: Vec<Option<(i64, [u8; 3])>> = vec![None; nu * nv];
        for v in 0..nv as i64 {
            if axis != 1 && v < v_from { continue; }
            for u in 0..nu as i64 {
                let mut t = t0;
                let mut prev_empty = false;
                while t >= 0 && t < tn {
                    if let Some(i) = cell(u, v, t) {
                        if w.solid[i] && prev_empty { depth[v as usize * nu + u as usize] = Some((t, w.rgb[i])); }
                        prev_empty = !w.solid[i];
                    }
                    t += sign;
                }
            }
        }
        let mut free: Vec<usize> = (0..depth.len()).filter(|&k| depth[k].is_some()).collect();
        for _round in 0..2 {
            if free.len() < 3 { break; }
            let pt = |k: usize| (k % nu, k / nu, depth[k].unwrap().0);
            let mut best: Option<([f64; 3], usize)> = None;
            for _ in 0..300 {
                let mut pick = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; pt(free[(rng % free.len() as u64) as usize]) };
                let (p1, p2, p3) = (pick(), pick(), pick());
                // Solve t = a u + b v + c.
                let (u1, v1, t1) = (p1.0 as f64, p1.1 as f64, p1.2 as f64);
                let (du2, dv2, dt2) = (p2.0 as f64 - u1, p2.1 as f64 - v1, p2.2 as f64 - t1);
                let (du3, dv3, dt3) = (p3.0 as f64 - u1, p3.1 as f64 - v1, p3.2 as f64 - t1);
                let det = du2 * dv3 - du3 * dv2;
                if det.abs() < 1e-6 { continue; }
                let a = (dt2 * dv3 - dt3 * dv2) / det;
                let b = (du2 * dt3 - du3 * dt2) / det;
                if a.hypot(b) > 1.2 { continue; } // > ~50 deg from facing the axis (bay-window walls stay in)
                let c = t1 - a * u1 - b * v1;
                let k = free.iter().filter(|&&k| { let q = pt(k); (q.2 as f64 - (a * q.0 as f64 + b * q.1 as f64 + c)).abs() <= WALL_INLIER as f64 }).count();
                if best.is_none_or(|bb| k > bb.1) { best = Some(([a, b, c], k)); }
            }
            let Some(([a, b, c], k)) = best else { break };
            let min_m2 = if axis == 1 { CEILING_MIN_M2 } else { WALL_MIN_M2 };
            if (k as f64) * e * e * mpu * mpu < min_m2 { break; }
            let inl: Vec<usize> = free.iter().copied().filter(|&k| { let q = pt(k); (q.2 as f64 - (a * q.0 as f64 + b * q.1 as f64 + c)).abs() <= WALL_INLIER as f64 }).collect();
            // Least squares on the inliers.
            let (mut s_uu, mut s_uv, mut s_vv, mut s_u, mut s_v, mut s_ut, mut s_vt, mut s_t) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for &kk in &inl { let q = pt(kk); let (u, v, t) = (q.0 as f64, q.1 as f64, q.2 as f64);
                s_uu += u * u; s_uv += u * v; s_vv += v * v; s_u += u; s_v += v; s_ut += u * t; s_vt += v * t; s_t += t; }
            let nn = inl.len() as f64;
            let m = DMat3::from_cols(DVec3::new(s_uu, s_uv, s_u), DVec3::new(s_uv, s_vv, s_v), DVec3::new(s_u, s_v, nn));
            let (mut a, mut b, mut c) = if m.determinant().abs() > 1e-9 { let r = m.inverse() * DVec3::new(s_ut, s_vt, s_t); (r.x, r.y, r.z) } else { (a, b, c) };
            let tilt = a.hypot(b).atan().to_degrees();
            let snapped = tilt <= WALL_AXIS_DEG as f64;
            if snapped {
                let mut ts: Vec<i64> = inl.iter().map(|&kk| pt(kk).2).collect();
                ts.sort_unstable();
                (a, b, c) = (0.0, 0.0, ts[ts.len() / 2] as f64);
            }
            // Columns whose wall surface lies within the fit band of the plane (and their neighbours) are flattened.
            let band = (WALL_GROW_DIST as i64).max((WALL_BULGE_M / mpu / e).ceil() as i64);
            let clear_band = (WALL_CLEAR_M / mpu / e).ceil() as i64;
            let mut cols: Vec<usize> = free.iter().copied().filter(|&kk| { let q = pt(kk); (q.2 as f64 - (a * q.0 as f64 + b * q.1 as f64 + c)).abs() <= band as f64 }).collect();
            let set: HashMap<usize, ()> = cols.iter().map(|&k| (k, ())).collect();
            let mut extra = Vec::new();
            for &kk in &cols { let (u, v) = ((kk % nu) as i64, (kk / nu) as i64);
                for (du, dv) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (uu, vv) = (u + du, v + dv);
                    if uu >= 0 && vv >= 0 && (uu as usize) < nu && (vv as usize) < nv {
                        let k2 = vv as usize * nu + uu as usize;
                        if !set.contains_key(&k2) && depth[k2].is_some_and(|(t, _)| (t as f64 - (a * uu as f64 + b * vv as f64 + c)).abs() <= band as f64 + 2.0) { extra.push(k2); }
                    }
                } }
            extra.sort_unstable(); extra.dedup();
            cols.extend(extra);
            let mut dev_max = 0i64;
            for &kk in &cols {
                let (u, v) = ((kk % nu) as i64, (kk / nu) as i64);
                let Some((hit, col)) = depth[kk] else { continue };
                let tp = (a * u as f64 + b * v as f64 + c).round() as i64;
                dev_max = dev_max.max((hit - tp).abs());
                // A bulge toward the room (wall surface in front of the plane): clear only the wall's own run, from its
                // surface to the plane, and only up to WALL_CLEAR_M; earlier runs (furniture) are never touched. The
                // ceiling is only filled (clearing under it notched the wall tops).
                let bulge = (tp - hit) * sign; // > 0: the wall surface is room-side of the plane
                if axis != 1 && bulge > 0 && bulge <= clear_band && cell(u, v, tp).is_some() {
                    let mut t = hit;
                    while (tp - t) * sign > 0 {
                        if let Some(i) = cell(u, v, t) { if w.solid[i] { w.solid[i] = false; cleared += 1; } }
                        t += sign;
                    }
                }
                // Fill the plane and 3 cells behind it in empty cells only (existing voxels keep their colours), where
                // the wall is within WALL_CLEAR_M of the plane either way: a deeper recess (a window bay, a doorway) is
                // left open rather than covered.
                if bulge.abs() <= clear_band {
                    for k in 0..=3 {
                        let Some(i) = cell(u, v, tp + sign * k) else { continue };
                        if !w.solid[i] { w.solid[i] = true; filled += 1; w.rgb[i] = col; }
                    }
                }
            }
            let area = inl.len() as f64 * e * e * mpu * mpu;
            planes.push(json!({"direction": label, "area_m2": (area * 100.0).round() / 100.0, "tilt_from_axis_deg": (tilt * 10.0).round() / 10.0,
                "axis_snapped": snapped, "columns_flattened": cols.len(), "max_deviation_cm": (dev_max as f64 * e * mpu * 1000.0).round() / 10.0}));
            free.retain(|kk| !set.contains_key(kk));
        }
    }
    let total: f64 = planes.iter().filter_map(|p| p["area_m2"].as_f64()).sum();
    json!({"summary": {"planes": planes.len(), "area_m2": (total * 10.0).round() / 10.0, "cells_cleared": cleared, "cells_filled": filled,
        "centre_cell": centre}, "planes": planes})
}

/// Empty cells reachable from the room centre (the floor layer's centroid, 1.5 m up) through empty cells.
fn room_space(w: &World, floor_y: f64, mpu: f64) -> Vec<bool> {
    let n = w.dims;
    let len = n[0] * n[1] * n[2];
    let e = w.edge as f64;
    let fl = ((floor_y - w.origin.y as f64) / e).floor() as i64;
    let mut reach = vec![false; len];
    if fl < 0 || fl as usize >= n[1] { return reach; }
    let (mut sx, mut sz, mut c) = (0.0, 0.0, 0.0);
    for z in 0..n[2] { for x in 0..n[0] { if w.solid[(z * n[1] + fl as usize) * n[0] + x] { sx += x as f64; sz += z as f64; c += 1.0; } } }
    if c == 0.0 { return reach; }
    let y = ((fl as f64 + 1.5 / mpu / e) as usize).min(n[1] - 1);
    let mut q = VecDeque::new();
    // Seed: the empty cells of a small box around the centre (the centre cell itself may be furniture).
    let (cx, cz) = ((sx / c) as i64, (sz / c) as i64);
    for dz in -3..=3 { for dx in -3..=3 {
        let (x, z) = (cx + dx, cz + dz);
        if x < 0 || z < 0 || x as usize >= n[0] || z as usize >= n[2] { continue; }
        let i = (z as usize * n[1] + y) * n[0] + x as usize;
        if !w.solid[i] && !reach[i] { reach[i] = true; q.push_back(i); }
    }}
    while let Some(i) = q.pop_front() {
        let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
        for j in [(x > 0).then(|| i - 1), (x + 1 < n[0]).then(|| i + 1), (y > 0).then(|| i - n[0]), (y + 1 < n[1]).then(|| i + n[0]),
            (z > 0).then(|| i - n[0] * n[1]), (z + 1 < n[2]).then(|| i + n[0] * n[1])].into_iter().flatten() {
            if !w.solid[j] && !reach[j] { reach[j] = true; q.push_back(j); }
        }
    }
    reach
}

/// Watertight check after flattening: the room's free space (from its centre) must not reach cells it did not
/// reach before (a new opening through a wall). Cells the flattening cleared next to newly reached space are put
/// back until no new space is reached (at most 8 rounds). Reports the openings found and the cells restored.
fn seal(w: &mut World, before: &[bool], reach0: &[bool], floor_y: f64, mpu: f64) -> Value {
    let n = w.dims;
    let nb = |i: usize| {
        let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
        [(x > 0).then(|| i - 1), (x + 1 < n[0]).then(|| i + 1), (y > 0).then(|| i - n[0]), (y + 1 < n[1]).then(|| i + n[0]),
            (z > 0).then(|| i - n[0] * n[1]), (z + 1 < n[2]).then(|| i + n[0] * n[1])]
    };
    let mut first = None;
    let mut restored = 0usize;
    let mut rounds = 0;
    let mut last = 0usize;
    for round in 0..8 {
        let reach = room_space(w, floor_y, mpu);
        // Newly reached cells that were solid or unreachable before.
        let new: Vec<usize> = (0..reach.len()).filter(|&i| reach[i] && !reach0[i] && !before[i]).collect();
        last = new.len();
        if first.is_none() { first = Some(new.len()); }
        rounds = round;
        if new.is_empty() { break; }
        // Put back the cleared cells bordering the newly reached space.
        let mut put = 0;
        for &i in &new {
            for j in nb(i).into_iter().flatten() {
                if before[j] && !w.solid[j] { w.solid[j] = true; put += 1; }
            }
        }
        restored += put;
        if put == 0 { break; }
    }
    // See-through check: from the room centre along each axis direction (and up), a column that had solid
    // somewhere outward before must still have; otherwise its cleared cells are put back.
    let e = w.edge as f64;
    let fl = ((floor_y - w.origin.y as f64) / e).floor() as i64;
    let (mut sx, mut sz, mut c) = (0.0, 0.0, 0.0);
    if fl >= 0 && (fl as usize) < n[1] {
        for z in 0..n[2] { for x in 0..n[0] { if w.solid[(z * n[1] + fl as usize) * n[0] + x] { sx += x as f64; sz += z as f64; c += 1.0; } } }
    }
    let (mut holes, mut hole_restored) = (0usize, 0usize);
    if c > 0.0 {
        let centre = [(sx / c) as i64, (sz / c) as i64, fl + (1.5 / mpu / e) as i64];
        let idx = |x: i64, y: i64, z: i64| (z as usize * n[1] + y as usize) * n[0] + x as usize;
        for (axis, sign) in [(0usize, 1i64), (0, -1), (2, 1), (2, -1), (1, 1)] {
            let (nu, nv, nt) = match axis { 0 => (n[2], n[1], n[0]), 2 => (n[0], n[1], n[2]), _ => (n[0], n[2], n[1]) };
            let t0 = match axis { 0 => centre[0], 2 => centre[1], _ => centre[2] };
            let at = |u: i64, v: i64, t: i64| match axis { 0 => idx(t, v, u), 2 => idx(u, v, t), _ => idx(u, t, v) };
            for v in 0..nv as i64 { for u in 0..nu as i64 {
                let mut t = t0.clamp(0, nt as i64 - 1);
                let (mut had, mut has) = (false, false);
                while t >= 0 && t < nt as i64 { let i = at(u, v, t); had |= before[i]; has |= w.solid[i]; if has { break; } t += sign; }
                if had && !has {
                    holes += 1;
                    let mut t = t0.clamp(0, nt as i64 - 1);
                    while t >= 0 && t < nt as i64 { let i = at(u, v, t); if before[i] && !w.solid[i] { w.solid[i] = true; hole_restored += 1; } t += sign; }
                }
            }}
        }
    }
    json!({"new_openings_before_repair": first.unwrap_or(0), "new_openings_after": last, "cells_restored": restored, "rounds": rounds,
        "room_touches_border": reach0_border(reach0, n), "see_through_columns_repaired": holes, "see_through_cells_restored": hole_restored})
}

/// Whether the room's free space reaches the grid border (then the reachability check above cannot see openings).
fn reach0_border(reach: &[bool], n: [usize; 3]) -> bool {
    (0..reach.len()).any(|i| reach[i] && {
        let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
        x == 0 || y == 0 || z == 0 || x + 1 == n[0] || y + 1 == n[1] || z + 1 == n[2]
    })
}

/// Planar patches: area (m^2) and fit (cells) a patch needs; smaller, flatter ones (furniture faces) qualify too.
const PLANE_MIN_M2: f64 = 0.5;
const PLANE_MIN_M2_FLAT: f64 = 0.25;
const PLANE_RMS: f32 = 1.0;
const PLANE_RMS_FLAT: f32 = 0.5;
/// Region growing: normal within this angle of the patch's, and within this many cells of its plane.
const PLANE_GROW_DEG: f32 = 15.0;
const PLANE_GROW_DIST: f32 = 1.5;
/// Planes within this angle of a grid axis are snapped onto a grid layer.
const PLANE_AXIS_DEG: f32 = 5.0;
/// Furniture patches: room-side bumps cleared up to this many cells off the plane.
const PLANE_BAND: f32 = 2.0;
/// Walls / ceiling: loose growth, RANSAC inlier band (cells), minimum areas, axis snap within WALL_AXIS_DEG; the
/// room-side band is WALL_GROW_DIST cells.
const WALL_GROW_DEG: f32 = 25.0;
const WALL_GROW_DIST: f32 = 4.0;
const WALL_INLIER: f32 = 2.0;
const WALL_MIN_M2: f64 = 1.0;
const CEILING_MIN_M2: f64 = 2.0;
const WALL_AXIS_DEG: f32 = 10.0;
/// The wall bulge band (m) is at least this deep.
const WALL_BULGE_M: f64 = 0.4;
/// Wall bulges toward the room are cleared up to this depth (m); deeper columns are left as they are.
const WALL_CLEAR_M: f64 = 0.22;
/// A patch goes single-colour when its frame colours' median absolute luminance deviation is at most this.
const PLANE_FLAT_MAD: f32 = 12.0;

/// Flatten large planar surfaces into clean single-colour faces (after colouring). Surface cells get normals from
/// the occupancy gradient (5x5x5). Walls and the ceiling come first with loose rules (see WALL_*: one RANSAC plane
/// per loosely grown region, snapped onto a grid layer within 10 deg of an axis, room-side bumps up to 2 cells
/// flattened). Other patches (furniture) are region-grown (normal within PLANE_GROW_DEG, plane distance within
/// PLANE_GROW_DIST cells), fitted by PCA and refined on their inliers, and kept at >= PLANE_MIN_M2 with RMS < 1
/// cell (or >= PLANE_MIN_M2_FLAT with RMS < 0.5: furniture faces). Each kept patch, within its footprint (+1
/// cell): a gap-free digital plane plus 3 cells behind are solid, room-side bumps up to 2 cells off the plane are
/// cleared (only when empty 3 cells further into the room, so furniture against it stays), and when its frame
/// colours are uniform (luminance MAD <= PLANE_FLAT_MAD; never floors, never without frame colours) all its cells
/// take their median. Planes within PLANE_AXIS_DEG of a grid axis are snapped onto a grid layer (no staircase).
fn flatten_planes(w: &mut World, mpu: f64, framed: bool, floor_y: f64) -> Value {
    let n = w.dims;
    let len = n[0] * n[1] * n[2];
    let e = w.edge as f64;
    let xyz = |i: usize| (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
    let at = |x: i64, y: i64, z: i64| -> Option<usize> {
        (x >= 0 && y >= 0 && z >= 0 && (x as usize) < n[0] && (y as usize) < n[1] && (z as usize) < n[2])
            .then(|| (z as usize * n[1] + y as usize) * n[0] + x as usize)
    };
    let is_surface = |solid: &[bool], i: usize| -> bool {
        let (x, y, z) = xyz(i);
        let (x, y, z) = (x as i64, y as i64, z as i64);
        solid[i] && [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)].iter()
            .any(|(dx, dy, dz)| at(x + dx, y + dy, z + dz).is_none_or(|j| !solid[j]))
    };
    // Outward normals (toward empty space) from the occupancy gradient.
    let surface: Vec<usize> = (0..len).into_par_iter().filter(|&i| is_surface(&w.solid, i)).collect();
    let normals: HashMap<usize, Vec3> = surface.par_iter().map(|&i| {
        let (x, y, z) = xyz(i);
        let mut g = Vec3::ZERO;
        for dz in -2i64..=2 { for dy in -2i64..=2 { for dx in -2i64..=2 {
            if let Some(j) = at(x as i64 + dx, y as i64 + dy, z as i64 + dz) {
                if !w.solid[j] { g += Vec3::new(dx as f32, dy as f32, dz as f32); }
            }
        }}}
        (i, g.normalize_or_zero())
    }).collect::<Vec<_>>().into_iter().collect();
    let pos = |i: usize| { let (x, y, z) = xyz(i); Vec3::new(x as f32, y as f32, z as f32) };
    let cos_grow = PLANE_GROW_DEG.to_radians().cos();
    let mut taken = vec![false; len];
    // (normal, offset, inliers, region, rms, kind, room-side band, axis-snap tolerance deg)
    let mut patches: Vec<(Vec3, f32, Vec<usize>, Vec<usize>, f32, &str, f32, f32)> = Vec::new();
    // Seeds in order of local flatness: cells whose 26-neighbour surface normals agree most.
    let mut seeds: Vec<(usize, f32)> = surface.par_iter().map(|&i| {
        let nrm = normals[&i];
        let (x, y, z) = xyz(i);
        let mut agree = 0.0;
        for dz in -1i64..=1 { for dy in -1i64..=1 { for dx in -1i64..=1 {
            if let Some(m) = at(x as i64 + dx, y as i64 + dy, z as i64 + dz).and_then(|j| normals.get(&j)) { agree += nrm.dot(*m); }
        }}}
        (i, agree)
    }).collect();
    seeds.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    // Walls and ceiling first, with loose rules: region-grown within WALL_GROW_DEG of the running normal and
    // WALL_GROW_DIST cells of the running plane, one RANSAC plane per region (WALL_INLIER cells band), kept at
    // >= WALL_MIN_M2 (walls) / CEILING_MIN_M2 (ceiling). A bowed fused wall becomes one plane.
    let floor_layer = (floor_y as f32 - w.origin.y) / w.edge;
    let ceiling_from = floor_layer + (1.8 / mpu) as f32 / w.edge;
    let class = |i: usize| -> u8 {
        let m = normals[&i];
        if m.y.abs() < WALL_GROW_DEG.to_radians().sin() { 1 } else if m.y < -WALL_GROW_DEG.to_radians().cos() && pos(i).y > ceiling_from { 2 } else { 0 }
    };
    let cos_wall = WALL_GROW_DEG.to_radians().cos();
    let mut rng = 0x9E3779B97F4A7C15u64;
    for &(seed, _) in &seeds {
        let kind = class(seed);
        if taken[seed] || kind == 0 || normals[&seed] == Vec3::ZERO { continue; }
        let mut nrm = normals[&seed];
        let mut d = nrm.dot(pos(seed));
        let mut sum_n = nrm;
        let mut region = vec![seed];
        let mut inq = HashMap::new();
        inq.insert(seed, ());
        let mut q = VecDeque::from([seed]);
        while let Some(i) = q.pop_front() {
            let (x, y, z) = xyz(i);
            for dz in -1i64..=1 { for dy in -1i64..=1 { for dx in -1i64..=1 {
                let Some(j) = at(x as i64 + dx, y as i64 + dy, z as i64 + dz) else { continue };
                if taken[j] || inq.contains_key(&j) { continue; }
                let Some(m) = normals.get(&j) else { continue };
                if class(j) != kind || m.dot(nrm) < cos_wall || (nrm.dot(pos(j)) - d).abs() > WALL_GROW_DIST { continue; }
                inq.insert(j, ());
                region.push(j);
                q.push_back(j);
                sum_n += *m;
                if region.len() % 64 == 0 {
                    nrm = sum_n.normalize_or_zero();
                    d = region.iter().map(|&c| nrm.dot(pos(c))).sum::<f32>() / region.len() as f32;
                }
            }}}
        }
        let min_m2 = if kind == 1 { WALL_MIN_M2 } else { CEILING_MIN_M2 };
        let need = ((min_m2 / (mpu * mpu)) / (e * e)).ceil() as usize;
        if region.len() < need { continue; }
        // RANSAC: 3-point planes within WALL_GROW_DEG of the region normal; most inliers within WALL_INLIER cells.
        let mean_n = sum_n.normalize_or_zero();
        let mut best: Option<(Vec3, f32, usize)> = None;
        for _ in 0..200 {
            let mut pick = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; pos(region[(rng % region.len() as u64) as usize]) };
            let (a, b, c) = (pick(), pick(), pick());
            let mut pn = (b - a).cross(c - a).normalize_or_zero();
            if pn == Vec3::ZERO { continue; }
            if pn.dot(mean_n) < 0.0 { pn = -pn; }
            if pn.dot(mean_n) < cos_wall { continue; }
            let pd = pn.dot(a);
            let k = region.iter().filter(|&&i| (pn.dot(pos(i)) - pd).abs() <= WALL_INLIER).count();
            if best.is_none_or(|bb| k > bb.2) { best = Some((pn, pd, k)); }
        }
        let Some((pn, pd, _)) = best else { continue };
        let inl: Vec<usize> = region.iter().copied().filter(|&i| (pn.dot(pos(i)) - pd).abs() <= WALL_INLIER).collect();
        if inl.len() < need { continue; }
        // Least-squares refit on the inliers.
        let c = inl.iter().map(|&i| pos(i).as_dvec3()).sum::<DVec3>() / inl.len() as f64;
        let mut cov = DMat3::ZERO;
        for &i in &inl { let dv = pos(i).as_dvec3() - c; cov += DMat3::from_cols(dv * dv.x, dv * dv.y, dv * dv.z); }
        let mut nn = smallest_eigvec(cov).as_vec3();
        if nn.dot(mean_n) < 0.0 { nn = -nn; }
        let dd = nn.dot(c.as_vec3());
        let rms = (inl.iter().map(|&i| (nn.dot(pos(i)) - dd).powi(2)).sum::<f32>() / inl.len() as f32).sqrt();
        for &i in &region { taken[i] = true; }
        // Bulges are flatten_walls' job: here walls clear no deeper than furniture faces.
        patches.push((nn, dd, inl, region, rms, if kind == 1 { "wall" } else { "ceiling" }, PLANE_BAND, WALL_AXIS_DEG));
    }
    let min_cells = ((PLANE_MIN_M2_FLAT / (mpu * mpu)) / (e * e)).ceil() as usize;
    for &(seed, _) in &seeds {
        if taken[seed] || normals[&seed] == Vec3::ZERO { continue; }
        let mut nrm = normals[&seed];
        let mut d = nrm.dot(pos(seed));
        let mut cells = vec![seed];
        let mut inq = HashMap::new();
        inq.insert(seed, ());
        let mut q = VecDeque::from([seed]);
        let mut sum_n = nrm;
        while let Some(i) = q.pop_front() {
            let (x, y, z) = xyz(i);
            for dz in -1i64..=1 { for dy in -1i64..=1 { for dx in -1i64..=1 {
                let Some(j) = at(x as i64 + dx, y as i64 + dy, z as i64 + dz) else { continue };
                if taken[j] || inq.contains_key(&j) { continue; }
                let Some(m) = normals.get(&j) else { continue };
                if m.dot(nrm) < cos_grow || (nrm.dot(pos(j)) - d).abs() > PLANE_GROW_DIST { continue; }
                inq.insert(j, ());
                cells.push(j);
                q.push_back(j);
                sum_n += *m;
                // Keep the patch plane through the running mean, with the running mean normal.
                if cells.len() % 64 == 0 {
                    nrm = sum_n.normalize_or_zero();
                    d = cells.iter().map(|&c| nrm.dot(pos(c))).sum::<f32>() / cells.len() as f32;
                }
            }}}
        }
        if cells.len() < min_cells { continue; }
        // PCA fit, refined once on the inliers within 1 cell.
        let fit = |cs: &[usize]| -> (Vec3, f32, f32) {
            let c = cs.iter().map(|&i| pos(i).as_dvec3()).sum::<DVec3>() / cs.len() as f64;
            let mut cov = DMat3::ZERO;
            for &i in cs { let d = pos(i).as_dvec3() - c; cov += DMat3::from_cols(d * d.x, d * d.y, d * d.z); }
            let mut nn = smallest_eigvec(cov).as_vec3();
            if nn.dot(sum_n) < 0.0 { nn = -nn; }
            let dd = nn.dot(c.as_vec3());
            let rms = (cs.iter().map(|&i| (nn.dot(pos(i)) - dd).powi(2)).sum::<f32>() / cs.len() as f32).sqrt();
            (nn, dd, rms)
        };
        let (n1, d1, _) = fit(&cells);
        let inl: Vec<usize> = cells.iter().copied().filter(|&i| (n1.dot(pos(i)) - d1).abs() <= 1.0).collect();
        if inl.len() < min_cells { continue; }
        let (nn, dd, rms) = fit(&inl);
        let area = inl.len() as f64 * e * e * mpu * mpu;
        let ok = (area >= PLANE_MIN_M2 && rms < PLANE_RMS) || (area >= PLANE_MIN_M2_FLAT && rms < PLANE_RMS_FLAT);
        if !ok { continue; }
        for &i in &cells { taken[i] = true; }
        patches.push((nn, dd, inl, cells, rms, "furniture", PLANE_BAND, PLANE_AXIS_DEG));
    }
    // Re-voxelize each patch.
    let (mut cleared, mut filled, mut snapped, mut recoloured, mut flat_patches) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut list = Vec::new();
    let mut total_area = 0.0;
    for (mut nn, mut dd, cells, region, rms, kind, band, axis_deg) in patches {
        // Snap near-axis planes onto a grid layer.
        let axis = [Vec3::X, Vec3::Y, Vec3::Z].into_iter().max_by(|a, b| a.dot(nn).abs().total_cmp(&b.dot(nn).abs())).unwrap();
        let tilt = axis.dot(nn).abs().clamp(-1.0, 1.0).acos().to_degrees();
        let near_axis = tilt <= axis_deg;
        if near_axis {
            nn = axis * axis.dot(nn).signum();
            let mut ds: Vec<f32> = cells.iter().map(|&i| nn.dot(pos(i))).collect();
            ds.sort_by(f32::total_cmp);
            dd = ds[ds.len() / 2].round();
            snapped += 1;
        }
        // In-plane axes and the footprint (dilated by 1).
        let t1 = if nn.y.abs() < 0.9 { nn.cross(Vec3::Y).normalize() } else { nn.cross(Vec3::X).normalize() };
        let t2 = nn.cross(t1);
        let key = |p: Vec3| ((p.dot(t1)).round() as i32, (p.dot(t2)).round() as i32);
        let mut foot: HashMap<(i32, i32), ()> = HashMap::new();
        for &i in &region { let (a, b) = key(pos(i)); for da in -1..=1 { for db in -1..=1 { foot.insert((a + da, b + db), ()); } } }
        // The largest room-side / behind deviation of the region from the final plane (cells).
        let devs: Vec<f32> = region.iter().map(|&i| nn.dot(pos(i)) - dd).collect();
        let (dev_room, dev_back) = (devs.iter().copied().fold(0.0f32, f32::max), devs.iter().copied().fold(0.0f32, f32::min));
        // Patch colour: median of its frame-sourced colours (else of all its colours).
        // (`framed`: the frames coloured the world; else `seen` is all true and says nothing.)
        let mut src: Vec<[u8; 3]> = if framed { cells.iter().filter(|&&i| w.seen[i]).map(|&i| w.rgb[i]).collect() } else { Vec::new() };
        let from_frames = !src.is_empty();
        if src.is_empty() { src = cells.iter().map(|&i| w.rgb[i]).collect(); }
        let col = [0, 1, 2].map(|k| { let mut v: Vec<u8> = src.iter().map(|c| c[k]).collect(); let m = v.len() / 2; *v.select_nth_unstable(m).1 });
        // A patterned patch (a rug, a bookshelf face) keeps its cells' colours: only uniform ones go flat.
        let luma = |c: &[u8; 3]| 0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32;
        let ml = luma(&col);
        let mut dev: Vec<f32> = src.iter().map(|c| (luma(c) - ml).abs()).collect();
        let mad = { let m = dev.len() / 2; *dev.select_nth_unstable_by(m, f32::total_cmp).1 };
        // Floors keep their colours (rugs), and so does any patch without frame colours (nothing to trust).
        let floor = nn.y > PLANE_AXIS_DEG.to_radians().cos();
        let flat = from_frames && !floor && mad <= PLANE_FLAT_MAD;
        // Gap-free (6-connected) digital plane: |s| <= half the L1 norm of the normal.
        let half = 0.5 * (nn.x.abs() + nn.y.abs() + nn.z.abs());
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &i in &region { lo = lo.min(pos(i)); hi = hi.max(pos(i)); }
        let (lo, hi) = ((lo - (band + 4.0)).max(Vec3::ZERO), (hi + (band + 4.0)).min(Vec3::new(n[0] as f32 - 1.0, n[1] as f32 - 1.0, n[2] as f32 - 1.0)));
        for z in lo.z as usize..=hi.z as usize { for y in lo.y as usize..=hi.y as usize { for x in lo.x as usize..=hi.x as usize {
            let p = Vec3::new(x as f32, y as f32, z as f32);
            let sd = nn.dot(p) - dd; // > 0: room side
            if sd > band + half || sd < -3.0 - half || !foot.contains_key(&key(p)) { continue; }
            let i = (z * n[1] + y) * n[0] + x;
            if sd > half {
                // A room-side bump: cleared unless something solid stands just beyond the band (furniture).
                if !w.solid[i] { continue; }
                let q = p + nn * (band + 2.0 - sd);
                let deeper = at(q.x.round() as i64, q.y.round() as i64, q.z.round() as i64).is_some_and(|j| w.solid[j]);
                if !deeper { w.solid[i] = false; cleared += 1; }
            } else {
                if !w.solid[i] { w.solid[i] = true; filled += 1; w.rgb[i] = col; w.seen[i] = from_frames; }
                else if flat { w.rgb[i] = col; w.seen[i] = from_frames; recoloured += 1; }
            }
        }}}
        let area = cells.len() as f64 * e * e * mpu * mpu;
        total_area += area;
        let cm = |c: f32| (c as f64 * e * mpu * 1000.0).round() / 10.0;
        list.push(json!({"kind": kind, "area_m2": (area * 100.0).round() / 100.0, "normal": nn.to_array().map(|v| (v * 1e3).round() / 1e3),
            "tilt_from_axis_deg": (tilt * 10.0).round() / 10.0, "region_cells": region.len(),
            "max_deviation_cm": {"room_side": cm(dev_room), "behind": cm(-dev_back)},
            "rms_cells": (rms * 100.0).round() / 100.0, "axis_snapped": near_axis, "colour": col, "colour_from_frames": from_frames,
            "flat_colour": flat, "luma_mad": (mad * 10.0).round() / 10.0}));
        if flat { flat_patches += 1; }
    }
    list.sort_by(|a, b| b["area_m2"].as_f64().unwrap_or(0.0).total_cmp(&a["area_m2"].as_f64().unwrap_or(0.0)));
    let summary = json!({"patches": list.len(), "area_m2": (total_area * 10.0).round() / 10.0, "axis_snapped": snapped,
        "flat_coloured_patches": flat_patches,
        "walls": list.iter().filter(|p| p["kind"] == "wall").count(), "ceilings": list.iter().filter(|p| p["kind"] == "ceiling").count(),
        "cells_cleared": cleared, "cells_filled": filled, "cells_flat_coloured": recoloured, "surface_cells": surface.len()});
    json!({"summary": summary, "patches": list})
}

/// The VXG1 binary (CONTRACT.md): chunk_count, then per non-empty 32^3 chunk its coords, cell count,
/// 4096-byte bit mask (bit lx + 32*(ly + 32*lz), LSB first) and the set cells' sRGB in bit order.
fn encode(w: &World) -> (Vec<u8>, usize, usize) {
    let nc = w.dims.map(|d| d.div_ceil(CHUNK));
    let mut out = b"VXG1".to_vec();
    out.extend_from_slice(&0u32.to_le_bytes());
    let (mut chunks, mut cells) = (0u32, 0usize);
    for cz in 0..nc[2] { for cy in 0..nc[1] { for cx in 0..nc[0] {
        let mut mask = [0u8; CHUNK * CHUNK * CHUNK / 8];
        let mut rgb = Vec::new();
        for lz in 0..CHUNK { for ly in 0..CHUNK { for lx in 0..CHUNK {
            let (x, y, z) = (cx * CHUNK + lx, cy * CHUNK + ly, cz * CHUNK + lz);
            if x >= w.dims[0] || y >= w.dims[1] || z >= w.dims[2] { continue; }
            let k = w.at(x, y, z);
            if !w.solid[k] { continue; }
            let bit = lx + CHUNK * (ly + CHUNK * lz);
            mask[bit >> 3] |= 1 << (bit & 7);
            rgb.extend_from_slice(&w.rgb[k]);
        }}}
        if rgb.is_empty() { continue; }
        for c in [cx, cy, cz] { out.extend_from_slice(&(c as i32).to_le_bytes()); }
        out.extend_from_slice(&((rgb.len() / 3) as u32).to_le_bytes());
        out.extend_from_slice(&mask);
        out.extend_from_slice(&rgb);
        chunks += 1;
        cells += rgb.len() / 3;
    }}}
    out[4..8].copy_from_slice(&chunks.to_le_bytes());
    (out, chunks as usize, cells)
}

/// Spawn at a filmed capture camera's own column, standing on the floor plane (+-0.15 m of `floor_y`; never a
/// furniture top) with a free 0.5 x 1.75 m body box; yaw toward the room centre (the floor layer's centroid);
/// pitch the camera's own tilt clamped to [-15, 0] deg. Each candidate's first frame is scored with a 32 x 18
/// ray grid at the camera's FOV (16:9) through the voxel world: rays that hit nothing within 8 m, or hit a cell
/// no capture frame saw, are unseen. Candidates over 25% unseen are dropped (all over: only those within 5 points
/// of the least unseen stay); the best has the
/// most clear distance ahead (eye and chest height, counted to 3 m), then the least unseen. Eye `EYE_M` above
/// the floor. Without any candidate: the first capture camera's x/z on the floor plane.
fn spawn(w: &World, capture: &[Value], first: &Value, floor_y: f64, mpu: f64) -> Value {
    let e = w.edge as f64;
    let cell = |v: f64, a: usize| ((v - w.origin[a] as f64) / e).floor() as i64;
    let inside = |x: i64, y: i64, z: i64| x >= 0 && y >= 0 && z >= 0 && (x as usize) < w.dims[0] && (y as usize) < w.dims[1] && (z as usize) < w.dims[2];
    let solid = |x: i64, y: i64, z: i64| inside(x, y, z) && w.solid[w.at(x as usize, y as usize, z as usize)];
    let half = BODY_HALF_M / mpu;
    let box_free = |wx: f64, wz: f64, fy: f64| -> bool {
        let (x0, x1, z0, z1) = (cell(wx - half, 0), cell(wx + half, 0), cell(wz - half, 2), cell(wz + half, 2));
        if x0 < 0 || z0 < 0 || x1 >= w.dims[0] as i64 || z1 >= w.dims[2] as i64 { return false; }
        let (y0, y1) = (cell(fy + 0.5 * e, 1), cell(fy + BODY_H_M / mpu, 1));
        (z0..=z1).all(|z| (y0..=y1).all(|y| (x0..=x1).all(|x| !solid(x, y, z))))
    };
    let tol = SPAWN_FLOOR_TOL_M / mpu;
    let floor_top = |x: f64, z: f64| -> Option<f64> {
        let (cx, cz) = (cell(x, 0), cell(z, 2));
        (cell(floor_y - tol, 1)..=cell(floor_y + tol, 1)).rev()
            .find(|&y| solid(cx, y, cz) && !solid(cx, y + 1, cz)).map(|y| w.origin.y as f64 + (y + 1) as f64 * e)
            .filter(|t| (t - floor_y).abs() <= tol + e)
    };
    // Free distance (m, up to 6) along yaw at eye and chest height.
    let clear = |x: f64, z: f64, fy: f64, yaw: f64| -> f64 {
        let (dx, dz) = (yaw.cos(), yaw.sin());
        let mut t = 0.0;
        while t < 6.0 / mpu {
            let (px, pz) = (x + dx * t, z + dz * t);
            if [EYE_M, 1.0].iter().any(|h| solid(cell(px, 0), cell(fy + h / mpu, 1), cell(pz, 2))) { break; }
            t += 0.5 * e;
        }
        t * mpu
    };
    // Room centre: centroid of the floor layer's columns.
    let fl = cell(floor_y - 0.5 * e, 1);
    let (mut sx, mut sz, mut nf) = (0.0, 0.0, 0.0);
    if fl >= 0 && (fl as usize) < w.dims[1] {
        for z in 0..w.dims[2] { for x in 0..w.dims[0] {
            if w.solid[w.at(x, fl as usize, z)] {
                sx += w.origin.x as f64 + (x as f64 + 0.5) * e;
                sz += w.origin.z as f64 + (z as f64 + 0.5) * e;
                nf += 1.0;
            }
        }}
    }
    let centre = if nf > 0.0 { [sx / nf, sz / nf] } else { [0.0, 0.0] };
    // Unseen fraction of the first frame from eye, looking (yaw, pitch) with vfov.
    let unseen = |eye: [f64; 3], yaw: f64, pitch: f64, vfov: f64| -> f64 {
        let f = DVec3::new(pitch.cos() * yaw.cos(), pitch.sin(), pitch.cos() * yaw.sin());
        let r = f.cross(DVec3::Y).normalize();
        let u = r.cross(f);
        let th = (vfov.to_radians() * 0.5).tan();
        let (cols, rows) = (32, 18);
        let mut bad = 0;
        for j in 0..rows { for i in 0..cols {
            let (nx, ny) = ((i as f64 + 0.5) / cols as f64 * 2.0 - 1.0, 1.0 - (j as f64 + 0.5) / rows as f64 * 2.0);
            let d = (f + r * (nx * th * 16.0 / 9.0) + u * (ny * th)).normalize();
            let mut t = 0.0;
            let mut hit = None;
            while t < 8.0 / mpu {
                let p = DVec3::from_array(eye) + d * t;
                let (x, y, z) = (cell(p.x, 0), cell(p.y, 1), cell(p.z, 2));
                if solid(x, y, z) { hit = Some(w.seen[w.at(x as usize, y as usize, z as usize)]); break; }
                t += 0.5 * e;
            }
            if hit != Some(true) { bad += 1; }
        }}
        bad as f64 / (cols * rows) as f64
    };
    let pose = |c: &Value| -> Option<([f64; 3], f64, f64)> {
        let p = viewer::v3(&c["position"])?;
        let l = viewer::v3(&c["look_at"])?;
        let d = [l[0] - p[0], l[1] - p[1], l[2] - p[2]];
        Some((p, d[1].atan2(d[0].hypot(d[2])).to_degrees(), c["vfov_deg"].as_f64().unwrap_or(60.0)))
    };
    // (x, z, floor, yaw, pitch, vfov, clear, unseen, capture index)
    let mut scored: Vec<(f64, f64, f64, f64, f64, f64, f64, f64, usize)> = Vec::new();
    for (k, c) in capture.iter().enumerate() {
        let Some((p, cam_pitch, vfov)) = pose(c) else { continue };
        let Some(fy) = floor_top(p[0], p[2]) else { continue };
        if !box_free(p[0], p[2], fy) { continue; }
        let (dx, dz) = (centre[0] - p[0], centre[1] - p[2]);
        let yaw = if dx.hypot(dz) * mpu > 0.3 { dz.atan2(dx) } else { continue };
        let pitch = cam_pitch.clamp(-15.0, 0.0).to_radians();
        let eye = [p[0], fy + EYE_M / mpu, p[2]];
        scored.push((p[0], p[2], fy, yaw, pitch, vfov, clear(p[0], p[2], fy, yaw), unseen(eye, yaw, pitch, vfov), k));
    }
    let ok: Vec<_> = scored.iter().filter(|s| s.7 <= 0.25).collect();
    // All over the limit: the least unseen ones (within 5 points of the best) compete on clear distance.
    let least = scored.iter().map(|s| s.7).fold(f64::MAX, f64::min);
    let pool: Vec<_> = if ok.is_empty() { scored.iter().filter(|s| s.7 <= least + 0.05).collect() } else { ok };
    let best = pool.into_iter().max_by(|a, b| {
        let key = |s: &(f64, f64, f64, f64, f64, f64, f64, f64, usize)| ((s.6.min(3.0) * 4.0).floor(), -s.7);
        let (ka, kb) = (key(a), key(b));
        ka.0.total_cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then(b.8.cmp(&a.8))
    }).copied();
    let candidates = scored.len();
    let (x, z, fy, yaw, pitch, vfov, open, unseen_frac, k, method) = match best {
        Some(s) => (s.0, s.1, s.2, s.3, s.4, s.5, s.6, s.7, Some(s.8), "capture_column_facing_room_centre"),
        None => {
            let p = viewer::v3(&first["position"]).unwrap_or([0.0, 1.5, 0.0]);
            (p[0], p[2], floor_y, 0.0, 0.0, first["vfov_deg"].as_f64().unwrap_or(60.0), 0.0, 1.0, None, "floor_plane")
        }
    };
    let eye = [x, fy + EYE_M / mpu, z];
    let r = 2.0 / mpu;
    let look = [eye[0] + r * pitch.cos() * yaw.cos(), eye[1] + r * pitch.sin(), eye[2] + r * pitch.cos() * yaw.sin()];
    let round = |a: [f64; 3]| a.map(|v| (v * 1e4).round() / 1e4 + 0.0);
    json!({"position": round(eye), "look_at": round(look), "floor_y": (fy * 1e4).round() / 1e4 + 0.0, "vfov_deg": vfov,
        "method": method, "pitch_deg": (pitch.to_degrees() * 10.0).round() / 10.0, "yaw_deg": (yaw.to_degrees() * 10.0).round() / 10.0,
        "clear_ahead_m": (open * 100.0).round() / 100.0, "unseen_frac": (unseen_frac * 1000.0).round() / 1000.0,
        "room_centre": centre.map(|v| (v * 1e3).round() / 1e3), "candidates": candidates,
        "capture_index": k, "capture_image": k.map(|k| capture[k]["image"].clone())})
}

/// Unit eigenvector of the smallest eigenvalue of the symmetric `m` (power iteration on trace*I - m).
fn smallest_eigvec(m: DMat3) -> DVec3 {
    let t = m.x_axis.x + m.y_axis.y + m.z_axis.z;
    let a = DMat3::from_diagonal(DVec3::splat(t)) - m;
    let mut v = DVec3::new(0.1, 1.0, 0.05).normalize();
    for _ in 0..200 { v = (a * v).normalize_or_zero(); if v == DVec3::ZERO { return DVec3::Y; } }
    v
}

/// True vertical: phone video is held about roll-free, so the capture cameras' right axes are near
/// horizontal and up is the direction least spanned by them (sum r r^T, smallest eigenvector, signed to the
/// package's +Y). A RANSAC floor plane through the low opaque splats wins when it has strong support and
/// agrees within 5 deg. Returns the levelling transform (a rotation about the floor point under the cameras)
/// when the correction exceeds 1 deg, and what was measured.
fn level(views: &[collision::View], g: &crate::ply::Gaussians, floor_y: f64, mpu: f64) -> (Option<DMat4>, Value) {
    let mut m = DMat3::ZERO;
    for v in views {
        let r = v.right().as_dvec3().normalize_or_zero();
        m += DMat3::from_cols(r * r.x, r * r.y, r * r.z);
    }
    let mut up_cam = smallest_eigvec(m);
    if up_cam.y < 0.0 { up_cam = -up_cam; }
    // Floor plane: RANSAC over opaque splats within 0.6 m of the package floor, normal within 20 deg of up_cam.
    let band = 0.6 / mpu;
    let pts: Vec<DVec3> = (0..g.len()).filter(|&i| g.alpha(i) >= 0.5)
        .map(|i| DVec3::new(g.pos[i][0] as f64, g.pos[i][1] as f64, g.pos[i][2] as f64))
        .filter(|p| p.is_finite() && (p.y - floor_y).abs() <= band).collect();
    let step = (pts.len() / 100_000).max(1);
    let pts: Vec<DVec3> = pts.into_iter().step_by(step).collect();
    let (mut best_n, mut best_a, mut best_in) = (DVec3::Y, DVec3::ZERO, 0usize);
    let tol = 0.02 / mpu;
    if pts.len() >= 100 {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = |n: usize| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed % n as u64) as usize };
        for _ in 0..400 {
            let (a, b, c) = (pts[rnd(pts.len())], pts[rnd(pts.len())], pts[rnd(pts.len())]);
            let mut n = (b - a).cross(c - a).normalize_or_zero();
            if n == DVec3::ZERO { continue; }
            if n.dot(up_cam) < 0.0 { n = -n; }
            if n.dot(up_cam) < 20f64.to_radians().cos() { continue; }
            let k = pts.iter().filter(|p| (**p - a).dot(n).abs() <= tol).count();
            if k > best_in { best_in = k; best_n = n; best_a = a; }
        }
        // Refine on the inliers: the normal of their best-fit plane.
        let near: Vec<DVec3> = pts.iter().copied().filter(|p| (*p - best_a).dot(best_n).abs() <= tol).collect();
        if near.len() >= 3 {
            let c = near.iter().copied().sum::<DVec3>() / near.len() as f64;
            let mut cov = DMat3::ZERO;
            for p in &near { let d = *p - c; cov += DMat3::from_cols(d * d.x, d * d.y, d * d.z); }
            let mut n = smallest_eigvec(cov);
            if n.dot(best_n) < 0.0 { n = -n; }
            if n.dot(best_n) > 5f64.to_radians().cos() { best_n = n; }
        }
    }
    let support = if pts.is_empty() { 0.0 } else { best_in as f64 / pts.len() as f64 };
    let agree = best_n.dot(up_cam).clamp(-1.0, 1.0).acos().to_degrees();
    let (up, method) = if support >= 0.2 && agree <= 5.0 { (best_n, "floor_plane") } else { (up_cam, "camera_right_axes") };
    let angle = up.dot(DVec3::Y).clamp(-1.0, 1.0).acos().to_degrees();
    let r2 = |v: f64| (v * 1e3).round() / 1e3;
    let info = json!({"angle_deg": r2(angle), "method": method, "up_estimate": up.to_array().map(|v| (v * 1e5).round() / 1e5),
        "camera_up": up_cam.to_array().map(|v| (v * 1e5).round() / 1e5),
        "camera_up_tilt_deg": r2(up_cam.dot(DVec3::Y).clamp(-1.0, 1.0).acos().to_degrees()),
        "floor_normal": best_n.to_array().map(|v| (v * 1e5).round() / 1e5), "floor_support": r2(support),
        "floor_vs_camera_deg": r2(agree), "floor_points": pts.len(), "applied": angle > 1.0});
    let c = views.iter().map(|v| v.c.as_dvec3()).sum::<DVec3>() / views.len() as f64;
    let pivot = DVec3::new(c.x, floor_y, c.z);
    let mut info = info;
    info["pivot"] = json!(pivot.to_array());
    if angle <= 1.0 { return (None, info); }
    let rot = DMat4::from_quat(DQuat::from_rotation_arc(up, DVec3::Y));
    (Some(DMat4::from_translation(pivot) * rot * DMat4::from_translation(-pivot)), info)
}

/// scene.json's capture cameras and start camera moved by `m` (the game's copy only).
fn level_scene(scene: &mut Value, m: &DMat4) {
    let mv = |v: &mut Value| if let Some(p) = viewer::v3(v) {
        let q = m.transform_point3(DVec3::from_array(p));
        *v = json!([q.x, q.y, q.z]);
    };
    if let Some(cams) = scene["capture_cameras"].as_array_mut() {
        for c in cams { mv(&mut c["position"]); mv(&mut c["look_at"]); }
    }
    if scene["start_camera"].is_object() { mv(&mut scene["start_camera"]["position"]); mv(&mut scene["start_camera"]["look_at"]); }
}

/// base64 data/*.js bytes of `raw` bytes.
fn js_size(raw: u64) -> u64 { raw.div_ceil(3) * 4 + 128 }

/// One voxel world at a given edge: the fusion, the game world, its binary and what made it.
struct Built {
    cm: f64,
    depth_h: usize,
    close_r: usize,
    views: usize,
    max_depth: f32,
    timings: [f64; 4],
    floor_y: f64,
    cams: Vec<Vec3>,
    clip: (Vec3, Vec3),
    world: World,
    stats: Value,
    carve: Carve,
    bin: Vec<u8>,
    count: usize,
}

/// Fuse `g` (levelled by `level_m`) at `cm` and build the game world.
fn build_at(g: &crate::ply::Gaussians, scene: &Value, ply_hint: Option<&Path>, level_m: Option<&DMat4>, path: &[Vec3], cm: f64,
    mpu: f64) -> Result<Built> {
    // Finer voxels want finer depth maps: 360 px tall at 5 cm (twice the collision export's), 600 at 3 cm.
    let depth_h = ((360.0 * 5.0 / cm).round() as usize).clamp(360, 720);
    let mut views = collision::scene_views(scene, None, ply_hint, depth_h)?;
    ensure!(!views.is_empty(), "no capture cameras");
    if let Some(m) = level_m { for v in &mut views { v.transform(m); } }
    let edge = (cm / 100.0 / mpu) as f32;
    let close_r = ((0.10 / mpu / edge as f64).round() as usize).max(2);
    // The whole room: depth out to at least MIN_DEPTH_M (the collision export's 3x median depth stops short of
    // far walls in small rooms), every surface point (not the 1-99% range: that cuts off far walls) and the
    // detected floor's extent, all within the crop box.
    let (mut clo, mut chi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for v in &views { clo = clo.min(v.c); chi = chi.max(v.c); }
    let detected = scene["floor"]["detected"].as_bool() == Some(true);
    let floor0 = if detected { scene["floor"]["y"].as_f64().unwrap_or(0.0) as f32 } else { clo.y - (2.0 / mpu) as f32 };
    let xz = (CROP_XZ_M / mpu) as f32;
    let clip = (Vec3::new(clo.x - xz, floor0 - (CROP_DOWN_M / mpu) as f32, clo.z - xz),
        Vec3::new(chi.x + xz, chi.y + (CROP_UP_M / mpu) as f32, chi.z + xz));
    let ext = &scene["floor"]["extent"];
    let include = match (detected, viewer::v3(&json!([ext["min"][0], 0, ext["min"][1]])), viewer::v3(&json!([ext["max"][0], 0, ext["max"][1]]))) {
        (true, Some(a), Some(b)) if ext["min"].is_array() && ext["max"].is_array() =>
            Some((Vec3::new(a[0] as f32, floor0 - 0.2 / mpu as f32, a[2] as f32), Vec3::new(b[0] as f32, chi.y, b[2] as f32))),
        _ => None,
    };
    let opts = collision::FuseOpts { voxel: edge, close: close_r, depth_scale: 3.0, min_depth: (MIN_DEPTH_M / mpu) as f32,
        bound_q: 0.0, include, clip: Some(clip) };
    let fused = collision::fuse(g, &views, scene, &opts)?;
    let max_depth = fused.max_depth;
    let (t_depth, t_fuse, t_vox) = (fused.t_depth, fused.t_fuse, fused.t_vox);
    let floor_y = fused.floor["y"].as_f64().or_else(|| scene["floor"]["y"].as_f64()).unwrap_or(0.0);
    let cams: Vec<Vec3> = views.iter().map(|v| v.c).collect();
    let t1 = Instant::now();
    let (world, stats, carve) = build_world(fused, g, &cams, path, mpu)?;
    let (bin, chunks, count) = encode(&world);
    let cm_actual = world.edge as f64 * mpu * 100.0;
    eprintln!("export-game: {count} voxels in {chunks} chunks at {cm_actual:.1} cm ({}x{}x{}), {} bytes",
        world.dims[0], world.dims[1], world.dims[2], bin.len());
    Ok(Built { cm, depth_h, close_r, views: views.len(), max_depth, timings: [t_depth, t_fuse, t_vox, t1.elapsed().as_secs_f64()],
        floor_y, cams, clip, world, stats, carve, bin, count })
}

/// The viewer's full model for splat mode's still frames: `full` (a web dir, scene.spz or candidate dir) or the
/// run's full model as export-viewer resolves it; None when that is the package's own `main_spz`.
fn display_full(full: Option<&Path>, run: &Path, main_spz: &Path) -> Result<Option<PathBuf>> {
    let full = match full {
        Some(f) => viewer::resolve_full(Some(f), run)?,
        None => viewer::resolve_full(None, run).unwrap_or(None),
    };
    Ok(full.filter(|f| fs::canonicalize(f).ok() != fs::canonicalize(main_spz).ok()))
}

/// Write the game folder; returns a summary (out, index, voxels, spawn, bytes).
pub fn run(a: &ExportGameArgs) -> Result<Value> {
    let t0 = Instant::now();
    let (pkg, run, mut scene) = viewer::resolve_package(&a.source)?;
    let main_spz = pkg.join(scene["splat"].as_str().unwrap_or("scene.spz"));
    ensure!(main_spz.is_file(), "{} not found", main_spz.display());
    // Package frames are metric (y_up_meters: meters_per_unit 1); there is no separate calibrated scale.
    let mpu = scene["meters_per_unit"].as_f64().filter(|m| m.is_finite() && *m > 0.0).unwrap_or(1.0);
    let out = a.out.clone().unwrap_or_else(|| pkg.parent().unwrap_or(&pkg).join("game"));
    let title = a.title.clone().unwrap_or_else(|| viewer::default_title(&run));

    let (mut g, splat_source) = collision::package_splats(&pkg, &scene)?;
    let ply_hint = scene["source"]["ply"].as_str().map(PathBuf::from);
    // Level the world: the page's camera is roll-free, so the scene's up must be true vertical.
    let views0 = collision::scene_views(&scene, None, ply_hint.as_deref(), 180)?;
    ensure!(!views0.is_empty(), "no capture cameras");
    let floor_guess = scene["floor"]["y"].as_f64().unwrap_or(0.0);
    let (level_m, mut level_info) = level(&views0, &g, floor_guess, mpu);
    drop(views0);
    if let Some(m) = &level_m {
        crate::to_scene_frame(&mut g, m);
        level_scene(&mut scene, m);
        eprintln!("export-game: levelled by {:.2} deg ({})", level_info["angle_deg"], level_info["method"]);
    }
    level_info["matrix"] = json!(level_m.unwrap_or(DMat4::IDENTITY).to_cols_array());
    let t_read = t0.elapsed().as_secs_f64();
    let capture = viewer::capture_path(&scene);
    let path: Vec<Vec3> = capture.iter().filter_map(|c| viewer::v3(&c["position"]))
        .map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)).collect();

    // Voxel edge: as asked, else the smallest of AUTO_CM within the voxel budget.
    let sizes: Vec<f64> = a.voxel_cm.map_or(AUTO_CM.to_vec(), |c| vec![c]);
    let mut tried = Vec::new();
    let mut chosen = None;
    for (k, &cm) in sizes.iter().enumerate() {
        let b = build_at(&g, &scene, ply_hint.as_deref(), level_m.as_ref(), &path, cm, mpu)?;
        let fits = b.count <= AUTO_MAX_VOXELS && js_size(b.bin.len() as u64) <= AUTO_MAX_VOXEL_BYTES;
        tried.push(json!({"cm": cm, "voxels": b.count, "binary_bytes": b.bin.len(), "fits": fits}));
        if fits || k + 1 == sizes.len() { chosen = Some(b); break; }
    }
    let Built { cm, depth_h, close_r, views, max_depth, timings, floor_y, cams, clip, mut world, stats: world_stats, carve, .. } =
        chosen.context("no voxel world")?;
    let image_colour = colour_from_images(&mut world, &scene, &g, level_m.as_ref(), ply_hint.as_deref(), mpu).unwrap_or_else(|e| {
        eprintln!("export-game: voxel colours from the splats only ({e:#})");
        json!({"error": format!("{e:#}")})
    });
    eprintln!("export-game: image colours {image_colour}");
    drop(g);
    let before = world.solid.clone();
    let reach0 = room_space(&world, floor_y, mpu);
    let mut walls = flatten_walls(&mut world, mpu, floor_y);
    eprintln!("export-game: walls {}", walls["summary"]);
    let planes = flatten_planes(&mut world, mpu, image_colour.get("error").is_none(), floor_y);
    eprintln!("export-game: planes {}", planes["summary"]);
    walls["watertight"] = seal(&mut world, &before, &reach0, floor_y, mpu);
    eprintln!("export-game: watertight {}", walls["watertight"]);
    drop((before, reach0));
    let (bin, chunks, count) = encode(&world);
    let cam = capture.first().cloned().or_else(|| scene["start_camera"].is_object().then(|| scene["start_camera"].clone()))
        .context("scene.json: no capture_cameras or start_camera for the spawn")?;
    let spawn = spawn(&world, &capture, &cam, floor_y, mpu);
    eprintln!("export-game: {cm} cm world; spawn {}", spawn["position"]);

    // Build next to the destination, then swap it in (as export-viewer).
    let name = out.file_name().context("--out has no folder name")?.to_string_lossy().into_owned();
    let parent = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if tmp.exists() {
        fs::remove_dir_all(&tmp)?;
    }
    let data = tmp.join("data");
    fs::create_dir_all(&data)?;
    let part_bytes = ((a.part_mb * 1e6) as usize).max(1);
    let (vparts, vjs) = viewer::write_parts(&bin, &data, "voxels", part_bytes)?;
    let spark = format!("window.__SPARK_SRC={};\n", serde_json::to_string(&viewer::spark_source()?)?);
    fs::write(data.join("spark.js"), &spark)?;

    // Splat mode: the interactive model (the package's) and, when the run has a separate one, the viewer's full
    // model for still frames (motion LOD in the page). Both levelled, minus the floaters at the capture path and
    // near the cameras that occupancy does not support; far background without view-dependent colour.
    let supported = |p: Vec3| -> bool {
        let c = ((p - world.origin) / world.edge).floor();
        let r = SUPPORT_CELLS;
        (-r..=r).any(|dz: i64| (-r..=r).any(|dy: i64| (-r..=r).any(|dx: i64| {
            let (x, y, z) = (c.x as i64 + dx, c.y as i64 + dy, c.z as i64 + dz);
            x >= 0 && y >= 0 && z >= 0 && (x as usize) < world.dims[0] && (y as usize) < world.dims[1] && (z as usize) < world.dims[2]
                && world.solid[world.at(x as usize, y as usize, z as usize)]
        })))
    };
    let (near2, big) = (((FLOATER_NEAR_M / mpu) as f32).powi(2), ((FLOATER_SCALE_M / mpu) as f32).ln());
    let sp = viewer::v3(&spawn["position"]).unwrap_or([0.0; 3]);
    let max_logit = (FAR_ALPHA / (1.0 - FAR_ALPHA)).ln();
    let prepare = |mut ps: crate::ply::Gaussians, key: &str| -> Result<(usize, u64, Value, Value)> {
        if let Some(m) = &level_m { crate::to_scene_frame(&mut ps, m); }
        let total = ps.len();
        // 0 keep, 1 dropped in the carved path, 2 dropped near a camera.
        let class: Vec<u8> = (0..ps.len()).into_par_iter().map(|i| {
            let p = Vec3::from(ps.pos[i]);
            let weak = ps.alpha(i) < FLOATER_ALPHA || ps.scale[i].iter().copied().fold(f32::MIN, f32::max) > big;
            if !weak { return 0; }
            let in_carve = carve.contains(p);
            if !in_carve && !cams.iter().any(|c| c.distance_squared(p) <= near2) { return 0; }
            if supported(p) { 0 } else if in_carve { 1 } else { 2 }
        }).collect();
        let keep: Vec<bool> = class.iter().map(|&c| c == 0).collect();
        let k = ps.sh_coeffs() * 3;
        let mut flattened = 0;
        for i in 0..ps.len() {
            let p = Vec3::from(ps.pos[i]);
            if !keep[i] || !(p.cmplt(clip.0).any() || p.cmpgt(clip.1).any()) { continue; }
            ps.rest[i * k..(i + 1) * k].fill(0.0);
            ps.opacity[i] = ps.opacity[i].min(max_logit);
            flattened += 1;
        }
        let count_class = |c: u8, near: bool| ps.pos.iter().zip(&class).filter(|(p, k)| **k == c
            && (!near || (((p[0] as f64 - sp[0]).powi(2) + (p[2] as f64 - sp[2]).powi(2)).sqrt() * mpu) <= 1.5)).count();
        let dropped = json!({"carve": count_class(1, false), "near_camera": count_class(2, false),
            "carve_within_1_5m_of_spawn": count_class(1, true), "near_camera_within_1_5m_of_spawn": count_class(2, true)});
        ps.retain(&keep);
        let n_dropped = total - ps.len();
        let spz_path = tmp.join(format!("{key}.game.spz"));
        crate::spz::write_spz(&ps, &spz_path)?;
        drop(ps);
        let spz = fs::read(&spz_path)?;
        let sh = viewer::spz_header(&spz_path)?;
        fs::remove_file(&spz_path)?;
        let (parts, js) = viewer::write_parts(&spz, &data, key, part_bytes)?;
        eprintln!("export-game: {key}: {total} splats, {n_dropped} floaters dropped, {flattened} far ones flattened");
        Ok((parts, spz.len() as u64, sh.clone(), json!({"model_splats": total, "dropped_splats": n_dropped, "dropped": dropped,
            "outside_room_flattened": flattened, "header": sh, "parts": parts, "bytes": spz.len(), "js_bytes": js})))
    };
    let (sparts, sbytes, sh, sstats) = prepare(crate::spz::read_package_splats(&pkg)?, "splat")?;
    let full_src = display_full(a.full.as_deref(), &run, &main_spz)?;
    let full = match &full_src {
        Some(f) => {
            let dir = f.parent().unwrap_or(Path::new("."));
            let g = if dir.join("scene.json").is_file() { crate::spz::read_package_splats(dir)? } else { crate::spz::read_spz(f)? };
            Some(prepare(g, "splatfull")?)
        }
        None => None,
    };
    drop(carve);
    let mip = scene["render_mode"].as_str().map_or(scene["stats"]["antialiased"].as_bool() == Some(true), |m| m == "mip");
    let r4 = |v: f32| (v as f64 * 1e5).round() / 1e5;
    let mpu_note = "scene.json meters_per_unit (the package frame is metric: y_up_meters, scaled by scale_estimate)";
    let cm_actual = (world.edge as f64 * mpu * 1e3).round() / 10.0;
    let cfg = json!({"title": title, "up": scene.get("up").cloned().unwrap_or(json!([0, 1, 0])),
        "metersPerUnit": mpu, "eyeHeightM": EYE_M,
        "level": {"matrix": level_info["matrix"], "pivot": level_info["pivot"], "angle_deg": level_info["angle_deg"]},
        "voxel": {"edge": r4(world.edge), "cm": cm_actual,
            "origin": world.origin.to_array().map(r4), "dims": world.dims, "chunk": CHUNK, "count": count,
            "key": "voxels", "parts": vparts, "bytes": bin.len()},
        "spawn": spawn,
        "splat": {"key": "splat", "parts": sparts, "bytes": sbytes, "mip": mip, "splats": sh["splats"], "sh_degree": sh["sh_degree"]},
        "splatFull": full.as_ref().map(|(parts, bytes, h, _)| json!({"key": "splatfull", "parts": parts, "bytes": bytes,
            "splats": h["splats"], "sh_degree": h["sh_degree"]})),
        "info": {"chunks": chunks, "scale_estimate": scene["scale_estimate"], "floor": scene["floor"],
            "collision_floor": world_stats["collision_floor"], "splat_source": splat_source}});
    let html = PAGE.replace("__TITLE__", &viewer::html_escape(&title))
        .replace("__CFG__", &serde_json::to_string(&cfg)?.replace("</", "<\\/"))
        .replace("__THREE__", viewer::THREE).replace("__THREE_ADDONS__", viewer::THREE_ADDONS);
    fs::write(tmp.join("index.html"), html)?;
    let mut build = json!({"generator": format!("vauss export-game {}", env!("CARGO_PKG_VERSION")), "source": pkg,
        "voxel_splat_source": splat_source, "spark": format!("{} (patched)", viewer::SPARK_VERSION),
        "meters_per_unit": mpu, "meters_per_unit_source": mpu_note, "level": level_info,
        "voxel": {"requested_cm": a.voxel_cm, "chosen_cm": cm, "auto": {"tried": tried, "max_voxels": AUTO_MAX_VOXELS,
                "max_voxel_js_bytes": AUTO_MAX_VOXEL_BYTES}, "edge_units": world.edge, "cm": cm_actual,
            "origin": world.origin.to_array(), "dims": world.dims, "count": count, "chunks": chunks,
            "binary_bytes": bin.len(), "parts": vparts, "js_bytes": vjs, "depth_height_px": depth_h,
            "closing_voxels": close_r, "max_depth_units": max_depth, "views": views, "world": world_stats,
            "image_colour": image_colour, "walls": walls, "planes": planes},
        "splat": {"file": main_spz, "stats": sstats},
        "splat_full": full_src.as_ref().map(|f| json!({"file": f, "stats": full.as_ref().map(|x| x.3.clone())})),
        "spawn": spawn,
        "timings_s": {"read": t_read, "depth": timings[0], "fuse": timings[1], "voxelize": timings[2], "world": timings[3]}});
    fs::write(tmp.join("build.json"), serde_json::to_string_pretty(&build)?)?;

    if out.exists() {
        fs::remove_dir_all(&out).with_context(|| format!("replacing {}", out.display()))?;
    }
    fs::rename(&tmp, &out)?;
    let bytes: u64 = walk_bytes(&out);
    build["bytes"] = json!(bytes);
    build["timings_s"]["total"] = json!(t0.elapsed().as_secs_f64());
    fs::write(out.join("build.json"), serde_json::to_string_pretty(&build)?)?;
    Ok(json!({"out": out, "index": out.join("index.html"), "title": title, "voxels": count, "chunks": chunks,
        "voxel_cm": cm_actual, "dims": world.dims, "spawn": build["spawn"], "splats": sh["splats"],
        "full_splats": full.as_ref().map(|x| x.2["splats"].clone()), "bytes": bytes,
        "elapsed_s": t0.elapsed().as_secs_f64()}))
}

fn walk_bytes(dir: &Path) -> u64 {
    fs::read_dir(dir).into_iter().flatten().flatten().map(|e| match e.metadata() {
        Ok(m) if m.is_dir() => walk_bytes(&e.path()),
        Ok(m) => m.len(),
        Err(_) => 0,
    }).sum()
}

/// The game world's occupancy only (export-viewer's camera collision): `(origin, edge, dims, solid, cells)` for
/// the package `pkg` at `voxel_cm`, built as [`run`] builds it (depth fusion, crop, thickening, capture-path
/// carve, floor slab); the colours are dropped. Solid cell (x, y, z) is `solid[(z * dims[1] + y) * dims[0] + x]`.
pub(crate) fn occupancy(pkg: &Path, scene: &Value, voxel_cm: f64) -> Result<(Vec3, f32, [usize; 3], Vec<bool>, usize)> {
    let mpu = scene["meters_per_unit"].as_f64().filter(|m| m.is_finite() && *m > 0.0).unwrap_or(1.0);
    let (g, _) = collision::package_splats(pkg, scene)?;
    let ply_hint = scene["source"]["ply"].as_str().map(PathBuf::from);
    let path: Vec<Vec3> = viewer::capture_path(scene).iter().filter_map(|c| viewer::v3(&c["position"]))
        .map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)).collect();
    // The package frame (not levelled): the viewer renders the package as is.
    let w = build_at(&g, scene, ply_hint.as_deref(), None, &path, voxel_cm, mpu)?.world;
    let cells = w.solid.iter().filter(|s| **s).count();
    Ok((w.origin, w.edge, w.dims, w.solid, cells))
}
