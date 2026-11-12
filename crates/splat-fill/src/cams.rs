//! Novel cameras over the free floor: every grid spot the walker could have stood on (flood fill from the capture
//! path through floor cells clear of furniture and walls), at several eye heights, looking all around.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::{Context, Result};
use glam::{DMat4, DVec2, DVec3};
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use splat_brush::render::{RenderCamera, Renderer};

use crate::FillArgs;

/// The y-up metric frame the floor lives in: `m` maps SfM units to metres with the floor at y = 0
/// (an export-web `transform_from_source`), `s` is metres per SfM unit.
#[derive(Clone, Copy, Debug)]
pub struct Metric {
    pub m: DMat4,
    pub inv: DMat4,
    pub s: f64,
}

impl Metric {
    /// From a JSON object with `transform_from_source` (column-major 4x4: export-web scene.json or a capture's frame.json).
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let cols: Vec<f64> = v["transform_from_source"].as_array().context("no transform_from_source")?
            .iter().filter_map(serde_json::Value::as_f64).collect();
        anyhow::ensure!(cols.len() == 16, "transform_from_source is not a 4x4");
        let m = DMat4::from_cols_slice(&cols);
        let s = m.x_axis.truncate().length();
        anyhow::ensure!(s > 0.0 && s.is_finite(), "transform_from_source has no scale");
        Ok(Self { m, inv: m.inverse(), s })
    }

    pub fn to_metric(&self, p: DVec3) -> DVec3 {
        self.m.transform_point3(p)
    }
}

/// A novel camera (cams.json row).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NovelCam {
    pub name: String,
    /// Horizontal distance to the nearest capture camera, metres.
    pub dist_m: f64,
    pub height_m: f64,
    pub yaw: f64,
    pub pitch: f64,
    /// Metric frame position and look-at point.
    pub position: [f64; 3],
    pub look_at: [f64; 3],
    pub vfov_deg: f64,
    /// Screen render: covered fraction, fraction of covered pixels nearer than 0.35 m, median depth (m).
    pub cov: f64,
    pub near: f64,
    pub median_depth_m: f64,
    /// Camera-to-world in SfM units, row-major, OpenCV axes.
    pub c2w: [[f64; 4]; 4],
}

/// OpenCV camera at `pos` looking at `look` (both metric, y up), as camera-to-world in SfM units.
pub fn look_c2w(metric: &Metric, pos: DVec3, look: DVec3) -> DMat4 {
    let z = (look - pos).normalize();
    let x = z.cross(DVec3::Y).normalize();
    let y = z.cross(x);
    let m = DMat4::from_cols(x.extend(0.0), y.extend(0.0), z.extend(0.0), pos.extend(1.0));
    let mut c = metric.inv * m;
    c.x_axis = c.x_axis.truncate().normalize().extend(0.0);
    c.y_axis = c.y_axis.truncate().normalize().extend(0.0);
    c.z_axis = c.z_axis.truncate().normalize().extend(0.0);
    c
}

pub fn render_camera(c2w: &DMat4, vfov_deg: f64, width: u32, height: u32) -> RenderCamera {
    RenderCamera::from_c2w(c2w.as_mat4(), vfov_deg.to_radians(), width, height)
}

pub fn rows(m: &DMat4) -> [[f64; 4]; 4] {
    m.transpose().to_cols_array_2d()
}

pub fn from_rows(r: &[[f64; 4]; 4]) -> DMat4 {
    DMat4::from_cols_array_2d(r).transpose()
}

pub fn rows_f32(m: &DMat4) -> [[f32; 4]; 4] {
    let r = rows(m);
    r.map(|row| row.map(|v| v as f32))
}

/// 2D convex hull (monotone chain), counter-clockwise.
fn convex_hull(mut p: Vec<DVec2>) -> Vec<DVec2> {
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: DVec2, a: DVec2, b: DVec2| (a - o).perp_dot(b - o);
    let mut h: Vec<DVec2> = Vec::with_capacity(2 * p.len());
    for pass in 0..2 {
        let start = h.len();
        let it: Box<dyn Iterator<Item = &DVec2>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in it {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.pop();
    }
    h
}

/// Largest signed distance of `q` outside the hull's edge lines (<= 0 inside): scipy's `max(equations @ [q, 1])`.
fn hull_distance(hull: &[DVec2], q: DVec2) -> f64 {
    let n = hull.len();
    if n < 3 {
        return 0.0;
    }
    (0..n).map(|i| {
        let (a, b) = (hull[i], hull[(i + 1) % n]);
        let e = (b - a).normalize();
        // Outward normal of a counter-clockwise edge.
        let normal = DVec2::new(e.y, -e.x);
        normal.dot(q - a)
    }).fold(f64::MIN, f64::max)
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let x = q / 100.0 * (v.len() - 1) as f64;
    let (i, f) = (x.floor() as usize, x.fract());
    if i + 1 < v.len() { v[i] * (1.0 - f) + v[i + 1] * f } else { v[i] }
}

/// Plan the novel cameras. `path` = capture camera centres (SfM units), `splats` = (SfM position, opacity).
pub fn plan(a: &FillArgs, metric: &Metric, path: &[DVec3], splats: &[([f32; 3], f32)], renderer: &Renderer, out: &Path) -> Result<Vec<NovelCam>> {
    let c: Vec<DVec3> = path.iter().map(|p| metric.to_metric(*p)).collect();
    let xm: Vec<DVec3> = splats.iter().filter(|(_, op)| *op > 0.5)
        .map(|(p, _)| metric.to_metric(DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64))).collect();
    anyhow::ensure!(!xm.is_empty() && !c.is_empty(), "no opaque splats or capture cameras to plan from");
    // Occupancy at 5 cm in plan view: opaque splats in the body-collision band (0.2-0.9 m: furniture and walls; rugs,
    // lamp arms and shelves overhead ignored). A cell needs `occ_min` splats so stray floaters don't block it.
    let g = 0.05;
    let room: Vec<&DVec3> = xm.iter().filter(|p| p.y > 0.1 && p.y < 2.3).collect();
    anyhow::ensure!(!room.is_empty(), "no splats between 0.1 and 2.3 m above the floor: is the metric frame right?");
    let mut xs: Vec<f64> = room.iter().map(|p| p.x).collect();
    let mut zs: Vec<f64> = room.iter().map(|p| p.z).collect();
    let lo = DVec2::new(percentile(&mut xs, 1.0), percentile(&mut zs, 1.0)) - 0.2;
    let hi = DVec2::new(percentile(&mut xs, 99.0), percentile(&mut zs, 99.0)) + 0.2;
    let nx = ((hi.x - lo.x) / g) as usize + 1;
    let nz = ((hi.y - lo.y) / g) as usize + 1;
    let cell = |x: f64, z: f64| -> Option<usize> {
        let (i, j) = (((x - lo.x) / g).floor(), ((z - lo.y) / g).floor());
        (i >= 0.0 && j >= 0.0 && (i as usize) < nx && (j as usize) < nz).then(|| i as usize * nz + j as usize)
    };
    let mut occ = vec![0u32; nx * nz];
    for p in xm.iter().filter(|p| p.y > a.band[0] && p.y < a.band[1]) {
        if let Some(k) = cell(p.x, p.z) {
            occ[k] += 1;
        }
    }
    let blocked: Vec<bool> = occ.iter().map(|&n| n >= a.occ_min).collect();
    // Free = no blocked cell within `clearance` (4-neighbour steps, like scipy's iterated cross dilation).
    let steps = (a.clearance / g).round() as u32;
    let mut dist = vec![u32::MAX; nx * nz];
    let mut queue = VecDeque::new();
    for (k, &b) in blocked.iter().enumerate() {
        if b {
            dist[k] = 0;
            queue.push_back(k);
        }
    }
    let neighbours = |k: usize| {
        let (i, j) = (k / nz, k % nz);
        let mut n = [None; 4];
        if i > 0 { n[0] = Some(k - nz); }
        if i + 1 < nx { n[1] = Some(k + nz); }
        if j > 0 { n[2] = Some(k - 1); }
        if j + 1 < nz { n[3] = Some(k + 1); }
        n
    };
    while let Some(k) = queue.pop_front() {
        for n in neighbours(k).into_iter().flatten() {
            if dist[n] == u32::MAX {
                dist[n] = dist[k] + 1;
                queue.push_back(n);
            }
        }
    }
    let free: Vec<bool> = dist.iter().map(|&d| d > steps).collect();
    // Inside = reachable floor: flood fill from the capture camera footprints through free cells.
    let seeds: Vec<usize> = c.iter().map(|p| {
        let i = (((p.x - lo.x) / g) as isize).clamp(0, nx as isize - 1) as usize;
        let j = (((p.z - lo.y) / g) as isize).clamp(0, nz as isize - 1) as usize;
        i * nz + j
    }).collect();
    let mut reach = vec![false; nx * nz];
    let passable = |k: usize, seed: &[bool]| free[k] || seed[k];
    let mut is_seed = vec![false; nx * nz];
    for &k in &seeds {
        is_seed[k] = true;
    }
    for &k in &seeds {
        if reach[k] {
            continue;
        }
        reach[k] = true;
        queue.push_back(k);
        while let Some(k) = queue.pop_front() {
            for n in neighbours(k).into_iter().flatten() {
                if !reach[n] && passable(n, &is_seed) {
                    reach[n] = true;
                    queue.push_back(n);
                }
            }
        }
    }
    // The free-space fill leaks through windows and doors into the far background: keep it within `hull_margin` of
    // the convex hull of the walked path.
    let hull = convex_hull(c.iter().map(|p| DVec2::new(p.x, p.z)).collect());
    let inside: Vec<bool> = (0..nx * nz).map(|k| {
        let q = lo + DVec2::new((k / nz) as f64, (k % nz) as f64) * g;
        reach[k] && free[k] && hull_distance(&hull, q) <= a.hull_margin
    }).collect();
    // Floor spots on the coarse grid, with their distance to the walked path.
    let mut spots = vec![];
    let mut x = lo.x + a.offset[0];
    while x < hi.x {
        let mut z = lo.y + a.offset[1];
        while z < hi.y {
            if cell(x, z).is_some_and(|k| inside[k]) {
                let d = c.iter().map(|p| DVec2::new(p.x - x, p.z - z).length()).fold(f64::MAX, f64::min);
                spots.push((DVec2::new(x, z), d));
            }
            z += a.grid;
        }
        x += a.grid;
    }
    anyhow::ensure!(!spots.is_empty(), "no free floor found next to the capture path");
    // Never train on a pose within 10 cm of an evaluation grid eye (e.g. measure's fixed grid.json).
    let exclude: Vec<DVec3> = match &a.exclude {
        Some(p) => {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p).with_context(|| format!("reading {}", p.display()))?)?;
            v["views"].as_array().context("exclude: no views")?.iter().filter_map(|w| {
                let r: [[f64; 4]; 4] = serde_json::from_value(w["c2w"].clone()).ok()?;
                Some(metric.to_metric(from_rows(&r).w_axis.truncate()))
            }).collect()
        }
        None => vec![],
    };
    let heights = &a.heights;
    let pitches = &a.pitches;
    let (sw, sh) = (160u32, 90u32);
    let mut rng = rand::rngs::StdRng::seed_from_u64(a.seed);
    let mut cams = vec![];
    for (spot, d) in &spots {
        let y0 = rng.random_range(0.0..a.yaw);
        for &hgt in heights {
            let mut yaw = y0;
            while yaw < y0 + 360.0 {
                let pitch = pitches[rng.random_range(0..pitches.len())];
                let (yr, pr) = (yaw.to_radians(), pitch.to_radians());
                let pos = DVec3::new(spot.x, hgt, spot.y);
                yaw += a.yaw;
                if exclude.iter().any(|e| e.distance(pos) < 0.10) {
                    continue;
                }
                let look = pos + DVec3::new(yr.sin() * pr.cos(), pr.sin(), -yr.cos() * pr.cos()) * 2.0;
                let c2w = look_c2w(metric, pos, look);
                let f = renderer.render(&render_camera(&c2w, a.vfov, sw, sh), true)?;
                let covered: Vec<f64> = f.alpha.iter().zip(&f.depth).filter(|(al, _)| **al > 0.5).map(|(_, z)| *z as f64 * metric.s).collect();
                let cov = covered.len() as f64 / f.alpha.len() as f64;
                let near = if covered.is_empty() { 1.0 } else { covered.iter().filter(|z| **z < 0.35).count() as f64 / covered.len() as f64 };
                // Facing a wall at arm's length, or looking out of the model.
                if near > 0.25 || cov < 0.3 {
                    continue;
                }
                let mut sorted = covered;
                let med = percentile(&mut sorted, 50.0);
                cams.push(NovelCam { name: format!("n{:05}", cams.len()), dist_m: *d, height_m: hgt, yaw: (yaw - a.yaw).rem_euclid(360.0), pitch,
                    position: pos.to_array(), look_at: look.to_array(), vfov_deg: a.vfov, cov, near, median_depth_m: med, c2w: rows(&c2w) });
            }
        }
    }
    // Plan-view map for the report: inside grey, blocked red, capture path blue, floor spots yellow.
    let mut img = image::RgbImage::new(nx as u32, nz as u32);
    for k in 0..nx * nz {
        let px = if blocked[k] { [200, 80, 60] } else if inside[k] { [60, 60, 60] } else { [0, 0, 0] };
        img.put_pixel((k / nz) as u32, (k % nz) as u32, image::Rgb(px));
    }
    for (pts, col) in [(c.iter().map(|p| DVec2::new(p.x, p.z)).collect::<Vec<_>>(), [80, 160, 255]), (spots.iter().map(|s| s.0).collect(), [255, 255, 0])] {
        for q in pts {
            if let Some(k) = cell(q.x, q.y) {
                img.put_pixel((k / nz) as u32, (k % nz) as u32, image::Rgb(col));
            }
        }
    }
    image::imageops::resize(&img, nx as u32 * 4, nz as u32 * 4, image::imageops::FilterType::Nearest).save(out.join("floor_map.png"))?;
    let by_dist = |lo: f64, hi: f64| cams.iter().filter(|c| c.dist_m >= lo && c.dist_m < hi).count();
    eprintln!("[fill] cams: {} floor spots (max {:.2} m from the path), {} views kept (0-1 m {}, 1-2 m {}, 2+ m {})",
        spots.len(), spots.iter().map(|s| s.1).fold(0.0, f64::max), cams.len(), by_dist(0.0, 1.0), by_dist(1.0, 2.0), by_dist(2.0, f64::MAX));
    Ok(cams)
}
