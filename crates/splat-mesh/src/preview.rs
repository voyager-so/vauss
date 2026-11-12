//! A small CPU rasterizer for mesh previews (z-buffered, perspective-correct colours).

use glam::{Vec2, Vec3};
use image::RgbImage;
use rayon::prelude::*;

use crate::{Camera, Mesh, vertex_normals};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shade {
    /// Vertex colours as captured, unlit.
    Color,
    /// Light grey with headlight shading: shows geometry only.
    Clay,
    /// Vertex colours with headlight shading.
    ColorLit,
}

/// Render `mesh` from `camera` (OpenCV axes) on `background`.
pub fn render(mesh: &Mesh, camera: &Camera, shade: Shade, background: [u8; 3]) -> RgbImage {
    let k = camera.intrinsics;
    let (w, h) = (k.width as usize, k.height as usize);
    let normals = vertex_normals(mesh);
    let cam_pts: Vec<Vec3> = mesh.positions.par_iter().map(|&p| camera.world_to_cam.transform_point3(p)).collect();
    let cam_n: Vec<Vec3> = normals.par_iter().map(|&n| camera.world_to_cam.transform_vector3(n)).collect();
    let near = 1e-4;
    // Screen-space triangles with a bounding box, culled to the image.
    struct Tri {
        idx: [u32; 3],
        s: [Vec2; 3],
        inv_z: [f32; 3],
        min: Vec2,
        max: Vec2,
    }
    let tris: Vec<Tri> = mesh
        .triangles
        .par_iter()
        .filter_map(|t| {
            let p = t.map(|i| cam_pts[i as usize]);
            if p.iter().any(|q| q.z <= near) {
                return None;
            }
            let s = p.map(|q| Vec2::new(k.fx * q.x / q.z + k.cx, k.fy * q.y / q.z + k.cy));
            let min = s[0].min(s[1]).min(s[2]);
            let max = s[0].max(s[1]).max(s[2]);
            if max.x < 0.0 || max.y < 0.0 || min.x >= w as f32 || min.y >= h as f32 {
                return None;
            }
            Some(Tri { idx: *t, s, inv_z: p.map(|q| 1.0 / q.z), min, max })
        })
        .collect();
    const BAND: usize = 16;
    let mut img = vec![0u8; w * h * 3];
    img.par_chunks_mut(w * 3 * BAND).enumerate().for_each(|(band, rows)| {
        let y0 = band * BAND;
        let nrows = rows.len() / (w * 3);
        let mut zbuf = vec![0f32; w * nrows]; // stores 1/z; larger is nearer
        for px in rows.chunks_exact_mut(3) {
            px.copy_from_slice(&background);
        }
        for t in &tris {
            if t.max.y < y0 as f32 || t.min.y >= (y0 + nrows) as f32 {
                continue;
            }
            let [a, b, c] = t.s;
            let area = (b - a).perp_dot(c - a);
            let tiny = t.max.x - t.min.x < 1.0 && t.max.y - t.min.y < 1.0;
            if tiny || area.abs() < 1e-12 {
                // Sub-pixel (or degenerate) triangle: may cover no pixel centre, so draw it as a
                // one-pixel point at its centroid, else dense far-away meshes render as speckle.
                let p = (a + b + c) / 3.0;
                let (x, y) = (p.x.floor(), p.y.floor());
                if x < 0.0 || x >= w as f32 || y < y0 as f32 || y >= (y0 + nrows) as f32 {
                    continue;
                }
                let zi = (y as usize - y0) * w + x as usize;
                let iz = (t.inv_z[0] + t.inv_z[1] + t.inv_z[2]) / 3.0;
                if iz <= zbuf[zi] {
                    continue;
                }
                zbuf[zi] = iz;
                let out = shade_px(mesh, &cam_n, &t.idx, [1.0 / 3.0; 3], p, &k, shade);
                rows[zi * 3..zi * 3 + 3].copy_from_slice(&out);
                continue;
            }
            let xs = t.min.x.max(0.0) as usize;
            let xe = (t.max.x.ceil() as usize).min(w);
            let ys = (t.min.y.max(y0 as f32) as usize).max(y0);
            let ye = (t.max.y.ceil() as usize).min(y0 + nrows);
            for y in ys..ye {
                for x in xs..xe {
                    let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = (b - p).perp_dot(c - p) / area;
                    let w1 = (c - p).perp_dot(a - p) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let iz = w0 * t.inv_z[0] + w1 * t.inv_z[1] + w2 * t.inv_z[2];
                    let zi = (y - y0) * w + x;
                    if iz <= zbuf[zi] {
                        continue;
                    }
                    zbuf[zi] = iz;
                    // Perspective-correct weights.
                    let pw = [w0 * t.inv_z[0] / iz, w1 * t.inv_z[1] / iz, w2 * t.inv_z[2] / iz];
                    let out = shade_px(mesh, &cam_n, &t.idx, pw, p, &k, shade);
                    let o = zi * 3;
                    rows[o..o + 3].copy_from_slice(&out);
                }
            }
        }
    });
    RgbImage::from_raw(w as u32, h as u32, img).expect("sized buffer")
}

#[inline]
fn shade_px(mesh: &Mesh, cam_n: &[Vec3], idx: &[u32; 3], pw: [f32; 3], p: Vec2, k: &crate::Intrinsics, shade: Shade) -> [u8; 3] {
    let mut col = Vec3::ZERO;
    let mut n = Vec3::ZERO;
    for (j, &vi) in idx.iter().enumerate() {
        let c = mesh.colors[vi as usize];
        col += Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) * pw[j];
        n += cam_n[vi as usize] * pw[j];
    }
    let view = (-(Vec3::new((p.x - k.cx) / k.fx, (p.y - k.cy) / k.fy, 1.0))).normalize();
    let n = n.normalize_or_zero();
    // Headlight plus a light from above-left (camera space: -y is up) so flat areas read.
    let key = Vec3::new(-0.4, -0.7, -0.6).normalize();
    let lit = 0.18 + 0.55 * n.dot(view).abs() + 0.27 * n.dot(key).abs();
    let out = match shade {
        Shade::Color => col,
        Shade::Clay => Vec3::new(120.0, 160.0, 190.0) * lit * 1.15,
        Shade::ColorLit => col * lit,
    };
    out.to_array().map(|v| v.round().clamp(0.0, 255.0) as u8)
}

/// Where to point preview cameras: the subject, not the whole (possibly huge) scene.
#[derive(Debug, Clone, Copy)]
pub struct Focus {
    pub center: Vec3,
    pub radius: f32,
    pub up: Vec3,
}

fn percentile(v: &mut [f32], q: f32) -> f32 {
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f32 * q) as usize]
}

/// Frame the subject. With capture cameras: their look-at point, and the 95th-percentile
/// distance of the vertices within the median camera distance of it; up = minus the mean
/// image-down axis. Without: a trimmed median centre and 90th-percentile radius of the vertices
/// near it (so far-away ground or background does not shrink the subject to a speck); up =
/// `up` if given, else the least-variance axis of those vertices (the ground normal), signed
/// toward the side with more mass.
pub fn focus(mesh: &Mesh, cameras: &[Camera], up: Option<Vec3>) -> Focus {
    let pts = &mesh.positions;
    let stride = (pts.len() / 200_000).max(1);
    let sample: Vec<Vec3> = pts.iter().step_by(stride).copied().collect();
    let median_point = |v: &[Vec3]| Vec3::from_array(std::array::from_fn(|a| percentile(&mut v.iter().map(|p| p[a]).collect::<Vec<_>>(), 0.5)));
    let (center, near): (Vec3, Vec<Vec3>) = if cameras.is_empty() {
        let c0 = median_point(&sample);
        let r50 = percentile(&mut sample.iter().map(|p| p.distance(c0)).collect::<Vec<_>>(), 0.5);
        let near: Vec<Vec3> = sample.iter().copied().filter(|p| p.distance(c0) <= 2.0 * r50).collect();
        let near = if near.len() >= 16 { near } else { sample.clone() };
        (median_point(&near), near)
    } else {
        let c = crate::look_target(cameras);
        let r = percentile(&mut cameras.iter().map(|cam| cam.position().distance(c)).collect::<Vec<_>>(), 0.5);
        let near: Vec<Vec3> = sample.iter().copied().filter(|p| p.distance(c) <= r).collect();
        (c, if near.len() >= 16 { near } else { sample.clone() })
    };
    let q = if cameras.is_empty() { 0.9 } else { 0.95 };
    let diag = mesh.bounds().map_or(1.0, |b| b.extent().length());
    let radius = percentile(&mut near.iter().map(|p| p.distance(center)).collect::<Vec<_>>(), q).max(0.02 * diag).max(1e-6);
    let up = up.map(Vec3::normalize).unwrap_or_else(|| {
        if !cameras.is_empty() {
            return -cameras.iter().map(|c| c.cam_to_world().y_axis.truncate()).sum::<Vec3>().normalize_or(Vec3::Z);
        }
        let mean = near.iter().sum::<Vec3>() / near.len() as f32;
        let mut cov = glam::Mat3::ZERO;
        for p in &near {
            let d = *p - mean;
            cov += glam::Mat3::from_cols(d * d.x, d * d.y, d * d.z);
        }
        let n = smallest_eigenvector(cov);
        let med = median_point(&near);
        if (mean - med).dot(n) >= 0.0 { n } else { -n }
    });
    Focus { center, radius, up }
}

/// Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (inverse power iteration).
fn smallest_eigenvector(m: glam::Mat3) -> Vec3 {
    let shift = m + glam::Mat3::IDENTITY * (m.x_axis.x + m.y_axis.y + m.z_axis.z) * 1e-6 + glam::Mat3::IDENTITY * 1e-12;
    let inv = shift.inverse();
    let mut v = Vec3::new(0.3, 0.5, 0.8).normalize();
    for _ in 0..50 {
        let next = (inv * v).normalize_or_zero();
        if !next.is_finite() || next == Vec3::ZERO {
            return Vec3::Z;
        }
        v = next;
    }
    v
}

/// `n` cameras on a ring around the focus, 25° above the horizon, 40° vertical field of view,
/// far enough back that the focus sphere fills the frame.
pub fn orbit_cameras(f: &Focus, n: usize, width: u32, height: u32) -> Vec<Camera> {
    let fov = 40f32.to_radians();
    let dist = f.radius / (fov * 0.5).sin() * 1.05;
    let side = f.up.any_orthonormal_vector();
    let fwd0 = f.up.cross(side);
    let elev = 25f32.to_radians();
    (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU + 0.6;
            let dir = (side * a.cos() + fwd0 * a.sin()) * elev.cos() + f.up * elev.sin();
            let eye = f.center + dir * dist;
            let fwd = (f.center - eye).normalize();
            let right = fwd.cross(f.up).normalize();
            let down = fwd.cross(right);
            let c2w = glam::Mat4::from_cols(right.extend(0.0), down.extend(0.0), fwd.extend(0.0), eye.extend(1.0));
            let fy = height as f32 * 0.5 / (fov * 0.5).tan();
            let k = crate::Intrinsics { fx: fy, fy, cx: width as f32 * 0.5, cy: height as f32 * 0.5, width, height };
            Camera { intrinsics: k, world_to_cam: c2w.inverse() }
        })
        .collect()
}

/// The standard preview set: `n` orbit views of the subject, each `size`² — colour and clay
/// side by side when the mesh has colour, else clay only. Used by every engine's mesh tool.
/// An empty mesh has nothing to frame: no previews.
pub fn orbit_previews(mesh: &Mesh, has_color: bool, cameras: &[Camera], up: Option<Vec3>, n: usize, size: u32) -> Vec<RgbImage> {
    if mesh.positions.is_empty() {
        return Vec::new();
    }
    let f = focus(mesh, cameras, up);
    let bg = [240, 242, 244];
    orbit_cameras(&f, n, size, size)
        .iter()
        .map(|cam| {
            let clay = render(mesh, cam, Shade::Clay, bg);
            if !has_color {
                return clay;
            }
            let color = render(mesh, cam, Shade::ColorLit, bg);
            let mut sheet = RgbImage::new(size * 2, size);
            image::imageops::replace(&mut sheet, &color, 0, 0);
            image::imageops::replace(&mut sheet, &clay, size as i64, 0);
            sheet
        })
        .collect()
}

/// Load a mesh file and write `geometry-{i}.png`-style previews via `save` (callers choose the
/// encoding). Returns the images in view order.
pub fn previews_for_file(path: &std::path::Path, cameras: &[Camera], up: Option<Vec3>, n: usize, size: u32) -> anyhow::Result<Vec<RgbImage>> {
    let loaded = crate::load_mesh(path)?;
    Ok(orbit_previews(&loaded.mesh, loaded.has_color, cameras, up, n, size))
}
