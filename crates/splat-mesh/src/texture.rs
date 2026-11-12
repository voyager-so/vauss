//! Texture atlas: every pair of triangles gets its own square cell (two right triangles split
//! along the diagonal, inset so neighbours never bleed), and the colour of each texel is baked
//! from rendered views: project the texel's surface point into each view, keep it where the
//! rendered depth agrees, and weight by how squarely the view sees the surface. Texels no view
//! sees fall back to the interpolated vertex colours.

use anyhow::{Result, bail};
use glam::{Vec2, Vec3};
use image::RgbImage;
use rayon::prelude::*;

use crate::{Camera, Mesh, View, decimate, vertex_normals};

#[derive(Debug, Clone, Copy)]
pub struct TextureOpts {
    /// Target atlas edge in pixels (square). Dense meshes grow it up to `max_size` rather than
    /// lose geometry: cells shrink to `min_cell` first, and only meshes that still don't fit
    /// are decimated (large decimation ratios flatten fine shapes).
    pub size: u32,
    pub max_size: u32,
    /// Textured meshes are unwelded (unique UVs per corner), so the GLB costs ~100 bytes per
    /// triangle; above this count the mesh is decimated first (mild ratios keep the shape).
    pub max_triangles: usize,
    /// Cell edge range in pixels for one triangle pair.
    pub min_cell: u32,
    pub max_cell: u32,
    /// Rendered depth must agree with the texel's depth to within this fraction.
    pub depth_tol: f32,
}

impl Default for TextureOpts {
    fn default() -> Self {
        Self { size: 2048, max_size: 4096, max_triangles: 500_000, min_cell: 4, max_cell: 16, depth_tol: 0.02 }
    }
}

/// An unwelded mesh (three vertices per triangle, in order) with UVs and its baked texture.
pub struct TexturedMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub texture: RgbImage,
    /// Fraction of covered texels that at least one view saw (the rest use vertex colours).
    pub seen_fraction: f32,
}

impl TexturedMesh {
    pub fn triangle_count(&self) -> usize {
        self.positions.len() / 3
    }
}

/// Pixel-space corners of triangle `i` in a `grid` x `grid` atlas of `cell`-pixel cells.
fn corners(i: usize, grid: u32, cell: u32) -> [Vec2; 3] {
    let p = (i / 2) as u32;
    let (x0, y0) = (((p % grid) * cell) as f32, ((p / grid) * cell) as f32);
    let (x1, y1) = (x0 + cell as f32, y0 + cell as f32);
    if i % 2 == 0 {
        [Vec2::new(x0 + 1.0, y0 + 1.0), Vec2::new(x1 - 2.0, y0 + 1.0), Vec2::new(x0 + 1.0, y1 - 2.0)]
    } else {
        [Vec2::new(x1 - 1.0, y1 - 1.0), Vec2::new(x0 + 2.0, y1 - 1.0), Vec2::new(x1 - 1.0, y0 + 2.0)]
    }
}

fn barycentric(p: Vec2, [a, b, c]: [Vec2; 3]) -> Vec3 {
    let (v0, v1, v2) = (b - a, c - a, p - a);
    let d = v0.perp_dot(v1);
    let v = v2.perp_dot(v1) / d;
    let w = v0.perp_dot(v2) / d;
    Vec3::new(1.0 - v - w, v, w)
}

/// Texel `index` (y * size + x), its triangle, and its surface point.
struct Texel {
    index: u32,
    tri: u32,
    pos: Vec3,
    fallback: [f32; 3],
}

/// Decimate `mesh` to fit the atlas, unwrap it, and bake colour from `render(camera)` views.
pub fn bake_texture(
    mesh: &Mesh,
    cameras: &[Camera],
    mut render: impl FnMut(&Camera) -> Result<View>,
    opts: &TextureOpts,
) -> Result<TexturedMesh> {
    if mesh.triangles.is_empty() {
        bail!("cannot texture an empty mesh");
    }
    let min_cell = opts.min_cell.max(3);
    let per_side = (opts.max_size.max(opts.size) / min_cell).max(1);
    let mut mesh = mesh.clone();
    decimate(&mut mesh, opts.max_triangles.min((2 * per_side * per_side) as usize));
    let ntri = mesh.triangles.len();
    let grid = ((ntri as f64 / 2.0).sqrt().ceil() as u32).max(1);
    let cell = (opts.size / grid).clamp(min_cell, opts.max_cell.max(min_cell));
    let size = grid * cell;
    let vn = vertex_normals(&mesh);
    let face_n: Vec<Vec3> = mesh
        .triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
            (b - a).cross(c - a).normalize_or_zero()
        })
        .collect();

    // Rasterise each triangle's cell, one pixel past its edges so bilinear lookups at the
    // borders read baked colour rather than background.
    let texels: Vec<Texel> = (0..ntri)
        .into_par_iter()
        .flat_map_iter(|i| {
            let uv = corners(i, grid, cell);
            let t = mesh.triangles[i].map(|v| v as usize);
            let (p, c) = (t.map(|v| mesh.positions[v]), t.map(|v| mesh.colors[v]));
            let lo = uv.iter().fold(Vec2::splat(f32::MAX), |m, v| m.min(*v)).floor() - 1.0;
            let hi = uv.iter().fold(Vec2::splat(f32::MIN), |m, v| m.max(*v)).ceil() + 1.0;
            let mut out = vec![];
            for y in lo.y.max(0.0) as u32..(hi.y as u32).min(size) {
                for x in lo.x.max(0.0) as u32..(hi.x as u32).min(size) {
                    let b = barycentric(Vec2::new(x as f32 + 0.5, y as f32 + 0.5), uv);
                    if b.min_element() < -0.35 {
                        continue;
                    }
                    let b = b.max(Vec3::ZERO);
                    let b = b / b.element_sum();
                    let pos = p[0] * b.x + p[1] * b.y + p[2] * b.z;
                    let fallback = std::array::from_fn(|k| c[0][k] as f32 * b.x + c[1][k] as f32 * b.y + c[2][k] as f32 * b.z);
                    out.push(Texel { index: y * size + x, tri: i as u32, pos, fallback });
                }
            }
            out.into_iter()
        })
        .collect();

    let mut acc = vec![[0f32; 4]; texels.len()];
    for cam in cameras {
        let view = render(cam)?;
        view.validate()?;
        if view.rgb.is_empty() {
            continue;
        }
        let k = cam.intrinsics;
        let (w, h) = (k.width as usize, k.height as usize);
        let eye = cam.position();
        texels.par_iter().zip(acc.par_iter_mut()).for_each(|(t, a)| {
            let pc = cam.world_to_cam.transform_point3(t.pos);
            if pc.z <= 1e-6 {
                return;
            }
            let (u, v) = (k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy);
            if u < 0.0 || v < 0.0 || u >= w as f32 || v >= h as f32 {
                return;
            }
            let i = v as usize * w + u as usize;
            let d = view.depth[i];
            if !(d > 0.0 && d.is_finite()) || (d - pc.z).abs() > opts.depth_tol * pc.z {
                return;
            }
            let cos = face_n[t.tri as usize].dot((eye - t.pos).normalize_or_zero()).abs();
            let alpha = view.alpha.get(i).copied().unwrap_or(1.0);
            let wgt = cos * cos * cos * alpha;
            if wgt <= 1e-4 {
                return;
            }
            for c in 0..3 {
                a[c] += view.rgb[3 * i + c] as f32 * wgt;
            }
            a[3] += wgt;
        });
    }

    let mut img = RgbImage::new(size, size);
    let mut filled = vec![false; (size * size) as usize];
    let mut seen = 0usize;
    for (t, a) in texels.iter().zip(&acc) {
        let rgb = if a[3] > 0.0 {
            seen += 1;
            [a[0] / a[3], a[1] / a[3], a[2] / a[3]]
        } else {
            t.fallback
        };
        img.put_pixel(t.index % size, t.index / size, image::Rgb(rgb.map(|v| v.round().clamp(0.0, 255.0) as u8)));
        filled[t.index as usize] = true;
    }
    dilate(&mut img, &mut filled, 2);

    let mut positions = Vec::with_capacity(ntri * 3);
    let mut normals = Vec::with_capacity(ntri * 3);
    let mut uvs = Vec::with_capacity(ntri * 3);
    for (i, t) in mesh.triangles.iter().enumerate() {
        for (k, &v) in t.iter().enumerate() {
            positions.push(mesh.positions[v as usize]);
            normals.push(vn[v as usize]);
            let c = corners(i, grid, cell)[k];
            uvs.push([c.x / size as f32, c.y / size as f32]);
        }
    }
    Ok(TexturedMesh { positions, normals, uvs, texture: img, seen_fraction: seen as f32 / texels.len().max(1) as f32 })
}

/// Grow filled texels into empty neighbours `passes` times (keeps mip/bilinear edges clean).
fn dilate(img: &mut RgbImage, filled: &mut [bool], passes: u32) {
    let (w, h) = img.dimensions();
    for _ in 0..passes {
        let snapshot = filled.to_vec();
        for y in 0..h {
            for x in 0..w {
                if snapshot[(y * w + x) as usize] {
                    continue;
                }
                let mut sum = [0u32; 3];
                let mut n = 0;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 || !snapshot[(ny as u32 * w + nx as u32) as usize] {
                        continue;
                    }
                    let p = img.get_pixel(nx as u32, ny as u32);
                    for c in 0..3 {
                        sum[c] += p[c] as u32;
                    }
                    n += 1;
                }
                if n > 0 {
                    img.put_pixel(x, y, image::Rgb(sum.map(|s| (s / n) as u8)));
                    filled[(y * w + x) as usize] = true;
                }
            }
        }
    }
}
