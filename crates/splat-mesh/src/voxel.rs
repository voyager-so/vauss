//! Coloured surface voxels from a (textured) mesh, for the Blender pack: every triangle is sampled
//! densely at <= half a voxel, each occupied cell takes the mean sampled colour (the base-colour
//! texture at the sample's UV, else the interpolated vertex colour), one-cell holes with >= 4 face
//! neighbours are filled, and only exposed cube faces are written (flat quads, vertex colours).
//! Surface sheets, not a solid fill: the interior of furniture and walls stays empty.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use glam::{IVec3, Mat4, Vec2, Vec3};
use hashbrown::HashMap;
use rayon::prelude::*;
use serde_json::json;

/// Triangles with per-corner UVs and colours, plus the base-colour texture when the mesh has one.
struct Surface {
    tris: Vec<[Vec3; 3]>,
    uvs: Vec<[Vec2; 3]>,
    colors: Vec<[Vec3; 3]>,
    texture: Option<image::RgbImage>,
}

#[derive(Debug, Clone)]
pub struct VoxelStats {
    pub edge: f32,
    pub source_triangles: usize,
    pub surface_samples: u64,
    pub occupied_cells: usize,
    pub one_cell_holes_filled: usize,
    pub exposed_faces: usize,
    pub triangles: usize,
    pub bytes: u64,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub elapsed_s: f64,
}

impl VoxelStats {
    pub fn to_json(&self) -> serde_json::Value {
        json!({"edge_units": self.edge, "source_triangles": self.source_triangles, "surface_samples": self.surface_samples,
            "occupied_cells": self.occupied_cells, "one_cell_holes_filled": self.one_cell_holes_filled,
            "exposed_faces": self.exposed_faces, "triangles": self.triangles, "bytes": self.bytes,
            "bounds_min": self.bounds_min, "bounds_max": self.bounds_max, "elapsed_s": (self.elapsed_s * 100.0).round() / 100.0})
    }
}

/// Bounds (min, max) of a GLB's triangles, for sizing the voxel grid before voxelizing.
pub fn glb_bounds(path: &Path) -> Result<(Vec3, Vec3)> {
    let s = read_glb(path)?;
    Ok(bounds(&s.tris))
}

fn bounds(tris: &[[Vec3; 3]]) -> (Vec3, Vec3) {
    tris.iter().flatten().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)))
}

/// Voxelize the GLB at `src` with cubes of edge `edge` (mesh units) and write them to `dst` as GLB,
/// in the mesh's own frame and on the global grid `floor(p / edge)`.
pub fn voxelize_glb(src: &Path, dst: &Path, edge: f32) -> Result<VoxelStats> {
    ensure!(edge.is_finite() && edge > 0.0, "voxel edge must be positive");
    let t0 = std::time::Instant::now();
    let s = read_glb(src)?;
    ensure!(!s.tris.is_empty(), "{}: no triangles", src.display());
    let step = edge * 0.5;
    let inv = 1.0 / edge;
    let tex = s.texture.as_ref();
    // Per-cell sums (sRGB 0..1) and sample counts; each rayon split gets its own map.
    let (cells, samples) = (0..s.tris.len())
        .into_par_iter()
        .with_min_len(4096)
        .fold(
            || (HashMap::<IVec3, [f32; 4]>::new(), 0u64),
            |(mut map, mut n), i| {
                let [a, b, c] = s.tris[i];
                let longest = (b - a).length().max((c - b).length()).max((a - c).length());
                let k = ((longest / step).ceil() as u32).max(1);
                let kf = k as f32;
                for u in 0..=k {
                    for v in 0..=(k - u) {
                        let (bu, bv) = (u as f32 / kf, v as f32 / kf);
                        let bw = 1.0 - bu - bv;
                        let p = a * bw + b * bu + c * bv;
                        let col = match tex {
                            Some(t) => {
                                let [ta, tb, tc] = s.uvs[i];
                                sample(t, ta * bw + tb * bu + tc * bv)
                            }
                            None => {
                                let [ca, cb, cc] = s.colors[i];
                                ca * bw + cb * bu + cc * bv
                            }
                        };
                        let cell = (p * inv).floor().as_ivec3();
                        let e = map.entry(cell).or_insert([0.0; 4]);
                        e[0] += col.x;
                        e[1] += col.y;
                        e[2] += col.z;
                        e[3] += 1.0;
                        n += 1;
                    }
                }
                (map, n)
            },
        )
        .reduce(
            || (HashMap::new(), 0),
            |(mut a, na), (b, nb)| {
                let (mut big, small) = if a.len() >= b.len() { (std::mem::take(&mut a), b) } else { (b, a) };
                for (k, v) in small {
                    let e = big.entry(k).or_insert([0.0; 4]);
                    for j in 0..4 {
                        e[j] += v[j];
                    }
                }
                (big, na + nb)
            },
        );
    let mut cells: HashMap<IVec3, Vec3> = cells.into_iter().map(|(k, v)| (k, Vec3::new(v[0], v[1], v[2]) / v[3])).collect();
    // One-cell holes: an empty cell with >= 4 occupied face neighbours takes their mean colour.
    // Larger uncaptured gaps stay open.
    let candidates: Vec<IVec3> = {
        let mut seen = hashbrown::HashSet::new();
        for c in cells.keys() {
            for d in DIRS {
                let n = *c + d;
                if !cells.contains_key(&n) {
                    seen.insert(n);
                }
            }
        }
        seen.into_iter().collect()
    };
    let fills: Vec<(IVec3, Vec3)> = candidates
        .par_iter()
        .filter_map(|c| {
            let (mut n, mut sum) = (0, Vec3::ZERO);
            for d in DIRS {
                if let Some(col) = cells.get(&(*c + d)) {
                    n += 1;
                    sum += *col;
                }
            }
            (n >= 4).then(|| (*c, sum / n as f32))
        })
        .collect();
    let filled = fills.len();
    cells.extend(fills);
    let occupied = cells.len();
    // Exposed faces: quads wound outward, four vertices each (flat colour per cube).
    let mut list: Vec<(IVec3, Vec3)> = cells.iter().map(|(k, v)| (*k, *v)).collect();
    list.par_sort_unstable_by_key(|(k, _)| (k.x, k.y, k.z));
    let faces: Vec<(IVec3, usize, [u8; 3])> = list
        .par_iter()
        .flat_map_iter(|(c, col)| {
            let rgb = srgb_to_linear_u8(*col);
            (0..6).filter(|&f| !cells.contains_key(&(*c + DIRS[f]))).map(move |f| (*c, f, rgb)).collect::<Vec<_>>()
        })
        .collect();
    let (bytes, lo, hi) = write_cubes(dst, &faces, edge)?;
    Ok(VoxelStats {
        edge,
        source_triangles: s.tris.len(),
        surface_samples: samples,
        occupied_cells: occupied,
        one_cell_holes_filled: filled,
        exposed_faces: faces.len(),
        triangles: faces.len() * 2,
        bytes,
        bounds_min: lo.to_array(),
        bounds_max: hi.to_array(),
        elapsed_s: t0.elapsed().as_secs_f64(),
    })
}

const DIRS: [IVec3; 6] = [IVec3::NEG_X, IVec3::X, IVec3::NEG_Y, IVec3::Y, IVec3::NEG_Z, IVec3::Z];
/// Unit-cube corners of each face in [`DIRS`] order, counter-clockwise seen from outside.
const QUADS: [[[i32; 3]; 4]; 6] = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
];

fn sample(t: &image::RgbImage, uv: Vec2) -> Vec3 {
    let (w, h) = t.dimensions();
    let x = ((uv.x.clamp(0.0, 1.0) * w as f32) as u32).min(w - 1);
    let y = ((uv.y.clamp(0.0, 1.0) * h as f32) as u32).min(h - 1);
    let p = t.get_pixel(x, y).0;
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0
}

/// glTF vertex colours are linear; the samples are sRGB.
fn srgb_to_linear_u8(c: Vec3) -> [u8; 3] {
    c.to_array().map(|v| {
        let v = v.clamp(0.0, 1.0);
        let l = if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
        (l * 255.0).round() as u8
    })
}

/// Stream the quads into a GLB: POSITION f32, COLOR_0 normalized u8 RGBA, u32 indices; no normals
/// (importers derive flat ones).
fn write_cubes(path: &Path, faces: &[(IVec3, usize, [u8; 3])], edge: f32) -> Result<(u64, Vec3, Vec3)> {
    use std::io::Write;
    let nv = faces.len() * 4;
    let ni = faces.len() * 6;
    let corner = |c: IVec3, f: usize, k: usize| (c + IVec3::from_array(QUADS[f][k])).as_vec3() * edge;
    let (lo, hi) = faces
        .par_iter()
        .map(|(c, f, _)| (corner(*c, *f, 0).min(corner(*c, *f, 2)), corner(*c, *f, 0).max(corner(*c, *f, 2))))
        .reduce(|| (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |a, b| (a.0.min(b.0), a.1.max(b.1)));
    let (len_p, len_c, len_i) = (nv * 12, nv * 4, ni * 4);
    let bin_len = len_p + len_c + len_i; // each a multiple of 4
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "voxels".into());
    let doc = json!({
        "asset": {"version": "2.0", "generator": "splat-mesh voxels"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0, "mode": 4}]}],
        "materials": [{"name": "voxel colour", "doubleSided": true,
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "buffers": [{"byteLength": bin_len}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len_p, "target": 34962},
            {"buffer": 0, "byteOffset": len_p, "byteLength": len_c, "target": 34962},
            {"buffer": 0, "byteOffset": len_p + len_c, "byteLength": len_i, "target": 34963}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": lo.to_array(), "max": hi.to_array()},
            {"bufferView": 1, "componentType": 5121, "normalized": true, "count": nv, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5125, "count": ni, "type": "SCALAR"}
        ]
    });
    let mut js = serde_json::to_vec(&doc)?;
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin_len;
    ensure!(total <= u32::MAX as usize, "voxel GLB over 4 GiB: use a larger voxel");
    let tmp = path.with_extension("glb.tmp");
    let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?);
    f.write_all(b"glTF")?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(total as u32).to_le_bytes())?;
    f.write_all(&(js.len() as u32).to_le_bytes())?;
    f.write_all(b"JSON")?;
    f.write_all(&js)?;
    f.write_all(&(bin_len as u32).to_le_bytes())?;
    f.write_all(b"BIN\0")?;
    // Encode in parallel blocks, write in order.
    for block in faces.chunks(1 << 18) {
        let buf: Vec<u8> = block.par_iter().flat_map_iter(|(c, f, _)| (0..4).flat_map(move |k| corner(*c, *f, k).to_array().map(f32::to_le_bytes)).flatten()).collect();
        f.write_all(&buf)?;
    }
    for block in faces.chunks(1 << 18) {
        let buf: Vec<u8> = block.iter().flat_map(|(_, _, rgb)| std::iter::repeat_n([rgb[0], rgb[1], rgb[2], 255], 4).flatten()).collect();
        f.write_all(&buf)?;
    }
    for (b, block) in faces.chunks(1 << 18).enumerate() {
        let base = (b << 18) as u32 * 4;
        let buf: Vec<u8> = (0..block.len() as u32)
            .flat_map(|q| [0u32, 1, 2, 0, 2, 3].map(|o| (base + q * 4 + o).to_le_bytes()))
            .flatten()
            .collect();
        f.write_all(&buf)?;
    }
    f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok((total as u64, lo, hi))
}

/// Triangles of every triangle primitive (node transforms applied), their UVs / vertex colours and the
/// first base-colour texture.
fn read_glb(path: &Path) -> Result<Surface> {
    let file = gltf::Gltf::open(path).with_context(|| format!("reading {}", path.display()))?;
    let blob = file.blob.clone();
    let buffers: Vec<Vec<u8>> = file
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Bin => blob.clone().context("GLB binary chunk missing"),
            gltf::buffer::Source::Uri(_) => anyhow::bail!("external glTF buffers are not supported"),
        })
        .collect::<Result<_>>()?;
    let mut s = Surface { tris: vec![], uvs: vec![], colors: vec![], texture: None };
    let scene = file.default_scene().or_else(|| file.scenes().next()).context("GLB scene missing")?;
    let mut stack: Vec<(gltf::Node<'_>, Mat4)> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let m = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let reader = prim.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
                let pos: Vec<Vec3> = reader.read_positions().context("GLB positions missing")?.map(|p| m.transform_point3(Vec3::from(p))).collect();
                let n = pos.len();
                let info = prim.material().pbr_metallic_roughness().base_color_texture();
                let uv: Vec<Vec2> = info
                    .as_ref()
                    .and_then(|i| reader.read_tex_coords(i.tex_coord()))
                    .map(|t| t.into_f32().map(Vec2::from).collect())
                    .unwrap_or_else(|| vec![Vec2::ZERO; n]);
                if s.texture.is_none() {
                    if let Some(gltf::image::Source::View { view, .. }) = info.as_ref().map(|i| i.texture().source().source()) {
                        let buf = buffers.get(view.buffer().index()).context("texture buffer missing")?;
                        s.texture = Some(image::load_from_memory(&buf[view.offset()..view.offset() + view.length()])?.to_rgb8());
                    }
                }
                let col: Vec<Vec3> = reader
                    .read_colors(0)
                    .map(|c| {
                        c.into_rgb_f32()
                            .map(|c| Vec3::from(c.map(|v| {
                                let v = v.clamp(0.0, 1.0);
                                if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
                            })))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![Vec3::splat(0.8); n]);
                let ids: Vec<u32> = reader.read_indices().map_or_else(|| (0..n as u32).collect(), |i| i.into_u32().collect());
                ensure!(ids.len() % 3 == 0 && ids.iter().all(|&i| (i as usize) < n), "invalid GLB indices");
                for t in ids.chunks_exact(3) {
                    let [a, b, c] = [t[0], t[1], t[2]].map(|i| i as usize);
                    s.tris.push([pos[a], pos[b], pos[c]]);
                    s.uvs.push([uv[a], uv[b], uv[c]]);
                    s.colors.push([col[a], col[b], col[c]]);
                }
            }
        }
        stack.extend(node.children().map(|c| (c, m)));
    }
    Ok(s)
}
