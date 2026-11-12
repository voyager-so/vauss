//! Mesh readers for previews: GLB (vertex colours, or the base-colour texture sampled at each
//! vertex's UV), OBJ and PLY (ASCII / binary, optional red/green/blue).

use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use glam::{Mat4, Vec3};

use crate::Mesh;

const GREY: [u8; 3] = [200, 200, 200];

/// Load a `.glb`, `.obj` or `.ply` mesh. Missing colours become grey; see [`Loaded::has_color`].
pub fn load_mesh(path: &Path) -> Result<Loaded> {
    let loaded = match path.extension().and_then(|s| s.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("glb" | "gltf") => glb(path)?,
        Some("obj") => obj(path)?,
        Some("ply") => ply(path)?,
        _ => bail!("unsupported mesh format: {}", path.display()),
    };
    let m = &loaded.mesh;
    ensure!(!m.positions.is_empty() && !m.triangles.is_empty(), "mesh has no renderable triangles");
    ensure!(m.positions.iter().all(|p| p.is_finite()), "mesh has non-finite coordinates");
    ensure!(m.triangles.iter().flatten().all(|&i| (i as usize) < m.positions.len()), "mesh face index exceeds vertex count");
    Ok(loaded)
}

pub struct Loaded {
    pub mesh: Mesh,
    pub has_color: bool,
}

fn polygon(tris: &mut Vec<[u32; 3]>, ids: &[u32]) {
    for i in 1..ids.len().saturating_sub(1) {
        tris.push([ids[0], ids[i], ids[i + 1]]);
    }
}

fn glb(path: &Path) -> Result<Loaded> {
    let file = gltf::Gltf::open(path)?;
    let mut buffers = vec![];
    for buffer in file.buffers() {
        buffers.push(match buffer.source() {
            gltf::buffer::Source::Bin => file.blob.clone().context("GLB binary chunk missing")?,
            gltf::buffer::Source::Uri(uri) => {
                ensure!(!uri.contains("..") && !uri.contains(':') && !uri.starts_with('/'), "unsafe buffer uri {uri}");
                std::fs::read(path.parent().unwrap_or(Path::new(".")).join(uri))?
            }
        });
    }
    let mut out = Loaded { mesh: Mesh::default(), has_color: false };
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
                let base = out.mesh.positions.len() as u32;
                let positions: Vec<Vec3> = reader.read_positions().context("GLB positions missing")?.map(|p| m.transform_point3(Vec3::from(p))).collect();
                let n = positions.len();
                out.mesh.positions.extend(positions);
                let colors: Option<Vec<[u8; 3]>> = if let Some(c) = reader.read_colors(0) {
                    // Stored linear; previews show sRGB.
                    Some(c.into_rgb_f32().map(|c| c.map(|v| (linear_to_srgb(v) * 255.0).round() as u8)).collect())
                } else {
                    texture_colors(&prim, &reader, &buffers)
                };
                out.has_color |= colors.is_some();
                out.mesh.colors.extend(colors.unwrap_or_else(|| vec![GREY; n]));
                let ids: Vec<u32> = reader.read_indices().map_or_else(|| (0..n as u32).collect(), |i| i.into_u32().collect());
                ensure!(ids.len() % 3 == 0, "incomplete GLB triangles");
                out.mesh.triangles.extend(ids.chunks_exact(3).map(|t| [t[0] + base, t[1] + base, t[2] + base]));
            }
        }
        stack.extend(node.children().map(|c| (c, m)));
    }
    Ok(out)
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Base-colour texture sampled (nearest) at each vertex's UV.
fn texture_colors<'a, 's, F>(prim: &gltf::Primitive<'a>, reader: &gltf::mesh::Reader<'a, 's, F>, buffers: &[Vec<u8>]) -> Option<Vec<[u8; 3]>>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let info = prim.material().pbr_metallic_roughness().base_color_texture()?;
    let uvs: Vec<[f32; 2]> = reader.read_tex_coords(info.tex_coord()).map(|t| t.into_f32().collect())?;
    let img = match info.texture().source().source() {
        gltf::image::Source::View { view, .. } => {
            let buf = buffers.get(view.buffer().index())?;
            image::load_from_memory(buf.get(view.offset()..view.offset() + view.length())?).ok()?
        }
        gltf::image::Source::Uri { .. } => return None,
    }
    .to_rgb8();
    let (w, h) = img.dimensions();
    Some(
        uvs.iter()
            .map(|uv| {
                let x = ((uv[0].rem_euclid(1.0)) * w as f32) as u32;
                let y = ((uv[1].rem_euclid(1.0)) * h as f32) as u32;
                img.get_pixel(x.min(w - 1), y.min(h - 1)).0
            })
            .collect(),
    )
}

fn obj(path: &Path) -> Result<Loaded> {
    let mut mesh = Mesh::default();
    let mut has_color = false;
    for line in BufReader::new(std::fs::File::open(path)?).lines() {
        let line = line?;
        let mut words = line.split_whitespace();
        match words.next() {
            Some("v") => {
                let v = words.map(str::parse::<f32>).collect::<Result<Vec<_>, _>>()?;
                ensure!(v.len() >= 3, "incomplete OBJ vertex");
                mesh.positions.push(Vec3::new(v[0], v[1], v[2]));
                // Common extension: `v x y z r g b` with colours in [0, 1].
                if v.len() >= 6 {
                    has_color = true;
                    mesh.colors.push([v[3], v[4], v[5]].map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8));
                } else {
                    mesh.colors.push(GREY);
                }
            }
            Some("f") => {
                let nv = mesh.positions.len() as i64;
                let ids = words
                    .map(|w| -> Result<u32> {
                        let i: i64 = w.split('/').next().context("OBJ face index missing")?.parse()?;
                        let i = if i < 0 { nv + i } else { i - 1 };
                        ensure!(i >= 0, "invalid OBJ face index");
                        Ok(i as u32)
                    })
                    .collect::<Result<Vec<_>>>()?;
                polygon(&mut mesh.triangles, &ids);
            }
            _ => {}
        }
    }
    Ok(Loaded { mesh, has_color })
}

struct Property {
    name: String,
    kind: String,
    count: Option<String>,
}

struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

enum Numbers<'a> {
    Ascii(std::str::SplitWhitespace<'a>),
    Binary(Cursor<&'a [u8]>, bool),
}

impl Numbers<'_> {
    fn get(&mut self, ty: &str) -> Result<f64> {
        match self {
            Self::Ascii(it) => Ok(it.next().context("truncated ASCII PLY")?.parse()?),
            Self::Binary(input, big) => {
                macro_rules! number {
                    ($t:ty, $n:expr) => {{
                        let mut b = [0u8; $n];
                        input.read_exact(&mut b)?;
                        if *big { <$t>::from_be_bytes(b) as f64 } else { <$t>::from_le_bytes(b) as f64 }
                    }};
                }
                Ok(match ty {
                    "char" | "int8" => number!(i8, 1),
                    "uchar" | "uint8" => number!(u8, 1),
                    "short" | "int16" => number!(i16, 2),
                    "ushort" | "uint16" => number!(u16, 2),
                    "int" | "int32" => number!(i32, 4),
                    "uint" | "uint32" => number!(u32, 4),
                    "float" | "float32" => number!(f32, 4),
                    "double" | "float64" => number!(f64, 8),
                    _ => bail!("unsupported PLY scalar: {ty}"),
                })
            }
        }
    }
}

fn ply(path: &Path) -> Result<Loaded> {
    let bytes = std::fs::read(path)?;
    let mut reader = Cursor::new(bytes.as_slice());
    let mut line = String::new();
    let mut format = String::new();
    let mut elements: Vec<Element> = vec![];
    reader.read_line(&mut line)?;
    ensure!(line.trim() == "ply", "missing PLY header");
    loop {
        line.clear();
        ensure!(reader.read_line(&mut line)? > 0, "truncated PLY header");
        let words = line.split_whitespace().collect::<Vec<_>>();
        match words.as_slice() {
            ["end_header"] => break,
            ["format", kind, _] => format = (*kind).into(),
            ["element", name, n] => elements.push(Element { name: (*name).into(), count: n.parse()?, properties: vec![] }),
            ["property", "list", count, kind, name] => elements
                .last_mut()
                .context("PLY property before element")?
                .properties
                .push(Property { name: (*name).into(), kind: (*kind).into(), count: Some((*count).into()) }),
            ["property", kind, name] => elements
                .last_mut()
                .context("PLY property before element")?
                .properties
                .push(Property { name: (*name).into(), kind: (*kind).into(), count: None }),
            _ => {}
        }
    }
    let data = &bytes[reader.position() as usize..];
    let mut numbers = match format.as_str() {
        "ascii" => Numbers::Ascii(std::str::from_utf8(data)?.split_whitespace()),
        "binary_little_endian" => Numbers::Binary(Cursor::new(data), false),
        "binary_big_endian" => Numbers::Binary(Cursor::new(data), true),
        _ => bail!("unsupported PLY format {format}"),
    };
    let mut mesh = Mesh::default();
    let mut has_color = false;
    for element in elements {
        ensure!(element.count <= bytes.len(), "implausible PLY element count");
        for _ in 0..element.count {
            let mut xyz = [None; 3];
            let mut rgb = [None; 3];
            let mut face = vec![];
            for p in &element.properties {
                if let Some(count_type) = &p.count {
                    let count = numbers.get(count_type)?;
                    ensure!(count >= 0.0 && count <= bytes.len() as f64 && count.fract() == 0.0, "invalid PLY list count");
                    for _ in 0..count as usize {
                        let v = numbers.get(&p.kind)?;
                        if element.name == "face" && matches!(p.name.as_str(), "vertex_indices" | "vertex_index") {
                            ensure!(v >= 0.0 && v.fract() == 0.0, "invalid PLY index");
                            face.push(v as u32);
                        }
                    }
                } else {
                    let v = numbers.get(&p.kind)?;
                    if element.name == "vertex" {
                        if let Some(a) = ["x", "y", "z"].iter().position(|n| *n == p.name) {
                            xyz[a] = Some(v as f32);
                        } else if let Some(a) = ["red", "green", "blue"].iter().position(|n| *n == p.name) {
                            let scale = if p.kind.starts_with("float") || p.kind == "double" { 255.0 } else { 1.0 };
                            rgb[a] = Some((v * scale).round().clamp(0.0, 255.0) as u8);
                        }
                    }
                }
            }
            if element.name == "vertex" {
                mesh.positions.push(Vec3::new(xyz[0].context("PLY x missing")?, xyz[1].context("PLY y missing")?, xyz[2].context("PLY z missing")?));
                match rgb {
                    [Some(r), Some(g), Some(b)] => {
                        has_color = true;
                        mesh.colors.push([r, g, b]);
                    }
                    _ => mesh.colors.push(GREY),
                }
            } else if element.name == "face" {
                polygon(&mut mesh.triangles, &face);
            }
        }
    }
    Ok(Loaded { mesh, has_color })
}
