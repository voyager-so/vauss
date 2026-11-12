//! GLB (glTF 2.0 binary) and binary PLY writers with vertex colours and normals; GLB with a
//! baked baseColor texture.

use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::json;

use crate::{Mesh, TexturedMesh, vertex_normals};

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// glTF stores vertex colours linear; we convert from the sRGB the renders were in. The mesh
/// is written in its own (training) frame, without the glTF Y-up conversion.
pub fn glb_bytes(mesh: &Mesh) -> Vec<u8> {
    let normals = vertex_normals(mesh);
    let nv = mesh.positions.len();
    let mut bin = Vec::with_capacity(nv * 28 + mesh.triangles.len() * 12);
    for p in &mesh.positions {
        bin.extend(p.to_array().iter().flat_map(|f| f.to_le_bytes()));
    }
    let off_n = bin.len();
    for n in &normals {
        bin.extend(n.to_array().iter().flat_map(|f| f.to_le_bytes()));
    }
    let off_c = bin.len();
    for c in &mesh.colors {
        bin.extend(c.map(|v| (srgb_to_linear(v) * 255.0).round() as u8));
        bin.push(255);
    }
    let off_i = bin.len();
    for t in &mesh.triangles {
        bin.extend(t.iter().flat_map(|i| i.to_le_bytes()));
    }
    let len_i = bin.len() - off_i;
    let b = mesh.bounds().map_or(([0.0; 3], [0.0; 3]), |b| (b.min.to_array(), b.max.to_array()));
    let doc = json!({
        "asset": {"version": "2.0", "generator": "splat-mesh"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "splat_mesh"}],
        "meshes": [{"name": "splat_mesh", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
            "indices": 3, "material": 0, "mode": 4
        }]}],
        "materials": [{"name": "vertex_color", "doubleSided": true,
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": off_n, "target": 34962},
            {"buffer": 0, "byteOffset": off_n, "byteLength": off_c - off_n, "target": 34962},
            {"buffer": 0, "byteOffset": off_c, "byteLength": off_i - off_c, "target": 34962},
            {"buffer": 0, "byteOffset": off_i, "byteLength": len_i, "target": 34963}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": b.0, "max": b.1},
            {"bufferView": 1, "componentType": 5126, "count": nv, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5121, "normalized": true, "count": nv, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": mesh.triangles.len() * 3, "type": "SCALAR"}
        ]
    });
    pack_glb(&doc, bin)
}

/// GLB container: JSON chunk + BIN chunk, both 4-byte aligned.
fn pack_glb(doc: &serde_json::Value, mut bin: Vec<u8>) -> Vec<u8> {
    let mut js = serde_json::to_vec(doc).expect("json");
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(b"glTF");
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((js.len() as u32).to_le_bytes());
    out.extend(b"JSON");
    out.extend(js);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(b"BIN\0");
    out.extend(bin);
    out
}

/// Non-indexed triangles (three vertices each) with normals, TEXCOORD_0 and an embedded PNG
/// baseColor texture.
pub fn glb_bytes_textured(mesh: &TexturedMesh) -> Result<Vec<u8>> {
    let nv = mesh.positions.len();
    let mut png = Vec::new();
    mesh.texture.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    let mut bin = Vec::with_capacity(nv * 32 + png.len());
    for p in &mesh.positions {
        bin.extend(p.to_array().iter().flat_map(|f| f.to_le_bytes()));
    }
    let off_n = bin.len();
    for n in &mesh.normals {
        bin.extend(n.to_array().iter().flat_map(|f| f.to_le_bytes()));
    }
    let off_t = bin.len();
    for t in &mesh.uvs {
        bin.extend(t.iter().flat_map(|f| f.to_le_bytes()));
    }
    let off_img = bin.len();
    bin.extend(&png);
    let (lo, hi) = mesh.positions.iter().fold((glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let doc = json!({
        "asset": {"version": "2.0", "generator": "splat-mesh"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "splat_mesh"}],
        "meshes": [{"name": "splat_mesh", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "material": 0, "mode": 4
        }]}],
        "materials": [{"name": "baked", "doubleSided": true,
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": off_n, "target": 34962},
            {"buffer": 0, "byteOffset": off_n, "byteLength": off_t - off_n, "target": 34962},
            {"buffer": 0, "byteOffset": off_t, "byteLength": off_img - off_t, "target": 34962},
            {"buffer": 0, "byteOffset": off_img, "byteLength": png.len()}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": lo.to_array(), "max": hi.to_array()},
            {"bufferView": 1, "componentType": 5126, "count": nv, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": nv, "type": "VEC2"}
        ]
    });
    Ok(pack_glb(&doc, bin))
}

pub fn write_glb_textured(mesh: &TexturedMesh, path: &Path) -> Result<()> {
    let tmp = path.with_extension("glb.tmp");
    std::fs::write(&tmp, glb_bytes_textured(mesh)?).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn write_glb(mesh: &Mesh, path: &Path) -> Result<()> {
    let tmp = path.with_extension("glb.tmp");
    std::fs::write(&tmp, glb_bytes(mesh)).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Binary little-endian PLY: float x y z nx ny nz, uchar red green blue; int face indices.
pub fn write_ply(mesh: &Mesh, path: &Path) -> Result<()> {
    let tmp = path.with_extension("ply.tmp");
    {
        let mut f = BufWriter::new(std::fs::File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?);
        write!(
            f,
            "ply\nformat binary_little_endian 1.0\ncomment splat-mesh\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n",
            mesh.positions.len(),
            mesh.triangles.len()
        )?;
        let normals = vertex_normals(mesh);
        for ((p, n), c) in mesh.positions.iter().zip(&normals).zip(&mesh.colors) {
            for v in p.to_array().into_iter().chain(n.to_array()) {
                f.write_all(&v.to_le_bytes())?;
            }
            f.write_all(c)?;
        }
        for t in &mesh.triangles {
            f.write_all(&[3])?;
            for i in t {
                f.write_all(&(*i as i32).to_le_bytes())?;
            }
        }
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn glb_layout() {
        let mesh = Mesh { positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y], colors: vec![[255, 0, 0]; 3], triangles: vec![[0, 1, 2]] };
        let b = glb_bytes(&mesh);
        assert_eq!(&b[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize, b.len());
        let jl = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
        let doc: serde_json::Value = serde_json::from_slice(&b[20..20 + jl]).unwrap();
        assert_eq!(doc["accessors"][3]["count"], 3);
        assert_eq!(&b[20 + jl + 4..20 + jl + 8], b"BIN\0");
    }
}
