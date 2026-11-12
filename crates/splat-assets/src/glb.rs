//! GLB normalization: Y-up, meters, origin at bottom-center, textures capped in size.
//!
//! Geometry is never rewritten: the scene's roots are wrapped in one new node whose TRS applies the
//! up-axis fix, the uniform scale, and the bottom-center offset. Bounds come from POSITION accessor
//! min/max (8 corners per accessor) carried through the node hierarchy of the default scene.

use std::collections::HashMap;
use std::io::Cursor;

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat4, DQuat, DVec3};
use image::imageops::FilterType;
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use serde::Serialize;
use serde_json::{Value, json};

const MAGIC: &[u8; 4] = b"glTF";
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeTarget {
    /// Keep the source units.
    None,
    /// Uniform scale so the height (Y extent) is this many meters.
    Height(f32),
    /// Uniform scale so the largest extent is this many meters.
    MaxDim(f32),
    /// Largest uniform scale that fits inside `[w, h, d]` (components <= 0 are ignored).
    Dims([f32; 3]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Up {
    /// glTF standard.
    #[default]
    Y,
    /// Source is Z-up: rotate -90 degrees about X.
    Z,
}

#[derive(Debug, Clone)]
pub struct NormalizeOpts {
    pub up: Up,
    pub target: SizeTarget,
    /// Longest texture side after normalization.
    pub max_texture: u32,
}

impl Default for NormalizeOpts {
    fn default() -> Self {
        Self { up: Up::Y, target: SizeTarget::None, max_texture: 1024 }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NormalizeReport {
    /// `[w, h, d]` after normalization.
    pub dims_m: [f32; 3],
    pub bytes_in: usize,
    pub bytes_out: usize,
    pub textures_resized: usize,
}

struct Glb {
    json: Value,
    bin: Option<Vec<u8>>,
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    let s = b.get(at..at + 4).context("truncated GLB")?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn parse_glb(b: &[u8]) -> Result<Glb> {
    ensure!(b.len() >= 12 && &b[..4] == MAGIC, "not a GLB (bad magic)");
    ensure!(u32_at(b, 4)? == 2, "unsupported glTF container version {}", u32_at(b, 4)?);
    let total = (u32_at(b, 8)? as usize).min(b.len());
    let (mut at, mut json, mut bin) = (12, None, None);
    while at + 8 <= total {
        let (len, kind) = (u32_at(b, at)? as usize, u32_at(b, at + 4)?);
        let data = b.get(at + 8..at + 8 + len).context("truncated GLB chunk")?;
        match kind {
            CHUNK_JSON if json.is_none() => json = Some(serde_json::from_slice::<Value>(data).context("GLB JSON chunk")?),
            CHUNK_BIN if bin.is_none() => bin = Some(data.to_vec()),
            _ => {}
        }
        at += 8 + len;
    }
    Ok(Glb { json: json.context("GLB has no JSON chunk")?, bin })
}

fn pad4(v: &mut Vec<u8>, fill: u8) {
    while !v.len().is_multiple_of(4) {
        v.push(fill);
    }
}

fn write_glb(json: &Value, bin: Option<&[u8]>) -> Result<Vec<u8>> {
    let mut j = serde_json::to_vec(json)?;
    pad4(&mut j, b' ');
    let bin = bin.map(|b| {
        let mut b = b.to_vec();
        pad4(&mut b, 0);
        b
    });
    let total = 12 + 8 + j.len() + bin.as_ref().map_or(0, |b| 8 + b.len());
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(total)?.to_le_bytes());
    out.extend_from_slice(&u32::try_from(j.len())?.to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&j);
    if let Some(b) = bin {
        out.extend_from_slice(&u32::try_from(b.len())?.to_le_bytes());
        out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        out.extend_from_slice(&b);
    }
    Ok(out)
}

fn vec3(v: &Value, default: f64) -> DVec3 {
    match v.as_array() {
        Some(a) if a.len() == 3 => DVec3::new(
            a[0].as_f64().unwrap_or(default),
            a[1].as_f64().unwrap_or(default),
            a[2].as_f64().unwrap_or(default),
        ),
        _ => DVec3::splat(default),
    }
}

fn local_matrix(node: &Value) -> DMat4 {
    if let Some(m) = node["matrix"].as_array()
        && m.len() == 16
    {
        let mut a = [0.0; 16];
        for (dst, src) in a.iter_mut().zip(m) {
            *dst = src.as_f64().unwrap_or(0.0);
        }
        return DMat4::from_cols_array(&a);
    }
    let r = match node["rotation"].as_array() {
        Some(q) if q.len() == 4 => {
            let f = |i: usize| q[i].as_f64().unwrap_or(0.0);
            DQuat::from_xyzw(f(0), f(1), f(2), f(3)).normalize()
        }
        _ => DQuat::IDENTITY,
    };
    DMat4::from_scale_rotation_translation(vec3(&node["scale"], 1.0), r, vec3(&node["translation"], 0.0))
}

/// Root nodes of the default scene; with no scenes, every node that is nobody's child.
fn scene_roots(json: &Value) -> Vec<usize> {
    let idx = |v: &Value| v.as_array().map(|a| a.iter().filter_map(|n| n.as_u64()).map(|n| n as usize).collect::<Vec<_>>());
    let scene = json["scene"].as_u64().unwrap_or(0) as usize;
    if let Some(roots) = idx(&json["scenes"][scene]["nodes"]) {
        return roots;
    }
    let n = json["nodes"].as_array().map_or(0, Vec::len);
    let mut is_child = vec![false; n];
    for node in json["nodes"].as_array().into_iter().flatten() {
        for c in idx(&node["children"]).unwrap_or_default() {
            if c < n {
                is_child[c] = true;
            }
        }
    }
    (0..n).filter(|&i| !is_child[i]).collect()
}

/// Accessor min/max; falls back to scanning float VEC3 data in the BIN chunk.
fn accessor_minmax(json: &Value, bin: Option<&[u8]>, acc: usize) -> Result<(DVec3, DVec3)> {
    let a = &json["accessors"][acc];
    if a["min"].as_array().is_some_and(|v| v.len() == 3) && a["max"].as_array().is_some_and(|v| v.len() == 3) {
        return Ok((vec3(&a["min"], 0.0), vec3(&a["max"], 0.0)));
    }
    ensure!(a["componentType"] == 5126 && a["type"] == "VEC3", "accessor {acc} lacks min/max and is not float VEC3");
    let bv = &json["bufferViews"][a["bufferView"].as_u64().context("accessor without bufferView or min/max")? as usize];
    ensure!(bv["buffer"].as_u64().unwrap_or(0) == 0 && bv.get("extensions").is_none(), "accessor {acc} data not in the GLB BIN chunk");
    let bin = bin.context("GLB has no BIN chunk")?;
    let base = bv["byteOffset"].as_u64().unwrap_or(0) as usize + a["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = bv["byteStride"].as_u64().unwrap_or(12) as usize;
    let count = a["count"].as_u64().unwrap_or(0) as usize;
    let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    for i in 0..count {
        let at = base + i * stride;
        let f = |k: usize| -> Result<f64> {
            let s = bin.get(at + 4 * k..at + 4 * k + 4).context("accessor out of BIN range")?;
            Ok(f64::from(f32::from_le_bytes([s[0], s[1], s[2], s[3]])))
        };
        let p = DVec3::new(f(0)?, f(1)?, f(2)?);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    ensure!(count > 0, "accessor {acc} is empty");
    Ok((lo, hi))
}

/// World-space AABB of all mesh POSITION data under `base * node transforms`.
fn bounds(json: &Value, bin: Option<&[u8]>, base: DMat4) -> Result<(DVec3, DVec3)> {
    let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    let nodes = json["nodes"].as_array().map_or(&[][..], Vec::as_slice);
    let mut stack: Vec<(usize, DMat4, usize)> = scene_roots(json).into_iter().map(|r| (r, base, 0)).collect();
    while let Some((n, parent, depth)) = stack.pop() {
        ensure!(depth < 256, "node hierarchy too deep (cycle?)");
        let Some(node) = nodes.get(n) else { continue };
        let world = parent * local_matrix(node);
        if let Some(mesh) = node["mesh"].as_u64() {
            for prim in json["meshes"][mesh as usize]["primitives"].as_array().into_iter().flatten() {
                let Some(acc) = prim["attributes"]["POSITION"].as_u64() else { continue };
                let (a, b) = accessor_minmax(json, bin, acc as usize)?;
                for i in 0..8 {
                    let c = DVec3::new(
                        if i & 1 == 0 { a.x } else { b.x },
                        if i & 2 == 0 { a.y } else { b.y },
                        if i & 4 == 0 { a.z } else { b.z },
                    );
                    let p = world.transform_point3(c);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        for c in node["children"].as_array().into_iter().flatten().filter_map(Value::as_u64) {
            stack.push((c as usize, world, depth + 1));
        }
    }
    ensure!(lo.x.is_finite() && hi.x.is_finite(), "GLB has no mesh geometry in its default scene");
    Ok((lo, hi))
}

fn up_rotation(up: Up) -> DQuat {
    match up {
        Up::Y => DQuat::IDENTITY,
        Up::Z => DQuat::from_rotation_x(-std::f64::consts::FRAC_PI_2),
    }
}

fn scale_for(target: SizeTarget, size: DVec3) -> f64 {
    const EPS: f64 = 1e-9;
    let s = match target {
        SizeTarget::None => 1.0,
        SizeTarget::Height(h) if size.y > EPS => f64::from(h) / size.y,
        SizeTarget::MaxDim(m) if size.max_element() > EPS => f64::from(m) / size.max_element(),
        SizeTarget::Dims(d) => [size.x, size.y, size.z]
            .iter()
            .zip(d)
            .filter(|(s, t)| **s > EPS && *t > 0.0)
            .map(|(s, t)| f64::from(t) / s)
            .fold(f64::INFINITY, f64::min),
        _ => 1.0,
    };
    if s.is_finite() && s > 0.0 { s } else { 1.0 }
}

fn dims(size: DVec3) -> [f32; 3] {
    [size.x as f32, size.y as f32, size.z as f32]
}

/// `[w, h, d]` of the default scene's bounds, as stored (no up fix, no scaling).
pub fn glb_dims(input: &[u8]) -> Result<[f32; 3]> {
    let g = parse_glb(input)?;
    let (lo, hi) = bounds(&g.json, g.bin.as_deref(), DMat4::IDENTITY)?;
    Ok(dims(hi - lo))
}

fn strip_extras(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.remove("extras");
            m.values_mut().for_each(strip_extras);
        }
        Value::Array(a) => a.iter_mut().for_each(strip_extras),
        _ => {}
    }
}

fn uses_ext(json: &Value, name: &str) -> bool {
    ["extensionsUsed", "extensionsRequired"]
        .iter()
        .any(|k| json[*k].as_array().is_some_and(|a| a.iter().any(|e| e == name)))
}

fn bv_range(bv: &Value) -> (usize, usize) {
    let off = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    (off, off + bv["byteLength"].as_u64().unwrap_or(0) as usize)
}

/// Re-encode one image if its longest side exceeds `max`: JPEG q85 when fully opaque, else PNG.
fn shrink_image(bytes: &[u8], max: u32) -> Result<Option<(Vec<u8>, &'static str)>> {
    let img = image::load_from_memory(bytes)?;
    if img.width().max(img.height()) <= max {
        return Ok(None);
    }
    let img = img.resize(max, max, FilterType::Lanczos3);
    let alpha = img.color().has_alpha() && img.to_rgba8().pixels().any(|p| p.0[3] < 255);
    let mut out = Vec::new();
    if alpha {
        img.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)?;
        Ok(Some((out, "image/png")))
    } else {
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85).encode_image(&DynamicImage::ImageRgb8(img.to_rgb8()))?;
        Ok(Some((out, "image/jpeg")))
    }
}

/// Downscale embedded png/jpeg images and repack the BIN chunk. Returns how many were resized.
/// Skipped (returns 0) when bufferViews are not plain slices of a single embedded buffer.
fn downscale_textures(json: &mut Value, bin: &mut Vec<u8>, max: u32) -> Result<usize> {
    if uses_ext(json, "EXT_meshopt_compression") || uses_ext(json, "KHR_meshopt_compression") {
        return Ok(0);
    }
    let buffers = json["buffers"].as_array().map_or(0, Vec::len);
    if buffers != 1 || json["buffers"][0].get("uri").is_some() {
        return Ok(0);
    }
    let views = json["bufferViews"].as_array().cloned().unwrap_or_default();
    if views.iter().any(|v| v["buffer"].as_u64().unwrap_or(0) != 0 || bv_range(v).1 > bin.len()) {
        return Ok(0);
    }
    let mut replaced: HashMap<usize, Vec<u8>> = HashMap::new();
    let mut count = 0;
    for img in json["images"].as_array_mut().into_iter().flatten() {
        let Some(bv) = img["bufferView"].as_u64().map(|b| b as usize) else { continue };
        let mime = img["mimeType"].as_str().unwrap_or("");
        if !matches!(mime, "image/png" | "image/jpeg") || replaced.contains_key(&bv) {
            continue;
        }
        let Some(view) = views.get(bv) else { continue };
        let (a, b) = bv_range(view);
        // Undecodable images are left as they are.
        if let Ok(Some((bytes, new_mime))) = shrink_image(&bin[a..b], max) {
            img["mimeType"] = json!(new_mime);
            replaced.insert(bv, bytes);
            count += 1;
        }
    }
    if replaced.is_empty() {
        return Ok(0);
    }
    let mut out = Vec::with_capacity(bin.len());
    for (i, view) in json["bufferViews"].as_array_mut().into_iter().flatten().enumerate() {
        pad4(&mut out, 0);
        let (a, b) = bv_range(view);
        let data = replaced.get(&i).map_or(&bin[a..b], Vec::as_slice);
        view["byteOffset"] = json!(out.len());
        view["byteLength"] = json!(data.len());
        out.extend_from_slice(data);
    }
    pad4(&mut out, 0);
    json["buffers"][0]["byteLength"] = json!(out.len());
    *bin = out;
    Ok(count)
}

/// Normalize a GLB: up-axis fix, uniform scale to `opts.target`, bottom-center origin, textures
/// capped at `opts.max_texture`, `extras` stripped.
pub fn normalize_glb(input: &[u8], opts: &NormalizeOpts) -> Result<(Vec<u8>, NormalizeReport)> {
    let Glb { mut json, mut bin } = parse_glb(input)?;
    let rot = up_rotation(opts.up);
    let (lo, hi) = bounds(&json, bin.as_deref(), DMat4::from_quat(rot))?;
    let size = hi - lo;
    let s = scale_for(opts.target, size);
    let t = -s * DVec3::new(f64::midpoint(lo.x, hi.x), lo.y, f64::midpoint(lo.z, hi.z));

    strip_extras(&mut json);
    let roots = scene_roots(&json);
    let nodes = json["nodes"].as_array_mut().context("GLB has no nodes")?;
    let wrapper = nodes.len();
    nodes.push(json!({
        "name": "normalize_root",
        "children": roots,
        "translation": [t.x, t.y, t.z],
        "rotation": [rot.x, rot.y, rot.z, rot.w],
        "scale": [s, s, s],
    }));
    let scene = json["scene"].as_u64().unwrap_or(0) as usize;
    if json["scenes"][scene].is_object() {
        json["scenes"][scene]["nodes"] = json!([wrapper]);
    } else {
        json["scenes"] = json!([{"nodes": [wrapper]}]);
        json["scene"] = json!(0);
    }

    let mut resized = 0;
    if let Some(b) = bin.as_mut() {
        resized = downscale_textures(&mut json, b, opts.max_texture)?;
    }
    let out = write_glb(&json, bin.as_deref())?;
    // Re-measure from the written file so the report reflects what was actually produced.
    let after = glb_dims(&out)?;
    if (f64::from(after[1]) - s * size.y).abs() > 1e-3 * (1.0 + s * size.y) {
        bail!("normalize self-check failed: height {} vs expected {}", after[1], s * size.y);
    }
    let report = NormalizeReport { dims_m: after, bytes_in: input.len(), bytes_out: out.len(), textures_resized: resized };
    Ok((out, report))
}

/// Fit any decodable image into a `size` x `size` white square (no crop) and encode lossless WebP.
pub fn thumb_webp(image_bytes: &[u8], size: u32) -> Result<Vec<u8>> {
    let img = image::load_from_memory(image_bytes).context("decode thumbnail source")?;
    let fit = img.resize(size, size, FilterType::Lanczos3).to_rgba8();
    let mut canvas = RgbaImage::from_pixel(size, size, Rgba([255, 255, 255, 255]));
    let (x, y) = ((size - fit.width()) / 2, (size - fit.height()) / 2);
    image::imageops::overlay(&mut canvas, &fit, i64::from(x), i64::from(y));
    let mut out = Vec::new();
    image::codecs::webp::WebPEncoder::new_lossless(&mut out).encode(canvas.as_raw(), size, size, image::ExtendedColorType::Rgba8)?;
    Ok(out)
}
