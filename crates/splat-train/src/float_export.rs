//! Lossless eval export for the common paired scorer
//! (`research/r-oracle/paired_score.py`): per held-out view, the float render
//! as a float32 CHW `.npy` in [0, 1] (no 8-bit round-trip), plus one
//! `index.jsonl` row in the schema `paired_export.capture` writes for the
//! STACK and NHT arms, so all arms are scored by the same scorer on the same
//! views. Rows are written sorted by view name (bytewise), the scorer's order.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use serde_json::json;
use sha2::{Digest, Sha256};

/// One exported view: its index row, keyed by name for sorting.
pub struct ExportedView {
    pub name: String,
    pub row: serde_json::Value,
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// numpy `.npy` v1.0 for a C-order little-endian float32 array.
fn npy_f32(shape: [usize; 3], data: &[f32]) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        shape[0], shape[1], shape[2]
    );
    // Magic (6) + version (2) + length (2) + header, padded to 64 bytes, '\n'-terminated.
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// The scorer's id of an eval image: its path relative to the dataset's
/// images directory (the last component named `images*`, e.g. `images_4`),
/// `/`-separated, so multi-camera datasets whose cameras reuse file names
/// (`cam0/0001.jpg`, `cam1/0001.jpg`) get distinct ids. Flat datasets keep
/// the bare file name, which is what `paired_score.py` lists.
pub fn view_id(image_path: &Path) -> Option<String> {
    let parts: Vec<&str> = image_path.iter().map(|c| c.to_str()).collect::<Option<_>>()?;
    let start = parts
        .iter()
        .rposition(|c| c.starts_with("images"))
        .map_or(parts.len().saturating_sub(1), |i| i + 1);
    let rel = parts.get(start..).filter(|r| !r.is_empty())?;
    Some(rel.join("/"))
}

/// Write `<dir>/<name>.npy` from an HWC float render (already clamped to
/// [0, 1]) and return its index row. `name` is a [`view_id`] (relative,
/// `/`-separated); `gt` is the decoded target image the render was scored
/// against.
pub fn write_view(dir: &Path, name: &str, hwc: &[f32], h: usize, w: usize, gt: &DynamicImage) -> Result<ExportedView> {
    ensure!(
        !name.is_empty() && !name.starts_with('/') && name.split('/').all(|c| !c.is_empty() && c != "." && c != ".."),
        "bad view name {name}"
    );
    ensure!(hwc.len() == h * w * 3, "render is not [H, W, 3]");
    ensure!(hwc.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)), "render not finite in [0, 1]");
    let gt = gt.to_rgb8();
    ensure!((gt.width() as usize, gt.height() as usize) == (w, h), "render and target sizes differ");

    let plane = h * w;
    let mut pred = vec![0.0f32; 3 * plane];
    let mut target_u8 = vec![0u8; 3 * plane];
    for (p, (px, gt_px)) in hwc.chunks_exact(3).zip(gt.pixels()).enumerate() {
        for c in 0..3 {
            pred[c * plane + p] = px[c];
            target_u8[c * plane + p] = gt_px.0[c];
        }
    }
    let target_f32: Vec<f32> = target_u8.iter().map(|&v| f32::from(v) / 255.0).collect();
    // Same check `paired_export` makes: the float target is an exact 8-bit image.
    let roundtrip = target_f32
        .iter()
        .map(|&f| (f - (f * 255.0).round() / 255.0).abs())
        .fold(0.0f32, f32::max);
    let target_f32_bytes: Vec<u8> = target_f32.iter().flat_map(|v| v.to_le_bytes()).collect();

    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path: PathBuf = dir.join(format!("{name}.npy"));
    ensure!(!path.exists(), "{} exists (duplicate view id?)", path.display());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let npy = npy_f32([3, h, w], &pred);
    std::fs::write(&path, &npy).with_context(|| format!("writing {}", path.display()))?;
    let row = json!({
        "view_id": name,
        "shape": [3, h, w],
        "target_uint8_chw_sha256": hex(&target_u8),
        "target_float_roundtrip_max": roundtrip,
        "prediction_sha256": hex(&npy),
        "target_float32_chw_sha256": hex(&target_f32_bytes),
    });
    Ok(ExportedView { name: name.to_owned(), row })
}

/// Write `<dir>/index.jsonl` with the rows sorted bytewise by view name.
pub fn write_index(dir: &Path, mut views: Vec<ExportedView>) -> Result<()> {
    views.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    let mut text = String::new();
    for v in views {
        text.push_str(&v.row.to_string());
        text.push('\n');
    }
    std::fs::write(dir.join("index.jsonl"), text).context("writing index.jsonl")
}
