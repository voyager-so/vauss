//! Neural appearance sidecar (`<stem>.nrl1.zlib`, scene.json `splat_neural`): the per-splat latents
//! and the shared decoder of a trainer neural-appearance PLY ([`crate::ply::Neural`]). The SPZ core
//! keeps the base colour as DC, so viewers without decoder support show the view-independent colour;
//! the exported viewer decoder evaluates the decoder per splat per view.
//!
//! zlib(magic "NRL1" | u32 count | u32 decoder_len | f32 decoder[decoder_len] | f32 latents[count][8]),
//! little-endian. Latents are in the exported splat order (the order `splat_overflow` indexes).
//!
//! Decoder (trainer `brush_render::kernels::neural`, version x5b-v1), for the unit view direction
//! `d = normalize(mean - camera)` in the source (trainer) frame:
//!   x  = [sin f0, cos f0, .., sin f7, cos f7, real SH bands 0..3 of d]   (32)
//!   h1 = ReLU(W1 x), h2 = ReLU(W2 h1), rgb = max(C0 * dc + tanh(W3 h2) + 0.5, 0); no biases.
//! The exported scene is rotated from the source frame, so the entry carries `source_from_scene`
//! (row-major 3x3 rotation) to take scene-frame directions back to the trainer's frame.

use std::io::Write;
use std::path::Path;

use anyhow::{Result, ensure};
use flate2::{Compression, write::ZlibEncoder};
use glam::DMat3;
use serde_json::{Value, json};

use crate::ply::{Gaussians, NEURAL_DECODER_LEN};

pub const FORMAT: &str = "neural-x5b-v1";

/// Write `<stem>.nrl1.zlib` into `dir` when `g` is a neural splat (else remove a stale one) and
/// return the scene.json `splat_neural` entry. `scene_from_source` is the export's rotation.
pub fn write_sidecar(g: &Gaussians, dir: &Path, stem: &str, scene_from_source: DMat3) -> Result<Option<Value>> {
    let name = format!("{stem}.nrl1.zlib");
    let Some(n) = &g.neural else {
        let _ = std::fs::remove_file(dir.join(&name));
        return Ok(None);
    };
    ensure!(n.latents.len() == g.len() && n.decoder.len() == NEURAL_DECODER_LEN, "neural data doesn't match the splats");
    let path = dir.join(&name);
    let mut enc = ZlibEncoder::new(std::io::BufWriter::new(std::fs::File::create(&path)?), Compression::fast());
    enc.write_all(b"NRL1")?;
    enc.write_all(&(g.len() as u32).to_le_bytes())?;
    enc.write_all(&(NEURAL_DECODER_LEN as u32).to_le_bytes())?;
    for v in n.decoder.iter().chain(n.latents.iter().flatten()) {
        enc.write_all(&v.to_le_bytes())?;
    }
    enc.finish()?.flush()?;
    let bytes = std::fs::metadata(&path)?.len();
    let r = scene_from_source.transpose();
    let rows: Vec<f64> = (0..3).flat_map(|i| (0..3).map(move |j| r.col(j)[i])).map(|v| (v * 1e9).round() / 1e9).collect();
    Ok(Some(json!({
        "format": FORMAT,
        "file": name,
        "count": g.len(),
        "latents": crate::ply::NEURAL_LATENTS,
        "decoder_len": NEURAL_DECODER_LEN,
        "source_from_scene": rows,
        "bytes": bytes,
    })))
}
