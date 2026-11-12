//! SPZ core plus lossless sidecars for what SPZ can't hold (the web viewer decodes them in
//! the exported viewer overflow decoder).
//!
//! - OVF1 (`<stem>.ovf1.zlib`): whole splats whose positions don't fit SPZ's signed 24-bit fixed
//!   point at 12 fractional bits (|x| >= 2048 units). They are left out of the SPZ core.
//! - DCP1 (`<stem>.dcp1.zlib`): float32 DC for core splats with any |DC| > 0.5/0.15, the largest
//!   value SPZ's 8-bit colour holds. The core keeps the clipped colour; the patch replaces it.
//!
//! Both are zlib(magic[4] | u32 total | u32 count | u32 indices[count] | f32 payload), little-endian.
//! Indices are ascending positions in the exported splat order; the core holds the other splats in
//! that same order. OVF1's payload is attribute-major over its records: positions[3], log scales[3],
//! quaternion wxyz[4], opacity logit[1], DC[3], higher SH [coef][channel]. DCP1's is DC[3] per record.

use std::io::Write;
use std::path::Path;

use anyhow::{Result, ensure};
use flate2::{Compression, write::ZlibEncoder};
use serde_json::{Value, json};

use crate::ply::Gaussians;
use crate::spz;

pub const FORMAT: &str = "spz-overflow-v1";
pub const POSITION_FRACTIONAL_BITS: u8 = 12;

/// Largest |DC| SPZ's colour byte holds: dc * 0.15 + 0.5 must stay in [0, 1].
pub fn dc_limit() -> f32 {
    0.5f32 / 0.15f32
}

pub(crate) fn position_fits(p: &[f32; 3]) -> bool {
    let scale = (1u32 << POSITION_FRACTIONAL_BITS) as f32;
    p.iter().all(|v| (-8388608.0..=8388607.0).contains(&((v * scale).round() as f64)))
}

fn write_block(path: &Path, magic: &[u8; 4], total: usize, ids: &[u32], payload: impl Iterator<Item = f32>) -> Result<u64> {
    let mut enc = ZlibEncoder::new(std::io::BufWriter::new(std::fs::File::create(path)?), Compression::best());
    enc.write_all(magic)?;
    enc.write_all(&(total as u32).to_le_bytes())?;
    enc.write_all(&(ids.len() as u32).to_le_bytes())?;
    for i in ids {
        enc.write_all(&i.to_le_bytes())?;
    }
    for v in payload {
        enc.write_all(&v.to_le_bytes())?;
    }
    enc.finish()?.flush()?;
    Ok(std::fs::metadata(path)?.len())
}

/// Write `<stem>.spz` (the in-range splats) and, when needed, the OVF1/DCP1 sidecars into `dir`.
/// Returns the scene.json `splat_overflow` entry (file names, counts, bytes).
pub fn write_package(g: &Gaussians, dir: &Path, stem: &str, options: spz::Options) -> Result<Value> {
    let n = g.len();
    ensure!(n > 0 && n <= u32::MAX as usize, "splat count {n} out of range");
    ensure!(g.pos.iter().flatten().all(|v| v.is_finite()), "SPZ requires finite positions");
    let fits: Vec<bool> = g.pos.iter().map(position_fits).collect();
    let ovf: Vec<u32> = (0..n).filter(|&i| !fits[i]).map(|i| i as u32).collect();
    let limit = dc_limit();
    let dcp: Vec<u32> = (0..n).filter(|&i| fits[i] && g.dc[i].iter().any(|v| v.abs() > limit)).map(|i| i as u32).collect();
    ensure!(ovf.len() < n, "every splat is beyond SPZ's position range");

    // Every core position fits at 12 bits, so the writer's adaptive precision picks 12.
    let core_name = format!("{stem}.spz");
    let core_bytes = if ovf.is_empty() {
        spz::write_spz_with_options(g, &dir.join(&core_name), options)?
    } else {
        let mut core = g.clone();
        core.retain(&fits);
        spz::write_spz_with_options(&core, &dir.join(&core_name), options)?
    };

    let k = g.sh_coeffs() * 3;
    let ovf_name = format!("{stem}.ovf1.zlib");
    let ovf_bytes = if ovf.is_empty() {
        None
    } else {
        let at = |i: &u32| *i as usize;
        let payload = ovf.iter().flat_map(|i| g.pos[at(i)])
            .chain(ovf.iter().flat_map(|i| g.scale[at(i)]))
            .chain(ovf.iter().flat_map(|i| g.rot[at(i)]))
            .chain(ovf.iter().map(|i| g.opacity[at(i)]))
            .chain(ovf.iter().flat_map(|i| g.dc[at(i)]))
            .chain(ovf.iter().flat_map(|i| g.rest[at(i) * k..(at(i) + 1) * k].iter().copied()));
        Some(write_block(&dir.join(&ovf_name), b"OVF1", n, &ovf, payload)?)
    };
    let dcp_name = format!("{stem}.dcp1.zlib");
    let dcp_bytes = if dcp.is_empty() {
        None
    } else {
        Some(write_block(&dir.join(&dcp_name), b"DCP1", n, &dcp, dcp.iter().flat_map(|i| g.dc[*i as usize]))?)
    };
    // Remove sidecars left by an earlier export into the same dir that this one doesn't need.
    for (name, bytes) in [(&ovf_name, ovf_bytes), (&dcp_name, dcp_bytes)] {
        if bytes.is_none() {
            let _ = std::fs::remove_file(dir.join(name));
        }
    }
    Ok(json!({
        "format": FORMAT,
        "core": core_name,
        "positions": ovf_bytes.map(|_| ovf_name),
        "dc": dcp_bytes.map(|_| dcp_name),
        "count": n,
        "core_count": n - ovf.len(),
        "overflow_count": ovf.len(),
        "dc_patch_count": dcp.len(),
        "sh_degree": g.sh_degree,
        "position_fractional_bits": POSITION_FRACTIONAL_BITS,
        "sh1_bits": options.sh1_bits,
        "sh_rest_bits": options.sh_rest_bits,
        "bytes": {"core": core_bytes, "positions": ovf_bytes, "dc": dcp_bytes,
            "total": core_bytes + ovf_bytes.unwrap_or(0) + dcp_bytes.unwrap_or(0)},
    }))
}
