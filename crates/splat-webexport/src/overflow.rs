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
    // A single SPZ cannot represent these splats. Supply a complete ordinary-importer
    // alternative in exactly the same frame, rather than calling the SPZ core a full scene.
    let standalone = format!("{stem}.ply");
    let needs_sidecars = !ovf.is_empty() || !dcp.is_empty();
    if needs_sidecars {
        crate::ply::write_ply(g, &dir.join(&standalone))?;
        eprintln!("export: {core_name} requires sidecars ({} omitted splats, {} colour patches); import {standalone} for the complete scene", ovf.len(), dcp.len());
    } else {
        let _ = std::fs::remove_file(dir.join(&standalone));
    }
    Ok(json!({
        "format": FORMAT,
        "standalone_splat": if needs_sidecars { standalone.as_str() } else { core_name.as_str() },
        "spz_standalone_complete": !needs_sidecars,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_import_retains_overflow_and_colour_and_removes_stale_fallback() {
        let dir = tempfile::tempdir().unwrap();
        // Failed-demo scale: almost every point lies outside fixed-12-bit SPZ range.
        let n = 73;
        let mut g = Gaussians {
            pos: (0..n).map(|i| if i == 0 { [0.0, 0.0, 1.0] } else { [i as f32 * 3173.2133, -7000.0, 9000.0] }).collect(),
            scale: vec![[-2.0, -3.0, -4.0]; n], rot: vec![[1.0, 0.0, 0.0, 0.0]; n],
            opacity: vec![1.5; n], dc: vec![[4.0, -5.0, 0.3]; n],
            sh_degree: 3, rest: (0..n * 45).map(|i| (i % 13) as f32 * 0.01).collect(),
            ..Default::default()
        };
        let o = spz::Options { version: 3, sh1_bits: 6, sh_rest_bits: 6, antialiased: false, symmetric_sh: true };
        let meta = write_package(&g, dir.path(), "scene", o).unwrap();
        assert_eq!(meta["core_count"], 1);
        assert_eq!(meta["overflow_count"], 72);
        assert_eq!(meta["standalone_splat"], "scene.ply");
        assert_eq!(meta["spz_standalone_complete"], false);
        // An ordinary PLY importer needs no manifest/sidecar knowledge and loses no attributes.
        let full = crate::ply::read_ply(&dir.path().join("scene.ply")).unwrap();
        assert_eq!(full.pos, g.pos);
        assert_eq!(full.scale, g.scale);
        assert_eq!(full.rot, g.rot);
        assert_eq!(full.opacity, g.opacity);
        assert_eq!(full.dc, g.dc);
        assert_eq!(full.rest, g.rest);
        for p in &mut g.pos { *p = [0.0, 1.0, 2.0]; }
        // Bright-colour patch alone must also select the complete file.
        let meta = write_package(&g, dir.path(), "scene", o).unwrap();
        assert_eq!(meta["overflow_count"], 0);
        assert_eq!(meta["standalone_splat"], "scene.ply");
        for c in &mut g.dc { *c = [0.2, 0.3, 0.4]; }
        let meta = write_package(&g, dir.path(), "scene", o).unwrap();
        assert_eq!(meta["standalone_splat"], "scene.spz");
        assert_eq!(meta["spz_standalone_complete"], true);
        assert!(!dir.path().join("scene.ply").exists());
        assert!(!dir.path().join("scene.ovf1.zlib").exists());
        assert!(!dir.path().join("scene.dcp1.zlib").exists());
    }
}
