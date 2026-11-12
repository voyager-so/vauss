//! export-viewer's camera collision grid (viewer.rs block 10, COLLIDE): the room's voxel occupancy, as
//! `vauss export-game` builds it (depth fusion, thickened walls and floor slab, the walked capture path
//! carved free). Keep the complete room occupancy (walls, floor, ceiling and furniture), not a tube
//! around the capture path. Grid bounds are storage extents, never artificial walls. The viewer removes
//! tiny disconnected voxel components before enabling floor-following capsule collision. Written as data/collide-NN.js parts of one "VXO1" binary (little endian):
//! ```text
//! magic "VXO1", u32 chunk_count, then per non-empty 32^3 chunk: i32 cx, cy, cz; u8[4096] mask
//! (bit lx + 32*(ly + 32*lz), LSB first; cell (cx*32+lx, ...) min corner = origin + edge * cell)
//! ```
//! i.e. the game's VXG1 without the cell count and colours. The grid is the package's `../game` export
//! when that was built from this package (`vauss run` writes the game first; its levelled voxels are mapped
//! back into the package frame with the inverse of build.json level.matrix, at >= 5 cm), else built here at
//! 5 cm in the package frame by the same code (`game::occupancy`, ~10 s for a room).
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use base64::Engine;
use glam::{DMat4, DVec3, Vec3};
use serde_json::{Value, json};

const CHUNK: usize = 32;
const MASK_BYTES: usize = CHUNK * CHUNK * CHUNK / 8;
/// Built grid edge (cm); a game grid finer than this is not reused (its chunks would be too many).
const BUILD_CM: f64 = 5.0;
/// The page's camera sphere radius (m).
const RADIUS_M: f64 = 0.25;
/// The page's stop distance (m): how far the camera keeps from what is in front of it (the hard radius in narrow spots).
const STOP_M: f64 = 0.7;

struct Occ {
    origin: [f64; 3],
    edge: f64,
    dims: [usize; 3],
    bin: Vec<u8>,
    cells: usize,
    chunks: usize,
    source: String,
    /// The room's true vertical in the package frame (the game's levelling), for the page's camera up.
    up: Option<[f64; 3]>,
    /// The levelled floor's height along `up` (the page's walk floor in that frame).
    floor: Option<f64>,
}

/// The game's levelled vertical in the package frame and the floor's height along it. The levelling
/// (`inv` = game -> package) rotates about a pivot on the floor, so the floor plane passes through the
/// pivot with normal `up`. None for an unlevelled game (the page then uses scene.json's up and floor).
fn level_floor(inv: &DMat4, pivot: Option<[f64; 3]>) -> (Option<[f64; 3]>, Option<f64>) {
    let up = inv.transform_vector3(DVec3::Y).normalize();
    if !(up.y > 0.0 && up.y < 1.0 - 1e-9) { return (None, None); }
    let r5 = |v: f64| (v * 1e5).round() / 1e5;
    (Some(up.to_array().map(r5)), pivot.map(|p| r5(up.dot(DVec3::from_array(p)))))
}

/// VXO1 from a dense solid grid.
fn encode(dims: [usize; 3], solid: &[bool]) -> (Vec<u8>, usize, usize) {
    let nc = dims.map(|d| d.div_ceil(CHUNK));
    let mut out = b"VXO1".to_vec();
    out.extend_from_slice(&0u32.to_le_bytes());
    let (mut chunks, mut cells) = (0u32, 0usize);
    for cz in 0..nc[2] { for cy in 0..nc[1] { for cx in 0..nc[0] {
        let mut mask = [0u8; MASK_BYTES];
        let mut n = 0;
        for lz in 0..CHUNK { for ly in 0..CHUNK { for lx in 0..CHUNK {
            let (x, y, z) = (cx * CHUNK + lx, cy * CHUNK + ly, cz * CHUNK + lz);
            if x >= dims[0] || y >= dims[1] || z >= dims[2] || !solid[(z * dims[1] + y) * dims[0] + x] { continue; }
            let bit = lx + CHUNK * (ly + CHUNK * lz);
            mask[bit >> 3] |= 1 << (bit & 7);
            n += 1;
        }}}
        if n == 0 { continue; }
        for c in [cx, cy, cz] { out.extend_from_slice(&(c as i32).to_le_bytes()); }
        out.extend_from_slice(&mask);
        chunks += 1;
        cells += n;
    }}}
    out[4..8].copy_from_slice(&chunks.to_le_bytes());
    (out, chunks as usize, cells)
}

/// The solid cells of a VXG1 binary (the game's voxels) as a dense grid of `dims`.
fn decode_vxg1(b: &[u8], dims: [usize; 3]) -> Result<Vec<bool>> {
    ensure!(b.len() >= 8 && &b[..4] == b"VXG1", "not a VXG1 binary");
    let count = u32::from_le_bytes(b[4..8].try_into()?) as usize;
    let mut solid = vec![false; dims[0] * dims[1] * dims[2]];
    let mut at = 8;
    for _ in 0..count {
        let head = b.get(at..at + 16 + MASK_BYTES).context("truncated VXG1 chunk")?;
        let c: Vec<usize> = (0..3).map(|k| i32::from_le_bytes(head[4 * k..4 * k + 4].try_into().unwrap()).max(0) as usize).collect();
        let n = u32::from_le_bytes(head[12..16].try_into()?) as usize;
        let mask = &head[16..];
        for bit in 0..CHUNK * CHUNK * CHUNK {
            if mask[bit >> 3] & (1 << (bit & 7)) == 0 { continue; }
            let (x, y, z) = (c[0] * CHUNK + bit % CHUNK, c[1] * CHUNK + (bit / CHUNK) % CHUNK, c[2] * CHUNK + bit / (CHUNK * CHUNK));
            ensure!(x < dims[0] && y < dims[1] && z < dims[2], "VXG1 cell outside dims");
            solid[(z * dims[1] + y) * dims[0] + x] = true;
        }
        at += 16 + MASK_BYTES + 3 * n;
    }
    ensure!(at == b.len(), "VXG1: {} trailing bytes", b.len().saturating_sub(at));
    Ok(solid)
}

/// The game export next to the package (`<pkg>/../game`, export-game's default) when it was built from `pkg`
/// (and after every file in it): its voxel world, brought back into the package frame (the game is levelled:
/// build.json level.matrix maps package -> game) on a BUILD_CM grid (at least the game's edge). A cell is solid
/// when any of its 2x2x2 sub-samples falls in a solid game cell, so thin walls stay closed.
fn from_game(pkg: &Path) -> Result<Option<Occ>> {
    let dir = pkg.parent().unwrap_or(pkg).join("game");
    let Ok(raw) = fs::read(dir.join("build.json")) else { return Ok(None) };
    let b: Value = serde_json::from_slice(&raw)?;
    // Same package, and written after it (a run's exports/web is rewritten in place by later stages).
    let mtime = |p: &Path| fs::metadata(p).and_then(|m| m.modified()).ok();
    let newer = mtime(&dir.join("build.json")).is_some_and(|t| fs::read_dir(pkg).into_iter().flatten().flatten()
        .all(|e| mtime(&e.path()).is_none_or(|m| m <= t)));
    let same = newer && b["source"].as_str().and_then(|s| fs::canonicalize(s).ok()) == fs::canonicalize(pkg).ok();
    if !same {
        eprintln!("export-viewer: {} is from another package or older; building the collision grid", dir.display());
        return Ok(None);
    }
    let v = &b["voxel"];
    let parts = v["parts"].as_u64().context("game build.json: voxel.parts")?;
    let mut bin = Vec::new();
    for i in 0..parts {
        let f = dir.join(format!("data/voxels-{i:02}.js"));
        let js = fs::read_to_string(&f).with_context(|| format!("reading {}", f.display()))?;
        let b64 = js.split("\"]=\"").nth(1).and_then(|s| s.split('"').next()).with_context(|| format!("{}: no data", f.display()))?;
        bin.extend(base64::engine::general_purpose::STANDARD.decode(b64)?);
    }
    ensure!(v["binary_bytes"].as_u64() == Some(bin.len() as u64), "game voxels: {} bytes, build.json says {}", bin.len(), v["binary_bytes"]);
    let go = crate::viewer::v3(&v["origin"]).context("game build.json: voxel.origin")?;
    let gd: Vec<usize> = v["dims"].as_array().context("game build.json: voxel.dims")?.iter().filter_map(|d| d.as_u64().map(|d| d as usize)).collect();
    ensure!(gd.len() == 3, "game build.json: voxel.dims");
    let gd = [gd[0], gd[1], gd[2]];
    let ge = v["edge_units"].as_f64().context("game build.json: voxel.edge_units")?;
    let gsolid = decode_vxg1(&bin, gd)?;
    // Package -> game frame; identity for a game built before levelling was recorded.
    let m: Vec<f64> = b["level"]["matrix"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
    let m = if m.len() == 16 { DMat4::from_cols_slice(&m) } else { DMat4::IDENTITY };
    let inv = m.inverse();
    // The game world's box in the package frame.
    let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
    for k in 0..8 {
        let c = DVec3::new(if k & 1 == 0 { 0.0 } else { gd[0] as f64 }, if k & 2 == 0 { 0.0 } else { gd[1] as f64 },
            if k & 4 == 0 { 0.0 } else { gd[2] as f64 });
        let p = inv.transform_point3(DVec3::from_array(go) + c * ge);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let mpu_edge = BUILD_CM / 100.0; // package units are metres (y_up_meters)
    let edge = ge.max(mpu_edge);
    let dims = ((hi - lo) / edge).ceil().to_array().map(|d| d.max(1.0) as usize);
    let mut solid = vec![false; dims[0] * dims[1] * dims[2]];
    for z in 0..dims[2] { for y in 0..dims[1] { for x in 0..dims[0] {
        let base = lo + DVec3::new(x as f64, y as f64, z as f64) * edge;
        let hit = (0..8).any(|k| {
            let off = DVec3::new(if k & 1 == 0 { 0.25 } else { 0.75 }, if k & 2 == 0 { 0.25 } else { 0.75 }, if k & 4 == 0 { 0.25 } else { 0.75 });
            let q = (m.transform_point3(base + off * edge) - DVec3::from_array(go)) / ge;
            let c = q.floor();
            c.cmpge(DVec3::ZERO).all() && (c.x as usize) < gd[0] && (c.y as usize) < gd[1] && (c.z as usize) < gd[2]
                && gsolid[((c.z as usize) * gd[1] + c.y as usize) * gd[0] + c.x as usize]
        });
        solid[(z * dims[1] + y) * dims[0] + x] = hit;
    }}}
    let (bin, chunks, cells) = encode(dims, &solid);
    let (up, floor) = level_floor(&inv, crate::viewer::v3(&b["level"]["pivot"]));
    eprintln!("export-viewer: collision from the game export {}", dir.display());
    // Served in the page: no local paths.
    Ok(Some(Occ { origin: lo.to_array(), edge, dims, bin, cells, chunks, up, floor,
        source: format!("game export ({} cm, levelled {} deg), resampled to the package frame", v["cm"], b["level"]["angle_deg"]) }))
}

fn build(pkg: &Path, scene: &Value) -> Result<Occ> {
    let (origin, edge, dims, solid, _) = crate::game::occupancy(pkg, scene, BUILD_CM)?;
    let (bin, chunks, cells) = encode(dims, &solid);
    let o: Vec3 = origin;
    Ok(Occ { origin: [o.x as f64, o.y as f64, o.z as f64], edge: edge as f64, dims, bin, cells, chunks, up: None, floor: None,
        source: format!("built ({BUILD_CM} cm depth fusion)") })
}

/// Write data/collide-NN.js for the package; returns the page's `collide` config.
pub(crate) fn write(pkg: &Path, scene: &Value, data: &Path, part_bytes: usize) -> Result<Value> {
    let t0 = Instant::now();
    let occ = match from_game(pkg) {
        Ok(Some(o)) => o,
        Ok(None) => build(pkg, scene)?,
        Err(e) => {
            eprintln!("export-viewer: game voxels unusable ({e:#}); building the collision grid");
            build(pkg, scene)?
        }
    };
    ensure!(occ.cells > 0, "empty collision grid");
    let (parts, js) = crate::viewer::write_parts(&occ.bin, data, "collide", part_bytes)?;
    let mpu = scene["meters_per_unit"].as_f64().filter(|m| m.is_finite() && *m > 0.0).unwrap_or(1.0);
    let r5 = |v: f64| (v * 1e5).round() / 1e5;
    let level_up = occ.up.map_or(Value::Null, |u| json!(u));
    eprintln!("export-viewer: collision grid {} cells, {} chunks, {}x{}x{} at {:.1} cm, {} bytes ({}; {:.1} s)", occ.cells, occ.chunks,
        occ.dims[0], occ.dims[1], occ.dims[2], occ.edge * mpu * 100.0, occ.bin.len(), occ.source, t0.elapsed().as_secs_f64());
    Ok(json!({"key": "collide", "parts": parts, "bytes": occ.bin.len(), "js_bytes": js, "format": "VXO1",
        "origin": occ.origin.map(r5), "edge": r5(occ.edge), "dims": occ.dims, "chunk": CHUNK, "cells": occ.cells,
        "chunks": occ.chunks, "cm": (occ.edge * mpu * 1e3).round() / 10.0, "radiusM": RADIUS_M, "stopM": STOP_M, "source": occ.source, "up": level_up,
        "floorHeight": occ.floor,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levelled_floor_height_is_measured_along_the_levelled_up() {
        // The game's levelling as game.rs builds it: a rotation about a floor pivot, package -> game.
        let pivot = DVec3::new(0.7, -0.2, 1.3);
        let tilt = DVec3::new(0.08, 1.0, -0.05).normalize();
        let m = DMat4::from_translation(pivot) * DMat4::from_quat(glam::DQuat::from_rotation_arc(tilt, DVec3::Y))
            * DMat4::from_translation(-pivot);
        let (up, floor) = level_floor(&m.inverse(), Some(pivot.to_array()));
        let (up, floor) = (DVec3::from_array(up.unwrap()), floor.unwrap());
        assert!(up.dot(tilt) > 1.0 - 1e-8);
        // Points on the levelled floor (game y == pivot y) sit at `floor` along `up` in the package frame.
        for (x, z) in [(0.0, 0.0), (3.0, -2.0), (-4.0, 5.0)] {
            let p = m.inverse().transform_point3(DVec3::new(pivot.x + x, pivot.y, pivot.z + z));
            assert!((p.dot(up) - floor).abs() < 1e-4, "{x} {z}: {} vs {floor}", p.dot(up));
        }
        assert_eq!(level_floor(&DMat4::IDENTITY, Some([0.0; 3])), (None, None));
    }
}
