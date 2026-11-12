//! First-paint package: a small SPZ of ~40K splats the viewer draws before anything else, so a
//! slow link shows the scene within about a second (1M-splat house at 25 Mbit/s on an M4: the
//! .RAD root chunk alone is 2.7 MB, first draw 3.2 s). Splats are drawn by weighted sampling
//! without replacement (Efraimidis-Spirakis, fixed seed), weight = opacity x projected area, so
//! the big surfaces that cover the frame come first while detail is still represented. SH stays at
//! the scene's degree: Spark 2.2 mis-renders splat files with fewer SH bands than it expects.
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::geom::Rng;
use crate::overflow::position_fits;
use crate::ply::Gaussians;
use crate::spz;

/// Splats in the first-paint file (1M house: about 1.1 MB of SPZ v3).
pub const FIRST_SPLATS: usize = 40_000;

/// Write `<dir>/<stem>.first.spz` and return its scene.json entry, or None when the scene is small
/// enough (under 2x [`FIRST_SPLATS`]) that its own SPZ is the first paint.
pub fn write_first(g: &Gaussians, dir: &Path, stem: &str, options: spz::Options) -> Result<Option<Value>> {
    let n = g.len();
    if n < 2 * FIRST_SPLATS {
        return Ok(None);
    }
    let limit = crate::overflow::dc_limit();
    // Sampling key ln(u) / w: the K largest keys are a weighted sample without replacement.
    let mut keyed: Vec<(f64, usize)> = (0..n)
        .into_par_iter()
        .filter(|&i| position_fits(&g.pos[i]) && g.dc[i].iter().all(|v| v.abs() <= limit))
        .map(|i| {
            let s = g.scale[i];
            let w = g.alpha(i) as f64 * ((2.0 / 3.0) * (s[0] + s[1] + s[2]) as f64).exp();
            let u = (Rng::new(0x5EED_F125 ^ i as u64).next() >> 11) as f64 / (1u64 << 53) as f64;
            (if w > 0.0 { u.max(1e-300).ln() / w } else { f64::NEG_INFINITY }, i)
        })
        .collect();
    let k = FIRST_SPLATS.min(keyed.len());
    keyed.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
    let mut idx: Vec<usize> = keyed[..k].iter().map(|&(_, i)| i).collect();
    idx.sort_unstable();
    let c = g.sh_coeffs() * 3;
    let sub = Gaussians {
        pos: idx.iter().map(|&i| g.pos[i]).collect(),
        scale: idx.iter().map(|&i| g.scale[i]).collect(),
        rot: idx.iter().map(|&i| g.rot[i]).collect(),
        opacity: idx.iter().map(|&i| g.opacity[i]).collect(),
        dc: idx.iter().map(|&i| g.dc[i]).collect(),
        rest: idx.iter().flat_map(|&i| g.rest[i * c..(i + 1) * c].iter().copied()).collect(),
        sh_degree: g.sh_degree,
        antialiased: g.antialiased,
        render_mode: g.render_mode,
        neural: None,
    };
    let file = format!("{stem}.first.spz");
    let bytes = spz::write_spz_with_options(&sub, &dir.join(&file), spz::Options { version: 3, ..options })?;
    Ok(Some(json!({"file": file, "bytes": bytes, "count": sub.len()})))
}
