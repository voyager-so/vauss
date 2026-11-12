//! Optional splat budget for a web package (`export-web --max-splats N`): keep the N most significant
//! splats (opacity x projected area, the LightGaussian global significance without per-view hit
//! counts). Off by default: pruning without retraining loses the fine detail (one scene, 4.66M -> 1M,
//! viewer vs the full package at 7 held-out views: 15.0 dB), so `vauss run --cloud` gets its 1M
//! interactive package from a capped refine instead.
use rayon::prelude::*;

use crate::ply::Gaussians;

/// Splat budget of `vauss run --cloud`'s interactive package (refined at this cap, not pruned).
pub const INTERACTIVE_SPLATS: usize = 1_000_000;

/// How the package was capped, for scene.json stats.
pub const METHOD: &str = "significance";

/// Significance of splat `i`: opacity x area (volume^(2/3) from the log scales).
fn significance(g: &Gaussians, i: usize) -> f32 {
    let s = g.scale[i];
    g.alpha(i) * ((2.0 / 3.0) * (s[0] + s[1] + s[2])).exp()
}

/// Keep the `max` most significant splats (all of them when `max` is 0 or not exceeded).
pub fn cap(g: &mut Gaussians, max: usize) {
    let n = g.len();
    if max == 0 || n <= max {
        return;
    }
    let mut order: Vec<(f32, usize)> = (0..n).into_par_iter().map(|i| (significance(g, i), i)).collect();
    order.select_nth_unstable_by(max - 1, |a, b| b.0.total_cmp(&a.0));
    let mut keep = vec![false; n];
    for &(_, i) in &order[..max] {
        keep[i] = true;
    }
    g.retain(&keep);
}
