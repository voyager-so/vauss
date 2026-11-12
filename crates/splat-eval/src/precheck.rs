//! Capture pre-check: a few seconds of local analysis of a video, before any upload or GPU work,
//! that tells the user in plain words whether it will work and how to re-shoot it.
//!
//! One decode at [`ANALYSIS_FPS`] gives a grey frame with a long edge of 640 px (light, sharpness)
//! and the frame plan's own 160 px motion track (total camera travel, the measure SfM failure messages
//! already use). On a 320 px copy, coarse-to-fine block matching between consecutive samples gives
//! the image motion (pan speed, yaw for room coverage), and between keyframes ~0.15 image widths
//! apart a homography fit: a camera that only turns (or sees one flat wall) moves the image by a
//! homography, so blocks that don't fit it are parallax, the depth signal SfM needs.
//!
//! Checks: length, light, sharpness, movement (travel and parallax), speed, coverage (rooms only).
//! Each is pass / warn / fail with a one-line fix; any fail stops `vauss run` before upload unless
//! overridden. Thresholds are calibrated on a corpus of capture outcomes (see the constants below).

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::path::Path;
use std::time::Instant;

use crate::frameplan::{self, ANALYSIS_FPS, MotionTracker, Transfer};

/// Long edge of the light / sharpness frame.
const BIG_EDGE: usize = 640;
/// Block size for the 320 px flow.
const BLOCK: usize = 16;
/// Keyframe pairs are evaluated once the image has moved this far (fraction of the long edge)...
const PAIR_SHIFT: f64 = 0.15;
/// ...or after this many samples (4 s).
const PAIR_MAX_SAMPLES: usize = 32;
/// Homography residual (px at a 320 px long edge) above which a block counts as parallax.
const PARALLAX_PX: f64 = 1.5;
/// Mean luma (0-255) below which a frame is dim (about 2 stops under mid grey).
const DIM_LUMA: f64 = 64.0;
/// Decoder threads over all parallel segments.
const DECODE_THREADS: usize = 10;
/// Sparse decode above this many decoded pixels (width x height x fps x duration): 2 min of 4K30 is
/// 3.0e10 and took ~19-22 s to decode in full on an M4; 2 min of 1080p30 (7.5e9) ~5 s stays full.
const SPARSE_PIXEL_FRAMES: f64 = 1.2e10;
/// Sparse windows: SPARSE_WINDOW_S decoded every SPARSE_PERIOD_S (over SEGMENT_S, so the 5 s
/// stretch warnings and the frames-stop-overlapping fail can still be seen inside one window). On an
/// M4, 2 min of 4K: 9.1 s (HEVC 30 fps) / 7.9 s (H.264 60 fps) of decode vs 18.8 / 16.5 s in full.
const SPARSE_WINDOW_S: f64 = 5.5;
const SPARSE_PERIOD_S: f64 = 15.0;
/// Assumed field of view across the image diagonal when neither the file nor the motion says: a
/// phone's main (1x, ~26 mm equivalent) camera, 79.5 deg.
const DIAG_FOV_DEG: f64 = 79.5;
/// Self-calibration: long-edge field of view searched in this range, from keyframe pairs that
/// moved at least FOV_PAIR_SHIFT of the long edge, needing FOV_MIN_PAIRS of them.
const FOV_MIN_DEG: f64 = 35.0;
const FOV_MAX_DEG: f64 = 130.0;
const FOV_PAIR_SHIFT: f64 = 0.1;
const FOV_MIN_PAIRS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Pass,
    /// Advice only: shown on its check line, but the video's overall verdict stays pass.
    Tip,
    Warn,
    Fail,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Tip => "tip",
            Verdict::Warn => "WARN",
            Verdict::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub id: &'static str,
    pub name: &'static str,
    pub verdict: Verdict,
    /// What we saw, in plain words.
    pub message: String,
    /// One-line re-shoot advice (empty on pass).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub fix: String,
}

#[derive(Debug, Clone, Default)]
pub struct CheckOptions {
    /// Judge it as a room (adds the coverage check and room-length advice).
    pub room: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaptureCheck {
    pub verdict: Verdict,
    pub summary: String,
    pub room: bool,
    pub checks: Vec<Check>,
    pub metrics: Value,
    pub seconds: f64,
}

impl CaptureCheck {
    /// Plain-text report for a terminal.
    pub fn render(&self) -> String {
        let mut s = format!("Capture check: {}\n", self.summary);
        for c in &self.checks {
            s += &format!("  [{:4}] {:<9} {}\n", c.verdict.label(), c.name, c.message);
            if !c.fix.is_empty() {
                s += &format!("         {:<9} Fix: {}\n", "", c.fix);
            }
        }
        if self.verdict != Verdict::Pass || self.checks.iter().any(|c| c.verdict == Verdict::Tip) {
            s += "  Filming guide: `vauss check --guide` (docs/filming-guide.md).\n";
        }
        s
    }
}

/// Whether a run goal reads like a room or other interior (turns on the coverage check).
pub fn goal_is_room(goal: &str) -> bool {
    let g = goal.to_lowercase();
    ["room", "kitchen", "bedroom", "bathroom", "office", "indoor", "interior", "apartment", "flat", "studio", "hallway", "lounge", "basement", "garage"]
        .iter()
        .any(|w| g.split(|c: char| !c.is_alphanumeric()).any(|t| t == *w || t.trim_end_matches('s') == *w))
}

/// The filming guide (docs/filming-guide.md), for `vauss check --guide`.
pub const FILMING_GUIDE: &str = include_str!("../../../docs/filming-guide.md");

/// Per-sample measurements.
#[derive(Debug, Clone, Default, Serialize)]
struct Sample {
    luma: f64,
    sharp: f64,
    /// Fine-over-coarse Laplacian energy (640 vs 320 px): low = soft at the pixel level.
    detail: f64,
    /// Vertical / horizontal gradient energy (1 = isotropic; a fast yaw smears horizontally).
    smear: f64,
    /// Image motion since the previous sample (fraction of the long edge), signed.
    dx: f64,
    dy: f64,
    /// Motion-blur length (px at the 640 px analysis frame) along x and along y, on every
    /// SMEAR_EVERY-th sample: None where no smear is measurable.
    blur_x: Option<f64>,
    blur_y: Option<f64>,
    /// On the same samples: corners (Shi-Tomasi, 640 px) and how many of them match the next measured
    /// sample (NCC at the flow-predicted spot): a cheap stand-in for SfM's feature matches.
    corners: Option<u32>,
    matches: Option<u32>,
    /// Share of textured blocks matched to the previous sample.
    tracked: f64,
    /// Share of blocks with texture.
    textured: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Pair {
    t0: f64,
    t1: f64,
    shift: f64,
    blocks: usize,
    /// Share of matched blocks off the best homography by more than PARALLAX_PX.
    off_share: f64,
    p80_px: f64,
    #[serde(skip)]
    h: Option<[f64; 9]>,
}

/// A grey image.
#[derive(Clone)]
struct Grey {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

impl Grey {
    fn half(&self) -> Grey {
        let (w, h) = (self.w / 2, self.h / 2);
        let mut px = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let s = self.px[2 * y * self.w + 2 * x] as u32
                    + self.px[2 * y * self.w + 2 * x + 1] as u32
                    + self.px[(2 * y + 1) * self.w + 2 * x] as u32
                    + self.px[(2 * y + 1) * self.w + 2 * x + 1] as u32;
                px[y * w + x] = (s / 4) as u8;
            }
        }
        Grey { w, h, px }
    }
}

/// Laplacian variance and horizontal / vertical gradient energy.
fn texture_stats(g: &Grey) -> (f64, f64, f64) {
    let (w, h) = (g.w, g.h);
    if w < 3 || h < 3 {
        return (0.0, 0.0, 0.0);
    }
    let p = |x: usize, y: usize| g.px[y * w + x] as f64;
    let (mut sum, mut sq, mut gx, mut gy) = (0f64, 0f64, 0f64, 0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = p(x, y);
            let l = p(x - 1, y) + p(x + 1, y) + p(x, y - 1) + p(x, y + 1) - 4.0 * c;
            sum += l;
            sq += l * l;
            gx += (p(x + 1, y) - c).powi(2);
            gy += (p(x, y + 1) - c).powi(2);
        }
    }
    let n = ((w - 2) * (h - 2)) as f64;
    (sq / n - (sum / n).powi(2), gx / n, gy / n)
}

fn sad(a: &Grey, b: &Grey, bx: usize, by: usize, size: usize, dx: i32, dy: i32) -> Option<u32> {
    let (x0, y0) = (bx as i32 + dx, by as i32 + dy);
    if x0 < 0 || y0 < 0 || x0 as usize + size > b.w || y0 as usize + size > b.h {
        return None;
    }
    let mut s = 0u32;
    for y in 0..size {
        let ra = &a.px[(by + y) * a.w + bx..(by + y) * a.w + bx + size];
        let rb = &b.px[(y0 as usize + y) * b.w + x0 as usize..(y0 as usize + y) * b.w + x0 as usize + size];
        s += ra.iter().zip(rb).map(|(&p, &q)| (p as i32 - q as i32).unsigned_abs()).sum::<u32>();
    }
    Some(s)
}

/// Mean absolute deviation of a block (its texture).
fn block_mad(g: &Grey, bx: usize, by: usize, size: usize) -> f64 {
    let vals = (0..size).flat_map(|y| g.px[(by + y) * g.w + bx..(by + y) * g.w + bx + size].iter().map(|&v| v as f64));
    let v: Vec<f64> = vals.collect();
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (x - m).abs()).sum::<f64>() / v.len() as f64
}

/// Global translation at a coarse level: exhaustive search minimising the mean abs difference over
/// the overlap (at least half the frame).
fn global_shift(a: &Grey, b: &Grey, prior: (i32, i32), radius: i32) -> (i32, i32) {
    let mut best = (f64::MAX, prior.0, prior.1);
    for dy in prior.1 - radius..=prior.1 + radius {
        for dx in prior.0 - radius..=prior.0 + radius {
            let (x0, x1) = (0.max(-dx), (a.w as i32).min(b.w as i32 - dx));
            let (y0, y1) = (0.max(-dy), (a.h as i32).min(b.h as i32 - dy));
            if x1 - x0 < a.w as i32 / 2 || y1 - y0 < a.h as i32 / 2 {
                continue;
            }
            let mut s = 0u64;
            for y in y0..y1 {
                let ra = &a.px[y as usize * a.w + x0 as usize..y as usize * a.w + x1 as usize];
                let rb = &b.px[(y + dy) as usize * b.w + (x0 + dx) as usize..(y + dy) as usize * b.w + (x1 + dx) as usize];
                s += ra.iter().zip(rb).map(|(&p, &q)| (p as i32 - q as i32).unsigned_abs() as u64).sum::<u64>();
            }
            let m = s as f64 / ((x1 - x0) * (y1 - y0)) as f64;
            if m < best.0 || (m == best.0 && dx.abs() + dy.abs() < best.1.abs() + best.2.abs()) {
                best = (m, dx, dy);
            }
        }
    }
    (best.1, best.2)
}

/// One matched block: centre in `a` and its displacement into `b` (px at the 320 level).
#[derive(Debug, Clone, Copy)]
struct Flow {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
}

/// Block flow from `a` to `b` (same-size 320 px greys) with a 4-level pyramid: a global shift at
/// 1/8 scale (searched around `prior`, px at full scale), then per-block refinement at 1/2 and full
/// scale with a sub-pixel parabola. Returns (matched blocks, textured block count).
fn block_flow(a: &Grey, b: &Grey, prior: (f64, f64)) -> (Vec<Flow>, usize) {
    let pa = [a.clone(), a.half(), a.half().half(), a.half().half().half()];
    let pb = [b.clone(), b.half(), b.half().half(), b.half().half().half()];
    let p3 = ((prior.0 / 8.0).round() as i32, (prior.1 / 8.0).round() as i32);
    let g3 = global_shift(&pa[3], &pb[3], p3, 6);
    let g2 = global_shift(&pa[2], &pb[2], (g3.0 * 2, g3.1 * 2), 2);
    let mut flows = vec![];
    let mut textured = 0;
    for by in (0..a.h.saturating_sub(BLOCK - 1)).step_by(BLOCK) {
        for bx in (0..a.w.saturating_sub(BLOCK - 1)).step_by(BLOCK) {
            let mad = block_mad(a, bx, by, BLOCK);
            if mad < 4.0 {
                continue;
            }
            textured += 1;
            let (mut dx, mut dy) = (g2.0 * 2, g2.1 * 2);
            for level in [1usize, 0] {
                let (la, lb) = (&pa[level], &pb[level]);
                let (sx, sy, size) = (bx >> level, by >> level, BLOCK >> level);
                if level == 0 {
                    dx *= 2;
                    dy *= 2;
                }
                let r = if level == 1 { 3 } else { 2 };
                let mut best = (u32::MAX, dx, dy);
                for oy in -r..=r {
                    for ox in -r..=r {
                        if let Some(s) = sad(la, lb, sx, sy, size, dx + ox, dy + oy) {
                            if s < best.0 {
                                best = (s, dx + ox, dy + oy);
                            }
                        }
                    }
                }
                if best.0 == u32::MAX {
                    break;
                }
                dx = best.1;
                dy = best.2;
                if level == 0 {
                    // Reject weak matches: residual comparable to the block's own texture.
                    let per_px = best.0 as f64 / (BLOCK * BLOCK) as f64;
                    if per_px > 0.6 * mad + 2.0 {
                        break;
                    }
                    let sub = |m: Option<u32>, p: Option<u32>| match (m, p) {
                        (Some(m), Some(p)) => {
                            let (m, c, p) = (m as f64, best.0 as f64, p as f64);
                            let d = m - 2.0 * c + p;
                            if d > 0.0 { (0.5 * (m - p) / d).clamp(-0.5, 0.5) } else { 0.0 }
                        }
                        _ => 0.0,
                    };
                    let fx = sub(sad(la, lb, bx, by, BLOCK, dx - 1, dy), sad(la, lb, bx, by, BLOCK, dx + 1, dy));
                    let fy = sub(sad(la, lb, bx, by, BLOCK, dx, dy - 1), sad(la, lb, bx, by, BLOCK, dx, dy + 1));
                    flows.push(Flow { x: (bx + BLOCK / 2) as f64, y: (by + BLOCK / 2) as f64, dx: dx as f64 + fx, dy: dy as f64 + fy });
                }
            }
        }
    }
    (flows, textured)
}

/// Flows whose block, matched back from `b` into `a`, lands within a pixel of where it started:
/// drops mismatches (repetitive or faint texture) that would otherwise read as depth change.
fn consistent(flows: Vec<Flow>, a: &Grey, b: &Grey) -> Vec<Flow> {
    flows
        .into_iter()
        .filter(|f| {
            let (bx, by) = ((f.x as usize).saturating_sub(BLOCK / 2), (f.y as usize).saturating_sub(BLOCK / 2));
            let (ix, iy) = (f.dx.round() as i32, f.dy.round() as i32);
            let (cx, cy) = (bx as i32 + ix, by as i32 + iy);
            if cx < 0 || cy < 0 || cx as usize + BLOCK > b.w || cy as usize + BLOCK > b.h {
                return false;
            }
            let mut best = (u32::MAX, 0, 0);
            for oy in -2..=2 {
                for ox in -2..=2 {
                    if let Some(s) = sad(b, a, cx as usize, cy as usize, BLOCK, -ix + ox, -iy + oy) {
                        if s < best.0 {
                            best = (s, ox, oy);
                        }
                    }
                }
            }
            best.0 < u32::MAX && best.1.abs() <= 1 && best.2.abs() <= 1
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// Least-squares homography (h33 = 1) from >= 4 correspondences, in normalised coordinates.
fn fit_homography(pts: &[(f64, f64, f64, f64)]) -> Option<[f64; 9]> {
    if pts.len() < 4 {
        return None;
    }
    let mut ata = [[0f64; 8]; 8];
    let mut atb = [0f64; 8];
    for &(x, y, u, v) in pts {
        for (row, rhs) in [([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u), ([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v)] {
            for i in 0..8 {
                atb[i] += row[i] * rhs;
                for j in 0..8 {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
    }
    // Gaussian elimination with partial pivoting.
    let mut m = ata;
    let mut r = atb;
    for c in 0..8 {
        let p = (c..8).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[p][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, p);
        r.swap(c, p);
        for i in c + 1..8 {
            let f = m[i][c] / m[c][c];
            for j in c..8 {
                m[i][j] -= f * m[c][j];
            }
            r[i] -= f * r[c];
        }
    }
    let mut h = [0f64; 8];
    for c in (0..8).rev() {
        h[c] = (r[c] - (c + 1..8).map(|j| m[c][j] * h[j]).sum::<f64>()) / m[c][c];
    }
    Some([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0])
}

fn apply(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let w = h[6] * x + h[7] * y + h[8];
    ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
}

/// Residuals (px) of each flow against the best homography (RANSAC, then a refit on inliers).
/// Also returns the homography (in the normalised coordinates x / scale - 0.5).
fn homography_residuals(flows: &[Flow], scale: f64) -> (Vec<f64>, Option<[f64; 9]>) {
    if flows.len() < 8 {
        return (vec![], None);
    }
    // Normalise to ~unit coordinates for conditioning.
    let pts: Vec<(f64, f64, f64, f64)> =
        flows.iter().map(|f| (f.x / scale - 0.5, f.y / scale - 0.5, (f.x + f.dx) / scale - 0.5, (f.y + f.dy) / scale - 0.5)).collect();
    let resid = |h: &[f64; 9]| -> Vec<f64> {
        pts.iter()
            .map(|&(x, y, u, v)| {
                let (pu, pv) = apply(h, x, y);
                ((pu - u).powi(2) + (pv - v).powi(2)).sqrt() * scale
            })
            .collect()
    };
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let mut best: Option<(usize, [f64; 9])> = None;
    for _ in 0..300 {
        let idx = [next(pts.len()), next(pts.len()), next(pts.len()), next(pts.len())];
        if idx[0] == idx[1] || idx[0] == idx[2] || idx[0] == idx[3] || idx[1] == idx[2] || idx[1] == idx[3] || idx[2] == idx[3] {
            continue;
        }
        let Some(h) = fit_homography(&idx.map(|i| pts[i])) else { continue };
        let n = resid(&h).iter().filter(|&&r| r < 1.0).count();
        if best.as_ref().is_none_or(|b| n > b.0) {
            best = Some((n, h));
        }
    }
    let Some((_, h)) = best else { return (vec![], None) };
    let r = resid(&h);
    let inliers: Vec<_> = pts.iter().zip(&r).filter(|(_, r)| **r < 1.0).map(|(p, _)| *p).collect();
    let h = fit_homography(&inliers).unwrap_or(h);
    (resid(&h), Some(h))
}

fn pair_stats(t0: f64, t1: f64, shift: f64, flows: &[Flow], scale: f64) -> Pair {
    let (r, h) = homography_residuals(flows, scale);
    let off = if r.is_empty() { 0.0 } else { r.iter().filter(|&&x| x > PARALLAX_PX).count() as f64 / r.len() as f64 };
    let p80 = quantile(&r, 0.8);
    Pair { t0, t1, shift, blocks: r.len(), off_share: off, p80_px: p80, h }
}

/// Grey code (BT.709 luma of an HLG / PQ frame) -> displayed sRGB code, so light is judged as seen.
fn hdr_luma_lut(t: Transfer) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (v, o) in lut.iter_mut().enumerate() {
        let e = v as f64 / 255.0;
        let nits = match t {
            Transfer::Hlg => {
                let (a, b, c) = (0.178_832_77, 0.284_668_92, 0.559_910_7);
                let scene = if e <= 0.5 { e * e / 3.0 } else { (((e - c) / a).exp() + b) / 12.0 };
                1000.0 * scene.powf(1.2)
            }
            Transfer::Pq => {
                let (m1, m2) = (2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0);
                let (c1, c2, c3) = (3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0);
                let p = e.powf(1.0 / m2);
                ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1) * 10000.0
            }
        };
        let y = nits / 203.0;
        let white = 1000.0 / 203.0;
        let l = (y * (1.0 + y / (white * white)) / (1.0 + y)).clamp(0.0, 1.0);
        let s = if l <= 0.003_130_8 { 12.92 * l } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        *o = (s * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// The video's HDR transfer (HLG / PQ, so light is judged as displayed) and the lens's 35 mm
/// equivalent focal length when the file records it (Apple's QuickTime camera tags), from ffprobe.
fn probe_extra(ffprobe: &Path, source: &Path) -> (Option<Transfer>, Option<f64>, String, Option<f64>) {
    let j: Value = std::process::Command::new(ffprobe)
        .args(["-v", "error", "-select_streams", "V:0", "-show_entries", "stream=color_transfer:stream_tags:format_tags", "-of", "json"])
        .arg(source)
        .stdin(std::process::Stdio::null())
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice(&o.stdout).ok())
        .unwrap_or(Value::Null);
    let transfer = match j["streams"][0]["color_transfer"].as_str().unwrap_or("").trim() {
        "arib-std-b67" => Some(Transfer::Hlg),
        "smpte2084" => Some(Transfer::Pq),
        _ => None,
    };
    let focal35 = j["format"]["tags"]["com.apple.quicktime.camera.focal_length.35mm_equivalent"]
        .as_str()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .or_else(|| j["format"]["tags"]["com.apple.quicktime.camera.focal_length.35mm_equivalent"].as_f64())
        .filter(|f| (8.0..300.0).contains(f));
    let tags = &j["format"]["tags"];
    let model = ["com.android.model", "com.apple.quicktime.model", "model"].iter().find_map(|k| tags[*k].as_str()).unwrap_or("").trim().to_string();
    // Exposure time (s) when a container tag records it (none of the phone files we have do).
    let exposure = [&j["format"]["tags"], &j["streams"][0]["tags"]]
        .iter()
        .filter_map(|t| t.as_object())
        .flat_map(|t| t.iter())
        .filter(|(k, _)| k.to_lowercase().contains("exposure"))
        .find_map(|(_, v)| v.as_str().and_then(|v| v.trim().parse::<f64>().ok()).or_else(|| v.as_f64()))
        .filter(|t| *t > 0.0 && *t < 1.0);
    (transfer, focal35, model, exposure)
}

/// Main-camera video field of view across the short edge (degrees) for phones whose files carry a
/// model tag but no lens tag (a prefix of it: Samsung tags carry a region suffix, SM-G950F/U),
/// measured from SfM intrinsics or turn self-calibration on their videos.
const MODEL_SHORT_FOV: &[(&str, f64)] = &[
    // A portrait room clip (900 x 1600, 1x main lens): COLMAP f = 1435 px -> 34.8 deg; its own turns say
    // 58.0 deg long (32 pairs) -> 34.6. (The earlier 63.5 came from f = 870 px at 1920, 0.45x the
    // long side: the 0.5x ultrawide, which no model-only table can tell apart from the 1x.)
    ("Pixel 10 Pro", 34.8),
    // Landscape 1920 x 1080 room tours, 1x (self-calibrated from 47-117 turning
    // pairs per clip: 61-64 deg across the long edge). Galaxy S8/S8+, S9/S9+, S20 FE, A50.
    ("SM-G950", 37.3),
    ("SM-G955", 37.3),
    ("SM-G960", 37.8),
    ("SM-G965", 37.8),
    ("SM-G780", 37.0),
    ("SM-G781", 37.0),
    ("SM-A505", 36.0),
];

/// Field of view (degrees) across the long edge for a diagonal field of view.
fn long_fov_from_diag(w: u32, h: u32, diag_deg: f64) -> f64 {
    let (l, s) = (w.max(h) as f64, w.min(h) as f64);
    2.0 * ((diag_deg.to_radians() / 2.0).tan() * l / (l * l + s * s).sqrt()).atan().to_degrees()
}

/// The camera's field of view across the long edge, self-calibrated from keyframe homographies: a
/// turning camera maps views by H = K R K^-1, so the focal length is the one that makes K^-1 H K
/// closest to a rotation (square pixels, centred principal point). Median over pairs that turned
/// enough to constrain it; None when too few did. `w`, `h` are the analysis frame in units of its
/// long edge (one of them is 1).
/// Returns (long-edge FOV, pairs used).
fn estimate_long_fov(pairs: &[Pair], w: f64, h: f64) -> Option<(f64, usize)> {
    let (cx, cy) = (w / 2.0 - 0.5, h / 2.0 - 0.5);
    let cost = |hm: &[f64; 9], fov: f64| {
        let f = 0.5 / (fov.to_radians() / 2.0).tan();
        // M = K^-1 H K with K = [[f,0,cx],[0,f,cy],[0,0,1]].
        let k = [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]];
        let ki = [[1.0 / f, 0.0, -cx / f], [0.0, 1.0 / f, -cy / f], [0.0, 0.0, 1.0]];
        let hh = [[hm[0], hm[1], hm[2]], [hm[3], hm[4], hm[5]], [hm[6], hm[7], hm[8]]];
        let mul = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| {
            let mut c = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    c[i][j] = (0..3).map(|t| a[i][t] * b[t][j]).sum();
                }
            }
            c
        };
        let m = mul(mul(ki, hh), k);
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det <= 0.0 {
            return f64::MAX;
        }
        let g = det.cbrt();
        let mut e = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let mtm: f64 = (0..3).map(|t| m[t][i] * m[t][j]).sum::<f64>() / (g * g);
                e += (mtm - if i == j { 1.0 } else { 0.0 }).powi(2);
            }
        }
        e
    };
    let mut best: Vec<f64> = pairs
        .iter()
        .filter(|p| p.shift >= FOV_PAIR_SHIFT && p.blocks >= 20)
        .filter_map(|p| {
            let hm = p.h.as_ref()?;
            let grid = (FOV_MIN_DEG as u32..=FOV_MAX_DEG as u32).map(|d| (d as f64, cost(hm, d as f64)));
            let (fov, c) = grid.min_by(|a, b| a.1.total_cmp(&b.1))?;
            // An edge of the search range means unconstrained (a translation, not a turn).
            (c.is_finite() && fov > FOV_MIN_DEG && fov < FOV_MAX_DEG).then_some(fov)
        })
        .collect();
    let n = best.len();
    (n >= FOV_MIN_PAIRS).then(|| (median(&mut best), n))
}

/// Motion-blur length (px) along x (`horizontal`) or y: a box smear of L px makes the derivative
/// along the motion anti-correlated at lag L (the two ends of the box), so the most negative
/// normalised autocorrelation of the derivative in lags 2..=MAX_BLUR_PX gives L. None when there is
/// no clear dip (no measurable smear, or too little texture).
fn blur_length(g: &Grey, horizontal: bool) -> Option<f64> {
    let (n, lines) = if horizontal { (g.w, g.h) } else { (g.h, g.w) };
    let at = |line: usize, i: usize| if horizontal { g.px[line * g.w + i] } else { g.px[i * g.w + line] } as f64;
    if n < MAX_BLUR_PX * 4 {
        return None;
    }
    let mut r = vec![0f64; MAX_BLUR_PX + 1];
    for line in (0..lines).step_by(3) {
        let d: Vec<f64> = (0..n - 1).map(|i| at(line, i + 1) - at(line, i)).collect();
        for (lag, acc) in r.iter_mut().enumerate() {
            *acc += d[..d.len() - lag].iter().zip(&d[lag..]).map(|(a, b)| a * b).sum::<f64>();
        }
    }
    if r[0] <= 0.0 {
        return None;
    }
    let (lag, v) = (2..=MAX_BLUR_PX).map(|l| (l, r[l] / r[0])).min_by(|a, b| a.1.total_cmp(&b.1))?;
    (v < -BLUR_DIP).then_some(lag as f64)
}

/// Runs of `true` at least `min` long, as [start, end) sample indices.
fn runs(flags: &[bool], min: usize) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut start = None;
    for (k, &f) in flags.iter().chain([false].iter()).enumerate() {
        match (f, start) {
            (true, None) => start = Some(k),
            (false, Some(a)) => {
                if k - a >= min {
                    out.push((a, k));
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// Seconds as m:ss.
fn clock(t: f64) -> String {
    let s = t.round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// One decoded sample: its per-frame statistics, the 320 px grey for flow and the 160 px motion grey.
struct Decoded {
    sample: Sample,
    half: Grey,
    small: Vec<u8>,
    /// Measured samples: the 640 px grey and its corners, for matching to the next measured sample.
    feat: Option<(Grey, Vec<(u16, u16)>)>,
}

/// Corners: Shi-Tomasi minimum eigenvalue of the 5x5 gradient structure tensor at or above
/// CORNER_MIN_EIG (both directions with ~8 grey levels a pixel), local maxima in 7x7, strongest
/// MAX_CORNERS.
fn corners(g: &Grey) -> Vec<(u16, u16)> {
    let (w, h) = (g.w, g.h);
    if w < 16 || h < 16 {
        return vec![];
    }
    let p = |x: usize, y: usize| g.px[y * w + x] as f32;
    let mut ixx = vec![0f32; w * h];
    let mut iyy = vec![0f32; w * h];
    let mut ixy = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = (p(x + 1, y - 1) + 2.0 * p(x + 1, y) + p(x + 1, y + 1) - p(x - 1, y - 1) - 2.0 * p(x - 1, y) - p(x - 1, y + 1)) / 8.0;
            let gy = (p(x - 1, y + 1) + 2.0 * p(x, y + 1) + p(x + 1, y + 1) - p(x - 1, y - 1) - 2.0 * p(x, y - 1) - p(x + 1, y - 1)) / 8.0;
            ixx[y * w + x] = gx * gx;
            iyy[y * w + x] = gy * gy;
            ixy[y * w + x] = gx * gy;
        }
    }
    // 5x5 box sums, separably (rows then columns).
    let box5 = |v: &[f32]| {
        let mut r = vec![0f32; w * h];
        for y in 0..h {
            for x in 2..w - 2 {
                r[y * w + x] = v[y * w + x - 2] + v[y * w + x - 1] + v[y * w + x] + v[y * w + x + 1] + v[y * w + x + 2];
            }
        }
        let mut c = vec![0f32; w * h];
        for y in 2..h - 2 {
            for x in 0..w {
                c[y * w + x] = r[(y - 2) * w + x] + r[(y - 1) * w + x] + r[y * w + x] + r[(y + 1) * w + x] + r[(y + 2) * w + x];
            }
        }
        c
    };
    let (sa, sb, sc) = (box5(&ixx), box5(&ixy), box5(&iyy));
    let mut eig = vec![0f32; w * h];
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let (a, b, c) = (sa[y * w + x], sb[y * w + x], sc[y * w + x]);
            eig[y * w + x] = (a + c) / 2.0 - (((a - c) / 2.0).powi(2) + b * b).sqrt();
        }
    }
    let mut found = vec![];
    for y in 8..h - 8 {
        for x in 8..w - 8 {
            let e = eig[y * w + x];
            if e < CORNER_MIN_EIG {
                continue;
            }
            let is_max = (y - 3..=y + 3).all(|yy| (x - 3..=x + 3).all(|xx| eig[yy * w + xx] <= e || (xx == x && yy == y)));
            if is_max {
                found.push((e, x as u16, y as u16));
            }
        }
    }
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    found.into_iter().take(MAX_CORNERS).map(|(_, x, y)| (x, y)).collect()
}

/// Corners of `a` found again in `b` (normalised cross-correlation >= MATCH_NCC of a 9x9 patch within
/// +-4 px of the flow-predicted spot `shift`, px).
fn match_corners(a: &Grey, pts: &[(u16, u16)], b: &Grey, shift: (f64, f64)) -> u32 {
    let patch = |g: &Grey, cx: i32, cy: i32| -> Option<Vec<f32>> {
        if cx < 4 || cy < 4 || cx + 4 >= g.w as i32 || cy + 4 >= g.h as i32 {
            return None;
        }
        let mut v: Vec<f32> = (-4..=4).flat_map(|dy| (-4..=4).map(move |dx| (dx, dy))).map(|(dx, dy)| g.px[(cy + dy) as usize * g.w + (cx + dx) as usize] as f32).collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        v.iter_mut().for_each(|x| *x -= m);
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        (n > 1e-3).then(|| v.into_iter().map(|x| x / n).collect())
    };
    let (sx, sy) = (shift.0.round() as i32, shift.1.round() as i32);
    pts.iter()
        .filter(|&&(x, y)| {
            let Some(pa) = patch(a, x as i32, y as i32) else { return false };
            let mut best = -1f32;
            for oy in -4..=4 {
                for ox in -4..=4 {
                    if let Some(pb) = patch(b, x as i32 + sx + ox, y as i32 + sy + oy) {
                        best = best.max(pa.iter().zip(&pb).map(|(p, q)| p * q).sum());
                    }
                }
            }
            best >= MATCH_NCC
        })
        .count() as u32
}


/// Motion track, consecutive-sample flow and keyframe pairs over contiguous samples starting at t0 (s):
/// (samples with flow filled in, pairs, total travel in image widths).
fn analyse(mut frames: Vec<Decoded>, gw: usize, gh: usize, scale: f64, n_blocks: usize, t0: f64) -> Result<(Vec<Sample>, Vec<Pair>, f64)> {
    if frames.is_empty() {
        return Ok((vec![], vec![], 0.0));
    }
    let mut tracker = MotionTracker::new(gw, gh);
    for f in &frames {
        tracker.push(&f.small);
    }
    let motion_widths: f64 = tracker.finish()?.iter().map(|t| t.1).sum();
    // Consecutive-sample flow (pan speed, yaw, tracking), in parallel.
    use rayon::prelude::*;
    let steps: Vec<(f64, f64, f64, f64)> = (1..frames.len())
        .into_par_iter()
        .map(|k| {
            let (flows, textured) = block_flow(&frames[k - 1].half, &frames[k].half, (0.0, 0.0));
            let mut dxs: Vec<f64> = flows.iter().map(|f| f.dx).collect();
            let mut dys: Vec<f64> = flows.iter().map(|f| f.dy).collect();
            let (dx, dy) = if flows.len() >= 4 { (median(&mut dxs) / scale, median(&mut dys) / scale) } else { (0.0, 0.0) };
            let tracked = if textured > 0 { flows.len() as f64 / textured as f64 } else { 0.0 };
            (dx, dy, tracked, textured as f64 / n_blocks.max(1) as f64)
        })
        .collect();
    for (k, &(dx, dy, tracked, textured)) in steps.iter().enumerate() {
        let s = &mut frames[k + 1].sample;
        (s.dx, s.dy, s.tracked, s.textured) = (dx, dy, tracked, textured);
    }
    // Corner matches from each measured sample to the next one, at the flow-predicted shift (640 px).
    let measured: Vec<usize> = (0..frames.len()).filter(|&k| frames[k].feat.is_some()).collect();
    let counts: Vec<(usize, u32)> = measured
        .par_windows(2)
        .map(|w| {
            let (i, j) = (w[0], w[1]);
            let shift = frames[i + 1..=j].iter().fold((0.0, 0.0), |a, f| (a.0 + f.sample.dx * BIG_EDGE as f64, a.1 + f.sample.dy * BIG_EDGE as f64));
            let (ga, pa) = frames[i].feat.as_ref().unwrap();
            (i, match_corners(ga, pa, &frames[j].feat.as_ref().unwrap().0, shift))
        })
        .collect();
    for (i, n) in counts {
        frames[i].sample.matches = Some(n);
    }
    for f in &mut frames {
        f.feat = None;
    }
    // Keyframe pairs ~PAIR_SHIFT apart for parallax: schedule from the flow, then fit in parallel.
    let mut schedule = vec![];
    let (mut key, mut acc) = (0usize, (0.0f64, 0.0f64));
    for i in 1..frames.len() {
        acc.0 += frames[i].sample.dx;
        acc.1 += frames[i].sample.dy;
        let shift = (acc.0 * acc.0 + acc.1 * acc.1).sqrt();
        if shift >= PAIR_SHIFT || i - key >= PAIR_MAX_SAMPLES {
            schedule.push((key, i, acc, shift));
            (key, acc) = (i, (0.0, 0.0));
        }
    }
    let pairs: Vec<Pair> = schedule
        .par_iter()
        .map(|&(k, i, acc, shift)| {
            let (flows, _) = block_flow(&frames[k].half, &frames[i].half, (acc.0 * scale, acc.1 * scale));
            let flows = consistent(flows, &frames[k].half, &frames[i].half);
            pair_stats(t0 + k as f64 / ANALYSIS_FPS, t0 + i as f64 / ANALYSIS_FPS, shift, &flows, scale)
        })
        .collect();
    Ok((frames.into_iter().map(|f| f.sample).collect(), pairs, motion_widths))
}

/// The sparse decode's windows (start s, samples) on one ANALYSIS_FPS timeline over `duration`: each
/// window analysed on its own (flow and keyframe pairs never span a gap); a gap's slots copy the nearest
/// decoded sample (light, sharpness, tracking) with the image motion interpolated between the turn
/// rates at the two window edges, so turn speed, yaw and coverage stay continuous. Travel is scaled by
/// the decoded share of the clip.
fn sparse_timeline(windows: Vec<(f64, Vec<Decoded>)>, duration: f64, gw: usize, gh: usize, scale: f64, n_blocks: usize) -> Result<(Vec<Sample>, Vec<Pair>, f64)> {
    let n_slots = ((duration * ANALYSIS_FPS).ceil() as usize).max(1);
    let mut slots: Vec<Option<Sample>> = vec![None; n_slots];
    let mut pairs = vec![];
    let (mut motion, mut covered) = (0.0, 0usize);
    for (t0, frames) in windows {
        let first = (t0 * ANALYSIS_FPS).round() as usize;
        let (samples, p, m) = analyse(frames, gw, gh, scale, n_blocks, t0)?;
        motion += m;
        pairs.extend(p);
        for (i, mut smp) in samples.into_iter().enumerate() {
            if i == 0 {
                // No flow into a window's first sample (its predecessor is in the gap): filled below.
                (smp.dx, smp.dy) = (f64::NAN, f64::NAN);
            }
            if let Some(slot) = slots.get_mut(first + i) {
                *slot = Some(smp);
                covered += 1;
            }
        }
    }
    let known: Vec<usize> = (0..n_slots).filter(|&k| slots[k].as_ref().is_some_and(|s| s.dx.is_finite())).collect();
    ensure!(!known.is_empty(), "ffmpeg decoded no frames for the capture check");
    // Turn rate (image motion per sample) near slot k from up to 8 decoded samples on one side.
    let rate = |k: usize, before: bool| {
        let near: Vec<&Sample> = if before {
            known.iter().rev().filter(|&&j| j < k).take(8).filter_map(|&j| slots[j].as_ref()).collect()
        } else {
            known.iter().filter(|&&j| j > k).take(8).filter_map(|&j| slots[j].as_ref()).collect()
        };
        (!near.is_empty()).then(|| (near.iter().map(|s| s.dx).sum::<f64>() / near.len() as f64, near.iter().map(|s| s.dy).sum::<f64>() / near.len() as f64))
    };
    let mut out = Vec::with_capacity(n_slots);
    for k in 0..n_slots {
        let measured = slots[k].as_ref().is_some_and(|s| s.dx.is_finite());
        if measured {
            out.push(slots[k].clone().unwrap());
            continue;
        }
        let nearest = *known.iter().min_by_key(|&&j| j.abs_diff(k)).unwrap();
        let mut smp = slots[k].clone().unwrap_or_else(|| slots[nearest].clone().unwrap());
        if slots[k].is_none() {
            (smp.blur_x, smp.blur_y) = (None, None);
        }
        let (prev, next) = (rate(k, true), rate(k, false));
        let (pk, nk) = (known.iter().rev().find(|&&j| j < k).copied(), known.iter().find(|&&j| j > k).copied());
        (smp.dx, smp.dy) = match (prev, next, pk, nk) {
            (Some(a), Some(b), Some(i), Some(j)) => {
                let w = (k - i) as f64 / (j - i) as f64;
                (a.0 + (b.0 - a.0) * w, a.1 + (b.1 - a.1) * w)
            }
            (Some(a), _, _, _) | (None, Some(a), _, _) => a,
            _ => (0.0, 0.0),
        };
        out.push(smp);
    }
    let travel = motion * n_slots as f64 / covered.max(1) as f64;
    Ok((out, pairs, travel))
}

/// Runs the check on a video file.
pub fn check_video(ffmpeg: &Path, source: &Path, opts: &CheckOptions) -> Result<CaptureCheck> {
    let start = Instant::now();
    let ffprobe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
    let probe = crate::video::probe(source, &ffprobe)?;
    let (width, height) = if probe.rotation_deg.rem_euclid(180) == 90 { (probe.height, probe.width) } else { (probe.width, probe.height) };
    ensure!(width > 0 && height > 0, "video dimensions are unavailable");
    let (transfer, focal35, model, exposure_tag) = probe_extra(&ffprobe, source);
    let lut = transfer.map(hdr_luma_lut);
    let t_probe = start.elapsed().as_secs_f64();

    // Big frame: long edge 640 (even), and the motion track's own 160 px grey under it.
    let s = BIG_EDGE as f64 / width.max(height) as f64;
    let bw = ((width as f64 * s).round() as usize).max(32) & !1;
    let bh = ((height as f64 * s).round() as usize).max(32) & !1;
    let (gw, gh) = frameplan::analysis_size(width, height);
    ensure!(bw >= gw, "analysis frame narrower than the motion track's");
    let filter = format!(
        "fps={ANALYSIS_FPS},split=2[a][b];[a]scale={bw}:{bh}:flags=area,format=gray[c];[b]scale={gw}:{gh}:flags=area,format=gray,pad={bw}:{gh}[d];[c][d]vstack=inputs=2"
    );
    let frame_bytes = bw * (bh + gh);
    // Decode in parallel time segments (each seeks to its start): a 2-min 4K clip is decode-bound.
    // At most DECODE_THREADS decoder threads in all, so the check stays light on a busy machine.
    // One decoder thread per segment: on an M4, 2 min of 4K30 HEVC took 14.1 s as 10 x 1 vs 16.5 s
    // as 4 x 2 (frame threads scale worse than independent segments).
    let budget = DECODE_THREADS.min(std::thread::available_parallelism().map_or(4, |n| n.get()));
    // Long or 4K clips (a full decode well over ~10 s on a laptop) decode SPARSE_WINDOW_S windows every
    // SPARSE_PERIOD_S instead: about half the frames. Gaps are filled from the windows (see below).
    let pixel_frames = width as f64 * height as f64 * probe.fps.max(1.0) * probe.duration_s;
    let sparse = pixel_frames > SPARSE_PIXEL_FRAMES && probe.duration_s > 2.0 * SPARSE_PERIOD_S;
    let segments = ((probe.duration_s / 10.0).ceil() as usize).clamp(1, budget);
    let bounds: Vec<Option<(f64, f64)>> = if sparse {
        let n = (probe.duration_s / SPARSE_PERIOD_S).ceil() as usize;
        (0..n).map(|k| k as f64 * SPARSE_PERIOD_S).map(|a| Some((a, (a + SPARSE_WINDOW_S).min(probe.duration_s)))).collect()
    } else if segments == 1 || probe.duration_s <= 0.0 {
        vec![None]
    } else {
        (0..segments).map(|k| Some((probe.duration_s * k as f64 / segments as f64, probe.duration_s * (k + 1) as f64 / segments as f64))).collect()
    };
    let threads = (budget / bounds.len()).max(1);
    // At most `budget` decoders at once (sparse clips can have more windows than that).
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<Result<Vec<Decoded>>>>> = bounds.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..budget.min(bounds.len()) {
            let (filter, lut, next, slots, bounds) = (&filter, &lut, &next, &slots, &bounds);
            scope.spawn(move || loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&range) = bounds.get(k) else { break };
                let result = (|| -> Result<Vec<Decoded>> {
                    let mut out = vec![];
                    frameplan::decode_sampled(ffmpeg, source, range, threads, filter, "gray", frame_bytes, |_, px| {
                        // HDR: every measure runs on the displayed (tone-mapped) grey, so an HLG video and its
                        // SDR transcode read the same (raw HLG codes shifted one room clip's turn estimate ~5%).
                        let map = |v: u8| lut.as_ref().map_or(v, |l| l[v as usize]);
                        let big = Grey { w: bw, h: bh, px: px[..bw * bh].iter().map(|&v| map(v)).collect() };
                        let small: Vec<u8> = (0..gh).flat_map(|y| px[(bh + y) * bw..(bh + y) * bw + gw].iter().map(|&v| map(v))).collect();
                        let luma = big.px.iter().map(|&v| v as f64).sum::<f64>() / big.px.len() as f64;
                        let (lap, gx, gy) = texture_stats(&big);
                        let half = big.half();
                        let (lap_half, _, _) = texture_stats(&half);
                        let measure = out.len() % SMEAR_EVERY == 0;
                        let pts = if measure { corners(&big) } else { vec![] };
                        let sample = Sample {
                            luma,
                            sharp: lap,
                            detail: if lap_half > 1e-6 { lap / lap_half } else { 0.0 },
                            smear: if gx > 1e-6 { gy / gx } else { 1.0 },
                            blur_x: if measure { blur_length(&big, true) } else { None },
                            blur_y: if measure { blur_length(&big, false) } else { None },
                            corners: measure.then_some(pts.len() as u32),
                            ..Default::default()
                        };
                        out.push(Decoded { sample, half, small, feat: measure.then_some((big, pts)) });
                        Ok(())
                    })?;
                    Ok(out)
                })();
                *slots[k].lock().unwrap() = Some(result);
            });
        }
    });
    let mut windows: Vec<(f64, Vec<Decoded>)> = vec![];
    for (b, slot) in bounds.iter().zip(slots) {
        let d = slot.into_inner().unwrap().unwrap_or_else(|| Err(anyhow::anyhow!("capture check decoder panicked")));
        windows.push((b.map_or(0.0, |r| r.0), d.with_context(|| format!("decoding {} for the capture check", source.display()))?));
    }
    ensure!(windows.iter().any(|w| !w.1.is_empty()), "ffmpeg decoded no frames from {}", source.display());
    let t_decode = start.elapsed().as_secs_f64() - t_probe;
    let scale = (bw.max(bh) / 2) as f64;
    let n_blocks = ((bw / 2) / BLOCK) * ((bh / 2) / BLOCK);
    let (samples, pairs, motion_widths) = if sparse {
        sparse_timeline(windows, probe.duration_s, gw, gh, scale, n_blocks)?
    } else {
        analyse(windows.into_iter().flat_map(|w| w.1).collect(), gw, gh, scale, n_blocks, 0.0)?
    };
    let t_flow = start.elapsed().as_secs_f64() - t_probe - t_decode;
    let t_pairs = 0.0;

    let duration = probe.duration_s.max(samples.len() as f64 / ANALYSIS_FPS);
    // Field of view: the file's lens tag, else self-calibrated from the turns, else a 1x phone lens.
    // Field of view: the file's lens tag, else the phone model's measured value, else self-calibrated
    // from the turns, else a 1x phone lens.
    let ratio = width.max(height) as f64 / width.min(height) as f64;
    let model_fov = MODEL_SHORT_FOV.iter().find(|(m, _)| !model.is_empty() && model.to_ascii_uppercase().starts_with(&m.to_ascii_uppercase())).map(|&(_, short)| {
        2.0 * ((short.to_radians() / 2.0).tan() * ratio).atan().to_degrees()
    });
    let estimated = estimate_long_fov(&pairs, bw as f64 / bw.max(bh) as f64, bh as f64 / bw.max(bh) as f64);
    let fov = match (focal35, model_fov, estimated) {
        (Some(f), _, _) => (long_fov_from_diag(width, height, 2.0 * (43.27 / 2.0 / f).atan().to_degrees()), "lens tag"),
        (None, Some(f), _) => (f, "phone model"),
        (None, None, Some((f, _))) => (f, "estimated"),
        _ => (long_fov_from_diag(width, height, DIAG_FOV_DEG), "assumed 1x lens"),
    };
    let mut c = judge(&samples, &pairs, motion_widths, duration, (width, height), fov, probe.fps, exposure_tag, &probe.codec, transfer, opts, start);
    c.metrics["model"] = json!(model);
    c.metrics["fov_estimate"] = json!(estimated.map(|(f, n)| json!({"long_deg": round2(f), "pairs": n})));
    if std::env::var("SPLAT_CHECK_DEBUG").as_deref() == Ok("1") {
        c.metrics["debug"] = json!({"samples": samples, "pairs": pairs});
    }
    c.metrics["timing_s"] = json!({"probe": round2(t_probe), "decode": round2(t_decode), "flow": round2(t_flow), "parallax": round2(t_pairs),
        "segments": bounds.len()});
    Ok(c)
}

#[allow(clippy::too_many_arguments)]
fn judge(
    samples: &[Sample],
    pairs: &[Pair],
    motion_widths: f64,
    duration: f64,
    (width, height): (u32, u32),
    (fov_long, fov_source): (f64, &str),
    source_fps: f64,
    exposure_tag: Option<f64>,
    codec: &str,
    transfer: Option<Transfer>,
    opts: &CheckOptions,
    start: Instant,
) -> CaptureCheck {
    let n = samples.len().max(1) as f64;
    let dt = 1.0 / ANALYSIS_FPS;
    let ratio = width.min(height) as f64 / width.max(height) as f64;
    let fov_short = 2.0 * ((fov_long.to_radians() / 2.0).tan() * ratio).atan().to_degrees();
    // Focal length in units of the long edge (dx, dy are fractions of it).
    let focal = 0.5 / (fov_long.to_radians() / 2.0).tan();
    // dx/dy are fractions of the long edge; the long edge is horizontal in a landscape frame.
    let (fov_x, fov_y) = if width >= height { (fov_long, fov_short) } else { (fov_short, fov_long) };
    // Per-sample pan speed (deg/s) and cumulative yaw (turning right = content moves left).
    let mut yaw = vec![0.0f64; samples.len()];
    let mut speed = vec![0.0f64; samples.len()];
    for (k, s) in samples.iter().enumerate().skip(1) {
        let yx = -(s.dx / focal).atan().to_degrees();
        let py = (s.dy / focal).atan().to_degrees();
        yaw[k] = yaw[k - 1] + yx;
        speed[k] = (yx * yx + py * py).sqrt() / dt;
    }
    let lumas: Vec<f64> = samples.iter().map(|s| s.luma).collect();
    let luma_med = quantile(&lumas, 0.5);
    let dim_share = lumas.iter().filter(|&&l| l < DIM_LUMA).count() as f64 / n;
    let dark_share = lumas.iter().filter(|&&l| l < 30.0).count() as f64 / n;
    let sharp: Vec<f64> = samples.iter().map(|s| s.sharp).collect();
    let sharp_med = quantile(&sharp, 0.5);
    let blurred_share = sharp.iter().filter(|&&v| v < 0.5 * sharp_med).count() as f64 / n;
    // Spikes: below half the median of the +-10 neighbouring samples (same content).
    let spikes = (0..sharp.len())
        .filter(|&k| {
            let w = &sharp[k.saturating_sub(10)..(k + 11).min(sharp.len())];
            sharp[k] < 0.5 * quantile(w, 0.5)
        })
        .count();
    let moving: Vec<f64> = speed.iter().skip(1).copied().collect();
    let speed_med = quantile(&moving, 0.5);
    let speed_p90 = quantile(&moving, 0.9);
    // Sharpness when slow vs overall: how much the motion costs.
    let mut by_speed: Vec<(f64, f64)> = speed.iter().zip(&sharp).map(|(&a, &b)| (a, b)).skip(1).collect();
    by_speed.sort_by(|a, b| a.0.total_cmp(&b.0));
    let slow_n = (by_speed.len() / 5).max(1);
    let slow_sharp = quantile(&by_speed.iter().take(slow_n).map(|p| p.1).collect::<Vec<_>>(), 0.5);
    let fast_sharp = quantile(&by_speed.iter().rev().take(slow_n).map(|p| p.1).collect::<Vec<_>>(), 0.5);
    let motion_blur_ratio = if slow_sharp > 0.0 { fast_sharp / slow_sharp } else { 1.0 };
    let detail_med = quantile(&samples.iter().map(|s| s.detail).collect::<Vec<_>>(), 0.5);
    let smear_med = quantile(&samples.iter().map(|s| s.smear).collect::<Vec<_>>(), 0.5);
    let textured_med = quantile(&samples.iter().skip(1).map(|s| s.textured).collect::<Vec<_>>(), 0.5);
    let tracked: Vec<f64> = samples.iter().skip(1).filter(|s| s.textured > 0.05).map(|s| s.tracked).collect();
    let lost_share = tracked.iter().filter(|&&t| t < 0.3).count() as f64 / tracked.len().max(1) as f64;
    // Longest run of lost samples, seconds.
    let mut lost_run = 0usize;
    let mut cur = 0usize;
    for s in samples.iter().skip(1) {
        if s.textured > 0.05 && s.tracked < 0.3 {
            cur += 1;
            lost_run = lost_run.max(cur);
        } else {
            cur = 0;
        }
    }
    let lost_run_s = lost_run as f64 * dt;
    // Depth change: the median share of blocks off each keyframe pair's homography (0 = the camera
    // only turned, or saw one flat surface). Unknown with fewer than 2 usable pairs.
    let fitted: Vec<&Pair> = pairs.iter().filter(|p| p.blocks >= 12).collect();
    let depth_change = (fitted.len() >= 2).then(|| quantile(&fitted.iter().map(|p| p.off_share).collect::<Vec<_>>(), 0.5));
    // SfM needs one good initial pair: the best pair decides the failure.
    let depth_best = (fitted.len() >= 2).then(|| fitted.iter().map(|p| p.off_share).fold(0.0, f64::max));
    let (yaw_min, yaw_max) = yaw.iter().fold((0.0f64, 0.0f64), |a, &y| (a.0.min(y), a.1.max(y)));
    let yaw_span = yaw_max - yaw_min;
    let mut sectors = [0.0f64; 8];
    for &y in &yaw {
        sectors[((y.rem_euclid(360.0) + 22.5) / 45.0) as usize % 8] += dt;
    }

    let mut checks = vec![];
    let mut add = |id: &'static str, name: &'static str, verdict: Verdict, message: String, fix: &str| {
        checks.push(Check { id, name, verdict, message, fix: if verdict == Verdict::Pass { String::new() } else { fix.to_string() } });
    };

    // Length.
    let short_warn = if opts.room { 30.0 } else { 10.0 };
    let (v, m) = if duration < 3.0 {
        (Verdict::Fail, format!("Too short: {duration:.1} s."))
    } else if duration < short_warn {
        (Verdict::Tip, format!("Short: {duration:.0} s."))
    } else if duration > LONG_WARN_S {
        (Verdict::Warn, format!("Long: {duration:.0} s. 1 minute is enough; a longer clip only takes longer to process."))
    } else {
        (Verdict::Pass, format!("{duration:.0} s."))
    };
    add("length", "Length", v, m, if opts.room { "Film for about 1 minute: a slow loop of the whole room." } else { "Film for at least 20-30 s, moving slowly around the subject." });

    // Light.
    let (v, m) = if luma_med < BLACK_LUMA {
        (Verdict::Fail, format!("Almost black: median brightness {luma_med:.0}/255."))
    } else if luma_med < DIM_LUMA || dim_share > 0.5 {
        (Verdict::Tip, format!("Dim: {:.0}% of the video is dark (median brightness {luma_med:.0}/255; aim for 100+).", dim_share * 100.0))
    } else {
        (Verdict::Pass, format!("Bright enough (median {luma_med:.0}/255{}).", if dim_share > 0.1 { format!(", {:.0}% dim", dim_share * 100.0) } else { String::new() }))
    };
    add("light", "Light", v, m, "Turn on every light and open the curtains; dim video blurs and gets noisy.");

    // Stretches (>= SEGMENT_S) where the camera turns fast or most frames blur: an average hides them
    // (one room clip: median 3.4 deg/s, but a 12.7 s ending around 20 deg/s).
    let window = (SEGMENT_S * ANALYSIS_FPS) as usize;
    let smooth_speed: Vec<f64> = (0..speed.len()).map(|k| quantile(&speed[k.saturating_sub(4)..(k + 5).min(speed.len())], 0.5)).collect();
    let fast_runs = runs(&smooth_speed.iter().map(|&v| v > SEGMENT_FAST_DEG_S).collect::<Vec<_>>(), window);
    let blurred: Vec<bool> = sharp.iter().map(|&v| v < 0.5 * sharp_med).collect();
    let blur_density: Vec<bool> = (0..blurred.len())
        .map(|k| {
            let w = &blurred[k.saturating_sub(window / 2)..(k + window / 2).min(blurred.len())];
            w.iter().filter(|&&b| b).count() * 2 > w.len()
        })
        .collect();
    let blur_runs = runs(&blur_density, window);
    // Motion smear in px of a 1080p frame, along each sample's direction of motion (unmeasured = 0),
    // capped by what the motion allows in one frame time (turn rate x 1/fps x focal length): texture
    // and compression alone read 6-9 px on still clips, but cannot be motion smear there.
    let to_1080 = 1920.0 / BIG_EDGE as f64;
    let focal_1080 = 960.0 / (fov_long.to_radians() / 2.0).tan();
    let frame_s = 1.0 / if source_fps > 1.0 { source_fps } else { 30.0 };
    let smear_px: Vec<Option<f64>> = samples
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let along = if s.dx.abs() >= s.dy.abs() { s.blur_x } else { s.blur_y };
            let most = (speed[k] * frame_s).to_radians() * focal_1080;
            (s.blur_x.is_some() || s.blur_y.is_some()).then(|| (along.unwrap_or(0.0) * to_1080).min(most))
        })
        .collect();
    let smear_smooth: Vec<bool> = (0..smear_px.len())
        .map(|k| {
            let w: Vec<f64> = smear_px[k.saturating_sub(window / 2)..(k + window / 2).min(smear_px.len())].iter().flatten().copied().collect();
            !w.is_empty() && quantile(&w, 0.5) > SMEAR_WARN_PX
        })
        .collect();
    let smear_runs = runs(&smear_smooth, window);
    // Estimated motion blur per frame (px at 1080p) = turn rate x focal length x exposure time. The
    // exposure comes from a container tag when there is one, else from brightness: a dim video has the
    // shutter open for the whole frame time (auto-exposure at its limit), a bright one about half.
    let (exposure, exposure_source) = match exposure_tag {
        Some(t) => (t.min(frame_s), "tag"),
        None if luma_med < EXPOSURE_FULL_FRAME_LUMA => (frame_s, "estimated (dim: whole frame time)"),
        None => (frame_s / 2.0, "estimated (bright: half the frame time)"),
    };
    let blur_est: Vec<f64> = speed.iter().map(|&v| v.to_radians() * focal_1080 * exposure).collect();
    let blur_est_med = quantile(&blur_est[1.min(blur_est.len())..], 0.5);
    let blur_local: Vec<f64> = (0..blur_est.len()).map(|k| quantile(&blur_est[k.saturating_sub(window / 2)..(k + window / 2).min(blur_est.len())], 0.5)).collect();
    // Tip stretches (>= TIP px for >= SEGMENT_S); one whose median reaches WARN px is a warning, so the
    // tier always matches the number shown.
    let tip_all = runs(&blur_local.iter().map(|&b| b >= BLUR_EST_TIP_PX).collect::<Vec<_>>(), window);
    let (blur_est_runs, blur_tip_runs): (Vec<_>, Vec<_>) = tip_all.into_iter().partition(|&(a, b)| quantile(&blur_est[a..b], 0.5) >= BLUR_EST_WARN_PX);
    let measured: Vec<f64> = smear_px.iter().flatten().copied().collect();
    let smear_px_med = quantile(&measured, 0.5);
    let ranges = |rs: &[(usize, usize)], what: &dyn Fn(usize, usize) -> String| {
        rs.iter().take(3).map(|&(a, b)| format!("{}-{} {}", clock(a as f64 * dt), clock(b as f64 * dt), what(a, b))).collect::<Vec<_>>().join(", ")
    };

    // Sharpness (blur from moving too fast is reported under speed). Only lots of blur warns (on the
    // dev clips: >= BLUR_WARN_TOTAL_S of blurred stretches went with worse results; short stretches,
    // measured smear and a blurry opening alone did not), the rest is a tip.
    let blur_total_s: f64 = blur_runs.iter().map(|&(a, b)| (b - a) as f64 * dt).sum();
    let blurry_in_places = |v: Verdict| {
        // A severe blur in the opening seconds matters more: the quick first view is built from them
        // (hlg-iphone16-oakland: 0:04-0:09 at 0.06-0.14x the clip's median sharpness broke it).
        let opening = blur_runs.iter().find(|&&(a, b)| (a as f64 * dt) < OPENING_S && quantile(&sharp[a..b], 0.5) < SEVERE_BLUR * sharp_med);
        match opening {
            Some(&(a, b)) => (v, format!(
                "Blurry in places: {}-{} is very blurry, and the quick first view is built from the opening seconds; start filming steady, or trim the start{}.",
                clock(a as f64 * dt), clock(b as f64 * dt),
                if blur_runs.len() > 1 { format!(" (also {})", ranges(&blur_runs.iter().copied().filter(|r| r.0 != a).collect::<Vec<_>>(), &|_, _| "mostly blurred frames".into())) } else { String::new() },
            )),
            None => (v, format!("Blurry in places: {}.", ranges(&blur_runs, &|_, _| "mostly blurred frames".into()))),
        }
    };
    let (v, m) = if blurred_share > BLURRED_WARN {
        (Verdict::Warn, format!("Blurry: {:.0}% of frames are much softer than the rest ({spikes} sudden blur spikes).", blurred_share * 100.0))
    } else if blur_total_s >= BLUR_WARN_TOTAL_S {
        blurry_in_places(Verdict::Warn)
    } else if !smear_runs.is_empty() {
        let at = |a: usize, b: usize| {
            let v: Vec<f64> = smear_px[a..b].iter().flatten().copied().collect();
            format!("~{:.0} px", quantile(&v, 0.5))
        };
        (Verdict::Tip, format!("Smeared by motion: {}; more light or a slower turn.", ranges(&smear_runs, &at)))
    } else if !blur_runs.is_empty() {
        blurry_in_places(Verdict::Tip)
    } else {
        (Verdict::Pass, format!("Sharp enough ({:.0}% blurred frames).", blurred_share * 100.0))
    };
    add("sharpness", "Sharpness", v, m, "Hold the phone steady with both hands, tap the screen to focus, add light, move slowly.");

    // Movement: travel and depth change.
    let flat = depth_change.is_some_and(|d| d < NO_DEPTH_CHANGE);
    let depth = depth_change.unwrap_or(0.0);
    let (v, m) = if motion_widths < LOW_MOTION_WIDTHS && depth_best.is_some_and(|d| d < NO_DEPTH_PAIR) {
        (Verdict::Fail, format!("The camera barely moved (about {motion_widths:.1} image widths, almost no depth change between views); there's no depth to build 3D from."))
    } else if flat {
        (Verdict::Tip, format!("Mostly turning in place: little depth change between views (median {:.0}% of the picture).", depth * 100.0))
    } else if depth_change.is_none() && motion_widths < LOW_MOTION_WIDTHS {
        (Verdict::Tip, format!("Little camera movement (about {motion_widths:.1} image widths), and views too different to measure depth change."))
    } else {
        (Verdict::Pass, match depth_change {
            Some(d) => format!("Moves enough (median {:.0}% of the picture shows depth change).", d * 100.0),
            None => format!("Moves enough (about {motion_widths:.1} image widths)."),
        })
    };
    add(
        "movement",
        "Movement",
        v,
        m,
        if opts.room { "Walk, don't spin: walk a slow arc 1-2 m across in front of each wall, at two heights." } else { "Walk around the subject or step sideways while filming; turning on the spot gives no depth." },
    );

    // Detail: corners per measured frame and how many match the next one (a stand-in for SfM's
    // verified matches; Spearman 0.84 with them per frame over 73 calibration clips).
    let corner_counts: Vec<f64> = samples.iter().filter_map(|s| s.corners).map(f64::from).collect();
    let match_counts: Vec<f64> = samples.iter().filter_map(|s| s.matches).map(f64::from).collect();
    let (corners_med, matches_med) = (quantile(&corner_counts, 0.5), quantile(&match_counts, 0.5));
    let measured_detail = !match_counts.is_empty();
    let (v, m) = if measured_detail && corners_med <= DETAIL_FAIL_CORNERS && matches_med <= DETAIL_FAIL_MATCHES {
        // A strong warning, not a fail: a room it fired on still built a usable model.
        (Verdict::Warn, format!("Too little detail to match: about {corners_med:.0} distinct points a frame and {matches_med:.0} matched between frames (plain walls, blur or darkness); the views may not tie together, so this video will probably fail."))
    } else if measured_detail && matches_med <= DETAIL_WARN_MATCHES {
        (Verdict::Warn, format!("Little detail to match: about {matches_med:.0} points matched between frames (aim for {:.0}+).", DETAIL_WARN_MATCHES + 8.0))
    } else {
        (Verdict::Pass, format!("Enough detail (about {corners_med:.0} points a frame, {matches_med:.0} matched between frames)."))
    };
    add("detail", "Detail", v, m, "Keep textured things in view (furniture, shelves, pictures, door frames), stay 1-2 m from plain walls, turn slower and add light.");

    // Speed / overlap.
    let blur = if motion_blur_ratio < 0.5 { format!("; moving frames are {:.0}% as sharp as still ones", motion_blur_ratio * 100.0) } else { String::new() };
    let (v, m) = if lost_run_s >= LOST_FAIL_S {
        (Verdict::Fail, format!("Too fast: frames stop overlapping for {lost_run_s:.0} s at a time (a hyperlapse or very fast pans)."))
    } else if !blur_est_runs.is_empty() || blur_est_med >= BLUR_EST_WARN_PX {
        let at = |a: usize, b: usize| format!("~{:.0} px of blur a frame, turning ~{:.0} degrees a second", quantile(&blur_est[a..b], 0.5), quantile(&speed[a..b], 0.5));
        let where_ = if blur_est_runs.is_empty() { format!("through most of the video (~{blur_est_med:.0} px a frame)") } else { ranges(&blur_est_runs, &at) };
        (Verdict::Warn, format!("Turning too fast for the light: {where_}; turn slower or add more light{blur}."))
    } else if !blur_tip_runs.is_empty() || blur_est_med >= BLUR_EST_TIP_PX {
        let at = |a: usize, b: usize| format!("{}-{}", clock(a as f64 * dt), clock(b as f64 * dt));
        let px = if blur_tip_runs.is_empty() { blur_est_med } else { blur_tip_runs.iter().map(|&(a, b)| quantile(&blur_est[a..b], 0.5)).fold(0.0, f64::max) };
        let when = if blur_tip_runs.is_empty() { "through most of the video".to_string() } else { blur_tip_runs.iter().take(3).map(|&(a, b)| at(a, b)).collect::<Vec<_>>().join(", ") };
        let light = if luma_med < EXPOSURE_FULL_FRAME_LUMA { "; more light also helps" } else { "" };
        (Verdict::Tip, format!("Move a bit slower for sharper results (about {px:.0} px of blur at {when}){light}."))
    } else if lost_share > LOST_WARN_SHARE {
        (Verdict::Tip, format!("Jumpy: {:.0}% of frames don't overlap the one before.", lost_share * 100.0))
    } else {
        (Verdict::Pass, format!("Steady pace (about {speed_med:.0} degrees a second, ~{blur_est_med:.0} px of blur a frame)."))
    };
    add("speed", "Speed", v, m, "Turn slower or add light: about one step a second, a full turn in 40 s or more; brighter rooms let the phone use a shorter shutter.");

    // Coverage (rooms).
    if opts.room {
        // Directions relative to where the camera first pointed; yaw grows turning right.
        const DIRECTIONS: [&str; 8] = ["where you started", "45 degrees right", "your right (90)", "135 degrees right", "behind you", "135 degrees left", "your left (90)", "45 degrees left"];
        // The turn estimate is good to ~15% (one room clip: 312 vs SfM 358 deg; its SDR transcode read 297, and
        // re-encoded HLG/SDR pairs differ 3-5%), so a clip that turned FULL_TURN_DEG or more may have gone
        // all the way round: no direction is called "never filmed", only glanced at (0 s).
        let full_turn = yaw_span >= FULL_TURN_DEG;
        let missing: Vec<&str> = (0..8).filter(|&k| sectors[k] < 1.0 && !full_turn).map(|k| DIRECTIONS[k]).collect();
        let thin: Vec<String> = (0..8)
            .filter(|&k| sectors[k] < SECTOR_MIN_S)
            .map(|k| format!("{} ({:.0} s)", DIRECTIONS[k], sectors[k]))
            .collect();
        // Only a clip that clearly never went round warns: per-direction seconds come from an estimate
        // that is ~15% off on most clips and far off on some (an ARKit room read 139 vs a true 512 deg),
        // and nothing we measure shows quick glances costing quality. Glanced directions are noted.
        let (v, m) = if yaw_span < COVERAGE_WARN_SPAN_DEG {
            (Verdict::Tip, format!("Part of the room is missing: never filmed toward {} (turned about {:.0} of 360 degrees).", missing.join(", "), yaw_span))
        } else if !thin.is_empty() {
            (Verdict::Pass, format!("Turned about {:.0} degrees; quick glances only: {} (aim for at least {SECTOR_MIN_S:.0} s each).", yaw_span.min(360.0), thin.join(", ")))
        } else {
            (Verdict::Pass, format!("Covers the room (every direction at least {SECTOR_MIN_S:.0} s, turned {yaw_span:.0} degrees)."))
        };
        add("coverage", "Coverage", v, m, "Point the camera at every wall, the corners and the floor; give each wall at least 4 s, with extra slow passes on bare walls.");
    }

    let verdict = checks.iter().map(|c| c.verdict).max().map_or(Verdict::Pass, |v| if v == Verdict::Tip { Verdict::Pass } else { v });
    let summary = match verdict {
        Verdict::Pass | Verdict::Tip => match checks.iter().filter(|c| c.verdict == Verdict::Tip).count() {
            0 => "looks good.".to_string(),
            1 => "looks good (one optional tip below).".to_string(),
            _ => "looks good (optional tips below).".to_string(),
        },
        Verdict::Warn => format!("will probably work, but {}.", checks.iter().filter(|c| c.verdict == Verdict::Warn).map(|c| c.name.to_lowercase()).collect::<Vec<_>>().join(", ") + " could be better"),
        Verdict::Fail => format!("this video won't work ({}).", checks.iter().filter(|c| c.verdict == Verdict::Fail).map(|c| c.name.to_lowercase()).collect::<Vec<_>>().join(", ")),
    };
    let metrics = json!({
        "duration_s": round2(duration), "width": width, "height": height, "codec": codec,
        "hdr": transfer.map(|t| format!("{t:?}").to_lowercase()), "samples": samples.len(),
        "luma_median": round2(luma_med), "dim_share": round2(dim_share), "dark_share": round2(dark_share),
        "sharpness_median": round2(sharp_med), "blurred_share": round2(blurred_share), "blur_spikes": spikes,
        "detail_median": round2(detail_med), "smear_median": round2(smear_med), "motion_blur_ratio": round2(motion_blur_ratio),
        "textured_median": round2(textured_med),
        "motion_widths": round2(motion_widths), "pan_deg_s_median": round2(speed_med), "pan_deg_s_p90": round2(speed_p90),
        "lost_share": round2(lost_share), "lost_run_s": round2(lost_run_s),
        "corners_median": round2(corners_med), "matches_median": round2(matches_med),
        "exposure_s": round2(exposure * 1000.0) / 1000.0, "exposure_source": exposure_source, "blur_est_px_median": round2(blur_est_med),
        "blur_est_segments_s": blur_est_runs.iter().map(|&(a, b)| [round2(a as f64 * dt), round2(b as f64 * dt)]).collect::<Vec<_>>(),
        "fast_segments_s": fast_runs.iter().map(|&(a, b)| [round2(a as f64 * dt), round2(b as f64 * dt)]).collect::<Vec<_>>(),
        "smear_px_median": round2(smear_px_med),
        "smear_segments_s": smear_runs.iter().map(|&(a, b)| [round2(a as f64 * dt), round2(b as f64 * dt)]).collect::<Vec<_>>(),
        "blur_segments_s": blur_runs.iter().map(|&(a, b)| [round2(a as f64 * dt), round2(b as f64 * dt)]).collect::<Vec<_>>(),
        "pairs": fitted.len(), "depth_change": depth_change.map(round2), "depth_change_best": depth_best.map(round2),
        "yaw_span_deg": round2(yaw_span), "yaw_sector_s": sectors.map(round2), "fov_deg": [round2(fov_x), round2(fov_y)], "fov_source": fov_source,
    });
    CaptureCheck { verdict, summary, room: opts.room, checks, metrics, seconds: round2(start.elapsed().as_secs_f64()) }
}

// Thresholds, calibrated on the outcomes of a 61-video corpus and a 20-clip room set. A FAIL needs
// a wide margin to every capture that succeeded; anything less certain is a WARN.

/// Image travel (widths, the frame plan's motion track) under which a clip may not have moved.
/// Alone it is no failure: slow glides that succeeded read 0.08 to 0.96.
const LOW_MOTION_WIDTHS: f64 = 1.0;
/// Share of blocks off the keyframe homography (needs >= 2 fitted keyframe pairs; a fast
/// walk-through can leave none: unknown, never a fail). FAIL (with under LOW_MOTION_WIDTHS of
/// travel) when even the best pair is under NO_DEPTH_PAIR: SfM needs one initial pair with depth.
/// Best pair: 0.16-0.20 and 0.12-0.15 on two clips with no initial pair vs >= 0.34 on
/// every success under 1 width (0.34 and 0.46). The median, under NO_DEPTH_CHANGE,
/// warns "mostly turning in place" (yagan 0.08 at 5.7 widths succeeded: warn only).
const NO_DEPTH_PAIR: f64 = 0.27;
const NO_DEPTH_CHANGE: f64 = 0.2;
/// Median luma under which the video is black (night-stroll succeeded at 12).
const BLACK_LUMA: f64 = 6.0;
/// Share of frames under half the clip's median sharpness (a dark cellar clip 0.41; successes <= 0.34).
const BLURRED_WARN: f64 = 0.35;
/// Clips longer than this warn: 1 minute covers a room, and the camera step slows with length (a
/// 2 min room clip at 5 fps: 606 frames, SfM 415 s). Capping long clips at 360 frames instead saved
/// 32% of the solve but placed 93% of the frames against 98%, so the clip
/// is kept whole and the user is told to film less.
const LONG_WARN_S: f64 = 75.0;
/// Blurred stretches (>= SEGMENT_S each) adding up to this many seconds warn; fewer are a tip (dev
/// clips: 14 reach it, 57% of them worse vs 29% of the rest; shorter totals showed no lift).
const BLUR_WARN_TOTAL_S: f64 = 10.0;
/// Estimated motion blur (px a frame at 1080p): a median or >= SEGMENT_S stretch at TIP gives a tip,
/// at WARN a warning (on the 98-clip set 43% reach 8 px, 26% 10, 16% 12). The exposure is the
/// whole frame time below EXPOSURE_FULL_FRAME_LUMA median luma, half of it above.
const BLUR_EST_TIP_PX: f64 = 8.0;
const BLUR_EST_WARN_PX: f64 = 12.0;
const EXPOSURE_FULL_FRAME_LUMA: f64 = 100.0;
/// Smear measured on every SMEAR_EVERY-th sample (2 a second), up to MAX_BLUR_PX at the 640 px frame;
/// a dip below -BLUR_DIP in the derivative's autocorrelation counts as a smear.
const SMEAR_EVERY: usize = 4;
/// A blurred stretch starting in the first OPENING_S seconds whose median sharpness is under
/// SEVERE_BLUR x the clip's gets the "opening" wording (the quick first view uses those seconds).
const OPENING_S: f64 = 15.0;
const SEVERE_BLUR: f64 = 0.25;
/// Corner detection and matching for the matchability signal.
const CORNER_MIN_EIG: f32 = 1600.0;
/// Detail FAILs at or under both (median corners a frame, median matches to the next measured frame):
/// on the calibration clips only two ARKit rooms (14/2, 70% placed; 42/4, 59%) and the cellar
/// clip (47/5, failed); every success with <= 100 corners had >= 44 matches. WARN at or under
/// DETAIL_WARN_MATCHES matches (groei and pegasus succeeded at 4-6 matches: fast pans, 400 corners).
const DETAIL_FAIL_CORNERS: f64 = 50.0;
const DETAIL_FAIL_MATCHES: f64 = 5.0;
const DETAIL_WARN_MATCHES: f64 = 12.0;
const MAX_CORNERS: usize = 400;
const MATCH_NCC: f32 = 0.8;
const MAX_BLUR_PX: usize = 40;
const BLUR_DIP: f64 = 0.05;
/// Smear (px of a 1920 px long edge, i.e. 1080p) whose >= SEGMENT_S stretch warns. Room renders at
/// held-out views (13 clips, within-clip vs the clip median) were no softer at 6-12 px (x1.13), x0.93
/// at 12-15 px and x0.69 above 15 px; held-out PSNR can't see it (the ground truth is the smeared
/// frame). Warn rate on the 98-clip set: 51% at 6 px, 27% at 12, 10% at 15 (no success in the 61-video corpus).
const SMEAR_WARN_PX: f64 = 15.0;
/// A stretch at least this long (s) turning faster than SEGMENT_FAST_DEG_S, or with most frames
/// blurred, warns with its time range; a room direction (45 degrees) filmed under SECTOR_MIN_S is noted.
const SEGMENT_S: f64 = 5.0;
const SEGMENT_FAST_DEG_S: f64 = 30.0;
const SECTOR_MIN_S: f64 = 4.0;
const FULL_TURN_DEG: f64 = 270.0;
/// Coverage warns only under this total turn (about half the room never in view): 8 of 37 room clips.
const COVERAGE_WARN_SPAN_DEG: f64 = 180.0;
/// Longest stretch (s) of samples that don't overlap the previous one: a hyperlapse 6.9 vs
/// <= 2.6 on successes (groei, a 5 s clip).
const LOST_FAIL_S: f64 = 5.0;
/// Share of samples that don't overlap the previous one (pegasus 0.40 and groei 0.88 succeeded).
const LOST_WARN_SHARE: f64 = 0.25;

/// Check result for a video, or None for a photo folder (photos are not checked).
pub fn check_input(ffmpeg: &Path, source: &Path, opts: &CheckOptions) -> Result<Option<CaptureCheck>> {
    if source.is_dir() {
        return Ok(None);
    }
    check_video(ffmpeg, source, opts).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(w: usize, h: usize, seed: u64) -> Grey {
        let mut s = seed;
        let mut px = vec![0u8; w * h];
        // Smooth-ish random texture: random values box-blurred once.
        let raw: Vec<u8> = (0..w * h)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 256) as u8
            })
            .collect();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let v: u32 = [(0, 0), (1, 0), (0, 1), (1, 1)].iter().map(|(a, b)| raw[(y + b - 1 + 1) * w + x + a - 1 + 1] as u32).sum();
                px[y * w + x] = (v / 4) as u8;
            }
        }
        Grey { w, h, px }
    }

    #[test]
    fn flow_finds_a_shift_and_a_shift_has_no_parallax() {
        let (w, h) = (360, 220);
        let src = noise(w, h, 7);
        let crop = |ox: usize, oy: usize| Grey { w: 320, h: 180, px: (0..180).flat_map(|y| src.px[(y + oy) * w + ox..(y + oy) * w + ox + 320].to_vec()).collect() };
        let (a, b) = (crop(20, 20), crop(31, 14));
        let (flows, _) = block_flow(&a, &b, (0.0, 0.0));
        assert!(flows.len() > 100);
        let mut dx: Vec<f64> = flows.iter().map(|f| f.dx).collect();
        let mut dy: Vec<f64> = flows.iter().map(|f| f.dy).collect();
        assert!((median(&mut dx) + 11.0).abs() < 0.5 && (median(&mut dy) - 6.0).abs() < 0.5);
        let p = pair_stats(0.0, 1.0, 0.1, &flows, 160.0);
        assert!(p.off_share < 0.05 && p.p80_px < 0.5, "{p:?}");
    }

    #[test]
    fn blur_length_reads_a_box_smear() {
        let (w, h) = (640, 360);
        let mut seed = 99u64;
        let raw: Vec<f64> = (0..w * h)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % 256) as f64
            })
            .collect();
        let sharp = Grey { w, h, px: raw.iter().map(|&v| v as u8).collect() };
        assert_eq!(blur_length(&sharp, true), None);
        for len in [5usize, 8, 14] {
            let px = (0..h)
                .flat_map(|y| {
                    let raw = &raw;
                    (0..w).map(move |x| ((x..(x + len).min(w)).map(|i| raw[y * w + i]).sum::<f64>() / len as f64) as u8)
                })
                .collect();
            let smeared = Grey { w, h, px };
            assert_eq!(blur_length(&smeared, true), Some(len as f64), "box {len}");
            assert_eq!(blur_length(&smeared, false), None, "box {len} across");
        }
    }

    #[test]
    fn homography_fit_recovers_a_projective_map() {
        let h = [1.02, 0.01, 0.03, -0.02, 0.98, 0.01, 0.05, -0.03, 1.0];
        let pts: Vec<_> = (0..30).map(|k| {
            let (x, y) = ((k % 6) as f64 / 6.0 - 0.5, (k / 6) as f64 / 5.0 - 0.5);
            let (u, v) = apply(&h, x, y);
            (x, y, u, v)
        }).collect();
        let f = fit_homography(&pts).unwrap();
        for (a, b) in f.iter().zip(&h) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn room_goals() {
        assert!(goal_is_room("my living room"));
        assert!(goal_is_room("Kitchen scan"));
        assert!(!goal_is_room("statue in the park"));
        assert!(!goal_is_room("mushroom"));
    }
}
