//! Clip triage for trip trailers: is a vacation clip good splat input?
//!
//! Splats need the camera to translate around a static scene (parallax). Per consecutive pair of
//! 4 fps gray samples (320 px long edge) we block-match a 16 px grid, fit one homography to the
//! rigid matches and read parallax off the residuals: a pan, tilt or zoom from one spot fits a
//! homography exactly, a walk past things at different depths does not. Blocks whose best match is
//! poor and unpeaked are non-rigid content (water, crowds, leaves in wind).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use splat_eval::video::{VideoStatsOptions, video_stats};

use crate::scene::{median, quantile};

/// Long edge (px) of the motion samples; every px measure below is at this scale.
pub const ANALYSIS_LONG_EDGE: usize = 320;
/// Sample cap (60 s at 4 fps).
pub const MAX_SAMPLES: usize = 240;
const MIN_SHORT_EDGE: usize = 96;
const BLOCK: usize = 16;
/// Grid inset from the frame border, so most matches stay in view.
const MARGIN: usize = 16;
/// Search ranges: quarter-res global shift (+-48 px full), half-res per block around it (+-16 px
/// full), full-res refinement.
const GLOBAL_RANGE: i32 = 12;
const COARSE_RANGE: i32 = 8;
const FINE_RANGE: i32 = 2;
/// Blocks with intensity variance below this are untextured (sky, walls) and skipped.
const MIN_BLOCK_VAR: f64 = 60.0;
/// Non-rigid block: best SAD/px above this (0..255) and not clearly below the SAD 2 px away.
const DYNAMIC_SAD: f64 = 12.0;
const DYNAMIC_PEAK_RATIO: f64 = 0.6;
/// Fewest blocks for a usable pair (homography has 8 unknowns).
const MIN_BLOCKS: usize = 12;
/// Homography refits keep this share of best-fitting points.
const TRIM_KEEP: f64 = 0.8;
/// Per-pair parallax = this quantile of residuals over all rigid blocks.
const PARALLAX_QUANTILE: f64 = 0.7;

// Decision thresholds: guesses, tune after a real run.
/// Shorter clips give too few distinct views.
pub const MIN_DURATION_S: f64 = 4.0;
/// Laplacian variance at 640x360 (splat_eval::video); below this is mush.
pub const MIN_SHARPNESS: f64 = 15.0;
/// Median rigid image motion per sample step (px) below this = tripod/static.
pub const STATIC_MOTION_PX: f64 = 0.5;
/// Share of near-identical consecutive samples above this = static.
pub const STATIC_FRAC_MAX: f64 = 0.5;
/// Homography residual (px/step) below this, or below this share of the motion, = pan/zoom.
/// Block-matching noise is ~0.1-0.25 px, so the px floor sits just above it; the ratio catches
/// fast pans whose blur/rolling shutter push residuals over the floor.
pub const PAN_PARALLAX_PX: f64 = 0.35;
pub const PAN_PARALLAX_RATIO: f64 = 0.03;
/// Median share of textured blocks that move non-rigidly above this = dynamic scene.
pub const MAX_DYNAMIC_FRAC: f64 = 0.27;
/// Median image motion per sample (px at 320 wide) above which we re-sample at 3x the rate.
pub const FAST_MOTION_PX: f64 = 8.0;

#[derive(Clone, Debug)]
pub struct TriageOptions {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub fps: f64,
}

impl Default for TriageOptions {
    fn default() -> Self {
        Self { ffmpeg: "ffmpeg".into(), ffprobe: "ffprobe".into(), fps: 4.0 }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct ClipTriage {
    pub file: String,
    pub duration_s: f64,
    /// Display size (rotation metadata applied).
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub sharpness_median: f64,
    pub static_frac: Option<f64>,
    pub orientation_suspect: bool,
    pub samples: usize,
    /// Sample pairs with enough textured blocks to judge.
    pub pairs_tracked: usize,
    /// Median share of grid blocks with texture.
    pub texture_frac: f64,
    /// Median rigid image motion per sample step (px at 320 px long edge).
    pub motion_px_median: f64,
    /// Median homography residual per step (px): depth parallax.
    pub parallax_px: f64,
    /// parallax_px / motion_px_median (robust to walking speed).
    pub parallax_ratio: f64,
    /// Share of tracked pairs whose own parallax clears PAN_PARALLAX_PX (how much of the clip translates).
    pub parallax_pair_frac: f64,
    /// Median share of textured blocks that move non-rigidly (water, people, foliage).
    pub dynamic_frac: f64,
    pub capturable: bool,
    /// 0..1; >= 0.5 iff capturable.
    pub score: f64,
    pub verdict: String,
    pub reason: String,
}

pub fn triage_clips(paths: &[PathBuf], opts: &TriageOptions) -> Vec<Result<ClipTriage>> {
    paths.iter().map(|p| triage_clip(p, opts).with_context(|| format!("triaging {}", p.display()))).collect()
}

pub fn triage_clip(path: &Path, opts: &TriageOptions) -> Result<ClipTriage> {
    let vs = video_stats(
        path,
        &VideoStatsOptions { fps: opts.fps, max_frames: Some(MAX_SAMPLES), ffmpeg: opts.ffmpeg.clone(), ffprobe: opts.ffprobe.clone() },
    )?;
    let p = &vs.probe;
    // ffmpeg autorotates on decode, so a 90 degree tag swaps the decoded size.
    let (dw, dh) = if p.rotation_deg.rem_euclid(180) == 90 { (p.height, p.width) } else { (p.width, p.height) };
    let (w, h) = analysis_size(dw, dh);
    let med = |v: Vec<f64>| if v.is_empty() { 0.0 } else { median(v) };
    let track = |fps: f64| -> Result<(usize, Vec<PairFlow>)> {
        let pyrs: Vec<Pyr> = decode(path, &TriageOptions { fps, ..opts.clone() }, w, h)?.into_iter().map(Pyr::new).collect();
        let pairs = pyrs.windows(2).map(|f| analyze_pair(&f[0], &f[1])).filter(|f| f.textured >= MIN_BLOCKS).collect();
        Ok((pyrs.len(), pairs))
    };
    let (mut samples, mut pairs) = track(opts.fps)?;
    let mut k = 1.0; // reported motion/parallax stay per 1/fps step
    // Fast motion (timelapses, quick walks) jumps past the block search between samples and reads
    // as moving content: look again at 3x the sample rate.
    if med(pairs.iter().filter_map(|f| f.motion).collect()) > FAST_MOTION_PX && opts.fps * 3.0 <= p.fps {
        (samples, pairs) = track(opts.fps * 3.0)?;
        k = 3.0;
    }
    let grid = grid(w, h).count().max(1) as f64;
    let parallaxes: Vec<f64> = pairs.iter().filter_map(|f| f.parallax).collect();
    let motion = k * med(pairs.iter().filter_map(|f| f.motion).collect());
    let parallax = k * med(parallaxes.clone());
    let mut t = ClipTriage {
        file: path.display().to_string(),
        duration_s: p.duration_s,
        width: dw,
        height: dh,
        fps: p.fps,
        sharpness_median: vs.frames.sharpness_median,
        static_frac: vs.frames.static_frac,
        orientation_suspect: vs.orientation_suspect,
        samples,
        pairs_tracked: pairs.len(),
        texture_frac: med(pairs.iter().map(|f| f.textured as f64 / grid).collect()),
        motion_px_median: motion,
        parallax_px: parallax,
        parallax_ratio: if motion > 0.0 { parallax / motion } else { 0.0 },
        parallax_pair_frac: parallaxes.iter().filter(|&&v| k * v > PAN_PARALLAX_PX).count() as f64
            / parallaxes.len().max(1) as f64,
        dynamic_frac: med(pairs.iter().map(|f| f.dynamic as f64 / f.textured as f64).collect()),
        capturable: false,
        score: 0.0,
        verdict: String::new(),
        reason: String::new(),
    };
    decide(&mut t);
    Ok(t)
}

fn ramp(v: f64, lo: f64, hi: f64) -> f64 {
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Fills capturable/score/verdict/reason from the measurements.
fn decide(t: &mut ClipTriage) {
    let tracked = t.pairs_tracked > 0;
    let mut why: Vec<String> = vec![];
    if t.duration_s < MIN_DURATION_S {
        why.push(format!("too short ({:.1} s; want {MIN_DURATION_S:.0}+ s)", t.duration_s));
    }
    if !tracked {
        why.push("too little texture to track (sky, fog, dark or blank frames)".into());
    }
    if t.sharpness_median < MIN_SHARPNESS {
        why.push(format!("too blurry (sharpness {:.0} < {MIN_SHARPNESS:.0})", t.sharpness_median));
    }
    if tracked && (t.motion_px_median < STATIC_MOTION_PX || t.static_frac.is_some_and(|f| f > STATIC_FRAC_MAX)) {
        why.push("camera barely moves (tripod/static shot); no viewpoints to triangulate".into());
    } else if tracked && (t.parallax_px < PAN_PARALLAX_PX || t.parallax_ratio < PAN_PARALLAX_RATIO) {
        why.push(format!("pure pan/zoom from one spot: no parallax ({:.2} px/step)", t.parallax_px));
    }
    if tracked && t.dynamic_frac > MAX_DYNAMIC_FRAC {
        why.push(format!("scene content moves ({:.0}% of texture: water/people/foliage)", t.dynamic_frac * 100.0));
    }
    let s = ramp(t.duration_s, 0.5 * MIN_DURATION_S, 1.5 * MIN_DURATION_S)
        * ramp(t.sharpness_median, 0.5 * MIN_SHARPNESS, 2.0 * MIN_SHARPNESS)
        * ramp(t.motion_px_median, 0.5 * STATIC_MOTION_PX, 2.0 * STATIC_MOTION_PX)
        * (1.0 - ramp(t.static_frac.unwrap_or(0.0), 0.5 * STATIC_FRAC_MAX, 1.5 * STATIC_FRAC_MAX))
        * ramp(t.parallax_px, 0.5 * PAN_PARALLAX_PX, 2.5 * PAN_PARALLAX_PX)
        * ramp(t.parallax_ratio, 0.5 * PAN_PARALLAX_RATIO, 3.0 * PAN_PARALLAX_RATIO)
        * (1.0 - ramp(t.dynamic_frac, 0.5 * MAX_DYNAMIC_FRAC, 1.5 * MAX_DYNAMIC_FRAC));
    t.capturable = why.is_empty();
    t.score = if t.capturable { 0.5 + 0.5 * s } else { 0.5 * s };
    t.verdict = if t.capturable { "capturable" } else { "not capturable" }.into();
    t.reason = if t.capturable {
        format!(
            "walk-around with parallax {:.1} px/step ({:.0}% of motion), {:.0}% moving content",
            t.parallax_px,
            t.parallax_ratio * 100.0,
            t.dynamic_frac * 100.0
        )
    } else {
        why.truncate(2);
        why.join("; ")
    };
}

/// Long edge ANALYSIS_LONG_EDGE, even short edge.
fn analysis_size(w: u32, h: u32) -> (usize, usize) {
    let l = ANALYSIS_LONG_EDGE;
    let short = |a: u32, b: u32| (((l as f64 * a as f64 / b.max(1) as f64) / 2.0).round() as usize * 2).clamp(MIN_SHORT_EDGE, l);
    if w >= h { (l, short(h, w)) } else { (short(w, h), l) }
}

/// Gray samples at `fps`, scaled to exactly w x h.
fn decode(path: &Path, opts: &TriageOptions, w: usize, h: usize) -> Result<Vec<Plane>> {
    let vf = format!("fps={},scale={w}:{h},format=gray", opts.fps);
    let mut child = Command::new(&opts.ffmpeg)
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-vf", &vf, "-frames:v", &MAX_SAMPLES.to_string(), "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("running {}", opts.ffmpeg.display()))?;
    let mut stderr = child.stderr.take().expect("piped stderr");
    let err_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        buf
    });
    let mut raw = Vec::new();
    let read = child.stdout.take().expect("piped stdout").read_to_end(&mut raw);
    let status = child.wait()?;
    let err = err_reader.join().unwrap_or_default();
    if !status.success() {
        bail!("ffmpeg failed on {} ({status}): {}", path.display(), crate::tail(&String::from_utf8_lossy(&err), 2000));
    }
    read.with_context(|| format!("reading ffmpeg frames for {}", path.display()))?;
    Ok(raw.chunks_exact(w * h).map(|c| Plane { w, h, px: c.to_vec() }).collect())
}

/// 8-bit gray image, row-major.
struct Plane {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

impl Plane {
    /// 2x2 box downsample.
    fn half(&self) -> Plane {
        let (w, h) = (self.w / 2, self.h / 2);
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            let (r0, r1) = (&self.px[2 * y * self.w..], &self.px[(2 * y + 1) * self.w..]);
            for x in 0..w {
                let s = r0[2 * x] as u32 + r0[2 * x + 1] as u32 + r1[2 * x] as u32 + r1[2 * x + 1] as u32;
                px.push(((s + 2) / 4) as u8);
            }
        }
        Plane { w, h, px }
    }
}

/// Full, 1/2 and 1/4 scale plus mean intensity (for exposure compensation).
struct Pyr {
    l: [Plane; 3],
    mean: f64,
}

impl Pyr {
    fn new(p: Plane) -> Self {
        let mean = p.px.iter().map(|&v| v as u64).sum::<u64>() as f64 / p.px.len().max(1) as f64;
        let h = p.half();
        let q = h.half();
        Self { l: [p, h, q], mean }
    }
}

/// Top-left corners of the full-res block grid.
fn grid(w: usize, h: usize) -> impl Iterator<Item = (usize, usize)> {
    let ys = (MARGIN..=h.saturating_sub(BLOCK + MARGIN)).step_by(BLOCK);
    ys.flat_map(move |y| (MARGIN..=w.saturating_sub(BLOCK + MARGIN)).step_by(BLOCK).map(move |x| (x, y)))
}

/// Sum of |a - b - off| over a `size` window at `at` in a and `bt` in b; None if bt leaves b.
fn sad(a: &Plane, at: (usize, usize), b: &Plane, bt: (i32, i32), size: (usize, usize), off: i32) -> Option<u32> {
    if bt.0 < 0 || bt.1 < 0 || bt.0 as usize + size.0 > b.w || bt.1 as usize + size.1 > b.h {
        return None;
    }
    let (bx, by) = (bt.0 as usize, bt.1 as usize);
    let mut s = 0u32;
    for r in 0..size.1 {
        let ra = &a.px[(at.1 + r) * a.w + at.0..][..size.0];
        let rb = &b.px[(by + r) * b.w + bx..][..size.0];
        s += ra.iter().zip(rb).map(|(&p, &q)| (p as i32 - q as i32 - off).unsigned_abs()).sum::<u32>();
    }
    Some(s)
}

/// Best integer displacement (and its SAD) within `r` of `center`.
fn search(a: &Plane, at: (usize, usize), b: &Plane, size: (usize, usize), center: (i32, i32), r: i32, off: i32) -> Option<((i32, i32), u32)> {
    let mut best: Option<((i32, i32), u32)> = None;
    for dy in center.1 - r..=center.1 + r {
        for dx in center.0 - r..=center.0 + r {
            if let Some(s) = sad(a, at, b, (at.0 as i32 + dx, at.1 as i32 + dy), size, off)
                && best.is_none_or(|(_, bs)| s < bs)
            {
                best = Some(((dx, dy), s));
            }
        }
    }
    best
}

fn block_var(p: &Plane, at: (usize, usize)) -> f64 {
    let (mut s, mut sq) = (0u64, 0u64);
    for r in 0..BLOCK {
        for &v in &p.px[(at.1 + r) * p.w + at.0..][..BLOCK] {
            s += v as u64;
            sq += v as u64 * v as u64;
        }
    }
    let n = (BLOCK * BLOCK) as f64;
    sq as f64 / n - (s as f64 / n).powi(2)
}

/// Sub-pixel flow of the textured block at `at` and whether it moves non-rigidly; None if untextured.
fn match_block(a: &Pyr, b: &Pyr, at: (usize, usize), prior: (i32, i32), off: i32) -> Option<((f64, f64), bool)> {
    if block_var(&a.l[0], at) < MIN_BLOCK_VAR {
        return None;
    }
    let half = (BLOCK / 2, BLOCK / 2);
    let (c, _) = search(&a.l[1], (at.0 / 2, at.1 / 2), &b.l[1], half, prior, COARSE_RANGE, off)?;
    let full = (BLOCK, BLOCK);
    let (d, best) = search(&a.l[0], at, &b.l[0], full, (2 * c.0, 2 * c.1), FINE_RANGE, off)?;
    let best = best as f64;
    let s = |dx: i32, dy: i32| sad(&a.l[0], at, &b.l[0], (at.0 as i32 + d.0 + dx, at.1 as i32 + d.1 + dy), full, off).map(f64::from);
    // Equiangular (V) fit: SAD is roughly linear in the offset near its minimum.
    let sub = |l: Option<f64>, r: Option<f64>| match (l, r) {
        (Some(l), Some(r)) if l.max(r) > best => ((l - r) / (2.0 * (l.max(r) - best))).clamp(-0.5, 0.5),
        _ => 0.0,
    };
    let flow = (d.0 as f64 + sub(s(-1, 0), s(1, 0)), d.1 as f64 + sub(s(0, -1), s(0, 1)));
    let ring = (-2i32..=2)
        .flat_map(|j| (-2i32..=2).map(move |i| (i, j)))
        .filter(|&(i, j)| i.abs().max(j.abs()) == 2)
        .filter_map(|(i, j)| s(i, j))
        .fold(f64::INFINITY, f64::min);
    let dynamic = best / (BLOCK * BLOCK) as f64 > DYNAMIC_SAD && best > DYNAMIC_PEAK_RATIO * ring;
    Some((flow, dynamic))
}

/// One sample pair: textured/dynamic block counts, plus motion and parallax (px) when enough
/// blocks are rigid.
struct PairFlow {
    textured: usize,
    dynamic: usize,
    motion: Option<f64>,
    parallax: Option<f64>,
}

fn analyze_pair(a: &Pyr, b: &Pyr) -> PairFlow {
    let off = (a.mean - b.mean).round() as i32;
    let (qa, qb) = (&a.l[2], &b.l[2]);
    let r = GLOBAL_RANGE.min((qa.w.min(qa.h) as i32 - 8) / 2).max(0);
    let ru = r as usize;
    let g = search(qa, (ru, ru), qb, (qa.w - 2 * ru, qa.h - 2 * ru), (0, 0), r, off).map_or((0, 0), |(d, _)| d);
    let (w, h) = (a.l[0].w, a.l[0].h);
    let mut pf = PairFlow { textured: 0, dynamic: 0, motion: None, parallax: None };
    let (mut src, mut dst) = (vec![], vec![]);
    for at in grid(w, h) {
        let Some((f, dynamic)) = match_block(a, b, at, (2 * g.0, 2 * g.1), off) else { continue };
        pf.textured += 1;
        if dynamic {
            pf.dynamic += 1;
        } else {
            let c = ((at.0 + BLOCK / 2) as f64, (at.1 + BLOCK / 2) as f64);
            src.push(c);
            dst.push((c.0 + f.0, c.1 + f.1));
        }
    }
    if src.len() < MIN_BLOCKS {
        return pf;
    }
    pf.motion = Some(median(src.iter().zip(&dst).map(|(s, d)| (d.0 - s.0).hypot(d.1 - s.1)).collect()));
    // Normalised coordinates keep the DLT well conditioned.
    let (cx, cy, k) = (w as f64 / 2.0, h as f64 / 2.0, w.max(h) as f64 / 2.0);
    let norm = |v: &[(f64, f64)]| -> Vec<(f64, f64)> { v.iter().map(|p| ((p.0 - cx) / k, (p.1 - cy) / k)).collect() };
    if let Some(res) = homography_residuals(&norm(&src[..]), &norm(&dst[..])) {
        let mut px: Vec<f64> = res.iter().map(|r| r * k).collect();
        pf.parallax = Some(quantile(&mut px, PARALLAX_QUANTILE));
    }
    pf
}

/// Residuals of a homography src -> dst, refit on the TRIM_KEEP best-fitting points.
fn homography_residuals(src: &[(f64, f64)], dst: &[(f64, f64)]) -> Option<Vec<f64>> {
    let mut keep: Vec<usize> = (0..src.len()).collect();
    let mut res = vec![];
    for _ in 0..4 {
        let hm = fit_homography(src, dst, &keep)?;
        res = src.iter().zip(dst).map(|(&s, &d)| {
            let (u, v) = apply(&hm, s);
            (u - d.0).hypot(v - d.1)
        }).collect();
        keep = (0..src.len()).collect();
        keep.sort_by(|&i, &j| res[i].total_cmp(&res[j]));
        keep.truncate(((src.len() as f64 * TRIM_KEEP).ceil() as usize).max(8));
    }
    Some(res)
}

fn apply(h: &[f64; 8], (x, y): (f64, f64)) -> (f64, f64) {
    let den = h[6] * x + h[7] * y + 1.0;
    ((h[0] * x + h[1] * y + h[2]) / den, (h[3] * x + h[4] * y + h[5]) / den)
}

/// Least-squares DLT with h33 = 1 over the points in `idx` (normal equations).
fn fit_homography(src: &[(f64, f64)], dst: &[(f64, f64)], idx: &[usize]) -> Option<[f64; 8]> {
    let mut m = [[0.0f64; 9]; 8];
    for &i in idx {
        let ((x, y), (u, v)) = (src[i], dst[i]);
        let rows = [([x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u], u), ([0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v], v)];
        for (a, rhs) in rows {
            for (mr, &ar) in m.iter_mut().zip(&a) {
                for (mc, &ac) in mr.iter_mut().zip(&a) {
                    *mc += ar * ac;
                }
                mr[8] += ar * rhs;
            }
        }
    }
    solve8(m)
}

/// Gauss-Jordan elimination with partial pivoting on an augmented 8x9 system.
fn solve8(mut m: [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let p = (col..8).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[p][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, p);
        let pivot = m[col];
        for (r, row) in m.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / pivot[col];
                for (x, &pv) in row[col..].iter_mut().zip(&pivot[col..]) {
                    *x -= f * pv;
                }
            }
        }
    }
    Some(std::array::from_fn(|i| m[i][8] / m[i][i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hash noise in 2 px cells: texture that survives the 1/4 downsample.
    fn tex(seed: u64, x: i64, y: i64) -> u8 {
        let mut v = (x.div_euclid(2) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (y.div_euclid(2) as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
        v ^= v >> 29;
        v = v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        v ^= v >> 32;
        (v & 0xff) as u8
    }

    fn frame(f: impl Fn(i64, i64) -> u8) -> Pyr {
        let (w, h) = (320usize, 180usize);
        let px = (0..h).flat_map(|y| (0..w).map(move |x| (x as i64, y as i64))).map(|(x, y)| f(x, y)).collect();
        Pyr::new(Plane { w, h, px })
    }

    #[test]
    fn translation_has_motion_but_no_parallax() {
        let a = frame(|x, y| tex(1, x, y));
        let b = frame(|x, y| tex(1, x - 5, y - 3));
        let p = analyze_pair(&a, &b);
        assert!(p.textured >= 100, "{}", p.textured);
        assert_eq!(p.dynamic, 0);
        let m = p.motion.unwrap();
        assert!((m - 34f64.sqrt()).abs() < 0.3, "motion {m}");
        assert!(p.parallax.unwrap() < 0.25, "parallax {:?}", p.parallax);
    }

    #[test]
    fn two_depth_layers_give_parallax() {
        // 48 px stripes alternate between a layer moving 3 px and one moving 8 px.
        let layer = |y: i64| (y / 48) % 2 == 1;
        let a = frame(|x, y| if layer(y) { tex(2, x, y) } else { tex(1, x, y) });
        let b = frame(|x, y| if layer(y) { tex(2, x - 8, y) } else { tex(1, x - 3, y) });
        let p = analyze_pair(&a, &b);
        let m = p.motion.unwrap();
        assert!((3.0..=8.0).contains(&m), "motion {m}");
        assert!(p.parallax.unwrap() > 1.0, "parallax {:?}", p.parallax);
    }

    #[test]
    fn unrelated_frames_are_dynamic() {
        let p = analyze_pair(&frame(|x, y| tex(1, x, y)), &frame(|x, y| tex(3, x, y)));
        assert!(p.dynamic as f64 > 0.8 * p.textured as f64, "{} of {}", p.dynamic, p.textured);
    }

    #[test]
    fn homography_fit_is_exact() {
        let h = [1.1, 0.05, 0.02, -0.03, 0.95, 0.01, 0.1, -0.05];
        let src: Vec<(f64, f64)> = (0..25).map(|i| ((i % 5) as f64 / 2.0 - 1.0, (i / 5) as f64 / 2.0 - 1.0)).collect();
        let dst: Vec<(f64, f64)> = src.iter().map(|&p| apply(&h, p)).collect();
        let res = homography_residuals(&src, &dst).unwrap();
        assert!(res.iter().all(|&r| r < 1e-9), "{res:?}");
    }
}
