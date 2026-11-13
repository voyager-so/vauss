//! Frame selection for video: how many frames to keep and where.
//!
//! Frames are kept at a rate (base ~2 per second), not a fixed count, so long walks keep their
//! overlap (a fixed 150 frames over 5 minutes split Copped Hall's SfM). Local density follows camera
//! motion, measured on a cheap low-resolution pass (block matching, in image widths per second),
//! within a +-4x band around the base rate: turns get dense frames, still stretches sparse ones. At
//! each planned instant the sharpest of ~3 neighbouring frames is kept, so spacing drifts by a frame
//! or two, not a whole slot. Black and duplicate frames are dropped.

use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Read};
use std::ops::Range;
use std::path::Path;
use std::process::{Command, Stdio};

/// The 150-frame floor never asks for more than this many frames per second.
const MAX_FLOOR_FPS: f64 = 8.0;
/// Analysis sample rate and width.
pub const ANALYSIS_FPS: f64 = 8.0;
/// Default quick-view prefix: 24 cameras with at most 0.15 image widths between them.
pub const QUICK_MOTION_WIDTHS: f64 = 24.0 * 0.15;
const ANALYSIS_WIDTH: usize = 160;
const BLOCK: usize = 16;
/// Mean abs grey difference (0-255 at ~160 px) below which a frame duplicates the last kept one.
pub const DUP_DIFF: f32 = 1.5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOptions {
    /// Average kept frames per second before the cap.
    pub base_fps: f64,
    /// Local density stays within base/band .. base*band.
    pub band: f64,
    /// Count cap (the request's target_count).
    pub max_count: u32,
    /// Floor (the previous fixed default), so short clips keep enough views for SfM and a held-out split.
    pub min_count: u32,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self { base_fps: 2.0, band: 4.0, max_count: 600, min_count: 150 }
    }
}

impl PlanOptions {
    /// Up to `max_count` frames; `motion_even` spaces them in even steps of camera motion (density
    /// proportional to motion, practically unbanded) instead of the rate plan's 4x band. A clip
    /// with no measurable motion still spreads them evenly in time (the band's floor).
    pub fn spaced(max_count: u32, motion_even: bool) -> Self {
        let band = if motion_even { 1e3 } else { Self::default().band };
        Self { max_count, band, ..Self::default() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub count: u32,
    /// base_fps x duration, before the cap.
    pub requested: u64,
    pub cap_hit: bool,
    /// Planned instants (seconds from the start of the decoded range), ascending.
    pub instants_s: Vec<f64>,
    pub duration_s: f64,
    /// Total camera motion in image widths.
    pub motion_widths: f64,
    /// Kept frames per second at the slowest and busiest parts of the clip.
    pub density_fps: (f64, f64),
    pub reason: String,
    pub options: PlanOptions,
    /// Cumulative motion (image widths) at sample times, for auditing kept-frame spacing.
    #[serde(skip)]
    pub motion_curve: Vec<(f64, f64)>,
}

impl Plan {
    pub fn motion_at(&self, t: f64) -> f64 {
        let c = &self.motion_curve;
        let j = c.partition_point(|&(s, _)| s < t);
        if j == 0 {
            return c.first().map_or(0.0, |p| p.1);
        }
        if j >= c.len() {
            return c.last().map_or(0.0, |p| p.1);
        }
        let ((t0, m0), (t1, m1)) = (c[j - 1], c[j]);
        if t1 > t0 { m0 + (m1 - m0) * (t - t0) / (t1 - t0) } else { m0 }
    }
    /// Spacing audit of kept frames (times in seconds): largest and median motion and time gaps.
    pub fn audit(&self, kept_times: &[f64]) -> Value {
        let gaps = kept_times.windows(2).map(|p| (self.motion_at(p[1]) - self.motion_at(p[0]), p[1] - p[0])).collect::<Vec<_>>();
        if gaps.is_empty() {
            return json!({});
        }
        let mut m = gaps.iter().map(|g| g.0).collect::<Vec<_>>();
        let mut s = gaps.iter().map(|g| g.1).collect::<Vec<_>>();
        m.sort_by(f64::total_cmp);
        s.sort_by(f64::total_cmp);
        json!({"max_gap_widths":round(*m.last().unwrap()),"median_gap_widths":round(m[m.len()/2]),
               "max_gap_s":round(*s.last().unwrap()),"median_gap_s":round(s[s.len()/2])})
    }
    pub fn metrics(&self) -> Value {
        json!({"count":self.count,"requested":self.requested,"cap_hit":self.cap_hit,"duration_s":round(self.duration_s),
               "motion_widths":round(self.motion_widths),"density_fps":[round(self.density_fps.0),round(self.density_fps.1)],
               "reason":self.reason,"options":self.options})
    }
}

fn round(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Spawns ffmpeg decoding `source` (optionally a time range) through `filter` to raw `pix_fmt` frames
/// of `frame_bytes`, calling `each(index, bytes)`. Returns the number of frames.
fn decode(
    ffmpeg: &Path,
    source: &Path,
    time_range: Option<(f64, f64)>,
    filter: &str,
    pix_fmt: &str,
    frame_bytes: usize,
    each: impl FnMut(u64, &[u8]) -> Result<()>,
) -> Result<u64> {
    decode_with(ffmpeg, source, time_range, None, false, filter, pix_fmt, frame_bytes, each)
}

/// [`decode`]; `sampled` = Some(decoder threads) also skips non-reference frames (`-skip_frame
/// noref`: B-frames nothing predicts from; h264 4K decodes ~2.5x faster) and the deblocking filter,
/// where any nearby, slightly blockier frame will do.
#[allow(clippy::too_many_arguments)]
fn decode_with(
    ffmpeg: &Path,
    source: &Path,
    time_range: Option<(f64, f64)>,
    sampled: Option<usize>,
    strict: bool,
    filter: &str,
    pix_fmt: &str,
    frame_bytes: usize,
    mut each: impl FnMut(u64, &[u8]) -> Result<()>,
) -> Result<u64> {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-v", "error", "-nostdin", "-threads", "4", "-filter_threads", "4"]);
    if strict { cmd.arg("-xerror"); }
    if let Some(threads) = sampled {
        // No deblocking either (h264/hevc in-loop filter): ~15-25% faster 4K decode on an M4, and the
        // check's statistics are taken on a 640 px downscale.
        cmd.args(["-skip_frame", "noref", "-skip_loop_filter", "all", "-threads", &threads.to_string()]);
    }
    if let Some((a, b)) = time_range {
        cmd.args(["-ss", &format!("{a:.3}"), "-to", &format!("{b:.3}")]);
    }
    cmd.arg("-i").arg(source);
    // 0:V skips attached pictures (cover art stored as an extra video stream).
    cmd.args(["-map", "0:V:0", "-vf", filter]).args(passthrough(ffmpeg)).args(["-threads", "4", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]);
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {}", ffmpeg.display()))?;
    let mut stdout = child.stdout.take().expect("piped");
    let mut stderr = child.stderr.take().expect("piped");
    let err = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });
    let mut buf = vec![0u8; frame_bytes];
    let mut i = 0u64;
    let outcome = loop {
        match stdout.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break Ok(()),
            Err(e) => break Err(anyhow::Error::from(e)),
        }
        if let Err(e) = each(i, &buf) {
            break Err(e);
        }
        i += 1;
    };
    drop(stdout);
    if outcome.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let err = err.join().unwrap_or_default();
    outcome?;
    if !status.success() && (strict || i == 0) {
        bail!("ffmpeg failed ({status}): {}", err.trim());
    }
    Ok(i)
}

/// `source` (or a time range of it) through `filter` as raw `pix_fmt` frames of `frame_bytes`,
/// skipping non-reference frames, with `threads` decoder threads (the capture check: any frame near
/// each sample instant will do).
#[allow(clippy::too_many_arguments)]
pub fn decode_sampled(
    ffmpeg: &Path,
    source: &Path,
    time_range: Option<(f64, f64)>,
    threads: usize,
    filter: &str,
    pix_fmt: &str,
    frame_bytes: usize,
    each: impl FnMut(u64, &[u8]) -> Result<()>,
) -> Result<u64> {
    decode_with(ffmpeg, source, time_range, Some(threads), false, filter, pix_fmt, frame_bytes, each)
}

/// `-fps_mode passthrough` needs ffmpeg 5.1+; older builds spell it `-vsync passthrough`.
fn passthrough(ffmpeg: &Path) -> [&'static str; 2] {
    // Callers can select different executables (including decoder wrappers). A process-wide
    // answer lets the first executable's version silently choose flags for every later one.
    let old = {
        let out = Command::new(ffmpeg).arg("-version").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        // "ffmpeg version 8.0.3-..." or "ffmpeg version n4.4.2"; unknown versions count as new.
        let v = out.split_whitespace().nth(2).unwrap_or("").trim_start_matches('n');
        let mut parts = v.split(|c: char| !c.is_ascii_digit()).filter_map(|x| x.parse::<u32>().ok());
        matches!((parts.next(), parts.next()), (Some(major), Some(minor)) if (major, minor) < (5, 1))
    };
    if old { ["-vsync", "passthrough"] } else { ["-fps_mode", "passthrough"] }
}

/// Input warnings the agent should see: several real video tracks (only the first is used) and a
/// mirrored display matrix (front camera; pixels are kept as stored).
pub fn video_warnings(ffmpeg: &Path, source: &Path) -> Vec<String> {
    let ffprobe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
    let Ok(out) = Command::new(ffprobe).args(["-v", "error", "-show_streams", "-of", "json"]).arg(source).output() else { return vec![] };
    let Ok(j) = serde_json::from_slice::<Value>(&out.stdout) else { return vec![] };
    let videos: Vec<&Value> = j["streams"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|s| s["codec_type"] == "video" && s["disposition"]["attached_pic"] != 1)
        .collect();
    let mut warnings = vec![];
    if videos.len() > 1 {
        warnings.push(format!("Source has {} video tracks (e.g. a multi-lens capture); only the first is used.", videos.len()));
    }
    let mirrored = videos.first().and_then(|s| s["side_data_list"].as_array()).into_iter().flatten().any(|d| {
        let m: Vec<f64> = d["displaymatrix"]
            .as_str()
            .unwrap_or("")
            .lines()
            .flat_map(|l| l.split(':').nth(1).unwrap_or("").split_whitespace().filter_map(|x| x.parse().ok()).collect::<Vec<f64>>())
            .collect();
        m.len() >= 5 && m[0] * m[4] - m[1] * m[3] < 0.0
    });
    if mirrored {
        warnings.push("Video is mirrored (front-camera display matrix); frames are kept as stored, so the scene is mirrored.".into());
    }
    warnings
}

/// Downsample a gray image by 2 (box filter).
fn half(px: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let (hw, hh) = (w / 2, h / 2);
    let mut out = vec![0u8; hw * hh];
    for y in 0..hh {
        for x in 0..hw {
            let s = px[2 * y * w + 2 * x] as u32
                + px[2 * y * w + 2 * x + 1] as u32
                + px[(2 * y + 1) * w + 2 * x] as u32
                + px[(2 * y + 1) * w + 2 * x + 1] as u32;
            out[y * hw + x] = (s / 4) as u8;
        }
    }
    (out, hw, hh)
}

fn sad(a: &[u8], b: &[u8], w: usize, h: usize, bx: usize, by: usize, size: usize, dx: i32, dy: i32) -> Option<u32> {
    let (x0, y0) = (bx as i32 + dx, by as i32 + dy);
    if x0 < 0 || y0 < 0 || x0 as usize + size > w || y0 as usize + size > h {
        return None;
    }
    let mut s = 0u32;
    for y in 0..size {
        let ra = &a[(by + y) * w + bx..(by + y) * w + bx + size];
        let rb = &b[(y0 as usize + y) * w + x0 as usize..(y0 as usize + y) * w + x0 as usize + size];
        s += ra.iter().zip(rb).map(|(&p, &q)| (p as i32 - q as i32).unsigned_abs()).sum::<u32>();
    }
    Some(s)
}

/// Whether the BLOCK-sized block at (bx, by) has enough texture (grey variance >= 25) to match.
fn textured_block(px: &[u8], w: usize, bx: usize, by: usize) -> bool {
    let block = (0..BLOCK).flat_map(|y| px[(by + y) * w + bx..(by + y) * w + bx + BLOCK].iter());
    let (n, sum, sq) = block.fold((0f64, 0f64, 0f64), |(n, s, q), &v| (n + 1.0, s + v as f64, q + (v as f64).powi(2)));
    sq / n - (sum / n).powi(2) >= 25.0
}

/// Whether any block of the frame can be matched (a black or flat frame has none).
fn textured(px: &[u8], w: usize, h: usize) -> bool {
    (0..h.saturating_sub(BLOCK - 1)).step_by(BLOCK)
        .any(|by| (0..w.saturating_sub(BLOCK - 1)).step_by(BLOCK).any(|bx| textured_block(px, w, bx, by)))
}

/// Image motion from `prev` to `cur` (same size gray), as a fraction of image width: trimmed mean of
/// per-block displacement found by coarse-to-fine block matching. Textureless blocks are skipped.
pub fn block_motion(prev: &[u8], cur: &[u8], w: usize, h: usize) -> f64 {
    let pyramid = |px: &[u8]| {
        let l0 = (px.to_vec(), w, h);
        let l1 = half(&l0.0, l0.1, l0.2);
        let l2 = half(&l1.0, l1.1, l1.2);
        [l0, l1, l2]
    };
    let (pa, pb) = (pyramid(prev), pyramid(cur));
    let mut mags = vec![];
    for by in (0..h.saturating_sub(BLOCK - 1)).step_by(BLOCK) {
        for bx in (0..w.saturating_sub(BLOCK - 1)).step_by(BLOCK) {
            if !textured_block(prev, w, bx, by) {
                continue;
            }
            let (mut dx, mut dy) = (0i32, 0i32);
            for level in (0..3).rev() {
                let (a, lw, lh) = (&pa[level].0, pa[level].1, pa[level].2);
                let b = &pb[level].0;
                let (sx, sy, size) = (bx >> level, by >> level, BLOCK >> level);
                if level < 2 {
                    dx *= 2;
                    dy *= 2;
                }
                let mut best = (u32::MAX, dx, dy);
                for oy in -3..=3 {
                    for ox in -3..=3 {
                        if let Some(s) = sad(a, b, lw, lh, sx, sy, size, dx + ox, dy + oy) {
                            if s < best.0 || (s == best.0 && (dx + ox).abs() + (dy + oy).abs() < best.1.abs() + best.2.abs()) {
                                best = (s, dx + ox, dy + oy);
                            }
                        }
                    }
                }
                dx = best.1;
                dy = best.2;
            }
            mags.push(((dx * dx + dy * dy) as f64).sqrt());
        }
    }
    if mags.is_empty() {
        return 0.0;
    }
    mags.sort_by(f64::total_cmp);
    let keep = &mags[..(mags.len() * 9).div_ceil(10)];
    keep.iter().sum::<f64>() / keep.len() as f64 / w as f64
}

/// Grey size the motion track is measured at (ANALYSIS_WIDTH wide, even height).
pub fn analysis_size(width: u32, height: u32) -> (usize, usize) {
    let w = ANALYSIS_WIDTH;
    (w, ((w as f64 * height as f64 / width as f64).round() as usize).max(BLOCK) & !1)
}

/// The motion track, one ANALYSIS_FPS sample (grey, [`analysis_size`]) at a time.
pub struct MotionTracker {
    w: usize,
    h: usize,
    reference: Option<Vec<u8>>,
    track: Vec<(f64, f64)>,
    /// First sample after the current reference.
    since: usize,
}

impl MotionTracker {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, reference: None, track: vec![], since: 0 }
    }

    pub fn push(&mut self, px: &[u8]) {
        // Motion is measured against a held reference frame, not the previous sample: block matching
        // is whole-pixel at 160 px, so slow sliders and orbits (<0.5 px per sample) would read 0. The
        // reference advances once the shift is clear; the measured motion is booked at that sample.
        const MIN_SHIFT: f64 = 2.0 / ANALYSIS_WIDTH as f64;
        let m = self.reference.as_ref().map_or(0.0, |r| block_motion(r, px, self.w, self.h));
        self.track.push((self.track.len() as f64 / ANALYSIS_FPS, 0.0));
        // A flat frame (a fade from black) never becomes the reference: block matching against it reads
        // 0 forever (rongankatu's black first frame froze its 57 s walk at 0 widths; 7.8 without it).
        if (self.reference.is_none() || m >= MIN_SHIFT) && textured(px, self.w, self.h) {
            // Spread the motion over the samples it accumulated across, so density stays smooth.
            let n = self.track.len() - self.since;
            for s in &mut self.track[self.since..] {
                s.1 = m / n as f64;
            }
            self.since = self.track.len();
            self.reference = Some(px.to_vec());
        }
    }

    /// Motion already booked to samples. Once a prefix crosses its budget, later samples cannot
    /// move that crossing forwards: the held reference only adds nonnegative motion.
    pub fn recorded_motion_widths(&self) -> f64 {
        self.track.iter().map(|sample| sample.1).sum()
    }

    pub fn finish(self) -> Result<Vec<(f64, f64)>> {
        ensure!(!self.track.is_empty(), "ffmpeg decoded no frames for motion analysis");
        Ok(self.track)
    }
}

/// Low-resolution pass: (sample time in seconds, motion since the previous sample in image widths).
pub fn motion_track(ffmpeg: &Path, source: &Path, time_range: Option<(f64, f64)>, width: u32, height: u32) -> Result<Vec<(f64, f64)>> {
    ensure!(width > 0 && height > 0, "video dimensions are unavailable");
    let (w, h) = analysis_size(width, height);
    let mut tracker = MotionTracker::new(w, h);
    decode(ffmpeg, source, time_range, &format!("fps={ANALYSIS_FPS},scale={w}:{h}:flags=area"), "gray", w * h, |_, px| {
        tracker.push(px);
        Ok(())
    })?;
    tracker.finish()
}

/// One decode of the whole video at ANALYSIS_FPS: each sample as `size` RGB and as the grey frame
/// [`motion_track`] measures (`width` x `height` is the display size): (sample index, rgb24, grey).
/// The grey takes the same filter chain as motion_track's own pass, stacked under the RGB frame as
/// grey RGB (lossless), so the track, and every plan made from it, is the same as from motion_track.
/// The product's video runs analyse once (inspect_input) and feed both extractions from it.
pub fn analysis_pass(
    ffmpeg: &Path,
    source: &Path,
    size: (u32, u32),
    width: u32,
    height: u32,
    mut each: impl FnMut(u64, &[u8], &[u8]) -> Result<()>,
) -> Result<u64> {
    let (w, h) = (size.0 as usize, size.1 as usize);
    let (gw, gh) = analysis_size(width, height);
    ensure!(w >= gw, "analysis frames narrower than the motion track's");
    let hdr = hdr_transfer(ffmpeg, source);
    let (rgb, grey_format, pix_fmt, pixel_bytes) = if hdr.is_some() {
        // The same 16-bit conversion and tone map as full extraction. Keep the motion branch
        // at gray8 before widening it: its samples must match motion_track exactly.
        (format!("scale={w}:{h}:flags=area:in_color_matrix=bt2020,format=gbrp16le"),
            "format=gray,format=rgb24,format=gbrp16le", "rgb48le", 6)
    } else {
        (format!("scale={w}:{h}:flags=area,format=rgb24"), "format=gray,format=rgb24", "rgb24", 3)
    };
    let filter = format!(
        "fps={ANALYSIS_FPS},split=2[a][b];[a]{rgb}[c];[b]scale={gw}:{gh}:flags=area,{grey_format},pad={w}:{gh}[d];[c][d]vstack=inputs=2"
    );
    let luts = hdr.map(|t| (transfer_lut(t), srgb_lut(), eetf_gain_lut()));
    let mut grey = vec![0u8; gw * gh];
    decode(ffmpeg, source, None, &filter, pix_fmt, w * (h + gh) * pixel_bytes, |i, px| {
        for y in 0..gh {
            let row = &px[((h + y) * w) * pixel_bytes..((h + y) * w + gw) * pixel_bytes];
            for (x, g) in grey[y * gw..(y + 1) * gw].iter_mut().enumerate() {
                *g = row[x * pixel_bytes + usize::from(hdr.is_some())];
            }
        }
        let rgb = &px[..w * h * pixel_bytes];
        match (&luts, hdr) {
            (Some((to_nits, srgb, gain)), Some(t)) => each(i, &tonemap(rgb, t, to_nits, srgb, gain), &grey),
            _ => each(i, rgb, &grey),
        }
    })
}

/// End (s) of the clip prefix whose cumulative motion stays within `max_widths` image widths, at
/// least `min_s` (and at most the track). The quick first-view SfM plans its few frames inside it:
/// showbus's 24 frames over the whole clip (14.3 widths, gaps of 0.6 widths) found no initial pair,
/// over its first 15 s (2.3 widths) all 24 registered.
pub fn motion_prefix(track: &[(f64, f64)], max_widths: f64, min_s: f64) -> f64 {
    let end = track.last().map_or(0.0, |t| t.0);
    let mut motion = 0.0;
    for &(t, m) in track {
        motion += m;
        if motion > max_widths {
            return t.max(min_s).min(end);
        }
    }
    end
}

/// Frame count and planned instants from a motion track. `frames_available` caps the count.
pub fn plan(track: &[(f64, f64)], duration_s: f64, frames_available: u64, opts: &PlanOptions) -> Plan {
    let duration = duration_s.max(track.last().map_or(0.0, |t| t.0)).max(1e-3);
    // Intervals between samples (plus a tail to the end), each with its motion rate in widths/s.
    let mut spans = vec![];
    let (mut motion, mut last_t) = (0.0, 0.0);
    let mut curve = vec![(0.0, 0.0)];
    for &(t, m) in track.iter().skip(1) {
        let dt = t - last_t;
        if dt > 0.0 {
            spans.push((last_t, dt, m / dt));
        }
        motion += m;
        curve.push((t, motion));
        last_t = t;
    }
    let tail_rate = spans.last().map_or(0.0, |s| s.2);
    if duration > last_t {
        spans.push((last_t, duration - last_t, tail_rate));
    }
    let requested = (opts.base_fps * duration).round().max(1.0) as u64;
    let cap = u64::from(opts.max_count).min(frames_available.max(1));
    // Floor: min_count, but never denser than MAX_FLOOR_FPS (a 10 s clip gets 80, not 150
    // near-duplicates).
    let floor = (u64::from(opts.min_count) as f64).min(MAX_FLOOR_FPS * duration).round().max(1.0) as u64;
    let target = requested.clamp(floor.min(cap), cap) as f64;
    let base = target / duration;
    let (lo, hi) = (base / opts.band, base * opts.band);
    let density = |k: f64, r: f64| (k * r).clamp(lo, hi);
    let integral = |k: f64| spans.iter().map(|&(_, dt, r)| density(k, r) * dt).sum::<f64>();
    // Scale k so the motion-following density integrates to the target count.
    let (mut a, mut b) = (0.0f64, 1.0f64);
    while integral(b) < target && b < 1e12 {
        b *= 4.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (a + b);
        if integral(mid) < target { a = mid } else { b = mid }
    }
    let k = b;
    let total = integral(k);
    // The density only shapes where frames go; the count is the target even when motion can't fill
    // it (a static or very slow clip would otherwise fall to target/band).
    let count = (target.round() as u64).clamp(1, cap.max(1));
    // Instants at cumulative density j + 0.5 (j = 0..count).
    let mut instants = Vec::with_capacity(count as usize);
    let scale = count as f64 / total.max(1e-9);
    let mut acc = 0.0;
    let mut j = 0u64;
    for &(t0, dt, r) in &spans {
        let d = density(k, r) * scale;
        while j < count && acc + d * dt >= j as f64 + 0.5 {
            instants.push(t0 + (j as f64 + 0.5 - acc) / d);
            j += 1;
        }
        acc += d * dt;
    }
    while (instants.len() as u64) < count {
        instants.push(duration);
    }
    let (mut dmin, mut dmax) = (f64::INFINITY, 0.0f64);
    for &(_, _, r) in &spans {
        let d = density(k, r) * scale;
        dmin = dmin.min(d);
        dmax = dmax.max(d);
    }
    let capped = if requested > cap { format!(", capped at {cap}") } else { String::new() };
    let follows = if opts.band >= 100.0 {
        "frames sit at even steps of motion".to_string()
    } else {
        format!("density follows motion within {:.0}x of the base", opts.band)
    };
    let reason = format!(
        "Rate-based: {duration:.0} s at a base of {:.1} kept frames/s asks for {requested} frames{capped} -> {count}. Camera moved {motion:.1} image widths; {follows} ({dmin:.2}-{dmax:.2} frames/s here), denser in turns. Actual spacing is in frame_plan.spacing.",
        opts.base_fps
    );
    Plan {
        count: count as u32,
        requested,
        cap_hit: requested > cap,
        instants_s: instants,
        duration_s: duration,
        motion_widths: motion,
        density_fps: (dmin, dmax),
        reason,
        options: opts.clone(),
        motion_curve: curve,
    }
}

/// Laplacian variance on a grey downsample (long edge ~640 px).
pub fn sharpness(img: &RgbImage) -> f32 {
    sharpness_raw(img.as_raw(), img.width(), img.height())
}

fn sharpness_raw(px: &[u8], width: u32, height: u32) -> f32 {
    let step = (width.max(height) / 640).max(1);
    let (w, h) = (width / step, height / step);
    let mut g = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let row = (y * step * width) as usize * 3;
        for x in 0..w {
            let i = row + (x * step) as usize * 3;
            g.push(0.299 * px[i] as f32 + 0.587 * px[i + 1] as f32 + 0.114 * px[i + 2] as f32);
        }
    }
    if w < 3 || h < 3 {
        return 0.0;
    }
    let (w, h) = (w as usize, h as usize);
    let (mut sum, mut sq) = (0f64, 0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let l = g[y * w + x - 1] + g[y * w + x + 1] + g[(y - 1) * w + x] + g[(y + 1) * w + x] - 4.0 * g[y * w + x];
            sum += l as f64;
            sq += (l as f64).powi(2);
        }
    }
    let n = ((w - 2) * (h - 2)) as f64;
    (sq / n - (sum / n).powi(2)) as f32
}

fn grey(img: &RgbImage, step: u32) -> (u32, u32, Vec<f32>) {
    let (w, h) = (img.width() / step, img.height() / step);
    let mut g = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x * step, y * step);
            g.push(0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32);
        }
    }
    (w, h, g)
}

/// ~160 px grey thumbnail rescaled to a standard contrast (std 50), so the duplicate test sees
/// motion in dark or flat footage as clearly as in normal footage.
fn thumb(img: &RgbImage) -> Vec<f32> {
    let g = grey(img, (img.width().max(img.height()) / 160).max(1)).2;
    let n = g.len().max(1) as f32;
    let mean = g.iter().sum::<f32>() / n;
    let std = (g.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
    let k = if std > 1.0 { 50.0 / std } else { 1.0 };
    g.iter().map(|v| (v - mean) * k).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractOptions {
    /// Decode every source frame, bypassing selection and black/duplicate filters.
    /// Any detected decoder corruption fails instead of dropping a frame.
    #[serde(default)]
    pub preserve_all: bool,
    pub uniform: bool,
    pub drop_black: bool,
    pub drop_duplicates: bool,
    pub rotate_deg: u32,
    /// Added to decoded frame indices for file names (first frame of the time range).
    pub start_index: u64,
    /// Half-open decoded-index bands that cannot supply training candidates.
    pub excluded: Vec<Range<u64>>,
    /// Maximum source-frame offset around an instant (product: 1; fixed test list: 0).
    pub pick_radius: u64,
    /// Research source-timeline extraction must not resample decoded source frames.
    /// Product callers retain constant-rate resampling.
    pub preserve_source_frames: bool,
    /// Crop applied at source resolution before scaling (black bars, see [`detect_black_bars`]).
    /// `size` passed to [`extract`] must be the scaled size of the cropped frame.
    pub crop: Option<Crop>,
    /// Decoded indices (the quick SfM's frames) added as planned frames where the plan is sparse,
    /// see [`with_fill`]. Absent from recipes without them, so those replay byte-identically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fill: Vec<u64>,
    /// Exact decoded indices already used by the quick solve. Nearby ordinary targets are
    /// replaced, so dense plans retain their frame budget; these frames get no pick window.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anchors: Vec<u64>,
}

/// A pixel rectangle `w x h` at `(x, y)` of the source frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Crop {
    pub w: u32,
    pub h: u32,
    pub x: u32,
    pub y: u32,
}

/// Luma at or below which a pixel counts as black for bar detection (8-bit).
const BAR_LUMA: u8 = 24;
/// A row/column is content when more than this share of its pixels is brighter than `BAR_LUMA`
/// in any sampled frame.
const BAR_CONTENT_FRAC: f64 = 0.005;
/// Largest bar kept per side, as a share of the dimension (2.39:1 in 16:9 is 12.3%).
const BAR_MAX_FRAC: f64 = 0.15;

/// Constant black bars (letterbox/pillarbox, encoder padding) common to every sampled grey frame
/// (`w x h`, 8-bit): rows/columns that are near-black in all of them. Returns the content
/// rectangle, with even offsets and sizes (4:2:0 crop), or `None` when there is nothing to crop or
/// a bar would exceed `BAR_MAX_FRAC` (a dark scene, not a bar). Training on the bars makes the
/// model fake them with view-locked splats, which some runs do and some don't: house-10s.mp4's
/// 3-row bars swung held-out PSNR by ~1 dB between runs.
pub fn detect_black_bars<'a>(w: usize, h: usize, frames: impl IntoIterator<Item = &'a [u8]>) -> Option<Crop> {
    let mut acc = BarAccumulator::new(w, h);
    for px in frames {
        acc.push(px);
    }
    acc.finish()
}

/// Streaming [`detect_black_bars`]: per row/column, the most above-`BAR_LUMA` pixels any frame had.
pub struct BarAccumulator {
    w: usize,
    h: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    frames: usize,
}

impl BarAccumulator {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, rows: vec![0; h], cols: vec![0; w], frames: 0 }
    }

    pub fn push(&mut self, px: &[u8]) {
        let (w, h) = (self.w, self.h);
        if px.len() < w * h || w == 0 {
            return;
        }
        self.frames += 1;
        let mut c = vec![0usize; w];
        for (y, row) in px[..w * h].chunks_exact(w).enumerate() {
            let mut r = 0;
            for (x, &v) in row.iter().enumerate() {
                if v > BAR_LUMA {
                    r += 1;
                    c[x] += 1;
                }
            }
            self.rows[y] = self.rows[y].max(r);
        }
        for (m, v) in self.cols.iter_mut().zip(c) {
            *m = (*m).max(v);
        }
    }

    pub fn finish(&self) -> Option<Crop> {
        let (w, h) = (self.w, self.h);
        if w < 16 || h < 16 || self.frames == 0 {
            return None;
        }
        let content = |count: usize, len: usize| count as f64 > BAR_CONTENT_FRAC * len as f64;
        let lead = |v: &[usize], len: usize| v.iter().take_while(|&&c| !content(c, len)).count();
        let trail = |v: &[usize], len: usize| v.iter().rev().take_while(|&&c| !content(c, len)).count();
        let (mut top, mut bottom) = (lead(&self.rows, w), trail(&self.rows, w));
        let (mut left, mut right) = (lead(&self.cols, h), trail(&self.cols, h));
        let ok = |b: usize, len: usize| (b as f64) <= BAR_MAX_FRAC * len as f64;
        if !(ok(top, h) && ok(bottom, h)) {
            (top, bottom) = (0, 0);
        }
        if !(ok(left, w) && ok(right, w)) {
            (left, right) = (0, 0);
        }
        if top + bottom + left + right == 0 {
            return None;
        }
        // Even offsets and sizes: round each bar up to even.
        let even_up = |b: usize| (b + 1) & !1;
        let (top, bottom, left, right) = (even_up(top), even_up(bottom), even_up(left), even_up(right));
        let (cw, ch) = ((w - left - right) & !1, (h - top - bottom) & !1);
        Some(Crop { w: cw as u32, h: ch as u32, x: left as u32, y: top as u32 })
    }
}

/// [`detect_black_bars`] over `BAR_SAMPLES` frames at evenly spaced times of a video, each decoded
/// grey at display size `width x height` by an input seek (cost independent of the video length:
/// a keyframe pass over an 8-min 4K walk took >10 min on the dev host).
pub fn detect_video_bars(ffmpeg: &Path, source: &Path, time_range: Option<(f64, f64)>, duration_s: f64, width: u32, height: u32) -> Result<Option<Crop>> {
    const BAR_SAMPLES: usize = 16;
    let (w, h) = (width as usize, height as usize);
    let (start, span) = time_range.map_or((0.0, duration_s), |(a, b)| (a, b - a));
    let mut acc = BarAccumulator::new(w, h);
    // Each seek decodes from the preceding keyframe; keep one decoder live per capture.
    for batch in (0..BAR_SAMPLES).collect::<Vec<_>>().chunks(1) {
        let children = batch
            .iter()
            .map(|&k| {
                let t = start + span * (k as f64 + 0.5) / BAR_SAMPLES as f64;
                Command::new(ffmpeg)
                    .args(["-v", "error", "-nostdin", "-threads", "4", "-filter_threads", "4", "-ss", &format!("{t:.3}")])
                    .arg("-i")
                    .arg(source)
                    .args(["-map", "0:V:0", "-frames:v", "1", "-vf", &format!("scale={w}:{h},format=gray"), "-threads", "4", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .with_context(|| format!("spawning {}", ffmpeg.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        for child in children {
            acc.push(&child.wait_with_output()?.stdout);
        }
    }
    Ok(acc.finish())
}

#[derive(Debug, Default, Serialize)]
pub struct Extracted {
    pub kept: Vec<Value>,
    pub decoded: u64,
    pub dropped_black: u32,
    pub dropped_duplicates: u32,
    /// Frames dropped as decoder-corrupted (see [`OFF_GREEN_MAX`]).
    pub dropped_corrupt: u32,
    pub moved_excluded: u32,
    pub dropped_excluded: u32,
    pub skipped_excluded_candidates: u64,
    /// "hlg" or "pq" when HDR frames were tone-mapped to SDR.
    pub tone_mapped: Option<&'static str>,
    /// Dark frames that also got a brightened copy in `../analysis/` (for SfM features only).
    pub analysis_lifted: u32,
    /// Fill frames added to the plan's (see [`with_fill`]).
    pub filled: u32,
    /// Exact quick frames retained for preview/full camera alignment.
    pub anchored: u32,
    pub warnings: Vec<String>,
}

/// Mean luma (0-255) below which a kept frame also gets a brightened analysis copy.
const DARK_LUMA: f32 = 60.0;

/// Off-green fraction above which a kept frame is treated as decoder-corrupted (see `off_green_fraction`):
/// ~4x the cleanest-scene maximum, ~2x below the corrupt minimum.
pub const OFF_GREEN_MAX: f32 = 0.35;

/// A few flagged frames are dropped and extraction goes on; more than this share of the planned
/// frames, or [`CORRUPT_RUN_MAX`] consecutive ones, fails the capture as decoder-corrupted
/// (the HDR corruption on one HLG clip hit every frame; one false hit must not fail a 216-frame room).
pub const CORRUPT_SHARE_MAX: f32 = 0.05;
pub const CORRUPT_RUN_MAX: usize = 3;

/// Fraction of (sampled) pixels whose green channel is crushed while blue is not: either
/// G < 0.05 with B > 0.25 (the blue/magenta blocks) or G more than 0.15 below both R and B
/// (the pale-magenta wash). Red alone does not count: a saturated red object (one ARKit room's red
/// curtain filled 35% of 17 consecutive frames with G ~ 0, B < 0.16) is real content, and it failed
/// that whole capture as "decoder-corrupted"; with the blue rule those frames score 0.003. ffmpeg 8.0's native-size yuv420p10 -> rgb48le conversion produced
/// 0.77-0.86 on HLG phone frames. Clean footage stays far below: max 0.09 over the datasets
/// (dycheck teddy), 0.001 on night-stroll-follow-A frame_000346 although its G < 0.05 on 91% of
/// pixels (dark pixels are dark in every channel, so a green-only rule would fail night scenes).
pub fn off_green_fraction(px: &[u8]) -> f32 {
    let (mut bad, mut n) = (0u32, 0u32);
    for p in px.chunks_exact(3).step_by(7) {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        n += 1;
        if (g < 13 && b > 64) || (r - g > 38 && b - g > 38) {
            bad += 1;
        }
    }
    bad as f32 / n.max(1) as f32
}

/// Analysis-only exposure lift: scales linear light so the median reaches ~0.1 (at most 10 stops)
/// without pushing the 99th percentile past white. Training keeps the original frame.
fn lift_exposure(img: &RgbImage) -> Option<RgbImage> {
    let to_lin = |v: u8| { let c = v as f32 / 255.0; if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) } };
    let lut: Vec<f32> = (0..=255u8).map(to_lin).collect();
    let mut lum: Vec<f32> = img.pixels().step_by(7).map(|p| 0.2126 * lut[p[0] as usize] + 0.7152 * lut[p[1] as usize] + 0.0722 * lut[p[2] as usize]).collect();
    lum.sort_by(f32::total_cmp);
    let (median, p99) = (lum[lum.len() / 2].max(1e-5), lum[lum.len() * 99 / 100].max(1e-5));
    let gain = (0.1 / median).min(1024.0).min(1.0 / p99).max(1.0);
    if gain < 1.25 {
        return None; // highlights already near white: nothing useful to lift
    }
    let enc = |l: f32| { let l = l.clamp(0.0, 1.0); ((if l <= 0.003_130_8 { 12.92 * l } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }) * 255.0).round() as u8 };
    let table: Vec<u8> = lut.iter().map(|&l| enc(l * gain)).collect();
    let mut out = img.clone();
    for p in out.pixels_mut() {
        for c in p.0.iter_mut() {
            *c = table[*c as usize];
        }
    }
    Some(out)
}

/// HDR transfer functions phones record (iPhone HLG / Dolby Vision 8.4 base layer, PQ HDR10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    Hlg,
    Pq,
}

/// What ffprobe says about the source's colour: its HDR transfer, and whether the stream otherwise
/// looks HDR (10/12-bit or BT.2020) so an unmatched transfer is reported instead of silently decoded as SDR.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColourProbe {
    pub transfer: Option<Transfer>,
    pub hdr_like: bool,
}

/// Parses `ffprobe -show_entries stream=color_transfer,color_primaries,pix_fmt -of csv=p=0` over
/// every video stream. Field order, trailing commas (ffmpeg 8 prints `arib-std-b67,` when a stream has
/// side data), whitespace and extra lines all vary between ffmpeg versions, so this matches tokens.
pub fn parse_colour_probe(stdout: &str) -> ColourProbe {
    let mut probe = ColourProbe::default();
    for token in stdout.split(|c: char| c == ',' || c.is_whitespace()).map(str::trim).filter(|t| !t.is_empty()) {
        let token = token.to_ascii_lowercase();
        match token.as_str() {
            "arib-std-b67" => probe.transfer = probe.transfer.or(Some(Transfer::Hlg)),
            "smpte2084" => probe.transfer = probe.transfer.or(Some(Transfer::Pq)),
            "bt2020" | "bt2020nc" | "bt2020c" | "bt2020-10" | "bt2020-12" => probe.hdr_like = true,
            t if t.starts_with("yuv") && (t.contains("p10") || t.contains("p12")) || t.starts_with("p010") || t.starts_with("p016") => probe.hdr_like = true,
            _ => {}
        }
    }
    probe.hdr_like |= probe.transfer.is_some();
    probe
}

/// The source's colour, from the ffprobe beside `ffmpeg`. Default (SDR) when ffprobe fails.
pub fn probe_colour(ffmpeg: &Path, source: &Path) -> ColourProbe {
    let ffprobe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-select_streams", "v", "-show_entries", "stream=color_transfer,color_primaries,pix_fmt", "-of", "csv=p=0"])
        .arg(source)
        .stdin(Stdio::null())
        .output();
    match out {
        Ok(out) => parse_colour_probe(&String::from_utf8_lossy(&out.stdout)),
        Err(_) => ColourProbe::default(),
    }
}

/// The source's HDR transfer. None for SDR or when unknown.
pub fn hdr_transfer(ffmpeg: &Path, source: &Path) -> Option<Transfer> {
    probe_colour(ffmpeg, source).transfer
}

/// SDR reference white (BT.2408) and assumed HDR peak, in nits.
const REF_WHITE: f32 = 203.0;
const HDR_PEAK: f32 = 1000.0;

/// PQ (ST 2084) inverse EOTF: nits -> signal 0..1.
fn pq_encode(nits: f64) -> f64 {
    let (m1, m2) = (2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0);
    let (c1, c2, c3) = (3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0);
    let y = (nits / 10000.0).max(0.0).powf(m1);
    ((c1 + c2 * y) / (1.0 + c3 * y)).powf(m2)
}

fn pq_decode(e: f64) -> f64 {
    let (m1, m2) = (2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0);
    let (c1, c2, c3) = (3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0);
    let p = e.max(0.0).powf(1.0 / m2);
    ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1) * 10000.0
}

/// ITU-R BT.2390 EETF (the libplacebo/mpv `bt.2390` curve): HDR_PEAK -> REF_WHITE nits, identity
/// below the knee and a Hermite roll-off above it, in PQ space. Input and output in nits.
fn bt2390_eetf(nits: f64) -> f64 {
    let peak = pq_encode(HDR_PEAK as f64);
    let e = (pq_encode(nits) / peak).min(1.0);
    let max_lum = pq_encode(REF_WHITE as f64) / peak;
    let ks = 1.5 * max_lum - 0.5;
    let e = if e > ks {
        let t = (e - ks) / (1.0 - ks);
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * ks + (t3 - 2.0 * t2 + t) * (1.0 - ks) + (-2.0 * t3 + 3.0 * t2) * max_lum
    } else {
        e
    };
    pq_decode(e.min(max_lum) * peak)
}

/// Samples of the EETF gain (output / input, output relative to REF_WHITE) over sqrt(nits / HDR_PEAK),
/// so the per-pixel tone curve is a lookup instead of four powf calls.
const EETF_LUT: usize = 4096;
fn eetf_gain_lut() -> Vec<f32> {
    (0..=EETF_LUT)
        .map(|i| {
            let u = i as f64 / EETF_LUT as f64;
            let nits = (u * u * HDR_PEAK as f64).max(1e-3);
            (bt2390_eetf(nits) / nits / REF_WHITE as f64) as f32
        })
        .collect()
}

/// Tone-maps one rgb48le frame (BT.2020, HLG or PQ) to 8-bit sRGB/BT.709 the way libplacebo's
/// `bt.2390` does with static metadata: transfer -> display light (HLG OOTF, gamma 1.2 at 1000 nits),
/// BT.2020 -> BT.709 gamut, then the BT.2390 EETF (1000 -> 203 nits) on max(R, G, B) with the colour
/// scaled linearly (hue and saturation kept), reference white 203 nits -> 1.0, sRGB encode.
/// A Pixel HLG room frame vs libplacebo bt.2390: ΔE2000 2.2 (the extended-Reinhard curve before was 5.8
/// and ~20 luma levels darker, because it put reference white at half brightness).
fn tonemap(px16: &[u8], transfer: Transfer, to_nits: &[f32], srgb: &[u8], gain: &[f32]) -> Vec<u8> {
    const M: [[f32; 3]; 3] = [[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]];
    use rayon::prelude::*;
    let mut out = vec![0u8; px16.len() / 2];
    // Parallel over chunks of pixels (row-sized work items); per-pixel math is independent.
    out.par_chunks_mut(3 * 1024).zip(px16.par_chunks(6 * 1024)).for_each(|(dst, src)| {
    for (o, p) in dst.chunks_exact_mut(3).zip(src.chunks_exact(6)) {
        let code = |i: usize| u16::from_le_bytes([p[2 * i], p[2 * i + 1]]) as usize;
        let (mut r, mut g, mut b) = (to_nits[code(0)], to_nits[code(1)], to_nits[code(2)]);
        if transfer == Transfer::Hlg {
            // to_nits holds scene light 0..1; OOTF scales by display luminance^(gamma-1).
            let ys = 0.2627 * r + 0.6780 * g + 0.0593 * b;
            let k = HDR_PEAK * ys.max(1e-6).powf(0.2);
            (r, g, b) = (r * k, g * k, b * k);
        }
        let lin = [
            (M[0][0] * r + M[0][1] * g + M[0][2] * b).max(0.0),
            (M[1][0] * r + M[1][1] * g + M[1][2] * b).max(0.0),
            (M[2][0] * r + M[2][1] * g + M[2][2] * b).max(0.0),
        ];
        let sig = lin[0].max(lin[1]).max(lin[2]);
        let x = (sig / HDR_PEAK).clamp(0.0, 1.0).sqrt() * EETF_LUT as f32;
        let i = (x as usize).min(EETF_LUT - 1);
        let scale = gain[i] + (gain[i + 1] - gain[i]) * (x - i as f32);
        for (o, c) in o.iter_mut().zip(lin) {
            *o = srgb[((c * scale).clamp(0.0, 1.0) * 4095.0) as usize];
        }
    }
    });
    out
}

/// 16-bit code -> scene light 0..1 (HLG) or display nits (PQ).
fn transfer_lut(transfer: Transfer) -> Vec<f32> {
    (0..65536u32)
        .map(|v| {
            let e = v as f32 / 65535.0;
            match transfer {
                Transfer::Hlg => {
                    let (a, b, c) = (0.178_832_77f32, 0.284_668_92f32, 0.559_910_7f32);
                    if e <= 0.5 { e * e / 3.0 } else { (((e - c) / a).exp() + b) / 12.0 }
                }
                Transfer::Pq => {
                    let (m1, m2) = (2610.0f32 / 16384.0, 2523.0f32 / 4096.0 * 128.0);
                    let (c1, c2, c3) = (3424.0f32 / 4096.0, 2413.0f32 / 4096.0 * 32.0, 2392.0f32 / 4096.0 * 32.0);
                    let p = e.powf(1.0 / m2);
                    ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1) * 10000.0
                }
            }
        })
        .collect()
}

fn srgb_lut() -> Vec<u8> {
    (0..4096)
        .map(|i| {
            let l = i as f32 / 4095.0;
            let v = if l <= 0.003_130_8 { 12.92 * l } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
            (v * 255.0).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Planned frames at least this far apart leave room for a fill frame between them (below it the
/// plan is denser than ~3 fps and a fill frame only duplicates a neighbour).
const FILL_MIN_SPACING_S: f64 = 0.3;
/// A fill frame's minimum distance from every planned frame.
const FILL_MIN_GAP_S: f64 = 0.1;

/// `targets` (sorted) plus each `fill` index that lies in a sparse stretch of the plan: its
/// neighbours are >= FILL_MIN_SPACING_S apart and it is >= FILL_MIN_GAP_S from both (and from
/// fill frames already added). It then gets the usual pick window without narrowing its
/// neighbours'. Forcing all 24 quick frames in (2fd10dc) added near-duplicates of dense plans:
/// night-soccer 114 -> 126 frames, which tripped the trainer's splat budget; on sparse walks
/// (cambridge-heath-rd 150 planned, 2 fps) the quick frames sit mid-gap and add views.
fn with_fill(targets: &[u64], fill: &[u64], fps: f64) -> (Vec<u64>, u32) {
    let (min_spacing, min_gap) = (FILL_MIN_SPACING_S * fps, FILL_MIN_GAP_S * fps);
    let mut out = targets.to_vec();
    let mut added = 0;
    let mut fill = fill.to_vec();
    fill.sort_unstable();
    fill.dedup();
    for f in fill {
        let k = out.partition_point(|&t| t < f);
        let prev = k.checked_sub(1).map(|j| out[j]);
        let next = out.get(k).copied();
        let gap = |t: Option<u64>| t.map_or(f64::INFINITY, |t| t.abs_diff(f) as f64);
        let spacing = match (prev, next) {
            (Some(p), Some(n)) => (n - p) as f64,
            _ => f64::INFINITY,
        };
        if spacing >= min_spacing && gap(prev).min(gap(next)) >= min_gap {
            out.insert(k, f);
            added += 1;
        }
    }
    (out, added)
}

/// Include exact quick-frame indices without adding near-duplicates to dense stretches. A
/// nearby ordinary target is substituted, never another anchor; sparse gaps gain a frame.
fn with_anchors(targets: &[u64], anchors: &std::collections::BTreeSet<u64>, fps: f64) -> Vec<u64> {
    let mut out = targets.to_vec();
    for &anchor in anchors {
        if out.binary_search(&anchor).is_ok() { continue; }
        let nearby = out.iter().enumerate()
            .filter(|(_, t)| !anchors.contains(*t) && t.abs_diff(anchor) as f64 <= FILL_MIN_GAP_S * fps)
            .min_by_key(|(_, t)| (t.abs_diff(anchor), **t))
            .map(|(i, _)| i);
        if let Some(i) = nearby { out.remove(i); }
        let i = out.partition_point(|&t| t < anchor);
        out.insert(i, anchor);
    }
    out
}

/// Decodes the video at `size` and, at each planned instant, keeps the sharpest (or, if uniform, the
/// nearest) of the ~3 frames around it; writes `frame_<source index>.jpg` into `dest`.
pub fn extract(
    ffmpeg: &Path,
    source: &Path,
    time_range: Option<(f64, f64)>,
    size: (u32, u32),
    fps: f64,
    plan: &Plan,
    opts: &ExtractOptions,
    dest: &Path,
) -> Result<Extracted> {
    ensure!(!opts.preserve_all || (time_range.is_none() && opts.excluded.is_empty() && opts.anchors.is_empty() && opts.fill.is_empty()), "preserve_all cannot trim, exclude or anchor a subset");
    ensure!(fps > 0.0 && fps <= 1000.0, "implausible video frame rate {fps}");
    let (w, h) = size;
    let excluded = |i: u64| opts.excluded.iter().any(|r| r.contains(&i));
    let last_frame = (plan.duration_s * fps).round().max(1.0) as u64 - 1;
    // Planned frame per instant (deduplicated), with a half-window of 1 frame unless neighbours are closer.
    let mut targets: Vec<u64> = plan.instants_s.iter().map(|t| (t * fps).round().max(0.0) as u64).collect();
    targets.dedup();
    let mut out = Extracted::default();
    if !opts.excluded.is_empty() {
        let original = targets.clone();
        targets.clear();
        for (k, &target) in original.iter().enumerate() {
            let prev = if k > 0 { target - original[k - 1] } else { u64::MAX };
            let next = original.get(k + 1).map_or(u64::MAX, |n| n - target);
            let radius = (prev.min(next).saturating_sub(1) / 2).min(opts.pick_radius);
            let free_in_window = (target.saturating_sub(radius)..=target.saturating_add(radius).min(last_frame)).any(|i| !excluded(i));
            if free_in_window {
                targets.push(target);
                continue;
            }
            // Every nearest free frame is either a clip endpoint or an exclusion boundary.
            // Filtering all bands also handles overlaps without requiring callers to merge them.
            let nearest = [0, last_frame].into_iter()
                .chain(opts.excluded.iter().flat_map(|r| [r.start.saturating_sub(1), r.end]))
                .filter(|&i| i <= last_frame && !excluded(i))
                .min_by_key(|&i| (i.abs_diff(target), i));
            if let Some(i) = nearest {
                targets.push(i);
                out.moved_excluded += 1;
            } else {
                out.dropped_excluded += 1;
            }
        }
        targets.sort_unstable();
        let n = targets.len();
        targets.dedup();
        out.dropped_excluded += (n - targets.len()) as u32;
    }
    if !opts.fill.is_empty() {
        let fill: Vec<u64> = opts.fill.iter().copied().filter(|&i| i <= last_frame && !excluded(i)).collect();
        (targets, out.filled) = with_fill(&targets, &fill, fps);
    }
    let anchors: std::collections::BTreeSet<u64> = opts.anchors.iter().copied()
        .filter(|&i| i <= last_frame && !excluded(i)).collect();
    targets = with_anchors(&targets, &anchors, fps);
    let half = |k: usize| -> u64 {
        if anchors.contains(&targets[k]) { return 0; }
        let prev = if k > 0 { targets[k] - targets[k - 1] } else { u64::MAX };
        let next = targets.get(k + 1).map_or(u64::MAX, |n| n - targets[k]);
        (prev.min(next).saturating_sub(1) / 2).min(opts.pick_radius)
    };
    let mut best: Option<(f32, u64, RgbImage)> = None;
    let mut k = 0usize;
    let mut last: Option<Vec<f32>> = None;
    let mut corrupt: Vec<String> = Vec::new();
    let (mut corrupt_run, mut corrupt_run_max, mut corrupt_anchors) = (0usize, 0usize, 0u32);
    let mut flush = |best: &mut Option<(f32, u64, RgbImage)>, out: &mut Extracted| -> Result<()> {
        let Some((s, idx, img)) = best.take() else { return Ok(()) };
        let th = thumb(&img);
        // Anchors already passed the quick extraction's black/duplicate checks. Rechecking
        // at a different scale can discard a shared camera and break the world alignment.
        let anchored = anchors.contains(&idx);
        if !opts.preserve_all && !anchored && opts.drop_duplicates && last.as_ref().is_some_and(|t| mean_abs_diff(&th, t) < DUP_DIFF) {
            out.dropped_duplicates += 1;
            return Ok(());
        }
        last = Some(th);
        let off_green = off_green_fraction(img.as_raw());
        let img = match opts.rotate_deg % 360 {
            90 => image::imageops::rotate90(&img),
            180 => image::imageops::rotate180(&img),
            270 => image::imageops::rotate270(&img),
            _ => img,
        };
        let src = idx + opts.start_index;
        let name = format!("frame_{src:06}.jpg");
        if off_green > OFF_GREEN_MAX {
            ensure!(!opts.preserve_all, "preserve_all: suspected decoder corruption in {name}; no frames may be silently dropped");
            corrupt.push(format!("{name} ({:.0}%)", off_green * 100.0));
            corrupt_run += 1;
            corrupt_run_max = corrupt_run_max.max(corrupt_run);
            out.dropped_corrupt += 1;
            if anchored { corrupt_anchors += 1; }
            return Ok(());
        }
        corrupt_run = 0;
        let file = std::fs::File::create(dest.join(&name))?;
        image::codecs::jpeg::JpegEncoder::new_with_quality(io::BufWriter::new(file), 95).encode_image(&img)?;
        let luma = img.as_raw().iter().map(|&v| v as u64).sum::<u64>() as f32 / img.as_raw().len() as f32;
        if let Some(lifted) = (luma < DARK_LUMA).then(|| lift_exposure(&img)).flatten() {
            let analysis = dest.parent().unwrap_or(dest).join("analysis");
            std::fs::create_dir_all(&analysis)?;
            let file = std::fs::File::create(analysis.join(&name))?;
            image::codecs::jpeg::JpegEncoder::new_with_quality(io::BufWriter::new(file), 95).encode_image(&lifted)?;
            out.analysis_lifted += 1;
        }
        out.kept.push(json!({"name": name, "source_index": src, "decoded_index": idx, "sharpness": s}));
        if anchored { out.anchored += 1; }
        Ok(())
    };
    // HDR (HLG/PQ) decodes at 16 bits and is tone-mapped here: the pinned ffmpeg has no zscale.
    let probe = probe_colour(ffmpeg, source);
    let hdr = probe.transfer;
    out.tone_mapped = hdr.map(|t| if t == Transfer::Hlg { "hlg" } else { "pq" });
    let luts = hdr.map(|t| (transfer_lut(t), srgb_lut(), eetf_gain_lut()));
    let resample = if opts.preserve_all || opts.preserve_source_frames { String::new() } else { format!("fps={fps},") };
    let resample = match opts.crop {
        Some(c) => format!("crop={}:{}:{}:{},{resample}", c.w, c.h, c.x, c.y),
        None => resample,
    };
    let (filter, pix_fmt, bytes) = match hdr {
        // ffmpeg 8.0's unscaled yuv420p10 -> rgb48le path (output size = source size) returns
        // garbage colour (an HLG clip at native 1920x1080); converting via gbrp16le avoids it.
        Some(_) => (format!("{resample}scale={w}:{h}:flags=area:in_color_matrix=bt2020,format=gbrp16le"), "rgb48le", (w * h * 6) as usize),
        None => (format!("{resample}scale={w}:{h}:flags=area"), "rgb24", (w * h * 3) as usize),
    };
    // fps= resamples variable-frame-rate phone video to constant rate, so frame i is at i/fps
    // exactly (planned instants and time_range_s starts don't drift).
    out.decoded = decode_with(ffmpeg, source, time_range, None, opts.preserve_all, &filter, pix_fmt, bytes, |i, raw| {
        while !opts.preserve_all && k < targets.len() && i > targets[k] + half(k) {
            flush(&mut best, &mut out)?;
            k += 1;
        }
        if !opts.preserve_all && (k >= targets.len() || i + half(k) < targets[k]) {
            return Ok(());
        }
        if !opts.preserve_all && excluded(i) {
            out.skipped_excluded_candidates += 1;
            return Ok(());
        }
        let mapped;
        let px = match (&luts, hdr) {
            (Some((to_nits, srgb, gain)), Some(t)) => {
                mapped = tonemap(raw, t, to_nits, srgb, gain);
                mapped.as_slice()
            }
            _ => raw,
        };
        // Black = nearly every pixel near zero (fades, lens cap); dark but real footage is kept.
        let near_black = px.iter().step_by(31).filter(|&&v| v < 16).count() as f32 / px.iter().step_by(31).count().max(1) as f32;
        if !opts.preserve_all && !anchors.contains(&i) && opts.drop_black && near_black > 0.98 {
            out.dropped_black += 1;
            return Ok(());
        }
        let s = sharpness_raw(px, w, h);
        if opts.preserve_all {
            best = Some((s, i, RgbImage::from_raw(w, h, px.to_vec()).expect("sized buffer")));
            return flush(&mut best, &mut out);
        }
        let replace = match &best {
            None => true,
            // Uniform keeps the frame nearest the planned instant.
            Some(b) if opts.uniform => i.abs_diff(targets[k]) < b.1.abs_diff(targets[k]),
            Some(b) => s > b.0,
        };
        if replace {
            best = Some((s, i, RgbImage::from_raw(w, h, px.to_vec()).expect("sized buffer")));
        }
        Ok(())
    })?;
    flush(&mut best, &mut out)?;
    drop(flush);
    ensure!(
        (out.anchored + corrupt_anchors) as usize == anchors.len(),
        "full extraction kept {} of {} quick-frame anchors ({corrupt_anchors} dropped as corrupted)",
        out.anchored,
        anchors.len()
    );
    // Decoder colour corruption (ffmpeg 8.0 native-size HDR, 3d5769c1) must never reach SfM/training:
    // flagged frames are dropped above; many or clustered flags mean the decode itself is broken.
    let checked = out.kept.len() + corrupt.len();
    ensure!(
        corrupt.len() as f32 <= CORRUPT_SHARE_MAX * checked as f32 && corrupt_run_max < CORRUPT_RUN_MAX,
        "extracted frames look decoder-corrupted (magenta/blue: green channel crushed on >{:.0}% of pixels) in {} of {} frames ({} in a row) from {} (hdr: {:?}): {}",
        OFF_GREEN_MAX * 100.0,
        corrupt.len(),
        checked,
        corrupt_run_max,
        source.display(),
        out.tone_mapped,
        corrupt.iter().take(12).cloned().collect::<Vec<_>>().join(", ")
    );
    out.warnings = video_warnings(ffmpeg, source);
    if !corrupt.is_empty() {
        let warning = format!("Dropped {} decoder-corrupted-looking frame(s): {}", corrupt.len(), corrupt.join(", "));
        eprintln!("[frames] WARNING: {warning}");
        out.warnings.push(warning);
    }
    if probe.hdr_like && hdr.is_none() {
        let warning = format!(
            "Video {} looks HDR (10-bit or BT.2020) but its transfer is neither HLG nor PQ, so frames were decoded as SDR and colours may look washed out.",
            source.display()
        );
        eprintln!("[frames] WARNING: {warning}");
        out.warnings.push(warning);
    }
    if out.kept.len() < 20 {
        out.warnings.push(format!(
            "Only {} distinct frames: too few views for a reliable reconstruction; record a longer clip that moves around the subject.",
            out.kept.len()
        ));
    }
    if out.analysis_lifted > 0 {
        out.warnings.push(format!(
            "{} dark frames: brightened copies are in analysis/ for SfM feature matching only; training uses the originals.",
            out.analysis_lifted
        ));
    }
    // Replay metadata lets cloud jobs send the compressed capture instead of its JPEG expansion.
    // The worker uses this same extraction path and verifies every output before using it.
    let write_recipe = || -> Result<()> {
        let mut hashes = std::collections::BTreeMap::new();
        for frame in &out.kept {
            let name = frame["name"].as_str().expect("frame name");
            hashes.insert(format!("images/{name}"), file_sha256(&dest.join(name))?);
            let analysis = dest.parent().unwrap_or(dest).join("analysis").join(name);
            if analysis.is_file() {
                hashes.insert(format!("analysis/{name}"), file_sha256(&analysis)?);
            }
        }
        let recipe = ReplayRecipe {
            source: source.canonicalize()?, source_sha256: file_sha256(source)?,
            ffmpeg: ffmpeg.canonicalize()?, ffmpeg_sha256: file_sha256(ffmpeg)?,
            ffprobe_sha256: file_sha256(&ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" }))?,
            time_range, size, fps, plan: plan.clone(), opts: opts.clone(), hashes,
        };
        std::fs::write(dest.parent().unwrap_or(dest).join("video-replay.json"), serde_json::to_vec(&recipe)?)?;
        Ok(())
    };
    if let Err(error) = write_recipe() { eprintln!("[video-replay] metadata unavailable; cloud will upload frames: {error:#}"); }
    Ok(out)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReplayRecipe {
    pub source: std::path::PathBuf,
    pub source_sha256: String,
    pub ffmpeg: std::path::PathBuf,
    pub ffmpeg_sha256: String,
    pub ffprobe_sha256: String,
    pub time_range: Option<(f64, f64)>,
    pub size: (u32, u32),
    pub fps: f64,
    pub plan: Plan,
    pub opts: ExtractOptions,
    pub hashes: std::collections::BTreeMap<String, String>,
}

fn file_sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

/// Internal cloud preamble. A mismatch fails before any SfM/training command is started.
pub fn replay_manifest(path: &Path) -> Result<()> {
    let tasks: Vec<Value> = serde_json::from_slice(&std::fs::read(path)?)?;
    for task in tasks {
        let recipe: ReplayRecipe = serde_json::from_value(task["recipe"].clone())?;
        ensure!(file_sha256(&recipe.source)? == recipe.source_sha256, "video source SHA mismatch");
        ensure!(file_sha256(&recipe.ffmpeg)? == recipe.ffmpeg_sha256, "video decoder SHA mismatch");
        ensure!(file_sha256(&recipe.ffmpeg.with_file_name("ffprobe"))? == recipe.ffprobe_sha256, "video probe SHA mismatch");
        let key = task["key"].as_str().context("missing replay key")?;
        ensure!(key.len() == 64 && key.bytes().all(|c| c.is_ascii_hexdigit()), "invalid replay key");
        let cache = std::path::PathBuf::from("/tmp/splat-video-frames").join(key);
        let valid = || recipe.hashes.iter().all(|(rel, sha)| file_sha256(&cache.join(rel)).is_ok_and(|got| got == *sha));
        if !valid() {
            let temp = cache.with_extension(format!("{}", std::process::id()));
            if temp.exists() { std::fs::remove_dir_all(&temp)?; }
            std::fs::create_dir_all(temp.join("images"))?;
            let result = extract(&recipe.ffmpeg, &recipe.source, recipe.time_range, recipe.size,
                recipe.fps, &recipe.plan, &recipe.opts, &temp.join("images"));
            if let Err(error) = result { std::fs::remove_dir_all(&temp).ok(); return Err(error); }
            for (rel, sha) in &recipe.hashes {
                if file_sha256(&temp.join(rel))? != *sha {
                    std::fs::remove_dir_all(&temp).ok();
                    bail!("video replay SHA mismatch for {rel}");
                }
            }
            if cache.exists() { std::fs::remove_dir_all(&cache)?; }
            std::fs::rename(&temp, &cache)?;
        }
        for mapping in task["copies"].as_array().context("missing replay copies")? {
            let from = mapping["from"].as_str().context("missing replay source")?;
            ensure!(recipe.hashes.contains_key(from), "unknown replay frame");
            let to = Path::new(mapping["to"].as_str().context("missing replay destination")?);
            std::fs::create_dir_all(to.parent().context("missing destination parent")?)?;
            std::fs::copy(cache.join(from), to)?;
        }
        println!("[video-replay] verified {} frames/files; restored {} copies", recipe.hashes.len(), task["copies"].as_array().unwrap().len());
    }
    Ok(())
}

/// Internal cloud step beside the workload, after [`replay_manifest`]: copy the regenerated frames into the run's
/// fal staging dir (`<stage_dir>/<sha256>`, fal/app.py `_stage_path`), so the run's later jobs find them staged
/// and neither upload nor replay them. Files on the /data network volume serialize at ~0.5 s per create, hence
/// 8 writers, and the host runs this in the background while the job computes.
pub fn stage_replay_manifest(path: &Path) -> Result<()> {
    use rayon::prelude::*;
    let tasks: Vec<Value> = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut jobs = std::collections::BTreeMap::new();
    for task in &tasks {
        let Some(dir) = task["stage_dir"].as_str() else { continue };
        let key = task["key"].as_str().context("missing replay key")?;
        let cache = std::path::PathBuf::from("/tmp/splat-video-frames").join(key);
        for mapping in task["copies"].as_array().context("missing replay copies")? {
            let from = mapping["from"].as_str().context("missing replay source")?;
            let sha = task["recipe"]["hashes"][from].as_str().context("unknown replay frame")?;
            ensure!(sha.len() == 64 && sha.bytes().all(|c| c.is_ascii_hexdigit()), "invalid frame sha256");
            jobs.insert(Path::new(dir).join(sha), cache.join(from));
        }
    }
    let t0 = std::time::Instant::now();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    let staged: Vec<bool> = pool.install(|| jobs.par_iter().map(|(dest, src)| -> Result<bool> {
        if dest.is_file() { return Ok(false); }
        std::fs::create_dir_all(dest.parent().context("stage dir")?)?;
        let tmp = dest.with_file_name(format!(".{}.{}", dest.file_name().unwrap().to_string_lossy(), std::process::id()));
        std::fs::copy(src, &tmp)?;
        std::fs::rename(&tmp, dest)?;
        Ok(true)
    }).collect::<Result<_>>())?;
    println!("[video-replay] staged {} of {} frames for later jobs in {:.1}s",
        staged.iter().filter(|s| **s).count(), jobs.len(), t0.elapsed().as_secs_f64());
    Ok(())
}

/// Remove replayed inputs from sync-back, preserving outputs a worker actually changed.
pub fn cleanup_replay_manifest(path: &Path) -> Result<()> {
    let tasks: Vec<Value> = serde_json::from_slice(&std::fs::read(path)?)?;
    for task in tasks {
        for mapping in task["copies"].as_array().context("missing replay copies")? {
            let from = mapping["from"].as_str().context("missing replay source")?;
            let to = Path::new(mapping["to"].as_str().context("missing replay destination")?);
            if file_sha256(to).is_ok_and(|sha| task["recipe"]["hashes"][from].as_str() == Some(&sha)) {
                std::fs::remove_file(to)?;
            }
        }
    }
    std::fs::remove_file(path)?;
    Ok(())
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// Long-edge-limited even output size for a `width` x `height` video.
pub fn output_size(width: u32, height: u32, max_edge: u32) -> (u32, u32) {
    let scale = (max_edge as f64 / width.max(height) as f64).min(1.0);
    let even = |v: f64| ((v.round() as u32) / 2 * 2).max(2);
    (even(width as f64 * scale), even(height as f64 * scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_anchors_replace_dense_targets_without_losing_any_shared_frames() {
        let dense: Vec<u64> = (0..116).map(|i| i * 3).collect();
        let anchors = [20, 60, 98, 114, 129, 300].into_iter().collect();
        let out = with_anchors(&dense, &anchors, 24.0);
        assert_eq!(out.len(), dense.len(), "dense plan gained redundant frames");
        assert!(anchors.iter().all(|a| out.binary_search(a).is_ok()));
        assert!(!out.contains(&21) && !out.contains(&99));
        assert!(out.windows(2).all(|w| w[0] < w[1]));
        // Two close quick cameras may replace ordinary targets, but cannot replace each other.
        let anchors = [2, 3].into_iter().collect();
        assert_eq!(with_anchors(&[0, 4, 8], &anchors, 30.0), [2, 3, 8]);
        // Sparse plans retain their existing coverage while gaining a shared camera mid-gap.
        let anchors = [45].into_iter().collect();
        assert_eq!(with_anchors(&[0, 30, 60, 90], &anchors, 30.0), [0, 30, 45, 60, 90]);
    }

    #[test]
    fn colour_probe_parses_ffmpeg_7_and_8_output() {
        let hlg = ColourProbe { transfer: Some(Transfer::Hlg), hdr_like: true };
        let pq = ColourProbe { transfer: Some(Transfer::Pq), hdr_like: true };
        // ffmpeg 7: one bare value per stream.
        assert_eq!(parse_colour_probe("arib-std-b67\n"), hlg);
        assert_eq!(parse_colour_probe("smpte2084\n"), pq);
        // ffmpeg 8 with side data: trailing comma (a Pixel HLG room clip printed `arib-std-b67,`).
        assert_eq!(parse_colour_probe("arib-std-b67,\n"), hlg);
        // All three fields, in ffprobe's own order, with CRLF and stray whitespace.
        assert_eq!(parse_colour_probe("yuv420p10le,bt2020,arib-std-b67,\r\n"), hlg);
        assert_eq!(parse_colour_probe("  yuv420p10le , smpte2084 ,bt2020  \n"), pq);
        // Multiple video streams: an SDR cover/thumbnail stream before the HDR one.
        assert_eq!(parse_colour_probe("yuvj420p,unknown,unknown\nyuv420p10le,bt2020,arib-std-b67,\n"), hlg);
        // SDR and unknown values.
        assert_eq!(parse_colour_probe("yuv420p,bt709,bt709\n"), ColourProbe::default());
        assert_eq!(parse_colour_probe("unknown,\n"), ColourProbe::default());
        assert_eq!(parse_colour_probe(""), ColourProbe::default());
        // HDR-looking stream whose transfer is not HLG/PQ (warned about, decoded as SDR).
        assert_eq!(parse_colour_probe("yuv420p10le,bt2020,bt2020-10,\n"), ColourProbe { transfer: None, hdr_like: true });
        assert_eq!(parse_colour_probe("ARIB-STD-B67"), hlg);
    }

    /// 10 s at 8 samples/s: slow (0.005 widths/sample) except a fast turn at 4-6 s (0.2/sample).
    fn turn_track() -> Vec<(f64, f64)> {
        (0..80).map(|i| (i as f64 / 8.0, if i == 0 { 0.0 } else if (33..=48).contains(&i) { 0.2 } else { 0.005 })).collect()
    }

    fn in_turn(plan: &Plan) -> usize {
        plan.instants_s.iter().filter(|&&t| (4.0..=6.0).contains(&t)).count()
    }

    /// Largest over smallest motion step between consecutive planned frames.
    fn gap_spread(plan: &Plan) -> f64 {
        let gaps: Vec<f64> = plan.instants_s.windows(2).map(|p| plan.motion_at(p[1]) - plan.motion_at(p[0])).collect();
        gaps.iter().copied().fold(0.0, f64::max) / gaps.iter().copied().fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn motion_even_spacing_packs_a_fast_turn() {
        let track = turn_track();
        let rate = plan(&track, 10.0, 300, &PlanOptions::spaced(24, false));
        let even = plan(&track, 10.0, 300, &PlanOptions::spaced(24, true));
        assert_eq!((rate.count, even.count), (24, 24));
        // The turn is 91% of the motion: motion-even spacing puts ~22 of 24 frames there, the
        // rate plan's 4x band ~19 (it keeps 0.6 frames/s through the slow stretches).
        assert!(in_turn(&even) >= 21, "{:?}", even.instants_s);
        assert!(in_turn(&even) > in_turn(&rate), "{} vs {}", in_turn(&even), in_turn(&rate));
        // Even motion steps between consecutive frames (the rate plan's differ ~2.5x).
        assert!(gap_spread(&even) < 1.3, "{}", gap_spread(&even));
        assert!(gap_spread(&rate) > 2.0, "{}", gap_spread(&rate));
        assert!(even.instants_s.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    fn fill_frames_only_join_sparse_stretches_of_the_plan() {
        // 30 fps: fill needs neighbours >= 9 frames apart and >= 3 frames from each.
        let planned = [0, 4, 8, 30, 60, 90];
        // 6: dense stretch (4 apart). 9: 1 from 8. 20: mid-gap (8..30), added. 22: 2 from the added 20.
        // 45: added. 95: after the last planned frame, 5 away, added. 30: already planned.
        let (out, added) = with_fill(&planned, &[95, 45, 30, 22, 20, 9, 6], 30.0);
        assert_eq!(out, vec![0, 4, 8, 20, 30, 45, 60, 90, 95]);
        assert_eq!(added, 3);
        // Night-soccer (24 fps, 116 planned ~3 frames apart): its quick frames are 0-2 frames from a
        // planned one, so none is added.
        let dense: Vec<u64> = (0..116).map(|i| i * 3).collect();
        assert_eq!(with_fill(&dense, &[20, 60, 98, 114, 129, 300], 24.0), (dense.clone(), 0));
        // Recipes without fill frames serialize as before.
        let opts: ExtractOptions = serde_json::from_value(json!({"uniform": false, "drop_black": true,
            "drop_duplicates": true, "rotate_deg": 0, "start_index": 0, "excluded": [], "pick_radius": 1,
            "preserve_source_frames": false, "crop": null})).unwrap();
        assert!(opts.fill.is_empty() && serde_json::to_value(&opts).unwrap().get("fill").is_none());
    }

    #[test]
    fn extract_options_from_a_forced_frames_recipe_still_parse() {
        // 2fd10dc..5a6d3a1 recipes carry the quick frames as `forced` (dropped: they changed the
        // full extraction's count and frames, night-soccer 116 planned -> 126 kept).
        let opts: ExtractOptions = serde_json::from_value(json!({"uniform": false, "drop_black": true,
            "drop_duplicates": true, "rotate_deg": 0, "start_index": 0, "excluded": [], "pick_radius": 1,
            "preserve_source_frames": false, "crop": null, "forced": [8, 28]})).unwrap();
        assert_eq!((opts.pick_radius, opts.uniform), (1, false));
    }

    #[test]
    fn a_black_first_frame_does_not_freeze_the_track() {
        // A textured 160x96 pattern panning 4 px per sample, after a fade in from black.
        let (w, h) = (160usize, 96usize);
        let pattern = |shift: usize| -> Vec<u8> {
            (0..w * h).map(|i| (((i % w + shift) * 37 ^ (i / w) * 91) % 251) as u8).collect()
        };
        let mut t = MotionTracker::new(w, h);
        t.push(&vec![0u8; w * h]);
        for k in 0..10 {
            t.push(&pattern(4 * k));
        }
        let total: f64 = t.finish().unwrap().iter().map(|s| s.1).sum();
        // 9 steps of 4 px over 160 px = 0.225 widths.
        assert!((total - 0.225).abs() < 0.03, "{total}");
    }

    #[test]
    fn motion_even_spacing_without_motion_spreads_in_time() {
        let still: Vec<(f64, f64)> = (0..80).map(|i| (i as f64 / 8.0, 0.0)).collect();
        let even = plan(&still, 10.0, 300, &PlanOptions::spaced(24, true));
        assert_eq!(even.count, 24);
        for (j, t) in even.instants_s.iter().enumerate() {
            assert!((t - (j as f64 + 0.5) * 10.0 / 24.0).abs() < 1e-6, "{j}: {t}");
        }
    }
}

#[cfg(test)]
mod black_bar_tests {
    use super::*;

    fn frame(w: usize, h: usize, top: usize, bottom: usize, left: usize, fill: u8) -> Vec<u8> {
        let mut px = vec![fill; w * h];
        for y in 0..h {
            for x in 0..w {
                if y < top || y >= h - bottom || x < left {
                    px[y * w + x] = 3;
                }
            }
        }
        px
    }

    #[test]
    fn letterbox_rows_are_cropped_even() {
        let (w, h) = (192, 108);
        let f = frame(w, h, 3, 3, 0, 120);
        let c = detect_black_bars(w, h, [f.as_slice()]).expect("bars");
        assert_eq!(c, Crop { w: 192, h: 100, x: 0, y: 4 });
    }

    #[test]
    fn dark_scene_and_clean_frames_are_kept() {
        let (w, h) = (160, 90);
        assert_eq!(detect_black_bars(w, h, [vec![120u8; w * h].as_slice()]), None);
        // A night frame whose top third is black: beyond BAR_MAX_FRAC, not a bar.
        let night = frame(w, h, 30, 0, 0, 60);
        assert_eq!(detect_black_bars(w, h, [night.as_slice()]), None);
        // Bars must be black in every frame: a frame with content in the top rows cancels them.
        let a = frame(w, h, 4, 0, 0, 120);
        let b = vec![120u8; w * h];
        assert_eq!(detect_black_bars(w, h, [a.as_slice(), b.as_slice()]), None);
    }

    #[test]
    fn pillarbox_columns_are_cropped() {
        let (w, h) = (200, 100);
        let f = frame(w, h, 0, 0, 10, 120);
        let c = detect_black_bars(w, h, [f.as_slice()]).expect("bars");
        assert_eq!((c.x, c.y, c.w, c.h), (10, 0, 190, 100));
    }
}

#[cfg(test)]
mod preserve_all_tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn passthrough_flags_follow_each_selected_executable() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let executable = |name: &str, version: &str| {
            let path = tmp.path().join(name);
            std::fs::write(&path, format!("#!/bin/sh\nprintf '%s\\n' 'ffmpeg version {version}'\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path
        };
        let modern = executable("modern", "8.0.3");
        let legacy = executable("legacy", "n4.4.2");
        assert_eq!(passthrough(&modern), ["-fps_mode", "passthrough"]);
        assert_eq!(passthrough(&legacy), ["-vsync", "passthrough"]);
        assert_eq!(passthrough(&modern), ["-fps_mode", "passthrough"]);
    }

    #[test]
    fn preserve_all_video_keeps_every_black_duplicate_frame_beyond_target() {
        let tmp = tempfile::tempdir().unwrap();
        let video = tmp.path().join("black.mkv");
        // Lossless, all-black video is both a black-frame and duplicate-filter regression.
        let status = Command::new("ffmpeg").args(["-v","error","-f","lavfi","-i","color=black:s=32x32:r=12:d=2","-c:v","ffv1"])
            .arg(&video).status().expect("ffmpeg is required for this regression");
        assert!(status.success());
        let dest = tmp.path().join("images");
        std::fs::create_dir(&dest).unwrap();
        let plan = plan(&[], 2.0, 24, &PlanOptions::spaced(2, false));
        let opts: ExtractOptions = serde_json::from_value(json!({"preserve_all":true,"uniform":false,
            "drop_black":true,"drop_duplicates":true,"rotate_deg":0,"start_index":0,"excluded":[],
            "pick_radius":1,"preserve_source_frames":false,"crop":null})).unwrap();
        let all = extract(Path::new("ffmpeg"), &video, None, (32,32), 12.0, &plan, &opts, &dest).unwrap();
        assert_eq!(all.decoded,24);
        assert_eq!(all.kept.len(),24);
        assert_eq!((all.dropped_black,all.dropped_duplicates), (0,0));
        for (i,k) in all.kept.iter().enumerate() { assert_eq!(k["source_index"],i); }
    }

    #[cfg(unix)]
    #[test]
    fn preserve_all_partial_decoder_failure_is_not_success() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let ffmpeg = tmp.path().join("decoder");
        std::fs::write(&ffmpeg, "#!/bin/sh\nprintf '\\000\\000\\000'\nexit 1\n").unwrap();
        std::fs::set_permissions(&ffmpeg, std::fs::Permissions::from_mode(0o755)).unwrap();
        let result = decode_with(&ffmpeg, Path::new("input"), None, None, true, "null", "rgb24", 3, |_, _| Ok(()));
        assert!(result.unwrap_err().to_string().contains("ffmpeg failed"));
    }
}
