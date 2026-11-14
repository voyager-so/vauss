//! The image circle of a circular (or cropped-circle) fisheye: the lens projects onto a disc and
//! the sensor outside it stays black in every frame. Training and SfM must ignore that area: black
//! pixels would otherwise pull splats toward black and the circle's sharp rim yields static
//! "features" that match across every frame.
//!
//! Detection is a cheap image test: the per-pixel maximum luma over a few frames is near black
//! outside the disc, the dark region touches the frame border, its edge fits a circle tightly and
//! the picture just inside that edge is clearly lit (a dark vignette on a night clip is not a rim).

use std::path::Path;

use anyhow::{Context, Result};
use image::{GrayImage, Luma, imageops::FilterType};
use serde::Serialize;

/// Long edge of the analysis copies.
const ANALYSIS_EDGE: u32 = 320;
/// Max luma (0-255) over the sampled frames at or below which a pixel counts as unlit.
const DARK: u8 = 16;
/// Least share of the frame outside the circle: below it there is no usable circle (full-frame
/// fisheye, or a few dead pixels in a corner).
const MIN_OUTSIDE: f64 = 0.03;
/// The fitted circle must classify this share of pixels like the dark-region test does.
const MIN_AGREEMENT: f64 = 0.97;
/// Median luma just inside the rim must reach this (vs DARK outside).
const MIN_RIM_LUMA: f64 = 48.0;
/// The mask stops this share of the radius inside the detected rim (vignetting, chromatic fringe).
pub const MASK_MARGIN: f64 = 0.02;

/// A detected image circle, in pixels of the analysed frames' full resolution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Circle {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
    pub width: u32,
    pub height: u32,
    /// Share of the frame outside the circle.
    pub outside_fraction: f64,
}

impl Circle {
    /// The circle at another resolution of the same frames (uniformly resized).
    pub fn scaled(&self, width: u32, height: u32) -> Circle {
        let (sx, sy) = (width as f64 / self.width as f64, height as f64 / self.height as f64);
        Circle { cx: self.cx * sx, cy: self.cy * sy, radius: self.radius * 0.5 * (sx + sy), width, height, ..*self }
    }

    /// White inside the circle (shrunk by [`MASK_MARGIN`]), black outside: the training/COLMAP
    /// mask convention (white keeps, black ignores).
    pub fn mask(&self) -> GrayImage {
        let r = self.radius * (1.0 - MASK_MARGIN);
        let r2 = r * r;
        GrayImage::from_fn(self.width, self.height, |x, y| {
            let (dx, dy) = (x as f64 + 0.5 - self.cx, y as f64 + 0.5 - self.cy);
            Luma([if dx * dx + dy * dy <= r2 { 255 } else { 0 }])
        })
    }
}

/// Detect the image circle shared by `frames` (paths to images of one camera). Uses up to 12
/// evenly spaced frames; frames of another size than the first are skipped. None when there is no
/// circle (most lenses).
pub fn detect(frames: &[impl AsRef<Path>]) -> Result<Option<Circle>> {
    if frames.is_empty() {
        return Ok(None);
    }
    let n = frames.len().min(12);
    let picks: Vec<&Path> = (0..n).map(|i| frames[i * (frames.len() - 1) / (n - 1).max(1)].as_ref()).collect();
    let (w, h) = image::image_dimensions(picks[0]).with_context(|| format!("reading {}", picks[0].display()))?;
    let mut lumas = Vec::with_capacity(n);
    for p in picks {
        let img = image::open(p).with_context(|| format!("reading {}", p.display()))?;
        if (img.width(), img.height()) == (w, h) {
            lumas.push(img.to_luma8());
        }
    }
    Ok(detect_luma(&lumas))
}

/// One circle per frame (each frame analysed alone), for multi-camera rigs whose lenses put the
/// circle in slightly different places: the shared test fails on such a set because the per-pixel
/// maximum is the union of the circles. None for frames without a circle or unreadable ones.
pub fn detect_each(frames: &[impl AsRef<Path> + Sync]) -> Vec<Option<Circle>> {
    use rayon::prelude::*;
    frames
        .par_iter()
        .map(|p| image::open(p.as_ref()).ok().and_then(|img| detect_luma(&[img.to_luma8()])))
        .collect()
}

/// [`detect`] on decoded luma frames of one size.
pub fn detect_luma(frames: &[GrayImage]) -> Option<Circle> {
    let first = frames.first()?;
    let (w, h) = first.dimensions();
    if w < 16 || h < 16 || frames.iter().any(|f| f.dimensions() != (w, h)) {
        return None;
    }
    let scale = (ANALYSIS_EDGE as f64 / w.max(h) as f64).min(1.0);
    let (aw, ah) = (((w as f64 * scale).round() as u32).max(8), ((h as f64 * scale).round() as u32).max(8));
    let mut max = GrayImage::new(aw, ah);
    for f in frames {
        // Triangle filtering averages a rim pixel with its lit neighbours: the dark test stays
        // conservative (a mixed pixel counts as lit).
        let small = if (aw, ah) == (w, h) { f.clone() } else { image::imageops::resize(f, aw, ah, FilterType::Triangle) };
        for (m, s) in max.iter_mut().zip(small.iter()) {
            *m = (*m).max(*s);
        }
    }
    let (sx, sy) = (w as f64 / aw as f64, h as f64 / ah as f64);
    let fit = fit_small(&max)?;
    Some(Circle {
        cx: fit.0 * sx,
        cy: fit.1 * sy,
        radius: fit.2 * 0.5 * (sx + sy),
        width: w,
        height: h,
        outside_fraction: fit.3,
    })
}

/// (cx, cy, radius, outside share) on the analysis grid, pixel-centre coordinates.
fn fit_small(max: &GrayImage) -> Option<(f64, f64, f64, f64)> {
    let (w, h) = max.dimensions();
    let idx = |x: u32, y: u32| (y * w + x) as usize;
    let dark: Vec<bool> = max.iter().map(|v| *v <= DARK).collect();
    // Unlit pixels connected to the frame border.
    let mut outside = vec![false; dark.len()];
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for x in 0..w {
        stack.push((x, 0));
        stack.push((x, h - 1));
    }
    for y in 0..h {
        stack.push((0, y));
        stack.push((w - 1, y));
    }
    while let Some((x, y)) = stack.pop() {
        let i = idx(x, y);
        if outside[i] || !dark[i] {
            continue;
        }
        outside[i] = true;
        if x > 0 { stack.push((x - 1, y)) }
        if x + 1 < w { stack.push((x + 1, y)) }
        if y > 0 { stack.push((x, y - 1)) }
        if y + 1 < h { stack.push((x, y + 1)) }
    }
    let n_out = outside.iter().filter(|v| **v).count();
    let outside_fraction = n_out as f64 / outside.len() as f64;
    if outside_fraction < MIN_OUTSIDE || outside_fraction > 0.9 {
        return None;
    }
    // Rim samples: lit pixels with an outside 4-neighbour, away from the frame border.
    let mut pts = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if outside[idx(x, y)] {
                continue;
            }
            if outside[idx(x - 1, y)] || outside[idx(x + 1, y)] || outside[idx(x, y - 1)] || outside[idx(x, y + 1)] {
                pts.push((x as f64 + 0.5, y as f64 + 0.5));
            }
        }
    }
    if pts.len() < 40 {
        return None;
    }
    // One refit without outliers: dark scene content touching the rim (a black sofa at the edge
    // of a single frame) joins the outside region and bends the first fit.
    let (cx, cy, r) = kasa(&pts)?;
    let inliers: Vec<(f64, f64)> = pts.iter().copied().filter(|(x, y)| ((x - cx).hypot(y - cy) - r).abs() <= 2.0_f64.max(0.03 * r)).collect();
    if inliers.len() * 5 < pts.len() * 4 {
        return None;
    }
    let pts = inliers;
    let (cx, cy, r) = kasa(&pts)?;
    if !(r.is_finite() && r >= 0.25 * w.min(h) as f64 && r <= w.max(h) as f64) {
        return None;
    }
    let mut res: Vec<f64> = pts.iter().map(|(x, y)| ((x - cx).hypot(y - cy) - r).abs()).collect();
    res.sort_by(f64::total_cmp);
    let (median, p90) = (res[res.len() / 2], res[res.len() * 9 / 10]);
    if median > 1.5_f64.max(0.01 * r) || p90 > 0.03 * r + 1.0 {
        return None;
    }
    // The circle must explain the dark region (letterbox bars and dark corners of a lit scene do not).
    let mut agree = 0usize;
    let mut rim = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let d = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy);
            agree += usize::from((d > r) == outside[idx(x, y)]);
            if (0.9 * r..0.96 * r).contains(&d) {
                rim.push(max.get_pixel(x, y)[0]);
            }
        }
    }
    if (agree as f64) < MIN_AGREEMENT * outside.len() as f64 || rim.len() < 20 {
        return None;
    }
    rim.sort_unstable();
    if (rim[rim.len() / 2] as f64) < MIN_RIM_LUMA {
        return None;
    }
    Some((cx, cy, r, outside_fraction))
}

/// Algebraic least-squares circle (Kasa): x^2 + y^2 + D x + E y + F = 0.
fn kasa(pts: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let (mut a, mut b) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
    for &(x, y) in pts {
        let row = [x, y, 1.0];
        let z = -(x * x + y * y);
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += row[i] * row[j];
            }
            b[i] += row[i] * z;
        }
    }
    let [d, e, f] = solve3(a, b)?;
    let (cx, cy) = (-d / 2.0, -e / 2.0);
    let r2 = cx * cx + cy * cy - f;
    (r2 > 0.0).then(|| (cx, cy, r2.sqrt()))
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..3 {
            let m = a[r][c] / a[c][c];
            for k in c..3 {
                a[r][k] -= m * a[c][k];
            }
            b[r] -= m * b[c];
        }
    }
    let mut x = [0.0; 3];
    for c in (0..3).rev() {
        x[c] = (b[c] - (c + 1..3).map(|k| a[c][k] * x[k]).sum::<f64>()) / a[c][c];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured frame with everything outside the circle black.
    fn disc(w: u32, h: u32, cx: f64, cy: f64, r: f64, seed: u32, outside: u8) -> GrayImage {
        GrayImage::from_fn(w, h, |x, y| {
            let d = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy);
            let tex = 60 + ((x * 7 + y * 13 + seed * 31) % 150) as u8;
            Luma([if d <= r { tex } else { outside }])
        })
    }

    #[test]
    fn finds_a_full_circle() {
        let frames: Vec<_> = (0..5).map(|s| disc(1280, 960, 640.0, 480.0, 450.0, s, 3)).collect();
        let c = detect_luma(&frames).unwrap();
        assert!((c.cx - 640.0).abs() < 6.0 && (c.cy - 480.0).abs() < 6.0 && (c.radius - 450.0).abs() < 8.0, "{c:?}");
        let m = c.mask();
        assert_eq!(m.get_pixel(640, 480)[0], 255);
        assert_eq!(m.get_pixel(5, 5)[0], 0);
        // The margin keeps the rim out.
        assert_eq!(m.get_pixel((640.0 + 448.0) as u32, 480)[0], 0);
    }

    #[test]
    fn finds_a_cropped_circle() {
        // Circle wider than the frame height: only the corners are dark.
        let frames: Vec<_> = (0..4).map(|s| disc(1920, 1080, 960.0, 540.0, 900.0, s, 0)).collect();
        let c = detect_luma(&frames).unwrap();
        assert!((c.radius - 900.0).abs() < 15.0, "{c:?}");
        assert!(c.outside_fraction > 0.05 && c.outside_fraction < 0.3);
        let half = c.scaled(960, 540);
        assert!((half.radius - c.radius / 2.0).abs() < 1e-9 && half.width == 960);
    }

    #[test]
    fn no_circle_for_full_frames_letterbox_or_dark_vignette() {
        let full: Vec<_> = (0..3).map(|s| disc(640, 480, 320.0, 240.0, 2000.0, s, 0)).collect();
        assert!(detect_luma(&full).is_none());
        let letterbox: Vec<_> = (0..3)
            .map(|s| GrayImage::from_fn(640, 480, |x, y| Luma([if !(60..420).contains(&y) { 0 } else { 60 + ((x + y + s) % 100) as u8 }])))
            .collect();
        assert!(detect_luma(&letterbox).is_none());
        // A night scene: dark corners fading out, nothing bright near the edge of the dark area.
        let night: Vec<_> = (0..3)
            .map(|_| {
                GrayImage::from_fn(640, 480, |x, y| {
                    let d = (x as f64 - 320.0).hypot(y as f64 - 240.0) / 400.0;
                    Luma([(40.0 * (1.0 - d).max(0.0)) as u8])
                })
            })
            .collect();
        assert!(detect_luma(&night).is_none());
    }

    #[test]
    fn per_frame_circles_for_a_rig_whose_lenses_differ() {
        let dir = tempfile::tempdir().unwrap();
        let mut paths = vec![];
        for (i, cx) in [300.0, 340.0, 300.0, 340.0].into_iter().enumerate() {
            let p = dir.path().join(format!("f{i}.png"));
            disc(640, 480, cx, 240.0, 200.0, i as u32, 0).save(&p).unwrap();
            paths.push(p);
        }
        // The union of two offset circles is not a circle...
        assert!(detect(&paths).unwrap().is_none());
        // ...but each frame has its own.
        let each = detect_each(&paths);
        assert!(each.iter().all(Option::is_some));
        assert!((each[1].unwrap().cx - 340.0).abs() < 4.0 && (each[0].unwrap().cx - 300.0).abs() < 4.0);
    }

    #[test]
    fn dark_content_at_the_rim_of_one_frame_is_tolerated() {
        // A black object inside the circle touching its rim joins the outside region.
        let mut f = disc(640, 480, 320.0, 240.0, 200.0, 1, 0);
        for y in 200..280 {
            for x in 110..150 {
                f.put_pixel(x, y, Luma([0]));
            }
        }
        let c = detect_luma(&[f]).unwrap();
        assert!((c.cx - 320.0).abs() < 4.0 && (c.radius - 200.0).abs() < 5.0, "{c:?}");
    }

    #[test]
    fn a_moving_bright_spot_outside_is_still_dark_most_frames_but_breaks_detection_safely() {
        // One frame lights the outside area: the per-pixel max is no longer dark there, so the
        // detector reports no circle rather than a wrong one.
        let mut frames: Vec<_> = (0..3).map(|s| disc(640, 480, 320.0, 240.0, 220.0, s, 0)).collect();
        frames.push(disc(640, 480, 320.0, 240.0, 2000.0, 9, 0));
        assert!(detect_luma(&frames).is_none());
    }
}
