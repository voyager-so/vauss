//! Sanity statistics for distractor masks. A mask meant for people/cars that covers the frame
//! centre or one huge blob has probably caught the subject (seen on pegasus: "car. truck. bus.
//! person." masked the statue).

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

/// Centre coverage above this, or one connected blob above LARGEST_SUSPECT of the frame, flags it.
pub const CENTRE_SUSPECT: f64 = 0.5;
pub const LARGEST_SUSPECT: f64 = 0.25;
/// Connected components are found on a copy at most this wide (enough for blob sizes, and cheap).
const WORK_WIDTH: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MaskStats {
    /// Fraction of the frame that is masked.
    pub coverage: f64,
    /// Fraction of the central third (both axes) that is masked.
    pub centre_coverage: f64,
    /// Largest 4-connected masked blob as a fraction of the frame.
    pub largest_component_frac: f64,
    pub object_masked_suspect: bool,
}

/// `masked` is row-major, true = masked out (the distractor).
pub fn mask_stats(width: usize, height: usize, masked: &[bool]) -> MaskStats {
    let n = width * height;
    if n == 0 || masked.len() != n {
        return MaskStats::default();
    }
    let coverage = masked.iter().filter(|&&m| m).count() as f64 / n as f64;
    let (x0, x1, y0, y1) = (width / 3, width - width / 3, height / 3, height - height / 3);
    let centre_n = ((x1 - x0) * (y1 - y0)).max(1);
    let centre = (y0..y1).flat_map(|y| (x0..x1).map(move |x| y * width + x)).filter(|&i| masked[i]).count();
    let centre_coverage = centre as f64 / centre_n as f64;

    let step = width.div_ceil(WORK_WIDTH).max(1);
    let (w, h) = (width.div_ceil(step), height.div_ceil(step));
    let small: Vec<bool> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| masked[(y * step) * width + x * step]).collect();
    let largest_component_frac = largest_component(w, h, &small) as f64 / (w * h) as f64;

    MaskStats {
        coverage,
        centre_coverage,
        largest_component_frac,
        object_masked_suspect: centre_coverage > CENTRE_SUSPECT || largest_component_frac > LARGEST_SUSPECT,
    }
}

/// Mask image on disk; pixels >= 128 are masked when `white_is_masked`, else pixels < 128.
pub fn mask_stats_file(path: &Path, white_is_masked: bool) -> Result<MaskStats> {
    let img = image::open(path).with_context(|| format!("reading mask {}", path.display()))?.into_luma8();
    let masked: Vec<bool> = img.as_raw().iter().map(|&v| (v >= 128) == white_is_masked).collect();
    Ok(mask_stats(img.width() as usize, img.height() as usize, &masked))
}

fn largest_component(w: usize, h: usize, on: &[bool]) -> usize {
    let mut seen = vec![false; on.len()];
    let mut stack = Vec::new();
    let mut best = 0;
    for start in 0..on.len() {
        if !on[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut size = 0;
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = (i % w, i / w);
            let neighbours = [
                (x > 0).then(|| i - 1),
                (x + 1 < w).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < h).then(|| i + w),
            ];
            for j in neighbours.into_iter().flatten() {
                if on[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        best = best.max(size);
    }
    best
}

/// Subject guard: the subject of an orbit sits in the
/// middle of the shot on most frames, real distractors wander through. A mask set is the subject
/// (a Spot robot clip: "person; car; bicycle" masked the robot) if (A) more than half of the centre
/// third is masked on more than CENTRE_HEAVY_FRAMES of frames, or (B) a centre-touching blob is
/// present on more than CENTRE_BLOB_FRAMES of frames. People walking through the middle (HPC
/// patio/spot ground truth) stay under both. Frames are judged on a GUARD_WIDTH-wide stride
/// subsample; the centre third is rows h/3..h-h/3, columns w/3..w-w/3 of that subsample.
pub const CENTRE_HEAVY: f64 = 0.5;
pub const CENTRE_HEAVY_FRAMES: f64 = 0.5;
pub const CENTRE_BLOB_FRAMES: f64 = 0.8;
/// A 4-connected blob counts when it covers at least BLOB_MIN_AREA of the frame and either covers
/// BLOB_CENTRE_TOUCH of the centre third or has at least half its area inside it.
pub const BLOB_MIN_AREA: f64 = 0.002;
pub const BLOB_CENTRE_TOUCH: f64 = 0.1;
const GUARD_WIDTH: usize = 128;

/// Per-frame centre statistics for the subject guard.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CentreFrame {
    /// Masked fraction of the centre third.
    pub centre_coverage: f64,
    /// A centre-touching blob is present.
    pub centre_blob: bool,
}

/// `masked` is row-major, true = masked out.
pub fn centre_frame(width: usize, height: usize, masked: &[bool]) -> CentreFrame {
    let step = width.div_ceil(GUARD_WIDTH).max(1);
    let (w, h) = (width.div_ceil(step), height.div_ceil(step));
    let small = (0..h).flat_map(|y| (0..w).map(move |x| masked[y * step * width + x * step])).collect::<Vec<_>>();
    let (y0, y1, x0, x1) = (h / 3, h - h / 3, w / 3, w - w / 3);
    let in_box = |i: usize| (y0..y1).contains(&(i / w)) && (x0..x1).contains(&(i % w));
    let box_area = (y1 - y0) * (x1 - x0);
    let box_masked = (0..w * h).filter(|&i| small[i] && in_box(i)).count();
    let mut label = vec![false; w * h];
    let mut centre_blob = false;
    for start in 0..w * h {
        if !small[start] || label[start] {
            continue;
        }
        label[start] = true;
        let (mut stack, mut area, mut inside) = (vec![start], 0usize, 0usize);
        while let Some(i) = stack.pop() {
            area += 1;
            inside += in_box(i) as usize;
            let (x, y) = (i % w, i / w);
            let next = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
            for j in next.into_iter().flatten() {
                if small[j] && !label[j] {
                    label[j] = true;
                    stack.push(j);
                }
            }
        }
        if area as f64 >= BLOB_MIN_AREA * (w * h) as f64
            && (inside as f64 >= BLOB_CENTRE_TOUCH * box_area as f64 || inside as f64 >= 0.5 * area as f64)
        {
            centre_blob = true;
            break;
        }
    }
    CentreFrame { centre_coverage: box_masked as f64 / box_area.max(1) as f64, centre_blob }
}

/// Set-level verdict of the subject guard.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubjectGuard {
    frames: usize,
    centre_heavy: usize,
    centre_blob: usize,
}

impl SubjectGuard {
    pub fn add(&mut self, frame: CentreFrame) {
        self.frames += 1;
        self.centre_heavy += (frame.centre_coverage > CENTRE_HEAVY) as usize;
        self.centre_blob += frame.centre_blob as usize;
    }

    /// Fraction of frames with more than CENTRE_HEAVY of the centre third masked (rule A).
    pub fn centre_heavy_frac(&self) -> f64 {
        self.centre_heavy as f64 / self.frames.max(1) as f64
    }

    /// Fraction of frames with a centre-touching blob (rule B).
    pub fn centre_blob_frac(&self) -> f64 {
        self.centre_blob as f64 / self.frames.max(1) as f64
    }

    /// The masks cover the subject: keep them for SfM features only.
    pub fn subject(&self) -> bool {
        self.centre_heavy_frac() > CENTRE_HEAVY_FRAMES || self.centre_blob_frac() > CENTRE_BLOB_FRAMES
    }
}

/// Masks a training loss should ignore, for a completed masks artifact directory. coverage.json's
/// `training_masks` names them (subject-guarded copies), or is null when the guard kept the masks
/// for SfM only; artifacts from before the guard train on `masks/`.
pub fn training_mask_dir(artifact: &Path) -> Result<Option<std::path::PathBuf>> {
    let path = artifact.join("coverage.json");
    let coverage: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?)?;
    Ok(match coverage.get("training_masks") {
        None => Some(artifact.join("masks")),
        Some(serde_json::Value::String(dir)) => Some(artifact.join(dir)),
        Some(_) => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, f: impl Fn(usize, usize) -> bool) -> Vec<bool> {
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect()
    }

    #[test]
    fn small_side_people_are_fine() {
        // two 20x40 blobs near the left and right edges of a 300x200 frame
        let m = frame(300, 200, |x, y| (10..30).contains(&x) && (100..140).contains(&y) || (260..280).contains(&x) && (20..60).contains(&y));
        let s = mask_stats(300, 200, &m);
        assert!((s.coverage - 1600.0 / 60000.0).abs() < 1e-12);
        assert_eq!(s.centre_coverage, 0.0);
        assert!(s.largest_component_frac < 0.02 && !s.object_masked_suspect);
    }

    #[test]
    fn centred_subject_is_suspect() {
        let m = frame(300, 200, |x, y| (90..210).contains(&x) && (50..150).contains(&y));
        let s = mask_stats(300, 200, &m);
        assert_eq!(s.centre_coverage, 1.0);
        assert!(s.object_masked_suspect);
    }

    #[test]
    fn one_huge_blob_off_centre_is_suspect() {
        let m = frame(600, 400, |x, _| x < 190);
        let s = mask_stats(600, 400, &m);
        assert_eq!(s.centre_coverage, 0.0);
        assert!(s.largest_component_frac > 0.3 && s.object_masked_suspect);
    }
}
