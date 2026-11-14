//! Fisheye input: the image-circle masks (black sensor area outside a circular fisheye's disc) for
//! SfM features and the training loss, and the evidence `--camera-model auto` uses to pick
//! OPENCV_FISHEYE.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::GrayImage;
use serde_json::json;
use splat_eval::lens_circle::Circle;

use crate::util::*;

/// Written into the SfM dataset when lens masks were made (the circle and how many masks).
pub(crate) const RECORD: &str = "lens_circle.json";
/// COLMAP-named (`<image name>.png`) feature masks, beside the dataset's images.
pub(crate) const FEATURE_MASKS: &str = "lens_masks";
/// The image-circle masks alone (`<image stem>.png`), for held-out scoring: the trainer never
/// sees outside the circle, so held-out metrics don't score it either (that area renders
/// unsupervised content where the ground truth is black; playroom scored 15.1 dB unmasked).
pub(crate) const EVAL_MASKS: &str = "lens_eval_masks";

/// The dataset's held-out scoring masks, when it has a circular fisheye's.
pub(crate) fn eval_mask_dir(dataset: &Path) -> Option<PathBuf> {
    Some(dataset.join(EVAL_MASKS)).filter(|d| d.is_dir())
}

/// The held-out scoring masks for a candidate's train_job.json: its dataset's lens masks, except
/// for surfel jobs (they render an undistorted pinhole copy, where fisheye-space masks don't apply).
pub(crate) fn eval_masks_for_job(candidate_dir: &Path) -> Option<PathBuf> {
    let job: serde_json::Value = read_json(&candidate_dir.join("train_job.json")).ok()?;
    if job["surfel"] == true {
        return None;
    }
    eval_mask_dir(Path::new(job["dataset"].as_str()?))
}

/// The image circles of a set of frames.
#[derive(Clone, Debug)]
pub(crate) struct Circles {
    /// The circle every frame shares, or (rigs) a typical one: the per-frame circle of median radius.
    pub representative: Circle,
    /// Per image (path under images/), when the frames don't share one circle: a multi-camera rig
    /// whose lenses sit slightly differently. None: `representative` applies to every frame.
    pub per_image: Option<BTreeMap<PathBuf, Circle>>,
}

/// Least share of sampled frames that must each show a circle for a per-frame (rig) result.
const PER_FRAME_MIN_SHARE: f64 = 2.0 / 3.0;

/// The image circles of the images under `images`, if they show circular-fisheye circles. One
/// circle shared by all frames (the per-pixel maximum over a sample; robust to dark scene content)
/// is tried first; otherwise, when most sampled frames each show a circle (a rig of fisheyes), every
/// frame gets its own, and a frame where none is found takes the previous frame's (frames of one
/// camera are usually consecutive), else the representative one. Failures to read count as no
/// circle: the masks are an improvement, never a reason to fail a solve.
pub(crate) fn detect(images: &Path) -> Option<Circles> {
    let frames = list_images(images).ok()?;
    match splat_eval::lens_circle::detect(&frames) {
        Ok(Some(c)) => return Some(Circles { representative: c, per_image: None }),
        Ok(None) => {}
        Err(e) => {
            eprintln!("[lens] image-circle check skipped: {e:#}");
            return None;
        }
    }
    let n = frames.len().min(12);
    if n < 2 {
        return None;
    }
    let sample: Vec<&PathBuf> = (0..n).map(|i| &frames[i * (frames.len() - 1) / (n - 1)]).collect();
    let found: Vec<Circle> = splat_eval::lens_circle::detect_each(&sample).into_iter().flatten().collect();
    if (found.len() as f64) < PER_FRAME_MIN_SHARE * n as f64 {
        return None;
    }
    let mut by_radius = found.clone();
    by_radius.sort_by(|a, b| a.radius.total_cmp(&b.radius));
    let representative = by_radius[by_radius.len() / 2];
    let each = splat_eval::lens_circle::detect_each(&frames);
    let mut per_image = BTreeMap::new();
    let mut previous: Option<Circle> = None;
    for (path, c) in frames.iter().zip(each) {
        let size = image::image_dimensions(path).ok();
        let c = c.or(previous.filter(|p| Some((p.width, p.height)) == size));
        if let Some(c) = c {
            previous = Some(c);
            per_image.insert(path.strip_prefix(images).ok()?.to_path_buf(), c);
        }
    }
    println!("[lens] frames don't share one image circle; per-frame circles for {} of {} frames (typical r={:.0}px)",
        per_image.len(), frames.len(), representative.radius);
    Some(Circles { representative, per_image: Some(per_image) })
}

/// Min of two masks of one size (white keeps).
fn and(mask: &mut GrayImage, other: &GrayImage) {
    for (m, o) in mask.iter_mut().zip(other.iter()) {
        *m = (*m).min(*o);
    }
}

/// The circle's mask at `size`, cached; None when `size` has another aspect ratio than the
/// circle's frames (a different camera).
fn mask_at(circle: &Circle, size: (u32, u32), cache: &mut BTreeMap<(u32, u32), Option<GrayImage>>) -> Option<GrayImage> {
    cache
        .entry(size)
        .or_insert_with(|| {
            let same_aspect = (size.0 as f64 * circle.height as f64 / (size.1 as f64 * circle.width as f64) - 1.0).abs() < 0.01;
            same_aspect.then(|| circle.scaled(size.0, size.1).mask())
        })
        .clone()
}

/// Save `mask` at `path` without writing through an existing hardlink (training masks are linked
/// from the mask artifact; overwriting in place would change the source).
fn replace(path: &Path, mask: &GrayImage) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let tmp = path.with_extension("lens.tmp.png");
    mask.save(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Lens masks for every image of the dataset `dir` (images/): training masks at `masks/<stem>.png`
/// (combined with any already there) and, when `features` is set, COLMAP feature masks at
/// `lens_masks/<name>.png` combined with the prior feature masks in `prior` (COLMAP or stem
/// names). Returns the feature mask dir. Records the circle in [`RECORD`].
pub(crate) fn write_masks(dir: &Path, circles: &Circles, features: bool, prior: Option<&Path>) -> Result<Option<PathBuf>> {
    let circle = &circles.representative;
    let images = dir.join("images");
    let feature_dir = dir.join(FEATURE_MASKS);
    let mut cache = BTreeMap::new();
    let (mut training, mut feature, mut skipped) = (0usize, 0usize, 0usize);
    for src in list_images(&images)? {
        let name = src.strip_prefix(&images)?.to_path_buf();
        let size = image::image_dimensions(&src)?;
        let own = circles.per_image.as_ref().map(|m| m.get(&name).copied());
        let mask = match own {
            // Rig: this frame's circle (none found: no mask).
            Some(Some(c)) => Some(c.scaled(size.0, size.1).mask()),
            Some(None) => None,
            None => mask_at(circle, size, &mut cache),
        };
        let Some(mask) = mask else {
            skipped += 1;
            continue;
        };
        let train_path = dir.join("masks").join(name.with_extension("png"));
        let mut train = mask.clone();
        if train_path.is_file() {
            let existing = image::open(&train_path)?.to_luma8();
            if existing.dimensions() == size {
                and(&mut train, &existing);
            }
        }
        replace(&train_path, &train)?;
        let stem = name.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        replace(&dir.join(EVAL_MASKS).join(format!("{stem}.png")), &mask)?;
        training += 1;
        if features {
            let mut feat = mask;
            let prior_mask = prior.and_then(|p| {
                [p.join(format!("{}.png", name.to_string_lossy())), p.join(name.with_extension("png"))].into_iter().find(|m| m.is_file())
            });
            if let Some(prior_mask) = prior_mask {
                let existing = image::open(&prior_mask)?.to_luma8();
                if existing.dimensions() == size {
                    and(&mut feat, &existing);
                }
            }
            replace(&feature_dir.join(format!("{}.png", name.to_string_lossy())), &feat)?;
            feature += 1;
        }
    }
    // Detector masks merged with the circle must stay on: the trainer's auto-mask gate would
    // otherwise switch the merged masks off (statue probe) and train the black area again.
    let marker = dir.join(crate::train::DETECTOR_MASKS_MARKER);
    let pinned = training > 0 && marker.is_file();
    if pinned {
        std::fs::remove_file(&marker)?;
    }
    write_json(&dir.join(RECORD), &json!({
        "detector_masks_pinned": pinned,
        "circle": circle, "per_frame": circles.per_image.is_some(), "mask_margin": splat_eval::lens_circle::MASK_MARGIN,
        "training_masks": training, "feature_masks": feature, "skipped_other_size": skipped,
    }))?;
    println!("[lens] {} image circle r={:.0}px at ({:.0}, {:.0}) on {}x{}: {training} training masks, {feature} feature masks",
        if circles.per_image.is_some() { "typical" } else { "shared" }, circle.radius, circle.cx, circle.cy, circle.width, circle.height);
    Ok((features && feature > 0).then_some(feature_dir))
}

/// A starting focal length for a fisheye solve without EXIF, as COLMAP's
/// `default_focal_length_factor` (focal = factor * longest image side). With an image circle the
/// lens reaches about 95 degrees at the rim (typical circular fisheyes cover 180-200 degrees);
/// without one, a full-frame fisheye covers about 150 degrees across its long side.
pub(crate) fn focal_factor(circle: Option<&Circle>) -> f64 {
    match circle {
        Some(c) => c.radius / 95f64.to_radians() / c.width.max(c.height) as f64,
        None => 0.5 / 75f64.to_radians(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle() -> Circle {
        Circle { cx: 32.0, cy: 24.0, radius: 20.0, width: 64, height: 48, outside_fraction: 0.6 }
    }

    #[test]
    fn masks_combine_with_existing_training_and_feature_masks_without_touching_links() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("sfm");
        std::fs::create_dir_all(dir.join("images/sub")).unwrap();
        std::fs::create_dir_all(dir.join("masks")).unwrap();
        for name in ["a.jpg", "sub/b.jpg"] {
            image::RgbImage::new(64, 48).save(dir.join("images").join(name)).unwrap();
        }
        image::RgbImage::new(32, 32).save(dir.join("images/other.jpg")).unwrap();
        // A distractor mask hiding the top-left quarter, hardlinked from a mask artifact.
        let source = tmp.path().join("artifact_a.png");
        GrayImage::from_fn(64, 48, |x, y| image::Luma([if x < 32 && y < 24 { 0 } else { 255 }])).save(&source).unwrap();
        std::fs::hard_link(&source, dir.join("masks/a.png")).unwrap();
        std::fs::write(dir.join(crate::train::DETECTOR_MASKS_MARKER), "{}").unwrap();
        let prior = tmp.path().join("prior");
        std::fs::create_dir_all(prior.join("sub")).unwrap();
        GrayImage::from_fn(64, 48, |x, _| image::Luma([if x > 40 { 0 } else { 255 }])).save(prior.join("sub/b.jpg.png")).unwrap();
        let features = write_masks(&dir, &Circles { representative: circle(), per_image: None }, true, Some(&prior)).unwrap().unwrap();
        let a = image::open(dir.join("masks/a.png")).unwrap().to_luma8();
        assert_eq!((a.get_pixel(30, 20)[0], a.get_pixel(34, 26)[0], a.get_pixel(1, 1)[0]), (0, 255, 0));
        // The artifact's mask is unchanged.
        assert_eq!(image::open(&source).unwrap().to_luma8().get_pixel(34, 26)[0], 255);
        assert_eq!(image::open(&source).unwrap().to_luma8().get_pixel(1, 47)[0], 255);
        let b = image::open(features.join("sub/b.jpg.png")).unwrap().to_luma8();
        assert_eq!((b.get_pixel(32, 24)[0], b.get_pixel(45, 24)[0], b.get_pixel(1, 1)[0]), (255, 0, 0));
        assert!(dir.join("masks/sub/b.png").is_file() && features.join("a.jpg.png").is_file());
        // Held-out scoring masks hold the circle alone (no distractor mask), by image stem.
        let eval = image::open(dir.join(EVAL_MASKS).join("a.png")).unwrap().to_luma8();
        assert_eq!((eval.get_pixel(30, 20)[0], eval.get_pixel(1, 1)[0]), (255, 0));
        assert_eq!(eval_mask_dir(&dir), Some(dir.join(EVAL_MASKS)));
        // A frame of another shape gets no mask.
        assert!(!dir.join("masks/other.png").exists());
        let record: serde_json::Value = read_json(&dir.join(RECORD)).unwrap();
        assert_eq!((record["training_masks"].as_u64(), record["skipped_other_size"].as_u64()), (Some(2), Some(1)));
        // The merged masks no longer carry the auto-unmask marker.
        assert!(!dir.join(crate::train::DETECTOR_MASKS_MARKER).exists() && record["detector_masks_pinned"] == true);
    }

    #[test]
    fn rigs_get_one_circle_per_frame() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("sfm");
        std::fs::create_dir_all(dir.join("images")).unwrap();
        // Two "cameras" whose circles sit 40 px apart, alternating frames.
        for i in 0..6u32 {
            let cx = if i % 2 == 0 { 300.0 } else { 340.0 };
            image::RgbImage::from_fn(640, 480, |x, y| {
                let d = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - 240.0);
                image::Rgb(if d > 200.0 { [0, 0, 0] } else { [60 + ((x * 7 + y * 3 + i) % 150) as u8, 90, 120] })
            }).save(dir.join(format!("images/f{i}.png"))).unwrap();
        }
        let circles = detect(&dir.join("images")).unwrap();
        let per = circles.per_image.as_ref().unwrap();
        assert_eq!(per.len(), 6);
        assert!((per[Path::new("f1.png")].cx - 340.0).abs() < 4.0 && (per[Path::new("f2.png")].cx - 300.0).abs() < 4.0);
        write_masks(&dir, &circles, false, None).unwrap();
        let m1 = image::open(dir.join("masks/f1.png")).unwrap().to_luma8();
        let m0 = image::open(dir.join("masks/f0.png")).unwrap().to_luma8();
        // x = 520 is inside camera B's circle (340 + 180) but outside camera A's (300 + 220 > 200).
        assert_eq!((m1.get_pixel(520, 240)[0], m0.get_pixel(520, 240)[0]), (255, 0));
    }

    #[test]
    fn focal_prior_follows_the_circle() {
        // A 190 degree circular fisheye filling 1080 px of height: f ~ 540 / 1.66.
        let c = Circle { cx: 960.0, cy: 540.0, radius: 540.0, width: 1920, height: 1080, outside_fraction: 0.5 };
        assert!((focal_factor(Some(&c)) * 1920.0 - 325.7).abs() < 1.0);
        assert!((focal_factor(None) - 0.382).abs() < 1e-3);
    }
}
