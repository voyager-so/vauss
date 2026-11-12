//! Per-view depth priors for training (goal 3, rooms): filled in a background
//! thread while training starts, so the first steps never wait on them.
//!
//! Source: sensor depth in the dataset (capture-app import): `depths.json` plus
//! `depths/<image stem>.png` (uint16 millimetres, z along the optical axis,
//! 0 = invalid). Stored as `1 / z`.
//!
//! On by default only with sensor depth. `SPLAT_DEPTH_PRIOR=0` turns the prior off; a number
//! sets its weight.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::Context;
use brush_dataset::scene::{DepthMap, SceneView};
use brush_train::depth_prior::{flat_weight, prior_size, resize_nearest};

/// Weight override from `SPLAT_DEPTH_PRIOR` (0 = off).
pub fn weight_override() -> anyhow::Result<Option<f32>> {
    std::env::var("SPLAT_DEPTH_PRIOR")
        .ok()
        .map(|v| v.trim().parse::<f32>().context("SPLAT_DEPTH_PRIOR must be a number (0 = off)"))
        .transpose()
}

/// Give every view an empty depth slot and start filling them. Returns the
/// worker; it logs its own progress and failures (a failure leaves the slots
/// empty: training runs without the prior).
pub fn attach(views: &mut [SceneView], dataset: &Path) -> std::thread::JoinHandle<()> {
    let mut work = Vec::with_capacity(views.len());
    for v in views.iter_mut() {
        let slot = Arc::new(OnceLock::new());
        v.depth = Some(slot.clone());
        work.push((image_file(dataset, v), slot));
    }
    let dataset = dataset.to_path_buf();
    std::thread::spawn(move || {
        let start = Instant::now();
        let n = work.len();
        match fill_from_files(&dataset, &work) {
            Ok(k) => eprintln!(
                "depth prior: {k}/{n} views from sensor in {:.1}s",
                start.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!("depth prior: off ({e:#})"),
        }
    })
}

fn image_file(dataset: &Path, v: &SceneView) -> PathBuf {
    let p = v.image.path();
    if p.is_absolute() { p.to_path_buf() } else { dataset.join(p) }
}

fn fill_from_files(dataset: &Path, work: &[(PathBuf, brush_dataset::scene::DepthSlot)]) -> anyhow::Result<usize> {
    let mut k = 0;
    for (img, slot) in work {
        let stem = img.file_stem().context("image without a name")?;
        let file = dataset.join("depths").join(stem).with_extension("png");
        let Ok(depth) = image::open(&file) else { continue };
        let depth = depth.into_luma16();
        let (dw, dh) = (depth.width() as usize, depth.height() as usize);
        let inv: Vec<f32> = depth.into_raw().into_iter().map(|mm| if mm == 0 { 0.0 } else { 1000.0 / mm as f32 }).collect();
        let (iw, ih) = image::image_dimensions(img).with_context(|| format!("reading {}", img.display()))?;
        let (w, h) = prior_size(iw as usize, ih as usize);
        let flat = gray_at(img, w, h).map(|g| flat_weight(&g, w, h)).unwrap_or_default();
        let _ = slot.set(Arc::new(DepthMap { w, h, inv_depth: resize_nearest(&inv, dw, dh, w, h), flat_weight: flat }));
        k += 1;
    }
    Ok(k)
}

/// Luminance of the photo at `(w, h)`.
fn gray_at(img: &Path, w: usize, h: usize) -> Option<Vec<f32>> {
    let small = image::open(img).ok()?.resize_exact(w as u32, h as u32, image::imageops::FilterType::Triangle).to_rgb32f();
    Some(small.pixels().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect())
}
