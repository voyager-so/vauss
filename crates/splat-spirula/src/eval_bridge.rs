//! Keep metric and COLMAP policy in splat-eval; adapt only JSON names from the reference backend.
use anyhow::Result;
use image::GrayImage;
use serde_json::Value;
use splat_eval::{MetricOptions, SfmHealthOptions};
use std::path::Path;

pub(crate) fn registered_names(path: &Path) -> Result<Vec<String>> {
    Ok(splat_eval::read_model(path)?.registered_names())
}
pub(crate) fn sfm_health(model: &Path, images: &Path, video: bool) -> Result<Value> {
    let opts = SfmHealthOptions {
        images_dir: Some(images.into()),
        video,
        ..Default::default()
    };
    Ok(serde_json::to_value(splat_eval::sfm_health(
        &splat_eval::read_model(model)?,
        model,
        &opts,
    )?)?)
}
pub(crate) fn summarize_frames(frames: &[GrayImage]) -> Result<Value> {
    let gray = frames
        .iter()
        .map(|f| splat_eval::video::Gray {
            width: f.width() as usize,
            height: f.height() as usize,
            data: f.as_raw().clone(),
        })
        .collect::<Vec<_>>();
    let mut result = serde_json::to_value(splat_eval::video::summarize_frames(&gray)?)?;
    result["static_fraction"] = result["static_frac"].clone();
    Ok(result)
}
pub(crate) fn evaluate(path: &Path, lpips: bool, models_dir: Option<&Path>) -> Result<Value> {
    let (pairs, missing) = splat_eval::spirula_pairs(path)?;
    let opts = MetricOptions {
        lpips,
        models_dir: models_dir.map(Path::to_path_buf),
        ..Default::default()
    };
    Ok(serde_json::to_value(splat_eval::evaluate_pairs(
        &pairs, &missing, &opts,
    )?)?)
}
