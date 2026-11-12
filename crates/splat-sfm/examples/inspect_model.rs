//! CPU-only density audit; all reprojection calculations are delegated to splat-eval.
use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use serde_json::json;
use splat_eval::{SfmHealthOptions, read_model, sfm_health};

#[derive(Parser)]
struct Args {
    sparse: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long)]
    reference: Option<PathBuf>,
    #[arg(long)]
    video: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut model = read_model(&args.sparse)?;
    for p in &mut model.points {
        p.error = 0.0;
    }
    let opts = SfmHealthOptions {
        images_dir: Some(args.images),
        video: args.video,
        ..Default::default()
    };
    let health = sfm_health(&model, &args.sparse, &opts)?;
    let mut histogram = BTreeMap::new();
    for point in &model.points {
        *histogram.entry(point.track_len).or_insert(0usize) += 1;
    }
    let mut groups = BTreeMap::new();
    for (name, two_view) in [("two_view", true), ("multi_view", false)] {
        let mut group = model.clone();
        group.points.retain(|p| (p.track_len == 2) == two_view);
        let h = sfm_health(&group, &args.sparse, &opts)?;
        groups.insert(name, json!({ "n_points": h.n_points, "mean_reproj_px": h.mean_reproj_err_px, "median_reproj_px": h.median_reproj_err_px }));
    }
    let unchanged = args.reference.map(|path| -> Result<_> {
        let reference = read_model(&path)?;
        let cameras = model.cameras.len() == reference.cameras.len() && model.cameras.iter().all(|(id, c)| reference.cameras.get(id).is_some_and(|r| c.model == r.model && c.width == r.width && c.height == r.height && c.params == r.params));
        let poses = model.images.len() == reference.images.len() && model.images.iter().all(|(id, im)| reference.images.get(id).is_some_and(|r| im.name == r.name && im.camera_id == r.camera_id && im.qvec == r.qvec && im.tvec == r.tvec));
        let mut rotation_delta = 0.0_f64;
        let mut translation_delta = 0.0_f64;
        for (id, im) in &model.images {
            if let Some(r) = reference.images.get(id) {
                let a = splat_eval::Model::rotation(im.qvec);
                let b = splat_eval::Model::rotation(r.qvec);
                for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
                    rotation_delta = rotation_delta.max((a - b).abs());
                }
                for (a, b) in im.tvec.iter().zip(r.tvec) {
                    translation_delta = translation_delta.max((a - b).abs());
                }
            }
        }
        Ok(json!({ "reference": path, "cameras_exactly_equal": cameras, "poses_exactly_equal": poses, "max_rotation_matrix_delta": rotation_delta, "max_translation_delta": translation_delta }))
    }).transpose()?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({ "health": health, "track_histogram": histogram, "groups": groups, "unchanged": unchanged })
        )?
    );
    Ok(())
}
