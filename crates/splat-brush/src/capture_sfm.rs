//! Prefer reconstructed cameras; retain sensor poses as a coverage fallback and
//! sensor depth as a training prior in the selected cameras' world units.
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use splat_contracts::{Artifact, ToolResult};
use splat_core::RunCtx;
use splat_eval::Model;

use crate::{
    capture::{self, Manifest},
    util::*,
};

pub(crate) fn is_capture(source: &Path) -> bool {
    source.join("capture.json").is_file()
}

/// Called on the client after the ordinary SfM worker has completed. A failed
/// solve also qualifies, but cancellation never starts a fallback.
pub(crate) fn complete(
    run: &dyn RunCtx,
    artifact: &Artifact,
    out: &Path,
    exit_code: i32,
) -> Result<Option<ToolResult>> {
    let frames_id = meta_str(artifact, "frames_id")?;
    let source = run.path(frames_id)?;
    if !is_capture(&source) || matches!(exit_code, 130 | 143) {
        return Ok(None);
    }
    let manifest: Manifest = read_json(&source.join("capture.json"))?;
    ensure!(
        manifest.frames.len() >= 2,
        "capture manifest has fewer than two frames"
    );
    let rec: Value = read_json(&out.join("reconstruction.json")).unwrap_or_default();
    let model = if exit_code == 0 && !rec.is_null() {
        splat_eval::read_model(&out.join("sparse/0")).unwrap_or_default()
    } else {
        Model::default()
    };
    let registered = model.images.len();
    let fallback = below_threshold(registered, manifest.frames.len());
    let selection = json!({
        "camera_source": if fallback { "capture_fallback" } else { "sfm" },
        "capture_format": manifest.format, "sfm_skipped": false,
        "sfm_registered": registered, "sfm_input": manifest.frames.len(),
        "sfm_registered_fraction": registered as f64 / manifest.frames.len() as f64,
        "capture_fallback_threshold": 0.9, "sfm_attempt_exit_code": exit_code,
        "sfm_wall_s": rec["wall_seconds"],
        "capture_pose_refinement_default": fallback,
        "sfm_attempt_failed": rec["capture_sfm_failed"].as_bool().unwrap_or(exit_code != 0),
        "sfm_attempt_error": rec["sfm_error"],
    });
    if fallback {
        // Preserve the failed/partial model as evidence, outside the loader's
        // sparse directory. Never mix its points with the metric phone poses.
        let sparse = out.join("sparse/0");
        if sparse.exists() && !out.join("sfm_attempt").exists() {
            fs::rename(&sparse, out.join("sfm_attempt"))?;
        }
        write_json(&out.join("capture_selection.json"), &selection)?;
        return capture::reconstruct(run, frames_id, artifact.clone(), out, &selection).map(Some);
    }
    capture::freeze_split(run, frames_id, artifact, &source, &manifest)?;
    if !out.join("capture.json").exists() {
        link_or_copy(&source.join("capture.json"), &out.join("capture.json"))?;
    }
    let scale = if source.join("depths.json").is_file() {
        Some(attach_depth(&source, out, &manifest, &model)?)
    } else {
        None
    };
    let mut selection = selection;
    if let Some(scale) = scale {
        selection["sensor_depth_alignment"] = scale;
    }
    write_json(&out.join("capture_selection.json"), &selection)?;
    Ok(None)
}

fn below_threshold(registered: usize, input: usize) -> bool {
    // Exactly 90% retains SfM, without floating-point boundary surprises.
    registered.saturating_mul(10) < input.saturating_mul(9)
}

pub(crate) fn training_overrides(
    dataset: &Path,
    preview: bool,
) -> Result<Option<serde_json::Map<String, Value>>> {
    if preview
        || !dataset.join("capture_selection.json").is_file()
        || std::env::var("SPLAT_POSE_REFINE").is_ok_and(|s| s.trim() != "1")
    {
        return Ok(None);
    }
    let selection: Value = read_json(&dataset.join("capture_selection.json"))?;
    if selection["camera_source"] != "capture_fallback" {
        return Ok(None);
    }
    let over = serde_json::Map::from_iter([
        ("pose_refine".into(), json!(true)),
        (
            "pose_align_eval_steps".into(),
            json!(crate::train::POSE_ALIGN_DEFAULT_STEPS),
        ),
    ]);
    Ok(Some(over))
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

/// Fit on TRAIN views only. Sparse observations give a direct depth-unit fit;
/// camera-centre distances are a fallback for exports with sparse/no depth hits.
fn depth_scale(source: &Path, manifest: &Manifest, model: &Model) -> Result<Value> {
    let names = model.registered_names();
    let heldout: BTreeSet<_> = manifest
        .heldout_names
        .clone()
        .unwrap_or_else(|| names.iter().step_by(8).cloned().collect())
        .into_iter()
        .collect();
    let frames: BTreeMap<_, _> = manifest
        .frames
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();
    let points: BTreeMap<_, _> = model
        .points
        .iter()
        .filter(|p| p.track_len >= 3 && p.error <= 3.0)
        .map(|p| (p.id, p))
        .collect();
    let mut ratios = vec![];
    let mut centers = vec![];
    let mut observations = 0;
    for image in model.images.values().filter(|i| !heldout.contains(&i.name)) {
        let Some(frame) = frames.get(image.name.as_str()) else {
            continue;
        };
        centers.push((
            frame.c2w.w_axis.truncate(),
            glam::DVec3::from_array(Model::center(image)),
        ));
        let path = source
            .join("depths")
            .join(Path::new(&image.name).with_extension("png"));
        if !path.is_file() {
            continue;
        }
        let depth = image::open(&path)?.to_luma16();
        let conf = source
            .join("depths_conf")
            .join(Path::new(&image.name).with_extension("png"));
        let conf = conf
            .is_file()
            .then(|| image::open(conf).map(|i| i.to_luma8()))
            .transpose()?;
        let camera = model
            .cameras
            .get(&image.camera_id)
            .context("SfM camera missing")?;
        let r = Model::rotation(image.qvec);
        let mut local = vec![];
        for (xy, id) in image
            .obs_xy
            .iter()
            .zip(&image.obs_point_id)
            .step_by(image.obs_xy.len().div_ceil(512).max(1))
        {
            let Some(point) = points.get(id) else {
                continue;
            };
            if !xy.iter().all(|v| v.is_finite())
                || xy[0] < 0.0
                || xy[1] < 0.0
                || xy[0] >= camera.width as f64
                || xy[1] >= camera.height as f64
            {
                continue;
            }
            let x = (xy[0] * depth.width() as f64 / camera.width as f64) as u32;
            let y = (xy[1] * depth.height() as f64 / camera.height as f64) as u32;
            if conf.as_ref().is_some_and(|c| {
                c.get_pixel(
                    x * c.width() / depth.width(),
                    y * c.height() / depth.height(),
                )[0] == 0
            }) {
                continue;
            }
            let sensor_z = depth.get_pixel(x, y)[0] as f64 * 0.001;
            let z = r[2].iter().zip(point.xyz).map(|(a, b)| a * b).sum::<f64>() + image.tvec[2];
            if sensor_z > 0.05 && z.is_finite() && z > 0.0 {
                local.push(z / sensor_z);
            }
        }
        if local.len() >= 16 {
            observations += local.len();
            ratios.push(median(&mut local).unwrap());
        }
    }
    let depth_frames = ratios.len();
    let method = if depth_frames >= 2 {
        "train_sparse_z_to_sensor_z"
    } else {
        ratios.clear();
        // Bound work while keeping widely spaced views for a stable baseline.
        let centers: Vec<_> = centers
            .iter()
            .step_by(centers.len().div_ceil(64).max(1))
            .collect();
        for (i, (a, b)) in centers.iter().enumerate() {
            for (c, d) in centers.iter().skip(i + 1) {
                let meters = a.distance(*c);
                let units = b.distance(*d);
                if meters >= 0.05 && units.is_finite() && units > 1e-6 {
                    ratios.push(units / meters);
                }
            }
        }
        "train_camera_distance_ratio"
    };
    let samples = ratios.len();
    let scale = median(&mut ratios)
        .context("cannot align sensor depth: no usable TRAIN sparse depths or camera baseline")?;
    ensure!(
        scale.is_finite() && scale > 0.0,
        "invalid sensor-depth scale"
    );
    let mut deviations: Vec<_> = ratios.iter().map(|r| (r / scale - 1.0).abs()).collect();
    Ok(
        json!({"units_per_meter":scale,"method":method,"train_samples":samples,
        "train_depth_frames":depth_frames,"train_depth_observations":observations,
        "median_relative_deviation":median(&mut deviations),"heldout_used":false}),
    )
}

fn attach_depth(source: &Path, out: &Path, manifest: &Manifest, model: &Model) -> Result<Value> {
    let mut alignment = depth_scale(source, manifest, model)?;
    let scale = alignment["units_per_meter"].as_f64().unwrap();
    fs::create_dir_all(out.join("depths"))?;
    let (mut valid, mut clipped) = (0u64, 0u64);
    for frame in manifest.frames.iter().filter(|f| f.has_depth) {
        let stem: PathBuf = Path::new(&frame.name).with_extension("png");
        let mut depth = image::open(source.join("depths").join(&stem))?.to_luma16();
        for value in depth.as_mut() {
            if *value == 0 {
                continue;
            }
            valid += 1;
            let z = (*value as f64 * scale).round();
            *value = if (1.0..=u16::MAX as f64).contains(&z) {
                z as u16
            } else {
                clipped += 1;
                0
            };
        }
        // Atomic replacement also breaks any prior hardlink, keeping source depth
        // immutable and repeated completion independent of the previous scale.
        let temp = out.join("depths").join(stem.with_extension("scaled.png"));
        depth.save(&temp)?;
        fs::rename(temp, out.join("depths").join(&stem))?;
        let conf = source.join("depths_conf").join(&stem);
        if conf.is_file() {
            fs::create_dir_all(out.join("depths_conf"))?;
            let dst = out.join("depths_conf").join(&stem);
            if !dst.exists() {
                link_or_copy(&conf, &dst)?;
            }
        }
    }
    ensure!(
        valid == 0 || clipped < valid,
        "scaled sensor depth has no representable values"
    );
    alignment["valid_pixels"] = json!(valid - clipped);
    alignment["clipped_pixels"] = json!(clipped);
    let mut metadata: Value = read_json(&source.join("depths.json"))?;
    metadata["kind"] = json!("metric");
    metadata["aligned_to_sfm"] = json!(true);
    metadata["units"] = json!("selected_world_units_x1000");
    metadata["units_per_meter"] = json!(scale);
    metadata["alignment"] = alignment.clone();
    metadata["scale_note"] = json!(
        "TRAIN-only scale fit; PNG/1000 is optical-axis z in the selected SfM world units. RGB pixel registration is preserved."
    );
    write_json(&out.join("depths.json"), &metadata)?;
    Ok(alignment)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sfm_coverage_boundary() {
        assert!(below_threshold(89, 100));
        assert!(!below_threshold(90, 100));
        assert!(!below_threshold(100, 100));
        assert!(below_threshold(8, 9));
        assert!(!below_threshold(9, 10));
        assert!(below_threshold(0, 100));
    }

    #[test]
    fn scale_excludes_heldout_depth_and_never_mutates_source() -> Result<()> {
        let source = tempfile::tempdir()?;
        let out = tempfile::tempdir()?;
        fs::create_dir(source.path().join("depths"))?;
        let frames: Vec<_> = (0..3)
            .map(|i| capture::Frame {
                name: format!("frame_{i:06}.jpg"),
                source_index: i,
                source_name: format!("{i}.jpg"),
                width: 480,
                height: 360,
                model: "PINHOLE".into(),
                params: vec![400., 400., 240., 180.],
                c2w: glam::DMat4::from_translation(glam::DVec3::new(i as f64, 0., 0.)),
                has_depth: true,
            })
            .collect();
        let manifest: Manifest = serde_json::from_value(json!({"schema_version":1,
            "format":"test","camera_convention":"opencv_c2w","units":"metres",
            "heldout_names":[frames[0].name],"frames":frames}))?;
        let mut model = Model::default();
        model.cameras.insert(
            1,
            splat_eval::colmap::Camera {
                id: 1,
                model: "PINHOLE".into(),
                width: 480,
                height: 360,
                params: vec![400., 400., 240., 180.],
            },
        );
        for id in 1..=25 {
            model.points.push(splat_eval::colmap::Point3D {
                id,
                xyz: [0., 0., 7.],
                rgb: [0; 3],
                error: 0.,
                track_len: 3,
            });
        }
        for (i, frame) in manifest.frames.iter().enumerate() {
            // Held-out depth intentionally disagrees with the training scale.
            image::ImageBuffer::from_pixel(
                4,
                3,
                image::Luma([if i == 0 { 1000u16 } else { 2000u16 }]),
            )
            .save(
                source
                    .path()
                    .join("depths")
                    .join(Path::new(&frame.name).with_extension("png")),
            )?;
            model.images.insert(
                i as u32,
                splat_eval::colmap::Image {
                    id: i as u32,
                    name: frame.name.clone(),
                    qvec: [1., 0., 0., 0.],
                    tvec: [-(i as f64) * 3.5, 0., 0.],
                    camera_id: 1,
                    num_points2d: 25,
                    obs_xy: vec![[240., 180.]; 25],
                    obs_point_id: (1..=25).collect(),
                },
            );
        }
        write_json(
            &source.path().join("depths.json"),
            &json!({"kind":"metric"}),
        )?;
        let first = source.path().join("depths/frame_000001.png");
        let source_hash = sha256_file(&first)?;
        // Simulate an existing hardlink and repeated completion: scaling must
        // replace the link, and never compound the previous scale.
        fs::create_dir(out.path().join("depths"))?;
        fs::hard_link(&first, out.path().join("depths/frame_000001.png"))?;
        for _ in 0..2 {
            let alignment = attach_depth(source.path(), out.path(), &manifest, &model)?;
            assert_eq!(alignment["units_per_meter"], 3.5);
            assert_eq!(alignment["train_depth_frames"], 2);
            assert_eq!(alignment["heldout_used"], false);
            assert_eq!(
                image::open(out.path().join("depths/frame_000001.png"))?
                    .to_luma16()
                    .get_pixel(0, 0)[0],
                7000
            );
            assert_eq!(sha256_file(&first)?, source_hash);
        }
        let metadata: Value = read_json(&out.path().join("depths.json"))?;
        assert_eq!(metadata["kind"], "metric");
        assert_eq!(metadata["aligned_to_sfm"], true);
        // An empty sensor prior should not reject an otherwise usable SfM.
        // Its scale can still be fitted from TRAIN camera centres.
        for frame in &manifest.frames {
            image::ImageBuffer::from_pixel(4, 3, image::Luma([0u16])).save(
                source
                    .path()
                    .join("depths")
                    .join(Path::new(&frame.name).with_extension("png")),
            )?;
        }
        let alignment = attach_depth(source.path(), out.path(), &manifest, &model)?;
        assert_eq!(alignment["method"], "train_camera_distance_ratio");
        assert_eq!(alignment["valid_pixels"], 0);
        assert_eq!(alignment["heldout_used"], false);
        Ok(())
    }
}
