//! CPU-only contract regression using the retained lane C authored calibration.
use crate::{BrushConfig, capture_sfm, engine, media, sfm, split, util::*};
use anyhow::Result;
use glam::{DMat4, DVec3};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::{Run, RunCtx};
use std::{fs, path::Path};

fn metadata() -> Value {
    serde_json::from_str(include_str!("../fixtures/trusted-lane-c/transforms.json")).unwrap()
}
fn source(root: &Path, metadata: &Value) -> Result<()> {
    fs::create_dir_all(root.join("images"))?;
    for f in metadata["frames"].as_array().unwrap() {
        image::RgbImage::from_pixel(
            f["w"].as_u64().unwrap() as u32,
            f["h"].as_u64().unwrap() as u32,
            image::Rgb([90, 120, 150]),
        )
        .save(root.join(f["file_path"].as_str().unwrap()))?;
    }
    write_json(&root.join("transforms.json"), metadata)
}
fn unavailable_helpers() -> BrushConfig {
    let missing = std::path::PathBuf::from("/unavailable-trusted-import-helper");
    BrushConfig {
        exe: missing.clone(),
        ffmpeg: missing.clone(),
        ffprobe: missing,
        colmap: None,
        models_dir: None,
        gpu_lock: None,
        vocab_tree: None,
        vocab_tree_fetch: Some(crate::VocabTreeFetch(std::sync::Arc::new(|| {
            panic!("SfM helper lookup in trusted mode")
        }))),
        sam: None,
        da3_weights: None,
        aliked_dir: None,
    }
}
fn authored(f: &Value) -> DMat4 {
    let a: Vec<f64> = f["transform_matrix"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect();
    let mut m = DMat4::from_cols_slice(&a).transpose();
    m.y_axis *= -1.0;
    m.z_axis *= -1.0;
    m
}
fn project(pose: DMat4, p: DVec3, k: [f64; 4]) -> glam::DVec2 {
    let p = pose.inverse().transform_point3(p);
    glam::dvec2(k[0] * p.x / p.z + k[2], k[1] * p.y / p.z + k[3])
}

#[test]
fn trusted_lane_c_source_worker_and_export_preserve_authored_cameras() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let meta = metadata();
    source(&tmp.path().join("source"), &meta)?;
    let run = Run::create(
        tmp.path().join("run"),
        tmp.path().join("source"),
        "test",
        "brush",
    )?
    .with_worker_exe(tmp.path().join("missing-worker"));
    let cfg = unavailable_helpers();
    let extracted = media::extract_frames(
        &cfg,
        &run,
        serde_json::from_value(json!({
            "trust_poses":true,"max_edge":1600
        }))?,
    )?;
    let frames = &extracted.artifacts[0];
    assert_eq!(extracted.metrics["frame_count"], 25);
    assert_eq!(frames.meta["preserve_all"], true);
    let imported = sfm::reconstruct_sfm(
        &cfg,
        &run,
        serde_json::from_value(json!({"frames_id":frames.id}))?,
    )?;
    assert_eq!(imported.status, Status::Succeeded);
    assert!(imported.job_id.is_none()); // No worker, helper, SfM retry or fallback.
    assert_eq!(imported.metrics["camera_source"], "imported");
    assert_eq!(imported.metrics["sfm_skipped"], true);
    assert_eq!(imported.metrics["reconstruction_quality"], "unmeasured");
    let sfm = &imported.artifacts[0];
    let dataset = run.path(&sfm.id)?;
    assert!(!dataset.join("sfm_job.json").exists());
    let split = split::define_split(&run, serde_json::from_value(json!({"sfm_id":sfm.id}))?)?;
    let frozen = splat_core::split::load(&run, &split.artifacts[0].id)?;
    assert_eq!(
        frozen.heldout_names,
        ["frame_000007.jpg", "frame_000015.jpg", "frame_000023.jpg"]
    );
    assert_eq!(frozen.train_names.len(), 22);
    assert!(
        frozen
            .train_names
            .iter()
            .all(|n| !frozen.heldout_names.contains(n))
    );
    // Build the actual training job. Its intentionally absent worker cannot launch a GPU.
    let staged = crate::train::train_splats(
        &cfg,
        &run,
        serde_json::from_value(json!({"sfm_id":sfm.id,"split_id":split.artifacts[0].id}))?,
    )?;
    let completion: Value = read_json(
        &run.root()
            .join("jobs")
            .join(staged.job_id.unwrap())
            .join("completion.json"),
    )?;
    assert_eq!(completion["error_code"], "worker_spawn_failed");
    let candidate = run
        .state()?
        .artifacts
        .into_values()
        .find(|a| a.kind == ArtifactKind::Candidate)
        .unwrap();
    let job: crate::TrainJob = read_json(&run.path(&candidate.id)?.join("train_job.json"))?;
    assert_eq!(job.train_names, frozen.train_names);
    assert_eq!(job.heldout_names, frozen.heldout_names);
    assert_eq!(job.config_overrides.as_ref().unwrap()["pose-refine"], false);
    let loaded =
        engine::runtime()?.block_on(engine::load_scene(&job.dataset, job.max_resolution))?;
    assert_eq!(loaded.views.len(), 25);
    let first = authored(&meta["frames"][0]);
    let landmarks = [
        DVec3::new(-0.4, -0.2, 3.),
        DVec3::new(0.5, -0.1, 3.7),
        DVec3::new(-0.1, 0.6, 4.2),
        DVec3::new(0.3, 0.2, 5.),
    ]
    .map(|p| first.transform_point3(p));
    assert!(
        (landmarks[1] - landmarks[0])
            .cross(landmarks[2] - landmarks[0])
            .length()
            > 0.1
    );
    let ply = tmp.path().join("splat.ply");
    let g = splat_webexport::ply::Gaussians {
        pos: landmarks.map(|p| p.as_vec3().to_array()).to_vec(),
        scale: vec![[-5.; 3]; 4],
        rot: vec![[1., 0., 0., 0.]; 4],
        opacity: vec![5.; 4],
        dc: vec![[0.; 3]; 4],
        ..Default::default()
    };
    splat_webexport::ply::write_ply(&g, &ply)?;
    let exported = splat_webexport::export_web_with_options(
        &ply,
        Some(&dataset),
        &tmp.path().join("web"),
        &splat_webexport::SceneOptions {
            lod: splat_webexport::LodMode::Skip,
            ..Default::default()
        },
    )?;
    let transform = DMat4::from_cols_array(&serde_json::from_value::<[f64; 16]>(
        exported["transform_from_source"].clone(),
    )?);
    let cameras = exported["capture_cameras"].as_array().unwrap();
    assert_eq!(cameras.len(), 25);
    let (mut max_worker_pose, mut max_export_pose, mut max_worker_k, mut max_projection) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (i, f) in meta["frames"].as_array().unwrap().iter().enumerate() {
        let expected = authored(f);
        let name = format!("frame_{i:06}.jpg");
        let camera = &loaded
            .views
            .iter()
            .find(|v| engine::view_name(v) == name)
            .unwrap()
            .camera;
        let worker = glam::Mat4::from(camera.local_to_world()).as_dmat4();
        for (a, b) in worker
            .to_cols_array()
            .into_iter()
            .zip(expected.to_cols_array())
        {
            max_worker_pose = max_worker_pose.max((a - b).abs());
        }
        assert!(worker.abs_diff_eq(expected, 3e-6), "worker pose {name}");
        let k = [
            f["fl_x"].as_f64().unwrap(),
            f["fl_y"].as_f64().unwrap(),
            f["cx"].as_f64().unwrap(),
            f["cy"].as_f64().unwrap(),
        ];
        assert_ne!(k[0], k[1]);
        let dims = glam::uvec2(
            f["w"].as_u64().unwrap() as u32,
            f["h"].as_u64().unwrap() as u32,
        );
        let focal = camera.focal(dims);
        let center = camera.center(dims);
        for (a, b) in [focal.x, focal.y, center.x, center.y].into_iter().zip(k) {
            max_worker_k = max_worker_k.max((a as f64 - b).abs());
            assert!((a as f64 - b).abs() < 2e-4);
        }
        let c = cameras.iter().find(|c| c["image"] == name).unwrap();
        let source_pose = DMat4::from_cols_array(&serde_json::from_value::<[f64; 16]>(
            c["source_c2w"].clone(),
        )?);
        let export_pose =
            DMat4::from_cols_array(&serde_json::from_value::<[f64; 16]>(c["c2w"].clone())?);
        assert!(
            source_pose.abs_diff_eq(expected, 2e-6),
            "export source pose {name}"
        );
        let mut expected_export = transform * expected;
        let scale = transform.x_axis.truncate().length();
        expected_export.x_axis /= scale;
        expected_export.y_axis /= scale;
        expected_export.z_axis /= scale;
        for (a, b) in export_pose
            .to_cols_array()
            .into_iter()
            .zip(expected_export.to_cols_array())
        {
            max_export_pose = max_export_pose.max((a - b).abs());
        }
        assert!(
            export_pose.abs_diff_eq(expected_export, 2e-6),
            "export pose {name}"
        );
        for (key, expected) in ["fx", "fy", "cx", "cy"].into_iter().zip(k) {
            assert!((c[key].as_f64().unwrap() - expected).abs() < 1e-9);
        }
        for p in landmarks {
            let residual = project(expected, p, k).distance(project(
                export_pose,
                transform.transform_point3(p),
                k,
            ));
            max_projection = max_projection.max(residual);
            assert!(
                residual < 0.002,
                "authored landmark projection {name}: {residual}"
            );
        }
    }
    eprintln!(
        "trusted_lane_c_regression {}",
        json!({"cameras":25,"train":22,"heldout":3,
        "sfm_skipped":true,"max_worker_pose_element_error":max_worker_pose,
        "max_export_pose_element_error":max_export_pose,"max_worker_k_error_px":max_worker_k,
        "max_landmark_projection_error_px":max_projection})
    );

    // Apply jobs from the real producer through the exact pre-GPU worker config path.
    // Merely inspecting TrainJob JSON missed a serde-name mismatch in these overrides.
    for preview in [false, true] {
        let staged = crate::train::train_splats(
            &cfg, &run,
            serde_json::from_value(json!({"sfm_id":sfm.id,
                "split_id":split.artifacts[0].id,"preview":preview}))?,
        )?;
        let produced: crate::TrainJob = read_json(
            &run.path(&staged.artifacts[0].id)?.join("train_job.json"))?;
        let schema = serde_json::to_value(brush_train::config::TrainConfig::default())?;
        let over = produced.config_overrides.as_ref().unwrap();
        assert_eq!(over["pose-refine"], false);
        assert_eq!(over["pose-refine-focal"], false);
        assert_eq!(over["pose-align-eval-steps"], 0);
        assert!(over.keys().all(|key| schema.get(key).is_some()));
        for env in [None, Some(false), Some(true)] {
            let config = crate::train::training_config(&produced, env)?;
            assert!(!config.pose_refine && !config.pose_refine_focal);
            assert_eq!(config.pose_align_eval_steps, 0);
            assert!(!config.camera_motion_blur);
            assert_eq!(config.total_train_iters, produced.total_steps);
        }
        // Trusted authority also wins over contradictory, otherwise-valid serialized overrides.
        let mut conflicting = produced.clone();
        conflicting.config_overrides = serde_json::from_value(json!({
            "pose-refine":true,"pose-refine-focal":true,
            "pose-align-eval-steps":100,"camera-motion-blur":true}))?;
        let config = crate::train::training_config(&conflicting, Some(true))?;
        assert!(!config.pose_refine && !config.pose_refine_focal && !config.camera_motion_blur);
        assert_eq!(config.pose_align_eval_steps, 0);
    }
    // Bad fields and types must fail before the worker can open a GPU lock or device.
    for overrides in [json!({"pose_refine":false}), json!({"unknown-field":1}),
        json!({"pose-refine":"not-a-boolean"})] {
        let mut invalid = job.clone();
        invalid.config_overrides = serde_json::from_value(overrides)?;
        invalid.gpu_lock = Some(tmp.path().join("must-not-open.lock"));
        let error = crate::train::run_job(invalid).unwrap_err().to_string();
        assert!(error.contains("unknown TrainConfig field") || error.contains("invalid type"), "{error}");
        assert!(!tmp.path().join("must-not-open.lock").exists());
    }
    // Legacy capture fallback and preview use the same canonical merge path, without
    // mutating process-global environment variables in parallel CPU tests.
    let fallback = tmp.path().join("fallback");
    fs::create_dir(&fallback)?;
    write_json(&fallback.join("capture_selection.json"), &json!({"camera_source":"capture_fallback"}))?;
    for preview in [false, true] {
        for env in [None, Some(false), Some(true)] {
            let mut fallback_job = job.clone();
            fallback_job.dataset = fallback.clone();
            fallback_job.preview = preview;
            fallback_job.config_overrides = capture_sfm::training_overrides_for(&fallback, preview, env)?;
            let config = crate::train::training_config(&fallback_job, env)?;
            let refine = !preview && env != Some(false);
            assert_eq!(config.pose_refine, refine);
            assert_eq!(config.pose_align_eval_steps, if refine { crate::train::POSE_ALIGN_DEFAULT_STEPS } else { 0 });
        }
    }
    // A high-coverage solve that changes K must fail both completion and the pre-GPU worker gate.
    let mut model = splat_eval::read_model(&dataset.join("sparse/0"))?;
    for c in model.cameras.values_mut() {
        c.params[1] *= 6.0;
    }
    splat_eval::colmap::write_cameras_bin(&dataset.join("sparse/0/cameras.bin"), &model.cameras)?;
    let error = crate::train::run_job(job).unwrap_err();
    assert!(
        error.to_string().contains("calibration health failed"),
        "{error:#}"
    );
    assert!(error.to_string().contains("frame_000000.jpg"));
    write_json(
        &dataset.join("reconstruction.json"),
        &json!({"health":{
        "n_input":25,"n_registered":25,"camera_extent":1.0,"points_median_dist_from_cameras":1.0,
        "cameras":{},"gates":{}},"component_sizes":[25]}),
    )?;
    let rejected = sfm::complete(&run, &sfm.id, 0, "")?;
    assert_eq!(rejected.status, Status::Failed);
    assert_eq!(rejected.error.unwrap().code, "calibration_health_failed");
    assert_eq!(rejected.metrics["calibration_health"]["pass"], false);
    Ok(())
}

#[test]
fn trusted_import_health_split_resize_and_default_policy() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let mut meta = metadata();
    meta["frames"] = Value::Array(meta["frames"].as_array().unwrap()[..3].to_vec());
    meta["test_filenames"] = json!([]);
    let input = tmp.path().join("source");
    source(&input, &meta)?;
    let cfg = unavailable_helpers();
    let run = Run::create(tmp.path().join("run"), &input, "test", "brush")?;
    let extracted = media::extract_frames(
        &cfg,
        &run,
        serde_json::from_value(json!({"trust_poses":true,"max_edge":512}))?,
    )?;
    let frames = &extracted.artifacts[0];
    let manifest: Value = read_json(&run.path(&frames.id)?.join("capture.json"))?;
    assert_eq!(manifest["frames"][0]["width"], 512);
    assert_eq!(manifest["frames"][0]["height"], 288);
    for (i, key) in ["fl_x", "fl_y", "cx", "cy"].iter().enumerate() {
        assert_eq!(
            manifest["frames"][0]["params"][i].as_f64().unwrap(),
            meta["frames"][0][key].as_f64().unwrap() / 2.
        );
    }
    let imported = sfm::reconstruct_sfm(
        &cfg,
        &run,
        serde_json::from_value(json!({"frames_id":frames.id}))?,
    )?;
    let split = split::define_split(
        &run,
        serde_json::from_value(json!({"sfm_id":imported.artifacts[0].id}))?,
    )?;
    let frozen = splat_core::split::load(&run, &split.artifacts[0].id)?;
    assert!(frozen.heldout_names.is_empty());
    assert_eq!(frozen.train_names.len(), 3);
    // Omitted test split also stays empty; default extraction still asks for SfM.
    meta.as_object_mut().unwrap().remove("test_filenames");
    write_json(&input.join("transforms.json"), &meta)?;
    let ordinary = media::extract_frames(
        &cfg,
        &run,
        serde_json::from_value(json!({"preserve_all":true}))?,
    )?;
    let failure = sfm::reconstruct_sfm(
        &cfg,
        &run,
        serde_json::from_value(json!({"frames_id":ordinary.artifacts[0].id}))?,
    )
    .unwrap_err();
    assert!(failure.to_string().contains("COLMAP not found"));
    let trusted = media::extract_frames(
        &cfg,
        &run,
        serde_json::from_value(json!({"trust_poses":true}))?,
    )?;
    let m: Value = read_json(&run.path(&trusted.artifacts[0].id)?.join("capture.json"))?;
    assert_eq!(m["heldout_names"], json!([]));
    let original = meta.clone();
    meta["frames"][0]["cx"] = json!(-50.25);
    meta["frames"][0]["cy"] = json!(700.75);
    write_json(&input.join("transforms.json"), &meta)?;
    let off_axis = media::extract_frames(
        &cfg,
        &run,
        serde_json::from_value(json!({"trust_poses":true}))?,
    )?;
    let m: Value = read_json(&run.path(&off_axis.artifacts[0].id)?.join("capture.json"))?;
    assert_eq!(m["frames"][0]["params"][2], -50.25);
    assert_eq!(m["frames"][0]["params"][3], 700.75);
    let mutations: Vec<(&str, Box<dyn Fn(&mut Value)>)> = vec![
        (
            "rotation",
            Box::new(|m| m["frames"][0]["transform_matrix"][0][0] = json!(2.)),
        ),
        (
            "homogeneous",
            Box::new(|m| m["frames"][0]["transform_matrix"][3][0] = json!(0.001)),
        ),
        (
            "intrinsics",
            Box::new(|m| m["frames"][0]["fl_x"] = json!(-1.)),
        ),
        ("dimensions", Box::new(|m| m["frames"][0]["w"] = json!(512))),
        (
            "calibration field cx",
            Box::new(|m| m["frames"][0]["cx"] = json!("invalid")),
        ),
        (
            "baseline",
            Box::new(|m| {
                let pose = m["frames"][0]["transform_matrix"].clone();
                for f in m["frames"].as_array_mut().unwrap() {
                    f["transform_matrix"] = pose.clone();
                }
            }),
        ),
        (
            "duplicate",
            Box::new(|m| m["frames"][1]["file_path"] = m["frames"][0]["file_path"].clone()),
        ),
        (
            "array",
            Box::new(|m| m["test_filenames"] = json!("images/00000.png")),
        ),
        (
            "duplicate",
            Box::new(|m| m["test_filenames"] = json!(["images/00000.png", "images/00000.png"])),
        ),
        (
            "disjoint",
            Box::new(|m| {
                m["train_filenames"] = json!(["images/00000.png"]);
                m["test_filenames"] = json!(["images/00000.png"]);
            }),
        ),
        ("empty", Box::new(|m| m["train_filenames"] = json!([]))),
        (
            "empty TRAIN",
            Box::new(|m| {
                m["test_filenames"] =
                    json!(["images/00000.png", "images/00001.png", "images/00002.png"])
            }),
        ),
    ];
    for (message, mutate) in mutations {
        let mut meta = original.clone();
        mutate(&mut meta);
        write_json(&input.join("transforms.json"), &meta)?;
        let error = media::extract_frames(
            &cfg,
            &run,
            serde_json::from_value(json!({"trust_poses":true}))?,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains(message),
            "expected {message}: {error:#}"
        );
    }
    fs::remove_file(input.join("transforms.json"))?;
    let error = media::extract_frames(
        &cfg,
        &run,
        serde_json::from_value(json!({"trust_poses":true}))?,
    )
    .unwrap_err();
    assert!(error.to_string().contains("supported calibrated capture"));
    Ok(())
}
