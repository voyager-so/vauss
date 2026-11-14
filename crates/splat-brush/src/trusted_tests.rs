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

/// Fake COLMAP: the extractor installs a database listing every authored frame, matching is a
/// no-op, and `point_triangulator` echoes its input cameras/images (poses untouched) with `points`
/// synthetic points. The images it was shown are recorded for the split assertion.
#[cfg(unix)]
fn fake_colmap(dir: &Path, points: usize) -> Result<std::path::PathBuf> {
    fake_colmap_with(dir, points, "")
}

/// `tri_extra` is shell run after the fake triangulator echoes its input (e.g. to sabotage it).
#[cfg(unix)]
fn fake_colmap_with(dir: &Path, points: usize, tri_extra: &str) -> Result<std::path::PathBuf> {
    // The fixture's held-out frames never reach COLMAP.
    let train: Vec<usize> = (0..25).filter(|i| ![7, 15, 23].contains(i)).collect();
    fake_colmap_for(dir, points, tri_extra, &train)
}

/// Same, for a database holding exactly the frames with these indices.
#[cfg(unix)]
fn fake_colmap_for(dir: &Path, points: usize, tri_extra: &str, frames: &[usize]) -> Result<std::path::PathBuf> {
    use std::{io::Write, os::unix::fs::PermissionsExt};
    let template = dir.join("template.db");
    let db = rusqlite::Connection::open(&template)?;
    // The shape COLMAP 4 writes: one single-camera rig and frame per image, a guessed camera.
    db.execute_batch("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
        CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER);
        CREATE TABLE rigs (rig_id INTEGER PRIMARY KEY, ref_sensor_id INTEGER, ref_sensor_type INTEGER);
        CREATE TABLE rig_sensors (rig_id INTEGER, sensor_id INTEGER, sensor_type INTEGER, sensor_from_rig BLOB);
        CREATE TABLE frames (frame_id INTEGER PRIMARY KEY, rig_id INTEGER);
        CREATE TABLE frame_data (frame_id INTEGER, data_id INTEGER, sensor_id INTEGER, sensor_type INTEGER);")?;
    for &i in frames {
        let i = i as i64;
        // Deliberately non-sequential, non-aligned ids.
        let (image, camera, rig, frame) = (i + 1, i + 11, i + 21, i + 31);
        db.execute("INSERT INTO images VALUES (?1, ?2, ?3)", rusqlite::params![image, format!("frame_{i:06}.jpg"), camera])?;
        db.execute("INSERT INTO cameras VALUES (?1, 2, 1024, 576, X'00', 0)", [camera])?;
        db.execute("INSERT INTO rigs VALUES (?1, ?2, 0)", [rig, camera])?;
        db.execute("INSERT INTO frames VALUES (?1, ?2)", [frame, rig])?;
        db.execute("INSERT INTO frame_data VALUES (?1, ?2, ?3, 0)", [frame, image, camera])?;
    }
    drop(db);
    let (mut file, staged) = tempfile::NamedTempFile::new_in(dir)?.into_parts();
    file.write_all(format!(r##"#!/bin/sh
if [ "$1" = '-h' ]; then echo 'COLMAP 4.2.1 (test)'; exit 0; fi
cmd="$1"; shift
while [ "$#" -gt 1 ]; do
  case "$1" in
    --output_path) out="$2";; --input_path) input="$2";;
    --image_path) images="$2";; --database_path) database="$2";;
  esac
  shift 2
done
if [ "$cmd" = 'feature_extractor' ]; then cp '{template}' "$database"; fi
if [ "$cmd" = 'point_triangulator' ]; then
  ls "$images" > '{seen}'
  # COLMAP 4 aborts unless the input rigs/frames/cameras match the database's ids.
  [ -s "$input/rigs.txt" ] && [ -s "$input/frames.txt" ] || {{ echo 'missing rigs/frames' >&2; exit 3; }}
  rm -rf '{input_seen}'; cp -r "$input" '{input_seen}'
  cp "$input/cameras.txt" "$input/images.txt" "$out/"
  {tri_extra}
  awk 'BEGIN {{ for (i = 1; i <= {points}; i++) printf "%d %f %f %f 10 20 30 0.5 1 0\n", i, i % 7, i % 11, 3 + i % 5 }}' > "$out/points3D.txt"
fi
"##, template = template.display(), seen = dir.join("seen.txt").display(), input_seen = dir.join("input_seen").display(), points = points, tri_extra = tri_extra).as_bytes())?;
    drop(file);
    fs::set_permissions(&staged, fs::Permissions::from_mode(0o755))?;
    let script = dir.join("fake colmap");
    fs::rename(&staged, &script)?;
    Ok(script)
}

/// Extract + trusted-import the lane C fixture; returns (run, sfm artifact id, result).
fn trusted_import(root: &Path, cfg: &BrushConfig) -> Result<(Run, ToolResult)> {
    let meta = metadata();
    source(&root.join("source"), &meta)?;
    let run = Run::create(root.join("run"), root.join("source"), "test", "brush")?
        .with_worker_exe(root.join("missing-worker"));
    let extracted = media::extract_frames(cfg, &run, serde_json::from_value(json!({"trust_poses":true,"max_edge":1600}))?)?;
    let imported = sfm::reconstruct_sfm(cfg, &run, serde_json::from_value(json!({"frames_id":extracted.artifacts[0].id}))?)?;
    assert_eq!(imported.status, Status::Succeeded);
    assert_eq!(imported.metrics["camera_source"], "imported");
    Ok((run, imported))
}

#[cfg(unix)]
#[test]
fn trusted_import_without_points_is_seeded_by_fixed_pose_triangulation() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let mut cfg = unavailable_helpers();
    cfg.colmap = Some(fake_colmap(tmp.path(), 1500)?);
    let (run, imported) = trusted_import(tmp.path(), &cfg)?;
    let dataset = run.path(&imported.artifacts[0].id)?;
    // Honest provenance and counts, no warning.
    assert_eq!(imported.metrics["seed_points"]["source"], "colmap_triangulated_fixed_poses");
    assert_eq!(imported.metrics["seed_points"]["points"], 1500);
    assert_eq!(imported.metrics["n_points3d"], 1500);
    assert_eq!(imported.metrics["seed_points"]["n_train_images"], 22);
    assert!(imported.warnings.is_empty(), "{:?}", imported.warnings);
    assert_eq!(read_json::<Value>(&dataset.join("health.json"))?["seed_points"]["points"], 1500);
    // Held-out views never reach feature extraction/triangulation.
    let seen = fs::read_to_string(tmp.path().join("seen.txt"))?;
    let seen: Vec<_> = seen.lines().collect();
    assert_eq!(seen.len(), 22);
    for held in ["frame_000007.jpg", "frame_000015.jpg", "frame_000023.jpg"] {
        assert!(!seen.contains(&held));
    }
    // Camera authority: the written model still holds all 25 authored poses, now with points.
    let model = splat_eval::read_model(&dataset.join("sparse/0"))?;
    assert_eq!(model.images.len(), 25);
    assert_eq!(model.points.len(), 1500);
    // The input model's ids/models match the database's (COLMAP 4 aborts otherwise).
    let seen = tmp.path().join("input_seen");
    let lines = |f: &str| -> Vec<Vec<String>> { fs::read_to_string(seen.join(f)).unwrap().lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty()).map(|l| l.split_whitespace().map(String::from).collect()).collect() };
    let (cams, rigs, frames) = (lines("cameras.txt"), lines("rigs.txt"), lines("frames.txt"));
    assert_eq!((cams.len(), rigs.len(), frames.len()), (22, 22, 22));
    for c in &cams { assert_eq!(c[1..4], ["OPENCV", "1024", "576"]); assert!((11..36).contains(&c[0].parse::<i64>()?)); }
    for r in &rigs { assert_eq!(r[1..3], ["1", "CAMERA"]); assert_eq!(r[3].parse::<i64>()? - r[0].parse::<i64>()?, -10); }
    for f in &frames {
        let n = |k: usize| f[k].parse::<i64>().unwrap();
        assert_eq!((n(0) - n(1), n(11) - n(12), n(1) - n(11)), (10, 10, 10));
        assert_eq!(f[9..11], ["1", "CAMERA"]);
    }
    // No scratch COLMAP model may live in the dataset root: the loader would prefer it to sparse/0.
    fn models(dir: &Path, found: &mut Vec<std::path::PathBuf>) {
        for e in fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() { models(&p, found) } else if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("cameras.")) { found.push(p) }
        }
    }
    let mut found = vec![];
    models(&dataset, &mut found);
    assert_eq!(found, [dataset.join("sparse/0/cameras.bin")], "scratch models leaked into the dataset");
    // The training loader gets seed points, so the +Z random initializer is not used.
    let split = split::define_split(&run, serde_json::from_value(json!({"sfm_id":imported.artifacts[0].id}))?)?;
    assert_eq!(splat_core::split::load(&run, &split.artifacts[0].id)?.heldout_names.len(), 3);
    let loaded = engine::runtime()?.block_on(engine::load_scene(&dataset, 1600))?;
    assert!(loaded.init.is_some(), "seeded trusted import must provide init points");
    Ok(())
}

#[cfg(unix)]
#[test]
fn trusted_import_with_too_few_triangulated_points_warns_and_reports_the_fallback() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let mut cfg = unavailable_helpers();
    cfg.colmap = Some(fake_colmap(tmp.path(), 10)?);
    let (run, imported) = trusted_import(tmp.path(), &cfg)?;
    let seed = &imported.metrics["seed_points"];
    assert_eq!(seed["source"], "none_plus_z_fallback");
    assert_eq!(seed["points"], 0);
    assert_eq!(seed["triangulated_points"], 10);
    assert!(imported.warnings.iter().any(|w| w.contains("lost >10 dB")), "{:?}", imported.warnings);
    let dataset = run.path(&imported.artifacts[0].id)?;
    assert_eq!(splat_eval::read_model(&dataset.join("sparse/0"))?.points.len(), 0);
    assert_eq!(read_json::<Value>(&dataset.join("health.json"))?["seed_points"]["source"], "none_plus_z_fallback");
    Ok(())
}

#[test]
fn trusted_import_without_colmap_warns_loudly() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let (run, imported) = trusted_import(tmp.path(), &unavailable_helpers())?;
    assert_eq!(imported.metrics["seed_points"]["source"], "none_plus_z_fallback");
    assert!(imported.warnings.iter().any(|w| w.contains("lost >10 dB") && w.contains("COLMAP not found")));
    let dataset = run.path(&imported.artifacts[0].id)?;
    let loaded = engine::runtime()?.block_on(engine::load_scene(&dataset, 1600))?;
    assert!(loaded.init.is_none());
    Ok(())
}

#[cfg(unix)]
#[test]
fn trusted_import_with_empty_heldout_still_loads_seed_points() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let mut meta = metadata();
    meta["frames"] = Value::Array(meta["frames"].as_array().unwrap()[..3].to_vec());
    meta["test_filenames"] = json!([]);
    source(&tmp.path().join("source"), &meta)?;
    let mut cfg = unavailable_helpers();
    cfg.colmap = Some(fake_colmap_for(tmp.path(), 1500, "", &[0, 1, 2])?);
    let run = Run::create(tmp.path().join("run"), tmp.path().join("source"), "test", "brush")?;
    let extracted = media::extract_frames(&cfg, &run, serde_json::from_value(json!({"trust_poses":true,"max_edge":1600}))?)?;
    let imported = sfm::reconstruct_sfm(&cfg, &run, serde_json::from_value(json!({"frames_id":extracted.artifacts[0].id}))?)?;
    assert_eq!(imported.metrics["seed_points"]["source"], "colmap_triangulated_fixed_poses");
    assert_eq!(imported.metrics["seed_points"]["n_train_images"], 3);
    let dataset = run.path(&imported.artifacts[0].id)?;
    let loaded = engine::runtime()?.block_on(engine::load_scene(&dataset, 1600))?;
    assert!(loaded.init.is_some(), "seed points must reach the loader");
    Ok(())
}

#[cfg(unix)]
#[test]
fn trusted_seeding_rejects_changed_intrinsics_and_kills_hangs() -> Result<()> {
    // COLMAP echoing a different image size is rejected, with the fallback warning.
    let tmp = tempfile::tempdir()?;
    let mut cfg = unavailable_helpers();
    cfg.colmap = Some(fake_colmap_with(tmp.path(), 1500, r#"awk '/^#/ {print; next} {$3=$3+1} 1' "$out/cameras.txt" > "$out/c.tmp" && mv "$out/c.tmp" "$out/cameras.txt""#)?);
    let (_run, imported) = trusted_import(tmp.path(), &cfg)?;
    assert_eq!(imported.metrics["seed_points"]["source"], "none_plus_z_fallback");
    assert!(imported.metrics["seed_points"]["reason"].as_str().unwrap().contains("intrinsics"));
    assert!(imported.warnings.iter().any(|w| w.contains("lost >10 dB")));

    // A hung triangulator is killed at the deadline instead of blocking the import.
    let tmp = tempfile::tempdir()?;
    let (run, mut imported) = trusted_import(tmp.path(), &unavailable_helpers())?;
    let hang = fake_colmap_with(tmp.path(), 1500, "sleep 7191907 & wait")?;
    let dataset = run.path(&imported.artifacts[0].id)?;
    let started = std::time::Instant::now();
    capture_sfm::seed_if_unseeded(&dataset, Some(&hang), std::time::Duration::from_secs(2), &mut imported)?;
    assert!(started.elapsed().as_secs() < 30);
    assert_eq!(imported.metrics["seed_points"]["source"], "none_plus_z_fallback");
    assert!(imported.metrics["seed_points"]["reason"].as_str().unwrap().contains("timed out"));
    // The shell's descendant must die with it. Scoped to this test's unique marker only.
    let alive = |marker: &str| fs::read_dir("/proc").unwrap().flatten().any(|e| {
        fs::read(e.path().join("cmdline")).is_ok_and(|c| String::from_utf8_lossy(&c).contains(marker))
    });
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(!alive("sleep\x007191907"), "timeout left a descendant running");

    // Non-finite poses from COLMAP are rejected (f64::max would hide a NaN).
    let tmp = tempfile::tempdir()?;
    let mut cfg = unavailable_helpers();
    cfg.colmap = Some(fake_colmap_with(tmp.path(), 1500, r#"awk '/^#/ || NF == 0 {print; next} {$6 = "nan"} 1' "$out/images.txt" > "$out/i.tmp" && mv "$out/i.tmp" "$out/images.txt""#)?);
    let (_run, imported) = trusted_import(tmp.path(), &cfg)?;
    assert_eq!(imported.metrics["seed_points"]["source"], "none_plus_z_fallback");
    assert!(imported.metrics["seed_points"]["reason"].as_str().unwrap().contains("not finite"), "{}", imported.metrics["seed_points"]);
    Ok(())
}

/// Real-COLMAP validation (the fake-COLMAP tests cannot see COLMAP's own model/rig/frame checks).
/// Skipped, with the reason printed, unless SPLAT_REAL_COLMAP (a COLMAP 4.x binary) and
/// SPLAT_REAL_CAPTURE (a nerfstudio capture dir with transforms.json + images/) are set.
/// Run: SPLAT_REAL_COLMAP=... SPLAT_REAL_CAPTURE=... cargo test -p splat-brush real_colmap -- --nocapture
#[test]
fn real_colmap_seeds_a_trusted_capture() -> Result<()> {
    let (Some(colmap), Some(capture)) = (std::env::var_os("SPLAT_REAL_COLMAP"), std::env::var_os("SPLAT_REAL_CAPTURE")) else {
        eprintln!("SKIPPED real_colmap_seeds_a_trusted_capture: set SPLAT_REAL_COLMAP and SPLAT_REAL_CAPTURE to run against real COLMAP");
        return Ok(());
    };
    let tmp = tempfile::tempdir()?;
    let mut cfg = unavailable_helpers();
    cfg.colmap = Some(colmap.into());
    let run = Run::create(tmp.path().join("run"), &capture, "test", "brush")?;
    let extracted = media::extract_frames(&cfg, &run, serde_json::from_value(json!({"trust_poses":true,"max_edge":1600}))?)?;
    let imported = sfm::reconstruct_sfm(&cfg, &run, serde_json::from_value(json!({"frames_id":extracted.artifacts[0].id}))?)?;
    eprintln!("seed_points {} warnings {:?}", imported.metrics["seed_points"], imported.warnings);
    assert_eq!(imported.metrics["seed_points"]["source"], "colmap_triangulated_fixed_poses", "{:?}", imported.metrics["seed_points"]);
    assert!(imported.metrics["seed_points"]["points"].as_u64().unwrap() >= 1000);
    let dataset = run.path(&imported.artifacts[0].id)?;
    let loaded = engine::runtime()?.block_on(engine::load_scene(&dataset, 1600))?;
    assert!(loaded.init.is_some());
    Ok(())
}
