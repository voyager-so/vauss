mod support;
use image::{DynamicImage, GenericImageView, RgbImage};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::{Backend, RunCtx};
use splat_spirula::{SpirulaBackend, SpirulaConfig, parse_progress, verify_membership};
use std::fs;
use support::{LocalRun, model, request};

#[test]
fn progress_handles_initial_metrics_unknown_eta_and_nonfinite_values() {
    let row=parse_progress("step 501/1000 (50%) splats 25752 [elapsed 00:01:03.274 | ETA 00:00:01.312] rgb_loss=0.1086 ssim=0.6702 psnr=20.65 num_added=1226").unwrap();
    assert_eq!(row["step"], 501);
    assert_eq!(row["splats"], 25752);
    assert_eq!(row["elapsed_s"], 63.274);
    let row = parse_progress(
        "step 1/200 (0%) splats 100 [elapsed 00:00:00.1 | ETA --:--] ssim=0 psnr=0 loss=1e999",
    )
    .unwrap();
    assert!(row["psnr"].is_null());
    assert!(row["ssim"].is_null());
    assert!(row["eta_s"].is_null());
    assert!(row["loss"].is_null());
    assert!(parse_progress("Eval psnr: 20").is_none());
}
#[test]
fn extraction_drops_black_and_duplicates_and_rotates() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("source");
    fs::create_dir(&source).unwrap();
    let a = RgbImage::from_fn(50, 30, |x, y| {
        image::Rgb([((x * 13 + y * 7) % 190 + 30) as u8, 80, 140])
    });
    RgbImage::new(50, 30).save(source.join("0.png")).unwrap();
    a.save(source.join("1.png")).unwrap();
    a.save(source.join("2.png")).unwrap();
    let b = RgbImage::from_fn(50, 30, |x, y| {
        image::Rgb([220, ((x * 3 + y * 9) % 150 + 70) as u8, 25])
    });
    b.save(source.join("3.png")).unwrap();
    let run = LocalRun::new(&tmp.path().join("run"), &source);
    let backend = SpirulaBackend::default();
    let r = backend
        .extract_frames(&run, request(json!({"target_count":20,"rotate_deg":90})))
        .unwrap();
    assert_eq!(r.status, Status::Succeeded, "{r:?}");
    assert_eq!(r.metrics["frame_count"], 2);
    assert_eq!(r.metrics["dropped_black"], 1);
    assert_eq!(r.metrics["dropped_duplicates"], 1);
    assert_eq!(
        image::open(run.root.join(&r.images[0]))
            .unwrap()
            .dimensions(),
        (30, 50)
    );
}
#[test]
fn one_frame_is_not_static_evidence() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("source");
    fs::create_dir(&source).unwrap();
    RgbImage::new(10, 10).save(source.join("0.png")).unwrap();
    let run = LocalRun::new(&tmp.path().join("run"), &source);
    let r = SpirulaBackend::default()
        .inspect_input(&run, InspectInput { sample_count: 1 })
        .unwrap();
    assert_eq!(r.status, Status::Succeeded, "{r:?}");
    assert!(r.metrics["static_frac"].is_null());
    assert_eq!(r.metrics["black_frac"], 1.0);
}
#[test]
fn unsupported_views_and_missing_mask_weights_are_explicit() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::with_config(SpirulaConfig {
        sam_model: "/missing/model.ggml".into(),
        ..Default::default()
    });
    assert_eq!(
        backend
            .render_views(
                &run,
                request(json!({"candidate_id":"none","views":"orbit"}))
            )
            .unwrap()
            .status,
        Status::Unsupported
    );
    let (a, _) = run.add(ArtifactKind::Frames, json!({}));
    assert_eq!(
        backend
            .mask_distractors(&run, request(json!({"frames_id":a.id})))
            .unwrap()
            .status,
        Status::Unsupported
    );
}
#[test]
fn freeze_reuses_same_split_and_blocks_changed_images_before_launch() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::default();
    let (sfm, path, split) = model(&run, &backend);
    let again = backend
        .define_split(
            &run,
            DefineSplit {
                sfm_id: sfm.id.clone(),
                interval: 2,
            },
        )
        .unwrap();
    assert_eq!(again.artifacts[0].id, split.id);
    assert_eq!(
        backend
            .define_split(
                &run,
                DefineSplit {
                    sfm_id: sfm.id.clone(),
                    interval: 3
                }
            )
            .unwrap()
            .status,
        Status::Failed
    );
    fs::write(path.join("images/000.png"), "changed").unwrap();
    let r = backend
        .train_splats(&run, request(json!({"sfm_id":sfm.id,"split_id":split.id})))
        .unwrap();
    assert_eq!(r.status, Status::Failed);
    assert!(
        r.error
            .unwrap()
            .message
            .contains("Frozen source image changed")
    );
    assert!(run.jobs.lock().unwrap().is_empty());
}
fn candidate(
    run: &LocalRun,
    sfm: &Artifact,
    path: &std::path::Path,
    split: &Artifact,
) -> (Artifact, std::path::PathBuf) {
    let (a, out) = run.add(
        ArtifactKind::Candidate,
        json!({"sfm_id":sfm.id,"split_id":split.id}),
    );
    fs::write(
        out.join("config.json"),
        r#"{"eval_mode":"interval","eval_interval":2}"#,
    )
    .unwrap();
    for (i, j) in [8, 2, 0, 6, 4].iter().enumerate() {
        for kind in ["gt", "render"] {
            fs::copy(
                path.join(format!("images/{j:03}.png")),
                out.join(format!("eval-{kind}-{i:05}.png")),
            )
            .unwrap();
        }
    }
    (a, out)
}
#[test]
fn membership_accepts_shuffle_rejects_training_view_and_extra_render() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::default();
    let (sfm, path, split) = model(&run, &backend);
    let (a, out) = candidate(&run, &sfm, &path, &split);
    fs::write(out.join("config.json"), r#"{"eval_mode":"interval","eval_interval":2,"outlier_threshold":Infinity,"lower":-Infinity,"nan":NaN,"label":"Infinity NaN \"quoted\""}"#).unwrap();
    assert_eq!(verify_membership(&run, &a).unwrap()["verified"], true);
    fs::copy(path.join("images/001.png"), out.join("eval-gt-00000.png")).unwrap();
    assert_eq!(verify_membership(&run, &a).unwrap()["verified"], false);
    fs::copy(path.join("images/008.png"), out.join("eval-gt-00000.png")).unwrap();
    fs::copy(
        out.join("eval-render-00000.png"),
        out.join("eval-render-extra.png"),
    )
    .unwrap();
    assert_eq!(verify_membership(&run, &a).unwrap()["verified"], false);
}
#[test]
fn membership_fails_closed_for_ambiguous_identical_images() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::default();
    let (sfm, path, split) = model(&run, &backend);
    let manifest = run.path(&split.id).unwrap().join("manifest.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    fs::copy(path.join("images/000.png"), path.join("images/001.png")).unwrap();
    value["image_sha256"]["001.png"] = value["image_sha256"]["000.png"].clone();
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    use sha2::{Digest, Sha256};
    let digest = format!("{:x}", Sha256::digest(fs::read(&manifest).unwrap()));
    run.artifacts
        .lock()
        .unwrap()
        .get_mut(&split.id)
        .unwrap()
        .meta
        .insert("sha256".into(), json!(digest));
    let (a, _) = candidate(&run, &sfm, &path, &split);
    assert_eq!(verify_membership(&run, &a).unwrap()["verified"], false);
}
#[test]
fn evaluation_uses_explicit_model_directory_without_env_mutation() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let models = tmp.path().join("explicit-empty-models");
    fs::create_dir(&models).unwrap();
    let backend = SpirulaBackend::with_config(SpirulaConfig {
        models_dir: Some(models.clone()),
        ..Default::default()
    });
    let (sfm, path, split) = model(&run, &backend);
    let (a, _) = candidate(&run, &sfm, &path, &split);
    let r = backend
        .evaluate(
            &run,
            Evaluate {
                candidate_id: a.id,
                split_id: split.id,
                lpips: true,
            },
        )
        .unwrap();
    assert_eq!(r.status, Status::Succeeded, "{r:?}");
    let missing = r.metrics["lpips_unavailable"].as_str().unwrap();
    assert!(missing.contains(models.to_str().unwrap()), "{missing}");
    assert!(missing.contains("explicit dir"), "{missing}");
}
#[test]
fn membership_rejects_duplicate_pairs_despite_correct_pair_count() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::default();
    let (sfm, path, split) = model(&run, &backend);
    let (a, out) = candidate(&run, &sfm, &path, &split);
    for kind in ["gt", "render"] {
        fs::copy(
            out.join(format!("eval-{kind}-00000.png")),
            out.join(format!("eval-{kind}-00004.png")),
        )
        .unwrap();
    }
    let report = verify_membership(&run, &a).unwrap();
    assert_eq!(report["verified"], false);
    assert_eq!(report["saved_pair_count"], report["expected_count"]);
    assert_eq!(report["missing_names"], json!(["004.png"]));
    assert!(
        report["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["identity_verified"] == true)
    );
    let r = backend
        .evaluate(
            &run,
            Evaluate {
                candidate_id: a.id,
                split_id: split.id,
                lpips: false,
            },
        )
        .unwrap();
    assert_eq!(r.status, Status::Failed);
    assert_eq!(r.error.unwrap().code, "split_mismatch");
}
#[test]
fn oversized_cpu_cache_fails_before_launch_and_downsampling_fits() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::with_config(SpirulaConfig {
        cpu_cache_max_bytes: 80_000,
        ..Default::default()
    });
    let (sfm, _, split) = model(&run, &backend);
    let r = backend
        .train_splats(&run, request(json!({"sfm_id":sfm.id,"split_id":split.id})))
        .unwrap();
    assert_eq!(r.status, Status::Failed);
    assert_eq!(r.error.unwrap().code, "cache_budget_exceeded");
    assert_eq!(r.metrics["estimated_cpu_cache_bytes"], 163_840);
    assert!(run.jobs.lock().unwrap().is_empty());
    let r = backend
        .train_splats(
            &run,
            request(json!({"sfm_id":sfm.id,"split_id":split.id,"resolution_divisor":2})),
        )
        .unwrap();
    assert_eq!(r.status, Status::Running, "{r:?}");
    assert_eq!(r.artifacts[0].meta["estimated_cpu_cache_bytes"], 71_680);
}
#[test]
fn masks_use_logged_names_and_completion_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::default();
    let (f, frames) = run.add(ArtifactKind::Frames, json!({}));
    fs::create_dir(frames.join("images")).unwrap();
    for name in ["003.jpg", "009.jpg"] {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(12, 10, image::Rgb([255, 255, 255])))
            .save(frames.join("images").join(name))
            .unwrap();
    }
    let (a, masks) = run.add(ArtifactKind::Masks, json!({"frames_id":f.id}));
    fs::create_dir(masks.join("raw_masks")).unwrap();
    image::GrayImage::from_pixel(12, 10, image::Luma([0]))
        .save(masks.join("raw_masks/frame_00000.png"))
        .unwrap();
    image::GrayImage::from_pixel(12, 10, image::Luma([255]))
        .save(masks.join("raw_masks/frame_00001.png"))
        .unwrap();
    let spec:JobSpec=serde_json::from_value(json!({"tool":"mask_distractors","argv":["unused"],"on_complete":"spirula:complete_mask_distractors","meta":{"artifact_id":a.id}})).unwrap();
    let jid = run.start_job(spec).unwrap();
    let job = run.root.join("jobs").join(jid);
    fs::write(
        job.join("stdout.log"),
        "frame 0: 009.jpg -- kept 0%\nframe 1: 003.jpg -- kept 100%\n",
    )
    .unwrap();
    let first = backend
        .on_job_complete(&run, "spirula:complete_mask_distractors", &job, 0)
        .unwrap();
    assert_eq!(first.status, Status::Succeeded, "{first:?}");
    assert_eq!(first.metrics["mask_count"], 2);
    assert_eq!(
        image::open(masks.join("masks/009.png"))
            .unwrap()
            .to_luma8()
            .get_pixel(0, 0)[0],
        0
    );
    let again = backend
        .on_job_complete(&run, "spirula:complete_mask_distractors", &job, 0)
        .unwrap();
    assert_eq!(
        serde_json::to_value(first).unwrap(),
        serde_json::to_value(again).unwrap()
    );
}
#[test]
fn train_flags_progress_and_interrupted_checkpoint_resume() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::with_config(SpirulaConfig {
        gpu_lock: Some("/tmp/test-gpu.lock".into()),
        ..Default::default()
    });
    let (sfm, _, split) = model(&run, &backend);
    let r = backend
        .train_splats(
            &run,
            request(json!({"sfm_id":sfm.id,"split_id":split.id,"total_steps":200})),
        )
        .unwrap();
    assert_eq!(r.status, Status::Running, "{r:?}");
    let jid = r.job_id.unwrap();
    let spec = run.jobs.lock().unwrap()[&jid].clone();
    for (flag, value) in [
        ("--save-full-checkpoint", "true"),
        ("--eval-mode", "interval"),
        ("--eval-interval", "2"),
        ("--disable-viewer", "true"),
        ("--keep-viewer-alive", "false"),
        ("--steps-per-save", "100"),
        ("--cache-images", "cpu"),
    ] {
        let i = spec.argv.iter().position(|s| s == flag).unwrap();
        assert_eq!(spec.argv[i + 1], value);
    }
    assert_eq!(spec.argv[0], "flock");
    assert!(!spec.argv.contains(&"--log-performance".into()));
    let path = run.path(&r.artifacts[0].id).unwrap();
    let checkpoint = path.join("step-100.ckpt");
    fs::create_dir(&checkpoint).unwrap();
    fs::write(
        checkpoint.join("splat.ply"),
        "ply\nelement vertex 100\nend_header\n",
    )
    .unwrap();
    fs::write(checkpoint.join("state.tar"), "fake").unwrap();
    let job = run.root.join("jobs").join(jid);
    fs::write(
        job.join("stdout.log"),
        "step 101/200 (50%) splats 100 [elapsed 00:00:01 | ETA --:--] psnr=22 ssim=0.6\nstep 201/",
    )
    .unwrap();
    assert_eq!(backend.job_progress(&run, &job)["step"], 101);
    assert_eq!(
        fs::read_to_string(path.join("training.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    let stopped = backend
        .on_job_complete(&run, &spec.on_complete, &job, 130)
        .unwrap();
    assert_eq!(stopped.status, Status::Failed);
    assert_eq!(stopped.error.unwrap().code, "interrupted");
    assert_eq!(stopped.metrics["resumable"], true);
    assert!(stopped.warnings[0].contains("99"));
    let resumed=backend.train_splats(&run,request(json!({"sfm_id":sfm.id,"split_id":split.id,"total_steps":300,"resume_candidate_id":r.artifacts[0].id}))).unwrap();
    assert_eq!(resumed.status, Status::Running, "{resumed:?}");
    let spec = run.jobs.lock().unwrap()[resumed.job_id.as_ref().unwrap()].clone();
    let index = spec.argv.iter().position(|s| s == "--resume").unwrap();
    assert_eq!(spec.argv[index + 1], checkpoint.to_str().unwrap());
    let index = spec
        .argv
        .iter()
        .position(|s| s == "--cache-images")
        .unwrap();
    assert_eq!(spec.argv[index + 1], "cpu");
}
#[test]
fn portable_config_does_not_require_flock() {
    let tmp = tempfile::tempdir().unwrap();
    let run = LocalRun::new(tmp.path(), tmp.path());
    let backend = SpirulaBackend::with_config(SpirulaConfig {
        binary: "spirula.exe".into(),
        gpu_lock: None,
        ..Default::default()
    });
    let (sfm, _, split) = model(&run, &backend);
    let r = backend
        .train_splats(&run, request(json!({"sfm_id":sfm.id,"split_id":split.id})))
        .unwrap();
    let jobs = run.jobs.lock().unwrap();
    assert_eq!(jobs[r.job_id.as_ref().unwrap()].argv[0], "spirula.exe");
}
