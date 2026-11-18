//! 360 rig solves (splat-core's panorama split): stage list, flags and the post-solve rig check.

use std::path::Path;

use splat_sfm::{Matching, Mapper, ReconstructOptions, RigOptions};

fn rig_options(dir: &Path, enforce: bool) -> ReconstructOptions {
    let config = dir.join("rig_config.json");
    std::fs::write(&config, "[]").unwrap();
    let mut opts = ReconstructOptions::new(dir.join("out"));
    opts.camera_model = "PINHOLE".into();
    opts.mapper = Mapper::Incremental;
    opts.matching = Matching::Sequential;
    opts.sequential_overlap = 6;
    opts.max_features = 4096;
    opts.max_image_size = 1024;
    opts.ransac_max_trials = 2000;
    opts.init_min_tri_angle = 8.0;
    opts.ba_global_ratio = Some(1.3);
    opts.rig = Some(RigOptions { config, camera_params: "512,512,512,512".into(), enforce, pairs: None });
    opts
}

fn args(stage: &splat_sfm::StageCommand) -> Vec<String> {
    stage.args.iter().map(|a| a.to_string_lossy().into_owned()).collect()
}

fn value(stage: &splat_sfm::StageCommand, key: &str) -> Option<String> {
    let a = args(stage);
    a.iter().position(|x| x == &format!("--{key}")).map(|i| a[i + 1].clone())
}

#[test]
fn rig_solve_runs_extractor_rig_configurator_matcher_mapper_with_the_rig_flags() {
    let tmp = tempfile::tempdir().unwrap();
    let opts = rig_options(tmp.path(), true);
    let stages = splat_sfm::commands(tmp.path(), &opts).unwrap();
    let names: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["feature_extractor", "rig_configurator", "sequential_matcher", "mapper"]);
    let [extract, rig, matcher, mapper] = &stages[..] else { unreachable!() };
    assert_eq!(value(extract, "ImageReader.camera_model").as_deref(), Some("PINHOLE"));
    assert_eq!(value(extract, "ImageReader.single_camera").as_deref(), Some("1"));
    assert_eq!(value(extract, "ImageReader.camera_params").as_deref(), Some("512,512,512,512"));
    assert_eq!(value(extract, "SiftExtraction.max_num_features").as_deref(), Some("4096"));
    assert_eq!(value(extract, "FeatureExtraction.max_image_size").as_deref(), Some("1024"));
    assert_eq!(value(rig, "rig_config_path").map(std::path::PathBuf::from), Some(tmp.path().join("rig_config.json")));
    for (key, v) in [("SequentialMatching.overlap", "6"), ("SequentialMatching.quadratic_overlap", "1"),
        ("SequentialMatching.expand_rig_images", "1"), ("FeatureMatching.rig_verification", "1"),
        ("FeatureMatching.skip_image_pairs_in_same_frame", "1"), ("TwoViewGeometry.max_num_trials", "2000")] {
        assert_eq!(value(matcher, key).as_deref(), Some(v), "{key}");
    }
    // Verification stays inside the matcher (rig verification).
    assert_eq!(value(matcher, "FeatureMatching.skip_geometric_verification"), None);
    for (key, v) in [("Mapper.ba_refine_sensor_from_rig", "0"), ("Mapper.ba_refine_focal_length", "0"),
        ("Mapper.ba_refine_principal_point", "0"), ("Mapper.ba_refine_extra_params", "0"),
        ("Mapper.init_min_tri_angle", "8"), ("Mapper.ba_global_frames_ratio", "1.3"), ("Mapper.ba_global_points_ratio", "1.3")] {
        assert_eq!(value(mapper, key).as_deref(), Some(v), "{key}");
    }
    // No flag twice (COLMAP rejects repeated options).
    for stage in &stages {
        let keys: Vec<String> = args(stage).into_iter().filter(|a| a.starts_with("--")).collect();
        let unique: std::collections::BTreeSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), keys.len(), "{}: {keys:?}", stage.name);
    }
}

#[test]
fn unrigged_baseline_has_no_rig_stage_or_rig_flags() {
    let tmp = tempfile::tempdir().unwrap();
    let opts = rig_options(tmp.path(), false);
    let stages = splat_sfm::commands(tmp.path(), &opts).unwrap();
    let names: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["feature_extractor", "sequential_matcher", "mapper"]);
    assert_eq!(value(&stages[0], "ImageReader.camera_params").as_deref(), Some("512,512,512,512"));
    for key in ["FeatureMatching.rig_verification", "FeatureMatching.skip_image_pairs_in_same_frame", "SequentialMatching.expand_rig_images"] {
        assert_eq!(value(&stages[1], key), None, "{key}");
    }
    assert_eq!(value(&stages[2], "Mapper.ba_refine_sensor_from_rig"), None);
    assert_eq!(value(&stages[2], "Mapper.ba_refine_focal_length").as_deref(), Some("0"));
}

#[test]
fn reused_database_leaves_only_the_mapper() {
    let tmp = tempfile::tempdir().unwrap();
    let mut opts = rig_options(tmp.path(), true);
    opts.reuse_database = Some(tmp.path().join("frozen.db"));
    opts.init_min_tri_angle = 4.0;
    let stages = splat_sfm::reused_stages(splat_sfm::commands(tmp.path(), &opts).unwrap(), &opts);
    let names: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["mapper"]);
    assert_eq!(value(&stages[0], "Mapper.init_min_tri_angle").as_deref(), Some("4"));
}

#[test]
fn rig_options_refuse_the_video_ladder_and_free_intrinsics() {
    let tmp = tempfile::tempdir().unwrap();
    let mut opts = rig_options(tmp.path(), true);
    opts.lens_last = true;
    assert!(splat_sfm::commands(tmp.path(), &opts).is_err());
    let mut opts = rig_options(tmp.path(), true);
    opts.camera_model = "RADIAL".into();
    assert!(splat_sfm::commands(tmp.path(), &opts).is_err());
    let mut opts = rig_options(tmp.path(), true);
    opts.mapper = Mapper::Global;
    assert!(splat_sfm::commands(tmp.path(), &opts).is_err());
    assert!(!splat_sfm::stage_needs_gpu_lock("rig_configurator"));
}

/// Row-major rotation about y by `deg`.
fn ry(deg: f64) -> [f64; 9] {
    let (s, c) = deg.to_radians().sin_cos();
    [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
}

fn mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut m = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            m[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    m
}

fn quat(m: &[f64; 9]) -> [f64; 4] {
    let q = glam::DQuat::from_mat3(&glam::DMat3::from_cols_array(m).transpose());
    [q.w, q.x, q.y, q.z]
}

/// A text model: 3 frames along x, each with 3 views at yaw 0/120/240 (+ `twist` on frame 1 view 2).
fn rig_model(dir: &Path, twist_deg: f64, shift: f64) -> Vec<[f64; 9]> {
    std::fs::create_dir_all(dir).unwrap();
    let layout = [ry(0.0), ry(120.0), ry(240.0)];
    std::fs::write(dir.join("cameras.txt"), "1 PINHOLE 64 64 32 32 32 32\n").unwrap();
    let mut images = String::new();
    let mut id = 1;
    for frame in 0..3 {
        let world_from_rig_r = ry(10.0 * frame as f64);
        let rig_from_world = [world_from_rig_r[0], world_from_rig_r[3], world_from_rig_r[6], world_from_rig_r[1], world_from_rig_r[4], world_from_rig_r[7], world_from_rig_r[2], world_from_rig_r[5], world_from_rig_r[8]];
        let centre = [frame as f64, 0.0, 0.0];
        for (view, cam_from_rig) in layout.iter().enumerate() {
            let mut r = mul(cam_from_rig, &rig_from_world);
            let mut c = centre;
            if frame == 1 && view == 2 {
                r = mul(&ry(twist_deg), &r);
                c[1] += shift;
            }
            // t = -R C
            let t = [0, 1, 2].map(|i| -(r[i * 3] * c[0] + r[i * 3 + 1] * c[1] + r[i * 3 + 2] * c[2]));
            let q = quat(&r);
            images.push_str(&format!("{id} {} {} {} {} {} {} {} 1 v{view:02}_f{frame}.jpg\n\n", q[0], q[1], q[2], q[3], t[0], t[1], t[2]));
            id += 1;
        }
    }
    std::fs::write(dir.join("images.txt"), images).unwrap();
    std::fs::write(dir.join("points3D.txt"), "").unwrap();
    layout.to_vec()
}

#[test]
fn rig_check_accepts_the_layout_and_rejects_a_twisted_or_shifted_view() {
    let tmp = tempfile::tempdir().unwrap();
    let prefixes: Vec<String> = (0..3).map(|i| format!("v{i:02}_")).collect();
    let frames: Vec<String> = (0..3).map(|i| format!("f{i}.jpg")).collect();
    let layout = rig_model(&tmp.path().join("ok"), 0.0, 0.0);
    let check = splat_sfm::check_rig(&tmp.path().join("ok"), &prefixes, &layout, &frames, 0.01, 1e-4).unwrap();
    assert_eq!((check.frames, check.views), (3, 9));
    assert!(check.worst_angle_deg < 1e-6 && check.worst_centre_rel < 1e-9 && (check.frame_step - 1.0).abs() < 1e-9);
    rig_model(&tmp.path().join("twist"), 0.05, 0.0);
    let error = splat_sfm::check_rig(&tmp.path().join("twist"), &prefixes, &layout, &frames, 0.01, 1e-4).unwrap_err();
    assert!(format!("{error:#}").contains("v02_f1.jpg is 0.05"), "{error:#}");
    rig_model(&tmp.path().join("shift"), 0.0, 0.001);
    let error = splat_sfm::check_rig(&tmp.path().join("shift"), &prefixes, &layout, &frames, 0.01, 1e-4).unwrap_err();
    assert!(format!("{error:#}").contains("centre"), "{error:#}");
}

#[test]
fn each_view_prefix_gets_its_own_camera_before_rig_configuration() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("database.db");
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute_batch("CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER);
        CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, camera_id INTEGER);
        INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (1, 256, 256, x'00', 1);").unwrap();
    for frame in 0..3 {
        for view in 0..3 {
            db.execute("INSERT INTO images (name, camera_id) VALUES (?1, 1)", [format!("v{view:02}_frame_{frame}.jpg")]).unwrap();
        }
    }
    // A name that merely matches the prefix as a LIKE pattern ('_' is a wildcard there) stays put.
    db.execute("INSERT INTO images (name, camera_id) VALUES ('v01xframe.jpg', 1)", []).unwrap();
    drop(db);
    let prefixes: Vec<String> = (0..3).map(|i| format!("v{i:02}_")).collect();
    assert_eq!(splat_sfm::split_cameras_by_prefix(&path, &prefixes).unwrap(), 3);
    let db = rusqlite::Connection::open(&path).unwrap();
    let camera = |name: &str| -> i64 { db.query_row("SELECT camera_id FROM images WHERE name = ?1", [name], |r| r.get(0)).unwrap() };
    assert_eq!((camera("v00_frame_0.jpg"), camera("v01_frame_2.jpg"), camera("v02_frame_1.jpg")), (1, 2, 3));
    assert_eq!(camera("v01xframe.jpg"), 1);
    let params: Vec<u8> = db.query_row("SELECT params FROM cameras WHERE camera_id = 3", [], |r| r.get(0)).unwrap();
    assert_eq!(params, [0]);
    drop(db);
    // Idempotent: a reused database keeps its cameras.
    assert_eq!(splat_sfm::split_cameras_by_prefix(&path, &prefixes).unwrap(), 3);
    let tmp_config = tmp.path().join("rig.json");
    std::fs::write(&tmp_config, serde_json::to_vec(&serde_json::json!([{"cameras": [{"image_prefix": "v00_"}, {"image_prefix": "v01_"}]}])).unwrap()).unwrap();
    assert_eq!(splat_sfm::rig_prefixes(&tmp_config).unwrap(), ["v00_", "v01_"]);
}

#[test]
fn rig_pairs_probe_retrieve_then_match_overlapping_views_with_rig_verification_last() {
    let tmp = tempfile::tempdir().unwrap();
    let mut opts = rig_options(tmp.path(), true);
    opts.vocab_tree = Some(tmp.path().join("tree.bin"));
    std::fs::write(tmp.path().join("tree.bin"), b"").unwrap();
    opts.rig.as_mut().unwrap().pairs = Some(splat_sfm::RigPairs::default());
    let stages = splat_sfm::commands(tmp.path(), &opts).unwrap();
    let names: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["feature_extractor", "rig_configurator", "rig_probe_matcher", "rig_loop_matcher", "rig_pair_matcher", "mapper"]);
    let programs: Vec<String> = stages.iter().map(|s| args(s)[0].clone()).collect();
    assert_eq!(programs, ["feature_extractor", "rig_configurator", "matches_importer", "vocab_tree_matcher", "matches_importer", "mapper"]);
    let [_, _, probe, retrieve, pairs, mapper] = &stages[..] else { unreachable!() };
    let out = tmp.path().join("out");
    assert_eq!(value(probe, "match_list_path").map(std::path::PathBuf::from), Some(out.join("rig_probe_pairs.txt")));
    assert_eq!(value(pairs, "match_list_path").map(std::path::PathBuf::from), Some(out.join("rig_pairs.txt")));
    assert_eq!(value(retrieve, "VocabTreeMatching.match_list_path").map(std::path::PathBuf::from), Some(out.join("rig_loop_queries.txt")));
    assert_eq!(value(retrieve, "VocabTreeMatching.num_images").as_deref(), Some("50"));
    assert_eq!(value(retrieve, "VocabTreeMatching.max_num_features").as_deref(), Some("512"));
    // Rig verification runs once, after every pair is matched.
    for (stage, rig) in [(probe, "0"), (retrieve, "0"), (pairs, "1")] {
        assert_eq!(value(stage, "FeatureMatching.rig_verification").as_deref(), Some(rig), "{}", stage.name);
        assert_eq!(value(stage, "FeatureMatching.skip_image_pairs_in_same_frame").as_deref(), Some("1"), "{}", stage.name);
        assert_eq!(value(stage, "TwoViewGeometry.max_num_trials").as_deref(), Some("2000"), "{}", stage.name);
        assert_eq!(value(stage, "SequentialMatching.expand_rig_images"), None);
    }
    assert_eq!(value(mapper, "Mapper.ba_refine_sensor_from_rig").as_deref(), Some("0"));
    for stage in &stages {
        let keys: Vec<String> = args(stage).into_iter().filter(|a| a.starts_with("--")).collect();
        let unique: std::collections::BTreeSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), keys.len(), "{}: {keys:?}", stage.name);
        assert_eq!(value(stage, "log_target").as_deref(), Some("stderr"));
    }
    // Without a tree there is no retrieval; a reused database maps only.
    opts.vocab_tree = None;
    let names: Vec<String> = splat_sfm::commands(tmp.path(), &opts).unwrap().into_iter().map(|s| s.name).collect();
    assert_eq!(names, ["feature_extractor", "rig_configurator", "rig_probe_matcher", "rig_pair_matcher", "mapper"]);
    opts.reuse_database = Some(tmp.path().join("frozen.db"));
    let names: Vec<String> = splat_sfm::reused_stages(splat_sfm::commands(tmp.path(), &opts).unwrap(), &opts).into_iter().map(|s| s.name).collect();
    assert_eq!(names, ["mapper"]);
}
