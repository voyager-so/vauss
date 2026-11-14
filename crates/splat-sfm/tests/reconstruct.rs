use std::fs;
use std::path::Path;

use splat_sfm::{ReconstructOptions, best_model, reconstruct};

#[test]
fn incoherent_geometry_is_a_hard_failure_for_photos_and_video() {
    let mut health = splat_eval::sfm_health(&Default::default(), Path::new("fixture"), &Default::default()).unwrap();
    let evidence = serde_json::from_str(include_str!("../../splat-eval/tests/fixtures/failed179.json")).unwrap();
    health.geometry = splat_eval::geometry::assess_geometry(&evidence, false);
    for video in [false, true] {
        let error = splat_sfm::ensure_video_continuity(&health, video).unwrap_err();
        assert!(error.downcast_ref::<splat_sfm::GeometryFailure>().is_some());
    }
    let evidence = serde_json::from_str(include_str!("../../splat-eval/tests/fixtures/good241.json")).unwrap();
    health.geometry = splat_eval::geometry::assess_geometry(&evidence, true);
    assert!(splat_sfm::ensure_video_continuity(&health, true).is_ok());
}

fn model(dir: &Path, images: usize, error: f64) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("cameras.txt"), "1 PINHOLE 100 80 100 100 50 40\n").unwrap();
    let mut text = String::new();
    for id in 1..=images {
        text.push_str(&format!("{id} 1 0 0 0 0 0 0 1 {id}.png\n78 94 7\n"));
    }
    fs::write(dir.join("images.txt"), text).unwrap();
    fs::write(
        dir.join("points3D.txt"),
        format!("7 1 2 4 100 100 100 {error} 1 0 2 0\n"),
    )
    .unwrap();
}

#[test]
fn selects_largest_component_and_recomputes_stored_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = tmp.path().join("images");
    fs::create_dir(&frames).unwrap();
    for id in 1..=3 {
        fs::write(frames.join(format!("{id}.png")), []).unwrap();
    }
    let sparse = tmp.path().join("sparse");
    model(&sparse.join("0"), 1, 999.0);
    model(&sparse.join("1"), 3, 999.0);
    let (dir, health) = best_model(&sparse, &frames, false, 8).unwrap();
    assert_eq!(dir, sparse.join("1"));
    assert_eq!(health.n_registered, 3);
    assert_eq!(health.reproj_source, "recomputed");
    assert!((health.mean_reproj_err_px.unwrap() - 5.0).abs() < 1e-12);
    assert!(
        fs::read_to_string(dir.join("points3D.txt"))
            .unwrap()
            .contains("999")
    );
}

#[cfg(unix)]
fn fake_colmap(dir: &Path, fail: bool) -> std::path::PathBuf {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    let script = dir.join("fake colmap");
    let (mut file, staged) = tempfile::NamedTempFile::new_in(dir).unwrap().into_parts();
    file.write_all(
        format!(
            r##"#!/bin/sh
if [ "$1" = '-h' ]; then echo 'COLMAP 4.2.1 (test)'; exit 0; fi
if [ '{fail}' = 'true' ]; then echo 'intentional stage failure' >&2; exit 7; fi
cmd="$1"
shift
while [ "$#" -gt 1 ]; do
  if [ "$1" = '--output_path' ]; then out="$2"; fi
  if [ "$1" = '--input_path' ]; then input="$2"; fi
  if [ "$1" = '--database_path' ] && [ ! -e "$2" ]; then : > "$2"; fi
  shift 2
done
if [ "$cmd" = 'point_triangulator' ]; then cp "$input/"*.txt "$out/"; fi
if [ "$cmd" = 'mapper' ]; then
  mkdir -p "$out/0"
  printf '1 PINHOLE 100 80 100 100 50 40\n' > "$out/0/cameras.txt"
  printf '1 1 0 0 0 0 0 0 1 1.png\n78 94 7\n' > "$out/0/images.txt"
  printf '7 1 2 4 100 100 100 99 1 0\n' > "$out/0/points3D.txt"
fi
"##
        )
        .as_bytes(),
    )
    .unwrap();
    // Close the writer before making the complete script executable and publishing it.
    drop(file);
    fs::set_permissions(&staged, fs::Permissions::from_mode(0o755)).unwrap();
    fs::rename(&staged, &script).unwrap();
    script
}

#[cfg(unix)]
#[test]
fn fresh_pipeline_handles_spaces_and_preserves_evidence() {
    let tmp = tempfile::tempdir().unwrap();
    let frames = tmp.path().join("input photos");
    fs::create_dir(&frames).unwrap();
    fs::write(frames.join("1.png"), []).unwrap();
    let mut opts = ReconstructOptions::new(tmp.path().join("a new run"));
    opts.colmap = fake_colmap(tmp.path(), false);
    opts.use_gpu = false;
    let report = reconstruct(&frames, &opts).unwrap();
    assert_eq!(report.health.n_registered, 1);
    assert_eq!(report.stages.len(), 5);
    assert!(opts.output_dir.join("mapped/0/cameras.txt").is_file());
    let evidence = fs::read(opts.output_dir.join("reconstruction.json")).unwrap();
    assert!(
        reconstruct(&frames, &opts)
            .unwrap_err()
            .to_string()
            .contains("fresh")
    );
    assert_eq!(
        evidence,
        fs::read(opts.output_dir.join("reconstruction.json")).unwrap()
    );
}

#[cfg(unix)]
#[test]
fn failed_stage_retains_log_and_does_not_report_success() {
    let tmp = tempfile::tempdir().unwrap();
    let mut opts = ReconstructOptions::new(tmp.path().join("failed"));
    opts.colmap = fake_colmap(tmp.path(), true);
    let err = reconstruct(tmp.path(), &opts).unwrap_err().to_string();
    assert!(err.contains("feature_extractor failed"));
    assert!(
        fs::read_to_string(opts.output_dir.join("feature_extractor.log"))
            .unwrap()
            .contains("intentional stage failure")
    );
    assert!(!opts.output_dir.join("reconstruction.json").exists());
}

#[cfg(unix)]
#[test]
fn native_child_keeps_gpu_lock_after_worker_is_killed() {
    use fs2::FileExt;
    use std::os::unix::fs::PermissionsExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let tmp = tempfile::tempdir().unwrap();
    let ready = tmp.path().join("ready");
    let finish = tmp.path().join("finish");
    let script = tmp.path().join("colmap");
    fs::write(
        &script,
        format!(
            r##"#!/bin/sh
if [ "$1" = '-h' ]; then echo 'COLMAP 4.2.1 (test)'; exit 0; fi
touch '{}'
i=0
while [ ! -e '{}' ] && [ "$i" -lt 500 ]; do sleep 0.01; i=$((i+1)); done
exit 7
"##,
            ready.display(),
            finish.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let lock_path = tmp.path().join("gpu.lock");
    let mut worker = Command::new(env!("CARGO_BIN_EXE_splat-sfm"))
        .arg(tmp.path())
        .arg("--output")
        .arg(tmp.path().join("run"))
        .arg("--colmap")
        .arg(script)
        .arg("--gpu-lock")
        .arg(&lock_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ready.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    worker.kill().unwrap();
    worker.wait().unwrap();
    assert!(ready.exists(), "native stage did not start");
    let lock = fs::OpenOptions::new().write(true).open(lock_path).unwrap();
    let blocked = lock.try_lock_exclusive().is_err();
    // Release the fixture child even when the assertion would fail.
    fs::write(finish, []).unwrap();
    assert!(
        blocked,
        "worker death released lock while native child was still alive"
    );
    while lock.try_lock_exclusive().is_err() {
        assert!(
            Instant::now() < deadline,
            "lock leaked after native child exited"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn global_mapper_takes_the_mapper_seed_and_runs_on_cpu() {
    let tmp = tempfile::tempdir().unwrap();
    let mut opts = ReconstructOptions::new(tmp.path().join("out"));
    opts.mapper = splat_sfm::Mapper::Global;
    opts.mapper_seed = 7;
    opts.use_gpu = true;
    let stages = splat_sfm::commands(tmp.path(), &opts).unwrap();
    let mapper = stages.iter().find(|s| s.name == "global_mapper").unwrap();
    let argv = mapper.args.iter().map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" ");
    assert!(argv.contains("--GlobalMapper.random_seed 7"), "{argv}");
    assert!(argv.contains("--GlobalMapper.gp_use_gpu 0") && argv.contains("--GlobalMapper.ba_ceres_use_gpu 0"), "{argv}");
}

#[test]
fn global_mapper_keeps_a_fixed_focal_fixed() {
    let tmp = tempfile::tempdir().unwrap();
    let mut opts = ReconstructOptions::new(tmp.path().join("out"));
    opts.mapper = splat_sfm::Mapper::Global;
    let argv = |opts: &ReconstructOptions| {
        let stages = splat_sfm::commands(tmp.path(), opts).unwrap();
        let names = stages.iter().map(|s| s.name.clone()).collect::<Vec<_>>();
        let mapper = stages.iter().find(|s| s.name == "global_mapper").unwrap();
        (names, mapper.args.iter().map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" "))
    };
    let (names, free) = argv(&opts);
    assert!(names.iter().any(|n| n == "view_graph_calibrator"), "{names:?}");
    assert!(!free.contains("ba_refine_focal_length"), "{free}");
    opts.fixed_focal_factor = Some(0.8);
    let (names, fixed) = argv(&opts);
    assert!(!names.iter().any(|n| n == "view_graph_calibrator"), "{names:?}");
    assert!(fixed.contains("--GlobalMapper.ba_refine_focal_length 0"), "{fixed}");
}

#[test]
fn only_cuda_feature_stages_take_the_gpu_lock() {
    for stage in ["feature_extractor", "exhaustive_matcher", "sequential_matcher"] {
        assert!(splat_sfm::stage_needs_gpu_lock(stage), "{stage}");
    }
    for stage in ["global_mapper", "mapper", "view_graph_calibrator", "bundle_adjuster"] {
        assert!(!splat_sfm::stage_needs_gpu_lock(stage), "{stage}");
    }
}

/// A failed global hybrid solve retries the low-angle rung with the incremental mapper (the angle is
/// an incremental option), not a second global solve.
#[cfg(unix)]
#[test]
fn failed_global_solve_retries_low_angle_incrementally() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let frames = tmp.path().join("images");
    fs::create_dir(&frames).unwrap();
    for id in 1..=4 {
        fs::write(frames.join(format!("{id}.png")), []).unwrap();
    }
    let calls = tmp.path().join("calls.log");
    // The smallest COLMAP database the ladder reads: one SIMPLE_RADIAL camera, no matches.
    let template = tmp.path().join("template.db");
    let db = rusqlite::Connection::open(&template).unwrap();
    db.execute_batch(
        "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER);
         CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
         CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
         CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB, config INTEGER);",
    )
    .unwrap();
    let params: Vec<u8> = [120.0f64, 50.0, 40.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    db.execute("INSERT INTO cameras VALUES (1, 2, 100, 80, ?1, 0)", [params]).unwrap();
    drop(db);
    let script = tmp.path().join("colmap");
    fs::write(
        &script,
        format!(
            r##"#!/bin/sh
if [ "$1" = '-h' ]; then echo 'COLMAP 4.2.1 (test)'; exit 0; fi
cmd="$1"
shift
echo "$cmd $*" >> '{calls}'
while [ "$#" -gt 1 ]; do
  if [ "$1" = '--output_path' ]; then out="$2"; fi
  if [ "$1" = '--input_path' ]; then input="$2"; fi
  if [ "$1" = '--database_path' ] && [ ! -e "$2" ]; then cp '{template}' "$2"; fi
  shift 2
done
if [ "$cmd" = 'point_triangulator' ] || [ "$cmd" = 'bundle_adjuster' ]; then cp "$input/"*.txt "$out/"; fi
if [ "$cmd" = 'global_mapper' ]; then n=1; elif [ "$cmd" = 'mapper' ]; then n=4; else exit 0; fi
mkdir -p "$out/0"
printf '1 PINHOLE 100 80 100 100 50 40\n' > "$out/0/cameras.txt"
: > "$out/0/images.txt"
i=1
while [ "$i" -le "$n" ]; do printf '%s 1 0 0 0 0 0 0 1 %s.png\n78 94 7\n' "$i" "$i" >> "$out/0/images.txt"; i=$((i+1)); done
printf '7 1 2 4 100 100 100 0.5 1 0\n' > "$out/0/points3D.txt"
"##,
            calls = calls.display(),
            template = template.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let vocab = tmp.path().join("vocab.bin");
    fs::write(&vocab, []).unwrap();
    let mut opts = ReconstructOptions::new(tmp.path().join("run"));
    opts.colmap = script;
    opts.use_gpu = false;
    opts.mapper = splat_sfm::Mapper::Global;
    opts.matching = splat_sfm::Matching::Hybrid;
    opts.vocab_tree = Some(vocab);
    let _ = reconstruct(&frames, &opts);
    let log = fs::read_to_string(&calls).unwrap();
    let mappers: Vec<&str> = log
        .lines()
        .filter_map(|l| l.split(' ').next())
        .filter(|c| *c == "global_mapper" || *c == "mapper")
        .collect();
    assert_eq!(mappers.first(), Some(&"global_mapper"), "{log}");
    assert_eq!(mappers.iter().filter(|c| **c == "global_mapper").count(), 1, "{log}");
    let low = log
        .lines()
        .find(|l| l.starts_with("mapper ") && l.contains("--Mapper.init_min_tri_angle"))
        .unwrap_or_else(|| panic!("no incremental low-angle retry:\n{log}"));
    assert!(low.contains("--Mapper.init_min_tri_angle 4"), "{low}");
}

#[cfg(unix)]
fn fixed_pose_run(model: &str, params: Vec<f64>, matching: splat_sfm::Matching) -> (Vec<String>, splat_eval::Model) {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let template = tmp.path().join("template.db");
    let db = rusqlite::Connection::open(&template).unwrap();
    db.execute_batch("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
        CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER);
        CREATE TABLE rigs (rig_id INTEGER PRIMARY KEY, ref_sensor_id INTEGER, ref_sensor_type INTEGER);
        CREATE TABLE rig_sensors (rig_id INTEGER, sensor_id INTEGER, sensor_type INTEGER, sensor_from_rig BLOB);
        CREATE TABLE frames (frame_id INTEGER PRIMARY KEY, rig_id INTEGER);
        CREATE TABLE frame_data (frame_id INTEGER, data_id INTEGER, sensor_id INTEGER, sensor_type INTEGER);
        INSERT INTO images VALUES (1,'a.png',1),(2,'b.png',3);
        INSERT INTO cameras VALUES (1,2,100,80,X'00',0),(3,2,100,80,X'00',0);
        INSERT INTO rigs VALUES (1,1,0),(3,3,0);
        INSERT INTO frames VALUES (1,1),(2,3);
        INSERT INTO frame_data VALUES (1,1,1,0),(2,2,3,0);").unwrap();
    drop(db);
    let ran = tmp.path().join("ran.txt");
    let script = tmp.path().join("fake colmap");
    let (mut file, staged) = tempfile::NamedTempFile::new_in(tmp.path()).unwrap().into_parts();
    write!(file, r##"#!/bin/sh
if [ "$1" = '-h' ]; then echo 'COLMAP 4.2.1 (test)'; exit 0; fi
cmd="$1"; shift
echo "$cmd" >> '{ran}'
while [ "$#" -gt 1 ]; do
  case "$1" in --output_path) out="$2";; --input_path) input="$2";; --database_path) database="$2";; esac
  shift 2
done
if [ "$cmd" = 'feature_extractor' ]; then cp '{template}' "$database"; fi
if [ "$cmd" = 'point_triangulator' ]; then cp "$input/cameras.txt" "$input/images.txt" "$out/"; printf '1 0 0 3 1 2 3 0.5 1 0\n' > "$out/points3D.txt"; fi
"##, ran = ran.display(), template = template.display()).unwrap();
    drop(file);
    fs::set_permissions(&staged, fs::Permissions::from_mode(0o755)).unwrap();
    fs::rename(&staged, &script).unwrap();
    let frames = tmp.path().join("frames");
    fs::create_dir(&frames).unwrap();
    let cam = |name: &str| splat_sfm::FixedCamera {
        name: name.into(), model: model.into(), width: 100, height: 80,
        params: params.clone(), qvec: [1., 0., 0., 0.], tvec: [0., 0., 0.],
    };
    let mut opts = ReconstructOptions::new(tmp.path().join("out"));
    opts.colmap = script;
    opts.use_gpu = false;
    opts.matching = matching;
    let seed = splat_sfm::triangulate_fixed_poses(&frames, &[cam("a.png"), cam("b.png")], &opts, std::time::Duration::from_secs(60)).unwrap();
    assert_eq!(seed.points, 1);
    let ran = fs::read_to_string(ran).unwrap();
    (ran.lines().map(String::from).collect(), splat_eval::read_model(&seed.model_dir).unwrap())
}

#[cfg(unix)]
#[test]
fn fixed_pose_triangulation_runs_the_verifier_for_sequential_matching() {
    let (ran, _) = fixed_pose_run("PINHOLE", vec![100., 100., 50., 40.], splat_sfm::Matching::Sequential);
    assert_eq!(ran, ["feature_extractor", "sequential_matcher", "geometric_verifier", "point_triangulator"]);
}

#[cfg(unix)]
#[test]
fn fixed_pose_triangulation_keeps_every_supported_camera_model_and_database_ids() {
    for (model, params) in [
        ("SIMPLE_PINHOLE", vec![100., 50., 40.]),
        ("PINHOLE", vec![100., 101., 50., 40.]),
        ("SIMPLE_RADIAL", vec![100., 50., 40., 0.01]),
        ("RADIAL", vec![100., 50., 40., 0.01, 0.001]),
        ("OPENCV", vec![100., 101., 50., 40., 0.01, 0.001, 0.0001, 0.0002]),
    ] {
        let (_, m) = fixed_pose_run(model, params.clone(), splat_sfm::Matching::Exhaustive);
        // Camera ids are the database's (1 and 3), not renumbered 1..n; models/params are the authored ones.
        assert_eq!(m.cameras.keys().copied().collect::<Vec<_>>(), [1, 3], "{model}");
        assert!(m.cameras.values().all(|c| c.model == model && c.params == params), "{model}");
        assert_eq!(m.images.values().map(|i| i.id).collect::<Vec<_>>(), [1, 2], "{model}");
    }
}
