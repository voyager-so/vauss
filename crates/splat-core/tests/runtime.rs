use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::{
    Run, RunCtx, atomic_json,
    jobs::{ProcessIdentity, identity_matches},
};
use std::{
    fs,
    io::Write,
    path::Path,
    process::Command,
    thread,
    time::{Duration, Instant},
};
const WORKER: &str = env!("CARGO_BIN_EXE_splat-core-fixture");

fn fixture() -> (tempfile::TempDir, Run) {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("input.mp4"), "fixture input").unwrap();
    let run = Run::create(
        dir.path().join("run"),
        dir.path().join("input.mp4"),
        "exercise runtime",
        "fake",
    )
    .unwrap()
    .with_worker_exe(WORKER);
    (dir, run)
}
fn job(run: &Run, action: &str, argument: &str) -> String {
    run.start_job(JobSpec {
        tool: "train_splats".into(),
        argv: vec![WORKER.into(), action.into(), argument.into()],
        env: Default::default(),
        cwd: None,
        on_complete: "core:exit".into(),
        meta: Default::default(),
    })
    .unwrap()
}
fn wait_file(path: &Path) -> Value {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !path.exists() {
        assert!(Instant::now() < deadline, "Waiting for {}", path.display());
        thread::sleep(Duration::from_millis(10));
    }
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
#[test]
fn journal_recovers_torn_tail_and_rejects_interior_corruption() {
    let (_tmp, run) = fixture();
    let path = run.root().join("journal.jsonl");
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{\"seq\":")
        .unwrap();
    let reopened = Run::open(run.root()).unwrap();
    assert_eq!(
        reopened.events().unwrap().last().unwrap().event,
        "journal_recovered"
    );
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, [b"broken\n".as_slice(), &bytes].concat()).unwrap();
    assert!(
        Run::open(run.root())
            .err()
            .unwrap()
            .to_string()
            .contains("Corrupt journal")
    );
}
#[test]
fn snapshot_is_rebuilt_and_source_changes_rejected() {
    let (_tmp, run) = fixture();
    fs::write(run.root().join("state.json"), "broken").unwrap();
    let reopened = Run::open(run.root()).unwrap();
    let snapshot: Value =
        serde_json::from_slice(&fs::read(run.root().join("state.json")).unwrap()).unwrap();
    assert_eq!(
        snapshot,
        serde_json::to_value(reopened.state().unwrap()).unwrap()
    );
    fs::write(run.source(), "changed").unwrap();
    assert!(
        Run::open(run.root())
            .err()
            .unwrap()
            .to_string()
            .contains("Source changed")
    );
}
#[test]
fn artifact_immutable_and_confined() {
    let (tmp, run) = fixture();
    fs::write(tmp.path().join("outside"), "x").unwrap();
    let mut a = Artifact {
        id: "bad".into(),
        kind: ArtifactKind::Frames,
        path: "../outside".into(),
        meta: Default::default(),
    };
    assert!(run.register(a.clone()).is_err());
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(tmp.path().join("outside"), run.root().join("escape")).unwrap();
        a.path = "escape".into();
        assert!(run.register(a.clone()).is_err());
    }
    let (id, path) = run.new_artifact_dir(ArtifactKind::Frames).unwrap();
    a.id = id;
    a.path = path
        .strip_prefix(run.root())
        .unwrap()
        .to_str()
        .unwrap()
        .into();
    run.register(a.clone()).unwrap();
    run.register(a.clone()).unwrap();
    a.meta.insert("changed".into(), json!(true));
    assert!(
        run.register(a)
            .unwrap_err()
            .to_string()
            .contains("immutable")
    );
}
#[test]
fn reregistering_an_artifact_with_float_meta_is_idempotent() {
    // The default serde_json parser reads 20.748512955752282 back as 20.74851295575228, so an artifact
    // whose meta holds full-precision floats (render_shot's render_s, alpha_mean) looked changed when the
    // dispatcher registered it a second time ("Artifact is immutable").
    let (_tmp, run) = fixture();
    let (id, path) = run.new_artifact_dir(ArtifactKind::Video).unwrap();
    let mut a = Artifact {
        id,
        kind: ArtifactKind::Video,
        path: path.strip_prefix(run.root()).unwrap().to_str().unwrap().into(),
        meta: Default::default(),
    };
    a.meta.insert("render_s".into(), json!(20.748512955752282_f64));
    a.meta.insert("alpha_mean".into(), json!(0.1_f64 + 0.2_f64));
    run.register(a.clone()).unwrap();
    run.register(a).unwrap();
}
#[test]
fn backend_crash_records_exit_never_relaunches() {
    let (_tmp, run) = fixture();
    let id = job(&run, "exit", "17");
    let result = run.jobs().wait(&id, 5.0).unwrap();
    assert_eq!(result.status, Status::Failed);
    assert_eq!(
        wait_file(&run.root().join("jobs").join(&id).join("completion.json"))["exit_code"],
        17
    );
    assert_eq!(
        Run::open(run.root())
            .unwrap()
            .jobs()
            .status(&id, 0.0)
            .unwrap()
            .status,
        Status::Failed
    );
    assert_eq!(run.state().unwrap().jobs.len(), 1);
}
#[test]
fn launcher_crash_reconciles_completed_job() {
    let (_tmp, run) = fixture();
    assert_eq!(
        Command::new(WORKER)
            .arg("start-crash")
            .arg(run.root())
            .status()
            .unwrap()
            .code(),
        Some(73)
    );
    let reopened = Run::open(run.root()).unwrap();
    let id = reopened
        .state()
        .unwrap()
        .jobs
        .keys()
        .next()
        .unwrap()
        .clone();
    assert_eq!(
        reopened.jobs().wait(&id, 5.0).unwrap().status,
        Status::Succeeded
    );
    assert_eq!(reopened.state().unwrap().jobs.len(), 1);
}
#[cfg(unix)]
#[test]
fn worker_crash_keeps_child_owned_until_exit() {
    let (_tmp, run) = fixture();
    let id = job(&run, "sleep", "700");
    let dir = run.root().join("jobs").join(&id);
    let worker = wait_file(&dir.join("worker.json"));
    wait_file(&dir.join("process.json"));
    unsafe {
        libc::kill(worker["pid"].as_u64().unwrap() as i32, libc::SIGKILL);
    }
    assert_eq!(
        Run::open(run.root())
            .unwrap()
            .jobs()
            .status(&id, 0.0)
            .unwrap()
            .status,
        Status::Running
    );
    let result = run.jobs().wait(&id, 5.0).unwrap();
    assert_eq!(result.status, Status::Failed);
    assert!(result.error.unwrap().message.contains("-255"));
}
#[test]
fn stop_sends_one_interrupt_waiting_for_checkpoint() {
    let (_tmp, run) = fixture();
    let checkpoint = run.root().join("checkpoint.json");
    let id = job(&run, "checkpoint", checkpoint.to_str().unwrap());
    wait_file(&checkpoint.with_extension("ready"));
    assert_eq!(run.jobs().stop(&id, 0.0).unwrap().status, Status::Running);
    assert_eq!(run.jobs().stop(&id, 5.0).unwrap().status, Status::Succeeded);
    assert_eq!(wait_file(&checkpoint)["saved"], true);
    assert_eq!(
        run.events()
            .unwrap()
            .iter()
            .filter(|e| e.event == "job_stop_requested")
            .count(),
        1
    );
}
#[test]
fn pid_reuse_not_signalled() {
    let (_tmp, run) = fixture();
    let id = job(&run, "sleep", "300");
    let path = run.root().join("jobs").join(&id).join("process.json");
    let mut identity = wait_file(&path);
    identity["start_time"] = json!("wrong");
    atomic_json(&path, &identity).unwrap();
    assert!(
        run.jobs()
            .stop(&id, 0.0)
            .unwrap()
            .warnings
            .iter()
            .any(|w| w.contains("identity changed"))
    );
    assert!(!path.parent().unwrap().join("stop.json").exists());
    assert_eq!(run.jobs().wait(&id, 5.0).unwrap().status, Status::Succeeded);
}
#[test]
fn rejects_unsafe_pid_and_invalid_ids() {
    for pid in [0, 1, u32::MAX] {
        assert!(
            identity_matches(&ProcessIdentity {
                pid,
                pgid: pid,
                start_time: "x".into(),
                boot_id: "x".into(),
                state: "x".into()
            })
            .is_err()
        );
    }
    let (_tmp, run) = fixture();
    assert!(run.jobs().status("../escape", 0.0).is_err());
}
#[test]
fn spawn_failure_durable() {
    let (_tmp, run) = fixture();
    let run = run.with_worker_exe("/does/not/exist");
    let id = job(&run, "exit", "0");
    assert_eq!(run.jobs().wait(&id, 5.0).unwrap().status, Status::Failed);
    assert_eq!(
        Run::open(run.root()).unwrap().state().unwrap().jobs.len(),
        1
    );
    let error = run.jobs().status(&id, 0.).unwrap().error.unwrap();
    assert_eq!(error.code, "worker_spawn_failed");
    assert!(error.message.contains("/does/not/exist"));
}

fn registry() -> std::sync::Arc<splat_core::Registry> {
    let mut registry = splat_core::Registry::new();
    registry.register_backend(std::sync::Arc::new(
        splat_core::FakeBackend::with_executable(WORKER),
    ));
    std::sync::Arc::new(registry)
}
#[test]
fn dispatch_normalizes_defaults_validates_bounds_and_caches() {
    let (_tmp, run) = fixture();
    let run = run.with_registry(registry());
    let first = splat_core::dispatch(&run, "extract_frames", json!({}));
    assert_eq!(first.status, Status::Succeeded);
    let second = splat_core::dispatch(
        &run,
        "extract_frames",
        json!({"target_count":150,"strategy":"sharpness_window"}),
    );
    assert_eq!(first.artifacts[0].id, second.artifacts[0].id);
    for payload in [
        json!({"target_count":1}),
        json!({"rotate_deg":45}),
        json!({"time_range_s":[2.,1.]}),
        json!({"unknown":true}),
    ] {
        let result = splat_core::dispatch(&run, "extract_frames", payload);
        assert_eq!(result.status, Status::Failed);
        assert_eq!(result.error.unwrap().code, "invalid_request");
    }
    assert_eq!(run.state().unwrap().artifacts.len(), 1);
}
#[test]
fn concurrent_dispatch_starts_one_job_and_crash_deduplicates() {
    let (_tmp, run) = fixture();
    let run = run.with_registry(registry());
    let frames = splat_core::dispatch(&run, "extract_frames", json!({})).artifacts[0]
        .id
        .clone();
    let input = json!({"frames_id":frames}).to_string();
    let children: Vec<_> = (0..3)
        .map(|_| {
            Command::new(WORKER)
                .args([
                    "dispatch",
                    run.root().to_str().unwrap(),
                    "reconstruct_sfm",
                    &input,
                ])
                .stdout(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    let results: Vec<ToolResult> = children
        .into_iter()
        .map(|c| serde_json::from_slice(&c.wait_with_output().unwrap().stdout).unwrap())
        .collect();
    let id = results[0].job_id.as_ref().unwrap();
    assert!(results.iter().all(|r| r.job_id.as_ref() == Some(id)));
    assert_eq!(run.jobs().wait(id, 5.).unwrap().status, Status::Succeeded);
    assert_eq!(run.state().unwrap().jobs.len(), 1);
    let other = json!({"frames_id":frames,"quality":"low"}).to_string();
    assert_eq!(
        Command::new(WORKER)
            .args([
                "dispatch-crash",
                run.root().to_str().unwrap(),
                "reconstruct_sfm",
                &other
            ])
            .status()
            .unwrap()
            .code(),
        Some(73)
    );
    let reopened = Run::open_with_registry(run.root(), registry(), true).unwrap();
    let resumed = splat_core::dispatch(
        &reopened,
        "reconstruct_sfm",
        serde_json::from_str(&other).unwrap(),
    );
    assert_eq!(
        run.jobs()
            .wait(resumed.job_id.as_ref().unwrap(), 5.)
            .unwrap()
            .status,
        Status::Succeeded
    );
    assert_eq!(run.state().unwrap().jobs.len(), 2);
}
#[test]
fn stopped_run_rejects_new_work_until_resume() {
    let (_tmp, run) = fixture();
    let run = run.with_registry(registry());
    run.journal("run_stopped", json!({"status":"stopping"}))
        .unwrap();
    assert_eq!(
        splat_core::dispatch(&run, "extract_frames", json!({})).status,
        Status::Failed
    );
    run.journal("run_started", json!({"status":"running"}))
        .unwrap();
    assert_eq!(
        splat_core::dispatch(&run, "extract_frames", json!({})).status,
        Status::Succeeded
    );
}
#[test]
fn finish_checks_candidate_evidence() {
    let (_tmp, run) = fixture();
    let run = run.with_registry(registry());
    assert_eq!(
        splat_core::dispatch(
            &run,
            "finish",
            json!({"status":"succeeded","explanation":"done"})
        )
        .status,
        Status::Failed
    );
    assert!(!run.root().join("agent/finish.json").exists());
    assert_eq!(
        splat_core::dispatch(
            &run,
            "finish",
            json!({"status":"refused","explanation":"no usable parallax"})
        )
        .status,
        Status::Succeeded
    );
    assert_eq!(run.state().unwrap().status, "refused");
    assert_eq!(
        wait_file(&run.root().join("agent/finish.json"))["status"],
        "refused"
    );
    let report = splat_core::report::write_report(&run).unwrap();
    assert!(report["metrics"]["psnr"].is_null());
    assert_eq!(report["summary"], "no usable parallax");
}

#[test]
fn finish_recovers_crash_between_marker_and_journal() {
    let (_tmp, run) = fixture();
    let run = run.with_registry(registry());
    let request = json!({"status":"refused","explanation":"no usable parallax","remaining_defects":["pure rotation"]});
    assert_eq!(
        splat_core::dispatch(&run, "finish", request.clone()).status,
        Status::Succeeded
    );
    let path = run.root().join("journal.jsonl");
    let events = run.events().unwrap();
    let journal = events
        .iter()
        .take_while(|e| e.event != "agent_finished")
        .map(|e| format!("{}\n", serde_json::to_string(e).unwrap()))
        .collect::<String>();
    fs::write(path, journal).unwrap();
    let reopened = Run::open_with_registry(run.root(), registry(), true).unwrap();
    assert_eq!(
        splat_core::dispatch(&reopened, "finish", request).status,
        Status::Succeeded
    );
    assert_eq!(reopened.state().unwrap().status, "refused");
    assert_eq!(
        reopened
            .events()
            .unwrap()
            .iter()
            .filter(|e| e.event == "agent_finished")
            .count(),
        1
    );
}

#[test]
fn partial_report_preserves_selection_and_resume_discards_old_finish() {
    let (_tmp, run) = fixture();
    let (id, dir) = run.new_artifact_dir(ArtifactKind::Candidate).unwrap();
    run.register(Artifact {
        id: id.clone(),
        kind: ArtifactKind::Candidate,
        path: dir
            .strip_prefix(run.root())
            .unwrap()
            .to_str()
            .unwrap()
            .into(),
        meta: Default::default(),
    })
    .unwrap();
    let result = splat_core::dispatch(
        &run,
        "finish",
        json!({
            "status":"partial", "candidate_id":id,
            "explanation":"Checkpoint retained; evaluation unavailable",
            "remaining_defects":["no held-out evaluation"]
        }),
    );
    assert_eq!(result.status, Status::Succeeded);
    let report = splat_core::report::write_report(&run).unwrap();
    assert_eq!(report["best_candidate"], id);
    assert_eq!(report["metrics"], json!({}));
    assert!(report["deliverables"]["splat"].is_null());
    run.journal("run_started", json!({"status":"running"}))
        .unwrap();
    run.journal(
        "run_finished",
        json!({"status":"failed","stop_reason":"retry failed"}),
    )
    .unwrap();
    let report = splat_core::report::write_report(&run).unwrap();
    assert_eq!(report["stop_reason"], "retry failed");
    assert!(report["best_candidate"].is_null());
    assert_eq!(report["remaining_defects"], json!([]));
    assert_eq!(report["summary"], "No successful evaluation is available.");
}

#[test]
fn report_links_selected_splat_and_matching_successful_mesh() {
    let (_tmp, run) = fixture();
    let (id, dir) = run.new_artifact_dir(ArtifactKind::Candidate).unwrap();
    let saved = dir.join("step-000000200.ckpt");
    fs::create_dir(&saved).unwrap();
    fs::write(saved.join("splat.ply"), "ply").unwrap();
    let relative = saved.strip_prefix(run.root()).unwrap();
    atomic_json(dir.join("checkpoint.json"), &json!({"checkpoint":relative})).unwrap();
    run.register(Artifact {
        id: id.clone(),
        kind: ArtifactKind::Candidate,
        path: dir
            .strip_prefix(run.root())
            .unwrap()
            .to_str()
            .unwrap()
            .into(),
        meta: Default::default(),
    })
    .unwrap();
    let mesh = dir.join("mesh.glb");
    fs::write(&mesh, "mesh").unwrap();
    run.journal("tool_result", json!({"operation_id":"mesh", "result":{
        "tool":"extract_mesh","status":"succeeded", "metrics":{"files":[mesh.strip_prefix(run.root()).unwrap()]},
        "artifacts":[{"meta":{"candidate_id":id}}]
    }})).unwrap();
    assert_eq!(
        splat_core::dispatch(
            &run,
            "finish",
            json!({"status":"partial","candidate_id":id,"explanation":"saved checkpoint"})
        )
        .status,
        Status::Succeeded
    );
    let report = splat_core::report::write_report(&run).unwrap();
    assert_eq!(
        report["deliverables"]["splat"],
        relative.join("splat.ply").to_str().unwrap()
    );
    assert_eq!(
        report["deliverables"]["mesh"],
        mesh.strip_prefix(run.root()).unwrap().to_str().unwrap()
    );
    let markdown = fs::read_to_string(run.root().join("report.md")).unwrap();
    assert!(markdown.contains("## Deliverables"));
    assert!(markdown.contains("[report.json](report.json)"));
    fs::remove_file(&mesh).unwrap();
    assert!(splat_core::report::write_report(&run).unwrap()["deliverables"]["mesh"].is_null());
    fs::remove_file(saved.join("splat.ply")).unwrap();
    fs::write(dir.join("checkpoint.ply"), "ply").unwrap();
    atomic_json(
        dir.join("checkpoint.json"),
        &json!({"ply":"checkpoint.ply"}),
    )
    .unwrap();
    assert_eq!(
        splat_core::report::write_report(&run).unwrap()["deliverables"]["splat"],
        dir.join("checkpoint.ply")
            .strip_prefix(run.root())
            .unwrap()
            .to_str()
            .unwrap()
    );
}

#[test]
fn missing_backend_is_distinct_from_worker_spawn_failure() {
    let (_tmp, run) = fixture();
    let id = run
        .start_job(JobSpec {
            tool: "train_splats".into(),
            argv: vec!["/missing/backend".into()],
            env: Default::default(),
            cwd: None,
            on_complete: "core:exit".into(),
            meta: Default::default(),
        })
        .unwrap();
    let result = run.jobs().wait(&id, 5.).unwrap();
    let error = result.error.unwrap();
    assert_eq!(error.code, "backend_spawn_failed");
    assert!(error.message.contains("/missing/backend"));
}

#[test]
fn camera_solve_below_ninety_percent_is_a_defect_and_flagged_in_the_report() {
    let (_tmp, run) = fixture();
    let register = |kind: ArtifactKind, meta: Value| {
        let (id, dir) = run.new_artifact_dir(kind).unwrap();
        let path = dir.strip_prefix(run.root()).unwrap().to_str().unwrap().into();
        run.register(Artifact { id: id.clone(), kind, path, meta: serde_json::from_value(meta).unwrap() }).unwrap();
        (id, dir)
    };
    let (sfm, sfm_dir) = register(ArtifactKind::Sfm, json!({}));
    let (candidate, _) = register(ArtifactKind::Candidate, json!({"sfm_id": sfm}));
    atomic_json(sfm_dir.join("health.json"), &json!({"n_registered": 11, "n_input": 14})).unwrap();
    assert_eq!(splat_core::sfm_coverage(&run, &candidate), Some((11, 14)));
    let defect = splat_core::sfm_coverage_defect(&run, &candidate).unwrap();
    assert!(defect.contains("11 of 14 frames (79%)"), "{defect}");
    atomic_json(sfm_dir.join("health.json"), &json!({"n_registered": 13, "n_input": 14})).unwrap();
    assert!(splat_core::sfm_coverage_defect(&run, &candidate).is_none());

    run.journal("tool_result", json!({"operation_id":"sfm", "result":{
        "tool":"reconstruct_sfm","status":"succeeded","metrics":{"n_registered":3,"n_input":37}}})).unwrap();
    let report = splat_core::report::write_report(&run).unwrap();
    assert_eq!(report["sfm"]["coverage_ok"], json!(false), "{report}");
    assert!(report["warnings"].to_string().contains("3 of 37 frames"), "{report}");
}
