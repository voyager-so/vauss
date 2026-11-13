use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
const BIN: &str = env!("CARGO_BIN_EXE_vauss");
/// The binary, offline: every run here is `--backend fake`, which needs no downloaded tool. A configured
/// FFmpeg that does not exist skips the capture pre-check (FFmpeg's only use in a fake run) instead of
/// downloading FFmpeg, and a configured vocabulary tree keeps the (unused) brush backend from fetching it.
fn splat() -> Command {
    let missing = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let mut command = Command::new(BIN);
    command.env("SPLAT_FFMPEG", missing.join("no-ffmpeg")).env("SPLAT_VOCAB_TREE", missing.join("no-vocab-tree"));
    command
}
fn invoke(args: &[&str]) -> Output {
    splat().args(args).output().unwrap()
}
fn parse(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn root(p: &Path) -> &str {
    p.to_str().unwrap()
}
#[test]
fn primary_command_and_compatibility_aliases_are_identical() {
    for flag in ["--version", "--help"] {
        let primary = Command::new(BIN).arg(flag).output().unwrap();
        assert!(primary.status.success());
        assert!(primary.stderr.is_empty());
        let text = String::from_utf8_lossy(&primary.stdout);
        assert!(text.contains("vauss"), "{text}");
        for alias in [env!("CARGO_BIN_EXE_splat"), env!("CARGO_BIN_EXE_voyager-splat")] {
            let output = Command::new(alias).arg(flag).output().unwrap();
            assert!(output.status.success());
            assert!(output.stderr.is_empty());
            assert_eq!(output.stdout, primary.stdout);
        }
    }
}

#[test]
fn schemas_and_json_file_validation() {
    assert!(parse(invoke(&["tools", "--json"]))["train_splats"].is_object());
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    parse(invoke(&[
        "init",
        root(&source),
        "--goal",
        "test",
        "--backend",
        "fake",
        "--run-dir",
        root(&run),
    ]));
    let request = tmp.path().join("request.json");
    fs::write(&request, "{\"target_count\":20}").unwrap();
    let result = parse(invoke(&[
        "tool",
        "extract_frames",
        "--run-dir",
        root(&run),
        "--input",
        &format!("@{}", request.display()),
    ]));
    assert_eq!(result["status"], "succeeded");
    let invalid = invoke(&[
        "tool",
        "extract_frames",
        "--run-dir",
        root(&run),
        "--input",
        "{\"target_count\":1}",
    ]);
    assert_eq!(invalid.status.code(), Some(5));
    assert_eq!(
        serde_json::from_slice::<Value>(&invalid.stdout).unwrap()["error"]["code"],
        "invalid_request"
    );
}
#[test]
fn worker_exit_preserves_tool_failures_and_write_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    parse(invoke(&["init", root(&source), "--goal", "test", "--backend", "fake",
        "--run-dir", root(&run)]));
    let input = tmp.path().join("input.json");
    let output = tmp.path().join("result.json");
    fs::write(&input, r#"{"target_count":20}"#).unwrap();
    let worker = invoke(&["__tool_job", root(&run), "extract_frames", root(&input), root(&output)]);
    assert!(worker.status.success(), "{}", String::from_utf8_lossy(&worker.stderr));
    let result: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(result["status"], "succeeded");

    // A persisted structured failure uses a clean transport exit; it stays a failed tool result.
    fs::write(&input, r#"{"target_count":"invalid"}"#).unwrap();
    let worker = invoke(&["__tool_job", root(&run), "extract_frames", root(&input), root(&output)]);
    assert!(worker.status.success());
    let result: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(result["status"], "failed");
    assert_eq!(result["error"]["code"], "tool_failed");
    let caller = invoke(&["tool", "extract_frames", "--run-dir", root(&run), "--input",
        r#"{"target_count":"invalid"}"#]);
    assert_eq!(caller.status.code(), Some(5));

    // An unwritable result must never take the successful early exit.
    let blocked = tmp.path().join("blocked.json");
    fs::create_dir(&blocked).unwrap();
    let worker = invoke(&["__tool_job", root(&run), "extract_frames", root(&input), root(&blocked)]);
    assert_eq!(worker.status.code(), Some(1));
}

#[test]
fn fixed_pipeline_and_resume_do_not_duplicate_jobs() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    let first = parse(invoke(&[
        "run",
        root(&source),
        "--goal",
        "exercise runtime",
        "--backend",
        "fake",
        "--driver",
        "fixed",
        "--steps",
        "200",
        "--run-dir",
        root(&run),
        "--json",
    ]));
    assert_eq!(first["status"], "succeeded");
    assert_eq!(first["metrics"]["synthetic"], true);
    // SfM, training, evaluation and mesh: evaluation runs as a job beside the mesh since 0debfed6.
    assert_eq!(first["jobs"].as_object().unwrap().len(), 4);
    assert!(run.join("exports/web/scene.json").is_file());
    let second = parse(invoke(&["resume", "--run-dir", root(&run), "--json"]));
    assert_eq!(first["best_candidate"], second["best_candidate"]);
    assert_eq!(first["jobs"], second["jobs"]);
    assert!(run.join("report.md").is_file());
    assert_eq!(
        parse(invoke(&["status", root(&run)]))["status"],
        "succeeded"
    );
}
#[test]
fn stop_blocks_then_resume_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    parse(invoke(&[
        "init",
        root(&source),
        "--goal",
        "test",
        "--backend",
        "fake",
        "--run-dir",
        root(&run),
    ]));
    assert_eq!(parse(invoke(&["stop", root(&run)]))["status"], "cancelled");
    assert_eq!(
        invoke(&["tool", "extract_frames", "--run-dir", root(&run)])
            .status
            .code(),
        Some(5)
    );
    assert_eq!(
        parse(invoke(&["resume", root(&run)]))["status"],
        "succeeded"
    );
}

#[test]
fn external_stop_waits_for_active_driver_and_reports_final_status() {
    use std::{
        process::Stdio,
        thread,
        time::{Duration, Instant},
    };
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    let mut driver = splat()
        .args([
            "run",
            root(&source),
            "--goal",
            "stop live driver",
            "--backend",
            "fake",
            "--driver",
            "fixed",
            "--run-dir",
            root(&run),
            "--json",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let running = fs::read(run.join("state.json"))
            .ok()
            .and_then(|data| serde_json::from_slice::<Value>(&data).ok())
            .is_some_and(|state| {
                state["jobs"]
                    .as_object()
                    .is_some_and(|jobs| jobs.values().any(|j| j["status"] == "running"))
            });
        if running {
            break;
        }
        if Instant::now() >= deadline || driver.try_wait().unwrap().is_some() {
            let _ = driver.kill();
            let output = driver.wait_with_output().unwrap();
            panic!(
                "Driver did not start a job: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(2));
    }
    // `vauss stop` never blocks on a busy run (4bea14fa): it asks the driver to stop and returns.
    let stopped = parse(invoke(&["stop", root(&run), "--wait-s", "5", "--json"]));
    assert_eq!(stopped["status"], "stop_requested");
    let deadline = Instant::now() + Duration::from_secs(10);
    while driver.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            let _ = driver.kill();
            driver.wait().unwrap();
            panic!("Stopped driver did not exit");
        }
        thread::sleep(Duration::from_millis(10));
    }
    let output = driver.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(5));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "cancelled");
    assert!(
        report["jobs"]
            .as_object()
            .unwrap()
            .values()
            .all(|j| j["status"] != "running")
    );
    assert_eq!(
        parse(invoke(&["status", root(&run)]))["status"],
        "cancelled"
    );
}

#[test]
fn mcp_uses_same_dispatch_and_keeps_stdout_as_protocol() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::Stdio;
    use std::time::Duration;
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    parse(invoke(&[
        "init",
        root(&source),
        "--goal",
        "test",
        "--backend",
        "fake",
        "--run-dir",
        root(&run),
    ]));
    let mut child = splat()
        .args(["serve", "--run-dir", root(&run)])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let output = child.stdout.take().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(output).lines() {
            if send.send(line).is_err() {
                break;
            }
        }
    });
    let read = || {
        serde_json::from_str::<Value>(
            &receive
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap(),
        )
        .unwrap()
    };
    writeln!(input,"{}",serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"cli-test","version":"1"}}})).unwrap();
    input.flush().unwrap();
    let initialized = read();
    assert_eq!(initialized["id"], 1);
    assert!(initialized["result"].is_object());
    writeln!(
        input,
        "{}",
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"})
    )
    .unwrap();
    writeln!(
        input,
        "{}",
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
    )
    .unwrap();
    input.flush().unwrap();
    // Every contract tool, in contract order: 17 plus run_default_pipeline (15abf2b7) and the
    // three edit tools (02f5348b).
    let listed: Vec<String> = read()["result"]["tools"].as_array().unwrap().iter()
        .map(|t| t["name"].as_str().unwrap().to_string()).collect();
    let contract: Vec<&str> = splat_contracts::tool_schemas().into_iter().map(|(name, _, _)| name).collect();
    assert_eq!(listed, contract);
    assert_eq!(listed.len(), 21);
    writeln!(input,"{}",serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"extract_frames","arguments":{"target_count":1}}})).unwrap();
    input.flush().unwrap();
    let invalid = read();
    assert_eq!(invalid["result"]["isError"], true);
    assert_eq!(
        invalid["result"]["structuredContent"]["error"]["code"],
        "invalid_request"
    );
    drop(input);
    let _ = child.kill();
    child.wait().unwrap();
}

#[test]
fn default_pipeline_leaves_agent_finish_and_goal_extras_open() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    parse(invoke(&["init", root(&source), "--goal", "a splat and mesh",
        "--backend", "fake", "--run-dir", root(&run)]));
    // No overrides must retain the CLI's explicit training budget.
    fs::write(run.join("fixed.json"), serde_json::json!({
        "steps":321, "target_count":40, "max_edge":800, "cap_max":100000,
        "resolution_divisor":1, "no_mesh":false, "no_lpips":true,
        "steps_explicit":true, "cap_max_explicit":true
    }).to_string()).unwrap();
    let result = parse(invoke(&["tool", "run_default_pipeline", "--run-dir", root(&run)]));
    assert_eq!(result["status"], "succeeded");
    assert_eq!(result["metrics"]["options"]["steps"], 321);
    assert_eq!(result["metrics"]["evaluated"], true);
    assert!(!run.join("agent/finish.json").exists());
    let state = parse(invoke(&["status", root(&run)]));
    assert!(!state["artifacts"].as_object().unwrap().values().any(|a| a["kind"] == "mesh"));
    let candidate = &result["metrics"]["candidate_id"];
    let mesh = parse(invoke(&["tool", "extract_mesh", "--run-dir", root(&run), "--input",
        &serde_json::json!({"candidate_id":candidate}).to_string()]));
    if mesh["status"] == "running" {
        let done = parse(invoke(&["tool", "job_status", "--run-dir", root(&run), "--input",
            &serde_json::json!({"job_id":mesh["job_id"],"wait_s":10}).to_string()]));
        assert_eq!(done["status"], "succeeded");
    }
    let finish = parse(invoke(&["tool", "finish", "--run-dir", root(&run), "--input",
        &serde_json::json!({"candidate_id":candidate,"status":"succeeded","explanation":"Reviewed output with requested mesh"}).to_string()]));
    assert_eq!(finish["status"], "succeeded");
}

/// Waits until a job of the run at `run` is running (the fake backend's jobs take ~150 ms).
#[cfg(unix)]
fn wait_for_running_job(run: &Path, follower: &mut std::process::Child) {
    use std::time::{Duration, Instant};
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let running = fs::read(run.join("state.json"))
            .ok()
            .and_then(|data| serde_json::from_slice::<Value>(&data).ok())
            .is_some_and(|state| {
                state["jobs"].as_object().is_some_and(|jobs| jobs.values().any(|j| j["status"] == "running"))
            });
        if running {
            return;
        }
        assert!(Instant::now() < deadline && follower.try_wait().unwrap().is_none(), "no job started");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Live descendants of `pid` (Linux /proc; empty elsewhere).
#[cfg(unix)]
fn descendants(pid: u32) -> Vec<u32> {
    let parents: Vec<(u32, u32)> = fs::read_dir("/proc").into_iter().flatten().flatten()
        .filter_map(|e| {
            let child: u32 = e.file_name().to_str()?.parse().ok()?;
            let stat = fs::read_to_string(e.path().join("stat")).ok()?;
            let ppid = stat.rsplit_once(')')?.1.split_whitespace().nth(1)?.parse().ok()?;
            Some((child, ppid))
        })
        .collect();
    let mut out = vec![];
    let mut frontier = vec![pid];
    while let Some(p) = frontier.pop() {
        for &(child, _) in parents.iter().filter(|(_, ppid)| *ppid == p) {
            out.push(child);
            frontier.push(child);
        }
    }
    out
}

#[cfg(unix)]
fn detached_run(source: &Path, run: &Path) -> Command {
    let mut command = splat();
    command
        .args(["run", root(source), "--goal", "detached", "--backend", "fake", "--driver", "fixed",
            "--steps", "200", "--run-dir", root(run), "--json"])
        .env("SPLAT_SUPERVISOR", "detached")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    command
}

/// An agent session that ends signals every process in its shell's tree: the run (a detached supervisor,
/// not a descendant of the launching process) finishes anyway, with no job left `job_owner_lost`.
#[cfg(unix)]
#[test]
fn detached_supervisor_outlives_its_launching_process() {
    use std::time::{Duration, Instant};
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    // Report and exit code come through the follower as before.
    let output = detached_run(&source, &run).output().unwrap();
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let state = parse(invoke(&["status", root(&run)]));
    assert_eq!(report["status"], state["status"]);
    assert_eq!(output.status.code(), Some(if report["status"] == "succeeded" { 0 } else { 5 }));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Supervisor: pid"));

    let run = tmp.path().join("run2");
    let mut follower = detached_run(&source, &run).spawn().unwrap();
    wait_for_running_job(&run, &mut follower);
    // As a session teardown does: SIGTERM to the launching process and every descendant of it.
    for pid in descendants(follower.id()).into_iter().chain([follower.id()]) {
        Command::new("kill").args(["-TERM", &pid.to_string()]).status().unwrap();
    }
    follower.wait().unwrap();
    // The supervisor finishes the run and records its exit code for a follower that is gone.
    let deadline = Instant::now() + Duration::from_secs(30);
    let exit = loop {
        let exit = fs::read_dir(run.join("supervisor-logs")).unwrap().flatten().map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "exit"));
        if let Some(exit) = exit {
            break exit;
        }
        assert!(Instant::now() < deadline, "the detached supervisor did not finish");
        std::thread::sleep(Duration::from_millis(50));
    };
    let state = parse(invoke(&["status", root(&run)]));
    assert!(matches!(state["status"].as_str(), Some("succeeded" | "failed")), "{state}");
    assert_eq!(fs::read_to_string(exit).unwrap(), if state["status"] == "succeeded" { "0" } else { "5" });
    let report: Value = serde_json::from_slice(&fs::read(run.join("report.json")).unwrap()).unwrap();
    let jobs = report["jobs"].as_object().unwrap();
    assert_eq!(jobs.len(), 4, "{report}");
    assert!(jobs.values().all(|j| j["status"] == "succeeded"), "{report}");
}

/// Ctrl-C in the launching terminal still stops the run (SIGINT is passed on to the supervisor).
#[cfg(unix)]
#[test]
fn detached_supervisor_stops_on_ctrl_c() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    let mut follower = detached_run(&source, &run).spawn().unwrap();
    wait_for_running_job(&run, &mut follower);
    assert!(Command::new("kill").args(["-INT", &follower.id().to_string()]).status().unwrap().success());
    let output = follower.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(130), "{}", String::from_utf8_lossy(&output.stderr));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["stop_reason"], "user_interrupt");
    assert_eq!(parse(invoke(&["status", root(&run)]))["status"], "cancelled");
}
/// A clean machine (no bundle, empty cache, no SPLAT_GITHUB_TOKEN) must see pinned release helpers with an
/// authenticated API fallback (ffmpeg-public, colmap, lpips) as downloadable: the anonymous URL is tried first.
#[test]
fn preflight_without_token_reports_release_helpers_downloadable() {
    let cache = tempfile::tempdir().unwrap();
    let output = Command::new(BIN)
        .args(["--json", "doctor", "--backend", "fake"])
        .env_remove("SPLAT_FFMPEG")
        .env_remove("FFMPEG_BIN")
        .env_remove("SPLAT_VOYAGER_FFMPEG_TREE")
        .env_remove("SPLAT_GITHUB_TOKEN")
        .env("SPLAT_CACHE_DIR", cache.path())
        .output()
        .unwrap();
    let line = String::from_utf8_lossy(&output.stdout).lines().last().unwrap_or_default().to_string();
    let report: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line}\n{}", String::from_utf8_lossy(&output.stderr)));
    let ffmpeg = &report["result"]["readiness"]["required"]["ffmpeg"];
    assert_eq!(ffmpeg["status"], "download_on_first_use", "{ffmpeg}");
    assert_eq!(ffmpeg["github_api_fallback"], true, "{ffmpeg}");
}

#[test]
fn input_contract_flags_and_api_schemas_are_explicit() {
    let help = invoke(&["run", "--help"]);
    assert!(help.status.success());
    let help = String::from_utf8_lossy(&help.stdout);
    for flag in ["--trust-poses", "--preserve-all", "--input-kind", "--pairs", "--camera-model"] { assert!(help.contains(flag), "{flag}: {help}"); }
    let schemas = parse(invoke(&["tools", "--json"]));
    for tool in ["extract_frames", "run_default_pipeline"] {
        for field in ["trust_poses", "preserve_all", "input_kind"] { assert!(schemas[tool]["properties"][field].is_object(), "{schemas}"); }
    }
    for field in ["pairs", "camera_model"] { assert!(schemas["run_default_pipeline"]["properties"][field].is_object()); }
}

#[test]
fn input_contract_pipeline_inherits_and_forwards_declared_controls() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("input.mp4");
    fs::write(&source, "fixture").unwrap();
    let run = tmp.path().join("run");
    parse(invoke(&["init", root(&source), "--goal", "a splat", "--backend", "fake", "--run-dir", root(&run)]));
    fs::write(run.join("fixed.json"), serde_json::json!({
        "steps":321,"target_count":40,"max_edge":800,"cap_max":100000,
        "resolution_divisor":1,"no_mesh":true,"no_lpips":true,"steps_explicit":true,
        "preserve_all":true,"input_kind":"video","pairs":"exhaustive","camera_model":"pinhole"
    }).to_string()).unwrap();
    let result = parse(invoke(&["tool", "run_default_pipeline", "--run-dir", root(&run)]));
    assert_eq!(result["status"], "succeeded", "{result}");
    assert_eq!(result["metrics"]["options"]["preserve_all"], true);
    let journal = fs::read_to_string(run.join("journal.jsonl")).unwrap();
    let rows: Vec<Value> = journal.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    // Normalized tool request events preserve exactly what the fixed pipeline sends to the backend.
    assert!(rows.iter().any(|row| row.to_string().contains("extract_frames") && row.to_string().contains("preserve_all")));
    assert!(rows.iter().any(|row| row.to_string().contains("reconstruct_sfm") && row.to_string().contains("exhaustive") && row.to_string().contains("pinhole")));
}

#[test]
fn input_contract_trusted_authority_rejects_conflicting_cli_selection() {
    let output = invoke(&["run", "missing-capture", "--trust-poses", "--target-count", "20"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used with"));
}
