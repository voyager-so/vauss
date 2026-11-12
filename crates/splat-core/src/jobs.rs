use crate::{
    Run, RunCtx, atomic_json, process,
    run::{lock_file, read_json},
    utc_now,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_contracts::*;
use std::{
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
use uuid::Uuid;

pub use crate::process::{ProcessIdentity, identity_matches, process_identity};

pub(crate) fn result(tool: &str, status: Status) -> ToolResult {
    ToolResult {
        schema_version: SCHEMA_VERSION,
        tool: tool.into(),
        status,
        job_id: None,
        artifacts: Vec::new(),
        metrics: Default::default(),
        images: Vec::new(),
        warnings: Vec::new(),
        error: None,
        wall_s: None,
    }
}
/// A job an identical request may attach to: still running, or succeeded. A failed or unsupported
/// job is never replayed; the retry starts a new one.
pub(crate) fn replayable(status: &Value) -> bool {
    status == "running" || status == "succeeded"
}
/// Status unsupported, code unsupported_capability.
pub(crate) fn unsupported(tool: &str, message: impl ToString) -> ToolResult {
    let mut r = failure(tool, "unsupported_capability", message);
    r.status = Status::Unsupported;
    r
}

pub(crate) fn failure(tool: &str, code: &str, message: impl ToString) -> ToolResult {
    let mut r = result(tool, Status::Failed);
    r.error = Some(ToolError {
        code: code.into(),
        message: message.to_string(),
        retryable: false,
        evidence: Vec::new(),
    });
    r
}

#[derive(Debug, Serialize, Deserialize)]
struct Completion {
    exit_code: i32,
    finished_at: String,
    wall_s: Option<f64>,
    error: Option<String>,
    #[serde(default)]
    error_code: Option<String>,
}

/// Whether a succeeded tool result's declared outputs are on disk: at least one artifact, every artifact
/// and image path present (a non-empty file or directory) and inside the run, and a recorded `size` /
/// `sha256` matching its file. A worker can abort in process teardown after writing a complete result
/// (evaluate, exit 134 "corrupted double-linked list", job_8f0b376b188c4e0a); anything less stays failed.
fn verified_outputs(root: &Path, result: &ToolResult) -> bool {
    let present = |rel: &str| -> Option<PathBuf> {
        let rel = Path::new(rel);
        if rel.as_os_str().is_empty() || rel.components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
            return None;
        }
        let path = root.join(rel);
        let meta = fs::metadata(&path).ok()?;
        let non_empty = if meta.is_dir() { fs::read_dir(&path).ok()?.next().is_some() } else { meta.len() > 0 };
        non_empty.then_some(path)
    };
    !result.artifacts.is_empty()
        && result.artifacts.iter().all(|a| {
            let Some(path) = present(&a.path) else { return false };
            let size = a.meta.get("size").or_else(|| a.meta.get("bytes"));
            let sha = a.meta.get("sha256");
            if size.is_none() && sha.is_none() {
                return true;
            }
            path.is_file()
                && size.is_none_or(|v| v.as_u64().is_some_and(|n| fs::metadata(&path).is_ok_and(|m| m.len() == n)))
                && sha.is_none_or(|v| v.as_str().is_some_and(|h| {
                    crate::hashcache::sha256(&path).is_ok_and(|got| got.eq_ignore_ascii_case(h))
                }))
        })
        && result.images.iter().all(|i| present(i).is_some())
}

pub struct JobRunner<'a> {
    run: &'a Run,
}
impl<'a> JobRunner<'a> {
    pub(crate) fn new(run: &'a Run) -> Self {
        Self { run }
    }
    fn directory(&self, id: &str) -> Result<PathBuf> {
        ensure!(
            id.starts_with("job_")
                && id.len() == 20
                && id[4..].bytes().all(|b| b.is_ascii_hexdigit()),
            "Invalid job ID"
        );
        let dir = self.run.root().join("jobs").join(id);
        ensure!(dir.join("request.json").is_file(), "Unknown job: {id}");
        Ok(dir)
    }
    pub fn start(&self, spec: JobSpec) -> Result<String> {
        ensure!(
            !spec.argv.is_empty() && !spec.argv[0].is_empty(),
            "Job argv must include an executable"
        );
        ensure!(
            !spec.on_complete.is_empty(),
            "Job completion handler must not be empty"
        );
        self.run.with_lock(|| {
            let operation_id = self.run.operation_id();
            if let Some(ref op) = operation_id {
                for (id, job) in self.run.state()?.jobs {
                    if job["operation_id"].as_str() == Some(op) && replayable(&job["status"]) {
                        return Ok(id);
                    }
                }
            }
            let id = format!("job_{}", &Uuid::new_v4().simple().to_string()[..16]);
            let dir = self.run.root().join("jobs").join(&id);
            fs::create_dir(&dir)?;
            atomic_json(dir.join("request.json"), &spec)?;
            self.run.journal(
                "job_planned",
                json!({"job_id":id,"tool":spec.tool,"operation_id":operation_id}),
            )?;
            let spawn = (|| -> Result<_> {
                let ownership = lock_file(&dir.join("worker.lock"), true)?;
                let log = OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(dir.join("runner.log"))?;
                let mut command = Command::new(&self.run.worker_exe);
                command
                    .arg("__splat_worker")
                    .arg(&dir)
                    .arg(ownership_arg(&ownership));
                command
                    .stdin(Stdio::null())
                    .stdout(log.try_clone()?)
                    .stderr(log);
                process::detach(&mut command, Some(&ownership), true)?;
                let child = command
                    .spawn()
                    .with_context(|| format!("Spawn worker {}", self.run.worker_exe.display()))?;
                let pid = child.id();
                if let Ok(identity) = process_identity(pid) {
                    atomic_json(dir.join("launch.json"), &identity)?;
                }
                // Close only; LOCK_UN would unlock the child's shared open file description.
                drop(ownership);
                // Reap while this launcher exists; a detached worker remains independent if it exits.
                thread::spawn(move || {
                    let mut child = child;
                    let _ = child.wait();
                });
                Ok(pid)
            })();
            match spawn {
                Ok(pid) => self
                    .run
                    .journal("job_started", json!({"job_id":id,"worker_pid":pid}))?,
                Err(error) => atomic_json(
                    dir.join("completion.json"),
                    &Completion {
                        exit_code: 127,
                        finished_at: utc_now(),
                        wall_s: None,
                        error: Some(format!("{error:#}")),
                        error_code: Some("worker_spawn_failed".into()),
                    },
                )?,
            }
            Ok(id)
        })
    }

    fn artifacts(&self, dir: &Path) -> Result<Vec<Artifact>> {
        let state = self.run.state()?;
        let id = dir.file_name().unwrap().to_string_lossy();
        let op = state
            .jobs
            .get(id.as_ref())
            .and_then(|j| j["operation_id"].as_str());
        let mut artifacts: Vec<Artifact> = op
            .and_then(|op| state.operations.get(op))
            .and_then(|o| o["result"]["artifacts"].as_array())
            .map(|a| serde_json::from_value(json!(a)))
            .transpose()?
            .unwrap_or_default();
        let spec: JobSpec = read_json(dir.join("request.json"))?;
        if let Some(a) = spec.meta.get("artifact") {
            let a: Artifact = serde_json::from_value(a.clone())?;
            if !artifacts.iter().any(|v| v.id == a.id) {
                artifacts.push(a);
            }
        }
        if let Some(a) = spec
            .meta
            .get("artifact_id")
            .and_then(Value::as_str)
            .and_then(|id| state.artifacts.get(id))
        {
            if !artifacts.iter().any(|v| v.id == a.id) {
                artifacts.push(a.clone());
            }
        }
        Ok(artifacts)
    }

    fn publish(&self, dir: &Path, mut result: ToolResult) -> Result<ToolResult> {
        for artifact in self.artifacts(dir)? {
            if !result.artifacts.iter().any(|a| a.id == artifact.id) {
                result.artifacts.push(artifact);
            }
        }
        for artifact in &result.artifacts {
            self.run.register(artifact.clone())?;
        }
        result.job_id = Some(dir.file_name().unwrap().to_string_lossy().into_owned());
        atomic_json(dir.join("result.json"), &result)?;
        self.run.with_lock(|| {
            let id = result.job_id.as_ref().unwrap();
            if self.run.state()?.jobs.get(id).map(|j| &j["result"])
                != Some(&serde_json::to_value(&result)?)
            {
                self.run
                    .journal("job_completed", json!({"job_id": id, "result": result}))?;
                if let Some(op) = self.run.state()?.jobs[id]["operation_id"].as_str() {
                    crate::progress::complete(self.run, op, result.status)?;
                }
            }
            Ok(())
        })?;
        Ok(result)
    }

    fn finalize(&self, dir: &Path) -> Result<ToolResult> {
        let _lock = lock_file(&dir.join("result.lock"), true)?;
        if dir.join("result.json").is_file() {
            return self.publish(dir, read_json(dir.join("result.json"))?);
        }
        let spec: JobSpec = read_json(dir.join("request.json"))?;
        let completion: Completion = read_json(dir.join("completion.json"))?;
        // Opening a run before installing its registry must not poison a completed job.
        let callback: Result<ToolResult> = if let Some(code) = &completion.error_code {
            Ok(failure(
                &spec.tool,
                code,
                format!(
                    "Job exited with code {}: {}",
                    completion.exit_code,
                    completion.error.as_deref().unwrap_or("job failed")
                ),
            ))
        } else if spec.on_complete.starts_with("shots:") || SHOT_TOOLS.contains(&spec.tool.as_str())
        {
            let Some(shots) = &self.run.registry.shots else {
                return Ok(self.awaiting_handler(&spec, dir));
            };
            shots.on_job_complete(self.run, &spec.on_complete, dir, completion.exit_code)
        } else if spec.on_complete == "core:tool_result" {
            let path = spec.meta.get("result_path").and_then(Value::as_str).context("Job has no result_path")?;
            match read_json::<ToolResult>(Path::new(path)) {
                Ok(r) => Ok(r),
                Err(_) => Ok(failure(&spec.tool, "job_exit", format!("Tool job exited with code {} without a result; see runner.log", completion.exit_code))),
            }
        } else if spec.on_complete == "core:exit" {
            Ok(result(
                &spec.tool,
                if completion.exit_code == 0 {
                    Status::Succeeded
                } else {
                    Status::Failed
                },
            ))
        } else {
            let Ok(backend) = self.run.registry.backend(&self.run.config().backend) else {
                return Ok(self.awaiting_handler(&spec, dir));
            };
            backend.on_job_complete(self.run, &spec.on_complete, dir, completion.exit_code)
        };
        let mut result = match callback {
            Ok(r) if r.status != Status::Running => r,
            Ok(_) => failure(
                &spec.tool,
                "completion_failed",
                "Completion callback returned running",
            ),
            Err(e) => failure(&spec.tool, "completion_failed", format!("{e:#}")),
        };
        let accepted = completion.exit_code != 0
            && spec.on_complete == "core:tool_result"
            && result.status == Status::Succeeded
            && verified_outputs(self.run.root(), &result);
        if accepted {
            let note = format!("accepted complete result despite exit {}", completion.exit_code);
            eprintln!("[job] {} {}: {note}", dir.file_name().unwrap().to_string_lossy(), spec.tool);
            result.warnings.push(format!("Backend {note} (its outputs were verified; teardown crashed after they were written)"));
        } else if completion.exit_code != 0 {
            if result.status == Status::Succeeded {
                result.status = Status::Failed;
            }
            if result.error.is_none() {
                result.error = Some(ToolError {
                    code: "job_exit".into(),
                    message: format!(
                        "Backend exited with code {}{}",
                        completion.exit_code,
                        completion
                            .error
                            .as_ref()
                            .map(|e| format!(": {e}"))
                            .unwrap_or_default()
                    ),
                    retryable: false,
                    evidence: Vec::new(),
                });
            }
        }
        result.wall_s = completion.wall_s.or(result.wall_s);
        self.publish(dir, result)
    }
    fn awaiting_handler(&self, spec: &JobSpec, dir: &Path) -> ToolResult {
        let mut r = result(&spec.tool, Status::Running);
        r.job_id = Some(dir.file_name().unwrap().to_string_lossy().into_owned());
        r.warnings
            .push("Process exited; completion is pending the backend registry".into());
        r
    }

    pub fn status(&self, id: &str, wait_s: f64) -> Result<ToolResult> {
        ensure!(
            wait_s.is_finite() && (0.0..=600.0).contains(&wait_s),
            "wait_s must be between 0 and 600"
        );
        let dir = self.directory(id)?;
        let deadline = Instant::now() + Duration::from_secs_f64(wait_s);
        loop {
            if dir.join("result.json").exists() || dir.join("completion.json").exists() {
                return self.finalize(&dir);
            }
            match lock_file(&dir.join("worker.lock"), false) {
                Ok(_lock) => {
                    let _launch = lock_file(&dir.join("launch.lock"), true)?;
                    if dir.join("completion.json").exists() {
                        continue;
                    }
                    let mut alive = false;
                    for name in ["process.json", "worker.json", "launch.json"] {
                        if dir.join(name).exists() && identity_matches(&read_json(dir.join(name))?)?
                        {
                            alive = true;
                            break;
                        }
                    }
                    #[cfg(windows)]
                    {
                        alive |= process::job_has_processes(id)?;
                    }
                    if !alive {
                        atomic_json(dir.join("completion.json"), &Completion { exit_code:-255, finished_at:utc_now(), wall_s:None,error:Some("Job owner exited without a completion marker; no automatic relaunch".into()),error_code:Some("job_owner_lost".into()) })?;
                        continue;
                    }
                }
                Err(e)
                    if e.downcast_ref::<std::io::Error>()
                        .is_some_and(|e| e.kind() == std::io::ErrorKind::WouldBlock) => {}
                Err(e) => return Err(e),
            }
            if Instant::now() >= deadline {
                let spec: JobSpec = read_json(dir.join("request.json"))?;
                let mut r = result(&spec.tool, Status::Running);
                r.job_id = Some(id.into());
                r.artifacts = self.artifacts(&dir)?;
                if dir.join("stop.json").exists() {
                    r.warnings
                        .push("Stop requested; waiting for checkpoint and process exit".into());
                }
                if let Ok(backend) = self.run.registry.backend(&self.run.config().backend) {
                    match backend.job_progress(self.run, &dir) {
                        Value::Object(metrics) => r.metrics.extend(metrics),
                        Value::Null => {}
                        value => {
                            r.metrics.insert("progress".into(), value);
                        }
                    }
                }
                return Ok(r);
            }
            thread::sleep(
                Duration::from_millis(50).min(deadline.saturating_duration_since(Instant::now())),
            );
        }
    }
    pub fn wait(&self, id: &str, wait_s: f64) -> Result<ToolResult> {
        self.status(id, wait_s)
    }
    pub fn stop(&self, id: &str, wait_s: f64) -> Result<ToolResult> {
        ensure!(
            wait_s.is_finite() && (0.0..=600.0).contains(&wait_s),
            "wait_s must be between 0 and 600"
        );
        let mut r = self.status(id, 0.0)?;
        if r.status != Status::Running {
            return Ok(r);
        }
        let dir = self.directory(id)?;
        let deadline = Instant::now() + Duration::from_secs_f64(wait_s.clamp(0.0, 5.0));
        while !dir.join("process.json").exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        if !dir.join("process.json").exists() {
            r.warnings
                .push("Process identity unavailable; no signal sent".into());
            return Ok(r);
        }
        {
            let _lock = lock_file(&dir.join("stop.lock"), true)?;
            if !dir.join("stop.json").exists() {
                let identity: ProcessIdentity = read_json(dir.join("process.json"))?;
                if !identity_matches(&identity)? {
                    r.warnings
                        .push("Process identity changed; no signal sent".into());
                    return Ok(r);
                }
                ensure!(
                    identity.pgid == identity.pid && identity.pid > 1,
                    "Refusing to signal a process group not owned by the job"
                );
                // A repeated interrupt can abort checkpoint saving, so persist before signaling.
                atomic_json(
                    dir.join("stop.json"),
                    &json!({"requested_at":utc_now(),"identity":identity}),
                )?;
                process::interrupt(&identity)?;
                self.run
                    .journal("job_stop_requested", json!({"job_id":id}))?;
            }
        }
        self.status(id, wait_s)
    }
    pub fn reconcile(&self) -> Result<Vec<ToolResult>> {
        self.run
            .state()?
            .jobs
            .keys()
            .map(|id| self.status(id, 0.0))
            .collect()
    }
}

#[cfg(unix)]
fn ownership_arg(file: &File) -> String {
    use std::os::fd::AsRawFd;
    file.as_raw_fd().to_string()
}
#[cfg(windows)]
fn ownership_arg(_file: &File) -> String {
    "0".into()
}

fn worker(dir: &Path, ownership: i32) -> Result<()> {
    #[cfg(windows)]
    process::protect_supervisor_console()?;
    #[cfg(unix)]
    let ownership = {
        use std::os::fd::FromRawFd;
        ensure!(ownership > 2, "Invalid ownership descriptor");
        unsafe { File::from_raw_fd(ownership) }
    };
    #[cfg(windows)]
    let ownership = {
        let _ = ownership;
        lock_file(&dir.join("worker.lock"), true)?
    };
    if dir.join("completion.json").exists() {
        return Ok(());
    }
    // The worker only waits for the backend and records how it ended. A signal meant for its launching
    // process tree (an agent session tearing down its shell) must not end it first: the backend gets the
    // same signal and exits on its own, and a worker gone before it left `job_owner_lost` in place of that
    // exit. A handler, unlike SIG_IGN, is reset for the exec'd backend.
    #[cfg(unix)]
    {
        extern "C" fn keep_waiting(_: libc::c_int) {}
        for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            // SAFETY: an empty handler is async-signal-safe.
            unsafe { libc::signal(signal, keep_waiting as *const () as libc::sighandler_t) };
        }
    }
    atomic_json(
        dir.join("worker.json"),
        &process_identity(std::process::id())?,
    )?;
    let started = Instant::now();
    let launch = (|| -> Result<i32> {
        let stdout = OpenOptions::new()
            .append(true)
            .create(true)
            .open(dir.join("stdout.log"))?;
        let stderr = OpenOptions::new()
            .append(true)
            .create(true)
            .open(dir.join("stderr.log"))?;
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg("__splat_exec")
            .arg(dir)
            .arg(ownership_arg(&ownership))
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(stderr);
        process::detach(&mut command, Some(&ownership), false)?;
        let status = command.spawn().context("Spawn backend bootstrap")?.wait()?;
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            Ok(status
                .code()
                .unwrap_or_else(|| -status.signal().unwrap_or(255)))
        }
        #[cfg(windows)]
        {
            while process::job_has_processes(&dir.file_name().unwrap().to_string_lossy())? {
                thread::sleep(Duration::from_millis(50));
            }
            Ok(status.code().unwrap_or(-255))
        }
    })();
    let (exit_code, error, error_code) = match launch {
        Ok(code) => {
            let exec_error: Option<Value> = if dir.join("exec_error.json").is_file() {
                Some(read_json(dir.join("exec_error.json"))?)
            } else {
                None
            };
            (
                code,
                exec_error
                    .as_ref()
                    .and_then(|v| v["message"].as_str())
                    .map(str::to_owned),
                exec_error
                    .as_ref()
                    .and_then(|v| v["code"].as_str())
                    .map(str::to_owned),
            )
        }
        Err(e) => (
            127,
            Some(format!("{e:#}")),
            Some("backend_bootstrap_failed".into()),
        ),
    };
    atomic_json(
        dir.join("completion.json"),
        &Completion {
            exit_code,
            finished_at: utc_now(),
            wall_s: Some(started.elapsed().as_secs_f64()),
            error,
            error_code,
        },
    )?;
    drop(ownership);
    Ok(())
}

fn exec_backend(dir: &Path, _ownership: i32) -> Result<i32> {
    let launch = lock_file(&dir.join("launch.lock"), true)?;
    // An orphaned, delayed bootstrap may never launch work after reconciliation failed it.
    if dir.join("completion.json").exists() {
        return Ok(127);
    }
    #[cfg(windows)]
    process::protect_supervisor_console()?;
    let spec: JobSpec = read_json(dir.join("request.json"))?;
    ensure!(!spec.argv.is_empty(), "Empty argv");
    #[cfg(windows)]
    let _job = process::job_object(&dir.file_name().unwrap().to_string_lossy())?;
    atomic_json(
        dir.join("process.json"),
        &process_identity(std::process::id())?,
    )?;
    drop(launch);
    // `vauss run --cloud`: GPU jobs run on fal; a fal failure fails the job (no local fallback).
    if crate::remote::mode().is_some() && crate::remote::wants(&spec) {
        return crate::remote::run(dir, &spec);
    }
    let mut command = Command::new(&spec.argv[0]);
    command.args(&spec.argv[1..]).envs(&spec.env);
    if let Some(cwd) = &spec.cwd {
        command.current_dir(cwd);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(command.exec()).with_context(|| format!("Execute backend {}", spec.argv[0]))
    }
    #[cfg(windows)]
    {
        Ok(command
            .status()
            .with_context(|| format!("Execute backend {}", spec.argv[0]))?
            .code()
            .unwrap_or(-255))
    }
}

/// Call before clap/async runtime initialization in every binary that launches jobs.
/// Worker processes write only to job logs; stdout remains clean for MCP and JSON CLI callers.
pub fn handle_internal_command() -> Option<Result<i32>> {
    let args: Vec<_> = std::env::args_os().collect();
    let action = args.get(1)?.to_str()?;
    match action {
        "__splat_fake" => Some((|| {
            let milliseconds: u64 = args
                .get(2)
                .and_then(|a| a.to_str())
                .unwrap_or("150")
                .parse()?;
            println!("synthetic fixture job");
            thread::sleep(Duration::from_millis(milliseconds));
            Ok(0)
        })()),
        "__splat_worker" | "__splat_exec" => Some((|| {
            let dir = Path::new(args.get(2).context("Missing job directory")?);
            let fd: i32 = args
                .get(3)
                .and_then(|a| a.to_str())
                .context("Missing ownership descriptor")?
                .parse()?;
            if action == "__splat_worker" {
                worker(dir, fd)?;
                Ok(0)
            } else {
                match exec_backend(dir, fd) {
                    Ok(code) => Ok(code),
                    Err(error) => {
                        atomic_json(
                            dir.join("exec_error.json"),
                            &json!({"code":"backend_spawn_failed","message":format!("{error:#}")}),
                        )?;
                        Err(error)
                    }
                }
            }
        })()),
        crate::supervise::LAUNCH_COMMAND => Some((|| {
            let prefix = args.get(2).context("Missing supervisor log prefix")?;
            crate::supervise::launch(Path::new(prefix), &args[3..])
        })()),
        #[cfg(windows)]
        "__splat_interrupt" => Some((|| {
            process::console_interrupt(
                args.get(2)
                    .and_then(|a| a.to_str())
                    .context("Missing pid")?
                    .parse()?,
            )?;
            Ok(0)
        })()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_outputs_accept_a_teardown_crash_and_missing_ones_do_not() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("artifacts/eval_1/comparisons")).unwrap();
        fs::write(root.join("artifacts/eval_1/metrics.json"), "{}").unwrap();
        fs::write(root.join("artifacts/eval_1/comparisons/00.jpg"), "jpg").unwrap();
        fs::write(root.join("artifacts/eval_1/scores.json"), "1234").unwrap();
        let artifact = |path: &str, size: Option<u64>| Artifact {
            id: "eval_1".into(),
            kind: ArtifactKind::Eval,
            path: path.into(),
            meta: size.map(|n| [("size".to_string(), json!(n))].into()).unwrap_or_default(),
        };
        let mut ok = result("evaluate", Status::Succeeded);
        ok.artifacts = vec![artifact("artifacts/eval_1", None), artifact("artifacts/eval_1/scores.json", Some(4))];
        ok.images = vec!["artifacts/eval_1/comparisons/00.jpg".into()];
        // (a) exit 134 after a succeeded result whose outputs are all there: accepted.
        assert!(verified_outputs(root, &ok));

        // (b) a missing image, a missing artifact, a size mismatch, an escaping path or no artifacts: failed.
        let mut missing_image = ok.clone();
        missing_image.images.push("artifacts/eval_1/comparisons/01.jpg".into());
        assert!(!verified_outputs(root, &missing_image));
        let mut missing_artifact = ok.clone();
        missing_artifact.artifacts.push(artifact("artifacts/eval_2", None));
        assert!(!verified_outputs(root, &missing_artifact));
        let mut wrong_size = ok.clone();
        wrong_size.artifacts[1] = artifact("artifacts/eval_1/scores.json", Some(5));
        assert!(!verified_outputs(root, &wrong_size));
        let mut escaping = ok.clone();
        escaping.artifacts[0] = artifact("../outside", None);
        assert!(!verified_outputs(root, &escaping));
        let mut nothing = ok.clone();
        nothing.artifacts.clear();
        assert!(!verified_outputs(root, &nothing));
    }
}
