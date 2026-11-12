//! Run state, journal, detached jobs and tool dispatch.
//! The `Backend` / `ShotTools` traits and the `RunCtx` surface below are the contract other
//! crates code against; this crate implements them.

use std::path::{Path, PathBuf};

use splat_contracts::*;

pub mod doctor;
pub mod edit;
mod fake;
pub mod jobs;
pub mod progress;
mod process;
mod registry;
pub mod fal;
mod hashcache;
pub mod remote;
pub mod report;
pub mod orientation;
pub mod resolution;
pub mod split;
pub mod supervise;
mod run;
mod tools;
pub use fake::FakeBackend;
pub use jobs::{JobRunner, handle_internal_command};
pub use registry::Registry;
pub use run::{JournalEvent, Run, RunConfig, RunState, STOP_REQUEST, SUPERVISOR_OWNER, atomic_json, supervisor_free, utc_now};
pub use tools::{JOB_TOOLS, MIN_SOLVED_FRACTION, dispatch, run_tool_job, sfm_coverage, sfm_coverage_defect};

/// What a backend or tool sees of a run. Implemented by `Run` in this crate.
pub trait RunCtx: Send + Sync {
    fn root(&self) -> &Path;
    fn run_id(&self) -> &str;
    fn source(&self) -> &Path;
    fn goal(&self) -> &str;
    /// Allocate a fresh artifact directory: -> (artifact_id, absolute dir).
    fn new_artifact_dir(&self, kind: ArtifactKind) -> anyhow::Result<(String, PathBuf)>;
    fn register(&self, artifact: Artifact) -> anyhow::Result<()>;
    fn artifact(&self, id: &str) -> anyhow::Result<Artifact>;
    fn path(&self, id: &str) -> anyhow::Result<PathBuf>;
    fn journal(&self, event: &str, data: serde_json::Value) -> anyhow::Result<()>;
    /// Start a detached, process-group-owned job that survives driver restarts -> job_id.
    fn start_job(&self, spec: JobSpec) -> anyhow::Result<String>;
}

/// A reconstruction engine: `spirula` (subprocess, GPL binary, never vendored), `brush`
/// (Apache, Rust, default for distribution). Long operations return Status::Running + job_id.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> serde_json::Value;
    fn inspect_input(&self, run: &dyn RunCtx, req: InspectInput) -> anyhow::Result<ToolResult>;
    fn extract_frames(&self, run: &dyn RunCtx, req: ExtractFrames) -> anyhow::Result<ToolResult>;
    fn mask_distractors(
        &self,
        run: &dyn RunCtx,
        req: MaskDistractors,
    ) -> anyhow::Result<ToolResult>;
    fn reconstruct_sfm(&self, run: &dyn RunCtx, req: ReconstructSfm) -> anyhow::Result<ToolResult>;
    fn define_split(&self, run: &dyn RunCtx, req: DefineSplit) -> anyhow::Result<ToolResult>;
    fn train_splats(&self, run: &dyn RunCtx, req: TrainSplats) -> anyhow::Result<ToolResult>;
    fn evaluate(&self, run: &dyn RunCtx, req: Evaluate) -> anyhow::Result<ToolResult>;
    fn render_views(&self, run: &dyn RunCtx, req: RenderViews) -> anyhow::Result<ToolResult>;
    fn extract_mesh(&self, run: &dyn RunCtx, req: ExtractMesh) -> anyhow::Result<ToolResult>;
    /// Object selection on a trained candidate (long job). Backends without it report unsupported.
    fn select_objects(&self, _run: &dyn RunCtx, _req: SelectObjects) -> anyhow::Result<ToolResult> {
        Ok(jobs::unsupported("select_objects", "this backend cannot select objects"))
    }
    /// Fill the hole a removal left (long job).
    fn inpaint_removed(&self, _run: &dyn RunCtx, _req: InpaintRemoved) -> anyhow::Result<ToolResult> {
        Ok(jobs::unsupported("inpaint_removed", "this backend cannot inpaint removed objects"))
    }
    /// Completion handler named by JobSpec.on_complete. Idempotent.
    fn on_job_complete(
        &self,
        run: &dyn RunCtx,
        handler: &str,
        job_dir: &Path,
        exit_code: i32,
    ) -> anyhow::Result<ToolResult>;
    /// Live progress of a running job, parsed on demand from its job_dir (stdout.log, metric
    /// files). Called by job_status; cheap and side-effect free apart from caching in job_dir.
    fn job_progress(&self, _run: &dyn RunCtx, _job_dir: &Path) -> serde_json::Value {
        serde_json::Value::Null
    }
}

/// Backend-independent shot tools (need only a PLY + SfM). Implemented by splat-shots.
pub trait ShotTools: Send + Sync {
    fn plan_shot(&self, run: &dyn RunCtx, req: PlanShot) -> anyhow::Result<ToolResult>;
    fn render_shot(&self, run: &dyn RunCtx, req: RenderShot) -> anyhow::Result<ToolResult>;
    fn generate_shot(&self, run: &dyn RunCtx, req: GenerateShot) -> anyhow::Result<ToolResult>;
    fn export_blender(&self, run: &dyn RunCtx, req: ExportBlender) -> anyhow::Result<ToolResult>;
    fn on_job_complete(
        &self,
        run: &dyn RunCtx,
        handler: &str,
        job_dir: &Path,
        exit_code: i32,
    ) -> anyhow::Result<ToolResult>;
}

/// Detach `command` from this process (own session/group) before spawning it.
pub fn detach_command(command: &mut std::process::Command) -> anyhow::Result<()> {
    process::detach(command, None, false)
}

/// Ask a process to exit (SIGTERM), or kill it (`force`). Missing processes are not an error.
pub fn terminate_pid(pid: u32, force: bool) -> anyhow::Result<()> {
    anyhow::ensure!(pid > 1 && pid != std::process::id(), "refusing to signal pid {pid}");
    #[cfg(unix)]
    {
        let signal = if force { libc::SIGKILL } else { libc::SIGTERM };
        if unsafe { libc::kill(pid as i32, signal) } < 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error.into());
            }
        }
    }
    #[cfg(windows)]
    {
        let _ = force;
        let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).status();
    }
    Ok(())
}
