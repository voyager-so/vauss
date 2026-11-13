//! Brush backend: the default, distributable engine (Rust + wgpu, Apache-2.0; NVIDIA, AMD,
//! Intel and Apple GPUs). Training and rendering use Brush's crates in-process inside
//! detached worker subprocesses (`splat __brush <train|render|sfm> <job.json>`); SfM runs
//! splat-sfm (COLMAP underneath); frames come from ffmpeg.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::{Backend, RunCtx};

mod align_eval;
mod capture;
mod calibration_health;
mod capture_depth;
mod capture_sfm;
#[cfg(test)]
mod trusted_tests;
mod depth_prior;
mod engine;
pub mod fixture;
mod glyphs;
mod inpaint;
mod media;
mod mesh;
mod model;
pub mod offpath;
pub mod render;
pub mod select;
mod select_worker;
mod sfm;
mod split;
mod train;
pub use train::{PseudoView, PseudoViews, TrainJob, WEB_PACKAGE};
mod util;
mod views;
mod warm_cap;

use util::*;

/// A lazy vocabulary-tree resolver (see [`BrushConfig::vocab_tree_fetch`]).
#[derive(Clone)]
pub struct VocabTreeFetch(pub std::sync::Arc<dyn Fn() -> anyhow::Result<PathBuf> + Send + Sync>);

impl std::fmt::Debug for VocabTreeFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VocabTreeFetch")
    }
}

#[derive(Debug, Clone)]
pub struct BrushConfig {
    /// Binary that understands `__brush` (normally the running `splat` binary).
    pub exe: PathBuf,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub colmap: Option<PathBuf>,
    /// Directory holding lpips_vgg.onnx / lpips_alex.onnx (splat-eval locates a default if None).
    pub models_dir: Option<PathBuf>,
    /// Serialise GPU work on shared dev hosts (exclusive file lock); None disables.
    pub gpu_lock: Option<PathBuf>,
    /// COLMAP SIFT vocabulary tree (asset colmap-vocab-tree-32k): video SfM matches sequentially
    /// with loop detection, filling in exhaustively only when that registers too little.
    /// None (and no [`Self::vocab_tree_fetch`]) keeps exhaustive matching.
    pub vocab_tree: Option<PathBuf>,
    /// Resolves the vocabulary tree on first use when `vocab_tree` is None (the CLI's asset cache,
    /// downloading it if needed), so only a video SfM pays for it.
    pub vocab_tree_fetch: Option<VocabTreeFetch>,
    /// Segmenter for select_objects (the Spirula SAM helper and its weights); None: unsupported.
    pub sam: Option<select::SamTool>,
    /// DA3-BASE weights for the SfM learned-pose rescue (SPLAT_DA3_WEIGHTS, asset da3-base); None
    /// skips the rescue with a warning.
    pub da3_weights: Option<PathBuf>,
    /// Directory holding aliked-n16rot.onnx and aliked-lightglue.onnx (COLMAP 3.13.0 release; ALIKED BSD-3,
    /// LightGlue Apache-2.0) for the SfM learned-feature fallback (SPLAT_ALIKED_DIR); None skips it.
    pub aliked_dir: Option<PathBuf>,
}

fn which(name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(&exe)).find(|p| p.is_file())
}

impl BrushConfig {
    /// The vocabulary tree a video SfM matches with: the configured one, else [`Self::vocab_tree_fetch`]'s
    /// (an error, e.g. offline with no cache, fails the solve instead of silently matching without loop
    /// detection). None when neither is set.
    pub fn video_vocab_tree(&self) -> anyhow::Result<Option<PathBuf>> {
        if let Some(path) = &self.vocab_tree {
            return Ok(Some(path.clone()));
        }
        self.vocab_tree_fetch.as_ref().map(|fetch| (fetch.0)()).transpose()
    }

    /// Defaults from the environment: SPLAT_BRUSH_EXE, SPLAT_FFMPEG, SPLAT_FFPROBE, SPLAT_COLMAP, SPLAT_MODELS_DIR,
    /// SPLAT_VOCAB_TREE, SPLAT_GPU_LOCK (else the dev root's, see [`splat_contracts::dev`]), else PATH.
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let gpu_lock = splat_contracts::dev::gpu_lock();
        Self {
            exe: env("SPLAT_BRUSH_EXE").or_else(|| std::env::current_exe().ok()).unwrap_or_else(|| "vauss".into()),
            ffmpeg: env("SPLAT_FFMPEG").or_else(|| which("ffmpeg")).unwrap_or_else(|| "ffmpeg".into()),
            ffprobe: env("SPLAT_FFPROBE").or_else(|| which("ffprobe")).unwrap_or_else(|| "ffprobe".into()),
            colmap: env("SPLAT_COLMAP").or_else(|| which("colmap")),
            models_dir: env("SPLAT_MODELS_DIR"),
            gpu_lock,
            vocab_tree: env("SPLAT_VOCAB_TREE"),
            vocab_tree_fetch: None,
            sam: None,
            da3_weights: env("SPLAT_DA3_WEIGHTS"),
            aliked_dir: env("SPLAT_ALIKED_DIR"),
        }
    }
}

pub struct BrushBackend {
    cfg: BrushConfig,
    /// Produces distractor masks (SAM 2.1 + Grounding DINO); its jobs complete through
    /// handlers prefixed `spirula:`. The masks artifact layout is shared, so SfM and training
    /// consume it natively.
    masker: Option<std::sync::Arc<dyn Backend>>,
}

impl BrushBackend {
    pub fn new(cfg: BrushConfig) -> Self {
        let _ = splat_eval::images::FFMPEG.set(cfg.ffmpeg.clone());
        Self { cfg, masker: None }
    }

    pub fn with_masker(cfg: BrushConfig, masker: std::sync::Arc<dyn Backend>) -> Self {
        Self { cfg, masker: Some(masker) }
    }
}

/// Exclusive lock on `path` held until the guard drops (no-op for None). Portable: uses
/// the OS file lock rather than the `flock` binary.
pub fn gpu_lock(path: Option<&Path>) -> anyhow::Result<Option<std::fs::File>> {
    let Some(path) = path else { return Ok(None) };
    let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(path).with_context(|| format!("opening {}", path.display()))?;
    fs4::fs_std::FileExt::lock_exclusive(&file)?;
    Ok(Some(file))
}

/// Submit `exe __brush <kind> <job.json>` as a detached core job for `artifact`.
pub(crate) fn start_worker(cfg: &BrushConfig, run: &dyn RunCtx, tool: &str, kind: &str, job: &Path, artifact: &Artifact) -> anyhow::Result<ToolResult> {
    let mut meta = std::collections::BTreeMap::new();
    meta.insert("artifact_id".into(), json!(artifact.id));
    let spec = JobSpec {
        tool: tool.into(),
        argv: vec![cfg.exe.to_string_lossy().into_owned(), "__brush".into(), kind.into(), job.to_string_lossy().into_owned()],
        env: Default::default(),
        cwd: Some(run.root().to_string_lossy().into_owned()),
        on_complete: format!("{kind}:{}", artifact.id),
        meta,
    };
    let job_id = run.start_job(spec)?;
    Ok(ToolResult { job_id: Some(job_id), artifacts: vec![artifact.clone()], ..result(tool, Status::Running) })
}

/// Entry point for the hidden worker subcommand: `args` are what follows `__brush`.
pub fn worker_main(args: Vec<String>) -> anyhow::Result<i32> {
    let [kind, job] = args.as_slice() else { anyhow::bail!("usage: __brush <train|render|align-eval|sfm|mesh|select|inpaint> <job.json>") };
    let job = Path::new(job);
    match kind.as_str() {
        "train" => train::run_job(read_json(job)?),
        "render" => views::run_job(read_json(job)?),
        "align-eval" => align_eval::run_job(read_json(job)?),
        "sfm" => sfm::run_job(read_json(job)?),
        "mesh" => mesh::run_job(read_json(job)?),
        "select" => select_worker::run_job(read_json(job)?),
        "inpaint" => inpaint::run_job(read_json(job)?),
        other => anyhow::bail!("unknown brush worker {other}"),
    }
}

fn job_log(job_dir: &Path) -> String {
    let mut log = tail_file(&job_dir.join("stderr.log"), 1200);
    if log.trim().is_empty() {
        log = tail_file(&job_dir.join("stdout.log"), 1200);
    }
    log
}

impl Backend for BrushBackend {
    fn name(&self) -> &'static str {
        "brush"
    }

    fn capabilities(&self) -> Value {
        json!({
            "backend": "brush", "inspect_input": true, "extract_frames": true,
            "mask_distractors": self.masker.as_ref().is_some_and(|m| m.capabilities()["mask_distractors"] == true),
            "sfm_inline_masks": self.cfg.sam.is_some(),
            "mask_uses": ["sfm_features", "training_ignore", "both"], "reconstruct_sfm": self.cfg.colmap.is_some(), "sfm_engine": "colmap",
            "define_split": "interval", "train_splats": true, "evaluate": true,
            "render_views": ["heldout", "training_sample", "orbit"], "render_depth": true,
            "extract_mesh": ["glb", "ply"], "select_objects": self.cfg.sam.is_some(), "inpaint_removed": true, "resume": "splats+lr_schedule", "mid_training_eval": true,
            "gpu_api": "wgpu (Vulkan/Metal/DX12)",
        })
    }

    fn inspect_input(&self, run: &dyn RunCtx, req: InspectInput) -> anyhow::Result<ToolResult> {
        guard("inspect_input", || media::inspect_input(&self.cfg, run, req))
    }

    fn extract_frames(&self, run: &dyn RunCtx, req: ExtractFrames) -> anyhow::Result<ToolResult> {
        guard("extract_frames", || media::extract_frames(&self.cfg, run, req))
    }

    fn mask_distractors(&self, run: &dyn RunCtx, req: MaskDistractors) -> anyhow::Result<ToolResult> {
        match &self.masker {
            Some(masker) => masker.mask_distractors(run, req),
            None => Ok(unsupported("mask_distractors", "no distractor segmenter configured (SAM 2.1 + Grounding DINO)")),
        }
    }

    fn reconstruct_sfm(&self, run: &dyn RunCtx, req: ReconstructSfm) -> anyhow::Result<ToolResult> {
        guard("reconstruct_sfm", || sfm::reconstruct_sfm(&self.cfg, run, req))
    }

    fn define_split(&self, run: &dyn RunCtx, req: DefineSplit) -> anyhow::Result<ToolResult> {
        guard("define_split", || split::define_split(run, req))
    }

    fn train_splats(&self, run: &dyn RunCtx, req: TrainSplats) -> anyhow::Result<ToolResult> {
        guard("train_splats", || train::train_splats(&self.cfg, run, req))
    }

    fn evaluate(&self, run: &dyn RunCtx, req: Evaluate) -> anyhow::Result<ToolResult> {
        guard("evaluate", || views::evaluate(&self.cfg, run, req))
    }

    fn render_views(&self, run: &dyn RunCtx, req: RenderViews) -> anyhow::Result<ToolResult> {
        guard("render_views", || views::render_views(&self.cfg, run, req))
    }

    fn extract_mesh(&self, run: &dyn RunCtx, req: ExtractMesh) -> anyhow::Result<ToolResult> {
        guard("extract_mesh", || mesh::extract_mesh(&self.cfg, run, req))
    }

    fn select_objects(&self, run: &dyn RunCtx, req: SelectObjects) -> anyhow::Result<ToolResult> {
        guard("select_objects", || select::select_objects(&self.cfg, run, req))
    }

    fn inpaint_removed(&self, run: &dyn RunCtx, req: InpaintRemoved) -> anyhow::Result<ToolResult> {
        guard("inpaint_removed", || inpaint::inpaint_removed(&self.cfg, run, req))
    }

    fn on_job_complete(&self, run: &dyn RunCtx, handler: &str, job_dir: &Path, exit_code: i32) -> anyhow::Result<ToolResult> {
        let (kind, artifact_id) = handler.split_once(':').with_context(|| format!("bad handler {handler}"))?;
        if kind == "spirula" {
            let masker = self.masker.as_ref().with_context(|| format!("no masker for handler {handler}"))?;
            return masker.on_job_complete(run, handler, job_dir, exit_code);
        }
        let log = job_log(job_dir);
        match kind {
            "train" => guard("train_splats", || train::complete(&self.cfg, run, artifact_id, exit_code, &log)),
            "sfm" => guard("reconstruct_sfm", || sfm::complete(run, artifact_id, exit_code, &log)),
            "mesh" => guard("extract_mesh", || mesh::complete(run, artifact_id, exit_code, &log)),
            "select" => guard("select_objects", || select::complete(run, artifact_id, exit_code, &log)),
            "inpaint" => guard("inpaint_removed", || inpaint::complete(run, artifact_id, exit_code, &log)),
            other => anyhow::bail!("unknown brush handler {other}"),
        }
    }

    fn job_progress(&self, _run: &dyn RunCtx, job_dir: &Path) -> Value {
        train::progress(job_dir)
    }
}
