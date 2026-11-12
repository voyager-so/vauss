//! Training: `train_splats` submits a detached worker (`splat __brush train job.json`) that
//! runs Brush's trainer in-process on the frozen train views, logs metrics to
//! training.jsonl, evaluates the held-out views mid-run, and checkpoints to PLY.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Context;
use brush_dataset::scene::{Scene, SceneView};
use brush_dataset::scene_loader::SceneLoader;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::config::TrainConfig;
use brush_train::eval::eval_stats;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use brush_train::{RandomSplatsConfig, create_random_splats, to_init_splats};
use burn::module::Module;
use burn::tensor::Device;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;

use crate::util::*;
use crate::views::stem;
use crate::{BrushConfig, engine, split, start_worker};

/// Written next to an SfM dataset's `masks/` when they came from `mask_distractors`.
pub const DETECTOR_MASKS_MARKER: &str = "masks_source.json";

/// Unknown keys are an error, not a silent no-op: a research knob sent to a binary that
/// lacks it must fail the job instead of training with defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainJob {
    pub dataset: PathBuf,
    pub out: PathBuf,
    /// First view only: publish the PLY without checkpoint or evaluation work.
    #[serde(default)]
    pub preview: bool,
    /// Video capture: camera-motion-blur detection stays off (see `reconstruct_job`).
    #[serde(default)]
    pub video: bool,
    pub train_names: Vec<String>,
    pub heldout_names: Vec<String>,
    pub total_steps: u32,
    /// MAX schedule a blurred photo capture moves to when blur mode switches on (unstretched
    /// knobs). Blurred photos, product path: 20k vs 45k sharp-GT 22.79 vs 21.43 dB, LPIPS-vgg 0.264 vs
    /// 0.321, final 353 vs 689 s: on 45k the fixed blur budget covers ~35% of the steps.
    #[serde(default)]
    pub blur_total_steps: Option<u32>,
    /// Resume: steps already done by `init_ply`'s run (LR schedule continues from here).
    pub start_step: u32,
    pub init_ply: Option<PathBuf>,
    /// `init_ply` is another candidate's model (warm start, e.g. the interactive refine), not this
    /// job's own checkpoint.
    #[serde(default)]
    pub warm_start: bool,
    /// `init_ply` is the fork another job is still training toward (fal /data, see `fork_ply`):
    /// wait up to this long for it, failing early on its `.none` marker.
    #[serde(default)]
    pub init_wait_s: Option<f64>,
    /// Cloud MAX: two thirds into training (or at its early stop), a model above `fork_min_splats` is also written
    /// here (fal /data) for the interactive refine, which was dispatched beside this job and starts
    /// from it instead of waiting for the final model. Otherwise `<fork_ply>.none` is written.
    #[serde(default)]
    pub fork_ply: Option<PathBuf>,
    #[serde(default)]
    pub fork_min_splats: u32,
    /// Treat a supplied PLY as a fresh initialization, including the degree-0 SH ramp.
    #[serde(default)]
    pub fresh_init_ply: bool,
    /// Reinitialize supplied PLY geometry with the same KNN/default attributes as SfM.
    #[serde(default)]
    pub plain_init_ply: bool,
    pub max_splats: u32,
    /// Enforce an explicit request cap on loaded splats as well as subsequent growth.
    #[serde(default)]
    pub cap_max_explicit: bool,
    /// Cloud MAX tier: with `max_splats` 0, unbounded outdoor scenes may grow
    /// to the full automatic budget (up to 8M, within device memory) instead
    /// of the local default's wall-time-bounded one.
    #[serde(default)]
    pub max_budget: bool,
    /// Stop converged MAX fitting early, then run the original complete QAT tail.
    #[serde(default)]
    pub early_stop_db: Option<f64>,
    /// Optional research schedule overrides; absent fields preserve the trainer defaults.
    #[serde(default)]
    pub growth_start_iter: Option<u32>,
    #[serde(default)]
    pub growth_stop_iter: Option<u32>,
    #[serde(default)]
    pub refine_every: Option<u32>,
    pub max_resolution: u32,
    pub eval_every: u32,
    pub checkpoint_every: u32,
    /// Seconds of training after which the worker stops early and finishes normally (final PLY,
    /// held-out eval) instead of being killed by its job's wall limit.
    #[serde(default)]
    pub time_budget_s: Option<f64>,
    pub sh_degree: u32,
    #[serde(default)]
    pub sh_degree_interval: Option<u32>,
    /// Optional research loss override (D-SSIM weight); absent keeps the trainer default.
    #[serde(default)]
    pub ssim_weight: Option<f32>,
    /// Train 2DGS surfels (set when the run's deliverables include a mesh).
    #[serde(default)]
    pub surfel: bool,
    /// Hosted MAX tier: neural appearance at 2.5M splats. Ignored for
    /// surfels, which keep SH.
    #[serde(default)]
    pub neural_max: bool,
    /// Research/ablation knob: TrainConfig fields (by serde name) to override, e.g.
    /// `{"mean_noise_weight": 0}`. Not exposed on any CLI.
    #[serde(default)]
    pub config_overrides: Option<serde_json::Map<String, Value>>,
    pub seed: u64,
    pub gpu_lock: Option<PathBuf>,
    /// `--cloud` jobs: the worker leaves held-out LPIPS to the evaluate job that follows on fal
    /// (it scores these renders, still staged there) and writes only the preview JPEGs, so neither
    /// the worker's exit nor the user's CPU waits on LPIPS.
    #[serde(default)]
    pub defer_lpips: bool,
    /// A `vauss run --cloud` job. Its full model above the interactive budget gets a capped refine
    /// whose package replaces this one's, so the worker skips this one's .RAD stream (`candidate_web`).
    #[serde(default)]
    pub cloud: bool,
    /// Generated views trained in beside the real ones (`vauss fill`).
    #[serde(default)]
    pub pseudo: Option<PseudoViews>,
}

/// Pseudo views for a warm-started fill-in fine-tune (`vauss fill`): fixer-repaired renders of
/// novel cameras, each an RGBA PNG whose alpha is the per-pixel confidence, listed in
/// `<dir>/views.json` (see [`PseudoView`]). A step trains on one with probability `frac`, with the
/// confidence-masked loss of [`brush_train::train::PseudoStep`] scaled by `weight` x the view's own
/// weight x a sine ramp-in over the first `ramp` of the steps; the means move at `pos_lr` x the
/// scheduled LR. From `stop` x the steps on (if set), only real views train (a real-only finish).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PseudoViews {
    pub dir: PathBuf,
    pub frac: f32,
    pub weight: f32,
    pub ramp: f32,
    pub pos_lr: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<f32>,
}

/// One row of a pseudo-view dir's `views.json` (`{"views": [...]}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoView {
    /// Image file in the dir: RGB = the repaired view, alpha = confidence.
    pub image: String,
    /// Camera-to-world, row-major 4x4, OpenCV axes (+z forward, +y down), SfM units.
    pub c2w: [[f32; 4]; 4],
    /// Pinhole intrinsics in pixels of the image.
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    /// Per-view factor on the job's weight.
    pub weight: f32,
    /// Mean alpha (confidence) of the image.
    pub conf_mean: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub step: u32,
    pub total_steps: u32,
    pub splats: u32,
    pub ply: String,
}

pub fn train_splats(cfg: &BrushConfig, run: &dyn RunCtx, mut req: TrainSplats) -> anyhow::Result<ToolResult> {
    let manifest = split::resolve(run, &req.split_id, &req.sfm_id)?;
    // Cloud and explicit --max use the MAX recipe; explicit steps/caps override its budget.
    // Mesh (surfel) goals keep the default recipe.
    let surfel = wants_mesh(run);
    let preset = if req.max_quality {
        Some(&splat_core::remote::BUDGET_PRESET)
    } else {
        splat_core::remote::preset()
    }.filter(|_| !surfel && !req.preview);
    // 0 = the trainer sizes the budget to the scene (logged as "Splat budget N").
    let max_splats = if req.cap_max_explicit { req.cap_max } else { preset.map_or(req.cap_max, |p| p.max_splats) };
    let max_budget = !req.cap_max_explicit && preset.is_some_and(|p| p.max_budget);
    let requested = (req.total_steps, req.cap_max);
    if req.resume_candidate_id.is_some() && req.init_candidate_id.is_some() {
        return Err(invalid("choose either checkpoint resume or warm start"));
    }
    let dataset = run.path(&req.sfm_id)?;
    let mut warnings = vec![];
    if let Some(p) = preset {
        if !req.total_steps_explicit {
            req.total_steps = req.total_steps.max(p.min_steps_for(manifest.train_names.len()));
        }
        warnings.push(format!(
            "MAX preset {}: {} steps, {} (requested {} steps, cap_max {}).",
            p.name,
            req.total_steps,
            if req.cap_max_explicit {
                format!("explicit cap {max_splats} kept")
            } else {
                match (max_splats, max_budget) {
                    (0, true) => "scene-adaptive splat budget up to 8M".to_string(),
                    (0, false) => "scene-adaptive splat budget up to 2M".to_string(),
                    (n, _) => format!("cap {n}"),
                }
            },
            requested.0,
            requested.1
        ));
    }
    let (mut init_ply, mut start_step) = (None, 0);
    if let Some(prior_id) = req.resume_candidate_id.as_ref().or(req.init_candidate_id.as_ref()) {
        let (prior, prior_dir) = get_artifact(run, prior_id, ArtifactKind::Candidate)?;
        if meta_str(&prior, "split_id")? != req.split_id || meta_str(&prior, "sfm_id")? != req.sfm_id {
            return Err(invalid("resume/warm start must use the original SfM and frozen split"));
        }
        if req.resume_candidate_id.is_some() {
            let ck: Checkpoint = read_json(&prior_dir.join("checkpoint.json"))
                .map_err(|_| invalid(format!("{prior_id} has no checkpoint to resume from")))?;
            if req.total_steps <= ck.step {
                return Err(invalid(format!("total_steps must exceed the checkpoint step {}", ck.step)));
            }
            start_step = ck.step;
            init_ply = Some(prior_dir.join(&ck.ply));
            warnings.push("Resume restores splats and the LR schedule; optimizer moments restart from zero.".into());
        } else {
            init_ply = Some(crate::views::candidate_ply(&prior_dir)?);
        }
    }
    let mut init_wait_s = None;
    if let Some(full_id) = &req.fork_of_candidate_id {
        if req.resume_candidate_id.is_some() || req.init_candidate_id.is_some() || splat_core::remote::mode().is_none() {
            return Err(invalid("fork_of_candidate_id is a cloud warm start of its own"));
        }
        let (full, _) = get_artifact(run, full_id, ArtifactKind::Candidate)?;
        if meta_str(&full, "split_id")? != req.split_id || meta_str(&full, "sfm_id")? != req.sfm_id {
            return Err(invalid("a forked refine must use the full training's SfM and frozen split"));
        }
        init_ply = Some(fork_path(run, full_id));
        init_wait_s = Some(splat_core::remote::job_timeout_s() - TIME_BUDGET_MARGIN_S);
    }
    // Transient down-weighting is always on in the trainer, so distraction_robustness needs no switch.
    if !matches!(req.floater_suppression, Strength::Off) {
        warnings.push("Brush has no floater_suppression; ignored.".into());
    }
    let max_edge = max_image_edge(&dataset.join("images"), &manifest.registered_names)?;
    let max_resolution = max_edge.div_ceil(req.resolution_divisor).max(64);
    let request = serde_json::to_value(&req)?;
    // MAX trains past the 30k reference schedule: stretch the absolute-step knobs by the same factor
    // (LR decay, progressive resolution, QAT tail and the refine cut-off already scale with total_steps).
    let stretch = preset
        .filter(|p| p.stretch_schedule && req.total_steps > 30_000)
        .map(|_| req.total_steps as f64 / 30_000.0);
    let defaults = TrainConfig::default();
    let stretched = |steps: u32| stretch.map(|s| (steps as f64 * s).round() as u32);
    let preset_name = preset.map_or("default", |p| p.name);
    let (artifact, out) = new_artifact(
        run,
        ArtifactKind::Candidate,
        json!({"sfm_id": req.sfm_id, "split_id": req.split_id, "request": request, "backend": "brush", "max_resolution": max_resolution,
            "preset": preset_name, "max_splats": max_splats, "max_budget": max_budget}),
    )?;
    // Blur mode is for photo sets (blurred-photos +3.8 dB). On video it cost driveway +564 s for
    // -0.12 dB / +0.009 LPIPS (bench aadaf5c vs 86143f5), and the detector can't tell the two apart.
    let video = read_json::<crate::sfm::SfmJob>(&dataset.join("sfm_job.json")).is_ok_and(|j| j.video);
    let config_overrides = crate::capture_sfm::training_overrides(&dataset, req.preview)?;
    let job = TrainJob {
        dataset,
        out: out.clone(),
        preview: req.preview,
        video,
        train_names: train_subset(&manifest.train_names, video, surfel),
        heldout_names: manifest.heldout_names.clone(),
        total_steps: req.total_steps,
        blur_total_steps: preset
            .filter(|p| p.max_budget && !req.total_steps_explicit && start_step == 0)
            .map(|_| BLUR_MAX_STEPS)
            .filter(|&t| t < req.total_steps),
        start_step,
        warm_start: req.init_candidate_id.is_some() || init_wait_s.is_some(),
        init_wait_s,
        // Video only: photo sets may refine poses, whose corrections live beside the final PLY.
        fork_ply: req.fork_min_splats.filter(|_| video && max_budget && start_step == 0 && init_ply.is_none()
            && splat_core::remote::mode().is_some()).map(|_| fork_path(run, &artifact.id)),
        fork_min_splats: req.fork_min_splats.unwrap_or(0),
        init_ply,
        fresh_init_ply: false,
        plain_init_ply: false,
        max_splats,
        cap_max_explicit: req.cap_max_explicit,
        max_budget,
        // Explicit caps need the full fit: garden-1M loses PLY quality with adaptive stopping.
        early_stop_db: preset.filter(|p| p.max_budget && !req.cap_max_explicit && start_step == 0).map(|_| 0.15),
        growth_start_iter: stretched(defaults.growth_start_iter),
        growth_stop_iter: stretched(defaults.growth_stop_iter),
        refine_every: stretched(defaults.refine_every),
        max_resolution,
        eval_every: req.eval_every.unwrap_or(if req.preview { req.total_steps } else { (req.total_steps / 5).clamp(500, 5000) }),
        // A cloud job's mid-run checkpoints are never read (a killed runner loses its /tmp out dir;
        // the time budget and stop both save one), and each 2M-splat PLY write took ~1 s: 22-39 s of
        // a 30k-step MAX fit. Its final checkpoint is still saved.
        checkpoint_every: if req.preview || splat_core::remote::mode().is_some() { req.total_steps } else { 500 },
        // A remote job is killed at its wall limit; leave time for setup, the final eval and the
        // result's sync-back so a slow run still ends with its latest splats.
        time_budget_s: splat_core::remote::mode().map(|_| splat_core::remote::job_timeout_s() - TIME_BUDGET_MARGIN_S),
        sh_degree: 3,
        sh_degree_interval: stretched(defaults.sh_degree_interval),
        // X39: MAX trains at D-SSIM weight 0.1 (default 0.2): flowers s1 +0.2 dB on the standard PLY.
        ssim_weight: preset.filter(|p| p.max_budget).map(|_| 0.1),
        surfel,
        neural_max: preset.is_some_and(|p| p.neural),
        config_overrides,
        seed: req.seed.unwrap_or(42),
        gpu_lock: cfg.gpu_lock.clone(),
        defer_lpips: splat_core::remote::mode().is_some() && !req.preview,
        cloud: splat_core::remote::mode().is_some(),
        pseudo: None,
    };
    write_json(&out.join("train_job.json"), &job)?;
    let mut r = start_worker(cfg, run, "train_splats", "train", &out.join("train_job.json"), &artifact)?;
    r.warnings.extend(warnings);
    r.metrics.insert("max_resolution".into(), json!(max_resolution));
    r.metrics.insert("preset".into(), json!(preset_name));
    // Scene-adaptive MAX budgets reach 8M splats on unbounded scenes.
    let forecast = vram_forecast_mib(if max_budget { 8_000_000 } else { max_splats }, max_resolution);
    r.metrics.insert("vram_forecast_mib".into(), json!(forecast));
    // Cloud runners have 48-96 GB; the 16 GB warning is for local GPUs. A MAX budget never warns: the
    // trainer clamps it to the detected device memory (splat-train's vram_splat_limit).
    if forecast > VRAM_WARN_MIB && !max_budget && (preset.is_none() || splat_core::remote::mode().is_none()) {
        r.warnings.push(format!(
            "VRAM forecast {forecast} MiB for cap_max {max_splats} at {max_resolution} px may not fit a 16 GB GPU; lower cap_max or raise resolution_divisor."
        ));
    }
    // Masks linked into the SfM dataset (purpose training_ignore/both) zero those pixels' loss.
    r.metrics.insert("loss_masked".into(), json!(job.dataset.join("masks").is_dir()));
    Ok(r)
}

/// A first-view preview that a quick cloud SfM job trains right after its solve ([`run_chained`]):
/// one upload and one fal queue for both, and no SfM result round trip before training (warm room
/// run 4f73f27c: the separate preview job cost 29.5 s for 19.1 s of training).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainedPreview {
    /// The preview's train job as [`train_splats`] would submit it, less what the solve decides:
    /// the split's names and `max_resolution` ([`run_chained`] fills them in).
    pub job: TrainJob,
    /// The request recorded on the candidate when the client adopts it ([`adopt_chained`]).
    pub request: TrainSplats,
    pub interval: u32,
}

/// The remote side's record of a [`ChainedPreview`] (`<candidate>/chained.json`).
const CHAINED_RECORD: &str = "chained.json";

/// Longest a chained preview may train before the SfM job gives up on it and returns its solve:
/// the 3.5k-step preview trains in ~10 s, and the client can still train a separate preview.
const CHAINED_TIMEOUT_S: f64 = 180.0;

/// The [`ChainedPreview`] for the quick SfM dataset `dataset`, trained into the (unregistered)
/// candidate dir `out`. Mirrors [`train_splats`] for a preview request: no MAX preset, explicit cap,
/// no checkpoints or held-out scoring.
pub(crate) fn chained_preview(cfg: &BrushConfig, sfm_id: &str, dataset: &Path, out: &Path, req: &PreviewTrain, video: bool) -> anyhow::Result<ChainedPreview> {
    let request: TrainSplats = serde_json::from_value(json!({"sfm_id": sfm_id, "split_id": "", "preview": true,
        "total_steps": req.total_steps, "cap_max": req.cap_max, "cap_max_explicit": true, "resolution_divisor": req.resolution_divisor}))?;
    let job = TrainJob {
        dataset: dataset.to_path_buf(),
        out: out.to_path_buf(),
        preview: true,
        video,
        train_names: vec![],
        heldout_names: vec![],
        total_steps: req.total_steps,
        blur_total_steps: None,
        start_step: 0,
        init_ply: None,
        warm_start: false,
        init_wait_s: None,
        fork_ply: None,
        fork_min_splats: 0,
        fresh_init_ply: false,
        plain_init_ply: false,
        max_splats: req.cap_max,
        cap_max_explicit: true,
        max_budget: false,
        early_stop_db: None,
        growth_start_iter: None,
        growth_stop_iter: None,
        refine_every: None,
        max_resolution: 0,
        eval_every: req.total_steps,
        checkpoint_every: req.total_steps,
        time_budget_s: splat_core::remote::mode().map(|_| splat_core::remote::job_timeout_s() - TIME_BUDGET_MARGIN_S),
        sh_degree: 3,
        sh_degree_interval: None,
        ssim_weight: None,
        surfel: false,
        neural_max: false,
        config_overrides: None,
        seed: 42,
        gpu_lock: cfg.gpu_lock.clone(),
        defer_lpips: false,
        cloud: splat_core::remote::mode().is_some(),
        pseudo: None,
    };
    // define_split's default interval: the pipeline's split of the quick solve.
    let interval = serde_json::from_value::<DefineSplit>(json!({"sfm_id": sfm_id}))?.interval;
    Ok(ChainedPreview { job, request, interval })
}

/// Training views by default: an even spread of this many registered non-held-out views. Dense frame
/// extraction feeds SfM (poses come from all frames) while training reads fewer, at about the old
/// cost (room 3: 69% of the uncapped gain at one-click compute). Photo sets and surfel (mesh)
/// training keep it: surfels upload a full-resolution grayscale image per view up front, and photo
/// sets over 150 views are unmeasured.
const DEFAULT_TRAIN_MAX_FRAMES: usize = 150;

/// Gaussian training on video: every registered non-held-out view, up to an even spread of this
/// many. On a 1-2 min handheld clip every frame clearly beat the 150-view spread on held-out views,
/// at about twice the training time. Video extraction keeps 5 fps up to 600 frames (media.rs
/// FULL_FRAME_FPS, target_count), so clips up to ~2 min train on every frame.
const VIDEO_TRAIN_MAX_FRAMES: usize = 600;

/// The training views: an even spread of SPLAT_TRAIN_MAX_FRAMES (0 = every view; default
/// VIDEO_TRAIN_MAX_FRAMES for gaussian training on video, else DEFAULT_TRAIN_MAX_FRAMES) of the
/// registered non-held-out names.
pub(crate) fn train_subset(names: &[String], video: bool, surfel: bool) -> Vec<String> {
    let cap = match std::env::var("SPLAT_TRAIN_MAX_FRAMES").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) => Some(n).filter(|&n| n >= 2),
        None => Some(if video && !surfel { VIDEO_TRAIN_MAX_FRAMES } else { DEFAULT_TRAIN_MAX_FRAMES }),
    };
    match cap {
        Some(n) if names.len() > n => (0..n).map(|i| names[i * (names.len() - 1) / (n - 1)].clone()).collect(),
        _ => names.to_vec(),
    }
}

/// SfM worker side: train `chain` on the finished solve in `dataset` as a `__brush train`
/// subprocess (its PROGRESS rows join the SfM job's log) and record its exit for [`adopt_chained`].
/// The split is define_split's for a fresh solve: every `interval`-th registered name held out.
pub(crate) fn run_chained(dataset: &Path, chain: &ChainedPreview) -> anyhow::Result<()> {
    let start = Instant::now();
    let names = crate::model::names(&dataset.join("sparse/0"))?;
    anyhow::ensure!(names.len() >= 2, "{} registered views: too few for a held-out split", names.len());
    let heldout: Vec<String> = names.iter().step_by(chain.interval.max(1) as usize).cloned().collect();
    let registered: std::collections::BTreeSet<&String> = names.iter().collect();
    let train_names: Vec<String> = registered.iter().filter(|n| !heldout.contains(n)).map(|n| (*n).clone()).collect();
    let mut job = chain.job.clone();
    job.max_resolution = max_image_edge(&dataset.join("images"), &names)?.div_ceil(chain.request.resolution_divisor).max(64);
    job.train_names = train_subset(&train_names, job.video, job.surfel);
    job.heldout_names = heldout;
    std::fs::create_dir_all(&job.out)?;
    let path = job.out.join("train_job.json");
    write_json(&path, &job)?;
    let exe = match std::env::var_os("SPLAT_BRUSH_EXE") {
        Some(p) => PathBuf::from(p),
        None => std::env::current_exe()?,
    };
    let mut child = std::process::Command::new(&exe).args(["__brush", "train"]).arg(&path).spawn().context("spawning the chained preview")?;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if start.elapsed().as_secs_f64() > CHAINED_TIMEOUT_S {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    let code = status.and_then(|s| s.code());
    // The first view needs only the web package, which the train worker built next to the splat. Its
    // PLYs (236 MB at 1M splats) would be published to the run's fal staging before the job returns:
    // 7-36 s of the first view (g3-speed first-view pairs, 2026-10-11). The full model's PLY is
    // published as before.
    let dropped_ply = code == Some(0) && job.out.join("web/scene.spz").is_file() && {
        let _ = std::fs::remove_file(job.out.join("checkpoint.ply"));
        std::fs::remove_file(job.out.join("splat.ply")).is_ok()
    };
    write_json(&job.out.join(CHAINED_RECORD), &json!({"exit": code, "timed_out": status.is_none(), "wall_s": start.elapsed().as_secs_f64(),
        "ply_dropped": dropped_ply}))?;
    println!("[sfm] chained preview: exit {code:?} in {:.1} s", start.elapsed().as_secs_f64());
    anyhow::ensure!(code == Some(0), "chained preview exited {code:?}");
    Ok(())
}

/// Client side, when the quick SfM `sfm_id` completed: register the preview its job trained
/// ([`run_chained`]) on the run's split for that solve (made here, as define_split makes it, so the
/// pipeline's define_split finds the same one). Its train job must name exactly that split's
/// views. Returns (split id, candidate id, the train summary).
pub(crate) fn adopt_chained(run: &dyn RunCtx, sfm_id: &str, chain: &ChainedPreview) -> anyhow::Result<(String, String, serde_json::Map<String, Value>)> {
    let out = &chain.job.out;
    let record: Value = read_json(&out.join(CHAINED_RECORD))?;
    anyhow::ensure!(record["exit"] == 0, "the chained preview exited {} (timed out: {})", record["exit"], record["timed_out"]);
    anyhow::ensure!(out.join("splat.ply").exists() || (record["ply_dropped"] == true && out.join("web/scene.spz").is_file()),
        "the chained preview left no splat.ply or web package");
    let summary: serde_json::Map<String, Value> = read_json(&out.join("train_summary.json"))?;
    let trained: TrainJob = read_json(&out.join("train_job.json"))?;
    let split = split::define_split(run, DefineSplit { sfm_id: sfm_id.into(), interval: chain.interval })?;
    anyhow::ensure!(split.status == Status::Succeeded, "define_split: {:?}", split.error.map(|e| e.message));
    let split_id = split.artifacts.first().context("define_split returned no split")?.id.clone();
    let manifest = split::resolve(run, &split_id, sfm_id)?;
    anyhow::ensure!(trained.train_names == train_subset(&manifest.train_names, trained.video, trained.surfel) && trained.heldout_names == manifest.heldout_names,
        "the chained preview trained on another split than {split_id}");
    let id = out.file_name().context("candidate dir")?.to_string_lossy().into_owned();
    let mut request = chain.request.clone();
    request.split_id = split_id.clone();
    let meta = json!({"sfm_id": sfm_id, "split_id": split_id, "request": request, "backend": "brush",
        "max_resolution": trained.max_resolution, "preset": "default", "max_splats": trained.max_splats, "max_budget": false,
        "chained": true});
    let Value::Object(meta) = meta else { unreachable!() };
    run.register(Artifact { id: id.clone(), kind: ArtifactKind::Candidate, path: rel(run, out), meta: meta.into_iter().collect() })?;
    Ok((split_id, id, summary))
}

/// Whether this run's deliverables include a mesh, so training should use
/// surfels. `--no-mesh` (recorded in fixed.json) opts out; the fixed pipeline
/// otherwise always extracts one; an agent run wants one when its goal is the
/// default (a textured GLB mesh) or names a mesh.
/// Surfel (2DGS) training when the user asked for a mesh: the run's goal names a mesh. Runs
/// that only get the fixed pipeline's side mesh (benchmarks, leaderboards) stay gaussian.
fn wants_mesh(run: &dyn RunCtx) -> bool {
    splat_core::remote::goal_wants_mesh(&run.goal())
}

/// Rough peak-VRAM forecast from the final splat count and image size, fitted
/// to L40S leaderboard peaks (~0.5 GiB base, ~2 KiB per splat including raster
/// workspaces, plus image buffers per pixel). A 0 cap means the trainer's
/// automatic cap (at most 1.2M).
fn vram_forecast_mib(cap_max: u32, max_resolution: u32) -> u64 {
    let splats = if cap_max == 0 { 1_200_000 } else { cap_max } as u64;
    let pixels = (max_resolution as u64).pow(2) * 3 / 4;
    600 + (splats * 2048 + pixels * 64) / (1024 * 1024)
}

/// Warn above this forecast (leaves headroom on a 16 GB card).
const VRAM_WARN_MIB: u64 = 12 * 1024;

/// Largest edge over every registered image (headers only, so this is cheap).
fn max_image_edge(images: &Path, names: &[String]) -> anyhow::Result<u32> {
    let mut edge = 0;
    for n in names {
        let (w, h) = image::image_dimensions(images.join(n)).with_context(|| format!("reading {n}"))?;
        edge = edge.max(w.max(h));
    }
    Ok(edge)
}

/// The interactive refine's fork in flight: its PLY being copied to fal /data.
type Fork = Option<std::thread::JoinHandle<()>>;

fn fork_none(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".none");
    PathBuf::from(p)
}

/// Write the fork (or its `.none` marker): the PLY goes to the job's local disk now (~1 s at 2M
/// splats) and on to fal /data in the background, renamed into place once complete, so training
/// does not wait on the network volume.
async fn write_fork(out: &Path, path: &Path, splats: &Splats, min_splats: u32, step: u32, up_axis: Option<glam::Vec3>) -> Fork {
    let n = splats.num_splats();
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(Path::new("/")));
    if n <= min_splats {
        let _ = std::fs::write(fork_none(path), format!("{n} splats\n"));
        emit(out, &json!({"event": "fork_skipped", "step": step, "splats": n}));
        return None;
    }
    let t = Instant::now();
    let local = std::env::temp_dir().join(format!("splat-fork-{}.ply", std::process::id()));
    if let Err(e) = engine::save_ply(splats.clone(), up_axis, &local).await {
        let _ = std::fs::write(fork_none(path), format!("save failed: {e:#}\n"));
        emit(out, &json!({"event": "fork_failed", "step": step, "error": format!("{e:#}")}));
        return None;
    }
    emit(out, &json!({"event": "fork_saved", "step": step, "splats": n, "save_s": t.elapsed().as_secs_f64()}));
    let (out, path) = (out.to_path_buf(), path.to_path_buf());
    Some(std::thread::spawn(move || {
        let t = Instant::now();
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        let copied = std::fs::copy(&local, &tmp).and_then(|_| std::fs::rename(&tmp, &path));
        let _ = std::fs::remove_file(&local);
        match copied {
            Ok(()) => emit(&out, &json!({"event": "fork_published", "step": step, "copy_s": t.elapsed().as_secs_f64()})),
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                let _ = std::fs::write(fork_none(&path), format!("copy failed: {e}\n"));
                emit(&out, &json!({"event": "fork_failed", "step": step, "error": e.to_string()}));
            }
        }
    }))
}

/// Training ended: finish the fork's copy, or mark it unavailable if training never reached the fork.
fn settle_fork(job: &TrainJob, fork: Option<Fork>) {
    match (fork, &job.fork_ply) {
        (Some(Some(copy)), _) => {
            let _ = copy.join();
        }
        (None, Some(path)) => {
            let _ = std::fs::write(fork_none(path), "no fork: training ended before reaching it\n");
        }
        _ => {}
    }
}

/// Wait for the fork at `path` (up to `wait_s`) and copy it to local disk: reads from the /data
/// network volume are slow. Fails on the `.none` marker, a stop, or the timeout.
fn wait_fork(out: &Path, path: &Path, wait_s: f64, stop: &AtomicBool) -> anyhow::Result<PathBuf> {
    let t = Instant::now();
    while !path.is_file() {
        let none = fork_none(path);
        if none.exists() {
            let why = std::fs::read_to_string(&none).unwrap_or_default();
            emit(out, &json!({"event": "fork_unavailable", "waited_s": t.elapsed().as_secs_f64(), "reason": why.trim()}));
            anyhow::bail!("fork unavailable: {}", why.trim());
        }
        anyhow::ensure!(!stop.load(Ordering::SeqCst), "stopped while waiting for the fork");
        anyhow::ensure!(t.elapsed().as_secs_f64() < wait_s, "no fork after {wait_s:.0} s");
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    let waited_s = t.elapsed().as_secs_f64();
    let local = std::env::temp_dir().join(format!("splat-fork-in-{}.ply", std::process::id()));
    std::fs::copy(path, &local).with_context(|| format!("copying {}", path.display()))?;
    // One refine reads each fork.
    let _ = std::fs::remove_file(path);
    emit(out, &json!({"event": "fork_ready", "waited_s": waited_s, "copy_s": t.elapsed().as_secs_f64() - waited_s}));
    Ok(local)
}

fn emit(out: &Path, row: &Value) {
    use std::io::Write;
    println!("PROGRESS {row}");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(out.join("training.jsonl")) {
        let _ = writeln!(f, "{row}");
    }
}

async fn score(splats: &Splats, views: &[SceneView], device: &Device, save: Option<&Path>) -> anyhow::Result<(f64, f64)> {
    score_with(splats, views, None, device, save).await
}

/// [`score`] with optional per-view cameras (test-time pose alignment) in place of the views' own.
async fn score_with(splats: &Splats, views: &[SceneView], cameras: Option<&[brush_render::camera::Camera]>, device: &Device, save: Option<&Path>) -> anyhow::Result<(f64, f64)> {
    let (mut psnr, mut ssim) = (0.0, 0.0);
    for (i, view) in views.iter().enumerate() {
        let gt = engine::load_gt(view).await?;
        let camera = cameras.map_or(&view.camera, |c| &c[i]);
        let sample = eval_stats(splats.clone(), camera, gt.clone(), engine::gt_alpha_mode(view), device).await?;
        let view_psnr = sample.psnr.clone().into_scalar_async::<f32>().await? as f64;
        // Heartbeat per scored view for stall watchdogs (output = liveness).
        eprintln!("[eval] view {}/{} psnr {view_psnr:.2}", i + 1, views.len());
        psnr += view_psnr;
        ssim += sample.ssim.clone().into_scalar_async::<f32>().await? as f64;
        if let Some(dir) = save {
            let s = stem(&engine::view_name(view));
            gt.to_rgb8().save(dir.join("gt").join(format!("{s}.png")))?;
            sample.save_to_disk(&dir.join("renders").join(format!("{s}.png"))).await?;
        }
    }
    let n = views.len().max(1) as f64;
    Ok((psnr / n, ssim / n))
}

/// Worker body. Exit 0 = finished; 130 = stopped by signal after writing a checkpoint.
pub fn run_job(job: TrainJob) -> anyhow::Result<i32> {
    anyhow::ensure!(!job.fresh_init_ply || (job.init_ply.is_some() && job.start_step == 0),
        "fresh_init_ply requires a PLY and start_step=0");
    anyhow::ensure!(!job.plain_init_ply || job.fresh_init_ply, "plain_init_ply requires fresh_init_ply");
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        // The job runner sends one SIGINT to the process group and waits; finish the step,
        // checkpoint, exit.
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst)).context("installing signal handler")?;
    }
    let wait = Instant::now();
    let _gpu = crate::gpu_lock(job.gpu_lock.as_deref())?;
    emit(&job.out, &json!({"event": "gpu_lock", "gpu_lock_wait_s": wait.elapsed().as_secs_f64()}));
    let rt = engine::runtime()?;
    rt.block_on(async {
        let result = train(job, stop).await;
        // Lagged render-count readbacks still in flight at process exit race the
        // driver's teardown (SIGSEGV after a finished run). Settle on every exit
        // path, including errors.
        brush_render::render::settle_lagged_counts().await;
        result
    })
}

async fn train(mut job: TrainJob, stop: Arc<AtomicBool>) -> anyhow::Result<i32> {
    let wall = Instant::now();
    let device = engine::init_device().await;
    // Surfels (mesh goals) train on a pinhole copy of a distorted dataset (video SfM ends RADIAL).
    let (dataset, _pinhole) = if job.surfel {
        match engine::surfel_dataset(&job.dataset, &job.out.join("pinhole_dataset")) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[train] WARNING: can't undistort the dataset for surfels ({e:#})");
                (job.dataset.clone(), None)
            }
        }
    } else {
        (job.dataset.clone(), None)
    };
    let scene = engine::load_scene(&dataset, job.max_resolution).await?;
    for w in &scene.warnings {
        eprintln!("[dataset] {w}");
    }
    let pick = |names: &[String]| -> anyhow::Result<Vec<SceneView>> {
        names
            .iter()
            .map(|n| scene.views.iter().find(|v| &engine::view_name(v) == n).cloned().with_context(|| format!("{n} missing from dataset")))
            .collect()
    };
    let mut train_views = pick(&job.train_names)?;
    let eval_views = pick(&job.heldout_names)?;
    // Surfels render pinhole cameras only (an unsupported lens would still render, distortion
    // ignored): anything the undistorter can't handle trains gaussians instead.
    if job.surfel && train_views.iter().chain(&eval_views).any(|v| !matches!(v.camera.camera_model, brush_render::kernels::camera_model::CameraModel::Pinhole)) {
        eprintln!("[train] WARNING: surfel (2DGS) training needs pinhole cameras and this dataset's lens can't be undistorted; training gaussians instead");
        emit(&job.out, &json!({"event": "surfel_fallback", "reason": "non-pinhole cameras"}));
        job.surfel = false;
    }
    device.seed(job.seed + job.start_step as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(job.seed);

    let up_axis = Some(scene.up_axis);
    let mode = if job.surfel { SplatRenderMode::Surfel } else { SplatRenderMode::Mip };
    // A forked refine: its PLY appears on fal /data when the full model's job reaches its fork.
    let fork_local = match (&job.init_ply, job.init_wait_s) {
        (Some(ply), Some(wait_s)) => Some(wait_fork(&job.out, ply, wait_s, &stop)?),
        _ => None,
    };
    let init_ply = fork_local.clone().or_else(|| job.init_ply.clone());
    let splats = if let Some(ply) = &init_ply {
        if job.plain_init_ply {
            engine::load_plain_init_ply(ply, job.max_splats as usize, mode, &device).await?
        } else {
            let cap = if job.cap_max_explicit { job.max_splats as usize } else { 0 };
            let select = if job.warm_start { crate::warm_cap::Select::Hybrid } else { crate::warm_cap::Select::Strided };
            let (splats, stats) = crate::warm_cap::load(ply, cap, select, mode, &train_views, job.max_resolution, &device).await?;
            emit(&job.out, &json!({"event": "warm_start", "stats": stats}));
            if let Some(local) = &fork_local {
                let _ = std::fs::remove_file(local);
            }
            splats
        }
    } else if let Some(init) = scene.init {
        eprintln!("init: render_mode={mode:?} (source=job)");
        let data = init.data.subsample(job.max_splats as usize);
        to_init_splats(data, mode, &device)
    } else {
        eprintln!("init: render_mode={mode:?} (source=job)");
        let cams: Vec<_> = train_views.iter().map(|v| v.camera).collect();
        create_random_splats(&RandomSplatsConfig::new(), &cams, None, &mut rng, mode, &device)
    };
    // A warm start / resume PLY keeps its parameters but trains in this job's mode.
    let mut splats = splats;
    splats.surfel = job.surfel;
    splats.render_mip &= !job.surfel;
    // SH degree schedule: fresh splats start at degree 0 and the trainer grows them.
    let mut splats = if job.init_ply.is_none() || job.fresh_init_ply {
        splats.with_sh_degree(0)
    } else {
        splats.with_sh_degree(job.sh_degree)
    };
    device.memory_cleanup();

    let mut config = TrainConfig::default();
    config.total_train_iters = job.total_steps;
    config.max_splats = job.max_splats;
    config.max_budget = job.max_budget;
    if job.surfel {
        config.apply_surfel_defaults();
    } else if job.neural_max {
        config.apply_neural_max();
    }
    if job.cap_max_explicit {
        config.max_splats = job.max_splats;
        config.max_budget = false;
    }
    // X56: a time-boxed preview gains more from steps than from deblur (9 sub-pose renders
    // per step once blur is detected): rooftop preview 22.85 dB with it off.
    // A warm start refines a model whose own job already decided blur mode: re-detecting on the
    // restarted schedule switched it on spuriously (garden 1M, 20k steps: -3.0 dB vs full, one panic).
    if job.preview || job.video || job.warm_start {
        config.camera_motion_blur = false;
    }
    // Person/distractor masks from mask_distractors (sfm.rs marks them) stay on only while
    // what they hide moves; a statue the detector reads as a person is unmasked after a probe.
    config.auto_masks = job.dataset.join(DETECTOR_MASKS_MARKER).is_file();
    if let Some(value) = job.sh_degree_interval {
        anyhow::ensure!(value > 0, "sh_degree_interval must be positive");
        config.sh_degree_interval = value;
    }
    if let Some(value) = job.ssim_weight {
        anyhow::ensure!((0.0..=1.0).contains(&value), "ssim_weight must be in [0, 1]");
        config.ssim_weight = value;
        eprintln!("x39 ssim_weight {value}");
    }
    if let Some(value) = job.growth_start_iter { config.growth_start_iter = value; }
    if let Some(value) = job.growth_stop_iter { config.growth_stop_iter = value; }
    if let Some(value) = job.refine_every {
        anyhow::ensure!(value > 0, "refine_every must be positive");
        config.refine_every = value;
    }
    // Joint camera refinement. SPLAT_POSE_REFINE=1 forces it on, =0 off (SPLAT_* env reaches cloud jobs too); unset,
    // it is on for video and for photo sets whose expected splat budget is at most POSE_REFINE_AUTO_MAX_SPLATS (2.25M) (decided below, from
    // the inputs, before the first step). Its per-step cost grows with the splat count: in-exec
    // steady state +1.7..1.9% at 1.9-2.05M (room, ETH3D delivery), +7.0..9.7% at 4.6-4.8M (driveway, garden), over the 3% budget there. When on: Goal-2 gate: ETH3D AUC@3 >= Spirula on 3/4 scenes (from 1/4), +1.9% in-exec
    // training time, held-out PSNR guards on bench + phone video. Refined training poses drift as
    // a gauge, so held-out views are pose-aligned before the final score by the evaluate job (unaligned is reported too).
    // Unaligned held-out PSNR is expected to drop (house -0.74 dB, others -0.05..-0.26): refined
    // training poses and unrefined test poses sit in slightly different gauges. Compare aligned to
    // aligned, including external references (the Spirula reference is scored aligned as well).
    let pose_refine_env = std::env::var("SPLAT_POSE_REFINE").ok().map(|v| v.trim() == "1");
    if pose_refine_env == Some(true) && !job.preview {
        config.pose_refine = true;
        config.pose_align_eval_steps = POSE_ALIGN_DEFAULT_STEPS;
    }
    if let Some(over) = &job.config_overrides {
        let mut merged = serde_json::to_value(&config)?;
        for (k, v) in over {
            anyhow::ensure!(merged.get(k).is_some(), "unknown TrainConfig field {k}");
            merged[k] = v.clone();
        }
        config = serde_json::from_value(merged)?;
        // Read each override back from the final config: proof it took effect.
        let applied = serde_json::to_value(&config)?;
        for k in over.keys() {
            eprintln!("config override applied: {k} = {}", applied[k]);
        }
    }
    // Goal 3 (rooms): sensor depth prior as a training regulariser (plain walls, dim rooms),
    // loaded in the background while training starts. SPLAT_DEPTH_PRIOR sets its weight (0 = off).
    if let Some(w) = crate::depth_prior::weight_override()? {
        config.depth_prior_weight = w;
    }
    // On only for inputs whose dataset carries metric sensor depth (capture imports: depths.json
    // kind "metric"). A monocular prior (Depth-Anything-V2-Small) on plain video lost:
    // LiDAR oracle on arkit-41254538 (same SfM, weight 0.5): sensor +0.29 dB held-out with off-path
    // sharpness kept, mono -0.80 dB and 16% softer off-path.
    let sensor_depth = read_json::<Value>(&job.dataset.join("depths.json")).is_ok_and(|d| d["kind"] == "metric");
    let _depth_worker = (sensor_depth && !job.surfel && !job.preview && !job.neural_max && config.depth_prior_weight > 0.0)
        .then(|| {
            eprintln!("depth prior: weight {} until {:.0}% of training, every {} steps",
                config.depth_prior_weight, config.depth_prior_end * 100.0, config.depth_prior_every);
            eprintln!("depth prior: planarity weight {}", config.depth_planar_weight);
            crate::depth_prior::attach(&mut train_views, &job.dataset)
        });
    if job.start_step > 0 {
        // Continue the exponential mean-LR decay where the previous run stopped.
        let decay = (config.lr_mean_end / config.lr_mean).powf(1.0 / job.total_steps as f64);
        config.lr_mean *= decay.powi(job.start_step as i32);
        config.lr_mean_end = config.lr_mean * decay.powi(job.total_steps as i32);
    }
    let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new_seeded(&config, &device, bounds, job.seed);
    let mut view_cams = Vec::with_capacity(train_views.len());
    let mut train_pixels = 0u64;
    for v in &train_views {
        let (w, h) = v.image.dimensions().await?;
        train_pixels += train_pixel_count(w, h, job.max_resolution);
        view_cams.push((v.camera.position, v.camera.focal(glam::uvec2(w, h)).x));
    }
    trainer.set_view_cams(view_cams);
    trainer.set_train_pixels(train_pixels);
    // Pose refine for video and photo sets (see SPLAT_POSE_REFINE above): stills are where camera precision matters most and
    // their budgets tend to be small. A warm start / resume follows its source candidate (its corrections, frozen).
    if pose_refine_env.is_none() && !config.pose_refine && !job.preview && !job.surfel && !job.neural_max {
        let on = if job.video && job.init_ply.is_none() {
            // Goal 3 (rooms, 4 public dev rooms, product path, aligned eval): held-out LPIPS better on 4/4,
            // PSNR +0.25 dB mean, off-path slightly sharper, ~+2% training time (goal3/recipe/RESULTS.md).
            eprintln!("pose refine auto (video): true");
            true
        } else if let Some(ply) = &job.init_ply {
            let carried = ply.parent().is_some_and(|d| d.join("pose_refine.json").exists());
            eprintln!("pose refine auto (photos): {carried} (source candidate refined: {carried})");
            carried
        } else {
            let means = splats.means().into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let budget = trainer.expected_splat_budget(means.chunks_exact(3).map(|m| glam::Vec3::new(m[0], m[1], m[2])));
            let on = budget <= POSE_REFINE_AUTO_MAX_SPLATS;
            eprintln!("pose refine auto (photos): {on} (expected splat budget {budget}, limit {POSE_REFINE_AUTO_MAX_SPLATS})");
            on
        };
        if on {
            config.pose_refine = true;
            config.pose_align_eval_steps = POSE_ALIGN_DEFAULT_STEPS;
            trainer.set_pose_refine(true);
        }
    }
    let pose_refine = trainer.set_pose_views(train_views.iter().map(|v| v.camera_index).collect());
    // A warm start or resume loads splats trained in its source candidate's refined gauge: carry the
    // corrections over (pose_refine.json beside the PLY) and keep them fixed, so this job neither
    // trains against the unrefined poses nor pays for refinement again.
    let mut warm_poses = 0usize;
    if let (Some(p), Some(ply)) = (trainer.pose_refine_mut(), job.init_ply.as_ref()) {
        if let Some(rows) = ply.parent().map(|d| d.join("pose_refine.json")).filter(|f| f.exists())
            .and_then(|f| read_json::<Value>(&f).ok()).and_then(|v| v["views"].as_array().cloned())
        {
            for row in rows {
                let (Some(name), Some(q), Some(t)) = (row["view"].as_str(), row["qvec"].as_array(), row["tvec"].as_array()) else { continue };
                let Some(i) = train_views.iter().position(|v| engine::view_name(v) == name) else { continue };
                let f = |a: &Vec<Value>, k: usize| a.get(k).and_then(Value::as_f64).unwrap_or(0.0) as f32;
                let w2c = glam::Affine3A::from_rotation_translation(
                    glam::Quat::from_xyzw(f(q, 1), f(q, 2), f(q, 3), f(q, 0)).normalize(),
                    glam::Vec3::new(f(t, 0), f(t, 1), f(t, 2)));
                p.set_correction(i, w2c * train_views[i].camera.local_to_world());
                warm_poses += 1;
            }
        }
        p.freeze();
    }
    if warm_poses > 0 {
        eprintln!("pose refine: {warm_poses} corrections carried over from the warm-start candidate (frozen)");
    }
    if config.pose_refine || config.pose_align_eval_steps > 0 {
        eprintln!("pose refine: {pose_refine} (focal {}), start {}, end {}, every {}, lr rot {} trans {} x extent {:.4}, eval alignment {} steps, ba step {} ({} iters/view, lr rot {} trans {})",
            config.pose_refine_focal, config.pose_refine_start, config.pose_refine_end, config.pose_refine_every, config.pose_refine_lr_rot, config.pose_refine_lr_trans,
            trainer.cam_extent(), config.pose_align_eval_steps, config.pose_ba_step, config.pose_ba_iters, config.pose_ba_lr_rot, config.pose_ba_lr_trans);
    }
    if job.surfel {
        // Multi-view consistency: cameras plus grayscale training images.
        let mut cams = Vec::with_capacity(train_views.len());
        let mut grays = Vec::with_capacity(train_views.len());
        for v in &train_views {
            let img = engine::load_gt(v).await?.to_rgb32f();
            let (w, h) = img.dimensions();
            let gray: Vec<f32> = img.pixels().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect();
            let t = burn::tensor::Tensor::<2>::from_data(burn::tensor::TensorData::new(gray, [h as usize, w as usize]), &device);
            cams.push(v.camera.clone());
            grays.push((t, glam::uvec2(w, h)));
        }
        trainer.set_multiview(brush_train::multiview::MultiView::new(cams, grays));
    }
    trainer.set_target_sh_degree(job.sh_degree);
    // Use training views only, never held-out views, for the early stop.
    // SPLAT_EARLY_STOP=0 trains the full schedule (no adaptive stop).
    let early_stop_db = job.early_stop_db
        .filter(|_| std::env::var("SPLAT_EARLY_STOP").map_or(true, |v| v.trim() != "0"));
    if let Some(db) = early_stop_db {
        anyhow::ensure!(db.is_finite() && db > 0.0, "early_stop_db must be positive");
        anyhow::ensure!(job.start_step == 0 && !job.surfel && !job.neural_max && config.qat_tail,
            "early stop requires fresh SH training with QAT");
    }
    // V1: the stop watches validation frames that the fit never sees: train PSNR kept rising
    // 0.23-0.46 dB per 3k on house while held-out PSNR fell (1M model 28.74 vs 3.12M 28.52).
    // They are evenly spaced like the frozen held-out split, so on video their neighbours
    // still train (the same leakage the held-out score has). The QAT tail trains on them too.
    let validation_idx: Vec<usize> = if early_stop_db.is_some() && train_views.len() >= VALIDATION_MIN_TRAIN {
        let k = (train_views.len() * VALIDATION_PERCENT / 100).clamp(VALIDATION_MIN, VALIDATION_MAX);
        (0..k).map(|i| (2 * i + 1) * train_views.len() / (2 * k)).collect()
    } else {
        Vec::new()
    };
    let validation_views: Vec<SceneView> = validation_idx.iter().map(|&i| train_views[i].clone()).collect();
    let mut fit_views: Vec<SceneView> = train_views.iter().enumerate()
        .filter(|(i, _)| !validation_idx.contains(i)).map(|(_, v)| v.clone()).collect();
    // Pose refinement keys its corrections by training view; the loader indexes fit_views.
    trainer.set_pose_loader_views((0..train_views.len()).filter(|i| !validation_idx.contains(i)).collect());
    // Progressive resolution: the loader is rebuilt when the image scale changes.
    let load_config = engine::load_config(job.max_resolution);
    let scaled_views = |views: &[SceneView], scale: f32| {
        let scene = Scene::new(views.to_vec());
        if scale < 1.0 { scene.with_image_scale(scale) } else { scene }
    };
    // A fill fine-tune (pseudo views) refines a converged model: full resolution from the start.
    let scale_at = |config: &TrainConfig, step: u32| if job.pseudo.is_some() { 1.0 } else { config.image_scale_at(step) };
    let mut image_scale = scale_at(&config, job.start_step);
    let mut loader = SceneLoader::new(&scaled_views(&fit_views, image_scale), job.seed + job.start_step as u64, &load_config);
    let mut pseudo = match &job.pseudo {
        Some(p) => Some(PseudoLoader::new(p, &load_config, job.seed, train_views.len()).await?),
        None => None,
    };
    if let Some(p) = &pseudo {
        emit(&job.out, &json!({"event": "pseudo_views", "n": p.views.len(), "frac": p.cfg.frac, "weight": p.cfg.weight,
            "ramp": p.cfg.ramp, "pos_lr": p.cfg.pos_lr, "stop": p.cfg.stop}));
    }

    let checkpoint_out = job.out.clone();
    let save_checkpoint = |splats: Splats, step: u32, total: u32| {
        let out = checkpoint_out.clone();
        async move {
            let n = splats.num_splats();
            engine::save_ply(splats, up_axis, &out.join("checkpoint.ply")).await?;
            write_json(&out.join("checkpoint.json"), &Checkpoint { step, total_steps: total, splats: n, ply: "checkpoint.ply".into() })
        }
    };

    emit(&job.out, &json!({"event": "start", "step": job.start_step, "total_steps": job.total_steps, "n_train": train_views.len(), "n_heldout": eval_views.len(), "n_validation": validation_views.len(), "init_splats": splats.num_splats(), "max_resolution": job.max_resolution, "adapter": brush_render::adapter_summary().map(|a| json!({"name": a.name, "backend": a.backend, "driver": a.driver, "driver_info": a.driver_info}))}));
    let train_start = Instant::now();
    let mut step = job.start_step;
    // The train-PSNR slope stop runs on fitted views alongside the validation stop.
    let convergence_views: Vec<_> = fit_views.iter().step_by((fit_views.len() / 8).max(1)).take(8).cloned().collect();
    // Training-view indices of the convergence views and, per validation view, its nearest fitted
    // view: early-stop checks score them with the refined gauge, or pose drift reads as a quality
    // change (one driveway pair stopped 9k steps later without this).
    let fit_idx: Vec<usize> = (0..train_views.len()).filter(|i| !validation_idx.contains(i)).collect();
    let convergence_idx: Vec<usize> = fit_idx.iter().step_by((fit_views.len() / 8).max(1)).take(8).copied().collect();
    let validation_near: Vec<usize> = validation_idx.iter().map(|&vi| {
        let p = train_views[vi].camera.position;
        *fit_idx.iter().min_by(|&&a, &&b| train_views[a].camera.position.distance_squared(p)
            .total_cmp(&train_views[b].camera.position.distance_squared(p))).unwrap_or(&vi)
    }).collect();
    // Convergence checks so far (step, train PSNR). The stop uses the least-squares slope over
    // the last CONVERGENCE_FIT checks: two raw gains below the threshold in a row fired on one
    // noisy dip and a single rise reset it (gimbal stopped at 30k in one run and 36k in another
    // with equal held-out PSNR at 30k).
    let mut convergence: Vec<(u32, f64)> = Vec::new();
    // Validation checks so far (step, validation PSNR).
    let mut validation: Vec<(u32, f64)> = Vec::new();
    let mut early_stopped = false;
    let mut actual_steps = 0u32;
    // The interactive refine's fork (`fork_ply`): written once, two thirds in or at the early stop.
    let mut fork: Option<Fork> = None;
    let mut blur_noted = false;
    while step < job.total_steps {
        if stop.load(Ordering::SeqCst) {
            save_checkpoint(splats.clone(), step, job.total_steps).await?;
            emit(&job.out, &json!({"event": "stopped", "step": step, "splats": splats.num_splats()}));
            return Ok(130);
        }
        let scale = scale_at(&config, step);
        if scale != image_scale {
            image_scale = scale;
            loader = SceneLoader::new(&scaled_views(&fit_views, scale), job.seed + step as u64, &load_config);
        }
        let picked = match pseudo.as_mut() {
            Some(p) => p.pick(step, job.total_steps).await,
            None => None,
        };
        let batch = match picked {
            Some((batch, step_cfg)) => {
                trainer.set_pseudo_step(step_cfg);
                batch
            }
            None => loader.next_batch().await,
        };
        let (stepped, stats) = trainer.step(batch, splats.train()).await;
        splats = stepped.valid();
        if !blur_noted && trainer.blur_mode() {
            blur_noted = true;
            emit(&job.out, &json!({"event": "blur_mode", "step": step, "note": brush_train::train::BLUR_MODE_NOTE}));
            if let Some(total) = job.blur_total_steps.filter(|&t| step < t && t < job.total_steps) {
                // The 20k schedule is not stretched: the step knobs go back to their defaults.
                let defaults = TrainConfig::default();
                config.total_train_iters = total;
                config.growth_start_iter = defaults.growth_start_iter;
                config.growth_stop_iter = defaults.growth_stop_iter;
                config.refine_every = defaults.refine_every;
                config.sh_degree_interval = defaults.sh_degree_interval;
                trainer.retarget(&config);
                if job.checkpoint_every == job.total_steps {
                    job.checkpoint_every = total;
                }
                job.eval_every = (total / 5).clamp(500, 5000);
                emit(&job.out, &json!({"event": "schedule", "step": step, "total_steps": total, "from_total_steps": job.total_steps, "reason": "blur_mode"}));
                job.total_steps = total;
            }
        }
        if step > 0 && step.is_multiple_of(config.refine_every) && step as f32 / job.total_steps as f32 <= 0.95 {
            let (refined, rs) = trainer.refine(step, splats).await;
            splats = refined;
            emit(&job.out, &json!({"event": "refine", "step": step, "added": rs.num_added, "split_oversized": rs.num_split_oversized,
                "split_high_grad": rs.num_split_high_grad, "pruned": rs.num_pruned, "pruned_non_finite": rs.num_pruned_non_finite,
                "relocated": rs.num_relocated, "total": rs.total_splats}));
        }
        step += 1;
        actual_steps += 1;
        // Fork two thirds in (30k of 45k), or at MAX's early stop if sooner (below): refines from the
        // end of densification (22.5k) lost LPIPS-vgg +0.002 on arkit-41254538 (2 seeds) vs the final
        // model; from 30k they matched it on arkit and pexels-livingroom-stove (g3-speed-v3 screen).
        if step == job.total_steps * 2 / 3 && fork.is_none() {
            if let Some(path) = &job.fork_ply {
                fork = Some(write_fork(&job.out, path, &splats, job.fork_min_splats, step, up_axis).await);
            }
        }
        if step == config.pose_ba_step && trainer.pose_refine().is_some() {
            let mut ba = pose_ba(&mut trainer, &splats, &train_views, &config, image_scale).await?;
            ba["event"] = json!("pose_ba");
            ba["step"] = json!(step);
            emit(&job.out, &ba);
        }
        if step.is_multiple_of(100) || step == job.total_steps {
            let loss = stats.loss.into_scalar_async::<f32>().await?;
            // Divergence alarm: the photometric loss is bounded (~2), so a
            // non-finite or huge value means training blew up. Stop without
            // overwriting the last good checkpoint.
            if !loss.is_finite() || loss > DIVERGED_LOSS {
                emit(&job.out, &json!({"event": "diverged", "step": step, "loss": loss.to_string(), "splats": splats.num_splats()}));
                eprintln!("training diverged at step {step}: loss {loss}");
                return Ok(EXIT_DIVERGED);
            }
            let elapsed = train_start.elapsed().as_secs_f64();
            let rate = actual_steps as f64 / elapsed.max(1e-6);
            emit(&job.out, &json!({"event": "step", "step": step, "total_steps": job.total_steps, "splats": splats.num_splats(), "loss": loss, "elapsed_s": elapsed, "eta_s": (job.total_steps - step) as f64 / rate}));
        }
        if step.is_multiple_of(job.eval_every) && step < job.total_steps && !eval_views.is_empty() {
            let t = Instant::now();
            let (psnr, ssim) = score(&splats, &eval_views, &device, None).await?;
            emit(&job.out, &json!({"event": "eval", "step": step, "scope": "heldout", "psnr": psnr, "ssim": ssim, "eval_s": t.elapsed().as_secs_f64()}));
        }
        if step.is_multiple_of(job.checkpoint_every) && step < job.total_steps {
            save_checkpoint(splats.clone(), step, job.total_steps).await?;
        }
        if job.time_budget_s.is_some_and(|b| train_start.elapsed().as_secs_f64() > b) && step < job.total_steps {
            emit(&job.out, &json!({"event": "time_budget", "step": step, "total_steps": job.total_steps, "splats": splats.num_splats()}));
            break;
        }
        if let Some(threshold) = early_stop_db.filter(|_| !early_stopped) {
            if step >= config.growth_stop_iter && step.is_multiple_of(3000) && step < job.total_steps * 19 / 20 {
                let conv_cams = trainer.pose_refine().map(|p| convergence_idx.iter().zip(&convergence_views)
                    .map(|(&i, v)| p.refined_camera(i, &v.camera)).collect::<Vec<_>>());
                let (psnr, _) = score_with(&splats, &convergence_views, conv_cams.as_deref(), &device, None).await?;
                convergence.push((step, psnr));
                // Fitted gain per 3k steps over the last CONVERGENCE_FIT checks.
                let gain = (convergence.len() >= CONVERGENCE_FIT)
                    .then(|| fit_slope(&convergence[convergence.len() - CONVERGENCE_FIT..]) * 3000.0);
                emit(&job.out, &json!({"event": "convergence", "step": step, "train_psnr": psnr,
                    "gain_db": gain, "threshold_db": threshold, "fit_points": CONVERGENCE_FIT}));
                let slope_fire = gain.is_some_and(|g| g < threshold);
                // Validation decides when there is a validation set: the train-PSNR slope stop cut
                // driveway at 36k for -0.12 dB vs a full 45k (ABBA 2v2, 30.24 vs 30.36); the slope
                // stays only as the fallback for captures too small to hold out frames.
                let validation_fire = if validation_views.is_empty() {
                    false
                } else {
                    let t = Instant::now();
                    let val_cams = trainer.pose_refine().map(|p| validation_near.iter().zip(&validation_views)
                        .map(|(&i, v)| p.refined_camera(i, &v.camera)).collect::<Vec<_>>());
                    let (psnr, _) = score_with(&splats, &validation_views, val_cams.as_deref(), &device, None).await?;
                    validation.push((step, psnr));
                    let gain = validation_gain(&validation);
                    emit(&job.out, &json!({"event": "validation", "step": step, "validation_psnr": psnr, "gain_db": gain,
                        "threshold_db": VALIDATION_GAIN_DB, "n_validation": validation_views.len(), "eval_s": t.elapsed().as_secs_f64()}));
                    gain.is_some_and(|g| g <= VALIDATION_GAIN_DB)
                };
                let fire = if validation_views.is_empty() { slope_fire } else { validation_fire };
                if fire {
                    if let Some(tail_start) = trainer.enter_qat_tail() {
                        if let (Some(path), true) = (&job.fork_ply, fork.is_none()) {
                            fork = Some(write_fork(&job.out, path, &splats, job.fork_min_splats, step, up_axis).await);
                        }
                        emit(&job.out, &json!({"event": "early_stop", "step": step, "tail_start": tail_start,
                            "tail_steps": job.total_steps - tail_start, "actual_steps": actual_steps,
                            "signal": if validation_fire { "validation_psnr" } else { "train_psnr" }}));
                        step = tail_start;
                        early_stopped = true;
                    }
                }
            }
        }
        // The QAT tail (reached early or on schedule) trains on the validation frames too.
        if !validation_views.is_empty() && fit_views.len() < train_views.len() && step >= job.total_steps - job.total_steps / 20 {
            fit_views = train_views.clone();
            trainer.set_pose_loader_views((0..train_views.len()).collect());
            loader = SceneLoader::new(&scaled_views(&fit_views, image_scale), job.seed + step as u64, &load_config);
            emit(&job.out, &json!({"event": "validation_rejoined", "step": step, "n_train": fit_views.len()}));
        }
    }
    let training_s = train_start.elapsed().as_secs_f64();
    settle_fork(&job, fork);
    if job.preview {
        engine::save_ply(splats.clone(), up_axis, &job.out.join("splat.ply")).await?;
        let summary = json!({"event":"done", "preview":true, "step":step, "splats":splats.num_splats(),
            "training_s":training_s, "wall_s":wall.elapsed().as_secs_f64()});
        write_json(&job.out.join("train_summary.json"), &summary)?;
        emit(&job.out, &summary);
        join_web_package(spawn_web_package());
        return Ok(0);
    }
    let _ = device.sync();
    trainer.finish_training();
    device.memory_cleanup();
    let tail = Instant::now();
    save_checkpoint(splats.clone(), step, job.total_steps).await?;
    let saved_s = tail.elapsed().as_secs_f64();
    // The final checkpoint is the model: link it rather than encode and write the same ~250
    // bytes per splat again (1.1 GB at 4.6M splats).
    let splat_ply = job.out.join("splat.ply");
    let _ = std::fs::remove_file(&splat_ply);
    if std::fs::hard_link(job.out.join("checkpoint.ply"), &splat_ply).is_err() {
        std::fs::copy(job.out.join("checkpoint.ply"), &splat_ply)?;
    }
    // The model is final: the fal runner starts staging it now, beside the scoring and web package below.
    emit(&job.out, &json!({"event": "model_saved", "step": step}));
    let dropped = brush_render::render::surfel_lens_dropped();
    if dropped > 0 {
        emit(&job.out, &json!({"event": "surfel_lens_dropped", "renders": dropped}));
    }
    // The web package needs only splat.ply: build it (CPU) while the held-out views are
    // scored (GPU) instead of after.
    let web = spawn_web_package();

    let eval_dir = job.out.join("eval");
    for d in ["gt", "renders"] {
        std::fs::create_dir_all(eval_dir.join(d))?;
    }
    // Test-time pose alignment of the held-out views (splats frozen) is the evaluate job's: both
    // arms of a pose-refine A/B must run it, or the refined arm's gauge drift is scored as error.
    // It ran here until 10-10 and cost ~40 s at the end of a driveway MAX job, which delayed the
    // job's results (the user's export). This job scores and saves unaligned renders and records
    // what evaluate needs (pose_refine.json plus these settings). Warm-start jobs (the interactive
    // refine stage) are not the scored candidate: no alignment.
    let warm_start = job.init_ply.is_some() && !job.fresh_init_ply;
    let align = (config.pose_align_eval_steps > 0 && !eval_views.is_empty() && !job.surfel && !job.neural_max && !warm_start)
        .then(|| json!({"steps": config.pose_align_eval_steps, "extent": trainer.cam_extent(), "ssim_weight": config.ssim_weight,
            "dataset": job.dataset, "max_resolution": job.max_resolution}));
    let (psnr, ssim) = score(&splats, &eval_views, &device, Some(&eval_dir)).await?;
    if let Some(p) = trainer.pose_refine() {
        write_json(&job.out.join("pose_refine.json"), &pose_refine_report(p, &train_views, trainer.cam_extent()))?;
    }
    // Train PSNR on up to 8 evenly spaced training views: the held-out gap flags overfitting.
    let stride = (train_views.len() / 8).max(1);
    let sample: Vec<SceneView> = train_views.iter().step_by(stride).take(8).cloned().collect();
    let (train_psnr, train_ssim) = score(&splats, &sample, &device, None).await?;
    // With pose refinement the training views' own (refined) cameras, as trained.
    let train_psnr_refined = match trainer.pose_refine() {
        Some(p) => {
            let cams: Vec<_> = (0..train_views.len()).step_by(stride).take(8).map(|i| p.refined_camera(i, &train_views[i].camera)).collect();
            Some(score_with(&splats, &sample, Some(&cams), &device, None).await?.0)
        }
        None => None,
    };
    let summary = json!({
        "event": "done", "step": step, "total_steps": job.total_steps, "actual_steps": actual_steps, "early_stopped": early_stopped, "splats": splats.num_splats(),
        "heldout_psnr_brush": psnr, "heldout_ssim_brush": ssim,
        "train_psnr": train_psnr, "train_ssim": train_ssim, "n_train_sampled": sample.len(),
        "training_s": training_s, "wall_s": wall.elapsed().as_secs_f64(),
        "pose_refine": trainer.pose_refine().is_some(), "train_psnr_refined": train_psnr_refined,
        // Held-out alignment for evaluate (see views::evaluate); the scores above are unaligned.
        "heldout_pose_align": align,
        "heldout_psnr_unaligned": align.as_ref().map(|_| psnr),
    });
    write_json(&job.out.join("train_summary.json"), &summary)?;
    emit(&job.out, &summary);
    let scored_s = tail.elapsed().as_secs_f64();
    if job.defer_lpips {
        // Publishable results need only these small JPEGs, not every GT/render PNG.
        // The PNGs stay staged on fal and evaluate reuses them at full resolution.
        let (pairs, _) = splat_eval::pair_dirs(&eval_dir.join("gt"), &eval_dir.join("renders"))?;
        let renders: Vec<PathBuf> = pairs.iter().take(8).map(|p| p.render.clone()).collect();
        let previews = splat_eval::budget_jpegs(&renders, &job.out.join("previews"), splat_eval::RESULT_IMAGE_BUDGET)?;
        let names: Vec<_> = previews.iter().map(|p| p.strip_prefix(&job.out).map(|p| p.to_string_lossy().into_owned())).collect::<Result<_, _>>()?;
        write_json(&job.out.join("previews.json"), &names)?;
    }
    let previews_s = tail.elapsed().as_secs_f64();
    join_web_package(web);
    eprintln!("final phases: checkpoint {saved_s:.1}s, scored {scored_s:.1}s, previews {previews_s:.1}s, web joined {:.1}s",
        tail.elapsed().as_secs_f64());
    Ok(0)
}

/// The pseudo views of a fill fine-tune ([`PseudoViews`]): their own loader beside the real one,
/// and the per-step choice between them.
struct PseudoLoader {
    cfg: PseudoViews,
    views: Vec<PseudoView>,
    loader: SceneLoader,
    rng: rand::rngs::StdRng,
    /// Pseudo batches carry view indices past the real views' (pose refine, blur caches and
    /// transient masks key on the index).
    index_offset: usize,
}

impl PseudoLoader {
    async fn new(cfg: &PseudoViews, load_config: &brush_dataset::config::LoadDatasetConfig, seed: u64, index_offset: usize) -> anyhow::Result<Self> {
        use brush_render::camera::focal_to_fov;
        use brush_render::kernels::camera_model::CameraModel;
        anyhow::ensure!((0.0..=1.0).contains(&cfg.frac), "pseudo frac must be in [0, 1]");
        anyhow::ensure!(cfg.weight >= 0.0 && cfg.pos_lr >= 0.0 && cfg.ramp >= 0.0, "pseudo weight, pos_lr and ramp must be >= 0");
        anyhow::ensure!(cfg.stop.is_none_or(|s| (0.0..=1.0).contains(&s)), "pseudo stop must be in [0, 1]");
        #[derive(Deserialize)]
        struct ViewsFile {
            views: Vec<PseudoView>,
        }
        let file: ViewsFile = read_json(&cfg.dir.join("views.json"))?;
        anyhow::ensure!(!file.views.is_empty(), "{}: no pseudo views", cfg.dir.display());
        let vfs = Arc::new(brush_vfs::BrushVfs::from_path(&cfg.dir).await.context("mounting the pseudo-view dir")?);
        let scene_views = file.views.iter().map(|v| {
            let c2w = glam::Mat4::from_cols_array_2d(&v.c2w).transpose();
            let (_, rotation, position) = c2w.to_scale_rotation_translation();
            let fov_x = focal_to_fov(v.fx as f64, v.width, &CameraModel::Pinhole);
            let fov_y = focal_to_fov(v.fy as f64, v.height, &CameraModel::Pinhole);
            let center = glam::vec2(v.cx / v.width as f32, v.cy / v.height as f32);
            let image = brush_dataset::scene::LoadImage::new(vfs.clone(), PathBuf::from(&v.image), None, load_config.max_resolution,
                Some(brush_render::AlphaMode::Masked), false);
            SceneView { image, camera: brush_render::camera::Camera::new(position, rotation.normalize(), fov_x, fov_y, center, CameraModel::Pinhole),
                camera_index: 0, depth: None }
        }).collect();
        let loader = SceneLoader::new(&Scene::new(scene_views), seed ^ 0x5eed_f111, load_config);
        Ok(Self { cfg: cfg.clone(), views: file.views, loader, rng: rand::rngs::StdRng::seed_from_u64(seed ^ 0xf111), index_offset })
    }

    /// With probability `frac` (none from `stop` x `total` on), the next pseudo batch and its step settings.
    async fn pick(&mut self, step: u32, total: u32) -> Option<(brush_dataset::scene::SceneBatch, brush_train::train::PseudoStep)> {
        use rand::RngExt;
        if self.cfg.stop.is_some_and(|s| step as f32 >= s * total as f32) {
            return None;
        }
        if self.rng.random::<f32>() >= self.cfg.frac {
            return None;
        }
        let mut batch = self.loader.next_batch().await;
        let v = &self.views[batch.view_index];
        let ramp = if self.cfg.ramp > 0.0 {
            (0.5 * std::f32::consts::PI * (step as f32 / (self.cfg.ramp * total as f32)).min(1.0)).sin()
        } else {
            1.0
        };
        let weight = self.cfg.weight * v.weight * ramp;
        // The trainer's pseudo loss wants the GT premultiplied by the confidence (opaque) and the
        // confidence itself as a separate mask.
        let [h, w] = batch.img_size();
        let packed: Vec<i32> = batch.img_packed.try_to_vec::<i32>().map_err(|e| anyhow::anyhow!("pseudo image: {e:?}")).ok()?;
        let mut mask = Vec::with_capacity(packed.len());
        let premult: Vec<i32> = packed.iter().map(|&p| {
            let [r, g, b, a] = p.to_le_bytes();
            mask.push(a as f32 / 255.0);
            let mul = |c: u8| ((c as u16 * a as u16 + 127) / 255) as u8;
            i32::from_le_bytes([mul(r), mul(g), mul(b), 255])
        }).collect();
        let mask_mean = mask.iter().sum::<f32>() / mask.len().max(1) as f32;
        batch.img_packed = burn::tensor::TensorData::new(premult, [h, w]);
        batch.has_alpha = false;
        batch.alpha_mode = brush_render::AlphaMode::Transparent;
        batch.view_index += self.index_offset;
        Some((batch, brush_train::train::PseudoStep { weight, pos_lr: self.cfg.pos_lr,
            mask: burn::tensor::TensorData::new(mask, [h, w, 1]), mask_mean }))
    }
}

/// Set by the `splat` binary: writes the finished candidate's web package (`<out>/web`) from
/// splat.ply and train_summary.json. [`run_job`] runs it on a thread beside the held-out
/// scoring (CPU export, GPU scoring) and waits for it before the worker exits.
pub static WEB_PACKAGE: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();

fn spawn_web_package() -> Option<std::thread::JoinHandle<()>> {
    WEB_PACKAGE.get().map(|f| std::thread::spawn(move || f()))
}

fn join_web_package(web: Option<std::thread::JoinHandle<()>>) {
    if let Some(Err(_)) = web.map(std::thread::JoinHandle::join) {
        eprintln!("web package: panicked");
    }
}

/// Held-out metrics (PSNR/SSIM/LPIPS) of the trainer's final eval renders.
fn score_heldout(dir: &Path, models_dir: Option<PathBuf>) -> anyhow::Result<splat_eval::EvalSummary> {
    let eval = dir.join("eval");
    let (pairs, missing) = splat_eval::pair_dirs(&eval.join("gt"), &eval.join("renders"))?;
    let opts = splat_eval::MetricOptions { lpips: true, models_dir, cc_lpips: false, ..Default::default() };
    splat_eval::evaluate_pairs(&pairs, &missing, &opts)
}

/// Test-time pose alignment steps per held-out view when pose refine is on.
pub(crate) const POSE_ALIGN_DEFAULT_STEPS: u32 = 200;
/// Auto pose refine (photo sets): the largest expected splat budget it switches on for. Its per-step cost
/// over in-exec steady-state pairs grows with the splat count, fit about -3.0% + 2.42% per million splats:
/// +1.7% room-every4 (1.91M), +1.9% ETH3D delivery (2.05M), +7.0% driveway (4.64M), +9.7% garden (4.79M),
/// ~2.4% here. ETH3D's budgets (0.68-2.17M) are under it; unbounded M360 photo sets (to 5M) over.
const POSE_REFINE_AUTO_MAX_SPLATS: u32 = 2_250_000;

/// Pose-only photometric BA (`pose_ba_step`): with the splats frozen, each training view's rigid
/// correction is aligned to its photo (`pose_ba_iters` Adam steps from its current correction, at
/// the training's current image scale) and stored; training then continues on the corrected poses
/// with refinement off. Returns the event fields (time, loss and correction size).
async fn pose_ba(trainer: &mut SplatTrainer, splats: &Splats, views: &[SceneView], config: &TrainConfig, image_scale: f32)
    -> anyhow::Result<Value> {
    use brush_train::pose_refine::{AlignLr, align_view, rigid_delta};
    let t = Instant::now();
    let train_splats = splats.clone().train();
    let extent = trainer.cam_extent();
    let lr = AlignLr { rot: config.pose_ba_lr_rot as f32, trans: config.pose_ba_lr_trans as f32 };
    let (mut loss_before, mut loss_after) = (0.0f64, 0.0f64);
    let mut rot: Vec<f32> = Vec::with_capacity(views.len());
    for (i, view) in views.iter().enumerate() {
        let Some(p) = trainer.pose_refine() else { break };
        let (base, init) = (p.view_focal_local(i), p.correction(i));
        let mut scaled = view.clone();
        if image_scale < 1.0 {
            scaled.image = scaled.image.with_scale(image_scale);
        }
        let gt = engine::load_gt(&scaled).await?;
        let a = align_view(&train_splats, &view.camera, base, init, &gt, engine::gt_alpha_mode(view),
            config.pose_ba_iters, lr, extent, config.ssim_weight).await?;
        if a.first_loss.is_finite() && a.last_loss.is_finite() {
            loss_before += a.first_loss as f64;
            loss_after += a.last_loss as f64;
        }
        rot.push(rigid_delta(&a.correction).0);
        if let Some(p) = trainer.pose_refine_mut() {
            p.set_correction(i, a.correction);
        }
    }
    rot.sort_by(f32::total_cmp);
    let n = views.len().max(1) as f64;
    Ok(json!({"views": views.len(), "iters": config.pose_ba_iters, "image_scale": image_scale, "ba_s": t.elapsed().as_secs_f64(),
        "loss_before": loss_before / n, "loss_after": loss_after / n,
        "rot_deg_median": rot.get(rot.len() / 2), "rot_deg_max": rot.last()}))
}

/// Per training view: the refined pose's change (rotation degrees, camera-centre translation in
/// scene units and camera extents) and the refined COLMAP world-to-camera (`qvec` w,x,y,z, `tvec`),
/// for pose-accuracy scoring (e.g. ETH3D AUC@3°) after refinement.
fn pose_refine_report(p: &brush_train::pose_refine::PoseRefine, views: &[SceneView], extent: f32) -> Value {
    use brush_train::pose_refine::{refined_w2c, rigid_delta};
    let corrections = p.corrections();
    let mut rot: Vec<f32> = Vec::with_capacity(views.len());
    let mut trans: Vec<f32> = Vec::with_capacity(views.len());
    let rows: Vec<Value> = views.iter().zip(&corrections).map(|(v, d)| {
        let (r, t) = rigid_delta(d);
        rot.push(r);
        trans.push(t);
        let (qvec, tvec) = refined_w2c(&v.camera, d);
        json!({"view": engine::view_name(v), "camera_index": v.camera_index, "rot_deg": r, "trans": t,
            "trans_rel": t / extent, "qvec": qvec, "tvec": tvec})
    }).collect();
    let stat = |mut x: Vec<f32>| {
        x.sort_by(f32::total_cmp);
        let q = |f: f64| x.get(((x.len().max(1) - 1) as f64 * f).round() as usize).copied();
        json!({"median": q(0.5), "p90": q(0.9), "max": x.last().copied()})
    };
    json!({"extent": extent, "updates": p.updates, "focal_scales": p.focal_scales(),
        "rot_deg": stat(rot), "trans": stat(trans), "views": rows})
}

/// Seconds of a remote job's wall limit kept free of training (see `TrainJob::time_budget_s`).
const TIME_BUDGET_MARGIN_S: f64 = 300.0;

/// Where a cloud MAX training of `candidate` leaves its fork model for the forked
/// interactive refine: fal's shared /data volume, which every runner of the account mounts.
fn fork_path(run: &dyn RunCtx, candidate: &str) -> PathBuf {
    PathBuf::from(format!("/data/splat-cloud/forks/{}/{candidate}.ply", run.run_id()))
}
/// MAX steps once blur mode switches on (see `TrainJob::blur_total_steps`).
const BLUR_MAX_STEPS: u32 = 20_000;

/// Worker exit code when the loss goes non-finite or explodes.
const EXIT_DIVERGED: i32 = 3;
/// Loss above this is divergence (L1 + SSIM terms stay around 0-2).
const DIVERGED_LOSS: f32 = 100.0;

pub fn complete(cfg: &BrushConfig, run: &dyn RunCtx, artifact_id: &str, exit_code: i32, log: &str) -> anyhow::Result<ToolResult> {
    let (artifact, dir) = get_artifact(run, artifact_id, ArtifactKind::Candidate)?;
    let checkpoint: Option<Checkpoint> = read_json(&dir.join("checkpoint.json")).ok();
    if exit_code != 0 || !dir.join("splat.ply").exists() {
        let code = if exit_code == 130 || exit_code == -2 {
            "interrupted"
        } else if exit_code == EXIT_DIVERGED {
            "diverged"
        } else if log.to_lowercase().contains("out of memory") {
            "oom"
        } else {
            "subprocess_failed"
        };
        let message = if code == "diverged" {
            format!("training diverged (loss became non-finite or exploded): {log}. Check poses first (flipped or drifting cameras), then resume from the last checkpoint or lower resolution.")
        } else {
            format!("training exited {exit_code}: {log}")
        };
        let mut r = failure("train_splats", code, message);
        if let Some(ck) = &checkpoint {
            r.metrics.insert("checkpoint".into(), json!(rel(run, &dir.join(&ck.ply))));
            r.metrics.insert("checkpoint_step".into(), json!(ck.step));
            r.metrics.insert("resumable".into(), json!(true));
            r.warnings.push(format!("Resume with resume_candidate_id={artifact_id}; up to 499 steps may be lost."));
        }
        r.artifacts = vec![artifact];
        return Ok(r);
    }
    let summary: serde_json::Map<String, Value> = read_json(&dir.join("train_summary.json"))?;
    if artifact.meta["request"]["preview"] == true {
        return Ok(ToolResult {
            artifacts: vec![artifact], metrics: summary.into_iter().collect(),
            ..result("train_splats", Status::Succeeded)
        });
    }
    let eval = dir.join("eval");
    let (pairs, missing) = splat_eval::pair_dirs(&eval.join("gt"), &eval.join("renders"))?;
    // A `--cloud` worker left LPIPS to the evaluate job on fal (its renders are stubs here): report
    // the trainer's own held-out PSNR/SSIM and never score on the user's CPU. Local jobs score here.
    let deferred = read_json::<Value>(&dir.join("train_job.json")).is_ok_and(|j| j["defer_lpips"] == true);
    let mut metrics = serde_json::Map::new();
    if deferred && !dir.join("metrics.json").is_file() {
        metrics.insert("psnr".into(), summary.get("heldout_psnr_brush").cloned().unwrap_or(Value::Null));
        metrics.insert("ssim".into(), summary.get("heldout_ssim_brush").cloned().unwrap_or(Value::Null));
        metrics.insert("heldout_lpips".into(), json!("evaluate"));
        metrics.insert("n_eval".into(), json!(pairs.len()));
    } else {
        let scores: splat_eval::EvalSummary = match read_json(&dir.join("metrics.json")) {
            Ok(scores) => scores,
            Err(_) => {
                let scores = score_heldout(&dir, cfg.models_dir.clone())?;
                write_json(&dir.join("metrics.json"), &scores)?;
                scores
            }
        };
        metrics.extend(scores.mean.iter().map(|(k, v)| (k.clone(), f64_metric(*v))));
        if let Some(why) = &scores.lpips_unavailable {
            metrics.insert("lpips_unavailable".into(), json!(why));
        }
        if let Some(note) = &scores.lpips_note {
            metrics.insert("lpips_note".into(), json!(note));
        }
        if let Some(scale) = &scores.lpips_scale {
            metrics.insert("lpips_scale".into(), json!(scale));
        }
        metrics.insert("n_eval".into(), json!(scores.n));
    }
    let lock_wait = std::fs::read_to_string(dir.join("training.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|r| r["event"] == "gpu_lock")
        .map(|r| r["gpu_lock_wait_s"].clone());
    metrics.insert("gpu_lock_wait_s".into(), lock_wait.unwrap_or(Value::Null));
    for k in ["train_psnr", "train_ssim", "training_s", "wall_s"] {
        metrics.insert(k.into(), summary.get(k).cloned().unwrap_or(Value::Null));
    }
    let (train, held) = (summary["train_psnr"].as_f64(), metrics.get("psnr").and_then(Value::as_f64));
    let gap = train.zip(held).map(|(t, h)| t - h);
    metrics.insert("train_heldout_psnr_gap".into(), json!(gap));
    metrics.insert("split_verified".into(), json!(missing.is_empty()));
    metrics.insert("exported_splats".into(), summary["splats"].clone());
    metrics.insert("splat_ply".into(), json!(rel(run, &dir.join("splat.ply"))));
    metrics.insert("checkpoint".into(), json!(rel(run, &dir.join("checkpoint.ply"))));
    metrics.insert("training_series".into(), json!(rel(run, &dir.join("training.jsonl"))));
    let blur_mode = std::fs::read_to_string(dir.join("training.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .any(|r| r["event"] == "blur_mode");
    metrics.insert("blur_mode".into(), json!(blur_mode));
    let mut warnings = vec![];
    if blur_mode {
        warnings.push(brush_train::train::BLUR_MODE_NOTE.into());
    }
    if let Some(total) = summary.get("total_steps").and_then(Value::as_u64).filter(|&t| summary["step"].as_u64().is_some_and(|s| s < t)) {
        warnings.push(format!(
            "Training stopped at step {} of {total}: the job's time limit was near. The result is the splats at that step.",
            summary["step"]
        ));
    }
    if gap.is_some_and(|g| g > 6.0) {
        warnings.push(format!("Train PSNR exceeds held-out by {:.1} dB: overfitting or poor view coverage.", gap.unwrap()));
    }
    let renders: Vec<PathBuf> = pairs.iter().take(8).map(|p| p.render.clone()).collect();
    let images = match read_json::<Vec<String>>(&dir.join("previews.json")) {
        Ok(names) => names.into_iter().map(|name| dir.join(name)).collect(),
        Err(_) => splat_eval::budget_jpegs(&renders, &dir.join("previews"), splat_eval::RESULT_IMAGE_BUDGET)?,
    };
    Ok(ToolResult {
        artifacts: vec![artifact],
        metrics: metrics.into_iter().collect(),
        images: images.iter().map(|p| rel(run, p)).collect(),
        warnings,
        ..result("train_splats", Status::Succeeded)
    })
}

/// Latest progress row from the worker's stdout (job_dir/stdout.log).
pub fn progress(job_dir: &Path) -> Value {
    let log = tail_file(&job_dir.join("stdout.log"), 64 * 1024);
    let mut out = serde_json::Map::new();
    for line in log.lines() {
        let Some(row) = line.strip_prefix("PROGRESS ") else { continue };
        let Ok(Value::Object(row)) = serde_json::from_str::<Value>(row) else { continue };
        match row.get("event").and_then(Value::as_str) {
            Some("eval") => {
                out.insert("heldout_psnr".into(), row["psnr"].clone());
                out.insert("heldout_ssim".into(), row["ssim"].clone());
                out.insert("eval_step".into(), row["step"].clone());
            }
            _ => {
                for (k, v) in row {
                    if k != "event" {
                        out.insert(k, v);
                    }
                }
            }
        }
    }
    Value::Object(out)
}

/// Pixels one view trains at: its native size, scaled down to fit
/// `max_resolution` on the long side (0 = native).
fn train_pixel_count(w: u32, h: u32, max_resolution: u32) -> u64 {
    let long = w.max(h).max(1) as f64;
    let scale = if max_resolution > 0 && long > max_resolution as f64 {
        max_resolution as f64 / long
    } else {
        1.0
    };
    (w as f64 * scale * h as f64 * scale) as u64
}

/// Validation set: VALIDATION_PERCENT of the training frames, clamped to
/// [VALIDATION_MIN, VALIDATION_MAX]; smaller captures keep the train-PSNR stop.
const VALIDATION_PERCENT: usize = 3;
const VALIDATION_MIN: usize = 4;
const VALIDATION_MAX: usize = 24;
const VALIDATION_MIN_TRAIN: usize = 60;
/// Stop once the best validation PSNR of the last 2 checks (6k steps) beats the best
/// before them by no more than this.
const VALIDATION_GAIN_DB: f64 = 0.05;

/// Gain of the best of the last 2 validation checks over the best earlier one.
fn validation_gain(checks: &[(u32, f64)]) -> Option<f64> {
    let best = |c: &[(u32, f64)]| c.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    (checks.len() >= 3).then(|| best(&checks[checks.len() - 2..]) - best(&checks[..checks.len() - 2]))
}

/// Convergence checks the early-stop slope is fitted over (3k steps apart).
const CONVERGENCE_FIT: usize = 3;

/// Least-squares slope of PSNR over step.
fn fit_slope(points: &[(u32, f64)]) -> f64 {
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0 as f64).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let cov: f64 = points.iter().map(|p| (p.0 as f64 - mean_x) * (p.1 - mean_y)).sum();
    let var: f64 = points.iter().map(|p| (p.0 as f64 - mean_x).powi(2)).sum();
    if var > 0.0 { cov / var } else { 0.0 }
}
