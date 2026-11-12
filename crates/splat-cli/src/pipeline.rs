use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use splat_contracts::{ArtifactKind, Status, ToolResult};
use splat_core::{Run, RunCtx, dispatch, report::write_report};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityPreset {
    Fast,
    Max,
}

#[derive(Debug, Clone, Default, clap::Args)]
pub struct PresetFlags {
    /// Quick preview only (14k steps, 1M splats, 1600 px); trades detail and evaluation for speed.
    #[arg(long, conflicts_with = "max")]
    pub fast: bool,
    /// Full-quality budget; on --cloud this is already the default.
    #[arg(long, conflicts_with = "fast")]
    pub max: bool,
}

impl PresetFlags {
    pub fn apply(&self, mut options: FixedOptions) -> FixedOptions {
        if self.fast {
            options.preset = Some(QualityPreset::Fast);
            options.steps = 7000;
            options.cap_max = 250_000;
            options.cap_max_explicit = true;
            options.max_edge = 800;
            options.no_mesh = true;
        } else if self.max {
            options.preset = Some(QualityPreset::Max);
            // train_splats raises this to the MAX floor for the capture's size.
            options.steps = splat_core::remote::SCALED_STEPS_MIN;
            options.resolution_divisor = 1;
            // The MAX recipe sizes its own budget unless --cap-max overrides it.
            options.cap_max_explicit = false;
        }
        if self.fast || self.max {
            options.steps_explicit = false;
        }
        options
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FixedOptions {
    pub steps: u32,
    pub target_count: u32,
    pub max_edge: u32,
    pub cap_max: u32,
    pub resolution_divisor: u32,
    pub no_mesh: bool,
    pub no_lpips: bool,
    #[serde(default)]
    pub preset: Option<QualityPreset>,
    #[serde(default)]
    pub steps_explicit: bool,
    #[serde(default)]
    pub cap_max_explicit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_every: Option<u32>,
}
impl Default for FixedOptions {
    fn default() -> Self {
        Self {
            steps: 7000,
            target_count: 600,
            max_edge: 1600,
            cap_max: 1_000_000,
            resolution_divisor: 1,
            no_mesh: false,
            no_lpips: false,
            preset: None,
            steps_explicit: false,
            cap_max_explicit: false,
            seed: None,
            eval_every: None,
        }
    }
}

impl FixedOptions {
    pub fn is_fast(&self) -> bool {
        self.preset == Some(QualityPreset::Fast)
    }

    fn training_request(&self, request: &mut Value) {
        request["total_steps_explicit"] = json!(self.steps_explicit);
        request["max_quality"] = json!(self.preset == Some(QualityPreset::Max));
        request["preview"] = json!(self.is_fast());
        if self.cap_max_explicit {
            request["cap_max"] = json!(self.cap_max);
            request["cap_max_explicit"] = json!(true);
        }
        if let Some(seed) = self.seed {
            request["seed"] = json!(seed);
        }
        if let Some(every) = self.eval_every {
            request["eval_every"] = json!(every);
        }
        // Cloud video MAX also leaves its end-of-densification model for the interactive refine
        // forked beside it (see `fork_refine`); the tool decides whether this training qualifies.
        if splat_core::remote::mode().is_some() && !self.is_fast() {
            request["fork_min_splats"] = json!(INTERACTIVE_SPLATS);
        }
    }

    /// Called before worker threads start; cloud.json records the selected preset for resume.
    pub fn configure_cloud_preset(&self) {
        let name = match self.preset {
            Some(QualityPreset::Fast) => "default",
            Some(QualityPreset::Max) => splat_core::remote::BUDGET_PRESET.name,
            None => return,
        };
        // SAFETY: the CLI calls this on the main thread before starting the driver or workers.
        unsafe { std::env::set_var("SPLAT_CLOUD_PRESET", name) };
    }
}
#[derive(Debug, Clone, Default, clap::Args)]
pub struct FixedOverrides {
    /// Training steps; overrides --fast/--max. Default run: 7000 (cloud may raise it).
    #[arg(long)]
    pub steps: Option<u32>,
    #[arg(long)]
    pub target_count: Option<u32>,
    #[arg(long)]
    pub max_edge: Option<u32>,
    /// Splat cap; overrides --fast/--max and the cloud budget.
    #[arg(long)]
    pub cap_max: Option<u32>,
    #[arg(long)]
    pub resolution_divisor: Option<u32>,
    #[arg(long, num_args=0..=1, default_missing_value="true")]
    pub no_mesh: Option<bool>,
    #[arg(long, num_args=0..=1, default_missing_value="true")]
    pub no_lpips: Option<bool>,
    /// Training RNG seed, for repeatable runs (default 42): the main training and its interactive refine;
    /// previews keep 42. Kept for `resume`.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Score the held-out views every N training steps (default: steps / 5, 500-5000): the main training
    /// and its interactive refine; previews score once at the end. Kept for `resume`.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub eval_every: Option<u32>,
}
impl FixedOverrides {
    pub fn apply(self, mut options: FixedOptions) -> FixedOptions {
        if let Some(v) = self.steps {
            options.steps = v;
            options.steps_explicit = true;
        }
        if let Some(v) = self.target_count {
            options.target_count = v;
        }
        if let Some(v) = self.max_edge {
            options.max_edge = v;
        }
        if let Some(v) = self.cap_max {
            options.cap_max = v;
            options.cap_max_explicit = true;
        }
        if let Some(v) = self.resolution_divisor {
            options.resolution_divisor = v;
        }
        if let Some(v) = self.no_mesh {
            options.no_mesh = v;
        }
        if let Some(v) = self.no_lpips {
            options.no_lpips = v;
        }
        if let Some(v) = self.seed {
            options.seed = Some(v);
        }
        if let Some(v) = self.eval_every {
            options.eval_every = Some(v);
        }
        options
    }
}
fn completed(run: &Run, name: &str, input: Value) -> Result<ToolResult> {
    let mut r = dispatch(run, name, input);
    while r.status == Status::Running {
        r = run
            .jobs()
            .wait(r.job_id.as_deref().context("Missing job_id")?, 0.25)?;
        ensure!(
            !matches!(run.state()?.status.as_str(), "stopping" | "cancelled"),
            "Run stopped"
        );
    }
    Ok(r)
}
/// How [`completed_within`] ended.
enum Bounded {
    /// The job ended (possibly after the stop, still `succeeded`).
    Done(ToolResult),
    /// Over budget: stopped and ended without succeeding.
    Stopped,
    /// Over budget and still running `stop_wait_s` after the stop: abandoned (journal `job_abandoned`), so
    /// `finish`, which refuses while a job runs, can proceed.
    Abandoned,
}

/// [`completed`], but stop the job after `budget_s` of waiting. The stop is joined for up to
/// SPLAT_MESH_STOP_WAIT_S (default 120 s): a remote job may not stop at once (sai-pixel5-plant's mesh
/// worker was fetching results for 200 s and ran on past a 60 s stop wait, and finish failed the run).
fn completed_within(run: &Run, name: &str, input: Value, budget_s: f64) -> Result<Bounded> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(budget_s);
    let mut r = dispatch(run, name, input);
    while r.status == Status::Running {
        let id = r.job_id.clone().context("Missing job_id")?;
        if std::time::Instant::now() >= deadline {
            let stop_wait = std::env::var("SPLAT_MESH_STOP_WAIT_S").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(120.0);
            let joined = std::time::Instant::now() + std::time::Duration::from_secs_f64(stop_wait.max(0.0));
            r = run.jobs().stop(&id, stop_wait.clamp(0.0, 60.0))?;
            while r.status == Status::Running && std::time::Instant::now() < joined {
                r = run.jobs().wait(&id, 5.)?;
            }
            return Ok(match r.status {
                Status::Running => {
                    run.journal("job_abandoned", json!({"job_id":id,"tool":name,"stop_wait_s":stop_wait}))?;
                    Bounded::Abandoned
                }
                Status::Succeeded => Bounded::Done(r),
                _ => Bounded::Stopped,
            });
        }
        r = run.jobs().wait(&id, 0.25)?;
        ensure!(
            !matches!(run.state()?.status.as_str(), "stopping" | "cancelled"),
            "Run stopped"
        );
    }
    Ok(Bounded::Done(r))
}
fn invoke(run: &Run, name: &str, input: Value) -> Result<ToolResult> {
    let r = completed(run, name, input)?;
    ensure!(
        r.status == Status::Succeeded,
        "{name}: {}",
        r.error
            .as_ref()
            .map(|e| e.message.clone())
            .unwrap_or_else(|| format!("{:?}", r.status))
    );
    Ok(r)
}
fn artifact(r: ToolResult, kind: ArtifactKind) -> Result<String> {
    r.artifacts
        .into_iter()
        .find(|a| a.kind == kind)
        .map(|a| a.id)
        .context("Expected artifact missing")
}

pub fn fixed_pipeline(run: &Run, options: &FixedOptions) -> Result<Value> {
    reconstruction_pipeline(run, options, false)
}

/// The MCP entry uses the fixed driver's implementation, leaving goal-specific extras and the
/// single finish verdict to the agent. CLI presets/overrides are already resolved in fixed.json.
pub fn run_default_pipeline(run: &Run, input: Value) -> Result<ToolResult> {
    let overrides: splat_contracts::RunDefaultPipeline = serde_json::from_value(input)?;
    let path = run.root().join("fixed.json");
    let options = if path.exists() {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        FixedOptions::default()
    };
    let options = FixedOverrides {
        steps: overrides.steps,
        target_count: overrides.target_count,
        max_edge: overrides.max_edge,
        cap_max: overrides.cap_max,
        resolution_divisor: overrides.resolution_divisor,
        no_lpips: overrides.no_lpips,
        no_mesh: None,
        seed: None,
        eval_every: None,
    }.apply(options);
    ensure!(options.steps > 0 && options.target_count > 0 && options.max_edge > 0
        && options.cap_max > 0 && options.resolution_divisor > 0, "Pipeline budgets must be positive");
    let started = std::time::Instant::now();
    run.journal("default_pipeline_started", json!({"options":options}))?;
    let output = reconstruction_pipeline(run, &options, true)?;
    let mut result: ToolResult = serde_json::from_value(json!({
        "schema_version":splat_contracts::SCHEMA_VERSION, "tool":"run_default_pipeline",
        "status":"succeeded", "job_id":null, "error":null, "wall_s":started.elapsed().as_secs_f64(),
        "metrics":{"candidate_id":output["candidate_id"], "evaluated":!options.is_fast(),
            "options":options, "evaluation":output["evaluation"]["metrics"]},
        "images":output["evaluation"]["images"].as_array().cloned().unwrap_or_default(),
    }))?;
    result.artifacts = run.state()?.artifacts.into_values().collect();
    if options.is_fast() {
        result.warnings.push("Fast preview: quality has not been evaluated; no mesh was generated.".into());
    }
    run.journal("default_pipeline_completed", serde_json::to_value(&result)?)?;
    Ok(result)
}

fn reconstruction_pipeline(run: &Run, options: &FixedOptions, agent_continues: bool) -> Result<Value> {
    let wants_preview = splat_core::remote::preset().is_some()
        && run.config().backend == "brush"
        && !splat_core::remote::goal_wants_mesh(run.goal());
    let inspection = invoke(run, "inspect_input", json!({}))?;
    // Videos: their full SfM takes minutes (night 6+). Photos from PHOTO_QUICK_MIN up: garden's (185)
    // full SfM took 279 s and its first view 5.8 min; the bench's room-every4 (78) waited 210 s for its
    // full SfM on a busy fleet.
    let video = inspection.metrics.get("input_kind").and_then(Value::as_str) == Some("video");
    let capture = inspection.metrics.get("input_kind").and_then(Value::as_str) == Some("capture");
    let photos = inspection.metrics.get("image_count").and_then(Value::as_u64).filter(|_| !video);
    let photo_quick = photos.filter(|&n| !capture && n >= PHOTO_QUICK_MIN);
    // A few keyframes are solved quickly and their preview trains, so first view doesn't wait
    // minutes for the full solve. Poll it during the detached full extraction as well: otherwise
    // a ready quick solve waited behind a 30-44 s decode before its training could even start.
    let quick = (wants_preview && (video || photo_quick.is_some()))
        .then(|| start_quick_sfm(run, options, photo_quick));
    let mut quick = QuickPreview::new(run, quick)?;
    // The quick SfM's upload (first view) has the run's uplink until its inputs are staged: every
    // other upload of the run (frame prestaging, the full SfM's own staging) yields to it on one
    // stream instead of waiting (remote::yields_upload). A video's full frames go up to fal while
    // they decode (FramePrestage): the full SfM then finds them there instead of sending them after.
    let quick_job = quick.solve.as_ref().and_then(|job| job.job_id.clone());
    if let Some(id) = &quick_job {
        if let Err(error) = splat_core::remote::set_upload_priority(run.root(), id) {
            eprintln!("Upload priority for the quick SfM not recorded: {error:#}");
        }
    }
    let frames_before: std::collections::BTreeSet<String> = run.state()?.artifacts.into_keys().collect();
    let mut prestage: Option<Option<splat_core::remote::FramePrestage>> = (!video).then_some(None);
    let extraction_started = std::time::Instant::now();
    let mut extraction = dispatch(run, "extract_frames", json!({"target_count":options.target_count,"max_edge":options.max_edge}));
    while extraction.status == Status::Running {
        quick.advance(run)?;
        let state = run.state()?;
        ensure!(!matches!(state.status.as_str(), "stopping" | "cancelled"), "Run stopped");
        if prestage.is_none() {
            let frames = state.artifacts.values()
                .find(|a| a.kind == ArtifactKind::Frames && !frames_before.contains(&a.id));
            if let Some(frames) = frames {
                let started = splat_core::remote::FramePrestage::start(
                    run.root(), run.path(&frames.id)?.join("images"), &run.config().source);
                if started.is_some() {
                    run.journal("frames_prestage_started", json!({"frames_id":frames.id,
                        "after_extraction_s":extraction_started.elapsed().as_secs_f64()}))?;
                }
                prestage = Some(started);
            }
        }
        extraction = run.jobs().wait(extraction.job_id.as_deref().context("Missing extraction job_id")?, 0.25)?;
    }
    if let Some(Some(prestage)) = prestage {
        run.journal("frames_prestaged", prestage.finish())?;
    }
    ensure!(extraction.status == Status::Succeeded, "extract_frames: {:?}", extraction.error.map(|e| e.message));
    let frames = artifact(extraction, ArtifactKind::Frames)?;
    // Videos may contain moving people; keep their feature masks inline on the SfM runner.
    // Photos only need automatic segmentation when the inspection samples suggest sky.
    let sky = inspection
        .metrics
        .get("sky_sample_fraction")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        >= 0.1;
    let prompt = if inspection.metrics.get("input_kind").and_then(Value::as_str) == Some("video") {
        Some("sky; person")
    } else {
        sky.then_some("sky; person")
    };
    let mut sfm_request = json!({"frames_id":frames});
    if let Some(prompt) = prompt {
        if run.backend_capabilities()["sfm_inline_masks"] == true {
            sfm_request["mask_prompt"] = json!(prompt);
        } else if run.backend_capabilities()["mask_distractors"] == true {
            if let Some(mask) = completed(
                run,
                "mask_distractors",
                json!({"frames_id":frames,"prompt":prompt,"purpose":"sfm_features"}),
            )
            .ok()
            .filter(|r| r.status == Status::Succeeded)
            .and_then(|r| artifact(r, ArtifactKind::Masks).ok())
            {
                sfm_request["mask_id"] = json!(mask);
            }
        }
    }
    // No wait for the quick upload here (it held the full SfM 60 s on a 16 s clip whose quick job
    // was still publishing the binary): the full job stages beside it on one stream until the quick
    // inputs are on fal (remote::yields_upload). Its binary publish waits on the host lock for the
    // quick job's, and both send any shared small file whole, which /stage writes atomically. The
    // full job uploads video only once its exact replay recipe proves that it saves bytes.
    let mut full = dispatch(run, "reconstruct_sfm", sfm_request.clone());
    while full.status == Status::Running {
        quick.advance(run)?;
        ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
        full = run.jobs().wait(full.job_id.as_deref().context("Missing SfM job_id")?, 0.25)?;
    }
    ensure!(full.status == Status::Succeeded, "reconstruct_sfm: {:?}", full.error.map(|e| e.message));
    let mut sfm = artifact(full, ArtifactKind::Sfm)?;
    if video {
        sfm = placement_retry(run, options, &mut quick, &frames, sfm, &sfm_request)?;
    }
    quick.stop_solve(run)?;
    let QuickPreview { training: fast, sfm: quick_sfm, published: previewed, .. } = quick;
    if let Some(quick) = &quick_sfm {
        // One world and export frame for every package: the full model trains in the first view's.
        crate::sfm_align::share_world(run, quick, &sfm);
    }
    let split = artifact(
        invoke(run, "define_split", json!({"sfm_id":sfm}))?,
        ArtifactKind::Split,
    )?;
    let request = json!({"sfm_id":sfm,"split_id":split,"total_steps":options.steps,"cap_max":options.cap_max,"resolution_divisor":options.resolution_divisor});
    if wants_preview && !previewed {
        // The preview is expendable: a failure must not prevent the full reconstruction.
        // Keep MAX's initialization and schedule unchanged until warm-start quality is proven.
        let quick = match fast {
            Some((job, quick_sfm)) => race_preview(run, options, job, &quick_sfm, "fast_sfm", &request).map_err(|error| {
                let _ = run.journal("fast_preview_failed", json!({"error":format!("{error:#}")}));
                error
            }),
            None => Err(anyhow::anyhow!("no quick preview")),
        };
        if let Err(error) = quick.or_else(|_| preview(run, options, &sfm, &split, &request)) {
            ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
            run.journal("preview_failed", json!({"error":format!("{error:#}")}))?;
            eprintln!("Preview unavailable: {error:#}; continuing full training.");
        }
    }
    train_to_finish(run, options, request, "Deterministic reconstruction pipeline completed", agent_continues)
}

/// Below this share of its frames placed, a video solve is retried once ([`placement_retry`]). Bench
/// (326e03c): ultrawide-room 0.50 and arkit-47333904 0.60 fall below it; the lowest passing room places 0.88.
const RETRY_BELOW_PLACED: f64 = 0.80;
/// A retry replaces the first solve only if it places this much more of its frames.
const RETRY_MIN_GAIN: f64 = 0.10;
/// The retry's COLMAP mapper seed (the solve itself tries seeds 0 and 1).
const RETRY_MAPPER_SEED: u64 = 3;
/// The retry's frame rate: the 2 fps sampling before 326e03c. On fal (masked, GPU features) ultrawide-room placed
/// 98% at 2 fps and 51% / 43% at 5 fps (thinned / unthinned, seed 3).
const RETRY_FRAME_FPS: f64 = 2.0;

/// (placed share, blown-out camera count) of a finished SfM artifact, from its health.json.
fn placement(run: &Run, sfm: &str) -> Result<(f64, usize)> {
    let health: Value = serde_json::from_slice(&std::fs::read(run.path(sfm)?.join("health.json"))?)?;
    let input = health["n_input"].as_f64().unwrap_or(0.0);
    let placed = if input > 0.0 { health["n_registered"].as_f64().unwrap_or(0.0) / input } else { 0.0 };
    let blown = health["pose_continuity"]["blown_out_frames"].as_array().map_or(0, Vec::len);
    Ok((placed, blown))
}

/// One retry of a video solve that placed under RETRY_BELOW_PLACED of its frames or kept a blown-out
/// camera section: frames re-extracted at RETRY_FRAME_FPS, unthinned, and another mapper seed.
/// ultrawide-room: 5 fps left the 0-10 s opening out (51%; 2 fps 98%); arkit-47333904: a 132-frame hole (60%);
/// halloween: 35 cameras joined ~13,800 units off. Kept only if it places RETRY_MIN_GAIN more of its frames and has no
/// blown-out section; any retry failure keeps the first solve.
fn placement_retry(run: &Run, options: &FixedOptions, quick: &mut QuickPreview, frames: &str, sfm: String, request: &Value) -> Result<String> {
    let (placed, blown) = placement(run, &sfm)?;
    if placed >= RETRY_BELOW_PLACED && blown == 0 {
        return Ok(sfm);
    }
    eprintln!("Camera solve placed {:.0}% of the frames ({blown} cameras far outside the rest): retrying it once.", placed * 100.0);
    let attempt = (|| -> Result<(String, f64, usize)> {
        let selection: Value = serde_json::from_slice(&std::fs::read(run.path(frames)?.join("selection.json"))?)?;
        let thinned = selection["metrics"]["low_detail_ranges_s"].as_array().is_some_and(|r| !r.is_empty());
        let resample = thinned || selection["metrics"]["frame_base_fps"].as_f64() != Some(RETRY_FRAME_FPS);
        let frames = if resample {
            let mut extraction = dispatch(run, "extract_frames", json!({"target_count":options.target_count,
                "max_edge":options.max_edge,"thin_low_detail":false,"base_fps":RETRY_FRAME_FPS}));
            while extraction.status == Status::Running {
                quick.advance(run)?;
                ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
                extraction = run.jobs().wait(extraction.job_id.as_deref().context("Missing extraction job_id")?, 0.25)?;
            }
            ensure!(extraction.status == Status::Succeeded, "extract_frames: {:?}", extraction.error.map(|e| e.message));
            artifact(extraction, ArtifactKind::Frames)?
        } else {
            frames.to_string()
        };
        let mut request = request.clone();
        request["frames_id"] = json!(frames);
        request["mapper_seed"] = json!(RETRY_MAPPER_SEED);
        let mut retry = dispatch(run, "reconstruct_sfm", request);
        while retry.status == Status::Running {
            quick.advance(run)?;
            ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
            retry = run.jobs().wait(retry.job_id.as_deref().context("Missing SfM job_id")?, 0.25)?;
        }
        ensure!(retry.status == Status::Succeeded, "reconstruct_sfm: {:?}", retry.error.map(|e| e.message));
        let id = artifact(retry, ArtifactKind::Sfm)?;
        let (placed, blown) = placement(run, &id)?;
        Ok((id, placed, blown))
    })();
    let (kept, record) = match attempt {
        Ok((id, retry_placed, retry_blown)) => {
            let keep = retry_blown == 0 && retry_placed >= placed + RETRY_MIN_GAIN;
            let record = json!({"first":sfm,"placed":placed,"blown_out":blown,"retry":id,
                "retry_placed":retry_placed,"retry_blown_out":retry_blown,"kept":if keep { "retry" } else { "first" }});
            (if keep { id } else { sfm }, record)
        }
        Err(error) => {
            ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
            let record = json!({"first":sfm,"placed":placed,"blown_out":blown,"error":format!("{error:#}"),"kept":"first"});
            (sfm, record)
        }
    };
    eprintln!("Camera solve retry: {record}");
    run.journal("sfm_placement_retry", record)?;
    Ok(kept)
}

/// Race a small first view against full MAX training. Dispatch is idempotent, so
/// `train_to_finish` below joins the already running full job (including its normal retry path).
fn preview(run: &Run, options: &FixedOptions, sfm: &str, split: &str, full_request: &Value) -> Result<()> {
    let first = dispatch(run, "train_splats", json!({"sfm_id":sfm,"split_id":split,"preview":true,
        // The video preview's class: FAST_STEPS at a <=960 px edge (here up to 1M splats). 7k against
        // X56's 14k steps at 1600 px, scored at the half-res edge: room 25.96 vs 25.98 dB, garden 28.58
        // vs 27.18 dB, training 45 -> 20 s and 85 -> 23 s (product tools on fal, 1 seed).
        "total_steps":FAST_STEPS,"cap_max":FAST_CAP,"cap_max_explicit":true,
        "resolution_divisor":options.max_edge.div_ceil(960).clamp(1, 8).max(options.resolution_divisor)}));
    // Train the photo preview alone, then start MAX: run side by side, the heavy MAX job slowed the
    // ~50 s preview 2.5-8x on the bench (room 3a73ae2: 241 -> 37 steps/s once MAX ramped), so the
    // first view landed with the final. MAX starts ~50 s later; the final has the most slack.
    let mut first = first;
    while first.status == Status::Running {
        ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
        first = run.jobs().wait(first.job_id.as_deref().context("Missing preview job_id")?, 0.25)?;
    }
    race_preview(run, options, first, sfm, "full_sfm", full_request)
}

/// Race the preview job `first` (trained on `sfm`) against full MAX training, dispatched here.
/// Dispatch is idempotent, so `train_to_finish` joins the already running full job.
fn race_preview(run: &Run, options: &FixedOptions, mut first: ToolResult, sfm: &str, poses: &str, full_request: &Value) -> Result<()> {
    let mut full_request = full_request.clone();
    options.training_request(&mut full_request);
    let mut full = dispatch(run, "train_splats", full_request);
    loop {
        ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
        if full.status == Status::Running {
            full = run.jobs().wait(full.job_id.as_deref().context("Missing full job_id")?, 0.)?;
        }
        // Never replace a ready full result with a slower preview. Publish before cancelling
        // the redundant job so cancellation latency cannot delay first view.
        if full.status == Status::Succeeded {
            crate::export_candidate(run, &artifact(full, ArtifactKind::Candidate)?)?;
            if first.status == Status::Running {
                run.jobs().stop(first.job_id.as_deref().context("Missing preview job_id")?, 60.)?;
            }
            return Ok(());
        }
        if first.status != Status::Running {
            break;
        }
        first = run.jobs().wait(first.job_id.as_deref().context("Missing preview job_id")?, 0.25)?;
    }
    ensure!(first.status == Status::Succeeded, "preview training: {:?}", first.error);
    publish_preview(run, &artifact(first, ArtifactKind::Candidate)?, sfm, poses)
}

/// Keyframes, edge and preview steps for the quick first-view SfM (a few hundred pairs even when
/// exhaustive).
/// Bench f6cb4a3: at 36 x 1000 px / 14k the quick path's upload (shared with the full SfM's) and
/// training kept driveway and gimbal over 2 min; this lighter preview is replaced at "ready".
/// 24 frames, not fewer: the full frame plan keeps every quick frame as an anchor (and fills sparse
/// stretches with them, frameplan::with_anchors/with_fill), so fewer quick frames would change the
/// full SfM's and the final model's inputs; and at FAST_GAP_WIDTHS 16 frames cover only 2.4 widths
/// of the room, while wider gaps lose the initial pair (showbus, below).
const FAST_FRAMES: u32 = 24;
/// Below the 960 px extraction floor (ExtractFrames::preview): the same 24 source frames (picked on
/// the analysis track, so the full plan's anchors are unchanged) at 720 px. Warm room run 4f73f27c
/// at 960 px: upload 6.3 MB in 10.9 s, COLMAP 9.6 s, 7k preview 19.1 s; 720 px is 0.56x the pixels.
const FAST_EDGE: u32 = 720;
/// Video: the quick frames cover the clip prefix with at most this much camera motion (image widths)
/// per frame. Showbus (quick SfM only): gaps of 0.60, 0.30 and 0.23 widths found no initial pair, 0.12
/// registered 23/24 and 0.10 24/24 (goal2/quickstretch).
const FAST_GAP_WIDTHS: f64 = 0.15;
/// Preview steps (quick and full-SfM previews). 7k -> 3.5k: warm room run 4f73f27c trained its 7k
/// preview (21 views at 960 px, at the 1M cap by the end) in 19.1 s of a 29.5 s job; growth is
/// clamped to the schedule, so 3.5k ends with fewer splats and trains in about 8-10 s. The preview
/// is a draft that the full model replaces: MAX training is unchanged.
const FAST_STEPS: u32 = 3_500;
/// Splat cap of the quick preview. 1M -> 500k: pexels-livingroom-stove's 3.5k preview reached 1M by
/// step 3000; its last 500 steps took 4.4 s of 13.6 s and its 23 MB SPZ took most of a 12.3 s result
/// fetch before the first view. The preview is a draft that the full model replaces.
const FAST_CAP: u32 = 500_000;
/// Photo sets from PHOTO_SUBSET_MIN: the quick SfM takes the first photo of each of this many runs in capture
/// (name) order (uniform: the others are not decoded; garden's sharpness pick decoded all 185 in
/// 15 s), at this edge: photos overlap less than video keyframes. Their frames keep the photos' source
/// indices, so they are a subset of the full SfM's (named alike) and share_world aligns the two.
const PHOTO_FAST_FRAMES: u32 = 32;
const PHOTO_FAST_EDGE: u32 = 1000;
/// Photo sets this large get the quick SfM (garden 185: first view 5.8 min -> 46-119 s; room-every4 78:
/// first view 210 s on the bench waiting for the full SfM). Under it (ETH3D courtyard, 38 photos: first
/// view 77-110 s) the full SfM is the first view, as before.
const PHOTO_QUICK_MIN: u64 = 40;
/// From this many photos the quick SfM takes a PHOTO_FAST_FRAMES subset (garden 185: 32/32 every run);
/// below, all of them at PHOTO_FAST_EDGE and low quality.
const PHOTO_SUBSET_MIN: u64 = 100;

/// The quick SfM on a few keyframes (its job, usually still running).
/// A video's keyframes sit at even steps of camera motion (each must overlap the next); photo sets
/// take every n-th photo (`photos`: the set's size, see PHOTO_FAST_FRAMES). It passes on
/// the preview floor (40% of its frames in one model): the full SfM replaces it.
fn start_quick_sfm(run: &Run, options: &FixedOptions, photos: Option<u64>) -> Result<ToolResult> {
    let request = match photos {
        None => json!({"target_count":FAST_FRAMES.min(options.target_count),"max_edge":FAST_EDGE.min(options.max_edge),"spacing":"motion",
            "max_motion_widths":FAST_FRAMES as f64 * FAST_GAP_WIDTHS,"preview":true}),
        Some(n) => {
            // Under PHOTO_SUBSET_MIN photos the quick SfM takes them all: every subset of room-every4
            // (78) split in two (uniform 32: 19+15, 39: 23+16, 52: 30+22; the first 32: 27/32), while all
            // 78 at 1000 px / low quality registered 78/78 in one model, SfM 15 s (research fal, SfM only).
            let count = if n < PHOTO_SUBSET_MIN { n as u32 } else { PHOTO_FAST_FRAMES }.min(options.target_count);
            json!({"target_count":count,"max_edge":PHOTO_FAST_EDGE.min(options.max_edge),"strategy":"uniform","sfm_copies":false})
        }
    };
    let frames = artifact(
        invoke(run, "extract_frames", request)?,
        ArtifactKind::Frames,
    )?;
    // Cloud: the same fal job trains the preview right after the solve (FAST_STEPS, FAST_CAP, full
    // frame size), saving the second job's upload, queue and the solve's result round trip; a
    // missing or failed chained preview falls back to its own train job (QuickPreview::advance).
    Ok(dispatch(run, "reconstruct_sfm", json!({"frames_id":frames,"quality":"low","preview":true,
        "preview_train":{"total_steps":FAST_STEPS,"cap_max":FAST_CAP,"resolution_divisor":1}})))
}

/// Until `job`'s inputs are on fal (the client marks its job dir), it ends, or QUICK_UPLOAD_WAIT
/// passes (clients without the marker: the dev drive.py path).
fn wait_inputs_staged(run: &Run, job: &ToolResult) -> Result<()> {
    let Some(id) = job.job_id.as_deref() else { return Ok(()) };
    let marker = run.root().join("jobs").join(id).join(splat_core::fal::INPUTS_STAGED);
    let started = std::time::Instant::now();
    while !marker.exists() && started.elapsed() < QUICK_UPLOAD_WAIT {
        ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
        if run.jobs().wait(id, 0.25)?.status != Status::Running {
            break;
        }
    }
    Ok(())
}

/// Longest a job waits for another's upload in [`wait_inputs_staged`] (clients without the marker).
const QUICK_UPLOAD_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// The quick solve and its first-view training, advanced while full extraction or SfM runs.
/// Every poll is nonblocking; optional preview failures leave the full pipeline available.
#[derive(Default)]
struct QuickPreview {
    solve: Option<ToolResult>,
    training: Option<(ToolResult, String)>,
    sfm: Option<String>,
    published: bool,
}

impl QuickPreview {
    fn failed(run: &Run, error: anyhow::Error) -> Result<()> {
        ensure!(!matches!(run.state()?.status.as_str(), "stopping" | "cancelled"), "Run stopped");
        run.journal("fast_preview_failed", json!({"error":format!("{error:#}")}))?;
        eprintln!("Quick preview unavailable: {error:#}; previewing after the full SfM.");
        Ok(())
    }

    fn new(run: &Run, solve: Option<Result<ToolResult>>) -> Result<Self> {
        let solve = match solve {
            Some(Ok(job)) => Some(job),
            Some(Err(error)) => { Self::failed(run, error)?; None }
            None => None,
        };
        Ok(Self { solve, ..Self::default() })
    }

    fn advance(&mut self, run: &Run) -> Result<()> {
        if let Some(mut job) = self.solve.take() {
            let started = (|| -> Result<()> {
                if job.status == Status::Running {
                    job = run.jobs().wait(job.job_id.as_deref().context("Missing quick SfM job_id")?, 0.)?;
                }
                if job.status == Status::Running {
                    self.solve = Some(job);
                    return Ok(());
                }
                ensure!(job.status == Status::Succeeded, "quick SfM: {:?}", job.error.map(|e| e.message));
                let chained = job.metrics.get("preview_candidate_id").and_then(Value::as_str).map(str::to_owned);
                let chain_error = job.metrics.get("preview_chain_error").cloned();
                let training_s = job.metrics.get("preview_training_s").cloned();
                let sfm = artifact(job, ArtifactKind::Sfm)?;
                // The quick SfM job trained the preview itself (one fal exec): publish it now.
                if let Some(candidate) = chained {
                    run.journal("fast_preview_chained", json!({"sfm_id":sfm,"candidate_id":candidate,"training_s":training_s}))?;
                    match publish_preview(run, &candidate, &sfm, "fast_sfm") {
                        Ok(()) => {
                            self.sfm = Some(sfm);
                            self.published = true;
                            return Ok(());
                        }
                        Err(error) => run.journal("fast_preview_chain_failed", json!({"candidate_id":candidate,"error":format!("{error:#}")}))?,
                    }
                } else if let Some(error) = chain_error {
                    run.journal("fast_preview_chain_failed", json!({"error":error}))?;
                }
                // Fallback: the preview as its own train job.
                let split = artifact(invoke(run, "define_split", json!({"sfm_id":sfm}))?, ArtifactKind::Split)?;
                let first = dispatch(run, "train_splats", json!({"sfm_id":sfm,"split_id":split,"preview":true,
                    "total_steps":FAST_STEPS,"cap_max":FAST_CAP,"cap_max_explicit":true,"resolution_divisor":1}));
                self.sfm = Some(sfm.clone());
                self.training = Some((first, sfm));
                Ok(())
            })();
            if let Err(error) = started { Self::failed(run, error)?; }
        }
        if let Some((mut job, sfm)) = self.training.take() {
            let published = (|| -> Result<()> {
                if job.status == Status::Running {
                    job = run.jobs().wait(job.job_id.as_deref().context("Missing preview job_id")?, 0.)?;
                }
                if job.status == Status::Running {
                    self.training = Some((job, sfm));
                    return Ok(());
                }
                ensure!(job.status == Status::Succeeded, "quick preview training: {:?}", job.error.map(|e| e.message));
                publish_preview(run, &artifact(job, ArtifactKind::Candidate)?, &sfm, "fast_sfm")?;
                self.published = true;
                Ok(())
            })();
            if let Err(error) = published { Self::failed(run, error)?; }
        }
        Ok(())
    }

    fn stop_solve(&mut self, run: &Run) -> Result<()> {
        if let Some(job) = self.solve.take() {
            if job.status == Status::Running {
                // Wait for it to settle: a stopped job still recorded as running fails the run's
                // `finish` (night-soccer, 2026-10-11: the quick job queued 6 min; the full SfM won).
                run.jobs().stop(job.job_id.as_deref().context("Missing quick SfM job_id")?, 30.)?;
            }
            run.journal("fast_preview_skipped", json!({"reason":"full SfM finished first"}))?;
        }
        Ok(())
    }
}

/// exports/web from a preview candidate trained on `sfm` (named explicitly: a run with a quick
/// and a full SfM must not export against whichever finished last).
fn publish_preview(run: &Run, candidate: &str, sfm: &str, poses: &str) -> Result<()> {
    let out = run.root().join("exports/web");
    // The train job wrote the package next to its splat (crate::brush_worker); older binaries didn't.
    if !crate::copy_candidate_web(run, candidate)? {
        splat_core::fal::materialize(run.root(), &run.path(candidate)?.join("splat.ply"))?;
        splat_webexport::export_web_with_options(
            &run.path(candidate)?.join("splat.ply"), Some(&run.path(sfm)?), &out,
            &splat_webexport::SceneOptions { lod: splat_webexport::LodMode::Skip, ..Default::default() },
        )?;
    }
    // The quick path's preview (few keyframes, low resolution, 7k steps) is a draft until "ready".
    let quality = if poses == "fast_sfm" { "quick" } else { "standard" };
    run.journal("preview_ready", json!({"candidate_id":candidate,"path":"exports/web","poses":poses,"preview_quality":quality}))?;
    eprintln!("Preview ready: {} (full quality training continues)", out.display());
    Ok(())
}

/// `vauss refine`: train the run's finished candidate further on its own SfM and frozen split (no
/// camera re-solve), resuming its final checkpoint (or warm-starting from its splat), then evaluate,
/// mesh (when `options.no_mesh` is off) and finish with the refined candidate.
pub fn refine_pipeline(run: &Run, options: &FixedOptions, prior: &str, cap_max: Option<u32>) -> Result<Value> {
    let artifact = run.artifact(prior)?;
    ensure!(artifact.kind == ArtifactKind::Candidate, "{prior} is not a candidate");
    let meta = |k: &str| artifact.meta.get(k).cloned().unwrap_or(Value::Null);
    let (sfm, split, prior_request) = (meta("sfm_id"), meta("split_id"), meta("request"));
    ensure!(sfm.is_string() && split.is_string(), "{prior} records no SfM/split to refine on");
    let mut request = json!({"sfm_id":sfm,"split_id":split,"total_steps":options.steps,
        "cap_max":cap_max.map(|c| json!(c)).or_else(|| prior_request.get("cap_max").cloned()).unwrap_or(json!(options.cap_max)),
        "cap_max_explicit":cap_max.is_some(),
        "resolution_divisor":if options.preset.is_some() { json!(options.resolution_divisor) } else {
            prior_request.get("resolution_divisor").cloned().unwrap_or(json!(options.resolution_divisor)) }});
    // A finished candidate keeps its final-step checkpoint: a true resume continues the LR schedule.
    let checkpoint = run.path(prior)?.join("checkpoint.json");
    // Presets give refinement a fresh short/full schedule, even when the prior already reached
    // that many steps. Explicit --steps retains its documented absolute checkpoint target.
    let resume = checkpoint.is_file() && (options.preset.is_none() || options.steps_explicit);
    request[if resume { "resume_candidate_id" } else { "init_candidate_id" }] = json!(prior);
    train_to_finish(run, options, request, &format!("Refined {prior} on the run's SfM and frozen split"), false)
}

/// Train (retrying once from the checkpoint of an interrupted job), evaluate on the frozen split,
/// mesh unless `no_mesh`, and finish with the trained candidate.
fn train_to_finish(run: &Run, options: &FixedOptions, mut request: Value, what: &str, agent_continues: bool) -> Result<Value> {
    options.training_request(&mut request);
    let split = request["split_id"].clone();
    let base = request.clone();
    let full = dispatch(run, "train_splats", request.clone());
    let mut fork = fork_refine(run, &full, &base)?;
    let mut train = completed(run, "train_splats", request.clone())?;
    let interrupted = train
        .job_id
        .as_ref()
        .is_some_and(|id| run.root().join("jobs").join(id).join("stop.json").is_file());
    let retryable = train
        .error
        .as_ref()
        .is_some_and(|e| e.retryable || e.code == "interrupted" || e.code == "job_interrupted");
    if train.status != Status::Succeeded && (interrupted || retryable) {
        // Backends validate full optimizer checkpoints; a PLY is only a warm start.
        if let Some(candidate) = train
            .artifacts
            .iter()
            .find(|a| a.kind == ArtifactKind::Candidate)
        {
            // The interrupted training will not write its fork: refine after the resumed one instead.
            if let Some(r) = fork.take() {
                stop_fork(run, &r, "full training interrupted")?;
            }
            request.as_object_mut().unwrap().remove("init_candidate_id");
            request["resume_candidate_id"] = json!(candidate.id);
            train = completed(run, "train_splats", request)?;
        }
    }
    if train.status != Status::Succeeded {
        if let Some(r) = fork.take() {
            stop_fork(run, &r, "full training failed")?;
        }
    }
    ensure!(
        train.status == Status::Succeeded,
        "train_splats: {}",
        train
            .error
            .as_ref()
            .map(|e| e.message.clone())
            .unwrap_or_else(|| format!("{:?}", train.status))
    );
    let candidate = artifact(train, ArtifactKind::Candidate)?;
    // A model within the interactive budget is its own package (the fork found nothing to refine).
    if splats(run, &candidate).is_none_or(|n| n <= INTERACTIVE_SPLATS as u64) {
        if let Some(r) = fork.take() {
            stop_fork(run, &r, "full model within the interactive budget")?;
        }
    }
    crate::export_candidate(run, &candidate)?;
    let candidate_path = run.path(&candidate)?.join("splat.ply");
    if options.is_fast() {
        if agent_continues {
            return Ok(json!({"candidate_id":candidate}));
        }
        // A preview is deliberately unevaluated: never label it a quality-validated success.
        let explanation = format!("{what}: fast preview exported; full training, held-out evaluation and mesh skipped.");
        invoke(run, "finish", json!({"status":"partial", "explanation":explanation,
            "candidate_id":candidate, "remaining_defects":["Fast preview: quality has not been evaluated; no mesh was generated."]}))?;
        run.journal("run_finished", json!({"status":"partial", "stop_reason":"fast_preview_complete"}))?;
        eprintln!("Preview ready: {} (fast preview only; quality not evaluated)", candidate_path.display());
        return write_report(run);
    }
    run.journal("candidate_ready", json!({"candidate_id":candidate,"path":candidate_path,"evaluation":"pending"}))?;
    eprintln!("Ready: {} (held-out evaluation follows)", run.root().join("exports/web").display());
    write_report(run)?;
    // Cloud models above the interactive budget: a capped refine runs on fal alongside evaluation
    // and mesh, and its package replaces the full model's as exports/web (the full PLY stays on fal
    // for renders and downloads). Pruning to the budget without retraining lost the fine detail.
    // The refine forked beside MAX is already running (or done); otherwise it starts now.
    let interactive = match fork {
        Some(r) => Some(Interactive { fallback: interactive_request(run, &candidate, &base)?, request: r }),
        None => interactive_request(run, &candidate, &base)?.map(|r| Interactive { request: r, fallback: None }),
    };
    let mut interactive_job = None;
    if let Some(Interactive { request: r, fallback }) = &interactive {
        let started = dispatch(run, "train_splats", r.clone());
        if fallback.is_none() {
            run.journal("interactive_started", json!({"from_candidate":candidate,"job_id":started.job_id,"status":started.status}))?;
        }
        interactive_job = Some(started);
    }
    // The mesh runs alongside evaluation when no edit will replace the candidate (dispatch is idempotent:
    // the wait below joins this job). After evaluation it gets mesh_wait_s() more, then the run finishes
    // without it: showbus's mesh job waited 472 s for its inputs on a cold runner and held run_finished
    // 12 min past the final splat.
    let eval_request = json!({"candidate_id":candidate,"split_id":split,"lpips":!options.no_lpips});
    if !agent_continues && !options.no_mesh && removal_target(run.goal()).is_none() {
        // One run's jobs must not upload the same files at once (/stage race): the mesh starts once the
        // refine's and evaluation's inputs are on fal.
        let eval_job = dispatch(run, "evaluate", eval_request.clone());
        for job in interactive_job.iter().chain([&eval_job]) {
            wait_inputs_staged(run, job)?;
        }
        let started = dispatch(run, "extract_mesh", json!({"candidate_id":candidate}));
        run.journal("mesh_started", json!({"candidate_id":candidate,"job_id":started.job_id,"status":started.status}))?;
    }
    let evaluation = invoke(run, "evaluate", eval_request)?;
    if agent_continues {
        if let Some(r) = interactive {
            r.publish(run)?;
        }
        write_report(run)?;
        return Ok(json!({"candidate_id":candidate, "evaluation":evaluation}));
    }
    let mut defects = Vec::new();
    let candidate = match removal_target(run.goal()) {
        Some(target) => remove_target(run, &candidate, &target, &mut defects)?,
        None => {
            if removal_unparsed(run.goal()) {
                defects.push(format!("{EDIT_FAIL} anything: the goal asks for a removal but names no object the edit tools can select (say \"remove the <object>\")."));
            }
            candidate
        }
    };
    // A camera solve that placed too few frames still trains on the rest, but the run says so.
    defects.extend(splat_core::sfm_coverage_defect(run, &candidate));
    let mut mesh_missing = false;
    if !options.no_mesh {
        let budget = mesh_wait_s();
        match completed_within(run, "extract_mesh", json!({"candidate_id":candidate}), budget)? {
            outcome @ (Bounded::Stopped | Bounded::Abandoned) => {
                // Over budget: the splat result is complete, so this is a note, not a reason for `partial`.
                mesh_missing = true;
                let abandoned = matches!(outcome, Bounded::Abandoned);
                run.journal("mesh_skipped", json!({"candidate_id":candidate,"budget_s":budget,"abandoned":abandoned,
                    "kind":"timeout","reason":format!("the mesh did not finish within {budget:.0} s after the evaluation, so the run finished without it; the splat is complete")}))?;
                defects.push(format!(
                    "{MESH_NOTE}{} it did not finish within {budget:.0} s of the evaluation, so the run finished without it (the splat result is complete). For a mesh: vauss tool extract_mesh --run-dir {} --input '{{\"candidate_id\":\"{candidate}\"}}'",
                    if abandoned { " (abandoned)" } else { "" },
                    run.root().display()
                ));
            }
            Bounded::Done(mesh) if mesh.status == Status::Unsupported => {
                mesh_missing = true;
                let why = mesh.error.map(|e| e.message).unwrap_or_else(|| "Mesh extraction is unavailable for this backend".into());
                run.journal("mesh_skipped", json!({"candidate_id":candidate,"kind":"unsupported","reason":why}))?;
                defects.push(why);
            }
            Bounded::Done(mesh) if mesh.status != Status::Succeeded => {
                // The splat result is already exported: a mesh failure is a defect of
                // this run, not a failed run.
                mesh_missing = true;
                let why = mesh.error.map(|e| e.message).unwrap_or_else(|| format!("{:?}", mesh.status));
                run.journal("mesh_failed", json!({"error": why, "kind": "failed", "reason": format!("mesh extraction failed: {why}")}))?;
                defects.push(format!("{MESH_FAILED} {why}"));
            }
            Bounded::Done(_) => {}
        }
    }
    // A requested removal that removed nothing fails loudly: the scene still shows the object.
    let edit_failed = defects.iter().find(|d| d.starts_with(EDIT_FAIL)).cloned();
    let status = if edit_failed.is_some() {
        "failed"
    } else if defects.iter().all(|d| d.starts_with(EDIT_NOTE) || d.starts_with(MESH_NOTE) || d.starts_with(MESH_FAILED)) {
        "succeeded"
    } else {
        "partial"
    };
    let mut explanation = if mesh_missing {
        format!("{what} with held-out evaluation; no mesh (see remaining defects).")
    } else {
        format!("{what} with held-out evaluation.")
    };
    if let Some(why) = &edit_failed {
        explanation = format!("The requested edit was not made. {why} The reconstruction itself finished ({}).", explanation.trim_end_matches('.'));
    }
    if let Some(note) = defects.iter().find(|d| d.starts_with(EDIT_NOTE)) {
        explanation.push_str(&format!(" {note} The held-out numbers are the unedited candidate's."));
    }
    if let Some(r) = interactive {
        r.publish(run)?;
    }
    // Record jobs that ended unobserved (a stopped quick solve) so `finish` sees them as done.
    run.jobs().reconcile()?;
    invoke(
        run,
        "finish",
        json!({"status":status,"explanation":explanation,"candidate_id":candidate,"remaining_defects":defects}),
    )?;
    run.journal(
        "run_finished",
        json!({"status":status,"stop_reason":"fixed_pipeline_complete"}),
    )?;
    write_report(run)
}

/// Splats of `vauss run --cloud`'s interactive package, and the refine steps that recover detail
/// after warm-starting from the full model reduced to it (splat-brush warm_cap.rs). 20k vs 10k:
/// 0.1-0.3 dB closer to the full model on 4 corpus scenes for ~+40-80 s of refine on an L40S.
/// 2M instead of 1M is closer still but draws at 33 fps at 1080p on an M4 (1M: 49).
const INTERACTIVE_SPLATS: u32 = splat_webexport::cap::INTERACTIVE_SPLATS as u32;
const INTERACTIVE_STEPS: u32 = 20_000;

/// The interactive refine to publish: `request`, or when it was forked beside MAX and failed (no
/// fork written, a lost runner), `fallback`, today's refine from the finished model.
struct Interactive {
    request: Value,
    fallback: Option<Value>,
}

/// Fork the interactive refine beside the full cloud MAX training `full` (dispatched with `request`)
/// when that training writes a fork model (`fork_ply` in its train_job.json: step 30k of 45k, or
/// its early stop): the refine starts from it instead of after MAX, its export and its result sync
/// (in-exec: 53 s of MAX left on arkit-41254538, 116 s on pexels-livingroom-stove, refine quality
/// equal to refining the final model). Its runner waits for the fork on fal /data. Returns the
/// refine's request.
fn fork_refine(run: &Run, full: &ToolResult, request: &Value) -> Result<Option<Value>> {
    let Some(candidate) = full.artifacts.iter().find(|a| a.kind == ArtifactKind::Candidate).map(|a| a.id.clone()) else {
        return Ok(None);
    };
    let job: Value = std::fs::read(run.path(&candidate)?.join("train_job.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
    if !job["fork_ply"].is_string() || full.status != Status::Running {
        return Ok(None);
    }
    // One run's jobs must not upload the same files at once (/stage race): the refine needs the
    // same SfM, so it starts once MAX's inputs are on fal (and then uploads nothing).
    wait_inputs_staged(run, full)?;
    let mut r = json!({"sfm_id":request["sfm_id"],"split_id":request["split_id"],"fork_of_candidate_id":candidate,
        "total_steps":INTERACTIVE_STEPS,"total_steps_explicit":true,"cap_max":INTERACTIVE_SPLATS,"cap_max_explicit":true,
        "resolution_divisor":request["resolution_divisor"]});
    carry_training_controls(request, &mut r);
    let started = dispatch(run, "train_splats", r.clone());
    run.journal("interactive_started", json!({"from_candidate":candidate,"job_id":started.job_id,"status":started.status,"forked":true}))?;
    Ok(Some(r))
}

/// Stop the forked refine `request` (its package is not needed or its fork will never come).
fn stop_fork(run: &Run, request: &Value, why: &str) -> Result<()> {
    let job = dispatch(run, "train_splats", request.clone());
    if let (Status::Running, Some(id)) = (job.status, job.job_id.as_deref()) {
        run.jobs().stop(id, 60.)?;
    }
    run.journal("interactive_fork_stopped", json!({"reason":why}))?;
    Ok(())
}

/// Splats of a trained candidate (its train_summary.json).
fn splats(run: &Run, candidate: &str) -> Option<u64> {
    let summary: Value = serde_json::from_slice(&std::fs::read(run.path(candidate).ok()?.join("train_summary.json")).ok()?).ok()?;
    summary["splats"].as_u64()
}

/// The capped refine of `candidate` for the interactive package: cloud runs whose model is above
/// [`INTERACTIVE_SPLATS`] only.
fn interactive_request(run: &Run, candidate: &str, request: &Value) -> Result<Option<Value>> {
    if splat_core::remote::mode().is_none() {
        return Ok(None);
    }
    if splats(run, candidate).is_none_or(|n| n <= INTERACTIVE_SPLATS as u64) {
        return Ok(None);
    }
    let mut r = json!({"sfm_id":request["sfm_id"],"split_id":request["split_id"],"init_candidate_id":candidate,
        "total_steps":INTERACTIVE_STEPS,"total_steps_explicit":true,"cap_max":INTERACTIVE_SPLATS,"cap_max_explicit":true,
        "resolution_divisor":request["resolution_divisor"]});
    carry_training_controls(request, &mut r);
    Ok(Some(r))
}

/// The run's `--seed` / `--eval-every`, when given, also apply to a training derived from `from`
/// (the interactive refine keeps its own step and splat budget). Previews keep their defaults.
fn carry_training_controls(from: &Value, to: &mut Value) {
    for key in ["seed", "eval_every"] {
        if let Some(v) = from.get(key).filter(|v| !v.is_null()) {
            to[key] = v.clone();
        }
    }
}

/// Wait for the capped refine and publish its package (written by its train job) as exports/web.
/// A failed refine leaves the full model's package in place.
impl Interactive {
    /// Publish the refine's package, or the fallback's when a forked refine failed.
    fn publish(self, run: &Run) -> Result<()> {
        if publish_interactive(run, self.request, self.fallback.is_some())? {
            return Ok(());
        }
        let Some(r) = self.fallback else { return Ok(()) };
        let started = dispatch(run, "train_splats", r.clone());
        run.journal("interactive_started", json!({"job_id":started.job_id,"status":started.status,"after_failed_fork":true}))?;
        publish_interactive(run, r, false).map(|_| ())
    }
}

/// Wait for the capped refine and publish its package (written by its train job) as exports/web;
/// whether it was published. A failed refine leaves the full model's package in place (`quiet`: a
/// fallback follows, so it is not reported as the run's).
fn publish_interactive(run: &Run, request: Value, quiet: bool) -> Result<bool> {
    let refined = completed(run, "train_splats", request)?;
    let candidate = match (refined.status, artifact(refined.clone(), ArtifactKind::Candidate)) {
        (Status::Succeeded, Ok(c)) => c,
        _ => {
            let why = refined.error.map(|e| e.message).unwrap_or_else(|| format!("{:?}", refined.status));
            run.journal(if quiet { "interactive_fork_failed" } else { "interactive_failed" }, json!({"error": why}))?;
            if !quiet {
                eprintln!("Interactive package unavailable ({why}); exports/web keeps the full model");
            }
            return Ok(false);
        }
    };
    if crate::copy_candidate_web(run, &candidate)? {
        run.journal("interactive_ready", json!({"candidate_id":candidate,"path":"exports/web","splats":INTERACTIVE_SPLATS}))?;
        eprintln!("Interactive package ({INTERACTIVE_SPLATS} splats): {}", run.root().join("exports/web").display());
        crate::spawn_collision(run, &candidate);
        Ok(true)
    } else {
        run.journal("interactive_failed", json!({"candidate_id":candidate,"error":"refined candidate has no web package"}))?;
        Ok(false)
    }
}

/// Prefix of the defect a requested removal that removed nothing leaves (run status `failed`).
const EDIT_FAIL: &str = "Could not remove";

/// Prefix of the informational remaining-defect an edit leaves (not a reason for `partial`).
const EDIT_NOTE: &str = "Edited:";

/// Prefix of the note a mesh over its time budget leaves (not a reason for `partial`: the splat is done).
const MESH_NOTE: &str = "Mesh skipped:";

/// Prefix of the defect a failed mesh job leaves (not a reason for `partial` either: the mesh is a
/// side output of an already exported, evaluated splat).
const MESH_FAILED: &str = "Mesh extraction failed:";

/// Seconds the fixed pipeline waits for the mesh after evaluation (it started alongside evaluation);
/// SPLAT_MESH_WAIT_S overrides. Dev corpus: mesh jobs took 20-160 s end to end; showbus's sat 472 s
/// on a cold runner's inputs.
fn mesh_wait_s() -> f64 {
    std::env::var("SPLAT_MESH_WAIT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(180.0)
}

/// The static object a goal asks to remove ("... and remove the black leather armchair from the
/// scene" -> "black leather armchair", singular), if any.
pub fn removal_target(goal: &str) -> Option<(String, bool)> {
    let goal = goal.to_lowercase();
    let (_, rest) = ["remove ", "delete ", "erase ", "get rid of ", "take out "]
        .iter()
        .filter_map(|verb| goal.find(verb).map(|i| (i, &goal[i + verb.len()..])))
        .min_by_key(|(i, _)| *i)?;
    // Grounding DINO matches nouns throughout the prompt: a location such as "on the
    // coffee table" otherwise selects the table too. Match the object before its location.
    let end = [" from ", " in ", " on ", " and ", ".", ",", ";", "!", " so ", " but "].iter().filter_map(|t| rest.find(t)).min().unwrap_or(rest.len());
    let phrase = rest[..end].trim();
    let (singular, noun) = match phrase.split_once(' ') {
        Some(("the" | "a" | "an" | "this" | "that", noun)) => (true, noun),
        Some(("all" | "every" | "any", noun)) => (false, noun.strip_prefix("the ").unwrap_or(noun)),
        _ => (false, phrase),
    };
    // Where the object is ("by the window", "next to the sofa") is not part of the detector noun.
    let noun = [" by ", " near ", " next to ", " beside ", " behind ", " in front of ", " on the ", " at the ", " against ", " under ", " to the left", " to the right", " that ", " which ", " with "]
        .iter()
        .filter_map(|t| noun.find(t))
        .min()
        .map_or(noun, |i| &noun[..i]);
    // Detector prompts are short nouns: a long phrase keeps its last four words (the head noun).
    let words: Vec<&str> = noun.split_whitespace().collect();
    let noun = &words[words.len().saturating_sub(4)..].join(" ");
    let noun = noun.trim();
    // Moving distractors are mask_distractors' job, not an edit of the splat.
    if noun.is_empty() || ["people", "person", "passers-by", "pedestrians", "traffic", "cars"].contains(&noun) {
        return None;
    }
    Some((noun.to_string(), singular && !noun.ends_with('s')))
}

/// The goal uses a removal verb but [`removal_target`] found no object to select (moving distractors
/// such as people or traffic excepted: masking handles those).
pub fn removal_unparsed(goal: &str) -> bool {
    let g = goal.to_lowercase();
    let verb = ["remove ", "delete ", "erase ", "get rid of ", "take out "].iter().any(|v| g.contains(v));
    let distractor = ["people", "person", "passers-by", "pedestrians", "traffic", "cars"].iter().any(|d| g.contains(d));
    verb && !distractor && removal_target(goal).is_none()
}

/// select_objects + remove_objects for the goal's target on `candidate`; the edited candidate, or
/// the original with a defect when nothing usable was selected.
fn remove_target(run: &Run, candidate: &str, (noun, singular): &(String, bool), defects: &mut Vec<String>) -> Result<String> {
    let select = completed(run, "select_objects", json!({"candidate_id":candidate,"prompt":noun,"instances":if *singular {"largest"} else {"all"}}))?;
    let n = select.metrics.get("n_selected").and_then(Value::as_u64).unwrap_or(0);
    if select.status != Status::Succeeded || n == 0 {
        let why = select.error.map(|e| e.message).unwrap_or_else(|| "nothing matched".into());
        defects.push(format!("{EDIT_FAIL} \"{noun}\": {why}"));
        return Ok(candidate.to_string());
    }
    let selection = artifact(select, ArtifactKind::Selection)?;
    let edited = artifact(invoke(run, "remove_objects", json!({"candidate_id":candidate,"selection_id":selection}))?, ArtifactKind::Candidate)?;
    run.journal("decision", json!({"text": format!("Removed \"{noun}\" ({n} splats, {selection}) from {candidate} -> {edited}")}))?;
    let fill = completed(run, "inpaint_removed", json!({"candidate_id":edited,"selection_id":selection}))?;
    if fill.status == Status::Succeeded {
        let filled = artifact(fill, ArtifactKind::Candidate)?;
        defects.push(format!("{EDIT_NOTE} \"{noun}\" removed ({n} splats) and the hole filled by inpainting ({filled})."));
        return Ok(filled);
    }
    let why = fill.error.map(|e| e.message).unwrap_or_else(|| format!("{:?}", fill.status));
    defects.push(format!("{EDIT_NOTE} \"{noun}\" removed ({n} splats); the hole it leaves is not filled ({why})."));
    Ok(edited)
}

#[cfg(test)]
mod tests {
    use super::{FixedOptions, FixedOverrides};
    use serde_json::json;

    #[test]
    fn seed_and_eval_every_reach_the_training_request_only_when_given() {
        let plain = FixedOverrides::default().apply(FixedOptions::default());
        let mut request = json!({"sfm_id": "s", "split_id": "p"});
        plain.training_request(&mut request);
        assert!(request.get("seed").is_none() && request.get("eval_every").is_none(), "unset flags add no keys");
        let saved = serde_json::to_value(&plain).unwrap();
        assert!(saved.get("seed").is_none() && saved.get("eval_every").is_none(), "fixed.json is unchanged without the flags");

        let set = FixedOverrides { seed: Some(7), eval_every: Some(100), ..Default::default() }.apply(FixedOptions::default());
        let mut request = json!({"sfm_id": "s", "split_id": "p"});
        set.training_request(&mut request);
        assert_eq!((request["seed"].clone(), request["eval_every"].clone()), (json!(7), json!(100)), "flags reach the request");
        let resumed: FixedOptions = serde_json::from_value(serde_json::to_value(&set).unwrap()).unwrap();
        assert_eq!((resumed.seed, resumed.eval_every), (Some(7), Some(100)), "resume keeps them");

        let mut refine = json!({"total_steps": 20000});
        super::carry_training_controls(&request, &mut refine);
        assert_eq!((refine["seed"].clone(), refine["eval_every"].clone()), (json!(7), json!(100)), "derived refine keeps them");
        let mut refine = json!({"total_steps": 20000});
        super::carry_training_controls(&json!({"sfm_id": "s"}), &mut refine);
        assert_eq!(refine, json!({"total_steps": 20000}), "nothing added when the run gave none");
    }
}
