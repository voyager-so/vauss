//! Optional Spirula release-binary backend. All linked code is permissively licensed;
//! Spirula and FFmpeg are separately located subprocesses, never runtime Python.
mod complete;
mod config;
mod eval_bridge;
mod media;
mod preview;
mod progress;
mod split;
mod split_view;
mod util;

use anyhow::{Context, Result, ensure};
pub use config::SpirulaConfig;
pub use progress::parse_progress;
use serde::Serialize;
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::{Backend, RunCtx};
pub use split::verify_membership;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use util::*;

#[derive(Clone, Debug, Default)]
pub struct SpirulaBackend {
    pub config: SpirulaConfig,
}
impl SpirulaBackend {
    pub fn new(binary: impl Into<PathBuf>) -> Self {
        Self::with_config(SpirulaConfig {
            binary: binary.into(),
            ..Default::default()
        })
    }
    pub fn with_config(config: SpirulaConfig) -> Self {
        let _ = splat_eval::images::FFMPEG.set(config.ffmpeg.clone());
        Self { config }
    }
    fn job(
        &self,
        run: &dyn RunCtx,
        tool: &str,
        args: Vec<String>,
        artifact: Artifact,
    ) -> Result<ToolResult> {
        self.start(run, tool, self.config.argv(args)?, artifact)
    }
    fn start(&self, run: &dyn RunCtx, tool: &str, argv: Vec<String>, artifact: Artifact) -> Result<ToolResult> {
        let spec = JobSpec {
            tool: tool.into(),
            argv,
            env: self.config.env(),
            cwd: Some(string(run.root())?),
            on_complete: format!("spirula:complete_{tool}"),
            meta: meta(json!({"artifact_id":artifact.id})),
        };
        let job_id = run.start_job(spec)?;
        let mut r = result(tool);
        r.status = Status::Running;
        r.job_id = Some(job_id);
        r.artifacts.push(artifact);
        Ok(r)
    }
}
/// `auto` pairs always match exhaustively: Spirula's own auto uses sequential + loop closure for
/// short video, the configuration that flipped truck cameras. Frame caps keep exhaustive tractable.
fn spirula_pairs(requested: Pairs) -> String {
    match requested {
        Pairs::Auto => "exhaustive".into(),
        other => word(other),
    }
}

fn word<T: Serialize>(value: T) -> String {
    serde_json::to_value(value)
        .unwrap()
        .as_str()
        .unwrap()
        .into()
}
/// Grounding DINO box score a frame needs for the mask prompt to count as present (Spirula default 0.3).
const PRESENCE_DETECTOR_THRESHOLD: f32 = 0.4;
macro_rules! args { ($($value:expr),* $(,)?) => { vec![$($value.to_string()),*] }; }
impl Backend for SpirulaBackend {
    fn name(&self) -> &'static str {
        "spirula"
    }
    fn capabilities(&self) -> Value {
        json!({"backend":"spirula","inspect_input":true,"extract_frames":true,"mask_distractors":self.config.mask_available(),"reconstruct_sfm":true,"define_split":"interval","train_splats":true,"evaluate":true,"render_views":["heldout"],"extract_mesh":["ply","obj","glb"],"resume":true,"mid_training_eval":false,"binary":self.config.binary})
    }
    fn inspect_input(&self, run: &dyn RunCtx, req: InspectInput) -> Result<ToolResult> {
        guarded(run, "inspect_input", || media::inspect(self, run, req))
    }
    fn extract_frames(&self, run: &dyn RunCtx, req: ExtractFrames) -> Result<ToolResult> {
        guarded(run, "extract_frames", || media::extract(self, run, req))
    }
    fn mask_distractors(&self, run: &dyn RunCtx, req: MaskDistractors) -> Result<ToolResult> {
        guarded(run, "mask_distractors", || {
            let (_, frames) = get(run, &req.frames_id, ArtifactKind::Frames)?;
            if !self.config.mask_available() {
                return Ok(unsupported(
                    "mask_distractors",
                    "missing_mask_weights",
                    "Configure local SAM 2.1 GGML and Grounding DINO checkpoints, with vocab.txt beside the detector; models.json records their provenance.",
                ));
            }
            self.config.verify_models()?;
            let (artifact, out) = new_artifact(
                run,
                ArtifactKind::Masks,
                json!({"frames_id":req.frames_id,"purpose":req.purpose,"prompt":req.prompt}),
            )?;
            let track = args![
                "sam",
                "track",
                "--model",
                string(&self.config.sam_model)?,
                "--detector",
                string(&self.config.sam_detector)?,
                "--frames",
                string(&frames.join("images"))?,
                "--text",
                req.prompt,
                "--remove-prompted",
                "--device",
                0
            ];
            // Two passes. At Grounding DINO's default box score 0.3 a frame with no person still gets a
            // "person" box on its biggest object (spot robot, a rooftop), which then drops out of
            // training. The quiet 0.4 pass is the per-frame presence test: complete.rs empties every frame it
            // finds nothing on. The logged 0.3 pass draws the masks, so frames with a person keep every body
            // 0.3 finds (a single 0.4 pass dropped close-up people). research/mask-noperson/NOTES.md.
            // Every path stays its own argv token so remote jobs stage them.
            let mut argv = self.config.argv(vec![])?;
            let binary = argv.pop().context("Missing Spirula binary")?;
            argv.extend(args![
                "sh",
                "-c",
                format!(r#"b=$0 presence=$1 masks=$2; shift 2; mkdir -p "$presence" && "$b" "$@" --detector-threshold {PRESENCE_DETECTOR_THRESHOLD} --out "$presence" > "$presence.log" 2>&1 || {{ tail -40 "$presence.log"; exit 1; }}; exec "$b" "$@" --out "$masks""#),
                binary,
                string(&out.join("presence_masks"))?,
                string(&out.join("raw_masks"))?
            ]);
            argv.extend(track);
            self.start(run, "mask_distractors", argv, artifact)
        })
    }
    fn reconstruct_sfm(&self, run: &dyn RunCtx, req: ReconstructSfm) -> Result<ToolResult> {
        guarded(run, "reconstruct_sfm", || {
            let (frames, path) = get(run, &req.frames_id, ArtifactKind::Frames)?;
            let input = frames
                .meta
                .get("input_kind")
                .and_then(Value::as_str)
                .unwrap_or("images");
            let mask = if let Some(id) = &req.mask_id {
                let (mask, masks) = get(run, id, ArtifactKind::Masks)?;
                ensure!(
                    field(&mask, "frames_id")? == req.frames_id,
                    "Mask artifact belongs to different frames"
                );
                ensure!(
                    masks.join("coverage.json").is_file(),
                    "Mask artifact is not complete"
                );
                Some((mask, masks))
            } else {
                None
            };
            let pairs = spirula_pairs(req.pairs);
            let (artifact, out) = new_artifact(
                run,
                ArtifactKind::Sfm,
                json!({"frames_id":req.frames_id,"sparse_dir":"sparse/0","input_kind":input,"mask_id":req.mask_id,"pairs":pairs,"pairs_requested":word(req.pairs)}),
            )?;
            link_directory(&path.join("images"), &out.join("images"))?;
            let camera = if matches!(req.camera_model, CameraModel::Auto) {
                if input == "video" {
                    "opencv".into()
                } else {
                    "pinhole".into()
                }
            } else {
                word(req.camera_model)
            };
            let mut argv = args![
                "sfm",
                "auto",
                string(&out.join("images"))?,
                "-o",
                string(&out)?,
                "--quality",
                word(req.quality),
                "--data-type",
                if input == "video" {
                    "video"
                } else {
                    "individual"
                },
                "--features",
                "sift",
                "--camera-model",
                camera,
                "--camera-mode",
                "single",
                "--pairs",
                pairs,
                "--threads",
                16,
                "--device",
                0
            ];
            if let Some((mask, masks)) = mask {
                let purpose = field(&mask, "purpose")?;
                ensure!(
                    matches!(purpose, "sfm_features" | "training_ignore" | "both"),
                    "Unknown mask purpose: {purpose}"
                );
                if matches!(purpose, "sfm_features" | "both") {
                    argv.extend(args!["--feature-masks", string(&masks.join("masks"))?]);
                }
                // The subject guard may have unmasked static pixels or kept the masks SfM-only.
                if purpose != "sfm_features"
                    && let Some(training) = splat_eval::training_mask_dir(&masks)?
                {
                    link_directory(&training, &out.join("masks"))?;
                }
            }
            self.job(run, "reconstruct_sfm", argv, artifact)
        })
    }
    fn define_split(&self, run: &dyn RunCtx, req: DefineSplit) -> Result<ToolResult> {
        guarded(run, "define_split", || split::define(run, req))
    }
    fn train_splats(&self, run: &dyn RunCtx, req: TrainSplats) -> Result<ToolResult> {
        guarded(run, "train_splats", || {
            ensure!(
                (200..=60000).contains(&req.total_steps),
                "total_steps must be 200..60000"
            );
            ensure!(
                (10000..=10000000).contains(&req.cap_max),
                "cap_max must be 10000..10000000"
            );
            ensure!(
                (1..=8).contains(&req.resolution_divisor),
                "resolution_divisor must be 1..8"
            );
            let split = split::resolve(run, &req.split_id, &req.sfm_id)?;
            ensure!(
                req.resume_candidate_id.is_none() || req.init_candidate_id.is_none(),
                "Choose either checkpoint resume or PLY warm start"
            );
            let (sfm, dataset) = get(run, &req.sfm_id, ArtifactKind::Sfm)?;
            let cache_bytes = media::cpu_cache_bytes(
                &dataset.join("images"),
                &split.registered_names,
                req.resolution_divisor,
            )?;
            if cache_bytes > self.config.cpu_cache_max_bytes {
                let mut r = error(
                    "train_splats",
                    "cache_budget_exceeded",
                    "Spirula needs CPU image caching for complete held-out coverage. Reduce frame count/max_edge, increase resolution_divisor, or raise SPLAT_SPIRULA_CPU_CACHE_MAX_BYTES after checking available RAM.",
                );
                r.metrics = meta(
                    json!({"estimated_cpu_cache_bytes": cache_bytes, "cpu_cache_max_bytes": self.config.cpu_cache_max_bytes}),
                );
                return Ok(r);
            }
            let mut extra = vec![];
            if let Some(prior_id) = req
                .resume_candidate_id
                .as_ref()
                .or(req.init_candidate_id.as_ref())
            {
                let (prior, previous) = get(run, prior_id, ArtifactKind::Candidate)?;
                ensure!(
                    field(&prior, "split_id")? == req.split_id
                        && field(&prior, "sfm_id")? == req.sfm_id,
                    "Resume/warm start must use the original SfM and frozen split"
                );
                let (step, checkpoint) = complete::checkpoint(&previous)?;
                if req.resume_candidate_id.is_some() {
                    ensure!(
                        checkpoint.join("state.tar").is_file(),
                        "Candidate has no full checkpoint state.tar for resume"
                    );
                    ensure!(
                        req.total_steps > step,
                        "total_steps must exceed the saved checkpoint step"
                    );
                    extra = args!["--resume", string(&checkpoint)?];
                } else {
                    extra = args!["--init-ply", string(&checkpoint.join("splat.ply"))?];
                }
            }
            let interval_names = split.registered_names.iter().step_by(split.interval as usize).cloned().collect::<Vec<_>>();
            let eval_mode = if interval_names == split.heldout_names { "interval" } else { "filename" };
            let (artifact, out) = new_artifact(
                run,
                ArtifactKind::Candidate,
                json!({"sfm_id":req.sfm_id,"frames_id":split.frames_id,"split_id":req.split_id,"request":req,"backend":"spirula","eval_mode":eval_mode,"cache_images":"cpu","estimated_cpu_cache_bytes":cache_bytes}),
            )?;
            let dataset = if eval_mode == "filename" {
                split_view::create(&dataset, field(&sfm, "sparse_dir")?, &split, &out)?
            } else {
                dataset
            };
            let mut argv = args![
                "train",
                "3dgs",
                "--data",
                string(&dataset)?,
                "--output-dir-prefix",
                string(out.parent().context("Artifact must have a parent")?)?,
                "--output-dir-name",
                out.file_name()
                    .unwrap()
                    .to_str()
                    .context("Invalid artifact name")?,
                "--num-iterations",
                req.total_steps,
                "--cap-max",
                req.cap_max,
                "--train-resolution-divisor",
                req.resolution_divisor,
                // Disk-prefetched eval can consume next-epoch frames before all
                // first-epoch frames arrive, duplicating pairs and omitting views.
                "--cache-images",
                "cpu",
                "--floater-suppression",
                word(req.floater_suppression),
                "--distraction-robustness",
                word(req.distraction_robustness),
                "--eval-mode",
                eval_mode,
                "--eval-interval",
                split.interval,
                "--save-eval-images",
                "true",
                "--save-full-checkpoint",
                "true",
                "--steps-per-save",
                100,
                "--save-only-latest-checkpoint",
                "true",
                "--keep-viewer-alive",
                "false",
                "--disable-viewer",
                "true",
                "--device",
                0
            ];
            argv.extend(extra);
            if dataset.join("masks").exists() {
                // Distractor masks: false = exclude masked pixels from the loss (true would train them as empty space).
                argv.extend(args!["--apply-loss-for-mask", "false"]);
            }
            self.job(run, "train_splats", argv, artifact)
        })
    }
    fn evaluate(&self, run: &dyn RunCtx, req: Evaluate) -> Result<ToolResult> {
        guarded(run, "evaluate", || {
            let start = Instant::now();
            let (candidate, path) = get(run, &req.candidate_id, ArtifactKind::Candidate)?;
            ensure!(
                field(&candidate, "split_id")? == req.split_id,
                "Candidate was trained against a different split"
            );
            let membership = verify_membership(run, &candidate)?;
            if membership["verified"] != true {
                let mut r = error(
                    "evaluate",
                    "split_mismatch",
                    "Saved eval pairs do not match the frozen split",
                );
                r.metrics = meta(membership);
                return Ok(r);
            }
            let (artifact, out) = new_artifact(
                run,
                ArtifactKind::Eval,
                json!({"candidate_id":req.candidate_id,"split_id":req.split_id}),
            )?;
            let scores =
                eval_bridge::evaluate(&path, req.lpips, self.config.models_dir.as_deref())?;
            write(&out.join("metrics.json"), &scores)?;
            let mut worst = scores["per_image"]
                .as_array()
                .context("Missing per_image scores")?
                .clone();
            worst.sort_by(|a, b| {
                a["psnr"]
                    .as_f64()
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&b["psnr"].as_f64().unwrap_or(f64::NEG_INFINITY))
            });
            for row in &mut worst {
                let pair = row["image"].as_str().context("Missing score image")?;
                row["source_name"] = membership["pairs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|p| p["pair"] == pair)
                    .context("Unmatched scored pair")?["source_name"]
                    .clone();
            }
            let pairs = worst
                .iter()
                .take(8)
                .map(|r| r["image"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            let comparisons = complete::comparisons(&path, &out, &pairs)?;
            let mut r = result("evaluate");
            r.artifacts.push(artifact);
            r.metrics = meta(scores["mean"].clone());
            r.metrics.extend(meta(json!({"n_eval":scores["n"],"heldout_coverage":1.0,"split_verified":true,"worst_views":worst.into_iter().take(8).collect::<Vec<_>>()})));
            if let Some(reason) = scores["lpips_unavailable"].as_str() {
                r.warnings.push(format!("LPIPS unavailable: {reason}"));
                r.metrics.insert("lpips_unavailable".into(), json!(reason));
            }
            if let Some(note) = scores["lpips_note"].as_str() {
                r.metrics.insert("lpips_note".into(), json!(note));
            }
            if let Some(scale) = scores["lpips_scale"].as_str() {
                r.metrics.insert("lpips_scale".into(), json!(scale));
            }
            let psnr = r
                .metrics
                .get("psnr")
                .and_then(Value::as_f64)
                .unwrap_or(f64::INFINITY);
            if r.metrics
                .get("cc_psnr")
                .and_then(Value::as_f64)
                .unwrap_or(psnr)
                - psnr
                > 1.5
            {
                r.warnings.push(
                    "Colour correction adds more than 1.5 dB; inspect exposure variation.".into(),
                );
            }
            if scores["n"].as_u64().unwrap_or(0) < 8 {
                r.warnings.push("Fewer than eight held-out views.".into());
            }
            write(&out.join("summary.json"), &r.metrics)?;
            r.images = comparisons
                .iter()
                .map(|p| rel(run, p))
                .collect::<Result<_>>()?;
            r.wall_s = Some(start.elapsed().as_secs_f64());
            Ok(r)
        })
    }
    fn render_views(&self, run: &dyn RunCtx, req: RenderViews) -> Result<ToolResult> {
        guarded(run, "render_views", || {
            if !matches!(req.views, ViewSet::Heldout) {
                return Ok(unsupported(
                    "render_views",
                    "unsupported_capability",
                    "Spirula headless mode exposes saved held-out pairs only",
                ));
            }
            ensure!((1..=48).contains(&req.max_views), "max_views must be 1..48");
            let (candidate, path) = get(run, &req.candidate_id, ArtifactKind::Candidate)?;
            let membership = verify_membership(run, &candidate)?;
            if membership["verified"] != true {
                let mut r = error(
                    "render_views",
                    "split_mismatch",
                    "Saved eval pairs do not match the frozen split",
                );
                r.metrics = meta(membership);
                return Ok(r);
            }
            let (artifact, out) = new_artifact(
                run,
                ArtifactKind::Images,
                json!({"candidate_id":req.candidate_id,"views":"heldout"}),
            )?;
            let pairs = membership["pairs"]
                .as_array()
                .unwrap()
                .iter()
                .take(req.max_views as usize)
                .map(|r| r["pair"].as_str().unwrap().into())
                .collect::<Vec<String>>();
            let comparisons = complete::comparisons(&path, &out, &pairs)?;
            let mut r = result("render_views");
            r.artifacts.push(artifact);
            r.metrics = meta(
                json!({"view_count":pairs.len(),"layout":"ground_truth|render","split_verified":true}),
            );
            r.images = comparisons
                .iter()
                .map(|p| rel(run, p))
                .collect::<Result<_>>()?;
            Ok(r)
        })
    }
    fn extract_mesh(&self, run: &dyn RunCtx, req: ExtractMesh) -> Result<ToolResult> {
        guarded(run, "extract_mesh", || {
            let (candidate, path) = get(run, &req.candidate_id, ArtifactKind::Candidate)?;
            complete::checkpoint(&path)?;
            if matches!(
                (req.format, req.color),
                (MeshFormat::Obj, MeshColor::Vertex) | (MeshFormat::Ply, MeshColor::Texture)
            ) {
                return Ok(unsupported(
                    "extract_mesh",
                    "unsupported_capability",
                    "Spirula does not support OBJ+vertex or PLY+texture",
                ));
            }
            ensure!(
                (256..=8192).contains(&req.texture_size),
                "texture_size must be 256..8192"
            );
            let (artifact, out) = new_artifact(
                run,
                ArtifactKind::Mesh,
                json!({"candidate_id":req.candidate_id,"format":req.format,"color":req.color}),
            )?;
            self.job(
                run,
                "extract_mesh",
                args![
                    "mesh",
                    string(&path)?,
                    "--data",
                    string(&if candidate.meta.get("eval_mode").and_then(Value::as_str) == Some("filename") {
                        path.join("training_data")
                    } else {
                        run.path(field(&candidate, "sfm_id")?)?
                    })?,
                    "--output",
                    string(&out.join("mesh"))?,
                    "--format",
                    word(req.format),
                    "--color",
                    word(req.color),
                    "--texture-size",
                    req.texture_size,
                    "--num-threads",
                    16,
                    "--device",
                    0
                ],
                artifact,
            )
        })
    }
    fn on_job_complete(
        &self,
        run: &dyn RunCtx,
        handler: &str,
        job_dir: &Path,
        exit_code: i32,
    ) -> Result<ToolResult> {
        complete::finish(run, handler, job_dir, exit_code)
    }
    fn job_progress(&self, run: &dyn RunCtx, job_dir: &Path) -> Value {
        (|| -> Result<Value> {
            let spec: JobSpec = read(&job_dir.join("request.json"))?;
            if spec.tool != "train_splats" {
                return Ok(Value::Null);
            }
            let id = spec.meta["artifact_id"]
                .as_str()
                .context("Missing job artifact_id")?;
            progress::refresh(
                &job_dir.join("stdout.log"),
                &run.path(id)?.join("training.jsonl"),
            )
        })()
        .unwrap_or_else(|e| json!({"progress_error":format!("{e:#}")}))
    }
}
