//! Structure from motion via splat-sfm (COLMAP 4.x underneath), run as a detached worker job.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splat_contracts::*;
use splat_core::RunCtx;

use crate::model;
use crate::util::*;
use crate::{BrushConfig, start_worker};

#[derive(Clone, Serialize, Deserialize)]
pub struct SfmJob {
    pub colmap: PathBuf,
    /// A calibrated export supplies a client-side pose fallback after the solve.
    #[serde(default)]
    pub capture: bool,
    /// Video at quick (low) quality: focal prior, fixed while mapping.
    #[serde(default)]
    pub quick_video: bool,
    /// A quick first-view solve ([`ReconstructSfm::preview`]): passes on the lower preview floor.
    #[serde(default)]
    pub preview: bool,
    pub dir: PathBuf,
    pub camera_model: String,
    pub video: bool,
    pub max_image_size: usize,
    pub max_features: usize,
    pub sequential: bool,
    #[serde(default)]
    pub pairs: Option<Pairs>,
    #[serde(default)]
    pub preserve_all: bool,
    pub mask_dir: Option<PathBuf>,
    #[serde(default)]
    pub mask_prompt: Option<String>,
    #[serde(default)]
    pub sam: Option<crate::select::SamTool>,
    /// The detector reads this sibling implicitly; record it so remote staging includes it.
    #[serde(default)]
    pub sam_vocab: Option<PathBuf>,
    /// Exposure-lifted copies of dark frames (frames/analysis/), used for feature extraction only.
    #[serde(default)]
    pub analysis_dir: Option<PathBuf>,
    /// Larger copies of the photos (frames/sfm_images/, same names): COLMAP extracts features and maps
    /// from these, and the model is then rescaled to the training images. Never with `analysis_dir`.
    #[serde(default)]
    pub sfm_images: Option<PathBuf>,
    pub gpu_lock: Option<PathBuf>,
    #[serde(default)]
    pub use_gpu: bool,
    #[serde(default)]
    pub vocab_tree: Option<PathBuf>,
    /// DA3-BASE weights for the learned-pose rescue ([`learned_fallback`]); None disables it. Named
    /// here so remote staging ships the file (sha256 cache) to the SfM job.
    #[serde(default)]
    pub da3_weights: Option<PathBuf>,
    /// ALIKED + LightGlue ONNX models for [`aliked_fallback`] (detector, matcher); None disables it.
    /// Named here so remote staging ships them to the SfM job.
    #[serde(default)]
    pub aliked: Option<(PathBuf, PathBuf)>,
    /// COLMAP mapper seed ([`splat_contracts::ReconstructSfm::mapper_seed`]); None keeps the default.
    #[serde(default)]
    pub mapper_seed: Option<u64>,
    /// Cloud quick solve: the first-view preview this job trains after the solve
    /// ([`ReconstructSfm::preview_train`], [`crate::train::run_chained`]).
    #[serde(default)]
    pub preview_train: Option<crate::train::ChainedPreview>,
    /// The video's lens field of view from the capture pre-check, for [`focal_guard`].
    #[serde(default)]
    pub focal_prior: Option<FocalPrior>,
}

/// A video's field of view across its long edge before any SfM: the file's lens tag, the phone
/// model's measured value, self-calibrated from the camera's turns (precheck), or an assumed 1x
/// lens; `source` says which.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FocalPrior {
    pub long_fov_deg: f64,
    pub source: String,
    /// Turning pairs behind a self-calibrated value (0 otherwise).
    #[serde(default)]
    pub pairs: u64,
    /// The source video's size (the pre-check's frame; extraction may crop black bars, then scale).
    pub source_size: (u32, u32),
}

/// Self-calibrated fields of view from fewer turning pairs don't decide that SfM's focal is off
/// (singapore: 5 pairs, 75-78 deg against a sane 63 deg SfM camera); they still set the re-solve's focal.
const FOCAL_PRIOR_MIN_PAIRS: u64 = 10;
/// SfM focal over the prior's beyond which the camera is implausible. Bench rooms (2026-10-12):
/// sane solves 0.79-1.26x the pre-check's value; the drifted ones 1.40-5.9x.
const FOCAL_PRIOR_RATIO: f64 = 1.4;
/// The fixed-focal re-solve may place this share of the free solve's frames: a plausible camera beats a
/// few extra frames on a degenerate one (busbee on fal: 532 against 535 at f 4672 px, k2 -24).
const FOCAL_RESOLVE_MIN_PLACED: f64 = 0.98;

impl FocalPrior {
    /// Reads the run's capture pre-check (capture_check.json); None when it didn't run (photos).
    fn from_capture_check(run: &dyn RunCtx) -> Option<Self> {
        // The check runs beside frame extraction and usually ends first; the job's upload waits for
        // it anyway (splat_core::remote's capture gate, 20 s), so wait as long here.
        let pending = run.root().join(splat_core::report::CAPTURE_PENDING);
        let start = std::time::Instant::now();
        while pending.exists() && start.elapsed().as_secs() < 20 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let check: serde_json::Value = read_json(&run.root().join(splat_core::report::CAPTURE_CHECK)).ok()?;
        Self::from_check_metrics(&check["metrics"])
    }

    /// The prior from capture_check.json's metrics.
    fn from_check_metrics(m: &serde_json::Value) -> Option<Self> {
        let fov = m["fov_deg"].as_array()?.iter().filter_map(|v| v.as_f64()).fold(f64::NAN, f64::max);
        let size = (m["width"].as_u64()? as u32, m["height"].as_u64()? as u32);
        let mut source = m["fov_source"].as_str().unwrap_or("unknown").to_string();
        let pairs = m["fov_estimate"]["pairs"].as_u64().unwrap_or(0);
        // A phone-model value can't tell which of the phone's lenses shot the video (room 4: a Pixel
        // 10 Pro 1x clip, f 1435 px at 1600, against a 95 deg ultrawide table value); this video's own
        // turns can, so enough of them replace it.
        let estimate = m["fov_estimate"]["long_deg"].as_f64().filter(|f| *f > 10.0 && *f < 170.0);
        let fov = match estimate {
            Some(est) if source == "phone model" && pairs >= FOCAL_PRIOR_MIN_PAIRS => {
                source = "estimated".into();
                est
            }
            _ => fov,
        };
        (fov > 10.0 && fov < 170.0 && size.0 > 0 && size.1 > 0).then(|| FocalPrior { long_fov_deg: fov, source, pairs, source_size: size })
    }

    /// Whether disagreeing with it marks SfM's focal implausible: a lens tag or a self-calibration
    /// from enough turns, both measured on this video's lens. A phone-model value (any of the
    /// phone's lenses) or an assumed lens only seeds the re-solve.
    fn trusted(&self) -> bool {
        match self.source.as_str() {
            "lens tag" => true,
            "estimated" => self.pairs >= FOCAL_PRIOR_MIN_PAIRS,
            _ => false,
        }
    }

    /// Focal length (px) for a `w x h` frame of this video: black-bar crops trim one axis before the
    /// uniform scale, so the larger of the two axis ratios is the scale.
    fn focal_px(&self, w: u64, h: u64) -> f64 {
        let (sw, sh) = (self.source_size.0 as f64, self.source_size.1 as f64);
        let scale = (w as f64 / sw).max(h as f64 / sh);
        0.5 * sw.max(sh) / (self.long_fov_deg.to_radians() / 2.0).tan() * scale
    }
}
/// CUDA builds say so in `colmap -h`'s first line ("... with CUDA ..."). Only those get use_gpu=1
/// and take the shared GPU lock; CPU builds (macOS, Windows-nocuda, our Linux build) need neither.
fn colmap_has_cuda(colmap: &std::path::Path) -> bool {
    std::process::Command::new(colmap)
        .arg("-h")
        .env("QT_QPA_PLATFORM", "offscreen")
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).lines().next().is_some_and(|l| l.contains("with CUDA")))
}

fn calibration_model(model: CameraModel, video: bool) -> anyhow::Result<&'static str> {
    Ok(match model {
        CameraModel::Auto if video => "RADIAL",
        CameraModel::Auto | CameraModel::Pinhole => "PINHOLE",
        CameraModel::Opencv => "OPENCV",
        CameraModel::OpencvFisheye => return Err(invalid("fisheye SfM is not supported by the brush backend yet")),
    })
}

fn matching_policy(job: &SfmJob) -> splat_sfm::Matching {
    use splat_sfm::Matching;
    match job.pairs {
        Some(Pairs::Exhaustive) => Matching::Exhaustive,
        Some(Pairs::Sequential) => Matching::Sequential,
        Some(Pairs::Auto) if job.video && job.vocab_tree.is_some() => Matching::Hybrid,
        Some(Pairs::Auto) if job.video => Matching::Sequential,
        Some(Pairs::Auto) => Matching::Exhaustive,
        // Already staged legacy jobs retain the policy encoded by their old fields.
        None if job.sequential => Matching::Sequential,
        None if job.vocab_tree.is_some() => Matching::Hybrid,
        None => Matching::Exhaustive,
    }
}

fn matching_name(job: &SfmJob) -> &'static str {
    match matching_policy(job) {
        splat_sfm::Matching::Sequential => "sequential",
        splat_sfm::Matching::Hybrid => "hybrid",
        splat_sfm::Matching::Exhaustive => "exhaustive",
    }
}

pub fn reconstruct_sfm(cfg: &BrushConfig, run: &dyn RunCtx, req: ReconstructSfm) -> anyhow::Result<ToolResult> {
    let (frames, frames_dir) = get_artifact(run, &req.frames_id, ArtifactKind::Frames)?;
    if frames.meta.get("trust_poses") == Some(&json!(true)) {
        anyhow::ensure!(matches!(req.camera_model, CameraModel::Auto) && matches!(req.pairs, Pairs::Auto)
            && req.mask_id.is_none() && req.mask_prompt.is_none() && !req.preview
            && req.preview_train.is_none() && req.mapper_seed.is_none(),
            "trusted camera authority is incompatible with SfM overrides, masks or previews");
        return crate::capture_sfm::import_trusted(run, &req.frames_id, cfg.colmap.as_deref());
    }
    let colmap = cfg.colmap.clone().ok_or_else(|| {
        invalid("COLMAP not found: set SPLAT_COLMAP to a COLMAP 4.x binary (or install it on PATH)")
    })?;
    let video = frames.meta.get("input_kind").and_then(|v| v.as_str()) == Some("video");
    let capture = crate::capture_sfm::is_capture(&frames_dir);
    let (artifact, out) = new_artifact(
        run,
        ArtifactKind::Sfm,
        json!({"frames_id": req.frames_id, "sparse_dir": "sparse/0", "input_kind": if capture { "capture" } else if video { "video" } else { "images" }, "mask_id": req.mask_id, "mask_prompt": req.mask_prompt, "engine": if capture { "colmap_with_capture_fallback" } else { "colmap" }}),
    )?;
    // The worker and chained preview need the supplied K prior before client completion.
    if capture { link_or_copy(&frames_dir.join("capture.json"), &out.join("capture.json"))?; }
    // Brush resolves images by name inside the dataset dir, so the dataset holds its own images.
    let images = out.join("images");
    std::fs::create_dir_all(&images)?;
    for src in list_images(&frames_dir.join("images"))? {
        let name = src.strip_prefix(frames_dir.join("images"))?;
        let dst = images.join(name);
        if let Some(p) = dst.parent() {
            std::fs::create_dir_all(p)?;
        }
        link_or_copy(&src, &dst)?;
    }
    let mut mask_dir = None;
    if let Some(mask_id) = &req.mask_id {
        let (mask, masks) = get_artifact(run, mask_id, ArtifactKind::Masks)?;
        if mask.meta.get("frames_id").and_then(|v| v.as_str()) != Some(req.frames_id.as_str()) {
            return Err(invalid("mask artifact belongs to different frames"));
        }
        if !masks.join("coverage.json").is_file() {
            return Err(invalid(format!("{mask_id} is not complete; wait for mask_distractors to finish")));
        }
        let purpose = mask.meta.get("purpose").and_then(|v| v.as_str()).unwrap_or("both");
        if matches!(purpose, "sfm_features" | "both") {
            mask_dir = Some(masks.join("masks"));
        }
        // Brush's loader pairs images/<x>.jpg with masks/<x>.png and multiplies the loss by it
        // (white keeps, black ignores), so masked distractors never pull on the splats.
        // The subject guard may have unmasked static pixels or kept the masks SfM-only.
        let training = matches!(purpose, "training_ignore" | "both").then(|| splat_eval::training_mask_dir(&masks)).transpose()?.flatten();
        if let Some(training) = training {
            for src in list_images(&training)? {
                let dst = out.join("masks").join(src.strip_prefix(&training)?);
                if let Some(p) = dst.parent() {
                    std::fs::create_dir_all(p)?;
                }
                link_or_copy(&src, &dst)?;
            }
            // Detector masks, not a user's: training keeps them only while they hide movers.
            write_json(&out.join(crate::train::DETECTOR_MASKS_MARKER), &json!({"source": "mask_distractors", "mask_id": mask_id}))?;
        }
    }
    // Photo alpha from extract_frames (frames/masks/, COLMAP names, 0 = ignore) masks features
    // when no distractor masks were requested.
    if mask_dir.is_none() && frames_dir.join("masks").is_dir() {
        mask_dir = Some(frames_dir.join("masks"));
    }
    let analysis_dir = Some(frames_dir.join("analysis")).filter(|d| d.is_dir());
    // Photos larger than the training frames: SfM on extract_frames' larger copies (see media::SFM_EDGE),
    // unless the exposure-lift overlay applies. Every training frame must have its copy.
    let sfm_images = Some(frames_dir.join("sfm_images")).filter(|d| analysis_dir.is_none() && d.is_dir()).filter(|d| {
        let names = |dir: &std::path::Path| -> Option<Vec<PathBuf>> {
            Some(list_images(dir).ok()?.iter().filter_map(|p| p.strip_prefix(dir).ok().map(PathBuf::from)).collect())
        };
        names(d).is_some_and(|n| Some(n) == names(&frames_dir.join("images")))
    });
    let (max_image_size, max_features) = match req.quality {
        Quality::Low => (1000, 4096),
        Quality::Medium => (1600, 8192),
        Quality::High => (3200, 16384),
    };
    // Preserve the explicit request through staging: video auto gets loop-aware sequential matching.
    let sequential = matches!(req.pairs, Pairs::Sequential);
    // Video: one shared camera with fx = fy (phone video has square pixels). OPENCV's free
    // fx/fy self-calibrated anisotropic, mismatched focals between the quick and full SfM of
    // one video (house demo: 383/505 at 960x540 vs 817/581 at 1600x900). RADIAL keeps
    // OPENCV's two radial terms.
    let camera_model = calibration_model(req.camera_model, video)?;
    let use_gpu = colmap_has_cuda(&colmap);
    // Only cloud jobs chain the preview: they save a second upload, queue and result round trip
    // (locally the separate train job costs nothing extra). Captures may switch to sensor poses
    // on the client after the solve, so they keep the separate preview job.
    let preview_train = match req.preview_train.as_ref().filter(|_| req.preview && !capture && splat_core::remote::mode().is_some()) {
        Some(train) => {
            let (_, candidate) = run.new_artifact_dir(ArtifactKind::Candidate)?;
            Some(crate::train::chained_preview(cfg, &artifact.id, &out, &candidate, train, video, frames.meta.get("preserve_all") == Some(&json!(true)))?)
        }
        None => None,
    };
    let job = SfmJob {
        capture,
        dir: out.clone(),
        camera_model: camera_model.into(),
        video,
        quick_video: video && matches!(req.quality, Quality::Low),
        preview: req.preview,
        max_image_size,
        max_features,
        sequential,
        pairs: Some(req.pairs),
        preserve_all: frames.meta.get("preserve_all") == Some(&json!(true)),
        mask_dir,
        sam: req.mask_prompt.as_ref().and(cfg.sam.clone()),
        sam_vocab: req.mask_prompt.as_ref().and(cfg.sam.as_ref())
            .and_then(|sam| sam.detector.parent().map(|p| p.join("vocab.txt"))),
        mask_prompt: req.mask_prompt,
        analysis_dir,
        sfm_images,
        // splat-sfm takes it only for stages that use the GPU (CUDA COLMAP, the learned seed),
        // so CPU-only COLMAP still doesn't queue behind GPU jobs.
        gpu_lock: cfg.gpu_lock.clone(),
        use_gpu,
        colmap,
        vocab_tree: if video && !matches!(req.pairs, Pairs::Exhaustive) { cfg.video_vocab_tree()? } else { None },
        // Sensor poses are the coverage fallback; a learned pose/training rescue
        // would add another competing trajectory and an unnecessary paid fit.
        da3_weights: (!capture).then(|| da3_weights(cfg)).flatten(),
        aliked: cfg.aliked_dir.as_ref()
            .map(|d| (d.join("aliked-n16rot.onnx"), d.join("aliked-lightglue.onnx")))
            .filter(|(a, b)| a.is_file() && b.is_file()),
        preview_train,
        mapper_seed: req.mapper_seed,
        focal_prior: video.then(|| FocalPrior::from_capture_check(run)).flatten(),
    };
    write_json(&out.join("sfm_job.json"), &job)?;
    let mut r = start_worker(cfg, run, "reconstruct_sfm", "sfm", &out.join("sfm_job.json"), &artifact)?;
    r.metrics.insert("matching".into(), json!(matching_name(&job)));
    r.metrics.insert("matching_requested".into(), json!(job.pairs));
    r.metrics.insert("loop_detection".into(), json!(!matches!(matching_policy(&job), splat_sfm::Matching::Exhaustive) && job.vocab_tree.is_some()));
    if video && !matches!(req.pairs, Pairs::Exhaustive) && job.vocab_tree.is_none() {
        r.warnings.push("No vocabulary tree configured: sequential matching uses quadratic long-range pairs without visual loop detection.".into());
    }
    Ok(r)
}

/// The quick video SfM's last resort after both attempts flipped: map again without the frames
/// the continuity check flagged, when they are a small minority (a bad registration, not a bad
/// map). House's quick SfM flipped frame 784 alone (104 deg off the full SfM, which is clean and
/// selects no frame in that 770-861 stretch) in both attempts, since the retry reuses the same
/// database; the preview then waited for the full SfM. Errors with the reason when not applicable.
fn quick_without_flipped(dir: &std::path::Path, colmap_dir: &std::path::Path, feature_images: &std::path::Path, opts: &mut splat_sfm::ReconstructOptions) -> anyhow::Result<splat_sfm::Reconstruction> {
    let rec: serde_json::Value = read_json(&colmap_dir.join("reconstruction.json"))?;
    let continuity = &rec["health"]["pose_continuity"];
    let flipped: std::collections::HashSet<String> =
        continuity["flipped_frames"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(str::to_owned)).collect();
    let registered = rec["health"]["n_registered"].as_u64().unwrap_or(0) as usize;
    anyhow::ensure!(!flipped.is_empty() && flipped.len() * 4 <= registered,
        "quick SfM flipped {} of {registered} frames in both attempts; too many to drop", flipped.len());
    println!("[sfm] quick SfM flipped again; mapping without {} flipped frame(s): {:?}", flipped.len(), flipped);
    let without = dir.join("feature_images_unflipped");
    let _ = std::fs::remove_dir_all(&without);
    for src in list_images(feature_images)? {
        let name = src.strip_prefix(feature_images)?;
        if flipped.contains(name.to_string_lossy().as_ref()) {
            continue;
        }
        let dst = without.join(name);
        if let Some(p) = dst.parent() {
            std::fs::create_dir_all(p)?;
        }
        link_or_copy(&src, &dst)?;
    }
    let retained = dir.join("colmap-flipped2");
    let _ = std::fs::remove_dir_all(&retained);
    std::fs::rename(colmap_dir, &retained)?;
    // A different image set: features are extracted again (the quick SfM is ~24 small frames).
    opts.reuse_database = None;
    splat_sfm::reconstruct(&without, opts)
}

/// Fewest frames of one coherent section worth training when under half the capture is placed.
const MIN_SECTION_FRAMES: u64 = 15;

/// One group dominates a split solve: the largest holds at least twice the frames of the next
/// (guildhall 258/71/52/...: one walk with gaps). Balanced groups more likely mix scenes and fail with
/// the separate-groups advice instead of training a section.
fn dominant_group(sizes: &[u64]) -> bool {
    let mut sorted = sizes.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    sorted.len() < 2 || sorted[0] >= 2 * sorted[1]
}

/// "frames <first>–<last> (<a>–<b>% of the capture)" for the registered frames of `dir`'s sparse/0, by
/// sorted image name; None if the model or images can't be read.
fn registered_section(dir: &std::path::Path) -> Option<String> {
    let model = splat_eval::read_model(&dir.join("sparse/0")).ok()?;
    let registered: std::collections::BTreeSet<String> = model.images.values().map(|i| i.name.clone()).collect();
    let names: Vec<String> = list_images(&dir.join("images")).ok()?.iter()
        .filter_map(|p| p.strip_prefix(dir.join("images")).ok().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    let at: Vec<usize> = names.iter().enumerate().filter(|(_, n)| registered.contains(*n)).map(|(i, _)| i).collect();
    let (first, last) = (*at.first()?, *at.last()?);
    let pct = |i: usize| 100 * i / names.len().max(1);
    let stem = |n: &str| std::path::Path::new(n).file_stem().map_or_else(|| n.to_owned(), |s| s.to_string_lossy().into_owned());
    Some(format!("frames {}–{} ({}–{}% of the capture)", stem(&names[first]), stem(&names[last]), pct(first), pct(last + 1)))
}

/// The configured DA3 weights (BrushConfig::da3_weights; the CLI fills it from SPLAT_DA3_WEIGHTS, the
/// models dir, the dev host or the da3-base asset cache), when the file exists.
fn da3_weights(cfg: &BrushConfig) -> Option<PathBuf> {
    cfg.da3_weights.clone().filter(|p| p.is_file())
}

/// Learned-pose rescue (DA3 seed + COLMAP triangulation/BA/mapper, crates/splat-sfm-learned) for
/// solves the product rejects anyway: COLMAP placed fewer than half the frames ([`complete`]'s
/// too_few_registered; sparse orbits: bicycle 24 views 3/24, groei timelapse 2/43). Such a model
/// is no baseline, so the guard keeps the learned one only if it is trainable, and it is used only
/// if it places at least half the frames. Solves at or above half never run it: there COLMAP is
/// trainable and the guard would need a held-out training check (paf-museum: learned placed
/// 220/220 vs 104 and rendered 1.9 dB worse). Any failure keeps COLMAP's result.
/// The learned rescue also runs (as a union, see [`learned_fallback`]) when COLMAP places fewer than
/// this fraction of the frames: goal 2's solve bar.
const RESCUE_BELOW_FRACTION: f64 = 0.9;
/// Above this many frames the half-to-90% band keeps COLMAP's model without running DA3. The
/// extraction's 600-frame cap: at 5 frames/s, arkit-47333904 (409 frames) lost its union under the
/// old 400 (COLMAP 94/176 -> union 136/176 at 2 fps; 247/409 and no rescue at 5 fps).
const RESCUE_BAND_MAX_FRAMES: usize = 600;
/// Above this many frames a below-half COLMAP model is kept without trying a learned replacement:
/// kept 0 of 4 times at 375-612 frames (sfm-ladder-plan.md 2026-10-12: living-spring 1406 s,
/// singapore 343 s, arkit-41125285 221 and 162 s of rescue). A COLMAP solve that made no model still
/// tries it (there is nothing else to keep).
const RESCUE_REPLACE_MAX_FRAMES: usize = 300;
/// A frame the union adds must observe this many 3D points (the leftover pass's floor).
const RESCUE_UNION_MIN_OBS: usize = 20;

/// A rescue (replacement or union) of a COLMAP solve that made a model must hold up on frames both placed:
/// [`rescue_check`] fits each model briefly and compares held-out PSNR on every RESCUE_CHECK_EVERY-th
/// shared frame (neither fit trains on them). Fraction placed doesn't predict a replacement's quality:
/// guildhall's (616/623 vs COLMAP 258) rendered 14.7 dB, medyk's (135/135 vs 46) 34.4 dB.
const RESCUE_CHECK_STEPS: u32 = 3000;
const RESCUE_CHECK_EVERY: usize = 8;
const RESCUE_CHECK_MIN_VIEWS: usize = 3;
/// A replacement is kept unless it scores more than this below COLMAP's fit on the shared held-out views.
const RESCUE_CHECK_MARGIN_DB: f64 = 0.2;
/// A union keeps COLMAP's frames and only adds the rest, so it gets a wider margin: dev unions with big
/// coverage gains sit at -0.17 to -0.20 dB (shengli-night-market 148 -> 203, copped-hall-first-floor/360
/// 304 -> 352); the bad union we know (-1.7 dB) is still rejected.
const RESCUE_CHECK_UNION_MARGIN_DB: f64 = 0.5;
const RESCUE_CHECK_MAX_SPLATS: u32 = 500_000;
const RESCUE_CHECK_MAX_RESOLUTION: u32 = 960;

/// Held-out check of a rescued model against COLMAP's (see [`RESCUE_CHECK_STEPS`]): the same short
/// preview fit (same seed, steps, cap) of each, as `__brush train` subprocesses, scored on the same
/// shared frames. Returns the record for health.json; `keep` is the decision.
/// `colmap_fit` caches COLMAP's score by held-out frames, so a second candidate with the same shared
/// frames doesn't refit it.
fn rescue_check(job: &SfmJob, colmap_model: &std::path::Path, rescue_model: &std::path::Path, out: &std::path::Path, margin_db: f64,
    colmap_fit: &mut Option<(Vec<String>, f64)>) -> anyhow::Result<serde_json::Value> {
    use anyhow::Context;
    let start = std::time::Instant::now();
    let _ = std::fs::remove_dir_all(out);
    let colmap_names = model::names(colmap_model)?;
    let rescue_names = model::names(rescue_model)?;
    let in_rescue: std::collections::BTreeSet<&String> = rescue_names.iter().collect();
    let mut common: Vec<&String> = colmap_names.iter().filter(|n| in_rescue.contains(n)).collect();
    common.sort();
    let heldout: Vec<String> = common.iter().skip(RESCUE_CHECK_EVERY / 2).step_by(RESCUE_CHECK_EVERY).map(|n| (*n).clone()).collect();
    anyhow::ensure!(heldout.len() >= RESCUE_CHECK_MIN_VIEWS, "{} shared frames: too few held-out views", common.len());
    let images = job.dir.join("images");
    let exe = match std::env::var_os("SPLAT_BRUSH_EXE") {
        Some(p) => PathBuf::from(p),
        None => std::env::current_exe()?,
    };
    let mut psnr = serde_json::Map::new();
    let mut n_train = serde_json::Map::new();
    for (arm, model_dir, names) in [("colmap", colmap_model, &colmap_names), ("rescue", rescue_model, &rescue_names)] {
        n_train.insert(arm.into(), json!(names.len() - heldout.len()));
        if let Some((_, p)) = colmap_fit.as_ref().filter(|(h, _)| arm == "colmap" && *h == heldout) {
            psnr.insert(arm.into(), json!(*p));
            continue;
        }
        let dir = out.join(arm);
        let dataset = dir.join("dataset");
        std::fs::create_dir_all(dataset.join("sparse/0"))?;
        for entry in std::fs::read_dir(model_dir)? {
            let p = entry?.path();
            if p.is_file() {
                std::fs::copy(&p, dataset.join("sparse/0").join(p.file_name().context("model file name")?))?;
            }
        }
        for n in names {
            let dst = dataset.join("images").join(n);
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            link_or_copy(&images.join(n), &dst)?;
        }
        let train_out = dir.join("train");
        std::fs::create_dir_all(&train_out)?;
        // A preview fit, scored by the trainer's mid-run held-out eval at RESCUE_CHECK_STEPS (one step
        // before the end): previews skip the final checkpoint and scoring work.
        let train_job = crate::train::TrainJob {
            dataset,
            out: train_out.clone(),
            preview: true,
            video: job.video,
            train_names: names.iter().filter(|n| !heldout.contains(*n)).cloned().collect(),
            heldout_names: heldout.clone(),
            total_steps: RESCUE_CHECK_STEPS + 1,
            blur_total_steps: None,
            start_step: 0,
            init_ply: None,
            warm_start: false,
            init_wait_s: None,
            fork_ply: None,
            fork_min_splats: 0,
            fresh_init_ply: false,
            plain_init_ply: false,
            max_splats: RESCUE_CHECK_MAX_SPLATS,
            cap_max_explicit: false,
            max_budget: false,
            early_stop_db: None,
            growth_start_iter: None,
            growth_stop_iter: None,
            refine_every: None,
            max_resolution: RESCUE_CHECK_MAX_RESOLUTION,
            eval_every: RESCUE_CHECK_STEPS,
            checkpoint_every: RESCUE_CHECK_STEPS + 1,
            time_budget_s: None,
            sh_degree: 3,
            sh_degree_interval: None,
            ssim_weight: None,
            surfel: false,
            neural_max: false,
            config_overrides: None,
            seed: 42,
            gpu_lock: job.gpu_lock.clone(),
            defer_lpips: false,
            cloud: false,
            pseudo: None,
        };
        let job_path = dir.join("train_job.json");
        write_json(&job_path, &train_job)?;
        let log = std::fs::File::create(dir.join("train.log"))?;
        let status = std::process::Command::new(&exe)
            .args(["__brush", "train"])
            .arg(&job_path)
            .stdout(log.try_clone()?)
            .stderr(log)
            .status()
            .context("spawning the rescue check fit")?;
        anyhow::ensure!(status.success(), "{arm} check fit exited {status} (see {})", dir.join("train.log").display());
        let rows = std::fs::read_to_string(train_out.join("training.jsonl"))?;
        let p = rows
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter(|v| v["event"] == "eval")
            .last()
            .and_then(|v| v["psnr"].as_f64())
            .with_context(|| format!("{arm} check fit wrote no held-out eval"))?;
        println!("[sfm] rescue check: {arm} held-out PSNR {p:.2} dB on {} shared frames ({} trained)", heldout.len(), names.len() - heldout.len());
        psnr.insert(arm.into(), json!(p));
        if arm == "colmap" {
            *colmap_fit = Some((heldout.clone(), p));
        }
    }
    let (c, r) = (psnr["colmap"].as_f64().unwrap_or(0.0), psnr["rescue"].as_f64().unwrap_or(0.0));
    let keep = r >= c - margin_db;
    let _ = std::fs::remove_dir_all(out.join("colmap/dataset"));
    let _ = std::fs::remove_dir_all(out.join("rescue/dataset"));
    Ok(json!({"colmap_psnr": c, "rescue_psnr": r, "delta_db": r - c, "margin_db": margin_db, "n_heldout": heldout.len(),
        "n_shared": common.len(), "n_train": n_train, "steps": RESCUE_CHECK_STEPS, "keep": keep, "wall_s": start.elapsed().as_secs_f64()}))
}

/// SIFT placed under this share of the frames (or made no model): try learned features.
const ALIKED_BELOW_FRACTION: f64 = 0.9;
/// ALIKED keypoints per frame (LightGlue's cost grows with them).
const ALIKED_MAX_FEATURES: usize = 4096;
/// Above this many frames the fallback is skipped: on 600-frame walks it cost 4-15 min of SfM past the
/// budget and changed no outcome (guildhall 258 -> 352 of 600, still partial; copped-hall-first-floor
/// found no initial pair); on a 40-frame sparse walk it turned 15/40 into 40/40.
const ALIKED_MAX_FRAMES: usize = 400;

/// Learned-feature fallback: when SIFT's solve placed under 90% of the frames or made no model, extract
/// ALIKED-N16ROT features and match them with LightGlue (COLMAP's built-in ONNX path) on the same frames:
/// sequential with quadratic long-range pairs for video (SIFT's vocabulary tree can't retrieve ALIKED
/// descriptors), exhaustive for up to 150 photos. The model that places more frames is kept in
/// `colmap_dir`; the other goes to colmap.sift / colmap.aliked (logs only). A SIFT trajectory that flipped is
/// the flip retry's business (an ALIKED one drops its flipped frames), a preview never waits on it, and a spent SfM budget skips it.
/// SPLAT_SFM_ALIKED=0 turns it off. Records aliked_fallback.json next to colmap/.
fn aliked_fallback(
    job: &SfmJob,
    images: &std::path::Path,
    colmap_dir: &std::path::Path,
    opts: &splat_sfm::ReconstructOptions,
    rec: anyhow::Result<splat_sfm::Reconstruction>,
) -> anyhow::Result<splat_sfm::Reconstruction> {
    let n = list_images(images).map(|v| v.len()).unwrap_or(0);
    let sift_registered = rec.as_ref().map(|r| r.health.n_registered).unwrap_or(0);
    let weak = match &rec {
        Ok(r) => (r.health.n_registered as f64) < ALIKED_BELOW_FRACTION * n as f64,
        Err(e) => e.downcast_ref::<splat_sfm::ContinuityFailure>().is_none(),
    };
    let Some((detector, matcher)) = job.aliked.clone() else { return rec };
    if !weak || job.preview || n < 3 || n > ALIKED_MAX_FRAMES || std::env::var("SPLAT_SFM_ALIKED").is_ok_and(|v| v == "0")
        // Video only on request (SPLAT_SFM_ALIKED=1): with every frame mapped it connects (shengli 125/183,
        // the same as SIFT, +170 s) but no video case gained; the photo set it fixed went 15/40 -> 40/40.
        || (job.video && !std::env::var("SPLAT_SFM_ALIKED").is_ok_and(|v| v == "1"))
        || splat_sfm::deadline_passed(opts, "the learned-feature (ALIKED) fallback")
    {
        return rec;
    }
    println!("[sfm] SIFT placed {sift_registered}/{n} frames: trying ALIKED + LightGlue features");
    let sift_dir = colmap_dir.with_extension("sift");
    let aliked_dir = colmap_dir.with_extension("aliked");
    let _ = std::fs::remove_dir_all(&sift_dir);
    let _ = std::fs::remove_dir_all(&aliked_dir);
    if colmap_dir.exists() {
        std::fs::rename(colmap_dir, &sift_dir)?;
    }
    let mut a = opts.clone();
    a.features = splat_sfm::Features::AlikedLightGlue { detector, matcher };
    a.vocab_tree = None;
    a.reuse_database = None;
    a.max_features = opts.max_features.min(ALIKED_MAX_FEATURES);
    a.matching = match job.pairs {
        Some(Pairs::Sequential) => splat_sfm::Matching::Sequential,
        Some(Pairs::Exhaustive) => splat_sfm::Matching::Exhaustive,
        _ if job.video || n > 150 => splat_sfm::Matching::Sequential,
        _ => splat_sfm::Matching::Exhaustive,
    };
    a.sequential_overlap = 20;
    a.quadratic_overlap = true;
    // Every frame, no keyframe map: ALIKED keyframe maps (every 3rd frame) connected only 2-4 images
    // (kobe 2/156, canada-water 4, copped-hall 4 vs SIFT's 53-126) and found no initial pair; the
    // photo-set path (no keyframes) connected 40/40.
    a.keyframe_stride = 1;
    let start = std::time::Instant::now();
    // A trajectory that fails pose continuity keeps its majority without the flipped frames (as the SIFT
    // path does past its budget): guildhall's ALIKED solve placed 595/600 and failed only that check.
    let attempt = match splat_sfm::reconstruct(images, &a) {
        Err(e) if job.video && e.downcast_ref::<splat_sfm::ContinuityFailure>().is_some() => {
            splat_sfm::without_flipped(images, &a).map_err(|why| e.context(why))
        }
        other => other,
    };
    let seconds = start.elapsed().as_secs_f64();
    let aliked_registered = attempt.as_ref().map(|r| r.health.n_registered).unwrap_or(0);
    let keep = attempt.is_ok() && (rec.is_err() || aliked_registered > sift_registered);
    let mut info = json!({"sift_registered": sift_registered, "aliked_registered": aliked_registered, "n_input": n,
        "seconds": seconds, "kept": if keep { "aliked" } else { "sift" }});
    match &attempt {
        Ok(_) => println!("[sfm] ALIKED placed {aliked_registered}/{n} frames in {seconds:.0} s: keeping {}", info["kept"]),
        Err(e) => {
            eprintln!("[sfm] ALIKED fallback failed ({seconds:.0} s): {e:#}");
            info["error"] = json!(format!("{e:#}"));
        }
    }
    let _ = write_json(&job.dir.join("aliked_fallback.json"), &info);
    let prune = |d: &std::path::Path| {
        for entry in std::fs::read_dir(d).into_iter().flatten().flatten() {
            let p = entry.path();
            let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else if p.extension().is_some_and(|e| e == "db" || e == "ply") { std::fs::remove_file(&p) } else { Ok(()) };
        }
    };
    if keep {
        prune(&sift_dir);
        return attempt;
    }
    // SIFT stays: its model paths point into colmap_dir.
    if colmap_dir.exists() {
        std::fs::rename(colmap_dir, &aliked_dir)?;
        prune(&aliked_dir);
    }
    if sift_dir.exists() {
        std::fs::rename(&sift_dir, colmap_dir)?;
    }
    rec
}

fn learned_fallback(
    job: &SfmJob,
    images: &std::path::Path,
    colmap_dir: &std::path::Path,
    holdout_every: usize,
    rec: anyhow::Result<splat_sfm::Reconstruction>,
) -> anyhow::Result<(splat_sfm::Reconstruction, Option<serde_json::Value>)> {
    // Geometry failures are failed solves even at 100% registration. Never train the failed
    // geometry as the rescue's baseline or keep it when rescue is disabled/unavailable.
    let rec = rec.and_then(|r| { splat_sfm::ensure_geometry_health(&r.health)?; Ok(r) });
    let n_input = list_images(images).map(|v| v.len()).unwrap_or(0);
    let colmap_registered = match &rec {
        Ok(r) => r.health.n_registered,
        // A failed solve (no model, or a video trajectory that failed pose continuity after the flip
        // retry: groei timelapse) keeps its database for diagnosis; the rescue reuses it.
        Err(_) => 0,
    };
    // A preview solve that meets its own floor (40% in one model, [`too_few_registered`]) passes as
    // it is: the rescue would only delay the first view (pegasus quick SfM 11/24).
    let preview_ok = job.preview && rec.as_ref().is_ok_and(|r| !too_few_registered(r.health.n_registered as u64, r.health.n_registered as u64, n_input as u64, true));
    // Below half the frames the learned model may replace COLMAP's (the guard decides). Between half and
    // the 90% solve bar (up to RESCUE_BAND_MAX_FRAMES) it may only ADD the frames COLMAP left out
    // (splat_sfm::union_learned): one room clip placed 79/150 (53%), losing a stretch of blank
    // walls, and the rescue never ran; a replacement there is unsafe (paf-museum -1.9 dB).
    let below_half = colmap_registered * 2 < n_input;
    let band = !below_half && rec.is_ok() && n_input <= RESCUE_BAND_MAX_FRAMES
        && (colmap_registered as f64) < RESCUE_BELOW_FRACTION * n_input as f64
        // On by default behind [`rescue_check`]: a union that costs more than RESCUE_CHECK_UNION_MARGIN_DB on
        // the frames COLMAP placed is rejected (unchecked unions lost up to 1.8 dB on shared views);
        // shengli-night-market's 198/206 vs COLMAP's 148 held -0.18 dB. SPLAT_SFM_RESCUE_UNION=0 turns it off.
        && !std::env::var("SPLAT_SFM_RESCUE_UNION").is_ok_and(|v| v == "0");
    // SPLAT_SFM_RESCUE=0 turns every learned rescue off (paired no-rescue arms).
    let off = std::env::var("SPLAT_SFM_RESCUE").is_ok_and(|v| v == "0");
    if off || n_input < 3 || !(below_half || band) || preview_ok || !colmap_dir.join("database.db").is_file() {
        return rec.map(|r| (r, None));
    }
    let bad_geometry = rec.as_ref().err().is_some_and(|e| e.downcast_ref::<splat_sfm::GeometryFailure>().is_some());
    if below_half && (rec.is_ok() || bad_geometry) && n_input > RESCUE_REPLACE_MAX_FRAMES {
        println!("[sfm] COLMAP placed {colmap_registered}/{n_input} frames: no learned-pose replacement above {RESCUE_REPLACE_MAX_FRAMES} frames");
        return rec.map(|r| (r, Some(json!({"triggered": false, "reason": format!("over {RESCUE_REPLACE_MAX_FRAMES} frames")}))));
    }
    let Some(weights) = job.da3_weights.as_deref() else {
        // Loud, not a warning: the solve is about to be rejected and the rescue that would have
        // run is missing only because no weights reached this machine (a cloud/fal job without
        // the file is the usual cause).
        let msg = format!(
            "COLMAP placed {colmap_registered}/{n_input} frames and the learned-pose rescue cannot run: DA3 weights \
             not installed (run `vauss fetch da3-base` or set SPLAT_DA3_WEIGHTS to da3-base-cam-f16.safetensors)"
        );
        eprintln!("[sfm] ERROR: {msg}");
        return match rec {
            Ok(r) => Ok((r, Some(json!({"triggered": false, "reason": "DA3 weights not installed", "error": msg})))),
            Err(e) => Err(e.context(msg)),
        };
    };
    let start = std::time::Instant::now();
    let out = job.dir.join("learned");
    let _ = std::fs::remove_dir_all(&out);
    match &rec {
        Ok(_) => println!("[sfm] COLMAP placed {colmap_registered}/{n_input} frames: trying the learned-pose rescue"),
        Err(e) => println!("[sfm] COLMAP failed ({e:#}): trying the learned-pose rescue"),
    }
    let mut info = json!({"triggered": true, "colmap_registered": colmap_registered, "colmap_failed": rec.is_err(), "n_input": n_input, "kept": "colmap", "candidates": {}});
    // Below half a union was tried next to the replacement (g3-room-sfm gate, 2026-10-11): on rooms its
    // added DA3 frames sat a median 5.6 deg off COLMAP's rotations (pose_agreement), and every union
    // lost the check by 0.56-1.8 dB, so below half stays replace-or-nothing.
    type Found = (PathBuf, splat_eval::SfmHealth, splat_sfm::ComponentInventory);
    let mut candidates: Vec<(&'static str, Found)> = Vec::new();
    let attempt = (|| -> anyhow::Result<()> {
        std::fs::create_dir_all(&out)?;
        // The seed writes SIMPLE_RADIAL cameras; photo databases hold PINHOLE ones.
        let db = out.join("seed-input.db");
        std::fs::copy(splat_sfm::frozen_database(colmap_dir), &db)?;
        splat_sfm::reset_camera(&db, "SIMPLE_RADIAL")?;
        let seed = splat_sfm_learned::learned_seed(images, &db, &job.colmap, job.gpu_lock.as_deref(), Some(weights), &out.join("seed"))?;
        info["inference_s"] = json!(seed.inference_seconds);
        let model = if seed.model_dir.join("0").is_dir() { seed.model_dir.join("0") } else { seed.model_dir.clone() };
        if let Ok(colmap) = &rec {
            // Do the learned poses agree with COLMAP's where both placed a frame? (Diagnostic only.)
            info["pose_agreement"] = splat_sfm::pose_agreement(&job.colmap, &colmap.model_dir, &model, &out.join("agreement"))
                .unwrap_or_else(|e| json!({"error": format!("{e:#}")}));
        }
        if band {
            let colmap_model = rec.as_ref().map_err(|e| anyhow::anyhow!("{e:#}"))?.model_dir.clone();
            let mut opts = splat_sfm::ReconstructOptions::new(colmap_dir);
            opts.colmap = job.colmap.clone();
            opts.video = job.video;
            opts.holdout_every = holdout_every;
            match splat_sfm::union_learned(images, &opts, &colmap_model, &model, &out.join("union"), RESCUE_UNION_MIN_OBS) {
                Ok(Some(found)) => {
                    splat_sfm::ensure_video_continuity(&found.1, job.video)?;
                    info["candidates"]["union"] = json!({"registered": found.1.n_registered});
                    candidates.push(("union", found));
                }
                Ok(None) => info["candidates"]["union"] = json!({"registered": colmap_registered, "added": 0}),
                Err(e) => return Err(e),
            }
            return Ok(());
        }
        let frames: Vec<String> = list_images(images)?
            .iter()
            .filter_map(|p| p.strip_prefix(images).ok().map(|n| n.to_string_lossy().into_owned()))
            .collect();
        let guard = splat_sfm_learned::guard_fallback(None, &model, &frames, None, &out);
        info["guard"] = json!(guard.reason);
        if guard.kept != splat_sfm_learned::Kept::Candidate {
            return Ok(());
        }
        let sparse = out.join("sparse");
        std::fs::create_dir_all(&sparse)?;
        std::fs::rename(&model, sparse.join("0"))?;
        let (dir, health) = splat_sfm::finalize_model(&sparse, images, job.video, holdout_every)?;
        splat_sfm::ensure_video_continuity(&health, job.video)?;
        info["learned_registered"] = json!(health.n_registered);
        info["candidates"]["replace"] = json!({"registered": health.n_registered});
        if health.n_registered * 2 < n_input {
            return Ok(());
        }
        candidates.push(("replace", (dir, health, splat_sfm::component_inventory(&sparse)?)));
        Ok(())
    })();
    // Any readable COLMAP model is a baseline, trainable or not (paf-museum: 86/244, the replacement
    // 244/244 rendered 12.7 dB): the rescue must hold up on the frames both placed. Only no model at
    // all, or too few shared frames for the check, keeps the rescue unchecked.
    let attempt = attempt.map(|()| match &rec {
        Ok(colmap) if !job.preview => {
            let mut kept: Option<(&'static str, Found, f64)> = None;
            let mut colmap_fit = None;
            let tried = !candidates.is_empty();
            for (mode, found) in candidates {
                let margin = if mode == "union" { RESCUE_CHECK_UNION_MARGIN_DB } else { RESCUE_CHECK_MARGIN_DB };
                match rescue_check(job, &colmap.model_dir, &found.0, &out.join(format!("check-{mode}")), margin, &mut colmap_fit) {
                    Ok(check) => {
                        let score = check["rescue_psnr"].as_f64().unwrap_or(f64::NEG_INFINITY);
                        let keep = check["keep"] == true;
                        info["candidates"][mode]["check"] = check;
                        if keep && kept.as_ref().is_none_or(|k| score > k.2) {
                            kept = Some((mode, found, score));
                        }
                    }
                    // An unusable check (too few shared frames, a failed fit) keeps the rescue, as before
                    // the check, unless another candidate passed its check.
                    Err(e) => {
                        eprintln!("[sfm] rescue check ({mode}) skipped: {e:#}");
                        info["candidates"][mode]["check"] = json!({"error": format!("{e:#}")});
                        if kept.is_none() {
                            kept = Some((mode, found, f64::NEG_INFINITY));
                        }
                    }
                }
            }
            if kept.is_none() && tried {
                info["guard"] = json!("held-out check: COLMAP's model scored better on the shared frames");
            }
            kept.map(|(mode, found, _)| {
                info["mode"] = json!(mode);
                info["check"] = info["candidates"][mode]["check"].clone();
                found
            })
        }
        _ => candidates.into_iter().next().map(|(mode, found)| {
            info["mode"] = json!(mode);
            found
        }),
    });
    info["wall_s"] = json!(start.elapsed().as_secs_f64());
    match (attempt, rec) {
        (Ok(Some((model_dir, health, components))), rec) => {
            info["kept"] = json!("learned");
            // The frames posed by DA3 rather than COLMAP (all of them for a replacement), one name per
            // line next to sparse/: pose refinement keys on it (a room clip's union cost held-out PSNR).
            let colmap_names: std::collections::BTreeSet<String> =
                rec.as_ref().ok().filter(|_| info["mode"] == "union").and_then(|r| model::names(&r.model_dir).ok()).unwrap_or_default().into_iter().collect();
            let added: Vec<String> = model::names(&model_dir).unwrap_or_default().into_iter().filter(|n| !colmap_names.contains(n)).collect();
            info["n_learned_poses"] = json!(added.len());
            if let Err(e) = std::fs::write(job.dir.join("learned_poses.txt"), added.iter().map(|n| format!("{n}\n")).collect::<String>()) {
                eprintln!("[sfm] learned_poses.txt: {e}");
            }
            println!("[sfm] learned-pose rescue kept: {}/{n_input} frames placed", health.n_registered);
            let rung = format!("learned-{}", info["mode"].as_str().unwrap_or("replace"));
            let rec = match rec {
                Ok(rec) => splat_sfm::Reconstruction { model_dir, health, components, rung, ..rec },
                Err(_) => splat_sfm::Reconstruction {
                    model_dir,
                    health,
                    components,
                    wall_seconds: start.elapsed().as_secs_f64(),
                    lock_wait_seconds: 0.0,
                    stages: Vec::new(),
                    colmap_version: String::new(),
                    environment: Default::default(),
                    rung,
                },
            };
            Ok((rec, Some(info)))
        }
        (attempt, rec) => {
            match attempt {
                Ok(_) => println!("[sfm] learned-pose rescue not kept: {}", info["guard"]),
                Err(e) => {
                    eprintln!("[sfm] learned-pose rescue failed: {e:#}");
                    info["error"] = json!(format!("{e:#}"));
                }
            }
            let _ = std::fs::remove_dir_all(&out);
            rec.map(|r| (r, Some(info)))
        }
    }
}

/// COLMAP looks masks up as `<mask_path>/<image name>.png` (frame_0001.jpg -> frame_0001.jpg.png);
/// mask artifacts name them by stem (frame_0001.png), frames/masks already uses COLMAP's names.
/// Link under COLMAP's names into `dst`,
/// which lives outside the dataset's images/ and masks/ dirs. Masks are drawn on the training
/// images; for larger feature images (sfm_images) they are resized to match (nearest neighbour).
fn colmap_masks(src: &std::path::Path, images: &std::path::Path, feature_images: &std::path::Path, dst: &std::path::Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dst)?;
    let mut linked = 0;
    for image in list_images(images)? {
        let name = image.strip_prefix(images)?;
        let colmap_name = src.join(format!("{}.png", name.to_string_lossy()));
        let mask = if colmap_name.is_file() { colmap_name } else { src.join(name).with_extension("png") };
        if !mask.is_file() {
            continue;
        }
        let target = dst.join(format!("{}.png", name.to_string_lossy()));
        if let Some(p) = target.parent() {
            std::fs::create_dir_all(p)?;
        }
        let size = (feature_images != images).then(|| image::image_dimensions(feature_images.join(name))).transpose()?;
        match size {
            Some((w, h)) if image::image_dimensions(&mask)? != (w, h) => {
                image::imageops::resize(&image::open(&mask)?.to_luma8(), w, h, image::imageops::FilterType::Nearest).save(&target)?;
            }
            _ => link_or_copy(&mask, &target)?,
        }
        linked += 1;
    }
    println!("[sfm] feature masks: {linked} images");
    Ok(dst.to_path_buf())
}

/// Inline SfM masks, per prompt label ("sky; person" runs "sky" and "person" separately, so each
/// frame's mask is judged by what fired). Indoors both labels fire falsely on rooms: blank walls,
/// carpets and clothes piles were masked to 29-49% of frame area on average (one cream wall 100%),
/// dropping 18-60% of verified matches on six ARKitScenes rooms (10% on the room that passed), and
/// the masked links broke the frame chain (rooms-sfm-triage.md).
/// A sky mask is kept only when it is anchored at the top of the frame and stays out of its bottom
/// band (frames are upright by now): a wall or carpet is not sky.
const SKY_TOP_ROWS: f64 = 0.15;
const SKY_MIN_TOP_COVER: f64 = 0.3;
const SKY_BOTTOM_ROWS: f64 = 1.0 / 3.0;
const SKY_MAX_BOTTOM_COVER: f64 = 0.2;
/// A person mask covering more than this share of the frame is a false fire on a static scene (a
/// clothes pile, a sofa): close-up people on the On-the-go clips peak at 0.49 of the frame
/// (research/mask-noperson), the rooms' false masks sit at 0.5-1.0.
const PERSON_MAX_SHARE: f64 = 0.5;

/// Whether one label's mask (0 = masked) is kept on a frame: Err names why it is dropped.
fn judge_mask(label: &str, mask: &image::GrayImage) -> Result<(), &'static str> {
    let (w, h) = mask.dimensions();
    let cover = |y0: u32, y1: u32| {
        let rows = &mask.as_raw()[(y0 * w) as usize..(y1 * w) as usize];
        rows.iter().filter(|v| **v < 128).count() as f64 / rows.len().max(1) as f64
    };
    match label {
        "sky" => {
            let top = ((h as f64 * SKY_TOP_ROWS).ceil() as u32).clamp(1, h);
            let bottom = ((h as f64 * SKY_BOTTOM_ROWS).ceil() as u32).clamp(1, h);
            if cover(0, top) < SKY_MIN_TOP_COVER {
                Err("sky not at the top")
            } else if cover(h - bottom, h) > SKY_MAX_BOTTOM_COVER {
                Err("sky reaches the bottom")
            } else {
                Ok(())
            }
        }
        "person" if cover(0, h) > PERSON_MAX_SHARE => Err("person mask too large"),
        _ => Ok(()),
    }
}

/// Segment on the SfM runner, before COLMAP, without another remote job or round trip.
/// The two-pass detector guard matches standalone masking, per label; masks only suppress features.
/// Writes sfm_masks.json: per label and frame, what fired, its share of the frame and the verdict.
fn inline_masks(
    job: &SfmJob,
    sam: &crate::select::SamTool,
    prompt: &str,
    lock: bool,
) -> anyhow::Result<PathBuf> {
    use anyhow::{Context, ensure};
    ensure!(
        job.sam_vocab.as_ref().is_some_and(|p| p.is_file()),
        "missing detector vocabulary"
    );
    let _lock = crate::gpu_lock(job.gpu_lock.as_deref().filter(|_| lock))?;
    let images = job.dir.join("images");
    let raw = job.dir.join("sfm_raw_masks");
    let presence = job.dir.join("sfm_presence_masks");
    for dir in [&raw, &presence] {
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
    }
    let labels: Vec<String> = prompt.split(';').map(|l| l.trim().to_lowercase()).filter(|l| !l.is_empty()).collect();
    ensure!(!labels.is_empty(), "empty mask prompt");
    let slug = |label: &str| label.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect::<String>();
    // Keep the existing two-pass presence guard: a 0.4 pass rejects absent distractors, while
    // 0.3 retains the full masks of people the detector did find (including close-ups).
    // Independent subprocesses overlap model loading and detection on the same worker.
    let segment = |label: &str, threshold: &str, out: &std::path::Path| -> anyhow::Result<Vec<u8>> {
        if out.exists() {
            std::fs::remove_dir_all(out)?;
        }
        std::fs::create_dir_all(out.parent().context("mask dir parent")?)?;
        let output = std::process::Command::new(&sam.binary)
            .env("OMP_NUM_THREADS", "8")
            .env("SS_LANG", "en")
            .args(["sam", "track", "--model"])
            .arg(&sam.model)
            .arg("--detector")
            .arg(&sam.detector)
            .arg("--frames")
            .arg(&images)
            .args([
                "--text",
                label,
                "--remove-prompted",
                "--detector-threshold",
                threshold,
                "--out",
            ])
            .arg(out)
            .args(["--device", "0"])
            .output()
            .context("starting SfM segmenter")?;
        let log = [&output.stdout[..], &output.stderr[..]].concat();
        std::fs::write(job.dir.join(format!("sfm_masks_{}_{threshold}.log", slug(label))), &log)?;
        ensure!(
            output.status.success(),
            "SfM segmenter ({label}) exited {}",
            output.status
        );
        Ok(log)
    };
    let logs = std::thread::scope(|scope| -> anyhow::Result<Vec<Vec<u8>>> {
        let passes: Vec<_> = labels
            .iter()
            .map(|label| {
                let (raw, presence) = (raw.join(slug(label)), presence.join(slug(label)));
                let segment = &segment;
                let presence_pass = scope.spawn(move || segment(label, "0.4", &presence));
                let mask_pass = scope.spawn(move || segment(label, "0.3", &raw));
                (presence_pass, mask_pass)
            })
            .collect();
        // Always wait for every child, including on failure, before using or removing masks.
        let joined: Vec<_> = passes
            .into_iter()
            .map(|(p, m)| (p.join().map_err(|_| anyhow::anyhow!("SfM presence worker panicked")), m.join().map_err(|_| anyhow::anyhow!("SfM mask worker panicked"))))
            .collect();
        joined.into_iter().map(|(p, m)| { p??; m? }).collect()
    })?;
    let source = list_images(&images)?;
    let mappings: Vec<_> = logs.iter().map(|log| crate::select_worker::kept_frames(&String::from_utf8_lossy(log))).collect();
    for mapping in &mappings {
        ensure!(
            mapping.len() == source.len(),
            "segmenter did not identify all frames"
        );
    }
    let masks = job.dir.join("sfm_masks");
    std::fs::create_dir_all(&masks)?;
    // Per frame: read every label's two passes, judge them, merge and write the feature mask. Frames
    // are independent, so they are spread over the cores (the mapper is waiting on this); results
    // are gathered in name order, so sfm_masks.json and every PNG are the same as a serial loop's.
    let merge = |src: &PathBuf| -> anyhow::Result<(serde_json::Value, f64, Vec<&'static str>)> {
        let name = src.strip_prefix(&images)?;
        let dims = image::image_dimensions(src)?;
        let mut mask = image::GrayImage::from_pixel(dims.0, dims.1, image::Luma([255]));
        let mut record = serde_json::Map::new();
        let mut verdicts = Vec::new();
        for (label, mapping) in labels.iter().zip(&mappings) {
            let index = mapping
                .iter()
                .find(|(_, n)| std::path::Path::new(n) == name)
                .map(|(i, _)| i)
                .context("segmenter omitted an input image")?;
            let label_mask = image::open(raw.join(slug(label)).join(format!("frame_{index:05}.png")))?.to_luma8();
            ensure!(label_mask.dimensions() == dims, "mask dimensions differ");
            let detected = image::open(presence.join(slug(label)).join(format!("frame_{index:05}.png")))?.to_luma8();
            let share = label_mask.as_raw().iter().filter(|v| **v < 128).count() as f64 / label_mask.as_raw().len().max(1) as f64;
            let verdict = if share == 0.0 || !detected.as_raw().iter().any(|v| *v < 128) {
                "absent"
            } else {
                match judge_mask(label, &label_mask) {
                    Ok(()) => {
                        for (pixel, m) in mask.as_mut().iter_mut().zip(label_mask.as_raw()) {
                            *pixel = (*pixel).min(*m);
                        }
                        "kept"
                    }
                    Err(why) => why,
                }
            };
            verdicts.push(verdict);
            record.insert(label.clone(), json!({"share": (share * 1000.0).round() / 1000.0, "verdict": verdict}));
        }
        let masked = mask.as_raw().iter().filter(|v| **v < 128).count() as f64 / mask.as_raw().len().max(1) as f64;
        let frame = json!({"name": name.to_string_lossy(), "masked_share": (masked * 1000.0).round() / 1000.0, "labels": record});
        // Preserve any alpha/explicit feature masks as well as the new sky mask.
        if let Some(existing) = &job.mask_dir {
            let colmap = existing.join(format!("{}.png", name.to_string_lossy()));
            let prior = if colmap.is_file() {
                colmap
            } else {
                existing.join(name).with_extension("png")
            };
            if prior.is_file() {
                let prior = image::open(prior)?.to_luma8();
                ensure!(
                    mask.dimensions() == prior.dimensions(),
                    "prior mask dimensions differ"
                );
                for (pixel, prior) in mask.as_mut().iter_mut().zip(prior.as_raw()) {
                    *pixel = (*pixel).min(*prior);
                }
            }
        }
        let target = masks.join(format!("{}.png", name.to_string_lossy()));
        std::fs::create_dir_all(target.parent().unwrap())?;
        mask.save(target)?;
        Ok((frame, masked, verdicts))
    };
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 8);
    let chunk = source.len().div_ceil(workers).max(1);
    let merged = std::thread::scope(|scope| {
        let merge = &merge;
        let handles: Vec<_> = source.chunks(chunk).map(|part| scope.spawn(move || part.iter().map(merge).collect::<Vec<_>>())).collect();
        handles.into_iter().map(|h| h.join().map_err(|_| anyhow::anyhow!("SfM mask merge worker panicked"))).collect::<anyhow::Result<Vec<_>>>()
    })?;
    let mut frames = Vec::new();
    let mut tally: std::collections::BTreeMap<String, std::collections::BTreeMap<String, usize>> = Default::default();
    let mut masked_shares = Vec::new();
    for result in merged.into_iter().flatten() {
        let (frame, masked, verdicts) = result?;
        for (label, verdict) in labels.iter().zip(verdicts) {
            *tally.entry(label.clone()).or_default().entry(verdict.into()).or_default() += 1;
        }
        masked_shares.push(masked);
        frames.push(frame);
    }
    let n = masked_shares.len().max(1) as f64;
    let summary = json!({
        "prompt": prompt,
        "labels": tally,
        "frames": masked_shares.len(),
        "frames_masked": masked_shares.iter().filter(|s| **s > 0.0).count(),
        "mean_masked_share": masked_shares.iter().sum::<f64>() / n,
        "frames_over_half": masked_shares.iter().filter(|s| **s > 0.5).count(),
    });
    for (label, counts) in &tally {
        println!("[sfm] feature masks {label}: {counts:?}");
    }
    println!("[sfm] feature masks: {} of {} frames masked, mean masked share {:.3}", summary["frames_masked"], summary["frames"], summary["mean_masked_share"].as_f64().unwrap_or(0.0));
    let mut out = summary.clone();
    out["per_frame"] = json!(frames);
    write_json(&job.dir.join("sfm_masks.json"), &out)?;
    Ok(masks)
}

/// Worker body: splat-sfm (COLMAP) into <dir>/colmap, then the chosen model becomes the
/// dataset's sparse/0 (Brush and the split read only that).
pub fn run_job(job: SfmJob) -> anyhow::Result<i32> {
    let start = std::time::Instant::now();
    let code = solve_job(job.clone(), start);
    // A chained preview never changes the solve's outcome: on failure the client trains it as a
    // separate job.
    if let (Ok(0), Some(chain)) = (&code, &job.preview_train) {
        let rec: serde_json::Value = read_json(&job.dir.join("reconstruction.json")).unwrap_or_default();
        let (registered, input) = (rec["health"]["n_registered"].as_u64().unwrap_or(0), rec["health"]["n_input"].as_u64().unwrap_or(0));
        let largest = rec["component_sizes"].as_array().and_then(|s| s.first()).and_then(|v| v.as_u64()).unwrap_or(0);
        let chained = if too_few_registered(registered, largest, input, true) {
            Err(anyhow::anyhow!("the solve placed {registered} of {input} frames, below the preview floor"))
        } else {
            crate::train::run_chained(&job.dir, chain)
        };
        if let Err(error) = chained {
            eprintln!("[sfm] chained preview skipped: {error:#}");
        }
    }
    code
}

fn solve_job(job: SfmJob, start: std::time::Instant) -> anyhow::Result<i32> {
    match run_sfm_job(job.clone()) {
        Err(error) if job.capture => {
            // A solve failure is recoverable only for calibrated inputs. Return
            // a completed attempt, so the client can select the sensor poses;
            // core correctly treats a nonzero worker exit as unrecoverable.
            let n_input = list_images(&job.dir.join("images"))?.len();
            write_json(&job.dir.join("reconstruction.json"), &json!({
                "health":{"n_input":n_input,"n_registered":0},
                "capture_sfm_failed":true,"sfm_error":format!("{error:#}"),
                "wall_seconds":start.elapsed().as_secs_f64(),
            }))?;
            eprintln!("[sfm] capture solve failed: {error:#}; selecting calibrated pose fallback");
            Ok(0)
        }
        other => other,
    }
}

/// The full video SfM's retry after a flipped trajectory: one radial term and an 8 deg initial pair, on
/// the incremental mapper even when the first solve was global (the initial pair is an incremental option).
fn flip_retry(opts: &mut splat_sfm::ReconstructOptions) {
    opts.mapper = splat_sfm::Mapper::Incremental;
    if opts.camera_model == "RADIAL" { opts.camera_model = "SIMPLE_RADIAL".into(); }
    opts.init_min_tri_angle = 8.0;
}

fn run_sfm_job(mut job: SfmJob) -> anyhow::Result<i32> {
    // Video: segment while COLMAP extracts unmasked features and matches them; splat-sfm waits
    // for the masks before mapping and drops the verified matches on masked keypoints. Driveway clip: the
    // segmenter's 31-38 s (per-frame SAM, not detection) overlap matching and verification. It
    // shares the GPU with them instead of taking the GPU lock.
    let late_masks = job.dir.join("sfm_masks");
    let late_done = late_masks.with_extension("done");
    let _ = std::fs::remove_file(&late_done);
    let mut masking = None;
    if let (Some(prompt), Some(sam)) = (&job.mask_prompt, &job.sam) {
        if job.video {
            let (job, sam, prompt, done) = (job.clone(), sam.clone(), prompt.clone(), late_done.clone());
            masking = Some(std::thread::spawn(move || {
                let start = std::time::Instant::now();
                let status = match inline_masks(&job, &sam, &prompt, false) {
                    Ok(_) => "ok",
                    Err(error) => {
                        eprintln!("[sfm] skipping optional masks: {error:#}");
                        "failed"
                    }
                };
                println!("[sfm] inline masks: {:.2}s (alongside matching)", start.elapsed().as_secs_f64());
                std::fs::write(&done, status)
            }));
        } else {
            let start = std::time::Instant::now();
            // Best effort, like the fixed driver's former standalone masking stage. Failure leaves
            // the original feature masks intact and never blocks a usable reconstruction.
            match inline_masks(&job, sam, prompt, true) {
                Ok(masks) => job.mask_dir = Some(masks),
                Err(error) => eprintln!("[sfm] skipping optional masks: {error:#}"),
            }
            println!("[sfm] inline masks: {:.2}s", start.elapsed().as_secs_f64());
        }
    }
    let colmap_dir = job.dir.join("colmap");
    let flipped_dir = job.dir.join("colmap-flipped");
    let _ = std::fs::remove_dir_all(&colmap_dir);
    let _ = std::fs::remove_dir_all(&flipped_dir);
    let _ = std::fs::remove_dir_all(job.dir.join("sparse"));
    let mut opts = splat_sfm::ReconstructOptions::new(&colmap_dir);
    opts.deadline = splat_sfm::budget_deadline();
    if let Some(seed) = job.mapper_seed {
        opts.mapper_seed = seed;
    }
    opts.colmap = job.colmap.clone();
    opts.gpu_lock = job.gpu_lock.clone();
    // Photos: incremental beat global on our data (truck-63: 13 s vs 23 s; pegasus-150: 46 s vs 79 s)
    // while the available COLMAP builds lack GPU bundle adjustment.
    // Full video SfM: COLMAP's global mapper (rotation averaging + global positioning) on every frame.
    // Rotation averaging keeps long handheld walks with many rotation-only pairs consistent, where
    // incremental solves can place whole stretches at wrong rotations. The quick video SfM (24
    // frames) stays incremental, and so do the recovery re-solves (low-angle, flip and
    // fixed-focal), whose options are incremental-only.
    // SPLAT_SFM_MAPPER=global|incremental overrides the first solve.
    opts.use_gpu = job.use_gpu;
    opts.mapper = if job.video && !job.quick_video { splat_sfm::Mapper::Global } else { splat_sfm::Mapper::Incremental };
    match std::env::var("SPLAT_SFM_MAPPER").as_deref().map(str::trim) {
        Ok("global") => opts.mapper = splat_sfm::Mapper::Global,
        Ok("incremental") => opts.mapper = splat_sfm::Mapper::Incremental,
        _ => {}
    }
    opts.matching = matching_policy(&job);
    opts.sequential_overlap = if job.video { 40 } else { 20 };
    // Video: sequential matching with vocab-tree loop detection, exhaustive fill-in when it
    // registers under 95%. Mapper seeds 1-3 (threads 1): 0/9 collapses on gimbal/night-stroll/rooftop;
    // rooftop +0.53 dB, matching 678 -> ~120 s; the others within seed noise.
    // Explicit sequential pairs on video keep the loop detection (no exhaustive fill-in), which long
    // walks need to close their loops.
    if !matches!(opts.matching, splat_sfm::Matching::Exhaustive) {
        opts.vocab_tree = job.vocab_tree.clone();
    }
    // Global BA at 1.3x model growth instead of COLMAP's 1.1x. Garden (185 photos, one exhaustive
    // db): 38 -> 18 global BAs, mapper 194 -> 161 s, same registration and reprojection. Video
    // (gimbal/night-stroll/kindel/moskvitch, mapper seeds 1-3): registration equal or higher on
    // every seed, mapper -8 to -43%.
    opts.ba_global_ratio = opts.ba_global_ratio.or(Some(1.3));
    // Two-view RANSAC stops early instead of at 10000 trials. Photos 500: most exhaustive photo
    // pairs don't overlap; in room, 23k of 41k RANSAC pairs had ~27 spurious matches and 0 inliers,
    // and each ran to the cap. Video 2000: rooftop extraction + matching 121 -> 56 s, verified
    // pairs -0.8%, inlier matches unchanged.
    opts.ransac_max_trials = if job.video { 2000 } else { 500 };
    if !job.video {
        // One local BA refinement pass instead of two: kitchen (279 photos) mapper -15% (2 seeds),
        // same registration, poses within 0.007 deg.
        opts.ba_local_refinements = opts.ba_local_refinements.or(Some(1));
        // One mapper thread up to 200 photos: COLMAP's multithreaded initial-pair search races, so
        // one room database picked a different pair run to run, and #19/#66 failed the gates (2 of
        // 13 bench runs, ~145 s SfM with the serial retry). One thread is deterministic. Mapper on
        // a Xeon 6962P, 8 threads -> 1: room (78) 18.2 -> 18.6 s, bicycle (194) 55.6 -> 47.5 s,
        // kitchen (279) 411 -> 485 s; larger sets keep the threads and the gate-failure retry.
        if list_images(&job.dir.join("images"))?.len() <= 200 {
            opts.mapper_threads = opts.mapper_threads.or(Some(1));
        }
    }
    // Video: the mapper keeps lens distortion fixed and one final bundle adjustment refines focal
    // length and distortion. Goresbridge (149 frames, one database): refining distortion while
    // mapping drifted the shared camera (focal 884 -> 1093-1431 px, k1 0.3-0.6) and split or threw
    // away 120+-frame models; with it fixed, one 148/149 model in 496 s instead of 114+49 in 953 s.
    opts.lens_last = job.video;
    // The quick video SfM (24 low-res frames) can't pin the focal length: night drifted to
    // ~100 degrees (406 px at 960 px) while its full SfM settled near 65, and the two models
    // then disagreed. Start it at a typical phone wide lens (70 degrees horizontal, frames carry
    // no EXIF focal), keep the focal fixed while mapping and refine it once at the end. The full
    // SfM is unchanged: fixing it moved the driveway clip's calibration 770 -> 880 px.
    if job.quick_video {
        opts.default_focal_length_factor = Some(0.5 / 35f64.to_radians().tan());
        opts.focal_last = true;
        // The 12-keyframe quick map keeps the mapper's threads: one thread (the full SfM's
        // deterministic keyframe mapper) took the driveway clip's quick SfM from 4.9 to 13.5 s on 18c1ce3.
        opts.mapper_threads = opts.mapper_threads.or(Some(opts.num_threads));
    }
    // Video maps every 3rd frame and the lens bundle adjustment refines that map; the other frames
    // are then registered against it. Global BA dominates the mapper (85-90% on goresbridge/driveway)
    // and grows faster than linearly with frames: driveway (150 frames) spent 307 of 480 s in the
    // mapper; every 3rd frame: mapper 62 -> 15 s, 150/150, 7k held-out PSNR +0.17 dB over every 2nd.
    // The quick SfM (24 frames) keeps every 2nd.
    if job.video {
        opts.keyframe_stride = if job.preserve_all { 1 } else if job.quick_video { 2 } else { 3 };
    }
    if job.video {
        // Global BA: 20 solver iterations and 2 refinement passes (COLMAP 50 / 5), as the final
        // triangulation already uses. Driveway clip (145 frames, one database): mapper 961 -> 464 s,
        // same 145/145 registered.
        opts.ba_global_max_iterations = Some(20);
        opts.ba_global_refinements = Some(2);
        // Video frames see mostly the same points; global BA skips the redundant ones. Driveway
        // clip (75 keyframes): mapper global BAs ~74 -> ~58 s, same registration.
        opts.ba_global_skip_redundant = true;
    }
    opts.video = job.video;
    opts.camera_model = job.camera_model.clone();
    opts.late_masks = masking.is_some().then(|| late_masks.clone());
    opts.max_image_size = job.max_image_size;
    opts.max_features = job.max_features;
    // Dark frames: extract features from their lifted analysis/ copies (same names, same size),
    // via an overlay of images/; the dataset keeps the originals for training. Photos larger than the
    // training frames: features and mapping on their larger copies (sfm_images), and the model is
    // rescaled to the training size below.
    let images = job.dir.join("images");
    let overlay = job.dir.join("feature_images");
    let _ = std::fs::remove_dir_all(&overlay);
    let feature_images = match (&job.analysis_dir, &job.sfm_images) {
        (None, Some(sfm)) => {
            println!("[sfm] feature images: {}", sfm.display());
            sfm.clone()
        }
        (Some(analysis), _) => {
            let mut lifted = 0;
            for src in list_images(&images)? {
                let name = src.strip_prefix(&images)?;
                let dst = overlay.join(name);
                if let Some(p) = dst.parent() {
                    std::fs::create_dir_all(p)?;
                }
                let alt = analysis.join(name);
                lifted += usize::from(alt.is_file());
                link_or_copy(if alt.is_file() { &alt } else { &src }, &dst)?;
            }
            println!("[sfm] feature images: {lifted} exposure-lifted");
            overlay.clone()
        }
        (None, None) => images.clone(),
    };
    // splat-sfm requires a fresh output dir, so COLMAP-named feature masks live beside it.
    let feature_masks = job.dir.join("feature_masks");
    let _ = std::fs::remove_dir_all(&feature_masks);
    opts.mask_dir = job.mask_dir.as_deref().map(|m| colmap_masks(m, &images, &feature_images, &feature_masks)).transpose()?;
    let rec = match splat_sfm::reconstruct(&feature_images, &opts) {
        // A flipped video trajectory fails the job. Retry once with one radial term and an
        // 8 deg initial pair: yagan flips in 2 of 3 OPENCV/16 runs, and this registered 150/150
        // and rendered +0.94 dB. Kept out of the default: -0.1 to -0.2 dB on unflipped videos.
        // Out of budget: keep this trajectory without its flipped frames instead of remapping.
        Err(e) if job.video && !job.quick_video && e.downcast_ref::<splat_sfm::ContinuityFailure>().is_some()
            && splat_sfm::deadline_passed(&opts, "the flip retry") =>
        {
            splat_sfm::without_flipped(&feature_images, &opts).map_err(|why| e.context(why))
        }
        Err(e) if job.video && e.downcast_ref::<splat_sfm::ContinuityFailure>().is_some() => {
            println!("[sfm] {e:#}; retrying with an 8 deg initial pair and guarded calibration");
            std::fs::rename(&colmap_dir, &flipped_dir)?;
            flip_retry(&mut opts);
            // Features and matches don't depend on the camera model: reuse them (half the solve).
            opts.reuse_database = Some(splat_sfm::frozen_database(&flipped_dir));
            match splat_sfm::reconstruct(&feature_images, &opts) {
                Err(e) if job.quick_video && e.downcast_ref::<splat_sfm::ContinuityFailure>().is_some() => {
                    quick_without_flipped(&job.dir, &colmap_dir, &feature_images, &mut opts).map_err(|why| e.context(why))
                }
                // The full SfM flipped again: keep the retry's trajectory without its flipped frames.
                Err(e) if e.downcast_ref::<splat_sfm::ContinuityFailure>().is_some() => {
                    splat_sfm::without_flipped(&feature_images, &opts).map_err(|why| e.context(why))
                }
                other => other,
            }
        }
        other => other,
    };
    let rec = aliked_fallback(&job, &feature_images, &colmap_dir, &opts, rec);
    let (rec, learned) = match learned_fallback(&job, &feature_images, &colmap_dir, opts.holdout_every, rec) {
        Ok(v) => v,
        Err(e) => {
            // A failed SfM keeps its logs, options and attempt models for diagnosis, but not the
            // derived scratch the success path also deletes: guildhall-interior-walk synced 4.5 GB
            // back for one failed SfM (feature-image overlay, mask PNGs; its four 939 MB databases
            // stay on fal, see splat_core::remote::kept_remote).
            if let Some(masking) = masking {
                let _ = masking.join();
            }
            for dir in [overlay.clone(), job.dir.join("feature_images_unflipped"), feature_masks.clone(),
                job.dir.join("sfm_raw_masks"), job.dir.join("sfm_presence_masks"), job.dir.join("sfm_masks")] {
                let _ = std::fs::remove_dir_all(dir);
            }
            return Err(e);
        }
    };
    let (mut rec, focal_guard) = if learned.as_ref().is_none_or(|l| l["kept"] != "learned") && rec.model_dir.starts_with(&colmap_dir) {
        focal_guard(&job, &opts, &feature_images, &colmap_dir, rec)
    } else {
        (rec, None)
    };
    // Earlier attempts' databases and models (splat_sfm's hybrid fallbacks colmap.sequential/ and
    // colmap.low-angle/, the flip retry's colmap-flipped/; hundreds of MB): scratch once the final
    // model exists, and nothing on the client reads them (the driveway clip's SfM synced 394 MB back, 332 MB of
    // it colmap.sequential/database.db).
    for dir in [flipped_dir.clone(), job.dir.join("colmap-flipped2"), job.dir.join("feature_images_unflipped"), colmap_dir.with_extension("sequential"), colmap_dir.with_extension("low-angle")] {
        let _ = std::fs::remove_dir_all(dir);
    }
    let _ = std::fs::remove_dir_all(&feature_masks);
    // Inline masks are SfM scratch, not training inputs. Keep detector logs for audit, but do
    // not download hundreds of raw/presence/normalized PNGs or restage them for training.
    if let Some(masking) = masking {
        masking.join().map_err(|_| anyhow::anyhow!("SfM mask thread panicked"))??;
    }
    let _ = std::fs::remove_file(&late_done);
    for name in ["sfm_raw_masks", "sfm_presence_masks", "sfm_masks"] {
        let path = job.dir.join(name);
        if path.exists() {
            std::fs::remove_dir_all(path)?;
        }
    }
    let _ = std::fs::remove_dir_all(&overlay);
    let mut feature_scale = None;
    if job.sfm_images.is_some() && feature_images != images {
        // Cameras and keypoints are in sfm_images pixels: rescale every model (the selected one and
        // the archived components) to the training images, then report health at that size.
        let mut models = vec![rec.model_dir.clone()];
        models.extend(model::models_by_size(&colmap_dir.join("components"))?.into_iter().map(|m| m.0));
        for dir in &models {
            feature_scale = feature_scale.or(rescale_to_images(dir, &images)?);
        }
        let mut m = splat_eval::read_model(&rec.model_dir)?;
        // Zeroed stored errors make splat-eval recompute them, as splat-sfm's own health does.
        m.points.iter_mut().for_each(|p| p.error = 0.0);
        rec.health = splat_eval::sfm_health(&m, &rec.model_dir, &splat_eval::SfmHealthOptions {
            images_dir: Some(images.clone()),
            video: job.video,
            holdout_every: opts.holdout_every,
            ..Default::default()
        })?;
    }
    splat_sfm::ensure_video_continuity(&rec.health, job.video)?;
    println!("[sfm] model {} ({} registered) in {:.1} s", rec.model_dir.display(), rec.health.n_registered, rec.wall_seconds);
    println!("[sfm] kept rung {}: {}/{} frames placed", rec.rung, rec.health.n_registered, rec.health.n_input);
    // finalize_model has already moved the losing components to colmap/components/; count
    // them with the selected model so multi-component gates fire.
    let mut components = Vec::new();
    if learned.as_ref().is_some_and(|l| l["kept"] == "learned") {
        // learned/sparse for a replacement, learned/union/sparse for a union.
        let sparse = rec.model_dir.parent().ok_or_else(|| anyhow::anyhow!("learned model has no parent"))?;
        components.extend(model::models_by_size(sparse)?);
        components.extend(model::models_by_size(&sparse.with_file_name("components"))?);
    } else {
        components.extend(model::models_by_size(&colmap_dir.join("sparse"))?);
        components.extend(model::models_by_size(&colmap_dir.join("components"))?);
    }
    components.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    std::fs::create_dir_all(job.dir.join("sparse"))?;
    std::fs::rename(&rec.model_dir, job.dir.join("sparse/0"))?;
    // Brush mounts the whole dataset dir: one trainable model, no database. Losing components
    // stay in colmap/components/ (the loader skips that dir) for split/merge decisions.
    for entry in std::fs::read_dir(&colmap_dir)? {
        let p = entry?.path();
        if p.is_dir() && p.file_name().is_some_and(|n| n != "components") {
            std::fs::remove_dir_all(&p)?;
        } else if p.extension().is_some_and(|e| e == "db" || e == "ply") {
            std::fs::remove_file(&p)?;
        }
    }
    let mut rec_json = serde_json::to_value(&rec)?;
    rec_json["component_sizes"] = json!(components.iter().map(|c| c.1).collect::<Vec<_>>());
    if let Some((sx, sy)) = feature_scale {
        rec_json["sfm_images_scale"] = json!([sx, sy]);
    }
    // What the inline feature masks removed, per label (per frame in sfm_masks.json beside it).
    if let Ok(mut masks) = read_json::<serde_json::Value>(&job.dir.join("sfm_masks.json")) {
        if let Some(m) = masks.as_object_mut() {
            m.remove("per_frame");
        }
        rec_json["feature_masks"] = masks;
    }
    if let Some(learned) = learned {
        rec_json["learned_fallback"] = learned;
        // Keep the guard's decision; the rest (database copies, intermediate models) is scratch, and
        // Brush mounts the whole dataset dir, so no second model may stay under it.
        let scratch = job.dir.join("learned");
        let _ = std::fs::rename(scratch.join("guard-fallback.json"), job.dir.join("learned-guard.json"));
        let _ = std::fs::remove_dir_all(&scratch);
    }
    if let Some(guard) = focal_guard {
        rec_json["focal_guard"] = guard;
    }
    write_json(&job.dir.join("reconstruction.json"), &rec_json)?;
    Ok(0)
}

/// Keep a losing SfM attempt's logs and JSON only: Brush mounts the dataset dir, which may hold one
/// model and no database (the attempt's are hundreds of MB).
fn prune_attempt(dir: &std::path::Path) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = entry.path();
        let _ = if p.is_dir() {
            std::fs::remove_dir_all(&p)
        } else if p.extension().is_some_and(|e| e == "db" || e == "ply") {
            std::fs::remove_file(&p)
        } else {
            Ok(())
        };
    }
}

/// The shared camera's focal length (px) and radial terms (k1, k2), for the models video SfM uses.
fn camera_intrinsics(model_dir: &std::path::Path) -> Option<(splat_eval::colmap::Camera, f64, f64, f64)> {
    let m = splat_eval::read_model(model_dir).ok()?;
    let cam = m.cameras.values().next()?.clone();
    let p = &cam.params;
    let (f, k1, k2) = match cam.model.as_str() {
        "SIMPLE_PINHOLE" => (p[0], 0.0, 0.0),
        "SIMPLE_RADIAL" => (p[0], p[3], 0.0),
        "RADIAL" => (p[0], p[3], p[4]),
        "PINHOLE" => ((p[0] + p[1]) / 2.0, 0.0, 0.0),
        "OPENCV" | "FULL_OPENCV" => ((p[0] + p[1]) / 2.0, p[4], p[5]),
        _ => return None,
    };
    Some((cam, f, k1, k2))
}

/// Why a video camera is implausible (empty when it isn't): a focal length no phone lens has (under
/// 0.3x or over 1.6x the long side: 1.6 is a 35 deg view, the ultrawide room clip is 0.46-0.49), radial
/// terms that only compensate a wrong focal (|k1| > 0.5, |k2| > 1), or a focal 1.4x off a trusted
/// (lens-specific) prior. Brumby/busbee/creswell's free-focal solves settled at 2.9-4.3x the long side with k2 up
/// to 55; every other bench room is 0.46-1.11x, |k1| <= 0.22, |k2| <= 0.43.
fn implausible_camera(cam: &splat_eval::colmap::Camera, f: f64, k1: f64, k2: f64, prior: Option<&FocalPrior>) -> Vec<String> {
    let long = cam.width.max(cam.height) as f64;
    let mut why = Vec::new();
    if !(0.3..=1.6).contains(&(f / long)) {
        why.push(format!("focal {f:.0} px is {:.2}x the long side", f / long));
    }
    if k1.abs() > 0.5 {
        why.push(format!("k1 {k1:.3}"));
    }
    if k2.abs() > 1.0 {
        why.push(format!("k2 {k2:.3}"));
    }
    if let Some(prior) = prior.filter(|p| p.trusted()) {
        let ratio = f / prior.focal_px(cam.width, cam.height);
        if !(1.0 / FOCAL_PRIOR_RATIO..=FOCAL_PRIOR_RATIO).contains(&ratio) {
            why.push(format!("focal {ratio:.2}x the {} value ({:.0} deg)", prior.source, prior.long_fov_deg));
        }
    }
    why
}

/// Intrinsics guard for video. A free focal length can drift into a degenerate solve: a narrow
/// lens whose radial terms bend it back (brumby f 1920 -> 6911 px at 1600 px wide, k1 2.63, k2 34.9;
/// the pre-check's turns say 61 deg, ~1340 px). Such a camera gets one re-solve on the same matches
/// with the focal fixed at the prior (the pre-check's lens tag, phone model, self-calibration or
/// assumed 1x lens) and one radial term (SIMPLE_RADIAL). The re-solve is kept only if its camera is
/// plausible, it places at least 98% as many frames, passes the continuity check (a flip or a collapsed
/// section fails the re-solve) and fails no more health gates; otherwise the original stays. Plausible cameras never
/// reach the re-solve. Returns the model to use and the decision for reconstruction.json.
fn focal_guard(
    job: &SfmJob, opts: &splat_sfm::ReconstructOptions, feature_images: &std::path::Path, colmap_dir: &std::path::Path,
    rec: splat_sfm::Reconstruction,
) -> (splat_sfm::Reconstruction, Option<serde_json::Value>) {
    if !job.video || job.camera_model != "RADIAL" || std::env::var("SPLAT_SFM_FOCAL_GUARD").as_deref() == Ok("0") {
        return (rec, None);
    }
    let Some((cam, f, k1, k2)) = camera_intrinsics(&rec.model_dir) else { return (rec, None) };
    let prior = job.focal_prior.clone();
    let why = implausible_camera(&cam, f, k1, k2, prior.as_ref());
    if why.is_empty() {
        return (rec, None);
    }
    // No pre-check (a run without one): a 1x phone lens, 65 deg across the long edge.
    let prior = prior.unwrap_or(FocalPrior {
        long_fov_deg: 65.0,
        source: "default 1x lens".into(),
        pairs: 0,
        source_size: (cam.width as u32, cam.height as u32),
    });
    let prior_f = prior.focal_px(cam.width, cam.height);
    let long = cam.width.max(cam.height) as f64;
    let before = json!({"model": cam.model, "focal_px": f, "k1": k1, "k2": k2, "registered": rec.health.n_registered,
        "reproj_px": rec.health.mean_reproj_err_px, "pass": rec.health.pass});
    let mut decision = json!({"reasons": why, "before": before, "prior": {"long_fov_deg": prior.long_fov_deg, "source": prior.source,
        "pairs": prior.pairs, "focal_px": prior_f}});
    println!("[sfm] implausible camera ({}); re-solving with the focal fixed at {prior_f:.0} px ({}, {:.0} deg)",
        why.join(", "), prior.source, prior.long_fov_deg);
    let free = colmap_dir.with_extension("free-focal");
    let database = free.join("database.db");
    let _ = std::fs::remove_dir_all(&free);
    if splat_sfm::deadline_passed(opts, "the intrinsics re-solve") || !colmap_dir.join("database.db").is_file()
        || std::fs::rename(colmap_dir, &free).is_err()
    {
        decision["kept"] = json!("original");
        decision["why"] = json!("no time or no database for a re-solve");
        return (rec, Some(decision));
    }
    let mut fixed = opts.clone();
    // Incremental, as before the global video default: its focal stays fixed through every BA, and
    // the flip retry below changes its initial pair.
    fixed.mapper = splat_sfm::Mapper::Incremental;
    fixed.camera_model = "SIMPLE_RADIAL".into();
    fixed.fixed_focal_factor = Some(prior_f / long);
    fixed.reuse_database = Some(database);
    fixed.reuse_matches = false;
    // The database already holds every match (and the late masks' filtering): map it, no rematch.
    fixed.matching = splat_sfm::Matching::Exhaustive;
    fixed.vocab_tree = None;
    fixed.late_masks = None;
    fixed.mask_dir = None;
    fixed.init_pair = None;
    let start = std::time::Instant::now();
    let mut retried = splat_sfm::reconstruct(feature_images, &fixed);
    // A flipped or collapsed trajectory gets the free solve's flip retry: an 8 deg initial pair.
    // Busbee's fixed-focal solve (CPU, 2026-10-12) placed 591 frames but collapsed two 12-frame turns.
    if retried.as_ref().is_err_and(|e| e.downcast_ref::<splat_sfm::ContinuityFailure>().is_some())
        && !splat_sfm::deadline_passed(opts, "the intrinsics re-solve's flip retry")
    {
        println!("[sfm] intrinsics re-solve failed pose continuity; retrying it with an 8 deg initial pair");
        let flipped = colmap_dir.with_extension("fixed-focal-flipped");
        let _ = std::fs::remove_dir_all(&flipped);
        if std::fs::rename(colmap_dir, &flipped).is_ok() {
            prune_attempt(&flipped);
            fixed.init_min_tri_angle = 8.0;
            retried = splat_sfm::reconstruct(feature_images, &fixed);
        }
    }
    decision["seconds"] = json!(start.elapsed().as_secs_f64());
    let failed = |h: &splat_eval::SfmHealth| h.gates.values().filter(|ok| !**ok).count();
    let verdict = match &retried {
        Err(e) => Err(format!("re-solve failed: {e:#}")),
        Ok(new) => {
            let after = camera_intrinsics(&new.model_dir);
            decision["after"] = json!({"model": after.as_ref().map(|a| a.0.model.clone()), "focal_px": after.as_ref().map(|a| a.1),
                "k1": after.as_ref().map(|a| a.2), "k2": after.as_ref().map(|a| a.3), "registered": new.health.n_registered,
                "reproj_px": new.health.mean_reproj_err_px, "pass": new.health.pass});
            match after {
                None => Err("re-solve has no camera".to_string()),
                Some((c, f, k1, k2)) if !implausible_camera(&c, f, k1, k2, None).is_empty() => {
                    Err(format!("re-solve camera still implausible ({})", implausible_camera(&c, f, k1, k2, None).join(", ")))
                }
                Some(_) if (new.health.n_registered as f64) < FOCAL_RESOLVE_MIN_PLACED * rec.health.n_registered as f64 => {
                    Err(format!("re-solve places {} frames, the original {}", new.health.n_registered, rec.health.n_registered))
                }
                Some(_) if (rec.health.pass && !new.health.pass) || failed(&new.health) > failed(&rec.health) => {
                    Err("re-solve fails more health gates".to_string())
                }
                Some(_) => Ok(()),
            }
        }
    };
    match (verdict, retried) {
        (Ok(()), Ok(new)) => {
            println!("[sfm] intrinsics re-solve kept: focal {f:.0} -> {prior_f:.0} px, {} -> {} frames",
                rec.health.n_registered, new.health.n_registered);
            prune_attempt(&free);
            decision["kept"] = json!("fixed_focal");
            (new, Some(decision))
        }
        (verdict, _) => {
            let why = verdict.err().unwrap_or_default();
            println!("[sfm] intrinsics re-solve rejected: {why}");
            let rejected = colmap_dir.with_extension("fixed-focal");
            let _ = std::fs::remove_dir_all(&rejected);
            if colmap_dir.exists() && std::fs::rename(colmap_dir, &rejected).is_ok() {
                prune_attempt(&rejected);
            }
            if let Err(e) = std::fs::rename(&free, colmap_dir) {
                eprintln!("[sfm] restoring the free-focal model failed: {e:#}");
            }
            decision["kept"] = json!("original");
            decision["why"] = json!(why);
            (rec, Some(decision))
        }
    }
}

/// Rescale a model reconstructed from larger copies of the images to the images in `images` (same
/// names): each camera takes the size of the images that use it. Returns the (x, y) scale applied.
fn rescale_to_images(model_dir: &std::path::Path, images: &std::path::Path) -> anyhow::Result<Option<(f64, f64)>> {
    let m = splat_eval::read_model(model_dir)?;
    let mut sizes = std::collections::BTreeMap::new();
    for im in m.images.values() {
        let (w, h) = image::image_dimensions(images.join(&im.name))?;
        let size = (u64::from(w), u64::from(h));
        if let Some(prev) = sizes.insert(im.camera_id, size).filter(|p| *p != size) {
            anyhow::bail!("camera {}: training images of different sizes ({prev:?}, {size:?})", im.camera_id);
        }
    }
    let scale = sizes.iter().find_map(|(id, (w, h))| m.cameras.get(id).map(|c| (*w as f64 / c.width as f64, *h as f64 / c.height as f64)));
    splat_eval::colmap::rescale_model_bin(model_dir, &sizes)?;
    Ok(scale)
}

pub fn complete(run: &dyn RunCtx, artifact_id: &str, exit_code: i32, log: &str) -> anyhow::Result<ToolResult> {
    let (artifact, dir) = get_artifact(run, artifact_id, ArtifactKind::Sfm)?;
    if let Some(imported) = crate::capture_sfm::complete(run, &artifact, &dir, exit_code)? { return Ok(imported); }
    let rec: serde_json::Value = read_json(&dir.join("reconstruction.json")).unwrap_or_default();
    if exit_code != 0 || !dir.join("sparse/0").is_dir() || rec.is_null() {
        let code = if log.contains("wrote no reconstruction") || log.contains("empty reconstruction") { "no_reconstruction" } else { "subprocess_failed" };
        let mut r = failure("reconstruct_sfm", code, format!("SfM exited {exit_code}: {log}"));
        // A capture with no camera movement fails here as "no initial pair" (shamian violinist) or
        // a mapper timeout (shamian statues): say that, and keep the log as evidence.
        if let (Some((msg, details)), Some(e)) = (low_parallax_message(capture_motion(run, &artifact, &dir)), r.error.as_mut()) {
            eprintln!("[sfm] {details}");
            let log = std::mem::replace(&mut e.message, msg);
            e.evidence.push(details);
            e.evidence.push(log);
        }
        r.artifacts = vec![artifact];
        return Ok(r);
    }
    let serde_json::Value::Object(mut metrics) = rec["health"].clone() else { anyhow::bail!("reconstruction.json has no health") };
    metrics.insert("registered_count".into(), metrics["n_registered"].clone());
    if let (Some(input), Some(registered)) = (metrics["n_input"].as_u64(), metrics["n_registered"].as_u64()) {
        metrics.insert("unregistered_count".into(), json!(input.saturating_sub(registered)));
        metrics.insert("unregistered_reason".into(), json!("not registered in the selected SfM model"));
    }
    let camera_extent = metrics["camera_extent"].as_f64().unwrap_or(0.0);
    let pts = metrics["points_median_dist_from_cameras"].as_f64().unwrap_or(0.0);
    // Pure rotation (tripod pan) leaves the cameras in a tiny ball relative to the scene.
    metrics.insert("baseline_ratio".into(), json!(if pts > 0.0 { camera_extent / pts } else { 0.0 }));
    let sizes = rec["component_sizes"].as_array().cloned().unwrap_or_default();
    metrics.insert("components".into(), json!(sizes.len()));
    // splat-sfm's inventory counts distinct frames across pieces; archived pieces whose frames
    // the largest model absorbed add nothing, so they don't count as a split.
    let union = rec["components"]["n_registered_union"].as_u64().unwrap_or_else(|| sizes.iter().filter_map(|v| v.as_u64()).sum());
    let largest = sizes.first().and_then(|v| v.as_u64()).unwrap_or(0);
    let split = union > largest;
    metrics.insert("component_sizes".into(), json!(sizes));
    metrics.insert("component_union".into(), json!(union));
    metrics.insert("largest_component_fraction".into(), json!(if union > 0 { largest as f64 / union as f64 } else { 0.0 }));
    metrics.insert("sfm_wall_s".into(), rec["wall_seconds"].clone());
    if !rec["sfm_images_scale"].is_null() {
        metrics.insert("sfm_images_scale".into(), rec["sfm_images_scale"].clone());
    }
    let job = read_json::<SfmJob>(&dir.join("sfm_job.json")).ok();
    if let Some(job) = &job {
        metrics.insert("matching".into(), json!(matching_name(job)));
        metrics.insert("matching_requested".into(), json!(job.pairs));
        metrics.insert("loop_detection".into(), json!(!matches!(matching_policy(job), splat_sfm::Matching::Exhaustive) && job.vocab_tree.is_some()));
        metrics.insert("camera_model".into(), json!(job.camera_model));
    }
    // The flip retry changed the camera model: report the one reconstructed.
    if dir.join("colmap-flipped").is_dir() {
        metrics.insert("sfm_flip_retry".into(), json!(true));
        let model = metrics["cameras"].as_object().and_then(|c| c.values().next()).map(|c| c["model"].clone());
        if let Some(model) = model {
            metrics.insert("camera_model".into(), model);
        }
    }
    metrics.insert("gpu_lock_wait_s".into(), rec["lock_wait_seconds"].clone());
    if let Ok(a) = read_json::<serde_json::Value>(&dir.join("aliked_fallback.json")) {
        metrics.insert("aliked_fallback".into(), a);
    }
    if !rec["learned_fallback"].is_null() {
        metrics.insert("learned_fallback".into(), rec["learned_fallback"].clone());
        metrics.insert("camera_source".into(), json!(if rec["learned_fallback"]["kept"] == "learned" { "learned" } else { "colmap" }));
    }
    for stage in rec["stages"].as_array().into_iter().flatten() {
        if let (Some(name), Some(sec)) = (stage["name"].as_str(), stage["seconds"].as_f64()) {
            metrics.insert(format!("{name}_s"), json!(sec));
        }
    }
    if let Ok(serde_json::Value::Object(selection)) = read_json::<serde_json::Value>(&dir.join("capture_selection.json")) {
        metrics.extend(selection);
    }
    let calibration = crate::calibration_health::dataset(&splat_eval::read_model(&dir.join("sparse/0"))?, &dir,
        job.as_ref().is_some_and(|j| j.video))?;
    metrics.insert("calibration_health".into(), calibration.clone());
    metrics.entry("camera_source").or_insert(json!("sfm"));
    metrics.entry("sfm_skipped").or_insert(json!(false));
    write_json(&dir.join("health.json"), &metrics)?;
    if let Err(error) = crate::calibration_health::enforce(&calibration) {
        let mut r = failure("reconstruct_sfm", "calibration_health_failed", error.to_string());
        r.artifacts = vec![artifact];
        r.metrics = metrics.into_iter().collect();
        return Ok(r);
    }
    let (registered, input) = (metrics["n_registered"].as_u64().unwrap_or(0), metrics["n_input"].as_u64().unwrap_or(0));
    let preview = job.as_ref().is_some_and(|j| j.preview);
    let too_few = too_few_registered(registered, largest, input, preview);
    // Too few placed but one coherent section of at least MIN_SECTION_FRAMES (walden hyperlapse: only
    // the lake end, 37/123): train that section and say which it is; the run finishes `partial`
    // (sfm_coverage_defect) instead of failing outright.
    let sizes_n: Vec<u64> = sizes.iter().filter_map(|v| v.as_u64()).collect();
    let section = (too_few && registered >= MIN_SECTION_FRAMES && !(split && sizes.len() > 1 && !dominant_group(&sizes_n)))
        .then(|| registered_section(&dir))
        .flatten();
    if let Some(section) = &section {
        metrics.insert("registered_section".into(), json!(section));
    }
    if too_few && section.is_none() {
        let mut r = failure(
            "reconstruct_sfm",
            "too_few_registered",
            if split && sizes.len() > 1 {
                let groups: Vec<String> = sizes.iter().filter_map(|v| v.as_u64()).map(|n| n.to_string()).collect();
                format!(
                    "The frames formed {} separate groups ({}); the largest has only {registered} of {input} frames. The input likely mixes different scenes or has gaps in coverage: use one continuous capture, or add frames that bridge the groups.",
                    groups.len(),
                    groups.join("/")
                )
            } else {
                format!("Only {registered} of {input} frames could be placed. Try footage with slower camera motion and more overlap between frames, or extract more frames from the same clip.")
            },
        );
        if let Some(e) = r.error.as_mut() {
            e.retryable = true;
            if let Some((msg, details)) = low_parallax_message(capture_motion(run, &artifact, &dir)) {
                eprintln!("[sfm] {details}");
                let log = std::mem::replace(&mut e.message, msg);
                e.evidence.push(details);
                e.evidence.push(log);
            }
            if metrics.get("learned_fallback").is_some_and(|l| l["reason"] == "DA3 weights not installed") {
                e.message.push_str(" (The learned-pose rescue for sparse captures was unavailable: DA3-BASE weights are not installed.)");
            }
        }
        r.metrics = metrics.into_iter().collect();
        r.artifacts = vec![artifact];
        return Ok(r);
    }
    let mut warnings: Vec<String> = metrics["gates"]
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(_, ok)| **ok != json!(true))
        .map(|(k, _)| format!("SfM health gate failed: {k}"))
        .collect();
    if let Some(section) = &section {
        warnings.push(format!(
            "Only {registered} of {input} frames could be placed; the scene is built from {section} only. The rest of the capture lacks overlap between frames: re-shoot it with slower camera motion."
        ));
    }
    if split {
        warnings.push(format!("Multiple reconstruction components: {} frames registered only outside the largest; training uses the largest.", union - largest));
    }
    if metrics.get("learned_fallback").is_some_and(|l| l["reason"] == "DA3 weights not installed") {
        warnings.push("The learned-pose rescue for sparse captures is unavailable: DA3-BASE weights are not installed.".into());
    }
    if metrics.get("camera_source").is_some_and(|v| v == "learned") {
        let lf = &metrics["learned_fallback"];
        let why = match lf["check"]["delta_db"].as_f64() {
            Some(d) => format!("kept by a held-out check against COLMAP's model ({d:+.2} dB on {} shared frames)", lf["check"]["n_heldout"]),
            None => "with no COLMAP baseline to check it against".into(),
        };
        warnings.push(format!(
            "COLMAP placed only {} of {} frames; cameras come from the learned-pose rescue (DA3 seed), {why}.",
            lf["colmap_registered"], lf["n_input"]
        ));
    }
    if metrics["baseline_ratio"].as_f64().is_some_and(|b| b < 0.1) {
        warnings.push("Camera baseline is tiny relative to the scene (likely pure rotation); 3D may not be recoverable.".into());
    }
    if let Some(chain) = job.as_ref().and_then(|j| j.preview_train.as_ref()) {
        match crate::train::adopt_chained(run, &artifact.id, chain) {
            Ok((split, candidate, summary)) => {
                metrics.insert("preview_split_id".into(), json!(split));
                metrics.insert("preview_candidate_id".into(), json!(candidate));
                metrics.insert("preview_training_s".into(), summary.get("training_s").cloned().unwrap_or_default());
                metrics.insert("preview_splats".into(), summary.get("splats").cloned().unwrap_or_default());
            }
            Err(error) => {
                metrics.insert("preview_chain_error".into(), json!(format!("{error:#}")));
            }
        }
    }
    Ok(ToolResult {
        artifacts: vec![artifact],
        metrics: metrics.into_iter().collect(),
        warnings,
        ..result("reconstruct_sfm", Status::Succeeded)
    })
}

/// Below this many image widths of camera travel over a whole clip there is too little baseline for
/// SfM's initial pair (shamian violinist 0.23, statues 0.54: no initial pair / mapper timeout).
const LOW_MOTION_WIDTHS: f64 = 1.0;
/// At or above this share of verified pairs low-parallax (COLMAP planar/panoramic), no pair has the
/// viewpoint change to triangulate from (shamian violinist 1.00, statues 0.93-0.98).
const LOW_PARALLAX_SHARE: f64 = 0.9;

/// How much the camera moved: the frame plan's total motion in image widths (videos only) and the
/// highest low-parallax pair share any mapper attempt measured (colmap*/motion.json).
fn capture_motion(run: &dyn RunCtx, sfm: &Artifact, dir: &std::path::Path) -> (Option<f64>, Option<f64>) {
    let widths = sfm.meta.get("frames_id").and_then(|v| v.as_str())
        .and_then(|id| get_artifact(run, id, ArtifactKind::Frames).ok())
        .and_then(|(_, frames)| read_json::<serde_json::Value>(&frames.join("selection.json")).ok())
        .and_then(|s| s["metrics"]["frame_plan"]["motion_widths"].as_f64());
    let share = std::fs::read_dir(dir).into_iter().flatten().flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("colmap"))
        .filter_map(|e| read_json::<serde_json::Value>(&e.path().join("motion.json")).ok())
        .filter_map(|m| m["low_parallax_share"].as_f64())
        .reduce(f64::max);
    (widths, share)
}

/// The user-facing reason for a failed or mostly-unplaced solve when the camera barely moved: a clip
/// under LOW_MOTION_WIDTHS of travel, or (photos too) a LOW_PARALLAX_SHARE of low-parallax pairs.
/// Returns the plain message and a details line with the measurements (it goes to the evidence).
fn low_parallax_message((widths, share): (Option<f64>, Option<f64>)) -> Option<(String, String)> {
    let pairs = share.filter(|s| *s >= LOW_PARALLAX_SHARE).map(|s| format!("{:.0}% of matched image pairs show almost no change in viewpoint", s * 100.0));
    const MOVED: &str = "The camera barely moved, so there's no depth to build a 3D scene from. Re-shoot while walking around the subject or stepping sideways, not just turning in place.";
    match (widths, pairs) {
        (Some(w), pairs) if w < LOW_MOTION_WIDTHS => {
            let about = if w < 0.1 { "less than 0.1".to_string() } else { format!("about {w:.1}") };
            let pairs = pairs.map(|p| format!("; {p}")).unwrap_or_default();
            Some((MOVED.into(), format!("Camera motion: {about} image widths over the whole clip{pairs}; aim for at least ~2 image widths.")))
        }
        // Plenty of image motion but no viewpoint change: the camera turned in place.
        (Some(w), Some(p)) => Some((
            "The camera turned in place instead of moving, so there's no depth to build a 3D scene from. Re-shoot while walking around the subject or stepping sideways, not just turning in place.".into(),
            format!("The view moved about {w:.1} image widths but the camera stayed in place ({p})."),
        )),
        (None, Some(p)) => Some((
            "The photos were taken from nearly the same spot, so there's no depth to build a 3D scene from. Re-take them while walking around the subject, stepping sideways between shots.".into(),
            format!("{p}."),
        )),
        (_, None) => None,
    }
}

/// Training on a handful of cameras only produces garbage, so fewer than half the frames placed
/// is a failure, not a warning (casual8: driveway 3/37, night-stroll 22/150). A quick preview solve
/// also passes with 40% of its frames in its largest model: its draft is replaced by the full SfM's
/// result, and waiting for that one costs the first view minutes (pegasus: the 24-frame quick SfM
/// kept 11 frames in one model, so the first view waited 197 s for the full SfM).
fn too_few_registered(registered: u64, largest: u64, input: u64, preview: bool) -> bool {
    registered * 2 < input && !(preview && largest * 5 >= input * 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_coverage_bad_geometry_routes_rescue_even_in_preview_and_never_returns_bad_model() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let images = dir.join("images");
        std::fs::create_dir(&images).unwrap();
        for i in 0..179 { std::fs::write(images.join(format!("frame_{i:06}.jpg")), []).unwrap(); }
        std::fs::write(dir.join("database.db"), []).unwrap();
        let job: SfmJob = serde_json::from_value(json!({"colmap":"unused", "dir":dir, "camera_model":"PINHOLE",
            "video":false, "preview":true, "max_image_size":1180, "max_features":8192, "sequential":true})).unwrap();
        let mut health = splat_eval::sfm_health(&splat_eval::Model::default(), dir, &Default::default()).unwrap();
        let evidence = serde_json::from_str(include_str!("../../splat-eval/tests/fixtures/failed179.json")).unwrap();
        health.geometry = splat_eval::geometry::assess_geometry(&evidence, false);
        health.n_registered = 179;
        health.n_input = 179;
        let rec = splat_sfm::Reconstruction { model_dir: dir.into(), health,
            components: splat_sfm::ComponentInventory { models: vec![], n_registered_union:179, largest_component_fraction:1.0 },
            wall_seconds:0.0, lock_wait_seconds:0.0, stages:vec![], colmap_version:String::new(), environment:Default::default(), rung:"test".into() };
        let err = learned_fallback(&job, &images, dir, 8, Ok(rec)).unwrap_err();
        assert!(err.downcast_ref::<splat_sfm::GeometryFailure>().is_some());
        // Reaching the missing-weights error proves the high-coverage preview did not bypass rescue.
        if std::env::var("SPLAT_SFM_RESCUE").as_deref() != Ok("0") {
            assert!(format!("{err:#}").contains("DA3 weights"));
        }
    }

    #[test]
    fn focal_guard_ignores_phone_model_priors() {
        let cam = |w, h| splat_eval::colmap::Camera { id: 1, model: "RADIAL".into(), width: w, height: h, params: vec![] };
        // Room 4: Pixel 10 Pro 1x, portrait 900 x 1600 from 2160 x 3840, free solve f 1435 px. The model
        // table said 95 deg (the ultrawide); the clip's own turns say 58 deg from 32 pairs.
        let room4 = json!({"fov_deg": [63.5, 95.46], "fov_source": "phone model", "fov_estimate": {"long_deg": 58.0, "pairs": 32},
            "width": 2160, "height": 3840});
        let prior = FocalPrior::from_check_metrics(&room4).unwrap();
        assert_eq!((prior.source.as_str(), prior.long_fov_deg), ("estimated", 58.0));
        assert!(implausible_camera(&cam(900, 1600), 1435.0, 0.05, -0.1, Some(&prior)).is_empty());
        // A phone-model value alone never decides: the 95 deg one, few turns to replace it.
        let few = json!({"fov_deg": [63.5, 95.46], "fov_source": "phone model", "fov_estimate": {"long_deg": 58.0, "pairs": 4},
            "width": 2160, "height": 3840});
        let prior = FocalPrior::from_check_metrics(&few).unwrap();
        assert_eq!(prior.source, "phone model");
        assert!(implausible_camera(&cam(900, 1600), 1435.0, 0.05, -0.1, Some(&prior)).is_empty());
        // Brumby / creswell's degenerate solves (landscape 1600 x 900) still fire, with or without a prior.
        for (f, fov, pairs) in [(4085.0, 61.0, 117), (4633.0, 73.0, 13)] {
            let m = json!({"fov_deg": [fov, 40.0], "fov_source": "estimated", "fov_estimate": {"long_deg": fov, "pairs": pairs},
                "width": 1920, "height": 1080});
            let prior = FocalPrior::from_check_metrics(&m).unwrap();
            assert!(!implausible_camera(&cam(1600, 900), f, 0.0, 0.0, Some(&prior)).is_empty(), "{f}");
            assert!(!implausible_camera(&cam(1600, 900), f, 0.0, 0.0, None).is_empty(), "{f}");
        }
    }

    #[test]
    fn low_parallax_message_needs_little_motion_or_mostly_low_parallax_pairs() {
        // shamian violinist: 0.23 widths, every pair low-parallax. Plain message, numbers in details.
        let (m, d) = low_parallax_message((Some(0.23), Some(1.0))).unwrap();
        assert!(m.starts_with("The camera barely moved, so there's no depth"), "{m}");
        assert!(!m.contains("widths"), "{m}");
        assert!(d.starts_with("Camera motion: about 0.2 image widths over the whole clip; 100% of"), "{d}");
        // Little motion alone; the share is unknown when the mapper never ran.
        assert!(low_parallax_message((Some(0.54), None)).unwrap().1.contains("about 0.5 image widths"));
        assert!(low_parallax_message((Some(0.04), Some(0.2))).unwrap().1.contains("less than 0.1"));
        // Enough motion and general pairs: keep the original message.
        assert_eq!(low_parallax_message((Some(14.3), Some(0.58))), None);
        assert_eq!(low_parallax_message((None, None)), None);
        // A video that moved but whose pairs are still nearly all low-parallax (pure rotation).
        let (m, d) = low_parallax_message((Some(3.0), Some(0.95))).unwrap();
        assert!(m.starts_with("The camera turned in place instead of moving"), "{m}");
        assert!(d.starts_with("The view moved about 3.0 image widths"), "{d}");
        // Photos: only the pair share applies.
        assert!(low_parallax_message((None, Some(0.92))).unwrap().0.starts_with("The photos were taken from nearly the same spot"));
        assert_eq!(low_parallax_message((None, Some(0.5))), None);
    }

    #[test]
    fn a_split_trains_its_section_only_when_one_group_dominates() {
        assert!(dominant_group(&[258, 71, 52, 45, 44, 40, 9])); // guildhall
        assert!(dominant_group(&[37]));
        assert!(!dominant_group(&[60, 55])); // two scenes mixed
        assert!(dominant_group(&[71, 258])); // order does not matter
    }

    #[test]
    fn preview_floor_is_forty_percent_in_one_model_and_full_keeps_half() {
        // Pegasus quick SfM: 11 of 24 in one model.
        assert!(too_few_registered(11, 11, 24, false));
        assert!(!too_few_registered(11, 11, 24, true));
        // 40% exactly passes the preview; just under fails both.
        assert!(!too_few_registered(10, 10, 25, true));
        assert!(too_few_registered(9, 9, 25, true));
        // The preview floor counts one model: 10 of 24 registered across pieces of 6 and 4 fails.
        assert!(too_few_registered(10, 6, 24, true));
        // Half registered passes either way; the preview is never stricter than the full rule.
        assert!(!too_few_registered(12, 12, 24, false));
        assert!(!too_few_registered(12, 7, 24, true));
    }
}

#[cfg(test)]
mod input_contract_tests {
    use super::*;

    #[test]
    fn input_contract_matching_override_survives_worker_serialization() {
        let mut job: SfmJob = serde_json::from_value(json!({"colmap":"colmap","dir":"sfm",
            "camera_model":"RADIAL","video":true,"max_image_size":1600,"max_features":8192,
            "sequential":false,"vocab_tree":"tree.bin","pairs":"exhaustive"})).unwrap();
        assert_eq!(matching_name(&job), "exhaustive");
        job.pairs = Some(Pairs::Auto);
        assert_eq!(matching_name(&job), "hybrid"); // sequential + visual loop closure, then coverage recovery
        job.pairs = Some(Pairs::Sequential);
        assert_eq!(matching_name(&job), "sequential");
        job.pairs = Some(Pairs::Auto);
        job.video = false;
        assert_eq!(matching_name(&job), "exhaustive");
        job.video = true;
        job.vocab_tree = None;
        assert_eq!(matching_name(&job), "sequential");
        let restored: SfmJob = serde_json::from_value(serde_json::to_value(&job).unwrap()).unwrap();
        assert_eq!(matching_name(&restored), "sequential");
        job.pairs = None;
        assert_eq!(matching_name(&job), "exhaustive");
        job.sequential = true;
        assert_eq!(matching_name(&job), "sequential");
    }

    #[test]
    fn input_contract_video_defaults_preserve_explicit_anisotropic_calibration() {
        assert_eq!(calibration_model(CameraModel::Auto, true).unwrap(), "RADIAL");
        assert_eq!(calibration_model(CameraModel::Auto, false).unwrap(), "PINHOLE");
        assert_eq!(calibration_model(CameraModel::Pinhole, true).unwrap(), "PINHOLE");
        assert_eq!(calibration_model(CameraModel::Opencv, true).unwrap(), "OPENCV");
        for model in ["PINHOLE", "OPENCV"] {
            let mut opts = splat_sfm::ReconstructOptions::new(std::path::Path::new("unused"));
            opts.camera_model = model.into();
            flip_retry(&mut opts);
            assert_eq!(opts.camera_model, model);
        }
    }
}
