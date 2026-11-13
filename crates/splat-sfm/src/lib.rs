//! COLMAP 4.2.1 reconstruction, including its integrated GLOMAP pipeline.
//! GPU-heavy stages hold the configured scheduling lock; incremental mapping does not.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::Serialize;
mod union;
pub use union::pose_agreement;
use splat_eval::{SfmHealth, SfmHealthOptions, read_model, sfm_health};

#[derive(Clone, Copy, Debug, Default, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Mapper {
    Global,
    #[default]
    Incremental,
}

#[derive(Clone, Copy, Debug, Default, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Matching {
    #[default]
    Exhaustive,
    Sequential,
    /// Sequential (with vocab-tree loop detection when `vocab_tree` is set); if that registers
    /// under 95% of the frames, exhaustive matching fills in the missing pairs and mapping reruns
    /// (after a collapse, the same matches are first mapped from a lower-angle initial pair).
    Hybrid,
}

/// Learned models are explicit local artifacts. Inference uses COLMAP's ONNX Runtime.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Features {
    #[default]
    Sift,
    AlikedLightGlue {
        detector: PathBuf,
        matcher: PathBuf,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct ReconstructOptions {
    /// Wall-clock deadline for starting further fallback attempts (hybrid exhaustive/low-angle
    /// fallbacks, the serial-mapper retry). Past it the best attempt so far is returned instead of
    /// spending the job's time box: copped-hall-first-floor (600 frames) burned its 3310 s box on an
    /// exhaustive fallback (1236 s of matching) and two remaps and was killed. Set by the caller;
    /// see [`budget_deadline`].
    #[serde(skip)]
    pub deadline: Option<Instant>,
    pub colmap: PathBuf,
    /// Must not exist. Fresh databases prevent cached work from contaminating timings.
    pub output_dir: PathBuf,
    /// Optional shared GPU lock. Do not set when the caller already owns the same flock.
    pub gpu_lock: Option<PathBuf>,
    pub mapper: Mapper,
    pub matching: Matching,
    pub features: Features,
    pub video: bool,
    pub single_camera: bool,
    pub camera_model: String,
    /// Incremental mapper's minimum triangulation angle (deg) for the initial pair.
    pub init_min_tri_angle: f64,
    /// Incremental-mapper RANSAC seed and thread count. The multithreaded mapper is not
    /// reproducible (night-stroll A/A from one database: 150 vs 147+11 images); measurement
    /// harnesses pin threads to 1 and sample paired seeds (rule:sfm-noise). Product: seed 0,
    /// `num_threads`. `new` reads SPLAT_SFM_MAPPER_SEED / SPLAT_SFM_MAPPER_THREADS (pin threads to 1
    /// for repeatable solves).
    pub mapper_seed: u64,
    pub mapper_threads: Option<usize>,
    /// Incremental mapper's initial image ids; the serial gate-failure retry drops it. COLMAP
    /// retries the pair for every further model, so a forced attempt can run long (room: 806 s
    /// instead of ~118 s).
    pub init_pair: Option<(u32, u32)>,
    /// Features and matches from an earlier run on the same frames: copied in, its camera reset
    /// to `camera_model`, and extraction and matching skipped.
    pub reuse_database: Option<PathBuf>,
    /// With `reuse_database` and exhaustive matching: match the pairs the database lacks.
    pub reuse_matches: bool,
    /// COLMAP masks: black excludes features; relative image name plus `.png`.
    pub mask_dir: Option<PathBuf>,
    /// Feature masks (COLMAP names, black = ignore) that another thread is still writing while
    /// features are extracted unmasked and matched: before mapping, wait for `<dir>.done` ("ok", or
    /// anything else for none) and drop the verified matches on masked keypoints.
    pub late_masks: Option<PathBuf>,
    pub max_image_size: usize,
    pub max_features: usize,
    pub first_octave: i32,
    pub num_threads: usize,
    pub use_gpu: bool,
    pub sequential_overlap: usize,
    /// Additional long-range candidates at powers of two in the video ordering.
    pub quadratic_overlap: bool,
    /// Explicit vocabulary-tree artifact enables visual loop retrieval; no implicit download.
    pub vocab_tree: Option<PathBuf>,
    /// Incremental mapper: global BA after the model grows by this factor (COLMAP default 1.1).
    pub ba_global_ratio: Option<f64>,
    /// Incremental mapper: local BA refinement passes (COLMAP default 2).
    pub ba_local_refinements: Option<usize>,
    /// Incremental mapper: solver iterations and refinement passes per global BA (COLMAP 50 / 5).
    pub ba_global_max_iterations: Option<usize>,
    pub ba_global_refinements: Option<usize>,
    /// Incremental mapper: global BA skips 3D points whose images are already covered by others
    /// (COLMAP's ba_global_ignore_redundant_points3D).
    pub ba_global_skip_redundant: bool,
    /// Incremental mapper keeps lens distortion fixed; one bundle adjustment refines focal length
    /// and distortion on the final model instead.
    pub lens_last: bool,
    /// Initial focal length as a factor of the image's longer side, when the frames carry no
    /// EXIF focal (COLMAP's own default is 1.2, a 45 degree FOV).
    pub default_focal_length_factor: Option<f64>,
    /// With `lens_last`, the incremental mapper also keeps the focal length fixed at its initial
    /// value; only the final lens bundle adjustment refines it, over every registered frame.
    pub focal_last: bool,
    /// Focal length fixed at this factor of the image's longer side for the whole solve: the
    /// database camera (extracted or reused) starts there and no mapper or bundle adjustment
    /// refines it; distortion is still refined. The intrinsics guard's re-solve (splat-brush sfm)
    /// when a free focal drifted off (brumby: 1920 -> 6911 px at 1600 px wide, k2 34.9).
    pub fixed_focal_factor: Option<f64>,
    /// Incremental mapper maps every n-th frame (sorted names, plus the last); the continuation
    /// step then registers the rest with those poses fixed. 1 maps every frame.
    pub keyframe_stride: usize,
    pub guided_matching: bool,
    pub ransac_max_trials: usize,
    pub min_track_length: usize,
    pub calibrate: bool,
    pub ba_iterations: usize,
    pub ba_max_iterations: usize,
    pub holdout_every: usize,
    /// The rung this attempt runs, for logs ("hybrid", "exhaustive", ...).
    #[serde(skip)]
    pub rung: &'static str,
}

impl ReconstructOptions {
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            deadline: None,
            colmap: "colmap".into(),
            output_dir: output_dir.into(),
            gpu_lock: None,
            mapper: Mapper::Incremental,
            matching: Matching::Exhaustive,
            features: Features::Sift,
            video: false,
            single_camera: true,
            camera_model: "SIMPLE_RADIAL".into(),
            init_min_tri_angle: 16.0,
            mapper_seed: env_parse("SPLAT_SFM_MAPPER_SEED").unwrap_or(0),
            mapper_threads: env_parse("SPLAT_SFM_MAPPER_THREADS"),
            init_pair: None,
            reuse_database: None,
            reuse_matches: false,
            mask_dir: None,
            late_masks: None,
            max_image_size: 1600,
            max_features: 8192,
            first_octave: -1,
            num_threads: 8,
            use_gpu: true,
            sequential_overlap: 10,
            quadratic_overlap: true,
            vocab_tree: None,
            ba_global_ratio: None,
            ba_local_refinements: None,
            lens_last: false,
            default_focal_length_factor: None,
            focal_last: false,
            fixed_focal_factor: None,
            keyframe_stride: 1,
            ba_global_max_iterations: None,
            ba_global_refinements: None,
            ba_global_skip_redundant: false,
            guided_matching: false,
            ransac_max_trials: 10000,
            min_track_length: 3,
            calibrate: true,
            ba_iterations: 3,
            ba_max_iterations: 200,
            holdout_every: 8,
            rung: "sfm",
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.max_image_size > 0 && self.max_features > 0,
            "image size and feature count must be positive"
        );
        ensure!(
            self.num_threads > 0 && self.holdout_every > 0,
            "thread count and holdout stride must be positive"
        );
        ensure!(
            self.sequential_overlap > 0,
            "sequential overlap must be positive"
        );
        ensure!(
            self.ba_iterations > 0 && self.ba_max_iterations > 0,
            "BA iteration counts must be positive"
        );
        ensure!(
            self.ransac_max_trials > 0 && self.min_track_length >= 2,
            "RANSAC trials must be positive and tracks need at least two views"
        );
        ensure!(
            [
                "SIMPLE_PINHOLE",
                "PINHOLE",
                "SIMPLE_RADIAL",
                "RADIAL",
                "OPENCV"
            ]
            .contains(&self.camera_model.as_str()),
            "camera model must support splat-eval reprojection"
        );
        if let Some(path) = &self.vocab_tree {
            ensure!(
                matches!(self.features, Features::Sift),
                "SIFT vocabulary trees cannot retrieve ALIKED descriptors"
            );
            ensure!(
                matches!(self.matching, Matching::Sequential | Matching::Hybrid),
                "vocabulary tree requires sequential matching"
            );
            ensure!(
                path.is_file(),
                "vocabulary tree missing: {}",
                path.display()
            );
        }
        if let Some(path) = &self.mask_dir {
            ensure!(path.is_dir(), "mask directory missing: {}", path.display());
        }
        if let Features::AlikedLightGlue { detector, matcher } = &self.features {
            ensure!(
                detector.is_file() && matcher.is_file(),
                "ALIKED and LightGlue require local ONNX artifacts"
            );
        }
        Ok(())
    }
}

/// A shell-free invocation. Adapters can execute these through their own job runtime.
#[derive(Clone, Debug)]
pub struct StageCommand {
    pub name: String,
    pub program: PathBuf,
    pub args: Vec<OsString>,
}

impl StageCommand {
    fn new(opts: &ReconstructOptions, name: &str) -> Self {
        Self {
            name: name.into(),
            program: opts.colmap.clone(),
            args: vec![name.into()],
        }
    }

    fn option(&mut self, key: &str, value: impl Into<OsString>) {
        self.args.push(format!("--{key}").into());
        self.args.push(value.into());
    }

    fn number(&mut self, key: &str, value: impl std::fmt::Display) {
        self.option(key, value.to_string());
    }
}

/// Builds commands without creating files. Paths can contain spaces; no shell is involved.
pub fn commands(frames_dir: &Path, opts: &ReconstructOptions) -> Result<Vec<StageCommand>> {
    opts.validate()?;
    let database = opts.output_dir.join("database.db");
    let mut extract = StageCommand::new(opts, "feature_extractor");
    extract.option("database_path", &database);
    extract.option("image_path", frames_dir);
    extract.option("ImageReader.camera_model", &opts.camera_model);
    if let Some(path) = &opts.mask_dir {
        extract.option("ImageReader.mask_path", path);
    }
    extract.number("ImageReader.single_camera", u8::from(opts.single_camera));
    if let Some(factor) = opts.fixed_focal_factor.or(opts.default_focal_length_factor) {
        extract.number("ImageReader.default_focal_length_factor", factor);
    }
    extract.number("FeatureExtraction.use_gpu", u8::from(opts.use_gpu));
    extract.number("FeatureExtraction.num_threads", opts.num_threads);
    extract.number("FeatureExtraction.max_image_size", opts.max_image_size);
    extract.number("SiftExtraction.max_num_features", opts.max_features);
    extract.number("SiftExtraction.first_octave", opts.first_octave);
    if let Features::AlikedLightGlue { detector, .. } = &opts.features {
        extract.option("FeatureExtraction.type", "ALIKED_N16ROT");
        extract.option("AlikedExtraction.n16rot_model_path", detector);
        extract.number("AlikedExtraction.max_num_features", opts.max_features);
    }

    let mut matching = StageCommand::new(
        opts,
        match opts.matching {
            Matching::Exhaustive => "exhaustive_matcher",
            Matching::Sequential | Matching::Hybrid => "sequential_matcher",
        },
    );
    matching.option("database_path", &database);
    if let Features::AlikedLightGlue { matcher, .. } = &opts.features {
        matching.option("FeatureMatching.type", "ALIKED_LIGHTGLUE");
        matching.option("AlikedMatching.lightglue_model_path", matcher);
    }
    matching.number("FeatureMatching.use_gpu", u8::from(opts.use_gpu));
    matching.number("FeatureMatching.num_threads", opts.num_threads);
    matching.number(
        "FeatureMatching.guided_matching",
        u8::from(opts.guided_matching),
    );
    matching.number("TwoViewGeometry.random_seed", 0);
    matching.number("TwoViewGeometry.max_num_trials", opts.ransac_max_trials);
    if matches!(opts.matching, Matching::Sequential | Matching::Hybrid) {
        matching.number("SequentialMatching.overlap", opts.sequential_overlap);
        matching.number(
            "SequentialMatching.quadratic_overlap",
            u8::from(opts.quadratic_overlap),
        );
        matching.number(
            "SequentialMatching.loop_detection",
            u8::from(opts.vocab_tree.is_some()),
        );
        if let Some(tree) = &opts.vocab_tree {
            matching.option("SequentialMatching.vocab_tree_path", tree);
            // Loop retrieval indexes each image's 2048 strongest features, not all ~10k: indexing is
            // single-threaded and was 10 of the driveway clip's 14 s of matching. Driveway 18 -> 9 s, the same
            // 1508 verified pairs.
            matching.number("SequentialMatching.loop_detection_max_num_features", 2048);
        }
    }
    // Sequential matching verifies each image's pairs as one small batch and waits for its slowest
    // pair (2.5 of 6 cores busy, GPU 3%). Match only, then verify every pair in large batches across
    // all cores: driveway (1508 pairs) 62 -> 16 + 22 s, the same verified pairs and inliers.
    let verify = matches!(opts.matching, Matching::Sequential | Matching::Hybrid)
        .then(|| {
            matching.number("FeatureMatching.skip_geometric_verification", 1);
            let mut verify = StageCommand::new(opts, "geometric_verifier");
            verify.option("database_path", &database);
            verify.number("num_threads", opts.num_threads);
            verify.number("TwoViewGeometry.random_seed", 0);
            verify.number("TwoViewGeometry.max_num_trials", opts.ransac_max_trials);
            verify
        });
    let mut stages = vec![extract, matching];
    stages.extend(verify);
    // A fixed focal skips the view-graph calibration, which would re-estimate it.
    if opts.calibrate && matches!(opts.mapper, Mapper::Global) && opts.fixed_focal_factor.is_none() {
        let mut calibrate = StageCommand::new(opts, "view_graph_calibrator");
        calibrate.option("database_path", &database);
        stages.push(calibrate);
    }
    let mut mapper = StageCommand::new(
        opts,
        match opts.mapper {
            Mapper::Global => "global_mapper",
            Mapper::Incremental => "mapper",
        },
    );
    mapper.option("database_path", database);
    mapper.option("image_path", frames_dir);
    mapper.option("output_path", opts.output_dir.join("sparse"));
    match opts.mapper {
        Mapper::Global => {
            mapper.number("GlobalMapper.num_threads", opts.num_threads);
            mapper.number("GlobalMapper.random_seed", opts.mapper_seed);
            // CPU global positioning and BA (the available COLMAP builds lack GPU bundle
            // adjustment); stage_needs_gpu_lock keeps this stage off the GPU queue.
            mapper.number("GlobalMapper.gp_use_gpu", 0);
            mapper.number("GlobalMapper.ba_ceres_use_gpu", 0);
            mapper.number("GlobalMapper.ba_num_iterations", opts.ba_iterations);
            // A fixed focal stays fixed through the global BA too. Distortion is still refined
            // while mapping (lens_last's fixed distortion is incremental-only); the lens BA after
            // mapping refines it again.
            if (opts.lens_last && opts.focal_last) || opts.fixed_focal_factor.is_some() {
                mapper.number("GlobalMapper.ba_refine_focal_length", 0);
            }
            mapper.number(
                "GlobalMapper.track_min_num_views_per_track",
                opts.min_track_length,
            );
            mapper.number("GlobalMapper.ba_min_track_length", opts.min_track_length);
            mapper.number(
                "GlobalMapper.ba_ceres_max_num_iterations",
                opts.ba_max_iterations,
            );
        }
        Mapper::Incremental => {
            mapper.number("Mapper.num_threads", keyframe_mapper_threads(opts));
            mapper.number("Mapper.random_seed", opts.mapper_seed);
            mapper.number("Mapper.ba_use_gpu", u8::from(opts.use_gpu));
            mapper.number("Mapper.init_min_tri_angle", opts.init_min_tri_angle);
            if let Some((a, b)) = opts.init_pair {
                mapper.number("Mapper.init_image_id1", a);
                mapper.number("Mapper.init_image_id2", b);
            }
            if let Some(ratio) = opts.ba_global_ratio {
                mapper.number("Mapper.ba_global_frames_ratio", ratio);
                mapper.number("Mapper.ba_global_points_ratio", ratio);
            }
            if let Some(n) = opts.ba_local_refinements {
                mapper.number("Mapper.ba_local_max_refinements", n);
            }
            if opts.lens_last {
                mapper.number("Mapper.ba_refine_extra_params", 0);
            }
            if (opts.lens_last && opts.focal_last) || opts.fixed_focal_factor.is_some() {
                mapper.number("Mapper.ba_refine_focal_length", 0);
            }
            if opts.keyframe_stride > 1 {
                mapper.option("Mapper.image_list_path", opts.output_dir.join("keyframes.txt"));
            }
            if let Some(n) = opts.ba_global_max_iterations {
                mapper.number("Mapper.ba_global_max_num_iterations", n);
            }
            if let Some(n) = opts.ba_global_refinements {
                mapper.number("Mapper.ba_global_max_refinements", n);
            }
            if opts.ba_global_skip_redundant {
                mapper.number("Mapper.ba_global_ignore_redundant_points3D", 1);
            }
        }
    }
    stages.push(mapper);
    // 4.2.1 uses log_target, not older glog flags.
    for stage in &mut stages {
        stage.option("log_target", "stderr");
    }
    Ok(stages)
}

#[derive(Debug, Serialize, serde::Deserialize)]
pub struct StageTiming {
    pub name: String,
    pub seconds: f64,
    pub argv: Vec<String>,
    pub log: PathBuf,
    pub gpu_lock_held: bool,
    /// The COLMAP child's CPU seconds (user + system, all threads; Linux, see [`wait_exited`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_seconds: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Reconstruction {
    pub model_dir: PathBuf,
    pub health: SfmHealth,
    pub components: ComponentInventory,
    pub wall_seconds: f64,
    pub lock_wait_seconds: f64,
    pub stages: Vec<StageTiming>,
    pub colmap_version: String,
    /// Performance/provider settings inherited by COLMAP (not the full process environment).
    pub environment: BTreeMap<String, String>,
    /// The ladder rung whose model this is ("hybrid", "exhaustive", "serial-remap",
    /// "flip-retry/hybrid", ...; callers append their own: "+drop-flipped", "learned-union").
    pub rung: String,
}

/// An attempt's database as every later rung reads it (see [`freeze_database`]).
pub const FROZEN_DB: &str = "frozen.db";

/// The database a rung after the attempt in `dir` maps: its frozen copy, or, for an attempt that
/// stopped before mapping (none frozen), its working database as before.
pub fn frozen_database(dir: &Path) -> PathBuf {
    Some(dir.join(FROZEN_DB)).filter(|p| p.is_file()).unwrap_or_else(|| dir.join("database.db"))
}

/// The SHA-256 recorded when `database` was frozen (None if it is not a frozen database).
fn frozen_sha(database: &Path) -> Option<String> {
    if database.file_name()? != FROZEN_DB {
        return None;
    }
    let record: serde_json::Value = serde_json::from_slice(&fs::read(database.with_extension("json")).ok()?).ok()?;
    record["sha256"].as_str().map(str::to_owned)
}

/// Copies `src` to a new `dst` (default permissions, not `src`'s) and returns the SHA-256 of the
/// bytes written, in one pass.
fn copy_hashed(src: &Path, dst: &Path) -> Result<String> {
    use sha2::Digest;
    let mut input = File::open(src).with_context(|| format!("reading {}", src.display()))?;
    let mut output = File::create(dst).with_context(|| format!("writing {}", dst.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        output.write_all(&buf[..n])?;
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Freezes the attempt's database in `dir` once its last content write (`after`, the late-mask
/// filter on video) is done: a read-only `frozen.db` beside it, its SHA-256 in `frozen.json`, from
/// which every later rung copies (the serial remap, the flip retry, the low-angle and exhaustive
/// fallbacks, the learned rescue's seed). 2026-10-12 CPU probe (COLMAP 4.2.1, 34 frames, the hybrid
/// attempt's stages after the mask filter: keyframe mapper, image_registrator, continuation and
/// leftover mappers, point_triangulator, lens BA): every table's content stayed identical, but each
/// COLMAP open rewrites the header (bytes 24-27 and 92-95, its write counters, and 96-99, the
/// version of the SQLite that last wrote it: rusqlite's 3.53.2 after the mask filter, COLMAP's
/// 3.53.4 after the mapper). That stamp is what the exhaustive prefetch's counter-masked hash missed
/// (Austin, 2026-10-12: "the hybrid database changed"). A frozen copy is never opened again, so a
/// plain SHA-256 identifies it.
fn freeze_database(dir: &Path, rung: &str, after: &str) -> Result<()> {
    let start = Instant::now();
    let frozen = dir.join(FROZEN_DB);
    let sha = copy_hashed(&dir.join("database.db"), &frozen)?;
    let mut permissions = fs::metadata(&frozen)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&frozen, permissions)?;
    let seconds = start.elapsed().as_secs_f64();
    fs::write(frozen.with_extension("json"), serde_json::to_vec_pretty(&serde_json::json!({
        "sha256": sha, "rung": rung, "after": after, "seconds": seconds,
    }))?)?;
    eprintln!("[sfm] frozen {rung} DB at {after} (sha {}, {seconds:.1} s)", &sha[..16]);
    Ok(())
}

thread_local! {
    /// CPU seconds of the COLMAP children this thread has run (see [`wait_exited`]).
    static CHILD_CPU: Cell<f64> = const { Cell::new(0.0) };
}

/// Waits until `child` exits, leaving it unreaped, and returns its CPU time (user + system over
/// all its threads) from /proc/<pid>/stat while it is a zombie: per child, not summed over the
/// whole process like getrusage(RUSAGE_CHILDREN).
fn wait_exited(child: &mut Child) -> Result<Option<f64>> {
    #[cfg(target_os = "linux")]
    loop {
        // SAFETY: a zeroed siginfo_t is valid; waitid fills it for our own child, and WNOWAIT leaves
        // the child for Child::wait to reap.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let flags = libc::WEXITED | libc::WNOWAIT;
        if unsafe { libc::waitid(libc::P_PID, child.id() as libc::id_t, &mut info, flags) } < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err.into());
        }
        // SAFETY: si_pid is set by waitid (0 with WNOHANG while the child runs).
        if unsafe { info.si_pid() } != 0 {
            let stat = fs::read_to_string(format!("/proc/{}/stat", child.id())).ok();
            // Fields after the parenthesised command: state is field 3, utime 14, stime 15.
            let ticks = stat.as_deref().and_then(|s| {
                let fields: Vec<&str> = s.rsplit_once(')')?.1.split_whitespace().collect();
                Some(fields.get(11)?.parse::<f64>().ok()? + fields.get(12)?.parse::<f64>().ok()?)
            });
            // SAFETY: sysconf has no preconditions.
            let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
            return Ok(ticks.filter(|_| hz > 0.0).map(|t| t / hz));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    #[cfg(not(target_os = "linux"))]
    loop {
        if child.try_wait()?.is_some() {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Runs a fresh reconstruction and writes options.json, stage logs and reconstruction.json.
/// A failed run is retained for diagnosis; its directory is never silently reused.
/// Initial-pair angle (deg) of the Hybrid retry after a collapsed sequential attempt.
const LOW_INIT_ANGLE: f64 = 4.0;
/// Exhaustive matching is quadratic: 600 frames took 1236 s on copped-hall-first-floor. Above this
/// many frames a hybrid attempt that registered under 95% is kept as it is (the caller's gates and
/// leftover pass deal with the rest) instead of falling back to exhaustive matching.
const EXHAUSTIVE_FALLBACK_MAX_FRAMES: usize = 300;
/// Default SfM fallback budget (SPLAT_SFM_BUDGET_S): 25 min of the job's 55 min box.
const DEFAULT_BUDGET_S: u64 = 1500;

/// The deadline for a job that starts now (SPLAT_SFM_BUDGET_S seconds, 0 = none).
pub fn budget_deadline() -> Option<Instant> {
    let s = env_parse::<u64>("SPLAT_SFM_BUDGET_S").unwrap_or(DEFAULT_BUDGET_S);
    (s > 0).then(|| Instant::now() + std::time::Duration::from_secs(s))
}

/// Union of COLMAP's model with a learned (DA3-seeded) model of the same database: COLMAP's poses stay as
/// they are; COLMAP's model_merger aligns the learned model to them on the shared frames and adds only
/// the frames COLMAP left out, the database's matches are re-triangulated with every pose fixed, and an
/// added frame that observes fewer than `min_obs` 3D points is removed. A replacement is not safe in the
/// half-to-90% band (paf-museum: DA3 placed 220/220 vs COLMAP's 104 but rendered -1.9 dB); a union
/// cannot move the frames COLMAP placed. `opts.output_dir` holds the database. Returns the finalized
/// model, or None when no frame was added.
pub fn union_learned(frames_dir: &Path, opts: &ReconstructOptions, colmap_model: &Path, learned_model: &Path, out: &Path, min_obs: usize)
    -> Result<Option<(PathBuf, SfmHealth, ComponentInventory)>>
{
    let before = read_model(colmap_model)?;
    let _ = fs::remove_dir_all(out);
    fs::create_dir_all(out)?;
    let added = union::append_learned(&opts.colmap, opts, colmap_model, learned_model, out)?;
    if added.is_empty() {
        return Ok(None);
    }
    // The append must leave COLMAP's frames bit for bit (TXT round trip prints full precision).
    let merged = read_model(&out.join("merged"))?;
    for (id, im) in &before.images {
        let m = merged.images.get(id).context("the union dropped a COLMAP frame")?;
        ensure!(im.qvec.iter().zip(&m.qvec).chain(im.tvec.iter().zip(&m.tvec)).all(|(x, y)| (x - y).abs() < 1e-9),
            "the union moved COLMAP frame {}", im.name);
    }
    let tri = out.join("sparse").join("0");
    fs::create_dir_all(&tri)?;
    run_stage(triangulation_command(frames_dir, &out.join("merged"), &tri, opts)?, &opts.output_dir, None)?;
    let after = read_model(&tri)?;
    let old: BTreeSet<&str> = before.images.values().map(|im| im.name.as_str()).collect();
    let weak: Vec<(String, usize)> = after.images.values().filter(|im| !old.contains(im.name.as_str()))
        .map(|im| (im.name.clone(), im.obs_point_id.len())).filter(|(_, n)| *n < min_obs).collect();
    if !weak.is_empty() {
        eprintln!("[sfm] learned union: dropping weakly observed frames {weak:?}");
        let names = out.join("weak.txt");
        fs::write(&names, weak.iter().map(|(n, _)| format!("{n}\n")).collect::<String>())?;
        let kept = out.join("sparse").join("1");
        fs::create_dir_all(&kept)?;
        let mut del = StageCommand::new(opts, "image_deleter");
        del.option("input_path", &tri);
        del.option("output_path", &kept);
        del.option("image_names_path", &names);
        del.option("log_target", "stderr");
        run_stage(del, &opts.output_dir, None)?;
        fs::remove_dir_all(&tri)?;
        fs::rename(&kept, &tri)?;
    }
    let (model_dir, health) = finalize_model(&out.join("sparse"), frames_dir, opts.video, opts.holdout_every)?;
    if health.n_registered <= before.images.len() {
        return Ok(None);
    }
    eprintln!("[sfm] learned union: {} -> {} frames placed (COLMAP's poses unchanged)", before.images.len(), health.n_registered);
    Ok(Some((model_dir, health, component_inventory(&out.join("sparse"))?)))
}

/// Whether the SfM budget is spent (callers deciding on their own retries); logs `what` is skipped.
pub fn deadline_passed(opts: &ReconstructOptions, what: &str) -> bool {
    past_deadline(opts, what)
}

/// Best-so-far for a video trajectory that failed pose continuity: the model just written to
/// `opts.output_dir` (its reconstruction.json and model) minus the frames the continuity check flagged
/// (they are the minority side of each flip), re-finalized. Used when the budget leaves no time for
/// another mapping, or a remap flipped again, so the run ends partial or succeeded instead of failing.
pub fn without_flipped(frames_dir: &Path, opts: &ReconstructOptions) -> Result<Reconstruction> {
    let dir = &opts.output_dir;
    let rec: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("reconstruction.json"))?)?;
    let flipped: Vec<String> = rec["health"]["pose_continuity"]["flipped_frames"].as_array().into_iter().flatten()
        .filter_map(|v| v.as_str().map(str::to_owned)).collect();
    let registered = rec["health"]["n_registered"].as_u64().unwrap_or(0) as usize;
    ensure!(!flipped.is_empty() && flipped.len() * 2 < registered,
        "{} of {registered} frames flipped: no majority trajectory to keep", flipped.len());
    let model = PathBuf::from(rec["model_dir"].as_str().context("reconstruction.json has no model_dir")?);
    let out = dir.join("unflipped");
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(out.join("0"))?;
    let names = dir.join("flipped_frames.txt");
    fs::write(&names, flipped.iter().map(|n| format!("{n}\n")).collect::<String>())?;
    let mut stage = StageCommand::new(opts, "image_deleter");
    stage.name = "drop_flipped_frames".into();
    stage.option("input_path", &model);
    stage.option("output_path", out.join("0"));
    stage.option("image_names_path", &names);
    stage.option("log_target", "stderr");
    let timing = run_stage(stage, dir, None)?;
    eprintln!("[sfm] keeping the trajectory without its {} flipped frame(s) of {registered}", flipped.len());
    let (model_dir, health) = finalize_model(&out, frames_dir, opts.video, opts.holdout_every)?;
    let result = Reconstruction {
        model_dir,
        health,
        components: component_inventory(&out)?,
        wall_seconds: rec["wall_seconds"].as_f64().unwrap_or(0.0) + timing.seconds,
        lock_wait_seconds: rec["lock_wait_seconds"].as_f64().unwrap_or(0.0),
        // Every attempt's stages so far, then the drop: each attempt's duration stays in the record.
        stages: serde_json::from_value::<Vec<StageTiming>>(rec["stages"].clone()).unwrap_or_default().into_iter().chain([timing]).collect(),
        colmap_version: rec["colmap_version"].as_str().unwrap_or_default().to_owned(),
        environment: BTreeMap::new(),
        rung: format!("{}+drop-flipped", rec["rung"].as_str().unwrap_or(opts.rung)),
    };
    fs::write(dir.join("reconstruction.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}

fn past_deadline(opts: &ReconstructOptions, what: &str) -> bool {
    let past = opts.deadline.is_some_and(|d| Instant::now() >= d);
    if past {
        eprintln!("[sfm] SfM budget spent: skipping {what}, keeping the best attempt so far");
    }
    past
}

pub fn reconstruct(frames_dir: &Path, opts: &ReconstructOptions) -> Result<Reconstruction> {
    let (rec, rung) = reconstruct_attempts(frames_dir, opts);
    let rec = rec?;
    if rec.health.pass || !matches!(opts.mapper, Mapper::Incremental) || past_deadline(opts, "the serial mapper retry") {
        return Ok(rec);
    }
    // A video chain that ended on its hybrid attempt maps once more only to re-roll the seed (its
    // keyframe mapper already runs one thread, see [`serial_mapper_retry`]): bench5 (22 jobs at 5 fps,
    // sfm-ladder-plan.md 2026-10-12) ran it on 11 such chains, median 119 s (p90 266 s, max 388 s),
    // and 0 of 11 passed the gates; its 6 wins were points-only tie-breaks or +1 frame.
    if matches!(opts.matching, Matching::Hybrid) && opts.video && rung.ends_with("hybrid") && keyframe_mapper_threads(opts) == 1 {
        eprintln!("[sfm] gates failed ({}); no serial remap after the hybrid attempt (a seed re-roll at one thread)", failed_gates(&rec.health));
        return Ok(rec);
    }
    serial_mapper_retry(frames_dir, opts, rec)
}

/// The ladder's rung names; a ladder on a reused database (the caller's flip retry) prefixes its own.
fn rung_name(opts: &ReconstructOptions, rung: &'static str) -> &'static str {
    let flip = matches!(opts.matching, Matching::Hybrid) && opts.reuse_database.is_some();
    match (flip, rung) {
        (false, rung) => rung,
        (true, "hybrid") => "flip-retry/hybrid",
        (true, "low-angle") => "flip-retry/low-angle",
        (true, "exhaustive") => "flip-retry/exhaustive",
        (true, "serial-remap") => "flip-retry/serial-remap",
        (true, _) => "flip-retry",
    }
}

/// Runs one rung on this thread and logs its wall time and its COLMAP children's CPU time (user +
/// system, see [`wait_exited`]).
fn run_rung(name: &str, f: impl FnOnce() -> Result<Reconstruction>) -> Result<Reconstruction> {
    let (start, cpu) = (Instant::now(), CHILD_CPU.get());
    let rec = f();
    eprintln!("[sfm] rung {name}: wall {:.1} s, cpu {:.1} s{}", start.elapsed().as_secs_f64(), CHILD_CPU.get() - cpu,
        if rec.is_err() { " (no model kept)" } else { "" });
    rec
}

/// The serial remap's options for `opts` (the ladder's), mapping the database `database`.
fn serial_options(opts: &ReconstructOptions, database: &Path) -> ReconstructOptions {
    let mut serial = opts.clone();
    if keyframe_mapper_threads(opts) == 1 {
        serial.mapper_seed = opts.mapper_seed + 1;
    }
    serial.mapper_threads = Some(1);
    serial.init_pair = None;
    serial.reuse_database = Some(database.to_path_buf());
    serial.reuse_matches = false;
    serial.rung = rung_name(opts, "serial-remap");
    serial
}

/// COLMAP 4.2.1's initial-pair search races at Mapper.num_threads > 1: one room database with seed
/// 0 picked #4/#1, #8/#23 and #18/#27 on one host, and #19/#66 (forced: 12k points instead of 22k,
/// 0.81 px, mapper >200 s instead of 17 s) failed the gates in 2 of 13 bench runs. One thread is
/// deterministic (#34/#14 twice, identical points3D.bin, good) and room's mapper +1.2 s. On a gate
/// failure, map the same database once more with one thread and keep the better model; a first
/// attempt that already ran one thread (the photo default) retries with the next seed instead.
fn serial_mapper_retry(frames_dir: &Path, opts: &ReconstructOptions, first: Reconstruction) -> Result<Reconstruction> {
    let raced = opts.output_dir.with_extension("raced");
    let _ = fs::remove_dir_all(&raced);
    fs::rename(&opts.output_dir, &raced)?;
    let serial = serial_options(opts, &frozen_database(&raced));
    eprintln!("[sfm] gates failed ({}); mapping the same matches with one mapper thread, seed {}", failed_gates(&first.health), serial.mapper_seed);
    let start = Instant::now();
    let retried = run_rung(serial.rung, || reconstruct_once(frames_dir, &serial));
    let seconds = start.elapsed().as_secs_f64();
    match retried {
        Ok(rec) if model_rank(&rec.health) > model_rank(&first.health) => {
            prune_attempt(&raced);
            with_earlier_attempts(&opts.output_dir, Ok(rec), &[("raced_mapper_attempt", first.wall_seconds, &raced)])
        }
        other => {
            if let Err(e) = &other {
                eprintln!("[sfm] serial mapper retry failed: {e:#}");
            }
            // Keep the first model where callers expect it; the retry's logs stay beside it.
            let serial_dir = opts.output_dir.with_extension("serial");
            let _ = fs::remove_dir_all(&serial_dir);
            if opts.output_dir.exists() {
                fs::rename(&opts.output_dir, &serial_dir)?;
                prune_attempt(&serial_dir);
            }
            fs::rename(&raced, &opts.output_dir)?;
            let mut rec = first;
            rec.stages.push(StageTiming {
                name: "serial_mapper_attempt".into(),
                seconds,
                argv: Vec::new(),
                log: serial_dir,
                gpu_lock_held: false,
                cpu_seconds: None,
            });
            rec.wall_seconds += seconds;
            fs::write(opts.output_dir.join("reconstruction.json"), serde_json::to_vec_pretty(&rec)?)?;
            Ok(rec)
        }
    }
}

/// Keep a losing attempt's logs and JSON only (its database and models are hundreds of MB).
fn prune_attempt(dir: &Path) {
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        let _ = if path.is_dir() {
            fs::remove_dir_all(&path)
        } else if path.extension().is_some_and(|e| e == "db" || e == "ply") {
            fs::remove_file(&path)
        } else {
            Ok(())
        };
    }
}

fn failed_gates(health: &SfmHealth) -> String {
    health.gates.iter().filter(|(_, ok)| !**ok).map(|(g, _)| g.as_str()).collect::<Vec<_>>().join(", ")
}

/// Passing models first, then fewer failed gates, more registered images, more points, lower error.
fn model_rank(h: &SfmHealth) -> (bool, std::cmp::Reverse<usize>, usize, usize, std::cmp::Reverse<u64>) {
    let failed = h.gates.values().filter(|ok| !**ok).count();
    (h.pass, std::cmp::Reverse(failed), h.n_registered, h.n_points, std::cmp::Reverse(h.reproj_rank()))
}

/// Whether a COLMAP stage takes the shared exclusive GPU lock when `use_gpu` is set: CUDA feature
/// extraction and matching do; both mappers run predominantly (global: entirely) on CPU.
pub fn stage_needs_gpu_lock(stage: &str) -> bool {
    matches!(stage, "feature_extractor" | "exhaustive_matcher" | "sequential_matcher")
}

/// `opts` for one attempt of the ladder whose options are `base`, named `rung`.
fn attempt_options(base: &ReconstructOptions, opts: ReconstructOptions, rung: &'static str) -> ReconstructOptions {
    ReconstructOptions { rung: rung_name(base, rung), ..opts }
}

/// The ladder's rungs before the serial remap; returns the last attempt's result and rung name.
fn reconstruct_attempts(frames_dir: &Path, opts: &ReconstructOptions) -> (Result<Reconstruction>, &'static str) {
    if !matches!(opts.matching, Matching::Hybrid) {
        return (run_rung(opts.rung, || reconstruct_once(frames_dir, opts)), opts.rung);
    }
    match hybrid_attempts(frames_dir, opts) {
        Ok(v) => v,
        Err(e) => (Err(e), "hybrid"),
    }
}

fn hybrid_attempts(frames_dir: &Path, opts: &ReconstructOptions) -> Result<(Result<Reconstruction>, &'static str)> {
    let started = Instant::now();
    let first = {
        let first_opts = attempt_options(opts, opts.clone(), "hybrid");
        run_rung(first_opts.rung, || reconstruct_once(frames_dir, &first_opts))
    };
    let first_seconds = started.elapsed().as_secs_f64();
    eprintln!("[sfm] hybrid attempt: {first_seconds:.0} s ({})", if first.is_ok() { "ok" } else { "failed" });
    // A flip with full registration is the caller's (retry) business; anything under 95% falls
    // back. Keeping partial models (>= half) let night-stroll ship 102/150 and fail its gates.
    let registered = fs::read(opts.output_dir.join("reconstruction.json")).ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|r| r["health"]["registered_frac"].as_f64());
    if registered.is_some_and(|f| f >= 0.95) {
        return Ok((first, "hybrid"));
    }
    // Large sets never fall back to quadratic exhaustive matching, and a spent budget starts no
    // fallback: a usable hybrid model (>= half the frames) is kept as it is.
    let large = frame_names(frames_dir)?.len() > EXHAUSTIVE_FALLBACK_MAX_FRAMES;
    if registered.is_some_and(|f| f >= 0.5) && (large || past_deadline(opts, "the exhaustive fallback")) {
        if large {
            eprintln!("[sfm] over {EXHAUSTIVE_FALLBACK_MAX_FRAMES} frames: keeping the hybrid attempt instead of an exhaustive fallback");
        }
        return Ok((first, "hybrid"));
    }
    let sequential = opts.output_dir.with_extension("sequential");
    let _ = fs::remove_dir_all(&sequential);
    fs::rename(&opts.output_dir, &sequential)?;
    // Every fallback maps a copy of the hybrid attempt's frozen database.
    let hybrid_db = frozen_database(&sequential);
    // A collapse may only mean no initial pair reaches the angle among nearby frames (driveway, a
    // slow slide: none at 16 deg, so exhaustive spent 4.5 min in failed pair searches and
    // SfM ran >18 min). Map the same matches from a 4 deg pair first. Not the default: at 4 deg
    // night-stroll starts from frames 4 apart and fragments (mapper 74 -> 335 s).
    let started_low = fs::read(sequential.join("motion.json")).ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|m| m["slow_translation"].as_bool() == Some(true));
    if registered.is_none_or(|f| f < 0.5) && opts.init_min_tri_angle > LOW_INIT_ANGLE && !started_low
        && !past_deadline(opts, "the low-angle retry")
    {
        let mut low = opts.clone();
        // The initial-pair angle is an incremental option: a failed global solve retries
        // incrementally, as every video solve did before the global default.
        low.mapper = Mapper::Incremental;
        low.init_min_tri_angle = LOW_INIT_ANGLE;
        low.reuse_database = Some(hybrid_db.clone());
        let low = attempt_options(opts, low, "low-angle");
        let low_start = Instant::now();
        let retried = run_rung(low.rung, || reconstruct_once(frames_dir, &low));
        let low_seconds = low_start.elapsed().as_secs_f64();
        let registered = fs::read(opts.output_dir.join("reconstruction.json")).ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|r| r["health"]["registered_frac"].as_f64());
        if registered.is_some_and(|f| f >= 0.95) {
            return Ok((with_earlier_attempts(&opts.output_dir, retried, &[("collapsed_attempt", first_seconds, &sequential)]), "low-angle"));
        }
        if retried.is_ok() && (large || past_deadline(opts, "the exhaustive fallback")) {
            // No exhaustive fallback for this set: the low-angle model is the best attempt.
            return Ok((with_earlier_attempts(&opts.output_dir, retried, &[("collapsed_attempt", first_seconds, &sequential)]), "low-angle"));
        }
        let low_dir = opts.output_dir.with_extension("low-angle");
        let _ = fs::remove_dir_all(&low_dir);
        fs::rename(&opts.output_dir, &low_dir)?;
        let exhaustive = exhaustive_fallback(frames_dir, opts, &hybrid_db);
        return Ok((with_earlier_attempts(
            &opts.output_dir,
            exhaustive,
            &[("collapsed_attempt", first_seconds, &sequential), ("low_angle_attempt", low_seconds, &low_dir)],
        ), "exhaustive"));
    }
    let exhaustive = exhaustive_fallback(frames_dir, opts, &hybrid_db);
    Ok((with_earlier_attempts(&opts.output_dir, exhaustive, &[("sequential_attempt", first_seconds, &sequential)]), "exhaustive"))
}

fn exhaustive_fallback(frames_dir: &Path, opts: &ReconstructOptions, database: &Path) -> Result<Reconstruction> {
    let mut exhaustive = opts.clone();
    exhaustive.matching = Matching::Exhaustive;
    exhaustive.vocab_tree = None;
    exhaustive.reuse_database = Some(database.to_path_buf());
    exhaustive.reuse_matches = true;
    let exhaustive = attempt_options(opts, exhaustive, "exhaustive");
    run_rung(exhaustive.rung, || reconstruct_once(frames_dir, &exhaustive))
}

/// Report the failed Hybrid attempts that preceded `rec` as its first stages, and count their time
/// in its wall time (a driveway run otherwise hid ~3 min of extraction, matching and a failed mapper).
fn with_earlier_attempts(output_dir: &Path, rec: Result<Reconstruction>, earlier: &[(&str, f64, &Path)]) -> Result<Reconstruction> {
    let mut rec = rec?;
    for (name, seconds, dir) in earlier.iter().rev() {
        rec.stages.insert(0, StageTiming {
            name: (*name).into(),
            seconds: *seconds,
            argv: Vec::new(),
            log: dir.to_path_buf(),
            gpu_lock_held: false,
            cpu_seconds: None,
        });
        rec.wall_seconds += seconds;
    }
    fs::write(output_dir.join("reconstruction.json"), serde_json::to_vec_pretty(&rec)?)?;
    Ok(rec)
}

fn reconstruct_once(frames_dir: &Path, opts: &ReconstructOptions) -> Result<Reconstruction> {
    let start = Instant::now();
    opts.validate()?;
    ensure!(
        frames_dir.is_dir(),
        "frames directory missing: {}",
        frames_dir.display()
    );
    let version = Command::new(&opts.colmap)
        .arg("-h")
        .output()
        .context("querying COLMAP version")?;
    ensure!(version.status.success(), "COLMAP version probe failed");
    let version = String::from_utf8_lossy(&version.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    ensure!(
        version.starts_with("COLMAP 4.2.1 "),
        "expected COLMAP 4.2.1, got {version:?}"
    );
    if let Some(parent) = opts
        .output_dir
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&opts.output_dir)
        .with_context(|| format!("output must be fresh: {}", opts.output_dir.display()))?;
    fs::create_dir(opts.output_dir.join("sparse"))?;
    fs::write(
        opts.output_dir.join("options.json"),
        serde_json::to_vec_pretty(opts)?,
    )?;
    let keyframes = opts.keyframe_stride > 1 && matches!(opts.mapper, Mapper::Incremental);
    if keyframes {
        write_keyframes(frames_dir, opts.keyframe_stride, &opts.output_dir.join("keyframes.txt"))?;
    }
    let mut lock_wait_seconds = 0.0;
    let mut timings = Vec::new();
    let mut stages = commands(frames_dir, opts)?;
    if let Some(database) = &opts.reuse_database {
        let sha = copy_hashed(database, &opts.output_dir.join("database.db"))?;
        if let Some(frozen) = frozen_sha(database) {
            ensure!(sha == frozen, "the copy of {} does not match its frozen SHA-256 ({sha} vs {frozen})", database.display());
            eprintln!("[sfm] {}: mapping a copy of the frozen DB (sha {}, matches)", opts.rung, &sha[..16]);
        }
        reset_camera(&opts.output_dir.join("database.db"), &opts.camera_model)?;
        if let Some(factor) = opts.fixed_focal_factor {
            set_focal(&opts.output_dir.join("database.db"), factor)?;
        }
        // Exhaustive matching on a reused database only adds the pairs it lacks (Hybrid fallback).
        let rematch = matches!(opts.matching, Matching::Exhaustive) && opts.reuse_matches;
        stages.retain(|s| {
            s.name != "feature_extractor"
                && (rematch || !(s.name.ends_with("_matcher") || s.name == "geometric_verifier"))
        });
    }
    let mut mapper_stage = None;
    for mut stage in stages {
        if matches!(stage.name.as_str(), "mapper" | "global_mapper") {
            if let Some(masks) = &opts.late_masks {
                apply_late_masks(masks, &opts.output_dir, &mut timings)?;
            }
            // The database's content is final here (see [`freeze_database`]).
            let after = timings.last().map_or("the database copy", |t: &StageTiming| t.name.as_str()).to_owned();
            freeze_database(&opts.output_dir, opts.rung, &after)?;
        }
        if stage.name == "mapper" && opts.video && matches!(opts.mapper, Mapper::Incremental)
            && opts.init_min_tri_angle > LOW_INIT_ANGLE
        {
            if let Err(e) = low_motion_init(&mut stage, &opts.output_dir) {
                eprintln!("[sfm] motion check skipped: {e:#}");
            }
        }
        // Mapping is predominantly CPU work: incremental's optional small GPU BA context may
        // coexist with other jobs, and the global mapper runs on CPU. Neither may monopolize the GPU
        // queue.
        let needs_gpu_lock = opts.use_gpu && stage_needs_gpu_lock(&stage.name);
        let lock_start = Instant::now();
        let gpu_lock = opts.gpu_lock.as_ref().filter(|_| needs_gpu_lock)
            .map(|path| -> Result<File> {
                let file = fs::OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .write(true)
                    .open(path)?;
                file.lock_exclusive()?;
                Ok(file)
            })
            .transpose()?;
        if gpu_lock.is_some() {
            lock_wait_seconds += lock_start.elapsed().as_secs_f64();
        }
        if stage.name == "mapper" {
            mapper_stage = Some(stage.clone());
            if keyframes && opts.keyframe_stride > 2 && opts.init_pair.is_none() && forced_keyframe_map(&stage, opts, &mut timings)? {
                continue;
            }
        }
        // Stages before the mapper run beside the late-mask segmenter (four `sam track` processes on
        // a 6-vCPU runner) and finish ~25 s before the masks do, so the masks are the critical path:
        // let them have the CPU first. Priority changes no output.
        let nice = opts
            .late_masks
            .as_ref()
            .filter(|masks| !masks.with_extension("done").is_file())
            .map_or(0, |_| PREMASK_NICE);
        timings.push(run_stage_at(stage, &opts.output_dir, gpu_lock.as_ref(), nice)?);
        // Release after this child exits; the next stage acquires only if it needs it.
    }
    // A keyframe map sparser than every 2nd frame that split: map every 2nd frame instead (the same
    // command and initial-pair settings). Night-soccer's every-3rd map split into 36 + 9 frames; every
    // 2nd frame maps 58 in one piece, which the keyframe lens BA and registration need.
    if keyframes && opts.keyframe_stride > 2 && !keyframe_map_complete(&opts.output_dir)? {
        if let Some(mut stage) = mapper_stage {
            let sparse = opts.output_dir.join("sparse");
            fs::rename(&sparse, opts.output_dir.join(format!("sparse-stride{}", opts.keyframe_stride)))?;
            fs::create_dir(&sparse)?;
            write_keyframes(frames_dir, 2, &opts.output_dir.join("keyframes.txt"))?;
            stage.name = "mapper_stride2".into();
            timings.push(run_stage(stage, &opts.output_dir, None)?);
        }
    }
    let (mapped_dir, _) = finalize_model(
        &opts.output_dir.join("sparse"),
        frames_dir,
        opts.video,
        opts.holdout_every,
    )?;
    let split = opts.output_dir.join("components").is_dir();
    let complete = keyframes && keyframe_map_complete(&opts.output_dir)?;
    // Keyframe map in one piece: the lens BA refines the keyframes' poses, focal length and
    // distortion here, before the other frames are added with that lens, instead of over every frame
    // at the end. Driveway clip (75 keyframes of 150): lens BA 97 -> 26 s. A map that still split keeps the
    // final lens BA over every frame: a lens fitted on one piece would be fixed for all the others.
    // So does `focal_last`, whose focal length only that final BA may refine.
    let lens_on_keyframes = opts.lens_last && complete && !opts.focal_last;
    if lens_on_keyframes {
        let refined = opts.output_dir.join("lens");
        fs::create_dir(&refined)?;
        timings.push(run_stage(lens_command(&mapped_dir, &refined, opts)?, &opts.output_dir, None)?);
        fs::rename(&mapped_dir, opts.output_dir.join("keyframes-unrefined"))?;
        fs::rename(&refined, &mapped_dir)?;
    }
    // Keyframe map: register the other frames against it, pose only (PnP + pose refinement, no BA);
    // the triangulator then adds their points. Driveway clip (74 frames): 3.8 s, 150/150, against 72 s of
    // local bundle adjustments in the continuation mapper, final reprojection error 0.425 vs 0.422 px.
    if complete {
        let registered = opts.output_dir.join("registered");
        fs::create_dir(&registered)?;
        timings.push(run_stage(registration_command(&mapped_dir, &registered, opts)?, &opts.output_dir, None)?);
        let (before, after) = (read_model(&mapped_dir)?, read_model(&registered).ok());
        match after {
            Some(after) if after.images.len() > before.images.len() && same_cameras(&before, &after) => {
                fs::rename(&mapped_dir, opts.output_dir.join("registration-seed"))?;
                fs::rename(&registered, &mapped_dir)?;
            }
            _ => fs::remove_dir_all(&registered)?,
        }
    }
    // A split video trajectory, or frames registration left out: grow the selected component with
    // the remaining frames, keeping its existing poses fixed. Keep the result only if it registers more.
    if split || (keyframes && read_model(&mapped_dir)?.images.len() < frame_names(frames_dir)?.len()) {
        let continued = opts.output_dir.join("continued");
        fs::create_dir(&continued)?;
        timings.push(run_stage(
            continuation_command(frames_dir, &mapped_dir, &continued, opts)?,
            &opts.output_dir,
            None,
        )?);
        let grown = read_model(&continued).map(|m| m.images.len()).unwrap_or(0);
        if grown > read_model(&mapped_dir)?.images.len() {
            fs::rename(&mapped_dir, opts.output_dir.join("continuation-seed"))?;
            fs::rename(&continued, &mapped_dir)?;
        } else {
            fs::remove_dir_all(&continued)?;
        }
    }
    // Frames still left out (no initial link, a component too small to keep, or too few 2D-3D
    // matches for COLMAP's 30-inlier floor to even try them) get one more pass with a laxer floor;
    // every frame it adds must observe enough 3D points. ETH3D pipes (14 photos) lost frames 3, 12
    // and 13 at 1600 and 2000 px but not at 2400 (one weak link decided registration);
    // sai-pixel5-plant's continuation never attempted 18 of 150 frames.
    let mut leftovers_grown = false;
    // Video too (SPLAT_SFM_LEFTOVERS_VIDEO=0 turns it off): sai-pixel5-plant 132 -> 137 of 150 frames
    // (partial -> succeeded), held-out PSNR on the same frozen views 28.50 -> 28.57 dB.
    let leftover_pass = env_parse::<u8>("SPLAT_SFM_LEFTOVERS").unwrap_or(1) == 1
        && (!opts.video || env_parse::<u8>("SPLAT_SFM_LEFTOVERS_VIDEO").unwrap_or(1) == 1);
    if leftover_pass && read_model(&mapped_dir)?.images.len() < frame_names(frames_dir)?.len() {
        let continued = opts.output_dir.join("leftovers");
        fs::create_dir(&continued)?;
        let mut stage = continuation_command(frames_dir, &mapped_dir, &continued, opts)?;
        stage.name = "leftover_mapper".into();
        stage.number("Mapper.abs_pose_min_num_inliers", LEFTOVER_MIN_INLIERS);
        stage.number("Mapper.max_reg_trials", 6);
        timings.push(run_stage(stage, &opts.output_dir, None)?);
        let before = read_model(&mapped_dir)?;
        // Drop the added frames that observe too few 3D points (a pose from a handful of matches),
        // keep the rest: sai-pixel5-plant's 18 leftovers saw 10-29 points each.
        let mut continued = continued;
        if let Ok(after) = read_model(&continued) {
            let weak = weak_new_frames(&before, &after);
            if !weak.is_empty() {
                eprintln!("[sfm] leftover pass: dropping weakly observed frames {weak:?}");
                let names = opts.output_dir.join("leftover-weak.txt");
                fs::write(&names, weak.iter().map(|(n, _)| format!("{n}\n")).collect::<String>())?;
                let kept = opts.output_dir.join("leftovers-kept");
                fs::create_dir(&kept)?;
                let mut stage = StageCommand::new(opts, "image_deleter");
                stage.option("input_path", &continued);
                stage.option("output_path", &kept);
                stage.option("image_names_path", &names);
                stage.option("log_target", "stderr");
                timings.push(run_stage(stage, &opts.output_dir, None)?);
                fs::remove_dir_all(&continued)?;
                continued = kept;
            }
        }
        let grown = read_model(&continued).ok().filter(|m| weak_new_frames(&before, m).is_empty()).map(|m| m.images.len()).unwrap_or(0);
        if grown > before.images.len() {
            leftovers_grown = true;
            fs::rename(&mapped_dir, opts.output_dir.join("leftover-seed"))?;
            fs::rename(&continued, &mapped_dir)?;
        } else {
            fs::remove_dir_all(&continued)?;
        }
    }
    // Estimate cameras first, then retain geometrically filtered two-view tracks.
    // Keep the original selected model for camera audits and initialization A/Bs.
    let init_dir = opts.output_dir.join("init");
    fs::create_dir(&init_dir)?;
    timings.push(run_stage(
        triangulation_command(frames_dir, &mapped_dir, &init_dir, opts)?,
        &opts.output_dir,
        None,
    )?);
    ensure_fixed_cameras(&mapped_dir, &init_dir)?;
    // Photo leftovers were placed with every other pose fixed: one global BA lets the whole model
    // settle around them (pipes at 1600 px: 14/14 either way, the late frames' AUC was the gap).
    // (Video: the final lens BA below already refines every pose.)
    if leftovers_grown && !opts.lens_last && env_parse::<u8>("SPLAT_SFM_LEFTOVER_BA").unwrap_or(1) == 1 {
        let refined = opts.output_dir.join("leftover-ba");
        fs::create_dir(&refined)?;
        let mut stage = StageCommand::new(opts, "bundle_adjuster");
        stage.name = "leftover_bundle_adjuster".into();
        stage.option("input_path", &init_dir);
        stage.option("output_path", &refined);
        stage.number("BundleAdjustment.refine_focal_length", u8::from(opts.fixed_focal_factor.is_none()));
        stage.number("BundleAdjustment.refine_principal_point", 0);
        stage.number("BundleAdjustment.refine_extra_params", 0);
        stage.number("BundleAdjustmentCeres.use_gpu", u8::from(opts.use_gpu));
        stage.option("log_target", "stderr");
        timings.push(run_stage(stage, &opts.output_dir, None)?);
        fs::remove_dir_all(&init_dir)?;
        fs::rename(&refined, &init_dir)?;
    }
    if opts.lens_last && !lens_on_keyframes {
        let refined = opts.output_dir.join("lens");
        fs::create_dir(&refined)?;
        timings.push(run_stage(lens_command(&init_dir, &refined, opts)?, &opts.output_dir, None)?);
        fs::remove_dir_all(&init_dir)?;
        fs::rename(&refined, &init_dir)?;
    }
    let archive = opts.output_dir.join("mapped");
    fs::create_dir(&archive)?;
    fs::rename(&mapped_dir, archive.join("0"))?;
    fs::rename(&init_dir, &mapped_dir)?;
    let (model_dir, health) = best_model(
        &opts.output_dir.join("sparse"),
        frames_dir,
        opts.video,
        opts.holdout_every,
    )?;
    let components = component_inventory(&opts.output_dir.join("sparse"))?;
    let result = Reconstruction {
        model_dir,
        health,
        components,
        wall_seconds: start.elapsed().as_secs_f64() - lock_wait_seconds,
        lock_wait_seconds,
        stages: timings,
        colmap_version: version,
        rung: opts.rung.to_owned(),
        environment: [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
        ]
        .into_iter()
        .filter_map(|key| std::env::var(key).ok().map(|value| (key.to_owned(), value)))
        .collect(),
    };
    fs::write(
        opts.output_dir.join("reconstruction.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    ensure_video_continuity(&result.health, opts.video).with_context(|| {
        format!("reconstruction and diagnostics retained in {}", opts.output_dir.display())
    })?;
    Ok(result)
}

/// Wait for `late_masks` (see [`ReconstructOptions::late_masks`]) and drop the verified matches on
/// masked keypoints from the database; both steps are reported as stages.
fn apply_late_masks(masks: &Path, output_dir: &Path, timings: &mut Vec<StageTiming>) -> Result<()> {
    let done = masks.with_extension("done");
    let start = Instant::now();
    while !done.is_file() {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let stage = |name: &str, seconds: f64| StageTiming {
        name: name.into(),
        seconds,
        argv: Vec::new(),
        log: masks.to_path_buf(),
        gpu_lock_held: false,
        cpu_seconds: None,
    };
    timings.push(stage("late_masks_wait", start.elapsed().as_secs_f64()));
    if fs::read_to_string(&done)?.trim() != "ok" {
        return Ok(());
    }
    let start = Instant::now();
    let (kept, dropped) = drop_masked_matches(&output_dir.join("database.db"), masks)?;
    eprintln!("[sfm] feature masks: dropped {dropped} of {} verified matches", kept + dropped);
    timings.push(stage("mask_filter", start.elapsed().as_secs_f64()));
    Ok(())
}

/// Remove verified (two-view geometry) matches whose keypoint lies under a black pixel of its
/// image's mask (`<masks>/<image name>.png`). Returns (kept, dropped). Repeating it is a no-op.
pub fn drop_masked_matches(database: &Path, masks: &Path) -> Result<(usize, usize)> {
    let mut db = rusqlite::Connection::open(database)?;
    let images: Vec<(i64, String)> = db
        .prepare("SELECT image_id, name FROM images")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut masked: std::collections::HashMap<i64, Vec<bool>> = Default::default();
    for (id, name) in images {
        let path = masks.join(format!("{name}.png"));
        if !path.is_file() {
            continue;
        }
        let mask = image::open(&path).with_context(|| format!("reading {}", path.display()))?.to_luma8();
        let (rows, cols, data): (i64, i64, Vec<u8>) = db.query_row(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, Option<Vec<u8>>>(2)?.unwrap_or_default())),
        )?;
        let coords: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))).collect();
        ensure!(coords.len() as i64 == rows * cols && cols >= 2, "keypoints of image {id} are malformed");
        let flags = coords
            .chunks_exact(cols as usize)
            .map(|k| {
                let (x, y) = (k[0].floor(), k[1].floor());
                x >= 0.0 && y >= 0.0 && mask.get_pixel_checked(x as u32, y as u32).is_some_and(|p| p[0] < 128)
            })
            .collect();
        masked.insert(id, flags);
    }
    let pairs: Vec<(i64, Vec<u8>)> = db
        .prepare("SELECT pair_id, data FROM two_view_geometries WHERE rows > 0")?
        .query_map([], |r| Ok((r.get(0)?, r.get::<_, Option<Vec<u8>>>(1)?.unwrap_or_default())))?
        .collect::<rusqlite::Result<_>>()?;
    let (mut kept, mut dropped) = (0, 0);
    let tx = db.transaction()?;
    for (pair, data) in pairs {
        let (a, b) = (pair / 2_147_483_647, pair % 2_147_483_647);
        let is_masked = |image: i64, idx: u32| masked.get(&image).and_then(|f| f.get(idx as usize)).copied().unwrap_or(false);
        let matches: Vec<[u32; 2]> = data
            .chunks_exact(8)
            .map(|m| [u32::from_le_bytes(m[..4].try_into().expect("4 bytes")), u32::from_le_bytes(m[4..].try_into().expect("4 bytes"))])
            .collect();
        let keep: Vec<[u32; 2]> = matches.iter().copied().filter(|[i, j]| !is_masked(a, *i) && !is_masked(b, *j)).collect();
        kept += keep.len();
        dropped += matches.len() - keep.len();
        if keep.len() < matches.len() {
            let blob: Vec<u8> = keep.iter().flat_map(|[i, j]| [i.to_le_bytes(), j.to_le_bytes()].concat()).collect();
            tx.execute(
                "UPDATE two_view_geometries SET rows = ?1, data = ?2 WHERE pair_id = ?3",
                rusqlite::params![keep.len() as i64, blob, pair],
            )?;
        }
    }
    tx.commit()?;
    Ok((kept, dropped))
}

/// Re-initialises every camera in a COLMAP database as `model` (focal and principal point kept,
/// distortion zero), so a reused database maps with a different camera model.
pub fn reset_camera(database: &Path, model: &str) -> Result<()> {
    // COLMAP model ids; the principal point follows one focal (SIMPLE_*, RADIAL) or two.
    let id = |m: &str| match m {
        "SIMPLE_PINHOLE" => Some((0, 3)),
        "PINHOLE" => Some((1, 4)),
        "SIMPLE_RADIAL" => Some((2, 4)),
        "RADIAL" => Some((3, 5)),
        "OPENCV" => Some((4, 8)),
        _ => None,
    };
    let (target, n) = id(model).with_context(|| format!("cannot reset a camera to {model}"))?;
    let db = rusqlite::Connection::open(database)?;
    let cameras: Vec<(i64, i64, Vec<u8>)> = db
        .prepare("SELECT camera_id, model, params FROM cameras")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (camera, source, blob) in cameras {
        let p: Vec<f64> = blob.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes"))).collect();
        let (f, cx, cy) = match source {
            0 | 2 | 3 => (p[0], p[1], p[2]),
            1 | 4 => (p[0], p[2], p[3]),
            other => anyhow::bail!("cannot reset camera model id {other}"),
        };
        let mut params = match target {
            1 | 4 => vec![f, f, cx, cy],
            _ => vec![f, cx, cy],
        };
        params.resize(n, 0.0);
        let blob: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        db.execute("UPDATE cameras SET model = ?1, params = ?2 WHERE camera_id = ?3", rusqlite::params![target, blob, camera])?;
    }
    Ok(())
}

/// Sets every camera's focal length in a COLMAP database to `factor` times its longer side (both
/// focals of a two-focal model); principal point and distortion are kept.
pub fn set_focal(database: &Path, factor: f64) -> Result<()> {
    let db = rusqlite::Connection::open(database)?;
    let cameras: Vec<(i64, i64, i64, i64, Vec<u8>)> = db
        .prepare("SELECT camera_id, model, width, height, params FROM cameras")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (camera, model, w, h, blob) in cameras {
        let mut p: Vec<f64> = blob.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes"))).collect();
        let f = factor * w.max(h) as f64;
        match model {
            0 | 2 | 3 => p[0] = f,
            1 | 4 => (p[0], p[1]) = (f, f),
            other => anyhow::bail!("cannot set the focal length of camera model id {other}"),
        }
        let blob: Vec<u8> = p.iter().flat_map(|v| v.to_le_bytes()).collect();
        db.execute("UPDATE cameras SET params = ?1, prior_focal_length = 1 WHERE camera_id = ?2", rusqlite::params![blob, camera])?;
    }
    Ok(())
}

/// The incremental mapper's threads. A keyframe map (~50 video frames) runs one: COLMAP's
/// multithreaded initial-pair search races, and the driveway clip's every-3rd-frame map (RADIAL) sometimes found
/// no 2 deg pair, relaxed to 1 deg, started from adjacent keyframes (#61/#64) and split, costing a
/// 68 s every-2nd-frame remap and a 25 s lens BA. One thread: the same #150/#22 every run, mapper
/// 16.0-16.4 -> 18.3-18.5 s. Continuation, registration and triangulation keep their threads.
fn keyframe_mapper_threads(opts: &ReconstructOptions) -> usize {
    opts.mapper_threads.unwrap_or(if opts.keyframe_stride > 1 { 1 } else { opts.num_threads })
}

/// Reject a known-bad video trajectory after the caller has saved its diagnostics.
/// Insufficient continuity evidence remains reported by the evaluator, not a fabricated pass.
pub fn ensure_video_continuity(health: &SfmHealth, video: bool) -> Result<()> {
    ensure_geometry_health(health)?;
    if video && health.gates.get("pose_continuity") == Some(&false) {
        return Err(ContinuityFailure.into());
    }
    Ok(())
}

/// Hard failure independent of input provenance/registration coverage. Kept separate from camera
/// flips: deleting stationary frames is not a repair for an inconsistent model scale.
pub fn ensure_geometry_health(health: &SfmHealth) -> Result<()> {
    if !health.geometry.pass {
        return Err(GeometryFailure.into());
    }
    Ok(())
}

#[derive(Debug)]
pub struct GeometryFailure;

impl std::fmt::Display for GeometryFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SfM geometry is incoherent (depth, translation or observation components); inspect health.geometry before training")
    }
}

impl std::error::Error for GeometryFailure {}

/// A video trajectory that failed pose continuity (a camera flip). Callers can find it under
/// added context with `downcast_ref` and retry with other mapper settings.
#[derive(Debug)]
pub struct ContinuityFailure;

impl std::fmt::Display for ContinuityFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("video camera trajectory failed pose continuity")
    }
}

impl std::error::Error for ContinuityFailure {}

#[derive(Clone, Debug, Serialize)]
pub struct ComponentSummary {
    pub model_dir: PathBuf,
    pub n_registered: usize,
    pub n_points: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ComponentInventory {
    pub models: Vec<ComponentSummary>,
    /// Distinct image names across all components, not the sum of component sizes.
    pub n_registered_union: usize,
    /// Largest component's share of all cameras registered in any component.
    pub largest_component_fraction: f64,
}

/// Inventory both mapper outputs and components archived by `finalize_model`.
/// Read-only: paths and original models are retained for inspecting fragmentation.
pub fn component_inventory(sparse_dir: &Path) -> Result<ComponentInventory> {
    let mut dirs = Vec::new();
    let archive = sparse_dir
        .parent()
        .context("sparse directory has no parent")?
        .join("components");
    for root in [sparse_dir, archive.as_path()] {
        if !root.exists() {
            continue;
        }
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            if path.is_dir()
                && (path.join("images.bin").is_file() || path.join("images.txt").is_file())
            {
                dirs.push(path);
            }
        }
    }
    dirs.sort();
    let mut names = BTreeSet::new();
    let mut models = Vec::new();
    for dir in dirs {
        let model = read_model(&dir).with_context(|| format!("reading {}", dir.display()))?;
        names.extend(model.images.values().map(|image| image.name.clone()));
        models.push(ComponentSummary {
            model_dir: dir,
            n_registered: model.images.len(),
            n_points: model.points.len(),
        });
    }
    let largest = models.iter().map(|model| model.n_registered).max().unwrap_or(0);
    Ok(ComponentInventory {
        models,
        n_registered_union: names.len(),
        largest_component_fraction: if names.is_empty() {
            0.0
        } else {
            largest as f64 / names.len() as f64
        },
    })
}

/// Put the selected component at `sparse/0` for consumers using the COLMAP convention.
/// Preserve other components in sibling `components/`; existing archives are never replaced.
/// External stage runners should call this after mapping instead of only `best_model`.
pub fn finalize_model(
    sparse_dir: &Path,
    frames_dir: &Path,
    video: bool,
    holdout_every: usize,
) -> Result<(PathBuf, SfmHealth)> {
    let (selected, mut health) = best_model(sparse_dir, frames_dir, video, holdout_every)?;
    let canonical = sparse_dir.join("0");
    let mut others: Vec<PathBuf> = fs::read_dir(sparse_dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<_>>()?;
    others.retain(|path| path.is_dir() && path != &selected);
    others.sort();
    if !others.is_empty() {
        let archive = sparse_dir
            .parent()
            .context("sparse directory has no parent")?
            .join("components");
        fs::create_dir(&archive)
            .with_context(|| format!("component archive must be fresh: {}", archive.display()))?;
        for path in others {
            let target = archive.join(path.file_name().context("component has no name")?);
            fs::rename(&path, &target)
                .with_context(|| format!("archiving component {}", path.display()))?;
        }
    }
    if selected != canonical {
        ensure!(
            !canonical.exists(),
            "canonical model path is occupied: {}",
            canonical.display()
        );
        fs::rename(&selected, &canonical)
            .with_context(|| format!("exporting selected component {}", selected.display()))?;
    }
    health.sparse_dir = canonical.to_string_lossy().into_owned();
    Ok((canonical, health))
}

/// Select the largest registered component, then point count, then lexical path.
/// Always recompute per-observation errors through splat-eval for comparable reports.
pub fn best_model(
    sparse_dir: &Path,
    frames_dir: &Path,
    video: bool,
    holdout_every: usize,
) -> Result<(PathBuf, SfmHealth)> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(sparse_dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    dirs.retain(|p| p.is_dir());
    dirs.sort();
    let mut best = None;
    for dir in dirs {
        let mut model = read_model(&dir).with_context(|| format!("reading {}", dir.display()))?;
        // Setting only the in-memory stored errors to zero selects splat-eval's recomputation
        // path; the model on disk is unchanged, and every camera model is validated above.
        for point in &mut model.points {
            point.error = 0.0;
        }
        let health = sfm_health(
            &model,
            &dir,
            &SfmHealthOptions {
                images_dir: Some(frames_dir.into()),
                video,
                holdout_every,
                ..Default::default()
            },
        )?;
        let rank = (health.n_registered, health.n_points);
        if best
            .as_ref()
            .is_none_or(|(_, h): &(PathBuf, SfmHealth)| rank > (h.n_registered, h.n_points))
        {
            best = Some((dir, health));
        }
    }
    let Some((dir, health)) = best else {
        bail!("mapper wrote no reconstruction in {}", sparse_dir.display());
    };
    ensure!(
        health.n_registered > 0 && health.n_points > 0,
        "mapper wrote an empty reconstruction"
    );
    Ok((dir, health))
}

/// Pose-fixed, CPU-only initialization pass after selecting the mapped component.
/// Output must be an empty directory distinct from input; the caller creates it.
/// Include this command after `commands` when using an external job runtime.
/// A slow translation (driveway, a slow slide): nearly every near pair (<= 2 frames apart) is
/// low-parallax (COLMAP config planar/panoramic) while most pairs overall are general. No 16 deg
/// initial pair exists among nearby frames, so the 16 deg attempt collapses (~90 s) before the
/// 4 deg retry: start there, at 2 deg and 50 inliers, where COLMAP's two relaxations of a 4 deg /
/// 100-inlier search end (the 4 and the halved-inlier searches found no pair either: driveway, 14-43 s;
/// the same initial pair #71/#135). After hybrid matching, near / all low-parallax shares: driveway
/// 0.97 / 0.23; goresbridge 1.00 / 0.58 (rotation, maps at 16 deg); gimbal 0.73 / 0.19;
/// night-stroll 0.62 / 0.56 (fragments at 4 deg); kindel 0.43 / 0.09; rooftop 0.08 / 0.13.
fn low_motion_init(mapper: &mut StageCommand, output_dir: &Path) -> Result<()> {
    let db = rusqlite::Connection::open(output_dir.join("database.db"))?;
    let mut names: Vec<(i64, String)> = db
        .prepare("SELECT image_id, name FROM images")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    names.sort_by(|a, b| a.1.cmp(&b.1));
    let order: std::collections::HashMap<i64, i64> = names.iter().enumerate().map(|(k, (id, _))| (*id, k as i64)).collect();
    let pairs: Vec<(i64, i64)> = db
        .prepare("SELECT pair_id, config FROM two_view_geometries WHERE rows > 0")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let low = |config: i64| matches!(config, 4..=6);
    let (mut near, mut near_low, mut all_low) = (0usize, 0usize, 0usize);
    for &(pair, config) in &pairs {
        let (a, b) = (pair / 2_147_483_647, pair % 2_147_483_647);
        all_low += usize::from(low(config));
        if let (Some(i), Some(j)) = (order.get(&a), order.get(&b)) {
            if (i - j).abs() <= 2 {
                near += 1;
                near_low += usize::from(low(config));
            }
        }
    }
    let near_share = near_low as f64 / near.max(1) as f64;
    let all_share = all_low as f64 / pairs.len().max(1) as f64;
    let slow = near >= 20 && near_share >= 0.9 && all_share < 0.4;
    if slow {
        let at = mapper.args.iter().position(|a| a == "--Mapper.init_min_tri_angle")
            .context("mapper has no init_min_tri_angle")?;
        mapper.args[at + 1] = (LOW_INIT_ANGLE / 2.0).to_string().into();
        mapper.number("Mapper.init_min_num_inliers", 50);
    }
    fs::write(
        output_dir.join("motion.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "near_pairs": near, "near_low_parallax_share": near_share,
            "low_parallax_share": all_share, "slow_translation": slow,
        }))?,
    )?;
    Ok(())
}

/// Whether the largest model under `<output_dir>/sparse` holds every keyframe of keyframes.txt.
fn keyframe_map_complete(output_dir: &Path) -> Result<bool> {
    let keyframes = fs::read_to_string(output_dir.join("keyframes.txt"))?.lines().filter(|l| !l.is_empty()).count();
    let mut largest = 0;
    for entry in fs::read_dir(output_dir.join("sparse"))? {
        let path = entry?.path();
        if path.is_dir() {
            largest = largest.max(read_model(&path).map(|m| m.images.len()).unwrap_or(0));
        }
    }
    Ok(largest >= keyframes)
}

/// The widest verified initial pair for a sparse keyframe map: of the keyframe pairs with a general
/// (not planar or panoramic) two-view geometry and at least 100 inliers, the one farthest apart in
/// frame order, then the most inliers. Driveway clip (every 3rd of 150 frames, a slow slide): #22/#150, 128
/// frames and 367 inliers, the pair COLMAP's successful searches start from; its search sometimes
/// started from adjacent keyframes (#121/#118) instead and split the map. The most inliers at >= 15
/// frames apart (#1/#16) lacks the parallax.
fn widest_init_pair(output_dir: &Path) -> Result<Option<(i64, i64)>> {
    let keyframes: BTreeSet<String> = fs::read_to_string(output_dir.join("keyframes.txt"))?.lines().map(str::to_owned).collect();
    let db = rusqlite::Connection::open(output_dir.join("database.db"))?;
    let mut names: Vec<(i64, String)> = db
        .prepare("SELECT image_id, name FROM images")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    names.sort_by(|a, b| a.1.cmp(&b.1));
    let order: std::collections::HashMap<i64, i64> = names.iter().enumerate()
        .filter(|(_, (_, name))| keyframes.contains(name))
        .map(|(k, (id, _))| (*id, k as i64))
        .collect();
    // COLMAP TwoViewGeometry configs 2 = CALIBRATED, 3 = UNCALIBRATED (general motion).
    let pairs: Vec<(i64, i64)> = db
        .prepare("SELECT pair_id, rows FROM two_view_geometries WHERE rows >= 100 AND config IN (2, 3)")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(pairs
        .into_iter()
        .filter_map(|(pair, rows)| {
            let (a, b) = (pair / 2_147_483_647, pair % 2_147_483_647);
            Some(((order.get(&a)? - order.get(&b)?).abs(), rows, a, b))
        })
        .max_by_key(|&(gap, rows, a, b)| (gap, rows, -a, -b))
        .map(|(_, _, a, b)| (a, b)))
}

/// Map the keyframes from [`widest_init_pair`], one model, with all threads (a fixed pair leaves no
/// initial-pair search to race). Returns false, leaving `sparse/` empty, when there is no such pair
/// or COLMAP finds it unsuitable; the caller then runs the ordinary search.
fn forced_keyframe_map(stage: &StageCommand, opts: &ReconstructOptions, timings: &mut Vec<StageTiming>) -> Result<bool> {
    let Some((a, b)) = widest_init_pair(&opts.output_dir)? else { return Ok(false) };
    let mut forced = stage.clone();
    if let Some(at) = forced.args.iter().position(|x| x == "--Mapper.num_threads") {
        forced.args[at + 1] = opts.num_threads.to_string().into();
    }
    forced.number("Mapper.init_image_id1", a);
    forced.number("Mapper.init_image_id2", b);
    // One model: COLMAP would otherwise retry the same pair for every further model.
    forced.number("Mapper.multiple_models", 0);
    forced.number("Mapper.init_num_trials", 3);
    let start = Instant::now();
    match run_stage(forced, &opts.output_dir, None) {
        Ok(timing) if fs::read_dir(opts.output_dir.join("sparse"))?.next().is_some() => {
            timings.push(timing);
            Ok(true)
        }
        result => {
            eprintln!("[sfm] keyframe map from #{a}/#{b} failed ({}); searching for an initial pair", result.err().map_or("no model".into(), |e| format!("{e:#}")));
            timings.push(StageTiming {
                name: "mapper_forced_pair".into(),
                seconds: start.elapsed().as_secs_f64(),
                argv: Vec::new(),
                log: opts.output_dir.join("mapper.log"),
                gpu_lock_held: false,
                cpu_seconds: None,
            });
            let sparse = opts.output_dir.join("sparse");
            fs::remove_dir_all(&sparse)?;
            fs::create_dir(&sparse)?;
            Ok(false)
        }
    }
}

/// Every `stride`-th image name of `frames_dir` (sorted) plus the last, one per line.
fn write_keyframes(frames_dir: &Path, stride: usize, path: &Path) -> Result<()> {
    let names = frame_names(frames_dir)?;
    ensure!(!names.is_empty(), "no frames in {}", frames_dir.display());
    let last = names.len() - 1;
    let keep: Vec<&str> = names.iter().enumerate()
        .filter(|(i, _)| i % stride == 0 || *i == last)
        .map(|(_, name)| name.as_str())
        .collect();
    fs::write(path, keep.join("\n") + "\n")?;
    Ok(())
}

/// Sorted image file names directly in `frames_dir`.
fn frame_names(frames_dir: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = fs::read_dir(frames_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            [".jpg", ".jpeg", ".png"].iter().any(|ext| lower.ends_with(ext))
        })
        .collect();
    names.sort();
    Ok(names)
}

/// Final bundle adjustment of `lens_last`: focal length and distortion (principal point fixed)
/// together with poses and points.
pub fn lens_command(model_dir: &Path, output_dir: &Path, opts: &ReconstructOptions) -> Result<StageCommand> {
    ensure!(model_dir != output_dir, "lens refinement input and output must differ");
    let mut stage = StageCommand::new(opts, "bundle_adjuster");
    stage.name = "lens_bundle_adjuster".into();
    stage.option("input_path", model_dir);
    stage.option("output_path", output_dir);
    stage.number("BundleAdjustment.refine_focal_length", u8::from(opts.fixed_focal_factor.is_none()));
    stage.number("BundleAdjustment.refine_principal_point", 0);
    stage.number("BundleAdjustment.refine_extra_params", 1);
    // One large BA over every frame (driveway 146-192 s on CPU, the biggest SfM stage). Ceres solves
    // up to 200 images with its dense GPU solver; same problem and optimum, CPU above that.
    stage.number("BundleAdjustmentCeres.use_gpu", u8::from(opts.use_gpu));
    stage.option("log_target", "stderr");
    Ok(stage)
}

pub fn triangulation_command(
    frames_dir: &Path,
    model_dir: &Path,
    output_dir: &Path,
    opts: &ReconstructOptions,
) -> Result<StageCommand> {
    opts.validate()?;
    ensure!(
        model_dir != output_dir,
        "triangulation input and output must differ"
    );
    let mut stage = StageCommand::new(opts, "point_triangulator");
    stage.option("database_path", opts.output_dir.join("database.db"));
    stage.option("image_path", frames_dir);
    stage.option("input_path", model_dir);
    stage.option("output_path", output_dir);
    stage.number("clear_points", 0);
    stage.number("refine_intrinsics", 0);
    stage.number("Mapper.fix_existing_frames", 1);
    // Poses are fixed here, so the mapper's one-thread determinism isn't needed (room: 1.9 s at
    // num_threads vs 2.8 s at one thread).
    stage.number("Mapper.num_threads", opts.num_threads);
    stage.number("Mapper.ba_use_gpu", 0);
    stage.number("Mapper.tri_ignore_two_view_tracks", 0);
    stage.number("Mapper.tri_min_angle", 1);
    stage.number("Mapper.filter_min_tri_angle", 1);
    stage.number("Mapper.ba_global_max_num_iterations", 30);
    stage.number("Mapper.ba_global_max_refinements", 2);
    stage.option("log_target", "stderr");
    Ok(stage)
}

/// One imported (authoritative) camera: world-to-camera pose and intrinsics, keyed by file name.
#[derive(Clone, Debug)]
pub struct FixedCamera {
    pub name: String,
    pub model: String,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
    /// World-to-camera rotation (w, x, y, z) and translation.
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
}

#[derive(Debug, Serialize)]
pub struct FixedPoseSeed {
    pub model_dir: PathBuf,
    pub points: usize,
    pub images: usize,
    pub stages: Vec<StageTiming>,
}

/// Triangulates points under imported cameras without ever re-solving them: features and matches
/// come from COLMAP, the poses/intrinsics are written into the input model (with the database's
/// own image/camera ids) and `point_triangulator` runs with fixed frames and intrinsics.
/// Every stage is killed past `timeout` in total. Always CPU. `frames_dir` must hold exactly the images to use (callers exclude held-out views). The returned
/// model's cameras/images are COLMAP's echo of the input; only its points are meaningful.
pub fn triangulate_fixed_poses(frames_dir: &Path, cameras: &[FixedCamera], opts: &ReconstructOptions, timeout: Duration) -> Result<FixedPoseSeed> {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            STAGE_DEADLINE.set(None);
        }
    }
    STAGE_DEADLINE.set(Some(Instant::now() + timeout));
    let _reset = Reset;
    // Authored intrinsics are per image: never a shared camera, never GPU (no scheduling lock here).
    let mut opts = opts.clone();
    opts.single_camera = false;
    opts.use_gpu = false;
    let opts = &opts;
    ensure!(cameras.len() >= 2, "need at least two imported cameras to triangulate");
    fs::create_dir_all(&opts.output_dir)?;
    let database = opts.output_dir.join("database.db");
    let mut stages = Vec::new();
    for mut stage in commands(frames_dir, opts)? {
        match stage.name.as_str() {
            "feature_extractor" => {
                // Every image keeps its own authored intrinsics.
                stage.option("ImageReader.single_camera_per_image", "1");
            }
            // Sequential matching skips verification and relies on this separate stage.
            "exhaustive_matcher" | "sequential_matcher" | "geometric_verifier" => {}
            _ => continue,
        }
        stages.push(run_stage(stage, &opts.output_dir, None)?);
    }
    let db = rusqlite::Connection::open(&database).context("opening the COLMAP database")?;
    // image -> (camera, frame, rig), straight from the database: COLMAP 4 requires the input
    // model's camera, rig and frame ids to match the database's.
    let rows: Vec<(i64, String, i64, i64, i64)> = db
        .prepare(
            "SELECT i.image_id, i.name, i.camera_id, f.frame_id, f.rig_id FROM images i \
             JOIN frame_data d ON d.data_id = i.image_id AND d.sensor_type = 0 \
             JOIN frames f ON f.frame_id = d.frame_id ORDER BY i.image_id",
        )
        .context("the COLMAP database has no rig/frame tables (COLMAP 4 expected)")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let n_images: i64 = db.query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))?;
    ensure!(rows.len() as i64 == n_images, "{} of {n_images} database images have no camera frame", rows.len());
    let unique: BTreeSet<i64> = rows.iter().map(|r| r.2).collect();
    ensure!(unique.len() == rows.len(), "the COLMAP database shares cameras between images; imported intrinsics are per image");
    let frames: BTreeSet<i64> = rows.iter().map(|r| r.3).collect();
    ensure!(frames.len() == rows.len(), "the COLMAP database shares frames between images");
    let rig_sensors: i64 = db.query_row("SELECT COUNT(*) FROM rig_sensors", [], |r| r.get(0))?;
    ensure!(rig_sensors == 0, "the COLMAP database has multi-sensor rigs; only single-camera rigs are supported");
    let by_name: BTreeMap<&str, &FixedCamera> = cameras.iter().map(|c| (c.name.as_str(), c)).collect();
    ensure!(rows.iter().all(|r| by_name.contains_key(r.1.as_str())), "the COLMAP database holds an image without an imported camera");
    ensure!(rows.len() >= 2, "the COLMAP database holds fewer than two of the imported cameras' images");
    // The extractor guessed intrinsics (SIMPLE_RADIAL...). Replace them with the authored cameras so
    // the database and the input model agree on model, size and parameters; keypoints stay valid
    // because the images are the ones the cameras describe.
    for (_, name, camera_id, _, _) in &rows {
        let c = by_name[name.as_str()];
        let model_id = colmap_model_id(&c.model)?;
        let (w, h): (i64, i64) = db.query_row("SELECT width, height FROM cameras WHERE camera_id = ?1", [camera_id], |r| Ok((r.get(0)?, r.get(1)?)))?;
        ensure!((w as u64, h as u64) == (c.width, c.height), "{name}: image is {w}x{h} but the imported camera is {}x{}", c.width, c.height);
        ensure!(c.params.len() == colmap_model_params(&c.model)? && c.params.iter().all(|v| v.is_finite()), "{name}: invalid {} parameters", c.model);
        let blob: Vec<u8> = c.params.iter().flat_map(|v| v.to_le_bytes()).collect();
        db.execute("UPDATE cameras SET model = ?1, params = ?2, prior_focal_length = 1 WHERE camera_id = ?3", rusqlite::params![model_id, blob, camera_id])?;
    }
    let ref_sensors: BTreeMap<i64, i64> = db
        .prepare("SELECT rig_id, ref_sensor_id FROM rigs")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(db);
    let input = opts.output_dir.join("fixed_input");
    fs::create_dir_all(&input)?;
    let (mut cams_txt, mut imgs_txt) = (String::from("# fixed imported cameras\n"), String::from("# fixed imported poses\n"));
    let (mut rigs_txt, mut frames_txt) = (String::from("# fixed single-camera rigs\n"), String::from("# fixed imported frames\n"));
    for (id, name, camera_id, frame_id, rig_id) in &rows {
        let c = by_name[name.as_str()];
        ensure!(ref_sensors.get(rig_id) == Some(camera_id), "{name}: rig {rig_id} is not the single-camera rig of camera {camera_id}");
        let params: Vec<String> = c.params.iter().map(|v| format!("{v:.17e}")).collect();
        let pose = format!("{:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}", c.qvec[0], c.qvec[1], c.qvec[2], c.qvec[3], c.tvec[0], c.tvec[1], c.tvec[2]);
        cams_txt += &format!("{camera_id} {} {} {} {}\n", c.model, c.width, c.height, params.join(" "));
        imgs_txt += &format!("{id} {pose} {camera_id} {name}\n\n");
        rigs_txt += &format!("{rig_id} 1 CAMERA {camera_id}\n");
        frames_txt += &format!("{frame_id} {rig_id} {pose} 1 CAMERA {camera_id} {id}\n");
    }
    fs::write(input.join("cameras.txt"), cams_txt)?;
    fs::write(input.join("images.txt"), imgs_txt)?;
    fs::write(input.join("rigs.txt"), rigs_txt)?;
    fs::write(input.join("frames.txt"), frames_txt)?;
    fs::write(input.join("points3D.txt"), "")?;
    let out = opts.output_dir.join("triangulated");
    fs::create_dir_all(&out)?;
    stages.push(run_stage(triangulation_command(frames_dir, &input, &out, opts)?, &opts.output_dir, None)?);
    let model = read_model(&out).context("reading the triangulated model")?;
    Ok(FixedPoseSeed { model_dir: out, points: model.points.len(), images: model.images.len(), stages })
}

/// COLMAP's database id for a camera model name.
fn colmap_model_id(model: &str) -> Result<i64> {
    Ok(match model {
        "SIMPLE_PINHOLE" => 0,
        "PINHOLE" => 1,
        "SIMPLE_RADIAL" => 2,
        "RADIAL" => 3,
        "OPENCV" => 4,
        other => bail!("unsupported imported camera model {other}"),
    })
}

fn colmap_model_params(model: &str) -> Result<usize> {
    Ok(match model {
        "SIMPLE_PINHOLE" => 3,
        "PINHOLE" => 4,
        "SIMPLE_RADIAL" => 4,
        "RADIAL" => 5,
        "OPENCV" => 8,
        other => bail!("unsupported imported camera model {other}"),
    })
}

/// Registers the database's other frames into `model_dir` by pose estimation alone (no bundle
/// adjustment, no new points); cameras and existing poses are unchanged.
pub fn registration_command(model_dir: &Path, output_dir: &Path, opts: &ReconstructOptions) -> Result<StageCommand> {
    opts.validate()?;
    ensure!(model_dir != output_dir, "registration input and output must differ");
    let mut stage = StageCommand::new(opts, "image_registrator");
    stage.option("database_path", opts.output_dir.join("database.db"));
    stage.option("input_path", model_dir);
    stage.option("output_path", output_dir);
    stage.number("Mapper.num_threads", opts.mapper_threads.unwrap_or(opts.num_threads));
    stage.number("Mapper.abs_pose_max_error", 16);
    stage.option("log_target", "stderr");
    Ok(stage)
}

/// CPU-only incremental mapping seeded by the selected component; its poses and
/// intrinsics stay fixed while unregistered frames are added to the same model.
pub fn continuation_command(
    frames_dir: &Path,
    model_dir: &Path,
    output_dir: &Path,
    opts: &ReconstructOptions,
) -> Result<StageCommand> {
    opts.validate()?;
    ensure!(model_dir != output_dir, "continuation input and output must differ");
    let mut stage = StageCommand::new(opts, "mapper");
    stage.name = "continuation_mapper".into();
    stage.option("database_path", opts.output_dir.join("database.db"));
    stage.option("image_path", frames_dir);
    stage.option("input_path", model_dir);
    stage.option("output_path", output_dir);
    stage.number("Mapper.num_threads", opts.mapper_threads.unwrap_or(opts.num_threads));
    stage.number("Mapper.random_seed", opts.mapper_seed);
    stage.number("Mapper.multiple_models", 0);
    stage.number("Mapper.fix_existing_frames", 1);
    stage.number("Mapper.ba_refine_focal_length", 0);
    stage.number("Mapper.ba_refine_extra_params", 0);
    stage.number("Mapper.ba_refine_sensor_from_rig", 0);
    stage.number("Mapper.abs_pose_max_error", 16);
    stage.number("Mapper.ba_use_gpu", 0);
    // Video: the selected mapper's global-BA schedule (ratio, iterations, refinements), not
    // COLMAP's defaults: only the new frames and points move here, and the lens BA re-refines all.
    if opts.video {
        if let Some(ratio) = opts.ba_global_ratio {
            stage.number("Mapper.ba_global_frames_ratio", ratio);
            stage.number("Mapper.ba_global_points_ratio", ratio);
        }
        if let Some(n) = opts.ba_global_max_iterations {
            stage.number("Mapper.ba_global_max_num_iterations", n);
        }
        if let Some(n) = opts.ba_global_refinements {
            stage.number("Mapper.ba_global_max_refinements", n);
        }
    }
    stage.option("log_target", "stderr");
    Ok(stage)
}

/// Leftover frames: COLMAP's absolute-pose inlier floor is 30; a leftover frame may have fewer
/// verified links into the model and still be well constrained.
const LEFTOVER_MIN_INLIERS: usize = 15;
/// A leftover frame added to a model must observe this many of its 3D points (a frame placed from a
/// handful of matches would otherwise ship as a confident but wrong camera).
const LEFTOVER_MIN_OBSERVATIONS: usize = 20;

fn weak_new_frames(before: &splat_eval::Model, after: &splat_eval::Model) -> Vec<(String, usize)> {
    let floor = LEFTOVER_MIN_OBSERVATIONS;
    let old: std::collections::HashSet<&str> = before.images.values().map(|im| im.name.as_str()).collect();
    after.images.values().filter(|im| !old.contains(im.name.as_str()))
        .map(|im| (im.name.clone(), im.obs_point_id.len())).filter(|(_, n)| *n < floor).collect()
}

fn same_cameras(a: &splat_eval::Model, b: &splat_eval::Model) -> bool {
    a.cameras.len() == b.cameras.len()
        && a.cameras.iter().all(|(id, cam)| {
            b.cameras.get(id).is_some_and(|c| {
                cam.model == c.model && cam.width == c.width && cam.height == c.height && cam.params == c.params
            })
        })
}

/// Reject an initialization pass that moved cameras. Quaternion sign/normalization
/// may change on serialization; compare rotation matrices with rounding tolerance.
pub fn ensure_fixed_cameras(before: &Path, after: &Path) -> Result<()> {
    let a = read_model(before)?;
    let b = read_model(after)?;
    ensure!(
        a.cameras.len() == b.cameras.len() && a.images.len() == b.images.len(),
        "triangulation changed camera/image counts"
    );
    for (id, cam) in &a.cameras {
        let same = b.cameras.get(id).is_some_and(|c| {
            cam.model == c.model
                && cam.width == c.width
                && cam.height == c.height
                && cam.params == c.params
        });
        ensure!(same, "triangulation changed camera {id}");
    }
    for (id, im) in &a.images {
        let other = b.images.get(id).context("triangulation removed an image")?;
        let r1 = splat_eval::Model::rotation(im.qvec);
        let r2 = splat_eval::Model::rotation(other.qvec);
        ensure!(
            im.name == other.name
                && im.camera_id == other.camera_id
                && im.tvec == other.tvec
                && r1
                    .iter()
                    .flatten()
                    .zip(r2.iter().flatten())
                    .all(|(a, b)| (a - b).abs() <= 1e-12),
            "triangulation changed pose/image {id}"
        );
    }
    Ok(())
}

/// Niceness of the COLMAP stages that run while the late masks are still being made (see
/// [`ReconstructOptions::late_masks`]).
const PREMASK_NICE: i32 = 10;

fn run_stage(
    stage: StageCommand,
    output_dir: &Path,
    gpu_lock: Option<&File>,
) -> Result<StageTiming> {
    run_stage_at(stage, output_dir, gpu_lock, 0)
}

/// [`run_stage`] with the child's niceness raised by `nice` (0 = inherit).
fn run_stage_at(
    stage: StageCommand,
    output_dir: &Path,
    gpu_lock: Option<&File>,
    nice: i32,
) -> Result<StageTiming> {
    #[cfg(not(unix))]
    let _ = (gpu_lock, nice);
    let log_path = output_dir.join(format!("{}.log", stage.name));
    let log = File::create(&log_path)?;
    let now = Instant::now();
    let mut command = Command::new(&stage.program);
    command
        .args(&stage.args)
        .env("QT_QPA_PLATFORM", "offscreen")
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // Keep flock ownership in the native process if the Rust worker dies first.
    // Only the child's copy becomes inheritable; the parent's fd stays CLOEXEC.
    #[cfg(unix)]
    if let Some(lock) = gpu_lock {
        use std::os::{fd::AsRawFd, unix::process::CommandExt};
        let fd = lock.as_raw_fd();
        // SAFETY: the closure only calls async-signal-safe fcntl after fork. The
        // lock lives until wait() reaps the child, so its descriptor is valid.
        unsafe {
            command.pre_exec(move || {
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    #[cfg(unix)]
    if nice > 0 {
        use std::os::unix::process::CommandExt;
        // SAFETY: nice(2) is async-signal-safe. Raising niceness never needs privileges; a failure
        // leaves the default priority, which is only slower.
        unsafe {
            command.pre_exec(move || {
                libc::nice(nice);
                Ok(())
            });
        }
    }
    let deadline = STAGE_DEADLINE.get();
    // With a deadline the child leads its own process group so a timeout takes its descendants too.
    #[cfg(unix)]
    if deadline.is_some() {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("starting {}", stage.name))?;
    let (cpu_seconds, status, timed_out) = match deadline {
        None => {
            let cpu = wait_exited(&mut child)?;
            (cpu, child.wait()?, false)
        }
        Some(deadline) => {
            // One owner polls, kills and reaps the same handle: no pid is signalled after reaping.
            let mut timed_out = false;
            let status = loop {
                if let Some(status) = child.try_wait()? {
                    break status;
                }
                if Instant::now() >= deadline {
                    timed_out = true;
                    #[cfg(unix)]
                    // SAFETY: the child is unreaped (try_wait just said so and only this thread
                    // reaps it), so its pid, which is its process-group id, is not recycled.
                    unsafe {
                        libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
                    }
                    let _ = child.kill();
                    break child.wait()?;
                }
                std::thread::sleep(Duration::from_millis(20));
            };
            (None, status, timed_out)
        }
    };
    ensure!(!timed_out, "{} timed out and was killed; see {}", stage.name, log_path.display());
    CHILD_CPU.set(CHILD_CPU.get() + cpu_seconds.unwrap_or(0.0));
    let elapsed = now.elapsed().as_secs_f64();
    ensure!(
        status.success(),
        "{} failed ({status}); see {}",
        stage.name,
        log_path.display()
    );
    Ok(StageTiming {
        name: stage.name,
        seconds: elapsed,
        argv: std::iter::once(stage.program.as_os_str())
            .chain(stage.args.iter().map(OsString::as_os_str))
            .map(|a| a.to_string_lossy().into_owned())
            .collect(),
        log: log_path,
        gpu_lock_held: gpu_lock.is_some(),
        cpu_seconds,
    })
}

thread_local! {
    /// Wall-clock limit for every stage started on this thread (see [`triangulate_fixed_poses`]).
    static STAGE_DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// An environment override; an unset or unparsable variable keeps the product default.
fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok()?.parse().ok()
}
