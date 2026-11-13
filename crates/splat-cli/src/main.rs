mod blender;
mod config;
mod effect;
mod game_source;
mod pipeline;
mod progress;
mod resources;
mod sfm_align;
use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use config::Config;
use pipeline::{FixedOptions, FixedOverrides, PresetFlags};
use serde_json::{Value, json};
use splat_contracts::{Status, tool_schemas};
use splat_core::{Run, RunCtx, dispatch, report::write_report};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(
    name = "vauss",
    bin_name = "vauss",
    version,
    about = "Reconstruct, evaluate and direct Gaussian splat scenes"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    json: bool,
    /// Accept sources whose long edge is under 480 px (results will be poor).
    #[arg(long, global = true)]
    allow_low_resolution: bool,
    #[command(subcommand)]
    command: Commands,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Driver {
    Fixed,
    Claude,
    Codex,
}
impl Driver {
    fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Claude => "claude",
            Self::Codex => "codex",
        }
    }
}
#[derive(Args)]
struct ExistingRun {
    #[arg(long, required_unless_present = "path", conflicts_with = "path")]
    run_dir: Option<PathBuf>,
    /// Run directory (also accepted as --run-dir).
    path: Option<PathBuf>,
}
impl ExistingRun {
    fn root(&self) -> &Path {
        self.run_dir.as_deref().or(self.path.as_deref()).unwrap()
    }
}
#[derive(Subcommand)]
enum Commands {
    Doctor {
        #[arg(long, default_value = "brush")]
        backend: String,
        #[arg(long)]
        run_dir: Option<PathBuf>,
    },
    Init {
        source: PathBuf,
        #[arg(long)]
        goal: String,
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long, default_value = "brush")]
        backend: String,
    },
    Tool {
        name: String,
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long = "input", default_value = "{}")]
        input: String,
    },
    Tools,
    Run {
        /// Video, image folder, or phone capture export (Record3D, Polycam, transforms.json,
        /// ARKitScenes); capture directories and .zip/.r3d archives supply depth and fallback poses.
        source: PathBuf,
        #[arg(long)]
        goal: String,
        #[arg(long, default_value = "brush")]
        backend: String,
        #[arg(long, value_enum, default_value = "claude")]
        driver: Driver,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        effort: Option<String>,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        non_interactive: bool,
        #[arg(long)]
        time_budget: Option<String>,
        /// Run GPU work (SfM, training, evaluation, mesh) on fal cloud GPUs; the agent stays local.
        #[arg(long)]
        cloud: bool,
        /// Download every result of the fal jobs (full-model PLYs, side mesh). By default those stay on fal
        /// /data, listed in <run>/remote_artifacts.json (`vauss pull <run>` downloads them); the run brings
        /// back the web package, metrics and logs. Also SPLAT_CLOUD_FETCH_ALL=1. Kept in cloud.json.
        #[arg(long, requires = "cloud")]
        fetch_all: bool,
        /// Judge the video as a room in the capture check (also on when the goal names a room).
        #[arg(long)]
        room: bool,
        /// Go on even when the capture check says the video won't work.
        #[arg(long)]
        ignore_check: bool,
        #[command(flatten)]
        fixed: FixedOverrides,
        #[command(flatten)]
        preset: PresetFlags,
    },
    /// Capture pre-check: seconds of local analysis that say whether a video will work and how to
    /// film it better (`vauss run` runs it first). JSON on stdout, a plain report on stderr;
    /// exit 6 when the video won't work.
    Check {
        #[arg(required_unless_present = "guide")]
        source: Option<PathBuf>,
        /// Judge it as a room (adds the coverage check).
        #[arg(long)]
        room: bool,
        /// Print the filming guide instead.
        #[arg(long)]
        guide: bool,
    },
    Status {
        #[command(flatten)]
        run: ExistingRun,
    },
    Logs {
        #[command(flatten)]
        run: ExistingRun,
        #[arg(long)]
        follow: bool,
    },
    Resume {
        #[command(flatten)]
        run: ExistingRun,
        #[arg(long, value_enum)]
        driver: Option<Driver>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        effort: Option<String>,
        #[arg(long)]
        time_budget: Option<String>,
        #[command(flatten)]
        fixed: FixedOverrides,
    },
    /// Train a finished run's result further on its own cameras (no SfM re-solve) and frozen split:
    /// resume its final checkpoint up to --steps, evaluate, mesh (if the run made one), re-export the
    /// web package, and make the refined candidate the run's result. In place, or into a copy with --run-dir.
    /// --fast/--max start a fresh schedule from its splat unless --steps explicitly sets a checkpoint target.
    Refine {
        /// The finished run to refine.
        run: PathBuf,
        /// Write the refined run here (a copy; the original stays untouched) instead of in place.
        #[arg(long)]
        run_dir: Option<PathBuf>,
        /// Total training steps; overrides the preset (default: 30000, or the cloud preset minimum).
        #[arg(long)]
        steps: Option<u32>,
        /// Splat cap; overrides the preset (default: the candidate's cap or the cloud budget).
        #[arg(long)]
        cap_max: Option<u32>,
        /// GPU work on fal with the current cloud preset (SPLAT_CLOUD_PRESET; default MAX). Without it a
        /// `--cloud` run keeps its own cloud preset and a local run stays local.
        #[arg(long)]
        cloud: bool,
        /// With --cloud: download every result (see `vauss run --fetch-all`).
        #[arg(long, requires = "cloud")]
        fetch_all: bool,
        #[command(flatten)]
        preset: PresetFlags,
    },
    /// Download the results a `vauss run --cloud` left on fal (<run>/remote_artifacts.json: full-model
    /// PLYs, side mesh) in place of their stubs.
    Pull {
        run: PathBuf,
    },
    Stop {
        #[command(flatten)]
        run: ExistingRun,
        #[arg(long, default_value_t = 60.)]
        wait_s: f64,
    },
    Serve {
        #[arg(long)]
        run_dir: PathBuf,
    },
    /// Web scene package (scene.spz + scene.json: +Y up, meters, floor at y=0) for the decorator.
    ExportWeb(splat_webexport::ExportWebArgs),
    /// Standalone double-click (file://) viewer folder for a run's web package: fast first picture,
    /// auto-swap to the full model, capture-pose views, navigation kept near the capture (?free=1: off).
    /// `vauss run` writes it to <run>/exports/viewer at the end.
    ExportViewer(splat_webexport::viewer::ExportViewerArgs),
    /// Double-click (file://) first-person voxel game for a run's web package: a 2-5 cm voxel world from
    /// the splats' depth fusion (shoot to break voxels; V toggles the splat look), gamepad or WASD/mouse.
    /// `vauss run` writes it to <run>/exports/game at the end (SPLAT_EXPORT_GAME=0: skip).
    ExportGame(splat_webexport::game::ExportGameArgs),
    /// Blender pack for a finished run: whole-scene textured mesh, full splat PLY, the web scene.spz, coloured surface voxels and
    /// import steps. `vauss run` writes it to <run>/exports/blender at the end (SPLAT_EXPORT_BLENDER=0: skip).
    ExportBlender(blender::ExportBlenderArgs),
    /// CPU collision mesh, convex pieces and Unity import folder in the exported splat frame.
    ExportCollision(ExportCollisionCmd),
    /// Off-path view check: render fixed views 0.5 m sideways / 30° turned from K capture frames for one or
    /// two models (GPU), and write labelled side-by-side contact sheets plus no-reference metrics
    /// (sharpness, holes, depth-edge density, near floaters) to OUT/offpath.json.
    Offpath(OffpathCmd),
    /// Generative fill-in for free roaming: render novel views on the free floor near the walked path, repair them
    /// with a fixer model (default NVIDIA Fixer; `--fixer difix-ref` is NON-COMMERCIAL) and train them back in at low
    /// weight beside the real frames. Writes <run>/fill (or --out) with the new splat at model/splat.ply.
    Fill(splat_fill::FillArgs),
    #[command(flatten)]
    Shots(splat_shots::cli::Cmd),
    /// View and decorate splat scenes in the browser (scene/run folders, .ply/.spz; none: a picker).
    #[command(alias = "decorate")]
    View(splat_assets::ViewArgs),
    #[command(subcommand, hide = true)]
    Assets(splat_assets::cli::Cmd),
    /// Render a B15 analytic-surface request with native 2DGS surfels (debug adapter).
    #[command(hide = true)]
    SurfelFixture { request: PathBuf },
    /// Programmable splat effects (effect graph v1): validate, apply, preview, render, export, bench.
    #[command(subcommand)]
    Effect(effect::Cmd),
    /// Download a pinned helper or model from assets.json into the cache. The asset's licence is printed
    /// and recorded in DOWNLOAD_NOTICE.txt. GPL helpers (spirula) need --allow-gpl. da3-base is
    /// Apache-2.0: never substitute DA3-LARGE/GIANT (CC-BY-NC-4.0, non-commercial).
    Fetch {
        asset: String,
        /// Accept a GPL-3.0 helper (Spirula): it is downloaded and run as a separate executable, never linked.
        #[arg(long)]
        allow_gpl: bool,
    },
}
static COMPACT_JSON: AtomicBool = AtomicBool::new(false);
fn print(value: &impl serde::Serialize) -> Result<()> {
    let json = if COMPACT_JSON.load(Ordering::Relaxed) {
        serde_json::to_string(value)?
    } else {
        serde_json::to_string_pretty(value)?
    };
    println!("{json}");
    Ok(())
}
/// The capture pre-check for `vauss run`, in the background alongside the run's local frame
/// extraction: `<run>/capture_check.pending` until it ends, then capture_check.json; remote jobs hold
/// their upload for it (splat_core::remote, at most 20 s) and refuse it on a FAIL. A FAIL (unless
/// `ignore`) also asks the run to stop. Photo folders are not checked, and an error in the check
/// never blocks a run. The handle yields whether the check stopped the run.
fn start_capture_check(cfg: &Config, root: &Path, source: &Path, room: bool, ignore: bool) -> Option<thread::JoinHandle<bool>> {
    if source.is_dir() {
        return None;
    }
    let ffmpeg = cfg.resolve_ffmpeg().ok()?;
    let pending = root.join(splat_core::report::CAPTURE_PENDING);
    fs::write(&pending, b"").ok()?;
    let (root, source) = (root.to_owned(), source.to_owned());
    Some(thread::spawn(move || {
        let stopped = match splat_eval::precheck::check_video(&ffmpeg, &source, &splat_eval::precheck::CheckOptions { room }) {
            Ok(c) => {
                eprint!("{}", c.render());
                let mut value = serde_json::to_value(&c).unwrap_or_default();
                value["ignored"] = json!(ignore);
                let _ = splat_core::atomic_json(root.join(splat_core::report::CAPTURE_CHECK), &value);
                let fail = c.verdict == splat_eval::precheck::Verdict::Fail;
                if fail && ignore {
                    eprintln!("Capture check failed; going on anyway (--ignore-check).");
                } else if fail {
                    eprintln!("Stopping before upload: {} Re-film it (`vauss check --guide`), or pass --ignore-check to try anyway.", c.summary);
                    let _ = splat_core::atomic_json(root.join(splat_core::STOP_REQUEST), &json!({"requested_at":splat_core::utc_now(),"reason":"capture_check"}));
                }
                fail && !ignore
            }
            Err(e) => {
                eprintln!("Capture check skipped: {e:#}");
                false
            }
        };
        let _ = fs::remove_file(&pending);
        stopped
    }))
}
/// The standalone viewer [`write_viewer`] keeps next to exports/web.
const VIEWER_INDEX: &str = "exports/viewer/index.html";
/// Rebuild <run>/exports/viewer from the run's web package (a run without one has nothing to view).
/// Never fails the run: the package stays viewable with `vauss view`. SPLAT_EXPORT_VIEWER=0 skips it
/// (internal harnesses and bench runners; the folder is ~1.3x the package).
fn write_viewer(root: &Path) {
    if std::env::var("SPLAT_EXPORT_VIEWER").as_deref() == Ok("0") || !root.join("exports/web/scene.json").is_file() {
        return;
    }
    if let Err(e) = splat_webexport::viewer::run(&splat_webexport::viewer::ExportViewerArgs::for_run(root)) {
        eprintln!("Viewer: not written ({e:#}); `vauss view {}` shows the scene", root.display());
    }
}
/// The run's local exports once it has finished: the standalone viewer, the voxel game and the Blender pack. Each
/// reads the run (exports/web, report.json, its candidates) and writes only its own folder (viewer and game build in a
/// temp dir and rename it in). The viewer reads the game's voxel grid, so game then viewer run on one thread and the
/// Blender pack beside them: its downloads from fal (mesh.glb, 55.5 MB in 192 s on 2026-10-12's Lego run) no longer
/// wait behind the viewer's and the game's CPU work. Same outputs as one after the other.
fn write_exports(root: &Path) {
    let timed = |what: &str, f: &dyn Fn(&Path)| {
        let t0 = std::time::Instant::now();
        f(root);
        (what.to_owned(), t0.elapsed().as_secs_f64())
    };
    let t0 = std::time::Instant::now();
    let times: Vec<(String, f64)> = std::thread::scope(|s| {
        // The viewer's collision grid reuses the game's voxel world, so the game goes first on its thread.
        let parts = [s.spawn(|| vec![timed("game", &write_game), timed("viewer", &write_viewer)]),
            s.spawn(|| vec![timed("Blender pack", &blender::write_default)])];
        parts.into_iter().filter_map(|h| h.join().inspect_err(|_| eprintln!("Exports: a build panicked")).ok()).flatten().collect()
    });
    eprintln!("Exports: {} (side by side: {:.1}s)",
        times.iter().map(|(w, t)| format!("{w} {t:.1}s")).collect::<Vec<_>>().join(", "), t0.elapsed().as_secs_f64());
}
/// The voxel game [`write_game`] keeps next to exports/web.
const GAME_INDEX: &str = "exports/game/index.html";
/// Rebuild <run>/exports/game (first-person voxel game) from the run's web package. Never fails the run.
/// SPLAT_EXPORT_GAME=0 skips it (internal harnesses and bench runners).
fn write_game(root: &Path) {
    if std::env::var("SPLAT_EXPORT_GAME").as_deref() == Ok("0") || !root.join("exports/web/scene.json").is_file() {
        return;
    }
    if let Err(e) = splat_webexport::game::run(&splat_webexport::game::ExportGameArgs::for_run(root)) {
        eprintln!("Game: not written ({e:#}); `vauss export-game {}` retries", root.display());
    }
}
fn end_summary(root: &Path, report: &Value) {
    eprintln!(
        "Run {}: {}",
        report["run_id"].as_str().unwrap_or(""),
        report["status"].as_str().unwrap_or("unknown")
    );
    for (label, key) in [
        ("Splat", "splat"),
        ("Mesh", "mesh"),
        ("Mesh (whole scene)", "mesh_full_scene"),
        ("Report", "report_md"),
        ("Report JSON", "report_json"),
    ] {
        let path = report["deliverables"][key]
            .as_str()
            .map(|p| root.join(p).display().to_string())
            .unwrap_or_else(|| "missing".into());
        eprintln!("{label}: {path}");
    }
    let viewer = root.join(VIEWER_INDEX);
    if viewer.is_file() {
        eprintln!("Viewer: {} (double-click to open)", viewer.display());
    }
    let game = root.join(GAME_INDEX);
    if game.is_file() {
        eprintln!("Game: {} (double-click to play; WASD/mouse or a gamepad)", game.display());
    }
    let blender = root.join(blender::DIR).join("README.md");
    if blender.is_file() {
        eprintln!("Blender: {} (mesh, splat, voxels; see README.md)", blender.parent().unwrap_or(root).display());
    }
    if let Some(c) = report["cloud"].as_object() {
        eprintln!(
            "Cloud GPU cost: ${:.4} actual over {} fal jobs (estimated ${:.2})",
            c["actual_usd"].as_f64().unwrap_or(0.),
            c["gpu_jobs"].as_u64().unwrap_or(0),
            c["estimate_usd"].as_f64().unwrap_or(0.)
        );
    }
}
fn open(root: &Path, cfg: &Config, verify: bool, prepare: bool) -> Result<Run> {
    let cfg = cfg.for_run(root)?;
    let config: splat_core::RunConfig = serde_json::from_slice(&fs::read(root.join("run.json"))?)?;
    Run::open_with_registry(root, cfg.registry(&config.backend, prepare)?, verify)
}
/// Refining geometry keeps the original camera authority and never requires a new solve.
fn open_for_refine(root: &Path, cfg: &Config) -> Result<(Run, FixedOptions)> {
    let config: splat_core::RunConfig = serde_json::from_slice(&fs::read(root.join("run.json"))?)?;
    let previous: FixedOptions = fs::read(root.join("fixed.json")).ok()
        .map(|b| serde_json::from_slice(&b)).transpose()?.unwrap_or_default();
    cfg.for_run(root)?.preflight_with_camera_policy(&config.backend, previous.trust_poses)?;
    Ok((open(root, cfg, true, true)?, previous))
}

fn budget(text: Option<&str>) -> Result<Option<Duration>> {
    text.map(|s| {
        let (n, m) = match s.chars().last() {
            Some('s') => (&s[..s.len() - 1], 1.),
            Some('m') => (&s[..s.len() - 1], 60.),
            Some('h') => (&s[..s.len() - 1], 3600.),
            _ => (s, 1.),
        };
        let seconds = n.parse::<f64>()? * m;
        ensure!(
            seconds.is_finite() && seconds > 0. && seconds <= 86400. * 365.,
            "Invalid time budget"
        );
        Ok(Duration::from_secs_f64(seconds))
    })
    .transpose()
}
fn stop(run: &Run, wait_s: f64) -> Result<Value> {
    run.journal("run_stopped", json!({"status":"stopping"}))?;
    let mut results = Vec::new();
    for id in run.state()?.jobs.keys() {
        results.push(run.jobs().stop(id, wait_s)?);
    }
    let status = if results.iter().any(|r| r.status == Status::Running) {
        "stopping"
    } else {
        "cancelled"
    };
    run.journal("run_stopped", json!({"status":status}))?;
    write_report(run)?;
    Ok(json!({"run_id":run.run_id(),"status":status,"jobs":results}))
}
/// Grace before a busy run's lock holder is sent SIGTERM, then SIGKILL after STOP_KILL_S more.
const STOP_GRACE_S: f64 = 20.0;
const STOP_KILL_S: f64 = 10.0;

/// `vauss stop`: never waits on a busy run. Writes a stop request (the driver polls it); if the run is
/// idle, stops jobs and writes the report now; if a tool holds the lock, a detached helper escalates.
fn request_stop(root: &Path, cfg: &Config, wait_s: f64) -> Result<Value> {
    let root = fs::canonicalize(root)?;
    splat_core::atomic_json(root.join(splat_core::STOP_REQUEST), &json!({"requested_at":splat_core::utc_now(),"pid":std::process::id()}))?;
    if splat_core::supervisor_free(&root) {
        return stop(&open(&root, cfg, false, false)?, wait_s);
    }
    let mut helper = std::process::Command::new(std::env::current_exe()?);
    helper.arg("__stop_escalate").arg(&root).arg(wait_s.to_string());
    helper.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    splat_core::detach_command(&mut helper)?;
    helper.spawn().context("spawning stop helper")?;
    let run = Run::open_unlocked(&root)?;
    Ok(json!({"run_id":run.run_id(),"status":"stop_requested",
        "note":format!("A tool is busy; the run was asked to stop. If it has not stopped within {STOP_GRACE_S:.0} s, the busy process is terminated and the stop completes. Check with `vauss status`.")}))
}

/// Detached helper for a busy `vauss stop`: wait for the lock, else SIGTERM then SIGKILL its holder,
/// then complete the stop (journal, stop jobs, report).
fn stop_escalate(args: Vec<String>) -> Result<()> {
    let [root, wait_s] = args.as_slice() else { anyhow::bail!("usage: __stop_escalate <run_dir> <wait_s>") };
    let root = Path::new(root);
    let wait = |secs: f64| {
        let deadline = Instant::now() + Duration::from_secs_f64(secs);
        while Instant::now() < deadline {
            if splat_core::supervisor_free(root) {
                return true;
            }
            thread::sleep(Duration::from_millis(200));
        }
        splat_core::supervisor_free(root)
    };
    let holder = || fs::read_to_string(root.join(splat_core::SUPERVISOR_OWNER)).ok().and_then(|s| s.trim().parse::<u32>().ok());
    if !wait(STOP_GRACE_S) {
        if let Some(pid) = holder() {
            eprintln!("stop: terminating busy process {pid}");
            splat_core::terminate_pid(pid, false)?;
        }
        if !wait(STOP_KILL_S) {
            if let Some(pid) = holder() {
                splat_core::terminate_pid(pid, true)?;
            }
        }
    }
    let cfg = Config::load(None)?;
    stop(&open(root, &cfg, false, false)?, wait_s.parse().unwrap_or(60.))?;
    Ok(())
}

/// `vauss fill <run>`: the run's result splat, its SfM dataset and its frozen split (every training name the solve
/// registered, not the trainer's 150-frame subset).
fn fill_inputs(args: &splat_fill::FillArgs) -> Result<splat_fill::Inputs> {
    let root = &args.source;
    let run = Run::open_unlocked(root)?;
    let candidate = finished_candidate(root)?;
    let artifact = run.artifact(&candidate)?;
    let meta = |k: &str| artifact.meta.get(k).and_then(Value::as_str).map(str::to_owned).with_context(|| format!("{candidate} has no {k}"));
    let (sfm_id, split_id) = (meta("sfm_id")?, meta("split_id")?);
    let dataset = run.path(&sfm_id)?;
    let split = splat_core::split::load(&run, &split_id)?;
    let registered: std::collections::BTreeSet<String> = splat_brush::render::scene_cameras(&dataset, 64)?.into_iter().map(|(n, _)| n).collect();
    let keep = |names: &[String]| names.iter().filter(|n| registered.contains(*n)).cloned().collect::<Vec<_>>();
    let ply = fill_candidate_ply(root, &run.path(&candidate)?, args.ply.as_deref())?;
    Ok(splat_fill::Inputs { dataset, ply, train_names: keep(&split.train_names), heldout_names: keep(&split.heldout_names),
        out: args.out.clone().unwrap_or_else(|| root.join("fill")) })
}

/// The splat `vauss fill <run>` starts from: `explicit`, else the candidate's model (compressed or not), downloaded
/// from fal first when a `--cloud` run left only its stub here.
fn fill_candidate_ply(root: &Path, candidate_dir: &Path, explicit: Option<&Path>) -> Result<PathBuf> {
    let ply = match explicit {
        Some(p) => p.to_path_buf(),
        None => ["splat.ply", "splat.ply.zst", "checkpoint.ply", "checkpoint.ply.zst"].iter().map(|f| candidate_dir.join(f))
            .find(|p| p.is_file()).with_context(|| format!("{} has no splat.ply", candidate_dir.display()))?,
    };
    splat_core::fal::materialize(root, &ply)?;
    Ok(ply)
}

/// `vauss refine` default: the full-quality step count (the cloud MAX preset's minimum).
const REFINE_STEPS: u32 = 30_000;

/// Candidate a finished run ended with: agent/finish.json, else the newest archived finish-<seq>.json
/// (an in-place refine archives it first, so a failed refine can be rerun).
fn finished_candidate(root: &Path) -> Result<String> {
    let agent = root.join("agent");
    let mut finishes: Vec<(u64, PathBuf)> = fs::read_dir(&agent)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let seq = if name == "finish.json" { u64::MAX } else { name.strip_prefix("finish-")?.strip_suffix(".json")?.parse().ok()? };
            Some((seq, e.path()))
        })
        .collect();
    finishes.sort();
    finishes
        .iter()
        .rev()
        .filter_map(|(_, p)| serde_json::from_slice::<Value>(&fs::read(p).ok()?).ok())
        .find(|f| matches!(f["status"].as_str(), Some("succeeded" | "partial")))
        .and_then(|f| f["candidate_id"].as_str().map(str::to_owned))
        .with_context(|| format!("{} has no finished candidate to refine; finish the run first (`vauss resume`)", root.display()))
}

/// Copy a run for `vauss refine --run-dir`: artifacts are hard-linked (immutable once written; remote
/// outputs merge back by rename), the journal, state, job records and finish record are copied. The
/// driver's transcripts, exports, reports, locks and fal staging records stay behind.
fn fork_run(source: &Path, dest: &Path) -> Result<()> {
    ensure!(!dest.join("run.json").exists(), "Run already exists: {}", dest.display());
    ensure!(splat_core::supervisor_free(source), "{} is busy; wait for it to finish", source.display());
    fn copy(src: &Path, dst: &Path, link: bool) -> Result<()> {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let (from, to) = (entry.path(), dst.join(entry.file_name()));
            let ty = entry.file_type()?;
            if ty.is_symlink() {
                let target = fs::read_link(&from)?;
                #[cfg(unix)]
                std::os::unix::fs::symlink(&target, &to)?;
                #[cfg(not(unix))]
                if from.is_dir() { copy(&from, &to, link)? } else { fs::copy(&from, &to).map(|_| ())? }
            } else if ty.is_dir() {
                copy(&from, &to, link)?;
            } else if !link || fs::hard_link(&from, &to).is_err() {
                fs::copy(&from, &to).with_context(|| format!("copy {}", from.display()))?;
            }
        }
        Ok(())
    }
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let skip = matches!(name.as_str(), "agent" | "exports" | "remote-stage" | "remote-out" | "report.md" | "report.json"
            | "fal-staged.json" | "fal-exp-claimed" | splat_core::STOP_REQUEST | splat_core::SUPERVISOR_OWNER)
            || name.ends_with(".lock")
            || name.ends_with(".tmp");
        if skip {
            continue;
        }
        let (from, to) = (entry.path(), dest.join(&name));
        if entry.file_type()?.is_dir() {
            copy(&from, &to, name == "artifacts")?;
        } else {
            fs::copy(&from, &to).with_context(|| format!("copy {}", from.display()))?;
        }
    }
    // The finish record names the candidate to refine (and lets a failed refine of the copy rerun).
    if source.join("agent/finish.json").is_file() {
        fs::create_dir_all(dest.join("agent"))?;
        fs::copy(source.join("agent/finish.json"), dest.join("agent/finish.json"))?;
    }
    Ok(())
}

fn drive(
    run: Run,
    driver: Driver,
    model: Option<String>,
    effort: Option<String>,
    time_budget: Option<Duration>,
    fixed: FixedOptions,
) -> Result<Value> {
    drive_with(run, driver, model, effort, time_budget, fixed, None)
}

/// `drive`, or with `refine` (a finished candidate) the refine pipeline instead of the driver.
fn drive_with(
    run: Run,
    driver: Driver,
    model: Option<String>,
    effort: Option<String>,
    time_budget: Option<Duration>,
    fixed: FixedOptions,
    refine: Option<(&str, Option<u32>)>,
) -> Result<Value> {
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(run.root().join("driver.lock"))?;
    fs2::FileExt::try_lock_exclusive(&lock).context("A driver already owns this run")?;
    let prior_finish = run.root().join("agent/finish.json");
    if prior_finish.is_file() && run.state()?.status != "succeeded" {
        fs::rename(
            &prior_finish,
            run.root()
                .join("agent")
                .join(format!("finish-{}.json", run.state()?.seq)),
        )?;
    }
    fixed.configure_cloud_preset();
    let _progress = progress::Reporter::start(&run);
    splat_core::atomic_json(run.root().join("fixed.json"), &fixed)?;
    let _ = fs::remove_file(run.root().join(splat_core::STOP_REQUEST));
    run.journal(
        "run_started",
        json!({"status":"running","driver":if refine.is_some() { "refine" } else { driver.name() },"model":model}),
    )?;
    let interrupted = Arc::new(AtomicBool::new(false));
    let flag = interrupted.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancel_harness = cancelled.clone();
    let finished = Arc::new(AtomicBool::new(false));
    let done = finished.clone();
    let observer = run.clone();
    let monitor = thread::spawn(move || {
        let started = Instant::now();
        while !done.load(Ordering::SeqCst) {
            let externally_stopped = fs::read(observer.root().join("state.json"))
                .ok()
                .and_then(|v| serde_json::from_slice::<Value>(&v).ok())
                .is_some_and(|v| matches!(v["status"].as_str(), Some("stopping" | "cancelled")))
                || observer.root().join(splat_core::STOP_REQUEST).exists();
            if interrupted.load(Ordering::SeqCst)
                || time_budget.is_some_and(|d| started.elapsed() >= d)
                || externally_stopped
            {
                cancel_harness.store(true, Ordering::SeqCst);
                // A stop request may name its reason (the capture check's "capture_check").
                let requested = fs::read(observer.root().join(splat_core::STOP_REQUEST))
                    .ok()
                    .and_then(|v| serde_json::from_slice::<Value>(&v).ok())
                    .and_then(|v| v["reason"].as_str().map(str::to_owned));
                let reason = if interrupted.load(Ordering::SeqCst) {
                    "user_interrupt".to_owned()
                } else if externally_stopped {
                    requested.unwrap_or_else(|| "user_stop".into())
                } else {
                    "time_budget".to_owned()
                };
                let _ = observer.journal("stop_requested", json!({"reason":reason}));
                if let Err(e) = stop(&observer, 60.) {
                    eprintln!("Stop: {e:#}");
                }
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
    });
    let outcome = match driver {
        _ if refine.is_some() => {
            let (prior, cap_max) = refine.unwrap();
            pipeline::refine_pipeline(&run, &fixed, prior, cap_max)
        }
        Driver::Fixed => pipeline::fixed_pipeline(&run, &fixed),
        Driver::Claude | Driver::Codex => (|| {
            let outcome = splat_agent::run_driver(
                &run,
                driver.name().parse()?,
                &splat_agent::DriverOptions {
                    model,
                    effort,
                    cancel: Some(cancelled),
                    ..Default::default()
                },
            )?;
            let stopping = matches!(run.state()?.status.as_str(), "stopping" | "cancelled");
            if let Some(reason) = outcome.unavailable.as_deref().filter(|_| !stopping) {
                // One-click never dies on a missing agent login: the harness failed before its first
                // tool call, so nothing has run yet and the fixed pipeline takes the run instead.
                eprintln!("[agent] {} driver unavailable ({reason}); continuing with the fixed pipeline", driver.name());
                run.journal("driver_fallback", json!({"from":driver.name(),"to":"fixed","reason":reason}))?;
                return pipeline::fixed_pipeline(&run, &fixed);
            }
            if !stopping {
                run.journal("run_finished", json!({"status":outcome.status,"stop_reason":outcome.error,"finish":outcome.finish}))?;
            }
            write_report(&run)
        })(),
    };
    finished.store(true, Ordering::SeqCst);
    let _ = monitor.join();
    match outcome {
        Ok(_) => {
            if !export_finished(&run)? {
                // A run that trained but has nothing to view did not succeed (gimbal: preview and export
                // both failed, yet the run said succeeded).
                run.journal("run_finished", json!({"status":"failed","stop_reason":"no viewable result: web export failed"}))?;
                eprintln!("No viewable result: the web export failed (see export_failed in the journal)");
            }
            write_report(&run)
        }
        Err(error) => {
            if !matches!(run.state()?.status.as_str(), "stopping" | "cancelled") {
                run.journal(
                    "run_finished",
                    json!({"status":"failed","stop_reason":format!("{error:#}")}),
                )?;
            }
            write_report(&run)
        }
    }
}
/// A finished run ends with a viewable web package (scene.spz + scene.json) of its chosen candidate
/// in exports/web (`--cloud`: the package the train job wrote on fal). False when the run finished with a candidate but no viewable package came of it (every export failed): the
/// caller fails the run.
fn export_finished(run: &Run) -> Result<bool> {
    let Some(finish) = fs::read(run.root().join("agent/finish.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
    else {
        return Ok(true);
    };
    let Some(candidate) = finish["candidate_id"].as_str() else { return Ok(true) };
    if !matches!(finish["status"].as_str(), Some("succeeded" | "partial")) {
        return Ok(true);
    }
    export_candidate(run, candidate)?;
    Ok(run.root().join("exports/web/scene.json").is_file())
}

/// `__brush` worker. A finished train job also writes its web package to `<candidate>/web`, in the
/// same process that trained it (beside its held-out scoring, see `splat_brush::WEB_PACKAGE`), so a
/// `--cloud` run brings back the package instead of the full model's PLY (which stays on fal for
/// renders and downloads).
fn brush_worker(args: Vec<String>) -> Result<i32> {
    if args.first().map(String::as_str) == Some("train") {
        if let Some(job) = args.get(1).map(PathBuf::from) {
            let _ = splat_brush::WEB_PACKAGE.set(Box::new(move || {
                if let Err(e) = candidate_web(&job) {
                    eprintln!("web package: {e:#}");
                }
            }));
        }
    }
    splat_brush::worker_main(args)
}

/// `<candidate>/web` for the train job at `job`: previews without the .RAD stream (first view),
/// full models with it, except cloud models above the interactive budget: their capped refine's
/// package (with its own .RAD) replaces this one as exports/web, and the .RAD build (~70 s for 2M
/// splats on a 6-vCPU runner, single-threaded) sat between the end of training and "ready".
fn candidate_web(job: &Path) -> Result<()> {
    let job: Value = serde_json::from_slice(&fs::read(job)?)?;
    let out = PathBuf::from(job["out"].as_str().context("train job has no out")?);
    let dataset = PathBuf::from(job["dataset"].as_str().context("train job has no dataset")?);
    // Splat count from the PLY header: the package is built while the trainer still scores
    // the model, before train_summary.json exists.
    let replaced = job["cloud"].as_bool() == Some(true)
        && ply_vertex_count(&out.join("splat.ply"))
            .is_some_and(|n| n > splat_webexport::cap::INTERACTIVE_SPLATS as u64);
    let lod = if job["preview"].as_bool() == Some(true) || replaced { splat_webexport::LodMode::Skip } else { splat_webexport::LodMode::Inline };
    let options = splat_webexport::SceneOptions { lod, ..Default::default() };
    let started = std::time::Instant::now();
    splat_webexport::export_web_with_options(&out.join("splat.ply"), Some(&dataset), &out.join("web"), &options)?;
    eprintln!("web package: {:.3}s", started.elapsed().as_secs_f64());
    Ok(())
}

/// `element vertex N` from a PLY header.
fn ply_vertex_count(path: &Path) -> Option<u64> {
    use std::io::BufRead;
    let reader = std::io::BufReader::new(fs::File::open(path).ok()?);
    for line in reader.split(b'\n').take(64) {
        let line = String::from_utf8_lossy(&line.ok()?).trim().to_owned();
        if let Some(n) = line.strip_prefix("element vertex ") {
            return n.trim().parse().ok();
        }
        if line == "end_header" {
            break;
        }
    }
    None
}

/// Publish `<candidate>/web` (written by the train job, see [`brush_worker`]) as exports/web (hardlinked files,
/// with its collision/ once built). False when the candidate has none (older runs, failed package): the caller
/// exports instead.
fn copy_candidate_web(run: &Run, candidate: &str) -> Result<bool> {
    let _lock = web_lock(run)?;
    publish_candidate_web(run, candidate)
}

/// exports/web is replaced by the pipeline and by a detached collision build: one at a time.
fn web_lock(run: &Run) -> Result<fs::File> {
    use fs2::FileExt;
    fs::create_dir_all(run.root().join("exports"))?;
    let file = fs::OpenOptions::new().create(true).truncate(false).write(true).open(run.root().join("exports/.web.lock"))?;
    file.lock_exclusive()?;
    Ok(file)
}

/// The candidate whose package exports/web holds (written by [`publish_candidate_web`]).
const WEB_CANDIDATE: &str = ".candidate";

/// [`copy_candidate_web`] under its lock.
fn publish_candidate_web(run: &Run, candidate: &str) -> Result<bool> {
    let src = run.path(candidate)?.join("web");
    if !src.join("scene.json").is_file() {
        return Ok(false);
    }
    let out = run.root().join("exports/web");
    let tmp = run.root().join(format!("exports/.web.tmp.{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    link_tree(&src, &tmp)?;
    fs::write(tmp.join(WEB_CANDIDATE), candidate)?;
    let _ = fs::remove_dir_all(&out);
    fs::rename(&tmp, &out)?;
    Ok(true)
}

/// Hardlink the files of `src` (recursively, minus dot entries) into a new `dst` (the package is 30-45 MB and both copies are
/// read-only from here on); copy when linking fails, e.g. across filesystems.
fn link_tree(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let (ty, dest) = (entry.file_type()?, dst.join(entry.file_name()));
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue; // in-progress builds (.collision-*), markers
        } else if ty.is_dir() {
            link_tree(&entry.path(), &dest)?;
        } else if ty.is_file() && fs::hard_link(entry.path(), &dest).is_err() {
            fs::copy(entry.path(), &dest)?;
        }
    }
    Ok(())
}

/// Export before post-training reporting work. Also used by the fixed driver's ready event.
fn export_candidate(run: &Run, candidate: &str) -> Result<()> {
    // The fixed pipeline already exported this candidate before evaluation. Agent runs reach here
    // for the first time; refinements have a new candidate and overwrite the old web package.
    if run.events()?.iter().rev().find(|e| e.event == "exported").is_some_and(|e| e.data["candidate_id"] == candidate)
        && run.root().join("exports/web/scene.spz").is_file()
    {
        return Ok(());
    }
    let ply = run.path(candidate)?.join("splat.ply");
    let out = run.root().join("exports/web");
    let operation = format!("export:{candidate}");
    splat_core::progress::begin(run, "export", &operation, &json!({}))?;
    match copy_candidate_web(run, candidate) {
        Ok(true) => {
            run.journal("exported", json!({"candidate_id":candidate,"path":"exports/web","from":"candidate web"}))?;
            splat_core::progress::complete(run, &operation, Status::Succeeded)?;
            eprintln!("Web package: {}", out.display());
            spawn_collision(run, candidate);
            return Ok(());
        }
        Ok(false) => {}
        Err(e) => eprintln!("Web package from the candidate failed ({e:#}); exporting it here"),
    }
    // Published (viewable) first; the .RAD LoD stream is added to the package afterwards. A `--cloud` model
    // kept on fal comes down first. The candidate's own SfM: a run can hold more than one (the quick
    // first-view solve).
    let sfm = run.artifact(candidate)?.meta.get("sfm_id").and_then(Value::as_str).map(|id| run.path(id)).transpose()?;
    match splat_core::fal::materialize(run.root(), &ply)
        .and_then(|()| splat_webexport::export_web_staged(&ply, sfm.as_deref(), &out, &splat_webexport::SceneOptions::default()))
    {
        Ok((_, lod)) => {
            run.journal("exported", json!({"candidate_id":candidate,"path":"exports/web"}))?;
            splat_core::progress::complete(run, &operation, Status::Succeeded)?;
            eprintln!("Web package: {}", out.display());
            if let Some(Err(e)) = lod.map(|l| l.finish()) {
                eprintln!("Web export: no LoD stream ({e:#}); the viewer uses scene.spz");
            }
            spawn_collision(run, candidate);
        }
        Err(e) => {
            run.journal("export_failed", json!({"candidate_id":candidate,"error":format!("{e:#}")}))?;
            splat_core::progress::complete(run, &operation, Status::Failed)?;
            eprintln!("Web export failed: {e:#}");
        }
    }
    Ok(())
}

/// Add collision/ to `candidate`'s package in a detached `vauss export-collision` (~30 s of CPU): nothing
/// viewable waits for it. The package is the candidate's own `<candidate>/web` when it has one (exports/web
/// is replaced by later candidates, e.g. the interactive refine), else exports/web. When it is done,
/// scene.json gains `collision` and the run journals `collision_ready {candidate_id, path}` (or
/// `collision_failed`); see [`journal_collision`].
pub(crate) fn spawn_collision(run: &Run, candidate: &str) {
    let no_mesh = fs::read(run.root().join("fixed.json")).ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .is_some_and(|options| options["no_mesh"] == true);
    if no_mesh { return; }
    let own = run.path(candidate).map(|p| p.join("web")).ok().filter(|w| w.join("scene.json").is_file());
    let out = own.unwrap_or_else(|| run.root().join("exports/web"));
    let log = run.root().join(format!("exports/collision-{candidate}.log"));
    // This run's own model and capture: a package built on fal records fal-side source paths.
    let ply = run.path(candidate).map(|p| p.join("splat.ply"));
    let sfm = run.artifact(candidate).ok().and_then(|a| a.meta.get("sfm_id").and_then(Value::as_str).map(str::to_string))
        .and_then(|id| run.path(&id).ok());
    let spawned = std::env::current_exe().and_then(|exe| {
        let file = fs::File::create(&log)?;
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("export-collision").arg(&out).arg("--run").arg(run.root()).arg("--candidate").arg(candidate);
        if let Ok(ply) = &ply { cmd.arg("--ply").arg(ply); }
        if let Some(sfm) = &sfm { cmd.arg("--data").arg(sfm); }
        cmd.stdin(std::process::Stdio::null()).stdout(file.try_clone()?).stderr(file);
        // Its own process group (Windows: no console), so stopping `vauss run` does not stop it.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        #[cfg(windows)]
        {
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            const DETACHED_PROCESS: u32 = 0x0000_0008;
            std::os::windows::process::CommandExt::creation_flags(&mut cmd, CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS);
        }
        cmd.spawn()
    });
    match spawned {
        Ok(child) => eprintln!("Collision: building in the background (pid {}, log {})", child.id(), log.display()),
        Err(e) => eprintln!("Collision: not started ({e})"),
    }
}

#[derive(Args)]
struct ExportCollisionCmd {
    #[command(flatten)]
    args: splat_webexport::collision::ExportCollisionArgs,
    /// Run whose journal records the result (`collision_ready` / `collision_failed`); set by a run's own export.
    #[arg(long, hide = true, requires = "candidate")]
    run: Option<PathBuf>,
    /// The run's candidate the package was exported from.
    #[arg(long, hide = true, requires = "run")]
    candidate: Option<String>,
}

#[derive(Args)]
struct OffpathCmd {
    /// LABEL=RUN_DIR[#CANDIDATE] (a finished run: its best or the named candidate and that candidate's SfM;
    /// a PLY kept on fal is read from fal /data inside fal jobs, else downloaded) or LABEL=PLY,SFM_DIR (SfM
    /// dataset or sparse model dir). Repeat for side-by-side arms (the first arm picks the anchor frames).
    #[arg(long = "arm", required = true)]
    arms: Vec<String>,
    #[arg(long)]
    out: PathBuf,
    /// Anchor frames spread along the capture path (5 views each: on path, left, right, turned left, turned right).
    #[arg(long, default_value_t = 6)]
    anchors: usize,
    /// Phase of the anchor spread (same seed + same SfM = same views).
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0.5)]
    shift_m: f32,
    #[arg(long, default_value_t = 30.0)]
    yaw_deg: f32,
    /// Assumed median distance from the camera to what it films; sets metres from the SfM point depths.
    #[arg(long, default_value_t = 2.5)]
    room_depth_m: f32,
    /// SfM units per metre for every arm, instead of the room-depth estimate.
    #[arg(long)]
    units_per_metre: Option<f32>,
    /// Covered pixels nearer than this count as near floaters.
    #[arg(long, default_value_t = 0.3)]
    near_m: f32,
    #[arg(long, default_value_t = 960)]
    max_resolution: u32,
    /// Tile width on the per-anchor sheets.
    #[arg(long, default_value_t = 640)]
    tile: u32,
}

fn offpath(cmd: &OffpathCmd) -> Result<Value> {
    let arms = cmd.arms.iter().map(|a| splat_brush::offpath::parse_arm(a)).collect::<Result<Vec<_>>>()?;
    let opts = splat_brush::offpath::Options {
        anchors: cmd.anchors, seed: cmd.seed, shift_m: cmd.shift_m, yaw_deg: cmd.yaw_deg, room_depth_m: cmd.room_depth_m,
        units_per_metre: cmd.units_per_metre, near_m: cmd.near_m, max_resolution: cmd.max_resolution, tile_width: cmd.tile,
    };
    let result = splat_brush::offpath::run(&arms, &opts, &cmd.out)?;
    Ok(json!({"offpath": cmd.out.join("offpath.json"), "contact": result["contact"], "sheets": result["sheets"], "arms": result["arms"]}))
}

fn export_collision(args: &splat_webexport::collision::ExportCollisionArgs) -> Result<Value> {
    // A `--cloud` run's model may be a stub of a file kept on fal: collision reads the full PLY.
    let scene = fs::read(args.input.join("scene.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    let ply = match (&args.ply, scene.as_ref().and_then(|s| s["source"]["ply"].as_str())) {
        (Some(p), _) => Some(p.clone()),
        (None, Some(p)) => Some(PathBuf::from(p)),
        (None, None) => splat_webexport::input_files(&args.input, args.data.as_deref()).ok().map(|(ply, _)| ply),
    };
    if let Some(ply) = ply {
        let run_root = ply.canonicalize().ok().and_then(|p| p.ancestors()
            .find(|d| d.join("artifacts").is_dir() && d.join("run.json").exists()).map(Path::to_path_buf));
        if let Some(root) = run_root { splat_core::fal::materialize(&root, &ply)?; }
    }
    splat_webexport::collision::run(args)
}

/// Journal a run's collision build: `collision_ready {candidate_id, path}` (`path`: the package, relative to the
/// run root, whose collision/ and scene.json `collision` are done) or `collision_failed {candidate_id, error}`.
/// A candidate package's collision is also carried into exports/web while that still publishes the candidate.
fn journal_collision(root: &Path, candidate: &str, args: &splat_webexport::collision::ExportCollisionArgs,
                     result: &Result<Value>) -> Result<()> {
    let run = Run::open_unlocked(root)?;
    let failed = |error: String| {
        eprintln!("Collision failed: {error}");
        run.journal("collision_failed", json!({"candidate_id":candidate,"error":error}))
    };
    if let Err(e) = result {
        return failed(format!("{e:#}"));
    }
    let pkg = fs::canonicalize(splat_webexport::collision::package_dir(args))?;
    let Ok(path) = pkg.strip_prefix(run.root()).map(Path::to_path_buf) else {
        return failed(format!("{} is outside the run", pkg.display()));
    };
    let lock = web_lock(&run)?;
    let published = fs::read_to_string(run.root().join("exports/web").join(WEB_CANDIDATE)).ok();
    if path == Path::new("exports/web") {
        // Built in place: valid only while exports/web still holds this candidate's model.
        if published.is_some_and(|c| c != candidate) {
            return failed("exports/web was replaced by another candidate during the build".into());
        }
    } else if published.as_deref() == Some(candidate) {
        if let Err(e) = publish_candidate_web(&run, candidate) {
            eprintln!("Collision: exports/web not updated ({e:#})");
        }
    }
    drop(lock);
    eprintln!("Collision ready: {}", pkg.display());
    run.journal("collision_ready", json!({"candidate_id":candidate,"path":path}))
}

fn logs(root: &Path, cfg: &Config, follow: bool) -> Result<()> {
    let run = open(root, cfg, false, false)?;
    let interrupted = Arc::new(AtomicBool::new(false));
    if follow {
        let flag = interrupted.clone();
        ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))?;
    }
    let mut offsets = BTreeMap::new();
    loop {
        let mut paths = vec![run.root().join("journal.jsonl")];
        for entry in fs::read_dir(run.root().join("jobs"))? {
            let entry = entry?;
            for name in ["stdout.log", "stderr.log", "runner.log"] {
                let p = entry.path().join(name);
                if p.is_file() {
                    paths.push(p);
                }
            }
        }
        paths.sort();
        for path in paths {
            let mut file = fs::File::open(&path)?;
            let offset = offsets.entry(path.clone()).or_insert(0);
            if file.metadata()?.len() < *offset {
                *offset = 0;
            }
            file.seek(SeekFrom::Start(*offset))?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            *offset = file.stream_position()?;
            if !bytes.is_empty() {
                println!(
                    "[{}]\n{}",
                    path.strip_prefix(run.root())?.display(),
                    String::from_utf8_lossy(&bytes)
                );
            }
        }
        if !follow || interrupted.load(Ordering::SeqCst) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }
}
/// The engine-spec 0.4 `doctor --json` envelope: one check per required (and optional) helper.
fn engine_doctor(report: &Value) -> Value {
    let readiness = &report["readiness"];
    let mut checks = vec![];
    for (group, optional) in [("required", false), ("optional", true)] {
        for (item, entry) in readiness[group].as_object().into_iter().flatten() {
            let mut check = json!({"item": item, "ok": entry["status"] != "missing", "optional": optional});
            for key in ["status", "source", "path", "fix", "error"] {
                if !entry[key].is_null() {
                    check[key] = entry[key].clone();
                }
            }
            if entry["status"] == "missing" {
                if let Some(fix) = entry["fix"].as_str() {
                    check["hint"] = json!(fix);
                }
            }
            checks.push(check);
        }
    }
    let ready = readiness["ready"].as_bool().unwrap_or(false);
    json!({
        "ok": ready,
        "result": {
            "engine": "splat",
            "version": env!("CARGO_PKG_VERSION"),
            "spec": "0.4",
            "requires": {},
            "ok": ready,
            "summary": if ready { "ready" } else { "requires-setup" },
            "checks": checks,
            "platform": report["platform"],
            "readiness": readiness,
        }
    })
}
/// Default run directory of `vauss run` without --run-dir.
fn new_run_root() -> PathBuf {
    PathBuf::from("runs").join(format!("run_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]))
}
fn execute(cli: Cli) -> Result<i32> {
    COMPACT_JSON.store(cli.json, Ordering::Relaxed);
    if cli.allow_low_resolution {
        // Single-threaded here; agent drivers and their MCP server children inherit it.
        unsafe { std::env::set_var(splat_core::resolution::OVERRIDE_ENV, "1") };
    }
    let cfg = Config::load(cli.config.as_deref())?;
    match cli.command {
        Commands::Doctor { backend, run_dir } => {
            let mut report = splat_core::doctor::doctor(&backend, run_dir.as_deref())?;
            report["readiness"] = cfg.readiness(&backend)?;
            if backend == "spirula" {
                if let Some(path) = report["readiness"]["required"]["spirula"]["path"].as_str() {
                    report["binaries"]["spirula"] = splat_core::doctor::binary_fingerprint(
                        Path::new(path),
                        &[&["--help"], &["sam", "--help"], &["sfm", "auto", "--help"], &["train", "--help-all"], &["mesh", "--help"]],
                    );
                }
            }
            report["distribution"] = cfg.offline_assets()?;
            report["backend_capabilities"] = cfg.registry(&backend, false)?.backend(&backend)?.capabilities();
            if let Some(root) = run_dir {
                splat_core::atomic_json(root.join("capabilities.json"), &report)?;
            }
            if cli.json {
                // `doctor --json` is the engine-spec 0.4 doctor Voyager reads (its last stdout line).
                print(&engine_doctor(&report))?;
            } else {
                print(&report)?;
            }
        }
        Commands::Init {
            source,
            goal,
            run_dir,
            backend,
        } => {
            cfg.registry(&backend, false)?.backend(&backend)?;
            let run = Run::create(run_dir, source, goal, backend)?;
            splat_core::atomic_json(run.root().join("toolchain.json"), &cfg)?;
            let mut value = serde_json::to_value(run.config())?;
            value["run_dir"] = json!(run.root());
            print(&value)?;
        }
        Commands::Tool {
            name,
            run_dir,
            input,
        } => {
            let content = if let Some(path) = input.strip_prefix('@') {
                fs::read_to_string(path)?
            } else if input == "-" {
                let mut data = String::new();
                std::io::stdin().read_to_string(&mut data)?;
                data
            } else {
                input
            };
            let prepare = splat_contracts::BACKEND_TOOLS.contains(&name.as_str())
                || splat_contracts::SHOT_TOOLS.contains(&name.as_str())
                || splat_contracts::EDIT_TOOLS.contains(&name.as_str())
                || name == "run_default_pipeline";
            let run = open(&run_dir, &cfg, true, prepare)?;
            // A `--cloud` run's GPU jobs stay on fal when started from `vauss tool`, as under `vauss run`.
            splat_core::remote::resume(&run_dir)?;
            let r = match serde_json::from_str(&content) {
                Ok(value) if name == "run_default_pipeline" => pipeline::run_default_pipeline(&run, value)?,
                Ok(value) => dispatch(&run, &name, value),
                Err(e) => {
                    let mut r = dispatch(&run, &name, Value::Null);
                    if let Some(error) = &mut r.error {
                        error.message = format!("Invalid JSON: {e}");
                    }
                    r
                }
            };
            print(&r)?;
            if matches!(r.status, Status::Failed | Status::Unsupported) {
                return Ok(5);
            }
        }
        Commands::Tools => {
            if cli.json {
                print(
                    &tool_schemas()
                        .into_iter()
                        .map(|(name, _, schema)| (name, schema))
                        .collect::<BTreeMap<_, _>>(),
                )?;
            } else {
                for (name, description, _) in tool_schemas() {
                    println!("{name}\t{description}");
                }
            }
        }
        Commands::Run {
            source,
            goal,
            backend,
            driver,
            model,
            effort,
            run_dir,
            non_interactive: _,
            time_budget,
            cloud,
            fetch_all,
            room,
            ignore_check,
            fixed,
            preset,
        } => {
            if splat_core::supervise::wanted(cloud) {
                // The run outlives this process (agent session restarts): it only follows the supervisor.
                let root = run_dir.clone().unwrap_or_else(new_run_root);
                let mut args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
                if run_dir.is_none() {
                    args.extend(["--run-dir".into(), root.clone().into_os_string()]);
                }
                return splat_core::supervise::run_detached(&root, &args);
            }
            if cloud {
                let tier = std::env::var("SPLAT_FAL_TIER").unwrap_or_else(|_| "product".into());
                splat_core::fal::warm_upload_stage(&tier);
            }
            let room = room || splat_eval::precheck::goal_is_room(&goal);
            let fixed = fixed.apply(preset.apply(FixedOptions::default()));
            fixed.configure_cloud_preset();
            let fast = fixed.is_fast();
            if fetch_all {
                // SAFETY: main thread, before the driver, MCP server or job workers start (as remote::resume);
                // remote::enable records it in cloud.json.
                unsafe { std::env::set_var(splat_core::remote::FETCH_ALL_ENV, "1") };
            }
            cfg.preflight_with_camera_policy(&backend, fixed.trust_poses)?;
            let registry = cfg.registry(&backend, true)?;
            registry.backend(&backend)?;
            let root = run_dir.unwrap_or_else(new_run_root);
            let run = Run::create_with_options(
                root,
                source,
                goal,
                backend,
                driver.name(),
                model.clone(),
            )?
            .with_registry(registry);
            splat_core::atomic_json(run.root().join("toolchain.json"), &cfg)?;
            eprintln!("Run: {}", run.root().display());
            let check = start_capture_check(&cfg, run.root(), run.source(), room, ignore_check);
            if cloud {
                let c = splat_core::remote::enable(run.root())?;
                eprintln!(
                    "Cloud GPU: fal {}, preset {}: estimated ${:.2} (up to ${:.2} with retries), about {} min",
                    c["gpu"].as_str().unwrap_or("?"),
                    c["training"].as_str().unwrap_or("default"),
                    c["estimate_usd"].as_f64().unwrap_or(0.),
                    c["estimate_high_usd"].as_f64().unwrap_or(0.),
                    c["estimate_minutes"].as_u64().unwrap_or(0)
                );
            }
            let root = run.root().to_owned();
            let report = drive(
                run,
                driver,
                model,
                effort,
                budget(time_budget.as_deref())?,
                fixed,
            );
            splat_core::remote::release(&root);
            let report = report?;
            write_exports(&root);
            end_summary(&root, &report);
            print(&report)?;
            if check.is_some_and(|h| h.join().unwrap_or(false)) {
                return Ok(6);
            }
            if report["stop_reason"] == "user_interrupt" {
                return Ok(130);
            }
            if report["status"] != "succeeded" && !(fast && report["status"] == "partial") {
                return Ok(5);
            }
        }
        Commands::Check { source, room, guide } => {
            if guide {
                print!("{}", splat_eval::precheck::FILMING_GUIDE);
                return Ok(0);
            }
            let source = source.context("a video to check")?;
            let ffmpeg = cfg.resolve_ffmpeg()?;
            let c = splat_eval::precheck::check_input(&ffmpeg, &source, &splat_eval::precheck::CheckOptions { room })?
                .context("photo folders are not checked; the capture check is for videos")?;
            eprint!("{}", c.render());
            print(&c)?;
            if c.verdict == splat_eval::precheck::Verdict::Fail {
                return Ok(6);
            }
        }
        Commands::Status { run } => {
            // Lock-free: answers immediately even while a tool holds the run's supervisor lock.
            let run = Run::open_unlocked(run.root())?;
            let mut state = serde_json::to_value(run.state_unlocked()?)?;
            state["progress"] = splat_core::progress::snapshot(state["progress"].clone(), state["status"].as_str().unwrap_or("running"));
            state["run_id"] = json!(run.run_id());
            print(&state)?;
        }
        Commands::Logs { run, follow } => logs(run.root(), &cfg, follow)?,
        Commands::Resume {
            run,
            driver,
            model,
            effort,
            time_budget,
            fixed,
        } => {
            let root = run.root();
            if splat_core::supervise::wanted(root.join("cloud.json").is_file()) {
                // As `vauss run --cloud`: the resumed run outlives this process.
                return splat_core::supervise::run_detached(root, &std::env::args_os().skip(1).collect::<Vec<_>>());
            }
            // Check helpers before open: preparing the backend would otherwise fail with a raw error.
            let config: splat_core::RunConfig = serde_json::from_slice(&fs::read(root.join("run.json"))?)?;
            let finished = fs::read(root.join("state.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                .is_some_and(|s| s["status"] == "succeeded");
            if !finished {
                let saved: FixedOptions = fs::read(root.join("fixed.json")).ok()
                    .map(|b| serde_json::from_slice(&b)).transpose()?.unwrap_or_default();
                cfg.for_run(root)?.preflight_with_camera_policy(&config.backend, fixed.trust_poses.unwrap_or(saved.trust_poses))?;
            }
            let run = open(root, &cfg, true, true)?;
            splat_core::remote::resume(root)?;
            if run.state()?.status == "succeeded" {
                let report = write_report(&run)?;
                write_exports(run.root());
                end_summary(run.root(), &report);
                print(&report)?;
                return Ok(0);
            }
            let driver = driver.unwrap_or(match run.config().driver.as_str() {
                "claude" => Driver::Claude,
                "codex" => Driver::Codex,
                _ => Driver::Fixed,
            });
            let model = model.or(run.config().model.clone());
            let previous = if run.root().join("fixed.json").is_file() {
                serde_json::from_slice(&fs::read(run.root().join("fixed.json"))?)?
            } else {
                FixedOptions::default()
            };
            let fixed = fixed.apply(previous);
            let root = run.root().to_owned();
            let report = drive(
                run,
                driver,
                model,
                effort,
                budget(time_budget.as_deref())?,
                fixed,
            );
            splat_core::remote::release(&root);
            let report = report?;
            write_exports(&root);
            end_summary(&root, &report);
            print(&report)?;
            if report["stop_reason"] == "user_interrupt" {
                return Ok(130);
            }
            if report["status"] != "succeeded" {
                return Ok(5);
            }
        }
        Commands::Refine { run: source, run_dir, steps, cap_max, cloud, fetch_all, preset } => {
            if fetch_all {
                // SAFETY: main thread, before the driver, MCP server or job workers start (as remote::resume).
                unsafe { std::env::set_var(splat_core::remote::FETCH_ALL_ENV, "1") };
            }
            let prior = finished_candidate(&source)?;
            let root = match run_dir {
                Some(dir) => {
                    fork_run(&source, &dir)?;
                    eprintln!("Refining a copy of {} in {}", source.display(), dir.display());
                    dir
                }
                None => source,
            };
            let (run, previous) = open_for_refine(&root, &cfg)?;
            let finish = run.root().join("agent/finish.json");
            if finish.is_file() {
                fs::rename(&finish, run.root().join(format!("agent/finish-{}.json", run.state()?.seq)))?;
            }
            // Refinement has its own step budget; a prior --fast does not opt this command into it.
            let meshed = run.state()?.operations.values().any(|op| op["tool"] == "extract_mesh" && op["result"]["status"] == "succeeded");
            let prior_max_edge = previous.max_edge;
            let fixed = FixedOptions { steps: REFINE_STEPS, no_mesh: !meshed,
                preset: None, steps_explicit: false, cap_max_explicit: false, ..previous };
            let mut fixed = FixedOverrides { steps, cap_max, ..Default::default() }.apply(preset.apply(fixed));
            if fixed.is_fast() {
                fixed.resolution_divisor = prior_max_edge.div_ceil(800).clamp(1, 8).max(fixed.resolution_divisor);
            }
            fixed.configure_cloud_preset();
            if cloud {
                let c = splat_core::remote::enable(run.root())?;
                eprintln!("Cloud GPU: fal {}, preset {}", c["gpu"].as_str().unwrap_or("?"), c["training"].as_str().unwrap_or("default"));
            } else {
                splat_core::remote::resume(run.root())?;
            }
            let fast = fixed.is_fast();
            let cap_max = cap_max.or_else(|| fixed.cap_max_explicit.then_some(fixed.cap_max));
            eprintln!("Refine: {prior} -> {} steps on the run's SfM and split{}", fixed.steps, if !fixed.no_mesh { ", then mesh" } else { "" });
            let root = run.root().to_owned();
            let report = drive_with(run, Driver::Fixed, None, None, None, fixed, Some((&prior, cap_max)));
            splat_core::remote::release(&root);
            let report = report?;
            write_exports(&root);
            end_summary(&root, &report);
            print(&report)?;
            if report["stop_reason"] == "user_interrupt" {
                return Ok(130);
            }
            if report["status"] != "succeeded" && !(fast && report["status"] == "partial") {
                return Ok(5);
            }
        }
        Commands::Stop { run, wait_s } => print(&request_stop(run.root(), &cfg, wait_s)?)?,
        Commands::Serve { run_dir } => {
            // Restore cloud routing before the MCP runtime or any workers start.
            splat_core::remote::resume(&run_dir)?;
            let run = Arc::new(open(&run_dir, &cfg, false, true)?);
            let context = run.clone();
            splat_mcp::serve(
                run,
                Arc::new(move |name, args| {
                    if name == "run_default_pipeline" {
                        pipeline::run_default_pipeline(&context, args)
                    } else {
                        Ok(dispatch(&context, name, args))
                    }
                }),
            )?;
        }
        Commands::ExportViewer(args) => print(&splat_webexport::viewer::run(&args)?)?,
        Commands::ExportGame(args) => print(&game_source::run(args)?)?,
        Commands::ExportBlender(args) => print(&blender::run(&args)?)?,
        Commands::ExportCollision(cmd) => {
            let result = export_collision(&cmd.args);
            if let (Some(root), Some(candidate)) = (&cmd.run, &cmd.candidate) {
                journal_collision(root, candidate, &cmd.args, &result)?;
            }
            print(&result?)?
        }
        Commands::Offpath(cmd) => print(&offpath(&cmd)?)?,
        Commands::Fill(args) => {
            let inputs = if args.source.join("run.json").is_file() { fill_inputs(&args)? } else { splat_fill::Inputs::from_dataset(&args)? };
            print(&splat_fill::run(&args, &inputs, &std::env::current_exe()?)?)?
        }
        Commands::ExportWeb(args) => {
            // A `--cloud` run's model (and mesh) may be stubs of files kept on fal: download what the export reads.
            let run_root = args.input.canonicalize().ok().and_then(|p| p.ancestors()
                .find(|d| d.join("artifacts").is_dir() && d.join("run.json").exists()).map(Path::to_path_buf));
            if let Some(root) = run_root {
                if let Ok((ply, mesh)) = splat_webexport::input_files(&args.input, args.data.as_deref()) {
                    splat_core::fal::materialize(&root, &ply)?;
                    if let Some(m) = mesh {
                        splat_core::fal::materialize(&root, &m)?;
                    }
                }
            }
            print(&splat_webexport::run(&args)?)?
        }
        Commands::Pull { run } => {
            let n = splat_core::fal::materialize_all(&run)?;
            eprintln!("{n} files downloaded from fal into {}", run.display());
        }
        Commands::Shots(command) => {
            let encode = matches!(&command, splat_shots::cli::Cmd::Shot {..} | splat_shots::cli::Cmd::Stabilize {..} | splat_shots::cli::Cmd::Generate {..});
            let ffmpeg = if encode { cfg.resolve_ffmpeg()? } else { cfg.ffmpeg.clone().unwrap_or_else(||"ffmpeg".into()) };
            let blender = cfg.blender.clone().unwrap_or_else(splat_shots::blender::blender_exe);
            splat_shots::cli::run_with(command, &splat_shots::encode::Ffmpeg::new(ffmpeg), &blender)?;
        },
        Commands::View(args) => splat_assets::server::run(args, cli.json)?,
        Commands::Assets(command) => splat_assets::cli::run(command)?,
        Commands::SurfelFixture { request } => splat_brush::fixture::run(&request)?,
        Commands::Effect(cmd) => return effect::run(cmd, cli.json),
        Commands::Fetch { asset, allow_gpl } => {
            let path = cfg.fetcher()?.ensure(
                &asset,
                if allow_gpl {
                    splat_fetch::LicenseConsent::AllowGpl
                } else {
                    splat_fetch::LicenseConsent::Default
                },
            )?;
            print(&json!({"asset":asset,"path":path}))?;
        }
    }
    Ok(0)
}
/// Worker for core JOB_TOOLS: `__tool_job <run_dir> <tool> <input.json> <result.json>`.
fn tool_job(args: Vec<String>) -> Result<()> {
    let [root, name, input, output] = args.as_slice() else {
        anyhow::bail!("usage: __tool_job <run_dir> <tool> <input.json> <result.json>");
    };
    let cfg = Config::load(None)?;
    let run = open(Path::new(root), &cfg, false, true)?;
    splat_core::run_tool_job(&run, name, Path::new(input), Path::new(output))?;
    // ToolResult is atomically written, including structured tool failures. ORT/GPU teardown
    // can abort after a complete evaluation ("corrupted double-linked list"), in Rust drops and in
    // the C exit handlers `process::exit` still runs. This dedicated worker has no further work.
    use std::io::Write;
    std::io::stdout().flush()?;
    std::io::stderr().flush()?;
    splat_core::supervise::exit_now(0)
}
/// Windows double-click: no arguments and the only process on its console (Explorer made one).
#[cfg(windows)]
fn launched_from_explorer() -> bool {
    let mut pids = [0u32; 2];
    // SAFETY: the buffer holds 2 u32s and its length is passed.
    std::env::args_os().len() == 1 && unsafe { windows_sys::Win32::System::Console::GetConsoleProcessList(pids.as_mut_ptr(), 2) } == 1
}
#[cfg(not(windows))]
fn launched_from_explorer() -> bool {
    false
}
fn main() {
    splat_core::supervise::take_env();
    if !std::env::args().nth(1).is_some_and(|a| a.starts_with("__")) {
        // Dev-host settings are off unless configured; when on, this says so once on stderr.
        splat_contracts::dev::config();
    }
    let outcome = if let Some(result) = splat_core::handle_internal_command() {
        result
    } else if std::env::args().nth(1).as_deref() == Some("__video_replay_cleanup") {
        std::env::args().nth(2).context("missing video replay manifest")
            .and_then(|p| splat_eval::frameplan::cleanup_replay_manifest(std::path::Path::new(&p))).map(|()| 0)
    } else if std::env::args().nth(1).as_deref() == Some("__video_replay_stage") {
        std::env::args().nth(2).context("missing video replay manifest")
            .and_then(|p| splat_eval::frameplan::stage_replay_manifest(std::path::Path::new(&p))).map(|()| 0)
    } else if std::env::args().nth(1).as_deref() == Some("__video_replay") {
        std::env::args().nth(2).context("missing video replay manifest")
            .and_then(|p| splat_eval::frameplan::replay_manifest(std::path::Path::new(&p))).map(|()| 0)
    } else if std::env::args().nth(1).as_deref() == Some("__fal_publish") {
        // Dev tooling, after installing a build: put it in fal's binary cache before any --cloud run needs it.
        std::env::args().skip(2).try_for_each(|p| splat_core::fal::publish_file(Path::new(&p)).map(drop)).map(|()| 0)
    } else if std::env::args().nth(1).as_deref() == Some("__brush") {
        brush_worker(std::env::args().skip(2).collect())
    } else if std::env::args().nth(1).as_deref() == Some("__stop_escalate") {
        stop_escalate(std::env::args().skip(2).collect()).map(|()| 0)
    } else if std::env::args().nth(1).as_deref() == Some("__frame_arms") {
        // Dev only: frame-selection ablation arms (M6).
        splat_eval::frame_arms::run(&std::env::args().skip(2).collect::<Vec<_>>()).map(|()| 0)
    } else if std::env::args().nth(1).as_deref() == Some("__align_sfm") {
        // Dev/ops: replay the full-onto-quick SfM alignment on two capture dirs (see sfm_align).
        sfm_align::align_dirs(&std::env::args().skip(2).collect::<Vec<_>>()).map(|()| 0)
    } else if std::env::args().nth(1).as_deref() == Some("__effect_remote") {
        std::env::args().nth(2).context("missing job dir").and_then(|d| effect::remote_child(Path::new(&d)))
    } else if std::env::args().nth(1).as_deref() == Some("__tool_job") {
        tool_job(std::env::args().skip(2).collect()).map(|()| 0)
    } else if std::env::args().nth(1).as_deref() == Some("__shots") {
        splat_shots::worker_main(&std::env::args().skip(2).collect::<Vec<_>>()).map(|()| 0)
    } else if launched_from_explorer() {
        splat_assets::server::run_with(splat_assets::ViewArgs::default(), splat_assets::server::Output::Banner).map(|()| 0)
    } else {
        execute(Cli::parse())
    };
    let code = match outcome {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error:#}");
            1
        }
    };
    splat_core::supervise::record_exit(code);
    std::process::exit(code);
}

#[cfg(test)]
mod fill_tests {
    use super::*;

    #[test]
    fn fill_finds_compressed_only_candidates() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("checkpoint.ply.zst"), b"\x28\xb5\x2f\xfd").unwrap();
        let ply = fill_candidate_ply(dir.path(), dir.path(), None).unwrap();
        assert_eq!(ply, dir.path().join("checkpoint.ply.zst"));
    }

    #[test]
    fn fill_materializes_a_cloud_stub_instead_of_reading_it() {
        // A `--cloud` run's model left on fal: the stub must go through fal::materialize (here: no staging ledger, so
        // it fails loudly) rather than reach the PLY reader.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("splat.ply"), format!("splat-fal-remote v1 {} 1234\n", "0".repeat(64))).unwrap();
        let err = fill_candidate_ply(dir.path(), dir.path(), None).unwrap_err();
        assert!(format!("{err:#}").contains("no longer on fal"), "{err:#}");
    }
}


#[cfg(test)]
mod trusted_refine_tests {
    use super::*;
    #[test]
    fn trusted_refine_opens_without_colmap_and_retains_saved_policy() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let source = tmp.path().join("source.png");
        fs::write(&source, b"fixture")?;
        let root = tmp.path().join("run");
        Run::create(&root, &source, "test", "brush")?;
        let helper = tmp.path().join("unused-media-helper");
        fs::write(&helper, b"unused")?;
        let cfg = Config { ffmpeg:Some(helper.clone()), ffprobe:Some(helper),
            colmap:Some(tmp.path().join("missing-colmap")), cache_dir:Some(tmp.path().join("cache")),
            ..Default::default() };
        assert!(open_for_refine(&root, &cfg).is_err());
        let options = FixedOptions { trust_poses:true, ..Default::default() };
        splat_core::atomic_json(root.join("fixed.json"), &options)?;
        let (run, saved) = open_for_refine(&root, &cfg)?;
        assert!(saved.trust_poses);
        assert_eq!(run.config().backend, "brush");
        assert!(run.state()?.jobs.is_empty());
        Ok(())
    }
}
