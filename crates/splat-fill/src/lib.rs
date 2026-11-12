//! `vauss fill`: the generative fill-in loop for free roaming.
//!
//! A trained splat is sharp along the walked path and falls apart a metre away from it. The loop
//! 1. plans novel cameras on the free floor near the path ([`cams`]),
//! 2. renders them (Rust, GPU),
//! 3. repairs the renders with a fixer model: an external Python step behind a directory protocol
//!    (renders in, repaired images out; `fixer/splat_fixer.py`), default NVIDIA Fixer,
//! 4. colour-fits each repaired view to its render, scores per-pixel confidence (LPIPS between the two:
//!    where the fixer rewrote a well-covered area a lot, trust it less), and
//! 5. trains the views back in at low weight beside the real frames, warm-started from the splat
//!    (`splat __brush train` with a [`splat_brush::PseudoViews`] section).
//!
//! Every stage writes its results into the work dir and is skipped when they are already there, so a
//! run can stop after any stage (`--until`) and continue later.

pub mod cams;

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use glam::{DMat4, DVec3};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_brush::render::{RenderCamera, Renderer};

use cams::{Metric, NovelCam};

/// The fixer runner (`splat_fixer.py fix|confidence`), written into the work dir at run time.
pub const FIXER_PY: &str = include_str!("../fixer/splat_fixer.py");

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Fixer {
    /// NVIDIA Fixer (Cosmos-Predict2 0.6B, one step): Apache-2.0 code, commercially usable weights.
    Fixer,
    /// Difix3D+ with a reference photo (nvidia/difix_ref). NON-COMMERCIAL licence: research use only.
    DifixRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Stage {
    Cams,
    Render,
    Fix,
    Train,
}

/// Generative fill-in for free roaming: render novel views near the walked path, repair them with a fixer model
/// and train them back in at low weight beside the real frames (warm start).
#[derive(Args, Clone, Debug)]
pub struct FillArgs {
    /// A finished run (its result splat, SfM and frozen split) or a COLMAP dataset dir (images/ + sparse/0; then
    /// --ply and --out are required).
    pub source: PathBuf,
    /// Splat to start from (run: the run's result).
    #[arg(long)]
    pub ply: Option<PathBuf>,
    /// Work dir (run: <run>/fill).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Dataset mode: training image names, one per line (default: every registered image not held out).
    #[arg(long)]
    pub train_list: Option<PathBuf>,
    /// Dataset mode: held-out image names, one per line, scored at the end and never trained on.
    #[arg(long)]
    pub heldout_list: Option<PathBuf>,
    /// Stop after this stage (cams, render, fix, train).
    #[arg(long, value_enum, default_value = "train")]
    pub until: Stage,
    /// Redo every stage even when its results exist.
    #[arg(long)]
    pub force: bool,
    /// Train in ready-made views instead of rendering and repairing them (e.g. from a video model): pseudo-view dirs
    /// in the trainer's format (views.json + RGBA PNGs, alpha = confidence; see docs/fill.md), comma-separated to
    /// merge several. Skips cams, render, fix and confidence (no fixer needed).
    #[arg(long, value_delimiter = ',')]
    pub views: Vec<PathBuf>,

    // ---- cameras ----
    /// Keep novel views at most this far (metres, horizontally) from the walked path.
    #[arg(long, default_value_t = 1.0)]
    pub max_dist: f64,
    /// Floor grid spacing, metres.
    #[arg(long, default_value_t = 0.4)]
    pub grid: f64,
    #[arg(long, num_args = 2, value_names = ["X", "Z"], default_values_t = [0.2, 0.2], hide = true)]
    pub offset: Vec<f64>,
    /// Eye heights above the floor, metres.
    #[arg(long, value_delimiter = ',', default_values_t = [1.1, 1.5, 1.8])]
    pub heights: Vec<f64>,
    /// Yaw step, degrees (8 directions at 45).
    #[arg(long, default_value_t = 45.0)]
    pub yaw: f64,
    /// Pitches to draw from, degrees.
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true, default_values_t = [-20.0, -10.0, 0.0, 10.0, 15.0])]
    pub pitches: Vec<f64>,
    /// Vertical field of view of the novel views, degrees.
    #[arg(long, default_value_t = 60.0)]
    pub vfov: f64,
    /// Clearance from furniture and walls for a floor spot, metres.
    #[arg(long, default_value_t = 0.25)]
    pub clearance: f64,
    /// Keep floor spots within this distance (metres) of the walked path's convex hull.
    #[arg(long, default_value_t = 2.0)]
    pub hull_margin: f64,
    #[arg(long, num_args = 2, default_values_t = [0.2, 0.9], hide = true)]
    pub band: Vec<f64>,
    #[arg(long, default_value_t = 10, hide = true)]
    pub occ_min: u32,
    /// Never place a novel camera within 10 cm of one of these views (JSON {"views":[{"c2w": 4x4 rows}]}, e.g. an
    /// evaluation grid).
    #[arg(long)]
    pub exclude: Option<PathBuf>,
    /// Metric frame (JSON with transform_from_source: export-web scene.json or frame.json; default: fitted to the capture
    /// like export-web does).
    #[arg(long)]
    pub frame: Option<PathBuf>,
    /// Use at most this many novel views (a random subset; 0 = all). For smoke tests.
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    /// Novel view width in pixels (16:9).
    #[arg(long, default_value_t = 1024)]
    pub width: u32,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    // ---- fixer ----
    /// Fixer model. `difix-ref` is NON-COMMERCIAL (Difix3D+ weights licence): research use only.
    #[arg(long, value_enum, default_value = "fixer")]
    pub fixer: Fixer,
    /// NVIDIA Fixer checkpoint: `stock` (pretrained) or a fine-tuned .pkl.
    #[arg(long, default_value = "stock")]
    pub fixer_ckpt: String,
    /// Python with the fixer environment (crates/splat-fill/fixer/setup_fixer.sh prints it).
    #[arg(long, default_value = "python3")]
    pub fixer_python: String,
    /// Any other fixer: a program and its arguments, split on whitespace and run directly (no shell), where the tokens
    /// {in}, {out} and {ref} become the dirs of same-named PNGs; it must write one repaired PNG per render into {out}.
    /// Replaces --fixer; confidence still runs through --fixer-python.
    #[arg(long)]
    pub fixer_cmd: Option<String>,
    /// GPUs to repair on, e.g. `0,1,2,3`: one fixer process per GPU (CUDA_VISIBLE_DEVICES), each on its share of the
    /// views. Default: one process on the default GPU.
    #[arg(long, value_delimiter = ',')]
    pub fixer_gpus: Vec<String>,
    /// Confidence = 1 - LPIPS(fixed, render) / tau, clamped to [floor, 1].
    #[arg(long, default_value_t = 0.8)]
    pub lp_tau: f64,
    #[arg(long, default_value_t = 0.3)]
    pub lp_floor: f64,

    // ---- training ----
    /// Warm-start fine-tune steps.
    #[arg(long, default_value_t = 6000)]
    pub steps: u32,
    /// Share of steps that train on a novel view.
    #[arg(long, default_value_t = 0.3)]
    pub pseudo_frac: f32,
    /// Loss weight of a novel view (x 0.5-1 by distance from the path).
    #[arg(long, default_value_t = 0.3)]
    pub pseudo_weight: f32,
    /// Novel views ramp in on a sine over this share of the steps.
    #[arg(long, default_value_t = 0.3)]
    pub ramp: f32,
    /// Position learning-rate factor on novel-view steps (they may shade geometry, not move it much).
    #[arg(long, default_value_t = 0.2)]
    pub pos_lr: f32,
    /// Share of the steps at the end that train on real frames only (0 = novel views to the last step).
    #[arg(long, default_value_t = 0.3)]
    pub real_finish: f32,
}

/// What the loop starts from.
#[derive(Clone, Debug, Serialize)]
pub struct Inputs {
    /// COLMAP dataset dir (images/ + sparse/0).
    pub dataset: PathBuf,
    pub ply: PathBuf,
    pub train_names: Vec<String>,
    pub heldout_names: Vec<String>,
    pub out: PathBuf,
}

fn read_list(path: &Path) -> Result<Vec<String>> {
    Ok(std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
        .lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect())
}

impl Inputs {
    /// Dataset mode: `source` is a COLMAP dir; --ply and --out are required.
    pub fn from_dataset(a: &FillArgs) -> Result<Self> {
        let ply = a.ply.clone().context("a dataset source needs --ply (the splat to start from)")?;
        let out = a.out.clone().context("a dataset source needs --out (the work dir)")?;
        ensure!(a.source.join("sparse/0").is_dir(), "{} is neither a run nor a COLMAP dataset (images/ + sparse/0)", a.source.display());
        let heldout_names = a.heldout_list.as_deref().map(read_list).transpose()?.unwrap_or_default();
        let train_names = match &a.train_list {
            Some(p) => read_list(p)?,
            None => {
                let held: BTreeSet<&String> = heldout_names.iter().collect();
                let scene = splat_brush::render::scene_cameras(&a.source, 64)?;
                scene.into_iter().map(|(n, _)| n).filter(|n| !held.contains(n)).collect()
            }
        };
        Ok(Self { dataset: a.source.clone(), ply, train_names, heldout_names, out })
    }
}

/// A rendered novel view waiting for its repaired image (render.json row).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Rendered {
    name: String,
    c2w: [[f64; 4]; 4],
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: u32,
    height: u32,
    dist_m: f64,
    /// Training photo that sees most of this view's surfaces (reference for reference-based fixers).
    reference: Option<String>,
}

fn write_json(path: &Path, v: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(v)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))
}

/// Splat count from a PLY header (plain or `.ply.zst`).
fn ply_count(path: &Path) -> Result<u32> {
    let mut r = BufReader::new(splat_webexport::ply::open_maybe_zstd(path)?);
    let mut line = String::new();
    loop {
        line.clear();
        ensure!(r.read_line(&mut line)? > 0, "{}: no end_header", path.display());
        if let Some(n) = line.trim().strip_prefix("element vertex ") {
            return Ok(n.trim().parse()?);
        }
        if line.trim() == "end_header" {
            bail!("{}: no vertex element", path.display());
        }
    }
}

/// Stage fingerprints (`stamps.json`): the inputs and settings each stage's outputs were made from. A stage whose
/// fingerprint changed (or `--force`) is redone and its outputs removed first; each fingerprint includes the one
/// before it, so a change flows down the chain. The fingerprint is recorded when a stage starts (`started.<stage>`)
/// as well as when it finishes, so an interrupted stage rerun with the same inputs keeps its finished per-view work.
struct Stamps {
    path: PathBuf,
    map: serde_json::Map<String, Value>,
    force: bool,
}

impl Stamps {
    fn open(out: &Path, force: bool) -> Self {
        let path = out.join("stamps.json");
        let map = read_json::<serde_json::Map<String, Value>>(&path).unwrap_or_default();
        Self { path, map, force }
    }

    /// Whether `stage` is done with exactly `key`. If not, the stage is about to run: its outputs are removed unless
    /// an earlier, interrupted run of it started with the same key (and no `--force`), and the key is recorded as started.
    fn fresh(&mut self, stage: &str, key: &Value, outputs: &[PathBuf]) -> Result<bool> {
        if !self.force && self.map.get(stage) == Some(key) {
            return Ok(true);
        }
        let started = format!("started.{stage}");
        let resume = !self.force && self.map.get(&started) == Some(key);
        self.map.remove(stage);
        self.map.insert(started, key.clone());
        write_json(&self.path, &self.map)?;
        if !resume {
            for p in outputs {
                if p.is_dir() {
                    std::fs::remove_dir_all(p)?;
                } else if p.exists() {
                    std::fs::remove_file(p)?;
                }
            }
        }
        Ok(false)
    }

    fn done(&mut self, stage: &str, key: &Value) -> Result<()> {
        self.map.insert(stage.into(), key.clone());
        write_json(&self.path, &self.map)
    }
}

/// Save via a temporary file and rename, so an interrupted run never leaves a truncated image behind.
fn save_atomic<P: image::Pixel<Subpixel = u8> + image::PixelWithColorType>(img: &image::ImageBuffer<P, Vec<u8>>, path: &Path) -> Result<()> {
    let tmp = path.with_file_name(format!(".{}.tmp.png", path.file_name().and_then(|n| n.to_str()).unwrap_or("img")));
    img.save(&tmp)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Names whose `<dir>/<name>.png` is missing, or does not decode at `size`; the broken ones are deleted.
fn invalid_pngs(dir: &Path, names: &[&str], size: (u32, u32)) -> Vec<String> {
    names.par_iter().filter_map(|n| {
        let p = dir.join(format!("{n}.png"));
        let ok = p.is_file() && image::open(&p).is_ok_and(|img| (img.width(), img.height()) == size);
        if !ok {
            let _ = std::fs::remove_file(&p);
        }
        (!ok).then(|| n.to_string())
    }).collect()
}

/// File identity for fingerprints: path, size and modification time.
fn file_id(p: &Path) -> Value {
    let meta = std::fs::metadata(p).ok();
    json!({"path": p, "len": meta.as_ref().map(|m| m.len()),
        "mtime": meta.and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs())})
}

/// Run the loop. `exe` understands `__brush train` (the running `splat` binary).
pub fn run(a: &FillArgs, inputs: &Inputs, exe: &Path) -> Result<Value> {
    let t0 = std::time::Instant::now();
    let out = &inputs.out;
    std::fs::create_dir_all(out)?;
    ensure!(!inputs.train_names.is_empty(), "no training images");
    let held: BTreeSet<&String> = inputs.heldout_names.iter().collect();
    ensure!(inputs.train_names.iter().all(|n| !held.contains(n)), "a held-out image is also in the training list");
    write_json(&out.join("inputs.json"), inputs)?;
    // Read before any expensive stage: training needs it at the end.
    let n0 = ply_count(&inputs.ply)?;
    ensure!(n0 > 0, "{} has no splats", inputs.ply.display());
    ensure!((0.0..1.0).contains(&a.real_finish), "--real-finish must be in [0, 1)");
    let mut stamps = Stamps::open(out, a.force);
    if !a.views.is_empty() {
        // Ready-made views: no cams, render, fix or confidence; straight to training.
        let (pseudo, n_views, views_key) = views_dirs(&a.views, out)?;
        eprintln!("[fill] {n_views} ready-made views from {}", a.views.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "));
        let mut summary = json!({"inputs": inputs, "views_dir": a.views, "n_views": n_views});
        if a.until != Stage::Train {
            return Ok(summary);
        }
        let t = std::time::Instant::now();
        summary["train"] = train_stage(a, inputs, exe, n0, &pseudo, n_views, json!({"views": views_key}), &mut stamps)?;
        summary["ply"] = json!(out.join("model/splat.ply"));
        summary["timings"] = json!({"train_s": (t.elapsed().as_secs_f64() * 10.0).round() / 10.0});
        summary["wall_s"] = json!(t0.elapsed().as_secs_f64());
        write_json(&out.join("fill.json"), &summary)?;
        return Ok(summary);
    }
    let metric = Metric::from_json(&match &a.frame {
        Some(p) => read_json::<Value>(p)?,
        None => splat_webexport::capture_frame(&inputs.dataset)?,
    })?;
    eprintln!("[fill] metric frame: {:.4} m per SfM unit", metric.s);
    let mut timings = serde_json::Map::new();
    let mut lap = |k: &str, t: &mut std::time::Instant| {
        timings.insert(format!("{k}_s"), json!((t.elapsed().as_secs_f64() * 10.0).round() / 10.0));
        *t = std::time::Instant::now();
    };
    let mut t = std::time::Instant::now();

    // Capture cameras at their training resolution (pinhole approximation).
    let scene = splat_brush::render::scene_cameras(&inputs.dataset, 1 << 16)?;
    let by_name: std::collections::HashMap<&str, &RenderCamera> = scene.iter().map(|(n, c)| (n.as_str(), c)).collect();
    let train: Vec<(&str, &RenderCamera)> = inputs.train_names.iter()
        .map(|n| by_name.get(n.as_str()).map(|c| (n.as_str(), *c)).with_context(|| format!("{n} is not in the dataset")))
        .collect::<Result<_>>()?;
    let path: Vec<DVec3> = inputs.train_names.iter().chain(&inputs.heldout_names)
        .filter_map(|n| by_name.get(n.as_str())).map(|c| c.position.as_dvec3()).collect();

    // One GPU device per process: the renderer serves both camera screening and the novel-view renders.
    let mut renderer: Option<Renderer> = None;
    let cams_path = out.join("cams.json");
    let cams_key = json!({"ply": file_id(&inputs.ply), "dataset": inputs.dataset, "train": inputs.train_names, "heldout": inputs.heldout_names,
        "frame": a.frame.as_deref().map(file_id), "exclude": a.exclude.as_deref().map(file_id), "grid": a.grid, "offset": a.offset,
        "heights": a.heights, "yaw": a.yaw, "pitches": a.pitches, "vfov": a.vfov, "clearance": a.clearance, "hull_margin": a.hull_margin,
        "band": a.band, "occ_min": a.occ_min, "seed": a.seed});
    let cams: Vec<NovelCam> = if stamps.fresh("cams", &cams_key, &[cams_path.clone()])? && cams_path.is_file() {
        read_json(&cams_path)?
    } else {
        let g = splat_webexport::ply::read_ply(&inputs.ply)?;
        let splats: Vec<([f32; 3], f32)> = (0..g.len()).map(|i| (g.pos[i], g.alpha(i))).collect();
        drop(g);
        let r = renderer.insert(Renderer::open(&inputs.ply)?);
        let cams = cams::plan(a, &metric, &path, &splats, r, out)?;
        write_json(&cams_path, &cams)?;
        stamps.done("cams", &cams_key)?;
        cams
    };
    lap("cams", &mut t);
    let mut chosen: Vec<&NovelCam> = cams.iter().filter(|c| c.dist_m <= a.max_dist).collect();
    if a.limit > 0 && chosen.len() > a.limit {
        chosen.shuffle(&mut rand::rngs::StdRng::seed_from_u64(0));
        chosen.truncate(a.limit);
        chosen.sort_by(|x, y| x.name.cmp(&y.name));
    }
    eprintln!("[fill] {} of {} novel views within {} m of the path", chosen.len(), cams.len(), a.max_dist);
    let mut summary = json!({"inputs": inputs, "m_per_unit": metric.s, "n_cams": cams.len(), "n_views": chosen.len(),
        "fixer": a.fixer, "fixer_cmd": a.fixer_cmd});
    if a.until == Stage::Cams {
        return Ok(summary);
    }

    // ---- render ----
    let (w, h) = (a.width, ((a.width as f64 * 9.0 / 16.0 / 8.0).round() * 8.0) as u32);
    let fix = out.join("fix");
    for d in ["in", "aux", "ref", "out", "fit", "conf"] {
        std::fs::create_dir_all(fix.join(d))?;
    }
    let render_path = fix.join("render.json");
    let wants_ref = a.fixer_cmd.is_some() || a.fixer == Fixer::DifixRef;
    let render_key = json!({"cams": cams_key, "max_dist": a.max_dist, "limit": a.limit, "width": a.width, "refs": wants_ref});
    let render_outputs = ["in", "aux", "ref"].map(|d| fix.join(d)).into_iter().chain([render_path.clone()]).collect::<Vec<_>>();
    let rendered: Vec<Rendered> = if stamps.fresh("render", &render_key, &render_outputs)? && render_path.is_file() {
        read_json(&render_path)?
    } else {
        for d in ["in", "aux", "ref"] {
            std::fs::create_dir_all(fix.join(d))?;
        }
        let r = match renderer.as_mut() {
            Some(r) => r,
            None => renderer.insert(Renderer::open(&inputs.ply)?),
        };
        let mut rows = Vec::with_capacity(chosen.len());
        for (i, c) in chosen.iter().enumerate() {
            let c2w = cams::from_rows(&c.c2w);
            let cam = cams::render_camera(&c2w, c.vfov_deg, w, h);
            let f = r.render(&cam, true)?;
            save_atomic(&f.rgb, &fix.join("in").join(format!("{}.png", c.name)))?;
            let alpha: Vec<u8> = f.alpha.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
            save_atomic(&image::GrayImage::from_raw(w, h, alpha).context("alpha")?, &fix.join("aux").join(format!("{}_a.png", c.name)))?;
            // Pixels nearer than 0.3 m (grazing furniture) are never trusted.
            let near: Vec<u8> = f.alpha.iter().zip(&f.depth).map(|(al, z)| if *al < 0.5 || *z as f64 * metric.s > 0.3 { 255 } else { 0 }).collect();
            save_atomic(&image::GrayImage::from_raw(w, h, near).context("near")?, &fix.join("aux").join(format!("{}_n.png", c.name)))?;
            let reference = wants_ref.then(|| best_reference(&f, &cam, &c2w, &train, metric.s)).flatten();
            if let Some(name) = &reference {
                let photo = image::open(inputs.dataset.join("images").join(name))?.to_rgb8();
                save_atomic(&image::imageops::resize(&photo, w, h, image::imageops::FilterType::Lanczos3), &fix.join("ref").join(format!("{}.png", c.name)))?;
            }
            rows.push(Rendered { name: c.name.clone(), c2w: c.c2w, fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy, width: w, height: h,
                dist_m: c.dist_m, reference });
            if i % 200 == 0 {
                eprintln!("[fill] render {}/{}", i + 1, chosen.len());
            }
        }
        write_json(&render_path, &rows)?;
        stamps.done("render", &render_key)?;
        rows
    };
    drop(renderer);
    lap("render", &mut t);
    if a.until == Stage::Render {
        return Ok(summary);
    }

    // ---- fix (external) + colour fit + confidence ----
    let runner = out.join("splat_fixer.py");
    std::fs::write(&runner, FIXER_PY)?;
    let names: Vec<&str> = rendered.iter().map(|r| r.name.as_str()).collect();
    let fix_key = json!({"render": render_key, "fixer": a.fixer, "ckpt": a.fixer_ckpt, "ckpt_file": Path::new(&a.fixer_ckpt).is_file().then(|| file_id(Path::new(&a.fixer_ckpt))),
        "cmd": a.fixer_cmd});
    if !stamps.fresh("fix", &fix_key, &[fix.join("out"), out.join("fixer.json")])? {
        std::fs::create_dir_all(fix.join("out"))?;
        if a.fixer == Fixer::DifixRef && a.fixer_cmd.is_none() {
            eprintln!("[fill] WARNING: difix-ref weights are licensed for NON-COMMERCIAL use only");
        }
        // What repairs the views (model, checkpoint hash, weights revision, licence): written once, before the workers.
        if a.fixer_cmd.is_none() {
            let model = if a.fixer == Fixer::Fixer { "fixer" } else { "difix_ref" };
            let status = Command::new(&a.fixer_python).arg(&runner).args(["provenance", "--model", model, "--ckpt", &a.fixer_ckpt, "--out"])
                .arg(out.join("fixer.json")).status().context("recording the fixer's provenance")?;
            if !status.success() {
                eprintln!("[fill] WARNING: fixer provenance not recorded ({status})");
            }
        }
        // A fixer that dies part-way leaves its finished views; broken or missing ones are redone (two tries).
        for attempt in 0..2 {
            let todo = invalid_pngs(&fix.join("out"), &names, (w, h));
            if todo.is_empty() {
                break;
            }
            eprintln!("[fill] fixer: {} of {} views to repair (try {})", todo.len(), names.len(), attempt + 1);
            let status = match &a.fixer_cmd {
                Some(cmd) => {
                    let (inp, outp, refp) = (fix.join("in"), fix.join("out"), fix.join("ref"));
                    let argv: Vec<std::ffi::OsString> = cmd.split_whitespace().map(|t| match t {
                        "{in}" => inp.clone().into_os_string(),
                        "{out}" => outp.clone().into_os_string(),
                        "{ref}" => refp.clone().into_os_string(),
                        t => t.into(),
                    }).collect();
                    let (prog, args) = argv.split_first().context("--fixer-cmd is empty")?;
                    Command::new(prog).args(args).status()
                }
                None => {
                    let model = match a.fixer {
                        Fixer::Fixer => "fixer",
                        Fixer::DifixRef => "difix_ref",
                    };
                    on_gpus(&a.fixer_gpus, |c| {
                        c.arg(&runner).args(fix_args(model, &a.fixer_ckpt, &fix, wants_ref));
                    }, &a.fixer_python)
                }
            }.context("running the fixer")?;
            ensure!(status.success(), "the fixer failed ({status})");
        }
        let left = invalid_pngs(&fix.join("out"), &names, (w, h)).len();
        ensure!(left == 0, "the fixer left {left} renders unrepaired (or unreadable) in {}", fix.join("out").display());
        stamps.done("fix", &fix_key)?;
    }
    lap("fixer", &mut t);
    let conf_key = json!({"fix": fix_key, "lp_tau": a.lp_tau, "lp_floor": a.lp_floor});
    let pseudo = out.join("pseudo");
    if !stamps.fresh("confidence", &conf_key, &[fix.join("fit"), fix.join("conf"), pseudo.clone()])? {
        for d in [fix.join("fit"), fix.join("conf"), pseudo.clone()] {
            std::fs::create_dir_all(d)?;
        }
        // Fixers drift colour: a 3x3 + bias fit of each repaired view onto its render, on well-covered pixels.
        rendered.par_iter().try_for_each(|r| -> Result<()> {
            let render = image::open(fix.join("in").join(format!("{}.png", r.name)))?.to_rgb8();
            let mut fixed = image::open(fix.join("out").join(format!("{}.png", r.name)))?.to_rgb8();
            if fixed.dimensions() != render.dimensions() {
                fixed = image::imageops::resize(&fixed, render.width(), render.height(), image::imageops::FilterType::Lanczos3);
            }
            let alpha = image::open(fix.join("aux").join(format!("{}_a.png", r.name)))?.to_luma8();
            colour_fit(&mut fixed, &render, &alpha);
            save_atomic(&fixed, &fix.join("fit").join(format!("{}.png", r.name)))
        })?;
        for _ in 0..2 {
            if invalid_pngs(&fix.join("conf"), &names, (w, h)).is_empty() {
                break;
            }
            let status = on_gpus(&a.fixer_gpus, |c| {
                c.arg(&runner).arg("confidence").arg("--a").arg(fix.join("fit")).arg("--b").arg(fix.join("in"))
                    .arg("--out").arg(fix.join("conf")).args(["--lp-tau", &a.lp_tau.to_string(), "--lp-floor", &a.lp_floor.to_string()]);
            }, &a.fixer_python).context("running the confidence step")?;
            ensure!(status.success(), "the confidence step failed ({status})");
        }
        ensure!(invalid_pngs(&fix.join("conf"), &names, (w, h)).is_empty(), "the confidence step left views unscored in {}", fix.join("conf").display());
    // Pseudo-view dir: RGBA (alpha = trust), views.json for the trainer.
    let views: Vec<splat_brush::PseudoView> = rendered.par_iter().map(|r| -> Result<splat_brush::PseudoView> {
        let fitted = image::open(fix.join("fit").join(format!("{}.png", r.name)))?.to_rgb8();
        let alpha = image::open(fix.join("aux").join(format!("{}_a.png", r.name)))?.to_luma8();
        let near = image::open(fix.join("aux").join(format!("{}_n.png", r.name)))?.to_luma8();
        let conf = image::open(fix.join("conf").join(format!("{}.png", r.name)))?.to_luma8();
        let (w, h) = fitted.dimensions();
        // Uncovered pixels (the fixer filled a hole) keep full trust; a hole has nothing to disagree with.
        let mask: Vec<f32> = (0..(w * h) as usize).map(|i| {
            let c = if alpha.as_raw()[i] < 128 { 1.0 } else { conf.as_raw()[i] as f32 / 255.0 };
            c * near.as_raw()[i] as f32 / 255.0
        }).collect();
        let mask = erode(&mask, w as usize, h as usize, 2);
        let mut rgba = image::RgbaImage::new(w, h);
        for (i, (p, q)) in rgba.pixels_mut().zip(fitted.pixels()).enumerate() {
            *p = image::Rgba([q[0], q[1], q[2], (mask[i] * 255.0).round() as u8]);
        }
        let file = format!("{}.png", r.name);
        save_atomic(&rgba, &pseudo.join(&file))?;
        // Per-view weight: full within 1 m of the path, fading to half at 3 m.
        let weight = if r.dist_m <= 1.0 { 1.0 } else { (1.0 - 0.25 * (r.dist_m - 1.0)).max(0.5) } as f32;
        Ok(splat_brush::PseudoView { image: file, c2w: cams::rows_f32(&cams::from_rows(&r.c2w)), fx: r.fx, fy: r.fy, cx: r.cx, cy: r.cy,
            width: r.width, height: r.height, weight, conf_mean: mask.iter().sum::<f32>() / mask.len() as f32 })
    }).collect::<Result<_>>()?;
    write_json(&pseudo.join("views.json"), &json!({"views": views}))?;
    stamps.done("confidence", &conf_key)?;
    }
    let views: Vec<splat_brush::PseudoView> = read_json::<Value>(&pseudo.join("views.json"))
        .and_then(|v| Ok(serde_json::from_value(v["views"].clone())?))?;
    let conf_mean = views.iter().map(|v| v.conf_mean as f64).sum::<f64>() / views.len().max(1) as f64;
    summary["fixer_provenance"] = read_json::<Value>(&out.join("fixer.json")).unwrap_or(Value::Null);
    summary["mean_confidence"] = json!(conf_mean);
    lap("confidence", &mut t);
    if a.until == Stage::Fix {
        summary["timings"] = Value::Object(timings);
        return Ok(summary);
    }

    // ---- train ----
    let trained = train_stage(a, inputs, exe, n0, &pseudo, views.len(), json!({"confidence": conf_key}), &mut stamps)?;
    lap("train", &mut t);
    summary["ply"] = json!(out.join("model/splat.ply"));
    summary["train"] = trained;
    summary["timings"] = Value::Object(timings);
    summary["wall_s"] = json!(t0.elapsed().as_secs_f64());
    write_json(&out.join("fill.json"), &summary)?;
    Ok(summary)
}

/// The warm-start fine-tune on the real frames plus the pseudo views in `pseudo` (`<out>/model`); `upstream` is the
/// fingerprint of what made the views. Returns the trainer's summary.
#[allow(clippy::too_many_arguments)]
fn train_stage(a: &FillArgs, inputs: &Inputs, exe: &Path, n0: u32, pseudo: &Path, n_views: usize, upstream: Value, stamps: &mut Stamps)
    -> Result<Value> {
    let model = inputs.out.join("model");
    let max_resolution = inputs.train_names.iter().chain(&inputs.heldout_names)
        .map(|n| image::image_dimensions(inputs.dataset.join("images").join(n)).map(|(w, h)| w.max(h)))
        .try_fold(0, |m, d| d.map(|d| m.max(d)))?;
    // Real-only finish: no novel views in the last `real_finish` of the steps.
    let stop = (a.real_finish > 0.0).then(|| 1.0 - a.real_finish);
    let job: splat_brush::TrainJob = serde_json::from_value(json!({
        "dataset": inputs.dataset, "out": model, "train_names": inputs.train_names, "heldout_names": inputs.heldout_names,
        "total_steps": a.steps, "start_step": 0, "init_ply": inputs.ply, "warm_start": true,
        // MCMC relocation keeps running; no growth past the starting count.
        "max_splats": n0, "cap_max_explicit": true, "max_resolution": max_resolution,
        "eval_every": a.steps, "checkpoint_every": a.steps, "sh_degree": 3, "seed": a.seed, "gpu_lock": null,
        // MCMC-style refine (relocation, no growth) every 100 steps, as in the reference loop.
        "refine_every": 100,
        "config_overrides": fill_recipe(),
        "pseudo": {"dir": pseudo, "frac": a.pseudo_frac, "weight": a.pseudo_weight, "ramp": a.ramp, "pos_lr": a.pos_lr, "stop": stop},
    }))?;
    let mut train_key = upstream;
    train_key["job"] = serde_json::to_value(&job)?;
    let job_path = model.join("fill_train_job.json");
    if !stamps.fresh("train", &train_key, &[model.clone()])? || !model.join("train_summary.json").is_file() {
        std::fs::create_dir_all(&model)?;
        write_json(&job_path, &job)?;
        let log = std::fs::File::create(model.join("train.log"))?;
        eprintln!("[fill] training: {} real + {} novel views, {} steps from {} splats (log {})", inputs.train_names.len(), n_views,
            a.steps, n0, model.join("train.log").display());
        let status = Command::new(exe).args(["__brush", "train"]).arg(&job_path).stdout(log.try_clone()?).stderr(log).status()
            .context("starting the train worker")?;
        ensure!(status.success(), "training failed ({status}); see {}", model.join("train.log").display());
        stamps.done("train", &train_key)?;
    }
    read_json(&model.join("train_summary.json"))
}

/// The views of a ready-made pseudo-view dir (`views.json` + RGBA PNGs), checked: at least one, every image there,
/// decodable with alpha at its listed size.
fn read_views(dir: &Path) -> Result<Vec<splat_brush::PseudoView>> {
    #[derive(Deserialize)]
    struct ViewsFile {
        views: Vec<splat_brush::PseudoView>,
    }
    let views = read_json::<ViewsFile>(&dir.join("views.json"))?.views;
    ensure!(!views.is_empty(), "{}: no views", dir.join("views.json").display());
    views.par_iter().try_for_each(|v| -> Result<()> {
        use image::ImageDecoder;
        let p = dir.join(&v.image);
        let dec = image::ImageReader::open(&p).with_context(|| format!("{}: missing view image", p.display()))?
            .with_guessed_format()?.into_decoder().with_context(|| format!("decoding {}", p.display()))?;
        ensure!(dec.dimensions() == (v.width, v.height), "{}: {:?}, views.json says {}x{}", p.display(), dec.dimensions(), v.width, v.height);
        ensure!(dec.color_type().has_alpha(), "{}: no alpha channel (alpha = per-pixel confidence)", p.display());
        Ok(())
    })?;
    Ok(views)
}

/// `--views`: checks each dir and returns the pseudo dir to train in (the dir itself, or for several a merged
/// `<out>/views` of links to them with image paths prefixed `<i>/`), the view count and a fingerprint (views.json
/// and image sizes and times).
fn views_dirs(dirs: &[PathBuf], out: &Path) -> Result<(PathBuf, usize, Value)> {
    use sha2::{Digest, Sha256};
    let mut all = vec![];
    let mut key = vec![];
    for (i, d) in dirs.iter().enumerate() {
        let d = d.canonicalize().with_context(|| format!("--views {}", d.display()))?;
        let views = read_views(&d)?;
        let mut h = Sha256::new();
        h.update(std::fs::read(d.join("views.json"))?);
        for v in &views {
            h.update(file_id(&d.join(&v.image)).to_string());
        }
        key.push(json!({"dir": d, "n": views.len(), "sha256": format!("{:x}", h.finalize())}));
        all.push((i, d, views));
    }
    let n: usize = all.iter().map(|(_, _, v)| v.len()).sum();
    if let [(_, d, _)] = &all[..] {
        return Ok((d.clone(), n, Value::Array(key)));
    }
    let merged = out.join("views");
    if merged.exists() {
        std::fs::remove_dir_all(&merged)?;
    }
    std::fs::create_dir_all(&merged)?;
    let mut views = Vec::with_capacity(n);
    for (i, d, vs) in all {
        link_dir(&d, &merged.join(i.to_string()))?;
        views.extend(vs.into_iter().map(|mut v| {
            v.image = format!("{i}/{}", v.image);
            v
        }));
    }
    write_json(&merged.join("views.json"), &json!({"views": views}))?;
    Ok((merged, n, Value::Array(key)))
}

#[cfg(unix)]
fn link_dir(src: &Path, dst: &Path) -> Result<()> {
    Ok(std::os::unix::fs::symlink(src, dst)?)
}

/// No symlinks without privileges on Windows: copy the dir instead.
#[cfg(not(unix))]
fn link_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        if e.file_type()?.is_dir() {
            link_dir(&e.path(), &dst.join(e.file_name()))?;
        } else {
            std::fs::copy(e.path(), dst.join(e.file_name()))?;
        }
    }
    Ok(())
}

/// Arguments of one fixer worker (`splat_fixer.py fix ...`); the same for every GPU shard, which adds only `--shard`.
fn fix_args(model: &str, ckpt: &str, fix: &Path, wants_ref: bool) -> Vec<std::ffi::OsString> {
    let mut v: Vec<std::ffi::OsString> = ["fix", "--model", model, "--ckpt", ckpt, "--in"].map(Into::into).into();
    v.extend([fix.join("in").into(), "--out".into(), fix.join("out").into()]);
    if wants_ref {
        v.extend(["--ref".into(), fix.join("ref").into()]);
    }
    v
}

/// Run `python <args>` once per GPU in `gpus` (each with CUDA_VISIBLE_DEVICES and `--shard i/N`), or once when empty;
/// fails if any process fails.
fn on_gpus(gpus: &[String], args: impl Fn(&mut Command), python: &str) -> std::io::Result<std::process::ExitStatus> {
    if gpus.len() <= 1 {
        let mut c = Command::new(python);
        args(&mut c);
        if let Some(g) = gpus.first() {
            c.env("CUDA_VISIBLE_DEVICES", g);
        }
        return c.status();
    }
    let children = gpus.iter().enumerate().map(|(i, g)| {
        let mut c = Command::new(python);
        args(&mut c);
        c.args(["--shard", &format!("{i}/{}", gpus.len())]).env("CUDA_VISIBLE_DEVICES", g);
        c.spawn()
    }).collect::<std::io::Result<Vec<_>>>()?;
    let mut worst = None;
    for mut child in children {
        let status = child.wait()?;
        if !status.success() || worst.is_none() {
            worst = Some(status);
        }
    }
    Ok(worst.expect("at least two GPUs"))
}

/// The warm-start fine-tune of the reference loop (research fill.py, gsplat), as trainer overrides. Learning rates:
/// means 0.1x the usual schedule (1.6e-5 -> 1.6e-7 of the scene extent), scales 2.5e-3, opacity 0.025, rotation 1e-3,
/// SH DC 2.5e-3 with higher bands 1/20 of it; opacity regulariser 0.01; no growth; no transient masking, opacity decay or
/// quantisation tail (the output is a PLY, not SPZ-bound). Differences that remain: the native regulariser runs for the
/// first third of the steps only, there is no scale regulariser, and refinement stops at 95% of the steps (85%).
pub fn fill_recipe() -> Value {
    json!({"lr-mean": 1.6e-5, "lr-mean-end": 1.6e-7, "lr-scale": 2.5e-3, "lr-opac": 0.025, "lr-rotation": 1e-3, "lr-coeffs-dc": 2.5e-3,
        "lr-coeffs-sh-scale": 20.0, "opacity-reg": 0.01, "opac-decay": 0.0, "growth-stop-iter": 0, "robust-loss": false, "qat-tail": false})
}

/// The training photo that sees most of the view's surface points (subsampled every 16 px), preferring a similar
/// look direction and a nearby camera.
fn best_reference(f: &splat_brush::render::Frame, cam: &RenderCamera, c2w: &DMat4, train: &[(&str, &RenderCamera)], s: f64) -> Option<String> {
    let mut pts = vec![];
    for y in (4..f.height).step_by(16) {
        for x in (4..f.width).step_by(16) {
            let i = (y * f.width + x) as usize;
            let d = f.depth[i] as f64;
            if f.alpha[i] > 0.5 && d > 0.0 {
                let pc = DVec3::new((x as f64 - cam.cx as f64) / cam.fx as f64 * d, (y as f64 - cam.cy as f64) / cam.fy as f64 * d, d);
                pts.push(c2w.transform_point3(pc));
            }
        }
    }
    if pts.is_empty() {
        return None;
    }
    let fwd = c2w.z_axis.truncate();
    let pos = c2w.w_axis.truncate();
    train.iter().map(|(name, tc)| {
        let r = tc.rotation.as_dquat().inverse();
        let p = tc.position.as_dvec3();
        let seen = pts.iter().filter(|x| {
            let q = r * (**x - p);
            let (u, v) = (tc.fx as f64 * q.x / q.z + tc.cx as f64, tc.fy as f64 * q.y / q.z + tc.cy as f64);
            q.z > 0.05 && u > 0.0 && u < tc.width as f64 && v > 0.0 && v < tc.height as f64
        }).count() as f64 / pts.len() as f64;
        let dir = (tc.rotation.as_dquat() * DVec3::Z).dot(fwd);
        (seen + 0.15 * dir - 0.05 * p.distance(pos) * s, name)
    }).max_by(|a, b| a.0.total_cmp(&b.0)).map(|(_, n)| n.to_string())
}

/// Least-squares 3x3 + bias colour map of `fixed` onto `render` over pixels with alpha > 0.8 (needs > 1000 of them).
fn colour_fit(fixed: &mut image::RgbImage, render: &image::RgbImage, alpha: &image::GrayImage) {
    let mut ata = [[0f64; 4]; 4];
    let mut atb = [[0f64; 3]; 4];
    let mut n = 0usize;
    for ((p, q), al) in fixed.pixels().zip(render.pixels()).zip(alpha.pixels()) {
        if al[0] <= 204 {
            continue;
        }
        let x = [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0, 1.0];
        for i in 0..4 {
            for j in 0..4 {
                ata[i][j] += x[i] * x[j];
            }
            for c in 0..3 {
                atb[i][c] += x[i] * q[c] as f64 / 255.0;
            }
        }
        n += 1;
    }
    if n <= 1000 {
        return;
    }
    // A tiny ridge keeps rank-deficient fits (grey or single-colour views) solvable, like lstsq's minimum-norm answer.
    for (i, row) in ata.iter_mut().enumerate() {
        row[i] += 1e-6 * n as f64;
    }
    let Some(m) = solve4(ata, atb) else { return };
    for p in fixed.pixels_mut() {
        let x = [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0, 1.0];
        for c in 0..3 {
            let v: f64 = (0..4).map(|i| x[i] * m[i][c]).sum();
            p[c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

/// Solve the 4x4 normal equations (Gauss-Jordan with partial pivoting) for three right-hand sides.
fn solve4(mut a: [[f64; 4]; 4], mut b: [[f64; 3]; 4]) -> Option<[[f64; 3]; 4]> {
    for col in 0..4 {
        let piv = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in 0..4 {
            if r != col {
                let f = a[r][col] / a[col][col];
                for k in 0..4 {
                    a[r][k] -= f * a[col][k];
                }
                for k in 0..3 {
                    b[r][k] -= f * b[col][k];
                }
            }
        }
    }
    for r in 0..4 {
        for k in 0..3 {
            b[r][k] /= a[r][r];
        }
    }
    Some(b)
}

/// Min filter over a (2r+1)^2 window clipped to the image (erodes the trusted area).
fn erode(m: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; m.len()];
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
            tmp[y * w + x] = m[y * w + a..=y * w + b].iter().copied().fold(f32::MAX, f32::min);
        }
    }
    let mut out = vec![0f32; m.len()];
    for y in 0..h {
        let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            out[y * w + x] = (a..=b).map(|yy| tmp[yy * w + x]).fold(f32::MAX, f32::min);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_fit_recovers_an_affine_map() {
        let (w, h) = (64, 32);
        let render = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 3) as u8, (y * 7) as u8, ((x + y) * 2) as u8]));
        // fixed = 0.8 * render + 20 (an affine drift the fit must undo)
        let mut fixed = image::RgbImage::from_fn(w, h, |x, y| {
            let p = render.get_pixel(x, y);
            image::Rgb([0, 1, 2].map(|c| (p[c] as f32 * 0.8 + 20.0).round() as u8))
        });
        let alpha = image::GrayImage::from_pixel(w, h, image::Luma([255]));
        colour_fit(&mut fixed, &render, &alpha);
        let err: f64 = fixed.pixels().zip(render.pixels()).map(|(a, b)| (0..3).map(|c| (a[c] as f64 - b[c] as f64).abs()).sum::<f64>()).sum::<f64>()
            / (w * h * 3) as f64;
        assert!(err < 1.0, "mean abs error {err}");
    }

    #[test]
    fn colour_fit_handles_grey_views() {
        let (w, h) = (64, 32);
        let render = image::RgbImage::from_fn(w, h, |x, _| image::Rgb([(x * 3) as u8; 3]));
        let mut fixed = image::RgbImage::from_fn(w, h, |x, _| image::Rgb([(x * 3 + 10) as u8; 3]));
        let alpha = image::GrayImage::from_pixel(w, h, image::Luma([255]));
        colour_fit(&mut fixed, &render, &alpha);
        let err: f64 = fixed.pixels().zip(render.pixels()).map(|(a, b)| (a[0] as f64 - b[0] as f64).abs()).sum::<f64>() / (w * h) as f64;
        assert!(err < 1.0, "grey view not fitted: mean abs error {err}");
    }

    #[test]
    fn ply_count_reads_compressed_headers() {
        let dir = std::env::temp_dir().join(format!("splat-fill-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = b"ply\nformat binary_little_endian 1.0\nelement vertex 7\nproperty float x\nend_header\n";
        std::fs::write(dir.join("a.ply"), header).unwrap();
        std::fs::write(dir.join("a.ply.zst"), zstd::encode_all(&header[..], 3).unwrap()).unwrap();
        assert_eq!(ply_count(&dir.join("a.ply")).unwrap(), 7);
        assert_eq!(ply_count(&dir.join("a.ply.zst")).unwrap(), 7);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_fixer_worker_writes_shared_provenance() {
        // Shared provenance is written once before the workers (they used to race on one temp file).
        let args = fix_args("fixer", "stock", Path::new("/w/fix"), true);
        assert!(!args.iter().any(|a| a == "--provenance"), "{args:?}");
        assert!(args.iter().any(|a| a == "--ref"));
        assert!(super::FIXER_PY.contains("def cmd_provenance"));
    }

    #[test]
    fn restart_with_same_inputs_keeps_finished_views() {
        let dir = std::env::temp_dir().join(format!("splat-fill-stamps-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("out")).unwrap();
        let key = json!({"k": 1});
        let outputs = [dir.join("out")];
        // First run starts the stage, finishes one view, then dies before `done`.
        assert!(!Stamps::open(&dir, false).fresh("fix", &key, &outputs).unwrap());
        std::fs::create_dir_all(dir.join("out")).unwrap();
        std::fs::write(dir.join("out/n00000.png"), b"done").unwrap();
        // Same inputs: the finished view survives the restart.
        assert!(!Stamps::open(&dir, false).fresh("fix", &key, &outputs).unwrap());
        assert!(dir.join("out/n00000.png").exists());
        // Changed inputs (or --force): it does not.
        assert!(!Stamps::open(&dir, false).fresh("fix", &json!({"k": 2}), &outputs).unwrap());
        assert!(!dir.join("out/n00000.png").exists());
        std::fs::create_dir_all(dir.join("out")).unwrap();
        std::fs::write(dir.join("out/n00000.png"), b"done").unwrap();
        assert!(!Stamps::open(&dir, true).fresh("fix", &json!({"k": 2}), &outputs).unwrap());
        assert!(!dir.join("out/n00000.png").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn views_dirs_validate_and_merge() {
        let dir = std::env::temp_dir().join(format!("splat-fill-views-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let view = |img: &str, w: u32| json!({"image": img, "c2w": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "fx": 8.0, "fy": 8.0, "cx": 8.0, "cy": 4.0, "width": w, "height": 8, "weight": 1.0, "conf_mean": 1.0});
        for d in ["a", "b"] {
            std::fs::create_dir_all(dir.join(d)).unwrap();
            image::RgbaImage::new(16, 8).save(dir.join(d).join("v.png")).unwrap();
            write_json(&dir.join(d).join("views.json"), &json!({"views": [view("v.png", 16)]})).unwrap();
        }
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let (p, n, _) = views_dirs(&[dir.join("a")], &out).unwrap();
        assert_eq!((p, n), (dir.join("a").canonicalize().unwrap(), 1));
        // Several dirs: one merged views.json whose images resolve through <out>/views/<i>/.
        let (p, n, _) = views_dirs(&[dir.join("a"), dir.join("b")], &out).unwrap();
        assert_eq!(n, 2);
        let merged = read_views(&p).unwrap();
        assert_eq!(merged[1].image, "1/v.png");
        // Wrong size, missing image, no alpha, no views: all refused.
        write_json(&dir.join("b/views.json"), &json!({"views": [view("v.png", 32)]})).unwrap();
        assert!(read_views(&dir.join("b")).is_err());
        write_json(&dir.join("b/views.json"), &json!({"views": [view("w.png", 16)]})).unwrap();
        assert!(read_views(&dir.join("b")).is_err());
        image::RgbImage::new(16, 8).save(dir.join("b/w.png")).unwrap();
        assert!(read_views(&dir.join("b")).is_err());
        write_json(&dir.join("b/views.json"), &json!({"views": []})).unwrap();
        assert!(read_views(&dir.join("b")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pseudo_stop_round_trips() {
        let p: splat_brush::PseudoViews = serde_json::from_value(json!({"dir": "/p", "frac": 0.3, "weight": 0.3, "ramp": 0.3, "pos_lr": 0.2})).unwrap();
        assert_eq!(p.stop, None);
        let p: splat_brush::PseudoViews = serde_json::from_value(json!({"dir": "/p", "frac": 0.3, "weight": 0.3, "ramp": 0.3, "pos_lr": 0.2, "stop": 0.7})).unwrap();
        assert_eq!(p.stop, Some(0.7));
        assert!(serde_json::from_value::<splat_brush::PseudoViews>(json!({"dir": "/p", "frac": 0.3, "weight": 0.3, "ramp": 0.3, "pos_lr": 0.2, "x": 1})).is_err());
    }

    #[test]
    fn erode_takes_the_window_minimum() {
        let mut m = vec![1f32; 25];
        m[12] = 0.0;
        let e = erode(&m, 5, 5, 1);
        assert_eq!(e.iter().filter(|v| **v == 0.0).count(), 9);
    }
}
