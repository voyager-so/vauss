//! `vauss effect …`: programmable splat effects (effect graph v1, *.effect.json) on a scene package:
//! validate, bake one time to a PLY, preview contact sheets with telemetry, render videos, export the
//! effect into a web package for the viewers, bench the evaluator.
//!
//! Every command prints one JSON object on stdout ({"ok":true,…} or {"ok":false,"error":{code,message}}).

mod cloud;
mod raster;
mod scene;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::{Subcommand, ValueEnum};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use splat_effects as fx;
use splat_webexport::ply::Gaussians;

use raster::{Pass, Raster};
use scene::{Pick, Scene};

#[derive(Subcommand)]
pub enum Cmd {
    /// Check an effect graph (and, with --package, that it resolves on that scene).
    Validate {
        graph: PathBuf,
        #[arg(long)]
        package: Option<PathBuf>,
    },
    /// Bake the effect at one time into a splat file (.ply, or .spz).
    Apply {
        scene: PathBuf,
        graph: PathBuf,
        /// Seconds into the effect.
        #[arg(long)]
        t: f32,
        #[arg(short = 'o', long = "out")]
        out: PathBuf,
        #[command(flatten)]
        common: Common,
    },
    /// Contact sheet: times × cameras × passes, plus telemetry JSON.
    Preview {
        scene: PathBuf,
        graph: PathBuf,
        /// A dir (→ contact.png + telemetry.json) or a .png path (→ <stem>.telemetry.json beside it).
        #[arg(short = 'o', long = "out")]
        out: PathBuf,
        /// Fractions of the duration (0,0.5,1), percentages (0,25,50%) or seconds (1.5s).
        #[arg(long, default_value = "0,25,50,75,100%")]
        times: String,
        /// N, auto[:N] (package start camera + spread capture cameras) or orbit:N.
        #[arg(long, default_value = "auto")]
        cams: String,
        #[arg(long, default_value = "rgb,mask,displacement")]
        passes: String,
        #[command(flatten)]
        size: Size,
        #[command(flatten)]
        common: Common,
    },
    /// Video over the effect's timeline (PNG frames on disk, then ffmpeg).
    Render {
        scene: PathBuf,
        graph: PathBuf,
        #[arg(short = 'o', long = "out")]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "fixed")]
        camera: CameraMode,
        /// Degrees the orbit camera turns over the effect.
        #[arg(long, default_value_t = 60.0)]
        orbit_deg: f32,
        #[arg(long)]
        fps: Option<f32>,
        #[command(flatten)]
        size: Size,
        /// Leave the PNG frames in <out stem>.frames/.
        #[arg(long)]
        keep_frames: bool,
        /// Write the PNG frames to <out stem>.frames/ and skip ffmpeg (the caller encodes).
        #[arg(long = "frames-only", alias = "no-encode")]
        no_encode: bool,
        #[command(flatten)]
        common: Common,
    },
    /// Add the effect to a web scene package: effects/<id>.effect.json (+ .mask.u8) and scene.json `effects`.
    Export {
        package: PathBuf,
        graph: PathBuf,
        /// Play on load in the viewer.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
        autoplay: bool,
        #[command(flatten)]
        common: Common,
    },
    /// Time the evaluator (and one render) on a scene.
    Bench {
        scene: PathBuf,
        graph: PathBuf,
        #[arg(long, default_value_t = 5)]
        reps: usize,
        /// Also write the result JSON here.
        #[arg(short = 'o', long = "out")]
        out: Option<PathBuf>,
        #[command(flatten)]
        size: Size,
        #[command(flatten)]
        common: Common,
    },
}

#[derive(clap::Args, Clone, Default)]
pub struct Common {
    /// Package of a Voyager scene root to use (default: active_splat; render: render_splat).
    #[arg(long)]
    splat: Option<String>,
    /// Render on a fal GPU instead of the local CPU.
    #[arg(long)]
    cloud: bool,
    /// Covariance update: exact (eigen refactor) or fast (polar).
    #[arg(long, default_value = "exact")]
    cov: String,
}

#[derive(clap::Args, Clone)]
pub struct Size {
    /// WxH, e.g. 1920x1080.
    #[arg(long)]
    size: Option<String>,
    #[arg(long)]
    width: Option<u32>,
    #[arg(long)]
    height: Option<u32>,
}

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Fixed,
    Orbit,
}

impl Size {
    fn resolve(&self, default_w: u32) -> Result<(u32, u32)> {
        let (mut w, mut h) = (None, None);
        if let Some(s) = &self.size {
            let (a, b) = s.split_once(['x', 'X']).with_context(|| format!("--size {s}: expected WxH"))?;
            (w, h) = (Some(a.trim().parse()?), Some(b.trim().parse()?));
        }
        let w = self.width.or(w).unwrap_or(default_w);
        let h = self.height.or(h).unwrap_or((w as f32 * 9.0 / 16.0).round() as u32);
        ensure!((16..=8192).contains(&w) && (16..=8192).contains(&h), "image size {w}x{h} out of range");
        Ok((w & !1, h & !1))
    }
    fn args(&self) -> Vec<String> {
        let mut a = vec![];
        if let Some(s) = &self.size {
            a.extend(["--size".into(), s.clone()]);
        }
        if let Some(w) = self.width {
            a.extend(["--width".into(), w.to_string()]);
        }
        if let Some(h) = self.height {
            a.extend(["--height".into(), h.to_string()]);
        }
        a
    }
}

/// Error code + message for the JSON envelope.
#[derive(Debug)]
struct Coded(&'static str, String);
impl std::fmt::Display for Coded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.1)
    }
}
impl std::error::Error for Coded {}

pub use cloud::child as remote_child;

pub fn run(cmd: Cmd, compact: bool) -> Result<i32> {
    let result = dispatch(cmd);
    let (code, value) = match result {
        Ok(mut v) => {
            v.as_object_mut().map(|o| o.insert("ok".into(), json!(true)));
            (0, v)
        }
        Err(e) => {
            let code = e.downcast_ref::<Coded>().map(|c| c.0)
                .or_else(|| e.downcast_ref::<GraphFailure>().map(|_| "invalid_graph")).unwrap_or("failed");
            let mut err = json!({"code": code, "message": format!("{e:#}")});
            if let Some(GraphFailure(issues)) = e.downcast_ref::<GraphFailure>() {
                err["issues"] = json!(issues);
            }
            (1, json!({"ok": false, "error": err}))
        }
    };
    // One line always: Voyager and the demo lane read the last stdout line.
    let _ = compact;
    println!("{}", serde_json::to_string(&value)?);
    Ok(code)
}

/// An invalid graph, with every issue found.
#[derive(Debug)]
struct GraphFailure(Vec<Value>);
impl std::fmt::Display for GraphFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msgs: Vec<String> = self.0.iter().map(|i| format!("{}: {}", i["path"].as_str().unwrap_or(""), i["message"].as_str().unwrap_or(""))).collect();
        write!(f, "invalid effect graph: {}", msgs.join("; "))
    }
}
impl std::error::Error for GraphFailure {}

fn load_graph(path: &Path) -> Result<(fx::Graph, Value)> {
    let text = std::fs::read_to_string(path).map_err(|e| Coded("graph_unreadable", format!("{}: {e}", path.display())))?;
    let raw: Value = serde_json::from_str(&text).map_err(|e| Coded("graph_not_json", format!("{}: {e}", path.display())))?;
    match fx::Graph::from_json(&text) {
        Ok(g) => Ok((g, raw)),
        Err(e) => Err(GraphFailure(vec![json!({"path": e.path, "message": e.msg})]).into()),
    }
}

fn dispatch(cmd: Cmd) -> Result<Value> {
    match cmd {
        Cmd::Validate { graph, package } => validate(&graph, package.as_deref()),
        Cmd::Apply { scene, graph, t, out, common } => apply(&scene, &graph, t, &out, &common),
        Cmd::Preview { scene, graph, out, times, cams, passes, size, common } => {
            if common.cloud {
                return preview_cloud(&scene, &graph, &out, &times, &cams, &passes, &size, &common);
            }
            preview(&scene, &graph, &out, &times, &cams, &passes, &size, &common)
        }
        Cmd::Render { scene, graph, out, camera, orbit_deg, fps, size, keep_frames, no_encode, common } => {
            let o = RenderOpts { camera, orbit_deg, fps, size, keep_frames, frames_only: no_encode };
            if common.cloud { render_cloud(&scene, &graph, &out, &o, &common) } else { render(&scene, &graph, &out, &o, &common) }
        }
        Cmd::Export { package, graph, autoplay, common } => export(&package, &graph, autoplay, &common),
        Cmd::Bench { scene, graph, reps, out, size, common } => {
            if common.cloud { bench_cloud(&scene, &graph, reps, out.as_deref(), &size, &common) } else { bench(&scene, &graph, reps, out.as_deref(), &size, &common) }
        }
    }
}

fn pick<'a>(common: &'a Common, default: Pick<'a>) -> Pick<'a> {
    common.splat.as_deref().map(Pick::Id).unwrap_or(default)
}

fn open_scene(path: &Path, p: Pick) -> Result<Scene> {
    scene::open(path, p).map_err(|e| Coded("scene_unreadable", format!("{e:#}")).into())
}

fn eval_opts(scene: &Scene, graph: &fx::Graph, common: &Common) -> Result<fx::EvalOpts> {
    let mut opts = fx::EvalOpts::default();
    opts.scene_radius = Some(scene.radius);
    opts.up = scene.up.to_array();
    if let Some(mpu) = scene.meta["meters_per_unit"].as_f64().filter(|m| *m > 0.0) {
        opts.units_per_meter = (1.0 / mpu) as f32;
    }
    opts.floor = scene.package.as_deref().and_then(package_floor);
    opts.cov = match common.cov.as_str() {
        "exact" | "" => fx::CovMode::Exact,
        "fast" => fx::CovMode::Fast,
        o => bail!("--cov {o}: exact or fast"),
    };
    for m in graph.object_masks() {
        let w = resolve_object(scene, &m.selection, m.object_ids.as_deref()).and_then(|mut w| {
            for u in &m.union {
                let wu = resolve_object(scene, u, None)?;
                w.iter_mut().zip(wu).for_each(|(a, b)| *a = a.max(b));
            }
            if let Some(r) = m.dilate.filter(|r| *r > 0.0) {
                w = dilate_weights(&scene.g.pos, &w, r);
            }
            Ok(w)
        });
        match w {
            Ok(w) => {
                opts.objects.insert(m.key(), w);
            }
            // An exported graph's box stands in where the selection itself is not here.
            Err(_) if m.bbox.is_some() => {}
            Err(e) => return Err(e),
        }
    }
    Ok(opts)
}

/// Grow a selection in 3D: every splat within `r` of a selected splat (weight >= 0.5) gets 1.
/// A voxel hash at cell `r` keeps it O(N · 27).
fn dilate_weights(pos: &[[f32; 3]], w: &[f32], r: f32) -> Vec<f32> {
    use std::collections::HashMap;
    let cell = |p: &[f32; 3]| [(p[0] / r).floor() as i32, (p[1] / r).floor() as i32, (p[2] / r).floor() as i32];
    let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
    for (i, p) in pos.iter().enumerate().filter(|(i, _)| w[*i] >= 0.5) {
        grid.entry(cell(p)).or_default().push(i as u32);
    }
    let r2 = r * r;
    pos.iter()
        .enumerate()
        .map(|(i, p)| {
            if w[i] >= 0.5 {
                return w[i];
            }
            let c = cell(p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(v) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            if v.iter().any(|&j| {
                                let q = pos[j as usize];
                                (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2) <= r2
                            }) {
                                return 1.0;
                            }
                        }
                    }
                }
            }
            w[i]
        })
        .collect()
}

/// The package's floor plane from `collision/floor.json` (`vauss export-collision`):
/// `collision_floor.normal` and its `y` (top of the floor slab) as `normal·p = y·|normal|`.
fn package_floor(pkg: &Path) -> Option<fx::graph::Floor> {
    let v: Value = serde_json::from_slice(&std::fs::read(pkg.join("collision").join("floor.json")).ok()?).ok()?;
    let f = &v["collision_floor"];
    let n: Vec<f32> = f["normal"].as_array()?.iter().filter_map(|x| x.as_f64().map(|x| x as f32)).collect();
    let n = glam::Vec3::new(*n.first()?, *n.get(1)?, *n.get(2)?).try_normalize()?;
    let y = f["y"].as_f64()? as f32;
    // floor.json's y is the plane height along +Y; for a tilted normal the plane passes through (0, y, 0).
    Some(fx::graph::Floor { normal: n.to_array(), offset: n.y * y })
}

/// Per-splat weights of an object mask, in package order. `selection` names a .mask.u8 in the
/// package's effects/ (one byte per splat, package order, as `export` writes), or a selection
/// artifact dir (selected_ids.npy / object_ids.npy, splat-core edit.rs). Selections index the
/// candidate's splat.ply; a package exported from it (scene.json `source.ply`) gets them mapped
/// through `transform_from_source` onto its own splats.
fn resolve_object(scene: &Scene, selection: &str, ids: Option<&[u32]>) -> Result<Vec<f32>> {
    let n = scene.g.len();
    let mut cands: Vec<PathBuf> = vec![PathBuf::from(selection)];
    if let Some(p) = &scene.package {
        cands.push(p.join(selection));
        cands.push(p.join("effects").join(selection));
        cands.push(p.join("effects").join(format!("{selection}.mask.u8")));
    }
    for c in cands {
        if c.is_file() && c.extension().is_some_and(|e| e == "u8") {
            let b = std::fs::read(&c)?;
            ensure!(b.len() == n, "{}: {} bytes for {} splats", c.display(), b.len(), n);
            return Ok(b.iter().map(|v| *v as f32 / 255.0).collect());
        }
        if c.is_dir() && c.join("selected_ids.npy").is_file() {
            let sel = splat_core::edit::read_npy_u32(&c.join("selected_ids.npy"))?;
            let obj = c.join("object_ids.npy");
            let obj = if obj.is_file() { splat_core::edit::read_npy_u32(&obj)? } else { vec![0; sel.len()] };
            let chosen: Vec<u32> = sel.iter().zip(&obj).filter(|(_, o)| ids.is_none_or(|ids| ids.contains(o))).map(|(i, _)| *i).collect();
            return match source_frame(scene, &c) {
                Some((src, m)) => map_to_package(scene, &src, &m, &chosen),
                None => {
                    let mut w = vec![0.0; n];
                    chosen.iter().filter(|i| (**i as usize) < n).for_each(|i| w[*i as usize] = 1.0);
                    Ok(w)
                }
            };
        }
    }
    Err(Coded("selection_not_found", format!("object mask selection {selection:?}: no <name>.mask.u8 or selection dir (selected_ids.npy) found")).into())
}

/// The candidate splat a package was exported from and the source→package matrix (column-major).
/// When scene.json's `source.ply` is gone or a stub of a file kept on fal, the same candidate in the
/// run holding the selection (`<run>/artifacts/<candidate>/splat.ply`, e.g. after `vauss pull`).
fn source_frame(scene: &Scene, selection: &Path) -> Option<(PathBuf, glam::DMat4)> {
    let real = |p: &Path| p.is_file() && std::fs::metadata(p).is_ok_and(|m| m.len() > 4096);
    let src = PathBuf::from(scene.meta["source"]["ply"].as_str()?);
    let src = if real(&src) {
        src
    } else {
        let rel: PathBuf = src.components().skip_while(|c| c.as_os_str() != "artifacts").collect();
        let run = selection.canonicalize().ok()?.ancestors().find(|a| a.join("artifacts").is_dir() && a.join("run.json").is_file())?.to_path_buf();
        Some(run.join(rel)).filter(|p| real(p))?
    };
    let m: Vec<f64> = scene.meta["transform_from_source"].as_array()?.iter().filter_map(Value::as_f64).collect();
    (m.len() == 16).then(|| (src, glam::DMat4::from_cols_slice(&m)))
}

/// Package splats matching the chosen candidate splats: same position after the export transform,
/// on the SPZ position grid (12 fractional bits), allowing one grid step of rounding.
fn map_to_package(scene: &Scene, src: &Path, m: &glam::DMat4, chosen: &[u32]) -> Result<Vec<f32>> {
    const Q: f64 = 4096.0;
    let key = |p: glam::DVec3| ((p.x * Q).round() as i64, (p.y * Q).round() as i64, (p.z * Q).round() as i64);
    let source = splat_webexport::ply::read_ply(src)?;
    let mut grid: std::collections::HashMap<(i64, i64, i64), u32> = std::collections::HashMap::with_capacity(scene.g.len());
    for (i, p) in scene.g.pos.iter().enumerate() {
        grid.entry(key(glam::DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64))).or_insert(i as u32);
    }
    let mut w = vec![0.0f32; scene.g.len()];
    let mut hit = 0usize;
    for &i in chosen {
        let Some(p) = source.pos.get(i as usize) else { continue };
        let (x, y, z) = key(m.transform_point3(glam::DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)));
        let found = (-1..=1).flat_map(|a| (-1..=1).flat_map(move |b| (-1..=1).map(move |c| (x + a, y + b, z + c)))).find_map(|k| grid.get(&k));
        if let Some(&j) = found {
            w[j as usize] = 1.0;
            hit += 1;
        }
    }
    eprintln!("[effect] selection: {hit} of {} candidate splats found in the package", chosen.len());
    ensure!(chosen.is_empty() || hit * 2 >= chosen.len(),
        "selection matched only {hit} of {} splats in this package (made on another candidate than {}?)", chosen.len(), src.display());
    Ok(w)
}

fn eval_err(e: fx::EvalError) -> anyhow::Error {
    Coded("selection_unresolved", e.to_string()).into()
}

fn validate(path: &Path, package: Option<&Path>) -> Result<Value> {
    let (graph, _) = load_graph(path)?;
    let mut out = json!({
        "effect_id": graph.effect_id, "duration_s": graph.duration_s, "fps": graph.fps,
        "nodes": graph.nodes.iter().map(|n| json!({"id": n.id, "type": n.op.type_name()})).collect::<Vec<_>>(),
    });
    if let Some(p) = package {
        let scene = open_scene(p, Pick::Active)?;
        let opts = eval_opts(&scene, &graph, &Common::default())?;
        let t0 = Instant::now();
        let mid = fx::evaluate(&scene.g, &graph, graph.duration_s * 0.5, &opts).map_err(eval_err)?;
        out["package"] = json!({
            "splats": scene.g.len(), "selected": mid.selected, "invalid_at_mid": mid.invalid,
            "eval_ms_mid": t0.elapsed().as_secs_f64() * 1e3, "bounds": scene.meta["bounds"], "up": scene.up.to_array(),
        });
    }
    Ok(out)
}

fn apply(scene_p: &Path, graph_p: &Path, t: f32, out: &Path, common: &Common) -> Result<Value> {
    let scene = open_scene(scene_p, pick(common, Pick::Active))?;
    let (graph, _) = load_graph(graph_p)?;
    let opts = eval_opts(&scene, &graph, common)?;
    let t0 = Instant::now();
    let frame = fx::evaluate(&scene.g, &graph, t, &opts).map_err(eval_err)?;
    let transform_ms = t0.elapsed().as_secs_f64() * 1e3;
    let g = frame.to_splats(&scene.g);
    if let Some(p) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p)?;
    }
    match out.extension().and_then(|e| e.to_str()) {
        Some("spz") => {
            splat_webexport::spz::write_spz(&g, out)?;
        }
        _ => splat_webexport::ply::write_ply(&g, out)?,
    }
    Ok(json!({"out": abs(out), "t": t, "splats": g.len(), "invalid": frame.invalid, "transform_ms": transform_ms, "frame_hash": frame_hash(&g)}))
}

/// sha256 (first 16 hex) of the evaluated splat state: identical for identical (rest, graph, t).
fn frame_hash(g: &Gaussians) -> String {
    let mut h = Sha256::new();
    for v in [&g.pos[..], &g.scale[..], &g.dc[..]] {
        v.iter().for_each(|x| x.iter().for_each(|f| h.update(f.to_le_bytes())));
    }
    g.rot.iter().for_each(|x| x.iter().for_each(|f| h.update(f.to_le_bytes())));
    g.opacity.iter().for_each(|f| h.update(f.to_le_bytes()));
    format!("{:x}", h.finalize())[..16].to_string()
}

fn abs(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

fn image_hash(img: &image::RgbImage) -> String {
    format!("{:x}", Sha256::digest(img.as_raw()))[..16].to_string()
}

/// Times in seconds from "0,25,50,75,100%" (percent), "0,0.5,1" (fractions) or "1.5s,3s" (seconds).
fn parse_times(s: &str, duration: f32) -> Result<Vec<f32>> {
    let pct = s.contains('%');
    s.split(',').filter(|t| !t.trim().is_empty()).map(|t| {
        let t = t.trim();
        let v = if let Some(x) = t.strip_suffix('s') {
            x.parse::<f32>()?
        } else if let Some(x) = t.strip_suffix('%') {
            x.parse::<f32>()? / 100.0 * duration
        } else if pct {
            t.parse::<f32>()? / 100.0 * duration
        } else {
            t.parse::<f32>()? * duration
        };
        ensure!(v.is_finite(), "--times {s}: bad value {t}");
        Ok(v.clamp(0.0, duration))
    }).collect::<Result<Vec<_>>>().and_then(|v| if v.is_empty() { bail!("--times is empty") } else { Ok(v) })
}

fn sheet_paths(out: &Path) -> (PathBuf, PathBuf) {
    if out.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")) {
        let stem = out.file_stem().unwrap().to_string_lossy();
        (out.to_path_buf(), out.with_file_name(format!("{stem}.telemetry.json")))
    } else {
        (out.join("contact.png"), out.join("telemetry.json"))
    }
}

#[allow(clippy::too_many_arguments)]
fn preview(scene_p: &Path, graph_p: &Path, out: &Path, times: &str, cams: &str, passes: &str, size: &Size, common: &Common) -> Result<Value> {
    let adapter = raster::pin_adapter();
    let scene = open_scene(scene_p, pick(common, Pick::Active))?;
    let (graph, _) = load_graph(graph_p)?;
    let opts = eval_opts(&scene, &graph, common)?;
    let times = parse_times(times, graph.duration_s)?;
    let cams = scene.pick_cams(cams)?;
    let passes: Vec<Pass> = passes.split(',').filter(|p| !p.trim().is_empty()).map(Pass::parse).collect::<Result<_>>()?;
    let (w, h) = size.resolve(480)?;
    let n = scene.g.len();
    let prep = fx::Prepared::new(&scene.g, &graph, &opts).map_err(eval_err)?;

    // Evaluate every time first: the displacement pass shares one colour scale over the sheet.
    let mut frames = Vec::new();
    for &t in &times {
        let t0 = Instant::now();
        let f = prep.frame(t);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let disp = fx::displacement(&scene.g, &f);
        frames.push((t, f, disp, ms));
    }
    let disp_max = {
        let mut all: Vec<f32> = frames.iter().flat_map(|f| f.2.iter().step_by((n / 50000).max(1)).copied()).filter(|d| *d > 0.0).collect();
        all.sort_by(f32::total_cmp);
        all.get(all.len().saturating_sub(1).min(all.len() * 99 / 100)).copied().unwrap_or(1.0)
    };
    // The mask pass shows where each node acts (its mask, curves held at 1), drawn on the moved splats.
    let selection = {
        let mut on = graph.clone();
        on.nodes.iter_mut().for_each(|n| n.window = None);
        fx::Prepared::new(&scene.g, &on, &opts).map_err(eval_err)?.frame(0.0).weight
    };
    let mut raster = Raster::new();
    let rows = passes.len() * cams.len();
    let mut tiles: Vec<Vec<image::RgbImage>> = vec![Vec::new(); rows];
    let mut rows_meta = Vec::new();
    for pass in &passes {
        for ci in 0..cams.len() {
            rows_meta.push(format!("cam{ci} {}", pass.name()));
        }
    }
    let mut frame_meta = Vec::new();
    for (t, f, disp, transform_ms) in &frames {
        let g = f.to_splats(&scene.g);
        let hash = frame_hash(&g);
        let mut render_ms = 0.0;
        let mut tile_meta = Vec::new();
        for (pi, pass) in passes.iter().enumerate() {
            let colors = raster::pass_colors(*pass, &selection, disp, disp_max);
            let raw = raster::raw(&g, colors.as_deref());
            let mut first = Some(raw);
            for (ci, cam) in cams.iter().enumerate() {
                let r0 = Instant::now();
                let img = match first.take() {
                    Some(raw) => raster.render(raw, cam, scene.up, w, h)?,
                    None => raster.render_again(cam, scene.up, w, h)?,
                };
                render_ms += r0.elapsed().as_secs_f64() * 1e3;
                tile_meta.push(json!({"cam": ci, "pass": pass.name(), "image_hash": image_hash(&img)}));
                tiles[pi * cams.len() + ci].push(img);
            }
        }
        let active = f.weight.iter().filter(|w| **w > 0.0).count();
        // Splats no node touched must be their rest state bit for bit (the scene stays crisp).
        let rest_changed = (0..n)
            .filter(|&i| f.weight[i] == 0.0)
            .filter(|&i| f.pos[i] != scene.g.pos[i] || f.rot[i] != scene.g.rot[i] || f.scale[i] != scene.g.scale[i] || f.opacity[i] != scene.g.opacity[i] || f.dc[i] != scene.g.dc[i])
            .count();
        let dmax = disp.iter().copied().fold(0.0f32, f32::max);
        frame_meta.push(json!({
            "t": t, "u": t / graph.duration_s.max(1e-9), "total": n, "selected": f.selected, "active": active, "untouched_changed": rest_changed, "invalid": f.invalid,
            "max_displacement": dmax, "transform_ms": transform_ms, "render_ms": render_ms, "frame_hash": hash, "tiles": tile_meta,
        }));
    }
    let col_labels: Vec<String> = times.iter().map(|t| format!("t={t:.2}s {:.0}%", t / graph.duration_s.max(1e-9) * 100.0)).collect();
    let sheet = raster::sheet(&tiles, &col_labels, &rows_meta);
    let (sheet_p, tele_p) = sheet_paths(out);
    if let Some(p) = sheet_p.parent() {
        std::fs::create_dir_all(p)?;
    }
    sheet.save(&sheet_p)?;
    let telemetry = json!({
        "effect_id": graph.effect_id, "scene": scene_p, "package": scene.package, "adapter": adapter,
        "total": n, "selected": frames.iter().map(|f| f.1.selected).max(), "invalid_max": frame_meta.iter().filter_map(|f| f["invalid"].as_u64()).max(),
        "transform_ms_total": frames.iter().map(|f| f.3).sum::<f64>(),
        "render_ms_total": frame_meta.iter().filter_map(|f| f["render_ms"].as_f64()).sum::<f64>(),
        "size": [w, h], "cams": cams, "passes": passes.iter().map(|p| p.name()).collect::<Vec<_>>(),
        "displacement_scale": disp_max, "sh_degree_rendered": raster::render_sh_degree(&scene.g), "cov": common.cov, "sh_view_rotation": fx::SH_VIEW_ROTATION,
        "layout": {"columns": col_labels, "rows": rows_meta}, "frames": frame_meta, "contact_sheet": sheet_p,
    });
    std::fs::write(&tele_p, serde_json::to_vec_pretty(&telemetry)?)?;
    Ok(json!({"contact_sheet": abs(&sheet_p), "telemetry": abs(&tele_p), "summary": summary(&telemetry), "outputs": [abs(&sheet_p), abs(&tele_p)]}))
}

/// The telemetry without per-tile detail, for the stdout envelope.
fn summary(t: &Value) -> Value {
    let mut s = t.clone();
    if let Some(frames) = s["frames"].as_array_mut() {
        for f in frames {
            f.as_object_mut().map(|o| o.remove("tiles"));
        }
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn preview_cloud(scene_p: &Path, graph_p: &Path, out: &Path, times: &str, cams: &str, passes: &str, size: &Size, common: &Common) -> Result<Value> {
    let pkg = scene::resolve(scene_p, pick(common, Pick::Active))?;
    let job = cloud::job(&sheet_paths(out).0)?;
    let s = job.add_scene(&pkg)?;
    let g = job.add_file(graph_p, "graph.effect.json")?;
    let mut args: Vec<String> = vec!["--json".into(), "effect".into(), "preview".into(), s.to_string_lossy().into(), g.to_string_lossy().into(),
        "-o".into(), job.outputs.join("preview").to_string_lossy().into(), "--times".into(), times.into(), "--cams".into(), cams.into(),
        "--passes".into(), passes.into(), "--cov".into(), common.cov.clone()];
    args.extend(size.args());
    let cost = job.run("effect_preview", args)?;
    let (sheet_p, tele_p) = sheet_paths(out);
    if let Some(p) = sheet_p.parent() {
        std::fs::create_dir_all(p)?;
    }
    job.take("preview/contact.png", &sheet_p)?;
    job.take("preview/telemetry.json", &tele_p)?;
    job.cleanup();
    let mut tele: Value = serde_json::from_slice(&std::fs::read(&tele_p)?)?;
    tele["contact_sheet"] = json!(sheet_p);
    tele["scene"] = json!(scene_p);
    tele["package"] = json!(pkg);
    tele["cloud"] = cost.clone();
    std::fs::write(&tele_p, serde_json::to_vec_pretty(&tele)?)?;
    Ok(json!({"contact_sheet": abs(&sheet_p), "telemetry": abs(&tele_p), "summary": summary(&tele), "cloud": cost, "outputs": [abs(&sheet_p), abs(&tele_p)]}))
}

struct RenderOpts {
    camera: CameraMode,
    orbit_deg: f32,
    fps: Option<f32>,
    size: Size,
    keep_frames: bool,
    frames_only: bool,
}

fn frames_dir(out: &Path) -> PathBuf {
    let stem = out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "render".into());
    out.with_file_name(format!("{stem}.frames"))
}

fn render(scene_p: &Path, graph_p: &Path, out: &Path, o: &RenderOpts, common: &Common) -> Result<Value> {
    let adapter = raster::pin_adapter();
    let scene = open_scene(scene_p, pick(common, Pick::Render))?;
    let (graph, _) = load_graph(graph_p)?;
    let opts = eval_opts(&scene, &graph, common)?;
    let (w, h) = o.size.resolve(1920)?;
    let fps = o.fps.unwrap_or(graph.fps).clamp(1.0, 120.0);
    let count = (graph.duration_s * fps).round().max(1.0) as usize + usize::from(!graph.looping);
    let dir = frames_dir(out);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir_all(&dir)?;
    let mut raster = Raster::new();
    let (mut transform_ms, mut render_ms, mut invalid_max) = (0.0, 0.0, 0usize);
    let mut hashes = Vec::new();
    let prep = fx::Prepared::new(&scene.g, &graph, &opts).map_err(eval_err)?;
    // On a GPU (fal) the WGSL evaluator moves the splats in place; the CPU path evaluates on the host.
    let gpu = if raster::gpu_allowed() { Some(raster.gpu_effect(&scene.g, &graph, &opts).map_err(|e| anyhow::anyhow!("{e}"))?) } else { None };
    let t_all = Instant::now();
    for k in 0..count {
        if let Some(gfx) = &gpu {
            let t = k as f32 / fps;
            let cam = match o.camera {
                CameraMode::Fixed => scene.cams[0],
                CameraMode::Orbit => scene.orbit(o.orbit_deg * k as f32 / (count.max(2) - 1) as f32),
            };
            let t0 = Instant::now();
            let (invalid, _, f) = raster.gpu_frame(gfx, t);
            transform_ms += t0.elapsed().as_secs_f64() * 1e3;
            invalid_max = invalid_max.max(invalid);
            if k == 0 || k + 1 == count {
                hashes.push(frame_hash(&fx::gpu::download(&f).to_splats(&scene.g)));
            }
            drop(f);
            let r0 = Instant::now();
            let img = raster.render_again(&cam, scene.up, w, h)?;
            render_ms += r0.elapsed().as_secs_f64() * 1e3;
            img.save(dir.join(format!("{k:05}.png")))?;
            if k % 30 == 0 {
                eprintln!("[effect] frame {}/{count} t={t:.2}s ({:.1}s elapsed, gpu eval)", k + 1, t_all.elapsed().as_secs_f64());
            }
            continue;
        }
        let t = k as f32 / fps;
        let cam = match o.camera {
            CameraMode::Fixed => scene.cams[0],
            CameraMode::Orbit => scene.orbit(o.orbit_deg * k as f32 / (count.max(2) - 1) as f32),
        };
        let t0 = Instant::now();
        let f = prep.frame(t);
        let g = f.to_splats(&scene.g);
        transform_ms += t0.elapsed().as_secs_f64() * 1e3;
        invalid_max = invalid_max.max(f.invalid);
        hashes.push(frame_hash(&g));
        drop(f);
        let r0 = Instant::now();
        let img = raster.render(raster::raw(&g, None), &cam, scene.up, w, h)?;
        render_ms += r0.elapsed().as_secs_f64() * 1e3;
        img.save(dir.join(format!("{k:05}.png")))?;
        if k % 10 == 0 {
            eprintln!("[effect] frame {}/{count} t={t:.2}s ({:.1}s elapsed)", k + 1, t_all.elapsed().as_secs_f64());
        }
    }
    let telemetry = json!({
        "effect_id": graph.effect_id, "scene": scene_p, "package": scene.package, "adapter": adapter, "total": scene.g.len(),
        "sh_degree_rendered": raster::render_sh_degree(&scene.g),
        "frames": count, "fps": fps, "size": [w, h], "camera": if o.camera == CameraMode::Fixed { "fixed" } else { "orbit" },
        "invalid_max": invalid_max, "transform_ms_total": transform_ms, "render_ms_total": render_ms,
        "evaluator": if gpu.is_some() { "gpu-wgsl" } else { "cpu" },
        "first_last_identical": hashes.first() == hashes.last(), "frame_hashes": hashes,
    });
    // Beside the frames dir, never in it: callers delete the PNGs and then the (empty) dir.
    let tele_p = out.with_extension("telemetry.json");
    std::fs::write(&tele_p, serde_json::to_vec_pretty(&telemetry)?)?;
    if o.frames_only {
        return Ok(json!({"mp4": null, "frames_dir": abs(&dir), "fps": fps, "frames": count, "frame_pattern": "%05d.png",
            "telemetry": abs(&tele_p), "summary": telemetry}));
    }
    finish_video(out, &dir, fps, telemetry, o.keep_frames)
}

/// Encode `dir`/%05d.png to `out` (H.264 mp4, else whatever ffmpeg infers from the extension),
/// reading frames from disk, under a memory cap where systemd-run exists.
fn finish_video(out: &Path, dir: &Path, fps: f32, mut telemetry: Value, keep: bool) -> Result<Value> {
    let t0 = Instant::now();
    let ff = std::env::var("SPLAT_FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let capped = Command::new("systemd-run").arg("--version").output().is_ok_and(|o| o.status.success());
    let mut cmd = if capped {
        let mut c = Command::new("systemd-run");
        c.args(["--user", "--scope", "--quiet", "-p", "MemoryHigh=6G", "-p", "MemoryMax=8G", &ff]);
        c
    } else {
        Command::new(&ff)
    };
    cmd.args(["-y", "-loglevel", "error", "-framerate", &format!("{fps}"), "-i"]).arg(dir.join("%05d.png"));
    if out.extension().is_some_and(|e| e == "mp4" || e == "mov") {
        // libx264 where this ffmpeg has it; LGPL builds (Voyager's) fall back to openh264, then MPEG-4.
        let encoders = Command::new(&ff).args(["-hide_banner", "-encoders"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        let has = |e: &str| encoders.lines().any(|l| l.split_whitespace().nth(1) == Some(e));
        if has("libx264") {
            cmd.args(["-c:v", "libx264", "-preset", "medium", "-crf", "16"]);
        } else if has("libopenh264") {
            cmd.args(["-c:v", "libopenh264", "-b:v", "16M"]);
        } else {
            cmd.args(["-c:v", "mpeg4", "-q:v", "2"]);
        }
        cmd.args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"]);
    }
    let status = cmd.arg(out).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).status().context("running ffmpeg")?;
    ensure!(status.success(), "ffmpeg failed encoding {}", out.display());
    telemetry["encode_ms"] = json!(t0.elapsed().as_secs_f64() * 1e3);
    let tele_p = out.with_extension("telemetry.json");
    std::fs::write(&tele_p, serde_json::to_vec_pretty(&telemetry)?)?;
    if !keep {
        std::fs::remove_dir_all(dir).ok();
    }
    let mut s = telemetry.clone();
    s.as_object_mut().map(|o| o.remove("frame_hashes"));
    let video = if out.extension().is_some_and(|e| e == "mp4") { "mp4" } else { "video" };
    Ok(json!({video: abs(out), "telemetry": abs(&tele_p), "summary": s, "frames_dir": keep.then(|| abs(dir)), "outputs": [abs(out), abs(&tele_p)]}))
}

fn render_cloud(scene_p: &Path, graph_p: &Path, out: &Path, o: &RenderOpts, common: &Common) -> Result<Value> {
    let pkg = scene::resolve(scene_p, pick(common, Pick::Render))?;
    let job = cloud::job(out)?;
    let s = job.add_scene(&pkg)?;
    let g = job.add_file(graph_p, "graph.effect.json")?;
    let remote_out = job.outputs.join("render.mp4");
    let mut args: Vec<String> = vec!["--json".into(), "effect".into(), "render".into(), s.to_string_lossy().into(), g.to_string_lossy().into(),
        "-o".into(), remote_out.to_string_lossy().into(), "--frames-only".into(), "--camera".into(),
        if o.camera == CameraMode::Fixed { "fixed".into() } else { "orbit".into() }, "--orbit-deg".into(), o.orbit_deg.to_string(),
        "--cov".into(), common.cov.clone()];
    if let Some(f) = o.fps {
        args.extend(["--fps".into(), f.to_string()]);
    }
    let mut size = o.size.clone();
    if size.size.is_none() && size.width.is_none() {
        size.size = Some("1920x1080".into());
    }
    args.extend(size.args());
    let cost = job.run("effect_render", args)?;
    if let Some(p) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p)?;
    }
    let dir = frames_dir(out);
    job.take("render.frames", &dir)?;
    let tele_p = out.with_extension("telemetry.json");
    job.take("render.telemetry.json", &tele_p)?;
    job.cleanup();
    let mut telemetry: Value = serde_json::from_slice(&std::fs::read(&tele_p)?)?;
    telemetry["scene"] = json!(scene_p);
    telemetry["package"] = json!(pkg);
    telemetry["cloud"] = cost;
    let fps = telemetry["fps"].as_f64().unwrap_or(30.0) as f32;
    if o.frames_only {
        std::fs::write(&tele_p, serde_json::to_vec_pretty(&telemetry)?)?;
        return Ok(json!({"mp4": null, "frames_dir": abs(&dir), "fps": fps, "frames": telemetry["frames"], "frame_pattern": "%05d.png",
            "telemetry": abs(&tele_p), "summary": telemetry}));
    }
    finish_video(out, &dir, fps, telemetry, o.keep_frames)
}

fn export(package: &Path, graph_p: &Path, autoplay: bool, common: &Common) -> Result<Value> {
    let (graph, _) = load_graph(graph_p)?;
    let targets: Vec<PathBuf> = match &common.splat {
        Some(id) => vec![scene::resolve(package, Pick::Id(id))?],
        None if scene::is_root(package) => scene::root_packages(package)?.into_iter().map(|(_, p)| p).collect(),
        None => vec![package.to_path_buf()],
    };
    let mut out = Vec::new();
    for pkg in targets {
        out.push(export_one(&pkg, &graph, autoplay, common).with_context(|| format!("exporting into {}", pkg.display()))?);
    }
    Ok(json!({"effect_id": graph.effect_id, "packages": out}))
}

/// effects/<id>.effect.json (+ <id>[.k].mask.u8 per object mask) and the scene.json `effects` entry.
fn export_one(pkg: &Path, graph: &fx::Graph, autoplay: bool, common: &Common) -> Result<Value> {
    let scene = open_scene(pkg, Pick::Active)?;
    ensure!(scene.package.is_some(), "{} is not a web scene package", pkg.display());
    let opts = eval_opts(&scene, graph, common)?;
    let id = &graph.effect_id;
    ensure!(!id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || "_-.".contains(c)), "effect_id {id:?}: use letters, digits, _ - .");
    let dir = pkg.join("effects");
    std::fs::create_dir_all(&dir)?;
    // Floor, gravity, aniso_max and per-node boxes resolved, so viewers never scan splats.
    let mut graph = fx::eval::resolve(&scene.g, graph, &opts)?;
    // Which splats the effect can ever touch (weight > 0 at some t, sampled densely over the
    // timeline). Viewers evaluate only these per frame; the rest is static.
    let active = active_mask(&scene.g, &graph, &opts)?;
    let active_rel = format!("effects/{id}.active.u8");
    write_atomic(&pkg.join(&active_rel), &active)?;
    let active_count = active.iter().filter(|a| **a != 0).count();
    let mut masks = Vec::new();
    let mut k = 0usize;
    for node in &mut graph.nodes {
        resolve_masks(&mut node.mask, &mut |o: &mut fx::ObjectMask| -> Result<()> {
            let Some(w) = opts.objects.get(&o.key()) else { return Ok(()) }; // keeps its own box
            let name = if k == 0 { format!("{id}.mask.u8") } else { format!("{id}.{k}.mask.u8") };
            k += 1;
            let bytes: Vec<u8> = w.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
            write_atomic(&dir.join(&name), &bytes)?;
            let bbox = weighted_box(&scene, w);
            masks.push(json!({"file": format!("effects/{name}"), "from": o.selection, "object_ids": o.object_ids, "splats": w.iter().filter(|v| **v > 0.0).count(), "box": bbox}));
            o.selection = name;
            o.object_ids = None;
            o.union.clear();
            o.dilate = None;
            o.bbox = bbox;
            Ok(())
        })?;
    }
    let text = graph.to_json_pretty();
    fx::Graph::from_json(&text).map_err(|e| anyhow::anyhow!("exported graph does not re-validate: {e}"))?;
    let rel = format!("effects/{id}.effect.json");
    write_atomic(&pkg.join(&rel), text.as_bytes())?;
    let scene_json = pkg.join("scene.json");
    let mut meta: Value = serde_json::from_slice(&std::fs::read(&scene_json)?)?;
    let entry = json!({"id": id, "path": rel, "autoplay": autoplay, "duration_s": graph.duration_s, "loop": graph.looping,
        "active": active_rel, "active_count": active_count});
    let list = meta.as_object_mut().context("scene.json is not an object")?.entry("effects").or_insert_with(|| json!([]));
    let list = list.as_array_mut().context("scene.json effects is not a list")?;
    list.retain(|e| e["id"].as_str() != Some(id));
    list.push(entry.clone());
    write_atomic(&scene_json, &serde_json::to_vec_pretty(&meta)?)?;
    Ok(json!({"package": abs(pkg), "effect": abs(&pkg.join(&rel)), "scene_json": abs(&scene_json), "entry": entry, "object_masks": masks}))
}

/// One byte per splat (package order): 1 when some node gives it a non-zero weight at one of
/// ACTIVE_SAMPLES times across the timeline (plus both ends), else 0.
fn active_mask(g: &Gaussians, graph: &fx::Graph, opts: &fx::EvalOpts) -> Result<Vec<u8>> {
    const ACTIVE_SAMPLES: usize = 96;
    let prep = fx::Prepared::new(g, graph, opts).map_err(eval_err)?;
    let mut active = vec![0u8; g.len()];
    for k in 0..=ACTIVE_SAMPLES {
        let f = prep.frame(graph.duration_s * k as f32 / ACTIVE_SAMPLES as f32);
        active.iter_mut().zip(&f.weight).for_each(|(a, w)| *a |= u8::from(*w > 0.0));
    }
    Ok(active)
}

fn resolve_masks(m: &mut fx::Mask, f: &mut dyn FnMut(&mut fx::ObjectMask) -> Result<()>) -> Result<()> {
    match m {
        fx::Mask::Object(o) => f(o),
        fx::Mask::Invert { mask } => resolve_masks(mask, f),
        fx::Mask::Mul { masks } => masks.iter_mut().try_for_each(|m| resolve_masks(m, f)),
        _ => Ok(()),
    }
}

/// Box around the splats weighted > 0.5 (1st-99th percentile per axis, so stray splats do not
/// blow it up), with a falloff of 10% of its largest side: the viewers' stand-in for the ids.
fn weighted_box(scene: &Scene, w: &[f32]) -> Option<fx::graph::BoxRegion> {
    let mut axes: [Vec<f32>; 3] = Default::default();
    for (p, _) in scene.g.pos.iter().zip(w).filter(|(_, w)| **w > 0.5) {
        (0..3).for_each(|k| axes[k].push(p[k]));
    }
    if axes[0].is_empty() {
        return None;
    }
    let (mut min, mut max) = ([0.0f32; 3], [0.0f32; 3]);
    for k in 0..3 {
        axes[k].sort_by(f32::total_cmp);
        let n = axes[k].len();
        (min[k], max[k]) = (axes[k][n / 100], axes[k][(n * 99 / 100).min(n - 1)]);
    }
    let side = (0..3).map(|k| max[k] - min[k]).fold(0.0f32, f32::max);
    Some(fx::graph::BoxRegion { min, max, falloff: (side * 0.1).max(0.02) })
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
}

fn stats(v: &[f64]) -> Value {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let mean = s.iter().sum::<f64>() / s.len().max(1) as f64;
    json!({"mean": mean, "min": s.first(), "median": s.get(s.len() / 2), "max": s.last(), "n": s.len()})
}

/// Peak resident memory of this process (VmHWM), MiB.
fn peak_rss_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: f64 = s.lines().find(|l| l.starts_with("VmHWM:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

/// Board memory in use (nvidia-smi), MiB; None without an NVIDIA GPU.
fn board_mib() -> Option<f64> {
    let o = Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().ok()?;
    String::from_utf8_lossy(&o.stdout).lines().next()?.trim().parse().ok()
}

/// Evaluator and render timings on one scene: CPU reference evaluator, the WGSL evaluator on the
/// render device, rest vs effect render of the same camera, memory.
fn bench(scene_p: &Path, graph_p: &Path, reps: usize, out: Option<&Path>, size: &Size, common: &Common) -> Result<Value> {
    let adapter = raster::pin_adapter();
    let board0 = board_mib();
    let t0 = Instant::now();
    let scene = open_scene(scene_p, pick(common, Pick::Active))?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let (graph, _) = load_graph(graph_p)?;
    let opts = eval_opts(&scene, &graph, common)?;
    let reps = reps.clamp(1, 100);
    let times: Vec<f32> = [0.25f32, 0.5, 0.75].iter().map(|u| u * graph.duration_s).collect();
    let t0 = Instant::now();
    let prep = fx::Prepared::new(&scene.g, &graph, &opts).map_err(eval_err)?;
    let prepare_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut cpu = Vec::new();
    let mut selected = 0;
    for _ in 0..reps {
        for &t in &times {
            let t0 = Instant::now();
            let f = prep.frame(t);
            cpu.push(t0.elapsed().as_secs_f64() * 1e3);
            selected = selected.max(f.selected);
        }
    }
    let rss_cpu = peak_rss_mib();
    let (w, h) = size.resolve(1280)?;
    let cam = scene.cams[0];
    let mut raster = Raster::new();
    // Baseline: the rest splats, no effect (first render warms up the pipelines).
    raster.render(raster::raw(&scene.g, None), &cam, scene.up, w, h)?;
    let mut base = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        raster.render_again(&cam, scene.up, w, h)?;
        base.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let gfx = raster.gpu_effect(&scene.g, &graph, &opts).map_err(|e| anyhow::anyhow!("{e}"))?;
    let _ = raster.gpu_frame(&gfx, times[0]);
    let (mut gpu, mut edited) = (Vec::new(), Vec::new());
    let mut invalid = 0;
    for _ in 0..reps {
        for &t in &times {
            let t0 = Instant::now();
            let (inv, _, f) = raster.gpu_frame(&gfx, t);
            gpu.push(t0.elapsed().as_secs_f64() * 1e3);
            invalid = invalid.max(inv);
            drop(f);
            let t0 = Instant::now();
            raster.render_again(&cam, scene.up, w, h)?;
            edited.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    let board1 = board_mib();
    let result = json!({
        "effect_id": graph.effect_id, "scene": abs(scene_p), "package": scene.package, "adapter": adapter,
        "splats": scene.g.len(), "selected": selected, "invalid_max": invalid, "reps": reps, "times_s": times, "size": [w, h],
        "load_ms": load_ms, "cpu_prepare_ms": prepare_ms, "cpu_frame_ms": stats(&cpu), "cpu_threads": std::thread::available_parallelism().map_or(0, |n| n.get()),
        "gpu_frame_ms": stats(&gpu), "render_rest_ms": stats(&base), "render_effect_ms": stats(&edited),
        "peak_rss_mib_after_cpu": rss_cpu, "peak_rss_mib": peak_rss_mib(),
        "board_mib_idle": board0, "board_mib_peak_seen": board1, "board_mib_added": board0.zip(board1).map(|(a, b)| b - a),
        "note": "gpu_frame_ms includes reading back two counts (a sync point); render_*_ms is one camera at size",
    });
    if let Some(o) = out {
        if let Some(p) = o.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(o, serde_json::to_vec_pretty(&result)?)?;
    }
    Ok(json!({"bench": result, "out": out.map(abs)}))
}

fn bench_cloud(scene_p: &Path, graph_p: &Path, reps: usize, out: Option<&Path>, size: &Size, common: &Common) -> Result<Value> {
    let pkg = scene::resolve(scene_p, pick(common, Pick::Active))?;
    let dst = out.map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("effect-bench.json"));
    let job = cloud::job(&dst)?;
    let s = job.add_scene(&pkg)?;
    let g = job.add_file(graph_p, "graph.effect.json")?;
    let mut args: Vec<String> = vec!["--json".into(), "effect".into(), "bench".into(), s.to_string_lossy().into(), g.to_string_lossy().into(),
        "-o".into(), job.outputs.join("bench.json").to_string_lossy().into(), "--reps".into(), reps.to_string(), "--cov".into(), common.cov.clone()];
    args.extend(size.args());
    let cost = job.run("effect_bench", args)?;
    job.take("bench.json", &dst)?;
    job.cleanup();
    let mut result: Value = serde_json::from_slice(&std::fs::read(&dst)?)?;
    result["scene"] = json!(abs(scene_p));
    result["package"] = json!(abs(&pkg));
    result["cloud"] = cost;
    std::fs::write(&dst, serde_json::to_vec_pretty(&result)?)?;
    Ok(json!({"bench": result, "out": abs(&dst)}))
}
