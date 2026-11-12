//! `splat export-web`: turn a trained splat + its COLMAP capture into a web scene package
//! (scene.spz + scene.json) in three.js coordinates: right-handed, +Y up, meters, floor at y=0.
//!
//! Steps: scene frame from the capture (splat-shots), floor = lowest large near-horizontal plane
//! (RANSAC over SfM points only), metric scale from capture-camera height above
//! the floor (handheld ~1.5 m), axis alignment to the dominant wall, crop splats no capture camera
//! sees, then SPZ (v3) + lossless overflow sidecars ([`overflow`]) + a Spark .RAD LoD stream ([`rad`])
//! + scene.json (SHARED FORMATS, schema 1)
//! and, when the run has a TSDF mesh, collider.glb re-rooted into the scene frame.

pub mod cap;
pub mod collision;
pub mod first;
pub mod game;
pub mod geom;
pub mod glb;
pub mod neural;
pub mod overflow;
pub mod ply;
pub mod rad;
pub mod shrot;
pub mod spz;
pub mod viewer;
mod viewer_collide;
pub mod sog;

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat { Spz, Sog }

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat3, DMat4, DQuat, DVec3};
use rayon::prelude::*;
use serde_json::{Value, json};
use splat_shots::scene::{SceneFrame, load_scene, quantile};

use crate::geom::{Plane, RansacOpts, Rng, ransac};

/// Typical eye-level phone height while walking a room.
pub const HANDHELD_CAMERA_HEIGHT_M: f64 = 1.5;

#[derive(clap::Args, Debug, Clone)]
pub struct ExportWebArgs {
    /// A `vauss run` dir, a splat PLY, or a dir holding splat.ply.
    pub input: PathBuf,
    /// Output scene directory.
    #[arg(long)]
    pub out: PathBuf,
    /// COLMAP capture dir; only needed for a bare PLY that isn't inside a `vauss run` dir.
    #[arg(long)]
    pub data: Option<PathBuf>,
    /// Convert the PLY as-is, preserving its coordinate frame; --out is a file.
    #[arg(long)]
    pub raw: bool,
    /// Output format; SOG requires Node>=22 and @playcanvas/splat-transform3.10.0.
    #[arg(long, value_enum, default_value_t = ExportFormat::Spz)]
    pub format: ExportFormat,
    /// Path to the pinned splat-transform executable (SOG only).
    #[arg(long, default_value = "splat-transform")]
    pub sog_encoder: PathBuf,
    /// SOG WebGPU adapter index; omitted uses the CPU encoder.
    #[arg(long)]
    pub sog_gpu: Option<u32>,
    /// SPZ container version (v4 requires a Zstd-capable viewer).
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(3..=4))]
    pub spz_version: u32,
    /// Quantization bits for SH degree1;8 retains the most appearance detail.
    #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=8))]
    pub sh1_bits: u8,
    /// Quantization bits for SH degrees2 and3.
    #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=8))]
    pub sh_rest_bits: u8,
    /// Use upstream double-rounding instead of the default symmetric SH rounding (SPZ only).
    #[arg(long)]
    pub upstream_sh_rounding: bool,
    /// The .RAD LoD stream: built after scene.spz + scene.json are published (the package is
    /// viewable meanwhile), inline before them, or skipped (quick previews).
    #[arg(long, value_enum, default_value_t = LodMode::After)]
    pub lod: LodMode,
    /// Refused: pruning a trained model to a budget without retraining lost 9.9-15 dB held-out
    /// (4.66M -> 1M on one scene; -11.6 dB on a room). Retrain at the cap instead (`vauss refine <run> --cap-max N`).
    #[arg(long, default_value_t = 0, hide = true)]
    pub max_splats: usize,
    /// Skip collision/ (physics meshes, built on CPU after scene.json is published).
    #[arg(long)]
    pub no_collision: bool,
}

/// When the Spark .RAD LoD stream (scene.rad) is built. Its build is single-threaded and most of
/// an export (house 1M: 16 of 19 s; 279k preview: 3.5 of 4.8 s), and the viewer shows scene.spz
/// whenever scene.json has no `splat_lod`.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LodMode {
    /// scene.spz + scene.json first, then scene.rad and scene.json again with `splat_lod`.
    #[default]
    After,
    /// scene.rad before scene.json is written (one scene.json, with `splat_lod`).
    Inline,
    /// No scene.rad (quick previews: a small SPZ paints sooner than the LoD stream).
    Skip,
}

pub fn run(a: &ExportWebArgs) -> Result<Value> {
    ensure!(a.max_splats == 0,
        "--max-splats is no longer supported: pruning a trained model without retraining loses 10-15 dB held-out. \
         Retrain at the budget instead: `vauss refine <run> --cap-max {}` (cloud runs above 1M already ship a 1M refine as exports/web)", a.max_splats);
    ensure!(!a.upstream_sh_rounding || a.format == ExportFormat::Spz, "--upstream-sh-rounding requires SPZ");
    if a.raw {
        let start = Instant::now();
        let g = ply::read_ply(&a.input)?;
        let options = spz::Options { version: a.spz_version, sh1_bits: a.sh1_bits, sh_rest_bits: a.sh_rest_bits, antialiased: g.antialiased, symmetric_sh: !a.upstream_sh_rounding };
        let bytes = match a.format {
            ExportFormat::Spz => spz::write_spz_with_options(&g, &a.out, options)?,
            ExportFormat::Sog => sog::write_sog(&g, &a.out, &a.sog_encoder, a.sog_gpu)?,
        };
        return Ok(json!({"path": a.out, "format": format!("{:?}", a.format).to_lowercase(), "bytes": bytes, "splats": g.len(), "spz_version": if a.format == ExportFormat::Spz { Some(options.version) } else { None },
            "sh1_bits": if a.format == ExportFormat::Spz { Some(options.sh1_bits) } else { None }, "sh_rest_bits": if a.format == ExportFormat::Spz { Some(options.sh_rest_bits) } else { None }, "antialiased": g.antialiased, "symmetric_sh": if a.format == ExportFormat::Spz { Some(options.symmetric_sh) } else { None },
            "position_fractional_bits": if a.format == ExportFormat::Spz { Some(spz::position_fractional_bits(&g)?) } else { None }, "position_clamped": if a.format == ExportFormat::Spz { Some(spz::position_clamped(&g)) } else { None }, "coordinate_transform": "identity", "elapsed_s": start.elapsed().as_secs_f64()}));
    }
    let options = SceneOptions { spz_version: a.spz_version, sh1_bits: a.sh1_bits, sh_rest_bits: a.sh_rest_bits, symmetric_sh: !a.upstream_sh_rounding, format: a.format, sog_encoder: a.sog_encoder.clone(), sog_gpu: a.sog_gpu, lod: a.lod, max_splats: a.max_splats, collision: !a.no_collision };
    export_web_with_options(&a.input, a.data.as_deref(), &a.out, &options)
}

/// Scene-package export settings; `Default` is what `vauss view` and a plain `splat export-web` use.
#[derive(Debug, Clone)]
pub struct SceneOptions {
    pub spz_version: u32,
    pub sh1_bits: u8,
    pub sh_rest_bits: u8,
    pub symmetric_sh: bool,
    pub format: ExportFormat,
    pub sog_encoder: PathBuf,
    pub sog_gpu: Option<u32>,
    pub lod: LodMode,
    /// Splat budget of the package (see [`cap`]); 0 keeps every splat.
    pub max_splats: usize,
    /// Add collision/ after the package is published ([`export_web_with_options`] only; previews skip it).
    pub collision: bool,
}
impl Default for SceneOptions {
    fn default() -> Self {
        Self { spz_version: 3, sh1_bits: 6, sh_rest_bits: 6, symmetric_sh: true, format: ExportFormat::Spz, sog_encoder: "splat-transform".into(), sog_gpu: None, lod: LodMode::After, max_splats: 0, collision: false }
    }
}

/// A published scene package whose scene.rad is still to be built ([`LodMode::After`]).
pub struct PendingLod {
    g: ply::Gaussians,
    out: PathBuf,
    scene: Value,
    t0: Instant,
}

impl PendingLod {
    /// Build scene.rad, then rewrite scene.json with `splat_lod` (both atomically); returns the new scene.json.
    pub fn finish(self) -> Result<Value> {
        let PendingLod { g, out, mut scene, t0 } = self;
        scene["splat_lod"] = rad::write_rad(&g, &out, "scene")?;
        scene["stats"]["total_s"] = json!(r2(t0.elapsed().as_secs_f64()));
        write_atomic(&out.join("scene.json"), &serde_json::to_vec_pretty(&scene)?)?;
        Ok(scene)
    }
}

/// Write `path` via a temporary file in the same dir and a rename, so readers never see a partial file.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).with_context(|| format!("write {}", path.display()))
}

/// What a scene is made of: splat PLY, COLMAP capture, optional TSDF mesh.
struct Inputs {
    ply: PathBuf,
    data: PathBuf,
    mesh: Option<PathBuf>,
}

/// The newest candidate's splat PLY: within a candidate `splat.ply` wins over `splat.ply.zst`; across candidates
/// the newest file wins, whichever name it has.
fn latest_candidate_ply(artifacts: &Path) -> Option<PathBuf> {
    std::fs::read_dir(artifacts)
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("candidate_"))
        .filter_map(|e| splat_shots::splat_ply_in(&e.path()))
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
        .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, p)| p)
}

fn latest_artifact(artifacts: &Path, prefix: &str, file: &str) -> Option<PathBuf> {
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(artifacts)
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
        .map(|e| e.path())
        .filter(|p| p.join(file).exists())
        .filter_map(|p| Some((std::fs::metadata(p.join(file)).ok()?.modified().ok()?, p)))
        .collect();
    v.sort();
    v.pop().map(|(_, p)| p)
}

/// The splat PLY and the mesh (if any) an export of `input` reads, e.g. to fetch a `--cloud` run's files first.
pub fn input_files(input: &Path, data: Option<&Path>) -> Result<(PathBuf, Option<PathBuf>)> {
    let i = resolve(input, data)?;
    Ok((i.ply, i.mesh))
}

/// A `vauss run --cloud` result left on fal: a small stub (splat_core::fal::remote_stub) in place of the file.
fn kept_on_fal(path: &Path) -> bool {
    let stub = std::fs::metadata(path).is_ok_and(|m| m.len() <= 256)
        && std::fs::read(path).is_ok_and(|b| b.starts_with(b"splat-fal-remote v1 "));
    if stub {
        eprintln!("{}: kept on fal (`vauss pull <run>` downloads it); exporting without collider.glb", path.display());
    }
    stub
}

fn resolve(input: &Path, data: Option<&Path>) -> Result<Inputs> {
    // `vauss run` dir (or anything inside one): artifacts/{candidate_*,sfm_*,mesh_*}
    let run_root = input.ancestors().find(|d| d.join("artifacts").is_dir() && d.join("run.json").exists());
    if let Some(root) = run_root {
        let arts = root.join("artifacts");
        let ply = if splat_shots::is_splat_ply_name(input) && input.is_file() {
            input.to_path_buf()
        } else {
            latest_candidate_ply(&arts).context("run has no trained splat yet")?
        };
        let data = match data {
            Some(d) => d.to_path_buf(),
            None => latest_artifact(&arts, "sfm_", "sparse").context("run has no SfM artifact")?,
        };
        let mesh = latest_artifact(&arts, "mesh_", "mesh.glb").map(|d| d.join("mesh.glb"));
        return Ok(Inputs { ply, data, mesh });
    }
    let ply = splat_shots::find_ply(input).with_context(|| format!("no splat PLY at {}", input.display()))?;
    let data = match data {
        Some(d) => d.to_path_buf(),
        // Next to the PLY, one level up, or in a subdir of the PLY's dir (e.g. a `capture` link).
        None => {
            let dir = ply.parent().unwrap_or(Path::new("."));
            let mut cands = vec![dir.to_path_buf()];
            cands.extend(dir.parent().map(Path::to_path_buf));
            if let Ok(rd) = std::fs::read_dir(dir) {
                cands.extend(rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
            }
            cands.into_iter().find(|d| splat_shots::scene::find_sparse_dir(d).is_ok())
        }
            .context("no COLMAP capture found next to the PLY; pass --data <capture dir>")?,
    };
    let mesh = ply.parent().map(|d| d.join("mesh.glb")).filter(|p| p.exists());
    Ok(Inputs { ply, data, mesh })
}

/// Title / license / source note for scene.json, from an optional `scene-source.json`
/// ({"title","license","source"}) next to the input, the PLY or in the capture dir.
fn source_info(input: &Path, i: &Inputs) -> Value {
    // A `vauss run` dir records its source video; a sidecar may sit next to it.
    let video_dir = input
        .ancestors()
        .find_map(|d| serde_json::from_slice::<Value>(&std::fs::read(d.join("run.json")).ok()?).ok())
        .and_then(|r| r["source"].as_str().map(|s| PathBuf::from(s).parent().map(Path::to_path_buf)))
        .flatten();
    [Some(input), i.ply.parent(), Some(i.data.as_path()), i.data.parent(), video_dir.as_deref()]
        .into_iter()
        .flatten()
        .map(|d| d.join("scene-source.json"))
        .find_map(|p| serde_json::from_slice::<Value>(&std::fs::read(p).ok()?).ok())
        .unwrap_or_else(|| json!({}))
}

/// Build a scene package in `out` and return its scene.json. `data` is only needed for a bare PLY.
pub fn export_web(input: &Path, data: Option<&Path>, out: &Path) -> Result<Value> {
    export_web_with_options(input, data, out, &SceneOptions::default())
}

pub fn export_web_with_options(input: &Path, data: Option<&Path>, out: &Path, o: &SceneOptions) -> Result<Value> {
    let scene = match export_web_staged(input, data, out, o)? {
        (_, Some(lod)) => lod.finish()?,
        (scene, None) => scene,
    };
    if !o.collision {
        return Ok(scene);
    }
    // After the package is published and complete: nothing viewable waits for physics.
    match collision::add_to_package(out) {
        Ok(_) => Ok(serde_json::from_slice(&std::fs::read(out.join("scene.json"))?)?),
        Err(e) => {
            eprintln!("collision: none ({e:#}); the package is unchanged");
            Ok(scene)
        }
    }
}

/// [`export_web_with_options`] up to the published package: scene.json is on disk (viewable) when
/// this returns. With [`LodMode::After`] the second value builds scene.rad and adds it to scene.json.
pub fn export_web_staged(input: &Path, data: Option<&Path>, out: &Path, o: &SceneOptions) -> Result<(Value, Option<PendingLod>)> {
    let format = o.format;
    let t0 = Instant::now();
    let inputs = resolve(input, data)?;
    let info = source_info(input, &inputs);
    let ply_path = inputs.ply.clone();
    let frame = load_scene(&inputs.data, splat_shots::scene_transform_near(&ply_path).as_deref())?;
    // Every package of a run shares one frame: the capture's pinned frame.json (see [`capture_frame`])
    // when present, else one fitted to this capture.
    let framing = match read_frame(&inputs.data)? {
        Some(f) => f,
        None => fit_frame(&frame)?,
    };
    let mut g = ply::read_ply(&ply_path)?;
    if g.is_empty() {
        bail!("{}: no splats", ply_path.display());
    }
    let n_in = g.len();
    let t_read = t0.elapsed().as_secs_f64();

    let m = framing.m;
    let scale = m.x_axis.truncate().length();
    let r = DMat3::from_mat4(m) * (1.0 / scale);
    let cams = scene_cams(&frame, &m);

    // Crop by visibility, not by a room box: a box sized from the floor/camera extent cut the far
    // half of real rooms and most of outdoor scenes (the metric scale can be off by 4x outdoors).
    let visible = visible_from_capture(&g, &frame);

    // ---- transform splats ----
    to_scene_frame(&mut g, &m);

    // ---- crop: splats no capture camera sees, and fully transparent specks ----
    let keep: Vec<bool> = (0..g.len()).into_par_iter().map(|i| visible[i] && g.alpha(i) >= 1.0 / 255.0).collect();
    g.retain(&keep);
    let n_cropped = g.len();
    // ---- optional splat budget (export-web --max-splats) ----
    cap::cap(&mut g, o.max_splats);
    let n_out = g.len();

    // ---- package ----
    std::fs::create_dir_all(out)?;
    let options = spz::Options { version: o.spz_version, sh1_bits: o.sh1_bits, sh_rest_bits: o.sh_rest_bits, antialiased: g.antialiased, symmetric_sh: o.symmetric_sh };
    let (splat_file, spz_bytes, splat_overflow) = match format {
        ExportFormat::Spz => {
            let pkg = overflow::write_package(&g, out, "scene", options)?;
            ("scene.spz", pkg["bytes"]["total"].as_u64().unwrap_or(0), Some(pkg))
        }
        ExportFormat::Sog => ("scene.sog", sog::write_sog(&g, &out.join("scene.sog"), &o.sog_encoder, o.sog_gpu)?, None),
    };
    let splat_neural = neural::write_sidecar(&g, out, "scene", r)?;
    // Coarse-first LoD stream for the viewer (Spark pages it over HTTP Range). Neural scenes need
    // every splat decoded with its latent, so they keep the full download only.
    let lod = if g.neural.is_none() { o.lod } else { LodMode::Skip };
    let splat_lod = if lod == LodMode::Inline { Some(rad::write_rad(&g, out, "scene")?) } else { None };
    // A ~40K-splat SPZ drawn before the .RAD / full SPZ on slow links (see splat_webexport::first);
    // only alongside a .RAD stream, which is what the viewer's coarse-first path loads.
    let splat_first = if format == ExportFormat::Spz && lod != LodMode::Skip { first::write_first(&g, out, "scene", options)? } else { None };
    if splat_first.is_none() {
        let _ = std::fs::remove_file(out.join("scene.first.spz"));
    }
    // A `vauss run --cloud` mesh left on fal is a small stub here (`vauss pull <run>` downloads it): no collider.
    let collider = match inputs.mesh.as_ref().filter(|m| !kept_on_fal(m)) {
        Some(mesh) => {
            glb::write_with_root_matrix(mesh, &out.join("collider.glb"), &m.to_cols_array())?;
            Some("collider.glb")
        }
        None => None,
    };
    let (mut smin, mut smax) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
    for p in &g.pos {
        let v = DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
        smin = smin.min(v);
        smax = smax.max(v);
    }
    let focus = framing.focus;
    let look = |c: &SceneCam| c.0 + c.1 * c.0.distance(focus).max(0.5);
    let stride = cams.len().div_ceil(32).max(1);
    let capture: Vec<Value> = cams
        .iter()
        .step_by(stride)
        .map(|c| json!({"position": r3(c.0), "look_at": r3(look(c)), "vfov_deg": r1(c.2), "image": c.3}))
        .collect();
    let name = out.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "scene".into());
    let scene = json!({
        "schema": 1,
        "name": name,
        "title": info.get("title").cloned().unwrap_or_else(|| json!(name)),
        "splat": splat_file,
        // SPZ only: the overflow sidecars that complete scene.spz (see splat_webexport::overflow).
        "splat_overflow": splat_overflow,
        // Neural-appearance models: latents + shared decoder (see splat_webexport::neural).
        "splat_neural": splat_neural,
        // Spark .RAD LoD tree of the same splats (see splat_webexport::rad); the viewer streams it
        // coarse-first when present and falls back to `splat`.
        "splat_lod": splat_lod,
        // ~40K-splat SPZ for the first paint (see splat_webexport::first); absent on small scenes.
        "splat_first": splat_first,
        // TSDF mesh of the capture in this frame (for collisions); absent when the run has none.
        "collider": collider,
        "coords": "y_up_meters",
        "meters_per_unit": 1.0,
        "scale_estimate": framing.fields["scale_estimate"],
        "up": [0.0, 1.0, 0.0],
        "floor": framing.fields["floor"],
        "walls": framing.fields["walls"],
        "focus": framing.fields["focus"],
        "radius": framing.fields["radius"],
        "start_camera": framing.fields["start_camera"],
        "capture_cameras": capture,
        "bounds": {"min": r3(smin), "max": r3(smax)},
        // Column-major 4x4 (three.js Matrix4.fromArray) mapping the source splat/COLMAP frame to this one.
        "transform_from_source": m.to_cols_array().iter().map(|v| (v * 1e9).round() / 1e9).collect::<Vec<_>>(),
        "source": {"ply": ply_path, "data": inputs.data, "mesh": inputs.mesh, "note": info.get("source"), "exporter": "splat export-web"},
        "license": info.get("license").cloned().unwrap_or_else(|| json!("unknown")),
        // The trainer's rasterizer settings (PLY comment SplatRenderMode); the viewer matches them.
        "render_mode": g.render_mode.as_str(),
        "stats": {"splats_in": n_in, "splats_out": n_out, "sh_degree": g.sh_degree, "splat_bytes": spz_bytes, "spz_bytes": if format == ExportFormat::Spz { Some(spz_bytes) } else { None }, "spz_version": if format == ExportFormat::Spz { Some(options.version) } else { None }, "position_fractional_bits": if format == ExportFormat::Spz { Some(overflow::POSITION_FRACTIONAL_BITS) } else { None }, "sh1_bits": if format == ExportFormat::Spz { Some(options.sh1_bits) } else { None }, "sh_rest_bits": if format == ExportFormat::Spz { Some(options.sh_rest_bits) } else { None }, "antialiased": options.antialiased, "symmetric_sh": if format == ExportFormat::Spz { Some(options.symmetric_sh) } else { None },
            "sfm_points": frame_points(&frame).len(), "plane_points": frame_points(&frame).len(),
            "read_s": r2(t_read), "total_s": r2(t0.elapsed().as_secs_f64()),
            // Seconds from start until scene.json (the viewable package) was written.
            "publish_s": r2(t0.elapsed().as_secs_f64())},
    });
    let mut scene = scene;
    // Median capture camera (see viewer::colmap_capture): lets viewers cap their FOV to the capture's.
    if let Some((aspect, height, focal)) = viewer::colmap_capture(&inputs.data) {
        scene["capture_aspect"] = json!((aspect * 1e5).round() / 1e5);
        scene["capture_height"] = json!(height);
        scene["capture_focal_px"] = json!(focal);
    }
    // Splats after the camera crop, and how many the package budget then dropped.
    scene["stats"]["splats_cropped"] = json!(n_cropped);
    scene["stats"]["splat_cap"] = json!({"max": o.max_splats, "method": cap::METHOD, "dropped": n_cropped - n_out});
    // collision/ is built after publishing (see [`collision::add_to_package`]); an earlier export's
    // would not match these splats.
    let _ = std::fs::remove_dir_all(out.join("collision"));
    // A stale scene.rad from an earlier export would not match these splats.
    if splat_lod.is_none() {
        let _ = std::fs::remove_file(out.join("scene.rad"));
    }
    write_atomic(&out.join("scene.json"), &serde_json::to_vec_pretty(&scene)?)?;
    let pending = (lod == LodMode::After).then(|| PendingLod { g, out: out.to_path_buf(), scene: scene.clone(), t0 });
    Ok((scene, pending))
}

/// Map splats (positions, scales, rotations and SH) from the source frame through the similarity `m`.
pub(crate) fn to_scene_frame(g: &mut ply::Gaussians, m: &DMat4) {
    let scale = m.x_axis.truncate().length();
    let r = DMat3::from_mat4(*m) * (1.0 / scale);
    let q = DQuat::from_mat3(&r);
    let ln_s = scale.ln() as f32;
    let shr = shrot::ShRotation::new(r);
    let k = g.sh_coeffs() * 3;
    g.pos.par_iter_mut().for_each(|p| {
        let v = m.transform_point3(DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64));
        *p = [v.x as f32, v.y as f32, v.z as f32];
    });
    g.scale.par_iter_mut().for_each(|s| s.iter_mut().for_each(|v| *v += ln_s));
    g.rot.par_iter_mut().for_each(|w| {
        let o = DQuat::from_xyzw(w[1] as f64, w[2] as f64, w[3] as f64, w[0] as f64);
        let n = (q * o).normalize();
        *w = [n.w as f32, n.x as f32, n.y as f32, n.z as f32];
    });
    if k > 0 {
        let coeffs = g.sh_coeffs();
        g.rest.par_chunks_mut(k).for_each(|c| shr.apply(c, coeffs));
    }
}

/// File in a capture (SfM) dir that pins its export frame; see [`capture_frame`].
pub const FRAME_FILE: &str = "frame.json";

/// The scene.json fields that depend only on the capture (never the splats); with the transform
/// they make up a capture's export frame.
const FRAME_FIELDS: [&str; 6] = ["scale_estimate", "floor", "walls", "focus", "radius", "start_camera"];

/// An export frame: source -> scene transform (world_from_src) and [`FRAME_FIELDS`].
struct Framing {
    m: DMat4,
    focus: DVec3,
    fields: serde_json::Map<String, Value>,
}

/// Capture camera in the scene frame: position, forward, vfov (deg), image name.
type SceneCam = (DVec3, DVec3, f64, String);

fn scene_cams(frame: &SceneFrame, m: &DMat4) -> Vec<SceneCam> {
    let r = DMat3::from_mat4(*m);
    frame.cameras.iter().map(|c| (m.transform_point3(c.center()), (r * c.forward()).normalize(), c.vfov_deg(), c.name.clone())).collect()
}

impl Framing {
    fn to_json(&self) -> Value {
        let mut v = json!({"schema": 1, "transform_from_source": self.m.to_cols_array()});
        v.as_object_mut().expect("object").extend(self.fields.clone());
        v
    }

    fn from_json(v: &Value) -> Result<Self> {
        let m: Vec<f64> = v["transform_from_source"].as_array().context("frame: no transform_from_source")?
            .iter().filter_map(Value::as_f64).collect();
        ensure!(m.len() == 16, "frame: transform_from_source is not a 4x4");
        let f: Vec<f64> = v["focus"].as_array().context("frame: no focus")?.iter().filter_map(Value::as_f64).collect();
        ensure!(f.len() == 3, "frame: focus is not a point");
        let fields = FRAME_FIELDS.iter().map(|k| Ok((k.to_string(), v.get(*k).cloned().with_context(|| format!("frame: no {k}"))?)))
            .collect::<Result<_>>()?;
        Ok(Self { m: DMat4::from_cols_slice(&m), focus: DVec3::from_slice(&f), fields })
    }
}

/// `data`'s pinned export frame (its [`FRAME_FILE`]), if any.
fn read_frame(data: &Path) -> Result<Option<Framing>> {
    let path = data.join(FRAME_FILE);
    if !path.is_file() {
        return Ok(None);
    }
    let v: Value = serde_json::from_slice(&std::fs::read(&path)?).with_context(|| format!("{}", path.display()))?;
    Framing::from_json(&v).map(Some).with_context(|| format!("{}", path.display()))
}

/// The export frame of the capture (COLMAP dir) `data` as frame.json content: its pinned
/// [`FRAME_FILE`] when present, else fitted to the capture. Written into another capture in the same
/// world (e.g. a full SfM aligned onto a quick one), every package exported from either gets the same
/// transform_from_source, focus, radius and start camera.
pub fn capture_frame(data: &Path) -> Result<Value> {
    if let Some(f) = read_frame(data)? {
        return Ok(f.to_json());
    }
    Ok(fit_frame(&load_scene(data, None)?)?.to_json())
}

/// Scene radius in meters the export frame may imply. The camera-height scale is a rough guess;
/// outside this band it is wrong, and past ~1 km the splats leave SPZ's position range.
const SCENE_RADIUS_M: std::ops::RangeInclusive<f64> = 0.05..=200.0;

/// Fit the export frame to a capture: floor, metric scale, axes, origin and start view.
fn fit_frame(frame: &SceneFrame) -> Result<Framing> {
    let mut frame = frame.clone();
    let frame_repaired = repair_frame(&mut frame);
    let frame = &frame;
    // ---- floor ----
    let pts = frame_points(frame);
    let mut floor = find_floor(&pts, frame);

    // ---- scale ----
    let cam_heights = |floor: &Option<Floor>| {
        let up = floor.as_ref().map(|f| f.plane.n).unwrap_or(frame.up);
        let floor_h = floor.as_ref().map(|f| f.plane.h).unwrap_or_else(|| up.dot(frame.target) + frame.ground);
        let cam_h: Vec<f64> = frame.centers().iter().map(|c| c.dot(up) - floor_h).collect();
        (up, floor_h, quantile(&mut cam_h.clone(), 0.5), cam_h)
    };
    let (mut up, mut floor_h, mut cam_h_med, mut cam_h) = cam_heights(&floor);
    // A plane the cameras sit below (a tabletop or ceiling under sparse SfM points) is not the floor:
    // fall back to the camera-derived frame rather than failing the export. A fitted floor plane is
    // real geometry; the camera-derived floor (5th-percentile point height) is not, so it must leave
    // the cameras clearly above it.
    let plausible = |h: f64, plane: bool| h > if plane { 0.02 } else { 0.1 } * frame.radius;
    if !plausible(cam_h_med, true) && floor.is_some() {
        floor = None;
        (up, floor_h, cam_h_med, cam_h) = cam_heights(&floor);
    }
    let iqr = quantile(&mut cam_h.clone(), 0.75) - quantile(&mut cam_h.clone(), 0.25);
    let mut confidence = (1.0 - 2.0 * iqr / cam_h_med).clamp(0.2, 0.8) * if floor.is_some() { 1.0 } else { 0.5 };
    let mut method = format!("camera_height_{HANDHELD_CAMERA_HEIGHT_M:.2}m");
    // Cameras at (or below) the floor estimate: no usable height. Assume a handheld room capture
    // whose cameras stand half their viewing distance above the floor, and put the floor there.
    let ref_h = if plausible(cam_h_med, floor.is_some()) {
        cam_h_med
    } else {
        method = "scene_radius_fallback".into();
        confidence = 0.2;
        cam_h_med = 0.5 * frame.radius;
        floor_h = quantile(&mut frame.centers().iter().map(|c| c.dot(up)).collect::<Vec<_>>(), 0.5) - cam_h_med;
        cam_h_med
    };
    // Keep the implied scene size sane whatever the height estimate said.
    let raw = (HANDHELD_CAMERA_HEIGHT_M / ref_h).clamp(SCENE_RADIUS_M.start() / frame.radius, SCENE_RADIUS_M.end() / frame.radius);
    if raw != HANDHELD_CAMERA_HEIGHT_M / ref_h {
        method = format!("{method}_clamped");
        confidence = 0.2;
    }
    // ln(scale) snapped to SPZ's 1/16 log-scale step (at most ~3% change to a rough estimate): log scales
    // the trainer snapped onto that grid (splat-train qat.rs) stay on it after the re-framing shift.
    let steps = spz::LOG_SCALE_STEPS;
    let scale = ((raw.ln() * steps).round() / steps).exp();

    // ---- orientation: up -> +Y, dominant wall normal -> +X ----
    let floor_pts: Vec<DVec3> = floor.as_ref().map(|f| f.inliers.clone()).unwrap_or_default();
    let walls_src = find_walls(&pts, up, floor_h, scale, frame);
    let ex = walls_src
        .first()
        .map(|w| w.n)
        .or_else(|| principal_horizontal(&floor_pts, up))
        .unwrap_or_else(|| splat_shots::scene::horizontal_basis(up).0);
    let ex = (ex - ex.dot(up) * up).normalize();
    let ez = ex.cross(up);
    let r = DMat3::from_cols(ex, up, ez).transpose(); // rows = new axes
    // Origin: floor centre (robust extent of floor inliers, else cameras) at y = 0.
    let horiz: Vec<DVec3> = if floor_pts.len() > 100 { floor_pts.clone() } else { frame.centers() };
    let lo_hi = |f: fn(DVec3) -> f64| {
        let mut v: Vec<f64> = horiz.iter().map(|p| f(r * *p)).collect();
        (quantile(&mut v, 0.02), quantile(&mut v.clone(), 0.98))
    };
    let (x0, x1) = lo_hi(|p| p.x);
    let (z0, z1) = lo_hi(|p| p.z);
    let origin_rot = DVec3::new(0.5 * (x0 + x1), floor_h, 0.5 * (z0 + z1));
    // world_from_src: p' = s * (R p - origin_rot)
    let m = DMat4::from_scale(DVec3::splat(scale)) * DMat4::from_translation(-origin_rot) * DMat4::from_mat3(r);
    let tp = |p: DVec3| m.transform_point3(p);

    let floor_ext = [(x0 - origin_rot.x) * scale, (z0 - origin_rot.z) * scale, (x1 - origin_rot.x) * scale, (z1 - origin_rot.z) * scale];
    let room_w = floor_ext[2] - floor_ext[0];
    let room_d = floor_ext[3] - floor_ext[1];
    if !(1.5..=40.0).contains(&room_w.max(room_d)) {
        confidence *= 0.5; // implausible room size for the assumed camera height
    }

    let cams = scene_cams(frame, &m);
    let focus = {
        let f = tp(frame.target);
        DVec3::new(f.x, f.y.max(0.0), f.z)
    };
    // Start view: a capture camera looking at the focus from far away (overview).
    let start = cams
        .iter()
        .filter(|c| c.1.dot((focus - c.0).normalize_or_zero()) > 0.8)
        .max_by(|a, b| horiz_dist(a.0, focus).total_cmp(&horiz_dist(b.0, focus)))
        .or(cams.first())
        .context("no capture cameras")?;
    let look = |c: &SceneCam| c.0 + c.1 * c.0.distance(focus).max(0.5);
    let walls: Vec<Value> = walls_src
        .iter()
        .map(|w| {
            // three.js Plane convention: normal . p + d = 0, normal pointing into the room.
            let n = r * w.n;
            let p_src = w.n * w.h;
            let p = tp(p_src);
            let mut n = DVec3::new(n.x, 0.0, n.z).normalize();
            let mut d = -n.dot(p);
            let cam_mean = cams.iter().map(|c| c.0).sum::<DVec3>() / cams.len() as f64;
            if n.dot(cam_mean) + d < 0.0 {
                n = -n;
                d = -d;
            }
            json!({"normal": r3(n), "d": r4(d), "inliers": w.inliers})
        })
        .collect();
    let floor_inlier_frac = floor.as_ref().map(|f| f.plane.inliers as f64 / pts.len() as f64).unwrap_or(0.0);
    let fields = json!({
        "scale_estimate": {"value": r4(scale), "method": method, "confidence": r2(confidence),
            "camera_height_median_units": r4(cam_h_med), "camera_height_iqr_units": r4(iqr),
            // The capture's focus/radius collapsed (see repair_frame) and were re-derived from SfM depth.
            "frame_repaired": frame_repaired},
        "floor": {"y": 0.0, "normal": [0.0, 1.0, 0.0], "inlier_frac": r4(floor_inlier_frac),
            "detected": floor.is_some(),
            "extent": {"min": [r3v(floor_ext[0]), r3v(floor_ext[1])], "max": [r3v(floor_ext[2]), r3v(floor_ext[3])]}},
        "walls": walls,
        "focus": r3(focus),
        "radius": r3v(frame.radius * scale),
        "start_camera": {"position": r3(start.0), "look_at": r3(look(start)), "vfov_deg": r1(start.2)},
    });
    Ok(Framing { m, focus, fields: fields.as_object().cloned().expect("object") })
}

/// Re-derive the capture focus and radius when they collapsed onto the cameras. A capture that
/// pans in place for part of the walk (76-frame room, 40 frames from one spot) pulls the optical
/// axes' convergence point onto that spot: radius 0.0006 against a 10.9 camera extent, and a
/// metric scale of ~1/radius that put every splat kilometres away. The typical depth of the SfM
/// points in front of the cameras is the scene's real size; true when the frame was repaired.
fn repair_frame(frame: &mut SceneFrame) -> bool {
    let centers = frame.centers();
    let fwd = frame.forwards();
    if centers.is_empty() || frame.points.is_empty() {
        return false;
    }
    // Subsample: this runs on every export, and only needs a robust median.
    let stride = frame.points.len().div_ceil(4096).max(1);
    let sample: Vec<DVec3> = frame.points.iter().step_by(stride).copied().collect();
    let in_view = |p: DVec3, c: DVec3, f: DVec3| {
        let v = p - c;
        v.dot(f) > 0.5 * v.length() // within ~60 deg of the axis
    };
    let mut depth: Vec<f64> = Vec::new();
    for (c, f) in centers.iter().zip(&fwd) {
        depth.extend(sample.iter().filter(|p| in_view(**p, *c, *f)).map(|p| p.distance(*c)));
    }
    let depth = quantile(&mut depth, 0.5);
    if !(depth.is_finite() && depth > 0.0) || frame.radius >= 0.2 * depth {
        return false;
    }
    // Focus: median of the points most cameras see (splat-shots' walk-through fallback).
    let vis: Vec<usize> = sample.iter().map(|p| centers.iter().zip(&fwd).filter(|(c, f)| in_view(*p, **c, **f)).count()).collect();
    let cut = quantile(&mut vis.iter().map(|v| *v as f64).collect::<Vec<_>>(), 0.75);
    let sel: Vec<DVec3> = sample.iter().zip(&vis).filter(|(_, v)| **v as f64 >= cut).map(|(p, _)| *p).collect();
    let med = |f: fn(&DVec3) -> f64| quantile(&mut sel.iter().map(f).collect::<Vec<_>>(), 0.5);
    let target = DVec3::new(med(|p| p.x), med(|p| p.y), med(|p| p.z));
    let radius = quantile(&mut centers.iter().map(|c| c.distance(target)).collect::<Vec<_>>(), 0.5);
    let collapsed = frame.radius;
    frame.target = if target.is_finite() { target } else { frame.target };
    frame.radius = if radius.is_finite() && radius >= 0.2 * depth { radius } else { depth };
    let mut near_h: Vec<f64> = frame.points.iter().filter(|p| p.distance(frame.target) < 3.0 * frame.radius).map(|p| (*p - frame.target).dot(frame.up)).collect();
    frame.ground = if near_h.is_empty() { 0.0 } else { quantile(&mut near_h, 0.05) };
    eprintln!("export-web: capture radius {collapsed:.2e} collapsed vs SfM depth {depth:.3}; using radius {:.3}", frame.radius);
    true
}

/// Splats whose 3-sigma footprint reaches into at least one capture camera's view frustum
/// (10% margin). Anything else never appeared in a training image: a floater, safe to drop.
fn visible_from_capture(g: &ply::Gaussians, frame: &SceneFrame) -> Vec<bool> {
    // (world-to-camera, tan half-fov x, tan half-fov y); COLMAP cameras look down +z.
    let views: Vec<(DMat4, f64, f64)> = frame
        .cameras
        .iter()
        .map(|c| (c.c2w.inverse(), 1.1 * c.width as f64 / (2.0 * c.fy), 1.1 * c.height as f64 / (2.0 * c.fy)))
        .collect();
    if views.is_empty() {
        return vec![true; g.len()];
    }
    (0..g.len())
        .into_par_iter()
        .map_init(
            || 0usize,
            |hint, i| {
                let p = DVec3::new(g.pos[i][0] as f64, g.pos[i][1] as f64, g.pos[i][2] as f64);
                let r = 3.0 * g.scale[i].iter().cloned().fold(f32::MIN, f32::max).exp() as f64;
                let seen = |(w2c, tx, ty): &(DMat4, f64, f64)| {
                    let q = w2c.transform_point3(p);
                    let z = q.z + r;
                    z > 0.0 && q.x.abs() <= tx * z + r && q.y.abs() <= ty * z + r
                };
                // Neighbouring splats tend to be seen by the same camera; try the last hit first.
                if seen(&views[*hint]) {
                    return true;
                }
                match views.iter().position(seen) {
                    Some(k) => {
                        *hint = k;
                        true
                    }
                    None => false,
                }
            },
        )
        .collect()
}

struct Floor {
    plane: Plane,
    inliers: Vec<DVec3>,
}

/// SfM points near the capture, in their stable source order. Floor, walls, scale and origin must
/// depend only on the capture: including trained splats makes refinement/pruning move the export
/// frame. Sparse captures fall back to the camera-derived axes and extents in the caller.
fn frame_points(frame: &SceneFrame) -> Vec<DVec3> {
    frame.points.iter().copied().filter(|p| p.distance(frame.target) < 4.0 * frame.radius).collect()
}

/// Lowest large near-horizontal plane below the capture cameras.
fn find_floor(pts: &[DVec3], frame: &SceneFrame) -> Option<Floor> {
    let up = frame.up;
    let mut cam_h: Vec<f64> = frame.centers().iter().map(|c| c.dot(up)).collect();
    let below = quantile(&mut cam_h, 0.1) - 0.05 * frame.radius;
    let thresh = 0.012 * frame.radius;
    let mut rng = Rng::new(0x5EED);
    let mut active: Vec<bool> = pts.iter().map(|p| p.dot(up) < below).collect();
    let mut found: Vec<(Plane, Vec<bool>)> = Vec::new();
    for _ in 0..6 {
        let opts = RansacOpts { iters: 1500, thresh, max_tilt_deg: 20.0, vertical: false };
        let Some((plane, mask)) = ransac(pts, &active, up, &opts, &mut rng) else { break };
        if plane.inliers < 300 {
            break;
        }
        for (a, m) in active.iter_mut().zip(&mask) {
            *a &= !*m;
        }
        found.push((plane, mask));
    }
    let best = found.iter().map(|f| f.0.inliers).max()?;
    // Lowest (along up) among the large ones, measured under the camera centroid.
    let c = frame.centers().iter().sum::<DVec3>() / frame.cameras.len() as f64;
    let height_at = |p: &Plane| {
        // point on plane below c along up
        let t = (p.h - p.n.dot(c)) / p.n.dot(up);
        (c + up * t).dot(up)
    };
    let (plane, mask) = found
        .into_iter()
        .filter(|f| f.0.inliers as f64 >= 0.3 * best as f64)
        .min_by(|a, b| height_at(&a.0).total_cmp(&height_at(&b.0)))?;
    let inliers: Vec<DVec3> = pts.iter().zip(&mask).filter(|(_, m)| **m).map(|(p, _)| *p).collect();
    Some(Floor { plane, inliers })
}

/// Large vertical planes between knee height and above the cameras; strongest first.
fn find_walls(pts: &[DVec3], up: DVec3, floor_h: f64, scale: f64, frame: &SceneFrame) -> Vec<Plane> {
    let (lo, hi) = (floor_h + 0.3 / scale, floor_h + 2.6 / scale);
    let mut active: Vec<bool> = pts.iter().map(|p| (lo..hi).contains(&p.dot(up))).collect();
    let n_active = active.iter().filter(|a| **a).count();
    let thresh = 0.03 / scale;
    let mut rng = Rng::new(0xA11);
    let mut walls = Vec::new();
    for _ in 0..6 {
        let opts = RansacOpts { iters: 1500, thresh, max_tilt_deg: 8.0, vertical: true };
        let Some((plane, mask)) = ransac(pts, &active, up, &opts, &mut rng) else { break };
        if (plane.inliers as f64) < (0.04 * n_active as f64).max(500.0) {
            break;
        }
        for (a, m) in active.iter_mut().zip(&mask) {
            *a &= !*m;
        }
        // Walls bound the room: most cameras on one side.
        let side = frame.centers().iter().filter(|c| plane.dist(**c) > 0.0).count();
        let frac = side as f64 / frame.cameras.len() as f64;
        // Skip near-duplicates (furniture flush against a wall already found).
        let dup = walls.iter().any(|w: &Plane| {
            let d = w.n.dot(plane.n);
            d.abs() > 0.97 && (plane.h - d.signum() * w.h).abs() < 0.4 / scale
        });
        if !(0.1..=0.9).contains(&frac) && !dup {
            walls.push(plane);
        }
    }
    walls
}

fn principal_horizontal(pts: &[DVec3], up: DVec3) -> Option<DVec3> {
    if pts.len() < 100 {
        return None;
    }
    let (e1, e2) = splat_shots::scene::horizontal_basis(up);
    let xy: Vec<(f64, f64)> = pts.iter().map(|p| (p.dot(e1), p.dot(e2))).collect();
    let n = xy.len() as f64;
    let (mx, my) = (xy.iter().map(|v| v.0).sum::<f64>() / n, xy.iter().map(|v| v.1).sum::<f64>() / n);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (x, y) in &xy {
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
        sxy += (x - mx) * (y - my);
    }
    let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    Some(e1 * th.cos() + e2 * th.sin())
}

fn horiz_dist(a: DVec3, b: DVec3) -> f64 {
    ((a.x - b.x).powi(2) + (a.z - b.z).powi(2)).sqrt()
}

fn r1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
fn r2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
fn r4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}
fn r3v(v: f64) -> f64 {
    (v * 1e3).round() / 1e3
}
fn r3(v: DVec3) -> [f64; 3] {
    [r3v(v.x), r3v(v.y), r3v(v.z)]
}

#[cfg(test)]
mod zstd_input_tests {
    use super::*;

    /// One-splat binary PLY (render mode tag mip) and its zstd copy.
    fn tiny_ply() -> Vec<u8> {
        let mut raw = b"ply\nformat binary_little_endian 1.0\ncomment SplatRenderMode: mip\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nproperty float opacity\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nend_header\n".to_vec();
        for x in [1.0_f32, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] {
            raw.extend(x.to_le_bytes());
        }
        raw
    }

    #[test]
    fn compressed_ply_reads_and_resolves_like_a_plain_one() {
        let dir = std::env::temp_dir().join(format!("splat-webexport-zst-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let raw = tiny_ply();
        let packed = zstd::stream::encode_all(&raw[..], 3).unwrap();

        // Readers: same splats and tag; a truncated stream is an error naming the file.
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("model.ply.zst");
        std::fs::write(&p, &packed).unwrap();
        let g = ply::read_ply(&p).unwrap();
        assert_eq!(g.pos, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(ply::read_render_mode(&p).unwrap(), g.render_mode);
        let cut = dir.join("cut.ply.zst");
        std::fs::write(&cut, &packed[..packed.len() / 2]).unwrap();
        let err = format!("{:#}", ply::read_ply(&cut).unwrap_err());
        assert!(err.contains("cut.ply.zst"), "{err}");

        // Ordinary export resolution: an explicit .ply.zst inside a run dir wins over the run's latest candidate.
        let run = dir.join("run");
        let cand = run.join("artifacts/candidate_a");
        std::fs::create_dir_all(&cand).unwrap();
        std::fs::create_dir_all(run.join("artifacts/sfm_a/sparse")).unwrap();
        std::fs::write(run.join("run.json"), "{}").unwrap();
        std::fs::write(cand.join("splat.ply"), &raw).unwrap();
        let explicit = run.join("refined.ply.zst");
        std::fs::write(&explicit, &packed).unwrap();
        assert_eq!(resolve(&explicit, None).unwrap().ply, explicit);
        assert_eq!(resolve(&run, None).unwrap().ply, cand.join("splat.ply"));

        // A newer candidate saved only as splat.ply.zst wins over an older plain one; a compressed-only run resolves.
        let newer = run.join("artifacts/candidate_b");
        std::fs::create_dir_all(&newer).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(newer.join("splat.ply.zst"), &packed).unwrap();
        assert_eq!(resolve(&run, None).unwrap().ply, newer.join("splat.ply.zst"));
        std::fs::remove_file(cand.join("splat.ply")).unwrap();
        assert_eq!(resolve(&run, None).unwrap().ply, newer.join("splat.ply.zst"));

        // Outside a run: a dir holding splat.ply.zst, with an explicit capture dir.
        let bare = dir.join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        std::fs::write(bare.join("splat.ply.zst"), &packed).unwrap();
        assert_eq!(resolve(&bare, Some(&run)).unwrap().ply, bare.join("splat.ply.zst"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
