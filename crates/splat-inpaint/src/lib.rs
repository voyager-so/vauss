//! Object removal + hole fill on a trained splat ("remove the X and fill the hole"), in-engine.
//! Clean-room from the InFusion / 3D-GIMP papers, ported from a Python (gsplat) reference
//! implementation. Steps:
//!
//! 1. Hole mask per view: an indicator render (removed splats white, others black, same
//!    geometry/opacity) through the ORIGINAL scene; hole = indicator > `hole_thr`, dilated.
//! 2. Unedited renders of every train view (the fine-tune targets) and the removed scene
//!    (RGB, depth, alpha) at the reference view.
//! 3. Reference view = the train view with the largest, most central hole. LaMa (default) or
//!    SD-1.5 inpainting (`inpainter = "sd15"`, diffusion on CUDA) fills the UNEDITED render's
//!    hole (crop around the hole, 512x512).
//! 4. Fill depth: Depth-Anything-V2-Small disparity fit to the removed scene's disparity on a
//!    ring around the hole (trimmed LSQ), or a trimmed plane fit to the ring's 3D points when
//!    that fits the ring better; clamped behind the removed object's surface.
//! 5. New splats from the hole pixels (DC colour, isotropic pixel-footprint scale); old splats
//!    in front of the fill inside the reference hole are pruned.
//! 6. Fine-tune ONLY the new splats (old ones frozen, no refine): the reference view supervises
//!    the whole filled image, the other train views supervise outside their holes against the
//!    unedited render.
//! 7. Output PLY = old (minus removed, minus pruned, records verbatim) + new; result.json;
//!    before/after PNGs of the reference view; held-out PSNR outside the hole vs the unedited
//!    render (how much of the rest of the scene moved).

pub mod diffusion;
pub mod geom;
pub mod gpu;
pub mod img;
pub mod models;
pub mod npy;
pub mod ply;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, ensure};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::gpu::{FtView, Host, Pin, SH_C0, render};
use crate::img::{Mask, RgbF, psnr_masked};

fn d_hole_thr() -> f32 {
    0.3
}
fn d_dilate() -> f32 {
    0.012
}
/// Footprint margin (fraction of max(H, W)): fill, prune and recolour stay inside the object's own
/// projected footprint dilated by this much; the wider `dilate` hole is only LaMa/depth context.
/// An old splat that this many other train views see, unoccluded, outside the object's footprint is a
/// real neighbour: never pruned or painted.
const OBSERVED_VIEWS: u32 = 2;
pub fn d_footprint_dilate() -> f32 {
    0.003
}
fn d_steps() -> u32 {
    1500
}
fn d_ref_every() -> u32 {
    3
}
fn d_max_new() -> usize {
    400_000
}
fn d_seed() -> u64 {
    42
}
pub fn d_inpainter() -> String {
    "lama".into()
}
pub fn d_diffusion_steps() -> u32 {
    30
}
pub fn d_guidance() -> f32 {
    7.5
}
pub fn d_recolour() -> bool {
    false
}
/// Pixels whose removed-scene render already shows an opaque surface near the fill depth get no
/// new splat (the scene saw that background from other views; recolouring handles its tint).
pub fn d_seen_alpha() -> f32 {
    2.0
}
/// A fill point seen (outside the object mask, opaque, matching depth) in at least this many other
/// training views is already reconstructed: no new splat there. 0 disables.
pub fn d_seen_views() -> u32 {
    3
}
pub fn d_recolour_pad() -> f32 {
    0.3
}
/// Other train views also supervise their hole pixels where the removed scene already shows an
/// opaque background (alpha above this), against the removed render: fill splats floating off the
/// surface (wrong fill depth) otherwise smear across the floor in side views. > 1 disables.
/// Only with monocular fill depth: a plane fill lies on the observed ring surface, and under the
/// object the removed scene is often a translucent blur (a bad target; a room clip's rug).
pub fn d_side_alpha() -> f32 {
    0.9
}
/// A fill point that sits >5% in front of an opaque removed-scene surface in at least this many
/// other training views floats off the real surface (wrong fill depth): no new splat. A correct
/// fill point never occludes observed background. Pixels inside that view's object hole count
/// only when `front_in_hole` (default: monocular fill depth yes, plane fill no): under the object
/// the removed scene is often a translucent blur whose expected depth mixes in the far background,
/// which wrongly drops correct plane fill. 0 disables.
pub fn d_front_views() -> u32 {
    1
}
/// Fine-tuned fill splats whose largest axis grew past this multiple of their initial (pixel
/// footprint) size, or whose two largest axes differ more than 8x, are dropped: they are the streaks a single
/// reference view stretches along its rays (seen as smears from the side). 0 disables.
pub fn d_max_stretch() -> f32 {
    3.0
}
/// Old splats in front of the fill are pruned as object leftovers unless their colour is within this
/// mean absolute RGB difference of the 2D fill at their reference pixel (the surface itself). 0
/// prunes them all.
pub fn d_prune_colour() -> f32 {
    0.0
}
/// A connected region of the reference hole where the removed scene is opaque in front of the fill,
/// covering at least this fraction of the hole, is a support surface: kept, not filled. 0 disables.
pub fn d_surface_frac() -> f32 {
    0.02
}

/// Unknown keys are an error (a knob sent to a binary that lacks it must fail the job).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InpaintJob {
    /// The UNEDITED splat (plain SH PLY, Brush layout).
    pub parent_ply: PathBuf,
    /// Vertex indices into `parent_ply` of the object to remove (`.npy`, integer dtype).
    pub selected_ids: PathBuf,
    /// SfM dataset dir: images/ + sparse/0.
    pub dataset: PathBuf,
    pub train_names: Vec<String>,
    #[serde(default)]
    pub heldout_names: Vec<String>,
    /// Training resolution cap (long edge) the parent was trained at.
    pub max_resolution: u32,
    /// Output dir: splat.ply, result.json, images/.
    pub out: PathBuf,
    #[serde(default = "d_steps")]
    pub steps: u32,
    #[serde(default = "d_hole_thr")]
    pub hole_thr: f32,
    /// Hole dilation radius as a fraction of max(H, W).
    #[serde(default = "d_dilate")]
    pub dilate: f32,
    /// Fill/prune/recolour margin around the object's footprint, fraction of max(H, W).
    #[serde(default = "d_footprint_dilate")]
    pub footprint_dilate: f32,
    /// One reference-view batch every this many steps (on average).
    #[serde(default = "d_ref_every")]
    pub ref_every: u32,
    #[serde(default = "d_max_new")]
    pub max_new: usize,
    /// Also seeds the diffusion inpainter's initial latent noise.
    #[serde(default = "d_seed")]
    pub seed: u64,
    /// 2D inpainter for the reference view: "lama" (LaMa, Apache-2.0, CPU; default) or "sd15" (Stable
    /// Diffusion 1.5 inpainting, CUDA when available). "sd15" downloads weights under the CreativeML
    /// OpenRAIL-M licence: commercial use is allowed subject to its use-based restrictions, which you must
    /// pass on to your users.
    #[serde(default = "d_inpainter")]
    pub inpainter: String,
    /// sd15 only: prompt (None = diffusion::DEFAULT_PROMPT).
    #[serde(default)]
    pub prompt: Option<String>,
    /// sd15 only: negative prompt (None = diffusion::DEFAULT_NEGATIVE).
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// sd15 only: DDIM steps.
    #[serde(default = "d_diffusion_steps")]
    pub diffusion_steps: u32,
    /// sd15 only: classifier-free guidance scale.
    #[serde(default = "d_guidance")]
    pub guidance: f32,
    /// Let the fine-tune recolour old splats (SH + opacity; geometry frozen) that project
    /// into the reference hole near the removed object: they baked its shadow/occlusion.
    #[serde(default = "d_recolour")]
    pub recolour: bool,
    /// Recolour candidates must lie in the removed set's robust 3D box padded by this
    /// fraction of its extent per axis.
    #[serde(default = "d_recolour_pad")]
    pub recolour_pad: f32,
    /// Skip new splats where the removed scene's alpha exceeds this and its depth is within 10%
    /// of the fill depth (already-seen background). > 1 disables.
    #[serde(default = "d_seen_alpha")]
    pub seen_alpha: f32,
    /// Drop new fill splats whose point the removed scene already shows in >= this many other
    /// training views (outside their object masks, alpha > 0.7, depth within 5%). 0 disables.
    #[serde(default = "d_seen_views")]
    pub seen_views: u32,
    #[serde(default = "d_side_alpha")]
    pub side_alpha: f32,
    #[serde(default = "d_front_views")]
    pub front_views: u32,
    #[serde(default)]
    pub front_in_hole: Option<bool>,
    #[serde(default = "d_max_stretch")]
    pub max_stretch: f32,
    #[serde(default = "d_prune_colour")]
    pub prune_colour: f32,
    #[serde(default = "d_surface_frac")]
    pub surface_frac: f32,
    /// Dir holding lama_fp32.onnx / depth_anything_v2_small.onnx; else $SPLAT_MODELS_DIR, else
    /// the pinned download (splat cache).
    #[serde(default)]
    pub models_dir: Option<PathBuf>,
    /// Keep the fine-tune dataset (renders + masks) under out/finetune for inspection.
    #[serde(default)]
    pub keep_finetune_dir: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InpaintResult {
    pub ply: PathBuf,
    pub n_input: usize,
    pub n_removed: usize,
    pub n_pruned: usize,
    /// Old splats whose SH + opacity the fine-tune retrained (geometry frozen).
    #[serde(default)]
    pub n_recolour: usize,
    pub n_seen_skipped: usize,
    pub n_seen_multiview: usize,
    /// Hole pixels of the other train views supervised against the removed render (side_alpha).
    #[serde(default)]
    pub side_px: usize,
    /// Fill points dropped as floating in front of observed background (front_views).
    #[serde(default)]
    pub n_front_dropped: usize,
    /// Fine-tuned fill splats dropped as stretched (max_stretch).
    #[serde(default)]
    pub n_stretch_dropped: usize,
    /// Prune candidates kept because their colour matches the fill (prune_colour).
    #[serde(default)]
    pub n_prune_kept_observed: usize,
    /// Reference hole pixels kept as support surface (surface_frac), and fill points skipped there.
    #[serde(default)]
    pub surface_px: usize,
    #[serde(default)]
    pub n_surface_skipped: usize,
    /// Old splats on the support surface recoloured to its observed rim (hoof marks).
    #[serde(default)]
    pub n_surface_painted: usize,
    /// Prune candidates kept because other views observe them outside the hole.
    #[serde(default)]
    pub n_prune_kept_seen: usize,
    pub n_new: usize,
    pub n_output: usize,
    pub stride: usize,
    pub ref_view: String,
    pub ref_hole_px: usize,
    pub views_with_hole: usize,
    pub dilate_px: usize,
    /// "plane" or "depth-anything-v2-small".
    pub depth_mode: String,
    pub ring_px: usize,
    pub ring_rel_err_da: f32,
    pub ring_rel_err_plane: f32,
    pub steps: u32,
    pub losses: Vec<(u32, f32)>,
    pub heldout_views: usize,
    /// Held-out PSNR outside the hole, edited vs unedited render (rest-of-scene stability).
    pub psnr_out_edit_vs_orig_mean: Option<f64>,
    pub psnr_out_edit_vs_orig_min: Option<f64>,
    pub psnr_out_removed_vs_orig_mean: Option<f64>,
    pub heldout: Vec<BTreeMap<String, serde_json::Value>>,
    /// Paths relative to `out`.
    pub images: Vec<String>,
    pub timings_s: BTreeMap<String, f64>,
    /// 2D inpainter used: "lama" or "sd15".
    #[serde(default)]
    pub inpainter: String,
    /// Its ONNX Runtime execution provider: "cpu" (LaMa), "cuda" or "cpu (<why not cuda>)".
    #[serde(default)]
    pub provider: String,
    pub models: BTreeMap<String, String>,
    pub warnings: Vec<String>,
}

fn progress(stage: &str, extra: serde_json::Value) {
    let mut row = serde_json::json!({"event": "stage", "stage": stage});
    if let (Some(r), serde_json::Value::Object(e)) = (row.as_object_mut(), extra) {
        r.extend(e);
    }
    println!("PROGRESS {row}");
}

static STAGE: std::sync::Mutex<&str> = std::sync::Mutex::new("start");

/// Record (and log) the current stage; a panic names it (see [`run`]).
pub fn stage(name: &'static str) {
    *STAGE.lock().unwrap_or_else(|e| e.into_inner()) = name;
    eprintln!("[inpaint] stage {name}");
}

/// Run the whole edit; blocks. Writes `out/splat.ply`, `out/result.json`, `out/images/*`.
pub fn run(job: InpaintJob) -> anyhow::Result<InpaintResult> {
    // A panic (burn asserts, kernel errors) prints the stage it happened in, with or
    // without RUST_BACKTRACE.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let at = *STAGE.lock().unwrap_or_else(|e| e.into_inner());
        eprintln!("[inpaint] PANIC in stage '{at}'");
        println!("PROGRESS {}", serde_json::json!({"event": "panic", "stage": at, "message": info.to_string()}));
        prev(info);
    }));
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async {
        let r = run_async(&job).await;
        brush_render::render::settle_lagged_counts().await;
        r
    })
}

async fn run_async(job: &InpaintJob) -> anyhow::Result<InpaintResult> {
    let t0 = Instant::now();
    let mut res = InpaintResult::default();
    let mut lap = {
        let mut t = Instant::now();
        move |res: &mut InpaintResult, k: &str| {
            res.timings_s.insert(k.into(), t.elapsed().as_secs_f64());
            t = Instant::now();
        }
    };
    std::fs::create_dir_all(job.out.join("images"))?;
    ensure!((1..=100_000).contains(&job.steps), "steps must be 1..=100000");
    ensure!(job.ref_every >= 1, "ref_every must be >= 1");
    ensure!(job.recolour_pad.is_finite() && job.recolour_pad >= 0.0, "recolour_pad must be finite and >= 0");
    ensure!(matches!(job.inpainter.as_str(), "lama" | "sd15"), "inpainter must be \"lama\" or \"sd15\", got {:?}", job.inpainter);

    // ---- load
    stage("load");
    let ply = ply::Ply::read(&job.parent_ply)?;
    for p in ["x", "y", "z", "opacity", "scale_0", "rot_0", "f_dc_0"] {
        ensure!(ply.has(p), "PLY has no {p} property (plain SH Brush layout expected)");
    }
    let (host, _up) = Host::load(&job.parent_ply).await?;
    let n = host.n();
    ensure!(n == ply.n, "PLY parsers disagree on the vertex count ({n} vs {})", ply.n);
    ensure!(ply.rest_count() == 3 * (host.k - 1), "f_rest count {} does not match SH k={}", ply.rest_count(), host.k);
    let ids = npy::read_indices(&job.selected_ids)?;
    let mut removed = vec![false; n];
    for &i in &ids {
        ensure!((i as usize) < n, "selected id {i} out of range for {n} splats");
        removed[i as usize] = true;
    }
    res.n_input = n;
    res.n_removed = removed.iter().filter(|&&r| r).count();
    ensure!(res.n_removed > 0, "selection is empty");
    let device = gpu::init_device().await;
    let views = gpu::load_views(&job.dataset, job.max_resolution).await?;
    let find = |name: &str| views.iter().position(|v| gpu::view_name(v) == name);
    let train: Vec<usize> = job.train_names.iter().filter_map(|n| find(n)).collect();
    let held: Vec<usize> = job.heldout_names.iter().filter_map(|n| find(n)).collect();
    ensure!(!train.is_empty(), "none of the train views are in the dataset");
    if train.len() < job.train_names.len() {
        res.warnings.push(format!("{} train views missing from the dataset", job.train_names.len() - train.len()));
    }
    let mut dims = vec![(0usize, 0usize); views.len()];
    for &i in train.iter().chain(&held) {
        // Native size capped to the training resolution (what the trainer's loader produces).
        let (w, h) = views[i].image.dimensions().await?;
        let long = w.max(h).max(1) as f32;
        let scale = if job.max_resolution > 0 && long > job.max_resolution as f32 { job.max_resolution as f32 / long } else { 1.0 };
        dims[i] = (((w as f32 * scale) as usize).max(1), ((h as f32 * scale) as usize).max(1));
    }
    progress("load", serde_json::json!({"n": n, "sh_k": host.k, "n_removed": res.n_removed, "n_train": train.len(), "n_heldout": held.len()}));
    lap(&mut res, "load_s");

    // ---- 1+2. hole masks; unedited renders = fine-tune targets
    stage("holes");
    let ft_dir = job.out.join("finetune");
    if ft_dir.exists() {
        std::fs::remove_dir_all(&ft_dir)?;
    }
    std::fs::create_dir_all(ft_dir.join("images"))?;
    std::fs::create_dir_all(ft_dir.join("masks"))?;
    let full = host.to_splats(&device);
    let keep_old: Vec<bool> = removed.iter().map(|r| !r).collect();
    let rem_splats = host.subset(&keep_old).to_splats(&device);
    let indicator = host.with_colour(|i| if removed[i] { 1.0 } else { 0.0 }, &device);
    let (w0, h0) = dims[train[0]];
    let rad = ((job.dilate * w0.max(h0) as f32).round() as usize).max(1);
    res.dilate_px = rad;
    let mut holes: BTreeMap<usize, Mask> = BTreeMap::new();
    let mut footprints: BTreeMap<usize, Mask> = BTreeMap::new();
    let foot_rad = ((job.footprint_dilate * w0.max(h0) as f32).round() as usize).max(1);
    for &i in train.iter().chain(&held) {
        let (w, h) = dims[i];
        let ind = render(&indicator, &views[i].camera, w, h, false).await?;
        let mut raw = Mask::new(w, h);
        for (p, m) in raw.data.iter_mut().enumerate() {
            *m = ind.rgb.data[3 * p] > job.hole_thr;
        }
        let hole = if raw.any() { raw.dilate(rad) } else { raw.clone() };
        footprints.insert(i, if raw.any() { raw.dilate(foot_rad) } else { raw });
        if train.contains(&i) {
            let orig = render(&full, &views[i].camera, w, h, false).await?;
            orig.rgb.save(&ft_dir.join("images").join(format!("{i:05}.png")))?;
            hole.to_gray(0, 255).save(ft_dir.join("masks").join(format!("{i:05}.png")))?;
        }
        holes.insert(i, hole);
    }
    drop(indicator);
    res.views_with_hole = holes.values().filter(|m| m.any()).count();
    progress("holes", serde_json::json!({"views_with_hole": res.views_with_hole, "dilate_px": rad}));
    lap(&mut res, "holes_s");

    // ---- 3. reference view + 2D inpaint (LaMa or SD-1.5)
    stage(if job.inpainter == "sd15" { "diffusion" } else { "lama" });
    let score = |i: usize| -> f64 {
        let m = &holes[&i];
        let Some((y0, y1, x0, x1)) = m.bbox() else { return -1.0 };
        let (h, w) = (m.h as f64, m.w as f64);
        let dc = (((y0 + y1) as f64 / 2.0 / h - 0.5).hypot((x0 + x1) as f64 / 2.0 / w - 0.5)) / 0.5f64.hypot(0.5);
        let border = y0 <= 1 || x0 <= 1 || y1 >= m.h - 1 || x1 >= m.w - 1;
        m.count() as f64 * (1.0 - 0.6 * dc) * if border { 0.4 } else { 1.0 }
    };
    let ri = *train.iter().max_by(|&&a, &&b| score(a).total_cmp(&score(b))).expect("train views");
    ensure!(score(ri) > 0.0, "the selected object is not visible in any train view");
    res.ref_view = gpu::view_name(&views[ri]);
    let (w, h) = dims[ri];
    let rcam = views[ri].camera;
    let pin = Pin::new(&rcam, w, h);
    let hole = holes[&ri].clone();
    // Fill, prune and recolour only inside the object's own footprint: observed neighbours in the
    // dilated ring (a rug fringe, a stove edge) were streaked by fill splats and recolouring.
    let foot = footprints[&ri].and(&hole);
    let base = render(&full, &rcam, w, h, true).await?;
    res.inpainter = job.inpainter.clone();
    let fill = if job.inpainter == "sd15" {
        let opts = diffusion::Sd15Options {
            prompt: job.prompt.clone().unwrap_or_else(|| diffusion::DEFAULT_PROMPT.into()),
            negative_prompt: job.negative_prompt.clone().unwrap_or_else(|| diffusion::DEFAULT_NEGATIVE.into()),
            steps: job.diffusion_steps,
            guidance: job.guidance,
            seed: job.seed,
        };
        let mut sd = diffusion::Sd15::load(job.models_dir.as_deref(), opts)?;
        let fill = models::inpaint_crop(&mut sd, &base.rgb, &hole)?;
        res.provider = sd.provider.clone();
        res.timings_s.extend(std::mem::take(&mut sd.timings));
        res.models.extend(std::mem::take(&mut sd.paths));
        fill
    } else {
        let lama_path = models::locate(job.models_dir.as_deref(), models::LAMA)?;
        let mut lama = models::Lama::load(&lama_path)?;
        let fill = models::inpaint_crop(&mut lama, &base.rgb, &hole)?;
        res.provider = "cpu".into();
        res.models.insert("lama".into(), lama_path.display().to_string());
        fill
    };
    progress("inpaint2d", serde_json::json!({"inpainter": res.inpainter, "provider": res.provider}));
    lap(&mut res, if job.inpainter == "sd15" { "diffusion_s" } else { "lama_s" });

    // ---- 4. fill depth
    stage("depth");
    let rem = render(&rem_splats, &rcam, w, h, true).await?;
    let d_rem = rem.depth.clone().expect("depth");
    let d_obj = base.depth.clone().expect("depth");
    let ring_mask = {
        let mut r = hole.dilate(2 * rad).and(&hole.not());
        for (p, m) in r.data.iter_mut().enumerate() {
            *m &= rem.alpha[p] > 0.9 && d_rem[p].is_finite() && d_rem[p] > 1e-4;
        }
        r
    };
    let ring: Vec<usize> = (0..w * h).filter(|&p| ring_mask.data[p]).collect();
    let hole_px: Vec<usize> = (0..w * h).filter(|&p| hole.data[p]).collect();
    ensure!(ring.len() >= 16, "only {} ring pixels around the hole have opaque background depth", ring.len());
    res.ring_px = ring.len();
    res.ref_hole_px = hole_px.len();
    let depth_path = models::locate(job.models_dir.as_deref(), models::DEPTH)?;
    let da = models::DepthAnything::load(&depth_path)?.run(&fill)?;
    res.models.insert("depth".into(), depth_path.display().to_string());
    let (s, t) = geom::robust_affine(
        &ring.iter().map(|&p| da[p]).collect::<Vec<_>>(),
        &ring.iter().map(|&p| 1.0 / d_rem[p]).collect::<Vec<_>>(),
    );
    let z_da = |p: usize| 1.0 / ((s * da[p] as f64 + t).max(1e-4)) as f32;
    let rel = |z: f32, p: usize| (z - d_rem[p]).abs() / d_rem[p];
    res.ring_rel_err_da = geom::median(ring.iter().map(|&p| rel(z_da(p), p)).collect());
    let ray = |p: usize| pin.ray(p % w, p / w);
    let ring_pts: Vec<Vec3> = ring.iter().map(|&p| pin.to_world(ray(p) * d_rem[p])).collect();
    let (c, nrm) = geom::fit_plane(&ring_pts);
    // z-depth where the pixel ray (z = 1 in camera space) meets the plane.
    let plane_z = |p: usize| {
        let dir = pin.rot * ray(p);
        (c - pin.pos).dot(nrm) / dir.dot(nrm)
    };
    res.ring_rel_err_plane = geom::median(ring.iter().map(|&p| {
        let z = plane_z(p);
        if z.is_finite() { rel(z, p) } else { 1e3 }
    }).collect());
    let plane_ok = hole_px.iter().all(|&p| plane_z(p) > 0.0) && res.ring_rel_err_plane < res.ring_rel_err_da;
    let zr: Vec<f32> = ring.iter().map(|&p| d_rem[p]).collect();
    let (zmin, zmax) = (zr.iter().copied().fold(f32::INFINITY, f32::min), zr.iter().copied().fold(0.0f32, f32::max));
    let z_fill: Vec<f32> = hole_px
        .iter()
        .map(|&p| {
            let z = if plane_ok { plane_z(p) } else { z_da(p) };
            let z = z.clamp(0.5 * zmin, 2.0 * zmax);
            let obj = if d_obj[p].is_finite() { d_obj[p] * 1.02 } else { 0.0 };
            z.max(obj)
        })
        .collect();
    res.depth_mode = if plane_ok { "plane" } else { "depth-anything-v2-small" }.into();
    progress("depth", serde_json::json!({"mode": res.depth_mode, "ring_rel_err_da": res.ring_rel_err_da, "ring_rel_err_plane": res.ring_rel_err_plane, "hole_px": hole_px.len(), "ref": res.ref_view}));
    lap(&mut res, "depth_s");

    // ---- 5. new splats + prune
    stage("init");
    let stride = ((hole_px.len() as f64 / job.max_new as f64).sqrt().ceil() as usize).max(1);
    res.stride = stride;
    let mut new = Host { means: vec![], rots: vec![], scales: vec![], sh: vec![], opac: vec![], k: host.k, mode: host.mode };
    let logit = (0.85f32 / 0.15).ln();
    // Multi-view "already seen" count per candidate fill point: the removed scene shows an opaque
    // surface at that point (depth within 5%) in another training view, outside that view's object
    // mask. Such floor was reconstructed from real observations; a fill layer from the one
    // reference view only adds streaks there at grazing angles (GHOST tool path, house).
    let cand: Vec<(usize, Vec3)> = hole_px
        .iter()
        .enumerate()
        .filter(|&(_, &p)| (p % w) % stride == 0 && (p / w) % stride == 0 && foot.data[p])
        .map(|(j, &p)| (j, pin.to_world(pin.ray(p % w, p / w) * z_fill[j])))
        .collect();
    // Support surface: hole pixels where the removed scene already renders an opaque surface in front
    // of the fill. A large connected region of them is real surface the object stood on (a statue's
    // pedestal top under a fill placed on the grass behind it): no fill and no pruning there. Small
    // regions are object leftovers (chair fragments) and are pruned as before.
    let surface = {
        let mut m = Mask::new(w, h);
        for (j, &p) in hole_px.iter().enumerate() {
            m.data[p] = rem.alpha[p] > 0.9 && d_rem[p].is_finite() && d_rem[p] < 0.97 * z_fill[j];
        }
        let min_area = (job.surface_frac * hole_px.len() as f32).ceil() as usize;
        let mut out = Mask::new(w, h);
        let mut seen_px = vec![false; w * h];
        for start in 0..w * h {
            if !m.data[start] || seen_px[start] || job.surface_frac <= 0.0 {
                continue;
            }
            let (mut comp, mut stack) = (vec![], vec![start]);
            seen_px[start] = true;
            while let Some(q) = stack.pop() {
                comp.push(q);
                let (x, y) = (q % w, q / w);
                let nb = [(x > 0).then(|| q - 1), (x + 1 < w).then(|| q + 1), (y > 0).then(|| q - w), (y + 1 < h).then(|| q + w)];
                for r in nb.into_iter().flatten() {
                    if m.data[r] && !seen_px[r] {
                        seen_px[r] = true;
                        stack.push(r);
                    }
                }
            }
            if comp.len() >= min_area {
                for q in comp {
                    out.data[q] = true;
                }
            }
        }
        out
    };
    res.surface_px = surface.count();
    let mut seen = vec![0u32; cand.len()];
    let mut front = vec![0u32; cand.len()];
    let side = job.side_alpha <= 1.0 && !plane_ok;
    let front_in_hole = job.front_in_hole.unwrap_or(!plane_ok);
    if job.seen_views > 0 || job.front_views > 0 || side {
        for &i in train.iter().filter(|&&i| i != ri) {
            let (wv, hv) = dims[i];
            let pv = Pin::new(&views[i].camera, wv, hv);
            let r = render(&rem_splats, &views[i].camera, wv, hv, true).await?;
            let dv = r.depth.as_ref().expect("depth");
            let hm = &holes[&i];
            if side && hm.any() {
                // Side supervision: this view's hole pixels the removed scene renders opaque.
                let (ip, mp) = (ft_dir.join("images").join(format!("{i:05}.png")), ft_dir.join("masks").join(format!("{i:05}.png")));
                let (mut im, mut mk) = (image::open(&ip)?.to_rgb8(), image::open(&mp)?.to_luma8());
                for p in (0..wv * hv).filter(|&p| hm.data[p] && r.alpha[p] > job.side_alpha) {
                    let (x, y) = ((p % wv) as u32, (p / wv) as u32);
                    let c = &r.rgb.data[3 * p..3 * p + 3];
                    im.put_pixel(x, y, image::Rgb(std::array::from_fn(|k| (c[k].clamp(0.0, 1.0) * 255.0).round() as u8)));
                    mk.put_pixel(x, y, image::Luma([255]));
                    res.side_px += 1;
                }
                im.save(&ip)?;
                mk.save(&mp)?;
            }
            for (k, (_, pw)) in cand.iter().enumerate() {
                let pc = pv.to_cam(*pw);
                if pc.z <= 0.01 {
                    continue;
                }
                let (u, v) = ((pc.x / pc.z * pv.fx + pv.cx).floor(), (pc.y / pc.z * pv.fy + pv.cy).floor());
                if u < 0.0 || v < 0.0 || u >= wv as f32 || v >= hv as f32 {
                    continue;
                }
                let q = v as usize * wv + u as usize;
                if (front_in_hole || !hm.data[q]) && r.alpha[q] > 0.7 && dv[q].is_finite() && pc.z < 0.95 * dv[q] {
                    front[k] += 1;
                }
                if !hm.data[q] && r.alpha[q] > 0.7 && dv[q].is_finite() && (dv[q] - pc.z).abs() < 0.05 * pc.z {
                    seen[k] += 1;
                }
            }
        }
    }
    for (k, &(j, pw)) in cand.iter().enumerate() {
        let p = hole_px[j];
        let (x, y) = (p % w, p / w);
        let z = z_fill[j];
        if surface.data[p] {
            res.n_surface_skipped += 1;
            continue;
        }
        if job.seen_views > 0 && seen[k] >= job.seen_views {
            res.n_seen_multiview += 1;
            continue;
        }
        if job.front_views > 0 && front[k] >= job.front_views {
            res.n_front_dropped += 1;
            continue;
        }
        if rem.alpha[p] > job.seen_alpha && d_rem[p].is_finite() && (d_rem[p] - z).abs() < 0.1 * z {
            res.n_seen_skipped += 1;
            continue;
        }
        let col = fill.px(x, y);
        let ls = (z / pin.fx * stride as f32 * 0.8).ln();
        new.means.extend([pw.x, pw.y, pw.z]);
        new.rots.extend([1.0, 0.0, 0.0, 0.0]);
        new.scales.extend([ls; 3]);
        new.opac.push(logit);
        new.sh.extend(col.map(|v| (v - 0.5) / SH_C0));
        new.sh.extend(std::iter::repeat_n(0.0, 3 * (host.k - 1)));
    }
    res.n_new = new.n();
    ensure!(res.n_new > 0 || job.recolour, "no new splats and recolour off: nothing to fine-tune");
    let mut zmap = vec![f32::INFINITY; w * h];
    for (j, &p) in hole_px.iter().enumerate() {
        // 0 on support surface: nothing there is "in front of the fill", so nothing is pruned.
        zmap[p] = if surface.data[p] { 0.0 } else { z_fill[j] };
    }
    let erode = hole.erode((rad / 2).max(1));
    // Only object leftovers are pruned: candidates must sit inside the removed set's robust 3D box
    // (2nd..98th percentile per axis, padded 15%). Without this a large object close to the camera
    // pruned scene surface (kitchen lego: 190k splats) and changed views outside the hole.
    let robust = {
        let mut ax: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        for i in (0..n).filter(|&i| !keep_old[i]) {
            let m = host.mean(i);
            ax[0].push(m.x);
            ax[1].push(m.y);
            ax[2].push(m.z);
        }
        ax.map(|mut v| {
            v.sort_by(f32::total_cmp);
            (v[(v.len() as f32 * 0.02) as usize], v[((v.len() as f32 * 0.98) as usize).min(v.len() - 1)])
        })
    };
    let in_box = |m: Vec3, pad: f32| {
        [m.x, m.y, m.z].iter().zip(&robust).all(|(&c, &(lo, hi))| {
            let p = (hi - lo) * pad;
            c >= lo - p && c <= hi + p
        })
    };
    // Reference-view pixel of splat i's centre (in front of the camera, inside the image).
    let ref_px = |i: usize| -> Option<(usize, f32)> {
        let pc = pin.to_cam(host.mean(i));
        if pc.z <= 0.01 {
            return None;
        }
        let (u, v) = ((pc.x / pc.z * pin.fx + pin.cx).floor(), (pc.y / pc.z * pin.fy + pin.cy).floor());
        (u >= 0.0 && v >= 0.0 && u < w as f32 && v < h as f32).then(|| (v as usize * w + u as usize, pc.z))
    };
    let mut keep = keep_old.clone();
    // Leftover candidates: old splats inside the removed set's box, in front of the fill inside the
    // reference footprint. Only those whose colour disagrees with the 2D fill at their reference
    // pixel are object leftovers (dark chair fragments on a wood floor, grass-coloured marks where a
    // statue's hooves stood); a candidate matching the fill is the surface itself (a pedestal top the
    // monocular fill depth put behind it) and stays.
    // A splat whose 3-sigma screen extent in the reference view leaves the footprint also draws the
    // neighbours (a floor splat under the basket reaching under the rug fringe): never pruned or painted.
    let within_foot = |i: usize, p: usize, z: f32| {
        let ls = &host.scales[3 * i..3 * i + 3];
        let r = (3.0 * ls[0].max(ls[1]).max(ls[2]).exp() * pin.fx / z).ceil() as isize;
        let (x, y) = ((p % w) as isize, (p / w) as isize);
        [(r, 0), (-r, 0), (0, r), (0, -r)].iter().all(|&(dx, dy)| {
            let (u, v) = (x + dx, y + dy);
            u >= 0 && v >= 0 && (u as usize) < w && (v as usize) < h && foot.data[v as usize * w + u as usize]
        })
    };
    let mut prune_cand: Vec<usize> = vec![];
    for i in 0..n {
        if !keep[i] || !in_box(host.mean(i), 0.15) {
            continue;
        }
        let Some((p, z)) = ref_px(i) else { continue };
        if !(erode.data[p] && foot.data[p] && z < zmap[p] * 0.97 && within_foot(i, p, z)) {
            continue;
        }
        let want = fill.px(p % w, p / w);
        let dc = &host.sh[3 * host.k * i..3 * host.k * i + 3];
        let diff = (0..3).map(|c| (0.5 + SH_C0 * dc[c] - want[c]).abs()).sum::<f32>() / 3.0;
        if job.prune_colour > 0.0 && diff <= job.prune_colour {
            res.n_prune_kept_observed += 1;
        } else {
            prune_cand.push(i);
        }
    }
    // Support surface inside the object's footprint keeps old splats the object stood on (hoof marks:
    // grass-coloured, never observed as stone). Their colour becomes the median of the observed rim
    // of that surface just outside the hole (DC; higher SH zeroed).
    let mut paint_cand: Vec<(usize, [f32; 3])> = vec![];
    if surface.any() {
        let rim = surface.dilate(2 * rad).and(&hole.not());
        let mut cols: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        for p in (0..w * h).filter(|&p| rim.data[p] && rem.alpha[p] > 0.9) {
            for (c, v) in cols.iter_mut().enumerate() {
                v.push(rem.rgb.data[3 * p + c]);
            }
        }
        if cols[0].len() >= 16 {
            let col = cols.map(|v| geom::median(v));
            for i in (0..n).filter(|&i| keep[i] && in_box(host.mean(i), 0.15)) {
                if let Some((p, z)) = ref_px(i) {
                    // Centre inside the footprint is enough: paint only sets the DC colour to the rim
                    // median, and hoof-mark splats straddle the footprint edge.
                    if surface.data[p] && foot.data[p] && d_rem[p].is_finite() && (z - d_rem[p]).abs() < 0.03 * d_rem[p] {
                        paint_cand.push((i, col));
                    }
                }
            }
        }
    }
    // Prune only what the object hid from the other views. A splat that other train views
    // see outside the object's footprint (nothing in front of it there) is an observed neighbour: a
    // rug fringe or a stove base beside a removed basket. Pruning those streaked them.
    // Leftover fragments of the object itself lie inside its hole in every view and still go.
    let mut observed: HashMap<usize, u32> = prune_cand.iter().map(|&i| (i, 0)).collect();
    if !observed.is_empty() {
        let ids: Vec<usize> = observed.keys().copied().collect();
        for &v in train.iter().filter(|&&v| v != ri) {
            let (wv, hv) = dims[v];
            let pv = Pin::new(&views[v].camera, wv, hv);
            let r = render(&rem_splats, &views[v].camera, wv, hv, true).await?;
            let dv = r.depth.as_ref().expect("depth");
            // The tight footprint, not the dilated hole: a fringe touching the object lies inside
            // the dilation in most views.
            let hm = &footprints[&v];
            for &i in &ids {
                let pc = pv.to_cam(host.mean(i));
                if pc.z <= 0.01 {
                    continue;
                }
                let (u, y) = ((pc.x / pc.z * pv.fx + pv.cx).floor(), (pc.y / pc.z * pv.fy + pv.cy).floor());
                if u < 0.0 || y < 0.0 || u >= wv as f32 || y >= hv as f32 {
                    continue;
                }
                let q = y as usize * wv + u as usize;
                // Unoccluded outside the footprint. No opacity test: thin rug tassels never render opaque.
                if !hm.data[q] && !(dv[q].is_finite() && dv[q] < 0.95 * pc.z) {
                    *observed.get_mut(&i).expect("candidate") += 1;
                }
            }
        }
    }
    let seen = |i: usize| observed.get(&i).is_some_and(|&c| c >= OBSERVED_VIEWS);
    for &i in &prune_cand {
        if seen(i) {
            res.n_prune_kept_seen += 1;
        } else {
            keep[i] = false;
            res.n_pruned += 1;
        }
    }
    // Paint keeps only the footprint guard: the hoof marks a statue leaves on its
    // pedestal are visible around the hooves, and painting them is the point.
    let paint: Vec<(usize, [f32; 3])> = paint_cand.into_iter().filter(|&(i, _)| keep[i]).collect();
    res.n_surface_painted = paint.len();
    // Recolourable: kept old splats projecting into the (dilated) reference hole near the removed
    // object. They baked its shadow/occlusion into their colour; the fine-tune retrains their SH +
    // opacity with geometry frozen.
    let recol: Vec<bool> = (0..n)
        .map(|i| job.recolour && keep[i] && in_box(host.mean(i), job.recolour_pad) && ref_px(i).is_some_and(|(p, _)| foot.data[p]))
        .collect();
    res.n_recolour = recol.iter().filter(|&&r| r).count();
    progress("init", serde_json::json!({"n_seen_multiview": res.n_seen_multiview, "n_surface_skipped": res.n_surface_skipped, "n_front_dropped": res.n_front_dropped, "n_seen_skipped": res.n_seen_skipped, "n_new": res.n_new, "stride": stride, "n_pruned": res.n_pruned, "n_prune_kept_observed": res.n_prune_kept_observed, "n_recolour": res.n_recolour, "n_surface_painted": res.n_surface_painted, "n_prune_kept_seen": res.n_prune_kept_seen}));
    lap(&mut res, "init_s");

    // ---- 6. fine-tune the new splats (+ the recolourable old splats' colour)
    stage("finetune_setup");
    fill.save(&ft_dir.join("images").join(format!("{ri:05}.png")))?;
    Mask::new(w, h).to_gray(255, 255).save(ft_dir.join("masks").join(format!("{ri:05}.png")))?;
    let ft_view = |i: usize| FtView {
        image: PathBuf::from(format!("images/{i:05}.png")),
        mask: PathBuf::from(format!("masks/{i:05}.png")),
        camera: views[i].camera,
        camera_index: views[i].camera_index,
    };
    let mut ft_views: Vec<FtView> = train.iter().map(|&i| ft_view(i)).collect();
    // The loader samples views uniformly: repeat the reference so it is ~1/ref_every of the batches.
    let others = train.len() - 1;
    let copies = if job.ref_every <= 1 { others.max(1) * 4 } else { (others as f64 / (job.ref_every - 1) as f64).round() as usize };
    ft_views.extend((1..copies.max(1)).map(|_| ft_view(ri)));
    // Order: [frozen old..., recolourable old..., new...]; each group keeps the PLY order.
    let frozen: Vec<bool> = keep.iter().zip(&recol).map(|(&k, &r)| k && !r).collect();
    let mut merged = host.subset(&frozen);
    let n_frozen = merged.n();
    merged.append(&host.subset(&recol));
    let n_old = merged.n();
    merged.append(&new);
    let init_ls: Vec<f32> = new.scales.iter().step_by(3).copied().collect();
    drop(new);
    drop(full);
    drop(rem_splats);
    let splats = merged.to_splats(&device);
    drop(merged);
    let tuned = gpu::finetune(splats, n_frozen, n_old, &ft_dir, &ft_views, job.steps, job.max_resolution, job.seed, &device).await?;
    res.steps = job.steps;
    res.losses = tuned.losses.clone();
    let splats = tuned.splats;
    lap(&mut res, "finetune_s");

    // ---- 7. output PLY: kept records verbatim (recoloured ones with their trained SH +
    // opacity, in place) + the trained new splats
    stage("ply");
    let trained = gpu::read_rows(&splats, n_frozen, host.k, host.mode).await?;
    let n_rec = n_old - n_frozen;
    let rest = host.k - 1;
    // Row i of `trained` (rows n_frozen.. of the fine-tuned splats) as a PLY property value.
    let value = |i: usize, name: &str| -> Option<f32> {
        let at = |v: &[f32], s: usize, j: usize| Some(v[s * i + j]);
        match name {
            "x" => at(&trained.means, 3, 0),
            "y" => at(&trained.means, 3, 1),
            "z" => at(&trained.means, 3, 2),
            "opacity" => at(&trained.opac, 1, 0),
            _ => {
                let idx = |pre: &str| name.strip_prefix(pre).and_then(|r| r.parse::<usize>().ok());
                if let Some(j) = idx("scale_") {
                    (j < 3).then(|| trained.scales[3 * i + j])
                } else if let Some(j) = idx("rot_") {
                    (j < 4).then(|| trained.rots[4 * i + j])
                } else if let Some(j) = idx("f_dc_") {
                    (j < 3).then(|| trained.sh[3 * host.k * i + j])
                } else if let Some(j) = idx("f_rest_") {
                    // Channel-major: f_rest_{ch * rest + c} = coefficient c + 1 of channel ch.
                    let (ch, c) = (j / rest.max(1), j % rest.max(1));
                    (ch < 3 && rest > 0).then(|| trained.sh[3 * host.k * i + 3 * (c + 1) + ch])
                } else {
                    None
                }
            }
        }
    };
    // Recoloured old records: only SH + opacity are rewritten (geometry and any other property
    // stay verbatim). `recol` indices ascend, matching their row order in `trained`.
    let mut ply = ply;
    for &(i, col) in &paint {
        ply.patch(i, |name| {
            if let Some(c) = name.strip_prefix("f_dc_").and_then(|r| r.parse::<usize>().ok()) {
                (c < 3).then(|| (col[c] - 0.5) / SH_C0)
            } else {
                name.starts_with("f_rest_").then_some(0.0)
            }
        });
    }
    for (r, i) in (0..n).filter(|&i| recol[i]).enumerate() {
        ply.patch(i, |name| (name == "opacity" || name.starts_with("f_dc_") || name.starts_with("f_rest_")).then(|| value(r, name)).flatten());
    }
    let mut extra = Vec::with_capacity((trained.n() - n_rec) * ply.stride);
    for i in n_rec..trained.n() {
        if job.max_stretch > 0.0 {
            // Flat (one thin axis) is a surface; the two largest axes measure a streak.
            let mut ls = [trained.scales[3 * i], trained.scales[3 * i + 1], trained.scales[3 * i + 2]];
            ls.sort_by(f32::total_cmp);
            if ls[2] - init_ls[i - n_rec] > job.max_stretch.ln() || ls[2] - ls[1] > 8f32.ln() {
                res.n_stretch_dropped += 1;
                continue;
            }
        }
        ply.encode(&mut extra, |name| value(i, name));
    }
    let out_ply = job.out.join("splat.ply");
    res.n_output = ply.write_filtered(&out_ply, &keep, &extra)?;
    res.ply = out_ply;
    lap(&mut res, "ply_s");

    // ---- before/after of the reference view; held-out stability outside the hole
    stage("metrics");
    let full = host.to_splats(&device);
    let rem_splats = host.subset(&keep_old).to_splats(&device);
    let after = render(&splats, &rcam, w, h, false).await?;
    let imgs = job.out.join("images");
    let mut overlay = base.rgb.clone();
    for &p in &hole_px {
        for (c, tint) in [1.0f32, 0.0, 0.0].into_iter().enumerate() {
            overlay.data[3 * p + c] = overlay.data[3 * p + c] * 0.4 + tint * 0.6;
        }
    }
    let named = [("before", &base.rgb), ("hole", &overlay), ("removed", &rem.rgb), ("fill", &fill), ("after", &after.rgb)];
    for (k, im) in named {
        im.save(&imgs.join(format!("{k}.png")))?;
        res.images.push(format!("images/{k}.png"));
    }
    let (top, left, side) = img::square_crop(&hole, 2.2, 256);
    let tile = |im: &RgbF| im.crop(top, left, side, side).resize(320, 320).to_rgb8();
    img::strip(&named.iter().map(|(_, im)| tile(im)).collect::<Vec<_>>()).save(imgs.join("strip.jpg"))?;
    res.images.push("images/strip.jpg".into());
    let (mut outs, mut rems) = (vec![], vec![]);
    for &i in &held {
        let (w, h) = dims[i];
        let m = holes[&i].not();
        let o = render(&full, &views[i].camera, w, h, false).await?;
        let e = render(&splats, &views[i].camera, w, h, false).await?;
        let r = render(&rem_splats, &views[i].camera, w, h, false).await?;
        let pe = psnr_masked(&e.rgb, &o.rgb, &m);
        let pr = psnr_masked(&r.rgb, &o.rgb, &m);
        outs.extend(pe);
        rems.extend(pr);
        let mut row = BTreeMap::new();
        row.insert("view".into(), serde_json::json!(gpu::view_name(&views[i])));
        row.insert("hole_frac".into(), serde_json::json!(holes[&i].count() as f64 / (w * h) as f64));
        row.insert("psnr_out_edit_vs_orig".into(), serde_json::json!(pe));
        row.insert("psnr_out_removed_vs_orig".into(), serde_json::json!(pr));
        res.heldout.push(row);
    }
    let mean = |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
    res.heldout_views = held.len();
    res.psnr_out_edit_vs_orig_mean = mean(&outs);
    res.psnr_out_edit_vs_orig_min = outs.iter().copied().reduce(f64::min);
    res.psnr_out_removed_vs_orig_mean = mean(&rems);
    lap(&mut res, "metrics_s");
    if !job.keep_finetune_dir {
        std::fs::remove_dir_all(&ft_dir).ok();
    }
    res.timings_s.insert("total_s".into(), t0.elapsed().as_secs_f64());
    write_json(&job.out.join("result.json"), &res)?;
    progress("done", serde_json::json!({"n_output": res.n_output, "total_s": res.timings_s["total_s"]}));
    Ok(res)
}

fn write_json(path: &Path, v: &impl Serialize) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(v)?).with_context(|| format!("writing {}", path.display()))
}

/// Worker entry: `splat __brush inpaint <job.json>` (splat-brush dispatches here).
pub fn run_job_file(job: &Path) -> anyhow::Result<InpaintResult> {
    let job: InpaintJob = serde_json::from_slice(&std::fs::read(job).with_context(|| format!("reading {}", job.display()))?)
        .with_context(|| format!("parsing {}", job.display()))?;
    run(job)
}
