//! `__brush align-eval <job.json>`: the product's held-out test-time pose alignment applied to any
//! 3DGS PLY (e.g. a Spirula reference checkpoint), so a reference is scored aligned like the
//! product. For each held-out view the splats stay frozen and only the view's rigid camera-space
//! correction is optimised (`pose_refine::align_view`, the same steps, LRs and loss as the trainer's
//! `pose_align_eval_steps`). Writes `align_eval.json` (per-view unaligned and aligned PSNR/SSIM as
//! this renderer sees them, plus each correction) and, with `sparse_out`, a copy of the dataset's
//! COLMAP model whose held-out images carry the aligned world-to-camera (training images byte-equal),
//! so another engine can render the aligned poses with its own renderer.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, ensure};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_train::config::TrainConfig;
use brush_train::eval::eval_stats;
use brush_train::pose_refine::{ALIGN_EVAL_LR, align_view, refined_w2c, rigid_delta};
use burn::module::Module;
use glam::{Affine3A, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::engine;
use crate::util::*;
use crate::views::stem;

#[derive(Serialize, Deserialize)]
pub struct AlignEvalJob {
    /// 3DGS PLY to align against (ours or another engine's; untagged PLYs render as plain 3DGS).
    pub ply: PathBuf,
    /// COLMAP dataset dir (images/ + sparse/0).
    pub dataset: PathBuf,
    /// Held-out image names (as in images.bin); every other registered view is a training view.
    pub heldout_names: Vec<String>,
    /// Alignment steps per view (the product's `pose_align_eval_steps`).
    #[serde(default = "default_steps")]
    pub steps: u32,
    /// Longest image edge; 0 = native size (what Spirula's eval renders).
    #[serde(default)]
    pub max_resolution: u32,
    /// Output dir for align_eval.json.
    pub out: PathBuf,
    /// Where to write the aligned COLMAP model (cameras.bin, images.bin, points3D.bin).
    #[serde(default)]
    pub sparse_out: Option<PathBuf>,
    /// The candidate's `pose_refine.json` (product evaluate): each held-out view starts from the
    /// refined correction of its nearest training camera, on its group's learned focal, with the
    /// trainer's extent; as the trainer's own end-of-job alignment did.
    #[serde(default)]
    pub pose_refine: Option<PathBuf>,
    /// Translation LR unit; default the refinement's extent, else 1.1 x the farthest training camera.
    #[serde(default)]
    pub extent: Option<f32>,
    /// Loss SSIM weight; default the trainer's default.
    #[serde(default)]
    pub ssim_weight: Option<f32>,
    /// Write each aligned view's render to `<renders_out>/<stem>.png` (as the trainer saves its eval renders).
    #[serde(default)]
    pub renders_out: Option<PathBuf>,
    #[serde(default)]
    pub gpu_lock: Option<PathBuf>,
}

fn default_steps() -> u32 {
    200
}

/// `w2c' = D w2c` in f64 on a COLMAP record (qvec w,x,y,z; tvec).
fn compose(d: &Affine3A, qvec: [f64; 4], tvec: [f64; 3]) -> ([f64; 4], [f64; 3]) {
    let (_, r, t) = d.to_scale_rotation_translation();
    let rd = DQuat::from_xyzw(r.x as f64, r.y as f64, r.z as f64, r.w as f64).normalize();
    let q = DQuat::from_xyzw(qvec[1], qvec[2], qvec[3], qvec[0]);
    let q2 = (rd * q).normalize();
    let t2 = rd * DVec3::from_array(tvec) + t.as_dvec3();
    ([q2.w, q2.x, q2.y, q2.z], t2.to_array())
}

/// A trained candidate's pose refinement (`pose_refine.json`), as test-time alignment starts from it.
struct Refinement {
    extent: f32,
    /// Learned focal scale per intrinsics group (camera index); empty or 1.0 when focal is not refined.
    focal_scales: Vec<f32>,
    /// (training camera centre, its refined correction `D`, with `w2c' = D w2c`).
    cameras: Vec<(glam::Vec3, Affine3A)>,
}

impl Refinement {
    fn read(path: &std::path::Path, train: &[&brush_dataset::scene::SceneView]) -> anyhow::Result<Self> {
        let v: serde_json::Value = read_json(path)?;
        let f = |a: &serde_json::Value, k: usize| a.get(k).and_then(serde_json::Value::as_f64).unwrap_or(0.0) as f32;
        let mut cameras = Vec::new();
        for row in v["views"].as_array().into_iter().flatten() {
            let Some(name) = row["view"].as_str() else { continue };
            let Some(view) = train.iter().find(|v| engine::view_name(v) == name) else { continue };
            let (q, t) = (&row["qvec"], &row["tvec"]);
            let w2c = Affine3A::from_rotation_translation(
                glam::Quat::from_xyzw(f(q, 1), f(q, 2), f(q, 3), f(q, 0)).normalize(), glam::Vec3::new(f(t, 0), f(t, 1), f(t, 2)));
            // As the warm start restores corrections: D = w2c' c2w.
            cameras.push((view.camera.position, w2c * view.camera.local_to_world()));
        }
        ensure!(!cameras.is_empty(), "{}: no training view corrections", path.display());
        let extent = v["extent"].as_f64().context("pose_refine.json: no extent")? as f32;
        let focal_scales = v["focal_scales"].as_array().into_iter().flatten().map(|s| s.as_f64().unwrap_or(1.0) as f32).collect();
        Ok(Self { extent, focal_scales, cameras })
    }

    /// (fixed camera-space base: the view group's learned focal, init: the nearest training camera's correction).
    fn start(&self, view: &brush_dataset::scene::SceneView) -> (Affine3A, Affine3A) {
        let s = self.focal_scales.get(view.camera_index).copied().unwrap_or(1.0);
        let base = Affine3A::from_scale(glam::Vec3::new(s, s, 1.0));
        let p = view.camera.position;
        let init = self.cameras.iter().min_by(|a, b| a.0.distance_squared(p).total_cmp(&b.0.distance_squared(p)))
            .map_or(Affine3A::IDENTITY, |c| c.1);
        (base, init)
    }
}

pub fn run_job(job: AlignEvalJob) -> anyhow::Result<i32> {
    let _gpu = crate::gpu_lock(job.gpu_lock.as_deref())?;
    std::fs::create_dir_all(&job.out)?;
    let rt = engine::runtime()?;
    rt.block_on(async {
        let result = align(job).await;
        // As in the train worker: lagged render-count readbacks still in flight at exit race the
        // driver's teardown (SIGSEGV after a finished job, seen on fal).
        brush_render::render::settle_lagged_counts().await;
        result
    })
}

async fn align(job: AlignEvalJob) -> anyhow::Result<i32> {
    let device = engine::init_device().await;
    let (splats, _) = engine::load_ply(&job.ply, SplatRenderMode::Default, &device).await?;
    let max_res = if job.max_resolution == 0 { u16::MAX as u32 } else { job.max_resolution };
    let scene = engine::load_scene(&job.dataset, max_res).await?;
    for w in &scene.warnings {
        eprintln!("[dataset] {w}");
    }
    // The trainer's translation LR unit: 1.1 x the farthest training camera from their mean.
    let train: Vec<_> = scene.views.iter().filter(|v| !job.heldout_names.contains(&engine::view_name(v))).collect();
    ensure!(!train.is_empty(), "no training views left");
    let refine = job.pose_refine.as_deref().map(|p| Refinement::read(p, &train)).transpose()?;
    let center = train.iter().map(|v| v.camera.position).sum::<glam::Vec3>() / train.len() as f32;
    let extent = job.extent.or(refine.as_ref().map(|r| r.extent))
        .unwrap_or_else(|| 1.1 * train.iter().map(|v| v.camera.position.distance(center)).fold(0.0, f32::max).max(1e-3));
    let ssim_weight = job.ssim_weight.unwrap_or(TrainConfig::default().ssim_weight);
    if let Some(dir) = &job.renders_out {
        std::fs::create_dir_all(dir)?;
    }
    let train_splats = splats.clone().train();
    let t = Instant::now();
    let mut rows = Vec::with_capacity(job.heldout_names.len());
    let mut aligned = std::collections::HashMap::new();
    for (i, name) in job.heldout_names.iter().enumerate() {
        let view = scene.views.iter().find(|v| &engine::view_name(v) == name).with_context(|| format!("{name} not in dataset"))?;
        let gt = engine::load_gt(view).await?;
        let mode = engine::gt_alpha_mode(view);
        let before = eval_stats(splats.clone(), &view.camera, gt.clone(), mode, &device).await?;
        let (base, init) = refine.as_ref().map_or((Affine3A::IDENTITY, Affine3A::IDENTITY), |r| r.start(view));
        let a = align_view(&train_splats, &view.camera, base, init, &gt, mode, job.steps,
            ALIGN_EVAL_LR, extent, ssim_weight).await?;
        let after = eval_stats(splats.clone(), &a.camera, gt, mode, &device).await?;
        if let Some(dir) = &job.renders_out {
            after.save_to_disk(&dir.join(format!("{}.png", stem(name)))).await?;
        }
        let (rot_deg, trans) = rigid_delta(&a.correction);
        // The aligned pose (COLMAP world-to-camera), so pose accuracy can cover held-out views too.
        let (qvec, tvec) = refined_w2c(&view.camera, &a.correction);
        let (p0, s0) = (before.psnr.into_scalar_async::<f32>().await? as f64, before.ssim.into_scalar_async::<f32>().await? as f64);
        let (p1, s1) = (after.psnr.into_scalar_async::<f32>().await? as f64, after.ssim.into_scalar_async::<f32>().await? as f64);
        eprintln!("[align] view {}/{} {name} psnr {p0:.2} -> {p1:.2} rot {rot_deg:.3} deg", i + 1, job.heldout_names.len());
        rows.push(json!({"image": stem(name), "name": name, "psnr_unaligned": p0, "ssim_unaligned": s0, "psnr": p1, "ssim": s1,
            "view": name, "qvec": qvec, "tvec": tvec, "rot_deg": rot_deg, "trans": trans, "trans_rel": trans / extent, "loss_before": a.first_loss, "loss_after": a.last_loss}));
        aligned.insert(name.clone(), a.correction);
    }
    let align_s = t.elapsed().as_secs_f64();
    let mean = |k: &str| rows.iter().map(|r| r[k].as_f64().unwrap_or(f64::NAN)).sum::<f64>() / rows.len().max(1) as f64;
    let mut max_rot = rows.iter().map(|r| r["rot_deg"].as_f64().unwrap_or(0.0)).collect::<Vec<_>>();
    max_rot.sort_by(f64::total_cmp);
    if let Some(dst) = &job.sparse_out {
        let src = job.dataset.join("sparse/0");
        std::fs::create_dir_all(dst)?;
        for f in ["cameras.bin", "points3D.bin"] {
            if src.join(f).exists() {
                std::fs::copy(src.join(f), dst.join(f))?;
            }
        }
        let mut images = splat_eval::colmap::read_images_bin(&src.join("images.bin"))?;
        let mut n = 0;
        for im in &mut images {
            if let Some(d) = aligned.get(&im.name) {
                (im.qvec, im.tvec) = compose(d, im.qvec, im.tvec);
                n += 1;
            }
        }
        ensure!(n == aligned.len(), "{} of {} held-out views found in images.bin", n, aligned.len());
        splat_eval::colmap::write_images_bin(&dst.join("images.bin"), &images)?;
    }
    write_json(&job.out.join("align_eval.json"), &json!({
        "schema": "splat/align-eval/v1",
        // Metrics below are this binary's renderer (brush eval_stats; no LPIPS): another engine's
        // scores at the aligned poses come from rendering `sparse_out` with that engine.
        "renderer": "splat",
        "ply": job.ply, "dataset": job.dataset, "max_resolution": job.max_resolution,
        "align": {"steps": job.steps, "lr_rot_rad": ALIGN_EVAL_LR.rot, "lr_trans_extents": ALIGN_EVAL_LR.trans,
            "lr_decay": 0.01, "optimizer": "adam", "loss": format!("{:.2} L1 + {ssim_weight:.2} D-SSIM", 1.0 - ssim_weight),
            "background": "black", "extent": extent, "splats": "frozen",
            "focal": if refine.is_some() { "the refinement's learned focal (fixed)" } else { "fixed (rigid pose only)" },
            "init": if refine.is_some() { "nearest training camera's refined correction" } else { "identity" }},
        "n": rows.len(), "align_s": align_s,
        "psnr_unaligned": mean("psnr_unaligned"), "psnr": mean("psnr"),
        "ssim_unaligned": mean("ssim_unaligned"), "ssim": mean("ssim"),
        "rot_deg_max": max_rot.last(), "rot_deg_median": max_rot.get(max_rot.len() / 2),
        "sparse_out": job.sparse_out, "per_image": rows,
    }))?;
    eprintln!("[align] {} views in {align_s:.1}s: psnr {:.3} -> {:.3}", rows.len(), mean("psnr_unaligned"), mean("psnr"));
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f64 composition on a COLMAP record matches the trainer's `refined_w2c` (f32 camera path).
    #[test]
    fn compose_matches_refined_w2c() {
        let q = DQuat::from_euler(glam::EulerRot::XYZ, 0.3, -1.1, 0.4);
        let t = DVec3::new(0.5, -1.8, 4.3);
        let w2c = Affine3A::from_rotation_translation(q.as_quat(), t.as_vec3());
        let c2w = w2c.inverse();
        let (_, rot, pos) = c2w.to_scale_rotation_translation();
        let cam = brush_render::camera::Camera::new(pos, rot, 1.0, 1.0, glam::Vec2::splat(0.5), Default::default());
        let d = Affine3A::from_rotation_translation(glam::Quat::from_rotation_y(0.01), glam::Vec3::new(0.002, -0.001, 0.003));
        let (qa, ta) = brush_train::pose_refine::refined_w2c(&cam, &d);
        let (qb, tb) = compose(&d, [q.w, q.x, q.y, q.z], t.to_array());
        let sign = if (qa[0] as f64) * qb[0] < 0.0 { -1.0 } else { 1.0 };
        for k in 0..4 {
            assert!((qa[k] as f64 - sign * qb[k]).abs() < 1e-5, "q{k}: {qa:?} vs {qb:?}");
        }
        for k in 0..3 {
            assert!((ta[k] as f64 - tb[k]).abs() < 1e-4, "t{k}: {ta:?} vs {tb:?}");
        }
    }
}
