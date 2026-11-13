//! `render_views` and `evaluate`: a short GPU worker renders named capture views (with their
//! ground truth) or an orbit from a candidate's PLY; the tool then scores / composes images.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, bail};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_train::eval::eval_stats;
use glam::UVec2;
use image::{GenericImage, RgbImage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;

use crate::render::{RenderCamera, orbit_cameras};
use crate::util::*;
use crate::{BrushConfig, engine, split};

#[derive(Serialize, Deserialize)]
pub struct RenderJob {
    pub ply: PathBuf,
    pub dataset: PathBuf,
    pub max_resolution: u32,
    /// Capture views to render with ground truth, by image name.
    pub names: Vec<String>,
    /// Orbit views to render instead (no ground truth).
    pub orbit: Option<usize>,
    pub out: PathBuf,
    pub depth: bool,
    pub gpu_lock: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ViewScore {
    pub name: String,
    pub stem: String,
    pub psnr: f64,
    pub ssim: f64,
}

pub fn stem(name: &str) -> String {
    Path::new(name).with_extension("").to_string_lossy().replace(['/', '\\'], "__")
}

/// Worker body: writes out/gt/<stem>.png, out/renders/<stem>.png (+ alpha/, depth/) and scores.json.
pub fn run_job(job: RenderJob) -> anyhow::Result<i32> {
    let wait = Instant::now();
    let _gpu = crate::gpu_lock(job.gpu_lock.as_deref())?;
    std::fs::create_dir_all(&job.out)?;
    write_json(&job.out.join("timing.json"), &json!({"gpu_lock_wait_s": wait.elapsed().as_secs_f64()}))?;
    let rt = engine::runtime()?;
    rt.block_on(async {
        let device = engine::init_device().await;
        let (splats, _) = engine::load_ply(&job.ply, SplatRenderMode::Default, &device).await?;
        // A surfel model was trained (and is scored) on the pinhole copy of a distorted dataset.
        let (dataset, _pinhole) = if splats.surfel {
            engine::surfel_dataset(&job.dataset, &job.out.join("pinhole_dataset"))?
        } else {
            (job.dataset.clone(), None)
        };
        let scene = engine::load_scene(&dataset, job.max_resolution).await?;
        for d in ["gt", "renders", "alpha", "depth"] {
            std::fs::create_dir_all(job.out.join(d))?;
        }
        let mut scores = vec![];
        let mut rendered_cameras = vec![];
        if let Some(n) = job.orbit {
            let mut capture = vec![];
            for v in &scene.views {
                let (w, h) = v.image.dimensions().await?;
                capture.push(RenderCamera::from_brush(&v.camera, w, h));
            }
            for (i, cam) in orbit_cameras(&capture, n).into_iter().enumerate() {
                let frame = engine::render(&splats, &cam.to_brush(), UVec2::new(cam.width, cam.height), job.depth).await?;
                write_frame(&job.out, &format!("orbit_{i:03}"), &frame)?;
            }
        } else {
            for (i, name) in job.names.iter().enumerate() {
                let view = scene
                    .views
                    .iter()
                    .find(|v| &engine::view_name(v) == name)
                    .with_context(|| format!("view {name} not in dataset"))?;
                let gt = engine::load_gt(view).await?;
                let size = UVec2::new(gt.width(), gt.height());
                let camera = RenderCamera::from_brush(&view.camera, size.x, size.y);
                rendered_cameras.push(json!({"source_name":name, "position":camera.position.to_array(),
                    "c2w_xyzw":camera.rotation.to_array(), "pinhole_K":[camera.fx,camera.fy,camera.cx,camera.cy],
                    "size":[size.x,size.y], "note":"K is the pinhole approximation; rendering retains the dataset lens model"}));
                let sample = eval_stats(splats.clone(), &view.camera, gt.clone(), engine::gt_alpha_mode(view), &device).await?;
                let s = stem(name);
                scores.push(ViewScore {
                    name: name.clone(),
                    stem: s.clone(),
                    psnr: sample.psnr.into_scalar_async::<f32>().await? as f64,
                    ssim: sample.ssim.into_scalar_async::<f32>().await? as f64,
                });
                // Heartbeat per scored view for stall watchdogs (output = liveness).
                eprintln!("[eval] view {}/{} {s} psnr {:.2}", i + 1, job.names.len(), scores[scores.len() - 1].psnr);
                gt.to_rgb8().save(job.out.join("gt").join(format!("{s}.png")))?;
                let frame = engine::render(&splats, &view.camera, size, job.depth).await?;
                write_frame(&job.out, &s, &frame)?;
            }
        }
        write_json(&job.out.join("scores.json"), &scores)?;
        if job.orbit.is_none() {
            let quality = splat_eval::view_quality::view_quality(scores.iter().map(|s| (s.name.as_str(), Some(s.psnr))));
            write_json(&job.out.join("view_quality.json"), &quality)?;
            if let Some(warning) = quality.warning() { eprintln!("[eval] {warning}"); }
            write_json(&job.out.join("camera_provenance.json"), &json!({"source":"dataset_cameras",
                "training_pose_corrections_applied":false, "views":rendered_cameras}))?;
        }
        let dropped = brush_render::render::surfel_lens_dropped();
        if dropped > 0 {
            write_json(&job.out.join("surfel_lens_dropped.json"), &json!({"renders": dropped}))?;
        }
        anyhow::Ok(0)
    })
}

fn write_frame(out: &Path, stem: &str, frame: &engine::RawFrame) -> anyhow::Result<()> {
    engine::to_rgb8(frame).save(out.join("renders").join(format!("{stem}.png")))?;
    engine::alpha8(frame).save(out.join("alpha").join(format!("{stem}.png")))?;
    if let Some(depth) = &frame.depth {
        // 16-bit PNG scaled to the frame's max depth (recorded alongside); 0 = no hit.
        let max = depth.iter().copied().filter(|d| d.is_finite()).fold(0f32, f32::max).max(1e-6);
        let px: Vec<u16> = depth.iter().map(|&d| if d.is_finite() { ((d / max).clamp(0.0, 1.0) * 65535.0) as u16 } else { 0 }).collect();
        image::ImageBuffer::<image::Luma<u16>, _>::from_raw(frame.width, frame.height, px)
            .expect("sized buffer")
            .save(out.join("depth").join(format!("{stem}.png")))?;
        write_json(&out.join("depth").join(format!("{stem}.json")), &json!({
            "max_depth": max,
            "encoding": "u16 = depth / max_depth * 65535, 0 = no hit",
            "float32": {
                "layout": "little-endian f32, row-major, width x height (normal: x,y,z per pixel)",
                "expected": format!("{stem}.expected.f32"),
                "median": format!("{stem}.median.f32"),
                "normal": format!("{stem}.normal.f32"),
                "alpha": format!("{stem}.alpha.f32"),
                "no_hit": "NaN depth / zero normal where alpha <= 1e-3; alpha is the raw accumulated opacity",
                "depth": "view-space z (along the camera forward axis)",
                "normal_space": "world, unit, camera-facing",
                "width": frame.width, "height": frame.height,
            },
        }))?;
        let f32_bytes = |v: &mut dyn Iterator<Item = f32>| v.flat_map(f32::to_le_bytes).collect::<Vec<u8>>();
        std::fs::write(out.join("depth").join(format!("{stem}.expected.f32")), f32_bytes(&mut depth.iter().copied()))?;
        std::fs::write(out.join("depth").join(format!("{stem}.alpha.f32")), f32_bytes(&mut frame.rgba.chunks_exact(4).map(|p| p[3])))?;
        if let Some(median) = &frame.median_depth {
            std::fs::write(out.join("depth").join(format!("{stem}.median.f32")), f32_bytes(&mut median.iter().copied()))?;
        }
        if let Some(normal) = &frame.normal {
            std::fs::write(out.join("depth").join(format!("{stem}.normal.f32")), f32_bytes(&mut normal.iter().flatten().copied()))?;
        }
    }
    Ok(())
}

/// Run the render worker synchronously (GPU work stays out of the tool-server process).
pub fn render_sync(cfg: &BrushConfig, job: &RenderJob) -> anyhow::Result<()> {
    std::fs::create_dir_all(&job.out)?;
    let spec = job.out.join("render_job.json");
    write_json(&spec, job)?;
    let out = Command::new(&cfg.exe).args(["__brush", "render"]).arg(&spec).output().context("spawning render worker")?;
    std::fs::write(job.out.join("render.log"), [&out.stdout[..], &out.stderr[..]].concat())?;
    if !out.status.success() {
        bail!("render worker exited {}: {}", out.status, tail_file(&job.out.join("render.log"), 1500));
    }
    Ok(())
}

/// Run the test-time alignment worker (`__brush align-eval`) synchronously; returns its align_eval.json.
fn align_sync(cfg: &BrushConfig, job: &crate::align_eval::AlignEvalJob) -> anyhow::Result<Value> {
    std::fs::create_dir_all(&job.out)?;
    let spec = job.out.join("align_job.json");
    write_json(&spec, job)?;
    let out = Command::new(&cfg.exe).args(["__brush", "align-eval"]).arg(&spec).output().context("spawning alignment worker")?;
    std::fs::write(job.out.join("align.log"), [&out.stdout[..], &out.stderr[..]].concat())?;
    if !out.status.success() {
        bail!("alignment worker exited {}: {}", out.status, tail_file(&job.out.join("align.log"), 1500));
    }
    read_json(&job.out.join("align_eval.json"))
}

/// The PLY of a candidate's latest (final or checkpoint) state.
pub fn candidate_ply(dir: &Path) -> anyhow::Result<PathBuf> {
    for name in ["splat.ply", "checkpoint.ply"] {
        if dir.join(name).is_file() {
            return Ok(dir.join(name));
        }
    }
    Err(invalid("candidate has no saved splat yet"))
}

/// Side-by-side gt|render composites, re-encoded so all of them fit the tool-result image budget.
pub fn comparisons(gt_dir: &Path, render_dir: &Path, stems: &[String], dest: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let full = dest.join("full");
    std::fs::create_dir_all(&full)?;
    let mut paths = vec![];
    for s in stems {
        let gt = image::open(gt_dir.join(format!("{s}.png")))?.to_rgb8();
        let rd = image::open(render_dir.join(format!("{s}.png")))?.to_rgb8();
        let mut canvas = RgbImage::new(gt.width() * 2, gt.height());
        canvas.copy_from(&gt, 0, 0)?;
        canvas.copy_from(&rd, gt.width(), 0)?;
        let p = full.join(format!("comparison-{s}.png"));
        canvas.save(&p)?;
        paths.push(p);
    }
    splat_eval::budget_jpegs(&paths, dest, splat_eval::RESULT_IMAGE_BUDGET)
}

fn add_lock_wait(out: &Path, metrics: &mut serde_json::Map<String, Value>) {
    if let Ok(t) = read_json::<Value>(&out.join("timing.json")) {
        metrics.insert("gpu_lock_wait_s".into(), t["gpu_lock_wait_s"].clone());
    }
}

fn candidate(run: &dyn RunCtx, id: &str) -> anyhow::Result<(Artifact, PathBuf, splat_core::split::ResolvedSplit, PathBuf)> {
    let (cand, dir) = get_artifact(run, id, ArtifactKind::Candidate)?;
    let (split_id, sfm_id) = (meta_str(&cand, "split_id")?.to_string(), meta_str(&cand, "sfm_id")?.to_string());
    let manifest = split::resolve(run, &split_id, &sfm_id)?;
    let dataset = run.path(&sfm_id)?;
    Ok((cand, dir, manifest, dataset))
}

fn max_resolution(cand: &Artifact) -> u32 {
    cand.meta.get("max_resolution").and_then(Value::as_u64).unwrap_or(1920) as u32
}

pub fn evaluate(cfg: &BrushConfig, run: &dyn RunCtx, req: Evaluate) -> anyhow::Result<ToolResult> {
    let start = Instant::now();
    let (cand, dir, manifest, dataset) = candidate(run, &req.candidate_id)?;
    let trusted_cameras = crate::capture_sfm::trusted(&dataset)?;
    if meta_str(&cand, "split_id")? != req.split_id {
        return Err(invalid("candidate was trained against a different split"));
    }
    let ply = candidate_ply(&dir)?;
    if manifest.heldout_names.is_empty() && trusted_cameras {
        let mut r = result("evaluate", Status::Succeeded);
        r.metrics.insert("evaluated".into(), json!(false));
        r.metrics.insert("reason".into(), json!("trusted import declares no held-out views; quality unmeasured"));
        r.metrics.insert("camera_source".into(), json!("imported"));
        r.metrics.insert("sfm_skipped".into(), json!(true));
        r.warnings.push("No authored held-out views; quality has not been evaluated.".into());
        return Ok(r);
    }
    let (artifact, out) = new_artifact(run, ArtifactKind::Eval, json!({"candidate_id": req.candidate_id, "split_id": req.split_id, "ply": rel(run, &ply)}))?;
    // train_splats already rendered the final splats on exactly these held-out views (same split,
    // same resolution): reuse them instead of rendering again. Score them when train_splats did not
    // (a `--cloud` worker leaves LPIPS to this job) or when LPIPS is wanted but was unavailable then.
    let trained = dir.join("eval");
    let reuse = ply == dir.join("splat.ply")
        && manifest.heldout_names.iter().all(|n| trained.join("renders").join(format!("{}.png", stem(n))).is_file());
    let src = if reuse {
        trained
    } else {
        render_sync(cfg, &RenderJob {
            ply: ply.clone(),
            dataset,
            max_resolution: max_resolution(&cand),
            names: manifest.heldout_names.clone(),
            orbit: None,
            out: out.clone(),
            depth: false,
            gpu_lock: cfg.gpu_lock.clone(),
        })?;
        out.clone()
    };
    // Test-time pose alignment (the trainer's pose refinement moves the training cameras' gauge):
    // when the candidate asks for it (train_summary heldout_pose_align), each held-out view's rigid
    // correction is fitted on the GPU with the splats frozen, starting from the nearest training
    // camera's refined correction, and the aligned renders are what is scored; the unaligned
    // scores stay beside them (psnr_unaligned, ssim_unaligned).
    let align = (!trusted_cameras && ply == dir.join("splat.ply"))
        .then(|| read_json::<Value>(&dir.join("train_summary.json")).ok())
        .flatten()
        .map(|t| t["heldout_pose_align"].clone())
        .filter(|a| a["steps"].as_u64().is_some_and(|s| s > 0));
    let aligned = match &align {
        Some(a) => Some(align_sync(cfg, &crate::align_eval::AlignEvalJob {
            ply: ply.clone(),
            // The trainer's dataset (masks included) at its resolution, as its held-out renders.
            dataset: a["dataset"].as_str().map(PathBuf::from).filter(|d| d.is_dir()).map_or_else(|| run.path(meta_str(&cand, "sfm_id")?), Ok)?,
            heldout_names: manifest.heldout_names.clone(),
            steps: a["steps"].as_u64().unwrap_or(0) as u32,
            max_resolution: a["max_resolution"].as_u64().filter(|&r| r > 0).map_or(max_resolution(&cand), |r| r as u32),
            out: out.join("aligned"),
            sparse_out: None,
            pose_refine: Some(dir.join("pose_refine.json")).filter(|p| p.is_file()),
            extent: a["extent"].as_f64().map(|e| e as f32),
            ssim_weight: a["ssim_weight"].as_f64().map(|w| w as f32),
            renders_out: Some(out.join("aligned").join("renders")),
            gpu_lock: cfg.gpu_lock.clone(),
        })?),
        None => None,
    };
    let render_dir = if aligned.is_some() { out.join("aligned").join("renders") } else { src.join("renders") };
    let (pairs, missing) = splat_eval::pair_dirs(&src.join("gt"), &render_dir)?;
    let cached = (reuse && aligned.is_none())
        .then(|| read_json::<splat_eval::EvalSummary>(&dir.join("metrics.json")).ok())
        .flatten()
        .filter(|s| !req.lpips || s.lpips_unavailable.is_none());
    let mut summary = match cached {
        Some(s) => s,
        None => {
            let opts = splat_eval::MetricOptions { lpips: req.lpips, models_dir: cfg.models_dir.clone(), cc_lpips: false, ..Default::default() };
            splat_eval::evaluate_pairs(&pairs, &missing, &opts)?
        }
    };
    if aligned.is_some() {
        // The same scorer on the unaligned renders (PSNR/SSIM only), per view and mean.
        let (pairs, missing) = splat_eval::pair_dirs(&src.join("gt"), &src.join("renders"))?;
        let opts = splat_eval::MetricOptions { lpips: false, models_dir: cfg.models_dir.clone(), cc_lpips: false, ..Default::default() };
        let unaligned = splat_eval::evaluate_pairs(&pairs, &missing, &opts)?;
        for k in ["psnr", "ssim"] {
            if let Some(v) = unaligned.mean.get(k) {
                summary.mean.insert(format!("{k}_unaligned"), *v);
            }
        }
        for row in &mut summary.per_image {
            let Some(u) = unaligned.per_image.iter().find(|u| u.get("image") == row.get("image")) else { continue };
            for k in ["psnr", "ssim"] {
                if let Some(v) = u.get(k) {
                    row.insert(format!("{k}_unaligned"), v.clone());
                }
            }
        }
    }
    let mut scores: Vec<ViewScore> = if let Some(a) = &aligned {
        a["per_image"].as_array().into_iter().flatten().map(|r| ViewScore {
            name: r["name"].as_str().unwrap_or_default().into(),
            stem: r["image"].as_str().unwrap_or_default().into(),
            psnr: r["psnr"].as_f64().unwrap_or(f64::NAN),
            ssim: r["ssim"].as_f64().unwrap_or(f64::NAN),
        }).collect()
    } else if reuse {
        let row = |s: &str, k: &str| summary.per_image.iter().find(|r| r.get("image").and_then(Value::as_str) == Some(s)).and_then(|r| r.get(k)).and_then(Value::as_f64).unwrap_or(f64::NAN);
        manifest.heldout_names.iter().map(|n| { let st = stem(n); ViewScore { name: n.clone(), psnr: row(&st, "psnr"), ssim: row(&st, "ssim"), stem: st } }).collect()
    } else {
        read_json(&out.join("scores.json"))?
    };
    let n = scores.len();
    if n != manifest.heldout_names.len() {
        bail!("rendered {n} of {} held-out views", manifest.heldout_names.len());
    }
    write_json(&out.join("metrics.json"), &summary)?;
    let mut metrics: serde_json::Map<String, Value> = summary.mean.iter().map(|(k, v)| (k.clone(), f64_metric(*v))).collect();
    let gpu_psnr = if let Some(a) = &aligned {
        a["psnr"].as_f64().unwrap_or(f64::NAN)
    } else if reuse {
        read_json::<Value>(&dir.join("train_summary.json")).ok().and_then(|t| t["heldout_psnr_brush"].as_f64()).unwrap_or(f64::NAN)
    } else {
        scores.iter().map(|s| s.psnr).sum::<f64>() / n.max(1) as f64
    };
    metrics.insert("brush_psnr".into(), f64_metric(gpu_psnr));
    metrics.insert("reused_train_renders".into(), json!(reuse && aligned.is_none()));
    if let Some(a) = &aligned {
        metrics.insert("pose_aligned_steps".into(), a["align"]["steps"].clone());
        metrics.insert("pose_align_s".into(), a["align_s"].clone());
        metrics.insert("brush_psnr_unaligned".into(), a["psnr_unaligned"].clone());
        // As the trainer wrote it before alignment moved here (pose-accuracy readers use the views' qvec/tvec).
        write_json(&out.join("eval_pose_align.json"), &json!({"steps": a["align"]["steps"], "unaligned_psnr": a["psnr_unaligned"],
            "psnr": a["psnr"], "align_s": a["align_s"], "align": a["align"], "views": a["per_image"]}))?;
    }
    metrics.insert("n_eval".into(), json!(summary.n));
    metrics.insert("heldout_coverage".into(), json!(summary.n as f64 / manifest.heldout_names.len().max(1) as f64));
    metrics.insert("split_verified".into(), json!(missing.is_empty()));
    if let Some(why) = &summary.lpips_unavailable {
        metrics.insert("lpips_unavailable".into(), json!(why));
    }
    if let Some(note) = &summary.lpips_note {
        metrics.insert("lpips_note".into(), json!(note));
    }
    if let Some(scale) = &summary.lpips_scale {
        metrics.insert("lpips_scale".into(), json!(scale));
    }
    // Scorer backend (ONNX Runtime provider); with lpips_scale it pins which LPIPS numbers compare.
    metrics.insert("lpips_device".into(), json!(summary.device));
    scores.sort_by(|a, b| a.psnr.total_cmp(&b.psnr));
    let view_quality = splat_eval::view_quality::view_quality(scores.iter().map(|s| (s.name.as_str(), Some(s.psnr))));
    metrics.insert("view_quality".into(), serde_json::to_value(&view_quality)?);
    metrics.insert("worst_views".into(), json!(scores.iter().take(8).map(|s| json!({"image": s.stem, "source_name": s.name, "psnr": s.psnr, "ssim": s.ssim})).collect::<Vec<_>>()));
    add_lock_wait(&out, &mut metrics);
    write_json(&out.join("summary.json"), &metrics)?;
    let worst: Vec<String> = scores.iter().take(8).map(|s| s.stem.clone()).collect();
    let images = comparisons(&src.join("gt"), &render_dir, &worst, &out.join("comparisons"))?;
    let mut warnings = vec![];
    warnings.extend(view_quality.warning());
    if metrics.get("cc_psnr").and_then(Value::as_f64).zip(metrics.get("psnr").and_then(Value::as_f64)).is_some_and(|(cc, p)| cc - p > 1.5) {
        warnings.push("Colour correction adds more than 1.5 dB; inspect exposure variation.".into());
    }
    if n < 8 {
        warnings.push("Fewer than eight held-out views.".into());
    }
    Ok(ToolResult {
        artifacts: vec![artifact],
        metrics: metrics.into_iter().collect(),
        images: images.iter().map(|p| rel(run, p)).collect(),
        warnings,
        wall_s: Some(start.elapsed().as_secs_f64()),
        ..result("evaluate", Status::Succeeded)
    })
}

pub fn render_views(cfg: &BrushConfig, run: &dyn RunCtx, req: RenderViews) -> anyhow::Result<ToolResult> {
    let start = Instant::now();
    let (cand, dir, manifest, dataset) = candidate(run, &req.candidate_id)?;
    let ply = candidate_ply(&dir)?;
    let k = req.max_views as usize;
    let pick = |names: &[String]| -> Vec<String> {
        let stride = (names.len() as f64 / k as f64).max(1.0);
        (0..k.min(names.len())).map(|i| names[(i as f64 * stride) as usize].clone()).collect()
    };
    let (mut names, orbit) = match req.views {
        ViewSet::Heldout => (pick(&manifest.heldout_names), None),
        ViewSet::TrainingSample => (pick(&manifest.train_names), None),
        ViewSet::Orbit => (vec![], Some(k)),
    };
    if !req.image_names.is_empty() {
        names = requested_views(&req, &manifest.train_names, &manifest.heldout_names)?;
    }
    let views = serde_json::to_value(req.views)?;
    let (artifact, out) = new_artifact(run, ArtifactKind::Images, json!({"candidate_id": req.candidate_id, "views": views, "ply": rel(run, &ply)}))?;
    render_sync(cfg, &RenderJob {
        ply,
        dataset,
        max_resolution: max_resolution(&cand),
        names,
        orbit,
        out: out.clone(),
        depth: true,
        gpu_lock: cfg.gpu_lock.clone(),
    })?;
    let mut metrics = serde_json::Map::new();
    let mut warnings = vec![];
    let images = if orbit.is_some() {
        metrics.insert("layout".into(), json!("render"));
        splat_eval::budget_jpegs(&list_images(&out.join("renders"))?, &out.join("previews"), splat_eval::RESULT_IMAGE_BUDGET)?
    } else {
        let scores: Vec<ViewScore> = read_json(&out.join("scores.json"))?;
        let quality = splat_eval::view_quality::view_quality(scores.iter().map(|s| (s.name.as_str(), Some(s.psnr))));
        warnings.extend(quality.warning());
        metrics.insert("view_quality".into(), serde_json::to_value(quality)?);
        metrics.insert("layout".into(), json!("ground_truth|render"));
        metrics.insert("views".into(), json!(scores));
        let stems: Vec<String> = scores.iter().map(|s| s.stem.clone()).collect();
        comparisons(&out.join("gt"), &out.join("renders"), &stems, &out.join("comparisons"))?
    };
    metrics.insert("view_count".into(), json!(images.len()));
    add_lock_wait(&out, &mut metrics);
    metrics.insert("depth_dir".into(), json!(rel(run, &out.join("depth"))));
    metrics.insert("alpha_dir".into(), json!(rel(run, &out.join("alpha"))));
    Ok(ToolResult {
        artifacts: vec![artifact],
        metrics: metrics.into_iter().collect(),
        images: images.iter().map(|p| rel(run, p)).collect(),
        warnings,
        wall_s: Some(start.elapsed().as_secs_f64()),
        ..result("render_views", Status::Succeeded)
    })
}

fn requested_views(req: &RenderViews, train: &[String], heldout: &[String]) -> anyhow::Result<Vec<String>> {
    anyhow::ensure!(!matches!(req.views, ViewSet::Orbit), "image_names cannot be used with orbit views");
    anyhow::ensure!(req.image_names.len() <= req.max_views as usize, "image_names exceeds max_views");
    let mut seen = std::collections::HashSet::new();
    for name in &req.image_names {
        anyhow::ensure!(train.contains(name) || heldout.contains(name), "unregistered image_names entry: {name}");
        anyhow::ensure!(seen.insert(name), "duplicate image_names entry: {name}");
    }
    Ok(req.image_names.clone())
}

#[cfg(test)]
mod qa_tests {
    use super::*;

    #[test]
    fn named_qa_keeps_requested_order_across_splits_and_rejects_unknown_views() {
        let mut req: RenderViews = serde_json::from_value(json!({"candidate_id":"c", "max_views":4,
            "image_names":["capture104.jpg", "capture100.jpg", "capture101.jpg", "capture103.jpg"]})).unwrap();
        let train = vec!["capture100.jpg".into(), "capture101.jpg".into(), "capture104.jpg".into()];
        let held = vec!["capture103.jpg".into()];
        assert_eq!(requested_views(&req, &train, &held).unwrap(), req.image_names);
        req.image_names.push("capture999.jpg".into());
        assert!(requested_views(&req, &train, &held).is_err());
        req.max_views = 8;
        assert!(requested_views(&req, &train, &held).is_err());
        req.image_names = vec![train[0].clone(), train[0].clone()];
        assert!(requested_views(&req, &train, &held).is_err());
        req.image_names.pop();
        req.views = ViewSet::Orbit;
        assert!(requested_views(&req, &train, &held).is_err());
    }
}
