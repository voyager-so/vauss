//! `__brush select` worker: segments the chosen views with `spirula sam track`, lifts the 2D
//! object masks onto the Gaussians by a gradient vote, and writes the selection plus before/after
//! strips (see [`crate::select`]).
//!
//! The vote: every splat gets a constant colour (SH degree 0, DC = 0, so colour 0.5 >= 0 and the
//! rasterizer's clamp passes gradients), and the loss sum(R * m) + sum(G * (1 - m)) on a black
//! background backprops dL/dDC_r = SH_C0 * sum_{p in m} alpha_i(p) T_i(p) = SH_C0 * in_i and
//! dL/dDC_g = SH_C0 * out_i. One render + backward per voting view.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, bail};
use brush_dataset::scene::SceneView;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::kernels::sh::SH_C0;
use brush_render::{TextureMode, bwd};
use burn::module::{Module, Param, ParamId};
use burn::tensor::{Int, Tensor, TensorData, s};
use glam::{UVec2, Vec3};
use image::imageops::{FilterType, resize};
use image::{GenericImage, GrayImage, Luma, Rgb, RgbImage};
use serde_json::json;

use crate::select::SelectJob;
use crate::util::*;
use crate::{engine, views};

const STRIP_VIEWS: usize = 4;
const STRIP_MAX_EDGE: u32 = 2048;

struct ViewData<'a> {
    name: String,
    stem: String,
    view: &'a SceneView,
    gt: RgbImage,
    /// Object mask at render size (true = object); None when SAM wrote no mask for the frame.
    mask: Option<Vec<bool>>,
    heldout: bool,
}

impl ViewData<'_> {
    fn size(&self) -> UVec2 {
        UVec2::new(self.gt.width(), self.gt.height())
    }
    fn mask_frac(&self) -> Option<f64> {
        self.mask.as_ref().map(|m| m.iter().filter(|&&b| b).count() as f64 / m.len().max(1) as f64)
    }
    fn has_object(&self) -> bool {
        self.mask.as_ref().is_some_and(|m| m.iter().any(|&b| b))
    }
}

/// Worker body: writes out/{masks,strips}/, sam.log, selected_ids.npy, object_ids.npy and
/// selection.json.
/// The frame copies and raw masks are deleted on exit either way (cloud runs sync out/ back).
pub fn run_job(job: SelectJob) -> anyhow::Result<i32> {
    let wait = Instant::now();
    let _gpu = crate::gpu_lock(job.gpu_lock.as_deref())?;
    let res = run(&job, wait.elapsed().as_secs_f64());
    for d in ["sam_frames", "raw_masks"] {
        let _ = std::fs::remove_dir_all(job.out.join(d));
    }
    res
}

fn run(job: &SelectJob, lock_wait: f64) -> anyhow::Result<i32> {
    let start = Instant::now();
    std::fs::create_dir_all(&job.out)?;
    let names: BTreeSet<String> = job.vote_views.iter().chain(&job.heldout_views).cloned().collect();
    if job.vote_views.is_empty() {
        bail!("select needs at least one vote view");
    }
    // SAM runs before this process opens the GPU, so the two never hold VRAM at once.
    eprintln!("[select] segmenting {} views for {:?}", names.len(), job.prompt);
    let sam_masks = segment(job, &names)?;
    let mut attached_job = job.clone();
    attached_job.prompt = "blanket; cloth; pillow".into();
    attached_job.out = job.out.join("attached");
    std::fs::create_dir_all(&attached_job.out)?;
    // Attachment evidence uses training views only.
    let attached_masks = segment(&attached_job, &job.vote_views.iter().cloned().collect())?;
    for d in ["sam_frames", "raw_masks"] {
        let _ = std::fs::remove_dir_all(attached_job.out.join(d));
    }
    // Other furniture, for keeping the volume hull off neighbouring objects (shelf beside a basket).
    let mut other_job = job.clone();
    other_job.prompt = other_objects_prompt(&job.prompt);
    other_job.out = job.out.join("other");
    std::fs::create_dir_all(&other_job.out)?;
    let other_masks = if other_job.prompt.is_empty() {
        HashMap::new()
    } else {
        segment(&other_job, &job.vote_views.iter().cloned().collect())?
    };
    for d in ["sam_frames", "raw_masks"] {
        let _ = std::fs::remove_dir_all(other_job.out.join(d));
    }
    let t_sam = start.elapsed().as_secs_f64();
    eprintln!("[select] sam: {} masks in {t_sam:.1}s", sam_masks.len());

    let rt = engine::runtime()?;
    rt.block_on(async {
        let t = Instant::now();
        let device = engine::init_device().await;
        let (splats, _) = engine::load_ply(&job.ply, SplatRenderMode::Default, &device).await?;
        if splats.decoder.is_some() {
            bail!("unsupported: neural appearance PLY (select_objects votes through SH colour)");
        }
        let n = splats.num_splats() as usize;
        let scene = engine::load_scene(&job.dataset, job.max_resolution).await?;
        let masks_dir = job.out.join("masks");
        std::fs::create_dir_all(&masks_dir)?;
        let heldout: BTreeSet<&String> = job.heldout_views.iter().collect();
        let mut views = vec![];
        for name in &names {
            let view = scene
                .views
                .iter()
                .find(|v| &engine::view_name(v) == name)
                .with_context(|| format!("view {name} not in dataset"))?;
            let gt = engine::load_gt(view).await?.to_rgb8();
            let stem = views::stem(name);
            let mask = sam_masks
                .get(name)
                .map(|m| if m.dimensions() == gt.dimensions() { m.clone() } else { resize(m, gt.width(), gt.height(), FilterType::Nearest) });
            if let Some(m) = &mask {
                m.save(masks_dir.join(format!("{stem}.png")))?;
            }
            let mask = mask.map(|m| m.as_raw().iter().map(|&v| v >= 128).collect());
            views.push(ViewData { name: name.clone(), stem, view, gt, mask, heldout: heldout.contains(name) });
        }
        let t_load = t.elapsed().as_secs_f64();
        eprintln!("[select] loaded {n} splats, {} views in {t_load:.1}s", views.len());

        // Vote.
        let t = Instant::now();
        let vote_set: BTreeSet<&String> = job.vote_views.iter().collect();
        let voting: Vec<&ViewData> = views.iter().filter(|v| vote_set.contains(&v.name) && v.mask.is_some()).collect();
        if voting.is_empty() {
            bail!("SAM wrote no mask for any vote view (see sam.log)");
        }
        let mut acc_in = Tensor::<1>::zeros([n], &device);
        let mut acc_out = Tensor::<1>::zeros([n], &device);
        for (i, v) in voting.iter().enumerate() {
            let size = v.size();
            let (w, h) = (size.x as usize, size.y as usize);
            // An exact forward render at this size first: the backward render sizes its
            // intersection buffer from the last count and would clamp a bigger view.
            let _ = brush_render::render_splats(splats.clone(), &v.view.camera, size, Vec3::ZERO, None, TextureMode::Float).await;
            let mut vote = splats.clone();
            vote.transforms = vote.transforms.set_require_grad(false);
            vote.raw_opacities = vote.raw_opacities.set_require_grad(false);
            vote.sh_coeffs = Param::initialized(ParamId::new(), Tensor::<3>::zeros([n, 1, 3], &device)).set_require_grad(true);
            let vote = vote.train();
            let out = bwd::render_splats(vote.clone(), &v.view.camera, size, Vec3::ZERO).await;
            let m: Vec<f32> = v.mask.as_ref().expect("voting views have masks").iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            let m = Tensor::<2>::from_data(TensorData::new(m, [h, w]), &vote.device());
            let r = out.img.clone().slice(s![.., .., 0..1]).reshape([h, w]);
            let g = out.img.slice(s![.., .., 1..2]).reshape([h, w]);
            let loss = r.mul(m.clone()).sum() + g.mul(m.neg().add_scalar(1.0)).sum();
            let mut grads = loss.backward();
            let grad = vote
                .sh_coeffs
                .val()
                .grad_remove(&mut grads)
                .context("vote render produced no SH gradient")?
                // Compact rows (0 = culled) -> one row per splat.
                .select(0, out.compact_from_global);
            acc_in = acc_in + grad.clone().slice(s![.., 0..1, 0..1]).reshape([n]);
            acc_out = acc_out + grad.slice(s![.., 0..1, 1..2]).reshape([n]);
            eprintln!("[select] vote {}/{} {}", i + 1, voting.len(), v.stem);
        }
        let vote_in: Vec<f32> = tensor_vec(acc_in).await?.into_iter().map(|x| x / SH_C0).collect();
        let vote_out: Vec<f32> = tensor_vec(acc_out).await?.into_iter().map(|x| x / SH_C0).collect();
        brush_render::render::settle_lagged_counts().await;
        let ratio: Vec<f32> = vote_in.iter().zip(&vote_out).map(|(&i, &o)| if i > 0.0 { i / (i + o.max(0.0)) } else { 0.0 }).collect();
        let means = tensor_vec(splats.means()).await?;
        let inside = hidden_inside(&means, &vote_in, &vote_out, &voting);
        let mut inst = instances(&means, &ratio, &vote_in, &vote_out, &inside, job.threshold, GROW_RATIO.min(job.threshold), job.largest_only);
        let old_ids = inst.ids.clone();
        let scales = tensor_vec(splats.scales()).await?;
        let opacity = tensor_vec(splats.opacities()).await?;
        let mut growth = grow_selection(&means, &scales, &opacity, &ratio, &inside, &mut inst);
        let attached_views: Vec<ViewData> = voting.iter().filter_map(|v| {
            attached_masks.get(&v.name).map(|m| {
                let m = resize(m, v.gt.width(), v.gt.height(), FilterType::Nearest);
                ViewData { name: v.name.clone(), stem: v.stem.clone(), view: v.view,
                    gt: v.gt.clone(), mask: Some(m.as_raw().iter().map(|&x| x >= 128).collect()), heldout: false }
            })
        }).collect();
        let attached_refs: Vec<&ViewData> = attached_views.iter().collect();
        let attached = hidden_inside(&means, &vote_in, &vote_out, &attached_refs);
        let other_views: Vec<ViewData> = voting.iter().filter_map(|v| {
            other_masks.get(&v.name).map(|m| {
                let m = resize(m, v.gt.width(), v.gt.height(), FilterType::Nearest);
                ViewData { name: v.name.clone(), stem: v.stem.clone(), view: v.view,
                    gt: v.gt.clone(), mask: Some(m.as_raw().iter().map(|&x| x >= 128).collect()), heldout: false }
            })
        }).collect();
        let other_refs: Vec<&ViewData> = other_views.iter().collect();
        let other = hidden_inside(&means, &vote_in, &vote_out, &other_refs);
        // COLMAP's world axes are arbitrary. Capture-camera up supplies the vertical axis.
        let up = voting.iter().map(|v| v.view.camera.rotation * -Vec3::Y).sum::<Vec3>().normalize_or_zero();
        growth["volume"] = prune_volume(&means, &scales, &opacity, &ratio, &attached, &inside, &other, up, &mut inst);
        let mut selected = vec![false; n];
        let mut object_of = vec![u32::MAX; n];
        for (&i, &o) in inst.ids.iter().zip(&inst.objects) {
            selected[i as usize] = true;
            object_of[i as usize] = o;
        }
        let ids: Vec<u32> = (0..n as u32).filter(|&i| selected[i as usize]).collect();
        let object_ids: Vec<u32> = ids.iter().map(|&i| object_of[i as usize]).collect();
        write_npy_u32(&job.out.join("selected_ids.npy"), &ids)?;
        write_npy_u32(&job.out.join("object_ids.npy"), &object_ids)?;
        let t_vote = t.elapsed().as_secs_f64();
        eprintln!(
            "[select] selected {} of {n} splats in {} objects (threshold {}, {} core, {} components, {} dropped as noise, voxel {:.4}) in {t_vote:.1}s",
            ids.len(), inst.stats.len(), job.threshold, inst.core, inst.components, inst.dropped, inst.voxel
        );

        // Eval: held-out IoU and strips.
        let t = Instant::now();
        let objects: Vec<serde_json::Value> = inst
            .stats
            .iter()
            .enumerate()
            .map(|(k, st)| {
                let members: Vec<u32> = ids.iter().zip(&object_ids).filter(|(_, o)| **o == k as u32).map(|(i, _)| *i).collect();
                let (lo, hi, c) = bbox(&means, &members);
                json!({"object_id": k, "n_gaussians": members.len(), "frac": members.len() as f64 / n.max(1) as f64,
                    "vote_share": st.vote_share, "bbox_min": lo, "bbox_max": hi, "centroid": c})
            })
            .collect();
        // Colour 1 for selected, 0 for the rest: the red channel over black is the selected
        // splats' share of each pixel's composite, occluders included.
        let dc: Vec<f32> = selected.iter().flat_map(|&s| [if s { 0.5 } else { -0.5 } / SH_C0; 3]).collect();
        let mut paint = splats.clone();
        paint.sh_coeffs = Param::initialized(ParamId::new(), Tensor::<3>::from_data(TensorData::new(dc, [n, 1, 3]), &device));
        let mut ious: HashMap<String, Option<f64>> = HashMap::new();
        let mut coverage: HashMap<String, f64> = HashMap::new();
        for v in views.iter().filter(|v| v.heldout) {
            let frame = engine::render(&paint, &v.view.camera, v.size(), false).await?;
            coverage.insert(v.name.clone(), frame.rgba.chunks_exact(4).filter(|px| px[0] > 0.5).count() as f64);
            let Some(mask) = &v.mask else { continue };
            let (mut inter, mut union) = (0usize, 0usize);
            for (px, &m) in frame.rgba.chunks_exact(4).zip(mask) {
                let r = px[0] > 0.5;
                inter += (r && m) as usize;
                union += (r || m) as usize;
            }
            ious.insert(v.name.clone(), (union > 0).then(|| inter as f64 / union as f64));
            eprintln!("[select] iou {}: {:?}", v.stem, ious[&v.name]);
        }
        let counted: Vec<f64> = views.iter().filter(|v| v.heldout && v.has_object()).filter_map(|v| ious.get(&v.name).copied().flatten()).collect();
        let iou_mean = (!counted.is_empty()).then(|| counted.iter().sum::<f64>() / counted.len() as f64);

        let strips_dir = job.out.join("strips");
        std::fs::create_dir_all(&strips_dir)?;
        let kept: Vec<u32> = (0..n as u32).filter(|&i| !selected[i as usize]).collect();
        let (object, rest) = (subset(&splats, &ids), subset(&splats, &kept));
        let mut old_selected = vec![false; n];
        for &i in &old_ids { old_selected[i as usize] = true; }
        let old_kept: Vec<u32> = (0..n as u32).filter(|&i| !old_selected[i as usize]).collect();
        let old_rest = subset(&splats, &old_kept);
        let growth_dir = job.out.join("growth_strips");
        std::fs::create_dir_all(&growth_dir)?;
        // Held-out views first, then vote views; views that show the object first within each.
        let mut order: Vec<&ViewData> = views.iter().collect();
        // Held-out views where the selection covers most pixels first (the edit shows best there).
        order.sort_by(|a, b| (!a.heldout, -coverage.get(&a.name).copied().unwrap_or(0.0)).partial_cmp(&(!b.heldout, -coverage.get(&b.name).copied().unwrap_or(0.0))).unwrap());
        let mut strips = vec![];
        for v in order.into_iter().take(STRIP_VIEWS) {
            let size = v.size();
            let mut tinted = v.gt.clone();
            if let Some(mask) = &v.mask {
                for (px, &m) in tinted.pixels_mut().zip(mask) {
                    if m {
                        *px = Rgb([0.4 * px[0] as f32 + 0.6 * 255.0, 0.4 * px[1] as f32, 0.4 * px[2] as f32].map(|c| c.round() as u8));
                    }
                }
            }
            let panels = [
                tinted,
                panel(Some(&splats), v.view, size, Vec3::ZERO).await?,
                panel(rest.as_ref(), v.view, size, Vec3::ZERO).await?,
                panel(object.as_ref(), v.view, size, Vec3::ONE).await?,
            ];
            let before = panel(old_rest.as_ref(), v.view, size, Vec3::ZERO).await?;
            let mut comparison = RgbImage::new(size.x * 2, size.y);
            comparison.copy_from(&before, 0, 0)?;
            comparison.copy_from(&panels[2], size.x, 0)?;
            comparison.save(growth_dir.join(format!("{}.jpg", v.stem)))?;
            let mut canvas = RgbImage::new(size.x * panels.len() as u32, size.y);
            for (k, p) in panels.iter().enumerate() {
                canvas.copy_from(p, k as u32 * size.x, 0)?;
            }
            let edge = canvas.width().max(canvas.height());
            if edge > STRIP_MAX_EDGE {
                let f = STRIP_MAX_EDGE as f64 / edge as f64;
                let (w, h) = (((canvas.width() as f64 * f).round() as u32).max(1), ((canvas.height() as f64 * f).round() as u32).max(1));
                canvas = resize(&canvas, w, h, FilterType::Triangle);
            }
            let rel = format!("strips/{}.jpg", v.stem);
            let file = std::io::BufWriter::new(std::fs::File::create(job.out.join(&rel))?);
            image::codecs::jpeg::JpegEncoder::new_with_quality(file, 90).encode_image(&canvas)?;
            eprintln!("[select] strip {rel}");
            strips.push(rel);
        }
        let t_eval = t.elapsed().as_secs_f64();

        let frac = ids.len() as f64 / n.max(1) as f64;
        let row = |v: &ViewData, iou: bool| {
            let mut r = json!({"name": v.name, "mask_frac": v.mask_frac()});
            if iou {
                r["iou"] = json!(ious.get(&v.name).copied().flatten());
            }
            r
        };
        let by_name: HashMap<&str, &ViewData> = views.iter().map(|v| (v.name.as_str(), v)).collect();
        let summary = json!({
            "prompt": job.prompt,
            "threshold": job.threshold,
            "n_gaussians": n,
            "n_selected": ids.len(),
            "frac_selected": frac,
            "instances": if job.largest_only { "largest" } else { "all" },
            "objects": objects,
            "growth": growth,
            "vote_views": job.vote_views.iter().map(|n| row(by_name[n.as_str()], false)).collect::<Vec<_>>(),
            "heldout_views": job.heldout_views.iter().map(|n| row(by_name[n.as_str()], true)).collect::<Vec<_>>(),
            "views_with_object": views.iter().filter(|v| v.has_object()).count(),
            "heldout_iou_mean": iou_mean,
            "strips": strips,
            "timings_s": {
                "gpu_lock_wait": lock_wait,
                "sam": t_sam,
                "load": t_load,
                "vote": t_vote,
                "eval": t_eval,
                "total": start.elapsed().as_secs_f64(),
            },
        });
        write_json(&job.out.join("selection.json"), &summary)?;
        eprintln!("[select] done: {} splats selected, held-out IoU {:?}", ids.len(), iou_mean);
        anyhow::Ok(0)
    })
}

/// Link the views into one flat dir in name (capture) order, run `spirula sam track` on it and
/// return each view's object mask (255 = object) at the source image size. Views SAM wrote no
/// mask for are absent.
fn segment(job: &SelectJob, names: &BTreeSet<String>) -> anyhow::Result<HashMap<String, GrayImage>> {
    let frames = job.out.join("sam_frames");
    let raw = job.out.join("raw_masks");
    for d in [&frames, &raw] {
        if d.exists() {
            std::fs::remove_dir_all(d)?;
        }
    }
    std::fs::create_dir_all(&frames)?;
    let mut flat_to_name = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        let src = job.dataset.join("images").join(name);
        if !src.is_file() {
            bail!("view image {} not found", src.display());
        }
        let flat = format!("{i:05}_{}", name.replace(['/', '\\'], "__"));
        link_or_copy(&src, &frames.join(&flat))?;
        flat_to_name.insert(flat, name.clone());
    }
    let out = Command::new(&job.sam.binary)
        .args(["sam", "track", "--model"])
        .arg(&job.sam.model)
        .arg("--detector")
        .arg(&job.sam.detector)
        .arg("--frames")
        .arg(&frames)
        .args(["--text", &job.prompt, "--remove-prompted", "--out"])
        .arg(&raw)
        .args(["--device", "0"])
        .output()
        .with_context(|| format!("spawning {}", job.sam.binary.display()))?;
    let log_path = job.out.join("sam.log");
    std::fs::write(&log_path, [&out.stdout[..], &out.stderr[..]].concat())?;
    if !out.status.success() {
        bail!("sam track exited {}: {}", out.status, tail_file(&log_path, 1500));
    }
    let log = std::fs::read_to_string(&log_path).unwrap_or_default();
    let index = kept_frames(&log);
    let mut masks = HashMap::new();
    let mut unmapped = 0;
    for p in list_images(&raw).unwrap_or_default() {
        let Some(id) = p.file_stem().and_then(|s| s.to_str()).and_then(|s| s.strip_prefix("frame_")).and_then(|s| s.parse::<u64>().ok()) else { continue };
        let name = index
            .get(&id)
            .and_then(|f| Path::new(f).file_name())
            .and_then(|f| flat_to_name.get(f.to_string_lossy().as_ref()));
        let Some(name) = name else {
            unmapped += 1;
            continue;
        };
        // Raw mask: black (< 128) = the prompted object.
        let gray = image::open(&p)?.to_luma8();
        let (w, h) = gray.dimensions();
        let obj = GrayImage::from_fn(w, h, |x, y| Luma([if gray.get_pixel(x, y)[0] < 128 { 255 } else { 0 }]));
        masks.insert(name.clone(), obj);
    }
    if masks.is_empty() && unmapped > 0 {
        bail!("sam.log does not map any of {unmapped} mask frames to an input view");
    }
    let missing: Vec<&String> = names.iter().filter(|n| !masks.contains_key(*n)).collect();
    if !missing.is_empty() {
        eprintln!("[select] warning: no SAM mask for {} views: {missing:?}", missing.len());
    }
    Ok(masks)
}

/// `frame N: <name> -- kept` lines of the tracker log: frame index -> input file name.
pub(crate) fn kept_frames(log: &str) -> BTreeMap<u64, String> {
    let mut map = BTreeMap::new();
    for line in log.lines() {
        let Some(end) = line.find(" -- kept") else { continue };
        let head = &line[..end];
        for (pos, _) in head.match_indices("frame") {
            let after = &head[pos + "frame".len()..];
            if !after.starts_with(char::is_whitespace) {
                continue;
            }
            let after = after.trim_start();
            let digits = after.len() - after.trim_start_matches(|c: char| c.is_ascii_digit()).len();
            let Some(name) = after[digits..].strip_prefix(':').map(str::trim) else { continue };
            if let (Ok(id), false) = (after[..digits].parse::<u64>(), name.is_empty()) {
                map.insert(id, name.to_string());
                break;
            }
        }
    }
    map
}

/// The splats at `ids` (None when empty).
fn subset(splats: &Splats, ids: &[u32]) -> Option<Splats> {
    if ids.is_empty() {
        return None;
    }
    if ids.len() == splats.num_splats() as usize {
        return Some(splats.clone());
    }
    let idx = Tensor::<1, Int>::from_data(TensorData::new(ids.iter().map(|&i| i as i32).collect::<Vec<_>>(), [ids.len()]), &splats.device());
    let mut s = splats.clone();
    s.transforms = s.transforms.map(|t| t.select(0, idx.clone()));
    s.sh_coeffs = s.sh_coeffs.map(|c| c.select(0, idx.clone()));
    s.raw_opacities = s.raw_opacities.map(|o| o.select(0, idx.clone()));
    s.min_scale = s.min_scale.map(|f| f.select(0, idx.clone()));
    Some(s)
}

/// Render `splats` over `bg`; no splats = the plain background.
async fn panel(splats: Option<&Splats>, view: &SceneView, size: UVec2, bg: Vec3) -> anyhow::Result<RgbImage> {
    Ok(match splats {
        Some(s) => engine::to_rgb8(&engine::render_bg(s, &view.camera, size, false, bg).await?),
        None => RgbImage::from_pixel(size.x, size.y, Rgb(bg.to_array().map(|c| (c * 255.0).round() as u8))),
    })
}

async fn tensor_vec<const D: usize>(t: Tensor<D>) -> anyhow::Result<Vec<f32>> {
    t.into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))
}

/// NumPy .npy v1.0: 1-D little-endian u32.
fn write_npy_u32(path: &Path, ids: &[u32]) -> anyhow::Result<()> {
    let mut header = format!("{{'descr': '<u4', 'fortran_order': False, 'shape': ({},), }}", ids.len());
    // magic(6) + version(2) + header_len(2) + header + '\n', padded to a multiple of 64.
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    bytes.extend(ids.iter().flat_map(|i| i.to_le_bytes()));
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Near neighbours of a kept object join it at this lower vote ratio (legs, thin parts, contact
/// shadow), one voxel ring around the object only.
const GROW_RATIO: f32 = 0.3;
/// Components with less than this share of the selected vote mass are noise (stray splats).
const NOISE_SHARE: f64 = 0.02;
/// A hidden splat belongs to an object when its centre projects inside the object masks in at
/// least this share of the voting views that see it in frame.
const HIDDEN_INSIDE: f32 = 0.8;
/// Lowest vote ratio of a seen splat that can still join by mask projection.
const INTERIOR_RATIO: f32 = 0.1;

/// For splats no voting view gave any weight (hidden), the share of the voting views with the
/// centre in frame and in front of the camera whose object mask contains the projected centre
/// (pinhole projection; lens distortion is ignored); -1 when never in frame.
fn hidden_inside(means: &[f32], vote_in: &[f32], vote_out: &[f32], views: &[&ViewData]) -> Vec<f32> {
    let cams: Vec<_> = views
        .iter()
        .map(|v| {
            let size = v.size();
            (v.view.camera.world_to_local(), v.view.camera.focal(size), v.view.camera.center(size), size, v.mask.as_ref().expect("voting views have masks"))
        })
        .collect();
    (0..vote_in.len())
        .map(|i| {
            let _ = (vote_in, vote_out);
            let q = Vec3::from_slice(&means[i * 3..i * 3 + 3]);
            let (mut seen, mut hit) = (0u32, 0u32);
            for (w2c, f, c, size, mask) in &cams {
                let l = w2c.transform_point3(q);
                if l.z <= 1e-4 {
                    continue;
                }
                let (u, v) = (f.x * l.x / l.z + c.x, f.y * l.y / l.z + c.y);
                if u < 0.0 || v < 0.0 || u >= size.x as f32 || v >= size.y as f32 {
                    continue;
                }
                seen += 1;
                hit += mask[v as usize * size.x as usize + u as usize] as u32;
            }
            if seen == 0 { -1.0 } else { hit as f32 / seen as f32 }
        })
        .collect()
}

struct ObjectStat {
    vote_share: f64,
}

struct Instances {
    ids: Vec<u32>,
    objects: Vec<u32>,
    stats: Vec<ObjectStat>,
    core: usize,
    components: usize,
    dropped: usize,
    voxel: f32,
}

fn bbox(means: &[f32], ids: &[u32]) -> (Option<[f32; 3]>, Option<[f32; 3]>, Option<[f32; 3]>) {
    if ids.is_empty() {
        return (None, None, None);
    }
    let (mut lo, mut hi, mut sum) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY), glam::DVec3::ZERO);
    for &i in ids {
        let p = Vec3::from_slice(&means[i as usize * 3..i as usize * 3 + 3]);
        (lo, hi, sum) = (lo.min(p), hi.max(p), sum + p.as_dvec3());
    }
    (Some(lo.to_array()), Some(hi.to_array()), Some((sum / ids.len() as f64).as_vec3().to_array()))
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Splits the voted splats (ratio > threshold) into objects: 3D connected components of their
/// centres on a voxel grid (voxel = 3x the median nearest-neighbour spacing of the voted splats, so
/// it follows the scene's arbitrary SfM scale), ranked by vote mass; components under NOISE_SHARE
/// of the mass are dropped. Then each kept object takes the splats with ratio > grow in its voxels
/// and the ring around them.
fn instances(means: &[f32], ratio: &[f32], vote_in: &[f32], vote_out: &[f32], inside: &[f32], threshold: f32, grow: f32, largest_only: bool) -> Instances {
    let p = |i: usize| Vec3::from_slice(&means[i * 3..i * 3 + 3]);
    let core: Vec<usize> = (0..ratio.len()).filter(|&i| ratio[i] > threshold && vote_in[i] > 0.0).collect();
    let empty = |core: usize| Instances { ids: vec![], objects: vec![], stats: vec![], core, components: 0, dropped: 0, voxel: 0.0 };
    if core.is_empty() {
        return empty(0);
    }
    let reference: Vec<Vec3> = core.iter().step_by(core.len().div_ceil(50_000)).map(|&i| p(i)).collect();
    let mut nn: Vec<f32> = core
        .iter()
        .step_by(core.len().div_ceil(2_000))
        .map(|&i| {
            let q = p(i);
            reference.iter().map(|r| r.distance_squared(q)).filter(|&d| d > 0.0).fold(f32::INFINITY, f32::min).sqrt()
        })
        .filter(|d| d.is_finite())
        .collect();
    nn.sort_by(f32::total_cmp);
    let voxel = (3.0 * nn.get(nn.len() / 2).copied().unwrap_or(0.0)).max(1e-6);
    let key = |q: Vec3| ((q.x / voxel).floor() as i64, (q.y / voxel).floor() as i64, (q.z / voxel).floor() as i64);
    let mut cells: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut cell_of = Vec::with_capacity(core.len());
    for &i in &core {
        let k = key(p(i));
        let next = cells.len();
        cell_of.push(*cells.entry(k).or_insert(next));
    }
    let mut parent: Vec<usize> = (0..cells.len()).collect();
    let neighbours = |k: (i64, i64, i64)| (-1..=1).flat_map(move |a| (-1..=1).flat_map(move |b| (-1..=1).map(move |c| (k.0 + a, k.1 + b, k.2 + c))));
    for (&k, &c) in &cells {
        for nk in neighbours(k) {
            if let Some(&d) = cells.get(&nk) {
                let (ra, rb) = (find(&mut parent, c), find(&mut parent, d));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut mass: HashMap<usize, f64> = HashMap::new();
    for (j, &i) in core.iter().enumerate() {
        let root = find(&mut parent, cell_of[j]);
        *mass.entry(root).or_default() += vote_in[i] as f64;
    }
    let total: f64 = mass.values().sum();
    let mut ranked: Vec<(usize, f64)> = mass.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let components = ranked.len();
    let mut kept: Vec<(usize, f64)> = ranked.iter().copied().filter(|&(_, m)| m >= NOISE_SHARE * total).collect();
    if kept.is_empty() {
        kept.push(ranked[0]);
    }
    if largest_only {
        kept.truncate(1);
    }
    let rank: HashMap<usize, u32> = kept.iter().enumerate().map(|(r, &(root, _))| (root, r as u32)).collect();
    // Voxel -> object for the kept objects' cells, then the grow pass over every candidate.
    let mut object_cell: HashMap<(i64, i64, i64), u32> = HashMap::new();
    for (&k, &c) in &cells {
        if let Some(&o) = rank.get(&find(&mut parent, c)) {
            object_cell.insert(k, o);
        }
    }
    // Each kept object's box (its core splats, padded by two voxels) bounds its hidden interior.
    let mut boxes = vec![(Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)); kept.len()];
    for (j, &i) in core.iter().enumerate() {
        if let Some(&o) = rank.get(&find(&mut parent, cell_of[j])) {
            let b = &mut boxes[o as usize];
            *b = (b.0.min(p(i)), b.1.max(p(i)));
        }
    }
    let pad = Vec3::splat(2.0 * voxel);
    // Splats with almost no weight in any voting view (under 1% of the core's median) are as good
    // as hidden: their ratio is noise.
    let mut weights: Vec<f32> = core.iter().map(|&i| vote_in[i] + vote_out[i]).collect();
    weights.sort_by(f32::total_cmp);
    let faint = 0.01 * weights.get(weights.len() / 2).copied().unwrap_or(0.0);
    let (mut ids, mut objects) = (vec![], vec![]);
    for i in 0..ratio.len() {
        let hidden = vote_in[i] + vote_out[i] <= faint;
        let q = p(i);
        let k = key(q);
        let object = if !hidden && ratio[i] > threshold {
            object_cell.get(&k).copied()
        } else if !hidden && ratio[i] > grow && vote_in[i] > 0.0 {
            // Near neighbours at a lower ratio (legs, thin parts, faint haze around the object): the
            // object's cells and their ring, else anywhere in its padded box.
            neighbours(k)
                .find_map(|nk| object_cell.get(&nk).copied())
                .or_else(|| boxes.iter().position(|b| q.cmpge(b.0 - pad).all() && q.cmple(b.1 + pad).all()).map(|o| o as u32))
        } else {
            None
        };
        // Interior and occluded splats (little or no weight in the composites) can't be voted in:
        // centres the object masks cover from most views, inside the object's padded box, join
        // unless clearly seen outside the masks.
        let object = object.or_else(|| {
            ((hidden || ratio[i] >= INTERIOR_RATIO) && inside[i] >= HIDDEN_INSIDE)
                .then(|| boxes.iter().position(|b| q.cmpge(b.0 - pad).all() && q.cmple(b.1 + pad).all()).map(|o| o as u32))
                .flatten()
        });
        if let Some(o) = object {
            ids.push(i as u32);
            objects.push(o);
        }
    }
    let stats = kept.iter().map(|&(_, m)| ObjectStat { vote_share: m / total.max(f64::MIN_POSITIVE) }).collect();
    Instances { ids, objects, stats, core: core.len(), components, dropped: ranked.len() - kept.len(), voxel }
}

/// A single dilation of the frozen selection's occupied volume. Newly accepted splats never
/// become seeds: otherwise weak votes can flood across a floor. Radius follows each seed's
/// rendered scale, capped relative to the object's size. Broad opaque splats are protected.
fn grow_selection(means: &[f32], scales: &[f32], opacity: &[f32], ratio: &[f32], inside: &[f32], inst: &mut Instances) -> serde_json::Value {
    if inst.ids.is_empty() {
        return json!({"added": 0, "protected_surfaces": 0});
    }
    let p = |i: usize| Vec3::from_slice(&means[3 * i..3 * i + 3]);
    let scale = |i: usize| Vec3::from_slice(&scales[3 * i..3 * i + 3]);
    let mut medians = vec![Vec::new(); inst.stats.len()];
    let mut bounds = vec![(Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)); inst.stats.len()];
    for (&i, &o) in inst.ids.iter().zip(&inst.objects) {
        let (i, o) = (i as usize, o as usize);
        let s = scale(i).max_element();
        if s.is_finite() && s > 0.0 {
            medians[o].push(s);
        }
        bounds[o] = (bounds[o].0.min(p(i)), bounds[o].1.max(p(i)));
    }
    let typical: Vec<f32> = medians.iter_mut().map(|v| {
        v.sort_by(f32::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(inst.voxel).max(1e-6)
    }).collect();
    let caps: Vec<f32> = bounds.iter().map(|(lo, hi)| ((hi - lo).max_element() * 0.06).max(1e-6)).collect();
    let cell = caps.iter().copied().fold(0.0f32, f32::max);
    let key = |q: Vec3| ((q.x / cell).floor() as i64, (q.y / cell).floor() as i64, (q.z / cell).floor() as i64);
    let mut occupied: HashMap<(i64, i64, i64), Vec<(Vec3, f32, u32)>> = HashMap::new();
    let mut selected = vec![false; ratio.len()];
    for (&i, &o) in inst.ids.iter().zip(&inst.objects) {
        let i = i as usize;
        selected[i] = true;
        let radius = (2.5 * scale(i).max_element()).min(caps[o as usize]);
        if radius.is_finite() && radius > 0.0 {
            occupied.entry(key(p(i))).or_default().push((p(i), radius * radius, o));
        }
    }
    let (mut added_vote, mut added_faint, mut protected) = (0usize, 0usize, 0usize);
    for i in 0..ratio.len() {
        if selected[i] || !(ratio[i] >= 0.08 || (opacity[i] < 0.1 && inside[i] >= 0.5)) {
            continue;
        }
        let q = p(i);
        let k = key(q);
        let mut nearest = (f32::INFINITY, u32::MAX);
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if let Some(seeds) = occupied.get(&(k.0 + x, k.1 + y, k.2 + z)) {
                        for &(centre, radius2, o) in seeds {
                            let d = q.distance_squared(centre);
                            if d <= radius2 && d < nearest.0 {
                                nearest = (d, o);
                            }
                        }
                    }
                }
            }
        }
        if nearest.1 == u32::MAX {
            continue;
        }
        let o = nearest.1 as usize;
        let s = scale(i);
        // Large high-opacity splats often span structural surfaces. Protect these even when a
        // chair mask happens to cover their centres. Also reject enormous faint splats.
        if !s.is_finite() || (opacity[i] >= 0.1 && s.max_element() > 3.0 * typical[o]) || s.max_element() > caps[o] {
            protected += 1;
            continue;
        }
        inst.ids.push(i as u32);
        inst.objects.push(o as u32);
        if ratio[i] >= 0.08 { added_vote += 1; } else { added_faint += 1; }
    }
    eprintln!("[select] growth: {} voted + {} faint, {} broad surfaces protected", added_vote, added_faint, protected);
    json!({"added": added_vote + added_faint, "added_vote": added_vote, "added_faint": added_faint,
        "protected_surfaces": protected, "opaque_large_added": 0, "radius_caps": caps, "median_scale": typical})
}


/// The volume hull skips a splat whose centre projects onto another detected object in at least
/// this share of the voting views that frame it and onto the selected object (or attached cloth) in
/// less: a neighbour inside the loose hull (shelf edge beside a basket, plant beside a sofa).
/// Interior splats project inside the object masks from nearly every view (projection ignores
/// occlusion), so they still join.
const VOLUME_INSIDE: f32 = 0.5;

/// Furniture SAM looks for to keep the volume hull off neighbours; nouns that overlap the selection
/// prompt are dropped (removing "the bookshelf" must not block the bookshelf).
const OTHER_OBJECTS: [&str; 10] = ["shelf", "bookcase", "cabinet", "table", "sofa", "chair", "plant", "lamp", "door", "television"];

fn other_objects_prompt(prompt: &str) -> String {
    let prompt = prompt.to_lowercase();
    let words: Vec<&str> = prompt.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() >= 4).collect();
    OTHER_OBJECTS
        .iter()
        .filter(|n| !prompt.contains(*n) && !words.iter().any(|w| n.contains(w)))
        .copied()
        .collect::<Vec<_>>()
        .join("; ")
}

/// Close frozen occupancy, fill enclosed holes, then expand one voxel. All geometry is in a
/// camera-up frame; the attachment column stops at 110% of the original selected height.
fn prune_volume(means: &[f32], scales: &[f32], opacity: &[f32], ratio: &[f32], attached: &[f32], inside: &[f32], other: &[f32], up: Vec3, inst: &mut Instances) -> serde_json::Value {
    type Cell = (i32, i32, i32);
    let up = if up.length_squared() > 0.5 { up } else { Vec3::Y };
    let right = up.any_orthonormal_vector();
    let forward = right.cross(up);
    let point = |i: usize| {
        let p = Vec3::from_slice(&means[3*i..3*i+3]);
        Vec3::new(p.dot(right), p.dot(up), p.dot(forward))
    };
    let neighbours = |k: Cell| [(k.0-1,k.1,k.2),(k.0+1,k.1,k.2),(k.0,k.1-1,k.2),(k.0,k.1+1,k.2),(k.0,k.1,k.2-1),(k.0,k.1,k.2+1)];
    let dilate = |cells: &HashSet<Cell>| -> HashSet<Cell> {
        cells.iter().flat_map(|&(x,y,z)| (-1..=1).flat_map(move |a| (-1..=1).flat_map(move |b| (-1..=1).map(move |c| (x+a,y+b,z+c))))).collect()
    };
    let mut selected: HashSet<u32> = inst.ids.iter().copied().collect();
    let mut total_hull = 0; let mut total_attached = 0; let mut protected = 0; let mut other_object = 0;
    let mut objects = vec![];
    for o in 0..inst.stats.len() {
        let seeds: Vec<usize> = inst.ids.iter().zip(&inst.objects).filter(|(_, obj)| **obj == o as u32).map(|(&i,_)| i as usize).collect();
        if seeds.is_empty() { continue; }
        let mut lo = Vec3::splat(f32::INFINITY); let mut hi = -lo;
        let mut sizes = vec![];
        for &i in &seeds {
            lo = lo.min(point(i)); hi = hi.max(point(i));
            sizes.push(Vec3::from_slice(&scales[3*i..3*i+3]).max_element());
        }
        sizes.sort_by(f32::total_cmp);
        let typical = sizes[sizes.len()/2];
        let extent = (hi-lo).max_element();
        let cell = inst.voxel.max(extent / 48.0).max(1e-6);
        let key = |p: Vec3| { let v = ((p-lo)/cell).floor(); (v.x as i32,v.y as i32,v.z as i32) };
        let occupied: HashSet<Cell> = seeds.iter().map(|&i| key(point(i))).collect();
        let mut closed = dilate(&dilate(&occupied));
        for _ in 0..2 {
            closed = closed.iter().copied().filter(|&(x,y,z)| (-1..=1).all(|a| (-1..=1).all(|b| (-1..=1).all(|c| closed.contains(&(x+a,y+b,z+c)))))).collect();
        }
        let max = key(hi);
        let inside_grid = |k: Cell| k.0 >= -3 && k.1 >= -3 && k.2 >= -3 && k.0 <= max.0+3 && k.1 <= max.1+3 && k.2 <= max.2+3;
        let mut outside = HashSet::from([(-3,-3,-3)]);
        let mut queue = VecDeque::from([(-3,-3,-3)]);
        while let Some(k) = queue.pop_front() {
            for n in neighbours(k) {
                if inside_grid(n) && !closed.contains(&n) && outside.insert(n) { queue.push_back(n); }
            }
        }
        for x in 0..=max.0 { for y in 0..=max.1 { for z in 0..=max.2 {
            if !outside.contains(&(x,y,z)) { closed.insert((x,y,z)); }
        } } }
        let hull = dilate(&closed);
        // A chair is an open shell: closing small gaps does not enclose the seat cavity.
        // Support planes through frozen seeds fill that cavity without using a loose AABB.
        let mut planes = vec![];
        for x in -1..=1 { for y in -1..=1 { for z in -1..=1 {
            if x == 0 && y == 0 && z == 0 { continue; }
            let normal = Vec3::new(x as f32, y as f32, z as f32).normalize();
            let support = seeds.iter().map(|&i| point(i).dot(normal)).fold(f32::NEG_INFINITY, f32::max);
            planes.push((normal, support + cell));
        } } }
        let footprint: HashSet<(i32,i32)> = hull.iter().map(|&(x,_,z)| (x,z)).collect();
        let (mut nh, mut na) = (0,0);
        for i in 0..ratio.len() {
            if selected.contains(&(i as u32)) { continue; }
            let p = point(i); let k = key(p);
            let in_hull = p.y >= lo.y && (hull.contains(&k) || planes.iter().all(|(normal, support)| p.dot(*normal) <= *support));
            let is_attachment = p.y >= lo.y && p.y <= hi.y + (hi.y-lo.y)*0.1
                && footprint.contains(&(k.0,k.2)) && (attached[i] >= 0.5 || ratio[i] > 0.15);
            if !in_hull && !is_attachment { continue; }
            if other[i] >= VOLUME_INSIDE && inside[i] < VOLUME_INSIDE && attached[i] < VOLUME_INSIDE { other_object += 1; continue; }
            let scale = Vec3::from_slice(&scales[3*i..3*i+3]);
            let flat = scale.min_element() < scale.max_element() * 0.25;
            if !scale.is_finite() || (opacity[i] >= 0.1 && flat && scale.max_element() > 3.0*typical) || scale.max_element() > extent*0.5 {
                protected += 1; continue;
            }
            selected.insert(i as u32); inst.ids.push(i as u32); inst.objects.push(o as u32);
            if in_hull { nh += 1; } else { na += 1; }
        }
        total_hull += nh; total_attached += na;
        objects.push(json!({"object":o,"voxel":cell,"hull_added":nh,"attached_added":na,"up":up.to_array(),"bbox_min":lo.to_array(),"bbox_max":hi.to_array()}));
    }
    eprintln!("[select] volume: {total_hull} hull + {total_attached} attachments; {protected} broad surfaces protected; {other_object} on other objects skipped");
    json!({"hull_added":total_hull,"attached_added":total_attached,"protected_surfaces":protected,"other_object_skipped":other_object,"objects":objects})
}
