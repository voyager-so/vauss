//! Warm start of a capped model (the interactive package's refine) from a larger candidate: which
//! splats the refine starts from. Default: half the cap by rendered contribution over the training
//! views, half an even stride over the rest. 1M from ~4.7M + the 20k refine, held-out vs the full
//! model (strided start + 10k before): driveway -0.73 (-1.22), garden -0.16 (-0.26), kitchen +0.21
//! (+0.05), yagan -0.30 (-0.59) dB. Contribution alone lost on garden (-0.50 at 20k): it drops the
//! many small detail splats. Merging dropped splats into kept ones instead of pruning matched
//! pruning once refined, and lost 17 dB unrefined.

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use brush_dataset::scene::SceneView;
use brush_render::bwd::render_splats;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_serde::import::SplatData;
use burn::prelude::Module;
use burn::tensor::{Device, Tensor, s};
use glam::Vec3;
use serde_json::{Value, json};

/// How a warm start picks `cap` splats from a larger PLY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Select {
    /// Every ceil(n/cap)-th splat in file order.
    Strided,
    /// Half the cap by contribution, the rest strided over the remaining splats.
    Hybrid,
}

/// Load `path` capped to `cap` splats with `select` (strided is `engine::load_capped_ply`).
pub async fn load(path: &Path, cap: usize, select: Select, untagged: SplatRenderMode, views: &[SceneView],
    max_resolution: u32, device: &Device) -> anyhow::Result<(Splats, Value)> {
    let t0 = Instant::now();
    let msg = brush_serde::load_splat_file(path, None).await.with_context(|| format!("loading {}", path.display()))?;
    let mode = msg.meta.render_mode.unwrap_or(untagged);
    let data = msg.data;
    let n = data.num_splats();
    // Neural appearance has no SH to score with: strided.
    if select == Select::Strided || cap == 0 || n <= cap || views.is_empty() || data.neural.is_some() {
        let data = data.subsample(cap);
        let kept = data.num_splats();
        return Ok((data.into_splats(device, mode), json!({"select": "strided", "from": n, "kept": kept})));
    }
    let score = contributions(data.clone().into_splats(device, mode), views, max_resolution, device).await?;
    let score_s = t0.elapsed().as_secs_f64();
    let top = cap / 2;
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.select_nth_unstable_by(top - 1, |&a, &b| score[b as usize].total_cmp(&score[a as usize]));
    let mut keep = vec![false; n];
    for &i in &order[..top] {
        keep[i as usize] = true;
    }
    if top < cap {
        // The rest: an even stride over the splats not yet kept, in file order.
        let rest: Vec<usize> = (0..n).filter(|&i| !keep[i]).collect();
        let step = rest.len() as f64 / (cap - top) as f64;
        for k in 0..cap - top {
            keep[rest[(k as f64 * step) as usize]] = true;
        }
    }
    let unseen = score.iter().filter(|&&s| s <= 0.0).count();
    let data = retain(&data, &keep);
    let stats = json!({"select": format!("{select:?}").to_lowercase(), "from": n, "kept": data.num_splats(),
        "unseen": unseen, "views": views.len(), "score_s": score_s, "total_s": t0.elapsed().as_secs_f64()});
    eprintln!("init: warm start {stats}");
    Ok((data.into_splats(device, mode), stats))
}

/// Per splat, its blend weight (alpha x transmittance) summed over every pixel of every view.
/// Rendered with all SH zero (colour 0.5, never clamped), the gradient of the summed image with
/// respect to a splat's DC coefficient is SH_C0 x that sum, per channel.
async fn contributions(splats: Splats, views: &[SceneView], max_resolution: u32, device: &Device) -> anyhow::Result<Vec<f32>> {
    let n = splats.num_splats() as usize;
    let mut acc: Tensor<1> = Tensor::zeros([n], device);
    for view in views {
        let (w, h) = view.image.dimensions().await?;
        let long = w.max(h).max(1) as f32;
        let scale = if max_resolution > 0 && long > max_resolution as f32 { max_resolution as f32 / long } else { 1.0 };
        let size = glam::uvec2(((w as f32 * scale).round() as u32).max(1), ((h as f32 * scale).round() as u32).max(1));
        let mut s: Splats = splats.clone().train();
        s.sh_coeffs = s.sh_coeffs.map(|c: Tensor<3>| c.zeros_like().require_grad());
        let out = render_splats(s.clone(), &view.camera, size, Vec3::ZERO).await;
        let loss = out.img.slice(s![.., .., 0..3]).sum();
        let mut grads = loss.backward();
        let g = s.sh_coeffs.val().grad_remove(&mut grads).context("no SH gradient")?
            .select(0, out.compact_from_global.clone())
            .slice(s![.., 0..1, ..])
            .sum_dim(2)
            .reshape([n]);
        acc = acc + g;
    }
    Ok(acc.into_data_async().await.map_err(|e| anyhow::anyhow!("{e:?}"))?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?)
}

/// The `keep` rows of `d`, in order.
fn retain(d: &SplatData, keep: &[bool]) -> SplatData {
    let n = d.num_splats();
    let pick = |v: &[f32]| -> Vec<f32> {
        let stride = v.len() / n;
        v.chunks_exact(stride).zip(keep).filter(|(_, k)| **k).flat_map(|(c, _)| c.iter().copied()).collect()
    };
    SplatData {
        means: pick(&d.means),
        rotations: d.rotations.as_deref().map(pick),
        log_scales: d.log_scales.as_deref().map(pick),
        sh_coeffs: d.sh_coeffs.as_deref().map(pick),
        raw_opacities: d.raw_opacities.as_deref().map(pick),
        neural: None,
    }
}
