//! `select_objects`: per-Gaussian object selection from a noun prompt on a trained candidate.
//!
//! The tool picks evenly spaced training views plus a few held-out views and starts a GPU worker
//! (`__brush select`, [`crate::select_worker`]) that segments the views with the SAM 2.1 +
//! Grounding DINO helper (`spirula sam track`, the same one mask_distractors uses), then votes per
//! Gaussian: w_i(p) = alpha_i(p) * T_i(p), the weight Gaussian i has in pixel p's composite,
//! summed inside (in_i) and outside (out_i) the object masks of the voting views. Selected:
//! in_i / (in_i + out_i) > threshold and in_i > 0. Clean-room from the training-free 2D-mask
//! lifting idea (gradient-weighted back-projection, arXiv 2411.15193); no third-party code.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// The SAM 2.1 + Grounding DINO segmenter: the Spirula executable (GPL, separate process) and its
/// weights (both Apache-2.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamTool {
    pub binary: PathBuf,
    pub model: PathBuf,
    pub detector: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectJob {
    pub ply: PathBuf,
    /// SfM dataset dir (images/ + sparse/N), loaded with Brush's loader at `max_resolution`.
    pub dataset: PathBuf,
    pub max_resolution: u32,
    /// Training views (image names) that are segmented and vote.
    pub vote_views: Vec<String>,
    /// Held-out views (image names): segmented for the IoU check and rendered as before/after
    /// strips; they never vote.
    pub heldout_views: Vec<String>,
    /// Detector prompt (short nouns separated by semicolons).
    pub prompt: String,
    pub threshold: f32,
    /// Keep only the 3D component with the most votes.
    #[serde(default)]
    pub largest_only: bool,
    pub sam: SamTool,
    /// The selection artifact dir; the worker writes everything here.
    pub out: PathBuf,
    pub gpu_lock: Option<PathBuf>,
}

use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;

use crate::util::*;
use crate::BrushConfig;

/// Held-out views segmented for the IoU check and rendered as before/after strips.
const HELDOUT_VIEWS: usize = 24;

/// `n` evenly spaced items of `names` (all of them when there are fewer).
fn spread(names: &[String], n: usize) -> Vec<String> {
    if names.len() <= n {
        return names.to_vec();
    }
    (0..n).map(|i| names[(i * (names.len() - 1) + (n - 1) / 2) / (n - 1).max(1)].clone()).collect()
}

pub fn select_objects(cfg: &BrushConfig, run: &dyn RunCtx, req: SelectObjects) -> anyhow::Result<ToolResult> {
    let Some(sam) = cfg.sam.clone() else {
        return Ok(unsupported("select_objects", "no segmenter configured (SAM 2.1 + Grounding DINO via the Spirula helper)"));
    };
    let (cand, dir) = get_artifact(run, &req.candidate_id, ArtifactKind::Candidate)?;
    if cand.meta.contains_key("edit") {
        return Err(invalid(format!("{} is already an edited candidate: select on its parent {}", req.candidate_id, cand.meta.get("parent_candidate_id").and_then(Value::as_str).unwrap_or("?"))));
    }
    let (split_id, sfm_id) = (meta_str(&cand, "split_id")?.to_string(), meta_str(&cand, "sfm_id")?.to_string());
    let split = crate::split::resolve(run, &split_id, &sfm_id)?;
    let ply = crate::views::candidate_ply(&dir)?;
    let (artifact, out) = new_artifact(
        run,
        ArtifactKind::Selection,
        json!({"candidate_id": req.candidate_id, "sfm_id": sfm_id, "split_id": split_id, "prompt": req.prompt, "threshold": req.threshold, "instances": req.instances, "ply": rel(run, &ply)}),
    )?;
    let job = SelectJob {
        ply,
        dataset: run.path(&sfm_id)?,
        max_resolution: cand.meta.get("max_resolution").and_then(Value::as_u64).unwrap_or(1920) as u32,
        vote_views: spread(&split.train_names, req.max_views as usize),
        heldout_views: spread(&split.heldout_names, HELDOUT_VIEWS),
        prompt: req.prompt,
        threshold: req.threshold as f32,
        largest_only: req.instances == Instances::Largest,
        sam,
        out: out.clone(),
        gpu_lock: cfg.gpu_lock.clone(),
    };
    let path = out.join("select_job.json");
    write_json(&path, &job)?;
    crate::start_worker(cfg, run, "select_objects", "select", &path, &artifact)
}

pub fn complete(run: &dyn RunCtx, artifact_id: &str, exit_code: i32, log: &str) -> anyhow::Result<ToolResult> {
    let (artifact, dir) = get_artifact(run, artifact_id, ArtifactKind::Selection)?;
    let summary: Option<serde_json::Map<String, Value>> = read_json(&dir.join("selection.json")).ok();
    let (0, Some(summary)) = (exit_code, summary) else {
        let mut r = failure("select_objects", "subprocess_failed", format!("selection worker exited {exit_code}: {log}"));
        r.artifacts = vec![artifact];
        return Ok(r);
    };
    let ids = splat_core::edit::read_npy_u32(&dir.join("selected_ids.npy"))?;
    if !dir.join("object_ids.npy").is_file() {
        splat_core::edit::write_npy_u32(&dir.join("object_ids.npy"), &vec![0; ids.len()])?;
    }
    let mut r = result("select_objects", Status::Succeeded);
    for k in ["n_gaussians", "n_selected", "frac_selected", "objects", "views_with_object", "heldout_iou_mean", "threshold", "timings_s"] {
        r.metrics.insert(k.into(), summary.get(k).cloned().unwrap_or(Value::Null));
    }
    r.metrics.insert("selected_ids".into(), json!(rel(run, &dir.join("selected_ids.npy"))));
    r.metrics.insert("masks".into(), json!(rel(run, &dir.join("masks"))));
    let strips: Vec<std::path::PathBuf> = summary.get("strips").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(|s| dir.join(s)).filter(|p| p.is_file()).collect();
    let images = splat_eval::budget_jpegs(&strips, &dir.join("previews"), splat_eval::RESULT_IMAGE_BUDGET)?;
    r.images = images.iter().map(|p| rel(run, p)).collect();
    if ids.is_empty() {
        r.warnings.push("Nothing selected: the detector found no matching object on the voting views, or its masks disagree across views. Try another noun.".into());
    }
    if summary.get("views_with_object").and_then(Value::as_u64).is_some_and(|n| n < 3) {
        r.warnings.push("The object was detected on fewer than 3 voting views: the selection is weakly supported.".into());
    }
    r.artifacts = vec![artifact];
    Ok(r)
}
