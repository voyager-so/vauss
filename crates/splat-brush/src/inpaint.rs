//! `inpaint_removed`: fill the hole `remove_objects` left. Submits a detached worker
//! (`splat __brush inpaint job.json`) running [`splat_inpaint::run`] on the UNEDITED parent
//! candidate and the removed ids; the output is a new candidate whose parent is the edited one.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;

use crate::util::*;
use crate::{BrushConfig, split, start_worker};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InpaintWorkerJob {
    pub job: splat_inpaint::InpaintJob,
    pub gpu_lock: Option<PathBuf>,
}

pub fn inpaint_removed(cfg: &BrushConfig, run: &dyn RunCtx, req: InpaintRemoved) -> anyhow::Result<ToolResult> {
    if !(200..=10_000).contains(&req.steps) {
        return Err(invalid("steps must be 200..=10000"));
    }
    let (edited, _) = get_artifact(run, &req.candidate_id, ArtifactKind::Candidate)?;
    if edited.meta.get("edit").and_then(Value::as_str) != Some("remove_objects") {
        return Err(invalid(format!("{} is not a remove_objects candidate: run remove_objects first and pass its candidate", req.candidate_id)));
    }
    if edited.meta.get("selection_id").and_then(Value::as_str) != Some(req.selection_id.as_str()) {
        return Err(invalid(format!(
            "{} was made from selection {}, not {}",
            req.candidate_id,
            edited.meta.get("selection_id").and_then(Value::as_str).unwrap_or("?"),
            req.selection_id
        )));
    }
    let parent_id = meta_str(&edited, "parent_candidate_id")?.to_string();
    let (selection, sel_dir) = get_artifact(run, &req.selection_id, ArtifactKind::Selection)?;
    if selection.meta.get("candidate_id").and_then(Value::as_str) != Some(parent_id.as_str()) {
        return Err(invalid(format!("selection {} was not made on {parent_id}", req.selection_id)));
    }
    let (_, parent_dir) = get_artifact(run, &parent_id, ArtifactKind::Candidate)?;
    let parent_ply = crate::views::candidate_ply(&parent_dir)?;
    // The same ids remove_objects dropped: the selection's, limited to removed_object_ids (empty = all).
    let ids = splat_core::edit::read_npy_u32(&sel_dir.join("selected_ids.npy"))?;
    let objects = sel_dir.join("object_ids.npy");
    let objects = if objects.is_file() { splat_core::edit::read_npy_u32(&objects)? } else { vec![0; ids.len()] };
    if objects.len() != ids.len() {
        return Err(invalid("object_ids.npy and selected_ids.npy differ in length"));
    }
    let only: Vec<u32> = edited
        .meta
        .get("removed_object_ids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as u32).collect())
        .unwrap_or_default();
    let mut removed: Vec<u32> = ids.iter().zip(&objects).filter(|(_, o)| only.is_empty() || only.contains(o)).map(|(i, _)| *i).collect();
    removed.sort_unstable();
    removed.dedup();
    if removed.is_empty() {
        return Err(invalid("the selection holds no Gaussians for the removed objects"));
    }
    let (split_id, sfm_id) = (meta_str(&edited, "split_id")?.to_string(), meta_str(&edited, "sfm_id")?.to_string());
    let manifest = split::resolve(run, &split_id, &sfm_id)?;
    let max_resolution = edited.meta.get("max_resolution").and_then(Value::as_u64).unwrap_or(1920) as u32;
    let mut meta = edited.meta.clone();
    meta.insert("parent_candidate_id".into(), json!(req.candidate_id));
    meta.insert("edit".into(), json!("inpaint_removed"));
    meta.insert("selection_id".into(), json!(req.selection_id));
    meta.insert("source_candidate_id".into(), json!(parent_id));
    meta.insert("steps".into(), json!(req.steps));
    let (artifact, out) = new_artifact(run, ArtifactKind::Candidate, Value::Object(meta.into_iter().collect()))?;
    let ids_path = out.join("removed_ids.npy");
    splat_core::edit::write_npy_u32(&ids_path, &removed)?;
    let job = InpaintWorkerJob {
        job: splat_inpaint::InpaintJob {
            parent_ply,
            selected_ids: ids_path,
            dataset: run.path(&sfm_id)?,
            train_names: manifest.train_names.clone(),
            heldout_names: manifest.heldout_names.clone(),
            max_resolution,
            out: out.clone(),
            steps: req.steps,
            hole_thr: 0.3,
            dilate: 0.012,
            footprint_dilate: splat_inpaint::d_footprint_dilate(),
            ref_every: 3,
            max_new: 400_000,
            seed: 42,
            inpainter: splat_inpaint::d_inpainter(),
            prompt: None,
            negative_prompt: None,
            diffusion_steps: splat_inpaint::d_diffusion_steps(),
            guidance: splat_inpaint::d_guidance(),
            recolour: splat_inpaint::d_recolour(),
            recolour_pad: splat_inpaint::d_recolour_pad(),
            seen_alpha: splat_inpaint::d_seen_alpha(),
            seen_views: splat_inpaint::d_seen_views(),
            side_alpha: splat_inpaint::d_side_alpha(),
            front_views: splat_inpaint::d_front_views(),
            front_in_hole: None,
            max_stretch: splat_inpaint::d_max_stretch(),
            prune_colour: splat_inpaint::d_prune_colour(),
            surface_frac: splat_inpaint::d_surface_frac(),
            models_dir: cfg.models_dir.clone(),
            keep_finetune_dir: false,
        },
        gpu_lock: cfg.gpu_lock.clone(),
    };
    let path = out.join("inpaint_job.json");
    write_json(&path, &job)?;
    let mut r = start_worker(cfg, run, "inpaint_removed", "inpaint", &path, &artifact)?;
    r.metrics.insert("n_removed".into(), json!(removed.len()));
    Ok(r)
}

/// Worker body.
pub fn run_job(job: InpaintWorkerJob) -> anyhow::Result<i32> {
    let _gpu = crate::gpu_lock(job.gpu_lock.as_deref())?;
    splat_inpaint::run(job.job)?;
    Ok(0)
}

pub fn complete(run: &dyn RunCtx, artifact_id: &str, exit_code: i32, log: &str) -> anyhow::Result<ToolResult> {
    let (artifact, dir) = get_artifact(run, artifact_id, ArtifactKind::Candidate)?;
    let summary: Option<serde_json::Map<String, Value>> = read_json(&dir.join("result.json")).ok();
    let (0, Some(summary), true) = (exit_code, summary, dir.join("splat.ply").is_file()) else {
        let mut r = failure("inpaint_removed", "subprocess_failed", format!("inpaint worker exited {exit_code}: {log}"));
        r.artifacts = vec![artifact];
        return Ok(r);
    };
    let mut r = result("inpaint_removed", Status::Succeeded);
    for k in [
        "n_input", "n_removed", "n_pruned", "n_new", "n_output", "ref_view", "ref_hole_px", "views_with_hole", "depth_mode",
        "ring_rel_err_da", "ring_rel_err_plane", "steps", "heldout_views", "psnr_out_edit_vs_orig_mean", "psnr_out_edit_vs_orig_min",
        "psnr_out_removed_vs_orig_mean", "timings_s",
    ] {
        r.metrics.insert(k.into(), summary.get(k).cloned().unwrap_or(Value::Null));
    }
    r.metrics.insert("splat_ply".into(), json!(rel(run, &dir.join("splat.ply"))));
    r.metrics.insert("result".into(), json!(rel(run, &dir.join("result.json"))));
    let shots: Vec<PathBuf> = ["strip.jpg", "before.png", "after.png"].iter().map(|f| dir.join("images").join(f)).filter(|p| p.is_file()).collect();
    let images = splat_eval::budget_jpegs(&shots, &dir.join("previews"), splat_eval::RESULT_IMAGE_BUDGET)?;
    r.images = images.iter().map(|p| rel(run, p)).collect();
    if let Some(w) = summary.get("warnings").and_then(Value::as_array) {
        r.warnings.extend(w.iter().filter_map(Value::as_str).map(String::from));
    }
    if summary.get("psnr_out_edit_vs_orig_mean").and_then(Value::as_f64).is_some_and(|p| p < 35.0) {
        r.warnings.push("Held-out views changed outside the hole (PSNR vs the unedited render < 35 dB): the fill leaks into the scene.".into());
    }
    r.artifacts = vec![artifact];
    Ok(r)
}
