//! Frozen held-out split: every Nth registered frame (sorted by name), owned by the source
//! frames so it survives SfM retries (splat_core::split resolves it against each SfM).

use serde_json::json;
use splat_contracts::*;
use splat_core::RunCtx;
use splat_core::split::{FrozenSplit, ResolvedSplit};

use crate::model;
use crate::util::*;

pub fn define_split(run: &dyn RunCtx, req: DefineSplit) -> anyhow::Result<ToolResult> {
    let (_, dataset) = get_artifact(run, &req.sfm_id, ArtifactKind::Sfm)?;
    let frames_id = splat_core::split::sfm_frames_id(run, &req.sfm_id)?;
    let names = model::names(&dataset.join("sparse/0"))?;
    // Frames-owned pointer; SfM-owned pointer from before shared splits.
    let pointer = run.path(&frames_id)?.join("frozen_split.json");
    let prior = [pointer.clone(), dataset.join("frozen_split.json")]
        .into_iter()
        .find(|p| p.exists())
        .map(|p| read_json::<serde_json::Value>(&p))
        .transpose()?;
    let (artifact, resolved) = if let Some(prior) = prior {
        let id = prior["split_id"].as_str().unwrap_or_default().to_string();
        let frozen = splat_core::split::load(run, &id)?;
        if frozen.interval != req.interval {
            return Err(invalid(format!("split {id} is already frozen for {frames_id} at interval {}", frozen.interval)));
        }
        (run.artifact(&id)?, splat_core::split::resolve(run, &frozen, &req.sfm_id, &names)?)
    } else {
        if names.len() < 2 {
            return Err(invalid("at least two registered views are needed for a held-out split"));
        }
        let heldout: Vec<String> = names.iter().step_by(req.interval as usize).cloned().collect();
        let train = names.iter().filter(|n| !heldout.contains(n)).cloned().collect();
        let image_sha256 = names
            .iter()
            .map(|n| Ok((n.clone(), sha256_file(&dataset.join("images").join(n))?)))
            .collect::<anyhow::Result<_>>()?;
        let frozen = FrozenSplit {
            frames_id: Some(frames_id.clone()),
            sfm_id: req.sfm_id.clone(),
            mode: "interval".into(),
            interval: req.interval,
            registered_names: names.clone(),
            heldout_names: heldout,
            train_names: train,
            image_sha256,
        };
        let (id, out) = run.new_artifact_dir(ArtifactKind::Split)?;
        std::fs::create_dir_all(&out)?;
        write_json(&out.join("manifest.json"), &frozen)?;
        let mut artifact = Artifact { id, kind: ArtifactKind::Split, path: rel(run, &out), meta: Default::default() };
        artifact.meta.insert("sfm_id".into(), json!(req.sfm_id));
        artifact.meta.insert("frames_id".into(), json!(frames_id));
        artifact.meta.insert("sha256".into(), json!(sha256_file(&out.join("manifest.json"))?));
        run.register(artifact.clone())?;
        write_json(&pointer, &json!({"split_id": artifact.id}))?;
        (artifact, splat_core::split::resolve(run, &frozen, &req.sfm_id, &names)?)
    };
    let count = resolved.heldout_names.len();
    let mut r = result("define_split", Status::Succeeded);
    r.metrics.insert("interval".into(), json!(req.interval));
    r.metrics.insert("n_train".into(), json!(resolved.train_names.len()));
    r.metrics.insert("n_heldout".into(), json!(count));
    r.metrics.insert("frames_id".into(), json!(resolved.frames_id));
    r.metrics.insert("split_sha256".into(), artifact.meta["sha256"].clone());
    if count < 8 {
        r.warnings.push("Fewer than eight held-out views; increase capture coverage.".into());
    }
    r.artifacts = vec![artifact];
    Ok(r)
}

/// The frozen split resolved against `sfm_id`'s registered frames (held-out names fixed).
pub fn resolve(run: &dyn RunCtx, split_id: &str, sfm_id: &str) -> anyhow::Result<ResolvedSplit> {
    let (_, dataset) = get_artifact(run, sfm_id, ArtifactKind::Sfm)?;
    let frozen = splat_core::split::load(run, split_id)?;
    splat_core::split::resolve(run, &frozen, sfm_id, &model::names(&dataset.join("sparse/0"))?)
}
