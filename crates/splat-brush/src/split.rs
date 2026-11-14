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
        // 360 datasets hold out whole panoramas (all views of every interval-th 360 frame).
        let panorama = splat_core::panorama::Manifest::read(&dataset)?.is_some();
        let heldout: Vec<String> = if panorama {
            splat_core::panorama::heldout_groups(&names, req.interval as usize).0
        } else {
            names.iter().step_by(req.interval as usize).cloned().collect()
        };
        let train = names.iter().filter(|n| !heldout.contains(n)).cloned().collect();
        let image_sha256 = names
            .iter()
            .map(|n| Ok((n.clone(), sha256_file(&dataset.join("images").join(n))?)))
            .collect::<anyhow::Result<_>>()?;
        let frozen = FrozenSplit {
            frames_id: Some(frames_id.clone()),
            sfm_id: req.sfm_id.clone(),
            mode: if panorama { "panorama_interval" } else { "interval" }.into(),
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
    // A 360 split counts held-out panoramas, not views.
    let panoramas = splat_core::panorama::Manifest::read(&dataset)?.map(|_| {
        resolved.heldout_names.iter().map(|n| splat_core::panorama::group_key(n).unwrap_or(n)).collect::<std::collections::BTreeSet<_>>().len()
    });
    let mut r = result("define_split", Status::Succeeded);
    r.metrics.insert("interval".into(), json!(req.interval));
    r.metrics.insert("n_train".into(), json!(resolved.train_names.len()));
    r.metrics.insert("n_heldout".into(), json!(count));
    r.metrics.insert("frames_id".into(), json!(resolved.frames_id));
    r.metrics.insert("split_sha256".into(), artifact.meta["sha256"].clone());
    if let Some(panoramas) = panoramas {
        r.metrics.insert("n_heldout_panoramas".into(), json!(panoramas));
        if panoramas < 8 {
            r.warnings.push("Fewer than eight held-out 360 frames; increase capture coverage.".into());
        }
    } else if count < 8 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use splat_core::Run;

    /// An SfM artifact over `frames_id` whose model registers `names` (images linked from the frames).
    fn sfm(run: &Run, frames_id: &str, frames: &std::path::Path, names: &[String]) -> String {
        let (artifact, out) = new_artifact(run, ArtifactKind::Sfm, json!({"frames_id": frames_id})).unwrap();
        std::fs::create_dir_all(out.join("images")).unwrap();
        std::fs::create_dir_all(out.join("sparse/0")).unwrap();
        for name in names {
            std::fs::copy(frames.join("images").join(name), out.join("images").join(name)).unwrap();
        }
        std::fs::write(out.join("sparse/0/cameras.txt"), "1 PINHOLE 8 8 4 4 4 4\n").unwrap();
        let images: String = names.iter().enumerate().map(|(i, n)| format!("{} 1 0 0 0 {i} 0 0 1 {n}\n\n", i + 1)).collect();
        std::fs::write(out.join("sparse/0/images.txt"), images).unwrap();
        std::fs::write(out.join("sparse/0/points3D.txt"), "").unwrap();
        splat_core::panorama::Manifest::new(8, 30.0, vec![]).write(&out).unwrap();
        artifact.id
    }

    #[test]
    fn panorama_split_holds_out_whole_frames_and_resolves_without_a_training_frame() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("source")).unwrap();
        let run = Run::create(tmp.path().join("run"), tmp.path().join("source"), "split", "brush").unwrap();
        let (frames, frames_dir) = new_artifact(&run, ArtifactKind::Frames, json!({"panorama": true})).unwrap();
        std::fs::create_dir_all(frames_dir.join("images")).unwrap();
        let mut names = Vec::new();
        for frame in 0..10 {
            for view in 0..3 {
                let name = format!("v{view:02}_frame_{frame:06}.jpg");
                image::RgbImage::from_pixel(8, 8, image::Rgb([frame * 20, view * 50, 7])).save(frames_dir.join("images").join(&name)).unwrap();
                names.push(name);
            }
        }
        let first = sfm(&run, &frames.id, &frames_dir, &names);
        let r = define_split(&run, DefineSplit { sfm_id: first.clone(), interval: 4 }).unwrap();
        assert_eq!(r.metrics["n_heldout"], 9);
        assert_eq!(r.metrics["n_heldout_panoramas"], 3);
        let frozen = splat_core::split::load(&run, &r.artifacts[0].id).unwrap();
        assert_eq!(frozen.mode, "panorama_interval");
        let group = |n: &String| splat_core::panorama::group_key(n).unwrap().to_string();
        let held: std::collections::BTreeSet<String> = frozen.heldout_names.iter().map(group).collect();
        let held_group = |n: &String| held.contains(&group(n));
        assert_eq!(held.iter().cloned().collect::<Vec<_>>(), ["frame_000000.jpg", "frame_000004.jpg", "frame_000008.jpg"]);
        assert!(frozen.train_names.iter().all(|n| !frozen.heldout_names.iter().any(|h| group(h) == group(n))));
        // A later solve that lost training frame 1 resolves; the held-out panoramas are unchanged.
        let without: Vec<String> = names.iter().filter(|n| !n.ends_with("frame_000001.jpg")).cloned().collect();
        let second = sfm(&run, &frames.id, &frames_dir, &without);
        let resolved = resolve(&run, &r.artifacts[0].id, &second).unwrap();
        assert_eq!(resolved.heldout_names, frozen.heldout_names);
        assert_eq!(resolved.train_names.len(), frozen.train_names.len() - 3);
        // A solve that lost one view of a held-out panorama still resolves without it; a view of a
        // held-out panorama never trains.
        let partial: Vec<String> = names.iter().filter(|n| n.as_str() != "v01_frame_000004.jpg").cloned().collect();
        let third = sfm(&run, &frames.id, &frames_dir, &partial);
        let resolved = resolve(&run, &r.artifacts[0].id, &third).unwrap();
        assert_eq!(resolved.heldout_names.len(), 8);
        assert!(resolved.train_names.iter().all(|n| !held_group(n)), "{:?}", resolved.train_names);
        // A held-out panorama with no view left fails, as a missing held-out frame does.
        let gone: Vec<String> = names.iter().filter(|n| !n.ends_with("frame_000004.jpg")).cloned().collect();
        let fourth = sfm(&run, &frames.id, &frames_dir, &gone);
        assert!(resolve(&run, &r.artifacts[0].id, &fourth).is_err());
    }
}
