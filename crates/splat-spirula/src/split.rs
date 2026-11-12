use crate::{eval_bridge, media::load_image, util::*};
use anyhow::{Context, Result, ensure};
use image::imageops::FilterType;
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;
use std::{
    collections::BTreeSet,
    fs,
    path::Path,
};

pub(crate) type Split = splat_core::split::FrozenSplit;
pub(crate) fn load(run: &dyn RunCtx, id: &str) -> Result<Split> {
    splat_core::split::load(run, id)
}
pub(crate) fn resolve(run: &dyn RunCtx, id: &str, sfm_id: &str) -> Result<splat_core::split::ResolvedSplit> {
    let split = load(run, id)?;
    let (sfm, dataset) = get(run, sfm_id, ArtifactKind::Sfm)?;
    let names = eval_bridge::registered_names(&dataset.join(field(&sfm, "sparse_dir")?))?;
    splat_core::split::resolve(run, &split, sfm_id, &names)
}
pub(crate) fn define(run: &dyn RunCtx, req: DefineSplit) -> Result<ToolResult> {
    ensure!((2..=50).contains(&req.interval), "interval must be 2..50");
    let (sfm, dataset) = get(run, &req.sfm_id, ArtifactKind::Sfm)?;
    let frames_id = splat_core::split::sfm_frames_id(run, &req.sfm_id)?;
    let frames = run.path(&frames_id)?;
    let pointer = frames.join("frozen_split.json");
    let legacy = dataset.join("frozen_split.json");
    let prior_pointer = if pointer.exists() { &pointer } else { &legacy };
    let (artifact, data) = if prior_pointer.exists() {
        let prior: Value = read(prior_pointer)?;
        let id = prior["split_id"]
            .as_str()
            .context("Missing split pointer")?;
        let data = load(run, id)?;
        ensure!(
            data.interval == req.interval,
            "Split is already frozen for these source frames at a different interval"
        );
        (run.artifact(id)?, data)
    } else {
        let names = eval_bridge::registered_names(&dataset.join(field(&sfm, "sparse_dir")?))?;
        ensure!(
            names.len() >= 2,
            "At least two registered views are needed for a held-out split"
        );
        let heldout: Vec<_> = names
            .iter()
            .step_by(req.interval as usize)
            .cloned()
            .collect();
        let train = names
            .iter()
            .filter(|n| !heldout.contains(n))
            .cloned()
            .collect();
        let hashes = names
            .iter()
            .map(|n| Ok((n.clone(), hash(&safe_child(&frames.join("images"), n)?)?)))
            .collect::<Result<_>>()?;
        let data = Split {
            frames_id: Some(frames_id.clone()),
            sfm_id: req.sfm_id.clone(),
            mode: "interval".into(),
            interval: req.interval,
            registered_names: names,
            heldout_names: heldout,
            train_names: train,
            image_sha256: hashes,
        };
        let (id, path) = run.new_artifact_dir(ArtifactKind::Split)?;
        fs::create_dir_all(&path)?;
        write(&path.join("manifest.json"), &data)?;
        let artifact = Artifact {
            id,
            kind: ArtifactKind::Split,
            path: rel(run, &path)?,
            meta: meta(json!({"frames_id":frames_id,"sfm_id":req.sfm_id,"sha256":hash(&path.join("manifest.json"))?})),
        };
        run.register(artifact.clone())?;
        (artifact, data)
    };
    let resolved = resolve(run, &artifact.id, &req.sfm_id)?;
    for path in [&pointer, &legacy] {
        if !path.exists() { write(path, &json!({"split_id":artifact.id}))?; }
    }
    let mut r = result("define_split");
    r.metrics = meta(
        json!({"interval":data.interval,"n_train":resolved.train_names.len(),"n_heldout":data.heldout_names.len(),"frames_id":frames_id,"split_sha256":artifact.meta["sha256"]}),
    );
    if data.heldout_names.len() < 8 {
        r.warnings
            .push("Fewer than eight held-out views; increase capture coverage.".into());
    }
    r.artifacts.push(artifact);
    Ok(r)
}
fn thumbnail(path: &Path) -> Result<Vec<u8>> {
    let (image, _) = load_image(path)?;
    Ok(image
        .resize_exact(64, 64, FilterType::CatmullRom)
        .to_rgb8()
        .into_raw())
}
/// Spirula shuffles eval pairs. Match every GT against all frozen source images,
/// require a unique match, and reject missing/extra renders or training views.
pub fn verify_membership(run: &dyn RunCtx, candidate: &Artifact) -> Result<Value> {
    let data = resolve(run, field(candidate, "split_id")?, field(candidate, "sfm_id")?)?;
    let dataset = run.path(&data.sfm_id)?.join("images");
    let path = run.path(&candidate.id)?;
    let config = read_engine_json(&path.join("config.json"))?;
    let mode = candidate.meta.get("eval_mode").and_then(Value::as_str).unwrap_or("interval");
    let report = if config["eval_mode"] != mode || (mode == "interval" && config["eval_interval"] != data.interval) {
        json!({"verified":false,"reason":"Resolved training config differs from frozen split"})
    } else {
        let thumbs = data
            .registered_names
            .iter()
            .map(|n| thumbnail(&safe_child(&dataset, n)?))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            thumbs.len() >= 2,
            "Split needs at least two source identities"
        );
        let mut rows = vec![];
        let all = images(&path)?;
        for gt in all.iter().filter(|p| {
            p.parent() == Some(path.as_path())
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("eval-gt-") && s.ends_with(".png"))
        }) {
            let pair = gt
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap()
                .strip_prefix("eval-gt-")
                .unwrap();
            let thumb = thumbnail(gt)?;
            let mut errors: Vec<(usize, f64)> = thumbs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    (
                        i,
                        t.iter()
                            .zip(&thumb)
                            .map(|(a, b)| (*a as f64 - *b as f64).abs())
                            .sum::<f64>()
                            / thumb.len() as f64,
                    )
                })
                .collect();
            errors.sort_by(|a, b| a.1.total_cmp(&b.1));
            let matched = errors[0].1 <= 3.0 && errors[1].1 - errors[0].1 > 0.25;
            rows.push(json!({"pair":pair,"source_name":data.registered_names[errors[0].0],"mae_255":errors[0].1,"identity_verified":matched,"render_exists":path.join(format!("eval-render-{pair}.png")).is_file()}));
        }
        let expected: BTreeSet<_> = data.heldout_names.iter().cloned().collect();
        let found: BTreeSet<String> = rows
            .iter()
            .filter(|r| r["identity_verified"] == true && r["render_exists"] == true)
            .map(|r| r["source_name"].as_str().unwrap().into())
            .collect();
        let render_count = all
            .iter()
            .filter(|p| {
                p.parent() == Some(path.as_path())
                    && p.file_name()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| s.starts_with("eval-render-") && s.ends_with(".png"))
            })
            .count();
        let verified = expected == found
            && rows.len() == expected.len()
            && rows
                .iter()
                .all(|r| r["identity_verified"] == true && r["render_exists"] == true)
            && render_count == expected.len();
        json!({"verified":verified,"method":"source_image_sha256_and_unique_gt_thumbnail_match","expected_count":expected.len(),"saved_pair_count":rows.len(),"missing_names":expected.difference(&found).collect::<Vec<_>>(),"unexpected_names":found.difference(&expected).collect::<Vec<_>>(),"pairs":rows})
    };
    write(&path.join("split_verification.json"), &report)?;
    Ok(report)
}
