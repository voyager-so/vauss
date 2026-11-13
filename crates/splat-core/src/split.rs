//! A frozen validation protocol belongs to source frames, independent of an SfM attempt.
use crate::RunCtx;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use splat_contracts::ArtifactKind;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenSplit {
    #[serde(default)]
    pub frames_id: Option<String>,
    /// The original SfM is provenance; it does not constrain later training attempts.
    pub sfm_id: String,
    pub mode: String,
    pub interval: u32,
    pub registered_names: Vec<String>,
    pub heldout_names: Vec<String>,
    pub train_names: Vec<String>,
    pub image_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedSplit {
    pub frames_id: String,
    pub sfm_id: String,
    pub interval: u32,
    pub registered_names: Vec<String>,
    pub heldout_names: Vec<String>,
    pub train_names: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("Held-out frames are not registered in SfM {sfm_id}: {names:?}")]
pub struct MissingHeldoutFrames {
    pub sfm_id: String,
    pub names: Vec<String>,
}

pub fn sfm_frames_id(run: &dyn RunCtx, sfm_id: &str) -> Result<String> {
    let sfm = run.artifact(sfm_id)?;
    ensure!(sfm.kind == ArtifactKind::Sfm, "{sfm_id} is not an SfM artifact");
    let id = sfm.meta.get("frames_id").and_then(serde_json::Value::as_str)
        .context("SfM artifact has no source frames_id")?;
    ensure!(run.artifact(id)?.kind == ArtifactKind::Frames, "{id} is not a frames artifact");
    Ok(id.into())
}

fn image_path(images: &Path, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    ensure!(!name.is_empty() && path.components().all(|p| matches!(p, Component::Normal(_))), "Invalid source frame name: {name}");
    Ok(images.join(path))
}

fn hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("Read frozen image {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Validate immutable manifest and source bytes. Legacy manifests infer frames_id from their original SfM.
pub fn load(run: &dyn RunCtx, split_id: &str) -> Result<FrozenSplit> {
    let artifact = run.artifact(split_id)?;
    ensure!(artifact.kind == ArtifactKind::Split, "{split_id} is not a split artifact");
    let path = run.path(split_id)?.join("manifest.json");
    let expected = artifact.meta.get("sha256").and_then(serde_json::Value::as_str)
        .context("Frozen split has no manifest digest")?;
    ensure!(hash(&path)? == expected, "Frozen split manifest has changed");
    let mut split: FrozenSplit = serde_json::from_slice(&fs::read(path)?)?;
    ensure!((2..=50).contains(&split.interval), "Invalid frozen split interval");
    let original_frames = sfm_frames_id(run, &split.sfm_id)?;
    if let Some(id) = &split.frames_id {
        ensure!(id == &original_frames, "Frozen split frame binding disagrees with its original SfM");
    }
    split.frames_id = Some(original_frames.clone());
    let registered: BTreeSet<_> = split.registered_names.iter().collect();
    let heldout: BTreeSet<_> = split.heldout_names.iter().collect();
    let train: BTreeSet<_> = split.train_names.iter().collect();
    ensure!(registered.len() == split.registered_names.len() && heldout.len() == split.heldout_names.len()
        && train.len() == split.train_names.len() && heldout.is_disjoint(&train)
        && heldout.union(&train).copied().collect::<BTreeSet<_>>() == registered,
        "Frozen split contains duplicate or inconsistent membership");
    ensure!((!heldout.is_empty() || (split.mode == "imported_explicit"
        && run.artifact(&original_frames)?.meta.get("trust_poses") == Some(&serde_json::json!(true)))) && !train.is_empty(), "Frozen split needs training and held-out frames");
    let images = run.path(&original_frames)?.join("images");
    for name in &split.registered_names {
        let expected = split.image_sha256.get(name).with_context(|| format!("Frozen split has no digest for {name}"))?;
        ensure!(hash(&image_path(&images, name)?)? == *expected, "Frozen source image changed: {name}");
    }
    Ok(split)
}

/// The caller reads its engine's registered image names; core stays independent of COLMAP/backend crates.
/// Training membership may grow or shrink. Held-out identities and source bytes stay fixed.
pub fn resolve(
    run: &dyn RunCtx,
    split: &FrozenSplit,
    sfm_id: &str,
    registered_names: &[String],
) -> Result<ResolvedSplit> {
    let frames_id = sfm_frames_id(run, sfm_id)?;
    let frozen_frames = split.frames_id.clone().map(Ok)
        .unwrap_or_else(|| sfm_frames_id(run, &split.sfm_id))?;
    ensure!(frames_id == frozen_frames, "Split belongs to frames {frozen_frames}, but SfM {sfm_id} belongs to {frames_id}");
    let registered: BTreeSet<_> = registered_names.iter().cloned().collect();
    ensure!(registered.len() == registered_names.len(), "SfM contains duplicate source frame names");
    let missing = split.heldout_names.iter().filter(|n| !registered.contains(*n)).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(MissingHeldoutFrames { sfm_id: sfm_id.into(), names: missing }.into());
    }
    let frames_images = run.path(&frames_id)?.join("images");
    let sfm_images = run.path(sfm_id)?.join("images");
    for name in &registered {
        let source_hash = hash(&image_path(&frames_images, name)?)?;
        if let Some(expected) = split.image_sha256.get(name) {
            ensure!(&source_hash == expected, "Frozen source image changed: {name}");
        }
        ensure!(hash(&image_path(&sfm_images, name)?)? == source_hash, "SfM image differs from source frame: {name}");
    }
    let heldout: BTreeSet<_> = split.heldout_names.iter().collect();
    let train_names = registered.iter().filter(|n| !heldout.contains(n)).cloned().collect::<Vec<_>>();
    ensure!(!train_names.is_empty(), "No training frames remain after resolving the frozen split");
    Ok(ResolvedSplit {
        frames_id,
        sfm_id: sfm_id.into(),
        interval: split.interval,
        registered_names: registered.into_iter().collect(),
        heldout_names: split.heldout_names.clone(),
        train_names,
    })
}
