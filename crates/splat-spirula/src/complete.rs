use crate::{
    eval_bridge,
    media::{jpeg, load_image},
    progress,
    split::verify_membership,
    util::*,
};
use anyhow::{Context, Result, ensure};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use regex::Regex;
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::LazyLock,
};

pub(crate) fn checkpoint(path: &Path) -> Result<(u32, PathBuf)> {
    let mut checkpoints = vec![];
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        let Some(step) = path
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("step-"))
            .and_then(|s| s.strip_suffix(".ckpt"))
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        if path.join("splat.ply").is_file() {
            checkpoints.push((step, path));
        }
    }
    checkpoints
        .into_iter()
        .max_by_key(|(step, _)| *step)
        .context("No saved splat checkpoint")
}
pub(crate) fn comparisons(candidate: &Path, out: &Path, pairs: &[String]) -> Result<Vec<PathBuf>> {
    let mut paths = vec![];
    for pair in pairs {
        let gt = image::open(safe_child(candidate, &format!("eval-gt-{pair}.png"))?)?;
        let rd = image::open(safe_child(candidate, &format!("eval-render-{pair}.png"))?)?;
        ensure!(
            gt.dimensions() == rd.dimensions(),
            "Eval image size mismatch: {pair}"
        );
        let mut canvas = RgbImage::new(gt.width() * 2, gt.height());
        image::imageops::replace(&mut canvas, &gt.to_rgb8(), 0, 0);
        image::imageops::replace(&mut canvas, &rd.to_rgb8(), gt.width() as i64, 0);
        let target = out.join(format!("comparison-{pair}.jpg"));
        jpeg(&target, &DynamicImage::ImageRgb8(canvas), 92)?;
        paths.push(target);
    }
    Ok(paths)
}
pub(crate) fn finish(run: &dyn RunCtx, handler: &str, job: &Path, code: i32) -> Result<ToolResult> {
    let spec: JobSpec = read(&job.join("request.json"))?;
    let id = spec
        .meta
        .get("artifact_id")
        .and_then(Value::as_str)
        .context("Missing completion artifact_id")?;
    let artifact = run.artifact(id)?;
    let path = run.path(id)?;
    let mut log = String::new();
    for name in ["stdout.log", "stderr.log"] {
        if let Ok(bytes) = fs::read(job.join(name)) {
            log.push_str(&String::from_utf8_lossy(&bytes));
            log.push('\n');
        }
    }
    let expected = format!("spirula:complete_{}", spec.tool);
    ensure!(
        handler == expected,
        "Unknown or mismatched Spirula completion handler: {handler}"
    );
    guarded(run, &spec.tool, || match spec.tool.as_str() {
        "mask_distractors" => masks(run, artifact, &path, code, &log),
        "reconstruct_sfm" => sfm(run, artifact, &path, code, &log),
        "train_splats" => train(run, artifact, &path, job, code, &log),
        "extract_mesh" => mesh(run, artifact, &path, code, &log),
        _ => anyhow::bail!("No Spirula completion handler for {}", spec.tool),
    })
}
fn with_artifact(mut r: ToolResult, a: Artifact) -> ToolResult {
    r.artifacts.push(a);
    r
}
static MASK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"frame\s+(\d+):\s+(.+?)\s+-- kept").unwrap());
fn masks(run: &dyn RunCtx, a: Artifact, path: &Path, code: i32, log: &str) -> Result<ToolResult> {
    if code != 0 {
        return Ok(failure("mask_distractors", code, a, log));
    }
    let source = run.path(field(&a, "frames_id")?)?.join("images");
    let frames = images(&source)?;
    let mut mapping = BTreeMap::new();
    for cap in MASK.captures_iter(log) {
        let id = cap[1].parse::<u64>()?;
        ensure!(
            mapping.insert(id, cap[2].to_owned()).is_none(),
            "Duplicate mask frame index: {id}"
        );
    }
    let raw = images(&path.join("raw_masks"))?
        .into_iter()
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with("frame_"))
                && p.extension().is_some_and(|s| s == "png")
        })
        .collect::<Vec<_>>();
    let names = frames
        .iter()
        .map(|p| string(p.strip_prefix(&source).unwrap()))
        .collect::<Result<BTreeSet<_>>>()?;
    let mapped = mapping.values().cloned().collect::<BTreeSet<_>>();
    if raw.is_empty()
        || raw.len() != frames.len()
        || mapped != names
        || mapping.len() != frames.len()
    {
        return Ok(with_artifact(
            error(
                "mask_distractors",
                "mask_coverage",
                "Mask log does not identify every input frame",
            ),
            a,
        ));
    }
    fs::create_dir_all(path.join("masks"))?;
    fs::create_dir_all(path.join("overlays"))?;
    let mut fractions = vec![];
    let mut sanity = BTreeMap::new();
    let mut previews = vec![];
    let mut seen = BTreeSet::new();
    let mut targets = BTreeSet::new();
    let mut subject = splat_eval::SubjectGuard::default();
    let presence_dir = path.join("presence_masks").is_dir();
    let mut emptied = vec![];
    for mask in &raw {
        let index = mask
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_prefix("frame_")
            .unwrap()
            .parse::<u64>()?;
        ensure!(seen.insert(index), "Duplicate mask image index: {index}");
        let name = mapping
            .get(&index)
            .context("Mask image index missing from log")?;
        let target = safe_child(&path.join("masks"), name)?.with_extension("png");
        ensure!(
            targets.insert(target.clone()),
            "Multiple source images map to the same mask filename"
        );
        fs::create_dir_all(target.parent().unwrap())?;
        let mut gray = image::open(mask)?.to_luma8();
        // Per-frame presence test (lib.rs mask_distractors): no box at the stricter detector score
        // means no person, so whatever the 0.3 pass masked there is the scene, not a distractor.
        let presence = path.join("presence_masks").join(mask.file_name().unwrap());
        if presence_dir && !image::open(&presence)?.to_luma8().as_raw().iter().any(|v| *v < 128) {
            if gray.as_raw().iter().any(|v| *v < 128) {
                emptied.push(name.clone());
            }
            gray.as_mut().fill(255);
        }
        let (original, _) = load_image(&safe_child(&source, name)?)?;
        if gray.dimensions() != original.dimensions() {
            return Ok(with_artifact(
                error(
                    "mask_distractors",
                    "mask_dimensions",
                    format!("Mask dimensions differ for {name}"),
                ),
                a,
            ));
        }
        gray.save(&target)?;
        let fraction =
            gray.as_raw().iter().filter(|v| **v < 128).count() as f64 / gray.as_raw().len() as f64;
        fractions.push(fraction);
        let masked = gray.as_raw().iter().map(|v| *v < 128).collect::<Vec<_>>();
        subject.add(splat_eval::centre_frame(gray.width() as usize, gray.height() as usize, &masked));
        sanity.insert(
            name.clone(),
            splat_eval::mask_stats(gray.width() as usize, gray.height() as usize, &masked),
        );
        if previews.len() < 8 {
            let mut overlay = original.to_rgb8();
            for (pixel, mask) in overlay.pixels_mut().zip(gray.pixels()) {
                let alpha = ((255 - mask[0]) / 2) as f64 / 255.0;
                let src = pixel.0;
                *pixel = Rgb([
                    (src[0] as f64 * (1.0 - alpha) + 255.0 * alpha).round() as u8,
                    (src[1] as f64 * (1.0 - alpha)).round() as u8,
                    (src[2] as f64 * (1.0 - alpha)).round() as u8,
                ]);
            }
            let preview = path
                .join("overlays")
                .join(mask.file_name().unwrap())
                .with_extension("jpg");
            jpeg(&preview, &DynamicImage::ImageRgb8(overlay), 92)?;
            previews.push(rel(run, &preview)?);
        }
    }
    let suspect = sanity.values().any(|s| s.object_masked_suspect);
    let mut metrics = json!({"mask_count":raw.len(),"masked_fraction_mean":fractions.iter().sum::<f64>()/fractions.len() as f64,"masked_fraction_max":fractions.iter().copied().fold(0.0,f64::max),"frames_with_detections":fractions.iter().filter(|f|**f>0.0).count(),"white_means":"keep", "object_masked_suspect":suspect, "per_frame":sanity});
    if presence_dir {
        metrics["presence_emptied_frames"] = json!(emptied);
    }
    let purpose = field(&a, "purpose")?.to_owned();
    let mut warnings = vec![];
    if purpose != "sfm_features" {
        let (guard, warning) = guard_training_masks(path, &targets, &subject, &purpose)?;
        metrics["training_masks"] = guard["training_masks"].clone();
        metrics["training_guard"] = guard;
        warnings.extend(warning);
    }
    write(&path.join("coverage.json"), &metrics)?;
    let mut r = with_artifact(result("mask_distractors"), a);
    r.metrics = meta(metrics);
    r.images = previews;
    if suspect {
        r.warnings.push("Distractor masks cover the frame centre or a large connected object; inspect overlays before using them for reconstruction or training.".into());
    }
    r.warnings.extend(warnings);
    Ok(r)
}
/// Subject guard for masks that reach the training loss: masks that sit in the frame centre on most
/// frames are the subject (B13 spot: "person; car; bicycle" masked the Spot robot), not distractors,
/// so they stay SfM-only (rules in splat_eval::SubjectGuard). Otherwise training_masks/ is a copy of
/// masks/. Training reads `training_masks` from coverage.json.
fn guard_training_masks(
    path: &Path,
    masks: &BTreeSet<PathBuf>,
    guard: &splat_eval::SubjectGuard,
    purpose: &str,
) -> Result<(Value, Option<String>)> {
    let subject = guard.subject();
    if !subject {
        let out = path.join("training_masks");
        for mask in masks {
            let target = out.join(mask.strip_prefix(path.join("masks"))?);
            fs::create_dir_all(target.parent().unwrap())?;
            fs::copy(mask, &target)?;
        }
    }
    let training_purpose = if subject { "sfm_features" } else { purpose };
    let report = json!({
        "requested_purpose": purpose,
        "training_purpose": training_purpose,
        "training_masks": (!subject).then_some("training_masks"),
        "centre_heavy_frame_frac": guard.centre_heavy_frac(),
        "centre_blob_frame_frac": guard.centre_blob_frac(),
    });
    let warning = subject.then(|| {
        format!(
            "Subject guard: the masks cover the frame centre (over half the centre third on {:.0}% of frames, a centre blob on {:.0}%), so they mark the static subject, not moving distractors; purpose {purpose} was downgraded to sfm_features: training keeps every pixel; SfM still ignores the masked features.",
            guard.centre_heavy_frac() * 100.0,
            guard.centre_blob_frac() * 100.0
        )
    });
    Ok((report, warning))
}
fn sfm(_run: &dyn RunCtx, a: Artifact, path: &Path, code: i32, log: &str) -> Result<ToolResult> {
    let sparse = path.join("sparse");
    let mut models = vec![];
    if sparse.is_dir() {
        for e in fs::read_dir(&sparse)? {
            let p = e?.path();
            if p.join("images.bin").exists() || p.join("images.txt").exists() {
                models.push(p);
            }
        }
    }
    models.sort();
    if models.is_empty() || !matches!(code, 0 | 3 | 4) {
        return Ok(failure("reconstruct_sfm", code, a, log));
    }
    let sizes = models
        .iter()
        .map(|p| Ok(eval_bridge::registered_names(p)?.len()))
        .collect::<Result<Vec<_>>>()?;
    let largest = sizes
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.cmp(b).then(ib.cmp(ia)))
        .unwrap()
        .0;
    if models[largest] != path.join(field(&a, "sparse_dir")?) {
        return Ok(with_artifact(
            error(
                "reconstruct_sfm",
                "unexpected_model_order",
                "Largest SfM model is not sparse/0",
            ),
            a,
        ));
    }
    let mut metrics = eval_bridge::sfm_health(
        &models[largest],
        &path.join("images"),
        a.meta.get("input_kind") == Some(&json!("video")),
    )?;
    metrics["components"] = json!(models.len());
    metrics["component_sizes"] = json!(sizes);
    metrics["exit_code"] = json!(code);
    write(&path.join("health.json"), &metrics)?;
    let mut r = with_artifact(result("reconstruct_sfm"), a);
    if let Some(gates) = metrics["gates"].as_object() {
        for (k, v) in gates {
            if *v == false {
                r.warnings.push(format!("SfM health gate failed: {k}"));
            }
        }
    }
    if models.len() > 1 {
        r.warnings
            .push("Multiple reconstruction components; training uses the largest.".into());
    }
    if code != 0 {
        r.warnings.push(format!(
            "Spirula returned a partial reconstruction (exit {code})."
        ));
    }
    r.metrics = meta(metrics);
    Ok(r)
}
fn train(
    run: &dyn RunCtx,
    a: Artifact,
    path: &Path,
    job: &Path,
    code: i32,
    log: &str,
) -> Result<ToolResult> {
    let series = path.join("training.jsonl");
    let last_training = progress::refresh(&job.join("stdout.log"), &series)?;
    let saved = checkpoint(path).ok();
    let resumable = saved
        .as_ref()
        .is_some_and(|(_, p)| p.join("state.tar").is_file());
    if let Some((_, checkpoint)) = &saved {
        write(
            &path.join("checkpoint.json"),
            &json!({"checkpoint":rel(run,checkpoint)?,"resumable":resumable}),
        )?;
    }
    if code != 0 {
        let mut r = failure("train_splats", code, a, log);
        if resumable {
            r.metrics.extend(meta(
                json!({"checkpoint":rel(run,&saved.as_ref().unwrap().1)?,"resumable":true}),
            ));
            r.warnings.push(
                "Resume uses the last periodic checkpoint; up to 99 training steps may be lost."
                    .into(),
            );
        }
        return Ok(r);
    }
    if !resumable {
        return Ok(with_artifact(
            error(
                "train_splats",
                "missing_checkpoint",
                "Training did not save a full checkpoint",
            ),
            a,
        ));
    }
    let checkpoint = saved.unwrap().1;
    let membership = verify_membership(run, &a)?;
    if membership["verified"] != true {
        let mut r = with_artifact(
            error(
                "train_splats",
                "split_mismatch",
                "Saved held-out membership differs from frozen manifest",
            ),
            a,
        );
        r.metrics = meta(membership);
        return Ok(r);
    }
    let native = read_engine_json(&path.join("metrics.json"))?;
    let mut metrics = BTreeMap::new();
    for (key, value) in native
        .as_object()
        .context("metrics.json must contain an object")?
    {
        if let Some(key) = key.strip_prefix("avg_") {
            metrics.insert(key.into(), value.clone());
        }
    }
    metrics.extend(meta(json!({"n_eval":native["num_eval_images"],"training_time_s":native["training_time"],"engine_vram_mb":native["engine_vram"],"split_verified":true,"training_series":rel(run,&series)?,"checkpoint":rel(run,&checkpoint)?})));
    let reader = BufReader::new(fs::File::open(checkpoint.join("splat.ply"))?);
    for line in reader.lines().take(256) {
        let line = line?;
        if let Some(n) = line.strip_prefix("element vertex ") {
            metrics.insert("exported_splats".into(), json!(n.parse::<u64>()?));
        }
        if line.trim() == "end_header" {
            break;
        }
    }
    let mut r = with_artifact(result("train_splats"), a);
    for key in ["psnr", "ssim"] {
        metrics.insert(format!("train_{key}"), last_training[key].clone());
    }
    metrics.insert("train_metrics_scope".into(), json!("last_training_batch"));
    metrics.insert("train_metrics_step".into(), last_training["step"].clone());
    r.warnings.push("Training PSNR/SSIM describe the last sampled batch after engine colour correction; they are not a full training-set evaluation.".into());
    r.metrics = metrics;
    r.images = images(path)?
        .iter()
        .filter(|p| {
            p.parent() == Some(path)
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("eval-render-"))
        })
        .take(8)
        .map(|p| rel(run, p))
        .collect::<Result<_>>()?;
    Ok(r)
}
static TOPOLOGY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"vertices:\s*(\d+),\s*faces:\s*(\d+)").unwrap());
fn mesh(run: &dyn RunCtx, a: Artifact, path: &Path, code: i32, log: &str) -> Result<ToolResult> {
    if code != 0 {
        return Ok(failure("extract_mesh", code, a, log));
    }
    let format = field(&a, "format")?;
    let mut files = vec![];
    let mut bytes = 0;
    for e in fs::read_dir(path)? {
        let p = e?.path();
        if p.is_file() {
            if p.file_name().is_some_and(|n| n != "mesh.json") {
                bytes += p.metadata()?.len();
            }
            if p.extension().and_then(|e| e.to_str()) == Some(format) {
                files.push(p);
            }
        }
    }
    files.sort();
    if files.is_empty()
        || files
            .iter()
            .any(|p| p.metadata().map(|m| m.len() == 0).unwrap_or(true))
    {
        return Ok(with_artifact(
            error(
                "extract_mesh",
                "missing_mesh",
                "Spirula did not produce the requested mesh format",
            ),
            a,
        ));
    }
    let mut metrics = json!({"files":files.iter().map(|p|rel(run,p)).collect::<Result<Vec<_>>>()?,"bytes":bytes,"coordinate_frame":"training","scene_transform":rel(run,&run.path(field(&a,"candidate_id")?)?.join("scene_transform.json"))?});
    if let Some(c) = TOPOLOGY.captures_iter(log).last() {
        metrics["vertices"] = json!(c[1].parse::<u64>()?);
        metrics["faces"] = json!(c[2].parse::<u64>()?);
    }
    let mut r = with_artifact(result("extract_mesh"), a);
    match crate::preview::render(&files[0], &path.join("geometry-preview")) {
        Ok(previews) => {
            r.images = previews
                .iter()
                .map(|p| rel(run, p))
                .collect::<Result<_>>()?;
            metrics["preview_style"] = json!("orbit_colour_and_geometry");
            r.warnings.push("Mesh previews show texture colour sampled per vertex (left) and shaded geometry (right); inspect the exported mesh for full texture quality.".into());
        }
        Err(e) => r
            .warnings
            .push(format!("Mesh saved, but geometry preview failed: {e:#}")),
    }
    write(&path.join("mesh.json"), &metrics)?;
    r.metrics = meta(metrics);
    Ok(r)
}
