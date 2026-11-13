use crate::{SpirulaBackend, eval_bridge, util::*};
use anyhow::{Context, Result, ensure};
use image::{DynamicImage, GrayImage, ImageDecoder, ImageReader, imageops::FilterType};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;
use splat_eval::frameplan;
use std::{
    collections::BTreeSet,
    fs,
    path::Path,
    process::Command,
    time::Instant,
};

pub(crate) fn load_image(path: &Path) -> Result<(DynamicImage, u8)> {
    if splat_eval::images::is_heif(path) {
        // HEIC/HEIF via ffmpeg (iPhone tile grids stitched), already upright.
        return Ok((splat_eval::images::load_heif(path)?.0, 1));
    }
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok((image, orientation.to_exif()))
}
pub(crate) fn cpu_cache_bytes(root: &Path, names: &[String], divisor: u32) -> Result<u64> {
    let mut bytes = 0_u64;
    for name in names {
        let decoder = ImageReader::open(safe_child(root, name)?)?
            .with_guessed_format()?
            .into_decoder()?;
        let (width, height) = decoder.dimensions();
        let component_bytes = match decoder.color_type() {
            image::ColorType::Rgb32F | image::ColorType::Rgba32F => 4,
            image::ColorType::L16
            | image::ColorType::La16
            | image::ColorType::Rgb16
            | image::ColorType::Rgba16 => 2,
            _ => 1,
        };
        let rgb = u64::from(width.div_ceil(divisor))
            .checked_mul(u64::from(height.div_ceil(divisor)))
            .and_then(|pixels| pixels.checked_mul(3 * component_bytes))
            .context("Image cache estimate overflow")?;
        // Reserve a full-resolution byte mask even when one is synthesized.
        let mask = u64::from(width) * u64::from(height);
        bytes = bytes
            .checked_add(rgb)
            .and_then(|sum| sum.checked_add(mask))
            .context("Image cache estimate overflow")?;
    }
    Ok(bytes)
}
pub(crate) fn jpeg(path: &Path, image: &DynamicImage, quality: u8) -> Result<()> {
    image::codecs::jpeg::JpegEncoder::new_with_quality(fs::File::create(path)?, quality)
        .encode_image(&image.to_rgb8())?;
    Ok(())
}
fn gray(image: &DynamicImage, w: u32, h: u32) -> GrayImage {
    image
        .grayscale()
        .resize_exact(w, h, FilterType::CatmullRom)
        .to_luma8()
}
fn run_command(cmd: &mut Command) -> Result<Vec<u8>> {
    let output = cmd.output().with_context(|| format!("Execute {cmd:?}"))?;
    ensure!(
        output.status.success(),
        "Subprocess failed: {}",
        tail(&String::from_utf8_lossy(&output.stderr), 2000)
    );
    Ok(output.stdout)
}
fn number(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}
pub(crate) fn probe(backend: &SpirulaBackend, source: &Path) -> Result<Value> {
    let raw = run_command(
        Command::new(&backend.config.ffprobe)
            .args([
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
            ])
            .arg(source),
    )?;
    let data: Value = serde_json::from_slice(&raw)?;
    let videos: Vec<_> = data["streams"]
        .as_array()
        .context("ffprobe streams unavailable")?
        .iter()
        .filter(|s| s["codec_type"] == "video")
        .collect();
    let s = videos
        .first()
        .context("no_video_track: source has no video stream")?;
    let mut rotation = number(&s["tags"]["rotate"]).unwrap_or(0.0);
    if let Some(sides) = s["side_data_list"].as_array() {
        for side in sides {
            if let Some(r) = number(&side["rotation"]) {
                rotation = r;
            }
        }
    }
    let duration = number(&s["duration"])
        .or_else(|| number(&data["format"]["duration"]))
        .unwrap_or(0.0);
    let rate = |v: &Value| -> Option<f64> {
        let (n, d) = v.as_str()?.split_once('/')?;
        let d = d.parse::<f64>().ok()?;
        let n = n.parse::<f64>().ok()?;
        (d > 0.0 && n > 0.0).then_some(n / d)
    };
    // WebM can report a bogus avg_frame_rate (cambridge_heath_rd: 30000/1); trust r_frame_rate then.
    let (avg, r) = (rate(&s["avg_frame_rate"]), rate(&s["r_frame_rate"]));
    let fps = match (avg, r) {
        (Some(a), Some(r)) if a > 1.5 * r => r,
        (Some(a), _) => a,
        (None, r) => r.unwrap_or(0.0),
    };
    let count = number(&s["nb_frames"]).unwrap_or((duration * fps).round());
    // Display size: ffmpeg autorotates, so a 90/270 phone video decodes as portrait (like brush).
    let (mut width, mut height) = (s["width"].clone(), s["height"].clone());
    if (rotation.round() as i64).rem_euclid(180) == 90 {
        std::mem::swap(&mut width, &mut height);
    }
    Ok(
        json!({"codec":s["codec_name"],"width":width,"height":height,"duration_s":duration,"fps":fps,"frame_count_estimate":count,"rotation_degrees":rotation,"rotation_metadata_present":rotation!=0.0,"upside_down_metadata":rotation.abs()%360.0==180.0,"visual_orientation":"unverified","video_streams":videos.len()}),
    )
}
pub(crate) fn inspect(
    backend: &SpirulaBackend,
    run: &dyn RunCtx,
    req: InspectInput,
) -> Result<ToolResult> {
    ensure!(
        (1..=48).contains(&req.sample_count),
        "sample_count must be 1..48"
    );
    let start = Instant::now();
    let (artifact, out) = new_artifact(
        run,
        ArtifactKind::Images,
        json!({"purpose":"input_inspection"}),
    )?;
    let mut warnings = vec![];
    let mut samples = vec![];
    let mut metrics = if run.source().is_dir() {
        let inputs = images(&image_dir(run.source()))?;
        ensure!(!inputs.is_empty(), "No input images");
        let count = inputs.len().min(req.sample_count as usize);
        let mut orientations = BTreeSet::new();
        for i in 0..count {
            let index = if count == 1 {
                0
            } else {
                i * (inputs.len() - 1) / (count - 1)
            };
            let (image, orientation) = load_image(&inputs[index])?;
            orientations.insert(orientation);
            let thumb = image.thumbnail(960, 960);
            jpeg(&out.join(format!("sample-{i:03}.jpg")), &thumb, 92)?;
            samples.push(gray(&thumb, 640, 360));
        }
        json!({"input_kind":"images","image_count":inputs.len(),"exif_orientations":orientations,"rotation_metadata_present":orientations.iter().any(|o|*o!=1),"upside_down_metadata":orientations.contains(&3),"visual_orientation":"unverified"})
    } else {
        let mut m = probe(backend, run.source())?;
        m["input_kind"] = json!("video");
        let duration = m["duration_s"].as_f64().unwrap();
        ensure!(duration > 0.0, "Video duration is unavailable");
        let fps = req.sample_count as f64 / duration;
        let raw = run_command(
            Command::new(&backend.config.ffmpeg)
                .args(["-v", "error", "-threads", "8", "-i"])
                .arg(run.source())
                .args([
                    "-vf",
                    &format!("fps={fps},scale=640:360"),
                    "-frames:v",
                    &req.sample_count.to_string(),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    "-",
                ]),
        )?;
        ensure!(
            !raw.is_empty() && raw.len() % (640 * 360) == 0,
            "FFmpeg did not return complete sample frames"
        );
        for frame in raw.chunks_exact(640 * 360) {
            samples.push(GrayImage::from_raw(640, 360, frame.to_vec()).unwrap());
        }
        run_command(
            Command::new(&backend.config.ffmpeg)
                .args(["-v", "error", "-threads", "8", "-i"])
                .arg(run.source())
                .args([
                    "-vf",
                    &format!("fps={fps},scale=960:960:force_original_aspect_ratio=decrease"),
                    "-frames:v",
                    &req.sample_count.to_string(),
                ])
                .arg(out.join("sample-%03d.jpg")),
        )?;
        if m["rotation_metadata_present"] == true {
            warnings
                .push("Rotation metadata detected; FFmpeg extraction applies autorotation.".into());
        }
        m
    };
    metrics.as_object_mut().unwrap().extend(
        eval_bridge::summarize_frames(&samples)?
            .as_object()
            .unwrap()
            .clone(),
    );
    if metrics["upside_down_metadata"] == true {
        warnings.push(
            "Source metadata indicates upside-down capture; inspect the corrected samples.".into(),
        );
    }
    warnings.push(
        "Visual upside-down orientation without metadata needs driver inspection of samples."
            .into(),
    );
    if metrics["sharpness_p10_over_median"].as_f64().unwrap_or(1.0) < 0.6 {
        warnings.push("Motion blur: prefer sharpness_window frame selection.".into());
    }
    if metrics["luminance_range"].as_f64().unwrap_or(0.0) > 40.0 {
        warnings.push("Exposure swings detected.".into());
    }
    if metrics["top_clipped_fraction"].as_f64().unwrap_or(0.0) > 0.05 {
        warnings.push("Clipped highlights in the top third; inspect sky coverage.".into());
    }
    metrics["orientation_suspect"] = json!(
        metrics["rotation_metadata_present"] == true || metrics["upside_down_metadata"] == true
    );
    metrics["static_frac"] = metrics["static_fraction"].clone();
    write(&out.join("inspection.json"), &metrics)?;
    let mut r = result("inspect_input");
    r.artifacts.push(artifact);
    r.metrics = meta(metrics);
    r.images = images(&out)?
        .iter()
        .map(|p| rel(run, p))
        .collect::<Result<_>>()?;
    r.warnings = warnings;
    r.wall_s = Some(start.elapsed().as_secs_f64());
    Ok(r)
}
pub(crate) fn extract(
    backend: &SpirulaBackend,
    run: &dyn RunCtx,
    req: ExtractFrames,
) -> Result<ToolResult> {
    ensure!(!req.preserve_all && matches!(req.input_kind, splat_contracts::InputKind::Auto),
        "preserve_all and declared input_kind require the brush backend");
    ensure!(
        (20..=1500).contains(&req.target_count),
        "target_count must be 20..1500"
    );
    ensure!(
        (480..=4096).contains(&req.max_edge),
        "max_edge must be 480..4096"
    );
    ensure!(
        matches!(req.rotate_deg, 0 | 90 | 180 | 270),
        "rotate_deg must be 0, 90, 180 or 270"
    );
    let start = Instant::now();
    let (artifact, out) = new_artifact(
        run,
        ArtifactKind::Frames,
        json!({"input_kind":if run.source().is_dir(){"images"}else{"video"}}),
    )?;
    let dest = out.join("images");
    fs::create_dir(&dest)?;
    let mut metrics = if run.source().is_dir() {
        ensure!(req.time_range_s.is_none(), "time_range_s requires a video");
        let source = image_dir(run.source());
        let inputs = images(&source)?;
        ensure!(!inputs.is_empty(), "No input images");
        let count = inputs.len().min(req.target_count as usize);
        let mut selected = vec![];
        // numpy.array_split puts the one-element-larger windows first.
        let mut lo = 0;
        for i in 0..count {
            let hi = lo + inputs.len() / count + usize::from(i < inputs.len() % count);
            let index = match req.strategy {
                FrameStrategy::Uniform => lo + (hi - lo) / 2,
                FrameStrategy::SharpnessWindow => {
                    let mut best = (lo, f64::NEG_INFINITY);
                    for (j, path) in inputs.iter().enumerate().take(hi).skip(lo) {
                        let (image, _) = load_image(path)?;
                        let score=eval_bridge::summarize_frames(&[gray(&image,640,360)])?["sharpness_median"].as_f64().unwrap();
                        if score > best.1 {
                            best = (j, score);
                        }
                    }
                    best.0
                }
            };
            let (image, _) = load_image(&inputs[index])?;
            let size = image.width().max(image.height());
            let image = if size > req.max_edge {
                image.resize(req.max_edge, req.max_edge, FilterType::Lanczos3)
            } else {
                image
            };
            // Keep EXIF (focal-length prior for SfM); orientation is already applied to the pixels.
            splat_eval::images::write_jpeg(&dest.join(format!("frame-{i:06}.jpg")), &image.to_rgb8(), 95, splat_eval::images::exif_upright(&inputs[index]))?;
            selected.push(string(inputs[index].strip_prefix(&source)?)?);
            lo = hi;
        }
        json!({"input_count":inputs.len(),"selected_source_names":selected})
    } else {
        let p = probe(backend, run.source())?;
        let duration = p["duration_s"].as_f64().unwrap_or(0.0);
        if let Some((begin, end)) = req.time_range_s {
            ensure!(
                begin.is_finite() && end.is_finite() && 0.0 <= begin && begin < end && end <= duration,
                "time_range_s must lie within the video duration"
            );
        }
        let (width, height) = (p["width"].as_u64().unwrap_or(0) as u32, p["height"].as_u64().unwrap_or(0) as u32);
        ensure!(width > 0 && height > 0, "Video dimensions are unavailable");
        let fps = p["fps"].as_f64().unwrap_or(0.0);
        let span = req.time_range_s.map_or(duration, |(a, b)| b - a);
        let available = (span * fps).round().max(1.0) as u64;
        // Shared with brush: frame count follows duration and camera motion (splat_eval::frameplan).
        let track = frameplan::motion_track(&backend.config.ffmpeg, run.source(), req.time_range_s, width, height)?;
        let plan = frameplan::plan(
            &track,
            span,
            available,
            &frameplan::PlanOptions::spaced(req.target_count, matches!(req.spacing, FrameSpacing::Motion)),
        );
        // Rotation, black and duplicate drops run below on the written files.
        let extracted = frameplan::extract(
            &backend.config.ffmpeg,
            run.source(),
            req.time_range_s,
            frameplan::output_size(width, height, req.max_edge),
            fps,
            &plan,
            &frameplan::ExtractOptions {
                uniform: matches!(req.strategy, FrameStrategy::Uniform),
                drop_black: false,
                drop_duplicates: false,
                rotate_deg: 0,
                start_index: req.time_range_s.map_or(0, |(a, _)| (a * fps).round() as u64),
                excluded: Vec::new(),
                pick_radius: 1,
                preserve_all: false, preserve_source_frames: false,
                crop: None,
                fill: Vec::new(),
                anchors: Vec::new(),
            },
            &dest,
        )?;
        let times = extracted.kept.iter().filter_map(|k| k["decoded_index"].as_f64()).map(|i| i / fps).collect::<Vec<_>>();
        let mut frame_plan = plan.metrics();
        frame_plan["spacing"] = plan.audit(&times);
        // HDR (HLG/PQ) phone video is tone-mapped to SDR for SfM and training.
        frame_plan["tone_mapped_from"] = serde_json::json!(extracted.tone_mapped);
        frame_plan["analysis_lifted"] = serde_json::json!(extracted.analysis_lifted);
        frame_plan["warnings"] = serde_json::json!(extracted.warnings);
        json!({"decoder":"ffmpeg","frames_decoded":extracted.decoded,"rotation_applied":p["rotation_degrees"],"frame_plan":frame_plan})
    };
    let mut dropped_black = 0;
    let mut dropped_duplicates = 0;
    let mut previous: Option<GrayImage> = None;
    for frame in images(&dest)? {
        let (mut rgb, _) = load_image(&frame)?;
        rgb = match req.rotate_deg {
            90 => rgb.rotate90(),
            180 => rgb.rotate180(),
            270 => rgb.rotate270(),
            _ => rgb,
        };
        if req.rotate_deg != 0 {
            if frame
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| matches!(s.to_ascii_lowercase().as_str(), "jpg" | "jpeg"))
            {
                jpeg(&frame, &rgb, 95)?;
            } else {
                rgb.save(&frame)?;
            }
        }
        let gray = gray(&rgb, 160, 90);
        let black =
            gray.as_raw().iter().filter(|v| **v < 16).count() as f64 / gray.as_raw().len() as f64;
        let diff = previous.as_ref().map(|p| {
            p.as_raw()
                .iter()
                .zip(gray.as_raw())
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .sum::<f64>()
                / gray.as_raw().len() as f64
        });
        if req.drop_black && black > 0.98 {
            fs::remove_file(&frame)?;
            dropped_black += 1;
        } else if req.drop_duplicates && diff.is_some_and(|d| d < 1.0) {
            fs::remove_file(&frame)?;
            dropped_duplicates += 1;
        } else {
            previous = Some(gray);
        }
    }
    let frames = images(&dest)?;
    ensure!(!frames.is_empty(), "Extraction wrote no images");
    metrics.as_object_mut().unwrap().extend(json!({"dropped_black":dropped_black,"dropped_duplicates":dropped_duplicates,"rotate_deg":req.rotate_deg,"frame_count":frames.len(),"strategy":req.strategy,"max_edge":req.max_edge}).as_object().unwrap().clone());
    write(
        &out.join("selection.json"),
        &json!({"request":req,"metrics":metrics,"files":frames.iter().map(|p|string(p.strip_prefix(&dest).unwrap())).collect::<Result<Vec<_>>>()?}),
    )?;
    let mut r = result("extract_frames");
    r.warnings.extend(
        metrics["frame_plan"]["warnings"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)),
    );
    r.artifacts.push(artifact);
    r.metrics = meta(metrics);
    r.images = frames
        .iter()
        .step_by((frames.len() / 8).max(1))
        .take(8)
        .map(|p| rel(run, p))
        .collect::<Result<_>>()?;
    r.wall_s = Some(start.elapsed().as_secs_f64());
    Ok(r)
}
