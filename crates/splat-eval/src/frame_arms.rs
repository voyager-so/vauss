//! Research-only frame selection and fixed source-video heldout extraction.
//!
//! splat __frame_arms VIDEO OUT_DIR FFMPEG --freeze
//! splat __frame_arms VIDEO OUT_DIR FFMPEG --heldout OUT_DIR/heldout.json [ARM...]

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use crate::frameplan::{self, ExtractOptions, PlanOptions};
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn motion_time(curve: &[(f64, f64)], motion: f64, duration: f64) -> f64 {
    if motion <= 0.0 { return 0.0; }
    for pair in curve.windows(2) {
        let ((t0, m0), (t1, m1)) = (pair[0], pair[1]);
        if m1 >= motion && m1 > m0 {
            return t0 + (t1 - t0) * (motion - m0) / (m1 - m0);
        }
    }
    duration
}

pub fn run(args: &[String]) -> Result<()> {
    let [video, out, ffmpeg, rest @ ..] = args else {
        anyhow::bail!("usage: frame_arms VIDEO OUT_DIR FFMPEG [--freeze | --heldout PATH] [ARM...]");
    };
    let (video, out, ffmpeg) = (Path::new(video), PathBuf::from(out), PathBuf::from(ffmpeg));
    let ffprobe = ffmpeg.with_file_name("ffprobe");
    let p = crate::video::probe(video, &ffprobe)?;
    ensure!(p.duration_s > 0.0 && p.fps > 0.0, "duration and fps required");
    let (w, h) = if p.rotation_deg.rem_euclid(180) == 90 { (p.height, p.width) } else { (p.width, p.height) };
    let size = frameplan::output_size(w, h, 1600);
    std::fs::create_dir_all(&out)?;
    // CPU ffprobe audits supply the actual packet PTS timeline, including nonzero stream origins.
    let source_times: Vec<f64> = serde_json::from_slice(&std::fs::read(out.join("source-pts.json"))?)?;
    ensure!(!source_times.is_empty() && source_times.windows(2).all(|v| v[0] < v[1]), "ordered source timestamps required");
    let frames_available = source_times.len() as u64;
    let freeze = rest.iter().any(|x| x == "--freeze");
    let heldout_path = rest.iter().position(|x| x == "--heldout")
        .map(|i| rest.get(i + 1).context("--heldout needs a path")).transpose()?;
    ensure!(!(freeze && heldout_path.is_some()), "--freeze and --heldout are separate steps");
    let mut arms = vec![];
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--freeze" => i += 1,
            "--heldout" => i += 2,
            arm => { arms.push(arm); i += 1; }
        }
    }
    let track = if let Some(path) = heldout_path {
        let j: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(j["source_sha256"] == hash_file(video)?, "heldout source hash differs");
        serde_json::from_value::<Vec<(f64, f64)>>(j["motion_track"].clone())?
    } else {
        frameplan::motion_track(&ffmpeg, video, None, w, h)?
    };
    let shared_plan = frameplan::plan(&track, p.duration_s, frames_available, &PlanOptions::default());
    if freeze {
        ensure!(arms.is_empty(), "freeze before extracting arms");
        let path = out.join("heldout.json");
        ensure!(!path.exists(), "heldout already frozen at {}; use another output directory", path.display());
        let instants: Vec<f64> = (0..).map(|k| 2.0 + 4.0 * k as f64).take_while(|&t| t <= *source_times.last().unwrap())
            .map(|t| source_times.iter().enumerate().min_by(|(ia, a), (ib, b)| (**a-t).abs().total_cmp(&(**b-t).abs()).then(ia.cmp(ib))).unwrap().0 as f64 / p.fps).collect();
        let test_plan = frameplan::Plan { count: instants.len() as u32, instants_s: instants, ..shared_plan.clone() };
        let images = out.join("test/images");
        std::fs::create_dir_all(&images)?;
        let ex = frameplan::extract(&ffmpeg, video, None, size, p.fps, &test_plan,
            &ExtractOptions { uniform: true, drop_black: true, drop_duplicates: false, rotate_deg: 0, start_index: 0, excluded: vec![], pick_radius: 0, preserve_all: false, preserve_source_frames: true, crop: None, fill: vec![], anchors: vec![] }, &images)?;
        let tests: Vec<Value> = ex.kept.iter().map(|k| {
            let index = k["decoded_index"].as_u64().expect("decoded index");
            let elapsed = index as f64 / p.fps;
            let t = source_times[index as usize];
            let m = shared_plan.motion_at(elapsed);
            let left_motion = motion_time(&shared_plan.motion_curve, m - 0.05, p.duration_s);
            let right_motion = motion_time(&shared_plan.motion_curve, m + 0.05, p.duration_s);
            let inside = |i: usize| (source_times[i]-t).abs() <= 0.5 + 1e-9
                || (i as f64 / p.fps >= left_motion - 1e-9 && i as f64 / p.fps <= right_motion + 1e-9);
            let left = (0..=index as usize).find(|&i| inside(i)).unwrap();
            let right = (index as usize..source_times.len()).rfind(|&i| inside(i)).unwrap();
            json!({"name": k["name"], "source_index": index, "time_s": t, "motion_widths": m,
                "excluded": [left, right+1], "band_s": [source_times[left], source_times[right]]})
        }).collect();
        ensure!(!tests.is_empty(), "no heldout frames");
        let test_bytes = serde_json::to_vec(&tests)?;
        let test_hash = format!("{:x}", Sha256::digest(&test_bytes));
        let payload = json!({"schema": "splat/m6-heldout/v1", "source_sha256": hash_file(video)?, "probe": p,
            "output_size": size, "test_frames": tests, "test_list_sha256": test_hash, "motion_track": track,
            "motion_curve": shared_plan.motion_curve, "source_times_s": source_times, "dropped_black": ex.dropped_black,
            "tone_mapped_from": ex.tone_mapped, "policy": {"offset_s": 2, "period_s": 4, "min_band_s": 0.5, "min_motion_widths": 0.05}});
        let bytes = serde_json::to_vec_pretty(&payload)?;
        std::fs::write(&path, &bytes)?;
        std::fs::write(out.join("heldout.sha256"), format!("{:x}  heldout.json\n", Sha256::digest(&bytes)))?;
        println!("frozen {} test frames: {}", tests.len(), path.display());
        return Ok(());
    }
    let (excluded, heldout_hash): (Vec<Range<u64>>, Option<String>) = if let Some(path) = heldout_path {
        let bytes = std::fs::read(path)?;
        let j: Value = serde_json::from_slice(&bytes)?;
        let ranges = j["test_frames"].as_array().context("test_frames list")?.iter().map(|v| {
            let a = v["excluded"][0].as_u64().context("band start")?;
            let b = v["excluded"][1].as_u64().context("band end")?;
            ensure!(a < b && b <= frames_available, "invalid exclusion band");
            Ok(a..b)
        }).collect::<Result<Vec<_>>>()?;
        (ranges, Some(format!("{:x}", Sha256::digest(&bytes))))
    } else { (vec![], None) };
    if arms.is_empty() { arms = vec!["rate_uniform", "rate_only", "adaptive", "no_sharpest", "fixed150"]; }
    for arm in arms {
        let defaults = PlanOptions::default();
        let (opts, uniform) = match arm {
            "fixed150" => (PlanOptions { band: 1.0, min_count: 150, max_count: 150, base_fps: 150.0 / p.duration_s.max(1e-3), ..defaults }, false),
            "adaptive" | "blur_reject" => (defaults, false),
            "rate_uniform" => (PlanOptions { band: 1.0, ..defaults }, true),
            "rate_only" => (PlanOptions { band: 1.0, ..defaults }, false),
            "no_sharpest" => (defaults, true),
            other => anyhow::bail!("unknown arm {other}"),
        };
        let plan = frameplan::plan(&track, p.duration_s, frames_available, &opts);
        let dir = out.join(arm);
        ensure!(!dir.exists(), "arm directory already exists: {}", dir.display());
        let images = dir.join("images");
        std::fs::create_dir_all(&images)?;
        let start = Instant::now();
        let mut ex = frameplan::extract(&ffmpeg, video, None, size, p.fps, &plan,
            &ExtractOptions { uniform, drop_black: true, drop_duplicates: true, rotate_deg: 0, start_index: 0, excluded: excluded.clone(), pick_radius: 1, preserve_all: false, preserve_source_frames: true, crop: None, fill: vec![], anchors: vec![] }, &images)
            .with_context(|| format!("arm {arm}"))?;
        let mut dropped_blur = 0;
        if arm == "blur_reject" {
            let original = ex.kept.clone();
            let mut retained = vec![true; original.len()];
            for k in 1..original.len().saturating_sub(1) {
                let mut scores: Vec<f64> = original[k.saturating_sub(5)..(k + 6).min(original.len())].iter()
                    .map(|v| v["sharpness"].as_f64().unwrap_or(0.0)).collect();
                scores.sort_by(f64::total_cmp);
                let prev = (0..k).rev().find(|&j| retained[j]).unwrap();
                let left = original[prev]["decoded_index"].as_u64().unwrap() as f64 / p.fps;
                let right = original[k + 1]["decoded_index"].as_u64().unwrap() as f64 / p.fps;
                if original[k]["sharpness"].as_f64().unwrap_or(0.0) < 0.5 * scores[scores.len()/2]
                    && plan.motion_at(right) - plan.motion_at(left) <= 0.15 {
                    retained[k] = false;
                    let name = original[k]["name"].as_str().unwrap();
                    std::fs::remove_file(images.join(name))?;
                    let analysis = dir.join("analysis").join(name);
                    if analysis.exists() { std::fs::remove_file(analysis)?; }
                    dropped_blur += 1;
                }
            }
            ex.kept = original.into_iter().zip(retained).filter_map(|(v, keep)| keep.then_some(v)).collect();
        }
        let times: Vec<f64> = ex.kept.iter().map(|k| k["decoded_index"].as_u64().unwrap() as f64 / p.fps).collect();
        for frame in &mut ex.kept {
            let index = frame["source_index"].as_u64().unwrap() as usize;
            frame["time_s"] = json!(source_times[index]);
            frame["motion_widths"] = json!(plan.motion_at(index as f64 / p.fps));
        }
        ensure!(ex.kept.iter().all(|k| !excluded.iter().any(|r| r.contains(&k["decoded_index"].as_u64().unwrap()))), "exclusion violation");
        let manifest = json!({"arm": arm, "video": video, "source_sha256": hash_file(video)?, "heldout_sha256": heldout_hash,
            "kept": ex.kept.len(), "frames": ex.kept, "dropped_black": ex.dropped_black,
            "dropped_duplicates": ex.dropped_duplicates, "moved_excluded": ex.moved_excluded, "dropped_excluded": ex.dropped_excluded,
            "skipped_excluded_candidates": ex.skipped_excluded_candidates, "dropped_blur": dropped_blur,
            "extract_s": start.elapsed().as_secs_f64(), "frame_plan": plan.metrics(), "spacing": plan.audit(&times),
            "warnings": ex.warnings, "tone_mapped_from": ex.tone_mapped});
        std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
        println!("{arm}: {} frames, spacing {}", ex.kept.len(), manifest["spacing"]);
    }
    Ok(())
}
