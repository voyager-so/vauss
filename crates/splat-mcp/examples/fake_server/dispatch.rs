//! Fake tool dispatch for driver dry runs and tests: plausible ToolResults and real image files,
//! no GPU, no backend. Long tools return Running and complete on the next job_status call.
//! Scenario "rotation" (env SPLAT_FAKE_SCENARIO) is a pure pan, so a driver should refuse.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};

use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use splat_contracts::{Artifact, ArtifactKind, SCHEMA_VERSION, Status, ToolError, ToolResult};
use splat_core::RunCtx;
use splat_mcp::Dispatch;

fn result(tool: &str, status: Status, metrics: Value) -> ToolResult {
    ToolResult {
        schema_version: SCHEMA_VERSION,
        tool: tool.into(),
        status,
        job_id: None,
        artifacts: vec![],
        metrics: serde_json::from_value(metrics).unwrap_or_default(),
        images: vec![],
        warnings: vec![],
        error: None,
        wall_s: None,
    }
}

/// A sky/grass frame with a grey blob whose position depends on `seed`; `blur` softens the blob.
fn scene(seed: u32, blur: bool) -> RgbImage {
    let (w, h) = (960, 540);
    let cx = 300 + (seed * 37) % 400;
    RgbImage::from_fn(w, h, |x, y| {
        let (dx, dy) = (x as f32 - cx as f32, y as f32 - 280.0);
        let d = (dx * dx / 110.0f32.powi(2) + dy * dy / 120.0f32.powi(2)).sqrt();
        let edge = if blur { 0.15 } else { 0.02 };
        let t = ((1.0 - d) / edge).clamp(0.0, 1.0);
        let bg = if y > 330 { [70.0, 120.0, 60.0] } else { [110.0, 150.0, 200.0] };
        let c: [f32; 3] = std::array::from_fn(|i| bg[i] * (1.0 - t) + 145.0 * t);
        Rgb(c.map(|v| v as u8))
    })
}

fn artifact(run: &dyn RunCtx, kind: ArtifactKind) -> anyhow::Result<(Artifact, std::path::PathBuf)> {
    let (id, dir) = run.new_artifact_dir(kind)?;
    let rel = dir.strip_prefix(run.root())?.display().to_string();
    let a = Artifact { id, kind, path: rel, meta: BTreeMap::new() };
    run.register(a.clone())?;
    Ok((a, dir))
}

pub fn fake_dispatch(run: Arc<dyn RunCtx>) -> Dispatch {
    let rotation = std::env::var("SPLAT_FAKE_SCENARIO").as_deref() == Ok("rotation");
    let jobs: Arc<Mutex<HashMap<String, (String, Option<ToolResult>)>>> = Default::default();
    Arc::new(move |name: &str, args: Value| -> anyhow::Result<ToolResult> {
        let run = run.as_ref();
        let mut r = match name {
            "inspect_input" => {
                let (_, dir) = artifact(run, ArtifactKind::Images)?;
                let mut r = result(name, Status::Succeeded, json!({"duration_s": 9.2, "fps": 30, "width": 1920, "height": 1080,
                    "sharpness_p10_over_median": 0.9, "orientation_suspect": false, "static_frac": 0.0, "black_frac": 0.0}));
                for i in 0..6 {
                    let p = dir.join(format!("sample_{i:02}.jpg"));
                    scene(i * 3, false).save(&p)?;
                    r.images.push(p.strip_prefix(run.root())?.display().to_string());
                }
                r
            }
            "extract_frames" => {
                let mut r = result(name, Status::Succeeded, json!({"kept": args["target_count"].as_u64().unwrap_or(150)}));
                r.artifacts.push(artifact(run, ArtifactKind::Frames)?.0);
                r
            }
            "define_split" => {
                let mut r = result(name, Status::Succeeded, json!({"n_train": 131, "n_heldout": 19}));
                r.artifacts.push(artifact(run, ArtifactKind::Split)?.0);
                r
            }
            "reconstruct_sfm" | "train_splats" | "extract_mesh" => {
                let job_id = format!("job_{:02}", jobs.lock().unwrap().len() + 1);
                jobs.lock().unwrap().insert(job_id.clone(), (name.into(), None));
                let mut r = result(name, Status::Running, json!({}));
                r.job_id = Some(job_id);
                r
            }
            "job_status" => {
                let id = args["job_id"].as_str().unwrap_or("").to_string();
                let mut jobs = jobs.lock().unwrap();
                let Some((tool, done)) = jobs.get_mut(&id) else {
                    let mut r = result(name, Status::Failed, json!({}));
                    r.error = Some(ToolError { code: "unknown_job".into(), message: id, retryable: false, evidence: vec![] });
                    return Ok(r);
                };
                if done.is_none() {
                    let (kind, metrics) = match tool.as_str() {
                        "reconstruct_sfm" if rotation => (ArtifactKind::Sfm, json!({"n_registered": 118, "n_input": 150,
                            "registered_frac": 0.79, "reproj_mean_px": 0.9, "track_len_median": 2, "camera_extent": 0.04,
                            "points_median_dist_from_cameras": 9.5, "gates_failed": ["registered_frac", "track_len_median"]})),
                        "reconstruct_sfm" => (ArtifactKind::Sfm, json!({"n_registered": 150, "n_input": 150, "registered_frac": 1.0,
                            "reproj_mean_px": 0.41, "track_len_median": 3, "camera_extent": 3.1, "points_median_dist_from_cameras": 2.4,
                            "gates_failed": []})),
                        "train_splats" => (ArtifactKind::Candidate, json!({"steps": 7000, "splats": 512000})),
                        _ => (ArtifactKind::Mesh, json!({"faces": 812000, "components": 3})),
                    };
                    let mut r = result(tool, Status::Succeeded, metrics);
                    r.artifacts.push(artifact(run, kind)?.0);
                    *done = Some(r);
                }
                done.clone().unwrap()
            }
            "evaluate" | "render_views" => {
                let (_, dir) = artifact(run, ArtifactKind::Eval)?;
                let mut r = result(name, Status::Succeeded, json!({"psnr": 24.1, "ssim": 0.71, "lpips_vgg": 0.31, "cc_psnr": 24.5,
                    "heldout_scored": 19, "heldout_expected": 19}));
                let n = args["max_views"].as_u64().unwrap_or(8).min(8) as u32;
                for i in 0..n {
                    let mut pair = RgbImage::new(1920, 540);
                    image::imageops::replace(&mut pair, &scene(i * 8, false), 0, 0);
                    image::imageops::replace(&mut pair, &scene(i * 8, true), 960, 0);
                    let p = dir.join(format!("compare_{i:02}.jpg"));
                    pair.save(&p)?;
                    r.images.push(p.strip_prefix(run.root())?.display().to_string());
                }
                r
            }
            "stop_job" | "record_decision" => {
                if name == "record_decision" {
                    run.journal("decision_recorded", args)?;
                }
                result(name, Status::Succeeded, json!({}))
            }
            "finish" => {
                // Stand-in for core's finish builtin.
                let path = run.root().join("agent").join("finish.json");
                std::fs::create_dir_all(path.parent().unwrap())?;
                std::fs::write(&path, serde_json::to_vec_pretty(&args)?)?;
                run.journal("agent_finished", args)?;
                result(name, Status::Succeeded, json!({"recorded": true}))
            }
            _ => {
                let mut r = result(name, Status::Unsupported, json!({}));
                r.error = Some(ToolError { code: "unsupported_capability".into(), message: format!("fake backend has no {name}"),
                    retryable: false, evidence: vec![] });
                r
            }
        };
        r.tool = name.into();
        Ok(r)
    })
}
