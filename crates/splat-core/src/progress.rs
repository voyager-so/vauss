//! Observational stage progress shared by CLI and agent tools. Fractions are measured, never a
//! clock disguised as a work counter. ETAs describe this stage, not the rest of an agent's plan.
use crate::{Run, RunCtx, RunState, utc_now};
use anyhow::Result;
use serde_json::{Value, json};
use splat_contracts::Status;
use std::fs;

fn stage(tool: &str) -> Option<(&'static str, usize, &'static str)> {
    Some(match tool {
        "extract_frames" => ("extract_frames", 1, "Extracting frames"),
        "mask_distractors" => ("masks", 2, "Masking distractions"),
        "reconstruct_sfm" => ("sfm", 3, "Placing cameras (SfM)"),
        "train_splats" => ("train", 4, "Training splats"),
        "export" => ("export", 5, "Exporting / ready"),
        "evaluate" => ("evaluate", 6, "Evaluating held-out views"),
        "extract_mesh" => ("mesh", 6, "Extracting mesh (optional)"),
        _ => return None,
    })
}

/// Seeds: Oct 9 journals in runs/rust-agent/cloud-max/{garden-default,garden-max-r3,
/// gate-budget-s2}/run: 185 frames at 1600px, extraction 5-10s, SfM 284-719s, masks
/// 29-166s. L40S training: 7k/41.48s (1M), 30k/429-491s (MAX); evaluation 238-349s.
/// Other GPU classes use conservative relative throughput priors, not claimed measurements.
/// Queue, upload and download time is variable and explicitly excluded from these estimates.
fn estimate(run: &Run, state: &RunState, tool: &str, req: &Value) -> (f64, u64) {
    let frames = state.operations.values().filter(|o| o["tool"] == "extract_frames")
        .find(|o| req["frames_id"].is_null() || o["result"]["artifacts"].as_array()
            .is_some_and(|a| a.iter().any(|a| a["id"] == req["frames_id"])));
    let n = frames.and_then(|o| o["result"]["metrics"]["frame_count"].as_f64())
        .or_else(|| req["target_count"].as_f64()).unwrap_or(185.);
    let edge = frames.and_then(|o| o["result"]["metrics"]["max_edge"].as_f64())
        .or_else(|| req["max_edge"].as_f64()).unwrap_or(1600.);
    let pixels = n * (edge / 1600.).powi(2);
    let mut steps = req["total_steps"].as_u64().unwrap_or(7000);
    let preset = crate::remote::preset().filter(|_| run.config().backend == "brush" && !crate::remote::goal_wants_mesh(&run.config().goal));
    if let Some(p) = preset { steps = steps.max(p.min_steps_for(n as usize) as u64); }
    let cloud = crate::remote::mode().is_some();
    let gpu = std::env::var("SPLAT_FAL_GPU").unwrap_or_else(|_| if cloud { "l40" } else { "unknown" }.into());
    let rate = match gpu.to_lowercase().as_str() {
        "h100" | "a100" => 220.,
        "rtxpro6000" | "5090" => 240.,
        "l40" | "l40s" | "4090" => 170.,
        _ => 80.,
    } / if preset.is_some_and(|p| p.max_budget) { 2.6 } else { 1. };
    let start = req["resume_candidate_id"].as_str().and_then(|id| state.artifacts.get(id))
        .and_then(|a| fs::read(run.root().join(&a.path).join("checkpoint.json")).ok())
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v["step"].as_u64()).unwrap_or(0);
    let seconds = match tool {
        "extract_frames" => n * 0.04,
        "mask_distractors" => pixels * 0.4,
        "reconstruct_sfm" => pixels * 2.3,
        "train_splats" => steps.saturating_sub(start) as f64 / rate,
        "evaluate" => pixels * if req["lpips"] == false { 0.15 } else { 1.5 },
        "export" => 10.,
        _ => 60.,
    };
    (seconds.max(1.), steps)
}

pub fn begin(run: &Run, tool: &str, operation: &str, request: &Value) -> Result<()> {
    let Some((stage, index, label)) = stage(tool) else { return Ok(()) };
    let (eta, steps) = estimate(run, &run.state()?, tool, request);
    run.journal("progress", json!({"stage":stage,"index":index,"stages":6,"label":label,
        "tool":tool,"operation_id":operation,"status":"running","fraction":null,
        "eta_s":eta,"eta_source":"model","eta_scope":"stage_compute","estimate_s":eta,
        "total_steps":if tool == "train_splats" { Some(steps) } else { None },
        "started_at":utc_now(),"updated_at":utc_now(),"elapsed_s":0.}))
}

pub fn complete(run: &Run, operation: &str, status: Status) -> Result<()> {
    if status == Status::Running { return Ok(()) }
    let state = run.state()?;
    let mut p = state.progress;
    if p["operation_id"] != operation || p["status"] != "running" { return Ok(()) }
    if status == Status::Succeeded && p["stage"] == "train" {
        p["step"] = p["total_steps"].clone();
        if let Some(n) = state.operations.get(operation).and_then(|o| o["result"]["metrics"]["exported_splats"].as_u64()) {
            p["splats"] = json!(n);
        }
    }
    p["status"] = json!(status);
    if status == Status::Succeeded { p["fraction"] = json!(1.); }
    p["eta_s"] = if status == Status::Succeeded { json!(0.) } else { Value::Null };
    p["elapsed_s"] = json!(age(&p, "started_at"));
    p["updated_at"] = json!(utc_now());
    p.as_object_mut().unwrap().remove("note");
    p.as_object_mut().unwrap().remove("remote_status");
    run.journal("progress", p)
}

fn age(p: &Value, key: &str) -> f64 {
    p[key].as_str().and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|t| (chrono::Utc::now() - t.with_timezone(&chrono::Utc)).num_milliseconds().max(0) as f64 / 1000.)
        .unwrap_or(0.)
}

/// Lock-free status/display projection. Expired estimates become unknown, never "0s left" while
/// the stage is still running. A missing remote step stream is explicitly described.
pub fn snapshot(mut p: Value, run_status: &str) -> Value {
    if p.is_null() { return p }
    if p["status"] == "running" {
        p["elapsed_s"] = json!(age(&p, "started_at"));
        if matches!(run_status, "failed" | "cancelled" | "stopping") {
            p["status"] = json!(run_status);
            p["eta_s"] = Value::Null;
        } else if let Some(eta) = p["eta_s"].as_f64() {
            let remaining = eta - age(&p, "updated_at");
            p["eta_s"] = if remaining > 0. { json!(remaining) } else { Value::Null };
        }
    }
    p
}

/// Called by the CLI observer, independent of how frequently an agent polls job_status.
/// Never finalizes a job or changes pipeline decisions. A nonblocking supervisor lock avoids waiting on
/// synchronous tools; begin/complete still record their boundary events.
pub fn refresh(run: &Run) -> Result<()> {
    run.try_with_lock(|| refresh_locked(run))?;
    Ok(())
}

fn refresh_locked(run: &Run) -> Result<()> {
    let state = run.state()?;
    let old = &state.progress;
    if old["status"] != "running" || age(old, "updated_at") < 2.
        || matches!(state.status.as_str(), "stopping" | "cancelled" | "failed") { return Ok(()) }
    let mut p = snapshot(old.clone(), &state.status);
    let job = state.jobs.iter().find(|(_, j)| j["operation_id"] == old["operation_id"] && j["status"] == "running");
    if let Some((id, _)) = job {
        p["job_id"] = json!(id);
        let dir = run.root().join("jobs").join(id);
        if let Ok(v) = fs::read(dir.join("remote-progress.json")) {
            if let Ok(v) = serde_json::from_slice::<Value>(&v) {
                p["remote_status"] = v["status"].clone();
                p["note"] = json!("Cloud queue and transfer time excluded; live steps unavailable");
                if v["status"] == "IN_PROGRESS" && old["remote_status"] != "IN_PROGRESS" {
                    p["eta_s"] = p["estimate_s"].clone();
                }
                if matches!(v["status"].as_str(), Some("STAGING" | "IN_QUEUE" | "FETCHING_OUTPUTS")) {
                    p["eta_s"] = Value::Null;
                }
            }
        }
        if old["stage"] == "train" {
            let metrics = run.registry.backend(&run.config().backend)?.job_progress(run, &dir);
            if let (Some(step), Some(total)) = (metrics["step"].as_u64(), metrics["total_steps"].as_u64()) {
                if total > 0 {
                    p.as_object_mut().unwrap().remove("note");
                    p["step"] = json!(step);
                    p["total_steps"] = json!(total);
                    p["fraction"] = json!((step as f64 / total as f64).min(1.));
                    p["splats"] = metrics["splats"].clone();
                    // Backend ETA uses measured steps / training seconds and handles checkpoint
                    // start offsets. Reuse only on a NEW step, so stale logs do not reset the clock.
                    if old["step"].as_u64() != Some(step) {
                        p["eta_s"] = metrics["eta_s"].clone();
                        p["eta_source"] = json!("measured");
                        if let Some(eta) = metrics["eta_s"].as_f64().filter(|v| *v > 0.) {
                            p["it_s"] = json!(total.saturating_sub(step) as f64 / eta);
                        }
                    }
                    if step >= total {
                        p["eta_s"] = Value::Null;
                        p["note"] = json!("Steps complete; finalizing checkpoint and outputs");
                    }
                }
            }
        }
    }
    p["updated_at"] = json!(utc_now());
    run.with_lock(|| {
        // A tool may have finished or another stage started while logs were being read.
        if run.state()?.progress == *old { run.journal("progress", p)?; }
        Ok(())
    })
}
