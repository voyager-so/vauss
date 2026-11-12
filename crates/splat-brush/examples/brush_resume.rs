//! Dev harness: interrupt a training job in an existing run, then resume it from its checkpoint.
//! `brush_resume RUN_DIR SFM_ID SPLIT_ID [STEPS] [STOP_AFTER_S]`

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use splat_brush::{BrushBackend, BrushConfig};
use splat_contracts::*;
use splat_core::{Backend, Registry, Run, RunCtx};

fn main() -> anyhow::Result<()> {
    if let Some(code) = splat_core::handle_internal_command() {
        std::process::exit(code?);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("__brush") {
        std::process::exit(splat_brush::worker_main(args[1..].to_vec())?);
    }
    let (root, sfm, split) = (&args[0], &args[1], &args[2]);
    let steps: u32 = args.get(3).map_or(Ok(3000), |s| s.parse())?;
    let stop_after: u64 = args.get(4).map_or(Ok(15), |s| s.parse())?;
    let backend = Arc::new(BrushBackend::new(BrushConfig::from_env()));
    let mut registry = Registry::new();
    registry.register_backend(backend.clone());
    let run = Run::open_with_registry(root, Arc::new(registry), true)?;
    let req: TrainSplats = serde_json::from_value(json!({"sfm_id": sfm, "split_id": split, "total_steps": steps}))?;
    let first = backend.train_splats(&run, req)?;
    let (job, cand) = (first.job_id.clone().unwrap(), first.artifacts[0].id.clone());
    std::thread::sleep(Duration::from_secs(stop_after));
    let progress = run.jobs().status(&job, 0.0)?;
    println!("before stop: {}", serde_json::to_string(&progress.metrics)?);
    let stopped = run.jobs().stop(&job, 90.0)?;
    println!("stopped: {:?} error={:?} metrics={}", stopped.status, stopped.error.as_ref().map(|e| &e.code), serde_json::to_string(&stopped.metrics)?);
    let req: TrainSplats = serde_json::from_value(json!({"sfm_id": sfm, "split_id": split, "total_steps": steps, "resume_candidate_id": cand}))?;
    let second = backend.train_splats(&run, req)?;
    println!("resume: {:?} warnings={:?}", second.status, second.warnings);
    let done = run.jobs().wait(second.job_id.as_ref().unwrap(), 900.0)?;
    let keys = ["psnr", "ssim", "lpips_vgg", "train_psnr", "training_s", "exported_splats"];
    let m: Vec<_> = keys.iter().map(|k| format!("{k}={}", done.metrics.get(*k).cloned().unwrap_or_default())).collect();
    println!("resumed run: {:?} {}", done.status, m.join(" "));
    let series = std::fs::read_to_string(run.path(&done.artifacts[0].id)?.join("training.jsonl"))?;
    println!("first rows of resumed series:\n{}", series.lines().take(2).collect::<Vec<_>>().join("\n"));
    Ok(())
}
