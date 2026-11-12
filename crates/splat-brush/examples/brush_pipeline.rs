//! Dev harness: drive the brush backend through a real `Run` without an LLM.
//! `cargo run --release -p splat-brush --example brush_pipeline -- RUN_DIR SOURCE [STEPS] [FRAMES]`

use std::sync::Arc;

use serde_json::{Value, json};
use splat_brush::{BrushBackend, BrushConfig};
use splat_contracts::*;
use splat_core::{Backend, Registry, Run};

fn show(r: &ToolResult) {
    let mut m = r.metrics.clone();
    for k in ["per_image", "cameras", "unregistered_runs", "worst_views", "views", "gates"] {
        if let Some(v) = m.get_mut(k) {
            if k == "gates" {
                continue;
            }
            *v = json!("…");
        }
    }
    println!(
        "{} {:?} wall={:?} job={:?}\n  metrics={}\n  images={} warnings={:?} error={:?}",
        r.tool, r.status, r.wall_s, r.job_id, Value::Object(m.into_iter().collect()), r.images.len(), r.warnings, r.error
    );
}

fn finish(run: &Run, r: ToolResult) -> anyhow::Result<ToolResult> {
    show(&r);
    let Some(id) = r.job_id.clone() else { return Ok(r) };
    loop {
        let s = run.jobs().wait(&id, 30.0)?;
        if s.status != Status::Running {
            show(&s);
            return Ok(s);
        }
        println!("  … {}", Value::Object(s.metrics.clone().into_iter().collect()));
    }
}

fn de<T: serde::de::DeserializeOwned>(v: Value) -> serde_json::Result<T> {
    serde_json::from_value(v)
}

fn main() -> anyhow::Result<()> {
    if let Some(code) = splat_core::handle_internal_command() {
        std::process::exit(code?);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("__brush") {
        std::process::exit(splat_brush::worker_main(args[1..].to_vec())?);
    }
    let (root, source) = (&args[0], &args[1]);
    let steps: u32 = args.get(2).map_or(Ok(7000), |s| s.parse())?;
    let frames: u32 = args.get(3).map_or(Ok(150), |s| s.parse())?;
    let backend = Arc::new(BrushBackend::new(BrushConfig::from_env()));
    let mut registry = Registry::new();
    registry.register_backend(backend.clone());
    let run = Run::create(root, source, "dev pipeline", "brush")?.with_registry(Arc::new(registry));

    finish(&run, backend.inspect_input(&run, de(json!({}))?)?)?;
    let fr = finish(&run, backend.extract_frames(&run, de(json!({"target_count": frames}))?)?)?;
    let frames_id = fr.artifacts[0].id.clone();
    let sfm = finish(&run, backend.reconstruct_sfm(&run, de(json!({"frames_id": frames_id}))?)?)?;
    let sfm_id = sfm.artifacts[0].id.clone();
    let split = finish(&run, backend.define_split(&run, de(json!({"sfm_id": sfm_id}))?)?)?;
    let split_id = split.artifacts[0].id.clone();
    let tr = finish(&run, backend.train_splats(&run, de(json!({"sfm_id": sfm_id, "split_id": split_id, "total_steps": steps}))?)?)?;
    let cand = tr.artifacts[0].id.clone();
    finish(&run, backend.evaluate(&run, de(json!({"candidate_id": cand, "split_id": split_id}))?)?)?;
    finish(&run, backend.render_views(&run, de(json!({"candidate_id": cand, "views": "orbit", "max_views": 8}))?)?)?;
    finish(&run, backend.render_views(&run, de(json!({"candidate_id": cand, "views": "training_sample", "max_views": 4}))?)?)?;
    Ok(())
}
