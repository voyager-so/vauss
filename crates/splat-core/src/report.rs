use crate::{Run, RunCtx, atomic_json, run::atomic_write};
use anyhow::Result;
use serde_json::{Value, json};
use splat_contracts::{Artifact, ArtifactKind};

fn existing_file(run: &Run, path: &std::path::Path) -> Option<String> {
    let path = run.root().join(path).canonicalize().ok()?;
    if !path.is_file() {
        return None;
    }
    Some(
        path.strip_prefix(run.root())
            .ok()?
            .to_string_lossy()
            .into_owned(),
    )
}

fn splat_file(run: &Run, candidate: Option<&str>) -> Option<String> {
    let dir = run.path(candidate?).ok()?;
    if dir.extension().is_some_and(|ext| ext == "ply") {
        return existing_file(run, &dir);
    }
    if let Some(path) = existing_file(run, &dir.join("splat.ply")) {
        return Some(path);
    }
    let checkpoint: Value =
        serde_json::from_slice(&std::fs::read(dir.join("checkpoint.json")).ok()?).ok()?;
    if let Some(ply) = checkpoint["ply"].as_str() {
        return existing_file(run, &dir.join(ply));
    }
    existing_file(
        run,
        &std::path::Path::new(checkpoint["checkpoint"].as_str()?).join("splat.ply"),
    )
}

/// SfM alignment residual (rms camera-centre error / scene radius) above which a run warns.
pub const SFM_ALIGN_WARN: f64 = 0.10;

/// The capture pre-check's result in a run directory (`vauss run` checks the video while frames
/// extract; [`crate::remote`] holds the first upload for it).
pub const CAPTURE_CHECK: &str = "capture_check.json";
/// Present while that check is still running.
pub const CAPTURE_PENDING: &str = "capture_check.pending";

/// Derive both reports from durable evidence; compare candidates only on the same frozen split.
pub fn write_report(run: &Run) -> Result<Value> {
    let events = run.events()?;
    let state = run.state()?;
    let mut evaluations = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for event in &events {
        if event.event != "operation_planned" || event.data["tool"] != "evaluate" {
            continue;
        }
        let Some(id) = event.data["operation_id"].as_str() else {
            continue;
        };
        if !seen.insert(id) {
            continue;
        }
        if let Some(op) = state.operations.get(id) {
            if op["result"]["status"] == "succeeded" {
                evaluations.push(json!({"candidate_id":op["request"]["candidate_id"],"split_id":op["request"]["split_id"],"metrics":op["result"]["metrics"],"images":op["result"]["images"]}));
            }
        }
    }
    let latest_split = evaluations.last().map(|e| &e["split_id"]);
    let mut best = evaluations
        .iter()
        .filter(|e| Some(&e["split_id"]) == latest_split)
        .max_by(|a, b| {
            let score = |e: &Value| {
                e["metrics"]["psnr"]
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .unwrap_or(f64::NEG_INFINITY)
            };
            score(a).total_cmp(&score(b))
        });
    let attempt_start = events
        .iter()
        .rposition(|e| e.event == "run_started")
        .unwrap_or(0);
    let attempt = &events[attempt_start..];
    let finished = attempt
        .iter()
        .rev()
        .find(|e| e.event == "run_finished")
        .map(|e| e.data.clone())
        .unwrap_or(json!({}));
    // Stops never write run_finished; their reason lives in stop_requested (driver) or run_stopped (`vauss stop`).
    let stop_reason = finished
        .get("stop_reason")
        .filter(|v| !v.is_null())
        .cloned()
        .or_else(|| {
            attempt
                .iter()
                .rev()
                .find(|e| e.event == "stop_requested")
                .map(|e| e.data["reason"].clone())
        })
        .or_else(|| {
            attempt
                .iter()
                .any(|e| e.event == "run_stopped")
                .then(|| json!("user_stop"))
        })
        .unwrap_or(Value::Null);
    let finish = finished
        .get("finish")
        .filter(|v| v.is_object())
        .cloned()
        .or_else(|| {
            attempt
                .iter()
                .rev()
                .find(|e| e.event == "agent_finished")
                .map(|e| e.data.clone())
        })
        .unwrap_or(json!({}));
    let selected = finish["candidate_id"].as_str();
    // An edited candidate (remove_objects) is reported with its parent's held-out evaluation.
    let scored_via = finish["scored_via_parent"]["candidate_id"].as_str();
    if let Some(id) = scored_via.or(selected) {
        best = evaluations.iter().rev().find(|e| e["candidate_id"] == id);
    }
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut sfm = json!({});
    for event in &events {
        if !matches!(
            event.event.as_str(),
            "tool_result" | "job_completed" | "tool_error"
        ) {
            continue;
        }
        let r = &event.data["result"];
        for warning in r["warnings"].as_array().into_iter().flatten() {
            if !warnings.contains(warning) {
                warnings.push(warning.clone());
            }
        }
        if r["error"].is_object() && !errors.contains(&r["error"]) {
            errors.push(r["error"].clone());
        }
        if r["tool"] == "reconstruct_sfm" && r["status"] == "succeeded" {
            sfm = r["metrics"].clone();
        }
    }
    // Success needs the solve to place >= MIN_SOLVED_FRACTION of the frames.
    if let (Some(registered), Some(input)) = (sfm["n_registered"].as_u64(), sfm["n_input"].as_u64()) {
        let ok = input == 0 || registered as f64 >= crate::MIN_SOLVED_FRACTION * input as f64;
        sfm["coverage_ok"] = json!(ok);
        if !ok {
            warnings.push(json!(format!("Camera solve placed only {registered} of {input} frames ({:.0}%).", 100.0 * registered as f64 / input as f64)));
        }
    }
    // A video's full SfM mapped onto its first view's world (splat-cli sfm_align): the camera-centre residual.
    if let Some(event) = events.iter().rev().find(|e| e.event == "sfm_aligned") {
        let alignment = &event.data["alignment"];
        sfm["alignment"] = alignment.clone();
        if alignment["rms_over_radius"].as_f64().is_some_and(|r| r > SFM_ALIGN_WARN) {
            warnings.push(json!(format!("The final model's world differs from the first view's by {:.0}% of the scene radius (SfM alignment residual); keyframes set on the preview may shift.",
                100.0 * alignment["rms_over_radius"].as_f64().unwrap_or(0.0))));
        }
    }
    let summary = finish["explanation"].as_str().unwrap_or(if best.is_some() {
        "Best evaluated candidate selected by held-out PSNR within its frozen split."
    } else {
        "No successful evaluation is available."
    });
    let selected = selected.or_else(|| best.and_then(|e| e["candidate_id"].as_str()));
    // Brush returns the subject mesh (files[0], the deliverable) plus a whole-scene mesh.
    let meshes = events.iter().rev().find_map(|event| {
        if !matches!(event.event.as_str(), "tool_result" | "job_completed") {
            return None;
        }
        let result = &event.data["result"];
        if result["tool"] != "extract_mesh" || result["status"] != "succeeded" {
            return None;
        }
        if let Some(candidate) = selected {
            let belongs = result["artifacts"]
                .as_array()?
                .iter()
                .any(|a| a["meta"]["candidate_id"] == candidate);
            if !belongs {
                return None;
            }
        }
        let subject = result["metrics"]["files"][0]
            .as_str()
            .and_then(|f| existing_file(run, std::path::Path::new(f)));
        let full_scene = result["metrics"]["full_scene"]["files"][0]
            .as_str()
            .and_then(|f| existing_file(run, std::path::Path::new(f)));
        // Room captures can have no cropped subject. Keep their usable whole-scene mesh in
        // both the primary mesh deliverable and the explicit whole-scene slot.
        subject.or_else(|| full_scene.clone()).map(|primary| (primary, full_scene))
    });
    let (mesh, mesh_full_scene) = meshes.map_or((None, None), |(s, f)| (Some(s), f));
    // Input defects (e.g. low_resolution) stay visible even if the agent's finish omits them.
    let mut remaining_defects = finish.get("remaining_defects").and_then(Value::as_array).cloned().unwrap_or_default();
    let input_defects = events
        .iter()
        .rev()
        .find(|e| e.event == "tool_result" && e.data["result"]["tool"] == "inspect_input")
        .and_then(|e| e.data["result"]["metrics"]["defects"].as_array().cloned())
        .unwrap_or_default();
    if let Some(note) = finish["scored_via_parent"]["note"].as_str() {
        if !remaining_defects.iter().any(|d| d.as_str() == Some(note)) {
            remaining_defects.push(json!(note));
        }
    }
    for defect in input_defects {
        if let Some(text) = defect["message"].as_str() {
            if !remaining_defects.iter().any(|d| d.as_str() == Some(text)) {
                remaining_defects.push(json!(text));
            }
        }
    }
    // Which trainer recipe produced the best candidate (`--cloud` runs: the MAX preset).
    let preset = selected.and_then(|id| state.artifacts.get(id)).map(|a| {
        json!({"name": a.meta.get("preset").and_then(Value::as_str).unwrap_or("default"),
            "max_splats": a.meta.get("max_splats").or_else(|| a.meta.get("request").and_then(|r| r.get("cap_max"))),
            "total_steps": a.meta.get("request").and_then(|r| r.get("total_steps")),
            "max_budget": a.meta.get("max_budget").and_then(Value::as_bool).unwrap_or(false)})
    });
    // The capture pre-check `vauss run` makes before any upload (capture_check.json): its warnings
    // lead the report's, so the user sees how to film it better next time; its optional tips share one line.
    let capture_check = std::fs::read(run.root().join(CAPTURE_CHECK)).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    let findings: Vec<&Value> = capture_check.iter().flat_map(|c| c["checks"].as_array().into_iter().flatten()).collect();
    let tips: Vec<&str> = findings.iter().filter(|c| c["verdict"] == "tip").filter_map(|c| c["message"].as_str()).collect();
    if !tips.is_empty() {
        warnings.insert(0, json!(format!("Capture check tips (optional): {}", tips.join(" "))));
    }
    for c in findings.iter().filter(|c| c["verdict"] == "warn" || c["verdict"] == "fail").rev() {
        warnings.insert(0, json!(format!("Capture check: {} {}", c["message"].as_str().unwrap_or(""), c["fix"].as_str().unwrap_or("")).trim().to_string()));
    }
    // Why there is no mesh, stated loudly (a UI can show "Mesh skipped: <reason>."): the pipeline's
    // last mesh_skipped / mesh_failed journal event, or --no-mesh. Null whenever a mesh exists.
    let mesh_skipped = if mesh.is_some() {
        Value::Null
    } else if let Some(e) = events.iter().rev().find(|e| matches!(e.event.as_str(), "mesh_skipped" | "mesh_failed") && e.data["kind"].is_string()) {
        json!({"kind": e.data["kind"], "reason": e.data["reason"]})
    } else if std::fs::read(run.root().join("fixed.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).is_some_and(|f| f["no_mesh"] == true) {
        json!({"kind": "not_requested", "reason": "no mesh was requested (--no-mesh)"})
    } else {
        Value::Null
    };
    if let Some(reason) = mesh_skipped["reason"].as_str().filter(|_| mesh_skipped["kind"] != "not_requested") {
        warnings.insert(0, json!(format!("Mesh skipped: {reason}.")));
    }
    let deliverables = json!({"splat":splat_file(run, selected),"mesh":mesh,"mesh_full_scene":mesh_full_scene,"report_md":"report.md","report_json":"report.json"});
    let report = json!({
        "schema_version":1,"run_id":run.run_id(),"generated_at":crate::utc_now(),"source":run.source(),"goal":run.goal(),"backend":run.config().backend,
        "status":state.status,"stop_reason":stop_reason,"summary":summary,"remaining_defects":remaining_defects,
        "best_candidate":selected,"metrics":best.map(|e|e["metrics"].clone()).unwrap_or(json!({})),"sfm":sfm,"deliverables":deliverables,
        "images":best.map(|e|e["images"].clone()).unwrap_or(json!([])),"evaluations":evaluations,
        "artifacts":state.artifacts.values().collect::<Vec<_>>(),"warnings":warnings,"errors":errors,"jobs":state.jobs,
        "training_preset":preset,"cloud":crate::remote::cost(run.root()),"capture_check":capture_check,
        "mesh_skipped":mesh_skipped
    });
    let mut markdown = format!(
        "# Reconstruction {}\n\nStatus: **{}**{}\n\n{}\n\n{}\n\nBest candidate: {}\n\n## Deliverables\n\n",
        run.run_id(),
        state.status,
        report["stop_reason"]
            .as_str()
            .and_then(|r| r.lines().next())
            .filter(|r| !r.is_empty())
            .map(|r| format!(" (stop reason: {r})"))
            .unwrap_or_default(),
        run.goal(),
        summary,
        report["best_candidate"].as_str().unwrap_or("none")
    );
    if let Some(p) = report["training_preset"].as_object() {
        markdown = markdown.replacen(
            "\n\n## Deliverables\n\n",
            &format!(
                "\n\nTraining preset: **{}** ({} steps, {})\n\n## Deliverables\n\n",
                p["name"].as_str().unwrap_or("default"),
                p["total_steps"].as_u64().map_or("?".into(), |n| n.to_string()),
                match p["max_splats"].as_u64() {
                    Some(0) if p["max_budget"].as_bool() == Some(true) => "scene-adaptive splat budget, up to 8M".to_string(),
                    Some(0) => "scene-adaptive splat budget, up to 2M".to_string(),
                    Some(n) => format!("cap {n} splats"),
                    None => "cap ?".to_string(),
                }
            ),
            1,
        );
    }
    if let Some(defects) = report["remaining_defects"].as_array().filter(|d| !d.is_empty()) {
        let mut section = String::from("## Remaining defects\n\n");
        for d in defects.iter().filter_map(Value::as_str) {
            section.push_str(&format!("- {d}\n"));
        }
        section.push_str("\n## Deliverables\n\n");
        markdown = markdown.replacen("## Deliverables\n\n", &section, 1);
    }

    for (label, key) in [
        ("Splat", "splat"),
        ("Mesh", "mesh"),
        ("Mesh (whole scene)", "mesh_full_scene"),
        ("Report", "report_md"),
        ("Report JSON", "report_json"),
    ] {
        match report["deliverables"][key].as_str() {
            Some(path) => markdown.push_str(&format!("- {label}: [{path}]({path})\n")),
            None => markdown.push_str(&format!("- {label}: missing\n")),
        }
    }
    if let Some(c) = report["cloud"].as_object() {
        markdown.push_str(&format!(
            "\n## Cloud GPU cost\n\nfal {}: ${:.4} actual over {} GPU jobs (estimated ${:.2}, up to ${:.2}).\n",
            c["gpu"].as_str().unwrap_or("?"),
            c["actual_usd"].as_f64().unwrap_or(0.),
            c["gpu_jobs"].as_u64().unwrap_or(0),
            c["estimate_usd"].as_f64().unwrap_or(0.),
            c["estimate_high_usd"].as_f64().unwrap_or(0.)
        ));
        markdown.push_str("\n| stage | job | GPU | wall | cost |\n|---|---|---|---|---|\n");
        for st in c["stages"].as_array().into_iter().flatten() {
            markdown.push_str(&format!(
                "| {} | {} | {} | {} s | ${:.4} |\n",
                st["tool"].as_str().unwrap_or("?"),
                st["job_id"].as_str().unwrap_or("?"),
                st["gpu"].as_str().unwrap_or("?"),
                st["wall_s"].as_f64().map_or("?".into(), |w| w.to_string()),
                st["usd"].as_f64().unwrap_or(0.)
            ));
        }
    }
    markdown.push_str(&format!(
        "\n## Metrics\n\n```json\n{}\n```\n\n## Artifacts\n\n",
        serde_json::to_string_pretty(&report["metrics"])?
    ));
    for a in state.artifacts.values() {
        markdown.push_str(&format!(
            "- [{}]({}) ({})\n",
            a.id,
            a.path,
            serde_json::to_value(a.kind)?.as_str().unwrap()
        ));
    }
    if let Some(images) = report["images"].as_array().filter(|a| !a.is_empty()) {
        markdown.push_str("\n## Held-out images\n\n");
        for path in images.iter().filter_map(Value::as_str) {
            markdown.push_str(&format!("![comparison]({path})\n"));
        }
    }
    if !warnings.is_empty() || !errors.is_empty() {
        markdown.push_str("\n## Limitations\n\n");
        for w in &warnings {
            markdown.push_str(&format!("- {}\n", w.as_str().unwrap_or("")));
        }
        for e in &errors {
            markdown.push_str(&format!(
                "- {}: {}\n",
                e["code"].as_str().unwrap_or("error"),
                e["message"].as_str().unwrap_or("")
            ));
        }
    }
    run.with_lock(|| {
        atomic_json(run.root().join("report.json"), &report)?;
        atomic_write(&run.root().join("report.md"), markdown.as_bytes())?;
        for suffix in ["md", "json"] {
            run.register(Artifact {
                id: format!("report_{suffix}"),
                kind: ArtifactKind::Report,
                path: format!("report.{suffix}"),
                meta: Default::default(),
            })?;
        }
        Ok(())
    })?;
    Ok(report)
}
