use crate::{
    Run, RunCtx, atomic_json,
    jobs::{failure, result},
};
use anyhow::{Context, Result, ensure};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use splat_contracts::*;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct InvalidRequest(String);

fn decode<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).map_err(|e| InvalidRequest(e.to_string()).into())
}
fn normalize<T: DeserializeOwned + Serialize>(value: Value) -> Result<Value> {
    Ok(serde_json::to_value(decode::<T>(value)?)?)
}

/// Static background the held-out views always show: never a training-ignore class, or it is never
/// trained (on one outdoor clip, sky masked out of training scored ~16 vs 23 dB). Fine for SfM features.
const STATIC_MASK_CLASSES: &[&str] = &["sky", "cloud", "building", "ground", "road", "street", "wall", "floor", "ceiling"];

/// Drops static classes from a mask request that reaches the training loss: with other classes
/// left the prompt keeps only those; with none left, "both" becomes SfM-only and
/// "training_ignore" is refused.
fn strip_static_mask_classes(value: &mut Value) -> Result<()> {
    if value["purpose"] == "sfm_features" {
        return Ok(());
    }
    let prompt = value["prompt"].as_str().unwrap_or_default().to_owned();
    let (fixed, moving): (Vec<&str>, Vec<&str>) = prompt
        .split([';', ',', '.'])
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .partition(|t| {
            let t = t.to_lowercase();
            let t = t.strip_suffix('s').unwrap_or(&t);
            STATIC_MASK_CLASSES.contains(&t)
        });
    if fixed.is_empty() {
        return Ok(());
    }
    if !moving.is_empty() {
        value["prompt"] = moving.join("; ").into();
    } else if value["purpose"] == "both" {
        value["purpose"] = "sfm_features".into();
    } else {
        return Err(InvalidRequest(format!(
            "{} is static background, not a distractor: training must keep it (use purpose sfm_features)",
            fixed.join("; ")
        ))
        .into());
    }
    Ok(())
}

fn request(name: &str, payload: Value) -> Result<Value> {
    let mut value = match name {
        "inspect_input" => normalize::<InspectInput>(payload),
        "extract_frames" => normalize::<ExtractFrames>(payload),
        "mask_distractors" => normalize::<MaskDistractors>(payload),
        "reconstruct_sfm" => normalize::<ReconstructSfm>(payload),
        "define_split" => normalize::<DefineSplit>(payload),
        "train_splats" => normalize::<TrainSplats>(payload),
        "evaluate" => normalize::<Evaluate>(payload),
        "render_views" => normalize::<RenderViews>(payload),
        "extract_mesh" => normalize::<ExtractMesh>(payload),
        "plan_shot" => normalize::<PlanShot>(payload),
        "render_shot" => normalize::<RenderShot>(payload),
        "generate_shot" => normalize::<GenerateShot>(payload),
        "export_blender" => normalize::<ExportBlender>(payload),
        "select_objects" => normalize::<SelectObjects>(payload),
        "remove_objects" => normalize::<RemoveObjects>(payload),
        "inpaint_removed" => normalize::<InpaintRemoved>(payload),
        "job_status" => normalize::<JobStatus>(payload),
        "stop_job" => normalize::<StopJob>(payload),
        "record_decision" => normalize::<RecordDecision>(payload),
        "finish" => normalize::<Finish>(payload),
        _ => Err(InvalidRequest(format!("Unknown tool: {name}")).into()),
    }?;
    let check = |ok: bool, message: &str| -> Result<()> {
        if ok {
            Ok(())
        } else {
            Err(InvalidRequest(message.into()).into())
        }
    };
    for (field, min, max) in [
        ("sample_count", 1., 48.),
        ("target_count", 20., 1500.),
        ("max_edge", 480., 4096.),
        ("interval", 2., 50.),
        ("total_steps", 200., 60000.),
        ("eval_every", 1., 4_294_967_295.),
        ("cap_max", 10000., 10000000.),
        ("resolution_divisor", 1., 8.),
        ("texture_size", 256., 8192.),
        ("wait_s", 0., 600.),
        ("max_views", 1., 64.),
        ("threshold", 0.1, 0.95),
        ("steps", 200., 10000.),
        ("height_offset", -2., 2.),
    ] {
        if field == "target_count" && value["preserve_all"] == true { continue; }
        if let Some(n) = value.get(field).and_then(Value::as_f64) {
            check(
                n.is_finite() && n >= min && n <= max,
                &format!("{field} must be between {min} and {max}"),
            )?;
        }
    }
    if name == "mask_distractors" {
        strip_static_mask_classes(&mut value)?;
    }
    if name == "extract_frames" {
        check(
            matches!(value["rotate_deg"].as_u64(), Some(0 | 90 | 180 | 270)),
            "rotate_deg must be 0, 90, 180 or 270",
        )?;
        if let Some(range) = value["time_range_s"].as_array() {
            check(
                range[0].as_f64().is_some_and(|a| a >= 0.) && range[1].as_f64() > range[0].as_f64(),
                "time_range_s must be nonnegative and increasing",
            )?;
        }
    }
    if name == "train_splats" {
        check(
            value["init_candidate_id"].is_null() || value["resume_candidate_id"].is_null(),
            "init_candidate_id and resume_candidate_id are mutually exclusive",
        )?;
    }
    if name == "render_views" {
        check(value["max_views"].as_f64().is_some_and(|n| n <= 48.), "max_views must be between 1 and 48")?;
    }
    if name == "select_objects" {
        check(value["max_views"].as_f64().is_some_and(|n| n >= 4.), "max_views must be between 4 and 64")?;
        check(
            value["prompt"].as_str().is_some_and(|s| !s.trim().is_empty() && s.split_whitespace().count() <= 12),
            "prompt must be short object nouns (\"bench\"; \"red car\"), not a sentence",
        )?;
    }
    if name == "plan_shot" {
        check(
            value["duration_s"]
                .as_f64()
                .is_some_and(|n| n > 0. && n <= 60.),
            "duration_s must be greater than 0 and at most 60",
        )?;
    }
    if name == "generate_shot" {
        check(
            value["models"].as_array().is_some_and(|m| {
                (1..=6).contains(&m.len())
                    && m.iter()
                        .all(|v| v.as_str().is_some_and(|s| !s.trim().is_empty()))
            }),
            "models must contain 1 to 6 nonempty model names",
        )?;
    }
    if name == "finish" || name == "record_decision" {
        check(
            value["explanation"]
                .as_str()
                .is_some_and(|s| !s.trim().is_empty()),
            "explanation must not be empty",
        )?;
    }
    Ok(value)
}

fn canonical(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: std::collections::BTreeMap<_, _> = map.into_iter().collect();
            Value::Object(sorted.into_iter().map(|(k, v)| (k, canonical(v))).collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical).collect()),
        other => other,
    }
}

fn invoke(run: &Run, name: &str, payload: Value) -> Result<ToolResult> {
    if name == "record_decision" {
        let mut decision = payload.clone();
        decision["text"] = payload["explanation"].clone();
        run.journal("decision", decision)?;
        run.journal("decision_recorded", payload)?;
        return Ok(result(name, Status::Succeeded));
    }
    if name == "finish" {
        return finish(run, decode(payload)?);
    }
    if name == "remove_objects" {
        return crate::edit::remove_objects(run, decode(payload)?);
    }
    if SHOT_TOOLS.contains(&name) {
        let Some(shots) = &run.registry.shots else {
            let mut r = failure(
                name,
                "unsupported_capability",
                "Shot tools are not installed",
            );
            r.status = Status::Unsupported;
            return Ok(r);
        };
        return match name {
            "plan_shot" => shots.plan_shot(run, decode(payload)?),
            "render_shot" => shots.render_shot(run, decode(payload)?),
            "generate_shot" => shots.generate_shot(run, decode(payload)?),
            "export_blender" => shots.export_blender(run, decode(payload)?),
            _ => unreachable!(),
        };
    }
    let backend = run.registry.backend(&run.config().backend)?;
    match name {
        "inspect_input" => backend.inspect_input(run, decode(payload)?),
        "extract_frames" => backend.extract_frames(run, decode(payload)?),
        "mask_distractors" => backend.mask_distractors(run, decode(payload)?),
        "reconstruct_sfm" => backend.reconstruct_sfm(run, decode(payload)?),
        "define_split" => backend.define_split(run, decode(payload)?),
        "train_splats" => backend.train_splats(run, decode(payload)?),
        "evaluate" => backend.evaluate(run, decode(payload)?),
        "render_views" => backend.render_views(run, decode(payload)?),
        "extract_mesh" => backend.extract_mesh(run, decode(payload)?),
        "select_objects" => backend.select_objects(run, decode(payload)?),
        "inpaint_removed" => backend.inpaint_removed(run, decode(payload)?),
        _ => unreachable!(),
    }
}

/// The candidate whose held-out evaluation speaks for `id`: itself, or for an edited candidate the
/// nearest unedited ancestor.
fn evaluated_ancestor(run: &Run, id: &str) -> Result<String> {
    let mut id = id.to_string();
    for _ in 0..16 {
        let artifact = run.artifact(&id)?;
        match artifact.meta.get("parent_candidate_id").and_then(Value::as_str) {
            Some(parent) if artifact.meta.contains_key("edit") => id = parent.to_string(),
            _ => return Ok(id),
        }
    }
    anyhow::bail!("candidate edit chain is too long")
}

/// Success needs the camera solve to place at least this fraction of the frames; below it a
/// run that trained anyway is `partial`, never `succeeded` (clips with 3/37 and 129/242 frames
/// placed used to finish `succeeded` with only a gate warning).
pub const MIN_SOLVED_FRACTION: f64 = 0.9;

/// The camera solve under `candidate_id` (through edits to the evaluated candidate): (registered,
/// input) frames from its SfM artifact's health.json. None when the candidate names no SfM or the
/// health is missing.
pub fn sfm_coverage(run: &Run, candidate_id: &str) -> Option<(u64, u64)> {
    let scored = evaluated_ancestor(run, candidate_id).ok()?;
    let sfm = run.artifact(&scored).ok()?.meta.get("sfm_id")?.as_str()?.to_owned();
    let health: Value = crate::run::read_json(&run.path(&sfm).ok()?.join("health.json")).ok()?;
    Some((health["n_registered"].as_u64()?, health["n_input"].as_u64()?))
}

/// The defect a run must report when its camera solve covers too few frames.
pub fn sfm_coverage_defect(run: &Run, candidate_id: &str) -> Option<String> {
    let (registered, input) = sfm_coverage(run, candidate_id)?;
    (input > 0 && (registered as f64) < MIN_SOLVED_FRACTION * input as f64).then(|| format!(
        "Camera solve placed only {registered} of {input} frames ({:.0}%); the scene is built from those frames only and unplaced views may be missing or blurry.",
        100.0 * registered as f64 / input as f64))
}

fn finish(run: &Run, req: Finish) -> Result<ToolResult> {
    let finish_path = run.root().join("agent/finish.json");
    let prior: Option<Value> = if finish_path.exists() {
        Some(crate::run::read_json(&finish_path)?)
    } else {
        None
    };
    if let Some(prior) = &prior {
        let mut request = prior.clone();
        request
            .as_object_mut()
            .context("Invalid finish record")?
            .remove("finished_at");
        request.as_object_mut().unwrap().remove("scored_via_parent");
        ensure!(
            request == serde_json::to_value(&req)?,
            "finish was already called"
        );
    }
    if matches!(req.status, RunStatus::Succeeded | RunStatus::Partial) {
        let id = req
            .candidate_id
            .as_deref()
            .context("succeeded/partial require candidate_id")?;
        ensure!(
            run.artifact(id)?.kind == ArtifactKind::Candidate,
            "candidate_id must refer to a candidate"
        );
        run.path(id)?;
        if req.status == RunStatus::Succeeded {
            let state = run.state()?;
            // An edited candidate (remove_objects) can't match held-out views that still show the
            // removed object: its parent's held-out evaluation stands in, and finish says so.
            let scored = evaluated_ancestor(run, id)?;
            let evaluated = state.operations.values().any(|op| {
                op["tool"] == "evaluate"
                    && op["request"]["candidate_id"] == scored.as_str()
                    && op["result"]["status"] == "succeeded"
                    && op["result"]["metrics"]["psnr"]
                        .as_f64()
                        .is_some_and(f64::is_finite)
            });
            ensure!(
                evaluated,
                "succeeded requires a successful held-out evaluation of the selected candidate"
            );
            if let Some(defect) = sfm_coverage_defect(run, id) {
                anyhow::bail!("succeeded requires the camera solve to place at least {:.0}% of the frames: {defect} Finish with status partial and list it in remaining_defects.", 100.0 * MIN_SOLVED_FRACTION);
            }
        }
    }
    let state = run.state()?;
    ensure!(
        !state.jobs.values().any(|job| job["status"] == "running"),
        "Cannot finish while jobs are running; wait for or stop them first"
    );
    let mut value = prior.unwrap_or(serde_json::to_value(&req)?);
    if value.get("finished_at").is_none() {
        value["finished_at"] = json!(crate::utc_now());
        if let Some(id) = req.candidate_id.as_deref() {
            let scored = evaluated_ancestor(run, id)?;
            if scored != id {
                value["scored_via_parent"] = json!({"candidate_id": scored, "note": format!(
                    "{id} is an edited candidate and was not scored itself; the held-out numbers are those of {scored}, before the edit."
                )});
            }
        }
    }
    std::fs::create_dir_all(finish_path.parent().unwrap())?;
    atomic_json(finish_path, &value)?;
    let events = run.events()?;
    if !events
        .iter()
        .any(|e| e.event == "agent_finished" && e.data == value)
    {
        run.journal("agent_finished", value.clone())?;
    }
    if !events
        .iter()
        .any(|e| e.event == "run_finished" && e.data["finish"] == value)
    {
        run.journal(
            "run_finished",
            json!({"status":req.status,"finish":value,"stop_reason":req.explanation}),
        )?;
    }
    let mut r = result("finish", Status::Succeeded);
    r.metrics.insert("run_status".into(), json!(req.status));
    Ok(r)
}

fn dispatch_inner(run: &Run, name: &str, payload: Value) -> Result<ToolResult> {
    let mut normalized = request(name, payload)?;
    let mut floor_warning = None;
    let mut auto_rotated = false;
    if name == "extract_frames" {
        if let Some(r) = crate::resolution::guard_extract(run) {
            run.journal("tool_error", json!({"tool":name,"result":r}))?;
            return Ok(r);
        }
        if normalized["rotate_deg"] == json!(0) && crate::orientation::source_upside_down(run) {
            normalized["rotate_deg"] = json!(180);
            auto_rotated = true;
        }
        let edge = crate::resolution::WARN_LONG_EDGE;
        // Quick first-view frames (`preview`) keep their edge: the full extraction keeps the floor.
        if normalized["max_edge"].as_u64().is_some_and(|e| e < u64::from(edge)) && normalized["preview"] != true && !crate::resolution::overridden() {
            floor_warning = Some(format!(
                "max_edge {} raised to {edge} px: smaller frames make camera solving unreliable (override with --allow-low-resolution).",
                normalized["max_edge"]
            ));
            normalized["max_edge"] = json!(edge);
        }
    }
    if name == "job_status" {
        let req: JobStatus = decode(normalized)?;
        return run.jobs().status(&req.job_id, req.wait_s);
    }
    if name == "stop_job" {
        let req: StopJob = decode(normalized)?;
        return run.jobs().stop(&req.job_id, 60.);
    }
    let operation_id = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&canonical(json!([
            run.config().backend,
            name,
            normalized
        ])))?)
    )[..24]
        .to_string();
    let (cached, job_id) = run.with_lock(|| {
        let state = run.state()?;
        ensure!(
            !matches!(state.status.as_str(), "stopping" | "cancelled"),
            "Run was stopped; use resume before scheduling more work"
        );
        for (id, job) in &state.jobs {
            if job["operation_id"] == operation_id && crate::jobs::replayable(&job["status"]) {
                return Ok((None, Some(id.clone())));
            }
        }
        // Only a success is replayed: a failed or unsupported result re-runs on an identical request,
        // since what made it fail (an unregistered artifact, a fal outage) may since have been fixed.
        if let Some(previous) = state
            .operations
            .get(&operation_id)
            .and_then(|o| o.get("result"))
            .filter(|r| r["status"] == "succeeded")
        {
            let r: ToolResult = serde_json::from_value(previous.clone())?;
            if name == "finish" && !run.root().join("agent/finish.json").exists() {
                return Ok((Some(invoke(run, name, normalized.clone())?), None));
            }
            for a in &r.artifacts {
                run.path(&a.id)?;
            }
            return Ok((Some(r), None));
        }
        run.journal(
            "operation_planned",
            json!({"operation_id":operation_id,"tool":name,"request":normalized}),
        )?;
        run.journal(
            "tool_call",
            json!({"operation_id":operation_id,"tool":name,"request":normalized}),
        )?;
        crate::progress::begin(run, name, &operation_id, &normalized)?;
        let mut r = run.in_operation(&operation_id, || {
            if is_job_tool(name) {
                start_tool_job(run, name, &operation_id, normalized)
            } else {
                invoke(run, name, normalized)
            }
        })?;
        if name == "inspect_input" {
            r = crate::resolution::guard_inspect(run, r);
            r = crate::orientation::judge_inspect(run, r);
        }
        if auto_rotated && r.status != Status::Failed {
            r.metrics.insert("auto_rotated_deg".into(), json!(180));
            if let Some(plan) = r.metrics.get_mut("frame_plan").filter(|p| p.is_object()) {
                plan["auto_rotated_deg"] = json!(180);
            }
            r.warnings.push("Source looks upside down (sky at the bottom in inspect samples): frames were rotated 180 degrees.".into());
        }
        r.warnings.extend(floor_warning);
        ensure!(
            r.schema_version == SCHEMA_VERSION && r.tool == name,
            "Backend returned a mismatched tool result"
        );
        if r.status == Status::Running {
            let id = r
                .job_id
                .as_ref()
                .context("Running result requires a job_id")?;
            ensure!(
                run.state()?
                    .jobs
                    .get(id)
                    .is_some_and(|j| j["operation_id"] == operation_id),
                "Running result refers to a job not owned by this operation"
            );
        }
        for artifact in &r.artifacts {
            run.register(artifact.clone())?;
        }
        run.journal(
            "tool_result",
            json!({"operation_id":operation_id,"tool":name,"result":r}),
        )?;
        crate::progress::complete(run, &operation_id, r.status)?;
        Ok((Some(r), None))
    })?;
    if let Some(r) = cached {
        return Ok(r);
    }
    run.jobs().status(&job_id.unwrap(), 0.)
}

/// Synchronous backend tools that can wait on the GPU lock or run for many minutes (renders + LPIPS).
/// They run as detached jobs so the caller gets a job_id at once and polls job_status.
pub const JOB_TOOLS: &[&str] = &["evaluate"];

/// Cloud extraction stays on the client's CPU, in a detached job: its full-video decode must not
/// hold the supervisor lock while a quick SfM finishes and the first view starts training.
/// render_views on a cloud run goes to fal like every other GPU job.
fn is_job_tool(name: &str) -> bool {
    JOB_TOOLS.contains(&name) || (matches!(name, "extract_frames" | "render_views") && crate::remote::mode().is_some())
}

fn start_tool_job(run: &Run, name: &str, operation_id: &str, payload: Value) -> Result<ToolResult> {
    let dir = run.root().join("jobs").join("tool-inputs");
    std::fs::create_dir_all(&dir)?;
    let input = dir.join(format!("{operation_id}.json"));
    let output = dir.join(format!("{operation_id}.result.json"));
    atomic_json(&input, &payload)?;
    let s = |p: &std::path::Path| p.to_string_lossy().into_owned();
    let mut meta = std::collections::BTreeMap::new();
    meta.insert("result_path".into(), json!(s(&output)));
    let id = run.jobs().start(JobSpec {
        tool: name.into(),
        argv: vec![s(&run.worker_exe), "__tool_job".into(), s(run.root()), name.into(), s(&input), s(&output)],
        env: Default::default(),
        cwd: None,
        on_complete: "core:tool_result".into(),
        meta,
    })?;
    let mut r = result(name, Status::Running);
    r.job_id = Some(id);
    Ok(r)
}

/// Worker side of a JOB_TOOLS job: run the backend tool synchronously in this process and write its
/// result for the core:tool_result completion handler. Called by the CLI's `__tool_job` command.
pub fn run_tool_job(run: &Run, name: &str, input: &std::path::Path, output: &std::path::Path) -> Result<()> {
    ensure!(JOB_TOOLS.contains(&name) || matches!(name, "extract_frames" | "render_views"), "{name} is not a job tool");
    let payload: Value = crate::run::read_json(input)?;
    let r = match invoke(run, name, payload) {
        Ok(r) => r,
        Err(e) => failure(name, "tool_failed", format!("{e:#}")),
    };
    atomic_json(output, &r)
}

/// Shared CLI, driver and MCP boundary. Errors are structured; every request is validated and journaled.
pub fn dispatch(run: &Run, name: &str, payload: Value) -> ToolResult {
    match dispatch_inner(run, name, payload) {
        Ok(result) => result,
        Err(error) => {
            let code = if error.downcast_ref::<InvalidRequest>().is_some() {
                "invalid_request"
            } else {
                "tool_failed"
            };
            let mut r = failure(name, code, format!("{error:#}"));
            if let Ok(state) = run.state() {
                if state.progress["tool"] == name {
                    if let Some(op) = state.progress["operation_id"].as_str() {
                        let _ = crate::progress::complete(run, op, Status::Failed);
                    }
                }
            }
            if let Err(e) = run.journal("tool_error", json!({"tool":name,"result":r})) {
                r.warnings.push(format!("Could not journal error: {e}"));
            }
            r
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, FakeBackend, Registry};
    use std::{path::Path, sync::Arc};

    /// FakeBackend whose define_split reports an unknown SfM as a failed result (as Brush's guard
    /// does) instead of an error, so the failure is journaled as a tool_result.
    struct Guarded(FakeBackend);
    impl Backend for Guarded {
        fn name(&self) -> &'static str { "guarded" }
        fn capabilities(&self) -> Value { self.0.capabilities() }
        fn inspect_input(&self, run: &dyn RunCtx, req: InspectInput) -> Result<ToolResult> { self.0.inspect_input(run, req) }
        fn extract_frames(&self, run: &dyn RunCtx, req: ExtractFrames) -> Result<ToolResult> { self.0.extract_frames(run, req) }
        fn mask_distractors(&self, run: &dyn RunCtx, req: MaskDistractors) -> Result<ToolResult> { self.0.mask_distractors(run, req) }
        fn reconstruct_sfm(&self, run: &dyn RunCtx, req: ReconstructSfm) -> Result<ToolResult> { self.0.reconstruct_sfm(run, req) }
        fn define_split(&self, run: &dyn RunCtx, req: DefineSplit) -> Result<ToolResult> {
            match run.path(&req.sfm_id) {
                Ok(_) => self.0.define_split(run, req),
                Err(e) => Ok(failure("define_split", "tool_failed", format!("{e:#}"))),
            }
        }
        fn train_splats(&self, run: &dyn RunCtx, req: TrainSplats) -> Result<ToolResult> { self.0.train_splats(run, req) }
        fn evaluate(&self, run: &dyn RunCtx, req: Evaluate) -> Result<ToolResult> { self.0.evaluate(run, req) }
        fn render_views(&self, run: &dyn RunCtx, req: RenderViews) -> Result<ToolResult> { self.0.render_views(run, req) }
        fn extract_mesh(&self, run: &dyn RunCtx, req: ExtractMesh) -> Result<ToolResult> { self.0.extract_mesh(run, req) }
        fn on_job_complete(&self, run: &dyn RunCtx, handler: &str, dir: &Path, code: i32) -> Result<ToolResult> {
            self.0.on_job_complete(run, handler, dir, code)
        }
    }

    #[test]
    fn train_splats_eval_every_must_be_positive_when_given() {
        let req = |extra: Value| {
            let mut r = serde_json::json!({"sfm_id": "s", "split_id": "p"});
            r.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            request("train_splats", r)
        };
        assert!(req(serde_json::json!({})).is_ok(), "omitted: the default cadence");
        assert!(req(serde_json::json!({"eval_every": null})).is_ok(), "null: the default cadence");
        assert!(req(serde_json::json!({"eval_every": 1})).is_ok(), "1 is the smallest cadence");
        let zero = req(serde_json::json!({"eval_every": 0})).expect_err("0 would silently disable mid-run scoring");
        assert!(zero.downcast_ref::<InvalidRequest>().is_some(), "rejected as an invalid request: {zero:#}");
    }

    #[test]
    fn identical_retry_reruns_a_failed_operation_and_replays_only_success() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        std::fs::write(tmp.path().join("input.mp4"), "fixture input")?;
        let mut registry = Registry::new();
        registry.register_backend(Arc::new(Guarded(FakeBackend::default())));
        let run = Run::create(tmp.path().join("run"), tmp.path().join("input.mp4"), "retry", "guarded")?
            .with_registry(Arc::new(registry));
        // The SfM directory exists but is not registered yet: define_split fails "unknown artifact".
        let (sfm_id, sfm_dir) = run.new_artifact_dir(ArtifactKind::Sfm)?;
        let request = json!({"sfm_id": sfm_id});
        let failed = dispatch(&run, "define_split", request.clone());
        assert_eq!(failed.status, Status::Failed);
        let operation_id = run.state()?.operations.keys().next().unwrap().clone();
        // A failed job of the same operation is not attached to either.
        run.journal("job_planned", json!({"job_id":"job_failed0000000","tool":"define_split","operation_id":operation_id}))?;
        run.journal("job_completed", json!({"job_id":"job_failed0000000","result":failed}))?;

        run.register(Artifact {
            id: sfm_id.clone(),
            kind: ArtifactKind::Sfm,
            path: sfm_dir.strip_prefix(run.root())?.to_string_lossy().into_owned(),
            meta: Default::default(),
        })?;
        let retried = dispatch(&run, "define_split", request.clone());
        assert_eq!(retried.status, Status::Succeeded, "{:?}", retried.error);
        assert!(retried.job_id.is_none());
        // The success is now replayed: same split, no third run of the tool.
        let replayed = dispatch(&run, "define_split", request);
        assert_eq!(replayed.artifacts[0].id, retried.artifacts[0].id);
        let planned = run.events()?.iter().filter(|e| e.event == "operation_planned").count();
        assert_eq!(planned, 2);
        Ok(())
    }
}
