use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use sha2::{Digest, Sha256};
use splat_contracts::*;
use splat_core::RunCtx;

pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "tif", "tiff", "bmp", "heic", "heif", "hif"];

pub fn write_json(path: &Path, data: &impl Serialize) -> anyhow::Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(data)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Image files under `dir`, recursively, sorted.
pub fn list_images(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).with_context(|| format!("listing {}", d.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_image(&path) {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// `source/images` when present, else `source`.
pub fn image_dir(source: &Path) -> PathBuf {
    let nested = source.join("images");
    if nested.is_dir() { nested } else { source.to_path_buf() }
}

pub fn rel(run: &dyn RunCtx, path: &Path) -> String {
    path.strip_prefix(run.root()).unwrap_or(path).to_string_lossy().replace('\\', "/")
}

pub fn result(tool: &str, status: Status) -> ToolResult {
    ToolResult {
        schema_version: SCHEMA_VERSION,
        tool: tool.into(),
        status,
        job_id: None,
        artifacts: vec![],
        metrics: BTreeMap::new(),
        images: vec![],
        warnings: vec![],
        error: None,
        wall_s: None,
    }
}

pub fn failure(tool: &str, code: &str, message: impl Into<String>) -> ToolResult {
    ToolResult {
        error: Some(ToolError { code: code.into(), message: message.into(), retryable: false, evidence: vec![] }),
        ..result(tool, Status::Failed)
    }
}

pub fn unsupported(tool: &str, message: impl Into<String>) -> ToolResult {
    ToolResult {
        error: Some(ToolError {
            code: "unsupported_capability".into(),
            message: message.into(),
            retryable: false,
            evidence: vec![],
        }),
        ..result(tool, Status::Unsupported)
    }
}

/// Typed input errors that tools report as `invalid_input` rather than bubbling up.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct InvalidInput(pub String);

pub fn invalid(msg: impl Into<String>) -> anyhow::Error {
    InvalidInput(msg.into()).into()
}

/// Convert tool-level errors into a failed ToolResult (the trait's Err is for core failures).
pub fn guard(tool: &str, f: impl FnOnce() -> anyhow::Result<ToolResult>) -> anyhow::Result<ToolResult> {
    match f() {
        Ok(r) => Ok(r),
        Err(e) => {
            if let Some(missing) = e.downcast_ref::<splat_core::split::MissingHeldoutFrames>() {
                let mut r = failure(tool, "missing_heldout_frames", format!("{e:#}"));
                if let Some(err) = r.error.as_mut() {
                    err.evidence = missing.names.clone();
                    err.retryable = true;
                }
                return Ok(r);
            }
            let code = if e.is::<InvalidInput>() { "invalid_input" } else { "tool_failed" };
            Ok(failure(tool, code, format!("{e:#}")))
        }
    }
}

pub fn new_artifact(run: &dyn RunCtx, kind: ArtifactKind, meta: Value) -> anyhow::Result<(Artifact, PathBuf)> {
    let (id, dir) = run.new_artifact_dir(kind)?;
    std::fs::create_dir_all(&dir)?;
    let meta = match meta {
        Value::Object(m) => m.into_iter().collect(),
        Value::Null => BTreeMap::new(),
        other => bail!("artifact meta must be an object, got {other}"),
    };
    let artifact = Artifact { id, kind, path: rel(run, &dir), meta };
    run.register(artifact.clone())?;
    Ok((artifact, dir))
}

pub fn get_artifact(run: &dyn RunCtx, id: &str, kind: ArtifactKind) -> anyhow::Result<(Artifact, PathBuf)> {
    let artifact = run.artifact(id).map_err(|e| invalid(format!("unknown artifact {id}: {e:#}")))?;
    if artifact.kind != kind {
        return Err(invalid(format!("{id} is {:?}, expected {kind:?}", artifact.kind)));
    }
    let path = run.path(id)?;
    Ok((artifact, path))
}

pub fn meta_str<'a>(a: &'a Artifact, key: &str) -> anyhow::Result<&'a str> {
    a.meta.get(key).and_then(Value::as_str).with_context(|| format!("{} has no meta.{key}", a.id))
}

/// Run a command, capturing output; error includes the tail of stderr.
pub fn run_capture(cmd: &mut Command) -> anyhow::Result<std::process::Output> {
    let out = cmd.output().with_context(|| format!("spawning {:?}", cmd.get_program()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: String = err.chars().rev().take(2000).collect::<Vec<_>>().into_iter().rev().collect();
        bail!("{:?} exited {}: {tail}", cmd.get_program(), out.status);
    }
    Ok(out)
}

/// Hard link, falling back to copy (different filesystem, no permission).
pub fn link_or_copy(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if std::fs::hard_link(src, dst).is_err() {
        std::fs::copy(src, dst).with_context(|| format!("copying {} -> {}", src.display(), dst.display()))?;
    }
    Ok(())
}

pub fn tail_file(path: &Path, max_chars: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let skip = text.chars().count().saturating_sub(max_chars);
    text.chars().skip(skip).collect()
}

pub fn f64_metric(v: f64) -> Value {
    serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number)
}
