use anyhow::{Context, Result, ensure};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use splat_contracts::*;
use splat_core::RunCtx;
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

pub(crate) fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(
        fs::File::open(path).with_context(|| format!("Read {}", path.display()))?,
    )
    .with_context(|| format!("Parse {}", path.display()))
}
pub(crate) fn read_engine_json(path: &Path) -> Result<Value> {
    // Spirula emits Python-style Infinity in config.json. Sanitize only bare
    // non-finite tokens, preserving strings and rejecting other malformed JSON.
    let text = fs::read_to_string(path)?;
    let mut output = String::with_capacity(text.len());
    let mut quoted = false;
    let mut escaped = false;
    let mut token = String::new();
    let flush = |output: &mut String, token: &mut String| {
        output.push_str(
            if matches!(token.as_str(), "Infinity" | "-Infinity" | "NaN") {
                "null"
            } else {
                token
            },
        );
        token.clear();
    };
    for c in text.chars() {
        if quoted {
            output.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                quoted = false;
            }
        } else if c == '"' {
            flush(&mut output, &mut token);
            quoted = true;
            output.push(c);
        } else if c.is_whitespace() || matches!(c, '{' | '}' | '[' | ']' | ':' | ',') {
            flush(&mut output, &mut token);
            output.push(c);
        } else {
            token.push(c);
        }
    }
    flush(&mut output, &mut token);
    serde_json::from_str(&output).with_context(|| format!("Parse engine JSON {}", path.display()))
}
pub(crate) fn write(path: &Path, data: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(data)?;
    atomic_write(path, &bytes)
}
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    // Unique staging files allow concurrent status polls without corrupting the series.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp-{}-{n}", std::process::id()));
    let result = (|| {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}
pub(crate) fn hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).with_context(|| format!("Hash {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub(crate) fn string(path: &Path) -> Result<String> {
    Ok(path.to_str().context("Path is not valid UTF-8")?.to_owned())
}
pub(crate) fn rel(run: &dyn RunCtx, path: &Path) -> Result<String> {
    string(path.strip_prefix(run.root())?)
}
pub(crate) fn meta(value: Value) -> BTreeMap<String, Value> {
    value.as_object().unwrap().clone().into_iter().collect()
}
pub(crate) fn field<'a>(artifact: &'a Artifact, key: &str) -> Result<&'a str> {
    artifact
        .meta
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("Missing {key} on {}", artifact.id))
}
pub(crate) fn new_artifact(
    run: &dyn RunCtx,
    kind: ArtifactKind,
    data: Value,
) -> Result<(Artifact, PathBuf)> {
    let (id, path) = run.new_artifact_dir(kind)?;
    fs::create_dir_all(&path)?;
    let a = Artifact {
        id,
        kind,
        path: rel(run, &path)?,
        meta: meta(data),
    };
    run.register(a.clone())?;
    Ok((a, path))
}
pub(crate) fn get(run: &dyn RunCtx, id: &str, kind: ArtifactKind) -> Result<(Artifact, PathBuf)> {
    let a = run.artifact(id)?;
    ensure!(a.kind == kind, "{id} is {:?}, expected {kind:?}", a.kind);
    Ok((a, run.path(id)?))
}
pub(crate) fn result(tool: &str) -> ToolResult {
    ToolResult {
        schema_version: SCHEMA_VERSION,
        tool: tool.into(),
        status: Status::Succeeded,
        job_id: None,
        artifacts: vec![],
        metrics: BTreeMap::new(),
        images: vec![],
        warnings: vec![],
        error: None,
        wall_s: None,
    }
}
pub(crate) fn error(tool: &str, code: &str, message: impl Into<String>) -> ToolResult {
    let mut r = result(tool);
    r.status = Status::Failed;
    r.error = Some(ToolError {
        code: code.into(),
        message: message.into(),
        retryable: false,
        evidence: vec![],
    });
    r
}
pub(crate) fn unsupported(tool: &str, code: &str, message: &str) -> ToolResult {
    let mut r = error(tool, code, message);
    r.status = Status::Unsupported;
    r
}
pub(crate) fn guarded(
    run: &dyn RunCtx,
    tool: &str,
    f: impl FnOnce() -> Result<ToolResult>,
) -> Result<ToolResult> {
    let outcome = (|| -> Result<ToolResult> {
        let mut result = f()?;
        if !result.images.is_empty() {
            let source = result
                .images
                .iter()
                .map(|p| run.root().join(p))
                .collect::<Vec<_>>();
            let artifact = result
                .artifacts
                .first()
                .context("Image result has no artifact")?;
            let previews = splat_eval::budget_jpegs(
                &source,
                &run.path(&artifact.id)?.join("driver-preview"),
                splat_eval::RESULT_IMAGE_BUDGET,
            )?;
            result.images = previews
                .iter()
                .map(|p| rel(run, p))
                .collect::<Result<_>>()?;
        }
        Ok(result)
    })();
    Ok(outcome.unwrap_or_else(|e| {
        if let Some(missing) = e.downcast_ref::<splat_core::split::MissingHeldoutFrames>() {
            let mut result = error(tool, "missing_heldout_frames", e.to_string());
            let error = result.error.as_mut().unwrap();
            error.evidence = missing.names.clone();
            error.retryable = true;
            result
        } else {
            error(tool, "invalid_input", format!("{e:#}"))
        }
    }))
}
pub(crate) fn images(path: &Path) -> Result<Vec<PathBuf>> {
    let mut found = vec![];
    if !path.exists() {
        return Ok(found);
    }
    for e in fs::read_dir(path)? {
        let p = e?.path();
        if p.is_dir() {
            found.extend(images(&p)?);
        } else if p.extension().and_then(|x| x.to_str()).is_some_and(|s| {
            matches!(
                s.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "tif" | "tiff" | "bmp" | "heic" | "heif" | "hif"
            )
        }) {
            found.push(p);
        }
    }
    found.sort();
    Ok(found)
}
pub(crate) fn safe_child(root: &Path, name: &str) -> Result<PathBuf> {
    ensure!(
        !name.is_empty()
            && Path::new(name)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "Unsafe image name: {name}"
    );
    Ok(root.join(name))
}
pub(crate) fn image_dir(source: &Path) -> PathBuf {
    if source.join("images").is_dir() {
        source.join("images")
    } else {
        source.into()
    }
}
pub(crate) fn link_directory(source: &Path, dest: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(source, dest)?;
    }
    #[cfg(windows)]
    {
        // Directory symlinks need extra privileges on Windows; copies work everywhere.
        copy_directory(source, dest)?;
    }
    Ok(())
}
#[cfg(windows)]
fn copy_directory(source: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    for e in fs::read_dir(source)? {
        let e = e?;
        let p = e.path();
        let d = dest.join(e.file_name());
        if p.is_dir() {
            copy_directory(&p, &d)?;
        } else {
            fs::copy(p, d)?;
        }
    }
    Ok(())
}
pub(crate) fn tail(s: &str, n: usize) -> String {
    s.chars()
        .rev()
        .take(n)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}
pub(crate) fn failure(tool: &str, code: i32, artifact: Artifact, log: &str) -> ToolResult {
    let lower = log.to_lowercase();
    let reason = if lower.contains("out of memory") || lower.contains("error_out_of_device_memory")
    {
        "oom"
    } else if matches!(code, 130 | -2 | -15 | 143) {
        "interrupted"
    } else if tool == "reconstruct_sfm" && code == 2 {
        "no_reconstruction"
    } else {
        "subprocess_failed"
    };
    let mut r = error(
        tool,
        reason,
        format!("Spirula exited {code}: {}", tail(log, 1600)),
    );
    r.artifacts.push(artifact);
    r.metrics = meta(json!({"exit_code":code}));
    r
}
