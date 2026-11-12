//! Input resolution guard. Low-resolution sources collapse SfM (320x180 Copped Hall frames registered
//! 34/150 cameras, with flips), and no later step can recover detail the source never had.

use crate::{Run, RunCtx, jobs::failure};
use serde_json::{Value, json};
use splat_contracts::{Status, ToolResult};
use std::path::{Path, PathBuf};

/// Long edge below this is a warning.
pub const WARN_LONG_EDGE: u32 = 960;
/// Long edge below this fails unless the user sets SPLAT_ALLOW_LOW_RESOLUTION=1 (`--allow-low-resolution`).
pub const FAIL_LONG_EDGE: u32 = 480;
pub const OVERRIDE_ENV: &str = "SPLAT_ALLOW_LOW_RESOLUTION";

pub fn overridden() -> bool {
    std::env::var(OVERRIDE_ENV).is_ok_and(|v| !matches!(v.trim(), "" | "0" | "false"))
}

fn images(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
            if path.is_dir() {
                stack.push(path);
            } else if matches!(ext.as_deref(), Some("jpg" | "jpeg" | "png" | "webp" | "tif" | "tiff" | "bmp" | "heic" | "heif" | "hif")) {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// HEIC/HEIF size from its `ispe` properties (meta/iprp/ipco); the largest is the full image
/// (an iPhone grid is 4032x3024 while each tile is 512x512).
fn heif_size(path: &Path) -> Option<(u32, u32)> {
    fn boxes(mut d: &[u8]) -> Vec<([u8; 4], &[u8])> {
        let mut out = vec![];
        while d.len() >= 8 {
            let size = u32::from_be_bytes(d[..4].try_into().unwrap()) as usize;
            let size = if size == 0 { d.len() } else { size };
            if size < 8 || size > d.len() {
                break;
            }
            out.push((d[4..8].try_into().unwrap(), &d[8..size]));
            d = &d[size..];
        }
        out
    }
    let file = std::fs::read(path).ok()?;
    let (_, meta) = boxes(&file).into_iter().find(|(k, _)| k == b"meta")?;
    let (_, iprp) = boxes(meta.get(4..)?).into_iter().find(|(k, _)| k == b"iprp")?;
    let (_, ipco) = boxes(iprp).into_iter().find(|(k, _)| k == b"ipco")?;
    boxes(ipco)
        .into_iter()
        .filter(|(k, b)| k == b"ispe" && b.len() >= 12)
        .map(|(_, b)| (u32::from_be_bytes(b[4..8].try_into().unwrap()), u32::from_be_bytes(b[8..12].try_into().unwrap())))
        .max_by_key(|&(w, h)| u64::from(w) * u64::from(h))
}

/// Median long edge of an image folder (`images/` inside it when present), from file headers only.
pub fn image_folder_long_edge(source: &Path) -> Option<u32> {
    let nested = source.join("images");
    let files = images(if nested.is_dir() { &nested } else { source });
    if files.is_empty() {
        return None;
    }
    let n = files.len().min(96);
    let mut edges = (0..n)
        .filter_map(|k| {
            // Sniff the format from the bytes: extension lookup misses e.g. upper-case `.JPG`.
            let path = &files[k * (files.len() - 1) / (n - 1).max(1)];
            image::ImageReader::open(path).ok()?.with_guessed_format().ok()?.into_dimensions().ok().or_else(|| heif_size(path))
        })
        .map(|(w, h)| w.max(h))
        .collect::<Vec<_>>();
    edges.sort_unstable();
    edges.get(edges.len() / 2).copied()
}

/// Source long edge: image folders are measured directly; videos use the latest inspect_input metrics.
fn source_long_edge(run: &Run, inspect: Option<&ToolResult>) -> Option<u32> {
    if run.source().is_dir() {
        return image_folder_long_edge(run.source());
    }
    let from = |m: &std::collections::BTreeMap<String, Value>| {
        Some(m.get("width")?.as_u64()?.max(m.get("height")?.as_u64()?) as u32)
    };
    if let Some(r) = inspect {
        return from(&r.metrics);
    }
    let state = run.state().ok()?;
    state.operations.values().rev().find_map(|op| {
        let r: ToolResult = serde_json::from_value(op.get("result")?.clone()).ok()?;
        (r.tool == "inspect_input").then(|| from(&r.metrics)).flatten()
    })
}

fn message(edge: u32, kind: &str, fail: bool) -> String {
    let ask = if kind == "video" { "the original full-resolution video" } else { "the original video or full-size photos" };
    let verdict = if fail {
        format!("too small to reconstruct (below {FAIL_LONG_EDGE} px)")
    } else {
        format!("below the recommended {WARN_LONG_EDGE} px; camera solving and detail will suffer")
    };
    let mut text = format!(
        "Low resolution: the source's long edge is {edge} px, {verdict}. Upscaling cannot recover detail that was lost; please provide {ask}."
    );
    if fail {
        text.push_str(&format!(" To try anyway, rerun with --allow-low-resolution ({OVERRIDE_ENV}=1)."));
    }
    text
}

/// Adds `long_edge_px` and a `low_resolution` defect to a successful inspect_input result; fails it
/// below FAIL_LONG_EDGE unless overridden.
pub fn guard_inspect(run: &Run, mut r: ToolResult) -> ToolResult {
    if r.status != Status::Succeeded {
        return r;
    }
    let Some(edge) = source_long_edge(run, Some(&r)) else { return r };
    r.metrics.insert("long_edge_px".into(), json!(edge));
    if edge >= WARN_LONG_EDGE {
        return r;
    }
    let kind = if run.source().is_dir() { "images" } else { "video" };
    let fail = edge < FAIL_LONG_EDGE && !overridden();
    let mut text = message(edge, kind, fail);
    if edge < FAIL_LONG_EDGE && !fail {
        text = text.replacen(
            &format!("below the recommended {WARN_LONG_EDGE} px; camera solving and detail will suffer"),
            &format!("below {FAIL_LONG_EDGE} px; continuing only because --allow-low-resolution is set, expect a poor result"),
            1,
        );
    }
    if fail {
        let mut failed = failure("inspect_input", "low_resolution", &text);
        failed.artifacts = r.artifacts;
        failed.metrics = r.metrics;
        failed.images = r.images;
        failed.metrics.insert("defects".into(), json!([{"code":"low_resolution","severity":"fail","long_edge_px":edge,"message":text}]));
        return failed;
    }
    let severity = if edge < FAIL_LONG_EDGE { "fail_overridden" } else { "warn" };
    r.metrics.insert("defects".into(), json!([{"code":"low_resolution","severity":severity,"long_edge_px":edge,"message":text}]));
    r.warnings.push(text);
    r
}

/// extract_frames refuses a source that inspect_input would fail, even if inspection was skipped.
pub fn guard_extract(run: &Run) -> Option<ToolResult> {
    let edge = source_long_edge(run, None)?;
    if edge >= FAIL_LONG_EDGE || overridden() {
        return None;
    }
    let kind = if run.source().is_dir() { "images" } else { "video" };
    Some(failure("extract_frames", "low_resolution", message(edge, kind, true)))
}
