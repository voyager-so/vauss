//! Backend-independent edits of a trained candidate: `remove_objects` drops a selection's Gaussians
//! from the candidate's PLY (CPU, every other PLY element copied byte for byte) into a child
//! candidate that render_views, extract_mesh, shots and exports use like any other.
use crate::{
    RunCtx,
    jobs::{failure, result},
};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use splat_contracts::*;
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::Path,
};

/// Reads a 1-D little-endian u32 `.npy` (selection ids).
pub fn read_npy_u32(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY", "{} is not a .npy file", path.display());
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len])?;
    ensure!(header.contains("'<u4'") || header.contains("'|u4'"), "{}: expected dtype <u4, header {header}", path.display());
    ensure!(header.contains("'fortran_order': False"), "{}: fortran order unsupported", path.display());
    Ok(bytes[start + header_len..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Writes a 1-D little-endian u32 `.npy`.
pub fn write_npy_u32(path: &Path, values: &[u32]) -> Result<()> {
    let mut header = format!("{{'descr': '<u4', 'fortran_order': False, 'shape': ({},), }}", values.len());
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + values.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    values.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
    fs::write(path, out)?;
    Ok(())
}

fn property_size(ty: &str) -> Result<usize> {
    Ok(match ty {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "int" | "uint" | "float" | "int32" | "uint32" | "float32" => 4,
        "double" | "float64" => 8,
        other => bail!("unsupported PLY property type {other}"),
    })
}

/// Copies `src` to `dst` without the vertex rows `drop` names (sorted, unique). Binary
/// little-endian PLYs whose vertex element has fixed-size properties; the vertex element may be
/// followed by other elements (copied as is). Returns (vertices before, after).
pub fn filter_ply(src: &Path, dst: &Path, drop: &[u32]) -> Result<(usize, usize)> {
    let mut reader = BufReader::new(fs::File::open(src).with_context(|| format!("opening {}", src.display()))?);
    let mut header = Vec::new();
    let (mut n, mut stride, mut in_vertex, mut vertex_first, mut seen_element) = (None, 0usize, false, false, false);
    loop {
        let mut line = String::new();
        ensure!(reader.read_line(&mut line)? > 0, "{}: PLY header has no end_header", src.display());
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["format", f, ..] => ensure!(*f == "binary_little_endian", "{}: only binary_little_endian PLYs are supported", src.display()),
            ["element", name, count] => {
                in_vertex = *name == "vertex";
                if in_vertex {
                    vertex_first = !seen_element;
                    n = Some(count.parse::<usize>()?);
                }
                seen_element = true;
            }
            ["property", "list", ..] if in_vertex => bail!("{}: list properties in the vertex element are unsupported", src.display()),
            ["property", ty, _] if in_vertex => stride += property_size(ty)?,
            _ => {}
        }
        let end = words.first() == Some(&"end_header");
        if !end || words.len() == 1 {
            header.push(line);
        }
        if end {
            break;
        }
    }
    let n = n.context("PLY has no vertex element")?;
    ensure!(vertex_first, "{}: the vertex element must come first", src.display());
    ensure!(drop.last().is_none_or(|&last| (last as usize) < n), "selection index out of range for {} vertices", n);
    let kept = n - drop.len();
    let mut out = std::io::BufWriter::new(fs::File::create(dst)?);
    let mut replaced = false;
    for line in &header {
        if !replaced && line.split_whitespace().take(2).eq(["element", "vertex"]) {
            writeln!(out, "element vertex {kept}")?;
            replaced = true;
        } else {
            out.write_all(line.as_bytes())?;
        }
    }
    let mut row = vec![0u8; stride];
    let mut next = drop.iter().peekable();
    for i in 0..n {
        reader.read_exact(&mut row).with_context(|| format!("{}: truncated vertex data", src.display()))?;
        if next.peek().is_some_and(|&&d| d as usize == i) {
            next.next();
            continue;
        }
        out.write_all(&row)?;
    }
    std::io::copy(&mut reader, &mut out)?;
    out.flush()?;
    Ok((n, kept))
}

fn candidate_ply(dir: &Path) -> Result<std::path::PathBuf> {
    ["splat.ply", "checkpoint.ply"].iter().map(|n| dir.join(n)).find(|p| p.is_file()).context("candidate has no saved splat yet")
}

fn rel(run: &dyn RunCtx, path: &Path) -> String {
    path.strip_prefix(run.root()).unwrap_or(path).to_string_lossy().into_owned()
}

pub fn remove_objects(run: &dyn RunCtx, req: RemoveObjects) -> Result<ToolResult> {
    let started = std::time::Instant::now();
    let parent = run.artifact(&req.candidate_id)?;
    ensure!(parent.kind == ArtifactKind::Candidate, "{} is not a candidate", req.candidate_id);
    let selection = run.artifact(&req.selection_id)?;
    ensure!(selection.kind == ArtifactKind::Selection, "{} is not a selection", req.selection_id);
    if selection.meta.get("candidate_id").and_then(Value::as_str) != Some(req.candidate_id.as_str()) {
        return Ok(failure(
            "remove_objects",
            "selection_mismatch",
            format!("{} was made on {}, not {}: select on this candidate first", req.selection_id, selection.meta.get("candidate_id").and_then(Value::as_str).unwrap_or("?"), req.candidate_id),
        ));
    }
    let sel_dir = run.path(&req.selection_id)?;
    let ids = read_npy_u32(&sel_dir.join("selected_ids.npy"))?;
    let objects = sel_dir.join("object_ids.npy");
    let objects = if objects.is_file() { read_npy_u32(&objects)? } else { vec![0; ids.len()] };
    ensure!(objects.len() == ids.len(), "object_ids.npy and selected_ids.npy differ in length");
    let mut drop: Vec<u32> = ids.iter().zip(&objects).filter(|(_, o)| req.object_ids.is_empty() || req.object_ids.contains(o)).map(|(i, _)| *i).collect();
    drop.sort_unstable();
    drop.dedup();
    if drop.is_empty() {
        return Ok(failure("remove_objects", "empty_selection", "The selection holds no Gaussians for these object_ids: nothing to remove"));
    }
    let src = candidate_ply(&run.path(&req.candidate_id)?)?;
    crate::fal::materialize(run.root(), &src)?; // a --cloud model kept on fal
    let (id, dir) = run.new_artifact_dir(ArtifactKind::Candidate)?;
    let (before, after) = filter_ply(&src, &dir.join("splat.ply"), &drop)?;
    // Everything the views/mesh/shot tools read from a candidate (sfm_id, split_id, max_resolution,
    // render mode...) carries over; the edit is recorded on top.
    let mut meta = parent.meta.clone();
    for k in ["parent_candidate_id", "edit", "selection_id", "removed_object_ids", "evaluated_via_parent"] {
        meta.remove(k);
    }
    meta.insert("parent_candidate_id".into(), json!(req.candidate_id));
    meta.insert("edit".into(), json!("remove_objects"));
    meta.insert("selection_id".into(), json!(req.selection_id));
    meta.insert("removed_object_ids".into(), json!(req.object_ids));
    meta.insert("prompt".into(), selection.meta.get("prompt").cloned().unwrap_or(Value::Null));
    let artifact = Artifact { id, kind: ArtifactKind::Candidate, path: rel(run, &dir), meta };
    let mut r = result("remove_objects", Status::Succeeded);
    r.artifacts.push(artifact);
    r.metrics.insert("splats_before".into(), json!(before));
    r.metrics.insert("splats_after".into(), json!(after));
    r.metrics.insert("removed".into(), json!(before - after));
    r.metrics.insert("splat_ply".into(), json!(rel(run, &dir.join("splat.ply"))));
    r.warnings.push("Edited candidate: it is not scored on the held-out views (they still show the removed object); its parent's evaluation is the quality number. Check it with render_views.".into());
    r.wall_s = Some(started.elapsed().as_secs_f64());
    Ok(r)
}
