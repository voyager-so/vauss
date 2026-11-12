//! Re-root a GLB under a node carrying a transform matrix (JSON-chunk edit; geometry untouched).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

const MAGIC: u32 = 0x4654_6c67; // "glTF"
const JSON_CHUNK: u32 = 0x4e4f_534a;

/// Copy `src` to `dst` with every scene's root nodes parented to a new node whose glTF `matrix`
/// (column-major 4x4) is `m`.
pub fn write_with_root_matrix(src: &Path, dst: &Path, m: &[f64; 16]) -> Result<()> {
    let bytes = std::fs::read(src).with_context(|| format!("reading {}", src.display()))?;
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    ensure!(bytes.len() >= 20 && u32_at(0) == MAGIC, "{}: not a GLB", src.display());
    let json_len = u32_at(12) as usize;
    ensure!(u32_at(16) == JSON_CHUNK, "{}: first chunk is not JSON", src.display());
    let mut doc: Value = serde_json::from_slice(&bytes[20..20 + json_len])?;
    let rest = &bytes[20 + json_len..]; // BIN chunk (header included), if any

    let nodes = doc["nodes"].as_array_mut().context("GLB has no nodes")?;
    let root = nodes.len();
    let mut children: Vec<Value> = Vec::new();
    for scene in doc["scenes"].as_array().cloned().unwrap_or_default() {
        children.extend(scene["nodes"].as_array().cloned().unwrap_or_default());
    }
    doc["nodes"].as_array_mut().unwrap().push(json!({
        "name": "scene_from_source",
        "matrix": m.iter().map(|v| (*v as f32) as f64).collect::<Vec<_>>(),
        "children": children,
    }));
    for scene in doc["scenes"].as_array_mut().into_iter().flatten() {
        scene["nodes"] = json!([root]);
    }

    let mut js = serde_json::to_vec(&doc)?;
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + rest.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(js.len() as u32).to_le_bytes());
    out.extend_from_slice(&JSON_CHUNK.to_le_bytes());
    out.extend_from_slice(&js);
    out.extend_from_slice(rest);
    std::fs::write(dst, out)?;
    Ok(())
}
