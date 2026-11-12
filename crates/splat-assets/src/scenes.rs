//! What `vauss view` can open: scene packages, run dirs and bare `.ply`/`.spz` files.
//!
//! A bare splat gets a minimal scene package (shared `scene.json` format) in a cache dir; the splat
//! itself is served from its original path, never copied.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};

/// The shared `scene.json` document.
pub type SceneJson = Value;

/// A resolved input: the dir the scene package is served (and generated items written) from,
/// plus files served from elsewhere by name.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// Canonical input path as given.
    pub input: PathBuf,
    pub scene_dir: PathBuf,
    pub aliases: HashMap<String, PathBuf>,
    /// Stem used for the scene id.
    pub stem: String,
}

pub fn cache_root() -> PathBuf {
    std::env::temp_dir().join("splat-view")
}

pub fn is_splat_file(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("ply") || e.eq_ignore_ascii_case("spz"))
}

/// A `vauss run --cloud` result kept on fal: a small stub (splat_core::fal::remote_stub) in place of the file.
fn is_fal_stub(p: &Path) -> bool {
    std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() <= 256)
        && std::fs::read(p).is_ok_and(|b| b.starts_with(b"splat-fal-remote v1 "))
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Newest splat in a run dir (depth <= 4): `**/splat.ply` and `**/*.spz` (exports/, checkpoints,
/// artifacts/*), plus top-level `*.ply`.
pub fn find_splat(dir: &Path) -> Option<PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                if depth < 4 && !name.starts_with('.') && name != "node_modules" {
                    walk(&p, depth + 1, out);
                }
            } else if is_fal_stub(&p) {
                // A `vauss run --cloud` model left on fal: only its stub is here.
            } else if name == "splat.ply" || name.to_ascii_lowercase().ends_with(".spz") || (depth == 1 && is_splat_file(&p)) {
                out.push(p);
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, 1, &mut found);
    found.into_iter().max_by_key(|p| mtime(p))
}

/// Name for a file-backed scene: the file stem, or its parent dir for generic stems (`splat.ply`).
fn file_stem(file: &Path) -> String {
    let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    if matches!(stem.as_str(), "splat" | "scene" | "point_cloud" | "checkpoint" | "model") {
        if let Some(parent) = file.parent().and_then(Path::file_name) {
            return parent.to_string_lossy().into_owned();
        }
    }
    stem
}

pub fn resolve(input: &Path) -> Result<Resolved> {
    let path = std::fs::canonicalize(input).with_context(|| format!("no such file or folder: {}", input.display()))?;
    if path.is_dir() && path.join("scene.json").is_file() {
        let stem = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "scene".into());
        return Ok(Resolved { input: path.clone(), scene_dir: path, aliases: HashMap::new(), stem });
    }
    // A `vauss run --cloud` run: its full model stays on fal, the result is the web package it brought back.
    let web = path.join("exports/web");
    if path.is_dir() && web.join("scene.json").is_file()
        && ["remote_artifacts.json", "fal-remote-kept.json"].iter().any(|r| path.join(r).is_file())
    {
        let stem = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "scene".into());
        return Ok(Resolved { input: path, scene_dir: web, aliases: HashMap::new(), stem });
    }
    let (file, stem) = if path.is_dir() {
        let f = find_splat(&path).with_context(|| format!("{} has no scene.json and no .ply/.spz splat", path.display()))?;
        (f, path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())
    } else if is_splat_file(&path) {
        let stem = file_stem(&path);
        (path.clone(), stem)
    } else {
        bail!("{} is not a scene folder, run folder, .ply or .spz", path.display());
    };
    let name = file.file_name().context("splat file name")?.to_string_lossy().into_owned();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    file.hash(&mut h);
    mtime(&file).hash(&mut h);
    // Packages from an older exporter format are rebuilt rather than reused.
    splat_webexport::overflow::FORMAT.hash(&mut h);
    splat_webexport::neural::FORMAT.hash(&mut h);
    // scene.json gained `render_mode`: older cached packages render mip splats as default.
    "render_mode".hash(&mut h);
    // scene.json gained `splat_lod` (scene.rad, the coarse-first LoD stream).
    "splat_lod".hash(&mut h);
    // scene.rad colour/scales went to f16 (8-bit colour clipped the extremes).
    "splat_lod_f16".hash(&mut h);
    let scene_dir = cache_root().join(format!("{:016x}", h.finish()));
    // Converted before (same file + mtime): reuse the package.
    if scene_dir.join("scene.json").is_file() && scene_dir.join("scene.spz").is_file() {
        return Ok(Resolved { input: path, scene_dir, aliases: HashMap::new(), stem });
    }
    std::fs::create_dir_all(&scene_dir).with_context(|| format!("create {}", scene_dir.display()))?;
    // Full conversion (floor at y=0, metres, start camera from the capture, scene.spz) when the
    // capture is found next to the splat or in its run; else a minimal package over the raw file.
    let src = if path.is_dir() { path.as_path() } else { file.as_path() };
    if file.extension().is_some_and(|e| e.eq_ignore_ascii_case("ply")) {
        match splat_webexport::export_web(src, None, &scene_dir) {
            Ok(_) => return Ok(Resolved { input: path, scene_dir, aliases: HashMap::new(), stem }),
            Err(e) => eprintln!("vauss view: {}: no full conversion ({e:#}); showing the raw splat", file.display()),
        }
    }
    let mut scene = scene_package_for(&file)?;
    scene["name"] = json!(crate::slug(&stem));
    scene["title"] = json!(stem);
    crate::write_atomic(&scene_dir.join("scene.json"), &serde_json::to_vec_pretty(&scene)?)?;
    Ok(Resolved { input: path, scene_dir, aliases: HashMap::from([(name, file)]), stem })
}

/// Minimal scene.json for a bare splat file (served next to it under its own file name).
pub fn scene_package_for(file: &Path) -> Result<SceneJson> {
    let name = file.file_name().context("splat file name")?.to_string_lossy().into_owned();
    let is_ply = name.to_ascii_lowercase().ends_with(".ply");
    let bounds = if is_ply { Some(ply_bounds(file)?) } else { None };
    let (focus, radius) = bounds.map_or(([0.0f32; 3], 3.0f32), |(lo, hi)| {
        let c = [0, 1, 2].map(|i| 0.5 * (lo[i] + hi[i]));
        let r = 0.5 * (0..3).map(|i| (hi[i] - lo[i]).powi(2)).sum::<f32>().sqrt();
        (c, r.max(0.1))
    });
    let position = [focus[0], focus[1] + 0.3 * radius, focus[2] + 1.6 * radius];
    let mut scene = json!({
        "schema": 1,
        "name": "scene",
        "title": "scene",
        "splat": name,
        "coords": "y_up_meters",
        "focus": focus,
        "radius": radius,
        "start_camera": {"position": position, "look_at": focus, "vfov_deg": 60},
        "source": {"path": file.to_string_lossy()},
    });
    // The trainer's render mode (PLY header comment), so the viewer matches its rasterizer.
    if is_ply {
        scene["render_mode"] = json!(splat_webexport::ply::read_render_mode(file)?.as_str());
    }
    if let Some((lo, hi)) = bounds {
        scene["bounds"] = json!({"min": lo, "max": hi});
    }
    Ok(scene)
}

const MAX_SAMPLES: usize = 200_000;

/// Robust (2nd..98th percentile per axis) bounds of a PLY's vertex x,y,z; samples at most
/// [`MAX_SAMPLES`] vertices. Handles `binary_little_endian` and `ascii`.
pub fn ply_bounds(path: &Path) -> Result<([f32; 3], [f32; 3])> {
    let mut r = BufReader::new(std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?);
    let (mut format, mut count, mut props, mut in_vertex, mut seen_vertex) = (String::new(), 0usize, Vec::<(String, usize)>::new(), false, false);
    loop {
        let mut line = String::new();
        if r.read_line(&mut line)? == 0 {
            bail!("{}: PLY header has no end_header", path.display());
        }
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.as_slice() {
            ["end_header"] => break,
            ["format", f, ..] => format = (*f).to_string(),
            ["element", name, n] => {
                in_vertex = *name == "vertex";
                if in_vertex {
                    count = n.parse()?;
                    seen_vertex = true;
                } else if !seen_vertex {
                    bail!("{}: vertex is not the first PLY element", path.display());
                }
            }
            ["property", "list", ..] if in_vertex => bail!("{}: list properties on vertices are not supported", path.display()),
            ["property", ty, name] if in_vertex => {
                let size = match *ty {
                    "char" | "uchar" | "int8" | "uint8" => 1,
                    "short" | "ushort" | "int16" | "uint16" => 2,
                    "int" | "uint" | "float" | "int32" | "uint32" | "float32" => 4,
                    "double" | "float64" => 8,
                    other => bail!("{}: unknown PLY type {other}", path.display()),
                };
                props.push(((*ty).to_string() + " " + name, size));
            }
            _ => {}
        }
    }
    let idx = |axis: &str| props.iter().position(|(p, _)| p.ends_with(&format!(" {axis}"))).with_context(|| format!("{}: PLY vertices have no {axis}", path.display()));
    let xyz = [idx("x")?, idx("y")?, idx("z")?];
    let step = count.div_ceil(MAX_SAMPLES).max(1);
    let mut axes: [Vec<f32>; 3] = Default::default();
    match format.as_str() {
        "binary_little_endian" => {
            let offsets: Vec<usize> = props.iter().scan(0, |o, (_, s)| Some(std::mem::replace(o, *o + s))).collect();
            let stride: usize = props.iter().map(|(_, s)| s).sum();
            if xyz.iter().any(|&k| !matches!(props[k].0.split(' ').next(), Some("float" | "float32"))) {
                bail!("{}: only float x,y,z are supported in binary PLY", path.display());
            }
            let mut buf = vec![0u8; stride];
            let mut v = 0;
            while v < count {
                r.read_exact(&mut buf).with_context(|| format!("{}: truncated PLY", path.display()))?;
                for (a, &k) in xyz.iter().enumerate() {
                    let o = offsets[k];
                    axes[a].push(f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]));
                }
                let skip = (step - 1).min(count - v - 1);
                r.seek_relative((skip * stride) as i64)?;
                v += step;
            }
        }
        "ascii" => {
            for line in r.lines().take(count).step_by(step) {
                let line = line?;
                let w: Vec<&str> = line.split_whitespace().collect();
                for (a, &k) in xyz.iter().enumerate() {
                    axes[a].push(w.get(k).context("short PLY vertex line")?.parse()?);
                }
            }
        }
        other => bail!("{}: unsupported PLY format {other:?}", path.display()),
    }
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    for (a, vals) in axes.iter_mut().enumerate() {
        vals.retain(|v| v.is_finite());
        if vals.is_empty() {
            bail!("{}: PLY has no finite vertices", path.display());
        }
        vals.sort_by(f32::total_cmp);
        let at = |q: f32| vals[((vals.len() - 1) as f32 * q).round() as usize];
        (lo[a], hi[a]) = (at(0.02), at(0.98));
    }
    Ok((lo, hi))
}

/// One openable thing for the picker.
#[derive(Debug, Clone, Serialize)]
pub struct Found {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    /// Unix seconds.
    pub modified: u64,
}

/// Scene dirs and splat files under `roots` (depth <= 3, at most 200, newest first).
pub fn scan(roots: &[PathBuf]) -> Vec<Found> {
    const CAP: usize = 200;
    fn secs(t: Option<SystemTime>) -> u64 {
        t.and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs())
    }
    fn walk(dir: &Path, depth: usize, out: &mut Vec<Found>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            if out.len() >= CAP {
                return;
            }
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || matches!(name.as_str(), "node_modules" | "target") {
                continue;
            }
            let Ok(meta) = e.metadata() else { continue };
            if meta.is_dir() {
                let sj = p.join("scene.json");
                if let Ok(m) = std::fs::metadata(&sj) {
                    let splat = std::fs::read(&sj).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| v["splat"].as_str().map(|s| p.join(s)));
                    let size = splat.and_then(|s| std::fs::metadata(s).ok()).map_or(0, |m| m.len());
                    out.push(Found { name, path: p, size, modified: secs(m.modified().ok()) });
                } else if depth < 3 {
                    walk(&p, depth + 1, out);
                }
            } else if is_splat_file(&p) {
                out.push(Found { name: file_stem(&p), path: p, size: meta.len(), modified: secs(meta.modified().ok()) });
            }
        }
    }
    let mut out = Vec::new();
    for root in roots {
        walk(root, 1, &mut out);
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_ply_bounds() {
        let dir = std::env::temp_dir().join(format!("splat-ply-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.ply");
        let mut s = String::from("ply\nformat ascii 1.0\nelement vertex 101\nproperty float x\nproperty float y\nproperty float z\nend_header\n");
        for i in 0..=100 {
            s += &format!("{i} {} 1\n", -i);
        }
        std::fs::write(&p, s).unwrap();
        let (lo, hi) = ply_bounds(&p).unwrap();
        assert_eq!((lo, hi), ([2.0, -98.0, 1.0], [98.0, -2.0, 1.0]));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
