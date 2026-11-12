//! Automatic Rodin preview -> Tripo detail. The preview survives every upgrade failure.
//! gltfpack 1.3 is a release helper (MIT), preflighted before any paid request. Outputs use
//! immutable URLs; all LODs share one normalized source and therefore the same object frame.
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_shots::fal::Fal;

use crate::{Catalog, CatalogItem, valid_id, write_atomic};
#[path = "detail_alignment.rs"]
mod alignment;
use crate::generate::{fal_call, find_glb_url};

pub const ENABLED: bool = true;
pub const ENDPOINT: &str = "tripo3d/h3.1/image-to-3d";
pub const USD_DETAIL: f64 = 0.30;
const MAX_BYTES: u64 = 5_000_000;

pub fn mesh_payload(image_url: &str) -> Value {
    json!({"image_url":image_url,"face_limit":100_000,"texture":true,"pbr":true,
        "texture_quality":"standard","geometry_quality":"standard",
        "orientation":"align_image","quad":false})
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lod {
    pub level: String,
    pub glb: String,
    pub bytes: u64,
    pub triangles: u64,
    pub seconds: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DetailReport {
    pub schema: u32,
    pub item: CatalogItem,
    pub preview_glb: String,
    pub source_bytes: usize,
    pub source_triangles: u64,
    pub preview_dims_m: [f32; 3],
    pub detail_dims_m: [f32; 3],
    /// Relative extent change after uniform fit. Not a reconstruction accuracy score.
    pub extent_delta: [f32; 3],
    pub yaw_correction_deg: f64,
    pub alignment_rms: f64,
    pub lods: Vec<Lod>,
    pub seconds: f64,
}

/// Metadata-only triangle count; supports ordinary and meshopt-compressed GLB accessors.
pub fn triangles(bytes: &[u8]) -> Result<u64> {
    ensure!(bytes.len() >= 20 && &bytes[..4] == b"glTF", "not a GLB");
    let n = u32::from_le_bytes(bytes[12..16].try_into()?) as usize;
    let j: Value = serde_json::from_slice(bytes.get(20..20 + n).context("truncated GLB JSON")?)?;
    let mut count = 0;
    for mesh in j["meshes"].as_array().context("GLB has no meshes")? {
        for p in mesh["primitives"].as_array().context("mesh has no primitives")? {
            ensure!(p["mode"].as_u64().unwrap_or(4) == 4, "detail requires triangle primitives");
            let a = p["indices"].as_u64().or_else(|| p["attributes"]["POSITION"].as_u64()).context("primitive has no position accessor")?;
            count += j["accessors"][a as usize]["count"].as_u64().context("accessor has no count")? / 3;
        }
    }
    ensure!(count > 0, "empty detail mesh");
    Ok(count)
}

fn packer() -> Result<PathBuf> {
    let name = if cfg!(windows) { "gltfpack.exe" } else { "gltfpack" };
    let path = if let Some(p) = std::env::var_os("SPLAT_GLTFPACK") {
        PathBuf::from(p)
    } else {
        std::env::current_exe().ok().and_then(|p| p.canonicalize().ok())
            .and_then(|p| p.parent().map(|p| p.join("tools").join("gltfpack").join(name)))
            .filter(|p| p.is_file()).unwrap_or_else(|| PathBuf::from(name))
    };
    let output = Command::new(&path).arg("-v").output().context("high detail needs bundled tools/gltfpack/gltfpack 1.3 or SPLAT_GLTFPACK; keeping preview")?;
    ensure!(output.status.success(), "gltfpack preflight failed; keeping preview");
    let version = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    ensure!(version.contains("gltfpack 1.3"), "high detail requires gltfpack 1.3, found {version}");
    Ok(path)
}

fn pack(tool: &Path, source: &Path, dst: &Path, ratio: f64, texture: u32, normal_uastc: bool) -> Result<()> {
    let mut cmd = Command::new(tool);
    cmd.arg("-i").arg(source).arg("-o").arg(dst)
        .args(["-cc", "-ce", "ext", "-tc", "-tq", "8", "-tj", "14", "-se", "0.005"])
        .arg("-tl").arg(texture.to_string()).arg("-si").arg(ratio.min(1.0).to_string());
    if normal_uastc { cmd.args(["-tu", "normal"]); }
    // File-backed diagnostics cannot fill a pipe and deadlock a long-running encoder.
    let stderr_path = dst.with_extension("stderr.txt");
    let stderr = std::fs::File::create(&stderr_path)?;
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(stderr).spawn().context("start gltfpack")?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            ensure!(status.success(), "gltfpack failed: {}", std::fs::read_to_string(&stderr_path).unwrap_or_default());
            let _ = std::fs::remove_file(stderr_path);
            return Ok(());
        }
        if start.elapsed() > Duration::from_secs(180) {
            let _ = child.kill(); // Only this exact child, never a process-name pattern.
            let _ = child.wait();
            bail!("gltfpack exceeded 180 seconds; keeping preview");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Package a local source without any fal call. Also used to replay the original bakeoff.
/// Reject implausibly changed extents instead of silently shrinking/reorienting a placed item.
pub fn compress_item(raw: &[u8], item: &CatalogItem, out_dir: &Path, log: &dyn Fn(&str)) -> Result<DetailReport> {
    ensure!(valid_id(&item.id), "invalid item id");
    ensure!(item.dims_m.iter().all(|x| x.is_finite() && *x > 0.0), "invalid preview dimensions");
    let tool = packer()?;
    let start = Instant::now();
    let source_triangles = triangles(raw)?;
    let preview = std::fs::read(out_dir.join("work").join(&item.id).join("raw.glb")).context("missing original preview geometry for alignment")?;
    let (normalized, detail_dims, yaw_correction_deg, alignment_rms) = alignment::align(raw, &preview, item.dims_m)?;
    let extent_delta = std::array::from_fn(|i| (detail_dims[i] / item.dims_m[i] - 1.0).abs());
    ensure!(extent_delta.iter().all(|d| *d <= 0.25), "detail extents differ from preview by >25% ({extent_delta:?}); keeping preview");
    let work = out_dir.join("work").join(&item.id).join("detail");
    std::fs::create_dir_all(&work)?;
    let source = work.join("normalized.glb");
    write_atomic(&source, &normalized)?;
    let version = uuid::Uuid::new_v4().simple().to_string();
    let stem = format!("{}-detail-{}", item.id, &version[..8]);
    let mut lods = Vec::new();
    for (level, target, texture) in [("high", 80_000, 2048), ("room", 25_000, 1024), ("far", 8_000, 512)] {
        let t = Instant::now();
        let temp = work.join(format!("{level}.glb"));
        let ratio = f64::from(target) / source_triangles as f64;
        log(&format!("compressing {level} detail"));
        pack(&tool, &source, &temp, ratio, texture, level == "high")?;
        if std::fs::metadata(&temp)?.len() > MAX_BYTES {
            log("detail exceeds 5 MB; retrying with ETC1S textures");
            pack(&tool, &source, &temp, ratio, texture, false)?;
        }
        ensure!(std::fs::metadata(&temp)?.len() <= MAX_BYTES, "compressed detail exceeds 5 MB; keeping preview");
        let bytes = std::fs::read(&temp)?;
        let glb = format!("items/{stem}-{level}.glb");
        let tris = triangles(&bytes)?;
        write_atomic(&out_dir.join(&glb), &bytes)?;
        lods.push(Lod { level: level.into(), glb, bytes: bytes.len() as u64, triangles: tris, seconds: t.elapsed().as_secs_f64() });
    }
    let mut upgraded = item.clone();
    upgraded.glb = lods[0].glb.clone();
    upgraded.dims_m = detail_dims;
    upgraded.license = format!("generated (fal {ENDPOINT}); approximate, not an exact product reconstruction");
    Ok(DetailReport { schema: 1, item: upgraded, preview_glb: item.glb.clone(), source_bytes: raw.len(), source_triangles,
        preview_dims_m: item.dims_m, detail_dims_m: detail_dims, extent_delta, yaw_correction_deg, alignment_rms, lods, seconds: start.elapsed().as_secs_f64() })
}

pub fn upgrade_item(fal: &Fal, item: &CatalogItem, out_dir: &Path, log: &dyn Fn(&str)) -> Result<CatalogItem> {
    ensure!(valid_id(&item.id), "invalid item id");
    let work = out_dir.join("work").join(&item.id);
    std::fs::create_dir_all(&work)?;
    let lock_path = work.join("detail.lock");
    let lock = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&lock_path)?;
    lock.try_lock().context("detail job already active for this item")?; // OS releases on crash.
    let report_path = out_dir.join("items").join(format!("{}-detail.json", item.id));
    if report_path.is_file() {
        let saved: DetailReport = serde_json::from_slice(&std::fs::read(&report_path)?)?;
        if saved.lods.iter().all(|l| out_dir.join(&l.glb).is_file()) {
            Catalog::upsert(&out_dir.join("catalog.json"), saved.item.clone())?;
            return Ok(saved.item);
        }
    }
    packer()?; // Never pay for an upgrade that cannot be packaged on this machine.
    let raw_path = work.join("detail-raw.glb");
    if !raw_path.is_file() {
        let image_url = std::fs::read_to_string(work.join("image_url.txt")).context("preview has no persisted source image URL")?;
        ensure!(!image_url.trim().is_empty(), "empty preview image URL");
        let tag = format!("furn-fs2-detail-{}", item.id);
        let output = fal_call(fal, ENDPOINT, &mesh_payload(image_url.trim()), &work.join("detail.request.json"),
            &tag, USD_DETAIL, Duration::from_secs(1200), log)?;
        write_atomic(&work.join("detail.output.json"), &serde_json::to_vec_pretty(&output)?)?;
        let url = find_glb_url(&output).context("Tripo output has no GLB")?;
        let tmp = work.join("detail-raw.download");
        fal.download(&url, &tmp)?;
        std::fs::rename(tmp, &raw_path)?;
    }
    let report = compress_item(&std::fs::read(raw_path)?, item, out_dir, log)?;
    write_atomic(&report_path, &serde_json::to_vec_pretty(&report)?)?;
    Catalog::upsert(&out_dir.join("catalog.json"), report.item.clone())?;
    log(&format!("high detail ready: {} bytes, {} triangles", report.lods[0].bytes, report.lods[0].triangles));
    Ok(report.item)
}
