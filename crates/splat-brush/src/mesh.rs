//! `extract_mesh`: a detached worker renders RGB + depth from the trained splat at the capture
//! cameras plus an orbit and fuses them with splat-mesh (TSDF -> marching cubes -> GLB/PLY).
//! The main artifact is cropped to the subject the cameras look at; the whole scene is a
//! second mesh artifact.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;
use splat_mesh::glam::Vec3;
use splat_mesh::{Aabb, Camera, Intrinsics, MeshOpts, TextureOpts, View, preview};

use crate::render::{RenderCamera, Renderer, orbit_cameras, scene_cameras};
use crate::util::*;
use crate::{BrushConfig, start_worker, views};

#[derive(Serialize, Deserialize)]
pub struct MeshJob {
    pub ply: PathBuf,
    pub dataset: PathBuf,
    pub max_resolution: u32,
    pub out: PathBuf,
    /// Uncropped whole-scene mesh goes here (a second mesh artifact).
    #[serde(default)]
    pub full_out: Option<PathBuf>,
    pub formats: Vec<String>,
    pub color: bool,
    /// Bake a baseColor texture of this edge into nonempty GLBs; None = vertex colours.
    #[serde(default)]
    pub texture_size: Option<u32>,
    pub orbit_views: usize,
    pub gpu_lock: Option<PathBuf>,
}

pub fn extract_mesh(cfg: &BrushConfig, run: &dyn RunCtx, req: ExtractMesh) -> anyhow::Result<ToolResult> {
    if matches!(req.format, MeshFormat::Obj) {
        return Ok(unsupported("extract_mesh", "the brush backend writes glb or ply meshes, not obj"));
    }
    let (cand, dir) = get_artifact(run, &req.candidate_id, ArtifactKind::Candidate)?;
    let ply = views::candidate_ply(&dir)?;
    let dataset = run.path(meta_str(&cand, "sfm_id")?)?;
    let format = serde_json::to_value(req.format)?;
    let color = serde_json::to_value(req.color)?;
    let (full, full_out) = new_artifact(run, ArtifactKind::Mesh, json!({"candidate_id": req.candidate_id, "format": format, "color": color, "ply": rel(run, &ply), "region": "full_scene"}))?;
    let (artifact, out) = new_artifact(run, ArtifactKind::Mesh, json!({"candidate_id": req.candidate_id, "format": format, "color": color, "ply": rel(run, &ply), "region": "subject", "full_mesh_id": full.id}))?;
    let job = MeshJob {
        ply,
        dataset,
        max_resolution: cand.meta.get("max_resolution").and_then(Value::as_u64).unwrap_or(1920).min(1280) as u32,
        out: out.clone(),
        full_out: Some(full_out),
        formats: vec![format.as_str().unwrap_or("glb").to_string()],
        color: !matches!(req.color, MeshColor::None),
        texture_size: matches!(req.color, MeshColor::Texture).then_some(req.texture_size),
        orbit_views: 24,
        gpu_lock: cfg.gpu_lock.clone(),
    };
    write_json(&out.join("mesh_job.json"), &job)?;
    let mut r = start_worker(cfg, run, "extract_mesh", "mesh", &out.join("mesh_job.json"), &artifact)?;
    r.artifacts.push(full);
    Ok(r)
}

fn to_mesh_camera(c: &RenderCamera) -> Camera {
    Camera::from_c2w(c.rotation, c.position, Intrinsics { fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy, width: c.width, height: c.height })
}

/// The subject the capture looks at: a cube around the cameras' common look-at point. Its
/// half-size is twice the 90th-percentile miss distance of the optical axes, kept between 0.4
/// and 0.7 of the median camera distance (orbits put the subject well inside their ring).
pub fn subject_bounds(capture: &[RenderCamera]) -> Option<Aabb> {
    if capture.len() < 3 {
        return None;
    }
    // Captures whose axes don't converge (walks, slides) have no subject to crop to.
    let cams: Vec<Camera> = capture.iter().map(to_mesh_camera).collect();
    let target = splat_mesh::converging_target(&cams)?;
    let mut dist: Vec<f32> = capture.iter().map(|c| (c.position - target).length()).collect();
    let mut miss: Vec<f32> = capture
        .iter()
        .map(|c| {
            let d = target - c.position;
            (d - c.forward() * d.dot(c.forward())).length()
        })
        .collect();
    dist.sort_by(f32::total_cmp);
    miss.sort_by(f32::total_cmp);
    let d = dist[dist.len() / 2];
    // A well-conditioned intersection alone also accepts room walks. An object orbit must
    // aim most optical axes near the same target, and keep it in front of nearly every view.
    // Camera audit (ahead %, p90 miss/d): truck84 (100, .680), garden (100, .063),
    // bicycle (100, .261), phone-bike (100, .259), gimbal (100, .008), driveway (0, .004;
    // already ill-conditioned). Room3 is (57.9, 1.040): ahead rejects this room walk,
    // while the .75 miss limit preserves the broad truck orbit that .35 rejected.
    let ahead = capture.iter().filter(|c| (target - c.position).dot(c.forward()) > 0.0).count();
    if 10 * ahead < 9 * capture.len() || miss[miss.len() * 9 / 10] > 0.75 * d {
        return None;
    }
    let half = (2.0 * miss[miss.len() * 9 / 10]).clamp(0.4 * d, 0.7 * d);
    (d.is_finite() && half > 0.0).then(|| Aabb { min: target - Vec3::splat(half), max: target + Vec3::splat(half) })
}

/// Most capture views a mesh fuses and bakes from (SPLAT_MESH_MAX_VIEWS overrides; 0 = all).
const MESH_MAX_VIEWS: usize = 200;

fn mesh_max_views() -> usize {
    std::env::var("SPLAT_MESH_MAX_VIEWS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(MESH_MAX_VIEWS)
}

/// At most `max` of `cams`, evenly spaced and in order (first and last kept); all when `max` is 0.
fn mesh_views<T: Clone>(cams: &[T], max: usize) -> Vec<T> {
    if max == 0 || cams.len() <= max {
        return cams.to_vec();
    }
    if max == 1 {
        return vec![cams[0].clone()];
    }
    (0..max).map(|i| cams[i * (cams.len() - 1) / (max - 1)].clone()).collect()
}

/// Worker body.
pub fn run_job(job: MeshJob) -> anyhow::Result<i32> {
    let _gpu = crate::gpu_lock(job.gpu_lock.as_deref())?;
    let renderer = Renderer::open(&job.ply)?;
    let capture: Vec<RenderCamera> = scene_cameras(&job.dataset, job.max_resolution)?.into_iter().map(|(_, c)| c).collect();
    let subject = subject_bounds(&capture);
    // Fusion and texture baking render every view, so their time grows with the frame count (room 3:
    // 1m22s at 170 frames, past the pipeline's mesh budget at 362). Evenly spaced views (capture order)
    // keep the coverage; the subject bounds and orbit still use every camera.
    let views = mesh_views(&capture, mesh_max_views());
    if views.len() < capture.len() {
        eprintln!("mesh: fusing {} of {} capture views (evenly spaced; SPLAT_MESH_MAX_VIEWS)", views.len(), capture.len());
    }
    let mut cams: Vec<Camera> = views.iter().map(to_mesh_camera).collect();
    let capture_cams: Vec<Camera> = views.iter().map(to_mesh_camera).collect();
    // Synthetic object orbits outside an interior room can carve its walls from the back.
    if subject.is_some() {
        cams.extend(orbit_cameras(&capture, job.orbit_views).iter().map(to_mesh_camera));
    }
    let mut targets = vec![];
    if subject.is_some() || job.full_out.is_none() {
        targets.push((job.out.clone(), subject));
    } else {
        eprintln!("Subject mesh skipped: camera axes do not form an object orbit; reconstructing the whole scene.");
        std::fs::create_dir_all(&job.out)?;
        for f in &job.formats {
            let path = job.out.join(format!("mesh.{f}"));
            if path.exists() { std::fs::remove_file(path)?; }
        }
        if job.out.join("previews_full").exists() {
            std::fs::remove_dir_all(job.out.join("previews_full"))?;
        }
        write_json(&job.out.join("mesh.json"), &json!({
            "region": "subject", "status": "skipped", "triangles": 0, "vertices": 0, "files": [],
            "warning": "Subject mesh skipped: capture cameras do not form an object orbit. Use the whole-scene mesh."
        }))?;
    }
    if let Some(full) = &job.full_out {
        targets.push((full.clone(), None));
    }
    let render = |c: &Camera| -> anyhow::Result<View> {
        let k = c.intrinsics;
        let rc = RenderCamera { position: c.position(), rotation: c.rotation(), fx: k.fx, fy: k.fy, cx: k.cx, cy: k.cy, width: k.width, height: k.height };
        let f = renderer.render(&rc, true)?;
        // Median depth (2DGS eq. 19): expected depth averages foreground and background
        // splats into floating surfaces (DTU scan 24: 5.63 -> 3.85 mm Chamfer, B15).
        let mut v = View { camera: *c, rgb: f.rgb.into_raw(), alpha: f.alpha, depth: f.median_depth };
        v.unpremultiply();
        Ok(v)
    };
    // Render every camera once for all regions (subject + whole scene).
    let opts: Vec<MeshOpts> = targets.iter().map(|(_, bounds)| MeshOpts {
        bounds: *bounds,
        crop_to_target: bounds.is_some(),
        // Whole rooms use adaptive voxels. 512 MiB covers voxel storage, with extra room for
        // hash tables, meshes and texture baking. Retain object extraction's existing detail.
        max_voxel_memory_bytes: bounds.is_none().then_some(512 * 1024 * 1024),
        // Room surfaces can be disconnected across uncaptured doorways and furniture.
        min_component_ratio: if bounds.is_none() { 0.001 } else { MeshOpts::default().min_component_ratio },
        ..MeshOpts::default()
    }).collect();
    let meshes = splat_mesh::meshes_from_renders(&cams, &render, &opts)?;
    for ((out, bounds), (mut mesh, stats)) in targets.into_iter().zip(meshes) {
        if !job.color {
            mesh.colors.iter_mut().for_each(|c| *c = [200, 200, 200]);
        }
        // Both regions carry the requested texture. Empty regions report evidence without
        // writing a placeholder GLB that would otherwise appear as a usable deliverable.
        let textured = match job.texture_size.filter(|_| {
            job.color && !mesh.triangles.is_empty() && job.formats.iter().any(|f| f == "glb")
        }) {
            Some(size) => Some(splat_mesh::bake_texture(&mesh, &capture_cams, &render, &TextureOpts { size, ..TextureOpts::default() })?),
            None => None,
        };
        let mut files = vec![];
        for f in &job.formats {
            let path = out.join(format!("mesh.{f}"));
            if mesh.triangles.is_empty() {
                // A resumed worker must remove an earlier empty placeholder as well.
                if path.exists() {
                    std::fs::remove_file(path)?;
                }
                continue;
            }
            match f.as_str() {
                "glb" if textured.is_some() => splat_mesh::write_glb_textured(textured.as_ref().expect("checked"), &path)?,
                "glb" => splat_mesh::write_glb(&mesh, &path)?,
                "ply" => splat_mesh::write_ply(&mesh, &path)?,
                other => anyhow::bail!("unsupported mesh format {other}"),
            }
            files.push(path);
        }
        if mesh.triangles.is_empty() && out.join("previews_full").exists() {
            std::fs::remove_dir_all(out.join("previews_full"))?;
        }
        std::fs::create_dir_all(out.join("previews_full"))?;
        for (i, img) in preview::orbit_previews(&mesh, job.color, &capture_cams, None, 4, 512).iter().enumerate() {
            img.save(out.join(format!("previews_full/orbit_{i}.png")))?;
        }
        let mut stats = stats.to_json();
        stats["files"] = json!(files);
        stats["region"] = json!(if bounds.is_some() { "subject" } else { "full_scene" });
        stats["bounds_policy"] = json!(if bounds.is_some() { "object_orbit" } else { "scene_depth" });
        stats["status"] = json!(if mesh.triangles.is_empty() { "empty" } else { "succeeded" });
        stats["mesh_bounds_min"] = json!(mesh.bounds().map(|b| b.min.to_array()));
        stats["mesh_bounds_max"] = json!(mesh.bounds().map(|b| b.max.to_array()));
        if mesh.triangles.is_empty() {
            stats["warning"] = json!("No surface in this region; no mesh file was exported. Room captures may have no single subject; use the whole-scene mesh.");
        }
        if let Some(t) = &textured {
            stats["texture"] = json!({"size": t.texture.width(), "triangles": t.triangle_count(), "seen_fraction": t.seen_fraction});
            t.texture.save(out.join("texture.png"))?;
        }
        write_json(&out.join("mesh.json"), &stats)?;
    }
    Ok(0)
}

/// Orbit previews in `dir`, none when it is missing: an empty mesh renders no previews, and the
/// remote result sync carries files only, so its empty `previews_full` never reaches the run.
fn list_previews(dir: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    if !dir.exists() {
        eprintln!("Mesh previews missing: {}", dir.display());
        return Ok(Vec::new());
    }
    list_images(dir)
}

pub fn complete(run: &dyn RunCtx, artifact_id: &str, exit_code: i32, log: &str) -> anyhow::Result<ToolResult> {
    let (artifact, dir) = get_artifact(run, artifact_id, ArtifactKind::Mesh)?;
    let full = artifact.meta.get("full_mesh_id").and_then(Value::as_str).map(|id| get_artifact(run, id, ArtifactKind::Mesh)).transpose()?;
    if exit_code != 0 {
        let mut r = failure("extract_mesh", "subprocess_failed", format!("mesh worker exited {exit_code}: {log}"));
        r.artifacts = std::iter::once(artifact).chain(full.map(|f| f.0)).collect();
        return Ok(r);
    }
    let Ok(Value::Object(mut metrics)) = read_json::<Value>(&dir.join("mesh.json")) else {
        let mut r = failure("extract_mesh", "subprocess_failed", format!("mesh worker exited {exit_code}: {log}"));
        r.artifacts = std::iter::once(artifact).chain(full.map(|f| f.0)).collect();
        return Ok(r);
    };
    let files = |m: &Value| -> Vec<String> { m["files"].as_array().into_iter().flatten().filter_map(Value::as_str).map(|p| rel(run, std::path::Path::new(p))).collect() };
    metrics.insert("files".into(), json!(files(&Value::Object(metrics.clone()))));
    metrics.insert("coordinate_frame".into(), json!("sfm"));
    let mut previews = list_previews(&dir.join("previews_full"))?;
    let mut artifacts = vec![artifact];
    if let Some((full, full_dir)) = full {
        if let Ok(m) = read_json::<Value>(&full_dir.join("mesh.json")) {
            let mut full_metrics = m;
            full_metrics["mesh_id"] = json!(full.id);
            full_metrics["files"] = json!(files(&full_metrics));
            metrics.insert("full_scene".into(), full_metrics);
            previews.extend(list_previews(&full_dir.join("previews_full"))?.into_iter().take(1));
        }
        artifacts.push(full);
    }
    let images = splat_eval::budget_jpegs(&previews, &dir.join("previews"), splat_eval::RESULT_IMAGE_BUDGET)?;
    let mut warnings = vec![];
    if metrics["triangles"].as_u64().unwrap_or(0) == 0 {
        warnings.push(metrics.get("warning").and_then(Value::as_str)
            .unwrap_or("Subject mesh is empty; no subject file was exported. Use the whole-scene mesh for room captures.").into());
    } else if images.is_empty() {
        warnings.push("Mesh has no previews.".into());
    }
    let has_subject = metrics["triangles"].as_u64().unwrap_or(0) > 0;
    let has_full = metrics.get("full_scene").is_some_and(|m| m["triangles"].as_u64().unwrap_or(0) > 0);
    if metrics.get("full_scene").is_some() && !has_full {
        warnings.push("Whole-scene mesh is empty; no whole-scene file was exported.".into());
    }
    let outcome = if has_subject || has_full {
        result("extract_mesh", Status::Succeeded)
    } else {
        failure("extract_mesh", "empty_mesh", "No surface could be reconstructed; no mesh files were exported.")
    };
    Ok(ToolResult {
        artifacts,
        metrics: metrics.into_iter().collect(),
        images: images.iter().map(|p| rel(run, p)).collect(),
        warnings,
        ..outcome
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_views_evenly_spaced() {
        let v: Vec<usize> = (0..362).collect();
        let s = mesh_views(&v, 200);
        assert_eq!(s.len(), 200);
        assert_eq!((s[0], s[199]), (0, 361));
        assert!(s.windows(2).all(|w| w[1] > w[0] && w[1] - w[0] <= 2));
        assert_eq!(mesh_views(&v[..170], 200).len(), 170);
        assert_eq!(mesh_views(&v, 0).len(), 362);
    }

    #[test]
    fn list_previews_tolerates_missing_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(list_previews(&tmp.path().join("previews_full")).unwrap().is_empty());
        let dir = tmp.path().join("present");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("orbit_0.png"), b"").unwrap();
        assert_eq!(list_previews(&dir).unwrap(), vec![dir.join("orbit_0.png")]);
    }
}
