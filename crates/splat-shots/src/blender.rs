//! Blender hand-off: a bundle directory that opens as a scene, plus a headless builder.
//!
//! Bundle: bundle.json (per-frame OpenCV c2w in the training frame, fov, fps, resolution, and the
//! Z-up transform `blender_from_train`), camera_path.json, mesh.glb (optional), points.ply (opaque
//! splat centres with base colour, light enough for Blender's PLY importer), splat.ply (the full
//! 3DGS PLY for Gaussian-splat add-ons), blender_build.py (runs inside Blender ->
//! scene.blend + scene.usdc + renders/).

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use glam::{DMat3, DMat4, DVec3};
use serde_json::json;

use crate::paths::{CameraPath, sample_path};
use crate::scene::SceneFrame;

const BUILD_SCRIPT: &str = include_str!("blender_build.py");
const SH_C0: f32 = 0.282_094_8;

/// Dev default: SPLAT_BLENDER, else `blender` on PATH.
pub fn blender_exe() -> PathBuf {
    std::env::var_os("SPLAT_BLENDER").map(PathBuf::from).unwrap_or_else(|| "blender".into())
}

/// Rotate scene up to +Z and move the focus point (dropped to the ground) to the origin.
pub fn blender_from_train(frame: &SceneFrame) -> DMat4 {
    let z = frame.up;
    let x = (if z.z.abs() < 0.9 { DVec3::Z } else { DVec3::X }).cross(z).normalize();
    let y = z.cross(x);
    let r = DMat3::from_cols(x, y, z).transpose(); // rows: new axes in train coords
    let t = -(r * (frame.target + frame.ground * frame.up));
    DMat4::from_cols(r.x_axis.extend(0.0), r.y_axis.extend(0.0), r.z_axis.extend(0.0), t.extend(1.0))
}

/// Opaque Gaussians' centres + SH0 colour as a plain xyz/rgb PLY. Returns the point count.
pub fn write_points_ply(splat_ply: &Path, out: &Path, min_opacity: f32) -> Result<usize> {
    let mut rd = BufReader::new(std::fs::File::open(splat_ply).with_context(|| format!("opening {}", splat_ply.display()))?);
    let (mut n, mut props, mut line) = (0usize, Vec::new(), String::new());
    loop {
        line.clear();
        if rd.read_line(&mut line)? == 0 {
            bail!("{}: no end_header", splat_ply.display());
        }
        let l = line.trim();
        if l == "end_header" {
            break;
        }
        let w: Vec<&str> = l.split_whitespace().collect();
        match w.as_slice() {
            ["element", "vertex", c] => n = c.parse()?,
            ["property", "float", name] => props.push(name.to_string()),
            ["property", ..] => bail!("{}: only all-float32 3DGS PLYs are supported", splat_ply.display()),
            ["format", f, ..] if *f != "binary_little_endian" => bail!("{}: expected binary_little_endian", splat_ply.display()),
            _ => {}
        }
    }
    let col = |name: &str| props.iter().position(|p| p == name).with_context(|| format!("PLY lacks {name}"));
    let idx = [col("x")?, col("y")?, col("z")?, col("f_dc_0")?, col("f_dc_1")?, col("f_dc_2")?, col("opacity")?];
    let mut row = vec![0u8; props.len() * 4];
    let mut body = Vec::new();
    let mut kept = 0usize;
    for _ in 0..n {
        rd.read_exact(&mut row)?;
        let f = |i: usize| f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
        if 1.0 / (1.0 + (-f(idx[6])).exp()) <= min_opacity {
            continue;
        }
        for &i in &idx[..3] {
            body.extend_from_slice(&f(i).to_le_bytes());
        }
        for &i in &idx[3..6] {
            body.push(((0.5 + SH_C0 * f(i)).clamp(0.0, 1.0) * 255.0) as u8);
        }
        kept += 1;
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
    write!(
        w,
        "ply\nformat binary_little_endian 1.0\nelement vertex {kept}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )?;
    w.write_all(&body)?;
    Ok(kept)
}

pub fn export_bundle(frame: &SceneFrame, splat_ply: &Path, path: &CameraPath, out_dir: &Path, mesh: Option<&Path>) -> Result<serde_json::Value> {
    std::fs::create_dir_all(out_dir)?;
    let frames = sample_path(path);
    let n_pts = write_points_ply(splat_ply, &out_dir.join("points.ply"), 0.5)?;
    let link = out_dir.join("splat.ply");
    let _ = std::fs::remove_file(&link);
    let abs = std::fs::canonicalize(splat_ply)?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(&abs, &link)?;
    #[cfg(not(unix))]
    std::fs::copy(&abs, &link)?; // symlinks need privileges on Windows
    if let Some(m) = mesh {
        std::fs::copy(m, out_dir.join("mesh.glb"))?;
    }
    std::fs::write(out_dir.join("blender_build.py"), BUILD_SCRIPT)?;
    std::fs::write(out_dir.join("camera_path.json"), serde_json::to_vec_pretty(path)?)?;
    // bundle.json matrices are row-major nested lists (what mathutils.Matrix expects).
    let rows = |m: &DMat4| m.transpose().to_cols_array_2d();
    let bundle = json!({
        "fps": path.fps,
        "width": path.width,
        "height": path.height,
        "c2w": frames.iter().map(|f| rows(&f.c2w)).collect::<Vec<_>>(),
        "vfov_rad": frames.iter().map(|f| f.vfov_deg.to_radians()).collect::<Vec<_>>(),
        "blender_from_train": rows(&blender_from_train(frame)),
        "mesh": mesh.map(|_| "mesh.glb"),
        "points": "points.ply",
        "splat": "splat.ply",
        "point_radius": 0.0015 * frame.radius,
    });
    std::fs::write(out_dir.join("bundle.json"), serde_json::to_vec(&bundle)?)?;
    Ok(json!({"bundle": out_dir, "points": n_pts, "frames": frames.len(), "mesh": mesh.is_some()}))
}

/// Run Blender headless on a bundle -> scene.blend, scene.usdc and workbench stills of `render_frames`.
pub fn build_blend(blender: &Path, bundle: &Path, render_frames: &[usize]) -> Result<serde_json::Value> {
    let mut cmd = Command::new(blender);
    cmd.args(["-b", "--factory-startup", "--python"]).arg(bundle.join("blender_build.py")).arg("--").arg(bundle);
    if !render_frames.is_empty() {
        cmd.arg("--render-frames").arg(render_frames.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","));
    }
    let out = cmd.output().with_context(|| format!("running {}", blender.display()))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let ok = stdout.lines().rev().find_map(|l| l.strip_prefix("BUILD_OK "));
    match (out.status.success(), ok) {
        (true, Some(j)) => {
            let mut v: serde_json::Value = serde_json::from_str(j)?;
            v["blend"] = json!(bundle.join("scene.blend"));
            Ok(v)
        }
        _ => {
            let all = format!("{stdout}{}", String::from_utf8_lossy(&out.stderr));
            bail!("blender build failed ({}): {}", out.status, crate::tail(&all, 3000))
        }
    }
}
