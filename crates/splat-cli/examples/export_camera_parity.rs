//! Hardware-GPU acceptance harness. Never run this on an interactive workstation.
//! cargo run --locked -p splat-cli --example export_camera_parity --
//!   --source splat.ply --data capture --package exports/web --image frame_000038.jpg --out parity
use anyhow::{Context, Result, ensure};
use clap::Parser;
use glam::{DMat4, Mat4};
use serde_json::{Value, json};
use splat_brush::render::{Frame, RawSplats, RenderCamera, Renderer};
use splat_webexport::{
    ply::{self, Gaussians, RenderMode},
    spz,
};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    source: PathBuf,
    /// COLMAP capture directory, with sparse/0 or cameras.bin/images.bin directly inside.
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    package: PathBuf,
    #[arg(long)]
    image: String,
    #[arg(long)]
    out: PathBuf,
    /// Exact, uncropped transformed PLY: only float32/GPU rounding is expected.
    #[arg(long, default_value_t = 50.0)]
    min_exact_psnr: f64,
    /// Ordinary import includes export cropping and SPZ quantization when applicable.
    #[arg(long, default_value_t = 30.0)]
    min_export_psnr: f64,
}

fn camera(pose: DMat4, width: u32, height: u32, k: [f64; 4]) -> RenderCamera {
    let pose = Mat4::from_cols_array(&pose.to_cols_array().map(|v| v as f32));
    let (_, rotation, position) = pose.to_scale_rotation_translation();
    RenderCamera {
        position,
        rotation,
        width,
        height,
        fx: k[0] as f32,
        fy: k[1] as f32,
        cx: k[2] as f32,
        cy: k[3] as f32,
    }
}

fn raw(g: &Gaussians) -> Result<RawSplats> {
    ensure!(
        g.neural.is_none() && matches!(g.render_mode, RenderMode::Default | RenderMode::Mip),
        "harness supports SH Gaussian default/mip models, not neural/surfel approximations"
    );
    let k = g.sh_coeffs() * 3;
    let sh_coeffs = (0..g.len())
        .flat_map(|i| {
            g.dc[i]
                .into_iter()
                .chain(g.rest[i * k..(i + 1) * k].iter().copied())
        })
        .collect();
    Ok(RawSplats {
        means: g.pos.iter().flatten().copied().collect(),
        rotations: g.rot.iter().flatten().copied().collect(),
        log_scales: g.scale.iter().flatten().copied().collect(),
        sh_coeffs,
        raw_opacities: g.opacity.clone(),
        mip: g.antialiased,
    })
}

fn metrics(a: &Frame, b: &Frame) -> Value {
    let pairs = a.rgb.as_raw().iter().zip(b.rgb.as_raw());
    let n = a.rgb.as_raw().len() as f64;
    let (mut sq, mut abs, mut max) = (0.0f64, 0.0f64, 0u8);
    for (a, b) in pairs {
        let d = a.abs_diff(*b);
        sq += (d as f64).powi(2);
        abs += d as f64;
        max = max.max(d);
    }
    let psnr = if sq == 0.0 {
        999.0
    } else {
        10.0 * (255.0f64.powi(2) / (sq / n)).log10()
    };
    let alpha_mae = a
        .alpha
        .iter()
        .zip(&b.alpha)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / a.alpha.len() as f64;
    json!({"psnr_db": psnr, "rgb_mae_255": abs/n, "rgb_max_error_255": max, "alpha_mae": alpha_mae})
}

fn main() -> Result<()> {
    let a = Args::parse();
    let scene: Value = serde_json::from_slice(&std::fs::read(a.package.join("scene.json"))?)?;
    let capture = splat_shots::scene::load_scene(
        &a.data,
        splat_shots::scene_transform_near(&a.source).as_deref(),
    )?;
    let source_camera = capture
        .cameras
        .iter()
        .find(|c| c.name == a.image)
        .context("image absent from source COLMAP")?;
    ensure!(
        matches!(source_camera.model.as_str(), "PINHOLE" | "SIMPLE_PINHOLE"),
        "undistort capture before this pinhole test"
    );
    let native_camera = camera(
        source_camera.c2w,
        source_camera.width,
        source_camera.height,
        [
            source_camera.fx,
            source_camera.fy,
            source_camera.cx,
            source_camera.cy,
        ],
    );
    let c = scene["capture_cameras"]
        .as_array()
        .context("no capture cameras")?
        .iter()
        .find(|c| c["image"] == a.image)
        .context("image absent from exact export cameras")?;
    let export_pose =
        DMat4::from_cols_array(&serde_json::from_value::<[f64; 16]>(c["c2w"].clone())?);
    let export_camera = camera(
        export_pose,
        c["width"].as_u64().context("width")? as u32,
        c["height"].as_u64().context("height")? as u32,
        [
            c["fx"].as_f64().context("fx")?,
            c["fy"].as_f64().context("fy")?,
            c["cx"].as_f64().context("cx")?,
            c["cy"].as_f64().context("cy")?,
        ],
    );
    ensure!(
        (native_camera.width, native_camera.height) == (export_camera.width, export_camera.height),
        "image dimensions differ"
    );
    let transform = DMat4::from_cols_array(&serde_json::from_value::<[f64; 16]>(
        scene["transform_from_source"].clone(),
    )?);
    let mut transformed = ply::read_ply(&a.source)?;
    raw(&transformed)?; // reject unsupported appearance before opening a GPU
    splat_webexport::to_scene_frame(&mut transformed, &transform);
    std::fs::create_dir_all(&a.out)?;
    let reference_path = a.out.join("transformed-reference.ply");
    ply::write_ply(&transformed, &reference_path)?;
    let standalone = scene["splat_overflow"]["standalone_splat"]
        .as_str()
        .unwrap_or("scene.spz");
    ensure!(
        standalone == "scene.spz" || standalone == "scene.ply",
        "unexpected standalone representation"
    );
    if standalone == "scene.spz" {
        ensure!(
            scene["splat_overflow"]["overflow_count"]
                .as_u64()
                .unwrap_or(0)
                == 0
                && scene["splat_overflow"]["dc_patch_count"]
                    .as_u64()
                    .unwrap_or(0)
                    == 0,
            "ordinary SPZ is incomplete; regenerate package"
        );
    }
    // Deliberately read ONLY the ordinary import file, not the sidecar-aware package reader.
    let ordinary = if standalone == "scene.ply" {
        ply::read_ply(&a.package.join(standalone))?
    } else {
        spz::read_spz(&a.package.join(standalone))?
    };
    raw(&ordinary)?;
    ensure!(
        Renderer::has_hardware_adapter()?,
        "requires parent-coordinated cloud/CI GPU"
    );
    let mut renderer = Renderer::open(&a.source)?;
    let native = renderer.render(&native_camera, false)?;
    native.rgb.save(a.out.join("native.png"))?;
    renderer.load(&reference_path)?;
    let exact = renderer.render(&export_camera, false)?;
    exact.rgb.save(a.out.join("exact-ply.png"))?;
    renderer.set_raw(raw(&ordinary)?);
    let exported = renderer.render(&export_camera, false)?;
    exported.rgb.save(a.out.join("ordinary-export.png"))?;
    let exact_metrics = metrics(&native, &exact);
    let export_metrics = metrics(&native, &exported);
    let passed = exact_metrics["psnr_db"].as_f64().unwrap() >= a.min_exact_psnr
        && export_metrics["psnr_db"].as_f64().unwrap() >= a.min_export_psnr;
    let report = json!({"passed":passed, "source":a.source, "package":a.package, "image":a.image,
        "ordinary_file":standalone, "source_splats":transformed.len(), "export_splats":ordinary.len(),
        "K":[source_camera.fx,source_camera.fy,source_camera.cx,source_camera.cy], "width":source_camera.width,"height":source_camera.height,
        "native_vs_exact_ply":exact_metrics,"native_vs_ordinary_export":export_metrics,
        "thresholds":{"exact_psnr_db":a.min_exact_psnr,"export_psnr_db":a.min_export_psnr},
        "scope":"same native rasterizer, all SH bands and declared mip mode, black background; not a Blender/film rasterizer equivalence claim"});
    std::fs::write(
        a.out.join("metrics.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        passed,
        "native/export parity thresholds failed; see metrics.json and PNGs"
    );
    Ok(())
}
