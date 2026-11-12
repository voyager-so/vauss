//! Mesh a Brush-trained splat end to end and write previews.
//!
//! `cargo run --release -p splat-mesh --example mesh_splat -- <splat.ply> <sfm dataset dir> <out dir>
//!  [--max-res 960] [--orbit 0] [--resolution 512] [--max-tris 1500000]`
//!
//! Writes mesh.glb, mesh.ply, stats.json and preview_<i>.png (splat | mesh colour | mesh clay)
//! for four capture views. `--cache DIR` saves the full-resolution renders (and splat previews)
//! on first use and reuses them afterwards without touching the GPU, for parameter sweeps;
//! `--trunc`, `--min-alpha`, `--edge-tol`, `--no-carve`, `--min-comp` override `MeshOpts`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{GenericImage, RgbImage};
use splat_brush::render::{RenderCamera, Renderer, orbit_cameras, scene_cameras};
use splat_mesh::preview::{Shade, render};
use splat_mesh::{Camera, Intrinsics, MeshOpts, View, mesh_from_renders, write_glb, write_ply};

fn to_mesh_cam(c: &RenderCamera) -> Camera {
    Camera::from_c2w(c.rotation, c.position, Intrinsics { fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy, width: c.width, height: c.height })
}

fn to_render_cam(c: &Camera) -> RenderCamera {
    let k = c.intrinsics;
    RenderCamera { position: c.position(), rotation: c.rotation(), fx: k.fx, fy: k.fy, cx: k.cx, cy: k.cy, width: k.width, height: k.height }
}

fn save_view(path: &Path, v: &View) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let k = v.camera.intrinsics;
    f.write_all(&k.width.to_le_bytes())?;
    f.write_all(&k.height.to_le_bytes())?;
    f.write_all(&v.rgb)?;
    for x in v.alpha.iter().chain(&v.depth) {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn load_view(path: &Path, camera: Camera) -> Result<View> {
    let mut b = vec![];
    std::fs::File::open(path)?.read_to_end(&mut b)?;
    let n = (u32::from_le_bytes(b[0..4].try_into()?) * u32::from_le_bytes(b[4..8].try_into()?)) as usize;
    let rgb = b[8..8 + 3 * n].to_vec();
    let fl: Vec<f32> = b[8 + 3 * n..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect();
    Ok(View { camera, rgb, alpha: fl[..n].to_vec(), depth: fl[n..].to_vec() })
}

/// Nearest-sample a cached full-resolution view down to `camera`'s (smaller) size.
fn downsample(v: &View, camera: Camera) -> View {
    let (src, dst) = (v.camera.intrinsics, camera.intrinsics);
    let n = (dst.width * dst.height) as usize;
    let (mut rgb, mut alpha, mut depth) = (vec![0; 3 * n], vec![0.0; n], vec![0.0; n]);
    for y in 0..dst.height {
        for x in 0..dst.width {
            let sx = ((x as f32 + 0.5) * src.width as f32 / dst.width as f32) as u32;
            let sy = ((y as f32 + 0.5) * src.height as f32 / dst.height as f32) as u32;
            let (i, j) = ((y * dst.width + x) as usize, (sy.min(src.height - 1) * src.width + sx.min(src.width - 1)) as usize);
            rgb[3 * i..3 * i + 3].copy_from_slice(&v.rgb[3 * j..3 * j + 3]);
            alpha[i] = v.alpha[j];
            depth[i] = v.depth[j];
        }
    }
    View { camera, rgb, alpha, depth }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str, default: u64| -> u64 {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let pos: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || !args[i - 1].starts_with("--"))).map(|(_, a)| a).collect();
    let (ply, dataset, out) = (PathBuf::from(pos[0]), PathBuf::from(pos[1]), PathBuf::from(pos[2]));
    std::fs::create_dir_all(&out)?;
    let t0 = std::time::Instant::now();
    let capture: Vec<RenderCamera> = scene_cameras(&dataset, flag("--max-res", 960) as u32)?.into_iter().map(|(_, c)| c).collect();
    let mut cams: Vec<RenderCamera> = capture.clone();
    cams.extend(orbit_cameras(&capture, flag("--orbit", 0) as usize));
    let fflag = |name: &str| -> Option<f32> { args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()) };
    let cache = args.iter().position(|a| a == "--cache").map(|i| PathBuf::from(&args[i + 1]));
    let cached = cache.as_ref().is_some_and(|c| c.join("done").is_file());
    let renderer = if cached { None } else { Some(Renderer::open(&ply).context("loading splat")?) };
    eprintln!("{:?} splats, {} cameras, load {:.1}s", renderer.as_ref().map(Renderer::num_splats), cams.len(), t0.elapsed().as_secs_f64());
    let mesh_cams: Vec<Camera> = cams.iter().map(to_mesh_cam).collect();
    let d = MeshOpts::default();
    let opts = MeshOpts {
        resolution: flag("--resolution", 512) as u32,
        max_triangles: Some(flag("--max-tris", 1_500_000) as usize),
        trunc_voxels: fflag("--trunc").unwrap_or(d.trunc_voxels),
        min_alpha: fflag("--min-alpha").unwrap_or(d.min_alpha),
        depth_edge_tol: fflag("--edge-tol").unwrap_or(d.depth_edge_tol),
        min_component_ratio: fflag("--min-comp").unwrap_or(d.min_component_ratio),
        smooth_iterations: fflag("--smooth").map_or(d.smooth_iterations, |v| v as u32),
        carve: !args.iter().any(|a| a == "--no-carve"),
        ..d
    };
    if let Some(c) = &cache {
        std::fs::create_dir_all(c)?;
    }
    let mut render_s = 0.0;
    let t = std::time::Instant::now();
    let (mesh, stats) = mesh_from_renders(
        &mesh_cams,
        |c| {
            let t = std::time::Instant::now();
            let idx = mesh_cams.iter().position(|m| m.world_to_cam == c.world_to_cam).expect("known camera");
            let full = mesh_cams[idx];
            let path = cache.as_ref().map(|d| d.join(format!("view_{idx:04}.bin")));
            let v = if cached {
                load_view(path.as_ref().expect("cache"), full)?
            } else {
                let f = renderer.as_ref().expect("renderer").render(&to_render_cam(c), true)?;
                let mut v = View { camera: *c, rgb: f.rgb.into_raw(), alpha: f.alpha, depth: f.depth };
                v.unpremultiply();
                if let Some(p) = &path
                    && c.intrinsics == full.intrinsics
                {
                    save_view(p, &v)?;
                }
                v
            };
            render_s += t.elapsed().as_secs_f64();
            Ok(if v.camera.intrinsics == c.intrinsics { v } else { downsample(&v, *c) })
        },
        &opts,
    )?;
    if let Some(c) = &cache {
        std::fs::write(c.join("done"), "")?;
    }
    let total = t.elapsed().as_secs_f64();
    let mut js = stats.to_json();
    js["render_s"] = render_s.into();
    js["mesh_total_s"] = total.into();
    js["opts"] = format!("{opts:?}").into();
    eprintln!("{js}");
    write_glb(&mesh, &out.join("mesh.glb"))?;
    write_ply(&mesh, &out.join("mesh.ply"))?;
    std::fs::write(out.join("stats.json"), serde_json::to_vec_pretty(&js)?)?;
    for (n, i) in [0, capture.len() / 4, capture.len() / 2, 3 * capture.len() / 4].into_iter().enumerate() {
        let rc = capture[i];
        let splat_png = cache.as_ref().map(|c| c.join(format!("splat_{n}.png")));
        let splat = match (&renderer, &splat_png) {
            (Some(r), p) => {
                let img = r.render(&rc, false)?.rgb;
                if let Some(p) = p {
                    img.save(p)?;
                }
                img
            }
            (None, Some(p)) => image::open(p)?.to_rgb8(),
            (None, None) => unreachable!("no renderer without a cache"),
        };
        let mc = to_mesh_cam(&rc);
        let color = render(&mesh, &mc, Shade::Color, [0, 0, 0]);
        let clay = render(&mesh, &mc, Shade::Clay, [0, 0, 0]);
        let (w, h) = (rc.width, rc.height);
        let mut canvas = RgbImage::new(w * 3, h);
        canvas.copy_from(&splat, 0, 0)?;
        canvas.copy_from(&color, w, 0)?;
        canvas.copy_from(&clay, 2 * w, 0)?;
        canvas.save(out.join(format!("preview_{n}.png")))?;
    }
    eprintln!("wrote {} in {:.1}s total", out.display(), t0.elapsed().as_secs_f64());
    Ok(())
}
