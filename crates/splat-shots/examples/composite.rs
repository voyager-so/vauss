//! Prototype driver for splat_shots::composite (becomes `splat composite` once the lead agrees).
//!   composite ground <data_dir>                                   -> plane JSON on stdout
//!   composite track <data_dir> <frames_dir> <masks_dir> <out_dir> <min_px> <times.json> <fps>
//!        frames_dir/<name>  masks_dir/<stem>.png (all people; split into connected components)
//!        -> out_dir/layers.json (+ cut-outs)
//!   composite matte <rgba_dir> <x,y,z> <height_world> <out.json>  -> one layer standing at x,y,z
//!   composite render <ply> <path.json> <shot_dir>
//!   composite run <shot_dir> <spec.json> <out_dir>
//! spec.json = {"layers": [...] | "layers_file": "...", "ground": {...}, "shadow": .., "gain": [..]}

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;
use splat_shots::composite::*;
use splat_shots::encode::Ffmpeg;
use splat_shots::layout::ShotDir;
use splat_shots::scene::load_scene;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("ground") => {
            let scene = load_scene(Path::new(&a[2]), None)?;
            let plane = fit_ground(&scene.points, scene.up)?;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({"ground": plane, "up": scene.up.to_array(), "target": scene.target.to_array(), "radius": scene.radius}))?);
        }
        Some("track") => {
            // track <data_dir> <frames_dir> <masks_dir> <out_dir> <min_px> <times.json> <fps>
            // times.json maps SfM image name -> source time (s); frame i of frames_dir is at i / fps and
            // gets a camera interpolated between the registered SfM images around it.
            let (data, frames, masks, out) = (Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4]), PathBuf::from(&a[5]));
            let min_px: usize = a[6].parse()?;
            let times: std::collections::BTreeMap<String, f64> = serde_json::from_str(&std::fs::read_to_string(&a[7])?)?;
            let fps: f64 = a[8].parse()?;
            let scene = load_scene(data, None)?;
            let ground = fit_ground(&scene.points, scene.up)?;
            let mut timed: Vec<(f64, &splat_shots::scene::CaptureCamera)> =
                scene.cameras.iter().filter_map(|c| times.get(&c.name).map(|t| (*t, c))).collect();
            timed.sort_by(|x, y| x.0.total_cmp(&y.0));
            anyhow::ensure!(timed.len() >= 2, "need 2+ registered cameras with times");
            let mut names: Vec<String> = std::fs::read_dir(frames)?.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
            names.sort();
            let cams: Vec<splat_shots::scene::CaptureCamera> = (0..names.len())
                .map(|i| {
                    let t = i as f64 / fps;
                    let k = timed.partition_point(|x| x.0 <= t).clamp(1, timed.len() - 1);
                    let ((t0, c0), (t1, c1)) = (timed[k - 1], timed[k]);
                    let w = ((t - t0) / (t1 - t0).max(1e-9)).clamp(0.0, 1.0);
                    let (_, r0, p0) = c0.c2w.to_scale_rotation_translation();
                    let (_, r1, p1) = c1.c2w.to_scale_rotation_translation();
                    let c2w = glam::DMat4::from_rotation_translation(r0.slerp(r1, w), p0.lerp(p1, w));
                    splat_shots::scene::CaptureCamera { name: names[i].clone(), c2w, width: c0.width, height: c0.height, fy: c0.fy }
                })
                .collect();
            let mut inputs = vec![];
            let unregistered = 0;
            for (n, cam) in names.iter().zip(&cams) {
                let stem = Path::new(n).file_stem().unwrap().to_string_lossy();
                let mp = masks.join(format!("{stem}.png"));
                let m = if mp.exists() { image::open(&mp)?.to_luma8() } else { image::GrayImage::new(1, 1) };
                inputs.push(TrackInput { camera: cam, image: frames.join(n), masks: components(&m, min_px) });
            }
            eprintln!("{} frames tracked, {} not registered by SfM", inputs.len(), unregistered);
            let radius = scene.radius;
            let layers = track_layers(&inputs, &ground, &out, 0.08 * radius, 5)?;
            std::fs::create_dir_all(&out)?;
            std::fs::write(out.join("layers.json"), serde_json::to_vec_pretty(&serde_json::json!({"layers": layers, "ground": ground}))?)?;
            for l in &layers {
                eprintln!("layer {}: {} frames, offset {}", l.name, l.frames.len(), l.offset);
            }
        }
        Some("matte") => {
            let dir = Path::new(&a[2]);
            let xyz: Vec<f64> = a[3].split(',').map(|s| s.parse()).collect::<Result<_, _>>()?;
            let height: f64 = a[4].parse()?;
            let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "png")).collect();
            files.sort();
            // Scale from the tallest pose so crouching doesn't shrink the person.
            let mut frames = vec![];
            let mut tallest = 0u32;
            for f in &files {
                let img = image::open(f)?.to_rgba8();
                let alpha = image::GrayImage::from_fn(img.width(), img.height(), |x, y| image::Luma([img.get_pixel(x, y).0[3]]));
                let Some((x0, y0, x1, y1)) = mask_bbox(&alpha) else { continue };
                tallest = tallest.max(y1 - y0 + 1);
                frames.push(LayerFrame { image: f.clone(), foot_px: [(x0 + x1) as f64 / 2.0, y1 as f64 + 0.5], px_per_world: 0.0, anchor: [xyz[0], xyz[1], xyz[2]] });
            }
            if frames.is_empty() {
                bail!("no matte frames with alpha in {}", dir.display());
            }
            // Steady the foot x (talking/walking in place): median-smooth over 9 frames.
            let xs: Vec<f64> = frames.iter().map(|f| f.foot_px[0]).collect();
            let ys: Vec<f64> = frames.iter().map(|f| f.foot_px[1]).collect();
            for (i, f) in frames.iter_mut().enumerate() {
                let (lo, hi) = (i.saturating_sub(4), (i + 4).min(xs.len() - 1));
                let mut wx = xs[lo..=hi].to_vec();
                let mut wy = ys[lo..=hi].to_vec();
                f.foot_px = [splat_shots::scene::median(std::mem::take(&mut wx)), splat_shots::scene::median(std::mem::take(&mut wy))];
                f.px_per_world = tallest as f64 / height;
            }
            let layer = Layer { name: dir.file_name().unwrap().to_string_lossy().into_owned(), frames, offset: 0 };
            std::fs::write(&a[5], serde_json::to_vec_pretty(&layer)?)?;
        }
        Some("render") => {
            // render <ply> <path.json> <shot_dir>: any hand-made CameraPath through the shot renderer
            let shot = PathBuf::from(&a[4]);
            std::fs::create_dir_all(&shot)?;
            let path: splat_shots::paths::CameraPath = serde_json::from_str(&std::fs::read_to_string(&a[3])?)?;
            std::fs::write(shot.join("path.json"), serde_json::to_vec_pretty(&path)?)?;
            let meta = splat_shots::render::render_path(Path::new(&a[2]), &path, &shot.join("render"), &Ffmpeg::from_env())?;
            println!("{}", serde_json::to_string_pretty(&meta)?);
        }
        Some("run") => {
            let shot = ShotDir(PathBuf::from(&a[2]));
            let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&a[3])?)?;
            if let Some(files) = v.get("layer_files").and_then(Value::as_array).cloned() {
                let mut layers = vec![];
                for f in files {
                    let lv: Value = serde_json::from_str(&std::fs::read_to_string(f.as_str().context("layer_files")?)?)?;
                    match lv.get("layers") {
                        Some(ls) => layers.extend(ls.as_array().unwrap().iter().cloned()),
                        None => layers.push(lv),
                    }
                }
                v["layers"] = Value::Array(layers);
                v.as_object_mut().unwrap().remove("layer_files");
            }
            let spec: CompositeSpec = serde_json::from_value(v)?;
            let out = composite_shot(&shot, &spec, Path::new(&a[4]), &Ffmpeg::from_env())?;
            println!("{}", serde_json::to_string_pretty(&out["layers"])?);
        }
        _ => bail!("usage: composite ground|track|matte|run ..."),
    }
    Ok(())
}
