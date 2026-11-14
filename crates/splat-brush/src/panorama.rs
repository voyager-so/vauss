//! split_panoramas: cut each 360 equirect frame into the 14 pinhole views of
//! [`splat_core::panorama::Layout::v14`], with SfM feature masks (each direction gives features to
//! one view), nadir training masks, `panorama.json` and COLMAP's `rig_config.json`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::Context;
use image::{GrayImage, RgbImage};
use serde_json::json;
use splat_contracts::*;
use splat_core::RunCtx;
use splat_core::panorama::{self, Manifest};

use crate::util::*;

/// Equirect frames are 2:1 within this many pixels.
pub const ASPECT_TOLERANCE_PX: u32 = 1;

/// Whether `width` x `height` is a 2:1 equirect frame.
pub fn is_two_to_one(width: u32, height: u32) -> bool {
    height > 0 && width.abs_diff(2 * height) <= ASPECT_TOLERANCE_PX
}

/// sRGB 8-bit -> linear light.
fn srgb_to_linear_table() -> [f32; 256] {
    std::array::from_fn(|i| {
        let c = i as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    })
}

/// Linear light -> sRGB 8-bit through a 4096-entry table (error below half a step).
struct Encode([u8; 4097]);

impl Encode {
    fn new() -> Self {
        Self(std::array::from_fn(|i| {
            let l = i as f32 / 4096.0;
            let c = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
            (c * 255.0).round().clamp(0.0, 255.0) as u8
        }))
    }

    fn get(&self, l: f32) -> u8 {
        // Linear interpolation between table entries keeps dark tones exact enough.
        let x = l.clamp(0.0, 1.0) * 4096.0;
        let i = x as usize;
        if i >= 4096 {
            return self.0[4096];
        }
        let (a, b) = (self.0[i] as f32, self.0[i + 1] as f32);
        (a + (b - a) * (x - i as f32)).round() as u8
    }
}

/// A frame decoded to linear light, 3 channels row-major.
struct Linear {
    w: u32,
    h: u32,
    px: Vec<f32>,
}

impl Linear {
    fn new(img: &RgbImage, table: &[f32; 256]) -> Self {
        Self { w: img.width(), h: img.height(), px: img.as_raw().iter().map(|&v| table[v as usize]).collect() }
    }

    /// Bilinear sample at pixel-index coordinates: u wraps, v clamps.
    fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let (w, h) = (self.w as i64, self.h as i64);
        let v = v.clamp(0.0, (h - 1) as f32);
        let (u0, v0) = (u.floor(), v.floor());
        let (fu, fv) = (u - u0, v - v0);
        let x0 = (u0 as i64).rem_euclid(w);
        let x1 = (x0 + 1).rem_euclid(w);
        let y0 = v0 as i64;
        let y1 = (y0 + 1).min(h - 1);
        let at = |x: i64, y: i64| ((y * w + x) * 3) as usize;
        let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        let p = &self.px;
        std::array::from_fn(|k| {
            let top = p[a + k] + (p[b + k] - p[a + k]) * fu;
            let bottom = p[c + k] + (p[d + k] - p[c + k]) * fu;
            top + (bottom - top) * fv
        })
    }
}

/// Render one view of `pano` through its lookup table.
fn render_view(pano: &Linear, table: &[[f32; 2]], s: u32, encode: &Encode) -> RgbImage {
    let mut out = Vec::with_capacity((s * s * 3) as usize);
    for &[u, v] in table {
        for c in pano.sample(u, v) {
            out.push(encode.get(c));
        }
    }
    RgbImage::from_raw(s, s, out).expect("view size")
}

/// The 14 views of one equirect image, in layout order.
pub fn split_image(img: &RgbImage, s: u32) -> Vec<RgbImage> {
    let pano = Linear::new(img, &srgb_to_linear_table());
    let encode = Encode::new();
    (0..panorama::VIEWS).map(|view| render_view(&pano, &panorama::lookup_table(view, s, img.width(), img.height()), s, &encode)).collect()
}

fn gray(s: u32, data: Vec<u8>) -> GrayImage {
    GrayImage::from_raw(s, s, data).expect("mask size")
}

pub fn split_panoramas(run: &dyn RunCtx, req: SplitPanoramas) -> anyhow::Result<ToolResult> {
    let start = Instant::now();
    let source = run.artifact(&req.frames_id).with_context(|| format!("frames artifact {}", req.frames_id))?;
    if source.kind != ArtifactKind::Frames {
        return Err(invalid(format!("{} is not a frames artifact", req.frames_id)));
    }
    if source.meta.get("panorama_source") != Some(&json!(true)) {
        return Err(invalid("split_panoramas needs frames extracted with panorama=true"));
    }
    if !(256..=2048).contains(&req.view_size) || !(0.0..=60.0).contains(&req.nadir_mask_deg) {
        return Err(invalid("view_size must be 256-2048 and nadir_mask_deg 0-60"));
    }
    let src_dir = run.path(&req.frames_id)?.join("images");
    let mut inputs = list_images(&src_dir)?;
    inputs.sort();
    if inputs.is_empty() {
        return Err(invalid("frames artifact has no images"));
    }
    let names: Vec<String> = inputs.iter().map(|p| p.strip_prefix(&src_dir).unwrap_or(p).to_string_lossy().replace('\\', "/")).collect();
    anyhow::ensure!(names.iter().all(|n| !n.contains('/')), "360 frames must sit directly in images/");
    let (w, h) = image::image_dimensions(&inputs[0])?;
    if !is_two_to_one(w, h) {
        return Err(invalid(format!("360 frames must be 2:1 equirectangular; {} is {w}x{h}", names[0])));
    }
    // Views wider than a quarter of the pano only upsample it.
    let s = req.view_size.min((w / 4).max(256));
    let (artifact, out) = new_artifact(run, ArtifactKind::Frames, json!({"input_kind": "images", "panorama": true,
        "views": panorama::VIEWS, "source_frames_id": req.frames_id, "chronology": "source_order"}))?;
    let images = out.join("images");
    let feature_masks = out.join("feature_masks");
    let train_masks = out.join("train_masks");
    for d in [&images, &feature_masks, &train_masks] {
        std::fs::create_dir_all(d)?;
    }
    // Views are written as .jpg; the manifest names them by their equirect frame.
    let frames: Vec<String> = names.iter().map(|n| Path::new(n).with_extension("jpg").to_string_lossy().into_owned()).collect();
    let manifest = Manifest::new(s, req.nadir_mask_deg, frames);
    // Masks are the same for every frame: compute each view's once, write one file per image
    // (COLMAP looks up `<name>.png`; Brush pairs `<stem>.png` under a masks/ folder).
    let owner: Vec<GrayImage> = (0..panorama::VIEWS).map(|v| gray(s, panorama::owner_mask(v, s, req.nadir_mask_deg))).collect();
    let nadir: Vec<GrayImage> = (0..panorama::VIEWS).map(|v| gray(s, panorama::nadir_mask(v, s, req.nadir_mask_deg))).collect();
    let tables: Vec<Vec<[f32; 2]>> = (0..panorama::VIEWS).map(|v| panorama::lookup_table(v, s, w, h)).collect();
    let (to_linear, encode) = (srgb_to_linear_table(), Encode::new());
    let next = AtomicUsize::new(0);
    // One failed frame stops the other workers instead of letting them convert the rest.
    let failed = std::sync::atomic::AtomicBool::new(false);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
    std::thread::scope(|scope| -> anyhow::Result<()> {
        let handles: Vec<_> = (0..workers).map(|_| scope.spawn(|| -> anyhow::Result<()> {
            loop {
                if failed.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(path) = inputs.get(i) else { return Ok(()) };
                let done = (|| -> anyhow::Result<()> {
                let img = image::open(path).with_context(|| format!("reading {}", path.display()))?.to_rgb8();
                if img.dimensions() != (w, h) {
                    return Err(invalid(format!("360 frames must all be {w}x{h}; {} is {}x{}", names[i], img.width(), img.height())));
                }
                let pano = Linear::new(&img, &to_linear);
                for view in 0..panorama::VIEWS {
                    let name = manifest.view_name(view, &manifest.frames[i]);
                    let rendered = render_view(&pano, &tables[view], s, &encode);
                    splat_eval::images::write_jpeg_compact(&images.join(&name), &rendered, 90, None)?;
                    owner[view].save(feature_masks.join(format!("{name}.png")))?;
                    let stem = Path::new(&name).file_stem().unwrap().to_string_lossy().into_owned();
                    nadir[view].save(train_masks.join(format!("{stem}.png")))?;
                }
                Ok(())
                })();
                if done.is_err() {
                    failed.store(true, Ordering::Relaxed);
                    return done;
                }
            }
        })).collect();
        for handle in handles {
            handle.join().map_err(|_| anyhow::anyhow!("panorama split worker panicked"))??;
        }
        Ok(())
    })?;
    manifest.write(&out)?;
    write_json(&out.join(panorama::RIG_CONFIG), &panorama::rig_config_json(s))?;
    let views = names.len() * panorama::VIEWS;
    let mut metrics = serde_json::Map::new();
    metrics.insert("panoramas".into(), json!(names.len()));
    metrics.insert("views".into(), json!(views));
    metrics.insert("view_size".into(), json!(s));
    metrics.insert("nadir_mask_deg".into(), json!(req.nadir_mask_deg));
    metrics.insert("source_size".into(), json!([w, h]));
    metrics.insert("seconds".into(), json!(start.elapsed().as_secs_f64()));
    // The fields other tools read from an extraction's selection.json (report counts).
    metrics.insert("input_count".into(), json!(views));
    metrics.insert("frame_count".into(), json!(views));
    metrics.insert("dropped_count".into(), json!(0));
    metrics.insert("input_count_basis".into(), json!("split_360_views"));
    metrics.insert("max_edge".into(), json!(s));
    write_json(&out.join("selection.json"), &json!({"request": req, "metrics": metrics, "frames": manifest.frames}))?;
    let previews: Vec<PathBuf> = manifest.frames.first().map(|f| (0..panorama::VIEWS).map(|v| images.join(manifest.view_name(v, f))).collect()).unwrap_or_default();
    Ok(ToolResult {
        artifacts: vec![artifact],
        metrics: metrics.into_iter().collect(),
        images: previews.iter().step_by(2).map(|p| rel(run, p)).collect(),
        wall_s: Some(start.elapsed().as_secs_f64()),
        ..result("split_panoramas", Status::Succeeded)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_core::panorama::{apply, cam_from_pano, equirect_dir, pixel_ray, transpose};

    /// An analytic colour of a direction, smooth and distinct per axis (sRGB values in 0-255).
    fn colour(d: [f64; 3]) -> [f64; 3] {
        [d[0], d[1], d[2]].map(|c| 127.5 + 100.0 * c)
    }

    fn analytic(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let c = colour(equirect_dir(x as f64 + 0.5, y as f64 + 0.5, w, h));
            image::Rgb(c.map(|v| v.round() as u8))
        })
    }

    #[test]
    fn views_sample_the_analytic_function_at_their_rays() {
        let (w, h, s) = (2048, 1024, 128);
        let views = split_image(&analytic(w, h), s);
        assert_eq!(views.len(), panorama::VIEWS);
        let mut worst = 0.0f64;
        for (view, img) in views.iter().enumerate() {
            let pano_from_cam = transpose(&cam_from_pano(view));
            for y in 0..s {
                for x in 0..s {
                    let mut d = apply(&pano_from_cam, pixel_ray(s, x, y));
                    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    d = d.map(|v| v / n);
                    let expected = colour(d);
                    let got = img.get_pixel(x, y).0;
                    for k in 0..3 {
                        worst = worst.max((got[k] as f64 - expected[k]).abs());
                    }
                }
            }
        }
        // 2/255: the 8-bit source quantisation (0.5) plus interpolation of a linear-light blend.
        assert!(worst <= 2.0, "worst error {worst}/255");
    }

    #[test]
    fn horizontal_wrap_is_continuous_across_the_seam() {
        // A pano whose left and right halves differ except for a band straddling the +-180 deg
        // seam: view 2 (yaw 180) must show the band continuously across its centre column.
        let (w, h, s) = (1024, 512, 64);
        let img = RgbImage::from_fn(w, h, |x, _| if x < 16 || x >= w - 16 { image::Rgb([250, 250, 250]) } else { image::Rgb([10, 10, 10]) });
        let view = &split_image(&img, s)[2];
        let row = s / 2;
        let (l, r) = (view.get_pixel(s / 2 - 1, row).0[0], view.get_pixel(s / 2, row).0[0]);
        assert!(l > 240 && r > 240, "seam columns {l} {r}");
        // Smooth across the seam: no column jumps more than the band's own edges.
        let jumps = (1..s).filter(|&x| view.get_pixel(x, row).0[0].abs_diff(view.get_pixel(x - 1, row).0[0]) > 100).count();
        assert!(jumps <= 2, "{jumps} jumps");
    }

    #[test]
    fn encode_round_trips_every_byte() {
        let (to_linear, encode) = (srgb_to_linear_table(), Encode::new());
        for v in 0..=255u8 {
            assert_eq!(encode.get(to_linear[v as usize]), v);
        }
    }

    #[test]
    fn aspect_check() {
        assert!(is_two_to_one(2048, 1024) && is_two_to_one(2049, 1024) && is_two_to_one(2047, 1024));
        assert!(!is_two_to_one(1920, 1080) && !is_two_to_one(2050, 1024) && !is_two_to_one(0, 0));
    }
}
