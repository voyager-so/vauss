//! Undistort a COLMAP dataset to PINHOLE cameras (COLMAP `image_undistorter`, no blank pixels).
//! Surfel (2DGS) rendering takes pinhole cameras only, while video SfM ends with RADIAL (COLMAP)
//! or SIMPLE_RADIAL (the learned-pose rescue): surfel training and evaluation load a pinhole copy
//! of the dataset ([`pinhole_dataset`]) and leave the source frames untouched.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use serde::Serialize;

use crate::colmap::{self, Camera};

/// Lens distortion in COLMAP's OPENCV / FULL_OPENCV form (the radial-only models are special cases).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lens {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    k: [f64; 6],
    p: [f64; 2],
}

impl Lens {
    /// The camera's intrinsics and distortion; `None` for a pinhole camera, an error for models
    /// this undistorter can't invert (fisheye, FOV).
    pub fn of(cam: &Camera) -> Result<Option<Lens>> {
        let p = &cam.params;
        let mut lens = Lens::default();
        match cam.model.as_str() {
            "SIMPLE_PINHOLE" | "PINHOLE" => return Ok(None),
            "SIMPLE_RADIAL" => (lens.fx, lens.fy, lens.cx, lens.cy, lens.k[0]) = (p[0], p[0], p[1], p[2], p[3]),
            "RADIAL" => (lens.fx, lens.fy, lens.cx, lens.cy, lens.k[0], lens.k[1]) = (p[0], p[0], p[1], p[2], p[3], p[4]),
            "OPENCV" => {
                (lens.fx, lens.fy, lens.cx, lens.cy) = (p[0], p[1], p[2], p[3]);
                (lens.k[0], lens.k[1], lens.p) = (p[4], p[5], [p[6], p[7]]);
            }
            "FULL_OPENCV" => {
                (lens.fx, lens.fy, lens.cx, lens.cy) = (p[0], p[1], p[2], p[3]);
                (lens.k[0], lens.k[1], lens.p) = (p[4], p[5], [p[6], p[7]]);
                lens.k[2..].copy_from_slice(&p[8..12]);
            }
            other => bail!("camera {}: cannot undistort a {other} camera to pinhole", cam.id),
        }
        Ok(Some(lens))
    }

    /// Radial factor at squared radius `r2` (FULL_OPENCV's rational form; the others have k3..k6 = 0).
    fn radial(&self, r2: f64) -> f64 {
        let [k1, k2, k3, k4, k5, k6] = self.k;
        (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    }

    /// Normalised undistorted -> distorted coordinates.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let f = self.radial(r2);
        let [p1, p2] = self.p;
        (x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x), y * f + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y))
    }

    /// Normalised distorted -> undistorted coordinates (fixed-point iteration, as COLMAP does).
    pub fn undistort(&self, xd: f64, yd: f64) -> (f64, f64) {
        let (mut x, mut y) = (xd, yd);
        for _ in 0..100 {
            let (dx, dy) = self.distort(x, y);
            let (ex, ey) = (dx - xd, dy - yd);
            x -= ex;
            y -= ey;
            if ex.abs() + ey.abs() < 1e-12 {
                break;
            }
        }
        (x, y)
    }

    /// The radial map r -> r * radial(r^2) still increases at `r` (past that point the polynomial
    /// folds back and two output pixels would read the same source pixel).
    fn monotonic_at(&self, r: f64) -> bool {
        let h = 1e-4 * r.max(1e-3);
        (r + h) * self.radial((r + h) * (r + h)) > r * self.radial(r * r)
    }
}

/// The pinhole camera an undistorted `w` x `h` image is rendered with: the same principal point and
/// the focal lengths scaled by the smallest factor in [0.5, 4] at which every border pixel of the
/// output reads inside the source image (no blank pixels; a pincushion lens zooms in, a barrel
/// lens zooms out up to where its polynomial would fold).
pub fn pinhole_for(lens: &Lens, w: u32, h: u32) -> (f64, f64) {
    let valid = |s: f64| {
        let (fx, fy) = (lens.fx * s, lens.fy * s);
        let mut border = (0..w).flat_map(|u| [(u, 0), (u, h - 1)]).chain((0..h).flat_map(|v| [(0, v), (w - 1, v)]));
        border.all(|(u, v)| {
            let (x, y) = ((u as f64 + 0.5 - lens.cx) / fx, (v as f64 + 0.5 - lens.cy) / fy);
            let (xd, yd) = lens.distort(x, y);
            let (su, sv) = (xd * lens.fx + lens.cx - 0.5, yd * lens.fy + lens.cy - 0.5);
            lens.monotonic_at((x * x + y * y).sqrt()) && (-0.5..=w as f64 - 0.5).contains(&su) && (-0.5..=h as f64 - 0.5).contains(&sv)
        })
    };
    let (mut lo, mut hi) = (0.5, 4.0);
    if valid(lo) {
        return (lens.fx * lo, lens.fy * lo);
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if valid(mid) { hi = mid } else { lo = mid }
    }
    (lens.fx * hi, lens.fy * hi)
}

/// Resample `src` (interleaved, `channels` per pixel) to the pinhole camera (fx, fy, lens.cx,
/// lens.cy), bilinear.
pub fn remap(src: &[u8], w: u32, h: u32, channels: usize, lens: &Lens, fx: f64, fy: f64) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let at = |x: i64, y: i64, c: usize| src[(y.clamp(0, h as i64 - 1) as usize * w as usize + x.clamp(0, w as i64 - 1) as usize) * channels + c] as f64;
    out.par_chunks_mut(channels * w as usize).enumerate().for_each(|(v, row)| {
        for u in 0..w as usize {
            let (x, y) = ((u as f64 + 0.5 - lens.cx) / fx, (v as f64 + 0.5 - lens.cy) / fy);
            let (xd, yd) = lens.distort(x, y);
            let (su, sv) = (xd * lens.fx + lens.cx - 0.5, yd * lens.fy + lens.cy - 0.5);
            let (x0, y0) = (su.floor(), sv.floor());
            let (ax, ay) = (su - x0, sv - y0);
            let (x0, y0) = (x0 as i64, y0 as i64);
            for c in 0..channels {
                let top = at(x0, y0, c) * (1.0 - ax) + at(x0 + 1, y0, c) * ax;
                let bottom = at(x0, y0 + 1, c) * (1.0 - ax) + at(x0 + 1, y0 + 1, c) * ax;
                row[channels * u + c] = (top * (1.0 - ay) + bottom * ay).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

/// Undistort the image file `src` into `dst` (same format; JPEG at quality 95).
fn remap_file(src: &Path, dst: &Path, lens: &Lens, fx: f64, fy: f64, size: (u64, u64)) -> Result<()> {
    let img = image::open(src).with_context(|| format!("reading {}", src.display()))?;
    let (w, h) = (img.width(), img.height());
    ensure!((u64::from(w), u64::from(h)) == size, "{}: image is {w}x{h}, camera is {}x{}", src.display(), size.0, size.1);
    let out = match img {
        image::DynamicImage::ImageLuma8(m) => image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, remap(m.as_raw(), w, h, 1, lens, fx, fy)).context("size")?),
        image::DynamicImage::ImageLumaA8(m) => image::DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_raw(w, h, remap(m.as_raw(), w, h, 2, lens, fx, fy)).context("size")?),
        image::DynamicImage::ImageRgba8(m) => image::DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w, h, remap(m.as_raw(), w, h, 4, lens, fx, fy)).context("size")?),
        other => {
            let m = other.to_rgb8();
            image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, remap(m.as_raw(), w, h, 3, lens, fx, fy)).context("size")?)
        }
    };
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let jpeg = dst.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"));
    if jpeg {
        let file = std::fs::File::create(dst).with_context(|| format!("writing {}", dst.display()))?;
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 95).encode_image(&out.to_rgb8())?;
    } else {
        out.save(dst).with_context(|| format!("writing {}", dst.display()))?;
    }
    Ok(())
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Undistorted {
    /// Camera id -> (model before, focal scale applied).
    pub cameras: BTreeMap<u32, (String, f64)>,
    pub images: usize,
    pub masks: usize,
    pub seconds: f64,
}

/// The binary model dir of a dataset (`sparse/0`, or `sparse` holding the files directly).
fn sparse_dir(dataset: &Path) -> Option<std::path::PathBuf> {
    [dataset.join("sparse/0"), dataset.join("sparse")].into_iter().find(|d| d.join("cameras.bin").is_file())
}

/// A pinhole copy of the SfM dataset `src` (images/ + sparse/0 [+ masks/]) at `dst`: PINHOLE
/// cameras, undistorted registered images and their masks, keypoints moved with their images.
/// `src` is never modified: its images are the frozen split's source frames. `None` (and nothing
/// written) when every camera already is pinhole; an error for lenses this can't invert (fisheye).
pub fn pinhole_dataset(src: &Path, dst: &Path) -> Result<Option<Undistorted>> {
    let start = std::time::Instant::now();
    let Some(sparse) = sparse_dir(src) else { bail!("{}: no binary COLMAP model to undistort", src.display()) };
    let model = colmap::read_model(&sparse)?;
    let mut lenses = BTreeMap::new();
    for (id, cam) in &model.cameras {
        if let Some(lens) = Lens::of(cam)? {
            lenses.insert(*id, lens);
        }
    }
    if lenses.is_empty() {
        return Ok(None);
    }
    let _ = std::fs::remove_dir_all(dst);
    let dst_sparse = dst.join("sparse/0");
    std::fs::create_dir_all(&dst_sparse)?;
    for entry in std::fs::read_dir(&sparse)? {
        let p = entry?.path();
        if p.is_file() {
            std::fs::copy(&p, dst_sparse.join(p.file_name().context("file name")?))?;
        }
    }
    let mut report = Undistorted::default();
    let mut cameras = model.cameras.clone();
    let mut pinholes = BTreeMap::new();
    for (id, lens) in &lenses {
        let cam = &model.cameras[id];
        let (fx, fy) = pinhole_for(lens, cam.width as u32, cam.height as u32);
        report.cameras.insert(*id, (cam.model.clone(), fx / lens.fx));
        cameras.insert(*id, Camera { id: *id, model: "PINHOLE".into(), width: cam.width, height: cam.height, params: vec![fx, fy, lens.cx, lens.cy] });
        pinholes.insert(*id, (fx, fy));
    }
    // Every registered image is written (a pinhole camera's image is copied as is), masks beside it.
    let masks_src = src.join("masks");
    let jobs: Vec<(String, u32)> = model.images.values().map(|im| (im.name.clone(), im.camera_id)).collect();
    let mask_counts: Vec<usize> = jobs.par_iter().map(|(name, cam_id)| -> Result<usize> {
        let size = (model.cameras[cam_id].width, model.cameras[cam_id].height);
        let (from, to) = (src.join("images").join(name), dst.join("images").join(name));
        let mut masks = 0;
        match lenses.get(cam_id) {
            Some(lens) => {
                let (fx, fy) = pinholes[cam_id];
                remap_file(&from, &to, lens, fx, fy, size)?;
                let stem = Path::new(name).with_extension("");
                for ext in ["png", "jpg", "jpeg"] {
                    for rel in [stem.with_extension(ext), std::path::PathBuf::from(format!("{name}.{ext}"))] {
                        let m = masks_src.join(&rel);
                        if m.is_file() {
                            remap_file(&m, &dst.join("masks").join(&rel), lens, fx, fy, size)?;
                            masks += 1;
                        }
                    }
                }
            }
            None => {
                std::fs::create_dir_all(to.parent().context("image dir")?)?;
                std::fs::copy(&from, &to).with_context(|| format!("copying {}", from.display()))?;
            }
        }
        Ok(masks)
    }).collect::<Result<_>>()?;
    let images_path = dst_sparse.join("images.bin");
    let mut records = colmap::read_images_bin(&images_path)?;
    for im in &mut records {
        let (Some(lens), Some(&(fx, fy))) = (lenses.get(&im.camera_id), pinholes.get(&im.camera_id)) else { continue };
        for (xy, _) in &mut im.points2d {
            let (x, y) = lens.undistort((xy[0] - lens.cx) / lens.fx, (xy[1] - lens.cy) / lens.fy);
            *xy = [x * fx + lens.cx, y * fy + lens.cy];
        }
    }
    colmap::write_images_bin(&images_path, &records)?;
    colmap::write_cameras_bin(&dst_sparse.join("cameras.bin"), &cameras)?;
    for txt in ["cameras.txt", "images.txt", "points3D.txt"] {
        let _ = std::fs::remove_file(dst_sparse.join(txt));
    }
    report.images = jobs.len();
    report.masks = mask_counts.iter().sum();
    report.seconds = start.elapsed().as_secs_f64();
    Ok(Some(report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(model: &str, params: Vec<f64>) -> Camera {
        Camera { id: 1, model: model.into(), width: 160, height: 90, params }
    }

    #[test]
    fn pinhole_cameras_are_left_alone() {
        assert!(Lens::of(&cam("PINHOLE", vec![100.0, 100.0, 80.0, 45.0])).unwrap().is_none());
        assert!(Lens::of(&cam("SIMPLE_PINHOLE", vec![100.0, 80.0, 45.0])).unwrap().is_none());
        assert!(Lens::of(&cam("OPENCV_FISHEYE", vec![100.0, 100.0, 80.0, 45.0, 0.1, 0.0, 0.0, 0.0])).is_err());
    }

    #[test]
    fn undistort_inverts_distort() {
        let lens = Lens::of(&cam("OPENCV", vec![100.0, 102.0, 80.0, 45.0, -0.12, 0.03, 0.001, -0.002])).unwrap().unwrap();
        for (x, y) in [(0.0, 0.0), (0.3, -0.2), (-0.7, 0.4), (0.8, 0.45)] {
            let (xd, yd) = lens.distort(x, y);
            let (xu, yu) = lens.undistort(xd, yd);
            assert!((xu - x).abs() < 1e-9 && (yu - y).abs() < 1e-9, "({x}, {y}) -> ({xu}, {yu})");
        }
    }

    #[test]
    fn no_blank_pixels_for_pincushion_or_barrel() {
        for k1 in [0.02, 0.15, -0.15] {
            let lens = Lens::of(&cam("RADIAL", vec![100.0, 80.0, 45.0, k1, 0.0])).unwrap().unwrap();
            let (fx, _) = pinhole_for(&lens, 160, 90);
            // Pincushion (k1 > 0) zooms in to stay inside the frame; barrel may zoom out.
            assert_eq!(fx > 100.0, k1 > 0.0, "k1 {k1}: fx {fx}");
            // The output's corner reads inside the source.
            let (x, y) = ((0.5 - 80.0) / fx, (0.5 - 45.0) / fx);
            let (xd, yd) = lens.distort(x, y);
            assert!(xd * 100.0 + 80.0 >= -1e-6 && yd * 100.0 + 45.0 >= -1e-6, "k1 {k1}");
        }
    }
}
