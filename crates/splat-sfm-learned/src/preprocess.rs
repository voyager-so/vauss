//! DA3 input preprocessing ("upper_bound_resize" at 504): resize so the long side is 504, round
//! each side to the nearest multiple of 14 with a second small resize, ImageNet-normalise.
//!
//! Port of depth_anything_3/utils/io/input_processor.py at 3d835ec1 (Apache-2.0, see NOTICE).
//! JPEGs are decoded exactly as libjpeg-turbo (PIL) does, see [`crate::jpeg`]. The resizes
//! reproduce `cv2.resize` on 8-bit RGB: INTER_AREA when shrinking, INTER_CUBIC when any side
//! grows.

use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use image::RgbImage;
use rayon::prelude::*;

use crate::model::PATCH;

const PROCESS_RES: u32 = 504;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Model input `[views, 3, height, width]` plus the frames' original size.
pub struct Preprocessed {
    pub data: Vec<f32>,
    pub views: usize,
    pub height: usize,
    pub width: usize,
    pub orig_width: u32,
    pub orig_height: u32,
}

pub fn preprocess(frames: &[PathBuf]) -> anyhow::Result<Preprocessed> {
    ensure!(!frames.is_empty(), "no frames");
    let images: Vec<(RgbImage, (u32, u32))> = frames
        .par_iter()
        .map(|path| load_resized(path))
        .collect::<anyhow::Result<_>>()?;
    let (first, orig) = &images[0];
    let (width, height) = first.dimensions();
    for ((img, size), path) in images.iter().zip(frames) {
        ensure!(
            size == orig && img.dimensions() == (width, height),
            "{}: {}x{} differs from the first frame's {}x{}; frames must share one resolution",
            path.display(),
            size.0,
            size.1,
            orig.0,
            orig.1
        );
    }
    let plane = (width * height) as usize;
    let mut data = vec![0.0f32; images.len() * 3 * plane];
    data.par_chunks_mut(3 * plane)
        .zip(&images)
        .for_each(|(out, (img, _))| {
            for (i, px) in img.pixels().enumerate() {
                for c in 0..3 {
                    out[c * plane + i] = (f32::from(px[c]) / 255.0 - MEAN[c]) / STD[c];
                }
            }
        });
    Ok(Preprocessed {
        data,
        views: images.len(),
        height: height as usize,
        width: width as usize,
        orig_width: orig.0,
        orig_height: orig.1,
    })
}

/// Decode and resize one frame; returns it with its original (width, height).
fn load_resized(path: &Path) -> anyhow::Result<(RgbImage, (u32, u32))> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let img = match crate::jpeg::decode(&bytes) {
        Some(img) => img,
        None => image::load_from_memory(&bytes)
            .with_context(|| format!("decoding {}", path.display()))?
            .into_rgb8(),
    };
    let (w, h) = img.dimensions();

    // Long side to 504 (Python `round`, i.e. ties to even).
    let longest = w.max(h);
    let mut img = img;
    if longest != PROCESS_RES {
        let scale = f64::from(PROCESS_RES) / f64::from(longest);
        let side = |v: u32| ((f64::from(v) * scale).round_ties_even() as u32).max(1);
        img = resize(&img, side(w), side(h), scale > 1.0);
    }

    // Each side to the nearest multiple of 14 (ties round up).
    let (rw, rh) = img.dimensions();
    let nearest = |v: u32| {
        let p = PATCH as u32;
        let down = v / p * p;
        if down + p - v <= v - down { down + p } else { down }.max(1)
    };
    let (nw, nh) = (nearest(rw), nearest(rh));
    if (nw, nh) != (rw, rh) {
        img = resize(&img, nw, nh, nw > rw || nh > rh);
    }
    Ok((img, (w, h)))
}

fn resize(img: &RgbImage, w: u32, h: u32, cubic: bool) -> RgbImage {
    if cubic { resize_cubic(img, w, h) } else { resize_area(img, w, h) }
}

/// `cv2.INTER_AREA` for downscaling: each output pixel is the coverage-weighted mean of the
/// source pixels under it (`computeResizeAreaTab`).
fn resize_area(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    let (sw, sh) = img.dimensions();
    let xs = area_taps(sw as usize, w as usize);
    let ys = area_taps(sh as usize, h as usize);
    let src = img.as_raw();
    let row_len = sw as usize * 3;
    let mut out = vec![0u8; (w * h * 3) as usize];
    out.par_chunks_mut(w as usize * 3).zip(&ys).for_each(|(dst, ytaps)| {
        let mut acc = vec![0.0f32; w as usize * 3];
        for &(sy, by) in ytaps {
            let row = &src[sy * row_len..][..row_len];
            for (dx, xtaps) in xs.iter().enumerate() {
                for c in 0..3 {
                    let v: f32 = xtaps.iter().map(|&(sx, ax)| ax * f32::from(row[sx * 3 + c])).sum();
                    acc[dx * 3 + c] += by * v;
                }
            }
        }
        for (d, v) in dst.iter_mut().zip(&acc) {
            *d = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    });
    RgbImage::from_raw(w, h, out).expect("buffer matches dimensions")
}

fn area_taps(ssize: usize, dsize: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = 1.0 / (dsize as f64 / ssize as f64);
    (0..dsize)
        .map(|d| {
            let fs1 = d as f64 * scale;
            let fs2 = fs1 + scale;
            let cell = scale.min(ssize as f64 - fs1);
            let s2 = (fs2.floor() as usize).min(ssize - 1);
            let s1 = (fs1.ceil() as usize).min(s2);
            let mut taps = Vec::new();
            if s1 as f64 - fs1 > 1e-3 {
                taps.push((s1 - 1, ((s1 as f64 - fs1) / cell) as f32));
            }
            for s in s1..s2 {
                taps.push((s, (1.0 / cell) as f32));
            }
            if fs2 - s2 as f64 > 1e-3 {
                taps.push((s2, ((fs2 - s2 as f64).min(1.0).min(cell) / cell) as f32));
            }
            taps
        })
        .collect()
}

/// `cv2.INTER_CUBIC` upscaling on 8-bit images as the opencv-python wheels run it, through Intel
/// IPP (`ippiResizeCubic`, B=0 C=0.75): f32 Keys weights, a horizontal then a vertical f32 pass,
/// rounded half to even, replicated borders. (OpenCV's own fixed-point path differs by one level
/// on ~3% of pixels.)
fn resize_cubic(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    let (sw, sh) = img.dimensions();
    let xs = cubic_taps(sw as usize, w as usize);
    let ys = cubic_taps(sh as usize, h as usize);
    let src = img.as_raw();
    let row_len = sw as usize * 3;
    let hrow = |sy: usize| -> Vec<f32> {
        let row = &src[sy * row_len..][..row_len];
        xs.iter()
            .flat_map(|taps| {
                (0..3).map(move |c| taps.iter().fold(0.0f32, |acc, &(sx, a)| acc + a * f32::from(row[sx * 3 + c])))
            })
            .collect()
    };
    let mut out = vec![0u8; (w * h * 3) as usize];
    out.par_chunks_mut(w as usize * 3).zip(&ys).for_each(|(dst, ytaps)| {
        let rows: Vec<(Vec<f32>, f32)> = ytaps.iter().map(|&(sy, b)| (hrow(sy), b)).collect();
        for (i, d) in dst.iter_mut().enumerate() {
            let v = rows.iter().fold(0.0f32, |acc, (r, b)| acc + b * r[i]);
            *d = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    });
    RgbImage::from_raw(w, h, out).expect("buffer matches dimensions")
}

/// Keys cubic (a = -0.75) taps per output index, `src = (dst + 0.5) * ssize / dsize - 0.5`;
/// taps clamped onto the same edge pixel are merged.
fn cubic_taps(ssize: usize, dsize: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = ssize as f64 / dsize as f64;
    (0..dsize)
        .map(|d| {
            let f = (d as f64 + 0.5) * scale - 0.5;
            let i = f.floor();
            let mut taps: Vec<(usize, f32)> = Vec::with_capacity(4);
            let x = (f - i) as f32;
            const A: f32 = -0.75;
            let near = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
            let far = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
            for (k, c) in [far(x + 1.0), near(x), near(1.0 - x), far(2.0 - x)].into_iter().enumerate() {
                let s = (i as isize - 1 + k as isize).clamp(0, ssize as isize - 1) as usize;
                match taps.last_mut() {
                    Some((last, a)) if *last == s => *a += c,
                    _ => taps.push((s, c)),
                }
            }
            taps
        })
        .collect()
}
