//! Host-side image and mask helpers (row-major, interleaved RGB f32 in [0, 1]).

use std::path::Path;

use image::imageops::FilterType;
use image::{ImageBuffer, Luma, Rgb, Rgb32FImage};

#[derive(Clone)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<bool>,
}

impl Mask {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![false; w * h] }
    }

    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&b| b).count()
    }

    pub fn any(&self) -> bool {
        self.data.iter().any(|&b| b)
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        self.data[y * self.w + x]
    }

    pub fn not(&self) -> Self {
        Self { w: self.w, h: self.h, data: self.data.iter().map(|b| !b).collect() }
    }

    pub fn and(&self, o: &Mask) -> Self {
        Self { w: self.w, h: self.h, data: self.data.iter().zip(&o.data).map(|(a, b)| *a && *b).collect() }
    }

    /// Square (2r+1)^2 max filter (= max_pool2d with stride 1, padding r).
    pub fn dilate(&self, r: usize) -> Self {
        if r == 0 {
            return self.clone();
        }
        let (w, h) = (self.w, self.h);
        let pass = |src: &[bool], len: usize, lines: usize, at: &dyn Fn(usize, usize) -> usize| -> Vec<bool> {
            let mut out = vec![false; src.len()];
            let mut prefix = vec![0u32; len + 1];
            for l in 0..lines {
                for i in 0..len {
                    prefix[i + 1] = prefix[i] + u32::from(src[at(l, i)]);
                }
                for i in 0..len {
                    let (a, b) = (i.saturating_sub(r), (i + r + 1).min(len));
                    out[at(l, i)] = prefix[b] > prefix[a];
                }
            }
            out
        };
        let rows = pass(&self.data, w, h, &|y, x| y * w + x);
        let data = pass(&rows, h, w, &|x, y| y * w + x);
        Self { w, h, data }
    }

    pub fn erode(&self, r: usize) -> Self {
        self.not().dilate(r).not()
    }

    /// (y0, y1, x0, x1), half-open. None when empty.
    pub fn bbox(&self) -> Option<(usize, usize, usize, usize)> {
        let (mut y0, mut y1, mut x0, mut x1) = (usize::MAX, 0, usize::MAX, 0);
        for y in 0..self.h {
            for x in 0..self.w {
                if self.data[y * self.w + x] {
                    y0 = y0.min(y);
                    y1 = y1.max(y + 1);
                    x0 = x0.min(x);
                    x1 = x1.max(x + 1);
                }
            }
        }
        (y0 != usize::MAX).then_some((y0, y1, x0, x1))
    }

    pub fn to_gray(&self, on: u8, off: u8) -> image::GrayImage {
        let data = self.data.iter().map(|&b| if b { on } else { off }).collect();
        image::GrayImage::from_raw(self.w as u32, self.h as u32, data).expect("sized")
    }
}

/// Square crop around the mask's bounding box: `scale` x its larger side, at least
/// `min_side`, clamped to the image. Returns (top, left, side).
pub fn square_crop(m: &Mask, scale: f32, min_side: usize) -> (usize, usize, usize) {
    let (y0, y1, x0, x1) = m.bbox().unwrap_or((0, m.h, 0, m.w));
    let side = ((((y1 - y0).max(x1 - x0)) as f32 * scale) as usize).max(min_side).min(m.h).min(m.w);
    let (cy, cx) = ((y0 + y1) / 2, (x0 + x1) / 2);
    let top = cy.saturating_sub(side / 2).min(m.h - side);
    let left = cx.saturating_sub(side / 2).min(m.w - side);
    (top, left, side)
}

#[derive(Clone)]
pub struct RgbF {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl RgbF {
    pub fn px(&self, x: usize, y: usize) -> [f32; 3] {
        let i = 3 * (y * self.w + x);
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    pub fn crop(&self, top: usize, left: usize, side_h: usize, side_w: usize) -> RgbF {
        let mut data = Vec::with_capacity(side_h * side_w * 3);
        for y in top..top + side_h {
            let s = 3 * (y * self.w + left);
            data.extend_from_slice(&self.data[s..s + 3 * side_w]);
        }
        RgbF { w: side_w, h: side_h, data }
    }

    /// Bicubic (Catmull-Rom) resize, clamped to [0, 1].
    pub fn resize(&self, w: usize, h: usize) -> RgbF {
        if (w, h) == (self.w, self.h) {
            return self.clone();
        }
        let img: Rgb32FImage = ImageBuffer::from_raw(self.w as u32, self.h as u32, self.data.clone()).expect("sized");
        let out = image::imageops::resize(&img, w as u32, h as u32, FilterType::CatmullRom);
        RgbF { w, h, data: out.into_raw().into_iter().map(|v| v.clamp(0.0, 1.0)).collect() }
    }

    pub fn to_rgb8(&self) -> image::RgbImage {
        let data = self.data.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        image::RgbImage::from_raw(self.w as u32, self.h as u32, data).expect("sized")
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        self.to_rgb8().save(path)?;
        Ok(())
    }
}

/// Nearest-neighbour mask resize (pixel centres).
pub fn resize_mask(m: &Mask, w: usize, h: usize) -> Mask {
    let mut out = Mask::new(w, h);
    for y in 0..h {
        let sy = (((y as f32 + 0.5) * m.h as f32 / h as f32) as usize).min(m.h - 1);
        for x in 0..w {
            let sx = (((x as f32 + 0.5) * m.w as f32 / w as f32) as usize).min(m.w - 1);
            out.data[y * w + x] = m.data[sy * m.w + sx];
        }
    }
    out
}

/// Bicubic resize of a single-channel float map.
pub fn resize_scalar(v: &[f32], w: usize, h: usize, nw: usize, nh: usize) -> Vec<f32> {
    let img: ImageBuffer<Luma<f32>, Vec<f32>> = ImageBuffer::from_raw(w as u32, h as u32, v.to_vec()).expect("sized");
    image::imageops::resize(&img, nw as u32, nh as u32, FilterType::CatmullRom).into_raw()
}

/// PSNR over pixels where `m` is set (None when empty).
pub fn psnr_masked(a: &RgbF, b: &RgbF, m: &Mask) -> Option<f64> {
    let (mut se, mut n) = (0.0f64, 0usize);
    for (i, &on) in m.data.iter().enumerate() {
        if on {
            for c in 0..3 {
                let d = (a.data[3 * i + c].clamp(0.0, 1.0) - b.data[3 * i + c].clamp(0.0, 1.0)) as f64;
                se += d * d;
            }
            n += 3;
        }
    }
    (n > 0).then(|| -10.0 * (se / n as f64).max(1e-12).log10())
}

/// Side-by-side strip of equally sized tiles.
pub fn strip(tiles: &[image::RgbImage]) -> image::RgbImage {
    let (tw, th) = tiles.first().map_or((1, 1), |t| t.dimensions());
    let mut out = image::RgbImage::from_pixel(tw * tiles.len() as u32, th, Rgb([255, 255, 255]));
    for (i, t) in tiles.iter().enumerate() {
        image::imageops::overlay(&mut out, t, (i as u32 * tw) as i64, 0);
    }
    out
}
