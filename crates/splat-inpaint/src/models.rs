//! 2D models through ONNX Runtime (CPU; `ort` load-dynamic like splat-eval's LPIPS):
//! LaMa big-lama (inpainting, fixed 512x512) and Depth-Anything-V2-Small (relative
//! inverse depth). Both Apache-2.0, pinned in splat-fetch's catalog. The diffusion
//! inpainter (SD-1.5, CUDA) is in `crate::diffusion`.
//!
//! Model files: `<models_dir>/<file>`, else `$SPLAT_MODELS_DIR/<file>`, else the pinned
//! download in the splat cache (`splat_fetch`, SHA-256 checked).

use std::path::{Path, PathBuf};

use anyhow::ensure;
use ort::session::Session;
use ort::value::TensorRef;

use crate::img::{Mask, RgbF, resize_mask, resize_scalar};

pub const LAMA: (&str, &str) = ("inpaint-lama", "lama_fp32.onnx");
pub const DEPTH: (&str, &str) = ("inpaint-depth-anything-v2-small", "depth_anything_v2_small.onnx");

pub fn locate(models_dir: Option<&Path>, (asset_id, file): (&str, &str)) -> anyhow::Result<PathBuf> {
    let asset = splat_fetch::asset_for(asset_id, "any")?;
    let env = std::env::var_os("SPLAT_MODELS_DIR").filter(|v| !v.is_empty()).map(PathBuf::from);
    for dir in models_dir.map(Path::to_path_buf).into_iter().chain(env) {
        let p = dir.join(file);
        if p.is_file() {
            let bytes = std::fs::metadata(&p)?.len();
            ensure!(bytes == asset.bytes, "{} is {bytes} bytes, expected {} (re-fetch it)", p.display(), asset.bytes);
            return Ok(p);
        }
    }
    let cache = splat_fetch::default_cache_dir()?;
    eprintln!("[inpaint] fetching {asset_id} ({} MB) into {}", asset.bytes >> 20, cache.display());
    splat_fetch::Fetcher::new(cache).ensure_asset(&asset, splat_fetch::LicenseConsent::Default)
}

fn session(path: &Path) -> anyhow::Result<Session> {
    session_on(path, false)
}

/// With `cuda`, the CUDA execution provider (onnxruntime-gpu; fails if unavailable).
fn session_on(path: &Path, cuda: bool) -> anyhow::Result<Session> {
    splat_eval::lpips::init_runtime(None)?;
    let load = || -> ort::Result<Session> {
        let b = Session::builder()?.with_intra_threads(splat_eval::lpips::ort_threads())?;
        let mut b = if cuda {
            b.with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?
        } else {
            b
        };
        b.commit_from_file(path)
    };
    load().map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))
}

fn input_names(s: &Session) -> Vec<String> {
    s.inputs().iter().map(|i| i.name().to_string()).collect()
}

/// A 2D inpainter at a fixed 512x512: [0,1] RGB in, [0,1] RGB out, `hole` = pixels to fill.
pub trait Inpaint2d {
    fn inpaint(&mut self, img: &RgbF, hole: &Mask) -> anyhow::Result<RgbF>;
}

/// Working resolution of every [`Inpaint2d`].
pub const INPAINT_SIZE: usize = 512;

pub struct Lama {
    session: Session,
    image: String,
    mask: String,
}

impl Lama {
    pub const SIZE: usize = INPAINT_SIZE;

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let session = session(path)?;
        let names = input_names(&session);
        ensure!(names.len() == 2, "LaMa: expected 2 inputs, got {names:?}");
        let pick = |want: &str, fallback: usize| names.iter().find(|n| n.contains(want)).cloned().unwrap_or_else(|| names[fallback].clone());
        let (image, mask) = (pick("image", 0), pick("mask", 1));
        eprintln!("[inpaint] LaMa inputs image={image} mask={mask}");
        Ok(Self { session, image, mask })
    }

    /// Inpaint a 512x512 [0,1] RGB image where `hole` is set. Returns [0,1] RGB.
    pub fn run(&mut self, img: &RgbF, hole: &Mask) -> anyhow::Result<RgbF> {
        let n = Self::SIZE;
        ensure!(img.w == n && img.h == n && hole.w == n && hole.h == n, "LaMa runs at {n}x{n}");
        let mut planar = vec![0f32; 3 * n * n];
        for i in 0..n * n {
            for c in 0..3 {
                planar[c * n * n + i] = img.data[3 * i + c];
            }
        }
        let m: Vec<f32> = hole.data.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let s = n as i64;
        let out = self.session.run(ort::inputs![
            self.image.as_str() => TensorRef::from_array_view(([1i64, 3, s, s], planar.as_slice()))?,
            self.mask.as_str() => TensorRef::from_array_view(([1i64, 1, s, s], m.as_slice()))?,
        ])?;
        let (shape, y) = out[0].try_extract_tensor::<f32>()?;
        ensure!(y.len() == 3 * n * n, "LaMa output shape {shape:?}");
        // The Carve export returns 0..255; normalise whichever range comes back.
        let max = y.iter().copied().fold(0.0f32, f32::max);
        let k = if max > 2.0 { 1.0 / 255.0 } else { 1.0 };
        let mut data = vec![0f32; 3 * n * n];
        for i in 0..n * n {
            for c in 0..3 {
                data[3 * i + c] = (y[c * n * n + i] * k).clamp(0.0, 1.0);
            }
        }
        Ok(RgbF { w: n, h: n, data })
    }
}

impl Inpaint2d for Lama {
    fn inpaint(&mut self, img: &RgbF, hole: &Mask) -> anyhow::Result<RgbF> {
        self.run(img, hole)
    }
}

/// Inpaint `hole` in `base` (full image): square crop around the hole (2x its size, at
/// least 192 px), resized to 512, inpainted, resized back, pasted inside the hole only.
pub fn inpaint_crop(model: &mut dyn Inpaint2d, base: &RgbF, hole: &Mask) -> anyhow::Result<RgbF> {
    let (top, left, side) = crate::img::square_crop(hole, 2.0, 192);
    let crop = base.crop(top, left, side, side);
    let mut cm = Mask::new(side, side);
    for y in 0..side {
        for x in 0..side {
            cm.data[y * side + x] = hole.get(left + x, top + y);
        }
    }
    let n = INPAINT_SIZE;
    let filled = model.inpaint(&crop.resize(n, n), &resize_mask(&cm, n, n))?.resize(side, side);
    let mut out = base.clone();
    for y in 0..side {
        for x in 0..side {
            if cm.data[y * side + x] {
                let (di, si) = (3 * ((top + y) * base.w + left + x), 3 * (y * side + x));
                out.data[di..di + 3].copy_from_slice(&filled.data[si..si + 3]);
            }
        }
    }
    Ok(out)
}

pub struct DepthAnything {
    session: Session,
    input: String,
}

impl DepthAnything {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Self::load_on(path, false)
    }

    /// [`Self::load`], on the CUDA execution provider when `cuda`.
    pub fn load_on(path: &Path, cuda: bool) -> anyhow::Result<Self> {
        let session = session_on(path, cuda)?;
        let names = input_names(&session);
        ensure!(names.len() == 1, "Depth-Anything: expected 1 input, got {names:?}");
        Ok(Self { input: names[0].clone(), session })
    }

    /// Relative inverse depth (larger = closer) at the image's resolution. Preprocessing
    /// follows the model's DPTImageProcessor config: keep aspect ratio, the side scaled
    /// least lands on 518, both sides rounded to multiples of 14, bicubic, ImageNet mean/std.
    pub fn run(&mut self, img: &RgbF) -> anyhow::Result<Vec<f32>> {
        self.run_at(img, img.w, img.h)
    }

    /// [`Self::run`], resized to `(w, h)` instead of the image's size.
    pub fn run_at(&mut self, img: &RgbF, out_w: usize, out_h: usize) -> anyhow::Result<Vec<f32>> {
        let (h, w) = (img.h as f64, img.w as f64);
        let (mut sh, mut sw) = (518.0 / h, 518.0 / w);
        if (1.0 - sw).abs() < (1.0 - sh).abs() {
            sh = sw;
        } else {
            sw = sh;
        }
        let m14 = |v: f64| (((v / 14.0).round() * 14.0) as usize).max(14);
        let (nh, nw) = (m14(sh * h), m14(sw * w));
        let small = img.resize(nw, nh);
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let mut planar = vec![0f32; 3 * nh * nw];
        for i in 0..nh * nw {
            for c in 0..3 {
                planar[c * nh * nw + i] = (small.data[3 * i + c] - mean[c]) / std[c];
            }
        }
        let out = self.session.run(ort::inputs![
            self.input.as_str() => TensorRef::from_array_view(([1i64, 3, nh as i64, nw as i64], planar.as_slice()))?,
        ])?;
        let (shape, d) = out[0].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        ensure!(dims.len() >= 2, "Depth-Anything output shape {dims:?}");
        let (oh, ow) = (dims[dims.len() - 2] as usize, dims[dims.len() - 1] as usize);
        ensure!(d.len() == oh * ow, "Depth-Anything output shape {dims:?}");
        Ok(resize_scalar(d, ow, oh, out_w, out_h))
    }
}
