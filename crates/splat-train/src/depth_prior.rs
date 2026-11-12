//! Monocular depth prior as a training regulariser (goal 3: plain walls and
//! dim rooms). A view's prior is inverse depth up to an unknown affine map
//! (Depth-Anything-class networks) or `1 / z` from sensor depth; the loss is
//! the Pearson correlation between it and the rendered inverse depth, which
//! ignores both scale and shift, so no per-frame alignment is needed. It is
//! taken over the whole image and over square patches: the global term fixes
//! the room layout, the patch term the local shape (a wall stays planar where
//! the photo has no texture to say so).

use burn::tensor::{Tensor, TensorData};
use brush_dataset::scene::DepthMap;

/// Patch side in pixels of the prior's resolution.
pub const PATCH: usize = 32;

/// Long side of the prior maps (and of the depth render): the depth term
/// is low-frequency, and a ~0.1 MP render keeps its step cost small.
pub const DEPTH_LONG_SIDE: usize = 384;

/// Pixels count as covered once the render's alpha passes this.
const MIN_ALPHA: f32 = 0.5;

/// A patch takes part when at least this share of it is covered and valid.
const MIN_PATCH_COVER: f32 = 0.5;

/// Per-row Pearson loss `1 - corr(p, t)` over the pixels with `m = 1`, and
/// the row's valid-pixel count. `p`, `t`, `m`: `[rows, k]`.
fn pearson_rows(p: Tensor<2>, t: Tensor<2>, m: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    let k = p.dims()[1] as f32;
    let cnt = m.clone().sum_dim(1);
    let inv_cnt = cnt.clone().clamp_min(1.0).recip();
    let mp = (p.clone() * m.clone()).sum_dim(1) * inv_cnt.clone();
    let mt = (t.clone() * m.clone()).sum_dim(1) * inv_cnt;
    let dp = (p - mp) * m.clone();
    let dt = (t - mt) * m;
    let cov = (dp.clone() * dt.clone()).sum_dim(1);
    let eps = 1e-8 * k;
    let vp = (dp.clone() * dp).sum_dim(1).add_scalar(eps);
    let vt = (dt.clone() * dt).sum_dim(1).add_scalar(eps);
    let corr = cov / (vp.sqrt() * vt.sqrt());
    (corr.neg().add_scalar(1.0), cnt)
}

/// Depth-prior loss for one view. `pred`: the inverse-depth render `[h, w, 4]`
/// (red = alpha-weighted inverse depth, alpha = coverage; see
/// `render_splats_inv_depth`), at the prior's resolution. Scalar `[1]`.
pub fn depth_prior_loss(pred: Tensor<3>, prior: &DepthMap) -> Tensor<1> {
    let [h, w, _] = pred.dims();
    debug_assert_eq!((h, w), (prior.h, prior.w));
    let device = pred.device();
    let target: Tensor<2> =
        Tensor::from_data(TensorData::new(prior.inv_depth.clone(), [h, w]), &device);
    let r = pred.clone().slice([0..h, 0..w, 0..1]).reshape([h, w]);
    let a = pred.slice([0..h, 0..w, 3..4]).reshape([h, w]);
    let m = a.clone().detach().greater_elem(MIN_ALPHA).float()
        * target.clone().greater_elem(0.0).float();
    let p = r / a.clamp_min(MIN_ALPHA);

    let flat = |x: Tensor<2>| x.reshape([1, h * w]);
    let (global, _) = pearson_rows(flat(p.clone()), flat(target.clone()), flat(m.clone()));
    let global = global.reshape([1]);

    let (ph, pw) = (h / PATCH, w / PATCH);
    if ph == 0 || pw == 0 {
        return global;
    }
    let patches = |x: Tensor<2>| {
        x.slice([0..ph * PATCH, 0..pw * PATCH])
            .reshape([ph, PATCH, pw, PATCH])
            .swap_dims(1, 2)
            .reshape([ph * pw, PATCH * PATCH])
    };
    let (local, cnt) = pearson_rows(patches(p), patches(target), patches(m));
    let keep = cnt.greater_elem(MIN_PATCH_COVER * (PATCH * PATCH) as f32).float();
    let n = keep.clone().sum().clamp_min(1.0);
    let local = (local * keep).sum() / n;
    global + local
}

/// Planarity term: edge-aware second-order smoothness of the rendered inverse
/// depth. A plane's inverse depth is affine in pixel coordinates, so its
/// second differences vanish; weighted by `flat_weight` (low texture) they
/// flatten plain walls the photo can't constrain, and leave edges alone.
/// Scale-free (divided by the mean rendered inverse depth). Scalar `[1]`.
pub fn planarity_loss(pred: Tensor<3>, prior: &DepthMap) -> Option<Tensor<1>> {
    let [h, w, _] = pred.dims();
    if prior.flat_weight.len() != h * w || h < 3 || w < 3 {
        return None;
    }
    let device = pred.device();
    let fw: Tensor<2> =
        Tensor::from_data(TensorData::new(prior.flat_weight.clone(), [h, w]), &device);
    let r = pred.clone().slice([0..h, 0..w, 0..1]).reshape([h, w]);
    let a = pred.slice([0..h, 0..w, 3..4]).reshape([h, w]);
    let m = a.clone().detach().greater_elem(MIN_ALPHA).float();
    let p = r / a.clamp_min(MIN_ALPHA);
    let scale = ((p.clone().detach() * m.clone()).sum() / m.clone().sum().clamp_min(1.0))
        .clamp_min(1e-6)
        .recip();
    let wm = fw * m;
    let dxx = p.clone().slice([0..h, 2..w]) - p.clone().slice([0..h, 1..w - 1]).mul_scalar(2.0)
        + p.clone().slice([0..h, 0..w - 2]);
    let dyy = p.clone().slice([2..h, 0..w]) - p.clone().slice([1..h - 1, 0..w]).mul_scalar(2.0)
        + p.slice([0..h - 2, 0..w]);
    // A second difference counts where all three taps are covered and flat.
    let wx = wm.clone().slice([0..h, 0..w - 2]) * wm.clone().slice([0..h, 1..w - 1]) * wm.clone().slice([0..h, 2..w]);
    let wy = wm.clone().slice([0..h - 2, 0..w]) * wm.clone().slice([1..h - 1, 0..w]) * wm.slice([2..h, 0..w]);
    let n = (wx.clone().sum() + wy.clone().sum()).clamp_min(1.0);
    let l = ((dxx.abs() * wx).sum() + (dyy.abs() * wy).sum()) / n;
    Some(l * scale.reshape([1]))
}

/// Planarity weight of a photo at the prior's size: `exp(-FLAT_K |grad I|)`
/// of its luminance (central differences), in [0, 1].
pub fn flat_weight(gray: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; w * h];
    let at = |x: usize, y: usize| gray[y * w + x];
    for y in 0..h {
        for x in 0..w {
            let gx = at((x + 1).min(w - 1), y) - at(x.saturating_sub(1), y);
            let gy = at(x, (y + 1).min(h - 1)) - at(x, y.saturating_sub(1));
            out[y * w + x] = (-FLAT_K * (gx * gx + gy * gy).sqrt()).exp();
        }
    }
    out
}

/// Gradient scale of [`flat_weight`]: a 0.05 luminance step over two pixels
/// halves the weight.
const FLAT_K: f32 = 14.0;

/// Resize a prior to `(w, h)` (nearest: invalid zeros must not blend into
/// valid values).
pub fn resize_nearest(src: &[f32], sw: usize, sh: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        let sy = ((y as f64 + 0.5) * sh as f64 / h as f64) as usize;
        let row = sy.min(sh - 1) * sw;
        for x in 0..w {
            let sx = ((x as f64 + 0.5) * sw as f64 / w as f64) as usize;
            out[y * w + x] = src[row + sx.min(sw - 1)];
        }
    }
    out
}

/// Prior size for an image of `(w, h)`: long side [`DEPTH_LONG_SIDE`] (never
/// upsampled), same aspect.
pub fn prior_size(w: usize, h: usize) -> (usize, usize) {
    let long = w.max(h);
    if long <= DEPTH_LONG_SIDE {
        return (w, h);
    }
    let s = DEPTH_LONG_SIDE as f64 / long as f64;
    (((w as f64 * s).round() as usize).max(1), ((h as f64 * s).round() as usize).max(1))
}
