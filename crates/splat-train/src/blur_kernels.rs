//! Fused camera-motion blur blend: the linear-radiance blend of a photo's
//! half-resolution exposure sub-renders with its sharp render, and the 2x
//! bilinear upsample it uses, as a handful of kernels with their own
//! backward. Replaces ~150 small tensor ops (slices, cats, stacks and their
//! autodiff transposes) per blur step.
//!
//! Upsampling is half-pixel-centred bilinear with clamped edges, per axis:
//! full sample `2i` is `0.75 t[i] + 0.25 t[i - 1]`, `2i + 1` is
//! `0.75 t[i] + 0.25 t[i + 1]`, then cropped to the full size; a one-sample
//! axis repeats. The blend is `((1 - m) sharp^G + m blurred)^(1/G)` with
//! `blurred = sum_k w_k sub_k^G` (or the sub-render mean), sub-render alpha
//! averaged, everything clamped to `1e-8` before the power as before.

use brush_cube::create_tensor;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::{
    Autodiff, AutodiffBackend, Backend, Dispatch, TensorMetadata,
    autodiff::{
        checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::FloatTensor,
};
use burn::cubecl::{self, CubeDim, calculate_cube_count_elemwise, cube, prelude::*};
use burn::tensor::{DType, Tensor as BTensor};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};

const WORKGROUP: u32 = 256;

/// Linear radiance floor before the gamma power.
const FLOOR: f32 = 1e-8;

mod ops {
    #![allow(unused_variables)]
    use burn::backend::fusion::custom::TensorSpec;
    use burn::backend::{Backend, tensor::FloatTensor};
    use burn::tensor::{DType, Shape};

    fn spec(dims: impl Into<Shape>) -> TensorSpec {
        TensorSpec::new(dims.into(), DType::F32)
    }

    /// `[1, C, sh, sw]` half-resolution blend of `[K, sh, sw, C]` sub-renders.
    pub(super) fn half_spec(subs: &TensorSpec) -> TensorSpec {
        spec([1, subs.shape[3], subs.shape[1], subs.shape[2]])
    }

    /// Per-pixel gradient planes of the blend backward: the blurred RGB,
    /// sub-render alpha when the output alpha comes from them, and the mask.
    pub(super) fn dfull_planes(half_channels: usize, sharp_channels: usize) -> usize {
        if half_channels > 3 {
            sharp_channels + 1
        } else {
            4
        }
    }

    #[burn::backend::backend_extension(Cube, Autodiff, Fusion)]
    pub trait BlurOps: Backend {
        /// 2x bilinear upsample of `[N, C, H, W]` cropped to `[N, C, h, w]`.
        #[fusion(meta = |t, h, w| spec([t.shape[0], t.shape[1], *h, *w]))]
        fn upsample2(t: FloatTensor<Self>, h: usize, w: usize) -> FloatTensor<Self>;

        /// Transpose of [`Self::upsample2`]: `[N, C, h, w]` to `[N, C, sh, sw]`.
        #[fusion(meta = |grad, sh, sw| spec([grad.shape[0], grad.shape[1], *sh, *sw]))]
        fn upsample2_bwd(grad: FloatTensor<Self>, sh: usize, sw: usize) -> FloatTensor<Self>;

        /// `[K, 3, sh, sw]` linear radiance of `[K, sh, sw, C]` sub-renders
        /// (the mixer's input). No gradient.
        #[fusion(meta = |subs| spec([subs.shape[0], 3, subs.shape[1], subs.shape[2]]))]
        fn blur_linear(subs: FloatTensor<Self>) -> FloatTensor<Self>;

        /// The blend of `sharp` `[h, w, C]` with the mean of `subs`
        /// `[K, sh, sw, C]` under a constant `mask` `[1, 1, sh, sw]` (no
        /// gradient), and the half-resolution blur `[1, C, sh, sw]` (no
        /// gradient): RGB blurred, then the mean sub-render alpha.
        #[fusion(meta = |sharp, subs, mask| (sharp.clone(), half_spec(subs)))]
        fn blur_blend_mean(
            sharp: FloatTensor<Self>,
            subs: FloatTensor<Self>,
            mask: FloatTensor<Self>,
        ) -> (FloatTensor<Self>, FloatTensor<Self>);

        /// [`Self::blur_blend_mean`] with per-pixel sub-render `weights`
        /// `[K, 1, sh, sw]`; the blend has gradients to `sharp`, `subs`,
        /// `weights` and `mask`.
        #[fusion(meta = |sharp, subs, weights, mask| (sharp.clone(), half_spec(subs)))]
        fn blur_blend_weighted(
            sharp: FloatTensor<Self>,
            subs: FloatTensor<Self>,
            weights: FloatTensor<Self>,
            mask: FloatTensor<Self>,
        ) -> (FloatTensor<Self>, FloatTensor<Self>);

        /// Blend of `sharp` `[h, w, C]` with a precomputed half-resolution
        /// blur `half` `[1, 3 or C, sh, sw]` and `mask` `[1, 1, sh, sw]`.
        /// Output alpha is the upsampled `half` alpha when it has one, else
        /// `sharp`'s. Gradients reach `sharp` only.
        #[fusion(meta = |sharp, half, mask| sharp.clone())]
        fn blur_blend_half(
            sharp: FloatTensor<Self>,
            half: FloatTensor<Self>,
            mask: FloatTensor<Self>,
        ) -> FloatTensor<Self>;

        /// Blend backward at full resolution: `d_sharp` `[h, w, C]` and,
        /// when `planes`, the per-pixel gradients `[P, h, w]` the
        /// half-resolution inputs gather (see [`dfull_planes`]); else `[1]`.
        #[fusion(meta = |sharp, half, mask, grad, planes| (
            sharp.clone(),
            if *planes {
                spec([dfull_planes(half.shape[1], sharp.shape[2]), sharp.shape[0], sharp.shape[1]])
            } else {
                spec([1])
            },
        ))]
        fn blur_bwd_full(
            sharp: FloatTensor<Self>,
            half: FloatTensor<Self>,
            mask: FloatTensor<Self>,
            grad: FloatTensor<Self>,
            planes: bool,
        ) -> (FloatTensor<Self>, FloatTensor<Self>);

        /// Sub-render gradient `[K, sh, sw, C]` of the mean blend.
        #[fusion(meta = |subs, dfull| subs.clone())]
        fn blur_bwd_half_mean(
            subs: FloatTensor<Self>,
            dfull: FloatTensor<Self>,
        ) -> FloatTensor<Self>;

        /// Gradients of the weighted blend w.r.t. `subs`, `weights` and the
        /// mask `[1, 1, sh, sw]`.
        #[fusion(meta = |subs, weights, dfull| (
            subs.clone(),
            weights.clone(),
            spec([1, 1, subs.shape[1], subs.shape[2]]),
        ))]
        fn blur_bwd_half_weighted(
            subs: FloatTensor<Self>,
            weights: FloatTensor<Self>,
            dfull: FloatTensor<Self>,
        ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

        /// `t` in the forward, its gradient scaled by `scale`.
        #[fusion(meta = |t, scale| t.clone())]
        fn scale_grad(t: FloatTensor<Self>, #[fusion(scalar)] scale: f32) -> FloatTensor<Self>;
    }
}
use ops::{BlurOps, dfull_planes};

// ---------------------------------------------------------------- kernels

/// Taps of full-resolution sample `o` on an `n`-sample half axis: the
/// sample's own half index, its neighbour, and their weights.
#[cube]
fn taps(o: u32, n: u32) -> (u32, u32, f32, f32) {
    let i = o / 2u32;
    let odd = o % 2u32 == 1u32;
    let next = select(i + 1u32 < n, i + 1u32, n - 1u32);
    let prev = select(i > 0u32, i, 1u32) - 1u32;
    let single = n == 1u32;
    (
        i,
        select(odd, next, prev),
        select(single, 1.0f32, 0.75f32),
        select(single, 0.0f32, 0.25f32),
    )
}

/// Bilinear sample of the `sw`-wide plane at `off`, rows first.
#[cube]
#[allow(clippy::too_many_arguments)]
fn up(
    t: &Tensor<f32>,
    off: u32,
    sw: u32,
    y0: u32,
    y1: u32,
    x0: u32,
    x1: u32,
    wy0: f32,
    wy1: f32,
    wx0: f32,
    wx1: f32,
) -> f32 {
    let a = t[(off + y0 * sw + x0) as usize] * wy0 + t[(off + y1 * sw + x0) as usize] * wy1;
    let b = t[(off + y0 * sw + x1) as usize] * wy0 + t[(off + y1 * sw + x1) as usize] * wy1;
    a * wx0 + b * wx1
}

/// Full-resolution sample `slot` (0..4) that half sample `i` feeds on an
/// axis of `n` half and `len` full samples, with its weight (0 when none).
#[cube]
#[allow(unused_assignments)]
fn feed(i: u32, n: u32, len: u32, #[comptime] slot: u32) -> (u32, f32) {
    // Samples 2i - 1 .. 2i + 2: the outer two lean on `i` at 0.25, the
    // inner two at 0.75 plus the clamped edge's 0.25.
    let o = 2u32 * i + slot;
    let mut weight = 0.0f32;
    if comptime![slot == 0] {
        weight = select(i > 0u32, 0.25f32, 0.0f32);
    } else if comptime![slot == 1] {
        weight = select(i == 0u32, 1.0f32, 0.75f32);
    } else if comptime![slot == 2] {
        weight = select(i + 1u32 == n, 1.0f32, 0.75f32);
    } else {
        weight = select(i + 1u32 < n, 0.25f32, 0.0f32);
    }
    // `o` is offset by one: slot 0 is sample 2i - 1.
    let valid = o >= 1u32 && o - 1u32 < len;
    (select(valid, o - 1u32, 0u32), select(valid, weight, 0.0f32))
}

/// Transposed upsample: the gradient half sample `(i, j)` gathers from the
/// `w`-wide full-resolution plane at `off`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn gather(g: &Tensor<f32>, off: u32, i: u32, j: u32, sh: u32, sw: u32, h: u32, w: u32) -> f32 {
    let mut acc = 0.0f32;
    #[unroll]
    for a in 0u32..4u32 {
        let (y, wy) = feed(i, sh, h, a);
        let mut row = 0.0f32;
        #[unroll]
        for b in 0u32..4u32 {
            let (x, wx) = feed(j, sw, w, b);
            row += g[(off + y * w + x) as usize] * wx;
        }
        acc += row * wy;
    }
    acc
}

#[cube]
fn linear(raw: f32, gamma: f32) -> f32 {
    f32::powf(f32::max(raw, FLOOR), gamma)
}

/// `d linear / d raw`, zero below the floor (the clamp's gradient).
#[cube]
fn linear_grad(raw: f32, gamma: f32) -> f32 {
    select(
        raw >= FLOOR,
        f32::powf(f32::max(raw, FLOOR), gamma - 1.0f32) * gamma,
        0.0f32,
    )
}

#[cube(launch)]
fn upsample_kernel(t: &Tensor<f32>, out: &mut Tensor<f32>, sh: u32, sw: u32, h: u32, w: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= out.len() as u32 {
        terminate!();
    }
    let plane = idx / (h * w);
    let pos = idx % (h * w);
    let (y0, y1, wy0, wy1) = taps(pos / w, sh);
    let (x0, x1, wx0, wx1) = taps(pos % w, sw);
    out[idx as usize] = up(t, plane * sh * sw, sw, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
}

#[cube(launch)]
fn upsample_bwd_kernel(
    grad: &Tensor<f32>,
    out: &mut Tensor<f32>,
    sh: u32,
    sw: u32,
    h: u32,
    w: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= out.len() as u32 {
        terminate!();
    }
    let plane = idx / (sh * sw);
    let pos = idx % (sh * sw);
    out[idx as usize] = gather(grad, plane * h * w, pos / sw, pos % sw, sh, sw, h, w);
}

#[cube(launch)]
fn linear_kernel(
    subs: &Tensor<f32>,
    out: &mut Tensor<f32>,
    plane: u32,
    gamma: f32,
    #[comptime] cn: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= out.len() as u32 {
        terminate!();
    }
    let pos = idx % plane;
    let kc = idx / plane;
    let k = kc / 3u32;
    out[idx as usize] = linear(subs[((k * plane + pos) * cn + kc % 3u32) as usize], gamma);
}

/// Half-resolution blur `[1, C, sh, sw]`: RGB `sum_k w_k linear(sub_k)` (or
/// the mean), then the mean sub-render alpha.
#[cube(launch)]
fn half_kernel(
    subs: &Tensor<f32>,
    weights: &Tensor<f32>,
    half: &mut Tensor<f32>,
    k: u32,
    plane: u32,
    gamma: f32,
    #[comptime] cn: u32,
    #[comptime] weighted: bool,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos >= plane {
        terminate!();
    }
    let count = f32::cast_from(k);
    #[unroll]
    for c in 0u32..cn {
        let mut acc = 0.0f32;
        for s in 0u32..k {
            let raw = subs[((s * plane + pos) * cn + c) as usize];
            if comptime![c < 3] {
                let v = linear(raw, gamma);
                if weighted {
                    acc += weights[(s * plane + pos) as usize] * v;
                } else {
                    acc += v;
                }
            } else {
                acc += raw;
            }
        }
        if comptime![c >= 3 || !weighted] {
            acc /= count;
        }
        half[(c * plane + pos) as usize] = acc;
    }
}

/// The blend at full resolution, one thread per pixel.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn blend_kernel(
    sharp: &Tensor<f32>,
    half: &Tensor<f32>,
    mask: &Tensor<f32>,
    out: &mut Tensor<f32>,
    h: u32,
    w: u32,
    sh: u32,
    sw: u32,
    gamma: f32,
    inv_gamma: f32,
    #[comptime] cn: u32,
    #[comptime] half_alpha: bool,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos >= h * w {
        terminate!();
    }
    let (y0, y1, wy0, wy1) = taps(pos / w, sh);
    let (x0, x1, wx0, wx1) = taps(pos % w, sw);
    let plane = sh * sw;
    let m = up(mask, 0u32, sw, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
    let keep = 1.0f32 - m;
    let base = pos * cn;
    #[unroll]
    for c in 0u32..cn {
        if comptime![c < 3] {
            let s = linear(sharp[(base + c) as usize], gamma);
            let b = up(half, c * plane, sw, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
            out[(base + c) as usize] = f32::powf(s * keep + b * m, inv_gamma);
        } else if half_alpha {
            out[(base + c) as usize] = up(half, c * plane, sw, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
        } else {
            out[(base + c) as usize] = sharp[(base + c) as usize];
        }
    }
}

/// Blend backward per full-resolution pixel: `d_sharp`, and (`planes`)
/// the gradients w.r.t. the upsampled blur RGB, alpha and mask.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn blend_bwd_full_kernel(
    sharp: &Tensor<f32>,
    half: &Tensor<f32>,
    mask: &Tensor<f32>,
    grad: &Tensor<f32>,
    d_sharp: &mut Tensor<f32>,
    dfull: &mut Tensor<f32>,
    h: u32,
    w: u32,
    sh: u32,
    sw: u32,
    gamma: f32,
    inv_gamma: f32,
    #[comptime] cn: u32,
    #[comptime] half_alpha: bool,
    #[comptime] planes: bool,
) {
    let pos = ABSOLUTE_POS as u32;
    let hw = h * w;
    if pos >= hw {
        terminate!();
    }
    let (y0, y1, wy0, wy1) = taps(pos / w, sh);
    let (x0, x1, wx0, wx1) = taps(pos % w, sw);
    let plane = sh * sw;
    let m = up(mask, 0u32, sw, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
    let keep = 1.0f32 - m;
    let base = pos * cn;
    let mut dm = 0.0f32;
    #[unroll]
    for c in 0u32..cn {
        let g = grad[(base + c) as usize];
        if comptime![c < 3] {
            let raw = sharp[(base + c) as usize];
            let s = linear(raw, gamma);
            let b = up(half, c * plane, sw, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
            let p = s * keep + b * m;
            let dp = g * (f32::powf(p, inv_gamma - 1.0f32) * inv_gamma);
            d_sharp[(base + c) as usize] = dp * keep * linear_grad(raw, gamma);
            if planes {
                dfull[(c * hw + pos) as usize] = dp * m;
                dm += dp * (b - s);
            }
        } else if half_alpha {
            d_sharp[(base + c) as usize] = 0.0f32;
            if planes {
                dfull[(c * hw + pos) as usize] = g;
            }
        } else {
            d_sharp[(base + c) as usize] = g;
        }
    }
    if planes {
        let mask_plane = comptime![if half_alpha { cn } else { 3u32 }];
        dfull[(mask_plane * hw + pos) as usize] = dm;
    }
}

/// Blend backward per half-resolution pixel: gathers the upsampled
/// gradients, then `d_subs` and (`weighted`) `d_weights` and `d_mask`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn blend_bwd_half_kernel(
    subs: &Tensor<f32>,
    weights: &Tensor<f32>,
    dfull: &Tensor<f32>,
    d_subs: &mut Tensor<f32>,
    d_weights: &mut Tensor<f32>,
    d_mask: &mut Tensor<f32>,
    k: u32,
    h: u32,
    w: u32,
    sh: u32,
    sw: u32,
    gamma: f32,
    #[comptime] cn: u32,
    #[comptime] weighted: bool,
) {
    let pos = ABSOLUTE_POS as u32;
    let plane = sh * sw;
    if pos >= plane {
        terminate!();
    }
    let (i, j) = (pos / sw, pos % sw);
    let hw = h * w;
    let d0 = gather(dfull, 0u32, i, j, sh, sw, h, w);
    let d1 = gather(dfull, hw, i, j, sh, sw, h, w);
    let d2 = gather(dfull, 2u32 * hw, i, j, sh, sw, h, w);
    let mut da = 0.0f32;
    if comptime![cn > 3] {
        da = gather(dfull, 3u32 * hw, i, j, sh, sw, h, w);
    }
    if weighted {
        d_mask[pos as usize] = gather(dfull, cn * hw, i, j, sh, sw, h, w);
    }
    let count = f32::cast_from(k);
    for s in 0u32..k {
        let base = (s * plane + pos) * cn;
        let mut wk = 1.0f32 / count;
        if weighted {
            wk = weights[(s * plane + pos) as usize];
        }
        let r0 = subs[base as usize];
        let r1 = subs[(base + 1u32) as usize];
        let r2 = subs[(base + 2u32) as usize];
        d_subs[base as usize] = d0 * wk * linear_grad(r0, gamma);
        d_subs[(base + 1u32) as usize] = d1 * wk * linear_grad(r1, gamma);
        d_subs[(base + 2u32) as usize] = d2 * wk * linear_grad(r2, gamma);
        if comptime![cn > 3] {
            d_subs[(base + 3u32) as usize] = da / count;
        }
        if weighted {
            d_weights[(s * plane + pos) as usize] =
                d0 * linear(r0, gamma) + d1 * linear(r1, gamma) + d2 * linear(r2, gamma);
        }
    }
}

// ------------------------------------------------------------- CubeBackend

fn f32_contiguous(t: CubeTensor) -> CubeTensor {
    assert_eq!(t.dtype, DType::F32, "blur blend runs in f32");
    into_contiguous(t)
}

fn elemwise(t: &CubeTensor, n: usize) -> (CubeCount, CubeDim) {
    let dim = CubeDim::new_1d(WORKGROUP);
    (calculate_cube_count_elemwise(&t.client, n.max(1), dim), dim)
}

/// A one-element binding for an argument a comptime flag leaves unused.
/// Never another argument's buffer: wgpu rejects a buffer bound twice.
fn placeholder(like: &CubeTensor) -> CubeTensor {
    create_tensor([1], &like.device, DType::F32)
}

fn dims4(t: &CubeTensor) -> [usize; 4] {
    let s = t.shape();
    assert_eq!(s.num_dims(), 4, "expected a 4-d tensor");
    [s[0], s[1], s[2], s[3]]
}

/// `[h, w, C]` of a sharp render (3 or 4 channels).
fn image_dims(sharp: &CubeTensor) -> (usize, usize, usize) {
    let s = sharp.shape();
    assert_eq!(s.num_dims(), 3, "expected an [h, w, C] render");
    assert!(s[2] == 3 || s[2] == 4, "blur blend expects 3 or 4 channels");
    (s[0], s[1], s[2])
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("blur blend size exceeds u32")
}

/// Half-resolution blur; `weights` is `None` for the mean.
fn launch_half(subs: &CubeTensor, weights: Option<&CubeTensor>) -> CubeTensor {
    let [k, sh, sw, cn] = dims4(subs);
    assert!(cn == 3 || cn == 4, "sub-renders have 3 or 4 channels");
    let half = create_tensor([1, cn, sh, sw], &subs.device, DType::F32);
    let (count, dim) = elemwise(subs, sh * sw);
    half_kernel::launch(
        &subs.client,
        count,
        dim,
        subs.clone().into_tensor_arg(),
        weights
            .cloned()
            .unwrap_or_else(|| placeholder(subs))
            .into_tensor_arg(),
        half.clone().into_tensor_arg(),
        u(k),
        u(sh * sw),
        GAMMA,
        u(cn),
        weights.is_some(),
    );
    half
}

fn launch_blend(sharp: &CubeTensor, half: &CubeTensor, mask: &CubeTensor) -> CubeTensor {
    let (h, w, cn) = image_dims(sharp);
    let [_, hc, sh, sw] = dims4(half);
    assert_eq!(dims4(mask), [1, 1, sh, sw], "mask matches the half size");
    assert!(
        h <= 2 * sh && w <= 2 * sw,
        "half size too small for the render"
    );
    let out = create_tensor([h, w, cn], &sharp.device, DType::F32);
    let (count, dim) = elemwise(sharp, h * w);
    blend_kernel::launch(
        &sharp.client,
        count,
        dim,
        sharp.clone().into_tensor_arg(),
        half.clone().into_tensor_arg(),
        mask.clone().into_tensor_arg(),
        out.clone().into_tensor_arg(),
        u(h),
        u(w),
        u(sh),
        u(sw),
        GAMMA,
        1.0 / GAMMA,
        u(cn),
        hc > 3,
    );
    out
}

fn launch_bwd_half(
    subs: CubeTensor,
    weights: Option<CubeTensor>,
    dfull: CubeTensor,
) -> (CubeTensor, Option<(CubeTensor, CubeTensor)>) {
    let subs = f32_contiguous(subs);
    let weights = weights.map(f32_contiguous);
    let dfull = f32_contiguous(dfull);
    let [k, sh, sw, cn] = dims4(&subs);
    let fs = dfull.shape();
    assert_eq!(
        fs[0],
        dfull_planes(cn, cn),
        "gradient planes of a sub-render blend"
    );
    let (h, w) = (fs[1], fs[2]);
    let device = &subs.device;
    let d_subs = create_tensor([k, sh, sw, cn], device, DType::F32);
    let extra = weights.as_ref().map(|_| {
        (
            create_tensor([k, 1, sh, sw], device, DType::F32),
            create_tensor([1, 1, sh, sw], device, DType::F32),
        )
    });
    let (d_w, d_m) = extra
        .clone()
        .unwrap_or_else(|| (placeholder(&subs), placeholder(&subs)));
    let (count, dim) = elemwise(&subs, sh * sw);
    blend_bwd_half_kernel::launch(
        &subs.client,
        count,
        dim,
        subs.clone().into_tensor_arg(),
        weights
            .clone()
            .unwrap_or_else(|| placeholder(&subs))
            .into_tensor_arg(),
        dfull.into_tensor_arg(),
        d_subs.clone().into_tensor_arg(),
        d_w.into_tensor_arg(),
        d_m.into_tensor_arg(),
        u(k),
        u(h),
        u(w),
        u(sh),
        u(sw),
        GAMMA,
        u(cn),
        weights.is_some(),
    );
    (d_subs, extra)
}

/// Display <-> linear radiance exponent (`train::GAMMA`).
const GAMMA: f32 = crate::train::GAMMA;

impl BlurOps for CubeBackend {
    fn upsample2(t: FloatTensor<Self>, h: usize, w: usize) -> FloatTensor<Self> {
        let t = f32_contiguous(t);
        let [n, c, sh, sw] = dims4(&t);
        assert!(
            h <= 2 * sh && w <= 2 * sw,
            "upsample2 crops, it cannot grow past 2x"
        );
        let out = create_tensor([n, c, h, w], &t.device, DType::F32);
        let (count, dim) = elemwise(&t, n * c * h * w);
        upsample_kernel::launch(
            &t.client,
            count,
            dim,
            t.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            u(sh),
            u(sw),
            u(h),
            u(w),
        );
        out
    }

    fn upsample2_bwd(grad: FloatTensor<Self>, sh: usize, sw: usize) -> FloatTensor<Self> {
        let grad = f32_contiguous(grad);
        let [n, c, h, w] = dims4(&grad);
        let out = create_tensor([n, c, sh, sw], &grad.device, DType::F32);
        let (count, dim) = elemwise(&grad, n * c * sh * sw);
        upsample_bwd_kernel::launch(
            &grad.client,
            count,
            dim,
            grad.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            u(sh),
            u(sw),
            u(h),
            u(w),
        );
        out
    }

    fn blur_linear(subs: FloatTensor<Self>) -> FloatTensor<Self> {
        let subs = f32_contiguous(subs);
        let [k, sh, sw, cn] = dims4(&subs);
        let out = create_tensor([k, 3, sh, sw], &subs.device, DType::F32);
        let (count, dim) = elemwise(&subs, k * 3 * sh * sw);
        linear_kernel::launch(
            &subs.client,
            count,
            dim,
            subs.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            u(sh * sw),
            GAMMA,
            u(cn),
        );
        out
    }

    fn blur_blend_mean(
        sharp: FloatTensor<Self>,
        subs: FloatTensor<Self>,
        mask: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (sharp, subs, mask) = (
            f32_contiguous(sharp),
            f32_contiguous(subs),
            f32_contiguous(mask),
        );
        assert_eq!(
            dims4(&subs)[3],
            image_dims(&sharp).2,
            "sub-renders match the sharp channels"
        );
        let half = launch_half(&subs, None);
        (launch_blend(&sharp, &half, &mask), half)
    }

    fn blur_blend_weighted(
        sharp: FloatTensor<Self>,
        subs: FloatTensor<Self>,
        weights: FloatTensor<Self>,
        mask: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (sharp, subs, mask) = (
            f32_contiguous(sharp),
            f32_contiguous(subs),
            f32_contiguous(mask),
        );
        let weights = f32_contiguous(weights);
        let [k, sh, sw, cn] = dims4(&subs);
        assert_eq!(
            cn,
            image_dims(&sharp).2,
            "sub-renders match the sharp channels"
        );
        assert_eq!(
            dims4(&weights),
            [k, 1, sh, sw],
            "one weight per sub-render pixel"
        );
        let half = launch_half(&subs, Some(&weights));
        (launch_blend(&sharp, &half, &mask), half)
    }

    fn blur_blend_half(
        sharp: FloatTensor<Self>,
        half: FloatTensor<Self>,
        mask: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let (sharp, half, mask) = (
            f32_contiguous(sharp),
            f32_contiguous(half),
            f32_contiguous(mask),
        );
        let hc = dims4(&half)[1];
        assert!(
            hc == 3 || hc == image_dims(&sharp).2,
            "half holds RGB or every channel"
        );
        launch_blend(&sharp, &half, &mask)
    }

    fn blur_bwd_full(
        sharp: FloatTensor<Self>,
        half: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        planes: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (sharp, half, mask) = (
            f32_contiguous(sharp),
            f32_contiguous(half),
            f32_contiguous(mask),
        );
        let grad = f32_contiguous(grad);
        let (h, w, cn) = image_dims(&sharp);
        let [_, hc, sh, sw] = dims4(&half);
        assert_eq!(
            grad.shape().as_slice(),
            sharp.shape().as_slice(),
            "gradient of the blend"
        );
        let d_sharp = create_tensor([h, w, cn], &sharp.device, DType::F32);
        let dfull = if planes {
            create_tensor([dfull_planes(hc, cn), h, w], &sharp.device, DType::F32)
        } else {
            create_tensor([1], &sharp.device, DType::F32)
        };
        let (count, dim) = elemwise(&sharp, h * w);
        blend_bwd_full_kernel::launch(
            &sharp.client,
            count,
            dim,
            sharp.clone().into_tensor_arg(),
            half.into_tensor_arg(),
            mask.into_tensor_arg(),
            grad.into_tensor_arg(),
            d_sharp.clone().into_tensor_arg(),
            dfull.clone().into_tensor_arg(),
            u(h),
            u(w),
            u(sh),
            u(sw),
            GAMMA,
            1.0 / GAMMA,
            u(cn),
            hc > 3,
            planes,
        );
        (d_sharp, dfull)
    }

    fn blur_bwd_half_mean(subs: FloatTensor<Self>, dfull: FloatTensor<Self>) -> FloatTensor<Self> {
        launch_bwd_half(subs, None, dfull).0
    }

    fn blur_bwd_half_weighted(
        subs: FloatTensor<Self>,
        weights: FloatTensor<Self>,
        dfull: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (d_subs, extra) = launch_bwd_half(subs, Some(weights), dfull);
        let (d_w, d_m) = extra.expect("weighted outputs");
        (d_subs, d_w, d_m)
    }

    fn scale_grad(t: FloatTensor<Self>, _scale: f32) -> FloatTensor<Self> {
        t
    }
}

// ---------------------------------------------------------------- Autodiff

#[derive(Debug)]
struct UpsampleBackward;

impl<B: Backend + BlurOps> Backward<B, 1> for UpsampleBackward {
    type State = (usize, usize);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let grad = grads.consume::<B>(&ops.node);
        let [parent] = ops.parents;
        if let Some(node) = parent {
            let (sh, sw) = ops.state;
            grads.register::<B>(node.id, B::upsample2_bwd(grad, sh, sw));
        }
    }
}

#[derive(Debug)]
struct ScaleGradBackward;

impl<B: Backend> Backward<B, 1> for ScaleGradBackward {
    type State = f32;

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let grad = grads.consume::<B>(&ops.node);
        let [parent] = ops.parents;
        if let Some(node) = parent {
            grads.register::<B>(node.id, B::float_mul_scalar(grad, ops.state.into()));
        }
    }
}

/// Inputs the blend backward recomputes from (the forward keeps no
/// per-pixel intermediates beyond the half-resolution blur).
#[derive(Debug, Clone)]
struct BlendState<B: Backend> {
    sharp: FloatTensor<B>,
    subs: Option<FloatTensor<B>>,
    weights: Option<FloatTensor<B>>,
    half: FloatTensor<B>,
    mask: FloatTensor<B>,
}

/// Parents `[sharp, subs, weights, mask]` (weighted), `[sharp, subs]`
/// (mean) or `[sharp]` (precomputed blur).
#[derive(Debug)]
struct BlendBackward;

impl<B: Backend + BlurOps, const N: usize> Backward<B, N> for BlendBackward {
    type State = BlendState<B>;

    fn backward(self, ops: Ops<Self::State, N>, grads: &mut Gradients, _: &mut Checkpointer) {
        let grad = grads.consume::<B>(&ops.node);
        let state = ops.state;
        let mut parents = ops.parents.into_iter();
        let sharp_parent = parents.next().flatten();
        let rest: Vec<_> = parents.collect();
        let planes = rest.iter().any(Option::is_some);
        let (d_sharp, dfull) = B::blur_bwd_full(state.sharp, state.half, state.mask, grad, planes);
        if let Some(node) = sharp_parent {
            grads.register::<B>(node.id, d_sharp);
        }
        if !planes {
            return;
        }
        let subs = state.subs.expect("half inputs carry their sub-renders");
        match state.weights {
            Some(weights) => {
                let half_grads: [_; 3] = B::blur_bwd_half_weighted(subs, weights, dfull).into();
                for (parent, g) in rest.into_iter().zip(half_grads) {
                    if let Some(node) = parent {
                        grads.register::<B>(node.id, g);
                    }
                }
            }
            None => {
                if let Some(Some(node)) = rest.into_iter().next() {
                    grads.register::<B>(node.id, B::blur_bwd_half_mean(subs, dfull));
                }
            }
        }
    }
}

impl<B: Backend + BlurOps, C: CheckpointStrategy> BlurOps for Autodiff<B, C> {
    fn upsample2(t: FloatTensor<Self>, h: usize, w: usize) -> FloatTensor<Self> {
        let prep = UpsampleBackward
            .prepare::<NoCheckpointing>([t.node()])
            .compute_bound()
            .stateful();
        let t = t.into_primitive();
        let shape = t.shape();
        let state = (shape[2], shape[3]);
        let out = B::upsample2(t, h, w);
        match prep {
            OpsKind::Tracked(prep) => prep.finish(state, out),
            OpsKind::UnTracked(prep) => prep.finish(out),
        }
    }

    fn upsample2_bwd(grad: FloatTensor<Self>, sh: usize, sw: usize) -> FloatTensor<Self> {
        <Self as AutodiffBackend>::from_inner(B::upsample2_bwd(grad.into_primitive(), sh, sw))
    }

    fn blur_linear(subs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Self as AutodiffBackend>::from_inner(B::blur_linear(subs.into_primitive()))
    }

    fn blur_blend_mean(
        sharp: FloatTensor<Self>,
        subs: FloatTensor<Self>,
        mask: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let prep = BlendBackward
            .prepare::<NoCheckpointing>([sharp.node(), subs.node()])
            .compute_bound()
            .stateful();
        let (sharp, subs, mask) = (
            sharp.into_primitive(),
            subs.into_primitive(),
            mask.into_primitive(),
        );
        let (out, half) = B::blur_blend_mean(sharp.clone(), subs.clone(), mask.clone());
        let out = match prep {
            OpsKind::Tracked(prep) => {
                let state = BlendState {
                    sharp,
                    subs: Some(subs),
                    weights: None,
                    half: half.clone(),
                    mask,
                };
                prep.finish(state, out)
            }
            OpsKind::UnTracked(prep) => prep.finish(out),
        };
        (out, <Self as AutodiffBackend>::from_inner(half))
    }

    fn blur_blend_weighted(
        sharp: FloatTensor<Self>,
        subs: FloatTensor<Self>,
        weights: FloatTensor<Self>,
        mask: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let prep = BlendBackward
            .prepare::<NoCheckpointing>([sharp.node(), subs.node(), weights.node(), mask.node()])
            .compute_bound()
            .stateful();
        let (sharp, subs) = (sharp.into_primitive(), subs.into_primitive());
        let (weights, mask) = (weights.into_primitive(), mask.into_primitive());
        let (out, half) =
            B::blur_blend_weighted(sharp.clone(), subs.clone(), weights.clone(), mask.clone());
        let out = match prep {
            OpsKind::Tracked(prep) => {
                let state = BlendState {
                    sharp,
                    subs: Some(subs),
                    weights: Some(weights),
                    half: half.clone(),
                    mask,
                };
                prep.finish(state, out)
            }
            OpsKind::UnTracked(prep) => prep.finish(out),
        };
        (out, <Self as AutodiffBackend>::from_inner(half))
    }

    fn blur_blend_half(
        sharp: FloatTensor<Self>,
        half: FloatTensor<Self>,
        mask: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let prep = BlendBackward
            .prepare::<NoCheckpointing>([sharp.node()])
            .compute_bound()
            .stateful();
        let (sharp, half, mask) = (
            sharp.into_primitive(),
            half.into_primitive(),
            mask.into_primitive(),
        );
        let out = B::blur_blend_half(sharp.clone(), half.clone(), mask.clone());
        match prep {
            OpsKind::Tracked(prep) => prep.finish(
                BlendState {
                    sharp,
                    subs: None,
                    weights: None,
                    half,
                    mask,
                },
                out,
            ),
            OpsKind::UnTracked(prep) => prep.finish(out),
        }
    }

    fn blur_bwd_full(
        sharp: FloatTensor<Self>,
        half: FloatTensor<Self>,
        mask: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        planes: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (a, b) = B::blur_bwd_full(
            sharp.into_primitive(),
            half.into_primitive(),
            mask.into_primitive(),
            grad.into_primitive(),
            planes,
        );
        (
            <Self as AutodiffBackend>::from_inner(a),
            <Self as AutodiffBackend>::from_inner(b),
        )
    }

    fn blur_bwd_half_mean(subs: FloatTensor<Self>, dfull: FloatTensor<Self>) -> FloatTensor<Self> {
        <Self as AutodiffBackend>::from_inner(B::blur_bwd_half_mean(
            subs.into_primitive(),
            dfull.into_primitive(),
        ))
    }

    fn blur_bwd_half_weighted(
        subs: FloatTensor<Self>,
        weights: FloatTensor<Self>,
        dfull: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (a, b, c) = B::blur_bwd_half_weighted(
            subs.into_primitive(),
            weights.into_primitive(),
            dfull.into_primitive(),
        );
        let lift = <Self as AutodiffBackend>::from_inner;
        (lift(a), lift(b), lift(c))
    }

    fn scale_grad(t: FloatTensor<Self>, scale: f32) -> FloatTensor<Self> {
        let prep = ScaleGradBackward
            .prepare::<NoCheckpointing>([t.node()])
            .compute_bound()
            .stateful();
        let t: FloatTensor<B> = t.into_primitive();
        match prep {
            OpsKind::Tracked(prep) => prep.finish(scale, t),
            OpsKind::UnTracked(prep) => prep.finish(t),
        }
    }
}

// ------------------------------------------------------------ tensor API

/// 2x half-pixel-centred bilinear upsample of `[N, C, H, W]`, cropped to
/// `[h, w]` (an odd full size has a half-size of `(h + 1) / 2`).
pub(crate) fn upsample2(t: BTensor<4>, h: usize, w: usize) -> BTensor<4> {
    BTensor::from_dispatch(<Dispatch as BlurOps>::upsample2(t.into_dispatch(), h, w))
}

/// `[K, 3, sh, sw]` linear radiance of stacked `[K, sh, sw, C]` sub-renders,
/// without gradient.
pub(crate) fn linear_subs(subs: BTensor<4>) -> BTensor<4> {
    BTensor::from_dispatch(<Dispatch as BlurOps>::blur_linear(subs.into_dispatch()))
}

/// The blur blend of the sharp render `[h, w, C]` with stacked sub-renders
/// `[K, sh, sw, C]`: `weights` `[K, 1, sh, sw]` and `mask` `[1, 1, sh, sw]`
/// from the mixer, or the sub-render mean under the constant `mask` when
/// `weights` is `None`. Returns the blend and the half-resolution blur
/// `[1, C, sh, sw]` (no gradient).
pub(crate) fn blend(
    sharp: BTensor<3>,
    subs: BTensor<4>,
    weights: Option<BTensor<4>>,
    mask: BTensor<4>,
) -> (BTensor<3>, BTensor<4>) {
    let (out, half) = match weights {
        Some(weights) => <Dispatch as BlurOps>::blur_blend_weighted(
            sharp.into_dispatch(),
            subs.into_dispatch(),
            weights.into_dispatch(),
            mask.into_dispatch(),
        ),
        None => <Dispatch as BlurOps>::blur_blend_mean(
            sharp.into_dispatch(),
            subs.into_dispatch(),
            mask.detach().into_dispatch(),
        ),
    };
    (BTensor::from_dispatch(out), BTensor::from_dispatch(half))
}

/// The blend of `sharp` `[h, w, C]` with a precomputed half-resolution blur
/// `[1, 3, sh, sw]` and mask (both constants); alpha stays `sharp`'s.
pub(crate) fn blend_half(sharp: BTensor<3>, half: BTensor<4>, mask: BTensor<4>) -> BTensor<3> {
    BTensor::from_dispatch(<Dispatch as BlurOps>::blur_blend_half(
        sharp.into_dispatch(),
        half.detach().into_dispatch(),
        mask.detach().into_dispatch(),
    ))
}

/// `img` in the forward, with its gradient scaled by `scale`.
pub(crate) fn scale_grad(img: BTensor<3>, scale: f32) -> BTensor<3> {
    BTensor::from_dispatch(<Dispatch as BlurOps>::scale_grad(
        img.into_dispatch(),
        scale,
    ))
}
