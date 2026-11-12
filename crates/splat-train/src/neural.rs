//! Derived from nerficg-project/efficient-gaussian-appearance at 4efbe67, Copyright 2026 The Efficient
//! Gaussian Appearance Authors, Apache-2.0 (see NOTICE);
//! upstream files Appearance/Neural.py and FasterGSVDA paper_configs/default.
//!
//! Training recipe for compact neural appearance (see
//! `brush_render::kernels::neural`), ported from the X5b reference
//! (Faster-GS VDA "Neural" appearance, paper_configs/default):
//! - base colour and latents start from the init colours / zeros; the shared
//!   decoder starts Xavier-uniform with its last layer zeroed, so the residual
//!   is exactly 0 until it trains;
//! - direction SH bands switch on one per `sh_degree_interval` steps;
//! - base/latent Adam LRs hold, then cosine-decay to 0 from 7/30 to 37/30 of
//!   training; the decoder has its own Adam (beta2 0.99) with a delayed
//!   exponential decay to 0.1x. Step counts are the reference's 30k schedule
//!   scaled to `total_train_iters`.

use brush_render::gaussian_splats::Splats;
use brush_render::kernels::neural::{
    NEURAL_DECODER_LEN, NEURAL_HIDDEN, NEURAL_INPUTS, NEURAL_ROW_COEFFS, NEURAL_W3,
};
use brush_render::kernels::sh::SH_C0;
use burn::tensor::{Tensor, TensorData, s};
use rand::RngExt;

/// Reference base-colour LR in colour units; rows store the base as a DC
/// coefficient (`base = C0 * c`), so the step is divided by C0.
const LR_BASE: f64 = 2.5e-4;
const LR_LATENT: f64 = 8e-3;
const LR_DECODER: f64 = 3e-3;
const REFERENCE_ITERS: f64 = 30_000.0;

pub(crate) fn base_lr() -> f64 {
    LR_BASE / f64::from(SH_C0)
}

/// Per-coefficient LR scales `[1, NEURAL_ROW_COEFFS, 1]` relative to
/// [`base_lr`]: coefficient 0 is the base, the rest hold the latents.
pub(crate) fn row_lr_scales(device: &burn::tensor::Device) -> Tensor<3> {
    let latent = (LR_LATENT / base_lr()) as f32;
    let mut scales = vec![latent; NEURAL_ROW_COEFFS as usize];
    scales[0] = 1.0;
    Tensor::<1>::from_floats(scales.as_slice(), device).reshape([1, NEURAL_ROW_COEFFS as i32, 1])
}

/// Hold, then cosine to 0 between 7k and 37k reference steps.
pub(crate) fn row_lr_factor(step: u32, total: u32) -> f64 {
    let k = f64::from(total.max(1)) / REFERENCE_ITERS;
    let (start, end) = (7_000.0 * k, 37_000.0 * k);
    let s = f64::from(step);
    if s <= start {
        1.0
    } else if s >= end {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * (s - start) / (end - start)).cos())
    }
}

/// 3DGS-style delayed exponential decay 3e-3 -> 3e-4 over the run, with a
/// 3k-reference-step sine warmup from 1e-8.
pub(crate) fn decoder_lr(step: u32, total: u32) -> f64 {
    let k = f64::from(total.max(1)) / REFERENCE_ITERS;
    let s = f64::from(step);
    let delay = 3_000.0 * k;
    let mult = 1e-8;
    let warm = mult + (1.0 - mult) * (0.5 * std::f64::consts::PI * (s / delay).clamp(0.0, 1.0)).sin();
    let t = (s / (REFERENCE_ITERS * k)).clamp(0.0, 1.0);
    let lerp = (LR_DECODER.ln() * (1.0 - t) + (LR_DECODER * 0.1).ln() * t).exp();
    warm * lerp
}

/// Colour tail: the last 3k of the reference's 30k steps train appearance
/// only. Means, scales, rotations and opacities stay fixed, with no
/// refinement, noise or opacity decay; base, latents and decoder keep
/// their schedules.
pub(crate) fn colour_tail(step: u32, total: u32) -> bool {
    step > total - total / 10
}

/// Convert freshly initialised SH splats: the DC coefficient becomes the
/// base, latents start at 0 and the decoder is newly initialised.
pub(crate) fn to_neural(splats: Splats, rng: &mut impl rand::Rng) -> Splats {
    let device = splats.device();
    let n = splats.num_splats() as usize;
    let dc = splats
        .sh_coeffs
        .val()
        .slice(s![.., 0..1, ..])
        .cast(burn::tensor::FloatDType::F32)
        .detach();
    let latents = Tensor::<3>::zeros([n, NEURAL_ROW_COEFFS as usize - 1, 3], &device);
    let rows = Tensor::cat(vec![dc, latents], 1);

    // Xavier-uniform per layer (as tiny-cuda-nn), last layer zero.
    let (i, h) = (NEURAL_INPUTS as usize, NEURAL_HIDDEN as usize);
    let mut w = Vec::with_capacity(NEURAL_DECODER_LEN as usize);
    for (fan_in, fan_out) in [(i, h), (h, h)] {
        let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
        w.extend((0..fan_in * fan_out).map(|_| rng.random_range(-a..a)));
    }
    assert_eq!(w.len(), NEURAL_W3 as usize);
    w.resize(NEURAL_DECODER_LEN as usize, 0.0);
    let decoder = Tensor::from_data(TensorData::new(w, [NEURAL_DECODER_LEN as usize]), &device);
    splats.with_neural_appearance(rows, decoder, 0)
}
