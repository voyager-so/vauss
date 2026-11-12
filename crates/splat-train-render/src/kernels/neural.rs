//! Derived from nerficg-project/efficient-gaussian-appearance at 4efbe67, Copyright 2026 The Efficient
//! Gaussian Appearance Authors, Apache-2.0 (see NOTICE);
//! upstream files Appearance/Neural.py and
//! FasterGSVDACudaBackend/FasterGSVDACudaBackend/rasterization/rtc/appearance_neural.cuh.
//!
//! Compact neural appearance: one tiny shared decoder evaluated once per
//! visible splat in the projection kernel (never per pixel).
//!
//! Each splat stores a base colour `b` (as an SH DC coefficient, `b = C0·c`)
//! and an 8-value latent `f` in its
//! `sh_coeffs` row (`NEURAL_ROW_COEFFS` coefficients, so refine/prune/reorder
//! and export carry it like any SH row). For the unit view direction `d`:
//!
//! ```text
//! x  = [sin f0, cos f0, …, sin f7, cos f7, Y0(d), Y1(d)…, Y3(d)]   (32)
//! h1 = ReLU(W1 x)        W1: 16×32
//! h2 = ReLU(W2 h1)       W2: 16×16
//! c  = ReLU(b + tanh(W3 h2) + 0.5)                                  W3: 3×16
//! ```
//!
//! The decoder shares the projection kernels' `min_scale` binding:
//! `[decoder(816) | floor(N)]`, the floor read at `gid + NEURAL_DECODER_LEN`.
//! That keeps the kernels within WebGPU's default eight storage buffers.
//!
//! No biases. SH bands 1..3 are gated by the active direction degree (the
//! training schedule enables one band per interval; band 0 is always on).
//! Weights are row-major `[out][in]`, packed W1, W2, W3 in one `[816]`
//! tensor.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

use super::sh::SH_C0;
use super::types::Vec3A;

/// Coefficients (of 3 floats) per splat row in neural mode: base(3) +
/// latent(8) + one pad float.
pub const NEURAL_ROW_COEFFS: u32 = 4;
pub const NEURAL_LATENTS: u32 = 8;
pub const NEURAL_INPUTS: u32 = 32;
pub const NEURAL_HIDDEN: u32 = 16;
pub const NEURAL_W1: u32 = 0;
pub const NEURAL_W2: u32 = NEURAL_W1 + NEURAL_HIDDEN * NEURAL_INPUTS;
pub const NEURAL_W3: u32 = NEURAL_W2 + NEURAL_HIDDEN * NEURAL_HIDDEN;
/// Logical decoder weights: 512 + 256 + 48.
pub const NEURAL_DECODER_LEN: u32 = NEURAL_W3 + 3 * NEURAL_HIDDEN;
/// Highest SH band of the direction input.
pub const NEURAL_MAX_DEGREE: u32 = 3;

/// Stage the decoder (the first `NEURAL_DECODER_LEN` values of `src`) in
/// shared memory: every lane reads every weight, so this turns ~800 global
/// loads per splat into broadcast shared reads. Must be reached by every
/// unit of the cube (it syncs).
#[cube]
pub fn stage_decoder(src: &Tensor<f32>, dst: &mut Shared<[f32]>) {
    let mut i = UNIT_POS;
    while i < NEURAL_DECODER_LEN {
        dst[i as usize] = src[i as usize];
        i += CUBE_DIM;
    }
    sync_cube();
}

/// Real SH bands 0..3 of a unit direction (tiny-cuda-nn's `sh_enc`
/// polynomials), written to `x[16..32]`. Bands above `active_degree` are 0.
#[cube]
pub fn write_direction_basis(x: &mut Array<f32>, v: Vec3A, active_degree: u32) {
    let (vx, vy, vz) = (v.x(), v.y(), v.z());
    let xy = vx * vy;
    let xz = vx * vz;
    let yz = vy * vz;
    let x2 = vx * vx;
    let y2 = vy * vy;
    let z2 = vz * vz;
    let m1 = select(active_degree >= 1u32, 1.0f32, 0.0f32);
    let m2 = select(active_degree >= 2u32, 1.0f32, 0.0f32);
    let m3 = select(active_degree >= 3u32, 1.0f32, 0.0f32);
    x[16] = 0.282_094_8f32;
    x[17] = m1 * (-0.488_602_5f32 * vy);
    x[18] = m1 * (0.488_602_5f32 * vz);
    x[19] = m1 * (-0.488_602_5f32 * vx);
    x[20] = m2 * (1.092_548_4f32 * xy);
    x[21] = m2 * (-1.092_548_4f32 * yz);
    x[22] = m2 * (0.946_174_7f32 * z2 - 0.315_391_57f32);
    x[23] = m2 * (-1.092_548_4f32 * xz);
    x[24] = m2 * (0.546_274_2f32 * x2 - 0.546_274_2f32 * y2);
    x[25] = m3 * (0.590_043_6f32 * vy * (-3.0f32 * x2 + y2));
    x[26] = m3 * (2.890_611_4f32 * xy * vz);
    x[27] = m3 * (0.457_045_8f32 * vy * (1.0f32 - 5.0f32 * z2));
    x[28] = m3 * (0.373_176_33f32 * vz * (5.0f32 * z2 - 3.0f32));
    x[29] = m3 * (0.457_045_8f32 * vx * (1.0f32 - 5.0f32 * z2));
    x[30] = m3 * (1.445_305_7f32 * vz * (x2 - y2));
    x[31] = m3 * (0.590_043_6f32 * vx * (-x2 + 3.0f32 * y2));
}

/// Activated colour of one splat for unit view direction `v`. `row` is the
/// splat's first element in `coeffs` (`global_gid * NEURAL_ROW_COEFFS * 3`).
#[cube]
pub fn neural_color<P: Float>(
    coeffs: &Tensor<P>,
    row: u32,
    decoder: &Shared<[f32]>,
    v: Vec3A,
    active_degree: u32,
) -> Vec3A {
    let r = row as usize;
    let mut x = Array::<f32>::new(NEURAL_INPUTS as usize);
    #[unroll]
    for i in 0..NEURAL_LATENTS as usize {
        let f = f32::cast_from(coeffs[r + 3 + i]);
        x[2 * i] = f.sin();
        x[2 * i + 1] = f.cos();
    }
    write_direction_basis(&mut x, v, active_degree);

    let mut h1 = Array::<f32>::new(NEURAL_HIDDEN as usize);
    #[unroll]
    for j in 0..NEURAL_HIDDEN as usize {
        let w = NEURAL_W1 as usize + j * NEURAL_INPUTS as usize;
        let mut acc = 0.0f32;
        #[unroll]
        for k in 0..NEURAL_INPUTS as usize {
            acc += decoder[w + k] * x[k];
        }
        h1[j] = max(acc, 0.0f32);
    }
    let mut h2 = Array::<f32>::new(NEURAL_HIDDEN as usize);
    #[unroll]
    for j in 0..NEURAL_HIDDEN as usize {
        let w = NEURAL_W2 as usize + j * NEURAL_HIDDEN as usize;
        let mut acc = 0.0f32;
        #[unroll]
        for k in 0..NEURAL_HIDDEN as usize {
            acc += decoder[w + k] * h1[k];
        }
        h2[j] = max(acc, 0.0f32);
    }
    let mut o = Array::<f32>::new(3usize);
    #[unroll]
    for c in 0..3usize {
        let w = NEURAL_W3 as usize + c * NEURAL_HIDDEN as usize;
        let mut acc = 0.0f32;
        #[unroll]
        for k in 0..NEURAL_HIDDEN as usize {
            acc += decoder[w + k] * h2[k];
        }
        o[c] = acc;
    }

    // The base is stored as an SH DC coefficient, so standard viewers that
    // read only `f_dc` show the view-independent colour exactly.
    let br = f32::cast_from(coeffs[r]) * SH_C0;
    let bg = f32::cast_from(coeffs[r + 1]) * SH_C0;
    let bb = f32::cast_from(coeffs[r + 2]) * SH_C0;
    Vec3A::new(
        max(br + o[0].tanh() + 0.5f32, 0.0f32),
        max(bg + o[1].tanh() + 0.5f32, 0.0f32),
        max(bb + o[2].tanh() + 0.5f32, 0.0f32),
    )
}
