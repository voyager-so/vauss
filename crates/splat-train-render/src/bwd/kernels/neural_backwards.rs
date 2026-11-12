//! Derived from nerficg-project/efficient-gaussian-appearance at 4efbe67, Copyright 2026 The Efficient
//! Gaussian Appearance Authors, Apache-2.0 (see NOTICE);
//! upstream file
//! FasterGSVDACudaBackend/FasterGSVDACudaBackend/rasterization/rtc/appearance_neural.cuh.
//!
//! Backward of the compact neural appearance (see `kernels::neural`).
//!
//! One thread per compact row (row 0 is the zero row, visible splat
//! `compact_gid` owns row `compact_gid + 1`), a fixed grid striding over the
//! rows. Each splat recomputes its decoder activations from the colour
//! gradient the rasterizer left in `v_rasterize_grads`, then writes:
//! - its row gradient `[base(3, DC units), latent(8), 0]` to `v_coeffs`,
//!   and the row's mean square to `v_coeffs_sq` (Adam's reduced second
//!   moment input);
//! - the view-direction chain rule onto its mean, added to the mean gradient
//!   `project_backwards` already wrote in `v_transforms`;
//! - its share of the shared decoder gradient. Lanes sum the 816 weight
//!   products with a 32-lane butterfly reduce-scatter (each lane ends up
//!   owning one weight of each 32-weight group), keep running sums in
//!   registers across the grid stride, then the subgroups of a cube add into
//!   shared memory and each cube writes one `[816]` partial row. The host sums
//!   the partials, so no per-splat weight gradient is ever materialized and
//!   no float atomics are needed.
//!
//! Requires exactly 32-wide subgroups.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

use crate::kernels::neural::{
    NEURAL_DECODER_LEN, NEURAL_HIDDEN, NEURAL_INPUTS, NEURAL_LATENTS, NEURAL_ROW_COEFFS,
    NEURAL_W1, NEURAL_W2, NEURAL_W3, stage_decoder, write_direction_basis,
};
use crate::kernels::sh::SH_C0;
use crate::kernels::types::{ProjectUniforms, Vec3A};

pub const WG_SIZE: u32 = 64;
/// Floats staged per row for the weight-gradient tiles (pass A: x | d1,
/// pass B: h1 | d2 | h2).
const STAGE: u32 = 48;
/// W1 weights per thread (512 / 128) and W2 weights per thread (256 / 128).
const W1_PER: u32 = NEURAL_HIDDEN * NEURAL_INPUTS / WG_SIZE;
const W2_PER: u32 = NEURAL_HIDDEN * NEURAL_HIDDEN / WG_SIZE;
const ROW: u32 = NEURAL_ROW_COEFFS * 3;

/// Butterfly reduce-scatter of 32 per-lane values across a 32-lane
/// subgroup: lane `l` ends with the subgroup sum of component `l` in `v[0]`.
#[cube]
fn reduce_scatter_32x32(v: &mut Array<f32>) {
    let lane = UNIT_POS_PLANE;
    let hi = (lane & 16u32) != 0u32;
    #[unroll]
    for i in 0..16usize {
        let keep = select(hi, v[i + 16], v[i]);
        let send = select(hi, v[i], v[i + 16]);
        v[i] = keep + plane_shuffle_xor(send, 16u32);
    }
    let hi = (lane & 8u32) != 0u32;
    #[unroll]
    for i in 0..8usize {
        let keep = select(hi, v[i + 8], v[i]);
        let send = select(hi, v[i], v[i + 8]);
        v[i] = keep + plane_shuffle_xor(send, 8u32);
    }
    let hi = (lane & 4u32) != 0u32;
    #[unroll]
    for i in 0..4usize {
        let keep = select(hi, v[i + 4], v[i]);
        let send = select(hi, v[i], v[i + 4]);
        v[i] = keep + plane_shuffle_xor(send, 4u32);
    }
    let hi = (lane & 2u32) != 0u32;
    #[unroll]
    for i in 0..2usize {
        let keep = select(hi, v[i + 2], v[i]);
        let send = select(hi, v[i], v[i + 2]);
        v[i] = keep + plane_shuffle_xor(send, 2u32);
    }
    let hi = (lane & 1u32) != 0u32;
    let keep = select(hi, v[1], v[0]);
    let send = select(hi, v[0], v[1]);
    v[0] = keep + plane_shuffle_xor(send, 1u32);
}

/// Gradient of the direction basis (bands 1..3, gated like the forward)
/// w.r.t. the unit direction, given the input adjoint `dx[16..32]`.
#[cube]
fn direction_basis_vjp(dx: &Array<f32>, v: Vec3A, active_degree: u32) -> Vec3A {
    let (x, y, z) = (v.x(), v.y(), v.z());
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    let mut gz = 0.0f32;
    if active_degree >= 1u32 {
        let c1 = 0.488_602_5f32;
        gy += -c1 * dx[17];
        gz += c1 * dx[18];
        gx += -c1 * dx[19];
    }
    if active_degree >= 2u32 {
        let a = 1.092_548_4f32;
        gx += a * y * dx[20];
        gy += a * x * dx[20];
        gy += -a * z * dx[21];
        gz += -a * y * dx[21];
        gz += 1.892_349_4f32 * z * dx[22];
        gx += -a * z * dx[23];
        gz += -a * x * dx[23];
        gx += a * x * dx[24];
        gy += -a * y * dx[24];
    }
    if active_degree >= 3u32 {
        let (x2, y2, z2) = (x * x, y * y, z * z);
        let b = 1.770_130_8f32;
        let c = 2.890_611_4f32;
        let e = 0.457_045_8f32;
        let f = 4.570_458f32;
        gx += -3.540_261_5f32 * x * y * dx[25];
        gy += (b * y2 - b * x2) * dx[25];
        gx += c * y * z * dx[26];
        gy += c * x * z * dx[26];
        gz += c * x * y * dx[26];
        gy += (e - 2.285_229f32 * z2) * dx[27];
        gz += -f * y * z * dx[27];
        gz += (5.597_645f32 * z2 - 1.119_529f32) * dx[28];
        gx += (e - 2.285_229f32 * z2) * dx[29];
        gz += -f * x * z * dx[29];
        gx += c * x * z * dx[30];
        gy += -c * y * z * dx[30];
        gz += 1.445_305_7f32 * (x2 - y2) * dx[30];
        gx += (b * y2 - b * x2) * dx[31];
        gy += 3.540_261_5f32 * x * y * dx[31];
    }
    Vec3A::new(gx, gy, gz)
}

/// One row's forward recompute and backward. All per-row arrays live in
/// this function only (like the forward's `neural_color`), so none of them
/// is live across the cube-wide passes; the activations the decoder gradient
/// needs are written straight to this thread's shared staging slots.
#[cube]
#[allow(clippy::too_many_arguments)]
fn row_backward<P: Float>(
    transforms: &Tensor<f32>,
    coeffs: &Tensor<P>,
    decoder: &Shared<[f32]>,
    gid: u32,
    row: u32,
    vc: Vec3A,
    v_transforms: &mut Tensor<f32>,
    v_coeffs: &mut Tensor<f32>,
    v_coeffs_sq: &mut Tensor<f32>,
    stage_a: &mut Shared<[f32]>,
    stage_b: &mut Shared<[f32]>,
    stage_go: &mut Shared<[f32]>,
    sa: u32,
    sgo: u32,
    u: ProjectUniforms,
    active_degree: u32,
) {
    let out = (row * ROW) as usize;
    let tb = (gid * 10u32) as usize;
    let mean = Vec3A::new(transforms[tb], transforms[tb + 1], transforms[tb + 2]);
    let u_world = mean.sub(u.camera_pos());
    let u_len = u_world.length();
    let dir = u_world.scale(1.0f32 / u_len);
    let r = (gid * ROW) as usize;
    let sa = sa as usize;

    let mut x = Array::<f32>::new(NEURAL_INPUTS as usize);
    #[unroll]
    for i in 0..NEURAL_LATENTS as usize {
        let f = f32::cast_from(coeffs[r + 3 + i]);
        x[2 * i] = f.sin();
        x[2 * i + 1] = f.cos();
    }
    write_direction_basis(&mut x, dir, active_degree);
    #[unroll]
    for k in 0..NEURAL_INPUTS as usize {
        stage_a[sa + k] = x[k];
    }
    let mut h1 = Array::<f32>::new(NEURAL_HIDDEN as usize);
    #[unroll]
    for j in 0..NEURAL_HIDDEN as usize {
        let w = NEURAL_W1 as usize + j * NEURAL_INPUTS as usize;
        let mut a = 0.0f32;
        #[unroll]
        for k in 0..NEURAL_INPUTS as usize {
            a += decoder[w + k] * x[k];
        }
        h1[j] = max(a, 0.0f32);
        stage_b[sa + j] = h1[j];
    }
    let mut h2 = Array::<f32>::new(NEURAL_HIDDEN as usize);
    #[unroll]
    for j in 0..NEURAL_HIDDEN as usize {
        let w = NEURAL_W2 as usize + j * NEURAL_HIDDEN as usize;
        let mut a = 0.0f32;
        #[unroll]
        for k in 0..NEURAL_HIDDEN as usize {
            a += decoder[w + k] * h1[k];
        }
        h2[j] = max(a, 0.0f32);
        stage_b[sa + 32 + j] = h2[j];
    }
    // Sum of squares of the row gradient, for Adam's per-row second moment.
    let mut sq = 0.0f32;
    let mut go = Array::<f32>::new(3usize);
    let mut gc = Array::<f32>::new(3usize);
    gc[0] = vc.x();
    gc[1] = vc.y();
    gc[2] = vc.z();
    #[unroll]
    for c in 0..3usize {
        let w = NEURAL_W3 as usize + c * NEURAL_HIDDEN as usize;
        let mut a = 0.0f32;
        #[unroll]
        for k in 0..NEURAL_HIDDEN as usize {
            a += decoder[w + k] * h2[k];
        }
        let t = a.tanh();
        // Colour = ReLU(base + tanh(o) + 0.5).
        let base = f32::cast_from(coeffs[r + c]) * SH_C0;
        let g = select(base + t + 0.5f32 > 0.0f32, gc[c], 0.0f32);
        v_coeffs[out + c] = g * SH_C0;
        sq += (g * SH_C0) * (g * SH_C0);
        go[c] = g * (1.0f32 - t * t);
        stage_go[sgo as usize + c] = go[c];
    }

    // Hidden adjoints through the ReLUs.
    let mut d2 = Array::<f32>::new(NEURAL_HIDDEN as usize);
    #[unroll]
    for k in 0..NEURAL_HIDDEN as usize {
        let mut a = 0.0f32;
        #[unroll]
        for c in 0..3usize {
            a += decoder[NEURAL_W3 as usize + c * NEURAL_HIDDEN as usize + k] * go[c];
        }
        d2[k] = select(h2[k] > 0.0f32, a, 0.0f32);
        stage_b[sa + 16 + k] = d2[k];
    }
    let mut d1 = Array::<f32>::new(NEURAL_HIDDEN as usize);
    #[unroll]
    for k in 0..NEURAL_HIDDEN as usize {
        let mut a = 0.0f32;
        #[unroll]
        for j in 0..NEURAL_HIDDEN as usize {
            a += decoder[NEURAL_W2 as usize + j * NEURAL_HIDDEN as usize + k] * d2[j];
        }
        d1[k] = select(h1[k] > 0.0f32, a, 0.0f32);
        stage_a[sa + NEURAL_INPUTS as usize + k] = d1[k];
    }
    let mut dx = Array::<f32>::new(NEURAL_INPUTS as usize);
    #[unroll]
    for k in 0..NEURAL_INPUTS as usize {
        let mut a = 0.0f32;
        #[unroll]
        for j in 0..NEURAL_HIDDEN as usize {
            a += decoder[NEURAL_W1 as usize + j * NEURAL_INPUTS as usize + k] * d1[j];
        }
        dx[k] = a;
    }

    // Latents through the sin/cos encoding.
    #[unroll]
    for i in 0..NEURAL_LATENTS as usize {
        let g = dx[2 * i] * x[2 * i + 1] - dx[2 * i + 1] * x[2 * i];
        v_coeffs[out + 3 + i] = g;
        sq += g * g;
    }
    v_coeffs[out + 11] = 0.0f32;
    v_coeffs_sq[row as usize] = sq / (ROW as f32);

    // View direction onto the mean: d(normalize(w))/dw = (I - d d^T) / |w|.
    let g_dir = direction_basis_vjp(&dx, dir, active_degree);
    let v_mean = g_dir.sub(dir.scale(dir.dot(g_dir))).scale(1.0f32 / u_len);
    let vb = (row * 10u32) as usize;
    v_transforms[vb] += v_mean.x();
    v_transforms[vb + 1] += v_mean.y();
    v_transforms[vb + 2] += v_mean.z();
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn neural_backwards_kernel<P: Float>(
    transforms: &Tensor<f32>,
    coeffs: &Tensor<P>,
    decoder_src: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    num_visible_gpu: &Tensor<u32>,
    v_rasterize_grads: &Tensor<f32>,
    v_transforms: &mut Tensor<f32>,
    v_coeffs: &mut Tensor<f32>,
    v_coeffs_sq: &mut Tensor<f32>,
    v_decoder_partials: &mut Tensor<f32>,
    u: ProjectUniforms,
    iters: u32,
) {
    let mut decoder = Shared::<[f32]>::new_slice(NEURAL_DECODER_LEN as usize);
    stage_decoder(decoder_src, &mut decoder);
    let t_id = UNIT_POS;
    let stride = (CUBE_COUNT as u32) * WG_SIZE;
    let num_rows = u.total_splats + 1u32;
    let num_visible = num_visible_gpu[0];
    let active_degree = u.sh_degree;

    // The decoder gradient is three small matrix products over rows: each
    // batch of WG_SIZE rows stages its activations in shared memory (pass A:
    // x | d1, pass B: h1 | d2 | h2, plus go) and every thread accumulates
    // the weights it owns over the batch in registers.
    let mut stage_a = Shared::<[f32]>::new_slice((WG_SIZE * STAGE) as usize);
    let mut stage_b = Shared::<[f32]>::new_slice((WG_SIZE * STAGE) as usize);
    let mut stage_go = Shared::<[f32]>::new_slice((WG_SIZE * 3u32) as usize);
    let mut acc1 = Array::<f32>::new(W1_PER as usize);
    let mut acc2 = Array::<f32>::new(W2_PER as usize);
    #[unroll]
    for i in 0..W1_PER as usize {
        acc1[i] = 0.0f32;
    }
    #[unroll]
    for i in 0..W2_PER as usize {
        acc2[i] = 0.0f32;
    }
    let mut acc3 = 0.0f32;
    let sa = t_id * STAGE;
    let sgo = t_id * 3u32;

    let mut row = (CUBE_POS as u32) * WG_SIZE + UNIT_POS;
    // Only rows 0..=num_visible are ever read (culled splats gather row 0).
    let live_iters = min(iters, (num_visible + stride) / stride);
    for _ in 0..live_iters {
        // Rows without a colour gradient stage zeros.
        #[unroll]
        for k in 0..STAGE as usize {
            stage_a[sa as usize + k] = 0.0f32;
            stage_b[sa as usize + k] = 0.0f32;
        }
        #[unroll]
        for c in 0..3usize {
            stage_go[sgo as usize + c] = 0.0f32;
        }
        if row < num_rows && row <= num_visible {
            let mut wrote = false;
            if row > 0u32 {
                let compact_gid = row - 1u32;
                let rg = (compact_gid * 10u32) as usize;
                let vc = Vec3A::new(
                    v_rasterize_grads[rg + 5],
                    v_rasterize_grads[rg + 6],
                    v_rasterize_grads[rg + 7],
                );
                if vc.x() != 0.0f32 || vc.y() != 0.0f32 || vc.z() != 0.0f32 {
                    wrote = true;
                    let gid = global_from_compact_gid[compact_gid as usize];
                    row_backward(
                        transforms, coeffs, &decoder, gid, row, vc, v_transforms, v_coeffs,
                        v_coeffs_sq, &mut stage_a, &mut stage_b, &mut stage_go, sa, sgo, u, active_degree,
                    );
                }
            }
            if !wrote {
                let out = (row * ROW) as usize;
                #[unroll]
                for k in 0..ROW as usize {
                    v_coeffs[out + k] = 0.0f32;
                }
                v_coeffs_sq[row as usize] = 0.0f32;
            }
        }
        sync_cube();
        // Pass A: dW1[j][k] += d1[j] * x[k]. A thread's weights share k
        // (WG_SIZE is a multiple of the row length), so x[k] is loaded once
        // per row; each weight still sums the batch in row order.
        let mut a1 = Array::<f32>::new(W1_PER as usize);
        #[unroll]
        for i in 0..W1_PER as usize {
            a1[i] = 0.0f32;
        }
        let k1 = t_id % NEURAL_INPUTS;
        for r in 0..WG_SIZE {
            let rb = r * STAGE;
            let xk = stage_a[(rb + k1) as usize];
            #[unroll]
            for i in 0..W1_PER {
                let j = (t_id + i * WG_SIZE) / NEURAL_INPUTS;
                a1[i as usize] += stage_a[(rb + NEURAL_INPUTS + j) as usize] * xk;
            }
        }
        #[unroll]
        for i in 0..W1_PER as usize {
            acc1[i] += a1[i];
        }
        // Pass B: dW2[j][k] += d2[j] * h1[k] (h1[k] shared the same way),
        // dW3[c][k] += go[c] * h2[k].
        let mut a2 = Array::<f32>::new(W2_PER as usize);
        #[unroll]
        for i in 0..W2_PER as usize {
            a2[i] = 0.0f32;
        }
        let k2 = t_id % NEURAL_HIDDEN;
        for r in 0..WG_SIZE {
            let rb = r * STAGE;
            let hk = stage_b[(rb + k2) as usize];
            #[unroll]
            for i in 0..W2_PER {
                let j = (t_id + i * WG_SIZE) / NEURAL_HIDDEN;
                a2[i as usize] += stage_b[(rb + 16u32 + j) as usize] * hk;
            }
        }
        #[unroll]
        for i in 0..W2_PER as usize {
            acc2[i] += a2[i];
        }
        if t_id < 3u32 * NEURAL_HIDDEN {
            let c = t_id / NEURAL_HIDDEN;
            let k = t_id % NEURAL_HIDDEN;
            let mut a = 0.0f32;
            for r in 0..WG_SIZE {
                a += stage_go[(r * 3u32 + c) as usize] * stage_b[(r * STAGE + 32u32 + k) as usize];
            }
            acc3 += a;
        }
        sync_cube();
        row += stride;
    }

    let base = (CUBE_POS as u32) * NEURAL_DECODER_LEN;
    #[unroll]
    for i in 0..W1_PER {
        let w = t_id + i * WG_SIZE;
        v_decoder_partials[(base + NEURAL_W1 + w) as usize] = acc1[i as usize];
    }
    #[unroll]
    for i in 0..W2_PER {
        let w = t_id + i * WG_SIZE;
        v_decoder_partials[(base + NEURAL_W2 + w) as usize] = acc2[i as usize];
    }
    if t_id < 3u32 * NEURAL_HIDDEN {
        v_decoder_partials[(base + NEURAL_W3 + t_id) as usize] = acc3;
    }
}
