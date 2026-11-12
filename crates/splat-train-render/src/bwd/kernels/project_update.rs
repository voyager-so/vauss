//! Backward projection fused with the geometry/opacity Adam step.
//!
//! One thread per splat. A visible splat runs the projection VJP on its
//! rasterize gradients in-thread and feeds the result straight into Adam, so
//! the compact transforms/opacity gradients are never written or gathered.
//! Every splat steps (dense Adam): culled ones decay their moments with a zero
//! gradient, exactly as the unfused path does. Both moments of each value
//! live packed in one u32 (`geo_moments`); the update math is f32.

use super::geo_moments::adam_packed;
use super::project_backwards::splat_vjp;
use crate::kernels::camera_model::CameraModel;
use crate::kernels::sh::SH_C0;
use crate::kernels::types::{ProjectUniforms, Quat, Vec3A};
use brush_cube::is_finite_f32;
use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

#[cube]
fn zero_color_row(color_dir: &mut Tensor<f32>, row: u32) {
    for k in 0u32..6u32 {
        color_dir[(row * 6u32 + k) as usize] = 0.0f32;
    }
}

/// `color_dir` is compact (`[total + 1, 6]`: colour gradient then view
/// direction, row 0 zero) for the SH step's `grad_rows` gather, which
/// rebuilds the coefficient gradient `Y_k(dir) * colour` itself; rows past
/// the visible count are never gathered, so they are left unwritten. `v_refine_weight` and `coeffs_grad_sq` (the
/// mean square of the splat's SH gradient, the SH second-moment input) are
/// written per splat, zero for culled ones, so neither needs a gather or a
/// capacity-wide reduction.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_update_kernel<P: Float>(
    transforms: &mut Tensor<f32>,
    sh_coeffs: &Tensor<P>,
    raw_opac: &mut Tensor<f32>,
    min_scale: &Tensor<f32>,
    compact_from_global: &Tensor<u32>,
    v_rasterize_grads: &Tensor<f32>,
    w_t: &mut Tensor<u32>,
    lr_scale: &Tensor<f32>,
    w_o: &mut Tensor<u32>,
    color_dir: &mut Tensor<f32>,
    v_refine_weight: &mut Tensor<f32>,
    coeffs_grad_sq: &mut Tensor<f32>,
    beta1: f32,
    beta2: f32,
    inv1: f32,
    inv2: f32,
    epsilon: f32,
    lr_opac: f32,
    opac_reg: f32,
    round_seed: u32,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] has_min_scale: bool,
    #[comptime] sh_degree: u32,
    #[comptime] camera_model: CameraModel,
) {
    let g = ABSOLUTE_POS as u32;
    if g >= u.total_splats {
        terminate!();
    }
    if g == 0u32 {
        zero_color_row(color_dir, 0u32);
    }
    let mut refine = 0.0f32;
    let mut grad_sq = 0.0f32;

    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut g2 = 0.0f32;
    let mut g3 = 0.0f32;
    let mut g4 = 0.0f32;
    let mut g5 = 0.0f32;
    let mut g6 = 0.0f32;
    let mut g7 = 0.0f32;
    let mut g8 = 0.0f32;
    let mut g9 = 0.0f32;
    let mut g_opac = 0.0f32;

    let row = compact_from_global[g as usize];
    if row > 0u32 {
        let rg_base = ((row - 1u32) * 10u32) as usize;
        let mut any_grad = false;
        for k in 0u32..10u32 {
            any_grad = any_grad || v_rasterize_grads[rg_base + k as usize] != 0.0f32;
        }
        if any_grad {
            let out_base = row * 6u32;
            let tb = (g * 10u32) as usize;
            let (v_mean, v_q, v_scale, v_raw) = splat_vjp::<P>(
                Vec3A::new(transforms[tb], transforms[tb + 1], transforms[tb + 2]),
                Vec3A::new(
                    f32::exp(transforms[tb + 7] + u.log_scale_offset),
                    f32::exp(transforms[tb + 8] + u.log_scale_offset),
                    f32::exp(transforms[tb + 9] + u.log_scale_offset),
                ),
                Quat::new(
                    transforms[tb + 3],
                    transforms[tb + 4],
                    transforms[tb + 5],
                    transforms[tb + 6],
                ),
                raw_opac[g as usize],
                sh_coeffs,
                min_scale,
                g,
                v_rasterize_grads,
                rg_base,
                color_dir,
                out_base,
                u,
                mip_splatting,
                has_min_scale,
                sh_degree,
                // Neural appearance keeps the plain backward.
                false,
                camera_model,
                true,
            );
            g0 = v_mean.x();
            g1 = v_mean.y();
            g2 = v_mean.z();
            g3 = v_q.w();
            g4 = v_q.x();
            g5 = v_q.y();
            g6 = v_q.z();
            g7 = v_scale.x();
            g8 = v_scale.y();
            g9 = v_scale.z();
            g_opac = v_raw;
            let v_refine_in = v_rasterize_grads[rg_base + 9];
            let refine_clean = select(is_finite_f32(v_refine_in), v_refine_in, 0.0f32);
            refine = clamp(refine_clean, 0.0f32, 1.0e32f32);
            // Mean square of the SH gradient row, from its DC entries
            // (`SH_C0 * colour`) exactly as the unfused backward does
            // (real SH: sum_k Y_k^2 = (d+1)^2 / 4pi).
            let c0 = SH_C0 * color_dir[out_base as usize];
            let c1 = SH_C0 * color_dir[(out_base + 1u32) as usize];
            let c2 = SH_C0 * color_dir[(out_base + 2u32) as usize];
            grad_sq = (c0 * c0 + c1 * c1 + c2 * c2) / 3.0f32;
        } else {
            zero_color_row(color_dir, row);
        }
    }
    v_refine_weight[g as usize] = refine;
    coeffs_grad_sq[g as usize] = grad_sq;

    // The VJP above read this splat's pre-step parameters; update them now.
    let tb = (g * 10u32) as usize;
    // Opacity regulariser (3DGS-MCMC, Kheradmand et al. 2024) on every splat
    // (culled ones too), its
    // `w · mean(opacity)`: d/d raw = (w/N) · o(1 - o). Pushes unneeded splats
    // towards zero opacity, where refine relocates them.
    let o = 1.0f32 / (1.0f32 + f32::exp(-raw_opac[g as usize]));
    g_opac += opac_reg * o * (1.0f32 - o);
    // Stochastic-rounding streams; the high byte decorrelates transforms
    // from opacity, which share element indices.
    let seed_t = round_seed ^ (10u32 << 24u32);
    let seed_o = round_seed ^ (1u32 << 24u32);
    adam_packed(transforms, w_t, tb, g0, lr_scale[0], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 1, g1, lr_scale[1], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 2, g2, lr_scale[2], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 3, g3, lr_scale[3], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 4, g4, lr_scale[4], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 5, g5, lr_scale[5], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 6, g6, lr_scale[6], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 7, g7, lr_scale[7], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 8, g8, lr_scale[8], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(transforms, w_t, tb + 9, g9, lr_scale[9], beta1, beta2, inv1, inv2, epsilon, seed_t);
    adam_packed(raw_opac, w_o, g as usize, g_opac, lr_opac, beta1, beta2, inv1, inv2, epsilon, seed_o);
}
