//! Backward projection.

use crate::kernels::camera_model::CameraModel;
use crate::kernels::camera_model::{calculate_project_jacobian, calculate_projection_vjp};
use crate::kernels::helpers::{
    apply_scale_floor, calc_cov2d, compensate_cov2d, read_quat_unorm, read_scale, world_to_cam,
};
use crate::kernels::sh::{num_sh_coeffs, sh_coeffs_to_color_vjp, sh_color_viewdir_vjp};
use crate::kernels::types::{Mat3, ProjectUniforms, Quat, Sym2, Vec3A};
use super::rasterize_backwards_planes::GRADS_SURFEL;
use brush_cube::{Vec2, is_finite_f32, sigmoid};
use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

/// Apply the VJP of `q -> q / |q|` to a downstream quaternion gradient.
#[cube]
fn apply_normalize_vjp(q: Quat, g: Quat) -> Quat {
    let lsq = q.dot(q);
    let l = f32::sqrt(lsq);
    let inv = 1.0f32 / (l * lsq);
    let qw = q.w();
    let qx = q.x();
    let qy = q.y();
    let qz = q.z();
    let gw = g.w();
    let gx = g.x();
    let gy = g.y();
    let gz = g.z();
    // Quat stored as `(w, x, y, z)`:
    //   cross_complex = -((w,x,y) * (x,y,w)) = (-w*x, -x*y, -y*w)
    //   cross_scalar  = -((w,x,y) * z)       = (-w*z, -x*z, -y*z)
    let cc0 = -qw * qx;
    let cc1 = -qx * qy;
    let cc2 = -qy * qw;
    let cs0 = -qw * qz;
    let cs1 = -qx * qz;
    let cs2 = -qy * qz;
    let q_sqr_w = qw * qw;
    let q_sqr_x = qx * qx;
    let q_sqr_y = qy * qy;
    let q_sqr_z = qz * qz;
    Quat::new(
        ((lsq - q_sqr_w) * gw + cc0 * gx + cc2 * gy + cs0 * gz) * inv,
        (cc0 * gw + (lsq - q_sqr_x) * gx + cc1 * gy + cs1 * gz) * inv,
        (cc2 * gw + cc1 * gx + (lsq - q_sqr_y) * gy + cs2 * gz) * inv,
        (cs0 * gw + cs1 * gx + cs2 * gy + (lsq - q_sqr_z) * gz) * inv,
    )
}

/// VJP of `quat_to_mat`. `v_r` is column-major like `quat_to_mat`'s output.
#[cube]
fn quat_to_mat_vjp(q: Quat, v_r: Mat3) -> Quat {
    let qw = q.w();
    let qx = q.x();
    let qy = q.y();
    let qz = q.z();
    let w_grad =
        qx * (v_r.c1_z - v_r.c2_y) + qy * (v_r.c2_x - v_r.c0_z) + qz * (v_r.c0_y - v_r.c1_x);
    let x_grad = -2.0f32 * qx * (v_r.c1_y + v_r.c2_z)
        + qy * (v_r.c0_y + v_r.c1_x)
        + qz * (v_r.c0_z + v_r.c2_x)
        + qw * (v_r.c1_z - v_r.c2_y);
    let y_grad = qx * (v_r.c0_y + v_r.c1_x) - 2.0f32 * qy * (v_r.c0_x + v_r.c2_z)
        + qz * (v_r.c1_z + v_r.c2_y)
        + qw * (v_r.c2_x - v_r.c0_z);
    let z_grad = qx * (v_r.c0_z + v_r.c2_x) + qy * (v_r.c1_z + v_r.c2_y)
        - 2.0f32 * qz * (v_r.c0_x + v_r.c1_y)
        + qw * (v_r.c0_y - v_r.c1_x);
    Quat::new(
        2.0f32 * w_grad,
        2.0f32 * x_grad,
        2.0f32 * y_grad,
        2.0f32 * z_grad,
    )
}

/// VJP of `Minv = inverse(M)` for symmetric 2x2 matrices.
///
/// Returns the gradient w.r.t. `M` as a `Sym2` (the upstream grad is
/// also symmetric since the rasterize backward writes the conic grad
/// in symmetric form).
#[cube]
fn inverse2x2_vjp(minv: Sym2, v_minv: Sym2) -> Sym2 {
    // -P * dP/dP * P. Writing both as symmetric Sym2 keeps the math at
    // 5 muladd-pairs instead of expanding to 4x4 dense.
    let tmp00 = -minv.c00 * v_minv.c00 + -minv.c01 * v_minv.c01;
    let tmp01 = -minv.c01 * v_minv.c00 + -minv.c11 * v_minv.c01;
    let tmp10 = -minv.c00 * v_minv.c01 + -minv.c01 * v_minv.c11;
    let tmp11 = -minv.c01 * v_minv.c01 + -minv.c11 * v_minv.c11;
    Sym2 {
        c00: tmp00 * minv.c00 + tmp10 * minv.c01,
        c01: tmp01 * minv.c00 + tmp11 * minv.c01,
        c11: tmp01 * minv.c01 + tmp11 * minv.c11,
    }
}

/// Zero one row of the compact gradient outputs.
#[cube]
fn write_zero_row(
    v_transforms: &mut Tensor<f32>,
    v_coeffs: &mut Tensor<f32>,
    v_raw_opac: &mut Tensor<f32>,
    v_refine_weight: &mut Tensor<f32>,
    row: u32,
    #[comptime] sh_degree: u32,
    #[comptime] neural: bool,
) {
    let vbase = (row * 10u32) as usize;
    for k in 0u32..10u32 {
        v_transforms[vbase + k as usize] = 0.0f32;
    }
    // Neural rows are written by `neural_backwards`.
    if comptime![!neural] {
        let coeff_base = row * comptime![num_sh_coeffs(sh_degree) * 3u32];
        for k in 0u32..comptime![num_sh_coeffs(sh_degree) * 3u32] {
            v_coeffs[(coeff_base + k) as usize] = 0.0f32;
        }
    }
    v_raw_opac[row as usize] = 0.0f32;
    v_refine_weight[row as usize] = 0.0f32;
}

/// Projection VJP for one visible splat: reads its rasterize gradients at
/// `rg_base`, writes the SH coefficient gradient at `coeff_out_base` and
/// returns the mean, quaternion and log-scale gradients plus the raw-opacity
/// gradient. Shared by the plain backward and the fused update.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn splat_vjp<P: Float>(
    // The splat's parameters, read by the caller (the fused update holds
    // the parameter buffers mutably).
    mean: Vec3A,
    scale_raw: Vec3A,
    quat_unorm: Quat,
    raw_opac: f32,
    sh_coeffs: &Tensor<P>,
    min_scale: &Tensor<f32>,
    global_gid: u32,
    v_rasterize_grads: &Tensor<f32>,
    rg_base: usize,
    v_coeffs: &mut Tensor<f32>,
    coeff_out_base: u32,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] has_min_scale: bool,
    #[comptime] sh_degree: u32,
    #[comptime] neural: bool,
    #[comptime] camera_model: CameraModel,
    // Write `[v_color, view dir]` (6 floats) at `coeff_out_base` instead of
    // the 3(d+1)^2 coefficient gradients; the SH step rebuilds those.
    #[comptime] color_dir: bool,
) -> (Vec3A, Quat, Vec3A, f32) {
    let v_mean2d_x = v_rasterize_grads[rg_base];
    let v_mean2d_y = v_rasterize_grads[rg_base + 1];
    let v_conics_x = v_rasterize_grads[rg_base + 2];
    let v_conics_y = v_rasterize_grads[rg_base + 3];
    let v_conics_z = v_rasterize_grads[rg_base + 4];
    let v_color_r = v_rasterize_grads[rg_base + 5];
    let v_color_g = v_rasterize_grads[rg_base + 6];
    let v_color_b = v_rasterize_grads[rg_base + 7];
    let v_alpha_in = v_rasterize_grads[rg_base + 8];

    let quat = quat_unorm.normalize();

    // Mip-Splatting floor, matching the forward.
    let opac_sig = sigmoid(raw_opac);
    let floor = apply_scale_floor(scale_raw, opac_sig, min_scale, global_gid, has_min_scale);
    let scale = floor.scale;
    let ratio_sq = floor.ratio_sq;

    // viewdir + SH VJP. d(normalize(u))/du = (I - vv^T)/|u|, so
    // v_u = (v_v - v * (v · v_v)) / |u|.
    // Neural appearance's colour and view-direction gradients come from
    // `neural_backwards`, which adds its direction term to the mean after;
    // it never takes the fused (`color_dir`) path.
    let v_mean_from_sh = if comptime![neural] {
        Vec3A::new(0.0f32, 0.0f32, 0.0f32)
    } else {
        let u_world = mean.sub(u.camera_pos());
        let u_len = u_world.length();
        let v = u_world.scale(1.0f32 / u_len);
        let coeff_base = global_gid * comptime![num_sh_coeffs(sh_degree) * 3u32];
        let v_color = Vec3A::new(v_color_r, v_color_g, v_color_b);
        if comptime![color_dir] {
            let o = coeff_out_base as usize;
            v_coeffs[o] = v_color_r;
            v_coeffs[o + 1] = v_color_g;
            v_coeffs[o + 2] = v_color_b;
            v_coeffs[o + 3] = v.x();
            v_coeffs[o + 4] = v.y();
            v_coeffs[o + 5] = v.z();
        } else {
            sh_coeffs_to_color_vjp(v_coeffs, coeff_out_base, sh_degree, v, v_color);
        }
        let v_v_sh = sh_color_viewdir_vjp(sh_coeffs, coeff_base, sh_degree, v, v_color);
        let v_dot_vv = v.dot(v_v_sh);
        v_v_sh.sub(v.scale(v_dot_vv)).scale(1.0f32 / u_len)
    };

    let mean_c = world_to_cam(mean, u);

    let r = quat.to_mat3();
    let m = r.mul_diag(scale);

    let raw_cov = calc_cov2d(scale, quat, mean_c, u, camera_model);
    let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    // Gradient w.r.t. the (compensated, clamped) opacity, zero if clamped.
    let v_opac_base = select(floor.opac_open, filter_comp * v_alpha_in, 0.0f32);
    let conic_inv = cov.inverse();
    let v_inv = Sym2 {
        c00: v_conics_x,
        c01: v_conics_y * 0.5f32,
        c11: v_conics_z,
    };
    let v_cov2d = inverse2x2_vjp(conic_inv, v_inv);

    // covar = M * M^T (symmetric).
    let covar = m.outer_product_self();

    // covar_c = R_cam * covar * R_cam^T (symmetric).
    let view_rot = u.view_rotation();
    let cov_c = covar.congruence(view_rot);

    let cam_jac = calculate_project_jacobian(
        mean_c,
        u.jacobian_clamp_limits,
        u.pinhole_params,
        camera_model,
    );
    let v_mean_c = calculate_projection_vjp(
        cam_jac,
        mean_c,
        cov_c,
        u,
        v_cov2d,
        Vec2::new(v_mean2d_x, v_mean2d_y),
        camera_model,
    );

    // v_covar_c = J^T * v_cov2d * J (2x2 sym → 3x3 sym).
    let vcc = cam_jac.transpose_congruence_sym2(v_cov2d);

    let v_mean = view_rot.transpose_mul_vec3(v_mean_c).add(v_mean_from_sh);

    // v_covar = R^T * v_covar_c * R (symmetric).
    // v_M = (v_covar + v_covar^T) * M = 2 * v_covar * M.
    let v_m = vcc.transpose_congruence(view_rot).scale(2.0f32).mul_mat3(m);

    // v_scale = (R[i] dot v_M[i]) * exp(log_scale), i.e. the gradient w.r.t.
    // the log of the (floored) scale. Chain through the floor with
    // `ratio_sq`, and add the opacity compensation's dependence on the raw
    // scales: d(opac_base)/d(log s_i) = opac_base * (1 - ratio_sq_i).
    let v_log_floored = Vec3A::new(
        r.col0().dot(v_m.col0()) * scale.x(),
        r.col1().dot(v_m.col1()) * scale.y(),
        r.col2().dot(v_m.col2()) * scale.z(),
    );
    let v_scale_exp = Vec3A::new(
        v_log_floored.x() * ratio_sq.x() + v_opac_base * floor.opac * (1.0f32 - ratio_sq.x()),
        v_log_floored.y() * ratio_sq.y() + v_opac_base * floor.opac * (1.0f32 - ratio_sq.y()),
        v_log_floored.z() * ratio_sq.z() + v_opac_base * floor.opac * (1.0f32 - ratio_sq.z()),
    );

    // grad for quat from covar: v_quat = normalize_vjp(quat) *
    // quat_to_mat_vjp(quat, v_M * diag(scale)).
    let q_grad = quat_to_mat_vjp(quat, v_m.mul_diag(scale));
    let v_q = apply_normalize_vjp(quat_unorm, q_grad);

    let v_raw = v_opac_base * floor.coef * opac_sig * (1.0f32 - opac_sig);
    (v_mean, v_q, v_scale_exp, v_raw)
}

/// Outputs are compact with a zero row in front: row 0 is what culled splats
/// gather through `compact_from_global`, visible splat `compact_gid` owns row
/// `compact_gid + 1`. Capacity extends to total_splats; inactive rows are zeroed
/// before any compact-map or raster-gradient read, so reductions remain defined.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_backwards_kernel<P: Float>(
    transforms: &Tensor<f32>,
    sh_coeffs: &Tensor<P>,
    raw_opac: &Tensor<f32>,
    min_scale: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    num_visible_gpu: &Tensor<u32>,
    v_rasterize_grads: &Tensor<f32>,
    v_transforms: &mut Tensor<f32>,
    v_coeffs: &mut Tensor<f32>,
    v_raw_opac: &mut Tensor<f32>,
    v_refine_weight: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] has_min_scale: bool,
    #[comptime] sh_degree: u32,
    #[comptime] neural: bool,
    #[comptime] camera_model: CameraModel,
    #[comptime] surfel: bool,
) {
    let row = ABSOLUTE_POS as u32;
    if row > u.total_splats {
        terminate!();
    }
    // Rows past the visible count are never read (culled splats gather
    // row 0), so they are left unwritten rather than zero-filled.
    if row > num_visible_gpu[0] {
        terminate!();
    }
    if row == 0u32 {
        write_zero_row(
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
            row,
            sh_degree,
            neural,
        );
        terminate!();
    }
    let compact_gid = row - 1u32;
    let global_gid = global_from_compact_gid[compact_gid as usize];
    if comptime![surfel] {
        surfel_backwards(
            transforms,
            sh_coeffs,
            raw_opac,
            min_scale,
            global_gid,
            compact_gid,
            row,
            v_rasterize_grads,
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
            u,
            has_min_scale,
            sh_degree,
        );
        terminate!();
    }

    // Read upstream rasterize grads first. rasterize_bwd only writes for
    // splats that contributed to a pixel; the rest write their zero row.
    let rg_base = (compact_gid * 10u32) as usize;
    let v_mean2d_x = v_rasterize_grads[rg_base];
    let v_mean2d_y = v_rasterize_grads[rg_base + 1];
    let v_conics_x = v_rasterize_grads[rg_base + 2];
    let v_conics_y = v_rasterize_grads[rg_base + 3];
    let v_conics_z = v_rasterize_grads[rg_base + 4];
    let v_color_r = v_rasterize_grads[rg_base + 5];
    let v_color_g = v_rasterize_grads[rg_base + 6];
    let v_color_b = v_rasterize_grads[rg_base + 7];
    let v_alpha_in = v_rasterize_grads[rg_base + 8];
    let v_refine_in = v_rasterize_grads[rg_base + 9];

    let any_grad = v_mean2d_x != 0.0f32
        || v_mean2d_y != 0.0f32
        || v_conics_x != 0.0f32
        || v_conics_y != 0.0f32
        || v_conics_z != 0.0f32
        || v_color_r != 0.0f32
        || v_color_g != 0.0f32
        || v_color_b != 0.0f32
        || v_alpha_in != 0.0f32
        || v_refine_in != 0.0f32;

    if !any_grad {
        write_zero_row(
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
            row,
            sh_degree,
            neural,
        );
        terminate!();
    }
    let vbase = (row * 10u32) as usize;
    let coeff_out_base = row * comptime![num_sh_coeffs(sh_degree) * 3u32];
    let tbase = (global_gid * 10u32) as usize;
    let (v_mean, v_q, v_scale_exp, v_raw) = splat_vjp::<P>(
        Vec3A::new(transforms[tbase], transforms[tbase + 1], transforms[tbase + 2]),
        read_scale(transforms, tbase, u.log_scale_offset),
        read_quat_unorm(transforms, tbase),
        raw_opac[global_gid as usize],
        sh_coeffs,
        min_scale,
        global_gid,
        v_rasterize_grads,
        rg_base,
        v_coeffs,
        coeff_out_base,
        u,
        mip_splatting,
        has_min_scale,
        sh_degree,
        neural,
        camera_model,
        false,
    );
    v_raw_opac[row as usize] = v_raw;

    // Make sure to keep refine weight >= 0 and finite. Helps with super large degenerate splats
    // that sum up their refine weight to some massive value.
    let refine_clean = select(is_finite_f32(v_refine_in), v_refine_in, 0.0f32);
    v_refine_weight[row as usize] = clamp(refine_clean, 0.0f32, 1.0e32f32);

    // Write gradients to the compact row.
    v_transforms[vbase] = v_mean.x();
    v_transforms[vbase + 1] = v_mean.y();
    v_transforms[vbase + 2] = v_mean.z();
    v_transforms[vbase + 3] = v_q.w();
    v_transforms[vbase + 4] = v_q.x();
    v_transforms[vbase + 5] = v_q.y();
    v_transforms[vbase + 6] = v_q.z();
    v_transforms[vbase + 7] = v_scale_exp.x();
    v_transforms[vbase + 8] = v_scale_exp.y();
    v_transforms[vbase + 9] = v_scale_exp.z();
}

/// Surfel (2DGS) projection backward. The raster backward hands over the
/// gradient of `M = K [a b c]` (rows), the camera-space normal, colour,
/// opacity and refine weight; `a = s_u W R e0`, `b = s_v W R e1`,
/// `c = W p + t`, `n = ±W R e2`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn surfel_backwards<P: Float>(
    transforms: &Tensor<f32>,
    sh_coeffs: &Tensor<P>,
    raw_opac: &Tensor<f32>,
    min_scale: &Tensor<f32>,
    global_gid: u32,
    compact_gid: u32,
    row: u32,
    v_rasterize_grads: &Tensor<f32>,
    v_transforms: &mut Tensor<f32>,
    v_coeffs: &mut Tensor<f32>,
    v_raw_opac: &mut Tensor<f32>,
    v_refine_weight: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] has_min_scale: bool,
    #[comptime] sh_degree: u32,
) {
    let rg = (compact_gid * GRADS_SURFEL) as usize;
    let mut any_grad = false;
    #[unroll]
    for i in 0..comptime![GRADS_SURFEL as usize] {
        any_grad = any_grad || v_rasterize_grads[rg + i] != 0.0f32;
    }
    if !any_grad {
        write_zero_row(
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
            row,
            sh_degree,
            // Surfels carry SH rows only.
            false,
        );
        terminate!();
    }
    // Gradient of M's columns (column j = (M0j, M1j, M2j)).
    let v_col0 = Vec3A::new(
        v_rasterize_grads[rg],
        v_rasterize_grads[rg + 3],
        v_rasterize_grads[rg + 6],
    );
    let v_col1 = Vec3A::new(
        v_rasterize_grads[rg + 1],
        v_rasterize_grads[rg + 4],
        v_rasterize_grads[rg + 7],
    );
    let v_col2 = Vec3A::new(
        v_rasterize_grads[rg + 2],
        v_rasterize_grads[rg + 5],
        v_rasterize_grads[rg + 8],
    );
    let v_n = Vec3A::new(
        v_rasterize_grads[rg + 9],
        v_rasterize_grads[rg + 10],
        v_rasterize_grads[rg + 11],
    );
    let v_color = Vec3A::new(
        v_rasterize_grads[rg + 12],
        v_rasterize_grads[rg + 13],
        v_rasterize_grads[rg + 14],
    );
    let v_alpha_in = v_rasterize_grads[rg + 15];
    let v_refine_in = v_rasterize_grads[rg + 16];

    let vbase = (row * 10u32) as usize;
    let coeff_out_base = row * comptime![num_sh_coeffs(sh_degree) * 3u32];
    let tbase = (global_gid * 10u32) as usize;
    let mean = Vec3A::new(
        transforms[tbase],
        transforms[tbase + 1],
        transforms[tbase + 2],
    );
    let scale_raw = read_scale(transforms, tbase, u.log_scale_offset);
    let quat_unorm = read_quat_unorm(transforms, tbase);
    let quat = quat_unorm.normalize();
    let opac_sig = sigmoid(raw_opac[global_gid as usize]);
    let floor = apply_scale_floor(scale_raw, opac_sig, min_scale, global_gid, has_min_scale);
    let scale = floor.scale;
    let ratio_sq = floor.ratio_sq;

    // SH colour and its view-direction dependence on the mean.
    let u_world = mean.sub(u.camera_pos());
    let u_len = u_world.length();
    let v = u_world.scale(1.0f32 / u_len);
    let coeff_base = global_gid * comptime![num_sh_coeffs(sh_degree) * 3u32];
    sh_coeffs_to_color_vjp(v_coeffs, coeff_out_base, sh_degree, v, v_color);
    let v_v_sh = sh_color_viewdir_vjp(sh_coeffs, coeff_base, sh_degree, v, v_color);
    let v_dot_vv = v.dot(v_v_sh);
    let v_mean_from_sh = v_v_sh.sub(v.scale(v_dot_vv)).scale(1.0f32 / u_len);

    // Opacity: no 2D blur compensation for surfels.
    let v_opac_base = select(floor.opac_open, v_alpha_in, 0.0f32);
    v_raw_opac[row as usize] = v_opac_base * floor.coef * opac_sig * (1.0f32 - opac_sig);
    let refine_clean = select(is_finite_f32(v_refine_in), v_refine_in, 0.0f32);
    v_refine_weight[row as usize] = clamp(refine_clean, 0.0f32, 1.0e32f32);

    // M = K X  =>  v_X = K^T v_M, column by column.
    let p = u.pinhole_params;
    let v_a = Vec3A::new(p.fx * v_col0.x(), p.fy * v_col0.y(), p.cx * v_col0.x() + p.cy * v_col0.y() + v_col0.z());
    let v_b = Vec3A::new(p.fx * v_col1.x(), p.fy * v_col1.y(), p.cx * v_col1.x() + p.cy * v_col1.y() + v_col1.z());
    let v_c = Vec3A::new(p.fx * v_col2.x(), p.fy * v_col2.y(), p.cx * v_col2.x() + p.cy * v_col2.y() + v_col2.z());

    let view_rot = u.view_rotation();
    let r = quat.to_mat3();
    let wr = view_rot.mul_mat3(r);
    let mean_c = world_to_cam(mean, u);
    // a = s_u (W R e0): d/ds_u = W R e0 . v_a; d/d(R e0) = s_u W^T v_a.
    let v_su = wr.col0().dot(v_a);
    let v_sv = wr.col1().dot(v_b);
    let flip = select(wr.col2().dot(mean_c) > 0.0f32, -1.0f32, 1.0f32);
    let v_r = Mat3::from_cols(
        view_rot.transpose_mul_vec3(v_a).scale(scale.x()),
        view_rot.transpose_mul_vec3(v_b).scale(scale.y()),
        view_rot.transpose_mul_vec3(v_n).scale(flip),
    );
    let q_grad = quat_to_mat_vjp(quat, v_r);
    let v_q = apply_normalize_vjp(quat_unorm, q_grad);
    let v_mean = view_rot.transpose_mul_vec3(v_c).add(v_mean_from_sh);

    // Log-scale gradients through the floor, plus the floor's opacity
    // compensation (which depends on all three raw scales).
    let v_scale_exp = Vec3A::new(
        v_su * scale.x() * ratio_sq.x() + v_opac_base * floor.opac * (1.0f32 - ratio_sq.x()),
        v_sv * scale.y() * ratio_sq.y() + v_opac_base * floor.opac * (1.0f32 - ratio_sq.y()),
        v_opac_base * floor.opac * (1.0f32 - ratio_sq.z()),
    );

    v_transforms[vbase] = v_mean.x();
    v_transforms[vbase + 1] = v_mean.y();
    v_transforms[vbase + 2] = v_mean.z();
    v_transforms[vbase + 3] = v_q.w();
    v_transforms[vbase + 4] = v_q.x();
    v_transforms[vbase + 5] = v_q.y();
    v_transforms[vbase + 6] = v_q.z();
    v_transforms[vbase + 7] = v_scale_exp.x();
    v_transforms[vbase + 8] = v_scale_exp.y();
    v_transforms[vbase + 9] = v_scale_exp.z();
}
