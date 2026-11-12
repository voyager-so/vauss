//! Camera gradient, reduced from the per-splat rasterizer gradients.
//!
//! The camera is perturbed by a camera-space affine map applied after the
//! world-to-camera transform, `x_c' = (I + A) x_c + b`, so the camera-space
//! covariance moves by `d S = A S + S A^T`. The camera centre moves by
//! `-M^-1 b` for view matrix `M` (A pivots on it), which is all the SH view
//! direction sees. Per
//! visible splat this recomputes the projection backward's camera-space
//! quantities and writes `[dA (row-major 3x3), db]`; the caller sums the rows.
//! A rigid twist `[rho, w]` is the special case `A = [w]x`, `b = rho`.

use crate::kernels::camera_model::CameraModel;
use crate::kernels::camera_model::{calculate_project_jacobian, calculate_projection_vjp};
use crate::kernels::helpers::{
    apply_scale_floor, calc_cov2d, compensate_cov2d, read_quat_unorm, read_scale, world_to_cam,
};
use crate::kernels::sh::{num_sh_coeffs, sh_color_viewdir_vjp};
use crate::kernels::types::{Mat3, ProjectUniforms, Sym2, Vec3A};
use brush_cube::Sym3;
use brush_cube::{Vec2, sigmoid};
use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

/// Width of one camera-gradient row: `dA` row-major, then `db`.
pub const POSE_DIM: u32 = 12;

/// Writes the row `[v x^T + 2 V S, b]`: `dA` from the mean (`v`, `x`) and the
/// covariance (`V`, `S`), then `db`.
#[cube]
fn write_pose_row(out: &mut Tensor<f32>, row: u32, v: Vec3A, x: Vec3A, vv: Sym3, s: Sym3, b: Vec3A) {
    let base = (row * POSE_DIM) as usize;
    let (v0, v1, v2) = (vv.row0(), vv.row1(), vv.row2());
    let (s0, s1, s2) = (s.row0(), s.row1(), s.row2());
    // (V S)_ij = V.row(i) . S.col(j), and S is symmetric.
    out[base] = v.x() * x.x() + 2.0f32 * v0.dot(s0);
    out[base + 1] = v.x() * x.y() + 2.0f32 * v0.dot(s1);
    out[base + 2] = v.x() * x.z() + 2.0f32 * v0.dot(s2);
    out[base + 3] = v.y() * x.x() + 2.0f32 * v1.dot(s0);
    out[base + 4] = v.y() * x.y() + 2.0f32 * v1.dot(s1);
    out[base + 5] = v.y() * x.z() + 2.0f32 * v1.dot(s2);
    out[base + 6] = v.z() * x.x() + 2.0f32 * v2.dot(s0);
    out[base + 7] = v.z() * x.y() + 2.0f32 * v2.dot(s1);
    out[base + 8] = v.z() * x.z() + 2.0f32 * v2.dot(s2);
    out[base + 9] = b.x();
    out[base + 10] = b.y();
    out[base + 11] = b.z();
}

#[cube]
fn cross(a: Vec3A, b: Vec3A) -> Vec3A {
    Vec3A::new(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )
}

/// `M^-T g`. Columns of `M^-T` are the column cross products over `det M`;
/// the view matrix is a rotation unless a camera-space affine bends it.
#[cube]
fn inverse_transpose_mul(m: Mat3, g: Vec3A) -> Vec3A {
    let (c0, c1, c2) = (m.col0(), m.col1(), m.col2());
    let k0 = cross(c1, c2);
    let k1 = cross(c2, c0);
    let k2 = cross(c0, c1);
    k0.scale(g.x())
        .add(k1.scale(g.y()))
        .add(k2.scale(g.z()))
        .scale(1.0f32 / c0.dot(k0))
}

#[cube]
fn write_zero_row(out: &mut Tensor<f32>, row: u32) {
    let base = (row * POSE_DIM) as usize;
    for k in 0u32..POSE_DIM {
        out[base + k as usize] = 0.0f32;
    }
}

/// One row per compact visible splat, matching `project_backwards_kernel`'s
/// inputs; rows past the visible count (up to the output's padded length) are
/// zero so the sum covers capacity.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn pose_backwards_kernel<P: Float>(
    transforms: &Tensor<f32>,
    sh_coeffs: &Tensor<P>,
    raw_opac: &Tensor<f32>,
    min_scale: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    num_visible_gpu: &Tensor<u32>,
    v_rasterize_grads: &Tensor<f32>,
    v_pose: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] has_min_scale: bool,
    #[comptime] sh_degree: u32,
    #[comptime] camera_model: CameraModel,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if (compact_gid * POSE_DIM) as usize >= v_pose.len() {
        terminate!();
    }
    if compact_gid >= u.total_splats || compact_gid >= num_visible_gpu[0] {
        write_zero_row(v_pose, compact_gid);
        terminate!();
    }
    let global_gid = global_from_compact_gid[compact_gid as usize];

    let rg_base = (compact_gid * 10u32) as usize;
    let v_mean2d = Vec2::new(v_rasterize_grads[rg_base], v_rasterize_grads[rg_base + 1]);
    let v_conics_x = v_rasterize_grads[rg_base + 2];
    let v_conics_y = v_rasterize_grads[rg_base + 3];
    let v_conics_z = v_rasterize_grads[rg_base + 4];
    let v_color = Vec3A::new(
        v_rasterize_grads[rg_base + 5],
        v_rasterize_grads[rg_base + 6],
        v_rasterize_grads[rg_base + 7],
    );
    let any_grad = v_mean2d.x() != 0.0f32
        || v_mean2d.y() != 0.0f32
        || v_conics_x != 0.0f32
        || v_conics_y != 0.0f32
        || v_conics_z != 0.0f32
        || v_color.x() != 0.0f32
        || v_color.y() != 0.0f32
        || v_color.z() != 0.0f32;
    if !any_grad {
        write_zero_row(v_pose, compact_gid);
        terminate!();
    }

    let tbase = (global_gid * 10u32) as usize;
    let mean = Vec3A::new(
        transforms[tbase],
        transforms[tbase + 1],
        transforms[tbase + 2],
    );
    let scale_raw = read_scale(transforms, tbase, u.log_scale_offset);
    let quat = read_quat_unorm(transforms, tbase).normalize();
    let opac_sig = sigmoid(raw_opac[global_gid as usize]);
    let scale = apply_scale_floor(scale_raw, opac_sig, min_scale, global_gid, has_min_scale).scale;

    // SH view direction: only the camera centre moves it.
    let u_world = mean.sub(u.camera_pos());
    let u_len = u_world.length();
    let dir = u_world.scale(1.0f32 / u_len);
    let coeff_base = global_gid * comptime![num_sh_coeffs(sh_degree) * 3u32];
    let v_dir = sh_color_viewdir_vjp(sh_coeffs, coeff_base, sh_degree, dir, v_color);
    let v_mean_from_sh = v_dir.sub(dir.scale(dir.dot(v_dir))).scale(1.0f32 / u_len);

    let mean_c = world_to_cam(mean, u);
    let raw_cov = calc_cov2d(scale, quat, mean_c, u, camera_model);
    let (cov, _filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    let conic = cov.inverse();
    // Same symmetric conic VJP as the projection backward.
    let p00 = conic.c00;
    let p01 = conic.c01;
    let p11 = conic.c11;
    let g00 = v_conics_x;
    let g01 = v_conics_y * 0.5f32;
    let g11 = v_conics_z;
    let t00 = -p00 * g00 - p01 * g01;
    let t01 = -p01 * g00 - p11 * g01;
    let t10 = -p00 * g01 - p01 * g11;
    let t11 = -p01 * g01 - p11 * g11;
    let v_cov2d = Sym2 {
        c00: t00 * p00 + t10 * p01,
        c01: t01 * p00 + t11 * p01,
        c11: t01 * p01 + t11 * p11,
    };

    let view_rot = u.view_rotation();
    let m = quat.to_mat3().mul_diag(scale);
    let cov_c = m.outer_product_self().congruence(view_rot);
    let cam_jac = calculate_project_jacobian(
        mean_c,
        u.jacobian_clamp_limits,
        u.pinhole_params,
        camera_model,
    );
    let v_mean_c =
        calculate_projection_vjp(cam_jac, mean_c, cov_c, u, v_cov2d, v_mean2d, camera_model);
    let v_cov_c = cam_jac.transpose_congruence_sym2(v_cov2d);

    let v_b = v_mean_c.add(inverse_transpose_mul(view_rot, v_mean_from_sh));
    write_pose_row(v_pose, compact_gid, v_mean_c, mean_c, v_cov_c, cov_c, v_b);
}
