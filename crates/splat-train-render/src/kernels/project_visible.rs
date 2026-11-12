//! Compute projected splat data for visible gaussians. PF already
//! culled non-finite-cov2d splats so this kernel trusts `calc_cov2d`.

use super::helpers::{
    AUX_LANES_GEOM, AUX_LANES_SURFEL, AUX_SURFEL_CENTER, AUX_SURFEL_NORMAL, apply_scale_floor,
    calc_cov2d, compensate_cov2d, is_finite_f32, read_quat_unorm, read_scale, sigmoid,
    surfel_normal, surfel_project, world_to_cam, write_projected_splat,
};
use super::neural::{NEURAL_DECODER_LEN, NEURAL_ROW_COEFFS, neural_color, stage_decoder};
use super::sh::{num_sh_coeffs, sh_coeffs_to_color};
use super::types::{ProjectUniforms, Splat, Vec3A};
use crate::kernels::camera_model::{CameraModel, project};
use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

/// View-dependent SH color of one splat with the 0.5 offset, NaN/Inf
/// scrubbed and clamped so the rasterize backward's gradient term can't
/// amplify past f32 range.
///
/// In neural mode (`decoder` staged, see [`stage_neural_decoder`]) the colour
/// comes from the shared decoder instead.
#[cube]
fn splat_color<P: Float>(
    coeffs: &Tensor<P>,
    decoder: ComptimeOption<Shared<[f32]>>,
    global_gid: u32,
    mean: Vec3A,
    u: ProjectUniforms,
    #[comptime] sh_degree: u32,
) -> Vec3A {
    // Viewdir. Safe to normalize: splats with length(mean - cam) == 0
    // would already be culled in PF.
    let v = mean.sub(u.camera_pos()).normalize();

    #[comptime]
    let color = match decoder {
        ComptimeOption::Some(decoder) => {
            // `u.sh_degree` is the decoder's active direction band here.
            let row = global_gid * comptime![NEURAL_ROW_COEFFS * 3u32];
            neural_color(coeffs, row, &decoder, v, u.sh_degree)
        }
        ComptimeOption::None => {
            let coeff_base = global_gid * comptime![num_sh_coeffs(sh_degree) * 3u32];
            let raw = sh_coeffs_to_color(coeffs, coeff_base, sh_degree, v);
            // SH-to-color offset.
            Vec3A::new(raw.x() + 0.5f32, raw.y() + 0.5f32, raw.z() + 0.5f32)
        }
    };
    let cr = color.x();
    let cg = color.y();
    let cb = color.z();

    Vec3A::new(
        clamp(select(is_finite_f32(cr), cr, 0.0f32), -100.0f32, 100.0f32),
        clamp(select(is_finite_f32(cg), cg, 0.0f32), -100.0f32, 100.0f32),
        clamp(select(is_finite_f32(cb), cb, 0.0f32), -100.0f32, 100.0f32),
    )
}

/// Neural mode stages the decoder, which rides at the front of the
/// `min_scale` binding (see `neural`), in shared memory; SH mode declares
/// nothing. Call before any unit can exit (it syncs).
#[cube]
fn stage_neural_decoder(
    min_scale: &Tensor<f32>,
    #[comptime] neural: bool,
) -> ComptimeOption<Shared<[f32]>> {
    if comptime![neural] {
        let mut decoder = Shared::<[f32]>::new_slice(NEURAL_DECODER_LEN as usize);
        stage_decoder(min_scale, &mut decoder);
        ComptimeOption::Some(decoder)
    } else {
        ComptimeOption::new_None()
    }
}

/// Row of the scale floor in the `min_scale` binding: neural mode puts the
/// decoder in front of it.
#[cube]
fn floor_gid(global_gid: u32, #[comptime] neural: bool) -> u32 {
    if comptime![neural] {
        global_gid + NEURAL_DECODER_LEN
    } else {
        global_gid
    }
}

// The `#[cube]` macro's terminal `write_projected_splat(...)` call expands to
// `expr();` plus a trailing `()` placeholder, which trips
// `semicolon_if_nothing_returned`. False positive — the macro already provides
// the semicolon. Silence here rather than in the kernel body.
#[allow(clippy::semicolon_if_nothing_returned)]
#[cube(launch)]
pub fn project_visible_kernel<P: Float>(
    transforms: &Tensor<f32>,
    coeffs: &Tensor<P>,
    raw_opacities: &Tensor<f32>,
    min_scale: &Tensor<f32>,
    global_from_presort_gid: &Tensor<u32>,
    num_visible: &Tensor<u32>,
    compact_from_global: &mut Tensor<u32>,
    projected: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] has_min_scale: bool,
    #[comptime] bwd_info: bool,
    #[comptime] sh_degree: u32,
    #[comptime] neural: bool,
    #[comptime] camera_model: CameraModel,
) {
    // Runs in PF's pre-sort order, which follows the global splat order, so
    // the transform / SH reads are near-sequential. The compact id of a
    // visible splat is its pre-sort index; depth order only matters for the
    // order intersections are emitted in (see `map_gaussians`).
    let decoder = stage_neural_decoder(min_scale, neural);
    let presort_idx = ABSOLUTE_POS as u32;
    if presort_idx >= num_visible[0] {
        terminate!();
    }

    let global_gid = global_from_presort_gid[presort_idx as usize];
    // Inverse map so the backward's compact gradients can be gathered per
    // global splat. Offset by one: row 0 of those buffers is the zero row
    // that culled splats point at.
    if bwd_info {
        compact_from_global[global_gid as usize] = presort_idx + 1u32;
    }

    // means(3) + quats(4) + log_scales(3)
    let base = (global_gid * 10u32) as usize;
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    let scale = read_scale(transforms, base, u.log_scale_offset);
    let quat_unorm = read_quat_unorm(transforms, base);
    let quat = quat_unorm.normalize();

    let opac_sig = sigmoid(raw_opacities[global_gid as usize]);
    // In neural mode `min_scale` is `[decoder | floor]` (see `neural`), which
    // keeps this kernel within WebGPU's default eight storage buffers.
    let floor = apply_scale_floor(
        scale,
        opac_sig,
        min_scale,
        floor_gid(global_gid, neural),
        has_min_scale,
    );

    let mean_c = world_to_cam(mean, u);
    let raw_cov = calc_cov2d(floor.scale, quat, mean_c, u, camera_model);
    let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    let opac = floor.opac * filter_comp;
    let conic = cov.inverse();

    let (mean2d_x, mean2d_y) = project(mean_c, u.pinhole_params, camera_model);
    let color = splat_color(coeffs, decoder, global_gid, mean, u, sh_degree);

    write_projected_splat(
        projected,
        presort_idx,
        Splat {
            xy_x: mean2d_x,
            xy_y: mean2d_y,
            conic_x: conic.c00,
            conic_y: conic.c01,
            conic_z: conic.c11,
            color_a: opac,
            color_r: color.x(),
            color_g: color.y(),
            color_b: color.z(),
        },
    );
}

/// Native-only variant for the geometry pass and surfels: also fills the
/// per-splat `splat_aux` side buffer (layout in `helpers::aux_lanes`). A
/// separate entry point keeps the default kernel within WebGPU's default
/// storage-buffer limit.
#[allow(clippy::semicolon_if_nothing_returned)]
#[cube(launch)]
pub fn project_visible_aux_kernel<P: Float>(
    transforms: &Tensor<f32>,
    coeffs: &Tensor<P>,
    raw_opacities: &Tensor<f32>,
    min_scale: &Tensor<f32>,
    global_from_presort_gid: &Tensor<u32>,
    num_visible: &Tensor<u32>,
    compact_from_global: &mut Tensor<u32>,
    projected: &mut Tensor<f32>,
    splat_aux: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] has_min_scale: bool,
    #[comptime] bwd_info: bool,
    #[comptime] sh_degree: u32,
    #[comptime] neural: bool,
    #[comptime] camera_model: CameraModel,
    #[comptime] surfel: bool,
) {
    let decoder = stage_neural_decoder(min_scale, neural);
    let presort_idx = ABSOLUTE_POS as u32;
    if presort_idx >= num_visible[0] {
        terminate!();
    }

    let global_gid = global_from_presort_gid[presort_idx as usize];
    if bwd_info {
        compact_from_global[global_gid as usize] = presort_idx + 1u32;
    }

    let base = (global_gid * 10u32) as usize;
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    let scale = read_scale(transforms, base, u.log_scale_offset);
    let quat_unorm = read_quat_unorm(transforms, base);
    let quat = quat_unorm.normalize();

    let opac_sig = sigmoid(raw_opacities[global_gid as usize]);
    let floor = apply_scale_floor(
        scale,
        opac_sig,
        min_scale,
        floor_gid(global_gid, neural),
        has_min_scale,
    );

    let mean_c = world_to_cam(mean, u);
    let (mean2d_x, mean2d_y) = project(mean_c, u.pinhole_params, camera_model);
    let color = splat_color(coeffs, decoder, global_gid, mean, u, sh_degree);

    if surfel {
        // PF only passed surfels whose projection is valid. The projected
        // row carries the binning ellipse; coverage reads `splat_aux`.
        let opac = floor.opac;
        let sp = surfel_project(floor.scale, quat, mean_c, u, f32::ln(opac * 255.0f32));
        let r0 = sp.m.row0();
        let r1 = sp.m.row1();
        let r2 = sp.m.row2();
        let n = surfel_normal(quat, mean_c, u);
        let b = (presort_idx * AUX_LANES_SURFEL) as usize;
        splat_aux[b] = r0.x();
        splat_aux[b + 1] = r0.y();
        splat_aux[b + 2] = r0.z();
        splat_aux[b + 3] = r1.x();
        splat_aux[b + 4] = r1.y();
        splat_aux[b + 5] = r1.z();
        splat_aux[b + 6] = r2.x();
        splat_aux[b + 7] = r2.y();
        splat_aux[b + 8] = r2.z();
        let c = b + AUX_SURFEL_CENTER as usize;
        splat_aux[c] = mean2d_x;
        splat_aux[c + 1] = mean2d_y;
        let nb = b + AUX_SURFEL_NORMAL as usize;
        splat_aux[nb] = n.x();
        splat_aux[nb + 1] = n.y();
        splat_aux[nb + 2] = n.z();
        write_projected_splat(
            projected,
            presort_idx,
            Splat {
                xy_x: sp.bin_x,
                xy_y: sp.bin_y,
                conic_x: sp.bin_conic.c00,
                conic_y: sp.bin_conic.c01,
                conic_z: sp.bin_conic.c11,
                color_a: opac,
                color_r: color.x(),
                color_g: color.y(),
                color_b: color.z(),
            },
        );
    } else {
        // Geometry pass for gaussians: center depth and the ellipsoid's
        // shortest axis in view space, flipped to face the camera.
        let raw_cov = calc_cov2d(floor.scale, quat, mean_c, u, camera_model);
        let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
        let opac = floor.opac * filter_comp;
        let conic = cov.inverse();

        let rot = u.view_rotation().mul_mat3(quat.to_mat3());
        let s = floor.scale;
        let x_min = s.x() <= s.y() && s.x() <= s.z();
        let y_min = !x_min && s.y() <= s.z();
        let (c0, c1, c2) = (rot.col0(), rot.col1(), rot.col2());
        let axis = Vec3A::new(
            select(x_min, c0.x(), select(y_min, c1.x(), c2.x())),
            select(x_min, c0.y(), select(y_min, c1.y(), c2.y())),
            select(x_min, c0.z(), select(y_min, c1.z(), c2.z())),
        );
        let flip = select(axis.dot(mean_c) > 0.0f32, -1.0f32, 1.0f32);
        let b = (presort_idx * AUX_LANES_GEOM) as usize;
        splat_aux[b] = mean_c.z();
        splat_aux[b + 1] = axis.x() * flip;
        splat_aux[b + 2] = axis.y() * flip;
        splat_aux[b + 3] = axis.z() * flip;
        write_projected_splat(
            projected,
            presort_idx,
            Splat {
                xy_x: mean2d_x,
                xy_y: mean2d_y,
                conic_x: conic.c00,
                conic_y: conic.c01,
                conic_z: conic.c11,
                color_a: opac,
                color_r: color.x(),
                color_g: color.y(),
                color_b: color.z(),
            },
        );
    }
}
