//! Project & cull pass.
//!
//! `project_forward` is the sole visibility gate. Guards are positive-
//! phrased so NaN reliably fails them (NaN comparisons are unordered).
//! Finite out-of-distribution values pass; calc_cov2d clamps overflow
//! internally.

use super::helpers::{
    apply_scale_floor, calc_cov2d, compensate_cov2d, compute_bbox_extent, count_contributing_tiles,
    get_tile_bbox, is_finite_f32, read_mean_viewspace, read_quat_unorm, read_scale, sigmoid,
    surfel_project, tile_rect, will_primitive_contribute,
};
use super::map_gaussians::SMALL_BBOX_TILES;
use super::types::{ProjectUniforms, Sym2, TileBbox};
use crate::kernels::camera_model::{CameraModel, project};
use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
// Mode-dependent values are assigned in comptime branches.
#[allow(unused_assignments)]
#[allow(clippy::too_many_arguments)]
pub fn project_forward_kernel(
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    min_scale: &Tensor<f32>,
    global_from_compact_gid: &mut Tensor<u32>,
    compact_from_global: &mut Tensor<u32>,
    opacities: &mut Tensor<f32>,
    depths: &mut Tensor<f32>,
    num_visible: &mut Tensor<Atomic<u32>>,
    intersect_counts: &mut Tensor<u32>,
    num_intersections: &mut Tensor<Atomic<u32>>,
    max_radius: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] has_min_scale: bool,
    #[comptime] bwd_info: bool,
    #[comptime] camera_model: CameraModel,
    #[comptime] planes: bool,
    #[comptime] surfel: bool,
) {
    let global_gid = ABSOLUTE_POS as u32;
    // Culling breaks out of this loop rather than terminating, so with
    // `planes` every lane reaches the cooperative tile count below.
    let mut visible = false;
    let mut depth = 0.0f32;
    let mut mx = 0.0f32;
    let mut my = 0.0f32;
    let mut c00 = 0.0f32;
    let mut c01 = 0.0f32;
    let mut c11 = 0.0f32;
    let mut power_threshold = 0.0f32;
    let mut bb_min_x = 0u32;
    let mut bb_min_y = 0u32;
    let mut bb_max_x = 0u32;
    let mut bb_max_y = 0u32;
    let mut radius = 0.0f32;
    if global_gid < u.total_splats {
        loop {
    // Defaults for culled splats: the backward's zero gradient row and a
    // zero opacity. Visible splats overwrite both further down.
    if bwd_info {
        compact_from_global[global_gid as usize] = 0u32;
    }
    opacities[global_gid as usize] = 0.0f32;

    // means(3) + quats(4) + log_scales(3)
    let base = (global_gid * 10u32) as usize;

    let mean_c = read_mean_viewspace(transforms, base, u);
    if !(mean_c.is_finite() && mean_c.z() <= 1.0e10f32) {
        break;
    }
    match camera_model {
        CameraModel::Pinhole => {
            if mean_c.z() < 0.01f32 {
                break;
            }
        }
        CameraModel::KannalaBrandt4(_)
        | CameraModel::RadialTangential8(_)
        | CameraModel::ThinPrismFisheye(_) => {
            let r = f32::sqrt(mean_c.x() * mean_c.x() + mean_c.y() * mean_c.y());
            let theta = r.atan2(mean_c.z());
            if theta > u.half_max_render_fov {
                break;
            }
        }
    }

    let scale = read_scale(transforms, base, u.log_scale_offset);
    if !scale.is_finite() {
        break;
    }

    let quat_unorm = read_quat_unorm(transforms, base);
    let qnorm_sq = quat_unorm.dot(quat_unorm);
    if !(qnorm_sq >= 1.0e-6f32 && is_finite_f32(qnorm_sq)) {
        break;
    }

    let raw_opac = raw_opacities[global_gid as usize];
    if !is_finite_f32(raw_opac) {
        break;
    }

    let quat = quat_unorm.normalize();

    // Mip-Splatting 3D filter, folded in here rather than on the host so
    // the raw params stay the kernel inputs.
    let floor = apply_scale_floor(
        scale,
        sigmoid(raw_opac),
        min_scale,
        global_gid,
        has_min_scale,
    );
    // Per-splat opacity with the floor folded in, for the trainer's noise gate.
    opacities[global_gid as usize] = floor.opac;

    let mut opac = floor.opac;
    let mut mean2d_x = 0.0f32;
    let mut mean2d_y = 0.0f32;
    if surfel {
        // 2DGS: bin by the screen ellipse covering the surfel's cutoff
        // footprint and its low-pass disk; no 2D blur compensation.
        if !(opac >= 1.0f32 / 255.0f32) {
            break;
        }
        power_threshold = f32::ln(opac * 255.0f32);
        let sp = surfel_project(floor.scale, quat, mean_c, u, power_threshold);
        if !sp.valid {
            break;
        }
        c00 = sp.bin_conic.c00;
        c01 = sp.bin_conic.c01;
        c11 = sp.bin_conic.c11;
        mean2d_x = sp.bin_x;
        mean2d_y = sp.bin_y;
    } else {
        let raw_cov = calc_cov2d(floor.scale, quat, mean_c, u, camera_model);
        let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
        opac = floor.opac * filter_comp;

        if !cov.is_finite() {
            break;
        }

        let (px, py) = project(mean_c, u.pinhole_params, camera_model);
        mean2d_x = px;
        mean2d_y = py;

        if !(opac >= 1.0f32 / 255.0f32) {
            break;
        }

        power_threshold = f32::ln(opac * 255.0f32);
        let inv = cov.inverse();
        c00 = inv.c00;
        c01 = inv.c01;
        c11 = inv.c11;
    }
    let conic = Sym2 { c00, c01, c11 };
    let (ex, ey) = compute_bbox_extent(conic, power_threshold);
    if !(ex >= 0.0f32 && ey >= 0.0f32) {
        break;
    }

    let img_w_f = u.img_w as f32;
    let img_h_f = u.img_h as f32;
    let on_screen = mean2d_x + ex > 0.0f32
        && mean2d_x - ex < img_w_f
        && mean2d_y + ey > 0.0f32
        && mean2d_y - ey < img_h_f;
    if !on_screen {
        break;
    }

    let tb = get_tile_bbox(mean2d_x, mean2d_y, ex, ey, u.tile_bw, u.tile_bh);
    bb_min_x = tb.min_x;
    bb_min_y = tb.min_y;
    bb_max_x = tb.max_x;
    bb_max_y = tb.max_y;
    mx = mean2d_x;
    my = mean2d_y;
    // Screen-space radius (pixels) for the small-splat prior.
    radius = f32::max(ex / img_w_f, ey / img_h_f);
    depth = mean_c.z();
    visible = true;
    break;
        }
    }

    // Count contributing tiles. Small boxes are walked by their own lane;
    // with subgroup ops larger ones are walked by the whole subgroup, so one
    // big splat no longer stalls its warp.
    let bb = TileBbox {
        min_x: bb_min_x,
        min_y: bb_min_y,
        max_x: bb_max_x,
        max_y: bb_max_y,
    };
    let conic = Sym2 { c00, c01, c11 };
    let bb_w = bb.max_x - bb.min_x;
    let num_tiles_bbox = select(visible, (bb.max_y - bb.min_y) * bb_w, 0u32);
    let alone_limit = comptime!(if planes { SMALL_BBOX_TILES } else { u32::MAX });
    let mut num_tiles_hit = 0u32;
    if visible && num_tiles_bbox <= alone_limit {
        num_tiles_hit = count_contributing_tiles(bb, mx, my, conic, power_threshold);
    }
    if planes {
        let mut pending = visible && num_tiles_bbox > SMALL_BBOX_TILES;
        while plane_any(pending) {
            let leader = plane_min(select(pending, UNIT_POS_PLANE, 0xFFFF_FFFFu32));
            let l_x = plane_shuffle(mx, leader);
            let l_y = plane_shuffle(my, leader);
            let l_conic = Sym2 {
                c00: plane_shuffle(conic.c00, leader),
                c01: plane_shuffle(conic.c01, leader),
                c11: plane_shuffle(conic.c11, leader),
            };
            let l_power = plane_shuffle(power_threshold, leader);
            let l_min_x = plane_shuffle(bb.min_x, leader);
            let l_min_y = plane_shuffle(bb.min_y, leader);
            let l_w = plane_shuffle(bb_w, leader);
            let l_n = plane_shuffle(num_tiles_bbox, leader);
            let mut hits = 0u32;
            let mut t = UNIT_POS_PLANE;
            while t < l_n {
                let tx = (t % l_w) + l_min_x;
                let ty = (t / l_w) + l_min_y;
                if will_primitive_contribute(tile_rect(tx, ty), l_x, l_y, l_conic, l_power) {
                    hits += 1u32;
                }
                t += PLANE_DIM;
            }
            let total = plane_sum(hits);
            if UNIT_POS_PLANE == leader {
                num_tiles_hit = total;
                pending = false;
            }
        }
    }

    if global_gid < u.total_splats {
        // Every splat writes its slots, culled ones zeros (their tile count
        // and radius never left 0), so neither buffer needs a fill.
        intersect_counts[global_gid as usize] = num_tiles_hit;
        max_radius[global_gid as usize] = radius;
    }
    if visible {
        Atomic::fetch_add(&num_intersections[0], num_tiles_hit);
        let write_id = Atomic::fetch_add(&num_visible[0], 1u32);
        global_from_compact_gid[write_id as usize] = global_gid;
        depths[write_id as usize] = depth;
    }
}

