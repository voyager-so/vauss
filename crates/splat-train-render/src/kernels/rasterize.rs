//! Tile-based gaussian rasterizer.
//!
//! One workgroup of `TILE_SIZE` threads per tile, each thread processes a
//! single pixel. Threads cooperate to load splats into a workgroup-shared
//! `local_batch` then iterate splats across all pixels.
//!
//! `bwd_info` enables: (a) writing `out_img` as 4 f32s (rgba) so the
//! backward kernel can recover the final color/alpha; (b) marking
//! `visible` splats; (c) shrinking `tile_offsets[tile*2+1]` to "one past
//! the last splat any pixel actually consumed" so the backward kernel's
//! outer loop ends early. When `bwd_info=false` the kernel writes a
//! packed u8x4 to `out_img` and skips the backward bookkeeping.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;
use burn::cubecl::std::FastDivmod;

use super::helpers::{
    ALPHA_CUTOFF_MID, AUX_SURFEL_CENTER, AUX_SURFEL_NORMAL, GEOM_OUT_CHANNELS, SURFEL_NEAR,
    is_finite_f32,
    aux_lanes, dist_depth, PROJECTED_LANES, PROJECTED_LANES_USIZE, TILE_SIZE, TILE_WIDTH,
    alpha_cutoff_weight, calc_sigma, map_1d_to_2d,
};
use super::types::{RasterizeUniforms, Sym2};

/// Default entry point: RGB only, with the argument list WebGPU's default
/// 8-storage-buffer limit allows.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rasterize_kernel(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    projected: &Tensor<f32>,
    out_img_packed: &mut Tensor<u32>,
    out_img_f32: &mut Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    visible: &mut Tensor<f32>,
    u: RasterizeUniforms,
    tile_bw_div: FastDivmod<u32>,
    #[comptime] bwd_info: bool,
    #[comptime] smooth_cutoff: bool,
) {
    // `projected` stands in for the unread aux buffer.
    rasterize_body(
        compact_gid_from_isect,
        tile_offsets,
        projected,
        out_img_packed,
        out_img_f32,
        global_from_compact_gid,
        visible,
        projected,
        u,
        tile_bw_div,
        bwd_info,
        smooth_cutoff,
        false,
        false,
        false,
    );
}

/// Native entry point for the geometry pass and/or surfels: one more buffer,
/// the per-splat `splat_aux`. With `geometry`, `out_img_f32` holds
/// `4 + GEOM_OUT_CHANNELS` floats per pixel (RGBA, then geometry).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rasterize_aux_kernel(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    projected: &Tensor<f32>,
    out_img_packed: &mut Tensor<u32>,
    out_img_f32: &mut Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    visible: &mut Tensor<f32>,
    splat_aux: &Tensor<f32>,
    u: RasterizeUniforms,
    tile_bw_div: FastDivmod<u32>,
    #[comptime] bwd_info: bool,
    #[comptime] smooth_cutoff: bool,
    #[comptime] geometry: bool,
    #[comptime] surfel: bool,
    #[comptime] surface_criterion: bool,
) {
    rasterize_body(
        compact_gid_from_isect,
        tile_offsets,
        projected,
        out_img_packed,
        out_img_f32,
        global_from_compact_gid,
        visible,
        splat_aux,
        u,
        tile_bw_div,
        bwd_info,
        smooth_cutoff,
        geometry,
        surfel,
        surface_criterion,
    );
}

#[cube]
// Mode-dependent values are assigned in comptime branches.
#[allow(unused_assignments)]
#[allow(clippy::too_many_arguments)]
fn rasterize_body(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    projected: &Tensor<f32>,
    out_img_packed: &mut Tensor<u32>,
    out_img_f32: &mut Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    visible: &mut Tensor<f32>,
    splat_aux: &Tensor<f32>,
    u: RasterizeUniforms,
    tile_bw_div: FastDivmod<u32>,
    #[comptime] bwd_info: bool,
    #[comptime] smooth_cutoff: bool,
    #[comptime] geometry: bool,
    #[comptime] surfel: bool,
    #[comptime] surface_criterion: bool,
) {
    let global_id = ABSOLUTE_POS as u32;
    let (pix_x, pix_y) = map_1d_to_2d(global_id, tile_bw_div);
    let pix_id = pix_x + pix_y * u.img_w;
    let pixel_coord_x = pix_x as f32 + 0.5f32;
    let pixel_coord_y = pix_y as f32 + 0.5f32;
    let tile_loc_x = pix_x / TILE_WIDTH;
    let tile_loc_y = pix_y / TILE_WIDTH;
    let tile_id = tile_loc_x + tile_loc_y * u.tile_bw;
    let inside = pix_x < u.img_w && pix_y < u.img_h;

    // Workgroup-shared splat batch + bookkeeping. The bwd-only `load_gid`
    // gets a comptime-tiny size when `bwd_info=false` so we don't pay
    // 1 KiB of static shared mem on the forward-only variant.
    // Plain gaussians stage each splat as two vec4s, (xy, c00, c01) and
    // (c11, alpha, r, g), plus b, so a pixel reads a splat in two loads.
    // Surfels keep the scalar layout.
    let mut local_batch = Shared::new_slice(
        comptime![if surfel { TILE_SIZE * PROJECTED_LANES } else { 1u32 }] as usize,
    );
    let vec_n = comptime![if surfel { 1u32 } else { TILE_SIZE }] as usize;
    let mut batch_a = Shared::<[Vector<f32, Const<4>>]>::new_slice(vec_n);
    let mut batch_b = Shared::<[Vector<f32, Const<4>>]>::new_slice(vec_n);
    let mut batch_c = Shared::<[f32]>::new_slice(vec_n);
    let mut load_gid =
        Shared::new_slice(comptime![if bwd_info { TILE_SIZE } else { 1u32 }] as usize);
    let aux_n = comptime![aux_lanes(surfel, geometry)];
    let mut local_aux =
        Shared::new_slice(comptime![if aux_n > 0 { TILE_SIZE * aux_n } else { 1u32 }] as usize);
    let num_done_atomic = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let max_useful_isect = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let mut range = Shared::new_slice(2usize);

    let local_idx = UNIT_POS;
    if local_idx == 0u32 {
        range[0] = tile_offsets[(tile_id * 2u32) as usize];
        range[1] = tile_offsets[(tile_id * 2u32 + 1u32) as usize];
        Atomic::store(&num_done_atomic[0], 0u32);
    }

    // Uniform-marked loads so the loop bounds + early-exit don't trip
    // WebGPU's "workgroupBarrier in non-uniform control flow" check.
    let range_lo = workgroup_uniform_load(&range[0]);
    let range_hi = workgroup_uniform_load(&range[1]);

    if comptime![bwd_info] && local_idx == 0u32 {
        Atomic::store(&max_useful_isect[0], range_lo);
    }

    let mut t_acc = 1.0f32;
    let mut pix_r = 0.0f32;
    let mut pix_g = 0.0f32;
    let mut pix_b = 0.0f32;
    let mut depth_acc = 0.0f32;
    let mut median_depth = 0.0f32;
    let mut median_id = 0xFFFF_FFFFu32;
    // Surfel renders outside training (mesh extraction, depth export):
    // unbiased-depth surface criterion (arXiv 2503.06587), first contributor
    // where O = sum (alpha_j + 0.1 G_j) exceeds 0.6, G_j the falloff without
    // opacity; the T > 0.5 median stands until then. Training keeps the
    // T > 0.5 median for its normal loss.
    let mut surf_o = 0.0f32;
    let mut surf_found = false;
    let mut dist_b = 0.0f32;
    let mut dist_c = 0.0f32;
    let mut n_x = 0.0f32;
    let mut n_y = 0.0f32;
    let mut n_z = 0.0f32;
    let mut done = !inside;
    let mut last_useful_isect = range_lo;

    if done {
        Atomic::fetch_add(&num_done_atomic[0], 1u32);
    }
    sync_cube();

    let mut batch_start = range_lo;
    while batch_start < range_hi {
        // Doubles as the sync between previous iter's local_batch reads
        // and this iter's overwrites.
        if workgroup_uniform_load_atomic(&num_done_atomic[0]) >= TILE_SIZE {
            break;
        }
        let remaining = min(TILE_SIZE, range_hi - batch_start);
        let load_isect_id = batch_start + local_idx;
        let mut compact_gid = 0u32;
        if local_idx < remaining {
            compact_gid = compact_gid_from_isect[load_isect_id as usize];
        }
        if local_idx < remaining {
            let src_base = (compact_gid * PROJECTED_LANES) as usize;
            if comptime![surfel] {
                let dst_base = (local_idx * PROJECTED_LANES) as usize;
                #[unroll]
                for lane in 0..PROJECTED_LANES_USIZE {
                    local_batch[dst_base + lane] = projected[src_base + lane];
                }
            } else {
                let mut a = Vector::<f32, Const<4>>::empty();
                a.insert(0usize, projected[src_base]);
                a.insert(1usize, projected[src_base + 1]);
                a.insert(2usize, projected[src_base + 2]);
                a.insert(3usize, projected[src_base + 3]);
                let mut b = Vector::<f32, Const<4>>::empty();
                b.insert(0usize, projected[src_base + 4]);
                b.insert(1usize, projected[src_base + 5]);
                b.insert(2usize, projected[src_base + 6]);
                b.insert(3usize, projected[src_base + 7]);
                batch_a[local_idx as usize] = a;
                batch_b[local_idx as usize] = b;
                batch_c[local_idx as usize] = projected[src_base + 8];
            }
            if comptime![bwd_info] {
                load_gid[local_idx as usize] = global_from_compact_gid[compact_gid as usize];
            }
            if comptime![aux_n > 0] {
                let g_src = (compact_gid * aux_n) as usize;
                let g_dst = (local_idx * aux_n) as usize;
                #[unroll]
                for lane in 0..comptime![aux_n as usize] {
                    local_aux[g_dst + lane] = splat_aux[g_src + lane];
                }
            }
        }
        sync_cube();

        let was_done = done;
        let mut t = 0u32;
        while !done && t < remaining {
            let dst_base = (t * PROJECTED_LANES) as usize;
            let mut va = Vector::<f32, Const<4>>::empty();
            let mut vb = Vector::<f32, Const<4>>::empty();
            let mut color_a = 0.0f32;
            if comptime![surfel] {
                color_a = local_batch[dst_base + 5];
            } else {
                va = batch_a[t as usize];
                vb = batch_b[t as usize];
                color_a = vb.extract(1usize);
            }
            let g = (t * aux_n) as usize;
            let mut sigma = 0.0f32;
            // Surfel intersection depth; gaussians take depth from `splat_aux`.
            let mut z = 0.0f32;
            if comptime![surfel] {
                // 2DGS: the pixel ray meets the surfel plane where both
                // homogeneous lines h_u = M0 - x M2 and h_v = M1 - y M2
                // vanish; (u, v) = (q.x, q.y) / q.z with q = h_u x h_v.
                let m20 = local_aux[g + 6];
                let m21 = local_aux[g + 7];
                let m22 = local_aux[g + 8];
                let hu_x = local_aux[g] - pixel_coord_x * m20;
                let hu_y = local_aux[g + 1] - pixel_coord_x * m21;
                let hu_z = local_aux[g + 2] - pixel_coord_x * m22;
                let hv_x = local_aux[g + 3] - pixel_coord_y * m20;
                let hv_y = local_aux[g + 4] - pixel_coord_y * m21;
                let hv_z = local_aux[g + 5] - pixel_coord_y * m22;
                let q_x = hu_y * hv_z - hu_z * hv_y;
                let q_y = hu_z * hv_x - hu_x * hv_z;
                let q_z = hu_x * hv_y - hu_y * hv_x;
                let su = q_x / q_z;
                let sv = q_y / q_z;
                let s3d = su * su + sv * sv;
                // Ray-plane intersection depth (camera z of
                // p + s_u t_u u + s_v t_v v), used for both branches below.
                z = m20 * su + m21 * sv + m22;
                // Screen-space low-pass (sigma = sqrt(2)/2 px), which also
                // covers grazing rays where the plane solve degenerates.
                let c = AUX_SURFEL_CENTER as usize;
                let dx = pixel_coord_x - local_aux[g + c];
                let dy = pixel_coord_y - local_aux[g + c + 1];
                let s2d = 2.0f32 * (dx * dx + dy * dy);
                sigma = 0.5f32 * min(s3d, s2d);
                // No intersection (ray parallel to the plane, NaN) or one
                // behind the near plane: the surfel does not contribute.
                if !(is_finite_f32(z) && z >= SURFEL_NEAR) {
                    sigma = -1.0f32;
                }
            } else {
                let xy_x = va.extract(0usize);
                let xy_y = va.extract(1usize);
                let conic = Sym2 {
                    c00: va.extract(2usize),
                    c01: va.extract(3usize),
                    c11: vb.extract(0usize),
                };
                sigma = calc_sigma(pixel_coord_x, pixel_coord_y, conic, xy_x, xy_y);
            }
            let alpha = min(0.999f32, color_a * f32::exp(-sigma));

            let w_cut = if comptime![smooth_cutoff] {
                alpha_cutoff_weight(alpha)
            } else {
                select(alpha >= ALPHA_CUTOFF_MID, 1.0f32, 0.0f32)
            };
            if sigma >= 0.0f32 && w_cut > 0.0f32 {
                let alpha_eff = alpha * w_cut;
                let next_t = t_acc * (1.0f32 - alpha_eff);
                if next_t <= 1.0e-4f32 {
                    done = true;
                } else {
                    if comptime![bwd_info] {
                        visible[load_gid[t as usize] as usize] = 1.0f32;
                    }
                    let vis = alpha_eff * t_acc;
                    let (cr, cg, cb) = if comptime![surfel] {
                        (
                            local_batch[dst_base + 6],
                            local_batch[dst_base + 7],
                            local_batch[dst_base + 8],
                        )
                    } else {
                        (vb.extract(2usize), vb.extract(3usize), batch_c[t as usize])
                    };
                    pix_r += max(cr, 0.0f32) * vis;
                    pix_g += max(cg, 0.0f32) * vis;
                    pix_b += max(cb, 0.0f32) * vis;
                    if comptime![geometry] {
                        let nb = comptime![if surfel { AUX_SURFEL_NORMAL } else { 1u32 }] as usize;
                        if comptime![!surfel] {
                            z = local_aux[g];
                        }
                        depth_acc += z * vis;
                        // Median (2DGS paper eq. 19): depth of the last
                        // contributor reached while transmittance > 0.5.
                        if comptime![surface_criterion] {
                            surf_o += alpha_eff + 0.1f32 * f32::exp(-sigma);
                            if !surf_found && (surf_o > 0.6f32 || t_acc > 0.5f32) {
                                median_depth = z;
                                median_id = batch_start + t;
                                surf_found = surf_o > 0.6f32;
                            }
                        } else if t_acc > 0.5f32 {
                            median_depth = z;
                            median_id = batch_start + t;
                        }
                        let m = dist_depth(z);
                        dist_b += m * vis;
                        dist_c += m * m * vis;
                        n_x += local_aux[g + nb] * vis;
                        n_y += local_aux[g + nb + 1] * vis;
                        n_z += local_aux[g + nb + 2] * vis;
                    }
                    t_acc = next_t;
                    last_useful_isect = batch_start + t + 1u32;
                }
            }
            t += 1u32;
        }
        if !was_done && done {
            Atomic::fetch_add(&num_done_atomic[0], 1u32);
        }
        batch_start += TILE_SIZE;
    }

    if inside {
        let final_r = pix_r + t_acc * u.bg_r;
        let final_g = pix_g + t_acc * u.bg_g;
        let final_b = pix_b + t_acc * u.bg_b;
        let final_a = 1.0f32 - t_acc;
        let out_ch = comptime![if geometry { 4u32 + GEOM_OUT_CHANNELS } else { 4u32 }];
        if comptime![geometry] {
            let g = (pix_id * out_ch + 4u32) as usize;
            out_img_f32[g] = depth_acc;
            out_img_f32[g + 1] = median_depth;
            out_img_f32[g + 5] = dist_b;
            out_img_f32[g + 6] = dist_c;
            out_img_f32[g + 7] = f32::reinterpret(median_id);
            out_img_f32[g + 2] = n_x;
            out_img_f32[g + 3] = n_y;
            out_img_f32[g + 4] = n_z;
        }
        if comptime![bwd_info] {
            let base = (pix_id * out_ch) as usize;
            out_img_f32[base] = final_r;
            out_img_f32[base + 1] = final_g;
            out_img_f32[base + 2] = final_b;
            out_img_f32[base + 3] = final_a;
        } else {
            let r = clamp(final_r * 255.0f32, 0.0f32, 255.0f32) as u32;
            let g = clamp(final_g * 255.0f32, 0.0f32, 255.0f32) as u32;
            let b = clamp(final_b * 255.0f32, 0.0f32, 255.0f32) as u32;
            let a = clamp(final_a * 255.0f32, 0.0f32, 255.0f32) as u32;
            let packed = r | (g << 8u32) | (b << 16u32) | (a << 24u32);
            out_img_packed[pix_id as usize] = packed;
        }
    }

    if comptime![bwd_info] {
        Atomic::fetch_max(&max_useful_isect[0], last_useful_isect);
        sync_cube();
        if local_idx == 0u32 {
            tile_offsets[(tile_id * 2u32 + 1u32) as usize] = Atomic::load(&max_useful_isect[0]);
        }
    }
}
