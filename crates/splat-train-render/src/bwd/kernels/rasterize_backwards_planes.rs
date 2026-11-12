//! Register-resident backward rasterizer for devices with subgroup ops.
//!
//! One 128-thread cube per tile, two pixels per thread; each thread keeps its
//! pixels' replay state and output adjoints in registers for the whole tile
//! list. Splats are staged in shared memory a batch at a time
//! and replayed front to back exactly like the forward pass, so barriers
//! are per batch, not per pixel. Per splat, the pixel gradients are summed
//! within each thread, then across the subgroup (a butterfly reduce-scatter
//! on 32-wide subgroups), and the subgroups' partials are summed in shared
//! memory once per batch: one atomic set per tile/splat, the same as the
//! per-splat kernel, but with no per-pixel barrier and no shared pixel state.
//! A splat no pixel of a subgroup touched skips that subgroup's reduction.
//!
//! Other subgroup widths let every subgroup leader emit its own partial sums,
//! which is correct for any width and only adds atomics.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

use super::rasterize_backwards::{AtomicAddF32, load_range, tile_origin};
use crate::kernels::helpers::{
    ALPHA_CUTOFF_BAND, ALPHA_CUTOFF_MID, AUX_LANES_SURFEL, AUX_SURFEL_CENTER, AUX_SURFEL_NORMAL,
    GEOM_OUT_CHANNELS, PROJECTED_LANES, PROJECTED_LANES_USIZE, SURFEL_NEAR, TILE_SIZE, TILE_WIDTH,
    DIST_FAR, DIST_NEAR, alpha_cutoff_weight, alpha_cutoff_weight_deriv, dist_depth, is_finite_f32,
};
use crate::kernels::types::RasterizeUniforms;

/// Threads per tile. Four subgroups replay a tile's list in parallel, so a
/// long list is not serialised on one subgroup. On L40S 128 beat both 32 and
/// 256 threads end to end.
pub const WG_SIZE: u32 = 128;
/// Splats staged in shared memory per batch.
const BATCH: u32 = 32;
/// Subgroups per tile when subgroups are 32 wide.
const NUM_WARPS: u32 = WG_SIZE / 32;
const PIXELS_PER_THREAD: usize = (TILE_SIZE / WG_SIZE) as usize;
/// Rows of the tile between a thread's consecutive pixels.
const ROW_STRIDE: u32 = WG_SIZE / TILE_WIDTH;

/// Gradient components per splat in `v_splats`: gaussians write
/// `[xy(2), conic(3), rgb(3), alpha, refine]`; surfels write
/// `[M rows(9), normal(3), rgb(3), alpha, refine]` (the low-pass center is a
/// function of M, so its gradient is folded into M).
pub const GRADS_GAUSSIAN: u32 = 10;
pub const GRADS_SURFEL: u32 = 17;
pub const fn grad_lanes(surfel: bool) -> u32 {
    if surfel { GRADS_SURFEL } else { GRADS_GAUSSIAN }
}

#[cube(launch)]
pub fn rasterize_backwards_planes_kernel<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    v_splats: &mut Tensor<Atomic<A::Storage>>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
    #[comptime] xor32: bool,
) {
    // `projected` stands in for the unread surfel buffer.
    backwards_planes_body::<A>(
        compact_gid_from_isect,
        tile_offsets,
        projected,
        projected,
        output,
        v_output,
        v_splats,
        u,
        smooth_cutoff,
        xor32,
        false,
    );
}

/// 2DGS surfels: replays the ray-plane coverage and the geometry channels
/// (expected depth, median depth, normal) the forward appended after RGBA,
/// and writes `GRADS_SURFEL` components per splat. Subgroup devices only.
#[cube(launch)]
pub fn rasterize_backwards_planes_surfel_kernel<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    splat_aux: &Tensor<f32>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    v_splats: &mut Tensor<Atomic<A::Storage>>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
    #[comptime] xor32: bool,
) {
    backwards_planes_body::<A>(
        compact_gid_from_isect,
        tile_offsets,
        projected,
        splat_aux,
        output,
        v_output,
        v_splats,
        u,
        smooth_cutoff,
        xor32,
        true,
    );
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn backwards_planes_body<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    splat_aux: &Tensor<f32>,
    output: &Tensor<f32>,
    v_output: &Tensor<f32>,
    v_splats: &mut Tensor<Atomic<A::Storage>>,
    u: RasterizeUniforms,
    #[comptime] smooth_cutoff: bool,
    #[comptime] xor32: bool,
    #[comptime] surfel: bool,
) {
    let n_grads = comptime![grad_lanes(surfel)];
    // Float channels per pixel of `output` / `v_output`.
    let out_ch = comptime![if surfel { 4u32 + GEOM_OUT_CHANNELS } else { 4u32 }];
    // Per-pixel surfel state, sized to one when unused.
    let px_s = comptime![if surfel { PIXELS_PER_THREAD } else { 1usize }];
    let (tile_id, tile_origin_x, tile_origin_y) = tile_origin(u.tile_bw);
    let pix_x = tile_origin_x + UNIT_POS % TILE_WIDTH;
    let first_pix_y = tile_origin_y + UNIT_POS / TILE_WIDTH;
    let pixel_coord_x = pix_x as f32 + 0.5f32;

    // Per-pixel replay state: remaining colour (final minus the background
    // pre-roll, walked down to zero) and transmittance. Out-of-image pixels
    // start with zero transmittance and never contribute.
    let mut state_r = Array::<f32>::new(PIXELS_PER_THREAD);
    let mut state_g = Array::<f32>::new(PIXELS_PER_THREAD);
    let mut state_b = Array::<f32>::new(PIXELS_PER_THREAD);
    let mut state_t = Array::<f32>::new(PIXELS_PER_THREAD);
    let mut v_o_r = Array::<f32>::new(PIXELS_PER_THREAD);
    let mut v_o_g = Array::<f32>::new(PIXELS_PER_THREAD);
    let mut v_o_b = Array::<f32>::new(PIXELS_PER_THREAD);
    let mut v_o_t = Array::<f32>::new(PIXELS_PER_THREAD);
    let mut refine_scale = Array::<f32>::new(PIXELS_PER_THREAD);
    // Surfel geometry channels: remaining depth / normal sums and their
    // adjoints, plus the median contributor's intersection index.
    let mut state_z = Array::<f32>::new(px_s);
    let mut state_nx = Array::<f32>::new(px_s);
    let mut state_ny = Array::<f32>::new(px_s);
    let mut state_nz = Array::<f32>::new(px_s);
    let mut v_o_z = Array::<f32>::new(px_s);
    let mut v_o_med = Array::<f32>::new(px_s);
    let mut v_o_nx = Array::<f32>::new(px_s);
    let mut v_o_ny = Array::<f32>::new(px_s);
    let mut v_o_nz = Array::<f32>::new(px_s);
    let mut state_db = Array::<f32>::new(px_s);
    let mut state_dc = Array::<f32>::new(px_s);
    let mut v_o_db = Array::<f32>::new(px_s);
    let mut v_o_dc = Array::<f32>::new(px_s);
    let mut med_id = Array::<u32>::new(px_s);
    let img_size_x = u.img_w as f32;
    let img_size_y = u.img_h as f32;

    #[unroll]
    for k in 0..PIXELS_PER_THREAD {
        let pix_y = first_pix_y + (k as u32) * ROW_STRIDE;
        state_r[k] = 0.0f32;
        state_g[k] = 0.0f32;
        state_b[k] = 0.0f32;
        state_t[k] = 0.0f32;
        v_o_r[k] = 0.0f32;
        v_o_g[k] = 0.0f32;
        v_o_b[k] = 0.0f32;
        v_o_t[k] = 0.0f32;
        refine_scale[k] = 0.0f32;
        if comptime![surfel] {
            state_z[k] = 0.0f32;
            state_nx[k] = 0.0f32;
            state_ny[k] = 0.0f32;
            state_nz[k] = 0.0f32;
            v_o_z[k] = 0.0f32;
            v_o_med[k] = 0.0f32;
            v_o_nx[k] = 0.0f32;
            v_o_ny[k] = 0.0f32;
            v_o_nz[k] = 0.0f32;
            state_db[k] = 0.0f32;
            state_dc[k] = 0.0f32;
            v_o_db[k] = 0.0f32;
            v_o_dc[k] = 0.0f32;
            med_id[k] = 0xFFFF_FFFFu32;
        }
        if pix_x < u.img_w && pix_y < u.img_h {
            let base = ((pix_x + pix_y * u.img_w) * out_ch) as usize;
            let final_a = output[base + 3];
            let t_final = 1.0f32 - final_a;
            state_r[k] = output[base] - t_final * u.bg_r;
            state_g[k] = output[base + 1] - t_final * u.bg_g;
            state_b[k] = output[base + 2] - t_final * u.bg_b;
            state_t[k] = 1.0f32;
            let vr = v_output[base];
            let vg = v_output[base + 1];
            let vb = v_output[base + 2];
            v_o_r[k] = vr;
            v_o_g[k] = vg;
            v_o_b[k] = vb;
            v_o_t[k] = (v_output[base + 3] - (u.bg_r * vr + u.bg_g * vg + u.bg_b * vb)) * t_final;
            refine_scale[k] = 1.0f32 / max(final_a, 1.0e-5f32);
            if comptime![surfel] {
                // Geometry channels have no background term.
                state_z[k] = output[base + 4];
                state_nx[k] = output[base + 6];
                state_ny[k] = output[base + 7];
                state_nz[k] = output[base + 8];
                state_db[k] = output[base + 9];
                state_dc[k] = output[base + 10];
                med_id[k] = u32::reinterpret(output[base + 11]);
                v_o_db[k] = v_output[base + 9];
                v_o_dc[k] = v_output[base + 10];
                v_o_z[k] = v_output[base + 4];
                v_o_med[k] = v_output[base + 5];
                v_o_nx[k] = v_output[base + 6];
                v_o_ny[k] = v_output[base + 7];
                v_o_nz[k] = v_output[base + 8];
            }
        }
    }

    let (range_lo, range_hi) = load_range(tile_offsets, tile_id);
    let mut batch = Shared::new_slice((BATCH * PROJECTED_LANES) as usize);
    let mut batch_gid = Shared::new_slice(BATCH as usize);
    let mut batch_aux =
        Shared::new_slice(comptime![if surfel { BATCH * AUX_LANES_SURFEL } else { 1u32 }] as usize);
    // Per-subgroup partial sums, [warp][splat][component], summed across
    // subgroups once per batch so each splat gets one atomic set per tile.
    let mut partials = Shared::new_slice(
        comptime![if xor32 { NUM_WARPS * BATCH * grad_lanes(surfel) } else { 1u32 }] as usize,
    );
    let warp = UNIT_POS / 32u32;

    let mut batch_start = range_lo;
    while batch_start < range_hi {
        let remaining = min(BATCH, range_hi - batch_start);
        if UNIT_POS < remaining {
            let compact_gid = compact_gid_from_isect[(batch_start + UNIT_POS) as usize];
            batch_gid[UNIT_POS as usize] = compact_gid;
            let src = (compact_gid * PROJECTED_LANES) as usize;
            let dst = (UNIT_POS * PROJECTED_LANES) as usize;
            #[unroll]
            for lane in 0..PROJECTED_LANES_USIZE {
                batch[dst + lane] = projected[src + lane];
            }
            if comptime![surfel] {
                let a_src = (compact_gid * AUX_LANES_SURFEL) as usize;
                let a_dst = (UNIT_POS * AUX_LANES_SURFEL) as usize;
                #[unroll]
                for lane in 0..comptime![AUX_LANES_SURFEL as usize] {
                    batch_aux[a_dst + lane] = splat_aux[a_src + lane];
                }
            }
        }
        sync_cube();

        let mut t = 0u32;
        while t < remaining {
            if comptime![surfel] {
                let b = (t * PROJECTED_LANES) as usize;
                let color_a = batch[b + 5];
                let color_r = batch[b + 6];
                let color_g = batch[b + 7];
                let color_b = batch[b + 8];
                let clamped_r = max(color_r, 0.0f32);
                let clamped_g = max(color_g, 0.0f32);
                let clamped_b = max(color_b, 0.0f32);
                let ab = (t * AUX_LANES_SURFEL) as usize;
                let r0x = batch_aux[ab];
                let r0y = batch_aux[ab + 1];
                let r0z = batch_aux[ab + 2];
                let r1x = batch_aux[ab + 3];
                let r1y = batch_aux[ab + 4];
                let r1z = batch_aux[ab + 5];
                let r2x = batch_aux[ab + 6];
                let r2y = batch_aux[ab + 7];
                let r2z = batch_aux[ab + 8];
                let ac = ab + AUX_SURFEL_CENTER as usize;
                let cen_x = batch_aux[ac];
                let cen_y = batch_aux[ac + 1];
                let an = ab + AUX_SURFEL_NORMAL as usize;
                let nrm_x = batch_aux[an];
                let nrm_y = batch_aux[an + 1];
                let nrm_z = batch_aux[an + 2];
                let inv_r2z = 1.0f32 / r2z;
                let sigma_max = f32::ln(color_a / (ALPHA_CUTOFF_MID - 0.5f32 * ALPHA_CUTOFF_BAND))
                    + 1.0e-3f32;
                let isect = batch_start + t;

                // [M rows(9), normal(3), rgb(3), alpha, refine].
                let mut gs = Array::<f32>::new(17usize);
                #[unroll]
                for i in 0..17usize {
                    gs[i] = 0.0f32;
                }
                let mut contributes = false;

                #[unroll]
                for k in 0..PIXELS_PER_THREAD {
                    let state_w = state_t[k];
                    let px = pixel_coord_x;
                    let py = (first_pix_y + (k as u32) * ROW_STRIDE) as f32 + 0.5f32;
                    // Forward: (u, v) = (q.x, q.y) / q.z, q = h_u x h_v,
                    // h_u = M0 - x M2, h_v = M1 - y M2; z = M2 . (u, v, 1).
                    let hu_x = r0x - px * r2x;
                    let hu_y = r0y - px * r2y;
                    let hu_z = r0z - px * r2z;
                    let hv_x = r1x - py * r2x;
                    let hv_y = r1y - py * r2y;
                    let hv_z = r1z - py * r2z;
                    let q_x = hu_y * hv_z - hu_z * hv_y;
                    let q_y = hu_z * hv_x - hu_x * hv_z;
                    let q_z = hu_x * hv_y - hu_y * hv_x;
                    let su = q_x / q_z;
                    let sv = q_y / q_z;
                    let s3d = su * su + sv * sv;
                    let z = r2x * su + r2y * sv + r2z;
                    let dxs = px - cen_x;
                    let dys = py - cen_y;
                    let s2d = 2.0f32 * (dxs * dxs + dys * dys);
                    let use_3d = s3d <= s2d;
                    let sigma = 0.5f32 * select(use_3d, s3d, s2d);
                    let valid = is_finite_f32(z) && z >= SURFEL_NEAR;
                    if state_w > 1.0e-4f32 && valid && sigma <= sigma_max {
                        let gaussian = f32::exp(-sigma);
                        let alpha = min(0.999f32, color_a * gaussian);
                        let w_cut = if comptime![smooth_cutoff] {
                            alpha_cutoff_weight(alpha)
                        } else {
                            select(alpha >= ALPHA_CUTOFF_MID, 1.0f32, 0.0f32)
                        };
                        if sigma >= 0.0f32 && w_cut > 0.0f32 {
                            let alpha_eff = alpha * w_cut;
                            let next_t = state_w * (1.0f32 - alpha_eff);
                            if next_t <= 1.0e-4f32 {
                                state_t[k] = 0.0f32;
                            } else {
                                contributes = true;
                                let vr = v_o_r[k];
                                let vg = v_o_g[k];
                                let vb = v_o_b[k];
                                let vz = v_o_z[k];
                                let vnx = v_o_nx[k];
                                let vny = v_o_ny[k];
                                let vnz = v_o_nz[k];
                                let sr = state_r[k];
                                let sg = state_g[k];
                                let sb = state_b[k];
                                let sz = state_z[k];
                                let snx = state_nx[k];
                                let sny = state_ny[k];
                                let snz = state_nz[k];
                                let vdb = v_o_db[k];
                                let vdc = v_o_dc[k];
                                let sdb = state_db[k];
                                let sdc = state_dc[k];
                                // Distortion channels composite m(z), m(z)^2.
                                let m = dist_depth(z);
                                let dm_dz = select(
                                    z > DIST_NEAR,
                                    DIST_FAR / (DIST_FAR - DIST_NEAR) * DIST_NEAR / (z * z),
                                    0.0f32,
                                );
                                let vis = alpha_eff * state_w;
                                gs[12] += select(color_r >= 0.0f32, vis * vr, 0.0f32);
                                gs[13] += select(color_g >= 0.0f32, vis * vg, 0.0f32);
                                gs[14] += select(color_b >= 0.0f32, vis * vb, 0.0f32);
                                gs[9] += vis * vnx;
                                gs[10] += vis * vny;
                                gs[11] += vis * vnz;

                                // Depth and normal composite like extra colour
                                // channels without a background term.
                                let ra = 1.0f32 / (1.0f32 - alpha_eff);
                                let dot = ((state_w * clamped_r - sr) * vr
                                    + (state_w * clamped_g - sg) * vg
                                    + (state_w * clamped_b - sb) * vb
                                    + (state_w * z - sz) * vz
                                    + (state_w * nrm_x - snx) * vnx
                                    + (state_w * nrm_y - sny) * vny
                                    + (state_w * nrm_z - snz) * vnz
                                    + (state_w * m - sdb) * vdb
                                    + (state_w * m * m - sdc) * vdc)
                                    * ra;
                                let v_alpha_eff = dot + v_o_t[k] * ra;
                                let v_alpha = if comptime![smooth_cutoff] {
                                    let dw_dalpha = alpha_cutoff_weight_deriv(alpha);
                                    v_alpha_eff * (w_cut + alpha * dw_dalpha)
                                } else {
                                    v_alpha_eff
                                };
                                // The median depth is this contributor's z.
                                let v_zval = vis * (vz + (vdb + 2.0f32 * m * vdc) * dm_dz)
                                    + select(isect == med_id[k], v_o_med[k], 0.0f32);
                                // Suppress the alpha-saturated footprint/opacity
                                // terms, as the gaussian path does.
                                let unsat = color_a * gaussian <= 0.999f32;
                                let v_sigma = select(unsat, -alpha * v_alpha, 0.0f32);
                                if unsat {
                                    gs[15] += v_alpha * gaussian;
                                }

                                let v_su = select(use_3d, v_sigma * su, 0.0f32) + v_zval * r2x;
                                let v_sv = select(use_3d, v_sigma * sv, 0.0f32) + v_zval * r2y;
                                let v_cx = select(use_3d, 0.0f32, -2.0f32 * v_sigma * dxs);
                                let v_cy = select(use_3d, 0.0f32, -2.0f32 * v_sigma * dys);
                                let inv_qz = 1.0f32 / q_z;
                                let vq_x = v_su * inv_qz;
                                let vq_y = v_sv * inv_qz;
                                let vq_z = -(v_su * su + v_sv * sv) * inv_qz;
                                // q = h_u x h_v: v_hu = h_v x v_q, v_hv = v_q x h_u.
                                let vhu_x = hv_y * vq_z - hv_z * vq_y;
                                let vhu_y = hv_z * vq_x - hv_x * vq_z;
                                let vhu_z = hv_x * vq_y - hv_y * vq_x;
                                let vhv_x = vq_y * hu_z - vq_z * hu_y;
                                let vhv_y = vq_z * hu_x - vq_x * hu_z;
                                let vhv_z = vq_x * hu_y - vq_y * hu_x;
                                // Low-pass center = (M02, M12) / M22.
                                let g_r0x = vhu_x;
                                let g_r0y = vhu_y;
                                let g_r0z = vhu_z + v_cx * inv_r2z;
                                let g_r1x = vhv_x;
                                let g_r1y = vhv_y;
                                let g_r1z = vhv_z + v_cy * inv_r2z;
                                let g_r2x = v_zval * su - px * vhu_x - py * vhv_x;
                                let g_r2y = v_zval * sv - px * vhu_y - py * vhv_y;
                                let g_r2z = v_zval - px * vhu_z - py * vhv_z
                                    - (v_cx * cen_x + v_cy * cen_y) * inv_r2z;
                                gs[0] += g_r0x;
                                gs[1] += g_r0y;
                                gs[2] += g_r0z;
                                gs[3] += g_r1x;
                                gs[4] += g_r1y;
                                gs[5] += g_r1z;
                                gs[6] += g_r2x;
                                gs[7] += g_r2y;
                                gs[8] += g_r2z;
                                // Refinement weight: this pixel's gradient
                                // w.r.t. a screen-space shift of the surfel
                                // (M -> T M, T a pixel translation).
                                let shift_x = g_r0x * r2x + g_r0y * r2y + g_r0z * r2z;
                                let shift_y = g_r1x * r2x + g_r1y * r2y + g_r1z * r2z;
                                gs[16] += f32::sqrt(
                                    shift_x * img_size_x * shift_x * img_size_x
                                        + shift_y * img_size_y * shift_y * img_size_y,
                                ) * refine_scale[k];

                                state_r[k] = sr - vis * clamped_r;
                                state_g[k] = sg - vis * clamped_g;
                                state_b[k] = sb - vis * clamped_b;
                                state_z[k] = sz - vis * z;
                                state_nx[k] = snx - vis * nrm_x;
                                state_ny[k] = sny - vis * nrm_y;
                                state_nz[k] = snz - vis * nrm_z;
                                state_db[k] = sdb - vis * m;
                                state_dc[k] = sdc - vis * m * m;
                                state_t[k] = next_t;
                            }
                        }
                    }
                }

                if comptime![xor32] {
                    let slot = ((warp * BATCH + t) * n_grads) as usize;
                    if plane_any(contributes) {
                        // The first 16 components through the butterfly,
                        // the refine weight through one plane_sum.
                        let refine = plane_sum(gs[16]);
                        let comp = reduce_scatter_32(&mut gs);
                        if (UNIT_POS_PLANE & 1u32) == 0u32 {
                            partials[slot + comp as usize] = gs[0];
                        }
                        if UNIT_POS_PLANE == 1u32 {
                            partials[slot + 16] = refine;
                        }
                    } else if UNIT_POS_PLANE < n_grads {
                        partials[slot + UNIT_POS_PLANE as usize] = 0.0f32;
                    }
                } else if plane_any(contributes) {
                    let base = (batch_gid[t as usize] * n_grads) as usize;
                    #[unroll]
                    for c in 0..comptime![GRADS_SURFEL as usize] {
                        let sum = plane_sum(gs[c]);
                        if UNIT_POS_PLANE == 0u32 {
                            A::add(&v_splats[base + c], sum);
                        }
                    }
                }
            } else {
            let b = (t * PROJECTED_LANES) as usize;
            let xy_x = batch[b];
            let xy_y = batch[b + 1];
            let c00 = batch[b + 2];
            let c01 = batch[b + 3];
            let c11 = batch[b + 4];
            let color_a = batch[b + 5];
            let color_r = batch[b + 6];
            let color_g = batch[b + 7];
            let color_b = batch[b + 8];
            let clamped_r = max(color_r, 0.0f32);
            let clamped_g = max(color_g, 0.0f32);
            let clamped_b = max(color_b, 0.0f32);
            let dx = xy_x - pixel_coord_x;
            // Below the low edge of the cutoff band every pixel is rejected,
            // so skip its exp. The margin keeps the exact test authoritative.
            let sigma_max = f32::ln(color_a / (ALPHA_CUTOFF_MID - 0.5f32 * ALPHA_CUTOFF_BAND))
                + 1.0e-3f32;

            let mut g_xy_x = 0.0f32;
            let mut g_xy_y = 0.0f32;
            let mut g_conic_x = 0.0f32;
            let mut g_conic_y = 0.0f32;
            let mut g_conic_z = 0.0f32;
            let mut g_rgb_r = 0.0f32;
            let mut g_rgb_g = 0.0f32;
            let mut g_rgb_b = 0.0f32;
            let mut g_alpha = 0.0f32;
            let mut g_refine = 0.0f32;
            let mut contributes = false;

            #[unroll]
            for k in 0..PIXELS_PER_THREAD {
                let state_w = state_t[k];
                let pix_y = first_pix_y + (k as u32) * ROW_STRIDE;
                let dy = xy_y - (pix_y as f32 + 0.5f32);
                let sigma = 0.5f32 * (c00 * dx * dx + c11 * dy * dy) + c01 * dx * dy;
                if state_w > 1.0e-4f32 && sigma <= sigma_max {
                    let gaussian = f32::exp(-sigma);
                    let alpha = min(0.999f32, color_a * gaussian);
                    let w_cut = if comptime![smooth_cutoff] {
                        alpha_cutoff_weight(alpha)
                    } else {
                        select(alpha >= ALPHA_CUTOFF_MID, 1.0f32, 0.0f32)
                    };
                    if sigma >= 0.0f32 && w_cut > 0.0f32 {
                        let alpha_eff = alpha * w_cut;
                        let next_t = state_w * (1.0f32 - alpha_eff);
                        if next_t <= 1.0e-4f32 {
                            state_t[k] = 0.0f32;
                        } else {
                            contributes = true;
                            let vr = v_o_r[k];
                            let vg = v_o_g[k];
                            let vb = v_o_b[k];
                            let sr = state_r[k];
                            let sg = state_g[k];
                            let sb = state_b[k];
                            let vis = alpha_eff * state_w;
                            // Gate the rgb VJP on the original (pre-clamp)
                            // sign: clamped values contribute no gradient.
                            g_rgb_r += select(color_r >= 0.0f32, vis * vr, 0.0f32);
                            g_rgb_g += select(color_g >= 0.0f32, vis * vg, 0.0f32);
                            g_rgb_b += select(color_b >= 0.0f32, vis * vb, 0.0f32);

                            let ra = 1.0f32 / (1.0f32 - alpha_eff);
                            let dot_rgb = ((state_w * clamped_r - sr) * vr
                                + (state_w * clamped_g - sg) * vg
                                + (state_w * clamped_b - sb) * vb)
                                * ra;
                            let v_alpha_eff = dot_rgb + v_o_t[k] * ra;
                            let v_alpha = if comptime![smooth_cutoff] {
                                let dw_dalpha = alpha_cutoff_weight_deriv(alpha);
                                v_alpha_eff * (w_cut + alpha * dw_dalpha)
                            } else {
                                v_alpha_eff
                            };
                            let v_sigma = -alpha * v_alpha;
                            let vxy_x = v_sigma * (c00 * dx + c01 * dy);
                            let vxy_y = v_sigma * (c01 * dx + c11 * dy);

                            // Suppress the alpha-saturated gradient term: at
                            // the cap the alpha derivative flattens.
                            if color_a * gaussian <= 0.999f32 {
                                g_conic_x += 0.5f32 * v_sigma * dx * dx;
                                g_conic_y += v_sigma * dx * dy;
                                g_conic_z += 0.5f32 * v_sigma * dy * dy;
                                g_xy_x += vxy_x;
                                g_xy_y += vxy_y;
                                g_alpha += v_alpha * gaussian;
                                let len = f32::sqrt(
                                    vxy_x * img_size_x * vxy_x * img_size_x
                                        + vxy_y * img_size_y * vxy_y * img_size_y,
                                );
                                g_refine += len * refine_scale[k];
                            }

                            state_r[k] = sr - vis * clamped_r;
                            state_g[k] = sg - vis * clamped_g;
                            state_b[k] = sb - vis * clamped_b;
                            state_t[k] = next_t;
                        }
                    }
                }
            }

            // Subgroup-uniform branch: every lane of a subgroup sees the
            // same vote, so the reductions below run in uniform control flow.
            if comptime![xor32] {
                let slot = ((warp * BATCH + t) * 10u32) as usize;
                if plane_any(contributes) {
                    let mut g = Array::<f32>::new(16usize);
                    g[0] = g_xy_x;
                    g[1] = g_xy_y;
                    g[2] = g_conic_x;
                    g[3] = g_conic_y;
                    g[4] = g_conic_z;
                    g[5] = g_rgb_r;
                    g[6] = g_rgb_g;
                    g[7] = g_rgb_b;
                    g[8] = g_alpha;
                    g[9] = g_refine;
                    #[unroll]
                    for i in 10..16usize {
                        g[i] = 0.0f32;
                    }
                    let comp = reduce_scatter_32(&mut g);
                    if (UNIT_POS_PLANE & 1u32) == 0u32 && comp < 10u32 {
                        partials[slot + comp as usize] = g[0];
                    }
                } else if UNIT_POS_PLANE < 10u32 {
                    partials[slot + UNIT_POS_PLANE as usize] = 0.0f32;
                }
            } else if plane_any(contributes) {
                let s_xy_x = plane_sum(g_xy_x);
                let s_xy_y = plane_sum(g_xy_y);
                let s_conic_x = plane_sum(g_conic_x);
                let s_conic_y = plane_sum(g_conic_y);
                let s_conic_z = plane_sum(g_conic_z);
                let s_rgb_r = plane_sum(g_rgb_r);
                let s_rgb_g = plane_sum(g_rgb_g);
                let s_rgb_b = plane_sum(g_rgb_b);
                let s_alpha = plane_sum(g_alpha);
                let s_refine = plane_sum(g_refine);
                if UNIT_POS_PLANE == 0u32 {
                    let base = (batch_gid[t as usize] * 10u32) as usize;
                    A::add(&v_splats[base], s_xy_x);
                    A::add(&v_splats[base + 1], s_xy_y);
                    A::add(&v_splats[base + 2], s_conic_x);
                    A::add(&v_splats[base + 3], s_conic_y);
                    A::add(&v_splats[base + 4], s_conic_z);
                    A::add(&v_splats[base + 5], s_rgb_r);
                    A::add(&v_splats[base + 6], s_rgb_g);
                    A::add(&v_splats[base + 7], s_rgb_b);
                    A::add(&v_splats[base + 8], s_alpha);
                    A::add(&v_splats[base + 9], s_refine);
                }
            }
            }
            t += 1u32;
        }
        if comptime![xor32] {
            // Sum the subgroup partials; a splat no subgroup touched sums to
            // exactly zero and is skipped.
            sync_cube();
            let mut idx = UNIT_POS;
            while idx < remaining * n_grads {
                let splat = idx / n_grads;
                let comp = idx % n_grads;
                let mut sum = 0.0f32;
                #[unroll]
                for w in 0..NUM_WARPS {
                    sum += partials[((w * BATCH + splat) * n_grads + comp) as usize];
                }
                if sum != 0.0f32 {
                    A::add(&v_splats[(batch_gid[splat as usize] * n_grads + comp) as usize], sum);
                }
                idx += WG_SIZE;
            }
        }
        // Keep this batch's shared reads ahead of the next batch's writes.
        sync_cube();
        batch_start += BATCH;
    }
}

/// Butterfly reduce-scatter of 16 per-lane values across a 32-lane
/// subgroup: 16 shuffles instead of 16 separate 5-step reductions. Each
/// halving step keeps the half selected by one lane bit and adds the
/// partner's copy of it. Returns the component this lane now holds the full
/// sum of (in `v[0]`); lanes `2c` and `2c + 1` both hold component `c`.
#[cube]
fn reduce_scatter_32(v: &mut Array<f32>) -> u32 {
    let lane = UNIT_POS_PLANE;
    let hi = (lane & 16u32) != 0u32;
    #[unroll]
    for i in 0..8usize {
        let keep = select(hi, v[i + 8], v[i]);
        let send = select(hi, v[i], v[i + 8]);
        v[i] = keep + plane_shuffle_xor(send, 16u32);
    }
    let hi = (lane & 8u32) != 0u32;
    #[unroll]
    for i in 0..4usize {
        let keep = select(hi, v[i + 4], v[i]);
        let send = select(hi, v[i], v[i + 4]);
        v[i] = keep + plane_shuffle_xor(send, 8u32);
    }
    let hi = (lane & 4u32) != 0u32;
    #[unroll]
    for i in 0..2usize {
        let keep = select(hi, v[i + 2], v[i]);
        let send = select(hi, v[i], v[i + 2]);
        v[i] = keep + plane_shuffle_xor(send, 4u32);
    }
    let hi = (lane & 2u32) != 0u32;
    let keep = select(hi, v[1], v[0]);
    let send = select(hi, v[0], v[1]);
    v[0] = keep + plane_shuffle_xor(send, 2u32);
    v[0] += plane_shuffle_xor(v[0], 1u32);
    (lane >> 1u32) & 15u32
}
