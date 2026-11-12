//! Map per-splat tile counts to per-intersection (tile_id, compact_gid)
//! pairs.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

use brush_scan::{BLOCK_SIZE, ELEMENTS_PER_THREAD, WG, block_scan, lds_index};

use super::helpers::{
    compute_bbox_extent, get_tile_bbox, read_main_splat, tile_rect, will_primitive_contribute,
};
use super::types::Sym2;

pub const WG_SIZE: u32 = 256;

/// Bounding boxes up to this many tiles are walked by their own lane; larger
/// ones by the whole subgroup, so one big splat no longer stalls its warp.
pub const SMALL_BBOX_TILES: u32 = 16;

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn map_gaussians_to_intersect_kernel(
    projected: &Tensor<f32>,
    compact_from_sorted: &Tensor<u32>,
    cum_hits_local: &Tensor<u32>,
    cum_hits_blocks: &Tensor<u32>,
    tile_id_from_isect: &mut Tensor<u32>,
    compact_gid_from_isect: &mut Tensor<u32>,
    tile_bw: u32,
    tile_bh: u32,
    num_visible: &Tensor<u32>,
    bounded_intersections: &mut Tensor<u32>,
    #[comptime] planes: bool,
) {
    // Threads walk the splats in depth order (`sorted`), so each tile's
    // intersections come out front to back after the stable tile sort; the
    // ids they emit are the compact (pre-sort) ids the projected rows use.
    let sorted = ABSOLUTE_POS as u32;
    // Training renders validate counts a render late, so an overflow clamps:
    // intersections past the capacity (the farthest splats) are dropped.
    let capacity = tile_id_from_isect.len() as u32;
    if sorted == 0u32 {
        // The scan's grand total: every visible splat's tile count, the
        // same sum projection accumulated into its intersection counter.
        let total = cum_hits_blocks[cum_hits_blocks.len() - 1];
        bounded_intersections[0] = min(total, capacity);
    }
    let active = sorted < num_visible[0];
    if !planes {
        if !active {
            terminate!();
        }
    }

    // Tile id past the valid range — radix-sorts after every real tile
    // and lives outside `tile_offsets`, so the rasterize pass never
    // visits these padded slots.
    let sentinel_tile_id = tile_bw * tile_bh;

    let read_idx = select(active, sorted, 0u32);
    let compact_gid = compact_from_sorted[read_idx as usize];
    let (xy_x, xy_y, conic, opac) = read_main_splat(projected, compact_gid);
    let power_threshold = f32::ln(opac * 255.0f32);
    let (ex, ey) = compute_bbox_extent(conic, power_threshold);
    let bb = get_tile_bbox(xy_x, xy_y, ex, ey, tile_bw, tile_bh);

    // Inclusive prefix sum: use cum[compact_gid - 1] as base (or 0 for first).
    // Index with `max(read_idx, 1) - 1` so the read is always in-bounds.
    let prev_idx = max(read_idx, 1u32) - 1u32;
    let base_isect_id = select(
        read_idx == 0u32,
        0u32,
        cum_hits(cum_hits_local, cum_hits_blocks, prev_idx),
    );
    // Slot budget reserved for this splat in PF. PF ran the same
    // `will_primitive_contribute` walk, so the emission loop below should
    // count the same tiles — but the two dispatches go through separate
    // shader optimisation passes, so cap emission at `pf_count` and pad any
    // leftover budget (belt and suspenders).
    let pf_count = cum_hits(cum_hits_local, cum_hits_blocks, read_idx) - base_isect_id;

    let bb_w = bb.max_x - bb.min_x;
    let num_tiles_bbox = select(active, (bb.max_y - bb.min_y) * bb_w, 0u32);

    let alone_limit = comptime!(if planes { SMALL_BBOX_TILES } else { u32::MAX });
    if num_tiles_bbox <= alone_limit {
        if active {
            let mut num_tiles_hit = 0u32;
            for tile_idx in 0u32..num_tiles_bbox {
                let tx = (tile_idx % bb_w) + bb.min_x;
                let ty = (tile_idx / bb_w) + bb.min_y;
                let rect = tile_rect(tx, ty);
                if will_primitive_contribute(rect, xy_x, xy_y, conic, power_threshold)
                    && num_tiles_hit < pf_count
                {
                    let isect_id = base_isect_id + num_tiles_hit;
                    if isect_id < capacity {
                        tile_id_from_isect[isect_id as usize] = tx + ty * tile_bw;
                        compact_gid_from_isect[isect_id as usize] = compact_gid;
                    }
                    num_tiles_hit += 1u32;
                }
            }
            // Pad the leftover budget with sentinel rows so no slot in
            // `[base_isect_id, base_isect_id + pf_count)` is left uninitialised.
            for pad_idx in num_tiles_hit..pf_count {
                let isect_id = base_isect_id + pad_idx;
                if isect_id < capacity {
                    tile_id_from_isect[isect_id as usize] = sentinel_tile_id;
                    compact_gid_from_isect[isect_id as usize] = compact_gid;
                }
            }
        }
    }

    if planes {
        // Large boxes: the subgroup walks one splat at a time, each lane
        // testing every PLANE_DIM-th tile; hits are packed with a prefix sum.
        // A splat's tiles are distinct, so their order within it is free.
        let mut pending = active && num_tiles_bbox > SMALL_BBOX_TILES;
        while plane_any(pending) {
            let leader = plane_min(select(pending, UNIT_POS_PLANE, 0xFFFF_FFFFu32));
            let l_x = plane_shuffle(xy_x, leader);
            let l_y = plane_shuffle(xy_y, leader);
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
            let l_base = plane_shuffle(base_isect_id, leader);
            let l_count = plane_shuffle(pf_count, leader);
            let l_gid = plane_shuffle(compact_gid, leader);

            let mut emitted = 0u32;
            let mut t0 = 0u32;
            while t0 < l_n {
                let t = t0 + UNIT_POS_PLANE;
                let tx = (t % l_w) + l_min_x;
                let ty = (t / l_w) + l_min_y;
                let hit = t < l_n
                    && will_primitive_contribute(tile_rect(tx, ty), l_x, l_y, l_conic, l_power);
                let h = select(hit, 1u32, 0u32);
                let slot = emitted + plane_exclusive_sum(h);
                if hit && slot < l_count {
                    let isect_id = l_base + slot;
                    if isect_id < capacity {
                        tile_id_from_isect[isect_id as usize] = tx + ty * tile_bw;
                        compact_gid_from_isect[isect_id as usize] = l_gid;
                    }
                }
                emitted += plane_sum(h);
                t0 += PLANE_DIM;
            }
            let mut pad = min(emitted, l_count) + UNIT_POS_PLANE;
            while pad < l_count {
                let isect_id = l_base + pad;
                if isect_id < capacity {
                    tile_id_from_isect[isect_id as usize] = sentinel_tile_id;
                    compact_gid_from_isect[isect_id as usize] = l_gid;
                }
                pad += PLANE_DIM;
            }
            if UNIT_POS_PLANE == leader {
                pending = false;
            }
        }
    }
}

/// Before projection: zero its two counters and write `0..len`, the
/// pre-sort indices the depth sort permutes, in one launch.
#[cube(launch)]
pub fn presort_index_kernel(
    num_visible: &mut Tensor<u32>,
    num_intersections: &mut Tensor<u32>,
    out: &mut Tensor<u32>,
) {
    let i = ABSOLUTE_POS;
    if i == 0usize {
        num_visible[0] = 0u32;
        num_intersections[0] = 0u32;
    }
    if i >= out.len() {
        terminate!();
    }
    out[i] = i as u32;
}

/// Inclusive prefix sum of the per-splat tile counts in depth order at
/// `i`, from [`scan_sorted_counts_kernel`]'s block-local scan and the
/// scanned block totals.
#[cube]
fn cum_hits(local: &Tensor<u32>, blocks: &Tensor<u32>, i: u32) -> u32 {
    let block = i / BLOCK_SIZE;
    // Index with `max(block, 1) - 1` so the read is always in-bounds.
    let offset = blocks[(max(block, 1u32) - 1u32) as usize];
    local[i as usize] + select(block > 0u32, offset, 0u32)
}

/// Block-local level of the prefix sum that places each splat's
/// intersections: gathers the per-splat tile counts in depth order (zero
/// over the unused capacity, so the scan is valid without a host
/// visible-count readback), scans each [`BLOCK_SIZE`] block inclusively into
/// `local` and writes the block totals for `brush_scan::scan_in_place` to
/// scan. Also zero-fills `visible` and `tile_offsets` (grid-stride) for the
/// rasterizer, which runs after it.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn scan_sorted_counts_kernel(
    compact_from_sorted: &Tensor<u32>,
    global_from_compact_gid: &Tensor<u32>,
    intersect_counts: &Tensor<u32>,
    num_visible: &Tensor<u32>,
    local: &mut Tensor<u32>,
    block_totals: &mut Tensor<u32>,
    visible: &mut Tensor<f32>,
    tile_offsets: &mut Tensor<u32>,
) {
    let n = local.len() as u32;
    let visible_count = num_visible[0];
    let block = CUBE_POS as u32;
    let base = block * BLOCK_SIZE;

    let mut lds = Shared::new_slice(BLOCK_SIZE as usize);
    for i in 0u32..ELEMENTS_PER_THREAD {
        let lin = i * WG + UNIT_POS;
        let idx = base + lin;
        let mut v = 0u32;
        if idx < n && idx < visible_count {
            let g = global_from_compact_gid[compact_from_sorted[idx as usize] as usize];
            v = intersect_counts[g as usize];
        }
        lds[lds_index(lin) as usize] = v;
    }
    sync_cube();

    let total = block_scan(&mut lds, 0u32, true);
    sync_cube();

    for i in 0u32..ELEMENTS_PER_THREAD {
        let lin = i * WG + UNIT_POS;
        let idx = base + lin;
        if idx < n {
            local[idx as usize] = lds[lds_index(lin) as usize];
        }
    }
    if UNIT_POS == 0u32 && block < block_totals.len() as u32 {
        block_totals[block as usize] = total;
    }

    // Grid-stride zero-fills: plain stores, nothing reads these here.
    let stride = CUBE_COUNT * CUBE_DIM as usize;
    let mut vi = ABSOLUTE_POS;
    while vi < visible.len() {
        visible[vi] = 0.0f32;
        vi += stride;
    }
    let mut ti = ABSOLUTE_POS;
    while ti < tile_offsets.len() {
        tile_offsets[ti] = 0u32;
        ti += stride;
    }
}
