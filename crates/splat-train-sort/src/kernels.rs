use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::frontend::CompilationArg;
use burn::cubecl::frontend::IndexMutExpand;
use burn::cubecl::prelude::*;

pub use brush_scan::{BLOCK_SIZE, ELEMENTS_PER_THREAD, WG};
use brush_scan::{block_scan, cube_exclusive_sum, cube_sum, lds_index};

pub const WG_USIZE: usize = WG as usize;
// 8-bit digits: half the passes (and dispatches) of 4-bit ones. The local
// histogram scan needs one thread per bin, so `WG` must cover `BIN_COUNT`.
pub const BITS_PER_PASS: u32 = 8;
pub const BIN_COUNT: u32 = 1 << BITS_PER_PASS;
pub const BIN_COUNT_USIZE: usize = BIN_COUNT as usize;
const DIGIT_MASK: u32 = BIN_COUNT - 1;
const _: () = assert!(WG >= BIN_COUNT);
/// Next-pass counters in the scatter: digit run x two partitions x next digit.
const NEXT_HISTOGRAM_LEN: usize = BIN_COUNT_USIZE * 2 * BIN_COUNT_USIZE;
/// Counting the next pass in the scatter needs those counters in shared
/// memory. That fits 4-bit digits (2 KiB) but not 8-bit ones (512 KiB, far
/// past the 16 KiB WebGPU guarantee), so 8-bit sorts count every pass.
pub const SCATTER_COUNT: bool = NEXT_HISTOGRAM_LEN * 4 <= 8192;
/// Sorts of at most this many partitions count each next pass in the
/// scatter. Its global atomics cost more than a count dispatch on big sorts.
pub const SCATTER_COUNT_MAX_WGS: u32 = 512;

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1u32) / b
}

#[cube(launch)]
pub fn sort_count_kernel(
    num_keys_arr: &Tensor<u32>,
    src: &Tensor<u32>,
    counts: &mut Tensor<u32>,
    shift: u32,
) {
    let num_keys = num_keys_arr[0];

    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let group_id = CUBE_POS as u32;

    if group_id >= num_wgs {
        terminate!();
    }

    let histogram = Shared::<[Atomic<u32>]>::new_slice(BIN_COUNT_USIZE);
    if UNIT_POS < BIN_COUNT {
        Atomic::store(&histogram[UNIT_POS as usize], 0u32);
    }
    sync_cube();

    let wg_block_start = BLOCK_SIZE * group_id;
    let mut data_index = wg_block_start + UNIT_POS;

    for _ in 0u32..ELEMENTS_PER_THREAD {
        if data_index < num_keys {
            let local_key = (src[data_index as usize] >> shift) & DIGIT_MASK;
            Atomic::fetch_add(&histogram[local_key as usize], 1u32);
        }
        data_index += WG;
    }
    sync_cube();
    if UNIT_POS < BIN_COUNT {
        let nw = div_ceil(num_keys, BLOCK_SIZE);
        counts[(UNIT_POS * nw + group_id) as usize] = Atomic::load(&histogram[UNIT_POS as usize]);
    }
}

#[cube(launch)]
pub fn sort_reduce_kernel(
    num_keys_arr: &Tensor<u32>,
    counts: &Tensor<u32>,
    reduced: &mut Tensor<u32>,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let group_id = CUBE_POS as u32;
    if group_id >= num_reduce_wgs {
        terminate!();
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;
    let bin_id = group_id / num_reduce_wg_per_bin;

    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * BLOCK_SIZE;

    let mut sum = 0u32;
    for i in 0u32..ELEMENTS_PER_THREAD {
        let data_index = base_index + i * WG + UNIT_POS;
        if data_index < num_wgs {
            sum += counts[(bin_offset + data_index) as usize];
        }
    }

    let total = cube_sum(sum);
    if UNIT_POS == 0u32 {
        reduced[group_id as usize] = total;
    }
}

#[cube(launch)]
pub fn sort_scan_kernel(num_keys_arr: &Tensor<u32>, reduced: &mut Tensor<u32>) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let mut lds = Shared::new_slice((WG * ELEMENTS_PER_THREAD) as usize);

    let mut carry = 0u32;
    let mut chunk_start = 0u32;
    while chunk_start < num_reduce_wgs {
        for i in 0u32..ELEMENTS_PER_THREAD {
            let lin = i * WG + UNIT_POS;
            let data_index = chunk_start + lin;
            let mut v = 0u32;
            if data_index < num_reduce_wgs {
                v = reduced[data_index as usize];
            }
            lds[lds_index(lin) as usize] = v;
        }
        sync_cube();

        let chunk_total = block_scan(&mut lds, carry, false);
        sync_cube();

        for i in 0u32..ELEMENTS_PER_THREAD {
            let lin = i * WG + UNIT_POS;
            let data_index = chunk_start + lin;
            if data_index < num_reduce_wgs {
                reduced[data_index as usize] = lds[lds_index(lin) as usize];
            }
        }
        // Also orders this chunk's scan reads ahead of the next chunk's writes.
        sync_cube();

        carry += chunk_total;
        chunk_start += BLOCK_SIZE;
    }
}

#[cube(launch)]
pub fn sort_scan_add_kernel(
    num_keys_arr: &Tensor<u32>,
    reduced: &Tensor<u32>,
    counts: &mut Tensor<u32>,
    next_counts: &mut Tensor<u32>,
    #[comptime] clear_next: bool,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let group_id = CUBE_POS as u32;
    if group_id >= num_reduce_wgs {
        terminate!();
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;
    let bin_id = group_id / num_reduce_wg_per_bin;
    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * ELEMENTS_PER_THREAD * WG;

    let mut lds = Shared::new_slice((WG * ELEMENTS_PER_THREAD) as usize);

    for i in 0u32..ELEMENTS_PER_THREAD {
        let lin = i * WG + UNIT_POS;
        let data_index = base_index + lin;
        // Gate explicitly so a stray OOB read is impossible regardless of
        // whether the backend implements robust-access clamping.
        let mut v = 0u32;
        if data_index < num_wgs {
            v = counts[(bin_offset + data_index) as usize];
            // The scatter accumulates the next pass's counts into this.
            if clear_next {
                next_counts[(bin_offset + data_index) as usize] = 0u32;
            }
        }
        lds[lds_index(lin) as usize] = v;
    }
    sync_cube();

    block_scan(&mut lds, reduced[group_id as usize], false);
    sync_cube();

    for i in 0u32..ELEMENTS_PER_THREAD {
        let lin = i * WG + UNIT_POS;
        let data_index = base_index + lin;
        if data_index < num_wgs {
            counts[(bin_offset + data_index) as usize] = lds[lds_index(lin) as usize];
        }
    }
}

#[cube(launch)]
pub fn sort_scatter_kernel(
    num_keys_arr: &Tensor<u32>,
    src: &Tensor<u32>,
    values: &Tensor<u32>,
    counts: &Tensor<u32>,
    out: &mut Tensor<u32>,
    out_values: &mut Tensor<u32>,
    next_counts: &mut Tensor<Atomic<u32>>,
    shift: u32,
    #[comptime] count_next: bool,
) {
    let num_keys = num_keys_arr[0];
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);

    let group_id = CUBE_POS as u32;
    if group_id >= num_wgs {
        terminate!();
    }

    let mut lds_keys = Shared::new_slice(WG_USIZE);
    let mut lds_values = Shared::new_slice(WG_USIZE);
    let mut lds_scratch = Shared::new_slice(WG_USIZE);
    let mut bin_offset_cache = Shared::new_slice(WG_USIZE);
    let local_histogram = Shared::<[Atomic<u32>]>::new_slice(BIN_COUNT_USIZE);
    // This cube's keys of one digit land in one contiguous run of at most
    // BLOCK_SIZE outputs, so the run touches at most two partitions of the
    // next pass. Count next-pass digits per (digit run, partition) here and
    // flush once, instead of a separate count dispatch over the output.
    let next_histogram = Shared::<[Atomic<u32>]>::new_slice(comptime!(if count_next {
        NEXT_HISTOGRAM_LEN
    } else {
        1usize
    }));
    let mut run_partition = Shared::new_slice(BIN_COUNT_USIZE);

    if UNIT_POS < BIN_COUNT {
        let run_start = counts[(UNIT_POS * num_wgs + group_id) as usize];
        bin_offset_cache[UNIT_POS as usize] = run_start;
        run_partition[UNIT_POS as usize] = run_start / BLOCK_SIZE;
    }
    if count_next {
        let mut i = UNIT_POS;
        while i < NEXT_HISTOGRAM_LEN as u32 {
            Atomic::store(&next_histogram[i as usize], 0u32);
            i += WG;
        }
    }
    sync_cube();
    let wg_block_start = BLOCK_SIZE * group_id;
    let block_index = wg_block_start + UNIT_POS;
    let mut data_index = block_index;
    for _ in 0u32..ELEMENTS_PER_THREAD {
        if UNIT_POS < BIN_COUNT {
            Atomic::store(&local_histogram[UNIT_POS as usize], 0u32);
        }

        let mut local_key = 0xFFFFFFFFu32;
        let mut local_value = 0u32;

        if data_index < num_keys {
            local_key = src[data_index as usize];
            local_value = values[data_index as usize];
        }

        let mut bit_shift = 0u32;
        while bit_shift < BITS_PER_PASS {
            let key_index = (local_key >> shift) & DIGIT_MASK;
            let bit_key = (key_index >> bit_shift) & 3u32;
            let packed_input = 1u32 << (bit_key * 8u32);

            // The packed per-bit-pair counters scan as plain u32 adds.
            let (exclusive_at_thread, total) = cube_exclusive_sum(packed_input);
            let bin_offsets = (total << 8u32) + (total << 16u32) + (total << 24u32);
            let local_sum = bin_offsets + exclusive_at_thread;
            let key_offset = (local_sum >> (bit_key * 8u32)) & 0xffu32;

            lds_keys[key_offset as usize] = local_key;
            lds_values[key_offset as usize] = local_value;
            sync_cube();
            local_key = lds_keys[UNIT_POS as usize];
            local_value = lds_values[UNIT_POS as usize];

            bit_shift += 2u32;
        }

        let key_index = (local_key >> shift) & DIGIT_MASK;
        Atomic::fetch_add(&local_histogram[key_index as usize], 1u32);
        sync_cube();

        // Inclusive scan of the bin counts, one thread per bin.
        let mut bin_count = 0u32;
        if UNIT_POS < BIN_COUNT {
            bin_count = Atomic::load(&local_histogram[UNIT_POS as usize]);
        }
        let (bin_exclusive, _) = cube_exclusive_sum(bin_count);
        if UNIT_POS < BIN_COUNT {
            lds_scratch[UNIT_POS as usize] = bin_exclusive + bin_count;
        }
        sync_cube();
        let global_offset = bin_offset_cache[key_index as usize];
        sync_cube();
        let mut local_offset = UNIT_POS;
        if key_index > 0u32 {
            local_offset -= lds_scratch[(key_index - 1u32) as usize];
        }
        let total_offset = global_offset + local_offset;
        if total_offset < num_keys {
            out[total_offset as usize] = local_key;
            out_values[total_offset as usize] = local_value;
            if count_next {
                let partition = total_offset / BLOCK_SIZE - run_partition[key_index as usize];
                let next_key = (local_key >> (shift + BITS_PER_PASS)) & DIGIT_MASK;
                Atomic::fetch_add(
                    &next_histogram[((key_index * 2u32 + partition) * BIN_COUNT + next_key) as usize],
                    1u32,
                );
            }
        }
        if UNIT_POS < BIN_COUNT {
            bin_offset_cache[UNIT_POS as usize] +=
                Atomic::load(&local_histogram[UNIT_POS as usize]);
        }
        sync_cube();
        data_index += WG;
    }

    if count_next {
        let mut i = UNIT_POS;
        while i < NEXT_HISTOGRAM_LEN as u32 {
            let c = Atomic::load(&next_histogram[i as usize]);
            if c > 0u32 {
                let run = i / (2u32 * BIN_COUNT);
                let partition = run_partition[run as usize] + (i / BIN_COUNT) % 2u32;
                let next_key = i % BIN_COUNT;
                Atomic::fetch_add(&next_counts[(next_key * num_wgs + partition) as usize], c);
            }
            i += WG;
        }
    }
}
