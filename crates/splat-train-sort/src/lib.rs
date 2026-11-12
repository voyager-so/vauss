mod kernels;

use brush_cube::CubeCount;
use brush_cube::create_tensor;
use brush_cube::create_tensor_from_slice;
use burn::backend::TensorMetadata;
use burn::cubecl::CubeDim;
use burn::cubecl::calculate_cube_count_elemwise;
use burn::tensor::DType;
use burn_wgpu::CubeTensor;

use kernels::{BIN_COUNT, BLOCK_SIZE, SCATTER_COUNT, SCATTER_COUNT_MAX_WGS, WG};

/// Perform a radix argsort on the input keys and values.
pub fn radix_argsort(
    input_keys: CubeTensor,
    input_values: CubeTensor,
    sorting_bits: u32,
) -> (CubeTensor, CubeTensor) {
    let max_n = input_keys.shape()[0] as u32;
    let num_keys = create_tensor_from_slice(&[max_n], &input_keys.device, DType::U32);
    radix_argsort_with_count(input_keys, input_values, sorting_bits, num_keys)
}

/// Sort the active prefix whose length is stored on the GPU. The input shapes
/// are capacities, and entries beyond the active prefix are never read.
/// The caller must guarantee `num_keys[0] <= input_keys.shape()[0]`.
pub fn radix_argsort_with_count(
    input_keys: CubeTensor,
    input_values: CubeTensor,
    sorting_bits: u32,
    num_keys_buf: CubeTensor,
) -> (CubeTensor, CubeTensor) {
    radix_argsort_skip_low_with_count(input_keys, input_values, 0, sorting_bits, num_keys_buf)
}

/// Like [`radix_argsort_with_count`], but sorts by key bits
/// `skip_bits..skip_bits + sorting_bits` only: keys that differ just in the
/// lowest `skip_bits` bits keep their input order.
pub fn radix_argsort_skip_low_with_count(
    input_keys: CubeTensor,
    input_values: CubeTensor,
    skip_bits: u32,
    sorting_bits: u32,
    num_keys_buf: CubeTensor,
) -> (CubeTensor, CubeTensor) {
    assert_eq!(
        input_keys.shape()[0],
        input_values.shape()[0],
        "Input keys and values must have the same number of elements"
    );
    assert!(skip_bits + sorting_bits <= 32, "Can only sort up to 32 bits");
    assert!(
        input_keys.is_contiguous(),
        "Please ensure input keys are contiguous"
    );
    assert!(
        input_values.is_contiguous(),
        "Please ensure input keys are contiguous"
    );

    let _span = tracing::trace_span!("Radix sort").entered();

    let client = input_keys.client.clone();
    let max_n = input_keys.shape()[0] as u32;
    let device = input_keys.device.clone();

    if max_n == 0 || sorting_bits == 0 {
        return (input_keys, input_values);
    }
    let passes = sorting_bits.div_ceil(kernels::BITS_PER_PASS);
    let max_needed_wgs = max_n.div_ceil(BLOCK_SIZE);

    // Calculate dispatch counts matching the original formula
    let num_wgs_count = max_n.div_ceil(BLOCK_SIZE);
    let num_reduce_wgs_count = num_wgs_count.div_ceil(BLOCK_SIZE) * BIN_COUNT;

    let cube_dim = CubeDim::new_1d(WG);

    assert_eq!(num_keys_buf.shape()[0], 1, "Expected one GPU count");
    let num_wgs =
        calculate_cube_count_elemwise(&client, max_n as usize, CubeDim::new_1d(BLOCK_SIZE));
    let num_reduce_wgs = calculate_cube_count_elemwise(
        &client,
        num_reduce_wgs_count as usize,
        CubeDim::new_single(),
    );

    // Small sorts count each next pass inside the scatter (when the digit
    // width lets its counters fit in shared memory), so only the first pass
    // dispatches a count. A pass's scan-add clears the other count
    // buffer, which its scatter then accumulates into. Big sorts count every
    // pass into alternating buffers too: the unused one is still bound, and a
    // binding may not alias another in the same dispatch.
    let scatter_counts = SCATTER_COUNT && max_needed_wgs <= SCATTER_COUNT_MAX_WGS;
    let counts: Vec<_> = (0..2)
        .map(|_| create_tensor([(max_needed_wgs * BIN_COUNT) as usize], &device, DType::I32))
        .collect();
    // Round up per-chunk totals to the scan block size. Padded entries are
    // never read or written by the scan kernels.
    let reduced_buf_size = num_reduce_wgs_count.div_ceil(BLOCK_SIZE).max(1) * BLOCK_SIZE;
    let reduced_buf = create_tensor([reduced_buf_size as usize], &device, DType::I32);

    // Inputs nobody else holds are the second ping-pong pair: one fresh pair
    // instead of two. Inputs a caller may still alias stay untouched, so both
    // output pairs are private then. A one-pass sort only needs one pair.
    let reuse_inputs = input_keys.can_mut() && input_values.can_mut();
    let mut outputs: Vec<_> = (0..if reuse_inputs { 1 } else { passes.min(2) })
        .map(|_| {
            (
                create_tensor([max_n as usize], &device, input_keys.dtype()),
                create_tensor([max_n as usize], &device, input_values.dtype()),
            )
        })
        .collect();
    if reuse_inputs && passes > 1 {
        outputs.push((input_keys.clone(), input_values.clone()));
    }
    let mut cur_keys = input_keys;
    let mut cur_vals = input_values;

    for pass in 0..passes {
        let count_buf = &counts[(pass % 2) as usize];
        let next_count_buf = &counts[((pass + 1) % 2) as usize];
        let count_next = scatter_counts && pass + 1 < passes;

        if pass == 0 || !scatter_counts {
            kernels::sort_count_kernel::launch(
                &client,
                num_wgs.clone(),
                cube_dim,
                num_keys_buf.clone().into_tensor_arg(),
                cur_keys.clone().into_tensor_arg(),
                count_buf.clone().into_tensor_arg(),
                skip_bits + pass * kernels::BITS_PER_PASS,
            );
        }

        {
            kernels::sort_reduce_kernel::launch(
                &client,
                num_reduce_wgs.clone(),
                cube_dim,
                num_keys_buf.clone().into_tensor_arg(),
                count_buf.clone().into_tensor_arg(),
                reduced_buf.clone().into_tensor_arg(),
            );
            kernels::sort_scan_kernel::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                cube_dim,
                num_keys_buf.clone().into_tensor_arg(),
                reduced_buf.clone().into_tensor_arg(),
            );

            kernels::sort_scan_add_kernel::launch(
                &client,
                num_reduce_wgs.clone(),
                cube_dim,
                num_keys_buf.clone().into_tensor_arg(),
                reduced_buf.clone().into_tensor_arg(),
                count_buf.clone().into_tensor_arg(),
                next_count_buf.clone().into_tensor_arg(),
                count_next,
            );
        }

        let (output_keys, output_values) = &outputs[(pass % 2) as usize];

        kernels::sort_scatter_kernel::launch(
            &client,
            num_wgs.clone(),
            cube_dim,
            num_keys_buf.clone().into_tensor_arg(),
            cur_keys.clone().into_tensor_arg(),
            cur_vals.clone().into_tensor_arg(),
            count_buf.clone().into_tensor_arg(),
            output_keys.clone().into_tensor_arg(),
            output_values.clone().into_tensor_arg(),
            next_count_buf.clone().into_tensor_arg(),
            skip_bits + pass * kernels::BITS_PER_PASS,
            count_next,
        );

        cur_keys = output_keys.clone();
        cur_vals = output_values.clone();
    }
    (cur_keys, cur_vals)
}

#[cfg(test)]
mod tests {
    use crate::radix_argsort;
    use brush_cube::{CubeDevice, MainBackendBase, create_tensor_from_slice};
    use burn::backend::ops::IntTensorOps;
    use burn::tensor::DType;

    use burn_wgpu::CubeTensor;
    use rand::RngExt;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    async fn read_i32(tensor: CubeTensor) -> Vec<i32> {
        let data = MainBackendBase::int_into_data(tensor)
            .await
            .expect("readback");
        data.as_slice::<i32>().expect("Wrong type").to_vec()
    }

    pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices = (0..data.len()).collect::<Vec<_>>();
        indices.sort_by_key(|&i| &data[i]);
        indices
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sorting() {
        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);

        for i in 0..128 {
            let keys_inp = [
                5 + i * 4,
                i,
                6,
                123,
                74657,
                123,
                999,
                2i32.pow(24) + 123,
                6,
                7,
                8,
                0,
                i * 2,
                16 + i,
                128 * i,
            ];

            let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

            let keys = create_tensor_from_slice(&keys_inp, &device, DType::I32);
            let values = create_tensor_from_slice(&values_inp, &device, DType::I32);
            let (ret_keys, ret_values) = radix_argsort(keys, values, 32);

            let ret_keys = read_i32(ret_keys).await;
            let ret_values = read_i32(ret_values).await;

            let inds = argsort(&keys_inp);

            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i] as u32).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i] as u32).collect();

            for (((key, val), ref_key), ref_val) in ret_keys
                .iter()
                .zip(&ret_values)
                .zip(ref_keys)
                .zip(ref_values)
            {
                assert_eq!(*key, ref_key as i32);
                assert_eq!(*val, ref_val as i32);
            }
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sorting_big() {
        // Simulate some data as one might find for a bunch of gaussians.
        let mut rng = rand::rng();
        let mut keys_inp = Vec::new();
        for i in 0..10000 {
            let start = rng.random_range(i..i + 150);
            let end = rng.random_range(start..start + 250);

            for j in start..end {
                if rng.random::<f32>() < 0.5 {
                    keys_inp.push(j);
                }
            }
        }

        let values_inp: Vec<_> = keys_inp.iter().map(|&x| x * 2 + 5).collect();

        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let keys = create_tensor_from_slice(&keys_inp, &device, DType::I32);
        let values = create_tensor_from_slice(&values_inp, &device, DType::I32);
        let (ret_keys, ret_values) = radix_argsort(keys, values, 32);

        let ret_keys = read_i32(ret_keys).await;
        let ret_values = read_i32(ret_values).await;

        let inds = argsort(&keys_inp);
        let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

        for (((key, val), ref_key), ref_val) in ret_keys
            .iter()
            .zip(&ret_values)
            .zip(ref_keys)
            .zip(ref_values)
        {
            assert_eq!(*key, ref_key as i32);
            assert_eq!(*val, ref_val as i32);
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sorting_large() {
        // Test with a ton of elements to verify 2D dispatch works correctly.
        const NUM_ELEMENTS: usize = 30_000_000;

        let mut rng = rand::rng();

        // Generate random keys with limited range to allow verification
        let keys_inp: Vec<u32> = (0..NUM_ELEMENTS)
            .map(|_| rng.random_range(0..1_000_000))
            .collect();
        let values_inp: Vec<u32> = (0..NUM_ELEMENTS).map(|i| i as u32).collect();

        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let keys = create_tensor_from_slice(&keys_inp, &device, DType::I32);
        let values = create_tensor_from_slice(&values_inp, &device, DType::I32);
        let (ret_keys, ret_values) = radix_argsort(keys, values, 32);

        let ret_keys_slice = read_i32(ret_keys).await;
        let ret_values_slice = read_i32(ret_values).await;

        assert_eq!(ret_keys_slice.len(), NUM_ELEMENTS);
        assert_eq!(ret_values_slice.len(), NUM_ELEMENTS);

        // Verify the output is sorted
        for i in 1..NUM_ELEMENTS {
            assert!(
                ret_keys_slice[i - 1] <= ret_keys_slice[i],
                "Keys not sorted at index {i}: {} > {}",
                ret_keys_slice[i - 1],
                ret_keys_slice[i]
            );
        }

        // Verify that values correspond to original indices that had those keys
        // Check a sample of indices to avoid O(n^2) verification
        let check_indices = [0, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 19_999_999];
        for &idx in &check_indices {
            let sorted_key = ret_keys_slice[idx] as u32;
            let original_idx = ret_values_slice[idx] as usize;
            assert_eq!(
                keys_inp[original_idx], sorted_key,
                "Value at index {idx} points to wrong original index"
            );
        }
    }

    // Regression test for a silent corruption in the radix sort that hits at
    // ~67M keys.
    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sorting_above_scan_block_size() {
        const NUM_ELEMENTS: usize = 70_000_000;

        let mut keys_inp: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();
        {
            use std::num::Wrapping;
            let mut state = Wrapping(0xD15EA5Eu64);
            for i in (1..keys_inp.len()).rev() {
                state += Wrapping(0x9E3779B97F4A7C15u64);
                let mut z = state.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                let j = (z as usize) % (i + 1);
                keys_inp.swap(i, j);
            }
        }
        let values_inp: Vec<u32> = (0..NUM_ELEMENTS as u32).collect();
        let mut expected_values = vec![0u32; NUM_ELEMENTS];
        for (i, &k) in keys_inp.iter().enumerate() {
            expected_values[k as usize] = i as u32;
        }

        let device = CubeDevice::Wgpu(brush_cube::test_helpers::test_device().await);
        let keys = create_tensor_from_slice(&keys_inp, &device, DType::I32);
        let values = create_tensor_from_slice(&values_inp, &device, DType::I32);
        let (ret_keys, ret_values) = radix_argsort(keys, values, 32);

        let ret_keys_slice = read_i32(ret_keys).await;
        let ret_values_slice = read_i32(ret_values).await;

        assert_eq!(ret_keys_slice.len(), NUM_ELEMENTS);
        assert_eq!(ret_values_slice.len(), NUM_ELEMENTS);

        for i in 0..NUM_ELEMENTS {
            assert_eq!(
                ret_keys_slice[i] as u32, i as u32,
                "key at sorted index {i} is {}, expected {i}",
                ret_keys_slice[i]
            );
            assert_eq!(
                ret_values_slice[i] as u32, expected_values[i],
                "value at sorted index {i} is {}, expected {}",
                ret_values_slice[i], expected_values[i]
            );
        }
    }
}
