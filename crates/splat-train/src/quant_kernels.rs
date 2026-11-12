//! Portable packed moment kernels, authored for this fork.
//! A workgroup reduces one block's absolute maximum and packs four signed
//! quantization codes per u32. Codes use a +128 bias so zero is exact.

use brush_cube::{MainBackendBase, create_tensor};
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    Backend, TensorMetadata, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::cubecl::{self, cube, prelude::*};
use burn::tensor::{DType, Shape, f16};

const BLOCK: usize = 128;

fn encoded_meta(input: &TensorSpec) -> (TensorSpec, TensorSpec) {
    let blocks = input.shape.num_elements().div_ceil(BLOCK);
    (
        TensorSpec::new(Shape::new([blocks * BLOCK / 4]), DType::I32),
        TensorSpec::new(Shape::new([blocks]), DType::F32),
    )
}

#[backend_extension(Cube, Fusion)]
pub(crate) trait QuantMomentOps: Backend {
    #[fusion(meta = encoded_meta)]
    fn encode_moment(input: FloatTensor<Self>) -> (IntTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |_codes, _scales, count| TensorSpec::new(Shape::new([*count]), DType::F32))]
    fn decode_moment(
        codes: IntTensor<Self>,
        scales: FloatTensor<Self>,
        count: usize,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |params, _grad, _grad_rows, codes, scales, _variance, _lr_scale, _beta, _factor, _inv1, _inv2, _epsilon, _lr, _round_seed, _row_width, _color_dir| (params.clone(), codes.clone(), scales.clone()))]
    fn step_moment(
        params: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        grad_rows: IntTensor<Self>,
        codes: IntTensor<Self>,
        scales: FloatTensor<Self>,
        variance: FloatTensor<Self>,
        lr_scale: FloatTensor<Self>,
        #[fusion(scalar)] beta: f32,
        #[fusion(scalar)] factor: f32,
        #[fusion(scalar)] inv1: f32,
        #[fusion(scalar)] inv2: f32,
        #[fusion(scalar)] epsilon: f32,
        #[fusion(scalar)] lr: f32,
        #[fusion(scalar)] round_seed: u32,
        row_width: u32,
        // `grad` is `[rows, 6]` (colour gradient, view direction): the SH
        // coefficient gradient `Y_k(dir) * colour` is rebuilt in-kernel.
        color_dir: bool,
    ) -> (FloatTensor<Self>, IntTensor<Self>, FloatTensor<Self>);

    /// `tensor[rows[i], ..] = values[i, ..]` for unique `rows`, one thread
    /// per assigned value. Burn's `select_assign` runs one thread per column
    /// and loops over every selected row, which serialises refinement.
    #[fusion(meta = |tensor, _rows, _values, _row_width| tensor.clone())]
    fn assign_rows(
        tensor: FloatTensor<Self>,
        rows: IntTensor<Self>,
        values: FloatTensor<Self>,
        row_width: u32,
    ) -> FloatTensor<Self>;

    /// Zero rows `rows` (unique, `row_width` values each) of a packed moment
    /// in place: code 128 decodes to exactly 0 whatever the block scale, so
    /// no decode/re-encode of the whole moment is needed.
    #[fusion(meta = |codes, _rows, _row_width| codes.clone())]
    fn zero_moment_rows(
        codes: IntTensor<Self>,
        rows: IntTensor<Self>,
        row_width: u32,
    ) -> IntTensor<Self>;

    /// Packed moment with its rows gathered by `perm` (`out` row `r` is input
    /// row `perm[r]`): each output block decodes its source values in-kernel
    /// and re-encodes, so no f32 copy of the moment is materialised.
    #[fusion(meta = |codes, scales, _perm, _row_width| (codes.clone(), scales.clone()))]
    fn permute_moment(
        codes: IntTensor<Self>,
        scales: FloatTensor<Self>,
        perm: IntTensor<Self>,
        row_width: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>);

    /// `out[r]` is 1 when row `r` (`row_width` values) holds a NaN or inf,
    /// else 0. One thread per row, so no `[N, width]` mask is materialised
    /// and no autotuned reduce runs.
    #[fusion(meta = |tensor, row_width| TensorSpec::new(Shape::new([tensor.shape.num_elements() / *row_width as usize]), DType::F32))]
    fn rows_non_finite(tensor: FloatTensor<Self>, row_width: u32) -> FloatTensor<Self>;

    /// Full-precision Adam over f32 params/moments, in place where owned.
    /// `grad` is compact: splat row `r` reads gradient row `grad_rows[r]`.
    #[fusion(meta = |params, _grad, _grad_rows, m, v, _lr_scale, _beta1, _beta2, _inv1, _inv2, _epsilon, _lr, _row_width| (params.clone(), m.clone(), v.clone()))]
    fn step_adam(
        params: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        grad_rows: IntTensor<Self>,
        m: FloatTensor<Self>,
        v: FloatTensor<Self>,
        lr_scale: FloatTensor<Self>,
        #[fusion(scalar)] beta1: f32,
        #[fusion(scalar)] beta2: f32,
        #[fusion(scalar)] inv1: f32,
        #[fusion(scalar)] inv2: f32,
        #[fusion(scalar)] epsilon: f32,
        #[fusion(scalar)] lr: f32,
        row_width: u32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}

impl QuantMomentOps for MainBackendBase {
    fn assign_rows(
        tensor: FloatTensor<Self>,
        rows: IntTensor<Self>,
        values: FloatTensor<Self>,
        row_width: u32,
    ) -> FloatTensor<Self> {
        let rows = rows_u32(rows);
        let tensor = if tensor.can_mut() { tensor } else { tensor.copy() };
        assert!(values.is_contiguous() && tensor.is_contiguous());
        assert_eq!(tensor.dtype, values.dtype);
        let n = values.shape().num_elements();
        assert_eq!(rows.shape().num_elements() * row_width as usize, n);
        if n == 0 {
            return tensor;
        }
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&tensor.client, n, dim);
        macro_rules! launch {
            ($t:ty) => {
                assign_rows_kernel::launch::<$t>(
                    &tensor.client, count, dim,
                    tensor.clone().into_tensor_arg(), rows.into_tensor_arg(),
                    values.into_tensor_arg(), row_width,
                )
            };
        }
        match tensor.dtype {
            DType::F32 => launch!(f32),
            DType::F16 => launch!(f16),
            dtype => panic!("assign_rows: unsupported dtype {dtype:?}"),
        }
        tensor
    }

    fn zero_moment_rows(
        codes: IntTensor<Self>,
        rows: IntTensor<Self>,
        row_width: u32,
    ) -> IntTensor<Self> {
        let rows = rows_u32(rows);
        let codes = if codes.can_mut() { codes } else { codes.copy() };
        assert!(codes.is_contiguous());
        assert_eq!(
            codes.dtype,
            DType::I32,
            "packed moment words must be 32-bit"
        );
        let n = rows.shape().num_elements() * row_width as usize;
        if n == 0 {
            return codes;
        }
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&codes.client, n, dim);
        zero_moment_rows_kernel::launch(
            &codes.client,
            count,
            dim,
            codes.clone().into_tensor_arg(),
            rows.into_tensor_arg(),
            row_width,
        );
        codes
    }

    fn permute_moment(
        codes: IntTensor<Self>,
        scales: FloatTensor<Self>,
        perm: IntTensor<Self>,
        row_width: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        let perm = rows_u32(perm);
        assert!(codes.is_contiguous() && scales.is_contiguous());
        assert_eq!(
            codes.dtype,
            DType::I32,
            "packed moment words must be 32-bit"
        );
        let blocks = scales.shape().num_elements();
        assert!(perm.shape().num_elements() * row_width as usize <= blocks * BLOCK);
        let out_codes = create_tensor([blocks * BLOCK / 4], &codes.device, DType::I32);
        let out_scales = create_tensor([blocks], &codes.device, DType::F32);
        let dim = CubeDim::new_1d(BLOCK as u32);
        let count = cubecl::calculate_cube_count_elemwise(&codes.client, blocks * BLOCK, dim);
        permute_kernel::launch(
            &codes.client,
            count,
            dim,
            codes.clone().into_tensor_arg(),
            scales.into_tensor_arg(),
            perm.into_tensor_arg(),
            out_codes.clone().into_tensor_arg(),
            out_scales.clone().into_tensor_arg(),
            row_width,
        );
        (out_codes, out_scales)
    }

    fn rows_non_finite(tensor: FloatTensor<Self>, row_width: u32) -> FloatTensor<Self> {
        assert!(tensor.is_contiguous());
        let rows = tensor.shape().num_elements() / row_width as usize;
        let out = create_tensor([rows], &tensor.device, DType::F32);
        if rows == 0 {
            return out;
        }
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&tensor.client, rows, dim);
        macro_rules! launch {
            ($t:ty) => {
                rows_non_finite_kernel::launch::<$t>(
                    &tensor.client, count, dim,
                    tensor.clone().into_tensor_arg(), out.clone().into_tensor_arg(), row_width,
                )
            };
        }
        match tensor.dtype {
            DType::F32 => launch!(f32),
            DType::F16 => launch!(f16),
            dtype => panic!("rows_non_finite: unsupported dtype {dtype:?}"),
        }
        out
    }

    fn encode_moment(input: FloatTensor<Self>) -> (IntTensor<Self>, FloatTensor<Self>) {
        assert!(input.is_contiguous());
        assert_eq!(input.dtype, DType::F32);
        let blocks = input.shape().num_elements().div_ceil(BLOCK);
        let codes = create_tensor([blocks * BLOCK / 4], &input.device, DType::I32);
        let scales = create_tensor([blocks], &input.device, DType::F32);
        let dim = CubeDim::new_1d(BLOCK as u32);
        let count = cubecl::calculate_cube_count_elemwise(&input.client, blocks * BLOCK, dim);
        encode_kernel::launch(
            &input.client,
            count,
            dim,
            input.clone().into_tensor_arg(),
            codes.clone().into_tensor_arg(),
            scales.clone().into_tensor_arg(),
        );
        (codes, scales)
    }

    fn decode_moment(
        codes: IntTensor<Self>,
        scales: FloatTensor<Self>,
        count: usize,
    ) -> FloatTensor<Self> {
        let out = create_tensor([count], &codes.device, DType::F32);
        let dim = CubeDim::new_1d(BLOCK as u32);
        let cubes = cubecl::calculate_cube_count_elemwise(&codes.client, count, dim);
        decode_kernel::launch(
            &codes.client,
            cubes,
            dim,
            codes.clone().into_tensor_arg(),
            scales.into_tensor_arg(),
            out.clone().into_tensor_arg(),
        );
        out
    }

    fn step_moment(
        params: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        grad_rows: IntTensor<Self>,
        codes: IntTensor<Self>,
        scales: FloatTensor<Self>,
        variance: FloatTensor<Self>,
        lr_scale: FloatTensor<Self>,
        beta: f32,
        factor: f32,
        inv1: f32,
        inv2: f32,
        epsilon: f32,
        lr: f32,
        round_seed: u32,
        row_width: u32,
        color_dir: bool,
    ) -> (FloatTensor<Self>, IntTensor<Self>, FloatTensor<Self>) {
        assert!(
            params.is_contiguous()
                && grad.is_contiguous()
                && grad_rows.is_contiguous()
                && variance.is_contiguous()
                && lr_scale.is_contiguous()
        );
        assert!(matches!(params.dtype, DType::F32 | DType::F16));
        for tensor in [&grad, &scales, &variance, &lr_scale] {
            assert_eq!(tensor.dtype, DType::F32, "quantized Adam computes in f32");
        }
        assert_eq!(
            codes.dtype,
            DType::I32,
            "packed moment words must be 32-bit"
        );
        let n = params.shape().num_elements();
        let blocks = n.div_ceil(BLOCK);
        let rows = n / row_width as usize;
        assert_eq!(grad_rows.shape().num_elements(), rows);
        let grad_width = if color_dir { 6 } else { row_width as usize };
        assert_eq!(grad.shape().num_elements() % grad_width, 0);
        assert_eq!(variance.shape().num_elements() * row_width as usize, n);
        assert_eq!(lr_scale.shape().num_elements(), row_width as usize);
        assert_eq!(codes.shape().num_elements(), blocks * BLOCK / 4);
        assert_eq!(scales.shape().num_elements(), blocks);
        let dim = CubeDim::new_1d(BLOCK as u32);
        let count = cubecl::calculate_cube_count_elemwise(&params.client, blocks * BLOCK, dim);
        // Reuse uniquely owned storage; preserve snapshots held by callers.
        // Seven tensor bindings plus runtime metadata fit the WebGPU default of 8.
        let out = if params.can_mut() {
            params
        } else {
            params.copy()
        };
        let out_codes = if codes.can_mut() { codes } else { codes.copy() };
        let out_scales = if scales.can_mut() {
            scales
        } else {
            scales.copy()
        };
        // Exactly 32-wide subgroups: one 128-value block per subgroup.
        let hw = &out.client.properties().hardware;
        let w4 = out.client.features().plane.contains(burn::cubecl::features::Plane::Ops)
            && hw.plane_size_min == 32
            && hw.plane_size_max == 32;
        let count_w4 = cubecl::calculate_cube_count_elemwise(&out.client, blocks * BLOCK / 4, dim);
        macro_rules! launch_w4 {
            ($elem:ty, $half:expr) => {
                step_kernel_w4::launch::<$elem>(
                    &out.client, count_w4, dim,
                    out.clone().into_tensor_arg(), grad.clone().into_tensor_arg(),
                    grad_rows.clone().into_tensor_arg(),
                    out_codes.clone().into_tensor_arg(), out_scales.clone().into_tensor_arg(),
                    variance.clone().into_tensor_arg(), lr_scale.clone().into_tensor_arg(),
                    beta, factor, inv1, inv2, epsilon, lr, round_seed, row_width, $half,
                    color_dir,
                )
            };
        }
        macro_rules! launch_step {
            ($elem:ty, $half:expr) => {
                step_kernel::launch::<$elem>(
                    &out.client, count, dim,
                    out.clone().into_tensor_arg(), grad.into_tensor_arg(),
                    grad_rows.into_tensor_arg(),
                    out_codes.clone().into_tensor_arg(), out_scales.clone().into_tensor_arg(),
                    variance.into_tensor_arg(), lr_scale.into_tensor_arg(),
                    beta, factor, inv1, inv2, epsilon, lr, round_seed, row_width, $half,
                    color_dir,
                )
            };
        }
        match (out.dtype, w4) {
            (DType::F16, true) => launch_w4!(f16, true),
            (DType::F32, true) => launch_w4!(f32, false),
            (DType::F16, false) => launch_step!(f16, true),
            (DType::F32, false) => launch_step!(f32, false),
            _ => unreachable!(),
        }
        (out, out_codes, out_scales)
    }

    fn step_adam(
        params: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        grad_rows: IntTensor<Self>,
        m: FloatTensor<Self>,
        v: FloatTensor<Self>,
        lr_scale: FloatTensor<Self>,
        beta1: f32,
        beta2: f32,
        inv1: f32,
        inv2: f32,
        epsilon: f32,
        lr: f32,
        row_width: u32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        for tensor in [&params, &grad, &m, &v, &lr_scale] {
            assert!(tensor.is_contiguous());
            assert_eq!(tensor.dtype, DType::F32, "Adam computes in f32");
        }
        assert!(grad_rows.is_contiguous());
        let n = params.shape().num_elements();
        assert_eq!(m.shape().num_elements(), n);
        assert_eq!(v.shape().num_elements(), n);
        assert_eq!(grad_rows.shape().num_elements() * row_width as usize, n);
        assert_eq!(lr_scale.shape().num_elements(), row_width as usize);
        let own = |t: FloatTensor<Self>| if t.can_mut() { t } else { t.copy() };
        let (params, m, v) = (own(params), own(m), own(v));
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&params.client, n, dim);
        adam_kernel::launch(
            &params.client, count, dim,
            params.clone().into_tensor_arg(), grad.into_tensor_arg(),
            grad_rows.into_tensor_arg(), m.clone().into_tensor_arg(),
            v.clone().into_tensor_arg(), lr_scale.into_tensor_arg(),
            beta1, beta2, inv1, inv2, epsilon, lr, row_width,
        );
        (params, m, v)
    }
}

#[cube(launch)]
fn adam_kernel(
    params: &mut Tensor<f32>,
    grad: &Tensor<f32>,
    grad_rows: &Tensor<u32>,
    m: &mut Tensor<f32>,
    v: &mut Tensor<f32>,
    lr_scale: &Tensor<f32>,
    beta1: f32,
    beta2: f32,
    inv1: f32,
    inv2: f32,
    epsilon: f32,
    lr: f32,
    row_width: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= params.len() {
        terminate!();
    }
    let row = index / row_width as usize;
    let col = index % row_width as usize;
    let g = grad[grad_rows[row] as usize * row_width as usize + col];
    let m1 = m[index] * beta1 + g * (1.0f32 - beta1);
    let v1 = v[index] * beta2 + g * g * (1.0f32 - beta2);
    m[index] = m1;
    v[index] = v1;
    let step = (m1 * inv1) / (f32::sqrt(v1 * inv2) + epsilon);
    params[index] = params[index] - step * (lr_scale[col] * lr);
}

#[cube]
fn pack_block(value: f32, codes: &mut Tensor<u32>, scales: &mut Tensor<f32>) {
    let lane = UNIT_POS as usize;
    let block = CUBE_POS;
    let mut scratch = Shared::<[f32]>::new_slice(BLOCK);
    scratch[lane] = f32::abs(value);
    sync_cube();
    let mut stride = 64usize;
    while stride > 0 {
        if lane < stride {
            scratch[lane] = f32::max(scratch[lane], scratch[lane + stride]);
        }
        sync_cube();
        stride /= 2;
    }
    let scale = scratch[0] / 127.0f32;
    if lane == 0 {
        scales[block] = scale;
    }
    let mut code = 128u32;
    if scale > 0.0f32 {
        code = (f32::round(clamp(value / scale, -127.0f32, 127.0f32)) + 128.0f32) as u32;
    }
    let mut packed = Shared::<[u32]>::new_slice(BLOCK);
    packed[lane] = code;
    sync_cube();
    if lane % 4 == 0 {
        codes[block * (BLOCK / 4) + lane / 4] = packed[lane]
            | (packed[lane + 1] << 8u32)
            | (packed[lane + 2] << 16u32)
            | (packed[lane + 3] << 24u32);
    }
}

#[cube]
fn unpack(codes: &Tensor<u32>, scales: &Tensor<f32>, index: usize) -> f32 {
    let code = (codes[index / 4] >> ((index % 4) as u32 * 8u32)) & 255u32;
    (code as f32 - 128.0f32) * scales[index / BLOCK]
}

#[cube(launch)]
fn encode_kernel(input: &Tensor<f32>, codes: &mut Tensor<u32>, scales: &mut Tensor<f32>) {
    if CUBE_POS >= scales.len() {
        terminate!();
    }
    let index = CUBE_POS * BLOCK + UNIT_POS as usize;
    let mut value = 0.0f32;
    if index < input.len() {
        value = input[index];
    }
    pack_block(value, codes, scales);
}

#[cube(launch)]
fn decode_kernel(codes: &Tensor<u32>, scales: &Tensor<f32>, out: &mut Tensor<f32>) {
    let index = ABSOLUTE_POS;
    if index < out.len() {
        out[index] = unpack(codes, scales, index);
    }
}

/// [`encode_kernel`] reading value `index` of the row-permuted moment from
/// the source codes: row `index / row_width` comes from source row `perm[row]`.
#[cube(launch)]
fn permute_kernel(
    codes: &Tensor<u32>,
    scales: &Tensor<f32>,
    perm: &Tensor<u32>,
    out_codes: &mut Tensor<u32>,
    out_scales: &mut Tensor<f32>,
    row_width: u32,
) {
    if CUBE_POS >= out_scales.len() {
        terminate!();
    }
    let index = CUBE_POS * BLOCK + UNIT_POS as usize;
    let row = index / row_width as usize;
    let mut value = 0.0f32;
    if row < perm.len() {
        let source = perm[row] as usize * row_width as usize + index % row_width as usize;
        value = unpack(codes, scales, source);
    }
    pack_block(value, out_codes, out_scales);
}

/// Whether `index`'s splat steps this round. The compact colour path is lazy:
/// row 0 of its gradient is the culled splats' zero row, and those splats
/// skip the update entirely (no moment decay), like gsplat's selective Adam.
#[cube]
fn row_visible(grad_rows: &Tensor<u32>, index: usize, row_width: u32, #[comptime] color_dir: bool) -> bool {
    let mut visible = true;
    if comptime![color_dir] {
        visible = grad_rows[index / row_width as usize] != 0u32;
    }
    visible
}

/// One value of the SH block step: rebuild or read its gradient, update the
/// first moment and the parameter, and return the new moment.
#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn step_value<P: Float>(
    params: &mut Tensor<P>,
    grad: &Tensor<f32>,
    grad_rows: &Tensor<u32>,
    variance: &Tensor<f32>,
    lr_scale: &Tensor<f32>,
    index: usize,
    old_moment: f32,
    beta: f32,
    factor: f32,
    inv1: f32,
    inv2: f32,
    epsilon: f32,
    lr: f32,
    round_seed: u32,
    row_width: u32,
    #[comptime] half_params: bool,
    #[comptime] color_dir: bool,
) -> f32 {
    // `grad` is compact: each splat's row is `grad_rows[row]`.
    let row = index / row_width as usize;
    let col = index % row_width as usize;
    let mut g = 0.0f32;
    if comptime![color_dir] {
        let base = grad_rows[row] as usize * 6;
        let basis = sh_basis((col / 3) as u32, grad[base + 3], grad[base + 4], grad[base + 5]);
        g = basis * grad[base + col % 3];
    } else {
        g = grad[grad_rows[row] as usize * row_width as usize + col];
    }
    let moment = old_moment * beta + g * factor;
    let denom = f32::sqrt(variance[row] * inv2) + epsilon;
    let delta = (moment * inv1 / denom) * (lr_scale[col] * lr);
    let mut updated = f32::cast_from(params[index]) - delta;
    if half_params {
        updated = stochastic_half(updated, index as u32, round_seed);
    }
    params[index] = P::cast_from(updated);
    moment
}

/// [`step_kernel`] for 32-wide subgroups: each 128-value block is one
/// subgroup and each lane owns one packed word (4 consecutive values), so
/// the block absmax is a single subgroup max and no workgroup barrier or
/// shuffle is needed. Same moments, scales and codes.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn step_kernel_w4<P: Float>(
    params: &mut Tensor<P>,
    grad: &Tensor<f32>,
    grad_rows: &Tensor<u32>,
    codes: &mut Tensor<u32>,
    scales: &mut Tensor<f32>,
    variance: &Tensor<f32>,
    lr_scale: &Tensor<f32>,
    beta: f32,
    factor: f32,
    inv1: f32,
    inv2: f32,
    epsilon: f32,
    lr: f32,
    round_seed: u32,
    row_width: u32,
    #[comptime] half_params: bool,
    #[comptime] color_dir: bool,
) {
    let block = CUBE_POS * (CUBE_DIM / 32u32) as usize + (UNIT_POS / 32u32) as usize;
    // Uniform per subgroup: a whole subgroup owns one block.
    if block >= scales.len() {
        terminate!();
    }
    let word = block * (BLOCK / 4) + UNIT_POS_PLANE as usize;
    // Lazy (visible-only) Adam on the compact colour path: a culled splat
    // (`grad_rows == 0`) keeps its parameters and moments untouched this
    // step. A block with no visible value is left as is: re-encoding the
    // unchanged moments would reproduce the same codes and scale.
    let mut any_visible = false;
    #[unroll]
    for j in 0..4usize {
        let index = word * 4 + j;
        if index < params.len() {
            any_visible = any_visible || row_visible(grad_rows, index, row_width, color_dir);
        }
    }
    if !plane_any(any_visible) {
        terminate!();
    }
    let old = codes[word];
    let old_scale = scales[block];
    let mut m = Array::<f32>::new(4usize);
    #[unroll]
    for j in 0..4usize {
        let code = (old >> (j as u32 * 8u32)) & 255u32;
        m[j] = select(word * 4 + j < params.len(), (code as f32 - 128.0f32) * old_scale, 0.0f32);
    }
    let mut word_path = false;
    if comptime![color_dir] {
        // A word's 4 values share one splat row when rows are a multiple of
        // 4 wide (SH degree 1 and 3): look up the row's gradient, direction
        // and variance once, and evaluate the SH basis for the (at most 2)
        // coefficients the word covers instead of once per value. Same
        // arithmetic per value as `step_value`.
        word_path = row_width % 4u32 == 0u32;
    }
    if word_path {
        let first = word * 4;
        let row = first / row_width as usize;
        let col0 = first % row_width as usize;
        let grad_row = grad_rows[row];
        if grad_row != 0u32 {
            let base = grad_row as usize * 6;
            let dir_x = grad[base + 3];
            let dir_y = grad[base + 4];
            let dir_z = grad[base + 5];
            let k0 = col0 / 3;
            let basis_0 = sh_basis(k0 as u32, dir_x, dir_y, dir_z);
            let basis_1 = sh_basis(k0 as u32 + 1u32, dir_x, dir_y, dir_z);
            let denom = f32::sqrt(variance[row] * inv2) + epsilon;
            #[unroll]
            for j in 0..4usize {
                let index = first + j;
                if index < params.len() {
                    let col = col0 + j;
                    let basis = select(col / 3 == k0, basis_0, basis_1);
                    let g = basis * grad[base + col % 3];
                    let moment = m[j] * beta + g * factor;
                    let delta = (moment * inv1 / denom) * (lr_scale[col] * lr);
                    let mut updated = f32::cast_from(params[index]) - delta;
                    if half_params {
                        updated = stochastic_half(updated, index as u32, round_seed);
                    }
                    params[index] = P::cast_from(updated);
                    m[j] = moment;
                }
            }
        }
    } else {
        #[unroll]
        for j in 0..4usize {
            let index = word * 4 + j;
            if index < params.len() && row_visible(grad_rows, index, row_width, color_dir) {
                m[j] = step_value(
                    params, grad, grad_rows, variance, lr_scale, index, m[j], beta, factor,
                    inv1, inv2, epsilon, lr, round_seed, row_width, half_params, color_dir,
                );
            }
        }
    }
    let local = f32::max(f32::max(f32::abs(m[0]), f32::abs(m[1])), f32::max(f32::abs(m[2]), f32::abs(m[3])));
    let absmax = plane_max(local);
    let scale = absmax / 127.0f32;
    if UNIT_POS_PLANE == 0u32 {
        scales[block] = scale;
    }
    let mut packed = 0u32;
    #[unroll]
    for j in 0..4usize {
        let mut code = 128u32;
        if scale > 0.0f32 {
            code = (f32::round(clamp(m[j] / scale, -127.0f32, 127.0f32)) + 128.0f32) as u32;
        }
        packed |= code << (j as u32 * 8u32);
    }
    codes[word] = packed;
}

#[cube(launch)]
// `g`'s zero initialiser is overwritten on both comptime branches.
#[allow(unused_assignments)]
fn step_kernel<P: Float>(
    params: &mut Tensor<P>,
    grad: &Tensor<f32>,
    grad_rows: &Tensor<u32>,
    codes: &mut Tensor<u32>,
    scales: &mut Tensor<f32>,
    variance: &Tensor<f32>,
    lr_scale: &Tensor<f32>,
    beta: f32,
    factor: f32,
    inv1: f32,
    inv2: f32,
    epsilon: f32,
    lr: f32,
    round_seed: u32,
    row_width: u32,
    #[comptime] half_params: bool,
    #[comptime] color_dir: bool,
) {
    if CUBE_POS >= scales.len() {
        terminate!();
    }
    let index = CUBE_POS * BLOCK + UNIT_POS as usize;
    let mut moment = 0.0f32;
    let mut visible = false;
    if index < params.len() {
        visible = row_visible(grad_rows, index, row_width, color_dir);
    }
    if index < params.len() && !visible {
        // Culled splat on the lazy colour path: moment and parameter unchanged.
        let code = (codes[index / 4] >> ((index % 4) as u32 * 8u32)) & 255u32;
        moment = (code as f32 - 128.0f32) * scales[index / BLOCK];
    }
    if visible {
        let code = (codes[index / 4] >> ((index % 4) as u32 * 8u32)) & 255u32;
        let old_moment = (code as f32 - 128.0f32) * scales[index / BLOCK];
        // `grad` is compact: each splat's row is `grad_rows[row]`.
        let row = index / row_width as usize;
        let col = index % row_width as usize;
        let mut g = 0.0f32;
        if comptime![color_dir] {
            let base = grad_rows[row] as usize * 6;
            let basis = sh_basis(
                (col / 3) as u32,
                grad[base + 3],
                grad[base + 4],
                grad[base + 5],
            );
            g = basis * grad[base + col % 3];
        } else {
            g = grad[grad_rows[row] as usize * row_width as usize + col];
        }
        moment = old_moment * beta + g * factor;
        let denom = f32::sqrt(variance[row] * inv2) + epsilon;
        let delta = (moment * inv1 / denom) * (lr_scale[index % row_width as usize] * lr);
        let mut updated = f32::cast_from(params[index]) - delta;
        if half_params {
            updated = stochastic_half(updated, index as u32, round_seed);
        }
        params[index] = P::cast_from(updated);
    }
    // Order storage reads before in-place scale/word writes by another lane.
    // Blocks own disjoint words and scales, so no cross-workgroup order is needed.
    sync_storage();
    pack_block(moment, codes, scales);
}

/// Real SH basis `Y_k(x, y, z)` for `k < 25`, term for term the renderer's
/// `sh_coeffs_to_color_vjp`, so the rebuilt gradient matches the materialized one.
#[cube]
fn sh_basis(k: u32, x: f32, y: f32, z: f32) -> f32 {
    let z2 = z * z;
    let fc1 = x * x - y * y;
    let fs1 = 2.0f32 * x * y;
    let fc2 = x * fc1 - y * fs1;
    let fs2 = x * fs1 + y * fc1;
    let p_sh6 = 0.946_174_7f32 * z2 - 0.315_391_57f32;
    let p_sh12 = z * (1.865_881_7f32 * z2 - 1.119_529f32);
    let f0b = -1.092_548_5f32 * z;
    let f0c = -2.285_229f32 * z2 + 0.457_045_8f32;
    let f1b = 1.445_305_7f32 * z;
    let f2a = -0.590_043_6f32;
    let f0d = z * (-4.683_326f32 * z2 + 2.007_139_6f32);
    let f1c = 3.311_611_4f32 * z2 - 0.473_087_35f32;
    let f2b = -1.770_130_8f32 * z;
    let f3a = 0.625_835_75f32;
    let fc3 = x * fc2 - y * fs2;
    let fs3 = x * fs2 + y * fc2;
    let mut b = 0.282_094_8f32;
    if k == 1u32 {
        b = -0.488_602_5f32 * y;
    } else if k == 2u32 {
        b = 0.488_602_5f32 * z;
    } else if k == 3u32 {
        b = -0.488_602_5f32 * x;
    } else if k == 4u32 {
        b = 0.546_274_24f32 * fs1;
    } else if k == 5u32 {
        b = f0b * y;
    } else if k == 6u32 {
        b = p_sh6;
    } else if k == 7u32 {
        b = f0b * x;
    } else if k == 8u32 {
        b = 0.546_274_24f32 * fc1;
    } else if k == 9u32 {
        b = f2a * fs2;
    } else if k == 10u32 {
        b = f1b * fs1;
    } else if k == 11u32 {
        b = f0c * y;
    } else if k == 12u32 {
        b = p_sh12;
    } else if k == 13u32 {
        b = f0c * x;
    } else if k == 14u32 {
        b = f1b * fc1;
    } else if k == 15u32 {
        b = f2a * fc2;
    } else if k == 16u32 {
        b = f3a * fs3;
    } else if k == 17u32 {
        b = f2b * fs2;
    } else if k == 18u32 {
        b = f1c * fs1;
    } else if k == 19u32 {
        b = f0d * y;
    } else if k == 20u32 {
        b = 1.984_313_5f32 * z * p_sh12 + -1.006_230_6f32 * p_sh6;
    } else if k == 21u32 {
        b = f0d * x;
    } else if k == 22u32 {
        b = f1c * fc1;
    } else if k == 23u32 {
        b = f2b * fc2;
    } else if k == 24u32 {
        b = f3a * fc3;
    }
    b
}

/// Choose adjacent representable half values with probability proportional to
/// distance. All arithmetic is f32; no master parameter or residual is retained.
#[cube]
fn stochastic_half(value: f32, index: u32, step: u32) -> f32 {
    let magnitude = f32::abs(value);
    let mut result = value;
    // Preserve non-finite values/overflow for the trainer's existing pruning.
    if magnitude <= 65504.0f32 {
        let (lower, upper) = if magnitude < 0.00006103515625f32 {
            // Half subnormals have constant spacing 2^-24.
            let lower = f32::floor(magnitude * 16777216.0f32) / 16777216.0f32;
            (lower, lower + 0.000000059604644775390625f32)
        } else {
            let bits = u32::reinterpret(magnitude) & 0xffffe000u32;
            (f32::reinterpret(bits), f32::reinterpret(bits + 8192u32))
        };
        // A per-element, per-step integer hash supplies 24 uniform random bits.
        // The same training seed/trajectory therefore reproduces these writes.
        let mut hash = index ^ (step * 0x9e3779b9u32);
        hash = (hash ^ (hash >> 16u32)) * 0x7feb352du32;
        hash = (hash ^ (hash >> 15u32)) * 0x846ca68bu32;
        hash = hash ^ (hash >> 16u32);
        let uniform = (hash >> 8u32) as f32 / 16777216.0f32;
        let probability = (magnitude - lower) / (upper - lower);
        result = select(uniform < probability, upper, lower);
        if value < 0.0f32 {
            result = -result;
        }
    }
    result
}

/// Row indices as contiguous `u32`, whatever integer type the caller has.
fn rows_u32(rows: IntTensor<MainBackendBase>) -> IntTensor<MainBackendBase> {
    use burn::backend::ops::IntTensorOps;
    let rows = if rows.dtype == DType::U32 {
        rows
    } else {
        MainBackendBase::int_cast(rows, burn::tensor::IntDType::U32)
    };
    assert!(rows.is_contiguous());
    rows
}

#[cube(launch)]
fn assign_rows_kernel<N: Numeric>(
    tensor: &mut Tensor<N>,
    rows: &Tensor<u32>,
    values: &Tensor<N>,
    row_width: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < values.len() as u32 {
        let r = pos / row_width;
        let c = pos % row_width;
        tensor[(rows[r as usize] * row_width + c) as usize] = values[pos as usize];
    }
}

/// One thread per zeroed value. Rows need not be word-aligned (SH0/SH2 rows
/// are 3/27 values), so neighbouring rows can share a word: clear then set
/// this value's byte atomically; the two ops touch only that byte.
#[cube(launch)]
fn zero_moment_rows_kernel(codes: &mut Tensor<Atomic<u32>>, rows: &Tensor<u32>, row_width: u32) {
    let pos = ABSOLUTE_POS as u32;
    if pos < rows.len() as u32 * row_width {
        let index = rows[(pos / row_width) as usize] * row_width + pos % row_width;
        let shift = (index % 4u32) * 8u32;
        let word = (index / 4u32) as usize;
        Atomic::fetch_and(&codes[word], 0xffffffffu32 ^ (255u32 << shift));
        Atomic::fetch_or(&codes[word], 128u32 << shift);
    }
}

#[cube(launch)]
fn rows_non_finite_kernel<F: Float>(tensor: &Tensor<F>, out: &mut Tensor<f32>, row_width: u32) {
    let r = ABSOLUTE_POS as u32;
    if r < out.len() as u32 {
        let mut bad = 0.0f32;
        for c in 0..row_width {
            // Exponent all ones = inf or NaN; the f32 cast keeps both, and
            // bit tests stay valid where shader compilers assume no NaN.
            let bits = u32::reinterpret(f32::cast_from(tensor[(r * row_width + c) as usize]));
            if bits & 0x7f80_0000u32 == 0x7f80_0000u32 {
                bad = 1.0f32;
            }
        }
        out[r as usize] = bad;
    }
}
