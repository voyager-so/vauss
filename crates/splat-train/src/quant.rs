//! Block-scaled 8-bit optimizer storage with a fused update kernel.
//!
//! Implements uniform symmetric block scaling (Dettmers et al.,
//! https://arxiv.org/abs/2110.02861), not the paper's nonlinear codebook.
//! Update math stays f32; no full first-moment buffer is retained or decoded
//! during a fused update. Refinement explicitly decodes before remapping rows.

#[path = "quant_kernels.rs"]
mod kernels;

use burn::backend::Dispatch;
use burn::tensor::{Device, Int, IntDType, Tensor};
use kernels::QuantMomentOps;

pub(crate) const BLOCK_VALUES: usize = 128;

/// Four biased zero codes: the packed word for four exact zeros.
const ZERO_WORD: i32 = 0x8080_8080u32 as i32;

#[derive(Clone, Copy)]
pub(crate) struct MomentUpdate {
    pub beta: f32,
    pub factor: f32,
    pub inv1: f32,
    pub inv2: f32,
    pub epsilon: f32,
    pub lr: f32,
    pub round_seed: u32,
}

#[derive(Clone)]
pub(crate) struct BlockMoment<const D: usize> {
    codes: Tensor<1, Int>,
    scales: Tensor<1>,
    shape: [usize; D],
}

impl<const D: usize> BlockMoment<D> {
    pub fn zeros(shape: [usize; D], device: &Device) -> Self {
        let count: usize = shape.iter().product();
        assert!(count > 0, "cannot quantize an empty moment");
        let blocks = count.div_ceil(BLOCK_VALUES);
        Self {
            codes: Tensor::<1, Int>::zeros([blocks * BLOCK_VALUES / 4], device).cast(IntDType::I32),
            scales: Tensor::zeros([blocks], device),
            shape,
        }
    }

    pub fn encode(values: Tensor<D>) -> Self {
        let shape = values.dims();
        let count: usize = shape.iter().product();
        assert!(count > 0, "cannot quantize an empty moment");
        let (codes, scales) = Dispatch::encode_moment(values.reshape([count]).into_dispatch());
        Self {
            codes: Tensor::from_dispatch(codes),
            scales: Tensor::from_dispatch(scales),
            shape,
        }
    }

    pub fn decode(self) -> Tensor<D> {
        let count: usize = self.shape.iter().product();
        Tensor::<1>::from_dispatch(Dispatch::decode_moment(
            self.codes.into_dispatch(),
            self.scales.into_dispatch(),
            count,
        ))
        .reshape(self.shape)
    }

    /// Rows gathered by `perm` (row `r` becomes row `perm[r]`), as
    /// `encode(decode().select(0, perm))` but without an f32 copy of the moment.
    pub fn permute_rows(self, perm: &Tensor<1, Int>) -> Self {
        let rows = self.shape[0];
        assert_eq!(perm.dims(), [rows]);
        let width = self.shape[1..].iter().product::<usize>();
        let (codes, scales) = Dispatch::permute_moment(
            self.codes.into_dispatch(),
            self.scales.into_dispatch(),
            perm.clone().into_dispatch(),
            u32::try_from(width).expect("row width exceeds u32"),
        );
        Self {
            codes: Tensor::from_dispatch(codes),
            scales: Tensor::from_dispatch(scales),
            shape: self.shape,
        }
    }

    /// Refinement: zero the unique `rows` in place and
    /// append `extra` zero rows. Exact, and nothing is decoded: padding past
    /// the last row already holds zero codes, new blocks get zero codes.
    pub fn reset_rows(self, rows: &Tensor<1, Int>, extra: usize) -> Self {
        let width = self.shape[1..].iter().product::<usize>();
        let device = self.scales.device();
        let mut codes = self.codes;
        let mut scales = self.scales;
        if rows.dims()[0] > 0 {
            codes = Tensor::from_dispatch(Dispatch::zero_moment_rows(
                codes.into_dispatch(),
                rows.clone().into_dispatch(),
                width as u32,
            ));
        }
        let mut shape = self.shape;
        shape[0] += extra;
        let blocks = shape.iter().product::<usize>().div_ceil(BLOCK_VALUES);
        let new_blocks = blocks - scales.dims()[0];
        if new_blocks > 0 {
            let words = Tensor::<1, Int>::full([new_blocks * BLOCK_VALUES / 4], ZERO_WORD, &device)
                .cast(IntDType::I32);
            codes = Tensor::cat(vec![codes, words], 0);
            scales = Tensor::cat(vec![scales, Tensor::zeros([new_blocks], &device)], 0);
        }
        Self {
            codes,
            scales,
            shape,
        }
    }

    /// `grad` is compact: splat row `r` reads gradient row `grad_rows[r]`.
    pub fn step(
        self,
        params: Tensor<D>,
        grad: Tensor<D>,
        grad_rows: Tensor<1, Int>,
        variance: Tensor<D>,
        lr_scale: Tensor<1>,
        update: MomentUpdate,
        color_dir: bool,
    ) -> (Tensor<D>, Self) {
        assert_eq!(params.dims(), self.shape);
        let count: usize = self.shape.iter().product();
        let rows = self.shape[0];
        assert_eq!(grad_rows.dims(), [rows]);
        let row_width = u32::try_from(count / rows).expect("row width exceeds u32");
        let grad_count: usize = grad.dims().iter().product();
        let (params, codes, scales) = Dispatch::step_moment(
            params.reshape([count]).into_dispatch(),
            grad.reshape([grad_count]).into_dispatch(),
            grad_rows.into_dispatch(),
            self.codes.into_dispatch(),
            self.scales.into_dispatch(),
            variance.reshape([rows]).into_dispatch(),
            lr_scale.into_dispatch(),
            update.beta,
            update.factor,
            update.inv1,
            update.inv2,
            update.epsilon,
            update.lr,
            update.round_seed,
            row_width,
            color_dir,
        );
        (
            Tensor::<1>::from_dispatch(params).reshape(self.shape),
            Self {
                codes: Tensor::from_dispatch(codes),
                scales: Tensor::from_dispatch(scales),
                shape: self.shape,
            },
        )
    }
}

/// Fused full-precision Adam: updates `params`, `m` and `v` (all `[rows, ..]`
/// f32) from a compact `grad`, where splat row `r` reads `grad_rows[r]`.
/// Returns the new `(params, m, v)`; owned buffers are updated in place.
pub(crate) fn adam_step<const D: usize>(
    params: Tensor<D>,
    grad: Tensor<D>,
    grad_rows: Tensor<1, Int>,
    (m, v): (Tensor<D>, Tensor<D>),
    lr_scale: Tensor<1>,
    update: MomentUpdate,
    beta2: f32,
) -> (Tensor<D>, Tensor<D>, Tensor<D>) {
    let shape = params.dims();
    let count: usize = shape.iter().product();
    let row_width = u32::try_from(count / shape[0]).expect("row width exceeds u32");
    let grad_count: usize = grad.dims().iter().product();
    let flat = |t: Tensor<D>| t.reshape([count]).into_dispatch();
    let (params, m, v) = Dispatch::step_adam(
        flat(params),
        grad.reshape([grad_count]).into_dispatch(),
        grad_rows.into_dispatch(),
        flat(m),
        flat(v),
        lr_scale.into_dispatch(),
        update.beta,
        beta2,
        update.inv1,
        update.inv2,
        update.epsilon,
        update.lr,
        row_width,
    );
    let back = |t| Tensor::<1>::from_dispatch(t).reshape(shape);
    (back(params), back(m), back(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[tokio::test]
    #[ignore = "GPU test: run under the shared exclusive GPU flock"]
    async fn gpu_round_trip_preserves_zero_and_isolates_outlier_blocks() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let mut values = vec![0.0f32; 3 * BLOCK_VALUES + 7];
        for (i, x) in values.iter_mut().enumerate().take(BLOCK_VALUES) {
            *x = (i as f32 - 64.0) * 1e-7;
        }
        values[BLOCK_VALUES] = 100.0;
        values[BLOCK_VALUES + 1] = -100.0;
        // Third block is all zero; tail also exercises padding and shape recovery.
        *values.last_mut().unwrap() = 0.5;
        let t = Tensor::<3>::from_data(
            TensorData::new(values.clone(), [values.len(), 1, 1]),
            &device,
        );
        let decoded = BlockMoment::encode(t).decode();
        assert_eq!(decoded.dims(), [values.len(), 1, 1]);
        let actual = decoded
            .into_data_async()
            .await
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        for (block, expected) in values.chunks(BLOCK_VALUES).enumerate() {
            let max = expected.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            for (offset, &x) in expected.iter().enumerate() {
                let y = actual[block * BLOCK_VALUES + offset];
                assert!(y.is_finite());
                assert!((x - y).abs() <= max / 254.0 + max * 1e-6, "{x} != {y}");
                if x == 0.0 {
                    assert_eq!(y, 0.0);
                }
            }
        }
    }
}

/// `t[rows[i], ..] = values[i, ..]` for unique `rows` (refinement row edits),
/// through the row-parallel kernel instead of burn's `select_assign`.
pub(crate) fn assign_rows<const D: usize>(
    t: Tensor<D>,
    rows: Tensor<1, Int>,
    values: Tensor<D>,
) -> Tensor<D> {
    let width = t.dims()[1..].iter().product::<usize>() as u32;
    Tensor::from_dispatch(Dispatch::assign_rows(
        t.into_dispatch(),
        rows.into_dispatch(),
        values.into_dispatch(),
        width,
    ))
}

/// Per-row NaN/inf test over a `[N, ..]` tensor as an `[N]` mask, without
/// materialising an elementwise mask.
pub(crate) fn rows_non_finite<const D: usize>(t: Tensor<D>) -> Tensor<1, burn::tensor::Bool> {
    let width = t.dims()[1..].iter().product::<usize>() as u32;
    let flags: Tensor<1> = Tensor::from_dispatch(Dispatch::rows_non_finite(t.into_dispatch(), width));
    flags.greater_elem(0.5)
}
