//! Exact fused attention for head dim 64 on any CubeCL device (WGSL included).
//!
//! Burn's `attention` autotunes between cubek's flash kernels and a materialized
//! fallback. On wgpu with the WGSL compiler the tensor-core flash variants cannot
//! run (no cooperative matrices), so only cubek's scalar "unit" kernel remains,
//! and its small tiles reach ~14% of fp32 peak. This kernel is a plain tiled
//! flash attention: each thread owns one query row (query and accumulator in
//! registers), the workgroup stages key/value tiles in shared memory, and the
//! softmax is computed online in f32. Memory is O(sequence); scores are never
//! materialized.

use burn::backend::{Backend, TensorMetadata};
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, Shape, Tensor};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};

/// Head dimension the kernel is specialized for (DA3-BASE).
pub const HEAD_DIM: usize = 64;

mod kernels {
    use burn::cubecl;
    use burn::cubecl::cube;
    use burn::cubecl::prelude::*;

    use super::HEAD_DIM;

    /// Query rows per workgroup (one per thread).
    pub const BQ: u32 = 64;
    /// Keys per shared-memory tile.
    pub const BK: u32 = 32;
    /// Score row stride in shared memory, padded to avoid bank conflicts.
    const S_STRIDE: u32 = BK + 1;
    const D: u32 = HEAD_DIM as u32;

    /// `out[bh, i, :] = softmax(q[bh, i, :] · k[bh, :, :]ᵀ · scale) · v[bh, :, :]`.
    /// Tensors are contiguous `[batch_heads, seq, 64]` f32.
    #[cube(launch)]
    pub fn flash_attention_kernel(
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        out: &mut Tensor<f32>,
        seq_q: u32,
        seq_kv: u32,
        scale: f32,
    ) {
        let bh = CUBE_POS_Y;
        let row = CUBE_POS_X * BQ + UNIT_POS;
        let active = row < seq_q;
        let q_base = ((bh * seq_q + select(active, row, 0u32)) * D) as usize;
        let kv_base = bh * seq_kv * D;

        let mut qr = Array::<f32>::new(HEAD_DIM);
        let mut o = Array::<f32>::new(HEAD_DIM);
        #[unroll]
        for d in 0..HEAD_DIM {
            qr[d] = q[q_base + d] * scale;
            o[d] = 0.0f32;
        }
        let mut m = f32::new(-3.0e38);
        let mut l = 0.0f32;

        let mut ks = Shared::<[f32]>::new_slice((BK * D) as usize);
        let mut vs = Shared::<[f32]>::new_slice((BK * D) as usize);
        let mut ss = Shared::<[f32]>::new_slice((BQ * S_STRIDE) as usize);
        let s_row = UNIT_POS * S_STRIDE;

        let mut start = 0u32;
        while start < seq_kv {
            let n = min(BK, seq_kv - start);
            // Cooperative, coalesced load of BK keys and values (zeros past the end).
            #[unroll]
            for i in 0..(BK * D / BQ) {
                let idx = i * BQ + UNIT_POS;
                let key = idx / D;
                let src = (kv_base + start * D + idx) as usize;
                let in_range = key < n;
                ks[idx as usize] = select(in_range, k[select(in_range, src, 0usize)], 0.0f32);
                vs[idx as usize] = select(in_range, v[select(in_range, src, 0usize)], 0.0f32);
            }
            sync_cube();

            // Scores for this tile, and their max.
            let mut tile_max = f32::new(-3.0e38);
            let mut j = 0u32;
            while j < n {
                let kb = (j * D) as usize;
                let mut acc = 0.0f32;
                #[unroll]
                for d in 0..HEAD_DIM {
                    acc += qr[d] * ks[kb + d];
                }
                ss[(s_row + j) as usize] = acc;
                tile_max = max(tile_max, acc);
                j += 1u32;
            }

            // Online softmax: rescale once per tile, then accumulate p · v.
            let m_new = max(m, tile_max);
            let corr = f32::exp(m - m_new);
            l *= corr;
            #[unroll]
            for d in 0..HEAD_DIM {
                o[d] *= corr;
            }
            let mut j = 0u32;
            while j < n {
                let p = f32::exp(ss[(s_row + j) as usize] - m_new);
                l += p;
                let vb = (j * D) as usize;
                #[unroll]
                for d in 0..HEAD_DIM {
                    o[d] += p * vs[vb + d];
                }
                j += 1u32;
            }
            m = m_new;
            sync_cube();
            start += BK;
        }

        if active {
            let inv = 1.0f32 / l;
            #[unroll]
            for d in 0..HEAD_DIM {
                out[q_base + d] = o[d] * inv;
            }
        }
    }
}

#[burn::backend::backend_extension(Cube, Fusion)]
pub trait FlashOps: Backend {
    /// Attention over contiguous `[batch_heads, seq, 64]` f32 tensors, default
    /// scale `1/sqrt(64)`, no mask.
    #[fusion(dtype = q, shape = q)]
    fn flash_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

impl FlashOps for CubeBackend {
    fn flash_attention(q: CubeTensor, k: CubeTensor, v: CubeTensor) -> CubeTensor {
        use burn::cubecl::prelude::{CubeCount, CubeDim};

        let (q, k, v) = (into_contiguous(q), into_contiguous(k), into_contiguous(v));
        let qd = q.shape().as_slice().to_vec();
        let kd = k.shape().as_slice().to_vec();
        assert!(
            qd.len() == 3 && kd.len() == 3 && qd[2] == HEAD_DIM && kd[2] == HEAD_DIM && qd[0] == kd[0],
            "flash_attention expects [bh, seq, 64] q/k/v"
        );
        assert_eq!(q.dtype, DType::F32, "flash_attention is f32 only");
        let (bh, seq_q, seq_kv) = (qd[0] as u32, qd[1] as u32, kd[1] as u32);

        let client = q.client.clone();
        let out = burn_cubecl::ops::numeric::zeros_client(
            client.clone(),
            q.device.clone(),
            Shape::new([qd[0], qd[1], HEAD_DIM]),
            DType::F32,
        );
        kernels::flash_attention_kernel::launch(
            &client,
            CubeCount::Static(seq_q.div_ceil(kernels::BQ), bh, 1),
            CubeDim::new_1d(kernels::BQ),
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            seq_q,
            seq_kv,
            1.0f32 / (HEAD_DIM as f32).sqrt(),
        );
        out
    }
}

/// Exact `softmax(q kᵀ / sqrt(64)) v` for `[b, heads, seq, 64]` f32 tensors, the
/// same as `burn::tensor::module::attention(q, k, v, None, None, Default::default())`.
pub fn attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    let [b, h, tq, d] = q.dims();
    let tk = k.dims()[2];
    debug_assert_eq!(d, HEAD_DIM);
    let flat = |x: Tensor<4>, t: usize| x.reshape([b * h, t, d]);
    let out = <burn::backend::Dispatch as FlashOps>::flash_attention(
        flat(q, tq).into_dispatch(),
        flat(k, tk).into_dispatch(),
        flat(v, tk).into_dispatch(),
    );
    Tensor::<3>::from_dispatch(out).reshape([b, h, tq, d])
}
