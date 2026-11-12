//! Fused fake quantisation for the quantisation-aware tail (`qat.rs`). The
//! tensor-op version this replaces materialised every slice, clamp and
//! concatenation, a few hundred bytes per splat of transients every tail
//! step. The split/join pair: one pass clamps each param into the export's
//! representable range, overwrites it with its quantised value and keeps its
//! offset from it as a small code, so the float params never sit next to a
//! quantised copy (the final snap is the split with its codes dropped).

use brush_cube::{MainBackendBase, create_tensor};
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    Backend, TensorMetadata, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::cubecl::{self, cube, prelude::*};
use burn::tensor::{DType, Shape, f16};

#[backend_extension(Cube, Fusion)]
pub(crate) trait QatOps: Backend {
    /// Raw opacities quantised to the export's 8-bit alpha.
    #[fusion(meta = |raw| raw.clone())]
    fn qat_opacities(raw: FloatTensor<Self>) -> FloatTensor<Self>;

    /// `[N, 10]` transforms' quantised values (means at the export's fixed
    /// point, log scales at 1/16, rotations as is; scalars are the export
    /// lattice's, see `qat.rs`) and each value's offset from its
    /// range-clamped param, as a signed 16-bit code of half a lattice step
    /// (two per word), rounded stochastically so it is unbiased. One pass, in place where owned: the quantised values
    /// overwrite the param, so the pool holds no second buffer of its size.
    #[fusion(meta = |t, _pos_scale, _pos_limit, _scale_lo, _scale_hi, _scale_shift, _scale_unshift, _seed| (t.clone(), TensorSpec::new(Shape::new([t.shape.num_elements() / 10 * TRANSFORM_CODE_WORDS]), DType::I32)))]
    fn qat_transforms_split(
        t: FloatTensor<Self>,
        #[fusion(scalar)] pos_scale: f32,
        #[fusion(scalar)] pos_limit: f32,
        #[fusion(scalar)] scale_lo: f32,
        #[fusion(scalar)] scale_hi: f32,
        #[fusion(scalar)] scale_shift: f32,
        #[fusion(scalar)] scale_unshift: f32,
        #[fusion(scalar)] seed: u32,
    ) -> (FloatTensor<Self>, IntTensor<Self>);

    /// `[N, K, 3]` SH coefficients' quantised values (DC clamped to its
    /// 8-bit range, the rest to [-1, 1] at 6 bits) and each value's offset
    /// from its range-clamped param as a signed 8-bit code of half a lattice
    /// step (four per word), rounded stochastically.
    /// In place where owned, like `qat_transforms_split`.
    #[fusion(meta = |sh, _k, _seed| (sh.clone(), TensorSpec::new(Shape::new([sh.shape.num_elements().div_ceil(4)]), DType::I32)))]
    fn qat_sh_split(
        sh: FloatTensor<Self>,
        k: u32,
        #[fusion(scalar)] seed: u32,
    ) -> (FloatTensor<Self>, IntTensor<Self>);

    /// Stepped transforms plus their decoded offsets (`qat_transforms_split`),
    /// in place where owned.
    #[fusion(meta = |t, _codes, _pos_scale| t.clone())]
    fn qat_transforms_join(
        t: FloatTensor<Self>,
        codes: IntTensor<Self>,
        #[fusion(scalar)] pos_scale: f32,
    ) -> FloatTensor<Self>;

    /// Stepped SH coefficients plus their decoded offsets (`qat_sh_split`),
    /// in place where owned.
    #[fusion(meta = |sh, _codes, _k| sh.clone())]
    fn qat_sh_join(sh: FloatTensor<Self>, codes: IntTensor<Self>, k: u32) -> FloatTensor<Self>;
}

fn owned(t: FloatTensor<MainBackendBase>) -> FloatTensor<MainBackendBase> {
    if t.can_mut() { t } else { t.copy() }
}

impl QatOps for MainBackendBase {
    fn qat_opacities(raw: FloatTensor<Self>) -> FloatTensor<Self> {
        assert!(raw.is_contiguous());
        assert_eq!(raw.dtype, DType::F32, "opacities are f32");
        let n = raw.shape().num_elements();
        let q = create_tensor([n], &raw.device, DType::F32);
        if n == 0 {
            return q;
        }
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&raw.client, n, dim);
        qat_opacities_kernel::launch(
            &raw.client, count, dim,
            raw.clone().into_tensor_arg(), q.clone().into_tensor_arg(),
            ALPHA_EDGE, 1.0 - ALPHA_EDGE,
        );
        q
    }

    fn qat_transforms_split(
        t: FloatTensor<Self>,
        pos_scale: f32,
        pos_limit: f32,
        scale_lo: f32,
        scale_hi: f32,
        scale_shift: f32,
        scale_unshift: f32,
        seed: u32,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let t = owned(t);
        assert!(t.is_contiguous());
        assert_eq!(t.dtype, DType::F32, "transforms are f32");
        let words = t.shape().num_elements() / 10 * TRANSFORM_CODE_WORDS;
        let codes = create_tensor([words], &t.device, DType::I32);
        if words == 0 {
            return (t, codes);
        }
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&t.client, words, dim);
        transforms_split_kernel::launch(
            &t.client, count, dim,
            t.clone().into_tensor_arg(), codes.clone().into_tensor_arg(),
            pos_scale, pos_limit, scale_lo, scale_hi, scale_shift, scale_unshift, 0.5 / pos_scale, seed,
        );
        (t, codes)
    }

    fn qat_sh_split(sh: FloatTensor<Self>, k: u32, seed: u32) -> (FloatTensor<Self>, IntTensor<Self>) {
        let sh = owned(sh);
        assert!(sh.is_contiguous());
        let n = sh.shape().num_elements();
        assert_eq!(sh.shape().num_dims(), 3, "SH coefficients are [N, K, 3]");
        let codes = create_tensor([n.div_ceil(4)], &sh.device, DType::I32);
        if n == 0 {
            return (sh, codes);
        }
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&sh.client, n.div_ceil(4), dim);
        macro_rules! launch {
            ($t:ty) => {
                sh_split_kernel::launch::<$t>(
                    &sh.client, count, dim,
                    sh.clone().into_tensor_arg(), codes.clone().into_tensor_arg(),
                    k, DC_MAX, COLOR_SCALE, 0.5 / COLOR_SCALE, seed,
                )
            };
        }
        match sh.dtype {
            DType::F32 => launch!(f32),
            DType::F16 => launch!(f16),
            dtype => panic!("qat_sh_split: unsupported dtype {dtype:?}"),
        }
        (sh, codes)
    }

    fn qat_transforms_join(t: FloatTensor<Self>, codes: IntTensor<Self>, pos_scale: f32) -> FloatTensor<Self> {
        let t = owned(t);
        assert!(t.is_contiguous() && codes.is_contiguous());
        assert_eq!(t.dtype, DType::F32, "transforms are f32");
        let words = codes.shape().num_elements();
        assert_eq!(words, t.shape().num_elements() / 10 * TRANSFORM_CODE_WORDS);
        if words == 0 {
            return t;
        }
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&t.client, words, dim);
        transforms_join_kernel::launch(
            &t.client, count, dim,
            t.clone().into_tensor_arg(), codes.into_tensor_arg(), 0.5 / pos_scale,
        );
        t
    }

    fn qat_sh_join(sh: FloatTensor<Self>, codes: IntTensor<Self>, k: u32) -> FloatTensor<Self> {
        let sh = owned(sh);
        assert!(sh.is_contiguous() && codes.is_contiguous());
        let n = sh.shape().num_elements();
        if n == 0 {
            return sh;
        }
        let dim = CubeDim::new_1d(256);
        let count = cubecl::calculate_cube_count_elemwise(&sh.client, n.div_ceil(4), dim);
        macro_rules! launch {
            ($t:ty) => {
                sh_join_kernel::launch::<$t>(
                    &sh.client, count, dim,
                    sh.clone().into_tensor_arg(), codes.into_tensor_arg(), k, 0.5 / COLOR_SCALE,
                )
            };
        }
        match sh.dtype {
            DType::F32 => launch!(f32),
            DType::F16 => launch!(f16),
            dtype => panic!("qat_sh_join: unsupported dtype {dtype:?}"),
        }
        sh
    }
}

pub(crate) const COLOR_SCALE: f32 = 0.15 * 255.0;
pub(crate) const DC_MAX: f32 = 127.5 / COLOR_SCALE;
/// Snapped opacity stays inside its code's rounding cell short of 0/1, so the
/// logit is finite and re-encoding returns the same code.
pub(crate) const ALPHA_EDGE: f32 = 0.25 / 255.0;

/// `-1`, `0` or `1`, like burn's `sign`.
#[cube]
fn sign<F: Float>(x: F) -> F {
    let mut s = F::new(0.0);
    if x < F::new(0.0) {
        s = F::new(-1.0);
    }
    if x > F::new(0.0) {
        s = F::new(1.0);
    }
    s
}

/// Rounding half away from zero for values already >= 0 (`qat::round_pos`).
#[cube]
fn round_pos<F: Float>(x: F) -> F {
    F::floor(x + F::new(0.5))
}

#[cube(launch)]
fn qat_opacities_kernel(raw: &Tensor<f32>, q: &mut Tensor<f32>, edge_lo: f32, edge_hi: f32) {
    let i = ABSOLUTE_POS;
    if i < raw.len() {
        // burn's sigmoid, op for op: exp(-ln(exp(-x) + 1)).
        let alpha = f32::exp(-f32::ln(f32::exp(-raw[i]) + 1.0));
        let code = f32::clamp(round_pos::<f32>(alpha * 255.0), 0.0, 255.0);
        let a = f32::clamp(code / 255.0, edge_lo, edge_hi);
        q[i] = f32::ln(a / (-a + 1.0));
    }
}

/// Uniform in (0, 1] from a 32-bit hash of (`key`, `seed`).
#[cube]
fn hash_uniform(key: u32, seed: u32) -> f32 {
    let mut h = key ^ (seed * 0x9e3779b9u32);
    h = (h ^ (h >> 16u32)) * 0x7feb352du32;
    h = (h ^ (h >> 15u32)) * 0x846ca68bu32;
    h = h ^ (h >> 16u32);
    ((h >> 8u32) + 1u32) as f32 / 16777216.0f32
}

/// Transforms value `c` (its column) clamped into the export's range.
#[cube]
fn transform_clamp(x: f32, c: usize, pos_limit: f32, scale_lo: f32, scale_hi: f32) -> f32 {
    let mut m = x;
    if c < 3 {
        m = f32::clamp(x, -pos_limit, pos_limit);
    } else if c >= 7 {
        m = f32::clamp(x, scale_lo, scale_hi);
    }
    m
}

/// A clamped transforms value's quantised value: SPZ's position fixed point
/// and 1/16 log-scale steps.
#[cube]
fn transform_quant(m: f32, c: usize, pos_scale: f32, scale_shift: f32, scale_unshift: f32) -> f32 {
    let mut q = m;
    if c < 3 {
        let code = m * pos_scale;
        q = sign::<f32>(code) * round_pos::<f32>(f32::abs(code)) / pos_scale;
    } else if c >= 7 {
        let code = f32::clamp(round_pos::<f32>((m + scale_shift) * 16.0), 0.0, 255.0);
        q = code / 16.0 + scale_unshift;
    }
    q
}

/// Half a lattice step of transforms column `c` (rotations: none).
#[cube]
fn transform_half(c: usize, pos_half: f32) -> f32 {
    let mut h = f32::new(0.0);
    if c < 3 {
        h = pos_half;
    } else if c >= 7 {
        h = f32::new(0.03125);
    }
    h
}

/// SH value clamped into the export's range (`dc`: a DC value).
#[cube]
fn sh_clamp<F: Float>(x: F, dc: bool, dc_max: f32) -> F {
    let mut d = F::clamp(x, F::new(-1.0), F::new(1.0));
    if dc {
        d = F::clamp(x, F::cast_from(-dc_max), F::cast_from(dc_max));
    }
    d
}

/// A clamped SH value's quantised value: SPZ's 8-bit DC and 6-bit higher SH.
#[cube]
fn sh_quant<F: Float>(d: F, dc: bool, color_scale: f32) -> F {
    // SPZ's 6-bit higher SH: 128 / bucket 4 steps per unit.
    let y = d * F::new(32.0);
    let code = sign::<F>(y) * round_pos::<F>(F::abs(y));
    let mut q = (F::clamp(code * F::new(4.0) + F::new(128.0), F::new(0.0), F::new(255.0)) - F::new(128.0))
        / F::new(128.0);
    if dc {
        let code = round_pos::<F>(d * F::cast_from(color_scale) + F::new(127.5));
        q = (F::clamp(code, F::new(0.0), F::new(255.0)) - F::new(127.5)) / F::cast_from(color_scale);
    }
    q
}

/// `r / h` as a signed code of `levels` steps each side, rounded
/// stochastically (`floor(v + u)`, unbiased), so offsets finer than a code
/// survive on average instead of rounding away every step.
#[cube]
fn offset_code(r: f32, h: f32, levels: f32, key: u32, seed: u32) -> f32 {
    let mut v = 0.0f32;
    if h > 0.0 {
        v = f32::clamp(r / h, -1.0, 1.0) * levels;
    }
    f32::clamp(f32::floor(v + 1.0 - hash_uniform(key, seed)), -levels, levels)
}

/// Offset code words per transforms row: two 16-bit codes per word for the
/// six quantised columns (means and log scales); rotations aren't quantised,
/// so they carry no offset.
const TRANSFORM_CODE_WORDS: usize = 3;

/// Transforms element of code word `w`'s slot `j`: the row's quantised
/// columns 0, 1, 2, 7, 8, 9 in order.
#[cube]
fn coded_transform(w: usize, j: u32) -> usize {
    let k = (w % 3) * 2 + j as usize;
    let mut col = k;
    if k >= 3 {
        col = k + 4;
    }
    (w / 3) * 10 + col
}

/// Two values per thread, one biased 16-bit offset code each.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn transforms_split_kernel(
    t: &mut Tensor<f32>,
    codes: &mut Tensor<u32>,
    pos_scale: f32,
    pos_limit: f32,
    scale_lo: f32,
    scale_hi: f32,
    scale_shift: f32,
    scale_unshift: f32,
    pos_half: f32,
    seed: u32,
) {
    let w = ABSOLUTE_POS;
    if w < codes.len() {
        let mut word = 0u32;
        for j in 0..2u32 {
            let e = coded_transform(w, j);
            let c = e % 10;
            let m = transform_clamp(t[e], c, pos_limit, scale_lo, scale_hi);
            let qe = transform_quant(m, c, pos_scale, scale_shift, scale_unshift);
            t[e] = qe;
            let h = transform_half(c, pos_half);
            let code = offset_code(m - qe, h, 32767.0, e as u32, seed);
            word |= ((code + 32768.0) as u32) << (j * 16u32);
        }
        codes[w] = word;
    }
}

#[cube(launch)]
fn transforms_join_kernel(t: &mut Tensor<f32>, codes: &Tensor<u32>, pos_half: f32) {
    let w = ABSOLUTE_POS;
    if w < codes.len() {
        let word = codes[w];
        for j in 0..2u32 {
            let e = coded_transform(w, j);
            let code = ((word >> (j * 16u32)) & 65535u32) as f32 - 32768.0;
            t[e] += code / 32767.0 * transform_half(e % 10, pos_half);
        }
    }
}

/// Four values per thread, one biased 8-bit offset code each.
#[cube(launch)]
fn sh_split_kernel<F: Float>(
    sh: &mut Tensor<F>,
    codes: &mut Tensor<u32>,
    k: u32,
    dc_max: f32,
    color_scale: f32,
    dc_half: f32,
    seed: u32,
) {
    let w = ABSOLUTE_POS;
    if w < codes.len() {
        let mut word = 0u32;
        for j in 0..4u32 {
            let e = w * 4 + j as usize;
            if e < sh.len() {
                let dc = (e as u32 / 3) % k == 0;
                let d = sh_clamp::<F>(sh[e], dc, dc_max);
                let qe = sh_quant::<F>(d, dc, color_scale);
                sh[e] = qe;
                let mut h = f32::new(0.015625);
                if dc {
                    h = dc_half;
                }
                let r = f32::cast_from(d) - f32::cast_from(qe);
                let code = offset_code(r, h, 127.0, e as u32, seed);
                word |= ((code + 128.0) as u32) << (j * 8u32);
            }
        }
        codes[w] = word;
    }
}

#[cube(launch)]
fn sh_join_kernel<F: Float>(sh: &mut Tensor<F>, codes: &Tensor<u32>, k: u32, dc_half: f32) {
    let w = ABSOLUTE_POS;
    if w < codes.len() {
        let word = codes[w];
        for j in 0..4u32 {
            let e = w * 4 + j as usize;
            if e < sh.len() {
                let code = ((word >> (j * 8u32)) & 255u32) as f32 - 128.0;
                let mut h = f32::new(0.015625);
                if (e as u32 / 3) % k == 0 {
                    h = dc_half;
                }
                sh[e] = F::cast_from(f32::cast_from(sh[e]) + code / 127.0 * h);
            }
        }
    }
}
