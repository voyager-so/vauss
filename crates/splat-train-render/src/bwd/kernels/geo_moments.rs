//! Packed geometry/opacity Adam moments, one u32 per value.
//!
//! With Adam's epsilon `e`, `u = m / (sqrt(v) + e)` and
//! `l = ln(1 + sqrt(v) / e)`. Low half: `u` sign-magnitude, 15-bit magnitude
//! over [0, GEO_U_MAX]. High half: `l` as a 16-bit code over [0, GEO_L_MAX].
//! Decode: `sqrt(v) = e (exp(l) - 1)`, `m = u e exp(l)`. Word 0 decodes to
//! exactly m = v = 0, so zero-filled rows are reset moments.
//! |u| <= ~7.3 for beta1 = 0.9, beta2 = 0.999; larger values clamp.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;

const GEO_U_MAX: f32 = 8.0;
const GEO_U_CODES: f32 = 32767.0;
const GEO_U_STEP: f32 = GEO_U_MAX / GEO_U_CODES;
const GEO_U_PER_UNIT: f32 = GEO_U_CODES / GEO_U_MAX;
// e^64 * 1e-15 ~ 6e12 bounds sqrt(v); a code step is ~0.1% relative in sqrt(v).
const GEO_L_MAX: f32 = 64.0;
const GEO_L_CODES: f32 = 65535.0;
const GEO_L_STEP: f32 = GEO_L_MAX / GEO_L_CODES;
const GEO_L_PER_UNIT: f32 = GEO_L_CODES / GEO_L_MAX;

/// `exp(l)` of a packed word; `sqrt(v) = e (exp(l) - 1)`, so code 0 is v = 0.
#[cube]
pub fn geo_growth(word: u32) -> f32 {
    f32::exp((word >> 16u32) as f32 * GEO_L_STEP)
}

/// Normalized first moment `u` of a packed word; `m = u e exp(l)`.
#[cube]
pub fn geo_normalized(word: u32) -> f32 {
    let magnitude = (word & 0x7fffu32) as f32 * GEO_U_STEP;
    select((word & 0x8000u32) != 0u32, -magnitude, magnitude)
}

/// `round_l`/`round_u` in [0, 1) add to the code before flooring: 0.5 rounds
/// to nearest, a uniform sample rounds stochastically. The optimizer needs
/// the latter: one beta2 decay moves `l` by ~5e-4, under half a code step,
/// so round-to-nearest would freeze decaying variance and momentum.
/// Out-of-range values clamp to the end codes.
#[cube]
pub fn pack_geo(m: f32, v: f32, epsilon: f32, round_l: f32, round_u: f32) -> u32 {
    let l = f32::ln(1.0f32 + f32::sqrt(f32::max(v, 0.0f32)) / epsilon) * GEO_L_PER_UNIT;
    let l_code = f32::floor(clamp(l + round_l, 0.0f32, GEO_L_CODES));
    // Normalize by the stored scale, so m decodes with only u's error.
    let scale = epsilon * f32::exp(l_code * GEO_L_STEP);
    let u = f32::abs(m) / scale * GEO_U_PER_UNIT;
    let u_code = f32::floor(clamp(u + round_u, 0.0f32, GEO_U_CODES)) as u32;
    let sign = select(m < 0.0f32, 0x8000u32, 0u32);
    ((l_code as u32) << 16u32) | sign | u_code
}

/// Per-element, per-step integer hash for stochastic rounding (the same
/// mixer the SH f16 writes use).
#[cube]
pub fn mix_hash(index: u32, step: u32) -> u32 {
    let mut hash = index ^ (step * 0x9e3779b9u32);
    hash = (hash ^ (hash >> 16u32)) * 0x7feb352du32;
    hash = (hash ^ (hash >> 15u32)) * 0x846ca68bu32;
    hash ^ (hash >> 16u32)
}

/// One f32 Adam step on `params[index]` with its packed moments in
/// `words[index]`, re-encoded with stochastic rounding from `seed`.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn adam_packed(
    params: &mut Tensor<f32>,
    words: &mut Tensor<u32>,
    index: usize,
    g: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    inv1: f32,
    inv2: f32,
    epsilon: f32,
    seed: u32,
) {
    let word = words[index];
    let growth = geo_growth(word);
    let root = epsilon * (growth - 1.0f32);
    let m1 = geo_normalized(word) * (epsilon * growth) * beta1 + g * (1.0f32 - beta1);
    let v1 = root * root * beta2 + g * g * (1.0f32 - beta2);
    let step = (m1 * inv1) / (f32::sqrt(v1 * inv2) + epsilon);
    params[index] = params[index] - step * lr;
    let hash = mix_hash(index as u32, seed);
    words[index] = pack_geo(
        m1,
        v1,
        epsilon,
        (hash >> 16u32) as f32 / 65536.0f32,
        (hash & 0xffffu32) as f32 / 65536.0f32,
    );
}

/// Round-to-nearest encode; decode-then-encode is idempotent.
#[cube(launch)]
pub fn encode_geo_kernel(m: &Tensor<f32>, v: &Tensor<f32>, words: &mut Tensor<u32>, epsilon: f32) {
    let index = ABSOLUTE_POS;
    if index < words.len() {
        words[index] = pack_geo(m[index], v[index], epsilon, 0.5f32, 0.5f32);
    }
}

#[cube(launch)]
pub fn decode_geo_kernel(
    words: &Tensor<u32>,
    m: &mut Tensor<f32>,
    v: &mut Tensor<f32>,
    epsilon: f32,
) {
    let index = ABSOLUTE_POS;
    if index < words.len() {
        let word = words[index];
        let growth = geo_growth(word);
        let root = epsilon * (growth - 1.0f32);
        m[index] = geo_normalized(word) * (epsilon * growth);
        v[index] = root * root;
    }
}

/// Zero `row_width` words of each (unique) row in `rows`: a reset moment.
#[cube(launch)]
pub fn zero_geo_rows_kernel(words: &mut Tensor<u32>, rows: &Tensor<u32>, row_width: u32) {
    let pos = ABSOLUTE_POS as u32;
    if pos < rows.len() as u32 * row_width {
        let index = rows[(pos / row_width) as usize] * row_width + pos % row_width;
        words[index as usize] = 0u32;
    }
}
