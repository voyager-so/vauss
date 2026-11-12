//! The one per-splat random stream every evaluator (Rust, WGSL, GLSL) shares.
//!
//! `h0 = pcg(i ^ seed·0x9E3779B9)`, `h1 = pcg(h0)`, `h2 = pcg(h1)`, ... (u32, wrapping), and a
//! hash becomes a float as `(h >> 8) · 2⁻²⁴`, exact in f32 and in [0, 1). `i` is the splat's
//! index in the package file. Never frame-dependent.

/// PCG-RXS-M-XS hash (Jarzynski & Olano 2020).
#[inline]
pub fn pcg_hash(x: u32) -> u32 {
    let s = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

#[inline]
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// The randoms of splat `i`: `r` (h0), `r3` (h1..h3) for explode jitter, and `axis` (h4..h6)
/// mapped to [-1, 1) for the explode spin axis (normalised by the user).
#[derive(Debug, Clone, Copy)]
pub struct SplatRandoms {
    pub r: f32,
    pub r3: [f32; 3],
    pub axis: [f32; 3],
}

#[inline]
pub fn splat_randoms(i: u32, seed: u32) -> SplatRandoms {
    let mut h = pcg_hash(i ^ seed.wrapping_mul(0x9E37_79B9));
    let r = unit(h);
    let mut next = || {
        h = pcg_hash(h);
        unit(h)
    };
    let r3 = [next(), next(), next()];
    let axis = [next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0];
    SplatRandoms { r, r3, axis }
}

