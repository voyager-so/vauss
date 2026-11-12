//! Render-specific cube helpers. Generic math (`Vec3A`, `Quat`, `Mat3`,
//! `Sym2`, `sigmoid`, `is_finite_*`, `calc_sigma`, `inverse_sym2`,
//! `det2_strict`) lives in [`brush_cube`] — re-exported here for
//! convenience.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;
use burn::cubecl::std::FastDivmod;

use super::types::{Mat3, PixelRect, ProjectUniforms, Quat, Splat, Sym2, TileBbox, Vec3A};
use crate::kernels::camera_model::{CameraModel, calculate_project_jacobian};
pub use brush_cube::{calc_sigma, is_finite_f32, sigmoid};

pub const TILE_WIDTH: u32 = 16;
pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

/// Smoothstep ramp centered at 1/255: zero below `MID - BAND/2`, one
/// above `MID + BAND/2`. C^1 in alpha. Selected at kernel-compile-time
/// via `RasterPass::BackwardSmoothCutoff` — the rasterizer's
/// `smooth_cutoff` comptime gate routes around this for the production
/// hard-step path.
pub const ALPHA_CUTOFF_MID: f32 = 1.0 / 255.0;
pub const ALPHA_CUTOFF_BAND: f32 = 1.0e-3;

#[cube]
pub fn alpha_cutoff_weight(alpha: f32) -> f32 {
    let t = f32::clamp(
        (alpha - (ALPHA_CUTOFF_MID - 0.5f32 * ALPHA_CUTOFF_BAND)) / ALPHA_CUTOFF_BAND,
        0.0f32,
        1.0f32,
    );
    t * t * (3.0f32 - 2.0f32 * t)
}

#[cube]
pub fn alpha_cutoff_weight_deriv(alpha: f32) -> f32 {
    let low = ALPHA_CUTOFF_MID - 0.5f32 * ALPHA_CUTOFF_BAND;
    let high = ALPHA_CUTOFF_MID + 0.5f32 * ALPHA_CUTOFF_BAND;
    let inside = alpha > low && alpha < high;
    let t = (alpha - low) / ALPHA_CUTOFF_BAND;
    select(
        inside,
        (6.0f32 * t - 6.0f32 * t * t) / ALPHA_CUTOFF_BAND,
        0.0f32,
    )
}

/// `f32` lanes per projected splat. Layout matches `Splat`:
///   0:xy_x, 1:xy_y, 2:conic_x, 3:conic_y, 4:conic_z, 5:color_a,
///   6:color_r, 7:color_g, 8:color_b.
pub const PROJECTED_LANES: u32 = 9;
pub const PROJECTED_LANES_USIZE: usize = PROJECTED_LANES as usize;

/// Per-visible-splat side buffer (`splat_aux`), indexed like `projected`.
/// Gaussian mode only fills it for the geometry pass:
///   0:view-space depth, 1..4:camera-facing view-space normal.
/// Surfel (2DGS) mode always fills it:
///   0..9: rows of the ray transform `M = K [a b c]` (screen-homogeneous
///   point = `M (u, v, 1)`), 9..11: projected center, 11..14: normal.
pub const AUX_LANES_GEOM: u32 = 4;
pub const AUX_LANES_SURFEL: u32 = 14;
pub const AUX_SURFEL_CENTER: u32 = 9;
pub const AUX_SURFEL_NORMAL: u32 = 11;

/// Surfel intersections closer than this (view-space z) are rejected, the
/// same near limit `project_forward` culls pinhole centers at.
pub const SURFEL_NEAR: f32 = 0.01;

/// Lanes per splat in `splat_aux`; 0 means the buffer is a placeholder.
pub const fn aux_lanes(surfel: bool, geometry: bool) -> u32 {
    if surfel {
        AUX_LANES_SURFEL
    } else if geometry {
        AUX_LANES_GEOM
    } else {
        0
    }
}

/// Per-pixel geometry outputs of the geometry pass:
///   0:sum(w*z) (divide by alpha for expected depth), 1:median depth
///   (last contributor while transmittance > 0.5; 0 if none),
///   2..5:sum(w*normal) in view space, 5:sum(w*m(z)), 6:sum(w*m(z)^2) for
///   the depth distortion (`A C - B^2` = sum over pairs of
///   w_i w_j (m_i - m_j)^2), 7:intersection index of the median contributor
///   (u32 bits; u32::MAX if none), which the backward uses to route the
///   median depth's gradient.
pub const GEOM_OUT_CHANNELS: u32 = 8;

/// Distortion depth map `m(z) = far / (far - near) * (1 - near / z)` (the
/// 2DGS release's NDC-like depth).
pub const DIST_NEAR: f32 = 0.2;
pub const DIST_FAR: f32 = 100.0;

/// 2DGS distortion depth: NDC-style map of view depth into [0, 1). Depths
/// below `DIST_NEAR` clamp to it: projection only culls below 0.01, and
/// between the two the unclamped map goes to -19 with a 1/z^2 gradient
/// (2DGS's own rasterizer culls at 0.2, so it never sees that range).
#[cube]
pub fn dist_depth(z: f32) -> f32 {
    DIST_FAR / (DIST_FAR - DIST_NEAR) * (1.0f32 - DIST_NEAR / max(z, DIST_NEAR))
}

#[cube]
pub fn compact_bits_16(v: u32) -> u32 {
    let mut x = v & 0x55555555u32;
    x = (x | (x >> 1u32)) & 0x33333333u32;
    x = (x | (x >> 2u32)) & 0x0F0F0F0Fu32;
    x = (x | (x >> 4u32)) & 0x00FF00FFu32;
    x = (x | (x >> 8u32)) & 0x0000FFFFu32;
    x
}

/// Decode a tile-internal Morton id to (px, py) coordinates within the image.
///
/// `tiles_per_row` arrives as a [`FastDivmod`] because the tile row/column
/// split is a runtime divide on the per-pixel path; the magic numbers are
/// precomputed host-side. `TILE_SIZE` is a constant, so those two stay plain.
#[cube]
pub fn map_1d_to_2d(id: u32, tiles_per_row: FastDivmod<u32>) -> (u32, u32) {
    let tile_id = id / TILE_SIZE;
    let within = id % TILE_SIZE;
    let (tile_y, tile_x) = tiles_per_row.div_mod(tile_id);
    let mx = compact_bits_16(within);
    let my = compact_bits_16(within >> 1u32);
    (tile_x * TILE_WIDTH + mx, tile_y * TILE_WIDTH + my)
}

/// Splat half-extent along x / y from the packed conic. Returns
/// `(-1, -1)` when the conic is degenerate so the caller bails on
/// `extent.x < 0`.
#[cube]
pub fn compute_bbox_extent(conic: Sym2, power_threshold: f32) -> (f32, f32) {
    let det = conic.c00 * conic.c11 - conic.c01 * conic.c01;
    let degenerate = det <= 0.0f32;
    let inv_det = select(degenerate, 0.0f32, 1.0f32 / det);
    let ex = f32::sqrt(2.0f32 * power_threshold * conic.c11 * inv_det);
    let ey = f32::sqrt(2.0f32 * power_threshold * conic.c00 * inv_det);
    (
        select(degenerate, -1.0f32, ex),
        select(degenerate, -1.0f32, ey),
    )
}

#[cube]
pub fn tile_rect(tx: u32, ty: u32) -> PixelRect {
    let min_x = (tx * TILE_WIDTH) as f32;
    let min_y = (ty * TILE_WIDTH) as f32;
    PixelRect {
        min_x,
        min_y,
        max_x: min_x + TILE_WIDTH as f32,
        max_y: min_y + TILE_WIDTH as f32,
    }
}

/// Pixel-space center +/- dims clamped to a `(bw, bh)` viewport. Used by
/// `get_tile_bbox` to compute the tile-grid bbox a splat covers.
#[cube]
pub fn get_bbox(cx: f32, cy: f32, dx: f32, dy: f32, bw: u32, bh: u32) -> TileBbox {
    let bwf = bw as f32;
    let bhf = bh as f32;
    TileBbox {
        min_x: clamp(cx - dx, 0.0f32, bwf) as u32,
        min_y: clamp(cy - dy, 0.0f32, bhf) as u32,
        max_x: clamp(cx + dx + 1.0f32, 0.0f32, bwf) as u32,
        max_y: clamp(cy + dy + 1.0f32, 0.0f32, bhf) as u32,
    }
}

#[cube]
pub fn get_tile_bbox(
    pix_cx: f32,
    pix_cy: f32,
    pix_ex: f32,
    pix_ey: f32,
    tile_bw: u32,
    tile_bh: u32,
) -> TileBbox {
    let tw = TILE_WIDTH as f32;
    get_bbox(
        pix_cx / tw,
        pix_cy / tw,
        pix_ex / tw,
        pix_ey / tw,
        tile_bw,
        tile_bh,
    )
}

/// 2D covariance from scale, quat, mean_c and view params. Returns the
/// symmetric covariance as `Sym2`, with a post-scale clamp so huge-but-
/// finite inputs don't overflow the `det` of the eventual conic.
#[cube]
pub fn calc_cov2d(
    scale: Vec3A,
    quat: Quat,
    mean_c: Vec3A,
    u: ProjectUniforms,
    #[comptime] camera_model: CameraModel,
) -> Sym2 {
    let ns = u.view_rotation().mul_mat3(quat.to_mat3()).mul_diag(scale);
    let cam_jac = calculate_project_jacobian(
        mean_c,
        u.jacobian_clamp_limits,
        u.pinhole_params,
        camera_model,
    );

    // V = J * N_s (J is 2x3, N_s is 3x3, V is 2x3).
    let v = cam_jac.mul_mat3(ns);

    // raw = V * V^T (2x2 symmetric).
    let raw = v.gram_matrix();

    // Clamp so max |entry| <= 1e18 — keeps det inside f32 range and
    // preserves PSD / off-diagonal-to-diagonal ratio for huge log_scale
    // training states.
    let lim = 1.0e18f32;
    let max_abs = raw.max_abs();
    let scale_down = select(max_abs > lim, lim / max_abs, 1.0f32);

    raw.scale(scale_down)
}

/// MIP-aware blur compensation. Adds `cov_blur` to the diagonal of the
/// passed-in cov2d and returns the new `Sym2` plus the compensation
/// factor (1.0 when `mip_splatting=false`).
#[cube]
pub fn compensate_cov2d(c: Sym2, #[comptime] mip_splatting: bool) -> (Sym2, f32) {
    let cov_blur = comptime![if mip_splatting { 0.1f32 } else { 0.3f32 }];
    let blurred = Sym2 {
        c00: c.c00 + cov_blur,
        c01: c.c01,
        c11: c.c11 + cov_blur,
    };
    let mut filter_comp = f32::cast_from(1.0f32);
    if comptime![mip_splatting] {
        let det_raw = max(c.det2_strict(), 0.0f32);
        let det_blurred = blurred.det2_strict();
        filter_comp = f32::sqrt(det_raw / det_blurred);
    }
    (blurred, filter_comp)
}

/// Walk the tiles in `bb` in linearised mod/div order and count those
/// that pass `will_primitive_contribute`. Shared between
/// `project_forward` and `map_gaussians_to_intersect` so both dispatches
/// run *byte-identical* loop bodies. Drift between the two counts would
/// leave uninitialised slots in `compact_gid_from_isect`; map_gaussians
/// pads with a sentinel `tile_id` defensively in case it still happens.
#[cube]
pub fn count_contributing_tiles(
    bb: TileBbox,
    xy_x: f32,
    xy_y: f32,
    conic: Sym2,
    power_threshold: f32,
) -> u32 {
    let bb_w = bb.max_x - bb.min_x;
    let num_tiles_bbox = (bb.max_y - bb.min_y) * bb_w;
    let mut num_tiles_hit = 0u32;
    for tile_idx in 0u32..num_tiles_bbox {
        let tx = (tile_idx % bb_w) + bb.min_x;
        let ty = (tile_idx / bb_w) + bb.min_y;
        let rect = tile_rect(tx, ty);
        if will_primitive_contribute(rect, xy_x, xy_y, conic, power_threshold) {
            num_tiles_hit += 1u32;
        }
    }
    num_tiles_hit
}

/// Conservative tile-vs-gaussian intersection test (StopThePop).
#[cube]
pub fn will_primitive_contribute(
    rect: PixelRect,
    mx: f32,
    my: f32,
    conic: Sym2,
    power_threshold: f32,
) -> bool {
    let x_left = mx < rect.min_x;
    let x_right = mx > rect.max_x;
    let in_x_range = !(x_left || x_right);
    let y_above = my < rect.min_y;
    let y_below = my > rect.max_y;
    let in_y_range = !(y_above || y_below);

    let mut hit = in_x_range && in_y_range;
    if !hit {
        let corner_x = select(x_left, rect.min_x, rect.max_x);
        let corner_y = select(y_above, rect.min_y, rect.max_y);
        let width = rect.max_x - rect.min_x;
        let height = rect.max_y - rect.min_y;
        let dxf = select(x_left, width, -width);
        let dyf = select(y_above, height, -height);
        let diff_x = mx - corner_x;
        let diff_y = my - corner_y;

        let tx_raw =
            (dxf * conic.c00 * diff_x + dxf * conic.c01 * diff_y) / (dxf * conic.c00 * dxf);
        let ty_raw =
            (dyf * conic.c01 * diff_x + dyf * conic.c11 * diff_y) / (dyf * conic.c11 * dyf);
        let tx = select(in_y_range, 0.0f32, clamp(tx_raw, 0.0f32, 1.0f32));
        let ty = select(in_x_range, 0.0f32, clamp(ty_raw, 0.0f32, 1.0f32));

        let max_x = corner_x + tx * dxf;
        let max_y = corner_y + ty * dyf;
        hit = calc_sigma(max_x, max_y, conic, mx, my) <= power_threshold;
    }
    hit
}

/// One surfel (2DGS) projected for binning and rasterization.
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct SurfelProj {
    /// Rows of `M = K [s_u t_u, s_v t_v, p_c]` in view space.
    pub m: Mat3,
    /// Center and conic of an ellipse covering every pixel where the surfel
    /// (or its screen-space low-pass) reaches `opac * G >= 1/255`, in the
    /// `calc_sigma <= power_threshold` convention the tile binning uses.
    pub bin_x: f32,
    pub bin_y: f32,
    pub bin_conic: Sym2,
    pub valid: bool,
}

/// Project a surfel: tangent axes `a = s_u W t_u`, `b = s_v W t_v` and view
/// center `c` give `M = K [a b c]`. The opacity cutoff ellipse
/// `u² + v² <= k²` (`k² = 2 power_threshold`) maps to the screen conic with
/// dual `M diag(k², k², -1) Mᵀ`; its center and shape come straight from
/// that dual. The ellipse is then grown to cover the low-pass disk of radius
/// `sqrt(power_threshold)` around the projected center (Minkowski bound
/// `(1 + 1/t) S + (1 + t) r² I`). Invalid when the cutoff ellipse touches
/// the camera plane.
#[cube]
pub fn surfel_project(
    scale: Vec3A,
    quat: Quat,
    mean_c: Vec3A,
    u: ProjectUniforms,
    power_threshold: f32,
) -> SurfelProj {
    let rot = u.view_rotation().mul_mat3(quat.to_mat3());
    let a = rot.col0().scale(scale.x());
    let b = rot.col1().scale(scale.y());
    let p = u.pinhole_params;
    let ka = Vec3A::new(p.fx * a.x() + p.cx * a.z(), p.fy * a.y() + p.cy * a.z(), a.z());
    let kb = Vec3A::new(p.fx * b.x() + p.cx * b.z(), p.fy * b.y() + p.cy * b.z(), b.z());
    let kc = Vec3A::new(
        p.fx * mean_c.x() + p.cx * mean_c.z(),
        p.fy * mean_c.y() + p.cy * mean_c.z(),
        mean_c.z(),
    );
    let m = Mat3::from_cols(ka, kb, kc);
    let r0 = m.row0();
    let r1 = m.row1();
    let r2 = m.row2();

    let k2 = 2.0f32 * power_threshold;
    let d00 = k2 * (r0.x() * r0.x() + r0.y() * r0.y()) - r0.z() * r0.z();
    let d01 = k2 * (r0.x() * r1.x() + r0.y() * r1.y()) - r0.z() * r1.z();
    let d11 = k2 * (r1.x() * r1.x() + r1.y() * r1.y()) - r1.z() * r1.z();
    let d02 = k2 * (r0.x() * r2.x() + r0.y() * r2.y()) - r0.z() * r2.z();
    let d12 = k2 * (r1.x() * r2.x() + r1.y() * r2.y()) - r1.z() * r2.z();
    let d22 = k2 * (r2.x() * r2.x() + r2.y() * r2.y()) - r2.z() * r2.z();

    // The ellipse is fully in front of the camera iff d22 < 0.
    let n = -d22;
    let front = n > 1.0e-12f32;
    let inv_n = select(front, 1.0f32 / n, 0.0f32);
    let x0 = -d02 * inv_n;
    let y0 = -d12 * inv_n;
    let s00 = d00 * inv_n + x0 * x0;
    let s01 = d01 * inv_n + x0 * y0;
    let s11 = d11 * inv_n + y0 * y0;

    let r2_filter = max(power_threshold, 0.0f32);
    // t = a / r (a = mean semi-axis) makes the bound tight for a circle:
    // ((1 + 1/t) a^2 + (1 + t) r^2 = (a + r)^2).
    let half_tr = max(0.5f32 * (s00 + s11), 1.0e-12f32);
    let t = clamp(f32::sqrt(half_tr / max(r2_filter, 1.0e-12f32)), 1.0e-3f32, 1.0e3f32);
    let ga = 1.0f32 + 1.0f32 / t;
    let gb = (1.0f32 + t) * r2_filter;
    let grown = Sym2 {
        c00: ga * s00 + gb,
        c01: ga * s01,
        c11: ga * s11 + gb,
    };
    let det = grown.c00 * grown.c11 - grown.c01 * grown.c01;
    let inv = grown.inverse();
    let bin_conic = Sym2 {
        c00: inv.c00 * k2,
        c01: inv.c01 * k2,
        c11: inv.c11 * k2,
    };
    let valid = front
        && det > 0.0f32
        && is_finite_f32(x0)
        && is_finite_f32(y0)
        && bin_conic.is_finite();
    SurfelProj {
        m,
        bin_x: x0,
        bin_y: y0,
        bin_conic,
        valid,
    }
}

/// Camera-facing view-space normal of a surfel: the rotated local +Z.
#[cube]
pub fn surfel_normal(quat: Quat, mean_c: Vec3A, u: ProjectUniforms) -> Vec3A {
    let n = u.view_rotation().mul_mat3(quat.to_mat3()).col2();
    n.scale(select(n.dot(mean_c) > 0.0f32, -1.0f32, 1.0f32))
}

/// Read just the spatial fields (xy + conic + alpha) of a projected
/// splat. Used by `map_gaussians_to_intersect`, which doesn't need the
/// colors.
#[cube]
pub fn read_main_splat(projected: &Tensor<f32>, idx: u32) -> (f32, f32, Sym2, f32) {
    let b = (idx * PROJECTED_LANES) as usize;
    (
        projected[b],
        projected[b + 1],
        Sym2 {
            c00: projected[b + 2],
            c01: projected[b + 3],
            c11: projected[b + 4],
        },
        projected[b + 5],
    )
}

/// Read one projected splat from the flat `Tensor<f32>` storage.
#[cube]
pub fn read_projected_splat(projected: &Tensor<f32>, idx: u32) -> Splat {
    let b = (idx * PROJECTED_LANES) as usize;
    Splat {
        xy_x: projected[b],
        xy_y: projected[b + 1],
        conic_x: projected[b + 2],
        conic_y: projected[b + 3],
        conic_z: projected[b + 4],
        color_a: projected[b + 5],
        color_r: projected[b + 6],
        color_g: projected[b + 7],
        color_b: projected[b + 8],
    }
}

#[cube]
pub fn write_projected_splat(projected: &mut Tensor<f32>, idx: u32, splat: Splat) {
    let b = (idx * PROJECTED_LANES) as usize;
    projected[b] = splat.xy_x;
    projected[b + 1] = splat.xy_y;
    projected[b + 2] = splat.conic_x;
    projected[b + 3] = splat.conic_y;
    projected[b + 4] = splat.conic_z;
    projected[b + 5] = splat.color_a;
    projected[b + 6] = splat.color_r;
    projected[b + 7] = splat.color_g;
    projected[b + 8] = splat.color_b;
}

/// View-space transform of a world-space mean using the project uniforms'
/// 3x4 viewmat (column-major).
#[cube]
pub fn world_to_cam(mean: Vec3A, u: ProjectUniforms) -> Vec3A {
    u.view_rotation().mul_vec3(mean).add(u.view_translation())
}

#[cube]
pub fn read_mean_viewspace(transforms: &Tensor<f32>, base: usize, u: ProjectUniforms) -> Vec3A {
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    world_to_cam(mean, u)
}

#[cube]
pub fn read_scale(transforms: &Tensor<f32>, base: usize, log_scale_offset: f32) -> Vec3A {
    Vec3A::new(
        f32::exp(transforms[base + 7] + log_scale_offset),
        f32::exp(transforms[base + 8] + log_scale_offset),
        f32::exp(transforms[base + 9] + log_scale_offset),
    )
}

/// Opacity clamp applied after the floor compensation, matching
/// `fold_min_scale` on the host.
const FLOOR_OPACITY_EPS: f32 = 1e-6;

/// One splat's scale and opacity after the Mip-Splatting 3D filter, plus
/// what the backward needs to differentiate through it.
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct ScaleFloor {
    /// `s' = sqrt(s² + f²)` per axis.
    pub scale: Vec3A,
    /// Base opacity `sigmoid(raw) · coef`, clamped.
    pub opac: f32,
    /// Energy compensation `Π s / s'`: the inflated splat keeps its mass.
    pub coef: f32,
    /// `d(log s') / d(log s) = s² / (s² + f²)` per axis.
    pub ratio_sq: Vec3A,
    /// Whether the opacity clamp lets gradient through.
    pub opac_open: bool,
}

/// Apply the floor `min_scale[gid]` to a splat's scale and sigmoid opacity,
/// matching `fold_min_scale` on the host. With `has_min_scale` false this is
/// the identity and `min_scale` is never read.
#[cube]
pub fn apply_scale_floor(
    scale: Vec3A,
    opac_sig: f32,
    min_scale: &Tensor<f32>,
    gid: u32,
    #[comptime] has_min_scale: bool,
) -> ScaleFloor {
    let mut floored = scale;
    let mut opac = opac_sig;
    let mut coef = 1.0f32;
    let mut ratio_sq = Vec3A::new(1.0f32, 1.0f32, 1.0f32);
    let mut opac_open = true;
    if has_min_scale {
        let f = min_scale[gid as usize];
        let f2 = f * f;
        floored = Vec3A::new(
            f32::sqrt(scale.x() * scale.x() + f2),
            f32::sqrt(scale.y() * scale.y() + f2),
            f32::sqrt(scale.z() * scale.z() + f2),
        );
        let rx = scale.x() / floored.x();
        let ry = scale.y() / floored.y();
        let rz = scale.z() / floored.z();
        coef = rx * ry * rz;
        ratio_sq = Vec3A::new(rx * rx, ry * ry, rz * rz);
        let o = opac_sig * coef;
        opac_open = o > FLOOR_OPACITY_EPS && o < 1.0f32 - FLOOR_OPACITY_EPS;
        opac = clamp(o, FLOOR_OPACITY_EPS, 1.0f32 - FLOOR_OPACITY_EPS);
    }
    ScaleFloor {
        scale: floored,
        opac,
        coef,
        ratio_sq,
        opac_open,
    }
}

#[cube]
pub fn read_quat_unorm(transforms: &Tensor<f32>, base: usize) -> Quat {
    Quat::new(
        transforms[base + 3],
        transforms[base + 4],
        transforms[base + 5],
        transforms[base + 6],
    )
}
