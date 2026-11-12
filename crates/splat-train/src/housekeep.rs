//! Per-step trainer bookkeeping as two kernels: refine-stat accumulation and
//! the low-opacity mean noise. Replaces ~15 small tensor ops (and a full
//! `[N, 3]` normal sample) per step.

use brush_cube::MainBackendBase;
use burn::backend::Dispatch;
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{Backend, TensorMetadata, backend_extension, tensor::FloatTensor};
use burn::cubecl::{self, cube, prelude::*};
use burn::tensor::{DType, Tensor as BTensor};

const WORKGROUP: u32 = 256;

#[backend_extension(Cube, Fusion)]
pub(crate) trait HousekeepOps: Backend {
    /// `(refine_norm, vis_weight, max_screen)` updated with one step's
    /// `refine_weight` (max), `visible` (sum) and `radius` (max).
    #[fusion(meta = |norm, vis, screen, _w, _v, _r| (norm.clone(), vis.clone(), screen.clone()))]
    fn gather_stats(
        refine_norm: FloatTensor<Self>,
        vis_weight: FloatTensor<Self>,
        max_screen: FloatTensor<Self>,
        refine_weight: FloatTensor<Self>,
        visible: FloatTensor<Self>,
        radius: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    /// Adds `N(0,1) · clamp((1-opacity)^150, 0, 1) · visible · scale`,
    /// clamped to ±`limit`, to the mean columns (0..3) of `[N, 10]` transforms.
    #[fusion(meta = |transforms, _o, _v, _s, _l, _seed| TensorSpec::new(transforms.shape.clone(), DType::F32))]
    fn noise_means(
        transforms: FloatTensor<Self>,
        opacities: FloatTensor<Self>,
        visible: FloatTensor<Self>,
        #[fusion(scalar)] scale: f32,
        #[fusion(scalar)] limit: f32,
        #[fusion(scalar)] seed: u32,
    ) -> FloatTensor<Self>;
}

fn owned(t: FloatTensor<MainBackendBase>) -> FloatTensor<MainBackendBase> {
    if t.can_mut() { t } else { t.copy() }
}

impl HousekeepOps for MainBackendBase {
    fn gather_stats(
        refine_norm: FloatTensor<Self>,
        vis_weight: FloatTensor<Self>,
        max_screen: FloatTensor<Self>,
        refine_weight: FloatTensor<Self>,
        visible: FloatTensor<Self>,
        radius: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let n = refine_norm.shape().num_elements();
        for t in [&vis_weight, &max_screen, &refine_weight, &visible, &radius] {
            assert_eq!(t.shape().num_elements(), n);
            assert_eq!(t.dtype, DType::F32);
            assert!(t.is_contiguous());
        }
        let (norm, vis, screen) = (owned(refine_norm), owned(vis_weight), owned(max_screen));
        let dim = CubeDim::new_1d(WORKGROUP);
        let count = cubecl::calculate_cube_count_elemwise(&norm.client, n, dim);
        stats_kernel::launch(
            &norm.client,
            count,
            dim,
            norm.clone().into_tensor_arg(),
            vis.clone().into_tensor_arg(),
            screen.clone().into_tensor_arg(),
            refine_weight.into_tensor_arg(),
            visible.into_tensor_arg(),
            radius.into_tensor_arg(),
        );
        (norm, vis, screen)
    }

    fn noise_means(
        transforms: FloatTensor<Self>,
        opacities: FloatTensor<Self>,
        visible: FloatTensor<Self>,
        scale: f32,
        limit: f32,
        seed: u32,
    ) -> FloatTensor<Self> {
        let n = opacities.shape().num_elements();
        assert_eq!(transforms.shape().num_elements(), n * 10);
        assert_eq!(visible.shape().num_elements(), n);
        assert!(transforms.is_contiguous() && opacities.is_contiguous() && visible.is_contiguous());
        let out = owned(transforms);
        let dim = CubeDim::new_1d(WORKGROUP);
        let count = cubecl::calculate_cube_count_elemwise(&out.client, n, dim);
        noise_kernel::launch(
            &out.client,
            count,
            dim,
            out.clone().into_tensor_arg(),
            opacities.into_tensor_arg(),
            visible.into_tensor_arg(),
            scale,
            limit,
            seed,
        );
        out
    }
}

#[cube(launch)]
fn stats_kernel(
    norm: &mut Tensor<f32>,
    vis: &mut Tensor<f32>,
    screen: &mut Tensor<f32>,
    weight: &Tensor<f32>,
    visible: &Tensor<f32>,
    radius: &Tensor<f32>,
) {
    let i = ABSOLUTE_POS;
    if i < norm.len() {
        norm[i] = f32::max(norm[i], weight[i]);
        vis[i] += visible[i];
        screen[i] = f32::max(screen[i], radius[i]);
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

#[cube(launch)]
fn noise_kernel(
    transforms: &mut Tensor<f32>,
    opacities: &Tensor<f32>,
    visible: &Tensor<f32>,
    scale: f32,
    limit: f32,
    seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < opacities.len() {
        let gate = clamp(f32::powf(1.0f32 - opacities[i], 150.0f32), 0.0f32, 1.0f32) * visible[i];
        if gate > 0.0f32 {
            let w = gate * scale;
            let base = (i as u32) * 4u32;
            // Box-Muller: two uniforms give two normals; a second pair gives the third.
            let r1 = f32::sqrt(-2.0f32 * f32::ln(hash_uniform(base, seed)));
            let a1 = 6.283_185_5f32 * hash_uniform(base + 1u32, seed);
            let r2 = f32::sqrt(-2.0f32 * f32::ln(hash_uniform(base + 2u32, seed)));
            let a2 = 6.283_185_5f32 * hash_uniform(base + 3u32, seed);
            let row = i * 10;
            transforms[row] += clamp(r1 * f32::cos(a1) * w, -limit, limit);
            transforms[row + 1] += clamp(r1 * f32::sin(a1) * w, -limit, limit);
            transforms[row + 2] += clamp(r2 * f32::cos(a2) * w, -limit, limit);
        }
    }
}

pub(crate) fn gather_stats(
    norm: BTensor<1>,
    vis: BTensor<1>,
    screen: BTensor<1>,
    weight: BTensor<1>,
    visible: BTensor<1>,
    radius: BTensor<1>,
) -> (BTensor<1>, BTensor<1>, BTensor<1>) {
    let (a, b, c) = Dispatch::gather_stats(
        norm.into_dispatch(),
        vis.into_dispatch(),
        screen.into_dispatch(),
        weight.into_dispatch(),
        visible.into_dispatch(),
        radius.into_dispatch(),
    );
    (BTensor::from_dispatch(a), BTensor::from_dispatch(b), BTensor::from_dispatch(c))
}

pub(crate) fn noise_means(
    transforms: BTensor<2>,
    opacities: BTensor<1>,
    visible: BTensor<1>,
    scale: f32,
    limit: f32,
    seed: u32,
) -> BTensor<2> {
    let dims = transforms.dims();
    let out = Dispatch::noise_means(
        transforms.reshape([dims[0] * dims[1]]).into_dispatch(),
        opacities.into_dispatch(),
        visible.into_dispatch(),
        scale,
        limit,
        seed,
    );
    BTensor::<1>::from_dispatch(out).reshape(dims)
}
