#![allow(clippy::match_wildcard_for_single_variants)]

use crate::fusion::bind;
use brush_cube::{MainBackend, MainBackendBase};
use burn::backend::{
    DispatchTensor, DispatchTensorKind,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{Int, Tensor};
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::Fusion;
use glam::Vec3;

use crate::{
    RenderAuxInner, SplatOps, backend_kind, camera::Camera, gaussian_splats::SplatRenderMode,
    render_aux::RenderOutput,
};
use burn_cubecl::CubeBackend;

// `Tensor<D>` is pinned to burn's `Dispatch` backend and brush only runs on
// wgpu, so these bridges assume a `DispatchTensorKind::Cube` (optionally
// wrapped in `Autodiff`) and panic otherwise. The forward render routes
// through the generated `Dispatch` impl; these serve the hand-rolled backward.

/// Extract the inner fusion-Wgpu float tensor from a non-autodiff
/// `Tensor<D>`.
pub fn unwrap_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        backend_kind!(bt) => bt.float(),
        other => panic!(
            "expected Wgpu tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// Extract the inner fusion-Wgpu int tensor from a non-autodiff
/// `Tensor<D, Int>`.
pub fn unwrap_wgpu_int<const D: usize>(t: Tensor<D, Int>) -> IntTensor<MainBackend> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        backend_kind!(bt) => bt.int(),
        other => panic!(
            "expected Wgpu int tensor, got: {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

fn is_autodiff<const D: usize>(t: &Tensor<D>) -> bool {
    matches!(
        t.clone().into_dispatch().kind,
        DispatchTensorKind::Autodiff(_)
    )
}

/// Put `t` on the same autodiff/inner backend variant as `reference`. Frozen
/// tensors like the 3D-filter floor live on the inner backend but get folded
/// against params that may be lifted, and mixing the two trips an assertion.
pub(crate) fn match_backend<const D: usize, const DR: usize>(
    t: Tensor<D>,
    reference: &Tensor<DR>,
) -> Tensor<D> {
    if is_autodiff(reference) {
        t.autodiff()
    } else {
        t.without_autodiff()
    }
}

/// Resolve pending fusion operations and return the underlying tensor.
pub fn resolve_to_cube_float<const D: usize>(tensor: Tensor<D>) -> CubeTensor {
    let fusion = unwrap_wgpu_float(tensor);
    let client = fusion.client.clone();
    client.resolve_tensor_float::<MainBackendBase>(fusion)
}

impl SplatOps for Fusion<CubeBackend> {
    #[allow(clippy::too_many_arguments)]
    async fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        log_scale_offset: f32,
        refine_weight: FloatTensor<Self>,
        coeffs_grad_sq: FloatTensor<Self>,
        pose_grad: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
    ) -> RenderOutput<Self> {
        render_fusion(
            camera,
            img_size,
            transforms,
            sh_coeffs,
            None,
            0,
            raw_opacities,
            min_scale,
            has_min_scale,
            log_scale_offset,
            refine_weight,
            coeffs_grad_sq,
            pose_grad,
            render_mode,
            background,
            pass,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn render_neural(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        decoder: FloatTensor<Self>,
        neural_degree: u32,
        raw_opacities: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        log_scale_offset: f32,
        refine_weight: FloatTensor<Self>,
        coeffs_grad_sq: FloatTensor<Self>,
        pose_grad: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: crate::gaussian_splats::RasterPass,
    ) -> RenderOutput<Self> {
        render_fusion(
            camera,
            img_size,
            transforms,
            sh_coeffs,
            Some(decoder),
            neural_degree,
            raw_opacities,
            min_scale,
            has_min_scale,
            log_scale_offset,
            refine_weight,
            coeffs_grad_sq,
            pose_grad,
            render_mode,
            background,
            pass,
        )
        .await
    }
}

/// [`SplatOps::render`] / [`SplatOps::render_neural`]: `decoder` is `None`
/// in SH mode.
#[allow(clippy::too_many_arguments)]
async fn render_fusion(
    camera: &Camera,
    img_size: glam::UVec2,
    transforms: FloatTensor<Fusion<CubeBackend>>,
    sh_coeffs: FloatTensor<Fusion<CubeBackend>>,
    decoder: Option<FloatTensor<Fusion<CubeBackend>>>,
    neural_degree: u32,
    raw_opacities: FloatTensor<Fusion<CubeBackend>>,
    min_scale: FloatTensor<Fusion<CubeBackend>>,
    has_min_scale: bool,
    log_scale_offset: f32,
    refine_weight: FloatTensor<Fusion<CubeBackend>>,
    coeffs_grad_sq: FloatTensor<Fusion<CubeBackend>>,
    pose_grad: FloatTensor<Fusion<CubeBackend>>,
    render_mode: SplatRenderMode,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
) -> RenderOutput<Fusion<CubeBackend>> {
    let client = transforms.client.clone();

    // Resolve fusion inputs to MainBackendBase tensors. This
    // drains any pending fusion operations into a concrete buffer.
    let base_transforms = client
        .clone()
        .resolve_tensor_float::<CubeBackend>(transforms);
    let base_sh_coeffs = client
        .clone()
        .resolve_tensor_float::<CubeBackend>(sh_coeffs);
    let base_decoder =
        decoder.map(|d| client.clone().resolve_tensor_float::<CubeBackend>(d));
    let base_raw_opac = client
        .clone()
        .resolve_tensor_float::<CubeBackend>(raw_opacities);
    let base_min_scale = client
        .clone()
        .resolve_tensor_float::<CubeBackend>(min_scale);
    let base_refine_weight = client
        .clone()
        .resolve_tensor_float::<CubeBackend>(refine_weight);
    let base_coeffs_grad_sq = client
        .clone()
        .resolve_tensor_float::<CubeBackend>(coeffs_grad_sq);
    let base_pose_grad = client
        .clone()
        .resolve_tensor_float::<CubeBackend>(pose_grad);

    let out = match base_decoder {
        None => <CubeBackend as SplatOps>::render(
            camera,
            img_size,
            base_transforms,
            base_sh_coeffs,
            base_raw_opac,
            base_min_scale,
            has_min_scale,
            log_scale_offset,
            base_refine_weight,
            base_coeffs_grad_sq,
            base_pose_grad,
            render_mode,
            background,
            pass,
        )
        .await,
        Some(decoder) => <CubeBackend as SplatOps>::render_neural(
            camera,
            img_size,
            base_transforms,
            base_sh_coeffs,
            decoder,
            neural_degree,
            base_raw_opac,
            base_min_scale,
            has_min_scale,
            log_scale_offset,
            base_refine_weight,
            base_coeffs_grad_sq,
            base_pose_grad,
            render_mode,
            background,
            pass,
        )
        .await,
    };

    let RenderOutput {
        out_img,
        aux,
        projected_splats,
        compact_gid_from_isect,
        splat_aux,
        project_uniforms,
        global_from_compact_gid,
        num_visible_gpu,
        compact_from_global,
    } = out;
    let RenderAuxInner {
        num_visible,
        num_intersections,
        visible,
        max_radius,
        opacities,
        tile_offsets,
        img_size,
    } = aux;

    let bind = |t| bind(&client, t);

    RenderOutput {
        out_img: bind(out_img),
        aux: RenderAuxInner {
            num_visible,
            num_intersections,
            visible: bind(visible),
            max_radius: bind(max_radius),
            opacities: bind(opacities),
            tile_offsets: bind(tile_offsets),
            img_size,
        },
        projected_splats: bind(projected_splats),
        compact_gid_from_isect: bind(compact_gid_from_isect),
        splat_aux: bind(splat_aux),
        project_uniforms,
        global_from_compact_gid: bind(global_from_compact_gid),
        num_visible_gpu: bind(num_visible_gpu),
        compact_from_global: bind(compact_from_global),
    }
}
