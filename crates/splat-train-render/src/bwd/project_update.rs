//! Projection backward fused with the geometry/opacity Adam step. Training
//! renders with [`RasterPass::BackwardFusedUpdate`]: the render backward then
//! stops at the rasterize gradients (registered as the transforms gradient)
//! and the trainer calls [`project_update`] in place of the transforms and
//! opacity Adam steps.
//!
//! [`RasterPass::BackwardFusedUpdate`]: crate::gaussian_splats::RasterPass::BackwardFusedUpdate

use crate::gaussian_splats::SplatRenderMode;
use crate::shaders::helpers::ProjectUniforms;
use brush_cube::create_tensor;
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    Backend, Dispatch, ExtensionType, TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use burn::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn::tensor::{DType, Int, IntDType, Shape, Tensor};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;

use super::kernels::project_update::{WG_SIZE, project_update_kernel};

#[derive(Debug, Clone, ExtensionType)]
#[extension_type(fusion)]
pub struct ProjectUpdateOut<B: Backend> {
    pub transforms: FloatTensor<B>,
    /// Packed transforms moments (see `geo_moments`).
    pub w_t: IntTensor<B>,
    pub raw_opac: FloatTensor<B>,
    pub w_o: IntTensor<B>,
    /// Compact `[total + 1, 6]` colour gradient + view direction, row 0
    /// zero; the SH step rebuilds the coefficient gradient from it.
    pub color_dir: FloatTensor<B>,
    /// Refine weight per splat `[total]`, zero for culled splats.
    pub v_refine_weight: FloatTensor<B>,
    /// Mean square of each splat's SH gradient `[total]`.
    pub coeffs_grad_sq: FloatTensor<B>,
}

#[allow(clippy::too_many_arguments)]
fn project_update_meta(
    transforms: &TensorSpec,
    _sh_coeffs: &TensorSpec,
    raw_opac: &TensorSpec,
    _min_scale: &TensorSpec,
    _has_min_scale: &bool,
    _compact_from_global: &TensorSpec,
    _v_combined: &TensorSpec,
    w_t: &TensorSpec,
    _lr_scale: &TensorSpec,
    w_o: &TensorSpec,
    _beta1: &f32,
    _beta2: &f32,
    _inv1: &f32,
    _inv2: &f32,
    _epsilon: &f32,
    _lr_opac: &f32,
    _opac_reg: &f32,
    _round_seed: &u32,
    project_uniforms: &ProjectUniforms,
    _render_mode: &SplatRenderMode,
) -> ProjectUpdateOutMetadata {
    let rows = project_uniforms.total_splats as usize + 1;
    ProjectUpdateOutMetadata {
        transforms: transforms.clone(),
        w_t: w_t.clone(),
        raw_opac: raw_opac.clone(),
        w_o: w_o.clone(),
        color_dir: TensorSpec::new(Shape::new([rows, 6]), DType::F32),
        v_refine_weight: TensorSpec::new(Shape::new([rows - 1]), DType::F32),
        coeffs_grad_sq: TensorSpec::new(Shape::new([rows - 1]), DType::F32),
    }
}

mod ops {
    #![allow(unused_variables)]
    use super::{ProjectUniforms, ProjectUpdateOut, SplatRenderMode, project_update_meta};
    use burn::backend::{
        Backend,
        tensor::{FloatTensor, IntTensor},
    };

    #[burn::backend::backend_extension(Cube, Fusion)]
    pub trait ProjectUpdateOps: Backend {
        #[allow(clippy::too_many_arguments)]
        #[fusion(meta = project_update_meta)]
        fn project_update(
            transforms: FloatTensor<Self>,
            sh_coeffs: FloatTensor<Self>,
            raw_opac: FloatTensor<Self>,
            min_scale: FloatTensor<Self>,
            has_min_scale: bool,
            compact_from_global: IntTensor<Self>,
            v_combined: FloatTensor<Self>,
            w_t: IntTensor<Self>,
            lr_scale: FloatTensor<Self>,
            w_o: IntTensor<Self>,
            #[fusion(scalar)] beta1: f32,
            #[fusion(scalar)] beta2: f32,
            #[fusion(scalar)] inv1: f32,
            #[fusion(scalar)] inv2: f32,
            #[fusion(scalar)] epsilon: f32,
            #[fusion(scalar)] lr_opac: f32,
            #[fusion(scalar)] opac_reg: f32,
            #[fusion(scalar)] round_seed: u32,
            project_uniforms: ProjectUniforms,
            render_mode: SplatRenderMode,
        ) -> ProjectUpdateOut<Self>;
    }
}
use ops::ProjectUpdateOps;

impl ProjectUpdateOps for CubeBackend {
    fn project_update(
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opac: FloatTensor<Self>,
        min_scale: FloatTensor<Self>,
        has_min_scale: bool,
        compact_from_global: IntTensor<Self>,
        v_combined: FloatTensor<Self>,
        w_t: IntTensor<Self>,
        lr_scale: FloatTensor<Self>,
        w_o: IntTensor<Self>,
        beta1: f32,
        beta2: f32,
        inv1: f32,
        inv2: f32,
        epsilon: f32,
        lr_opac: f32,
        opac_reg: f32,
        round_seed: u32,
        project_uniforms: ProjectUniforms,
        render_mode: SplatRenderMode,
    ) -> ProjectUpdateOut<Self> {
        let _span = tracing::trace_span!("ProjectUpdate").entered();
        // Updated in place where this op holds the only reference.
        let own = |t: FloatTensor<Self>| {
            let t = into_contiguous(t);
            if t.can_mut() { t } else { t.copy() }
        };
        let (transforms, raw_opac) = (own(transforms), own(raw_opac));
        let (w_t, w_o) = (own(w_t), own(w_o));
        for t in [&transforms, &raw_opac] {
            assert_eq!(t.dtype, DType::F32, "fused update computes in f32");
        }
        for t in [&w_t, &w_o] {
            assert!(
                matches!(t.dtype, DType::U32 | DType::I32),
                "packed moments must be 32-bit words"
            );
        }
        let sh_coeffs = into_contiguous(sh_coeffs);
        let min_scale = into_contiguous(min_scale);
        let lr_scale = into_contiguous(lr_scale);
        let v_combined = into_contiguous(v_combined);
        let compact_from_global = into_contiguous(compact_from_global);
        let total = project_uniforms.total_splats as usize;
        assert_eq!(transforms.shape().num_elements(), total * 10);
        assert_eq!(w_t.shape().num_elements(), total * 10);
        assert_eq!(raw_opac.shape().num_elements(), total);
        assert_eq!(w_o.shape().num_elements(), total);
        assert_eq!(lr_scale.shape().num_elements(), 10);

        let device = transforms.device.clone();
        let client = transforms.client.clone();
        let rows = total + 1;
        let color_dir = create_tensor([rows, 6], &device, DType::F32);
        let v_refine_weight = create_tensor([total], &device, DType::F32);
        let coeffs_grad_sq = create_tensor([total], &device, DType::F32);

        let mip = matches!(render_mode, SplatRenderMode::Mip);
        let cube_dim = CubeDim::new_1d(WG_SIZE);
        macro_rules! launch {
            ($elem:ty) => {
                project_update_kernel::launch::<$elem>(
                    &client,
                    calculate_cube_count_elemwise(&client, total.max(1), cube_dim),
                    cube_dim,
                    transforms.clone().into_tensor_arg(),
                    sh_coeffs.clone().into_tensor_arg(),
                    raw_opac.clone().into_tensor_arg(),
                    min_scale.into_tensor_arg(),
                    compact_from_global.into_tensor_arg(),
                    v_combined.into_tensor_arg(),
                    w_t.clone().into_tensor_arg(),
                    lr_scale.into_tensor_arg(),
                    w_o.clone().into_tensor_arg(),
                    color_dir.clone().into_tensor_arg(),
                    v_refine_weight.clone().into_tensor_arg(),
                    coeffs_grad_sq.clone().into_tensor_arg(),
                    beta1,
                    beta2,
                    inv1,
                    inv2,
                    epsilon,
                    lr_opac,
                    opac_reg,
                    round_seed,
                    project_uniforms.to_launch_object(),
                    mip,
                    has_min_scale,
                    project_uniforms.sh_degree,
                    project_uniforms.camera_model,
                )
            };
        }
        match sh_coeffs.dtype {
            DType::F16 => launch!(burn::tensor::f16),
            DType::F32 => launch!(f32),
            dtype => panic!("SH coefficients must use f16 or f32, got {dtype:?}"),
        }

        ProjectUpdateOut {
            transforms,
            w_t,
            raw_opac,
            w_o,
            color_dir,
            v_refine_weight,
            coeffs_grad_sq,
        }
    }
}

/// What the trainer needs from the fused-update render to step.
#[derive(Clone, Copy, Debug)]
pub struct ProjectUpdateCtx {
    pub project_uniforms: ProjectUniforms,
    pub render_mode: SplatRenderMode,
}

/// Adam step parameters shared by transforms and opacity (same step count).
#[derive(Clone, Copy, Debug)]
pub struct FusedAdam {
    pub beta1: f32,
    pub beta2: f32,
    pub inv1: f32,
    pub inv2: f32,
    pub epsilon: f32,
    pub lr_opac: f32,
    /// Gradient weight of the opacity regulariser `opac_reg · Σ opacity`
    /// (3DGS-MCMC's `w · mean(opacity)`, so `w / N`); 0 = off.
    pub opac_reg: f32,
    /// Stochastic-rounding stream for the packed moments (the step count).
    pub round_seed: u32,
}

pub struct ProjectUpdateResult {
    pub transforms: Tensor<2>,
    /// Packed transforms moments `[N, 10]` (see `geo_moments`).
    pub w_t: Tensor<2, Int>,
    pub raw_opac: Tensor<1>,
    pub w_o: Tensor<1, Int>,
    pub color_dir: Tensor<2>,
    pub v_refine_weight: Tensor<1>,
    /// `[N, 1, 1]`, the SH step's reduced second-moment input.
    pub coeffs_grad_sq: Tensor<3>,
}

/// Projection VJP + Adam for transforms and opacity, on inner tensors.
/// `v_combined` is the transforms gradient a `BackwardFusedUpdate` render
/// registered; `lr_scale` is the per-column transforms LR `[10]` (base LR 1).
/// `w_t`/`w_o` are the packed Adam moments ([`super::geo_moments`]), stepped
/// in place; `adam.epsilon` must be the codec's epsilon.
#[allow(clippy::too_many_arguments)]
pub fn project_update(
    ctx: &ProjectUpdateCtx,
    transforms: Tensor<2>,
    sh_coeffs: Tensor<3>,
    raw_opac: Tensor<1>,
    (min_scale, has_min_scale): (Tensor<1>, bool),
    compact_from_global: Tensor<1, Int>,
    v_combined: Tensor<2>,
    w_t: Tensor<2, Int>,
    lr_scale: Tensor<1>,
    w_o: Tensor<1, Int>,
    adam: FusedAdam,
) -> ProjectUpdateResult {
    let t_shape = transforms.dims();
    let o_shape = raw_opac.dims();
    // The renderer already makes this u32; a cast would still copy it.
    let rows = if compact_from_global.dtype() == DType::U32 {
        compact_from_global
    } else {
        compact_from_global.cast(IntDType::U32)
    };
    let out = <Dispatch as ProjectUpdateOps>::project_update(
        transforms.into_dispatch(),
        sh_coeffs.into_dispatch(),
        raw_opac.into_dispatch(),
        min_scale.into_dispatch(),
        has_min_scale,
        rows.into_dispatch(),
        v_combined.into_dispatch(),
        w_t.into_dispatch(),
        lr_scale.into_dispatch(),
        w_o.into_dispatch(),
        adam.beta1,
        adam.beta2,
        adam.inv1,
        adam.inv2,
        adam.epsilon,
        adam.lr_opac,
        adam.opac_reg,
        adam.round_seed,
        ctx.project_uniforms,
        ctx.render_mode,
    );
    let t2 = |t| Tensor::<2>::from_dispatch(t).reshape(t_shape);
    let t1 = |t| Tensor::<1>::from_dispatch(t).reshape(o_shape);
    ProjectUpdateResult {
        transforms: t2(out.transforms),
        w_t: Tensor::<2, Int>::from_dispatch(out.w_t).reshape(t_shape),
        raw_opac: t1(out.raw_opac),
        w_o: Tensor::<1, Int>::from_dispatch(out.w_o).reshape(o_shape),
        color_dir: Tensor::from_dispatch(out.color_dir),
        v_refine_weight: t1(out.v_refine_weight),
        coeffs_grad_sq: t1(out.coeffs_grad_sq).reshape([o_shape[0], 1, 1]),
    }
}
