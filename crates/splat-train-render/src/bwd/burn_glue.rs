#![allow(clippy::match_wildcard_for_single_variants)]

use crate::{
    SplatOps,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    kernels::neural::{NEURAL_DECODER_LEN, NEURAL_ROW_COEFFS},
    sh::sh_coeffs_for_degree,
    shaders::helpers::ProjectUniforms,
};
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::{Autodiff, ExtensionType, fusion::custom::TensorSpec};
use burn::{
    backend::{
        AutodiffBackend, Backend,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{DType, Int, Shape, Tensor},
};
use glam::Vec3;

/// Final gradients w.r.t. splat inputs from the project backward pass.
#[derive(Debug, Clone, ExtensionType)]
#[extension_type(fusion)]
pub(crate) struct SplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
}

/// [`SplatGrads`] of the neural-appearance backward, plus the dense shared
/// decoder gradient `[NEURAL_DECODER_LEN]` and each compact row's gradient
/// mean square `[rows]` (Adam's reduced second moment input).
#[derive(Debug, Clone, ExtensionType)]
#[extension_type(fusion)]
pub(crate) struct NeuralSplatGrads<B: Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_coeffs: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    pub v_refine_weight: FloatTensor<B>,
    pub v_decoder: FloatTensor<B>,
    pub v_coeffs_sq: FloatTensor<B>,
}

/// Concrete backward kernels behind [`SplatOps::render`]. Deliberately not
/// `: SplatOps`: fewer backends have backward kernels than can render, and
/// these run on concrete tensors from the inner backend's `Backward` impl.
/// Wrapped so the allow reaches the generated Fusion impl, which binds every
/// ordinary argument whether its metadata expression reads it or not.
mod bwd_ops {
    #![allow(unused_variables)]
    use super::{
        NeuralSplatGrads, ProjectUniforms, SplatGrads, SplatRenderMode, project_bwd_metadata,
        project_bwd_neural_metadata,
    };
    use burn::backend::{
        Backend,
        tensor::{FloatTensor, IntTensor},
    };
    use burn::backend::fusion::custom::TensorSpec;
    use burn::tensor::{DType, Shape};
    use glam::Vec3;

    #[burn::backend::backend_extension(Fusion)]
    pub(crate) trait SplatBwdOps: Backend {
        /// Returns sparse `v_combined` `[num_visible, 10]` indexed by
        /// `compact_gid`: eight projected-splat gradients, then the raw opacity
        /// gradient and the refinement weight.
        #[allow(clippy::too_many_arguments)]
        #[fusion(dtype = v_output, shape = Shape::new([
            projected_splats[0],
            crate::bwd::kernels::rasterize_backwards_planes::grad_lanes(*surfel) as usize,
        ]))]
        fn rasterize_bwd(
            out_img: FloatTensor<Self>,
            projected_splats: FloatTensor<Self>,
            splat_aux: FloatTensor<Self>,
            surfel: bool,
            compact_gid_from_isect: IntTensor<Self>,
            tile_offsets: IntTensor<Self>,
            num_visible_gpu: IntTensor<Self>,
            background: Vec3,
            img_size: glam::UVec2,
            v_output: FloatTensor<Self>,
            smooth_cutoff: bool,
        ) -> FloatTensor<Self>;

        /// Writes a zero row, then one row per visible splat in `compact_gid`
        /// order; the caller gathers those per global splat through
        /// `compact_from_global`. `sh_coeffs` is the forward's input, so the
        /// kernel can backprop `v_color` through the SH basis to the mean.
        #[allow(clippy::too_many_arguments)]
        #[fusion(meta = project_bwd_metadata)]
        fn project_bwd(
            transforms: FloatTensor<Self>,
            sh_coeffs: FloatTensor<Self>,
            raw_opac: FloatTensor<Self>,
            min_scale: FloatTensor<Self>,
            has_min_scale: bool,
            global_from_compact_gid: IntTensor<Self>,
            num_visible_gpu: IntTensor<Self>,
            project_uniforms: ProjectUniforms,
            render_mode: SplatRenderMode,
            v_combined: FloatTensor<Self>,
        ) -> SplatGrads<Self>;

        /// [`Self::project_bwd`] for neural appearance: `sh_coeffs` holds
        /// the compact neural rows and `decoder` the shared decoder, whose
        /// gradient comes back summed over the visible splats.
        #[allow(clippy::too_many_arguments)]
        #[fusion(meta = project_bwd_neural_metadata)]
        fn project_bwd_neural(
            transforms: FloatTensor<Self>,
            sh_coeffs: FloatTensor<Self>,
            decoder: FloatTensor<Self>,
            raw_opac: FloatTensor<Self>,
            min_scale: FloatTensor<Self>,
            has_min_scale: bool,
            global_from_compact_gid: IntTensor<Self>,
            num_visible_gpu: IntTensor<Self>,
            project_uniforms: ProjectUniforms,
            render_mode: SplatRenderMode,
            v_combined: FloatTensor<Self>,
        ) -> NeuralSplatGrads<Self>;

        /// Camera gradient `[12]`: a camera-space affine map `(I + A) x + b`
        /// after world-to-camera, as `[A row-major, b]`, summed over splats.
        #[allow(clippy::too_many_arguments)]
        #[fusion(meta = |_t, _sh, _o, _ms, _hms, _g, _n, _u, _rm, _v| TensorSpec::new(Shape::new([12]), DType::F32))]
        fn pose_bwd(
            transforms: FloatTensor<Self>,
            sh_coeffs: FloatTensor<Self>,
            raw_opac: FloatTensor<Self>,
            min_scale: FloatTensor<Self>,
            has_min_scale: bool,
            global_from_compact_gid: IntTensor<Self>,
            num_visible_gpu: IntTensor<Self>,
            project_uniforms: ProjectUniforms,
            render_mode: SplatRenderMode,
            v_combined: FloatTensor<Self>,
        ) -> FloatTensor<Self>;
    }
}
pub(crate) use bwd_ops::SplatBwdOps;

/// Output shapes, which Fusion needs before the kernel runs. Input splat count
/// bounds the visible rows, so sizing never depends on a device count readback.
fn project_bwd_metadata(
    _transforms: &TensorSpec,
    _sh_coeffs: &TensorSpec,
    _raw_opac: &TensorSpec,
    _min_scale: &TensorSpec,
    _has_min_scale: &bool,
    _global_from_compact_gid: &TensorSpec,
    _num_visible_gpu: &TensorSpec,
    project_uniforms: &ProjectUniforms,
    _render_mode: &SplatRenderMode,
    _v_combined: &TensorSpec,
) -> SplatGradsMetadata {
    let rows = project_uniforms.total_splats as usize + 1;
    let coeffs = sh_coeffs_for_degree(project_uniforms.sh_degree) as usize;
    let f32_spec = |shape: Shape| TensorSpec::new(shape, DType::F32);
    SplatGradsMetadata {
        v_transforms: f32_spec(Shape::new([rows, 10])),
        v_coeffs: f32_spec(Shape::new([rows, coeffs, 3])),
        v_raw_opac: f32_spec(Shape::new([rows])),
        v_refine_weight: f32_spec(Shape::new([rows])),
    }
}

#[allow(clippy::too_many_arguments)]
fn project_bwd_neural_metadata(
    _transforms: &TensorSpec,
    _sh_coeffs: &TensorSpec,
    _decoder: &TensorSpec,
    _raw_opac: &TensorSpec,
    _min_scale: &TensorSpec,
    _has_min_scale: &bool,
    _global_from_compact_gid: &TensorSpec,
    _num_visible_gpu: &TensorSpec,
    project_uniforms: &ProjectUniforms,
    _render_mode: &SplatRenderMode,
    _v_combined: &TensorSpec,
) -> NeuralSplatGradsMetadata {
    let rows = project_uniforms.total_splats as usize + 1;
    let f32_spec = |shape: Shape| TensorSpec::new(shape, DType::F32);
    NeuralSplatGradsMetadata {
        v_transforms: f32_spec(Shape::new([rows, 10])),
        v_coeffs: f32_spec(Shape::new([rows, NEURAL_ROW_COEFFS as usize, 3])),
        v_raw_opac: f32_spec(Shape::new([rows])),
        v_refine_weight: f32_spec(Shape::new([rows])),
        v_decoder: f32_spec(Shape::new([NEURAL_DECODER_LEN as usize])),
        v_coeffs_sq: f32_spec(Shape::new([rows])),
    }
}

/// State saved during forward pass for backward computation.
#[derive(Debug, Clone)]
struct GaussianBackwardState<B: Backend> {
    transforms: FloatTensor<B>,
    sh_coeffs: FloatTensor<B>,
    /// The shared decoder in neural mode; `None` for SH.
    decoder: Option<FloatTensor<B>>,
    raw_opacity: FloatTensor<B>,
    min_scale: FloatTensor<B>,
    has_min_scale: bool,

    projected_splats: FloatTensor<B>,
    splat_aux: FloatTensor<B>,
    project_uniforms: ProjectUniforms,
    global_from_compact_gid: IntTensor<B>,
    num_visible_gpu: IntTensor<B>,
    compact_from_global: IntTensor<B>,

    out_img: FloatTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,

    render_mode: SplatRenderMode,
    pass: crate::gaussian_splats::RasterPass,
    background: Vec3,
    img_size: glam::UVec2,
}

#[derive(Debug)]
struct RenderBackwards;

/// Render parents in SH mode (the last catches the camera gradient); neural
/// appearance adds the decoder as a seventh.
const NUM_BWD_ARGS: usize = 6;

type RenderOpsKind<B, const N: usize> =
    OpsKind<RenderBackwards, B, GaussianBackwardState<B>, NoCheckpointing, N>;

/// The render op's autodiff prep: SH mode carries no decoder node.
enum RenderPrep<B: Backend> {
    Sh(RenderOpsKind<B, NUM_BWD_ARGS>),
    Neural(RenderOpsKind<B, { NUM_BWD_ARGS + 1 }>),
}

// Implement gradient registration when rendering backwards.
impl<B: Backend + SplatBwdOps, const N: usize> Backward<B, N> for RenderBackwards {
    type State = GaussianBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("render_gaussians backwards").entered();

        let state = ops.state;
        let v_output = grads.consume::<B>(&ops.node);

        // Register gradients for parent nodes (This code is already skipped entirely
        // if no parent nodes require gradients).
        let mut parents = ops.parents.into_iter();
        let mut next_parent = || parents.next().flatten();
        let transforms_parent = next_parent();
        let refine_weight = next_parent();
        let coeffs_parent = next_parent();
        let raw_opacity_parent = next_parent();
        let coeffs_grad_sq_parent = next_parent();
        let pose_parent = next_parent();
        let decoder_parent = next_parent();

        let v_combined = B::rasterize_bwd(
            state.out_img,
            state.projected_splats,
            state.splat_aux,
            state.render_mode == SplatRenderMode::Surfel,
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.num_visible_gpu.clone(),
            state.background,
            state.img_size,
            v_output,
            state.pass.smooth_cutoff(),
        );

        // The pose kernel follows the SH colour path (pose learning is SH-only).
        if let Some(node) = pose_parent
            && state.decoder.is_none()
        {
            let v_pose = B::pose_bwd(
                state.transforms.clone(),
                state.sh_coeffs.clone(),
                state.raw_opacity.clone(),
                state.min_scale.clone(),
                state.has_min_scale,
                state.global_from_compact_gid.clone(),
                state.num_visible_gpu.clone(),
                state.project_uniforms,
                state.render_mode,
                v_combined.clone(),
            );
            grads.register::<B>(node.id, v_pose);
        }

        // The trainer runs the projection VJP inside its Adam step
        // (`project_update`), reading the rasterize gradients from here.
        if state.pass.fused_update() {
            if let Some(node) = transforms_parent {
                grads.register::<B>(node.id, v_combined);
            }
            return;
        }

        let (splat_grads, v_decoder) = match state.decoder {
            None => (
                B::project_bwd(
                    state.transforms,
                    state.sh_coeffs,
                    state.raw_opacity,
                    state.min_scale,
                    state.has_min_scale,
                    state.global_from_compact_gid,
                    state.num_visible_gpu,
                    state.project_uniforms,
                    state.render_mode,
                    v_combined,
                ),
                None,
            ),
            Some(decoder) => {
                let g = B::project_bwd_neural(
                    state.transforms,
                    state.sh_coeffs,
                    decoder,
                    state.raw_opacity,
                    state.min_scale,
                    state.has_min_scale,
                    state.global_from_compact_gid,
                    state.num_visible_gpu,
                    state.project_uniforms,
                    state.render_mode,
                    v_combined,
                );
                let grads = SplatGrads {
                    v_transforms: g.v_transforms,
                    v_coeffs: g.v_coeffs,
                    v_raw_opac: g.v_raw_opac,
                    v_refine_weight: g.v_refine_weight,
                };
                (grads, Some((g.v_decoder, g.v_coeffs_sq)))
            }
        };
        let (v_decoder, v_coeffs_sq) = v_decoder.unzip();

        // The kernels write compact gradients with a zero row in front, and
        // `compact_from_global` points culled splats at that row, so one
        // gather expands each to the dense param shape. burn runs each as its
        // own fused kernel rather than folding it into the optimizer's, so the
        // dense gradient is still written, but only once and with no
        // zero-fill. `tests/fusion.rs` holds that shape in place.
        let inv = state.compact_from_global;
        let dense = |compact: FloatTensor<B>| B::float_select(compact, 0, inv.clone());
        let compact_coeffs = splat_grads.v_coeffs.clone();


        if let Some(node) = refine_weight {
            grads.register::<B>(node.id, dense(splat_grads.v_refine_weight));
        }

        // Parameter gradients stay compact: `[total + 1, ..]` rows in compact
        // order, row 0 zero. The trainer's fused Adam reads them through
        // [`SplatOutputDiff::compact_from_global`], so no dense f32 gradient
        // (240 B/splat at SH3) is ever written.
        // Renders whose gradients are summed (several sub-poses of one photo)
        // each have their own compact order, so those register dense rows.
        let layout = |compact: FloatTensor<B>| {
            if state.pass.dense_grads() {
                dense(compact)
            } else {
                compact
            }
        };
        if let Some(node) = transforms_parent {
            grads.register::<B>(node.id, layout(splat_grads.v_transforms));
        }

        if let Some(node) = coeffs_parent {
            grads.register::<B>(node.id, layout(splat_grads.v_coeffs));
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<B>(node.id, layout(splat_grads.v_raw_opac));
        }

        // Already dense: summed over every visible splat in the kernel.
        if let (Some(node), Some(v_decoder)) = (decoder_parent, v_decoder) {
            grads.register::<B>(node.id, v_decoder);
        }

        // The SH second moment Adam wants is a mean over each splat's
        // coefficients. Reducing the compact rows and gathering the small
        // result skips squaring the dense `[N, coeffs, 3]` gradient, which
        // burn will not fuse into the reduce that consumes it.
        if let Some(node) = coeffs_grad_sq_parent {
            let rows = burn::backend::TensorMetadata::shape(&compact_coeffs)[0];
            let mean = if let Some(sq) = v_coeffs_sq {
                // Neural rows (base + latents) have no SH identity: the
                // backward kernel writes the mean square over the whole row.
                B::float_reshape(sq, Shape::new([rows, 1, 1]))
            } else {
                // Every coefficient gradient is `Y_k(v) * v_color`, and real SH
                // satisfy sum_k Y_k(v)^2 = (d+1)^2 / 4pi, so the mean square over
                // all 3(d+1)^2 values equals the DC row's sum of squares / 3. Read
                // 3 floats per row instead of 3(d+1)^2.
                let dc = B::float_slice(
                    compact_coeffs,
                    &[
                        burn::backend::Slice::from(0..rows),
                        burn::backend::Slice::from(0..1),
                        burn::backend::Slice::from(0..3),
                    ],
                );
                let sq = B::float_mul(dc.clone(), dc);
                let sq = B::float_sum_dims(sq, &[1, 2]);
                B::float_div_scalar(sq, burn::tensor::Scalar::Float(3.0))
            };
            grads.register::<B>(node.id, B::float_select(mean, 0, inv));
        }
    }
}

pub struct SplatOutputDiff {
    /// Rendered RGBA image, on the autodiff graph (this is what the loss backprops through).
    pub img: Tensor<3>,
    /// Surfels: the geometry channels (`kernels::helpers::GEOM_OUT_CHANNELS`:
    /// depth sum, median depth, normal sum, median contributor), on the same
    /// autodiff graph. `None` for gaussians.
    pub geom: Option<Tensor<3>>,
    pub num_visible: u32,
    /// Per-splat visibility aux — on the **inner** backend (no gradients).
    pub visible: Tensor<1>,
    /// Per-splat max screen radius aux — on the **inner** backend (no gradients).
    pub max_radius: Tensor<1>,
    /// Per-splat opacity with the scale floor folded in — on the **inner**
    /// backend (no gradients). Zero for culled splats.
    pub opacities: Tensor<1>,
    pub refine_weight_holder: Tensor<1>,
    /// Catches the per-splat mean square of the SH gradient (see
    /// [`SplatOps::render`]). Its gradient is `[N, 1, 1]`, broadcasting over
    /// the coefficients, so it feeds Adam's second moment directly.
    /// This statistic is for one render/backward; it cannot be added across
    /// renders to obtain the mean square of an accumulated SH gradient.
    pub coeffs_grad_sq_holder: Tensor<3>,
    /// Row of each splat's compact gradient (0 = culled, the zero row). The
    /// transforms, SH and opacity gradients are registered in this layout.
    pub compact_from_global: Tensor<1, Int>,
    /// Set for `RasterPass::BackwardFusedUpdate` renders.
    pub fused_update: Option<super::project_update::ProjectUpdateCtx>,
    /// Catches the camera gradient `[12]`: `[A row-major, b]` of a
    /// camera-space affine map `(I + A) x + b` applied after world-to-camera.
    /// Tracked only when the render was asked to learn the pose.
    pub pose_grad_holder: Tensor<1>,
}

/// Render splats on a differentiable device.
///
/// Panics if the device is not autodiff-enabled.
pub async fn render_splats(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
) -> SplatOutputDiff {
    render_splats_with_pass(
        splats,
        camera,
        img_size,
        background,
        crate::gaussian_splats::RasterPass::Backward,
        false,
    )
    .await
}

/// Like [`render_splats`] but picks the
/// [`crate::gaussian_splats::RasterPass`] (dense gradients when several
/// renders are summed, or the finite-diff tests' smooth cutoff), and with
/// `learn_pose` tracks [`SplatOutputDiff::pose_grad_holder`].
pub async fn render_splats_with_pass(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
    learn_pose: bool,
) -> SplatOutputDiff {
    splats.clone().validate_values().await;

    let device = splats.device();
    assert!(
        device.is_autodiff(),
        "brush_render::bwd::render_splats requires an autodiff-enabled device"
    );

    // Gradient catchers only: no kernel reads their values, so they are
    // allocated without a fill.
    let refine_weight_holder = Tensor::<1>::empty([1], &device).require_grad();
    let coeffs_grad_sq_holder = Tensor::<3>::empty([1, 1, 1], &device).require_grad();
    let pose_grad_holder = Tensor::<1>::empty([12], &device);
    let pose_grad_holder = if learn_pose {
        pose_grad_holder.require_grad()
    } else {
        pose_grad_holder
    };

    // The 3D-filter floor is applied inside the projection kernels. It lives
    // on the inner backend and carries no gradient, so lifting it onto the
    // autodiff device is just a wrap.
    let (min_scale, has_min_scale) = splats.min_scale_arg();
    let (decoder, neural_degree) = splats.decoder_arg();

    let render_mode = splats.render_mode();

    assert!(
        pass.bwd_info(),
        "render_splats_with_pass requires a Backward variant"
    );

    let output = match decoder {
        None => <burn::backend::Dispatch as SplatOps>::render(
            camera,
            img_size,
            splats.transforms.val().into_dispatch(),
            splats.sh_coeffs.val().into_dispatch(),
            splats.raw_opacities.val().into_dispatch(),
            min_scale.autodiff().into_dispatch(),
            has_min_scale,
            0.0,
            refine_weight_holder.clone().into_dispatch(),
            coeffs_grad_sq_holder.clone().into_dispatch(),
            pose_grad_holder.clone().into_dispatch(),
            render_mode,
            background,
            pass,
        )
        .await,
        Some(decoder) => <burn::backend::Dispatch as SplatOps>::render_neural(
            camera,
            img_size,
            splats.transforms.val().into_dispatch(),
            splats.sh_coeffs.val().into_dispatch(),
            decoder.into_dispatch(),
            neural_degree,
            splats.raw_opacities.val().into_dispatch(),
            min_scale.autodiff().into_dispatch(),
            has_min_scale,
            0.0,
            refine_weight_holder.clone().into_dispatch(),
            coeffs_grad_sq_holder.clone().into_dispatch(),
            pose_grad_holder.clone().into_dispatch(),
            render_mode,
            background,
            pass,
        )
        .await,
    };

    let out: Tensor<3> = Tensor::from_dispatch(output.out_img);
    let [h, w, c] = out.dims();
    let (img, geom) = if c > 4 {
        (out.clone().slice([0..h, 0..w, 0..4]), Some(out.slice([0..h, 0..w, 4..c])))
    } else {
        (out, None)
    };
    SplatOutputDiff {
        img,
        geom,
        num_visible: output.aux.num_visible,
        visible: Tensor::from_dispatch(output.aux.visible).without_autodiff(),
        max_radius: Tensor::from_dispatch(output.aux.max_radius).without_autodiff(),
        opacities: Tensor::from_dispatch(output.aux.opacities).without_autodiff(),
        refine_weight_holder,
        coeffs_grad_sq_holder,
        compact_from_global: Tensor::<1, Int>::from_dispatch(output.compact_from_global)
            .without_autodiff(),
        fused_update: pass.fused_update().then(|| super::project_update::ProjectUpdateCtx {
            project_uniforms: output.project_uniforms,
            render_mode,
        }),
        pose_grad_holder,
    }
}

/// Differentiable inverse-depth render for depth-prior losses: every splat is
/// drawn with a flat colour `depth_unit / z` (`z` its view-space depth), so on
/// a zero background each pixel's red channel is the alpha-weighted inverse
/// depth and its alpha the coverage. The colour is computed from the means on
/// the autodiff graph, so the gradient reaches positions through both the
/// colour and the footprint (and opacities and shapes through the blend). No
/// kernel change: it is a plain SH degree-0 render with dense gradients
/// (`slot`, nonzero), summed with the step's photo render by autodiff.
pub async fn render_splats_inv_depth(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    depth_unit: f32,
    slot: u8,
) -> SplatOutputDiff {
    let device = splats.device();
    assert!(device.is_autodiff(), "render_splats_inv_depth requires an autodiff-enabled device");
    let w2c = camera.world_to_local();
    let row = glam::vec3(w2c.matrix3.x_axis.z, w2c.matrix3.y_axis.z, w2c.matrix3.z_axis.z);
    let t = w2c.translation.z;
    let transforms = splats.transforms.val();
    let n = transforms.dims()[0];
    let means = transforms.clone().slice([0..n, 0..3]).cast(DType::F32);
    let row = Tensor::<1>::from_floats([row.x, row.y, row.z], &device).reshape([1, 3]);
    let z = (means * row).sum_dim(1).add_scalar(t);
    // The projection clamps colours to +-100 but its backward does not mask
    // the clamp, so the colour must stay inside it: splats nearer than
    // `depth_unit / MAX_INV_DEPTH` (and those behind the camera, culled
    // anyway) take that depth, with no gradient through it. Unbounded, a
    // splat at the near plane drew ~1e4 and its 1/z^2 gradient overflowed
    // Adam's second moment (NaN splats from step ~900).
    const MAX_INV_DEPTH: f32 = 50.0;
    let inv = z.clamp_min(depth_unit / MAX_INV_DEPTH).recip().mul_scalar(depth_unit);
    let coeffs = inv
        .sub_scalar(0.5)
        .div_scalar(crate::shaders::SH_C0)
        .reshape([n, 1, 1])
        .repeat_dim(2, 3);

    let refine_weight_holder = Tensor::<1>::empty([1], &device).require_grad();
    let coeffs_grad_sq_holder = Tensor::<3>::empty([1, 1, 1], &device).require_grad();
    let pose_grad_holder = Tensor::<1>::empty([12], &device);
    let (min_scale, has_min_scale) = splats.min_scale_arg();
    let render_mode = splats.render_mode();
    let pass = crate::gaussian_splats::RasterPass::BackwardDense { slot };
    let output = <burn::backend::Dispatch as SplatOps>::render(
        camera,
        img_size,
        transforms.into_dispatch(),
        coeffs.into_dispatch(),
        splats.raw_opacities.val().into_dispatch(),
        min_scale.autodiff().into_dispatch(),
        has_min_scale,
        0.0,
        refine_weight_holder.clone().into_dispatch(),
        coeffs_grad_sq_holder.clone().into_dispatch(),
        pose_grad_holder.clone().into_dispatch(),
        render_mode,
        Vec3::ZERO,
        pass,
    )
    .await;
    let out: Tensor<3> = Tensor::from_dispatch(output.out_img);
    let [h, w, c] = out.dims();
    let img = if c > 4 { out.slice([0..h, 0..w, 0..4]) } else { out };
    SplatOutputDiff {
        img,
        geom: None,
        num_visible: output.aux.num_visible,
        visible: Tensor::from_dispatch(output.aux.visible).without_autodiff(),
        max_radius: Tensor::from_dispatch(output.aux.max_radius).without_autodiff(),
        opacities: Tensor::from_dispatch(output.aux.opacities).without_autodiff(),
        refine_weight_holder,
        coeffs_grad_sq_holder,
        compact_from_global: Tensor::<1, Int>::from_dispatch(output.compact_from_global)
            .without_autodiff(),
        fused_update: None,
        pose_grad_holder,
    }
}

impl<B: Backend + SplatOps + SplatBwdOps, C: CheckpointStrategy> SplatOps for Autodiff<B, C> {
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
    ) -> crate::RenderOutput<Self> {
        render_autodiff(
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
    ) -> crate::RenderOutput<Self> {
        render_autodiff(
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
/// in SH mode, which keeps main's five autodiff parents.
#[allow(clippy::too_many_arguments)]
async fn render_autodiff<B: Backend + SplatOps + SplatBwdOps, C: CheckpointStrategy>(
    camera: &Camera,
    img_size: glam::UVec2,
    transforms: FloatTensor<Autodiff<B, C>>,
    sh_coeffs: FloatTensor<Autodiff<B, C>>,
    decoder: Option<FloatTensor<Autodiff<B, C>>>,
    neural_degree: u32,
    raw_opacities: FloatTensor<Autodiff<B, C>>,
    min_scale: FloatTensor<Autodiff<B, C>>,
    has_min_scale: bool,
    log_scale_offset: f32,
    refine_weight: FloatTensor<Autodiff<B, C>>,
    coeffs_grad_sq: FloatTensor<Autodiff<B, C>>,
    pose_grad: FloatTensor<Autodiff<B, C>>,
    render_mode: SplatRenderMode,
    background: Vec3,
    pass: crate::gaussian_splats::RasterPass,
) -> crate::RenderOutput<Autodiff<B, C>> {
    let nodes = [
        transforms.node(),
        refine_weight.node(),
        sh_coeffs.node(),
        raw_opacities.node(),
        coeffs_grad_sq.node(),
        pose_grad.node(),
    ];
    let prep_nodes = match &decoder {
        None => RenderPrep::Sh(
            RenderBackwards
                .prepare::<NoCheckpointing>(nodes)
                .compute_bound()
                .stateful(),
        ),
        Some(decoder) => {
            let [t, r, s, o, c, p] = nodes;
            RenderPrep::Neural(
                RenderBackwards
                    .prepare::<NoCheckpointing>([t, r, s, o, c, p, decoder.node()])
                    .compute_bound()
                    .stateful(),
            )
        }
    };

    let transforms_inner: FloatTensor<B> = transforms.primitive().clone();
    let sh_inner: FloatTensor<B> = sh_coeffs.into_primitive();
    let raw_opac_inner: FloatTensor<B> = raw_opacities.primitive().clone();
    let min_scale_inner: FloatTensor<B> = min_scale.into_primitive();
    let decoder_inner: Option<FloatTensor<B>> = decoder.map(|d| d.into_primitive());

    let output = match decoder_inner.clone() {
        None => <B as SplatOps>::render(
            camera,
            img_size,
            transforms_inner.clone(),
            sh_inner.clone(),
            raw_opac_inner.clone(),
            min_scale_inner.clone(),
            has_min_scale,
            log_scale_offset,
            refine_weight.into_primitive(),
            coeffs_grad_sq.into_primitive(),
            pose_grad.primitive().clone(),
            render_mode,
            background,
            pass,
        )
        .await,
        Some(decoder) => <B as SplatOps>::render_neural(
            camera,
            img_size,
            transforms_inner.clone(),
            sh_inner.clone(),
            decoder,
            neural_degree,
            raw_opac_inner.clone(),
            min_scale_inner.clone(),
            has_min_scale,
            log_scale_offset,
            refine_weight.into_primitive(),
            coeffs_grad_sq.into_primitive(),
            pose_grad.primitive().clone(),
            render_mode,
            background,
            pass,
        )
        .await,
    };

    output.clone().validate().await;

    // SH and neural preps differ only in their parent count.
    macro_rules! finish {
        ($prep:expr) => {
            match $prep {
                OpsKind::Tracked(prep) => {
                    let state = GaussianBackwardState {
                        transforms: transforms_inner,
                        sh_coeffs: sh_inner,
                        decoder: decoder_inner,
                        raw_opacity: raw_opac_inner,
                        min_scale: min_scale_inner,
                        has_min_scale,
                        out_img: output.out_img.clone(),
                        projected_splats: output.projected_splats.clone(),
                        splat_aux: output.splat_aux.clone(),
                        project_uniforms: output.project_uniforms,
                        tile_offsets: output.aux.tile_offsets.clone(),
                        compact_gid_from_isect: output.compact_gid_from_isect.clone(),
                        render_mode,
                        pass,
                        global_from_compact_gid: output.global_from_compact_gid.clone(),
                        num_visible_gpu: output.num_visible_gpu.clone(),
                        compact_from_global: output.compact_from_global.clone(),
                        background,
                        img_size,
                    };
                    prep.finish(state, output.out_img)
                }
                OpsKind::UnTracked(prep) => prep.finish(output.out_img),
            }
        };
    }
    let img_ad: FloatTensor<Autodiff<B, C>> = match prep_nodes {
        RenderPrep::Sh(prep) => finish!(prep),
        RenderPrep::Neural(prep) => finish!(prep),
    };

    // Lift the remaining float aux onto the autodiff graph. None of these
    // carry a backward — they only feed refine bookkeeping — but the
    // extension trait's output is uniformly `RenderOutput<Autodiff<B, C>>`, so they
    // ride along as untracked autodiff tensors. Int tensors share the
    // inner backend's primitive and pass through unchanged.
    let lift = <Autodiff<B, C> as AutodiffBackend>::from_inner;

    crate::RenderOutput {
        out_img: img_ad,
        aux: crate::RenderAuxInner {
            num_visible: output.aux.num_visible,
            num_intersections: output.aux.num_intersections,
            visible: lift(output.aux.visible),
            max_radius: lift(output.aux.max_radius),
            opacities: lift(output.aux.opacities),
            tile_offsets: output.aux.tile_offsets,
            img_size: output.aux.img_size,
        },
        projected_splats: lift(output.projected_splats),
        splat_aux: lift(output.splat_aux),
        compact_gid_from_isect: output.compact_gid_from_isect,
        project_uniforms: output.project_uniforms,
        global_from_compact_gid: output.global_from_compact_gid,
        num_visible_gpu: output.num_visible_gpu,
        compact_from_global: output.compact_from_global,
    }
}
