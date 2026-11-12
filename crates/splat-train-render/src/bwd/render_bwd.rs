use crate::gaussian_splats::SplatRenderMode;
use crate::kernels::types::RasterizeUniformsLaunch;
use crate::kernels::neural::{NEURAL_DECODER_LEN, NEURAL_ROW_COEFFS};
use crate::sh::sh_coeffs_for_degree;
use brush_cube::create_tensor;
use burn::backend::TensorMetadata;
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::cubecl::CubeCount;
use burn::cubecl::CubeDim;
use burn::cubecl::calculate_cube_count_elemwise;
use burn::cubecl::features::{AtomicUsage, Plane};
use burn::cubecl::ir::{ElemType, FloatKind, Type};
use burn::tensor::DType;
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use glam::{Vec3, uvec2};

use crate::bwd::burn_glue::{NeuralSplatGrads, SplatBwdOps, SplatGrads};
use crate::bwd::kernels;
use crate::shaders::helpers::ProjectUniforms;

/// Rows per partial sum of the camera gradient (see `pose_bwd`).
const POSE_SUM_BLOCK: usize = 256;

impl SplatBwdOps for CubeBackend {
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
    ) -> FloatTensor<Self> {
        let _span = tracing::trace_span!("rasterize_bwd").entered();

        let v_output = into_contiguous(v_output);
        let device = out_img.device.clone();
        let num_visible = projected_splats.shape()[0];
        let client = projected_splats.client.clone();

        // Sparse [capacity, grad lanes] indexed by compact_gid. Only the
        // rows of visible splats are accumulated into and read, so only those
        // are zeroed: the device-resident count bounds the fill, as the
        // host's count lags a step behind on training renders.
        let lanes = kernels::rasterize_backwards_planes::grad_lanes(surfel);
        let v_combined = create_tensor([num_visible, lanes as usize], &device, DType::F32);
        let cube_dim = CubeDim::new_1d(256);
        kernels::rasterize_backwards::zero_visible_rows_kernel::launch(
            &client,
            calculate_cube_count_elemwise(&client, num_visible * lanes as usize, cube_dim),
            cube_dim,
            v_combined.clone().into_tensor_arg(),
            num_visible_gpu.into_tensor_arg(),
            lanes,
        );

        let tile_bounds = uvec2(
            img_size.x.div_ceil(crate::shaders::helpers::TILE_WIDTH),
            img_size.y.div_ceil(crate::shaders::helpers::TILE_WIDTH),
        );

        let hard_floats = client
            .properties()
            .atomic_type_usage(Type::atomic(Type::new(ElemType::Float(FloatKind::F32))))
            .contains(AtomicUsage::Add);

        let cube_count = CubeCount::Static(tile_bounds.x, tile_bounds.y, 1);
        let uniforms = RasterizeUniformsLaunch::new(
            tile_bounds.x,
            img_size.x,
            img_size.y,
            background.x,
            background.y,
            background.z,
        );
        // Subgroup reductions let pixels keep their replay state in
        // registers; without them, fall back to the per-splat kernel.
        let planes = client.features().plane.contains(Plane::Ops);
        // Exactly 32-wide subgroups (NVIDIA, Apple) get the butterfly
        // reduce-scatter; other widths reduce one component at a time.
        let hardware = &client.properties().hardware;
        let xor32 = hardware.plane_size_min == 32 && hardware.plane_size_max == 32;

        tracing::trace_span!("RasterizeBackwards").in_scope(|| {
            use kernels::rasterize_backwards::{
                CasAtomicAdd, HfAtomicAdd, SPLAT_BATCH, rasterize_backwards_kernel,
            };
            use kernels::rasterize_backwards_planes::{
                WG_SIZE, rasterize_backwards_planes_kernel,
            };
            macro_rules! launch {
                ($kernel:ident, $atomic:ty, $wg:expr $(, $extra:expr)*) => {
                    $kernel::launch::<$atomic>(
                        &client,
                        cube_count,
                        CubeDim::new_1d($wg),
                        compact_gid_from_isect.into_tensor_arg(),
                        tile_offsets.into_tensor_arg(),
                        projected_splats.into_tensor_arg(),
                        out_img.into_tensor_arg(),
                        v_output.into_tensor_arg(),
                        v_combined.clone().into_tensor_arg(),
                        uniforms,
                        smooth_cutoff,
                        $($extra),*
                    )
                };
            }
            if surfel {
                // Surfels have only the subgroup backward.
                assert!(planes, "surfel (2DGS) training needs a GPU with subgroup operations");
                let splat_aux = into_contiguous(splat_aux);
                use kernels::rasterize_backwards_planes::rasterize_backwards_planes_surfel_kernel;
                macro_rules! launch_surfel {
                    ($atomic:ty) => {
                        rasterize_backwards_planes_surfel_kernel::launch::<$atomic>(
                            &client,
                            cube_count,
                            CubeDim::new_1d(WG_SIZE),
                            compact_gid_from_isect.into_tensor_arg(),
                            tile_offsets.into_tensor_arg(),
                            projected_splats.into_tensor_arg(),
                            splat_aux.into_tensor_arg(),
                            out_img.into_tensor_arg(),
                            v_output.into_tensor_arg(),
                            v_combined.clone().into_tensor_arg(),
                            uniforms,
                            smooth_cutoff,
                            xor32,
                        )
                    };
                }
                if hard_floats {
                    launch_surfel!(HfAtomicAdd);
                } else {
                    launch_surfel!(CasAtomicAdd);
                }
                return;
            }
            match (planes, hard_floats) {
                (true, true) => launch!(rasterize_backwards_planes_kernel, HfAtomicAdd, WG_SIZE, xor32),
                (true, false) => launch!(rasterize_backwards_planes_kernel, CasAtomicAdd, WG_SIZE, xor32),
                (false, true) => launch!(rasterize_backwards_kernel, HfAtomicAdd, SPLAT_BATCH),
                (false, false) => launch!(rasterize_backwards_kernel, CasAtomicAdd, SPLAT_BATCH),
            }
        });

        v_combined
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> SplatGrads<Self> {
        project_bwd_cube(
            transforms,
            sh_coeffs,
            None,
            raw_opac,
            min_scale,
            has_min_scale,
            global_from_compact_gid,
            num_visible_gpu,
            project_uniforms,
            render_mode,
            v_combined,
        )
        .0
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> NeuralSplatGrads<Self> {
        let (g, neural) = project_bwd_cube(
            transforms,
            sh_coeffs,
            Some(decoder),
            raw_opac,
            min_scale,
            has_min_scale,
            global_from_compact_gid,
            num_visible_gpu,
            project_uniforms,
            render_mode,
            v_combined,
        );
        let (v_decoder, v_coeffs_sq) = neural.expect("neural backward returns the decoder gradient");
        NeuralSplatGrads {
            v_transforms: g.v_transforms,
            v_coeffs: g.v_coeffs,
            v_raw_opac: g.v_raw_opac,
            v_refine_weight: g.v_refine_weight,
            v_decoder,
            v_coeffs_sq,
        }
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> FloatTensor<Self> {
        let _span = tracing::trace_span!("pose_bwd").entered();
        use kernels::pose_backwards::{POSE_DIM, WG_SIZE, pose_backwards_kernel};

        let transforms = into_contiguous(transforms);
        let sh_coeffs = into_contiguous(sh_coeffs);
        let raw_opac = into_contiguous(raw_opac);
        let min_scale = into_contiguous(min_scale);
        let device = transforms.device.clone();
        let client = transforms.client.clone();

        // One row per possible visible splat, padded to whole reduction
        // blocks; the kernel zeros the rest.
        let rows = (project_uniforms.total_splats as usize).max(1).div_ceil(POSE_SUM_BLOCK) * POSE_SUM_BLOCK;
        let v_pose = create_tensor([rows, POSE_DIM as usize], &device, DType::F32);
        let cube_dim = CubeDim::new_1d(WG_SIZE);
        macro_rules! launch_sh {
            ($elem:ty) => {
                pose_backwards_kernel::launch::<$elem>(
                    &client,
                    calculate_cube_count_elemwise(&client, rows, cube_dim),
                    cube_dim,
                    transforms.into_tensor_arg(),
                    sh_coeffs.into_tensor_arg(),
                    raw_opac.into_tensor_arg(),
                    min_scale.into_tensor_arg(),
                    global_from_compact_gid.into_tensor_arg(),
                    num_visible_gpu.into_tensor_arg(),
                    v_combined.into_tensor_arg(),
                    v_pose.clone().into_tensor_arg(),
                    project_uniforms.to_launch_object(),
                    matches!(render_mode, SplatRenderMode::Mip),
                    has_min_scale,
                    project_uniforms.sh_degree,
                    project_uniforms.camera_model,
                )
            };
        }
        match sh_coeffs.dtype {
            DType::F16 => launch_sh!(burn::tensor::f16),
            DType::F32 => launch_sh!(f32),
            dtype => panic!("SH coefficients must use f16 or f32, got {dtype:?}"),
        }
        // Two-level sum: a single sum over the rows leaves 12 outputs, each
        // reducing millions of rows (one cube per output); summing blocks of
        // rows first keeps both reductions parallel.
        let blocks = rows / POSE_SUM_BLOCK;
        let v_pose = Self::float_reshape(v_pose, [blocks, POSE_SUM_BLOCK, POSE_DIM as usize].into());
        let partial = Self::float_sum_dim(v_pose, 1);
        let partial = Self::float_reshape(partial, [blocks, POSE_DIM as usize].into());
        let summed = Self::float_sum_dim(partial, 0);
        Self::float_reshape(summed, [POSE_DIM as usize].into())
    }
}

/// The projection backward; with `decoder` (neural appearance) it also runs
/// `neural_backwards` and returns the dense decoder gradient and the per-row
/// gradient mean square.
#[allow(clippy::too_many_arguments)]
fn project_bwd_cube(
    transforms: FloatTensor<CubeBackend>,
    sh_coeffs: FloatTensor<CubeBackend>,
    decoder: Option<FloatTensor<CubeBackend>>,
    raw_opac: FloatTensor<CubeBackend>,
    min_scale: FloatTensor<CubeBackend>,
    has_min_scale: bool,
    global_from_compact_gid: IntTensor<CubeBackend>,
    num_visible_gpu: IntTensor<CubeBackend>,
    project_uniforms: ProjectUniforms,
    render_mode: SplatRenderMode,
    v_combined: FloatTensor<CubeBackend>,
) -> (
    SplatGrads<CubeBackend>,
    Option<(FloatTensor<CubeBackend>, FloatTensor<CubeBackend>)>,
) {
    let _span = tracing::trace_span!("project_bwd").entered();

    // The screen-area regulariser only acts in this backward kernel, so we
    // stamp the weight onto the uniforms here rather than in the forward.
    let transforms = into_contiguous(transforms);
    let sh_coeffs = into_contiguous(sh_coeffs);
    let raw_opac = into_contiguous(raw_opac);
    let min_scale = into_contiguous(min_scale);

    let device = transforms.device.clone();
    let client = transforms.client.clone();

    let mip_splat = matches!(render_mode, SplatRenderMode::Mip);

    // Capacity-sized compact outputs with a zero row in front. The kernel
    // zeros every inactive row; later coefficient reductions see no garbage.
    let rows = project_uniforms.total_splats as usize + 1;
    let neural = decoder.is_some();
    let coeffs = if neural {
        NEURAL_ROW_COEFFS as usize
    } else {
        sh_coeffs_for_degree(project_uniforms.sh_degree) as usize
    };
    let v_transforms = create_tensor([rows, 10], &device, DType::F32);
    let v_coeffs = create_tensor([rows, coeffs, 3], &device, DType::F32);
    let v_raw_opac = create_tensor([rows], &device, DType::F32);
    let v_refine_weight = create_tensor([rows], &device, DType::F32);

    // The neural backward runs after the projection backward and reads
    // its inputs again; SH mode hands them straight to the kernel.
    let neural_inputs = decoder.map(|decoder| {
        (
            into_contiguous(decoder),
            transforms.clone(),
            sh_coeffs.clone(),
            global_from_compact_gid.clone(),
            num_visible_gpu.clone(),
            v_combined.clone(),
        )
    });

    let uniforms = project_uniforms.to_launch_object();
    let cube_dim = CubeDim::new_1d(kernels::project_backwards::WG_SIZE);

    tracing::trace_span!("ProjectBackwards").in_scope(|| {
        macro_rules! launch_sh {
            ($elem:ty) => {
                kernels::project_backwards::project_backwards_kernel::launch::<$elem>(
                    &client,
                    calculate_cube_count_elemwise(&client, rows, cube_dim),
                    cube_dim,
                    transforms.into_tensor_arg(),
                    sh_coeffs.into_tensor_arg(),
                    raw_opac.into_tensor_arg(),
                    min_scale.into_tensor_arg(),
                    global_from_compact_gid.into_tensor_arg(),
                    num_visible_gpu.into_tensor_arg(),
                    v_combined.into_tensor_arg(),
                    v_transforms.clone().into_tensor_arg(),
                    v_coeffs.clone().into_tensor_arg(),
                    v_raw_opac.clone().into_tensor_arg(),
                    v_refine_weight.clone().into_tensor_arg(),
                    uniforms,
                    mip_splat,
                    has_min_scale,
                    if neural { 0 } else { project_uniforms.sh_degree },
                    neural,
                    project_uniforms.camera_model,
                    matches!(render_mode, SplatRenderMode::Surfel),
                )
            };
        }
        match sh_coeffs.dtype {
            DType::F16 => launch_sh!(burn::tensor::f16),
            DType::F32 => launch_sh!(f32),
            dtype => panic!("SH coefficients must use f16 or f32, got {dtype:?}"),
        }
    });

    let neural_out = neural_inputs.map(
        |(decoder, transforms, sh_coeffs, global_from_compact_gid, num_visible_gpu, v_combined)| {
        let hardware = &client.properties().hardware;
        assert!(
            client.features().plane.contains(Plane::Ops)
                && hardware.plane_size_min == 32
                && hardware.plane_size_max == 32,
            "neural appearance training needs 32-wide subgroups"
        );
        use kernels::neural_backwards::{WG_SIZE, neural_backwards_kernel};
        // A bounded grid striding over the rows keeps the per-cube
        // decoder partials small; the host sums them.
        let cubes = rows.div_ceil(WG_SIZE as usize).clamp(1, 1024);
        let iters = rows.div_ceil(cubes * WG_SIZE as usize) as u32;
        let partials = create_tensor([cubes, NEURAL_DECODER_LEN as usize], &device, DType::F32);
        let v_coeffs_sq = create_tensor([rows], &device, DType::F32);
        tracing::trace_span!("NeuralBackwards").in_scope(|| {
            macro_rules! launch_neural {
                ($elem:ty) => {
                    neural_backwards_kernel::launch::<$elem>(
                        &client,
                        CubeCount::Static(cubes as u32, 1, 1),
                        CubeDim::new_1d(WG_SIZE),
                        transforms.into_tensor_arg(),
                        sh_coeffs.into_tensor_arg(),
                        decoder.into_tensor_arg(),
                        global_from_compact_gid.into_tensor_arg(),
                        num_visible_gpu.into_tensor_arg(),
                        v_combined.into_tensor_arg(),
                        v_transforms.clone().into_tensor_arg(),
                        v_coeffs.clone().into_tensor_arg(),
                        v_coeffs_sq.clone().into_tensor_arg(),
                        partials.clone().into_tensor_arg(),
                        project_uniforms.to_launch_object(),
                        iters,
                    )
                };
            }
            match sh_coeffs.dtype {
                DType::F16 => launch_neural!(burn::tensor::f16),
                DType::F32 => launch_neural!(f32),
                dtype => panic!("neural rows must use f16 or f32, got {dtype:?}"),
            }
        });
        let summed = CubeBackend::float_sum_dim(partials, 0);
        (CubeBackend::float_reshape(summed, [NEURAL_DECODER_LEN as usize].into()), v_coeffs_sq)
    });

    (
        SplatGrads {
            v_transforms,
            v_coeffs,
            v_raw_opac,
            v_refine_weight,
        },
        neural_out,
    )
}
