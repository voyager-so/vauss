use crate::camera::{calculate_jacobian_clamp_limits, max_render_theta};
use crate::{
    RenderAuxInner, SplatOps,
    camera::Camera,
    dim_check::DimCheck,
    gaussian_splats::{RasterPass, SplatRenderMode},
    get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets},
    kernels,
    render_aux::RenderOutput,
    sh::sh_degree_from_coeffs,
    shaders,
};
use brush_cube::create_tensor;
use brush_scan::scan_in_place;
use brush_sort::radix_argsort_with_count;
use burn::backend::TensorMetadata;
use burn::backend::ops::TransactionPrimitive;
use burn::backend::ops::{FloatTensorOps, TransactionOps};
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::cubecl::CubeDim;
use burn::cubecl::calculate_cube_count_elemwise;
use burn::tensor::DType;
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use glam::{Vec3, uvec2};
use kernels::types::RasterizeUniformsLaunch;
use std::f32::consts::PI;
use std::collections::HashMap;
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::task::Poll;

// Capacity hints own no GPU buffers and are shared across async task threads.
// The allocator can reuse equal-sized buffers; every in-flight render still
// owns its buffers, so concurrent renders cannot overwrite one another.
type CapacityKey = (u16, u16, u32, u32, u8);

#[derive(Clone, Copy)]
struct Capacity {
    isects: usize,
    // Splat count the hint was measured at, to scale it after refine growth.
    splats: u32,
    // The current window: renders seen, their largest and summed requirement,
    // renders that needed more than the capacity (lagged training renders
    // clamp those), and a histogram of intersections per splat.
    window_len: u32,
    window_max: usize,
    window_sum: u64,
    window_over: u32,
    ratios: RatioHistogram,
    // Whether a full window has been seen (until then the hint only grows).
    settled: bool,
}

/// Renders per capacity window. A window ends by sizing the hint to the
/// window's p99 intersections per splat (with `QUANTILE_HEADROOM`), but never
/// below its largest render: an overflowing (clamped) training render grows
/// the hint for the next one, and the window keeps covering it.
const CAPACITY_WINDOW: u32 = 200;
/// Intersections-per-splat quantile the hint covers.
const CAPACITY_QUANTILE: f64 = 0.99;
/// Headroom over the quantile, for the window-to-window drift.
const QUANTILE_HEADROOM: f64 = 1.25;

/// Log-spaced histogram of intersections per splat: 8 buckets per octave from
/// 2^-10 to 2^10, so a quantile is read to within 9% (rounded up).
#[derive(Clone, Copy)]
struct RatioHistogram([u32; RATIO_BUCKETS]);
const RATIO_BUCKETS: usize = 160;

impl RatioHistogram {
    fn bucket(ratio: f64) -> usize {
        ((ratio.max(1e-9).log2() * 8.0).floor() as i64 + 80).clamp(0, RATIO_BUCKETS as i64 - 1) as usize
    }
    fn push(&mut self, ratio: f64) {
        self.0[Self::bucket(ratio)] += 1;
    }
    /// Upper edge of the bucket holding quantile `q`.
    fn quantile(&self, q: f64) -> f64 {
        let total: u64 = self.0.iter().map(|&c| c as u64).sum();
        let rank = ((total as f64) * q).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (i, &c) in self.0.iter().enumerate() {
            seen += c as u64;
            if seen >= rank {
                return 2f64.powf((i as f64 - 80.0 + 1.0) / 8.0);
            }
        }
        2f64.powf((RATIO_BUCKETS as f64 - 80.0) / 8.0)
    }
}

/// Set when a capacity hint changes size: the next render releases the pool's
/// pages of the old size before allocating at the new one.
static CAPACITY_CHANGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Capacity {
    fn for_splats(self, total_splats: u32, limit: usize) -> usize {
        let grown = (self.isects as u128 * total_splats.max(self.splats) as u128)
            / self.splats.max(1) as u128;
        (grown.min(limit as u128) as usize).max(1)
    }
}

fn intersection_capacities() -> &'static Mutex<HashMap<CapacityKey, Capacity>> {
    static CAPACITIES: OnceLock<Mutex<HashMap<CapacityKey, Capacity>>> = OnceLock::new();
    CAPACITIES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn isect_log(line: String) {
    log::info!("{line}");
}

fn record_capacity(key: CapacityKey, required: usize, splats: u32, limit: usize) {
    let mut capacities = intersection_capacities().lock().unwrap();
    if capacities.len() >= 32 && !capacities.contains_key(&key) {
        capacities.clear();
    }
    let desired = capacity_with_headroom(required, limit);
    let entry = capacities.entry(key).or_insert(Capacity {
        isects: desired,
        splats,
        window_len: 0,
        window_max: 0,
        window_sum: 0,
        window_over: 0,
        ratios: RatioHistogram([0; RATIO_BUCKETS]),
        settled: false,
    });
    let current = entry.for_splats(splats, limit);
    entry.window_len += 1;
    entry.window_max = entry.window_max.max(required);
    entry.window_sum += required as u64;
    entry.ratios.push(required as f64 / splats.max(1) as f64);
    if required > current {
        entry.window_over += 1;
    }
    // Before the first full window, grow at once (headroom below 1.3x -> back
    // to 1.5x). After it, grow only on an overflow, to just cover it: a lagged
    // training render that overflowed has already clamped, so the next one
    // must not. Each growth re-allocates every intersection-sized buffer and
    // strands the old pages in the pool, so the hint must not creep up on
    // every new max.
    let grow_to = if !entry.settled {
        (required.saturating_add(required.div_ceil(10) * 3) > current).then_some(desired)
    } else {
        (required > current).then(|| covering_max(required, limit))
    };
    if let Some(grown) = grow_to {
        let grown = grown.max(current);
        isect_log(format!(
            "Intersection capacity grow: required={required}, splats={splats}, capacity {current} -> {grown}"
        ));
        *entry = Capacity { isects: grown, splats, ..*entry };
        CAPACITY_CHANGED.store(true, Ordering::Relaxed);
    }
    if entry.window_len >= CAPACITY_WINDOW {
        // Size the hint to the window's p99 intersections per splat, and never
        // below its largest render (a view seen once a window would otherwise
        // overflow every window). A change under 25% either way keeps the
        // current buffers (no re-allocation).
        let ratio = entry.ratios.quantile(CAPACITY_QUANTILE) * QUANTILE_HEADROOM;
        let target = ((splats as f64 * ratio).ceil() as usize)
            .max(covering_max(entry.window_max, limit))
            .clamp(1, limit);
        let current = entry.for_splats(splats, limit);
        let resize = target > current || target < current - current / 4;
        isect_log(format!(
            "Intersection capacity window: renders={}, mean={}, max={}, p99/splat={:.2}, over_capacity={}, splats={splats}, capacity={current}{}",
            entry.window_len,
            entry.window_sum / entry.window_len.max(1) as u64,
            entry.window_max,
            entry.ratios.quantile(CAPACITY_QUANTILE),
            entry.window_over,
            if resize { format!(" -> {target}") } else { String::new() }
        ));
        if resize {
            *entry = Capacity { isects: target, splats, ..*entry };
            CAPACITY_CHANGED.store(true, Ordering::Relaxed);
        }
        *entry = Capacity {
            window_len: 0,
            window_max: 0,
            window_sum: 0,
            window_over: 0,
            ratios: RatioHistogram([0; RATIO_BUCKETS]),
            settled: true,
            ..*entry
        };
    }
}

/// Bytes one intersection holds at the arena's peak: tile and splat ids, the
/// radix sort's ping-pong pair, and the hint's headroom.
const BYTES_PER_INTERSECTION: u64 = 24;

/// Most intersections one render may allocate. Each intersection buffer must
/// fit one storage binding, and the arena must fit what the device budget has
/// left: past that a render clamps (drops its farthest splats' tiles) instead
/// of running out of memory. The budget is only consulted for arenas big
/// enough to matter, so ordinary renders pay nothing.
fn intersection_ceiling(client: &burn::cubecl::client::Client, wanted: usize, limit: usize) -> usize {
    let page = (client.properties().memory.max_page_size / 4) as usize;
    let mut cap = limit.min(page);
    if let Some(budget) = brush_cube::vram::device_budget()
        && (wanted as u64).saturating_mul(BYTES_PER_INTERSECTION) > budget / 4
    {
        let used = client.memory_usage().bytes_in_use;
        cap = cap.min((budget.saturating_sub(used) / BYTES_PER_INTERSECTION) as usize);
    }
    cap.max(1 << 16).min(limit)
}

/// Capacity covering a seen requirement, with 5% for refine growth between
/// windows.
fn covering_max(required: usize, limit: usize) -> usize {
    required.saturating_add(required.div_ceil(20)).min(limit).max(1)
}

fn capacity_with_headroom(required: usize, limit: usize) -> usize {
    required.saturating_add(required.div_ceil(2)).min(limit).max(1)
}

type CountReadback = Pin<Box<dyn Future<Output = (u32, u32)> + Send>>;

// Training renders do not wait for their own counts: the next training render
// at the same size settles them, a step late, after the GPU has long finished
// that projection. This lets the CPU encode step N+1 while the GPU runs step N.
struct PendingCounts {
    readback: CountReadback,
    capacity: usize,
    splats: u32,
    limit: usize,
}

fn pending_counts() -> &'static Mutex<HashMap<CapacityKey, PendingCounts>> {
    static PENDING: OnceLock<Mutex<HashMap<CapacityKey, PendingCounts>>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(HashMap::new()))
}

// Lagged training renders and how many of them clamped their intersections.
static LAGGED_RENDERS: AtomicU64 = AtomicU64::new(0);
static LAGGED_OVERFLOWS: AtomicU64 = AtomicU64::new(0);

/// Settle the previous training render's counts at this size: grow the
/// capacity hint and report a clamped (overflowed) step. Returns its counts.
async fn settle_pending(key: CapacityKey) -> Option<(u32, u32)> {
    let pending = pending_counts().lock().unwrap().remove(&key)?;
    let (num_visible, num_intersections) = pending
        .readback
        .instrument(tracing::trace_span!("Settle lagged render counts"))
        .await;
    let renders = LAGGED_RENDERS.fetch_add(1, Ordering::Relaxed) + 1;
    if num_intersections as usize > pending.capacity {
        let overflows = LAGGED_OVERFLOWS.fetch_add(1, Ordering::Relaxed) + 1;
        let dropped = 1.0 - pending.capacity as f64 / num_intersections as f64;
        log::error!(
            "Intersection capacity overflow: required={num_intersections}, capacity={}, \
             dropped={:.3}% (farthest splats), overflows={overflows}/{renders} renders",
            pending.capacity,
            dropped * 100.0,
        );
        isect_log(format!(
            "Intersection capacity overflow: required={num_intersections}, capacity={}, dropped={:.3}%, overflows={overflows}/{renders}",
            pending.capacity,
            dropped * 100.0,
        ));
    }
    record_capacity(key, num_intersections as usize, pending.splats, pending.limit);
    Some((num_visible, num_intersections))
}

/// Settle every lagged training-render count. Call once training ends: a
/// readback still in flight at process exit races the driver's teardown.
pub async fn settle_lagged_counts() {
    let keys: Vec<_> = pending_counts().lock().unwrap().keys().copied().collect();
    for key in keys {
        settle_pending(key).await;
    }
    // One line per run so an overflow (a clamped step) is visible in any log.
    eprintln!(
        "lagged render counts: {} overflows / {} renders",
        LAGGED_OVERFLOWS.load(Ordering::Relaxed),
        LAGGED_RENDERS.load(Ordering::Relaxed),
    );
}

fn read_counts(
    num_visible: IntTensor<CubeBackend>,
    num_intersections: IntTensor<CubeBackend>,
) -> CountReadback {
    let tp = TransactionPrimitive::<CubeBackend>::new(
        vec![], vec![], vec![num_visible, num_intersections], vec![],
    );
    let readback = <CubeBackend as TransactionOps<CubeBackend>>::tr_execute(tp)
        .instrument(tracing::trace_span!("Read render counts"));
    Box::pin(async move {
        let data = readback.await.expect("Failed to read counts");
        let read = |i: usize| data.read_ints[i].clone().try_into_vec::<u32>().expect("render count")[0];
        (read(0), read(1))
    })
}
use tracing::Instrument;

#[doc(hidden)]
pub fn calc_tile_bounds(img_size: glam::UVec2) -> glam::UVec2 {
    uvec2(
        img_size.x.div_ceil(shaders::helpers::TILE_WIDTH),
        img_size.y.div_ceil(shaders::helpers::TILE_WIDTH),
    )
}

impl SplatOps for CubeBackend {
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
        _refine_weight: FloatTensor<Self>,
        _coeffs_grad_sq: FloatTensor<Self>,
        _pose_grad: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: RasterPass,
    ) -> RenderOutput<Self> {
        render_cube(
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
        _refine_weight: FloatTensor<Self>,
        _coeffs_grad_sq: FloatTensor<Self>,
        _pose_grad: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: RasterPass,
    ) -> RenderOutput<Self> {
        render_cube(
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
            render_mode,
            background,
            pass,
        )
        .await
    }
}

static SURFEL_LENS_DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Surfel renders in this process whose camera had lens distortion, rendered as the pinhole with the
/// same focal length and principal point. Should stay 0: surfel jobs load a pinhole copy of a
/// distorted dataset; jobs report a nonzero count.
pub fn surfel_lens_dropped() -> u64 {
    SURFEL_LENS_DROPPED.load(std::sync::atomic::Ordering::Relaxed)
}

/// [`SplatOps::render`] / [`SplatOps::render_neural`]: `decoder` is `None`
/// in SH mode.
#[allow(clippy::too_many_arguments)]
async fn render_cube(
    camera: &Camera,
    img_size: glam::UVec2,
    transforms: FloatTensor<CubeBackend>,
    sh_coeffs: FloatTensor<CubeBackend>,
    decoder: Option<FloatTensor<CubeBackend>>,
    neural_degree: u32,
    raw_opacities: FloatTensor<CubeBackend>,
    min_scale: FloatTensor<CubeBackend>,
    has_min_scale: bool,
    log_scale_offset: f32,
    render_mode: SplatRenderMode,
    background: Vec3,
    pass: RasterPass,
) -> RenderOutput<CubeBackend> {
    assert!(
        img_size[0] > 0 && img_size[1] > 0,
        "Can't render images with 0 size."
    );
    let bwd_info = pass.bwd_info();
    let smooth_cutoff = pass.smooth_cutoff();

    let transforms = into_contiguous(transforms);
    let sh_coeffs = into_contiguous(sh_coeffs);
    let raw_opacities = into_contiguous(raw_opacities);
    let min_scale = into_contiguous(min_scale);
    let neural = decoder.is_some();

    DimCheck::new()
        .check_dims("transforms", &transforms, &["D".into(), 10.into()])
        .check_dims("sh_coeffs", &sh_coeffs, &["D".into(), "C".into(), 3.into()])
        .check_dims("raw_opacities", &raw_opacities, &["D".into()]);

    let total_splats = transforms.shape()[0] as u32;
    // In neural mode the uniform carries the decoder's active direction
    // band instead (the SH kernels are compiled out).
    let sh_degree = if neural {
        assert_eq!(
            sh_coeffs.shape()[1],
            kernels::neural::NEURAL_ROW_COEFFS as usize,
            "neural appearance rows must have NEURAL_ROW_COEFFS coefficients"
        );
        neural_degree.min(kernels::neural::NEURAL_MAX_DEGREE)
    } else {
        sh_degree_from_coeffs(sh_coeffs.shape()[1] as u32)
    };
    // The kernels specialise on the SH degree; neural mode reads its band
    // from the uniform, so one compiled variant serves every band.
    let sh_degree_ct = if neural { 0 } else { sh_degree };
    let mip_splat = matches!(render_mode, SplatRenderMode::Mip);
    let surfel = matches!(render_mode, SplatRenderMode::Surfel);
    // Surfel (2DGS) ray-splat intersection is pinhole-only. Training and evaluation undistort a
    // distorted dataset first (splat_eval::undistort::pinhole_dataset); any other caller gets the
    // same focal length and principal point without the lens terms instead of a panic.
    let pinholed;
    let camera = if surfel && !matches!(camera.camera_model, crate::kernels::camera_model::CameraModel::Pinhole) {
        if SURFEL_LENS_DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            eprintln!("[render] WARNING: surfel (2DGS) rendering is pinhole-only: lens distortion ignored (the model will misalign); see surfel_lens_dropped()");
        }
        let f = camera.focal(img_size);
        pinholed = Camera {
            fov_x: crate::camera::focal_to_fov(f.x as f64, img_size.x, &crate::kernels::camera_model::CameraModel::Pinhole),
            fov_y: crate::camera::focal_to_fov(f.y as f64, img_size.y, &crate::kernels::camera_model::CameraModel::Pinhole),
            camera_model: crate::kernels::camera_model::CameraModel::Pinhole,
            ..*camera
        };
        &pinholed
    } else {
        camera
    };
    // Surfel training always renders the geometry channels: the geometry
    // losses read them and the backward replays them.
    let geometry = pass.geometry() || (surfel && bwd_info);

    // Cull splats beyond the lens' diagonal fov with some margin, but never
    // past the angle where the distortion polynomial folds back on itself:
    // those would project mirrored into the image with huge radii.
    let half_max_render_fov =
        (((camera.fov_x as f32).hypot(camera.fov_y as f32) * 1.05).min(2.0 * PI - 1e-6) * 0.5)
            .min(max_render_theta(&camera.camera_model) as f32);
    let pinhole_params = camera.build_pinhole_params(img_size);

    let world_to_local = camera.world_to_local();
    // The view-direction origin, which a camera-space affine can move.
    let center = world_to_local.inverse().translation;
    let mut project_uniforms = shaders::helpers::ProjectUniforms {
        viewmat: glam::Mat4::from(world_to_local).to_cols_array_2d(),
        camera_model: camera.camera_model,
        half_max_render_fov,
        pinhole_params,
        camera_position: [center.x, center.y, center.z, 0.0],
        img_size: img_size.into(),
        tile_bounds: calc_tile_bounds(img_size).into(),
        sh_degree,
        total_splats,
        num_visible: 0, // num_visible — not yet known.
        log_scale_offset,
        jacobian_clamp_limits: calculate_jacobian_clamp_limits(
            img_size,
            pinhole_params,
            camera.camera_model,
        ),
    };

    let device = transforms.device.clone();
    let client = transforms.client.clone();

    let (
        global_from_presort_gid,
        compact_from_global,
        opacities,
        depths,
        intersect_counts,
        max_radius,
        num_visible_buf,
        num_intersections_buf,
        presort_index,
    ) = {
        let project_uniforms: &shaders::helpers::ProjectUniforms = &project_uniforms;
        let _span = tracing::trace_span!("ProjectSplats").entered();

        let total_splats = project_uniforms.total_splats as usize;
        // The presort kernel zeros both counters; projection writes every
        // splat's tile count and radius (zero when culled). No fills.
        let num_visible_buf = create_tensor([1], &device, DType::U32);
        let num_intersections_buf = create_tensor([1], &device, DType::U32);
        let intersect_counts = create_tensor([total_splats], &device, DType::U32);
        let max_radius = create_tensor([total_splats], &device, DType::F32);
        let presort_index = create_tensor([total_splats.max(1)], &device, DType::U32);
        let cube_dim = CubeDim::new_1d(256);
        kernels::map_gaussians::presort_index_kernel::launch(
            &client,
            calculate_cube_count_elemwise(&client, total_splats.max(1), cube_dim),
            cube_dim,
            num_visible_buf.clone().into_tensor_arg(),
            num_intersections_buf.clone().into_tensor_arg(),
            presort_index.clone().into_tensor_arg(),
        );

        let global_from_presort_gid = create_tensor([total_splats.max(1)], &device, DType::U32);
        // Only backward needs the inverse map. Forward kernels never access
        // this placeholder; backward writes every entry, so no zero-fill.
        let inverse_map_len = if bwd_info { total_splats } else { 1 };
        let compact_from_global = create_tensor([inverse_map_len], &device, DType::U32);
        let opacities = create_tensor([total_splats], &device, DType::F32);
        let depths = create_tensor([total_splats.max(1)], &device, DType::F32);

        let uniforms = project_uniforms.to_launch_object();
        let cube_dim = CubeDim::new_1d(kernels::project_forward::WG_SIZE);

        kernels::project_forward::project_forward_kernel::launch(
            &client,
            calculate_cube_count_elemwise(&client, total_splats, cube_dim),
            cube_dim,
            transforms.clone().into_tensor_arg(),
            raw_opacities.clone().into_tensor_arg(),
            min_scale.clone().into_tensor_arg(),
            global_from_presort_gid.clone().into_tensor_arg(),
            compact_from_global.clone().into_tensor_arg(),
            opacities.clone().into_tensor_arg(),
            depths.clone().into_tensor_arg(),
            num_visible_buf.clone().into_tensor_arg(),
            intersect_counts.clone().into_tensor_arg(),
            num_intersections_buf.clone().into_tensor_arg(),
            max_radius.clone().into_tensor_arg(),
            uniforms,
            mip_splat,
            has_min_scale,
            bwd_info,
            camera.camera_model,
            client
                .features()
                .plane
                .contains(burn::cubecl::features::Plane::Ops),
            surfel,
        );
        (
            global_from_presort_gid,
            compact_from_global,
            opacities,
            depths,
            intersect_counts,
            max_radius,
            num_visible_buf,
            num_intersections_buf,
            presort_index,
        )
    };

    // Submit the count copy now, but do not stop CPU submission while it
    // waits for projection. All following kernels use GPU-resident counts.
    let mut count_readback = read_counts(num_visible_buf.clone(), num_intersections_buf.clone());
    let mut count_data = poll_fn(|cx| Poll::Ready(match count_readback.as_mut().poll(cx) {
        Poll::Ready(data) => Some(data),
        Poll::Pending => None,
    })).await;
    let mut validated_counts: Option<(u32, u32)> = None;

    let mip_splat = matches!(render_mode, SplatRenderMode::Mip);
    let img_size: glam::UVec2 = project_uniforms.img_size.into();
    let tile_bounds: glam::UVec2 = project_uniforms.tile_bounds.into();
    // Visibility cannot exceed the number of input splats, so this arena
    // needs no overflow retry. Its inactive prefix-scan entries are zero.
    let num_visible_sz = (total_splats as usize).max(1);
    // Project + shade the visible splats in pre-sort order (near-sequential
    // reads), then depth-sort pre-sort indices and gather the rows.
    let projected_presort = create_tensor(
        [num_visible_sz, kernels::helpers::PROJECTED_LANES_USIZE],
        &device,
        DType::F32,
    );
    // Surfels and the geometry pass carry a per-splat side buffer and use
    // the `_aux` kernel entry points; the default path launches exactly
    // the RGB kernels.
    let aux_lanes = kernels::helpers::aux_lanes(surfel, geometry) as usize;
    let use_aux = aux_lanes > 0;
    let splat_aux = create_tensor([num_visible_sz * aux_lanes.max(1)], &device, DType::F32);
    // Neural mode: the decoder rides in front of the floor in one binding
    // (see `kernels::neural`).
    let visible_min_scale = match decoder {
        Some(decoder) => {
            let decoder = into_contiguous(decoder);
            if has_min_scale {
                into_contiguous(<CubeBackend>::float_cat(vec![decoder, min_scale], 0))
            } else {
                decoder
            }
        }
        None => min_scale,
    };
    tracing::trace_span!("ProjectVisible").in_scope(|| {
        let uniforms = project_uniforms.to_launch_object();
        let cube_dim = CubeDim::new_1d(kernels::project_visible::WG_SIZE);
        macro_rules! launch_sh {
            ($elem:ty) => {
                if use_aux {
                    kernels::project_visible::project_visible_aux_kernel::launch::<$elem>(
                        &client,
                        calculate_cube_count_elemwise(&client, num_visible_sz, cube_dim),
                        cube_dim,
                        transforms.into_tensor_arg(),
                        sh_coeffs.into_tensor_arg(),
                        raw_opacities.into_tensor_arg(),
                        visible_min_scale.into_tensor_arg(),
                        global_from_presort_gid.clone().into_tensor_arg(),
                        num_visible_buf.clone().into_tensor_arg(),
                        compact_from_global.clone().into_tensor_arg(),
                        projected_presort.clone().into_tensor_arg(),
                        splat_aux.clone().into_tensor_arg(),
                        uniforms,
                        mip_splat,
                        has_min_scale,
                        bwd_info,
                        sh_degree_ct,
                        neural,
                        camera.camera_model,
                        surfel,
                    )
                } else {
                    kernels::project_visible::project_visible_kernel::launch::<$elem>(
                        &client,
                        calculate_cube_count_elemwise(&client, num_visible_sz, cube_dim),
                        cube_dim,
                        transforms.into_tensor_arg(),
                        sh_coeffs.into_tensor_arg(),
                        raw_opacities.into_tensor_arg(),
                        visible_min_scale.into_tensor_arg(),
                        global_from_presort_gid.clone().into_tensor_arg(),
                        num_visible_buf.clone().into_tensor_arg(),
                        compact_from_global.clone().into_tensor_arg(),
                        projected_presort.clone().into_tensor_arg(),
                        uniforms,
                        mip_splat,
                        has_min_scale,
                        bwd_info,
                        sh_degree_ct,
                        neural,
                        camera.camera_model,
                    )
                }
            };
        }
        match sh_coeffs.dtype {
            DType::F16 => launch_sh!(burn::tensor::f16),
            DType::F32 => launch_sh!(f32),
            dtype => panic!("SH coefficients must use f16 or f32, got {dtype:?}"),
        }
    });
    let (_, compact_from_sorted) = tracing::trace_span!("DepthSort").in_scope(|| {
        // Full 32-bit keys: dropping low mantissa bits ties near-equal
        // depths, and ties then fall back to the atomic (unordered)
        // projection order, so renders stop being deterministic.
        radix_argsort_with_count(depths, presort_index, 32, num_visible_buf.clone())
    });
    // Compact ids are pre-sort indices from here on.
    let global_from_compact_gid = global_from_presort_gid;
    let projected_splats = projected_presort;
    // Created once for every capacity retry below: a retry's intersection
    // list extends the clamped one, so it marks a superset of the visible
    // splats and rewrites the offsets of every tile either list touches.
    let visible_len = if bwd_info { total_splats as usize } else { 1 };
    let visible = create_tensor([visible_len], &device, DType::F32);
    let tile_offsets = create_tensor(
        [tile_bounds.y as usize, tile_bounds.x as usize, 2],
        &device,
        DType::U32,
    );
    // Each splat's tile count in depth order, prefix-summed in two launches:
    // block-local scans (fused with the gather and the zero-fills of
    // `visible` and `tile_offsets`), then one cube scans the block totals.
    // `map_gaussians` adds a block's offset itself.
    let (cum_hits_local, cum_hits_blocks) =
        tracing::trace_span!("PrefixSumGaussHits").in_scope(|| {
            let block = brush_scan::BLOCK_SIZE as usize;
            let local = create_tensor([num_visible_sz], &device, DType::U32);
            let blocks = create_tensor([num_visible_sz.div_ceil(block)], &device, DType::U32);
            kernels::map_gaussians::scan_sorted_counts_kernel::launch(
                &client,
                calculate_cube_count_elemwise(
                    &client,
                    num_visible_sz,
                    CubeDim::new_1d(brush_scan::BLOCK_SIZE),
                ),
                CubeDim::new_1d(brush_scan::WG),
                compact_from_sorted.clone().into_tensor_arg(),
                global_from_compact_gid.clone().into_tensor_arg(),
                intersect_counts.into_tensor_arg(),
                num_visible_buf.clone().into_tensor_arg(),
                local.clone().into_tensor_arg(),
                blocks.clone().into_tensor_arg(),
                visible.clone().into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
            );
            scan_in_place(&blocks);
            (local, blocks)
        });
    let num_tiles = tile_bounds.x * tile_bounds.y;
    let limit = ((total_splats as u64) * (num_tiles as u64))
        .min(u32::MAX as u64).max(1) as usize;
    let device_id = client.service_id().device;
    let capacity_key = (
        device_id.type_id,
        device_id.index_id,
        img_size.x,
        img_size.y,
        pass.count_slot(),
    );
    // Training renders size the arena from the previous step's count and
    // never wait for their own: an overflow clamps (dropping the farthest
    // splats' tiles) and grows the hint for the next step. Forward-only
    // renders (eval, export) and the first render at a size stay exact.
    let mut lagged_counts = None;
    if pass.training() {
        lagged_counts = settle_pending(capacity_key).await;
    }
    if CAPACITY_CHANGED.swap(false, Ordering::Relaxed) {
        // Pages freed while their render is in flight stay resident.
        let _ = client.sync().await;
        client.memory_cleanup();
    }
    let cached = intersection_capacities().lock().unwrap().get(&capacity_key).copied();
    let lagged = pass.training() && cached.is_some();
    let wanted = cached
        .map(|c| c.for_splats(total_splats, limit))
        .unwrap_or(num_visible_sz.saturating_mul(2).max(4096))
        .min(limit).max(1);
    let mut ceiling = intersection_ceiling(&client, wanted, limit);
    let mut buffer_size = wanted.min(ceiling);
    loop {
        let bounded_intersections = create_tensor([1], &device, DType::U32);
        let tile_id_from_isect = create_tensor([buffer_size], &device, DType::U32);
        let compact_gid_from_isect = create_tensor([buffer_size], &device, DType::U32);
        tracing::trace_span!("MapGaussiansToIntersect").in_scope(|| {
            let cube_dim = CubeDim::new_1d(kernels::map_gaussians::WG_SIZE);
            kernels::map_gaussians::map_gaussians_to_intersect_kernel::launch(
                &client,
                calculate_cube_count_elemwise(&client, num_visible_sz, cube_dim),
                cube_dim,
                projected_splats.clone().into_tensor_arg(),
                compact_from_sorted.clone().into_tensor_arg(),
                cum_hits_local.clone().into_tensor_arg(),
                cum_hits_blocks.clone().into_tensor_arg(),
                tile_id_from_isect.clone().into_tensor_arg(),
                compact_gid_from_isect.clone().into_tensor_arg(),
                project_uniforms.tile_bounds[0],
                project_uniforms.tile_bounds[1],
                num_visible_buf.clone().into_tensor_arg(),
                bounded_intersections.clone().into_tensor_arg(),
                client.features().plane.contains(burn::cubecl::features::Plane::Ops),
            );
        });
        let bits = u32::BITS - num_tiles.leading_zeros();
        let (tile_id_from_isect, compact_gid_from_isect) = tracing::trace_span!("Tile sort")
            .in_scope(|| radix_argsort_with_count(tile_id_from_isect, compact_gid_from_isect, bits, bounded_intersections.clone()));
        let cube_dim = CubeDim::new_1d(256);
        tracing::trace_span!("GetTileOffsets").in_scope(|| {
            get_tile_offsets::launch(
                &client,
                calculate_cube_count_elemwise(
                    &client,
                    buffer_size,
                    CubeDim::new_1d(cube_dim.x * CHECKS_PER_ITER),
                ),
                cube_dim,
                bounded_intersections.into_tensor_arg(),
                num_tiles,
                tile_id_from_isect.into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
            );
        });
        // Geometry appends its channels after RGBA in the float image.
        let out_dim = if geometry {
            4 + kernels::helpers::GEOM_OUT_CHANNELS as usize
        } else if bwd_info {
            4
        } else {
            1
        };
        let out_img = create_tensor(
            [img_size.y as usize, img_size.x as usize, out_dim],
            &device,
            DType::F32,
        );
        let (out_packed_arg, out_f32_arg) = if bwd_info {
            (create_tensor([1], &device, DType::U32), out_img.clone())
        } else {
            (out_img.clone(), create_tensor([1], &device, DType::F32))
        };
        tracing::trace_span!("Rasterize").in_scope(|| {
            let uniforms = RasterizeUniformsLaunch::new(
                project_uniforms.tile_bounds[0],
                project_uniforms.img_size[0],
                project_uniforms.img_size[1],
                background.x,
                background.y,
                background.z,
            );
            // One cube per tile, one thread per pixel in it.
            let cube_dim = CubeDim::new_1d(shaders::helpers::TILE_SIZE);
            let cube_count = calculate_cube_count_elemwise(
                &client,
                (num_tiles * shaders::helpers::TILE_SIZE) as usize,
                cube_dim,
            );
            if use_aux {
                kernels::rasterize::rasterize_aux_kernel::launch(
                    &client,
                    cube_count,
                    cube_dim,
                    compact_gid_from_isect.clone().into_tensor_arg(),
                    tile_offsets.clone().into_tensor_arg(),
                    projected_splats.clone().into_tensor_arg(),
                    out_packed_arg.into_tensor_arg(),
                    out_f32_arg.into_tensor_arg(),
                    global_from_compact_gid.clone().into_tensor_arg(),
                    visible.clone().into_tensor_arg(),
                    splat_aux.clone().into_tensor_arg(),
                    uniforms,
                    project_uniforms.tile_bounds[0],
                    bwd_info,
                    smooth_cutoff,
                    geometry,
                    surfel,
                    surfel && pass.surface_criterion(),
                );
            } else {
                kernels::rasterize::rasterize_kernel::launch(
                    &client,
                    cube_count,
                    cube_dim,
                    compact_gid_from_isect.clone().into_tensor_arg(),
                    tile_offsets.clone().into_tensor_arg(),
                    projected_splats.clone().into_tensor_arg(),
                    out_packed_arg.into_tensor_arg(),
                    out_f32_arg.into_tensor_arg(),
                    global_from_compact_gid.clone().into_tensor_arg(),
                    visible.clone().into_tensor_arg(),
                    uniforms,
                    // Precomputed divisor for the tiles-per-row split in the
                    // per-pixel index math.
                    project_uniforms.tile_bounds[0],
                    bwd_info,
                    smooth_cutoff,
                );
            }
        });
        if lagged {
            pending_counts().lock().unwrap().insert(capacity_key, PendingCounts {
                readback: match count_data.take() {
                    Some(counts) => Box::pin(std::future::ready(counts)),
                    None => count_readback,
                },
                capacity: buffer_size,
                splats: total_splats,
                limit,
            });
            // CPU counts are a step stale and only feed stats. Kernels use
            // the GPU-resident counts.
            let (num_visible, num_intersections) = lagged_counts.unwrap_or((0, 0));
            project_uniforms.num_visible = total_splats;
            return RenderOutput {
                out_img,
                aux: RenderAuxInner {
                    num_visible: num_visible.min(total_splats),
                    num_intersections,
                    visible,
                    max_radius,
                    opacities,
                    tile_offsets,
                    img_size,
                },
                projected_splats,
                compact_gid_from_isect,
                splat_aux,
                project_uniforms,
                global_from_compact_gid,
                compact_from_global,
                num_visible_gpu: num_visible_buf,
            };
        }
        let (num_visible, num_intersections) = if let Some(counts) = validated_counts {
            counts
        } else {
            let (num_visible, num_intersections) = match count_data.take() {
                Some(counts) => counts,
                None => count_readback.as_mut()
                    .instrument(tracing::trace_span!("Validate render counts"))
                    .await,
            };
            validated_counts = Some((num_visible, num_intersections));
            (num_visible, num_intersections)
        };
        assert!(num_visible <= total_splats, "Projection visible count overflow");
        assert!((num_intersections as usize) <= limit, "Projection intersection count overflow");
        if num_intersections as usize > buffer_size && buffer_size < ceiling {
            // Map clamped the intersection list, so this render is
            // incomplete. Exact renders never expose that; retry.
            log::info!("Intersection capacity retry: required={num_intersections}, capacity={buffer_size}");
            tracing::trace_span!("Retry intersection capacity",
                required = num_intersections, capacity = buffer_size).in_scope(|| {
                let wanted = capacity_with_headroom(num_intersections as usize, limit);
                ceiling = intersection_ceiling(&client, wanted, limit);
                buffer_size = wanted.min(ceiling);
            });
            continue;
        }
        if num_intersections as usize > buffer_size {
            // At the memory ceiling: keep the clamped list rather than run out.
            log::error!(
                "Intersection memory ceiling: required={num_intersections}, ceiling={buffer_size}, \
                 dropped={:.3}% (farthest splats' tiles)",
                100.0 * (1.0 - buffer_size as f64 / num_intersections as f64)
            );
        }
        record_capacity(capacity_key, num_intersections as usize, total_splats, limit);
        project_uniforms.num_visible = num_visible;
        return RenderOutput {
            out_img,
            aux: RenderAuxInner {
                num_visible,
                num_intersections,
                visible,
                max_radius,
                opacities,
                tile_offsets,
                img_size,
            },
            // Backward projection uses the validated visible count; only
            // its raster-gradient scratch retains this padded capacity.
            projected_splats,
            compact_gid_from_isect,
            splat_aux,
            project_uniforms,
            global_from_compact_gid,
            num_visible_gpu: num_visible_buf,
            compact_from_global,
        };
    }
}
