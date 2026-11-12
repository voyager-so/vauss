use std::f32::consts::FRAC_1_SQRT_2;

use crate::blur_kernels;
use crate::neural;
use crate::{
    adam_scaled::{AdamScaled, AdamState},
    config::TrainConfig,
    mask_gate::MaskGate,
    msg::{RefineStats, TrainStepStats},
    mixer::Mixer,
    multinomial::multinomial_sample,
    quat_vec::quaternion_vec_multiply,
    splat_init::bounds_from_pos,
    stats::RefineRecord,
    trajectory::Trajectories,
};
use brush_dataset::scene::SceneBatch;
use brush_loss::{ImageLossConfig, image_loss};
use brush_render::bwd::project_update::{FusedAdam, project_update};
use brush_render::bwd::render_splats_with_pass;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RasterPass, Splats, render_splats_exposure};
use brush_render::{AlphaMode, bounding_box::BoundingBox, sh::sh_coeffs_for_degree};
use burn::{
    module::{Module, Param},
    tensor::{Bool, Device, Gradients, Int, Tensor, TensorData, activation::sigmoid, s},
};

use hashbrown::HashSet;
use rand::{RngExt, SeedableRng};
use burn::optim::GradientsParams;
use tracing::{Instrument, trace_span};

pub const BOUND_PERCENTILE: f32 = 0.8;

const MIN_OPACITY: f32 = 1.0 / 255.0;

/// Fraction of training after which the Mip-Splatting 3D-filter floor stops
/// being recomputed and is held frozen (still applied), so splats settle
/// against a fixed target instead of chasing a moving floor.
const MIN_SCALE_FREEZE_FRAC: f32 = 0.9;

/// The three per-parameter Adam states of a [`Splats`] module, owned directly
/// so the trainer can update LR scaling every step and surgically edit the
/// momentum tensors during refine — all GPU-side, no record round-trips.
struct SplatOptim {
    adam: AdamScaled,
    transforms: AdamState<2>,
    sh_coeffs: AdamState<3>,
    opacities: AdamState<1>,
    /// Shared neural-appearance decoder: its own Adam (beta2 0.99), as it
    /// gets a dense gradient from every visible splat each step.
    decoder_adam: AdamScaled,
    decoder: AdamState<1>,
}

/// Step one parameter: pull its gradient, run Adam on the inner
/// (autodiff-free) tensor, and re-wrap tracking. Parameters without a
/// gradient this step are left untouched. The unfused path (surfels).
#[allow(clippy::too_many_arguments)]
fn step_param<const D: usize>(
    adam: &AdamScaled,
    lr: f64,
    param: Param<Tensor<D>>,
    state: &mut AdamState<D>,
    grads: &mut Gradients,
    grad_rows: Option<Tensor<1, Int>>,
    grad_sq_mean: Option<Tensor<D>>,
) -> Param<Tensor<D>> {
    param.map(|t| {
        let Some(grad) = t.grad_remove(grads) else {
            return t;
        };
        let stepped = adam.step(lr, t.inner(), &grad, grad_rows, grad_sq_mean, state);
        Tensor::from_inner(stepped).require_grad()
    })
}

pub struct SplatTrainer {
    config: TrainConfig,
    /// Per-step multiplier of the exponential mean-LR schedule:
    /// `lr(n) = lr_mean * decay^(n-1)`.
    lr_mean_decay: f64,
    /// Per-column LR scales for `transforms` (`means(3) + rotations(4) +
    /// log_scales(3)`) with the mean columns zeroed, and a mask of those
    /// columns: the scheduled mean LR is mixed in on the device each step so
    /// the optimizer never waits on a host upload.
    lr_scaling_fixed: Tensor<2>,
    lr_mean_columns: Tensor<2>,
    refine_record: Option<RefineRecord>,
    optim: Option<SplatOptim>,
    /// Splat count when growth started; the budget curve grows from it.
    growth_start_count: Option<u32>,
    /// Total training pixels (see `set_train_pixels`); 0 = unknown.
    train_pixels: u64,
    /// Share of splats far outside the camera rig, measured at the first
    /// refine (≈ the SfM init); scales the automatic budget.
    far_share: Option<f32>,
    /// Most splats the device's memory budget holds (see [`vram_splat_limit`]).
    vram_splat_limit: Option<u32>,
    ssim_enabled: bool,
    bounds: BoundingBox,
    step_count: u32,
    max_sh_degree: u32,
    /// Dataset units per metre the exports are written in.
    units_per_meter: f32,
    /// The export lattice, fixed when the quantisation-aware tail starts.
    qat_lattice: Option<crate::qat::Lattice>,
    /// SH degree the schedule grows towards (see `SH_DEGREE_INTERVAL`).
    target_sh_degree: Option<u32>,
    rng: rand::rngs::StdRng,
    /// Per-train-view (world center, focal in px at native res) for the
    /// Mip-Splatting 3D filter. Empty disables it. The floor itself lives on
    /// the splats (recomputed at each refine), not here.
    view_cams: Vec<(glam::Vec3, f32)>,
    /// Training cameras and grayscale images for the surfel multi-view
    /// consistency losses (mesh-goal runs); `None` disables them.
    multiview: Option<crate::multiview::MultiView>,
    /// World scale of the mean LR: 1.1x the largest camera distance from the
    /// camera centroid (3DGS's scene extent). Set with the view cameras.
    cam_extent: Option<f32>,
    /// `--auto-masks`: keeps or drops the detector masks after a probe;
    /// created on the first batch with masked alpha.
    mask_gate: Option<MaskGate>,
    /// Frozen rows as they were at the first step, on the inner device;
    /// written back after every step: transforms of the first
    /// max(`freeze_prefix`, `freeze_geometry_prefix`) rows, and (sh_coeffs,
    /// raw_opacities) of the first `freeze_prefix` rows (None when that is 0).
    frozen: Option<(Tensor<2>, Option<(Tensor<3>, Tensor<1>)>)>,
    /// Per-photo exposure trajectories, created on the first motion step.
    trajectories: Option<Trajectories>,
    /// One entry per recent step, `None` where nothing was learned.
    pending_poses: std::collections::VecDeque<Option<PendingPose>>,
    /// Joint camera refinement (`config.pose_refine`), set up by
    /// [`Self::set_pose_views`].
    pose_refine: Option<crate::pose_refine::PoseRefine>,
    /// Camera gradients of recent steps, applied `POSE_READ_DELAY` steps later.
    pending_refine: std::collections::VecDeque<Option<(crate::pose_refine::PoseGradTicket, Tensor<1>)>>,
    mixer: Option<(Mixer, burn::optim::ModuleOptimizer)>,
    /// Relative loss reduction of the blend over the sharp render, per step of
    /// `BLUR_TEST`.
    blur_test: Vec<BlurSample>,
    /// Decided at the end of `BLUR_TEST`.
    blurred_capture: Option<bool>,
    /// Per photo of a blurred capture: the half-resolution linear blur image
    /// `[1, 3, H/2, W/2]` and mask `[1, 1, H/2, W/2]` from its last full
    /// exposure step.
    blur_cache: Vec<Option<BlurCache>>,
    /// Per photo: visits since full steps stopped running on every visit.
    cached_visits: Vec<u32>,
    /// Wall time spent in full exposure steps so far.
    blur_full_spent: std::time::Duration,
    #[cfg(not(target_family = "wasm"))]
    lpips: Option<lpips::LpipsModel>,
    /// Set when a refine found non-finite splats while the depth prior was
    /// on: the prior stays off for the rest of the run (a bad prior must
    /// never take a product run down).
    depth_prior_off: bool,
    /// Set by [`Self::set_pseudo_step`] for the next step only.
    pseudo_step: Option<PseudoStep>,
}

/// A step on a generated (pseudo) view, e.g. a fixer-repaired render from the fill loop
/// (`vauss fill`). Its loss is the confidence-masked objective of the reference implementation:
/// `(1 - w) * sum(m |pred - gt|) / (3 sum(m)) + w * (1 - SSIM(pred * m, gt * m))`, scaled by
/// `weight`, where `m` is the per-pixel confidence (`mask`, `[H, W, 1]`, mean `mask_mean`) and
/// the batch's GT is already premultiplied by it (opaque, no alpha). The means move at `pos_lr`
/// times the scheduled LR. Such a view has no pose to refine, no exposure model and no transient
/// masking; its view index is not a training view's.
#[derive(Clone, Debug)]
pub struct PseudoStep {
    pub weight: f32,
    pub pos_lr: f32,
    pub mask: TensorData,
    pub mask_mean: f32,
}

/// Per-coefficient SH LR scales `[1, num_coeffs, 1]`: DC (band 0) uses full
/// LR, bands 1+ are divided by `sh_scale`.
fn sh_lr_scales(num_coeffs: usize, sh_scale: f32, device: &Device) -> Tensor<3> {
    let mut scales = vec![1.0f32; num_coeffs];
    for s in &mut scales[1..] {
        *s = 1.0 / sh_scale;
    }
    Tensor::<1>::from_floats(scales.as_slice(), device).reshape([1, num_coeffs as i32, 1])
}

/// Fraction of training at which the splat count reaches its cap.
const GROWTH_FULL_AT: f64 = 0.8;

/// Step from which each training photo learns its exposure trajectory.
const POSE_START_ITER: u32 = 1000;

/// Renders per blurred training photo, one per trajectory sub-pose, blended
/// in linear radiance (camera-motion blur).
const SUB_POSES: usize = crate::trajectory::SUB_POSES;

/// Until the capture is found blurred, only its start, middle and end
/// sub-poses render: enough for the blur test, at a third of the cost on the
/// sharp captures that never get past it.
const DETECT_POSES: [usize; 3] = [0, SUB_POSES / 2, SUB_POSES - 1];
const ALL_POSES: [usize; SUB_POSES] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

/// Steps whose losses decide whether the capture is motion-blurred: if the
/// blend of exposure renders fits the photos better than their sharp renders
/// (median over the window) the capture is blurred; otherwise the exposure
/// renders stop for the rest of training.
const BLUR_TEST: std::ops::Range<u32> = 1250..1350;

/// Median relative loss reduction (over the positive loss) above which the
/// capture counts as blurred. Only photo sets reach this test: video never uses blur
/// mode (splat-brush train.rs). Measured 2026-10-24 (brush-cli, 30k/45k schedules,
/// caps auto to 8M): sharp M360/T&T/Deep Blending -0.11 to 0.08, except M360 kitchen
/// 0.125-0.166 (52 runs; blur mode there cost 1.6 dB and 2.5x the wall). Deblur-NeRF
/// real scenes 0.164-0.58, where blur mode gains 0.9-4.3 dB on 8/10; it loses on
/// decoration (0.116-0.22, -1.9 dB) and heron (0.35-0.41, -0.6 dB). The old cut 0.095
/// fired on kitchen. At 0.17 kitchen stays off in 52/52 runs and parterre (gain +1.0 dB)
/// stays on in 23/25: the two overlap, so no cut separates them every run.
const BLUR_MIN_GAIN: f32 = 0.17;

/// One `BLUR_TEST` step: the blend's loss reduction over the sharp render, the
/// sharp render's training loss, and that loss in positive form.
struct BlurSample {
    reduction: f32,
    sharp: f32,
    positive: f32,
}

/// Shown to the user once the exposure model switches on.
pub const BLUR_MODE_NOTE: &str = "blurry capture detected; sharper result, longer training";

/// Total wall time a blurred capture spends in full exposure steps (every
/// sub-pose rendered), whatever its size. Full steps run on every visit for
/// the first `BLUR_EVERY_VISIT` of it; then each photo blends its fresh sharp
/// render with the blur image and mask cached at its last full step (about the
/// cost of a plain step), refreshing the cache with a full step every
/// `BLUR_REFRESH_EVERY`-th visit until the budget is spent. After that no full
/// step runs: cached photos keep blending, uncached ones train plainly.
const BLUR_FULL_BUDGET: std::time::Duration = std::time::Duration::from_secs(330);
const BLUR_EVERY_VISIT: f64 = 0.6;
const BLUR_REFRESH_EVERY: u32 = 8;

/// Trajectory LR, decaying linearly to `TRAJECTORY_LR_END` at the last step.
const TRAJECTORY_LR: f64 = 1e-4;
const TRAJECTORY_LR_END: f64 = 1e-6;

/// Loss weight on the trajectory's departure from rigid sub-poses.
const ORTHOGONALITY_WEIGHT: f32 = 1e-4;

/// Display <-> linear radiance exponent for blending sub-renders.
pub(crate) const GAMMA: f32 = 2.2;

/// Step from which the [`Mixer`] blends the sub-renders; before it they and
/// the sharp render are averaged equally.
const MIXER_START_ITER: u32 = 3000;

/// Mixer LR, decaying linearly to `MIXER_LR_END` at the last step.
const MIXER_LR: f64 = 1e-4;
const MIXER_LR_END: f64 = 1e-6;

/// Loss weight on the mean blend mask, preferring the sharp render.
const MASK_WEIGHT: f32 = 1e-3;

/// Steps a photo's pose gradients wait before being read back, so the read
/// never waits on the GPU queue. Each photo comes round once per epoch, so the
/// delay costs nothing.
const POSE_READ_DELAY: usize = 4;

/// Steps of pose-refinement camera gradients read back together (see the
/// readback in `step`). A view's update lands up to this many steps later;
/// it comes round about once per epoch.
const POSE_REFINE_READ_BATCH: usize = 32;

/// Camera gradient of one photo's sampled sub-pose, read back
/// `POSE_READ_DELAY` steps later.
struct PendingPose {
    view: usize,
    sub_pose: usize,
    /// The sub-render's camera-space map `T^-1` (see [`sub_pose_camera`]).
    local: glam::Mat4,
    grad: Tensor<1>,
    lr: f32,
}

/// Lagged-count slot of the depth-prior render (after the exposure sub-renders').
const DEPTH_PRIOR_SLOT: u8 = 2 + SUB_POSES as u8;

/// Fresh splats start at SH degree 0 and gain a degree every this many steps.
pub const SH_DEGREE_INTERVAL: u32 = 1000;

/// Automatic splat budget (`max_splats` = 0) for a compact scene.
/// Training bytes per splat at the run's peak (params, optimizer state and the
/// per-splat share of a step's transients; garden 2-5M slope), and the fixed
/// part, for the count ceiling the device budget allows.
const PEAK_BYTES_PER_SPLAT: u64 = 1300;
const PEAK_BASE_BYTES: u64 = 300 << 20;

/// Most splats the device budget holds, when the device reports its memory.
/// Intersection spikes of single views are bounded separately, in the renderer.
fn vram_splat_limit() -> Option<u32> {
    brush_cube::vram::device_budget().map(|budget| {
        (budget.saturating_sub(PEAK_BASE_BYTES) / PEAK_BYTES_PER_SPLAT).clamp(100_000, u32::MAX as u64) as u32
    })
}

const AUTO_MAX_SPLATS: u32 = 1_200_000;

/// Ceiling of the automatic budget for local training: 30k-step wall grows
/// ~60 s per extra million splats on an L40S, and at 2M outdoor scenes
/// (bicycle, garden) still finish within Spirula's default wall (~200 s).
const AUTO_MAX_SPLATS_LOCAL: u32 = 2_000_000;

/// Ceiling for the cloud MAX tier (`max_budget`). At an explicit 8M cap (30k,
/// RTX PRO 6000) the NVML peak is ~4.9 GB at ~7.6M splats, on 1 MP (bicycle)
/// and 1.4 MP (moskvitch 1600x900) images, within a 16 GB GPU. Above 5M,
/// unbounded scenes keep gaining LPIPS vs gsplat-MCMC 8M (bicycle: our 5M
/// ties it on test views, 6M/8M beat it on 3/3 seeds) at 0.17-0.23x its train
/// time; MAX wall there grows ~45% from 5M to 8M (parity row 6e, 2026-10-24).
const AUTO_MAX_SPLATS_MAX: u32 = 8_000_000;

/// Base of the automatic budget for the cloud MAX tier, before the far-scene
/// scale: 2.0M instead of [`AUTO_MAX_SPLATS`]. With the MAX 45k schedule it
/// is part of the best-stack M360-9 result (X38, within 0.24-0.33 dB of NHT,
/// better LPIPS); the default and local tiers keep 1.2M.
const AUTO_MAX_SPLATS_MAX_BASE: u32 = 2_000_000;

/// Scenes whose geometry reaches far beyond the cameras (unbounded outdoor
/// captures) need more splats: the automatic budget grows by
/// `FAR_BUDGET_GAIN` × the share of initial splats farther than
/// `FAR_RADIUS` camera-spreads from the camera centroid, above
/// `FAR_SHARE_FLOOR`. Measured on the SfM points: Mip-NeRF 360 indoor
/// scenes and Deep Blending rooms 0.00-0.02; truck, garden, bicycle, flowers
/// 0.22-0.48. More splats there buy +0.2-0.4 dB and much better LPIPS (1.2M
/// to 3M on bicycle/garden/stump); indoors they buy nothing.
const FAR_RADIUS: f32 = 2.0;
const FAR_SHARE_FLOOR: f32 = 0.05;
const FAR_BUDGET_GAIN: f32 = 10.0;

/// The automatic budget never exceeds one splat per this many training
/// pixels. Small-image captures (phone video frames, ~0.2 MP) otherwise get
/// ~1.2M splats for ~20-40M pixels and overfit: held-out PSNR peaks around
/// 12k steps and then falls 1.5-2.5 dB by 30k. At 30 the cap binds only on
/// such captures; standard scenes (100M+ training pixels) are unchanged.
const MIN_PIXELS_PER_SPLAT: u64 = 30;

/// Training images larger than this (pixels) skip the Mip-Splatting 3D filter.
const MIP_3D_FILTER_MAX_PIXELS: u64 = 300_000;

fn inv_sigmoid(x: Tensor<1>) -> Tensor<1> {
    (x.clone() / (1.0f32 - x)).log()
}

/// Per-splat world-space scale floor for the Mip-Splatting 3D filter:
/// `f_i = sqrt(factor) · min_v(||mean_i - cam_v|| / focal_px_v)`. `means` and
/// the result are on the inner (non-autodiff) backend; `f` is a frozen
/// constant. Returns `None` if disabled or there are no cameras.
fn compute_min_scale(
    means: &Tensor<2>,
    view_cams: &[(glam::Vec3, f32)],
    factor: f32,
) -> Option<Tensor<1>> {
    if factor <= 0.0 || view_cams.is_empty() {
        return None;
    }
    let device = means.device();
    let n = means.dims()[0];

    // Views in chunks: dist/focal as one [n, chunk] op, min over the chunk.
    // (One small kernel chain per chunk instead of per view.)
    const CHUNK: usize = 16;
    let mut min_ratio: Option<Tensor<1>> = None;
    for views in view_cams.chunks(CHUNK) {
        let k = views.len();
        let centers: Vec<f32> = views.iter().flat_map(|(c, _)| [c.x, c.y, c.z]).collect();
        let inv_focal: Vec<f32> = views.iter().map(|(_, f)| 1.0 / f.max(1e-6)).collect();
        let centers = Tensor::<1>::from_floats(centers.as_slice(), &device).reshape([1, k, 3]);
        let inv_focal = Tensor::<1>::from_floats(inv_focal.as_slice(), &device).reshape([1, k]);
        let diff = means.clone().reshape([n, 1, 3]) - centers;
        let dist = diff.clone().mul(diff).sum_dim(2).reshape([n, k]).sqrt();
        let ratio = (dist * inv_focal).min_dim(1).reshape([n]);
        min_ratio = Some(match min_ratio {
            Some(m) => m.min_pair(ratio),
            None => ratio,
        });
    }
    min_ratio.map(|r| r.mul_scalar(factor.sqrt()))
}

/// Write the frozen prefix rows back over the stepped splats (Adam, noise
/// and the tail's quantisation touched every row). Needs a fixed SH degree
/// and no refine: the prefix rows must stay rows `0..k`. Transforms cover the
/// geometry-only band too; SH/opacity only the fully frozen rows.
fn restore_frozen(
    mut splats: Splats,
    frozen: &(Tensor<2>, Option<(Tensor<3>, Tensor<1>)>),
) -> Splats {
    let (t0, colour) = frozen;
    let kg = t0.dims()[0];
    // The optimizer stores SH in f16 after its first update; the frozen copy
    // was taken before that, so match each live tensor's dtype.
    let n = splats.num_splats() as usize;
    assert!(kg <= n, "frozen prefix {kg} exceeds the {n} splats (refine must be off)");
    splats.transforms = splats.transforms.map(|t| {
        let t = t.inner();
        let out = Tensor::cat(vec![t0.clone().cast(t.dtype()), t.slice(s![kg..n, ..])], 0);
        Tensor::from_inner(out).require_grad()
    });
    let Some((sh0, o0)) = colour else { return splats };
    let k = sh0.dims()[0];
    assert_eq!(
        sh0.dims()[1],
        splats.sh_coeffs.val().dims()[1],
        "frozen prefix needs a fixed SH degree"
    );
    splats.sh_coeffs = splats.sh_coeffs.map(|t| {
        let t = t.inner();
        let out = Tensor::cat(vec![sh0.clone().cast(t.dtype()), t.slice(s![k..n, .., ..])], 0);
        Tensor::from_inner(out).require_grad()
    });
    splats.raw_opacities = splats.raw_opacities.map(|t| {
        let t = t.inner();
        let out = Tensor::cat(vec![o0.clone().cast(t.dtype()), t.slice(s![k..n])], 0);
        Tensor::from_inner(out).require_grad()
    });
    splats
}

pub async fn get_splat_bounds(splats: Splats, percentile: f32) -> BoundingBox {
    let means: Vec<f32> = splats
        .means()
        .into_data_async()
        .await
        .expect("Failed to fetch splat data")
        .try_to_vec()
        .expect("Failed to get means");
    bounds_from_pos(percentile, &means)
}

impl SplatTrainer {
    pub fn new(config: &TrainConfig, device: &Device, bounds: BoundingBox) -> Self {
        Self::new_seeded(config, device, bounds, 42)
    }

    pub fn new_seeded(
        config: &TrainConfig,
        device: &Device,
        bounds: BoundingBox,
        seed: u64,
    ) -> Self {
        // The per-step decay reaching lr_mean_end at the last iteration. With
        // one iteration or fewer there is nothing to decay over (and the
        // exponent 1/iters would be undefined), so hold the LR.
        let decay = if config.total_train_iters > 1 {
            (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_train_iters as f64)
        } else {
            1.0
        };

        let ssim_enabled = config.ssim_weight > 0.0;

        // Optimizer state lives on the inner device.
        let opt_device = device.clone().inner();
        let (rot, scale) = (config.lr_rotation as f32, config.lr_scale as f32);
        let lr_scaling_fixed = Tensor::<1>::from_floats(
            [0.0, 0.0, 0.0, rot, rot, rot, rot, scale, scale, scale],
            &opt_device,
        )
        .reshape([1, 10]);
        let lr_mean_columns = Tensor::<1>::from_floats(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &opt_device,
        )
        .reshape([1, 10]);

        // Growth is gated on the global iter. LOD phases run past
        // total_train_iters but their refines should never grow — clamp
        // here so growth_stop is never effectively past end-of-training,
        // and growth_start never past growth_stop.
        let mut config = config.clone();
        let blurred_capture = (!config.camera_motion_blur).then_some(false);
        config.growth_stop_iter = config.growth_stop_iter.min(config.total_train_iters);
        config.growth_start_iter = config.growth_start_iter.min(config.growth_stop_iter);

        #[cfg(not(target_family = "wasm"))]
        let lpips = (config.lpips_loss_weight > 0.0).then(|| lpips::load_vgg_lpips(device));

        Self {
            config,
            lr_mean_decay: decay,
            lr_scaling_fixed,
            lr_mean_columns,
            optim: None,
            growth_start_count: None,
            train_pixels: 0,
            far_share: None,
            vram_splat_limit: vram_splat_limit(),
            refine_record: None,
            ssim_enabled,
            bounds,
            step_count: 0,
            max_sh_degree: 0,
            units_per_meter: 1.0,
            qat_lattice: None,
            target_sh_degree: None,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            view_cams: Vec::new(),
            multiview: None,
            cam_extent: None,
            mask_gate: None,
            frozen: None,
            trajectories: None,
            pending_poses: std::collections::VecDeque::new(),
            pose_refine: None,
            pending_refine: std::collections::VecDeque::new(),
            mixer: None,
            blur_test: Vec::new(),
            blurred_capture,
            blur_cache: Vec::new(),
            cached_visits: Vec::new(),
            blur_full_spent: std::time::Duration::ZERO,
            #[cfg(not(target_family = "wasm"))]
            lpips,
            depth_prior_off: false,
            pseudo_step: None,
        }
    }

    /// Percentile bounding box of the splats, refreshed on each refine.
    pub fn bounds(&self) -> BoundingBox {
        self.bounds
    }

    /// Moves the run to a new schedule mid-training (a blurred capture's shorter MAX schedule):
    /// takes the step-indexed knobs from `config` and re-derives the mean-LR decay, as if
    /// training had started with them.
    pub fn retarget(&mut self, config: &TrainConfig) {
        let c = &mut self.config;
        c.total_train_iters = config.total_train_iters;
        c.refine_every = config.refine_every;
        c.sh_degree_interval = config.sh_degree_interval;
        c.growth_stop_iter = config.growth_stop_iter.min(c.total_train_iters);
        c.growth_start_iter = config.growth_start_iter.min(c.growth_stop_iter);
        if c.total_train_iters > 1 {
            self.lr_mean_decay = (c.lr_mean_end / c.lr_mean).powf(1.0 / c.total_train_iters as f64);
        }
        if let Some(p) = self.pose_refine.as_mut() {
            p.set_total(c.total_train_iters);
        }
    }

    /// Enable joint camera refinement (when `config.pose_refine`; SH 3DGS
    /// only) for the training views, given each one's intrinsics group. Call
    /// after [`Self::set_view_cams`] (the translation LR is in camera extents).
    /// Returns whether refinement is on.
    pub fn set_pose_views(&mut self, view_group: Vec<usize>) -> bool {
        let on = self.config.pose_refine && !self.config.surfel && !self.config.neural;
        self.pose_refine = on.then(|| {
            let extent = self.cam_extent.unwrap_or(self.bounds.median_size());
            crate::pose_refine::PoseRefine::new(&self.config, view_group, extent)
        });
        on
    }

    /// The loader's view index -> training view index (see [`Self::set_pose_views`]).
    pub fn set_pose_loader_views(&mut self, loader_to_view: Vec<usize>) {
        if let Some(p) = self.pose_refine.as_mut() {
            p.set_loader_views(loader_to_view);
        }
    }

    pub fn pose_refine(&self) -> Option<&crate::pose_refine::PoseRefine> {
        self.pose_refine.as_ref()
    }

    /// Mutable refinement state (the pose-only BA sets the corrections).
    pub fn pose_refine_mut(&mut self) -> Option<&mut crate::pose_refine::PoseRefine> {
        self.pose_refine.as_mut()
    }

    /// Translation unit of pose refinement and alignment: the camera extent.
    pub fn cam_extent(&self) -> f32 {
        self.cam_extent.unwrap_or(self.bounds.median_size())
    }

    /// Research early-stop: skip converged fitting steps, retaining the full QAT tail.
    pub fn enter_qat_tail(&mut self) -> Option<u32> {
        let total = self.config.total_train_iters;
        let start = total - total / crate::qat::TAIL_FRACTION;
        if !self.config.qat_tail || self.config.neural || self.step_count >= start {
            return None;
        }
        self.step_count = start;
        Some(start)
    }

    /// Test-time appearance for held-out views (see
    /// [`crate::eval::EvalAppearance`]); the PPISP controller plugs in here.
    /// `None` until a trained appearance model exists.
    pub fn eval_appearance(&self) -> Option<std::sync::Arc<dyn crate::eval::EvalAppearance>> {
        None
    }

    /// Training cameras and grayscale images for the surfel multi-view losses.
    pub fn set_multiview(&mut self, mv: crate::multiview::MultiView) {
        self.multiview = Some(mv);
    }

    /// Supply per-train-view (world center, focal-px at native res): they set
    /// the mean LR's world scale and enable the Mip-Splatting 3D filter
    /// (gated on `config.min_scale_factor > 0`).
    pub fn set_view_cams(&mut self, view_cams: Vec<(glam::Vec3, f32)>) {
        if !view_cams.is_empty() {
            let n = view_cams.len() as f32;
            let center = view_cams.iter().map(|(c, _)| *c).sum::<glam::Vec3>() / n;
            let max_dist = view_cams.iter().map(|(c, _)| c.distance(center)).fold(0.0, f32::max);
            self.cam_extent = Some(1.1 * max_dist.max(1e-3));
        }
        self.view_cams = view_cams;
    }

    /// SH degree to grow towards (SH degree schedule); fresh initial
    /// splats should then start at degree 0.
    /// Final splat count: `max_splats`, or when that is 0 the count Spirula's
    /// growth rate (x1.05 per 100 steps) reaches from `n0` over the growth
    /// window, at most [`Self::auto_budget`].
    fn splat_cap(&self, n0: u32) -> u32 {
        if self.config.max_splats > 0 {
            return self.config.max_splats;
        }
        let end = self.config.growth_stop_iter.min(self.config.total_train_iters);
        let steps = end.saturating_sub(self.config.growth_start_iter) as f64;
        let factor = 1.05f64.powf(steps / 100.0);
        ((n0 as f64 * factor) as u32).min(self.auto_budget()).max(n0)
    }

    /// The automatic budget before the growth-rate limit: scaled up for
    /// scenes reaching far beyond the cameras, at most one splat per
    /// `MIN_PIXELS_PER_SPLAT` training pixels.
    fn auto_budget(&self) -> u32 {
        self.auto_budget_at(self.far_share.unwrap_or(0.0))
    }

    /// The splat budget this run is expected to train at, from its inputs alone (before any step):
    /// the explicit cap, else the automatic budget with the far share measured on the initial
    /// splats' `means` (the trainer measures it again at the first refine). An upper bound: the
    /// growth-rate limit can stop below it. Call after [`Self::set_view_cams`] and
    /// [`Self::set_train_pixels`].
    pub fn expected_splat_budget(&self, means: impl Iterator<Item = glam::Vec3>) -> u32 {
        if self.config.max_splats > 0 {
            return self.config.max_splats;
        }
        self.auto_budget_at(self.measure_far_share(means))
    }

    /// Switch joint camera refinement on or off before [`Self::set_pose_views`].
    pub fn set_pose_refine(&mut self, on: bool) {
        self.config.pose_refine = on;
    }

    fn auto_budget_at(&self, far: f32) -> u32 {
        let by_pixels = match self.train_pixels {
            0 => u32::MAX,
            px => (px / MIN_PIXELS_PER_SPLAT).min(u32::MAX as u64) as u32,
        };
        let scale = 1.0 + FAR_BUDGET_GAIN * (far - FAR_SHARE_FLOOR).max(0.0);
        let (base, ceiling) = if self.config.max_budget {
            (AUTO_MAX_SPLATS_MAX_BASE, AUTO_MAX_SPLATS_MAX)
        } else {
            (AUTO_MAX_SPLATS, AUTO_MAX_SPLATS_LOCAL)
        };
        ((base as f32 * scale) as u32)
            .min(ceiling)
            .min(by_pixels)
            .min(self.vram_splat_limit.unwrap_or(u32::MAX))
    }

    /// Share of `means` (`[x, y, z]` rows) farther than `FAR_RADIUS` camera
    /// spreads (95th-percentile camera distance) from the camera centroid. 0 without view cameras.
    fn measure_far_share(&self, means: impl Iterator<Item = glam::Vec3>) -> f32 {
        if self.view_cams.is_empty() {
            return 0.0;
        }
        let n = self.view_cams.len() as f32;
        let center = self.view_cams.iter().map(|(c, _)| *c).sum::<glam::Vec3>() / n;
        // The 95th-percentile camera distance, not the farthest: halloween's 5 fps solve placed 35
        // of 172 cameras ~13,800 units off (a broken join), the max-based spread read 0% far (56% on
        // the 2 fps solve) and the budget fell 5.0M -> 2.0M.
        let mut dists: Vec<f32> = self.view_cams.iter().map(|(c, _)| c.distance(center)).collect();
        dists.sort_unstable_by(f32::total_cmp);
        let spread = dists[(dists.len() - 1) * 95 / 100];
        let (mut far, mut total) = (0usize, 0usize);
        for m in means {
            total += 1;
            far += usize::from(m.distance(center) > FAR_RADIUS * spread.max(1e-6));
        }
        far as f32 / total.max(1) as f32
    }

    /// Dataset units per metre the exports are written in.
    pub fn set_units_per_meter(&mut self, units_per_meter: f32) {
        self.units_per_meter = units_per_meter;
    }

    /// Total training pixels (all train views at training resolution);
    /// bounds the automatic splat budget.
    pub fn set_train_pixels(&mut self, pixels: u64) {
        self.train_pixels = pixels;
        let per_view = pixels / self.view_cams.len().max(1) as u64;
        eprintln!(
            "Mip 3D filter {} ({per_view} px per training image, threshold {MIP_3D_FILTER_MAX_PIXELS})",
            if per_view <= MIP_3D_FILTER_MAX_PIXELS { "on" } else { "off" }
        );
    }

    pub fn set_target_sh_degree(&mut self, degree: u32) {
        self.target_sh_degree = Some(degree);
    }

    /// Raise the splats' SH degree by one, zero-padding the new coefficients and
    /// their Adam moments and rebuilding the per-band LR scales.
    fn grow_sh_degree(&mut self, splats: Splats) -> Splats {
        let degree = splats.sh_degree() + 1;
        let splats = splats.with_sh_degree(degree);
        let num_coeffs = sh_coeffs_for_degree(degree) as usize;
        if let Some(optim) = self.optim.as_mut() {
            let opt_device = splats.device().inner();
            optim.sh_coeffs.scaling = Some(sh_lr_scales(
                num_coeffs,
                self.config.lr_coeffs_sh_scale,
                &opt_device,
            ));
            // The first moment is [N, K, 3]; the reduced second moment is
            // [N, 1, 1] and stays as is.
            optim.sh_coeffs.map_momentum(|m| {
                let [n, cur, c] = m.dims();
                if c == 1 || cur >= num_coeffs {
                    return m;
                }
                let zeros = Tensor::<3>::zeros([n, num_coeffs - cur, c], &m.device());
                Tensor::cat(vec![m, zeros], 1)
            });
        }
        self.max_sh_degree = self.max_sh_degree.max(degree);
        splats
    }

    pub async fn step(&mut self, batch: SceneBatch, splats: Splats) -> (Splats, TrainStepStats) {
        let (mut splats, stats) = self.step_inner(batch, splats).await;
        // The quantisation tail's last step snaps the params onto the export
        // lattice. Done here, once the step's renders, gradients and fused
        // context are dropped: while any of them still references the params,
        // the in-place snap has to copy them, a fresh SH-sized pool page at the
        // very end of training (+176 MiB board on room-2M, the run's peak).
        if self.config.qat_tail
            && !self.config.neural
            && self.step_count == self.config.total_train_iters
            && let Some(lattice) = &self.qat_lattice
        {
            splats = crate::qat::snap(splats, lattice);
            if let Some(frozen) = &self.frozen {
                splats = restore_frozen(splats, frozen);
            }
        }
        (splats, stats)
    }

    /// Make the next [`Self::step`] a pseudo-view step (see [`PseudoStep`]).
    pub fn set_pseudo_step(&mut self, pseudo: PseudoStep) {
        self.pseudo_step = Some(pseudo);
    }

    async fn step_inner(&mut self, batch: SceneBatch, splats: Splats) -> (Splats, TrainStepStats) {
        let mut splats = splats;
        let pseudo = self.pseudo_step.take();
        if self.config.neural {
            if splats.decoder.is_none() {
                splats = neural::to_neural(splats, &mut self.rng);
            }
            // One more direction band per interval (band 0 always on).
            splats.neural_degree = (self.step_count / self.config.sh_degree_interval.max(1))
                .min(brush_render::kernels::neural::NEURAL_MAX_DEGREE);
        } else if self.step_count > 0
            && self.step_count.is_multiple_of(self.config.sh_degree_interval.max(1))
            && let Some(target) = self.target_sh_degree
            && splats.sh_degree() < target
        {
            splats = self.grow_sh_degree(splats);
        }

        // Track max SH degree from the first splats we see.
        if self.step_count == 0 {
            self.max_sh_degree = splats.sh_degree();
            let k = self.config.freeze_prefix as usize;
            // Geometry-only band [k, kg); kg == k when unused (the plain freeze).
            let kg = (self.config.freeze_geometry_prefix as usize).max(k);
            if kg > 0 {
                assert!(
                    kg <= splats.num_splats() as usize,
                    "freeze prefix {kg} exceeds the {} splats",
                    splats.num_splats()
                );
                // Copied through the host once, so no view of the live
                // parameter buffers (which the fused step updates in place)
                // is kept.
                async fn own<const D: usize>(t: Tensor<D>) -> Tensor<D> {
                    let device = t.device();
                    let data = t.into_data_async().await.expect("reading frozen splats");
                    Tensor::from_data(data, &device)
                }
                let colour = if k > 0 {
                    Some((
                        own(splats.sh_coeffs.val().inner().slice(s![0..k, .., ..])).await,
                        own(splats.raw_opacities.val().inner().slice(s![0..k])).await,
                    ))
                } else {
                    None
                };
                self.frozen = Some((
                    own(splats.transforms.val().inner().slice(s![0..kg, ..])).await,
                    colour,
                ));
            }
        }
        self.step_count += 1;

        let [img_h, img_w] = batch.img_size();
        let depth_prior = batch.depth.clone();
        // The view's refined camera; with a ticket its pose learns this step.
        let (camera, pose_ticket) = match self.pose_refine.as_ref() {
            Some(p) if !splats.surfel && !self.config.neural && pseudo.is_none() => {
                p.camera(&batch.camera, batch.view_index, self.step_count)
            }
            _ => (batch.camera, None),
        };

        let device = splats.device();
        crate::memory::log_phase(&device, self.step_count, "step_start", splats.num_splats());
        let has_alpha = batch.has_alpha;
        // GT lives on the GPU as packed `[H, W]` u32 (RGBA u8). All mixing
        // (bg compositing, alpha matching, mask) is folded into the loss
        // kernels; no f32 GT image is ever materialised here.
        // GT is pure data — never differentiated. Build it on the inner
        // backend so it doesn't inherit the autodiff device's residual
        // checkpointing flag (the LPIPS `unpack_gt_rgb` path, via
        // `unwrap_wgpu_int`, expects a clean Wgpu tensor).
        let gt_packed: Tensor<2, Int> =
            Tensor::from_data(batch.img_packed, &device.clone().inner());
        let img_size = glam::uvec2(img_w as u32, img_h as u32);
        let base = &self.config.background_color;
        let base_bg = glam::Vec3::new(base[0], base[1], base[2]);
        let background = sample_background_color(
            base_bg,
            self.config.background_noise_strength,
            &mut self.rng,
        );

        let median_scale = self.bounds.median_size();

        let lr_unit =
            self.config.lr_mean * self.lr_mean_decay.powi(self.step_count as i32 - 1);
        let lr_mean = lr_unit
            * self.cam_extent.unwrap_or(median_scale) as f64
            * pseudo.as_ref().map_or(1.0, |p| p.pos_lr as f64);
        let masked_alpha = batch.alpha_mode == AlphaMode::Masked;
        if self.config.auto_masks && masked_alpha && pseudo.is_none() && self.mask_gate.is_none() {
            self.mask_gate = Some(MaskGate::new(self.config.total_train_iters));
        }
        if let Some(g) = self.mask_gate.as_mut() {
            g.maybe_decide(self.step_count).await;
        }
        // The detector's masks as loaded, for the gate's probe; the loss sees
        // them only while the gate keeps them. Pseudo views (alpha is a
        // confidence, not a mask) are never gated.
        let gate = self.mask_gate.as_ref().filter(|_| pseudo.is_none());
        let mask_gt = gate.map(|_| gt_packed.clone());
        let gt_packed = match gate {
            Some(g) => g.apply(gt_packed),
            None => gt_packed,
        };
        // Until the gate turns the masks on, the view trains exactly as an
        // unmasked one: no mask flag, no alpha compositing.
        let unmasked = gate.is_some_and(|g| !g.masks_on());
        let has_alpha = has_alpha && !unmasked;
        let masked_alpha = masked_alpha && !unmasked;
        let do_alpha_match = has_alpha && !masked_alpha && self.config.match_alpha_weight > 0.0;

        let total = self.config.total_train_iters;
        let tail = total / crate::qat::TAIL_FRACTION;
        // Neural rows are latents, not SH: they skip the SPZ lattice.
        let qat = self.config.qat_tail
            && !self.config.neural
            && self.step_count > total - tail
            && self.step_count <= total;
        // The quantisation-aware tail decays every LR 10x, settling params into their codes.
        let tail_lr = if qat {
            0.1f64.powf((self.step_count + tail - total) as f64 / tail as f64)
        } else {
            1.0
        };

        // The optimizer runs with base LR 1 and these per-column scales.
        let lr_scaling = (self.lr_scaling_fixed.clone()
            + self.lr_mean_columns.clone() * lr_mean as f32)
            .mul_scalar(tail_lr as f32);

        let mut qat_residuals = None;
        let pose_active = self.step_count > POSE_START_ITER;
        let progress = self.step_count as f64 / self.config.total_train_iters.max(1) as f64;
        let active = (self.step_count.saturating_sub(POSE_START_ITER) as f64
            / self.config.total_train_iters.saturating_sub(POSE_START_ITER).max(1) as f64)
            .min(1.0);
        let trajectory_lr = TRAJECTORY_LR + (TRAJECTORY_LR_END - TRAJECTORY_LR) * active;
        // Pose learning follows the SH colour path, so neural appearance runs
        // without the exposure model. The quantisation tail keeps it: blurred
        // photos fit against sharp renders would re-learn the blur.
        let exposure = !splats.surfel
            && !self.config.neural
            && pseudo.is_none()
            && pose_active
            && self.blurred_capture != Some(false);
        if self.blur_cache.len() <= batch.view_index {
            self.blur_cache.resize(batch.view_index + 1, None);
            self.cached_visits.resize(batch.view_index + 1, 0);
        }
        let step_start = std::time::Instant::now();
        let every_visit = self.blur_full_spent.as_secs_f64()
            < BLUR_FULL_BUDGET.as_secs_f64() * BLUR_EVERY_VISIT;
        let budget_left = self.blur_full_spent < BLUR_FULL_BUDGET;
        // A photo's cache from a lower training resolution is upsampled when
        // blended, so only a photo with no cache needs a full step.
        let (cached, full_allowed) = if exposure && self.blurred_capture == Some(true) && !every_visit {
            let visits = &mut self.cached_visits[batch.view_index];
            *visits += 1;
            let refresh = budget_left && visits.is_multiple_of(BLUR_REFRESH_EVERY);
            let cached = (!refresh)
                .then(|| self.blur_cache[batch.view_index].clone())
                .flatten();
            (cached, budget_left)
        } else {
            (None, true)
        };
        // A full exposure step renders every sub-pose; a cached one only the
        // photo's own camera; once the budget is spent an uncached photo
        // trains plainly.
        let motion = exposure && cached.is_none() && full_allowed;
        // Renders registered dense gradients (several summed): motion steps and depth-prior steps.
        let mut dense_step = motion;
        let transforms = motion.then(|| {
            let seed = self.rng.random::<u64>();
            self.trajectories
                .get_or_insert_with(|| Trajectories::new(median_scale, seed))
                .transforms(batch.view_index, c2w_rows(&camera))
        });
        let poses: &[usize] = if self.blurred_capture == Some(true) {
            &ALL_POSES
        } else {
            &DETECT_POSES
        };
        // One sub-render carries the backward, scaled to stand for all.
        let sampled = poses[self.rng.random_range(0..poses.len())];
        let sampled_local = transforms
            .as_ref()
            .map_or(glam::Mat4::IDENTITY, |t| t[sampled].inverse());

        let (
            mut grads,
            visible,
            opacities,
            num_visible,
            loss_inner,
            max_radius,
            compact_from_global,
            fused_ctx,
            unfused_stats,
            pose_grad,
            refine_grad,
        ) = {
            // The splats already carry their 3D-filter floor (set at refine);
            // the render path folds it in. Optimizer/refine work on raw params.
            if qat {
                // The tail's buffers have the sizes of the steps before it (the
                // offset codes aside), so its steps reuse the pool's pages as is.
                if self.qat_lattice.is_none() {
                    let lattice = crate::qat::Lattice::new(&splats, self.units_per_meter).await;
                    self.qat_lattice = Some(lattice);
                }
                let (quantised, residuals) = crate::qat::quantise(
                    splats,
                    self.qat_lattice.as_ref().unwrap(),
                    self.step_count,
                );
                splats = quantised;
                qat_residuals = Some(residuals);
            }
            // The tail renders and steps the quantised params themselves.
            let render_input = splats.clone();
            // Projection backward runs inside the transforms/opacity step;
            // surfels and neural appearance keep the plain backward (the fused
            // kernel is SH 3DGS only).
            // A blurred photo renders along its learned camera path at
            // `SUB_POSES` exposure times, plus once at its own camera; only the
            // sampled sub-render and the sharp one run backward. Their summed
            // gradients need dense rows, and each render keeps its lagged
            // counts in its own slot so none waits on the one before it.
            // Depth prior (on every `depth_prior_every`-th step of a view that
            // has one, decaying to 0 at `depth_prior_end`): an inverse-depth
            // render at the prior's size whose dense gradients add to the
            // photo render's, so that one renders dense too.
            let depth_step = {
                let c = &self.config;
                let every = c.depth_prior_every.max(1);
                let end = c.depth_prior_end as f64;
                depth_prior
                    .clone()
                    .filter(|_| {
                        !motion
                            && !self.depth_prior_off
                            && !qat
                            && !splats.surfel
                            && !self.config.neural
                            && c.depth_prior_weight > 0.0
                            && progress < end
                            && self.step_count.is_multiple_of(every)
                    })
                    .map(|map| {
                        let decay = (1.0 - progress / end) as f32 * every as f32;
                        (map, c.depth_prior_weight * decay, c.depth_planar_weight * decay)
                    })
            };
            dense_step |= depth_step.is_some();
            let pass = if dense_step {
                RasterPass::BackwardDense { slot: 1 }
            } else if splats.surfel || self.config.neural {
                RasterPass::Backward
            } else {
                RasterPass::BackwardFusedUpdate
            };
            // Multi-view: the neighbor view is rendered first, as a plain
            // geometry pass (no backward; a second differentiable render
            // interleaved with the primary one corrupts the primary's backward).
            let mv_pre = {
                let c = &self.config;
                let mv_on = splats.surfel
                    && self.step_count > c.surfel_step(c.surfel_mv_start)
                    && self.step_count % c.surfel_mv_every.max(1) == 0
                    && (c.surfel_mv_geo_weight > 0.0 || c.surfel_mv_ncc_weight > 0.0);
                match self.multiview.as_ref().filter(|_| mv_on) {
                    Some(mv) => match mv.view_of(&camera).and_then(|v| mv.neighbor(v, self.step_count).map(|n| (v, n))) {
                        Some((view, ncam)) => {
                            let nout =
                                brush_render::gaussian_splats::render_splats_geometry_train_depth(
                                    splats.clone(),
                                    ncam,
                                    img_size,
                                    background,
                                )
                                    .await;
                            if self.step_count <= c.surfel_step(c.surfel_mv_start) + c.surfel_mv_every.max(1) {
                                eprintln!(
                                    "Multi-view on at {}: geo {} ncc {} samples {} every {}",
                                    self.step_count,
                                    c.surfel_mv_geo_weight,
                                    c.surfel_mv_ncc_weight,
                                    c.surfel_mv_ncc_samples,
                                    c.surfel_mv_every
                                );
                            }
                            Some((view, nout.geom.detach()))
                        }
                        None => None,
                    },
                    None => None,
                }
            };
            // Rendered first, like the exposure sub-renders: its own slot and size.
            let depth_out = match &depth_step {
                Some((map, w, planar_w)) => {
                    let out = brush_render::bwd::render_splats_inv_depth(
                        render_input.clone(),
                        &camera,
                        glam::uvec2(map.w as u32, map.h as u32),
                        median_scale,
                        DEPTH_PRIOR_SLOT,
                    )
                    .instrument(trace_span!("DepthPrior"))
                    .await;
                    Some((out.img, map.clone(), *w, *planar_w))
                }
                None => None,
            };
            let mut outs = Vec::with_capacity(2);
            let mut images = Vec::with_capacity(poses.len() + 1);
            // The blur is low-pass along the camera path: its sub-renders go
            // at half resolution, a quarter of the raster work and memory.
            let sub_size = (img_size + 1) / 2;
            if let Some(transforms) = &transforms {
                for &k in poses {
                    let sub_camera = sub_pose_camera(&camera, &transforms[k]);
                    let slot = 2 + k as u8;
                    if k == sampled {
                        let out = render_splats_with_pass(
                            render_input.clone(),
                            &sub_camera,
                            sub_size,
                            background,
                            RasterPass::BackwardDense { slot },
                            true,
                        )
                        .instrument(trace_span!("Forward"))
                        .await;
                        images.push(scale_grad(out.img.clone(), poses.len() as f32));
                        outs.push(out);
                    } else {
                        let img = render_splats_exposure(
                            render_input.valid(),
                            &sub_camera,
                            sub_size,
                            background,
                            slot,
                        )
                        .instrument(trace_span!("Forward"))
                        .await;
                        images.push(Tensor::from_inner(img));
                    }
                }
            }
            let out = render_splats_with_pass(
                render_input,
                &camera,
                img_size,
                background,
                pass,
                pose_ticket.is_some(),
            )
            .instrument(trace_span!("Forward"))
            .await;
            images.push(out.img.clone());
            outs.push(out);

            crate::memory::log_phase(&device, self.step_count, "forward", splats.num_splats());
            let weighted = self.step_count > MIXER_START_ITER;
            let (pred_image, mask_mean) = if motion {
                let mixer = &self
                    .mixer
                    .get_or_insert_with(|| {
                        (
                            Mixer::new(SUB_POSES, &device),
                            burn::optim::AdamConfig::new().init(),
                        )
                    })
                    .0;
                let (pred, mask_mean, cache) = blend(images, weighted.then_some(mixer));
                if self.blurred_capture == Some(true) {
                    self.blur_cache[batch.view_index] = Some(cache);
                }
                (pred, mask_mean)
            } else if let Some((blurred, mask)) = cached {
                (blend_cached(outs[0].img.clone(), blurred, mask), None)
            } else {
                (outs[0].img.clone(), None)
            };
            // Visibility and growth statistics follow the photo's own camera.
            let diff_out = outs.pop().expect("the photo's own render");
            let sub_outs = outs;
            let refine_weight_holder = diff_out.refine_weight_holder;
            let visible = diff_out.visible;
            let max_radius = diff_out.max_radius;
            let opacities = diff_out.opacities;
            let compact_from_global = diff_out.compact_from_global;
            let geom = diff_out.geom;

            // RGB loss is `(1 - w) * L1 + (-w) * SSIM` per pixel. Bg
            // compositing always runs in the kernel; for synthesised opaque
            // alpha or zero bg it's a no-op. Mask multiplies the loss-map
            // by `gt.a`; for synthesised opaque alpha that's a no-op too.
            // Alpha matching needs a real alpha source (synthesised
            // a = 1 would pull predicted alpha to fully opaque); we feed
            // `pred` with 4 channels and the kernel's `c == 3` workgroup
            // emits `|pred.a - gt.a|` into the alpha channel.
            let (l1_w, ssim_w) = if self.ssim_enabled {
                (1.0 - self.config.ssim_weight, -self.config.ssim_weight)
            } else {
                (1.0, 0.0)
            };
            // Only composite when there's a real alpha channel and a non-zero
            // bg to mix in; the kernel skips the per-pixel `(1-a)*bg` math
            // entirely when this is None.
            // A pseudo view's alpha is a confidence, never transparency: no compositing.
            let composite_bg = (has_alpha && pseudo.is_none() && background != glam::Vec3::ZERO)
                .then_some(background);
            if let (Some(g), Some(gt)) = (self.mask_gate.as_mut(), mask_gt.as_ref()) {
                g.record(self.step_count, &pred_image.clone().inner(), gt);
            }
            let base_gt = gt_packed.clone();
            // Robust loss: zero the GT alpha on transient pixels (people,
            // cars) and run the masked loss, so they neither pull colours nor
            // drive densification. User/SAM masks stay hard zeros. Skipped
            // when alpha is real transparency rather than a mask.
            let robust = self.config.robust_loss
                && pseudo.is_none()
                && self.step_count >= brush_loss::ROBUST_START_ITER
                && (!has_alpha || masked_alpha);
            let loss_gt = if robust {
                brush_loss::robust_gt(pred_image.clone().inner(), base_gt)
            } else {
                base_gt
            };
            let cfg = ImageLossConfig {
                l1_weight: l1_w,
                ssim_weight: ssim_w,
                composite_bg,
                mask: masked_alpha || robust,
                alpha_weight: if do_alpha_match {
                    self.config.match_alpha_weight
                } else {
                    0.0
                },
            };
            // The kernel takes the RGBA image as rendered.
            // `loss` is only reassigned by the LPIPS path below, which is
            // compiled out on wasm — so `mut` is unused there.
            let sharp_loss = (motion && BLUR_TEST.contains(&self.step_count))
                .then(|| image_loss(diff_out.img.clone().detach(), loss_gt.clone(), cfg).inner());
            let mut loss = match &pseudo {
                Some(p) => {
                    let m: Tensor<3> = Tensor::from_data(p.mask.clone(), &device);
                    let pred_m = pred_image.clone().slice(s![.., .., 0..3]) * m;
                    let gt_m: Tensor<3> = Tensor::from_inner(brush_loss::unpack_gt_rgb(loss_gt.clone(), None));
                    let l1 = (pred_m.clone() - gt_m).abs().mean() / p.mask_mean.max(1e-3);
                    let ssim_only = ImageLossConfig { l1_weight: 0.0, ..cfg };
                    l1 * l1_w + image_loss(pred_m, loss_gt, ssim_only)
                }
                None => image_loss(pred_image.clone(), loss_gt, cfg),
            };
            // Read each step: the window is short, and a loss tensor kept for
            // later can be overwritten by the kernels that reuse its buffer.
            if let Some(sharp_loss) = sharp_loss {
                let both: Vec<f32> = Tensor::cat(vec![sharp_loss, loss.clone().inner()], 0)
                    .into_data_async()
                    .await
                    .expect("Failed to read blur-test losses")
                    .try_to_vec()
                    .expect("losses are f32");
                // The training loss `(1 - w) L1 - w SSIM` crosses zero on a
                // sharp, high-SSIM capture, so the gain is relative to its
                // positive form `(1 - w) L1 + w (1 - SSIM)` (= loss + w).
                self.blur_test.push(BlurSample {
                    reduction: both[0] - both[1],
                    sharp: both[0],
                    positive: both[0] - ssim_w,
                });
            }
            if let Some(mask_mean) = mask_mean {
                loss = loss + mask_mean * MASK_WEIGHT;
            }
            if let Some(p) = &pseudo {
                loss = loss * p.weight;
            }

            // LPIPS still needs an f32 RGB tensor for VGG. Materialising it
            // here costs ~99 MB at 4K, only when LPIPS is enabled.
            #[cfg(not(target_family = "wasm"))]
            if let Some(lpips) = &self.lpips {
                let gt_rgb = brush_loss::unpack_gt_rgb(gt_packed.clone(), composite_bg);
                let gt_rgb_diff: Tensor<3> = Tensor::from_inner(gt_rgb);
                loss = loss
                    + lpips.lpips(
                        pred_image.clone().slice(s![.., .., 0..3]).unsqueeze_dim(0),
                        gt_rgb_diff.unsqueeze_dim(0),
                    ) * self.config.lpips_loss_weight;
            }

            if let Some((depth_img, map, w, planar_w)) = depth_out {
                if planar_w > 0.0
                    && let Some(planar) = crate::depth_prior::planarity_loss(depth_img.clone(), &map)
                {
                    loss = loss + planar * planar_w;
                }
                loss = loss + crate::depth_prior::depth_prior_loss(depth_img, &map) * w;
            }

            // 2DGS surfel geometry losses (normal consistency, depth
            // distortion), each switched on at its own step.
            let geom_for_mv = geom.clone();
            let loss = match geom.filter(|_| self.config.surfel) {
                Some(geom) => {
                    let c = &self.config;
                    let on = |start: u32, weight: f32| {
                        if self.step_count >= start { weight } else { 0.0 }
                    };
                    let pin = camera.build_pinhole_params(img_size);
                    let geo = brush_loss::surfel_geometry_loss(
                        pred_image.clone(),
                        geom,
                        [pin.fx, pin.fy, pin.cx, pin.cy],
                        on(c.surfel_step(c.surfel_normal_start), c.surfel_normal_weight),
                        on(c.surfel_step(c.surfel_dist_start), c.surfel_dist_weight),
                    );
                    let loss = match geo {
                        Some(geo) => loss + geo,
                        None => loss,
                    };
                    // Multi-view consistency against the neighbor render.
                    match (self.multiview.as_ref(), mv_pre) {
                        (Some(mv), Some((view, ngeom))) => {
                            let (mv_geo, mv_ncc) = crate::multiview::consistency_losses(
                                mv,
                                view,
                                self.step_count,
                                img_size,
                                geom_for_mv.clone().expect("surfel geometry"),
                                ngeom,
                                c.surfel_mv_ncc_samples as usize,
                            );
                            let every = c.surfel_mv_every.max(1) as f32;
                            loss + mv_geo * (c.surfel_mv_geo_weight * every) + mv_ncc * (c.surfel_mv_ncc_weight * every)
                        }
                        _ => loss,
                    }
                }
                None => loss,
            };

            // Strip the autodiff graph off the loss so consumers can read the
            // scalar later without keeping the backward pass alive.
            let loss_inner = loss.clone().inner();
            crate::memory::log_phase(&device, self.step_count, "loss", splats.num_splats());
            let mut grads = splats.bwd_validate(loss).await;
            crate::memory::log_phase(&device, self.step_count, "backward", splats.num_splats());

            if weighted
                && motion
                && let Some((mixer, optim)) = self.mixer.take()
            {
                let lr = MIXER_LR + (MIXER_LR_END - MIXER_LR) * progress.min(1.0);
                let mixer_grads = GradientsParams::from_module(&mut grads, &mixer);
                self.mixer = Some(optim_step(optim, lr, mixer, mixer_grads));
            }
            let pose_grad = sub_outs
                .iter()
                .find_map(|o| o.pose_grad_holder.grad_remove(&mut grads))
                .map(Tensor::without_autodiff);
            let refine_grad = pose_ticket
                .is_some()
                .then(|| diff_out.pose_grad_holder.grad_remove(&mut grads))
                .flatten()
                .map(Tensor::without_autodiff);

            // The fused update produces the refine weight and SH gradient
            // itself; the plain backward registers them on the holders.
            let fused_ctx = diff_out.fused_update;
            let unfused_stats = fused_ctx.is_none().then(|| {
                let refine_weight = sub_outs
                    .iter()
                    .map(|o| &o.refine_weight_holder)
                    .chain([&refine_weight_holder])
                    .map(|holder| {
                        holder
                            .grad_remove(&mut grads)
                            .expect("XY gradients need to be calculated.")
                            .without_autodiff()
                    })
                    .reduce(|a, b| a + b)
                    .expect("at least one render");
                // A per-render statistic: with summed renders Adam squares the
                // summed gradient instead.
                let coeffs_grad_sq = if dense_step {
                    None
                } else {
                    diff_out
                        .coeffs_grad_sq_holder
                        .grad_remove(&mut grads)
                        .map(Tensor::without_autodiff)
                };
                (refine_weight, coeffs_grad_sq)
            });

            (
                grads,
                visible,
                opacities,
                diff_out.num_visible,
                loss_inner,
                max_radius,
                compact_from_global,
                fused_ctx,
                unfused_stats,
                pose_grad,
                refine_grad,
            )
        };

        // The optimizer strips autodiff before stepping, so optimizer state
        // (scaling, momentum) lives on the inner device.
        let opt_device = device.clone().inner();
        let optimizer =
            self.optim.get_or_insert_with(|| {
                // Neural rows are stored like SH rows: f16 values, a block-q8
                // first moment and one reduced second moment per row.
                let sh_coeffs = if self.config.neural {
                    AdamState::new(Some(neural::row_lr_scales(&opt_device)), true)
                } else {
                    let sh_degree = splats.sh_degree();
                    let num_coeffs = sh_coeffs_for_degree(sh_degree) as usize;
                    let sh_lr_scales =
                        sh_lr_scales(num_coeffs, self.config.lr_coeffs_sh_scale, &opt_device);
                    AdamState::new(Some(sh_lr_scales), true)
                };

                SplatOptim {
                    adam: AdamScaled::new(1e-15),
                    transforms: AdamState::new(None, false),
                    sh_coeffs,
                    opacities: AdamState::new(None, false),
                    decoder_adam: AdamScaled::new(1e-15).with_beta_2(0.99),
                    decoder: AdamState::new(None, false),
                }
            });

        optimizer.transforms.scaling = Some(lr_scaling.clone());
        let colour_tail = self.config.neural
            && neural::colour_tail(self.step_count, self.config.total_train_iters);

        // Neural appearance never takes the fused update (see the pass choice
        // above), so its rows step from their own gradients.
        let refine_weight = if let Some(fused_ctx) = fused_ctx {
            let mut refine_weight = None;
            splats = trace_span!("Optimizer step").in_scope(|| {
                // Transforms + opacity: projection VJP and Adam in one kernel.
                // It reads the pre-step SH coefficients, so it runs before SH.
                let v_combined = splats
                    .transforms
                    .val()
                    .grad_remove(&mut grads)
                    .expect("transforms gradient");
                let sh_now = splats.sh_coeffs.val().inner();
                // Parameters are taken through `Param::map` so the fused op holds
                // the only reference and updates them in place.
                let min_scale = splats.min_scale_arg();
                let mut fused_out = None;
                let mut new_opacities = None;
                let opacity_param = splats.raw_opacities;
                splats.transforms = splats.transforms.map(|t| {
                    let transforms = t.inner();
                    let mut new_transforms = None;
                    new_opacities = Some(opacity_param.map(|o| {
                        let raw_opac = o.inner();
                        let epsilon = optimizer.adam.epsilon();
                        let (w_t, time) = optimizer.transforms.take_packed(
                            transforms.dims(),
                            &opt_device,
                            epsilon,
                        );
                        let (w_o, time_o) =
                            optimizer.opacities.take_packed(raw_opac.dims(), &opt_device, epsilon);
                        assert_eq!(time, time_o, "transforms and opacity step together");
                        let (beta1, beta2, inv1, inv2, epsilon) = optimizer.adam.coefficients(time);
                        // Opacity regulariser for the first third of training
                        // (10k of 30k), then off so the kept splats re-sharpen.
                        let opac_reg = if self.step_count <= self.config.total_train_iters / 3 {
                            self.config.opacity_reg / raw_opac.dims()[0].max(1) as f32
                        } else {
                            0.0
                        };
                        let out = trace_span!("Transforms step").in_scope(|| {
                            project_update(
                                &fused_ctx,
                                transforms,
                                sh_now,
                                raw_opac,
                                min_scale,
                                compact_from_global.clone(),
                                v_combined,
                                w_t,
                                lr_scaling.reshape([10]),
                                w_o,
                                FusedAdam {
                                    beta1,
                                    beta2,
                                    inv1,
                                    inv2,
                                    epsilon,
                                    lr_opac: (self.config.lr_opac * tail_lr) as f32,
                                    opac_reg,
                                    round_seed: time as u32,
                                },
                            )
                        });
                        optimizer.transforms.put_packed(out.w_t, time, epsilon);
                        optimizer.opacities.put_packed(out.w_o, time, epsilon);
                        new_transforms = Some(out.transforms);
                        fused_out = Some((out.color_dir, out.coeffs_grad_sq, out.v_refine_weight));
                        Tensor::from_inner(out.raw_opac).require_grad()
                    }));
                    Tensor::from_inner(new_transforms.expect("fused step ran")).require_grad()
                });
                splats.raw_opacities = new_opacities.expect("fused step ran");
                let (color_dir, grad_sq, v_refine_weight) = fused_out.expect("fused step ran");
                refine_weight = Some(v_refine_weight);
                splats.sh_coeffs = trace_span!("SH Coeffs step").in_scope(|| {
                    splats.sh_coeffs.map(|t| {
                        let stepped = optimizer.adam.step_sh_color(
                            self.config.lr_coeffs_dc * tail_lr,
                            t.inner(),
                            &color_dir,
                            compact_from_global,
                            grad_sq,
                            &mut optimizer.sh_coeffs,
                        );
                        Tensor::from_inner(stepped).require_grad()
                    })
                });
                splats
            });
            refine_weight.expect("set by the fused step")
        } else {
            let (refine_weight, coeffs_grad_sq) = unfused_stats.expect("plain backward stats");
            // Summed renders registered dense gradients.
            let grad_rows = (!dense_step).then(|| compact_from_global.clone());
            splats = trace_span!("Optimizer step").in_scope(|| {
                if !colour_tail {
                    splats.transforms = trace_span!("Transforms step").in_scope(|| {
                        step_param(
                            &optimizer.adam,
                            1.0,
                            splats.transforms,
                            &mut optimizer.transforms,
                            &mut grads,
                            grad_rows.clone(),
                            None,
                        )
                    });
                }
                let lr_coeffs = if self.config.neural {
                    let total = self.config.total_train_iters;
                    neural::base_lr() * neural::row_lr_factor(self.step_count, total)
                } else {
                    self.config.lr_coeffs_dc * tail_lr
                };
                splats.sh_coeffs = trace_span!("SH Coeffs step").in_scope(|| {
                    step_param(
                        &optimizer.adam,
                        lr_coeffs,
                        splats.sh_coeffs,
                        &mut optimizer.sh_coeffs,
                        &mut grads,
                        // The renderer registers gradients in compact rows.
                        grad_rows.clone(),
                        coeffs_grad_sq,
                    )
                });
                // As the reference, the decoder sits out the band-0 phase: its last
                // layer stays zero, so the residual and the latent gradients do too.
                if splats.neural_degree > 0
                    && let Some(decoder) = splats.decoder.take()
                {
                    let lr = neural::decoder_lr(self.step_count, self.config.total_train_iters);
                    splats.decoder = Some(trace_span!("Decoder step").in_scope(|| {
                        step_param(
                            &optimizer.decoder_adam,
                            lr,
                            decoder,
                            &mut optimizer.decoder,
                            &mut grads,
                            // Already dense: summed over the visible splats.
                            None,
                            None,
                        )
                    }));
                }
                if !colour_tail {
                    splats.raw_opacities = trace_span!("Opacity step").in_scope(|| {
                        step_param(
                            &optimizer.adam,
                            self.config.lr_opac * tail_lr,
                            splats.raw_opacities,
                            &mut optimizer.opacities,
                            &mut grads,
                            grad_rows,
                            None,
                        )
                    });
                }
                splats
            });
            refine_weight
        };

        if let Some(residuals) = qat_residuals {
            splats = crate::qat::restore(splats, residuals);
        }
        crate::memory::log_phase(&device, self.step_count, "optimizer", splats.num_splats());
        trace_span!("Housekeeping").in_scope(|| {
            // Refine state accumulates on the inner (non-autodiff) device.
            // Kept after the optimizer so it doesn't sit between the
            // backward's gradient gathers and the step that consumes them.
            let device = splats.device().inner();
            let record = self
                .refine_record
                .take()
                .unwrap_or_else(|| RefineRecord::new(splats.num_splats(), &device));
            self.refine_record =
                Some(record.gather_stats(refine_weight, visible.clone(), max_radius));
        });

        // Noise on low-opacity visible splats' means (one kernel, hash-based
        // normals), using the forward's floored opacities and no autodiff graph.
        let seed = self.rng.random::<u32>();
        let scale = self.config.mean_noise_weight * lr_unit as f32 * median_scale;
        if !colour_tail {
            splats.transforms = splats.transforms.map(|t| {
                let out = crate::housekeep::noise_means(
                    t.inner(),
                    opacities,
                    visible,
                    scale,
                    median_scale,
                    seed,
                );
                Tensor::from_inner(out).require_grad()
            });
        }
        if let Some(frozen) = &self.frozen {
            splats = restore_frozen(splats, frozen);
        }

        // Apply pose gradients from `POSE_READ_DELAY` steps ago: that backward
        // has long run, so the readback doesn't wait on queued work.
        let pending = pose_grad.map(|grad| PendingPose {
            view: batch.view_index,
            sub_pose: sampled,
            local: sampled_local,
            grad,
            lr: trajectory_lr as f32,
        });
        self.pending_poses.push_back(pending);
        if self.pending_poses.len() > POSE_READ_DELAY
            && let Some(Some(prev)) = self.pending_poses.pop_front()
        {
            self.apply_pose(prev).await;
        }
        if self.pose_refine.is_some() {
            self.pending_refine.push_back(pose_ticket.zip(refine_grad));
            // Read the oldest POSE_REFINE_READ_BATCH steps' gradients in one readback once
            // POSE_READ_DELAY newer steps are queued, rather than one readback (and queue
            // flush) per step.
            if self.pending_refine.len() >= POSE_REFINE_READ_BATCH + POSE_READ_DELAY {
                let (tickets, grads): (Vec<_>, Vec<_>) =
                    self.pending_refine.drain(..POSE_REFINE_READ_BATCH).flatten().unzip();
                if !grads.is_empty() {
                    let g: Vec<f32> = Tensor::cat(grads, 0)
                        .into_data_async()
                        .await
                        .expect("Failed to read camera gradients")
                        .try_to_vec()
                        .expect("Camera gradients are f32");
                    if let Some(p) = self.pose_refine.as_mut() {
                        for (ticket, g) in tickets.into_iter().zip(g.chunks(12)) {
                            p.apply(ticket, g);
                        }
                    }
                }
            }
        }

        if self.step_count == BLUR_TEST.end && self.blurred_capture.is_none() {
            self.blurred_capture = Some(self.blur_verdict());
        }

        let stats = TrainStepStats {
            num_visible,
            lr_mean,
            lr_rotation: self.config.lr_rotation,
            lr_scale: self.config.lr_scale,
            lr_coeffs: self.config.lr_coeffs_dc,
            lr_opac: self.config.lr_opac,
            loss: loss_inner,
        };

        if motion && self.blurred_capture == Some(true) {
            self.blur_full_spent += step_start.elapsed();
        }
        crate::memory::log_phase(&device, self.step_count, "step_end", splats.num_splats());
        (splats, stats)
    }

    /// Whether the blend fit the photos better than their sharp renders over
    /// `BLUR_TEST` (median relative loss reduction above `BLUR_MIN_GAIN`).
    fn blur_verdict(&mut self) -> bool {
        let samples = std::mem::take(&mut self.blur_test);
        if samples.is_empty() {
            return false;
        }
        let median = |mut values: Vec<f32>| {
            values.sort_by(f32::total_cmp);
            values[values.len() / 2]
        };
        let gain = median(
            samples
                .iter()
                .map(|s| s.reduction / s.positive.max(f32::EPSILON))
                .collect(),
        );
        let blurred = gain > BLUR_MIN_GAIN;
        if blurred {
            eprintln!("{BLUR_MODE_NOTE}");
        }
        eprintln!(
            "camera-motion deblur {}: median blend-over-sharp loss reduction {gain:.4} \
             (reduction {:.5}, positive loss {:.5}, signed loss {:.5}, old |signed| ratio {:.4})",
            if blurred { "on" } else { "off (sharp capture)" },
            median(samples.iter().map(|s| s.reduction).collect()),
            median(samples.iter().map(|s| s.positive).collect()),
            median(samples.iter().map(|s| s.sharp).collect()),
            median(
                samples
                    .iter()
                    .map(|s| s.reduction / s.sharp.abs().max(f32::EPSILON))
                    .collect()
            ),
        );
        blurred
    }

    async fn apply_pose(&mut self, pending: PendingPose) {
        let g: Vec<f32> = pending
            .grad
            .into_data_async()
            .await
            .expect("Failed to read pose gradients")
            .try_to_vec()
            .expect("Pose gradients are f32");
        // The kernel's gradient is w.r.t. `E` in `(I + E) M` for the render's
        // camera-space map `M = T^-1`; then dL/dM = G M^-T and, through the
        // inverse, dL/dT = -M^T G.
        let g_e = glam::Mat4::from_cols_array_2d(&[
            [g[0], g[3], g[6], 0.0],
            [g[1], g[4], g[7], 0.0],
            [g[2], g[5], g[8], 0.0],
            [g[9], g[10], g[11], 0.0],
        ]);
        let d_t = -(pending.local.transpose() * g_e);
        let rows = d_t.transpose().to_cols_array();
        let grad_t: [f32; 12] = std::array::from_fn(|i| rows[i]);
        if let Some(trajectories) = self.trajectories.as_mut() {
            trajectories.step(
                pending.view,
                pending.sub_pose,
                grad_t,
                ORTHOGONALITY_WEIGHT,
                pending.lr,
            );
        }
    }

    /// Release state that only a future training step would use. Call only
    /// after the final step; appearance and scene bounds remain available
    /// for evaluation and reporting.
    pub fn finish_training(&mut self) {
        self.optim = None;
        self.refine_record = None;
    }

    /// Whether the capture was found motion-blurred and trains with the
    /// exposure model.
    pub fn blur_mode(&self) -> bool {
        self.blurred_capture == Some(true)
    }

    /// Learned exposure trajectories, carried into a fresh trainer (LOD phases).
    pub fn take_trajectories(&mut self) -> Option<Trajectories> {
        self.trajectories.take()
    }

    pub fn set_trajectories(&mut self, trajectories: Option<Trajectories>) {
        self.trajectories = trajectories;
    }

    pub async fn refine(&mut self, iter: u32, splats: Splats) -> (Splats, RefineStats) {
        if self.config.neural && neural::colour_tail(iter, self.config.total_train_iters) {
            let total_splats = splats.num_splats();
            let stats = RefineStats {
                num_added: 0,
                num_split_oversized: 0,
                num_split_high_grad: 0,
                num_pruned: 0,
                num_pruned_non_finite: 0,
                num_relocated: 0,
                total_splats,
            };
            return (splats, stats);
        }
        crate::memory::log_phase(&splats.device(), iter, "refine_start", splats.num_splats());
        // Return the finished step's free pages, sized for the old splat count,
        // before refine allocates new-count buffers; sync first so they really go.
        #[cfg(not(target_family = "wasm"))]
        let _ = splats.device().sync();
        #[cfg(target_family = "wasm")]
        wasm_gpu_wait(&splats).await;
        splats.device().memory_cleanup();
        let progress = iter as f32 / self.config.total_train_iters.max(1) as f32;
        // Refine manipulates the canonical (un-floored) params, so bake the
        // current 3D-filter floor into them first — split/clone/prune then see
        // the splat's true scales with no double-apply. A freshly recomputed
        // floor is attached at the end (below), once positions/count are known.
        let mut splats = splats.bake_min_scale();
        let device = splats.device();

        let refiner = self
            .refine_record
            .take()
            .expect("Can only refine if refine stats are initialized");

        let max_allowed_bounds = self.bounds.extent.max_element() * 100.0;

        // If not refining, update splat to step with gradients applied.
        // Prune dead splats. This ALWAYS happen even if we're not "refining" anymore.
        let optim = self
            .optim
            .take()
            .expect("Can only refine after optimizer is initialized");
        let alpha_mask = splats.opacities().lower_elem(MIN_OPACITY);
        let scales = splats.scales();

        // Note: we do NOT cull on a minimum scale. A genuinely flat splat
        // (a thin "pancake" representing a surface) legitimately has a tiny
        // smallest axis, so there's no correct min-scale threshold — the
        // non-finite check below still removes actually-degenerate splats.
        let scale_big = scales
            .clone()
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Remove splats that are way out of bounds.
        let center = self.bounds.center;
        let bound_center =
            Tensor::<1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        let splat_dists = (splats.means() - bound_center).abs();
        let bound_mask = splat_dists
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Prune parameter that's NaN. One thread per row: an elementwise
        // [N, width] boolean mask here was refine's live-memory peak at the
        // cap (2 x 192 B/splat for SH).
        fn row_non_finite(t: &Tensor<2>) -> Tensor<1, Bool> {
            crate::quant::rows_non_finite(t.clone())
        }
        let transforms_bad = row_non_finite(&splats.transforms.val());
        let sh_bad = row_non_finite(&splats.sh_coeffs.val().flatten(1, 2));
        let opac_bad = row_non_finite(&splats.raw_opacities.val().unsqueeze_dim(1));
        let non_finite_mask = transforms_bad.bool_or(sh_bad).bool_or(opac_bad);
        let prune_mask = alpha_mask
            .bool_or(scale_big)
            .bool_or(bound_mask)
            .bool_or(non_finite_mask.clone());

        // Every host decision below comes from ONE readback (a single GPU sync)
        // of per-splat columns: prune, non-finite, resample weight
        // (opacity × visible), oversized, growth weight (refine weight ×
        // visible), and the means for the scene bounds.
        let n = splats.num_splats() as usize;
        let vis_f = refiner.vis_mask().float();
        let oversized = if self.config.split_at_screen_size > 0.0 {
            refiner
                .above_screen_size(self.config.split_at_screen_size)
                .float()
        } else {
            Tensor::zeros([n], &device)
        };
        let table = Tensor::cat(
            vec![
                prune_mask.float().unsqueeze_dim::<2>(1),
                non_finite_mask.float().unsqueeze_dim(1),
                (splats.opacities() * vis_f.clone()).unsqueeze_dim(1),
                oversized.unsqueeze_dim(1),
                (refiner.refine_weight_norm.clone() * vis_f).unsqueeze_dim(1),
                splats.means(),
            ],
            1,
        );
        const COLS: usize = 8;
        let host: Vec<f32> = table
            .into_data_async()
            .await
            .expect("Failed to read refine table")
            .try_into_vec()
            .expect("Failed to read refine table");
        let rows = || host.chunks_exact(COLS);
        if self.far_share.is_none() && self.config.max_splats == 0 {
            let far = self.measure_far_share(rows().map(|r| glam::Vec3::new(r[5], r[6], r[7])));
            self.far_share = Some(far);
            eprintln!(
                "Splat budget {} ({:.0}% of the scene far outside the cameras)",
                self.auto_budget(),
                far * 100.0
            );
        }
        // Pruned rows stay where they are as free slots: the children below
        // are written into them, so refine never gathers or remaps the
        // per-splat state. If everything would be pruned, prune nothing.
        let mut dead: Vec<i32> = rows()
            .enumerate()
            .filter(|(_, r)| r[0] != 0.0)
            .map(|(i, _)| i as i32)
            .collect();
        if dead.len() == n {
            log::warn!("Trying to create empty splat!");
            dead.clear();
        }
        let alive = (n - dead.len()) as u32;
        let is_dead = |r: &[f32]| r[0] != 0.0 && !dead.is_empty();
        let num_pruned_non_finite = rows().filter(|r| r[1] != 0.0).count() as u32;
        if num_pruned_non_finite > 0 && !self.depth_prior_off && self.config.depth_prior_weight > 0.0 {
            self.depth_prior_off = true;
            let line = format!(
                "depth prior: {num_pruned_non_finite} non-finite splats at step {}; depth prior off for the rest of training",
                self.step_count
            );
            log::warn!("{line}");
            eprintln!("{line}");
        }
        // Bounds from the surviving splats' means (children of splits land
        // next to their parents, so this matches the post-refine scene).
        let kept_means: Vec<f32> = rows()
            .filter(|r| !is_dead(r))
            .flat_map(|r| r[5..8].iter().copied())
            .collect();
        let new_bounds = bounds_from_pos(BOUND_PERCENTILE, &kept_means);
        // Column `c` per splat, zero for pruned rows so they never parent.
        let alive_col = |c: usize| -> Vec<f32> {
            rows()
                .map(|r| if is_dead(r) { 0.0 } else { r[c] })
                .collect()
        };

        let pruned_count = dead.len() as u32;
        let mut split_inds = HashSet::new();

        // Always replace dead gaussians, so that the pruned budget is reused.
        if pruned_count > 0 {
            // Replacement weighting: opacity × visibility.
            let resampled_inds = multinomial_sample(&mut self.rng, &alive_col(2), pruned_count);
            split_inds.extend(resampled_inds);
        }

        // Force-split splats that are too big on screen (every refine). Rather
        // than killing them (the old `kill_at_screen_size`), we split them and
        // shrink the children down to `split_at_screen_size` on screen — see
        // `refine_splats`. Capped by the remaining splat budget.
        let pre_oversized = split_inds.len();
        if self.config.split_at_screen_size > 0.0 {
            let n0 = self.growth_start_count.unwrap_or(alive);
            let mut budget = self
                .splat_cap(n0)
                .saturating_sub(alive + split_inds.len() as u32);
            for (ind, flag) in alive_col(3).into_iter().enumerate() {
                if budget == 0 {
                    break;
                }
                if flag != 0.0 && split_inds.insert(ind as i32) {
                    budget -= 1;
                }
            }
        }
        let num_split_oversized = (split_inds.len() - pre_oversized) as u32;

        let pre_high_grad = split_inds.len();
        if iter >= self.config.growth_start_iter && iter < self.config.growth_stop_iter {
            // Budget curve: grow exponentially from the count at growth start
            // to the cap at `GROWTH_FULL_AT` of training (or growth stop), so
            // early steps run on few splats and the final count is set up
            // front. Replacements of dead splats (above) keep the count; this
            // adds on top of them, sampling splits by refine weight.
            let before = splats.num_splats();
            let n0 = *self.growth_start_count.get_or_insert(before) as f64;
            let cap = self.splat_cap(n0 as u32) as f64;
            let start = self.config.growth_start_iter as f64;
            let full = (GROWTH_FULL_AT * self.config.total_train_iters as f64)
                .min(self.config.growth_stop_iter as f64)
                .max(start + 1.0);
            let t = ((iter as f64 - start) / (full - start)).clamp(0.0, 1.0);
            let target =
                ((n0 * (cap / n0.max(1.0)).max(1.0).powf(t)).ceil() as u32).min(cap as u32);
            let cur_splats = alive + split_inds.len() as u32;
            let grow_count = target.saturating_sub(cur_splats);
            if grow_count > 0 {
                let weights = alive_col(4);
                let positive = weights
                    .iter()
                    .filter(|w| w.is_finite() && **w > 0.0)
                    .count() as u32;
                let growth_inds =
                    multinomial_sample(&mut self.rng, &weights, grow_count.min(positive));
                split_inds.extend(growth_inds);
            }
        }

        let num_split_high_grad = (split_inds.len() - pre_high_grad) as u32;
        let refine_count = split_inds.len();
        // Per-splat max on-screen extent, used by `refine_splats` to cap the
        // split shrink so oversized splats' children land at `split_at_screen_size`.
        let screen_sizes = refiner.max_screen_size;
        // The readback synced the GPU, so the step's transient buffers are
        // free: release them before refine allocates, so the growth below
        // doesn't stack on a pool still holding a whole step's working set.
        device.memory_cleanup();
        let parents: Vec<i32> = split_inds.into_iter().collect();
        splats = self.refine_splats(&device, optim, splats, parents, dead, screen_sizes, iter);

        crate::memory::log_phase(&device, iter, "refine_remap", splats.num_splats());
        // Periodically order the splats along a Morton (z-order) curve of
        // their means, so splats close in space are close in memory (better
        // cache locality in projection and rasterization).
        if iter.is_multiple_of(MORTON_EVERY) {
            splats = self.morton_sort(splats, &new_bounds).await;
            crate::memory::log_phase(&device, iter, "morton", splats.num_splats());
        }
        self.bounds = new_bounds;
        crate::memory::log_phase(&device, iter, "refine_bounds", splats.num_splats());
        // Wait for the GPU before returning free pages: a page freed while
        // in-flight work still uses it stays resident until that work ends,
        // so the next step's fresh pages would stack on top of it.
        #[cfg(not(target_family = "wasm"))]
        let _ = device.sync();
        #[cfg(target_family = "wasm")]
        wasm_gpu_wait(&splats).await;
        device.memory_cleanup();
        crate::memory::log_phase(&device, iter, "refine_cleanup", splats.num_splats());

        // Recompute the per-splat 3D-filter floor against the new positions/
        // count and attach it — the floor is part of the splat from here until
        // the next refine. Past the freeze fraction we stop refreshing and leave
        // it baked in, so the tail settles against fixed params.
        // The filter only pays off on low-resolution captures (measured: it
        // costs LPIPS and PSNR on standard-resolution scenes, helps PSNR on
        // ~0.2 MP On-the-go images), so it applies only below
        // MIP_3D_FILTER_MAX_PIXELS per training image. Neural appearance
        // never uses it: the floor halves the neural gain over SH (kitchen).
        let pixels_per_view = self.train_pixels / self.view_cams.len().max(1) as u64;
        let small_images = self.train_pixels == 0 || pixels_per_view <= MIP_3D_FILTER_MAX_PIXELS;
        // Surfels never take the floor (its opacity compensation spans all three
        // axes), nor does neural appearance.
        if progress < MIN_SCALE_FREEZE_FRAC && small_images && !splats.surfel && !self.config.neural {
            // `splats` is already on the inner backend here, so `means()` is too.
            // No-op when there are no view cameras (e.g. unit tests).
            let means = splats.means();
            if let Some(f) =
                compute_min_scale(&means, &self.view_cams, self.config.min_scale_factor)
            {
                splats = splats.with_min_scale(f);
            }
        }

        let splat_count = splats.num_splats();

        (
            splats,
            RefineStats {
                num_added: refine_count as u32,
                num_split_oversized,
                num_split_high_grad,
                num_pruned: pruned_count,
                num_pruned_non_finite,
                // refine_splats refills dead slots first, from any child.
                num_relocated: pruned_count.min(refine_count as u32),
                total_splats: splat_count,
            },
        )
    }

    /// Permute every per-splat tensor (params and optimizer moments) into
    /// Morton order of the means within `bounds`.
    async fn morton_sort(&mut self, mut splats: Splats, bounds: &BoundingBox) -> Splats {
        let means: Vec<f32> = splats
            .means()
            .into_data_async()
            .await
            .expect("Failed to read means")
            .try_into_vec()
            .expect("Failed to read means");
        let order = morton_order(&means, bounds);
        let device = splats.device();
        let perm: Tensor<1, Int> =
            Tensor::from_data(TensorData::new(order, [means.len() / 3]), &device);
        // One tensor at a time, and each source's pages go back to the device
        // before the next copy: the copies differ in size, so a source page
        // left in the pool is never reused and every copy would stack on the
        // live set (+215 MiB at garden 1M, the run's peak). This way the
        // re-sort costs at most its largest tensor on top of the live set.
        splats.transforms = splats.transforms.map(|t| t.select(0, perm.clone()));
        release_pages(&splats).await;
        splats.sh_coeffs = splats.sh_coeffs.map(|t| t.select(0, perm.clone()));
        release_pages(&splats).await;
        splats.raw_opacities = splats.raw_opacities.map(|t| t.select(0, perm.clone()));
        splats.min_scale = splats.min_scale.map(|t| t.select(0, perm.clone()));
        release_pages(&splats).await;
        if let Some(optim) = self.optim.as_mut() {
            optim.transforms.permute_rows(&perm);
            release_pages(&splats).await;
            optim.sh_coeffs.permute_rows(&perm);
            release_pages(&splats).await;
            optim.opacities.permute_rows(&perm);
        }
        splats
    }

    fn refine_splats(
        &mut self,
        device: &Device,
        mut optim: SplatOptim,
        mut splats: Splats,
        parents: Vec<i32>,
        dead: Vec<i32>,
        screen_sizes: Tensor<1>,
        iter: u32,
    ) -> Splats {
        let refine_count = parents.len();
        // Every dead slot is refilled by a child: one replacement was sampled
        // per pruned splat, and growth only adds on top of that. A NaN-heavy
        // state can leave fewer distinct parents than dead rows; the surplus
        // slots stay in place as empty, invisible splats (pruned again later).
        let mut dead = dead;
        let empty = if dead.len() > refine_count { dead.split_off(refine_count) } else { Vec::new() };
        let filled = dead.len();

        if refine_count > 0 {
            // Rows written in place: the parents, then the dead slots taking
            // the first `filled` children. The remaining children are appended.
            let mut slots = parents.clone();
            slots.extend(&dead);
            let ints = |v: Vec<i32>| {
                let len = v.len();
                Tensor::<1, Int>::from_data(TensorData::new(v, [len]), device)
            };
            let refine_inds = ints(parents);
            let slot_inds = ints(slots);
            let dead_inds = (filled > 0).then(|| ints(dead));

            let cur_transforms = splats.transforms.val().select(0, refine_inds.clone());
            let cur_means = cur_transforms.clone().slice(s![.., 0..3]);
            let cur_rots_raw = cur_transforms.clone().slice(s![.., 3..7]);
            let magnitudes = Tensor::clamp_min(
                Tensor::sum_dim(cur_rots_raw.clone().powi_scalar(2), 1).sqrt(),
                1e-32,
            );
            let cur_rots = cur_rots_raw.clone() / magnitudes;
            let cur_log_scale = cur_transforms.slice(s![.., 7..10]);
            let cur_sh_coeffs = splats.sh_coeffs.val().select(0, refine_inds.clone());
            let cur_raw_opac = splats.raw_opacities.val().select(0, refine_inds.clone());

            let cur_scales = cur_log_scale.clone().exp();

            let cur_opac = sigmoid(cur_raw_opac);
            let inv_opac: Tensor<1> = 1.0 - cur_opac;
            // Post-split child opacity as a power law in transmittance,
            // p = 0.5 would keep the transmittance for cloning splats but as we offset them
            // choose a higher p.
            let new_opac: Tensor<1> = 1.0 - inv_opac.powf_scalar(FRAC_1_SQRT_2);
            let new_raw_opac = inv_sigmoid(new_opac.clamp(MIN_OPACITY, 1.0 - MIN_OPACITY));

            // Smooth covariance-aware split. Per-axis shrink + mass-conserving
            // deterministic offset (one child at +offset, the other at -offset).
            // Children inherit the
            // parent's rotation; the split is the scale shrink + ±offset.
            let cur_scales_sq = cur_scales.clone().powi_scalar(2);
            // Surfels split only within their tangent plane: masking the
            // normal axis (index 2) out gives it ratio 0, i.e. no shrink and
            // no offset.
            let cur_scales_sq = if splats.surfel {
                cur_scales_sq * Tensor::<1>::from_floats([1.0, 1.0, 0.0], device).reshape([1, 3])
            } else {
                cur_scales_sq
            };
            let max_scale_sq = cur_scales_sq.clone().max_dim(1).clamp_min(1e-30);
            let ratio = cur_scales_sq / max_scale_sq;
            // Max-axis shrink factor `k` (per splat). The standard split uses
            // 1/√2 (mass-conserving). When `split_at_screen_size` is set, splats
            // that are too big on screen shrink harder so their children land at
            // (at most) the cap: `k = min(1/√2, split_at_screen_size / screen)`.
            // Splats already within √2× of the cap are unaffected (min → 1/√2).
            let k_per_axis: Tensor<2> = if self.config.split_at_screen_size > 0.0 {
                let k_max = screen_sizes
                    .select(0, refine_inds.clone())
                    .unsqueeze_dim(1)
                    .clamp_min(1e-6)
                    .recip()
                    .mul_scalar(self.config.split_at_screen_size)
                    .clamp_max(FRAC_1_SQRT_2);
                -(ratio * (-k_max + 1.0)) + 1.0
            } else {
                -(ratio * (1.0_f32 - FRAC_1_SQRT_2)) + 1.0
            };
            let offset_factor = (-k_per_axis.clone().powi_scalar(2) + 1.0)
                .clamp_min(0.0)
                .sqrt();
            let offset_local = offset_factor * cur_scales;
            let samples = quaternion_vec_multiply(cur_rots.clone(), offset_local);
            let new_log_scales = cur_log_scale + k_per_axis.log();
            // Keep a surfel's ignored normal log-scale at its smaller tangent one.
            let new_log_scales = if splats.surfel {
                let tangent_log = new_log_scales.slice(s![.., 0..2]);
                let normal_log = tangent_log.clone().min_dim(1);
                Tensor::cat(vec![tangent_log, normal_log], 1)
            } else {
                new_log_scales
            };
            let child_rots = cur_rots;

            let parent_transforms = Tensor::cat(
                vec![
                    cur_means.clone() - samples.clone(),
                    cur_rots_raw,
                    new_log_scales.clone(),
                ],
                1,
            );
            // Child sits at parent_mean + samples (parent moves to
            // parent_mean - samples) — anti-correlated, centroid-preserving.
            // Build new transforms row: means(3) + rotations(4) + log_scales(3)
            let child_transforms =
                Tensor::cat(vec![cur_means + samples, child_rots, new_log_scales], 1);

            // Parents then children, in `slots` order plus the appended tail.
            splats.transforms = splats.transforms.map(|t| {
                write_rows(
                    t,
                    Some(slot_inds.clone()),
                    Tensor::cat(vec![parent_transforms, child_transforms], 0),
                )
            });
            splats.raw_opacities = splats.raw_opacities.map(|t| {
                write_rows(
                    t,
                    Some(slot_inds.clone()),
                    Tensor::cat(vec![new_raw_opac.clone(), new_raw_opac], 0),
                )
            });
            // The row writes can copy their whole target (39.7 MB transforms,
            // 95.4 MB SH at 1M). Return the old transforms page before the SH
            // write so the two copies never coexist.
            #[cfg(not(target_family = "wasm"))]
            {
                let _ = device.sync();
                device.memory_cleanup();
            }
            splats.sh_coeffs = splats
                .sh_coeffs
                .map(|t| write_rows(t, dead_inds, cur_sh_coeffs));

            // Optimizer state lives on the inner (non-autodiff) device.
            let opt_inds = slot_inds.to_device(&device.clone().inner());
            let extra = refine_count - filled;
            optim.transforms.reset_rows(&opt_inds, extra);
            optim.sh_coeffs.reset_rows(&opt_inds, extra);
            optim.opacities.reset_rows(&opt_inds, extra);
        }

        if !empty.is_empty() {
            log::warn!("Refine: {} dead slots without a replacement left empty", empty.len());
            let k = empty.len();
            let rows = Tensor::<1, Int>::from_data(TensorData::new(empty, [k]), device);
            // Finite, invisible rows: origin, identity rotation, tiny scale,
            // zero colour, opacity ~2e-9.
            let row = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -10.0, -10.0, -10.0];
            let transforms: Vec<f32> = row.iter().copied().cycle().take(k * row.len()).collect();
            let transforms = Tensor::<2>::from_data(TensorData::new(transforms, [k, row.len()]), device);
            splats.transforms = splats
                .transforms
                .map(|t| write_rows(t, Some(rows.clone()), transforms));
            splats.raw_opacities = splats
                .raw_opacities
                .map(|t| write_rows(t, Some(rows.clone()), Tensor::full([k], -20.0, device)));
            splats.sh_coeffs = splats.sh_coeffs.map(|t| {
                let [_, coeffs, channels] = t.dims();
                let zeros = Tensor::<3>::zeros([k, coeffs, channels], device).cast(t.dtype());
                write_rows(t, Some(rows.clone()), zeros)
            });
            let opt_rows = rows.to_device(&device.clone().inner());
            optim.transforms.reset_rows(&opt_rows, 0);
            optim.sh_coeffs.reset_rows(&opt_rows, 0);
            optim.opacities.reset_rows(&opt_rows, 0);
        }

        let train_t = (iter as f32 / self.config.total_train_iters.max(1) as f32).clamp(0.0, 1.0);
        let t_shrink_strength = 1.0 - train_t;
        let minus_opac = self.config.opac_decay * t_shrink_strength;

        // Lower opacity slowly over time.
        splats.raw_opacities = splats.raw_opacities.map(|f| {
            let new_opac = sigmoid(f) - minus_opac;
            inv_sigmoid(new_opac.clamp(1e-12, 1.0 - 1e-12))
        });

        self.optim = Some(optim);
        splats
    }
}

/// Write `values` rows into `t`: the first `rows.len()` in place at `rows`
/// (unique), the rest appended.
fn write_rows<const D: usize>(
    t: Tensor<D>,
    rows: Option<Tensor<1, Int>>,
    values: Tensor<D>,
) -> Tensor<D> {
    let total = values.dims()[0];
    let n = rows.as_ref().map_or(0, |r| r.dims()[0]);
    let t = match rows {
        Some(rows) => crate::quant::assign_rows(t, rows, values.clone().narrow(0, 0, n)),
        None => t,
    };
    if total > n {
        Tensor::cat(vec![t, values.narrow(0, n, total - n)], 0)
    } else {
        t
    }
}

/// Wait for the GPU, then return the pool's free pages to the device.
async fn release_pages(splats: &Splats) {
    #[cfg(not(target_family = "wasm"))]
    let _ = splats.device().sync();
    #[cfg(target_family = "wasm")]
    wasm_gpu_wait(splats).await;
    splats.device().memory_cleanup();
}

/// Steps between Morton re-sorts of the splats (at the refine on that step).
const MORTON_EVERY: u32 = 1000;

/// Row order of `means` (`[x, y, z]` per splat) by the 30-bit Morton code of
/// each mean within `bounds`.
fn morton_order(means: &[f32], bounds: &BoundingBox) -> Vec<i32> {
    fn spread(v: u32) -> u32 {
        // Insert two zero bits between each of the low 10 bits.
        let mut x = v & 0x3ff;
        x = (x | (x << 16)) & 0x0300_00ff;
        x = (x | (x << 8)) & 0x0300_f00f;
        x = (x | (x << 4)) & 0x030c_30c3;
        (x | (x << 2)) & 0x0924_9249
    }
    let lo = bounds.center - bounds.extent;
    let size = (bounds.extent * 2.0).max(glam::Vec3::splat(1e-6));
    let mut keyed: Vec<(u32, i32)> = means
        .chunks_exact(3)
        .enumerate()
        .map(|(i, m)| {
            let q = ((glam::Vec3::new(m[0], m[1], m[2]) - lo) / size)
                .clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
                * 1023.0;
            let q = q.as_uvec3();
            (spread(q.x) | (spread(q.y) << 1) | (spread(q.z) << 2), i as i32)
        })
        .collect();
    keyed.sort_unstable_by_key(|&(k, _)| k);
    keyed.into_iter().map(|(_, i)| i).collect()
}

/// Sample a background color: base + uniform noise in [-strength, +strength], clamped to [0, 1].
fn sample_background_color<R: rand::Rng + ?Sized>(
    base: glam::Vec3,
    strength: f32,
    rng: &mut R,
) -> glam::Vec3 {
    if strength <= 0.0 {
        return base.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
    }
    use rand::RngExt as _;
    let noise = glam::Vec3::new(
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
    );
    (base + noise).clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}

/// splat patch (wasm only): `Device::sync` blocks, which the browser's event loop can't. Await a
/// one-element read-back instead; with the single CubeCL stream it waits for the same work.
#[cfg(target_family = "wasm")]
async fn wasm_gpu_wait(splats: &Splats) {
    let _ = splats.opacities().slice([0..1]).into_data_async().await;
}

/// Blend a photo's renders in linear radiance (display RGB raised to
/// [`GAMMA`]): `renders` are the half-resolution exposure sub-renders followed
/// by the full-resolution render at the photo's own camera. With a [`Mixer`]
/// the result is its per-pixel blend at half resolution, upsampled (and the
/// mean mask, for the loss); without, the plain average. Other channels
/// (alpha) average over the sub-renders. The blend, upsampling and their
/// backward are fused kernels ([`crate::blur_kernels`]).
fn blend(
    mut renders: Vec<Tensor<3>>,
    mixer: Option<&Mixer>,
) -> (Tensor<3>, Option<Tensor<1>>, BlurCache) {
    let sharp = renders.pop().expect("the sharp render");
    let count = renders.len();
    // [K, H/2, W/2, C], the renders' own layout.
    let subs = Tensor::stack::<4>(renders, 0);
    let [_, sh, sw, _] = subs.dims();
    // Both blends are `(1 - m) * sharp + m * blurred`; the plain average has
    // `blurred` the sub-render mean and `m = K / (K + 1)`.
    let (weights, mask) = if let Some(mixer) = mixer {
        let (weights, mask) = mixer.forward(blur_kernels::linear_subs(subs.clone().detach()));
        (Some(weights), mask)
    } else {
        let m = count as f32 / (count + 1) as f32;
        (None, Tensor::full([1, 1, sh, sw], m, &sharp.device()))
    };
    let mask_mean = mixer.is_some().then(|| mask.clone().mean());
    // Both are smooth: f16 halves a cache that grows with the photo count.
    let cache_mask = mask.clone().detach().cast(burn::tensor::FloatDType::F16);
    let (rgb, half) = blur_kernels::blend(sharp, subs, weights, mask);
    let blurred = half.slice(s![.., 0..3]).cast(burn::tensor::FloatDType::F16);
    (rgb, mask_mean, (blurred, cache_mask))
}

/// A photo's linear blur image and mask at half resolution, detached, in f16.
type BlurCache = (Tensor<4>, Tensor<4>);

/// `(1 - m) * sharp + m * blurred` in linear radiance with the photo's cached
/// blur image and mask; only the fresh sharp render carries gradients.
fn blend_cached(sharp: Tensor<3>, blurred: Tensor<4>, mask: Tensor<4>) -> Tensor<3> {
    let [h, _, _] = sharp.dims();
    // A cache from a lower training resolution doubles until it reaches this
    // photo's half resolution; the blend upsamples it to full.
    let grow = |t: Tensor<4>| {
        let mut t = t.cast(burn::tensor::FloatDType::F32);
        while t.dims()[2] * 2 < h.div_ceil(2) * 2 && t.dims()[2] * 2 <= h {
            let [_, _, th, tw] = t.dims();
            t = blur_kernels::upsample2(t, th * 2, tw * 2);
        }
        t
    };
    blur_kernels::blend_half(sharp, grow(blurred), grow(mask))
}

fn optim_step(
    mut optim: burn::optim::ModuleOptimizer,
    lr: f64,
    mixer: Mixer,
    grads: GradientsParams,
) -> (Mixer, burn::optim::ModuleOptimizer) {
    let mixer = optim.step(lr, mixer, grads);
    (mixer, optim)
}

/// `img` in the forward, with its gradient scaled by `scale`.
fn scale_grad(img: Tensor<3>, scale: f32) -> Tensor<3> {
    blur_kernels::scale_grad(img, scale)
}

/// The photo's camera-to-world as a row-major 3x4.
fn c2w_rows(camera: &Camera) -> [f32; 12] {
    let r = glam::Mat3::from_quat(camera.rotation);
    let p = camera.position;
    [
        r.x_axis.x, r.y_axis.x, r.z_axis.x, p.x, //
        r.x_axis.y, r.y_axis.y, r.z_axis.y, p.y, //
        r.x_axis.z, r.y_axis.z, r.z_axis.z, p.z,
    ]
}

/// `camera` moved to camera-to-world `c2w * transform`: the world-to-camera
/// becomes `transform^-1 * w2c`, carried as a camera-space affine (composed
/// with the camera's own, e.g. its pose-refinement correction; the trajectory
/// gradient in `apply_pose` is unchanged by a right factor).
fn sub_pose_camera(camera: &Camera, transform: &glam::Mat4) -> Camera {
    Camera {
        local_affine: glam::Affine3A::from_mat4(transform.inverse()) * camera.local_affine,
        ..*camera
    }
}



#[cfg(test)]
#[path = "blur_tests.rs"]
mod blur_tests;
