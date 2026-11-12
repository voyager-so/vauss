use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Clone, Parser, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    /// Internal schedule override; not exposed as a CLI option.
    #[arg(skip = 1000u32)]
    #[serde(default = "default_sh_degree_interval")]
    pub sh_degree_interval: u32,
    /// Internal: robust loss (zero GT alpha on transient pixels after
    /// `ROBUST_START_ITER`). On by default; off for ablations.
    #[arg(skip = true)]
    #[serde(default = "default_true")]
    pub robust_loss: bool,
    /// Internal: the first `freeze_prefix` splats are held fixed (their
    /// parameters are restored after every step), so only the splats after
    /// them train. Used by object-removal inpainting to fit new fill splats
    /// without moving the scene. 0 = off.
    #[arg(skip = 0u32)]
    #[serde(default)]
    pub freeze_prefix: u32,
    /// Internal: rows `freeze_prefix..freeze_geometry_prefix` keep their
    /// geometry (means, rotations, scales restored after every step) while SH
    /// colour and opacity train. Used by object-removal inpainting to recolour
    /// old splats that baked the object's shadow. 0 (or <= `freeze_prefix`) =
    /// off: the freeze is exactly the `freeze_prefix` one.
    #[arg(skip = 0u32)]
    #[serde(default)]
    pub freeze_geometry_prefix: u32,
    /// Internal: quantisation-aware tail (last 1/20 of training). On by
    /// default; off for short fine-tunes whose output is not SPZ-bound.
    #[arg(skip = true)]
    #[serde(default = "default_true")]
    pub qat_tail: bool,
    /// Internal: the camera-motion blur test and exposure model (9 sub-pose
    /// renders per step once a capture is found blurred). On by default; off
    /// for time-boxed previews, where the slower steps cost more than the
    /// deblur gains (X56: rooftop MAX reached 3.9k steps in 60 s with it on).
    #[arg(skip = true)]
    #[serde(default = "default_true")]
    pub camera_motion_blur: bool,
    /// Internal: multi-view consistency weights (geometric round trip, patch
    /// NCC) and their start step; 0 = off. See `multiview`.
    #[arg(skip = 0.0f32)]
    #[serde(default)]
    pub surfel_mv_geo_weight: f32,
    #[arg(skip = 0.0f32)]
    #[serde(default)]
    pub surfel_mv_ncc_weight: f32,
    #[arg(skip = 7000u32)]
    #[serde(default = "default_mv_start")]
    pub surfel_mv_start: u32,
    /// Internal: photometric (NCC) samples per multi-view step.
    #[arg(skip = 102_400u32)]
    #[serde(default = "default_mv_ncc_samples")]
    pub surfel_mv_ncc_samples: u32,
    /// Internal: apply the multi-view terms on every Nth step only, with
    /// their weights scaled by N (the same expected gradient at 1/N of the
    /// neighbor renders and patch work).
    #[arg(skip = 3u32)]
    #[serde(default = "default_mv_every")]
    pub surfel_mv_every: u32,
    /// Internal: weight of the monocular depth-prior loss (patch + global
    /// Pearson correlation of rendered vs prior inverse depth, see
    /// `depth_prior`). Applies only to batches that carry a depth prior; it
    /// decays linearly to 0 at `depth_prior_end` (fraction of training).
    #[arg(skip = 0.5f32)]
    #[serde(default = "default_depth_prior_weight")]
    pub depth_prior_weight: f32,
    #[arg(skip = 1.0f32)]
    #[serde(default = "default_depth_prior_end")]
    pub depth_prior_end: f32,
    /// Internal: weight of the planarity term (edge-aware second-order
    /// smoothness of rendered inverse depth on low-texture pixels), on the
    /// depth-prior steps and schedule.
    #[arg(skip = 0.0f32)]
    #[serde(default)]
    pub depth_planar_weight: f32,
    /// Internal: the depth term runs on every Nth step only, its weight
    /// scaled by N (a step with it renders twice and skips the fused update).
    #[arg(skip = 2u32)]
    #[serde(default = "default_depth_prior_every")]
    pub depth_prior_every: u32,
    /// Internal: train 2DGS surfels with the geometry losses below.
    #[arg(skip = false)]
    #[serde(default)]
    pub surfel: bool,
    /// Internal: compact neural appearance (per-splat base + latents and a
    /// shared decoder) instead of SH; see `neural.rs`.
    #[arg(skip = false)]
    #[serde(default)]
    pub neural: bool,
    /// Internal: weight of the surfel normal-consistency loss.
    #[arg(skip = 0.05f32)]
    #[serde(default = "default_surfel_normal_weight")]
    pub surfel_normal_weight: f32,
    /// Internal: step at which the surfel normal loss turns on.
    #[arg(skip = 7000u32)]
    #[serde(default = "default_surfel_normal_start")]
    pub surfel_normal_start: u32,
    /// Internal: weight of the surfel depth-distortion loss.
    #[arg(skip = 1000.0f32)]
    #[serde(default = "default_surfel_dist_weight")]
    pub surfel_dist_weight: f32,
    /// Internal: step at which the surfel distortion loss turns on.
    #[arg(skip = 3000u32)]
    #[serde(default = "default_surfel_dist_start")]
    pub surfel_dist_start: u32,
    /// Internal: joint camera refinement (per-view SE(3) correction, see
    /// `pose_refine`). Off by default; `SPLAT_POSE_REFINE=1` on the product path.
    #[arg(skip = false)]
    #[serde(default)]
    pub pose_refine: bool,
    /// Internal: with `pose_refine`, also refine one focal scale per
    /// intrinsics group (a `config_overrides` field).
    #[arg(skip = false)]
    #[serde(default)]
    pub pose_refine_focal: bool,
    /// Internal: step after which camera refinement starts (then a linear LR
    /// warm-up of 1000 steps, decaying 100x by the last step); 0 = 30% of the
    /// steps, once training reaches full resolution (screened 10-10: earlier
    /// starts and 1e-3 LRs drifted house's low-track frames, -1.95 dB).
    #[arg(skip = 0u32)]
    #[serde(default = "default_pose_refine_start")]
    pub pose_refine_start: u32,
    /// Internal: per-visit Adam LR of the rotation correction (radians).
    #[arg(skip = 5e-4f64)]
    #[serde(default = "default_pose_refine_lr_rot")]
    pub pose_refine_lr_rot: f64,
    /// Internal: per-visit Adam LR of the translation correction, in units of
    /// the camera extent.
    #[arg(skip = 1e-4f64)]
    #[serde(default = "default_pose_refine_lr_trans")]
    pub pose_refine_lr_trans: f64,
    /// Internal: Adam LR of the log focal scale (updated on every view of
    /// its group, so far smaller than the per-view LRs).
    #[arg(skip = 2e-5f64)]
    #[serde(default = "default_pose_refine_lr_focal")]
    pub pose_refine_lr_focal: f64,
    /// Internal: trust region of the per-view correction: at most this many
    /// degrees of rotation and this fraction of the camera extent of
    /// translation (0 = unbounded). Useful corrections measured 0.01-0.25 deg
    /// and <0.3% extent; unconstrained views (house's first video frames)
    /// drifted to 1.4 deg / 8%.
    #[arg(skip = 0.0f64)]
    #[serde(default)]
    pub pose_refine_max_rot_deg: f64,
    #[arg(skip = 0.0f64)]
    #[serde(default)]
    pub pose_refine_max_trans: f64,
    /// Internal: refinement window end. After this step the poses are frozen
    /// and no camera gradient is requested again (no pose kernel, no
    /// readback); the LR warm-up and 100x decay span the window. 0 = refine
    /// to the last step.
    #[arg(skip = 0u32)]
    #[serde(default)]
    pub pose_refine_end: u32,
    /// Internal: request the camera gradient on every n-th step only (1 =
    /// every step). The pose kernel and readback cost scale by 1/n; each view
    /// learns on about 1/n of its visits.
    #[arg(skip = 1u32)]
    #[serde(default = "default_pose_refine_every")]
    pub pose_refine_every: u32,
    /// Internal: pose-only photometric bundle adjustment instead of joint
    /// refinement. At this step training pauses, every training view's rigid
    /// correction is optimised with the splats frozen (`pose_ba_iters` Adam
    /// steps per view, LR decaying 100x), and training continues on the
    /// corrected poses with refinement off. 0 = off. Needs `pose_refine`.
    #[arg(skip = 0u32)]
    #[serde(default)]
    pub pose_ba_step: u32,
    /// Internal: Adam steps per training view of the pose-only BA.
    #[arg(skip = 30u32)]
    #[serde(default = "default_pose_ba_iters")]
    pub pose_ba_iters: u32,
    /// Internal: pose-only BA start LRs, rotation (radians) and translation
    /// (camera extents).
    #[arg(skip = 1e-3f64)]
    #[serde(default = "default_pose_ba_lr_rot")]
    pub pose_ba_lr_rot: f64,
    #[arg(skip = 2e-4f64)]
    #[serde(default = "default_pose_ba_lr_trans")]
    pub pose_ba_lr_trans: f64,
    /// Internal: test-time pose alignment steps per held-out view before the
    /// final held-out score (splats frozen); 0 = off. Defaults to 200 with
    /// `pose_refine` on the product path (`config_overrides` sets it),
    /// and must be set on both arms of a pose-refine A/B.
    #[arg(skip = 0u32)]
    #[serde(default)]
    pub pose_align_eval_steps: u32,
    /// Total number of steps to train for.
    #[arg(
        long,
        visible_alias = "total-steps",
        help_heading = "Training options",
        default_value = "30000"
    )]
    pub total_train_iters: u32,

    #[arg(long, help_heading = "Training options")]
    pub render_mode: Option<SplatRenderMode>,

    /// Start learning rate for the mean parameters, in units of the camera
    /// extent (1.1x the largest camera distance from the camera centroid).
    #[arg(long, help_heading = "Training options", default_value = "1.6e-4")]
    pub lr_mean: f64,

    /// End learning rate for the mean parameters, in units of the camera extent.
    #[arg(long, help_heading = "Training options", default_value = "1.6e-6")]
    pub lr_mean_end: f64,

    /// Noise added to the means of low opacity gaussians, per unit of the
    /// (unscaled) mean learning rate, in units of the median scene size.
    #[arg(long, help_heading = "Training options", default_value = "6.25")]
    pub mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[arg(long, help_heading = "Training options", default_value = "10.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "0.012")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    pub lr_scale: f64,

    /// Learning rate for the rotation parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_rotation: f64,

    /// Final number of splats. Growth follows an exponential curve up to this
    /// count. 0 picks it from the initial count: Spirula's growth rate (x1.05
    /// per 100 steps) over the growth window, at most an automatic budget
    /// sized to the scene (1.2M for compact scenes, more for unbounded ones).
    #[arg(long, help_heading = "Refine options", default_value = "0")]
    pub max_splats: u32,

    /// Internal (cloud MAX tier, set via `TrainJob::max_budget`): with
    /// `max_splats` 0, let unbounded outdoor scenes grow to the full automatic
    /// budget (up to 8M, within device memory) instead of the local default's
    /// wall-time-bounded one (up to 2M).
    #[arg(skip = false)]
    #[serde(default)]
    pub max_budget: bool,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[arg(
        long,
        help_heading = "Refine options",
        default_value = "200",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub refine_every: u32,

    /// Iteration at which splat growth starts. Clamped to `growth_stop_iter`.
    #[arg(long, help_heading = "Refine options", default_value = "500")]
    pub growth_start_iter: u32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Split any splat whose max screen-space extent exceeds this fraction of
    /// the image dimension, shrinking the children so they land at (at most)
    /// this size on screen. 0 disables.
    #[arg(long, help_heading = "Refine options", default_value = "0.5")]
    pub split_at_screen_size: f32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// Internal: weight of the opacity regulariser `w · mean(opacity)`
    /// (3DGS-MCMC), applied for the first third of training.
    #[arg(skip = 0.005f32)]
    #[serde(default = "default_opacity_reg")]
    pub opacity_reg: f32,

    /// Factor of the opacity decay.
    #[arg(long, help_heading = "Training options", default_value = "0.004")]
    pub opac_decay: f32,

    /// Detector masks (person/distractor masks, alpha mode masked) stay on
    /// only while what they hide moves: after a short probe the masks are
    /// dropped when the static scene already explains their pixels (a statue
    /// read as a person). Leave off for masks a user drew on purpose.
    #[arg(long, help_heading = "Training options")]
    #[serde(default)]
    pub auto_masks: bool,

    /// Mip-Splatting 3D-filter strength (the paper's `s`): each splat gets a
    /// per-splat world-space scale floor `sqrt(min_scale_factor) · pixel size
    /// at the nearest observing camera`, i.e. a ~0.32px std-dev floor at the
    /// default. Folded into scales/opacity at render (and baked at export),
    /// never optimized. 0 disables the filter.
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub min_scale_factor: f32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    pub lpips_loss_weight: f32,

    /// Base background color (R,G,B) used during training.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "0,0,0",
        value_delimiter = ',',
        num_args = 3
    )]
    pub background_color: Vec<f32>,

    /// Strength of random noise added to the background color each step.
    /// Noise is uniform in [-strength, +strength], clamped to [0, 1].
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub background_noise_strength: f32,

    /// Number of LOD levels to generate after initial training (0 = disabled).
    #[arg(long, help_heading = "LOD options", default_value = "0")]
    pub lod_levels: u32,

    /// Number of refinement training steps per LOD level.
    #[arg(long, help_heading = "LOD options", default_value = "5000")]
    pub lod_refine_steps: u32,

    /// Percentage of gaussians to keep at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_decimation_keep: u32,

    /// Percentage to scale source images at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_image_scale: u32,

    /// Scene scale used for random splat initialization.
    /// When no init is provided, splats are randomly placed
    /// inside camera frustums up to this depth. By default this is
    /// estimated from the camera spacing (with a 1m minimum).
    #[arg(long, help_heading = "Training options")]
    pub random_init_scene_scale: Option<f32>,
}

fn default_sh_degree_interval() -> u32 { 1000 }
fn default_true() -> bool { true }
fn default_depth_prior_weight() -> f32 { 0.5 }
fn default_depth_prior_end() -> f32 { 1.0 }
fn default_depth_prior_every() -> u32 { 2 }
fn default_opacity_reg() -> f32 { 0.005 }
fn default_mv_start() -> u32 { 7000 }
fn default_mv_ncc_samples() -> u32 { 102_400 }
fn default_mv_every() -> u32 { 3 }
fn default_surfel_normal_weight() -> f32 { 0.05 }
fn default_surfel_normal_start() -> u32 { 7000 }
fn default_surfel_dist_weight() -> f32 { 1000.0 }
fn default_surfel_dist_start() -> u32 { 3000 }
fn default_pose_refine_start() -> u32 { 0 }
fn default_pose_refine_lr_rot() -> f64 { 5e-4 }
fn default_pose_refine_lr_trans() -> f64 { 1e-4 }
fn default_pose_refine_lr_focal() -> f64 { 2e-5 }
fn default_pose_refine_every() -> u32 { 1 }
fn default_pose_ba_iters() -> u32 { 30 }
fn default_pose_ba_lr_rot() -> f64 { 1e-3 }
fn default_pose_ba_lr_trans() -> f64 { 2e-4 }

impl Default for TrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}

impl TrainConfig {
    /// Surfel (2DGS, mesh-goal) training recipe. Measured on DTU (B15 scorer,
    /// 30k): no opacity decay, the 3DGS/2DGS opacity learning rate, and no
    /// robust transient mask (clean captures lose real signal to it).
    pub fn apply_surfel_defaults(&mut self) {
        self.surfel = true;
        self.opac_decay = 0.0;
        // Never validated with surfel/mesh training.
        self.opacity_reg = 0.0;
        self.lr_opac = 0.05;
        self.robust_loss = false;
        // PGSR-style multi-view consistency (registered g1-dtu5-pgsr-mv1-v1).
        self.surfel_mv_geo_weight = 0.03;
        self.surfel_mv_ncc_weight = 0.15;
    }

    /// The neural cloud preset (`max-neural-2.5m`, opt-in; the hosted MAX
    /// tier is SH `max_budget`): neural appearance with a 2.5M splat budget,
    /// about the VRAM of SH at 1.5M. The 3D filter floor stays off in neural mode.
    pub fn apply_neural_max(&mut self) {
        self.neural = true;
        self.max_splats = 2_500_000;
    }

    pub fn total_iters(&self) -> u32 {
        self.total_train_iters + self.lod_levels * self.lod_refine_steps
    }

    /// Training image scale at `iter` (main phase): progressive resolution,
    /// 1/4 for the first 10% of steps, 1/2 until 30%, then full.
    /// A surfel schedule step (`surfel_*_start`, set for the 2DGS/PGSR 30k
    /// recipe) as the same fraction of this run's length: a 7k mesh run
    /// starts the normal and multi-view terms at 1633, not after its end.
    pub fn surfel_step(&self, start: u32) -> u32 {
        (u64::from(start) * u64::from(self.total_train_iters) / 30_000) as u32
    }

    pub fn image_scale_at(&self, iter: u32) -> f32 {
        let t = iter as f32 / self.total_train_iters.max(1) as f32;
        if t < 0.1 {
            0.25
        } else if t < 0.3 {
            0.5
        } else {
            1.0
        }
    }
}
