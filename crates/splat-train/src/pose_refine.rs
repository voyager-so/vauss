//! Joint camera refinement during training (`TrainConfig::pose_refine`,
//! default off), and test-time pose alignment of held-out views for a fair
//! evaluation of it. Design note: `~/data/splat/goal2/pose-refine.md`.
//!
//! Each training view carries a rigid camera-space correction `D` (and each
//! intrinsics group, with `pose_refine_focal`, a focal scale `F =
//! diag(s, s, 1)`), rendered through [`Camera::local_affine`] as `M = F D`:
//! world-to-camera becomes `F D w2c`. The renderer's pose kernel (shared with
//! the camera-motion blur trajectories) returns `dL/dE` for the perturbation
//! `(I + E) M`, `E = [A | b]`. A left se(3) step `D <- exp(xi) D`, `xi = (w,
//! v)`, perturbs `M` by `A = F [w]x F^-1`, `b = F v`; a log-focal step `d` by
//! `A = diag(d, d, 0)`. Both are stepped with a per-view (per-group) Adam.
//!
//! Held-out views are never refined by training. With
//! `pose_align_eval_steps` > 0 each held-out view's pose is aligned against its
//! ground truth with the splats frozen before it is scored (the standard
//! pose-optimisation protocol, e.g. BARF/NeRF--): without it the training
//! poses' drift biases held-out PSNR against the refined arm.

use brush_dataset::scene::view_to_packed_data;
use brush_loss::{ImageLossConfig, image_loss};
use brush_render::AlphaMode;
use brush_render::bwd::render_splats_with_pass;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{RasterPass, Splats};
use burn::tensor::{Int, Tensor};
use glam::{Affine3A, Mat3, Quat, Vec3};
use image::DynamicImage;

/// Adam constants of the camera parameters.
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

/// Refinement start when `pose_refine_start` is 0: this fraction of the steps
/// (training reaches full resolution at 30%).
const DEFAULT_START_FRACTION: f64 = 0.3;

fn resolve_start(configured: u32, total: u32) -> u32 {
    if configured > 0 { configured } else { (total as f64 * DEFAULT_START_FRACTION) as u32 }
}

/// Linear LR warm-up after `pose_refine_start`, in steps.
const WARMUP_STEPS: u32 = 1000;
/// The LR decays exponentially to this fraction at the last step.
const LR_END_FRACTION: f64 = 0.01;

#[derive(Clone, Copy)]
struct Adam<const N: usize> {
    m: [f32; N],
    v: [f32; N],
    t: i32,
}

impl<const N: usize> Default for Adam<N> {
    fn default() -> Self {
        Self { m: [0.0; N], v: [0.0; N], t: 0 }
    }
}

impl<const N: usize> Adam<N> {
    /// The Adam direction for gradient `g` (bias-corrected `m / sqrt(v)`).
    fn direction(&mut self, g: [f32; N]) -> [f32; N] {
        self.t += 1;
        let c1 = 1.0 - BETA1.powi(self.t);
        let c2 = 1.0 - BETA2.powi(self.t);
        std::array::from_fn(|i| {
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * g[i];
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * g[i] * g[i];
            (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + EPS)
        })
    }
}

/// One view's correction: rigid camera-space map and its optimizer.
#[derive(Clone, Copy)]
struct ViewPose {
    d: Affine3A,
    adam: Adam<6>,
}

impl Default for ViewPose {
    fn default() -> Self {
        Self { d: Affine3A::IDENTITY, adam: Adam::default() }
    }
}

#[derive(Clone, Copy, Default)]
struct GroupFocal {
    log_scale: f32,
    adam: Adam<1>,
}

/// `exp` of a camera-space se(3) step (rotation by the axis-angle `w`, then
/// translation by `v`): a first-order-exact retraction, all the small steps
/// need.
fn se3_step(w: Vec3, v: Vec3) -> Affine3A {
    Affine3A::from_rotation_translation(Quat::from_scaled_axis(w), v)
}

fn skew(w: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, w.z, -w.y),
        Vec3::new(-w.z, 0.0, w.x),
        Vec3::new(w.y, -w.x, 0.0),
    )
}

/// The kernel's `[A row-major | b]` camera gradient as (`G_A`, `g_b`).
fn split_grad(g: &[f32]) -> (Mat3, Vec3) {
    let ga = Mat3::from_cols(
        Vec3::new(g[0], g[3], g[6]),
        Vec3::new(g[1], g[4], g[7]),
        Vec3::new(g[2], g[5], g[8]),
    );
    (ga, Vec3::new(g[9], g[10], g[11]))
}

/// Frobenius inner product.
fn dot3(a: Mat3, b: Mat3) -> f32 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis) + a.z_axis.dot(b.z_axis)
}

/// `dL/d(w, v)` of the left step `exp(xi) D` given `dL/dE` at `M = F D`.
fn pose_grad(ga: Mat3, gb: Vec3, focal: f32) -> [f32; 6] {
    let f = Mat3::from_diagonal(Vec3::new(focal, focal, 1.0));
    let f_inv = Mat3::from_diagonal(Vec3::new(1.0 / focal, 1.0 / focal, 1.0));
    let dw = |axis: Vec3| dot3(ga, f * skew(axis) * f_inv);
    let gv = f.transpose() * gb;
    [dw(Vec3::X), dw(Vec3::Y), dw(Vec3::Z), gv.x, gv.y, gv.z]
}

/// `d` projected into the trust region: rotation angle at most `max_rot`
/// (radians), translation at most `max_trans`; a bound of 0 is no bound.
fn clamp_rigid(d: Affine3A, max_rot: f32, max_trans: f32) -> Affine3A {
    let (_, rot, mut t) = d.to_scale_rotation_translation();
    let (axis, mut angle) = rot.to_axis_angle();
    if angle > std::f32::consts::PI {
        angle -= 2.0 * std::f32::consts::PI;
    }
    let rot = if max_rot > 0.0 && angle.abs() > max_rot {
        Quat::from_axis_angle(axis, max_rot * angle.signum())
    } else {
        rot
    };
    if max_trans > 0.0 && t.length() > max_trans {
        t *= max_trans / t.length();
    }
    Affine3A::from_rotation_translation(rot, t)
}

/// Rotation angle (degrees) and translation length of a rigid correction.
pub fn rigid_delta(d: &Affine3A) -> (f32, f32) {
    let (_, rot, t) = d.to_scale_rotation_translation();
    (rot.to_axis_angle().1.to_degrees().abs().min(360.0), t.length())
}

/// Per-view refined camera of the training set.
pub struct PoseRefine {
    views: Vec<ViewPose>,
    focal: Vec<GroupFocal>,
    /// Loader view index -> training view (`views`) index, and each training
    /// view's intrinsics group.
    loader_to_view: Vec<usize>,
    view_group: Vec<usize>,
    refine_focal: bool,
    lr_rot: f64,
    lr_trans: f64,
    lr_focal: f64,
    /// Trust region: max rotation (radians) and translation (scene units); 0 = none.
    max_rot: f32,
    max_trans: f32,
    start: u32,
    /// `pose_refine_start` as configured (0 = a fraction of the steps).
    start_override: u32,
    /// Window end (`pose_refine_end`; 0 = the last step): no learning after it.
    end: u32,
    /// Pose-only BA mode (`pose_ba_step`): the corrections are set by
    /// [`Self::set_correction`], never learned during training.
    ba: bool,
    /// Learn on every `every`-th step only (`pose_refine_every`).
    every: u32,
    total: u32,
    pub updates: u64,
}

pub struct PoseGradTicket {
    view: usize,
    focal: f32,
    lr_scale: f64,
}

impl PoseRefine {
    /// `view_group[i]` is training view `i`'s intrinsics group; `extent` the
    /// scene's camera extent (translation LR unit).
    pub fn new(config: &crate::config::TrainConfig, view_group: Vec<usize>, extent: f32) -> Self {
        let n = view_group.len();
        let groups = view_group.iter().copied().max().map_or(0, |g| g + 1);
        Self {
            views: vec![ViewPose::default(); n],
            focal: vec![GroupFocal::default(); groups],
            loader_to_view: (0..n).collect(),
            view_group,
            refine_focal: config.pose_refine_focal,
            lr_rot: config.pose_refine_lr_rot,
            lr_trans: config.pose_refine_lr_trans * extent as f64,
            lr_focal: config.pose_refine_lr_focal,
            max_rot: (config.pose_refine_max_rot_deg as f32).to_radians(),
            max_trans: config.pose_refine_max_trans as f32 * extent,
            start_override: config.pose_refine_start,
            start: resolve_start(config.pose_refine_start, config.total_train_iters),
            end: config.pose_refine_end,
            ba: config.pose_ba_step > 0,
            every: config.pose_refine_every.max(1),
            total: config.total_train_iters,
            updates: 0,
        }
    }

    pub fn set_total(&mut self, total: u32) {
        self.start = resolve_start(self.start_override, total);
        self.total = total;
    }

    /// The loader's view order (e.g. training views minus validation views).
    pub fn set_loader_views(&mut self, loader_to_view: Vec<usize>) {
        self.loader_to_view = loader_to_view;
    }

    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    /// LR multiplier at `step` (1-based): 0 before `start` and after the
    /// window `end`, linear warm-up (at most a quarter of the window), then
    /// exponential decay to `LR_END_FRACTION` at the window's (or the run's)
    /// last step. Always 0 in BA mode.
    pub fn lr_scale(&self, step: u32) -> f64 {
        let last = if self.end > 0 { self.end.min(self.total) } else { self.total };
        if self.ba || step <= self.start || step > last {
            return 0.0;
        }
        let since = (step - self.start) as f64;
        let span = last.saturating_sub(self.start).max(1) as f64;
        let warm = (since / (WARMUP_STEPS as f64).min(span / 4.0).max(1.0)).min(1.0);
        warm * LR_END_FRACTION.powf((since / span).min(1.0))
    }

    /// Camera-space focal map of training view `view` (identity without
    /// focal refinement): the fixed part of its camera in a pose-only BA.
    pub fn view_focal_local(&self, view: usize) -> Affine3A {
        self.heldout_local(self.view_group[view])
    }

    /// Training view `view`'s current rigid correction.
    pub fn correction(&self, view: usize) -> Affine3A {
        self.views[view].d
    }

    /// Set training view `view`'s rigid correction (pose-only BA), projected
    /// into the trust region.
    pub fn set_correction(&mut self, view: usize, d: Affine3A) {
        self.views[view].d = clamp_rigid(d, self.max_rot, self.max_trans);
        self.updates += 1;
    }

    fn view_of(&self, loader_index: usize) -> Option<usize> {
        self.loader_to_view.get(loader_index).copied().filter(|&v| v < self.views.len())
    }

    fn focal_scale(&self, group: usize) -> f32 {
        self.focal.get(group).map_or(1.0, |g| g.log_scale.exp())
    }

    /// Camera-space map `M = F D` of training view `view`.
    fn local(&self, view: usize) -> Affine3A {
        let s = self.focal_scale(self.view_group[view]);
        Affine3A::from_scale(Vec3::new(s, s, 1.0)) * self.views[view].d
    }

    /// The batch camera with its refined pose, and a ticket for its gradient
    /// when the pose learns this step.
    pub fn camera(&self, camera: &Camera, loader_index: usize, step: u32) -> (Camera, Option<PoseGradTicket>) {
        let Some(view) = self.view_of(loader_index) else {
            return (*camera, None);
        };
        let refined = Camera { local_affine: self.local(view) * camera.local_affine, ..*camera };
        let lr_scale = if step.is_multiple_of(self.every) { self.lr_scale(step) } else { 0.0 };
        let ticket = (lr_scale > 0.0).then(|| PoseGradTicket {
            view,
            focal: self.focal_scale(self.view_group[view]),
            lr_scale,
        });
        (refined, ticket)
    }

    /// Apply one view's camera gradient `[A row-major | b]` (12 floats).
    pub fn apply(&mut self, ticket: PoseGradTicket, g: &[f32]) {
        if g.len() < 12 || g.iter().any(|x| !x.is_finite()) {
            return;
        }
        let (ga, gb) = split_grad(g);
        let pose = &mut self.views[ticket.view];
        let dir = pose.adam.direction(pose_grad(ga, gb, ticket.focal));
        let lr_r = (self.lr_rot * ticket.lr_scale) as f32;
        let lr_t = (self.lr_trans * ticket.lr_scale) as f32;
        let w = -Vec3::new(dir[0], dir[1], dir[2]) * lr_r;
        let v = -Vec3::new(dir[3], dir[4], dir[5]) * lr_t;
        pose.d = clamp_rigid(se3_step(w, v) * pose.d, self.max_rot, self.max_trans);
        if self.refine_focal {
            let group = self.view_group[ticket.view];
            let f = &mut self.focal[group];
            let d = f.adam.direction([ga.x_axis.x + ga.y_axis.y])[0];
            f.log_scale -= d * (self.lr_focal * ticket.lr_scale) as f32;
        }
        self.updates += 1;
    }

    /// Learned focal scale of each intrinsics group (1 without focal refinement).
    pub fn focal_scales(&self) -> Vec<f32> {
        (0..self.focal.len()).map(|g| self.focal_scale(g)).collect()
    }

    /// Camera-space map for a view outside training (held-out) in `group`:
    /// the learned focal only, never a pose.
    pub fn heldout_local(&self, group: usize) -> Affine3A {
        let s = self.focal_scale(group);
        Affine3A::from_scale(Vec3::new(s, s, 1.0))
    }

    /// Training view `view`'s camera with its refined pose and focal.
    pub fn refined_camera(&self, view: usize, camera: &Camera) -> Camera {
        Camera { local_affine: self.local(view) * camera.local_affine, ..*camera }
    }

    /// Stop learning: the current corrections are used as they are and no camera gradient is
    /// requested again.
    pub fn freeze(&mut self) {
        self.start_override = u32::MAX;
        self.start = u32::MAX;
    }

    /// Per training view: the rigid correction `D` (camera-space).
    pub fn corrections(&self) -> Vec<Affine3A> {
        self.views.iter().map(|v| v.d).collect()
    }
}

/// COLMAP-convention world-to-camera (`qvec` w,x,y,z and `tvec`) of `camera`
/// with its rigid camera-space correction `d` applied (`w2c' = D w2c`).
pub fn refined_w2c(camera: &Camera, d: &Affine3A) -> ([f32; 4], [f32; 3]) {
    let w2c = *d * camera.local_to_world().inverse();
    let (_, q, t) = w2c.to_scale_rotation_translation();
    let q = q.normalize();
    ([q.w, q.x, q.y, q.z], t.to_array())
}

/// Result of [`align_view`].
pub struct Alignment {
    /// The held-out camera with the aligned camera-space map.
    pub camera: Camera,
    /// Rigid part of the alignment (camera-space).
    pub correction: Affine3A,
    pub first_loss: f32,
    pub last_loss: f32,
}

/// Start LRs of [`align_view`]: rotation (radians) and translation (camera
/// extents) per Adam step; both decay 100x over the steps.
#[derive(Clone, Copy)]
pub struct AlignLr {
    pub rot: f32,
    pub trans: f32,
}

/// Held-out test-time alignment: 1e-3 rad and 1e-3 extents.
pub const ALIGN_EVAL_LR: AlignLr = AlignLr { rot: 1e-3, trans: 1e-3 };

/// Test-time pose alignment of one held-out view: the splats stay frozen and
/// only the view's rigid camera-space correction is optimised for `steps`
/// steps (Adam, LR decaying 100x) against its ground truth, on the training
/// loss, on the reference black background. `base_local` is the view's fixed
/// camera-space map (the learned focal); `init` the correction to start from
/// (identity for a held-out view; the current one in a pose-only BA);
/// `extent` the translation LR unit.
#[allow(clippy::too_many_arguments)]
pub async fn align_view(
    splats: &Splats,
    camera: &Camera,
    base_local: Affine3A,
    init: Affine3A,
    gt_img: &DynamicImage,
    alpha_mode: AlphaMode,
    steps: u32,
    lr: AlignLr,
    extent: f32,
    ssim_weight: f32,
) -> anyhow::Result<Alignment> {
    let device = splats.device();
    let res = glam::uvec2(gt_img.width(), gt_img.height());
    let (gt_data, has_alpha) = view_to_packed_data(gt_img.clone(), alpha_mode);
    let gt: Tensor<2, Int> = Tensor::from_data(gt_data, &device.clone().inner());
    let cfg = ImageLossConfig {
        l1_weight: 1.0 - ssim_weight,
        ssim_weight: -ssim_weight,
        composite_bg: None,
        mask: has_alpha && alpha_mode == AlphaMode::Masked,
        alpha_weight: 0.0,
    };
    let focal = base_local.matrix3.x_axis.x;
    let mut d = init;
    let mut adam = Adam::<6>::default();
    let (mut first_loss, mut last_loss) = (f32::NAN, f32::NAN);
    for k in 0..steps {
        let cam = Camera { local_affine: base_local * d * camera.local_affine, ..*camera };
        let out = render_splats_with_pass(
            splats.clone(),
            &cam,
            res,
            Vec3::ZERO,
            RasterPass::BackwardFusedUpdate,
            true,
        )
        .await;
        let loss = image_loss(out.img.clone(), gt.clone(), cfg);
        let loss_val = loss.clone().inner();
        let mut grads = loss.backward();
        let Some(g) = out.pose_grad_holder.grad_remove(&mut grads) else {
            anyhow::bail!("pose alignment: the render produced no camera gradient");
        };
        drop(grads);
        let g: Vec<f32> = g.into_data_async().await?.try_to_vec().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let l: f32 = loss_val.into_scalar_async().await?;
        if k == 0 {
            first_loss = l;
        }
        last_loss = l;
        if g.iter().any(|x| !x.is_finite()) {
            break;
        }

        let (ga, gb) = split_grad(&g);
        let dir = adam.direction(pose_grad(ga, gb, focal));
        // Start LRs decaying 100x over the steps.
        let decay = 0.01f32.powf(k as f32 / steps.max(1) as f32);
        let w = -Vec3::new(dir[0], dir[1], dir[2]) * lr.rot * decay;
        let v = -Vec3::new(dir[3], dir[4], dir[5]) * lr.trans * decay * extent;
        d = se3_step(w, v) * d;
    }
    Ok(Alignment {
        camera: Camera { local_affine: base_local * d * camera.local_affine, ..*camera },
        correction: d,
        first_loss,
        last_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The analytic `dL/dw` of `pose_grad` matches a finite difference of a
    /// linear functional of `M = F exp(w) D`.
    #[test]
    fn pose_grad_matches_finite_difference() {
        let focal = 1.07f32;
        let f = Affine3A::from_scale(Vec3::new(focal, focal, 1.0));
        let d = se3_step(Vec3::new(0.01, -0.02, 0.03), Vec3::new(0.1, 0.0, -0.2));
        // L(M) = <G, A(M)> + <gb, b(M)> with M expressed relative to M0:
        // E = M M0^-1 - I.
        let ga = Mat3::from_cols(
            Vec3::new(0.3, -1.2, 0.5),
            Vec3::new(0.7, 0.2, -0.4),
            Vec3::new(-0.9, 1.1, 0.6),
        );
        let gb = Vec3::new(0.25, -0.5, 0.75);
        let m0 = f * d;
        let loss = |xi: [f32; 6]| {
            let m = f * se3_step(Vec3::new(xi[0], xi[1], xi[2]), Vec3::new(xi[3], xi[4], xi[5])) * d;
            let e = m * m0.inverse();
            dot3(ga, e.matrix3.into()) + gb.dot(e.translation.into())
        };
        let analytic = pose_grad(ga, gb, focal);
        for i in 0..6 {
            let h = 1e-3;
            let mut p = [0.0; 6];
            p[i] = h;
            let mut n = [0.0; 6];
            n[i] = -h;
            let fd = (loss(p) - loss(n)) / (2.0 * h);
            assert!((fd - analytic[i]).abs() < 2e-3, "component {i}: fd {fd} analytic {}", analytic[i]);
        }
    }

    #[test]
    fn clamp_rigid_bounds_rotation_and_translation() {
        let d = se3_step(Vec3::new(0.0, 0.05, 0.0), Vec3::new(3.0, 4.0, 0.0));
        let c = clamp_rigid(d, 0.01, 1.0);
        let (r, t) = rigid_delta(&c);
        assert!((r - 0.01f32.to_degrees()).abs() < 1e-3, "{r}");
        assert!((t - 1.0).abs() < 1e-5, "{t}");
        let (r0, t0) = rigid_delta(&clamp_rigid(d, 0.0, 0.0));
        assert!((r0 - 0.05f32.to_degrees()).abs() < 1e-3 && (t0 - 5.0).abs() < 1e-4);
    }

    #[test]
    fn lr_schedule_warms_up_and_decays() {
        let mut config = crate::config::TrainConfig::default();
        config.pose_refine = true;
        config.pose_refine_start = 1000;
        config.total_train_iters = 30_000;
        let p = PoseRefine::new(&config, vec![0; 4], 1.0);
        assert_eq!(p.lr_scale(1000), 0.0);
        assert!((p.lr_scale(1500) - 0.5 * LR_END_FRACTION.powf(500.0 / 29_000.0)).abs() < 1e-9);
        assert!((p.lr_scale(30_000) - LR_END_FRACTION).abs() < 1e-9);
    }

    #[test]
    fn lr_window_stops_learning_and_ba_never_learns() {
        let mut config = crate::config::TrainConfig::default();
        config.pose_refine = true;
        config.pose_refine_start = 1000;
        config.pose_refine_end = 6000;
        config.total_train_iters = 45_000;
        let p = PoseRefine::new(&config, vec![0; 4], 1.0);
        assert_eq!(p.lr_scale(1000), 0.0);
        assert!((p.lr_scale(2000) - LR_END_FRACTION.powf(0.2)).abs() < 1e-9);
        assert!((p.lr_scale(6000) - LR_END_FRACTION).abs() < 1e-9);
        assert_eq!(p.lr_scale(6001), 0.0);
        config.pose_refine_end = 0;
        config.pose_ba_step = 5000;
        let p = PoseRefine::new(&config, vec![0; 4], 1.0);
        assert_eq!(p.lr_scale(2000), 0.0);
    }
}
