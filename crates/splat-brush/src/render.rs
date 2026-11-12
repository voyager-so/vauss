//! Arbitrary-camera rendering of a trained splat (RGB + alpha + depth), in-process.
//!
//! Coordinates are the SfM world frame the splat was trained in. Cameras use OpenCV/COLMAP
//! axes: +x right, +y down, +z forward; `rotation` is camera-to-world.

use std::path::Path;

use brush_render::camera::{Camera, focal_to_fov};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
/// Splats on a renderer's device (what `Renderer::splats` hands out).
pub type GpuSplats = Splats;
use brush_render::kernels::camera_model::CameraModel;
use burn::tensor::Device;
use glam::{Mat3, Mat4, Quat, UVec2, Vec3};
use image::RgbImage;

use crate::engine;

pub use glam;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderCamera {
    pub position: Vec3,
    /// Camera-to-world rotation, OpenCV axes.
    pub rotation: Quat,
    /// Pinhole intrinsics in pixels.
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl RenderCamera {
    /// From a camera-to-world matrix (OpenCV axes) and a vertical field of view (radians),
    /// principal point at the image centre.
    pub fn from_c2w(c2w: Mat4, fov_y: f64, width: u32, height: u32) -> Self {
        let (_, rotation, position) = c2w.to_scale_rotation_translation();
        let f = (height as f64 / 2.0 / (fov_y / 2.0).tan()) as f32;
        Self { position, rotation, fx: f, fy: f, cx: width as f32 / 2.0, cy: height as f32 / 2.0, width, height }
    }

    /// Camera at `eye` looking at `target`, with image-up as close to `world_up` as possible.
    pub fn look_at(eye: Vec3, target: Vec3, world_up: Vec3, fov_y: f64, width: u32, height: u32) -> Self {
        let fwd = (target - eye).normalize();
        let right = fwd.cross(world_up).normalize();
        let down = fwd.cross(right);
        let c2w = Mat4::from_cols(right.extend(0.0), down.extend(0.0), fwd.extend(0.0), eye.extend(1.0));
        Self::from_c2w(c2w, fov_y, width, height)
    }

    pub fn c2w(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position)
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }

    pub fn fov_y(&self) -> f64 {
        focal_to_fov(self.fy as f64, self.height, &CameraModel::Pinhole)
    }

    pub(crate) fn to_brush(self) -> Camera {
        let fov_x = focal_to_fov(self.fx as f64, self.width, &CameraModel::Pinhole);
        let center = glam::vec2(self.cx / self.width as f32, self.cy / self.height as f32);
        Camera::new(self.position, self.rotation, fov_x, self.fov_y(), center, CameraModel::Pinhole)
    }

    /// Pinhole approximation of a Brush capture camera (drops lens distortion).
    pub(crate) fn from_brush(cam: &Camera, width: u32, height: u32) -> Self {
        let size = glam::uvec2(width, height);
        let f = cam.focal(size);
        let c = cam.center(size);
        Self { position: cam.position, rotation: cam.rotation, fx: f.x, fy: f.y, cx: c.x, cy: c.y, width, height }
    }
}

/// Splats in the trainer's raw parameterisation, flat row-major: means [N,3], rotations [N,4]
/// (w,x,y,z), log scales [N,3], SH [N,coeffs,3] (DC first), opacity logits [N].
pub struct RawSplats {
    pub means: Vec<f32>,
    pub rotations: Vec<f32>,
    pub log_scales: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
    pub raw_opacities: Vec<f32>,
    /// Mip-Splatting antialiasing (the package's `render_mode == "mip"`).
    pub mip: bool,
}

impl RawSplats {
    fn into_splats(self, device: &Device) -> Splats {
        let mode = if self.mip { SplatRenderMode::Mip } else { SplatRenderMode::Default };
        Splats::from_raw(self.means, self.rotations, self.log_scales, self.sh_coeffs, self.raw_opacities, mode, device)
    }
}

pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Composited on black.
    pub rgb: RgbImage,
    /// Accumulated opacity in [0, 1], row-major.
    pub alpha: Vec<f32>,
    /// Expected view-space z-depth (along `forward`, sum(w z) / sum(w)), 0 where alpha < 0.05.
    /// Empty unless requested.
    pub depth: Vec<f32>,
    /// Median view-space z-depth (2DGS convention), 0 where alpha < 0.05. Empty unless requested.
    pub median_depth: Vec<f32>,
    /// World-space unit normal facing the camera, zero where alpha < 0.05. Empty unless requested.
    pub normal: Vec<[f32; 3]>,
}

/// A loaded splat on the GPU. Holds its own async runtime; methods block.
pub struct Renderer {
    rt: tokio::runtime::Runtime,
    #[allow(dead_code)]
    device: Device,
    splats: Splats,
    up_axis: Option<Vec3>,
    background: Vec3,
}

impl Renderer {
    pub fn open(ply: &Path) -> anyhow::Result<Self> {
        let rt = engine::runtime()?;
        let (device, (splats, up_axis)) = rt.block_on(async {
            let device = engine::init_device().await;
            let loaded = engine::load_ply(ply, SplatRenderMode::Default, &device).await?;
            anyhow::Ok((device, loaded))
        })?;
        Ok(Self { rt, device, splats, up_axis, background: Vec3::ZERO })
    }

    /// From raw trainer-parameterised arrays (see [`RawSplats`]) instead of a PLY.
    /// `cpu`: a software adapter only (see [`engine::init_device_cpu`]), never a GPU.
    pub fn from_raw(raw: RawSplats, cpu: bool) -> anyhow::Result<Self> {
        let rt = engine::runtime()?;
        let device = if cpu { rt.block_on(engine::init_device_cpu())? } else { rt.block_on(engine::init_device()) };
        let splats = raw.into_splats(&device);
        Ok(Self { rt, device, splats, up_axis: None, background: Vec3::ZERO })
    }

    /// Whether this machine has a hardware GPU adapter to render on (see [`engine::has_hardware_adapter`]).
    pub fn has_hardware_adapter() -> anyhow::Result<bool> {
        Ok(engine::runtime()?.block_on(engine::has_hardware_adapter()))
    }

    /// Draw the splats of another PLY from now on (same GPU runtime: wgpu/cubecl allow one device
    /// per process, so a second [`Renderer::open`] would panic).
    pub fn load(&mut self, ply: &Path) -> anyhow::Result<()> {
        let (splats, up_axis) = self.rt.block_on(engine::load_ply(ply, SplatRenderMode::Default, &self.device))?;
        self.splats = splats;
        self.up_axis = up_axis;
        Ok(())
    }

    /// Replace the splats drawn (same GPU runtime): one animated frame of an effect.
    pub fn set_raw(&mut self, raw: RawSplats) {
        self.splats = raw.into_splats(&self.device);
    }

    /// The splats drawn, on this renderer's device (e.g. the rest state a GPU effect reads).
    pub fn splats(&self) -> &Splats {
        &self.splats
    }

    /// Draw these splats (already on this renderer's device) from now on.
    pub fn set_splats(&mut self, splats: Splats) {
        self.splats = splats;
    }

    /// Run `f` on this renderer's async runtime (GPU readbacks block on it).
    pub fn block_on<T>(&self, f: impl std::future::Future<Output = T>) -> T {
        self.rt.block_on(f)
    }

    pub fn num_splats(&self) -> u32 {
        self.splats.num_splats()
    }

    /// Up direction recorded in the PLY (Brush estimates it from the capture cameras).
    pub fn up_axis(&self) -> Option<Vec3> {
        self.up_axis
    }

    /// Linear RGB in [0, 1] composited behind the splats; black by default.
    pub fn set_background(&mut self, rgb: Vec3) {
        self.background = rgb;
    }

    pub fn render(&self, camera: &RenderCamera, with_depth: bool) -> anyhow::Result<Frame> {
        self.render_brush(&camera.to_brush(), UVec2::new(camera.width, camera.height), with_depth)
    }

    pub(crate) fn render_brush(&self, camera: &Camera, size: UVec2, with_depth: bool) -> anyhow::Result<Frame> {
        let raw = self.rt.block_on(engine::render_bg(&self.splats, camera, size, with_depth, self.background))?;
        Ok(Frame {
            width: raw.width,
            height: raw.height,
            rgb: engine::to_rgb8(&raw),
            alpha: raw.rgba.chunks_exact(4).map(|p| p[3]).collect(),
            depth: raw.depth.as_deref().map(|d| mask_depth(d, &raw.rgba)).unwrap_or_default(),
            median_depth: raw.median_depth.as_deref().map(|d| mask_depth(d, &raw.rgba)).unwrap_or_default(),
            normal: raw
                .normal
                .map(|n| n.into_iter().zip(raw.rgba.chunks_exact(4)).map(|(n, p)| if p[3] >= 0.05 { n } else { [0.0; 3] }).collect())
                .unwrap_or_default(),
        })
    }
}

fn mask_depth(depth: &[f32], rgba: &[f32]) -> Vec<f32> {
    depth.iter().zip(rgba.chunks_exact(4)).map(|(&z, p)| if p[3] >= 0.05 && z.is_finite() { z } else { 0.0 }).collect()
}

/// Capture cameras of an SfM dataset dir (images/ + sparse/0) as Brush loads them, at the
/// image size Brush trains at (`max_resolution` cap on the long edge). Sorted by image name.
pub fn scene_cameras(dataset: &Path, max_resolution: u32) -> anyhow::Result<Vec<(String, RenderCamera)>> {
    let rt = engine::runtime()?;
    rt.block_on(async {
        let scene = engine::load_scene(dataset, max_resolution).await?;
        let mut out = Vec::with_capacity(scene.views.len());
        for view in &scene.views {
            let (w, h) = view.image.dimensions().await?;
            out.push((engine::view_name(view), RenderCamera::from_brush(&view.camera, w, h)));
        }
        Ok(out)
    })
}

/// A ring of `n` cameras orbiting the point the capture cameras look at, at their mean
/// distance and height, facing inward. `up` is minus the mean capture image-down axis.
pub fn orbit_cameras(capture: &[RenderCamera], n: usize) -> Vec<RenderCamera> {
    if capture.is_empty() || n == 0 {
        return vec![];
    }
    let up = -capture.iter().map(|c| c.rotation * Vec3::Y).sum::<Vec3>().normalize();
    let target = look_target(capture);
    let mean_pos = capture.iter().map(|c| c.position).sum::<Vec3>() / capture.len() as f32;
    let height = (mean_pos - target).dot(up);
    let mut radial: Vec<f32> = capture
        .iter()
        .map(|c| {
            let d = c.position - target;
            (d - up * d.dot(up)).length()
        })
        .collect();
    radial.sort_by(f32::total_cmp);
    let radius = radial[radial.len() / 2].max(1e-3);
    let start = {
        let d = capture[0].position - target;
        (d - up * d.dot(up)).normalize_or(up.any_orthonormal_vector())
    };
    let side = up.cross(start);
    let first = &capture[0];
    (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let eye = target + up * height + (start * a.cos() + side * a.sin()) * radius;
            RenderCamera::look_at(eye, target, up, first.fov_y(), first.width, first.height)
        })
        .collect()
}

/// Least-squares point closest to all optical axes (where the capture is aimed).
pub fn look_target(capture: &[RenderCamera]) -> Vec3 {
    let mut a = Mat3::ZERO;
    let mut b = Vec3::ZERO;
    for c in capture {
        let d = c.forward();
        let m = Mat3::IDENTITY - Mat3::from_cols(d * d.x, d * d.y, d * d.z);
        a += m;
        b += m * c.position;
    }
    if a.determinant().abs() < 1e-6 {
        // Parallel axes (e.g. a straight walk): fall back to a point ahead of the cameras.
        let mean = capture.iter().map(|c| c.position + c.forward()).sum::<Vec3>();
        return mean / capture.len() as f32;
    }
    a.inverse() * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_at_faces_target() {
        let cam = RenderCamera::look_at(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO, Vec3::NEG_Y, 1.0, 64, 48);
        assert!((cam.fov_y() - 1.0).abs() < 1e-5);
        assert!((cam.forward() - Vec3::Z).length() < 1e-5);
        // Image-down points along -world_up.
        assert!((cam.rotation * Vec3::Y - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn orbit_circles_target() {
        let ring: Vec<RenderCamera> = (0..12)
            .map(|i| {
                let a = i as f32 / 12.0 * std::f32::consts::TAU;
                RenderCamera::look_at(Vec3::new(a.cos() * 3.0, -1.0, a.sin() * 3.0), Vec3::ZERO, Vec3::NEG_Y, 0.8, 64, 48)
            })
            .collect();
        let target = look_target(&ring);
        assert!(target.length() < 1e-3, "{target}");
        for cam in orbit_cameras(&ring, 8) {
            let d = cam.position - Vec3::new(0.0, -1.0, 0.0);
            assert!((d.length() - 3.0).abs() < 1e-3);
            assert!(cam.forward().dot((Vec3::ZERO - cam.position).normalize()) > 0.999);
        }
    }
}
