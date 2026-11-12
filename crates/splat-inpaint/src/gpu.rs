//! Brush/Burn plumbing: device, scene views, host splat arrays, renders and the frozen-prefix
//! fine-tune. (Mirrors splat-brush's engine.rs, which this crate cannot depend on: splat-brush
//! calls into us.)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use brush_dataset::config::LoadDatasetConfig;
use brush_dataset::load_image::LoadImage;
use brush_dataset::scene::{Scene, SceneView};
use brush_dataset::scene_loader::SceneLoader;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::{TextureMode, render_splats, render_splats_geometry};
use brush_train::config::TrainConfig;
use brush_train::train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds};
use brush_vfs::BrushVfs;
use burn::module::Module;
use burn::tensor::{Device, FloatDType, Tensor, s};
use burn_wgpu::graphics::AutoGraphicsApi;
use burn_wgpu::{MemoryConfiguration, RuntimeOptions, WgpuDevice};
use glam::{UVec2, Vec3};

use crate::img::RgbF;

pub const SH_C0: f32 = 0.282_094_8;

pub async fn init_device() -> Device {
    let options = RuntimeOptions { tasks_max: 64, memory_config: MemoryConfiguration::ExclusivePages };
    let setup = burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), options).await;
    let adapter = setup.adapter.get_info();
    eprintln!("[inpaint] adapter {} ({:?}, driver {} {})", adapter.name, adapter.backend, adapter.driver, adapter.driver_info);
    brush_render::install_default_device_autotune(&setup);
    WgpuDevice::default().into()
}

pub fn load_config(max_resolution: u32) -> LoadDatasetConfig {
    LoadDatasetConfig {
        max_frames: None,
        max_resolution,
        eval_split_every: None,
        eval_names: Vec::new(),
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
        invert_masks: false,
        units_per_meter: 1.0,
        max_scene_batch_cache_size: 6 << 30,
    }
}

/// Image path relative to images/, as COLMAP names it.
pub fn view_name(view: &SceneView) -> String {
    let p = view.image.path();
    let p = p.strip_prefix("images").unwrap_or(p);
    p.to_string_lossy().replace('\\', "/")
}

/// All registered views of a COLMAP dataset dir, at the training resolution cap.
pub async fn load_views(dataset: &Path, max_resolution: u32) -> anyhow::Result<Vec<SceneView>> {
    let vfs = Arc::new(BrushVfs::from_path(dataset).await.context("mounting dataset")?);
    let loaded = brush_dataset::load_dataset(vfs, &load_config(max_resolution)).await?;
    let mut views: Vec<SceneView> = loaded.dataset.train.views.iter().cloned().collect();
    if let Some(eval) = &loaded.dataset.eval {
        views.extend(eval.views.iter().cloned());
    }
    Ok(views)
}

/// Pinhole view of a Brush camera at a pixel size (distortion dropped), OpenCV axes.
#[derive(Clone, Copy, Debug)]
pub struct Pin {
    pub pos: Vec3,
    /// Camera-to-world.
    pub rot: glam::Quat,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub w: usize,
    pub h: usize,
}

impl Pin {
    pub fn new(cam: &Camera, w: usize, h: usize) -> Self {
        let size = UVec2::new(w as u32, h as u32);
        let (f, c) = (cam.focal(size), cam.center(size));
        Self { pos: cam.position, rot: cam.rotation, fx: f.x, fy: f.y, cx: c.x, cy: c.y, w, h }
    }

    /// Camera-space ray through pixel (x, y) with z = 1.
    pub fn ray(&self, x: usize, y: usize) -> Vec3 {
        Vec3::new((x as f32 + 0.5 - self.cx) / self.fx, (y as f32 + 0.5 - self.cy) / self.fy, 1.0)
    }

    pub fn to_world(&self, pc: Vec3) -> Vec3 {
        self.rot * pc + self.pos
    }

    pub fn to_cam(&self, pw: Vec3) -> Vec3 {
        self.rot.inverse() * (pw - self.pos)
    }
}

/// Splat parameters on the host, Brush layout: SH `[n, k, 3]` (k coefficients per channel).
#[derive(Clone)]
pub struct Host {
    pub means: Vec<f32>,
    pub rots: Vec<f32>,
    pub scales: Vec<f32>,
    pub sh: Vec<f32>,
    pub opac: Vec<f32>,
    pub k: usize,
    pub mode: SplatRenderMode,
}

impl Host {
    pub async fn load(path: &Path) -> anyhow::Result<(Self, Option<Vec3>)> {
        let msg = brush_serde::load_splat_file(path, None).await.with_context(|| format!("loading {}", path.display()))?;
        anyhow::ensure!(msg.data.neural.is_none(), "neural-appearance PLYs are unsupported");
        let mode = msg.meta.render_mode.unwrap_or(SplatRenderMode::Default);
        anyhow::ensure!(mode != SplatRenderMode::Surfel, "surfel (2DGS) PLYs are unsupported");
        let d = msg.data;
        let n = d.means.len() / 3;
        let sh = d.sh_coeffs.unwrap_or_else(|| vec![0.0; n * 3]);
        let k = sh.len() / n.max(1) / 3;
        let host = Self {
            rots: d.rotations.unwrap_or_else(|| [1.0, 0.0, 0.0, 0.0].repeat(n)),
            scales: d.log_scales.unwrap_or_else(|| vec![-4.0; n * 3]),
            opac: d.raw_opacities.unwrap_or_else(|| vec![0.0; n]),
            means: d.means,
            sh,
            k,
            mode,
        };
        Ok((host, msg.meta.up_axis))
    }

    pub fn n(&self) -> usize {
        self.opac.len()
    }

    pub fn mean(&self, i: usize) -> Vec3 {
        Vec3::new(self.means[3 * i], self.means[3 * i + 1], self.means[3 * i + 2])
    }

    pub fn subset(&self, keep: &[bool]) -> Self {
        let pick = |v: &[f32], stride: usize| -> Vec<f32> {
            v.chunks_exact(stride).zip(keep).filter(|(_, k)| **k).flat_map(|(c, _)| c.iter().copied()).collect()
        };
        Self {
            means: pick(&self.means, 3),
            rots: pick(&self.rots, 4),
            scales: pick(&self.scales, 3),
            sh: pick(&self.sh, 3 * self.k),
            opac: pick(&self.opac, 1),
            k: self.k,
            mode: self.mode,
        }
    }

    pub fn append(&mut self, o: &Host) {
        assert_eq!(self.k, o.k, "SH coefficient counts differ");
        self.means.extend_from_slice(&o.means);
        self.rots.extend_from_slice(&o.rots);
        self.scales.extend_from_slice(&o.scales);
        self.sh.extend_from_slice(&o.sh);
        self.opac.extend_from_slice(&o.opac);
    }

    pub fn to_splats(&self, device: &Device) -> Splats {
        Splats::from_raw(
            self.means.clone(),
            self.rots.clone(),
            self.scales.clone(),
            self.sh.clone(),
            self.opac.clone(),
            self.mode,
            device,
        )
    }

    /// Same geometry and opacity, SH degree 0 with colour `value(i)` on every channel.
    pub fn with_colour(&self, value: impl Fn(usize) -> f32, device: &Device) -> Splats {
        let sh: Vec<f32> = (0..self.n()).flat_map(|i| [(value(i) - 0.5) / SH_C0; 3]).collect();
        Splats::from_raw(
            self.means.clone(),
            self.rots.clone(),
            self.scales.clone(),
            sh,
            self.opac.clone(),
            self.mode,
            device,
        )
    }
}

pub struct Frame {
    pub rgb: RgbF,
    pub alpha: Vec<f32>,
    /// Expected view-space z (sum w z / sum w), NaN where alpha < 1e-3.
    pub depth: Option<Vec<f32>>,
}

async fn tensor_vec(t: Tensor<3>) -> anyhow::Result<Vec<f32>> {
    t.cast(FloatDType::F32).into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))
}

/// Render on black.
pub async fn render(splats: &Splats, cam: &Camera, w: usize, h: usize, with_depth: bool) -> anyhow::Result<Frame> {
    let size = UVec2::new(w as u32, h as u32);
    let (rgba, depth) = if with_depth {
        let geo = render_splats_geometry(splats.clone(), cam, size, Vec3::ZERO).await;
        let rgba = tensor_vec(geo.img).await?;
        let geom = tensor_vec(geo.geom).await?;
        let ch = brush_render::kernels::helpers::GEOM_OUT_CHANNELS as usize;
        let depth = rgba.chunks_exact(4).zip(geom.chunks_exact(ch)).map(|(p, g)| if p[3] > 1e-3 { g[0] / p[3] } else { f32::NAN }).collect();
        (rgba, Some(depth))
    } else {
        let (img, _) = render_splats(splats.clone(), cam, size, Vec3::ZERO, None, TextureMode::Float).await;
        (tensor_vec(img).await?, None)
    };
    let rgb = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let alpha = rgba.chunks_exact(4).map(|p| p[3]).collect();
    Ok(Frame { rgb: RgbF { w, h, data: rgb }, alpha, depth })
}

/// One fine-tune view: target image + loss mask on disk (relative to the fine-tune dir).
pub struct FtView {
    pub image: PathBuf,
    pub mask: PathBuf,
    pub camera: Camera,
    pub camera_index: usize,
}

pub struct Finetuned {
    pub splats: Splats,
    pub losses: Vec<(u32, f32)>,
}

/// Train only the splats after the first `n_frozen` for `steps` steps (rows
/// `n_frozen..n_geometry_frozen` train SH colour + opacity only, geometry held): no refine (no
/// densify/prune/opacity reset), no QAT tail, no robust transient masking, full resolution
/// from step 0, black background (the targets are renders on black).
pub async fn finetune(
    splats: Splats,
    n_frozen: usize,
    n_geometry_frozen: usize,
    dir: &Path,
    views: &[FtView],
    steps: u32,
    max_resolution: u32,
    seed: u64,
    device: &Device,
) -> anyhow::Result<Finetuned> {
    let vfs = Arc::new(BrushVfs::from_path(dir).await.context("mounting fine-tune dir")?);
    let scene_views: Vec<SceneView> = views
        .iter()
        .map(|v| SceneView {
            image: LoadImage::new(vfs.clone(), v.image.clone(), Some(v.mask.clone()), max_resolution, None, false),
            camera: v.camera,
            camera_index: v.camera_index,
            depth: None,
        })
        .collect();
    let mut config = TrainConfig::default();
    config.total_train_iters = steps;
    config.freeze_prefix = n_frozen as u32;
    config.freeze_geometry_prefix = n_geometry_frozen as u32;
    config.qat_tail = false;
    config.robust_loss = false;
    config.background_noise_strength = 0.0;
    config.opacity_reg = 0.0;
    config.mean_noise_weight = 0.0;
    let sh_degree = splats.sh_degree();
    crate::stage("finetune_bounds");
    let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new_seeded(&config, device, bounds, seed);
    let mut cams = vec![];
    for v in &scene_views {
        let (w, h) = v.image.dimensions().await?;
        cams.push((v.camera.position, v.camera.focal(UVec2::new(w, h)).x));
    }
    trainer.set_view_cams(cams);
    trainer.set_target_sh_degree(sh_degree);
    let mut loader = SceneLoader::new(&Scene::new(scene_views), seed, &load_config(max_resolution));
    let mut splats = splats;
    let mut losses = vec![];
    crate::stage("finetune_first_step");
    for step in 1..=steps {
        if step == 2 {
            crate::stage("finetune");
        }
        let batch = loader.next_batch().await;
        let (stepped, stats) = trainer.step(batch, splats.train()).await;
        splats = stepped.valid();
        if step % 100 == 0 || step == steps {
            let loss = stats.loss.into_scalar_async::<f32>().await?;
            anyhow::ensure!(loss.is_finite(), "fine-tune diverged at step {step} (loss {loss})");
            println!("PROGRESS {}", serde_json::json!({"event": "step", "stage": "finetune", "step": step, "total_steps": steps, "loss": loss}));
            losses.push((step, loss));
        }
    }
    Ok(Finetuned { splats, losses })
}

/// Rows `from..` of the trained splats back on the host (SH truncated/padded to `k`).
pub async fn read_rows(splats: &Splats, from: usize, k: usize, mode: SplatRenderMode) -> anyhow::Result<Host> {
    let n = splats.num_splats() as usize;
    let vec2 = |t: Tensor<2>| async move { t.into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}")) };
    // The trainer keeps SH in f16: read everything back as f32.
    let t = vec2(splats.transforms.val().slice(s![from..n, ..]).cast(FloatDType::F32)).await?;
    let sh = splats.sh_coeffs.val().slice(s![from..n, .., ..]).cast(FloatDType::F32);
    let kk = sh.dims()[1];
    let sh = tensor_vec(sh).await?;
    let opac = splats.raw_opacities.val().slice(s![from..n]).cast(FloatDType::F32).into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let m = n - from;
    let mut host = Host { means: vec![], rots: vec![], scales: vec![], sh: vec![], opac, k, mode };
    for r in t.chunks_exact(10) {
        host.means.extend_from_slice(&r[0..3]);
        host.rots.extend_from_slice(&r[3..7]);
        host.scales.extend_from_slice(&r[7..10]);
    }
    for i in 0..m {
        for c in 0..k {
            for ch in 0..3 {
                host.sh.push(if c < kk { sh[(i * kk + c) * 3 + ch] } else { 0.0 });
            }
        }
    }
    Ok(host)
}
