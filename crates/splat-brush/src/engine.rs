//! Brush/Burn plumbing shared by the training and render workers.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use brush_dataset::config::LoadDatasetConfig;
use brush_dataset::scene::SceneView;
use brush_render::camera::Camera;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::{AlphaMode, TextureMode, render_splats, render_splats_geometry};
use brush_serde::SplatMessage;
use brush_vfs::BrushVfs;
use burn::tensor::{Device, Tensor};
use burn_wgpu::graphics::AutoGraphicsApi;
use burn_wgpu::{MemoryConfiguration, RuntimeOptions, WgpuDevice};
use glam::{UVec2, Vec3};
use image::{DynamicImage, RgbImage};

pub fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread().enable_all().build()?)
}

/// Initialise wgpu (Vulkan / Metal / DX12, whichever the platform offers) and return the device.
pub async fn init_device() -> Device {
    let options = RuntimeOptions { tasks_max: 64, memory_config: MemoryConfiguration::ExclusivePages };
    let setup = burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), options).await;
    let adapter = setup.adapter.get_info();
    eprintln!("ADAPTER {}", serde_json::json!({
        "name": adapter.name, "vendor": adapter.vendor, "device": adapter.device,
        "backend": format!("{:?}", adapter.backend),
        "device_type": format!("{:?}", adapter.device_type),
        "driver": adapter.driver, "driver_info": adapter.driver_info,
    }));
    brush_render::install_default_device_autotune(&setup);
    WgpuDevice::default().into()
}

/// Whether a hardware adapter exists (Metal on macOS, Vulkan/DX12 on a discrete or integrated GPU).
/// Software adapters (lavapipe, llvmpipe, WARP) don't count.
pub async fn has_hardware_adapter() -> bool {
    use wgpu::DeviceType::{DiscreteGpu, IntegratedGpu, VirtualGpu};
    let adapters = wgpu::Instance::default().enumerate_adapters(wgpu::Backends::all()).await;
    adapters.iter().any(|a| matches!(a.get_info().device_type, DiscreteGpu | IntegratedGpu | VirtualGpu))
}

/// The default device, which `CUBECL_WGPU_DEFAULT_DEVICE=Cpu` (set by the caller) makes the CPU one.
fn cpu_device() -> WgpuDevice {
    WgpuDevice::default()
}

/// A CPU (software Vulkan, e.g. lavapipe) device only: never a GPU. The caller limits the Vulkan
/// ICDs (VK_ICD_FILENAMES) and sets CUBECL_WGPU_DEFAULT_DEVICE=Cpu; Auto would skip a CPU-only Vulkan for a GPU on another API.
pub async fn init_device_cpu() -> anyhow::Result<Device> {
    let options = RuntimeOptions { tasks_max: 64, memory_config: MemoryConfiguration::ExclusivePages };
    let setup = burn_wgpu::init_setup_async::<burn_wgpu::graphics::Vulkan>(&cpu_device(), options).await;
    let adapter = setup.adapter.get_info();
    eprintln!("ADAPTER {}", serde_json::json!({"name": adapter.name, "backend": format!("{:?}", adapter.backend),
        "device_type": format!("{:?}", adapter.device_type), "driver": adapter.driver}));
    anyhow::ensure!(format!("{:?}", adapter.device_type) == "Cpu", "CPU render asked for, got {} ({:?})", adapter.name, adapter.device_type);
    brush_render::install_default_device_autotune(&setup);
    Ok(cpu_device().into())
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

pub struct LoadedScene {
    /// All registered views, sorted by image name.
    pub views: Vec<SceneView>,
    pub init: Option<SplatMessage>,
    pub warnings: Vec<String>,
    pub up_axis: Vec3,
}

/// Load a COLMAP dataset dir (images/ + sparse/N) with Brush's loader.
pub async fn load_scene(dataset: &Path, max_resolution: u32) -> anyhow::Result<LoadedScene> {
    let vfs = Arc::new(BrushVfs::from_path(dataset).await.context("mounting dataset")?);
    let loaded = brush_dataset::load_dataset(vfs, &load_config(max_resolution)).await?;
    let up_axis = loaded.dataset.estimate_up();
    let mut views: Vec<SceneView> = loaded.dataset.train.views.iter().cloned().collect();
    if let Some(eval) = &loaded.dataset.eval {
        views.extend(eval.views.iter().cloned());
    }
    views.sort_by_key(|v| view_name(v));
    Ok(LoadedScene { views, init: loaded.init_splat, warnings: loaded.warnings, up_axis })
}

/// A dataset to render surfels (2DGS, pinhole cameras only): `dataset` itself when its cameras are
/// pinhole, else a pinhole copy built at `scratch` (see [`splat_eval::undistort::pinhole_dataset`];
/// the source frames stay untouched for the frozen split). The copy lives while the returned
/// guard does. Errors for lenses that can't be undistorted (fisheye).
pub fn surfel_dataset(dataset: &Path, scratch: &Path) -> anyhow::Result<(std::path::PathBuf, Option<ScratchDir>)> {
    match splat_eval::undistort::pinhole_dataset(dataset, scratch)? {
        Some(u) => {
            eprintln!(
                "[dataset] surfels render pinhole cameras: undistorted {} images and {} masks in {:.1}s (camera: model, focal scale) {:?}",
                u.images, u.masks, u.seconds, u.cameras
            );
            Ok((scratch.to_path_buf(), Some(ScratchDir(scratch.to_path_buf()))))
        }
        None => Ok((dataset.to_path_buf(), None)),
    }
}

/// Removes its directory when dropped (job scratch that must not sync back).
pub struct ScratchDir(std::path::PathBuf);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Image path relative to the dataset's images/ dir, as COLMAP names it.
pub fn view_name(view: &SceneView) -> String {
    let p = view.image.path();
    let p = p.strip_prefix("images").unwrap_or(p);
    p.to_string_lossy().replace('\\', "/")
}

/// `untagged` is the render mode for a PLY without a `SplatRenderMode` comment
/// (other engines' exports, dense point clouds); our exports carry the tag.
pub async fn load_ply(path: &Path, untagged: SplatRenderMode, device: &Device) -> anyhow::Result<(Splats, Option<Vec3>)> {
    load_capped_ply(path, 0, untagged, device).await
}

/// Preserve PLY attributes while limiting the initialization before allocating GPU tensors.
/// A zero cap keeps every splat.
pub async fn load_capped_ply(path: &Path, max_splats: usize, untagged: SplatRenderMode, device: &Device) -> anyhow::Result<(Splats, Option<Vec3>)> {
    let msg = brush_serde::load_splat_file(path, None).await.with_context(|| format!("loading {}", path.display()))?;
    let mode = msg.meta.render_mode.unwrap_or(untagged);
    eprintln!("init: render_mode={mode:?} (source={})", if msg.meta.render_mode.is_some() { "ply-tag" } else { "job" });
    let before = msg.data.num_splats();
    let data = msg.data.subsample(max_splats);
    if data.num_splats() < before {
        eprintln!("init: explicit cap {max_splats} subsampled {before} splats to {}", data.num_splats());
    }
    Ok((data.into_splats(device, mode), msg.meta.up_axis))
}

/// Plain dense geometry uses the exact SfM attribute initializer, without imported uncertainty/opacity.
/// An untagged PLY takes the job's `mode`, like an SfM start.
pub async fn load_plain_init_ply(path: &Path, max_splats: usize, mode: SplatRenderMode, device: &Device) -> anyhow::Result<Splats> {
    let msg = brush_serde::load_splat_file(path, None).await.with_context(|| format!("loading {}", path.display()))?;
    let tagged = msg.meta.render_mode;
    let mode = tagged.unwrap_or(mode);
    eprintln!("init: render_mode={mode:?} (source={})", if tagged.is_some() { "ply-tag" } else { "job" });
    let mut data = msg.data.subsample(max_splats);
    data.rotations = None;
    data.log_scales = None;
    data.raw_opacities = None;
    Ok(brush_train::to_init_splats(data, mode, device))
}

/// Atomic PLY write (temp file + rename), so a reader never sees a partial checkpoint.
pub async fn save_ply(splats: Splats, up_axis: Option<Vec3>, path: &Path) -> anyhow::Result<()> {
    let bytes = brush_serde::splat_to_ply(splats, up_axis).await?;
    let tmp = path.with_extension("ply.tmp");
    tokio::fs::write(&tmp, bytes).await?;
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}

/// Rendered view: straight (not premultiplied) RGB is what was composited on black.
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    /// RGBA float, row-major, composited on the background.
    pub rgba: Vec<f32>,
    /// Expected camera-space depth (z along the view axis), NaN where alpha < 1e-3.
    pub depth: Option<Vec<f32>>,
    /// Median depth (last contributor while transmittance > 0.5), NaN where alpha < 1e-3.
    pub median_depth: Option<Vec<f32>>,
    /// Alpha-weighted, renormalised world-space normal facing the camera; zero where alpha < 1e-3.
    pub normal: Option<Vec<[f32; 3]>>,
}

async fn tensor_vec(t: Tensor<3>) -> anyhow::Result<Vec<f32>> {
    Ok(t.into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?)
}

pub async fn render(splats: &Splats, camera: &Camera, size: UVec2, with_depth: bool) -> anyhow::Result<RawFrame> {
    render_bg(splats, camera, size, with_depth, Vec3::ZERO).await
}

pub async fn render_bg(splats: &Splats, camera: &Camera, size: UVec2, with_depth: bool, background: Vec3) -> anyhow::Result<RawFrame> {
    if !with_depth {
        let (img, _) = render_splats(splats.clone(), camera, size, background, None, TextureMode::Float).await;
        let rgba = tensor_vec(img).await?;
        return Ok(RawFrame { width: size.x, height: size.y, rgba, depth: None, median_depth: None, normal: None });
    }
    // One raster pass: colour plus per-pixel depth/normal sums (see `GEOM_OUT_CHANNELS`).
    let geo = render_splats_geometry(splats.clone(), camera, size, background).await;
    let rgba = tensor_vec(geo.img).await?;
    let geom = tensor_vec(geo.geom).await?;
    let channels = brush_render::kernels::helpers::GEOM_OUT_CHANNELS as usize;
    let n = rgba.len() / 4;
    let (mut depth, mut median, mut normal) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for (px, g) in rgba.chunks_exact(4).zip(geom.chunks_exact(channels)) {
        if px[3] > 1e-3 {
            depth.push(g[0] / px[3]);
            median.push(g[1]);
            // View-space sum -> world; camera.rotation maps local to world.
            let nw = (camera.rotation * Vec3::new(g[2], g[3], g[4])).normalize_or_zero();
            normal.push(nw.to_array());
        } else {
            depth.push(f32::NAN);
            median.push(f32::NAN);
            normal.push([0.0; 3]);
        }
    }
    Ok(RawFrame { width: size.x, height: size.y, rgba, depth: Some(depth), median_depth: Some(median), normal: Some(normal) })
}

pub fn to_rgb8(frame: &RawFrame) -> RgbImage {
    let data = frame.rgba.chunks_exact(4).flat_map(|p| [0, 1, 2].map(|c| (p[c].clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
    RgbImage::from_raw(frame.width, frame.height, data).expect("sized buffer")
}

pub fn alpha8(frame: &RawFrame) -> image::GrayImage {
    let data = frame.rgba.chunks_exact(4).map(|p| (p[3].clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    image::GrayImage::from_raw(frame.width, frame.height, data).expect("sized buffer")
}

/// Ground truth for a view exactly as training sees it (resolution cap, alpha handling).
pub async fn load_gt(view: &SceneView) -> anyhow::Result<DynamicImage> {
    Ok(view.image.load().await?)
}

pub fn gt_alpha_mode(view: &SceneView) -> AlphaMode {
    view.image.alpha_mode()
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel as Lens;
    use splat_eval::colmap::{Camera, ImageRecord, write_cameras_bin, write_images_bin};

    /// Upload one tensor of `shape` f32s and read it back. Each value is a hash of its absolute
    /// index (24 bits, exact in f32), so a chunk written or read at the wrong offset, repeated
    /// or reordered is caught (a plain index mod 2^24 repeats every 64 MiB, the chunk size).
    async fn upload_roundtrip(device: &Device, shape: [usize; 3]) {
        use burn::tensor::TensorData;
        let n = shape.iter().product::<usize>();
        let value = |i: usize| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32;
        let data: Vec<f32> = (0..n).map(value).collect();
        let t = Tensor::<3>::from_data(TensorData::new(data, shape), device);
        let back = t.into_data_async().await.unwrap().try_into_vec::<f32>().unwrap();
        assert_eq!(back.len(), n);
        if let Some(i) = back.iter().enumerate().position(|(i, v)| *v != value(i)) {
            panic!("{n} floats: wrong value at {i}: {}", back[i]);
        }
    }

    /// The nyc-apartment `--max` crash: at 5.87M splats a PLY's SH coefficients ([n, 16, 3] f32,
    /// 1.13 GB) went to the GPU as one queue write, which wgpu staged in one host-visible buffer,
    /// and the RTX PRO 6000's driver refused it ("wgpu error: Out of Memory", ~2 GB of 96 GB in
    /// use). Large writes and reads are now split (cubecl-wgpu `MAX_WRITE_CHUNK`). Covers chunk boundaries
    /// and the largest model `--max` allows (8M splats, 1.5 GB), up and back down.
    #[tokio::test]
    #[ignore = "GPU test: run under the shared exclusive GPU flock"]
    async fn large_uploads_are_exact() {
        let device = init_device().await;
        upload_roundtrip(&device, [(64 << 20) / 4 * 2 + 5, 1, 1]).await;
        upload_roundtrip(&device, [8_000_000, 16, 3]).await;
    }

    /// The phone-austin-iphone14-room crash: a mesh goal trains surfels, which render pinhole
    /// cameras only, and video SfM ends with a distorted camera (here the DA3 rescue's
    /// SIMPLE_RADIAL). Surfel jobs load a pinhole copy; the source frames (the frozen split's
    /// bytes) stay untouched. CPU only.
    #[test]
    fn surfel_dataset_is_pinhole_and_leaves_source_frames_alone() {
        assert!(splat_core::remote::goal_wants_mesh("3D splat scene and whole scene mesh"));
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("sfm");
        let (sparse, images) = (src.join("sparse/0"), src.join("images"));
        std::fs::create_dir_all(&sparse).unwrap();
        std::fs::create_dir_all(&images).unwrap();
        let (w, h) = (96u32, 64u32);
        let cam = Camera { id: 1, model: "SIMPLE_RADIAL".into(), width: w as u64, height: h as u64, params: vec![60.0, 48.0, 32.0, 0.02] };
        write_cameras_bin(&sparse.join("cameras.bin"), &[(1, cam)].into_iter().collect()).unwrap();
        let mut records = Vec::new();
        for (i, tx) in [0.0, 0.2, 0.4].into_iter().enumerate() {
            let name = format!("frame_{i:06}.jpg");
            RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 2) as u8, (y * 3) as u8, ((x ^ y) * 4) as u8])).save(images.join(&name)).unwrap();
            records.push(ImageRecord { id: i as u32 + 1, qvec: [1.0, 0.0, 0.0, 0.0], tvec: [tx, 0.0, 5.0], camera_id: 1, name, points2d: vec![([48.0 + 60.0 * tx / 5.0, 32.0], 1)] });
        }
        write_images_bin(&sparse.join("images.bin"), &records).unwrap();
        // points3D.bin: one point at the origin seen by all three images.
        let mut pts = Vec::new();
        pts.extend(1u64.to_le_bytes());
        pts.extend(1u64.to_le_bytes());
        [0.0f64; 3].iter().for_each(|v| pts.extend(v.to_le_bytes()));
        pts.extend([200u8, 100, 50]);
        pts.extend(0.5f64.to_le_bytes());
        pts.extend(3u64.to_le_bytes());
        for i in 1..=3i32 {
            pts.extend(i.to_le_bytes());
            pts.extend(0i32.to_le_bytes());
        }
        std::fs::write(sparse.join("points3D.bin"), pts).unwrap();
        let source_bytes: Vec<Vec<u8>> = records.iter().map(|r| std::fs::read(images.join(&r.name)).unwrap()).collect();

        let load = |dir: &Path| runtime().unwrap().block_on(load_scene(dir, 1600)).unwrap();
        // The radial camera is what reached the surfel renderer.
        assert!(load(&src).views.iter().all(|v| matches!(v.camera.camera_model, Lens::RadialTangential8(_))));
        let scratch = tmp.path().join("pinhole_dataset");
        let (dataset, guard) = surfel_dataset(&src, &scratch).unwrap();
        assert_eq!(dataset, scratch);
        let scene = load(&dataset);
        assert_eq!(scene.views.len(), 3);
        assert!(scene.views.iter().all(|v| matches!(v.camera.camera_model, Lens::Pinhole)));
        for r in &records {
            assert_eq!(image::image_dimensions(scratch.join("images").join(&r.name)).unwrap(), (w, h));
        }
        // Source frames are byte-identical (the frozen split hashes them).
        for (r, bytes) in records.iter().zip(&source_bytes) {
            assert_eq!(&std::fs::read(images.join(&r.name)).unwrap(), bytes, "{} changed", r.name);
        }
        drop(guard);
        assert!(!scratch.exists());
        // A pinhole dataset is used as is.
        let (same, guard) = surfel_dataset(&dataset_copy_pinhole(&tmp, &src), &tmp.path().join("unused")).unwrap();
        assert!(guard.is_none() && same.ends_with("pinhole_src"));
    }

    /// `src` with its camera rewritten as PINHOLE (images unchanged).
    fn dataset_copy_pinhole(tmp: &tempfile::TempDir, src: &Path) -> std::path::PathBuf {
        let dst = tmp.path().join("pinhole_src");
        std::fs::create_dir_all(dst.join("sparse/0")).unwrap();
        for f in ["images.bin", "points3D.bin"] {
            std::fs::copy(src.join("sparse/0").join(f), dst.join("sparse/0").join(f)).unwrap();
        }
        let cam = Camera { id: 1, model: "PINHOLE".into(), width: 96, height: 64, params: vec![60.0, 60.0, 48.0, 32.0] };
        write_cameras_bin(&dst.join("sparse/0/cameras.bin"), &[(1, cam)].into_iter().collect()).unwrap();
        dst
    }
}
