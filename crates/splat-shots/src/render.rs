//! Render a CameraPath from a splat PLY to RGB / depth / alpha frames and browser-playable videos.
//!
//! Engine: splat-brush's renderer (wgpu, any GPU).

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc::sync_channel;
use std::time::Instant;

use anyhow::{Context, Result};
use image::{GrayImage, ImageBuffer, Luma, RgbImage};

use crate::encode::Ffmpeg;
use crate::layout::RenderMeta;
use crate::paths::{CameraPath, PathFrame, sample_path};
use crate::scene::quantile;

/// One rendered view. `depth` is expected view-space z, 0 where nothing was hit.
pub struct Frame {
    pub rgb: RgbImage,
    pub alpha: Vec<f32>,
    pub depth: Vec<f32>,
}

/// Something that renders a splat from an OpenCV camera, one frame at a time.
pub trait FrameRenderer {
    fn engine(&self) -> &'static str;
    fn render(&mut self, frame: &PathFrame, width: u32, height: u32) -> Result<Frame>;
}

/// Renders every frame of `path`, writes `out_dir/{rgb,alpha,depth}/` + videos + render.json.
/// `gpu_guard` (the shared GPU lock) is dropped as soon as the raster pass ends: PNG writing and
/// encoding run without it.
pub fn render_with(
    renderer: &mut dyn FrameRenderer,
    path: &CameraPath,
    out_dir: &Path,
    gpu_guard: Option<File>,
    ff: &Ffmpeg,
) -> Result<RenderMeta> {
    let t0 = Instant::now();
    let frames = sample_path(path);
    let (w, h) = (path.width, path.height);
    for sub in ["rgb", "alpha", "depth"] {
        std::fs::create_dir_all(out_dir.join(sub))?;
    }
    // Depth is normalised over the whole shot (per-frame normalisation makes geometry pulse), so
    // pass 1 spools raw depth to a temp file and samples it for the near/far quantiles.
    let spool_path = out_dir.join("depth.spool");
    let mut spool = BufWriter::new(File::create(&spool_path)?);
    let (mut samples, mut alpha_means) = (Vec::new(), Vec::new());
    let mut raster_s = 0.0;
    // PNG encoding is slower than rasterising; a few writer threads keep the GPU loop short.
    let (tx, rx) = sync_channel::<(usize, RgbImage, GrayImage)>(8);
    let rx = Mutex::new(rx);
    std::thread::scope(|s| -> Result<()> {
        let writers: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| -> Result<()> {
                    while let Ok((i, rgb, a)) = rx.lock().unwrap().recv() {
                        rgb.save(out_dir.join("rgb").join(format!("{i:05}.png")))?;
                        a.save(out_dir.join("alpha").join(format!("{i:05}.png")))?;
                    }
                    Ok(())
                })
            })
            .collect();
        let raster = (|| -> Result<()> {
            for (i, f) in frames.iter().enumerate() {
                let fr = renderer.render(f, w, h)?;
                let a: GrayImage =
                    ImageBuffer::from_fn(w, h, |x, y| Luma([(fr.alpha[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0) as u8]));
                alpha_means.push(fr.alpha.iter().map(|v| *v as f64).sum::<f64>() / fr.alpha.len() as f64);
                samples.extend(fr.depth.iter().step_by(61).filter(|d| **d > 0.0).map(|d| *d as f64));
                for d in &fr.depth {
                    spool.write_all(&d.to_le_bytes())?;
                }
                if tx.send((i, fr.rgb, a)).is_err() {
                    break; // a writer failed; its error is reported below
                }
            }
            Ok(())
        })();
        raster_s = t0.elapsed().as_secs_f64();
        drop(gpu_guard);
        drop(tx);
        for wr in writers {
            wr.join().expect("png writer panicked")?;
        }
        raster
    })?;
    spool.flush()?;
    drop(spool);
    let (near, far) = if samples.is_empty() { (1.0, 2.0) } else { (quantile(&mut samples, 0.01), quantile(&mut samples, 0.99)) };
    let mut rd = BufReader::new(File::open(&spool_path)?);
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for i in 0..frames.len() {
        rd.read_exact(&mut buf)?;
        let img: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_fn(w, h, |x, y| {
            let k = ((y * w + x) * 4) as usize;
            let d = f32::from_le_bytes(buf[k..k + 4].try_into().unwrap()) as f64;
            let inv = if d > 0.0 { 1.0 / d.clamp(near, far) } else { 1.0 / far };
            Luma([(((inv - 1.0 / far) / (1.0 / near - 1.0 / far)) * 65535.0).round() as u16])
        });
        img.save(out_dir.join("depth").join(format!("{i:05}.png")))?;
    }
    std::fs::remove_file(&spool_path)?;
    finish(ff, out_dir, path, RenderMeta {
        frames: frames.len(),
        width: w,
        height: h,
        engine: renderer.engine().into(),
        depth_near: near,
        depth_far: far,
        alpha_mean: alpha_means.iter().sum::<f64>() / alpha_means.len().max(1) as f64,
        alpha_min_frame: alpha_means.iter().copied().fold(f64::INFINITY, f64::min),
        render_s: raster_s,
    })
}

fn finish(ff: &Ffmpeg, out_dir: &Path, path: &CameraPath, meta: RenderMeta) -> Result<RenderMeta> {
    encode_passes(ff, out_dir, path.fps)?;
    std::fs::write(out_dir.join("render.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(meta)
}

/// Encode the rgb/depth/alpha frame dirs under `out_dir` to `<pass>.{mp4|webm}`.
pub fn encode_passes(ff: &Ffmpeg, out_dir: &Path, fps: u32) -> Result<()> {
    for (pass, gray) in [("rgb", false), ("depth", true), ("alpha", true)] {
        ff.encode_video(&out_dir.join(pass), fps, &out_dir.join(pass), gray)?;
    }
    Ok(())
}

struct Brush(splat_brush::render::Renderer);

impl FrameRenderer for Brush {
    fn engine(&self) -> &'static str {
        "brush"
    }

    fn render(&mut self, f: &PathFrame, width: u32, height: u32) -> Result<Frame> {
        let c2w = f.c2w.as_mat4();
        let cam = splat_brush::render::RenderCamera::from_c2w(c2w, f.vfov_deg.to_radians(), width, height);
        let fr = self.0.render(&cam, true)?;
        Ok(Frame { rgb: fr.rgb, alpha: fr.alpha, depth: fr.depth })
    }
}

pub fn render_path(ply: &Path, path: &CameraPath, out_dir: &Path, ff: &Ffmpeg) -> Result<RenderMeta> {
    // The shared dev-host GPU lock; absent elsewhere.
    let guard = splat_brush::gpu_lock(splat_contracts::dev::gpu_lock().as_deref())?;
    let mut r = Brush(splat_brush::render::Renderer::open(ply).with_context(|| format!("loading {}", ply.display()))?);
    render_with(&mut r, path, out_dir, guard, ff)
}

/// Render several paths of one splat with a single renderer (the GPU runtime can only be set up
/// once per process). The GPU lock is taken per path and dropped after each raster pass.
pub fn render_paths(ply: &Path, jobs: &[(&CameraPath, &Path)], ff: &Ffmpeg) -> Result<Vec<RenderMeta>> {
    let lock = splat_contracts::dev::gpu_lock();
    let mut r: Option<Brush> = None;
    let mut metas = vec![];
    for (path, out_dir) in jobs {
        let guard = splat_brush::gpu_lock(lock.as_deref())?;
        if r.is_none() {
            r = Some(Brush(splat_brush::render::Renderer::open(ply).with_context(|| format!("loading {}", ply.display()))?));
        }
        metas.push(render_with(r.as_mut().unwrap(), path, out_dir, guard, ff)?);
    }
    Ok(metas)
}
