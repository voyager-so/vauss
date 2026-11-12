//! On-disk layout of one shot (same as the Python reference):
//!   path.json  risk.json  reference.<ext>  shot.json
//!   render/{rgb,depth,alpha}/%05d.png  render/{rgb,depth,alpha}.{mp4|webm}  render/render.json
//!   gen/<model>.{mp4,json,request.json}  gen/rgb.fal.mp4 (model input)  blender/

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths::CameraPath;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderMeta {
    pub frames: usize,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub engine: String,
    /// Depth PNGs hold disparity normalised over the whole shot: v = (1/d - 1/far) / (1/near - 1/far).
    pub depth_near: f64,
    pub depth_far: f64,
    pub alpha_mean: f64,
    pub alpha_min_frame: f64,
    /// GPU raster time (the part that holds the GPU lock).
    pub render_s: f64,
}

pub struct ShotDir(pub PathBuf);

impl ShotDir {
    pub fn path(&self) -> Result<CameraPath> {
        let p = self.0.join("path.json");
        serde_json::from_slice(&std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?)
            .with_context(|| format!("parsing {}", p.display()))
    }
    pub fn render_dir(&self) -> PathBuf {
        self.0.join("render")
    }
    pub fn gen_dir(&self) -> PathBuf {
        self.0.join("gen")
    }
    /// Encoded video of a render pass: `render/<pass>.mp4` (H.264) or `.webm` (VP9), whichever the
    /// encoder wrote (the newer one if a shot somehow holds both).
    pub fn video(&self, pass: &str) -> PathBuf {
        let stem = self.render_dir().join(pass);
        let (mp4, webm) = (stem.with_extension("mp4"), stem.with_extension("webm"));
        match (mp4.exists(), webm.exists()) {
            (true, true) if crate::encode::newer_than(&mp4, &webm) => mp4,
            (_, true) => webm,
            _ => mp4,
        }
    }
    pub fn rgb_video(&self) -> PathBuf {
        self.video("rgb")
    }
    pub fn depth_video(&self) -> PathBuf {
        self.video("depth")
    }
    pub fn frame(&self, pass: &str, i: usize) -> PathBuf {
        self.render_dir().join(pass).join(format!("{i:05}.png"))
    }
    pub fn render_meta(&self) -> Result<RenderMeta> {
        Ok(serde_json::from_slice(&std::fs::read(self.render_dir().join("render.json"))?)?)
    }
    pub fn reference(&self) -> Option<PathBuf> {
        std::fs::read_dir(&self.0)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.file_stem().is_some_and(|s| s == "reference"))
    }
}
