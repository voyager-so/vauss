//! Rendering effect frames through the engine's rasterizer (splat-brush / splat-train-render) and
//! laying them out as contact sheets. The adapter is the machine's GPU (Metal, Vulkan, DX12) when it
//! has one, else the CPU (lavapipe); `SPLAT_EFFECT_DEVICE=cpu` forces the CPU (see [`gpu_allowed`]).

use std::sync::OnceLock;

use anyhow::{Context, Result};
use glam::Vec3;
use image::{Rgb, RgbImage};
use splat_brush::render::{RawSplats, RenderCamera, Renderer};
use splat_webexport::ply::Gaussians;

use super::scene::Cam;

const C0: f32 = 0.282_094_8;
const LAVAPIPE: &str = "/usr/share/vulkan/icd.d/lvp_icd.json";

/// Whether effect renders run on a hardware GPU, resolved once per process:
/// - `SPLAT_EFFECT_DEVICE=cpu`: the CPU (software) adapter only. Our dev hosts set this: their GPU is off limits.
/// - `SPLAT_EFFECT_DEVICE=gpu` or `SPLAT_EFFECT_GPU=1` (fal jobs; `--cloud` sets it): the default GPU adapter.
/// - unset / `auto`: the best hardware adapter (Metal on macOS, a discrete or integrated GPU on
///   Windows/Linux), falling back to the CPU adapter only when there is none.
pub fn gpu_allowed() -> bool {
    static GPU: OnceLock<bool> = OnceLock::new();
    *GPU.get_or_init(|| {
        if std::env::var("SPLAT_EFFECT_GPU").is_ok_and(|v| v == "1") {
            return true;
        }
        match std::env::var("SPLAT_EFFECT_DEVICE").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
            "cpu" => false,
            "gpu" => true,
            "" | "auto" => Renderer::has_hardware_adapter().unwrap_or(false),
            o => {
                eprintln!("[effect] SPLAT_EFFECT_DEVICE={o:?} is not cpu, gpu or auto; using auto");
                Renderer::has_hardware_adapter().unwrap_or(false)
            }
        }
    })
}

/// Pin wgpu to the CPU adapter unless effect renders run on a GPU ([`gpu_allowed`]). Call before any GPU init.
pub fn pin_adapter() -> &'static str {
    if gpu_allowed() {
        return "gpu";
    }
    // Single-threaded here, before the runtime starts: Vulkan sees lavapipe only.
    unsafe {
        if std::path::Path::new(LAVAPIPE).exists() {
            std::env::set_var("VK_ICD_FILENAMES", LAVAPIPE);
        }
        std::env::set_var("CUBECL_WGPU_DEFAULT_DEVICE", "Cpu");
    }
    "cpu-lavapipe"
}

/// Which colours the splats carry in a render pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pass {
    Rgb,
    /// Node weight (mask × curve) per splat: grey 0 → orange 1.
    Mask,
    /// |μ' − μ| over the sheet's largest displacement: black → blue → green → yellow → red.
    Displacement,
}

impl Pass {
    pub fn parse(s: &str) -> Result<Pass> {
        Ok(match s.trim() {
            "rgb" | "beauty" => Pass::Rgb,
            "mask" | "selection" => Pass::Mask,
            "displacement" | "motion" | "disp" => Pass::Displacement,
            o => anyhow::bail!("unknown pass {o:?} (rgb, mask, displacement)"),
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Pass::Rgb => "rgb",
            Pass::Mask => "mask",
            Pass::Displacement => "displacement",
        }
    }
}

fn dc_of(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| (v - 0.5) / C0)
}

fn heat(x: f32) -> [f32; 3] {
    let x = x.clamp(0.0, 1.0);
    let stops = [[0.05, 0.05, 0.08], [0.1, 0.25, 0.9], [0.1, 0.8, 0.3], [0.95, 0.9, 0.1], [0.95, 0.15, 0.05]];
    let f = x * 4.0;
    let i = (f as usize).min(3);
    let u = f - i as f32;
    [0, 1, 2].map(|k| stops[i][k] * (1.0 - u) + stops[i + 1][k] * u)
}

/// lavapipe's maxStorageBufferRange: one storage buffer holds every splat's SH.
const CPU_MAX_BUFFER: usize = 128 << 20;

/// SH degree the renderer draws `g` with: full on a GPU; on the CPU adapter the highest degree whose
/// SH buffer fits its storage-buffer limit (degree 3 fits ~700K splats, 1 fits ~2.8M).
pub fn render_sh_degree(g: &Gaussians) -> u32 {
    if gpu_allowed() {
        return g.sh_degree;
    }
    (0..=g.sh_degree).rev().find(|d| g.len() * ((d + 1) * (d + 1)) as usize * 12 <= CPU_MAX_BUFFER).unwrap_or(0)
}

/// Raw arrays for the rasterizer. `colors` replaces the appearance (flat DC only) for debug passes.
pub fn raw(g: &Gaussians, colors: Option<&[[f32; 3]]>) -> RawSplats {
    let n = g.len();
    let sh_coeffs = match colors {
        Some(c) => c.iter().flat_map(|c| dc_of(*c)).collect(),
        None => {
            let rest = g.sh_coeffs();
            let d = render_sh_degree(g);
            let keep = ((d + 1) * (d + 1) - 1) as usize;
            let mut out = Vec::with_capacity(n * (keep + 1) * 3);
            for i in 0..n {
                out.extend_from_slice(&g.dc[i]);
                out.extend_from_slice(&g.rest[i * rest * 3..(i * rest + keep) * 3]);
            }
            out
        }
    };
    RawSplats {
        means: g.pos.iter().flatten().copied().collect(),
        rotations: g.rot.iter().flatten().copied().collect(),
        log_scales: g.scale.iter().flatten().copied().collect(),
        sh_coeffs,
        raw_opacities: g.opacity.clone(),
        mip: g.antialiased,
    }
}

pub fn pass_colors(pass: Pass, weight: &[f32], disp: &[f32], disp_max: f32) -> Option<Vec<[f32; 3]>> {
    match pass {
        Pass::Rgb => None,
        Pass::Mask => Some(weight.iter().map(|w| {
            let w = w.clamp(0.0, 1.0);
            [0.22 + 0.76 * w, 0.22 + 0.33 * w, 0.24 - 0.2 * w]
        }).collect()),
        Pass::Displacement => Some(disp.iter().map(|d| heat(d / disp_max.max(1e-9))).collect()),
    }
}

/// One renderer per process (the GPU runtime is set up once); splats are swapped per frame.
pub struct Raster {
    r: Option<Renderer>,
}

impl Raster {
    pub fn new() -> Self {
        Self { r: None }
    }

    pub fn render(&mut self, splats: RawSplats, cam: &Cam, up: Vec3, w: u32, h: u32) -> Result<RgbImage> {
        match &mut self.r {
            Some(r) => r.set_raw(splats),
            None => self.r = Some(Renderer::from_raw(splats, !gpu_allowed())?),
        }
        self.render_again(cam, up, w, h)
    }

    /// The splats of the last `render` from another camera.
    pub fn render_again(&mut self, cam: &Cam, up: Vec3, w: u32, h: u32) -> Result<RgbImage> {
        let r = self.r.as_ref().context("render_again before render")?;
        let c = RenderCamera::look_at(cam.position, cam.look_at, up, (cam.vfov_deg as f64).to_radians(), w, h);
        Ok(r.render(&c, false)?.rgb)
    }
}

/// An effect evaluated on the renderer's device by splat-effects' WGSL evaluator: the rest splats
/// go up once, each frame is one dispatch whose output the rasterizer reads in place.
pub struct GpuFx {
    fx: splat_effects::gpu::GpuEffect,
    rest: splat_brush::render::GpuSplats,
}

impl Raster {
    pub fn gpu_effect(&mut self, rest: &Gaussians, graph: &splat_effects::Graph, opts: &splat_effects::EvalOpts) -> Result<GpuFx> {
        let raw = raw(rest, None);
        match &mut self.r {
            Some(r) => r.set_raw(raw),
            None => self.r = Some(Renderer::from_raw(raw, !gpu_allowed())?),
        }
        let rest = self.r.as_ref().unwrap().splats().clone();
        let fx = splat_effects::gpu::GpuEffect::new(&rest, graph, opts)?;
        Ok(GpuFx { fx, rest })
    }

    /// Evaluate `t` on the device and draw it next; returns `(invalid, selected)` (two numbers read back).
    pub fn gpu_frame(&mut self, g: &GpuFx, t: f32) -> (usize, usize, splat_effects::gpu::GpuFrame) {
        let f = g.fx.frame(&g.rest, t);
        let counts = f.counts();
        self.r.as_mut().unwrap().set_splats(f.splats.clone());
        (counts.0, counts.1, f)
    }
}

/// Grid of tiles with a header row (column labels) and a label strip over each tile.
pub fn sheet(tiles: &[Vec<RgbImage>], col_labels: &[String], row_labels: &[String]) -> RgbImage {
    let (tw, th) = tiles.first().and_then(|r| r.first()).map(|t| t.dimensions()).unwrap_or((1, 1));
    let (pad, head, side) = (4u32, 26u32, 150u32);
    let cols = tiles.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let rows = tiles.len() as u32;
    let mut out = RgbImage::from_pixel(side + cols * (tw + pad) + pad, head + rows * (th + pad) + pad, Rgb([18, 18, 22]));
    for (c, l) in col_labels.iter().enumerate() {
        text(&mut out, side + pad + c as u32 * (tw + pad) + 4, 6, l, 2, Rgb([235, 235, 235]));
    }
    for (r, row) in tiles.iter().enumerate() {
        let y = head + r as u32 * (th + pad);
        if let Some(l) = row_labels.get(r) {
            for (k, part) in l.split(' ').enumerate() {
                text(&mut out, 6, y + 6 + k as u32 * 18, part, 2, Rgb([235, 235, 235]));
            }
        }
        for (c, t) in row.iter().enumerate() {
            image::imageops::replace(&mut out, t, (side + pad + c as u32 * (tw + pad)) as i64, y as i64);
        }
    }
    out
}

/// 5x7 bitmap text (lowercase, digits, a little punctuation), `scale` px per dot.
pub fn text(img: &mut RgbImage, x: u32, y: u32, s: &str, scale: u32, color: Rgb<u8>) {
    let mut cx = x;
    for ch in s.chars() {
        let g = glyph(ch.to_ascii_lowercase());
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if bits >> (4 - col) & 1 == 1 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let (px, py) = (cx + col * scale + dx, y + row as u32 * scale + dy);
                            if px < img.width() && py < img.height() {
                                img.put_pixel(px, py, color);
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'a' => [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
        'b' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E],
        'c' => [0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E],
        'd' => [0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F],
        'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        'f' => [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08],
        'g' => [0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        'h' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11],
        'i' => [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
        'j' => [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],
        'k' => [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
        'l' => [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
        'n' => [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
        'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
        'p' => [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
        'q' => [0x00, 0x00, 0x0D, 0x13, 0x0F, 0x01, 0x01],
        'r' => [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
        's' => [0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E],
        't' => [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],
        'u' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],
        'v' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'w' => [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],
        'x' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
        'y' => [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        'z' => [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '/' => [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        _ => [0; 7],
    }
}
