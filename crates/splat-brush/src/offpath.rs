//! Off-path view check: fixed cameras a short step off the capture path (0.5 m sideways, 30°
//! turned) rendered for one or two trained models, written as a labelled side-by-side contact
//! sheet with simple no-reference metrics per view.
//!
//! Views: K anchor frames spread evenly along the capture (registered image names in order; a
//! seed shifts the phase), and for each the anchor itself plus four variants: shifted
//! `shift_m` left/right along the camera's right axis, and turned `yaw_deg` left/right about the
//! camera's own up axis (position kept). Only the SfM model (sparse/0) is read, never images, so
//! two arms on the same SfM get identical views; an arm on another SfM takes each anchor's frame
//! by name (or the nearest frame number) in its own model.
//!
//! Metric scale: SfM units are arbitrary. Unless given, units per metre = the median over
//! frames of each frame's median SfM-point depth, divided by `room_depth_m` (2.5 m: a phone
//! held in a room mostly sees walls/furniture 2–3 m away). Each arm uses its own model's scale.
//!
//! Metrics per rendered view (no reference image exists off the path):
//! - `sharpness`: Laplacian variance of the grey image (as the capture pre-check scores frames);
//! - `hole_frac`: pixels with accumulated opacity < 0.5 (uncovered, see-through);
//! - `depth_edge_frac`: of covered neighbour pairs, those whose rendered depth jumps > 10 %
//!   (floaters and smeared shells break up surfaces; a clean wall is smooth);
//! - `near_frac`: covered pixels rendered nearer than `near_m` (floaters hanging in front of the
//!   camera).
//!
//! Each view's rendered depth is drawn under its colour tiles, colour-mapped in metres on one
//! range per view shared by all arms (pooled 2nd–98th percentile; near = yellow, far = purple,
//! uncovered = black), so a filmed wall should read as a smooth gradient in every arm.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use glam::{DQuat, Quat, Vec3};
use image::{RgbImage, imageops};
use serde::Serialize;
use serde_json::{Value, json};

use crate::glyphs::{draw_text, text_width};
use crate::render::{Frame, RenderCamera, Renderer};
use crate::util::{read_json, write_json};

#[derive(Debug, Clone)]
pub struct Arm {
    pub label: String,
    pub ply: PathBuf,
    /// COLMAP sparse model dir (cameras/images/points3D).
    pub sparse: PathBuf,
    /// Render-time diagnostics (`;sh=N` / `;f3d=X` after the arm spec): cap the SH degree
    /// (view-dependent overfit check) and fold a Mip-Splatting 3D filter of strength X built
    /// from the capture cameras (aliasing check).
    pub sh_degree: Option<u32>,
    pub filter3d: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub anchors: usize,
    pub seed: u64,
    pub shift_m: f32,
    pub yaw_deg: f32,
    pub room_depth_m: f32,
    /// Overrides the depth-based scale estimate (same value for every arm).
    pub units_per_metre: Option<f32>,
    pub near_m: f32,
    /// Long-edge cap of the rendered views.
    pub max_resolution: u32,
    /// Width of one tile on the per-anchor sheets.
    pub tile_width: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self { anchors: 6, seed: 0, shift_m: 0.5, yaw_deg: 30.0, room_depth_m: 2.5, units_per_metre: None, near_m: 0.3, max_resolution: 960, tile_width: 640 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Anchor,
    Left,
    Right,
    YawLeft,
    YawRight,
}

impl Variant {
    pub const ALL: [Variant; 5] = [Variant::Anchor, Variant::Left, Variant::Right, Variant::YawLeft, Variant::YawRight];

    pub fn name(self) -> &'static str {
        match self {
            Variant::Anchor => "anchor",
            Variant::Left => "left",
            Variant::Right => "right",
            Variant::YawLeft => "yaw_left",
            Variant::YawRight => "yaw_right",
        }
    }

    fn caption(self, o: &Options) -> String {
        match self {
            Variant::Anchor => "ON PATH".into(),
            Variant::Left => format!("{} M LEFT", o.shift_m),
            Variant::Right => format!("{} M RIGHT", o.shift_m),
            Variant::YawLeft => format!("TURN {} LEFT", o.yaw_deg),
            Variant::YawRight => format!("TURN {} RIGHT", o.yaw_deg),
        }
    }
}

/// The off-path camera: translated along the camera's right axis (+x, OpenCV) by `shift` SfM
/// units, or turned about its own up axis (-y) by `yaw` radians; positive turns to the right.
pub fn variant_camera(cam: &RenderCamera, v: Variant, shift: f32, yaw: f32) -> RenderCamera {
    let right = cam.rotation * Vec3::X;
    let mut out = *cam;
    match v {
        Variant::Anchor => {}
        Variant::Left => out.position -= right * shift,
        Variant::Right => out.position += right * shift,
        // Rotating about camera +y (image down) by +a swings forward (+z) toward +x (right).
        Variant::YawLeft => out.rotation = (cam.rotation * Quat::from_rotation_y(-yaw)).normalize(),
        Variant::YawRight => out.rotation = (cam.rotation * Quat::from_rotation_y(yaw)).normalize(),
    }
    out
}

/// A COLMAP model's registered frames as pinhole render cameras (distortion dropped, principal
/// point kept) at the long-edge cap, sorted by name, plus each frame's median SfM-point depth.
pub struct PathCameras {
    pub frames: Vec<(String, RenderCamera)>,
    pub median_depths: Vec<f32>,
}

pub fn load_cameras(sparse: &Path, max_resolution: u32) -> anyhow::Result<PathCameras> {
    let model = splat_eval::read_model(sparse).with_context(|| format!("reading SfM model {}", sparse.display()))?;
    let points: HashMap<u64, [f64; 3]> = model.points.iter().map(|p| (p.id, p.xyz)).collect();
    let mut images: Vec<_> = model.images.values().collect();
    images.sort_by(|a, b| a.name.cmp(&b.name));
    let (mut frames, mut median_depths) = (vec![], vec![]);
    for im in images {
        let cam = model.cameras.get(&im.camera_id).with_context(|| format!("{}: camera {} missing", im.name, im.camera_id))?;
        let (fx, fy, cx, cy) = pinhole(&cam.model, &cam.params).with_context(|| format!("{}: camera model {}", im.name, cam.model))?;
        let s = (max_resolution as f64 / cam.width.max(cam.height) as f64).min(1.0);
        let (w, h) = (((cam.width as f64 * s).round() as u32).max(1), ((cam.height as f64 * s).round() as u32).max(1));
        let w2c = DQuat::from_xyzw(im.qvec[1], im.qvec[2], im.qvec[3], im.qvec[0]).normalize();
        let t = glam::DVec3::from_array(im.tvec);
        let c2w = w2c.inverse();
        let position = -(c2w * t);
        let sx = w as f64 / cam.width as f64;
        let sy = h as f64 / cam.height as f64;
        frames.push((im.name.clone(), RenderCamera {
            position: position.as_vec3(),
            rotation: c2w.as_quat().normalize(),
            fx: (fx * sx) as f32,
            fy: (fy * sy) as f32,
            cx: (cx * sx) as f32,
            cy: (cy * sy) as f32,
            width: w,
            height: h,
        }));
        let mut z: Vec<f64> = im
            .obs_point_id
            .iter()
            .filter_map(|id| points.get(id))
            .map(|p| (w2c * glam::DVec3::from_array(*p) + t).z)
            .filter(|z| *z > 0.0)
            .collect();
        z.sort_by(f64::total_cmp);
        median_depths.push(z.get(z.len() / 2).copied().unwrap_or(f64::NAN) as f32);
    }
    if frames.is_empty() {
        bail!("{}: no registered frames", sparse.display());
    }
    Ok(PathCameras { frames, median_depths })
}

/// (fx, fy, cx, cy) of a COLMAP camera; distortion parameters are ignored.
fn pinhole(model: &str, p: &[f64]) -> Option<(f64, f64, f64, f64)> {
    match model {
        "SIMPLE_PINHOLE" | "SIMPLE_RADIAL" | "RADIAL" | "SIMPLE_RADIAL_FISHEYE" | "RADIAL_FISHEYE" | "FOV" if p.len() >= 3 => Some((p[0], p[0], p[1], p[2])),
        _ if p.len() >= 4 => Some((p[0], p[1], p[2], p[3])),
        _ => None,
    }
}

/// SfM units per metre: median frame depth over the assumed room depth.
pub fn units_per_metre(median_depths: &[f32], room_depth_m: f32) -> anyhow::Result<f32> {
    let mut d: Vec<f32> = median_depths.iter().copied().filter(|d| d.is_finite() && *d > 0.0).collect();
    if d.is_empty() {
        bail!("no SfM point depths to estimate the scene scale from (pass --units-per-metre)");
    }
    d.sort_by(f32::total_cmp);
    Ok(d[d.len() / 2] / room_depth_m)
}

/// `k` names spread evenly over `names` (in order), the phase fixed by `seed`.
pub fn pick_anchors(names: &[String], k: usize, seed: u64) -> Vec<String> {
    let n = names.len();
    let k = k.min(n);
    if k == 0 {
        return vec![];
    }
    // splitmix64 -> a phase in [0, 1) of one stride.
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let phase = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
    let stride = n as f64 / k as f64;
    (0..k).map(|i| names[(((i as f64 + phase) * stride) as usize).min(n - 1)].clone()).collect()
}

/// Split a frame name into (non-digit prefix, trailing number), e.g. "frame_000033.jpg" -> ("frame_", 33).
fn frame_number(name: &str) -> Option<(&str, u64)> {
    let stem = name.rsplit_once('.').map_or(name, |(s, _)| s);
    let digits = stem.len() - stem.chars().rev().take_while(char::is_ascii_digit).count();
    let n = stem[digits..].parse().ok()?;
    Some((&stem[..digits], n))
}

/// `target` itself if registered, else the registered frame with the same prefix and the nearest
/// frame number (ties: the earlier one).
pub fn match_name<'a>(target: &str, names: &'a [String]) -> Option<&'a String> {
    if let Some(n) = names.iter().find(|n| *n == target) {
        return Some(n);
    }
    let (prefix, num) = frame_number(target)?;
    names
        .iter()
        .filter_map(|n| frame_number(n).filter(|(p, _)| *p == prefix).map(|(_, k)| (k.abs_diff(num), k, n)))
        .min_by_key(|&(d, k, _)| (d, k))
        .map(|(_, _, n)| n)
}

#[derive(Debug, Clone, Serialize)]
pub struct ViewMetrics {
    pub sharpness: f32,
    pub hole_frac: f32,
    pub depth_edge_frac: f32,
    pub near_frac: f32,
    /// Sharpness of the nearest capture photo (by position and viewing direction) at the view's
    /// size, when the arm's SfM dir has its images: how sharp the input is where this view looks.
    pub source_sharpness: f32,
    pub source_frame: String,
}

/// See the module docs. `near` is in SfM units; depth 0 marks an uncovered pixel.
pub fn view_metrics(frame: &Frame, near: f32) -> ViewMetrics {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let n = (w * h).max(1) as f32;
    let hole_frac = frame.alpha.iter().filter(|a| **a < 0.5).count() as f32 / n;
    let d = &frame.depth;
    let (mut pairs, mut jumps, mut near_px) = (0usize, 0usize, 0usize);
    if d.len() == w * h {
        let covered = |i: usize| d[i] > 0.0 && frame.alpha[i] >= 0.5;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if !covered(i) {
                    continue;
                }
                if d[i] < near {
                    near_px += 1;
                }
                for j in [(x + 1 < w).then_some(i + 1), (y + 1 < h).then_some(i + w)].into_iter().flatten() {
                    if covered(j) {
                        pairs += 1;
                        if (d[i] - d[j]).abs() > 0.1 * d[i].min(d[j]) {
                            jumps += 1;
                        }
                    }
                }
            }
        }
    }
    ViewMetrics {
        sharpness: splat_eval::frameplan::sharpness(&frame.rgb),
        hole_frac,
        depth_edge_frac: jumps as f32 / pairs.max(1) as f32,
        near_frac: near_px as f32 / n,
        source_sharpness: 0.0,
        source_frame: String::new(),
    }
}

/// The capture frame nearest `cam` (metres apart plus the turn in radians) and its photo's
/// sharpness at the view's size (cached per photo and size).
fn source_sharpness(
    images: &Path,
    frames: &[(String, RenderCamera)],
    cam: &RenderCamera,
    upm: f32,
    cache: &mut HashMap<(PathBuf, u32, u32), f32>,
) -> anyhow::Result<(String, f32)> {
    let cost = |c: &RenderCamera| c.position.distance(cam.position) / upm + c.forward().angle_between(cam.forward());
    let (name, _) = frames.iter().min_by(|a, b| cost(&a.1).total_cmp(&cost(&b.1))).context("no capture frames")?;
    let path = images.join(name);
    if !path.is_file() {
        return Ok((name.clone(), 0.0));
    }
    let key = (path.clone(), cam.width, cam.height);
    if let Some(s) = cache.get(&key) {
        return Ok((name.clone(), *s));
    }
    let img = image::open(&path).with_context(|| format!("reading {}", path.display()))?
        .resize_exact(cam.width, cam.height, imageops::FilterType::Triangle)
        .to_rgb8();
    let s = splat_eval::frameplan::sharpness(&img);
    cache.insert(key, s);
    Ok((name.clone(), s))
}

/// Mip-Splatting 3D filter for a trained model: per splat `sqrt(factor) * min over capture
/// cameras of distance / focal` (world size of one pixel at the nearest camera; the render
/// cameras' focal lengths).
fn filter3d_floor(r: &Renderer, frames: &[(String, RenderCamera)], factor: f32) -> anyhow::Result<burn::tensor::Tensor<1>> {
    let means = r.splats().means();
    let device = means.device();
    let means: Vec<f32> = r.block_on(means.into_data_async())?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let cams: Vec<(Vec3, f32)> = frames.iter().map(|(_, c)| (c.position, c.fx.max(1e-6))).collect();
    let k = factor.max(0.0).sqrt();
    let floor: Vec<f32> = means
        .chunks_exact(3)
        .map(|m| {
            let p = Vec3::new(m[0], m[1], m[2]);
            k * cams.iter().map(|(c, f)| p.distance(*c) / f).fold(f32::INFINITY, f32::min)
        })
        .collect();
    Ok(burn::tensor::Tensor::from_data(burn::tensor::TensorData::new(floor, [means.len() / 3]), &device))
}

/// One planned view: an anchor frame and a variant, with each arm's camera.
struct View {
    anchor: String,
    variant: Variant,
    /// Per arm: (frame name used in that arm's SfM, camera).
    cams: Vec<(String, RenderCamera)>,
}

fn camera_json(c: &RenderCamera) -> Value {
    json!({"position": c.position.to_array(), "rotation_xyzw_c2w": c.rotation.to_array(), "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
           "width": c.width, "height": c.height})
}

/// Render the off-path views for every arm into `out`; returns (and writes) out/offpath.json.
pub fn run(arms: &[Arm], o: &Options, out: &Path) -> anyhow::Result<Value> {
    if arms.is_empty() || arms.len() > 4 {
        bail!("give one to four arms");
    }
    std::fs::create_dir_all(out)?;
    let mut paths = vec![];
    let mut scales = vec![];
    for arm in arms {
        let pc = load_cameras(&arm.sparse, o.max_resolution)?;
        let upm = match o.units_per_metre {
            Some(u) => u,
            None => units_per_metre(&pc.median_depths, o.room_depth_m)?,
        };
        scales.push(upm);
        paths.push(pc);
    }
    let names0: Vec<String> = paths[0].frames.iter().map(|(n, _)| n.clone()).collect();
    let anchors = pick_anchors(&names0, o.anchors, o.seed);
    let yaw = o.yaw_deg.to_radians();
    let mut views = vec![];
    for anchor in &anchors {
        let mut base = vec![];
        for pc in &paths {
            let names: Vec<String> = pc.frames.iter().map(|(n, _)| n.clone()).collect();
            let m = match_name(anchor, &names).with_context(|| format!("anchor {anchor}: no matching frame in an arm's SfM"))?;
            let cam = pc.frames.iter().find(|(n, _)| n == m).map(|(_, c)| *c).expect("matched name is registered");
            base.push((m.clone(), cam));
        }
        for v in Variant::ALL {
            let cams = base.iter().zip(&scales).map(|((n, c), upm)| (n.clone(), variant_camera(c, v, o.shift_m * upm, yaw))).collect();
            views.push(View { anchor: anchor.clone(), variant: v, cams });
        }
    }

    // Render arm by arm (one model on the GPU at a time).
    let mut metrics: Vec<Vec<ViewMetrics>> = vec![];
    let mut tiles: Vec<Vec<RgbImage>> = vec![];
    // Per arm, per view: rendered depth in metres (0 = uncovered).
    let mut depths: Vec<Vec<Vec<f32>>> = vec![];
    // One renderer (one GPU device per process); each arm's PLY replaces the previous one.
    let mut renderer: Option<Renderer> = None;
    let mut source_cache: HashMap<(PathBuf, u32, u32), f32> = HashMap::new();
    for (ai, arm) in arms.iter().enumerate() {
        let dir = out.join("renders").join(sanitize(&arm.label));
        std::fs::create_dir_all(&dir)?;
        match renderer.as_mut() {
            None => renderer = Some(Renderer::open(&arm.ply).with_context(|| format!("loading {}", arm.ply.display()))?),
            Some(r) => r.load(&arm.ply).with_context(|| format!("loading {}", arm.ply.display()))?,
        }
        let r = renderer.as_mut().expect("loaded");
        if let Some(d) = arm.sh_degree {
            let capped = r.splats().clone().with_sh_degree(d.min(r.splats().sh_degree()));
            r.set_splats(capped);
        }
        if let Some(factor) = arm.filter3d {
            let floor = filter3d_floor(r, &paths[ai].frames, factor)?;
            let mut splats = r.splats().clone();
            splats.min_scale = Some(floor);
            r.set_splats(splats);
        }
        let renderer = renderer.as_ref().expect("loaded");
        eprintln!("[offpath] {}: {} splats, {} views (sh cap {:?}, 3D filter {:?})", arm.label, renderer.num_splats(), views.len(),
            arm.sh_degree, arm.filter3d);
        let near = o.near_m * scales[ai];
        let (mut ms, mut ts, mut ds) = (vec![], vec![], vec![]);
        let images = arm.sparse.ancestors().map(|d| d.join("images")).find(|d| d.is_dir());
        for (vi, view) in views.iter().enumerate() {
            let frame = renderer.render(&view.cams[ai].1, true)?;
            let mut m = view_metrics(&frame, near);
            if let Some(images) = &images {
                let (name, sharp) = source_sharpness(images, &paths[ai].frames, &view.cams[ai].1, scales[ai], &mut source_cache)?;
                m.source_sharpness = sharp;
                m.source_frame = name;
            }
            eprintln!("[offpath] {} view {}/{} {} {} sharp {:.0} (source {:.0} {}) holes {:.3} dedge {:.3} near {:.4}", arm.label, vi + 1, views.len(),
                view.anchor, view.variant.name(), m.sharpness, m.source_sharpness, m.source_frame, m.hole_frac, m.depth_edge_frac, m.near_frac);
            frame.rgb.save(dir.join(format!("{}__{}.png", stem(&view.anchor), view.variant.name())))?;
            ms.push(m);
            ds.push(frame.depth.iter().zip(&frame.alpha).map(|(&d, &a)| if a >= 0.5 && d > 0.0 { d / scales[ai] } else { 0.0 }).collect::<Vec<f32>>());
            ts.push(frame.rgb);
        }
        metrics.push(ms);
        tiles.push(ts);
        depths.push(ds);
    }

    // Depth tiles: one metric range per view, shared by the arms.
    let mut ranges = vec![];
    let mut depth_tiles: Vec<Vec<RgbImage>> = vec![vec![]; arms.len()];
    for (vi, view) in views.iter().enumerate() {
        let range = depth_range(depths.iter().map(|d| d[vi].as_slice()));
        for (ai, arm) in arms.iter().enumerate() {
            let (w, h) = tiles[ai][vi].dimensions();
            let img = depth_image(&depths[ai][vi], w, h, range);
            img.save(out.join("renders").join(sanitize(&arm.label)).join(format!("{}__{}__depth.png", stem(&view.anchor), view.variant.name())))?;
            depth_tiles[ai].push(img);
        }
        ranges.push(range);
    }

    // Sheets: one per anchor (5 rows: on path + 4 off-path; one column per arm), plus an overview.
    let labels: Vec<&str> = arms.iter().map(|a| a.label.as_str()).collect();
    let mut sheets = vec![];
    for (k, anchor) in anchors.iter().enumerate() {
        let rows: Vec<usize> = (0..views.len()).filter(|&i| views[i].anchor == *anchor).collect();
        let sheet = contact_sheet(&views, &rows, &tiles, &depth_tiles, &ranges, &metrics, &labels, o, o.tile_width);
        let p = out.join(format!("sheet_{k:02}_{}.png", stem(anchor)));
        sheet.save(&p)?;
        sheets.push(p);
    }
    let all: Vec<usize> = (0..views.len()).collect();
    let overview = contact_sheet(&views, &all, &tiles, &depth_tiles, &ranges, &metrics, &labels, o, o.tile_width.min(360));
    let overview_path = out.join("contact.jpg");
    splat_eval::images::write_jpeg(&overview_path, &overview, 88, None)?;

    let summary: Vec<Value> = arms.iter().enumerate().map(|(ai, arm)| {
        let mean = |pred: &dyn Fn(Variant) -> bool, f: &dyn Fn(&ViewMetrics) -> f32| {
            let v: Vec<f32> = views.iter().zip(&metrics[ai]).filter(|(v, _)| pred(v.variant)).map(|(_, m)| f(m)).collect();
            v.iter().sum::<f32>() / v.len().max(1) as f32
        };
        let group = |pred: &dyn Fn(Variant) -> bool| json!({
            "sharpness": mean(pred, &|m| m.sharpness), "source_sharpness": mean(pred, &|m| m.source_sharpness), "hole_frac": mean(pred, &|m| m.hole_frac),
            "depth_edge_frac": mean(pred, &|m| m.depth_edge_frac), "near_frac": mean(pred, &|m| m.near_frac)});
        let on = mean(&|v| v == Variant::Anchor, &|m| m.sharpness);
        let off = mean(&|v| v != Variant::Anchor, &|m| m.sharpness);
        json!({
            "label": arm.label, "ply": arm.ply, "sparse": arm.sparse, "units_per_metre": scales[ai],
            "sh_degree_cap": arm.sh_degree, "filter3d": arm.filter3d,
            "scale_source": if o.units_per_metre.is_some() { "given" } else { "median SfM-point depth / room_depth_m" },
            "on_path": group(&|v| v == Variant::Anchor),
            "off_path": group(&|v| v != Variant::Anchor),
            "shifted": group(&|v| matches!(v, Variant::Left | Variant::Right)),
            "turned": group(&|v| matches!(v, Variant::YawLeft | Variant::YawRight)),
            "off_path_sharpness_ratio": off / on.max(1e-6),
        })
    }).collect();
    let result = json!({
        "schema": "offpath-v1",
        "options": {"anchors": o.anchors, "seed": o.seed, "shift_m": o.shift_m, "yaw_deg": o.yaw_deg, "room_depth_m": o.room_depth_m,
                    "units_per_metre": o.units_per_metre, "near_m": o.near_m, "max_resolution": o.max_resolution},
        "anchors": anchors,
        "arms": summary,
        "views": views.iter().enumerate().map(|(vi, v)| json!({
            "anchor": v.anchor, "variant": v.variant.name(), "depth_range_m": ranges[vi],
            "arms": v.cams.iter().enumerate().map(|(ai, (n, c))| json!({"label": arms[ai].label, "frame": n, "camera": camera_json(c), "metrics": metrics[ai][vi]})).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "sheets": sheets,
        "contact": overview_path,
    });
    write_json(&out.join("offpath.json"), &result)?;
    Ok(result)
}

fn stem(name: &str) -> String {
    sanitize(Path::new(name).with_extension("").to_string_lossy().as_ref())
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

const BG: [u8; 3] = [24, 24, 24];
const FG: [u8; 3] = [235, 235, 235];
const DIM: [u8; 3] = [170, 170, 170];

/// Pooled 2nd–98th percentile of the covered depths of one view across arms (metres).
fn depth_range<'a>(views: impl Iterator<Item = &'a [f32]>) -> [f32; 2] {
    let mut d: Vec<f32> = views.flat_map(|v| v.iter().copied().filter(|d| *d > 0.0).step_by(7)).collect();
    if d.is_empty() {
        return [0.0, 1.0];
    }
    d.sort_by(f32::total_cmp);
    let q = |p: f64| d[((d.len() - 1) as f64 * p) as usize];
    let (lo, hi) = (q(0.02), q(0.98));
    [lo, hi.max(lo + 1e-3)]
}

/// Viridis stops (CC0), reversed so near is bright.
const DEPTH_STOPS: [[f32; 3]; 5] = [[253.0, 231.0, 37.0], [94.0, 201.0, 98.0], [33.0, 145.0, 140.0], [59.0, 82.0, 139.0], [68.0, 1.0, 84.0]];

fn depth_colour(t: f32) -> [u8; 3] {
    let x = t.clamp(0.0, 1.0) * (DEPTH_STOPS.len() - 1) as f32;
    let i = (x as usize).min(DEPTH_STOPS.len() - 2);
    let f = x - i as f32;
    std::array::from_fn(|c| (DEPTH_STOPS[i][c] * (1.0 - f) + DEPTH_STOPS[i + 1][c] * f).round() as u8)
}

/// Depth (metres, 0 = uncovered) colour-mapped over `range`; uncovered pixels black.
pub fn depth_image(depth: &[f32], w: u32, h: u32, range: [f32; 2]) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let d = depth.get((y * w + x) as usize).copied().unwrap_or(0.0);
        image::Rgb(if d > 0.0 { depth_colour((d - range[0]) / (range[1] - range[0])) } else { [0, 0, 0] })
    })
}

/// Rows = views, columns = arms; a header with the arm labels, a caption over each row (anchor
/// frame, variant, depth range), the colour tiles, each tile's metrics, then the depth tiles.
#[allow(clippy::too_many_arguments)]
fn contact_sheet(views: &[View], rows: &[usize], tiles: &[Vec<RgbImage>], depth_tiles: &[Vec<RgbImage>], ranges: &[[f32; 2]],
                 metrics: &[Vec<ViewMetrics>], labels: &[&str], o: &Options, tile_w: u32) -> RgbImage {
    let (pad, scale) = (8u32, if tile_w >= 480 { 2 } else { 1 });
    let line = 7 * scale + 6;
    let first = &tiles[0][rows[0]];
    let tile_h = (tile_w as f64 * first.height() as f64 / first.width() as f64).round() as u32;
    let cols = tiles.len() as u32;
    let row_h = line + tile_h + line + tile_h + pad;
    let width = pad + cols * (tile_w + pad);
    let height = line + pad + rows.len() as u32 * row_h;
    let mut img = RgbImage::from_pixel(width, height, image::Rgb(BG));
    for (c, label) in labels.iter().enumerate() {
        let x = pad + c as u32 * (tile_w + pad);
        draw_text(&mut img, x + (tile_w.saturating_sub(text_width(label, scale + 1))) / 2, 3, label, scale + 1, FG);
    }
    for (r, &vi) in rows.iter().enumerate() {
        let y = line + pad + r as u32 * row_h;
        let v = &views[vi];
        let [lo, hi] = ranges[vi];
        draw_text(&mut img, pad, y + 3, &format!("{}  {}   DEPTH {lo:.1}-{hi:.1} M", v.anchor, v.variant.caption(o)), scale, FG);
        for c in 0..cols as usize {
            let x = pad + c as u32 * (tile_w + pad);
            let t = imageops::resize(&tiles[c][vi], tile_w, tile_h, imageops::FilterType::Triangle);
            imageops::replace(&mut img, &t, x as i64, (y + line) as i64);
            let m = &metrics[c][vi];
            let text = format!("SHARP {:.0}  HOLES {:.1}%  DEDGE {:.1}%  NEAR {:.2}%", m.sharpness, m.hole_frac * 100.0, m.depth_edge_frac * 100.0, m.near_frac * 100.0);
            draw_text(&mut img, x, y + line + tile_h + 3, &text, scale, DIM);
            let d = imageops::resize(&depth_tiles[c][vi], tile_w, tile_h, imageops::FilterType::Nearest);
            imageops::replace(&mut img, &d, x as i64, (y + line + tile_h + line) as i64);
        }
    }
    img
}

/// A finished run's candidate PLY (default: report.json `best_candidate`) and its SfM model (the
/// candidate's `sfm_id` in state.json). A PLY kept on fal is read at its /data path when that is
/// mounted (inside fal jobs), else downloaded in place of its stub.
pub fn resolve_run(run: &Path, candidate: Option<&str>) -> anyhow::Result<(PathBuf, PathBuf)> {
    let report: Value = read_json(&run.join("report.json")).context("reading the run's report.json")?;
    let cand = match candidate {
        Some(c) => c,
        None => report["best_candidate"].as_str().context("report.json: no best_candidate")?,
    };
    let state: Value = read_json(&run.join("state.json"))?;
    let sfm = state["artifacts"][cand]["meta"]["sfm_id"].as_str().with_context(|| format!("state.json: {cand} has no sfm_id"))?;
    let sparse = sparse_dir(&run.join("artifacts").join(sfm))?;
    let ply = run.join("artifacts").join(cand).join("splat.ply");
    let rel = format!("artifacts/{cand}/splat.ply");
    if ply.is_file() && splat_core::fal::remote_stub(&ply).is_none() {
        return Ok((ply, sparse));
    }
    let kept: Option<PathBuf> = read_json::<Value>(&run.join("remote_artifacts.json")).ok()
        .and_then(|v| v[&rel]["fal_path"].as_str().map(PathBuf::from)).filter(|p| p.is_file());
    if let Some(p) = kept {
        return Ok((p, sparse));
    }
    if !ply.is_file() {
        bail!("{}: missing (pruned?) and not readable from fal /data here", ply.display());
    }
    splat_core::fal::materialize(run, &ply)?;
    Ok((ply, sparse))
}

/// The sparse model dir: `p` itself, or p/sparse/0, or p/sparse.
pub fn sparse_dir(p: &Path) -> anyhow::Result<PathBuf> {
    for d in [p.to_path_buf(), p.join("sparse/0"), p.join("sparse")] {
        if d.join("images.bin").is_file() || d.join("images.txt").is_file() {
            return Ok(d);
        }
    }
    bail!("{}: no COLMAP model (images.bin) here or under sparse/0", p.display())
}

/// `LABEL=RUN_DIR[#CANDIDATE_ID]` or `LABEL=PLY,SFM_DIR`, then optional `;sh=N` / `;f3d=X`.
pub fn parse_arm(spec: &str) -> anyhow::Result<Arm> {
    let mut parts = spec.split(';');
    let spec = parts.next().unwrap_or_default();
    let (mut sh_degree, mut filter3d) = (None, None);
    for opt in parts {
        match opt.split_once('=') {
            Some(("sh", v)) => sh_degree = Some(v.parse().with_context(|| format!("--arm {spec}: sh={v}"))?),
            Some(("f3d", v)) => filter3d = Some(v.parse().with_context(|| format!("--arm {spec}: f3d={v}"))?),
            _ => bail!("--arm {spec}: unknown option {opt} (sh=N, f3d=X)"),
        }
    }
    let (label, rest) = spec.split_once('=').with_context(|| format!("--arm {spec}: expected LABEL=RUN_DIR or LABEL=PLY,SFM_DIR"))?;
    let (ply, sparse) = match rest.split_once(',') {
        Some((ply, sfm)) => (PathBuf::from(ply), sparse_dir(Path::new(sfm))?),
        None => match rest.split_once('#') {
            Some((run, cand)) => resolve_run(Path::new(run), Some(cand))?,
            None => resolve_run(Path::new(rest), None)?,
        },
    };
    Ok(Arm { label: label.to_string(), ply, sparse, sh_degree, filter3d })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> RenderCamera {
        RenderCamera::look_at(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 5.0), Vec3::NEG_Y, 1.0, 64, 48)
    }

    #[test]
    fn variants_shift_along_right_and_turn_about_up() {
        let c = cam();
        let right = c.rotation * Vec3::X;
        let l = variant_camera(&c, Variant::Left, 0.5, 0.0);
        let r = variant_camera(&c, Variant::Right, 0.5, 0.0);
        assert!((l.position - (c.position - right * 0.5)).length() < 1e-5);
        assert!((r.position - (c.position + right * 0.5)).length() < 1e-5);
        assert_eq!(l.rotation, c.rotation);
        let yaw = 30f32.to_radians();
        let yr = variant_camera(&c, Variant::YawRight, 0.5, yaw);
        let yl = variant_camera(&c, Variant::YawLeft, 0.5, yaw);
        assert_eq!(yr.position, c.position);
        assert!((yr.forward().angle_between(c.forward()) - yaw).abs() < 1e-4);
        // Turning right swings the view toward the camera's right; up is unchanged.
        assert!(yr.forward().dot(right) > 0.4 && yl.forward().dot(right) < -0.4);
        assert!(((yr.rotation * Vec3::Y) - (c.rotation * Vec3::Y)).length() < 1e-5);
    }

    #[test]
    fn anchors_are_spread_and_seeded() {
        let names: Vec<String> = (0..100).map(|i| format!("frame_{i:06}.jpg")).collect();
        let a = pick_anchors(&names, 6, 0);
        assert_eq!(a, pick_anchors(&names, 6, 0));
        assert_eq!(a.len(), 6);
        let idx: Vec<usize> = a.iter().map(|n| names.iter().position(|m| m == n).unwrap()).collect();
        assert!(idx.windows(2).all(|w| (16..=17).contains(&(w[1] - w[0]))), "{idx:?}");
        assert_ne!(pick_anchors(&names, 6, 1), a);
        assert_eq!(pick_anchors(&names[..3], 6, 0).len(), 3);
    }

    #[test]
    fn match_name_exact_then_nearest_number() {
        let names: Vec<String> = ["frame_000004.jpg", "frame_000010.jpg", "other_000005.jpg"].map(String::from).to_vec();
        assert_eq!(match_name("frame_000010.jpg", &names).unwrap(), "frame_000010.jpg");
        assert_eq!(match_name("frame_000006.jpg", &names).unwrap(), "frame_000004.jpg");
        assert_eq!(match_name("frame_000008.jpg", &names).unwrap(), "frame_000010.jpg");
        assert_eq!(match_name("frame_000007.jpg", &names).unwrap(), "frame_000004.jpg");
        assert!(match_name("img.jpg", &names).is_none());
    }

    #[test]
    fn depth_tiles_share_a_range_and_black_out_holes() {
        let a: Vec<f32> = (0..100).map(|i| 1.0 + i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..100).map(|i| if i < 50 { 0.0 } else { 3.0 }).collect();
        let r = depth_range([a.as_slice(), b.as_slice()].into_iter());
        assert!(r[0] >= 1.0 && r[0] < 1.1 && (r[1] - 3.0).abs() < 1e-6, "{r:?}");
        let img = depth_image(&b, 10, 10, r);
        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(img.get_pixel(9, 9).0, [68, 1, 84]);
        assert_eq!(depth_colour(0.0), [253, 231, 37]);
    }

    #[test]
    fn scale_from_median_depth() {
        assert!((units_per_metre(&[5.0, f32::NAN, 4.0, 6.0], 2.5).unwrap() - 2.0).abs() < 1e-6);
        assert!(units_per_metre(&[f32::NAN], 2.5).is_err());
    }

    #[test]
    fn metrics_see_holes_edges_and_near_splats() {
        let (w, h) = (20u32, 10u32);
        let n = (w * h) as usize;
        let mut rgb = RgbImage::new(w, h);
        for (x, _, p) in rgb.enumerate_pixels_mut() {
            *p = image::Rgb([if x % 2 == 0 { 200 } else { 50 }; 3]);
        }
        // Left half a smooth wall at depth 2; right half a step to depth 4; last column uncovered.
        let mut depth = vec![2.0f32; n];
        let mut alpha = vec![1.0f32; n];
        for y in 0..h as usize {
            for x in 10..w as usize {
                depth[y * w as usize + x] = 4.0;
            }
            alpha[y * w as usize + 19] = 0.0;
            depth[y * w as usize + 19] = 0.0;
        }
        depth[0] = 0.1;
        let f = Frame { width: w, height: h, rgb, alpha, depth, median_depth: vec![], normal: vec![] };
        let m = view_metrics(&f, 0.3);
        assert!((m.hole_frac - 0.05).abs() < 1e-6);
        assert!((m.near_frac - 1.0 / n as f32).abs() < 1e-6);
        // 10 step edges + 2 around the near pixel, of the covered neighbour pairs.
        let pairs = (18 * 10 + 19 * 9) as f32;
        assert!((m.depth_edge_frac - 12.0 / pairs).abs() < 1e-6, "{}", m.depth_edge_frac);
        assert!(m.sharpness > 0.0);
    }
}
