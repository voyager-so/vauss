//! Put matted people (or any cut-out layer) into a rendered splat shot with 3D-consistent placement.
//!
//! Each layer is a sequence of RGBA cut-outs plus, per frame, where its feet stand in the splat's
//! world (`anchor`) and how many source pixels one world unit spans at that depth (`px_per_world`).
//! For every shot frame the anchor is projected into the new camera and the cut-out is scaled by
//! `(fy_new / z_new) / px_per_world`, so the person keeps their real size and parallax as the camera
//! moves (a camera-facing billboard). The splat depth pass occludes the layer (goal posts, a
//! teammate in front), and a soft contact shadow is laid on the ground plane under the feet.
//!
//! Layers come from either a matte of any clip plus a chosen spot and height (virtual set), or
//! from tracking people in the clip the splat was built from (reshoot): `track_layers` ray-casts
//! each person's foot pixel through the SfM camera onto the ground plane.
//!
//! Output (`out_dir`): `rgb/%05d.png` + `rgb.mp4` (composite), `mask/%05d.png` + `mask.mp4`
//! (where layers and shadows landed: the region a video model may repaint), `composite.json`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::{DMat4, DVec3};
use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage, RgbaImage};
use serde::{Deserialize, Serialize};

use crate::encode::Ffmpeg;
use crate::layout::ShotDir;
use crate::paths::{PathFrame, sample_path};
use crate::scene::{CaptureCamera, horizontal_basis, quantile};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Plane {
    pub point: [f64; 3],
    /// Unit normal, pointing up.
    pub normal: [f64; 3],
}

impl Plane {
    fn hit(&self, o: DVec3, d: DVec3) -> Option<DVec3> {
        let n = DVec3::from_array(self.normal);
        let den = d.dot(n);
        if den.abs() < 1e-9 {
            return None;
        }
        let t = (DVec3::from_array(self.point) - o).dot(n) / den;
        (t > 0.0).then(|| o + t * d)
    }
}

/// Robust ground plane through sparse points: RANSAC over planes whose normal is within 30 deg of
/// `up`, inlier band 1% of the point spread, then least-squares height refit along the normal.
pub fn fit_ground(points: &[DVec3], up: DVec3) -> Result<Plane> {
    if points.len() < 20 {
        bail!("too few sparse points ({}) for a ground plane", points.len());
    }
    let mut hs: Vec<f64> = points.iter().map(|p| p.dot(up)).collect();
    let spread = quantile(&mut hs.clone(), 0.95) - quantile(&mut hs, 0.05);
    let band = (spread * 0.02).max(1e-6);
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let (mut best, mut best_n) = ((points[0], up), 0usize);
    for _ in 0..2000 {
        let (a, b, c) = (points[rnd(points.len())], points[rnd(points.len())], points[rnd(points.len())]);
        let mut n = (b - a).cross(c - a);
        if n.length() < 1e-12 {
            continue;
        }
        n = n.normalize();
        if n.dot(up) < 0.0 {
            n = -n;
        }
        if n.dot(up) < 30f64.to_radians().cos() {
            continue;
        }
        let count = points.iter().filter(|p| ((**p - a).dot(n)).abs() < band).count();
        // Prefer the lowest big plane: a field, not a wall of hedges or a stand.
        if count > best_n {
            (best, best_n) = ((a, n), count);
        }
    }
    let (a, n) = best;
    let inl: Vec<DVec3> = points.iter().copied().filter(|p| ((*p - a).dot(n)).abs() < band).collect();
    let c = inl.iter().sum::<DVec3>() / inl.len().max(1) as f64;
    Ok(Plane { point: c.to_array(), normal: n.to_array() })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerFrame {
    /// RGBA cut-out (any size; usually the source frame size).
    pub image: PathBuf,
    /// Pixel in `image` that stands on the ground (bottom centre of the person).
    pub foot_px: [f64; 2],
    /// Source pixels per world unit at the person's depth.
    pub px_per_world: f64,
    /// World point under the feet (on the ground plane).
    pub anchor: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub frames: Vec<LayerFrame>,
    /// Layer frame used for shot frame i is `i + offset`, clamped (holds first/last pose).
    #[serde(default)]
    pub offset: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeSpec {
    pub layers: Vec<Layer>,
    pub ground: Plane,
    /// Contact shadow darkening at the centre, 0..1.
    #[serde(default = "d_shadow")]
    pub shadow: f32,
    /// Shadow radius in person heights.
    #[serde(default = "d_shadow_radius")]
    pub shadow_radius: f64,
    /// Splat surfaces closer than (1 - tol) * person depth hide the person.
    #[serde(default = "d_tol")]
    pub occlusion_tol: f64,
    /// Per-channel gain on the layer colour (crude light match; 1 = as filmed).
    #[serde(default = "d_gain")]
    pub gain: [f32; 3],
    /// Replace what the capture never saw above the horizon (low alpha, or surfaces floating
    /// higher than this many world units above the ground) with a sky gradient.
    #[serde(default)]
    pub sky: Option<Sky>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sky {
    pub max_height: f64,
    /// sRGB at the horizon and at 45 deg up.
    pub horizon: [u8; 3],
    pub zenith: [u8; 3],
}

fn d_shadow() -> f32 {
    0.45
}
fn d_shadow_radius() -> f64 {
    0.28
}
fn d_tol() -> f64 {
    0.08
}
fn d_gain() -> [f32; 3] {
    [1.0; 3]
}

impl CompositeSpec {
    pub fn new(layers: Vec<Layer>, ground: Plane) -> Self {
        Self { layers, ground, shadow: d_shadow(), shadow_radius: d_shadow_radius(), occlusion_tol: d_tol(), gain: d_gain(), sky: None }
    }
}

fn bilinear(img: &RgbaImage, x: f64, y: f64) -> [f32; 4] {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let mut out = [0f32; 4];
    for (dx, dy, wgt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
        let (px, py) = (x0 + dx, y0 + dy);
        if px < 0 || py < 0 || px >= w || py >= h {
            continue;
        }
        let p = img.get_pixel(px as u32, py as u32).0;
        let a = p[3] as f32 / 255.0;
        // premultiplied accumulation so edges don't pick up the matte's background colour
        for c in 0..3 {
            out[c] += wgt * a * p[c] as f32;
        }
        out[3] += wgt * a;
    }
    if out[3] > 1e-6 {
        let a = out[3];
        for c in out.iter_mut().take(3) {
            *c /= a;
        }
    }
    out
}

struct Cam {
    c2w: DMat4,
    w2c: DMat4,
    fy: f64,
    cx: f64,
    cy: f64,
}

impl Cam {
    fn new(f: &PathFrame, w: u32, h: u32) -> Self {
        Self { c2w: f.c2w, w2c: f.c2w.inverse(), fy: f.fy(h), cx: w as f64 / 2.0, cy: h as f64 / 2.0 }
    }
    fn project(&self, p: DVec3) -> Option<(f64, f64, f64)> {
        let q = self.w2c.transform_point3(p);
        (q.z > 1e-6).then(|| (self.fy * q.x / q.z + self.cx, self.fy * q.y / q.z + self.cy, q.z))
    }
    fn ray(&self, x: f64, y: f64) -> (DVec3, DVec3) {
        let d = DVec3::new((x - self.cx) / self.fy, (y - self.cy) / self.fy, 1.0);
        (self.c2w.w_axis.truncate(), self.c2w.transform_vector3(d).normalize())
    }
}

/// Metric z from the shot's normalised disparity PNG (0 where nothing was rendered -> far).
fn depth_z(v: u16, near: f64, far: f64) -> f64 {
    let inv = v as f64 / 65535.0 * (1.0 / near - 1.0 / far) + 1.0 / far;
    1.0 / inv.max(1e-12)
}

/// Composite every layer into every frame of a rendered shot.
pub fn composite_shot(shot: &ShotDir, spec: &CompositeSpec, out_dir: &Path, ff: &Ffmpeg) -> Result<serde_json::Value> {
    let path = shot.path()?;
    let meta = shot.render_meta()?;
    let frames = sample_path(&path);
    let (w, h) = (meta.width, meta.height);
    for d in ["rgb", "mask"] {
        std::fs::create_dir_all(out_dir.join(d))?;
    }
    let n = frames.len().min(meta.frames);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let visible = std::sync::Mutex::new(vec![0usize; spec.layers.len()]);
    std::thread::scope(|s| -> Result<()> {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            return Ok(());
                        }
                        let vis = composite_frame(shot, spec, &frames[i], i, w, h, meta.depth_near, meta.depth_far, out_dir)?;
                        let mut v = visible.lock().unwrap();
                        for (k, ok) in vis.into_iter().enumerate() {
                            v[k] += ok as usize;
                        }
                    }
                })
            })
            .collect();
        for wk in workers {
            wk.join().expect("composite worker panicked")?;
        }
        Ok(())
    })?;
    let rgb = ff.encode_video(&out_dir.join("rgb"), path.fps, &out_dir.join("rgb"), false)?;
    let mask = ff.encode_video(&out_dir.join("mask"), path.fps, &out_dir.join("mask"), true)?;
    let summary = serde_json::json!({
        "frames": n, "width": w, "height": h, "fps": path.fps,
        "layers": spec.layers.iter().zip(visible.into_inner().unwrap()).map(|(l, v)| serde_json::json!({"name": l.name, "frames": l.frames.len(), "visible_frames": v})).collect::<Vec<_>>(),
        "rgb": rgb, "mask": mask, "spec": spec,
    });
    std::fs::write(out_dir.join("composite.json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn composite_frame(
    shot: &ShotDir,
    spec: &CompositeSpec,
    f: &PathFrame,
    i: usize,
    w: u32,
    h: u32,
    near: f64,
    far: f64,
    out_dir: &Path,
) -> Result<Vec<bool>> {
    let mut rgb: RgbImage = image::open(shot.frame("rgb", i)).with_context(|| format!("shot frame {i}"))?.to_rgb8();
    let depth: ImageBuffer<Luma<u16>, Vec<u16>> = image::open(shot.frame("depth", i))?.to_luma16();
    let mut mask = GrayImage::new(w, h);
    let cam = Cam::new(f, w, h);
    let up = DVec3::from_array(spec.ground.normal);
    if let Some(sky) = &spec.sky {
        let alpha = image::open(shot.frame("alpha", i))?.to_luma8();
        let g0 = DVec3::from_array(spec.ground.point);
        for y in 0..h {
            for x in 0..w {
                let (o, d) = cam.ray(x as f64 + 0.5, y as f64 + 0.5);
                let elev = d.dot(up);
                if elev <= 0.0 {
                    continue;
                }
                // distance along the ray from z-depth: z = t * (ray . camera forward)
                let fwd = cam.c2w.z_axis.truncate();
                let t = depth_z(depth.get_pixel(x, y).0[0], near, far) / d.dot(fwd).max(1e-6);
                let height = (o + t * d - g0).dot(up);
                let a = alpha.get_pixel(x, y).0[0] as f64 / 255.0;
                // soft weights so the skyline doesn't alias
                let w_alpha = ((0.97 - a) / 0.15).clamp(0.0, 1.0);
                let w_high = ((height - sky.max_height) / (0.25 * sky.max_height)).clamp(0.0, 1.0);
                let wgt = w_alpha.max(w_high) as f32;
                if wgt <= 0.0 {
                    continue;
                }
                let k = (elev / std::f64::consts::FRAC_1_SQRT_2).clamp(0.0, 1.0) as f32;
                let px = rgb.get_pixel_mut(x, y);
                for c in 0..3 {
                    let s = sky.horizon[c] as f32 * (1.0 - k) + sky.zenith[c] as f32 * k;
                    px.0[c] = (px.0[c] as f32 * (1.0 - wgt) + s * wgt).round() as u8;
                }
            }
        }
    }

    // Resolve each layer's frame and its depth in this camera; draw far to near.
    let mut order: Vec<(usize, &LayerFrame, f64, f64, f64)> = vec![];
    for (k, l) in spec.layers.iter().enumerate() {
        if l.frames.is_empty() {
            continue;
        }
        let j = (i as i64 + l.offset).clamp(0, l.frames.len() as i64 - 1) as usize;
        let lf = &l.frames[j];
        if let Some((u, v, z)) = cam.project(DVec3::from_array(lf.anchor)) {
            order.push((k, lf, u, v, z));
        }
    }
    order.sort_by(|a, b| b.4.total_cmp(&a.4));
    let mut vis = vec![false; spec.layers.len()];

    for (k, lf, u, v, z) in order {
        let img: RgbaImage = image::open(&lf.image).with_context(|| format!("layer image {}", lf.image.display()))?.to_rgba8();
        let scale = (cam.fy / z) / lf.px_per_world; // dst px per src px
        let height_world = alpha_height(&img) as f64 / lf.px_per_world;

        // Contact shadow: soft disc on the ground plane under the feet.
        if spec.shadow > 0.0 && height_world > 0.0 {
            let anchor = DVec3::from_array(lf.anchor);
            let r = spec.shadow_radius * height_world;
            let (e1, e2) = horizontal_basis(up);
            let mut bb = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for a in 0..16 {
                let t = a as f64 / 16.0 * std::f64::consts::TAU;
                if let Some((x, y, _)) = cam.project(anchor + 2.0 * r * (t.cos() * e1 + t.sin() * e2)) {
                    bb = (bb.0.min(x), bb.1.min(y), bb.2.max(x), bb.3.max(y));
                }
            }
            let (x0, y0) = (bb.0.floor().max(0.0) as u32, bb.1.floor().max(0.0) as u32);
            let (x1, y1) = (bb.2.ceil().min(w as f64 - 1.0).max(0.0) as u32, bb.3.ceil().min(h as f64 - 1.0).max(0.0) as u32);
            for y in y0..=y1.max(y0) {
                for x in x0..=x1.max(x0) {
                    if x >= w || y >= h {
                        continue;
                    }
                    let (o, d) = cam.ray(x as f64 + 0.5, y as f64 + 0.5);
                    let Some(p) = spec.ground.hit(o, d) else { continue };
                    let dist = (p - anchor).length() / r;
                    let g = (-dist * dist * 1.5).exp() as f32 * spec.shadow;
                    if g < 0.01 {
                        continue;
                    }
                    // Only darken the ground itself, not something standing in front of it.
                    let zs = depth_z(depth.get_pixel(x, y).0[0], near, far);
                    let zp = cam.w2c.transform_point3(p).z;
                    if zs < zp * (1.0 - 2.0 * spec.occlusion_tol) {
                        continue;
                    }
                    let px = rgb.get_pixel_mut(x, y);
                    for c in 0..3 {
                        px.0[c] = (px.0[c] as f32 * (1.0 - g)).round() as u8;
                    }
                    let m = mask.get_pixel_mut(x, y);
                    m.0[0] = m.0[0].max((g / spec.shadow.max(1e-3) * 255.0) as u8);
                }
            }
        }

        // Billboard: dst pixel -> src pixel = foot + (dst - proj(anchor)) / scale.
        let (sw, sh) = (img.width() as f64, img.height() as f64);
        let (fx, fy) = (lf.foot_px[0], lf.foot_px[1]);
        let (x0, y0) = (u + (0.0 - fx) * scale, v + (0.0 - fy) * scale);
        let (x1, y1) = (u + (sw - fx) * scale, v + (sh - fy) * scale);
        let (xa, ya) = (x0.floor().max(0.0) as i64, y0.floor().max(0.0) as i64);
        let (xb, yb) = (x1.ceil().min(w as f64) as i64, y1.ceil().min(h as f64) as i64);
        let foot_band = v - 0.06 * height_world * cam.fy / z; // feet touch the ground: never self-occlude there
        for y in ya..yb {
            for x in xa..xb {
                let sx = fx + (x as f64 + 0.5 - u) / scale - 0.5;
                let sy = fy + (y as f64 + 0.5 - v) / scale - 0.5;
                let p = bilinear(&img, sx, sy);
                if p[3] < 0.004 {
                    continue;
                }
                let zs = depth_z(depth.get_pixel(x as u32, y as u32).0[0], near, far);
                if zs < z * (1.0 - spec.occlusion_tol) && (y as f64) < foot_band {
                    continue;
                }
                let px = rgb.get_pixel_mut(x as u32, y as u32);
                for c in 0..3 {
                    let src = (p[c] * spec.gain[c]).clamp(0.0, 255.0);
                    px.0[c] = (px.0[c] as f32 * (1.0 - p[3]) + src * p[3]).round() as u8;
                }
                let m = mask.get_pixel_mut(x as u32, y as u32);
                m.0[0] = m.0[0].max((p[3] * 255.0) as u8);
                vis[k] = true;
            }
        }
    }
    rgb.save(out_dir.join("rgb").join(format!("{i:05}.png")))?;
    mask.save(out_dir.join("mask").join(format!("{i:05}.png")))?;
    Ok(vis)
}

/// Height in pixels of the alpha > 0.5 rows.
fn alpha_height(img: &RgbaImage) -> u32 {
    let rows: Vec<u32> = (0..img.height()).filter(|&y| (0..img.width()).any(|x| img.get_pixel(x, y).0[3] > 127)).collect();
    match (rows.first(), rows.last()) {
        (Some(a), Some(b)) => b - a + 1,
        _ => 0,
    }
}

/// Bounding box (x0, y0, x1, y1) of mask > 127, or None if empty.
pub fn mask_bbox(m: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let mut bb: Option<(u32, u32, u32, u32)> = None;
    for (x, y, p) in m.enumerate_pixels() {
        if p.0[0] > 127 {
            bb = Some(match bb {
                None => (x, y, x, y),
                Some((a, b, c, d)) => (a.min(x), b.min(y), c.max(x), d.max(y)),
            });
        }
    }
    bb
}

/// Connected components (4-neighbour) of mask > 127 with at least `min_px` pixels.
pub fn components(m: &GrayImage, min_px: usize) -> Vec<GrayImage> {
    let (w, h) = m.dimensions();
    let mut label = vec![0u32; (w * h) as usize];
    let mut out = vec![];
    let mut next = 0u32;
    for start in 0..(w * h) as usize {
        if label[start] != 0 || m.as_raw()[start] <= 127 {
            continue;
        }
        next += 1;
        let mut stack = vec![start];
        let mut pix = vec![];
        label[start] = next;
        while let Some(p) = stack.pop() {
            pix.push(p);
            let (x, y) = ((p as u32) % w, (p as u32) / w);
            let mut push = |q: usize| {
                if label[q] == 0 && m.as_raw()[q] > 127 {
                    label[q] = next;
                    stack.push(q);
                }
            };
            if x > 0 {
                push(p - 1);
            }
            if x + 1 < w {
                push(p + 1);
            }
            if y > 0 {
                push(p - w as usize);
            }
            if y + 1 < h {
                push(p + w as usize);
            }
        }
        if pix.len() >= min_px {
            let mut g = GrayImage::new(w, h);
            for p in pix {
                g.as_mut()[p] = 255;
            }
            out.push(g);
        }
    }
    out
}

/// Source frame for tracking: image + per-person masks (one mask per instance, same size).
pub struct TrackInput<'a> {
    pub camera: &'a CaptureCamera,
    pub image: PathBuf,
    pub masks: Vec<GrayImage>,
}

/// Turn per-frame person masks from the capture clip into layers: cut-outs written to
/// `out_dir/<layer>/%05d.png`, anchors by ray-casting the foot pixel onto the ground plane.
/// Instances are linked frame to frame by nearest anchor (in world units, gated by `max_step`).
/// Anchors and scales are smoothed over `smooth` frames to steady the billboard.
pub fn track_layers(inputs: &[TrackInput], ground: &Plane, out_dir: &Path, max_step: f64, smooth: usize) -> Result<Vec<Layer>> {
    struct Track {
        frames: Vec<Option<LayerFrame>>,
        last: DVec3,
        last_seen: usize,
    }
    let mut tracks: Vec<Track> = vec![];
    for (fi, inp) in inputs.iter().enumerate() {
        let src = image::open(&inp.image).with_context(|| format!("{}", inp.image.display()))?.to_rgb8();
        let cam = &inp.camera;
        let (cx, cy) = (cam.width as f64 / 2.0, cam.height as f64 / 2.0);
        let s = src.width() as f64 / cam.width as f64; // frames may be downscaled vs the SfM images
        let mut taken = vec![false; tracks.len()];
        for m in &inp.masks {
            let Some((x0, _, x1, y1)) = mask_bbox(m) else { continue };
            let foot = [(x0 + x1) as f64 / 2.0, y1 as f64 + 0.5];
            let d = DVec3::new((foot[0] / s - cx) / cam.fy, (foot[1] / s - cy) / cam.fy, 1.0);
            let (o, dir) = (cam.center(), cam.c2w.transform_vector3(d).normalize());
            let Some(anchor) = ground.hit(o, dir) else { continue };
            let z = cam.c2w.inverse().transform_point3(anchor).z;
            let px_per_world = cam.fy * s / z;
            let ti = tracks
                .iter()
                .enumerate()
                .filter(|(k, t)| !taken.get(*k).copied().unwrap_or(false) && t.last.distance(anchor) < max_step * (fi - t.last_seen) as f64)
                .min_by(|a, b| a.1.last.distance(anchor).total_cmp(&b.1.last.distance(anchor)))
                .map(|(k, _)| k);
            let k = match ti {
                Some(k) => k,
                None => {
                    tracks.push(Track { frames: vec![None; inputs.len()], last: anchor, last_seen: fi });
                    taken.push(false);
                    tracks.len() - 1
                }
            };
            taken[k] = true;
            let dir = out_dir.join(format!("p{k}"));
            std::fs::create_dir_all(&dir)?;
            let cut: RgbaImage = ImageBuffer::from_fn(src.width(), src.height(), |x, y| {
                let Rgb([r, g, b]) = *src.get_pixel(x, y);
                image::Rgba([r, g, b, m.get_pixel(x, y).0[0]])
            });
            let image = dir.join(format!("{fi:05}.png"));
            cut.save(&image)?;
            tracks[k].frames[fi] = Some(LayerFrame { image, foot_px: foot, px_per_world, anchor: anchor.to_array() });
            tracks[k].last = anchor;
            tracks[k].last_seen = fi;
        }
    }
    let mut layers = vec![];
    for (k, t) in tracks.into_iter().enumerate() {
        let seen: Vec<usize> = (0..t.frames.len()).filter(|i| t.frames[*i].is_some()).collect();
        if seen.len() < 3 {
            continue; // flicker detections
        }
        // Fill gaps by holding the nearest seen frame, then smooth anchor + scale.
        let (first, last) = (seen[0], *seen.last().unwrap());
        let mut fr: Vec<LayerFrame> = vec![];
        let mut held = t.frames[first].clone().unwrap();
        for i in first..=last {
            if let Some(f) = &t.frames[i] {
                held = f.clone();
            }
            fr.push(held.clone());
        }
        let raw = fr.clone();
        let r = smooth / 2;
        for i in 0..fr.len() {
            let (a, b) = (i.saturating_sub(r), (i + r).min(fr.len() - 1));
            let cnt = (b - a + 1) as f64;
            let mut acc = DVec3::ZERO;
            let mut ppw = 0.0;
            for f in &raw[a..=b] {
                acc += DVec3::from_array(f.anchor);
                ppw += f.px_per_world;
            }
            fr[i].anchor = (acc / cnt).to_array();
            fr[i].px_per_world = ppw / cnt;
        }
        layers.push(Layer { name: format!("p{k}"), frames: fr, offset: -(first as i64) });
    }
    Ok(layers)
}
