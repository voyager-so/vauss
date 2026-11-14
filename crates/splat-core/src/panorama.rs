//! 360 equirect input: the geometry of splitting each equirectangular frame into 14 pinhole views
//! that share one rig pose (COLMAP 4.2.1 `python/pycolmap/panorama.py` conventions:
//! `get_virtual_rotations`, `spherical_img_from_cam`). Pure geometry and file helpers, no image I/O.
//!
//! Frames: camera x right, y down, z forward. A pano direction `d` maps to equirect pixel
//! coordinates by yaw = atan2(x, z), pitch = -atan2(y, hypot(x, z)), u = W(1 + yaw/pi)/2,
//! v = H(1 - 2 pitch/pi)/2 (continuous coordinates, pixel centres at +0.5).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// Row-major 3x3 rotation.
pub type Mat3 = [[f64; 3]; 3];

/// One virtual view: name prefix and its pitch (up positive) and yaw, in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    pub prefix: &'static str,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
}

/// The view layout of a 360 split.
#[derive(Debug, Clone)]
pub struct Layout {
    pub views: Vec<View>,
}

/// Views per 360 frame in [`Layout::v14`].
pub const VIEWS: usize = 14;
/// The manifest written next to split views (frames folder and SfM dataset folder).
pub const MANIFEST: &str = "panorama.json";
/// COLMAP rig_configurator input written next to the manifest.
pub const RIG_CONFIG: &str = "rig_config.json";
/// Default half-angle of the masked cone below the camera (operator, tripod).
pub const DEFAULT_NADIR_MASK_DEG: f32 = 30.0;

impl Layout {
    /// COLMAP's 12 views (4 level, 4 at +35 deg offset by 45 deg, 4 at -35 deg) reordered level
    /// first, plus straight up and straight down. 90 x 90 deg each; view 0 is the rig reference.
    pub fn v14() -> Self {
        const PREFIXES: [&str; VIEWS] =
            ["v00_", "v01_", "v02_", "v03_", "v04_", "v05_", "v06_", "v07_", "v08_", "v09_", "v10_", "v11_", "v12_", "v13_"];
        let mut views = Vec::with_capacity(VIEWS);
        let rings: [(f64, f64); 3] = [(0.0, 0.0), (35.0, 45.0), (-35.0, 0.0)];
        for (pitch, offset) in rings {
            for k in 0..4 {
                views.push((pitch, offset + 90.0 * k as f64));
            }
        }
        views.push((90.0, 0.0));
        views.push((-90.0, 0.0));
        Self { views: views.into_iter().zip(PREFIXES).map(|((pitch_deg, yaw_deg), prefix)| View { prefix, pitch_deg, yaw_deg }).collect() }
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    pub fn prefixes(&self) -> Vec<String> {
        self.views.iter().map(|v| v.prefix.to_string()).collect()
    }

    /// Forward axis of every view in pano coordinates.
    pub fn forwards(&self) -> Vec<[f64; 3]> {
        (0..self.len()).map(|i| self.cam_from_pano(i)[2]).collect()
    }

    /// cam_from_pano = Rx(-pitch) . Ry(-yaw) (scipy `from_euler("XY", [-pitch, -yaw])`).
    pub fn cam_from_pano(&self, view: usize) -> Mat3 {
        let v = self.views[view];
        mul(&rx(-v.pitch_deg.to_radians()), &ry(-v.yaw_deg.to_radians()))
    }
}

/// [`Layout::v14`]'s rotation of `view`.
pub fn cam_from_pano(view: usize) -> Mat3 {
    Layout::v14().cam_from_pano(view)
}

fn rx(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn ry(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

pub fn transpose(a: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            m[j][i] = v;
        }
    }
    m
}

pub fn apply(a: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

/// Unit quaternion (w, x, y, z) of a rotation matrix, w >= 0.
pub fn quaternion(m: &Mat3) -> [f64; 4] {
    let t = m[0][0] + m[1][1] + m[2][2];
    let q = if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    };
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let sign = if q[0] < 0.0 { -1.0 } else { 1.0 };
    q.map(|v| sign * v / n)
}

/// Rotation matrix of a unit quaternion (w, x, y, z).
pub fn rotation(q: [f64; 4]) -> Mat3 {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// The camera-frame ray of pixel (x, y) of an `s` x `s` view (fx = fy = cx = cy = s/2).
pub fn pixel_ray(s: u32, x: u32, y: u32) -> [f64; 3] {
    let f = s as f64 / 2.0;
    [(x as f64 + 0.5 - f) / f, (y as f64 + 0.5 - f) / f, 1.0]
}

/// Continuous equirect coordinates (u, v) of a pano direction, in [0, W] x [0, H].
pub fn equirect_uv(d: [f64; 3], w: u32, h: u32) -> (f64, f64) {
    let yaw = d[0].atan2(d[2]);
    let pitch = -d[1].atan2(d[0].hypot(d[2]));
    (w as f64 * (1.0 + yaw / std::f64::consts::PI) / 2.0, h as f64 * (1.0 - 2.0 * pitch / std::f64::consts::PI) / 2.0)
}

/// Pano direction (unit) of continuous equirect coordinates (inverse of [`equirect_uv`]).
pub fn equirect_dir(u: f64, v: f64, w: u32, h: u32) -> [f64; 3] {
    let yaw = (2.0 * u / w as f64 - 1.0) * std::f64::consts::PI;
    let pitch = (1.0 - 2.0 * v / h as f64) * std::f64::consts::FRAC_PI_2;
    let (sp, cp) = pitch.sin_cos();
    [cp * yaw.sin(), -sp, cp * yaw.cos()]
}

/// Per pixel of `view` (row-major, `s * s` entries): the equirect pixel-index coordinates
/// (u - 0.5, v - 0.5) it samples from a `w` x `h` panorama. Sample bilinearly, wrapping u and
/// clamping v.
pub fn lookup_table(view: usize, s: u32, w: u32, h: u32) -> Vec<[f32; 2]> {
    let pano_from_cam = transpose(&cam_from_pano(view));
    let mut table = Vec::with_capacity((s * s) as usize);
    for y in 0..s {
        for x in 0..s {
            let (u, v) = equirect_uv(apply(&pano_from_cam, pixel_ray(s, x, y)), w, h);
            table.push([(u - 0.5) as f32, (v - 0.5) as f32]);
        }
    }
    table
}

/// The view of [`Layout::v14`] that owns pano direction `d`: among the views whose 90 deg
/// frustum contains `d`, the one with the largest forward-axis dot product (ties go to the lower
/// index). The frustum condition matters only next to the level/up/down ring corners, where the
/// nearest axis can belong to a view that does not see the direction (yaw 225, pitch -10.8 is
/// nearest to view 6's axis but 45.8 deg below its centre).
pub fn owner(d: [f64; 3]) -> usize {
    owner_of(&rotations(), d)
}

fn rotations() -> Vec<Mat3> {
    let layout = Layout::v14();
    (0..layout.len()).map(|i| layout.cam_from_pano(i)).collect()
}

fn owner_of(rotations: &[Mat3], d: [f64; 3]) -> usize {
    let mut best = (usize::MAX, f64::NEG_INFINITY);
    for (i, m) in rotations.iter().enumerate() {
        let c = apply(m, d);
        let inside = c[2] > 0.0 && c[0].abs() <= c[2] && c[1].abs() <= c[2];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if inside && c[2] / n > best.1 {
            best = (i, c[2] / n);
        }
    }
    assert!(best.0 != usize::MAX, "the 14-view layout covers the sphere");
    best.0
}

/// Whether pano direction `d` lies inside the cone of half-angle `nadir_deg` around straight
/// down (pano +y). 0 disables it.
pub fn in_nadir(d: [f64; 3], nadir_deg: f32) -> bool {
    if nadir_deg <= 0.0 {
        return false;
    }
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    d[1] / n > (nadir_deg as f64).to_radians().cos()
}

fn view_mask(view: usize, s: u32, keep: impl Fn([f64; 3]) -> bool) -> Vec<u8> {
    let pano_from_cam = transpose(&cam_from_pano(view));
    let mut mask = Vec::with_capacity((s * s) as usize);
    for y in 0..s {
        for x in 0..s {
            mask.push(if keep(apply(&pano_from_cam, pixel_ray(s, x, y))) { 255 } else { 0 });
        }
    }
    mask
}

/// SfM feature mask of `view` (row-major, 255 keep, 0 ignore): the pixels whose ray this view
/// owns (each part of the sphere gives features to exactly one view) and that lie outside the
/// nadir cone.
pub fn owner_mask(view: usize, s: u32, nadir_deg: f32) -> Vec<u8> {
    let rotations = rotations();
    view_mask(view, s, |d| owner_of(&rotations, d) == view && !in_nadir(d, nadir_deg))
}

/// Training mask of `view` (row-major): 0 inside the nadir cone, 255 elsewhere.
pub fn nadir_mask(view: usize, s: u32, nadir_deg: f32) -> Vec<u8> {
    view_mask(view, s, |d| !in_nadir(d, nadir_deg))
}

/// COLMAP `rig_configurator` input: one rig whose sensors are the 14 views, view 0 the reference
/// (cam_from_rig = cam_from_pano), every sensor a PINHOLE of side `s`.
pub fn rig_config_json(s: u32) -> serde_json::Value {
    let layout = Layout::v14();
    let f = s as f64 / 2.0;
    let cameras: Vec<serde_json::Value> = layout
        .views
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut camera = serde_json::json!({"image_prefix": v.prefix, "camera_model_name": "PINHOLE", "camera_params": [f, f, f, f]});
            if i == 0 {
                camera["ref_sensor"] = serde_json::json!(true);
            } else {
                camera["cam_from_rig_rotation"] = serde_json::json!(quaternion(&layout.cam_from_pano(i)));
                camera["cam_from_rig_translation"] = serde_json::json!([0.0, 0.0, 0.0]);
            }
            camera
        })
        .collect();
    serde_json::json!([{"cameras": cameras}])
}

/// `panorama.json`: how a frames folder (and the SfM dataset made from it) was split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub views: u32,
    pub view_size: u32,
    pub nadir_mask_deg: f32,
    /// Prefix of the view whose images stand for the camera path (one per 360 frame).
    pub path_prefix: String,
    pub prefixes: Vec<String>,
    /// The equirect frame names, in capture order.
    pub frames: Vec<String>,
    /// Row-major cam_from_pano of each view.
    pub cam_from_pano: Vec<[f64; 9]>,
}

impl Manifest {
    pub fn new(view_size: u32, nadir_mask_deg: f32, frames: Vec<String>) -> Self {
        let layout = Layout::v14();
        Self {
            version: 1,
            views: layout.len() as u32,
            view_size,
            nadir_mask_deg,
            path_prefix: layout.views[0].prefix.into(),
            prefixes: layout.prefixes(),
            frames,
            cam_from_pano: (0..layout.len()).map(|i| {
                let m = layout.cam_from_pano(i);
                [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]]
            }).collect(),
        }
    }

    /// The manifest in `dir`, if it holds one.
    pub fn read(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join(MANIFEST);
        if !path.is_file() {
            return Ok(None);
        }
        let manifest: Self = serde_json::from_slice(&std::fs::read(&path)?).with_context(|| format!("reading {}", path.display()))?;
        ensure!(manifest.version == 1, "{}: unsupported panorama manifest version {}", path.display(), manifest.version);
        ensure!(manifest.prefixes.len() == manifest.views as usize && manifest.cam_from_pano.len() == manifest.views as usize,
            "{}: views, prefixes and rotations disagree", path.display());
        Ok(Some(manifest))
    }

    pub fn write(&self, dir: &Path) -> Result<()> {
        let path = dir.join(MANIFEST);
        std::fs::write(&path, serde_json::to_vec_pretty(self)?).with_context(|| format!("writing {}", path.display()))
    }

    /// The view name of `frame` seen by view `view`.
    pub fn view_name(&self, view: usize, frame: &str) -> String {
        format!("{}{frame}", self.prefixes[view])
    }

    pub fn rotation(&self, view: usize) -> Mat3 {
        let m = self.cam_from_pano[view];
        [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
    }
}

/// The 360 frame a view image belongs to: its name after the `vNN_` prefix.
pub fn group_key(name: &str) -> Option<&str> {
    let b = name.as_bytes();
    (b.len() > 4 && b[0] == b'v' && b[1].is_ascii_digit() && b[2].is_ascii_digit() && b[3] == b'_').then(|| &name[4..])
}

/// The names that stand for the camera path: in a 360 dataset (`panorama.json` in `dataset_dir`)
/// only the path view's names, one per 360 frame in capture order; otherwise `names` unchanged.
pub fn path_names(dataset_dir: &Path, names: &[String]) -> Vec<String> {
    match Manifest::read(dataset_dir) {
        Ok(Some(manifest)) => path_names_for(&manifest, names),
        _ => names.to_vec(),
    }
}

fn path_names_for(manifest: &Manifest, names: &[String]) -> Vec<String> {
    let order: std::collections::HashMap<&str, usize> = manifest.frames.iter().enumerate().map(|(i, f)| (f.as_str(), i)).collect();
    let mut path: Vec<&String> = names.iter().filter(|n| n.starts_with(&manifest.path_prefix)).collect();
    path.sort_by_key(|n| (group_key(n).and_then(|g| order.get(g)).copied().unwrap_or(usize::MAX), n.as_str()));
    path.into_iter().cloned().collect()
}

/// Held-out names of a whole-panorama split: registered view names grouped by [`group_key`],
/// groups sorted, every `interval`-th group (starting with the first) held out with all its views.
/// Holding out single views would leak the overlapping pixels of the same 360 frame's other views.
/// Names without a view prefix are their own group. Returns (held-out names, held-out groups).
pub fn heldout_groups(names: &[String], interval: usize) -> (Vec<String>, usize) {
    let mut groups: std::collections::BTreeMap<&str, Vec<&String>> = std::collections::BTreeMap::new();
    for name in names {
        groups.entry(group_key(name).unwrap_or(name)).or_default().push(name);
    }
    let held: Vec<&Vec<&String>> = groups.values().step_by(interval.max(1)).collect();
    let mut heldout: Vec<String> = held.iter().flat_map(|g| g.iter().map(|n| (*n).clone())).collect();
    heldout.sort();
    (heldout, held.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// Deterministic directions spread over the sphere.
    fn directions(n: usize) -> impl Iterator<Item = [f64; 3]> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(move |_| {
            let z = 2.0 * next() - 1.0;
            let phi = 2.0 * std::f64::consts::PI * next();
            let r = (1.0 - z * z).sqrt();
            [r * phi.cos(), z, r * phi.sin()]
        })
    }

    #[test]
    fn rotations_are_orthonormal_and_view_zero_is_identity() {
        let layout = Layout::v14();
        assert_eq!(layout.len(), VIEWS);
        for i in 0..VIEWS {
            let m = layout.cam_from_pano(i);
            let p = mul(&m, &transpose(&m));
            for (r, row) in p.iter().enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    assert!(close(v, if r == c { 1.0 } else { 0.0 }, 1e-12), "view {i}");
                }
            }
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            assert!(close(det, 1.0, 1e-12));
        }
        let identity = layout.cam_from_pano(0);
        assert_eq!(identity, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn centre_pixels_sample_the_documented_equirect_points() {
        let (w, h, s) = (2048u32, 1024u32, 64u32);
        let centre = |view: usize| {
            // The four pixels around the centre average to the exact centre ray.
            let t = lookup_table(view, s, w, h);
            let at = |x: u32, y: u32| t[(y * s + x) as usize];
            let p = [at(s / 2 - 1, s / 2 - 1), at(s / 2, s / 2 - 1), at(s / 2 - 1, s / 2), at(s / 2, s / 2)];
            (p, equirect_uv(apply(&transpose(&cam_from_pano(view)), [0.0, 0.0, 1.0]), w, h))
        };
        let (_, (u, v)) = centre(0);
        assert!(close(u, w as f64 / 2.0, 1e-9) && close(v, h as f64 / 2.0, 1e-9));
        let (_, (u, _)) = centre(1);
        assert!(close(u, 3.0 * w as f64 / 4.0, 1e-9));
        let (_, (_, v)) = centre(12);
        assert!(close(v, 0.0, 1e-9));
        let (_, (_, v)) = centre(13);
        assert!(close(v, h as f64, 1e-9));
        // The table's centre pixels straddle those points (index coordinates are -0.5).
        let (p, _) = centre(0);
        let mean_u = p.iter().map(|q| q[0] as f64).sum::<f64>() / 4.0 + 0.5;
        let mean_v = p.iter().map(|q| q[1] as f64).sum::<f64>() / 4.0 + 0.5;
        assert!(close(mean_u, w as f64 / 2.0, 1e-3) && close(mean_v, h as f64 / 2.0, 1e-3));
        let (p, _) = centre(12);
        assert!(p.iter().all(|q| q[1] < 8.0), "zenith view centre samples row 0: {p:?}");
        let (p, _) = centre(13);
        assert!(p.iter().all(|q| q[1] > h as f32 - 9.0), "nadir view centre samples the last row: {p:?}");
    }

    #[test]
    fn layout_angles_match_the_table() {
        let layout = Layout::v14();
        for (i, v) in layout.views.iter().enumerate() {
            let f = layout.cam_from_pano(i)[2];
            let pitch = (-f[1]).asin().to_degrees();
            assert!(close(pitch, v.pitch_deg, 1e-9), "view {i} pitch {pitch}");
            if v.pitch_deg.abs() < 90.0 {
                let yaw = f[0].atan2(f[2]).to_degrees().rem_euclid(360.0);
                assert!(close(yaw, v.yaw_deg, 1e-9), "view {i} yaw {yaw}");
            }
        }
        assert_eq!(layout.views[4].prefix, "v04_");
        assert_eq!((layout.views[4].pitch_deg, layout.views[4].yaw_deg), (35.0, 45.0));
        assert_eq!((layout.views[8].pitch_deg, layout.views[8].yaw_deg), (-35.0, 0.0));
    }

    #[test]
    fn rig_config_quaternions_round_trip() {
        let config = rig_config_json(1024);
        let cameras = config[0]["cameras"].as_array().unwrap();
        assert_eq!(cameras.len(), VIEWS);
        assert_eq!(cameras[0]["ref_sensor"], true);
        assert!(cameras[0].get("cam_from_rig_rotation").is_none());
        for (i, camera) in cameras.iter().enumerate() {
            assert_eq!(camera["image_prefix"], format!("v{i:02}_"));
            assert_eq!(camera["camera_model_name"], "PINHOLE");
            assert_eq!(camera["camera_params"], serde_json::json!([512.0, 512.0, 512.0, 512.0]));
            if i == 0 {
                continue;
            }
            assert_eq!(camera["cam_from_rig_translation"], serde_json::json!([0.0, 0.0, 0.0]));
            let q: Vec<f64> = camera["cam_from_rig_rotation"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let m = rotation([q[0], q[1], q[2], q[3]]);
            let expected = cam_from_pano(i);
            for r in 0..3 {
                for c in 0..3 {
                    assert!(close(m[r][c], expected[r][c], 1e-12), "view {i}");
                }
            }
        }
        // All quaternion branches.
        for m in [rx(3.0), ry(3.0), mul(&rx(1.5), &ry(2.9)), [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]] {
            let back = rotation(quaternion(&m));
            for r in 0..3 {
                for c in 0..3 {
                    assert!(close(back[r][c], m[r][c], 1e-12));
                }
            }
        }
    }

    #[test]
    fn owner_masks_cover_every_direction_outside_the_nadir_exactly_once() {
        let s = 256u32;
        for nadir in [DEFAULT_NADIR_MASK_DEG, 0.0] {
            let masks: Vec<Vec<u8>> = (0..VIEWS).map(|v| owner_mask(v, s, nadir)).collect();
            let rotations: Vec<Mat3> = (0..VIEWS).map(cam_from_pano).collect();
            let (mut covered, mut masked) = (0usize, 0usize);
            for d in directions(100_000) {
                let mut hits = 0;
                for (view, m) in rotations.iter().enumerate() {
                    let c = apply(m, d);
                    if c[2] <= 0.0 {
                        continue;
                    }
                    let f = s as f64 / 2.0;
                    let (x, y) = (c[0] / c[2] * f + f, c[1] / c[2] * f + f);
                    if x < 0.0 || y < 0.0 || x >= s as f64 || y >= s as f64 {
                        continue;
                    }
                    if masks[view][(y as u32 * s + x as u32) as usize] == 255 {
                        hits += 1;
                    }
                }
                // Within about a pixel of the cone's edge either answer is right.
                let cone_edge = nadir > 0.0 && (d[1].acos() - (nadir as f64).to_radians()).abs() < 4.0 / s as f64;
                if in_nadir(d, nadir) {
                    masked += 1;
                    assert!(hits == 0 || cone_edge, "nadir direction {d:?} kept");
                } else if cone_edge {
                    covered += usize::from(hits == 1);
                } else {
                    // Pixel quantisation: a direction within a pixel of an ownership edge may land
                    // on the neighbour's boundary pixel. Allow none uncovered and rare doubles.
                    assert!(hits >= 1 || near_boundary(d, s), "direction {d:?} uncovered");
                    if hits == 1 {
                        covered += 1;
                    }
                }
            }
            if nadir == 0.0 {
                assert_eq!(masked, 0);
            } else {
                // Cone solid angle fraction (1 - cos 30) / 2 = 6.7%.
                assert!((5_500..8_000).contains(&masked), "{masked}");
            }
            assert!(covered as f64 >= 0.99 * (100_000 - masked) as f64, "{covered} covered once");
        }
    }

    /// Whether `d` is within about a pixel of an ownership boundary (ties between two views).
    fn near_boundary(d: [f64; 3], s: u32) -> bool {
        // Some direction within ~2 pixels has another owner.
        let e = 4.0 / s as f64;
        let view = owner(d);
        [[e, 0.0, 0.0], [-e, 0.0, 0.0], [0.0, e, 0.0], [0.0, -e, 0.0], [0.0, 0.0, e], [0.0, 0.0, -e]]
            .iter().any(|o| owner([d[0] + o[0], d[1] + o[1], d[2] + o[2]]) != view)
    }

    #[test]
    fn owned_rays_lie_inside_their_view() {
        let rotations = rotations();
        for d in directions(100_000) {
            let view = owner_of(&rotations, d);
            let c = apply(&cam_from_pano(view), d);
            assert!(c[2] > 0.0 && (c[0] / c[2]).abs() <= 1.0 && (c[1] / c[2]).abs() <= 1.0, "{d:?} owned by {view} outside its frustum");
        }
    }

    #[test]
    fn training_masks_are_black_only_inside_the_cone() {
        let s = 128;
        for view in 0..VIEWS {
            let mask = nadir_mask(view, s, 30.0);
            let pano_from_cam = transpose(&cam_from_pano(view));
            for y in 0..s {
                for x in 0..s {
                    let d = apply(&pano_from_cam, pixel_ray(s, x, y));
                    assert_eq!(mask[(y * s + x) as usize] == 0, in_nadir(d, 30.0));
                }
            }
            assert!(nadir_mask(view, s, 0.0).iter().all(|&v| v == 255));
        }
        // The level views see none of a 30 deg cone; the nadir view is mostly black.
        assert!(nadir_mask(0, s, 30.0).iter().all(|&v| v == 255));
        assert!(nadir_mask(13, s, 30.0).iter().filter(|&&v| v == 0).count() > (s * s / 4) as usize);
    }

    #[test]
    fn equirect_round_trip() {
        for d in directions(1000) {
            let (u, v) = equirect_uv(d, 2048, 1024);
            let back = equirect_dir(u, v, 2048, 1024);
            for k in 0..3 {
                assert!(close(back[k], d[k], 1e-9));
            }
        }
    }

    #[test]
    fn heldout_groups_are_whole_panoramas_every_interval() {
        let mut names = Vec::new();
        for frame in 0..20 {
            for view in 0..VIEWS {
                // Frame 5 lost two views in SfM.
                if frame == 5 && view < 2 {
                    continue;
                }
                names.push(format!("v{view:02}_frame_{frame:06}.jpg"));
            }
        }
        let (heldout, groups) = heldout_groups(&names, 8);
        assert_eq!(groups, 3);
        let held: std::collections::BTreeSet<&str> = heldout.iter().map(|n| group_key(n).unwrap()).collect();
        assert_eq!(held.into_iter().collect::<Vec<_>>(), ["frame_000000.jpg", "frame_000008.jpg", "frame_000016.jpg"]);
        assert_eq!(heldout.len(), 3 * VIEWS);
        // No group in both sets.
        let train: Vec<&String> = names.iter().filter(|n| !heldout.contains(n)).collect();
        assert!(train.iter().all(|n| !heldout.iter().any(|h| group_key(h) == group_key(n))));
        assert_eq!(train.len() + heldout.len(), names.len());
    }

    #[test]
    fn group_keys_and_path_names() {
        assert_eq!(group_key("v04_frame_000017.jpg"), Some("frame_000017.jpg"));
        assert_eq!(group_key("frame_000017.jpg"), None);
        assert_eq!(group_key("v4_x.jpg"), None);
        let dir = tempfile::tempdir().unwrap();
        let names: Vec<String> = ["v01_b.jpg", "v00_b.jpg", "v00_a.jpg", "v13_a.jpg", "v00_c.jpg"].map(String::from).to_vec();
        assert_eq!(path_names(dir.path(), &names), names);
        let manifest = Manifest::new(1024, 30.0, vec!["c.jpg".into(), "a.jpg".into(), "b.jpg".into()]);
        manifest.write(dir.path()).unwrap();
        assert_eq!(Manifest::read(dir.path()).unwrap().unwrap(), manifest);
        assert_eq!(path_names(dir.path(), &names), ["v00_c.jpg", "v00_a.jpg", "v00_b.jpg"]);
        assert_eq!(manifest.view_name(4, "a.jpg"), "v04_a.jpg");
        assert_eq!(manifest.rotation(5), cam_from_pano(5));
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.path().join(MANIFEST)).unwrap()).unwrap();
        assert_eq!(json["path_prefix"], "v00_");
        assert_eq!(json["prefixes"].as_array().unwrap().len(), VIEWS);
        assert_eq!(json["cam_from_pano"][0], serde_json::json!([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
    }
}
