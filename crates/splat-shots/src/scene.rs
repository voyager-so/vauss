//! Scene frame for shot planning: capture cameras, sparse points, up-vector, focus point, scale.
//!
//! Everything is in the splat's training frame. COLMAP poses are mapped through Spirula's
//! `scene_transform.json` when one sits next to the checkpoint (identity on our test runs).
//! Camera convention throughout: OpenCV (x right, y down, z forward), c2w 4x4.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::{DMat3, DMat4, DVec3};
use splat_eval::colmap::{Model, read_model};

#[derive(Debug, Clone)]
pub struct CaptureCamera {
    pub name: String,
    pub c2w: DMat4,
    pub width: u32,
    pub height: u32,
    pub fy: f64,
}

impl CaptureCamera {
    pub fn center(&self) -> DVec3 {
        self.c2w.w_axis.truncate()
    }
    pub fn forward(&self) -> DVec3 {
        self.c2w.z_axis.truncate()
    }
    pub fn vfov_deg(&self) -> f64 {
        (2.0 * (self.height as f64 / (2.0 * self.fy)).atan()).to_degrees()
    }
}

#[derive(Debug, Clone)]
pub struct SceneFrame {
    pub cameras: Vec<CaptureCamera>,
    pub points: Vec<DVec3>,
    /// Unit world up.
    pub up: DVec3,
    /// Focus point the capture looks at.
    pub target: DVec3,
    /// Median camera -> target distance; the scene's natural unit.
    pub radius: f64,
    /// Height (along up) of the 5th percentile of nearby points, relative to target.
    pub ground: f64,
    /// Capture photos, named as in the SfM model.
    pub image_dir: Option<PathBuf>,
}

impl SceneFrame {
    pub fn centers(&self) -> Vec<DVec3> {
        self.cameras.iter().map(|c| c.center()).collect()
    }
    pub fn forwards(&self) -> Vec<DVec3> {
        self.cameras.iter().map(|c| c.forward()).collect()
    }
    pub fn median_vfov_deg(&self) -> f64 {
        median(self.cameras.iter().map(|c| c.vfov_deg()).collect())
    }

    /// Azimuth (rad) of a point around the target in the plane perpendicular to up.
    pub fn azimuth(&self, p: DVec3) -> f64 {
        let (e1, e2) = horizontal_basis(self.up);
        let v = p - self.target;
        v.dot(e2).atan2(v.dot(e1))
    }

    /// Arc of azimuths covered by capture cameras (360 minus the largest gap), degrees.
    pub fn azimuth_coverage_deg(&self) -> f64 {
        let mut az: Vec<f64> = self.centers().iter().map(|c| self.azimuth(*c).to_degrees()).collect();
        if az.len() < 2 {
            return 0.0;
        }
        az.sort_by(f64::total_cmp);
        let mut gap = az[0] + 360.0 - az[az.len() - 1];
        for w in az.windows(2) {
            gap = gap.max(w[1] - w[0]);
        }
        360.0 - gap
    }

    /// Index of the capture camera closest to a pose among those facing within `max_angle_deg`.
    pub fn nearest_capture(&self, c2w: &DMat4, max_angle_deg: f64) -> usize {
        let (pos, fwd) = (c2w.w_axis.truncate(), c2w.z_axis.truncate());
        let cos_max = max_angle_deg.to_radians().cos();
        let score = |c: &CaptureCamera| {
            let d = c.center().distance(pos);
            if c.forward().dot(fwd) < cos_max { d + 1e6 } else { d }
        };
        (0..self.cameras.len()).min_by(|&a, &b| score(&self.cameras[a]).total_cmp(&score(&self.cameras[b]))).unwrap_or(0)
    }

    pub fn summary(&self) -> serde_json::Value {
        let h: Vec<f64> = self.centers().iter().map(|c| (*c - self.target).dot(self.up)).collect();
        serde_json::json!({
            "num_cameras": self.cameras.len(),
            "num_points": self.points.len(),
            "up": self.up.to_array(),
            "target": self.target.to_array(),
            "radius": self.radius,
            "camera_height_median": median(h),
            "azimuth_coverage_deg": self.azimuth_coverage_deg(),
            "vfov_deg_median": self.median_vfov_deg(),
        })
    }
}

pub fn horizontal_basis(up: DVec3) -> (DVec3, DVec3) {
    let a = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let e1 = (a - a.dot(up) * up).normalize();
    (e1, up.cross(e1))
}

pub fn median(mut v: Vec<f64>) -> f64 {
    quantile(&mut v, 0.5)
}

/// Linear-interpolated quantile (numpy's default).
pub fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let pos = q * (v.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos.fract());
    if i + 1 < v.len() { v[i] * (1.0 - f) + v[i + 1] * f } else { v[i] }
}

/// Same search order as Spirula's COLMAP detection; the model with the largest images.bin wins.
pub fn find_sparse_dir(data_dir: &Path) -> Result<PathBuf> {
    ["sparse/0", "colmap/sparse/0", "sparse", "colmap", "."]
        .iter()
        .map(|p| data_dir.join(p))
        .filter(|d| d.join("images.bin").exists() || d.join("images.txt").exists())
        .max_by_key(|d| std::fs::metadata(d.join("images.bin")).map(|m| m.len()).unwrap_or(0))
        .with_context(|| format!("no COLMAP model under {}", data_dir.display()))
}

/// 4x4 similarity from Spirula's scene_transform.json; identity if absent.
pub fn read_train_from_world(path: &Path) -> Result<DMat4> {
    if !path.exists() {
        return Ok(DMat4::IDENTITY);
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let rows: Vec<Vec<f64>> = serde_json::from_value(v["train_from_world"]["matrix_4x4"].clone())
        .context("scene_transform.json: train_from_world.matrix_4x4")?;
    if rows.len() != 4 || rows.iter().any(|r| r.len() != 4) {
        bail!("scene_transform.json: matrix_4x4 is not 4x4");
    }
    Ok(DMat4::from_cols_array_2d(&std::array::from_fn(|c| std::array::from_fn(|r| rows[r][c]))))
}

fn cameras_from_model(model: &Model) -> Result<Vec<CaptureCamera>> {
    let mut cams = Vec::with_capacity(model.images.len());
    for im in model.images.values() {
        let cam = model.cameras.get(&im.camera_id).context("image references a missing camera")?;
        // SIMPLE_* / FOV-style models share one focal length
        let fy = match cam.model.as_str() {
            "SIMPLE_PINHOLE" | "SIMPLE_RADIAL" | "RADIAL" | "FOV" | "SIMPLE_RADIAL_FISHEYE" | "RADIAL_FISHEYE" => cam.params[0],
            _ => cam.params[1],
        };
        let r = Model::rotation(im.qvec);
        let r_w2c = DMat3::from_cols_array_2d(&std::array::from_fn(|c| std::array::from_fn(|row| r[row][c])));
        let r_c2w = r_w2c.transpose();
        let center = DVec3::from_array(Model::center(im));
        cams.push(CaptureCamera {
            name: im.name.clone(),
            c2w: DMat4::from_cols(
                r_c2w.x_axis.extend(0.0),
                r_c2w.y_axis.extend(0.0),
                r_c2w.z_axis.extend(0.0),
                center.extend(1.0),
            ),
            width: cam.width as u32,
            height: cam.height as u32,
            fy,
        });
    }
    cams.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(cams)
}

fn nearest_point_to_rays(origins: &[DVec3], dirs: &[DVec3]) -> Option<DVec3> {
    let (mut a, mut b) = (DMat3::ZERO, DVec3::ZERO);
    for (o, d) in origins.iter().zip(dirs) {
        let p = DMat3::IDENTITY - outer(*d, *d);
        a += p;
        b += p * *o;
    }
    (a.determinant().abs() > 1e-9).then(|| a.inverse() * b)
}

fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

pub fn load_scene(data_dir: &Path, scene_transform: Option<&Path>) -> Result<SceneFrame> {
    let model = read_model(&find_sparse_dir(data_dir)?)?;
    let mut cams = cameras_from_model(&model)?;
    if cams.is_empty() {
        bail!("COLMAP model has no registered images");
    }
    let mut pts: Vec<DVec3> = model.points.iter().map(|p| DVec3::from_array(p.xyz)).collect();
    if let Some(st) = scene_transform {
        let t = read_train_from_world(st)?;
        if !t.abs_diff_eq(DMat4::IDENTITY, 1e-12) {
            let s = glam::DMat3::from_mat4(t).determinant().cbrt();
            for c in &mut cams {
                let mut m = t * c.c2w;
                for axis in [&mut m.x_axis, &mut m.y_axis, &mut m.z_axis] {
                    *axis /= s; // keep camera rotation orthonormal
                }
                c.c2w = m;
            }
            for p in &mut pts {
                *p = t.transform_point3(*p);
            }
        }
    }
    let centers: Vec<DVec3> = cams.iter().map(|c| c.center()).collect();
    let fwd: Vec<DVec3> = cams.iter().map(|c| c.forward()).collect();

    // Camera y points down in OpenCV; handheld/drone captures are mostly level, so the mean camera
    // "up" is a reliable world up regardless of how SfM gauged the frame.
    let up = (-cams.iter().map(|c| c.c2w.y_axis.truncate()).sum::<DVec3>()).normalize();

    // Focus: where optical axes converge (orbit-style captures). Walk-throughs have near-parallel
    // axes, so the solution lands far away or behind the cameras; then fall back to the median of
    // sparse points in front of most cameras.
    let mean_c = centers.iter().sum::<DVec3>() / centers.len() as f64;
    let extent = centers.iter().map(|c| c.distance(mean_c)).fold(0.0, f64::max) + 1e-9;
    let converge = nearest_point_to_rays(&centers, &fwd).filter(|t| {
        let in_front = centers.iter().zip(&fwd).filter(|(c, f)| (*t - **c).dot(**f) > 0.0).count();
        in_front as f64 > 0.8 * centers.len() as f64 && t.distance(mean_c) < 3.0 * extent
    });
    let target = match converge {
        Some(t) => t,
        None if !pts.is_empty() => {
            let vis: Vec<f64> = pts
                .iter()
                .map(|p| {
                    centers.iter().zip(&fwd).filter(|(c, f)| {
                        let v = *p - **c;
                        v.dot(**f) > 0.5 * v.length() // within ~60 deg of the axis
                    })
                    .count() as f64
                })
                .collect();
            let cut = quantile(&mut vis.clone(), 0.75);
            let sel: Vec<&DVec3> = pts.iter().zip(&vis).filter(|(_, v)| **v >= cut).map(|(p, _)| p).collect();
            let m = |f: fn(&DVec3) -> f64| median(sel.iter().map(|p| f(p)).collect());
            DVec3::new(m(|p| p.x), m(|p| p.y), m(|p| p.z))
        }
        None => mean_c,
    };
    let radius = median(centers.iter().map(|c| c.distance(target)).collect());
    // Ignore points far outside the capture (sky, distant background) for the ground estimate.
    let mut near_h: Vec<f64> =
        pts.iter().filter(|p| p.distance(target) < 3.0 * radius).map(|p| (*p - target).dot(up)).collect();
    let ground = if near_h.is_empty() { 0.0 } else { quantile(&mut near_h, 0.05) };
    let images = data_dir.join("images");
    Ok(SceneFrame { cameras: cams, points: pts, up, target, radius, ground, image_dir: images.is_dir().then_some(images) })
}
