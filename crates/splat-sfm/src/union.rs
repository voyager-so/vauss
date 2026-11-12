//! Union of COLMAP's model with a learned (DA3-seeded) model of the same database: COLMAP's frames keep
//! their poses bit for bit; the learned model contributes only the frames COLMAP left out, brought into
//! COLMAP's world by a Sim3 fitted to the camera centres both placed. COLMAP's model_merger is not used:
//! it re-adjusted the shared frames (2.3 deg on one room clip).
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use glam::{DMat3, DQuat, DVec3};

/// x -> s R x + t, mapping the learned model's world into COLMAP's. Adapted from splat-cli's sfm_align.
struct Sim3 {
    scale: f64,
    rotation: DQuat,
    translation: DVec3,
}

impl Sim3 {
    fn apply(&self, p: DVec3) -> DVec3 {
        self.scale * (self.rotation * p) + self.translation
    }

    /// Least squares, refitted twice without pairs off by more than 3x the median residual.
    fn fit_robust(pairs: &[(DVec3, DVec3)]) -> Result<Self> {
        let mut used = pairs.to_vec();
        let mut sim = Self::fit(&used)?;
        for _ in 0..2 {
            let r: Vec<f64> = pairs.iter().map(|(a, b)| sim.apply(*a).distance(*b)).collect();
            let mut sorted = r.clone();
            sorted.sort_by(f64::total_cmp);
            let cut = 3.0 * sorted[sorted.len() / 2];
            let kept: Vec<(DVec3, DVec3)> = pairs.iter().zip(&r).filter(|(_, e)| **e <= cut).map(|(p, _)| *p).collect();
            if kept.len() == used.len() || kept.len() < 3 {
                break;
            }
            used = kept;
            sim = Self::fit(&used)?;
        }
        Ok(sim)
    }

    fn fit(pairs: &[(DVec3, DVec3)]) -> Result<Self> {
        ensure!(pairs.len() >= 3, "fewer than 3 shared frames");
        let n = pairs.len() as f64;
        let (ma, mb) = pairs.iter().fold((DVec3::ZERO, DVec3::ZERO), |(a, b), (p, q)| (a + *p, b + *q));
        let (ma, mb) = (ma / n, mb / n);
        let mut s = [[0.0f64; 3]; 3];
        let mut var_a = 0.0;
        for (p, q) in pairs {
            let (a, b) = ((*p - ma).to_array(), (*q - mb).to_array());
            for i in 0..3 {
                for j in 0..3 {
                    s[i][j] += a[i] * b[j];
                }
                var_a += a[i] * a[i];
            }
        }
        ensure!(var_a > 0.0, "shared camera centres coincide");
        let [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = s;
        let nm = [
            [xx + yy + zz, yz - zy, zx - xz, xy - yx],
            [yz - zy, xx - yy - zz, xy + yx, zx + xz],
            [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
            [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
        ];
        let v = top_eigenvector(nm);
        let rotation = DQuat::from_xyzw(v[1], v[2], v[3], v[0]).normalize();
        let num: f64 = pairs.iter().map(|(p, q)| (*q - mb).dot(rotation * (*p - ma))).sum();
        let scale = num / var_a;
        ensure!(scale.is_finite() && scale > 0.0, "degenerate Sim3 (scale {scale})");
        Ok(Self { scale, rotation, translation: mb - scale * (rotation * ma) })
    }
}

fn top_eigenvector(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let off: f64 = (0..4).flat_map(|i| (0..4).filter(move |j| *j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let best = (0..4).max_by(|i, j| a[*i][*i].total_cmp(&a[*j][*j])).unwrap_or(0);
    [v[0][best], v[1][best], v[2][best], v[3][best]]
}

/// One image of a COLMAP TXT model: id, world-to-camera pose, camera id, name.
struct TxtImage {
    id: u32,
    q: DQuat,
    t: DVec3,
    camera: u32,
    name: String,
}

fn read_images_txt(path: &Path) -> Result<Vec<TxtImage>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
    let mut out = Vec::new();
    for l in lines.iter().step_by(2) {
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() < 10 {
            continue;
        }
        let n = |i: usize| -> Result<f64> { Ok(f[i].parse::<f64>()?) };
        out.push(TxtImage {
            id: f[0].parse()?,
            q: DQuat::from_xyzw(n(2)?, n(3)?, n(4)?, n(1)?).normalize(),
            t: DVec3::new(n(5)?, n(6)?, n(7)?),
            camera: f[8].parse()?,
            name: f[9..].join(" "),
        });
    }
    Ok(out)
}

fn centre(q: DQuat, t: DVec3) -> DVec3 {
    -(DMat3::from_quat(q).transpose() * t)
}

/// COLMAP's model plus the learned model's other frames in COLMAP's world (TXT round trip so the
/// frames/rigs records stay consistent), written as a binary model in `out`. Returns the names added.
/// Both models as TXT under `out` (colmap_txt/, learned_txt/) and their images.
fn to_txt(colmap: &Path, colmap_model: &Path, learned_model: &Path, out: &Path) -> Result<(PathBuf, Vec<TxtImage>, Vec<TxtImage>)> {
    let (a, b) = (out.join("colmap_txt"), out.join("learned_txt"));
    for (src, dst) in [(colmap_model, &a), (learned_model, &b)] {
        fs::create_dir_all(dst)?;
        let st = std::process::Command::new(colmap).env("QT_QPA_PLATFORM", "offscreen")
            .args(["model_converter", "--input_path"]).arg(src).arg("--output_path").arg(dst).args(["--output_type", "TXT"])
            .output()?;
        ensure!(st.status.success(), "model_converter to TXT failed: {}", String::from_utf8_lossy(&st.stderr));
    }
    let (ca, cb) = (read_images_txt(&a.join("images.txt"))?, read_images_txt(&b.join("images.txt"))?);
    Ok((a, ca, cb))
}

fn shared_centres(ca: &[TxtImage], cb: &[TxtImage]) -> Vec<(DVec3, DVec3)> {
    cb.iter().filter_map(|l| ca.iter().find(|c| c.name == l.name).map(|c| (centre(l.q, l.t), centre(c.q, c.t)))).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
}

/// How well the learned model's poses agree with COLMAP's on the frames both placed, after the same
/// Sim3 the union uses: camera-centre residuals in units of COLMAP's median step between consecutive
/// shared frames, and rotation differences in degrees. Tells bad learned poses (large residuals) from a
/// held-out check that is biased against a bigger model (small residuals, still rejected).
pub fn pose_agreement(colmap: &Path, colmap_model: &Path, learned_model: &Path, scratch: &Path) -> Result<serde_json::Value> {
    let _ = fs::remove_dir_all(scratch);
    fs::create_dir_all(scratch)?;
    let result = (|| {
        let (_, ca, cb) = to_txt(colmap, colmap_model, learned_model, scratch)?;
        let pairs = shared_centres(&ca, &cb);
        let sim = Sim3::fit_robust(&pairs)?;
        let mut shared: Vec<(&TxtImage, &TxtImage)> = cb.iter().filter_map(|l| ca.iter().find(|c| c.name == l.name).map(|c| (c, l))).collect();
        shared.sort_by(|x, y| x.0.name.cmp(&y.0.name));
        let step = median(shared.windows(2).map(|w| centre(w[0].0.q, w[0].0.t).distance(centre(w[1].0.q, w[1].0.t))).collect());
        let residual: Vec<f64> = shared.iter().map(|(c, l)| sim.apply(centre(l.q, l.t)).distance(centre(c.q, c.t)) / step.max(1e-12)).collect();
        // World-to-camera rotation of the learned frame in COLMAP's world vs COLMAP's own.
        let angle: Vec<f64> = shared.iter().map(|(c, l)| {
            let r = (l.q * sim.rotation.inverse()).normalize();
            (r * c.q.inverse()).normalize().w.abs().min(1.0).acos().to_degrees() * 2.0
        }).collect();
        let over = |v: &[f64], t: f64| v.iter().filter(|x| **x > t).count();
        let mut sorted = residual.clone();
        sorted.sort_by(f64::total_cmp);
        let p90 = sorted.get(sorted.len() * 9 / 10).copied().unwrap_or(f64::NAN);
        Ok(serde_json::json!({
            "n_shared": shared.len(),
            "median_step": step,
            "centre_residual_steps_median": median(residual.clone()),
            "centre_residual_steps_p90": p90,
            "frames_off_by_over_3_steps": over(&residual, 3.0),
            "rotation_deg_median": median(angle.clone()),
            "frames_rotated_over_5_deg": over(&angle, 5.0),
        }))
    })();
    let _ = fs::remove_dir_all(scratch);
    result
}

pub(crate) fn append_learned(colmap: &Path, opts: &crate::ReconstructOptions, colmap_model: &Path, learned_model: &Path, out: &Path) -> Result<Vec<String>> {
    let (a, ca, cb) = to_txt(colmap, colmap_model, learned_model, out)?;
    let names: BTreeSet<&str> = ca.iter().map(|i| i.name.as_str()).collect();
    let ids: BTreeSet<u32> = ca.iter().map(|i| i.id).collect();
    let pairs = shared_centres(&ca, &cb);
    let sim = Sim3::fit_robust(&pairs)?;
    let rig = fs::read_to_string(a.join("frames.txt"))?.lines().find(|l| !l.starts_with('#')).and_then(|l| l.split_whitespace().nth(1).map(str::to_owned)).unwrap_or_else(|| "1".into());
    let (mut images, mut frames) = (fs::read_to_string(a.join("images.txt"))?, fs::read_to_string(a.join("frames.txt"))?);
    let mut added = Vec::new();
    for l in cb.iter().filter(|l| !names.contains(l.name.as_str()) && !ids.contains(&l.id)) {
        // world_A = s R world_B + t; camera coordinates scaled by s: R' = R_b R^T, t' = s t_b - R' t.
        let rot = (l.q * sim.rotation.inverse()).normalize();
        let tr = sim.scale * l.t - rot * sim.translation;
        let pose = format!("{} {} {} {} {} {} {}", rot.w, rot.x, rot.y, rot.z, tr.x, tr.y, tr.z);
        images.push_str(&format!("{} {pose} {} {}\n\n", l.id, l.camera, l.name));
        frames.push_str(&format!("{} {rig} {pose} 1 CAMERA {} {}\n", l.id, l.camera, l.id));
        added.push(l.name.clone());
    }
    fs::write(a.join("images.txt"), images)?;
    fs::write(a.join("frames.txt"), frames)?;
    let merged = out.join("merged");
    fs::create_dir_all(&merged)?;
    let st = std::process::Command::new(colmap).env("QT_QPA_PLATFORM", "offscreen")
        .args(["model_converter", "--input_path"]).arg(&a).arg("--output_path").arg(&merged).args(["--output_type", "BIN"])
        .output()?;
    ensure!(st.status.success(), "model_converter to BIN failed: {}", String::from_utf8_lossy(&st.stderr));
    let _ = opts;
    Ok(added)
}

#[allow(dead_code)]
pub(crate) fn merged_dir(out: &Path) -> PathBuf {
    out.join("merged")
}
