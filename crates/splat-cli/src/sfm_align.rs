//! One world for every package of a run. A video's first view trains on a quick keyframe SfM and
//! the full model on the full SfM: two independent COLMAP worlds, so keyframes authored on the
//! preview pointed elsewhere after the swap to the full package. [`share_world`] maps the full SfM
//! onto the quick one (a Sim3 fitted to the camera centres both solved) before the full model trains,
//! and pins the quick capture's export frame (splat_webexport::FRAME_FILE) in both SfMs, so every
//! package exports with the same transform_from_source, focus, radius and start camera.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat3, DMat4, DQuat, DVec3};
use serde_json::{Value, json};
use splat_core::{Run, RunCtx};

/// Least shared images for a Sim3 (3 non-collinear centres determine one).
const MIN_SHARED: usize = 3;

/// Align the run's full SfM `full` onto its quick SfM `quick` (artifact ids) and pin the shared export
/// frame. Never fails the run: with too few shared images (or an unreadable model) the full SfM
/// keeps its own world and frame, and the run journals why.
pub fn share_world(run: &Run, quick: &str, full: &str) {
    let aligned = run.path(quick).and_then(|q| run.path(full).map(|f| (q, f))).and_then(|(q, f)| align(&q, &f, quick));
    let _ = match aligned {
        Ok(record) => {
            if record["rms_over_radius"].as_f64().is_some_and(|r| r > splat_core::report::SFM_ALIGN_WARN) {
                eprintln!("Warning: the full SfM fits the preview's world only loosely; keyframes set on the preview may shift.");
            }
            eprintln!("SfM aligned onto the preview's: {} shared images, residual rms {:.4} ({:.2}% of the scene radius)",
                record["shared"], record["rms"].as_f64().unwrap_or(f64::NAN), 100.0 * record["rms_over_radius"].as_f64().unwrap_or(f64::NAN));
            run.journal("sfm_aligned", json!({"sfm_id":full,"to_sfm_id":quick,"alignment":record}))
        }
        Err(error) => {
            eprintln!("Warning: full SfM not aligned to the preview's ({error:#}); packages may not share one frame.");
            run.journal("sfm_align_skipped", json!({"sfm_id":full,"to_sfm_id":quick,"error":format!("{error:#}")}))
        }
    };
}

/// `splat __align_sfm <quick_dir> <full_dir>`: [`share_world`] on two capture dirs; prints the record.
pub fn align_dirs(args: &[String]) -> Result<()> {
    let [quick, full] = args else { bail!("usage: splat __align_sfm <quick_sfm_dir> <full_sfm_dir>") };
    let quick = Path::new(quick);
    let id = quick.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    println!("{}", serde_json::to_string_pretty(&align(quick, Path::new(full), &id)?)?);
    Ok(())
}

/// Fit the Sim3 from `full`'s world into `quick`'s and write both frame files (the full one
/// composes the Sim3 into the quick frame's transform); the alignment record. Idempotent: a full SfM already aligned to `quick_id` is left as is.
fn align(quick: &Path, full: &Path, quick_id: &str) -> Result<Value> {
    let frame_file = full.join(splat_webexport::FRAME_FILE);
    if let Some(done) = fs::read(&frame_file).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .map(|v| v["aligned_to"].clone()).filter(|a| a["sfm"] == quick_id)
    {
        return Ok(done);
    }
    let (qs, fs_) = (splat_shots::scene::find_sparse_dir(quick)?, splat_shots::scene::find_sparse_dir(full)?);
    ensure!(fs_.join("images.bin").is_file() && qs.join("images.bin").is_file(), "not a binary COLMAP model");
    ensure!(trivial_rigs(&fs_.join("rigs.bin"))?, "multi-sensor rig: a scaled world would change its baselines");
    let q_poses = fs::read(qs.join("images.bin"))?;
    let q_centres: BTreeMap<String, DVec3> = images(&q_poses)?.into_iter().map(|i| { let c = i.centre(); (i.name, c) }).collect();
    let f_images = fs::read(fs_.join("images.bin"))?;
    let pairs: Vec<(DVec3, DVec3)> = images(&f_images)?.into_iter()
        .filter_map(|i| Some((i.centre(), *q_centres.get(&i.name)?))).collect();
    ensure!(pairs.len() >= MIN_SHARED, "only {} images solved by both SfMs (need {MIN_SHARED})", pairs.len());
    let sim = Sim3::fit_robust(&pairs)?;
    let residuals: Vec<f64> = pairs.iter().map(|(a, b)| sim.apply(*a).distance(*b)).collect();
    let rms = (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();

    // The export frame the preview was (or will be) packaged in: fitted to the quick capture alone.
    let mut frame = splat_webexport::capture_frame(quick)?;
    // Source units -> scene metres: the length of the transform's first column.
    let to_scene = (0..3).map(|k| frame["transform_from_source"][k].as_f64().unwrap_or(f64::NAN).powi(2)).sum::<f64>().sqrt();
    let radius = frame["radius"].as_f64().unwrap_or(f64::NAN) / to_scene;
    let q = sim.rotation;
    let record = json!({"sfm": quick_id, "shared": pairs.len(), "inliers": sim.inliers,
        "scale": sim.scale, "rotation_wxyz": [q.w, q.x, q.y, q.z], "translation": sim.translation.to_array(),
        "rms": rms, "max": residuals.iter().cloned().fold(0.0, f64::max), "rms_m": rms * to_scene,
        "rms_over_radius": rms / radius});

    // The full SfM keeps its own bytes (training is invariant to a similarity, and a rewritten
    // model no longer matches the SfM job's outputs kept on fal: night re-uploaded 36 MB, +125 s).
    // Its export frame instead maps full-world source points through the Sim3 into the quick
    // world, then through the quick capture's frame, so both packages share one scene.
    if !quick.join(splat_webexport::FRAME_FILE).is_file() {
        write_new(&quick.join(splat_webexport::FRAME_FILE), &serde_json::to_vec_pretty(&frame)?)?;
    }
    let quick_from_full = DMat4::from_scale_rotation_translation(DVec3::splat(sim.scale), sim.rotation, sim.translation);
    let m: Vec<f64> = frame["transform_from_source"].as_array().context("frame: no transform_from_source")?
        .iter().filter_map(Value::as_f64).collect();
    ensure!(m.len() == 16, "frame: transform_from_source is not a 4x4");
    frame["transform_from_source"] = json!((DMat4::from_cols_slice(&m) * quick_from_full).to_cols_array());
    // Last: the marker that this SfM's frame maps into the quick world.
    frame["aligned_to"] = record.clone();
    write_new(&frame_file, &serde_json::to_vec_pretty(&frame)?)?;
    Ok(record)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("align.tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))
}

/// x -> s R x + t, mapping full-SfM world coordinates into the quick SfM's.
struct Sim3 {
    scale: f64,
    rotation: DQuat,
    translation: DVec3,
    inliers: usize,
}

impl Sim3 {
    fn apply(&self, p: DVec3) -> DVec3 {
        self.scale * (self.rotation * p) + self.translation
    }

    /// Least-squares fit (Umeyama scale, Horn rotation), refitted without centres off by more than
    /// 3x the median residual (a mis-solved frame in either SfM).
    fn fit_robust(pairs: &[(DVec3, DVec3)]) -> Result<Self> {
        let mut used: Vec<(DVec3, DVec3)> = pairs.to_vec();
        let mut sim = Self::fit(&used)?;
        for _ in 0..2 {
            let mut r: Vec<f64> = pairs.iter().map(|(a, b)| sim.apply(*a).distance(*b)).collect();
            let cut = 3.0 * splat_shots::scene::quantile(&mut r.clone(), 0.5);
            let kept: Vec<(DVec3, DVec3)> = pairs.iter().zip(r.drain(..)).filter(|(_, e)| *e <= cut).map(|(p, _)| *p).collect();
            if kept.len() == used.len() || kept.len() < MIN_SHARED {
                break;
            }
            used = kept;
            sim = Self::fit(&used)?;
        }
        Ok(sim)
    }

    fn fit(pairs: &[(DVec3, DVec3)]) -> Result<Self> {
        let n = pairs.len() as f64;
        let (ma, mb) = pairs.iter().fold((DVec3::ZERO, DVec3::ZERO), |(a, b), (p, q)| (a + *p, b + *q));
        let (ma, mb) = (ma / n, mb / n);
        // Cross-covariance S[i][j] = sum a_i b_j over centred points.
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
        // Horn (1987): the rotation is the top eigenvector of N, as a (w, x, y, z) quaternion.
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
        Ok(Self { scale, rotation, translation: mb - scale * (rotation * ma), inliers: pairs.len() })
    }

}

/// Eigenvector of the largest eigenvalue of a symmetric 4x4 (cyclic Jacobi).
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

struct Image {
    name: String,
    q: DQuat,
    t: DVec3,
}

impl Image {
    fn centre(&self) -> DVec3 {
        -(DMat3::from_quat(self.q).transpose() * self.t)
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self.b.get(self.at..self.at + n).context("truncated COLMAP file")?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn pose(&mut self) -> Result<(DQuat, DVec3)> {
        let (w, x, y, z) = (self.f64()?, self.f64()?, self.f64()?, self.f64()?);
        Ok((DQuat::from_xyzw(x, y, z, w), DVec3::new(self.f64()?, self.f64()?, self.f64()?)))
    }
    fn done(&self) -> Result<()> {
        ensure!(self.at == self.b.len(), "unexpected trailing bytes in a COLMAP file");
        Ok(())
    }
}

/// images.bin records: id, qvec (w x y z), tvec, camera id, name\0, points2D (x, y, point3D id).
fn images(b: &[u8]) -> Result<Vec<Image>> {
    let mut r = Reader { b, at: 0 };
    let n = r.u64()?;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        r.u32()?;
        let (q, t) = r.pose()?;
        r.u32()?;
        let end = b[r.at..].iter().position(|c| *c == 0).context("unterminated image name")?;
        let name = String::from_utf8_lossy(r.take(end + 1)?.split_last().map(|(_, s)| s).unwrap_or_default()).into_owned();
        let points = r.u64()? as usize;
        r.take(points * 24)?;
        out.push(Image { name, q, t });
    }
    r.done()?;
    Ok(out)
}

/// Every rig has a single sensor (sensor_from_rig is the identity, unaffected by scale). No rigs.bin:
/// a pre-3.12 model, one camera per image.
fn trivial_rigs(path: &Path) -> Result<bool> {
    let Ok(b) = fs::read(path) else { return Ok(true) };
    let mut r = Reader { b: &b, at: 0 };
    for _ in 0..r.u64()? {
        r.u32()?;
        match r.u32()? {
            0 => {}
            1 => {
                r.take(8)?;
            }
            _ => return Ok(false),
        }
    }
    if r.at != b.len() {
        bail!("rigs.bin: unexpected layout");
    }
    Ok(true)
}
