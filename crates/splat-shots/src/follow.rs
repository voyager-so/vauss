//! Follow-capture ("fix my footage"): a smooth camera path fitted to the capture's own trajectory.
//!
//! The SfM model holds one pose per extracted video frame. Frame files are named after their
//! source frame index (`frame_000123.jpg`, `00123.jpg`), so `index / src_fps` recovers when each
//! pose happened in the original clip. We drop mis-registered poses, then fit position and view
//! direction with a Gaussian-weighted local-quadratic smoother over time (no shrink at the ends),
//! level the horizon (zero roll about world up) and write keyframes on the original timing. The
//! render covers the full frame, unlike crop-and-warp stabilisers.

use glam::{DMat3, DVec3};
use serde::{Deserialize, Serialize};

use crate::paths::{CameraPath, Interpolation, Keyframe};
use crate::scene::{SceneFrame, median};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowSpec {
    /// Frame rate of the video the capture frames were extracted from.
    pub src_fps: f64,
    /// Gaussian sigma (seconds) of the trajectory smoother; larger = steadier, less faithful.
    pub smooth_s: f64,
    /// Separate sigma for the view direction (handheld shake is mostly rotation).
    pub smooth_rot_s: f64,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
}

impl Default for FollowSpec {
    fn default() -> Self {
        Self { src_fps: 30.0, smooth_s: 0.8, smooth_rot_s: 1.0, fps: 24, width: 1280, height: 720 }
    }
}

/// How much the smoother removed, in scene radii and degrees, per capture pose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowReport {
    pub poses: usize,
    pub rejected: Vec<String>,
    /// Source-clip time span the path covers (seconds); cut the "before" clip to this.
    pub src_start_s: f64,
    pub src_end_s: f64,
    pub timing_from_names: bool,
    pub pos_jitter_rms_radii: f64,
    pub rot_jitter_rms_deg: f64,
    pub rot_jitter_max_deg: f64,
    /// Frame-to-frame view rotation (deg/s) of the capture vs the smoothed path, 95th percentile.
    pub capture_turn_p95_deg_s: f64,
    pub smooth_turn_p95_deg_s: f64,
    pub spec: FollowSpec,
}

/// Trailing digits of the file stem: the source frame index.
fn frame_index(name: &str) -> Option<u64> {
    let stem = std::path::Path::new(name).file_stem()?.to_str()?;
    let digits: String = stem.chars().rev().take_while(char::is_ascii_digit).collect::<Vec<_>>().into_iter().rev().collect();
    digits.parse().ok()
}

struct Sample {
    t: f64,
    pos: DVec3,
    fwd: DVec3,
    name: String,
}

/// Gaussian-weighted local-quadratic fit at `t`: removes shake without cutting the corners of
/// orbits and turns (a local-linear or plain Gaussian fit pulls curved paths inward), and does not
/// shrink at the ends. Falls back to linear, then to the weighted mean, when the window is thin.
fn local_fit(ts: &[f64], vs: &[DVec3], t: f64, sigma: f64) -> DVec3 {
    let mut m = [0.0f64; 5]; // sum w d^k
    let mut mv = [DVec3::ZERO; 3]; // sum w d^k v
    for (tj, vj) in ts.iter().zip(vs) {
        let d = (tj - t) / sigma; // unitless, keeps the system well conditioned
        if d.abs() > 3.0 {
            continue;
        }
        let w = (-0.5 * d * d).exp();
        let mut p = w;
        for (k, mk) in m.iter_mut().enumerate() {
            *mk += p;
            if k < 3 {
                mv[k] += p * *vj;
            }
            p *= d;
        }
    }
    if m[0] <= 0.0 {
        let i = ts.partition_point(|x| *x < t).min(ts.len() - 1);
        return vs[i];
    }
    let a = DMat3::from_cols_array(&[m[0], m[1], m[2], m[1], m[2], m[3], m[2], m[3], m[4]]);
    if a.determinant().abs() > 1e-6 * m[0].powi(3) {
        let inv = a.inverse();
        // intercept = first row of inv . [mv0, mv1, mv2]
        return inv.x_axis.x * mv[0] + inv.y_axis.x * mv[1] + inv.z_axis.x * mv[2];
    }
    let det = m[0] * m[2] - m[1] * m[1];
    if det.abs() > 1e-9 * m[0] * m[0] {
        return (m[2] * mv[0] - m[1] * mv[1]) / det;
    }
    mv[0] / m[0]
}

fn p95(mut v: Vec<f64>) -> f64 {
    crate::scene::quantile(&mut v, 0.95)
}

fn turn_rates(ts: &[f64], fwd: &[DVec3]) -> Vec<f64> {
    ts.windows(2)
        .zip(fwd.windows(2))
        .filter(|(t, _)| t[1] > t[0])
        .map(|(t, f)| f[0].dot(f[1]).clamp(-1.0, 1.0).acos().to_degrees() / (t[1] - t[0]))
        .collect()
}

pub fn plan_follow(frame: &SceneFrame, spec: &FollowSpec, name: &str) -> anyhow::Result<(CameraPath, FollowReport)> {
    anyhow::ensure!(frame.cameras.len() >= 4, "follow_capture needs at least 4 registered capture frames");
    let by_name = frame.cameras.iter().all(|c| frame_index(&c.name).is_some());
    let mut samples: Vec<Sample> = frame
        .cameras
        .iter()
        .enumerate()
        .map(|(i, c)| Sample {
            // Without numbered names assume the frames were sampled evenly at 4 fps (the extract default).
            t: if by_name { frame_index(&c.name).unwrap() as f64 / spec.src_fps } else { i as f64 / 4.0 },
            pos: c.center(),
            fwd: c.forward(),
            name: c.name.clone(),
        })
        .collect();
    samples.sort_by(|a, b| a.t.total_cmp(&b.t));

    // Mis-registered frames jump away from their neighbours: compare each pose to the straight-line
    // prediction from the poses either side of it (from the next two at the ends).
    let n = samples.len();
    let lerp = |a: &Sample, b: &Sample, t: f64| a.pos + (b.pos - a.pos) * ((t - a.t) / (b.t - a.t).max(1e-9));
    let dev: Vec<f64> = (0..n)
        .map(|i| {
            let (a, b) = match i {
                0 => (1, 2),
                _ if i == n - 1 => (n - 3, n - 2),
                _ => (i - 1, i + 1),
            };
            samples[i].pos.distance(lerp(&samples[a], &samples[b], samples[i].t))
        })
        .collect();
    let cut = (6.0 * median(dev.clone())).max(0.02 * frame.radius);
    let mut rejected = vec![];
    let mut keep = vec![];
    for (i, s) in samples.into_iter().enumerate() {
        if dev[i] > cut { rejected.push(s.name) } else { keep.push(s) }
    }
    let samples = keep;
    anyhow::ensure!(samples.len() >= 4, "follow_capture: too few consistent capture poses");

    let ts: Vec<f64> = samples.iter().map(|s| s.t).collect();
    let ps: Vec<DVec3> = samples.iter().map(|s| s.pos).collect();
    let fs: Vec<DVec3> = samples.iter().map(|s| s.fwd).collect();
    let smooth_pos = |t: f64| local_fit(&ts, &ps, t, spec.smooth_s.max(1e-3));
    let smooth_fwd = |t: f64| local_fit(&ts, &fs, t, spec.smooth_rot_s.max(1e-3)).normalize();

    // Horizon: sample_path builds each frame with look_at_c2w about world up, so roll is zero.
    let up = frame.up;
    let (t0, t1) = (ts[0], ts[ts.len() - 1]);
    let duration_s = t1 - t0;
    // Keyframes every 1/8 s of output: dense enough that Catmull-Rom adds no wobble of its own.
    let nk = ((duration_s * 8.0).ceil() as usize + 1).max(2);
    // 9:16 keeps the capture's vertical field of view.
    let vfov = frame.median_vfov_deg();
    let look_dist = frame.radius;
    let keyframes: Vec<Keyframe> = (0..nk)
        .map(|k| {
            let u = k as f64 / (nk - 1) as f64;
            let tq = t0 + (t1 - t0) * u;
            let p = smooth_pos(tq);
            let f = smooth_fwd(tq);
            Keyframe { t: duration_s * u, position: p.to_array(), look_at: (p + f * look_dist).to_array(), vfov_deg: vfov, roll_deg: 0.0 }
        })
        .collect();

    let pos_d: Vec<f64> = ts.iter().zip(&ps).map(|(t, p)| p.distance(smooth_pos(*t)) / frame.radius).collect();
    let rot_d: Vec<f64> = ts.iter().zip(&fs).map(|(t, f)| f.dot(smooth_fwd(*t)).clamp(-1.0, 1.0).acos().to_degrees()).collect();
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    let sm_f: Vec<DVec3> = ts.iter().map(|t| smooth_fwd(*t)).collect();
    let report = FollowReport {
        poses: ts.len(),
        rejected,
        src_start_s: r3(t0),
        src_end_s: r3(t1),
        timing_from_names: by_name,
        pos_jitter_rms_radii: r3(rms(&pos_d)),
        rot_jitter_rms_deg: r3(rms(&rot_d)),
        rot_jitter_max_deg: r3(rot_d.iter().copied().fold(0.0, f64::max)),
        capture_turn_p95_deg_s: r3(p95(turn_rates(&ts, &fs))),
        smooth_turn_p95_deg_s: r3(p95(turn_rates(&ts, &sm_f))),
        spec: spec.clone(),
    };
    let path = CameraPath {
        name: name.to_string(),
        fps: spec.fps,
        width: spec.width,
        height: spec.height,
        duration_s,
        up: up.to_array(),
        keyframes,
        interpolation: Interpolation::CatmullRom,
        movement: None,
    };
    Ok((path, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_index_parses_trailing_digits() {
        assert_eq!(frame_index("frame_000123.jpg"), Some(123));
        assert_eq!(frame_index("00042.png"), Some(42));
        assert_eq!(frame_index("img.jpg"), None);
    }

    #[test]
    fn local_fit_keeps_curves_and_removes_shake() {
        let ts: Vec<f64> = (0..200).map(|i| i as f64 / 30.0).collect();
        let line: Vec<DVec3> = ts.iter().map(|t| DVec3::new(*t, 2.0 * t * t, 0.0)).collect();
        let shaky: Vec<DVec3> = ts.iter().zip(&line).map(|(t, p)| *p + DVec3::Z * 0.05 * (t * 40.0).sin()).collect();
        for t in [0.0, 3.0, ts[199]] {
            let exact = DVec3::new(t, 2.0 * t * t, 0.0);
            assert!(local_fit(&ts, &line, t, 0.5).distance(exact) < 1e-9); // no shrink at the ends
            assert!(local_fit(&ts, &shaky, t, 0.5).distance(exact) < 0.01);
        }
    }
}
