//! Camera paths: parametric moves placed in the scene frame, a keyframe JSON format, and risk.
//!
//! A `MoveSpec` is what the agent writes from a text description ("slow orbit at dusk, 5s").
//! `plan_path` expands it into a `CameraPath` of explicit keyframes (world positions + look-at),
//! the format the renderer, the Blender export and the agent share and can hand-edit.
//! Distances in a MoveSpec are in scene radii (median capture camera -> focus distance), so the
//! same spec works on a tabletop object and a building.

use std::f64::consts::PI;
use std::sync::LazyLock;

use glam::{DMat4, DVec3};
use regex::Regex;
use serde::{Deserialize, Serialize};
use splat_contracts::ShotMove;

use crate::scene::{SceneFrame, horizontal_basis};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ease {
    Linear,
    EaseInOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Cw,
    Ccw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveSpec {
    #[serde(rename = "move")]
    pub movement: ShotMove,
    pub duration_s: f64,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    /// None = median of the capture cameras.
    pub vfov_deg: Option<f64>,
    pub ease: Ease,
    /// Orbit: arc in degrees; other moves: scene radii travelled.
    pub amount: Option<f64>,
    /// Orbit radius / start distance relative to the capture (1.0 = as captured).
    pub distance_scale: f64,
    /// Scene radii along up.
    pub height_offset: f64,
    /// Capture image name to start from; None = best observed.
    pub start_camera: Option<String>,
    pub direction: Direction,
    /// Look-at override in the training frame.
    pub target: Option<[f64; 3]>,
    /// The text the spec came from (kept for video generation).
    pub prompt: Option<String>,
}

impl MoveSpec {
    pub fn new(movement: ShotMove) -> Self {
        Self {
            movement,
            duration_s: 5.0,
            fps: 24,
            width: 1280,
            height: 720,
            vfov_deg: None,
            ease: Ease::EaseInOut,
            amount: None,
            distance_scale: 1.0,
            height_offset: 0.0,
            start_camera: None,
            direction: Direction::Ccw,
            target: None,
            prompt: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Keyframe {
    pub t: f64,
    pub position: [f64; 3],
    pub look_at: [f64; 3],
    pub vfov_deg: f64,
    #[serde(default)]
    pub roll_deg: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    CatmullRom,
    Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraPath {
    pub name: String,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub duration_s: f64,
    pub up: [f64; 3],
    pub keyframes: Vec<Keyframe>,
    #[serde(default = "d_catmull")]
    pub interpolation: Interpolation,
    #[serde(rename = "move", default)]
    pub movement: Option<MoveSpec>,
}

fn d_catmull() -> Interpolation {
    Interpolation::CatmullRom
}

impl CameraPath {
    pub fn frame_count(&self) -> usize {
        ((self.duration_s * self.fps as f64).round() as usize).max(1)
    }
}

// ---- text -> MoveSpec (no-LLM fallback; the agent normally fills PlanShot fields itself) ----

static MOVE_WORDS: LazyLock<Vec<(Regex, ShotMove)>> = LazyLock::new(|| {
    [
        (r"figure[- ]?(8|eight)", ShotMove::FigureEight),
        (r"orbit|circle|turntable|around", ShotMove::Orbit),
        (r"push[- ]?in|dolly[- ]?in|move in|zoom in|approach", ShotMove::PushIn),
        (r"pull[- ]?(out|back)|dolly[- ]?out|zoom out|reveal", ShotMove::PullOut),
        (r"crane[- ]?down|boom[- ]?down|descend", ShotMove::CraneDown),
        (r"crane|boom|rise|rising|jib", ShotMove::CraneUp),
        (r"dolly[- ]?left|truck[- ]?left|slide left", ShotMove::DollyLeft),
        (r"dolly|truck|slide|track", ShotMove::DollyRight),
    ]
    .into_iter()
    .map(|(p, m)| (Regex::new(p).unwrap(), m))
    .collect()
});

fn re(p: &str) -> Regex {
    Regex::new(p).unwrap()
}

pub fn default_amount(m: ShotMove) -> f64 {
    match m {
        ShotMove::Orbit => 90.0,
        ShotMove::FigureEight => 0.15,
        _ => 0.35,
    }
}

/// Parse a text description; `movement` (when given) wins over the words in the text.
pub fn spec_from_prompt(prompt: &str, movement: Option<ShotMove>) -> MoveSpec {
    let p = prompt.to_lowercase();
    let mv = movement
        .or_else(|| MOVE_WORDS.iter().find(|(r, _)| r.is_match(&p)).map(|(_, m)| *m))
        .unwrap_or(ShotMove::Orbit);
    let mut s = MoveSpec::new(mv);
    s.prompt = Some(prompt.to_string());
    if let Some(c) = re(r"(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)").captures(&p) {
        s.duration_s = c[1].parse().unwrap_or(5.0);
    }
    let speed = if re(r"slow|gentle|subtle").is_match(&p) {
        0.6
    } else if re(r"fast|quick|whip").is_match(&p) {
        1.4
    } else {
        1.0
    };
    s.amount = Some(if let Some(c) = re(r"(\d+)\s*(?:°|deg)").captures(&p) {
        c[1].parse().unwrap()
    } else if matches!(mv, ShotMove::Orbit) && re(r"full|360").is_match(&p) {
        360.0
    } else {
        default_amount(mv) * speed
    });
    if p.contains("clockwise") && !p.contains("counter") {
        s.direction = Direction::Cw;
    }
    if re(r"low[- ]angle|from below").is_match(&p) {
        s.height_offset = -0.25;
    } else if re(r"high angle|overhead|bird").is_match(&p) {
        s.height_offset = 0.5;
    }
    if re(r"vertical|9:16|portrait").is_match(&p) {
        (s.width, s.height) = (720, 1280);
    }
    s
}

// ---- planning ----

fn ease(u: f64, kind: Ease) -> f64 {
    match kind {
        Ease::Linear => u,
        Ease::EaseInOut => 0.5 - 0.5 * (PI * u).cos(),
    }
}

/// Best-observed capture camera: facing the target most directly, at a typical distance.
fn pick_start(frame: &SceneFrame, spec: &MoveSpec) -> anyhow::Result<usize> {
    if let Some(name) = &spec.start_camera {
        return frame
            .cameras
            .iter()
            .position(|c| &c.name == name)
            .ok_or_else(|| anyhow::anyhow!("start_camera {name:?} is not a capture image"));
    }
    let score = |c: &crate::scene::CaptureCamera| {
        let to_t = frame.target - c.center();
        let d = to_t.length();
        to_t.dot(c.forward()) / d - 0.5 * (d / frame.radius).ln().abs()
    };
    Ok((0..frame.cameras.len())
        .max_by(|&a, &b| score(&frame.cameras[a]).total_cmp(&score(&frame.cameras[b])))
        .unwrap())
}

/// Start a partial orbit on the densest run of capture azimuths so it stays observed.
fn orbit_start_azimuth(frame: &SceneFrame, spec: &MoveSpec, arc: f64) -> anyhow::Result<f64> {
    if spec.start_camera.is_some() {
        return Ok(frame.azimuth(frame.cameras[pick_start(frame, spec)?].center()));
    }
    let az: Vec<f64> = frame.cameras.iter().map(|c| frame.azimuth(c.center())).collect();
    let (mut best, mut best_n) = (0.0, 0usize);
    for k in 0..72 {
        let a0 = -PI + 2.0 * PI * k as f64 / 72.0;
        let n = az.iter().filter(|a| (*a - a0).rem_euclid(2.0 * PI) <= arc).count();
        if n > best_n {
            (best, best_n) = (a0, n);
        }
    }
    Ok(if spec.direction == Direction::Ccw { best } else { best + arc })
}

pub fn plan_path(frame: &SceneFrame, spec: &MoveSpec, name: &str) -> anyhow::Result<CameraPath> {
    let vfov = spec.vfov_deg.unwrap_or_else(|| frame.median_vfov_deg());
    let (up, r) = (frame.up, frame.radius);
    let target = spec.target.map(DVec3::from_array).unwrap_or(frame.target);
    let n = ((spec.duration_s * 4.0).round() as usize + 1).max(2);
    let us: Vec<f64> = (0..n).map(|i| ease(i as f64 / (n - 1) as f64, spec.ease)).collect();
    let amount = spec.amount.unwrap_or_else(|| default_amount(spec.movement));
    let mut pos = Vec::with_capacity(n);
    let mut look = vec![target; n];

    if matches!(spec.movement, ShotMove::Orbit) {
        let (e1, e2) = horizontal_basis(up);
        let rel: Vec<DVec3> = frame.centers().iter().map(|c| *c - frame.target).collect();
        let ring = crate::scene::median(rel.iter().map(|v| (*v - v.dot(up) * up).length()).collect())
            * spec.distance_scale;
        let height = crate::scene::median(rel.iter().map(|v| v.dot(up)).collect()) + spec.height_offset * r;
        let arc = amount.to_radians();
        let a0 = orbit_start_azimuth(frame, spec, arc)?;
        let sign = if spec.direction == Direction::Ccw { 1.0 } else { -1.0 };
        for u in &us {
            let a = a0 + sign * arc * u;
            pos.push(target + ring * (a.cos() * e1 + a.sin() * e2) + height * up);
        }
    } else {
        let cam = &frame.cameras[pick_start(frame, spec)?];
        let mut start = cam.center() + spec.height_offset * r * up;
        if spec.distance_scale != 1.0 {
            start = target + (start - target) * spec.distance_scale;
        }
        let to_t = target - start;
        let dist = to_t.length();
        let fwd = to_t / dist;
        let right = fwd.cross(up).normalize();
        let cam_up = right.cross(fwd);
        let step = amount * r;
        for (i, u) in us.iter().enumerate() {
            pos.push(match spec.movement {
                // never fly through the subject
                ShotMove::PushIn => start + fwd * step.min(0.8 * dist) * u,
                ShotMove::PullOut => start + fwd * step.min(0.8 * dist) * (1.0 - u),
                ShotMove::CraneUp => start + up * step * u,
                ShotMove::CraneDown => start - up * step * u,
                ShotMove::DollyLeft | ShotMove::DollyRight => {
                    let s = if matches!(spec.movement, ShotMove::DollyRight) { 1.0 } else { -1.0 } * (u - 0.5);
                    look[i] = target + right * step * 0.5 * s; // parallax, not a pan
                    start + right * step * s
                }
                ShotMove::FigureEight => {
                    let th = 2.0 * PI * i as f64 / (n - 1) as f64; // closed loop, no easing
                    start + step * (th.sin() * right + th.sin() * th.cos() * cam_up)
                }
                ShotMove::Orbit => unreachable!(),
            });
        }
    }
    let keyframes = (0..n)
        .map(|i| Keyframe {
            t: spec.duration_s * i as f64 / (n - 1) as f64,
            position: pos[i].to_array(),
            look_at: look[i].to_array(),
            vfov_deg: vfov,
            roll_deg: 0.0,
        })
        .collect();
    Ok(CameraPath {
        name: name.to_string(),
        fps: spec.fps,
        width: spec.width,
        height: spec.height,
        duration_s: spec.duration_s,
        up: up.to_array(),
        keyframes,
        interpolation: Interpolation::CatmullRom,
        movement: Some(spec.clone()),
    })
}

// ---- sampling ----

/// Uniform Catmull-Rom through keyframe values, clamped ends.
fn catmull_rom(ts: &[f64], vals: &[[f64; 5]], t: f64) -> [f64; 5] {
    let last = ts.len() - 1;
    let i = ts.partition_point(|&x| x <= t).saturating_sub(1).min(last - 1);
    let s = (t - ts[i]) / (ts[i + 1] - ts[i]).max(1e-9);
    let (p0, p1, p2, p3) = (vals[i.saturating_sub(1)], vals[i], vals[i + 1], vals[(i + 2).min(last)]);
    std::array::from_fn(|k| {
        0.5 * (2.0 * p1[k]
            + (-p0[k] + p2[k]) * s
            + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * s * s
            + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * s * s * s)
    })
}

fn linear(ts: &[f64], vals: &[[f64; 5]], t: f64) -> [f64; 5] {
    let i = ts.partition_point(|&x| x <= t).saturating_sub(1).min(ts.len() - 2);
    let s = ((t - ts[i]) / (ts[i + 1] - ts[i]).max(1e-9)).clamp(0.0, 1.0);
    std::array::from_fn(|k| vals[i][k] * (1.0 - s) + vals[i + 1][k] * s)
}

pub fn look_at_c2w(pos: DVec3, look: DVec3, up: DVec3, roll_deg: f64) -> DMat4 {
    let z = (look - pos).normalize();
    let mut x = z.cross(up); // camera right
    if x.length() < 1e-6 {
        x = z.cross(if z.x.abs() < 0.9 { DVec3::X } else { DVec3::Y }); // looking along up
    }
    let mut x = x.normalize();
    let mut y = z.cross(x);
    if roll_deg != 0.0 {
        let (s, c) = roll_deg.to_radians().sin_cos();
        (x, y) = (c * x + s * y, -s * x + c * y);
    }
    DMat4::from_cols(x.extend(0.0), y.extend(0.0), z.extend(0.0), pos.extend(1.0))
}

/// One sampled frame: OpenCV c2w and vertical fov.
#[derive(Debug, Clone, Copy)]
pub struct PathFrame {
    pub c2w: DMat4,
    pub vfov_deg: f64,
}

impl PathFrame {
    pub fn fy(&self, height: u32) -> f64 {
        height as f64 / (2.0 * (self.vfov_deg.to_radians() / 2.0).tan())
    }
}

/// Frames at the path's fps; count = round(duration * fps).
pub fn sample_path(path: &CameraPath) -> Vec<PathFrame> {
    let kf = &path.keyframes;
    let ts: Vec<f64> = kf.iter().map(|k| k.t).collect();
    let n = path.frame_count();
    let pv: Vec<[f64; 5]> = kf.iter().map(|k| [k.position[0], k.position[1], k.position[2], k.vfov_deg, k.roll_deg]).collect();
    let lv: Vec<[f64; 5]> = kf.iter().map(|k| [k.look_at[0], k.look_at[1], k.look_at[2], 0.0, 0.0]).collect();
    let up = DVec3::from_array(path.up);
    (0..n)
        .map(|i| {
            if kf.len() == 1 {
                let k = &kf[0];
                let c2w = look_at_c2w(k.position.into(), k.look_at.into(), up, k.roll_deg);
                return PathFrame { c2w, vfov_deg: k.vfov_deg };
            }
            let t = if n == 1 { ts[0] } else { ts[0] + (ts[ts.len() - 1] - ts[0]) * i as f64 / (n - 1) as f64 };
            let f = if path.interpolation == Interpolation::CatmullRom && kf.len() > 2 { catmull_rom } else { linear };
            let (p, l) = (f(&ts, &pv, t), f(&ts, &lv, t));
            PathFrame {
                c2w: look_at_c2w(DVec3::new(p[0], p[1], p[2]), DVec3::new(l[0], l[1], l[2]), up, p[4]),
                vfov_deg: p[3],
            }
        })
        .collect()
}

// ---- risk ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Risk {
    pub risk: String,
    pub novelty_mean: f64,
    pub novelty_max: f64,
    pub frac_frames_far: f64,
    pub per_frame: Vec<f64>,
}

/// How far each frame strays from what the capture observed.
///
/// Per frame: distance to the nearest capture camera whose view direction is within 30 deg, in
/// scene radii (2.0 if none looks that way). Within ~0.25 radii of a similarly-oriented capture
/// camera renders cleanly; beyond ~0.6 expect floaters and holes.
pub fn score_risk(frame: &SceneFrame, frames: &[PathFrame]) -> Risk {
    let cos30 = 30f64.to_radians().cos();
    let d: Vec<f64> = frames
        .iter()
        .map(|f| {
            let (pos, fwd) = (f.c2w.w_axis.truncate(), f.c2w.z_axis.truncate());
            frame
                .cameras
                .iter()
                .filter(|c| c.forward().dot(fwd) > cos30)
                .map(|c| c.center().distance(pos) / frame.radius)
                .fold(2.0, f64::min)
        })
        .collect();
    let worst = d.iter().copied().fold(0.0, f64::max);
    let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
    Risk {
        risk: if worst < 0.25 { "low" } else if worst < 0.6 { "medium" } else { "high" }.into(),
        novelty_mean: r3(d.iter().sum::<f64>() / d.len().max(1) as f64),
        novelty_max: r3(worst),
        frac_frames_far: r3(d.iter().filter(|x| **x > 0.6).count() as f64 / d.len().max(1) as f64),
        per_frame: d.into_iter().map(r3).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_parsing() {
        let s = spec_from_prompt("slow orbit at dusk, 5s", None);
        assert!(matches!(s.movement, ShotMove::Orbit));
        assert_eq!(s.duration_s, 5.0);
        assert!((s.amount.unwrap() - 54.0).abs() < 1e-9);
        assert!(matches!(spec_from_prompt("dramatic push-in, 3 seconds", None).movement, ShotMove::PushIn));
        assert!(matches!(spec_from_prompt("crane down over the roof", None).movement, ShotMove::CraneDown));
        assert!(matches!(spec_from_prompt("figure 8 around the statue", None).movement, ShotMove::FigureEight));
        assert_eq!(spec_from_prompt("pull back to reveal the street, vertical", None).height, 1280);
        // explicit move wins and gets its own default amount
        let s = spec_from_prompt("orbit", Some(ShotMove::PushIn));
        assert!(matches!(s.movement, ShotMove::PushIn) && (s.amount.unwrap() - 0.35).abs() < 1e-9);
    }

    #[test]
    fn look_at_is_right_handed_opencv() {
        let m = look_at_c2w(DVec3::new(0.0, 0.0, -5.0), DVec3::ZERO, DVec3::new(0.0, -1.0, 0.0), 0.0);
        let (x, y, z) = (m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate());
        assert!(z.abs_diff_eq(DVec3::Z, 1e-12));
        assert!(y.abs_diff_eq(DVec3::Y, 1e-12)); // world up is -y, camera y points down
        assert!(x.cross(y).abs_diff_eq(z, 1e-12));
    }

    #[test]
    fn camera_path_json_roundtrip() {
        let p = CameraPath {
            name: "t".into(), fps: 24, width: 64, height: 64, duration_s: 1.0, up: [0.0, 0.0, 1.0],
            keyframes: vec![
                Keyframe { t: 0.0, position: [1.0, 0.0, 0.0], look_at: [0.0; 3], vfov_deg: 40.0, roll_deg: 0.0 },
                Keyframe { t: 1.0, position: [0.0, 1.0, 0.0], look_at: [0.0; 3], vfov_deg: 50.0, roll_deg: 0.0 },
            ],
            interpolation: Interpolation::CatmullRom, movement: None,
        }; // fmt: skip
        let q: CameraPath = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        let (a, b) = (sample_path(&p), sample_path(&q));
        assert_eq!(a.len(), 24);
        assert!(a.iter().zip(&b).all(|(x, y)| x.c2w.abs_diff_eq(y.c2w, 1e-12)));
        assert!((a[23].vfov_deg - 50.0).abs() < 1e-9);
    }
}
