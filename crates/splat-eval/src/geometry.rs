//! Conservative, similarity-invariant geometry diagnostics. Camera stillness alone is not a
//! failure: depths and the observation graph provide evidence independent of camera motion.

use crate::colmap::Model;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

const MIN_OBSERVATIONS: usize = 16;
const MAX_DEPTH_RATIO: f64 = 1000.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometryView {
    pub name: String,
    pub center: [f64; 3],
    pub median_depth: Option<f64>,
    pub observations: usize,
    pub invalid_depths: usize,
}

/// Small reproducible evidence record; no source images or arbitrary world-unit thresholds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometryEvidence {
    pub views: Vec<GeometryView>,
    /// Connected components of cameras joined by observations of the same extant 3D point.
    pub component_sizes: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TranslationJump {
    pub image: String,
    pub step_over_typical: f64,
    pub step_over_depth: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeometryHealth {
    /// "ok", "failed", or "insufficient". Insufficient evidence never asserts sound geometry.
    pub status: String,
    pub depth_p90_over_p10: Option<f64>,
    pub depth_scale_discontinuity: bool,
    pub invalid_depth_frames: Vec<String>,
    pub translation_jumps: Vec<TranslationJump>,
    pub component_sizes: Vec<usize>,
    pub disconnected: bool,
    pub insufficient_frames: Vec<String>,
    pub pass: bool,
}

fn quantile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    Some(v[((v.len() - 1) as f64 * fraction).round() as usize])
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2])
}

fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i];
    }
    i
}

/// Views of one 360 frame whose camera centres agree within this share of their scene depth form
/// one camera group (see [`geometry_evidence`]). A rig solve puts them at one point; solving the
/// views without the rig (the acceptance baseline) leaves solve noise, but they still share a
/// centre physically, so a centre that far off is a real disagreement.
const RIG_CENTRE_TOLERANCE: f64 = 0.02;

pub fn geometry_evidence(model: &Model) -> GeometryEvidence {
    let points: HashMap<_, _> = model.points.iter().map(|p| (p.id, p.xyz)).collect();
    let mut images: Vec<_> = model.images.values().collect();
    images.sort_by(|a, b| a.name.cmp(&b.name));
    let mut parents: Vec<_> = (0..images.len()).collect();
    let mut owner = HashMap::new();
    let views: Vec<_> = images
        .iter()
        .enumerate()
        .map(|(i, im)| {
            let r = Model::rotation(im.qvec);
            let mut depths = Vec::new();
            let mut invalid = 0;
            for pid in &im.obs_point_id {
                let Some(p) = points.get(pid) else { continue };
                if let Some(&j) = owner.get(pid) {
                    let a = root(&mut parents, i);
                    let b = root(&mut parents, j);
                    parents[a] = b;
                } else {
                    owner.insert(*pid, i);
                }
                let z = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + im.tvec[2];
                if z.is_finite() && z > 0.0 {
                    depths.push(z);
                } else {
                    invalid += 1;
                }
            }
            GeometryView {
                name: im.name.clone(),
                center: Model::center(im),
                median_depth: quantile(&depths, 0.5),
                observations: depths.len() + invalid,
                invalid_depths: invalid,
            }
        })
        .collect();
    // 360 rig frames (splat-core's panorama split: `vNN_<frame>`): the views of one frame share its
    // pose through the rig, not through 3D points (each direction gives features to one view), so
    // they are one camera group when their centres coincide.
    let mut frames: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, im) in images.iter().enumerate() {
        let b = im.name.as_bytes();
        if b.len() > 4 && b[0] == b'v' && b[1].is_ascii_digit() && b[2].is_ascii_digit() && b[3] == b'_' {
            frames.entry(&im.name[4..]).or_default().push(i);
        }
    }
    for members in frames.values().filter(|m| m.len() > 1) {
        let first = members[0];
        let scale = members.iter().filter_map(|&i| views[i].median_depth).fold(0.0f64, f64::max).max(1e-9);
        for &i in &members[1..] {
            if distance(&views[first].center, &views[i].center) <= RIG_CENTRE_TOLERANCE * scale {
                let a = root(&mut parents, i);
                let b = root(&mut parents, first);
                parents[a] = b;
            }
        }
    }
    let mut sizes = BTreeMap::new();
    for (i, _) in views
        .iter()
        .enumerate()
        .filter(|(_, v)| v.observations >= MIN_OBSERVATIONS)
    {
        *sizes.entry(root(&mut parents, i)).or_insert(0) += 1;
    }
    let mut component_sizes: Vec<_> = sizes.into_values().collect();
    component_sizes.sort_unstable_by(|a, b| b.cmp(a));
    GeometryEvidence {
        views,
        component_sizes,
    }
}

/// Only chronological captures get translation gates. Photo sets still get depth/graph gates.
pub fn assess_geometry(e: &GeometryEvidence, video: bool) -> GeometryHealth {
    let sufficient = |v: &&GeometryView| v.observations >= MIN_OBSERVATIONS;
    let depths: Vec<_> = e
        .views
        .iter()
        .filter(sufficient)
        .filter_map(|v| v.median_depth)
        .filter(|d| d.is_finite() && *d > 0.0)
        .collect();
    let ratio = (depths.len() >= 10)
        .then(|| quantile(&depths, 0.9).unwrap() / quantile(&depths, 0.1).unwrap());
    let depth_scale_discontinuity = ratio.is_some_and(|r| r > MAX_DEPTH_RATIO);
    let invalid_depth_frames: Vec<_> = e
        .views
        .iter()
        .filter(|v| {
            v.center.iter().any(|x| !x.is_finite())
                || (v.observations >= MIN_OBSERVATIONS
                    && v.invalid_depths as f64 / v.observations as f64 > 0.1)
        })
        .map(|v| v.name.clone())
        .collect();
    let insufficient_frames: Vec<_> = e
        .views
        .iter()
        .filter(|v| v.observations < MIN_OBSERVATIONS)
        .map(|v| v.name.clone())
        .collect();
    // Two substantial disconnected camera groups cannot define a common scale/translation.
    // Tiny weakly observed fragments stay visible in the inventory, rather than guessing a pose.
    let disconnected = e.component_sizes.get(1).is_some_and(|&n| n >= 3);
    let mut translation_jumps = Vec::new();
    if video && e.views.len() >= 5 {
        let gaps: Vec<_> = e
            .views
            .windows(2)
            .map(|w| {
                crate::sfm::frame_number(&w[0].name)
                    .zip(crate::sfm::frame_number(&w[1].name))
                    .and_then(|((a, x), (b, y))| (a == b && y > x).then_some((y - x) as f64))
            })
            .collect();
        let typical_gap =
            quantile(&gaps.iter().flatten().copied().collect::<Vec<_>>(), 0.5).unwrap_or(1.0);
        let steps: Vec<_> = e
            .views
            .windows(2)
            .zip(&gaps)
            .map(|(w, g)| {
                g.filter(|g| *g <= 4.0 * typical_gap)
                    .map(|g| distance(&w[0].center, &w[1].center) / g)
            })
            .collect();
        let typical =
            quantile(&steps.iter().flatten().copied().collect::<Vec<_>>(), 0.5).unwrap_or(0.0);
        for i in 1..e.views.len() - 1 {
            let (Some(before), Some(after), Some(depth)) =
                (steps[i - 1], steps[i], e.views[i].median_depth)
            else {
                continue;
            };
            if e.views[i].observations < MIN_OBSERVATIONS || depth <= 0.0 {
                continue;
            }
            let jump = before.min(after);
            let chord = distance(&e.views[i - 1].center, &e.views[i + 1].center)
                / (gaps[i - 1].unwrap() + gaps[i].unwrap());
            // An isolated out-and-back jump, large both relative to ordinary motion and observed
            // scene depth. Zero baseline (a stationary segment) needs the depth evidence too.
            if jump > 20.0 * typical && jump > 10.0 * chord && jump * typical_gap > 0.5 * depth {
                translation_jumps.push(TranslationJump {
                    image: e.views[i].name.clone(),
                    step_over_typical: jump / typical.max(jump * f64::EPSILON),
                    step_over_depth: jump * typical_gap / depth,
                });
            }
        }
    }
    let pass = !depth_scale_discontinuity
        && invalid_depth_frames.is_empty()
        && !disconnected
        && translation_jumps.is_empty();
    GeometryHealth {
        status: if !pass {
            "failed"
        } else if depths.len() < 10 || !insufficient_frames.is_empty() {
            "insufficient"
        } else {
            "ok"
        }
        .into(),
        depth_p90_over_p10: ratio,
        depth_scale_discontinuity,
        invalid_depth_frames,
        translation_jumps,
        component_sizes: e.component_sizes.clone(),
        disconnected,
        insufficient_frames,
        pass,
    }
}

pub fn geometry_health(model: &Model, video: bool) -> GeometryHealth {
    assess_geometry(&geometry_evidence(model), video)
}

#[cfg(test)]
mod rig_tests {
    use super::*;
    use crate::colmap::{Camera, Image, Point3D};

    /// 4 frames along x, each with a forward and a backward view at the same centre; every point is
    /// seen by one view direction only (as with 360 feature ownership masks).
    fn rig_model(prefixed: bool, offset: f64) -> Model {
        let mut model = Model { cameras: Default::default(), images: Default::default(), points: vec![] };
        model.cameras.insert(1, Camera { id: 1, model: "PINHOLE".into(), width: 64, height: 64, params: vec![32.0, 32.0, 32.0, 32.0] });
        for (side, z) in [(0u64, 5.0), (1, -5.0)] {
            for i in 0..20u64 {
                model.points.push(Point3D { id: side * 100 + i, xyz: [i as f64 * 0.2 - 2.0, 0.1 * i as f64, z], rgb: [0; 3], error: 0.0, track_len: 4 });
            }
        }
        let mut id = 1;
        for k in 0..4 {
            for (side, q) in [(0u64, [1.0, 0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.0, 0.0])] {
                let c = [k as f64 + if side == 1 { offset } else { 0.0 }, 0.0, 0.0];
                let t = if side == 0 { [-c[0], 0.0, 0.0] } else { [c[0], 0.0, 0.0] };
                let name = if prefixed { format!("v{side:02}_frame_{k}.jpg") } else { format!("cam{side}_frame_{k}.jpg") };
                let ids: Vec<u64> = (0..20).map(|i| side * 100 + i).collect();
                model.images.insert(id, Image { id, name, qvec: q, tvec: t, camera_id: 1, num_points2d: 20, obs_xy: vec![[0.0; 2]; 20], obs_point_id: ids });
                id += 1;
            }
        }
        model
    }

    #[test]
    fn views_of_one_rig_frame_are_one_camera_group() {
        let rig = geometry_evidence(&rig_model(true, 0.0));
        assert_eq!(rig.component_sizes, [8]);
        assert!(!assess_geometry(&rig, false).disconnected);
        // Without the view prefix, or with distinct centres, the two directions stay disconnected.
        assert_eq!(geometry_evidence(&rig_model(false, 0.0)).component_sizes, [4, 4]);
        assert_eq!(geometry_evidence(&rig_model(true, 0.5)).component_sizes, [4, 4]);
        // Solve noise in an unrigged baseline (centres 1 cm apart at 5 m) still groups.
        assert_eq!(geometry_evidence(&rig_model(true, 0.01)).component_sizes, [8]);
    }
}
