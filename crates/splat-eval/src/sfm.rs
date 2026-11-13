//! SfM health report and gates. Port of eval/sfm_health.py (same JSON fields and gates).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::colmap::{Camera, Model};

#[derive(Clone, Debug)]
pub struct SfmHealthOptions {
    /// Input image folder (recursive) to count unregistered frames and runs.
    pub images_dir: Option<PathBuf>,
    /// Number of input frames when `images_dir` is not given.
    pub num_input: Option<usize>,
    /// Video frames: reprojection gate 2.0 px instead of 1.5 px.
    pub video: bool,
    /// Held-out stride for the unregistered-holdout coverage check.
    pub holdout_every: usize,
    /// Pose-continuity gate: max rotation between frame-order neighbours, in degrees per frame
    /// step (scaled by the gap for dropped frames, capped at 90). None turns the gate off.
    pub max_pose_jump_deg: Option<f64>,
}

impl Default for SfmHealthOptions {
    fn default() -> Self {
        Self { images_dir: None, num_input: None, video: false, holdout_every: 8, max_pose_jump_deg: Some(30.0) }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PoseJump {
    pub from: String,
    pub to: String,
    pub deg: f64,
    /// Frame steps between the two (1 = adjacent; more when frames between them did not register).
    pub gap: f64,
    /// 3D points both images observe, and that count over the smaller image's points.
    pub shared_points: usize,
    pub shared_frac: f64,
    /// Counts as a flip (video only; photo-set jumps are evidence, unconfirmed).
    pub corroborated: bool,
}

/// Gated for video input only (see sfm_health). Registered cameras in frame order (file name;
/// trailing number when present): sudden rotation
/// jumps and cameras facing against their neighbours. Catches mappers that flip a block of
/// cameras while keeping reprojection error low (global mapper on truck: 18/63 flipped ~180 deg).
#[derive(Clone, Debug, Serialize)]
pub struct PoseContinuity {
    /// "flipped" or "ok" (video), "not_trajectory" (photo set with median neighbour rotation >
    /// 20 deg) or "insufficient" (under half the pairs comparable). Gated for video only.
    pub status: String,
    pub threshold_deg: f64,
    /// Neighbour pairs in name order, and how many were compared (same name sequence).
    pub n_pairs: usize,
    pub n_compared: usize,
    pub median_step_deg: f64,
    pub max_jump_deg: f64,
    pub jumps: Vec<PoseJump>,
    /// Frames outside the largest jump-free stretch, plus frames facing > 90 deg away from the
    /// average forward of their neighbours. Position-only candidates below are advisory.
    pub n_flipped: usize,
    pub flipped_frames: Vec<String>,
    /// Video: runs of COLLAPSE_RUN or more consecutive cameras whose centres sit on one point
    /// (see [`collapsed_runs`]).
    pub collapsed_frames: Vec<String>,
    /// Video: a section of BLOWN_OUT_MIN or more cameras far outside the rest (see [`blown_out`]).
    pub blown_out_frames: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CameraSummary {
    pub model: String,
    pub size: [u64; 2],
    pub f: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct UnregisteredRun {
    pub first: String,
    pub length: usize,
}

/// Same JSON as `eval/sfm_health.py --json`.
#[derive(Clone, Debug, Serialize)]
pub struct SfmHealth {
    pub sparse_dir: String,
    pub cameras: BTreeMap<String, CameraSummary>,
    pub n_input: usize,
    pub n_registered: usize,
    pub registered_frac: f64,
    pub n_points: usize,
    pub mean_reproj_err_px: Option<f64>,
    pub median_reproj_err_px: Option<f64>,
    pub mean_track_len: Option<f64>,
    pub median_track_len: Option<f64>,
    /// Points seen in >= 3 images. Gated instead of the median track length, which dense two-view
    /// initialisation drags to 2 while adding geometry (truck: more 3+-view points, median 2).
    pub n_points_multiview: usize,
    pub points_per_image_mean: f64,
    pub points_per_image_min: usize,
    pub images_under_100_points: usize,
    pub camera_extent: f64,
    pub points_median_dist_from_cameras: f64,
    pub unregistered_runs: Vec<UnregisteredRun>,
    pub unregistered_holdouts: Vec<String>,
    /// "points3D" (stored), "recomputed" (all stored errors were 0) or "not recorded".
    pub reproj_source: String,
    /// Observations whose reprojection error is non-finite or above `REPROJ_OVERFLOW_PX`: left out of
    /// the mean and median, and any one fails the reprojection gate.
    pub reproj_overflowed: usize,
    /// None when the gate is off or fewer than 3 cameras registered.
    pub pose_continuity: Option<PoseContinuity>,
    pub geometry: crate::geometry::GeometryHealth,
    pub gates: BTreeMap<String, bool>,
    pub pass: bool,
    /// Which gate set produced `pass`, so reports scored under older gates stay traceable.
    pub gate_policy: String,
}

/// v1 gated median_track_len>=3; v2 gates points_multiview>=1000 instead; v3 adds pose_continuity
/// for video input (photo sets: reported only); v4 adds provenance-independent depth/graph
/// geometry and depth-corroborated video translation impulses. Stillness alone is advisory.
pub const GATE_POLICY: &str = "sfm-gates-v4";

/// A recomputed error above this is an overflow, not a measurement: a point at the camera plane makes
/// x/z blow up (4e3 to 2.7e61 px means over a set of room runs, medians 0.2 to 1.1 px). Far above the
/// 2 px gate times any model's observation count (at most 0.73M), so a model passing on the mean
/// never holds one and a model with one already failed on the mean.
pub const REPROJ_OVERFLOW_PX: f64 = 1e7;

/// File extensions counted as input frames (matched case-insensitively).
pub const IMG_EXTS: [&str; 8] = ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "exr"];

/// Health of a model read from `sparse_dir` (the dir is only recorded in the report).
pub fn sfm_health(model: &Model, sparse_dir: &Path, opts: &SfmHealthOptions) -> Result<SfmHealth> {
    let errs: Vec<f64> = model.points.iter().map(|p| p.error).collect();
    let (errs, reproj_source) = if !errs.is_empty() && errs.iter().all(|&e| e == 0.0) {
        match reprojection_errors(model) {
            Some(e) => (e, "recomputed"),
            None => (Vec::new(), "not recorded"),
        }
    } else {
        (errs, "points3D")
    };
    let n_errs = errs.len();
    let errs: Vec<f64> = errs.into_iter().filter(|e| e.is_finite() && *e <= REPROJ_OVERFLOW_PX).collect();
    let reproj_overflowed = n_errs - errs.len();

    let input_names = opts.images_dir.as_deref().map(input_images).transpose()?;
    let n_reg = model.images.len();
    let n_input = match &input_names {
        Some(names) => names.len(),
        None => opts.num_input.filter(|&n| n > 0).unwrap_or(n_reg),
    };

    let centers: Vec<[f64; 3]> = model.images.values().map(Model::center).collect();
    let med_center: [f64; 3] = std::array::from_fn(|k| median(&centers.iter().map(|c| c[k]).collect::<Vec<_>>()));
    let camera_extent = centers.iter().map(|c| dist(c, &med_center)).fold(0.0, f64::max);
    let points_median_dist_from_cameras = if model.points.is_empty() {
        0.0
    } else {
        median(&model.points.iter().map(|p| dist(&p.xyz, &med_center)).collect::<Vec<_>>())
    };
    let obs: Vec<usize> = model.images.values().map(|im| im.obs_point_id.len()).collect();
    let tracks: Vec<f64> = model.points.iter().map(|p| p.track_len as f64).collect();

    // Unregistered runs (video): consecutive input frames missing from the model = a tracking break.
    let (mut unregistered_runs, mut unregistered_holdouts) = (Vec::new(), Vec::new());
    if let Some(names) = &input_names {
        let registered: HashSet<&str> = model.images.values().map(|im| im.name.as_str()).collect();
        let missing: Vec<usize> = (0..names.len()).filter(|&i| !registered.contains(names[i].as_str())).collect();
        for (k, &i) in missing.iter().enumerate() {
            if k == 0 || i != missing[k - 1] + 1 {
                unregistered_runs.push(UnregisteredRun { first: names[i].clone(), length: 0 });
            }
            unregistered_runs.last_mut().expect("run started above").length += 1;
        }
        let every = opts.holdout_every.max(1);
        unregistered_holdouts = missing.iter().filter(|&&i| i % every == 0).map(|&i| names[i].clone()).collect();
    }

    let registered_frac = n_reg as f64 / n_input.max(1) as f64;
    let mean_reproj_err_px = mean(&errs);
    let median_track_len = (!tracks.is_empty()).then(|| median(&tracks));
    let n_points_multiview = model.points.iter().filter(|p| p.track_len >= 3).count();
    let points_per_image_mean = if n_reg > 0 { obs.iter().sum::<usize>() as f64 / n_reg as f64 } else { 0.0 };
    let images_under_100_points = obs.iter().filter(|&&n| n < 100).count();

    let mut gates = BTreeMap::from([
        ("registered_frac>=0.95".to_string(), registered_frac >= 0.95),
        ("points_multiview>=1000".into(), n_points_multiview >= 1000),
        ("points_per_image_mean>=300".into(), points_per_image_mean >= 300.0),
        ("images_under_100_points<=2%".into(), images_under_100_points as f64 <= 0.02 * n_reg.max(1) as f64),
        ("no_unregistered_run>5".into(), unregistered_runs.iter().all(|r| r.length <= 5)),
        ("points_not_degenerate".into(), model.points.len() >= 1000),
    ]);
    if mean_reproj_err_px.is_some() || reproj_overflowed > 0 {
        // Python formats the float threshold: "2.0" / "1.5".
        let (max, label) = if opts.video { (2.0, "2.0") } else { (1.5, "1.5") };
        let ok = reproj_overflowed == 0 && mean_reproj_err_px.is_some_and(|m| m <= max);
        gates.insert(format!("mean_reproj_err<={label}px"), ok);
    }
    let pose_continuity = opts.max_pose_jump_deg.and_then(|t| pose_continuity(model, t, opts.video));
    // Gated for video only: handheld photo sets legitimately turn 90-180 deg between shots (DB
    // drjohnson's official model has 32 such pairs) and shared points do not separate those from
    // flips, so there the jumps are reported, not gated.
    if let Some(pc) = pose_continuity.as_ref().filter(|pc| opts.video && pc.status != "insufficient") {
        gates.insert("pose_continuity".into(), pc.n_flipped == 0);
    }
    let geometry = crate::geometry::geometry_health(model, opts.video);
    gates.insert("geometry_coherent".into(), geometry.pass);
    let pass = gates.values().all(|&ok| ok);

    Ok(SfmHealth {
        sparse_dir: sparse_dir.display().to_string(),
        cameras: model.cameras.iter().map(|(id, c)| (id.to_string(), summary(c))).collect(),
        n_input,
        n_registered: n_reg,
        registered_frac,
        n_points: model.points.len(),
        mean_reproj_err_px,
        median_reproj_err_px: (!errs.is_empty()).then(|| median(&errs)),
        mean_track_len: mean(&tracks),
        median_track_len,
        n_points_multiview,
        points_per_image_mean,
        points_per_image_min: obs.iter().copied().min().unwrap_or(0),
        images_under_100_points,
        camera_extent,
        points_median_dist_from_cameras,
        unregistered_runs,
        unregistered_holdouts,
        reproj_source: reproj_source.into(),
        reproj_overflowed,
        pose_continuity,
        geometry,
        gates,
        pass,
        gate_policy: GATE_POLICY.into(),
    })
}

impl SfmHealth {
    /// Reprojection key for ranking models, lower is better: the mean in micro-pixels, worst
    /// (u64::MAX) with an overflowed observation or no recorded error, so neither can win on it.
    pub fn reproj_rank(&self) -> u64 {
        match self.mean_reproj_err_px {
            Some(e) if self.reproj_overflowed == 0 && e.is_finite() => (e * 1e6) as u64,
            _ => u64::MAX,
        }
    }
}

/// (sequence, number): the name without its trailing digits, and those digits as a number
/// ("cam1/frame_000123.png" -> ("cam1/frame_", 123)).
pub(crate) fn frame_number(name: &str) -> Option<(String, i64)> {
    let p = Path::new(name);
    let stem = p.file_stem()?.to_str()?;
    let prefix = stem.trim_end_matches(|c: char| c.is_ascii_digit());
    let n = stem[prefix.len()..].parse().ok()?;
    Some((p.with_file_name(prefix).to_string_lossy().into_owned(), n))
}

fn rel_angle_deg(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    // trace(A B^T)
    let tr: f64 = (0..3).map(|i| (0..3).map(|j| a[i][j] * b[i][j]).sum::<f64>()).sum();
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

const FORWARD_WINDOW: usize = 3;

/// Any jump above this is flagged; also the per-pair threshold for photo sets (non-video).
const MAX_ALLOWED_JUMP_DEG: f64 = 90.0;
/// Photo collections, not trajectories: median neighbour rotation above this (Deep Blending
/// drjohnson/playroom ~40 deg; video and walk-arounds <= ~15 deg) means no flip test.
const TRAJECTORY_MEDIAN_DEG: f64 = 20.0;
/// Video: a jump between neighbours more than this many median frame gaps apart (unregistered
/// frames in between) is listed but not counted as a flip.
const MAX_EVIDENCE_GAP_FACTOR: f64 = 4.0;
/// Below this fraction of comparable neighbour pairs the check is "insufficient" (not gated).
const MIN_COMPARED_FRAC: f64 = 0.5;

/// Fewest consecutive registered cameras that make a collapsed section.
const COLLAPSE_RUN: usize = 10;
/// A section whose camera centres all lie within this fraction of the model's camera extent was
/// attached without parallax. HLG Oakland's unmasked solve (weak matches across a passing train)
/// placed 142/147 frames and passed every gate, but its 64 pre-train cameras sat within 1.4e-5 of one
/// point (extent ~2): shared held-out PSNR 27.4 -> 15.1 dB against the split solve. Over 74 bench
/// and dev models the tightest windows were collapses at 3e-7..8e-6, turns in place at 3e-4..9e-4
/// (an ARKit room among them) and normal walks from 2e-3 up.
const COLLAPSE_SPREAD_FRAC: f64 = 1e-4;

/// Indices of cameras (name order) inside any window of COLLAPSE_RUN consecutive centres whose RMS
/// distance from the window mean is under COLLAPSE_SPREAD_FRAC of the model's camera extent (90th
/// percentile distance from the centroid of all centres).
pub fn collapsed_runs(centres: &[[f64; 3]]) -> Vec<usize> {
    let n = centres.len();
    if n < COLLAPSE_RUN {
        return vec![];
    }
    let mean = |c: &[[f64; 3]]| -> [f64; 3] {
        let k = c.len() as f64;
        std::array::from_fn(|i| c.iter().map(|p| p[i]).sum::<f64>() / k)
    };
    let dist = |a: &[f64; 3], b: &[f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let centroid = mean(centres);
    let mut d: Vec<f64> = centres.iter().map(|c| dist(c, &centroid)).collect();
    d.sort_by(f64::total_cmp);
    let extent = d[(n - 1) * 9 / 10];
    if extent <= 0.0 {
        return vec![];
    }
    let mut out = vec![false; n];
    for start in 0..=n - COLLAPSE_RUN {
        let w = &centres[start..start + COLLAPSE_RUN];
        let m = mean(w);
        let rms = (w.iter().map(|c| dist(c, &m).powi(2)).sum::<f64>() / COLLAPSE_RUN as f64).sqrt();
        if rms < COLLAPSE_SPREAD_FRAC * extent {
            out[start..start + COLLAPSE_RUN].iter_mut().for_each(|f| *f = true);
        }
    }
    (0..n).filter(|&i| out[i]).collect()
}

/// A camera this many median distances from the median camera centre sits outside the capture.
const BLOWN_OUT_FACTOR: f64 = 20.0;
/// Fewest such cameras that make a blown-out section (a lone far camera is left to the other checks).
const BLOWN_OUT_MIN: usize = 10;

/// Indices of cameras whose centre lies over BLOWN_OUT_FACTOR median distances from the per-axis
/// median centre, when there are at least BLOWN_OUT_MIN of them: a section joined at the wrong scale
/// or place, the mirror of [`collapsed_runs`]. One room clip's 5 fps solve placed 35 opening cameras
/// ~13,800 units off (6,444 median distances; its far share read 0% and the splat budget fell
/// 5.0M -> 2.0M). Over the bench solves the passing rooms peak at 5 such cameras (44.9x); besides
/// that clip only two older solves (20 and 18 cameras) have a section.
pub fn blown_out(centres: &[[f64; 3]]) -> Vec<usize> {
    let n = centres.len();
    if n < BLOWN_OUT_MIN {
        return vec![];
    }
    let median = |mut v: Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let centre: [f64; 3] = std::array::from_fn(|i| median(centres.iter().map(|c| c[i]).collect()));
    let dist: Vec<f64> = centres.iter()
        .map(|c| ((c[0] - centre[0]).powi(2) + (c[1] - centre[1]).powi(2) + (c[2] - centre[2]).powi(2)).sqrt())
        .collect();
    let typical = median(dist.clone());
    if typical <= 0.0 {
        return vec![];
    }
    let far: Vec<usize> = (0..n).filter(|&i| dist[i] > BLOWN_OUT_FACTOR * typical).collect();
    if far.len() < BLOWN_OUT_MIN { vec![] } else { far }
}

/// `threshold_deg` is per source frame for video (frame-number gaps scale it, so every-4th-frame
/// extraction or dropped frames are fine); photo sets only flag jumps above 90 degrees.
pub fn pose_continuity(model: &Model, threshold_deg: f64, video: bool) -> Option<PoseContinuity> {
    let mut ims: Vec<_> = model.images.values().collect();
    if ims.len() < 3 {
        return None;
    }
    ims.sort_by(|a, b| a.name.cmp(&b.name));
    let seq: Vec<Option<(String, i64)>> = ims.iter().map(|im| frame_number(&im.name)).collect();
    let numbered = seq.iter().all(Option::is_some);
    // Source-frame gap between neighbours i-1 and i; None when they are not comparable.
    let gap = |i: usize| -> Option<i64> {
        if !numbered {
            return Some(1);
        }
        let (Some((pa, a)), Some((pb, b))) = (&seq[i - 1], &seq[i]) else { return None };
        // big gaps (sparse registration, sharpness-picked frames) fall back to the 90 deg limit
        (pa == pb && b > a).then_some(b - a)
    };
    let rots: Vec<[[f64; 3]; 3]> = ims.iter().map(|im| Model::rotation(im.qvec)).collect();
    let pts: Vec<HashSet<u64>> = ims.iter().map(|im| im.obs_point_id.iter().copied().collect()).collect();

    // A jump across many unregistered frames is no evidence of a flip: the camera may really have
    // turned in between. Pegasus's 24-frame quick SfM (median gap 11 source frames) lost frames
    // 96-168 and joined 35 -> 178 across a 145 deg walk around the statue; the full SfM, with frames
    // in between, agrees with that turn to 0.3 deg, yet the gap-105 jump failed every first view.
    // Known real flips sit at 0.7-1.2x the median gap (groei, moskvitch, house 772 -> 784).
    let median_gap = {
        let gaps: Vec<f64> = (1..ims.len()).filter_map(|i| gap(i)).map(|g| g as f64).collect();
        if gaps.is_empty() { 1.0 } else { median(&gaps) }
    };
    let mut jumps = Vec::new();
    let mut max_jump_deg: f64 = 0.0;
    let mut cuts = Vec::new();
    // Segment of each frame between uncorroborated jumps: a real turn across unregistered frames
    // ends the forward-direction neighbourhood below.
    let mut segment = vec![0usize; ims.len()];
    let mut evidence = false;
    let mut n_compared = 0;
    let mut steps = Vec::new();
    for i in 1..ims.len() {
        let Some(gap) = gap(i) else { continue }; // a different sequence
        n_compared += 1;
        let deg = rel_angle_deg(&rots[i - 1], &rots[i]);
        steps.push(deg);
        max_jump_deg = max_jump_deg.max(deg);
        let allowed = if video { (threshold_deg * gap as f64).min(MAX_ALLOWED_JUMP_DEG) } else { MAX_ALLOWED_JUMP_DEG };
        if deg > allowed {
            let shared_points = pts[i - 1].intersection(&pts[i]).count();
            let shared_frac = shared_points as f64 / pts[i - 1].len().min(pts[i].len()).max(1) as f64;
            // Photo sets: no model-only signal separates flips from real wide turns on DEV. Flipped
            // cameras share ~0 points with their neighbour (truck global 0/5 jumps, auditorium
            // 3-19), while real turns share many (playroom up to 536, drjohnson 286), so they are
            // listed with this evidence but never counted.
            let corroborated = video && gap as f64 <= MAX_EVIDENCE_GAP_FACTOR * median_gap;
            jumps.push(PoseJump { from: ims[i - 1].name.clone(), to: ims[i].name.clone(), deg, gap: gap as f64, shared_points, shared_frac, corroborated });
            // Every video jump still toggles the two-state flip below (house: frame 784 flips at gap
            // 12 and returns at gap 77), but only a corroborated one makes the trajectory flipped.
            if video {
                cuts.push(i);
            }
            evidence |= corroborated;
            if video && !corroborated {
                segment[i..].iter_mut().for_each(|s| *s += 1);
            }
        }
    }

    // Mapper flips are two-state (a mirrored/rotated copy of the true pose, e.g. the far side of a
    // symmetric object), so each jump toggles between the two; the smaller side is the flipped one.
    let mut flipped = vec![false; ims.len()];
    if evidence {
        let mut side = vec![false; ims.len()];
        let mut cur = false;
        let mut next_cut = cuts.iter().peekable();
        for (i, s) in side.iter_mut().enumerate() {
            if next_cut.next_if_eq(&&i).is_some() {
                cur = !cur;
            }
            *s = cur;
        }
        let n_toggled = side.iter().filter(|&&s| s).count();
        // ties: the side holding the first frame counts as correct
        let bad = n_toggled * 2 > ims.len();
        for (f, &s) in flipped.iter_mut().zip(&side) {
            *f = s != bad;
        }
    }
    // Viewing direction (camera +z in world = third row of R) against the average of the
    // neighbours not already flagged, on the same side of any uncorroborated jump: pegasus's quick
    // SfM joined 73 -> 178 across a real 170 deg walk around the statue, and the far side's frames
    // in the window flagged both frames at the turn as flipped (no first view).
    let fwd: Vec<[f64; 3]> = rots.iter().map(|r| r[2]).collect();
    let by_jumps = flipped.clone();
    // photo sets are not trajectories: neighbours by name need not face the same way
    for i in (0..ims.len()).filter(|&i| video && !by_jumps[i]) {
        let lo = i.saturating_sub(FORWARD_WINDOW);
        let hi = (i + FORWARD_WINDOW + 1).min(ims.len());
        let mut avg = [0.0; 3];
        for j in (lo..hi).filter(|&j| j != i && !by_jumps[j] && segment[j] == segment[i]) {
            for k in 0..3 {
                avg[k] += fwd[j][k];
            }
        }
        let n = (avg[0] * avg[0] + avg[1] * avg[1] + avg[2] * avg[2]).sqrt();
        if n > 1e-9 && (fwd[i][0] * avg[0] + fwd[i][1] * avg[1] + fwd[i][2] * avg[2]) / n < 0.0 {
            flipped[i] = true;
        }
    }
    // Position-only collapse/outlier evidence is advisory: a legitimate hold or turn in place
    // can produce exactly these centres. Independent depth/graph evidence gates geometry.
    let mut collapsed_frames = Vec::new();
    // So are a blown-out section's.
    let mut blown_out_frames = Vec::new();
    if video {
        let centres: Vec<[f64; 3]> = ims.iter().zip(&rots).map(|(im, r)| {
            std::array::from_fn(|i| -(r[0][i] * im.tvec[0] + r[1][i] * im.tvec[1] + r[2][i] * im.tvec[2]))
        }).collect();
        for i in collapsed_runs(&centres) {
            collapsed_frames.push(ims[i].name.clone());
        }
        for i in blown_out(&centres) {
            blown_out_frames.push(ims[i].name.clone());
        }
    }
    let flipped_frames: Vec<String> = ims.iter().zip(&flipped).filter(|(_, f)| **f).map(|(im, _)| im.name.clone()).collect();
    let n_pairs = ims.len() - 1;
    let median_step_deg = if steps.is_empty() { 0.0 } else { median(&steps) };
    let not_trajectory = !video && median_step_deg > TRAJECTORY_MEDIAN_DEG;
    let status = if (n_compared as f64) < MIN_COMPARED_FRAC * n_pairs as f64 {
        "insufficient"
    } else if !flipped_frames.is_empty() {
        "flipped"
    } else if not_trajectory {
        "not_trajectory"
    } else {
        "ok"
    };
    Some(PoseContinuity {
        status: status.into(),
        threshold_deg,
        n_pairs,
        n_compared,
        median_step_deg,
        max_jump_deg,
        jumps,
        n_flipped: flipped_frames.len(),
        flipped_frames,
        collapsed_frames,
        blown_out_frames,
    })
}

/// Per-observation reprojection errors recomputed from poses and intrinsics; None if any camera
/// model is unsupported (supported: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV).
/// Also None when no image has an observation; observations of unknown points are skipped.
pub fn reprojection_errors(model: &Model) -> Option<Vec<f64>> {
    let lut: HashMap<u64, [f64; 3]> = model.points.iter().map(|p| (p.id, p.xyz)).collect();
    let mut out = Vec::new();
    let mut any = false;
    for im in model.images.values().filter(|im| !im.obs_point_id.is_empty()) {
        any = true;
        let cam = model.cameras.get(&im.camera_id)?;
        let r = Model::rotation(im.qvec);
        for (pid, xy) in im.obs_point_id.iter().zip(&im.obs_xy) {
            let Some(x) = lut.get(pid) else { continue };
            let pc: [f64; 3] = std::array::from_fn(|i| r[i][0] * x[0] + r[i][1] * x[1] + r[i][2] * x[2] + im.tvec[i]);
            let [u, v] = project(cam, pc)?;
            out.push((u - xy[0]).hypot(v - xy[1]));
        }
    }
    any.then_some(out)
}

/// Project a camera-frame point to pixels; None for unsupported models.
fn project(cam: &Camera, pc: [f64; 3]) -> Option<[f64; 2]> {
    let p = &cam.params;
    let at = |i: usize| p.get(i).copied().unwrap_or(0.0);
    let (x, y) = (pc[0] / pc[2], pc[1] / pc[2]);
    let r2 = x * x + y * y;
    let (fx, fy, cx, cy, xd, yd) = match cam.model.as_str() {
        "SIMPLE_PINHOLE" | "SIMPLE_RADIAL" | "RADIAL" => {
            let rad = 1.0 + at(3) * r2 + at(4) * r2 * r2;
            (at(0), at(0), at(1), at(2), x * rad, y * rad)
        }
        "PINHOLE" | "OPENCV" => {
            let [k1, k2, p1, p2] = if cam.model == "OPENCV" { [at(4), at(5), at(6), at(7)] } else { [0.0; 4] };
            let rad = 1.0 + k1 * r2 + k2 * r2 * r2;
            let xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            (at(0), at(1), at(2), at(3), xd, yd)
        }
        _ => return None,
    };
    Some([fx * xd + cx, fy * yd + cy])
}

/// `f` is the physical [fx, fy]: single-focal models (SIMPLE_*, RADIAL, RADIAL_FISHEYE) store one
/// focal, so params[1] is a principal point there, not fy.
fn summary(c: &Camera) -> CameraSummary {
    let single = c.model.starts_with("SIMPLE_") || matches!(c.model.as_str(), "RADIAL" | "RADIAL_FISHEYE");
    let f = match (single, c.params.first()) {
        (true, Some(&f)) => vec![f, f],
        _ => c.params.iter().take(2).copied().collect(),
    };
    CameraSummary { model: c.model.clone(), size: [c.width, c.height], f }
}

/// Image files under `dir` (recursive), as sorted relative paths.
fn input_images(dir: &Path) -> Result<Vec<String>> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
        for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                walk(root, &path, out)?;
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| IMG_EXTS.contains(&e.to_ascii_lowercase().as_str()))
            {
                out.push(path.strip_prefix(root)?.to_string_lossy().into_owned());
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// numpy median: the mean of the two middle values for even lengths; NaN when empty.
fn median(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut v = v.to_vec();
    let (lo, &mut hi, _) = v.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 1 {
        hi
    } else {
        let below = lo.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (below + hi) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap::{Image, Point3D};

    fn camera(id: u32, model: &str, params: &[f64]) -> Camera {
        Camera { id, model: model.into(), width: 100, height: 80, params: params.to_vec() }
    }

    fn image(id: u32, name: &str, qvec: [f64; 4], tvec: [f64; 3], obs: &[([f64; 2], u64)]) -> Image {
        Image {
            id,
            name: name.into(),
            qvec,
            tvec,
            camera_id: id,
            num_points2d: obs.len() + 1,
            obs_xy: obs.iter().map(|o| o.0).collect(),
            obs_point_id: obs.iter().map(|o| o.1).collect(),
        }
    }

    fn point(id: u64, xyz: [f64; 3]) -> Point3D {
        Point3D { id, xyz, rgb: [0; 3], error: 0.0, track_len: 2 }
    }

    /// Image 1: PINHOLE at the origin. Image 2: SIMPLE_RADIAL, rotated 90 degrees about z, t = (0, 0, 1).
    fn two_camera_model() -> Model {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        Model {
            cameras: BTreeMap::from([
                (1, camera(1, "PINHOLE", &[100.0, 100.0, 50.0, 40.0])),
                (2, camera(2, "SIMPLE_RADIAL", &[200.0, 10.0, 20.0, 0.1])),
            ]),
            images: BTreeMap::from([
                // (1,2,4) -> (75, 90), observed 3-4-5 off; (0,0,5) -> (50, 40) exactly.
                (1, image(1, "f1.png", [1.0, 0.0, 0.0, 0.0], [0.0; 3], &[([78.0, 94.0], 7), ([50.0, 40.0], 8)])),
                // (1,2,4) -> camera (-2, 1, 5) -> r2 0.2, radial 1.02 -> (-71.6, 60.8); observed 2 px off.
                // Point 99 does not exist and is skipped.
                (2, image(2, "f2.png", [h, 0.0, 0.0, h], [0.0, 0.0, 1.0], &[([-71.6, 62.8], 7), ([0.0, 0.0], 99)])),
            ]),
            points: vec![point(7, [1.0, 2.0, 4.0]), point(8, [0.0, 0.0, 5.0])],
        }
    }

    #[test]
    fn reprojection_two_cameras() {
        let mut e = reprojection_errors(&two_camera_model()).unwrap();
        e.sort_by(f64::total_cmp);
        let want = [0.0, 2.0, 5.0];
        assert_eq!(e.len(), 3);
        for (a, b) in e.iter().zip(want) {
            assert!((a - b).abs() < 1e-9, "{e:?}");
        }
    }

    #[test]
    fn reprojection_unsupported_model() {
        let mut m = two_camera_model();
        m.cameras.get_mut(&2).unwrap().model = "OPENCV_FISHEYE".into();
        assert!(reprojection_errors(&m).is_none());
    }

    #[test]
    fn numpy_median() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn overflowed_reprojection_never_wins() {
        let opts = SfmHealthOptions::default();
        // Errors 6, 2 and 5 px: mean 13/3, all finite.
        let mut clean = two_camera_model();
        clean.images.get_mut(&1).unwrap().obs_xy[1] = [50.0, 46.0];
        // Errors 0, 2 and 5 px plus two points at camera 1's plane: z = 1e-60 overflows to ~1e62 px and
        // z = 0 to NaN (which old code cast to 0, the best rank).
        let mut broken = two_camera_model();
        broken.points.extend([point(9, [1.0, 0.0, 1e-60]), point(10, [1.0, 0.0, 0.0])]);
        let im = broken.images.get_mut(&1).unwrap();
        im.obs_xy.extend([[0.0, 0.0], [0.0, 0.0]]);
        im.obs_point_id.extend([9, 10]);

        let clean = sfm_health(&clean, Path::new("sparse/0"), &opts).unwrap();
        let broken = sfm_health(&broken, Path::new("sparse/0"), &opts).unwrap();
        assert_eq!((clean.reproj_overflowed, broken.reproj_overflowed), (0, 2));
        assert!((broken.mean_reproj_err_px.unwrap() - 7.0 / 3.0).abs() < 1e-9);
        assert!((clean.mean_reproj_err_px.unwrap() - 13.0 / 3.0).abs() < 1e-9);
        assert!(!broken.gates["mean_reproj_err<=1.5px"]);
        assert!(broken.reproj_rank() > clean.reproj_rank());
    }

    #[test]
    fn health_runs_holdouts_and_gates() {
        let dir = std::env::temp_dir().join(format!("splat-eval-sfm-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        for i in 0..10 {
            std::fs::write(dir.join(format!("f{i}.png")), b"").unwrap();
        }
        std::fs::write(dir.join("sub/x.JPG"), b"").unwrap();
        std::fs::write(dir.join("notes.txt"), b"").unwrap();

        let opts = SfmHealthOptions { images_dir: Some(dir.clone()), ..Default::default() };
        let r = sfm_health(&two_camera_model(), Path::new("sparse/0"), &opts).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!((r.n_input, r.n_registered, r.n_points), (11, 2, 2));
        assert_eq!(r.reproj_source, "recomputed");
        assert!((r.mean_reproj_err_px.unwrap() - 7.0 / 3.0).abs() < 1e-9);
        assert!((r.median_reproj_err_px.unwrap() - 2.0).abs() < 1e-9);
        let runs: Vec<_> = r.unregistered_runs.iter().map(|u| (u.first.as_str(), u.length)).collect();
        assert_eq!(runs, [("f0.png", 1), ("f3.png", 8)]);
        assert_eq!(r.unregistered_holdouts, ["f0.png", "f8.png"]);
        assert_eq!((r.points_per_image_mean, r.points_per_image_min, r.images_under_100_points), (2.0, 2, 2));
        assert!(!r.gates["no_unregistered_run>5"]);
        assert!(!r.gates["mean_reproj_err<=1.5px"]);
        assert!(!r.pass);

        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["cameras"]["2"]["f"], serde_json::json!([200.0, 200.0]));
        assert_eq!(json["cameras"]["1"]["size"], serde_json::json!([100, 80]));
    }

    /// Frames every 10 source frames turning 5 deg each; `turn` adds a step rotation from `from` on.
    fn yaw_track(frames: &[i64], turn: impl Fn(i64) -> f64) -> Model {
        let images = frames.iter().enumerate().map(|(k, &f)| {
            let half = (5.0 * k as f64 + turn(f)).to_radians() / 2.0;
            let id = k as u32 + 1;
            (id, image(id, &format!("frame_{f:06}.jpg"), [half.cos(), 0.0, half.sin(), 0.0], [0.0; 3], &[]))
        });
        Model { cameras: BTreeMap::from([(1, camera(1, "PINHOLE", &[100.0, 100.0, 50.0, 40.0]))]), images: images.collect(), points: vec![] }
    }

    #[test]
    fn collapsed_sections_are_flagged_and_moving_trajectories_are_not() {
        // Camera centres (name order) of HLG Oakland's unmasked solve, whose 64 pre-train cameras
        // collapsed onto one point, and of the masked solve of the same frames (64 moving cameras).
        let fixture: serde_json::Value = serde_json::from_str(include_str!("../testdata/hlg_camera_centres.json")).unwrap();
        let centres = |key: &str| -> Vec<[f64; 3]> {
            fixture[key].as_array().unwrap().iter().map(|row| {
                let c = row[1].as_array().unwrap();
                std::array::from_fn(|i| c[i].as_f64().unwrap())
            }).collect()
        };
        let collapsed = collapsed_runs(&centres("hlg_unmasked_collapsed"));
        assert_eq!(collapsed, (0..64).collect::<Vec<_>>());
        assert!(collapsed_runs(&centres("hlg_masked")).is_empty());
        // A steady walk with a pause of a few frames is not a collapse.
        let mut walk: Vec<[f64; 3]> = (0..40).map(|k| [0.1 * k as f64, 0.0, 0.0]).collect();
        walk[10..16].fill([1.0, 0.0, 0.0]);
        assert!(collapsed_runs(&walk).is_empty());
    }

    #[test]
    fn blown_out_sections_are_flagged_and_long_walks_are_not() {
        // A room clip's 5 fps solve: a 137-camera room plus 35 opening cameras ~13,800 units off.
        let mut cams: Vec<[f64; 3]> = (0..137).map(|k| [3.0 * (k as f64 * 0.1).cos(), 3.0 * (k as f64 * 0.1).sin(), 0.0]).collect();
        cams.extend((0..35).map(|k| [13_800.0 + k as f64, 0.0, 0.0]));
        assert_eq!(blown_out(&cams), (137..172).collect::<Vec<_>>());
        // Five far cameras (a passing room's 5 at up to 45x) are not a section.
        cams.truncate(142);
        assert!(blown_out(&cams).is_empty());
        // A straight walk: the far end is a few median distances out.
        let walk: Vec<[f64; 3]> = (0..200).map(|k| [0.1 * k as f64, 0.0, 0.0]).collect();
        assert!(blown_out(&walk).is_empty());
    }

    #[test]
    fn continuity_ignores_turns_across_unregistered_stretches_but_flags_flips() {
        // Pegasus: a 150 deg turn while frames 50-150 failed to register is a real turn, not a flip.
        let gap: Vec<i64> = [0, 10, 20, 30, 40].into_iter().chain([150, 160, 170, 180, 190, 200]).collect();
        let pc = pose_continuity(&yaw_track(&gap, |f| if f >= 150 { 150.0 } else { 0.0 }), 30.0, true).unwrap();
        assert_eq!(pc.status, "ok", "{pc:?}");
        assert!(pc.jumps.iter().all(|j| !j.corroborated) && pc.jumps.len() == 1, "{pc:?}");
        // Pegasus: a 170 deg turn across the gap; the frames at the turn face away from
        // the far side's, which must not count as their neighbours.
        let pc = pose_continuity(&yaw_track(&gap, |f| if f >= 150 { 170.0 } else { 0.0 }), 30.0, true).unwrap();
        assert_eq!(pc.status, "ok", "{pc:?}");
        // The same 150 deg jump between neighbours at the usual spacing is a flip.
        let dense: Vec<i64> = (0..11).map(|k| 10 * k).collect();
        let pc = pose_continuity(&yaw_track(&dense, |f| if f == 50 { 150.0 } else { 0.0 }), 30.0, true).unwrap();
        assert_eq!(pc.status, "flipped", "{pc:?}");
        assert_eq!(pc.flipped_frames, ["frame_000050.jpg"]);
        // House: one frame flips at the usual spacing and the return jump crosses a registration gap;
        // only that frame is flipped, not everything after it.
        let house: Vec<i64> = (0..8).map(|k| 10 * k).chain((0..6).map(|k| 200 + 10 * k)).collect();
        let pc = pose_continuity(&yaw_track(&house, |f| if f == 70 { 150.0 } else { 0.0 }), 30.0, true).unwrap();
        assert_eq!(pc.status, "flipped", "{pc:?}");
        assert_eq!(pc.flipped_frames, ["frame_000070.jpg"], "{pc:?}");
    }
}
