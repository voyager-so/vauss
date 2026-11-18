//! Rig-aware image pairs for 360 solves (splat-core's panorama split: every 360 frame is one rig
//! of pinhole views that share a centre, and each view keeps only the features of the directions it
//! owns).
//!
//! COLMAP's sequential matcher with `expand_rig_images` matches every view of a frame against all
//! views of each neighbouring frame (14 x 14 per frame pair). With feature ownership only the views
//! whose owned directions overlap can share a match, so most of those pairs cost a full descriptor
//! match and give nothing (Temple3: 1,629 s of a 1,880 s solve). Instead:
//!
//! 1. **Probe**: match each view with the same view of the next two frames (14 pairs per frame
//!    pair). [`write_probe`] writes that list.
//! 2. **Rotations**: [`write_pairs`] fits each probed frame pair's relative rotation to the
//!    verified inliers' bearing rays (Kabsch; the baseline between neighbouring frames is small
//!    next to the scene), and chains them in capture order.
//! 3. **Pairs**: for frame pairs at the quadratic offsets 1, 2, 4, ... keep the view pairs whose
//!    owned directions overlap once the rotation is applied, widened by a margin for rotation error
//!    and parallax (which grows with the offset: the probes' residuals measure it per frame step).
//!    A frame pair without a chained rotation keeps all view pairs, and rotations that disagree
//!    around a probed triangle are not trusted.
//!
//! Loop closure stays with vocabulary-tree retrieval ([`write_loop_queries`]: every 10th image,
//! as the sequential matcher's loop detection).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat3, DQuat, DVec3};
use serde::Serialize;

/// Rig-aware pair selection (see the module docs).
#[derive(Clone, Debug, Serialize)]
pub struct RigPairs {
    /// Frame offsets 1, 2, 4, ..., 2^(levels-1); 6 are the old sequential matcher's offsets (1-32),
    /// whose long baselines matter for distant content (Ballintoy up and zenith views: 4 levels lost 0.4 dB).
    pub levels: usize,
    /// Extra angle (degrees) around each owned direction that still counts as overlap (rotation
    /// error); the measured parallax per frame step times the offset is added on top.
    pub margin_deg: f64,
    /// A view pair is matched when its overlap is at least this share of either view's directions
    /// (each direction spread over the margin ring).
    pub min_share: f64,
    /// Features per image that loop retrieval indexes and queries. Indexing runs on one thread
    /// (Temple3, 1,008 views: 0.2 s per view at 2,048 features).
    pub loop_features: usize,
}

impl Default for RigPairs {
    fn default() -> Self {
        Self { levels: 6, margin_deg: 10.0, min_share: 0.03, loop_features: 512 }
    }
}

/// The probe's frame offsets: each view against the same view 1 and 2 frames later.
const PROBE_OFFSETS: [usize; 2] = [1, 2];
/// The sequential matcher's loop detection period (every n-th image queries the vocabulary tree).
pub const LOOP_PERIOD: usize = 10;
/// A probed frame pair needs this many consistent inliers for its rotation to count.
const MIN_ROTATION_INLIERS: usize = 30;
/// Inliers farther than this from the fitted rotation are dropped before the refit.
const MAX_RESIDUAL_DEG: f64 = 5.0;
/// Sample directions on the sphere for the overlap test.
const SAMPLES: usize = 3000;

/// One view of the rig: name prefix and cam_from_rig.
#[derive(Clone, Debug)]
pub struct RigView {
    pub prefix: String,
    pub cam_from_rig: DMat3,
}

/// The rig's views (rig_configurator input, one rig) and the shared pinhole (fx, fy, cx, cy).
#[derive(Clone, Debug)]
pub struct Rig {
    pub views: Vec<RigView>,
    pub pinhole: [f64; 4],
}

impl Rig {
    pub fn read(config: &Path, camera_params: &str) -> Result<Self> {
        let rigs: serde_json::Value = serde_json::from_slice(&fs::read(config).with_context(|| format!("reading {}", config.display()))?)?;
        let cameras = rigs.as_array().and_then(|r| r.first()).and_then(|r| r["cameras"].as_array())
            .with_context(|| format!("{} has no rig cameras", config.display()))?;
        let mut views = Vec::new();
        for c in cameras {
            let prefix = c["image_prefix"].as_str().context("rig camera without image_prefix")?.to_owned();
            let cam_from_rig = if c["ref_sensor"].as_bool() == Some(true) {
                DMat3::IDENTITY
            } else {
                let q: Vec<f64> = c["cam_from_rig_rotation"].as_array().context("rig camera without cam_from_rig_rotation")?
                    .iter().filter_map(|v| v.as_f64()).collect();
                ensure!(q.len() == 4, "cam_from_rig_rotation needs 4 values");
                DMat3::from_quat(DQuat::from_xyzw(q[1], q[2], q[3], q[0]).normalize())
            };
            views.push(RigView { prefix, cam_from_rig });
        }
        let p: Vec<f64> = camera_params.split(',').map(|v| v.trim().parse::<f64>()).collect::<Result<_, _>>()
            .with_context(|| format!("camera params {camera_params:?}"))?;
        ensure!(p.len() == 4 && p[0] > 0.0 && p[1] > 0.0, "rig pairs need PINHOLE params f,f,c,c, got {camera_params:?}");
        Ok(Self { views, pinhole: [p[0], p[1], p[2], p[3]] })
    }

    /// The view and frame key of an image name (longest matching prefix).
    fn split<'a>(&self, name: &'a str) -> Option<(usize, &'a str)> {
        self.views.iter().enumerate().filter(|(_, v)| name.starts_with(v.prefix.as_str()))
            .max_by_key(|(_, v)| v.prefix.len()).map(|(i, v)| (i, &name[v.prefix.len()..]))
    }

    /// The view that owns rig direction `d`: among the views whose frustum holds it, the one whose
    /// axis is closest (splat-core panorama::owner). None outside every frustum.
    pub fn owner(&self, d: DVec3) -> Option<usize> {
        let (tx, ty) = (self.pinhole[2] / self.pinhole[0], self.pinhole[3] / self.pinhole[1]);
        let mut best: Option<(usize, f64)> = None;
        for (i, v) in self.views.iter().enumerate() {
            let c = v.cam_from_rig * d;
            if c.z > 0.0 && c.x.abs() <= c.z * tx && c.y.abs() <= c.z * ty && best.is_none_or(|(_, z)| c.z > z) {
                best = Some((i, c.z));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Rig-frame unit ray of keypoint (x, y) of `view`.
    fn ray(&self, view: usize, x: f64, y: f64) -> DVec3 {
        let [fx, fy, cx, cy] = self.pinhole;
        self.views[view].cam_from_rig.transpose() * DVec3::new((x - cx) / fx, (y - cy) / fy, 1.0).normalize()
    }
}

/// The database's images grouped as frames (sorted by frame key: the sequential matcher's order)
/// of views.
struct Frames {
    /// Frame key -> view -> (image id, name).
    frames: Vec<(String, BTreeMap<usize, (i64, String)>)>,
}

fn read_frames(db: &rusqlite::Connection, rig: &Rig) -> Result<Frames> {
    let mut by_key: BTreeMap<String, BTreeMap<usize, (i64, String)>> = BTreeMap::new();
    let rows: Vec<(i64, String)> = db.prepare("SELECT image_id, name FROM images")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
    for (id, name) in rows {
        let Some((view, key)) = rig.split(&name) else { bail!("image {name} has no rig view prefix") };
        by_key.entry(key.to_owned()).or_default().insert(view, (id, name));
    }
    Ok(Frames { frames: by_key.into_iter().collect() })
}

fn write_lines(path: &Path, lines: impl IntoIterator<Item = String>) -> Result<usize> {
    let mut out = std::io::BufWriter::new(fs::File::create(path).with_context(|| format!("writing {}", path.display()))?);
    let mut n = 0;
    for line in lines {
        writeln!(out, "{line}")?;
        n += 1;
    }
    out.flush()?;
    Ok(n)
}

/// Writes the probe pairs (each view against the same view 1 and 2 frames later) to `path`.
pub fn write_probe(database: &Path, rig: &Rig, path: &Path) -> Result<usize> {
    let db = rusqlite::Connection::open(database)?;
    let frames = read_frames(&db, rig)?.frames;
    let mut lines = Vec::new();
    for (i, (_, views)) in frames.iter().enumerate() {
        for off in PROBE_OFFSETS {
            let Some((_, other)) = frames.get(i + off) else { continue };
            for (view, (_, name)) in views {
                if let Some((_, name2)) = other.get(view) {
                    lines.push(format!("{name} {name2}"));
                }
            }
        }
    }
    write_lines(path, lines)
}

/// Writes the loop-retrieval query images to `path`: every [`LOOP_PERIOD`]-th image by name, as
/// COLMAP's sequential matcher queries its vocabulary tree.
pub fn write_loop_queries(database: &Path, path: &Path) -> Result<usize> {
    let db = rusqlite::Connection::open(database)?;
    let mut names: Vec<String> = db.prepare("SELECT name FROM images")?
        .query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
    names.sort();
    write_lines(path, names.into_iter().step_by(LOOP_PERIOD))
}

/// What [`write_pairs`] chose, written next to the pair list.
#[derive(Debug, Serialize)]
pub struct PairsSummary {
    pub frames: usize,
    pub views: usize,
    /// Probed frame pairs whose rotation was fitted and kept.
    pub rotations_fitted: usize,
    pub rotations_probed: usize,
    /// Fitted rotations dropped because they disagreed around a triangle (f, f+1, f+2).
    pub rotations_inconsistent: usize,
    /// Pieces of the rotation chain (1: every frame's rotation is known relative to the first).
    pub rotation_chains: usize,
    pub frame_pairs: usize,
    /// Frame pairs matched with every view pair (no chained rotation).
    pub frame_pairs_unknown: usize,
    pub image_pairs: usize,
    /// Every view pair of the same frame pairs (COLMAP's rig expansion at these offsets).
    pub image_pairs_all_views: usize,
    /// Median rotation residual of the fitted probes (degrees).
    pub median_residual_deg: Option<f64>,
    /// Parallax per frame step (degrees) that widens the margin with the frame offset.
    pub step_parallax_deg: f64,
}

/// Rotations that disagree by more than this around a probed triangle are not trusted.
const MAX_TRIANGLE_DEG: f64 = 2.0;
/// The margin never grows past this.
const MAX_MARGIN_DEG: f64 = 45.0;

/// Fits each probed frame pair's rotation, chains them and writes the rig-aware pair list to
/// `path` (one "name1 name2" per line) and a summary beside it.
pub fn write_pairs(database: &Path, rig: &Rig, opts: &RigPairs, path: &Path) -> Result<PairsSummary> {
    let db = rusqlite::Connection::open(database)?;
    let frames = read_frames(&db, rig)?.frames;
    let probes = probe_rays(&db, rig, &frames)?;
    let views: Vec<BTreeSet<usize>> = frames.iter().map(|(_, v)| v.keys().copied().collect()).collect();
    let (pairs, summary) = choose_pairs(rig, &views, &probes, opts);
    write_lines(path, pairs.into_iter().map(|(f, a, g, b)| format!("{} {}", frames[f].1[&a].1, frames[g].1[&b].1)))?;
    fs::write(path.with_extension("json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(summary)
}

/// Rig-frame ray pairs (first frame, second frame) of the verified inliers between the frames of
/// each probed frame pair.
fn probe_rays(db: &rusqlite::Connection, rig: &Rig, frames: &[(String, BTreeMap<usize, (i64, String)>)]) -> Result<BTreeMap<(usize, usize), Vec<(DVec3, DVec3)>>> {
    let mut where_is: HashMap<i64, (usize, usize)> = HashMap::new();
    for (f, (_, views)) in frames.iter().enumerate() {
        for (&v, (id, _)) in views {
            where_is.insert(*id, (f, v));
        }
    }
    // Inliers as (image1, keypoint1, image2, keypoint2), image1 in the earlier frame.
    let mut probed: BTreeMap<(usize, usize), Vec<(i64, u32, i64, u32)>> = BTreeMap::new();
    let rows: Vec<(i64, Vec<u8>)> = db.prepare("SELECT pair_id, data FROM two_view_geometries WHERE rows > 0")?
        .query_map([], |r| Ok((r.get(0)?, r.get::<_, Option<Vec<u8>>>(1)?.unwrap_or_default())))?
        .collect::<rusqlite::Result<_>>()?;
    for (pair, data) in rows {
        // COLMAP pair ids: smaller image id * 2^31-1 + larger; matches list the smaller id first.
        let (a, b) = (pair / 2_147_483_647, pair % 2_147_483_647);
        let (Some(&(fa, _)), Some(&(fb, _))) = (where_is.get(&a), where_is.get(&b)) else { continue };
        let swap = fb < fa;
        let (f1, f2, i1, i2) = if swap { (fb, fa, b, a) } else { (fa, fb, a, b) };
        if !PROBE_OFFSETS.contains(&(f2 - f1)) {
            continue;
        }
        let entry = probed.entry((f1, f2)).or_default();
        for m in data.chunks_exact(8) {
            let x = u32::from_le_bytes(m[..4].try_into().expect("4 bytes"));
            let y = u32::from_le_bytes(m[4..].try_into().expect("4 bytes"));
            entry.push(if swap { (i1, y, i2, x) } else { (i1, x, i2, y) });
        }
    }
    let used: BTreeSet<i64> = probed.values().flatten().flat_map(|m| [m.0, m.2]).collect();
    let mut keypoints: HashMap<i64, Vec<[f32; 2]>> = HashMap::new();
    let mut stmt = db.prepare("SELECT rows, cols, data FROM keypoints WHERE image_id = ?1")?;
    for id in used {
        let (rows, cols, data): (i64, i64, Vec<u8>) = stmt.query_row([id], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, Option<Vec<u8>>>(2)?.unwrap_or_default())))?;
        ensure!(cols >= 2 && data.len() as i64 == rows * cols * 4, "image {id}: unexpected keypoints blob");
        let floats: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))).collect();
        keypoints.insert(id, floats.chunks_exact(cols as usize).map(|k| [k[0], k[1]]).collect());
    }
    let ray = |id: i64, k: u32| -> Option<DVec3> {
        let p = keypoints.get(&id)?.get(k as usize)?;
        Some(rig.ray(where_is[&id].1, p[0] as f64, p[1] as f64))
    };
    Ok(probed.into_iter().map(|(key, m)| (key, m.iter().filter_map(|&(a, ka, b, kb)| Some((ray(a, ka)?, ray(b, kb)?))).collect())).collect())
}

/// The view pairs to match, as (frame, view, later frame, view), from the probes' ray pairs.
/// `views[f]` holds the views frame `f` has.
pub fn choose_pairs(rig: &Rig, views: &[BTreeSet<usize>], probes: &BTreeMap<(usize, usize), Vec<(DVec3, DVec3)>>, opts: &RigPairs)
    -> (Vec<(usize, usize, usize, usize)>, PairsSummary) {
    let n = views.len();
    // Relative rotations (rays in the later frame = R * rays in the earlier one).
    let mut fits: HashMap<(usize, usize), Fit> = probes.iter().filter_map(|(&key, rays)| Some((key, fit_rotation(rays)?))).collect();
    // A rotation that disagrees around a triangle (f, f+1, f+2) may have locked onto something
    // that moves with the camera: drop all three, so those frames fall back to every view pair.
    let mut inconsistent = BTreeSet::new();
    for f in 0..n.saturating_sub(2) {
        if let (Some(a), Some(b), Some(c)) = (fits.get(&(f, f + 1)), fits.get(&(f + 1, f + 2)), fits.get(&(f, f + 2))) {
            if angle_deg(b.r * a.r, c.r) > MAX_TRIANGLE_DEG {
                inconsistent.extend([(f, f + 1), (f + 1, f + 2), (f, f + 2)]);
            }
        }
    }
    let fitted = fits.len();
    fits.retain(|k, _| !inconsistent.contains(k));
    // Chain in capture order: each frame from the frame before it, else from the one before that.
    let mut absolute: Vec<(usize, DMat3)> = Vec::with_capacity(n);
    let mut chains = 0;
    for f in 0..n {
        let from = PROBE_OFFSETS.iter().find_map(|&off| {
            let g = f.checked_sub(off)?;
            let (chain, rg) = absolute[g];
            fits.get(&(g, f)).map(|fit| (chain, fit.r * rg))
        });
        absolute.push(from.unwrap_or_else(|| {
            chains += 1;
            (chains, DMat3::IDENTITY)
        }));
    }
    // Translation shows as residual parallax: its typical size per frame step (the median over
    // the neighbouring frames' 90th-percentile residuals) widens the margin at longer offsets.
    let mut step: Vec<f64> = fits.iter().filter(|(k, _)| k.1 - k.0 == 1).map(|(_, f)| f.p90_deg).collect();
    step.sort_by(f64::total_cmp);
    let step_parallax = step.get(step.len() / 2).copied().unwrap_or(0.0);
    let samples = sphere_samples(SAMPLES);
    let owners: Vec<Option<usize>> = samples.iter().map(|&d| rig.owner(d)).collect();
    let nv = rig.views.len();
    let all: Vec<(usize, usize)> = (0..nv).flat_map(|a| (0..nv).map(move |b| (a, b))).collect();
    let (mut pairs, mut frame_pairs, mut unknown, mut all_views) = (Vec::new(), 0, 0, 0);
    for level in 0..opts.levels {
        let offset = 1usize << level;
        let ring = margin_ring((opts.margin_deg + offset as f64 * step_parallax).min(MAX_MARGIN_DEG));
        for f in 0..n.saturating_sub(offset) {
            let g = f + offset;
            frame_pairs += 1;
            all_views += views[f].len() * views[g].len();
            let ((cf, rf), (cg, rg)) = (absolute[f], absolute[g]);
            let view_pairs = if cf == cg {
                overlapping_views(rig, &samples, &owners, &ring, rg * rf.transpose(), opts.min_share)
            } else {
                unknown += 1;
                all.clone()
            };
            pairs.extend(view_pairs.into_iter().filter(|(a, b)| views[f].contains(a) && views[g].contains(b)).map(|(a, b)| (f, a, g, b)));
        }
    }
    let mut residuals: Vec<f64> = fits.values().map(|f| f.median_deg).collect();
    residuals.sort_by(f64::total_cmp);
    let summary = PairsSummary {
        frames: n,
        views: nv,
        rotations_fitted: fits.len(),
        rotations_probed: probes.len(),
        rotations_inconsistent: fitted - fits.len(),
        rotation_chains: chains,
        frame_pairs,
        frame_pairs_unknown: unknown,
        image_pairs: pairs.len(),
        image_pairs_all_views: all_views,
        median_residual_deg: residuals.get(residuals.len() / 2).copied(),
        step_parallax_deg: step_parallax,
    };
    (pairs, summary)
}

fn angle_deg(a: DMat3, b: DMat3) -> f64 {
    let m = a * b.transpose();
    ((m.x_axis.x + m.y_axis.y + m.z_axis.z - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

/// A fitted rotation and its residuals (degrees).
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub r: DMat3,
    pub median_deg: f64,
    pub p90_deg: f64,
}

/// The rotation R with b ~= R a over the ray pairs: a fit, then a refit on the pairs within
/// [`MAX_RESIDUAL_DEG`]. None below [`MIN_ROTATION_INLIERS`].
pub fn fit_rotation(pairs: &[(DVec3, DVec3)]) -> Option<Fit> {
    let fit = |pairs: &[&(DVec3, DVec3)]| -> Option<DMat3> {
        let mut h = DMat3::ZERO;
        for (a, b) in pairs {
            // sum b a^T
            h += DMat3::from_cols(*b * a.x, *b * a.y, *b * a.z);
        }
        polar_rotation(h)
    };
    if pairs.len() < MIN_ROTATION_INLIERS {
        return None;
    }
    let r = fit(&pairs.iter().collect::<Vec<_>>())?;
    let limit = MAX_RESIDUAL_DEG.to_radians();
    let kept: Vec<&(DVec3, DVec3)> = pairs.iter().filter(|(a, b)| (r * *a).angle_between(*b) <= limit).collect();
    if kept.len() < MIN_ROTATION_INLIERS {
        return None;
    }
    let r = fit(&kept)?;
    let mut res: Vec<f64> = kept.iter().map(|(a, b)| (r * *a).angle_between(*b).to_degrees()).collect();
    res.sort_by(f64::total_cmp);
    Some(Fit { r, median_deg: res[res.len() / 2], p90_deg: res[res.len() * 9 / 10] })
}

/// The closest rotation to `h` (orthogonal polar factor, Newton iteration). None when `h` is
/// degenerate or the closest orthogonal matrix is a reflection.
fn polar_rotation(h: DMat3) -> Option<DMat3> {
    if !(h.determinant() > 1e-12 * h.to_cols_array().iter().map(|v| v.abs()).sum::<f64>().powi(3)) {
        return None;
    }
    let mut x = h;
    for _ in 0..50 {
        let next = (x + x.inverse().transpose()) * 0.5;
        let done = (next - x).to_cols_array().iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-12;
        x = next;
        if done {
            break;
        }
    }
    (x.determinant() > 0.0 && x.is_finite()).then_some(x)
}

/// Fibonacci sphere directions.
fn sphere_samples(n: usize) -> Vec<DVec3> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..n).map(|i| {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - y * y).sqrt();
        let t = golden * i as f64;
        DVec3::new(r * t.cos(), y, r * t.sin())
    }).collect()
}

/// Rotations that move a direction by `margin_deg` in 8 directions, plus the identity.
fn margin_ring(margin_deg: f64) -> Vec<DQuat> {
    let mut ring = vec![DQuat::IDENTITY];
    if margin_deg > 0.0 {
        // Applied in each sample's own tangent frame (see overlapping_views).
        for k in 0..8 {
            let phi = k as f64 * std::f64::consts::FRAC_PI_4;
            ring.push(DQuat::from_axis_angle(DVec3::new(phi.cos(), phi.sin(), 0.0), margin_deg.to_radians()));
        }
    }
    ring
}

/// View pairs (a in the first frame, b in the second) whose owned directions overlap under
/// `r` (second-frame ray = r * first-frame ray): the share of a's directions, each spread over the
/// margin ring, that lands in b reaches `min_share` (either way round).
fn overlapping_views(rig: &Rig, samples: &[DVec3], owners: &[Option<usize>], ring: &[DQuat], r: DMat3, min_share: f64) -> Vec<(usize, usize)> {
    let nv = rig.views.len();
    let mut weight = vec![vec![0.0f64; nv]; nv];
    let mut per_view = vec![0usize; nv];
    for (d, owner) in samples.iter().zip(owners) {
        let Some(a) = *owner else { continue };
        per_view[a] += 1;
        // Ring rotations act in the tangent frame at the moved direction.
        let to_moved = DQuat::from_rotation_arc(DVec3::Z, r * *d);
        for q in ring {
            if let Some(b) = rig.owner(to_moved * (*q * DVec3::Z)) {
                weight[a][b] += 1.0 / ring.len() as f64;
            }
        }
    }
    let share = |a: usize, b: usize| weight[a][b] / per_view[a].max(1) as f64;
    let mut pairs = Vec::new();
    for a in 0..nv {
        for b in 0..nv {
            // b's share of the reverse direction is the same overlap seen from b (r inverted).
            if weight[a][b] > 0.0 && (share(a, b) >= min_share || weight[a][b] / per_view[b].max(1) as f64 >= min_share) {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 14-view layout of splat-core's panorama split (level ring, +35 ring offset 45 deg,
    /// -35 ring, up, down), cam_from_rig = Rx(-pitch) Ry(-yaw).
    pub(crate) fn v14() -> Rig {
        let mut views = Vec::new();
        let mut angles = Vec::new();
        for (pitch, offset) in [(0.0f64, 0.0f64), (35.0, 45.0), (-35.0, 0.0)] {
            for k in 0..4 {
                angles.push((pitch, offset + 90.0 * k as f64));
            }
        }
        angles.push((90.0, 0.0));
        angles.push((-90.0, 0.0));
        for (i, (p, y)) in angles.into_iter().enumerate() {
            let m = DMat3::from_rotation_x(-p.to_radians()) * DMat3::from_rotation_y(-y.to_radians());
            views.push(RigView { prefix: format!("v{i:02}_"), cam_from_rig: m });
        }
        Rig { views, pinhole: [512.0, 512.0, 512.0, 512.0] }
    }

    #[test]
    fn rays_round_trip_and_owners_cover_the_sphere() {
        let rig = v14();
        // The centre pixel of each view is its axis.
        for (i, v) in rig.views.iter().enumerate() {
            let d = rig.ray(i, 512.0, 512.0);
            assert!((v.cam_from_rig * d - DVec3::Z).length() < 1e-12);
            assert_eq!(rig.owner(d), Some(i));
        }
        assert!(sphere_samples(500).iter().all(|&d| rig.owner(d).is_some()));
    }

    #[test]
    fn rotation_fit_recovers_a_yaw_and_rejects_too_few_rays() {
        let truth = DMat3::from_rotation_y(7f64.to_radians()) * DMat3::from_rotation_x(1f64.to_radians());
        let mut pairs: Vec<(DVec3, DVec3)> = sphere_samples(200).into_iter().map(|a| (a, truth * a)).collect();
        // Outliers are dropped by the refit.
        for p in pairs.iter_mut().step_by(10) {
            p.1 = -p.1;
        }
        let fit = fit_rotation(&pairs).unwrap();
        assert!((fit.r - truth).to_cols_array().iter().all(|v| v.abs() < 1e-9), "{:?}", fit.r);
        assert!(fit.median_deg < 1e-6 && fit.p90_deg < 1e-6);
        assert!(fit_rotation(&pairs[..20]).is_none());
    }

    #[test]
    fn overlap_without_rotation_keeps_the_views_near_each_view() {
        let rig = v14();
        let samples = sphere_samples(SAMPLES);
        let owners: Vec<_> = samples.iter().map(|&d| rig.owner(d)).collect();
        let pairs = overlapping_views(&rig, &samples, &owners, &margin_ring(10.0), DMat3::IDENTITY, 0.03);
        // Every view keeps itself, the set is symmetric, and far fewer than 14 x 14 pairs remain.
        for a in 0..14 {
            assert!(pairs.contains(&(a, a)));
        }
        for &(a, b) in &pairs {
            assert!(pairs.contains(&(b, a)), "({a}, {b})");
        }
        assert!(pairs.len() <= 70, "{}", pairs.len());
        // The level view 0 never pairs with the opposite level view 2 or the zenith/nadir views.
        for b in [2, 12, 13] {
            assert!(!pairs.contains(&(0, b)), "0-{b}");
        }
    }

    #[test]
    fn overlap_follows_the_rotation() {
        let rig = v14();
        let samples = sphere_samples(SAMPLES);
        let owners: Vec<_> = samples.iter().map(|&d| rig.owner(d)).collect();
        // The second frame is turned 90 deg in yaw: what view 0 saw, view 1 or 3 now sees.
        let r = DMat3::from_rotation_y(-90f64.to_radians());
        let pairs = overlapping_views(&rig, &samples, &owners, &margin_ring(10.0), r, 0.03);
        let level0: Vec<usize> = pairs.iter().filter(|p| p.0 == 0).map(|p| p.1).filter(|&b| b < 4).collect();
        assert_eq!(level0.len(), 1, "{pairs:?}");
        let b = level0[0];
        assert!(b == 1 || b == 3);
        // The direction view 0 looks along lands on view b's axis.
        assert!((rig.views[b].cam_from_rig * (r * DVec3::Z) - DVec3::Z).length() < 1e-9);
    }

    /// Probe rays of a rig turning `yaw_step` degrees per frame about the vertical, with no
    /// translation: sample directions seen from both frames of each probed pair.
    fn synthetic_probes(n: usize, yaw_step: f64) -> BTreeMap<(usize, usize), Vec<(DVec3, DVec3)>> {
        let turn = |f: usize| DMat3::from_rotation_y((yaw_step * f as f64).to_radians());
        let mut probes = BTreeMap::new();
        for f in 0..n {
            for off in PROBE_OFFSETS {
                let g = f + off;
                if g >= n {
                    continue;
                }
                let r = turn(g) * turn(f).transpose();
                let rays: Vec<(DVec3, DVec3)> = sphere_samples(400).into_iter().map(|d| (d, r * d)).collect();
                probes.insert((f, g), rays);
            }
        }
        probes
    }

    #[test]
    fn chosen_pairs_follow_the_chained_turn_and_fall_back_without_it() {
        let rig = v14();
        let n = 12;
        let views: Vec<BTreeSet<usize>> = vec![(0..14).collect(); n];
        let opts = RigPairs::default();
        let probes = synthetic_probes(n, 10.0);
        let (pairs, s) = choose_pairs(&rig, &views, &probes, &opts);
        assert_eq!((s.rotations_fitted, s.rotation_chains, s.frame_pairs_unknown, s.rotations_inconsistent), (probes.len(), 1, 0, 0));
        assert!(s.step_parallax_deg < 1e-6);
        assert_eq!(s.frame_pairs, (n - 1) + (n - 2) + (n - 4) + (n - 8));
        assert!(s.image_pairs * 2 < s.image_pairs_all_views, "{s:?}");
        // Frames 0 and 8 are 80 deg apart: view 0's axis lands closest to the level view whose
        // axis is nearest the turned direction, and that pair is kept.
        let turned = DMat3::from_rotation_y(80f64.to_radians()) * DVec3::Z;
        let nearest = (0..4).max_by(|&a, &b| (rig.views[a].cam_from_rig * turned).z.total_cmp(&(rig.views[b].cam_from_rig * turned).z)).unwrap();
        assert!(pairs.contains(&(0, 0, 8, nearest)), "0 -> {nearest}");
        // Without the frame 5 probes the chain breaks in two: pairs across it keep every view.
        let mut broken = probes.clone();
        broken.retain(|k, _| k.1 != 5);
        let (pairs, s) = choose_pairs(&rig, &views, &broken, &opts);
        assert_eq!(s.rotation_chains, 2);
        assert!(s.frame_pairs_unknown > 0);
        assert_eq!(pairs.iter().filter(|p| p.0 == 4 && p.2 == 5).count(), 14 * 14);
        // A probe that disagrees with its triangle is dropped with its triangle.
        let mut bad = probes.clone();
        for ray in bad.get_mut(&(3, 5)).unwrap() {
            ray.1 = DMat3::from_rotation_y(0.3) * ray.1;
        }
        let (_, s) = choose_pairs(&rig, &views, &bad, &opts);
        assert_eq!(s.rotations_inconsistent, 3, "{s:?}");
        // Frame 4 still chains from frame 2; frame 5 has no trusted link and starts a new chain.
        assert_eq!(s.rotation_chains, 2, "{s:?}");
    }

    #[test]
    fn write_pairs_reads_colmap_tables_in_either_id_order() {
        let rig = v14();
        let (n, step) = (6usize, 12.0f64);
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("database.db");
        let db = rusqlite::Connection::open(&database).unwrap();
        db.execute_batch("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER);
            CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB, config INTEGER);").unwrap();
        // Later frames get smaller ids, so stored matches list the later frame's keypoint first.
        let id = |f: usize, v: usize| ((n - 1 - f) * 14 + v + 1) as i64;
        let mut keypoints: HashMap<i64, Vec<[f32; 6]>> = HashMap::new();
        let mut matches: BTreeMap<(i64, i64), Vec<[u32; 2]>> = BTreeMap::new();
        let turn = |f: usize| DMat3::from_rotation_y((step * f as f64).to_radians());
        let pixel = |v: usize, d: DVec3| -> Option<[f32; 2]> {
            let c = rig.views[v].cam_from_rig * d;
            let (x, y) = (512.0 * c.x / c.z + 512.0, 512.0 * c.y / c.z + 512.0);
            (c.z > 0.0 && (0.0..1024.0).contains(&x) && (0.0..1024.0).contains(&y)).then_some([x as f32, y as f32])
        };
        for f in 0..n {
            for v in 0..14 {
                db.execute("INSERT INTO images VALUES (?1, ?2, 1)", rusqlite::params![id(f, v), format!("v{v:02}_frame_{f:03}.jpg")]).unwrap();
            }
            for off in PROBE_OFFSETS {
                let g = f + off;
                if g >= n {
                    continue;
                }
                let r = turn(g) * turn(f).transpose();
                for d in sphere_samples(4000) {
                    let Some(v) = rig.owner(d) else { continue };
                    let (Some(p), Some(q)) = (pixel(v, d), pixel(v, r * d)) else { continue };
                    let (a, b) = (id(f, v), id(g, v));
                    let ka = keypoints.entry(a).or_default();
                    ka.push([p[0], p[1], 1.0, 0.0, 0.0, 1.0]);
                    let ia = (ka.len() - 1) as u32;
                    let kb = keypoints.entry(b).or_default();
                    kb.push([q[0], q[1], 1.0, 0.0, 0.0, 1.0]);
                    let ib = (kb.len() - 1) as u32;
                    let (lo, hi, m) = if a < b { (a, b, [ia, ib]) } else { (b, a, [ib, ia]) };
                    matches.entry((lo, hi)).or_default().push(m);
                }
            }
        }
        for (image, kps) in &keypoints {
            let blob: Vec<u8> = kps.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            db.execute("INSERT INTO keypoints VALUES (?1, ?2, 6, ?3)", rusqlite::params![image, kps.len() as i64, blob]).unwrap();
        }
        for ((lo, hi), m) in &matches {
            let blob: Vec<u8> = m.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            db.execute("INSERT INTO two_view_geometries VALUES (?1, ?2, 2, ?3, 2)", rusqlite::params![lo * 2_147_483_647 + hi, m.len() as i64, blob]).unwrap();
        }
        drop(db);
        let path = dir.path().join("rig_pairs.txt");
        let s = write_pairs(&database, &rig, &RigPairs::default(), &path).unwrap();
        assert_eq!((s.frames, s.rotations_fitted, s.rotation_chains, s.frame_pairs_unknown), (n, 2 * n - 3, 1, 0), "{s:?}");
        assert!(s.median_residual_deg.unwrap() < 0.1, "{s:?}");
        let lines = std::fs::read_to_string(&path).unwrap();
        assert_eq!(lines.lines().count(), s.image_pairs);
        assert!(lines.lines().all(|l| { let w: Vec<&str> = l.split(' ').collect(); w.len() == 2 && w[0].ends_with(".jpg") && w[1].ends_with(".jpg") }));
        assert!(lines.lines().any(|l| l == "v00_frame_000.jpg v00_frame_001.jpg"));
        assert!(path.with_extension("json").is_file());
        // Probe list: each view with the same view 1 and 2 frames on.
        let probe = dir.path().join("probe.txt");
        assert_eq!(write_probe(&database, &rig, &probe).unwrap(), 14 * ((n - 1) + (n - 2)));
        assert_eq!(write_loop_queries(&database, &dir.path().join("loop.txt")).unwrap(), (14 * n).div_ceil(LOOP_PERIOD));
    }
}
