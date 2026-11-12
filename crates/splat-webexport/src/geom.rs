//! Plane fitting: RANSAC for horizontal floors and vertical walls, PCA refinement.

use glam::{DMat3, DVec3};

/// Small deterministic PRNG (splitmix64) so exports are reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Plane n.p = h (unit n).
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub n: DVec3,
    pub h: f64,
    pub inliers: usize,
}

impl Plane {
    pub fn dist(&self, p: DVec3) -> f64 {
        p.dot(self.n) - self.h
    }
}

/// Least-squares plane through points (smallest-eigenvector of the covariance), oriented along `hint`.
pub fn fit_plane(pts: &[DVec3], hint: DVec3) -> Option<Plane> {
    if pts.len() < 3 {
        return None;
    }
    let c = pts.iter().sum::<DVec3>() / pts.len() as f64;
    let mut cov = DMat3::ZERO;
    for p in pts {
        let d = *p - c;
        cov += DMat3::from_cols(d * d.x, d * d.y, d * d.z);
    }
    let n = smallest_eigvec(cov);
    let n = if n.dot(hint) < 0.0 { -n } else { n };
    Some(Plane { n, h: n.dot(c), inliers: pts.len() })
}

/// Eigenvector of the smallest eigenvalue of a symmetric 3x3 (inverse power iteration on a shift).
fn smallest_eigvec(m: DMat3) -> DVec3 {
    // Shift so the smallest eigenvalue becomes the largest of (tr*I - m), then power-iterate.
    let tr = m.x_axis.x + m.y_axis.y + m.z_axis.z;
    let s = DMat3::from_diagonal(DVec3::splat(tr)) - m;
    let mut v = DVec3::new(0.577, 0.577, 0.577);
    for _ in 0..200 {
        let w = s * v;
        let l = w.length();
        if l < 1e-300 {
            break;
        }
        v = w / l;
    }
    v
}

pub struct RansacOpts {
    pub iters: usize,
    pub thresh: f64,
    /// Candidate normals must be within this angle of `axis` (horizontal planes) ...
    pub max_tilt_deg: f64,
    /// ... or, if true, perpendicular to `axis` within that angle (vertical planes).
    pub vertical: bool,
}

/// Best plane by inlier count among `active` points; returns plane + inlier mask.
pub fn ransac(pts: &[DVec3], active: &[bool], axis: DVec3, o: &RansacOpts, rng: &mut Rng) -> Option<(Plane, Vec<bool>)> {
    let idx: Vec<usize> = (0..pts.len()).filter(|&i| active[i]).collect();
    if idx.len() < 50 {
        return None;
    }
    let cos_tol = o.max_tilt_deg.to_radians().cos();
    let sin_tol = o.max_tilt_deg.to_radians().sin();
    let mut best: Option<Plane> = None;
    for _ in 0..o.iters {
        let n = if o.vertical {
            let (a, b) = (pts[idx[rng.below(idx.len())]], pts[idx[rng.below(idx.len())]]);
            let n = (b - a).cross(axis);
            if n.length() < 1e-9 {
                continue;
            }
            let n = n.normalize();
            (n, a)
        } else {
            let (a, b, c) = (pts[idx[rng.below(idx.len())]], pts[idx[rng.below(idx.len())]], pts[idx[rng.below(idx.len())]]);
            let n = (b - a).cross(c - a);
            if n.length() < 1e-12 {
                continue;
            }
            let n = n.normalize();
            let n = if n.dot(axis) < 0.0 { -n } else { n };
            if n.dot(axis) < cos_tol {
                continue;
            }
            (n, a)
        };
        let (n, a) = n;
        if o.vertical && n.dot(axis).abs() > sin_tol {
            continue;
        }
        let h = n.dot(a);
        let count = idx.iter().filter(|&&i| (pts[i].dot(n) - h).abs() < o.thresh).count();
        if best.is_none_or(|b| count > b.inliers) {
            best = Some(Plane { n, h, inliers: count });
        }
    }
    let rough = best?;
    // Refine on inliers, then recount.
    let inl: Vec<DVec3> = idx.iter().map(|&i| pts[i]).filter(|p| rough.dist(*p).abs() < o.thresh).collect();
    let mut plane = fit_plane(&inl, rough.n).unwrap_or(rough);
    if !o.vertical && plane.n.dot(axis) < cos_tol {
        plane = rough;
    }
    let mut mask = vec![false; pts.len()];
    let mut count = 0;
    for &i in &idx {
        if plane.dist(pts[i]).abs() < o.thresh {
            mask[i] = true;
            count += 1;
        }
    }
    plane.inliers = count;
    Some((plane, mask))
}
