//! Robust fits for the fill depth: trimmed affine (disparity) and trimmed plane.

use glam::{DMat3, DVec3, Vec3};

/// Indices of the `keep` fraction (at least 16) of smallest residuals.
fn trim(res: &[f64], keep: f64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..res.len()).collect();
    idx.sort_by(|&a, &b| res[a].total_cmp(&res[b]));
    idx.truncate(((res.len() as f64 * keep) as usize).max(16).min(res.len()));
    idx
}

/// Least squares `y ~ s x + t`, refit 3 times on the best 80 %. Returns (s, t).
pub fn robust_affine(x: &[f32], y: &[f32]) -> (f64, f64) {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    let mut sol = (1.0, 0.0);
    for _ in 0..3 {
        let n = idx.len() as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for &i in &idx {
            let (a, b) = (x[i] as f64, y[i] as f64);
            sx += a;
            sy += b;
            sxx += a * a;
            sxy += a * b;
        }
        let det = n * sxx - sx * sx;
        sol = if det.abs() > 1e-12 {
            let s = (n * sxy - sx * sy) / det;
            (s, (sy - s * sx) / n)
        } else {
            (0.0, sy / n.max(1.0))
        };
        let res: Vec<f64> = x.iter().zip(y).map(|(&a, &b)| (sol.0 * a as f64 + sol.1 - b as f64).abs()).collect();
        idx = trim(&res, 0.8);
    }
    sol
}

/// Eigenvector of the smallest eigenvalue of a symmetric 3x3 (cyclic Jacobi).
fn smallest_eigvec(m: DMat3) -> DVec3 {
    let mut a = m.to_cols_array_2d(); // a[col][row]; symmetric so the layout does not matter
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-15 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let k = (0..3).min_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap_or(2);
    DVec3::new(v[0][k], v[1][k], v[2][k]).normalize_or(DVec3::Z)
}

/// Plane through points, refit 3 times on the best 80 % by distance. Returns (centre, unit normal).
pub fn fit_plane(p: &[Vec3]) -> (Vec3, Vec3) {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    let (mut c, mut n) = (DVec3::ZERO, DVec3::Z);
    for _ in 0..3 {
        c = idx.iter().map(|&i| p[i].as_dvec3()).sum::<DVec3>() / idx.len().max(1) as f64;
        let mut cov = DMat3::ZERO;
        for &i in &idx {
            let d = p[i].as_dvec3() - c;
            cov += DMat3::from_cols(d * d.x, d * d.y, d * d.z);
        }
        n = smallest_eigvec(cov);
        let res: Vec<f64> = p.iter().map(|q| (q.as_dvec3() - c).dot(n).abs()).collect();
        idx = trim(&res, 0.8);
    }
    (c.as_vec3(), n.as_vec3())
}

/// Median of the finite values (NaN when there are none).
pub fn median(mut v: Vec<f32>) -> f32 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f32::NAN;
    }
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, f32::total_cmp).1
}
