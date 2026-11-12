//! Colour correction of a render onto its ground truth before scoring (cc_* metrics).

use rayon::prelude::*;

use crate::images::Rgb;

const QUAD_TERMS: usize = 10;
const UPPER: usize = QUAD_TERMS * (QUAD_TERMS + 1) / 2;
const CHUNK: usize = 1 << 14;

/// Zip-NeRF eval colour correction (camp_zipnerf `color_correct`, Apache-2.0; reimplemented):
/// per channel least squares over [r*r, r*g, r*b, g*g, g*b, b*b, r, g, b, 1], 5 iterations,
/// ignoring pixels clipped (< eps or > 1-eps, eps = 0.5/255) in the render or the GT, in f64.
pub fn correct_quadratic(render: &Rgb, gt: &Rgb, mask: Option<&[f32]>) -> Rgb {
    const ITERS: usize = 5;
    const EPS: f64 = 0.5 / 255.0;
    let unclipped = |z: f64| (EPS..=1.0 - EPS).contains(&z);
    let n = render.width * render.height;
    let mut cur: Vec<[f64; 3]> = pixels(render);
    let reference: Vec<[f64; 3]> = pixels(gt);
    // Per pixel and channel: unclipped in the initial render and the GT, and kept by the mask.
    let fixed: Vec<[bool; 3]> = (0..n)
        .map(|p| {
            let keep = mask.is_none_or(|m| m[p] > 0.0);
            std::array::from_fn(|c| keep && unclipped(cur[p][c]) && unclipped(reference[p][c]))
        })
        .collect();

    for _ in 0..ITERS {
        // Normal equations per channel: upper triangle of A^T M A and A^T M ref.
        let zero = || [[0f64; UPPER + QUAD_TERMS]; 3];
        let sums = cur
            .par_chunks(CHUNK)
            .zip(reference.par_chunks(CHUNK))
            .zip(fixed.par_chunks(CHUNK))
            .fold(zero, |mut acc, ((cur, reference), fixed)| {
                for ((z, r), f) in cur.iter().zip(reference).zip(fixed) {
                    let a = quad_terms(z);
                    for c in 0..3 {
                        if !(f[c] && unclipped(z[c])) {
                            continue;
                        }
                        let s = &mut acc[c];
                        let mut k = 0;
                        for i in 0..QUAD_TERMS {
                            for j in i..QUAD_TERMS {
                                s[k] += a[i] * a[j];
                                k += 1;
                            }
                            s[UPPER + i] += a[i] * r[c];
                        }
                    }
                }
                acc
            })
            .reduce(zero, |mut a, b| {
                for (x, y) in a.iter_mut().flatten().zip(b.iter().flatten()) {
                    *x += y;
                }
                a
            });
        #[allow(clippy::needless_range_loop)] // symmetric fill reads clearer indexed
        let warp: [[f64; QUAD_TERMS]; 3] = std::array::from_fn(|c| {
            let s = &sums[c];
            let mut m = [[0f64; QUAD_TERMS]; QUAD_TERMS];
            let mut k = 0;
            for i in 0..QUAD_TERMS {
                for j in i..QUAD_TERMS {
                    (m[i][j], m[j][i]) = (s[k], s[k]);
                    k += 1;
                }
            }
            solve_symmetric(m, std::array::from_fn(|i| s[UPPER + i]))
        });
        cur.par_iter_mut().for_each(|z| {
            let a = quad_terms(z);
            *z = std::array::from_fn(|c| dot(&a, &warp[c]).clamp(0.0, 1.0));
        });
    }
    to_rgb(&cur, render.width, render.height)
}

/// 4x3 affine fit A so that [r,g,b,1] @ A ~= gt, clamped to [0,1].
pub fn correct_affine(render: &Rgb, gt: &Rgb, mask: Option<&[f32]>) -> Rgb {
    let x = pixels(render);
    let y = pixels(gt);
    let mut xtx = [[0f64; 4]; 4];
    let mut xty = [[0f64; 4]; 3];
    for p in 0..x.len() {
        if mask.is_some_and(|m| m[p] <= 0.0) {
            continue;
        }
        let a = [x[p][0], x[p][1], x[p][2], 1.0];
        for i in 0..4 {
            for j in 0..4 {
                xtx[i][j] += a[i] * a[j];
            }
            for c in 0..3 {
                xty[c][i] += a[i] * y[p][c];
            }
        }
    }
    // metrics.py casts the fitted matrix to float32 before applying it.
    let warp: [[f64; 4]; 3] =
        std::array::from_fn(|c| solve_symmetric(xtx, xty[c]).map(|v| v as f32 as f64));
    let out: Vec<[f64; 3]> = x
        .iter()
        .map(|z| {
            let a = [z[0], z[1], z[2], 1.0];
            std::array::from_fn(|c| dot(&a, &warp[c]).clamp(0.0, 1.0))
        })
        .collect();
    to_rgb(&out, render.width, render.height)
}

fn quad_terms(z: &[f64; 3]) -> [f64; QUAD_TERMS] {
    let [r, g, b] = *z;
    [r * r, r * g, r * b, g * g, g * b, b * b, r, g, b, 1.0]
}

fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn pixels(im: &Rgb) -> Vec<[f64; 3]> {
    let n = im.width * im.height;
    (0..n)
        .map(|p| std::array::from_fn(|c| im.data[c * n + p] as f64))
        .collect()
}

fn to_rgb(px: &[[f64; 3]], width: usize, height: usize) -> Rgb {
    let n = px.len();
    let mut data = vec![0f32; 3 * n];
    for (p, z) in px.iter().enumerate() {
        for c in 0..3 {
            data[c * n + p] = z[c] as f32;
        }
    }
    Rgb {
        width,
        height,
        data,
    }
}

/// Minimum-norm least-squares solution of the symmetric PSD system `m x = b` via a cyclic Jacobi
/// eigendecomposition. Like torch.linalg.lstsq (gelsy, rcond = eps * N), directions with eigenvalue
/// below rcond * max|eigenvalue| are dropped, so singular systems (constant images, everything
/// clipped or masked) give a finite answer instead of NaN; an all-zero system gives x = 0.
#[allow(clippy::needless_range_loop)] // Jacobi rotations index rows and columns of m and v
fn solve_symmetric<const N: usize>(mut m: [[f64; N]; N], b: [f64; N]) -> [f64; N] {
    let mut v = [[0f64; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..N)
            .flat_map(|i| (0..N).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| m[i][j] * m[i][j])
            .sum();
        let diag: f64 = (0..N).map(|i| m[i][i] * m[i][i]).sum();
        if off <= f64::EPSILON * f64::EPSILON * diag || off == 0.0 {
            break;
        }
        for p in 0..N {
            for q in p + 1..N {
                if m[p][q] == 0.0 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let (mkp, mkq) = (m[k][p], m[k][q]);
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..N {
                    let (mpk, mqk) = (m[p][k], m[q][k]);
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let lmax = (0..N).map(|i| m[i][i].abs()).fold(0.0, f64::max);
    let cutoff = f64::EPSILON * N as f64 * lmax;
    let mut x = [0f64; N];
    for k in 0..N {
        let l = m[k][k];
        if l.is_nan() || l <= cutoff {
            continue;
        }
        let coef = (0..N).map(|i| v[i][k] * b[i]).sum::<f64>() / l;
        for i in 0..N {
            x[i] += coef * v[i][k];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize) -> Rgb {
        let n = w * h;
        let mut data = vec![0f32; 3 * n];
        for p in 0..n {
            let (x, y) = ((p % w) as f32 / w as f32, (p / w) as f32 / h as f32);
            data[p] = 0.1 + 0.6 * x;
            data[n + p] = 0.15 + 0.5 * y;
            data[2 * n + p] = 0.2 + 0.3 * x + 0.3 * y;
        }
        Rgb {
            width: w,
            height: h,
            data,
        }
    }

    fn shifted(im: &Rgb) -> Rgb {
        // A cross-channel affine drift, kept inside (eps, 1 - eps) so no pixel is clipped.
        let n = im.width * im.height;
        let d = &im.data;
        let mut data = vec![0f32; 3 * n];
        for p in 0..n {
            let (r, g, b) = (d[p], d[n + p], d[2 * n + p]);
            data[p] = 0.8 * r + 0.05 * g + 0.05;
            data[n + p] = 0.9 * g + 0.1 * b - 0.02;
            data[2 * n + p] = 0.7 * b + 0.1 * r + 0.1;
        }
        Rgb { data, ..im.clone() }
    }

    fn max_diff(a: &Rgb, b: &Rgb) -> f32 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn undoes_affine_shift() {
        let gt = ramp(64, 48);
        let render = shifted(&gt);
        assert!(max_diff(&render, &gt) > 0.05);
        assert!(max_diff(&correct_quadratic(&render, &gt, None), &gt) < 1e-5);
        assert!(max_diff(&correct_affine(&render, &gt, None), &gt) < 1e-5);
    }

    #[test]
    fn mask_restricts_the_fit() {
        let gt = ramp(64, 48);
        let mut render = shifted(&gt);
        let n = 64 * 48;
        let mask: Vec<f32> = (0..n)
            .map(|p| if p % 64 < 32 { 1.0 } else { 0.0 })
            .collect();
        for p in (0..n).filter(|p| mask[*p] == 0.0) {
            render.data[p] = 0.9; // garbage outside the mask must not affect the fit
        }
        let out = correct_affine(&render, &gt, Some(&mask));
        let kept = (0..n)
            .filter(|p| mask[*p] > 0.0)
            .map(|p| (out.data[p] - gt.data[p]).abs())
            .fold(0.0, f32::max);
        assert!(kept < 1e-5, "{kept}");
    }

    #[test]
    fn singular_systems_stay_finite() {
        let constant = Rgb {
            width: 8,
            height: 8,
            data: vec![0.5; 192],
        };
        let gt = ramp(8, 8);
        for out in [
            correct_quadratic(&constant, &gt, None),
            correct_affine(&constant, &gt, None),
        ] {
            assert!(
                out.data
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            );
        }
        // Everything masked out: the fit is empty and the output is all zeros, like torch.
        let out = correct_quadratic(&gt, &gt, Some(&[0.0; 64]));
        assert!(out.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn solver_matches_direct_solution() {
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let x = [1.0, -2.0, 0.5];
        let b: [f64; 3] = std::array::from_fn(|i| dot(&m[i], &x));
        let got = solve_symmetric(m, b);
        for i in 0..3 {
            assert!((got[i] - x[i]).abs() < 1e-12);
        }
    }
}
