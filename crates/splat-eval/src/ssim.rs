//! SSIM matching torchmetrics 1.9 `structural_similarity_index_measure(data_range=1,
//! gaussian_kernel=True, kernel_size=11, sigma=1.5)`: reflect padding by 5, Gaussian window
//! (separable here, a 2D outer product there), K1 = 0.01, K2 = 0.03, variances clamped at 0, and the
//! mean over every channel and pixel of the unpadded size. torchmetrics >= 1.0 does not crop the
//! border before the mean on this path (older releases did), so neither do we.

use rayon::prelude::*;

use crate::images::Rgb;

const RADIUS: usize = 5; // int(3.5 * sigma + 0.5) for sigma = 1.5
const TAPS: usize = 2 * RADIUS + 1;
const SIGMA: f64 = 1.5;
const C1: f64 = 0.01 * 0.01;
const C2: f64 = 0.03 * 0.03;

pub fn ssim(render: &Rgb, gt: &Rgb) -> f64 {
    assert_eq!(
        (render.width, render.height),
        (gt.width, gt.height),
        "ssim: size mismatch"
    );
    let (w, h) = (render.width, render.height);
    if w == 0 || h == 0 {
        return f64::NAN;
    }
    let kernel = gaussian();
    let plane = w * h;
    let sum: f64 = (0..3)
        .into_par_iter()
        .map(|c| {
            let x = &render.data[c * plane..(c + 1) * plane];
            let y = &gt.data[c * plane..(c + 1) * plane];
            channel_sum(x, y, w, h, &kernel)
        })
        .sum();
    sum / (3 * plane) as f64
}

fn gaussian() -> [f64; TAPS] {
    let mut k = [0.0; TAPS];
    for (i, v) in k.iter_mut().enumerate() {
        let d = (i as f64 - RADIUS as f64) / SIGMA;
        *v = (-d * d / 2.0).exp();
    }
    let s: f64 = k.iter().sum();
    k.map(|v| v / s)
}

/// torch `F.pad(mode="reflect")` index (edge not repeated), bouncing for images smaller than the pad.
fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let m = i.rem_euclid(period);
    (if m < n as isize { m } else { period - m }) as usize
}

/// Sum of the SSIM map over one channel.
fn channel_sum(x: &[f32], y: &[f32], w: usize, h: usize, k: &[f64; TAPS]) -> f64 {
    let ph = h + 2 * RADIUS;
    let cols: Vec<usize> = (0..w + 2 * RADIUS)
        .map(|i| reflect(i as isize - RADIUS as isize, w))
        .collect();
    // Horizontal pass over the reflect-padded rows: [mu_x, mu_y, xx, yy, xy] per (padded row, x).
    let mut horiz = vec![[0f64; 5]; ph * w];
    horiz.par_chunks_mut(w).enumerate().for_each(|(pr, out)| {
        let row = reflect(pr as isize - RADIUS as isize, h) * w;
        for (ox, o) in out.iter_mut().enumerate() {
            let mut acc = [0f64; 5];
            for (t, &kv) in k.iter().enumerate() {
                let i = row + cols[ox + t];
                let (a, b) = (x[i] as f64, y[i] as f64);
                acc[0] += kv * a;
                acc[1] += kv * b;
                acc[2] += kv * a * a;
                acc[3] += kv * b * b;
                acc[4] += kv * a * b;
            }
            *o = acc;
        }
    });
    (0..h)
        .into_par_iter()
        .map(|oy| {
            let mut row_sum = 0.0;
            for ox in 0..w {
                let mut s = [0f64; 5];
                for (t, &kv) in k.iter().enumerate() {
                    let v = &horiz[(oy + t) * w + ox];
                    for j in 0..5 {
                        s[j] += kv * v[j];
                    }
                }
                let [mx, my, xx, yy, xy] = s;
                let (mx2, my2, mxy) = (mx * mx, my * my, mx * my);
                let vx = (xx - mx2).max(0.0);
                let vy = (yy - my2).max(0.0);
                let cov = xy - mxy;
                row_sum +=
                    ((2.0 * mxy + C1) * (2.0 * cov + C2)) / ((mx2 + my2 + C1) * (vx + vy + C2));
            }
            row_sum
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> f32) -> Rgb {
        let mut data = vec![0.0; 3 * w * h];
        for c in 0..3 {
            for yy in 0..h {
                for xx in 0..w {
                    data[c * w * h + yy * w + xx] = f(c, xx, yy);
                }
            }
        }
        Rgb {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn identical_is_one() {
        let a = img(37, 23, |c, x, y| {
            ((x * 7 + y * 13 + c * 5) % 17) as f32 / 16.0
        });
        assert!((ssim(&a, &a) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reflect_matches_torch() {
        let idx: Vec<usize> = (-3..9).map(|i| reflect(i, 6)).collect();
        assert_eq!(idx, [3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2]);
    }

    #[test]
    fn torchmetrics_reference_value() {
        // torchmetrics 1.9 in float64 on the same 3x23x37 image and its 0.75x copy.
        let a = img(37, 23, |c, x, y| {
            ((x * 7 + y * 13 + c * 5) % 17) as f32 / 16.0
        });
        let mut b = a.clone();
        b.data.iter_mut().for_each(|v| *v *= 0.75);
        let s = ssim(&a, &b);
        assert!((s - 0.9218450205404968).abs() < 1e-12, "{s}");
    }
}
