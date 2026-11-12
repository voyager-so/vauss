//! Rotating real spherical-harmonic coefficients (3DGS basis and sign convention).
//!
//! For each band l the rotated coefficients are a linear map D_l of the originals. Rather than
//! hand-deriving Wigner matrices for this basis, D_l is fitted by least squares on sample directions:
//! sum_m c'_m Y_m(d) = sum_m c_m Y_m(R^T d) for all d. The basis is exact, so the fit is exact.

use glam::{DMat3, DVec3};

const C1: f64 = 0.488_602_511_902_919_9;
const C2: [f64; 5] = [1.092_548_430_592_079_2, -1.092_548_430_592_079_2, 0.315_391_565_252_520_05, -1.092_548_430_592_079_2, 0.546_274_215_296_039_6];
const C3: [f64; 7] = [
    -0.590_043_589_926_643_5, 2.890_611_442_640_554, -0.457_045_799_464_465_8, 0.373_176_332_590_115_4,
    -0.457_045_799_464_465_8, 1.445_305_721_320_277, -0.590_043_589_926_643_5,
];

/// Higher-order basis (bands 1..=3, 15 values) at unit direction d.
fn basis(d: DVec3) -> [f64; 15] {
    let (x, y, z) = (d.x, d.y, d.z);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    [
        -C1 * y,
        C1 * z,
        -C1 * x,
        C2[0] * x * y,
        C2[1] * y * z,
        C2[2] * (2.0 * zz - xx - yy),
        C2[3] * x * z,
        C2[4] * (xx - yy),
        C3[0] * y * (3.0 * xx - yy),
        C3[1] * x * y * z,
        C3[2] * y * (4.0 * zz - xx - yy),
        C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        C3[4] * x * (4.0 * zz - xx - yy),
        C3[5] * z * (xx - yy),
        C3[6] * x * (xx - 3.0 * yy),
    ]
}

/// Per-band rotation matrices for the higher-order coefficients under world rotation `r`.
pub struct ShRotation {
    /// Band l (1..=3) occupies offset l*l-1, size 2l+1; dense row-major blocks.
    bands: Vec<Vec<f64>>,
}

impl ShRotation {
    pub fn new(r: DMat3) -> Self {
        let dirs: Vec<DVec3> = (0..96)
            .map(|i| {
                // Fibonacci sphere
                let t = (i as f64 + 0.5) / 96.0;
                let zc = 1.0 - 2.0 * t;
                let ph = i as f64 * std::f64::consts::PI * (3.0 - 5f64.sqrt());
                let s = (1.0 - zc * zc).sqrt();
                DVec3::new(s * ph.cos(), s * ph.sin(), zc)
            })
            .collect();
        let rt = r.transpose();
        let bands = (1..=3usize)
            .map(|l| {
                let (off, n) = (l * l - 1, 2 * l + 1);
                // Normal equations: (A^T A) D = A^T B
                let mut ata = vec![0.0; n * n];
                let mut atb = vec![0.0; n * n];
                for d in &dirs {
                    let a = basis(*d);
                    let b = basis(rt * *d);
                    for i in 0..n {
                        for j in 0..n {
                            ata[i * n + j] += a[off + i] * a[off + j];
                            atb[i * n + j] += a[off + i] * b[off + j];
                        }
                    }
                }
                solve(n, ata, atb)
            })
            .collect();
        Self { bands }
    }

    /// Rotate one splat's coefficients in place; layout [coef][channel], `coeffs` per channel.
    pub fn apply(&self, rest: &mut [f32], coeffs: usize) {
        let mut tmp = [0.0f64; 7];
        for (l, d) in self.bands.iter().enumerate() {
            let (off, n) = ((l + 1) * (l + 1) - 1, 2 * l + 3);
            if off + n > coeffs {
                break;
            }
            for c in 0..3 {
                for (i, t) in tmp.iter_mut().enumerate().take(n) {
                    *t = (0..n).map(|j| d[i * n + j] * rest[(off + j) * 3 + c] as f64).sum();
                }
                for (i, t) in tmp.iter().enumerate().take(n) {
                    rest[(off + i) * 3 + c] = *t as f32;
                }
            }
        }
    }
}

/// Solve A X = B for X (n x n, row-major) by Gauss-Jordan with partial pivoting.
fn solve(n: usize, mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs())).unwrap();
        for k in 0..n {
            a.swap(col * n + k, piv * n + k);
            b.swap(col * n + k, piv * n + k);
        }
        let p = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= p;
            b[col * n + k] /= p;
        }
        for row in 0..n {
            if row != col {
                let f = a[row * n + col];
                for k in 0..n {
                    a[row * n + k] -= f * a[col * n + k];
                    b[row * n + k] -= f * b[col * n + k];
                }
            }
        }
    }
    b
}
