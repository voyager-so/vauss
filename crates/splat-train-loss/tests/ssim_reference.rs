//! SSIM against an independent CPU reference.
//!
//! The reference is the 3DGS / pytorch-msssim `ssim()`: an 11x11 Gaussian
//! window at sigma 1.5, 'same' convolution with zero padding, C1 = 0.01²,
//! C2 = 0.03², averaged over every pixel and channel. It runs in f64. The
//! forward map is compared pixel by pixel; the gradient of the mean is
//! compared with a central finite difference of the reference at probe
//! pixels (corners, edges, tile seams, interior), so it shares no
//! derivation with the kernels' analytic backward.
//!
//! Three sizes cover both training paths: below `FUSED_SSIM_MAX_PIXELS` the
//! backward blurs coefficients the forward stored; above it (2048x2049) the
//! backward recomputes them. 360x480 is the DyCheck frame size the 4D
//! prototype trained on.

use brush_loss::{ImageLossConfig, image_loss, image_loss_eval};
use burn::tensor::{Device, Int, Tensor, TensorData};
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg(target_family = "wasm")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const C1: f64 = 0.01 * 0.01;
const C2: f64 = 0.03 * 0.03;

fn taps() -> [f64; 11] {
    let mut w = [0.0; 11];
    for (i, v) in w.iter_mut().enumerate() {
        let x = i as f64 - 5.0;
        *v = (-x * x / (2.0 * 1.5 * 1.5)).exp();
    }
    let s: f64 = w.iter().sum();
    w.map(|v| v / s)
}

/// Deterministic image-like content: smooth shading plus hashed texture, in
/// `[0, 1]`, `[H, W, 3]`.
fn scene(h: usize, w: usize, seed: u32) -> Vec<f64> {
    let mut out = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut k = (y as u32).wrapping_mul(73_856_093)
                    ^ (x as u32).wrapping_mul(19_349_663)
                    ^ (c as u32 + seed).wrapping_mul(83_492_791);
                k ^= k >> 13;
                k = k.wrapping_mul(0x5bd1_e995);
                k ^= k >> 15;
                let noise = (k % 1000) as f64 / 1000.0 - 0.5;
                let v = 0.5
                    + 0.3 * ((x as f64 / 7.0 + c as f64).sin() * (y as f64 / 5.0).cos())
                    + 0.15 * noise;
                out.push(v.clamp(0.0, 1.0));
            }
        }
    }
    out
}

/// Per-pixel SSIM map `[H, W, 3]` (separable zero-padded blur).
fn ssim_map(x: &[f64], y: &[f64], h: usize, w: usize) -> Vec<f64> {
    let g = taps();
    let blur = |f: &dyn Fn(usize) -> f64| -> Vec<f64> {
        let mut tmp = vec![0.0; h * w * 3];
        for r in 0..h {
            for col in 0..w {
                for c in 0..3 {
                    let mut s = 0.0;
                    for (k, gk) in g.iter().enumerate() {
                        let xx = col as isize + k as isize - 5;
                        if xx >= 0 && (xx as usize) < w {
                            s += gk * f((r * w + xx as usize) * 3 + c);
                        }
                    }
                    tmp[(r * w + col) * 3 + c] = s;
                }
            }
        }
        let mut out = vec![0.0; h * w * 3];
        for r in 0..h {
            for col in 0..w {
                for c in 0..3 {
                    let mut s = 0.0;
                    for (k, gk) in g.iter().enumerate() {
                        let yy = r as isize + k as isize - 5;
                        if yy >= 0 && (yy as usize) < h {
                            s += gk * tmp[(yy as usize * w + col) * 3 + c];
                        }
                    }
                    out[(r * w + col) * 3 + c] = s;
                }
            }
        }
        out
    };
    let mu1 = blur(&|i| x[i]);
    let mu2 = blur(&|i| y[i]);
    let x2 = blur(&|i| x[i] * x[i]);
    let y2 = blur(&|i| y[i] * y[i]);
    let xy = blur(&|i| x[i] * y[i]);
    (0..h * w * 3)
        .map(|i| ssim_from(mu1[i], mu2[i], x2[i], y2[i], xy[i]))
        .collect()
}

fn ssim_from(mu1: f64, mu2: f64, x2: f64, y2: f64, xy: f64) -> f64 {
    let s1 = x2 - mu1 * mu1;
    let s2 = y2 - mu2 * mu2;
    let s12 = xy - mu1 * mu2;
    ((2.0 * mu1 * mu2 + C1) * (2.0 * s12 + C2))
        / ((mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2))
}

/// SSIM at one pixel and channel, direct 11x11 window.
fn ssim_at(x: &[f64], y: &[f64], h: usize, w: usize, r: usize, col: usize, c: usize) -> f64 {
    let g = taps();
    let (mut m1, mut m2, mut a, mut b, mut ab) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, gi) in g.iter().enumerate() {
        let yy = r as isize + i as isize - 5;
        if yy < 0 || yy as usize >= h {
            continue;
        }
        for (j, gj) in g.iter().enumerate() {
            let xx = col as isize + j as isize - 5;
            if xx < 0 || xx as usize >= w {
                continue;
            }
            let k = (yy as usize * w + xx as usize) * 3 + c;
            let wt = gi * gj;
            m1 += wt * x[k];
            m2 += wt * y[k];
            a += wt * x[k] * x[k];
            b += wt * y[k] * y[k];
            ab += wt * x[k] * y[k];
        }
    }
    ssim_from(m1, m2, a, b, ab)
}

/// `d sum(SSIM) / d x[r, col, c]` by a central difference: only the pixels
/// within the window radius of the probe change.
fn ssim_sum_fd(x: &[f64], y: &[f64], h: usize, w: usize, r: usize, col: usize, c: usize) -> f64 {
    let eps = 1e-5;
    let i = (r * w + col) * 3 + c;
    let local = |xv: &[f64]| {
        let mut s = 0.0;
        for rr in r.saturating_sub(5)..(r + 6).min(h) {
            for cc in col.saturating_sub(5)..(col + 6).min(w) {
                s += ssim_at(xv, y, h, w, rr, cc, c);
            }
        }
        s
    };
    let mut xp = x.to_vec();
    xp[i] += eps;
    let mut xm = x.to_vec();
    xm[i] -= eps;
    (local(&xp) - local(&xm)) / (2.0 * eps)
}

async fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_to_vec()
        .expect("vec")
}

fn gt_tensor(gt: &[f64], h: usize, w: usize, device: &Device) -> Tensor<2, Int> {
    let packed: Vec<i32> = gt
        .chunks_exact(3)
        .map(|p| {
            let b = |v: f64| (v * 255.0).round() as u32;
            (b(p[0]) | b(p[1]) << 8 | b(p[2]) << 16 | 255 << 24) as i32
        })
        .collect();
    Tensor::from_data(TensorData::new(packed, [h, w]), device)
}

async fn check_size(h: usize, w: usize) {
    let device =
        burn::tensor::Device::from(brush_cube::test_helpers::test_device().await).autodiff();
    // GT is quantised to u8 by the packing; the reference sees the same bytes.
    let gt: Vec<f64> = scene(h, w, 3)
        .into_iter()
        .map(|v| (v * 255.0).round() / 255.0)
        .collect();
    // pred: GT shifted one pixel, contrast-reduced, plus its own texture, so
    // SSIM sits mid-range as in early training.
    let tex = scene(h, w, 11);
    let pred: Vec<f32> = (0..h * w * 3)
        .map(|i| {
            let px = i / 3;
            let (r, col) = (px / w, px % w);
            let src = (r * w + (col + 1).min(w - 1)) * 3 + i % 3;
            (0.7 * gt[src] + 0.3 * tex[i]) as f32
        })
        .collect();
    let pred64: Vec<f64> = pred.iter().map(|v| f64::from(*v)).collect();
    let gt_t = gt_tensor(&gt, h, w, &device);
    let cfg = ImageLossConfig {
        l1_weight: 0.0,
        ssim_weight: 1.0,
        composite_bg: None,
        mask: false,
        alpha_weight: 0.0,
    };

    // Forward: the per-pixel map, and the reduced training total.
    let reference = ssim_map(&pred64, &gt, h, w);
    let pred_t = Tensor::<1>::from_floats(pred.as_slice(), &device).reshape([h, w, 3]);
    let map = to_vec(image_loss_eval(pred_t.clone(), gt_t.clone(), cfg)).await;
    let max_err = map
        .iter()
        .zip(&reference)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0, f64::max);
    let ref_mean = reference.iter().sum::<f64>() / reference.len() as f64;
    assert!(
        max_err < 1e-3,
        "{h}x{w}: SSIM map differs from reference by {max_err} (ref mean {ref_mean})"
    );

    let pred_g = pred_t.require_grad();
    let loss = image_loss(pred_g.clone(), gt_t, cfg);
    let total = to_vec(loss.clone()).await[0];
    assert!(
        (f64::from(total) - ref_mean).abs() < 1e-4,
        "{h}x{w}: mean SSIM {total} vs reference {ref_mean}"
    );

    // Backward: d mean(SSIM) / d pred, scaled by the element count so the
    // compared values are O(1).
    let grads = loss.backward();
    let grad = to_vec(pred_g.grad(&grads).expect("grad")).await;
    let n = (h * w * 3) as f64;
    let probes = [
        (0, 0, 0),
        (0, w - 1, 1),
        (h - 1, 0, 2),
        (h - 1, w - 1, 0),
        (0, w / 2, 1),
        (h / 2, 0, 2),
        (2, 3, 0),
        (7, 8, 1),
        (15, 16, 2),
        (16, 15, 0),
        (h / 2, w / 2, 1),
        (h - 3, w - 6, 2),
    ];
    let mut failed = Vec::new();
    for (r, col, c) in probes {
        let numerical = ssim_sum_fd(&pred64, &gt, h, w, r, col, c);
        let analytical = f64::from(grad[(r * w + col) * 3 + c]) * n;
        let tol = 2e-3 + 0.01 * numerical.abs().max(analytical.abs());
        if (numerical - analytical).abs() > tol {
            failed.push(format!(
                "({r},{col}) c{c}: reference {numerical:.5} vs kernel {analytical:.5}"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "{h}x{w}: SSIM gradient mismatches:\n  {}",
        failed.join("\n  ")
    );
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_matches_reference_small() {
    check_size(45, 61).await;
}

#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_matches_reference_dycheck_size() {
    check_size(360, 480).await;
}

/// Above `FUSED_SSIM_MAX_PIXELS`: the recomputing backward.
#[wasm_bindgen_test(unsupported = tokio::test)]
async fn ssim_matches_reference_large() {
    check_size(2048, 2049).await;
}

/// `l1_dssim(w)` is the training objective `(1 - w)·L1 - w·SSIM`.
#[test]
fn l1_dssim_subtracts_ssim() {
    let cfg = ImageLossConfig::l1_dssim(0.2);
    assert_eq!((cfg.l1_weight, cfg.ssim_weight), (0.8, -0.2));
    assert!(cfg.composite_bg.is_none() && !cfg.mask && cfg.alpha_weight == 0.0);
}
