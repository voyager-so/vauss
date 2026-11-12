//! The fused blur blend against the tensor-op implementation it replaced.
//! Run on any device; on this host via lavapipe:
//! `cargo test -p splat-train blur_tests -- --ignored`.

use super::{GAMMA, blend, blend_cached};
use crate::blur_kernels;
use crate::mixer::Mixer;
use burn::tensor::{Device, Distribution, Tensor, s};

// ------------------------------------------------- reference (pre-fusion)

/// The tensor-op blend that `blend` replaced, with the mixer abstracted to
/// `mix` (linear sub-renders `[K, 3, sh, sw]` to weights and mask).
#[allow(clippy::type_complexity)]
fn ref_blend(
    mut renders: Vec<Tensor<3>>,
    mix: Option<&dyn Fn(Tensor<4>) -> (Tensor<4>, Tensor<4>)>,
) -> (Tensor<3>, Option<Tensor<1>>, Tensor<4>, Tensor<4>) {
    let middle = renders.pop().expect("the sharp render");
    let count = renders.len();
    let [h, w, channels] = middle.dims();
    let linear = |img: &Tensor<3>| {
        img.clone()
            .slice(s![.., .., 0..3])
            .clamp_min(1e-8)
            .powf_scalar(GAMMA)
    };
    let subs = Tensor::stack::<4>(renders.iter().map(linear).collect(), 0).permute([0, 3, 1, 2]);
    let sharp = linear(&middle).permute([2, 0, 1]).reshape([1, 3, h, w]);
    let up = |t: Tensor<4>| ref_upsample2(t, h, w);
    let (blurred, mask) = if let Some(mix) = mix {
        let (weights, mask) = mix(subs.clone().detach());
        ((weights * subs).sum_dim(0), mask)
    } else {
        let [_, _, sh, sw] = subs.dims();
        let m = count as f32 / (count + 1) as f32;
        (
            subs.mean_dim(0),
            Tensor::full([1, 1, sh, sw], m, &sharp.device()),
        )
    };
    let cache = (blurred.clone().detach(), mask.clone().detach());
    let mask_mean = mix.is_some().then(|| mask.clone().mean());
    let (blurred, mask) = (up(blurred), up(mask));
    let rgb = sharp * (mask.clone().neg() + 1.0) + blurred * mask;
    let rgb = rgb
        .reshape([3, h, w])
        .permute([1, 2, 0])
        .powf_scalar(1.0 / GAMMA);
    let rgb = if channels > 3 {
        let alpha = renders
            .into_iter()
            .map(|img| img.slice(s![.., .., 3..channels]))
            .reduce(|a, b| a + b)
            .expect("at least one sub-render")
            / count as f32;
        let alpha = up(alpha.permute([2, 0, 1]).unsqueeze_dim(0))
            .squeeze_dim::<3>(0)
            .permute([1, 2, 0]);
        Tensor::cat(vec![rgb, alpha], 2)
    } else {
        rgb
    };
    (rgb, mask_mean, cache.0, cache.1)
}

fn ref_blend_cached(sharp: Tensor<3>, blurred: Tensor<4>, mask: Tensor<4>) -> Tensor<3> {
    let [h, w, channels] = sharp.dims();
    let lin = sharp
        .clone()
        .slice(s![.., .., 0..3])
        .clamp_min(1e-8)
        .powf_scalar(GAMMA)
        .permute([2, 0, 1])
        .reshape([1, 3, h, w]);
    let grow = |t: Tensor<4>| {
        let mut t = t.cast(burn::tensor::FloatDType::F32);
        while t.dims()[2] * 2 < h.div_ceil(2) * 2 && t.dims()[2] * 2 <= h {
            let [_, _, th, tw] = t.dims();
            t = ref_upsample2(t, th * 2, tw * 2);
        }
        ref_upsample2(t, h, w)
    };
    let (blurred, mask) = (grow(blurred), grow(mask));
    let rgb = (lin * (mask.clone().neg() + 1.0) + blurred * mask)
        .reshape([3, h, w])
        .permute([1, 2, 0])
        .powf_scalar(1.0 / GAMMA);
    if channels > 3 {
        Tensor::cat(vec![rgb, sharp.slice(s![.., .., 3..channels])], 2)
    } else {
        rgb
    }
}

fn ref_upsample2(t: Tensor<4>, h: usize, w: usize) -> Tensor<4> {
    fn axis(t: Tensor<4>, dim: usize) -> Tensor<4> {
        let n = t.dims()[dim];
        let prev = Tensor::cat(
            vec![t.clone().narrow(dim, 0, 1), t.clone().narrow(dim, 0, n - 1)],
            dim,
        );
        let next = Tensor::cat(
            vec![
                t.clone().narrow(dim, 1, n - 1),
                t.clone().narrow(dim, n - 1, 1),
            ],
            dim,
        );
        let even = t.clone() * 0.75 + prev * 0.25;
        let odd = t * 0.75 + next * 0.25;
        let mut dims = even.dims();
        dims[dim] *= 2;
        Tensor::stack::<5>(vec![even, odd], dim + 1).reshape(dims)
    }
    let [_, _, sh, sw] = t.dims();
    let t = if sh > 1 {
        axis(t, 2)
    } else {
        t.repeat_dim(2, 2)
    };
    let t = if sw > 1 {
        axis(t, 3)
    } else {
        t.repeat_dim(3, 2)
    };
    t.narrow(2, 0, h).narrow(3, 0, w)
}

// ------------------------------------------------------------------ helpers

async fn device() -> Device {
    Device::from(brush_cube::test_helpers::test_device().await).autodiff()
}

async fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("readback")
        .try_to_vec::<f32>()
        .expect("f32")
}

fn uniform<const D: usize>(shape: [usize; D], lo: f64, hi: f64, device: &Device) -> Tensor<D> {
    Tensor::random(shape, Distribution::Uniform(lo, hi), device)
}

/// A render with values in [0, 1] and a few exact zeros and negatives (the
/// `1e-8` clamp, gradient and all).
fn render(shape: [usize; 3], device: &Device) -> Tensor<3> {
    let t = uniform(shape, 0.02, 1.0, device);
    let pick = uniform(shape, 0.0, 1.0, device);
    let t = t.mask_fill(pick.clone().lower_elem(0.03), 0.0);
    t.mask_fill(pick.greater_elem(0.98), -0.1)
}

/// Worst error of `got` against `want`, relative to `max(|want|, floor *
/// max|want|)`, so near-zero entries are judged against the tensor's scale.
async fn compare<const D: usize>(what: &str, got: Tensor<D>, want: Tensor<D>, tol: f32) -> f32 {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let (got, want) = (values(got).await, values(want).await);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let floor = (scale * 1e-2).max(1e-30);
    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite at {i}");
        let err = (g - w).abs() / w.abs().max(floor);
        assert!(
            err <= tol,
            "{what}: element {i}: fused {g} vs reference {w} (rel {err:e} > {tol:e})"
        );
        worst = worst.max(err);
    }
    println!("{what:<44} n={:<6} max rel err {worst:.2e}", got.len());
    worst
}

const FWD_TOL: f32 = 1e-5;
const GRAD_TOL: f32 = 1e-4;

// -------------------------------------------------------------------- tests

#[tokio::test]
#[ignore = "GPU test: run under lavapipe or the shared GPU flock"]
async fn upsample2_matches_tensor_ops() {
    let device = device().await;
    // (n, c, sh, sw) -> (h, w): odd and even crops, one-sample axes.
    let cases = [
        ([2, 3, 17, 23], [33, 45]),
        ([1, 4, 17, 23], [34, 46]),
        ([1, 2, 8, 8], [16, 16]),
        ([1, 2, 1, 5], [1, 9]),
        ([1, 2, 1, 5], [2, 10]),
        ([1, 3, 6, 1], [11, 2]),
        ([1, 1, 1, 1], [1, 1]),
        ([1, 1, 1, 1], [2, 2]),
        ([1, 1, 2, 2], [3, 3]),
    ];
    for ([n, c, sh, sw], [h, w]) in cases {
        let t = uniform([n, c, sh, sw], -1.0, 1.0, &device);
        let rw = uniform([n, c, h, w], -1.0, 1.0, &device);
        let a = t.clone().require_grad();
        let b = t.require_grad();
        let fused = blur_kernels::upsample2(a.clone(), h, w);
        let reference = ref_upsample2(b.clone(), h, w);
        let name = format!("upsample2 {:?}->{:?}", [n, c, sh, sw], [h, w]);
        compare(
            &format!("{name} fwd"),
            fused.clone(),
            reference.clone(),
            FWD_TOL,
        )
        .await;
        let ga = (fused * rw.clone()).sum().backward();
        let gb = (reference * rw).sum().backward();
        compare(
            &format!("{name} grad"),
            a.grad(&ga).unwrap(),
            b.grad(&gb).unwrap(),
            GRAD_TOL,
        )
        .await;
    }
}

/// Inputs of one blend case, as leaves.
struct Case {
    sharp: Tensor<3>,
    subs: Vec<Tensor<3>>,
    weights: Tensor<4>,
    mask: Tensor<4>,
    rw: Tensor<3>,
}

fn case(k: usize, c: usize, [h, w]: [usize; 2], device: &Device) -> Case {
    let (sh, sw) = (h.div_ceil(2), w.div_ceil(2));
    Case {
        sharp: render([h, w, c], device),
        subs: (0..k).map(|_| render([sh, sw, c], device)).collect(),
        weights: uniform([k, 1, sh, sw], 0.0, 1.0, device),
        mask: uniform([1, 1, sh, sw], 0.05, 0.95, device),
        rw: uniform([h, w, c], -1.0, 1.0, device),
    }
}

/// Leaves with gradients: `(sharp, subs, weights, mask)`.
fn leaves(case: &Case) -> (Tensor<3>, Vec<Tensor<3>>, Tensor<4>, Tensor<4>) {
    (
        case.sharp.clone().detach().require_grad(),
        case.subs
            .iter()
            .map(|t| t.clone().detach().require_grad())
            .collect(),
        case.weights.clone().detach().require_grad(),
        case.mask.clone().detach().require_grad(),
    )
}

/// Full sizes (half sizes `ceil(size / 2)`): odd, even, and small. The
/// tensor-op reference gets 4-channel alpha wrong at 5x3 (checked against a
/// host computation, which the fused blend matches), so the small case is
/// 9x7.
const SIZES: [[usize; 2]; 3] = [[33, 45], [34, 46], [9, 7]];

#[tokio::test]
#[ignore = "GPU test: run under lavapipe or the shared GPU flock"]
async fn weighted_blend_matches_tensor_ops() {
    let device = device().await;
    for k in [3, 9] {
        for c in [3, 4] {
            for size in SIZES {
                let case = case(k, c, size, &device);
                let name = format!("weighted K={k} C={c} {size:?}");

                let (sharp, subs, weights, mask) = leaves(&case);
                let (fused, half) = blur_kernels::blend(
                    sharp.clone(),
                    Tensor::stack::<4>(subs.clone(), 0),
                    Some(weights.clone()),
                    mask.clone(),
                );
                let ga = (fused.clone() * case.rw.clone()).sum().backward();

                let (rsharp, rsubs, rweights, rmask) = leaves(&case);
                let mix = |_: Tensor<4>| (rweights.clone(), rmask.clone());
                let mut renders = rsubs.clone();
                renders.push(rsharp.clone());
                let (reference, _, blurred, _) = ref_blend(renders, Some(&mix));
                let gb = (reference.clone() * case.rw.clone()).sum().backward();

                compare(&format!("{name} fwd"), fused, reference, FWD_TOL).await;
                compare(
                    &format!("{name} blur cache"),
                    half.slice(s![.., 0..3]),
                    blurred,
                    FWD_TOL,
                )
                .await;
                compare(
                    &format!("{name} d sharp"),
                    sharp.grad(&ga).unwrap(),
                    rsharp.grad(&gb).unwrap(),
                    GRAD_TOL,
                )
                .await;
                for (i, (a, b)) in subs.iter().zip(&rsubs).enumerate() {
                    compare(
                        &format!("{name} d sub {i}"),
                        a.grad(&ga).unwrap(),
                        b.grad(&gb).unwrap(),
                        GRAD_TOL,
                    )
                    .await;
                }
                compare(
                    &format!("{name} d weights"),
                    weights.grad(&ga).unwrap(),
                    rweights.grad(&gb).unwrap(),
                    GRAD_TOL,
                )
                .await;
                compare(
                    &format!("{name} d mask"),
                    mask.grad(&ga).unwrap(),
                    rmask.grad(&gb).unwrap(),
                    GRAD_TOL,
                )
                .await;
            }
        }
    }
}

#[tokio::test]
#[ignore = "GPU test: run under lavapipe or the shared GPU flock"]
async fn mean_blend_matches_tensor_ops() {
    let device = device().await;
    for k in [3, 9] {
        for c in [3, 4] {
            for size in SIZES {
                let case = case(k, c, size, &device);
                let name = format!("mean K={k} C={c} {size:?}");

                let (sharp, subs, _, _) = leaves(&case);
                let mut renders = subs.clone();
                renders.push(sharp.clone());
                let (fused, mask_mean, (blurred, mask)) = blend(renders, None);
                assert!(mask_mean.is_none());
                let ga = (fused.clone() * case.rw.clone()).sum().backward();

                let (rsharp, rsubs, _, _) = leaves(&case);
                let mut renders = rsubs.clone();
                renders.push(rsharp.clone());
                let (reference, _, rblurred, rmask) = ref_blend(renders, None);
                let gb = (reference.clone() * case.rw.clone()).sum().backward();

                compare(&format!("{name} fwd"), fused, reference, FWD_TOL).await;
                // The cache is f16.
                let f32 = burn::tensor::FloatDType::F32;
                compare(
                    &format!("{name} blur cache (f16)"),
                    blurred.cast(f32),
                    rblurred,
                    1e-3,
                )
                .await;
                compare(
                    &format!("{name} mask cache (f16)"),
                    mask.cast(f32),
                    rmask,
                    1e-3,
                )
                .await;
                compare(
                    &format!("{name} d sharp"),
                    sharp.grad(&ga).unwrap(),
                    rsharp.grad(&gb).unwrap(),
                    GRAD_TOL,
                )
                .await;
                for (i, (a, b)) in subs.iter().zip(&rsubs).enumerate() {
                    compare(
                        &format!("{name} d sub {i}"),
                        a.grad(&ga).unwrap(),
                        b.grad(&gb).unwrap(),
                        GRAD_TOL,
                    )
                    .await;
                }
            }
        }
    }
}

/// `blend` itself with a real mixer, the sampled sub-render's gradient
/// scaled as in training (gradients reach the mixer through its outputs).
#[tokio::test]
#[ignore = "GPU test: run under lavapipe or the shared GPU flock"]
async fn mixer_blend_matches_tensor_ops() {
    let device = device().await;
    for (k, c, size) in [(9, 4, [33, 45]), (3, 3, [34, 46])] {
        let case = case(k, c, size, &device);
        let name = format!("mixer K={k} C={c} {size:?}");
        let mixer = Mixer::new(k, &device);
        let sampled = k / 2;

        let (sharp, subs, _, _) = leaves(&case);
        let mut renders: Vec<_> = subs.iter().map(|t| t.clone().detach()).collect();
        renders[sampled] = super::scale_grad(subs[sampled].clone(), k as f32);
        renders.push(sharp.clone());
        let (fused, mask_mean, _) = blend(renders, Some(&mixer));
        let mask_mean = mask_mean.expect("mixer mask mean");
        let ga = ((fused.clone() * case.rw.clone()).sum() + mask_mean.clone() * 3.0).backward();

        let (rsharp, rsubs, _, _) = leaves(&case);
        let mut renders: Vec<_> = rsubs.iter().map(|t| t.clone().detach()).collect();
        let fixed = rsubs[sampled].clone().detach();
        renders[sampled] = fixed.clone() + (rsubs[sampled].clone() - fixed) * k as f32;
        renders.push(rsharp.clone());
        let mix = |subs: Tensor<4>| mixer.forward(subs);
        let (reference, rmask_mean, _, _) = ref_blend(renders, Some(&mix));
        let rmask_mean = rmask_mean.expect("mixer mask mean");
        let gb =
            ((reference.clone() * case.rw.clone()).sum() + rmask_mean.clone() * 3.0).backward();

        compare(&format!("{name} fwd"), fused, reference, FWD_TOL).await;
        compare(&format!("{name} mask mean"), mask_mean, rmask_mean, FWD_TOL).await;
        compare(
            &format!("{name} d sharp"),
            sharp.grad(&ga).unwrap(),
            rsharp.grad(&gb).unwrap(),
            GRAD_TOL,
        )
        .await;
        compare(
            &format!("{name} d sampled sub"),
            subs[sampled].grad(&ga).unwrap(),
            rsubs[sampled].grad(&gb).unwrap(),
            GRAD_TOL,
        )
        .await;
        for (i, sub) in subs.iter().enumerate().filter(|&(i, _)| i != sampled) {
            assert!(sub.grad(&ga).is_none(), "sub-render {i} is detached");
        }
        // The mixer's own parameters: its mask head through `weights` and `mask`.
        let p = mixer.mask_head_param();
        compare(
            &format!("{name} d mixer mask head"),
            p.grad(&ga).unwrap(),
            p.grad(&gb).unwrap(),
            GRAD_TOL,
        )
        .await;
        let p = mixer.weight_head_param();
        compare(
            &format!("{name} d mixer weight head"),
            p.grad(&ga).unwrap(),
            p.grad(&gb).unwrap(),
            GRAD_TOL,
        )
        .await;
    }
}

#[tokio::test]
#[ignore = "GPU test: run under lavapipe or the shared GPU flock"]
async fn cached_blend_matches_tensor_ops() {
    use burn::tensor::{TensorData, f16};
    let device = device().await;
    // (full size, cache size): the photo's own half size, and a cache from a
    // lower training resolution that grows first.
    let cases = [
        ([33, 45], [17, 23]),
        ([34, 46], [17, 23]),
        ([33, 45], [5, 6]),
        ([5, 3], [3, 2]),
    ];
    // f16-representable values, as the f16 training cache (fused) and as
    // f32 (reference: fused into its upsample, f16 inputs would run the
    // reference's arithmetic at reduced precision).
    let cache = |shape: [usize; 4], lo: f32, hi: f32, seed: u32| {
        let n: usize = shape.iter().product();
        let v: Vec<f16> = (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8) as f32
                    / 16_777_216.0;
                f16::from_f32(lo + (hi - lo) * x)
            })
            .collect();
        let wide: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
        (
            Tensor::<4>::from_data(
                TensorData::new(v, shape),
                (&device, burn::tensor::DType::F16),
            ),
            Tensor::<4>::from_data(TensorData::new(wide, shape), &device),
        )
    };
    for c in [3, 4] {
        for ([h, w], [ch, cw]) in cases {
            let name = format!("cached C={c} {:?} cache {:?}", [h, w], [ch, cw]);
            let sharp0 = render([h, w, c], &device);
            let (blurred, rblurred) = cache([1, 3, ch, cw], 0.0, 1.0, 17);
            let (mask, rmask) = cache([1, 1, ch, cw], 0.05, 0.95, 91);
            assert_eq!(blurred.dtype(), burn::tensor::DType::F16);
            let rw = uniform([h, w, c], -1.0, 1.0, &device);

            let sharp = sharp0.clone().require_grad();
            let fused = blend_cached(sharp.clone(), blurred, mask);
            let ga = (fused.clone() * rw.clone()).sum().backward();
            let rsharp = sharp0.require_grad();
            let reference = ref_blend_cached(rsharp.clone(), rblurred, rmask);
            let gb = (reference.clone() * rw).sum().backward();

            compare(&format!("{name} fwd"), fused, reference, FWD_TOL).await;
            compare(
                &format!("{name} d sharp"),
                sharp.grad(&ga).unwrap(),
                rsharp.grad(&gb).unwrap(),
                GRAD_TOL,
            )
            .await;
        }
    }
}

#[tokio::test]
#[ignore = "GPU test: run under lavapipe or the shared GPU flock"]
async fn scale_grad_is_identity_with_scaled_gradient() {
    let device = device().await;
    let x0 = uniform([7, 5, 4], -1.0, 1.0, &device);
    let rw = uniform([7, 5, 4], -1.0, 1.0, &device);
    let x = x0.clone().require_grad();
    let y = super::scale_grad(x.clone(), 9.0);
    compare("scale_grad fwd", y.clone(), x0, 0.0).await;
    let grads = (y * rw.clone()).sum().backward();
    compare("scale_grad grad", x.grad(&grads).unwrap(), rw * 9.0, 1e-6).await;
}

/// Kernel launches of one training-like blur blend, forward and backward:
/// `BLUR_COUNT=reference|fused` (`BLUR_COUNT_MEAN` for the plain average)
/// with cubecl's profiling logger on (`[profiling] logger = { level =
/// "minimal", file = ... }` in a cubecl.toml in the working directory). The
/// probe runs `BLUR_COUNT_REPS` times; one warm blend is the log-line
/// difference between 2 and 1 reps (the first pass also autotunes).
#[tokio::test]
#[ignore = "launch-count probe; needs a cubecl.toml profiling logger"]
async fn blur_launch_count() {
    let Ok(mode) = std::env::var("BLUR_COUNT") else {
        return;
    };
    let reps: usize = std::env::var("BLUR_COUNT_REPS").map_or(1, |r| r.parse().unwrap());
    let weighted = std::env::var("BLUR_COUNT_MEAN").is_err();
    let device = device().await;
    let (k, c) = (9, 4);
    let case = case(k, c, [34, 46], &device);
    let sampled = k / 2;
    for _ in 0..reps {
        let (sharp, subs, weights, mask) = leaves(&case);
        let mut renders: Vec<_> = subs.iter().map(|t| t.clone().detach()).collect();
        // Stand-in mixer: both paths build its input; the CNN's own
        // launches are the same either way.
        let mix = |_: Tensor<4>| (weights.clone(), mask.clone());
        let f16 = burn::tensor::FloatDType::F16;
        let (out, mask_mean, blurred, cache_mask) = if mode == "reference" {
            let fixed = subs[sampled].clone().detach();
            renders[sampled] = fixed.clone() + (subs[sampled].clone() - fixed) * k as f32;
            renders.push(sharp.clone());
            let (out, mean, b, m) = ref_blend(renders, weighted.then_some(&mix as _));
            (out, mean, b.cast(f16), m.cast(f16))
        } else {
            renders[sampled] = super::scale_grad(subs[sampled].clone(), k as f32);
            renders.push(sharp.clone());
            if weighted {
                // `blend` with the stand-in mixer.
                let sharp = renders.pop().unwrap();
                let stacked = Tensor::stack::<4>(renders, 0);
                let (w, m) = mix(blur_kernels::linear_subs(stacked.clone().detach()));
                let cm = m.clone().detach().cast(f16);
                let (out, half) = blur_kernels::blend(sharp, stacked, Some(w), m);
                (
                    out,
                    None::<Tensor<1>>,
                    half.slice(s![.., 0..3]).cast(f16),
                    cm,
                )
            } else {
                let (out, mean, (b, m)) = blend(renders, None);
                (out, mean, b, m)
            }
        };
        // The mixer's mask-mean loss is the same reduce either way (and
        // lavapipe's reduce autotune trips on it), so it stays out.
        let _ = mask_mean;
        let loss = (out * case.rw.clone()).sum();
        let grads = loss.backward();
        // Read back without conversion kernels.
        async fn settle<const D: usize>(t: Tensor<D>) {
            t.into_data_async().await.expect("readback");
        }
        settle(blurred).await;
        settle(cache_mask).await;
        settle(sharp.grad(&grads).unwrap()).await;
        settle(subs[sampled].grad(&grads).unwrap()).await;
        if weighted {
            settle(weights.grad(&grads).unwrap()).await;
            settle(mask.grad(&grads).unwrap()).await;
        }
    }
    // The profiling logger writes asynchronously.
    std::thread::sleep(std::time::Duration::from_secs(2));
}
