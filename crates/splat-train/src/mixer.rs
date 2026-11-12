//! Learned per-pixel blend of a photo's exposure sub-renders.
//!
//! Camera motion rarely blurs a photo uniformly, so the blurry image is
//! modelled as `(1 - m) * middle + m * sum_k w_k * I_k`: the sharp render at
//! the photo's own camera, blended per pixel (mask `m`) with a per-pixel
//! weighting `w` (softmax over the sub-renders) of the renders along the
//! camera path. One small CNN shared by every photo predicts `w` and `m` from
//! the sub-renders themselves, detached, so the scene and camera gradients
//! only flow through the blend. The CNN sees the renders box-downsampled to
//! about `MIXER_PIXELS` and its outputs are upsampled, which keeps it a small,
//! resolution-independent fraction of a training step.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{InstanceNorm, InstanceNormConfig, PaddingConfig2d};
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::module::avg_pool2d;
use burn::tensor::{Device, Distribution, Tensor};

const FEATURES: usize = 32;

/// Pixels the CNN works at: a quarter of a 600x400 photo per side.
const MIXER_PIXELS: usize = 15_000;

#[derive(Module, Debug)]
pub struct Mixer {
    conv1: Conv2d,
    norm1: InstanceNorm,
    conv2: Conv2d,
    norm2: InstanceNorm,
    /// 1x1 convolution `FEATURES -> 1`, no bias: per-render blend logit.
    weight_head: Param<Tensor<4>>,
    /// 1x1 convolution over every render's features, `K * FEATURES -> 1`, no
    /// bias, as `[K, FEATURES, 1, 1]`: the blend mask logit.
    mask_head: Param<Tensor<4>>,
}

impl Mixer {
    /// A mixer for `sub_renders` renders per photo.
    pub fn new(sub_renders: usize, device: &Device) -> Self {
        let conv = |c_in: usize, c_out: usize, size: usize| {
            let pad = size / 2;
            Conv2dConfig::new([c_in, c_out], [size, size])
                .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
                .init(device)
        };
        let norm = || {
            InstanceNormConfig::new(FEATURES)
                .with_affine(false)
                .init(device)
        };
        // Kaiming-uniform bound 1/sqrt(fan_in), as a default 1x1 conv.
        let head = |shape: [usize; 4], fan_in: usize| {
            let bound = 1.0 / (fan_in as f64).sqrt();
            Param::from_tensor(Tensor::random(
                shape,
                Distribution::Uniform(-bound, bound),
                device,
            ))
        };
        Self {
            conv1: conv(3, FEATURES, 5),
            norm1: norm(),
            conv2: conv(FEATURES, FEATURES, 3),
            norm2: norm(),
            weight_head: head([1, FEATURES, 1, 1], FEATURES),
            mask_head: head([sub_renders, FEATURES, 1, 1], sub_renders * FEATURES),
        }
    }

    /// Blend weights `[K, 1, H, W]` (softmax over the renders) and mask
    /// `[1, 1, H, W]` for the sub-renders `[K, 3, H, W]` (linear radiance).
    ///
    /// The 1x1 heads are channel sums over the downsampled features, so the
    /// whole batch goes through the CNN in a handful of kernel launches.
    pub fn forward(&self, renders: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let [k, _, h, w] = renders.dims();
        let factor = ((h * w) as f64 / MIXER_PIXELS as f64).sqrt().ceil().max(1.0) as usize;
        let renders = avg_pool2d(
            renders,
            [factor; 2],
            [factor; 2],
            [0, 0],
            true,
            true,
        );
        let [_, _, sh, sw] = renders.dims();
        // Bilinear upsampling as `A_h X A_w^T` (the backend has no bilinear
        // interpolate backward).
        let device = renders.device();
        let rows = linear_resize(h, sh, &device).reshape([1, 1, h, sh]);
        let cols = linear_resize(w, sw, &device).transpose().reshape([1, 1, sw, w]);
        let upsample = |t: Tensor<4>| rows.clone().matmul(t).matmul(cols.clone());
        let features = self.norm1.forward(relu(self.conv1.forward(renders)));
        let features = self.norm2.forward(relu(self.conv2.forward(features)));
        let logits = (features.clone() * self.weight_head.val()).sum_dim(1);
        let mask = (features * self.mask_head.val()).sum_dim(1).sum_dim(0);
        let weights = softmax(upsample(logits), 0);
        let mask = sigmoid(upsample(mask));
        debug_assert_eq!(weights.dims(), [k, 1, h, w]);
        (weights, mask)
    }
}

#[cfg(test)]
impl Mixer {
    pub fn weight_head_param(&self) -> Tensor<4> {
        self.weight_head.val()
    }

    pub fn mask_head_param(&self) -> Tensor<4> {
        self.mask_head.val()
    }
}

/// `[out, size]` matrix of half-pixel-centred linear interpolation weights
/// resizing `size` samples to `out`.
fn linear_resize(out: usize, size: usize, device: &Device) -> Tensor<2> {
    let mut weights = vec![0.0f32; out * size];
    let scale = size as f32 / out as f32;
    for o in 0..out {
        let src = ((o as f32 + 0.5) * scale - 0.5).clamp(0.0, (size - 1) as f32);
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(size - 1);
        let f = src - lo as f32;
        weights[o * size + lo] += 1.0 - f;
        weights[o * size + hi] += f;
    }
    Tensor::<1>::from_floats(weights.as_slice(), device).reshape([out as i32, size as i32])
}
