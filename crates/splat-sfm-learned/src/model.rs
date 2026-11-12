//! DA3-BASE camera-pose path in Burn: the DINOv2 ViT-B/14 backbone as Depth Anything 3 runs it
//! (local/global alternating attention from `ALT_START`, QK-norm and 2D RoPE from layer 4,
//! "saddle_balanced" reference-view selection, camera tokens) followed by the camera decoder.
//!
//! Ported from depth_anything_3/model/{dinov2/vision_transformer.py, dinov2/layers/*.py,
//! reference_view_selector.py, cam_dec.py} at 3d835ec1 (Apache-2.0, see NOTICE). Only the
//! last layer's camera token feeds the decoder, so no other layer outputs are kept, and that
//! (global) layer runs with the camera tokens as its only queries.

use std::path::Path;

use anyhow::{Context, ensure};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::store::{FloatCastAdapter, KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::activation::{gelu, relu};
use burn::tensor::{DType, Device, Int, Tensor, TensorData, s};

pub(crate) const PATCH: usize = 14;
const DIM: usize = 768;
const HEADS: usize = 12;
const HEAD_DIM: usize = DIM / HEADS;
const DEPTH: usize = 12;
const MLP_DIM: usize = 4 * DIM;
/// First layer with alternating attention, camera tokens, QK-norm and RoPE (da3-base.yaml).
const ALT_START: usize = 4;
const ROPE_FREQ: f64 = 100.0;
/// `pos_embed` was trained for 518x518 inputs: a 37x37 patch grid.
const POS_GRID: usize = 37;
/// Largest activation buffer to create: wgpu limits single buffers (2 GiB here).
const CHUNK_BYTES: usize = 512 << 20;
/// `THRESH_FOR_REF_SELECTION`: reference-view reordering needs at least this many views.
const REF_SELECTION_MIN_VIEWS: usize = 3;

#[derive(Module, Debug)]
struct Attention {
    qkv: Linear,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    proj: Linear,
}

#[derive(Module, Debug)]
struct LayerScale {
    gamma: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

#[derive(Module, Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

#[derive(Module, Debug)]
struct PatchEmbed {
    proj: Conv2d,
}

#[derive(Module, Debug)]
struct Backbone {
    patch_embed: PatchEmbed,
    cls_token: Param<Tensor<3>>,
    camera_token: Param<Tensor<3>>,
    pos_embed: Param<Tensor<3>>,
    blocks: Vec<Block>,
}

#[derive(Module, Debug)]
struct CamDec {
    fc1: Linear,
    fc2: Linear,
    fc_t: Linear,
    fc_qvec: Linear,
    fc_fov: Linear,
}

/// DA3-BASE backbone + camera decoder.
#[derive(Module, Debug)]
pub struct Da3Pose {
    backbone: Backbone,
    cam_dec: CamDec,
}

/// Per-token RoPE tables `[tokens, HEAD_DIM]`, broadcast over batch and heads.
struct Rope {
    cos: Tensor<4>,
    sin: Tensor<4>,
}

impl Rope {
    /// `positions` are (y, x) per token. The head dim splits into a vertical and a horizontal
    /// half; each half is a 1D rotary embedding over 16 frequencies, laid out `[f, f]`.
    fn new(positions: &[(usize, usize)], device: &Device) -> Self {
        let half = HEAD_DIM / 2;
        let freqs: Vec<f32> = (0..half / 2)
            .map(|j| (1.0 / ROPE_FREQ.powf((2 * j) as f64 / half as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(positions.len() * HEAD_DIM);
        let mut sin = Vec::with_capacity(positions.len() * HEAD_DIM);
        for &(y, x) in positions {
            for p in [y, x] {
                for _ in 0..2 {
                    for f in &freqs {
                        let a = p as f32 * f;
                        cos.push(a.cos());
                        sin.push(a.sin());
                    }
                }
            }
        }
        let shape = [1, 1, positions.len(), HEAD_DIM];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
        }
    }

    fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        // rotate_half within each 32-wide axis half: [a, b, c, d] -> [-b, a, -d, c].
        let q = HEAD_DIM / 4;
        let part = |i: usize| x.clone().slice(s![.., .., .., i * q..(i + 1) * q]);
        let rotated = Tensor::cat(vec![part(1).neg(), part(0), part(3).neg(), part(2)], 3);
        x * self.cos.clone() + rotated * self.sin.clone()
    }
}

impl Attention {
    /// Per-head `[b, HEADS, t, HEAD_DIM]` queries, keys and values for `x` (already normed),
    /// with QK-norm and RoPE applied.
    fn qkv_heads(&self, x: Tensor<3>, rope: Option<&Rope>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [b, t, _] = x.dims();
        let qkv = self.qkv.forward(x);
        let heads = |i: usize| {
            qkv.clone()
                .slice(s![.., .., i * DIM..(i + 1) * DIM])
                .reshape([b, t, HEADS, HEAD_DIM])
                .swap_dims(1, 2)
        };
        let (mut q, mut k, v) = (heads(0), heads(1), heads(2));
        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = qn.forward(q);
            k = kn.forward(k);
        }
        if let Some(rope) = rope {
            q = rope.apply(q);
            k = rope.apply(k);
        }
        (q, k, v)
    }
}

impl LayerScale {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        x * self.gamma.val().reshape([1, 1, DIM])
    }
}

impl Block {
    /// Self-attention block over `x` `[b, t, DIM]`.
    fn forward(&self, x: Tensor<3>, rope: Option<&Rope>) -> Tensor<3> {
        let (q, k, v) = self.attn.qkv_heads(self.norm1.forward(x.clone()), rope);
        self.finish(x, scaled_attention(q, k, v))
    }

    /// Global attention over the tokens of all `chunks` (`[1, t_c, DIM]` each, with their RoPE
    /// tables), computed chunk by chunk so no buffer exceeds `CHUNK_BYTES`: keys and values are
    /// gathered per group of heads, then every chunk of queries attends over all of them. With
    /// `queries`, only those tokens are computed and returned (as the single output).
    fn forward_global(&self, chunks: Vec<(Tensor<3>, &Rope)>, queries: Option<(Tensor<3>, &Rope)>) -> Vec<Tensor<3>> {
        let total: usize = chunks.iter().map(|(x, _)| x.dims()[1]).sum();
        let group = (CHUNK_BYTES / (total * HEAD_DIM * 4)).clamp(1, HEADS);
        let groups: Vec<_> = (0..HEADS).step_by(group).map(|h| h..(h + group).min(HEADS)).collect();
        let (mut ks, mut vs) = (vec![Vec::new(); groups.len()], vec![Vec::new(); groups.len()]);
        let mut qs = Vec::new();
        for (x, rope) in &chunks {
            let (q, k, v) = self.attn.qkv_heads(self.norm1.forward(x.clone()), Some(*rope));
            for (g, heads) in groups.iter().enumerate() {
                ks[g].push(k.clone().slice(s![.., heads.clone()]));
                vs[g].push(v.clone().slice(s![.., heads.clone()]));
            }
            if queries.is_none() {
                qs.push(q);
            }
        }
        let ks: Vec<Tensor<4>> = ks.into_iter().map(|k| Tensor::cat(k, 2)).collect();
        let vs: Vec<Tensor<4>> = vs.into_iter().map(|v| Tensor::cat(v, 2)).collect();
        let attend = |q: Tensor<4>| {
            let parts = groups
                .iter()
                .enumerate()
                .map(|(g, heads)| {
                    let q = q.clone().slice(s![.., heads.clone()]);
                    scaled_attention(q, ks[g].clone(), vs[g].clone())
                })
                .collect();
            Tensor::cat(parts, 1)
        };
        match queries {
            Some((xq, rope)) => {
                let (q, _, _) = self.attn.qkv_heads(self.norm1.forward(xq.clone()), Some(rope));
                vec![self.finish(xq, attend(q))]
            }
            None => chunks.into_iter().zip(qs).map(|((x, _), q)| self.finish(x, attend(q))).collect(),
        }
    }

    /// Residual attention output (`o` `[b, HEADS, t, HEAD_DIM]`) and the MLP for the tokens `x`.
    fn finish(&self, x: Tensor<3>, o: Tensor<4>) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let attn = self.attn.proj.forward(o.swap_dims(1, 2).reshape([b, t, DIM]));
        let x = x + self.ls1.forward(attn);
        let h = self.mlp.fc2.forward(gelu(self.mlp.fc1.forward(self.norm2.forward(x.clone()))));
        x + self.ls2.forward(h)
    }
}

impl Da3Pose {
    fn init(device: &Device) -> Self {
        let linear = |i, o| LinearConfig::new(i, o).init(device);
        let norm = |d| LayerNormConfig::new(d).with_epsilon(1e-6).init(device);
        let blocks = (0..DEPTH)
            .map(|i| {
                // q/k norms use nn.LayerNorm's default eps (1e-5); the block norms use 1e-6.
                let qk_norm = || (i >= ALT_START).then(|| LayerNormConfig::new(HEAD_DIM).init(device));
                Block {
                    norm1: norm(DIM),
                    attn: Attention {
                        qkv: linear(DIM, 3 * DIM),
                        q_norm: qk_norm(),
                        k_norm: qk_norm(),
                        proj: linear(DIM, DIM),
                    },
                    ls1: LayerScale {
                        gamma: Param::from_tensor(Tensor::ones([DIM], device)),
                    },
                    norm2: norm(DIM),
                    mlp: Mlp {
                        fc1: linear(DIM, MLP_DIM),
                        fc2: linear(MLP_DIM, DIM),
                    },
                    ls2: LayerScale {
                        gamma: Param::from_tensor(Tensor::ones([DIM], device)),
                    },
                }
            })
            .collect();
        let zeros = |shape: [usize; 3]| Param::from_tensor(Tensor::zeros(shape, device));
        Self {
            backbone: Backbone {
                patch_embed: PatchEmbed {
                    proj: Conv2dConfig::new([3, DIM], [PATCH, PATCH])
                        .with_stride([PATCH, PATCH])
                        .init(device),
                },
                cls_token: zeros([1, 1, DIM]),
                camera_token: zeros([1, 2, DIM]),
                pos_embed: zeros([1, 1 + POS_GRID * POS_GRID, DIM]),
                blocks,
            },
            cam_dec: CamDec {
                fc1: linear(2 * DIM, 2 * DIM),
                fc2: linear(2 * DIM, 2 * DIM),
                fc_t: linear(2 * DIM, 3),
                fc_qvec: linear(2 * DIM, 4),
                fc_fov: linear(2 * DIM, 2),
            },
        }
    }

    /// Load the backbone and camera decoder from a DA3-BASE safetensors file (the upstream
    /// `model.safetensors`, or a reduced one holding just these tensors, f32 or f16).
    pub fn load(path: &Path, device: &Device) -> anyhow::Result<Self> {
        let remap = KeyRemapper::from_patterns(vec![
            (r"^model\.", ""),
            (r"^backbone\.pretrained\.", "backbone."),
            (r"^cam_dec\.backbone\.0\.", "cam_dec.fc1."),
            (r"^cam_dec\.backbone\.2\.", "cam_dec.fc2."),
            (r"^cam_dec\.fc_fov\.0\.", "cam_dec.fc_fov."),
        ])?;
        let mut store = SafetensorsStore::from_file(path)
            // fp16 weight files are widened to the f32 model on load.
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .remap(remap)
            .map_indices_contiguous(false);
        let mut model = Self::init(device);
        let result = model
            .load_from(&mut store)
            .map_err(|e| anyhow::anyhow!("loading DA3 weights from {}: {e}", path.display()))?;
        ensure!(
            result.errors.is_empty() && result.missing.is_empty(),
            "DA3 weights {}: errors {:?}, missing {:?}",
            path.display(),
            result.errors,
            result.missing
        );
        Ok(model)
    }

    /// `images` `[S, 3, H, W]` row-major on the host (ImageNet-normalised, H and W multiples
    /// of 14) to the camera decoder input `[S, 2*DIM]` (local and global camera token of the
    /// last layer) and the pose encoding `[S, 9]` (t xyz, quat xyzw, fov_h, fov_w), both in
    /// input order.
    pub fn forward(&self, images: &[f32], views: usize, h: usize, w: usize, device: &Device) -> anyhow::Result<(Tensor<2>, Tensor<2>)> {
        let feat = self.backbone.forward(images, views, h, w, device)?;
        Ok((feat.clone(), self.cam_dec.forward(feat)))
    }
}

impl Backbone {
    /// The views run in chunks of frames sized so that the largest per-chunk activation (the
    /// MLP hidden layer) stays within `CHUNK_BYTES`: wgpu caps the size of a single buffer, and
    /// a few hundred views of 721 tokens exceed it. Global attention gathers keys and values
    /// across the chunks (see [`Block::forward_global`]).
    fn forward(&self, images: &[f32], views: usize, h: usize, w: usize, device: &Device) -> anyhow::Result<Tensor<2>> {
        let (gh, gw) = (h / PATCH, w / PATCH);
        let n = 1 + gh * gw;
        let per = (CHUNK_BYTES / (n * MLP_DIM * 4)).max(1);
        let ranges: Vec<_> = (0..views).step_by(per).map(|a| a..(a + per).min(views)).collect();

        let pos = self.pos_embed_for(gh, gw)?;
        let image_len = 3 * h * w;
        let mut xs: Vec<Tensor<3>> = ranges
            .iter()
            .map(|r| {
                let data = images[r.start * image_len..r.end * image_len].to_vec();
                let img = Tensor::<4>::from_data(TensorData::new(data, [r.len(), 3, h, w]), device);
                let patches = self.patch_embed.proj.forward(img).reshape([r.len(), DIM, gh * gw]).swap_dims(1, 2);
                let cls = self.cls_token.val().expand([r.len(), 1, DIM]);
                Tensor::cat(vec![cls, patches], 1) + pos.clone()
            })
            .collect();

        // Local attention sees per-view grid positions offset by one, the camera/cls token sits
        // at (0, 0); global attention gives every patch (1, 1).
        let local_pos: Vec<_> = std::iter::once((0, 0))
            .chain((0..gh * gw).map(|k| (k / gw + 1, k % gw + 1)))
            .collect();
        let local_rope = Rope::new(&local_pos, device);
        let global_rope = |frames: usize| {
            let pos: Vec<_> = (0..frames)
                .flat_map(|_| std::iter::once((0, 0)).chain(std::iter::repeat_n((1, 1), gh * gw)))
                .collect();
            Rope::new(&pos, device)
        };
        let mut global_ropes = std::collections::BTreeMap::new();
        for r in &ranges {
            global_ropes.entry(r.len()).or_insert_with(|| global_rope(r.len()));
        }
        let cam_rope = Rope::new(&vec![(0, 0); views], device);
        let cams = |xs: &[Tensor<3>]| Tensor::cat(xs.iter().map(|x| x.clone().slice(s![.., 0..1])).collect(), 0);

        let mut ref_view = None;
        let mut local_cam = cams(&xs);
        for (i, block) in self.blocks.iter().enumerate() {
            if i == ALT_START - 1 && views >= REF_SELECTION_MIN_VIEWS {
                let cls = cams(&xs).reshape([views, DIM]).into_data().try_into_vec::<f32>()?;
                let r = saddle_balanced(&cls, views);
                let order = reference_first_order(r, views);
                xs = ranges
                    .iter()
                    .map(|range| {
                        let frames = order[range.clone()].iter().map(|&v| xs[v / per].clone().slice(s![v % per..v % per + 1]));
                        Tensor::cat(frames.collect(), 0)
                    })
                    .collect();
                ref_view = Some(r);
            }
            if i == ALT_START {
                // The reference view (now first) gets camera token 0, every other view token 1.
                let tokens = self.camera_token.val();
                xs = xs
                    .into_iter()
                    .zip(&ranges)
                    .map(|(x, r)| {
                        let mut cam = Vec::new();
                        if r.start == 0 {
                            cam.push(tokens.clone().slice(s![.., 0..1]));
                        }
                        let others = r.len() - usize::from(r.start == 0);
                        if others > 0 {
                            cam.push(tokens.clone().slice(s![.., 1..2]).expand([others, 1, DIM]));
                        }
                        Tensor::cat(vec![Tensor::cat(cam, 0), x.slice(s![.., 1..])], 1)
                    })
                    .collect();
            }
            let global = i >= ALT_START && i % 2 == 1;
            if global {
                let chunks: Vec<_> = xs
                    .iter()
                    .zip(&ranges)
                    .map(|(x, r)| (x.clone().reshape([1, r.len() * n, DIM]), &global_ropes[&r.len()]))
                    .collect();
                if i == DEPTH - 1 {
                    // Last layer: only the camera tokens feed the decoder.
                    let q = cams(&xs).reshape([1, views, DIM]);
                    let out = block.forward_global(chunks, Some((q, &cam_rope)));
                    xs = vec![out.into_iter().next().expect("one output").reshape([views, 1, DIM])];
                } else {
                    let out = block.forward_global(chunks, None);
                    xs = out.into_iter().zip(&ranges).map(|(x, r)| x.reshape([r.len(), n, DIM])).collect();
                }
            } else {
                let rope = (i >= ALT_START).then_some(&local_rope);
                xs = xs.into_iter().map(|x| block.forward(x, rope)).collect();
                local_cam = cams(&xs);
            }
        }

        let out = Tensor::cat(vec![local_cam, cams(&xs)], 2).reshape([views, 2 * DIM]);
        Ok(match ref_view {
            Some(r) => out.select(0, index(&restore_order(r, views), device)),
            None => out,
        })
    }

    /// `pos_embed` for a `gh x gw` grid: DINOv2's bicubic resize of the 37x37 table, including
    /// its `+0.1` scale-factor offset (`interpolate_pos_encoding`), computed on the host.
    fn pos_embed_for(&self, gh: usize, gw: usize) -> anyhow::Result<Tensor<3>> {
        let table = self.pos_embed.val();
        let device = table.device();
        if gh == POS_GRID && gw == POS_GRID {
            return Ok(table);
        }
        let data = table.into_data().try_into_vec::<f32>()?;
        let (cls, grid) = data.split_at(DIM);
        let ry = resize_weights(POS_GRID, gh, (gh as f64 + 0.1) / POS_GRID as f64);
        let rx = resize_weights(POS_GRID, gw, (gw as f64 + 0.1) / POS_GRID as f64);
        let mut out = cls.to_vec();
        out.resize(DIM * (1 + gh * gw), 0.0);
        for (oy, wy) in ry.iter().enumerate() {
            for (ox, wx) in rx.iter().enumerate() {
                let dst = &mut out[DIM * (1 + oy * gw + ox)..][..DIM];
                for &(sy, ay) in wy {
                    for &(sx, ax) in wx {
                        let a = (ay * ax) as f32;
                        let src = &grid[DIM * (sy * POS_GRID + sx)..][..DIM];
                        for (d, v) in dst.iter_mut().zip(src) {
                            *d += a * v;
                        }
                    }
                }
            }
        }
        Ok(Tensor::from_data(TensorData::new(out, [1, 1 + gh * gw, DIM]), &device))
    }
}

impl CamDec {
    fn forward(&self, feat: Tensor<2>) -> Tensor<2> {
        let h = relu(self.fc2.forward(relu(self.fc1.forward(feat))));
        let t = self.fc_t.forward(h.clone());
        let q = self.fc_qvec.forward(h.clone());
        let fov = relu(self.fc_fov.forward(h));
        Tensor::cat(vec![t, q, fov], 1)
    }
}

/// PyTorch bicubic (A = -0.75, align_corners=False, edge-clamped) taps per output index when
/// `interpolate` is given a scale factor: `src = (dst + 0.5) / scale - 0.5`.
fn resize_weights(src: usize, dst: usize, scale: f64) -> Vec<Vec<(usize, f64)>> {
    (0..dst)
        .map(|o| {
            let f = (o as f64 + 0.5) / scale - 0.5;
            let i = f.floor();
            let t = f - i;
            cubic_coeffs(t)
                .into_iter()
                .enumerate()
                .map(|(k, c)| ((i as isize - 1 + k as isize).clamp(0, src as isize - 1) as usize, c))
                .collect()
        })
        .collect()
}

pub(crate) fn cubic_coeffs(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let near = |x: f64| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let far = |x: f64| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
}

/// "saddle_balanced": the view whose cls token is closest to the middle of the range on mean
/// similarity to the others, feature norm and feature variance.
fn saddle_balanced(cls: &[f32], views: usize) -> usize {
    let rows: Vec<&[f32]> = cls.chunks(DIM).collect();
    let norms: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt())
        .collect();
    let unit: Vec<Vec<f64>> = rows
        .iter()
        .zip(&norms)
        .map(|(r, n)| r.iter().map(|&v| f64::from(v) / n).collect())
        .collect();
    let sim: Vec<f64> = unit
        .iter()
        .map(|a| {
            let total: f64 = unit.iter().map(|b| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>()).sum();
            (total - 1.0) / (views - 1) as f64
        })
        .collect();
    let var: Vec<f64> = unit
        .iter()
        .map(|u| {
            let mean = u.iter().sum::<f64>() / DIM as f64;
            u.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (DIM - 1) as f64
        })
        .collect();
    let normalise = |m: &[f64]| -> Vec<f64> {
        let lo = m.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = m.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        m.iter().map(|v| (v - lo) / (hi - lo + 1e-8)).collect()
    };
    let (sim, norms, var) = (normalise(&sim), normalise(&norms), normalise(&var));
    let score = |i: usize| (sim[i] - 0.5).abs() + (norms[i] - 0.5).abs() + (var[i] - 0.5).abs();
    (1..views).fold(0, |best, i| if score(i) < score(best) { i } else { best })
}

/// `reorder_by_reference`: position 0 takes the reference view, the rest keep their order.
fn reference_first_order(r: usize, views: usize) -> Vec<usize> {
    std::iter::once(r).chain((0..views).filter(|&v| v != r)).collect()
}

/// `restore_original_order`: inverse of [`reference_first_order`].
fn restore_order(r: usize, views: usize) -> Vec<usize> {
    (0..views)
        .map(|v| match v.cmp(&r) {
            std::cmp::Ordering::Less => v + 1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => v,
        })
        .collect()
}

fn index(order: &[usize], device: &Device) -> Tensor<1, Int> {
    let order: Vec<i64> = order.iter().map(|&v| v as i64).collect();
    let len = order.len();
    Tensor::from_data(TensorData::new(order, [len]), device)
}

/// The weights file named by `SPLAT_DA3_WEIGHTS`: normally `da3-base-cam-f16.safetensors`, the
/// backbone + camera-decoder tensors of DA3-BASE `model.safetensors` cast to fp16 (183 MB).
/// `explicit` if given (a job spec names its weights so remote staging ships them), else `SPLAT_DA3_WEIGHTS`.
pub(crate) fn weights_path(explicit: Option<&std::path::Path>) -> anyhow::Result<std::path::PathBuf> {
    let path = match explicit {
        Some(p) => p.to_path_buf(),
        None => std::path::PathBuf::from(std::env::var_os("SPLAT_DA3_WEIGHTS").context(
            "SPLAT_DA3_WEIGHTS must point at the DA3-BASE weights (da3-base-cam-f16.safetensors)",
        )?),
    };
    ensure!(path.is_file(), "SPLAT_DA3_WEIGHTS={} is not a file", path.display());
    Ok(path)
}

/// `softmax(q kᵀ / sqrt(d)) v` over `[b, heads, seq, d]`, via the fused kernel in
/// [`crate::flash`] (exact, O(seq) memory; Burn's autotuned attention only has
/// its slow scalar flash kernel on WGSL).
fn scaled_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    crate::flash::attention(q, k, v)
}
