// Derived from Brush (Apache-2.0), crates/brush-train/src/adam_scaled.rs
// at 1388f74c6fe0236f68ee4915564bf00e9d2e3747.
//! Brush Adam with block-scaled 8-bit SH first moments and f16 SH parameters.
//!
//! SH updates use f32 arithmetic and stochastic rounding to f16 storage.
//! Reduced SH variance and geometry parameters remain f32. The fused 3DGS
//! update stores geometry/opacity moments packed, one u32 per value
//! ([`brush_render::bwd::geo_moments`]); the unfused path keeps f32 moments.

use crate::quant::{BlockMoment, MomentUpdate, assign_rows};
use brush_render::bwd::geo_moments;

use burn::tensor::{DType, Device, ElementConversion, FloatDType, Int, IntDType, Tensor};

/// Adam with per-parameter second-moment reduction (via [`AdamState::reduce_moment_2`])
/// and per-component learning-rate scaling (via [`AdamState::scaling`]).
///
/// Hand-rolled rather than built on burn's `Adam`, which has neither the
/// per-component LR scaling nor the second-moment reduction below. The
/// trainer also edits momentum tensors in place during refine (split/prune)
/// and swaps the transforms LR scaling every step; burn's low-level
/// `Optimizer` trait does now hand back live `Tensor` state rather than only
/// serialized records, so that part is no longer a blocker — the two missing
/// behaviours are what keep this separate.
#[derive(Clone)]
pub(crate) struct AdamScaled {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

/// Per-parameter momentum state. When `reduce_moment_2` is set on the owning
/// [`AdamState`], the second moment has size 1 in trailing dims;
/// [`AdamState::map_momentum`] callers must stay shape-agnostic along those.
#[derive(Clone)]
pub(crate) struct MomentumState<const D: usize> {
    moments: Moments<D>,
    pub time: usize,
}

#[derive(Clone)]
enum Moments<const D: usize> {
    /// First moment plus an f32, possibly row-reduced, second moment (SH,
    /// and the unfused geometry path).
    Split {
        first: StoredMoment<D>,
        second: Tensor<D>,
    },
    /// Fused geometry/opacity: both moments packed in one u32 word per
    /// value, encoded with Adam's `epsilon`. A zero word is `m = v = 0`.
    Packed { words: Tensor<D, Int>, epsilon: f32 },
}

impl<const D: usize> Moments<D> {
    /// Full f32 `(m, v)`.
    fn decode(self) -> (Tensor<D>, Tensor<D>) {
        match self {
            Self::Split { first, second } => (first.decode(), second),
            Self::Packed { words, epsilon } => geo_moments::decode(words, epsilon),
        }
    }
}

/// Compressed state uses a fused f32 update; refinement decodes explicitly.
/// SH parameters use f16; gradients and update arithmetic remain f32.
#[derive(Clone)]
enum StoredMoment<const D: usize> {
    Full(Tensor<D>),
    Block8(BlockMoment<D>),
}

impl<const D: usize> StoredMoment<D> {
    fn decode(self) -> Tensor<D> {
        match self {
            Self::Full(tensor) => tensor,
            Self::Block8(moment) => moment.decode(),
        }
    }

    fn map(self, f: impl FnOnce(Tensor<D>) -> Tensor<D>) -> Self {
        match self {
            Self::Full(tensor) => Self::Full(f(tensor)),
            Self::Block8(moment) => Self::Block8(BlockMoment::encode(f(moment.decode()))),
        }
    }
}

/// Per-parameter optimizer state.
#[derive(Clone)]
pub(crate) struct AdamState<const D: usize> {
    pub momentum: Option<MomentumState<D>>,
    /// Per-component learning rate scaling (e.g. different LR for means vs
    /// rotations vs scales within the transforms tensor).
    pub scaling: Option<Tensor<D>>,
    /// When true, the second moment is reduced to a scalar per row. Set by the
    /// caller when initializing state for parameters where per-element variance
    /// is not needed.
    pub reduce_moment_2: bool,
}

impl<const D: usize> AdamState<D> {
    pub fn new(scaling: Option<Tensor<D>>, reduce_moment_2: bool) -> Self {
        Self {
            momentum: None,
            scaling,
            reduce_moment_2,
        }
    }

    /// Packed moments for the fused step (zeros before the first step) and
    /// the step count that update makes; f32 moments are packed first.
    /// Pair with [`Self::put_packed`]; `epsilon` is Adam's.
    pub fn take_packed(
        &mut self,
        shape: [usize; D],
        device: &Device,
        epsilon: f32,
    ) -> (Tensor<D, Int>, usize) {
        assert!(!self.reduce_moment_2, "fused Adam keeps full second moments");
        match self.momentum.take() {
            Some(MomentumState {
                moments: Moments::Packed { words, epsilon: e },
                time,
            }) => {
                assert_eq!(e, epsilon, "codec epsilon must match Adam's");
                (words, time + 1)
            }
            Some(old) => {
                let (m, v) = old.moments.decode();
                (geo_moments::encode(m, v, epsilon), old.time + 1)
            }
            None => (geo_moments::zeros(shape, device), 1),
        }
    }

    pub fn put_packed(&mut self, words: Tensor<D, Int>, time: usize, epsilon: f32) {
        self.momentum = Some(MomentumState {
            moments: Moments::Packed { words, epsilon },
            time,
        });
    }

    /// Apply `map_fn` to both momentum tensors (no-op before the first step).
    /// `map_fn` must be shape-agnostic along trailing dims since the second
    /// moment may have size-1 trailing dims under `reduce_moment_2`. Packed
    /// moments are decoded, mapped and re-encoded (round-to-nearest, which
    /// is idempotent for unchanged values).
    pub fn map_momentum(&mut self, map_fn: impl Fn(Tensor<D>) -> Tensor<D>) {
        self.momentum = self.momentum.take().map(|mut moment| {
            moment.moments = match moment.moments {
                Moments::Split { first, second } => Moments::Split {
                    first: first.map(&map_fn),
                    second: map_fn(second),
                },
                Moments::Packed { words, epsilon } => {
                    let (m, v) = geo_moments::decode(words, epsilon);
                    Moments::Packed {
                        words: geo_moments::encode(map_fn(m), map_fn(v), epsilon),
                        epsilon,
                    }
                }
            };
            moment
        });
    }

    /// Gather both moments' rows by `perm` (no-op before the first step).
    /// Packed moments are permuted without decoding to f32.
    pub fn permute_rows(&mut self, perm: &Tensor<1, Int>) {
        self.momentum = self.momentum.take().map(|mut moment| {
            moment.moments = match moment.moments {
                Moments::Split { first, second } => Moments::Split {
                    first: match first {
                        StoredMoment::Full(t) => StoredMoment::Full(t.select(0, perm.clone())),
                        StoredMoment::Block8(m) => StoredMoment::Block8(m.permute_rows(perm)),
                    },
                    second: second.select(0, perm.clone()),
                },
                Moments::Packed { words, epsilon } => Moments::Packed {
                    words: words.select(0, perm.clone()),
                    epsilon,
                },
            };
            moment
        });
    }

    /// Refinement: zero both moments of the unique `rows` in place
    /// and append `extra` zero rows (no-op before the first
    /// step). Packed moments are edited without decoding (a zero word is a
    /// zero moment).
    pub fn reset_rows(&mut self, rows: &Tensor<1, Int>, extra: usize) {
        self.momentum = self.momentum.take().map(|mut moment| {
            moment.moments = match moment.moments {
                Moments::Split { first, second } => Moments::Split {
                    first: match first {
                        StoredMoment::Full(t) => StoredMoment::Full(reset_rows(t, rows, extra)),
                        StoredMoment::Block8(m) => StoredMoment::Block8(m.reset_rows(rows, extra)),
                    },
                    second: reset_rows(second, rows, extra),
                },
                Moments::Packed { mut words, epsilon } => {
                    if rows.dims()[0] > 0 {
                        words = geo_moments::zero_rows(words, rows.clone());
                    }
                    if extra > 0 {
                        let mut shape = words.dims();
                        shape[0] = extra;
                        let tail = geo_moments::zeros(shape, &words.device());
                        words = Tensor::cat(vec![words, tail], 0);
                    }
                    Moments::Packed { words, epsilon }
                }
            };
            moment
        });
    }
}

/// Zero the unique `rows` of `t` in place and append
/// `extra` zero rows.
/// `[0, 1, .., n)` built as a `[h, w]` outer sum of two ~sqrt(n) ranges. Fused
/// `arange` lowers to a cumsum whose CubeCL kernel is quadratic: at 2M rows (dense
/// blur-step gradients) a plain arange cost ~290 ms per step.
fn identity_rows(n: usize, device: &Device) -> Tensor<1, Int> {
    let w = ((n as f64).sqrt().ceil() as usize).max(1);
    let h = n.div_ceil(w);
    let hi = Tensor::<1, Int>::arange(0..h as i64, device).mul_scalar(w as i64);
    let lo = Tensor::<1, Int>::arange(0..w as i64, device);
    (hi.unsqueeze_dim::<2>(1) + lo.unsqueeze_dim::<2>(0))
        .reshape([h * w])
        .slice([0..n])
}

fn reset_rows<const D: usize>(mut t: Tensor<D>, rows: &Tensor<1, Int>, extra: usize) -> Tensor<D> {
    let mut shape = t.dims();
    let device = t.device();
    let num_rows = rows.dims()[0];
    if num_rows > 0 {
        shape[0] = num_rows;
        t = assign_rows(t, rows.clone(), Tensor::zeros(shape, &device));
    }
    if extra > 0 {
        shape[0] = extra;
        t = Tensor::cat(vec![t, Tensor::zeros(shape, &device)], 0);
    }
    t
}

impl AdamScaled {
    pub fn new(epsilon: f32) -> Self {
        Self {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon,
        }
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn with_beta_2(mut self, beta_2: f32) -> Self {
        self.beta_2 = beta_2;
        self
    }

    /// `(beta1, beta2, inv1, inv2, epsilon)` for step `time` (1-based).
    pub fn coefficients(&self, time: usize) -> (f32, f32, f32, f32, f32) {
        (
            self.beta_1,
            self.beta_2,
            1.0 / (1.0 - self.beta_1.powi(time as i32)),
            1.0 / (1.0 - self.beta_2.powi(time as i32)),
            self.epsilon,
        )
    }

    /// One Adam step for a single parameter. `tensor` and `grad` live on the
    /// inner (non-autodiff) backend; `state` is updated in place.
    ///
    /// `grad_sq_mean` supplies the reduced second moment when the caller can
    /// produce it more cheaply than squaring `grad`. It must equal what
    /// `mean_trailing_dims(grad * grad)` would give, and is only read when
    /// [`AdamState::reduce_moment_2`] is set.
    ///
    /// With `grad_rows`, `grad` is compact (see [`Self::step_quantized`]).
    pub fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: &Tensor<D>,
        grad_rows: Option<Tensor<1, Int>>,
        grad_sq_mean: Option<Tensor<D>>,
        state: &mut AdamState<D>,
    ) -> Tensor<D> {
        if state.reduce_moment_2 && D > 1 {
            return self.step_quantized(lr, tensor, grad, grad_rows, grad_sq_mean, state, false);
        }
        if let Some(grad_rows) = grad_rows {
            return self.step_compact(lr, tensor, grad, grad_rows, state);
        }
        self.step_full(lr, tensor, grad, grad_sq_mean, state)
    }

    /// Geometry/opacity update from a compact gradient in one fused kernel,
    /// so neither a dense gradient nor any moment temporaries are allocated.
    fn step_compact<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: &Tensor<D>,
        grad_rows: Tensor<1, Int>,
        state: &mut AdamState<D>,
    ) -> Tensor<D> {
        assert!(!state.reduce_moment_2, "compact Adam keeps full second moments");
        let shape = tensor.dims();
        let device = tensor.device();
        let (m, v, time) = match state.momentum.take() {
            Some(old) => {
                let (m, v) = old.moments.decode();
                (m, v, old.time + 1)
            }
            None => (Tensor::zeros(shape, &device), Tensor::zeros(shape, &device), 1),
        };
        let width = shape[1..].iter().product::<usize>();
        let mut scale_shape = shape;
        scale_shape[0] = 1;
        let lr_scale = match &state.scaling {
            Some(s) => s.clone().expand(scale_shape).reshape([width]),
            None => Tensor::<1>::ones([width], &device),
        };
        let (tensor, m, v) = crate::quant::adam_step(
            tensor,
            grad.clone(),
            grad_rows,
            (m, v),
            lr_scale,
            MomentUpdate {
                beta: self.beta_1,
                factor: 1.0 - self.beta_1,
                inv1: 1.0 / (1.0 - self.beta_1.powi(time as i32)),
                inv2: 1.0 / (1.0 - self.beta_2.powi(time as i32)),
                epsilon: self.epsilon,
                lr: lr as f32,
                round_seed: 0,
            },
            self.beta_2,
        );
        state.momentum = Some(MomentumState {
            moments: Moments::Split {
                first: StoredMoment::Full(m),
                second: v,
            },
            time,
        });
        tensor
    }

    /// Geometry update; also used as the f32 reference in the codec regression.
    fn step_full<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: &Tensor<D>,
        grad_sq_mean: Option<Tensor<D>>,
        state: &mut AdamState<D>,
    ) -> Tensor<D> {
        let (grad, momentum) = self.transform(
            grad,
            grad_sq_mean,
            state.momentum.take(),
            state.reduce_moment_2,
        );
        state.momentum = Some(momentum);

        let delta = if let Some(scale) = &state.scaling {
            grad * (scale.clone() * lr).unsqueeze()
        } else {
            grad * lr
        };
        tensor - delta
    }

    /// Update SH params and packed first moments together; variance remains a
    /// small f32 tensor with one value per splat.
    ///
    /// `grad_rows` maps each splat to its row of a compact `grad` (the
    /// renderer's layout, row 0 zero for culled splats); the reduced variance
    /// must then be supplied. Without it `grad` is dense.
    /// SH step from the fused update's compact `[rows, 6]` colour gradient +
    /// view direction (splat `r` reads row `grad_rows[r]`): the coefficient
    /// gradient `Y_k(dir) * colour` is rebuilt inside the moment kernel, so
    /// the `[rows, coeffs, 3]` gradient is never materialized.
    pub fn step_sh_color(
        &self,
        lr: f64,
        tensor: Tensor<3>,
        color_dir: &Tensor<2>,
        grad_rows: Tensor<1, Int>,
        grad_sq_mean: Tensor<3>,
        state: &mut AdamState<3>,
    ) -> Tensor<3> {
        assert!(state.reduce_moment_2, "SH keeps the reduced second moment");
        let [rows, width] = color_dir.dims();
        assert_eq!(width, 6);
        // Shape-only stand-in for the generic path; the kernel reads it as
        // `[rows, 6]`.
        let grad = color_dir.clone().reshape([rows, 2, 3]);
        self.step_quantized(lr, tensor, &grad, Some(grad_rows), Some(grad_sq_mean), state, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_quantized<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: &Tensor<D>,
        grad_rows: Option<Tensor<1, Int>>,
        grad_sq_mean: Option<Tensor<D>>,
        state: &mut AdamState<D>,
        color_dir: bool,
    ) -> Tensor<D> {
        // Canonical half storage begins with the first SH update. Renderer and
        // backward accept half coefficients but produce f32 gradients.
        // splat patch (wasm only): a browser may not expose `shader-f16` (Chrome on
        // Linux/NVIDIA doesn't); keep f32 there. Every kernel below has an f32 variant.
        #[cfg(target_family = "wasm")]
        let half = tensor.device().supports_dtype(DType::F16);
        #[cfg(not(target_family = "wasm"))]
        let half = true;
        let tensor = if half && tensor.dtype() != DType::F16 {
            tensor.cast(FloatDType::F16)
        } else {
            tensor
        };
        let shape = tensor.dims();
        let grad_rows = grad_rows.unwrap_or_else(|| {
            identity_rows(shape[0], &tensor.device()).cast(IntDType::U32)
        });
        let grad_sq = grad_sq_mean.unwrap_or_else(|| {
            assert_eq!(grad.dims(), shape, "compact gradients need a supplied variance");
            mean_trailing_dims(grad.clone().powi_scalar(2))
        });
        let (moment_1, moment_2, time) = match state.momentum.take() {
            Some(old) => {
                let Moments::Split { first, second } = old.moments else {
                    unreachable!("SH moments are never packed");
                };
                let first = match first {
                    StoredMoment::Block8(first) => first,
                    StoredMoment::Full(first) => BlockMoment::encode(first),
                };
                let stepped = second.clone() * self.beta_2 + grad_sq * (1.0 - self.beta_2);
                // Lazy colour path: culled splats (gradient row 0) keep their
                // variance, matching the moment kernel's skip.
                let second = if color_dir {
                    let culled = grad_rows.clone().equal_elem(0).reshape(second.dims());
                    stepped.mask_where(culled, second)
                } else {
                    stepped
                };
                (first, second, old.time + 1)
            }
            None => (
                BlockMoment::zeros(shape, &tensor.device()),
                grad_sq * (1.0 - self.beta_2),
                1,
            ),
        };
        let mut scale_shape = shape;
        scale_shape[0] = 1;
        let width: usize = scale_shape.iter().product();
        let scale = match &state.scaling {
            Some(s) => s.clone().expand(scale_shape).reshape([width]),
            None => Tensor::<1>::ones([width], &tensor.device()),
        };
        let (tensor, first) = moment_1.step(
            tensor,
            grad.clone(),
            grad_rows,
            moment_2.clone(),
            scale,
            MomentUpdate {
                beta: self.beta_1,
                factor: 1.0 - self.beta_1,
                inv1: 1.0 / (1.0 - self.beta_1.powi(time as i32)),
                inv2: 1.0 / (1.0 - self.beta_2.powi(time as i32)),
                epsilon: self.epsilon,
                lr: lr as f32,
                round_seed: time as u32,
            },
            color_dir,
        );
        state.momentum = Some(MomentumState {
            moments: Moments::Split {
                first: StoredMoment::Block8(first),
                second: moment_2,
            },
            time,
        });
        tensor
    }

    fn transform<const D: usize>(
        &self,
        grad: &Tensor<D>,
        grad_sq_mean: Option<Tensor<D>>,
        momentum_state: Option<MomentumState<D>>,
        reduce_moment_2: bool,
    ) -> (Tensor<D>, MomentumState<D>) {
        let grad_sq_for_moment = if reduce_moment_2 && D > 1 {
            grad_sq_mean.unwrap_or_else(|| mean_trailing_dims(grad.clone().powi_scalar(2)))
        } else {
            grad.clone().powi_scalar(2)
        };

        let (moment_1, moment_2, time) = if let Some(state) = momentum_state {
            let (m, v) = state.moments.decode();
            (
                m.mul_scalar(self.beta_1)
                    .add(grad.clone().mul_scalar(1.0 - self.beta_1)),
                v.mul_scalar(self.beta_2)
                    .add(grad_sq_for_moment.mul_scalar(1.0 - self.beta_2)),
                state.time + 1,
            )
        } else {
            (
                grad.clone().mul_scalar(1.0 - self.beta_1),
                grad_sq_for_moment.mul_scalar(1.0 - self.beta_2),
                1,
            )
        };
        let state = MomentumState {
            moments: Moments::Split {
                first: StoredMoment::Full(moment_1.clone()),
                second: moment_2.clone(),
            },
            time,
        };

        let time = (state.time as i32).elem();
        let moment_1_corrected = moment_1.mul_scalar(1f32 / (1f32 - self.beta_1.powi(time)));
        let moment_2_corrected = moment_2.mul_scalar(1f32 / (1f32 - self.beta_2.powi(time)));
        // moment_2_corrected broadcasts when it has reduced trailing dims
        let grad = moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));
        (grad, state)
    }
}

/// Reduce to a single mean per row by averaging across all trailing dims (1..D).
/// Result has size 1 in each trailing dim so it broadcasts back to the full shape.
fn mean_trailing_dims<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    debug_assert!(D > 1, "mean_trailing_dims requires D > 1");
    let shape = t.dims();
    let trailing_count: usize = shape[1..].iter().product();

    // Reduce over the trailing dims directly; a flatten + reshape would end
    // burn's fusion block.
    let dims: Vec<usize> = (1..D).collect();
    t.sum_dims(&dims) / trailing_count as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData, s};

    #[tokio::test]
    #[ignore = "GPU test: run under the shared exclusive GPU flock"]
    async fn quantized_moment_tracks_float_updates_and_survives_refine() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let full_adam = AdamScaled {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-15,
        };
        let quant_adam = full_adam.clone();
        const ROWS: usize = 13;
        const COEFFS: usize = 16;
        const WIDTH: usize = COEFFS * 3;
        const ELEMENTS: usize = ROWS * WIDTH;
        let mut full = Tensor::<3>::zeros([ROWS, COEFFS, 3], &device);
        let mut quant = full.clone();
        let snapshot = quant.clone();
        let mut rates = vec![0.05f32; COEFFS];
        rates[0] = 1.0;
        let scale = Tensor::<1>::from_floats(rates.as_slice(), &device).reshape([1, COEFFS, 1]);
        let mut fs = AdamState::new(Some(scale.clone()), true);
        let mut qs = AdamState::new(Some(scale), true);
        let mut state_snapshot = None;
        for step in 0..200 {
            if step == 1 {
                let saved = qs.momentum.as_ref().unwrap().moments.clone();
                let values = saved.clone().decode().0.into_data_async().await.unwrap();
                state_snapshot = Some((saved, values));
            }
            let gradient: Vec<f32> = (0..ELEMENTS)
                .map(|i| {
                    if i >= ELEMENTS - WIDTH {
                        0.0
                    } else {
                        ((i as f32 * 0.37 + step as f32 * 0.03).sin()) * 0.001
                    }
                })
                .collect();
            let reduced: Vec<f32> = gradient
                .chunks_exact(WIDTH)
                .map(|row| row.iter().map(|x| x * x).sum::<f32>() / WIDTH as f32)
                .collect();
            let grad = Tensor::from_data(TensorData::new(gradient, [ROWS, COEFFS, 3]), &device);
            // Production supplies the renderer's pre-reduced variance; also
            // exercise the fallback on alternating steps.
            let supplied = (step % 2 == 0)
                .then(|| Tensor::from_data(TensorData::new(reduced, [ROWS, 1, 1]), &device));
            full = full_adam.step_full(0.001, full, &grad, None, &mut fs);
            quant = quant_adam.step(0.001, quant, &grad, None, supplied, &mut qs);
        }
        let (saved_state, expected_state) = state_snapshot.unwrap();
        assert_eq!(
            saved_state.decode().0.into_data_async().await.unwrap(),
            expected_state
        );
        let saved = snapshot
            .into_data_async()
            .await
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            saved.iter().all(|x| *x == 0.0),
            "update mutated a shared snapshot"
        );
        let f = full
            .into_data_async()
            .await
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let q = quant
            .cast(FloatDType::F32)
            .into_data_async()
            .await
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let max_error = f
            .iter()
            .zip(&q)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 0.003, "quantization drift {max_error}");
        assert!(q.iter().all(|x| x.is_finite()));
        assert!(q[ELEMENTS - WIDTH..].iter().all(|x| *x == 0.0));
        // Prune, then append a child with zero momentum. Both full m and
        // reduced v must preserve row count but retain their trailing shapes.
        qs.map_momentum(|t| {
            let t = t.slice(s![0..5, .., ..]);
            let mut shape = t.dims();
            shape[0] = 1;
            Tensor::cat(vec![t, Tensor::zeros(shape, &device)], 0)
        });
        let (m, v) = qs.momentum.unwrap().moments.decode();
        assert_eq!(m.dims(), [6, COEFFS, 3]);
        assert_eq!(v.dims(), [6, 1, 1]);
        let child = m
            .slice(s![5..6, .., ..])
            .into_data_async()
            .await
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(child.iter().all(|x| *x == 0.0));
    }
}
