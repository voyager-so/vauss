//! Automatic distractor-mask gate (`--auto-masks`). Detector masks (person
//! masks from `mask_distractors`, alpha mode Masked) are ignored at first:
//! the run trains unmasked through a probe window, comparing the residual on
//! the pixels the masks cover with the residual elsewhere. Static content a
//! detector mislabels (a statue read as a person) is consistent across views
//! and fits like the rest of the scene, so the ratio stays near 1 and the
//! masks are never used: the run is the unmasked run. People who move never
//! agree across views and keep the ratio high; then the masks switch on for
//! the rest of the run.

use burn::tensor::{Int, Tensor, s};

/// Inside/outside residual ratio above which the masks are kept.
const RATIO: f64 = 2.0;
/// Probe window as fractions of the run: at 30k steps 3000..4500, the
/// half-resolution phase, once coarse geometry exists.
const WINDOW: (f64, f64) = (0.10, 0.15);
/// The masks cover less than this share of the probed pixels: nothing to
/// decide, keep them (they cost nothing).
const MIN_COVERAGE: f64 = 1e-4;
/// Opaque alpha byte of the packed RGBA GT.
const ALPHA_OPAQUE: i32 = 0xff00_0000_u32 as i32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Probe,
    On,
    Off,
}

pub(crate) struct MaskGate {
    start: u32,
    end: u32,
    state: State,
    /// Device-side running sums over the window: (Σ e·m, Σ m, Σ e·a, Σ a),
    /// `e` the per-pixel mean RGB L1, `a` the GT alpha, `m = 1 − a`.
    acc: Option<Tensor<1>>,
}

impl MaskGate {
    pub fn new(total_steps: u32) -> Self {
        let (a, b) = WINDOW;
        let at = |f: f64| ((f * f64::from(total_steps)) as u32).max(1);
        let gate = Self {
            start: at(a),
            end: at(b).max(at(a) + 1),
            state: State::Probe,
            acc: None,
        };
        let line = format!(
            "mask gate: masks off, probing steps {}..{} (masks on if inside/outside residual > {})",
            gate.start, gate.end, RATIO
        );
        log::info!("{line}");
        eprintln!("{line}");
        gate
    }

    fn sums(pred: &Tensor<3>, gt_packed: &Tensor<2, Int>) -> Tensor<1> {
        let [h, w, _] = pred.dims();
        let gt = brush_loss::unpack_gt_rgb(gt_packed.clone(), None);
        let err = (pred.clone().slice(s![.., .., 0..3]) - gt).abs().mean_dim(2);
        let alpha = gt_packed
            .clone()
            .bitwise_right_shift_scalar(24)
            .bitwise_and_scalar(0xff)
            .float()
            .reshape([h, w, 1])
            / 255.0;
        let masked = alpha.clone().neg().add_scalar(1.0);
        Tensor::cat(
            vec![
                (err.clone() * masked.clone()).sum(),
                masked.sum(),
                (err * alpha.clone()).sum(),
                alpha.sum(),
            ],
            0,
        )
    }

    /// Accumulate this step's residuals. `pred` is the detached `[H, W, C]`
    /// render, `gt_packed` the GT whose alpha is the detector mask.
    pub fn record(&mut self, step: u32, pred: &Tensor<3>, gt_packed: &Tensor<2, Int>) {
        if self.state != State::Probe || !(self.start..self.end).contains(&step) {
            return;
        }
        let sums = Self::sums(pred, gt_packed);
        self.acc = Some(match self.acc.take() {
            Some(a) => a + sums,
            None => sums,
        });
    }

    /// Decide at the end of the window (one readback).
    pub async fn maybe_decide(&mut self, step: u32) {
        if self.state != State::Probe || step < self.end {
            return;
        }
        let Some(acc) = self.acc.take() else {
            // No masked view in the window: keep the masks, nothing measured.
            self.state = State::On;
            return;
        };
        let (inside, outside, cover) = stats(&read(acc).await);
        let ratio = inside / outside.max(1e-9);
        self.state = if cover < MIN_COVERAGE || ratio > RATIO {
            State::On
        } else {
            State::Off
        };
        let line = format!(
            "mask gate: step {step} inside {inside:.4} outside {outside:.4} ratio {ratio:.3} coverage {cover:.4} -> masks {}",
            if self.state == State::On { "ON" } else { "OFF (what they hide is static)" }
        );
        log::info!("{line}");
        eprintln!("{line}");
    }

    pub fn masks_on(&self) -> bool {
        self.state == State::On
    }

    /// The GT the loss sees: the masks only once the gate is On. The probe
    /// trains unmasked, so a run whose masks end Off is the unmasked run.
    pub fn apply(&self, gt_packed: Tensor<2, Int>) -> Tensor<2, Int> {
        if self.state != State::On {
            gt_packed.bitwise_or_scalar(ALPHA_OPAQUE)
        } else {
            gt_packed
        }
    }
}

async fn read(t: Tensor<1>) -> Vec<f32> {
    t.into_data_async()
        .await
        .expect("mask gate readback")
        .try_into_vec()
        .expect("mask gate f32 sums")
}

/// (mean residual inside the masks, outside, masked share of pixels).
fn stats(v: &[f32]) -> (f64, f64, f64) {
    let [em, m, ea, a] = [v[0], v[1], v[2], v[3]].map(f64::from);
    (em / m.max(1.0), ea / a.max(1.0), m / (m + a).max(1.0))
}
