//! Quantisation-aware tail. Over the last twentieth of training each step
//! renders the splats as the SPZ export stores them (`splat_webexport::spz`,
//! same lattice): SH DC 8-bit, higher SH at the export's 6 bits, log scales at
//! 1/16, opacity 8-bit, positions at the fixed point the export will pick
//! (fixed when the tail starts). Gradients w.r.t. those quantised values update
//! the float params (straight-through), which first get projected into the
//! representable range so clipping can't absorb unbounded drift; the tail's
//! learning rates decay 10x to settle them into their codes. The last step
//! snaps the params onto the lattice, so eval and export see exactly the
//! delivered model. Rotations keep full precision (their 10-bit smallest-three
//! codes measured no loss).

#[path = "qat_kernels.rs"]
mod kernels;

use brush_render::gaussian_splats::Splats;
use burn::backend::Dispatch;
use burn::tensor::{Int, Tensor};
use kernels::QatOps;

/// The last `1 / TAIL_FRACTION` of `total_train_iters` trains quantisation-aware.
pub(crate) const TAIL_FRACTION: u32 = 20;

const SCALE_STEPS: f32 = 16.0;
const SCALE_MIN: f32 = -10.0;
const SCALE_MAX: f32 = SCALE_MIN + 255.0 / SCALE_STEPS;

/// SPZ positions are signed 24-bit fixed point with at most 12 fractional bits.
const POS_FRACTIONAL_BITS: i32 = 12;
const POS_MAX: f32 = 8_388_607.0;

/// The export's units and position fixed point, in training units.
pub(crate) struct Lattice {
    /// `ln(units_per_meter)`: exported log scales are shifted by this.
    ln_units: f32,
    /// Position codes per training unit.
    pos_scale: f32,
}

impl Lattice {
    /// The finest position fixed point that holds every current mean, like
    /// `spz::position_fractional_bits` on the export.
    pub(crate) async fn new(splats: &Splats, units_per_meter: f32) -> Self {
        let max = splats
            .means()
            .abs()
            .max()
            .into_data_async()
            .await
            .expect("Failed to read splat extent")
            .try_to_vec::<f32>()
            .expect("Failed to read splat extent")[0]
            * units_per_meter;
        let bits = (0..=POS_FRACTIONAL_BITS)
            .rev()
            .find(|&b| (max * 2f32.powi(b)).round() <= POS_MAX)
            .unwrap_or(0);
        Self {
            ln_units: units_per_meter.ln(),
            pos_scale: 2f32.powi(bits) * units_per_meter,
        }
    }
}

/// Raw opacities at the export's 8-bit alpha.
fn opacities(raw: Tensor<1>) -> Tensor<1> {
    Tensor::from_dispatch(Dispatch::qat_opacities(raw.into_dispatch()))
}

/// What [`restore`] adds back to stepped quantised params: each param's
/// offset from its quantised value as codes of half its lattice step,
/// 16-bit for the f32 transforms and 8-bit for the f16 SH (finer than half
/// precision there anyway); opacity logits, unbounded, in f32.
pub(crate) struct Residuals {
    transforms: Tensor<1, Int>,
    sh: Tensor<1, Int>,
    opacities: Tensor<1>,
    pos_scale: f32,
}

/// The tail step's params: each replaced by its quantised value (what the
/// step renders and steps), its offset from it kept as [`Residuals`] for
/// [`restore`]. Straight-through without a second full-size copy: the float
/// params are never held next to their quantised values. Codes round
/// stochastically, so offsets finer than a code survive on average.
pub(crate) fn quantise(mut splats: Splats, lattice: &Lattice, seed: u32) -> (Splats, Residuals) {
    let ln_units = lattice.ln_units;
    let mut t_codes = None;
    splats.transforms = splats.transforms.map(|t| {
        let (quant, codes) = Dispatch::qat_transforms_split(
            t.inner().into_dispatch(),
            lattice.pos_scale,
            POS_MAX / lattice.pos_scale,
            SCALE_MIN - ln_units,
            SCALE_MAX - ln_units,
            ln_units - SCALE_MIN,
            SCALE_MIN - ln_units,
            seed,
        );
        t_codes = Some(Tensor::from_dispatch(codes));
        Tensor::from_inner(Tensor::from_dispatch(quant)).require_grad()
    });
    let mut sh_codes = None;
    splats.sh_coeffs = splats.sh_coeffs.map(|c| {
        let k = c.dims()[1] as u32;
        let (quant, codes) = Dispatch::qat_sh_split(c.inner().into_dispatch(), k, seed);
        sh_codes = Some(Tensor::from_dispatch(codes));
        Tensor::from_inner(Tensor::from_dispatch(quant)).require_grad()
    });
    let mut o_res = None;
    splats.raw_opacities = splats.raw_opacities.map(|o| {
        let raw = o.inner();
        let quant = opacities(raw.clone());
        o_res = Some(raw - quant.clone());
        Tensor::from_inner(quant).require_grad()
    });
    let residuals = Residuals {
        transforms: t_codes.expect("mapped"),
        sh: sh_codes.expect("mapped"),
        opacities: o_res.expect("mapped"),
        pos_scale: lattice.pos_scale,
    };
    (splats, residuals)
}

/// Stepped quantised params back to float params: each moves by its
/// quantised value's step (straight-through).
pub(crate) fn restore(mut splats: Splats, residuals: Residuals) -> Splats {
    let Residuals { transforms, sh, opacities, pos_scale } = residuals;
    splats.transforms = splats.transforms.map(|t| {
        let joined = Dispatch::qat_transforms_join(
            t.inner().into_dispatch(),
            transforms.into_dispatch(),
            pos_scale,
        );
        Tensor::from_inner(Tensor::from_dispatch(joined)).require_grad()
    });
    splats.sh_coeffs = splats.sh_coeffs.map(|t| {
        let k = t.dims()[1] as u32;
        let joined = Dispatch::qat_sh_join(t.inner().into_dispatch(), sh.into_dispatch(), k);
        Tensor::from_inner(Tensor::from_dispatch(joined)).require_grad()
    });
    splats.raw_opacities = splats
        .raw_opacities
        .map(|t| Tensor::from_inner(t.inner() + opacities).require_grad());
    splats
}

/// Replace the params with their quantised values: [`quantise`] with its
/// offset codes dropped. Its passes overwrite the params in place, where a
/// full quantised copy of the SH and transforms would be fresh pool pages at
/// the end of training (+175 MiB board on room-1.2M, the run's peak).
pub(crate) fn snap(splats: Splats, lattice: &Lattice) -> Splats {
    quantise(splats, lattice, 0).0
}
