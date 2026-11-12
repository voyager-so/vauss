//! Logical tensor storage budget, not a prediction of peak device allocation.
//!
//! Brush 1388f74 stores transforms `[N,10]`, opacity `[N]`, SH `[N,K,3]`,
//! full first moments, and one second moment per SH row. Geometry second
//! moments are not reduced. Gradients, raster/sort workspaces, image buffers,
//! allocator caching, and simultaneous old/new tensors during refinement are
//! additional costs. A capacity claim requires a measured training run.

/// Optional allocator snapshots for phase attribution. These are host allocator
/// readings, not synchronized GPU peaks; diagnostic runs must also sample VRAM.
/// The log guard avoids querying the allocator during ordinary training.
pub(crate) fn log_phase(device: &burn::tensor::Device, step: u32, phase: &str, splats: u32) {
    if log::log_enabled!(target: "splat_memory", log::Level::Debug) {
        if let Some(usage) = device.memory_pool_usage() {
            log::debug!(target: "splat_memory",
                "step={step} phase={phase} splats={splats} live={} reserved={} padding={} allocations={}",
                usage.bytes_in_use, usage.bytes_reserved, usage.bytes_padding, usage.number_allocs);
        }
    }
}

/// Persistent storage choices. The compressed layouts are planning estimates
/// until a matching implementation is selected and measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageLayout {
    /// Upstream Brush: all parameter and moment values are f32.
    BrushF32,
    /// SH first moment: signed 8-bit blocks of 128 values with f32 scales.
    /// Reduced SH second moment and geometry moments remain f32.
    ShFirstMoment8,
    /// Same optimizer layout, with canonical 16-bit SH parameters.
    ShFirstMoment8HalfCoefficients,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorMemory {
    pub parameters: u64,
    pub optimizer: u64,
    /// Three f32 refinement accumulators and an optional f32 mip scale floor.
    pub refinement: u64,
    /// Full f32 parameter gradients, excluding raster intermediates.
    pub gradients: u64,
}

impl TensorMemory {
    pub fn persistent_bytes(self) -> u64 {
        self.parameters + self.optimizer + self.refinement
    }

    pub fn with_gradients_bytes(self) -> u64 {
        self.persistent_bytes() + self.gradients
    }
}

/// Compute logical tensor bytes using checked arithmetic. Supports SH0–SH3.
/// `None` means the degree is unsupported or the byte count overflows u64.
/// This intentionally does not return a maximum splat count for a GPU: there
/// is no scene-independent bound on tile intersections or refinement peaks.
pub fn tensor_memory(
    splats: u64,
    sh_degree: u32,
    layout: StorageLayout,
    mip_filter: bool,
) -> Option<TensorMemory> {
    if sh_degree > 3 {
        return None;
    }
    let coefficients = 3 * u64::from(sh_degree + 1).pow(2);
    let sh_elements = splats.checked_mul(coefficients)?;
    let geometry_bytes = splats.checked_mul(11 * 4)?;
    let sh_width = match layout {
        StorageLayout::ShFirstMoment8HalfCoefficients => 2,
        _ => 4,
    };
    let parameters = geometry_bytes.checked_add(sh_elements.checked_mul(sh_width)?)?;
    let sh_first_moment = match layout {
        StorageLayout::BrushF32 => sh_elements.checked_mul(4)?,
        _ => {
            // Packed u32 values and a scale for every (possibly partial) block.
            let packed = sh_elements.div_ceil(128).checked_mul(128)?;
            packed.checked_add(sh_elements.div_ceil(128).checked_mul(4)?)?
        }
    };
    let optimizer = geometry_bytes
        .checked_mul(2)?
        .checked_add(sh_first_moment)?
        .checked_add(splats.checked_mul(4)?)?;
    let refinement = splats.checked_mul(if mip_filter { 16 } else { 12 })?;
    let gradients = geometry_bytes.checked_add(sh_elements.checked_mul(4)?)?;
    let budget = TensorMemory {
        parameters,
        optimizer,
        refinement,
        gradients,
    };
    // Ensure totals can also be represented, not just individual fields.
    parameters
        .checked_add(optimizer)?
        .checked_add(refinement)?
        .checked_add(gradients)?;
    Some(budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_sh3_accounts_for_reduced_variance() {
        let b = tensor_memory(1, 3, StorageLayout::BrushF32, true).unwrap();
        assert_eq!(b.parameters, 59 * 4);
        assert_eq!(b.optimizer, (11 * 2 + 48 + 1) * 4);
        assert_eq!(b.persistent_bytes(), 536);
        assert_eq!(b.with_gradients_bytes(), 772);
    }

    #[test]
    fn ten_million_sh3_budget() {
        let n = 10_000_000;
        let fp32 = tensor_memory(n, 3, StorageLayout::BrushF32, true).unwrap();
        let q8 = tensor_memory(n, 3, StorageLayout::ShFirstMoment8, true).unwrap();
        let half =
            tensor_memory(n, 3, StorageLayout::ShFirstMoment8HalfCoefficients, true).unwrap();
        assert_eq!(fp32.persistent_bytes(), 5_360_000_000);
        assert_eq!(q8.persistent_bytes(), 3_935_000_000);
        assert_eq!(half.persistent_bytes(), 2_975_000_000);
        assert_eq!(half.with_gradients_bytes(), 5_335_000_000);
    }

    #[test]
    fn packing_and_tail_scales_are_counted() {
        let b = tensor_memory(1, 0, StorageLayout::ShFirstMoment8, false).unwrap();
        assert_eq!(b.optimizer, 88 + 128 + 4 + 4);
        assert_eq!(
            tensor_memory(0, 3, StorageLayout::BrushF32, true)
                .unwrap()
                .with_gradients_bytes(),
            0
        );
        assert!(tensor_memory(1, 4, StorageLayout::BrushF32, true).is_none());
        assert!(tensor_memory(u64::MAX, 3, StorageLayout::BrushF32, true).is_none());
    }
}
