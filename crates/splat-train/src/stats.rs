use burn::tensor::{Bool, Device, Tensor};
use tracing::trace_span;

pub(crate) struct RefineRecord {
    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    pub refine_weight_norm: Tensor<1>,
    pub vis_weight: Tensor<1>,
    pub max_screen_size: Tensor<1>,
}

impl RefineRecord {
    pub(crate) fn new(num_points: u32, device: &Device) -> Self {
        Self {
            refine_weight_norm: Tensor::<1>::zeros([num_points as usize], device),
            vis_weight: Tensor::<1>::zeros([num_points as usize], device),
            max_screen_size: Tensor::<1>::zeros([num_points as usize], device),
        }
    }

    /// Visible splats whose max 2D screen-space extent (as a fraction of the
    /// image dim) exceeds `threshold` — i.e. the "too big on screen" outliers.
    pub(crate) fn above_screen_size(&self, threshold: f32) -> Tensor<1, Bool> {
        self.max_screen_size
            .clone()
            .greater_elem(threshold)
            .bool_and(self.vis_mask())
    }

    /// One fused kernel: max refine weight, visible count, max screen radius.
    pub(crate) fn gather_stats(
        self,
        refine_weight: Tensor<1>,
        visible: Tensor<1>,
        screen_radius: Tensor<1>,
    ) -> Self {
        let _span = trace_span!("Gather stats").entered();
        let (refine_weight_norm, vis_weight, max_screen_size) = crate::housekeep::gather_stats(
            self.refine_weight_norm,
            self.vis_weight,
            self.max_screen_size,
            refine_weight,
            visible,
            screen_radius,
        );
        Self {
            refine_weight_norm,
            vis_weight,
            max_screen_size,
        }
    }

    pub(crate) fn vis_mask(&self) -> Tensor<1, Bool> {
        self.vis_weight.clone().greater_elem(0.0)
    }
}
