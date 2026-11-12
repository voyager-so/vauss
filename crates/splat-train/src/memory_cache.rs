//! A bounded cache of immutable packed training images on one inner device.
//!
//! The caller supplies a stable index within one training scene. Construct a
//! new cache if the scene, image preprocessing, or device changes. Resolution
//! changes replace that view's old entry. A zero budget uploads on every visit.

use std::collections::HashMap;

use burn::tensor::{DType, Device, Int, Tensor, TensorData};

struct Entry {
    tensor: Tensor<2, Int>,
    bytes: u64,
    last_used: u64,
}

/// Logical resident tensor bytes are bounded by `budget_bytes`. Driver staging
/// allocations and allocator caches are outside this budget and must be measured.
pub struct GpuImageCache {
    device: Device,
    budget_bytes: u64,
    resident_bytes: u64,
    clock: u64,
    entries: HashMap<usize, Entry>,
}

impl GpuImageCache {
    pub fn new(device: &Device, budget_bytes: u64) -> Self {
        Self {
            device: device.clone().inner(),
            budget_bytes,
            resident_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
        }
    }

    /// Upload or return a cached `[height, width]` packed RGBA8 image.
    /// Data for a given view and resolution must be immutable in this cache.
    pub fn get_or_upload(&mut self, view_index: usize, packed: TensorData) -> Tensor<2, Int> {
        assert_eq!(
            packed.shape.len(),
            2,
            "expected packed [height, width] image"
        );
        assert_eq!(
            packed.dtype,
            DType::I32,
            "expected packed RGBA8 in i32 storage"
        );
        let shape = [packed.shape[0], packed.shape[1]];
        let bytes = packed.bytes.len() as u64;
        if self.budget_bytes == 0 {
            return Tensor::from_data(packed, &self.device);
        }
        // Saturation is practically unreachable. Clearing avoids losing LRU
        // ordering if a caller does perform 2^64 cache operations.
        if self.clock == u64::MAX {
            self.clear();
        }
        self.clock += 1;
        if let Some(entry) = self.entries.get_mut(&view_index) {
            if entry.tensor.dims() == shape {
                entry.last_used = self.clock;
                return entry.tensor.clone();
            }
            // Progressive resolution must release the previous GPU image.
            let old = self.entries.remove(&view_index).expect("entry exists");
            self.resident_bytes -= old.bytes;
        }
        // An oversized image bypasses caching without evicting other views.
        // Evict before upload, so cache-held images do not coexist with the
        // new upload unnecessarily. In-flight tensor clones may still live.
        if bytes <= self.budget_bytes {
            while self.resident_bytes > self.budget_bytes - bytes {
                let oldest = *self
                    .entries
                    .iter()
                    .min_by_key(|(_, e)| e.last_used)
                    .expect("resident bytes imply an entry")
                    .0;
                self.resident_bytes -= self.entries.remove(&oldest).unwrap().bytes;
            }
        }
        let tensor = Tensor::from_data(packed, &self.device);
        if bytes <= self.budget_bytes {
            self.entries.insert(
                view_index,
                Entry {
                    tensor: tensor.clone(),
                    bytes,
                    last_used: self.clock,
                },
            );
            self.resident_bytes += bytes;
        }
        tensor
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.resident_bytes = 0;
        self.clock = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "GPU test: run under the shared exclusive GPU flock"]
    async fn cache_obeys_budget_lru_and_resolution_changes() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let image = || TensorData::new(vec![1i32, 2, 3, 4], [2, 2]);
        let mut cache = GpuImageCache::new(&device, 32);
        let _ = cache.get_or_upload(0, image());
        let _ = cache.get_or_upload(1, image());
        assert_eq!(cache.resident_bytes(), 32); // Exact budget is admitted.
        let _ = cache.get_or_upload(0, image());
        let t = cache.get_or_upload(2, image());
        assert!(cache.entries.contains_key(&0));
        assert!(!cache.entries.contains_key(&1));
        assert_eq!(
            t.into_data_async()
                .await
                .unwrap()
                .try_to_vec::<i32>()
                .unwrap(),
            [1, 2, 3, 4]
        );
        let _ = cache.get_or_upload(0, TensorData::new(vec![9i32; 16], [4, 4]));
        assert!(!cache.entries.contains_key(&0));
        assert!(cache.entries.contains_key(&2));
        assert_eq!(cache.resident_bytes(), 16);
        cache.clear();
        assert_eq!(cache.resident_bytes(), 0);
        let mut streaming = GpuImageCache::new(&device, 0);
        let _ = streaming.get_or_upload(0, image());
        assert_eq!(streaming.resident_bytes(), 0);
        assert!(streaming.entries.is_empty());
    }
}
