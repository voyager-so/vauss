//! Reuse of compute bind groups across launches.
//!
//! Every launch used to build a fresh [`wgpu::BindGroup`], which shows up as a
//! large share of server-thread CPU time for workloads that issue many small
//! dispatches against buffers the memory pools keep reusing. A bind group is
//! fully determined by the pipeline's layout and, per binding, the buffer and
//! the bound range, so it is cached under exactly that key: a hit can only
//! ever bind the same buffer ranges a freshly built group would.
//!
//! A cached group (and its key) holds strong references to its buffers and
//! pipeline. To never keep freed memory alive, the storages backing bound
//! buffers evict every entry binding a buffer when they deallocate it (see
//! [`WgpuStorage`](crate::WgpuStorage)), and the cache is cleared entirely once
//! it exceeds [`MAX_ENTRIES`], which bounds how long it can keep a dropped
//! pipeline alive.

use crate::WgpuResource;
use core::hash::{Hash, Hasher};
use cubecl_environment::collections::HashMap;
use cubecl_environment::sync::{Arc, Mutex};
use hashbrown::Equivalent;
use wgpu::{Buffer, ComputePipeline};

/// The cache is cleared entirely once it holds more entries than this.
const MAX_ENTRIES: usize = 4096;

/// A handle to a shared bind group cache. Cloning shares the same cache.
#[derive(Clone, Default, Debug)]
pub(crate) struct BindGroupCache {
    entries: Arc<Mutex<HashMap<BindGroupKey, wgpu::BindGroup>>>,
}

impl BindGroupCache {
    /// Get the bind group binding `resources` (in order) for `pipeline`, building
    /// and caching it on a miss.
    pub(crate) fn get_or_create(
        &self,
        device: &wgpu::Device,
        pipeline: &ComputePipeline,
        resources: &[WgpuResource],
    ) -> wgpu::BindGroup {
        let lookup = BindGroupLookup {
            pipeline,
            resources,
        };
        let mut entries = self.entries.lock();
        if let Some(bind_group) = entries.get(&lookup) {
            return bind_group.clone();
        }

        let bind_entries = resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.as_wgpu_bind_resource(),
            })
            .collect::<Vec<_>>();
        let group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group_layout,
            entries: &bind_entries,
        });

        if entries.len() >= MAX_ENTRIES {
            entries.clear();
        }
        entries.insert(BindGroupKey::new(pipeline, resources), bind_group.clone());
        bind_group
    }

    /// Drop every cached bind group that binds `buffer`, so the cache never
    /// keeps a freed buffer alive.
    pub(crate) fn evict_buffer(&self, buffer: &Buffer) {
        self.entries
            .lock()
            .retain(|key, _| !key.bindings.iter().any(|(b, _, _)| b == buffer));
    }
}

/// Owned cache key: the pipeline, and per binding the buffer and bound range.
#[derive(Debug, PartialEq, Eq)]
struct BindGroupKey {
    pipeline: ComputePipeline,
    bindings: Vec<(Buffer, u64, u64)>,
}

impl BindGroupKey {
    fn new(pipeline: &ComputePipeline, resources: &[WgpuResource]) -> Self {
        Self {
            pipeline: pipeline.clone(),
            bindings: resources
                .iter()
                .map(|r| (r.buffer.clone(), r.offset, r.size))
                .collect(),
        }
    }
}

/// Borrowed form of [`BindGroupKey`], so a hit allocates nothing.
struct BindGroupLookup<'a> {
    pipeline: &'a ComputePipeline,
    resources: &'a [WgpuResource],
}

// Both key forms hash the same sequence, as `Equivalent` requires.
fn hash_key<'a, H: Hasher>(
    state: &mut H,
    pipeline: &ComputePipeline,
    bindings: impl ExactSizeIterator<Item = (&'a Buffer, u64, u64)>,
) {
    pipeline.hash(state);
    state.write_usize(bindings.len());
    for (buffer, offset, size) in bindings {
        buffer.hash(state);
        state.write_u64(offset);
        state.write_u64(size);
    }
}

impl Hash for BindGroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_key(
            state,
            &self.pipeline,
            self.bindings.iter().map(|(b, o, s)| (b, *o, *s)),
        );
    }
}

impl Hash for BindGroupLookup<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_key(
            state,
            self.pipeline,
            self.resources.iter().map(|r| (&r.buffer, r.offset, r.size)),
        );
    }
}

impl Equivalent<BindGroupKey> for BindGroupLookup<'_> {
    fn equivalent(&self, key: &BindGroupKey) -> bool {
        *self.pipeline == key.pipeline
            && self.resources.len() == key.bindings.len()
            && self
                .resources
                .iter()
                .zip(key.bindings.iter())
                .all(|(r, (b, o, s))| r.buffer == *b && r.offset == *o && r.size == *s)
    }
}
