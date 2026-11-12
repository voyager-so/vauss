use super::bind_group_cache::BindGroupCache;
use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::{num::NonZeroU64, ptr::NonNull};
use wgpu::BufferUsages;

/// Minimum buffer size in bytes. The WebGPU spec requires buffer sizes > 0, and shaders
/// declare typed arrays (e.g. `array<vec4<f32>>`) that impose a minimum binding size.
/// 32 bytes covers the largest possible binding type (`vec4<f64>`).
const MIN_BUFFER_SIZE: u64 = 32;

// splat patch (wasm only): wgpu's WebGPU backend never calls `GPUBuffer.destroy()` on drop (its
// `Drop for WebBuffer` is a no-op), so a freed buffer keeps its VRAM until the JS garbage
// collector gets to the wrapper. Freed buffers are queued here and destroyed by the stream right
// after its next tasks submit: every command that used them was encoded before the free, so it
// is in that submit or an earlier one, and WebGPU defers the destroy until submitted work ends.
#[cfg(target_family = "wasm")]
thread_local! {
    static PENDING_DESTROY: core::cell::RefCell<Vec<wgpu::Buffer>> = const { core::cell::RefCell::new(Vec::new()) };
}

/// Buffers freed since the last tasks submit (wasm only), for the stream to destroy once it
/// has submitted.
#[cfg(target_family = "wasm")]
pub(crate) fn take_pending_destroy() -> Vec<wgpu::Buffer> {
    PENDING_DESTROY.with(|p| core::mem::take(&mut *p.borrow_mut()))
}

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, WgpuMemory>,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
    #[allow(unused, reason = "keep it simple")]
    vk_storage: bool,
    /// Bind groups that may reference this storage's buffers. A dealloc evicts
    /// the ones binding the freed buffer so the cache never keeps it alive.
    bind_groups: Option<BindGroupCache>,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// A buffer's persistent host mapping. Crate private: writes through it need an idle queue.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HostPtr(pub(crate) NonNull<u8>);

// SAFETY: Only the stream owning the buffer writes through it.
unsafe impl Send for HostPtr {}
unsafe impl Sync for HostPtr {}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug, Clone)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer host mapping, if host visible.
    #[new(default)]
    pub(crate) host_ptr: Option<HostPtr>,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
}

/// The memory that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuMemory {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer host mapping, if host visible.
    #[new(default)]
    pub(crate) host_ptr: Option<HostPtr>,
}

impl WgpuResource {
    /// Return the binding view of the buffer.
    pub fn as_wgpu_bind_resource(&self) -> wgpu::BindingResource<'_> {
        // wgpu enforces 4-byte alignment for buffer binding sizes per the WebGPU spec.
        // - https://github.com/gfx-rs/wgpu/pull/8041
        //
        // This padding is safe because:
        // 1. In checked mode, bounds checks prevent reading beyond the logical size.
        // 2. In unchecked mode, OOB access is already undefined behavior.
        //
        // For zero-sized resources, pass None (use rest of buffer from offset).
        // The allocator guarantees the buffer is at least MIN_BUFFER_SIZE bytes.
        let size = NonZeroU64::new(self.size.next_multiple_of(4));

        let binding = wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size,
        };
        wgpu::BindingResource::Buffer(binding)
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(
        mem_alignment: usize,
        device: wgpu::Device,
        usages: BufferUsages,
        vk_storage: bool,
    ) -> Self {
        Self {
            memory: HashMap::new(),
            device,
            buffer_usages: usages,
            mem_alignment,
            vk_storage,
            bind_groups: None,
        }
    }

    /// Evict from `cache` every bind group over a buffer this storage frees.
    pub(crate) fn with_bind_group_cache(mut self, cache: BindGroupCache) -> Self {
        self.bind_groups = Some(cache);
        self
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        let memory = self
            .memory
            .get(&handle.id)
            .ok_or_else(|| IoError::StorageHandleNotFound {
                reason: format!("{} in the wgpu buffer storage", handle.id).into(),
                backtrace: BackTrace::capture(),
            })?;
        Ok(WgpuResource {
            host_ptr: memory.host_ptr,
            ..WgpuResource::new(
                memory.buffer.clone(),
                memory.address,
                handle.offset(),
                handle.size(),
            )
        })
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let alloc_size = size.max(MIN_BUFFER_SIZE);

        let memory = self.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_size,
            usage: self.buffer_usages,
            mapped_at_creation: false,
        })?;

        self.memory.insert(id, memory);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        let memory = self.memory.remove(&id);
        #[cfg(not(target_family = "wasm"))]
        if let (Some(bind_groups), Some(memory)) = (&self.bind_groups, memory) {
            bind_groups.evict_buffer(&memory.buffer);
        }
        #[cfg(target_family = "wasm")]
        if let Some(memory) = memory {
            if let Some(bind_groups) = &self.bind_groups {
                bind_groups.evict_buffer(&memory.buffer);
            }
            PENDING_DESTROY.with(|p| p.borrow_mut().push(memory.buffer));
        }
    }

    fn flush(&mut self) {
        // We don't wait for dealloc
    }
}

impl WgpuStorage {
    #[cfg(feature = "spirv")]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        if self.vk_storage {
            // wgpu currently doesn't expose this, even though it's used internally for acceleration
            // structures. While we could use the acceleration structure input flag, that would
            // then require ray tracing to be supported. So we need to allocate manually, then import
            // the native buffer into wgpu using `from_raw_managed`.
            // This actually skips some of the buffer batching stuff we don't really want in `gpu_allocator`.
            crate::backend::vulkan::create_storage_buffer(&self.device, desc)
        } else {
            Ok(WgpuMemory::new(self.device.create_buffer(desc), None))
        }
    }

    #[cfg(not(feature = "spirv"))]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        Ok(WgpuMemory::new(self.device.create_buffer(desc), None))
    }
}
