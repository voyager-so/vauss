mod base;
mod wgsl;

#[cfg(feature = "spirv")]
pub mod vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
pub mod metal;

#[cfg(windows)]
pub mod dx12;

pub use base::*;

/// Allocator hints for the Vulkan/DX12 suballocator (gpu-allocator).
///
/// `MemoryUsage` suballocates everything up to 64 MiB out of shared 64 MiB
/// blocks. The memory pool's pages are mostly 10–95 MiB, so a 40 MiB page
/// pins a whole 64 MiB block and the tail is wasted: on a 1M-splat garden run
/// that slack is up to ~200 MiB of board memory that no pool accounts for, and
/// it peaks when refine frees and re-creates its pages. Our pool already
/// batches small buffers into pages, so keep the shared blocks at the minimum
/// size (4 MiB): anything bigger gets its own exactly-sized allocation.
pub(crate) fn memory_hints() -> wgpu::MemoryHints {
    const BLOCK: u64 = 4 * 1024 * 1024;
    wgpu::MemoryHints::Manual {
        suballocated_device_memory_block_size: BLOCK..BLOCK,
    }
}
