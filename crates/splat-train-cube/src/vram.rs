//! Usable device memory for sizing training (e.g. the splat cap), detected once
//! at device setup. Vulkan reports device-local heaps and, with
//! VK_EXT_memory_budget, what the driver will actually give this process;
//! other backends report nothing and callers fall back to a conservative size.

use burn_wgpu::WgpuSetup;

/// Device-local bytes this process can expect to use, if the backend says.
pub fn detect(setup: &WgpuSetup) -> Option<u64> {
    #[cfg(not(target_family = "wasm"))]
    if let Some(bytes) = vulkan(&setup.adapter) {
        return Some(bytes);
    }
    let _ = setup;
    None
}

/// Largest device-local heap, capped by its VK_EXT_memory_budget budget when
/// the driver supports the extension (the budget accounts for other processes).
#[cfg(not(target_family = "wasm"))]
fn vulkan(adapter: &wgpu::Adapter) -> Option<u64> {
    use ash::vk;
    // SAFETY: read-only queries on the adapter's own instance and physical device.
    unsafe {
        let hal = adapter.as_hal::<wgpu::hal::api::Vulkan>()?;
        let instance = hal.shared_instance().raw_instance();
        let device = hal.raw_physical_device();
        let budget_ext = instance
            .enumerate_device_extension_properties(device)
            .unwrap_or_default()
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceMemoryProperties2::default();
        if budget_ext {
            props = props.push_next(&mut budget);
        }
        instance.get_physical_device_memory_properties2(device, &mut props);
        let memory = props.memory_properties;
        (0..memory.memory_heap_count as usize)
            .filter(|&i| memory.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|i| {
                let size = memory.memory_heaps[i].size;
                if budget_ext && budget.heap_budget[i] > 0 {
                    size.min(budget.heap_budget[i])
                } else {
                    size
                }
            })
            .max()
    }
}

static BUDGET: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();

/// Share of usable device memory training may plan to fill.
const BUDGET_FRACTION: f64 = 0.85;

/// Record the detected device memory once. `SPLAT_MEM_BUDGET_MB` overrides the
/// budget (internal, for forcing the memory guard on a large device).
pub fn set_device_memory(bytes: Option<u64>) {
    let forced = std::env::var("SPLAT_MEM_BUDGET_MB").ok().and_then(|v| v.parse::<u64>().ok());
    let budget = forced
        .map(|mb| mb << 20)
        .or_else(|| bytes.map(|b| (b as f64 * BUDGET_FRACTION) as u64));
    let _ = BUDGET.set(budget);
}

/// Bytes training may plan to hold on the device, when known.
pub fn device_budget() -> Option<u64> {
    BUDGET.get().copied().flatten()
}
