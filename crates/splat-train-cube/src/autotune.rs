//! Shipped tuning results for the default device, imported before its first tensor operation.
use burn::cubecl::{
    Device, bundle,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    future,
    std::throughput::measure_memory_curve,
    throughput::{MemoryAccess, MemorySpec, ThroughputError, ThroughputKey, ThroughputMode, sweep_size},
};
use burn::tensor::Bytes;
use burn_wgpu::{WgpuDevice, WgpuSetup};
use std::sync::OnceLock;

/// The adapter the default device runs on, recorded once at setup so results
/// can prove which GPU and driver produced them.
#[derive(Clone, Debug)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub vendor: u32,
    pub device: u32,
    /// Usable device-local memory, when the backend reports it.
    pub vram_bytes: Option<u64>,
}

static ADAPTER: OnceLock<AdapterSummary> = OnceLock::new();

/// The default device's adapter, once [`install_default_device_autotune`] ran.
pub fn adapter_summary() -> Option<&'static AdapterSummary> {
    ADAPTER.get()
}

/// Seed the default device's cache only for an adapter and compiler we measured.
/// Unknown hardware and explicitly disabled persistence retain ordinary autotuning.
/// Call after device setup, before creating tensors; custom device IDs use separate namespaces.
pub fn install_default_device_autotune(setup: &WgpuSetup) {
    let info = setup.adapter.get_info();
    let summary = ADAPTER.get_or_init(|| AdapterSummary {
        name: info.name.clone(),
        backend: format!("{:?}", info.backend),
        driver: info.driver.clone(),
        driver_info: info.driver_info.clone(),
        vendor: info.vendor,
        device: info.device,
        vram_bytes: crate::vram::detect(setup),
    });
    crate::vram::set_device_memory(summary.vram_bytes);
    log::info!(
        "wgpu adapter: {} ({}), driver {} {}, usable VRAM {}",
        summary.name,
        summary.backend,
        summary.driver,
        summary.driver_info,
        summary.vram_bytes.map_or("unknown".to_owned(), |b| format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64))
    );
    if !CubeClRuntimeConfig::get().autotune.disable_cache
        && Device::Wgpu(WgpuDevice::default()).client().name() == "wgpu<wgsl>"
    {
        import_seed(setup);
    }
    probe_throughput_unless_cached();
}

/// Measure the memory ceilings autotune's roofline bounds read, before any training buffer exists,
/// when nothing (a seed or an earlier run) cached them for this device. Every memory probe holds a
/// ~2 GiB pool whatever size it measures; run lazily mid-training it stacks on the live set. Same
/// keys and measurement, so the same kernel picks, and cached for later runs.
fn probe_throughput_unless_cached() {
    let client = Device::Wgpu(WgpuDevice::default()).client();
    let smallest = ThroughputKey {
        mode: ThroughputMode::Memory(MemorySpec::new(MemoryAccess::Copy, sweep_size(0))),
    };
    // A lookup: the probe only runs on a miss, and its error is never cached.
    if client.measure_throughput(smallest, || Err(ThroughputError::Unsupported)).is_ok() {
        return;
    }
    let start = std::time::Instant::now();
    let curve = measure_memory_curve(&client, MemoryAccess::Copy);
    // Pages freed while their probe is still in flight stay resident.
    let _ = future::block_on(client.sync());
    client.memory_cleanup();
    log::info!("Throughput probe at init: {} points in {:.2?}", curve.points().len(), start.elapsed());
}

/// Seed the cache from the shipped bundle for a measured adapter.
fn import_seed(setup: &WgpuSetup) {
    let info = setup.adapter.get_info();
    let backend = format!("{:?}", info.backend);
    let bytes: &[u8] = match (info.name.as_str(), info.vendor, info.device, backend.as_str()) {
        ("NVIDIA L40S", 0x10de, 0x26b9, "Vulkan") => include_bytes!("../autotune/l40s.bundle"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 0x10de, 0x2bb5, "Vulkan") => {
            include_bytes!("../autotune/rtx-pro-6000.bundle")
        }
        ("NVIDIA H100 80GB HBM3", 0x10de, 0x2330, "Vulkan") => include_bytes!("../autotune/h100.bundle"),
        ("NVIDIA GeForce RTX 5080", 4318, 11266, "Vulkan") => {
            include_bytes!("../autotune/rtx-5080.bundle")
        }
        ("Apple M4", 0, 0, "Metal") => include_bytes!("../autotune/apple-m4.bundle"),
        _ => return,
    };
    match bundle::EmbeddedBundle::open(Bytes::from_bytes_vec(bytes.to_vec())) {
        Ok(seed) => {
            let report = bundle::import(&seed);
            log::info!("Autotune seed for {}: {} imported, {} already cached", info.name, report.imported, report.skipped);
        }
        Err(error) => log::warn!("Ignoring invalid autotune seed for {}: {error}", info.name),
    }
}
