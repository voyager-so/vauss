//! Lists every wgpu adapter per graphics API and mirrors how cubecl-wgpu brings up the
//! default device: `AutoGraphicsApi` settles on the first of Vulkan, DX12, GL with a non-CPU
//! adapter (else the first with any adapter), `request_adapter(HighPerformance)` picks the
//! adapter there, and the device is opened with all adapter features and limits.
//! Prints one JSON object; exits 1 when no device can be created.

use wgpu::{Backend, DeviceType, Features};

fn instance(api: Backend) -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: api.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

fn describe(a: &wgpu::Adapter) -> String {
    let i = a.get_info();
    format!(
        "{{\"name\":{:?},\"device_type\":\"{:?}\",\"driver\":{:?},\"driver_info\":{:?},\"backend\":\"{:?}\",\"subgroups\":{},\"shader_f16\":{}}}",
        i.name,
        i.device_type,
        i.driver,
        i.driver_info,
        i.backend,
        a.features().contains(Features::SUBGROUP),
        a.features().contains(Features::SHADER_F16)
    )
}

fn main() {
    let mut apis = Vec::new();
    let (mut gpu_api, mut software_api) = (None, None);
    for api in [Backend::Vulkan, Backend::Dx12, Backend::Gl] {
        let adapters = pollster::block_on(instance(api).enumerate_adapters(api.into()));
        apis.push(format!("\"{api:?}\":[{}]", adapters.iter().map(describe).collect::<Vec<_>>().join(",")));
        if adapters.iter().any(|a| a.get_info().device_type != DeviceType::Cpu) {
            gpu_api = gpu_api.or(Some(api));
        } else if !adapters.is_empty() {
            software_api = software_api.or(Some(api));
        }
    }
    let settled = gpu_api.or(software_api);
    let adapter = settled.and_then(|api| {
        pollster::block_on(instance(api).request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()
    });
    let (selected, device) = match &adapter {
        None => ("null".to_string(), "\"no adapter on any API\"".to_string()),
        Some(a) => {
            let result = pollster::block_on(a.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: a.features().difference(Features::MAPPABLE_PRIMARY_BUFFERS),
                required_limits: a.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                // SAFETY: cubecl enables the same experimental features for its device.
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            }));
            let device = match result {
                Ok(_) => "\"ok\"".to_string(),
                Err(e) => format!("{:?}", format!("{e:?}")),
            };
            (describe(a), device)
        }
    };
    println!(
        "{{\"adapters\":{{{}}},\"settled_api\":\"{}\",\"auto_selected\":{selected},\"request_device\":{device}}}",
        apis.join(","),
        settled.map_or("none".to_string(), |a| format!("{a:?}"))
    );
    if device != "\"ok\"" {
        std::process::exit(1);
    }
}
