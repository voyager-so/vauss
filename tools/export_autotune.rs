//! Export a warmed CubeCL environment using its native portable bundle format.
//! Usage: export_autotune CACHE_DB OUTPUT DEVICE_LABEL OS ARCH
use cubecl_environment::bundle::{BundleFormat, EnvironmentInfo, ExportOptions, export};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: export_autotune CACHE_DB OUTPUT DEVICE_LABEL OS ARCH".into());
    }
    let options = ExportOptions {
        name: format!("splat autotune {}", args[3]),
        environments: vec![EnvironmentInfo {
            label: args[3].clone(),
            devices: vec![args[3].clone()],
            os: args[4].clone(),
            arch: args[5].clone(),
        }],
        namespaces: vec!["autotune".into(), "throughput".into()],
        format: BundleFormat::Flat,
    };
    let manifest = export(&[&args[1]], &args[2], &options)?;
    println!("Exported {} ({})", manifest.name, manifest.cubecl_version);
    Ok(())
}
