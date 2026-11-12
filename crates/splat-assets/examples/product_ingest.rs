//! Usage: product_ingest inspect URL | generate request.json output-directory
use anyhow::{Context, Result, bail};
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("inspect") => println!(
            "{}",
            serde_json::to_string_pretty(&splat_assets::product::inspect(
                args.get(2).context("URL required")?
            )?)?
        ),
        Some("bounds") => println!(
            "{:?}",
            splat_assets::glb_dims(&std::fs::read(args.get(2).context("GLB required")?)?)?
        ),
        Some("generate") => {
            let req = serde_json::from_slice(&std::fs::read(
                args.get(2).context("request.json required")?,
            )?)?;
            let out = std::path::Path::new(args.get(3).context("output directory required")?);
            let fal = splat_shots::fal::Fal::from_env()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&splat_assets::product::ingest(
                    &fal,
                    &req,
                    out,
                    |s| eprintln!("{s}")
                )?)?
            );
        }
        _ => bail!("usage: product_ingest inspect URL | generate request.json output-directory"),
    }
    Ok(())
}
