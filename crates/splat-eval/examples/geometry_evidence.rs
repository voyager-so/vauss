//! Reduce a COLMAP model to the evidence consumed by the geometry gates (CPU only).
//! Usage: cargo run -p splat-eval --example geometry_evidence -- path/to/sparse/0
fn main() -> anyhow::Result<()> {
    let path = std::env::args_os().nth(1).expect("model directory");
    let model = splat_eval::read_model(std::path::Path::new(&path))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&splat_eval::geometry::geometry_evidence(&model))?
    );
    Ok(())
}
