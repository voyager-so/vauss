//! Finalize existing operator-run COLMAP stages using the same export path as reconstruct().
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use splat_sfm::{component_inventory, ensure_video_continuity, finalize_model};

#[derive(Parser)]
struct Args {
    sparse: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long)]
    video: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (model_dir, health) = finalize_model(&args.sparse, &args.images, args.video, 8)?;
    let components = component_inventory(&args.sparse)?;
    let report = serde_json::json!({
        "model_dir": model_dir,
        "health": health,
        "components": components,
    });
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(args.sparse.join("finalization.json"), &json)?;
    println!("{json}");
    ensure_video_continuity(&health, args.video)
}
