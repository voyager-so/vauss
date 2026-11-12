//! Operator entry point for comparing native and external SfM models fairly.
//! Stored COLMAP point errors are not per-observation averages: always recompute.
use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::Parser;
use splat_eval::{SfmHealthOptions, read_model, sfm_health};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    #[arg(long)]
    images: PathBuf,
    #[arg(long)]
    video: bool,
    #[arg(long, default_value_t = 8)]
    holdout_every: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.images.is_dir(), "image directory is missing");
    ensure!(args.holdout_every > 0, "holdout stride must be positive");
    let mut model = read_model(&args.model)?;
    for point in &mut model.points {
        point.error = 0.0;
    }
    let health = sfm_health(&model, &args.model, &SfmHealthOptions {
        images_dir: Some(args.images),
        video: args.video,
        holdout_every: args.holdout_every,
        ..Default::default()
    })?;
    println!("{}", serde_json::to_string_pretty(&health)?);
    Ok(())
}
