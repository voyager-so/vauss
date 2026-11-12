use anyhow::Result;
use clap::Parser;
use splat_sfm::{Features, Mapper, Matching, ReconstructOptions, reconstruct};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Fresh COLMAP 4.2.1 reconstruction with recomputed SfM health")]
struct Cli {
    frames: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    gpu_lock: Option<PathBuf>,
    #[arg(long, default_value = "colmap")]
    colmap: PathBuf,
    #[arg(long, value_enum, default_value = "incremental")]
    mapper: Mapper,
    #[arg(long, value_enum, default_value = "exhaustive")]
    matching: Matching,
    #[arg(long, requires = "lightglue_model")]
    aliked_model: Option<PathBuf>,
    #[arg(long, requires = "aliked_model")]
    lightglue_model: Option<PathBuf>,
    #[arg(long)]
    video: bool,
    #[arg(long, default_value_t = 1600)]
    max_image_size: usize,
    #[arg(long, default_value_t = 8192)]
    max_features: usize,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    first_octave: i32,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    #[arg(long, default_value_t = 10)]
    overlap: usize,
    #[arg(long)]
    vocab_tree: Option<PathBuf>,
    #[arg(long)]
    guided: bool,
    #[arg(long, default_value_t = 10000)]
    ransac_max_trials: usize,
    #[arg(long, default_value_t = 3)]
    min_track_length: usize,
    #[arg(long)]
    no_calibrate: bool,
    #[arg(long)]
    cpu: bool,
    #[arg(long, default_value_t = 3)]
    ba_iterations: usize,
    #[arg(long, default_value_t = 200)]
    ba_max_iterations: usize,
    #[arg(long, default_value = "SIMPLE_RADIAL")]
    camera_model: String,
    #[arg(long)]
    mask_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let opts = ReconstructOptions {
        colmap: cli.colmap,
        gpu_lock: cli.gpu_lock,
        mapper: cli.mapper,
        matching: cli.matching,
        features: match (cli.aliked_model, cli.lightglue_model) {
            (Some(detector), Some(matcher)) => Features::AlikedLightGlue { detector, matcher },
            _ => Features::Sift,
        },
        video: cli.video,
        max_image_size: cli.max_image_size,
        max_features: cli.max_features,
        first_octave: cli.first_octave,
        num_threads: cli.threads,
        sequential_overlap: cli.overlap,
        vocab_tree: cli.vocab_tree,
        guided_matching: cli.guided,
        ransac_max_trials: cli.ransac_max_trials,
        min_track_length: cli.min_track_length,
        calibrate: !cli.no_calibrate,
        use_gpu: !cli.cpu,
        ba_iterations: cli.ba_iterations,
        ba_max_iterations: cli.ba_max_iterations,
        camera_model: cli.camera_model,
        mask_dir: cli.mask_dir,
        ..ReconstructOptions::new(&cli.output)
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&reconstruct(&cli.frames, &opts)?)?
    );
    Ok(())
}
