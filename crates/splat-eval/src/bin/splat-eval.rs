//! Dev CLI mirroring eval/metrics.py, eval/sfm_health.py and eval/video_stats.py; prints JSON.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use splat_eval::metrics::ColorCorrection;
use splat_eval::{MetricOptions, SfmHealthOptions, VideoStatsOptions};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// GT_DIR RENDER_DIR, or a Spirula run dir alone (eval-gt-*/eval-render-*).
    Metrics {
        gt_dir: PathBuf,
        render_dir: Option<PathBuf>,
        /// Skip LPIPS on purpose; the output then carries `"lpips": null`. Without it, LPIPS that
        /// cannot run (no ONNX Runtime, no models) is an error, never a PSNR-only result.
        #[arg(long)]
        no_lpips: bool,
        #[arg(long)]
        no_cc: bool,
        #[arg(long)]
        mask_dir: Option<PathBuf>,
        #[arg(long)]
        models_dir: Option<PathBuf>,
        #[arg(long)]
        resize: bool,
        /// Spirula run dir: held-out views the run should have evaluated; fewer unique ones is an error.
        #[arg(long)]
        expect: Option<usize>,
    },
    Sfm {
        sparse_dir: PathBuf,
        #[arg(long)]
        images: Option<PathBuf>,
        #[arg(long)]
        num_input: Option<usize>,
        #[arg(long)]
        video: bool,
        #[arg(long, default_value_t = 8)]
        holdout_every: usize,
        /// Pose-continuity gate threshold (degrees per frame step); 0 turns the gate off.
        #[arg(long, default_value_t = 30.0)]
        max_pose_jump_deg: f64,
    },
    Video {
        videos: Vec<PathBuf>,
        #[arg(long, default_value_t = 4.0)]
        fps: f64,
    },
}

fn main() -> Result<()> {
    let out = match Cli::parse().cmd {
        Cmd::Metrics { gt_dir, render_dir, no_lpips, no_cc, mask_dir, models_dir, resize, expect } => {
            let (pairs, missing) = match &render_dir {
                Some(r) => splat_eval::pair_dirs(&gt_dir, r)?,
                None => {
                    // Spirula can export one view twice and skip another: score unique views only.
                    let (pairs, missing, total) = splat_eval::spirula_unique_pairs(&gt_dir)?;
                    eprintln!("spirula eval: {} unique of {total} exported views", pairs.len());
                    if let Some(n) = expect {
                        anyhow::ensure!(pairs.len() >= n, "spirula eval: {} unique views, expected {n}", pairs.len());
                    }
                    (pairs, missing)
                }
            };
            let opts = MetricOptions {
                lpips: !no_lpips,
                models_dir,
                color_correction: (!no_cc).then_some(ColorCorrection::Quadratic),
                cc_lpips: true,
                mask_dir,
                resize,
            };
            let summary = splat_eval::evaluate_pairs(&pairs, &missing, &opts)?;
            if let Some(why) = &summary.lpips_unavailable {
                anyhow::bail!(
                    "LPIPS requested but unavailable: {why}\n\
                     Set ORT_DYLIB_PATH to an ONNX Runtime 1.28 libonnxruntime.so (`vauss doctor` installs one in \
                     ~/.local/share/splat/runtimes/onnxruntime/). Or pass \
                     --no-lpips to score without LPIPS on purpose (\"lpips\": null)."
                );
            }
            let mut v = serde_json::to_value(summary)?;
            if no_lpips {
                v["lpips"] = serde_json::Value::Null;
            }
            v
        }
        Cmd::Sfm { sparse_dir, images, num_input, video, holdout_every, max_pose_jump_deg } => {
            let model = splat_eval::read_model(&sparse_dir)?;
            let max_pose_jump_deg = (max_pose_jump_deg > 0.0).then_some(max_pose_jump_deg);
            let opts = SfmHealthOptions { images_dir: images, num_input, video, holdout_every, max_pose_jump_deg };
            serde_json::to_value(splat_eval::sfm_health(&model, &sparse_dir, &opts)?)?
        }
        Cmd::Video { videos, fps } => {
            let opts = VideoStatsOptions { fps, ..Default::default() };
            let stats = videos
                .iter()
                .map(|v| Ok((v.display().to_string(), serde_json::to_value(splat_eval::video_stats(v, &opts)?)?)))
                .collect::<Result<serde_json::Map<_, _>>>()?;
            serde_json::Value::Object(stats)
        }
    };
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
