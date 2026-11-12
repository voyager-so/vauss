//! Evaluation for splat runs: image metrics (PSNR, SSIM, colour-corrected variants, LPIPS via ONNX
//! Runtime), COLMAP model reading + SfM health, and cheap video capture statistics.
//! Also: distractor-mask sanity stats and a shared image byte budget for tool results.
//! Ports of eval/metrics.py, eval/sfm_health.py and eval/video_stats.py; numbers match them.

pub mod colmap;
pub mod frame_arms;
pub mod frameplan;
pub mod color;
pub mod images;
pub mod lpips;
pub mod masks;
pub mod metrics;
pub mod precheck;
pub mod preview;
pub mod sfm;
pub mod ssim;
pub mod undistort;
pub mod video;

pub use colmap::{read_model, Model};
pub use images::{pair_dirs, spirula_pairs, spirula_unique_pairs, Pair};
pub use masks::{centre_frame, mask_stats, mask_stats_file, training_mask_dir, CentreFrame, MaskStats, SubjectGuard};
pub use metrics::{evaluate_pairs, EvalSummary, MetricOptions};
pub use preview::{budget_jpegs, RESULT_IMAGE_BUDGET};
pub use sfm::{sfm_health, SfmHealth, SfmHealthOptions};
pub use video::{video_stats, VideoStats, VideoStatsOptions};
