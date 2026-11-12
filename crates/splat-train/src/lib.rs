#![recursion_limit = "256"]

pub mod config;
pub mod depth_prior;
pub mod eval;
#[cfg(not(target_family = "wasm"))]
pub mod float_export;
pub mod lod;
mod mixer;
pub mod msg;
pub mod neural;
pub mod multiview;
pub mod pose_refine;
pub mod trajectory;
pub mod train;

#[path = "optim.rs"]
mod adam_scaled;
mod mask_gate;
pub mod memory;
pub mod memory_cache;
mod blur_kernels;
mod housekeep;
mod multinomial;
mod qat;
mod quant;
mod quat_vec;
mod stats;

mod splat_init;

pub use splat_init::{RandomSplatsConfig, create_random_splats, to_init_splats};
