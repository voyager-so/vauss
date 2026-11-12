//! Programmable splat effects: effect graph v1 and its evaluators.
//!
//! # API (kept small and stable; the CLI and the web viewer build on it)
//!
//! ```ignore
//! let rest: Splats = splat_webexport::ply::read_ply(path)?;          // PLY parameter space
//! let graph = Graph::from_json(&std::fs::read_to_string(fx)?)?;      // parses + validates
//! let mut opts = EvalOpts::default();                                // cov Exact, up +Y
//! for m in graph.object_masks() { opts.objects.insert(m.key(), weights_for(m)); }
//! let frame: Frame = evaluate(&rest, &graph, t, &opts)?;            // one t
//! let prep = Prepared::new(&rest, &graph, &opts)?;                   // many t: prep.frame(t)
//! write_ply(&frame.to_splats(&rest), out)?;
//! ```
//!
//! - `Splats` is `splat_webexport::ply::Gaussians`: pos, rot (w,x,y,z), log scale, opacity logit,
//!   SH DC. `Frame` has the same layout and keeps the splat count and order. Invalid splats are
//!   counted (`frame.invalid`) and reset to rest with opacity [`DROPPED_LOGIT`].
//! - `frame.weight` is the max node weight per splat (mask pass); `frame.selected` counts
//!   non-zero ones; [`displacement`] gives `|Δμ|`.
//! - Object masks are resolved by the caller from the selection npy files into per-splat
//!   weights keyed by [`ObjectMask::key`]; a mask's `box` is the fallback without weights.
//! - GPU (`gpu` feature, default): [`gpu::GpuEffect::new`]`(&brush_splats, &graph, &opts)` once,
//!   then `.frame(&brush_splats, t)` -> [`gpu::GpuFrame`] whose `splats` feed
//!   `brush_render::render_splats` with no host copies; [`gpu::upload`] / [`gpu::download`] convert.
//! - Randoms: [`hash`] (pcg, shared with WGSL/GLSL). Math: [`eval`] module docs.
//! - `golden.json` is the cross-evaluator reference (200 hashed splats x 4 graphs x 5 times):
//!   `cargo run -p splat-effects --bin splat-effects-golden -- --check|--write`, or
//!   [`golden::generate`] / [`golden::compare`] from a CLI.

pub mod eval;
pub mod golden;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod graph;
pub mod hash;

pub use eval::{CovMode, DROPPED_LOGIT, EvalError, EvalOpts, Frame, Prepared, SH_VIEW_ROTATION, displacement, evaluate, scene_radius};
pub use graph::{Curve, Graph, GraphError, Mask, Node, ObjectMask, Op, Window};
pub use splat_webexport::ply::Gaussians as Splats;
