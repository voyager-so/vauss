//! Golden vectors for the effect evaluators.
//!
//! `splat-effects-golden --write [path]` regenerates `crates/splat-effects/golden.json` from the
//! CPU reference; `--check [path]` re-evaluates on the CPU and compares; `--check-gpu cpu|gpu
//! [path]` runs the WGSL evaluator (`cpu` = wgpu software adapter, `gpu` = default adapter, needs
//! a GPU) and compares. Positions/alpha/dc/rot to 1e-4, Σ' and s' relative 1e-4 (CPU) / 2e-3 (GPU).
//! The GPU check gates on the exact-mode fields; fast-mode q' is reported only.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden.json");
    let mode = args.first().map(String::as_str).unwrap_or("");
    let path_arg = if mode == "--check-gpu" { args.get(2) } else { args.get(1) };
    let path = path_arg.map_or(default, PathBuf::from);
    let (doc, tol) = match mode {
        "--write" | "--check" => (splat_effects::golden::generate(), 1e-4),
        #[cfg(feature = "gpu")]
        "--check-gpu" => {
            let kind = args.get(1).map_or("cpu", String::as_str);
            let device = splat_effects::gpu::device(kind);
            let t0 = std::time::Instant::now();
            let doc = splat_effects::golden::generate_with(&|rest, g, t, opts| {
                splat_effects::gpu::evaluate_via_gpu(rest, g, t, opts, &device).expect("gpu eval")
            });
            println!("gpu ({kind}) golden evaluated in {:.2}s", t0.elapsed().as_secs_f64());
            (doc, 2e-3)
        }
        _ => {
            println!("usage: splat-effects-golden --write|--check [golden.json] | --check-gpu cpu|gpu [golden.json]");
            std::process::exit(2);
        }
    };
    if mode == "--write" {
        std::fs::write(&path, serde_json::to_string(&doc).expect("json")).expect("write golden");
        println!("wrote {}", path.display());
        return;
    }
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("read golden")).expect("parse golden");
    use splat_effects::golden::{ALL_FIELDS, EXACT_FIELDS, compare};
    if mode == "--check-gpu" {
        match compare(&want, &doc, tol, &["fast_rot"]) {
            Ok(w) => println!("fast-mode q' (not gated): {w:?}"),
            Err(e) => println!("fast-mode q' (not gated; polar is ill-defined at singular J): {}", e.lines().next().unwrap_or("")),
        }
    }
    let fields = if mode == "--check-gpu" { EXACT_FIELDS } else { ALL_FIELDS };
    match compare(&want, &doc, tol, fields) {
        Ok(worst) => println!("golden OK ({}, tol {tol:.0e}): max |d| {worst:?}", path.display()),
        Err(e) => {
            println!("golden MISMATCH: {e}");
            std::process::exit(1);
        }
    }
}
