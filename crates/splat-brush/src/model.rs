//! COLMAP sparse model helpers on top of splat-eval's reader.

use std::path::{Path, PathBuf};

pub use splat_eval::Model;

pub fn read_model(dir: &Path) -> anyhow::Result<Model> {
    splat_eval::read_model(dir)
}

/// Registered image names of the model in `dir`, sorted.
pub fn names(dir: &Path) -> anyhow::Result<Vec<String>> {
    Ok(read_model(dir)?.registered_names())
}

/// Sparse model directories under `sparse/`, largest (most registered images) first.
pub fn models_by_size(sparse: &Path) -> anyhow::Result<Vec<(PathBuf, usize)>> {
    let mut out = vec![];
    if !sparse.is_dir() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(sparse)? {
        let p = entry?.path();
        if p.join("images.bin").exists() || p.join("images.txt").exists() {
            let n = read_model(&p)?.images.len();
            out.push((p, n));
        }
    }
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(out)
}
