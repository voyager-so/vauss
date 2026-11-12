//! Spark `.RAD` level-of-detail package: Spark 2.x's `build-lod` pipeline (bhatt-lod
//! tree, base 1.75, 64K-splat chunks, coarse-first; f16 colour and scales), built in-process from the exported splats with
//! Spark's own MIT `spark-lib` pinned to the v2.2.0 release the viewer ships. The viewer pages it
//! over HTTP Range requests: first paint from the root chunks, detail streamed by view priority,
//! GPU memory bounded by Spark's splat pager. SPZ (+ sidecars) stays the download format.

use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use spark_lib::bhatt_lod;
use spark_lib::chunk_tree;
use spark_lib::decoder::{ChunkReceiver, MultiDecoder};
use spark_lib::gsplat::GsplatArray;
use spark_lib::rad::{RadEncoder, RadRgbEncoding, RadScalesEncoding};
use spark_lib::tsplat::{Tsplat, TsplatArray};

use crate::ply::{Gaussians, write_ply};

/// bhatt-lod merge base: `build-lod --quality` default.
const LOD_BASE: f32 = 1.75;

/// Write `<dir>/<stem>.rad` from `g` and return its stats for scene.json.
pub fn write_rad(g: &Gaussians, dir: &Path, stem: &str) -> Result<Value> {
    let t0 = Instant::now();
    // Through a PLY so Spark's own decoder applies its conventions (sigmoid opacity, exp scales,
    // SH0 -> rgb, SH layout) exactly as `build-lod scene.ply` would.
    let tmp = tempfile::Builder::new().suffix(".ply").tempfile_in(dir)?;
    write_ply(g, tmp.path())?;
    let bytes = std::fs::read(tmp.path())?;
    drop(tmp);
    let mut decoder = MultiDecoder::new(GsplatArray::new(), None, Some("scene.ply"));
    decoder.push(&bytes).context("spark-lib: PLY decode")?;
    decoder.finish().context("spark-lib: PLY decode")?;
    drop(bytes);
    let mut splats = decoder.into_splats();
    let n_in = splats.len();
    // build-lod drops splats it can't place in the tree; the export already cropped near-transparent ones.
    splats.retain(|s| s.opacity() > 0.0 && s.max_scale() > 0.0 && s.quaternion().is_finite() && s.quaternion().length() > 0.0 && s.center().is_finite());
    let quiet = |_: &str| {};
    bhatt_lod::compute_lod_tree(&mut splats, LOD_BASE, quiet);
    let n_lod = splats.len();
    chunk_tree::chunk_tree(&mut splats, 0, quiet);
    splats.encode_lod_opacity();
    let file = format!("{stem}.rad");
    let path = dir.join(&file);
    // Written beside and renamed into place: a viewer may already be serving this package.
    let part = dir.join(format!("{file}.part-{}", std::process::id()));
    let mut w = BufWriter::new(std::fs::File::create(&part)?);
    // Encoding: f16 colour and log scales instead of build-lod's defaults (8-bit colour over the
    // 1st-99th percentile range, which clips the brightest/darkest 1%; 8-bit log scales). Garden on
    // an M4 at full detail vs the trainer: 43.7 -> 47.8 dB (SPZ 48.6) for +7% bytes. SH and
    // orientation stay default: f16 there renders wrong in Spark 2.2.0 (30.9 dB).
    RadEncoder::new(splats)
        .with_rgb_encoding(RadRgbEncoding::F16)
        .with_scales_encoding(RadScalesEncoding::LnF16)
        .encode(&mut w)
        .context("spark-lib: RAD encode")?;
    w.flush()?;
    drop(w);
    std::fs::rename(&part, &path).with_context(|| format!("write {}", path.display()))?;
    let bytes = std::fs::metadata(&path)?.len();
    Ok(json!({"file": file, "bytes": bytes, "splats_in": n_in, "splats_lod": n_lod, "lod_base": LOD_BASE,
        "encoder": "spark-lib 2.2.0 bhatt-lod, rgb f16, scales ln-f16", "elapsed_s": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0}))
}
