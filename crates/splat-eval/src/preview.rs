//! Images handed to a driver share one byte budget per tool result (MCP / Claude SDK reject
//! results over ~1 MB), so they are downscaled and re-encoded as JPEG to fit.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;

/// Default raw JPEG total for one tool result. MCP sends images base64 (x4/3), so this is ~750 KB
/// on the wire, leaving headroom under the ~1 MB limit for the JSON.
pub const RESULT_IMAGE_BUDGET: usize = 560_000;
const START_EDGE: u32 = 1024;
const MIN_EDGE: u32 = 128;
const QUALITY: u8 = 80;

/// Write `<out_dir>/<stem>.jpg` for each input so that all outputs together stay within
/// `max_total_bytes`: each image gets an equal share, and its long edge shrinks (from 1024 px, by
/// 3/4 per step) until it fits. Errors if the share is too small even at 128 px.
pub fn budget_jpegs(paths: &[PathBuf], out_dir: &Path, max_total_bytes: usize) -> Result<Vec<PathBuf>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let share = max_total_bytes / paths.len();
    let mut out = Vec::with_capacity(paths.len());
    for (i, path) in paths.iter().enumerate() {
        let img = image::open(path).with_context(|| format!("reading {}", path.display()))?.into_rgb8();
        let mut edge = START_EDGE.min(img.width().max(img.height()));
        let bytes = loop {
            let small = if img.width().max(img.height()) > edge {
                image::DynamicImage::ImageRgb8(img.clone()).resize(edge, edge, FilterType::Triangle).into_rgb8()
            } else {
                img.clone()
            };
            let mut buf = Vec::new();
            JpegEncoder::new_with_quality(&mut buf, QUALITY).encode_image(&small)?;
            if buf.len() <= share || edge <= MIN_EDGE {
                break buf;
            }
            edge = (edge * 3 / 4).max(MIN_EDGE);
        };
        ensure!(bytes.len() <= share, "{}: {} bytes at {MIN_EDGE} px exceeds its {share}-byte share", path.display(), bytes.len());
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        // Prefix with the index so equal stems from different dirs don't collide.
        let dst = out_dir.join(format!("{i:02}-{stem}.jpg"));
        std::fs::write(&dst, &bytes).with_context(|| format!("writing {}", dst.display()))?;
        out.push(dst);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noisy_images_fit_one_budget() {
        let dir = std::env::temp_dir().join(format!("splat-eval-preview-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut seed = 1u32;
        let paths: Vec<PathBuf> = (0..8)
            .map(|i| {
                let img = image::RgbImage::from_fn(1920, 1080, |_, _| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    image::Rgb([(seed >> 24) as u8, (seed >> 16) as u8, (seed >> 8) as u8])
                });
                let p = dir.join(format!("in{i}.png"));
                img.save(&p).unwrap();
                p
            })
            .collect();
        let out = budget_jpegs(&paths, &dir.join("out"), RESULT_IMAGE_BUDGET).unwrap();
        let total: u64 = out.iter().map(|p| std::fs::metadata(p).unwrap().len()).sum();
        assert_eq!(out.len(), 8);
        assert!(total as usize <= RESULT_IMAGE_BUDGET, "{total}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
