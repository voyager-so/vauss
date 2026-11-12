//! Per-image metrics averaged over images (the 3DGS / gsplat convention). Port of eval/metrics.py.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::color::{correct_affine, correct_quadratic};
use crate::images::{Pair, Rgb, load_mask, load_rgb};
use crate::lpips::{Lpips, Net};
use crate::ssim::ssim;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColorCorrection {
    /// Zip-NeRF eval colour correction (quadratic+linear+bias, 5 iterations, clipped pixels ignored).
    Quadratic,
    Affine,
}

#[derive(Clone, Debug)]
pub struct MetricOptions {
    /// Compute LPIPS (VGG and Alex). If the ONNX models or runtime are unavailable the summary says
    /// why in `lpips_unavailable` and the lpips fields are absent; never estimated.
    pub lpips: bool,
    /// Directory holding lpips_vgg.onnx / lpips_alex.onnx (see lpips::LpipsModels::locate).
    pub models_dir: Option<PathBuf>,
    /// None = no cc_* variants.
    pub color_correction: Option<ColorCorrection>,
    /// Also run LPIPS on the colour-corrected variant (cc_lpips_*). Off halves the LPIPS work
    /// (CPU LPIPS dominates scoring time) while keeping cc_psnr / cc_ssim.
    pub cc_lpips: bool,
    /// Masks (white = keep) named like the GT stem. PSNR over kept pixels; SSIM/LPIPS full-frame
    /// on the masked images (same as metrics.py).
    pub mask_dir: Option<PathBuf>,
    /// Resize renders to GT size (bicubic) instead of failing on a size mismatch.
    pub resize: bool,
}

impl Default for MetricOptions {
    fn default() -> Self {
        Self {
            lpips: true,
            models_dir: None,
            color_correction: Some(ColorCorrection::Quadratic),
            cc_lpips: true,
            mask_dir: None,
            resize: false,
        }
    }
}

/// Same JSON shape as `eval/metrics.py --json`. Metric keys: psnr, ssim, lpips_vgg, lpips_alex,
/// lpips_vgg_3dgs (the 3DGS-code variant papers report; see Lpips::distance_3dgs) and the cc_
/// prefixed variants. per_image rows are {"image": stem, <metric>: value, ...}.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalSummary {
    pub gt_dir: String,
    pub render_dir: String,
    pub cc: Option<ColorCorrection>,
    pub n: usize,
    pub missing_renders: Vec<String>,
    pub device: String,
    pub seconds: f64,
    pub mean: BTreeMap<String, f64>,
    pub per_image: Vec<BTreeMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lpips_unavailable: Option<String>,
    /// e.g. the bounded CPU fallback (see lpips::CPU_MAX_EDGE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lpips_note: Option<String>,
    /// "full" or "cpu<=512" (see Lpips::scale); absent when LPIPS did not run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lpips_scale: Option<String>,
}

/// Score each pair, then average. Errors if `pairs` is empty or sizes mismatch without `resize`.
/// `missing` (GT stems with no render) is carried into the summary.
pub fn evaluate_pairs(
    pairs: &[Pair],
    missing: &[String],
    opts: &MetricOptions,
) -> Result<EvalSummary> {
    let Some(first) = pairs.first() else {
        bail!("no image pairs to evaluate")
    };
    let mut lpips_unavailable = None;
    let mut lpips = if opts.lpips {
        match Lpips::locate(opts.models_dir.as_deref()).and_then(|dir| Lpips::load(&dir, &NETS)) {
            Ok(l) => Some(l),
            Err(e) => {
                lpips_unavailable = Some(format!("{e:#}"));
                None
            }
        }
    } else {
        None
    };

    if let Some(l) = lpips.as_ref().filter(|l| l.scale() != "full") {
        eprintln!("warning: LPIPS scale {}: absolute LPIPS is not comparable to full-resolution or published numbers (paired deltas are)", l.scale());
    }
    let t0 = Instant::now();
    // LPIPS sessions are &mut and run one image at a time, so with LPIPS on, a bounded chunk of
    // images is scored in parallel and held for the LPIPS pass; without it everything is parallel.
    let chunk = if lpips.is_some() { 8 } else { pairs.len() };
    let mut rows: Vec<(String, BTreeMap<String, f64>)> = Vec::with_capacity(pairs.len());
    for group in pairs.chunks(chunk) {
        let keep_images = lpips.is_some();
        let scored: Vec<Scored> = group
            .par_iter()
            .map(|p| score(p, opts, keep_images))
            .collect::<Result<_>>()?;
        for mut s in scored {
            if let Some(l) = lpips.as_mut()
                && let Err(e) = add_lpips(l, &mut s)
            {
                lpips_unavailable = Some(format!("LPIPS failed on {}: {e:#}", s.stem));
                lpips = None;
                s.metrics.retain(|k, _| !k.contains("lpips_"));
                rows.iter_mut()
                    .for_each(|(_, m)| m.retain(|k, _| !k.contains("lpips_")));
            }
            // Heartbeat per scored view: CPU LPIPS keeps the GPU idle for minutes, and stall
            // watchdogs count output as liveness.
            let m = &s.metrics;
            let f = |k: &str| m.get(k).map_or_else(|| "-".to_string(), |v| format!("{v:.4}"));
            eprintln!("[eval] view {}/{} {} psnr {} lpips {} ({:.0} s)", rows.len() + 1, pairs.len(), s.stem,
                f("psnr"), f("lpips_vgg"), t0.elapsed().as_secs_f64());
            rows.push((s.stem, s.metrics));
        }
    }
    let seconds = (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0;

    let mean = rows[0]
        .1
        .keys()
        .map(|k| {
            (
                k.clone(),
                rows.iter().map(|(_, m)| m[k]).sum::<f64>() / rows.len() as f64,
            )
        })
        .collect();
    let per_image = rows
        .into_iter()
        .map(|(stem, metrics)| {
            let mut row: BTreeMap<String, serde_json::Value> =
                metrics.into_iter().map(|(k, v)| (k, v.into())).collect();
            row.insert("image".into(), stem.into());
            row
        })
        .collect();
    let parent = |p: &Path| {
        p.parent()
            .map(|d| d.display().to_string())
            .unwrap_or_default()
    };
    Ok(EvalSummary {
        gt_dir: parent(&first.gt),
        render_dir: parent(&first.render),
        cc: opts.color_correction,
        n: pairs.len(),
        missing_renders: missing.to_vec(),
        // PSNR/SSIM/cc always run on the CPU; this names where LPIPS ran.
        device: lpips.as_ref().map_or("cpu", |l| l.provider).into(),
        seconds,
        mean,
        per_image,
        lpips_note: lpips.as_ref().and_then(Lpips::note),
        lpips_scale: lpips.as_ref().map(Lpips::scale),
        lpips_unavailable,
    })
}

const NETS: [Net; 2] = [Net::Vgg, Net::Alex];

fn lpips_key(net: Net) -> &'static str {
    match net {
        Net::Vgg => "lpips_vgg",
        Net::Alex => "lpips_alex",
    }
}

/// One image's metrics, plus the (masked) images LPIPS still has to see.
struct Scored {
    stem: String,
    metrics: BTreeMap<String, f64>,
    gt: Option<Rgb>,
    variants: Vec<(&'static str, Rgb)>,
}

fn score(pair: &Pair, opts: &MetricOptions, keep_images: bool) -> Result<Scored> {
    let stem = &pair.stem;
    let gt = load_rgb(&pair.gt, None)?;
    let size = (gt.width, gt.height);
    let render = if opts.resize {
        load_rgb(&pair.render, Some(size))?
    } else {
        let r = load_rgb(&pair.render, None)?;
        if (r.width, r.height) != size {
            bail!(
                "size mismatch for {stem}: gt {}x{} render {}x{} (use --resize)",
                size.0,
                size.1,
                r.width,
                r.height
            );
        }
        r
    };
    let mask = match &opts.mask_dir {
        Some(dir) => find_mask(dir, stem)?
            .map(|p| load_mask(&p, size))
            .transpose()?,
        None => None,
    };

    let corrected = opts.color_correction.map(|cc| match cc {
        ColorCorrection::Quadratic => correct_quadratic(&render, &gt, mask.as_deref()),
        ColorCorrection::Affine => correct_affine(&render, &gt, mask.as_deref()),
    });
    let variants: Vec<(&'static str, Rgb)> = std::iter::once(("", render))
        .chain(corrected.map(|c| ("cc_", c)))
        .collect();

    // Like metrics.py: masked pixels are zeroed in both images; PSNR counts kept pixels only.
    let (gt, variants) = match &mask {
        Some(m) => (
            apply_mask(gt, m),
            variants
                .into_iter()
                .map(|(pre, r)| (pre, apply_mask(r, m)))
                .collect(),
        ),
        None => (gt, variants),
    };
    let mut metrics = BTreeMap::new();
    for (prefix, r) in &variants {
        metrics.insert(format!("{prefix}psnr"), psnr(r, &gt, mask.as_deref()));
        metrics.insert(format!("{prefix}ssim"), ssim(r, &gt));
    }
    Ok(Scored {
        stem: stem.clone(),
        metrics,
        gt: keep_images.then_some(gt),
        variants: if keep_images { variants.into_iter().filter(|(pre, _)| pre.is_empty() || opts.cc_lpips).collect() } else { Vec::new() },
    })
}

fn add_lpips(lpips: &mut Lpips, s: &mut Scored) -> Result<()> {
    let gt = s.gt.take().context("images not kept for LPIPS")?;
    for (prefix, r) in std::mem::take(&mut s.variants) {
        for net in NETS {
            s.metrics.insert(
                format!("{prefix}{}", lpips_key(net)),
                lpips.distance(net, &r, &gt)?,
            );
        }
        s.metrics.insert(format!("{prefix}lpips_vgg_3dgs"), lpips.distance_3dgs(Net::Vgg, &r, &gt)?);
    }
    Ok(())
}

/// metrics.py takes the first `<mask_dir>/<stem>.*`; sorted here so the pick is deterministic.
fn find_mask(dir: &Path, stem: &str) -> Result<Option<PathBuf>> {
    let prefix = format!("{stem}.");
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading mask dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&prefix))
        })
        .collect();
    hits.sort();
    Ok(hits.into_iter().next())
}

fn apply_mask(mut im: Rgb, mask: &[f32]) -> Rgb {
    let n = mask.len();
    for (i, v) in im.data.iter_mut().enumerate() {
        *v *= mask[i % n];
    }
    im
}

/// -10 log10(MSE) over all pixels/channels, or over kept pixels when `mask` (h*w of 0/1) is given.
/// MSE is clamped at 1e-10.
pub fn psnr(render: &crate::images::Rgb, gt: &crate::images::Rgb, mask: Option<&[f32]>) -> f64 {
    let n = render.width * render.height;
    let sq = |(i, (&r, &g)): (usize, (&f32, &f32))| {
        let m = mask.map_or(1.0, |m| m[i % n]) as f64;
        let d = r as f64 * m - g as f64 * m;
        d * d
    };
    let sse: f64 = render
        .data
        .par_iter()
        .zip(&gt.data)
        .enumerate()
        .map(sq)
        .sum();
    let count = match mask {
        Some(m) => (m.iter().map(|&v| v as f64).sum::<f64>() * 3.0).max(1.0),
        None => (3 * n) as f64,
    };
    -10.0 * (sse / count).max(1e-10).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(w: usize, h: usize, seed: u32) -> Rgb {
        let mut s = seed;
        let data = (0..3 * w * h)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as f32 / 255.0
            })
            .collect();
        Rgb {
            width: w,
            height: h,
            data,
        }
    }

    #[test]
    fn identical_images_cap_at_100_db_and_ssim_one() {
        let a = noise(40, 30, 1);
        assert_eq!(psnr(&a, &a, None), 100.0);
        assert!((ssim(&a, &a) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn psnr_known_value_and_mask() {
        let a = Rgb {
            width: 2,
            height: 1,
            data: vec![0.0; 6],
        };
        let mut b = a.clone();
        b.data = vec![0.1, 0.0, 0.1, 0.0, 0.1, 0.0]; // error only in pixel 0
        assert!((psnr(&b, &a, None) - 23.010299956639813).abs() < 1e-5); // mse 0.005
        assert!((psnr(&b, &a, Some(&[1.0, 0.0])) - 20.0).abs() < 1e-5); // mse 0.01 over kept
        assert_eq!(psnr(&b, &a, Some(&[0.0, 1.0])), 100.0);
    }
}
