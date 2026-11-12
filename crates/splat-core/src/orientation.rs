//! Upside-down video (pixels stored rotated 180 with no metadata, e.g. a phone held inverted):
//! (videos only; photo sets keep the evidence and warning but are never auto-rotated)
//! inspect_input flags it from its samples (sky in the bottom third), and extract_frames then
//! rotates by 180 automatically in every pipeline. Backend-agnostic, so brush and spirula agree.

use crate::{Run, RunCtx};
use serde_json::{Value, json};
use splat_contracts::{Status, ToolResult};

/// Share of samples that must look upside down.
const FLIPPED_SHARE: f64 = 0.6;

/// (bottom - top) luma and (bottom - top) blue-minus-red over the outer thirds of a sample.
fn thirds(img: &image::RgbImage) -> Option<(f64, f64)> {
    let (w, h) = img.dimensions();
    if h < 6 || w == 0 {
        return None;
    }
    let band = |y0: u32, y1: u32| {
        let (mut luma, mut blue, mut n) = (0f64, 0f64, 0f64);
        for y in y0..y1 {
            for x in (0..w).step_by(2) {
                let p = img.get_pixel(x, y);
                luma += 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64;
                blue += p[2] as f64 - p[0] as f64;
                n += 1.0;
            }
        }
        (luma / n.max(1.0), blue / n.max(1.0))
    };
    let ((top_l, top_b), (bot_l, bot_b)) = (band(0, h / 3), band(h * 2 / 3, h));
    Some((bot_l - top_l, bot_b - top_b))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

/// Clip-level decision, deliberately conservative (a wrong flip is worse than a miss). Calibrated
/// on cambridge_heath_rd (stored upside down: bottom +68 luma, +5 bluer) against upright pegasus,
/// yagan, groei, copped and one more clip (bottom always less blue than the top: -4 to -92).
const MIN_BRIGHTER: f64 = 30.0;
const MIN_BLUER: f64 = 2.0;

/// Adds `visual_orientation` (and `sky_bottom_fraction`) to a successful inspect_input result when
/// the backend didn't judge it, from the result's sample images.
pub fn judge_inspect(run: &Run, mut r: ToolResult) -> ToolResult {
    if r.status != Status::Succeeded || r.metrics.get("visual_orientation").and_then(Value::as_str) == Some("likely_upside_down") {
        return r;
    }
    // A source that records its rotation (display matrix / EXIF) is decoded upright by ffmpeg; only
    // pixels stored upside down with no metadata (cambridge_heath_rd) need the content test.
    if r.metrics.get("rotation_metadata_present") == Some(&json!(true)) {
        return r;
    }
    let samples: Vec<image::RgbImage> = r
        .images
        .iter()
        .filter_map(|p| image::open(run.root().join(p)).ok())
        .map(|i| i.thumbnail(320, 320).to_rgb8())
        .collect();
    if samples.is_empty() {
        return r;
    }
    let thirds: Vec<(f64, f64)> = samples.iter().filter_map(thirds).collect();
    // Per-sample sky test (brush's inspect rule) and the clip-median brighter-and-bluer bottom.
    let share = thirds.iter().filter(|&&(l, b)| l > 15.0 && b > 8.0).count() as f64 / thirds.len().max(1) as f64;
    let (dl, db) = (median(thirds.iter().map(|t| t.0).collect()), median(thirds.iter().map(|t| t.1).collect()));
    r.metrics.entry("sky_bottom_fraction".into()).or_insert(json!(share));
    r.metrics.insert("orientation_evidence".into(), json!({"bottom_minus_top_luma": dl, "bottom_minus_top_blue": db}));
    if share > FLIPPED_SHARE || (dl > MIN_BRIGHTER && db > MIN_BLUER) {
        r.metrics.insert("visual_orientation".into(), json!("likely_upside_down"));
    }
    r
}

/// True when the latest inspect_input judged a video source upside down. Photo sets are never
/// rotated from content: truck84 (upright Tanks and Temples photos, no EXIF orientation, a bright
/// blue-grey pavement in the bottom third) was rotated 180 degrees on every run and scored 13.9 dB
/// held-out with 73/84 registered; upright, the same photos give 22.6 dB and 84/84. Cameras record
/// photo orientation in EXIF, and the cambridge calibration case is a video.
pub fn source_upside_down(run: &Run) -> bool {
    let Ok(state) = run.state() else { return false };
    state
        .operations
        .values()
        .filter_map(|op| serde_json::from_value::<ToolResult>(op.get("result")?.clone()).ok())
        .filter(|r| r.tool == "inspect_input" && r.status == Status::Succeeded)
        .last()
        .is_some_and(|r| {
            r.metrics.get("input_kind").and_then(Value::as_str) == Some("video")
                && r.metrics.get("visual_orientation").and_then(Value::as_str) == Some("likely_upside_down")
                && r.metrics.get("rotation_metadata_present") != Some(&json!(true))
        })
}
