//! Source probing, capture-quality stats and frame selection (ffmpeg/ffprobe subprocesses).

use splat_eval::frameplan;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{Context, bail};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage, imageops::FilterType};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::RunCtx;

use crate::BrushConfig;
use crate::util::*;

/// Mean luma below this is a black/fade frame.
const BLACK_LUMA: f32 = 12.0;
/// Mean abs grey difference (0-255 at ~160 px) below which a window duplicates the last kept frame.
const DUP_DIFF: f32 = 1.5;
/// Long edge of the photo copies SfM reads (`sfm_images/`) when the photos exceed `max_edge`.
/// ETH3D (38-44 photos, pose AUC@3 vs laser GT): 1600-px frames lost 3 of 14 frames on pipes every
/// time (.554); the same SfM on 2400-px frames registered 14/14 (.926) and lifted courtyard
/// .931 -> .945, delivery_area .960 -> .969, terrains .926 -> .935, for +30-40% SfM wall.
/// COLMAP still extracts SIFT at its own max_image_size; the 1600-px loss came from our Lanczos +
/// JPEG re-encode at 1600. `SPLAT_SFM_MAX_EDGE` overrides (0: off).
pub const SFM_EDGE: u32 = 2400;
/// Photo copies resized and encoded at once in the background while the next photos decode.
const SFM_WRITERS: usize = 4;

fn sfm_edge() -> u32 {
    std::env::var("SPLAT_SFM_MAX_EDGE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(SFM_EDGE)
}

fn rotate(img: RgbImage, deg: u32) -> RgbImage {
    match deg % 360 {
        90 => image::imageops::rotate90(&img),
        180 => image::imageops::rotate180(&img),
        270 => image::imageops::rotate270(&img),
        _ => img,
    }
}

#[derive(Debug, Clone)]
pub struct VideoInfo {
    /// Display size, after rotation metadata.
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frame_count: u64,
    pub duration_s: f64,
    pub codec: String,
    pub rotation: i32,
    pub video_streams: usize,
}

fn parse_rate(r: Option<&str>) -> f64 {
    let Some((n, d)) = r.and_then(|r| r.split_once('/')) else { return 0.0 };
    match (n.parse::<f64>(), d.parse::<f64>()) {
        (Ok(n), Ok(d)) if d != 0.0 => n / d,
        _ => 0.0,
    }
}

pub fn probe(cfg: &BrushConfig, video: &Path) -> anyhow::Result<VideoInfo> {
    let out = run_capture(Command::new(&cfg.ffprobe).args(["-v", "error", "-show_streams", "-show_format", "-of", "json"]).arg(video))?;
    let data: Value = serde_json::from_slice(&out.stdout)?;
    let streams: Vec<&Value> = data["streams"]
        .as_array()
        .map(|s| s.iter().filter(|s| s["codec_type"] == "video").collect())
        .unwrap_or_default();
    let Some(s) = streams.first() else { return Err(invalid("no_video_track: source has no video stream")) };
    // WebM often reports a bogus avg_frame_rate (cambridge_heath_rd: 30000/1); trust r_frame_rate then.
    let avg = parse_rate(s["avg_frame_rate"].as_str());
    let r = parse_rate(s["r_frame_rate"].as_str());
    let fps = if avg > 0.0 && avg <= 1.5 * if r > 0.0 { r } else { avg } { avg } else { r };
    let num = |v: &Value| v.as_str().and_then(|x| x.parse::<f64>().ok()).or_else(|| v.as_f64());
    let duration = num(&s["duration"]).or_else(|| num(&data["format"]["duration"])).unwrap_or(0.0);
    let mut rotation = num(&s["tags"]["rotate"]).unwrap_or(0.0) as i32;
    if let Some(sides) = s["side_data_list"].as_array() {
        for side in sides {
            if let Some(r) = side["rotation"].as_f64() {
                rotation = r as i32;
            }
        }
    }
    let (mut w, mut h) = (s["width"].as_u64().unwrap_or(0) as u32, s["height"].as_u64().unwrap_or(0) as u32);
    if rotation.rem_euclid(180) == 90 {
        std::mem::swap(&mut w, &mut h);
    }
    let frame_count = num(&s["nb_frames"]).map(|n| n as u64).filter(|&n| n > 0).unwrap_or((duration * fps).round() as u64);
    Ok(VideoInfo {
        width: w,
        height: h,
        fps,
        frame_count,
        duration_s: duration,
        codec: s["codec_name"].as_str().unwrap_or("?").into(),
        rotation,
        video_streams: streams.len(),
    })
}

pub fn is_video(source: &Path) -> bool {
    source.is_file() && !is_image(source)
}

fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let (mut x, mut y) = (a.as_bytes(), b.as_bytes());
    while !x.is_empty() && !y.is_empty() {
        if x[0].is_ascii_digit() && y[0].is_ascii_digit() {
            let nx = x.iter().take_while(|c| c.is_ascii_digit()).count();
            let ny = y.iter().take_while(|c| c.is_ascii_digit()).count();
            let ax = &x[..nx]; let by = &y[..ny];
            let ax = &ax[ax.iter().take_while(|&&c| c == b'0').count()..];
            let by = &by[by.iter().take_while(|&&c| c == b'0').count()..];
            let order = ax.len().cmp(&by.len()).then_with(|| ax.cmp(by));
            if order != Ordering::Equal { return order; }
            x = &x[nx..]; y = &y[ny..];
        } else {
            let order = x[0].cmp(&y[0]);
            if order != Ordering::Equal { return order; }
            x = &x[1..]; y = &y[1..];
        }
    }
    x.len().cmp(&y.len()).then_with(|| a.cmp(b))
}

fn scaled_size(w: u32, h: u32, max_edge: u32) -> (u32, u32) {
    let s = (max_edge as f64 / w.max(h) as f64).min(1.0);
    let even = |x: u32| (((x as f64 * s) / 2.0).round() as u32 * 2).max(2);
    (even(w), even(h))
}

/// Decode a video once as scaled RGB frames: (source frame index, frame).
fn decode_video(
    cfg: &BrushConfig,
    video: &Path,
    size: (u32, u32),
    filter_prefix: &str,
    time_range: Option<(f64, f64)>,
    mut each: impl FnMut(u64, RgbImage) -> anyhow::Result<()>,
) -> anyhow::Result<u64> {
    let (w, h) = size;
    let mut cmd = Command::new(&cfg.ffmpeg);
    cmd.args(["-v", "error", "-nostdin", "-threads", "4", "-filter_threads", "4"]);
    if let Some((a, b)) = time_range {
        cmd.args(["-ss", &format!("{a:.3}"), "-to", &format!("{b:.3}")]);
    }
    cmd.arg("-i").arg(video);
    cmd.args(["-map", "0:v:0", "-vf", &format!("{filter_prefix}scale={w}:{h}:flags=area"), "-fps_mode", "passthrough"]);
    cmd.args(["-threads", "4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().context("spawning ffmpeg")?;
    let mut stdout = child.stdout.take().expect("piped");
    let stderr = child.stderr.take().expect("piped");
    let err_reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = { stderr }.read_to_string(&mut s);
        s
    });
    let mut buf = vec![0u8; (w * h * 3) as usize];
    let mut i = 0u64;
    let outcome = loop {
        match stdout.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break Ok(()),
            Err(e) => break Err(e.into()),
        }
        if let Err(e) = each(i, RgbImage::from_raw(w, h, buf.clone()).expect("sized buffer")) {
            break Err(e);
        }
        i += 1;
    };
    drop(stdout);
    if outcome.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let err = err_reader.join().unwrap_or_default();
    outcome?;
    if !status.success() && i == 0 {
        bail!("ffmpeg failed on {}: {}", video.display(), err.trim());
    }
    Ok(i)
}

fn grey(img: &RgbImage, step: u32) -> (u32, u32, Vec<f32>) {
    let (w, h) = ((img.width() / step).max(1), (img.height() / step).max(1));
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x * step, y * step);
            out.push((p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0);
        }
    }
    (w, h, out)
}

/// Variance of the 4-neighbour Laplacian at ~640 px (resolution independent).
pub fn sharpness(img: &RgbImage) -> f32 {
    let step = (img.width().max(img.height()) / 640).max(1);
    let (w, h, g) = grey(img, step);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let (w, h) = (w as usize, h as usize);
    let mut vals = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = g[y * w + x];
            vals.push(g[y * w + x - 1] + g[y * w + x + 1] + g[(y - 1) * w + x] + g[(y + 1) * w + x] - 4.0 * c);
        }
    }
    variance(&vals)
}

fn variance(v: &[f32]) -> f32 {
    let n = v.len().max(1) as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n) as f32
}

fn thumb(img: &RgbImage) -> Vec<f32> {
    grey(img, (img.width().max(img.height()) / 160).max(1)).2
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

fn percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    let t = (rank - lo as f64) as f32;
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

/// Same keys and definitions as eval/video_stats.py `summarize_frames`, plus the
/// sky-at-the-bottom upside-down heuristic (bottom third brighter by >15 and bluer than the top).
fn summarize(frames: &[RgbImage]) -> serde_json::Map<String, Value> {
    let mut lap = Vec::new();
    let mut lum = Vec::new();
    let mut black = 0usize;
    let mut top_clipped = 0f64;
    let mut top_total = 0f64;
    let mut flipped = 0usize;
    let mut sky = 0usize;
    let mut greys: Vec<Vec<f32>> = Vec::new();
    for f in frames {
        lap.push(sharpness(f));
        let (w, h, g) = grey(f, (f.width().max(f.height()) / 640).max(1));
        lum.push(g.iter().sum::<f32>() / g.len() as f32);
        if g.iter().filter(|&&v| v < 16.0).count() as f32 / g.len() as f32 > 0.98 {
            black += 1;
        }
        let third = (h / 3) as usize * w as usize;
        top_clipped += g[..third].iter().filter(|&&v| v >= 236.0).count() as f64;
        top_total += third as f64;
        let bottom = &g[g.len() - third..];
        let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len().max(1) as f32;
        // Mean (blue - red) per pixel over a band of rows: sky is clearly blue-shifted.
        let blue = |y0: u32, y1: u32| {
            let mut d = 0f64;
            for y in y0..y1 {
                for x in 0..f.width() {
                    let p = f.get_pixel(x, y);
                    d += p[2] as f64 - p[0] as f64;
                }
            }
            d / ((y1 - y0) as f64 * f.width() as f64).max(1.0)
        };
        let fh = f.height();
        let (blue_bottom, blue_top) = (blue(fh * 2 / 3, fh), blue(0, fh / 3));
        if mean(bottom) - mean(&g[..third]) > 15.0 && blue_bottom > 8.0 && blue_bottom > blue_top + 8.0 {
            flipped += 1;
        }
        // Color + brightness together avoid treating clipped white indoor ceilings as sky.
        // Check both ends because orientation correction happens after inspection.
        if (blue_top > 8.0 && mean(&g[..third]) > 100.0)
            || (blue_bottom > 8.0 && mean(bottom) > 100.0)
        {
            sky += 1;
        }
        greys.push(thumb(f));
    }
    let static_n = greys.windows(2).filter(|p| mean_abs_diff(&p[0], &p[1]) < 1.0).count();
    let mut sorted = lap.clone();
    sorted.sort_by(f32::total_cmp);
    let median = percentile(&sorted, 50.0);
    let p10 = percentile(&sorted, 10.0);
    let (lmin, lmax) = lum.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let n = frames.len().max(1) as f64;
    let mut m = serde_json::Map::new();
    m.insert("sample_count".into(), json!(frames.len()));
    m.insert("sharpness_median".into(), json!(median));
    m.insert("sharpness_p10".into(), json!(p10));
    m.insert("sharpness_p10_over_median".into(), json!(p10 / median.max(1e-6)));
    m.insert("luminance_min".into(), json!(lmin));
    m.insert("luminance_max".into(), json!(lmax));
    m.insert("luminance_range".into(), json!(lmax - lmin));
    m.insert("black_frac".into(), json!(black as f64 / n));
    m.insert("top_clipped_fraction".into(), json!(top_clipped / top_total.max(1.0)));
    let static_frac = (frames.len() > 1).then(|| static_n as f64 / (frames.len() - 1) as f64);
    m.insert("static_fraction".into(), json!(static_frac));
    m.insert("static_frac".into(), json!(static_frac));
    m.insert("sky_sample_fraction".into(), json!(sky as f64 / n));
    m.insert("sky_bottom_fraction".into(), json!(flipped as f64 / n));
    m
}

/// Load an image with its EXIF orientation applied.
pub fn load_oriented(path: &Path) -> anyhow::Result<(DynamicImage, u8)> {
    if splat_eval::images::is_heif(path) {
        // HEIC/HEIF via ffmpeg (iPhone tile grids stitched), already upright.
        return Ok((splat_eval::images::load_heif(path)?.0, 1));
    }
    let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orientation = decoder.orientation()?;
    let exif = orientation.to_exif();
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok((img, exif))
}

/// Where a run keeps its one analysis pass over the capture video (never staged to the cloud).
const VIDEO_ANALYSIS: &str = "video-analysis";
/// Long edge of the analysis samples kept as quick-SfM candidates: at least the quick SfM's frames
/// (the pipeline's FAST_EDGE, 720 px, a `preview` extraction below the 960 px floor).
const CANDIDATE_EDGE: u32 = 960;

/// inspect_input's decode of a video, shared with both extract_frames calls so a `vauss run` decodes
/// the capture twice (this and the full extraction) instead of five times (inspect, and a motion pass
/// plus an extraction pass for each of the quick and full frames). showbus (AV1 1080p60, 78 s) on a
/// loaded host: inspect 94 s, quick extract 212 s, full extract 289 s before any SfM ran.
#[derive(serde::Serialize, serde::Deserialize)]
struct VideoAnalysis {
    /// The capture it was made from (path, size, mtime, probe).
    key: Value,
    /// Candidate frame size (ANALYSIS_FPS samples at CANDIDATE_EDGE, files `s<index>.jpg`).
    size: (u32, u32),
    track: Vec<(f64, f64)>,
    /// Laplacian sharpness of each sample.
    sharpness: Vec<f32>,
    /// Last cached candidate. The motion track and inspection statistics still cover the whole
    /// capture; only JPEGs beyond the default preview prefix are omitted.
    candidates_through: u64,
    /// Source frame indices the quick extraction kept: the full extraction adds those that fall in
    /// sparse stretches of its plan (frameplan::ExtractOptions::fill).
    #[serde(default)]
    quick_picks: Vec<u64>,
    #[serde(default)]
    tone_mapped: Option<String>,
}

/// Quick-frame anchors also exist for HDR captures, which intentionally skip VideoAnalysis.
/// Keep their indices separately so the full extraction can share the preview's cameras.
#[derive(serde::Serialize, serde::Deserialize)]
struct QuickPicks {
    key: Value,
    rotate_deg: u32,
    indices: Vec<u64>,
}

impl QuickPicks {
    fn load(run: &dyn RunCtx, source: &Path, info: &VideoInfo, rotate_deg: u32) -> Option<Vec<u64>> {
        let picks: Self = read_json(&VideoAnalysis::dir(run).join("quick-picks.json")).ok()?;
        (picks.key == VideoAnalysis::key(source, info) && picks.rotate_deg == rotate_deg).then_some(picks.indices)
    }

    fn save(run: &dyn RunCtx, source: &Path, info: &VideoInfo, rotate_deg: u32, indices: Vec<u64>) -> anyhow::Result<()> {
        let dir = VideoAnalysis::dir(run);
        std::fs::create_dir_all(&dir)?;
        write_json(&dir.join("quick-picks.json"), &Self { key:VideoAnalysis::key(source, info), rotate_deg, indices })
    }
}

/// The whole capture's black-bar crop ([`frameplan::detect_video_bars`]), measured once per run.
/// Its 16 seeks each decode from the preceding keyframe: on arkit-41254538 (4 s GOPs) that was most
/// of the quick extraction's 10.9 s (no frames decoded) and was repeated by the full extraction.
/// inspect_input measures it alongside its analysis decode; both extractions read it back.
#[derive(serde::Serialize, serde::Deserialize)]
struct VideoBars {
    key: Value,
    crop: Option<frameplan::Crop>,
}

impl VideoBars {
    fn path(dir: &Path) -> PathBuf {
        dir.join("bars.json")
    }

    /// Detect and record the crop (the same arguments as extract_frames without a time range). The
    /// record is best effort: without it the next extraction detects again.
    fn detect(ffmpeg: &Path, source: &Path, info: &VideoInfo, dir: &Path) -> anyhow::Result<Option<frameplan::Crop>> {
        let crop = frameplan::detect_video_bars(ffmpeg, source, None, info.duration_s, info.width, info.height)?;
        let record = std::fs::create_dir_all(dir).map_err(anyhow::Error::from)
            .and_then(|_| write_json(&Self::path(dir), &Self { key: VideoAnalysis::key(source, info), crop }));
        if let Err(error) = record {
            eprintln!("black-bar crop not recorded: {error:#}");
        }
        Ok(crop)
    }

    /// The recorded crop of `source` (`Some(None)`: it has no bars), if one matches it.
    fn load(dir: &Path, source: &Path, info: &VideoInfo) -> Option<Option<frameplan::Crop>> {
        let bars: Self = read_json(&Self::path(dir)).ok()?;
        (bars.key == VideoAnalysis::key(source, info)).then_some(bars.crop)
    }
}

impl VideoAnalysis {
    fn key(source: &Path, info: &VideoInfo) -> Value {
        let meta = std::fs::metadata(source).ok();
        let mtime_ns = meta.as_ref().and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .and_then(|d| u64::try_from(d.as_nanos()).ok());
        // JSON float parsing can change an epoch timestamp by an ULP, silently invalidating the
        // cache just written by inspection. Fingerprints use exact integers, including the probe's
        // float bit patterns. The cache version requires sharpness across the whole timeline,
        // while candidate JPEG coverage remains bounded to the preview prefix.
        // Main's BT.2390 tone mapper changes HDR candidate pixels; reject older analysis caches.
        json!({"motion": 7, "source": source.canonicalize().ok(), "len": meta.map(|m| m.len()), "mtime_ns": mtime_ns,
            "width": info.width, "height": info.height, "fps_bits": info.fps.to_bits(), "duration_bits": info.duration_s.to_bits()})
    }

    fn dir(run: &dyn RunCtx) -> PathBuf {
        run.root().join(VIDEO_ANALYSIS)
    }

    /// The run's analysis of `source`, if inspect_input made one for it.
    fn load(run: &dyn RunCtx, source: &Path, info: &VideoInfo) -> Option<Self> {
        let a: Self = serde_json::from_slice(&std::fs::read(Self::dir(run).join("analysis.json")).ok()?).ok()?;
        (a.key == Self::key(source, info) && a.sharpness.len() == a.track.len()).then_some(a)
    }

    fn save(&self, run: &dyn RunCtx) -> anyhow::Result<()> {
        write_json(&Self::dir(run).join("analysis.json"), self)
    }

    /// Decode `source` once at ANALYSIS_FPS: the motion track, candidates for the quick frames, and
    /// every `stat_step`-th sample (at most 400) at 640 px for inspection statistics. HDR candidates
    /// use the extraction's 16-bit decode and SDR tone map too.
    fn run(cfg: &BrushConfig, run: &dyn RunCtx, source: &Path, info: &VideoInfo, stat_step: u64) -> anyhow::Result<Option<(Self, Vec<RgbImage>)>> {
        if info.width == 0 || info.height == 0 {
            return Ok(None);
        }
        let tone_mapped = frameplan::hdr_transfer(&cfg.ffmpeg, source).map(|t|
            if t == frameplan::Transfer::Hlg { "hlg".into() } else { "pq".into() });
        let dir = Self::dir(run);
        std::fs::create_dir_all(&dir)?;
        // A repeated inspection may replace a longer candidate cache in the same run.
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.file_name().is_some_and(|name| name.to_string_lossy().starts_with('s'))
                && path.extension().is_some_and(|ext| ext == "jpg") {
                std::fs::remove_file(path)?;
            }
        }
        let size = scaled_size(info.width, info.height, CANDIDATE_EDGE);
        let (gw, gh) = frameplan::analysis_size(info.width, info.height);
        let stat_size = scaled_size(info.width, info.height, 640);
        let mut tracker = frameplan::MotionTracker::new(gw, gh);
        let mut stats = Vec::new();
        let mut prefix_end = None;
        let mut candidates_through = 0;
        // Sharpness and the candidate JPEGs on worker threads, so the decode never waits on them
        // (driveway 4K: inspect 4.5 -> 12.3 s with them inline on a loaded host).
        let sharpness = std::sync::Mutex::new(Vec::<(u64, f32)>::new());
        let (tx, rx) = std::sync::mpsc::sync_channel::<(u64, RgbImage, bool)>(16);
        let rx = std::sync::Mutex::new(rx);
        std::thread::scope(|scope| -> anyhow::Result<()> {
            let workers: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(|| -> anyhow::Result<()> {
                        loop {
                            let next = rx.lock().unwrap().recv();
                            let Ok((i, img, cache)) = next else { return Ok(()) };
                            sharpness.lock().unwrap().push((i, frameplan::sharpness(&img)));
                            if cache {
                                splat_eval::images::write_jpeg(&dir.join(format!("s{i:06}.jpg")), &img, 95, None)?;
                            }
                        }
                    })
                })
                .collect();
            let decoded = frameplan::analysis_pass(&cfg.ffmpeg, source, size, info.width, info.height, |i, px, grey| {
                let img = RgbImage::from_raw(size.0, size.1, px.to_vec()).context("analysis frame size")?;
                tracker.push(grey);
                if i % stat_step == 0 && stats.len() < 400 {
                    stats.push(image::imageops::resize(&img, stat_size.0, stat_size.1, FilterType::Triangle));
                }
                // Keep one neighbour after the prefix for the sharpest-of-three pick. The
                // tracker may book motion back to earlier samples, so this conservative stop
                // always includes the final prefix that motion_prefix computes on the full track.
                if prefix_end.is_none() && i as f64 / frameplan::ANALYSIS_FPS >= QUICK_MIN_WINDOW_S
                    && tracker.recorded_motion_widths() > frameplan::QUICK_MOTION_WIDTHS {
                    prefix_end = Some(i);
                }
                let cache = prefix_end.is_none_or(|end| i <= end + 1);
                if cache {
                    candidates_through = i;
                }
                // The full planner needs the whole timeline's detail signal, while only preview
                // candidates need JPEGs. Keep measuring after the cached prefix without more files.
                tx.send((i, img, cache)).map_err(|_| anyhow::anyhow!("analysis writers stopped"))?;
                Ok(())
            });
            drop(tx);
            for w in workers {
                w.join().map_err(|_| anyhow::anyhow!("analysis writer panicked"))??;
            }
            decoded.map(|_| ())
        })?;
        let mut sharpness = sharpness.into_inner().unwrap();
        sharpness.sort_by_key(|s| s.0);
        anyhow::ensure!(!sharpness.is_empty(), "analysis wrote no samples");
        let track = tracker.finish()?;
        let mut all_sharpness = vec![0.0; track.len()];
        for (i, value) in sharpness { all_sharpness[i as usize] = value; }
        let analysis = Self { key: Self::key(source, info), size, track, sharpness:all_sharpness,
            candidates_through, quick_picks: vec![], tone_mapped };
        analysis.save(run)?;
        Ok(Some((analysis, stats)))
    }
}

pub fn inspect_input(cfg: &BrushConfig, run: &dyn RunCtx, req: InspectInput) -> anyhow::Result<ToolResult> {
    let start = Instant::now();
    let source = run.source();
    let (artifact, out) = new_artifact(run, ArtifactKind::Images, json!({"purpose": "input_inspection"}))?;
    let mut warnings = vec![];
    let mut metrics = serde_json::Map::new();
    let samples: Vec<RgbImage>;
    let capture = crate::capture::load(run)?;
    if source.is_dir() || capture.is_some() {
        let inputs = if let Some(capture) = &capture { capture.frames.iter().map(|f| f.image.clone()).collect() } else { list_images(&image_dir(source))? };
        if inputs.is_empty() {
            return Err(invalid("no input images"));
        }
        let mut orientations = std::collections::BTreeSet::new();
        let mut frames = Vec::new();
        // Stats over up to 96 evenly spaced images; thumbnails for the driver from those.
        let stat_n = inputs.len().min(96);
        for k in 0..stat_n {
            let path = &inputs[k * (inputs.len() - 1) / (stat_n - 1).max(1)];
            let (img, o) = if capture.is_some() { (image::open(path)?, 1) } else { load_oriented(path)? };
            orientations.insert(o);
            frames.push(img.resize(640, 640, FilterType::Triangle).to_rgb8());
        }
        metrics.insert("input_kind".into(), json!("images"));
        if let Some(c) = &capture {
            metrics.insert("input_kind".into(), json!("capture"));
            metrics.insert("capture_format".into(), json!(c.format));
            metrics.insert("sensor_depth_frames".into(), json!(c.frames.iter().filter(|f| f.depth.is_some()).count()));
            let dims = image::image_dimensions(&c.frames[0].image)?;
            metrics.insert("width".into(), json!(dims.0));
            metrics.insert("height".into(), json!(dims.1));
        }
        metrics.insert("image_count".into(), json!(inputs.len()));
        metrics.insert("exif_orientations".into(), json!(orientations));
        metrics.insert("rotation_metadata_present".into(), json!(orientations.iter().any(|&o| o != 1)));
        metrics.insert("upside_down_metadata".into(), json!(orientations.contains(&3)));
        let n = (req.sample_count as usize).min(inputs.len());
        for i in 0..n {
            let path = &inputs[i * (inputs.len() - 1) / (n - 1).max(1)];
            let img = if capture.is_some() { image::open(path)? } else { load_oriented(path)?.0 };
            img.resize(960, 960, FilterType::Triangle).to_rgb8().save(out.join(format!("sample-{i:03}.jpg")))?;
        }
        samples = frames;
    } else {
        let info = probe(cfg, source)?;
        if info.duration_s <= 0.0 {
            return Err(invalid("video duration is unavailable"));
        }
        metrics.insert("input_kind".into(), json!("video"));
        metrics.insert("codec".into(), json!(info.codec));
        metrics.insert("width".into(), json!(info.width));
        metrics.insert("height".into(), json!(info.height));
        metrics.insert("duration_s".into(), json!(info.duration_s));
        metrics.insert("fps".into(), json!(info.fps));
        metrics.insert("frame_count_estimate".into(), json!(info.frame_count));
        metrics.insert("rotation_degrees".into(), json!(info.rotation));
        metrics.insert("rotation_metadata_present".into(), json!(info.rotation != 0));
        metrics.insert("upside_down_metadata".into(), json!(info.rotation.rem_euclid(360) == 180));
        metrics.insert("video_streams".into(), json!(info.video_streams));
        // ~4 fps stat samples (at most 400) in one decode pass, which also makes the run's video
        // analysis (motion track and quick-frame candidates) for extract_frames.
        let samples_8fps = (info.duration_s * frameplan::ANALYSIS_FPS).ceil() as u64;
        // The extractions' black-bar seeks run alongside this decode instead of serially inside
        // each extraction. A failure records nothing: extract_frames then detects (and reports) it.
        let bars = (crop_bars_enabled() && info.width > 0 && info.height > 0).then(|| {
            let (ffmpeg, source, info, dir) = (cfg.ffmpeg.clone(), source.to_path_buf(), info.clone(), VideoAnalysis::dir(run));
            std::thread::spawn(move || VideoBars::detect(&ffmpeg, &source, &info, &dir))
        });
        let analysis = VideoAnalysis::run(cfg, run, source, &info, (frameplan::ANALYSIS_FPS / 4.0) as u64 * (samples_8fps / 800 + 1));
        if let Some(bars) = bars {
            match bars.join() {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => eprintln!("black-bar detection during inspection failed: {error:#}"),
                Err(_) => eprintln!("black-bar detection during inspection panicked"),
            }
        }
        let mut frames = match analysis? {
            Some((analysis, stats)) => {
                metrics.insert("analysis_samples".into(), json!(analysis.track.len()));
                metrics.insert("cached_preview_candidates".into(), json!(analysis.candidates_through + 1));
                stats
            }
            None => Vec::new(),
        };
        if frames.is_empty() {
            let step = ((info.fps / 4.0).round() as u64).max(1).max(info.frame_count / 400);
            let size = scaled_size(info.width, info.height, 640);
            decode_video(cfg, source, size, &format!("select='not(mod(n\\,{step}))',"), None, |_, f| {
                frames.push(f);
                Ok(())
            })?;
        }
        if frames.is_empty() {
            bail!("ffmpeg decoded no frames");
        }
        let n = (req.sample_count as usize).min(frames.len());
        for i in 0..n {
            frames[i * (frames.len() - 1) / (n - 1).max(1)].save(out.join(format!("sample-{i:03}.jpg")))?;
        }
        if info.rotation != 0 {
            warnings.push("Rotation metadata detected; ffmpeg extraction applies autorotation.".into());
        }
        samples = frames;
    }
    metrics.extend(summarize(&samples));
    let sky_bottom = metrics["sky_bottom_fraction"].as_f64().unwrap_or(0.0);
    let suspect = metrics["rotation_metadata_present"] == json!(true) || metrics["upside_down_metadata"] == json!(true) || sky_bottom > 0.6;
    metrics.insert("orientation_suspect".into(), json!(suspect));
    metrics.insert("visual_orientation".into(), json!(if sky_bottom > 0.6 { "likely_upside_down" } else { "unverified" }));
    if sky_bottom > 0.6 {
        // Photos: a bright ground (truck84's pavement) trips the sky test, and rotating upright
        // photos broke their SfM (73/84, 13.9 dB vs 84/84, 22.6 dB); cameras record photo
        // orientation in EXIF, so only videos get the rotate suggestion.
        if metrics["input_kind"] == json!("video") {
            warnings.push(format!("{:.0}% of samples look upside down (bright, blue bottom third); check samples and consider rotate_deg=180.", sky_bottom * 100.0));
        } else {
            warnings.push(format!("{:.0}% of samples have a bright, blue bottom third; photos keep their EXIF orientation (look at the samples before rotating).", sky_bottom * 100.0));
        }
    }
    if metrics["upside_down_metadata"] == json!(true) {
        warnings.push("Source metadata indicates upside-down capture; inspect the corrected samples.".into());
    }
    let get = |k: &str| metrics.get(k).and_then(Value::as_f64).unwrap_or(0.0);
    if get("sharpness_p10_over_median") < 0.6 {
        warnings.push("Motion blur: prefer sharpness_window frame selection.".into());
    }
    if get("luminance_range") > 40.0 {
        warnings.push("Exposure swings detected.".into());
    }
    if get("top_clipped_fraction") > 0.05 {
        warnings.push("Clipped highlights in the top third; inspect sky coverage.".into());
    }
    write_json(&out.join("inspection.json"), &metrics)?;
    Ok(ToolResult {
        artifacts: vec![artifact],
        metrics: metrics.into_iter().collect(),
        images: list_images(&out)?.iter().map(|p| rel(run, p)).collect(),
        warnings,
        wall_s: Some(start.elapsed().as_secs_f64()),
        ..result("inspect_input", Status::Succeeded)
    })
}

/// The SfM copy of a photo whose source exceeds `max_edge`: its decoded, upright original.
type SfmSource = Option<DynamicImage>;

struct Selector {
    window: f64,
    cur_win: u64,
    best: Option<(f32, u64, RgbImage, Option<String>, Option<Vec<u8>>, SfmSource)>,
    last_thumb: Option<Vec<f32>>,
    uniform: bool,
    drop_black: bool,
    drop_duplicates: bool,
    rotate_deg: u32,
    dest: PathBuf,
    start_index: u64,
    kept: Vec<Value>,
    black: u32,
    dup: u32,
    /// `<frames>/sfm_images` and its long edge; photo copies there are written in the background.
    sfm_dest: PathBuf,
    sfm_edge: u32,
    sfm_writers: std::collections::VecDeque<std::thread::JoinHandle<anyhow::Result<(u32, u32)>>>,
    /// Sizes of the kept training frames and of the SfM copies written.
    train_sizes: Vec<(u32, u32)>,
    sfm_sizes: Vec<(u32, u32)>,
}

impl Selector {
    /// Whether input `i` can still be kept: uniform keeps the first frame of each window, so the
    /// rest of the window need not be decoded (the quick photo SfM picks 32 of garden's 185).
    fn wants(&self, i: u64) -> bool {
        !(self.uniform && self.best.is_some() && (i as f64 / self.window) as u64 == self.cur_win)
    }

    fn push(&mut self, i: u64, img: RgbImage, label: Option<String>, exif: Option<Vec<u8>>, sfm: SfmSource) -> anyhow::Result<()> {
        let win = (i as f64 / self.window) as u64;
        if win != self.cur_win {
            self.flush()?;
            self.cur_win = win;
        }
        let luma = img.as_raw().iter().map(|&v| v as u64).sum::<u64>() as f32 / img.as_raw().len() as f32;
        if self.drop_black && luma < BLACK_LUMA {
            self.black += 1;
            return Ok(());
        }
        if self.uniform && self.best.is_some() {
            return Ok(());
        }
        let s = sharpness(&img);
        if self.best.as_ref().is_none_or(|b| s > b.0) {
            self.best = Some((s, i, img, label, exif, sfm));
        }
        Ok(())
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        let Some((s, idx, img, source_name, exif, sfm)) = self.best.take() else { return Ok(()) };
        let th = thumb(&img);
        if self.drop_duplicates && self.last_thumb.as_ref().is_some_and(|t| mean_abs_diff(&th, t) < DUP_DIFF) {
            self.dup += 1;
            return Ok(());
        }
        self.last_thumb = Some(th);
        let img = rotate(img, self.rotate_deg);
        let src = idx + self.start_index;
        let name = format!("frame_{src:06}.jpg");
        // The SfM copy: same frame, rotation, encoder and EXIF, at long edge min(native, sfm_edge).
        if let Some(native) = sfm {
            if self.sfm_writers.len() >= SFM_WRITERS {
                self.join_sfm_writer()?;
            }
            let (path, edge, deg, exif) = (self.sfm_dest.join(&name), self.sfm_edge, self.rotate_deg, exif.clone());
            self.sfm_writers.push_back(std::thread::spawn(move || {
                let img = if native.width().max(native.height()) > edge { native.resize(edge, edge, FilterType::Lanczos3) } else { native };
                let img = rotate(img.to_rgb8(), deg);
                splat_eval::images::write_jpeg_compact(&path, &img, 90, exif)?;
                Ok(img.dimensions())
            }));
        }
        self.train_sizes.push(img.dimensions());
        // Photos keep their EXIF (focal-length prior for SfM); orientation is already applied.
        // Quality 90, 4:2:0: every frame is uploaded on the user's uplink before SfM; 43% fewer
        // bytes than q95 4:4:4, and the 41.8 dB encoding error is far below a splat's (~26 dB).
        splat_eval::images::write_jpeg_compact(&self.dest.join(&name), &img, 90, exif)?;
        self.kept.push(json!({"name": name, "source_index": src, "sharpness": s, "source_name": source_name}));
        Ok(())
    }

    fn join_sfm_writer(&mut self) -> anyhow::Result<()> {
        if let Some(w) = self.sfm_writers.pop_front() {
            self.sfm_sizes.push(w.join().map_err(|_| anyhow::anyhow!("SfM frame writer panicked"))??);
        }
        Ok(())
    }

    /// Wait for the SfM copies. Keep `sfm_images/` only when every kept frame has one and the copies
    /// mix no more sizes than the training frames (COLMAP's single camera skips images of another size;
    /// 2000- and 3000-px photos are both 1600 px for training but 2000 and 2400 px here). Returns the
    /// metrics to record.
    fn finish_sfm(&mut self) -> anyhow::Result<serde_json::Map<String, Value>> {
        while !self.sfm_writers.is_empty() {
            self.join_sfm_writer()?;
        }
        let mut m = serde_json::Map::new();
        let written = self.sfm_sizes.len();
        if written == 0 {
            return Ok(m);
        }
        let distinct = |sizes: &[(u32, u32)]| sizes.iter().collect::<std::collections::BTreeSet<_>>().len();
        let skipped = if written < self.train_sizes.len() {
            Some("some photos are not larger than max_edge")
        } else if distinct(&self.sfm_sizes) > distinct(&self.train_sizes) {
            Some("photo sizes would differ at the SfM edge")
        } else {
            None
        };
        if let Some(reason) = skipped {
            std::fs::remove_dir_all(&self.sfm_dest)?;
            m.insert("sfm_images_skipped".into(), json!(reason));
        } else {
            m.insert("sfm_images".into(), json!(written));
            m.insert("sfm_max_edge".into(), json!(self.sfm_edge));
        }
        Ok(m)
    }
}

/// Black-bar cropping in `extract_frames` (default on; `SPLAT_CROP_BARS=0` disables it).
fn crop_bars_enabled() -> bool {
    std::env::var("SPLAT_CROP_BARS").map_or(true, |v| v.trim() != "0")
}

/// Black bars of a photo set: detected on up to 12 evenly spaced photos, which must all share one
/// size; returns that size and the crop.
fn detect_photo_bars(inputs: &[PathBuf]) -> Option<((u32, u32), frameplan::Crop)> {
    let step = inputs.len().div_ceil(12).max(1);
    let mut size = None;
    let mut greys = Vec::new();
    for path in inputs.iter().step_by(step) {
        let img = load_oriented(path).ok()?.0.to_luma8();
        let dims = img.dimensions();
        if *size.get_or_insert(dims) != dims {
            return None;
        }
        greys.push(img.into_raw());
    }
    let (w, h) = size?;
    frameplan::detect_black_bars(w as usize, h as usize, greys.iter().map(|g| g.as_slice())).map(|c| ((w, h), c))
}

/// Full-extraction base frame rate (kept frames per second of video, before target_count's cap).
const FULL_FRAME_FPS: f64 = 5.0;

/// Shortest clip prefix the quick extraction plans in (see ExtractFrames::max_motion_widths).
const QUICK_MIN_WINDOW_S: f64 = 5.0;

/// Keep the preview's small camera budget before a sustained loss of detail. Motion measured
/// during a blurred pan or a foreground occlusion can dominate the budget even though those
/// frames cannot be matched. HLG Oakland's passing train consumed 15 of 22 quick views; the
/// clear prefix registers all 24 with the same SfM settings. A relative threshold keeps normally
/// soft captures intact, and a full second below it avoids reacting to individual weak samples.
fn quick_sharp_prefix(sharpness: &[f32], end: f64) -> f64 {
    let count = ((end * frameplan::ANALYSIS_FPS).floor() as usize + 1).min(sharpness.len());
    let samples = &sharpness[..count];
    let mut sorted: Vec<f32> = samples.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    if sorted.is_empty() {
        return end;
    }
    sorted.sort_unstable_by(f32::total_cmp);
    let reference = sorted[(sorted.len() - 1) * 3 / 4];
    // Flat or uniformly soft footage has no clear reference to cut against.
    if reference < 100.0 {
        return end;
    }
    let mut run = 0usize;
    for (i, &value) in samples.iter().enumerate() {
        run = if value.is_finite() && value < reference * 0.15 { run + 1 } else { 0 };
        if run >= frameplan::ANALYSIS_FPS as usize {
            let start = i + 1 - run;
            // Retain at least 2.5 s, and stop a sample before the dip so the sharpest-of-three
            // selector has a clear neighbour at its final planned instant.
            let before = start.saturating_sub(1) as f64 / frameplan::ANALYSIS_FPS;
            if before >= 2.5 {
                return end.min(before);
            }
        }
    }
    end
}

/// Exclude only sustained losses relative to a clear reference; uniformly soft captures and
/// transient defocus retain their original plans. Pad by one sample for neighbouring-frame picks.
fn low_detail_ranges(sharpness: &[f32], duration: f64) -> Vec<(f64, f64)> {
    let mut sorted: Vec<f32> = sharpness.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    if sorted.is_empty() {
        return vec![];
    }
    sorted.sort_unstable_by(f32::total_cmp);
    let reference = sorted[(sorted.len() - 1) * 3 / 4];
    if reference < 100.0 {
        return vec![];
    }
    let mut ranges: Vec<(f64, f64)> = vec![];
    let mut start = None;
    for i in 0..=sharpness.len() {
        let low = sharpness.get(i).is_some_and(|v| v.is_finite() && *v < reference * 0.15);
        if low {
            start.get_or_insert(i);
        } else if let Some(first) = start.take() {
            if i - first >= frameplan::ANALYSIS_FPS as usize {
                let a = (first.saturating_sub(1) as f64 / frameplan::ANALYSIS_FPS).min(duration);
                let b = ((i + 1) as f64 / frameplan::ANALYSIS_FPS).min(duration);
                if let Some(last) = ranges.last_mut().filter(|last| last.1 >= a) {
                    last.1 = b;
                } else if b > a {
                    ranges.push((a, b));
                }
            }
        }
    }
    ranges
}

/// Views kept per second inside an omitted low-detail range. Zero left the range uncovered (room 3:
/// off-path smear at an anchor inside its dropped 3.6 s); a thin sample keeps some coverage there
/// without letting the blur dominate the budget again.
const LOW_DETAIL_FPS: f64 = 2.0;

/// LOW_DETAIL_FPS, or SPLAT_LOW_DETAIL_FPS. Thinning is opt-in (ExtractFrames::thin_low_detail): on the ultrawide
/// room clip it cut the 0-10 s opening off the room at every density tried (1-4 views/s).
fn low_detail_fps() -> f64 {
    std::env::var("SPLAT_LOW_DETAIL_FPS").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| *v > 0.0).unwrap_or(LOW_DETAIL_FPS)
}

/// Run the existing density planner on clear time, then map its instants back to source time, plus a
/// thin LOW_DETAIL_FPS sample inside each omitted range. The original motion curve stays intact so the
/// spacing audit still exposes capture breaks.
fn reallocate_clear_views(plan: &mut frameplan::Plan, sharpness: &[f32], fps: f64) -> Vec<(f64, f64)> {
    let ranges = low_detail_ranges(sharpness, plan.duration_s);
    if ranges.is_empty() {
        return ranges;
    }
    let mut clear = vec![];
    let mut previous = 0.0;
    for &(a, b) in &ranges {
        if a > previous {
            clear.push((previous, a));
        }
        previous = b;
    }
    if previous < plan.duration_s {
        clear.push((previous, plan.duration_s));
    }
    let duration: f64 = clear.iter().map(|&(a, b)| b - a).sum();
    // Keep enough actual source frames for the original camera budget; never turn a mostly soft
    // capture into a tiny sharp fragment or duplicate views to fill its request.
    if duration < 2.5 || duration * fps < plan.count as f64 {
        return vec![];
    }
    let mut compressed = vec![(0.0, 0.0)];
    let mut mapping = vec![];
    let mut elapsed = 0.0;
    for &(a, b) in &clear {
        mapping.push((elapsed, elapsed + b - a, a));
        let mut previous = a;
        for end in plan.motion_curve.iter().map(|p| p.0).filter(|&t| t > a && t < b).chain(std::iter::once(b)) {
            elapsed += end - previous;
            compressed.push((elapsed, (plan.motion_at(end) - plan.motion_at(previous)).max(0.0)));
            previous = end;
        }
    }
    let sparse: Vec<f64> = ranges.iter().flat_map(|&(a, b)| {
        let n = ((b - a) * low_detail_fps()).round().max(1.0) as usize;
        (0..n).map(move |i| a + (i as f64 + 0.5) * (b - a) / n as f64)
    }).collect();
    let clear_count = plan.count.saturating_sub(sparse.len() as u32).max(1);
    let options = frameplan::PlanOptions {
        base_fps: clear_count as f64 / duration,
        min_count: clear_count,
        max_count: clear_count,
        ..plan.options.clone()
    };
    let clear_plan = frameplan::plan(&compressed, duration, (duration * fps) as u64, &options);
    plan.instants_s = clear_plan.instants_s.iter().map(|&t| {
        let j = mapping.partition_point(|&(_, end, _)| end <= t).min(mapping.len() - 1);
        let (start, end, source) = mapping[j];
        source + (t - start).clamp(0.0, end - start)
    }).chain(sparse).collect();
    plan.instants_s.sort_by(f64::total_cmp);
    plan.density_fps = (0.0, clear_plan.density_fps.1);
    plan.reason.push_str(&format!(" Redistributed the camera budget across {:.2} s of clear views, keeping {} view/s in {:.2} s of sustained detail loss.", duration, low_detail_fps(), plan.duration_s - duration));
    ranges
}

/// Source-frame ranges the selector must not pick from: each omitted range minus a one-frame
/// neighbourhood around its thin-sample instants (the selector picks within one frame).
fn low_detail_exclusions(ranges: &[(f64, f64)], instants: &[f64], fps: f64) -> Vec<std::ops::Range<u64>> {
    let mut out = vec![];
    for &(a, b) in ranges {
        let (mut from, to) = ((a * fps).ceil() as u64, (b * fps).ceil() as u64);
        for &t in instants.iter().filter(|&&t| t >= a && t < b) {
            let f = (t * fps).round() as u64;
            if f.saturating_sub(1) > from {
                out.push(from..f - 1);
            }
            from = from.max(f + 2);
        }
        if to > from {
            out.push(from..to);
        }
    }
    out
}

/// The quick extraction from `a`'s samples: at each planned instant the sharpest sample within one
/// sample (the nearest when uniform), cropped like the decode would be and resized to `size`; written
/// through `sel` (rotation, black/duplicate drops) as `frame_<source index>.jpg`.
fn pick_candidates(
    run: &dyn RunCtx,
    a: &VideoAnalysis,
    plan: &frameplan::Plan,
    fps: f64,
    crop: Option<(frameplan::Crop, u32, u32)>,
    size: (u32, u32),
    sel: &mut Selector,
) -> anyhow::Result<frameplan::Extracted> {
    let n = a.track.len() as i64;
    let mut picks: Vec<i64> = Vec::new();
    for &t in &plan.instants_s {
        let j = (t * frameplan::ANALYSIS_FPS).round() as i64;
        let near = (j - 1..=j + 1).filter(|&k| (0..n).contains(&k));
        let pick = if sel.uniform { Some(j.clamp(0, n - 1)) } else { near.max_by(|&x, &y| a.sharpness[x as usize].total_cmp(&a.sharpness[y as usize])) };
        if let Some(p) = pick.filter(|p| picks.last() != Some(p)) {
            picks.push(p);
        }
    }
    let dir = VideoAnalysis::dir(run);
    let window = sel.window;
    sel.window = 1.0;
    let decoded = picks.len() as u64;
    for p in picks {
        let mut img = image::open(dir.join(format!("s{p:06}.jpg")))?.to_rgb8();
        if let Some((c, w, h)) = crop {
            let (sx, sy) = (img.width() as f64 / w as f64, img.height() as f64 / h as f64);
            let (x, y) = ((c.x as f64 * sx).round() as u32, (c.y as f64 * sy).round() as u32);
            let (cw, ch) = (((c.w as f64 * sx).round() as u32).min(img.width() - x), ((c.h as f64 * sy).round() as u32).min(img.height() - y));
            img = image::imageops::crop_imm(&img, x, y, cw, ch).to_image();
        }
        if img.dimensions() != size {
            img = image::imageops::resize(&img, size.0, size.1, FilterType::Triangle);
        }
        let source_index = (p as f64 / frameplan::ANALYSIS_FPS * fps).round() as u64;
        sel.push(source_index, img, None, None, None)?;
    }
    sel.flush()?;
    sel.window = window;
    let kept = std::mem::take(&mut sel.kept)
        .into_iter()
        .map(|mut k| {
            k["decoded_index"] = k["source_index"].clone();
            k
        })
        .collect();
    Ok(frameplan::Extracted { decoded, kept, dropped_black: sel.black, dropped_duplicates: sel.dup, ..Default::default() })
}

pub fn extract_frames(cfg: &BrushConfig, run: &dyn RunCtx, req: ExtractFrames) -> anyhow::Result<ToolResult> {
    if let Some(capture) = crate::capture::load(run)? {
        anyhow::ensure!(req.input_kind == InputKind::Auto, "calibrated captures retain sensor provenance; input_kind must be auto");
        anyhow::ensure!(!req.preserve_all || (req.time_range_s.is_none() && req.max_motion_widths.is_none() && !req.preview), "preserve_all requires the full capture");
        return crate::capture::extract(run, req, capture);
    }
    let start = Instant::now();
    if ![0, 90, 180, 270].contains(&req.rotate_deg) {
        return Err(invalid("rotate_deg must be 0, 90, 180 or 270"));
    }
    let source = run.source();
    let video = is_video(source);
    anyhow::ensure!(!video || req.input_kind != InputKind::Images, "input_kind=images cannot describe a video file");
    anyhow::ensure!(!req.preserve_all || (req.time_range_s.is_none() && req.max_motion_widths.is_none() && !req.preview),
        "preserve_all requires the full capture (no time_range_s, max_motion_widths or preview)");
    let ordered_video = video || req.input_kind == InputKind::Video;
    let (artifact, out) = new_artifact(run, ArtifactKind::Frames, json!({"input_kind": if ordered_video { "video" } else { "images" },
        "source_kind": if video { "video_file" } else { "image_files" }, "preserve_all": req.preserve_all,
        "chronology": if ordered_video { "source_order" } else { "unspecified" }}))?;
    let dest = out.join("images");
    std::fs::create_dir_all(&dest)?;
    let mut sel = Selector {
        window: 1.0,
        cur_win: 0,
        best: None,
        last_thumb: None,
        uniform: matches!(req.strategy, FrameStrategy::Uniform),
        drop_black: req.drop_black && !req.preserve_all,
        drop_duplicates: req.drop_duplicates && !req.preserve_all,
        rotate_deg: req.rotate_deg,
        dest: dest.clone(),
        start_index: 0,
        kept: vec![],
        black: 0,
        dup: 0,
        sfm_dest: out.join("sfm_images"),
        sfm_edge: sfm_edge(),
        sfm_writers: Default::default(),
        train_sizes: vec![],
        sfm_sizes: vec![],
    };
    let mut metrics = serde_json::Map::new();
    let decoded;
    if video {
        let info = probe(cfg, source)?;
        let mut estimate = info.frame_count;
        if let Some((a, b)) = req.time_range_s {
            if !(0.0 <= a && a < b && b <= info.duration_s + 1e-3) {
                return Err(invalid("time_range_s must lie within the video duration"));
            }
            estimate = ((b - a) * info.fps) as u64;
            sel.start_index = (a * info.fps).round() as u64;
        }
        // Frame count follows duration and camera motion (shared with spirula: splat_eval::frameplan).
        // inspect_input's analysis supplies the motion track (and the quick frames) when it covers
        // this decode.
        let mut analysis = req.time_range_s.is_none().then(|| VideoAnalysis::load(run, source, &info)).flatten();
        let mut span = req.time_range_s.map_or(info.duration_s, |(a, b)| b - a);
        let mut time_range = req.time_range_s;
        let mut track = match &analysis {
            _ if req.preserve_all => vec![],
            Some(a) => a.track.clone(),
            None => frameplan::motion_track(&cfg.ffmpeg, source, req.time_range_s, info.width, info.height)?,
        };
        // The quick SfM's few frames must each overlap the next: plan them inside the prefix whose
        // motion stays within max_motion_widths (showbus: 24 frames over 14.3 widths found no initial
        // pair; over its first 15 s, 2.3 widths, all 24 registered).
        if let Some(widths) = req.max_motion_widths.filter(|_| req.time_range_s.is_none()) {
            let motion_end = frameplan::motion_prefix(&track, widths, QUICK_MIN_WINDOW_S);
            let end = analysis.as_ref().map_or(motion_end, |a| {
                let cached = &a.sharpness[..a.sharpness.len().min(a.candidates_through as usize + 1)];
                quick_sharp_prefix(cached, motion_end)
            });
            if end < motion_end {
                metrics.insert("quick_blur_window_s".into(), json!(end));
            }
            if end + 0.5 < span {
                track.retain(|t| t.0 <= end);
                span = end;
                estimate = (end * info.fps) as u64;
                time_range = Some((0.0, end));
                metrics.insert("motion_window_s".into(), json!(end));
            }
        }
        let mut options = frameplan::PlanOptions::spaced(req.target_count, matches!(req.spacing, FrameSpacing::Motion));
        // Full extractions keep FULL_FRAME_FPS (capped by target_count and the source frames): denser
        // views give SfM the overlap it needs (ARKit 42899041: 154 -> 376 cameras, +1.8 dB on shared
        // held-out views; Austin +8.9, LA +1.3); gaussian training on video reads every frame up to 600 (train.rs
        // VIDEO_TRAIN_MAX_FRAMES). SPLAT_FRAME_FPS overrides it (2 = the old sampling).
        if req.max_motion_widths.is_none() {
            let fps = req.base_fps.filter(|v| *v > 0.0)
                .or_else(|| std::env::var("SPLAT_FRAME_FPS").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| *v > 0.0))
                .unwrap_or(FULL_FRAME_FPS);
            options.base_fps = fps;
            metrics.insert("frame_base_fps".into(), json!(fps));
        }
        let mut plan = frameplan::plan(&track, span, estimate, &options);
        let omitted = if req.max_motion_widths.is_none() && req.time_range_s.is_none()
            && !req.preserve_all && req.thin_low_detail && matches!(req.strategy, FrameStrategy::SharpnessWindow) {
            analysis.as_ref().map_or_else(Vec::new, |a| reallocate_clear_views(&mut plan, &a.sharpness, info.fps))
        } else { vec![] };
        if !omitted.is_empty() {
            metrics.insert("low_detail_ranges_s".into(), json!(omitted));
            metrics.insert("low_detail_omitted_s".into(), json!(omitted.iter().map(|&(a, b)| b - a).sum::<f64>()));
        }
        // The clear prefix has fewer 8 fps candidates than the original window. A motion-even
        // plan can ask for >8 views/s locally and collapse several slots onto one sample (Oakland:
        // 24 slots -> 19 views -> only 5 cameras). Even time samples retain the camera budget;
        // use them only when every gap still meets the requested motion-overlap bound.
        if metrics.contains_key("quick_blur_window_s") && plan.count > 0 {
            let uniform: Vec<f64> = (0..plan.count).map(|i| (i as f64 + 0.5) * span / plan.count as f64).collect();
            let max_gap = uniform.windows(2).map(|p| plan.motion_at(p[1]) - plan.motion_at(p[0])).fold(0.0f64, f64::max);
            if req.max_motion_widths.is_some_and(|widths| max_gap <= widths / req.target_count.max(1) as f64) {
                plan.instants_s = uniform;
                plan.density_fps = (plan.count as f64 / span, plan.count as f64 / span);
                plan.reason.push_str(" Clear preview views are evenly spaced in time within the motion-overlap bound.");
                sel.uniform = true;
                metrics.insert("quick_clear_uniform".into(), json!(true));
            }
        }
        // Constant black bars (letterbox, encoder padding) are cropped before scaling, so SfM,
        // training and held-out GT all see the same content-only frames.
        let crop = if crop_bars_enabled() {
            // The whole capture's crop is measured once (VideoBars): by inspect_input, else by the
            // first extraction, for the next one.
            let dir = VideoAnalysis::dir(run);
            match req.time_range_s.is_none().then(|| VideoBars::load(&dir, source, &info)).flatten() {
                Some(crop) => {
                    metrics.insert("black_bars_cached".into(), json!(true));
                    crop
                }
                None => {
                    let detected = if req.time_range_s.is_none() {
                        VideoBars::detect(&cfg.ffmpeg, source, &info, &dir)
                    } else {
                        frameplan::detect_video_bars(&cfg.ffmpeg, source, req.time_range_s, info.duration_s, info.width, info.height)
                    };
                    detected.unwrap_or_else(|e| {
                        eprintln!("black-bar detection skipped: {e:#}");
                        None
                    })
                }
            }
        } else {
            None
        };
        let (crop_w, crop_h) = crop.map_or((info.width, info.height), |c| (c.w, c.h));
        if let Some(c) = crop {
            metrics.insert("black_bar_crop".into(), json!({"source": [info.width, info.height], "crop": c}));
        }
        let size = scaled_size(crop_w, crop_h, req.max_edge);
        let quick = req.max_motion_widths.is_some();
        // Only a windowed quick extraction comes from the samples: one within its motion budget is the
        // same extraction (and the same frames) as before.
        let windowed = quick && time_range.is_some();
        let from_candidates = analysis.as_ref().filter(|a| windowed && size.0.max(size.1) <= a.size.0.max(a.size.1)
            && plan.instants_s.iter().all(|t| (t * frameplan::ANALYSIS_FPS).round() as u64 + 1 <= a.candidates_through));
        let quick_picks = if !req.preserve_all && !quick && req.time_range_s.is_none() {
            // The separate manifest includes rotation. Falling back to the analysis cache's
            // indices after a rotation mismatch would silently anchor a different extraction.
            QuickPicks::load(run, source, &info, req.rotate_deg).unwrap_or_default()
        } else { vec![] };
        let extracted = match from_candidates {
            // The quick frames come from the analysis samples: no decode at all.
            Some(a) => {
                let picked = pick_candidates(run, a, &plan, info.fps, crop.map(|c| (c, info.width, info.height)), size, &mut sel)?;
                metrics.insert("from_analysis".into(), json!(true));
                metrics.insert("input_count".into(), json!(info.frame_count));
                frameplan::Extracted { tone_mapped: match a.tone_mapped.as_deref() {
                    Some("hlg") => Some("hlg"), Some("pq") => Some("pq"), _ => None,
                }, ..picked }
            }
            None => frameplan::extract(
                &cfg.ffmpeg,
                source,
                time_range,
                size,
                info.fps,
                &plan,
                &frameplan::ExtractOptions {
                    uniform: sel.uniform,
                    drop_black: req.drop_black && !req.preserve_all,
                    drop_duplicates: req.drop_duplicates && !req.preserve_all,
                    rotate_deg: req.rotate_deg,
                    start_index: sel.start_index,
                    excluded: low_detail_exclusions(&omitted, &plan.instants_s, info.fps),
                    pick_radius: 1,
                    preserve_all: req.preserve_all, preserve_source_frames: req.preserve_all,
                    crop,
                    // The quick frames where the plan leaves room for them (sparse walks gain views;
                    // dense plans such as night-soccer's take none).
                    fill: quick_picks.clone(),
                    anchors: quick_picks,
                },
                &dest,
            )?,
        };
        if quick {
            let picks: Vec<u64> = extracted.kept.iter().filter_map(|k| k["decoded_index"].as_u64()).collect();
            QuickPicks::save(run, source, &info, req.rotate_deg, picks.clone())?;
            if let Some(a) = analysis.as_mut() {
                a.quick_picks = picks;
                a.save(run)?;
            }
        }
        let times = extracted.kept.iter().filter_map(|k| k["decoded_index"].as_f64()).map(|i| i / info.fps).collect::<Vec<_>>();
        let mut frame_plan = if req.preserve_all {
            json!({"mode":"preserve_all","count":extracted.kept.len(),
                "reason":"Every decoded source frame; target_count and sampling plan ignored."})
        } else { plan.metrics() };
        if !req.preserve_all { frame_plan["spacing"] = plan.audit(&times); }
        // HDR (HLG/PQ) phone video is tone-mapped to SDR for SfM and training.
        frame_plan["tone_mapped_from"] = serde_json::json!(extracted.tone_mapped);
        frame_plan["analysis_lifted"] = serde_json::json!(extracted.analysis_lifted);
        frame_plan["quick_filled"] = serde_json::json!(extracted.filled);
        frame_plan["quick_anchored"] = serde_json::json!(extracted.anchored);
        frame_plan["dropped_corrupt"] = serde_json::json!(extracted.dropped_corrupt);
        frame_plan["warnings"] = serde_json::json!(extracted.warnings);
        metrics.insert("frame_plan".into(), frame_plan);
        decoded = extracted.decoded;
        sel.kept = extracted.kept;
        sel.black = extracted.dropped_black;
        sel.dup = extracted.dropped_duplicates;
        metrics.insert("decoder".into(), json!("ffmpeg"));
        metrics.insert("fps".into(), json!(info.fps));
        metrics.insert("rotation_applied".into(), json!(info.rotation));
    } else {
        if req.time_range_s.is_some() {
            return Err(invalid("time_range_s requires a video"));
        }
        let root = image_dir(source);
        let mut inputs = list_images(&root)?;
        if ordered_video { inputs.sort_by(|a, b| natural_cmp(&a.to_string_lossy(), &b.to_string_lossy())); }
        if inputs.is_empty() {
            return Err(invalid("no input images"));
        }
        sel.window = if req.preserve_all { 1.0 } else { (inputs.len() as f64 / req.target_count as f64).max(1.0) };
        // Photos larger than the training frames also get an SfM copy (see SFM_EDGE).
        let sfm_copies = req.sfm_copies && sel.sfm_edge > req.max_edge;
        if sfm_copies {
            std::fs::create_dir_all(&sel.sfm_dest)?;
        }
        // Black bars common to a sample of the photos (all of one size) are cropped from each.
        let photo_crop = if crop_bars_enabled() { detect_photo_bars(&inputs) } else { None };
        if let Some((size, c)) = photo_crop {
            metrics.insert("black_bar_crop".into(), json!({"source": [size.0, size.1], "crop": c}));
        }
        let mut decoded_images = 0;
        for (i, path) in inputs.iter().enumerate() {
            if !sel.wants(i as u64) {
                continue;
            }
            let native = load_oriented(path)?.0;
            decoded_images += 1;
            let native = match photo_crop {
                Some((size, c)) if (native.width(), native.height()) == size => native.crop_imm(c.x, c.y, c.w, c.h),
                _ => native,
            };
            let (img, sfm) = if native.width().max(native.height()) > req.max_edge {
                (native.resize(req.max_edge, req.max_edge, FilterType::Lanczos3), sfm_copies.then_some(native))
            } else {
                (native, None)
            };
            // Cut-out photos: real transparency becomes a COLMAP mask (transparent = ignore),
            // named after the frame this input becomes (masks/frame_NNNNNN.jpg.png).
            if img.color().has_alpha() {
                let alpha = img.to_rgba8();
                if alpha.pixels().any(|p| p[3] < 250) {
                    let mask = image::GrayImage::from_fn(alpha.width(), alpha.height(), |x, y| image::Luma([if alpha.get_pixel(x, y)[3] > 127 { 255 } else { 0 }]));
                    let masks = out.join("masks");
                    std::fs::create_dir_all(&masks)?;
                    mask.save(masks.join(format!("frame_{i:06}.jpg.png")))?;
                }
            }
            let name = path.strip_prefix(&root).unwrap_or(path).to_string_lossy().into_owned();
            sel.push(i as u64, img.to_rgb8(), Some(name), splat_eval::images::exif_upright(path), sfm)?;
        }
        decoded = decoded_images;
        metrics.insert("input_count".into(), json!(inputs.len()));
    }
    sel.flush()?;
    if sel.sfm_dest.is_dir() {
        let sfm = sel.finish_sfm()?;
        if sfm.is_empty() {
            std::fs::remove_dir(&sel.sfm_dest)?;
        }
        metrics.extend(sfm);
    }
    let masks = out.join("masks");
    if masks.is_dir() {
        // Keep masks only for frames that were kept.
        let kept: std::collections::HashSet<String> = sel.kept.iter().filter_map(|k| k["name"].as_str().map(|n| format!("{n}.png"))).collect();
        for entry in std::fs::read_dir(&masks)? {
            let p = entry?.path();
            if !p.file_name().is_some_and(|n| kept.contains(n.to_string_lossy().as_ref())) {
                std::fs::remove_file(p)?;
            }
        }
        metrics.insert("alpha_masks".into(), json!("masks"));
        metrics
            .entry("frame_plan")
            .or_insert(json!({"warnings":[]}))["warnings"]
            .as_array_mut()
            .map(|w| w.push(json!("Images have transparency: alpha saved as masks/ (transparent = ignored); use it as an SfM/training mask.")));
    }
    if sel.kept.is_empty() {
        return Err(invalid("extraction wrote no frames"));
    }
    metrics.insert("frames_decoded".into(), json!(decoded));
    metrics.entry("input_count").or_insert(json!(decoded));
    let input = metrics["input_count"].as_u64().unwrap();
    let selected = sel.kept.len() as u64;
    let corrupt = metrics.get("frame_plan").and_then(|p| p["dropped_corrupt"].as_u64()).unwrap_or(0);
    let dropped = input.saturating_sub(selected);
    anyhow::ensure!(!req.preserve_all || dropped == 0, "preserve_all: selected {selected} of {input} inputs");
    metrics.insert("selected_count".into(), json!(selected));
    metrics.insert("dropped_count".into(), json!(dropped));
    metrics.insert("drop_reasons".into(), json!({"black":sel.black,"duplicate":sel.dup,"decoder_corrupt":corrupt,
        "sampling":dropped.saturating_sub(u64::from(sel.black) + u64::from(sel.dup) + corrupt)}));
    metrics.insert("input_count_basis".into(), json!(if !video { "image_files" } else if metrics.get("from_analysis") == Some(&json!(true)) { "probed_source_frames" } else if req.preserve_all { "decoded_source_frames" } else { "decoded_sampling_timeline" }));
    metrics.insert("preserve_all".into(), json!(req.preserve_all));
    metrics.insert("selection_policy".into(), json!(if req.preserve_all { "all" } else { "sample_and_filter" }));
    metrics.insert("input_kind".into(), json!(if ordered_video { "video" } else { "images" }));
    metrics.insert("frame_count".into(), json!(sel.kept.len()));
    metrics.insert("dropped_black".into(), json!(sel.black));
    metrics.insert("dropped_duplicates".into(), json!(sel.dup));
    metrics.insert("rotate_deg".into(), json!(req.rotate_deg));
    metrics.insert("strategy".into(), serde_json::to_value(req.strategy)?);
    metrics.insert("max_edge".into(), json!(req.max_edge));
    let mut sharp: Vec<f32> = sel.kept.iter().filter_map(|k| k["sharpness"].as_f64()).map(|v| v as f32).collect();
    sharp.sort_by(f32::total_cmp);
    metrics.insert("sharpness_median_kept".into(), json!(percentile(&sharp, 50.0)));
    write_json(&out.join("selection.json"), &json!({"request": req, "metrics": metrics, "frames": sel.kept}))?;
    let frames = list_images(&dest)?;
    let stride = (frames.len() / 8).max(1);
    let warnings = metrics
        .get("frame_plan")
        .and_then(|p| p["warnings"].as_array())
        .map(|w| w.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    Ok(ToolResult {
        artifacts: vec![artifact],
        warnings,
        metrics: metrics.into_iter().collect(),
        images: frames.iter().step_by(stride).take(8).map(|p| rel(run, p)).collect(),
        wall_s: Some(start.elapsed().as_secs_f64()),
        ..result("extract_frames", Status::Succeeded)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_preview_keeps_clear_views_before_a_sustained_detail_loss() {
        let mut samples = vec![800.0; 33];
        samples.extend(vec![60.0; 39]);
        assert_eq!(quick_sharp_prefix(&samples, 8.875), 4.0);
        // A short dip in a textured capture must not shorten its preview.
        let mut samples = vec![800.0; 72];
        samples[33..40].fill(60.0);
        assert_eq!(quick_sharp_prefix(&samples, 8.875), 8.875);
    }

    #[test]
    fn quick_preview_preserves_soft_captures_and_enough_initial_views() {
        assert_eq!(quick_sharp_prefix(&[60.0; 72], 8.875), 8.875);
        let mut samples = vec![800.0; 12];
        samples.extend(vec![60.0; 12]);
        assert_eq!(quick_sharp_prefix(&samples, 2.875), 2.875);
        // Uncached tail slots are not measurements and the caller leaves them out.
        assert_eq!(quick_sharp_prefix(&[], 8.875), 8.875);
    }

    #[test]
    fn full_plan_spends_its_camera_budget_on_clear_views_and_audits_the_break() {
        let track: Vec<_> = (0..160).map(|i| (i as f64 / 8.0, if (33..72).contains(&i) { 0.3 } else { 0.015 })).collect();
        let mut sharpness = vec![800.0; 160];
        sharpness[33..72].fill(60.0);
        let mut plan = frameplan::plan(&track, 20.0, 600, &frameplan::PlanOptions::spaced(150, true));
        let original_motion = plan.motion_at(20.0);
        let omitted = reallocate_clear_views(&mut plan, &sharpness, 30.0);
        assert_eq!(omitted, vec![(4.0, 9.125)]);
        assert_eq!(plan.instants_s.len(), 150);
        assert!(plan.instants_s.windows(2).all(|w| w[0] < w[1]));
        // A thin LOW_DETAIL_FPS (2 views/s) sample stays inside the omitted 5.1 s.
        assert_eq!(plan.instants_s.iter().filter(|&&t| (4.0..9.125).contains(&t)).count(), 10);
        let excluded = low_detail_exclusions(&omitted, &plan.instants_s, 30.0);
        assert!(plan.instants_s.iter().all(|&t| !excluded.iter().any(|r| r.contains(&((t * 30.0).round() as u64)))));
        assert!(plan.instants_s.iter().filter(|&&t| t < 4.0).count() > 10);
        assert!(plan.instants_s.iter().filter(|&&t| t > 9.125).count() > 10);
        // Filtering influences placement, never hides the motion across the missing interval.
        assert_eq!(plan.motion_at(20.0), original_motion);
        // The thin sample spaces the 5.1 s break at 0.51 s (vs ~0.11 s in clear time) and each of those
        // steps crosses over a view width of the break's fast motion, which the audit reports.
        let audit = plan.audit(&plan.instants_s);
        assert_eq!(audit["max_gap_s"].as_f64().unwrap(), 0.51);
        assert!(audit["max_gap_widths"].as_f64().unwrap() > 1.0);
        assert!(audit["median_gap_widths"].as_f64().unwrap() < 0.1);
    }

    #[test]
    fn full_plan_preserves_soft_captures_transient_dips_and_short_clear_fragments() {
        let track: Vec<_> = (0..80).map(|i| (i as f64 / 8.0, 0.02)).collect();
        for sharpness in [vec![60.0; 80], {
            let mut samples = vec![800.0; 80];
            samples[24..31].fill(60.0);
            samples
        }, {
            let mut samples = vec![60.0; 80];
            samples[..16].fill(800.0);
            samples
        }] {
            let mut plan = frameplan::plan(&track, 10.0, 300, &frameplan::PlanOptions::spaced(80, true));
            let original = plan.instants_s.clone();
            assert!(reallocate_clear_views(&mut plan, &sharpness, 30.0).is_empty());
            assert_eq!(plan.instants_s, original);
        }
    }

    /// Textured photos (distinct per index, so none is dropped as a duplicate).
    fn photos(dir: &Path, sizes: &[(u32, u32)]) {
        std::fs::create_dir_all(dir).unwrap();
        for (i, &(w, h)) in sizes.iter().enumerate() {
            let k = i as u32 * 37 + 11;
            let img = RgbImage::from_fn(w, h, |x, y| image::Rgb([((x * k / 7) ^ (y * 3)) as u8, ((y * k / 5) ^ x) as u8, ((x + y) * k / 9) as u8]));
            img.save(dir.join(format!("p{i:02}.jpg"))).unwrap();
        }
    }

    fn extract(sizes: &[(u32, u32)], rotate_deg: u32) -> (tempfile::TempDir, PathBuf, ToolResult) {
        let tmp = tempfile::tempdir().unwrap();
        photos(&tmp.path().join("photos"), sizes);
        let run = splat_core::Run::create(tmp.path().join("run"), tmp.path().join("photos"), "test", "brush").unwrap();
        let req: ExtractFrames = serde_json::from_value(json!({"max_edge": 1600, "rotate_deg": rotate_deg})).unwrap();
        let r = extract_frames(&BrushConfig::from_env(), &run, req).unwrap();
        let dir = run.path(&r.artifacts[0].id).unwrap();
        (tmp, dir, r)
    }

    fn dims(dir: &Path) -> Vec<(u32, u32)> {
        list_images(dir).unwrap().iter().map(|p| image::image_dimensions(p).unwrap()).collect()
    }

    #[test]
    fn photos_over_max_edge_get_sfm_copies() {
        // 2000 px: copy at native size; 3000 px: copy at the SfM edge; training frames at 1600 either way.
        for (size, train, want) in [((2000, 1500), (1600, 1200), (2000, 1500)), ((3000, 2000), (1600, 1067), (2400, 1600))] {
            let (_tmp, dir, r) = extract(&[size; 3], 0);
            assert_eq!(dims(&dir.join("images")), vec![train; 3]);
            assert_eq!(dims(&dir.join("sfm_images")), vec![want; 3]);
            let names = |d: &Path| list_images(d).unwrap().iter().map(|p| p.file_name().unwrap().to_owned()).collect::<Vec<_>>();
            assert_eq!(names(&dir.join("images")), names(&dir.join("sfm_images")));
            assert_eq!(r.metrics["sfm_max_edge"], json!(SFM_EDGE));
            assert_eq!(r.metrics["sfm_images"], json!(3));
        }
        // Rotation applies to both copies.
        let (_tmp, dir, _) = extract(&[(3000, 2000); 2], 90);
        assert_eq!(dims(&dir.join("images")), vec![(1067, 1600); 2]);
        assert_eq!(dims(&dir.join("sfm_images")), vec![(1600, 2400); 2]);
    }

    #[test]
    fn no_sfm_copies_unless_every_photo_exceeds_max_edge_at_one_size() {
        // At or under max_edge: training frames are the photos' own size, nothing else written.
        let (_tmp, dir, r) = extract(&[(1600, 1200), (1200, 900)], 0);
        assert!(!dir.join("sfm_images").exists());
        assert!(r.metrics.get("sfm_images").is_none() && r.metrics.get("sfm_images_skipped").is_none());
        // Mixed: one photo not larger than max_edge.
        let (_tmp, dir, r) = extract(&[(3000, 2000), (1500, 1000)], 0);
        assert!(!dir.join("sfm_images").exists());
        assert!(r.metrics["sfm_images_skipped"].is_string());
        // 2000 and 3000 px share the 1600-px training size but not an SfM size: COLMAP's single
        // camera would drop images, so no copies.
        let (_tmp, dir, r) = extract(&[(3000, 2000), (2100, 1400)], 0);
        assert!(!dir.join("sfm_images").exists());
        assert!(r.metrics["sfm_images_skipped"].is_string());
    }
}

#[cfg(test)]
mod input_contract_tests {
    use super::*;
    use splat_core::Run;

    #[test]
    fn input_contract_241_images_preserve_duplicates_black_and_chronology() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        std::fs::create_dir(&source).unwrap();
        for i in 1..=241 {
            let value = if i == 1 { 0 } else { 120 };
            RgbImage::from_pixel(16, 16, image::Rgb([value; 3])).save(source.join(format!("frame_{i}.png"))).unwrap();
        }
        let run = Run::create(tmp.path().join("run"), &source, "input contract", "brush").unwrap();
        let cfg = BrushConfig::from_env();
        let req = |v| serde_json::from_value::<ExtractFrames>(v).unwrap();
        // A requested count alone does not override filtering.
        let filtered = extract_frames(&cfg, &run, req(json!({"target_count":241}))).unwrap();
        assert_eq!(filtered.metrics["input_count"], 241);
        assert!(filtered.metrics["frame_count"].as_u64().unwrap() < 241);
        let all = extract_frames(&cfg, &run, req(json!({"target_count":20,"preserve_all":true,"input_kind":"video"}))).unwrap();
        assert_eq!(all.metrics["input_count"], 241);
        assert_eq!(all.metrics["selected_count"], 241);
        assert_eq!(all.metrics["dropped_count"], 0);
        assert_eq!(all.metrics["dropped_black"], 0);
        assert_eq!(all.metrics["dropped_duplicates"], 0);
        assert_eq!(all.artifacts[0].meta["input_kind"], "video");
        let selection: Value = read_json(&run.path(&all.artifacts[0].id).unwrap().join("selection.json")).unwrap();
        for (i, frame) in selection["frames"].as_array().unwrap().iter().enumerate() {
            assert_eq!(frame["source_name"], format!("frame_{}.png", i + 1));
            assert_eq!(frame["source_index"], i);
        }
        assert!(extract_frames(&cfg, &run, req(json!({"preserve_all":true,"time_range_s":[0,1]}))).is_err());
    }
}
