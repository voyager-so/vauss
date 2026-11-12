//! Cheap capture-quality statistics from grayscale samples piped out of ffmpeg.
//! Port of eval/video_stats.py plus the inspect-time fields the agent reads.

use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use serde_json::Value;

/// Sample size fed to the statistics (matches eval/video_stats.py).
pub const SAMPLE_WIDTH: usize = 640;
pub const SAMPLE_HEIGHT: usize = 360;
/// `upside_down_score` above this marks the capture orientation-suspect (gsplat_frames.py).
pub const UPSIDE_DOWN_THRESHOLD: f64 = 0.6;

#[derive(Clone, Debug)]
pub struct VideoStatsOptions {
    pub fps: f64,
    pub max_frames: Option<usize>,
    /// ffmpeg / ffprobe executables (located by doctor/config); defaults to PATH lookup.
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Default for VideoStatsOptions {
    fn default() -> Self {
        Self { fps: 4.0, max_frames: None, ffmpeg: "ffmpeg".into(), ffprobe: "ffprobe".into() }
    }
}

/// Grayscale frame, 8-bit, row-major.
#[derive(Clone, Debug)]
pub struct Gray {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FrameStats {
    pub sample_count: usize,
    pub sharpness_median: f64,
    pub sharpness_p10: f64,
    pub sharpness_p10_over_median: f64,
    pub luminance_min: f64,
    pub luminance_max: f64,
    pub luminance_range: f64,
    /// Frames with > 98% of pixels below 16.
    pub black_frac: f64,
    pub top_clipped_fraction: f64,
    /// Consecutive samples with mean abs difference < 1 (out of 255).
    pub static_frac: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Probe {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_s: f64,
    pub codec: String,
    /// Display rotation from stream side data / rotate tag, degrees (0 if none).
    pub rotation_deg: i32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct VideoStats {
    #[serde(flatten)]
    pub probe: Probe,
    #[serde(flatten)]
    pub frames: FrameStats,
    /// Fraction of samples whose bottom third is clearly brighter (> 15/255) and bluer than the
    /// top third, i.e. sky at the bottom. Outdoor-capture heuristic ported from gsplat_frames.py.
    pub upside_down_score: Option<f64>,
    /// Rotation metadata present, or `upside_down_score` above [`UPSIDE_DOWN_THRESHOLD`].
    pub orientation_suspect: bool,
}

/// Sample at `fps` (scaled to 640x360 gray) and summarise.
pub fn video_stats(path: &Path, opts: &VideoStatsOptions) -> Result<VideoStats> {
    let probe = probe(path, &opts.ffprobe)?;
    let (frames, upside_down_score) = sample_stats(path, opts)?;
    let orientation_suspect = probe.rotation_deg != 0 || upside_down_score > UPSIDE_DOWN_THRESHOLD;
    Ok(VideoStats { probe, frames, upside_down_score: Some(upside_down_score), orientation_suspect })
}

pub fn probe(path: &Path, ffprobe: &Path) -> Result<Probe> {
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-show_streams", "-show_format", "-of", "json"])
        .arg(path)
        .output()
        .with_context(|| format!("running {}", ffprobe.display()))?;
    if !out.status.success() {
        bail!("ffprobe failed on {} ({}): {}", path.display(), out.status, tail(&out.stderr));
    }
    let data: Value = serde_json::from_slice(&out.stdout).context("parsing ffprobe JSON")?;
    parse_probe(&data).with_context(|| format!("probing {}", path.display()))
}

/// Reads the first video stream of `ffprobe -show_streams -show_format -of json` output.
pub fn parse_probe(data: &Value) -> Result<Probe> {
    let s = data["streams"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|s| s["codec_type"] == "video")
        .context("no_video_track: source has no video stream")?;
    // A Display Matrix side-data rotation wins over the legacy `rotate` tag.
    let mut rotation = number(&s["tags"]["rotate"]).unwrap_or(0.0);
    for side in s["side_data_list"].as_array().into_iter().flatten() {
        if let Some(r) = number(&side["rotation"]) {
            rotation = r;
        }
    }
    // Some webm muxers write a bogus avg_frame_rate (cambridge_heath_rd: 30000/1 vs r 30/1); an
    // average above the base rate is impossible, so fall back to r_frame_rate then.
    let fps = match (rate(&s["avg_frame_rate"]), rate(&s["r_frame_rate"])) {
        (Some(avg), Some(base)) if avg > base => base,
        (Some(avg), _) => avg,
        (None, base) => base.unwrap_or(0.0),
    };
    let duration_s = number(&s["duration"]).or_else(|| number(&data["format"]["duration"])).unwrap_or(0.0);
    let dim = |k: &str| {
        s[k].as_u64().and_then(|v| u32::try_from(v).ok()).with_context(|| format!("video stream has no {k}"))
    };
    Ok(Probe {
        width: dim("width")?,
        height: dim("height")?,
        fps,
        duration_s,
        codec: s["codec_name"].as_str().unwrap_or_default().to_owned(),
        rotation_deg: rotation.round() as i32,
    })
}

/// ffprobe emits most numbers as strings ("71.680000"); side-data rotation is a JSON number.
fn number(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str()?.trim().parse().ok())
}

/// "num/den" -> Some(rate) when positive.
fn rate(v: &Value) -> Option<f64> {
    let (n, d) = v.as_str()?.split_once('/')?;
    let r = n.parse::<f64>().ok()? / d.parse::<f64>().ok()?;
    (r.is_finite() && r > 0.0).then_some(r)
}

/// One decode pass streamed frame by frame. Each output frame is four stacked 640x360 gray planes:
/// ffmpeg's own gray conversion (byte-identical to `-pix_fmt gray` in video_stats.py, which is a
/// range-expanded luma we can't reproduce exactly by hand) followed by R, G and B.
fn sample_stats(path: &Path, opts: &VideoStatsOptions) -> Result<(FrameStats, f64)> {
    ensure!(opts.fps > 0.0 && opts.max_frames != Some(0), "fps and max_frames must be positive");
    let (w, h) = (SAMPLE_WIDTH, SAMPLE_HEIGHT);
    let graph = format!(
        "fps={},split[a][b];[a]scale={w}:{h},format=gray[y];\
         [b]scale={w}:{h},format=gbrp,extractplanes=r+g+b[r][g][c];[y][r][g][c]vstack=inputs=4",
        opts.fps
    );
    let mut cmd = Command::new(&opts.ffmpeg);
    cmd.args(["-v", "error", "-nostdin", "-i"]).arg(path).args(["-filter_complex", &graph]);
    if let Some(n) = opts.max_frames {
        cmd.args(["-frames:v", &n.to_string()]);
    }
    let mut child = cmd
        .args(["-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("running {}", opts.ffmpeg.display()))?;
    let mut stderr = child.stderr.take().expect("piped stderr");
    let stderr_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        buf
    });
    let mut stdout = child.stdout.take().expect("piped stdout");

    let plane = w * h;
    let mut acc = Accumulator::new(w, h)?;
    let mut upside = UpsideDown::default();
    let (mut cur, mut prev) = (vec![0u8; 4 * plane], vec![0u8; 4 * plane]);
    let mut have_prev = false;
    let read = loop {
        match read_frame(&mut stdout, &mut cur) {
            Ok(true) => {
                acc.push(&cur[..plane], have_prev.then(|| &prev[..plane]));
                upside.push(&cur[plane..], w, h);
                std::mem::swap(&mut cur, &mut prev);
                have_prev = true;
            }
            Ok(false) => break Ok(()),
            Err(e) => break Err(e),
        }
    };
    drop(stdout);
    let status = child.wait()?;
    let err = stderr_reader.join().unwrap_or_default();
    if !status.success() {
        bail!("ffmpeg failed on {} ({status}): {}", path.display(), tail(&err));
    }
    read.with_context(|| format!("reading ffmpeg frames for {}", path.display()))?;
    let stats = acc.finish().with_context(|| format!("no frames decoded from {}: {}", path.display(), tail(&err)))?;
    Ok((stats, upside.score()))
}

/// Fills `buf` with one frame; Ok(false) on a clean end of stream.
fn read_frame(r: &mut impl Read, buf: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated frame")),
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

/// Last ~2000 chars of a tool's stderr, for error messages.
fn tail(stderr: &[u8]) -> String {
    let s = String::from_utf8_lossy(stderr);
    let s = s.trim();
    match s.char_indices().rev().nth(1999) {
        Some((i, _)) => format!("...{}", &s[i..]),
        None if s.is_empty() => "(no stderr)".into(),
        None => s.to_owned(),
    }
}

pub fn summarize_frames(frames: &[Gray]) -> Result<FrameStats> {
    let first = frames.first().context("summarize_frames: no frames")?;
    let mut acc = Accumulator::new(first.width, first.height)?;
    let mut prev: Option<&Gray> = None;
    for f in frames {
        ensure!(
            (f.width, f.height) == (first.width, first.height),
            "frame size {}x{} differs from {}x{}",
            f.width,
            f.height,
            first.width,
            first.height
        );
        ensure!(f.data.len() == f.width * f.height, "frame buffer is {} bytes, expected {}x{}", f.data.len(), f.width, f.height);
        acc.push(&f.data, prev.map(|p| p.data.as_slice()));
        prev = Some(f);
    }
    acc.finish()
}

/// Per-frame statistics gathered on the fly so long videos need not be buffered.
struct Accumulator {
    width: usize,
    height: usize,
    lap_var: Vec<f64>,
    lum: Vec<f64>,
    diffs: Vec<f64>,
    black: usize,
    top_clipped: u64,
}

impl Accumulator {
    fn new(width: usize, height: usize) -> Result<Self> {
        ensure!(width >= 3 && height >= 3, "expected grayscale frames of size 3x3 or larger, got {width}x{height}");
        Ok(Self { width, height, lap_var: vec![], lum: vec![], diffs: vec![], black: 0, top_clipped: 0 })
    }

    fn push(&mut self, px: &[u8], prev: Option<&[u8]>) {
        let (w, h) = (self.width, self.height);
        let n = (w * h) as f64;
        self.lap_var.push(laplacian_var(px, w, h));
        self.lum.push(px.iter().map(|&p| p as u64).sum::<u64>() as f64 / n);
        if px.iter().filter(|&&p| p < 16).count() as f64 / n > 0.98 {
            self.black += 1;
        }
        self.top_clipped += px[..(h / 3) * w].iter().filter(|&&p| p >= 236).count() as u64;
        if let Some(prev) = prev {
            let d: u64 = px.iter().zip(prev).map(|(&a, &b)| a.abs_diff(b) as u64).sum();
            self.diffs.push(d as f64 / n);
        }
    }

    fn finish(self) -> Result<FrameStats> {
        let count = self.lap_var.len();
        ensure!(count > 0, "no frames");
        let mut lap = self.lap_var;
        lap.sort_by(f64::total_cmp);
        let (median, p10) = (percentile(&lap, 50.0), percentile(&lap, 10.0));
        let lum_min = self.lum.iter().copied().fold(f64::INFINITY, f64::min);
        let lum_max = self.lum.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let top_pixels = (count * (self.height / 3) * self.width) as f64;
        let static_frac = (!self.diffs.is_empty())
            .then(|| self.diffs.iter().filter(|&&d| d < 1.0).count() as f64 / self.diffs.len() as f64);
        Ok(FrameStats {
            sample_count: count,
            sharpness_median: median,
            sharpness_p10: p10,
            sharpness_p10_over_median: p10 / median.max(1e-6),
            luminance_min: lum_min,
            luminance_max: lum_max,
            luminance_range: lum_max - lum_min,
            black_frac: self.black as f64 / count as f64,
            top_clipped_fraction: self.top_clipped as f64 / top_pixels,
            static_frac,
        })
    }
}

/// Population variance of the interior 4-neighbour Laplacian, computed exactly in integers
/// (numpy computes it in float32; differences are rounding only).
fn laplacian_var(px: &[u8], w: usize, h: usize) -> f64 {
    let (mut sum, mut sq) = (0i64, 0i64);
    for y in 1..h - 1 {
        let (up, row, down) = (&px[(y - 1) * w..y * w], &px[y * w..(y + 1) * w], &px[(y + 1) * w..(y + 2) * w]);
        for x in 1..w - 1 {
            let l = row[x - 1] as i64 + row[x + 1] as i64 + up[x] as i64 + down[x] as i64 - 4 * row[x] as i64;
            sum += l;
            sq += l * l;
        }
    }
    let n = ((w - 2) * (h - 2)) as i128;
    ((n * sq as i128 - (sum as i128).pow(2)) as f64) / (n * n) as f64
}

/// numpy.percentile with the default linear interpolation; `sorted` must be non-empty.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let pos = q / 100.0 * (sorted.len() - 1) as f64;
    let (lo, frac) = (pos.floor() as usize, pos.fract());
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

#[derive(Default)]
struct UpsideDown {
    frames: usize,
    flipped: usize,
}

impl UpsideDown {
    /// `rgb` holds the R, G and B planes of one w x h frame, back to back.
    fn push(&mut self, rgb: &[u8], w: usize, h: usize) {
        let band = (h / 3) * w;
        let (top, bottom) = (third_means(rgb, w * h, 0, band), third_means(rgb, w * h, w * h - band, band));
        self.frames += 1;
        if bottom.0 - top.0 > 15.0 && bottom.1 > top.1 {
            self.flipped += 1;
        }
    }

    fn score(&self) -> f64 {
        if self.frames == 0 { 0.0 } else { self.flipped as f64 / self.frames as f64 }
    }
}

/// (mean RGB brightness, mean blueness B - (R+G)/2) of `len` pixels starting at `start` in each plane.
fn third_means(rgb: &[u8], plane: usize, start: usize, len: usize) -> (f64, f64) {
    let sum = |c: usize| rgb[c * plane + start..c * plane + start + len].iter().map(|&p| p as u64).sum::<u64>() as f64;
    let (r, g, b) = (sum(0), sum(1), sum(2));
    let n = len as f64;
    ((r + g + b) / (3.0 * n), (b - (r + g) / 2.0) / n)
}
