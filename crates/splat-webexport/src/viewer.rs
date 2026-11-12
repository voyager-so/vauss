//! `vauss export-viewer`: a double-click (file://) Gaussian-splat viewer folder for a web package,
//! tuned for weak or busy GPUs. `vauss run` writes one at `<run>/exports/viewer/` by default.
//!
//! The page ([`PAGE`], viewer/index.html) is a Spark 2.2.0 viewer. Each block is tagged in the template:
//!  1. FAST OPEN: data/first.js (~40K splats) renders first; the model data comes in parts
//!     (data/<key>-NN.js, decoded part by part with yields), so the first picture stays live.
//!  2. IDLE RULE (stale sort): keeps drawing >= 3 s after a camera move and until 1 s after the last
//!     scene change / sort / LoD activity; POST-SORT FRAME: never idles before a settled frame was drawn
//!     with the newest sort's order.
//!  3. RE-SORT AFTER SWAP: every mesh show/hide/add forces a Spark re-sort and counts as a move.
//!  4. AUTO FULL: opens on the interactive (<= 1M) model and swaps in the full model (a cloud run's
//!     uncapped candidate) once it is decoded. Off: ?autofull=0 or ?variant=.
//!     MOTION LOD: both stay loaded, each in its own Spark renderer (layers 1 and 2). While the camera
//!     moves the canvas shows the interactive model; once still, the full model is sorted for that view
//!     off screen (a fresh sort after settling), then crossfaded in over 280 ms (both frames drawn once
//!     into targets, blended in a full-screen pass; the last frame is the full model's). After a move
//!     the last (reduced-resolution)
//!     moving frame stays up while the full model sorts (at most 4 s, ?fullwait=ms) and the fade starts
//!     from it, so the resolution step is inside the same blend. A move cuts back to the 1M at once.
//!     ?motionlod=0: the full model while moving too; ?fade=0: cut instead of the crossfade.
//!  5. Named capture views preserve framing; navigation has no capture-path or heading leash.
//!  6. VIEW PICKER: "Jump to view…" whenever the camera is not at a named view.
//!  7. SCRIPT HOOKS: window.__viewer.setCamera / dbg() / three for scripted checks.
//!  8. NAVIGATION: frame-time exponential damping (~20 ms), double-click / double-tap Gaussian-raycast
//!     pivot with a brief marker and 600 ms turn; one-finger orbit, two-finger pan + pinch; trackpad pinch
//!     dollies, Shift-scroll pans and physical Ctrl-scroll orbits. WASD/QE fly at >= 1.5 m/s (scaled by
//!     metersPerUnit and scene size), with eased velocity, Shift x3 and Ctrl x0.25.
//!  9. MOVING RESOLUTION: starts at full DPR, shrinks 20% after a >18 ms moving frame, and probes upwards
//!     by 8% after 60 consecutive clean frames. ?movefloor= keeps its default 0.65 floor.
//! 10. COLLIDE: soft camera collision against the room's voxel occupancy (data/collide-NN.js, see
//!     [`crate::viewer_collide`]; never fails the export: without a grid the page has no collision). The
//!     camera is a 0.25 m sphere: each frame's move (fly, orbit, pan, dolly, pivot turn) is resolved in
//!     sub-steps per axis, so it stops at walls and slides along them; a camera already inside solid may move
//!     only where the penetration does not grow. Orbit / dolly is a third-person boom: the pivot -> camera ray
//!     is marched through the grid (DDA) and the radius shortened before a wall (at once), then eased back
//!     out. Named-view jumps are exempt. No capture-path clamp. Off: ?collide=0 (or --no-collide).
//! 11. WALK: rooms (a package with a collision grid) open in first-person walk mode; O toggles orbit, ?mode=orbit
//!     opens in it (keeping the named view's look_at as the pivot). Drag looks, WASD / arrows walk level with the
//!     floor, Shift x3, scroll steps along the view, right-drag slides, double-click walks toward a surface, R resets
//!     the view, H hides the help line; Q/E (down / up) are orbit / fly only. The camera's up is the room's levelled
//!     vertical (collision grid `up`, from the game's levelling), with any up vector given an orthonormal floor basis.
//!     Walk collision is a standing capsule (1.5 m eye height, hard spheres, no stop distance) that slides along
//!     walls and steps up at most 20 cm. Spawns (named views, R, orbit -> walk) stand on the floor: CFG.floorHeight
//!     (exported along the chosen up) where it supports the spot, else the first support below the eye; an occupied
//!     spawn searches nearby for clear space. Movement never passes through solid. The grid's bounds are storage,
//!     not walls; near-camera splats remain visible.
//!
//! Each data/*.js is a classic script setting window.__SPLAT_DATA[key] = base64 bytes (script tags load
//! over file://; fetch and HTTP Range do not). Spark (MIT, World Labs) is embedded with the same edits as
//! the Spark SH compatibility patch below, applied here with checked counts. three.js loads from jsDelivr.
//! URL flags: ?mode=walk|orbit ?variant=interactive|lod|full ?autofull=0 ?motionlod=0 ?fade=0 ?collide=0 ?budget=N ?bench=1 ?fps=0 ?fast=0
//! ?exact=0 ?idle=0 ?throttle=0 ?movefloor=0..1 (default 0.65 of native DPR while moving).
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use serde_json::{Value, json};

const PAGE: &str = include_str!("viewer/index.html");
/// Upstream @sparkjsdev/spark@2.2.0 dist/spark.module.js (sha256 d5c3b372…54ead7), zstd -19.
const SPARK_ZST: &[u8] = include_bytes!("viewer/spark-2.2.0.module.js.zst");
pub(crate) const SPARK_VERSION: &str = "2.2.0";
pub(crate) const THREE: &str = "https://cdn.jsdelivr.net/npm/three@0.186.0/build/three.module.js";
pub(crate) const THREE_ADDONS: &str = "https://cdn.jsdelivr.net/npm/three@0.186.0/examples/jsm/";
const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

#[derive(clap::Args, Debug, Clone)]
pub struct ExportViewerArgs {
    /// A `vauss run` dir (uses exports/web) or a web package dir.
    pub source: PathBuf,
    /// Output folder (default: <run>/exports/viewer, or <package>/../viewer).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Full model: web dir, scene.spz or candidate dir (default: a cloud run's uncapped candidate, from its journal).
    #[arg(long)]
    pub full: Option<PathBuf>,
    /// Page title (default: the run's source file name).
    #[arg(long)]
    pub title: Option<String>,
    /// Capture poses in the view picker when --views is not given (spread over the capture).
    #[arg(long, default_value_t = 6)]
    pub cameras: usize,
    /// Named views JSON: [{"label","capture":i} | {"label","image":"frame_x.jpg"} | {"label","frame":N}
    /// (optional "pitch"/"yaw" deg) | {"label","position","look_at","vfov_deg"}]; the first is the start view.
    #[arg(long)]
    pub views: Option<PathBuf>,
    /// Capture tan(hfov/2)/tan(vfov/2) (default: the COLMAP cameras.bin of the package's source.data).
    #[arg(long)]
    pub capture_aspect: Option<f64>,
    /// Omit capture-path metadata (legacy option; navigation is always free of path/pose guards).
    #[arg(long)]
    pub no_guard: bool,
    /// Default LoD splat budget (Spark lodSplatCount).
    #[arg(long, default_value_t = 1_000_000)]
    pub budget: u64,
    /// No interactive SPZ: open on the LoD stream.
    #[arg(long)]
    pub no_interactive_spz: bool,
    /// Raw MB per data/<key>-NN.js part.
    #[arg(long, default_value_t = 4.0)]
    pub part_mb: f64,
    /// No camera collision (no data/collide-NN.js voxel grid).
    #[arg(long)]
    pub no_collide: bool,
}

impl ExportViewerArgs {
    /// The defaults `vauss run` uses for `<run>`.
    pub fn for_run(run: &Path) -> Self {
        Self { source: run.to_owned(), out: None, full: None, title: None, cameras: 6, views: None,
            capture_aspect: None, no_guard: false, budget: 1_000_000, no_interactive_spz: false, part_mb: 4.0, no_collide: false }
    }
}

// ---------------------------------------------------------------- Spark patches
// Same edits as the Spark SH compatibility patch below (Vite plugin), applied to the release bundle:
//  * sparkjsdev/spark#465 (SH exponent rounds up; JS, GLSL and the 30 inlined wasm sites),
//  * sparkjsdev/spark#424 (paged extended splats with SH3: 2.2.0's pager allocates too few SH
//    textures, so the second page upload throws and the .rad stream stops after the root chunk).
// Every count is checked, so a different bundle fails the export instead of shipping unpatched.
const JS_FIX: (&str, &str) = ("const base = Math.floor(Math.log2(maxAbs));", "const base = Math.ceil(Math.log2(maxAbs));");
const GLSL_FIX: (&str, &str) = ("int base = clamp(int(floor(log2(maxAbs))) + 15, 0, 31);",
    "int base = clamp(int(ceil(log2(maxAbs))) + 15, 0, 31);");
const PAGER_FIX: [(&str, &str); 3] = [
    ("  ensureShTextures(numSh) {\n    this.curSh = Math.max(this.curSh, numSh);\n",
     "  ensureShTextures(numTextures) {\n"),
    ("    for (let i = 0; i < this.curSh; i++) {\n      if (this.shTextures[i].value === emptyShTextures[i]) {",
     "    for (let i = 0; i < numTextures; i++) {\n      if (this.shTextures[i].value === emptyShTextures[i]) {"),
    ("    const numSh = Math.min(shArrays.length, 3);\n    this.ensureShTextures(numSh);",
     "    this.curSh = Math.max(this.curSh, Math.min(shArrays.length, 3));\n    this.ensureShTextures(shArrays.length);"),
];
const WASM_FROM: [u8; 7] = [0x8E, 0x43, 0x00, 0x00, 0x70, 0x41, 0x92]; // f32.floor; f32.const 15.0; f32.add
const WASM_SITES: usize = 30;
const F32_CEIL: u8 = 0x8D;

fn replace_once(code: &str, (from, to): (&str, &str)) -> Result<String> {
    let n = code.matches(from).count();
    ensure!(n == 1, "spark patch: expected 1 {:?}, found {n}", &from[..from.len().min(60)]);
    Ok(code.replacen(from, to, 1))
}

/// The embedded Spark bundle with the patches above.
pub fn spark_source() -> Result<String> {
    let mut code = String::new();
    zstd::stream::read::Decoder::new(SPARK_ZST)?.read_to_string(&mut code)?;
    code = replace_once(&code, JS_FIX)?;
    code = replace_once(&code, GLSL_FIX)?;
    for fix in PAGER_FIX {
        code = replace_once(&code, fix)?;
    }
    // Inlined wasm modules: base64 runs of >= 100000 chars (optionally '='-padded) that decode to "\0asm".
    let is_b64 = |c: u8| c.is_ascii_alphanumeric() || c == b'+' || c == b'/';
    let bytes = code.as_bytes();
    let (mut out, mut patched, mut i, mut copied) = (String::with_capacity(code.len()), 0, 0, 0);
    while i < bytes.len() {
        if !is_b64(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && is_b64(bytes[i]) {
            i += 1;
        }
        let run_end = i;
        while i < bytes.len() && i - run_end < 2 && bytes[i] == b'=' {
            i += 1;
        }
        if run_end - start < 100_000 {
            continue;
        }
        let Ok(mut raw) = B64.decode(&code[start..i]) else { continue };
        if !raw.starts_with(b"\0asm") {
            continue;
        }
        let mut k = 0;
        while let Some(at) = raw[k..].windows(WASM_FROM.len()).position(|w| w == WASM_FROM) {
            raw[k + at] = F32_CEIL;
            patched += 1;
            k += at + 1;
        }
        out.push_str(&code[copied..start]);
        out.push_str(&B64.encode(&raw));
        copied = i;
    }
    out.push_str(&code[copied..]);
    ensure!(patched == WASM_SITES, "spark patch: expected {WASM_SITES} wasm sites, found {patched}");
    // The bundle is imported from a blob: URL; its source map would 404.
    if let Some(at) = out.rfind("\n//# sourceMappingURL=") {
        if !out[at + 1..].trim_end().contains('\n') {
            out.truncate(at + 1);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- inputs
fn journal(run: &Path) -> Vec<Value> {
    fs::read_to_string(run.join("journal.jsonl"))
        .map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
        .unwrap_or_default()
}

/// The full model's candidate dir when a cloud run's interactive (capped) refine replaced exports/web.
fn full_candidate(run: &Path) -> Option<PathBuf> {
    let (mut started, mut ready) = (None, false);
    for ev in journal(run) {
        match ev["event"].as_str() {
            Some("interactive_started") => {
                started = ev["data"]["from_candidate"].as_str().map(str::to_owned);
                ready = false;
            }
            Some("interactive_ready") => ready = true,
            _ => {}
        }
    }
    started.filter(|_| ready).map(|c| run.join("artifacts").join(c))
}

fn spz_in(web: &Path) -> Option<PathBuf> {
    let scene: Value = serde_json::from_slice(&fs::read(web.join("scene.json")).ok()?).ok()?;
    let f = web.join(scene["splat"].as_str().unwrap_or("scene.spz"));
    f.is_file().then_some(f)
}

pub(crate) fn resolve_full(arg: Option<&Path>, run: &Path) -> Result<Option<PathBuf>> {
    let Some(p) = arg.map(Path::to_owned).or_else(|| full_candidate(run)) else { return Ok(None) };
    if p.is_file() && p.extension().is_some_and(|e| e == "spz") {
        return Ok(Some(p));
    }
    if let Some(spz) = [p.clone(), p.join("web")].iter().find_map(|w| spz_in(w)) {
        return Ok(Some(spz));
    }
    let msg = format!("full model: no scene.spz under {} (export it with `splat export-web --lod skip`)", p.display());
    if arg.is_some() {
        bail!(msg);
    }
    eprintln!("export-viewer: {msg}; the viewer has the interactive model only");
    Ok(None)
}

pub(crate) fn spz_header(p: &Path) -> Result<Value> {
    let file = fs::File::open(p).with_context(|| format!("reading {}", p.display()))?;
    let mut head = [0u8; 16];
    let mut magic = [0u8; 2];
    let mut f = std::io::BufReader::new(file);
    f.read_exact(&mut magic)?;
    ensure!(magic == [0x1f, 0x8b], "{}: not a gzip SPZ (v2/v3)", p.display());
    let file = fs::File::open(p)?;
    flate2::read::MultiGzDecoder::new(file).read_exact(&mut head).with_context(|| format!("{}: bad SPZ", p.display()))?;
    let u32_at = |o: usize| u32::from_le_bytes(head[o..o + 4].try_into().unwrap());
    ensure!(u32_at(0) == 0x5053474E, "{}: not an SPZ", p.display());
    Ok(json!({"version": u32_at(4), "splats": u32_at(8), "sh_degree": head[12], "fractional_bits": head[13]}))
}

fn rad_header(p: &Path) -> Result<Value> {
    let mut f = fs::File::open(p)?;
    let mut head = [0u8; 8];
    f.read_exact(&mut head)?;
    ensure!(&head[..4] == b"RAD0", "{}: not a .rad", p.display());
    let mut meta = vec![0u8; u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize];
    f.read_exact(&mut meta)?;
    let meta: Value = serde_json::from_slice(&meta)?;
    let chunks = meta["chunks"].as_array().cloned().unwrap_or_default();
    ensure!(!chunks.iter().any(|c| c.get("filename").is_some_and(|f| !f.is_null())),
        "{}: chunked .rad (separate chunk files) cannot be embedded", p.display());
    Ok(json!({"nodes": meta["count"], "max_sh": meta["maxSh"], "chunks": chunks.len()}))
}

pub(crate) fn write_data_js(raw: &[u8], dst: &Path, key: &str) -> Result<u64> {
    let mut js = String::with_capacity(raw.len() * 4 / 3 + 128);
    js.push_str("window.__SPLAT_DATA=window.__SPLAT_DATA||{};\nwindow.__SPLAT_DATA[\"");
    js.push_str(key);
    js.push_str("\"]=\"");
    B64.encode_string(raw, &mut js);
    js.push_str("\";\n");
    fs::write(dst, &js)?;
    Ok(js.len() as u64)
}

/// data/<key>-NN.js: base64 of consecutive raw slices (the page decodes them one at a time).
pub(crate) fn write_parts(raw: &[u8], data: &Path, key: &str, part_bytes: usize) -> Result<(usize, u64)> {
    let n = raw.len().div_ceil(part_bytes).max(1);
    let step = raw.len().div_ceil(n).max(1);
    let (mut js, mut parts) = (0, 0);
    for part in raw.chunks(step) {
        let k = format!("{key}-{parts:02}");
        js += write_data_js(part, &data.join(format!("{k}.js")), &k)?;
        parts += 1;
    }
    Ok((parts, js))
}

// ---------------------------------------------------------------- capture path + named views
pub(crate) fn v3(v: &Value) -> Option<[f64; 3]> {
    let a = v.as_array()?;
    (a.len() == 3).then(|| [a[0].as_f64().unwrap_or(0.), a[1].as_f64().unwrap_or(0.), a[2].as_f64().unwrap_or(0.)])
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt().max(1e-12);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// FOV aspect (tan-half-FOV width/height, = width/height for square pixels), height (px) and
/// vertical focal (px at that height) of the median capture camera, from <data>/sparse/0/cameras.bin
/// (or <data>/cameras.bin). scene.json's `capture_aspect` / `capture_height` / `capture_focal_px`.
pub fn colmap_capture(d: &Path) -> Option<(f64, f64, f64)> {
    let params = |model: i32| -> Option<usize> {
        Some(match model { 0 => 3, 1 | 2 | 8 => 4, 3 | 7 | 9 => 5, 4 | 5 => 8, 6 | 10 => 12, _ => return None })
    };
    for p in [d.join("sparse/0/cameras.bin"), d.join("cameras.bin")] {
        let Ok(b) = fs::read(&p) else { continue };
        let mut at = 0;
        let mut take = |n: usize| -> Option<&[u8]> { let s = b.get(at..at + n)?; at += n; Some(s) };
        let count = u64::from_le_bytes(take(8)?.try_into().ok()?);
        let mut asp = Vec::new();
        for _ in 0..count {
            let h = take(24)?;
            let model = i32::from_le_bytes(h[4..8].try_into().ok()?);
            let w = u64::from_le_bytes(h[8..16].try_into().ok()?) as f64;
            let hh = u64::from_le_bytes(h[16..24].try_into().ok()?) as f64;
            let prm: Vec<f64> = take(8 * params(model)?)?.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
            let (fx, fy) = if matches!(model, 0 | 2 | 3 | 8 | 9) { (prm[0], prm[0]) } else { (prm[0], prm[1]) };
            if w > 0. && hh > 0. && fx > 0. && fy > 0. {
                asp.push(((w / fx) / (hh / fy), hh, fy));
            }
        }
        if !asp.is_empty() {
            asp.sort_by(|a, b| a.0.total_cmp(&b.0));
            return Some(asp[asp.len() / 2]);
        }
    }
    None
}

fn frame_index(name: Option<&str>) -> Option<u64> {
    let stem = Path::new(name?).file_stem()?.to_str()?;
    stem.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).last()?.parse().ok()
}

/// The package's capture cameras in capture order (frame number in the image name, else as listed).
pub(crate) fn capture_path(scene: &Value) -> Vec<Value> {
    let cams: Vec<Value> = scene["capture_cameras"].as_array().into_iter().flatten()
        .filter(|c| v3(&c["position"]).is_some() && v3(&c["look_at"]).is_some()).cloned().collect();
    let mut order: Vec<usize> = (0..cams.len()).collect();
    order.sort_by_key(|&i| {
        let f = frame_index(cams[i]["image"].as_str());
        (f.is_none(), f.unwrap_or(0), i)
    });
    order.into_iter().map(|i| cams[i].clone()).collect()
}

/// Rodrigues rotation of v about axis by deg.
fn rot(v: [f64; 3], axis: [f64; 3], deg: f64) -> [f64; 3] {
    let k = unit(axis);
    let (s, c) = deg.to_radians().sin_cos();
    let (kxv, kv) = (cross(k, v), dot(k, v));
    std::array::from_fn(|i| v[i] * c + kxv[i] * s + k[i] * kv * (1. - c))
}

/// Capture camera c's look_at with its look direction turned by yaw (deg, + left about up) then pitch (deg, + up).
fn turned(c: &Value, up: [f64; 3], pitch: f64, yaw: f64) -> Value {
    if pitch == 0. && yaw == 0. {
        return c["look_at"].clone();
    }
    let p = v3(&c["position"]).unwrap();
    let d = rot(sub(v3(&c["look_at"]).unwrap(), p), up, yaw);
    let d = rot(d, cross(d, up), pitch);
    json!([0, 1, 2].map(|i| ((p[i] + d[i]) * 1e4).round() / 1e4))
}

/// `n` capture poses spread over the capture: farthest-point sampling on position (in scene radii) plus
/// view direction (1 - cos), seeded at the pose nearest the start camera; returned in capture order.
fn spread_poses(path: &[Value], n: usize, seed: Option<[f64; 3]>, radius: f64) -> Vec<usize> {
    if path.len() <= n {
        return (0..path.len()).collect();
    }
    let pose = |c: &Value| { let p = v3(&c["position"]).unwrap(); (p, unit(sub(v3(&c["look_at"]).unwrap(), p))) };
    let poses: Vec<_> = path.iter().map(pose).collect();
    let dist = |a: usize, b: usize| {
        let d = sub(poses[a].0, poses[b].0);
        dot(d, d).sqrt() / radius.max(1e-6) + (1. - dot(poses[a].1, poses[b].1))
    };
    let first = seed.map_or(0, |s| (0..poses.len())
        .min_by(|&a, &b| { let (da, db) = (sub(poses[a].0, s), sub(poses[b].0, s)); dot(da, da).total_cmp(&dot(db, db)) })
        .unwrap());
    let mut picked = vec![first];
    let mut near: Vec<f64> = (0..poses.len()).map(|i| dist(i, first)).collect();
    while picked.len() < n {
        let next = (0..poses.len()).max_by(|&a, &b| near[a].total_cmp(&near[b])).unwrap();
        picked.push(next);
        for i in 0..poses.len() {
            near[i] = near[i].min(dist(i, next));
        }
    }
    picked.sort_unstable();
    picked.dedup();
    picked
}

fn named_views(scene: &Value, path: &[Value], views_file: Option<&Path>, n_default: usize) -> Result<Vec<Value>> {
    let up = v3(&scene["up"]).unwrap_or([0., 1., 0.]);
    let at = |label: &str, c: &Value, pitch: f64, yaw: f64| json!({"label": label, "position": c["position"],
        "look_at": turned(c, up, pitch, yaw), "vfov_deg": c["vfov_deg"]});
    let mut out = Vec::new();
    if let Some(vf) = views_file {
        let views: Vec<Value> = serde_json::from_slice(&fs::read(vf).with_context(|| format!("reading {}", vf.display()))?)?;
        for v in views {
            let label = v["label"].as_str().with_context(|| format!("{}: a view without a label", vf.display()))?;
            if !v["position"].is_null() && !v["look_at"].is_null() {
                let vfov = if v["vfov_deg"].is_null() { path.first().map_or(json!(50), |c| c["vfov_deg"].clone()) } else { v["vfov_deg"].clone() };
                out.push(json!({"label": label, "position": v["position"], "look_at": v["look_at"], "vfov_deg": vfov}));
                continue;
            }
            ensure!(!path.is_empty(), "view {label:?} names a capture pose but scene.json has no capture_cameras");
            let k = if let Some(k) = v["capture"].as_u64() {
                ensure!((k as usize) < path.len(), "view {label:?}: capture {k} out of range (0..{})", path.len() - 1);
                k as usize
            } else if let Some(img) = v["image"].as_str() {
                path.iter().position(|c| c["image"].as_str() == Some(img))
                    .with_context(|| format!("view {label:?}: no capture camera with image {img:?}"))?
            } else if let Some(f) = v["frame"].as_f64() {
                (0..path.len()).min_by(|&a, &b| {
                    let d = |i: usize| (frame_index(path[i]["image"].as_str()).map_or(1e9, |x| x as f64) - f).abs();
                    d(a).total_cmp(&d(b))
                }).unwrap()
            } else {
                bail!("view {label:?}: needs capture, image, frame or position+look_at");
            };
            out.push(at(label, &path[k], v["pitch"].as_f64().unwrap_or(0.), v["yaw"].as_f64().unwrap_or(0.)));
        }
        ensure!(!out.is_empty(), "{}: no views", vf.display());
        return Ok(out);
    }
    let start = &scene["start_camera"];
    if start.is_object() {
        let mut s = start.clone();
        s["label"] = json!("Start view");
        out.push(s);
    }
    let radius = scene["radius"].as_f64().unwrap_or(2.);
    for i in spread_poses(path, n_default.max(2), v3(&start["position"]), radius) {
        let c = &path[i];
        let label = match c["image"].as_str() { Some(img) => format!("Capture {} ({img})", i + 1), None => format!("Capture {}", i + 1) };
        out.push(at(&label, c, 0., 0.));
    }
    if out.is_empty() {
        let f = v3(&scene["focus"]).unwrap_or([0.; 3]);
        out.push(json!({"label": "Start view", "position": [f[0], f[1], f[2] + 2. * radius], "look_at": f, "vfov_deg": 50}));
    }
    Ok(out)
}

/// JSON for an inline <script>: no literal '<' (only ever inside strings, as \u003c), so metadata such as a
/// title cannot close the element (`</script>`) or open a comment (`<!--`).
pub(crate) fn script_json(v: &Value) -> Result<String> {
    Ok(serde_json::to_string(v)?.replace('<', "\\u003c"))
}

/// WALK floor for the page in the frame of its up vector: (floorHeight, floorExtent), JSON null when unknown (the
/// page then finds the floor below the eye). A levelled collision grid (`collide.up`) carries its own floor height
/// along that up; otherwise scene.json's detected floor (y, along a +Y scene up) and its x/z extent.
fn walk_floor(scene: &Value, collide: &Value) -> (Value, Value) {
    let floor = &scene["floor"];
    let detected = floor["detected"].as_bool() == Some(true);
    let extent = if detected && floor["extent"]["min"].is_array() && floor["extent"]["max"].is_array() { floor["extent"].clone() } else { Value::Null };
    if collide.get("up").is_some_and(Value::is_array) {
        let h = collide.get("floorHeight").filter(|h| h.is_number()).cloned();
        return match h { Some(h) => (h, extent), None => (Value::Null, Value::Null) };
    }
    let up = v3(&scene["up"]).map(unit).unwrap_or([0.0, 1.0, 0.0]);
    match floor["y"].as_f64() {
        Some(y) if detected && up[1] > 1.0 - 1e-6 => (json!(y), extent),
        _ => (Value::Null, Value::Null),
    }
}

pub(crate) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#x27;")
}

pub(crate) fn default_title(run: &Path) -> String {
    fs::read(run.join("run.json")).ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|r| Path::new(r["source"].as_str()?).file_stem()?.to_str().map(str::to_owned))
        .unwrap_or_else(|| "Splat viewer".into())
}

/// The web package and run dir `source` (a `vauss run` dir or a web package dir) names, and its scene.json.
pub(crate) fn resolve_package(source: &Path) -> Result<(PathBuf, PathBuf, Value)> {
    let src = fs::canonicalize(source).with_context(|| format!("{} not found", source.display()))?;
    let pkg = if src.join("exports/web/scene.json").is_file() { src.join("exports/web") } else { src.clone() };
    let run = if pkg.parent().is_some_and(|p| p.file_name().is_some_and(|n| n == "exports")) {
        pkg.parent().and_then(Path::parent).unwrap().to_owned()
    } else {
        src.clone()
    };
    let sj = pkg.join("scene.json");
    ensure!(sj.is_file(), "{} not found", sj.display());
    let scene: Value = serde_json::from_slice(&fs::read(&sj)?)?;
    Ok((pkg, run, scene))
}

/// Write the viewer folder; returns a summary (out, index, models, views, bytes).
pub fn run(a: &ExportViewerArgs) -> Result<Value> {
    let (pkg, run, scene) = resolve_package(&a.source)?;
    if !scene["splat_neural"].is_null() {
        eprintln!("export-viewer: warning: neural appearance is ignored");
    }
    let ov = &scene["splat_overflow"];
    if ov["overflow_count"].as_u64().is_some_and(|n| n > 0) || !ov["dc"].is_null() || !ov["positions"].is_null() {
        eprintln!("export-viewer: warning: splat_overflow sidecars are ignored");
    }
    let main_spz = pkg.join(scene["splat"].as_str().unwrap_or("scene.spz"));
    let lod = pkg.join(scene["splat_lod"]["file"].as_str().unwrap_or("scene.rad"));
    let first = pkg.join(scene["splat_first"]["file"].as_str().unwrap_or("scene.first.spz"));
    ensure!(main_spz.is_file(), "{} not found", main_spz.display());
    let out = a.out.clone().unwrap_or_else(|| pkg.parent().unwrap_or(&pkg).join("viewer"));
    let full = resolve_full(a.full.as_deref(), &run)?;
    let title = a.title.clone().unwrap_or_else(|| default_title(&run));

    // Build next to the destination, then swap it in: an open viewer never sees a half-written folder.
    let name = out.file_name().context("--out has no folder name")?.to_string_lossy().into_owned();
    let parent = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if tmp.exists() {
        fs::remove_dir_all(&tmp)?;
    }
    let data = tmp.join("data");
    fs::create_dir_all(&data)?;

    let ih = spz_header(&main_spz)?;
    let mut info = serde_json::Map::new();
    info.insert("interactive".into(), ih.clone());
    let mut variants: Vec<(&str, &str, PathBuf, String, &str)> = Vec::new();
    // Millions of splats to two significant digits ("1M", "4.6M"), as the page's status line shows them.
    let mio = |h: &Value| {
        let m = h["splats"].as_f64().unwrap_or(0.) / 1e6;
        let digits = if m >= 10. { 0 } else if m >= 1. { 1 } else { 2 };
        let s = format!("{m:.digits$}");
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_owned() } else { s }
    };
    if first.is_file() {
        variants.push(("first", "First paint", first.clone(), "first-paint sample".into(), "spz"));
    }
    if !(a.no_interactive_spz && lod.is_file()) {
        variants.push(("interactive", "Interactive", main_spz.clone(), format!("{}M splats", mio(&ih)), "spz"));
    }
    if lod.is_file() {
        info.insert("lod".into(), rad_header(&lod)?);
        variants.push(("lod", "Interactive (LoD)", lod.clone(), format!("{}M splats, LoD", mio(&ih)), "rad"));
    }
    if let Some(f) = full.as_ref().filter(|f| fs::canonicalize(f).ok() != fs::canonicalize(&main_spz).ok()) {
        let fh = spz_header(f)?;
        variants.push(("full", "Full quality", f.clone(), format!("{}M splats", mio(&fh)), "spz"));
        info.insert("full".into(), fh);
    }

    let part_bytes = ((a.part_mb * 1e6) as usize).max(1);
    let mut meta = Vec::new();
    for (key, label, path, note, kind) in &variants {
        let raw = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let mut m = json!({"key": key, "label": label, "note": note, "kind": kind, "bytes": raw.len()});
        if *key == "first" {  // one small script: it has to be up before anything else
            m["src"] = json!(format!("data/{key}.js"));
            m["js_bytes"] = json!(write_data_js(&raw, &data.join(format!("{key}.js")), key)?);
        } else {
            let (parts, js) = write_parts(&raw, &data, key, part_bytes)?;
            m["parts"] = json!(parts);
            m["js_bytes"] = json!(js);
        }
        meta.push(m);
    }
    fs::write(data.join("spark.js"), format!("window.__SPARK_SRC={};\n", serde_json::to_string(&spark_source()?)?))?;
    // COLLIDE: best effort; any failure ships the viewer without collision.
    let collide = if a.no_collide { Value::Null } else {
        crate::viewer_collide::write(&pkg, &scene, &data, part_bytes).unwrap_or_else(|e| {
            eprintln!("export-viewer: no camera collision: {e:#}");
            for f in fs::read_dir(&data).into_iter().flatten().flatten() {
                if f.file_name().to_string_lossy().starts_with("collide-") { let _ = fs::remove_file(f.path()); }
            }
            Value::Null
        })
    };

    let capture = capture_path(&scene);
    let capture_metadata: &[Value] = if a.no_guard { &[] } else { &capture };
    let cams = named_views(&scene, &capture, a.views.as_deref(), a.cameras)?;
    let labels: std::collections::HashSet<_> = cams.iter().map(|c| c["label"].as_str().unwrap_or("")).collect();
    ensure!(labels.len() == cams.len(), "named view labels must be unique");
    let capture_geometry = scene["source"]["data"].as_str().and_then(|d| colmap_capture(Path::new(d)));
    let aspect = a.capture_aspect.or_else(|| capture_geometry.map(|c| c.0));
    let mip = scene["render_mode"].as_str().map_or(scene["stats"]["antialiased"].as_bool() == Some(true), |m| m == "mip");
    // WALK: the room's levelled vertical (the game's floor / camera levelling) when the collision grid has one,
    // so walking and looking keep the horizon level; else scene.json's up.
    let up = collide.get("up").filter(|u| u.is_array()).cloned()
        .unwrap_or_else(|| scene.get("up").cloned().unwrap_or(json!([0, 1, 0])));
    let (floor_height, floor_extent) = walk_floor(&scene, &collide);
    let cfg = json!({"title": title, "up": up, "floorHeight": floor_height, "floorExtent": floor_extent,
        "focus": scene.get("focus").cloned().unwrap_or(json!([0, 0, 0])), "radius": scene["radius"].as_f64().unwrap_or(2.),
        "mip": mip, "budget": a.budget, "variants": meta, "cameras": cams,
        // Capture metadata for inspection / scripted comparisons, never a navigation boundary.
        "path": capture_metadata.iter().enumerate().map(|(i, c)| json!({"n": c["image"].as_str().map_or(i.to_string(), str::to_owned),
            "p": c["position"], "l": c["look_at"]})).collect::<Vec<_>>(),
        "capAspect": aspect.map(|x| (x * 1e5).round() / 1e5), "capHeight": capture_geometry.map(|c| c.1),
        "capFocal": capture_geometry.map(|c| c.2),
        "metersPerUnit": scene["meters_per_unit"].as_f64().unwrap_or(1.),
        "collide": collide,
        "info": info});
    let t = html_escape(&title);
    let html = PAGE.replace("__TITLE__", &t).replace("__CFG__", &script_json(&cfg)?)
        .replace("__THREE__", THREE).replace("__THREE_ADDONS__", THREE_ADDONS);
    fs::write(tmp.join("index.html"), html)?;
    // build.json is served with the page: the package's folder name, never its local path.
    let build = json!({"generator": format!("vauss export-viewer {}", env!("CARGO_PKG_VERSION")),
        "source": pkg.file_name().map(|n| n.to_string_lossy().into_owned()),
        "full": full, "spark": format!("{SPARK_VERSION} (patched)"), "guard_poses": capture_metadata.len(),
        "capture_aspect": aspect, "capture_height": capture_geometry.map(|c| c.1), "capture_focal_px": capture_geometry.map(|c| c.2),
        "models": info, "views": cams.iter().map(|c| c["label"].clone()).collect::<Vec<_>>(), "collide": collide});
    fs::write(tmp.join("build.json"), serde_json::to_string_pretty(&build)?)?;

    if out.exists() {
        fs::remove_dir_all(&out).with_context(|| format!("replacing {}", out.display()))?;
    }
    fs::rename(&tmp, &out)?;
    let bytes: u64 = fs::read_dir(out.join("data"))?.flatten().filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
    Ok(json!({"out": out, "index": out.join("index.html"), "source": pkg, "title": title, "models": build["models"],
        "views": build["views"], "guard_poses": capture_metadata.len(), "capture_aspect": aspect, "collide": !build["collide"].is_null(), "bytes": bytes}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_cfg_cannot_close_its_script() {
        let title = "</script><script>globalThis.pr8Injected=true</script><!--";
        let cfg = json!({"title": title, "cameras": [{"label": "</SCRIPT >"}]});
        let s = script_json(&cfg).unwrap();
        assert!(!s.contains('<'), "{s}");
        assert_eq!(serde_json::from_str::<Value>(&s).unwrap(), cfg);
        let page = PAGE.replace("__TITLE__", &html_escape(title)).replace("__CFG__", &s);
        // The template's own script elements only: a title cannot add one.
        assert_eq!(page.matches("<script").count(), PAGE.matches("<script").count());
        assert!(!page.contains("pr8Injected=true</script>"));
    }

    #[test]
    fn walk_floor_follows_the_chosen_up() {
        let extent = json!({"min": [-2.5, -2.7], "max": [2.5, 2.7]});
        let scene = json!({"up": [0, 1, 0], "floor": {"y": 0.0, "detected": true, "extent": extent}});
        assert_eq!(walk_floor(&scene, &Value::Null), (json!(0.0), extent.clone()));
        // A levelled grid: its own floor height along its up, or none (never scene y along another up).
        let levelled = json!({"up": [0.08, 0.996, 0.0], "floorHeight": 0.031});
        assert_eq!(walk_floor(&scene, &levelled), (json!(0.031), extent.clone()));
        assert_eq!(walk_floor(&scene, &json!({"up": [0.08, 0.996, 0.0], "floorHeight": null})), (Value::Null, Value::Null));
        // Undetected floors and non-+Y scene frames export nothing: the page searches below the eye.
        let undetected = json!({"up": [0, 1, 0], "floor": {"y": 0.0, "detected": false}});
        assert_eq!(walk_floor(&undetected, &Value::Null), (Value::Null, Value::Null));
        let z_up = json!({"up": [0, 0, 1], "floor": {"y": 0.0, "detected": true, "extent": extent}});
        assert_eq!(walk_floor(&z_up, &Value::Null), (Value::Null, Value::Null));
        assert_eq!(walk_floor(&json!({}), &Value::Null), (Value::Null, Value::Null));
    }
}
