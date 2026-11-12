//! LPIPS (VGG and AlexNet) through ONNX Runtime. Models are exported once from the torchmetrics /
//! richzhang LPIPS v0.1 weights; at runtime only the .onnx files are needed (no Python).
//!
//! Model IO (crates/splat-eval/tools/export_lpips_onnx.py): inputs `in0`, `in1` float32 [1,3,H,W]
//! already in [-1,1], output `distance` float32 [1,1,1,1]. The [0,1] -> [-1,1] step that
//! torchmetrics does for `normalize=True` (`2x - 1`) happens here in Rust; the ImageNet
//! ScalingLayer is inside the graph.
//!
//! ONNX Runtime is loaded at runtime (`ort` feature `load-dynamic`), so the binary stays a single
//! file; see [`init_runtime`] for where the shared library is looked up.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use ort::session::Session;
use ort::value::TensorRef;

use crate::images::Rgb;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Net {
    Vgg,
    Alex,
}

impl Net {
    pub fn file_name(self) -> &'static str {
        match self {
            Net::Vgg => "lpips_vgg.onnx",
            Net::Alex => "lpips_alex.onnx",
        }
    }

    /// Size of the pinned export (models.json); a cheap guard against truncated or foreign files.
    fn expected_bytes(self) -> u64 {
        match self {
            Net::Vgg => 58_889_359,
            Net::Alex => 9_900_294,
        }
    }
}

#[cfg(target_os = "windows")]
const ORT_LIB: &str = "onnxruntime.dll";
#[cfg(target_os = "macos")]
const ORT_LIB: &str = "libonnxruntime.dylib";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const ORT_LIB: &str = "libonnxruntime.so";

/// Platform data dir (`dirs::data_dir` semantics) without the extra dependency.
fn data_dir() -> Option<PathBuf> {
    let env = |k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if cfg!(target_os = "windows") {
        env("APPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Application Support"))
    } else {
        env("XDG_DATA_HOME").or_else(|| env("HOME").map(|h| h.join(".local/share")))
    }
}

/// Path of the ONNX Runtime library that was loaded; `None` until [`init_runtime`] succeeds.
static RUNTIME: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Load the ONNX Runtime shared library (ONNX Runtime 1.28.x). Resolution: explicit path, else
/// $ORT_DYLIB_PATH, else `<data_dir>/splat/runtimes/onnxruntime/<libonnxruntime.so|.dylib|onnxruntime.dll>`.
/// Idempotent: after the first success further calls are no-ops (a process can load only one runtime).
pub fn init_runtime(dylib: Option<&Path>) -> Result<()> {
    let mut loaded = RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
    if loaded.is_some() {
        return Ok(());
    }
    let (path, source) = if let Some(p) = dylib {
        (p.to_path_buf(), "explicit path")
    } else if let Some(p) = std::env::var_os("ORT_DYLIB_PATH").filter(|v| !v.is_empty()) {
        (PathBuf::from(p), "$ORT_DYLIB_PATH")
    } else {
        let dir = data_dir().context(
            "ONNX Runtime not found: no platform data dir ($HOME/$XDG_DATA_HOME/%APPDATA% unset) and \
             $ORT_DYLIB_PATH unset; run `vauss doctor` to fetch it",
        )?;
        (dir.join("splat/runtimes/onnxruntime").join(ORT_LIB), "default runtime dir")
    };
    if !path.is_file() {
        bail!(
            "ONNX Runtime library not found at {} ({source}); run `vauss doctor` to fetch it, or set \
             $ORT_DYLIB_PATH to a {ORT_LIB} from ONNX Runtime 1.28",
            path.display()
        );
    }
    ort::init_from(&path)
        .map_err(|e| {
            anyhow::anyhow!(
                "loading ONNX Runtime from {} ({source}): {e}; run `vauss doctor` to install a compatible one",
                path.display()
            )
        })?
        .with_name("splat-eval")
        .commit();
    *loaded = Some(path);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CudaMode {
    Off,
    Auto,
    Required,
}

/// $SPLAT_EVAL_ORT_CUDA: unset = CUDA when the runtime supports it (onnxruntime-gpu), else CPU;
/// "1" = require CUDA (error if unavailable); "0" = CPU only.
fn cuda_mode() -> CudaMode {
    match std::env::var("SPLAT_EVAL_ORT_CUDA").as_deref() {
        Ok("0") => CudaMode::Off,
        Ok("1") => CudaMode::Required,
        _ => CudaMode::Auto,
    }
}

/// CPU fallback bound: LPIPS-VGG at full resolution took 255 s on 19 pegasus views using ~11
/// cores, so on CPU the images are box-downscaled to this edge.
pub const CPU_MAX_EDGE: usize = 512;

/// $SPLAT_EVAL_LPIPS_SCALE=full: score LPIPS at full resolution on the CPU too (slow; for absolute
/// numbers without a GPU, and for CPU-vs-CUDA equivalence checks). Unset = the bounded fallback.
fn cpu_full_scale() -> bool {
    std::env::var("SPLAT_EVAL_LPIPS_SCALE").is_ok_and(|v| v == "full")
}

/// ORT intra-op thread count: $SPLAT_EVAL_ORT_THREADS when set to a positive integer, else
/// `available_parallelism` (respects the sched affinity mask and cgroup CPU quota). Always passed
/// explicitly: with the default (0) ORT sizes the pool from the host's physical cores and pins each
/// thread with pthread_setaffinity_np, which fails inside a container cpuset (cloud runners) and
/// leaves ORT effectively single-threaded. An explicit count makes ORT skip affinity pinning.
pub fn ort_threads() -> usize {
    std::env::var("SPLAT_EVAL_ORT_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
}

/// Integer box downscale so the longer edge is <= max_edge.
fn box_downscale(img: &Rgb, max_edge: usize) -> Option<Rgb> {
    let f = img.width.max(img.height).div_ceil(max_edge);
    if f <= 1 {
        return None;
    }
    let (w, h) = (img.width / f, img.height / f);
    let mut data = vec![0.0f32; 3 * w * h];
    let norm = 1.0 / (f * f) as f32;
    for c in 0..3 {
        let src = &img.data[c * img.width * img.height..];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for dy in 0..f {
                    let row = (y * f + dy) * img.width + x * f;
                    acc += src[row..row + f].iter().sum::<f32>();
                }
                data[c * w * h + y * w + x] = acc * norm;
            }
        }
    }
    Some(Rgb { width: w, height: h, data })
}

/// Libraries the CUDA execution provider loads by soname at run time. When one is missing, ORT
/// aborts the whole process mid-inference ("Cannot load symbol cublasLtGetVersion", exit -6)
/// instead of failing the session, so Auto mode checks they resolve before choosing CUDA.
#[cfg(target_os = "linux")]
const CUDA_SONAMES: [&str; 4] = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"];

/// None when every CUDA library resolves from this process (and the provider loads next to the
/// runtime); else why not.
#[cfg(target_os = "linux")]
fn cuda_libs_missing() -> Option<String> {
    let open = |name: &str| {
        let Ok(c) = std::ffi::CString::new(name) else { return false };
        // SAFETY: dlopen with a valid C string; the handle is intentionally kept (libraries stay loaded).
        !unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) }.is_null()
    };
    let provider = RUNTIME
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|p| p.parent().map(|d| d.join("libonnxruntime_providers_cuda.so")));
    match provider {
        Some(p) if p.is_file() => {
            if !open(&p.to_string_lossy()) {
                return Some(format!("{} does not load", p.display()));
            }
        }
        _ => return Some("CPU-only ONNX Runtime (no libonnxruntime_providers_cuda.so next to it)".into()),
    }
    let missing: Vec<&str> = CUDA_SONAMES.into_iter().filter(|n| !open(n)).collect();
    (!missing.is_empty()).then(|| format!("{} not on the library path", missing.join(", ")))
}

#[cfg(not(target_os = "linux"))]
fn cuda_libs_missing() -> Option<String> {
    None
}

/// Loaded ONNX sessions for the requested nets.
pub struct Lpips {
    sessions: Vec<(Net, Session)>,
    /// Execution provider the sessions run on: "cuda" or "cpu".
    pub provider: &'static str,
    /// Why CUDA was not used (Auto mode), for the metrics note.
    cpu_reason: Option<String>,
    /// CPU sessions score at full resolution ($SPLAT_EVAL_LPIPS_SCALE=full) instead of <= CPU_MAX_EDGE.
    cpu_full: bool,
}

impl Lpips {
    /// Set when LPIPS runs in the bounded CPU fallback (scores are at reduced resolution).
    pub fn note(&self) -> Option<String> {
        (self.provider == "cpu").then(|| {
            let why = self.cpu_reason.as_deref().unwrap_or("no CUDA onnxruntime");
            if self.cpu_full {
                format!("cpu, full resolution (SPLAT_EVAL_LPIPS_SCALE=full), {} threads ({why})", ort_threads())
            } else {
                format!("cpu fallback: LPIPS on images downscaled to <= {CPU_MAX_EDGE} px, {} threads ({why})", ort_threads())
            }
        })
    }

    /// Resolution LPIPS is scored at: "full", or "cpu<=512" for the CPU fallback, whose absolute
    /// values are not comparable to full-resolution / published numbers (paired deltas are).
    pub fn scale(&self) -> String {
        if self.provider == "cpu" && !self.cpu_full { format!("cpu<={CPU_MAX_EDGE}") } else { "full".into() }
    }
}

impl Lpips {
    /// Models dir resolution: explicit dir, else $SPLAT_MODELS_DIR, else the platform data dir
    /// (`<data_dir>/splat/models`). Errors name the missing file.
    pub fn locate(models_dir: Option<&Path>) -> Result<PathBuf> {
        let (dir, source) = if let Some(d) = models_dir {
            (d.to_path_buf(), "explicit dir")
        } else if let Some(d) = std::env::var_os("SPLAT_MODELS_DIR").filter(|v| !v.is_empty()) {
            (PathBuf::from(d), "$SPLAT_MODELS_DIR")
        } else {
            let d = data_dir().context(
                "no models dir: pass one or set $SPLAT_MODELS_DIR (no platform data dir found)",
            )?;
            (d.join("splat/models"), "default models dir")
        };
        let missing: Vec<_> = [Net::Vgg, Net::Alex]
            .into_iter()
            .map(Net::file_name)
            .filter(|f| !dir.join(f).is_file())
            .collect();
        ensure!(
            missing.is_empty(),
            "LPIPS models missing in {} ({source}): {}; run `vauss doctor` to fetch them",
            dir.display(),
            missing.join(", ")
        );
        Ok(dir)
    }

    pub fn load(models_dir: &Path, nets: &[Net]) -> Result<Self> {
        init_runtime(None)?;
        let mut sessions = Vec::with_capacity(nets.len());
        let mut provider = "cpu";
        let cpu_reason = match cuda_mode() {
            CudaMode::Auto => cuda_libs_missing(),
            CudaMode::Off => Some("SPLAT_EVAL_ORT_CUDA=0".into()),
            CudaMode::Required => None,
        };
        for &net in nets {
            if sessions.iter().any(|(n, _)| *n == net) {
                continue;
            }
            let path = models_dir.join(net.file_name());
            let bytes = std::fs::metadata(&path)
                .with_context(|| format!("LPIPS model {}", path.display()))?
                .len();
            ensure!(
                bytes == net.expected_bytes(),
                "LPIPS model {} is {bytes} bytes, expected {} (re-fetch it)",
                path.display(),
                net.expected_bytes()
            );
            let load = |cuda: bool| -> ort::Result<Session> {
                // Explicit intra-op threads on both paths (the CUDA session still has a CPU pool),
                // so ORT never pins threads; see ort_threads.
                let b = Session::builder()?.with_intra_threads(ort_threads())?;
                let mut b = if cuda {
                    // TF32 off: full FP32 convolutions, so CUDA scores match the CPU path (<= 1e-4 per view).
                    b.with_execution_providers([ort::ep::CUDA::default().with_tf32(false).build().error_on_failure()])?
                } else {
                    b
                };
                b.commit_from_file(&path)
            };
            let fail = |e: ort::Error| anyhow::anyhow!("loading LPIPS model {}: {e}", path.display());
            // CUDA needs onnxruntime-gpu plus the CUDA/cuDNN libraries; Auto falls back to CPU without them.
            let session = match cuda_mode() {
                CudaMode::Off => load(false).map_err(fail)?,
                CudaMode::Required => {
                    provider = "cuda";
                    load(true).map_err(fail)?
                }
                CudaMode::Auto if cpu_reason.is_some() => load(false).map_err(fail)?,
                CudaMode::Auto => match load(true) {
                    Ok(s) => {
                        provider = "cuda";
                        s
                    }
                    Err(_) => load(false).map_err(fail)?,
                },
            };
            sessions.push((net, session));
        }
        let cpu_reason = if provider == "cpu" { cpu_reason.or_else(|| Some("CUDA provider failed to load".into())) } else { None };
        Ok(Self { sessions, provider, cpu_reason, cpu_full: cpu_full_scale() })
    }

    /// Distance between two images of the same size in [0,1] (scaled to [-1,1] internally, which is
    /// torchmetrics `normalize=True`). Lower is better.
    pub fn distance(&mut self, net: Net, a: &Rgb, b: &Rgb) -> Result<f64> {
        self.distance_scaled(net, a, b, true)
    }

    /// The 3DGS-code variant (graphdeco `lpipsPyTorch`, = torchmetrics `normalize=False`): the same
    /// network fed [0,1] images unscaled. Reads ~0.04 lower than [`Self::distance`] on Mip-NeRF 360;
    /// it is what most 3DGS-family papers report as "LPIPS", so only use it to compare with them.
    pub fn distance_3dgs(&mut self, net: Net, a: &Rgb, b: &Rgb) -> Result<f64> {
        self.distance_scaled(net, a, b, false)
    }

    fn distance_scaled(&mut self, net: Net, a: &Rgb, b: &Rgb, to_pm1: bool) -> Result<f64> {
        ensure!(
            (a.width, a.height) == (b.width, b.height),
            "LPIPS: size mismatch {}x{} vs {}x{}",
            a.width,
            a.height,
            b.width,
            b.height
        );
        let n = 3 * a.width * a.height;
        ensure!(a.data.len() == n && b.data.len() == n, "LPIPS: image buffers are not 3xHxW");
        let small = if self.provider == "cpu" && !self.cpu_full { box_downscale(a, CPU_MAX_EDGE).zip(box_downscale(b, CPU_MAX_EDGE)) } else { None };
        let (a, b) = match &small {
            Some((sa, sb)) => (sa, sb),
            None => (a, b),
        };
        let session = self
            .sessions
            .iter_mut()
            .find_map(|(n, s)| (*n == net).then_some(s))
            .with_context(|| format!("LPIPS {net:?} model not loaded"))?;

        let scale = |img: &Rgb| img.data.iter().map(|&v| if to_pm1 { 2.0 * v - 1.0 } else { v }).collect::<Vec<f32>>();
        let (a, b, shape) = (scale(a), scale(b), [1, 3, a.height as i64, a.width as i64]);
        let outputs = session.run(ort::inputs![
            "in0" => TensorRef::from_array_view((shape, a.as_slice()))?,
            "in1" => TensorRef::from_array_view((shape, b.as_slice()))?,
        ])?;
        let (_, d) = outputs["distance"].try_extract_tensor::<f32>()?;
        ensure!(d.len() == 1, "LPIPS: expected a scalar distance, got {} values", d.len());
        Ok(f64::from(d[0]))
    }
}
