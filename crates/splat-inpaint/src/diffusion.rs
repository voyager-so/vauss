//! Stable-Diffusion-1.5 inpainting (9-channel UNet) through ONNX Runtime, written from the
//! standard latent-diffusion description: CLIP BPE tokenizer -> text encoder (uncond + cond for
//! classifier-free guidance) -> VAE-encode the masked image -> DDIM loop over the UNet with
//! input concat(latents, mask, masked-image latents) -> VAE decode. Fixed 512x512, fp16 models
//! (the pinned diffusers ONNX export, see splat-fetch's `inpaint-sd15-*` assets).
//!
//! Execution provider: CUDA when the runtime and libraries support it, else CPU with a loud log
//! line ($SPLAT_INPAINT_ORT_CUDA: unset = auto, "1" = require CUDA, "0" = CPU only). The UNet
//! takes ~minutes per image on CPU (if the CPU provider has the fp16 kernels at all).
//!
//! Determinism: the initial latent noise comes from `seed` (SplitMix64 + Box-Muller). The VAE
//! encoder export samples its posterior with an unseeded RandomNormalLike, so the masked-image
//! latents carry a tiny run-to-run jitter (std of the posterior, ~1e-3 of the latent scale).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, ensure};
use half::f16;
use ort::session::Session;
use ort::value::{TensorElementType, TensorRef};

use crate::img::{Mask, RgbF};
use crate::models::{Inpaint2d, locate};

pub const TEXT_ENCODER: (&str, &str) = ("inpaint-sd15-text-encoder", "sd15_inpaint_text_encoder_fp16.onnx");
pub const UNET: (&str, &str) = ("inpaint-sd15-unet", "sd15_inpaint_unet_fp16.onnx");
pub const VAE_ENCODER: (&str, &str) = ("inpaint-sd15-vae-encoder", "sd15_inpaint_vae_encoder_fp16.onnx");
pub const VAE_DECODER: (&str, &str) = ("inpaint-sd15-vae-decoder", "sd15_inpaint_vae_decoder_fp16.onnx");
pub const VOCAB: (&str, &str) = ("inpaint-sd15-clip-vocab", "sd15_clip_vocab.json");
pub const MERGES: (&str, &str) = ("inpaint-sd15-clip-merges", "sd15_clip_merges.txt");
pub const SCHEDULER: (&str, &str) = ("inpaint-sd15-scheduler", "sd15_inpaint_scheduler_config.json");

pub const DEFAULT_PROMPT: &str = "empty floor, clean background, photorealistic";
pub const DEFAULT_NEGATIVE: &str = "object, furniture, person, text, blurry, artifacts";

/// SD-1.5 VAE latent scaling factor.
const VAE_SCALE: f32 = 0.18215;
const SIZE: usize = 512;
const LAT: usize = SIZE / 8;

// ---------------------------------------------------------------- CLIP BPE tokenizer

const BOS: i64 = 49406;
const EOS: i64 = 49407;
const CTX: usize = 77;

/// CLIP's byte-level BPE (lower-cased, whitespace-collapsed text; `</w>` marks word ends).
pub struct ClipTokenizer {
    vocab: HashMap<String, i64>,
    ranks: HashMap<(String, String), usize>,
    byte_char: [char; 256],
    pat: regex::Regex,
}

/// GPT-2/CLIP byte -> printable unicode char table: printable Latin-1 bytes map to themselves,
/// the rest to 256.. in byte order.
fn bytes_to_unicode() -> [char; 256] {
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut table = ['\0'; 256];
    let mut extra = 0;
    for b in 0..256u32 {
        let c = if printable(b) {
            b
        } else {
            extra += 1;
            255 + extra
        };
        table[b as usize] = char::from_u32(c).expect("valid char");
    }
    table
}

impl ClipTokenizer {
    pub fn load(vocab: &Path, merges: &Path) -> anyhow::Result<Self> {
        let vocab: HashMap<String, i64> = serde_json::from_slice(&std::fs::read(vocab).with_context(|| format!("reading {}", vocab.display()))?)
            .with_context(|| format!("parsing {}", vocab.display()))?;
        ensure!(vocab.get("<|startoftext|>") == Some(&BOS) && vocab.get("<|endoftext|>") == Some(&EOS), "CLIP vocab: unexpected special-token ids");
        let text = std::fs::read_to_string(merges).with_context(|| format!("reading {}", merges.display()))?;
        // Skip the "#version" header; CLIP uses the first 49152 - 256 - 2 merges.
        let ranks: HashMap<(String, String), usize> = text
            .lines()
            .skip(1)
            .take(49152 - 256 - 2)
            .filter_map(|l| l.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
            .enumerate()
            .map(|(i, p)| (p, i))
            .collect();
        ensure!(ranks.len() > 40_000, "CLIP merges: only {} merges in {}", ranks.len(), merges.display());
        let pat = regex::Regex::new(r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")?;
        Ok(Self { vocab, ranks, byte_char: bytes_to_unicode(), pat })
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(String::from).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        while word.len() > 1 {
            let best = word
                .windows(2)
                .filter_map(|w| self.ranks.get(&(w[0].clone(), w[1].clone())).map(|&r| (r, w[0].clone(), w[1].clone())))
                .min_by_key(|(r, _, _)| *r);
            let Some((_, a, b)) = best else { break };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(std::mem::take(&mut word[i]));
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }

    /// 77 ids: BOS, up to 75 tokens, EOS, EOS padding.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![BOS];
        for m in self.pat.find_iter(&clean) {
            let mapped: String = m.as_str().bytes().map(|b| self.byte_char[b as usize]).collect();
            ids.extend(self.bpe(&mapped).iter().filter_map(|t| self.vocab.get(t).copied()));
        }
        ids.truncate(CTX - 1);
        ids.resize(CTX, EOS);
        ids
    }
}

// ---------------------------------------------------------------- RNG + scheduler

/// SplitMix64 with Box-Muller normals (seeded, platform-independent).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n + 1);
        while v.len() < n {
            let (r, th) = ((-2.0 * self.uniform().ln()).sqrt(), std::f64::consts::TAU * self.uniform());
            v.push((r * th.cos()) as f32);
            v.push((r * th.sin()) as f32);
        }
        v.truncate(n);
        v
    }
}

/// The subset of diffusers' scheduler_config.json the DDIM sampler uses.
#[derive(Debug, serde::Deserialize)]
struct SchedulerConfig {
    #[serde(default = "d_beta_start")]
    beta_start: f64,
    #[serde(default = "d_beta_end")]
    beta_end: f64,
    #[serde(default = "d_beta_schedule")]
    beta_schedule: String,
    #[serde(default = "d_train_steps")]
    num_train_timesteps: usize,
    #[serde(default = "d_steps_offset")]
    steps_offset: usize,
    #[serde(default)]
    set_alpha_to_one: bool,
    #[serde(default = "d_prediction")]
    prediction_type: String,
    #[serde(default = "d_spacing")]
    timestep_spacing: String,
}
fn d_beta_start() -> f64 {
    0.00085
}
fn d_beta_end() -> f64 {
    0.012
}
fn d_beta_schedule() -> String {
    "scaled_linear".into()
}
fn d_train_steps() -> usize {
    1000
}
fn d_steps_offset() -> usize {
    1
}
fn d_prediction() -> String {
    "epsilon".into()
}
fn d_spacing() -> String {
    "leading".into()
}

/// Deterministic DDIM (eta = 0) over `steps` of the train schedule, "leading" spacing.
struct Ddim {
    alphas_cumprod: Vec<f64>,
    final_alpha: f64,
    timesteps: Vec<usize>,
    ratio: usize,
}

impl Ddim {
    fn new(cfg: &SchedulerConfig, steps: usize) -> anyhow::Result<Self> {
        ensure!(cfg.beta_schedule == "scaled_linear", "scheduler: beta_schedule {} not supported (scaled_linear only)", cfg.beta_schedule);
        ensure!(cfg.prediction_type == "epsilon", "scheduler: prediction_type {} not supported (epsilon only)", cfg.prediction_type);
        ensure!(cfg.timestep_spacing == "leading", "scheduler: timestep_spacing {} not supported (leading only)", cfg.timestep_spacing);
        let t = cfg.num_train_timesteps;
        ensure!(t >= 2 && (1..=t).contains(&steps), "diffusion_steps must be 1..={t}");
        let (s0, s1) = (cfg.beta_start.sqrt(), cfg.beta_end.sqrt());
        let mut ac = Vec::with_capacity(t);
        let mut acc = 1.0f64;
        for i in 0..t {
            let beta = (s0 + (s1 - s0) * i as f64 / (t - 1) as f64).powi(2);
            acc *= 1.0 - beta;
            ac.push(acc);
        }
        let ratio = t / steps;
        let timesteps: Vec<usize> = (0..steps).rev().map(|i| (i * ratio + cfg.steps_offset).min(t - 1)).collect();
        let final_alpha = if cfg.set_alpha_to_one { 1.0 } else { ac[0] };
        Ok(Self { alphas_cumprod: ac, final_alpha, timesteps, ratio })
    }

    /// x_{t-1} from x_t and the predicted noise (in place).
    fn step(&self, t: usize, eps: &[f32], x: &mut [f32]) {
        let a_t = self.alphas_cumprod[t];
        let a_prev = t.checked_sub(self.ratio).map_or(self.final_alpha, |p| self.alphas_cumprod[p]);
        let (sa, sb) = (a_t.sqrt(), (1.0 - a_t).sqrt());
        let (pa, pb) = (a_prev.sqrt(), (1.0 - a_prev).sqrt());
        for (xi, &e) in x.iter_mut().zip(eps) {
            let x0 = (*xi as f64 - sb * e as f64) / sa;
            *xi = (pa * x0 + pb * e as f64) as f32;
        }
    }
}

// ---------------------------------------------------------------- ONNX sessions on CUDA / CPU

#[derive(Clone, Copy, PartialEq, Eq)]
enum CudaMode {
    Off,
    Auto,
    Required,
}

fn cuda_mode() -> CudaMode {
    match std::env::var("SPLAT_INPAINT_ORT_CUDA").as_deref() {
        Ok("0") => CudaMode::Off,
        Ok("1") => CudaMode::Required,
        _ => CudaMode::Auto,
    }
}

/// Same check as splat-eval's LPIPS: ORT aborts the process mid-inference when a CUDA library
/// the provider needs is missing, so Auto mode checks they resolve before choosing CUDA. The
/// loaded runtime's directory is read from /proc/self/maps (splat-eval keeps its path private).
#[cfg(target_os = "linux")]
fn cuda_libs_missing() -> Option<String> {
    const SONAMES: [&str; 4] = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"];
    let open = |name: &str| {
        let Ok(c) = std::ffi::CString::new(name) else { return false };
        // SAFETY: dlopen with a valid C string; the handle is intentionally kept (libraries stay loaded).
        !unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) }.is_null()
    };
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let runtime = maps
        .lines()
        .filter_map(|l| l.split_whitespace().nth(5))
        .find(|p| Path::new(p).file_name().is_some_and(|f| f.to_string_lossy().starts_with("libonnxruntime.so")))
        .map(PathBuf::from);
    let provider = runtime.as_deref().and_then(Path::parent).map(|d| d.join("libonnxruntime_providers_cuda.so"));
    match provider {
        Some(p) if p.is_file() => {
            if !open(&p.to_string_lossy()) {
                return Some(format!("{} does not load", p.display()));
            }
        }
        Some(p) => return Some(format!("CPU-only ONNX Runtime (no {})", p.display())),
        None => return Some("loaded libonnxruntime.so not found in /proc/self/maps".into()),
    }
    let missing: Vec<&str> = SONAMES.into_iter().filter(|n| !open(n)).collect();
    (!missing.is_empty()).then(|| format!("{} not on the library path", missing.join(", ")))
}

#[cfg(not(target_os = "linux"))]
fn cuda_libs_missing() -> Option<String> {
    None
}

/// Loads every session on one provider: CUDA if possible, else CPU (logged loudly).
struct Loader {
    cuda: bool,
    cpu_reason: Option<String>,
}

impl Loader {
    fn new() -> anyhow::Result<Self> {
        splat_eval::lpips::init_runtime(None)?;
        let (cuda, cpu_reason) = match cuda_mode() {
            CudaMode::Off => (false, Some("SPLAT_INPAINT_ORT_CUDA=0".to_string())),
            CudaMode::Required => (true, None),
            CudaMode::Auto => match cuda_libs_missing() {
                Some(why) => (false, Some(why)),
                None => (true, None),
            },
        };
        Ok(Self { cuda, cpu_reason })
    }

    fn load(&mut self, path: &Path) -> anyhow::Result<Session> {
        let build = |cuda: bool| -> ort::Result<Session> {
            // Explicit intra-op threads (ORT never pins threads; see splat_eval::lpips::ort_threads).
            let b = Session::builder()?.with_intra_threads(splat_eval::lpips::ort_threads())?;
            let mut b = if cuda { b.with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])? } else { b };
            b.commit_from_file(path)
        };
        let fail = |e: ort::Error| anyhow::anyhow!("loading {}: {e}", path.display());
        if self.cuda {
            match build(true) {
                Ok(s) => return Ok(s),
                Err(e) if cuda_mode() == CudaMode::Required => return Err(fail(e)),
                Err(e) => {
                    self.cuda = false;
                    self.cpu_reason = Some(format!("CUDA provider failed: {e}"));
                }
            }
        }
        eprintln!(
            "[inpaint] WARNING: SD-1.5 inpainting on the ORT CPU provider ({}); expect minutes per image, and fp16 kernels may be missing",
            self.cpu_reason.as_deref().unwrap_or("?")
        );
        build(false).map_err(fail)
    }

    fn provider(&self) -> String {
        if self.cuda { "cuda".into() } else { format!("cpu ({})", self.cpu_reason.as_deref().unwrap_or("?")) }
    }
}

fn input_name(s: &Session, want: &str, model: &str) -> anyhow::Result<String> {
    let names: Vec<&str> = s.inputs().iter().map(|i| i.name()).collect();
    ensure!(names.contains(&want), "{model}: no input {want:?} (inputs {names:?})");
    Ok(want.to_string())
}

fn input_type(s: &Session, name: &str) -> Option<TensorElementType> {
    s.inputs().iter().find(|i| i.name() == name).and_then(|i| i.dtype().tensor_type())
}

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

/// First output (or the named one) as f32.
fn output_f32(out: &ort::session::SessionOutputs, name: Option<&str>) -> anyhow::Result<(Vec<i64>, Vec<f32>)> {
    let v = match name.and_then(|n| out.get(n)) {
        Some(v) => v,
        None => &out[0],
    };
    if let Ok((shape, y)) = v.try_extract_tensor::<f16>() {
        return Ok((shape.iter().copied().collect(), y.iter().map(|x| x.to_f32()).collect()));
    }
    let (shape, y) = v.try_extract_tensor::<f32>()?;
    Ok((shape.iter().copied().collect(), y.to_vec()))
}

// ---------------------------------------------------------------- pipeline

pub struct Sd15Options {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
}

pub struct Sd15 {
    tokenizer: ClipTokenizer,
    text_encoder: Option<Session>,
    unet: Session,
    vae_encoder: Session,
    vae_decoder: Session,
    sched: SchedulerConfig,
    opts: Sd15Options,
    /// Text embeddings [uncond; cond] (2 x 77 x 768), computed on the first run.
    text: Option<Vec<f16>>,
    /// "cuda" or "cpu (<why>)".
    pub provider: String,
    /// Per-stage seconds (diffusion_load_s, diffusion_text_s, diffusion_vae_encode_s,
    /// diffusion_unet_s, diffusion_vae_decode_s).
    pub timings: BTreeMap<String, f64>,
    /// Model file paths by role.
    pub paths: BTreeMap<String, String>,
}

impl Sd15 {
    pub fn load(models_dir: Option<&Path>, opts: Sd15Options) -> anyhow::Result<Self> {
        let t0 = Instant::now();
        ensure!((1..=1000).contains(&opts.steps), "diffusion_steps must be 1..=1000");
        ensure!(opts.guidance.is_finite() && opts.guidance >= 0.0, "guidance must be finite and >= 0");
        let mut paths = BTreeMap::new();
        let mut get = |role: &str, a: (&str, &str)| -> anyhow::Result<PathBuf> {
            let p = locate(models_dir, a)?;
            paths.insert(format!("sd15_{role}"), p.display().to_string());
            Ok(p)
        };
        let (vocab, merges, sched) = (get("vocab", VOCAB)?, get("merges", MERGES)?, get("scheduler", SCHEDULER)?);
        let (te, unet, enc, dec) = (get("text_encoder", TEXT_ENCODER)?, get("unet", UNET)?, get("vae_encoder", VAE_ENCODER)?, get("vae_decoder", VAE_DECODER)?);
        let tokenizer = ClipTokenizer::load(&vocab, &merges)?;
        let sched: SchedulerConfig = serde_json::from_slice(&std::fs::read(&sched)?).with_context(|| format!("parsing {}", sched.display()))?;
        let mut loader = Loader::new()?;
        let text_encoder = loader.load(&te)?;
        let unet = loader.load(&unet)?;
        let vae_encoder = loader.load(&enc)?;
        let vae_decoder = loader.load(&dec)?;
        let provider = loader.provider();
        eprintln!("[inpaint] SD-1.5 inpainting on {provider}: {} steps, guidance {}, seed {}", opts.steps, opts.guidance, opts.seed);
        let mut timings = BTreeMap::new();
        timings.insert("diffusion_load_s".into(), t0.elapsed().as_secs_f64());
        Ok(Self { tokenizer, text_encoder: Some(text_encoder), unet, vae_encoder, vae_decoder, sched, opts, text: None, provider, timings, paths })
    }

    fn lap(&mut self, k: &str, t: Instant) {
        *self.timings.entry(k.into()).or_default() += t.elapsed().as_secs_f64();
    }

    /// [uncond; cond] last_hidden_state, 2 x 77 x 768 fp16.
    fn encode_text(&mut self) -> anyhow::Result<Vec<f16>> {
        let t = Instant::now();
        let te = self.text_encoder.as_mut().context("text encoder already released")?;
        let ids: Vec<i64> = [self.opts.negative_prompt.as_str(), self.opts.prompt.as_str()].iter().flat_map(|p| self.tokenizer.encode(p)).collect();
        let name = input_name(te, "input_ids", "text encoder")?;
        let shape = [2i64, CTX as i64];
        let out = match input_type(te, &name) {
            Some(TensorElementType::Int64) => te.run(ort::inputs![name.as_str() => TensorRef::from_array_view((shape, ids.as_slice()))?])?,
            _ => {
                let ids32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
                te.run(ort::inputs![name.as_str() => TensorRef::from_array_view((shape, ids32.as_slice()))?])?
            }
        };
        let (dims, h) = output_f32(&out, Some("last_hidden_state"))?;
        ensure!(dims == [2, CTX as i64, 768], "text encoder output shape {dims:?} (expected [2, 77, 768])");
        drop(out);
        // The text encoder is needed once per job; free its memory for the UNet.
        self.text_encoder = None;
        self.lap("diffusion_text_s", t);
        Ok(to_f16(&h))
    }

    /// VAE-encode an NCHW [-1,1] 512x512 image -> scaled latents (4 x 64 x 64).
    fn vae_encode(&mut self, planar: &[f32]) -> anyhow::Result<Vec<f32>> {
        let t = Instant::now();
        let name = input_name(&self.vae_encoder, "sample", "VAE encoder")?;
        let x = to_f16(planar);
        let s = SIZE as i64;
        let out = self.vae_encoder.run(ort::inputs![name.as_str() => TensorRef::from_array_view(([1i64, 3, s, s], x.as_slice()))?])?;
        let (dims, z) = output_f32(&out, Some("latent_sample"))?;
        ensure!(dims == [1, 4, LAT as i64, LAT as i64], "VAE encoder output shape {dims:?} (expected [1, 4, 64, 64])");
        drop(out);
        self.lap("diffusion_vae_encode_s", t);
        Ok(z.iter().map(|v| v * VAE_SCALE).collect())
    }

    fn vae_decode(&mut self, latents: &[f32]) -> anyhow::Result<Vec<f32>> {
        let t = Instant::now();
        let name = input_name(&self.vae_decoder, "latent_sample", "VAE decoder")?;
        let z: Vec<f16> = latents.iter().map(|&v| f16::from_f32(v / VAE_SCALE)).collect();
        let l = LAT as i64;
        let out = self.vae_decoder.run(ort::inputs![name.as_str() => TensorRef::from_array_view(([1i64, 4, l, l], z.as_slice()))?])?;
        let (dims, y) = output_f32(&out, Some("sample"))?;
        ensure!(dims == [1, 3, SIZE as i64, SIZE as i64], "VAE decoder output shape {dims:?} (expected [1, 3, 512, 512])");
        drop(out);
        self.lap("diffusion_vae_decode_s", t);
        Ok(y)
    }
}

impl Inpaint2d for Sd15 {
    fn inpaint(&mut self, img: &RgbF, hole: &Mask) -> anyhow::Result<RgbF> {
        let n = SIZE;
        ensure!(img.w == n && img.h == n && hole.w == n && hole.h == n, "SD-1.5 inpainting runs at {n}x{n}");
        let text = match self.text.take() {
            Some(t) => t,
            None => self.encode_text()?,
        };
        // Masked image in [-1, 1], hole pixels zeroed (diffusers' convention), NCHW.
        let mut planar = vec![0f32; 3 * n * n];
        for i in 0..n * n {
            if !hole.data[i] {
                for c in 0..3 {
                    planar[c * n * n + i] = img.data[3 * i + c] * 2.0 - 1.0;
                }
            }
        }
        let masked = self.vae_encode(&planar)?;
        // Latent mask: a cell is "hole" when any of its 8x8 pixels is, so no partly-hole cell
        // keeps the zeroed (grey) masked-image content that is pasted back inside the hole.
        let mut mask = vec![0f32; LAT * LAT];
        for y in 0..n {
            for x in 0..n {
                if hole.data[y * n + x] {
                    mask[(y / 8) * LAT + x / 8] = 1.0;
                }
            }
        }
        let ddim = Ddim::new(&self.sched, self.opts.steps as usize)?;
        let plane = LAT * LAT;
        let mut latents = Rng(self.opts.seed).normals(4 * plane);
        let (sample, timestep, ctx) = (
            input_name(&self.unet, "sample", "UNet")?,
            input_name(&self.unet, "timestep", "UNet")?,
            input_name(&self.unet, "encoder_hidden_states", "UNet")?,
        );
        let ts_f32 = input_type(&self.unet, &timestep) == Some(TensorElementType::Float32);
        let ts_i64 = input_type(&self.unet, &timestep) == Some(TensorElementType::Int64);
        let t_unet = Instant::now();
        let mut x = vec![f16::ZERO; 2 * 9 * plane];
        // Mask and masked-image latents are constant over the loop.
        for b in 0..2 {
            let o = b * 9 * plane;
            for (j, &m) in mask.iter().enumerate() {
                x[o + 4 * plane + j] = f16::from_f32(m);
            }
            for (j, &v) in masked.iter().enumerate() {
                x[o + 5 * plane + j] = f16::from_f32(v);
            }
        }
        let l = LAT as i64;
        let total = ddim.timesteps.len();
        for (k, &t) in ddim.timesteps.iter().enumerate() {
            for b in 0..2 {
                for (j, &v) in latents.iter().enumerate() {
                    x[b * 9 * plane + j] = f16::from_f32(v);
                }
            }
            let x_in = TensorRef::from_array_view(([2i64, 9, l, l], x.as_slice()))?;
            let c_in = TensorRef::from_array_view(([2i64, CTX as i64, 768], text.as_slice()))?;
            let (t16, t32, t64) = ([f16::from_f32(t as f32); 2], [t as f32; 2], [t as i64; 2]);
            let out = if ts_f32 {
                self.unet.run(ort::inputs![sample.as_str() => x_in, timestep.as_str() => TensorRef::from_array_view(([2i64], &t32[..]))?, ctx.as_str() => c_in])?
            } else if ts_i64 {
                self.unet.run(ort::inputs![sample.as_str() => x_in, timestep.as_str() => TensorRef::from_array_view(([2i64], &t64[..]))?, ctx.as_str() => c_in])?
            } else {
                self.unet.run(ort::inputs![sample.as_str() => x_in, timestep.as_str() => TensorRef::from_array_view(([2i64], &t16[..]))?, ctx.as_str() => c_in])?
            };
            let (dims, eps) = output_f32(&out, None)?;
            ensure!(dims == [2, 4, l, l], "UNet output shape {dims:?} (expected [2, 4, 64, 64])");
            drop(out);
            // Classifier-free guidance: uncond + g * (cond - uncond).
            let g = self.opts.guidance;
            let (u, c) = eps.split_at(4 * plane);
            let guided: Vec<f32> = u.iter().zip(c).map(|(&u, &c)| u + g * (c - u)).collect();
            ddim.step(t, &guided, &mut latents);
            if k == 0 || (k + 1) % 10 == 0 || k + 1 == total {
                eprintln!("[inpaint] sd15 step {}/{total} t={t} ({:.1}s)", k + 1, t_unet.elapsed().as_secs_f64());
            }
        }
        self.lap("diffusion_unet_s", t_unet);
        ensure!(latents.iter().all(|v| v.is_finite()), "SD-1.5: non-finite latents after sampling (fp16 overflow?)");
        let y = self.vae_decode(&latents)?;
        self.text = Some(text);
        let mut data = vec![0f32; 3 * n * n];
        for i in 0..n * n {
            for c in 0..3 {
                data[3 * i + c] = ((y[c * n * n + i] + 1.0) * 0.5).clamp(0.0, 1.0);
            }
        }
        Ok(RgbF { w: n, h: n, data })
    }
}
