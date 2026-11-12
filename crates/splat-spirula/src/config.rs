use crate::util::{hash, read, safe_child, string};
use anyhow::{Result, ensure};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    env,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct SpirulaConfig {
    pub binary: PathBuf,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub sam_model: PathBuf,
    pub sam_detector: PathBuf,
    pub models_manifest: Option<PathBuf>,
    /// LPIPS ONNX model directory; None uses splat-eval's platform discovery.
    pub models_dir: Option<PathBuf>,
    /// Predecoded RGB/mask cache estimate; excludes engine and decoder scratch space.
    pub cpu_cache_max_bytes: u64,
    /// Dev-host coordination only; None elsewhere (flock is not portable).
    pub gpu_lock: Option<PathBuf>,
}
impl Default for SpirulaConfig {
    fn default() -> Self {
        let dev_root = splat_contracts::dev::dev_root();
        // SPLAT_MODELS_DIR, else the dev root's models/; neither: bare file names (the CLI fills these in).
        let models = env::var_os("SPLAT_MODELS_DIR").map(PathBuf::from).or_else(|| dev_root.map(|r| r.join("models")));
        let model = |name: &str| models.as_ref().map_or_else(|| PathBuf::from(name), |m| m.join(name));
        let manifest = env::var_os("SPLAT_AGENT_MODELS_JSON").map(PathBuf::from).unwrap_or_else(|| model("models.json"));
        Self {
            binary: env::var_os("SPLAT_AGENT_SPIRULA")
                .map(PathBuf::from)
                .or_else(|| dev_root.map(|r| r.join("build/release/spirula")).filter(|p| p.is_file()))
                .unwrap_or_else(|| "spirula".into()),
            ffmpeg: configured("SPLAT_FFMPEG", "ffmpeg"),
            ffprobe: configured("SPLAT_FFPROBE", "ffprobe"),
            sam_model: env::var_os("SPLAT_AGENT_SAM_MODEL")
                .map(PathBuf::from)
                .unwrap_or_else(|| model("sam2.1_hiera_small_f16.ggml")),
            sam_detector: env::var_os("SPLAT_AGENT_SAM_DETECTOR")
                .map(PathBuf::from)
                .unwrap_or_else(|| model("grounding-dino-tiny.safetensors")),
            models_manifest: manifest.is_file().then_some(manifest),
            models_dir: models.filter(|m| m.is_dir()),
            cpu_cache_max_bytes: env::var("SPLAT_SPIRULA_CPU_CACHE_MAX_BYTES")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|n| *n > 0)
                .unwrap_or(4 * 1024 * 1024 * 1024),
            gpu_lock: splat_contracts::dev::gpu_lock().filter(|_| cfg!(target_os = "linux")),
        }
    }
}
fn configured(key: &str, fallback: &str) -> PathBuf {
    env::var_os(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| fallback.into())
}
impl SpirulaConfig {
    pub(crate) fn argv(&self, args: Vec<String>) -> Result<Vec<String>> {
        let mut out = vec![];
        if let Some(lock) = &self.gpu_lock {
            out.extend(["flock".into(), "--no-fork".into(), string(lock)?]);
        }
        out.push(string(&self.binary)?);
        out.extend(args);
        Ok(out)
    }
    pub(crate) fn env(&self) -> BTreeMap<String, String> {
        let mut env = BTreeMap::from([
            ("SS_LANG".into(), "en".into()),
            ("OMP_NUM_THREADS".into(), "8".into()),
        ]);
        let icd = "/usr/share/vulkan/icd.d/nvidia_icd.json";
        if Path::new(icd).is_file() {
            env.insert("VK_ICD_FILENAMES".into(), icd.into());
        }
        // Spirula locates its FFmpeg subprocess through PATH.
        if let Some(parent) = self.ffmpeg.parent().filter(|p| !p.as_os_str().is_empty()) {
            let mut paths = vec![parent.to_path_buf()];
            paths.extend(std::env::split_paths(
                &std::env::var_os("PATH").unwrap_or_default(),
            ));
            if let Ok(path) = std::env::join_paths(paths) {
                env.insert("PATH".into(), path.to_string_lossy().into_owned());
            }
        }
        env
    }
    pub(crate) fn mask_available(&self) -> bool {
        self.sam_model.is_file()
            && self.sam_detector.is_file()
            && self
                .sam_detector
                .parent()
                .is_some_and(|p| p.join("vocab.txt").is_file())
    }
    pub(crate) fn verify_models(&self) -> Result<()> {
        ensure!(
            self.sam_model.extension().is_some_and(|s| s == "ggml"),
            "SAM requires GGML weights, not the original .pt checkpoint"
        );
        let Some(manifest) = &self.models_manifest else {
            return Ok(());
        };
        let value: Value = read(manifest)?;
        let rows = value["models"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("models.json must contain a models array"))?;
        for model in [
            &self.sam_model,
            &self.sam_detector,
            &self.sam_detector.parent().unwrap().join("vocab.txt"),
        ] {
            let name = model
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            let row = rows
                .iter()
                .find(|r| r["file"].as_str() == Some(name))
                .ok_or_else(|| anyhow::anyhow!("No manifest entry for {name}"))?;
            safe_child(manifest.parent().unwrap_or(Path::new(".")), name)?;
            if let Some(digest) = row["sha256"].as_str() {
                ensure!(hash(model)? == digest, "Model checksum mismatch: {name}");
            }
        }
        Ok(())
    }
}
