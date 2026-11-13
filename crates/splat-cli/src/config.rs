use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_core::{FakeBackend, Registry};
use splat_fetch::{Fetcher, LicenseConsent};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// Spirula is a prebuilt GUI-capable binary: on a headless Linux box without libOpenGL it dies at load
/// time, which would otherwise surface only later in a job log. Probe `--help` once and name what is missing.
fn spirula_starts(path: &Path, why: &str) -> Result<()> {
    let out = std::process::Command::new(path).arg("--help").stdin(std::process::Stdio::null()).output()
        .with_context(|| format!("{why}: cannot start Spirula ({})", path.display()))?;
    let err = String::from_utf8_lossy(&out.stderr);
    if let Some(line) = err.lines().find(|l| l.contains("error while loading shared libraries")) {
        let lib = line.split(": ").nth(2).unwrap_or(line).trim();
        bail!("{why}: Spirula cannot start on this machine, a system library is missing ({lib}). On Debian/Ubuntu \
               install it (libOpenGL.so.0: `apt install libopengl0`; libGLX/libEGL: `libglx0 libegl1`), or skip \
               object selection / distractor masks");
    }
    Ok(())
}

/// Release archives stage helpers at `<exe_dir>/tools/<asset-id>/<entrypoint>`, mirroring the cache layout.
pub fn bundled(id: &str) -> Option<PathBuf> {
    let asset = splat_fetch::asset_for(id, splat_fetch::host_target()).ok()?;
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    // tools/ sits beside the binary in the splat archive, and beside bin/ in the Voyager engine archive.
    let dir = exe.parent()?;
    [dir.join("tools"), dir.parent()?.join("tools")]
        .into_iter()
        .map(|tools| tools.join(&asset.id).join(&asset.entrypoint))
        .find(|path| path.is_file())
}

fn dev_colmap() -> Option<PathBuf> {
    let path = splat_contracts::dev::dev_root()?.join("tools/colmap-4.2.1/bin/colmap");
    path.is_file().then_some(path)
}

/// Plain-language remedy for a dependency `readiness` could not resolve.
fn fix(name: &str, entry: &Value) -> String {
    let error = entry["error"].as_str().unwrap_or("");
    if error.contains("SPLAT_GITHUB_TOKEN") {
        return format!("use a release archive that bundles tools/{name}/, set SPLAT_GITHUB_TOKEN, or set SPLAT_MODELS_DIR to a folder with the model files");
    }
    let configured = entry["source"] == "configured";
    match name {
        "ffmpeg" if configured => "point SPLAT_FFMPEG (or config `ffmpeg`) at an existing ffmpeg with ffprobe beside it".into(),
        "ffmpeg" => "install FFmpeg and set SPLAT_FFMPEG to the ffmpeg binary (ffprobe must sit beside it)".into(),
        "colmap" if configured => "point SPLAT_COLMAP (or config `colmap`) at an existing COLMAP 4.x binary".into(),
        "colmap" => "install COLMAP 4.x and put `colmap` on PATH or set SPLAT_COLMAP; or use a release archive that bundles tools/colmap/".into(),
        "spirula" => "set SPLAT_AGENT_SPIRULA to the Spirula binary, or use --backend brush".into(),
        "onnxruntime" => "set SPLAT_ONNXRUNTIME to an ONNX Runtime 1.28 library".into(),
        _ => "set SPLAT_MODELS_DIR to a folder containing the model files".into(),
    }
}

fn file_status(path: &Path, source: &str) -> Value {
    if path.is_file() {
        json!({"status":"ready","source":source,"path":path})
    } else {
        json!({"status":"missing","source":source,"path":path,"error":"configured path does not exist"})
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub ffmpeg: Option<PathBuf>,
    pub ffprobe: Option<PathBuf>,
    pub spirula: Option<PathBuf>,
    pub colmap: Option<PathBuf>,
    pub voyager_ffmpeg_tree: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    pub onnxruntime: Option<PathBuf>,
    pub blender: Option<PathBuf>,
}
impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = path
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("SPLAT_CONFIG").map(PathBuf::from));
        let mut cfg = if let Some(path) = path {
            serde_json::from_slice::<Self>(
                &std::fs::read(&path).with_context(|| format!("Read config {}", path.display()))?,
            )?
        } else {
            Self::default()
        };
        fn env(keys: &[&str]) -> Option<PathBuf> {
            keys.iter().find_map(|k| {
                std::env::var_os(k)
                    .filter(|v| !v.is_empty())
                    .map(PathBuf::from)
            })
        }
        cfg.ffmpeg = env(&["SPLAT_FFMPEG", "FFMPEG_BIN"]).or(cfg.ffmpeg);
        cfg.ffprobe = env(&["SPLAT_FFPROBE", "FFPROBE_BIN"]).or(cfg.ffprobe);
        cfg.colmap = env(&["SPLAT_COLMAP", "COLMAP_BIN"]).or(cfg.colmap);
        cfg.spirula = env(&["SPLAT_AGENT_SPIRULA", "SPIRULA_BIN"]).or(cfg.spirula);
        cfg.voyager_ffmpeg_tree = env(&["SPLAT_VOYAGER_FFMPEG_TREE"]).or(cfg.voyager_ffmpeg_tree);
        cfg.cache_dir = env(&["SPLAT_CACHE_DIR"]).or(cfg.cache_dir);
        cfg.models_dir = env(&["SPLAT_MODELS_DIR"]).or(cfg.models_dir);
        cfg.onnxruntime = env(&["SPLAT_ONNXRUNTIME", "ORT_DYLIB_PATH"]).or(cfg.onnxruntime);
        cfg.blender = env(&["SPLAT_BLENDER"]).or(cfg.blender);
        Ok(cfg)
    }
    pub fn for_run(&self, root: &Path) -> Result<Self> {
        let path = root.join("toolchain.json");
        let mut previous = if path.is_file() {
            serde_json::from_slice::<Self>(&std::fs::read(path)?)?
        } else {
            Self::default()
        };
        previous.ffmpeg = self.ffmpeg.clone().or(previous.ffmpeg);
        previous.ffprobe = self.ffprobe.clone().or(previous.ffprobe);
        previous.spirula = self.spirula.clone().or(previous.spirula);
        previous.colmap = self.colmap.clone().or(previous.colmap);
        previous.voyager_ffmpeg_tree = self
            .voyager_ffmpeg_tree
            .clone()
            .or(previous.voyager_ffmpeg_tree);
        previous.cache_dir = self.cache_dir.clone().or(previous.cache_dir);
        previous.models_dir = self.models_dir.clone().or(previous.models_dir);
        previous.onnxruntime = self.onnxruntime.clone().or(previous.onnxruntime);
        previous.blender = self.blender.clone().or(previous.blender);
        Ok(previous)
    }
    pub fn fetcher(&self) -> Result<Fetcher> {
        let fetcher = Fetcher::new(self.cache_root()?);
        Ok(if let Ok(token) = std::env::var("SPLAT_GITHUB_TOKEN") {
            fetcher.with_github_token(token)
        } else {
            fetcher
        })
    }
    pub fn cache_root(&self) -> Result<PathBuf> {
        self.cache_dir.clone().map(Ok).unwrap_or_else(splat_fetch::default_cache_dir)
    }
    pub fn registry(&self, backend: &str, prepare: bool) -> Result<Arc<Registry>> {
        let ready = if prepare && matches!(backend, "brush" | "spirula") {
            self.prepared(backend)?
        } else {
            self.clone()
        };
        let mut registry = Registry::new();
        registry.register_backend(Arc::new(FakeBackend::default()));
        registry.set_shots(Arc::new(crate::resources::PreparedShots(ready.clone())));
        for name in ["brush", "spirula"] {
            registry.register_backend(Arc::new(crate::resources::PreparedBackend::new(name, ready.clone())?));
        }
        Ok(Arc::new(registry))
    }
    pub fn prepared(&self, backend: &str) -> Result<Self> {
        let mut ready = self.clone();
        let ffmpeg = self.resolve_ffmpeg()?;
        let ffprobe = self.ffprobe.clone().unwrap_or_else(|| ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" }));
        ensure!(ffprobe.is_file(), "ffprobe missing next to FFmpeg; configure ffprobe explicitly");
        ready.ffmpeg = Some(ffmpeg);
        ready.ffprobe = Some(ffprobe);
        if backend == "spirula" {
            ready.spirula = Some(self.spirula_binary("selected optional backend Spirula")?);
        }
        Ok(ready)
    }
    /// The Spirula executable: configured, dev host, bundled, already fetched with consent (`vauss fetch
    /// spirula --allow-gpl`), else downloaded only when SPLAT_ALLOW_GPL=1 (GPL, separate executable).
    pub fn spirula_binary(&self, why: &str) -> Result<PathBuf> {
        self.locate_spirula(why).and_then(|path| spirula_starts(&path, why).map(|()| path))
    }
    fn locate_spirula(&self, why: &str) -> Result<PathBuf> {
        let binary = self.spirula.clone().unwrap_or_else(||splat_spirula::SpirulaConfig::default().binary);
        if binary.is_file() {
            return Ok(binary);
        }
        ensure!(self.spirula.is_none(), "Configured Spirula does not exist: {}", binary.display());
        if let Some(path) = bundled("spirula") {
            return Ok(path);
        }
        let asset = splat_fetch::asset_for("spirula", splat_fetch::host_target())?;
        if let Some(path) = self.fetcher()?.cached(&asset)? {
            return Ok(path);
        }
        ensure!(std::env::var("SPLAT_ALLOW_GPL").is_ok_and(|v| v == "1"),
            "{why} needs Spirula ({}, a separate executable, never linked): {}. Download it once with \
             `vauss fetch spirula --allow-gpl`, or set SPLAT_ALLOW_GPL=1 to allow the download on first use",
            asset.license, asset.source_url);
        eprintln!("Preparing {why} ({}, separate executable, SPLAT_ALLOW_GPL=1): {}",asset.license,asset.source_url);
        self.ensure("spirula", LicenseConsent::AllowGpl)
    }
    pub fn resolve_ffmpeg(&self) -> Result<PathBuf> {
        let bundled = (self.ffmpeg.is_none() && self.voyager_ffmpeg_tree.is_none())
            .then(|| bundled("ffmpeg"))
            .flatten();
        self.fetcher()?.ffmpeg(self.ffmpeg.as_deref().or(bundled.as_deref()), self.voyager_ffmpeg_tree.as_deref())
    }
    /// First-use resolution: a helper staged in the release archive wins over the verified cache/download.
    /// The COLMAP vocabulary tree for video loop detection (asset colmap-vocab-tree-32k): cached, or
    /// downloaded now (says so on stderr).
    pub fn vocab_tree(&self) -> Result<PathBuf> {
        const ID: &str = "colmap-vocab-tree-32k";
        if let Some(path) = bundled(ID) {
            return Ok(path);
        }
        let fetcher = self.fetcher()?;
        let asset = splat_fetch::asset_for(ID, splat_fetch::host_target())?;
        if let Some(path) = fetcher.cached(&asset)? {
            return Ok(path);
        }
        eprintln!("Downloading the COLMAP vocabulary tree for video loop detection ({ID}, {:.1} MB, {})",
            asset.bytes as f64 / 1e6, asset.license);
        fetcher.ensure_asset(&asset, LicenseConsent::Default).with_context(|| format!(
            "video camera solve needs the COLMAP vocabulary tree ({ID}) for loop detection and it could not be \
             downloaded; run `vauss fetch {ID}` once online, or set SPLAT_VOCAB_TREE to the file"))
    }
    pub fn ensure(&self, id: &str, consent: LicenseConsent) -> Result<PathBuf> {
        if let Some(path) = bundled(id) {
            return Ok(path);
        }
        self.fetcher()?.ensure(id, consent)
    }
    /// Offline answer to "what will `vauss run` need on this machine": each dependency's source, or what
    /// first use will download, or why it cannot be resolved. Never downloads.
    pub fn readiness(&self, backend: &str) -> Result<Value> {
        self.readiness_with_camera_policy(backend, false)
    }
    fn readiness_with_camera_policy(&self, backend: &str, trust_poses: bool) -> Result<Value> {
        let fetcher = self.fetcher()?;
        let token = std::env::var("SPLAT_GITHUB_TOKEN").is_ok_and(|t| !t.is_empty());
        let locate = |id: &str| -> Value {
            if let Some(path) = bundled(id) {
                return json!({"status":"ready","source":"bundled","path":path});
            }
            let asset = match splat_fetch::asset_for(id, splat_fetch::host_target()) {
                Ok(asset) => asset,
                Err(error) => return json!({"status":"missing","error":error.to_string()}),
            };
            if let Ok(Some(path)) = fetcher.cached(&asset) {
                return json!({"status":"ready","source":"cache","path":path});
            }
            // The download tries the public URL first; github_api_url is only the authenticated fallback for
            // a repository the caller can't read anonymously, so it never makes an uncached asset missing here.
            // The real request decides, and names SPLAT_GITHUB_TOKEN if the anonymous one is refused.
            json!({"status":"download_on_first_use","asset":asset.id,"version":asset.version,"bytes":asset.bytes,"license":asset.license,"requires_gpl_consent":asset.requires_gpl_consent,"github_api_fallback":asset.github_api_url.is_some(),"github_token":token})
        };
        let file = file_status;
        let mut required = serde_json::Map::new();
        let mut optional = serde_json::Map::new();
        required.insert("ffmpeg".into(), match (&self.ffmpeg, &self.voyager_ffmpeg_tree) {
            (Some(path), _) => file(path, "configured"),
            (None, Some(tree)) => file(&tree.join("bin").join(if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" }), "voyager"),
            (None, None) => locate("ffmpeg"),
        });
        if backend == "brush" && !trust_poses {
            let found = self.colmap.clone().map(|p| (p, "configured"))
                .or_else(|| bundled("colmap").map(|p| (p, "bundled")))
                .or_else(|| splat_brush::BrushConfig::from_env().colmap.map(|p| (p, "env_or_path")))
                .or_else(|| dev_colmap().map(|p| (p, "dev_host")));
            required.insert("colmap".into(), match found {
                Some((path, source)) => file(&path, source),
                None => locate("colmap"),
            });
        }
        if matches!(backend, "spirula" | "brush") {
            let binary = self.spirula.clone().unwrap_or_else(|| splat_spirula::SpirulaConfig::default().binary);
            let status = if binary.is_file() {
                json!({"status":"ready","source":"configured","path":binary})
            } else if self.spirula.is_some() {
                file(&binary, "configured")
            } else {
                locate("spirula")
            };
            // Brush needs Spirula only as the distractor-mask helper.
            if backend == "spirula" { required.insert("spirula".into(), status); } else { optional.insert("spirula".into(), status); }
            for id in ["spirula-sam2", "spirula-grounding-dino", "grounding-dino-vocab"] {
                optional.insert(id.into(), self.model(id, &locate));
            }
        }
        optional.insert("onnxruntime".into(), match &self.onnxruntime {
            Some(path) => file(path, "configured"),
            None => locate("onnxruntime"),
        });
        for id in ["lpips-alex", "lpips-vgg"] {
            optional.insert(id.into(), self.model(id, &locate));
        }
        for (name, entry) in required.iter_mut().chain(optional.iter_mut()) {
            if entry["status"] == "missing" {
                entry["fix"] = json!(fix(name, entry));
            }
        }
        let blockers = required.iter().filter(|(_, v)| v["status"] == "missing").map(|(k, _)| k.clone()).collect::<Vec<_>>();
        let degraded = optional.iter().filter(|(_, v)| v["status"] == "missing").map(|(k, _)| k.clone()).collect::<Vec<_>>();
        let download_bytes = required.values().chain(optional.values()).filter(|v| v["status"] == "download_on_first_use").filter_map(|v| v["bytes"].as_u64()).sum::<u64>();
        let drivers = json!({
            "fixed": true,
            "claude": splat_core::doctor::find_binary("claude", "CLAUDE_BIN").is_some(),
            "codex": splat_core::doctor::find_binary("codex", "CODEX_BIN").is_some(),
        });
        Ok(json!({
            "ready": blockers.is_empty(),
            "blockers": blockers,
            "degraded": degraded,
            "first_use_download_bytes": download_bytes,
            "required": required,
            "optional": optional,
            "drivers": drivers,
            "note": "optional items only degrade results (missing LPIPS -> lpips_unavailable; missing SAM -> no distractor masks); `--driver fixed` needs no agent CLI",
        }))
    }
    /// Fail before the run is created or an agent starts, in plain text, when a required helper cannot
    /// be resolved. Optional gaps only print a warning. A missing agent CLI is not fatal: the run falls
    /// back to the fixed pipeline (splat_agent::DriverResult::unavailable).
    pub fn preflight(&self, backend: &str) -> Result<()> {
        self.preflight_with_camera_policy(backend, false)
    }
    pub fn preflight_with_camera_policy(&self, backend: &str, trust_poses: bool) -> Result<()> {
        ensure!(!trust_poses || backend == "brush", "--trust-poses requires the brush backend");
        if !matches!(backend, "brush" | "spirula") {
            return Ok(());
        }
        let report = self.readiness_with_camera_policy(backend, trust_poses)?;
        let mut problems = vec![];
        for (name, entry) in report["required"].as_object().into_iter().flatten() {
            if entry["status"] == "missing" {
                problems.push(format!("  - {name}: {}\n    fix: {}", entry["error"].as_str().unwrap_or("not found"), entry["fix"].as_str().unwrap_or("")));
            }
        }
        for (name, entry) in report["optional"].as_object().into_iter().flatten() {
            if entry["status"] == "missing" {
                eprintln!("Warning: {name} unavailable ({}); results will be degraded. Fix: {}", entry["error"].as_str().unwrap_or("not found"), entry["fix"].as_str().unwrap_or(""));
            }
        }
        ensure!(
            problems.is_empty(),
            "Vauss cannot start this run on {} ({}):\n{}\nRun `vauss doctor --backend {backend}` for the full report.",
            splat_fetch::host_target(),
            backend,
            problems.join("\n")
        );
        Ok(())
    }
    fn model(&self, id: &str, locate: &dyn Fn(&str) -> Value) -> Value {
        let dev = splat_contracts::dev::dev_root().map(|r| r.join("models"));
        let (dir, source) = match (&self.models_dir, &dev) {
            (Some(dir), _) => (dir.as_path(), "models_dir"),
            (None, _) if bundled(id).is_some() => return locate(id),
            (None, Some(dev)) if dev.is_dir() => (dev.as_path(), "dev_host"),
            (None, _) => return locate(id),
        };
        match splat_fetch::asset_for(id, splat_fetch::host_target()) {
            Ok(asset) => match Path::new(&asset.entrypoint).file_name() {
                Some(name) => file_status(&dir.join(name), source),
                None => json!({"status":"missing","error":"asset has no filename"}),
            },
            Err(error) => json!({"status":"missing","error":error.to_string()}),
        }
    }
    pub fn backend(&self, backend: &str) -> Result<Arc<dyn splat_core::Backend>> {
        let mut spirula = splat_spirula::SpirulaConfig::default();
        if let Some(binary) = &self.spirula {
            spirula.binary = binary.clone();
        }
        if let Some(ffmpeg) = &self.ffmpeg {
            spirula.ffmpeg = ffmpeg.clone();
        }
        if let Some(ffprobe) = &self.ffprobe {
            spirula.ffprobe = ffprobe.clone();
        }
        if let Some(models) = &self.models_dir {
            spirula.models_dir = Some(models.clone());
            spirula.sam_model = models.join("sam2.1_hiera_small_f16.ggml");
            spirula.sam_detector = models.join("grounding-dino-tiny.safetensors");
            spirula.models_manifest = Some(models.join("models.json")).filter(|p| p.is_file());
        }
        if backend == "spirula" { return Ok(Arc::new(splat_spirula::SpirulaBackend::with_config(spirula))); }
        ensure!(backend == "brush", "Unknown backend: {backend}");
        let mut brush = splat_brush::BrushConfig::from_env();
        brush.ffmpeg = spirula.ffmpeg.clone();
        brush.ffprobe = spirula.ffprobe.clone();
        brush.models_dir = self.models_dir.clone().or(brush.models_dir);
        // Explicit config/env, then the archive's pinned build, then PATH, then the dev host.
        brush.colmap = self.colmap.clone().or_else(|| bundled("colmap")).or(brush.colmap).or_else(dev_colmap);
        // Video SfM's loop detection, resolved when a video solve starts (other commands never fetch it).
        // Offline with no cache, that solve fails with how to get the tree.
        if brush.vocab_tree.is_none() {
            let cfg = self.clone();
            brush.vocab_tree_fetch = Some(splat_brush::VocabTreeFetch(Arc::new(move || cfg.vocab_tree())));
        }
        // SfM learned-pose rescue: explicit SPLAT_DA3_WEIGHTS, the models dir's reduced f16 file (the dev
        // host's when none is configured), else the da3-base asset if `vauss fetch da3-base` cached it (541 MB,
        // never downloaded implicitly). None: the rescue is skipped with a warning.
        brush.da3_weights = brush.da3_weights.clone().filter(|p| p.is_file()).or_else(|| {
            let models = self.models_dir.clone().or_else(|| splat_contracts::dev::dev_root().map(|r| r.join("models")))?;
            Some(models.join("da3-base/da3-base-cam-f16.safetensors")).filter(|p| p.is_file())
        }).or_else(|| {
            let asset = splat_fetch::asset_for("da3-base", splat_fetch::host_target()).ok()?;
            self.fetcher().ok()?.cached(&asset).ok().flatten()
        });
        // SfM learned-feature fallback (ALIKED + LightGlue ONNX): explicit SPLAT_ALIKED_DIR, else the models dir's.
        brush.aliked_dir = brush.aliked_dir.clone().filter(|p| p.is_dir()).or_else(|| {
            let models = self.models_dir.clone().or_else(|| splat_contracts::dev::dev_root().map(|r| r.join("models")))?;
            Some(models.join("aliked")).filter(|p| p.join("aliked-n16rot.onnx").is_file() && p.join("aliked-lightglue.onnx").is_file())
        });
        // Distractor segmentation (SAM 2.1 + Grounding DINO) runs as Spirula's `sam track` helper;
        // Brush consumes the resulting masks itself (SfM feature masks, loss masks).
        // select_objects segments with the same helper and weights, in its own GPU worker.
        if spirula.binary.is_file() && spirula.sam_model.is_file() && spirula.sam_detector.is_file() {
            brush.sam = Some(splat_brush::select::SamTool {
                binary: spirula.binary.clone(),
                model: spirula.sam_model.clone(),
                detector: spirula.sam_detector.clone(),
            });
        }
        let masker = Arc::new(splat_spirula::SpirulaBackend::with_config(spirula));
        Ok(Arc::new(splat_brush::BrushBackend::with_masker(brush, masker)))
    }
    pub fn offline_assets(&self) -> Result<serde_json::Value> {
        let fetcher = self.fetcher()?;
        let assets=splat_fetch::catalog()?.into_iter().filter(|a|a.target==splat_fetch::host_target()||a.target=="any").map(|a|{
            let cached=fetcher.cached(&a);
            serde_json::json!({"id":a.id,"target":a.target,"version":a.version,"license":a.license,"cached":cached.as_ref().ok().and_then(|p|p.clone()),"error":cached.err().map(|e|e.to_string())})
        }).collect::<Vec<_>>();
        Ok(serde_json::json!({"config":self,"assets":assets}))
    }
}

#[cfg(test)]
mod spirula_consent {
    use super::Config;

    #[test]
    fn trusted_readiness_does_not_require_or_resolve_colmap() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config { colmap: Some(dir.path().join("missing-colmap")),
            cache_dir: Some(dir.path().join("cache")), ..Default::default() };
        let normal = cfg.readiness("brush").unwrap();
        assert_eq!(normal["required"]["colmap"]["status"], "missing");
        let trusted = cfg.readiness_with_camera_policy("brush", true).unwrap();
        assert!(trusted["required"].get("colmap").is_none());
        assert!(cfg.preflight_with_camera_policy("spirula", true).unwrap_err().to_string().contains("brush"));
    }

    #[test]
    fn never_fetched_without_consent() {
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: the only test in this binary that touches these variables.
        unsafe {
            std::env::set_var("SPLAT_AGENT_SPIRULA", dir.path().join("absent"));
            std::env::remove_var("SPLAT_ALLOW_GPL");
        }
        let cfg = Config { cache_dir: Some(dir.path().join("cache")), ..Default::default() };
        let err = format!("{:#}", cfg.spirula_binary("the object segmenter").unwrap_err());
        assert!(err.contains("vauss fetch spirula --allow-gpl") && err.contains("SPLAT_ALLOW_GPL=1"), "{err}");
        assert!(!dir.path().join("cache").exists(), "nothing may be downloaded without consent");
    }

    #[cfg(unix)]
    #[test]
    fn missing_system_library_is_named() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let fake = dir.path().join("spirula");
        std::fs::write(&fake, "#!/bin/sh\necho 'spirula: error while loading shared libraries: libOpenGL.so.0: cannot open shared object file: No such file or directory' >&2\nexit 127\n").unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let cfg = Config { spirula: Some(fake), ..Default::default() };
        let err = format!("{:#}", cfg.spirula_binary("the object segmenter").unwrap_err());
        assert!(err.contains("libOpenGL.so.0") && err.contains("apt install libopengl0"), "{err}");
    }
}
