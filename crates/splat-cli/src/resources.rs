use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use splat_contracts::*;
use splat_core::{Backend, RunCtx, ShotTools};
use splat_fetch::LicenseConsent;
use std::{fs, io::Read, path::{Path, PathBuf}, sync::Arc};

const LPIPS: &[&str] = &["lpips-vgg", "lpips-alex"];
const SAM: &[&str] = &["spirula-sam2", "spirula-grounding-dino", "grounding-dino-vocab"];

fn hash(path: &Path) -> Result<String> {
    let mut reader = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn models(cfg: &Config, ids: &[&str]) -> Result<PathBuf> {
    let assets = ids.iter().map(|id| splat_fetch::asset_for(id, splat_fetch::host_target())).collect::<Result<Vec<_>>>()?;
    let names = assets.iter().map(|a|Path::new(&a.entrypoint).file_name().context("Model asset has no filename")).collect::<Result<Vec<_>>>()?;
    if let Some(dir) = &cfg.models_dir {
        for name in &names {
            ensure!(dir.join(name).is_file(), "Configured models directory is missing {}", dir.join(name).display());
        }
        return Ok(dir.clone());
    }
    let all_bundled = ids.iter().all(|id| crate::config::bundled(id).is_some());
    if let Some(dev) = splat_contracts::dev::dev_root().map(|r| r.join("models"))
        && !all_bundled && names.iter().all(|name| dev.join(name).is_file()) { return Ok(dev); }
    let fingerprint = format!("{:x}",Sha256::digest(serde_json::to_vec(&assets)?));
    let dir = cfg.cache_root()?.join("model-sets").join(&fingerprint[..24]);
    fs::create_dir_all(&dir)?;
    let lock = fs::OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("assemble.lock"))?;
    fs2::FileExt::lock_exclusive(&lock)?;
    for (asset, name) in assets.iter().zip(&names) {
        let target = dir.join(name);
        if target.is_file() && hash(&target)? == asset.sha256 { continue; }
        eprintln!("Preparing {} ({})", asset.id, asset.license);
        let source = cfg.ensure(&asset.id, LicenseConsent::Default)?;
        let temp = dir.join(format!(".{}.tmp",uuid::Uuid::new_v4()));
        // Copy rather than hard-link: a user modifying assembled models must not corrupt the asset cache.
        fs::copy(&source,&temp)?;
        fs::File::open(&temp)?.sync_all()?;
        if target.exists() { fs::remove_file(&target)?; }
        fs::rename(&temp, &target)?;
    }
    let rows = assets.iter().zip(names).map(|(a,name)|json!({"name":a.id,"file":name.to_string_lossy(),"url":a.url,"sha256":a.sha256,"bytes":a.bytes,"license":a.license,"license_url":a.license_url,"source_url":a.source_url})).collect::<Vec<_>>();
    splat_core::atomic_json(dir.join("models.json"), &json!({"models":rows}))?;
    Ok(dir)
}

fn lpips(cfg: &Config) -> Result<Config> {
    let mut ready = cfg.clone();
    ready.models_dir = Some(models(cfg, LPIPS)?);
    if cfg.onnxruntime.is_none() && splat_eval::lpips::init_runtime(None).is_ok() {
        return Ok(ready);
    }
    let runtime = match &cfg.onnxruntime {
        Some(path) => path.clone(),
        None => cfg.ensure("onnxruntime", LicenseConsent::Default)?,
    };
    splat_eval::lpips::init_runtime(Some(&runtime))?;
    ready.onnxruntime = Some(runtime);
    Ok(ready)
}

/// Tool-specific preparation keeps status, stop, and completion reconciliation offline.
pub struct PreparedBackend {
    name: &'static str,
    cfg: Config,
    backend: Arc<dyn Backend>,
}
impl PreparedBackend {
    pub fn new(name: &'static str, cfg: Config) -> Result<Self> {
        Ok(Self { name, backend:cfg.backend(name)?, cfg })
    }
}
macro_rules! forward {
    ($method:ident, $request:ty) => {
        fn $method(&self, run: &dyn RunCtx, req: $request) -> Result<ToolResult> {
            self.backend.$method(run, req)
        }
    };
}
impl Backend for PreparedBackend {
    fn name(&self) -> &'static str { self.name }
    fn capabilities(&self) -> serde_json::Value { self.backend.capabilities() }
    forward!(inspect_input, InspectInput);
    forward!(extract_frames, ExtractFrames);
    forward!(define_split, DefineSplit);
    forward!(train_splats, TrainSplats);
    forward!(render_views, RenderViews);
    forward!(extract_mesh, ExtractMesh);

    fn reconstruct_sfm(&self, run: &dyn RunCtx, req: ReconstructSfm) -> Result<ToolResult> {
        let capture = self.name == "brush" && run.artifact(&req.frames_id)?.meta.get("input_kind") == Some(&json!("capture"));
        if capture || self.name != "brush" || self.backend.capabilities()["reconstruct_sfm"] == true {
            return self.backend.reconstruct_sfm(run, req);
        }
        let mut ready = self.cfg.clone();
        eprintln!("Preparing COLMAP for reconstruction");
        ready.colmap = Some(self.cfg.ensure("colmap", LicenseConsent::Default)?);
        ready.backend(self.name)?.reconstruct_sfm(run, req)
    }
    fn mask_distractors(&self, run: &dyn RunCtx, req: MaskDistractors) -> Result<ToolResult> {
        if self.backend.capabilities()["mask_distractors"] == true {
            return self.backend.mask_distractors(run, req);
        }
        let mut ready = self.cfg.clone();
        if self.name == "brush" {
            ready.spirula = Some(self.cfg.spirula_binary("the distractor-mask helper (Spirula sam track)")?);
        }
        ready.models_dir = Some(models(&self.cfg, SAM)?);
        ready.backend(self.name)?.mask_distractors(run, req)
    }
    fn select_objects(&self, run: &dyn RunCtx, req: SelectObjects) -> Result<ToolResult> {
        if self.name != "brush" || self.backend.capabilities()["select_objects"] == true {
            return self.backend.select_objects(run, req);
        }
        let mut ready = self.cfg.clone();
        ready.spirula = Some(self.cfg.spirula_binary("the object segmenter (Spirula sam track)")?);
        ready.models_dir = Some(models(&self.cfg, SAM)?);
        ready.backend(self.name)?.select_objects(run, req)
    }
    fn inpaint_removed(&self, run: &dyn RunCtx, req: InpaintRemoved) -> Result<ToolResult> {
        self.backend.inpaint_removed(run, req)
    }
    fn evaluate(&self, run: &dyn RunCtx, req: Evaluate) -> Result<ToolResult> {
        let no_heldout = run.artifact(&req.split_id).ok().and_then(|a| run.path(&a.id).ok())
            .and_then(|p| fs::read(p.join("manifest.json")).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|v| v["mode"] == "imported_explicit" && v["heldout_names"].as_array().is_some_and(Vec::is_empty));
        if !req.lpips || no_heldout { return self.backend.evaluate(run, req); }
        match lpips(&self.cfg) {
            Ok(ready) => ready.backend(self.name)?.evaluate(run, req),
            Err(error) => {
                let mut request = req;
                request.lpips = false;
                let mut result = self.backend.evaluate(run, request)?;
                let reason = format!("{error:#}");
                result.metrics.insert("lpips_unavailable".into(), json!(reason));
                result.warnings.push(format!("LPIPS unavailable: {reason}"));
                Ok(result)
            }
        }
    }
    fn on_job_complete(&self, run: &dyn RunCtx, handler: &str, dir: &Path, exit_code: i32) -> Result<ToolResult> {
        self.backend.on_job_complete(run, handler, dir, exit_code)
    }
    fn job_progress(&self, run: &dyn RunCtx, dir: &Path) -> serde_json::Value {
        self.backend.job_progress(run, dir)
    }
}

pub struct PreparedShots(pub Config);
impl PreparedShots {
    fn tools(&self, encode: bool) -> Result<splat_shots::Shots> {
        let mut shots = splat_shots::Shots::new();
        if encode {
            shots = shots.with_ffmpeg(self.0.resolve_ffmpeg()?);
        } else if let Some(path) = &self.0.ffmpeg {
            shots = shots.with_ffmpeg(path);
        }
        if let Some(path) = &self.0.blender {
            shots = shots.with_blender(path);
        }
        Ok(shots)
    }
}
impl ShotTools for PreparedShots {
    fn plan_shot(&self, run: &dyn RunCtx, req: PlanShot) -> Result<ToolResult> {
        self.tools(false)?.plan_shot(run,req)
    }
    fn render_shot(&self, run: &dyn RunCtx, req: RenderShot) -> Result<ToolResult> {
        self.tools(true)?.render_shot(run,req)
    }
    fn generate_shot(&self, run: &dyn RunCtx, req: GenerateShot) -> Result<ToolResult> {
        self.tools(true)?.generate_shot(run,req)
    }
    fn export_blender(&self, run: &dyn RunCtx, req: ExportBlender) -> Result<ToolResult> {
        self.tools(false)?.export_blender(run,req)
    }
    fn on_job_complete(&self, run: &dyn RunCtx, handler: &str, dir: &Path, exit_code: i32) -> Result<ToolResult> {
        self.tools(false)?.on_job_complete(run,handler,dir,exit_code)
    }
}
