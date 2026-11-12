//! `--cloud`: run an effect command on a fal GPU through the engine's remote-job path
//! (splat_core::remote, the one `vauss run --cloud` jobs use). A throwaway run root holds the
//! inputs under artifacts/fxin and the outputs under artifacts/fxout, which sync back.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use splat_contracts::JobSpec;

pub struct Job {
    pub root: PathBuf,
    pub inputs: PathBuf,
    pub outputs: PathBuf,
}

/// A fresh run root next to `near` (an output path), canonical so remote staging maps its paths.
pub fn job(near: &Path) -> Result<Job> {
    let parent = near.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let root = parent.join(format!(".fx-cloud-{}", uuid::Uuid::new_v4().simple()));
    fs::create_dir_all(root.join("artifacts/fxin"))?;
    fs::create_dir_all(root.join("artifacts/fxout"))?;
    fs::create_dir_all(root.join("jobs"))?;
    let root = root.canonicalize()?;
    Ok(Job { inputs: root.join("artifacts/fxin"), outputs: root.join("artifacts/fxout"), root })
}

fn link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    if fs::hard_link(src, dst).is_err() {
        fs::copy(src, dst).with_context(|| format!("copying {}", src.display()))?;
    }
    Ok(())
}

impl Job {
    /// Stage a scene input: a package dir (its scene.json and the files the splat reader opens) or a file.
    pub fn add_scene(&self, scene: &Path) -> Result<PathBuf> {
        if scene.is_file() {
            let dst = self.inputs.join(scene.file_name().context("scene path has no file name")?);
            link_or_copy(scene, &dst)?;
            return Ok(dst);
        }
        let dst = self.inputs.join("package");
        fs::create_dir_all(&dst)?;
        let meta: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
        let mut files = vec!["scene.json".to_string()];
        files.extend(meta["splat"].as_str().map(String::from));
        // spz-overflow-v1 sidecars (positions / dc) that complete scene.spz.
        if let Some(o) = meta["splat_overflow"].as_object() {
            for k in ["positions", "dc"] {
                files.extend(o.get(k).and_then(Value::as_str).map(String::from));
            }
        }
        files.extend(meta["splat_neural"]["file"].as_str().map(String::from));
        // The physical effects' floor plane (eval_opts reads it; without it the floor is y = 0).
        files.push("collision/floor.json".to_string());
        for f in files {
            let src = scene.join(&f);
            if src.is_file() {
                if let Some(parent) = dst.join(&f).parent() {
                    fs::create_dir_all(parent)?;
                }
                link_or_copy(&src, &dst.join(&f))?;
            }
        }
        // Exported effects and their object masks (.mask.u8), which graphs name relative to the package.
        if let Ok(rd) = fs::read_dir(scene.join("effects")) {
            fs::create_dir_all(dst.join("effects"))?;
            for e in rd.flatten().filter(|e| e.path().is_file()) {
                link_or_copy(&e.path(), &dst.join("effects").join(e.file_name()))?;
            }
        }
        Ok(dst)
    }

    pub fn add_file(&self, src: &Path, name: &str) -> Result<PathBuf> {
        let dst = self.inputs.join(name);
        link_or_copy(src, &dst)?;
        Ok(dst)
    }

    /// Run `args` (the `splat` arguments, absolute paths under this root) on fal; returns the job's cost.
    pub fn run(&self, tool: &str, args: Vec<String>) -> Result<Value> {
        let exe = std::env::current_exe()?.canonicalize()?;
        let dir = self.root.join("jobs").join(format!("job_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]));
        fs::create_dir_all(&dir)?;
        let mut argv = vec![exe.to_string_lossy().into_owned()];
        argv.extend(args);
        let spec = JobSpec {
            tool: tool.into(),
            argv,
            env: BTreeMap::from([("SPLAT_EFFECT_GPU".to_string(), "1".to_string())]),
            cwd: None,
            on_complete: String::new(),
            meta: BTreeMap::new(),
        };
        fs::write(dir.join("spec.json"), serde_json::to_vec_pretty(&spec)?)?;
        let mut cmd = Command::new(&exe);
        cmd.arg("__effect_remote").arg(&dir).stdin(Stdio::null())
            .stdout(fs::File::create(dir.join("stdout.log"))?)
            .stderr(fs::File::create(dir.join("stderr.log"))?)
            .env("SPLAT_GPU_REMOTE", "fal");
        for (k, v) in [("SPLAT_FAL_TEAM", "product"), ("SPLAT_FAL_LANE", "fx-render"), ("SPLAT_FAL_TIMEOUT_S", "1500")] {
            if std::env::var_os(k).is_none() {
                cmd.env(k, v);
            }
        }
        if std::env::var("FAL_KEY").map_or(true, |k| k.trim().is_empty()) {
            // Read inside the process (the configured key command), handed to the child through its environment only.
            let key = splat_contracts::dev::fal_key().map_err(|e| anyhow::anyhow!("--cloud: {e}"))?;
            cmd.env("FAL_KEY", key);
        }
        eprintln!("[effect] fal job {} (logs {})", tool, dir.display());
        let status = cmd.status().context("starting the fal job")?;
        let cost: Option<Value> = fs::read(dir.join("remote_cost.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
        if !status.success() {
            let tail = fs::read_to_string(dir.join("stderr.log")).unwrap_or_default();
            let tail: String = tail.lines().rev().take(25).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
            bail!("fal job failed (exit {:?}); logs in {}\n{tail}", status.code(), dir.display());
        }
        Ok(json!({"usd": cost.as_ref().and_then(|c| c["usd"].as_f64()), "gpu": cost.as_ref().and_then(|c| c["gpu"].as_str())}))
    }

    /// Move an output file or dir out of the job to `dst` (replacing it).
    pub fn take(&self, rel: &str, dst: &Path) -> Result<()> {
        let src = self.outputs.join(rel);
        ensure!(src.exists(), "fal job produced no {rel}");
        if dst.is_dir() {
            fs::remove_dir_all(dst)?;
        } else if dst.exists() {
            fs::remove_file(dst)?;
        }
        if fs::rename(&src, dst).is_err() {
            if src.is_dir() {
                fs::create_dir_all(dst)?;
                for e in fs::read_dir(&src)? {
                    let e = e?;
                    fs::copy(e.path(), dst.join(e.file_name()))?;
                }
            } else {
                fs::copy(&src, dst)?;
            }
        }
        Ok(())
    }

    pub fn cleanup(self) {
        fs::remove_dir_all(&self.root).ok();
    }
}

/// `splat __effect_remote <job dir>`: the child that waits on the fal request (its prints go to the job logs).
pub fn child(dir: &Path) -> Result<i32> {
    let spec: JobSpec = serde_json::from_slice(&fs::read(dir.join("spec.json"))?)?;
    let code = splat_core::remote::run(dir, &spec)?;
    eprintln!("[effect] remote exit {code}");
    Ok(code)
}
