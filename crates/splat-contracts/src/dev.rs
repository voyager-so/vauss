//! Optional developer-host settings: a shared GPU lock, the fal request log, spend ledger and hard
//! stop, local builds and models. All of it is off unless configured, so a normal install never
//! touches any of it.
//!
//! Each setting comes from the environment, else from `$XDG_CONFIG_HOME/splat/dev.toml` (default
//! `~/.config/splat/dev.toml`):
//!
//! ```toml
//! # SPLAT_DEV_ROOT: gpu.lock, fal-spend.jsonl, runs/fal/, models/, build/release/, tools/ live here.
//! dev_root = "/data/splat"
//! # SPLAT_FAL_STOP_DRIVE: script whose `true-total --json` enforces the fal spend hard stop.
//! fal_stop_drive = "/path/to/drive.py"
//! # SPLAT_FAL_KEY_CMD: command that prints the fal key when FAL_KEY is unset.
//! fal_key_cmd = "my-secrets get FAL_KEY"
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DevConfig {
    pub dev_root: Option<PathBuf>,
    pub fal_stop_drive: Option<PathBuf>,
    pub fal_key_cmd: Option<String>,
}

impl DevConfig {
    /// The file's settings (`file` is dev.toml's text, if any), each overridden by its non-empty
    /// environment variable (`env` reads one).
    pub fn resolve(env: impl Fn(&str) -> Option<String>, file: Option<&str>) -> Result<Self, String> {
        let mut cfg: Self = match file {
            Some(text) => toml::from_str(text).map_err(|e| e.to_string())?,
            None => Self::default(),
        };
        let env = |k: &str| env(k).filter(|v| !v.trim().is_empty());
        if let Some(v) = env("SPLAT_DEV_ROOT") {
            cfg.dev_root = Some(v.into());
        }
        if let Some(v) = env("SPLAT_FAL_STOP_DRIVE") {
            cfg.fal_stop_drive = Some(v.into());
        }
        if let Some(v) = env("SPLAT_FAL_KEY_CMD") {
            cfg.fal_key_cmd = Some(v);
        }
        Ok(cfg)
    }
}

/// `$XDG_CONFIG_HOME/splat/dev.toml`, else `~/.config/splat/dev.toml`.
pub fn config_path() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = var("XDG_CONFIG_HOME").or_else(|| var("HOME").or_else(|| var("USERPROFILE")).map(|h| h.join(".config")))?;
    Some(base.join("splat").join("dev.toml"))
}

/// This process's settings, read once. When a dev root is set, says so on stderr (once), so a
/// missing hard stop is visible.
pub fn config() -> &'static DevConfig {
    static CONFIG: OnceLock<DevConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let env = |k: &str| std::env::var(k).ok();
        let path = config_path();
        let text = path.as_deref().and_then(|p| std::fs::read_to_string(p).ok());
        let cfg = DevConfig::resolve(env, text.as_deref()).unwrap_or_else(|e| {
            eprintln!("splat: ignoring {}: {e}", path.as_deref().unwrap_or(Path::new("dev.toml")).display());
            DevConfig::resolve(env, None).unwrap_or_default()
        });
        if let Some(root) = &cfg.dev_root {
            // Mirrors splat-core's spend ledger + check_budget: with the ledger on, a missing script refuses submits.
            let ledger_off = std::env::var("SPLAT_FAL_SPEND").is_ok_and(|v| v.is_empty() || v == "0");
            let ledger = !ledger_off && (std::env::var_os("SPLAT_FAL_SPEND").is_some() || root.join("runs/fal").is_dir());
            let stop = match &cfg.fal_stop_drive {
                Some(d) if d.is_file() => "on".to_owned(),
                _ if !ledger => "off (no spend ledger)".to_owned(),
                Some(d) => format!("script missing ({}): fal submits are refused", d.display()),
                None => "no fal_stop_drive configured: fal submits are refused".to_owned(),
            };
            eprintln!("splat: dev root: {}, fal hard stop: {stop}", root.display());
        }
        cfg
    })
}

/// The configured dev root, if any.
pub fn dev_root() -> Option<&'static Path> {
    config().dev_root.as_deref()
}

/// The shared GPU lock file: `SPLAT_GPU_LOCK`, else `<dev root>/gpu.lock` when the dev root exists.
/// None (the default): GPU jobs take no lock.
pub fn gpu_lock() -> Option<PathBuf> {
    std::env::var_os("SPLAT_GPU_LOCK")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| dev_root().filter(|r| r.is_dir()).map(|r| r.join("gpu.lock")))
}

/// The fal API key: `FAL_KEY`, else the trimmed stdout of the configured `fal_key_cmd` (run by the
/// shell). The key is never printed.
pub fn fal_key() -> Result<String, String> {
    if let Ok(k) = std::env::var("FAL_KEY")
        && !k.trim().is_empty()
    {
        return Ok(k.trim().to_owned());
    }
    let Some(cmd) = &config().fal_key_cmd else {
        return Err("FAL_KEY is not set (set it to your fal.ai API key)".to_owned());
    };
    let mut shell = if cfg!(windows) { Command::new("cmd") } else { Command::new("sh") };
    shell.arg(if cfg!(windows) { "/C" } else { "-c" }).arg(cmd);
    let out = shell
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("FAL_KEY is not set and fal_key_cmd could not start: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("FAL_KEY is not set and fal_key_cmd failed ({}): {}", out.status, err.trim()));
    }
    let key = String::from_utf8(out.stdout).map_err(|_| "fal_key_cmd printed non-UTF-8 output".to_owned())?;
    let key = key.trim();
    if key.is_empty() {
        return Err("FAL_KEY is not set and fal_key_cmd printed nothing".to_owned());
    }
    Ok(key.to_owned())
}

#[cfg(test)]
mod tests {
    use super::DevConfig;

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| vars.iter().find(|(name, _)| *name == k).map(|(_, v)| (*v).to_owned())
    }

    #[test]
    fn off_without_env_or_file() {
        assert_eq!(DevConfig::resolve(env(&[]), None).unwrap(), DevConfig::default());
    }

    #[test]
    fn file_alone() {
        let file = "dev_root = \"/d\"\nfal_stop_drive = \"/x/drive.py\"\nfal_key_cmd = \"keytool get\"\n";
        let cfg = DevConfig::resolve(env(&[]), Some(file)).unwrap();
        assert_eq!(cfg.dev_root.as_deref(), Some(std::path::Path::new("/d")), "dev_root from file");
        assert_eq!(cfg.fal_stop_drive.as_deref(), Some(std::path::Path::new("/x/drive.py")), "drive from file");
        assert_eq!(cfg.fal_key_cmd.as_deref(), Some("keytool get"), "key command from file");
    }

    #[test]
    fn env_wins_per_key_and_empty_env_is_unset() {
        let file = "dev_root = \"/file\"\nfal_key_cmd = \"from-file\"\n";
        let vars = [("SPLAT_DEV_ROOT", "/env"), ("SPLAT_FAL_KEY_CMD", " "), ("SPLAT_FAL_STOP_DRIVE", "/env/drive.py")];
        let cfg = DevConfig::resolve(env(&vars), Some(file)).unwrap();
        assert_eq!(cfg.dev_root.as_deref(), Some(std::path::Path::new("/env")), "env overrides the file");
        assert_eq!(cfg.fal_key_cmd.as_deref(), Some("from-file"), "blank env keeps the file's value");
        assert_eq!(cfg.fal_stop_drive.as_deref(), Some(std::path::Path::new("/env/drive.py")), "env alone sets a key");
    }

    #[test]
    fn env_alone() {
        let cfg = DevConfig::resolve(env(&[("SPLAT_DEV_ROOT", "/env")]), None).unwrap();
        assert_eq!(cfg.dev_root.as_deref(), Some(std::path::Path::new("/env")), "env without a file");
        assert_eq!(cfg.fal_stop_drive, None, "no drive configured");
    }

    #[test]
    fn bad_file_is_an_error() {
        assert!(DevConfig::resolve(env(&[]), Some("dev_rot = \"/typo\"")).is_err(), "unknown keys are rejected");
        assert!(DevConfig::resolve(env(&[]), Some("dev_root = ")).is_err(), "invalid TOML is rejected");
    }
}
