//! SOG adapter to MIT @playcanvas/splat-transform 3.10.0.
//! Node >= 22 and the pinned package are runtime dependencies, never downloaded implicitly.
use std::path::Path;
use std::process::{Command, Stdio};
use anyhow::{Context, Result, ensure};
use crate::ply::{self, Gaussians};

pub const ENCODER_VERSION: &str = "3.10.0";

pub fn write_sog(g: &Gaussians, path: &Path, encoder: &Path, gpu: Option<u32>) -> Result<u64> {
    ensure!(!g.is_empty(), "SOG requires at least one splat");
    ensure!(g.sh_degree <= 3, "SOG adapter supports SH degrees 0..3");
    let output = Command::new(encoder).arg("--version").output().with_context(|| format!("starting {}: install @playcanvas/splat-transform@{ENCODER_VERSION} with Node >= 22, or pass --sog-encoder", encoder.display()))?;
    let version = format!("{} {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    ensure!(output.status.success() && version.split_whitespace().any(|s| s.trim_start_matches('v') == ENCODER_VERSION),
        "SOG requires splat-transform {ENCODER_VERSION}; got {} ({}): {}", version.trim(), output.status, String::from_utf8_lossy(&output.stderr));
    // Keep the temporary PLY private and remove it on success or failure.
    let tmp = tempfile::Builder::new().prefix("splat-sog-").tempdir_in(path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")))?;
    let source = tmp.path().join("source.ply");
    ply::write_ply(g, &source)?;
    let destination = tmp.path().join("scene.sog");
    let status = Command::new(encoder)
        .args(["--gpu", &gpu.map(|v| v.to_string()).unwrap_or_else(|| "cpu".into()), "--sh-iterations", "10"])
        .arg(&source).arg(&destination)
        .stdout(Stdio::null()).stderr(Stdio::inherit()).status().context("running SOG encoder")?;
    ensure!(status.success(), "SOG encoder exited {status}");
    ensure!(destination.is_file(), "SOG encoder produced no archive");
    std::fs::rename(&destination, path).context("installing encoded SOG")?;
    Ok(std::fs::metadata(path)?.len())
}
