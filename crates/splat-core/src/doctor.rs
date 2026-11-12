use anyhow::Result;
use serde_json::{Value, json};
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

pub fn find_binary(name: &str, env_key: &str) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(env_key)
        .map(PathBuf::from)
        .filter(|p| p.is_file())
    {
        return Some(path);
    }
    let executable = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.into()
    };
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(&executable))
            .find(|p| p.is_file())
    })
}
fn capture(program: &Path, args: &[&str]) -> Value {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    match command.spawn() {
        Err(error) => json!({"path":program,"error":error.to_string()}),
        Ok(mut child) => {
            let mut stdout = child.stdout.take().unwrap();
            let mut stderr = child.stderr.take().unwrap();
            let out = thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = stdout.read_to_end(&mut bytes);
                bytes
            });
            let err = thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = stderr.read_to_end(&mut bytes);
                bytes
            });
            let deadline = Instant::now() + Duration::from_secs(10);
            let (code, timed_out) = loop {
                match child.try_wait() {
                    Ok(Some(code)) => break (code.code(), false),
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(50))
                    }
                    _ => {
                        let _ = child.kill();
                        let code = child.wait().ok().and_then(|c| c.code());
                        break (code, true);
                    }
                }
            };
            let stdout = out.join().unwrap_or_default();
            let stderr = err.join().unwrap_or_default();
            json!({"path":program,"argv":args,"exit_code":code,"timed_out":timed_out,"stdout":String::from_utf8_lossy(&stdout[..stdout.len().min(32768)]),"stderr":String::from_utf8_lossy(&stderr[..stderr.len().min(32768)])})
        }
    }
}
/// Pins exactly which build a run used: SHA-256 plus the output of each probe command (e.g. subcommand help).
pub fn binary_fingerprint(path: &Path, probes: &[&[&str]]) -> Value {
    use sha2::{Digest, Sha256};
    let sha256 = std::fs::File::open(path).and_then(|mut f| {
        let mut digest = Sha256::new();
        std::io::copy(&mut f, &mut digest)?;
        Ok(format!("{:x}", digest.finalize()))
    });
    let bytes = std::fs::metadata(path).map(|m| m.len()).ok();
    let help = probes.iter().map(|args| capture(path, args)).collect::<Vec<_>>();
    match sha256 {
        Ok(sha256) => json!({"path":path,"present":true,"sha256":sha256,"bytes":bytes,"help":help}),
        Err(error) => json!({"path":path,"present":false,"error":error.to_string()}),
    }
}
/// Offline inventory. Dependency downloads are explicit CLI actions or first-use backend work.
pub fn doctor(backend: &str, output_dir: Option<&Path>) -> Result<Value> {
    let mut disk_path = output_dir
        .map(Path::to_path_buf)
        .unwrap_or(std::env::current_dir()?);
    while !disk_path.exists() {
        if !disk_path.pop() {
            disk_path = std::env::current_dir()?;
            break;
        }
    }
    let mut binaries = serde_json::Map::new();
    for (name, key, args) in [
        ("ffmpeg", "FFMPEG_BIN", vec!["-version"]),
        ("ffprobe", "FFPROBE_BIN", vec!["-version"]),
        ("colmap", "COLMAP_BIN", vec!["-h"]),
        ("spirula", "SPIRULA_BIN", vec!["--help"]),
        ("claude", "CLAUDE_BIN", vec!["--version"]),
        ("codex", "CODEX_BIN", vec!["--version"]),
    ] {
        let path = find_binary(name, key).or_else(|| {
            let p = splat_contracts::dev::dev_root()?.join("build/release/spirula");
            (name == "spirula" && p.is_file()).then_some(p)
        });
        binaries.insert(
            name.into(),
            path.as_ref()
                .map(|p| capture(p, &args))
                .unwrap_or(json!({"present":false,"configure_env":key})),
        );
    }
    let gpu = if let Some(nvidia) = find_binary("nvidia-smi", "NVIDIA_SMI_BIN") {
        if let Some(lock) = splat_contracts::dev::gpu_lock().filter(|_| cfg!(unix)) {
            match crate::run::lock_file(&lock, false) {
                Ok(_lock) => capture(
                    &nvidia,
                    &[
                        "--query-gpu=name,driver_version,memory.total,memory.free",
                        "--format=csv,noheader",
                    ],
                ),
                Err(error)
                    if error
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|e| e.kind() == std::io::ErrorKind::WouldBlock) =>
                {
                    json!({"busy":true,"note":"GPU job owns the device lock; inventory skipped"})
                }
                Err(error) => return Err(error),
            }
        } else {
            capture(
                &nvidia,
                &[
                    "--query-gpu=name,driver_version,memory.total,memory.free",
                    "--format=csv,noheader",
                ],
            )
        }
    } else {
        json!({"nvidia_smi":false,"note":"Brush uses wgpu; NVIDIA tooling is optional on Apple/AMD/Intel GPUs"})
    };
    let report = json!({"time":crate::utc_now(),"backend":backend,"platform":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH},"disk":{"path":disk_path,"free_bytes":fs2::available_space(&disk_path)?,"total_bytes":fs2::total_space(&disk_path)?},"gpu":gpu,"binaries":binaries});
    if let Some(dir) = output_dir {
        std::fs::create_dir_all(dir)?;
        crate::atomic_json(dir.join("capabilities.json"), &report)?;
    }
    Ok(report)
}
