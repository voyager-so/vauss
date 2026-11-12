//! sha256 of big local files (the splat binary, model weights, a capture video, big run outputs), cached across
//! processes in `<cache dir>/sha256-cache.json` keyed by the file's identity: device and inode, size and mtime (ns).
//! Every `--cloud` job hashed every big external file again: ~600 MB per SfM job (splat binary 145 MB, DA3 weights
//! 183 MB, ALIKED 46 MB, ffmpeg/ffprobe 63 MB, ...), read cold from disk on a client held to one core. A file
//! replaced by rename has a new inode, and one written in place a new mtime, so a hit is the same bytes. The run's
//! staged copies are hardlinks of its files (same inode), so a later job finds an earlier job's hashes. A miss (or
//! any cache error) hashes the file as before; the cache never changes what is hashed, only how often.
use anyhow::Result;
use serde_json::{Value, json};
use std::{fs, io::Read, path::Path, path::PathBuf};

/// Smaller files are hashed directly: reading them costs less than the cache's own I/O.
const MIN_BYTES: u64 = 4 << 20;
/// Entries kept (the oldest hashed dropped first).
const MAX_ENTRIES: usize = 512;

fn cache_file() -> Option<PathBuf> {
    let dir = if let Some(d) = std::env::var_os("SPLAT_CACHE_DIR").filter(|v| !v.is_empty()) {
        PathBuf::from(d)
    } else if cfg!(windows) {
        PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("splat/cache")
    } else if let Some(d) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(d).join("splat")
    } else {
        PathBuf::from(std::env::var_os("HOME")?).join(if cfg!(target_os = "macos") { "Library/Caches/splat" } else { ".cache/splat" })
    };
    Some(dir.join("sha256-cache.json"))
}

/// The file's identity: (key, size, mtime ns). Unix: device and inode (hardlinks share it); elsewhere the canonical path.
fn identity(path: &Path, meta: &fs::Metadata) -> Option<(String, u64, u128)> {
    let mtime = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
    #[cfg(unix)]
    let key = { use std::os::unix::fs::MetadataExt; format!("{}:{}", meta.dev(), meta.ino()) };
    #[cfg(not(unix))]
    let key = path.canonicalize().ok()?.to_string_lossy().into_owned();
    let _ = path;
    Some((key, meta.len(), mtime))
}

fn hash(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn now_s() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// sha256 (hex) of the file at `path`, from the cache when its identity matches.
pub fn sha256(path: &Path) -> Result<String> {
    sha256_with(path, cache_file())
}

fn sha256_with(path: &Path, file: Option<PathBuf>) -> Result<String> {
    let meta = fs::metadata(path)?;
    if meta.len() < MIN_BYTES {
        return hash(path);
    }
    let (Some(file), Some((key, size, mtime))) = (file, identity(path, &meta)) else { return hash(path) };
    let read = || -> Option<Value> {
        let mut text = String::new();
        fs::File::open(&file).ok()?.read_to_string(&mut text).ok()?;
        serde_json::from_str(&text).ok()
    };
    let mut all = read().filter(Value::is_object).unwrap_or_else(|| json!({}));
    let e = &all[key.as_str()];
    if e["size"].as_u64() == Some(size) && e["mtime_ns"].as_str() == Some(mtime.to_string().as_str()) {
        if let Some(sha) = e["sha256"].as_str() {
            return Ok(sha.to_string());
        }
    }
    let sha = hash(path)?;
    // Best effort, last writer wins: a lost entry is only hashed again.
    let map = all.as_object_mut().expect("object");
    map.insert(key, json!({"size": size, "mtime_ns": mtime.to_string(), "sha256": sha, "path": path, "used": now_s()}));
    if map.len() > MAX_ENTRIES {
        let mut by_use: Vec<(u64, String)> = map.iter().map(|(k, v)| (v["used"].as_u64().unwrap_or(0), k.clone())).collect();
        by_use.sort();
        for (_, k) in by_use.into_iter().take(map.len() - MAX_ENTRIES) {
            map.remove(&k);
        }
    }
    let write = || -> Result<()> {
        fs::create_dir_all(file.parent().unwrap_or(Path::new(".")))?;
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let tmp = file.with_file_name(format!(".sha256-cache.{}.{nanos}.tmp", std::process::id()));
        fs::write(&tmp, serde_json::to_vec(&all)?)?;
        fs::rename(&tmp, &file).inspect_err(|_| drop(fs::remove_file(&tmp)))?;
        Ok(())
    };
    let _ = write();
    Ok(sha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hit_needs_the_same_file_size_and_mtime() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("splat-hashcache-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let cache = Some(dir.join("sha256-cache.json"));
        let f = dir.join("big.bin");
        fs::write(&f, vec![1u8; MIN_BYTES as usize])?;
        let first = sha256_with(&f, cache.clone())?;
        assert_eq!(first, hash(&f)?);
        assert!(fs::read_to_string(dir.join("sha256-cache.json"))?.contains(&first));
        assert_eq!(sha256_with(&f, cache.clone())?, first);
        // A hardlink is the same file (a job's staged copy of a run file).
        fs::hard_link(&f, dir.join("link.bin"))?;
        assert_eq!(sha256_with(&dir.join("link.bin"), cache.clone())?, first);
        // Same size, new content: a new mtime (or inode after a rename) misses.
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&f, vec![2u8; MIN_BYTES as usize])?;
        assert_eq!(sha256_with(&f, cache.clone())?, hash(&f)?);
        assert_ne!(sha256_with(&f, cache)?, first);
        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
