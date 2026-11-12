//! Driver pipeline cache persisted across processes (splat patch).
//!
//! Every new process otherwise pays the driver's SPIR-V -> machine code
//! compile for each compute pipeline (`vkCreateComputePipelines`, ~4 s per
//! cold training run on an L40S). Where the device supports
//! [`wgpu::Features::PIPELINE_CACHE`] (Vulkan only), one [`wgpu::PipelineCache`]
//! is loaded from disk at server creation, handed to every compute pipeline,
//! and written back in the background.
//!
//! - File: `<dir>/<wgpu::util::pipeline_cache_key>.bin`, where `<dir>` is
//!   `$SPLAT_PIPELINE_CACHE_DIR` or else `<user cache dir>/splat/wgpu-pipeline-cache`.
//!   The key names the vendor/device; wgpu additionally validates the driver
//!   and wgpu version in the data header, and `fallback: true` makes stale or
//!   corrupt data fall back to an empty cache.
//! - Saving never runs on the server thread and never relies on process exit:
//!   a saver thread writes the cache once pipeline creation has been quiet for
//!   [`QUIET`] (or at most every [`MAX_DELAY`] while a burst continues), and
//!   once more when the server is dropped. Writes are atomic (temp file +
//!   rename), so a crash mid-save or concurrent processes cannot leave a torn
//!   file. I/O errors are logged once at debug level and otherwise ignored.
//! - The file is framed (magic, length, FNV-1a checksum of the driver data);
//!   anything that does not verify is never handed to the driver: the run
//!   starts cold instead.
//! - What each run got is logged at info level: bytes loaded (or why the file
//!   was rejected) and every save with the number of driver compiles so far,
//!   so result rows can record the cache state.

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// Save once no new pipeline was created for this long. Short on purpose: the
/// eval/export burst at the end of training is followed within a second by
/// process exit, which does not drop the server, so a longer debounce lost
/// those pipelines on every run (measured: ~0.5 s of a seeded truck run).
const QUIET: Duration = Duration::from_millis(250);
/// Save at least this often while pipelines keep being created. A save copies
/// the whole cache (a few MB) on this thread, never the server's.
const MAX_DELAY: Duration = Duration::from_secs(2);
/// A pipeline creation slower than this was a driver compile (cache miss;
/// ~13 ms each on an L40S); faster ones were served from the cache. Only
/// misses make the cache worth saving, so a warm run never wakes the saver
/// and never copies the cache (`get_data` reads the whole cache, and the
/// driver may serialise it against pipeline creation on the server thread).
const COMPILE_THRESHOLD: Duration = Duration::from_millis(2);

/// File header: magic, then data length and FNV-1a 64 of the data (both LE).
const MAGIC: &[u8; 8] = b"SPLATPC1";
const HEADER: usize = 24;

#[derive(Debug)]
pub(crate) struct PersistentPipelineCache {
    cache: wgpu::PipelineCache,
    /// Driver compiles (cache misses) this process, for the save log.
    compiles: Arc<AtomicU64>,
    /// Signals "a pipeline was created"; dropping it asks the saver for a
    /// final save and stops it.
    notify: Option<mpsc::Sender<()>>,
    saver: Option<JoinHandle<()>>,
}

impl PersistentPipelineCache {
    /// Returns `None` (pipelines are then created without a cache, as before)
    /// when the device lacks [`wgpu::Features::PIPELINE_CACHE`] or no cache
    /// location can be determined.
    pub(crate) fn new(device: &wgpu::Device) -> Option<Self> {
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return None;
        }
        let key = wgpu::util::pipeline_cache_key(&device.adapter_info())?;
        let path = cache_dir()?.join(format!("{key}.bin"));
        let file = std::fs::read(&path).ok();
        let data = match file.as_deref().map(unframe) {
            None => {
                log::info!("[cubecl-wgpu] pipeline cache {}: none, cold start", path.display());
                None
            }
            Some(Ok(data)) => {
                log::info!(
                    "[cubecl-wgpu] pipeline cache {}: loaded {} bytes",
                    path.display(),
                    data.len()
                );
                Some(data.to_vec())
            }
            Some(Err(why)) => {
                log::info!(
                    "[cubecl-wgpu] pipeline cache {}: rejected ({why}), cold start",
                    path.display()
                );
                None
            }
        };
        // SAFETY: the data is only ever produced by `PipelineCache::get_data`
        // for a device with the same key; wgpu validates its header (adapter,
        // driver and wgpu version) and `fallback: true` turns anything
        // unusable into an empty cache instead of an error.
        let cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("splat persistent pipeline cache"),
                data: data.as_deref(),
                fallback: true,
            })
        };
        log::debug!(
            "[cubecl-wgpu] pipeline cache {} ({} bytes loaded)",
            path.display(),
            data.as_ref().map_or(0, Vec::len)
        );
        let (notify, rx) = mpsc::channel();
        let saver_cache = cache.clone();
        let loaded_hash = data.as_deref().map(fnv1a);
        let compiles = Arc::new(AtomicU64::new(0));
        let saver_compiles = compiles.clone();
        let saver = std::thread::Builder::new()
            .name("wgpu-pipeline-cache".into())
            .spawn(move || saver_loop(rx, saver_cache, path, loaded_hash, saver_compiles))
            .ok()?;
        Some(Self {
            cache,
            compiles,
            notify: Some(notify),
            saver: Some(saver),
        })
    }

    pub(crate) fn cache(&self) -> &wgpu::PipelineCache {
        &self.cache
    }

    /// Call after each pipeline created with [`Self::cache`], with how long
    /// the creation took; only driver compiles schedule a save.
    pub(crate) fn pipeline_created(&self, took: Duration) {
        if took < COMPILE_THRESHOLD {
            return;
        }
        self.compiles.fetch_add(1, Ordering::Relaxed);
        if let Some(notify) = &self.notify {
            let _ = notify.send(());
        }
    }
}

impl Drop for PersistentPipelineCache {
    fn drop(&mut self) {
        // Best effort: closing the channel makes the saver write any unsaved
        // pipelines and exit.
        self.notify.take();
        if let Some(saver) = self.saver.take() {
            let _ = saver.join();
        }
    }
}

fn saver_loop(
    rx: mpsc::Receiver<()>,
    cache: wgpu::PipelineCache,
    path: PathBuf,
    mut saved_hash: Option<u64>,
    compiles: Arc<AtomicU64>,
) {
    let mut logged_error = false;
    // Block until the first pipeline of a burst (or shutdown).
    while rx.recv().is_ok() {
        let burst_start = Instant::now();
        let mut open = true;
        loop {
            let left = MAX_DELAY.saturating_sub(burst_start.elapsed());
            if left.is_zero() {
                break;
            }
            match rx.recv_timeout(QUIET.min(left)) {
                Ok(()) => continue,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    open = false;
                    break;
                }
            }
        }
        save(&cache, &path, &mut saved_hash, &mut logged_error, &compiles);
        if !open {
            return;
        }
    }
}

fn save(
    cache: &wgpu::PipelineCache,
    path: &Path,
    saved_hash: &mut Option<u64>,
    logged: &mut bool,
    compiles: &AtomicU64,
) {
    let Some(data) = cache.get_data() else {
        return;
    };
    let hash = fnv1a(&data);
    // A warm run re-creates only cached pipelines: nothing new to write.
    if *saved_hash == Some(hash) {
        return;
    }
    match write_atomic(path, &frame(&data, hash)) {
        Ok(()) => {
            *saved_hash = Some(hash);
            log::info!(
                "[cubecl-wgpu] pipeline cache {}: saved {} bytes after {} driver compiles",
                path.display(),
                data.len(),
                compiles.load(Ordering::Relaxed)
            );
        }
        Err(err) if !*logged => {
            *logged = true;
            log::debug!(
                "[cubecl-wgpu] could not save pipeline cache {}: {err}",
                path.display()
            );
        }
        Err(_) => {}
    }
}

fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Unique per process so concurrent runs never share a temp file.
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// FNV-1a 64: stable across builds, unlike `DefaultHasher`.
fn fnv1a(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn frame(data: &[u8], hash: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + data.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(data);
    out
}

/// The driver data inside a framed file, or why it is unusable.
fn unframe(file: &[u8]) -> Result<&[u8], &'static str> {
    if file.len() < HEADER || &file[..8] != MAGIC {
        return Err("bad header");
    }
    let len = u64::from_le_bytes(file[8..16].try_into().unwrap());
    let hash = u64::from_le_bytes(file[16..24].try_into().unwrap());
    let data = &file[HEADER..];
    if data.len() as u64 != len {
        return Err("truncated");
    }
    if fnv1a(data) != hash {
        return Err("checksum mismatch");
    }
    Ok(data)
}

/// `$SPLAT_PIPELINE_CACHE_DIR`, else the platform user cache directory
/// (`$XDG_CACHE_HOME` or `~/.cache` on Linux, `~/Library/Caches` on macOS,
/// `%LOCALAPPDATA%` on Windows) plus `splat/wgpu-pipeline-cache`.
fn cache_dir() -> Option<PathBuf> {
    let env = |name: &str| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(dir) = env("SPLAT_PIPELINE_CACHE_DIR") {
        return Some(dir);
    }
    let base = if cfg!(windows) {
        env("LOCALAPPDATA")?
    } else if cfg!(target_os = "macos") {
        env("HOME")?.join("Library/Caches")
    } else {
        env("XDG_CACHE_HOME").or_else(|| Some(env("HOME")?.join(".cache")))?
    };
    Some(base.join("splat").join("wgpu-pipeline-cache"))
}
