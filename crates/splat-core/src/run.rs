use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, bail, ensure};
use parking_lot::ReentrantMutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use splat_contracts::*;
use uuid::Uuid;

use crate::{JobRunner, Registry, RunCtx};

pub fn utc_now() -> String {
    chrono::Utc::now().to_rfc3339()
}

pub fn atomic_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    atomic_write(path.as_ref(), &bytes)
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("File has no parent directory")?;
    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        Uuid::new_v4().simple()
    ));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        #[cfg(not(windows))]
        fs::rename(&tmp, path)?;
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            use windows_sys::Win32::Storage::FileSystem::{
                MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
            };
            let src: Vec<u16> = tmp.as_os_str().encode_wide().chain(Some(0)).collect();
            let dst: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
            if unsafe {
                MoveFileExW(
                    src.as_ptr(),
                    dst.as_ptr(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
                )
            } == 0
            {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = fs::remove_file(tmp);
    result
}

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    serde_json::from_reader(File::open(path).with_context(|| format!("Read {}", path.display()))?)
        .with_context(|| format!("Decode {}", path.display()))
}

pub const SUPERVISOR_OWNER: &str = "supervisor.owner";
pub const STOP_REQUEST: &str = "stop_request.json";

/// True when no process holds the run's supervisor lock right now (probe only; nothing is held).
pub fn supervisor_free(root: &Path) -> bool {
    lock_file(&root.join("supervisor.lock"), false).is_ok()
}

pub(crate) fn lock_file(path: &Path, blocking: bool) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    if blocking {
        fs2::FileExt::lock_exclusive(&file)?;
    } else if let Err(error) = fs2::FileExt::try_lock_exclusive(&file) {
        // Windows reports a held lock as ERROR_LOCK_VIOLATION; callers match WouldBlock.
        if error.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
            return Err(std::io::Error::from(std::io::ErrorKind::WouldBlock).into());
        }
        return Err(error.into());
    }
    Ok(file)
}

fn fingerprint(source: &Path) -> Result<String> {
    fn files(path: &Path, list: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            // Do not recurse through directory symlinks (cycles and mutable external trees).
            if ty.is_dir() {
                files(&entry.path(), list)?;
            } else if entry.path().is_file() {
                list.push(entry.path());
            }
        }
        Ok(())
    }
    let mut digest = Sha256::new();
    let mut paths = Vec::new();
    if source.is_dir() {
        files(source, &mut paths)?;
        paths.sort();
    } else {
        paths.push(source.to_path_buf());
    }
    let mut buffer = vec![0; 1024 * 1024];
    for path in paths {
        if source.is_dir() {
            digest.update(path.strip_prefix(source)?.to_string_lossy().as_bytes());
            digest.update([0]);
        }
        let mut stream = File::open(path)?;
        loop {
            let n = stream.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
        }
        digest.update([0]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub schema_version: u32,
    pub run_id: String,
    pub source: PathBuf,
    pub source_sha256: String,
    pub goal: String,
    pub backend: String,
    pub driver: String,
    pub model: Option<String>,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JournalEvent {
    pub schema_version: u32,
    pub seq: u64,
    pub time: String,
    pub run_id: String,
    pub event: String,
    pub data: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunState {
    #[serde(default)]
    pub progress: Value,
    pub seq: u64,
    pub status: String,
    pub artifacts: BTreeMap<String, Artifact>,
    pub jobs: BTreeMap<String, Value>,
    pub operations: BTreeMap<String, Value>,
}

impl Default for RunState {
    fn default() -> Self {
        Self {
            progress: Value::Null,
            seq: 0,
            status: "created".into(),
            artifacts: BTreeMap::new(),
            jobs: BTreeMap::new(),
            operations: BTreeMap::new(),
        }
    }
}

#[derive(Default)]
struct LockState {
    depth: usize,
    file: Option<File>,
    operation_id: Option<String>,
}
struct Inner {
    root: PathBuf,
    config: RunConfig,
    lock: ReentrantMutex<RefCell<LockState>>,
}

#[derive(Clone)]
pub struct Run {
    inner: Arc<Inner>,
    pub(crate) registry: Arc<Registry>,
    pub(crate) worker_exe: PathBuf,
}

impl Run {
    pub fn create(
        root: impl AsRef<Path>,
        source: impl AsRef<Path>,
        goal: impl Into<String>,
        backend: impl Into<String>,
    ) -> Result<Self> {
        Self::create_with_options(root, source, goal, backend, "fixed", None)
    }

    pub fn create_with_options(
        root: impl AsRef<Path>,
        source: impl AsRef<Path>,
        goal: impl Into<String>,
        backend: impl Into<String>,
        driver: impl Into<String>,
        model: Option<String>,
    ) -> Result<Self> {
        let source = fs::canonicalize(source.as_ref()).context("Source does not exist")?;
        fs::create_dir_all(root.as_ref())?;
        let root = fs::canonicalize(root.as_ref())?;
        ensure!(
            !source.is_dir() || !root.starts_with(&source),
            "Run directory cannot be inside the source directory"
        );
        let _lock = lock_file(&root.join(".init.lock"), true)?;
        ensure!(
            !root.join("run.json").exists(),
            "Run already exists: {}",
            root.display()
        );
        let config = RunConfig {
            schema_version: SCHEMA_VERSION,
            run_id: format!("run_{}", &Uuid::new_v4().simple().to_string()[..16]),
            source_sha256: fingerprint(&source)?,
            source,
            goal: goal.into(),
            backend: backend.into(),
            driver: driver.into(),
            model,
            created_at: utc_now(),
        };
        for name in ["artifacts", "jobs", "exports"] {
            fs::create_dir_all(root.join(name))?;
        }
        atomic_json(root.join("run.json"), &config)?;
        let run = Self::load(root, Arc::new(Registry::default()))?;
        run.journal("run_created", json!({"config": config}))?;
        Ok(run)
    }

    fn load(root: PathBuf, registry: Arc<Registry>) -> Result<Self> {
        let config: RunConfig = read_json(root.join("run.json"))?;
        ensure!(
            config.schema_version == SCHEMA_VERSION,
            "Unsupported run schema version"
        );
        Ok(Self {
            inner: Arc::new(Inner {
                root,
                config,
                lock: ReentrantMutex::new(RefCell::new(LockState::default())),
            }),
            registry,
            worker_exe: std::env::current_exe()?,
        })
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_registry(root, Arc::new(Registry::default()), true)
    }

    pub fn open_with_registry(
        root: impl AsRef<Path>,
        registry: Arc<Registry>,
        verify_source: bool,
    ) -> Result<Self> {
        let run = Self::load(fs::canonicalize(root)?, registry)?;
        run.with_lock(|| atomic_json(run.root().join("state.json"), &run.state()?))?;
        if verify_source {
            ensure!(
                run.source().exists() && fingerprint(run.source())? == run.config().source_sha256,
                "Source changed or disappeared since run creation"
            );
        }
        run.jobs().reconcile()?;
        Ok(run)
    }

    pub fn with_registry(mut self, registry: Arc<Registry>) -> Self {
        self.registry = registry;
        self
    }
    /// Executable implementing handle_internal_command; defaults to the current binary.
    pub fn with_worker_exe(mut self, executable: impl Into<PathBuf>) -> Self {
        self.worker_exe = executable.into();
        self
    }
    /// The run backend's capability map (empty when the backend is not registered).
    pub fn backend_capabilities(&self) -> Value {
        self.registry.backend(&self.config().backend).map(|b| b.capabilities()).unwrap_or(Value::Null)
    }
    pub fn config(&self) -> &RunConfig {
        &self.inner.config
    }
    pub fn jobs(&self) -> JobRunner<'_> {
        JobRunner::new(self)
    }

    pub(crate) fn with_lock<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let guard = self.inner.lock.lock();
        {
            let mut state = guard.borrow_mut();
            if state.depth == 0 {
                state.file = Some(lock_file(&self.root().join("supervisor.lock"), true)?);
                // Lets `vauss stop` find (and if needed signal) the process holding the lock.
                let _ = fs::write(self.root().join(SUPERVISOR_OWNER), std::process::id().to_string());
            }
            state.depth += 1;
        }
        struct Release<'a>(&'a RefCell<LockState>);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                let mut s = self.0.borrow_mut();
                s.depth -= 1;
                if s.depth == 0 {
                    s.file.take();
                }
            }
        }
        let _release = Release(&guard);
        f()
    }

    /// Best-effort observers must never wait behind a synchronous tool or delay cancellation.
    pub(crate) fn try_with_lock<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<Option<T>> {
        let Some(guard) = self.inner.lock.try_lock() else { return Ok(None) };
        if guard.borrow().depth > 0 { return f().map(Some); }
        let file = match lock_file(&self.root().join("supervisor.lock"), false) {
            Ok(file) => file,
            Err(e) if e.downcast_ref::<std::io::Error>().is_some_and(|e| e.kind() == std::io::ErrorKind::WouldBlock) => return Ok(None),
            Err(e) => return Err(e),
        };
        guard.borrow_mut().file = Some(file);
        guard.borrow_mut().depth = 1;
        struct Release<'a>(&'a RefCell<LockState>);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                let mut state = self.0.borrow_mut();
                state.depth = 0;
                state.file.take();
            }
        }
        let _release = Release(&guard);
        f().map(Some)
    }

    pub(crate) fn operation_id(&self) -> Option<String> {
        self.inner.lock.lock().borrow().operation_id.clone()
    }
    pub(crate) fn in_operation<T>(&self, id: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.with_lock(|| {
            self.inner.lock.lock().borrow_mut().operation_id = Some(id.into());
            struct Clear<'a>(&'a Run);
            impl Drop for Clear<'_> {
                fn drop(&mut self) {
                    self.0.inner.lock.lock().borrow_mut().operation_id = None;
                }
            }
            let _clear = Clear(self);
            f()
        })
    }

    fn read_events(&self, repair: bool) -> Result<Vec<JournalEvent>> {
        let path = self.root().join("journal.jsonl");
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        let mut events: Vec<JournalEvent> = Vec::new();
        let mut offset = 0;
        for line in bytes.split_inclusive(|b| *b == b'\n') {
            if !line.ends_with(b"\n") && !repair {
                // A lock-free reader can see a record mid-append; it is not yet part of the journal.
                break;
            }
            if !line.ends_with(b"\n") && repair {
                let file = OpenOptions::new().write(true).open(&path)?;
                file.set_len(offset as u64)?;
                file.sync_all()?;
                self.append_event(
                    &mut events,
                    "journal_recovered",
                    json!({"discarded_bytes": line.len()}),
                )?;
                return Ok(events);
            }
            let event: JournalEvent = serde_json::from_slice(line)
                .with_context(|| format!("Corrupt journal at record {}", events.len() + 1))?;
            ensure!(
                line.ends_with(b"\n")
                    && event.seq == events.len() as u64 + 1
                    && event.run_id == self.run_id()
                    && event.schema_version == SCHEMA_VERSION,
                "Corrupt journal at record {}",
                events.len() + 1
            );
            events.push(event);
            offset += line.len();
        }
        Ok(events)
    }

    pub fn events(&self) -> Result<Vec<JournalEvent>> {
        self.with_lock(|| self.read_events(true))
    }
    /// Opens a run for reading without the supervisor lock (no state.json write, no reconcile), so
    /// `status` answers while a tool holds the lock.
    pub fn open_unlocked(root: impl AsRef<Path>) -> Result<Self> {
        Self::load(fs::canonicalize(root)?, Arc::new(Registry::default()))
    }
    /// State from the append-only journal without taking the supervisor lock.
    pub fn state_unlocked(&self) -> Result<RunState> {
        Self::project(&self.read_events(false)?)
    }
    pub fn state(&self) -> Result<RunState> {
        self.with_lock(|| Self::project(&self.read_events(true)?))
    }

    fn project(events: &[JournalEvent]) -> Result<RunState> {
        let mut state = RunState::default();
        for event in events {
            state.seq = event.seq;
            let d = &event.data;
            let key = |name: &str| -> Result<String> {
                Ok(d[name]
                    .as_str()
                    .with_context(|| format!("Corrupt journal: {} requires {name}", event.event))?
                    .into())
            };
            match event.event.as_str() {
                "progress" => state.progress = d.clone(),
                "run_started" => {
                    state.progress = Value::Null;
                    state.status = d["status"].as_str().unwrap_or("running").into();
                }
                "artifact_registered" => {
                    let a: Artifact = serde_json::from_value(d["artifact"].clone())?;
                    state.artifacts.insert(a.id.clone(), a);
                }
                "operation_planned" => {
                    state.operations.insert(key("operation_id")?, d.clone());
                }
                "tool_result" => {
                    state
                        .operations
                        .entry(key("operation_id")?)
                        .or_insert(json!({}))["result"] = d["result"].clone();
                }
                "job_planned" => {
                    let mut job = d.clone();
                    job["status"] = json!("running");
                    state.jobs.insert(key("job_id")?, job);
                }
                "job_completed" => {
                    let job = state.jobs.entry(key("job_id")?).or_insert(json!({}));
                    job["status"] = d["result"]["status"].clone();
                    job["result"] = d["result"].clone();
                    if let Some(op) = job["operation_id"]
                        .as_str()
                        .and_then(|id| state.operations.get_mut(id))
                    {
                        op["result"] = d["result"].clone();
                    }
                }
                // A stopped job the pipeline stopped waiting for: no longer counts as running (finish).
                "job_abandoned" => {
                    let job = state.jobs.entry(key("job_id")?).or_insert(json!({}));
                    if job["status"] == "running" {
                        job["status"] = json!("abandoned");
                    }
                }
                "run_finished" | "run_stopped" => {
                    state.status = d["status"].as_str().unwrap_or("running").into()
                }
                _ => {}
            }
        }
        Ok(state)
    }

    fn append_event(&self, events: &mut Vec<JournalEvent>, event: &str, data: Value) -> Result<()> {
        let record = JournalEvent {
            schema_version: SCHEMA_VERSION,
            seq: events.len() as u64 + 1,
            time: utc_now(),
            run_id: self.run_id().into(),
            event: event.into(),
            data,
        };
        let mut bytes = serde_json::to_vec(&record)?;
        bytes.push(b'\n');
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(self.root().join("journal.jsonl"))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        events.push(record);
        atomic_json(self.root().join("state.json"), &Self::project(events)?)
    }
}

impl RunCtx for Run {
    fn root(&self) -> &Path {
        &self.inner.root
    }
    fn run_id(&self) -> &str {
        &self.inner.config.run_id
    }
    fn source(&self) -> &Path {
        &self.inner.config.source
    }
    fn goal(&self) -> &str {
        &self.inner.config.goal
    }
    fn new_artifact_dir(&self, kind: ArtifactKind) -> Result<(String, PathBuf)> {
        let kind = serde_json::to_value(kind)?;
        let id = format!(
            "{}_{}",
            kind.as_str().unwrap(),
            &Uuid::new_v4().simple().to_string()[..12]
        );
        let path = self.root().join("artifacts").join(&id);
        fs::create_dir(&path)?;
        Ok((id, path))
    }
    fn register(&self, artifact: Artifact) -> Result<()> {
        ensure!(!artifact.id.is_empty(), "Artifact ID must not be empty");
        let relative = Path::new(&artifact.path);
        ensure!(
            !relative.is_absolute()
                && !relative
                    .components()
                    .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_))),
            "Artifact path must stay within the run root"
        );
        let path =
            fs::canonicalize(self.root().join(relative)).context("Artifact path does not exist")?;
        ensure!(
            path.starts_with(self.root()),
            "Artifact path must stay within the run root"
        );
        self.with_lock(|| {
            if let Some(previous) = self.state()?.artifacts.get(&artifact.id) {
                if serde_json::to_value(previous)? == serde_json::to_value(&artifact)? {
                    return Ok(());
                }
                bail!("Artifact is immutable: {}", artifact.id);
            }
            self.journal("artifact_registered", json!({"artifact": artifact}))
        })
    }
    fn artifact(&self, id: &str) -> Result<Artifact> {
        self.state()?
            .artifacts
            .remove(id)
            .with_context(|| format!("Unknown artifact: {id}"))
    }
    fn path(&self, id: &str) -> Result<PathBuf> {
        let path = fs::canonicalize(self.root().join(self.artifact(id)?.path))?;
        ensure!(
            path.starts_with(self.root()),
            "Artifact path escaped the run root"
        );
        Ok(path)
    }
    fn journal(&self, event: &str, data: Value) -> Result<()> {
        self.with_lock(|| self.append_event(&mut self.read_events(true)?, event, data))
    }
    fn start_job(&self, spec: JobSpec) -> Result<String> {
        self.jobs().start(spec)
    }
}
