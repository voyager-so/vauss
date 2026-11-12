//! Native fal client for `vauss run --cloud`: what `fal/drive.py exec` does, without Python.
//! Stages inputs privately on the fal account's /data volume through the app's `/stage` endpoint
//! (chunked, content-addressed by sha256, under a scope derived from the caller's key; nothing goes to
//! fal's publicly readable storage), submits an ExecJob to the product-tier
//! `splat-gpu-<gpu>-product-<version>` app through fal's queue API, polls it, and brings back
//! stdout/stderr and the new/changed files of the sync-back inputs. The runner deletes the staged
//! inputs when the job ends (and sweeps leftovers after 24 h); a large result is read back through
//! `/stage` and deleted. Every request sets `X-Fal-Store-IO: 0` so fal keeps no copy of payloads.
//!
//! Auth is `FAL_KEY` from the environment. The key is never printed.
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use std::sync::Arc;

/// The fal account that hosts the runner apps; SPLAT_FAL_OWNER points `--cloud` at your own deployment.
fn owner() -> String {
    std::env::var("SPLAT_FAL_OWNER").ok().map(|o| o.trim().to_string()).filter(|o| !o.is_empty()).unwrap_or_else(|| "Nullframe".into())
}
const QUEUE: &str = "https://queue.fal.run";
/// fal's synchronous endpoint: raw uploads go straight to a stage runner (no queue, no base64).
const SYNC: &str = "https://fal.run";
/// Raw bytes per `/stage` request (base64 in JSON); halved on HTTP 413.
const STAGE_CHUNK: usize = 16 << 20;
/// The runner apps' request protocol (app.py MIN_CLIENT_PROTO).
const CLIENT_PROTO: u32 = 3;
/// COLMAP for Brush SfM, provisioned by the runner from this conda lock (as drive.py --colmap does).
const COLMAP_LOCK: &str = include_str!("../assets/colmap-linux-explicit.txt");

/// $/s while a runner works on a job (drive.py GPUS).
fn price_per_s(gpu: &str) -> f64 {
    match gpu {
        "rtxpro6000" => 0.0008,
        _ => 0.000575,
    }
}

pub struct Input {
    pub name: String,
    pub path: PathBuf,
    /// New/changed files under this input come back into `path`.
    pub sync_back: bool,
}

pub struct Exec {
    pub gpu: String,
    /// Local observer snapshot; never sent to fal.
    pub progress_path: PathBuf,
    pub argv: Vec<String>,
    pub inputs: Vec<Input>,
    pub env: BTreeMap<String, String>,
    pub timeout_s: f64,
    pub colmap: bool,
    /// The run's staging ledger: with one, staged inputs (and the job's sync-back outputs) stay on
    /// fal /data for the run's later jobs until `release`; without, the runner deletes them after the job.
    pub ledger: Option<PathBuf>,
    /// Dev bookkeeping of each request (registry link, submitted.jsonl rows); never fails the job.
    pub track: Track,
    /// Created once the inputs are on fal (the pipeline holds a competing upload until then).
    pub staged_marker: Option<PathBuf>,
    /// File names or artifact-relative directory prefixes (ending in /) left on fal with a ledger: a local
    /// stub stands in, see [`remote_stub`]. Empty: every result comes back (`SPLAT_CLOUD_FETCH_ALL=1`).
    pub keep_remote: Vec<String>,
}

/// File name of [`Exec::staged_marker`] in a job's dir.
pub const INPUTS_STAGED: &str = "fal-inputs-staged";

/// What links a native request to the dev host's bookkeeping: submit/end rows in the request log
/// ([`submitted_log`]) and the spend ledger. All best effort.
#[derive(Default)]
pub struct Track {
    pub team: String,
    pub lane: String,
    /// The job (`splat-cloud/<job id>`).
    pub tag: String,
    /// The run's root dir, recorded in the request log.
    pub run_root: Option<PathBuf>,
}

/// The dev host's request log (`<dev root>/runs/fal/submitted.jsonl`, see [`splat_contracts::dev`]).
/// Written only where its directory already exists, so user machines get nothing; `SPLAT_FAL_SUBMITTED`
/// overrides the path (empty or 0: off).
fn submitted_log() -> Option<PathBuf> {
    match std::env::var("SPLAT_FAL_SUBMITTED") {
        Ok(v) if v.is_empty() || v == "0" => None,
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => Some(splat_contracts::dev::dev_root()?.join("runs/fal/submitted.jsonl")).filter(|p| p.parent().is_some_and(Path::is_dir)),
    }
}

/// One line appended under fal/jsonl_log.py's shared lock (`<dir>/.locks/<name>.lock`; rotation takes it exclusive).
fn append_row(path: &Path, row: &Value) -> Result<()> {
    use std::io::Write;
    let dir = path.parent().context("no parent")?;
    let locks = dir.join(".locks");
    fs::create_dir_all(&locks)?;
    let lock = fs::OpenOptions::new().create(true).append(true)
        .open(locks.join(format!("{}.lock", path.file_name().context("no name")?.to_string_lossy())))?;
    fs2::FileExt::lock_shared(&lock)?;
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(format!("{row}\n").as_bytes())?;
    Ok(())
}

/// One row of runs/fal/submitted.jsonl.
fn reqlog(event: &str, mut row: Value) {
    let Some(path) = submitted_log() else { return };
    row["ts"] = json!(chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false));
    row["unix"] = json!(now());
    row["event"] = json!(event);
    row["pid"] = json!(std::process::id());
    // fal/recover.py skips these: results come back through this process only, it has nothing to resume.
    row["client"] = json!("splat-native");
    if let Err(e) = append_row(&path, &row) {
        println!("[fal] could not log {event} to {} ({e:#})", path.display());
    }
}

/// The dev host's spend ledger (`<dev root>/fal-spend.jsonl`, never rotated): what the project / team
/// hard stops and the burn governor sum. Same rule as submitted_log: only where `<dev root>/runs/fal`
/// exists; `SPLAT_FAL_SPEND` overrides the path (empty or 0: off, no ledger rows and no stop).
fn spend_ledger() -> Option<PathBuf> {
    match std::env::var("SPLAT_FAL_SPEND") {
        Ok(v) if v.is_empty() || v == "0" => None,
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => {
            let root = splat_contracts::dev::dev_root()?;
            Some(root.join("fal-spend.jsonl")).filter(|_| root.join("runs/fal").is_dir())
        }
    }
}

fn ledger_rows(path: &Path) -> Vec<Value> {
    fs::read_to_string(path).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
}

/// The project hard stop before each submit (dev host only: where the spend ledger is). The total is the
/// configured stop script's (`fal_stop_drive` / `SPLAT_FAL_STOP_DRIVE`, see [`splat_contracts::dev`]):
/// `<drive> true-total --json` (ledger + today's unledgered fal billing), run with `<dev root>/tools/fal-venv/bin/python`
/// (`SPLAT_FAL_STOP_PYTHON` overrides). Refuses when true_total >= project_stop_usd, and when the ledger is on
/// but no stop script is configured or it or its python is missing; if the script fails or takes over 10 s, warns and
/// lets the submit through (fail open).
fn check_budget() -> Result<()> {
    if spend_ledger().is_none() {
        return Ok(());
    }
    let dev = splat_contracts::dev::config();
    let python = std::env::var("SPLAT_FAL_STOP_PYTHON").ok().filter(|v| !v.is_empty()).map(PathBuf::from)
        .or_else(|| dev.dev_root.as_ref().map(|r| r.join("tools/fal-venv/bin/python")))
        .unwrap_or_else(|| "python3".into());
    let python = python.display().to_string();
    let drive = dev.fal_stop_drive.as_ref().filter(|d| d.is_file()).with_context(|| format!(
        "the fal spend ledger is on but the hard-stop script is {}: set fal_stop_drive in {} \
         (or SPLAT_FAL_STOP_DRIVE), or SPLAT_FAL_SPEND=0 to turn the ledger off; refusing to submit",
        dev.fal_stop_drive.as_ref().map_or("not configured".to_owned(), |d| format!("missing ({})", d.display())),
        splat_contracts::dev::config_path().map_or("dev.toml".to_owned(), |p| p.display().to_string())))?;
    let drive = drive.display().to_string();
    ensure!(!python.contains('/') || Path::new(&python).is_file(),
        "the fal spend ledger is on but its hard-stop python is missing ({python}); set SPLAT_FAL_STOP_PYTHON; refusing to submit");
    let run = || -> Result<Value> {
        let mut child = std::process::Command::new(&python).args([drive.as_str(), "true-total", "--json"])
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null())
            .spawn().with_context(|| format!("spawn {python}"))?;
        let t0 = Instant::now();
        while child.try_wait()?.is_none() {
            if t0.elapsed() > Duration::from_secs(10) {
                child.kill().ok();
                child.wait().ok();
                bail!("timed out after 10 s");
            }
            thread::sleep(Duration::from_millis(100));
        }
        let out = child.wait_with_output()?;
        ensure!(out.status.success(), "exit {}", out.status);
        let line = String::from_utf8_lossy(&out.stdout).lines().rev().find(|l| l.trim_start().starts_with('{'))
            .context("no JSON line")?.to_string();
        Ok(serde_json::from_str(&line)?)
    };
    let t = match run() {
        Ok(t) => t,
        Err(e) => {
            println!("[budget] could not read the project total ({drive} true-total: {e:#}); submitting without the hard stop check");
            return Ok(());
        }
    };
    let (Some(total), Some(stop)) = (t["true_total"].as_f64(), t["project_stop_usd"].as_f64()) else {
        println!("[budget] {drive} true-total gave no true_total/project_stop_usd; submitting without the hard stop check");
        return Ok(());
    };
    ensure!(total < stop, "project fal spend ${total:.2} (ledger ${:.2} + unledgered today ${:.2} per fal billing) >= \
        project_stop_usd ${stop:.0} (fal-budget.json): project-wide hard stop, refusing to submit",
        t["ledger_total"].as_f64().unwrap_or(0.), t["unledgered_today"].as_f64().unwrap_or(0.));
    Ok(())
}

/// GPU busy seconds, cold start, and $ of a finished request from the runner's timing (drive.py
/// `_summarize`: billed while the runner works on it, plus setup when this request paid the cold start).
fn bill(gpu: &str, resp: &Value, submit: f64) -> Option<(f64, bool, f64)> {
    let t = &resp["timing"];
    let f = |k: &str| t[k].as_f64().unwrap_or(0.);
    if f("received_unix") <= 0. {
        return None;
    }
    let cold = f("setup_started_unix") >= submit - 1.;
    let setup = f("setup_done_unix") - f("setup_started_unix");
    let busy = (f("done_unix") - f("received_unix")).max(0.);
    Some((busy, cold, price_per_s(gpu) * (busy + if cold { setup.max(0.) } else { 0. })))
}

impl Track {
    fn submitted(&self, request_id: &str, app: &str, attempt: u32) {
        let mut row = json!({"tag": self.tag, "app": app, "request_id": request_id, "team": self.team,
            "lane": self.lane, "exp": "", "attempt": attempt});
        if let Some(root) = &self.run_root {
            row["run"] = json!(root.display().to_string());
        }
        reqlog("submit", row);
    }

    fn ended(&self, request_id: &str, how: &str) {
        reqlog("end", json!({"request_id": request_id, "how": how}));
    }

    /// The request's one fal-spend.jsonl row, as drive.py exec's (`_record_spend`): kind "exec" when the runner
    /// answered (result, runner error, refusal; busy/cold from its timing), "exec-cancelled" when this process
    /// cancelled it, "exec-unfinished" on a client error (both: busy = time seen IN_PROGRESS, an estimate).
    /// Skipped when a row already carries the request id (e.g. `drive.py cancel` from the time-box killer);
    /// fal/redispatch.py then sees a row and writes no "orphan" one for it.
    #[allow(clippy::too_many_arguments)]
    fn spent(&self, request_id: &str, app: &str, gpu: &str, kind: &str, how: &str, busy: f64, cold: bool, usd: f64) {
        let Some(path) = spend_ledger() else { return };
        if ledger_rows(&path).iter().any(|e| e["request_id"].as_str() == Some(request_id)) {
            println!("[fal] spend ledger already has {request_id}; not adding another row");
            return;
        }
        let row = json!({
            "ts": chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
            "kind": kind, "tag": self.tag, "usd": (usd * 10000.).round() / 10000., "backend": "fal",
            "app": app, "request_id": request_id, "busy_s": (busy * 10.).round() / 10., "cold": cold,
            "team": self.team, "lane": self.lane, "exp": "", "gpu": gpu, "how": how, "client": "splat-native",
        });
        if let Err(e) = append_row(&path, &row) {
            println!("[fal] could not ledger {request_id} spend to {} ({e:#})", path.display());
        }
    }
}

/// Record staged shas in the run's ledger (`{"app", "scope", "shas"}`) so `release` can drop them. A ledger
/// copied in from another run (its scope is not this run's) starts over: the shas it lists live in the other
/// run's scope and are not this run's to drop.
fn ledger_add(ledger: &Path, app: &str, scope: &str, shas: impl IntoIterator<Item = String>) -> Result<()> {
    let mut l: Value = fs::read(ledger).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_else(|| json!({}));
    let mut all: std::collections::BTreeSet<String> = if l["scope"].as_str().is_some_and(|s| s != scope) {
        Default::default()
    } else {
        l["shas"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect()
    };
    all.extend(shas);
    l["app"] = json!(app);
    l["scope"] = json!(scope);
    l["shas"] = json!(all);
    let tmp = ledger.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec(&l)?)?;
    fs::rename(tmp, ledger)?;
    Ok(())
}

/// A run's private staging dir on fal /data (fal/app.py `_stage_dir`, as each job sees it) and which of `shas`
/// are staged there already: a job finds those without any upload.
pub fn run_staged(ledger: &Path, shas: &[String]) -> Result<(String, std::collections::BTreeSet<String>)> {
    let (key, http) = (key()?, client()?);
    let tier = std::env::var("SPLAT_FAL_TIER").unwrap_or_else(|_| "product".into());
    let app = format!("splat-stage-{tier}-{}", app_version());
    let stager = Stager { http: &http, key: &key, app: &app, scope: stage_scope(&key, Some(ledger)), chunk: STAGE_CHUNK };
    let have = if shas.is_empty() { Default::default() } else { stager.have(shas)? };
    Ok((format!("/data/splat-fal/product/{}", stager.scope), have))
}

/// Record files a job stages into its run's staging dir itself, so `release` drops them with the rest.
pub fn ledger_record(ledger: &Path, shas: impl IntoIterator<Item = String>) -> Result<()> {
    let key = key()?;
    let tier = std::env::var("SPLAT_FAL_TIER").unwrap_or_else(|_| "product".into());
    ledger_add(ledger, &format!("splat-stage-{tier}-{}", app_version()), &stage_scope(&key, Some(ledger)), shas)
}

/// Who has the uplink while [`prestage`] sends: the quick SfM's few frames go first, so first view does not wait
/// behind a capture video. `paused`: every send stalls mid-body (nothing is cut or resent); `opened`: the quick
/// inputs are staged, so the video takes the whole uplink. Before that it trickles out on a few rate-limited
/// streams, so little of it sits in socket buffers when a pause starts.
#[derive(Default)]
pub struct PrestageGate {
    pub stop: std::sync::atomic::AtomicBool,
    pub paused: std::sync::atomic::AtomicBool,
    pub opened: std::sync::atomic::AtomicBool,
    /// Bytes of chunks sent before the gate opened (for the run's journal).
    pub early_bytes: std::sync::atomic::AtomicU64,
    /// Chunk sends that failed and were retried.
    pub retries: std::sync::atomic::AtomicU64,
}

/// The stage app joins a big file's chunks one by one on fal /data at ~0.3 s each: a 72 MB file committed in
/// 31 s as 144 x 512 KB, 11.7 s as 36 x 2 MB, 6.0 s as 18 x 4 MB (S3 commit probe, 2026-10-10). So few big
/// chunks, each on its own stream.
const PRESTAGE_CHUNK: usize = 8 << 20;
/// Before the gate opens: this many streams, each held to this rate (~ the host uplink in all).
const PRESTAGE_EARLY_STREAMS: usize = 4;
const PRESTAGE_EARLY_RATE: f64 = 1.0e6;

/// Stage `path` into its run's staging dir on fal ahead of the job that needs it (content-addressed, as `exec`
/// stages inputs, so that job's have-check finds it), as `gate` allows. Returns the bytes sent (0: staged
/// already); `gate.stop` aborts, leaving nothing committed.
pub fn prestage(ledger: &Path, path: &Path, gate: &Arc<PrestageGate>) -> Result<u64> {
    use std::sync::{Mutex, atomic::Ordering};
    let (key, http) = (key()?, client()?);
    let tier = std::env::var("SPLAT_FAL_TIER").unwrap_or_else(|_| "product".into());
    let app = format!("splat-stage-{tier}-{}", app_version());
    let stager = Stager { http: &http, key: &key, app: &app, scope: stage_scope(&key, Some(ledger)), chunk: PRESTAGE_CHUNK };
    let sha = sha256_file(path)?;
    ledger_add(ledger, &app, &stager.scope, [sha.clone()])?;
    if stager.have(&[sha.clone()])?.contains(&sha) {
        return Ok(0);
    }
    let len = fs::metadata(path)?.len() as usize;
    let chunks = len.div_ceil(PRESTAGE_CHUNK).max(1);
    let halted = || gate.stop.load(Ordering::Relaxed) || stopping();
    let next = Mutex::new(0usize);
    let err = Mutex::new(None);
    thread::scope(|s| {
        for worker in 0..chunks.min(STAGE_PARALLEL) {
            let (next, err, stager, sha, halted) = (&next, &err, &stager, &sha, &halted);
            s.spawn(move || {
                let result = (|| -> Result<()> {
                    loop {
                        // Until the gate opens only the first few streams send.
                        while worker >= PRESTAGE_EARLY_STREAMS && !gate.opened.load(Ordering::Relaxed) {
                            ensure!(!halted() && err.lock().unwrap().is_none(), "stopped");
                            thread::sleep(Duration::from_millis(20));
                        }
                        ensure!(!halted() && err.lock().unwrap().is_none(), "stopped");
                        let index = {
                            let mut n = next.lock().unwrap();
                            if *n >= chunks {
                                return Ok(());
                            }
                            *n += 1;
                            *n - 1
                        };
                        let size = PRESTAGE_CHUNK.min(len - index * PRESTAGE_CHUNK);
                        let early = !gate.opened.load(Ordering::Relaxed);
                        stager.put_gated(path, sha, index, size, gate)?;
                        if early {
                            gate.early_bytes.fetch_add(size as u64, Ordering::Relaxed);
                        }
                    }
                })();
                if let Err(e) = result {
                    err.lock().unwrap().get_or_insert(e);
                }
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    ensure!(!halted(), "stopped");
    stager.commit(&sha, chunks)?;
    Ok(len as u64)
}

/// Stage `files` into the run's staging dir on fal ahead of the job that sends them, `streams` at a time,
/// content-addressed as `exec` stages inputs, so that job's have-check (and `remote::pack_stage`, which leaves
/// staged files out of its tars) finds them. Files up to [`PUBLISH_CHUNK`] go up whole; bigger ones in
/// [`PRESTAGE_CHUNK`] parts under this process's upload nonce, then a commit. Once `stop` is set nothing
/// more is sent (a big file stopped between parts is left to the job). Returns (files, bytes) sent.
pub fn stage_files(ledger: &Path, files: &[PathBuf], streams: usize, stop: &std::sync::atomic::AtomicBool) -> Result<(usize, u64)> {
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    let (key, http) = (key()?, client()?);
    let tier = std::env::var("SPLAT_FAL_TIER").unwrap_or_else(|_| "product".into());
    let app = format!("splat-stage-{tier}-{}", app_version());
    let stager = Stager { http: &http, key: &key, app: &app, scope: stage_scope(&key, Some(ledger)), chunk: PRESTAGE_CHUNK };
    // Small files are read once: the sha256 and the upload hold the same bytes. A big file is hashed and
    // read again per part; a part that changed meanwhile fails the commit's sha256 check (the job re-sends it).
    let mut blobs: BTreeMap<String, (PathBuf, Option<Vec<u8>>)> = BTreeMap::new();
    for path in files {
        let len = fs::metadata(path)?.len() as usize;
        if len <= PUBLISH_CHUNK {
            let data = fs::read(path)?;
            blobs.insert(format!("{:x}", Sha256::digest(&data)), (path.clone(), Some(data)));
        } else {
            blobs.insert(sha256_file(path)?, (path.clone(), None));
        }
    }
    if blobs.is_empty() {
        return Ok((0, 0));
    }
    let shas: Vec<String> = blobs.keys().cloned().collect();
    ledger_add(ledger, &app, &stager.scope, shas.iter().cloned())?;
    let have = stager.have(&shas)?;
    let todo: Vec<(&String, &(PathBuf, Option<Vec<u8>>))> = blobs.iter().filter(|(sha, _)| !have.contains(*sha)).collect();
    let halted = || stop.load(Ordering::Relaxed) || stopping();
    let (sent, bytes) = (AtomicUsize::new(0), AtomicU64::new(0));
    parallel(&todo, streams.max(1), |(sha, (path, data))| {
        if halted() {
            return Ok(());
        }
        let len = match data {
            Some(data) => {
                stager.put_raw(sha, None, data.to_vec())?;
                data.len()
            }
            None => {
                let len = fs::metadata(path)?.len() as usize;
                let chunks = len.div_ceil(PRESTAGE_CHUNK);
                for index in 0..chunks {
                    if halted() {
                        return Ok(());
                    }
                    stager.put_chunk(path, sha, index, PRESTAGE_CHUNK.min(len - index * PRESTAGE_CHUNK))?;
                }
                stager.commit(sha, chunks)?;
                len
            }
        };
        sent.fetch_add(1, Ordering::Relaxed);
        bytes.fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    })?;
    Ok((sent.into_inner(), bytes.into_inner()))
}

/// Which of `paths` (under the binary cache /data/splat-cloud/files) exist on fal /data, so a job uploads
/// what is missing up front instead of failing once on a cold runner to find out.
pub fn cached(paths: &[String]) -> Result<Vec<bool>> {
    let (key, http) = (key()?, client()?);
    let tier = std::env::var("SPLAT_FAL_TIER").unwrap_or_else(|_| "product".into());
    let app = format!("splat-stage-{tier}-{}", app_version());
    let stager = Stager { http: &http, key: &key, app: &app, scope: stage_scope(&key, None), chunk: STAGE_CHUNK };
    let r = stager.call(json!({"op": "exists", "paths": paths}))?;
    Ok(paths.iter().enumerate().map(|(i, _)| r["exists"][i].as_bool().unwrap_or(false)).collect())
}

/// Publish a big file (the splat binary, model weights) to fal's binary cache before its job runs:
/// `<cache>/<sha>/<name>` under /data/splat-cloud/files, shared by every run. The file goes up zstd-compressed
/// (the splat binary shrinks ~4x) and is joined, decompressed and sha256-checked by the stage app. A per-sha
/// lock on this host makes concurrent runs of a new build wait for one upload instead of each sending it.
/// Ok(false) when the stage app predates op `cache_commit`: the caller keeps the per-job upload.
/// Small chunks, many streams: fal's gateway gives each stream well under 1 MB/s. The 49 MB zstd splat binary
/// (2026-10-09, stage v17): 16 MB x 8 took 70.6 s, 4 MB x 8 35.8 s, 2 MB x 32 17.2 s, 1 MB x 32 15.1 s plus a
/// slower commit. Uncompressed in 16 MB x 8 (the per-job path) it took 237 s.
pub(crate) const PUBLISH_CHUNK: usize = 2 << 20;
const PUBLISH_PARALLEL: usize = 32;

pub fn publish_cached(path: &Path, sha: &str, name: &str) -> Result<bool> {
    let (key, http) = (key()?, client()?);
    let tier = std::env::var("SPLAT_FAL_TIER").unwrap_or_else(|_| "product".into());
    let app = format!("splat-stage-{tier}-{}", app_version());
    let stager = Stager { http: &http, key: &key, app: &app, scope: stage_scope(&key, None), chunk: PUBLISH_CHUNK };
    let cached_path = format!("/data/splat-cloud/files/{sha}/{name}");
    let locks = std::env::temp_dir().join("splat-fal-publish");
    fs::create_dir_all(&locks)?;
    let lock = fs::OpenOptions::new().create(true).truncate(false).write(true).open(locks.join(format!("{sha}.lock")))?;
    let waited = Instant::now();
    fs2::FileExt::lock_exclusive(&lock)?;
    if waited.elapsed() > Duration::from_secs(1) {
        // Another process (a concurrent run, or a publish of a new build) was sending this file.
        println!("[fal] waited {:.1}s for another process publishing {name} to the fal binary cache", waited.elapsed().as_secs_f64());
    }
    if cached(std::slice::from_ref(&cached_path))?.first() == Some(&true) {
        return Ok(true);
    }
    let t0 = Instant::now();
    let raw = fs::read(path)?;
    let packed = zstd::bulk::compress(&raw, 3)?;
    let chunks: Vec<(usize, &[u8])> = packed.chunks(stager.chunk).enumerate().collect();
    parallel(&chunks, PUBLISH_PARALLEL, |(i, data)| stager.put_raw(sha, Some(*i), data.to_vec()))?;
    let r = match stager.call(json!({"op": "cache_commit", "sha256": sha, "name": name, "index": chunks.len(), "encoding": "zstd",
        "upload": upload_id()})) {
        Ok(r) => r,
        Err(e) if format!("{e:#}").contains("unknown op") => return Ok(false),
        Err(e) => return Err(e),
    };
    ensure!(r["ok"].as_bool() == Some(true), "fal cache_commit: {}", r.to_string().chars().take(300).collect::<String>());
    println!("[fal] published {name} ({:.1} MB, {:.1} MB zstd) to the fal binary cache in {:.1}s",
        raw.len() as f64 / 1e6, packed.len() as f64 / 1e6, t0.elapsed().as_secs_f64());
    Ok(true)
}

/// Publish `path` to the binary cache under its own file name and sha256, as a `--cloud` run of it would on its first
/// job (`splat __fal_publish`, run by dev tooling right after installing a build, so that first run doesn't pay ~25 s).
pub fn publish_file(path: &Path) -> Result<bool> {
    let name = path.file_name().context("publish: path has no file name")?.to_string_lossy().to_string();
    publish_cached(path, &sha256_file(path)?, &name)
}

/// Per-run record (next to the staging ledger) of results kept on fal instead of downloaded: run-relative path ->
/// {name, sha256, size, fal_path, app, scope}. `vauss pull <run>` (or [`materialize`]) downloads them.
pub const REMOTE_KEPT: &str = "remote_artifacts.json";
/// The record's name before 2026-10-10 (runs started earlier).
const REMOTE_KEPT_LEGACY: &str = "fal-remote-kept.json";
const STUB_MAGIC: &str = "splat-fal-remote v1 ";

/// Big results a `--cloud` job leaves on fal (`keep`: file names, see [`Exec::keep_remote`]): the full model's
/// PLYs (the run gets the candidate's web package) and, unless the goal is a mesh, the mesh GLBs. Held-out
/// renders may also stay staged until evaluation; small worker-generated previews come back.
fn kept_remote(rel: &str, keep: &[String], has_previews: bool) -> bool {
    let Some(rest) = rel.strip_prefix("artifacts/") else { return false };
    let Some((_, artifact_rel)) = rest.split_once('/') else { return false };
    let name = artifact_rel.rsplit('/').next().unwrap_or_default();
    keep.iter().any(|k| if k.ends_with('/') { has_previews && artifact_rel.starts_with(k) } else { k == name })
}

/// Every result of the run at `run_root` kept on fal, downloaded in place of its stub. Returns how many came down.
pub fn materialize_all(run_root: &Path) -> Result<usize> {
    let mut n = 0;
    for record in [REMOTE_KEPT, REMOTE_KEPT_LEGACY] {
        let Ok(bytes) = fs::read(run_root.join(record)) else { continue };
        let v: Value = serde_json::from_slice(&bytes)?;
        for rel in v.as_object().into_iter().flatten().map(|(k, _)| k) {
            let path = run_root.join(rel);
            if remote_stub(&path).is_some() {
                materialize(run_root, &path)?;
                n += 1;
            }
        }
    }
    Ok(n)
}

/// A local stand-in for a result kept on fal: `(sha256, size)` of the real file. Later jobs of the run stage it
/// from fal by sha; [`materialize`] downloads it when something here needs the bytes.
pub fn remote_stub(path: &Path) -> Option<(String, u64)> {
    let meta = fs::metadata(path).ok()?;
    if meta.len() > 256 {
        return None;
    }
    let text = fs::read_to_string(path).ok()?;
    let mut parts = text.strip_prefix(STUB_MAGIC)?.split_whitespace();
    Some((parts.next()?.to_string(), parts.next()?.parse().ok()?))
}

/// Replace the stub at `path` (in the run at `run_root`) with the real file, read back from the run's staging on
/// fal. No-op for a real file.
pub fn materialize(run_root: &Path, path: &Path) -> Result<()> {
    let Some((sha, size)) = remote_stub(path) else { return Ok(()) };
    // The kept record names the app and scope each result was staged under: a run copy's inherited stubs are
    // in the source run's scope, not in the one its own (later) ledger records.
    let entry = kept_entry(run_root, &sha);
    let ledger: Value = match fs::read(run_root.join(STAGE_LEDGER_NAME)) {
        Ok(b) => serde_json::from_slice(&b)?,
        Err(_) if entry.as_ref().is_some_and(|e| e["scope"].is_string() && e["app"].is_string()) => json!({}),
        Err(e) => return Err(e).context("the run's staging ledger is gone: its model is no longer on fal"),
    };
    let (key, http) = (key()?, client()?);
    let field = |name: &str| entry.as_ref().and_then(|e| e[name].as_str()).or(ledger[name].as_str()).map(String::from);
    let app = field("app").context("staging ledger: no app")?;
    let scope = field("scope").context("staging ledger: no scope")?;
    let stager = Stager { http: &http, key: &key, app: &app, scope, chunk: PUBLISH_CHUNK };
    let t0 = Instant::now();
    let chunks: Vec<usize> = (0..(size as usize).div_ceil(stager.chunk).max(1)).collect();
    let parts = std::sync::Mutex::new(vec![Vec::new(); chunks.len()]);
    parallel(&chunks, FETCH_PARALLEL, |&i| {
        let (data, _) = stager.get_raw(&format!("staged:{sha}"), i * stager.chunk)?;
        parts.lock().unwrap()[i] = data;
        Ok(())
    })
    .with_context(|| match kept_blob(run_root, &sha) {
        // devtools/prune/fal_keep_cited.py copied it to the blob store before the product scope's cleanup.
        Some(blob) => format!("{} is no longer in the run's fal scope; a kept copy is at {blob}: \
            fal-venv python devtools/prune/fal_keep_cited.py pull {}", path.display(), run_root.display()),
        None => format!("{}: download from fal failed", path.display()),
    })?;
    let data = parts.into_inner().unwrap().concat();
    use sha2::{Digest, Sha256};
    ensure!(format!("{:x}", Sha256::digest(&data)) == sha, "{}: downloaded copy does not match", path.display());
    write_atomic(path, |tmp| Ok(fs::write(tmp, &data)?))?;
    eprintln!("[fal] downloaded {} ({:.1} MB) from fal in {:.1}s", path.display(), size as f64 / 1e6, t0.elapsed().as_secs_f64());
    Ok(())
}

/// The run's kept record entry (see [`REMOTE_KEPT`]) of the file with this sha, preferring one that names a
/// `blob_path`.
fn kept_entry(run_root: &Path, sha: &str) -> Option<Value> {
    let entries: Vec<Value> = [REMOTE_KEPT, REMOTE_KEPT_LEGACY].iter()
        .filter_map(|r| fs::read(run_root.join(r)).ok())
        .filter_map(|b| serde_json::from_slice::<Value>(&b).ok())
        .flat_map(|v| v.as_object().map(|o| o.values().filter(|e| e["sha256"] == sha).cloned().collect::<Vec<_>>()).unwrap_or_default())
        .collect();
    entries.iter().find(|e| e["blob_path"].is_string()).or(entries.first()).cloned()
}

/// `blob_path` of the kept file with this sha in the run's record (set by devtools/prune/fal_keep_cited.py).
fn kept_blob(run_root: &Path, sha: &str) -> Option<String> {
    kept_entry(run_root, sha)?["blob_path"].as_str().map(String::from)
}

/// fal's content-addressed blob store (fal/app.py `_fetch`): `<KEPT_BLOBS>/<sha256>`.
const KEPT_BLOBS: &str = "/data/splat-fal/blobs";

/// The blob-store copy of the kept result `stub` (content `sha`) a job may read in place of its expired staged
/// copy: the `blob_path` recorded next to the job's staging ledger, else in the record of the run holding the
/// stub (a run copy's stub keeps its source's record). Only paths the runner reads from the store count.
fn kept_blob_for(ledger: Option<&Path>, stub: &Path, sha: &str) -> Option<String> {
    let store = format!("{KEPT_BLOBS}/{sha}");
    ledger.and_then(Path::parent).into_iter()
        .chain(stub.ancestors().skip(1).filter(|d| [REMOTE_KEPT, REMOTE_KEPT_LEGACY].iter().any(|r| d.join(r).is_file())))
        .filter_map(|root| kept_blob(root, sha))
        .find(|blob| *blob == store)
}

/// Whether the kept result `stub` (content `sha`) was staged under a scope other than `scope` (the job's run):
/// a run copy's stub of its source's result, which `materialize` reads from the source's scope.
fn inherited_stub(stub: &Path, sha: &str, scope: &str) -> bool {
    stub.ancestors().skip(1)
        .find_map(|d| kept_entry(d, sha))
        .is_some_and(|e| e["scope"].as_str().is_some_and(|s| s != scope))
}

/// The run's staging ledger file name (remote.rs STAGE_LEDGER).
const STAGE_LEDGER_NAME: &str = "fal-staged.json";

fn app_version() -> String {
    std::env::var("SPLAT_FAL_APP_VERSION").unwrap_or_else(|_| "v21".into())
}

/// Shas of the results the run next to `ledger` keeps on fal (its [`REMOTE_KEPT`] record).
fn kept_shas(ledger: &Path) -> std::collections::BTreeSet<String> {
    [REMOTE_KEPT, REMOTE_KEPT_LEGACY].iter()
        .filter_map(|r| fs::read(ledger.with_file_name(r)).ok())
        .filter_map(|b| serde_json::from_slice::<Value>(&b).ok())
        .flat_map(|v| v.as_object().into_iter().flatten().filter_map(|(_, e)| e["sha256"].as_str().map(String::from)).collect::<Vec<_>>())
        .collect()
}

/// What `release` of `ledger` (`l`, its content) may drop: `(scope, shas)`, or None when the ledger is not this
/// run's. Staging is per run (the scope hashes the ledger's path), and a run copied with its ledger (cp -a, rsync)
/// carries the source's scope and shas: dropping them would delete the source's staged inputs and kept outputs
/// (the source then fails "staged copy is gone"). A ledger without a scope predates per-run scopes; it is left
/// to the 24 h sweep too. Results left on fal (stubs, see [`remote_stub`]) stay staged for on-request download.
fn release_plan(ledger: &Path, l: &Value, key: &str) -> Option<(String, Vec<String>)> {
    let own = stage_scope(key, Some(&ledger.parent()?.canonicalize().ok()?.join(ledger.file_name()?)));
    if l["scope"].as_str() != Some(own.as_str()) {
        return None;
    }
    let keep = kept_shas(ledger);
    let shas = l["shas"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from))
        .filter(|s| !keep.contains(s)).collect();
    Some((own, shas))
}

/// End of a `--cloud` run: delete everything it staged on fal /data (under its own scope only, see
/// [`release_plan`]), then the ledger.
pub fn release(ledger: &Path) -> Result<()> {
    let Ok(bytes) = fs::read(ledger) else { return Ok(()) };
    let l: Value = serde_json::from_slice(&bytes)?;
    let key = key()?;
    let Some((scope, shas)) = release_plan(ledger, &l, &key) else {
        eprintln!("[fal] run end: the staging ledger is another run's (a copy) or predates per-run scopes; \
            nothing deleted from fal /data (the 24 h sweep removes it)");
        return Ok(());
    };
    let keep = kept_shas(ledger);
    let app = l["app"].as_str().context("staging ledger: no app")?;
    let http = client()?;
    let stager = Stager { http: &http, key: &key, app, scope, chunk: STAGE_CHUNK };
    stager.call(json!({"op": "drop", "shas": shas}))?;
    if keep.is_empty() {
        fs::remove_file(ledger)?;
    } else {
        // materialize() still finds the kept files' app and scope.
        let mut l = l.clone();
        l["shas"] = json!(keep);
        fs::write(ledger, serde_json::to_vec(&l)?)?;
    }
    // stderr: stdout carries `--json` reports.
    eprintln!("[fal] run end: deleted {} staged files from fal /data", shas.len());
    Ok(())
}

pub struct Outcome {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub cost_usd: f64,
    pub gpu_label: String,
    pub synced: usize,
}

fn key() -> Result<String> {
    std::env::var("FAL_KEY").ok().filter(|k| !k.is_empty()).context("--cloud needs FAL_KEY in the environment")
}

fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3600))
        .connect_timeout(Duration::from_secs(30))
        .build()?)
}

/// An HTTP answer no retry changes (4xx other than 429): [`passes`] gives up on it at once.
#[derive(Debug)]
struct Permanent(String);
impl std::fmt::Display for Permanent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Permanent {}

/// Waits between passes over a finished request's result (its JSON, then its files): ~13 min in all. One
/// [`retry`] round is ~1 min, and the shared host's uplink, saturated by other lanes, timed out connects for
/// longer (a driveway job, 2026-10-10: `fal get_raw … Connection timed out (os error 110)`).
const RESULT_PASS_WAITS_S: [u64; 6] = [15, 30, 60, 120, 240, 300];

/// `f` again after each of `waits` while it fails on transport or server errors: a request that ran must not
/// fail for a fetch the network dropped. Gives up at once on a [`Permanent`] error or when the job is stopping.
fn passes<T>(what: &str, waits: &[Duration], mut f: impl FnMut() -> Result<T>) -> Result<T> {
    for (pass, wait) in waits.iter().enumerate() {
        match f() {
            Err(e) if !stopping() && !e.chain().any(|c| c.is::<Permanent>()) => {
                println!("[fal] {what}: {e:#}; pass {} of {} in {}s", pass + 2, waits.len() + 1, wait.as_secs());
                thread::sleep(*wait);
            }
            r => return r,
        }
    }
    f()
}

fn result_waits() -> Vec<Duration> {
    RESULT_PASS_WAITS_S.iter().map(|s| Duration::from_secs(*s)).collect()
}

/// Run `f` with backoff on transport errors, 429 and 5xx (fal gateway hiccups).
fn retry<T>(what: &str, mut f: impl FnMut() -> Result<reqwest::blocking::Response>, ok: impl Fn(reqwest::blocking::Response) -> Result<T>) -> Result<T> {
    let mut last = None;
    for attempt in 0..5u32 {
        match f() {
            Ok(r) if r.status().is_success() => return ok(r),
            Ok(r) if r.status().as_u16() == 429 || r.status().is_server_error() => {
                last = Some(format!("{what}: HTTP {}", r.status()));
            }
            Ok(r) => {
                let status = r.status();
                let body = r.text().unwrap_or_default();
                return Err(Permanent(format!("{what}: HTTP {status}: {}", body.chars().take(2000).collect::<String>())).into());
            }
            Err(e) => last = Some(format!("{what}: {e:#}")),
        }
        let wait = 2u64.pow(attempt + 1);
        eprintln!("[fal] {}; retry {} in {wait}s", last.as_deref().unwrap_or_default(), attempt + 1);
        thread::sleep(Duration::from_secs(wait));
    }
    bail!("{} (gave up after 5 attempts)", last.unwrap_or_default())
}

/// A file's sha256; big files' come from the cross-process cache ([`crate::hashcache`]) when the file is unchanged.
fn sha256_file(path: &Path) -> Result<String> {
    crate::hashcache::sha256(path)
}

/// The private namespace on fal's /data volume: a hash of the key (never the key) and, for a run, of its
/// ledger path, so concurrent runs on one key never delete each other's staged files.
fn stage_scope(key: &str, ledger: Option<&Path>) -> String {
    use sha2::{Digest, Sha256};
    let run = ledger.map(|l| l.to_string_lossy().into_owned()).unwrap_or_default();
    format!("{:x}", Sha256::digest(format!("splat-stage/v1:{key}:{run}").as_bytes()))[..32].to_string()
}

/// Wake the upload app while the client inspects/decodes the capture. Control requests normally
/// use its separate ctl twin, which does not warm the app serving put_raw. This sends no capture
/// data and is best effort: a timeout never delays or fails the run.
pub fn warm_upload_stage(tier: &str) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let app = format!("splat-stage-{tier}-{}", app_version());
    STARTED.call_once(|| {
        thread::spawn(move || {
            let started = Instant::now();
            let result = (|| -> Result<()> {
                let (key, http) = (key()?, client()?);
                let stager = Stager { http: &http, key: &key, app: &app,
                    scope: stage_scope(&key, None), chunk: STAGE_CHUNK };
                let body = json!({"op": "have", "shas": [], "scope": stager.scope,
                    "client_rev": format!("splat-{}", env!("CARGO_PKG_VERSION")), "client_proto": CLIENT_PROTO});
                stager.call_sync(&app, &body, Duration::from_secs(180))?;
                Ok(())
            })();
            reqlog("stage_warm", json!({"app": app, "wall_s": started.elapsed().as_secs_f64(),
                "ok": result.is_ok(), "capture_bytes": 0}));
        });
    });
}

/// How long a request may wait for a runner, then run, before it is cancelled and resubmitted.
#[derive(Clone, Copy)]
struct Deadlines {
    queued: Duration,
    running: Duration,
}

/// The /stage app (`splat-stage-<tier>-<version>`, a CPU machine; fal/app.py SplatStage): a request waits
/// up to 15 min for a cold runner and runs up to 3 min (its request_timeout is 5 min).
const STAGE_DEADLINES: Deadlines = Deadlines { queued: Duration::from_secs(900), running: Duration::from_secs(180) };

/// One request through fal's queue (`/<path>` of `app`) and its JSON response. With deadlines, a request
/// stuck in the queue or on a runner (fal once held a lost /stage request as running for 2 h) is
/// cancelled and resubmitted, up to 3 times.
fn queue_call(http: &reqwest::blocking::Client, key: &str, app: &str, path: &str, body: &Value, poll: Duration,
              deadlines: Option<Deadlines>) -> Result<Value> {
    let tries = if deadlines.is_some() { 3 } else { 1 };
    for attempt in 1..=tries {
        match queue_call_once(http, key, app, path, body, poll, deadlines)? {
            Some(v) => return Ok(v),
            None if attempt < tries => println!("[fal] {path} request stuck past its deadline; cancelled, resubmitting ({attempt}/{tries})"),
            None => {}
        }
    }
    bail!("fal {app}/{path}: {tries} requests in a row got stuck (queued or running past their deadline); fal may be degraded")
}

/// None: the request passed a deadline and was cancelled.
fn queue_call_once(http: &reqwest::blocking::Client, key: &str, app: &str, path: &str, body: &Value, poll: Duration,
                   deadlines: Option<Deadlines>) -> Result<Option<Value>> {
    let submitted: Value = retry(
        &format!("fal {path} submit"),
        || Ok(http.post(format!("{QUEUE}/{owner}/{app}/{path}", owner = owner()))
            .header("Authorization", format!("Key {key}"))
            .header("X-Fal-Store-IO", "0")
            .json(body)
            .send()?),
        |r| Ok(r.json()?),
    )?;
    let status_url = submitted["status_url"].as_str().context("fal submit: no status_url")?.to_string();
    let response_url = submitted["response_url"].as_str().context("fal submit: no response_url")?.to_string();
    let cancel_url = submitted["cancel_url"].as_str().map(String::from).unwrap_or_else(|| {
        format!("{QUEUE}/{owner}/{app}/requests/{}/cancel", submitted["request_id"].as_str().unwrap_or(""), owner = owner())
    });
    let (submitted_at, mut running_since) = (Instant::now(), None);
    loop {
        let st: Value = retry(
            &format!("fal {path} status"),
            || Ok(http.get(&status_url).header("Authorization", format!("Key {key}")).send()?),
            |r| Ok(r.json()?),
        )?;
        match st["status"].as_str() {
            Some("COMPLETED") => break,
            Some("IN_PROGRESS") if running_since.is_none() => running_since = Some(Instant::now()),
            _ => {}
        }
        if let Some(d) = deadlines {
            let late = match running_since {
                Some(t) => t.elapsed() > d.running,
                None => submitted_at.elapsed() > d.queued,
            };
            if late {
                // Best effort: fal cancels queued requests; a running one ends at the app's request_timeout.
                http.put(&cancel_url).header("Authorization", format!("Key {key}")).send().ok();
                return Ok(None);
            }
        }
        thread::sleep(poll);
    }
    retry(
        &format!("fal {path} result"),
        || Ok(http.get(&response_url).header("Authorization", format!("Key {key}")).send()?),
        |r| Ok(r.json()?),
    )
    .map(Some)
}

struct Stager<'a> {
    http: &'a reqwest::blocking::Client,
    key: &'a str,
    app: &'a str,
    scope: String,
    chunk: usize,
}

/// A [`Stager`] that owns its parts, for a request on a detached thread ([`hedged`]).
#[derive(Clone)]
struct OwnedStager {
    http: reqwest::blocking::Client,
    key: String,
    app: String,
    scope: String,
    chunk: usize,
}

impl OwnedStager {
    fn get(&self) -> Stager<'_> {
        Stager { http: &self.http, key: &self.key, app: &self.app, scope: self.scope.clone(), chunk: self.chunk }
    }
}

/// Requests of this process a [`hedged`] call sent twice, and those the second copy answered first.
static HEDGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static HEDGE_WINS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `(sent twice, second copy answered first)` so far in this process.
fn hedge_counts() -> (usize, usize) {
    use std::sync::atomic::Ordering;
    (HEDGES.load(Ordering::Relaxed), HEDGE_WINS.load(Ordering::Relaxed))
}

/// `first()`, and when it has not answered within `after`, `second()` alongside it: the first Ok of the two. fal's
/// stage apps stall for minutes at a time, one app (all its runners) at once: on 2026-10-12 the product upload app
/// answered nothing for 180 s (a stage_warm timed out) while its ctl twin answered have-checks in 0.3 s, and single
/// 0.8-2 MB put_raw requests took 140-207 s, single 2 MB get_raw chunks 144-189 s (product-time-breakdown.md). So
/// `second` goes to the other app where it can. An Err from `first` before `after` is returned as is, or with
/// `fallback` `second` then runs alone (the caller's own fallback, now also its hedge). Both failing returns the last
/// error. The slower copy runs on, detached, until its own request timeout: only idempotent requests are hedged
/// (reads, and whole files the stage app sha256-checks and renames into place, see fal/app.py `_put_raw`).
fn hedged<T: Send + 'static>(what: &str, route: &str, after: Duration, fallback: bool,
                              first: impl FnOnce() -> Result<T> + Send + 'static,
                              second: impl FnOnce() -> Result<T> + Send + 'static) -> Result<T> {
    use std::sync::{atomic::Ordering, mpsc};
    let (tx, rx) = mpsc::channel::<(bool, Result<T>)>();
    let t0 = Instant::now();
    let tx1 = tx.clone();
    thread::spawn(move || drop(tx1.send((false, first()))));
    match rx.recv_timeout(after) {
        Ok((_, Ok(v))) => return Ok(v),
        Ok((_, Err(e))) if !fallback || stopping() => return Err(e),
        Ok((_, Err(e))) => {
            println!("[fal] {what} failed ({e:#}); sent again {route}");
            return second().with_context(|| format!("{what} (after: {e:#})"));
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => bail!("{what}: request thread ended without an answer"),
        Err(mpsc::RecvTimeoutError::Timeout) => {}
    }
    HEDGES.fetch_add(1, Ordering::Relaxed);
    println!("[fal] hedge: {what} unanswered after {:.1}s; sent again {route}", t0.elapsed().as_secs_f64());
    thread::spawn(move || drop(tx.send((true, second()))));
    let mut last = None;
    for (hedge, r) in rx.iter().take(2) {
        match r {
            Ok(v) => {
                if hedge {
                    HEDGE_WINS.fetch_add(1, Ordering::Relaxed);
                }
                println!("[fal] hedge: {what} answered by the {} copy in {:.1}s", if hedge { "second" } else { "first" },
                    t0.elapsed().as_secs_f64());
                return Ok(v);
            }
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("{what}: request threads ended without an answer")))
}

/// Small /stage ops sent to fal's sync endpoint first, with their timeout; the queue (deadlines, resubmission) is the
/// fallback. Each saves the queue's submit + 300 ms poll round, several per job on the first-view path. Long or
/// off-path ops (cache_commit, drop, delete, get) stay on the queue.
/// A hung sync request falls back to the queue sooner: room's binary-cache `exists` waited out 20 s
/// before its queued retry (bench 18c1ce3). Bench `have` checks: p50 0.6 s, p90 1.0 s, p99 8 s.
const SYNC_OPS: [(&str, Duration); 3] =
    [("have", Duration::from_secs(5)), ("exists", Duration::from_secs(3)), ("commit", Duration::from_secs(120))];

/// When [`Stager::call_sync_hedged`] sends a sync op again: have/exists answer in ~0.1-1 s (bench p90 1.0 s), a commit
/// joins its chunks on /data at ~0.3 s each (`index`: the chunk count).
fn hedge_after_op(op: &str, body: &Value) -> Duration {
    match op {
        "commit" => Duration::from_secs(10 + body["index"].as_u64().unwrap_or(0)),
        _ => Duration::from_millis(1500),
    }
}

/// When a whole-file upload or a result fetch of `bytes` is sent again ([`hedged`]): `base` plus the bytes at `rate`
/// (bytes/s), well under one gateway stream's usual ~0.75 MB/s up and ~1.5 MB/s down, so only a stalled request hedges.
fn hedge_after(base: f64, bytes: usize, rate: f64) -> Duration {
    Duration::from_secs_f64(base + bytes as f64 / rate)
}

/// A job whose new inputs total at most this many bytes is a small upload: its whole files (up to the stager's chunk)
/// go up hedged ([`Stager::put_whole`]). Bigger uploads keep their parallel streams unhedged: there the shared uplink
/// is the limit, and a re-sent request only takes from the others. Chunks of big files are never hedged: two uploads
/// of one chunk race its commit on fal (HTTP 500, 2026-10-10).
const HEDGE_JOB_BYTES: u64 = 8 << 20;
/// A hedged whole-file put_raw's own timeout: its hedge covers a stall, so it need not wait out the app's 300 s.
fn whole_put_timeout(bytes: usize) -> Duration {
    Duration::from_secs_f64(30. + bytes as f64 / 50e3)
}

/// Whole input files up to SMALL_FILE bytes go to the ctl twin in `put_many` batches of up to SMALL_BATCH bytes
/// ([`Stager::put_small`]), not one put_raw each to the upload app; bigger ones need put_raw's parallel raw streams.
const SMALL_FILE: usize = 256 << 10;
const SMALL_BATCH: usize = 1 << 20;
/// A `put_many` batch's timeout before its bytes: then its files go up with put_raw.
const SMALL_PUT_TIMEOUT: Duration = Duration::from_secs(8);

/// This process's chunked uploads go to their own parts dir on fal (fal/app.py `_parts_dir`): two jobs uploading one
/// file into a run's scope at once cannot delete each other's chunks on commit (IR1, 2026-10-10). Stage apps before
/// the fix ignore it and share one dir.
fn upload_id() -> &'static str {
    static ID: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ID.get_or_init(|| {
        use sha2::{Digest, Sha256};
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        format!("{:x}", Sha256::digest(format!("{}:{nanos}", std::process::id()).as_bytes()))[..16].to_string()
    })
}

/// Per stage app: whether its control/fetch twin (`splat-stage-ctl-...`, fal/app.py SplatStageCtl) answers. Probed once
/// per process; uploads stay on the main app, so control ops and fetches never queue behind slow put_raw streams.
static CTL_LIVE: std::sync::Mutex<BTreeMap<String, bool>> = std::sync::Mutex::new(BTreeMap::new());

impl Stager<'_> {
    /// The app for control ops (/stage) and fetches (/get_raw, /get_many): the ctl twin when it is deployed (a 404 or
    /// any probe failure keeps the main app), never for put_raw. SPLAT_FAL_STAGE_CTL=0 turns it off.
    fn ctl(&self) -> String {
        let Some(twin) = self.app.strip_prefix("splat-stage-").map(|rest| format!("splat-stage-ctl-{rest}")) else {
            return self.app.to_string();
        };
        if std::env::var("SPLAT_FAL_STAGE_CTL").is_ok_and(|v| v == "0") { return self.app.to_string(); }
        let mut live = CTL_LIVE.lock().unwrap();
        let up = *live.entry(twin.clone()).or_insert_with(|| {
            let body = json!({"op": "have", "shas": [], "scope": self.scope,
                "client_rev": format!("splat-{}", env!("CARGO_PKG_VERSION")), "client_proto": CLIENT_PROTO});
            self.http.post(format!("{SYNC}/{owner}/{twin}/stage", owner = owner())).header("Authorization", format!("Key {}", self.key))
                .header("X-Fal-Store-IO", "0").timeout(Duration::from_secs(10)).json(&body).send()
                .is_ok_and(|r| r.status().is_success())
        });
        static NOTED: std::sync::Once = std::sync::Once::new();
        NOTED.call_once(|| println!("[fal] ctl app: {}", if up { twin.as_str() } else { self.app }));
        if up { twin } else { self.app.to_string() }
    }

    /// The ctl twin stopped answering (404): the main app for the rest of this process.
    fn ctl_gone(&self, app: &str) {
        if app != self.app {
            println!("[fal] {app} not found; control ops and fetches go to {}", self.app);
            CTL_LIVE.lock().unwrap().insert(app.to_string(), false);
        }
    }

    /// `body` with this stager's scope and the client's revision and protocol, as every /stage request carries.
    fn stamped(&self, mut body: Value) -> Value {
        body["scope"] = json!(self.scope);
        body["client_rev"] = json!(format!("splat-{}", env!("CARGO_PKG_VERSION")));
        body["client_proto"] = json!(CLIENT_PROTO);
        body
    }

    fn call(&self, body: Value) -> Result<Value> {
        let body = self.stamped(body);
        let op = body["op"].as_str().unwrap_or_default().to_string();
        let mut app = self.ctl();
        let resp = match SYNC_OPS.iter().find(|(o, _)| *o == op).map(|(_, t)| self.call_sync_hedged(&app, &body, *t)) {
            Some(Ok(resp)) => resp,
            other => {
                if let Some(Err(e)) = other {
                    if format!("{e:#}").starts_with("HTTP 404") {
                        self.ctl_gone(&app);
                        app = self.app.to_string();
                    }
                    println!("[fal] /stage {op} on the sync endpoint failed ({e:#}); retrying through the queue");
                    // A commit whose sync request finished on the runner but not here: done already.
                    if let ("commit", Some(sha)) = (op.as_str(), body["sha256"].as_str()) {
                        if self.have(&[sha.to_string()]).is_ok_and(|h| h.contains(sha)) {
                            return Ok(json!({"ok": true}));
                        }
                    }
                }
                queue_call(self.http, self.key, &app, "stage", &body, Duration::from_millis(300), Some(STAGE_DEADLINES))?
            }
        };
        if resp.get("detail").is_some() && resp.get("ok").is_none() && resp.get("have").is_none() {
            bail!("fal /stage {}: {}", body["op"], resp["detail"].to_string().chars().take(500).collect::<String>());
        }
        Ok(resp)
    }

    /// One /stage request on fal's sync endpoint (no queue submit/poll: ~0.1 s vs ~0.5 s for a have-check).
    fn call_sync(&self, app: &str, body: &Value, timeout: Duration) -> Result<Value> {
        let r = self.http.post(format!("{SYNC}/{owner}/{app}/stage", owner = owner()))
            .header("Authorization", format!("Key {}", self.key))
            .header("X-Fal-Store-IO", "0")
            .timeout(timeout)
            .json(body)
            .send()?;
        let status = r.status();
        ensure!(status.is_success(), "HTTP {status}: {}", r.text().unwrap_or_default().chars().take(300).collect::<String>());
        Ok(r.json()?)
    }

    /// [`Self::call_sync`] on `app`, sent again to the main stage app when unanswered after [`hedge_after_op`]. Every
    /// SYNC_OP is idempotent: have/exists read; a commit of chunks another commit joined already finds the file in
    /// place (fal/app.py `_stage` op commit), and a commit never deletes the parts before the joined file is renamed in.
    fn call_sync_hedged(&self, app: &str, body: &Value, timeout: Duration) -> Result<Value> {
        let op = body["op"].as_str().unwrap_or_default().to_string();
        let (first, second) = (self.owned(), self.owned());
        let (app1, body1, body2) = (app.to_string(), body.clone(), body.clone());
        hedged(&format!("/stage {op} on {app}"), &format!("to {}", self.app), hedge_after_op(&op, body), false,
            move || first.get().call_sync(&app1, &body1, timeout),
            move || { let s = second.get(); s.call_sync(s.app, &body2, timeout) })
    }

    fn owned(&self) -> OwnedStager {
        OwnedStager { http: self.http.clone(), key: self.key.to_string(), app: self.app.to_string(), scope: self.scope.clone(),
            chunk: self.chunk }
    }

    /// Which of `shas` are already staged under this scope.
    fn have(&self, shas: &[String]) -> Result<std::collections::BTreeSet<String>> {
        let r = self.call(json!({"op": "have", "shas": shas}))?;
        Ok(r["have"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect())
    }

    /// Raw bytes to the stage app's /put_raw on fal's sync endpoint: a whole small file (index None,
    /// sha256-checked by the runner) or chunk `index` of a big one. One stream through fal's gateway gets
    /// ~0.75 MB/s, so callers run several at once (8 reached ~4.75 MB/s, about this host's uplink).
    fn put_raw(&self, sha: &str, index: Option<usize>, data: Vec<u8>) -> Result<()> {
        self.put_raw_within(sha, index, data, Duration::from_secs(300))
    }

    /// [`Self::put_raw`] with each attempt's timeout.
    fn put_raw_within(&self, sha: &str, index: Option<usize>, data: Vec<u8>, timeout: Duration) -> Result<()> {
        let url = format!("{SYNC}/{owner}/{}/put_raw", self.app, owner = owner());
        let mut query = vec![("scope", self.scope.clone()), ("sha256", sha.to_string()), ("client_proto", CLIENT_PROTO.to_string())];
        if let Some(i) = index {
            query.push(("index", i.to_string()));
            query.push(("upload", upload_id().to_string()));
        }
        retry(
            "fal put_raw",
            || Ok(self.http.post(&url)
                .query(&query)
                .header("Authorization", format!("Key {}", self.key))
                .header("X-Fal-Store-IO", "0")
                .header("Content-Type", "application/octet-stream")
                .timeout(timeout)
                .body(data.clone())
                .send()?),
            |r| {
                let v: Value = r.json()?;
                ensure!(v["ok"].as_bool() == Some(true), "fal put_raw: {}", v.to_string().chars().take(300).collect::<String>());
                Ok(())
            },
        )
    }

    /// Small whole files in one /stage `put_many` request (base64, each sha256-checked by the runner) on the ctl twin's
    /// sync endpoint. The ctl twin's runners never carry put_raw streams, so a job whose frames are staged already does
    /// not wait on the upload app: demo e1bb3141's full train job staged 3 tiny files in 38.4 s because that app took
    /// 38.1 s to answer anything (its stage_warm too; 4 other runs' warms took 21-41 s in the same minute) while the
    /// ctl twin answered the have-check in 0.3 s. Err: the caller sends the files with put_raw instead.
    fn put_small(&self, files: &[(&str, &[u8])]) -> Result<()> {
        use base64::Engine;
        let bytes: usize = files.iter().map(|f| f.1.len()).sum();
        let list: Vec<Value> = files.iter()
            .map(|(sha, data)| json!({"sha256": sha, "data": base64::engine::general_purpose::STANDARD.encode(data)}))
            .collect();
        let body = self.stamped(json!({"op": "put_many", "files": list}));
        let app = self.ctl();
        // ~0.1 s on a live runner; the base64 body at a congested gateway stream's ~100 KB/s on top.
        let timeout = SMALL_PUT_TIMEOUT + Duration::from_secs_f64(bytes as f64 * 4.0 / 3.0 / 100e3);
        let resp = self.call_sync(&app, &body, timeout).inspect_err(|e| {
            if format!("{e:#}").starts_with("HTTP 404") {
                self.ctl_gone(&app);
            }
        })?;
        ensure!(resp["ok"].as_bool() == Some(true) && resp["files"].as_u64() == Some(files.len() as u64),
            "fal /stage put_many: {}", resp.to_string().chars().take(300).collect::<String>());
        Ok(())
    }

    /// [`Self::put_small`], and when it has not answered within [`hedge_after`], each file with put_raw to the upload
    /// app alongside it (also the fallback when it fails). Both write the same sha256-checked files.
    fn put_small_hedged(&self, files: &[(&str, &[u8])]) -> Result<()> {
        let bytes: usize = files.iter().map(|f| f.1.len()).sum();
        let owned: Arc<Vec<(String, Vec<u8>)>> = Arc::new(files.iter().map(|(s, d)| (s.to_string(), d.to_vec())).collect());
        let (first, second, own2) = (self.owned(), self.owned(), owned.clone());
        hedged(&format!("put_many of {} small inputs ({:.1} MB)", files.len(), bytes as f64 / 1e6),
            &format!("with put_raw to {}", self.app), hedge_after(3., bytes, 250e3), true,
            move || first.get().put_small(&owned.iter().map(|(s, d)| (s.as_str(), d.as_slice())).collect::<Vec<_>>()),
            move || {
                let s = second.get();
                parallel(&own2, STAGE_PARALLEL, |(sha, data)| s.put_raw_within(sha, None, data.clone(), whole_put_timeout(data.len())))
            })
    }

    /// A whole input file of a small upload: put_raw to the upload app, and when it has not answered within
    /// [`hedge_after`] (or failed), the same file as a `put_many` on the ctl twin.
    fn put_whole(&self, sha: &str, data: &[u8]) -> Result<()> {
        let owned = Arc::new((sha.to_string(), data.to_vec()));
        let (first, second, own2) = (self.owned(), self.owned(), owned.clone());
        hedged(&format!("put_raw of {:.1} MB", data.len() as f64 / 1e6), "as put_many to the ctl app",
            hedge_after(4., data.len(), 500e3), true,
            move || first.get().put_raw_within(&owned.0, None, owned.1.clone(), whole_put_timeout(owned.1.len())),
            move || second.get().put_small(&[(own2.0.as_str(), own2.1.as_slice())]))
    }

    /// Chunk `index` (`size` bytes) of a big file as [`prestage`]'s `gate` allows: the body stalls while paused and
    /// is held to [`PRESTAGE_EARLY_RATE`] until the gate opens; fal hiccups retry with backoff, as [`retry`].
    fn put_gated(&self, path: &Path, sha: &str, index: usize, size: usize, gate: &Arc<PrestageGate>) -> Result<()> {
        use std::io::{Seek, SeekFrom};
        use std::sync::atomic::Ordering;
        /// The chunk's bytes, given out as the gate allows (a token bucket while it is closed).
        struct Gated { data: Arc<Vec<u8>>, at: usize, gate: Arc<PrestageGate>, tokens: f64, last: Instant }
        impl Read for Gated {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let left = self.data.len() - self.at;
                if left == 0 || buf.is_empty() {
                    return Ok(0);
                }
                let mut n = buf.len().min(left).min(64 << 10);
                loop {
                    if self.gate.stop.load(Ordering::Relaxed) || stopping() {
                        return Err(std::io::Error::other("prestage stopped"));
                    }
                    let now = Instant::now();
                    if self.gate.opened.load(Ordering::Relaxed) {
                        break;
                    }
                    if !self.gate.paused.load(Ordering::Relaxed) {
                        self.tokens = (self.tokens + PRESTAGE_EARLY_RATE * (now - self.last).as_secs_f64()).min(256e3);
                        if self.tokens >= 16e3 {
                            n = n.min(self.tokens as usize);
                            self.tokens -= n as f64;
                            self.last = now;
                            break;
                        }
                    }
                    self.last = now;
                    thread::sleep(Duration::from_millis(10));
                }
                buf[..n].copy_from_slice(&self.data[self.at..self.at + n]);
                self.at += n;
                Ok(n)
            }
        }
        let mut file = fs::File::open(path)?;
        file.seek(SeekFrom::Start((index * self.chunk) as u64))?;
        let mut buf = vec![0u8; size];
        file.read_exact(&mut buf)?;
        let data = Arc::new(buf);
        let url = format!("{SYNC}/{owner}/{}/put_raw", self.app, owner = owner());
        let query = [("scope", self.scope.clone()), ("sha256", sha.to_string()), ("client_proto", CLIENT_PROTO.to_string()),
            ("index", index.to_string()), ("upload", upload_id().to_string())];
        let mut failures = 0u32;
        loop {
            let body = Gated { data: data.clone(), at: 0, gate: gate.clone(), tokens: 64e3, last: Instant::now() };
            let sent = self.http.post(&url)
                .query(&query)
                .header("Authorization", format!("Key {}", self.key))
                .header("X-Fal-Store-IO", "0")
                .header("Content-Type", "application/octet-stream")
                .timeout(Duration::from_secs(300))
                .body(reqwest::blocking::Body::sized(body, size as u64))
                .send();
            let problem = match sent {
                Ok(r) if r.status().is_success() => {
                    let v: Value = r.json()?;
                    ensure!(v["ok"].as_bool() == Some(true), "fal put_raw: {}", v.to_string().chars().take(300).collect::<String>());
                    return Ok(());
                }
                Ok(r) if r.status().as_u16() == 429 || r.status().is_server_error() => format!("HTTP {}", r.status()),
                Ok(r) => {
                    let status = r.status();
                    bail!("fal put_raw: HTTP {status}: {}", r.text().unwrap_or_default().chars().take(2000).collect::<String>());
                }
                Err(_) if gate.stop.load(Ordering::Relaxed) || stopping() => bail!("stopped"),
                Err(e) => format!("{e:#}"),
            };
            failures += 1;
            gate.retries.fetch_add(1, Ordering::Relaxed);
            ensure!(failures < 5, "fal put_raw: {problem} (gave up after 5 attempts)");
            let wait = 2u64.pow(failures);
            eprintln!("[fal] put_raw: {problem}; retry {failures} in {wait}s");
            thread::sleep(Duration::from_secs(wait));
        }
    }

    /// Chunk `index` (`len` bytes) of a big file.
    fn put_chunk(&self, path: &Path, sha: &str, index: usize, len: usize) -> Result<()> {
        use std::io::{Seek, SeekFrom};
        let mut file = fs::File::open(path)?;
        file.seek(SeekFrom::Start((index * self.chunk) as u64))?;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf)?;
        self.put_raw(sha, Some(index), buf)
    }

    /// Join a big file's chunks (the runner checks the sha256).
    fn commit(&self, sha: &str, chunks: usize) -> Result<()> {
        self.call(json!({"op": "commit", "sha256": sha, "index": chunks, "upload": upload_id()}))?;
        Ok(())
    }

    /// A result file the runner kept on /data (under its result_data_path), read back in chunks.
    fn fetch(&self, name: &str) -> Result<Vec<u8>> {
        let (first, size) = self.get_raw(name, 0)?;
        let chunks = size.div_ceil(self.chunk).max(1);
        if chunks == 1 {
            ensure!(first.len() == size, "fal get_raw {name}: {} of {size} bytes", first.len());
            return Ok(first);
        }
        // The rest of a big file in parallel too: one gateway stream is ~0.75 MB/s.
        let parts = std::sync::Mutex::new(vec![Vec::new(); chunks]);
        parts.lock().unwrap()[0] = first;
        parallel(&(1..chunks).collect::<Vec<_>>(), 4, |&i| {
            let (data, _) = self.get_raw(name, i * self.chunk)?;
            ensure!(data.len() == self.chunk.min(size - i * self.chunk), "fal get_raw {name}: short chunk {i}");
            parts.lock().unwrap()[i] = data;
            Ok(())
        })?;
        let out = parts.into_inner().unwrap().concat();
        ensure!(out.len() == size, "fal get_raw {name}: {} of {size} bytes", out.len());
        Ok(out)
    }

    /// Several small result files in one request (the stage app's /get_many: one tar, member i = names[i]).
    fn get_many(&self, names: &[String]) -> Result<Vec<Vec<u8>>> {
        let url = format!("{SYNC}/{owner}/{}/get_many", self.ctl(), owner = owner());
        let body = json!({"scope": self.scope, "names": names, "client_proto": CLIENT_PROTO});
        let tar_bytes = retry(
            "fal get_many",
            || Ok(self.http.post(&url)
                .header("Authorization", format!("Key {}", self.key))
                .header("X-Fal-Store-IO", "0")
                .timeout(Duration::from_secs(300))
                .json(&body)
                .send()?),
            |r| Ok(r.bytes()?.to_vec()),
        )?;
        let mut out = vec![None; names.len()];
        for entry in tar::Archive::new(&tar_bytes[..]).entries()? {
            let mut entry = entry?;
            let i: usize = entry.path()?.to_string_lossy().parse().context("fal get_many: bad member name")?;
            ensure!(i < names.len(), "fal get_many: member {i} of {}", names.len());
            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;
            out[i] = Some(data);
        }
        out.into_iter().enumerate().map(|(i, d)| d.with_context(|| format!("fal get_many: {} missing", names[i]))).collect()
    }

    /// One chunk of a result file as raw bytes from the stage app's /get_raw (fal's sync endpoint),
    /// and the file's size.
    fn get_raw(&self, name: &str, offset: usize) -> Result<(Vec<u8>, usize)> {
        let url = format!("{SYNC}/{owner}/{}/get_raw", self.ctl(), owner = owner());
        let query = [("scope", self.scope.clone()), ("name", name.to_string()), ("offset", offset.to_string()),
            ("length", self.chunk.to_string()), ("client_proto", CLIENT_PROTO.to_string())];
        retry(
            "fal get_raw",
            || Ok(self.http.post(&url)
                .query(&query)
                .header("Authorization", format!("Key {}", self.key))
                .header("X-Fal-Store-IO", "0")
                .timeout(Duration::from_secs(300))
                .send()?),
            |r| {
                let size = r.headers().get("x-stage-size").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok())
                    .context("fal get_raw: no x-stage-size")?;
                Ok((r.bytes()?.to_vec(), size))
            },
        )
    }

    fn delete(&self, name: &str) -> Result<()> {
        self.call(json!({"op": "delete", "name": name}))?;
        Ok(())
    }
}

/// Job staging and result fetches move PUBLISH_CHUNK pieces, this many in flight (the stage app serves 8
/// per runner on up to 8 runners): each gateway stream is slow, so many small ones beat a few big ones.
const STAGE_PARALLEL: usize = PUBLISH_PARALLEL;
/// Fewer for fetches: with 4 runs fetching at once, 32 each (240 groups) took room's 488 MB from 34 to 91 s.
const FETCH_PARALLEL: usize = 12;

/// `f` over `items` with up to `n` running at once; stops at, and returns, the first error.
fn parallel<T: Sync>(items: &[T], n: usize, f: impl Fn(&T) -> Result<()> + Sync) -> Result<()> {
    use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};
    let (next, err) = (AtomicUsize::new(0), Mutex::new(None));
    thread::scope(|s| {
        for _ in 0..n.min(items.len()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() || err.lock().unwrap().is_some() {
                    break;
                }
                if let Err(e) = f(&items[i]) {
                    err.lock().unwrap().get_or_insert(e);
                    break;
                }
            });
        }
    });
    err.into_inner().unwrap().map_or(Ok(()), Err)
}

/// One raw upload: a whole small file, or one chunk of a big file (`data`: from the copy read while hashing).
enum Upload<'a> {
    /// Small whole files, one `put_many` request ([`SMALL_FILE`]).
    Small(Vec<(&'a str, &'a [u8])>),
    File { sha: &'a str, data: &'a [u8] },
    Part { sha: &'a str, index: usize, data: &'a [u8] },
    Chunk { sha: &'a str, path: &'a Path, index: usize, len: usize },
}

/// `files` in order, cut into `put_many` batches of at most [`SMALL_BATCH`] bytes (a bigger file alone).
fn small_batches<'a>(files: Vec<(&'a str, &'a [u8])>) -> Vec<Vec<(&'a str, &'a [u8])>> {
    let (mut batches, mut batch, mut bytes): (Vec<Vec<_>>, Vec<_>, usize) = (Vec::new(), Vec::new(), 0);
    for file in files {
        if !batch.is_empty() && bytes + file.1.len() > SMALL_BATCH {
            batches.push(std::mem::take(&mut batch));
            bytes = 0;
        }
        bytes += file.1.len();
        batch.push(file);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

fn files_of(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    if root.is_file() {
        return Ok(vec![(String::new(), root.to_path_buf())]);
    }
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                out.push((p.strip_prefix(root)?.to_string_lossy().into_owned(), p));
            }
        }
    }
    out.sort();
    Ok(out)
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.)
}

/// Set when this job process is told to stop (the CLI's stop, time budget or Ctrl-C signals the job's process
/// group): the request it is waiting on is cancelled instead of running on, billed, with nobody waiting.
static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn stopping() -> bool {
    STOP.load(std::sync::atomic::Ordering::SeqCst)
}

/// Submit `job`, wait for it, and bring results back. Progress goes to stdout (the job log).
pub fn exec(job: &Exec, tier: &str) -> Result<Outcome> {
    // One handler per process (a job worker runs one exec at a time); an error means one is set already.
    ctrlc::set_handler(|| STOP.store(true, std::sync::atomic::Ordering::SeqCst)).ok();
    let key = key()?;
    let http = client()?;
    let version = app_version();
    // A side GPU app for A/B tests of fal/app.py (the stage app and its /data scope stay shared).
    let gpu_version = std::env::var("SPLAT_FAL_GPU_APP_VERSION").unwrap_or_else(|_| version.clone());
    let app = format!("splat-gpu-{}-{tier}-{gpu_version}", job.gpu);
    let stage_app = format!("splat-stage-{tier}-{version}");
    let stager =
        Stager { http: &http, key: &key, app: &stage_app, scope: stage_scope(&key, job.ledger.as_deref()), chunk: PUBLISH_CHUNK };

    // Also covers standalone tool workers, which do not go through `vauss run`.
    warm_upload_stage(tier);
    // Inputs, per file and content-addressed; the runner deletes them when the job ends, so every
    // job stages its own (the colmap lock too).
    let mut inputs = Vec::new();
    // Small files are read once, hashed and later uploaded from that same copy: run files can change while
    // a job stages (its own log under jobs/), and a file uploaded after it changed fails the runner's sha256
    // check every time (2026-10-09: evaluate jobs failed with put_raw HTTP 500).
    let mut blobs: Vec<(String, PathBuf, u64, Option<Vec<u8>>)> = Vec::new();
    let mut remote_only: Vec<(String, PathBuf)> = Vec::new();
    for input in &job.inputs {
        let mut files = Vec::new();
        for (rel, path) in files_of(&input.path)? {
            let meta = fs::metadata(&path)?;
            if let Some((sha, _)) = remote_stub(&path) {
                // A result kept on fal: the job reads the staged copy; it cannot be uploaded from here.
                files.push(json!({"path": rel, "sha256": sha, "mode": 0o644}));
                remote_only.push((sha, path));
                continue;
            }
            let (sha, data) = if meta.len() <= STAGE_CHUNK as u64 {
                use sha2::{Digest, Sha256};
                let data = fs::read(&path)?;
                (format!("{:x}", Sha256::digest(&data)), Some(data))
            } else {
                (sha256_file(&path)?, None)
            };
            #[cfg(unix)]
            let mode = { use std::os::unix::fs::PermissionsExt; meta.permissions().mode() & 0o777 };
            #[cfg(not(unix))]
            let mode = 0o644;
            let len = data.as_ref().map_or(meta.len(), |d| d.len() as u64);
            files.push(json!({"path": rel, "sha256": sha, "mode": mode}));
            blobs.push((sha, path, len, data));
        }
        inputs.push(json!({"name": input.name, "is_file": input.path.is_file(), "files": files}));
    }
    let colmap_lock = if job.colmap {
        let path = std::env::temp_dir().join("splat-colmap-linux-explicit.txt");
        fs::write(&path, COLMAP_LOCK)?;
        let sha = sha256_file(&path)?;
        blobs.push((sha.clone(), path, COLMAP_LOCK.len() as u64, Some(COLMAP_LOCK.as_bytes().to_vec())));
        json!({"url": "", "sha256": sha, "name": "colmap-lock.txt"})
    } else {
        Value::Null
    };
    let t0 = Instant::now();
    let hedged0 = hedge_counts();
    let shas: Vec<String> = blobs.iter().map(|b| b.0.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    if let Some(ledger) = &job.ledger {
        ledger_add(ledger, &stage_app, &stager.scope, shas.iter().cloned())?;
    }
    let remote_shas: Vec<String> = remote_only.iter().map(|r| r.0.clone()).collect();
    let have = stager.have(&[shas.clone(), remote_shas].concat())?;
    let have_s = t0.elapsed().as_secs_f64();
    for (sha, path) in remote_only.iter().filter(|(sha, _)| !have.contains(sha)) {
        // The runner reads an input missing from the run's scope from the account's blob store (fal/app.py
        // `src` -> `_fetch`), where devtools/prune/fal_keep_cited.py copies cited runs' kept results.
        match kept_blob_for(job.ledger.as_deref(), path, sha) {
            Some(blob) => println!("[fal] {}: staged copy expired; the job reads the kept copy {blob}", path.display()),
            None if inherited_stub(path, sha, &stager.scope) => bail!("{} is kept on fal in the scope of the run this \
                one was copied from; a job reads only its own run's staged files. Download the copy's kept results \
                first (`vauss pull <run>`), or copy runs with their real files", path.display()),
            None => bail!("{} is kept on fal and its staged copy is gone (staged files expire after 24 h; only cited \
                runs' results are kept, in {KEPT_BLOBS}); rerun training", path.display()),
        }
    }
    // Small files go up whole, big ones in chunks, STAGE_PARALLEL at a time; each big file is committed
    // once all its chunks are up.
    let (mut new, mut new_bytes) = (0, 0u64);
    let mut done = std::collections::BTreeSet::new();
    let (mut uploads, mut bigs) = (Vec::new(), Vec::new());
    let mut largest: Vec<(u64, &Path)> = Vec::new();
    let mut small: Vec<(&str, &[u8])> = Vec::new();
    for (sha, path, len, data) in &blobs {
        if have.contains(sha) || !done.insert(sha.clone()) {
            continue;
        }
        new += 1;
        new_bytes += len;
        largest.push((*len, path));
        let len = *len as usize;
        if let Some(data) = data.as_deref().filter(|d| d.len() <= SMALL_FILE) {
            small.push((sha.as_str(), data));
        } else if let Some(data) = data.as_deref().filter(|d| d.len() <= stager.chunk) {
            uploads.push(Upload::File { sha, data });
        } else if let Some(data) = data {
            let parts: Vec<_> = data.chunks(stager.chunk).collect();
            bigs.push((sha.as_str(), parts.len()));
            uploads.extend(parts.into_iter().enumerate().map(|(index, data)| Upload::Part { sha, index, data }));
        } else {
            let chunks = len.div_ceil(stager.chunk);
            for index in 0..chunks {
                uploads.push(Upload::Chunk { sha, path, index, len: stager.chunk.min(len - index * stager.chunk) });
            }
            bigs.push((sha.as_str(), chunks));
        }
    }
    let small_files = small.len();
    uploads.extend(small_batches(small).into_iter().map(Upload::Small));
    // Small batches first (they go to the ctl twin, not the upload app's streams), then biggest first, so a large
    // file does not start last and run alone.
    uploads.sort_by_key(|u| std::cmp::Reverse(match u {
        Upload::Small(_) => usize::MAX,
        Upload::File { data, .. } | Upload::Part { data, .. } => data.len(),
        Upload::Chunk { len, .. } => *len,
    }));
    let requests = uploads.len() + bigs.len();
    let small_upload = new_bytes <= HEDGE_JOB_BYTES;
    let sent = std::sync::atomic::AtomicU64::new(0);
    let last_note = std::sync::Mutex::new(Instant::now());
    parallel(&uploads, STAGE_PARALLEL, |u| {
        ensure!(!stopping(), "stopped while staging inputs");
        let bytes = match u {
            Upload::Small(files) => {
                let bytes = files.iter().map(|f| f.1.len() as u64).sum();
                stager.put_small_hedged(files)?;
                bytes
            }
            Upload::File { sha, data } if small_upload => stager.put_whole(sha, data).map(|_| data.len() as u64)?,
            Upload::File { sha, data } => stager.put_raw(sha, None, data.to_vec()).map(|_| data.len() as u64)?,
            Upload::Part { sha, index, data } => stager.put_raw(sha, Some(*index), data.to_vec()).map(|_| data.len() as u64)?,
            Upload::Chunk { sha, path, index, len } => stager.put_chunk(path, sha, *index, *len).map(|_| *len as u64)?,
        };
        let sent = sent.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed) + bytes;
        let mut last = last_note.lock().unwrap();
        if last.elapsed() > Duration::from_secs(30) {
            println!("[fal] staging: {:.1} of {:.1} MB in {:.0}s", sent as f64 / 1e6, new_bytes as f64 / 1e6,
                t0.elapsed().as_secs_f64());
            *last = Instant::now();
        }
        Ok(())
    })?;
    parallel(&bigs, STAGE_PARALLEL, |(sha, chunks)| match stager.commit(sha, *chunks) {
        // Chunks gone before the commit (the stage sweep, or a stage app with the shared legacy parts dir): send again.
        Err(e) if format!("{e:#}").contains("retryable") => {
            println!("[fal] {sha}: {e:#}; sending its chunks again");
            for u in uploads.iter().filter(|u| matches!(u, Upload::Part { sha: s, .. } | Upload::Chunk { sha: s, .. } if s == sha)) {
                match u {
                    Upload::Part { index, data, .. } => stager.put_raw(sha, Some(*index), data.to_vec())?,
                    Upload::Chunk { path, index, len, .. } => stager.put_chunk(path, sha, *index, *len)?,
                    Upload::File { .. } | Upload::Small(_) => {}
                }
            }
            stager.commit(sha, *chunks)
        }
        r => r,
    })?;
    let (hedges, hedge_wins) = (hedge_counts().0 - hedged0.0, hedge_counts().1 - hedged0.1);
    println!("[fal] inputs: {} files, staged {new} new ({:.1} MB) privately on fal /data in {:.1}s \
        (have-check {have_s:.1}s, {requests} upload requests, {small_files} small via ctl, {hedges} hedged, \
        {hedge_wins} answered by the hedge)",
        blobs.len(), new_bytes as f64 / 1e6, t0.elapsed().as_secs_f64());
    if let Some(marker) = &job.staged_marker {
        fs::write(marker, "").ok();
    }
    // What a re-upload was: a later job of the run should find earlier outputs staged already.
    largest.sort_by_key(|l| std::cmp::Reverse(l.0));
    for (len, path) in largest.iter().take(3).filter(|l| l.0 >= 8 << 20) {
        println!("[fal]   new: {} ({:.1} MB)", path.display(), *len as f64 / 1e6);
    }

    let mut attempts = 0;
    loop {
        attempts += 1;
        // drive.py's project hard stop (dev host only).
        check_budget()?;
        let submit_unix = now();
        let mut running_since: Option<f64> = None;
        let body = json!({
            "argv": job.argv, "inputs": inputs,
            "sync_back": job.inputs.iter().filter(|i| i.sync_back).map(|i| &i.name).collect::<Vec<_>>(),
            "env": job.env, "cwd": "{out}", "timeout_s": job.timeout_s, "autotune_cache": true,
            "colmap_lock": colmap_lock, "stage_scope": stager.scope, "stage_keep": job.ledger.is_some(),
            "client_submit_unix": submit_unix,
            "client_rev": format!("splat-{}", env!("CARGO_PKG_VERSION")), "client_proto": CLIENT_PROTO,
        });
        let submitted: Value = retry(
            "fal submit",
            || Ok(http.post(format!("{QUEUE}/{owner}/{app}/exec", owner = owner()))
                .header("Authorization", format!("Key {key}"))
                .header("X-Fal-Store-IO", "0")
                .json(&body)
                .send()?),
            |r| Ok(r.json()?),
        )?;
        let request_id = submitted["request_id"].as_str().unwrap_or("?").to_string();
        println!("[fal] request {request_id} on {app}");
        job.track.submitted(&request_id, &app, attempts);
        // Every way out of this request (result, error, runner refusal) closes its submitted.jsonl row.
        let resp = (|| -> Result<Value> {
        let status_url = submitted["status_url"].as_str().context("fal submit: no status_url")?.to_string();
        let response_url = submitted["response_url"].as_str().context("fal submit: no response_url")?.to_string();
        let mut last = String::new();
        let mut seen_logs = std::collections::BTreeSet::new();
        // Polls that fail on the network are retried, for as long as RESULT_PASS_WAITS_S: the request runs on.
        let mut unreachable_since: Option<Instant> = None;
        loop {
            let st: Value = match retry(
                "fal status",
                || Ok(http.get(&status_url).query(&[("logs", "1")]).header("Authorization", format!("Key {key}")).send()?),
                |r| Ok(r.json()?),
            ) {
                Ok(st) => {
                    unreachable_since = None;
                    st
                }
                Err(e) if !stopping() && !e.chain().any(|c| c.is::<Permanent>())
                    && unreachable_since.get_or_insert_with(Instant::now).elapsed().as_secs() < RESULT_PASS_WAITS_S.iter().sum() => {
                    println!("[fal] {request_id}: status unavailable ({e:#}); polling again in 30s");
                    thread::sleep(Duration::from_secs(30));
                    continue;
                }
                Err(e) => return Err(e),
            };
            let s = st["status"].as_str().unwrap_or("").to_string();
            let _ = crate::atomic_json(&job.progress_path, &json!({"status":s,"updated_at":crate::utc_now()}));
            // Current app buffers child stdout until completion. Future/deployed streaming apps
            // may expose it here; forward only recognized training rows, once per log record.
            for log in st["logs"].as_array().into_iter().flatten() {
                let message = log["message"].as_str().unwrap_or("");
                if (message.contains("PROGRESS ") || message.contains("step "))
                    && seen_logs.insert(log.to_string()) {
                    if let Some(i) = message.find("PROGRESS ") { println!("{}", &message[i..]); }
                    else { println!("{message}"); }
                }
            }
            if s == "IN_PROGRESS" && running_since.is_none() {
                running_since = Some(now());
            }
            if s != last {
                println!("[fal] {request_id}: {s}");
                last = s.clone();
            }
            if s == "COMPLETED" {
                break;
            }
            if stopping() {
                // Queued: fal's cancel drops it. Running: the runner polls its abort flag and kills the job.
                if let Some(url) = submitted["cancel_url"].as_str() {
                    http.put(url).header("Authorization", format!("Key {key}")).send().ok();
                }
                let aborted = stager.call(json!({"op": "abort", "name": request_id}));
                println!("[fal] stopped: cancelled {request_id}{}",
                    aborted.err().map(|e| format!(" (abort flag not written: {e:#})")).unwrap_or_default());
                bail!("stopped: fal request {request_id} cancelled");
            }
            // Completion is on the first-view critical path twice (SfM and preview).
            // A three-second poll adds up to six seconds of idle time between them.
            thread::sleep(Duration::from_millis(500));
        }
        passes(&format!("{request_id}: result"), &result_waits(), || retry(
            "fal result",
            || Ok(http.get(&response_url).header("Authorization", format!("Key {key}")).send()?),
            |r| Ok(r.json()?),
        ))
        })();
        let how = match &resp {
            Ok(r) if r["error"].is_null() => "fetched".to_string(),
            Ok(r) if r["retry_other_runner"].as_bool() == Some(true) => "retry_other_runner".into(),
            Ok(_) => "runner error".into(),
            Err(e) => format!("client error: {}", format!("{e:#}").chars().take(200).collect::<String>()),
        };
        // Ledger first: redispatch.py orphans an ended request that has no spend row.
        let (kind, busy, cold, usd) = match resp.as_ref().ok().and_then(|r| bill(&job.gpu, r, submit_unix)) {
            Some((busy, cold, usd)) => ("exec", busy, cold, usd),
            None => {
                let busy = running_since.map_or(0., |t| (now() - t).max(0.));
                (if stopping() { "exec-cancelled" } else if resp.is_ok() { "exec" } else { "exec-unfinished" },
                    busy, false, price_per_s(&job.gpu) * busy)
            }
        };
        job.track.spent(&request_id, &app, &job.gpu, kind, &how, busy, cold, usd);
        job.track.ended(&request_id, &how);
        let resp = resp?;
        if let Some(err) = resp["error"].as_str() {
            if resp["retry_other_runner"].as_bool() == Some(true) && attempts < 4 {
                println!("[fal] runner refused ({err}); retrying on another runner");
                continue;
            }
            bail!("fal runner error: {}", err.chars().rev().take(3000).collect::<String>().chars().rev().collect::<String>());
        }
        if let Some(ledger) = &job.ledger {
            let outs = resp["staged_outputs"].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from));
            ledger_add(ledger, &stage_app, &stager.scope, outs.collect::<Vec<_>>())?;
        }
        let _ = crate::atomic_json(&job.progress_path, &json!({"status":"FETCHING_OUTPUTS","updated_at":crate::utc_now()}));
        return finish(&stager, job, &resp, submit_unix).with_context(|| format!(
            "fal request {request_id} finished (exit {}) but its results could not be fetched; they stay on fal /data for 24 h",
            resp["exit_code"]));
    }
}

fn gunzip_b64(v: &Value) -> Vec<u8> {
    use base64::Engine;
    let Some(s) = v.as_str() else { return Vec::new() };
    let Ok(gz) = base64::engine::general_purpose::STANDARD.decode(s) else { return Vec::new() };
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut out).ok();
    out
}

/// Where a result entry (`io/<input>/<rel>`) goes locally: a sync-back input's file.
fn sync_target(job: &Exec, path: &str) -> Option<PathBuf> {
    let (name, rel) = path.strip_prefix("io/")?.split_once('/')?;
    let input = job.inputs.iter().find(|i| i.name == name && i.sync_back)?;
    (!rel.split('/').any(|c| c == ".." || c.is_empty())).then(|| input.path.join(rel))
}

fn write_atomic(target: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = target.with_file_name(format!(".{name}.fal-tmp")); // unique per file: fetches run in parallel
    write(&tmp)?;
    fs::rename(&tmp, target)?;
    Ok(())
}

fn finish(stager: &Stager, job: &Exec, resp: &Value, submit: f64) -> Result<Outcome> {
    use base64::Engine;
    let mut resp = resp.clone();
    let mut synced = 0;
    if let Some(b) = resp["result_tgz"].as_str() {
        let tgz = base64::engine::general_purpose::STANDARD.decode(b)?;
        let archive = || tar::Archive::new(flate2::read::GzDecoder::new(&tgz[..]));
        // Python's tarfile stores a file whose inode it already added as a hard link to the first path. Unpacking
        // that entry alone links to a path that exists only in the archive and fails ("No such file or directory
        // when hard linking io/run/jobs/<other job>/runner.log", phone-halloween eval, 2026-10-12): read the
        // linked-to entries' bytes in a first pass and write each linked file from them.
        let mut linked: BTreeMap<String, Option<Vec<u8>>> = BTreeMap::new();
        for entry in archive().entries()? {
            let entry = entry?;
            if entry.header().entry_type().is_hard_link() && sync_target(job, &entry.path()?.to_string_lossy()).is_some() {
                if let Some(to) = entry.link_name()? {
                    linked.insert(to.to_string_lossy().into_owned(), None);
                }
            }
        }
        for entry in archive().entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_string_lossy().into_owned();
            if let Some(slot) = linked.get_mut(&path).filter(|_| entry.header().entry_type().is_file()) {
                let mut data = Vec::new();
                entry.read_to_end(&mut data)?;
                if let Some(target) = sync_target(job, &path) {
                    write_atomic(&target, |tmp| Ok(fs::write(tmp, &data)?))?;
                    synced += 1;
                }
                *slot = Some(data);
                continue;
            }
            let Some(target) = sync_target(job, &path) else { continue };
            if entry.header().entry_type().is_hard_link() {
                let to = entry.link_name()?.map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
                let Some(Some(data)) = linked.get(&to) else { bail!("fal result {path}: hard link to {to}, not in the archive") };
                write_atomic(&target, |tmp| Ok(fs::write(tmp, data)?))?;
            } else {
                write_atomic(&target, |tmp| Ok(entry.unpack(tmp).map(drop)?))?;
            }
            synced += 1;
        }
    }
    // Big results stay on the private /data volume: read back what we need (each distinct content once,
    // e.g. checkpoint.ply and splat.ply of one candidate are the same bytes; small files batched into one
    // request each, big ones chunk by chunk; FETCH_PARALLEL requests in flight), then delete the directory.
    if let Some(dir) = resp["result_data_path"].as_str().map(String::from) {
        use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};
        let t0 = Instant::now();
        type Uses = Vec<(String, Option<String>, Option<PathBuf>)>;
        let mut wanted: BTreeMap<String, (usize, Uses)> = BTreeMap::new();
        // The full model's PLYs stay on fal when the run keeps its staging (they are staged outputs): stubs here.
        let staged: std::collections::BTreeSet<&str> =
            resp["staged_outputs"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        let mut kept = Vec::new();
        // Entries marked "staged" were not copied into the results dir: the runner staged them once, under the
        // scope, and they read back as "staged:<sha>".
        let mut by_stage = std::collections::BTreeSet::new();
        // Old train workers did not return ready-to-use JPEG previews. Their PNGs
        // must still download so the local finalizer can make those previews.
        let preview_roots: std::collections::BTreeSet<&str> = resp["result_files"].as_array().into_iter().flatten()
            .filter_map(|f| f["path"].as_str()?.strip_suffix("/previews.json")).collect();
        for f in resp["result_files"].as_array().cloned().unwrap_or_default() {
            if let (Some(true), Some(sha)) = (f["staged"].as_bool(), f["sha256"].as_str()) {
                by_stage.insert(sha.to_string());
            }
            let Some(path) = f["path"].as_str() else { continue };
            let field = f["field"].as_str().map(String::from);
            let target = sync_target(job, path);
            if field.is_none() && target.is_none() {
                continue; // {out} files: nothing reads them locally
            }
            let rel = path.strip_prefix("io/").and_then(|p| p.split_once('/')).map(|(_, r)| r).unwrap_or("");
            if let (Some(target), Some(sha), Some(size), true) =
                (&target, f["sha256"].as_str(), f["size"].as_u64(), job.ledger.is_some() && kept_remote(rel, &job.keep_remote,
                    path.split_once("/eval/").is_some_and(|(root, _)| preview_roots.contains(root))))
            {
                if staged.contains(sha) {
                    write_atomic(target, |tmp| Ok(fs::write(tmp, format!("{STUB_MAGIC}{sha} {size}\n"))?))?;
                    kept.push((rel.to_string(), sha.to_string(), size));
                    synced += 1;
                    continue;
                }
            }
            let key = f["sha256"].as_str().map_or_else(|| format!("path:{path}"), String::from);
            let size = f["size"].as_u64().map_or(usize::MAX, |s| s as usize);
            wanted.entry(key).or_insert_with(|| (size, Vec::new())).1.push((path.to_string(), field, target));
        }
        if let (Some(ledger), false) = (&job.ledger, kept.is_empty()) {
            let record = ledger.with_file_name(REMOTE_KEPT);
            let mut all: Value = fs::read(&record).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_else(|| json!({}));
            for (rel, sha, size) in &kept {
                // fal/app.py _stage_path: the run's private staging dir on the account's /data volume.
                all[rel.as_str()] = json!({"name": rel.rsplit('/').next(), "sha256": sha, "size": size,
                    "fal_path": format!("/data/splat-fal/product/{}/{sha}", stager.scope), "app": stager.app,
                    "scope": stager.scope});
            }
            write_atomic(&record, |tmp| Ok(fs::write(tmp, serde_json::to_vec_pretty(&all)?)?))?;
            println!("[fal] kept on fal (not downloaded): {} files, {:.0} MB", kept.len(),
                kept.iter().map(|k| k.2).sum::<u64>() as f64 / 1e6);
        }
        let wanted: Vec<_> = wanted.into_iter().collect();
        enum Fetch {
            Many(Vec<usize>),
            /// Chunk `.1` of big file `.0` (size known).
            Chunk(usize, usize),
            One(usize),
        }
        let mut items = Vec::new();
        // Chunks still missing per big file; the request that brings the last one places the file.
        let mut parts: BTreeMap<usize, Vec<Option<Vec<u8>>>> = BTreeMap::new();
        let (mut batch, mut batch_bytes) = (Vec::new(), 0usize);
        for (i, (_, (size, _))) in wanted.iter().enumerate() {
            if *size == usize::MAX {
                items.push(Fetch::One(i));
                continue;
            }
            if *size > stager.chunk / 4 {
                let chunks = size.div_ceil(stager.chunk).max(1);
                items.extend((0..chunks).map(|c| Fetch::Chunk(i, c)));
                parts.insert(i, vec![None; chunks]);
                continue;
            }
            if batch_bytes + size > stager.chunk && !batch.is_empty() {
                items.push(Fetch::Many(std::mem::take(&mut batch)));
                batch_bytes = 0;
            }
            batch.push(i);
            batch_bytes += size;
        }
        if !batch.is_empty() {
            items.push(Fetch::Many(batch));
        }
        let (fields, written, bytes) = (Mutex::new(Vec::new()), AtomicUsize::new(0), AtomicUsize::new(0));
        let name = |i: usize| {
            let key = &wanted[i].0;
            if by_stage.contains(key) { format!("staged:{key}") } else { format!("{dir}/{}", wanted[i].1.1[0].0) }
        };
        let place = |i: usize, data: Vec<u8>| -> Result<()> {
            let (key, (_, uses)) = &wanted[i];
            let path = &uses[0].0;
            if !key.starts_with("path:") {
                use sha2::{Digest, Sha256};
                ensure!(&format!("{:x}", Sha256::digest(&data)) == key, "fal result {path}: sha256 mismatch");
            }
            bytes.fetch_add(data.len(), Ordering::Relaxed);
            // Identical files (a candidate's splat.ply and checkpoint.ply, 283 MB each on room) are one file on
            // disk: later copies are hardlinks of the first. Writers replace files by rename, never in place.
            let mut first: Option<&PathBuf> = None;
            for (_, field, target) in uses {
                if let Some(field) = field {
                    fields.lock().unwrap().push((field.clone(), serde_json::from_slice::<Value>(&data)?));
                } else if let Some(target) = target {
                    match first {
                        Some(src) => write_atomic(target, |tmp| match fs::hard_link(src, tmp) {
                            Ok(()) => Ok(()),
                            Err(_) => Ok(fs::write(tmp, &data)?),
                        })?,
                        None => write_atomic(target, |tmp| Ok(fs::write(tmp, &data)?))?,
                    }
                    first.get_or_insert(target);
                    written.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        };
        let parts = Mutex::new(parts);
        let slowest = Mutex::new(0f64);
        // Items fetched and placed: a later pass (after a network failure) fetches only the rest.
        let done = Mutex::new(std::collections::BTreeSet::new());
        let items: Vec<(usize, Fetch)> = items.into_iter().enumerate().collect();
        // One flat pool of chunk-sized requests: a gateway stream gives ~1.5 MB/s, and nested pools
        // (4 files x 4 chunks) kept only ~7 busy (room: 424 MB in 38.7 s).
        passes("fetching results", &result_waits(), || parallel(&items, FETCH_PARALLEL, |(k, item)| {
          if done.lock().unwrap().contains(k) {
              return Ok(());
          }
          let t = Instant::now();
          let r = match item {
            Fetch::One(i) => place(*i, stager.fetch(&name(*i))?),
            Fetch::Chunk(i, c) => {
                let size = wanted[*i].1.0;
                let (data, _) = stager.get_raw(&name(*i), c * stager.chunk)?;
                ensure!(data.len() == stager.chunk.min(size - c * stager.chunk), "fal get_raw {}: short chunk {c}", name(*i));
                let whole = {
                    let mut parts = parts.lock().unwrap();
                    let file = parts.get_mut(i).context("fal fetch: no parts")?;
                    file[*c] = Some(data);
                    if file.iter().all(Option::is_some) { parts.remove(i) } else { None }
                };
                match whole {
                    Some(file) => {
                        let chunks = file.len();
                        let placed = place(*i, file.into_iter().flatten().collect::<Vec<_>>().concat());
                        if placed.is_err() {
                            // A bad copy (sha256 mismatch): every chunk of the file comes again next pass.
                            parts.lock().unwrap().insert(*i, vec![None; chunks]);
                            let mut done = done.lock().unwrap();
                            for (k, _) in items.iter().filter(|(_, f)| matches!(f, Fetch::Chunk(j, _) if j == i)) {
                                done.remove(k);
                            }
                        }
                        placed
                    }
                    None => Ok(()),
                }
            }
            Fetch::Many(is) => {
                let datas = stager.get_many(&is.iter().map(|&i| name(i)).collect::<Vec<_>>())?;
                is.iter().zip(datas).try_for_each(|(&i, data)| place(i, data))
            }
          };
          if r.is_ok() {
              done.lock().unwrap().insert(*k);
          }
          let mut slowest = slowest.lock().unwrap();
          *slowest = slowest.max(t.elapsed().as_secs_f64());
          r
        }))?;
        for (field, v) in fields.into_inner().unwrap() {
            resp[field] = v;
        }
        synced += written.into_inner();
        // Everything is here: a failed cleanup must not fail the job (the stage sweep ends the dir after 24 h).
        if let Err(e) = stager.delete(&dir) {
            println!("[fal] results fetched; could not delete them on fal ({e:#}); the stage sweep removes them");
        }
        println!("[fal] fetched {} result files ({:.1} MB, {} distinct) privately from fal /data in {:.1}s \
            ({} fetch groups, slowest {:.1}s)",
            wanted.iter().map(|w| w.1.1.len()).sum::<usize>(), bytes.into_inner() as f64 / 1e6, wanted.len(),
            t0.elapsed().as_secs_f64(), items.len(), slowest.into_inner().unwrap());
    }
    let t = &resp["timing"];
    let f = |k: &str| t[k].as_f64().unwrap_or(0.);
    let cold = f("setup_started_unix") >= submit - 1.;
    let setup = f("setup_done_unix") - f("setup_started_unix");
    if f("received_unix") > 0. {
        // Where a job's time outside the command went (queue includes a cold runner's start).
        println!("[fal] runner: queue {:.1}s{}, inputs {:.1}s, command {:.1}s, results {:.1}s",
            f("received_unix") - submit, if cold { format!(" (cold start, setup {setup:.1}s)") } else { String::new() },
            f("staged_unix") - f("received_unix"), f("finished_unix") - f("started_unix"), f("done_unix") - f("finished_unix"));
    }
    // Host contention over the command (app.py _runner_host): a slow job on a busy host vs a slow job.
    if let Some(h) = resp["runner_host"].as_object() {
        println!("[fal] runner host: {} cpu={:?}",
            h.iter().map(|(k, v)| format!("{k}={}", v.as_str().map_or_else(|| v.to_string(), String::from)))
                .collect::<Vec<_>>().join(" "),
            resp["cpu_model"].as_str().unwrap_or(""));
    }
    let cost = bill(&job.gpu, &resp, submit).map_or(0., |b| b.2);
    let gpu_label = resp["gpu_label"].as_str().unwrap_or(&job.gpu).to_string();
    let exit_code = resp["exit_code"].as_i64().map(|c| c as i32).unwrap_or(1);
    let outcome = Outcome {
        exit_code,
        stdout: gunzip_b64(&resp["stdout_gz"]),
        stderr: gunzip_b64(&resp["stderr_gz"]),
        cost_usd: (cost * 10000.).round() / 10000.,
        gpu_label,
        synced,
    };
    Ok(outcome)
}

#[cfg(test)]
mod retention_tests {
    use super::kept_remote;

    #[test]
    fn defer_only_artifact_eval_images_with_worker_previews() {
        let keep = ["splat.ply", "eval/gt/", "eval/renders/"].map(String::from);
        assert!(kept_remote("artifacts/candidate_1/splat.ply", &keep, false));
        assert!(kept_remote("artifacts/candidate_1/eval/gt/frame.png", &keep, true));
        assert!(kept_remote("artifacts/candidate_1/eval/renders/frame.png", &keep, true));
        for path in ["artifacts/candidate_1/web/scene.spz", "artifacts/candidate_1/previews/frame.jpg",
                     "artifacts/candidate_1/previews.json", "artifacts/eval_1/comparisons/frame.jpg",
                     "artifacts/candidate_1/eval/renders-extra/frame.png", "exports/eval/gt/frame.png"] {
            assert!(!kept_remote(path, &keep, true), "must download {path}");
        }
        assert!(!kept_remote("artifacts/candidate_1/eval/renders/frame.png", &keep, false), "older workers need local PNG previews");
        assert!(!kept_remote("artifacts/candidate_1/eval/renders/frame.png", &[], true), "fetch-all must download PNGs");
    }
}

#[cfg(test)]
mod stage_ctl_tests {
    use super::*;

    /// Live check of the ctl routing (`FAL_KEY` in the environment): prints which app control ops use, and that a
    /// have-check answers there. cargo test -p splat-core --lib stage_ctl_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn stage_ctl_probe() -> Result<()> {
        let (key, http) = (key()?, client()?);
        let app = format!("splat-stage-product-{}", app_version());
        let stager = Stager { http: &http, key: &key, app: &app, scope: stage_scope(&key, None), chunk: STAGE_CHUNK };
        let t = Instant::now();
        let have = stager.have(&[])?;
        println!("ctl app: {} (have-check ok, {} staged, {:.2}s)", stager.ctl(), have.len(), t.elapsed().as_secs_f64());
        Ok(())
    }
}

#[cfg(test)]
mod kept_blob_tests {
    use super::*;

    /// An expired staged copy of a kept result falls back to its blob-store copy (house-run-sel, night-B:
    /// "staged copy is gone", 2026-10-10) only when the run's record names the store path for that content.
    #[test]
    fn expired_stub_reads_the_recorded_blob_store_copy() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let run = tmp.path().join("run");
        let stub = run.join("jobs/job_1/remote-stage/artifacts/candidate_1/splat.ply");
        fs::create_dir_all(stub.parent().unwrap())?;
        let (sha, other) = ("a".repeat(64), "b".repeat(64));
        fs::write(&stub, format!("{STUB_MAGIC}{sha} 10\n"))?;
        let ledger = run.join(STAGE_LEDGER_NAME);
        let record = |entries: Value| fs::write(run.join(REMOTE_KEPT), serde_json::to_vec(&entries).unwrap());
        // Not kept (no blob_path): no fallback.
        record(json!({"artifacts/candidate_1/splat.ply": {"sha256": sha, "fal_path": "/data/splat-fal/product/s/x"}}))?;
        assert_eq!(kept_blob_for(Some(&ledger), &stub, &sha), None);
        // Kept by fal_keep_cited.py.
        let blob = format!("{KEPT_BLOBS}/{sha}");
        record(json!({"artifacts/candidate_1/splat.ply": {"sha256": sha, "blob_path": blob}}))?;
        assert_eq!(kept_blob_for(Some(&ledger), &stub, &sha).as_deref(), Some(blob.as_str()));
        // Found from the stub's run when the job has no ledger.
        assert_eq!(kept_blob_for(None, &stub, &sha).as_deref(), Some(blob.as_str()));
        // Another content's blob, or a path the runner does not read, never stands in.
        assert_eq!(kept_blob_for(Some(&ledger), &stub, &other), None);
        record(json!({"artifacts/candidate_1/splat.ply": {"sha256": sha, "blob_path": format!("/elsewhere/{sha}")}}))?;
        assert_eq!(kept_blob_for(Some(&ledger), &stub, &sha), None);
        Ok(())
    }
}

#[cfg(test)]
mod staged_release_tests {
    use super::*;

    fn shas(l: &Path) -> Vec<String> {
        let l: Value = serde_json::from_slice(&fs::read(l).unwrap()).unwrap();
        l["shas"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
    }

    /// A run copied with its staging ledger (cp -a) never drops the source's staged shas: its release is a no-op
    /// until it stages files itself, and then drops only those, under its own scope (e2e-lo plant-cpoff/-cpon,
    /// 2026-10-10: copies of plant-0 sharing its ledger and kept record).
    #[test]
    fn staged_release_drops_only_the_runs_own_scope() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let (orig, copy) = (tmp.path().canonicalize()?.join("orig"), tmp.path().canonicalize()?.join("copy"));
        fs::create_dir_all(&orig)?;
        fs::create_dir_all(&copy)?;
        let (key, app) = ("k", "splat-stage-product-v21");
        let (a, b, c) = ("a".repeat(64), "b".repeat(64), "c".repeat(64));
        let orig_ledger = orig.join(STAGE_LEDGER_NAME);
        let orig_scope = stage_scope(key, Some(&orig_ledger));
        ledger_add(&orig_ledger, app, &orig_scope, [a.clone(), b.clone()])?;
        // b is a kept result (a stub): it stays staged at run end.
        let record = json!({"artifacts/mesh_1/mesh.glb": {"sha256": b, "scope": orig_scope, "app": app}});
        fs::write(orig.join(REMOTE_KEPT), serde_json::to_vec(&record)?)?;
        let l: Value = serde_json::from_slice(&fs::read(&orig_ledger)?)?;
        assert_eq!(release_plan(&orig_ledger, &l, key), Some((orig_scope.clone(), vec![a.clone()])));

        // The copy: same ledger and record bytes, another path.
        let copy_ledger = copy.join(STAGE_LEDGER_NAME);
        fs::copy(&orig_ledger, &copy_ledger)?;
        fs::copy(orig.join(REMOTE_KEPT), copy.join(REMOTE_KEPT))?;
        let l: Value = serde_json::from_slice(&fs::read(&copy_ledger)?)?;
        assert_eq!(release_plan(&copy_ledger, &l, key), None);
        // A ledger without a scope (before per-run scopes) is never dropped either.
        assert_eq!(release_plan(&copy_ledger, &json!({"app": app, "shas": [a]}), key), None);

        // Its first job records its own scope and starts the sha list over.
        let copy_scope = stage_scope(key, Some(&copy_ledger));
        assert_ne!(copy_scope, orig_scope);
        ledger_add(&copy_ledger, app, &copy_scope, [c.clone()])?;
        assert_eq!(shas(&copy_ledger), vec![c.clone()]);
        let l: Value = serde_json::from_slice(&fs::read(&copy_ledger)?)?;
        assert_eq!(release_plan(&copy_ledger, &l, key), Some((copy_scope.clone(), vec![c.clone()])));
        // Later jobs of the same run keep adding.
        ledger_add(&copy_ledger, app, &copy_scope, [a.clone()])?;
        assert_eq!(shas(&copy_ledger), vec![a.clone(), c.clone()]);

        // The copy's inherited stub is recognised as the source's (materialize reads it from that scope).
        let stub = copy.join("jobs/job_1/remote-stage/artifacts/mesh_1/mesh.glb");
        fs::create_dir_all(stub.parent().unwrap())?;
        fs::write(&stub, format!("{STUB_MAGIC}{b} 10\n"))?;
        assert!(inherited_stub(&stub, &b, &copy_scope));
        assert!(!inherited_stub(&stub, &b, &orig_scope));
        assert_eq!(kept_entry(&copy, &b).unwrap()["scope"], json!(orig_scope));
        Ok(())
    }
}

#[cfg(test)]
mod result_pass_tests {
    use super::*;

    /// A finished request's result survives network failures that outlast one retry round (a driveway job: remote
    /// exit 0, then `fal get_raw … Connection timed out (os error 110)` failed the job).
    #[test]
    fn small_inputs_batch_up_to_the_limit() {
        let (a, b) = (vec![0u8; SMALL_BATCH / 2], vec![0u8; SMALL_BATCH / 2 + 1]);
        assert!(small_batches(Vec::new()).is_empty());
        let one = small_batches(vec![("x", &b"1"[..]), ("y", &b"22"[..]), ("z", &b""[..])]);
        assert_eq!(one.iter().map(Vec::len).collect::<Vec<_>>(), [3]);
        let split = small_batches(vec![("a", &a[..]), ("b", &a[..]), ("c", &b[..]), ("d", &b"4"[..]), ("e", &a[..])]);
        assert_eq!(split.iter().map(|g| g.iter().map(|f| f.0).collect::<String>()).collect::<Vec<_>>(), ["ab", "cd", "e"]);
    }

    #[test]
    fn passes_ride_out_transport_errors_but_not_permanent_ones() {
        let waits = [Duration::ZERO; 3];
        let mut calls = 0;
        let r = passes("fetch", &waits, || {
            calls += 1;
            if calls < 3 { bail!("fal get_raw: Connection timed out (os error 110) (gave up after 5 attempts)") } else { Ok(calls) }
        });
        assert_eq!(r.unwrap(), 3);
        // Every pass fails: the last error comes back after 1 + waits.len() tries.
        calls = 0;
        let r: Result<()> = passes("fetch", &waits, || {
            calls += 1;
            bail!("timed out")
        });
        assert!(r.is_err());
        assert_eq!(calls, 4);
        // A 4xx answer is final, also under context.
        calls = 0;
        let r: Result<()> = passes("fetch", &waits, || {
            calls += 1;
            Err(anyhow::Error::from(Permanent("fal get_raw: HTTP 404".into()))).context("fetching results")
        });
        assert!(format!("{:#}", r.unwrap_err()).contains("HTTP 404"));
        assert_eq!(calls, 1);
    }
}

#[cfg(test)]
mod hedge_tests {
    use super::*;

    const SLOW: Duration = Duration::from_secs(5);
    const AFTER: Duration = Duration::from_millis(50);

    #[test]
    fn a_stalled_request_is_answered_by_its_hedge() {
        let t0 = Instant::now();
        let r = hedged("test", "again", AFTER, false, || { thread::sleep(SLOW); Ok(1) }, || Ok(2));
        assert_eq!(r.unwrap(), 2);
        // The stalled first copy runs on, detached: the caller does not wait for it.
        assert!(t0.elapsed() < SLOW / 2);
    }

    #[test]
    fn a_prompt_answer_sends_nothing_again() {
        let second = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ran = second.clone();
        let r = hedged("test", "again", SLOW, false, || Ok(1), move || { ran.store(true, std::sync::atomic::Ordering::SeqCst); Ok(2) });
        assert_eq!(r.unwrap(), 1);
        thread::sleep(AFTER);
        assert!(!second.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn a_prompt_failure_falls_back_only_when_asked() {
        let r: Result<i32> = hedged("test", "again", SLOW, false, || bail!("HTTP 404: gone"), || Ok(2));
        assert!(format!("{:#}", r.unwrap_err()).starts_with("HTTP 404"), "callers read the first error as is");
        assert_eq!(hedged("test", "again", SLOW, true, || bail!("HTTP 500"), || Ok(2)).unwrap(), 2);
    }

    #[test]
    fn both_failing_returns_an_error_and_a_late_first_answer_still_counts() {
        // The first copy answers after the hedge went out (polls are 250 ms apart).
        let late = Duration::from_millis(600);
        let r: Result<i32> = hedged("test", "again", AFTER, false, move || { thread::sleep(late); bail!("first") }, || bail!("second"));
        assert!(r.is_err());
        let r = hedged("test", "again", AFTER, false, move || { thread::sleep(late); Ok(1) }, || { thread::sleep(SLOW); Ok(2) });
        assert_eq!(r.unwrap(), 1);
    }

    #[test]
    fn hedge_deadlines_scale_with_size() {
        assert!(hedge_after(4., 2 << 20, 500e3) > hedge_after(4., 256 << 10, 500e3));
        assert_eq!(hedge_after_op("have", &json!({})), Duration::from_millis(1500));
        assert_eq!(hedge_after_op("commit", &json!({"index": 5})), Duration::from_secs(15));
        assert!(whole_put_timeout(2 << 20) > hedge_after(4., 2 << 20, 500e3));
    }
}
