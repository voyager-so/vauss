//! Remote GPU jobs: with `SPLAT_GPU_REMOTE=fal`, a job's backend command runs on a fal GPU through
//! F1's `fal/drive.py exec` instead of on the local GPU. The agent loop, MCP server and CPU tools
//! stay local; only the detached job's argv moves.
//!
//! The remote side sees the same absolute paths as this host: the job's inputs are uploaded
//! (per-file, content-addressed, so unchanged files upload once; a many-file dir as one tar, see
//! [`pack_stage`]) and symlinked back to their local paths under `sh -c`, so backend argv and job JSON files need no rewriting. Outputs come back as
//! the new/changed files of a staged copy of the run directory and are merged into the run.
//! A fal failure fails the job; there is no local fallback.
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use splat_contracts::JobSpec;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

/// `SPLAT_GPU_REMOTE` value, when remote GPU execution is on.
pub fn mode() -> Option<String> {
    std::env::var("SPLAT_GPU_REMOTE").ok().filter(|v| !v.is_empty() && v != "0" && v != "local")
}

/// Tools whose jobs run on the remote GPU. Shot jobs (hosted video models, Blender) stay local.
/// `--cloud` users may have no usable GPU, so distractor masking (SAM on Vulkan) runs remotely too.
const REMOTE_TOOLS: &[&str] = &["reconstruct_sfm", "mask_distractors", "train_splats", "extract_mesh", "evaluate", "render_views", "select_objects", "inpaint_removed"];

/// Cloud training presets. Every splat-only `--cloud` run gets the MAX tier, [`BUDGET_PRESET`]
/// (cloud runners have 48-96 GB): SH at the trainer's scene-adaptive budget with its full ceiling
/// (O2's `max_budget`: unbounded outdoor scenes grow up to 8M, from a 2.0M base, so compact/indoor scenes get ~2.0M), at
/// least 30k steps. Garden, 3 same-batch pairs vs the 7k/1M default: +1.01 +/- 0.04 dB, LPIPS
/// better. Since X27 (2026-10-09) it trains up to 45k (250 steps per training image, at least
/// 20k: [`Preset::min_steps_for`]) with the 30k schedule stretched past 30k
/// (`stretch_schedule`): counter/kitchen/garden x 2 seeds vs same-exec 30k, standard PLY:
/// +0.184 / +0.188 / -0.045 dB, LPIPS better on all three, wall x1.35 (research/x27/RESULTS.md). Opt-ins via SPLAT_CLOUD_PRESET: `max-neural-2.5m` (perf A2's neural appearance at a
/// fixed 2.5M cap: wins PSNR, but SH at the same 2.5M beat it on LPIPS on 3/4 scenes with 20-70%
/// less wall, so it is not the default), `steps-30k` (the local adaptive budget, up to 2M, at
/// 30k) and `default` (the request's recipe, for paired A/B baselines). Mesh (surfel) goals keep
/// the default recipe: surfels at the 5M budget scored worse on pegasus.
pub struct Preset {
    pub name: &'static str,
    pub min_steps: u32,
    /// 0: the trainer sizes the budget to the scene.
    pub max_splats: u32,
    /// With `max_splats` 0: the full adaptive ceiling (8M) instead of the local one (2M).
    pub max_budget: bool,
    pub neural: bool,
    /// Past 30k steps, stretch the trainer's absolute-step schedule knobs (growth window, refine
    /// interval, SH degree interval) by total_steps / 30k, so the run keeps the 30k schedule's shape.
    pub stretch_schedule: bool,
    /// Non-zero: the step floor is this many steps per training image, between
    /// [`SCALED_STEPS_MIN`] and `min_steps`, so small captures don't overfit a long schedule.
    pub steps_per_image: u32,
}

/// Lowest step floor of a capture-scaled preset.
pub const SCALED_STEPS_MIN: u32 = 20_000;

impl Preset {
    /// The step floor for a run training on `train_images` images. truck84 (80 train images,
    /// MAX on L40, same SfM/split): 45k 232 s, held-out PSNR 14.32 / LPIPS-vgg 0.466; 30k 161 s,
    /// 15.07 / 0.444; 20k 88 s, 15.16 / 0.451. The 45k gain (X27) was measured on 185-279 image scenes.
    pub fn min_steps_for(&self, train_images: usize) -> u32 {
        if self.steps_per_image == 0 {
            return self.min_steps;
        }
        let scaled = u32::try_from(train_images).unwrap_or(u32::MAX).saturating_mul(self.steps_per_image);
        scaled.clamp(SCALED_STEPS_MIN.min(self.min_steps), self.min_steps)
    }
}
// Quality first: the default never trains below the 45k MAX recipe. Capture-scaled
// steps (250/image, 20k floor) cost 0.4-0.56 dB on room/night/driveway vs 45k on the same SfM.
// The id "max-budget-5m" is historical (the ceiling was 5M until 2026-10-24); it is kept because run records name it.
pub const BUDGET_PRESET: Preset =
    Preset { name: "max-budget-5m", min_steps: 45_000, max_splats: 0, max_budget: true, neural: false, stretch_schedule: true, steps_per_image: 0 };
pub const NEURAL_PRESET: Preset = Preset {
    name: "max-neural-2.5m",
    min_steps: 30_000,
    max_splats: 2_500_000,
    max_budget: false,
    neural: true,
    stretch_schedule: false,
    steps_per_image: 0,
};
pub const STEPS_PRESET: Preset =
    Preset { name: "steps-30k", min_steps: 30_000, max_splats: 0, max_budget: false, neural: false, stretch_schedule: false, steps_per_image: 0 };

/// The training preset for this process's run: MAX ([`BUDGET_PRESET`]) when GPU jobs go to the
/// cloud, unless SPLAT_CLOUD_PRESET names another one.
pub fn preset() -> Option<&'static Preset> {
    mode().and(selected_preset())
}

/// The preset SPLAT_CLOUD_PRESET selects for a cloud run (`enable` records it before the remote
/// mode is set).
fn selected_preset() -> Option<&'static Preset> {
    match std::env::var("SPLAT_CLOUD_PRESET").unwrap_or_default().as_str() {
        "default" => None,
        n if n == NEURAL_PRESET.name => Some(&NEURAL_PRESET),
        n if n == STEPS_PRESET.name => Some(&STEPS_PRESET),
        _ => Some(&BUDGET_PRESET),
    }
}

fn preset_name() -> &'static str {
    selected_preset().map_or("default", |p| p.name)
}

/// `vauss run --cloud --fetch-all` / `SPLAT_CLOUD_FETCH_ALL=1`: download every result of the run's fal jobs.
/// By default the big ones a viewable result does not need stay on fal /data (see [`kept_remote`]).
pub const FETCH_ALL_ENV: &str = "SPLAT_CLOUD_FETCH_ALL";

pub fn fetch_all() -> bool {
    std::env::var(FETCH_ALL_ENV).is_ok_and(|v| matches!(v.trim(), "1" | "true" | "yes"))
}

/// Result file names (under `artifacts/<id>/`) a `--cloud` job leaves on fal unless [`fetch_all`]: the full
/// model's PLYs (the run shows the candidate's web package, written on fal) and, when the goal is not a mesh,
/// the side mesh (283 MB x 2 and 2 x ~57 MB on a MAX run). Local readers download a stub's file on demand
/// (`fal::materialize`: web export fallback, preview, edit, shots / Blender export, `splat export-web`,
/// `vauss pull`).
fn kept_remote(run_root: &Path, tool: &str) -> Vec<String> {
    if fetch_all() {
        return Vec::new();
    }
    let mut keep = vec!["splat.ply".to_string(), "checkpoint.ply".to_string()];
    // The train worker returns small preview JPEGs. Keep its full-resolution held-out images
    // staged for evaluate, which scores them (LPIPS), so they do not delay web publication.
    if tool == "train_splats" {
        keep.extend(["eval/gt/", "eval/renders/"].map(String::from));
    }
    // COLMAP databases (up to ~1 GB per SfM attempt, and its frozen copy) stay on fal: nothing on the
    // client reads them, and a failed SfM keeps one per attempt for diagnosis (guildhall: 4 x 939 MB
    // synced back).
    if tool == "reconstruct_sfm" {
        keep.extend(["database.db", "frozen.db"].map(String::from));
    }
    let goal = fs::read(run_root.join("run.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v["goal"].as_str().map(str::to_owned)).unwrap_or_default();
    if !goal_wants_mesh(&goal) {
        keep.extend(["mesh.glb", "mesh.ply"].map(String::from));
    }
    keep
}

pub fn wants(spec: &JobSpec) -> bool {
    REMOTE_TOOLS.contains(&spec.tool.as_str())
}

/// Files of the run that only the local side may write; never staged or merged back.
fn local_only(rel: &Path) -> bool {
    let first = rel.components().next().map(|c| c.as_os_str().to_string_lossy().into_owned());
    let name = rel.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    name.ends_with(".lock")
        // Atomic-write temp files (`.state.json.<id>.tmp`) vanish mid-staging.
        || (name.starts_with('.') && name.ends_with(".tmp"))
        || (rel.components().count() == 1 && matches!(name.as_str(), "state.json" | "journal.jsonl"))
        || first.as_deref() == Some("agent")
}

/// Never uploaded: locks, driver transcripts, viewer exports and supervisor output. (state.json and the journal are staged so
/// tool jobs can reopen the run, but `local_only` keeps them from coming back.)
fn never_staged(rel: &Path) -> bool {
    let name = rel.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    name.ends_with(".lock")
        // Atomic-write temps (`.state.json.<id>.tmp`): the client renames them away mid-walk
        // (arkit-42447230 failed evaluate staging on one).
        || (name.starts_with('.') && name.ends_with(".tmp"))
        || rel.components().next().is_some_and(|c| ["agent", "exports", crate::supervise::LOG_DIR, "video-analysis"].iter().any(|n| c.as_os_str() == *n))
        || rel.components().any(|c| ["remote-stage", "remote-out", "remote-packs"].iter().any(|n| c.as_os_str() == *n))
}

/// Env forwarded to the remote command besides the job's own env: the CLI config the worker
/// inherited (tool and model paths), which tool jobs need to reopen the run.
fn forwarded_env() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| {
            (k.starts_with("SPLAT_") && !k.starts_with("SPLAT_FAL_") && k != "SPLAT_GPU_REMOTE" && k != "SPLAT_GPU_LOCK")
                || matches!(k.as_str(), "OMP_NUM_THREADS" | "RUST_LOG" | "RUST_BACKTRACE")
        })
        // The image's own GPU ONNX Runtime is used remotely (see ORT_PREAMBLE).
        .filter(|(k, _)| k != "SPLAT_ONNXRUNTIME")
        // A local pipeline-cache dir means nothing remotely; the job gets PIPELINE_CACHE (see run()).
        .filter(|(k, _)| k != "SPLAT_PIPELINE_CACHE_DIR")
        .collect()
}

fn strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|v| strings(v, out)),
        Value::Object(o) => o.values().for_each(|v| strings(v, out)),
        _ => {}
    }
}

/// Drop a leading `flock [opts] <lockfile>`: the remote GPU is ours alone.
fn strip_flock(argv: &[String]) -> Vec<String> {
    if argv.first().map(|a| Path::new(a).file_name() == Some("flock".as_ref())) != Some(true) {
        return argv.to_vec();
    }
    let mut i = 1;
    while i < argv.len() && argv[i].starts_with('-') {
        i += 1;
    }
    argv[(i + 1).min(argv.len())..].to_vec()
}

fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Hardlink (or copy across filesystems) `src` into `dst`, recursively. `tool_job`: skip a
/// candidate's resume checkpoint when its final splat.ply is there (evaluate, render and mesh read
/// splat.ply; only resumed training reads the checkpoint, and that is not a tool job). On MAX runs
/// that is ~0.8 GB less to upload per job. `sfm_images`: also stage a frames artifact's sfm_images/
/// (the larger photo copies only the SfM job reads; training and tools use the dataset's images/).
fn link_tree(src: &Path, dst: &Path, root: &Path, tool_job: bool, sfm_images: bool) -> Result<()> {
    if src.is_dir() {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let rel = entry.path().strip_prefix(root).map(Path::to_path_buf).unwrap_or_default();
            if never_staged(&rel) || (tool_job && entry.file_name() == "checkpoint.ply" && src.join("splat.ply").is_file())
                || (!sfm_images && is_sfm_images(&rel))
            {
                continue;
            }
            link_tree(&entry.path(), &dst.join(entry.file_name()), root, tool_job, sfm_images)?;
        }
    } else if src.is_file() {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        if fs::hard_link(src, dst).is_err() {
            match fs::copy(src, dst) {
                // Gone between the directory listing and the link (rewritten by the running client):
                // whatever replaced it is not part of this job's inputs.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && !src.exists() => {}
                r => {
                    r.with_context(|| format!("stage {}", src.display()))?;
                }
            }
        }
    }
    Ok(())
}

/// `artifacts/<frames id>/sfm_images` (see splat-brush extract_frames).
fn is_sfm_images(rel: &Path) -> bool {
    let parts: Vec<_> = rel.components().collect();
    parts.len() == 3 && parts[0].as_os_str() == "artifacts" && parts[2].as_os_str() == "sfm_images"
}

/// Dirs under `dir` (relative to `root`) that hold no entries at all.
fn empty_dirs_under(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(dir)?.peekable();
    if entries.peek().is_none() {
        out.push(dir.strip_prefix(root)?.to_path_buf());
    }
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() && !path.is_symlink() {
            empty_dirs_under(root, &path, out)?;
        }
    }
    Ok(())
}

/// A staged file's identity: size, mtime and (unix) inode.
type Stamp = (u64, Option<std::time::SystemTime>, u64);

fn stamp(meta: &fs::Metadata) -> Stamp {
    #[cfg(unix)]
    let ino = { use std::os::unix::fs::MetadataExt; meta.ino() };
    #[cfg(not(unix))]
    let ino = 0;
    (meta.len(), meta.modified().ok(), ino)
}

/// Stamps of every file in the stage before the job runs, so [`merge_back`] can tell the job's
/// outputs from inputs it never touched.
fn stage_stamps(dir: &Path, out: &mut BTreeMap<PathBuf, Stamp>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            stage_stamps(&entry.path(), out)?;
        } else if ty.is_file() {
            out.insert(entry.path(), stamp(&entry.metadata()?));
        }
    }
    Ok(())
}

/// Move new/changed files from the stage into the run. A staged input the job left as it was is
/// never merged: the run's copy may have changed since staging (e.g. collision rewrites a
/// candidate's web/scene.json while a later job that staged that candidate runs), and an
/// unchanged hardlink is the same file anyway. Returns the number of files merged.
fn merge_back(stage: &Path, run_root: &Path, dir: &Path, staged: &BTreeMap<PathBuf, Stamp>) -> Result<usize> {
    let mut n = 0;
    if !dir.is_dir() {
        return Ok(0);
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let rel = path.strip_prefix(stage)?.to_path_buf();
        // Only job outputs come back (artifacts, tool-job results); the run's own files and other
        // jobs' bookkeeping are written locally.
        let output = rel.starts_with("artifacts") || rel.starts_with("jobs/tool-inputs") || rel == Path::new("jobs");
        if local_only(&rel) || !output {
            continue;
        }
        let target = run_root.join(&rel);
        if path.is_dir() {
            fs::create_dir_all(&target)?;
            n += merge_back(stage, run_root, &path, staged)?;
            continue;
        }
        if staged.get(&path).is_some_and(|s| fs::metadata(&path).is_ok_and(|m| stamp(&m) == *s)) {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if let (Ok(a), Ok(b)) = (fs::metadata(&path), fs::metadata(&target)) {
                if a.ino() == b.ino() && a.dev() == b.dev() {
                    continue;
                }
            }
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        if fs::rename(&path, &target).is_err() {
            fs::copy(&path, &target).with_context(|| format!("merge {}", target.display()))?;
        }
        n += 1;
    }
    Ok(n)
}

/// The run-relative directory an input path belongs to: `artifacts/<id>` for artifact files,
/// else the top-level entry.
fn stage_key(rel: &Path) -> PathBuf {
    let parts: Vec<_> = rel.components().filter_map(|c| match c {
        Component::Normal(s) => Some(s.to_owned()),
        _ => None,
    }).collect();
    match parts.as_slice() {
        [first, second, ..] if first == "artifacts" => PathBuf::from(first).join(second),
        [first, ..] => PathBuf::from(first),
        [] => PathBuf::new(),
    }
}

/// A staged dir (per [`stage_key`]) with more than this many small files goes up as tars. The runner app places
/// staged files 16 at a time from fal's /data network volume: a few ms a file on a typical host, ~1.6 s on a slow one
/// (01a127c1: 297 files across 4 dirs, 472 s before the command ran). Up to 32 files is two such rounds, about what
/// the tars cost (requests and an extract), so smaller dirs stay per-file.
const PACK_MIN_FILES: usize = 32;

/// Bytes per tar: one whole-file upload. A bigger tar goes up in chunks and is joined on /data at ~0.3 s a chunk
/// (one 61.6 MB tar of showbus's 158 SfM inputs: 63 s to stage vs 29 s per-file); tars of this size go up in
/// parallel like the files they replace, and the runner reads ~1/6 as many files.
const PACK_TAR_BYTES: u64 = crate::fal::PUBLISH_CHUNK as u64;

/// A tar starts at each file with probability (its size / this), see [`pack_stage`]. SfM datasets of showbus, counter,
/// room and garden (158-312 small files): 3.2-5.6x fewer files to read, and 98-99% of the bytes packed for the SfM job
/// are in the same tars when training stages the dir again with the SfM outputs added (1 MB: 2.5-3.9x fewer, 98-100%;
/// no path cuts: up to 7.4x fewer, but 0-96%).
const PACK_TAR_TARGET: u64 = PACK_TAR_BYTES;

/// Files up to this size are packed; bigger ones stay per-file (their bytes, not the per-file cost, dominate).
const PACK_FILE_MAX: u64 = PACK_TAR_BYTES / 2;

/// One tar of a staged dir's small files: uploaded as an input (content-addressed like any other, so a later job of
/// the run sending the same files finds it staged), extracted into the run by the job's preamble.
struct Pack {
    tar: PathBuf,
    /// Staged path and sha256 of each member.
    members: Vec<(PathBuf, String)>,
}

/// Move the small files (up to [`PACK_FILE_MAX`]) of every staged dir holding more than [`PACK_MIN_FILES`] of them
/// out of `stage` into deterministic tars of up to [`PACK_TAR_BYTES`] under `out` (sorted, run-relative paths, zeroed
/// owner and mtime, the files' modes): the same files give the same tars, so the same sha256s. Big files and remote
/// stubs stay per-file; they are often staged on fal already as an earlier job's outputs.
///
/// `have` answers which sha256s the run's staging scope on fal holds already (`fal::run_staged`). Small files
/// staged there per-file (frames a video replay or an earlier per-file job staged) stay per-file inputs, which cost
/// no upload; packing them would send them again in new tars. The [`PACK_MIN_FILES`] threshold applies to what is
/// left. If `have` fails nothing is packed. Returns the packs and how many small files stayed per-file as staged.
fn pack_stage(stage: &Path, out: &Path, have: impl FnOnce(&[String]) -> Result<BTreeSet<String>>) -> Result<(Vec<Pack>, usize)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let p = entry?.path();
            if p.is_dir() {
                walk(&p, out)?;
            } else if p.is_file() {
                out.push(p);
            }
        }
        Ok(())
    }
    /// A member's bytes in the tar: header, a GNU long-name record for long paths, data, each 512-byte padded.
    fn entry_bytes(rel: &str, len: u64) -> u64 {
        let pad = |n: u64| n.div_ceil(512) * 512;
        512 + if rel.len() >= 100 { 512 + pad(rel.len() as u64 + 1) } else { 0 } + pad(len)
    }
    let mut files = Vec::new();
    walk(stage, &mut files)?;
    let mut groups: BTreeMap<PathBuf, Vec<(String, PathBuf, u64)>> = BTreeMap::new();
    for path in files {
        let rel = path.strip_prefix(stage)?.to_path_buf();
        let len = fs::metadata(&path)?.len();
        if rel.components().count() < 2 || len > PACK_FILE_MAX || crate::fal::remote_stub(&path).is_some() {
            continue;
        }
        groups.entry(stage_key(&rel)).or_default().push((rel.to_string_lossy().into_owned(), path, len));
    }
    groups.retain(|_, g| g.len() > PACK_MIN_FILES);
    if groups.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let mut hashed: BTreeMap<PathBuf, String> = BTreeMap::new();
    for (_, path, _) in groups.values().flatten() {
        hashed.insert(path.clone(), sha256(path)?);
    }
    let shas: Vec<String> = hashed.values().cloned().collect::<BTreeSet<_>>().into_iter().collect();
    let have = match have(&shas) {
        Ok(have) => have,
        Err(e) => {
            println!("[remote] fal staging check failed ({e:#}); not packing, every input stays per-file");
            return Ok((Vec::new(), 0));
        }
    };
    let mut kept = 0;
    for group in groups.values_mut() {
        let before = group.len();
        group.retain(|(_, path, _)| !have.contains(&hashed[path]));
        kept += before - group.len();
    }
    // A frame is also linked into each SfM dataset. Per-file staging deduplicates those aliases by
    // content, but putting each dataset into its own tar used to upload the pixels twice. Pack one
    // copy and restore the other paths with tar hardlinks after all regular packs are extracted.
    let mut representatives = BTreeMap::<String, PathBuf>::new();
    let mut aliases = Vec::new();
    let mut packs: Vec<Pack> = Vec::new();
    let mut groups: Vec<_> = groups.into_iter().filter(|(_, g)| g.len() > PACK_MIN_FILES).collect();
    // Training sends the SfM dataset again, but not the original frame artifact. Keep the regular
    // pixel entries at the SfM paths so its content-addressed archives remain reusable there.
    groups.sort_by_key(|(key, _)| (!key.file_name().is_some_and(|n| n.to_string_lossy().starts_with("sfm_")), key.clone()));
    for (_, mut group) in groups {
        group.sort();
        group.retain(|(rel, path, _)| {
            let sha = &hashed[path];
            if let Some(target) = representatives.get(sha) {
                aliases.push((rel.clone(), path.clone(), target.clone(), sha.clone()));
                false
            } else {
                representatives.insert(sha.clone(), path.clone());
                true
            }
        });
        // Consecutive runs of the sorted files, each tar (with its 1 KB end marker) within PACK_TAR_BYTES. A tar also
        // starts at a file whose path hash falls under its share of PACK_TAR_TARGET, so boundaries follow the paths,
        // not the position: files added to the dir (an SfM job's outputs, staged again by training) change only the
        // tars around them, and the rest are found staged instead of sent again.
        let mut chunks: Vec<Vec<(String, PathBuf, u64)>> = Vec::new();
        let mut size = u64::MAX;
        for file in group {
            use sha2::{Digest, Sha256};
            let bytes = entry_bytes(&file.0, file.2);
            let h = u64::from_le_bytes(Sha256::digest(file.0.as_bytes())[..8].try_into().unwrap());
            let cut = (h as f64 / u64::MAX as f64) < file.2 as f64 / PACK_TAR_TARGET as f64;
            if cut || size.saturating_add(bytes) > PACK_TAR_BYTES - 1024 {
                chunks.push(Vec::new());
                size = 0;
            }
            size += bytes;
            chunks.last_mut().unwrap().push(file);
        }
        for chunk in chunks {
            fs::create_dir_all(out)?;
            let tar = out.join(format!("p{}.tar", packs.len()));
            let mut builder = tar::Builder::new(std::io::BufWriter::new(fs::File::create(&tar)?));
            let mut members = Vec::new();
            for (rel, path, _) in chunk {
                use sha2::{Digest, Sha256};
                // Read once: the tar and the member's sha256 hold the same bytes even if the file changes meanwhile.
                let data = fs::read(&path)?;
                #[cfg(unix)]
                let mode = { use std::os::unix::fs::PermissionsExt; fs::metadata(&path)?.permissions().mode() & 0o777 };
                #[cfg(not(unix))]
                let mode = 0o644;
                let mut header = tar::Header::new_gnu();
                header.set_size(data.len() as u64);
                header.set_mode(mode);
                header.set_mtime(0);
                header.set_uid(0);
                header.set_gid(0);
                header.set_entry_type(tar::EntryType::Regular);
                builder.append_data(&mut header, &rel, &data[..])?;
                members.push((path, format!("{:x}", Sha256::digest(&data))));
            }
            builder.into_inner()?.into_inner().map_err(|e| e.into_error())?;
            packs.push(Pack { tar, members });
        }
    }
    if !aliases.is_empty() {
        let tar = out.join(format!("p{}.tar", packs.len()));
        let mut builder = tar::Builder::new(std::io::BufWriter::new(fs::File::create(&tar)?));
        let mut members = Vec::new();
        for (rel, path, target, sha) in aliases {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::Link);
            header.set_size(0);
            header.set_mode(0o644);
            header.set_mtime(0);
            header.set_uid(0);
            header.set_gid(0);
            builder.append_link(&mut header, rel, target.strip_prefix(stage)?)?;
            members.push((path, sha));
        }
        builder.into_inner()?.into_inner().map_err(|e| e.into_error())?;
        packs.push(Pack { tar, members });
    }
    for (path, _) in packs.iter().flat_map(|p| &p.members) {
        fs::remove_file(path)?;
    }
    Ok((packs, kept))
}

/// Run `spec` on the remote GPU from the job directory `dir`; returns the remote exit code.
/// stdout/stderr (this process's, i.e. the job's logs) carry drive.py's progress and the remote
/// command's output.
pub fn run(dir: &Path, spec: &JobSpec) -> Result<i32> {
    run_inner(dir, spec, true)
}

/// Longest a job waits for a still-running capture check (from the first job that waits): past it
/// the run goes on and the check reports late, so a slow check never holds a user for minutes.
const CAPTURE_WAIT_S: f64 = 20.0;

/// Holds a job's upload for the run's capture pre-check (`vauss run` runs it alongside the local
/// frame extraction) and refuses it when the check failed, so a video that won't work spends nothing.
fn capture_gate(run_root: &Path) -> Result<()> {
    use crate::report::{CAPTURE_CHECK, CAPTURE_PENDING};
    let pending = run_root.join(CAPTURE_PENDING);
    if pending.exists() {
        // The first waiter starts the shared clock.
        let clock = run_root.join("capture_check.wait");
        let _ = fs::OpenOptions::new().write(true).create_new(true).open(&clock);
        let since = fs::metadata(&clock).and_then(|m| m.modified()).ok();
        let left = || since.and_then(|t| t.elapsed().ok()).map_or(0.0, |e| CAPTURE_WAIT_S - e.as_secs_f64());
        while pending.exists() && left() > 0.0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    let check: Value = fs::read(run_root.join(CAPTURE_CHECK)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
    ensure!(
        check["verdict"] != "fail" || check["ignored"] == true,
        "Capture check: {} Nothing was uploaded (`vauss check --guide`; --ignore-check runs it anyway).",
        check["summary"].as_str().unwrap_or("this video won't work.")
    );
    Ok(())
}

/// The client-side phases of a job before its fal request (`[remote] pre-script: <phase> <secs>s` in the job's
/// stdout): Lego's first job spent 184 s between launch.json and remote.sh with no line saying where (2026-10-12).
struct Phases {
    start: std::time::Instant,
    last: std::time::Instant,
}

impl Phases {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Self { start: now, last: now }
    }

    fn done(&mut self, phase: &str) {
        println!("[remote] pre-script: {phase} {:.2}s", self.last.elapsed().as_secs_f64());
        self.last = std::time::Instant::now();
    }

    fn total(&self) {
        println!("[remote] pre-script: total {:.2}s", self.start.elapsed().as_secs_f64());
    }
}

fn run_inner(dir: &Path, spec: &JobSpec, replay_video: bool) -> Result<i32> {
    let mode = mode().unwrap_or_default();
    ensure!(mode == "fal", "SPLAT_GPU_REMOTE={mode:?}: only \"fal\" is supported");
    let mut phases = Phases::new();
    capture_gate(&dir.parent().and_then(Path::parent).context("job dir is not <run>/jobs/<id>")?.canonicalize()?)?;
    phases.done("capture gate");
    let _ = crate::atomic_json(dir.join("remote-progress.json"), &serde_json::json!({"status":"STAGING","updated_at":crate::utc_now()}));
    // Native fal client by default; SPLAT_FAL_DRIVE=<fal/drive.py> selects the dev path.
    let drive = std::env::var("SPLAT_FAL_DRIVE").ok().filter(|d| !d.is_empty());
    let native = drive.is_none();
    let drive = drive.unwrap_or_default();
    let python = std::env::var("SPLAT_FAL_PYTHON").unwrap_or_else(|_| "python3".into());
    let team = std::env::var("SPLAT_FAL_TEAM").unwrap_or_else(|_| "product".into());
    let lane = std::env::var("SPLAT_FAL_LANE").or_else(|_| std::env::var("SPLAT_LANE")).unwrap_or_else(|_| "agent".into());
    let gpu = std::env::var("SPLAT_FAL_GPU").unwrap_or_else(|_| DEFAULT_GPU.into());

    let run_root = dir.parent().and_then(Path::parent).context("job dir is not <run>/jobs/<id>")?.canonicalize()?;
    let argv = strip_flock(&spec.argv);
    ensure!(!argv.is_empty(), "empty argv after removing flock");
    let mut env: BTreeMap<String, String> = forwarded_env();
    // Host search paths stay local: the image has its own, and staging would upload every directory
    // they name (a job's PATH holding ~/.local/bin shipped ~1 GB of the user's tools to fal).
    env.extend(
        spec.env.iter().filter(|(k, _)| !["VK_ICD_FILENAMES", "PATH", "LD_LIBRARY_PATH"].contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())),
    );
    let tool_job = argv.get(1).map(String::as_str) == Some("__tool_job");

    // Every absolute path the command can reach: argv, env, JSON job files named in argv, and for
    // tool jobs (which reopen the run) the run's own config files.
    let mut tokens: Vec<String> = argv.iter().chain(env.values()).cloned().collect();
    let mut json_files: Vec<PathBuf> = argv.iter().map(PathBuf::from).filter(|p| p.extension().is_some_and(|e| e == "json") && p.is_file()).collect();
    if tool_job {
        json_files.extend(["run.json", "toolchain.json", "fixed.json"].iter().map(|f| run_root.join(f)).filter(|p| p.is_file()));
    }
    for f in &json_files {
        if let Ok(mut v) = serde_json::from_slice::<Value>(&fs::read(f)?) {
            // The capture itself: tool jobs reopen the run without verifying it (`__tool_job`), and
            // their inputs are its extracted frames. Staging it re-uploaded every photo (268 MB on
            // room) for each evaluate.
            if f.file_name().is_some_and(|n| n == "run.json") {
                v.as_object_mut().map(|o| o.remove("source"));
            }
            strings(&v, &mut tokens);
        }
    }

    let mut stage_keys: BTreeSet<PathBuf> = BTreeSet::new();
    let mut externals: BTreeSet<PathBuf> = BTreeSet::new();
    let mut lock_dirs: BTreeSet<String> = BTreeSet::new();
    let mut colmap: BTreeSet<PathBuf> = BTreeSet::new();
    // Only a job that names an sfm_images dir (the SfM job's sfm_job.json) uploads it.
    let mut sfm_images = false;
    for t in tokens.iter().flat_map(|t| t.split(':')) {
        let p = Path::new(t);
        if !p.is_absolute() || t.starts_with("/proc") || t.starts_with("/dev") {
            continue;
        }
        if t.ends_with(".lock") {
            // In-process GPU users (Brush) open their lock path from the job JSON; give it a home.
            lock_dirs.insert(parent_str(p));
            continue;
        }
        let canon = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        if let Ok(rel) = canon.strip_prefix(&run_root) {
            sfm_images |= rel.ancestors().any(is_sfm_images);
            stage_keys.insert(stage_key(rel));
            continue;
        }
        // COLMAP (Brush SfM) is provisioned by the fal app itself; never upload the 5.6 GB tool dir.
        if p.file_name().is_some_and(|n| n == "colmap") {
            if !tool_job {
                colmap.insert(p.to_path_buf());
            }
            continue;
        }
        // System tools exist in the image; shared libraries (onnxruntime) may not.
        let system = ["/usr/", "/bin/", "/etc/", "/lib", "/sbin/", "/opt/"].iter().any(|s| t.starts_with(s))
            || t.contains("onnxruntime");
        if p.exists() && (!system || (t.starts_with("/usr/lib") && p.is_file())) {
            externals.insert(p.to_path_buf());
        }
    }
    if let Some(id) = spec.meta.get("artifact_id").and_then(Value::as_str) {
        stage_keys.insert(PathBuf::from("artifacts").join(id));
    }
    if tool_job {
        // The tool reopens the run, which checks every job in state.json: stage all of it.
        stage_keys.clear();
        for entry in fs::read_dir(&run_root)? {
            let name = PathBuf::from(entry?.file_name());
            if !never_staged(&name) {
                stage_keys.insert(name);
            }
        }
    }
    stage_keys.remove(Path::new(""));

    let stage = dir.join("remote-stage");
    if stage.exists() {
        fs::remove_dir_all(&stage)?;
    }
    fs::create_dir_all(&stage)?;
    for key in &stage_keys {
        let src = run_root.join(key);
        if src.exists() {
            link_tree(&src, &stage.join(key), &run_root, tool_job, sfm_images)?;
        } else {
            fs::create_dir_all(stage.join(key))?;
        }
    }
    phases.done("staging walk and link_tree");

    let replay = if replay_video && cfg!(target_os = "linux") && matches!(argv.get(1).map(String::as_str), Some("__brush" | "__tool_job")) {
        let ledger = native.then(|| run_root.join(STAGE_LEDGER));
        prepare_video_replay(&run_root, &stage, Path::new(&argv[0]), &mut externals, ledger.as_deref(), REPLAY_MIN_SAVING)?
    } else { None };
    phases.done("video replay prep");
    let mut empty_dirs = Vec::new();
    empty_dirs_under(&stage, &stage, &mut empty_dirs)?;
    let mut staged = BTreeMap::new();
    stage_stamps(&stage, &mut staged)?;
    // Many-file dirs go up as one tar each (native client; drive.py packs inputs itself).
    let packs_dir = dir.join("remote-packs");
    fs::remove_dir_all(&packs_dir).ok();
    let (packs, kept) = if native {
        let ledger = run_root.join(STAGE_LEDGER);
        pack_stage(&stage, &packs_dir, |shas| crate::fal::run_staged(&ledger, shas).map(|(_, have)| have))?
    } else {
        (Vec::new(), 0)
    };
    if !packs.is_empty() || kept > 0 {
        let n: usize = packs.iter().map(|p| p.members.len()).sum();
        let bytes: u64 = packs.iter().filter_map(|p| fs::metadata(&p.tar).ok()).map(|m| m.len()).sum();
        println!("[remote] packed {n} files into {} tars ({:.1} MB; {kept} already staged, kept per-file)", packs.len(), bytes as f64 / 1e6);
    }
    phases.done("pack_stage");

    // Remote preamble: same absolute paths as here. The run root always points at the staged copy;
    // other inputs are linked where the image lacks them. fal reuses runners, so a symlink left by an
    // earlier job (possibly dangling) is replaced; real files in the image are kept.
    let run_str = run_root.to_string_lossy().into_owned();
    // fal's COLMAP build needs AVX-512, which only the RTX PRO 6000 runners guarantee.
    let gpu = if colmap.is_empty() { gpu } else { "rtxpro6000".to_string() };
    // Large files (the splat and Spirula binaries) live in fal's /data keyed by sha256, so a new
    // version uploads once in total, not once per machine: try without uploading, upload on a miss.
    let mut big = Vec::new();
    let mut small = Vec::new();
    for ext in &externals {
        if replay.as_ref().is_some_and(|r| r.sources.contains(ext)) {
            small.push((ext.clone(), ext.clone()));
        } else if ext.is_file() && (replay.as_ref().is_some_and(|r| r.cached.contains(ext)) || fs::metadata(ext)?.len() > BIG_FILE) {
            big.push((ext.clone(), crate::hashcache::sha256(ext)?));
        } else if ext.is_dir() && ext.join("lpips_vgg.onnx").is_file() {
            // The models dir: upload a hardlinked subset (LPIPS and small files such as the
            // detector vocab) and link it at the same path. Weights a job names in its argv (SAM,
            // Grounding DINO) go through the sha256 cache like any big file; DA3 stays local.
            let subset = dir.join("remote-models");
            fs::create_dir_all(&subset)?;
            for entry in fs::read_dir(ext)? {
                let entry = entry?;
                let name = entry.file_name();
                let n = name.to_string_lossy();
                let small_file = entry.metadata().is_ok_and(|m| m.is_file() && m.len() < 1 << 20);
                if n.starts_with("lpips") || small_file {
                    link_tree(&ext.join(&name), &subset.join(&name), ext, false, false)?;
                }
            }
            small.push((ext.clone(), subset));
        } else {
            small.push((ext.clone(), ext.clone()));
        }
    }
    phases.done("big-file hashing");
    let timeout = job_timeout_s();
    // The dev experiment registry must never gate a user's run: opt out whenever this drive.py has
    // one. Checked per job, since drive.py can change under a long run.
    let help = if native { String::new() } else {
        Command::new(&python).arg(&drive).args(["exec", "--help"]).output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default()
    };
    let no_exp = help.contains("--no-exp");
    let tier = std::env::var("SPLAT_FAL_TIER").unwrap_or_else(|_| "product".into());
    let mut code = -255;
    // Native client: ask fal which big files are already in the binary cache and upload the missing ones on
    // the first try (a miss used to cost a whole extra GPU request on a cold runner, exit MISS, ~2-3 min).
    let mut missing: Vec<bool> = if native && !big.is_empty() {
        let paths: Vec<String> =
            big.iter().map(|(ext, sha)| format!("{CACHE}/{sha}/{}", ext.file_name().unwrap().to_string_lossy())).collect();
        crate::fal::cached(&paths).map(|c| c.iter().map(|hit| !hit).collect()).unwrap_or_else(|_| vec![false; big.len()])
    } else {
        vec![false; big.len()]
    };
    phases.done("binary cache check");
    // Until the run's first-view job (the quick SfM) has its inputs on fal, this job's own inputs go up on
    // YIELD_STREAMS beside it, while the binary publish below waits on the host lock for the same binary that job
    // is publishing. The full SfM used to wait up to 60 s for that upload before even starting its own. Once the
    // priority job is staged, `exec` sends the rest at full width (its have-check skips what went up here).
    let early = if native && yields_upload(&run_root, Some(dir)) {
        let mut files = Vec::new();
        input_files(&stage, &mut files)?;
        files.extend(packs.iter().map(|p| p.tar.clone()));
        for (_, from) in &small {
            input_files(from, &mut files)?;
        }
        files.retain(|f| crate::fal::remote_stub(f).is_none());
        let (root, own, ledger) = (run_root.clone(), dir.to_path_buf(), run_root.join(STAGE_LEDGER));
        Some(std::thread::spawn(move || -> Result<(usize, u64, usize)> {
            let stop = std::sync::atomic::AtomicBool::new(false);
            let (mut sent, mut bytes, mut offered) = (0, 0, 0);
            for batch in files.chunks(8) {
                if !yields_upload(&root, Some(&own)) {
                    break;
                }
                let (n, b) = crate::fal::stage_files(&ledger, batch, YIELD_STREAMS, &stop)?;
                (sent, bytes, offered) = (sent + n, bytes + b, offered + batch.len());
            }
            Ok((sent, bytes, offered))
        }))
    } else {
        None
    };
    // Publish missing big files to the shared cache up front (compressed, once per host for concurrent runs);
    // anything that fails here still goes up with the job as before.
    for (i, (ext, sha)) in big.iter().enumerate() {
        if missing[i] {
            let name = ext.file_name().unwrap().to_string_lossy();
            match crate::fal::publish_cached(ext, sha, &name) {
                Ok(true) => missing[i] = false,
                Ok(false) => {}
                Err(e) => println!("[remote] publishing {name} to the fal cache failed ({e:#}); uploading it with the job"),
            }
        }
    }
    phases.done("binary publish");
    if let Some(early) = early {
        match early.join() {
            Ok(Ok((sent, bytes, offered))) => println!("[remote] yielded to the run's first-view upload: sent {sent} of {offered} \
                inputs ({:.1} MB) on {YIELD_STREAMS} stream(s) meanwhile", bytes as f64 / 1e6),
            Ok(Err(e)) => println!("[remote] early staging stopped ({e:#}); the job stages the rest"),
            Err(_) => println!("[remote] early staging panicked; the job stages the rest"),
        }
        phases.done("prestage yield wait");
    }
    phases.total();
    for upload in [false, true] {
        // fal reuses runners, and every link this preamble makes dangles once its job's exec dir is
        // deleted; `mkdir -p` under a dangling link fails ("mkdir: Already exists"). `mkd` drops
        // dangling links among the path's ancestors first.
        let mut script = vec![
            "set -e".to_string(),
            r#"mkd() { d=$1; while [ "$d" != / ] && [ "$d" != . ]; do { [ -L "$d" ] && [ ! -e "$d" ]; } && rm -f "${d:?}"; d=$(dirname "$d"); done; mkdir -p "$1"; }"#.to_string(),
            format!("mkd {}", quote(&parent_str(&run_root))),
        ];
        script.push(format!("ln -sfn {{run}} {}", quote(&run_str)));
        // fal uploads files only: recreate every staged dir that is empty (job output dirs, and
        // artifact dirs a failed job registered but never filled, which reopening the run checks).
        // Ask the stage, not the live run: a file can vanish from the run after it was staged.
        for rel in &empty_dirs {
            script.push(format!("mkd {}", quote(&run_root.join(rel).to_string_lossy())));
        }
        let mut cmd = Command::new(&python);
        let mut inputs = vec![crate::fal::Input { name: "run".into(), path: stage.clone(), sync_back: true }];
        // Packed dirs: extract into the staged run (mtimes now, as placed files get), and record the members' sha256
        // so the job's end deletes those it left as they were: the runner returns every file of the run it did not
        // place itself, and an untouched input must not come back as an output.
        for (i, pack) in packs.iter().enumerate() {
            let name = format!("p{i}");
            inputs.push(crate::fal::Input { name: name.clone(), path: pack.tar.clone(), sync_back: false });
            script.push(format!(
                "(cd {{run}} && tar -xmf {{{name}}} && tar -tf {{{name}}} --quoting-style=literal | grep -v '/$' | tr '\\n' '\\0' \
                 | xargs -0 -r sha256sum -- > {{{name}}}.sha256)"
            ));
        }
        // User runs go to the product-tier fal apps (own runner cap), never the dev experiment apps.
        cmd.env("SPLAT_FAL_TIER", &tier);
        cmd.arg(&drive).arg("exec").args(["--team", &team, "--lane", &lane, "--gpu", &gpu]);
        if !colmap.is_empty() {
            cmd.arg("--colmap");
        }
        if no_exp {
            cmd.arg("--no-exp");
        }
        // drive.py exec detaches by default (exit 0 at submit): this job needs the outputs before it ends.
        if help.contains("--attach") {
            cmd.arg("--attach");
        }
        for c in &colmap {
            let p = quote(&c.to_string_lossy());
            script.push(format!("{{ [ -L {p} ] || [ ! -e {p} ]; }} && {{ mkd {}; ln -sfn {{colmap}} {p}; }}", quote(&parent_str(c))));
        }
        cmd.arg("--io").arg(format!("run={}", stage.display()));
        for (i, (ext, upload_from)) in small.iter().enumerate() {
            let name = format!("x{i}");
            cmd.arg("--in").arg(format!("{name}={}", upload_from.display()));
            inputs.push(crate::fal::Input { name: name.clone(), path: upload_from.clone(), sync_back: false });
            let p = quote(&ext.to_string_lossy());
            script.push(format!("{{ [ -L {p} ] || [ ! -e {p} ]; }} && {{ mkd {}; ln -sfn {{{name}}} {p}; }}", quote(&parent_str(ext))));
        }
        for (i, (ext, sha)) in big.iter().enumerate() {
            let name = format!("b{i}");
            let cached = format!("{CACHE}/{sha}/{}", ext.file_name().unwrap().to_string_lossy());
            let fill = if upload || missing[i] {
                cmd.arg("--in").arg(format!("{name}={}", ext.display()));
                inputs.push(crate::fal::Input { name: name.clone(), path: ext.clone(), sync_back: false });
                format!(r#"cp -p {{{name}}} "$c.$$" && mv "$c.$$" "$c""#)
            } else {
                format!(r#"echo "splat-cloud: {sha} not cached on fal" >&2; exit {MISS}"#)
            };
            let p = quote(&ext.to_string_lossy());
            // Run from a copy on the runner's own disk: reading model weights straight from the /data network
            // volume cost ~21-30 s per process (truck84 masks: 23.5 s to the first frame vs 2.2 s from a local
            // copy, which takes ~1.5 s). fal reuses runners, so a warm runner keeps the copy for later jobs.
            // The runner's /tmp is its overlay disk, not RAM; a new copy evicts older shas of the same file name.
            let local = format!("{LOCAL_CACHE}/{sha}/{}", ext.file_name().unwrap().to_string_lossy());
            script.push(format!(
                "c={}; [ -e \"$c\" ] || {{ mkdir -p \"$(dirname \"$c\")\"; {fill}; }}; l={}; [ -e \"$l\" ] || {{ mkdir -p \"$(dirname \"$l\")\" && cp -p \"$c\" \"$l.$$\" && mv \"$l.$$\" \"$l\" && for o in {LOCAL_CACHE}/*/{n}; do [ \"$o\" = \"$l\" ] || {{ [ -e \"$o\" ] && rm -f \"${{o:?}}\" && rmdir \"$(dirname \"$o\")\" 2>/dev/null; }} || true; done; }} || {{ rm -f \"${{l:?}}.$$\"; l=$c; }}; {{ [ -L {p} ] || [ ! -e {p} ]; }} && {{ mkd {}; ln -sfn \"$l\" {p}; }}",
                quote(&cached),
                quote(&local),
                quote(&parent_str(ext)),
                n = quote(&ext.file_name().unwrap().to_string_lossy()),
            ));
        }
        for d in &lock_dirs {
            script.push(format!("mkd {}", quote(d)));
        }
        script.push(format!("cd {}", quote(spec.cwd.as_deref().unwrap_or(&run_str))));
        for (k, v) in &env {
            script.push(format!("export {k}={}", quote(v)));
        }
        // The runner app persists the CubeCL/pipeline caches only for jobs that send it a `binary`;
        // these run the splat binary from a script, so point the driver pipeline cache at /data
        // (shared by every runner; the file is keyed by device, framed and written atomically).
        script.push(format!("export SPLAT_PIPELINE_CACHE_DIR={PIPELINE_CACHE}"));
        script.push(ORT_PREAMBLE.to_string());
        if let Some(r) = &replay {
            let manifest = quote(&r.manifest.to_string_lossy());
            script.push(format!("{} __video_replay {manifest} || exit 87", quote(&argv[0])));
            // Stage the frames for the run's later jobs while this one works; waited for before it exits.
            if r.stages {
                script.push(format!("{} __video_replay_stage {manifest} & replay_stage=$!", quote(&argv[0])));
            }
        }
        let command = argv.iter().map(|a| quote(a)).collect::<Vec<_>>().join(" ");
        let cleanup = replay.as_ref().map(|r| format!("{}{} __video_replay_cleanup {} || true; ",
            if r.stages { "wait $replay_stage || true; " } else { "" }, quote(&argv[0]), quote(&r.manifest.to_string_lossy()))).unwrap_or_default()
            + &(0..packs.len()).map(|i| format!(
                "(cd {{run}} && sha256sum -c {{p{i}}}.sha256 2>/dev/null | sed -n 's/: OK$//p' | tr '\\n' '\\0' | xargs -0 -r rm -f --) || true; "
            )).collect::<String>();
        // SPLAT_REMOTE_VRAM_LOG=1: sample board memory (nvidia-smi memory.used, MiB, every 500 ms)
        // into the job's artifact, which syncs back. The first row is the idle baseline.
        let vram_log = std::env::var("SPLAT_REMOTE_VRAM_LOG").is_ok_and(|v| v == "1");
        match spec.meta.get("artifact_id").and_then(Value::as_str).filter(|_| vram_log) {
            Some(id) => {
                let log = quote(&run_root.join("artifacts").join(id).join("vram_board_mib.csv").to_string_lossy());
                script.push(format!(
                    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > {log} 2>/dev/null < /dev/null & vram=$!"
                ));
                script.push(format!("rc=0; {command} || rc=$?; kill $vram 2>/dev/null; {cleanup}exit $rc"));
            }
            None if !cleanup.is_empty() => script.push(format!("rc=0; {command} || rc=$?; {cleanup}exit $rc")),
            None => script.push(format!("exec {command}")),
        }
        cmd.args(["--timeout", &timeout.to_string(), "--out"]).arg(dir.join("remote-out"));
        cmd.args(["sh", "--", "-c", &script.join("\n")]);
        fs::write(dir.join("remote.sh"), script.join("\n"))?;
        println!(
            "[remote] fal exec: {} staged dirs, {} inputs, {} cached binaries{}",
            stage_keys.len(),
            small.len(),
            big.len(),
            if upload { " (uploading to fill the cache)" } else { "" }
        );
        code = if native {
            let job = crate::fal::Exec {
                gpu: gpu.clone(),
                progress_path: dir.join("remote-progress.json"),
                argv: vec!["sh".into(), "-c".into(), script.join("\n")],
                inputs,
                env: BTreeMap::new(),
                timeout_s: timeout,
                colmap: !colmap.is_empty(),
                ledger: Some(run_root.join(STAGE_LEDGER)),
                staged_marker: Some(dir.join(crate::fal::INPUTS_STAGED)),
                keep_remote: kept_remote(&run_root, &spec.tool),
                track: crate::fal::Track {
                    team: team.clone(),
                    lane: lane.clone(),
                    tag: format!("splat-cloud/{}", dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
                    run_root: Some(run_root.clone()),
                },
            };
            let out = crate::fal::exec(&job, &tier)?;
            use std::io::Write;
            std::io::stdout().write_all(&out.stdout).ok();
            std::io::stderr().write_all(&out.stderr).ok();
            // Same closing line as drive.py, which job_cost() and remote_ran() read.
            eprintln!(
                "[{}] exit {} | {} | {} files synced | ~${} ({team}/{lane})",
                crate::utc_now(),
                out.exit_code,
                out.gpu_label,
                out.synced,
                out.cost_usd
            );
            out.exit_code
        } else {
            cmd.status().with_context(|| format!("spawn {python} {drive}"))?.code().unwrap_or(-255)
        };
        if code != MISS || big.is_empty() || upload {
            break;
        }
        println!("[remote] binary not yet cached on fal; uploading it once");
    }
    if code == 87 && replay.is_some() {
        println!("[remote] video replay could not verify identical frames; retrying with original JPEG uploads");
        fs::remove_dir_all(&stage).ok();
        return run_inner(dir, spec, false);
    }
    // A packed input back unchanged (the job's end did not get to delete it, e.g. a killed job) is no output.
    for (path, sha) in packs.iter().flat_map(|p| &p.members) {
        if path.is_file() && sha256(path).is_ok_and(|s| s == *sha) {
            fs::remove_file(path)?;
        }
    }
    let merged = merge_back(&stage, &run_root, &stage, &staged)?;
    println!("[remote] exit {code}; merged {merged} new/changed files into the run");
    fs::remove_dir_all(&stage).ok();
    fs::remove_dir_all(dir.join("remote-models")).ok();
    fs::remove_dir_all(&packs_dir).ok();
    if let Some(usd) = job_cost(dir) {
        crate::atomic_json(dir.join("remote_cost.json"), &serde_json::json!({"usd": usd, "gpu": gpu, "tool": spec.tool}))?;
    }
    if code != 0 && !remote_ran(dir) {
        bail!("fal exec failed before the remote command ran (exit {code}); see stderr.log");
    }
    Ok(code)
}

/// Least JPEG upload (bytes, net of the source video) that video replay must save to pay for decoding the frames
/// again on the runner (~3-10 s a job). Inputs stage at ~1.9-4.8 MB/s from one host. Gimbal's 15.7 MB net saving
/// was a loss: its replayed full-SfM job took ~+37 s overall (uxbench aadaf5c, G2-REPORT.md).
const REPLAY_MIN_SAVING: u64 = 32 << 20;

/// The capture video a native `--cloud` run's jobs would replay frames from (`frames`: an extracted frames
/// artifact dir with its replay recipe), when its jobs can replay at all.
pub fn replay_source(frames: &Path) -> Option<PathBuf> {
    let recipe: Value = serde_json::from_slice(&fs::read(frames.join("video-replay.json")).ok()?).ok()?;
    replay_video(Path::new(recipe["source"].as_str()?))
}

/// `source` (a run's capture video, before any frames are extracted from it) when this run's jobs can replay
/// frames from it: the video they would send.
pub fn replay_video(source: &Path) -> Option<PathBuf> {
    let native = std::env::var("SPLAT_FAL_DRIVE").map_or(true, |d| d.is_empty());
    if !(cfg!(target_os = "linux") && native && mode().as_deref() == Some("fal")) { return None; }
    let source = source.canonicalize().ok()?;
    source.is_file().then_some(source)
}

/// Whether a job sending the frames in `frames` will replay them from the source video, by
/// [`prepare_video_replay`]'s rule for frames not yet staged on fal (its first job).
pub fn replay_pays(frames: &Path) -> bool {
    let Some(source) = replay_source(frames) else { return false };
    let Ok(raw) = fs::read(frames.join("video-replay.json")) else { return false };
    let Ok(recipe) = serde_json::from_slice::<Value>(&raw) else { return false };
    let mut unique = BTreeMap::new();
    for (rel, sha) in recipe["hashes"].as_object().into_iter().flatten() {
        if let (Some(sha), Ok(meta)) = (sha.as_str(), fs::metadata(frames.join(rel))) {
            unique.insert(sha.to_string(), meta.len());
        }
    }
    let source_len = fs::metadata(&source).map_or(u64::MAX, |m| m.len());
    unique.values().sum::<u64>() >= source_len.saturating_add(REPLAY_MIN_SAVING)
}

/// Upload `source` into the run's fal staging dir now, as `gate` allows (see [`crate::fal::prestage`]); bytes sent.
pub fn prestage(run_root: &Path, source: &Path, gate: &std::sync::Arc<crate::fal::PrestageGate>) -> Result<u64> {
    let run_root = run_root.canonicalize()?;
    capture_gate(&run_root)?;
    crate::fal::prestage(&run_root.join(STAGE_LEDGER), source, gate)
}

/// Upload streams for [`FramePrestage`] (~0.75 MB/s each through fal's gateway): the 54 MB of arkit-41254538's
/// full frames in ~25 s, inside its 66 s extraction, leaving uplink for the preview job's own staging.
const FRAME_PRESTAGE_STREAMS: usize = 3;

/// `jobs/<this>` names the run's first-view job (the quick SfM): until its inputs are on fal, every other
/// upload of the run yields the uplink to it ([`yields_upload`]).
const UPLOAD_PRIORITY: &str = "fal-upload-priority";

/// Upload streams of a run's other uploads while they yield to its priority job: its binary publish and staging
/// run 32 streams, so one stream takes a small share of the uplink and still makes progress (the full SfM used to
/// wait up to 60 s for the quick upload, then send everything at once beside it).
const YIELD_STREAMS: usize = 1;

/// Every file under `root` (a file or a dir), following links as staging does.
fn input_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if root.is_file() {
        out.push(root.to_path_buf());
    } else if root.is_dir() {
        for entry in fs::read_dir(root)? {
            input_files(&entry?.path(), out)?;
        }
    }
    Ok(())
}

/// Give `job_id`'s uploads the run's uplink until its inputs are staged (see [`yields_upload`]).
pub fn set_upload_priority(run_root: &Path, job_id: &str) -> Result<()> {
    let jobs = run_root.join("jobs");
    fs::create_dir_all(&jobs)?;
    crate::atomic_json(jobs.join(UPLOAD_PRIORITY), &serde_json::json!({"job_id": job_id}))
}

/// Whether an upload of the run (by the job in `own`, if any) should leave the uplink to the run's priority job:
/// that job is another one, is running and its inputs are not yet staged on fal.
pub fn yields_upload(run_root: &Path, own: Option<&Path>) -> bool {
    let Ok(raw) = fs::read(run_root.join("jobs").join(UPLOAD_PRIORITY)) else { return false };
    let Some(id) = serde_json::from_slice::<Value>(&raw).ok().and_then(|v| v["job_id"].as_str().map(String::from)) else { return false };
    let dir = run_root.join("jobs").join(&id);
    if own.and_then(Path::file_name).is_some_and(|n| n == id.as_str()) {
        return false;
    }
    dir.is_dir() && !dir.join(crate::fal::INPUTS_STAGED).exists() && !dir.join("completion.json").exists()
}

/// The full extraction's frames going up to the run's fal staging while the extraction still decodes, so the
/// full SfM job finds them staged instead of sending them after the decode (arkit-41254538: 54.4 MB, 19.8 s
/// between the extraction's end and the SfM request). A frame is sent once a later frame exists: the extractor
/// writes `images/frame_<source index>.jpg` one at a time in name order, so only the newest can be partial.
/// Content addressing makes the job's own staging the check: a frame changed after it was sent is just sent
/// again under its new sha256. Never more than the capture video plus [`REPLAY_MIN_SAVING`] goes up early,
/// the most a job would send as JPEGs rather than replay them from the video.
pub struct FramePrestage {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<Value>>,
}

impl FramePrestage {
    /// Start sending the frames written to `images` (native fal client only). Until the run's priority job
    /// (the quick SfM) has its inputs staged, it sends on [`YIELD_STREAMS`] only.
    pub fn start(run_root: &Path, images: PathBuf, source: &Path) -> Option<Self> {
        let native = std::env::var("SPLAT_FAL_DRIVE").map_or(true, |d| d.is_empty());
        if !(native && mode().as_deref() == Some("fal")) { return None; }
        let run_root = run_root.canonicalize().ok()?;
        let budget = replay_video(source).and_then(|s| fs::metadata(s).ok())
            .map_or(u64::MAX, |m| m.len().saturating_add(REPLAY_MIN_SAVING));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        // The thread never journals: the run lock belongs to the supervisor's loop ([`Self::finish`] reports).
        let thread = std::thread::spawn(move || {
            use std::sync::atomic::Ordering;
            let started = std::time::Instant::now();
            let (mut sent, mut files, mut bytes, mut over_budget) = (BTreeSet::new(), 0usize, 0u64, false);
            let mut planned = 0u64;
            let outcome = (|| -> Result<()> {
                capture_gate(&run_root)?;
                let ledger = run_root.join(STAGE_LEDGER);
                while !flag.load(Ordering::Relaxed) && !over_budget {
                    let mut written: Vec<PathBuf> = match fs::read_dir(&images) {
                        Ok(dir) => dir.filter_map(|e| e.ok().map(|e| e.path()))
                            .filter(|p| p.extension().is_some_and(|e| e == "jpg")).collect(),
                        Err(_) => Vec::new(),
                    };
                    written.sort();
                    written.pop();
                    // While yielding, small batches: the stream count follows the priority job's staging.
                    let yielding = yields_upload(&run_root, None);
                    let max_batch = if yielding { 4 } else { usize::MAX };
                    let mut batch = Vec::new();
                    for path in written.into_iter().filter(|p| !sent.contains(p)).take(max_batch) {
                        let len = fs::metadata(&path)?.len();
                        if planned + len > budget { over_budget = true; break; }
                        planned += len;
                        batch.push(path);
                    }
                    if batch.is_empty() {
                        std::thread::sleep(std::time::Duration::from_millis(250));
                        continue;
                    }
                    let streams = if yielding { YIELD_STREAMS } else { FRAME_PRESTAGE_STREAMS };
                    let (n, b) = crate::fal::stage_files(&ledger, &batch, streams, &flag)?;
                    files += n;
                    bytes += b;
                    sent.extend(batch);
                }
                Ok(())
            })();
            let mut summary = serde_json::json!({"frames_sent":files,"mb_sent":bytes as f64 / 1e6,"frames_seen":sent.len(),
                "wall_s":started.elapsed().as_secs_f64(),"budget_reached":over_budget});
            if let Err(e) = outcome { summary["error"] = Value::from(format!("{e:#}")); }
            summary
        });
        Some(Self { stop, thread: Some(thread) })
    }

    /// Stop sending (requests in flight finish) and report what went up, before a job that stages the same
    /// frames is dispatched: two uploads of one chunk race on fal (/stage HTTP 500).
    pub fn finish(mut self) -> Value {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.thread.take().and_then(|t| t.join().ok()).unwrap_or(Value::Null)
    }
}

impl Drop for FramePrestage {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// What [`prepare_video_replay`] set up for a job.
struct Replay {
    manifest: PathBuf,
    /// Decoder binaries: fal's shared binary cache (content-addressed, uploaded once ever).
    cached: BTreeSet<PathBuf>,
    /// Capture videos: the run's own staging, uploaded alongside the other inputs. The binary cache cost
    /// 20-30 s for a 2.8-9.4 MB video, a serial step before the job.
    sources: BTreeSet<PathBuf>,
    /// The worker also stages the regenerated frames into the run's fal staging dir, for later jobs.
    stages: bool,
}

/// Whether `path` (symlinks followed) is an ELF executable, not a script needing an interpreter.
fn native_executable(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && magic == *b"\x7fELF"
}

/// Replace only byte-matched JPEG inputs in the staged copy; the local originals remain intact.
/// All aliases (SfM datasets as well as frame artifacts) must be omitted to avoid uploading them.
/// With the run's staging `ledger` (native client), frames already staged on fal (an earlier job replayed them
/// and staged them, or uploaded them) stay as normal inputs: replay runs on a run's first job only.
fn prepare_video_replay(run: &Path, stage: &Path, binary: &Path, externals: &mut BTreeSet<PathBuf>, ledger: Option<&Path>,
    min_saving: u64) -> Result<Option<Replay>> {
    fn files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() { files(&entry.path(), out)?; }
            else if entry.path().extension().is_some_and(|e| e == "jpg") { out.push(entry.path()); }
        }
        Ok(())
    }
    let artifacts = run.join("artifacts");
    if !artifacts.is_dir() || !binary.is_file() { return Ok(None); }
    let mut staged = Vec::new();
    files(stage, &mut staged)?;
    let mut tasks = Vec::new();
    let (mut cached, mut sources) = (BTreeSet::new(), BTreeSet::new());
    let mut stage_dir: Option<String> = None;
    let mut new_shas = Vec::new();
    let binary_sha = crate::hashcache::sha256(binary)?;
    for entry in fs::read_dir(artifacts)? {
        let path = entry?.path().join("video-replay.json");
        if !path.is_file() { continue; }
        let raw = fs::read(&path)?;
        let Ok(recipe) = serde_json::from_slice::<Value>(&raw) else { continue; };
        let Some(source) = recipe["source"].as_str().map(PathBuf::from) else { continue; };
        let Some(ffmpeg) = recipe["ffmpeg"].as_str().map(PathBuf::from) else { continue; };
        let ffprobe = ffmpeg.with_file_name("ffprobe");
        // Configured decoders without their matching probe cannot reproduce HDR decisions.
        if !source.is_file() || !ffmpeg.is_file() || !ffprobe.is_file() { continue; }
        // The runner execs the decoders itself: a wrapper script (the bench harness's `#!/usr/bin/python3` ffmpeg
        // that pins decoding to a core) has no interpreter in the engine image, so its replay exits 87 every time
        // and the job runs again with the JPEG uploads (4 of 19 phone rooms on 326e03c, 2026-10-12).
        if let Some(p) = [&ffmpeg, &ffprobe].into_iter().find(|p| !native_executable(p)) {
            println!("[remote] video replay skipped: decoder {} is not a native executable the runner can run", p.display());
            continue;
        }
        let Some(hashes) = recipe["hashes"].as_object() else { continue; };
        // (staged input, recipe key, sha256)
        let mut matched = Vec::new();
        for input in &staged {
            if !input.is_file() { continue; }
            for (from, expected) in hashes {
                if input.file_name() != Path::new(from).file_name() { continue; }
                let Some(expected) = expected.as_str() else { continue; };
                if sha256(input)? != expected { continue; }
                matched.push((input.clone(), from.clone(), expected.to_string()));
                break;
            }
        }
        if matched.is_empty() { continue; }
        let unique: BTreeSet<String> = matched.iter().map(|m| m.2.clone()).collect();
        let have = match ledger {
            Some(ledger) => match crate::fal::run_staged(ledger, &unique.iter().cloned().collect::<Vec<_>>()) {
                Ok((dir, have)) => { stage_dir = Some(dir); have }
                Err(e) => { println!("[remote] video replay: fal staging check failed ({e:#}); replaying every frame"); BTreeSet::new() }
            },
            None => BTreeSet::new(),
        };
        // Frames staged on fal already cost nothing to send: they stay as normal inputs.
        matched.retain(|m| !have.contains(&m.2));
        let mut unique_bytes = BTreeMap::new();
        for (input, _, sha) in &matched { unique_bytes.insert(sha.clone(), fs::metadata(input)?.len()); }
        let bytes: u64 = unique_bytes.values().sum();
        let source_len = fs::metadata(&source)?.len();
        if bytes < source_len + min_saving {
            if !have.is_empty() && matched.is_empty() {
                println!("[remote] video replay: all {} frames already staged on fal; no replay", have.len());
            } else {
                println!("[remote] video replay skipped: {:.2} MB of JPEGs to send vs a {:.2} MB video (needs {:.0} MB saved)",
                    bytes as f64 / 1e6, source_len as f64 / 1e6, min_saving as f64 / 1e6);
            }
            continue;
        }
        let copies: Vec<Value> = matched.iter().map(|(input, from, _)| -> Result<Value> {
            Ok(serde_json::json!({"from": from, "to": run.join(input.strip_prefix(stage)?)}))
        }).collect::<Result<_>>()?;
        use sha2::{Digest, Sha256};
        let mut key = Sha256::new(); key.update(&raw); key.update(&binary_sha);
        tasks.push(serde_json::json!({"recipe": recipe, "key": format!("{:x}", key.finalize()), "copies": copies,
            "stage_dir": stage_dir}));
        if stage_dir.is_some() { new_shas.extend(unique_bytes.keys().cloned()); }
        for (input, _, _) in matched { fs::remove_file(input)?; }
        for path in [ffmpeg, ffprobe] { externals.insert(path.clone()); cached.insert(path); }
        externals.insert(source.clone());
        sources.insert(source);
        println!("[remote] video replay replaces {:.2} MB of unique JPEG inputs", bytes as f64 / 1e6);
    }
    if tasks.is_empty() { return Ok(None); }
    // The worker stages these into the run's staging dir; `release` drops them with the run.
    let stages = !new_shas.is_empty() && ledger.is_some_and(|l| crate::fal::ledger_record(l, new_shas).is_ok());
    if !stages { for task in &mut tasks { task["stage_dir"] = Value::Null; } }
    let name = ".video-replay-manifest.json";
    fs::write(stage.join(name), serde_json::to_vec(&tasks)?)?;
    Ok(Some(Replay { manifest: run.join(name), cached, sources, stages }))
}

/// LPIPS on fal: the image ships onnxruntime-gpu 1.28 with its CUDA libraries in /opt/ort (as the
/// perf runner's Job.ort_gpu uses); the host's Ubuntu libonnxruntime needs system libs fal lacks.
const ORT_PREAMBLE: &str = r#"site=$(ls -d /opt/ort/lib/python3*/site-packages 2>/dev/null | head -1)
if [ -n "$site" ]; then
  lib=$(ls "$site"/onnxruntime/capi/libonnxruntime.so* | head -1)
  export ORT_DYLIB_PATH="$lib" SPLAT_ONNXRUNTIME="$lib" SPLAT_EVAL_ORT_CUDA=1
  export LD_LIBRARY_PATH="$(dirname "$lib"):$(find "$site/nvidia" -type d -name lib | tr '
' ':')${LD_LIBRARY_PATH:-}"
fi"#;

/// Every GPU job of a run on one class, the fastest, where the product app keeps a warm runner
/// (fal/drive.py PRODUCT_WARM_GPU): SfM needs it anyway (AVX-512 COLMAP), and a job on another
/// class waited ~54 s for a cold runner.
const DEFAULT_GPU: &str = "rtxpro6000";
/// Persistent fal storage for large inputs, keyed by sha256.
const CACHE: &str = "/data/splat-cloud/files";
/// Runner-local copies of CACHE files (binaries, model weights), keyed by sha256; fal reuses runners.
const LOCAL_CACHE: &str = "/tmp/splat-cloud-local";
/// Persistent Vulkan pipeline cache of `--cloud` jobs on fal (`<device key>.bin` inside).
const PIPELINE_CACHE: &str = "/data/splat-cloud/wgpu-pipeline-cache";
/// Files outside the run (tools, models) from this size go to the shared cache, once across runs. At 16 MB the
/// 9.5 MB COLMAP vocab tree went up with every SfM job (chunked plus a ~7 s commit: most of the driveway clip's 22.6 s
/// quick-SfM staging).
const BIG_FILE: u64 = 4 << 20;
/// Remote exit code for "cached binary missing" (retried once with the upload).
const MISS: i32 = 86;

fn sha256(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// drive.py's closing line: "... | ~$0.0123 (team/lane; total ~$..) | <dir>".
fn job_cost(dir: &Path) -> Option<f64> {
    let log = fs::read_to_string(dir.join("stderr.log")).ok()?;
    let costs: Vec<f64> = log
        .lines()
        .filter(|l| l.contains("] exit ") && l.contains(" | ~$"))
        .filter_map(|l| l[l.find(" | ~$")? + 5..].split_whitespace().next()?.parse().ok())
        .collect();
    (!costs.is_empty()).then(|| (costs.iter().sum::<f64>() * 10000.0).round() / 10000.0)
}

/// Up-front estimate (USD, USD with retries, wall minutes). Measured Oct 9 on fal (L40S train,
/// RTX PRO 6000 SfM), fixed driver, totals incl. mesh, Mip-NeRF 360 garden (185 frames):
/// max-budget-5m $0.67 +/- 0.07 / 30 +/- 3 min over 3 runs (budget 3.56M, train ~$0.30; at 45k
/// since X27 train is ~x1.35, so ~$0.78 / ~32 min);
/// max-neural-2.5m $0.51 / 22 min (1 run); default 7k/1M $0.38 +/- 0.09 / 20 min (SfM dominates).
/// Smaller captures cost less (pegasus video: ~$0.26 default); SfM grows with the frame count.
/// Cold starts, a new version's binary upload and retries can more than double it.
fn estimate(preset: Option<&Preset>) -> (f64, f64, u32) {
    match preset {
        Some(p) if p.max_budget => (0.80, 2.20, 32),
        Some(_) => (0.55, 1.50, 25),
        None => (0.40, 1.20, 20),
    }
}

/// A goal that asks for a mesh trains surfels (2DGS), which keep the default recipe.
pub fn goal_wants_mesh(goal: &str) -> bool {
    let goal = goal.to_lowercase();
    ["mesh", "glb", "surface"].iter().any(|w| goal.contains(w))
}

/// `vauss run --cloud`: GPU jobs of this run go to fal. Checks drive.py and its Python, records the
/// choice (so `resume` keeps it) with the cost estimate, and exports the env the job workers read.
pub fn enable(run_root: &Path) -> Result<Value> {
    // Native fal client (FAL_KEY) unless SPLAT_FAL_DRIVE points at the dev drive.py.
    let drive = std::env::var("SPLAT_FAL_DRIVE").ok().filter(|d| !d.is_empty()).map(PathBuf::from);
    let python = std::env::var("SPLAT_FAL_PYTHON").ok().map(PathBuf::from).unwrap_or_else(|| "python3".into());
    let gpu = std::env::var("SPLAT_FAL_GPU").unwrap_or_else(|_| DEFAULT_GPU.into());
    match &drive {
        Some(drive) => {
            let help = Command::new(&python).arg(drive).args(["exec", "--help"]).output()
                .with_context(|| format!("--cloud: run {} {}", python.display(), drive.display()))?;
            ensure!(help.status.success(), "--cloud: {} exec --help failed: {}", drive.display(), String::from_utf8_lossy(&help.stderr));
        }
        None => ensure!(
            std::env::var("FAL_KEY").is_ok_and(|k| !k.is_empty()),
            "--cloud needs FAL_KEY (your fal.ai API key) in the environment"
        ),
    }
    // What training will actually run: mesh goals keep the default recipe even under MAX.
    let goal = fs::read(run_root.join("run.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v["goal"].as_str().map(str::to_owned)).unwrap_or_default();
    let training = selected_preset().filter(|_| !goal_wants_mesh(&goal));
    let max = training.is_some();
    let (usd, high, minutes) = estimate(training);
    let cloud = serde_json::json!({"provider": "fal", "gpu": gpu, "client": if drive.is_some() { "drive.py" } else { "native" },
        "drive": drive, "python": python,
        "estimate_usd": usd, "estimate_high_usd": high, "estimate_minutes": minutes,
        "preset": preset_name(), "fetch_all": fetch_all(), "training": if max { preset_name() } else if preset_name() != "default" { "default (mesh goal)" } else { "default" }});
    crate::atomic_json(run_root.join("cloud.json"), &cloud)?;
    resume(run_root)?;
    Ok(cloud)
}

/// Wall limit of one remote GPU job (SPLAT_FAL_TIMEOUT_S, default 3300 s): the runner kills
/// the job's process group past it.
pub fn job_timeout_s() -> f64 {
    std::env::var("SPLAT_FAL_TIMEOUT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(3300.)
}

/// Re-export the cloud env for a run created with `--cloud` (no-op otherwise).
pub fn resume(run_root: &Path) -> Result<bool> {
    let Ok(bytes) = fs::read(run_root.join("cloud.json")) else { return Ok(false) };
    let cloud: Value = serde_json::from_slice(&bytes)?;
    // SAFETY: called on the main thread before the driver, MCP server or job workers start.
    unsafe {
        std::env::set_var("SPLAT_GPU_REMOTE", "fal");
        for (key, field) in [("SPLAT_FAL_DRIVE", "drive"), ("SPLAT_FAL_PYTHON", "python"), ("SPLAT_FAL_GPU", "gpu"), ("SPLAT_CLOUD_PRESET", "preset")] {
            // A native run (drive = null) must stay native on resume.
            if std::env::var_os(key).is_none() {
                if let Some(v) = cloud[field].as_str() {
                    std::env::set_var(key, v);
                }
            }
        }
        if cloud["fetch_all"].as_bool() == Some(true) && std::env::var_os(FETCH_ALL_ENV).is_none() {
            std::env::set_var(FETCH_ALL_ENV, "1");
        }
    }
    Ok(true)
}

/// Inputs and outputs a `--cloud` run keeps staged on fal /data between its jobs (fal::release).
const STAGE_LEDGER: &str = "fal-staged.json";

/// End of a `--cloud` run (finished or failed): delete what it staged on fal /data.
pub fn release(run_root: &Path) {
    if let Err(e) = crate::fal::release(&run_root.join(STAGE_LEDGER)) {
        eprintln!("[fal] could not delete staged run files now ({e:#}); fal sweeps them after 24 h");
    }
}

/// Estimated vs actual fal spend of a `--cloud` run, for the report; None for local runs.
pub fn cost(run_root: &Path) -> Option<Value> {
    let cloud: Value = serde_json::from_slice(&fs::read(run_root.join("cloud.json")).ok()?).ok()?;
    let read = |p: PathBuf| fs::read(p).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    let mut stages = Vec::new();
    for entry in fs::read_dir(run_root.join("jobs")).ok()?.flatten() {
        let Some(cost) = read(entry.path().join("remote_cost.json")) else { continue };
        let done = read(entry.path().join("completion.json"));
        stages.push(serde_json::json!({
            "job_id": entry.file_name().to_string_lossy(), "tool": cost["tool"], "gpu": cost["gpu"], "usd": cost["usd"],
            "wall_s": done.as_ref().and_then(|d| d["wall_s"].as_f64()).map(|w| w.round()),
            "finished_at": done.as_ref().map(|d| d["finished_at"].clone()),
        }));
    }
    stages.sort_by(|a, b| a["finished_at"].as_str().cmp(&b["finished_at"].as_str()));
    let actual: f64 = stages.iter().filter_map(|s| s["usd"].as_f64()).sum();
    Some(serde_json::json!({"provider": cloud["provider"], "gpu": cloud["gpu"], "estimate_usd": cloud["estimate_usd"],
        "estimate_high_usd": cloud["estimate_high_usd"], "actual_usd": (actual * 10000.0_f64).round() / 10000.0,
        "gpu_jobs": stages.len(), "stages": stages}))
}

fn parent_str(p: &Path) -> String {
    p.parent().unwrap_or(Path::new("/")).to_string_lossy().into_owned()
}

/// drive.py prints "... exit <code> | <gpu> ..." once the remote command has run.
fn remote_ran(dir: &Path) -> bool {
    fs::read_to_string(dir.join("stderr.log")).is_ok_and(|s| s.lines().any(|l| l.contains("] exit ") && l.contains(" | ")))
}

#[cfg(test)]
mod video_replay_tests {
    use super::*;

    #[test]
    fn staging_skips_atomic_write_temps_and_files_that_vanish() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let run = temp.path().join("run");
        fs::create_dir_all(run.join("artifacts/eval_1"))?;
        fs::write(run.join("state.json"), b"{}")?;
        fs::write(run.join(".state.json.123.tmp"), b"{}")?;
        fs::write(run.join("artifacts/eval_1/metrics.json"), b"{}")?;
        let stage = temp.path().join("stage");
        link_tree(&run, &stage, &run, true, true)?;
        assert!(stage.join("state.json").is_file() && stage.join("artifacts/eval_1/metrics.json").is_file());
        assert!(!stage.join(".state.json.123.tmp").exists());
        // A file renamed away between the listing and the link is skipped, not an error.
        link_tree(&run.join("gone.json"), &stage.join("gone.json"), &run, true, true)?;
        assert!(!stage.join("gone.json").exists());
        Ok(())
    }

    #[test]
    fn many_file_dirs_pack_deterministically_and_leave_big_files_staged() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let mut shas = Vec::new();
        for round in 0..2 {
            let stage = temp.path().join(format!("stage{round}"));
            for i in 0..=PACK_MIN_FILES {
                let f = stage.join(format!("artifacts/sfm_1/images/f{i:03}.jpg"));
                fs::create_dir_all(f.parent().unwrap())?;
                fs::write(f, vec![i as u8; 200 << 10])?;
            }
            fs::write(stage.join("artifacts/sfm_1/big.bin"), vec![0u8; PACK_FILE_MAX as usize + 1])?;
            fs::create_dir_all(stage.join("artifacts/few"))?;
            fs::write(stage.join("artifacts/few/a.json"), b"{}")?;
            fs::write(stage.join("run.json"), b"{}")?;
            let packs = pack_stage(&stage, &temp.path().join(format!("packs{round}")), |_| Ok(BTreeSet::new()))?.0;
            assert_eq!(packs.iter().map(|p| p.members.len()).sum::<usize>(), PACK_MIN_FILES + 1);
            assert!(packs.len() >= 4 && packs.iter().all(|p| fs::metadata(&p.tar).is_ok_and(|m| m.len() <= PACK_TAR_BYTES)));
            // Members leave the stage; big files, small dirs and root files stay per-file.
            assert!(!stage.join("artifacts/sfm_1/images/f000.jpg").exists());
            assert!(stage.join("artifacts/sfm_1/big.bin").is_file() && stage.join("artifacts/few/a.json").is_file());
            assert!(stage.join("run.json").is_file());
            let mut archive = tar::Archive::new(fs::File::open(&packs[0].tar)?);
            let first = archive.entries()?.next().unwrap()?;
            assert_eq!(first.path()?.to_string_lossy(), "artifacts/sfm_1/images/f000.jpg");
            assert_eq!(first.header().mtime()?, 0);
            shas.push(packs.iter().map(|p| sha256(&p.tar)).collect::<Result<Vec<_>>>()?);
        }
        assert_eq!(shas[0], shas[1], "same files, same tar bytes");
        // A new file changes only the tars around it.
        let stage = temp.path().join("stage2");
        for i in 0..=PACK_MIN_FILES {
            let f = stage.join(format!("artifacts/sfm_1/images/f{i:03}.jpg"));
            fs::create_dir_all(f.parent().unwrap())?;
            fs::write(f, vec![i as u8; 200 << 10])?;
        }
        fs::write(stage.join("artifacts/sfm_1/images/f010b.jpg"), b"new")?;
        let packs = pack_stage(&stage, &temp.path().join("packs2"), |_| Ok(BTreeSet::new()))?.0;
        let new: BTreeSet<String> = packs.iter().map(|p| sha256(&p.tar)).collect::<Result<_>>()?;
        assert!(shas[0].iter().filter(|s| !new.contains(*s)).count() <= 2, "{} of {} tars changed", shas[0].len(), new.len());
        Ok(())
    }

    #[test]
    fn packed_frame_aliases_upload_pixels_once_and_restore_every_path() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let stage = temp.path().join("stage");
        let n = PACK_MIN_FILES + 1;
        let mut expected = BTreeMap::new();
        for i in 0..n {
            for artifact in ["frames_1", "sfm_1", "sfm_2"] {
                let rel = format!("artifacts/{artifact}/images/f{i:03}.jpg");
                let path = stage.join(&rel);
                fs::create_dir_all(path.parent().unwrap())?;
                let bytes = vec![i as u8; 200 << 10];
                fs::write(&path, &bytes)?;
                expected.insert(rel, sha256(&path)?);
            }
        }
        let (packs, _) = pack_stage(&stage, &temp.path().join("packs"), |_| Ok(BTreeSet::new()))?;
        let uploaded: u64 = packs.iter().map(|p| fs::metadata(&p.tar).unwrap().len()).sum();
        // Three names per image need tar headers and GNU long-name records, but only one
        // pixel payload. Leave 10% for that metadata; another pixel copy exceeds it easily.
        let pixel_bytes = (n * (200 << 10)) as u64;
        assert!(uploaded < pixel_bytes * 11 / 10, "aliased pixels were uploaded again: {uploaded}");
        let restored = temp.path().join("restored");
        fs::create_dir_all(&restored)?;
        let mut regular = 0;
        for pack in packs {
            let mut archive = tar::Archive::new(fs::File::open(&pack.tar)?);
            for entry in archive.entries()? {
                let entry = entry?;
                if entry.header().entry_type().is_file() {
                    regular += 1;
                    assert!(entry.path()?.starts_with("artifacts/sfm_1"), "pixels moved out of the reusable dataset archive");
                }
            }
            tar::Archive::new(fs::File::open(pack.tar)?).unpack(&restored)?;
        }
        assert_eq!(regular, n);
        for (rel, sha) in expected {
            assert_eq!(sha256(&restored.join(rel))?, sha);
        }
        Ok(())
    }

    #[test]
    fn files_staged_on_fal_already_are_not_packed() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let write = |stage: &Path, n: usize| -> Result<Vec<String>> {
            (0..n).map(|i| {
                let f = stage.join(format!("artifacts/frames/f{i:03}.jpg"));
                fs::create_dir_all(f.parent().unwrap())?;
                fs::write(&f, format!("frame {i}"))?;
                sha256(&f)
            }).collect()
        };
        let n = PACK_MIN_FILES * 3;
        // Every frame staged (video replay staged them, or an earlier per-file job): nothing is packed.
        let stage = temp.path().join("all");
        let shas = write(&stage, n)?;
        let (packs, kept) = pack_stage(&stage, &temp.path().join("p-all"), |asked| {
            assert_eq!(asked.len(), n);
            Ok(shas.iter().cloned().collect())
        })?;
        assert!(packs.is_empty() && kept == n);
        assert_eq!(fs::read_dir(stage.join("artifacts/frames"))?.count(), n);
        // Most staged: the rest is under the threshold and stays per-file too.
        let stage = temp.path().join("most");
        let shas = write(&stage, n)?;
        let (packs, kept) = pack_stage(&stage, &temp.path().join("p-most"), |_| Ok(shas[PACK_MIN_FILES..].iter().cloned().collect()))?;
        assert!(packs.is_empty() && kept == n - PACK_MIN_FILES);
        // Some staged: only the others are packed, the staged ones stay in the stage.
        let stage = temp.path().join("some");
        let shas = write(&stage, n)?;
        let (packs, kept) = pack_stage(&stage, &temp.path().join("p-some"), |_| Ok(shas[..PACK_MIN_FILES].iter().cloned().collect()))?;
        let members: BTreeSet<String> = packs.iter().flat_map(|p| &p.members).map(|m| m.1.clone()).collect();
        assert_eq!(kept, PACK_MIN_FILES);
        assert_eq!(members, shas[PACK_MIN_FILES..].iter().cloned().collect());
        assert_eq!(fs::read_dir(stage.join("artifacts/frames"))?.count(), PACK_MIN_FILES);
        // The staging check fails: nothing is packed (never send more than per-file staging would).
        let stage = temp.path().join("fail");
        write(&stage, n)?;
        let (packs, kept) = pack_stage(&stage, &temp.path().join("p-fail"), |_| anyhow::bail!("offline"))?;
        assert!(packs.is_empty() && kept == 0);
        assert_eq!(fs::read_dir(stage.join("artifacts/frames"))?.count(), n);
        Ok(())
    }

    #[test]
    fn sfm_images_staged_only_for_jobs_that_name_them() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let run = temp.path().join("run");
        for f in ["artifacts/frames_1/images/frame_000000.jpg", "artifacts/frames_1/sfm_images/frame_000000.jpg", "artifacts/frames_1/selection.json"] {
            fs::create_dir_all(run.join(f).parent().unwrap())?;
            fs::write(run.join(f), b"x")?;
        }
        let key = run.join("artifacts/frames_1");
        for (want, stage) in [(false, temp.path().join("train")), (true, temp.path().join("sfm"))] {
            link_tree(&key, &stage, &run, false, want)?;
            assert!(stage.join("images/frame_000000.jpg").is_file() && stage.join("selection.json").is_file());
            assert_eq!(stage.join("sfm_images/frame_000000.jpg").is_file(), want);
        }
        assert!(is_sfm_images(Path::new("artifacts/frames_1/sfm_images")));
        assert!(!is_sfm_images(Path::new("artifacts/frames_1/sfm_images/frame_000000.jpg")));
        assert!(!is_sfm_images(Path::new("artifacts/frames_1/images")));
        Ok(())
    }

    #[test]
    fn video_staging_removes_verified_aliases_only_and_preserves_local_frames() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let run = temp.path().join("run");
        let stage = temp.path().join("stage");
        let original = run.join("artifacts/frames/images/frame_000001.jpg");
        fs::create_dir_all(original.parent().unwrap())?;
        fs::write(&original, vec![42; 100])?;
        let source = temp.path().join("source.mp4");
        let ffmpeg = temp.path().join("ffmpeg");
        let binary = temp.path().join("splat");
        for path in [&source, &binary] { fs::write(path, b"test")?; }
        for path in [&ffmpeg, &temp.path().join("ffprobe")] { fs::write(path, b"\x7fELFtest")?; }
        let recipe = serde_json::json!({"source":source,"ffmpeg":ffmpeg,"hashes":{"images/frame_000001.jpg":sha256(&original)?}});
        fs::write(run.join("artifacts/frames/video-replay.json"), serde_json::to_vec(&recipe)?)?;
        let alias = run.join("artifacts/sfm/images/frame_000001.jpg");
        fs::create_dir_all(alias.parent().unwrap())?;
        fs::hard_link(&original, &alias)?;
        let edited = run.join("artifacts/edited/images/frame_000001.jpg");
        fs::create_dir_all(edited.parent().unwrap())?;
        fs::write(&edited, b"different pixels")?;
        link_tree(&run, &stage, &run, false, false)?;
        let mut externals = BTreeSet::new();
        let manifest = prepare_video_replay(&run, &stage, &binary, &mut externals, None, 0)?.unwrap().manifest;
        assert!(original.is_file() && alias.is_file());
        assert!(!stage.join(original.strip_prefix(&run)?).exists());
        assert!(!stage.join(alias.strip_prefix(&run)?).exists());
        assert!(stage.join(edited.strip_prefix(&run)?).is_file());
        let tasks: Value = serde_json::from_slice(&fs::read(stage.join(manifest.strip_prefix(&run)?))?)?;
        assert_eq!(tasks[0]["copies"].as_array().unwrap().len(), 2);
        assert!(externals.contains(&source) && externals.contains(&ffmpeg));
        Ok(())
    }

    #[test]
    fn video_staging_keeps_quick_frames_when_source_is_larger() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let run = temp.path().join("run");
        let stage = temp.path().join("stage");
        let image = run.join("artifacts/frames/images/frame_000001.jpg");
        fs::create_dir_all(image.parent().unwrap())?;
        fs::write(&image, b"small")?;
        let source = temp.path().join("source.mp4");
        let ffmpeg = temp.path().join("ffmpeg");
        fs::write(&source, vec![42; 100])?;
        fs::write(&ffmpeg, b"\x7fELFtest")?;
        fs::write(temp.path().join("ffprobe"), b"\x7fELFtest")?;
        let recipe = serde_json::json!({"source":source,"ffmpeg":ffmpeg,"hashes":{"images/frame_000001.jpg":sha256(&image)?}});
        fs::write(run.join("artifacts/frames/video-replay.json"), serde_json::to_vec(&recipe)?)?;
        link_tree(&run, &stage, &run, false, false)?;
        assert!(prepare_video_replay(&run, &stage, &ffmpeg, &mut BTreeSet::new(), None, 0)?.is_none());
        assert!(stage.join(image.strip_prefix(&run)?).is_file());
        Ok(())
    }

    #[test]
    fn video_staging_keeps_frames_below_the_break_even_saving() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let run = temp.path().join("run");
        let stage = temp.path().join("stage");
        let image = run.join("artifacts/frames/images/frame_000001.jpg");
        fs::create_dir_all(image.parent().unwrap())?;
        fs::write(&image, vec![42; 100])?;
        let (source, ffmpeg) = (temp.path().join("source.mp4"), temp.path().join("ffmpeg"));
        fs::write(&source, b"test")?;
        for path in [&ffmpeg, &temp.path().join("ffprobe")] { fs::write(path, b"\x7fELFtest")?; }
        let recipe = serde_json::json!({"source":source,"ffmpeg":ffmpeg,"hashes":{"images/frame_000001.jpg":sha256(&image)?}});
        fs::write(run.join("artifacts/frames/video-replay.json"), serde_json::to_vec(&recipe)?)?;
        link_tree(&run, &stage, &run, false, false)?;
        let mut externals = BTreeSet::new();
        // 96 bytes saved net of the source: below a 97-byte break-even, at it with 96.
        assert!(prepare_video_replay(&run, &stage, &ffmpeg, &mut externals, None, 97)?.is_none());
        assert!(stage.join(image.strip_prefix(&run)?).is_file() && externals.is_empty());
        let replay = prepare_video_replay(&run, &stage, &ffmpeg, &mut externals, None, 96)?.unwrap();
        assert!(replay.sources.contains(&source) && !replay.cached.contains(&source) && !replay.stages);
        Ok(())
    }

    #[test]
    fn video_staging_skips_a_script_decoder() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let run = temp.path().join("run");
        let stage = temp.path().join("stage");
        let image = run.join("artifacts/frames/images/frame_000001.jpg");
        fs::create_dir_all(image.parent().unwrap())?;
        fs::write(&image, vec![42; 100])?;
        let (source, ffmpeg) = (temp.path().join("source.mp4"), temp.path().join("ffmpeg"));
        fs::write(&source, b"test")?;
        fs::write(&ffmpeg, b"#!/usr/bin/python3\n")?;
        fs::write(temp.path().join("ffprobe"), b"\x7fELFtest")?;
        let recipe = serde_json::json!({"source":source,"ffmpeg":ffmpeg,"hashes":{"images/frame_000001.jpg":sha256(&image)?}});
        fs::write(run.join("artifacts/frames/video-replay.json"), serde_json::to_vec(&recipe)?)?;
        link_tree(&run, &stage, &run, false, false)?;
        let mut externals = BTreeSet::new();
        assert!(prepare_video_replay(&run, &stage, &ffmpeg, &mut externals, None, 0)?.is_none());
        assert!(stage.join(image.strip_prefix(&run)?).is_file() && externals.is_empty());
        Ok(())
    }
}
