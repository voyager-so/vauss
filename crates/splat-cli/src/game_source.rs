//! `vauss export-game` for runs without exports/web (stripped, pruned or older runs): export the run's web
//! candidate (the one its exports/web came from) against that candidate's own SfM into a package next to the
//! game folder, then build the game from it. A candidate PLY kept on fal is fetched from the run's private
//! staging (splat_core::fal::materialize) into that package's work dir, never into the run.
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use splat_webexport::game::ExportGameArgs;

/// `export-game` for `a.source`, building a web package first when it is a run without exports/web.
pub fn run(mut a: ExportGameArgs) -> Result<Value> {
    let src = fs::canonicalize(&a.source).with_context(|| format!("{} not found", a.source.display()))?;
    if src.join("exports/web/scene.json").is_file() || src.join("scene.json").is_file() || !src.join("run.json").is_file() {
        return splat_webexport::game::run(&a);
    }
    let out = a.out.clone().unwrap_or_else(|| src.join("exports/game"));
    // A user's run gets its exports/web back; with --out nothing is written into the run.
    let pkg = if a.out.is_some() {
        let name = out.file_name().context("--out has no folder name")?.to_string_lossy().into_owned();
        out.with_file_name(format!("{name}-web"))
    } else {
        src.join("exports/web")
    };
    let (candidate, sfm, full) = web_candidate(&src)?;
    if a.full.is_none() {
        a.full = full;
    }
    // A candidate whose train job wrote its web package next to its splat: use that package as is.
    let own = src.join("artifacts").join(&candidate).join("web");
    let own_pkg = own.join("scene.json").is_file();
    let pkg = if own_pkg { own } else { pkg };
    let ply_rel = format!("artifacts/{candidate}/splat.ply");
    if !own_pkg && !reusable(&pkg, &candidate) {
        let work = pkg.with_file_name(format!(".{}-src", pkg.file_name().unwrap_or_default().to_string_lossy()));
        fs::create_dir_all(&work)?;
        let ply = local_ply(&src, &ply_rel, &work)?;
        eprintln!("export-game: {} has no exports/web; exporting {candidate} (SfM {}) to {}", src.display(),
            sfm.file_name().unwrap_or_default().to_string_lossy(), pkg.display());
        let r = splat_webexport::export_web_with_options(&ply, Some(&sfm), &pkg, &splat_webexport::SceneOptions {
            lod: splat_webexport::LodMode::Skip, max_splats: 1_000_000, ..Default::default() });
        // The PLY copy is only needed for the export (the game falls back to the package splats).
        if ply.starts_with(&work) { let _ = fs::remove_dir_all(&work); }
        r?;
        let mut scene: Value = serde_json::from_slice(&fs::read(pkg.join("scene.json"))?)?;
        scene["source"]["run"] = json!(src);
        scene["source"]["candidate"] = json!(candidate);
        fs::write(pkg.join("scene.json"), serde_json::to_vec_pretty(&scene)?)?;
    }
    if a.title.is_none() {
        a.title = run_title(&src);
    }
    a.source = pkg.clone();
    a.out = Some(out);
    let mut summary = splat_webexport::game::run(&a)?;
    summary["package"] = json!(pkg);
    summary["candidate"] = json!(candidate);
    Ok(summary)
}

/// A package this command exported from `candidate` before (kept next to the game folder).
fn reusable(pkg: &Path, candidate: &str) -> bool {
    fs::read(pkg.join("scene.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .is_some_and(|s| s["source"]["candidate"] == candidate && pkg.join(s["splat"].as_str().unwrap_or("scene.spz")).is_file())
}

/// The candidate the run's exports/web came from (journal: the last exported / interactive_ready /
/// preview_ready), else report.json's splat deliverable, else the run's last capped refine of another
/// candidate (an interactive model), else its last candidate with a web package; that candidate's SfM dir
/// (state.json meta); and the full model's web dir when the candidate is an interactive refine of one.
fn web_candidate(run: &Path) -> Result<(String, PathBuf, Option<PathBuf>)> {
    let events: Vec<Value> = fs::read_to_string(run.join("journal.jsonl")).unwrap_or_default().lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok()).collect();
    let state: Value = serde_json::from_slice(&fs::read(run.join("state.json")).context("the run has no state.json")?)?;
    let arts = &state["artifacts"];
    let has_web = |c: &str| run.join("artifacts").join(c).join("web/scene.json").is_file();
    let mut candidate = events.iter()
        .filter(|e| matches!(e["event"].as_str(), Some("exported" | "interactive_ready" | "preview_ready")))
        .filter_map(|e| e["data"]["candidate_id"].as_str().map(str::to_owned)).last();
    if candidate.is_none() {
        let report: Value = fs::read(run.join("report.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        candidate = report["deliverables"]["splat"].as_str()
            .and_then(|p| Path::new(p).parent()?.file_name()?.to_str().map(str::to_owned));
    }
    if candidate.is_none() {
        // Candidates in registration order.
        let order: Vec<String> = events.iter().filter_map(|e| {
            let list = e["data"]["result"]["artifacts"].as_array()?;
            Some(list.iter().filter_map(|a| a["id"].as_str().filter(|id| id.starts_with("candidate_")).map(str::to_owned)).collect::<Vec<_>>())
        }).flatten().collect();
        let interactive = |c: &String| { let r = &arts[c.as_str()]["meta"];
            r["max_splats"].as_u64().is_some_and(|m| m > 0) && r["request"]["init_candidate_id"].is_string() };
        candidate = order.iter().rev().find(|c| has_web(c) && interactive(c)).or_else(|| order.iter().rev().find(|c| has_web(c))).cloned();
    }
    let candidate = candidate.context("the run has no exported candidate (journal), splat deliverable (report.json) or candidate web package")?;
    let meta = &arts[&candidate]["meta"];
    let sfm = meta["sfm_id"].as_str().or(meta["request"]["sfm_id"].as_str())
        .with_context(|| format!("state.json: {candidate} names no sfm_id"))?;
    let dir = run.join("artifacts").join(sfm);
    if !dir.is_dir() { bail!("{}: the candidate's SfM is missing", dir.display()); }
    // The full model: the candidate this interactive model was refined from (journal or request).
    let from = events.iter().filter(|e| e["event"] == "interactive_started")
        .filter_map(|e| e["data"]["from_candidate"].as_str()).last().map(str::to_owned)
        .or_else(|| meta["request"]["init_candidate_id"].as_str().map(str::to_owned).filter(|_| meta["max_splats"].as_u64().is_some_and(|m| m > 0)));
    let full = from.filter(|f| *f != candidate).map(|f| run.join("artifacts").join(f).join("web")).filter(|w| w.join("scene.spz").is_file());
    Ok((candidate, dir, full))
}

/// The candidate PLY as a real local file: the run's own when present, else (a fal stub, or stripped from the
/// run but recorded in remote_artifacts.json) downloaded from the run's private fal staging into `work`.
fn local_ply(run: &Path, rel: &str, work: &Path) -> Result<PathBuf> {
    let path = run.join(rel);
    if path.is_file() && splat_core::fal::remote_stub(&path).is_none() {
        return Ok(path);
    }
    let (sha, size) = match splat_core::fal::remote_stub(&path) {
        Some(s) => s,
        None => {
            let kept: Value = [splat_core::fal::REMOTE_KEPT, "fal-remote-kept.json"].iter()
                .find_map(|r| serde_json::from_slice(&fs::read(run.join(r)).ok()?).ok())
                .with_context(|| format!("{} is missing and the run records no copy on fal", path.display()))?;
            let e = &kept[rel];
            (e["sha256"].as_str().with_context(|| format!("{rel}: not in the run's fal record"))?.to_owned(),
                e["size"].as_u64().with_context(|| format!("{rel}: no size in the run's fal record"))?)
        }
    };
    let dst = work.join("splat.ply");
    if fs::metadata(&dst).is_ok_and(|m| m.len() == size) {
        return Ok(dst);
    }
    // A stub in the work dir; materialize reads the run's records (app, scope) and replaces it with the file.
    fs::write(&dst, format!("splat-fal-remote v1 {sha} {size}\n"))?;
    splat_core::fal::materialize(run, &dst).with_context(|| format!("downloading {rel} from the run's fal staging"))?;
    Ok(dst)
}

/// The run's source file name (run.json), as export-viewer titles it.
fn run_title(run: &Path) -> Option<String> {
    let r: Value = serde_json::from_slice(&fs::read(run.join("run.json")).ok()?).ok()?;
    Path::new(r["source"].as_str()?).file_stem()?.to_str().map(str::to_owned)
}
