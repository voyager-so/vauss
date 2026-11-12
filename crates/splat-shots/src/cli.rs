//! Shot commands that work on a raw trained output (no Run), plus the job worker entry.
//!
//!   splat __shots shot RUN_DIR --out DIR --prompt "slow orbit at dusk, 5s" --scene "a stone statue"
//!   splat __shots generate SHOT_DIR --prompt "..." --model luma_ray32_v2v --model wan_vace_depth
//!   splat __shots review SHOT_DIR... --out index.html
//!
//! RUN_DIR is a Spirula output dir (config.json + scene_transform.json + step-*.ckpt/splat.ply);
//! `--ply`/`--data` override for other layouts.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::json;

use crate::encode::Ffmpeg;
use crate::layout::ShotDir;
use crate::paths::spec_from_prompt;
use crate::scene::load_scene;
use crate::{ad, blender, gen_prompt, generate, plan_into, plan_path_into, render, review, scene_transform_near};

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(clap::Args)]
pub struct Source {
    /// Trained output dir (Spirula layout).
    pub run_dir: Option<PathBuf>,
    /// Splat PLY (default: latest step-*.ckpt/splat.ply in RUN_DIR).
    #[arg(long)]
    pub ply: Option<PathBuf>,
    /// COLMAP dataset dir (default: `data` in RUN_DIR/config.json).
    #[arg(long)]
    pub data: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Print the scene frame (up, focus, radius, coverage) used for planning.
    Scene(Source),
    /// Plan + render (+ generate, + Blender bundle) one shot.
    Shot {
        #[command(flatten)]
        src: Source,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "slow orbit, 5s")]
        prompt: String,
        /// Render this CameraPath JSON (training frame) instead of planning from --prompt.
        #[arg(long)]
        path: Option<PathBuf>,
        /// What the scene is, for the video model.
        #[arg(long, default_value = "")]
        scene: String,
        #[arg(long, default_value = "")]
        style: String,
        #[arg(long = "model")]
        models: Vec<String>,
        #[arg(long)]
        mesh: Option<PathBuf>,
        #[arg(long)]
        blender: bool,
    },
    /// Product ad: plan 5 shots (Claude, unless DIR/plan.json exists), render, restyle, generate, cut a titled spot.
    Ad {
        #[command(flatten)]
        src: Source,
        #[arg(long)]
        out: PathBuf,
        /// What the product is ("a white leather sneaker").
        #[arg(long)]
        product: String,
        #[arg(long, default_value = "")]
        title: String,
        #[arg(long, default_value = "")]
        tagline: String,
        /// Music track for the cut (record its license).
        #[arg(long)]
        music: Option<PathBuf>,
    },
    /// Fix shaky footage: fit a smooth path to the capture's own camera trajectory (same timing,
    /// level horizon), render it, cut the matching "before" clip, optionally generate.
    Stabilize {
        #[command(flatten)]
        src: Source,
        #[arg(long)]
        out: PathBuf,
        /// The original clip the capture frames came from (cut to the same span as `capture.mp4`).
        #[arg(long)]
        video: Option<PathBuf>,
        /// 16:9 or 9:16.
        #[arg(long, default_value = "16:9")]
        aspect: String,
        #[arg(long, default_value = "")]
        scene: String,
        #[arg(long, default_value = "")]
        style: String,
        #[arg(long = "model")]
        models: Vec<String>,
    },
    /// Triage clips: can each be reconstructed (camera moves through a static scene with parallax)?
    Triage {
        clips: Vec<PathBuf>,
        /// Write the table as JSON here (also printed).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Send an already-rendered shot to video model(s). Calls third-party hosted models on fal.ai, billed
    /// to your FAL_KEY; use of the outputs is governed by each model provider's terms.
    Generate {
        shot_dir: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long = "model", required = true)]
        models: Vec<String>,
    },
    /// Side-by-side HTML review of shot dirs.
    Review {
        shot_dirs: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "Shot review")]
        title: String,
    },
}

fn resolve(src: &Source) -> Result<(PathBuf, PathBuf)> {
    let cfg = src.run_dir.as_ref().map(|d| d.join("config.json")).filter(|p| p.exists());
    let data = match (&src.data, &cfg) {
        (Some(d), _) => d.clone(),
        (None, Some(c)) => spirula_config_data(&std::fs::read_to_string(c)?).context("config.json has no \"data\"")?,
        _ => anyhow::bail!("give RUN_DIR with config.json, or --data"),
    };
    let ply = match (&src.ply, &src.run_dir) {
        (Some(p), _) => p.clone(),
        (None, Some(d)) => crate::find_ply(d).with_context(|| format!("no splat.ply under {}", d.display()))?,
        _ => anyhow::bail!("give RUN_DIR or --ply"),
    };
    Ok((ply, data))
}

/// Spirula's config.json is not strict JSON (it writes `Infinity`), so pull out just `"data"`.
fn spirula_config_data(text: &str) -> Option<PathBuf> {
    let re = regex::Regex::new(r#""data"\s*:\s*("(?:[^"\\]|\\.)*")"#).unwrap();
    let quoted = re.captures(text)?.get(1)?.as_str();
    serde_json::from_str::<String>(quoted).ok().map(PathBuf::from)
}

/// Frame rate of a video via the ffprobe next to ffmpeg (or on PATH).
fn probe_fps(ff: &Ffmpeg, video: &Path) -> Option<f64> {
    let probe = ff.path().with_file_name(ff.path().file_name()?.to_str()?.replace("ffmpeg", "ffprobe"));
    let probe = if probe.exists() { probe } else { PathBuf::from("ffprobe") };
    let out = std::process::Command::new(probe)
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0"])
        .arg(video)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let (n, d) = s.trim().split_once('/')?;
    let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
    (d > 0.0).then(|| n / d)
}

/// The original clip over the span the path covers, at the path's fps and frame size (center crop),
/// so before/after line up frame for frame.
fn cut_capture(ff: &Ffmpeg, video: &Path, rep: &crate::follow::FollowReport, path: &crate::paths::CameraPath, stem: &Path) -> Result<PathBuf> {
    let (w, h) = (path.width, path.height);
    let vf = format!(
        "fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
        fps = path.fps
    );
    let (ss, t) = (format!("{:.3}", rep.src_start_s), format!("{:.3}", rep.src_end_s - rep.src_start_s));
    let n = path.frame_count().to_string();
    let input = ["-ss", &ss, "-t", &t, "-i", video.to_str().context("non-UTF-8 path")?];
    ff.encode_clip(&input, &vf, &["-an", "-frames:v", &n], stem)
}

fn write_shot_json(dir: &Path, name: &str, prompt: &str, generated: &[generate::GenRecord]) -> Result<()> {
    let shot = ShotDir(dir.to_path_buf());
    let risk: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("risk.json"))?)?;
    let rec = json!({
        "name": name,
        "risk": {"risk": risk["risk"], "novelty_mean": risk["novelty_mean"], "novelty_max": risk["novelty_max"], "frac_frames_far": risk["frac_frames_far"]},
        "render": shot.render_meta()?,
        "prompt": prompt,
        "generated": generated,
    });
    std::fs::write(dir.join("shot.json"), serde_json::to_vec_pretty(&rec)?)?;
    Ok(())
}

/// Run with tools from SPLAT_FFMPEG / SPLAT_BLENDER or PATH (dev and job workers).
pub fn run(cmd: Cmd) -> Result<()> {
    run_with(cmd, &Ffmpeg::from_env(), &blender::blender_exe())
}

/// Run with tool paths the caller already resolved (the `splat` CLI's toolchain config).
pub fn run_with(cmd: Cmd, ff: &Ffmpeg, blender_bin: &Path) -> Result<()> {
    match cmd {
        Cmd::Scene(src) => {
            let (ply, data) = resolve(&src)?;
            println!("{}", serde_json::to_string_pretty(&load_scene(&data, scene_transform_near(&ply).as_deref())?.summary())?);
        }
        Cmd::Shot { src, out, prompt, path: path_file, scene, style, models, mesh, blender: with_blender } => {
            let (ply, data) = resolve(&src)?;
            let frame = load_scene(&data, scene_transform_near(&ply).as_deref())?;
            let spec = spec_from_prompt(&prompt, None);
            let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "shot".into());
            let (path, risk) = match &path_file {
                Some(f) => {
                    let p: crate::paths::CameraPath = serde_json::from_slice(&std::fs::read(f)?).with_context(|| format!("parsing {}", f.display()))?;
                    let r = plan_path_into(&frame, &p, &out)?;
                    (p, r)
                }
                None => plan_into(&frame, &spec, &out, &name)?,
            };
            let meta = render::render_path(&ply, &path, &out.join("render"), ff)?;
            let gen_text = gen_prompt(&path, &scene, &style);
            let generated = if models.is_empty() { vec![] } else { generate::generate(ff, &out, &gen_text, &models)? };
            write_shot_json(&out, &name, &gen_text, &generated)?;
            let mut summary = json!({"risk": risk.risk, "novelty_max": risk.novelty_max, "render": meta, "generated": generated});
            if with_blender {
                let b = blender::export_bundle(&frame, &ply, &path, &out.join("blender"), mesh.as_deref())?;
                let n = path.frame_count();
                summary["blender"] = json!({"bundle": b, "build": blender::build_blend(blender_bin, &out.join("blender"), &[0, n / 2, n - 1])?});
            }
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Cmd::Ad { src, out, product, title, tagline, music } => {
            let (ply, data) = resolve(&src)?;
            let frame = load_scene(&data, scene_transform_near(&ply).as_deref())?;
            let title = if title.is_empty() { product.to_uppercase() } else { title };
            // The first run only plans: Claude writes DIR/plan.json (built-in plan if Claude is
            // unavailable) for review; the next run renders, generates and cuts it.
            let plan_file = out.join("plan.json");
            if !plan_file.exists() {
                std::fs::create_dir_all(&out)?;
                let (p, rec) = ad::claude_plan(&frame, &product, &title, &tagline);
                std::fs::write(out.join("planner.json"), serde_json::to_vec_pretty(&rec)?)?;
                let plan = p.unwrap_or_else(|| {
                    eprintln!("[ad] planner gave no plan ({}); using the built-in one", rec["error"]);
                    ad::default_plan(&product, &title, &tagline)
                });
                std::fs::write(&plan_file, serde_json::to_vec_pretty(&plan)?)?;
                println!("{}", serde_json::to_string_pretty(&plan)?);
                eprintln!("[ad] wrote {}; review or edit it, then run the same command again to make the ad", plan_file.display());
                return Ok(());
            }
            let plan = serde_json::from_slice(&std::fs::read(&plan_file)?).with_context(|| format!("parsing {}", plan_file.display()))?;
            let rec = ad::make_ad(ff, ad::AdArgs { ply: &ply, frame: &frame, out: &out, plan, music })?;
            println!("{}", serde_json::to_string_pretty(&rec)?);
        }
        Cmd::Stabilize { src, out, video, aspect, scene, style, models } => {
            let (ply, data) = resolve(&src)?;
            let frame = load_scene(&data, scene_transform_near(&ply).as_deref())?;
            let src_fps = video.as_deref().and_then(|v| probe_fps(ff, v)).unwrap_or(30.0);
            let (width, height) = if aspect == "9:16" { (720, 1280) } else { (1280, 720) };
            let spec = crate::follow::FollowSpec { src_fps, width, height, ..Default::default() };
            let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "stabilize".into());
            let (path, report) = crate::follow::plan_follow(&frame, &spec, &name)?;
            let risk = crate::plan_path_into(&frame, &path, &out)?;
            std::fs::write(out.join("follow.json"), serde_json::to_vec_pretty(&report)?)?;
            let capture = match &video {
                Some(v) => Some(cut_capture(ff, v, &report, &path, &out.join("capture"))?),
                None => None,
            };
            let meta = render::render_path(&ply, &path, &out.join("render"), ff)?;
            let look = if style.is_empty() { "as captured, natural daylight" } else { style.as_str() };
            let parts: Vec<&str> = [scene.trim(), look.trim()].into_iter().filter(|s| !s.is_empty()).collect();
            let gen_text = format!(
                "{}. Smooth stabilized gimbal walk-along, the camera path and scene layout follow the input exactly. Photorealistic, sharp detail, stable geometry, no new objects.",
                parts.join(". ")
            );
            let generated = if models.is_empty() { vec![] } else { generate::generate(ff, &out, &gen_text, &models)? };
            write_shot_json(&out, &name, &gen_text, &generated)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({"follow": report, "risk": risk.risk, "novelty_max": risk.novelty_max, "capture": capture, "render": meta, "generated": generated}))?
            );
        }
        Cmd::Triage { clips, out } => {
            let ffprobe = ff.path().with_file_name(ff.path().file_name().and_then(|n| n.to_str()).unwrap_or("ffmpeg").replace("ffmpeg", "ffprobe"));
            let opts = crate::triage::TriageOptions {
                ffmpeg: ff.path().to_path_buf(),
                ffprobe: if ffprobe.exists() { ffprobe } else { "ffprobe".into() },
                ..Default::default()
            };
            let rows: Vec<serde_json::Value> = clips
                .iter()
                .zip(crate::triage::triage_clips(&clips, &opts))
                .map(|(c, r)| match r {
                    Ok(t) => serde_json::to_value(t).unwrap_or_default(),
                    Err(e) => json!({"file": c, "capturable": false, "verdict": "error", "reason": format!("{e:#}")}),
                })
                .collect();
            for r in &rows {
                eprintln!("{:<40} {:<15} {}", r["file"].as_str().unwrap_or(""), r["verdict"].as_str().unwrap_or(""), r["reason"].as_str().unwrap_or(""));
            }
            let text = serde_json::to_string_pretty(&rows)?;
            if let Some(o) = out {
                std::fs::write(o, &text)?;
            }
            println!("{text}");
        }
        Cmd::Generate { shot_dir, prompt, models } => {
            let recs = generate::generate(ff, &shot_dir, &prompt, &models)?;
            let name = shot_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if shot_dir.join("render/render.json").exists() {
                write_shot_json(&shot_dir, &name, &prompt, &recs)?;
            }
            println!("{}", serde_json::to_string_pretty(&recs)?);
            let failed: Vec<_> = recs.iter().filter(|r| r.error.is_some()).map(|r| r.model.as_str()).collect();
            anyhow::ensure!(failed.len() < recs.len(), "every model failed: {failed:?}");
        }
        Cmd::Review { shot_dirs, out, title } => {
            println!("{}", review::write_review(&shot_dirs, &out, &title)?.display());
        }
    }
    Ok(())
}
