//! Product ad from one trained splat: plan five distinct shots, render each, restyle its opening
//! frame into an ad still, regenerate the move with a video model, and cut a titled 15-20 s spot.
//!
//!   splat ad RUN_DIR --out DIR --product "a white leather sneaker" --title "AIRLINE ONE" --tagline "Walk lighter."
//!
//! Layout of DIR:
//!   plan.json            the shot plan: the first run only writes it (Claude); edit it, run again
//!   planner.json         the planner's prompt, raw reply and wall time
//!   <shot>/              a normal shot dir (path.json, render/, gen/) plus capture.<ext> (the
//!                        nearest real photo) and reference.png (the restyled opening still, which
//!                        is what Wan VACE gets as its appearance reference)
//!   ad.mp4               the cut (title card, shots with crossfades, end card, optional music)
//!   ad.json              what went into the cut and the fal spend
//!
//! Every step is idempotent: a shot whose path is unchanged is not re-rendered, a finished still or
//! clip is not re-bought, so editing one shot in plan.json and rerunning redoes only that shot.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_contracts::ShotMove;

use crate::encode::Ffmpeg;
use crate::fal::Fal;
use crate::generate::{self, GenRecord};
use crate::layout::ShotDir;
use crate::paths::{Direction, MoveSpec};
use crate::scene::SceneFrame;
use crate::{plan_into, render};

/// Image edit model that turns the rough opening frame into a styled ad still.
pub const STILL_ENDPOINT: &str = "fal-ai/nano-banana-pro/edit";
pub const STILL_USD: f64 = 0.15;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdShot {
    /// Short slug, also the shot dir name.
    pub name: String,
    #[serde(rename = "move")]
    pub movement: ShotMove,
    /// The look: set, light, palette. Drives the restyled still and the video prompt.
    pub style: String,
    /// Camera language for the video model ("slow macro push-in, shallow depth of field").
    pub camera: String,
    #[serde(default = "d_dur")]
    pub duration_s: f64,
    /// Orbit: degrees; other moves: scene radii.
    #[serde(default)]
    pub amount: Option<f64>,
    #[serde(default = "d_one")]
    pub distance_scale: f64,
    /// Scene radii along up (negative = low angle).
    #[serde(default)]
    pub height_offset: f64,
    #[serde(default)]
    pub vfov_deg: Option<f64>,
    #[serde(default)]
    pub clockwise: bool,
    /// Seconds of this shot used in the cut.
    #[serde(default = "d_cut")]
    pub cut_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdPlan {
    pub product: String,
    pub title: String,
    #[serde(default)]
    pub tagline: String,
    pub shots: Vec<AdShot>,
}

fn d_dur() -> f64 {
    5.0
}
fn d_one() -> f64 {
    1.0
}
fn d_cut() -> f64 {
    3.2
}

impl AdShot {
    pub fn spec(&self) -> MoveSpec {
        let mut s = MoveSpec::new(self.movement);
        s.duration_s = self.duration_s;
        s.amount = self.amount;
        s.distance_scale = self.distance_scale;
        s.height_offset = self.height_offset;
        s.vfov_deg = self.vfov_deg;
        s.direction = if self.clockwise { Direction::Cw } else { Direction::Ccw };
        s.prompt = Some(format!("{}. {}", self.camera, self.style));
        s
    }
}

/// Five moves x five looks; what the planner starts from and falls back to.
pub fn default_plan(product: &str, title: &str, tagline: &str) -> AdPlan {
    let shot = |name: &str, mv, style: &str, camera: &str| AdShot {
        name: name.into(),
        movement: mv,
        style: style.into(),
        camera: camera.into(),
        duration_s: 5.0,
        amount: None,
        distance_scale: 1.0,
        height_offset: 0.0,
        vfov_deg: None,
        clockwise: false,
        cut_s: 3.2,
    };
    let mut hero = shot("hero_orbit", ShotMove::Orbit, "on a seamless white studio sweep, soft high-key softbox light, clean shadow under the product", "slow hero orbit around the product");
    hero.amount = Some(70.0);
    let mut macro_ = shot("macro_push", ShotMove::PushIn, "moody dark set, black background, a single hard rim light tracing the edges, faint haze", "slow macro push-in toward the product, shallow depth of field");
    (macro_.amount, macro_.vfov_deg) = (Some(0.45), Some(32.0));
    let mut crane = shot("reveal_crane", ShotMove::CraneUp, "outdoors at golden hour, warm low sun, long soft shadows, lens flare", "low-angle crane up revealing the product");
    (crane.amount, crane.height_offset) = (Some(0.45), -0.2);
    let mut dolly = shot("rack_dolly", ShotMove::DollyRight, "neon cyberpunk night, magenta and cyan neon reflections, wet glossy surface", "lateral dolly with a rack focus from foreground to the product");
    (dolly.amount, dolly.vfov_deg) = (Some(0.5), Some(38.0));
    let mut top = shot("top_spin", ShotMove::Orbit, "minimal pastel set, soft peach and mint paper backdrop, diffuse even light", "high top-down spin over the product");
    (top.amount, top.height_offset, top.distance_scale, top.clockwise) = (Some(90.0), 0.8, 0.6, true);
    AdPlan { product: product.into(), title: title.into(), tagline: tagline.into(), shots: vec![hero, macro_, crane, dolly, top] }
}

/// Ask Claude (the `claude` CLI, print mode) to plan the five shots for this product and scene.
/// Returns the plan and a record of the exchange; None when the CLI is missing or the reply
/// doesn't parse (the caller then uses the default plan).
pub fn claude_plan(frame: &SceneFrame, product: &str, title: &str, tagline: &str) -> (Option<AdPlan>, Value) {
    let example = serde_json::to_string_pretty(&default_plan(product, title, tagline)).unwrap_or_default();
    let prompt = format!(
        "You are the director of a 15-20 second product ad made from a Gaussian splat of a real phone capture.\n\
         Product: {product}\nTitle card: {title}\nTagline: {tagline}\n\
         Scene frame (scene radius = median capture distance; azimuth coverage tells you which orbit arcs were actually seen):\n{scene}\n\n\
         Plan exactly 5 shots, each a different camera move AND a different look, ordered for a good edit \
         (open wide/hero, build detail, end on a strong product moment). Moves: orbit, push_in, pull_out, crane_up, crane_down, dolly_left, dolly_right. \
         Looks should vary across: studio white seamless, moody dark with rim light, outdoor golden hour, neon cyberpunk, minimal pastel (adapt them to the product). \
         Stay near the captured views: orbit arcs within the coverage, |height_offset| <= 0.8, distance_scale 0.5-1.3, push_in amount <= 0.6. \
         `style` describes set + light + palette (no camera words); `camera` is camera language for a video model. cut_s per shot 2.6-3.8, total of cut_s 14-17.\n\
         Reply with ONLY the JSON plan, same schema as this example:\n{example}",
        scene = serde_json::to_string_pretty(&frame.summary()).unwrap_or_default(),
    );
    let t0 = Instant::now();
    let out = Command::new("claude")
        .args(["-p", "--output-format", "json", "--model", "opus", "--tools", "", "--setting-sources", "", "--disable-slash-commands"])
        .arg(&prompt)
        .output();
    let wall_s = t0.elapsed().as_secs_f64();
    let mut rec = json!({"planner": "claude -p --model opus", "prompt": prompt, "wall_s": (wall_s * 10.0).round() / 10.0});
    let Ok(out) = out else {
        rec["error"] = "claude CLI not runnable".into();
        return (None, rec);
    };
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    if let Ok(v) = serde_json::from_str::<Value>(&stdout) {
        rec["claude"] = json!({"terminal_reason": v["terminal_reason"], "is_error": v["is_error"], "cost_usd": v["total_cost_usd"], "session_id": v["session_id"]});
    }
    if !out.status.success() || stdout.trim().is_empty() {
        rec["stderr"] = crate::tail(&String::from_utf8_lossy(&out.stderr), 1000).into();
    }
    let reply = serde_json::from_str::<Value>(&stdout)
        .ok()
        .and_then(|v| v["result"].as_str().map(str::to_string))
        .unwrap_or(stdout);
    rec["reply"] = reply.clone().into();
    let body = match (reply.find('{'), reply.rfind('}')) {
        (Some(a), Some(b)) if b > a => &reply[a..=b],
        _ => "",
    };
    match serde_json::from_str::<AdPlan>(body) {
        Ok(p) if !p.shots.is_empty() => (Some(p), rec),
        Ok(_) => {
            rec["error"] = "plan has no shots".into();
            (None, rec)
        }
        Err(e) => {
            rec["error"] = format!("unparseable plan: {e}").into();
            (None, rec)
        }
    }
}

fn video_prompt(plan: &AdPlan, s: &AdShot) -> String {
    format!(
        "Premium commercial product film of {product}. {style}. {camera}. The camera path and scene layout follow the input exactly; \
         the product keeps its exact shape, colors, materials and logos. Photorealistic, sharp detail, stable geometry, no text, no people.",
        product = plan.product.trim_end_matches('.'),
        style = s.style.trim_end_matches('.'),
        camera = s.camera.trim_end_matches('.'),
    )
}

fn still_prompt(plan: &AdPlan, s: &AdShot) -> String {
    format!(
        "Turn the first image into a premium advertising still: {style}. Keep {product} exactly as it is: identical shape, proportions, \
         colors, materials, logos and its position, size and angle in the frame; keep the camera viewpoint and framing. Change only the \
         set, background, surface and lighting. The second image is a real photo of the same product for detail. Photorealistic, no text, no people.",
        style = s.style.trim_end_matches('.'),
        product = plan.product.trim_end_matches('.'),
    )
}

/// Video models, preferred first.
const MODELS: &[&str] = &["wan_vace_depth"];
/// Title-card font (Bebas Neue, SIL OFL 1.1, see BebasNeue-OFL.txt), written next to the cut.
const FONT: &[u8] = include_bytes!("BebasNeue-Regular.ttf");

/// Append one paid call to DIR/spend.jsonl (tag = `<DIR name>-<call>`).
fn log_spend(out: &Path, tag: &str, endpoint: &str, usd: f64, request_id: &str) {
    let ts = Command::new("date").arg("--iso-8601=seconds").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let line = json!({"ts": ts, "kind": "job", "tag": tag, "usd": usd, "endpoint": endpoint, "request_id": request_id}).to_string();
    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(out.join("spend.jsonl")) {
        let _ = writeln!(fh, "{line}");
    }
}

/// Restyle the shot's opening frame (plus the real capture photo) into `reference.png`.
fn styled_still(fal: &Fal, out: &Path, tag: &str, shot: &ShotDir, prompt: &str) -> Result<PathBuf> {
    let dst = shot.0.join("reference.png");
    if dst.exists() {
        return Ok(dst);
    }
    // reference.dl is the still as downloaded; it is kept until converted, so a crash after
    // paying for it never pays twice.
    let tmp = shot.0.join("reference.dl");
    if !tmp.exists() {
        let mut urls = vec![fal.upload(&shot.frame("rgb", 0))?];
        if let Some(cap) = std::fs::read_dir(&shot.0)?.flatten().map(|e| e.path()).find(|p| p.file_stem().is_some_and(|s| s == "capture")) {
            urls.push(fal.upload(&cap)?);
        }
        let payload = json!({"prompt": prompt, "image_urls": urls, "num_images": 1, "aspect_ratio": "16:9", "output_format": "png", "resolution": "1K"});
        let sub = fal.submit(STILL_ENDPOINT, &payload)?;
        let res = fal.wait(&sub, Duration::from_secs(600))?;
        log_spend(out, &format!("{tag}-still-{}", shot.0.file_name().unwrap_or_default().to_string_lossy()), STILL_ENDPOINT, STILL_USD, &sub.request_id);
        let url = res["images"][0]["url"].as_str().with_context(|| format!("still has no image: {res}"))?;
        fal.download(url, &tmp)?;
        std::fs::write(shot.0.join("reference.json"), serde_json::to_vec_pretty(&json!({"endpoint": STILL_ENDPOINT, "request_id": sub.request_id, "prompt": prompt, "url": url}))?)?;
    }
    // Wan wants the reference at the video's size; the edit model may return a different aspect.
    let img = image::ImageReader::open(&tmp)?.with_guessed_format()?.decode()?.resize_to_fill(1280, 720, image::imageops::FilterType::Lanczos3);
    img.save(&dst)?;
    std::fs::remove_file(&tmp)?;
    Ok(dst)
}

pub struct AdArgs<'a> {
    pub ply: &'a Path,
    pub frame: &'a SceneFrame,
    pub out: &'a Path,
    pub plan: AdPlan,
    pub music: Option<PathBuf>,
}

/// Plan, render, restyle, generate and cut. Returns ad.json's content.
///
/// Without a fal key (e.g. on a headless GPU box) it stops after the renders and cuts from them;
/// rerunning where a key is available reuses the renders and adds the stills and generated clips.
pub fn make_ad(ff: &Ffmpeg, a: AdArgs) -> Result<Value> {
    let t0 = Instant::now();
    let tag = a.out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "ad".into());
    let models: Vec<String> = MODELS.iter().map(|m| m.to_string()).collect();
    std::fs::create_dir_all(a.out)?;
    std::fs::write(a.out.join("plan.json"), serde_json::to_vec_pretty(&a.plan)?)?;
    // Plan + render every shot first (GPU), then all fal work runs concurrently.
    let mut dirs = vec![];
    let mut todo: Vec<(crate::paths::CameraPath, PathBuf)> = vec![];
    for s in &a.plan.shots {
        let dir = a.out.join(&s.name);
        let shot = ShotDir(dir.clone());
        let old = std::fs::read(dir.join("path.json")).ok();
        let (path, risk) = plan_into(a.frame, &s.spec(), &dir, &s.name)?;
        // plan_into copies the nearest capture photo to reference.<ext>; keep it as capture.<ext>
        // so `reference` can be the restyled still.
        for e in std::fs::read_dir(&dir)?.flatten() {
            let p = e.path();
            if p.file_stem().is_some_and(|x| x == "reference") && p.extension().is_some_and(|x| x != "png" && x != "json" && x != "dl") {
                std::fs::rename(&p, p.with_file_name(format!("capture.{}", p.extension().unwrap().to_string_lossy())))?;
            }
        }
        let changed = old.as_deref() != Some(std::fs::read(dir.join("path.json"))?.as_slice());
        if changed || !shot.render_dir().join("render.json").exists() {
            let _ = std::fs::remove_dir_all(shot.gen_dir());
            let _ = std::fs::remove_file(dir.join("reference.png"));
            todo.push((path, shot.render_dir()));
        }
        eprintln!("[ad] {}: planned ({} risk, novelty max {:.2})", s.name, risk.risk, risk.novelty_max);
        dirs.push(shot);
    }
    let jobs: Vec<(&crate::paths::CameraPath, &Path)> = todo.iter().map(|(p, d)| (p, d.as_path())).collect();
    for m in render::render_paths(a.ply, &jobs, ff)? {
        eprintln!("[ad] rendered {} frames, alpha mean {:.3}, raster {:.1}s", m.frames, m.alpha_mean, m.render_s);
    }
    let mut gens: Vec<Vec<GenRecord>> = vec![vec![]; dirs.len()];
    let fal = Fal::from_env().map_err(|e| eprintln!("[ad] no fal key ({e:#}): cutting from the splat renders")).ok();
    if let Some(fal) = &fal {
        gens = std::thread::scope(|sc| {
            let hs: Vec<_> = a
                .plan
                .shots
                .iter()
                .zip(&dirs)
                .map(|(s, shot)| {
                    let (plan, a, tag, models) = (&a.plan, &a, &tag, &models);
                    sc.spawn(move || -> Result<Vec<GenRecord>> {
                        styled_still(fal, a.out, tag, shot, &still_prompt(plan, s))?;
                        let prev_ok: Vec<String> = models.iter().filter(|m| shot.gen_dir().join(format!("{m}.mp4")).exists()).cloned().collect();
                        let recs = generate::generate(ff, &shot.0, &video_prompt(plan, s), models)?;
                        for r in &recs {
                            if r.error.is_none() && !prev_ok.contains(&r.model) {
                                log_spend(a.out, &format!("{tag}-{}-{}", s.name, r.model), &r.endpoint, r.usd_estimate.unwrap_or(0.0), r.request_id.as_deref().unwrap_or(""));
                            }
                        }
                        eprintln!("[ad] {}: generated {:?}", s.name, recs.iter().map(|r| (r.model.as_str(), r.error.is_none())).collect::<Vec<_>>());
                        Ok(recs)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("ad thread panicked")).collect::<Result<Vec<_>>>()
        })?;
    }
    // Per shot: the first model (in --model order) that produced a clip, else the vauss render.
    let clips: Vec<(PathBuf, String)> = dirs
        .iter()
        .zip(&gens)
        .map(|(shot, recs)| {
            models
                .iter()
                .find_map(|m| recs.iter().find(|r| &r.model == m && r.error.is_none()).and_then(|r| r.video.clone()).map(|v| (v, m.clone())))
                .unwrap_or_else(|| (shot.rgb_video(), "splat_render".into()))
        })
        .collect();
    let cut = cut_ad(ff, &a, &clips)?;
    let usd: f64 = std::fs::read_to_string(a.out.join("spend.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok()?["usd"].as_f64())
        .sum();
    let rec = json!({
        "ad": cut,
        "plan": a.plan,
        "shots": a.plan.shots.iter().zip(&clips).map(|(s, (v, src))| json!({"name": s.name, "clip": v, "source": src, "cut_s": s.cut_s})).collect::<Vec<_>>(),
        "fal_usd_total": (usd * 100.0).round() / 100.0,
        "wall_s": t0.elapsed().as_secs_f64().round(),
    });
    std::fs::write(a.out.join("ad.json"), serde_json::to_vec_pretty(&rec)?)?;
    Ok(rec)
}

fn esc(t: &str) -> String {
    t.replace('\\', "\\\\").replace('\'', "\u{2019}").replace(':', "\\:").replace('%', "\\%")
}

/// Title card + shots (crossfaded) + end card, 1280x720 24 fps, H.264 (or VP9) with optional music.
fn cut_ad(ff: &Ffmpeg, a: &AdArgs, clips: &[(PathBuf, String)]) -> Result<PathBuf> {
    let (w, h, fps, xd) = (1280, 720, 24, 0.35_f64);
    let (title_s, end_s) = (2.0_f64, 2.6_f64);
    let font_path = a.out.join("title-font.ttf");
    std::fs::write(&font_path, FONT)?;
    let font = format!("fontfile='{}'", font_path.display());
    let codec = ff.codec();
    let (pre, vf_tail, enc_args) = codec.parts();
    let mut args: Vec<String> = vec!["-y".into(), "-hide_banner".into(), "-loglevel".into(), "error".into()];
    args.extend(pre.iter().map(|s| s.to_string()));
    let mut fg = String::new();
    // [0] title card, [1..=n] shots, [n+1] end card, [n+2] music
    let card = |d: f64| format!("color=c=0x0b0b0d:s={w}x{h}:r={fps}:d={d}");
    args.extend(["-f".into(), "lavfi".into(), "-i".into(), card(title_s)]);
    for (v, _) in clips {
        args.extend(["-i".into(), v.to_string_lossy().into_owned()]);
    }
    args.extend(["-f".into(), "lavfi".into(), "-i".into(), card(end_s)]);
    let n = clips.len();
    let title = esc(&a.plan.title);
    let tagline = esc(&a.plan.tagline);
    fg += &format!(
        "[0:v]drawtext={font}:text='{title}':fontsize=150:fontcolor=white:x=(w-tw)/2:y=(h-th)/2:alpha='min(1,t/0.6)',format=yuv420p,setsar=1[c0];"
    );
    for (i, s) in a.plan.shots.iter().enumerate().take(n) {
        // skip the first 0.4 s: generated clips often settle in
        fg += &format!(
            "[{k}:v]trim=start=0.4:duration={d},setpts=PTS-STARTPTS,fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=yuv420p,setsar=1[c{k}];",
            k = i + 1,
            d = s.cut_s + xd,
        );
    }
    fg += &format!(
        "[{k}:v]drawtext={font}:text='{title}':fontsize=130:fontcolor=white:x=(w-tw)/2:y=(h-th)/2-50:alpha='min(1,t/0.5)',\
         drawtext={font}:text='{tagline}':fontsize=48:fontcolor=0xd0d0d0:x=(w-tw)/2:y=(h/2)+60:alpha='min(1,max(0,(t-0.4)/0.6))',format=yuv420p,setsar=1[c{k}];",
        k = n + 1
    );
    let lens: Vec<f64> = std::iter::once(title_s).chain(a.plan.shots.iter().take(n).map(|s| s.cut_s + xd)).chain([end_s]).collect();
    let mut acc = lens[0];
    let mut prev = "c0".to_string();
    for k in 1..lens.len() {
        let off = acc - xd;
        let trans = if k == 1 || k == lens.len() - 1 { "fadeblack" } else { "fade" };
        let label = if k == lens.len() - 1 { "vcut".to_string() } else { format!("x{k}") };
        fg += &format!("[{prev}][c{k}]xfade=transition={trans}:duration={xd}:offset={off:.3}[{label}];");
        acc = off + lens[k];
        prev = label;
    }
    let total = acc;
    fg += &format!("[vcut]{vf_tail}[vout];");
    let mut maps = vec!["-map".to_string(), "[vout]".into()];
    if let Some(m) = &a.music {
        args.extend(["-i".into(), m.to_string_lossy().into_owned()]);
        fg += &format!("[{k}:a]atrim=duration={total:.3},afade=t=in:d=0.3,afade=t=out:st={:.3}:d=1.5,volume=0.9[aout];", total - 1.5, k = n + 2);
        maps.extend(["-map".into(), "[aout]".into(), "-c:a".into(), (if codec.ext() == "mp4" { "aac" } else { "libopus" }).into(), "-b:a".into(), "192k".into()]);
    }
    let fg = fg.trim_end_matches(';').to_string();
    args.extend(["-filter_complex".into(), fg]);
    args.extend(maps);
    args.extend(enc_args.iter().map(|s| s.to_string()));
    let out = a.out.join(format!("ad.{}", codec.ext()));
    args.extend(["-r".into(), fps.to_string()]);
    if codec.ext() == "mp4" {
        args.extend(["-movflags".into(), "+faststart".into()]);
    }
    args.push(out.to_string_lossy().into_owned());
    let o = Command::new(ff.path()).args(&args).output().with_context(|| format!("running {}", ff.path().display()))?;
    if !o.status.success() {
        bail!("ad cut failed: {}", String::from_utf8_lossy(&o.stderr));
    }
    eprintln!("[ad] cut {} ({total:.1} s)", out.display());
    Ok(out)
}
