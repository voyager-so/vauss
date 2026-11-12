//! Rough render -> hosted video model(s). One record per model at `gen/<model>.json`; the queue
//! request is persisted at `gen/<model>.request.json` first, so a rerun resumes rather than re-pays.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::encode::Ffmpeg;
use crate::fal::{Fal, HttpError, Submitted};
use crate::layout::ShotDir;
use crate::models::{Ctx, Input, model};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenRecord {
    pub model: String,
    pub endpoint: String,
    pub note: String,
    pub payload: Value,
    pub request_id: Option<String>,
    pub wall_s: Option<f64>,
    pub usd_estimate: Option<f64>,
    /// Absolute path of the downloaded clip.
    pub video: Option<PathBuf>,
    pub video_url: Option<String>,
    pub error: Option<String>,
}

fn local_input(shot: &ShotDir, input: Input, ff: &Ffmpeg) -> Result<PathBuf> {
    Ok(match input {
        Input::RgbVideo => ff.upload_mp4(&shot.rgb_video(), &shot.gen_dir().join("rgb.fal.mp4"))?,
        Input::Depth16Video => {
            let (src, out) = (shot.depth_video(), shot.gen_dir().join("depth16.mp4"));
            if !crate::encode::newer_than(&out, &src) {
                ff.conform(&src, 16, 81, &out)?;
            }
            out
        }
        Input::DepthFull16Video => {
            let (src, out) = (shot.depth_video(), shot.gen_dir().join("depth16full.mp4"));
            if !crate::encode::newer_than(&out, &src) {
                ff.conform(&src, 16, crate::models::wan_frames(shot.path()?.duration_s), &out)?;
            }
            out
        }
        Input::FirstFrame => shot.frame("rgb", 0),
        Input::LastFrame => shot.frame("rgb", shot.render_meta()?.frames - 1),
        Input::Reference => shot.reference().unwrap_or_else(|| shot.frame("rgb", 0)),
    })
}

fn run_one(fal: &Fal, ff: &Ffmpeg, shot: &ShotDir, key: &str, prompt: &str, duration_s: f64) -> Result<GenRecord> {
    let m = model(key).with_context(|| format!("unknown video model {key:?}"))?;
    if m.h264_only && !matches!(ff.codec(), crate::encode::Codec::H264(_)) {
        bail!("{key} needs H.264 input and this ffmpeg has no working H.264 encoder (needs a Vulkan Video GPU, macOS, Windows, or openh264)");
    }
    let gen_dir = shot.gen_dir();
    let req_path = gen_dir.join(format!("{key}.request.json"));
    let t0 = Instant::now();
    let (sub, payload): (Submitted, Value) = if req_path.exists() {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&req_path)?)?;
        (serde_json::from_value(v["submitted"].clone())?, v["payload"].clone())
    } else {
        let urls = m.inputs.iter().map(|i| fal.upload(&local_input(shot, *i, ff)?)).collect::<Result<Vec<_>>>()?;
        let p = shot.path()?;
        let aspect_ratio = if p.height > p.width { "9:16" } else { "16:9" };
        let payload = (m.payload)(&Ctx { prompt, duration_s, urls: &urls, aspect_ratio });
        let sub = fal.submit(m.endpoint, &payload)?;
        let v = serde_json::json!({"submitted": sub, "payload": payload});
        std::fs::write(&req_path, serde_json::to_vec_pretty(&v)?)?;
        (sub, payload)
    };
    // A model-side failure (4xx on the result, or no video) is final: drop the persisted request so
    // a retry resubmits. Network errors keep it, so a retry resumes polling instead of paying again.
    let out = fal.wait(&sub, Duration::from_secs(1800)).inspect_err(|e| {
        if e.downcast_ref::<HttpError>().is_some_and(|h| (400..500).contains(&h.0)) {
            let _ = std::fs::remove_file(&req_path);
        }
    })?;
    let Some(url) = out["video"]["url"].as_str().map(str::to_string) else {
        let _ = std::fs::remove_file(&req_path);
        bail!("model output has no video.url: {}", out.to_string().chars().take(500).collect::<String>());
    };
    let dst = gen_dir.join(format!("{key}.mp4"));
    fal.download(&url, &dst)?;
    Ok(GenRecord {
        model: key.into(),
        endpoint: m.endpoint.into(),
        note: m.note.into(),
        payload,
        request_id: Some(sub.request_id),
        wall_s: Some((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0),
        usd_estimate: (m.usd_estimate)(duration_s),
        video: Some(dst),
        video_url: Some(url),
        error: None,
    })
}

/// Generate with each model concurrently. Already-finished models are skipped (idempotent).
pub fn generate(ff: &Ffmpeg, shot_dir: &Path, prompt: &str, models: &[String]) -> Result<Vec<GenRecord>> {
    let shot = ShotDir(shot_dir.to_path_buf());
    if !shot.rgb_video().exists() {
        bail!("shot is not rendered: {}", shot.rgb_video().display());
    }
    std::fs::create_dir_all(shot.gen_dir())?;
    let duration_s = shot.path()?.duration_s;
    let fal = Fal::from_env()?;
    let records = std::thread::scope(|s| {
        let handles: Vec<_> = models
            .iter()
            .map(|key| {
                let (fal, shot) = (&fal, &shot);
                s.spawn(move || {
                    let rec_path = shot.gen_dir().join(format!("{key}.json"));
                    let prev = std::fs::read(&rec_path).ok().and_then(|b| serde_json::from_slice::<GenRecord>(&b).ok());
                    if let Some(prev) = prev
                        && prev.error.is_none()
                        && prev.video.as_ref().is_some_and(|v| v.exists())
                    {
                        return prev;
                    }
                    let rec = run_one(fal, ff, shot, key, prompt, duration_s).unwrap_or_else(|e| GenRecord {
                        model: key.clone(),
                        endpoint: model(key).map(|m| m.endpoint.to_string()).unwrap_or_default(),
                        note: String::new(),
                        payload: Value::Null,
                        request_id: None,
                        wall_s: None,
                        usd_estimate: None,
                        video: None,
                        video_url: None,
                        error: Some(format!("{e:#}")),
                    });
                    if let Ok(bytes) = serde_json::to_vec_pretty(&rec) {
                        let _ = std::fs::write(&rec_path, bytes);
                    }
                    rec
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("generation thread panicked")).collect::<Vec<_>>()
    });
    Ok(records)
}
