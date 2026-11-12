//! Prompt (or image) -> image -> 3D GLB -> normalized catalog item, on fal; plus scene restyling.
//!
//! Every fal request is persisted under `<out_dir>/work/<id>/` before polling, and finished steps
//! leave their outputs there, so rerunning with the same id resumes instead of paying twice.
//! Endpoints and payloads live in small fns here so a model can be swapped in one place.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_shots::fal::{Fal, HttpError, Submitted};

use crate::glb::{NormalizeOpts, SizeTarget, normalize_glb, thumb_webp};
use crate::{Catalog, CatalogItem, category_target, guess_category, log_spend, new_id, valid_id, write_atomic};

pub const IMAGE_ENDPOINT: &str = "fal-ai/flux/schnell";
pub const MESH_ENDPOINT: &str = "fal-ai/hyper3d/rodin/v2.5/fast";
pub const RESTYLE_ENDPOINT: &str = "fal-ai/flux-pro/kontext";
/// Per-call USD estimates for the spend ledger (placeholders until checked against invoices).
pub const USD_IMAGE: f64 = 0.003;
pub const USD_MESH: f64 = 0.10;
pub const USD_RESTYLE: f64 = 0.04;

const THUMB_PX: u32 = 256;

pub fn image_payload(prompt: &str) -> Value {
    json!({
        "prompt": format!("{prompt}, single object, product photo, three-quarter view, centered, isolated on pure white background, soft studio lighting, entire object visible, no shadow"),
        "image_size": "square_hd",
        "num_inference_steps": 4,
        "num_images": 1,
    })
}

pub fn mesh_payload(image_url: &str) -> Value {
    json!({"image_urls": [image_url], "material": "PBR"})
}

pub fn restyle_payload(prompt: &str, image_url: &str) -> Value {
    json!({"prompt": prompt, "image_url": image_url, "output_format": "jpeg"})
}

const KEEP: &str = "Keep every object, wall, piece of furniture, and the camera viewpoint exactly the same; change only the lighting and season.";

pub const PRESETS: &[&str] = &["golden_hour", "sunset", "night", "snow", "autumn"];

pub fn preset_prompt(preset: &str) -> Option<String> {
    let look = match preset {
        "golden_hour" => "Relight this exact scene at golden hour: warm low sun through the windows, long soft shadows.",
        "sunset" => "Relight this exact scene at sunset: deep orange and pink light from the windows, dimmer interior, warm glow on surfaces.",
        "night" => "Relight this exact scene at night: dark blue sky outside the windows, warm interior lamps on, soft pools of lamplight.",
        "snow" => "Show this exact scene on a snowy winter day: snow visible outside the windows, cool soft overcast daylight.",
        "autumn" => "Show this exact scene on an autumn afternoon: orange and red foliage outside the windows, warm diffuse daylight.",
        _ => return None,
    };
    Some(format!("{look} {KEEP}"))
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenRequest {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing)]
    pub image_b64: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub dims_m: Option<[f32; 3]>,
    /// Reuse an earlier item id to resume its work dir (new id when absent).
    #[serde(default)]
    pub id: Option<String>,
    /// Category override (else guessed from the prompt/name).
    #[serde(default)]
    pub category: Option<String>,
    /// Spend-log tag prefix (default "app-decorator-gen"); the item id is appended.
    #[serde(default)]
    pub tag: Option<String>,
}

/// Decode a (possibly `data:` prefixed) base64 image; non png/jpeg inputs are converted to PNG.
/// Returns `(bytes, extension, mime)`.
pub fn decode_image_b64(s: &str) -> Result<(Vec<u8>, &'static str, &'static str)> {
    let data = s.split_once("base64,").map_or(s, |(_, d)| d).trim();
    let bytes = base64::engine::general_purpose::STANDARD.decode(data).context("image_b64 is not valid base64")?;
    match image::guess_format(&bytes).context("image_b64 is not a recognized image")? {
        image::ImageFormat::Png => Ok((bytes, "png", "image/png")),
        image::ImageFormat::Jpeg => Ok((bytes, "jpg", "image/jpeg")),
        _ => {
            let img = image::load_from_memory(&bytes)?;
            let mut out = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;
            Ok((out, "png", "image/png"))
        }
    }
}

/// Submit (or resume) one fal request persisted at `req_path`, wait, return its output.
pub(crate) fn fal_call(fal: &Fal, endpoint: &str, payload: &Value, req_path: &Path, tag: &str, usd: f64, timeout: Duration, log: &dyn Fn(&str)) -> Result<Value> {
    let sub: Submitted = if req_path.exists() {
        let v: Value = serde_json::from_slice(&std::fs::read(req_path)?)?;
        let sub: Submitted = serde_json::from_value(v["submitted"].clone())?;
        log(&format!("resuming {endpoint} request {}", sub.request_id));
        sub
    } else {
        let sub = fal.submit(endpoint, payload)?;
        write_atomic(req_path, &serde_json::to_vec_pretty(&json!({"submitted": sub, "payload": payload}))?)?;
        log_spend(tag, usd, endpoint, &sub.request_id, log);
        log(&format!("submitted {endpoint} request {}", sub.request_id));
        sub
    };
    // Model-side failure (4xx on the result) is final: drop the persisted request so a retry
    // resubmits. Network errors keep it so a retry resumes polling.
    fal.wait(&sub, timeout).inspect_err(|e| {
        if e.downcast_ref::<HttpError>().is_some_and(|h| (400..500).contains(&h.0)) {
            let _ = std::fs::remove_file(req_path);
        }
    })
}

/// URL of the GLB in a mesh model's output: known keys first, then any `.glb` string anywhere.
pub fn find_glb_url(v: &Value) -> Option<String> {
    for p in ["/model_mesh/url", "/model_glb/url", "/model_urls/glb/url", "/glb/url", "/mesh/url"] {
        if let Some(s) = v.pointer(p).and_then(Value::as_str) {
            return Some(s.to_string());
        }
    }
    fn walk(v: &Value) -> Option<String> {
        match v {
            Value::String(s) if s.split(['?', '#']).next().is_some_and(|b| b.to_ascii_lowercase().ends_with(".glb")) => Some(s.clone()),
            Value::Object(m) => m.values().find_map(walk),
            Value::Array(a) => a.iter().find_map(walk),
            _ => None,
        }
    }
    walk(v)
}

fn first_image_url(out: &Value) -> Result<String> {
    out.pointer("/images/0/url")
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("model output has no images[0].url: {}", out.to_string().chars().take(500).collect::<String>()))
}

fn existing(dir: &Path, stem: &str) -> Option<PathBuf> {
    ["jpg", "png"].iter().map(|e| dir.join(format!("{stem}.{e}"))).find(|p| p.is_file())
}

/// Download `url` into `dir/<stem>.<jpg|png>`, naming by the actual content.
fn download_image(fal: &Fal, url: &str, dir: &Path, stem: &str) -> Result<PathBuf> {
    let tmp = dir.join(format!("{stem}.download"));
    fal.download(url, &tmp)?;
    let ext = match image::guess_format(&std::fs::read(&tmp)?) {
        Ok(image::ImageFormat::Png) => "png",
        _ => "jpg",
    };
    let dst = dir.join(format!("{stem}.{ext}"));
    std::fs::rename(&tmp, &dst)?;
    Ok(dst)
}

/// Generate one catalog item into `out_dir` (`items/`, `work/`, `catalog.json`).
pub fn generate_item(fal: &Fal, req: &GenRequest, out_dir: &Path, log: impl Fn(&str)) -> Result<CatalogItem> {
    let log: &dyn Fn(&str) = &log;
    let id = match &req.id {
        Some(id) if valid_id(id) => id.clone(),
        Some(id) => bail!("invalid item id {id:?}"),
        None => new_id(req.name.as_deref().or(req.prompt.as_deref()).unwrap_or("item")),
    };
    let work = out_dir.join("work").join(&id);
    std::fs::create_dir_all(&work)?;
    // A resume with only an id picks up the original request.
    let saved = work.join("request.json");
    let req = if req.prompt.is_none() && req.image_b64.is_none() && saved.exists() {
        let mut r: GenRequest = serde_json::from_slice(&std::fs::read(&saved)?)?;
        r.id = Some(id.clone());
        r
    } else {
        write_atomic(&saved, &serde_json::to_vec_pretty(req)?)?;
        req.clone()
    };
    let prompt = req.prompt.as_deref().map(str::trim).filter(|p| !p.is_empty());
    let tag = format!("{}-{id}", req.tag.as_deref().unwrap_or("app-decorator-gen"));
    log(&format!("item {id}"));

    // 1. Source image: uploaded by the caller, or text -> image.
    let url_file = work.join("image_url.txt");
    let image_path = if let Some(p) = existing(&work, "image") {
        p
    } else if let Some(b64) = &req.image_b64 {
        let (bytes, ext, _) = decode_image_b64(b64)?;
        let p = work.join(format!("image.{ext}"));
        std::fs::write(&p, bytes)?;
        p
    } else if let Some(prompt) = prompt {
        log(&format!("text -> image ({IMAGE_ENDPOINT})"));
        let out = fal_call(fal, IMAGE_ENDPOINT, &image_payload(prompt), &work.join("image.request.json"), &tag, USD_IMAGE, Duration::from_secs(300), log)?;
        let url = first_image_url(&out)?;
        std::fs::write(&url_file, &url)?;
        download_image(fal, &url, &work, "image")?
    } else {
        bail!("generate needs a prompt or an image");
    };

    // 2. Image -> 3D.
    let raw = work.join("raw.glb");
    if !raw.is_file() {
        let image_url = match std::fs::read_to_string(&url_file) {
            Ok(u) if !u.trim().is_empty() => u.trim().to_string(),
            _ => {
                log("uploading source image");
                let (ctype, name) = if image_path.extension().is_some_and(|e| e == "png") { ("image/png", "image.png") } else { ("image/jpeg", "image.jpg") };
                let url = fal.upload_bytes(name, ctype, std::fs::read(&image_path)?)?;
                std::fs::write(&url_file, &url)?;
                url
            }
        };
        log(&format!("image -> 3D ({MESH_ENDPOINT})"));
        let out = fal_call(fal, MESH_ENDPOINT, &mesh_payload(&image_url), &work.join("mesh.request.json"), &tag, USD_MESH, Duration::from_secs(1200), log)?;
        std::fs::write(work.join("mesh.output.json"), serde_json::to_vec_pretty(&out)?)?;
        let glb_url = find_glb_url(&out).with_context(|| format!("mesh output has no .glb URL: {}", out.to_string().chars().take(500).collect::<String>()))?;
        log("downloading mesh");
        let tmp = work.join("raw.glb.download");
        fal.download(&glb_url, &tmp)?;
        std::fs::rename(&tmp, &raw)?;
    }

    // 3. Normalize + thumbnail.
    let label = req.name.as_deref().or(prompt).unwrap_or("item");
    let category = req.category.clone().unwrap_or_else(|| guess_category(label).to_string());
    let target = req.dims_m.map_or_else(|| category_target(&category), SizeTarget::Dims);
    log(&format!("normalizing as {category} ({target:?})"));
    let (glb, report) = normalize_glb(&std::fs::read(&raw)?, &NormalizeOpts { target, ..NormalizeOpts::default() })?;
    let items = out_dir.join("items");
    write_atomic(&items.join(format!("{id}.glb")), &glb)?;
    write_atomic(&items.join(format!("{id}.webp")), &thumb_webp(&std::fs::read(&image_path)?, THUMB_PX)?)?;
    log(&format!("dims {:.2} x {:.2} x {:.2} m, {} KB", report.dims_m[0], report.dims_m[1], report.dims_m[2], report.bytes_out / 1024));

    // 4. Catalog.
    let item = CatalogItem {
        id: id.clone(),
        name: req.name.clone().unwrap_or_else(|| prompt.unwrap_or("Uploaded item").chars().take(60).collect()),
        category,
        glb: format!("items/{id}.glb"),
        thumb: format!("items/{id}.webp"),
        dims_m: report.dims_m,
        license: format!("generated (fal {MESH_ENDPOINT}); endpoint tagged for commercial use by fal"),
        source: prompt.map_or_else(|| "uploaded image".to_string(), str::to_string),
        generated: true,
        product: None,
    };
    Catalog::upsert(&out_dir.join("catalog.json"), item.clone())?;
    log("done");
    Ok(item)
}

#[derive(Debug, Clone, Deserialize)]
pub struct RestyleRequest {
    pub image_b64: String,
    /// Accepted for API compatibility; the current restyle model does not take depth.
    #[serde(default)]
    pub depth_b64: Option<String>,
    #[serde(default, alias = "style")]
    pub preset: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    /// Spend-log tag prefix (default "app-decorator-restyle"); the job id is appended.
    #[serde(default)]
    pub tag: Option<String>,
}

/// The prompt a restyle request resolves to: preset text, plus the caller's prompt when given.
pub fn restyle_prompt(req: &RestyleRequest) -> Result<String> {
    let extra = req.prompt.as_deref().map(str::trim).filter(|p| !p.is_empty());
    match (req.preset.as_deref().and_then(preset_prompt), extra) {
        (Some(p), Some(e)) => Ok(format!("{p} {e}")),
        (Some(p), None) => Ok(p),
        (None, Some(e)) => Ok(format!("{e} {KEEP}")),
        (None, None) => bail!("restyle needs a known preset (golden_hour|sunset|night|snow|autumn) or a prompt"),
    }
}

/// Restyle one rendered view; writes the result to `dst` (JPEG). `work` holds the persisted request.
pub fn restyle(fal: &Fal, req: &RestyleRequest, work: &Path, dst: &Path, tag: &str, log: impl Fn(&str)) -> Result<()> {
    let log: &dyn Fn(&str) = &log;
    let prompt = restyle_prompt(req)?;
    std::fs::create_dir_all(work)?;
    let (bytes, ext, mime) = decode_image_b64(&req.image_b64)?;
    log("uploading view");
    let url = fal.upload_bytes(&format!("view.{ext}"), mime, bytes)?;
    log(&format!("restyling ({RESTYLE_ENDPOINT})"));
    let out = fal_call(fal, RESTYLE_ENDPOINT, &restyle_payload(&prompt, &url), &work.join("restyle.request.json"), tag, USD_RESTYLE, Duration::from_secs(600), log)?;
    let result = first_image_url(&out)?;
    if let Some(dir) = dst.parent() {
        std::fs::create_dir_all(dir)?;
    }
    fal.download(&result, dst)?;
    log("done");
    Ok(())
}
