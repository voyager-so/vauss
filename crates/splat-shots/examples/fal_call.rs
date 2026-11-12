//! One generic fal queue call for app prototypes (reshoot / virtual set):
//!   fal_call <endpoint> <payload.json> <out_dir> <tag> <usd_estimate>
//! String values "@/abs/path" in the payload are uploaded to fal storage and replaced by URLs.
//! The submitted request is persisted at out_dir/request.json first, so a rerun resumes instead of
//! paying twice. Every media URL in the output is downloaded into out_dir. The paid call is
//! appended to $SPLAT_SPEND_LOG, else the dev root's fal-spend.jsonl (splat_contracts::dev), else
//! out_dir/fal-spend.jsonl.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use splat_shots::fal::{Fal, Submitted};

fn spend_log(out: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os("SPLAT_SPEND_LOG").filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    match splat_contracts::dev::dev_root() {
        Some(root) => root.join("fal-spend.jsonl"),
        None => out.join("fal-spend.jsonl"),
    }
}

fn upload_refs(fal: &Fal, v: &mut Value) -> Result<()> {
    match v {
        Value::String(s) if s.starts_with("@/") => {
            let url = fal.upload(Path::new(&s[1..]))?;
            eprintln!("uploaded {} -> {url}", &s[1..]);
            *s = url;
        }
        Value::Array(a) => a.iter_mut().try_for_each(|x| upload_refs(fal, x))?,
        Value::Object(o) => o.values_mut().try_for_each(|x| upload_refs(fal, x))?,
        _ => {}
    }
    Ok(())
}

fn media_urls(v: &Value, key: &str, out: &mut Vec<(String, String)>) {
    match v {
        Value::String(s) if s.starts_with("http") && [".mp4", ".png", ".jpg", ".webm", ".webp", ".mov", ".zip"].iter().any(|e| s.contains(e)) => {
            out.push((key.to_string(), s.clone()))
        }
        Value::Array(a) => a.iter().enumerate().for_each(|(i, x)| media_urls(x, &format!("{key}{i}"), out)),
        Value::Object(o) => o.iter().for_each(|(k, x)| media_urls(x, if key.is_empty() { k } else { key }, out)),
        _ => {}
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() == 6, "usage: fal_call <endpoint> <payload.json> <out_dir> <tag> <usd_estimate>");
    let (endpoint, out, tag, usd) = (&a[1], PathBuf::from(&a[3]), &a[4], a[5].parse::<f64>()?);
    std::fs::create_dir_all(&out)?;
    let fal = Fal::from_env()?;
    let req_path = out.join("request.json");
    let t0 = Instant::now();
    let sub: Submitted = if req_path.exists() {
        serde_json::from_value(serde_json::from_str::<Value>(&std::fs::read_to_string(&req_path)?)?["submitted"].clone())?
    } else {
        let mut payload: Value = serde_json::from_str(&std::fs::read_to_string(&a[2])?).context("payload json")?;
        upload_refs(&fal, &mut payload)?;
        let sub = fal.submit(endpoint, &payload)?;
        std::fs::write(&req_path, serde_json::to_vec_pretty(&json!({"submitted": sub, "payload": payload}))?)?;
        let line = json!({"ts": chrono::Local::now().to_rfc3339(), "kind": "job", "tag": tag, "usd": usd, "endpoint": endpoint, "request_id": sub.request_id});
        std::fs::OpenOptions::new().create(true).append(true).open(spend_log(&out))?.write_all(format!("{line}\n").as_bytes())?;
        eprintln!("submitted {endpoint} request {}", sub.request_id);
        sub
    };
    let result = fal.wait(&sub, Duration::from_secs(3600))?;
    std::fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    let mut urls = vec![];
    media_urls(&result, "", &mut urls);
    for (i, (k, u)) in urls.iter().enumerate() {
        let ext = [".mp4", ".png", ".jpg", ".webm", ".webp", ".mov", ".zip"].iter().find(|e| u.contains(*e)).unwrap();
        let dst = out.join(format!("{k}{}{ext}", if i == 0 { String::new() } else { format!("_{i}") }));
        fal.download(u, &dst)?;
        eprintln!("downloaded {}", dst.display());
    }
    println!("{}", json!({"request_id": sub.request_id, "wall_s": t0.elapsed().as_secs_f64(), "files": urls.len()}));
    Ok(())
}
