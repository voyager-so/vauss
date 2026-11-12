//! Minimal fal.ai client: storage upload, queue submit, poll, fetch. Blocking.
//!
//! The key comes from FAL_KEY, else the configured key command ([`splat_contracts::dev::fal_key`]);
//! it is held in memory only and never logged or written.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const QUEUE: &str = "https://queue.fal.run";
const REST: &str = "https://rest.fal.ai";

/// HTTP status of a failed fal call; 4xx from a result fetch means the request itself failed.
#[derive(Debug)]
pub struct HttpError(pub u16);

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HTTP {}", self.0)
    }
}

impl std::error::Error for HttpError {}

pub struct Fal {
    key: String,
    http: reqwest::blocking::Client,
}

/// A submitted request; persisted before polling so an interrupted worker resumes instead of paying twice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Submitted {
    pub endpoint: String,
    pub request_id: String,
    pub status_url: String,
    pub response_url: String,
}

fn key_from_env() -> Result<String> {
    splat_contracts::dev::fal_key().map_err(anyhow::Error::msg)
}

impl Fal {
    pub fn from_env() -> Result<Self> {
        let http = reqwest::blocking::Client::builder().timeout(Duration::from_secs(300)).build()?;
        Ok(Self { key: key_from_env()?, http })
    }

    fn auth(&self) -> String {
        format!("Key {}", self.key)
    }

    fn json(&self, req: reqwest::blocking::RequestBuilder) -> Result<Value> {
        let resp = req.header("Authorization", self.auth()).header("Accept", "application/json").send()?;
        let (status, url) = (resp.status(), resp.url().path().to_string());
        let text = resp.text()?;
        if !status.is_success() {
            let body = text.chars().take(2000).collect::<String>();
            return Err(anyhow::Error::new(HttpError(status.as_u16())).context(format!("fal {url} -> {status}: {body}")));
        }
        Ok(if text.trim().is_empty() { Value::Null } else { serde_json::from_str(&text)? })
    }

    /// Upload a local file to fal storage; returns a URL the models can fetch.
    pub fn upload(&self, path: &Path) -> Result<String> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let ctype = match ext.as_str() {
            "mp4" => "video/mp4",
            "webm" => "video/webm",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            _ => bail!("upload: unsupported file type {}", path.display()),
        };
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("upload");
        self.upload_bytes(name, ctype, std::fs::read(path)?)
    }

    /// Upload in-memory bytes to fal storage as `name` with content type `ctype`.
    pub fn upload_bytes(&self, name: &str, ctype: &str, bytes: Vec<u8>) -> Result<String> {
        let init = self.json(
            self.http
                .post(format!("{REST}/storage/upload/initiate?storage_type=gcs"))
                .json(&json!({"file_name": name, "content_type": ctype})),
        )?;
        let (put, url) = (init["upload_url"].as_str().context("upload_url")?, init["file_url"].as_str().context("file_url")?);
        let resp = self.http.put(put).header("Content-Type", ctype).body(bytes).send()?;
        if !resp.status().is_success() {
            bail!("fal storage PUT -> {}", resp.status());
        }
        Ok(url.to_string())
    }

    pub fn submit(&self, endpoint: &str, payload: &Value) -> Result<Submitted> {
        let v = self.json(self.http.post(format!("{QUEUE}/{endpoint}")).json(payload))?;
        let s = |k: &str| v[k].as_str().map(str::to_string).with_context(|| format!("submit response lacks {k}"));
        Ok(Submitted {
            endpoint: endpoint.to_string(),
            request_id: s("request_id")?,
            status_url: s("status_url")?,
            response_url: s("response_url")?,
        })
    }

    /// Poll until COMPLETED, then fetch the output.
    pub fn wait(&self, sub: &Submitted, timeout: Duration) -> Result<Value> {
        let t0 = Instant::now();
        loop {
            let st = self.json(self.http.get(&sub.status_url))?;
            if st["status"] == "COMPLETED" {
                return self.json(self.http.get(&sub.response_url));
            }
            if t0.elapsed() > timeout {
                bail!("{} request {} not done after {:?}", sub.endpoint, sub.request_id, timeout);
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    }

    pub fn download(&self, url: &str, dst: &Path) -> Result<()> {
        let resp = self.http.get(url).timeout(Duration::from_secs(600)).send()?.error_for_status()?;
        std::fs::write(dst, resp.bytes()?)?;
        Ok(())
    }
}
