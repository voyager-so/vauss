//! `vauss view [INPUT...]`: local server for the decorator web app over one or more splat scenes.
//! Inputs are scene folders (scene.json), run folders or bare `.ply`/`.spz` files (see [`crate::scenes`]).
//!
//!   /scenes/<id>/              the web app for one scene, with everything below relative to it:
//!     scene/*                    the scene package (scene.json, the splat, restyle/...)
//!     catalog/catalog.json       builtin catalog + <scene>/generated/catalog.json, item paths `items/<file>`
//!     catalog/items/<file>       <scene>/generated/items, then the builtin catalog's items
//!     api/health, api/generate[/{id}], api/layout[/{id}], api/restyle[/{id}]
//!   /                          one input: the app over the first scene (so /scene/, /catalog/, /api/ work
//!                              as with the old `splat decorate`); none or several: a scene picker
//!   /api/scenes, /api/open, /api/open/upload, /api/quit   picker API (same-origin only)
//!
//! Web app: --web DIR, otherwise a page linking to the scene API.
//! Optional catalog: --catalog DIR; no catalog is bundled.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as UrlPath, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri, header};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_shots::fal::Fal;
use tokio::io::AsyncWriteExt;
use tokio::sync::watch;
use tokio_stream::StreamExt;
use tower::ServiceExt;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};

use crate::generate::{GenRequest, RestyleRequest, generate_item, restyle, restyle_prompt};
use crate::scenes::{self, Found};
use crate::{CatalogItem, valid_id, write_atomic};

#[derive(clap::Args, Debug, Clone, Default)]
pub struct ViewArgs {
    /// Scene folders, run folders, or .ply/.spz files. None: open a scene picker.
    pub inputs: Vec<PathBuf>,
    /// Port to bind; without it, 8787 or the next free port is used.
    #[arg(long)]
    pub port: Option<u16>,
    /// Don't open the browser.
    #[arg(long)]
    pub no_open: bool,
    /// Serve the web app from this dir instead of the embedded build (development).
    #[arg(long, hide = true)]
    pub web: Option<PathBuf>,
    /// Builtin catalog dir (catalog.json + items/) instead of the embedded one (development).
    #[arg(long, hide = true)]
    pub catalog: Option<PathBuf>,
    /// Accepted for compatibility; opening the browser is the default.
    #[arg(long, hide = true)]
    pub open: bool,
}

/// `splat decorate` is a hidden alias of `vauss view`.
pub type DecorateArgs = ViewArgs;

/// What [`serve`] serves. The host is always 127.0.0.1.
#[derive(Debug, Clone, Default)]
pub struct ServeOpts {
    /// Scene dirs, run dirs or .ply/.spz files; empty serves the picker at `/`.
    pub inputs: Vec<PathBuf>,
    /// None: 8787, else the next free port in 8788..=8836, else any. Some(p): exactly p or an error.
    pub port: Option<u16>,
    /// Optional external web app directory.
    pub web: Option<PathBuf>,
    /// Optional external catalog directory.
    pub catalog: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SceneInfo {
    pub id: String,
    /// Absolute URL of the app for this scene.
    pub url: String,
    /// The input (canonical).
    pub path: PathBuf,
}

const PICKER: &str = include_str!("picker.html");

#[derive(Debug, Clone, Serialize)]
struct GenJob {
    status: &'static str,
    /// High-detail upgrade after the fast item: upgrading|ready|preview_only (absent when off).
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<&'static str>,
    item: Option<CatalogItem>,
    error: Option<String>,
    log: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RestyleJob {
    status: &'static str,
    image_url: Option<String>,
    error: Option<String>,
}

enum Web {
    Dir(PathBuf),
    Stub,
}

enum Builtin {
    Dir(PathBuf),
    None,
}

/// The fal client, shared by every scene. The key is resolved once in the background at startup;
/// until then (and when it fails) /api/health reports the AI features off.
#[derive(Default)]
struct Ai {
    fal: Mutex<Option<Arc<Fal>>>,
    available: AtomicBool,
}

impl Ai {
    /// Blocking: may run the configured key command. Retried on use, so a key added later works.
    fn fal(&self) -> Result<Arc<Fal>> {
        let mut slot = lock(&self.fal);
        if let Some(f) = slot.as_ref() {
            return Ok(f.clone());
        }
        let f = Arc::new(Fal::from_env()?);
        *slot = Some(f.clone());
        self.available.store(true, Ordering::Relaxed);
        Ok(f)
    }

    fn health(&self) -> Json<Value> {
        let on = self.available.load(Ordering::Relaxed);
        Json(json!({"generate": on, "restyle": on, "product": on && crate::product::ENABLED}))
    }
}

impl Drop for Ai {
    fn drop(&mut self) {
        // The blocking HTTP client must not be dropped inside the async runtime.
        if let Some(fal) = self.fal.get_mut().unwrap_or_else(PoisonError::into_inner).take() {
            std::thread::spawn(move || drop(fal));
        }
    }
}

/// Web app, builtin catalog and fal client, shared by every scene.
struct Assets {
    web: Web,
    builtin: Builtin,
    ai: Ai,
}

impl Assets {
    fn pick(web: Option<&Path>, catalog: Option<&Path>) -> Self {
        let web = match web {
            Some(d) => Web::Dir(d.to_path_buf()),
            None => Web::Stub,
        };
        let builtin = match catalog {
            Some(d) => Builtin::Dir(d.to_path_buf()),
            None => Builtin::None,
        };
        Self { web, builtin, ai: Ai::default() }
    }

    fn catalog_json(&self) -> Option<Vec<u8>> {
        match &self.builtin {
            Builtin::Dir(d) => std::fs::read(d.join("catalog.json")).ok(),
            _ => None,
        }
    }

    fn catalog_item(&self, file: &str) -> Option<Response> {
        match &self.builtin {
            Builtin::Dir(d) => [d.join("items").join(file), d.join(file)].iter().find(|p| p.is_file()).map(|p| file_response(p)),
            _ => None,
        }
    }
}

/// One scene's state; generated items, layouts and restyles are written under `scene_dir`.
struct AppState {
    scene_dir: PathBuf,
    /// Scene files served from elsewhere, by name (a bare splat's original file).
    aliases: HashMap<String, PathBuf>,
    assets: Arc<Assets>,
    gen_jobs: Mutex<HashMap<String, GenJob>>,
    restyle_jobs: Mutex<HashMap<String, RestyleJob>>,
}

type Shared = Arc<AppState>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl AppState {
    fn generated(&self) -> PathBuf {
        self.scene_dir.join("generated")
    }
}

fn err(code: StatusCode, msg: impl std::fmt::Display) -> Response {
    (code, Json(json!({"error": msg.to_string()}))).into_response()
}

fn mime(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("").to_ascii_lowercase().as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "glb" => "model/gltf-binary",
        "gltf" => "model/gltf+json",
        "webp" => "image/webp",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "wasm" => "application/wasm",
        "woff2" => "font/woff2",
        "txt" | "md" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// Brotli/gzip (by Accept-Encoding) for text the browser parses: the app bundle (3.6 MB of JS, 1.2 MB
/// gzipped; 1.4 s at 25 Mbit/s uncompressed), scene.json, catalog JSON. Splat payloads (.spz, .rad,
/// .ply: octet-stream) stay as they are: already compressed, and the .rad is fetched by byte range.
/// tower-http never compresses range responses or ones that already carry a Content-Encoding.
fn compressible(_: StatusCode, _: axum::http::Version, headers: &HeaderMap, _: &axum::http::Extensions) -> bool {
    let ty = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
    let text = ty.starts_with("text/") || ["javascript", "json", "svg", "wasm", "xml"].iter().any(|t| ty.contains(t));
    let len = headers.get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()?.parse::<u64>().ok());
    text && len.is_none_or(|n| n >= 1024)
}

fn bytes_response(name: &str, bytes: &'static [u8]) -> Response {
    ([(header::CONTENT_TYPE, mime(name))], bytes).into_response()
}

fn file_response(path: &Path) -> Response {
    match std::fs::read(path) {
        Ok(b) => ([(header::CONTENT_TYPE, mime(&path.to_string_lossy()))], b).into_response(),
        Err(_) => err(StatusCode::NOT_FOUND, "not found"),
    }
}

/// Serve through a tower-http file service (ranges, conditional GETs), forcing our MIME type.
async fn serve_via<S>(svc: S, req: Request, name: &str) -> Response
where
    S: tower::Service<Request, Error = std::convert::Infallible>,
    S::Response: IntoResponse,
{
    let Ok(res) = svc.oneshot(req).await;
    let mut res = res.into_response();
    if res.status().is_success() || res.status() == StatusCode::PARTIAL_CONTENT {
        res.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static(mime(name)));
    }
    res
}

/// Builtin + generated catalogs, item paths rewritten to `items/<file>` (served by catalog/items).
fn merged_catalog(state: &AppState) -> Value {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let generated = state.generated().join("catalog.json");
    let sources = [state.assets.catalog_json().map(|b| ("builtin catalog".to_string(), b)), std::fs::read(&generated).ok().map(|b| (generated.display().to_string(), b))];
    for (name, bytes) in sources.into_iter().flatten() {
        let Ok(cat) = serde_json::from_slice::<Value>(&bytes) else {
            eprintln!("warning: unparseable catalog {name}");
            continue;
        };
        for mut item in cat["items"].as_array().cloned().unwrap_or_default() {
            let Some(id) = item["id"].as_str().map(str::to_string) else { continue };
            if !seen.insert(id) {
                continue;
            }
            for k in ["glb", "thumb"] {
                if let Some(file) = item[k].as_str().and_then(|p| p.rsplit('/').next()).map(str::to_string) {
                    item[k] = json!(format!("items/{file}"));
                }
            }
            items.push(item);
        }
    }
    json!({"schema": crate::CATALOG_SCHEMA, "items": items})
}

async fn catalog_json(State(s): State<Shared>) -> Response {
    Json(merged_catalog(&s)).into_response()
}

async fn catalog_item(State(s): State<Shared>, UrlPath(file): UrlPath<String>) -> Response {
    if file.is_empty() || file.starts_with('.') || file.contains(['/', '\\']) || file.contains("..") {
        return err(StatusCode::BAD_REQUEST, "bad file name");
    }
    let generated = s.generated().join("items").join(&file);
    if generated.is_file() {
        return file_response(&generated);
    }
    s.assets.catalog_item(&file).unwrap_or_else(|| err(StatusCode::NOT_FOUND, "not found"))
}

/// Files of the scene package; aliased names come from their original path.
async fn scene_file(State(s): State<Shared>, UrlPath(path): UrlPath<String>, mut req: Request) -> Response {
    if path.split(['/', '\\']).any(|c| c == "..") {
        return err(StatusCode::BAD_REQUEST, "bad path");
    }
    if let Some(orig) = s.aliases.get(&path) {
        return serve_via(ServeFile::new(orig), req, &path).await;
    }
    let raw = req.uri().path().strip_prefix("/scene").unwrap_or("/").to_string();
    match raw.parse::<Uri>() {
        Ok(uri) => *req.uri_mut() = uri,
        Err(_) => return err(StatusCode::BAD_REQUEST, "bad path"),
    }
    serve_via(ServeDir::new(&s.scene_dir), req, &path).await
}

async fn health(State(s): State<Shared>) -> Json<Value> {
    s.assets.ai.health()
}

fn short_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..12].to_string()
}

async fn post_generate(State(s): State<Shared>, Json(req): Json<GenRequest>) -> Response {
    let has_prompt = req.prompt.as_deref().is_some_and(|p| !p.trim().is_empty());
    if !has_prompt && req.image_b64.is_none() {
        return err(StatusCode::BAD_REQUEST, "need prompt or image_b64");
    }
    let job_id = short_id();
    lock(&s.gen_jobs).insert(job_id.clone(), GenJob { status: "running", detail: None, item: None, error: None, log: Vec::new() });
    let (state, jid) = (s.clone(), job_id.clone());
    tokio::task::spawn_blocking(move || {
        let push = |m: &str| {
            println!("[generate {jid}] {m}");
            if let Some(j) = lock(&state.gen_jobs).get_mut(&jid) {
                j.log.push(m.to_string());
            }
        };
        let res = state.assets.ai.fal().and_then(|fal| generate_item(&fal, &req, &state.generated(), push));
        if let Err(e) = &res {
            push(&format!("failed: {e:#}"));
        }
        let fast = res.as_ref().ok().cloned();
        if let Some(j) = lock(&state.gen_jobs).get_mut(&jid) {
            match res {
                Ok(item) => (j.status, j.item) = ("done", Some(item)),
                Err(e) => (j.status, j.error) = ("failed", Some(format!("{e:#}"))),
            }
            if fast.is_some() && crate::detail::ENABLED {
                j.detail = Some("upgrading");
            }
        }
        // The fast item is usable now; the high-detail version replaces it when ready.
        if let (Some(item), true) = (fast, crate::detail::ENABLED) {
            let up = state.assets.ai.fal().and_then(|fal| crate::detail::upgrade_item(&fal, &item, &state.generated(), &push));
            if let Err(e) = &up {
                push(&format!("detail upgrade failed: {e:#}"));
            }
            if let Some(j) = lock(&state.gen_jobs).get_mut(&jid) {
                match up {
                    Ok(item) => (j.detail, j.item) = (Some("ready"), Some(item)),
                    Err(_) => j.detail = Some("preview_only"),
                }
            }
        }
    });
    Json(json!({"job_id": job_id})).into_response()
}

async fn post_product(State(s): State<Shared>, Json(req): Json<crate::product::ProductRequest>) -> Response {
    if !crate::product::ENABLED {
        return err(StatusCode::NOT_IMPLEMENTED, "product import is not available yet");
    }
    // Request validation belongs to product::ingest (its error is the job's error).
    let job_id = short_id();
    lock(&s.gen_jobs).insert(job_id.clone(), GenJob { status: "running", detail: None, item: None, error: None, log: Vec::new() });
    let (state, jid) = (s.clone(), job_id.clone());
    tokio::task::spawn_blocking(move || {
        let push = |m: &str| {
            println!("[product {jid}] {m}");
            if let Some(j) = lock(&state.gen_jobs).get_mut(&jid) {
                j.log.push(m.to_string());
            }
        };
        let res = state.assets.ai.fal().and_then(|fal| crate::product::ingest(&fal, &req, &state.generated(), &push));
        if let Err(e) = &res {
            push(&format!("failed: {e:#}"));
        }
        if let Some(j) = lock(&state.gen_jobs).get_mut(&jid) {
            match res {
                Ok(item) => (j.status, j.item) = ("done", Some(item)),
                Err(e) => (j.status, j.error) = ("failed", Some(format!("{e:#}"))),
            }
        }
    });
    Json(json!({"job_id": job_id})).into_response()
}

async fn get_generate(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> Response {
    lock(&s.gen_jobs).get(&id).cloned().map_or_else(|| err(StatusCode::NOT_FOUND, "no such job"), |j| Json(j).into_response())
}

async fn post_layout(State(s): State<Shared>, Json(body): Json<Value>) -> Response {
    let id = short_id();
    let path = s.scene_dir.join("layouts").join(format!("{id}.json"));
    match serde_json::to_vec_pretty(&body).map_err(anyhow::Error::from).and_then(|b| write_atomic(&path, &b)) {
        Ok(()) => Json(json!({"id": id})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn get_layout(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> Response {
    if !valid_id(&id) {
        return err(StatusCode::BAD_REQUEST, "bad id");
    }
    file_response(&s.scene_dir.join("layouts").join(format!("{id}.json")))
}

async fn post_restyle(State(s): State<Shared>, Json(req): Json<RestyleRequest>) -> Response {
    if let Err(e) = restyle_prompt(&req) {
        return err(StatusCode::BAD_REQUEST, e);
    }
    let job_id = short_id();
    lock(&s.restyle_jobs).insert(job_id.clone(), RestyleJob { status: "running", image_url: None, error: None });
    let (state, jid) = (s.clone(), job_id.clone());
    tokio::task::spawn_blocking(move || {
        let dir = state.scene_dir.join("restyle");
        let dst = dir.join(format!("{jid}.jpg"));
        let log = |m: &str| println!("[restyle {jid}] {m}");
        let tag = format!("{}-{jid}", req.tag.as_deref().unwrap_or("app-decorator-restyle"));
        let res = state.assets.ai.fal().and_then(|fal| restyle(&fal, &req, &dir.join("work").join(&jid), &dst, &tag, log));
        if let Err(e) = &res {
            log(&format!("failed: {e:#}"));
        }
        if let Some(j) = lock(&state.restyle_jobs).get_mut(&jid) {
            match res {
                // Relative to the app page, which may live under /scenes/<id>/.
                Ok(()) => (j.status, j.image_url) = ("done", Some(format!("scene/restyle/{jid}.jpg"))),
                Err(e) => (j.status, j.error) = ("failed", Some(format!("{e:#}"))),
            }
        }
    });
    Json(json!({"job_id": job_id})).into_response()
}

async fn get_restyle(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> Response {
    lock(&s.restyle_jobs).get(&id).cloned().map_or_else(|| err(StatusCode::NOT_FOUND, "no such job"), |j| Json(j).into_response())
}

fn stub_page() -> Response {
    Html(
        "<!doctype html><meta charset=utf-8><title>vauss view</title>\
         <body style=\"font-family:system-ui;margin:2rem\"><h1>vauss view</h1>\
         <p>Pass --web DIR to serve an external web app, or use vauss export-viewer to create a standalone viewer.</p>\
         <ul><li><a href=scene/scene.json>scene.json</a></li><li><a href=catalog/catalog.json>catalog.json</a></li>\
         <li><a href=api/health>api/health</a></li></ul></body>",
    )
    .into_response()
}

async fn web_response(assets: &Assets, req: Request) -> Response {
    match &assets.web {
        Web::Dir(d) => {
            let name = req.uri().path().to_string();
            let name = if name.ends_with('/') { "index.html".to_string() } else { name };
            serve_via(ServeDir::new(d).fallback(ServeFile::new(d.join("index.html"))), req, &name).await
        }
        Web::Stub => stub_page(),
    }
}

async fn web_fallback(State(s): State<Shared>, req: Request) -> Response {
    web_response(&s.assets, req).await
}

fn scene_router(state: Shared) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/generate", post(post_generate))
        .route("/api/generate/{id}", get(get_generate))
        .route("/api/product", post(post_product))
        .route("/api/product/{id}", get(get_generate))
        .route("/api/layout", post(post_layout))
        .route("/api/layout/{id}", get(get_layout))
        .route("/api/restyle", post(post_restyle))
        .route("/api/restyle/{id}", get(get_restyle))
        .route("/catalog/catalog.json", get(catalog_json))
        .route("/catalog/items/{file}", get(catalog_item))
        .route("/scene/{*path}", get(scene_file))
        .fallback(web_fallback)
        .layer(DefaultBodyLimit::max(64 << 20))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Stops a server (graceful, then forced after a few seconds).
#[derive(Clone)]
pub struct Stopper(Arc<watch::Sender<bool>>);

impl Stopper {
    fn new() -> Self {
        Self(Arc::new(watch::channel(false).0))
    }

    pub fn stop(&self) {
        self.0.send_replace(true);
    }

    pub async fn stopped(&self) {
        let mut rx = self.0.subscribe();
        let _ = rx.wait_for(|v| *v).await;
    }
}

struct Entry {
    info: SceneInfo,
    router: Router,
}

struct Global {
    assets: Arc<Assets>,
    /// Startup inputs first; `/scene/` etc. at the root alias the first one.
    scenes: RwLock<Vec<Entry>>,
    /// `/` is the picker (zero or several inputs).
    picker: bool,
    stop: Stopper,
    base: String,
}

impl Global {
    /// Blocking (resolves the input, may parse a PLY). Re-opening a path returns its scene.
    fn register(&self, input: &Path) -> Result<SceneInfo> {
        let r = scenes::resolve(input)?;
        let mut list = self.scenes.write().unwrap_or_else(PoisonError::into_inner);
        if let Some(e) = list.iter().find(|e| e.info.path == r.input) {
            return Ok(e.info.clone());
        }
        let base = crate::slug(&r.stem);
        let mut id = base.clone();
        for n in 2.. {
            if !list.iter().any(|e| e.info.id == id) {
                break;
            }
            id = format!("{base}-{n}");
        }
        let state = Arc::new(AppState {
            scene_dir: r.scene_dir,
            aliases: r.aliases,
            assets: self.assets.clone(),
            gen_jobs: Mutex::new(HashMap::new()),
            restyle_jobs: Mutex::new(HashMap::new()),
        });
        let info = SceneInfo { url: format!("{}scenes/{id}/", self.base), id, path: r.input };
        list.push(Entry { info: info.clone(), router: scene_router(state) });
        Ok(info)
    }

    fn router(&self, id: Option<&str>) -> Option<Router> {
        let list = self.scenes.read().unwrap_or_else(PoisonError::into_inner);
        match id {
            Some(id) => list.iter().find(|e| e.info.id == id),
            None => list.first(),
        }
        .map(|e| e.router.clone())
    }

    fn infos(&self) -> Vec<SceneInfo> {
        self.scenes.read().unwrap_or_else(PoisonError::into_inner).iter().map(|e| e.info.clone()).collect()
    }
}

type G = Arc<Global>;

/// Everything not handled by the picker API: per-scene routers, the picker page, the web app.
async fn dispatch(State(g): State<G>, mut req: Request) -> Response {
    let path = req.uri().path().to_string();
    let (router, rest) = if let Some(rest) = path.strip_prefix("/scenes/") {
        let Some((id, tail)) = rest.split_once('/') else {
            return Redirect::temporary(&format!("/scenes/{rest}/")).into_response();
        };
        match g.router(Some(id)) {
            Some(r) => (r, format!("/{tail}")),
            None => return err(StatusCode::NOT_FOUND, format!("no scene {id:?}")),
        }
    } else if g.picker && (path == "/" || path == "/index.html") {
        return Html(PICKER).into_response();
    } else if let Some(r) = g.router(None) {
        (r, path)
    } else {
        return web_response(&g.assets, req).await;
    };
    let pq = match req.uri().query() {
        Some(q) => format!("{rest}?{q}"),
        None => rest,
    };
    match pq.parse::<Uri>() {
        Ok(uri) => *req.uri_mut() = uri,
        Err(_) => return err(StatusCode::BAD_REQUEST, "bad path"),
    }
    let Ok(res) = router.oneshot(req).await;
    res
}

/// Browsers send Origin on cross-site POSTs; only this server's own pages may use the picker API.
fn same_origin(headers: &HeaderMap) -> bool {
    let Some(origin) = headers.get(header::ORIGIN).and_then(|o| o.to_str().ok()) else { return true };
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
    origin == format!("http://{host}")
}

async fn api_scenes(State(g): State<G>) -> Response {
    let mut roots: Vec<PathBuf> = std::env::current_dir().into_iter().collect();
    roots.extend(crate::data_dir().map(|d| d.join("splat")));
    let recent: Vec<Found> = tokio::task::spawn_blocking(move || scenes::scan(&roots)).await.unwrap_or_default();
    Json(json!({"scenes": g.infos(), "recent": recent})).into_response()
}

#[derive(Deserialize)]
struct OpenReq {
    path: String,
}

async fn register(g: G, path: PathBuf) -> Response {
    match tokio::task::spawn_blocking(move || g.register(&path)).await {
        Ok(Ok(info)) => Json(json!({"id": info.id, "url": info.url})).into_response(),
        Ok(Err(e)) => err(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn api_open(State(g): State<G>, headers: HeaderMap, Json(req): Json<OpenReq>) -> Response {
    if !same_origin(&headers) {
        return err(StatusCode::FORBIDDEN, "cross-origin request");
    }
    // Accept pasted paths: surrounding quotes (Windows "Copy as path") and a leading ~.
    let p = req.path.trim().trim_matches('"').trim();
    let path = match (p.strip_prefix("~/").or_else(|| p.strip_prefix("~\\")), std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(p),
    };
    register(g, path).await
}

#[derive(Deserialize)]
struct UploadQuery {
    name: String,
}

/// Raw request body -> `<cache>/uploads/<random>/<name>`, then opened like any other file.
async fn api_upload(State(g): State<G>, headers: HeaderMap, Query(q): Query<UploadQuery>, body: Body) -> Response {
    if !same_origin(&headers) {
        return err(StatusCode::FORBIDDEN, "cross-origin request");
    }
    let name: String = q.name.rsplit(['/', '\\']).next().unwrap_or("").chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    if name.starts_with('.') || !scenes::is_splat_file(Path::new(&name)) {
        return err(StatusCode::BAD_REQUEST, "upload a .ply or .spz file");
    }
    let path = scenes::cache_root().join("uploads").join(short_id()).join(&name);
    let res: Result<()> = async {
        tokio::fs::create_dir_all(path.parent().context("upload dir")?).await?;
        let mut f = tokio::fs::File::create(&path).await?;
        let mut stream = body.into_data_stream();
        while let Some(chunk) = stream.next().await {
            f.write_all(&chunk?).await?;
        }
        f.flush().await?;
        Ok(())
    }
    .await;
    if let Err(e) = res {
        let _ = tokio::fs::remove_file(&path).await;
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("upload: {e:#}"));
    }
    register(g, path).await
}

async fn api_quit(State(g): State<G>, headers: HeaderMap) -> Response {
    if !same_origin(&headers) {
        return err(StatusCode::FORBIDDEN, "cross-origin request");
    }
    g.stop.stop();
    Json(json!({"ok": true})).into_response()
}

async fn bind(port: Option<u16>) -> Result<tokio::net::TcpListener> {
    let ip = Ipv4Addr::LOCALHOST;
    if let Some(p) = port {
        return tokio::net::TcpListener::bind((ip, p)).await.with_context(|| format!("bind {ip}:{p} (port given with --port)"));
    }
    for p in (8787..=8836).chain([0]) {
        if let Ok(l) = tokio::net::TcpListener::bind((ip, p)).await {
            return Ok(l);
        }
    }
    bail!("no free local port")
}

/// A running server (see [`serve`]).
pub struct Server {
    pub addr: SocketAddr,
    /// `http://127.0.0.1:PORT/`
    pub url: String,
    /// Scenes from [`ServeOpts::inputs`], in order.
    pub scenes: Vec<SceneInfo>,
    stop: Stopper,
    task: tokio::task::JoinHandle<Result<()>>,
}

impl Server {
    /// Ask the server to stop; [`Server::wait`] returns once it has.
    pub fn shutdown(&self) {
        self.stop.stop();
    }

    /// A cloneable handle that stops this server.
    pub fn stopper(&self) -> Stopper {
        self.stop.clone()
    }

    /// Until the server stops (shutdown, /api/quit, or an error).
    pub async fn wait(self) -> Result<()> {
        self.task.await.context("server task")?
    }
}

/// Bind 127.0.0.1, register the inputs, start serving on the current tokio runtime.
pub async fn serve(opts: ServeOpts) -> Result<Server> {
    let listener = bind(opts.port).await?;
    let addr = listener.local_addr()?;
    let url = format!("http://{addr}/");
    let stop = Stopper::new();
    let assets = Arc::new(Assets::pick(opts.web.as_deref(), opts.catalog.as_deref()));
    let a = assets.clone();
    tokio::task::spawn_blocking(move || {
        let _ = a.ai.fal();
    });
    let g: G = Arc::new(Global {
        assets,
        scenes: RwLock::new(Vec::new()),
        picker: opts.inputs.len() != 1,
        stop: stop.clone(),
        base: url.clone(),
    });
    let mut scenes = Vec::new();
    for input in opts.inputs {
        let g = g.clone();
        scenes.push(tokio::task::spawn_blocking(move || g.register(&input)).await??);
    }
    let app = Router::new()
        .route("/api/health", get(|State(g): State<G>| async move { g.assets.ai.health() }))
        .route("/api/scenes", get(api_scenes))
        .route("/api/open", post(api_open))
        .route("/api/open/upload", post(api_upload).layer(DefaultBodyLimit::disable()))
        .route("/api/quit", post(api_quit))
        .fallback(dispatch)
        .layer(CompressionLayer::new().compress_when(compressible))
        .with_state(g);
    let s = stop.clone();
    let task = tokio::spawn(async move {
        let graceful = axum::serve(listener, app).with_graceful_shutdown({
            let s = s.clone();
            async move { s.stopped().await }
        });
        // Open downloads would hold a graceful shutdown forever; give them a few seconds.
        tokio::select! {
            r = async { graceful.await } => r.context("server"),
            () = async { s.stopped().await; tokio::time::sleep(Duration::from_secs(3)).await } => Ok(()),
        }
    });
    Ok(Server { addr, url, scenes, stop, task })
}

/// A server on its own thread and runtime; stops on [`ServerHandle::shutdown`] or drop.
pub struct ServerHandle {
    addr: SocketAddr,
    url: String,
    scenes: Vec<SceneInfo>,
    stop: Stopper,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
}

impl ServerHandle {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn scenes(&self) -> &[SceneInfo] {
        &self.scenes
    }

    pub fn shutdown(&self) {
        self.stop.stop();
    }

    /// Block until the server stops (shutdown, /api/quit, or an error).
    pub fn wait(mut self) -> Result<()> {
        self.thread.take().map_or(Ok(()), |t| t.join().map_err(|_| anyhow::anyhow!("server thread panicked"))?)
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.stop.stop();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

type Started = (SocketAddr, String, Vec<SceneInfo>, Stopper);

/// Blocking: start [`serve`] on a new thread with its own runtime; returns once listening.
pub fn spawn(opts: ServeOpts) -> Result<ServerHandle> {
    let (tx, rx) = std::sync::mpsc::channel::<Result<Started>>();
    let thread = std::thread::Builder::new().name("splat-view".into()).spawn(move || -> Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        let res = rt.block_on(async {
            match serve(opts).await {
                Ok(server) => {
                    let _ = tx.send(Ok((server.addr, server.url.clone(), server.scenes.clone(), server.stopper())));
                    server.wait().await
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    Ok(())
                }
            }
        });
        rt.shutdown_timeout(Duration::from_secs(1));
        res
    })?;
    match rx.recv() {
        Ok(Ok((addr, url, scenes, stop))) => Ok(ServerHandle { addr, url, scenes, stop, thread: Some(thread) }),
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(e)
        }
        Err(_) => bail!("server thread exited before listening"),
    }
}

// ---------- CLI ----------

/// How `vauss view` reports itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    Text,
    /// First stdout line is one JSON object.
    Json,
    /// Friendly banner for a double-clicked exe (Windows); waits for Enter on error.
    Banner,
}

pub fn open_browser(url: &str) {
    let mut cmd = if cfg!(windows) {
        // No console window flash, unlike `cmd /C start`.
        let mut c = std::process::Command::new("rundll32");
        c.args(["url.dll,FileProtocolHandler", url]);
        c
    } else {
        let mut c = std::process::Command::new(if cfg!(target_os = "macos") { "open" } else { "xdg-open" });
        c.arg(url);
        c
    };
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    match cmd.spawn() {
        Ok(mut child) => {
            std::thread::spawn(move || child.wait());
        }
        Err(e) => eprintln!("could not open a browser ({e}); open {url} yourself"),
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let term = async {
            match signal(SignalKind::terminate()) {
                Ok(mut s) => {
                    s.recv().await;
                }
                Err(_) => std::future::pending::<()>().await,
            }
        };
        tokio::select! { () = ctrl_c => {}, () = term => {} }
    }
    #[cfg(not(unix))]
    ctrl_c.await;
}

fn process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let Ok(pid) = i32::try_from(pid) else { return false };
        // SAFETY: signal 0 only checks that the process exists.
        unsafe { libc::kill(pid, 0) == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM) }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
        const STILL_ACTIVE: u32 = 259;
        // SAFETY: the handle is checked and closed; GetExitCodeProcess writes one u32.
        unsafe {
            let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if h.is_null() {
                return false;
            }
            let mut code = 0u32;
            let ok = GetExitCodeProcess(h, &mut code) != 0;
            CloseHandle(h);
            ok && code == STILL_ACTIVE
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        true
    }
}

/// Resolves when the process in $VOYAGER_OWNER_PID exits (polled every 3 s); never when unset.
async fn owner_gone() {
    let Some(pid) = std::env::var("VOYAGER_OWNER_PID").ok().and_then(|p| p.trim().parse::<u32>().ok()) else {
        return std::future::pending().await;
    };
    while process_alive(pid) {
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
    eprintln!("vauss view: owner process {pid} exited; stopping");
}

/// `vauss view` / `splat decorate`.
pub fn run(args: ViewArgs, json: bool) -> Result<()> {
    run_with(args, if json { Output::Json } else { Output::Text })
}

pub fn run_with(args: ViewArgs, out: Output) -> Result<()> {
    let opts = ServeOpts { inputs: args.inputs, port: args.port, web: args.web, catalog: args.catalog };
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let res = rt.block_on(async {
        let server = serve(opts).await?;
        let url = &server.url;
        match out {
            Output::Json => {
                let line = json!({"url": url, "port": server.addr.port(), "pid": std::process::id(), "scenes": server.scenes});
                let mut stdout = std::io::stdout().lock();
                writeln!(stdout, "{line}")?;
                stdout.flush()?;
            }
            Output::Text => {
                println!("vauss view: {url}");
                for s in &server.scenes {
                    println!("  {}  {}  ({})", s.id, s.url, s.path.display());
                }
                if server.scenes.len() != 1 {
                    println!("  open {url} to pick a scene");
                }
                println!("  Ctrl-C to stop");
            }
            Output::Banner => {
                println!();
                println!("  splat viewer is running at {url}");
                println!("  Your browser should open by itself; if not, paste that address into it.");
                println!("  Close this window to stop.");
            }
        }
        if !args.no_open {
            open_browser(url);
        }
        let stop = server.stopper();
        tokio::spawn(async move {
            tokio::select! { () = shutdown_signal() => {}, () = owner_gone() => {} }
            stop.stop();
        });
        server.wait().await
    });
    rt.shutdown_timeout(Duration::from_secs(1));
    if out == Output::Banner {
        if let Err(e) = &res {
            println!("\n  splat viewer stopped: {e:#}\n  Press Enter to close.");
            let _ = std::io::stdin().read_line(&mut String::new());
        }
    }
    res
}
