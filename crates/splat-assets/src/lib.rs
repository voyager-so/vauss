//! Decorator assets: GLB normalization, a fal-backed prompt -> image -> 3D -> catalog item pipeline,
//! and the `vauss view` local server (the decorator web app over one or more splat scenes).
//!
//! Shared catalog format (`catalog.json`):
//!   {"schema":1,"items":[{"id","name","category","glb":"items/<id>.glb","thumb":"items/<id>.webp",
//!     "dims_m":[w,h,d],"license","source","generated":bool}]}
//! Every item GLB is Y-up, in meters, with its origin at the bottom-center of its bounds.

pub mod cli;
pub mod detail;
pub mod generate;
pub mod glb;
pub mod product;
pub mod scenes;
pub mod server;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub use glb::{NormalizeOpts, NormalizeReport, SizeTarget, Up, glb_dims, normalize_glb, thumb_webp};
pub use server::{DecorateArgs, ServeOpts, Server, ServerHandle, ViewArgs, serve, spawn};

pub const CATALOG_SCHEMA: u32 = 1;
/// Team ledger of paid fal calls, used when its directory exists (see [`spend_log_path`]).

/// Per-user data dir: $XDG_DATA_HOME or ~/.local/share (Linux), ~/Library/Application Support
/// (macOS), %LOCALAPPDATA% (Windows).
pub fn data_dir() -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if cfg!(windows) {
        env("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Application Support"))
    } else {
        env("XDG_DATA_HOME").or_else(|| env("HOME").map(|h| h.join(".local/share")))
    }
}

/// Ledger of paid fal calls: $SPLAT_SPEND_LOG, else the dev root's team ledger (`<dev root>/fal-spend.jsonl`,
/// see [`splat_contracts::dev`]) when the dev root exists, else `<data_dir>/splat/fal-spend.jsonl` (parent created).
pub fn spend_log_path() -> PathBuf {
    if let Some(p) = std::env::var_os("SPLAT_SPEND_LOG").filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    if let Some(root) = splat_contracts::dev::dev_root().filter(|r| r.is_dir()) {
        return root.join("fal-spend.jsonl");
    }
    let dir = data_dir().unwrap_or_else(std::env::temp_dir).join("splat");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("fal-spend.jsonl")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogItem {
    pub id: String,
    pub name: String,
    pub category: String,
    pub glb: String,
    pub thumb: String,
    pub dims_m: [f32; 3],
    pub license: String,
    pub source: String,
    #[serde(default)]
    pub generated: bool,
    /// Real-product provenance (product import): {"url","approximate":true,"note"}.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub schema: u32,
    pub items: Vec<CatalogItem>,
}

impl Default for Catalog {
    fn default() -> Self {
        Self { schema: CATALOG_SCHEMA, items: Vec::new() }
    }
}

/// Serializes read-modify-write of catalog files within this process (server jobs run concurrently).
static CATALOG_LOCK: Mutex<()> = Mutex::new(());

impl Catalog {
    /// Missing file reads as an empty catalog.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        serde_json::from_slice(&std::fs::read(path)?).with_context(|| format!("parse {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        write_atomic(path, &serde_json::to_vec_pretty(self)?)
    }

    /// Append (or replace by id) one item in the catalog at `path`.
    pub fn upsert(path: &Path, item: CatalogItem) -> Result<()> {
        let _guard = CATALOG_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut cat = Self::load(path)?;
        cat.items.retain(|i| i.id != item.id);
        cat.items.push(item);
        cat.save(path)
    }
}

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).with_context(|| format!("write {}", path.display()))
}

/// Real-world default size per category, with the keywords that select it.
pub struct Category {
    pub name: &'static str,
    pub target: SizeTarget,
    pub keywords: &'static [&'static str],
}

pub const DEFAULT_CATEGORY: &str = "object";

pub const CATEGORIES: &[Category] = &[
    Category { name: "sofa", target: SizeTarget::MaxDim(2.1), keywords: &["sofa", "couch", "sectional", "loveseat", "settee"] },
    Category { name: "armchair", target: SizeTarget::Height(0.9), keywords: &["armchair", "arm chair", "recliner", "lounge chair", "accent chair", "club chair", "wingback"] },
    Category { name: "dining_chair", target: SizeTarget::Height(0.9), keywords: &["dining chair", "kitchen chair"] },
    Category { name: "chair", target: SizeTarget::Height(0.85), keywords: &["chair", "office chair"] },
    Category { name: "stool", target: SizeTarget::Height(0.65), keywords: &["stool", "ottoman", "pouf"] },
    Category { name: "coffee_table", target: SizeTarget::Height(0.45), keywords: &["coffee table", "cocktail table"] },
    Category { name: "side_table", target: SizeTarget::Height(0.55), keywords: &["side table", "end table", "nightstand", "bedside table", "night stand"] },
    Category { name: "dining_table", target: SizeTarget::Height(0.76), keywords: &["dining table", "kitchen table", "table"] },
    Category { name: "desk", target: SizeTarget::Height(0.75), keywords: &["desk", "writing desk"] },
    Category { name: "floor_lamp", target: SizeTarget::Height(1.6), keywords: &["floor lamp", "standing lamp", "torchiere", "arc lamp"] },
    Category { name: "table_lamp", target: SizeTarget::Height(0.5), keywords: &["table lamp", "desk lamp", "lamp"] },
    Category { name: "small_plant", target: SizeTarget::Height(0.5), keywords: &["small plant", "succulent", "cactus", "potted herb", "bonsai"] },
    Category { name: "plant", target: SizeTarget::Height(1.4), keywords: &["plant", "tree", "fern", "monstera", "ficus", "palm"] },
    Category { name: "bookshelf", target: SizeTarget::Height(1.9), keywords: &["bookshelf", "bookcase", "shelving", "shelf", "book shelf"] },
    Category { name: "tv_console", target: SizeTarget::Height(0.55), keywords: &["tv console", "tv stand", "media console", "sideboard", "credenza", "tv cabinet"] },
    Category { name: "tv", target: SizeTarget::Height(0.75), keywords: &["tv", "television"] },
    Category { name: "bed", target: SizeTarget::MaxDim(2.0), keywords: &["bed", "daybed"] },
    Category { name: "rug", target: SizeTarget::MaxDim(2.4), keywords: &["rug", "carpet"] },
    Category { name: "painting", target: SizeTarget::MaxDim(0.8), keywords: &["painting", "artwork", "art print", "poster", "picture frame", "framed print", "wall art"] },
    Category { name: "vase", target: SizeTarget::Height(0.35), keywords: &["vase", "urn"] },
];

/// Size target for a category name (unknown names get a 1 m max dimension).
pub fn category_target(category: &str) -> SizeTarget {
    CATEGORIES.iter().find(|c| c.name == category).map_or(SizeTarget::MaxDim(1.0), |c| c.target)
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_string).collect()
}

/// Keyword match on whole words (plural `s`/`es` allowed); the longest keyword wins, ties go to the
/// earlier category. Returns [`DEFAULT_CATEGORY`] when nothing matches.
pub fn guess_category(prompt: &str) -> &'static str {
    let ws = words(prompt);
    let word_eq = |w: &str, k: &str| w == k || w.strip_suffix('s').is_some_and(|b| b == k) || w.strip_suffix("es").is_some_and(|b| b == k);
    let mut best: Option<(usize, &'static str)> = None;
    for c in CATEGORIES {
        for kw in c.keywords {
            let kws = words(kw);
            let hit = ws.windows(kws.len()).any(|win| win.iter().zip(&kws).all(|(w, k)| word_eq(w, k)));
            if hit && best.is_none_or(|(n, _)| kws.len() > n) {
                best = Some((kws.len(), c.name));
            }
        }
    }
    best.map_or(DEFAULT_CATEGORY, |(_, n)| n)
}

/// Lowercase ASCII slug, at most 40 chars.
pub fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= 40 {
            break;
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "item".into() } else { out }
}

/// `slug(text)-<6 hex>`.
pub fn new_id(text: &str) -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    format!("{}-{}", slug(text), &hex[..6])
}

/// Ids name files and URL segments: only `[a-z0-9_-]`.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Append one paid call to [`spend_log_path`]. Failures are reported to `log`, never fatal.
pub fn log_spend(tag: &str, usd: f64, endpoint: &str, request_id: &str, log: &dyn Fn(&str)) {
    let line = serde_json::json!({
        "ts": chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
        "kind": "job",
        "tag": tag,
        "usd": usd,
        "endpoint": endpoint,
        "request_id": request_id,
    });
    let path = spend_log_path();
    let res = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| writeln!(f, "{line}"));
    if let Err(e) = res {
        log(&format!("warning: could not append to {}: {e}", path.display()));
    }
}
