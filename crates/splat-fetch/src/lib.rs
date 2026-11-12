//! Pinned, on-demand assets. This synchronous API belongs on a blocking thread in async callers.
//! No helper is linked or executed here; callers choose when to launch the returned path.

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CATALOG: &str = include_str!("../assets.json");
const RECEIPT: &str = ".splat-receipt.json";
const MAX_UNPACKED: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub id: String,
    /// Rust target triple, or "any" for model weights.
    pub target: String,
    pub version: String,
    pub url: String,
    /// Pinned API asset URL for an internal GitHub release. Never a moving tag lookup.
    #[serde(default)]
    pub github_api_url: Option<String>,
    pub sha256: String,
    pub bytes: u64,
    pub format: Format,
    /// Some upstream releases wrap their actual distribution ZIP in another ZIP.
    #[serde(default)]
    pub inner_zip: Option<String>,
    /// Relative path inside the archive, or the output filename for a raw file.
    pub entrypoint: String,
    pub executable: bool,
    pub license: String,
    pub license_url: String,
    pub source_url: String,
    #[serde(default)]
    pub requires_gpl_consent: bool,
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Raw,
    Zip,
    TarGz,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LicenseConsent {
    #[default]
    Default,
    /// Caller has shown the separate GPL license and the user selected this helper.
    AllowGpl,
}

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("no pinned download for {id} on {target}; configure a local path instead")]
    Unavailable { id: String, target: String },
    #[error(
        "{id} is a separate {license} download ({url}); explicitly select this optional helper first"
    )]
    LicenseRequired {
        id: String,
        license: String,
        url: String,
    },
    #[error("SHA-256 mismatch: expected {expected}, received {actual}")]
    Checksum { expected: String, actual: String },
    #[error(
        "{id} is on a GitHub release that is not public yet; set SPLAT_GITHUB_TOKEN to a token with read access to the repository, use a release archive that bundles tools/{id}/, or configure a local copy (SPLAT_FFMPEG for ffmpeg)"
    )]
    AuthenticationRequired { id: String },
}

pub fn catalog() -> anyhow::Result<Vec<Asset>> {
    let assets: Vec<Asset> = serde_json::from_str(CATALOG)?;
    for asset in &assets {
        asset.validate()?;
    }
    Ok(assets)
}

pub fn host_target() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("windows", "x86_64") => "x86_64-pc-windows-msvc",
        _ => "unsupported",
    }
}

pub fn asset_for(id: &str, target: &str) -> anyhow::Result<Asset> {
    let id = if id == "ffmpeg" { "ffmpeg-public" } else { id };
    catalog()?
        .into_iter()
        .find(|a| a.id == id && (a.target == target || a.target == "any"))
        .ok_or_else(|| {
            FetchError::Unavailable {
                id: id.into(),
                target: target.into(),
            }
            .into()
        })
}

pub fn default_cache_dir() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os("SPLAT_CACHE_DIR").filter(|v| !v.is_empty()) {
        return Ok(path.into());
    }
    if cfg!(windows) {
        return std::env::var_os("LOCALAPPDATA")
            .map(|p| PathBuf::from(p).join("splat/cache"))
            .context("LOCALAPPDATA is unset; set SPLAT_CACHE_DIR");
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(path).join("splat"));
    }
    let home = std::env::var_os("HOME").context("HOME is unset; set SPLAT_CACHE_DIR")?;
    Ok(PathBuf::from(home).join(if cfg!(target_os = "macos") {
        "Library/Caches/splat"
    } else {
        ".cache/splat"
    }))
}

impl Asset {
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            !self.id.is_empty()
                && self
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b)),
            "invalid asset id"
        );
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid SHA-256 for {}",
            self.id
        );
        ensure!(self.url.starts_with("https://"), "asset URL must use HTTPS");
        ensure!(self.bytes > 0, "asset size must be pinned");
        if let Some(url) = &self.github_api_url {
            let url = reqwest::Url::parse(url)?;
            let parts: Vec<_> = url.path().split('/').collect();
            ensure!(
                url.scheme() == "https"
                    && url.host_str() == Some("api.github.com")
                    && url.port().is_none()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.query().is_none()
                    && url.fragment().is_none()
                    && parts.len() == 7
                    && parts[1] == "repos"
                    && !parts[2].is_empty()
                    && !parts[3].is_empty()
                    && parts[4] == "releases"
                    && parts[5] == "assets"
                    && parts[6].parse::<u64>().is_ok(),
                "invalid GitHub release asset API URL"
            );
        }
        safe_relative(Path::new(&self.entrypoint))?;
        if let Some(inner) = &self.inner_zip {
            safe_relative(Path::new(inner))?;
        }
        Ok(())
    }

    fn key(&self) -> String {
        format!("{}-{}", self.id, self.sha256)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Receipt {
    asset: Asset,
    files: BTreeMap<String, String>,
    #[serde(default)]
    links: BTreeMap<String, String>,
}

pub struct Fetcher {
    root: PathBuf,
    github_token: Option<String>,
}

impl Fetcher {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            root: cache_dir.into(),
            github_token: None,
        }
    }

    /// Explicit caller-provided token, used only for pinned api.github.com assets.
    /// It is never stored in a receipt, URL or log, or sent to other download hosts.
    pub fn with_github_token(mut self, token: impl Into<String>) -> Self {
        self.github_token = Some(token.into());
        self
    }

    pub fn ensure(&self, id: &str, consent: LicenseConsent) -> anyhow::Result<PathBuf> {
        self.ensure_asset(&asset_for(id, host_target())?, consent)
    }

    /// Explicit configuration wins over Voyager's staged tree, then the public LGPL pin.
    pub fn ffmpeg(
        &self,
        configured: Option<&Path>,
        voyager_tree: Option<&Path>,
    ) -> anyhow::Result<PathBuf> {
        let name = if cfg!(windows) {
            "ffmpeg.exe"
        } else {
            "ffmpeg"
        };
        if let Some(path) = configured
            .map(Path::to_path_buf)
            .or_else(|| voyager_tree.map(|p| p.join("bin").join(name)))
        {
            ensure!(
                path.is_file(),
                "configured FFmpeg is missing: {}",
                path.display()
            );
            return Ok(path);
        }
        self.ensure("ffmpeg", LicenseConsent::Default)
    }

    /// Offline integrity check. Never downloads, creates directories, or accepts a partial install.
    pub fn cached(&self, asset: &Asset) -> anyhow::Result<Option<PathBuf>> {
        asset.validate()?;
        let dir = self.root.join(asset.key());
        let Ok(bytes) = fs::read(dir.join(RECEIPT)) else {
            return Ok(None);
        };
        let Ok(receipt) = serde_json::from_slice::<Receipt>(&bytes) else {
            return Ok(None);
        };
        if receipt.asset.sha256 != asset.sha256
            || receipt.asset.url != asset.url
            || !receipt.files.contains_key(&asset.entrypoint)
            || (asset.id == "spirula-grounding-dino" && !receipt.files.contains_key("vocab.txt"))
        {
            return Ok(None);
        }
        for (relative, expected) in receipt.files {
            safe_relative(Path::new(&relative))?;
            let path = dir.join(relative);
            if !fs::symlink_metadata(&path).is_ok_and(|m| m.is_file()) {
                return Ok(None);
            }
            if !hash_file(&path).is_ok_and(|actual| actual == expected) {
                return Ok(None);
            }
        }
        for (relative, expected) in receipt.links {
            safe_relative(Path::new(&relative))?;
            let link = dir.join(relative);
            if fs::read_link(&link).is_ok_and(|p| p == Path::new(&expected))
                && link
                    .canonicalize()
                    .is_ok_and(|p| dir.canonicalize().is_ok_and(|root| p.starts_with(root)))
            {
                continue;
            }
            return Ok(None);
        }
        Ok(Some(dir.join(&asset.entrypoint)))
    }

    pub fn ensure_asset(&self, asset: &Asset, consent: LicenseConsent) -> anyhow::Result<PathBuf> {
        asset.validate()?;
        if asset.requires_gpl_consent && consent != LicenseConsent::AllowGpl {
            return Err(FetchError::LicenseRequired {
                id: asset.id.clone(),
                license: asset.license.clone(),
                url: asset.license_url.clone(),
            }
            .into());
        }
        fs::create_dir_all(&self.root)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join(format!("{}.lock", asset.key())))?;
        // OS locks release on process death; no stale PID files or half-installed cache entries.
        lock.lock()?;
        if let Some(path) = self.cached(asset)? {
            return Ok(path);
        }
        eprintln!(
            "Downloading {} {} ({})\nLicense: {}\nSource: {}",
            asset.id, asset.version, asset.license, asset.license_url, asset.source_url
        );
        let temp = tempfile::tempdir_in(&self.root)?;
        let archive = temp.path().join("download");
        let client = reqwest::blocking::Client::builder()
            .https_only(true)
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(1800))
            .user_agent(concat!("splat/", env!("CARGO_PKG_VERSION")))
            .build()?;
        let token = self.github_token.as_deref().filter(|t| !t.is_empty());
        let response = open_download(&client, asset, token)?;
        let mut output = File::create(&archive)?;
        copy_verified(response, &mut output, asset)?;
        output.sync_all()?;
        drop(output);
        let install = temp.path().join("install");
        fs::create_dir(&install)?;
        unpack(&archive, &install, asset)?;
        if let Some(inner) = &asset.inner_zip {
            let nested = temp.path().join("nested");
            fs::create_dir(&nested)?;
            let mut nested_asset = asset.clone();
            nested_asset.format = Format::Zip;
            unpack(&install.join(inner), &nested, &nested_asset)?;
            fs::remove_dir_all(&install)?;
            fs::rename(&nested, &install)?;
        }
        let entrypoint = install.join(&asset.entrypoint);
        ensure!(
            entrypoint.is_file(),
            "archive does not contain {}",
            asset.entrypoint
        );
        if asset.executable {
            set_executable(&entrypoint)?;
        }
        // FFmpeg's sibling ffprobe must remain runnable as well.
        if asset.id == "ffmpeg-public" {
            set_executable(&install.join(if asset.entrypoint.ends_with(".exe") {
                "ffprobe.exe"
            } else {
                "ffprobe"
            }))?;
        }
        if matches!(asset.id.as_str(), "lpips-alex" | "lpips-vgg") {
            fs::write(
                install.join("MODEL_LICENSES.txt"),
                include_str!("../licenses/lpips.txt"),
            )?;
        }
        if asset.id == "colmap" && asset.target == "x86_64-pc-windows-msvc" {
            // Upstream launches bin\colmap.exe via COLMAP.bat, which points QT_PLUGIN_PATH at
            // plugins\. Run directly, Qt aborts without its "windows" platform plugin; qt.conf
            // beside the executable gives Qt the same plugin path for every caller.
            fs::write(
                install.join("bin").join("qt.conf"),
                "[Paths]\nPlugins = ../plugins\n",
            )?;
        }
        if asset.id == "spirula-grounding-dino" {
            // Spirula resolves its tokenizer beside the detector, not via a CLI argument.
            let vocab = self.ensure("grounding-dino-vocab", LicenseConsent::Default)?;
            fs::copy(vocab, install.join("vocab.txt"))?;
        }
        let mut files = BTreeMap::new();
        let mut links = BTreeMap::new();
        inventory(&install, &install, &mut files, &mut links)?;
        let receipt = Receipt {
            asset: asset.clone(),
            files,
            links,
        };
        fs::write(install.join(RECEIPT), serde_json::to_vec_pretty(&receipt)?)?;
        fs::write(
            install.join("DOWNLOAD_NOTICE.txt"),
            format!(
                "{} {}\nLicense: {}\n{}\nSource: {}\nDownloaded from: {}\nSHA-256: {}\n{}\n",
                asset.id,
                asset.version,
                asset.license,
                asset.license_url,
                asset.source_url,
                asset.url,
                asset.sha256,
                asset.note
            ),
        )?;
        let destination = self.root.join(asset.key());
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::rename(&install, &destination)?;
        Ok(destination.join(&asset.entrypoint))
    }
}

fn copy_verified(
    mut input: impl Read,
    output: &mut impl Write,
    asset: &Asset,
) -> anyhow::Result<()> {
    let mut hash = Sha256::new();
    let mut size = 0;
    let mut buffer = [0; 65536];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        size += n as u64;
        ensure!(
            size <= asset.bytes,
            "download exceeds pinned size of {} bytes",
            asset.bytes
        );
        hash.update(&buffer[..n]);
        output.write_all(&buffer[..n])?;
    }
    ensure!(
        size == asset.bytes,
        "incomplete download: expected {}, received {size} bytes",
        asset.bytes
    );
    let actual = format!("{:x}", hash.finalize());
    if actual != asset.sha256.to_ascii_lowercase() {
        return Err(FetchError::Checksum {
            expected: asset.sha256.clone(),
            actual,
        }
        .into());
    }
    Ok(())
}

fn safe_relative(path: &Path) -> anyhow::Result<()> {
    let text = path.to_string_lossy();
    ensure!(
        !text.is_empty() && !text.contains(['\\', ':']) && !path.is_absolute(),
        "unsafe archive path: {}",
        path.display()
    );
    ensure!(
        path.components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)),
        "unsafe archive path: {}",
        path.display()
    );
    Ok(())
}

fn unpack(archive: &Path, destination: &Path, asset: &Asset) -> anyhow::Result<()> {
    let mut remaining = MAX_UNPACKED;
    let mut links = Vec::new();
    match asset.format {
        Format::Raw => {
            fs::copy(archive, destination.join(&asset.entrypoint))?;
        }
        Format::Zip => {
            let mut zip = zip::ZipArchive::new(File::open(archive)?)?;
            for i in 0..zip.len() {
                let mut entry = zip.by_index_with_options(
                    i,
                    zip::read::ZipReadOptions::new().enforce_header_consistency(true),
                )?;
                let path = PathBuf::from(entry.name().context("invalid ZIP entry name")?.as_ref());
                safe_relative(&path)?;
                if entry.is_symlink() {
                    ensure!(entry.size() < 4096, "archive link target is too long");
                    let mut target = String::new();
                    entry.read_to_string(&mut target)?;
                    safe_relative(Path::new(&target))?;
                    links.push((path, PathBuf::from(target)));
                    continue;
                }
                if entry.is_dir() {
                    fs::create_dir_all(destination.join(path))?;
                    continue;
                }
                let size = entry.size();
                let executable = entry.unix_mode().is_some_and(|mode| mode & 0o111 != 0);
                let output = destination.join(path);
                write_entry(&mut entry, &output, size, &mut remaining)?;
                if executable {
                    set_executable(&output)?;
                }
            }
        }
        Format::TarGz => {
            let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(File::open(archive)?));
            for entry in tar.entries()? {
                let mut entry = entry?;
                let path = entry.path()?.into_owned();
                safe_relative(&path)?;
                let kind = entry.header().entry_type();
                if kind.is_dir() {
                    fs::create_dir_all(destination.join(path))?;
                    continue;
                }
                if kind.is_symlink() {
                    let target = entry
                        .link_name()?
                        .context("missing link target")?
                        .into_owned();
                    safe_relative(&target)?;
                    links.push((path, target));
                    continue;
                }
                ensure!(kind.is_file(), "archive links/devices are not allowed");
                let mode = entry.header().mode()?;
                let size = entry.size();
                let output = destination.join(path);
                write_entry(&mut entry, &output, size, &mut remaining)?;
                if mode & 0o111 != 0 {
                    set_executable(&output)?;
                }
            }
        }
    }
    // Defer links until all regular files exist: extraction can never traverse a link.
    // Only relative, in-tree, non-dangling links are accepted (macOS framework bundles).
    while !links.is_empty() {
        let before = links.len();
        let mut pending = Vec::new();
        for (relative, target) in links {
            let path = destination.join(&relative);
            let parent = path.parent().context("missing link parent")?;
            let Ok(resolved) = parent.join(&target).canonicalize() else {
                pending.push((relative, target));
                continue;
            };
            ensure!(
                resolved.starts_with(destination.canonicalize()?),
                "archive link escapes destination"
            );
            fs::create_dir_all(parent)?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &path)?;
            #[cfg(not(unix))]
            bail!("archive symlinks are not supported on this platform");
        }
        ensure!(pending.len() < before, "dangling or cyclic archive links");
        links = pending;
    }
    Ok(())
}

fn write_entry(
    input: &mut impl Read,
    path: &Path,
    size: u64,
    remaining: &mut u64,
) -> anyhow::Result<()> {
    *remaining = remaining
        .checked_sub(size)
        .context("archive exceeds extraction size limit")?;
    fs::create_dir_all(path.parent().context("missing parent")?)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let copied = io::copy(&mut input.take(size + 1), &mut file)?;
    ensure!(copied == size, "archive entry size mismatch");
    Ok(())
}

fn set_executable(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn hash_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    io::copy(&mut file, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn inventory(
    root: &Path,
    dir: &Path,
    files: &mut BTreeMap<String, String>,
    links: &mut BTreeMap<String, String>,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            inventory(root, &path, files, links)?;
        } else if kind.is_file() {
            files.insert(
                path.strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
                hash_file(&path)?,
            );
        } else if kind.is_symlink() {
            links.insert(
                path.strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
                fs::read_link(path)?.to_string_lossy().into_owned(),
            );
        } else {
            bail!("unexpected file type in asset cache")
        }
    }
    Ok(())
}

/// The asset's download, anonymous first: its public URL works for every caller once the repository
/// is public, and an expired or wrong token can't break it. Only when that URL answers as a release
/// the caller can't read does a token fetch it through the GitHub API.
fn open_download(
    client: &reqwest::blocking::Client,
    asset: &Asset,
    token: Option<&str>,
) -> anyhow::Result<reqwest::blocking::Response> {
    let public = client.get(&asset.url).send()?;
    let Some(api) = asset.github_api_url.as_deref().filter(|_| needs_token(public.status())) else {
        return Ok(public.error_for_status()?);
    };
    let required = || FetchError::AuthenticationRequired {
        id: asset.id.clone(),
    };
    let Some(token) = token else {
        return Err(required().into());
    };
    let authed = client
        .get(api)
        .bearer_auth(token)
        .header(reqwest::header::ACCEPT, "application/octet-stream")
        .send()?;
    if needs_token(authed.status()) {
        let status = authed.status();
        return Err(anyhow::Error::from(required()).context(format!(
            "SPLAT_GITHUB_TOKEN was refused for {} (HTTP {status}): it needs read access to the repository",
            asset.id
        )));
    }
    Ok(authed.error_for_status()?)
}

/// An anonymous download of a GitHub release asset answered as a repository the caller can't read.
fn needs_token(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 401 | 403 | 404)
}

#[cfg(test)]
mod tests;
