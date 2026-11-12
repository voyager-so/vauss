//! Public product page/photos -> approximate, dimension-sourced catalog item.
//! Blocking entry points belong on a worker thread. No category dimensions are invented.
use crate::{
    Catalog, CatalogItem, NormalizeOpts, SizeTarget, guess_category, log_spend, new_id,
    normalize_glb, thumb_webp, valid_id, write_atomic,
};
use anyhow::{Context, Result, bail, ensure};
use reqwest::{Url, blocking::Client, redirect::Policy};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_shots::fal::{Fal, Submitted};
use std::{
    io::Read,
    net::{IpAddr, ToSocketAddrs},
    path::Path,
    time::Duration,
};

pub const ENABLED: bool = true;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimensions {
    /// Width, height, depth in metres, not package dimensions.
    pub dims_m: [f32; 3],
    pub source: String,
    pub evidence: String,
    /// Null means the source gives no measurement tolerance; it does not mean zero error.
    pub uncertainty_m: Option<[f32; 3]>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProductRequest {
    #[serde(alias = "product_url")]
    pub url: Option<String>,
    #[serde(default)]
    pub images_b64: Vec<String>,
    pub tag: Option<String>,
    /// Explicitly selected views of the same product/variant; overrides page image selection.
    #[serde(default)]
    pub image_urls: Vec<String>,
    pub dimensions: Option<Dimensions>,
    pub name: Option<String>,
    pub id: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductPage {
    pub url: String,
    pub retrieved_at: String,
    pub name: String,
    pub sku: Option<String>,
    pub image_urls: Vec<String>,
    pub dimensions: Option<Dimensions>,
    /// Discovery only. Public accessibility does not grant redistribution rights.
    pub manufacturer_model_urls: Vec<String>,
}

/// Conservative public-address policy: excludes private, shared, test, transition and special ranges.
fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => {
            let [x, y, z, w] = a.octets();
            !(x == 0
                || x == 10
                || x == 127
                || x >= 224
                || (x == 100 && (64..128).contains(&y))
                || (x == 169 && y == 254)
                || (x == 168 && y == 63 && z == 129 && w == 16) // Azure platform virtual IP
                || (x == 172 && (16..32).contains(&y))
                || (x == 192 && (y == 168 || y == 0 || (y == 88 && z == 99)))
                || (x == 198 && (y == 18 || y == 19 || (y == 51 && z == 100)))
                || (x == 203 && y == 0 && z == 113))
        }
        IpAddr::V6(a) => {
            let s = a.segments();
            (s[0] & 0xe000) == 0x2000
                && s[0] != 0x2002
                && !(s[0] == 0x2001 && (s[1] < 0x200 || s[1] == 0xdb8))
                && !(s[0] == 0x3fff && s[1] < 0x1000)
        }
    }
}
/// HTTPS only, no credentials/nonstandard ports/proxies. Every redirect resolves and pins all
/// validated addresses on a fresh client, preventing a second DNS lookup (rebinding).
pub fn fetch_public(input: &str, limit: usize) -> Result<(String, Vec<u8>)> {
    let mut url = Url::parse(input).context("invalid public URL")?;
    for _ in 0..6 {
        ensure!(
            url.scheme() == "https"
                && url.username().is_empty()
                && url.password().is_none()
                && url.port_or_known_default() == Some(443),
            "only public HTTPS URLs on port 443 without credentials are accepted"
        );
        let host = url.host_str().context("URL has no host")?;
        let host = host.trim_matches(['[', ']']);
        ensure!(
            host.contains('.') || host.contains(':'),
            "local hostnames are not accepted"
        );
        let addrs: Vec<_> = (host, 443).to_socket_addrs()?.collect();
        ensure!(
            !addrs.is_empty() && addrs.iter().all(|a| public_ip(a.ip())),
            "URL resolves to a non-public address"
        );
        let client = Client::builder()
            .no_proxy()
            .redirect(Policy::none())
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .resolve_to_addrs(host, &addrs)
            .user_agent("SplatProductIngest/1.0")
            .build()?;
        let response = client.get(url.clone()).send()?;
        if response.status().is_redirection() {
            let next = response
                .headers()
                .get(reqwest::header::LOCATION)
                .context("redirect without location")?
                .to_str()?;
            url = url.join(next)?;
            continue;
        }
        let response = response.error_for_status()?;
        ensure!(
            response.content_length().is_none_or(|n| n <= limit as u64),
            "public resource too large"
        );
        let mut bytes = Vec::new();
        response.take(limit as u64 + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= limit, "public resource too large");
        return Ok((url.to_string(), bytes));
    }
    bail!("too many redirects")
}
fn typed(v: &Value, name: &str) -> bool {
    v["@type"].as_str() == Some(name)
        || v["@type"]
            .as_array()
            .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(name)))
}
fn nodes<'a>(v: &'a Value, out: &mut Vec<&'a Value>) {
    if let Some(a) = v.as_array() {
        for x in a {
            nodes(x, out);
        }
    }
    if v.is_object() {
        out.push(v);
        if let Some(a) = v.get("@graph") {
            nodes(a, out);
        }
    }
}
fn metres(v: &Value) -> Option<f32> {
    let (n, u) = if let Some(s) = v.as_str() {
        let i = s.find(|c: char| !c.is_ascii_digit() && c != '.' && !c.is_whitespace())?;
        (
            s[..i].trim().parse::<f32>().ok()?,
            s[i..].trim().to_lowercase(),
        )
    } else {
        let n = v["value"]
            .as_f64()
            .map(|n| n as f32)
            .or_else(|| v["value"].as_str()?.parse().ok())?;
        (
            n,
            v["unitCode"]
                .as_str()
                .or_else(|| v["unitText"].as_str())?
                .to_lowercase(),
        )
    };
    let f = match u.as_str() {
        "m" | "mtr" | "metre" | "meter" => 1.,
        "cm" | "cmt" => 0.01,
        "mm" | "mmt" => 0.001,
        "in" | "inch" | "inches" | "inh" => 0.0254,
        "ft" | "foot" | "feet" | "fot" => 0.3048,
        _ => return None,
    };
    let n = n * f;
    (n.is_finite() && n > 0. && n <= 30.).then_some(n)
}
fn urls(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => {
            for x in a {
                urls(x, out);
            }
        }
        Value::Object(_) => {
            if let Some(s) = v["contentUrl"].as_str().or_else(|| v["url"].as_str()) {
                out.push(s.into())
            }
        }
        _ => {}
    }
}
/// Strict JSON-LD Product parser. Ambiguous variant pages fail rather than selecting silently.
pub fn parse_page(url: &str, html: &str) -> Result<ProductPage> {
    let doc = Html::parse_document(html);
    let selector = Selector::parse("script[type='application/ld+json']").unwrap();
    let values: Vec<Value> = doc
        .select(&selector)
        .filter_map(|e| serde_json::from_str(&e.inner_html()).ok())
        .collect();
    let mut all = Vec::new();
    for v in &values {
        nodes(v, &mut all);
    }
    let products: Vec<_> = all
        .iter()
        .copied()
        .filter(|v| typed(v, "Product"))
        .collect();
    ensure!(
        products.len() == 1,
        "page must contain one unambiguous JSON-LD Product; supply selected photos and sourced dimensions instead"
    );
    let p = products[0];
    let dimensions = match (metres(&p["width"]),metres(&p["height"]),metres(&p["depth"])) {
        (Some(w),Some(h),Some(d)) => Some(Dimensions {dims_m:[w,h,d],source:url.into(),evidence:json!({"width":p["width"],"height":p["height"],"depth":p["depth"],"path":"JSON-LD Product"}).to_string(),uncertainty_m:None}),
        _=>None,
    };
    let mut images = Vec::new();
    urls(&p["image"], &mut images);
    let mut models = Vec::new();
    for v in all {
        if typed(v, "3DModel") {
            urls(&v["encoding"], &mut models);
        }
    }
    let base = Url::parse(url)?;
    let resolve = |items: Vec<String>| {
        items
            .into_iter()
            .filter_map(|s| base.join(&s).ok().map(|u| u.to_string()))
            .collect()
    };
    Ok(ProductPage {
        url: url.into(),
        retrieved_at: chrono::Utc::now().to_rfc3339(),
        name: p["name"].as_str().unwrap_or("Product").into(),
        sku: p["sku"]
            .as_str()
            .or_else(|| p["mpn"].as_str())
            .map(str::to_string),
        image_urls: resolve(images),
        dimensions,
        manufacturer_model_urls: resolve(models),
    })
}

pub fn inspect(url: &str) -> Result<ProductPage> {
    let (url, bytes) = fetch_public(url, 8 * 1024 * 1024)?;
    parse_page(
        &url,
        std::str::from_utf8(&bytes).context("product page is not UTF-8")?,
    )
}

/// Generate an approximate product, retaining source/achieved dimensions and evidence beside GLB.
/// Page-only mode uses the primary photo: blindly feeding room/detail/packaging gallery images
/// to multiview can create extra geometry. Callers may select 1–5 matching views explicitly.
pub fn ingest(
    fal: &Fal,
    req: &ProductRequest,
    out_dir: &Path,
    log: impl Fn(&str),
) -> Result<CatalogItem> {
    let id = req.id.clone().unwrap_or_else(|| new_id("product"));
    ensure!(valid_id(&id), "invalid product id");
    let work = out_dir.join("work").join(&id);
    std::fs::create_dir_all(&work)?;
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(work.join("ingest.lock"))?;
    lock_file
        .try_lock()
        .context("this product is already being ingested")?;
    // A directory is immutable with respect to its input, avoiding accidental cached-job reuse.
    let request_bytes = serde_json::to_vec_pretty(req)?;
    let saved = work.join("product-request.json");
    if saved.exists() {
        ensure!(
            std::fs::read(&saved)? == request_bytes,
            "id already belongs to a different product request"
        );
    } else {
        write_atomic(&saved, &request_bytes)?;
    }
    let page_path = work.join("product-page.json");
    let page: Option<ProductPage> = if let Some(url) = &req.url {
        if page_path.exists() {
            Some(serde_json::from_slice(&std::fs::read(&page_path)?)?)
        } else {
            let p = inspect(url)?;
            write_atomic(&page_path, &serde_json::to_vec_pretty(&p)?)?;
            Some(p)
        }
    } else {
        None
    };
    let dims = req.dimensions.clone().or_else(||page.as_ref()?.dimensions.clone()).context("product dimensions missing; supply width/height/depth with a source and evidence (no category defaults)")?;
    ensure!(
        dims.dims_m
            .iter()
            .all(|n| n.is_finite() && *n > 0. && *n <= 30.),
        "dimensions must be finite positive metres <=30"
    );
    ensure!(
        !dims.source.trim().is_empty() && !dims.evidence.trim().is_empty(),
        "dimension source and evidence are required"
    );
    ensure!(
        dims.uncertainty_m
            .is_none_or(|v| v.iter().all(|n| n.is_finite() && *n >= 0.)),
        "invalid dimension uncertainty"
    );
    ensure!(
        req.images_b64.is_empty() || req.image_urls.is_empty(),
        "choose uploaded photos or public photo URLs, not both"
    );
    let views = if !req.images_b64.is_empty() {
        (0..req.images_b64.len())
            .map(|i| format!("uploaded-photo-{i}"))
            .collect()
    } else if req.image_urls.is_empty() {
        page.as_ref()
            .context("need product URL or photos")?
            .image_urls
            .iter()
            .take(1)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        req.image_urls.clone()
    };
    ensure!(
        (1..=5).contains(&views.len()),
        "select 1–5 photos of the same product"
    );
    let mut data_urls = Vec::new();
    use base64::Engine;
    for (i, url) in views.iter().enumerate() {
        let path = work.join(format!("view-{i}.jpg"));
        let bytes = if path.exists() {
            std::fs::read(&path)?
        } else {
            let bytes = if req.images_b64.is_empty() {
                fetch_public(url, 16 * 1024 * 1024)?.1
            } else {
                let text = &req.images_b64[i];
                ensure!(text.len() <= 24 * 1024 * 1024, "uploaded image too large");
                let data = text
                    .split_once("base64,")
                    .map_or(text.as_str(), |(_, data)| data);
                base64::engine::general_purpose::STANDARD.decode(data)?
            };
            let mut reader =
                image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
            let mut limits = image::Limits::default();
            limits.max_image_width = Some(8192);
            limits.max_image_height = Some(8192);
            limits.max_alloc = Some(256 * 1024 * 1024);
            reader.limits(limits);
            let rgba = reader.decode()?.thumbnail(2048, 2048).to_rgba8();
            let img = image::RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
                let p = rgba.get_pixel(x, y).0;
                let a = u16::from(p[3]);
                image::Rgb(
                    [0, 1, 2].map(|i| ((u16::from(p[i]) * a + 255 * (255 - a) + 127) / 255) as u8),
                )
            });
            let mut jpeg = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95).encode_image(&img)?;
            write_atomic(&path, &jpeg)?;
            jpeg
        };
        data_urls.push(format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        ));
    }
    // The API accepts integer axis constraints (1..=2048), not metre units.
    // Normalize the measured aspect ratio to a longest side of 1000.
    let longest = dims.dims_m.into_iter().fold(0.0_f32, f32::max);
    let axes = dims
        .dims_m
        .map(|d| (d / longest * 1000.).round().clamp(1., 2048.) as u32);
    let bbox = json!({"width":axes[0],"height":axes[1],"length":axes[2]});
    let endpoint = crate::generate::MESH_ENDPOINT;
    let raw = work.join("raw.glb");
    let submitted_path = work.join("mesh-request.json");
    let tag = format!(
        "{}-{id}",
        req.tag.as_deref().unwrap_or("app-decorator-product")
    );
    if !raw.exists() {
        let submitted: Submitted = if submitted_path.exists() {
            serde_json::from_slice(&std::fs::read(&submitted_path)?)?
        } else {
            let payload = json!({"image_urls":data_urls,"material":"PBR","geometry_file_format":"glb","quality_mesh_option":"20K Triangle","seed":42,"bbox_condition":bbox});
            write_atomic(
                &work.join("mesh-settings.json"),
                &serde_json::to_vec_pretty(
                    &json!({"image_files":(0..views.len()).map(|i|format!("view-{i}.jpg")).collect::<Vec<_>>(),"material":"PBR","geometry_file_format":"glb","quality_mesh_option":"20K Triangle","seed":42,"bbox_condition":bbox}),
                )?,
            )?;
            let sub = fal.submit(endpoint, &payload)?;
            write_atomic(&submitted_path, &serde_json::to_vec_pretty(&sub)?)?;
            log_spend(
                &tag,
                crate::generate::USD_MESH,
                endpoint,
                &sub.request_id,
                &log,
            );
            sub
        };
        log(&format!("product mesh request {}", submitted.request_id));
        let output = fal.wait(&submitted, Duration::from_secs(1200))?;
        write_atomic(
            &work.join("mesh-output.json"),
            &serde_json::to_vec_pretty(&output)?,
        )?;
        let url = crate::generate::find_glb_url(&output).context("mesh output has no GLB")?;
        let (_, bytes) = fetch_public(&url, 128 * 1024 * 1024)?;
        write_atomic(&raw, &bytes)?;
    }
    let (glb, report) = normalize_glb(
        &std::fs::read(&raw)?,
        &NormalizeOpts {
            target: SizeTarget::Dims(dims.dims_m),
            ..Default::default()
        },
    )?;
    let name = req
        .name
        .as_deref()
        .or_else(|| page.as_ref().map(|p| p.name.as_str()))
        .unwrap_or("Product");
    let source = req.url.clone().unwrap_or_else(|| views.join(" "));
    let product_note = format!(
        "Approximate shape fitted within sourced {:.1} × {:.1} × {:.1} cm; room scale estimated",
        dims.dims_m[0] * 100.,
        dims.dims_m[1] * 100.,
        dims.dims_m[2] * 100.
    );
    let item = CatalogItem {
        id: id.clone(),
        name: name.into(),
        category: guess_category(name).into(),
        glb: format!("items/{id}.glb"),
        thumb: format!("items/{id}.webp"),
        dims_m: report.dims_m,
        license: "Generated reconstruction; source product/image rights remain with their owners; redistribution rights unverified".into(),
        source: format!("Approximate reconstruction from {source}; dimensions sourced, mesh uniformly fitted; room scale estimated"),
        generated: true,
        product: Some(json!({
            "url": page.as_ref().map(|p| &p.url),
            "approximate": true,
            "note": product_note
        })),
    };
    let residual: Vec<f32> = report
        .dims_m
        .iter()
        .zip(dims.dims_m)
        .map(|(a, b)| a - b)
        .collect();
    let provenance = json!({"schema":1,"recorded_at":chrono::Utc::now().to_rfc3339(),"identity":"approximate_reconstruction","product":page,"photos":views,"dimensions":dims,"source_dims_m":dims.dims_m,"resulting_dims_m":report.dims_m,"dimension_residual_m":residual,"bbox_condition":bbox,"scaling":"uniform fit inside source envelope; not exact axis dimensions","room_fit":"room metric scale must be independently calibrated","manufacturer_model":"links are discovery candidates, not verified licenses or imported geometry","endpoint":endpoint,"request":serde_json::from_slice::<Value>(&std::fs::read(&submitted_path)?)?,"usd_estimate":crate::generate::USD_MESH,"tag":tag});
    write_atomic(
        &out_dir.join(format!("items/{id}.provenance.json")),
        &serde_json::to_vec_pretty(&provenance)?,
    )?;
    write_atomic(&out_dir.join(&item.glb), &glb)?;
    write_atomic(
        &out_dir.join(&item.thumb),
        &thumb_webp(&std::fs::read(work.join("view-0.jpg"))?, 256)?,
    )?;
    Catalog::upsert(&out_dir.join("catalog.json"), item.clone())?;
    Ok(item)
}
