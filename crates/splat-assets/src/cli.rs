//! `splat assets ...` (hidden): build catalog items without the server.
//!
//!   splat assets normalize in.glb out.glb --category sofa --thumb-from photo.jpg --thumb out.webp
//!   splat assets normalize in.glb out.glb --dims 2.0,0.8,0.9 --up z
//!   splat assets restyle start.png --out SCENE/restyle      (all five presets, for static deploys)
//!   splat assets generate --prompt "mid-century walnut armchair" --out SCENE/generated

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Subcommand;

use crate::generate::{GenRequest, PRESETS, RESTYLE_ENDPOINT, RestyleRequest, generate_item, restyle};
use crate::glb::{NormalizeOpts, NormalizeReport, SizeTarget, Up, normalize_glb, thumb_webp};
use crate::{CATEGORIES, category_target, write_atomic};

#[derive(Subcommand)]
pub enum Cmd {
    /// Normalize a GLB (Y-up, meters, bottom-center origin, textures <= --max-texture).
    Normalize {
        input: PathBuf,
        output: PathBuf,
        /// Size by the category's real-world default (see `splat assets categories`).
        #[arg(long, conflicts_with_all = ["height", "dims"])]
        category: Option<String>,
        /// Uniform scale to this height in meters.
        #[arg(long, conflicts_with = "dims")]
        height: Option<f32>,
        /// Fit inside w,h,d meters (uniform scale).
        #[arg(long, value_delimiter = ',', num_args = 3)]
        dims: Option<Vec<f32>>,
        /// Source up axis: y (glTF default) or z.
        #[arg(long, default_value = "y")]
        up: String,
        #[arg(long, default_value_t = 1024)]
        max_texture: u32,
        /// Image to build a thumbnail from.
        #[arg(long, requires = "thumb")]
        thumb_from: Option<PathBuf>,
        /// Thumbnail output (.webp, 256 px).
        #[arg(long, requires = "thumb_from")]
        thumb: Option<PathBuf>,
    },
    /// Generate one item on fal (prompt or image -> 3D -> normalized catalog item). Calls third-party
    /// hosted models on fal.ai, billed to your FAL_KEY; use of the outputs is governed by each model
    /// provider's terms.
    Generate {
        #[arg(long)]
        prompt: Option<String>,
        /// Source image instead of text -> image.
        #[arg(long)]
        image: Option<PathBuf>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        category: Option<String>,
        #[arg(long, value_delimiter = ',', num_args = 3)]
        dims: Option<Vec<f32>>,
        /// Resume an earlier item's work dir.
        #[arg(long)]
        id: Option<String>,
        /// Output dir (items/, work/, catalog.json), usually SCENE/generated.
        #[arg(long)]
        out: PathBuf,
    },
    /// Restyle one view with every preset (or the given ones) for static deploys:
    /// writes OUT/<preset>.jpg and OUT/restyle.json.
    Restyle {
        image: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Presets to run (default: all five).
        #[arg(long = "preset")]
        presets: Vec<String>,
        /// Extra prompt text appended to every preset.
        #[arg(long)]
        prompt: Option<String>,
    },
    /// List categories and their default sizes.
    Categories,
}

fn dims3(v: &[f32]) -> Result<[f32; 3]> {
    <[f32; 3]>::try_from(v).ok().context("--dims needs w,h,d")
}

/// Normalize `input` into `output`; the CLI-facing wrapper around [`normalize_glb`].
pub fn normalize_file(input: &Path, output: &Path, target: SizeTarget, up: Up, max_texture: u32) -> Result<NormalizeReport> {
    let bytes = std::fs::read(input).with_context(|| format!("read {}", input.display()))?;
    let (glb, report) = normalize_glb(&bytes, &NormalizeOpts { up, target, max_texture })?;
    write_atomic(output, &glb)?;
    Ok(report)
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Normalize { input, output, category, height, dims, up, max_texture, thumb_from, thumb } => {
            let target = match (category, height, dims) {
                (Some(c), ..) => {
                    if !CATEGORIES.iter().any(|k| k.name == c) {
                        println!("note: unknown category {c:?}, using a 1 m max dimension");
                    }
                    category_target(&c)
                }
                (_, Some(h), _) => SizeTarget::Height(h),
                (.., Some(d)) => SizeTarget::Dims(dims3(&d)?),
                _ => SizeTarget::None,
            };
            let up = match up.as_str() {
                "y" | "Y" => Up::Y,
                "z" | "Z" => Up::Z,
                other => bail!("--up must be y or z, got {other:?}"),
            };
            let report = normalize_file(&input, &output, target, up, max_texture)?;
            if let (Some(src), Some(dst)) = (thumb_from, thumb) {
                write_atomic(&dst, &thumb_webp(&std::fs::read(&src)?, 256)?)?;
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Generate { prompt, image, name, category, dims, id, out } => {
            use base64::Engine;
            let image_b64 = image.map(|p| std::fs::read(&p).map(|b| base64::engine::general_purpose::STANDARD.encode(b))).transpose()?;
            let req = GenRequest { prompt, image_b64, name, dims_m: dims.as_deref().map(dims3).transpose()?, id, category, tag: None };
            let fal = splat_shots::fal::Fal::from_env()?;
            let item = generate_item(&fal, &req, &out, |m| println!("{m}"))?;
            println!("{}", serde_json::to_string_pretty(&item)?);
        }
        Cmd::Restyle { image, out, presets, prompt } => {
            use base64::Engine;
            let presets = if presets.is_empty() { PRESETS.iter().map(|p| (*p).to_owned()).collect() } else { presets };
            let image_b64 = base64::engine::general_purpose::STANDARD.encode(std::fs::read(&image)?);
            let fal = splat_shots::fal::Fal::from_env()?;
            let stem = image.file_stem().and_then(|s| s.to_str()).unwrap_or("view").to_owned();
            let results: Vec<(String, Result<()>)> = std::thread::scope(|s| {
                let jobs: Vec<_> = presets
                    .iter()
                    .map(|preset| {
                        let req = RestyleRequest { image_b64: image_b64.clone(), depth_b64: None, preset: Some(preset.clone()), prompt: prompt.clone(), tag: None };
                        let (fal, out, stem) = (&fal, &out, &stem);
                        s.spawn(move || {
                            let tag = format!("app-decorator-restyle-{stem}-{preset}");
                            let r = restyle(fal, &req, &out.join("work").join(preset), &out.join(format!("{preset}.jpg")), &tag, |m| println!("{preset}: {m}"));
                            (preset.clone(), r)
                        })
                    })
                    .collect();
                jobs.into_iter().map(|j| j.join().expect("restyle thread panicked")).collect()
            });
            let mut manifest = serde_json::Map::new();
            for (preset, r) in &results {
                match r {
                    Ok(()) => {
                        manifest.insert(preset.clone(), serde_json::json!(format!("{preset}.jpg")));
                    }
                    Err(e) => println!("{preset}: FAILED {e:#}"),
                }
            }
            write_atomic(&out.join("restyle.json"), &serde_json::to_vec_pretty(&serde_json::json!({"source": image.file_name().and_then(|n| n.to_str()), "endpoint": RESTYLE_ENDPOINT, "presets": manifest}))?)?;
            if results.iter().any(|(_, r)| r.is_err()) {
                bail!("some presets failed");
            }
        }
        Cmd::Categories => {
            for c in CATEGORIES {
                println!("{:<14} {:?}  [{}]", c.name, c.target, c.keywords.join(", "));
            }
        }
    }
    Ok(())
}
