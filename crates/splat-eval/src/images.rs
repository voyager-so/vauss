//! Loading images as planar f32 RGB in [0,1] and pairing ground truth with renders.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{DynamicImage, ImageReader, RgbImage, imageops::FilterType};

pub const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"];

/// Planar RGB, values in [0,1]: data[c * h * w + y * w + x].
#[derive(Clone, Debug)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

/// A ground-truth image and the render of the same view. `stem` is the shared key.
#[derive(Clone, Debug)]
pub struct Pair {
    pub stem: String,
    pub gt: PathBuf,
    pub render: PathBuf,
}

/// Pairs by file stem between two folders (gt/00012.png <-> renders/00012.jpg).
/// Returns (pairs sorted by stem, gt stems with no render).
pub fn pair_dirs(gt_dir: &Path, render_dir: &Path) -> Result<(Vec<Pair>, Vec<String>)> {
    let gts = list_images(gt_dir, image_stem)?;
    let rds = list_images(render_dir, image_stem)?;
    Ok(join(gts, rds))
}

fn image_stem(name: &str) -> Option<String> {
    let p = Path::new(name);
    let ext = p.extension()?.to_str()?.to_ascii_lowercase();
    if !IMAGE_EXTS.contains(&ext.as_str()) {
        return None;
    }
    p.file_stem()?.to_str().map(str::to_owned)
}

/// Spirula `--save-eval-images 1` output: eval-gt-NNNNN.png <-> eval-render-NNNNN.png in one dir.
pub fn spirula_pairs(run_dir: &Path) -> Result<(Vec<Pair>, Vec<String>)> {
    let keyed = |prefix: &'static str| {
        move |name: &str| {
            name.strip_prefix(prefix)?
                .strip_suffix(".png")
                .map(str::to_owned)
        }
    };
    let gts = list_images(run_dir, keyed("eval-gt-"))?;
    let rds = list_images(run_dir, keyed("eval-render-"))?;
    Ok(join(gts, rds))
}

/// `spirula_pairs` minus repeated views: Spirula's eval can export one camera twice (same GT
/// pixels) and skip another. Returns (unique pairs, gt stems with no render, pairs before dedupe).
pub fn spirula_unique_pairs(run_dir: &Path) -> Result<(Vec<Pair>, Vec<String>, usize)> {
    use std::hash::{Hash, Hasher};
    let (pairs, missing) = spirula_pairs(run_dir)?;
    let total = pairs.len();
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for p in pairs {
        let im = image::open(&p.gt).with_context(|| format!("decoding {}", p.gt.display()))?.to_rgb8();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (im.dimensions(), im.as_raw()).hash(&mut h);
        if seen.insert(h.finish()) {
            unique.push(p);
        }
    }
    Ok((unique, missing, total))
}

/// Files in `dir` keyed by `key(file_name)`. Visited in sorted order so that, like metrics.py's
/// dict build, the last path wins when two files share a key (00012.jpg vs 00012.png -> png).
fn list_images(
    dir: &Path,
    key: impl Fn(&str) -> Option<String>,
) -> Result<BTreeMap<String, PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            continue;
        }
        paths.push(entry.path());
    }
    paths.sort();
    Ok(paths
        .into_iter()
        .filter_map(|p| Some((key(p.file_name()?.to_str()?)?, p)))
        .collect())
}

fn join(
    gts: BTreeMap<String, PathBuf>,
    mut rds: BTreeMap<String, PathBuf>,
) -> (Vec<Pair>, Vec<String>) {
    let mut pairs = Vec::new();
    let mut missing = Vec::new();
    for (stem, gt) in gts {
        match rds.remove(&stem) {
            Some(render) => pairs.push(Pair { stem, gt, render }),
            None => missing.push(stem),
        }
    }
    (pairs, missing)
}

fn open(path: &Path) -> Result<DynamicImage> {
    ImageReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("decoding {}", path.display()))
}

/// Load as RGB (8-bit /255, like PIL `convert("RGB")`). With `size`, resize bicubic to (w, h).
pub fn load_rgb(path: &Path, size: Option<(usize, usize)>) -> Result<Rgb> {
    let mut im: RgbImage = open(path)?.to_rgb8();
    if let Some((w, h)) = size
        && (im.width() as usize, im.height() as usize) != (w, h)
    {
        // Catmull-Rom is the a = -0.5 cubic PIL's BICUBIC uses, and `resize` widens the support on
        // downscale like PIL's antialiasing; close to PIL, not bit-exact.
        im = image::imageops::resize(&im, w as u32, h as u32, FilterType::CatmullRom);
    }
    let (w, h) = (im.width() as usize, im.height() as usize);
    let mut data = vec![0f32; 3 * w * h];
    for (i, px) in im.pixels().enumerate() {
        for c in 0..3 {
            data[c * w * h + i] = px.0[c] as f32 / 255.0;
        }
    }
    Ok(Rgb {
        width: w,
        height: h,
        data,
    })
}

/// Mask: white (>= 128) = keep. Resized nearest to `size`. Returns h*w values of 0.0/1.0.
pub fn load_mask(path: &Path, size: (usize, usize)) -> Result<Vec<f32>> {
    let rgb = open(path)?.to_rgb8();
    let (sw, sh) = (rgb.width() as usize, rgb.height() as usize);
    // PIL convert("L"): ITU-R 601-2 luma in 16.16 fixed point.
    let luma: Vec<u8> = rgb
        .pixels()
        .map(|p| {
            let [r, g, b] = p.0.map(u32::from);
            ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16) as u8
        })
        .collect();
    let (w, h) = size;
    // PIL NEAREST: source index floor((dst + 0.5) * src / dst).
    let src = |d: usize, n_src: usize, n_dst: usize| {
        (((d as f64 + 0.5) * n_src as f64 / n_dst as f64) as usize).min(n_src - 1)
    };
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let sy = src(y, sh, h);
        for x in 0..w {
            let sx = src(x, sw, w);
            out.push(if luma[sy * sw + sx] >= 128 { 1.0 } else { 0.0 });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("splat-eval-images-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_png(path: &Path, w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) {
        RgbImage::from_fn(w, h, |x, y| image::Rgb(f(x, y)))
            .save(path)
            .unwrap();
    }

    #[test]
    fn pairs_by_stem_and_reports_missing() {
        let (g, r) = (tmpdir("pg"), tmpdir("pr"));
        for s in ["a", "b", "c"] {
            write_png(&g.join(format!("{s}.png")), 2, 2, |_, _| [0; 3]);
        }
        write_png(&r.join("a.JPG.png"), 2, 2, |_, _| [0; 3]);
        write_png(&r.join("a.png"), 2, 2, |_, _| [0; 3]);
        write_png(&r.join("c.bmp"), 2, 2, |_, _| [0; 3]);
        std::fs::write(r.join("b.txt"), "x").unwrap();
        let (pairs, missing) = pair_dirs(&g, &r).unwrap();
        let stems: Vec<_> = pairs.iter().map(|p| p.stem.as_str()).collect();
        assert_eq!(stems, ["a", "c"]);
        assert_eq!(pairs[1].render, r.join("c.bmp"));
        assert_eq!(missing, ["b"]);
    }

    #[test]
    fn spirula_pairs_by_suffix() {
        let d = tmpdir("sp");
        for n in ["00000", "00001"] {
            write_png(&d.join(format!("eval-gt-{n}.png")), 2, 2, |_, _| [0; 3]);
        }
        write_png(&d.join("eval-render-00001.png"), 2, 2, |_, _| [0; 3]);
        let (pairs, missing) = spirula_pairs(&d).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].stem, "00001");
        assert_eq!(missing, ["00000"]);
    }

    #[test]
    fn load_rgb_is_planar_over_255_and_mask_thresholds() {
        let d = tmpdir("ld");
        let p = d.join("x.png");
        write_png(&p, 3, 2, |x, y| [(x * 100) as u8, (y * 255) as u8, 128]);
        let im = load_rgb(&p, None).unwrap();
        assert_eq!((im.width, im.height), (3, 2));
        assert_eq!(im.data[2], 200.0 / 255.0); // R at (2, 0)
        assert_eq!(im.data[6 + 3], 1.0); // G at (0, 1)
        assert_eq!(im.data[12], 128.0 / 255.0);
        assert_eq!(load_rgb(&p, Some((6, 4))).unwrap().data.len(), 3 * 24);

        let m = d.join("m.png");
        write_png(&m, 2, 1, |x, _| if x == 0 { [255; 3] } else { [127; 3] });
        assert_eq!(load_mask(&m, (2, 1)).unwrap(), [1.0, 0.0]);
        assert_eq!(
            load_mask(&m, (4, 2)).unwrap(),
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        );
    }
}

/// Raw EXIF of an image with its orientation tag reset to "upright" (callers write upright pixels),
/// so a re-encoded copy keeps the camera's focal-length prior for SfM.
pub fn exif_upright(path: &Path) -> Option<Vec<u8>> {
    use image::ImageDecoder;
    if is_heif(path) {
        let mut exif = heif_exif(path)?;
        let _ = image::metadata::Orientation::remove_from_exif_chunk(&mut exif);
        return Some(exif);
    }
    let mut decoder = ImageReader::open(path).ok()?.with_guessed_format().ok()?.into_decoder().ok()?;
    let mut exif = decoder.exif_metadata().ok()??;
    let _ = image::metadata::Orientation::remove_from_exif_chunk(&mut exif);
    Some(exif)
}

/// JPEG encode with optional EXIF (from `exif_upright`).
pub fn write_jpeg(path: &Path, img: &RgbImage, quality: u8, exif: Option<Vec<u8>>) -> Result<()> {
    use image::ImageEncoder;
    let file = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?);
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(file, quality);
    if let Some(exif) = exif {
        let _ = encoder.set_exif_metadata(exif);
    }
    encoder.write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)?;
    Ok(())
}

/// Like `write_jpeg`, but with 4:2:0 chroma and optimized Huffman tables, for frames that get uploaded:
/// at quality 90 a room frame is 334 KB instead of 586 KB (q95 4:4:4), 41.8 dB from the source.
pub fn write_jpeg_compact(path: &Path, img: &RgbImage, quality: u8, exif: Option<Vec<u8>>) -> Result<()> {
    use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
    let mut encoder = Encoder::new_file(path, quality).with_context(|| format!("creating {}", path.display()))?;
    encoder.set_sampling_factor(SamplingFactor::R_4_2_0);
    encoder.set_optimized_huffman_tables(true);
    if let Some(exif) = exif {
        let mut app1 = b"Exif\0\0".to_vec();
        app1.extend_from_slice(&exif);
        // Over a segment's 64 KB limit the frame goes without EXIF, as with the image crate.
        let _ = encoder.add_app_segment(1, &app1);
    }
    let (w, h) = (u16::try_from(img.width())?, u16::try_from(img.height())?);
    encoder.encode(img.as_raw(), w, h, ColorType::Rgb).with_context(|| format!("encoding {}", path.display()))?;
    Ok(())
}

/// ffmpeg used to decode HEIC/HEIF photos (the image crate cannot); backends set it once.
pub static FFMPEG: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

pub fn is_heif(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "heic" | "heif" | "hif"))
}

/// Children of an ISOBMFF box payload: (type, payload).
fn boxes(mut data: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = vec![];
    while data.len() >= 8 {
        let size = u32::from_be_bytes(data[..4].try_into().unwrap()) as u64;
        let kind: [u8; 4] = data[4..8].try_into().unwrap();
        let (header, size) = match size {
            1 if data.len() >= 16 => (16, u64::from_be_bytes(data[8..16].try_into().unwrap())),
            0 => (8, data.len() as u64),
            n => (8, n),
        };
        if size < header as u64 || size > data.len() as u64 {
            break;
        }
        out.push((kind, &data[header..size as usize]));
        data = &data[size as usize..];
    }
    out
}

fn be(b: &[u8], n: usize) -> Option<u64> {
    (b.len() >= n).then(|| b[..n].iter().fold(0u64, |a, &x| (a << 8) | x as u64))
}

/// EXIF (TIFF) of a HEIF file: the `Exif` item located through iinf + iloc in the meta box.
pub fn heif_exif(path: &Path) -> Option<Vec<u8>> {
    let file = std::fs::read(path).ok()?;
    let (_, meta) = boxes(&file).into_iter().find(|(k, _)| k == b"meta")?;
    let children = boxes(meta.get(4..)?); // full box: version + flags
    // iinf: entry_count then infe boxes; find the Exif item's ID.
    let (_, iinf) = children.iter().find(|(k, _)| k == b"iinf")?;
    let skip = if iinf[0] == 0 { 6 } else { 8 };
    let exif_id = boxes(iinf.get(skip..)?).into_iter().find_map(|(k, infe)| {
        if &k != b"infe" || infe[0] < 2 {
            return None;
        }
        let (id, rest) = if infe[0] == 2 { (be(&infe[4..], 2)?, &infe[6..]) } else { (be(&infe[4..], 4)?, &infe[8..]) };
        (rest.get(2..6)? == b"Exif").then_some(id)
    })?;
    // iloc: find that item's extents.
    let (_, iloc) = children.iter().find(|(k, _)| k == b"iloc")?;
    let version = iloc[0];
    let (off_size, len_size) = ((iloc[4] >> 4) as usize, (iloc[4] & 15) as usize);
    let (base_size, idx_size) = ((iloc[5] >> 4) as usize, if version > 0 { (iloc[5] & 15) as usize } else { 0 });
    let mut p = 6;
    let id_size = if version < 2 { 2 } else { 4 };
    let count = be(&iloc[p..], id_size)?;
    p += id_size;
    for _ in 0..count {
        let id = be(&iloc[p..], id_size)?;
        p += id_size;
        let method = if version > 0 { let m = be(&iloc[p..], 2)? & 15; p += 2; m } else { 0 };
        p += 2; // data_reference_index
        let base = if base_size > 0 { be(&iloc[p..], base_size)? } else { 0 };
        p += base_size;
        let extents = be(&iloc[p..], 2)?;
        p += 2;
        let mut data = vec![];
        for _ in 0..extents {
            p += idx_size;
            let off = if off_size > 0 { be(&iloc[p..], off_size)? } else { 0 };
            p += off_size;
            let len = be(&iloc[p..], len_size)?;
            p += len_size;
            if id == exif_id && method == 0 {
                let start = (base + off) as usize;
                data.extend_from_slice(file.get(start..start + len as usize)?);
            }
        }
        if id == exif_id {
            // Payload: 4-byte offset to the TIFF header (usually past "Exif\0\0").
            let tiff = be(&data, 4)? as usize + 4;
            return Some(data.get(tiff..)?.to_vec());
        }
    }
    None
}

/// Decodes a HEIC/HEIF photo with ffmpeg: iPhone photos are a grid of 512x512 HEVC tiles that
/// ffmpeg reports but does not assemble, so they are stitched with xstack and cropped. EXIF
/// orientation is applied; the returned EXIF has orientation reset (pixels are upright).
pub fn load_heif(path: &Path) -> Result<(DynamicImage, Option<Vec<u8>>)> {
    use std::process::Command;
    let ffmpeg = FFMPEG.get().cloned().unwrap_or_else(|| PathBuf::from("ffmpeg"));
    let ffprobe = ffmpeg.with_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
    let probe = Command::new(&ffprobe)
        .args(["-v", "error", "-show_stream_groups", "-show_streams", "-of", "json"])
        .arg(path)
        .output()
        .with_context(|| format!("running {}", ffprobe.display()))?;
    let j: serde_json::Value = serde_json::from_slice(&probe.stdout).context("ffprobe output")?;
    let grid = j["stream_groups"].as_array().into_iter().flatten().find(|g| g["type"] == "Tile Grid");
    let (filter, w, h) = if let Some(c) = grid.and_then(|g| g["components"].get(0)) {
        let tiles = c["subcomponents"].as_array().context("tile grid without tiles")?;
        let inputs: String = tiles.iter().map(|t| format!("[0:{}]", t["stream_index"])).collect();
        let layout = tiles
            .iter()
            .map(|t| format!("{}_{}", t["tile_horizontal_offset"], t["tile_vertical_offset"]))
            .collect::<Vec<_>>()
            .join("|");
        let (w, h) = (c["width"].as_u64().context("grid width")?, c["height"].as_u64().context("grid height")?);
        let crop = format!("crop={w}:{h}:{}:{}", c["horizontal_offset"], c["vertical_offset"]);
        let f = if tiles.len() == 1 { format!("{inputs}{crop}") } else { format!("{inputs}xstack=inputs={}:layout={layout},{crop}", tiles.len()) };
        (f, w, h)
    } else {
        // Single coded image: the largest video stream is the primary.
        let s = j["streams"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|s| s["codec_type"] == "video")
            .max_by_key(|s| s["width"].as_u64().unwrap_or(0) * s["height"].as_u64().unwrap_or(0))
            .context("HEIF has no image")?;
        let (w, h) = (s["width"].as_u64().unwrap_or(0), s["height"].as_u64().unwrap_or(0));
        (format!("[0:{}]copy", s["index"]), w, h)
    };
    let out = Command::new(&ffmpeg)
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-filter_complex", &format!("{filter}[img]"), "-map", "[img]", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .with_context(|| format!("running {}", ffmpeg.display()))?;
    let rgb = RgbImage::from_raw(w as u32, h as u32, out.stdout).with_context(|| {
        format!("decoding {} failed: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim())
    })?;
    let mut img = DynamicImage::ImageRgb8(rgb);
    let mut exif = heif_exif(path);
    if let Some(e) = exif.as_mut() {
        if let Some(o) = image::metadata::Orientation::remove_from_exif_chunk(e) {
            img.apply_orientation(o);
        }
    }
    Ok((img, exif))
}
