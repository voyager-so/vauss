//! Niantic SPZ writer: v3 gzip and v4 independent Zstd attribute streams; [`read_spz`] inverts it
//! (and reads upstream v2), [`read_package_splats`] reloads a web scene package.
//!
//! Written from the MIT-licensed format description at github.com/nianticlabs/spz:
//! 16-byte header {magic "NGSP", version, numPoints, shDegree, fractionalBits, flags, reserved},
//! then positions (24-bit fixed point), alphas, colors, scales, rotations (smallest three, 4 bytes),
//! SH ([coef][channel], 8-bit). Coordinates are stored as given; SPZ's native frame is RUB, which is
//! the three.js frame our exporter produces, so no axis flips are applied here.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use flate2::{Compression, write::GzEncoder};
use serde_json::Value;

use crate::ply::{Gaussians, RenderMode};

const MAGIC: u32 = 0x5053_474e;

// `splat-train`'s quantisation-aware tail (`qat.rs`) trains against these
// quantisers and the default 6/6 SH bits: change both together.
const FRACTIONAL_BITS: u8 = 12;
const COLOR_SCALE: f32 = 0.15;
/// Log-scale bytes are `(ln s + 10) * 16`: one step is 1/16 in log space.
pub const LOG_SCALE_STEPS: f64 = 16.0;
/// Highest alpha byte written. Byte 255 decodes to alpha 1.0, an infinite opacity logit for
/// logit-space loaders (upstream SPZ, Brush), which broke the most opaque splats (-0.9 dB on one scene).
const ALPHA_MAX: f32 = 254.0;
/// SPZ quantization is lossy. Bits 8/8 retain appearance more accurately than upstream 5/4.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub version: u32,
    pub sh1_bits: u8,
    pub sh_rest_bits: u8,
    pub antialiased: bool,
    /// Round once to the final SH bucket, avoiding upstream double-rounding bias.
    pub symmetric_sh: bool,
}
impl Default for Options {
    fn default() -> Self { Self { version: 4, sh1_bits: 6, sh_rest_bits: 6, antialiased: false, symmetric_sh: true } }
}

fn to_u8(x: f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

fn quantize_sh(x: f32, bucket: i32, symmetric: bool) -> u8 {
    if symmetric {
        let q = ((x * 128.0 / bucket as f32).round() * bucket as f32 + 128.0) as i32;
        return q.clamp(0, 255) as u8;
    }
    let q = (x * 128.0).round() as i32 + 128;
    ((q + bucket / 2) / bucket * bucket).clamp(0, 255) as u8
}

/// Smallest-three quaternion: 2-bit index of the largest |component| (xyzw order), then three
/// 10-bit fields (sign bit + 9-bit magnitude scaled by 1/sqrt(1/2)), packed little-endian.
fn pack_quat(wxyz: [f32; 4]) -> [u8; 4] {
    let mut q = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]];
    let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 0.0 {
        q.iter_mut().for_each(|v| *v /= n);
    } else {
        q = [0.0, 0.0, 0.0, 1.0];
    }
    let largest = (0..4).max_by(|&a, &b| q[a].abs().total_cmp(&q[b].abs())).unwrap();
    let negate = q[largest] < 0.0;
    let mut comp = largest as u32;
    for (i, v) in q.iter().enumerate() {
        if i != largest {
            let negbit = ((*v < 0.0) ^ negate) as u32;
            let mag = (511.0 * (v.abs() / std::f32::consts::FRAC_1_SQRT_2) + 0.5) as u32;
            comp = (comp << 10) | (negbit << 9) | mag.min(511);
        }
    }
    comp.to_le_bytes()
}

/// Position precision: always upstream's 12 fractional bits (|x| < 2048 units).
///
/// This used to drop to fewer bits until every coordinate fit, so a handful of distant
/// background splats coarsened the whole scene: -0.88 dB on bonsai, -0.38 room, -0.41 stump
/// vs upstream's fixed 12 (Mip-NeRF 360, 9 scenes; identical to upstream wherever it picked 12).
/// Coordinates past the range are now clamped to it (see [`position_clamped`]) instead; the
/// scene package keeps those splats losslessly
/// in its OVF1 sidecar, so only the raw single-file export clamps.
pub fn position_fractional_bits(g: &Gaussians) -> Result<u8> {
    ensure!(g.pos.iter().flatten().all(|v| v.is_finite()), "SPZ requires finite positions");
    Ok(FRACTIONAL_BITS)
}

const POS_CODE_MAX: f32 = 8_388_607.0;

/// Splats with a coordinate outside SPZ's 24-bit range at 12 fractional bits (clamped on write).
pub fn position_clamped(g: &Gaussians) -> usize {
    let scale = (1u32 << FRACTIONAL_BITS) as f32;
    g.pos.iter().filter(|p| p.iter().any(|v| (v * scale).round().abs() > POS_CODE_MAX)).count()
}

pub fn write_spz(g: &Gaussians, path: &Path) -> Result<u64> {
    write_spz_with_options(g, path, Options { version: 3, sh1_bits: 6, sh_rest_bits: 6, antialiased: g.antialiased, symmetric_sh: true })
}

pub fn write_spz_with_options(g: &Gaussians, path: &Path, options: Options) -> Result<u64> {
    ensure!(matches!(options.version, 3 | 4), "SPZ version must be 3 or 4");
    ensure!((1..=8).contains(&options.sh1_bits) && (1..=8).contains(&options.sh_rest_bits), "SH bits must be 1..8");
    ensure!(g.sh_degree <= 3, "SPZ holds SH degree <=3");
    ensure!(!g.is_empty() && g.len() <= i32::MAX as usize, "SPZ requires 1..INT32_MAX splats");
    let fractional_bits = position_fractional_bits(g)?;
    ensure!(g.scale.iter().flatten().chain(g.rot.iter().flatten()).chain(g.dc.iter().flatten()).chain(g.rest.iter()).all(|v| v.is_finite()), "SPZ requires finite attributes");
    ensure!(g.opacity.iter().all(|v| !v.is_nan()), "SPZ opacity cannot be NaN");
    let n = g.len();
    let k = g.sh_coeffs();
    let mut buf = Vec::with_capacity(16 + n * (9 + 1 + 3 + 3 + 4 + k * 3));
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&options.version.to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&[g.sh_degree as u8, fractional_bits, u8::from(options.antialiased), 0]);
    let fx = (1u32 << fractional_bits) as f32;
    for p in &g.pos {
        for v in p {
            // Clamped, not wrapped: a far splat stays at the range edge.
            let q = (v * fx).round().clamp(-POS_CODE_MAX - 1.0, POS_CODE_MAX) as i32;
            buf.extend_from_slice(&q.to_le_bytes()[..3]);
        }
    }
    for i in 0..n {
        buf.push(to_u8((g.alpha(i) * 255.0).min(ALPHA_MAX)));
    }
    for c in &g.dc {
        for v in c {
            buf.push(to_u8(v * (COLOR_SCALE * 255.0) + 0.5 * 255.0));
        }
    }
    for s in &g.scale {
        for v in s {
            buf.push(to_u8((v + 10.0) * 16.0));
        }
    }
    for r in &g.rot {
        buf.extend_from_slice(&pack_quat(*r));
    }
    let (b1, brest) = (1 << (8 - options.sh1_bits), 1 << (8 - options.sh_rest_bits));
    for (j, v) in g.rest.iter().enumerate() {
        let coef = (j / 3) % k.max(1);
        buf.push(quantize_sh(*v, if coef < 3 { b1 } else { brest }, options.symmetric_sh));
    }
    if options.version == 4 {
        // Header/TOC layout from Niantic SPZ MIT format, pin affd0ec.
        // Empty SH buffers are omitted, as required for SH0.
        let lengths = [n * 9, n, n * 3, n * 3, n * 4, n * k * 3];
        let mut offset = 16;
        let mut streams = Vec::new();
        for length in lengths.into_iter().filter(|&len| len > 0) {
            let compressed = zstd::bulk::compress(&buf[offset..offset + length], 12)?;
            streams.push((compressed, length as u64));
            offset += length;
        }
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut header = [0u8; 32];
        header[..16].copy_from_slice(&buf[..16]);
        header[15] = streams.len() as u8;
        header[16..20].copy_from_slice(&32u32.to_le_bytes());
        file.write_all(&header)?;
        for (data, size) in &streams {
            file.write_all(&(data.len() as u64).to_le_bytes())?;
            file.write_all(&size.to_le_bytes())?;
        }
        for (data, _) in streams { file.write_all(&data)?; }
        file.flush()?;
        return Ok(std::fs::metadata(path)?.len());
    }
    write_gzip(path, &buf)?;
    Ok(std::fs::metadata(path)?.len())
}

/// Input bytes per independently deflated piece of [`write_gzip`].
const GZIP_PIECE: usize = 4 << 20;

/// `buf` as one gzip member, deflated in [`GZIP_PIECE`] pieces on all cores (pigz's layout: each
/// piece ends on a sync flush, the last one finishes the stream). It decompresses to exactly `buf`
/// with any gzip reader; pieces don't share a window, so it is ~0.1% larger than one serial pass.
/// A serial pass was ~9 of the 10.3 s web package of a 4.65M-splat MAX model on a 6-vCPU runner.
fn write_gzip(path: &Path, buf: &[u8]) -> Result<()> {
    use rayon::prelude::*;
    let pieces: Vec<&[u8]> = buf.chunks(GZIP_PIECE).collect();
    if pieces.len() <= 1 {
        let mut enc = GzEncoder::new(std::fs::File::create(path)?, Compression::default());
        enc.write_all(buf)?;
        enc.finish()?;
        return Ok(());
    }
    let last = pieces.len() - 1;
    let parts: Vec<(Vec<u8>, flate2::Crc)> = pieces
        .par_iter()
        .enumerate()
        .map(|(i, piece)| -> Result<_> {
            // Raw deflate driven directly: flate2's writer lost the tail of a sync flush (zlib-rs).
            let mut deflate = flate2::Compress::new(Compression::default(), false);
            let flush = if i == last { flate2::FlushCompress::Finish } else { flate2::FlushCompress::Sync };
            let mut data = Vec::with_capacity(piece.len() + piece.len() / 8 + 4096);
            loop {
                if data.len() == data.capacity() {
                    data.reserve(1 << 20);
                }
                let status = deflate.compress_vec(&piece[deflate.total_in() as usize..], &mut data, flush)?;
                let consumed = deflate.total_in() as usize == piece.len();
                // Finish: the end of the stream; Sync: all input in and the flush fitted in the output.
                if status == flate2::Status::StreamEnd || (i != last && consumed && data.len() < data.capacity()) {
                    break;
                }
            }
            let mut crc = flate2::Crc::new();
            crc.update(piece);
            Ok((data, crc))
        })
        .collect::<Result<_>>()?;
    let mut crc = flate2::Crc::new();
    for (_, c) in &parts {
        crc.combine(c);
    }
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    // GzEncoder's default header: no name, mtime 0, xfl 0, OS unknown (255).
    file.write_all(&[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255])?;
    for (data, _) in &parts {
        file.write_all(data)?;
    }
    file.write_all(&crc.sum().to_le_bytes())?;
    file.write_all(&(buf.len() as u32).to_le_bytes())?;
    file.flush()?;
    Ok(())
}

/// Inverse of [`pack_quat`]: wxyz, unit length.
fn unpack_quat(b: [u8; 4]) -> [f32; 4] {
    let mut comp = u32::from_le_bytes(b);
    let largest = (comp >> 30) as usize;
    let (mut q, mut sum) = ([0f32; 4], 0f32);
    // The last packed component sits in the low bits.
    for i in (0..4).rev().filter(|&i| i != largest) {
        let mag = (comp & 511) as f32 / 511.0 * std::f32::consts::FRAC_1_SQRT_2;
        q[i] = if comp & 512 != 0 { -mag } else { mag };
        sum += mag * mag;
        comp >>= 10;
    }
    q[largest] = (1.0 - sum).max(0.0).sqrt();
    [q[3], q[0], q[1], q[2]]
}

/// v4: 32-byte header, TOC of (compressed, raw) u64 lengths, then one Zstd frame per stream.
/// Returns the v3-shaped body (16-byte header + concatenated streams).
fn v4_body(file: &[u8]) -> Result<Vec<u8>> {
    ensure!(file.len() >= 32, "truncated v4 header");
    let n_streams = file[15] as usize;
    let toc = u32::from_le_bytes(file[16..20].try_into().unwrap()) as usize;
    let u64_at = |o: usize| -> Result<usize> { Ok(u64::from_le_bytes(file.get(o..o + 8).context("truncated v4 TOC")?.try_into().unwrap()) as usize) };
    let mut out = file[..16].to_vec();
    out[15] = 0;
    let mut at = toc + 16 * n_streams;
    for s in 0..n_streams {
        let (clen, raw) = (u64_at(toc + 16 * s)?, u64_at(toc + 16 * s + 8)?);
        let data = file.get(at..at + clen).context("truncated v4 stream")?;
        let d = zstd::bulk::decompress(data, raw)?;
        ensure!(d.len() == raw, "v4 stream {s}: {} bytes, TOC says {raw}", d.len());
        out.extend_from_slice(&d);
        at += clen;
    }
    Ok(out)
}

/// Read an SPZ (v2/v3 gzip, v4 Zstd streams) back into the trainer's raw parameterisation, splats
/// in file order. Lossy only by the writer's quantisation; alpha bytes 0/255 decode to a finite
/// logit (alpha 0.5/255 from each end). The antialiased flag sets `RenderMode::Mip`.
pub fn read_spz(path: &Path) -> Result<Gaussians> {
    let file = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let decode = || -> Result<Gaussians> {
        let buf = if file.starts_with(&[0x1f, 0x8b]) {
            let mut out = Vec::new();
            flate2::read::MultiGzDecoder::new(&file[..]).read_to_end(&mut out)?;
            out
        } else {
            v4_body(&file)?
        };
        decode_body(&buf)
    };
    decode().with_context(|| format!("{}: bad SPZ", path.display()))
}

fn decode_body(buf: &[u8]) -> Result<Gaussians> {
    ensure!(buf.len() >= 16, "truncated header");
    let u32_at = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
    ensure!(u32_at(0) == MAGIC, "not an SPZ (magic {:#x})", u32_at(0));
    let version = u32_at(4);
    ensure!((2..=4).contains(&version), "SPZ version {version} unsupported (2..4)");
    let n = u32_at(8) as usize;
    let (sh_degree, fb, flags) = (buf[12] as u32, buf[13], buf[14]);
    ensure!(sh_degree <= 3, "SH degree {sh_degree}");
    ensure!(fb < 24, "fractional bits {fb}");
    let mut g = Gaussians { sh_degree, antialiased: flags & 1 != 0, ..Default::default() };
    g.render_mode = if g.antialiased { RenderMode::Mip } else { RenderMode::Default };
    let k = g.sh_coeffs();
    let rot_bytes = if version == 2 { 3 } else { 4 };
    let want = 16 + n * (9 + 1 + 3 + 3 + rot_bytes + k * 3);
    ensure!(buf.len() == want, "{n} splats need {want} bytes, got {}", buf.len());
    let mut at = 16;
    let mut take = |len: usize| {
        let s = &buf[at..at + len];
        at += len;
        s
    };
    let fx = (1u32 << fb) as f32;
    g.pos = take(n * 9).chunks_exact(9).map(|p| std::array::from_fn(|j| {
        let b = &p[j * 3..j * 3 + 3];
        i32::from_le_bytes([b[0], b[1], b[2], if b[2] & 0x80 != 0 { 0xff } else { 0 }]) as f32 / fx
    })).collect();
    g.opacity = take(n).iter().map(|&b| {
        let a = (b as f32 / 255.0).clamp(0.5 / 255.0, 254.5 / 255.0);
        (a / (1.0 - a)).ln()
    }).collect();
    g.dc = take(n * 3).chunks_exact(3).map(|c| std::array::from_fn(|j| (c[j] as f32 / 255.0 - 0.5) / COLOR_SCALE)).collect();
    g.scale = take(n * 3).chunks_exact(3).map(|s| std::array::from_fn(|j| s[j] as f32 / LOG_SCALE_STEPS as f32 - 10.0)).collect();
    g.rot = take(n * rot_bytes).chunks_exact(rot_bytes).map(|r| {
        if version == 2 {
            // Upstream v2: xyz bytes, w >= 0 recovered from unit length.
            let v: [f32; 3] = std::array::from_fn(|j| r[j] as f32 / 127.5 - 1.0);
            [(1.0 - v.iter().map(|x| x * x).sum::<f32>()).max(0.0).sqrt(), v[0], v[1], v[2]]
        } else {
            unpack_quat(r.try_into().unwrap())
        }
    }).collect();
    g.rest = take(n * k * 3).iter().map(|&b| (b as f32 - 128.0) / 128.0).collect();
    Ok(g)
}

/// zlib(magic | u32 total | u32 count | u32 ids[count] | f32 payload[count * width]).
fn read_overflow_block(path: &Path, magic: &[u8; 4], total: usize, width: usize) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?).read_to_end(&mut raw)?;
    let u = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
    ensure!(raw.len() >= 12 && &raw[..4] == magic && u(4) as usize == total, "{}: not a {} block for {total} splats", path.display(), String::from_utf8_lossy(magic));
    let count = u(8) as usize;
    ensure!(count <= total && raw.len() == 12 + count * 4 * (1 + width), "{}: bad length", path.display());
    let ids: Vec<u32> = (0..count).map(|i| u(12 + 4 * i)).collect();
    ensure!(ids.windows(2).all(|w| w[0] < w[1]) && ids.last().is_none_or(|&i| (i as usize) < total), "{}: bad indices", path.display());
    let values = raw[12 + 4 * count..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    Ok((ids, values))
}

/// Apply `spz-overflow-v1` sidecars in the viewer's order (the exported viewer overflow decoder): DCP1
/// patches core DC, OVF1 splats are appended after the core in their exported order.
fn apply_overflow(g: &mut Gaussians, meta: &Value, dir: &Path) -> Result<()> {
    let format = meta["format"].as_str().unwrap_or("");
    ensure!(format == crate::overflow::FORMAT, "unsupported splat_overflow format {format:?}");
    let count = |key: &str| meta[key].as_u64().map(|v| v as usize).with_context(|| format!("splat_overflow lacks {key}"));
    let (total, core) = (count("count")?, count("core_count")?);
    ensure!(g.len() == core, "core has {} splats, scene.json says {core}", g.len());
    ensure!(meta["sh_degree"].as_u64().is_none_or(|d| d == g.sh_degree as u64), "splat_overflow SH degree disagrees with the core");
    let k = g.sh_coeffs() * 3;
    let file = |key: &str| meta[key].as_str().map(|f| dir.join(f));
    let (ovf_ids, ovf) = match file("positions") {
        Some(p) => read_overflow_block(&p, b"OVF1", total, 14 + k)?,
        None => Default::default(),
    };
    let (dcp_ids, dcp) = match file("dc") {
        Some(p) => read_overflow_block(&p, b"DCP1", total, 3)?,
        None => Default::default(),
    };
    ensure!(ovf_ids.len() == count("overflow_count")? && dcp_ids.len() == count("dc_patch_count")?, "splat_overflow: record counts don't match scene.json");
    ensure!(core + ovf_ids.len() == total, "core {core} + overflow {} != count {total}", ovf_ids.len());
    // DC patch: exported index -> core index (minus the overflow splats before it).
    let mut o = 0;
    for (r, &id) in dcp_ids.iter().enumerate() {
        while o < ovf_ids.len() && ovf_ids[o] < id { o += 1; }
        ensure!(ovf_ids.get(o) != Some(&id), "DCP1 patches overflow splat {id}");
        g.dc[id as usize - o] = std::array::from_fn(|c| dcp[r * 3 + c]);
    }
    let m = ovf_ids.len();
    let v = |base: usize, j: usize, w: usize, c: usize| ovf[base * m + j * w + c];
    for j in 0..m {
        g.pos.push(std::array::from_fn(|c| v(0, j, 3, c)));
        g.scale.push(std::array::from_fn(|c| v(3, j, 3, c)));
        g.rot.push(std::array::from_fn(|c| v(6, j, 4, c)));
        g.opacity.push(v(10, j, 1, 0));
        g.dc.push(std::array::from_fn(|c| v(11, j, 3, c)));
    }
    g.rest.extend_from_slice(&ovf[14 * m..]);
    Ok(())
}

/// The splats of a web scene package `<dir>/scene.json`, in the viewer's splat order: its `splat`
/// file (SPZ or PLY) plus any `spz-overflow-v1` sidecars. scene.json's `render_mode` wins over the
/// SPZ antialiased flag (it also distinguishes surfel).
pub fn read_package_splats(package_dir: &Path) -> Result<Gaussians> {
    let meta_path = package_dir.join("scene.json");
    let meta: Value = serde_json::from_slice(&std::fs::read(&meta_path).with_context(|| format!("reading {}", meta_path.display()))?)
        .with_context(|| format!("parsing {}", meta_path.display()))?;
    let splat = meta["splat"].as_str().with_context(|| format!("{}: no splat file", meta_path.display()))?;
    let path = package_dir.join(splat);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let mut g = match ext.as_str() {
        "spz" => read_spz(&path)?,
        "ply" => crate::ply::read_ply(&path)?,
        _ => bail!("{}: can't read splat format {ext:?}", path.display()),
    };
    if let Some(ovf) = meta.get("splat_overflow").filter(|v| !v.is_null()) {
        apply_overflow(&mut g, ovf, package_dir).with_context(|| format!("{}: splat_overflow", package_dir.display()))?;
    }
    if let Some(mode) = meta["render_mode"].as_str().and_then(crate::ply::render_mode_comment) {
        g.render_mode = mode;
        g.antialiased = mode == RenderMode::Mip;
    }
    Ok(g)
}

