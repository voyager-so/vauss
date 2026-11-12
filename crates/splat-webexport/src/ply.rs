//! 3DGS PLY reader (binary_little_endian, all-float properties; what Brush and Spirula export).

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

/// Gaussians in the trainer's raw parameterisation.
#[derive(Debug, Clone, Default)]
pub struct Gaussians {
    pub pos: Vec<[f32; 3]>,
    /// Log scales.
    pub scale: Vec<[f32; 3]>,
    /// Quaternion w, x, y, z (PLY rot_0..rot_3), not necessarily normalised.
    pub rot: Vec<[f32; 4]>,
    /// Opacity logit.
    pub opacity: Vec<f32>,
    /// SH degree-0 coefficients (f_dc_*).
    pub dc: Vec<[f32; 3]>,
    /// Higher-order SH, coefficient-major with RGB inner: [coef][channel], `sh_coeffs` per splat.
    pub rest: Vec<f32>,
    pub sh_degree: u32,
    /// `render_mode == Mip`.
    pub antialiased: bool,
    pub render_mode: RenderMode,
    /// Compact neural appearance (trainer PLY comment `Neural appearance: x5b-v1`), else None.
    pub neural: Option<Neural>,
}

/// Neural appearance: per-splat latents plus one shared view-dependent decoder (the trainer's
/// `brush_render::kernels::neural`; the web viewer evaluates it in the exported viewer decoder).
/// The base colour stays in `dc`, so SH-only paths show the view-independent colour.
#[derive(Debug, Clone, Default)]
pub struct Neural {
    /// PLY `f_latent_0..7`, one row per splat.
    pub latents: Vec<[f32; NEURAL_LATENTS]>,
    /// PLY element `neural_decoder` (property `w`): W1 16x32 | W2 16x16 | W3 3x16, row-major [out][in].
    pub decoder: Vec<f32>,
}

pub const NEURAL_LATENTS: usize = 8;
pub const NEURAL_DECODER_LEN: usize = 16 * 32 + 16 * 16 + 3 * 16;

impl Gaussians {
    pub fn len(&self) -> usize {
        self.pos.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }
    /// Higher-order coefficients per channel: 3, 8, 15 for degrees 1..3.
    pub fn sh_coeffs(&self) -> usize {
        ((self.sh_degree + 1) * (self.sh_degree + 1) - 1) as usize
    }
    pub fn alpha(&self, i: usize) -> f32 {
        1.0 / (1.0 + (-self.opacity[i]).exp())
    }

    /// Keep splats where `keep[i]`.
    pub fn retain(&mut self, keep: &[bool]) {
        let k = self.sh_coeffs() * 3;
        let mut j = 0;
        for i in 0..self.len() {
            if keep[i] {
                self.pos[j] = self.pos[i];
                self.scale[j] = self.scale[i];
                self.rot[j] = self.rot[i];
                self.opacity[j] = self.opacity[i];
                self.dc[j] = self.dc[i];
                if let Some(n) = &mut self.neural {
                    n.latents[j] = n.latents[i];
                }
                if k > 0 {
                    self.rest.copy_within(i * k..(i + 1) * k, j * k);
                }
                j += 1;
            }
        }
        self.pos.truncate(j);
        self.scale.truncate(j);
        self.rot.truncate(j);
        self.opacity.truncate(j);
        self.dc.truncate(j);
        self.rest.truncate(j * k);
        if let Some(n) = &mut self.neural {
            n.latents.truncate(j);
        }
    }
}

/// The trainer's render mode, from its PLY comment `SplatRenderMode: mip|default|surfel`
/// (splat-train-serde export). Untagged files render as `Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    #[default]
    Default,
    /// Anti-aliased: 0.1 px low-pass with opacity compensation (vs plain 0.3 px for `Default`).
    Mip,
    Surfel,
}

impl RenderMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Mip => "mip",
            Self::Surfel => "surfel",
        }
    }
}

pub(crate) fn render_mode_comment(m: &str) -> Option<RenderMode> {
    match m.to_ascii_lowercase().as_str() {
        "mip" => Some(RenderMode::Mip),
        "default" => Some(RenderMode::Default),
        "surfel" => Some(RenderMode::Surfel),
        _ => None,
    }
}

/// Render mode from the PLY header alone (the last `SplatRenderMode` comment wins, as in the
/// trainer's import).
pub fn read_render_mode(path: &Path) -> Result<RenderMode> {
    read_render_mode_from(path).with_context(|| format!("reading the header of {}", path.display()))
}

fn read_render_mode_from(path: &Path) -> Result<RenderMode> {
    let mut rd = BufReader::new(open_maybe_zstd(path)?);
    let (mut mode, mut line) = (RenderMode::Default, Vec::new());
    loop {
        line.clear();
        if rd.read_until(b'\n', &mut line)? == 0 {
            bail!("{}: no end_header", path.display());
        }
        let l = String::from_utf8_lossy(&line);
        let w: Vec<&str> = l.split_whitespace().collect();
        match w.as_slice() {
            ["end_header"] => return Ok(mode),
            ["comment", "SplatRenderMode:", m] => mode = render_mode_comment(m).unwrap_or(mode),
            _ => {}
        }
    }
}

/// `path` for reading; a zstd-compressed file (`.ply.zst`, sniffed by its magic) is decompressed as it streams.
/// zstd is the lossless transfer format for trained PLYs (~3.7x smaller at level 3): see docs/performance.md.
pub fn open_maybe_zstd(path: &Path) -> Result<Box<dyn Read + Send>> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut magic = [0u8; 4];
    let compressed = file.read_exact(&mut magic).is_ok() && magic == [0x28, 0xb5, 0x2f, 0xfd];
    std::io::Seek::rewind(&mut file)?;
    Ok(if compressed { Box::new(zstd::stream::read::Decoder::new(file)?) } else { Box::new(file) })
}

pub fn read_ply(path: &Path) -> Result<Gaussians> {
    read_ply_from(path).with_context(|| format!("reading {}", path.display()))
}

fn read_ply_from(path: &Path) -> Result<Gaussians> {
    let mut rd = BufReader::with_capacity(1 << 22, open_maybe_zstd(path)?);
    let (mut n, mut props, mut line) = (0usize, Vec::<String>::new(), String::new());
    let mut render_mode = RenderMode::Default;
    // Elements after `vertex`: only the neural decoder (one float `w` per weight) is understood.
    let (mut element, mut n_decoder, mut neural_comment) = (String::new(), None::<usize>, false);
    loop {
        line.clear();
        if rd.read_line(&mut line)? == 0 {
            bail!("{}: no end_header", path.display());
        }
        let l = line.trim();
        if l == "end_header" {
            break;
        }
        let w: Vec<&str> = l.split_whitespace().collect();
        match w.as_slice() {
            ["comment", "SplatRenderMode:", m] if m.eq_ignore_ascii_case("2dgs") => bail!("2DGS export is not supported"),
            ["comment", "SplatRenderMode:", m] => render_mode = render_mode_comment(m).unwrap_or(render_mode),
            ["comment", "Neural", "appearance:", v] => {
                ensure!(*v == "x5b-v1", "{}: unsupported neural appearance version {v}", path.display());
                neural_comment = true;
            }
            ["element", "vertex", c] => {
                element = "vertex".into();
                n = c.parse()?;
            }
            ["element", "neural_decoder", c] if element == "vertex" => {
                element = "neural_decoder".into();
                n_decoder = Some(c.parse()?);
            }
            ["element", other, ..] => bail!("{}: unexpected element {other}", path.display()),
            ["property", "float", "w"] if element == "neural_decoder" => {}
            ["property", ..] if element == "neural_decoder" => bail!("{}: neural_decoder must hold one float w", path.display()),
            ["property", "float", name] => props.push(name.to_string()),
            ["property", ..] => bail!("{}: only all-float32 3DGS PLYs are supported", path.display()),
            ["format", f, ..] if *f != "binary_little_endian" => bail!("{}: expected binary_little_endian", path.display()),
            _ => {}
        }
    }
    let col = |name: &str| props.iter().position(|p| p == name).with_context(|| format!("PLY lacks {name}"));
    let base = [
        col("x")?, col("y")?, col("z")?,
        col("scale_0")?, col("scale_1")?, col("scale_2")?,
        col("rot_0")?, col("rot_1")?, col("rot_2")?, col("rot_3")?,
        col("opacity")?, col("f_dc_0")?, col("f_dc_1")?, col("f_dc_2")?,
    ];
    let n_rest = props.iter().filter(|p| p.starts_with("f_rest_")).count();
    let per_channel = n_rest / 3;
    let sh_degree = match per_channel {
        0 => 0,
        3 => 1,
        8 => 2,
        15 => 3,
        24 => 4,
        _ => bail!("{}: {n_rest} f_rest_* properties is not a valid SH layout", path.display()),
    };
    let rest_idx: Vec<usize> = (0..n_rest).map(|i| col(&format!("f_rest_{i}"))).collect::<Result<_>>()?;
    let latent_idx: Option<Vec<usize>> = if n_decoder.is_some() || neural_comment {
        ensure!(n_decoder == Some(NEURAL_DECODER_LEN), "{}: neural PLY needs a {NEURAL_DECODER_LEN}-weight neural_decoder", path.display());
        Some((0..NEURAL_LATENTS).map(|i| col(&format!("f_latent_{i}"))).collect::<Result<_>>()?)
    } else {
        None
    };
    let mut g = Gaussians { sh_degree, antialiased: render_mode == RenderMode::Mip, render_mode, ..Default::default() };
    let mut latents = Vec::with_capacity(if latent_idx.is_some() { n } else { 0 });
    g.pos.reserve(n);
    g.rest.reserve(n * n_rest);
    let mut row = vec![0u8; props.len() * 4];
    for _ in 0..n {
        rd.read_exact(&mut row)?;
        let f = |i: usize| f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
        g.pos.push([f(base[0]), f(base[1]), f(base[2])]);
        g.scale.push([f(base[3]), f(base[4]), f(base[5])]);
        g.rot.push([f(base[6]), f(base[7]), f(base[8]), f(base[9])]);
        g.opacity.push(f(base[10]));
        g.dc.push([f(base[11]), f(base[12]), f(base[13])]);
        // PLY f_rest is channel-major (all R coefficients, then G, then B); store [coef][channel].
        for k in 0..per_channel {
            for c in 0..3 {
                g.rest.push(f(rest_idx[c * per_channel + k]));
            }
        }
        if let Some(li) = &latent_idx {
            latents.push(std::array::from_fn(|i| f(li[i])));
        }
    }
    if latent_idx.is_some() {
        let mut w = vec![0u8; NEURAL_DECODER_LEN * 4];
        rd.read_exact(&mut w).with_context(|| format!("{}: truncated neural_decoder", path.display()))?;
        let decoder = w.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        g.neural = Some(Neural { latents, decoder });
    }
    Ok(g)
}

/// Serialize the raw 3DGS parameterization for a format adapter. No axis or gamma transform.
/// Standard 3DGS property order (no normals), f_rest channel-major; the render mode rides in the
/// same `SplatRenderMode` comment [`read_ply`] parses. Neural appearance is not written (Spark and
/// splat-transform read this file). Atomic: written to `<path>.tmp`, then renamed.
pub fn write_ply(g: &Gaussians, path: &Path) -> Result<()> {
    use std::io::{BufWriter, Write};
    let (n, k) = (g.len(), g.sh_coeffs());
    ensure!(g.scale.len() == n && g.rot.len() == n && g.opacity.len() == n && g.dc.len() == n && g.rest.len() == n * k * 3, "write_ply: attribute lengths disagree");
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    let write = || -> Result<()> {
        let mut out = BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?);
        writeln!(out, "ply\nformat binary_little_endian 1.0")?;
        let mode = if g.antialiased { RenderMode::Mip } else { g.render_mode };
        if mode != RenderMode::Default { writeln!(out, "comment SplatRenderMode: {}", mode.as_str())?; }
        writeln!(out, "element vertex {n}")?;
        for name in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] { writeln!(out, "property float {name}")?; }
        for i in 0..k * 3 { writeln!(out, "property float f_rest_{i}")?; }
        for name in ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] { writeln!(out, "property float {name}")?; }
        writeln!(out, "end_header")?;
        let mut row = Vec::with_capacity((14 + k * 3) * 4);
        for i in 0..n {
            row.clear();
            let rest = &g.rest[i * k * 3..(i + 1) * k * 3];
            // [coef][channel] -> channel-major.
            let vals = g.pos[i].iter().chain(&g.dc[i]).copied()
                .chain((0..3).flat_map(|c| (0..k).map(move |j| rest[j * 3 + c])))
                .chain(std::iter::once(g.opacity[i])).chain(g.scale[i]).chain(g.rot[i]);
            for v in vals { row.extend_from_slice(&v.to_le_bytes()); }
            out.write_all(&row)?;
        }
        out.flush()?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path).with_context(|| format!("installing {}", path.display()))
}
