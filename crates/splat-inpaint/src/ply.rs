//! Byte-level binary little-endian PLY (one `vertex` element), so the output keeps the
//! input's header and its old records verbatim: only the vertex count changes and new
//! records are appended in the same property layout.

use std::path::Path;

use anyhow::{Context, bail, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            other => bail!("unsupported PLY property type {other}"),
        })
    }

    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Prop {
    pub name: String,
    ty: Ty,
    offset: usize,
}

pub struct Ply {
    /// Header text up to and including `end_header\n`.
    header: String,
    pub props: Vec<Prop>,
    pub stride: usize,
    pub n: usize,
    /// `n * stride` bytes of vertex records.
    pub data: Vec<u8>,
}

impl Ply {
    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let end = b"end_header\n";
        let head_len = bytes
            .windows(end.len())
            .position(|w| w == end)
            .map(|p| p + end.len())
            .context("PLY: no end_header")?;
        let header = std::str::from_utf8(&bytes[..head_len]).context("PLY header is not UTF-8")?.to_string();
        let (mut props, mut n, mut elements, mut in_vertex, mut binary_le) = (vec![], None, vec![], false, false);
        let mut offset = 0;
        for line in header.lines() {
            let p: Vec<&str> = line.split_whitespace().collect();
            match p.as_slice() {
                ["format", "binary_little_endian", _] => binary_le = true,
                ["element", name, count] => {
                    elements.push(name.to_string());
                    in_vertex = *name == "vertex";
                    if in_vertex {
                        n = Some(count.parse::<usize>().context("PLY vertex count")?);
                    }
                }
                ["property", "list", ..] if in_vertex => bail!("PLY: list properties are unsupported"),
                ["property", ty, name] if in_vertex => {
                    let ty = Ty::parse(ty)?;
                    props.push(Prop { name: name.to_string(), ty, offset });
                    offset += ty.size();
                }
                _ => {}
            }
        }
        ensure!(binary_le, "binary little-endian PLY only");
        ensure!(
            elements == ["vertex"],
            "plain SH PLY only (elements {elements:?}; neural-appearance PLYs are unsupported)"
        );
        let n = n.context("PLY: no vertex element")?;
        let stride = offset;
        ensure!(bytes.len() >= head_len + n * stride, "PLY truncated: {} vertices of {stride} bytes", n);
        let data = bytes[head_len..head_len + n * stride].to_vec();
        Ok(Self { header, props, stride, n, data })
    }

    pub fn has(&self, name: &str) -> bool {
        self.props.iter().any(|p| p.name == name)
    }

    /// Number of `f_rest_*` properties.
    pub fn rest_count(&self) -> usize {
        self.props.iter().filter(|p| p.name.starts_with("f_rest_")).count()
    }

    /// Records `i` for which `keep[i]`, in order, followed by `extra` (already encoded).
    pub fn write_filtered(&self, path: &Path, keep: &[bool], extra: &[u8]) -> anyhow::Result<usize> {
        ensure!(keep.len() == self.n, "keep mask has {} entries for {} vertices", keep.len(), self.n);
        ensure!(extra.len().is_multiple_of(self.stride), "extra records are not whole vertices");
        let kept = keep.iter().filter(|&&k| k).count();
        let total = kept + extra.len() / self.stride;
        let mut header = String::with_capacity(self.header.len() + 8);
        let mut replaced = false;
        for line in self.header.split_inclusive('\n') {
            if !replaced && line.starts_with("element vertex ") {
                header.push_str(&format!("element vertex {total}\n"));
                replaced = true;
            } else {
                header.push_str(line);
            }
        }
        let mut out = Vec::with_capacity(header.len() + total * self.stride);
        out.extend_from_slice(header.as_bytes());
        for (i, rec) in self.data.chunks_exact(self.stride).enumerate() {
            if keep[i] {
                out.extend_from_slice(rec);
            }
        }
        out.extend_from_slice(extra);
        let tmp = path.with_extension("ply.tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(&tmp, path)?;
        Ok(total)
    }

    /// Encode one record in this layout; properties `value` returns None for are zero.
    pub fn encode(&self, out: &mut Vec<u8>, value: impl FnMut(&str) -> Option<f32>) {
        let start = out.len();
        out.resize(start + self.stride, 0);
        self.write_props(&mut out[start..], value);
    }

    /// Overwrite properties of record `i` in place; those `value` returns None for stay verbatim.
    pub fn patch(&mut self, i: usize, value: impl FnMut(&str) -> Option<f32>) {
        let mut rec = self.data[i * self.stride..(i + 1) * self.stride].to_vec();
        self.write_props(&mut rec, value);
        self.data[i * self.stride..(i + 1) * self.stride].copy_from_slice(&rec);
    }

    fn write_props(&self, rec: &mut [u8], mut value: impl FnMut(&str) -> Option<f32>) {
        for p in &self.props {
            let Some(v) = value(&p.name) else { continue };
            let dst = &mut rec[p.offset..p.offset + p.ty.size()];
            match p.ty {
                Ty::I8 => dst.copy_from_slice(&(v.round() as i8).to_le_bytes()),
                Ty::U8 => dst.copy_from_slice(&(v.round().clamp(0.0, 255.0) as u8).to_le_bytes()),
                Ty::I16 => dst.copy_from_slice(&(v.round() as i16).to_le_bytes()),
                Ty::U16 => dst.copy_from_slice(&(v.round().max(0.0) as u16).to_le_bytes()),
                Ty::I32 => dst.copy_from_slice(&(v.round() as i32).to_le_bytes()),
                Ty::U32 => dst.copy_from_slice(&(v.round().max(0.0) as u32).to_le_bytes()),
                Ty::F32 => dst.copy_from_slice(&v.to_le_bytes()),
                Ty::F64 => dst.copy_from_slice(&(v as f64).to_le_bytes()),
            }
        }
    }
}
