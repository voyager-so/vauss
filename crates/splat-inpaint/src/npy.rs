//! Minimal `.npy` reader for 1-D integer index arrays (selection `selected_ids.npy`).

use std::path::Path;

use anyhow::{Context, bail, ensure};

pub fn read_indices(path: &Path) -> anyhow::Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY", "{} is not a .npy file", path.display());
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            ensure!(bytes.len() >= 12, "truncated .npy header");
            (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
        }
        v => bail!(".npy version {v} unsupported"),
    };
    ensure!(bytes.len() >= start + header_len, "truncated .npy header");
    let header = std::str::from_utf8(&bytes[start..start + header_len]).context(".npy header")?;
    let field = |key: &str| -> anyhow::Result<&str> {
        let i = header.find(&format!("'{key}'")).with_context(|| format!(".npy header has no {key}: {header}"))?;
        let rest = header[i + key.len() + 2..].trim_start().trim_start_matches(':').trim_start();
        Ok(rest)
    };
    let descr = field("descr")?;
    let descr = descr.trim_start_matches('\'');
    let descr = &descr[..descr.find('\'').context(".npy descr")?];
    // Fortran order only matters for >1-D arrays, which are rejected below.
    let _ = field("fortran_order")?;
    let shape = field("shape")?;
    let shape = &shape[1..shape.find(')').context(".npy shape")?];
    let dims: Vec<usize> = shape.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap_or(0)).collect();
    let count: usize = dims.iter().product();
    ensure!(dims.len() <= 1 || dims.iter().filter(|&&d| d > 1).count() <= 1, ".npy indices must be 1-D, got shape {dims:?}");
    let data = &bytes[start + header_len..];
    let (size, little) = match descr {
        "<u4" | "<i4" => (4, true),
        "<u8" | "<i8" => (8, true),
        "<u2" | "<i2" => (2, true),
        "|u1" | "|i1" => (1, true),
        d => bail!(".npy dtype {d} unsupported for indices"),
    };
    ensure!(little && data.len() >= count * size, ".npy data truncated");
    let signed = descr.contains('i');
    let mut out = Vec::with_capacity(count);
    for c in data.chunks_exact(size).take(count) {
        let v: i128 = match (size, signed) {
            (1, false) => c[0] as i128,
            (1, true) => c[0] as i8 as i128,
            (2, false) => u16::from_le_bytes([c[0], c[1]]) as i128,
            (2, true) => i16::from_le_bytes([c[0], c[1]]) as i128,
            (4, false) => u32::from_le_bytes(c.try_into()?) as i128,
            (4, true) => i32::from_le_bytes(c.try_into()?) as i128,
            (8, false) => u64::from_le_bytes(c.try_into()?) as i128,
            _ => i64::from_le_bytes(c.try_into()?) as i128,
        };
        ensure!((0..=u32::MAX as i128).contains(&v), "index {v} out of range");
        out.push(v as u32);
    }
    Ok(out)
}
