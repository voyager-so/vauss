//! Spirula accepts filename-based splits, not explicit image lists. Rename a private dataset view
//! while retaining camera IDs, poses, observations, sparse points, image bytes, and masks.
use crate::util::write;
use anyhow::{Context, Result, ensure};
use splat_core::split::ResolvedSplit;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

fn link(source: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = target.parent() { fs::create_dir_all(parent)?; }
    fs::hard_link(source, target).or_else(|_| fs::copy(source, target).map(|_| ()))
        .with_context(|| format!("Link split view {}", source.display()))
}

fn u64_le(input: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn rewrite_binary(source: &Path, target: &Path, aliases: &BTreeMap<String, String>) -> Result<()> {
    let mut input = BufReader::new(fs::File::open(source)?);
    let mut output = BufWriter::new(fs::File::create(target)?);
    let count = u64_le(&mut input)?;
    ensure!(count == aliases.len() as u64, "COLMAP membership changed while constructing split view");
    output.write_all(&count.to_le_bytes())?;
    for _ in 0..count {
        // image_id + qvec + tvec + camera_id; keep the floating-point bit patterns unchanged.
        let mut header = [0; 64];
        input.read_exact(&mut header)?;
        output.write_all(&header)?;
        let mut name = Vec::new();
        input.read_until(0, &mut name)?;
        ensure!(name.pop() == Some(0), "Unterminated COLMAP image name");
        let source_name = std::str::from_utf8(&name)?;
        let alias = aliases.get(source_name).with_context(|| format!("Unknown COLMAP image {source_name}"))?;
        output.write_all(alias.as_bytes())?;
        output.write_all(&[0])?;
        let points = u64_le(&mut input)?;
        output.write_all(&points.to_le_bytes())?;
        let bytes = points.checked_mul(24).context("COLMAP observation count overflow")?;
        ensure!(std::io::copy(&mut (&mut input).take(bytes), &mut output)? == bytes, "Truncated COLMAP observations");
    }
    ensure!(input.read(&mut [0])? == 0, "Trailing COLMAP image records");
    output.flush()?;
    Ok(())
}

fn rewrite_text(source: &Path, target: &Path, aliases: &BTreeMap<String, String>) -> Result<()> {
    let header = regex::Regex::new(r"^(\s*(?:\S+\s+){9})(.+?)\s*$")?;
    let input = BufReader::new(fs::File::open(source)?);
    let mut output = BufWriter::new(fs::File::create(target)?);
    let mut image_line = true;
    let mut count = 0;
    for line in input.lines() {
        let line = line?;
        if line.trim_start().starts_with('#') || (image_line && line.trim().is_empty()) {
            writeln!(output, "{line}")?;
            continue;
        }
        if image_line {
            let fields = header.captures(&line).context("Invalid COLMAP text image record")?;
            let alias = aliases.get(&fields[2]).with_context(|| format!("Unknown COLMAP image {}", &fields[2]))?;
            writeln!(output, "{}{alias}", &fields[1])?;
            count += 1;
        } else {
            writeln!(output, "{line}")?;
        }
        image_line = !image_line;
    }
    ensure!(image_line && count == aliases.len(), "Truncated COLMAP text model");
    output.flush()?;
    Ok(())
}

pub(crate) fn create(dataset: &Path, sparse_dir: &str, split: &ResolvedSplit, candidate: &Path) -> Result<PathBuf> {
    let out = candidate.join("training_data");
    let sparse = out.join("sparse/0");
    fs::create_dir_all(&sparse)?;
    fs::create_dir(out.join("images"))?;
    let heldout = split.heldout_names.iter().collect::<BTreeSet<_>>();
    let mut aliases = BTreeMap::new();
    for (index, name) in split.registered_names.iter().enumerate() {
        let suffix = Path::new(name).extension().and_then(|s| s.to_str()).context("Source image has no extension")?;
        let kind = if heldout.contains(name) { "eval" } else { "train" };
        let alias = format!("{kind}_{index:06}.{suffix}");
        link(&dataset.join("images").join(name), &out.join("images").join(&alias))?;
        for (original, renamed) in [
            (PathBuf::from(name), PathBuf::from(&alias)),
            (Path::new(name).with_extension("png"), Path::new(&alias).with_extension("png")),
            (PathBuf::from(format!("{name}.png")), PathBuf::from(format!("{alias}.png"))),
        ] {
            let mask = dataset.join("masks").join(original);
            let dest = out.join("masks").join(renamed);
            if mask.is_file() && !dest.exists() { link(&mask, &dest)?; }
        }
        aliases.insert(name.clone(), alias);
    }
    let source_sparse = dataset.join(sparse_dir);
    let binary = source_sparse.join("images.bin").exists();
    let extension = if binary { "bin" } else { "txt" };
    for name in ["cameras", "points3D"] {
        let name = format!("{name}.{extension}");
        link(&source_sparse.join(&name), &sparse.join(name))?;
    }
    let source = source_sparse.join(format!("images.{extension}"));
    let target = sparse.join(format!("images.{extension}"));
    if binary { rewrite_binary(&source, &target, &aliases)?; }
    else { rewrite_text(&source, &target, &aliases)?; }
    write(&out.join("source_names.json"), &aliases)?;
    write(&candidate.join("resolved_split.json"), split)?;
    Ok(out)
}
