//! Sensor depth sidecars shared with the depth-loss lane: uint16 millimetres,
//! optical-axis z, 0 invalid. Resizing changes pixel sampling, never metric scale.
use crate::{capture::Frame, util::write_json};
use anyhow::{Context, Result, ensure};
use image::{ImageBuffer, Luma, imageops::FilterType};
use serde_json::json;
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub(crate) struct Source {
    pub path: PathBuf,
    pub confidence: Option<PathBuf>,
    pub raw_size: Option<(u32, u32)>,
    /// PNG units; EXR / Record3D float32 depth is always in metres.
    pub meters_per_unit: f64,
}

fn lzfse(path: &Path, expected: usize) -> Result<Vec<u8>> {
    ensure!(
        expected <= 128 * 1024 * 1024,
        "sensor depth frame is too large"
    );
    let mut decoder = lzfse_rust::LzfseRingDecoder::default();
    let mut bytes = vec![];
    decoder
        .reader(fs::File::open(path)?)
        .take(expected as u64 + 1)
        .read_to_end(&mut bytes)
        .context("invalid Record3D LZFSE data")?;
    ensure!(
        bytes.len() == expected,
        "Record3D depth/confidence dimensions disagree with decompressed data: expected {expected}, got {}",
        bytes.len()
    );
    Ok(bytes)
}

fn read(source: &Source) -> Result<(u32, u32, Vec<f64>)> {
    match source
        .path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "depth" => {
            let (w, h) = source.raw_size.context("Record3D depth has no size")?;
            let bytes = lzfse(&source.path, w as usize * h as usize * 4)?;
            Ok((
                w,
                h,
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
                    .collect(),
            ))
        }
        "exr" => {
            use exr::prelude::*;
            let image = read()
                .no_deep_data()
                .largest_resolution_level()
                .all_channels()
                .first_valid_layer()
                .all_attributes()
                .from_file(&source.path)?;
            let layer = image.layer_data;
            let channels = &layer.channel_data.list;
            let channel = ["Z", "depth", "Y", "R"]
                .iter()
                .find_map(|name| {
                    channels
                        .iter()
                        .find(|c| c.name.to_string().eq_ignore_ascii_case(name))
                })
                .or_else(|| (channels.len() == 1).then(|| &channels[0]))
                .context("EXR has no depth channel")?;
            let values = channel
                .sample_data
                .values_as_f32()
                .map(f64::from)
                .collect::<Vec<_>>();
            ensure!(
                values.len() == layer.size.0 * layer.size.1,
                "subsampled EXR depth channels are unsupported"
            );
            Ok((layer.size.0 as u32, layer.size.1 as u32, values))
        }
        "png" | "tif" | "tiff" => {
            ensure!(
                source.meters_per_unit.is_finite() && source.meters_per_unit > 0.0,
                "invalid depth_unit_scale_factor"
            );
            let image = image::open(&source.path)?;
            ensure!(
                matches!(image, image::DynamicImage::ImageLuma16(_)),
                "sensor depth PNG/TIFF must be single-channel uint16"
            );
            let d = image.to_luma16();
            Ok((
                d.width(),
                d.height(),
                d.as_raw()
                    .iter()
                    .map(|v| *v as f64 * source.meters_per_unit)
                    .collect(),
            ))
        }
        other => anyhow::bail!(
            "unsupported sensor depth extension {other}; use EXR, Record3D .depth or uint16 PNG"
        ),
    }
}

pub(crate) fn write(out: &Path, name: &str, source: &Source, w: u32, h: u32) -> Result<()> {
    let (dw, dh, values) = read(source)?;
    ensure!(
        dw > 0 && dh > 0 && (dw as f64 / dh as f64 / (w as f64 / h as f64) - 1.0).abs() < 0.025,
        "sensor depth and RGB must be registered with the same aspect ratio"
    );
    let bytes = values
        .iter()
        .map(|z| {
            let mm = z * 1000.0;
            if mm.is_finite() && (1.0..=u16::MAX as f64).contains(&mm) {
                mm.round() as u16
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let raw = ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(dw, dh, bytes)
        .context("invalid sensor depth shape")?;
    let depth = image::imageops::resize(&raw, w, h, FilterType::Nearest);
    fs::create_dir_all(out.join("depths"))?;
    let stem = Path::new(name).with_extension("png");
    depth.save(out.join("depths").join(&stem))?;
    if let Some(conf) = &source.confidence {
        let bytes = lzfse(conf, dw as usize * dh as usize)?
            .into_iter()
            .map(|x| match x {
                0 => 0,
                1 => 128,
                2 => 255,
                _ => 0,
            })
            .collect();
        let confidence =
            image::GrayImage::from_raw(dw, dh, bytes).context("invalid confidence shape")?;
        fs::create_dir_all(out.join("depths_conf"))?;
        image::imageops::resize(&confidence, w, h, FilterType::Nearest)
            .save(out.join("depths_conf").join(&stem))?;
    }
    write_json(
        &out.join("depths.json"),
        &json!({"kind":"metric","source":"lidar","aligned_to_sfm":false,
        "units":"millimetres","depth_convention":"optical_axis_z","invalid_value":0,
        "scale_note":"Camera world coordinates preserved in metres; depth scale factor 1.0; RGB/depth nearest-neighbour pixel resize only."}),
    )?;
    Ok(())
}

/// A bounded, coloured surface seed from training frames' sensor depth. Held-out
/// frame indices (default every eighth) do not contribute their RGB or depth.
pub(crate) fn initial_points(
    out: &Path,
    frames: &[Frame],
    heldout: Option<&[String]>,
) -> Result<Vec<([f64; 3], [u8; 3])>> {
    let mut points = vec![];
    let mut voxels = BTreeSet::new();
    let depth_frames = frames.iter().filter(|f| f.has_depth).count();
    let budget = 100_000usize;
    for (i, f) in frames.iter().enumerate() {
        if !f.has_depth
            || heldout.map_or(i % 8 == 0, |names| names.contains(&f.name))
            || f.model != "PINHOLE"
        {
            continue;
        }
        let stem = Path::new(&f.name).with_extension("png");
        let d = image::open(out.join("depths").join(&stem))?.to_luma16();
        let conf = out.join("depths_conf").join(&stem);
        let conf = conf
            .is_file()
            .then(|| image::open(conf).map(|i| i.to_luma8()))
            .transpose()?;
        let rgb = image::open(out.join("images").join(&f.name))?.to_rgb8();
        let step = ((f.width as f64 * f.height as f64 * depth_frames as f64 / budget as f64)
            .sqrt()
            .ceil() as usize)
            .max(1);
        for y in (0..f.height).step_by(step) {
            for x in (0..f.width).step_by(step) {
                let z = d.get_pixel(x, y)[0] as f64 * 0.001;
                if z <= 0.05 || z > 50.0 || conf.as_ref().is_some_and(|c| c.get_pixel(x, y)[0] == 0)
                {
                    continue;
                }
                let p = glam::DVec3::new(
                    (x as f64 - f.params[2]) * z / f.params[0],
                    (y as f64 - f.params[3]) * z / f.params[1],
                    z,
                );
                let p = f.c2w.transform_point3(p);
                let voxel = p.to_array().map(|x| (x / 0.02).round() as i64);
                if voxels.insert(voxel) {
                    points.push((p.to_array(), rgb.get_pixel(x, y).0));
                }
                if points.len() >= budget {
                    return Ok(points);
                }
            }
        }
    }
    Ok(points)
}
