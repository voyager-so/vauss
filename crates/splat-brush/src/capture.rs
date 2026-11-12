//! Capture-app cameras become ordinary COLMAP datasets. World coordinates and metric
//! scale are preserved; only OpenGL camera axes are changed to right/down/forward.
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
    time::Instant,
};

use crate::{capture_depth, util::*};
use anyhow::{Context, Result, bail, ensure};
use glam::{DMat3, DMat4, DQuat, DVec3};
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use splat_contracts::{Artifact, ArtifactKind, ExtractFrames, Status, ToolResult};
use splat_core::RunCtx;

const MANIFEST: &str = "capture.json";

#[derive(Clone)]
pub(crate) struct SourceFrame {
    pub image: PathBuf,
    pub camera: Frame,
    pub depth: Option<capture_depth::Source>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Frame {
    pub name: String,
    pub source_index: usize,
    pub source_name: String,
    pub width: u32,
    pub height: u32,
    pub model: String,
    pub params: Vec<f64>,
    /// Camera-to-world, OpenCV camera axes (+X right, +Y down, +Z forward).
    pub c2w: DMat4,
    pub has_depth: bool,
}

pub(crate) struct Capture {
    pub format: String,
    pub frames: Vec<SourceFrame>,
    pub init_ply: Option<PathBuf>,
    pub heldout_images: Option<Vec<PathBuf>>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Manifest {
    schema_version: u32,
    pub format: String,
    camera_convention: String,
    units: String,
    pub frames: Vec<Frame>,
    #[serde(default)]
    pub heldout_names: Option<Vec<String>>,
}

fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|n| n.is_finite())
        .with_context(|| format!("missing or invalid {key}"))
}

fn matrix(v: &Value) -> Result<DMat4> {
    let rows = v.as_array().context("transform_matrix must be 4x4")?;
    ensure!(
        rows.len() == 4
            && rows
                .iter()
                .all(|r| r.as_array().is_some_and(|a| a.len() == 4)),
        "transform_matrix must be 4x4"
    );
    let a = rows
        .iter()
        .flat_map(|r| r.as_array().unwrap())
        .map(|x| {
            x.as_f64()
                .filter(|x| x.is_finite())
                .context("non-finite transform")
        })
        .collect::<Result<Vec<_>>>()?;
    let m = DMat4::from_cols_slice(&a).transpose();
    validate_pose(m)?;
    Ok(m)
}

fn validate_pose(m: DMat4) -> Result<()> {
    ensure!(
        m.is_finite() && m.row(3).abs_diff_eq(glam::DVec4::W, 1e-5),
        "invalid homogeneous camera transform"
    );
    let r = DMat3::from_mat4(m);
    ensure!(
        (r.determinant() - 1.0).abs() < 0.02
            && (r.transpose() * r).abs_diff_eq(DMat3::IDENTITY, 0.02),
        "camera transform rotation must be orthonormal (no reflection or scale)"
    );
    Ok(())
}

fn opengl_to_cv(mut m: DMat4) -> DMat4 {
    m.y_axis *= -1.0;
    m.z_axis *= -1.0;
    m
}

fn quaternion_pose(q: &[f64], t: &[f64]) -> Result<DMat4> {
    ensure!(
        q.len() == 4 && t.len() == 3 && q.iter().chain(t).all(|n| n.is_finite()),
        "invalid quaternion/translation"
    );
    let q = DQuat::from_xyzw(q[0], q[1], q[2], q[3]);
    ensure!(
        (q.length() - 1.0).abs() < 0.02,
        "camera quaternion must have unit length"
    );
    Ok(DMat4::from_rotation_translation(
        q.normalize(),
        DVec3::from_slice(t),
    ))
}

/// No traversal or symlink escape from the capture root, including for referenced depth/PLY.
fn inside(root: &Path, base: &Path, name: &str) -> Result<PathBuf> {
    let p = Path::new(name);
    ensure!(
        !p.is_absolute()
            && p.components()
                .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)),
        "capture path must be relative without '..': {name}"
    );
    let p = base.join(p);
    let p = if p.extension().is_none() && !p.is_file() {
        p.with_extension("png")
    } else {
        p
    };
    let resolved = p
        .canonicalize()
        .with_context(|| format!("missing capture file {}", p.display()))?;
    ensure!(
        resolved.starts_with(root.canonicalize()?) && resolved.is_file(),
        "capture path escapes input directory: {name}"
    );
    Ok(resolved)
}

/// Unpack once per run. Reject traversal, symlinks, and archives with unbounded expansion.
fn input_root(run: &dyn RunCtx) -> Result<Option<PathBuf>> {
    let source = run.source();
    if source.is_dir() {
        return Ok(Some(source.to_path_buf()));
    }
    if !source
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| matches!(s.to_ascii_lowercase().as_str(), "zip" | "r3d"))
    {
        return Ok(None);
    }
    let dest = run.root().join("capture_input");
    if dest.join(".complete").is_file() {
        return Ok(Some(dest));
    }
    if dest.exists() {
        fs::remove_dir_all(&dest)?;
    }
    fs::create_dir_all(&dest)?;
    let mut zip = zip::ZipArchive::new(fs::File::open(source)?)
        .context("capture input must be a ZIP archive")?;
    let mut bytes = 0u64;
    ensure!(zip.len() <= 100_000, "too many capture archive entries");
    for i in 0..zip.len() {
        let mut file = zip.by_index_with_options(
            i,
            zip::read::ZipReadOptions::new().enforce_header_consistency(true),
        )?;
        let p = file.enclosed_name().context("unsafe capture ZIP path")?;
        ensure!(
            !file.is_symlink(),
            "symlinks are not allowed in capture ZIPs"
        );
        bytes = bytes
            .checked_add(file.size())
            .context("capture archive size overflow")?;
        ensure!(
            bytes <= 16 * 1024 * 1024 * 1024 && file.size() <= 2 * 1024 * 1024 * 1024,
            "capture archive is too large"
        );
        let p = dest.join(p);
        if file.is_dir() {
            fs::create_dir_all(&p)?;
        } else {
            fs::create_dir_all(p.parent().context("ZIP path has no parent")?)?;
            let mut out = fs::File::create(p)?;
            std::io::copy(&mut file, &mut out)?;
        }
    }
    fs::write(dest.join(".complete"), b"1")?;
    Ok(Some(dest))
}

/// Detect a capture root or a single capture nested under an export's wrapper folder.
pub(crate) fn load(run: &dyn RunCtx) -> Result<Option<Capture>> {
    let Some(root) = input_root(run)? else {
        return Ok(None);
    };
    let mut dirs = vec![root.clone()];
    let mut at = 0;
    while at < dirs.len() {
        let dir = &dirs[at];
        if dir.strip_prefix(&root)?.components().count() < 3 {
            for e in fs::read_dir(dir)? {
                let e = e?;
                if e.file_type()?.is_dir()
                    && !e.file_name().to_string_lossy().starts_with('.')
                    && e.file_name() != "__MACOSX"
                {
                    dirs.push(e.path());
                }
            }
        }
        at += 1;
    }
    dirs.sort();
    let mut captures = vec![];
    for dir in dirs {
        let record_meta = if let Some(meta) = ["metadata", "metadata.json"]
            .iter()
            .map(|s| dir.join(s))
            .find(|p| p.is_file())
        {
            let v: Value = read_json(&meta)?;
            if v["poses"].is_array() && v["K"].is_array() {
                Some(v)
            } else {
                None
            }
        } else {
            None
        };
        let capture = if let Some(v) = record_meta {
            Some(record3d(&root, &dir, &v)?)
        } else if dir.join("keyframes").is_dir() {
            Some(polycam(&root, &dir.join("keyframes"))?)
        } else if dir.join("corrected_cameras").is_dir()
            || (dir.join("cameras").is_dir() && dir.join("images").is_dir())
            || dir.join("cameras.json").is_file()
        {
            Some(polycam(&root, &dir)?)
        } else if dir.join("transforms.json").is_file() {
            Some(nerfstudio(&root, &dir)?)
        } else if dir.join("lowres_wide.traj").is_file() {
            Some(arkitscenes(&root, &dir)?)
        } else {
            None
        };
        if let Some(c) = capture {
            // keyframes is a subdirectory of the already detected Polycam capture.
            if captures
                .iter()
                .any(|prev: &Capture| capture_first(prev) == capture_first(&c))
            {
                continue;
            }
            captures.push(c);
        }
    }
    ensure!(
        captures.len() <= 1,
        "multiple captures in input: pass one capture directory or ZIP at a time"
    );
    let mut capture = captures.pop();
    if let Some(c) = &mut capture {
        ensure!(
            c.frames.len() >= 2,
            "capture needs at least two calibrated frames"
        );
        // Product frame names already identify decoded video instants. Keep them when
        // attaching exported calibration to those pixels, including for paired evaluation.
        let original_indices: Option<Vec<usize>> = c
            .frames
            .iter()
            .map(|f| {
                let digits = f
                    .camera
                    .source_name
                    .strip_prefix("frame_")?
                    .strip_suffix(".jpg")?;
                (digits.len() >= 6 && digits.bytes().all(|b| b.is_ascii_digit()))
                    .then(|| digits.parse().ok())
                    .flatten()
            })
            .collect();
        if let Some(indices) = original_indices {
            let unique: std::collections::BTreeSet<_> = indices.iter().copied().collect();
            ensure!(
                unique.len() == indices.len(),
                "duplicate capture frame names"
            );
            for (f, index) in c.frames.iter_mut().zip(indices) {
                f.camera.name = f.camera.source_name.clone();
                f.camera.source_index = index;
            }
        }
        for f in &mut c.frames {
            validate_pose(f.camera.c2w)?;
            ensure!(
                f.camera.width > 0
                    && f.camera.height > 0
                    && f.camera.params.iter().all(|p| p.is_finite())
                    && f.camera.params[0] > 0.0
                    && f.camera.params[1] > 0.0,
                "invalid capture intrinsics for {}",
                f.image.display()
            );
            if let Some(depth) = &mut f.depth {
                for path in std::iter::once(&mut depth.path).chain(depth.confidence.iter_mut()) {
                    let resolved = path.canonicalize()?;
                    ensure!(
                        resolved.starts_with(root.canonicalize()?) && resolved.is_file(),
                        "sensor depth/confidence path escapes input directory"
                    );
                    *path = resolved;
                }
            }
        }
    } else if !run.source().is_dir() {
        bail!(
            "capture archive has no supported cameras: expected Record3D metadata, Polycam cameras, transforms.json or lowres_wide.traj"
        );
    }
    Ok(capture)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opengl_camera_matches_colmap_reprojection() {
        // Non-identity pose and an off-centre pixel catch transposes, inverse
        // confusion, quaternion order and either missing axis sign conversion.
        let gl = DMat4::from_rotation_translation(
            DQuat::from_rotation_y(0.43) * DQuat::from_rotation_z(-0.17),
            DVec3::new(1.2, -0.6, 2.3),
        );
        let cv = opengl_to_cv(gl);
        let world = gl.transform_point3(DVec3::new(0.7, 0.4, -3.2));
        let expected = cv.inverse().transform_point3(world);
        assert!(expected.abs_diff_eq(DVec3::new(0.7, -0.4, 3.2), 1e-10));
        let q = DQuat::from_mat3(&DMat3::from_mat4(cv.inverse()));
        let rotation = splat_eval::Model::rotation([q.w, q.x, q.y, q.z]);
        let t = cv.inverse().w_axis.truncate().to_array();
        let p: [f64; 3] = std::array::from_fn(|i| {
            rotation[i][0] * world.x + rotation[i][1] * world.y + rotation[i][2] * world.z + t[i]
        });
        let uv = [813.0 * p[0] / p[2] + 317.0, 827.0 * p[1] / p[2] + 243.0];
        assert!((uv[0] - (813.0 * 0.7 / 3.2 + 317.0)).abs() < 1e-8);
        assert!((uv[1] - (827.0 * (-0.4) / 3.2 + 243.0)).abs() < 1e-8);
    }

    #[test]
    fn record3d_intrinsics_and_scalar_last_quaternion() -> Result<()> {
        let dir = tempfile::tempdir()?;
        fs::create_dir(dir.path().join("rgb"))?;
        image::RgbImage::new(640, 480).save(dir.path().join("rgb/0.jpg"))?;
        let q = DQuat::from_rotation_y(0.35);
        let v = json!({"w":640,"h":480,"K":[811,0,0,0,823,0,319,239,1],"poses":[[q.x,q.y,q.z,q.w,1,2,3]]});
        let c = record3d(dir.path(), dir.path(), &v)?;
        assert_eq!(c.frames[0].camera.params, vec![811., 823., 319., 239.]);
        let p = c.frames[0]
            .camera
            .c2w
            .transform_point3(DVec3::new(0., 0., 2.));
        let expected = DVec3::new(1., 2., 3.) + q * DVec3::new(0., 0., -2.);
        assert!(p.abs_diff_eq(expected, 1e-9));
        Ok(())
    }

    #[test]
    fn rejects_reflections_scaled_poses_and_path_escape() -> Result<()> {
        assert!(validate_pose(DMat4::from_scale(DVec3::splat(2.))).is_err());
        assert!(validate_pose(DMat4::from_scale(DVec3::new(1., -1., 1.))).is_err());
        let dir = tempfile::tempdir()?;
        assert!(inside(dir.path(), dir.path(), "../outside.jpg").is_err());
        assert!(inside(dir.path(), dir.path(), "/tmp/outside.jpg").is_err());
        Ok(())
    }
}

fn capture_first(c: &Capture) -> Option<&PathBuf> {
    c.frames.first().map(|f| &f.image)
}

fn frame(
    image: PathBuf,
    index: usize,
    w: u32,
    h: u32,
    model: String,
    params: Vec<f64>,
    c2w: DMat4,
    depth: Option<capture_depth::Source>,
) -> SourceFrame {
    SourceFrame {
        camera: Frame {
            name: format!("frame_{index:06}.jpg"),
            source_index: index,
            source_name: image
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            width: w,
            height: h,
            model,
            params,
            c2w,
            has_depth: depth.is_some(),
        },
        image,
        depth,
    }
}

fn record3d(root: &Path, dir: &Path, v: &Value) -> Result<Capture> {
    let rgb = ["rgb", "rgbd", "images"]
        .iter()
        .map(|s| dir.join(s))
        .find(|p| p.is_dir())
        .context("Record3D export has no rgb/rgbd/images directory")?;
    let poses = v["poses"].as_array().context("Record3D poses missing")?;
    let k = v["K"].as_array().context("Record3D K missing")?;
    ensure!(
        k.len() == 9,
        "Record3D K must contain 9 column-major values"
    );
    let k = k
        .iter()
        .map(|x| x.as_f64().context("invalid Record3D K"))
        .collect::<Result<Vec<_>>>()?;
    let (w, h) = (number(v, "w")? as u32, number(v, "h")? as u32);
    let mut images = list_images(&rgb)?;
    images.sort_by_key(|p| {
        p.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    });
    let mut frames = vec![];
    for (ordinal, image) in images.into_iter().enumerate() {
        let index = image
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(ordinal);
        let pose = poses
            .get(index)
            .and_then(Value::as_array)
            .with_context(|| format!("Record3D has no pose for frame {index}"))?;
        let p = pose
            .iter()
            .map(|p| p.as_f64().context("invalid Record3D pose"))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            p.len() == 7,
            "Record3D poses must be [qx,qy,qz,qw,tx,ty,tz]"
        );
        let image = inside(
            root,
            &rgb,
            image
                .file_name()
                .unwrap()
                .to_str()
                .context("non-UTF8 frame name")?,
        )?;
        let stem = image.file_stem().unwrap().to_string_lossy();
        let depth = [
            rgb.join(format!("{stem}.depth")),
            dir.join("depth").join(format!("{stem}.exr")),
            dir.join("depth").join(format!("{stem}.png")),
            rgb.join(format!("{stem}.exr")),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .map(|path| capture_depth::Source {
            path,
            confidence: Some(rgb.join(format!("{stem}.conf"))).filter(|p| p.is_file()),
            raw_size: Some((
                v["dw"].as_u64().unwrap_or(w as u64) as u32,
                v["dh"].as_u64().unwrap_or(h as u64) as u32,
            )),
            meters_per_unit: 0.001,
        });
        frames.push(frame(
            image,
            index,
            w,
            h,
            "PINHOLE".into(),
            vec![k[0], k[4], k[6], k[7]],
            opengl_to_cv(quaternion_pose(&p[..4], &p[4..])?),
            depth,
        ));
    }
    Ok(Capture {
        format: "record3d".into(),
        frames,
        init_ply: None,
        heldout_images: None,
    })
}

fn nerfstudio(root: &Path, dir: &Path) -> Result<Capture> {
    let v: Value = read_json(&dir.join("transforms.json"))?;
    let frames = v["frames"]
        .as_array()
        .context("transforms.json has no frames")?;
    let mut out = vec![];
    for (i, f) in frames.iter().enumerate() {
        let get = |key: &str| f[key].as_f64().or_else(|| v[key].as_f64());
        let image = inside(
            root,
            dir,
            f["file_path"].as_str().context("frame has no file_path")?,
        )?;
        let dims = image::image_dimensions(&image)?;
        let (w, h) = (
            get("w").unwrap_or(dims.0 as f64) as u32,
            get("h").unwrap_or(dims.1 as f64) as u32,
        );
        let fx = get("fl_x")
            .or_else(|| get("camera_angle_x").map(|a| 0.5 * w as f64 / (0.5 * a).tan()))
            .context("frame has no fl_x or camera_angle_x")?;
        let fy = get("fl_y")
            .or_else(|| get("camera_angle_y").map(|a| 0.5 * h as f64 / (0.5 * a).tan()))
            .unwrap_or(fx);
        let mut params = vec![
            fx,
            fy,
            get("cx").unwrap_or(w as f64 * 0.5),
            get("cy").unwrap_or(h as f64 * 0.5),
        ];
        let inferred_model = if ["k1", "k2", "k3", "k4", "p1", "p2"]
            .iter()
            .any(|k| get(k).unwrap_or(0.0) != 0.0)
        {
            "OPENCV"
        } else {
            "PINHOLE"
        };
        let camera_model = f["camera_model"]
            .as_str()
            .or(v["camera_model"].as_str())
            .unwrap_or(inferred_model)
            .to_ascii_uppercase();
        let model = match camera_model.as_str() {
            "PINHOLE" | "PERSPECTIVE" => "PINHOLE",
            "OPENCV" => {
                ensure!(
                    get("k4").unwrap_or(0.0) == 0.0,
                    "Nerfstudio OPENCV k4 is an eighth-order radial coefficient; export undistorted pixels or an explicit supported camera model"
                );
                params.extend(["k1", "k2", "p1", "p2"].map(|k| get(k).unwrap_or(0.0)));
                if get("k3").unwrap_or(0.0) != 0.0 {
                    params.extend([get("k3").unwrap(), 0.0, 0.0, 0.0]);
                    "FULL_OPENCV"
                } else {
                    "OPENCV"
                }
            }
            "FULL_OPENCV" => {
                params.extend(
                    ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"].map(|k| get(k).unwrap_or(0.0)),
                );
                "FULL_OPENCV"
            }
            "OPENCV_FISHEYE" => {
                params.extend(["k1", "k2", "k3", "k4"].map(|k| get(k).unwrap_or(0.0)));
                "OPENCV_FISHEYE"
            }
            other => bail!("unsupported capture camera_model {other}"),
        };
        let depth = f["depth_file_path"]
            .as_str()
            .map(|p| {
                Ok::<_, anyhow::Error>(capture_depth::Source {
                    path: inside(root, dir, p)?,
                    confidence: None,
                    raw_size: None,
                    meters_per_unit: get("depth_unit_scale_factor").unwrap_or(0.001),
                })
            })
            .transpose()?;
        out.push(frame(
            image,
            i,
            w,
            h,
            model.into(),
            params,
            opengl_to_cv(matrix(&f["transform_matrix"])?),
            depth,
        ));
    }
    let init_ply = v["ply_file_path"]
        .as_str()
        .map(|p| inside(root, dir, p))
        .transpose()?;
    Ok(Capture {
        format: "nerfstudio".into(),
        frames: out,
        init_ply,
        heldout_images: v["test_filenames"]
            .as_array()
            .map(|names| {
                names
                    .iter()
                    .map(|n| {
                        inside(
                            root,
                            dir,
                            n.as_str().context("test_filenames must contain paths")?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?,
    })
}

fn polycam(root: &Path, dir: &Path) -> Result<Capture> {
    let corrected = dir.join("corrected_cameras").is_dir();
    let cameras = dir.join(if corrected {
        "corrected_cameras"
    } else {
        "cameras"
    });
    let images = dir.join(if corrected {
        "corrected_images"
    } else {
        "images"
    });
    ensure!(
        images.is_dir(),
        "Polycam cameras require matching {} (do not mix corrected cameras with original images)",
        images.display()
    );
    let aggregate: Option<Value> = dir
        .join("cameras.json")
        .is_file()
        .then(|| read_json(&dir.join("cameras.json")))
        .transpose()?;
    let mut out = vec![];
    for (i, image) in list_images(&images)?.into_iter().enumerate() {
        let stem = image.file_stem().unwrap().to_string_lossy();
        let v: Value = if let Some(all) = &aggregate {
            let items = all["cameras"].as_array().or_else(|| all.as_array());
            if let Some(items) = items {
                items
                    .iter()
                    .find(|v| {
                        ["file_path", "filename", "image_path"].iter().any(|k| {
                            v[k].as_str()
                                .is_some_and(|p| Path::new(p).file_stem() == image.file_stem())
                        })
                    })
                    .cloned()
                    .context("Polycam cameras.json has no matching image entry")?
            } else {
                all.get(stem.as_ref())
                    .cloned()
                    .context("Polycam cameras.json has no image stem entry")?
            }
        } else {
            read_json(&cameras.join(format!("{stem}.json")))?
        };
        let mut rows = vec![vec![0.0; 4]; 4];
        rows[3][3] = 1.0;
        for (r, row) in rows.iter_mut().take(3).enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = number(&v, &format!("t_{r}{c}"))?;
            }
        }
        let depth = [
            dir.join("depth").join(format!("{stem}.png")),
            dir.join("depths").join(format!("{stem}.png")),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .map(|path| capture_depth::Source {
            path,
            confidence: None,
            raw_size: None,
            meters_per_unit: 0.001,
        });
        let image = inside(
            root,
            &images,
            image
                .file_name()
                .unwrap()
                .to_str()
                .context("non-UTF8 image")?,
        )?;
        // Polycam's world axes need no rotation: only camera axes change. A global
        // axis permutation in Nerfstudio's utility is cosmetic and would also move depth points.
        out.push(frame(
            image,
            i,
            number(&v, "width")? as u32,
            number(&v, "height")? as u32,
            "PINHOLE".into(),
            vec![
                number(&v, "fx")?,
                number(&v, "fy")?,
                number(&v, "cx")?,
                number(&v, "cy")?,
            ],
            opengl_to_cv(matrix(&json!(rows))?),
            depth,
        ));
    }
    Ok(Capture {
        format: "polycam".into(),
        frames: out,
        init_ply: None,
        heldout_images: None,
    })
}

fn arkitscenes(root: &Path, dir: &Path) -> Result<Capture> {
    let mut poses = vec![];
    for line in fs::read_to_string(dir.join("lowres_wide.traj"))?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let p = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            p.len() == 7 && p.iter().all(|x| x.is_finite()),
            "invalid ARKitScenes trajectory row"
        );
        let w2c = DMat4::from_rotation_translation(
            DQuat::from_scaled_axis(DVec3::from_slice(&p[1..4])),
            DVec3::from_slice(&p[4..7]),
        );
        poses.push((p[0], w2c.inverse()));
    }
    let rgb = if dir.join("vga_wide").is_dir() {
        "vga_wide"
    } else {
        "lowres_wide"
    };
    let mut out = vec![];
    for (i, image) in list_images(&dir.join(rgb))?.into_iter().enumerate() {
        let stem = image
            .file_stem()
            .unwrap()
            .to_str()
            .context("non-UTF8 frame")?;
        let time = stem
            .rsplit('_')
            .next()
            .context("no ARKit timestamp")?
            .parse::<f64>()?;
        let &(pt, pose) = poses
            .iter()
            .min_by(|a, b| (a.0 - time).abs().total_cmp(&(b.0 - time).abs()))
            .context("ARKitScenes trajectory is empty")?;
        ensure!(
            (pt - time).abs() <= 0.02,
            "no ARKitScenes pose within 20 ms of {stem}"
        );
        let intr = dir
            .join(format!("{rgb}_intrinsics"))
            .join(format!("{stem}.pincam"));
        let p = fs::read_to_string(intr)?
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(p.len() == 6, "ARKitScenes pincam needs w h fx fy cx cy");
        let depth = Some(dir.join("lowres_depth").join(format!("{stem}.png")))
            .filter(|p| p.is_file())
            .map(|path| capture_depth::Source {
                path,
                confidence: None,
                raw_size: None,
                meters_per_unit: 0.001,
            });
        let image = inside(
            root,
            &dir.join(rgb),
            image.file_name().unwrap().to_str().unwrap(),
        )?;
        out.push(frame(
            image,
            i,
            p[0] as u32,
            p[1] as u32,
            "PINHOLE".into(),
            p[2..].to_vec(),
            pose,
            depth,
        ));
    }
    Ok(Capture {
        format: "arkitscenes".into(),
        frames: out,
        init_ply: None,
        heldout_images: None,
    })
}

pub(crate) fn extract(
    run: &dyn RunCtx,
    req: ExtractFrames,
    capture: Capture,
) -> Result<ToolResult> {
    let start = Instant::now();
    ensure!(
        req.rotate_deg == 0 && req.time_range_s.is_none(),
        "capture cameras are calibrated to their stored pixels: rotate_deg/time_range_s are unavailable"
    );
    ensure!(
        req.target_count >= 2 && req.max_edge > 0,
        "captures require target_count >= 2 and positive max_edge"
    );
    let (artifact, out) = new_artifact(
        run,
        ArtifactKind::Frames,
        json!({"input_kind":"capture", "capture_format":capture.format}),
    )?;
    fs::create_dir_all(out.join("images"))?;
    let n = capture.frames.len().min(req.target_count as usize);
    let indices: Vec<usize> = (0..n)
        .map(|i| i * (capture.frames.len() - 1) / (n - 1).max(1))
        .collect();
    let mut frames = vec![];
    let mut kept = vec![];
    for index in indices {
        let source = &capture.frames[index];
        let rgb = image::open(&source.image)?;
        let rgb = if rgb.width().max(rgb.height()) > req.max_edge {
            rgb.resize(req.max_edge, req.max_edge, FilterType::Lanczos3)
        } else {
            rgb
        }
        .to_rgb8();
        let mut f = source.camera.clone();
        let (sx, sy) = (
            rgb.width() as f64 / f.width as f64,
            rgb.height() as f64 / f.height as f64,
        );
        ensure!(
            (sx / sy - 1.0).abs() < 0.025,
            "capture RGB aspect ratio differs from calibration for {}",
            source.image.display()
        );
        for p in [0, 2] {
            f.params[p] *= sx;
        }
        for p in [1, 3] {
            f.params[p] *= sy;
        }
        f.width = rgb.width();
        f.height = rgb.height();
        let dest = out.join("images").join(&f.name);
        let source_dims = image::image_dimensions(&source.image)?;
        if source_dims == (f.width, f.height)
            && source
                .image
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("jpg") || s.eq_ignore_ascii_case("jpeg"))
        {
            // Recompressing calibrated JPEGs changes held-out pixels for no benefit.
            link_or_copy(&source.image, &dest)?;
        } else {
            splat_eval::images::write_jpeg_compact(&dest, &rgb, 90, None)?;
        }
        if let Some(depth) = &source.depth {
            capture_depth::write(&out, &f.name, depth, f.width, f.height)?;
        }
        kept.push(json!({"name":f.name,"source_index":f.source_index,"source_name":f.source_name}));
        frames.push(f);
    }
    if let Some(ply) = capture.init_ply {
        link_or_copy(&ply, &out.join("init.ply"))?;
    }
    let heldout_names = capture
        .heldout_images
        .as_ref()
        .map(|images| {
            images
                .iter()
                .map(|image| {
                    let source = capture
                        .frames
                        .iter()
                        .find(|f| &f.image == image)
                        .context("test_filenames contains a frame absent from the capture")?;
                    ensure!(
                        frames.iter().any(|f| f.name == source.camera.name),
                        "requested held-out frame was subsampled away"
                    );
                    Ok(source.camera.name.clone())
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    write_json(
        &out.join(MANIFEST),
        &Manifest {
            schema_version: 1,
            format: capture.format.clone(),
            camera_convention: "opencv_camera_to_world".into(),
            units: "source (metres for sensor captures)".into(),
            frames,
            heldout_names,
        },
    )?;
    let metrics = json!({"input_kind":"capture","capture_format":capture.format,"frame_count":kept.len(),"frames_decoded":capture.frames.len(),"sensor_depth_frames":kept.iter().filter(|k| out.join("depths").join(Path::new(k["name"].as_str().unwrap()).with_extension("png")).is_file()).count(),"max_edge":req.max_edge,"dropped_black":0,"dropped_duplicates":0});
    write_json(
        &out.join("selection.json"),
        &json!({"request":req,"metrics":metrics,"frames":kept}),
    )?;
    Ok(ToolResult {
        artifacts: vec![artifact],
        metrics: metrics.as_object().unwrap().clone().into_iter().collect(),
        wall_s: Some(start.elapsed().as_secs_f64()),
        ..result("extract_frames", Status::Succeeded)
    })
}

/// Restore calibrated cameras after an SfM attempt registered fewer than 90%.
pub(crate) fn reconstruct(
    run: &dyn RunCtx, frames_id: &str, artifact: Artifact, out: &Path,
    selection: &Value,
) -> Result<ToolResult> {
    let (_, source) = get_artifact(run, frames_id, ArtifactKind::Frames)?;
    let start = Instant::now();
    let manifest: Manifest = read_json(&source.join(MANIFEST))?;
    for dir in ["images", "depths", "depths_conf"] {
        if source.join(dir).is_dir() {
            fs::create_dir_all(out.join(dir))?;
            for entry in fs::read_dir(source.join(dir))? {
                let e = entry?;
                let dst = out.join(dir).join(e.file_name());
                // SfM already linked RGB from these source files. Copying over
                // that hardlink would truncate the original captured pixels.
                if !dst.exists() { link_or_copy(&e.path(), &dst)?; }
            }
        }
    }
    for name in ["init.ply", "depths.json", MANIFEST] {
        if source.join(name).is_file() {
            if !out.join(name).exists() { link_or_copy(&source.join(name), &out.join(name))?; }
        }
    }
    let sparse = out.join("sparse/0");
    fs::create_dir_all(&sparse)?;
    let mut cameras = BTreeMap::new();
    let mut images = vec![];
    for (i, f) in manifest.frames.iter().enumerate() {
        let id = u32::try_from(i + 1)?;
        cameras.insert(
            id,
            splat_eval::colmap::Camera {
                id,
                model: f.model.clone(),
                width: f.width as u64,
                height: f.height as u64,
                params: f.params.clone(),
            },
        );
        let w2c = f.c2w.inverse();
        let q = DQuat::from_mat3(&DMat3::from_mat4(w2c)).normalize();
        images.push(splat_eval::colmap::ImageRecord {
            id,
            name: f.name.clone(),
            camera_id: id,
            qvec: [q.w, q.x, q.y, q.z],
            tvec: w2c.w_axis.truncate().to_array(),
            points2d: vec![],
        });
    }
    splat_eval::colmap::write_cameras_bin(&sparse.join("cameras.bin"), &cameras)?;
    splat_eval::colmap::write_images_bin(&sparse.join("images.bin"), &images)?;
    let points =
        capture_depth::initial_points(&out, &manifest.frames, manifest.heldout_names.as_deref())?;
    let mut file = fs::File::create(sparse.join("points3D.bin"))?;
    file.write_all(&(points.len() as u64).to_le_bytes())?;
    for (i, (xyz, rgb)) in points.iter().enumerate() {
        file.write_all(&((i + 1) as u64).to_le_bytes())?;
        for x in xyz {
            file.write_all(&x.to_le_bytes())?;
        }
        file.write_all(rgb)?;
        file.write_all(&0f64.to_le_bytes())?;
        file.write_all(&0u64.to_le_bytes())?;
    }
    let mut health = json!({"n_input":manifest.frames.len(),"n_registered":manifest.frames.len(),"registered_fraction":1.0,"n_points3d":points.len(),"engine":"capture_import","capture_format":manifest.format,"sfm_skipped":false});
    health.as_object_mut().unwrap().extend(selection.as_object().unwrap().clone());
    write_json(&out.join("health.json"), &health)?;
    freeze_split(run, frames_id, &artifact, &source, &manifest)?;
    Ok(ToolResult {
        artifacts: vec![artifact],
        metrics: health.as_object().unwrap().clone().into_iter().collect(),
        wall_s: Some(start.elapsed().as_secs_f64()),
        warnings: vec![format!("SfM placed {} of {} frames (below 90%); using capture-app poses with training pose refinement.", selection["sfm_registered"], manifest.frames.len())],
        ..result("reconstruct_sfm", Status::Succeeded)
    })
}

pub(crate) fn freeze_split(
    run: &dyn RunCtx, frames_id: &str, artifact: &Artifact, source: &Path,
    manifest: &Manifest,
) -> Result<()> {
    if source.join("frozen_split.json").is_file() { return Ok(()); }
    if let Some(heldout) = manifest.heldout_names.clone() {
        let names: Vec<_> = manifest.frames.iter().map(|f| f.name.clone()).collect();
        let heldout_set: std::collections::BTreeSet<_> = heldout.iter().collect();
        ensure!(
            !heldout.is_empty()
                && heldout_set.len() == heldout.len()
                && heldout.len() < names.len(),
            "invalid capture held-out split"
        );
        let frozen = splat_core::split::FrozenSplit {
            frames_id: Some(frames_id.into()),
            sfm_id: artifact.id.clone(),
            mode: "capture_explicit".into(),
            interval: 8,
            train_names: names
                .iter()
                .filter(|n| !heldout_set.contains(n))
                .cloned()
                .collect(),
            image_sha256: names
                .iter()
                .map(|n| Ok((n.clone(), sha256_file(&source.join("images").join(n))?)))
                .collect::<Result<_>>()?,
            registered_names: names,
            heldout_names: heldout,
        };
        let (id, dir) = run.new_artifact_dir(ArtifactKind::Split)?;
        fs::create_dir_all(&dir)?;
        write_json(&dir.join("manifest.json"), &frozen)?;
        let split = splat_contracts::Artifact {
            id,
            kind: ArtifactKind::Split,
            path: rel(run, &dir),
            meta: BTreeMap::from([
                (
                    "sha256".into(),
                    json!(sha256_file(&dir.join("manifest.json"))?),
                ),
                ("sfm_id".into(), json!(artifact.id)),
                ("frames_id".into(), json!(frames_id)),
            ]),
        };
        run.register(split.clone())?;
        write_json(
            &source.join("frozen_split.json"),
            &json!({"split_id":split.id}),
        )?;
    }
    Ok(())
}
