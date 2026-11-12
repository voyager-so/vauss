//! Learned camera fallback: Depth Anything 3 (DA3-BASE) camera poses and intrinsics in pure
//! Rust (Burn, wgpu), used as a seed for COLMAP triangulation + bundle adjustment.
//!
//! Only the pose path is ported (ViT-B backbone + camera decoder; see NOTICE). Weights are read
//! from the file named by `SPLAT_DA3_WEIGHTS`: `da3-base-cam-f16.safetensors` (just the tensors
//! this port loads, fp16; widened to f32 on load), or the upstream DA3-BASE `model.safetensors`.
//! [`learned_seed`] is the whole fallback: predict, seed, COLMAP triangulation + bundle
//! adjustment, then a COLMAP mapper pass. [`guard_fallback`] decides whether its model replaces
//! COLMAP's: only if it registers no fewer frames, fits no worse, is not far more weakly
//! connected and wins the held-out check.

mod colmap;
mod flash;
mod guard;
mod jpeg;
mod model;
mod preprocess;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, ensure};
use brush_cube::MainDevice;
use burn::tensor::Device;
use fs2::FileExt;

pub use model::Da3Pose;
pub use preprocess::{Preprocessed, preprocess};
pub use colmap::Cut;
pub use guard::{Clause, Guard, GuardStep, Kept, MAX_REPROJECTION_RATIO, MIN_CONDUCTANCE_RATIO, ModelSummary, guard_fallback};

/// One frame's predicted camera. Intrinsics are at the frame's original resolution.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LearnedCamera {
    pub name: String,
    pub w2c_rotation: [[f64; 3]; 3],
    pub w2c_translation: [f64; 3],
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// Raw network outputs, in frame order: the camera decoder input (`[views, 1536]`, row-major)
/// and the pose encoding (t xyz, quat xyzw, fov_h, fov_w; camera-to-world).
pub struct PoseOutput {
    pub cam_feat: Vec<f32>,
    pub pose_enc: Vec<[f32; 9]>,
}

/// Result of [`learned_seed`].
#[derive(Clone, Debug)]
pub struct LearnedSeed {
    /// The polished sparse model (`out_dir/polished`).
    pub model_dir: PathBuf,
    pub n_frames: usize,
    /// DA3 inference: device creation, weight load, forward and readback.
    pub inference_seconds: f64,
    /// The GPU memory pool's reserved bytes after inference: the allocator's high-water mark
    /// (excludes the driver/context overhead outside the pool).
    pub peak_gpu_bytes: Option<u64>,
    pub lock_wait_seconds: f64,
}

/// Which cameras to train on. Registered-frame count is not a quality measure (paf-museum: the
/// fallback placed 220 frames vs COLMAP's 104 and rendered 1.9 dB worse on shared held-out
/// frames), so learned cameras replace a trainable COLMAP model only on held-out render quality
/// ([`pick_by_heldout`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraSource {
    Colmap,
    Learned,
}

/// A model can be trained on when it places a quarter of the input and at least 20 frames, or
/// three quarters of the input for captures shorter than that (ETH3D pipes: 14/14 frames).
pub fn trainable(registered: usize, n_input: usize) -> bool {
    registered * 4 >= n_input && registered >= 20.min((3 * n_input).div_ceil(4))
}

/// Held-out check result: mean PSNR of each arm's short training run on the same held-out frames
/// (frames both models registered). Learned wins only if strictly better.
pub fn pick_by_heldout(colmap_psnr: f64, learned_psnr: f64) -> CameraSource {
    if learned_psnr > colmap_psnr { CameraSource::Learned } else { CameraSource::Colmap }
}

/// Learned-pose reconstruction of `frames_dir` against an existing COLMAP `database` (features
/// and matches): DA3 cameras seed `out_dir/seed`, then triangulation and bundle adjustment run
/// three times with tightening thresholds, followed by point and image filtering
/// (`out_dir/final`) and a COLMAP mapper pass continuing from that model (`out_dir/polished`).
/// With `gpu_lock`, an exclusive lock on that file is held around the DA3 inference only; the
/// COLMAP stages are CPU. Everything is written under `out_dir`; the caller's database is only
/// read (COLMAP works on copies there). `weights`: the DA3 weights file (`None`: `SPLAT_DA3_WEIGHTS`).
pub fn learned_seed(
    frames_dir: &Path,
    database: &Path,
    colmap: &Path,
    gpu_lock: Option<&Path>,
    weights: Option<&Path>,
    out_dir: &Path,
) -> anyhow::Result<LearnedSeed> {
    let frames = list_frames(frames_dir)?;
    let input = preprocess(&frames)?;

    let lock_start = Instant::now();
    let gpu_lock = gpu_lock
        .map(|path| -> anyhow::Result<File> {
            let file = OpenOptions::new().create(true).truncate(false).write(true).open(path)?;
            file.lock_exclusive()?;
            Ok(file)
        })
        .transpose()?;
    let lock_wait_seconds = if gpu_lock.is_some() { lock_start.elapsed().as_secs_f64() } else { 0.0 };
    let start = Instant::now();
    let device: Device = MainDevice::default().into();
    let out = run(&load_model_from(weights, &device)?, &input, &device)?;
    let peak_gpu_bytes = device.memory_pool_usage().map(|u| u.bytes_reserved);
    let inference_seconds = start.elapsed().as_secs_f64();
    drop(gpu_lock);
    let cams = cameras_from_pose_enc(&frame_names(&frames)?, &out.pose_enc, &input);

    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out_dir = &out_dir.canonicalize()?;
    let db = copy_database(database, &out_dir.join("database.db"))?;
    let (ids, width, height) = read_database(&db)?;
    let dir = |name: &str| out_dir.join(name);
    write_colmap_seed(&cams, &ids, width, height, &dir("seed"))?;

    let db = db.to_string_lossy().into_owned();
    let images = frames_dir.canonicalize()?.to_string_lossy().into_owned();
    let triangulate = |input: &str, output: &str, extra: &[&str]| {
        let mut args = vec!["--database_path", &db, "--image_path", &images, "--clear_points", "1"];
        args.extend_from_slice(extra);
        colmap_stage(colmap, out_dir, "point_triangulator", input, output, &args)
    };
    let adjust = |input: &str, output: &str| {
        let args = ["--BundleAdjustment.refine_focal_length", "1", "--BundleAdjustment.refine_extra_params", "1"];
        colmap_stage(colmap, out_dir, "bundle_adjuster", input, output, &args)
    };
    // Flags as in the validated refine.sh: loose thresholds with the seed intrinsics fixed, then
    // 8 px, then COLMAP defaults.
    triangulate("seed", "tri0", &[
        "--refine_intrinsics", "0",
        "--Mapper.tri_create_max_angle_error", "8",
        "--Mapper.tri_continue_max_angle_error", "8",
        "--Mapper.tri_merge_max_reproj_error", "16",
        "--Mapper.tri_complete_max_reproj_error", "16",
        "--Mapper.filter_max_reproj_error", "16",
        "--Mapper.tri_min_angle", "1",
    ])?;
    adjust("tri0", "ba0")?;
    triangulate("ba0", "tri1", &[
        "--refine_intrinsics", "1",
        "--Mapper.filter_max_reproj_error", "8",
        "--Mapper.tri_merge_max_reproj_error", "8",
        "--Mapper.tri_complete_max_reproj_error", "8",
    ])?;
    adjust("tri1", "ba1")?;
    triangulate("ba1", "tri2", &["--refine_intrinsics", "1"])?;
    adjust("tri2", "ba2")?;
    colmap_stage(colmap, out_dir, "point_filtering", "ba2", "filtered", &["--max_reproj_error", "4", "--min_tri_angle", "1.5"])?;
    colmap_stage(colmap, out_dir, "image_filterer", "filtered", "final", &["--min_num_observations", "10"])?;
    // Continue incremental mapping from the learned model (on a fresh database copy, as in the
    // validated polish.sh): registers what the seed missed and refines the rest.
    let mapper_db = copy_database(database, &out_dir.join("mapper.db"))?;
    let mapper_db = mapper_db.to_string_lossy().into_owned();
    colmap_stage(colmap, out_dir, "mapper", "final", "polished", &["--database_path", &mapper_db, "--image_path", &images])?;
    Ok(LearnedSeed {
        model_dir: dir("polished"),
        n_frames: frames.len(),
        inference_seconds,
        peak_gpu_bytes,
        lock_wait_seconds,
    })
}

/// Copy a COLMAP database; a WAL-mode database may keep committed pages in its -wal file.
fn copy_database(src: &Path, dst: &Path) -> anyhow::Result<PathBuf> {
    for suffix in ["", "-wal"] {
        let (from, to) = (path_with_suffix(src, suffix), path_with_suffix(dst, suffix));
        if suffix.is_empty() || from.exists() {
            std::fs::copy(&from, &to).with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
        }
    }
    Ok(dst.to_path_buf())
}

fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Image files (jpg/jpeg/png, any case) directly in `dir`, sorted by name.
pub fn list_frames(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut frames = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let path = entry?.path();
        let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase());
        if path.is_file() && matches!(ext.as_deref(), Some("jpg" | "jpeg" | "png")) {
            frames.push(path);
        }
    }
    frames.sort();
    ensure!(!frames.is_empty(), "no jpg/png frames in {}", dir.display());
    Ok(frames)
}

/// `name -> (image_id, camera_id)` and the single camera's size from a COLMAP database.
fn read_database(db: &Path) -> anyhow::Result<(BTreeMap<String, (u32, u32)>, u32, u32)> {
    let con = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("opening {}", db.display()))?;
    let mut stmt = con.prepare("SELECT camera_id, width, height FROM cameras")?;
    let cameras = stmt
        .query_map([], |r| Ok((r.get::<_, u32>(0)?, r.get::<_, u32>(1)?, r.get::<_, u32>(2)?)))?
        .collect::<Result<Vec<_>, _>>()?;
    let [(_, width, height)] = cameras[..] else {
        anyhow::bail!("{}: expected one camera, found {}", db.display(), cameras.len())
    };
    let mut stmt = con.prepare("SELECT image_id, name, camera_id FROM images")?;
    let ids = stmt
        .query_map([], |r| Ok((r.get::<_, String>(1)?, (r.get::<_, u32>(0)?, r.get::<_, u32>(2)?))))?
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok((ids, width, height))
}

/// Run one COLMAP stage `out_dir/input -> out_dir/output`; on failure, report its stderr tail.
fn colmap_stage(colmap: &Path, out_dir: &Path, stage: &str, input: &str, output: &str, args: &[&str]) -> anyhow::Result<()> {
    let output_dir = out_dir.join(output);
    std::fs::create_dir_all(&output_dir).with_context(|| format!("creating {}", output_dir.display()))?;
    let logs = out_dir.join("logs");
    std::fs::create_dir_all(&logs)?;
    let out = Command::new(colmap)
        .arg(stage)
        .arg("--input_path")
        .arg(out_dir.join(input))
        .arg("--output_path")
        .arg(&output_dir)
        .args(args)
        .current_dir(out_dir)
        .env("QT_QPA_PLATFORM", "offscreen")
        // COLMAP's glog files go to $TMPDIR (default /tmp); keep them under out_dir.
        .env("TMPDIR", &logs)
        .output()
        .with_context(|| format!("running {} {stage}", colmap.display()))?;
    std::fs::write(logs.join(format!("{output}.{stage}.txt")), &out.stderr)?;
    if !out.status.success() {
        let text = String::from_utf8_lossy(if out.stderr.is_empty() { &out.stdout } else { &out.stderr }).into_owned();
        let lines: Vec<&str> = text.lines().collect();
        let tail = lines[lines.len().saturating_sub(30)..].join("\n");
        anyhow::bail!("colmap {stage} ({input} -> {output}) failed with {}:\n{tail}", out.status);
    }
    Ok(())
}

fn frame_names(frames: &[PathBuf]) -> anyhow::Result<Vec<String>> {
    frames
        .iter()
        .map(|f| {
            f.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .with_context(|| format!("frame path {} has no file name", f.display()))
        })
        .collect()
}

/// Load DA3-BASE from `weights`, or `SPLAT_DA3_WEIGHTS` when `None`.
pub fn load_model_from(weights: Option<&Path>, device: &Device) -> anyhow::Result<Da3Pose> {
    Da3Pose::load(&model::weights_path(weights)?, device)
}

/// Run the backbone and camera decoder on a preprocessed batch.
pub fn run(model: &Da3Pose, input: &Preprocessed, device: &Device) -> anyhow::Result<PoseOutput> {
    let (feat, pose) = model.forward(&input.data, input.views, input.height, input.width, device)?;
    let cam_feat = feat.into_data().try_into_vec::<f32>()?;
    let pose = pose.into_data().try_into_vec::<f32>()?;
    let pose_enc = pose
        .chunks_exact(9)
        .map(|c| std::array::from_fn(|i| c[i]))
        .collect();
    Ok(PoseOutput { cam_feat, pose_enc })
}

/// Decode pose encodings (`pose_encoding_to_extri_intri` + `affine_inverse`) into world-to-
/// camera poses, with intrinsics rescaled from the network input to the original frame size.
pub fn cameras_from_pose_enc(
    names: &[String],
    pose_enc: &[[f32; 9]],
    input: &Preprocessed,
) -> Vec<LearnedCamera> {
    let (w, h) = (input.width as f64, input.height as f64);
    let (ow, oh) = (f64::from(input.orig_width), f64::from(input.orig_height));
    names
        .iter()
        .zip(pose_enc)
        .map(|(name, pe)| {
            let pe = pe.map(f64::from);
            let [t0, t1, t2, i, j, k, r, fov_h, fov_w] = pe;
            let two_s = 2.0 / (i * i + j * j + k * k + r * r);
            let c2w = [
                [1.0 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
                [two_s * (i * j + k * r), 1.0 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
                [two_s * (i * k - j * r), two_s * (j * k + i * r), 1.0 - two_s * (i * i + j * j)],
            ];
            let rot: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| c2w[b][a]));
            let t = [t0, t1, t2];
            let trans = std::array::from_fn(|a| -(0..3).map(|b| rot[a][b] * t[b]).sum::<f64>());
            let focal = |side: f64, fov: f64| (side / 2.0) / (fov / 2.0).tan().max(1e-6);
            LearnedCamera {
                name: name.clone(),
                w2c_rotation: rot,
                w2c_translation: trans,
                fx: focal(w, fov_w) * ow / w,
                fy: focal(h, fov_h) * oh / h,
                cx: ow / 2.0,
                cy: oh / 2.0,
            }
        })
        .collect()
}

/// Write a poses-only COLMAP TEXT model (`cameras.txt`, `images.txt`, empty `points3D.txt`)
/// keyed to the caller's database IDs (`name -> (image_id, camera_id)`): one SIMPLE_RADIAL
/// camera with the median focal over all frames, principal point at the centre and k = 0.
/// Frames missing from `ids` are skipped.
pub fn write_colmap_seed(
    cams: &[LearnedCamera],
    ids: &BTreeMap<String, (u32, u32)>,
    width: u32,
    height: u32,
    out_dir: &Path,
) -> anyhow::Result<()> {
    for c in cams {
        let finite = c.w2c_rotation.iter().flatten().chain(&c.w2c_translation).chain([&c.fx, &c.fy]);
        ensure!(finite.copied().all(f64::is_finite), "non-finite learned pose for {}", c.name);
    }
    let mut camera_ids: Vec<u32> = ids.values().map(|&(_, c)| c).collect();
    camera_ids.sort_unstable();
    camera_ids.dedup();
    ensure!(camera_ids.len() == 1, "expected one shared camera in the database, got {camera_ids:?}");
    let camera_id = camera_ids[0];
    ensure!(!cams.is_empty(), "no learned cameras");

    let mut focals: Vec<f64> = cams.iter().flat_map(|c| [c.fx, c.fy]).collect();
    focals.sort_by(f64::total_cmp);
    let mid = focals.len() / 2;
    let f = if focals.len() % 2 == 0 { (focals[mid - 1] + focals[mid]) / 2.0 } else { focals[mid] };

    let (w, h) = (f64::from(width), f64::from(height));
    let mut images = String::new();
    for c in cams {
        let Some(&(image_id, _)) = ids.get(&c.name) else { continue };
        let q = rotation_to_quat(&c.w2c_rotation);
        let [tx, ty, tz] = c.w2c_translation;
        writeln!(images, "{image_id} {} {} {} {} {tx} {ty} {tz} {camera_id} {}\n", q[0], q[1], q[2], q[3], c.name)?;
    }

    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let write = |file: &str, text: &str| {
        let path = out_dir.join(file);
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
    };
    write("cameras.txt", &format!("{camera_id} SIMPLE_RADIAL {width} {height} {f} {} {} 0\n", w / 2.0, h / 2.0))?;
    write("images.txt", &images)?;
    write("points3D.txt", "")
}

/// Shepperd's method (largest diagonal term for stability); unit (w, x, y, z).
fn rotation_to_quat(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let mut q = [0.0; 4];
    if trace > 0.0 {
        let s = 2.0 * (1.0 + trace).sqrt();
        q = [s / 4.0, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s];
    } else {
        let i = (0..3).fold(0, |m, d| if r[d][d] > r[m][m] { d } else { m });
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let s = 2.0 * (1.0 + r[i][i] - r[j][j] - r[k][k]).sqrt();
        q[0] = (r[k][j] - r[j][k]) / s;
        q[1 + i] = s / 4.0;
        q[1 + j] = (r[j][i] + r[i][j]) / s;
        q[1 + k] = (r[k][i] + r[i][k]) / s;
    }
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / n)
}
