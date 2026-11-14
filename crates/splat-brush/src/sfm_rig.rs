//! The 360 camera solve: 14 exact pinhole views per frame (splat-core panorama split) as one COLMAP
//! rig. A short ladder: sequential matching with loop detection, the incremental mapper with fixed
//! intrinsics and rig, one low-angle retry from the same database, then the rig check. None of the
//! video ladder (keyframes, lens-last, flip retry, focal guard, learned fallbacks) applies: they
//! assume one camera whose name order is time order.

use serde_json::json;
use splat_core::panorama::{self, Manifest};

use crate::model;
use crate::sfm::SfmJob;
use crate::util::*;

/// Rig check tolerances (spec): rotation relative to view 0, and camera centres relative to the
/// median step between consecutive frames.
const RIG_MAX_ANGLE_DEG: f64 = 0.01;
const RIG_MAX_CENTRE_REL: f64 = 1e-4;
/// Registered share below which the solve maps again at a 4 deg initial pair.
const RETRY_BELOW: f64 = 0.95;

/// The options of the first 360 solve (also used by the tests).
pub(crate) fn rig_options(job: &SfmJob, manifest: &Manifest, colmap_dir: &std::path::Path) -> splat_sfm::ReconstructOptions {
    let mut opts = splat_sfm::ReconstructOptions::new(colmap_dir);
    opts.deadline = splat_sfm::budget_deadline();
    if let Some(seed) = job.mapper_seed {
        opts.mapper_seed = seed;
    }
    opts.colmap = job.colmap.clone();
    opts.gpu_lock = job.gpu_lock.clone();
    opts.use_gpu = job.use_gpu;
    opts.mapper = splat_sfm::Mapper::Incremental;
    opts.matching = splat_sfm::Matching::Sequential;
    opts.sequential_overlap = 6;
    opts.quadratic_overlap = true;
    opts.vocab_tree = job.vocab_tree.clone();
    opts.camera_model = "PINHOLE".into();
    opts.single_camera = true;
    opts.init_min_tri_angle = 8.0;
    opts.ba_global_ratio = Some(1.3);
    opts.ransac_max_trials = 2000;
    // Every core: ~1,400 small views make matching CPU-bound on the CPU-only Linux COLMAP (Temple3,
    // 1,008 views on 32 vCPUs at the default 8 threads: 1.75 s per image, ~29 min of matching).
    opts.num_threads = std::thread::available_parallelism().map_or(opts.num_threads, |n| n.get().max(opts.num_threads));
    opts.max_image_size = job.max_image_size;
    opts.max_features = job.max_features;
    opts.video = false;
    let f = manifest.view_size as f64 / 2.0;
    opts.rig = Some(splat_sfm::RigOptions {
        config: job.dir.join(panorama::RIG_CONFIG),
        camera_params: format!("{f},{f},{f},{f}"),
        enforce: !job.rig_off,
    });
    opts
}

fn share(rec: &splat_sfm::Reconstruction) -> f64 {
    rec.health.n_registered as f64 / rec.health.n_input.max(1) as f64
}

/// Distinct 360 frames with at least one registered view.
fn panoramas_registered(model_dir: &std::path::Path) -> anyhow::Result<usize> {
    let names = model::names(model_dir)?;
    Ok(names.iter().filter_map(|n| panorama::group_key(n)).collect::<std::collections::BTreeSet<_>>().len())
}

pub(crate) fn run_rig_job(job: &SfmJob) -> anyhow::Result<i32> {
    let dir = &job.dir;
    let manifest = Manifest::read(dir)?.ok_or_else(|| anyhow::anyhow!("360 SfM dataset has no {}", panorama::MANIFEST))?;
    let colmap_dir = dir.join("colmap");
    let first_dir = dir.join("colmap.rig-first");
    for d in [&colmap_dir, &first_dir, &dir.join("sparse")] {
        let _ = std::fs::remove_dir_all(d);
    }
    let images = dir.join("images");
    let mut opts = rig_options(job, &manifest, &colmap_dir);
    let feature_masks = dir.join("feature_masks");
    let _ = std::fs::remove_dir_all(&feature_masks);
    opts.mask_dir = job.mask_dir.as_deref().map(|m| crate::sfm::colmap_masks(m, &images, &images, &feature_masks)).transpose()?;
    println!("[sfm] 360 solve: {} frames x {} views, rig {}", manifest.frames.len(), manifest.views, if job.rig_off { "off" } else { "on" });
    let first = splat_sfm::reconstruct(&images, &opts);
    let first_share = first.as_ref().map_or(0.0, share);
    let mut retried = None;
    // One low-angle retry from the frozen database when the first solve failed or placed too little.
    let rec = if first_share < RETRY_BELOW && !splat_sfm::deadline_passed(&opts, "the 360 low-angle retry") {
        match &first {
            Ok(r) => println!("[sfm] 360 solve placed {}/{} views; mapping again at a 4 deg initial pair", r.health.n_registered, r.health.n_input),
            Err(e) => println!("[sfm] 360 solve failed ({e:#}); mapping again at a 4 deg initial pair"),
        }
        std::fs::rename(&colmap_dir, &first_dir)?;
        let mut low = opts.clone();
        low.init_min_tri_angle = 4.0;
        low.reuse_database = Some(splat_sfm::frozen_database(&first_dir));
        let second = splat_sfm::reconstruct(&images, &low);
        retried = Some(json!({"first_registered": first.as_ref().ok().map(|r| r.health.n_registered),
            "first_error": first.as_ref().err().map(|e| format!("{e:#}")),
            "retry_registered": second.as_ref().ok().map(|r| r.health.n_registered)}));
        match (first, second) {
            // Keep whichever registers more. The first attempt ran in colmap/ (its model_dir points
            // there), so moving its directory back makes its paths valid again.
            (Ok(a), Ok(b)) if a.health.n_registered >= b.health.n_registered => {
                std::fs::remove_dir_all(&colmap_dir)?;
                std::fs::rename(&first_dir, &colmap_dir)?;
                a
            }
            (_, Ok(b)) => b,
            (Ok(a), Err(e)) => {
                eprintln!("[sfm] 360 low-angle retry failed: {e:#}");
                let _ = std::fs::remove_dir_all(&colmap_dir);
                std::fs::rename(&first_dir, &colmap_dir)?;
                a
            }
            (Err(a), Err(b)) => return Err(b.context(format!("360 solve failed twice (first: {a:#})"))),
        }
    } else {
        first?
    };
    let _ = std::fs::remove_dir_all(&first_dir);
    let _ = std::fs::remove_dir_all(&feature_masks);
    let rig_check = if job.rig_off {
        None
    } else {
        let layout: Vec<[f64; 9]> = manifest.cam_from_pano.clone();
        let check = splat_sfm::check_rig(&rec.model_dir, &manifest.prefixes, &layout, &manifest.frames, RIG_MAX_ANGLE_DEG, RIG_MAX_CENTRE_REL)?;
        println!("[sfm] rig check: {} frames, worst {:.5} deg, centres {:.1e} of a frame step", check.frames, check.worst_angle_deg, check.worst_centre_rel);
        Some(check)
    };
    println!("[sfm] model {} ({} registered) in {:.1} s", rec.model_dir.display(), rec.health.n_registered, rec.wall_seconds);
    let mut components = Vec::new();
    components.extend(model::models_by_size(&colmap_dir.join("sparse"))?);
    components.extend(model::models_by_size(&colmap_dir.join("components"))?);
    components.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    std::fs::create_dir_all(dir.join("sparse"))?;
    std::fs::rename(&rec.model_dir, dir.join("sparse/0"))?;
    // Brush mounts the whole dataset dir: one trainable model, no database (as the main ladder).
    for entry in std::fs::read_dir(&colmap_dir)? {
        let p = entry?.path();
        if p.is_dir() && p.file_name().is_some_and(|n| n != "components") {
            std::fs::remove_dir_all(&p)?;
        } else if p.extension().is_some_and(|e| e == "db" || e == "ply") {
            std::fs::remove_file(&p)?;
        }
    }
    let registered = panoramas_registered(&dir.join("sparse/0"))?;
    let mut rec_json = serde_json::to_value(&rec)?;
    rec_json["component_sizes"] = json!(components.iter().map(|c| c.1).collect::<Vec<_>>());
    rec_json["panorama"] = json!({"views": manifest.views, "panoramas_registered": registered,
        "panoramas_input": manifest.frames.len(), "rig": !job.rig_off, "rig_check": rig_check, "low_angle_retry": retried});
    write_json(&dir.join("reconstruction.json"), &rec_json)?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use splat_core::RunCtx;
    use splat_core::panorama::equirect_dir;

    /// Value noise in [0, 1] on a lattice of `cell` units, hashed per face.
    fn noise(u: f64, v: f64, cell: f64, seed: u64) -> f64 {
        let hash = |i: i64, j: i64| -> f64 {
            let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            (h >> 11) as f64 / (1u64 << 53) as f64
        };
        let (x, y) = (u / cell, v / cell);
        let (i, j) = (x.floor(), y.floor());
        let (fx, fy) = (x - i, y - j);
        let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
        let (i, j) = (i as i64, j as i64);
        let a = hash(i, j) + (hash(i + 1, j) - hash(i, j)) * sx;
        let b = hash(i, j + 1) + (hash(i + 1, j + 1) - hash(i, j + 1)) * sx;
        a + (b - a) * sy
    }

    /// Colour of the inside of a 6 x 3 x 6 m box room (y down, floor at +1.5) seen from `c` along `d`.
    fn room(c: [f64; 3], d: [f64; 3]) -> [u8; 3] {
        let half = [3.0, 1.5, 3.0];
        let (mut t, mut axis) = (f64::INFINITY, 0);
        for k in 0..3 {
            if d[k].abs() > 1e-12 {
                let tk = ((if d[k] > 0.0 { half[k] } else { -half[k] }) - c[k]) / d[k];
                if tk < t {
                    t = tk;
                    axis = k;
                }
            }
        }
        let p = [c[0] + t * d[0], c[1] + t * d[1], c[2] + t * d[2]];
        let (u, v) = match axis { 0 => (p[2], p[1]), 1 => (p[0], p[2]), _ => (p[0], p[1]) };
        let seed = axis as u64 * 2 + u64::from(d[axis] > 0.0);
        let g = 0.55 * noise(u, v, 0.06, seed) + 0.3 * noise(u, v, 0.25, seed + 7) + 0.15 * noise(u, v, 1.0, seed + 13);
        let tint = noise(u, v, 0.5, seed + 21);
        [(255.0 * g) as u8, (255.0 * (0.7 * g + 0.3 * tint)) as u8, (255.0 * (0.8 * g + 0.2 * (1.0 - tint))) as u8]
    }

    fn render(c: [f64; 3], w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| image::Rgb(room(c, equirect_dir(x as f64 + 0.5, y as f64 + 0.5, w, h))))
    }

    /// Real-COLMAP 360 solve (spec section 7). Skipped, with the reason printed, unless SPLAT_REAL_COLMAP
    /// names a COLMAP 4.2.1 binary.
    /// Run: SPLAT_REAL_COLMAP=... cargo test -p splat-brush real_colmap_solves_a_rigged -- --nocapture
    #[test]
    fn real_colmap_solves_a_rigged_360_circle() -> anyhow::Result<()> {
        let Some(colmap) = std::env::var_os("SPLAT_REAL_COLMAP") else {
            eprintln!("SKIPPED real_colmap_solves_a_rigged_360_circle: set SPLAT_REAL_COLMAP to run against real COLMAP");
            return Ok(());
        };
        let tmp = tempfile::tempdir()?;
        // SPLAT_RIG_TEST_KEEP=1 keeps the work dir for diagnosis.
        let (root, _guard) = if std::env::var_os("SPLAT_RIG_TEST_KEEP").is_some() { (tmp.keep(), None) } else { (tmp.path().to_path_buf(), Some(tmp)) };
        eprintln!("work dir {}", root.display());
        let source = root.join("source");
        std::fs::create_dir(&source)?;
        let n = 16;
        let truth: Vec<[f64; 3]> = (0..n).map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            [0.5 * a.cos(), 0.0, 0.5 * a.sin()]
        }).collect();
        for (i, c) in truth.iter().enumerate() {
            render(*c, 1024, 512).save(source.join(format!("pano_{i:03}.png")))?;
        }
        let run = splat_core::Run::create(root.join("run"), &source, "pano rig", "brush")?;
        let cfg = crate::BrushConfig::from_env();
        let extracted = crate::media::extract_frames(&cfg, &run, serde_json::from_value(json!({"panorama": true, "input_kind": "video", "drop_duplicates": false}))?)?;
        assert_eq!(extracted.metrics["frame_count"], n);
        let split = crate::panorama::split_panoramas(&run, splat_contracts::SplitPanoramas { frames_id: extracted.artifacts[0].id.clone(), view_size: 256, nadir_mask_deg: 30.0 })?;
        let frames = run.path(&split.artifacts[0].id)?;
        // The SfM dataset as reconstruct_sfm lays it out.
        let dir = root.join("sfm");
        for sub in ["images", "masks"] {
            std::fs::create_dir_all(dir.join(sub))?;
        }
        for src in list_images(&frames.join("images"))? {
            std::fs::copy(&src, dir.join("images").join(src.file_name().unwrap()))?;
        }
        for src in list_images(&frames.join("train_masks"))? {
            std::fs::copy(&src, dir.join("masks").join(src.file_name().unwrap()))?;
        }
        for name in [panorama::MANIFEST, panorama::RIG_CONFIG] {
            std::fs::copy(frames.join(name), dir.join(name))?;
        }
        let colmap = std::path::PathBuf::from(colmap);
        let job: SfmJob = serde_json::from_value(json!({"colmap": colmap, "dir": dir, "camera_model": "PINHOLE", "video": false,
            "max_image_size": 256, "max_features": 4096, "sequential": true, "mask_dir": frames.join("feature_masks"),
            "gpu_lock": null, "rig": dir}))?;
        assert_eq!(run_rig_job(&job)?, 0);
        let rec: serde_json::Value = read_json(&dir.join("reconstruction.json"))?;
        eprintln!("360 solve: {}", rec["panorama"]);
        assert_eq!(rec["panorama"]["panoramas_registered"], n, "{}", rec["panorama"]);
        assert_eq!(rec["component_sizes"].as_array().map(Vec::len), Some(1), "one model: {}", rec["component_sizes"]);
        assert!(rec["panorama"]["rig_check"]["worst_angle_deg"].as_f64().unwrap() <= RIG_MAX_ANGLE_DEG);
        // Similarity alignment through the reference view's rotation (truth cameras are axis-aligned).
        let model = splat_eval::read_model(&dir.join("sparse/0"))?;
        let rot = |q: [f64; 4]| glam::DMat3::from_quat(glam::DQuat::from_xyzw(q[1], q[2], q[3], q[0]).normalize());
        let mut pairs = Vec::new();
        let mut world_rotation = None;
        for im in model.images.values().filter(|im| im.name.starts_with("v00_")) {
            let i: usize = im.name["v00_frame_".len()..im.name.len() - 4].parse()?;
            let r = rot(im.qvec);
            world_rotation.get_or_insert(r);
            pairs.push((glam::DVec3::from_array(truth[i]), -(r.transpose() * glam::DVec3::from_array(im.tvec))));
        }
        let r = world_rotation.unwrap();
        let est: Vec<glam::DVec3> = pairs.iter().map(|p| r * p.1).collect();
        let (mt, me) = (pairs.iter().map(|p| p.0).sum::<glam::DVec3>() / n as f64, est.iter().sum::<glam::DVec3>() / n as f64);
        let spread = |v: &mut dyn Iterator<Item = f64>| v.sum::<f64>().sqrt();
        let scale = spread(&mut pairs.iter().map(|p| (p.0 - mt).length_squared())) / spread(&mut est.iter().map(|e| (*e - me).length_squared()));
        let worst = pairs.iter().zip(&est).map(|(p, e)| ((*e - me) * scale - (p.0 - mt)).length()).fold(0.0, f64::max);
        eprintln!("frame centres: worst {worst:.4} m after similarity alignment (scale {scale:.3})");
        assert!(worst <= 0.02, "frame centres off by {worst} m (2% of the 1 m diameter)");
        // Vauss's readers take the multi-sensor model.
        let loaded = crate::engine::runtime()?.block_on(crate::engine::load_scene(&dir, 256))?;
        drop(loaded);
        // The unrigged baseline (SPLAT_PANO_RIG=0) on the same views: its masked nadir views stay
        // unregistered, so it takes the low-angle retry and keeps the better attempt.
        let base = root.join("sfm-norig");
        for sub in ["images", "masks"] {
            std::fs::create_dir_all(base.join(sub))?;
            for src in list_images(&dir.join(sub))? {
                std::fs::copy(&src, base.join(sub).join(src.file_name().unwrap()))?;
            }
        }
        for name in [panorama::MANIFEST, panorama::RIG_CONFIG] {
            std::fs::copy(dir.join(name), base.join(name))?;
        }
        let mut job = job;
        job.dir = base.clone();
        job.rig = Some(base.clone());
        job.rig_off = true;
        assert_eq!(run_rig_job(&job)?, 0);
        let rec: serde_json::Value = read_json(&base.join("reconstruction.json"))?;
        eprintln!("unrigged baseline: {} registered of {}; {}", rec["health"]["n_registered"], rec["health"]["n_input"], rec["panorama"]);
        assert_eq!(rec["panorama"]["rig"], false);
        // It placed under 95% of the views, so the retry ran and the first attempt was kept.
        assert!(rec["panorama"]["low_angle_retry"].is_object(), "{}", rec["panorama"]);
        assert!(base.join("sparse/0").is_dir() && !base.join("colmap.rig-first").exists());
        Ok(())
    }
}
