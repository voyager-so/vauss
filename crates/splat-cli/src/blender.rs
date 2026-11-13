//! `<run>/exports/blender/`: the run's results packed for Blender — the whole-scene textured mesh, the
//! full splat and the web export's scene.spz (hardlinks, never recomputed), coloured surface voxels from
//! that mesh, and a README with the import steps. `vauss run` writes it at the end; `vauss export-blender <run>` rebuilds it.

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

pub const DIR: &str = "exports/blender";
/// Physical voxel edge when the run's web export has a metric scale estimate.
const VOXEL_M: f64 = 0.05;
/// Voxels along the scene's longest axis when it has none (and the bounds a metric edge may span).
const VOXELS_PER_EXTENT: f32 = 100.0;

#[derive(clap::Args, Debug)]
pub struct ExportBlenderArgs {
    /// Run folder.
    pub run: PathBuf,
    /// Output folder (default <run>/exports/blender).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Voxel edge in the run's scene units (default: 5 cm by the metric scale estimate, else 1/100 of the
    /// scene's longest axis).
    #[arg(long)]
    pub voxel: Option<f32>,
}

/// `vauss run`'s default pack; never fails the run. SPLAT_EXPORT_BLENDER=0 skips it (internal harnesses).
pub fn write_default(root: &Path) {
    if std::env::var("SPLAT_EXPORT_BLENDER").as_deref() == Ok("0") || !root.join("report.json").is_file() {
        return;
    }
    if let Err(e) = run(&ExportBlenderArgs { run: root.to_owned(), out: None, voxel: None }) {
        eprintln!("Blender pack: not written ({e:#}); `vauss export-blender {}` retries", root.display());
    }
}

fn deliverable(root: &Path, report: &Value, key: &str) -> Option<PathBuf> {
    let p = root.join(report["deliverables"][key].as_str()?);
    p.is_file().then_some(p)
}

/// Hardlink (copy across filesystems) `src` to `dst`, downloading a `--cloud` result left on fal first.
fn place(root: &Path, src: &Path, dst: &Path) -> Result<()> {
    splat_core::fal::materialize(root, src).with_context(|| format!("downloading {} from fal", src.display()))?;
    if src.canonicalize().ok().zip(dst.canonicalize().ok()).is_some_and(|(a, b)| a == b) {
        return Ok(());
    }
    let _ = fs::remove_file(dst);
    if fs::hard_link(src, dst).is_err() {
        fs::copy(src, dst).with_context(|| format!("copying {}", src.display()))?;
    }
    Ok(())
}

/// The run's web export scene.json, if any.
fn web_scene(root: &Path) -> Option<Value> {
    serde_json::from_slice(&fs::read(root.join("exports/web/scene.json")).ok()?).ok()
}

/// Meters per scene unit from the web export's frame fit, when it is more than a fallback guess.
fn metric_scale(root: &Path) -> Option<f64> {
    let scene = web_scene(root)?;
    let s = &scene["scale_estimate"];
    let (v, conf) = (s["value"].as_f64()?, s["confidence"].as_f64().unwrap_or(0.));
    (v.is_finite() && v > 0. && conf > 0.2).then_some(v)
}

pub fn run(a: &ExportBlenderArgs) -> Result<Value> {
    let root = a.run.as_path();
    let report: Value = serde_json::from_slice(&fs::read(root.join("report.json")).context("the run has no report.json (unfinished?)")?)?;
    let out = a.out.clone().unwrap_or_else(|| root.join(DIR));
    fs::create_dir_all(&out)?;
    ensure!(root.join("exports/web").canonicalize().ok() != Some(out.canonicalize()?),
        "Blender output must not overwrite the source web package; choose another --out directory");
    let t0 = std::time::Instant::now();
    let mut files: Vec<String> = vec![];
    let mut warnings = vec![];
    let splat = deliverable(root, &report, "splat");
    if let Some(src) = &splat {
        place(root, src, &out.join("splat.ply"))?;
        files.push("splat.ply".into());
    } else {
        warnings.push("the run has no splat PLY".to_string());
    }
    // The web SPZ can be only a core; ordinary importers do not read its sidecars.
    // Deliver a full PLY in that same frame whenever either positions or colours need patches.
    let web = web_scene(root);
    for stale in ["scene.spz", "scene.ply", "cameras.json", "import_cameras.py"] {
        let _ = fs::remove_file(out.join(stale));
    }
    if let Some(w) = &web {
        if !w["splat_neural"].is_null() {
            warnings.push("Blender's ordinary importer does not evaluate the neural appearance sidecar; exported colour is the SH/base-colour approximation.".into());
        }
        if w["splat"] == "scene.spz" {
            let ov = &w["splat_overflow"];
            let incomplete = ["overflow_count", "dc_patch_count"].iter().any(|k| ov[k].as_u64().unwrap_or(0) > 0)
                || ov["positions"].as_str().is_some() || ov["dc"].as_str().is_some();
            if incomplete {
                let src = root.join("exports/web/scene.ply");
                if ov["standalone_splat"] == "scene.ply" && src.is_file() {
                    place(root, &src, &out.join("scene.ply"))?;
                } else {
                    // Older exports have no standalone file: reconstruct all the sidecars.
                    let g = splat_webexport::spz::read_package_splats(&root.join("exports/web"))?;
                    splat_webexport::ply::write_ply(&g, &out.join("scene.ply"))?;
                }
                files.push("scene.ply".into());
                warnings.push(format!("Web scene.spz is not standalone: {} omitted splats and {} colour patches. Delivered complete scene.ply instead.", ov["overflow_count"], ov["dc_patch_count"]));
            } else if root.join("exports/web/scene.spz").is_file() {
                place(root, &root.join("exports/web/scene.spz"), &out.join("scene.spz"))?;
                files.push("scene.spz".into());
            }
        }
        if w["capture_cameras"].as_array().is_some_and(|cs| !cs.is_empty() && cs.iter().all(|c| c["source_c2w"].is_array() && c["K"].is_array())) {
            let cameras = json!({"schema": 1, "camera_convention": w["camera_convention"], "capture_cameras": w["capture_cameras"], "transform_from_source": w["transform_from_source"], "bounds": w["bounds"]});
            fs::write(out.join("cameras.json"), serde_json::to_vec_pretty(&cameras)?)?;
            fs::write(out.join("import_cameras.py"), splat_webexport::cameras::BLENDER_IMPORT_SCRIPT)?;
            files.extend(["cameras.json".into(), "import_cameras.py".into()]);
        } else {
            warnings.push("Exact capture calibration is absent; rerun export-web then export-blender to include cameras.json. Approximate look-at poses are not exact cameras.".into());
        }
    }
    if web.is_none() {
        warnings.push("No web scene metadata: exact capture cameras are unavailable. Run export-web then export-blender.".into());
    }
    let mesh = deliverable(root, &report, "mesh_full_scene").or_else(|| deliverable(root, &report, "mesh")).filter(|p| p.extension().is_some_and(|e| e == "glb"));
    let scale = metric_scale(root);
    let mut voxels = Value::Null;
    if let Some(src) = &mesh {
        let dst = out.join("mesh.glb");
        place(root, src, &dst)?;
        files.push("mesh.glb".into());
        let (lo, hi) = splat_mesh::voxel::glb_bounds(&dst)?;
        let extent = (hi - lo).max_element();
        let auto = extent / VOXELS_PER_EXTENT;
        // A metric edge is kept within 4x of the extent-based one: the scale estimate can be far off outdoors.
        let edge = a.voxel.unwrap_or_else(|| scale.map_or(auto, |s| ((VOXEL_M / s) as f32).clamp(auto / 4., auto * 4.)));
        let name = voxel_name(edge, scale);
        let stats = splat_mesh::voxel::voxelize_glb(&dst, &out.join(&name), edge)?;
        files.push(name);
        voxels = stats.to_json();
    } else {
        warnings.push("the run has no mesh: no mesh.glb or voxels (run with a mesh goal, or `vauss refine` with mesh)".to_string());
    }
    // A previous pack's voxels at another size would be stale next to this one.
    for e in fs::read_dir(&out)?.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if n.starts_with("voxels-") && n.ends_with(".glb") && !files.contains(&n) {
            let _ = fs::remove_file(e.path());
        }
    }
    let gaussians = ply_vertices(&out.join("splat.ply"));
    let readme = readme(gaussians, web.as_ref(), &files, &voxels, scale);
    fs::write(out.join("README.md"), readme)?;
    files.push("README.md".into());
    // Why the pack has no mesh (and so no voxels), mirrored from report.json for Voyager.
    let mesh_skipped = if mesh.is_some() { Value::Null } else { report["mesh_skipped"].clone() };
    let summary = json!({"out": out, "files": files, "gaussians": gaussians, "voxels": voxels, "meters_per_unit": scale, "warnings": warnings, "mesh_skipped": mesh_skipped, "elapsed_s": (t0.elapsed().as_secs_f64() * 100.).round() / 100.});
    fs::write(out.join("blender.json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(summary)
}

/// `element vertex N` from a PLY header.
fn ply_vertices(path: &Path) -> Option<u64> {
    use std::io::BufRead;
    let f = std::io::BufReader::new(fs::File::open(path).ok()?);
    for line in f.split(b'\n').take(200) {
        let line = String::from_utf8(line.ok()?).ok()?;
        if let Some(n) = line.trim().strip_prefix("element vertex ") {
            return n.trim().parse().ok();
        }
        if line.trim() == "end_header" {
            break;
        }
    }
    None
}

fn voxel_name(edge: f32, scale: Option<f64>) -> String {
    match scale {
        Some(s) => format!("voxels-{}cm.glb", ((edge as f64 * s * 100.).round() as i64).max(1)),
        None => "voxels.glb".into(),
    }
}

fn readme(gaussians: Option<u64>, web: Option<&Value>, files: &[String], voxels: &Value, scale: Option<f64>) -> String {
    let has = |f: &str| files.iter().any(|x| x == f);
    let vox = files.iter().find(|f| f.starts_with("voxels"));
    let mut s = String::from("# Blender pack\n\nResults of this run for Blender. Tested with Blender 5.3, which imports Gaussian splats natively; older Blender needs the KIRI add-on (see the end).\n\n## Files\n\n");
    if has("scene.spz") {
        let n = web.and_then(|w| w["stats"]["splats_out"].as_u64()).map(|n| format!(" ({n} Gaussians)")).unwrap_or_default();
        s += &format!(
            "- `scene.spz`: the run's web export{n} (the same file as `exports/web/scene.spz`): upright (Z up in Blender), in metres, floor at 0, with all its colour (spherical harmonics). Blender 5.3+: File → Import → SPZ. No add-on needed. The easiest way to see the splat.\n"
        );
    }
    if has("scene.ply") {
        s += "- `scene.ply`: complete web-export splat, including all overflow positions and colour patches, in the web scene frame. Import as Stanford PLY with numeric XYZ unchanged. The incomplete web SPZ is deliberately not included.\n";
    }
    if has("splat.ply") {
        s += &format!(
            "- `splat.ply`: the full Gaussian splat{}, with all its colour (spherical harmonics), opacity, scale and rotation, in the run's reconstruction frame (lines up with the mesh and voxels). Blender 5.3+: File → Import → Stanford PLY.\n",
            gaussians.map(|n| format!(" ({n} Gaussians)")).unwrap_or_default()
        );
    }
    if has("mesh.glb") {
        s += "- `mesh.glb`: the whole-scene surface mesh with its colour texture. File → Import → glTF 2.0 (.glb/.gltf). No add-on needed.\n";
    }
    if let Some(v) = vox {
        let edge = voxels["edge_units"].as_f64().unwrap_or(0.);
        let size = match scale {
            Some(m) => format!("about {:.1} cm cubes ({edge:.4} scene units; metric scale estimated from camera height)", edge * m * 100.),
            None => format!("cubes of {edge:.4} scene units (1/100 of the scene's longest side; no metric scale)"),
        };
        s += &format!(
            "- `{v}`: coloured surface voxels, {size}: {} cubes, {} triangles. File → Import → glTF 2.0. Each cube takes the mean colour of the mesh surface inside it; one-cell holes are filled; only outside faces are kept. These are surface sheets: walls and furniture are not filled solid.\n",
            voxels["occupied_cells"], voxels["triangles"]
        );
    }
    if has("cameras.json") {
        s += "\n## Exact capture camera\n\n`cameras.json` preserves every capture image, native width/height, unequal fx/fy, principal point, lens model and column-major OpenCV camera-to-world matrices. `source_c2w` matches the raw `splat.ply`; `c2w` matches the transformed web scene. No look-at reconstruction is needed.\n\nRun `blender scene.blend --python import_cameras.py -- --image <capture-image-name> --frame source` after importing `splat.ply` with numeric XYZ unchanged (do not apply the orientation-only rotation below). Use `--frame scene` for `scene.ply`, also with numeric XYZ unchanged, or `--frame spz` for Blender's SPZ importer (+90 degrees about X). The script selects one camera and sets its native resolution, pixel aspect and principal-point shifts. Rotate/scale the object and camera together if you want another world convention. Distorted lenses are rejected: undistort and re-export first. This guarantees calibrated framing, not equality between Blender and the native Gaussian rasterizer.\n";
    }
    if !has("cameras.json") {
        s += "\nExact capture cameras are unavailable in this pack. Regenerate export-web then export-blender with the current version; a position/look-at/FOV record cannot recover unequal focal lengths, principal point or roll.\n";
    }
    s += "\n## Import the splat (Blender 5.3 and later)\n\n";
    if has("scene.spz") {
        s += "- **`scene.spz`** (recommended): File → Import → SPZ (Python: `bpy.ops.wm.spz_import(filepath=...)`). It loads upright, in metres, with the floor at 0; a million splats import in well under a second. \
The capture cameras in the run's `exports/web/scene.json` (`capture_cameras`, +Y up) line up with it once mapped to Blender's Z up (rotate +90° about X).\n";

    }
    if has("splat.ply") {
        s += "- **`splat.ply`**: File → Import → Stanford PLY (Python: `bpy.ops.wm.ply_import(filepath=...)`) imports it as a Gaussian-splat point cloud. Rotate it **−90° about X** to stand it upright. \
Its frame is the run's reconstruction frame, not the web export's: it differs from `scene.spz` in offset and scale, and the mesh and voxels line up with it, not with `scene.spz`.\n";
    }
    s += "\nSplats render in Cycles and EEVEE with a default emissive shader that looks like the capture. To relight them, give the point cloud a material with an Attribute node `radiance:base` → Diffuse BSDF (optionally mixed with some Emission). \
Use Cycles for relit shots and for splats mixed with meshes: EEVEE does not show splats in mesh reflections.\n\n\
Blender 5.3 limits: it cannot export splats, and **Apply Transform corrupts splat scale and rotation**: move, rotate and scale the object instead of applying its transform.\n\n";
    s += "## Orientation and scale\n\n\
`splat.ply`, `mesh.glb` and the voxels share the run's reconstruction frame and line up with each other. Blender's glTF importer turns glTF's Y-up into Z-up; rotate the imported PLY splat to match (above). Scale or move all objects together. \
`scene.spz` is in its own upright, metric frame and does not line up with them.\n\n";
    s += &match scale {
        Some(m) => format!("Estimated scale of the reconstruction frame: one scene unit ≈ {m:.3} m (from typical handheld camera height, not measured). Scale the PLY, mesh and voxels together by {m:.3} for approximate metres; measure a known dimension for accuracy.\n\n"),
        None => "The reconstruction frame's units are not metres: measure a known dimension in Blender and scale the PLY, mesh and voxels together.\n\n".into(),
    };
    s += "## Blender older than 5.3: KIRI 3DGS Render add-on\n\n\
1. Blender 5.1 or 5.2: download `3dgs_render_by_kiri_engine_5.1.0.zip` from https://github.com/Kiri-Innovation/3dgs-render-blender-addon/releases/tag/v5.1.0 (keep it zipped). \
Blender 4.3–4.5: use KIRI 4.1.5 instead, https://github.com/Kiri-Innovation/3dgs-render-blender-addon/releases/tag/v4.1.5.\n\
2. Edit → Preferences → Get Extensions → menu (top right) → Install from Disk → the ZIP. Enable **3DGS Render by KIRI Engine**.\n\
3. In the 3D Viewport press **N**, open the **3DGS Render** tab, choose **Edit**, then Import → PLY and pick `splat.ply`. For large scenes use the **Verts** option.\n\
4. Switch the viewport to Render mode to see the Gaussians composited.\n\n\
Before 5.3, the Gaussian look only shows through the add-on's renderer; Blender's own PLY import sees the splat as coloured points. \
Rotate the KIRI-imported splat about X until it lines up with the mesh, if KIRI has not already done so.\n\n";
    s += "The mesh and voxels approximate the captured surfaces, with the captured lighting baked into their colours: uncaptured areas can leave holes and floaters. They are not watertight or CAD-accurate. \
Files here may be hardlinks to the run's own results: save edited scenes and exports under new names.\n";
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use splat_webexport::{overflow, ply, spz};

    #[test]
    fn blender_pack_never_recommends_an_incomplete_spz_and_recovers_old_sidecars() {
        let root = tempfile::tempdir().unwrap();
        let web = root.path().join("exports/web");
        fs::create_dir_all(&web).unwrap();
        fs::write(root.path().join("report.json"), "{\"deliverables\":{}}").unwrap();
        let g = ply::Gaussians {
            pos: vec![[0.0, 0.0, 1.0], [3173.0, 6000.0, -8000.0]],
            scale: vec![[-2.0; 3]; 2], rot: vec![[1.0, 0.0, 0.0, 0.0]; 2],
            opacity: vec![1.0; 2], dc: vec![[0.2; 3]; 2], ..Default::default()
        };
        let opts = spz::Options { version: 3, sh1_bits: 6, sh_rest_bits: 6, antialiased: false, symmetric_sh: true };
        let meta = overflow::write_package(&g, &web, "scene", opts).unwrap();
        fs::write(web.join("scene.json"), serde_json::to_vec(&json!({"splat":"scene.spz", "splat_overflow":meta})).unwrap()).unwrap();
        let unsafe_args = ExportBlenderArgs { run: root.path().to_path_buf(), out: Some(web.clone()), voxel: None };
        assert!(run(&unsafe_args).unwrap_err().to_string().contains("source web package"));
        assert!(web.join("scene.spz").exists());
        let args = ExportBlenderArgs { run: root.path().to_path_buf(), out: None, voxel: None };
        let out = root.path().join(DIR);
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("scene.spz"), "stale partial export").unwrap();
        let summary = run(&args).unwrap();
        assert_eq!(ply::read_ply(&out.join("scene.ply")).unwrap().pos, g.pos);
        assert!(!out.join("scene.spz").exists());
        assert!(summary["warnings"].as_array().unwrap().iter().any(|v| v.as_str().unwrap().contains("not standalone")));
        let readme = fs::read_to_string(out.join("README.md")).unwrap();
        assert!(readme.contains("complete web-export splat"));
        assert!(!readme.contains("**`scene.spz`** (recommended)"));
        // Pre-fix package has only sidecars, no complete PLY: recover all positions there too.
        fs::remove_file(web.join("scene.ply")).unwrap();
        run(&args).unwrap();
        let recovered = ply::read_ply(&out.join("scene.ply")).unwrap();
        assert_eq!(recovered.pos, g.pos);
        // A later compatible export must remove the old full-scene fallback, not leave stale files.
        let mut small = g;
        small.pos[1] = [1.0, 2.0, 3.0];
        let meta = overflow::write_package(&small, &web, "scene", opts).unwrap();
        fs::write(web.join("scene.json"), serde_json::to_vec(&json!({"splat":"scene.spz", "splat_overflow":meta})).unwrap()).unwrap();
        run(&args).unwrap();
        assert!(out.join("scene.spz").exists());
        assert!(!out.join("scene.ply").exists());
    }
}
