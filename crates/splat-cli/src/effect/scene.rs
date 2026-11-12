//! What an effect runs on: a web scene package (`splat export-web`: scene.json + scene.spz, +Y up,
//! meters), a Voyager scene root (scene.json listing `splats/<id>/` packages), or a bare .ply/.spz.
//! Splat order is file order, so splat i here is splat i in every viewer of the package.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use glam::Vec3;
use serde_json::Value;
use splat_webexport::ply::Gaussians;

pub struct Scene {
    pub g: Gaussians,
    /// The package dir (scene.json + splats), when the input is one.
    pub package: Option<PathBuf>,
    pub meta: Value,
    pub up: Vec3,
    #[allow(dead_code)]
    pub focus: Vec3,
    pub radius: f32,
    /// The package's start camera first, then its capture cameras.
    pub cams: Vec<Cam>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Cam {
    pub position: Vec3,
    pub look_at: Vec3,
    pub vfov_deg: f32,
}

/// Which package of a Voyager scene root to load (`--splat`), else the root's `role` default.
pub enum Pick<'a> {
    Id(&'a str),
    /// `active_splat` (interactive) for previews and bakes.
    Active,
    /// `render_splat` (full quality) for videos.
    Render,
}

fn vec3(v: &Value) -> Option<Vec3> {
    let a = v.as_array()?;
    if a.len() != 3 {
        return None;
    }
    Some(Vec3::new(a[0].as_f64()? as f32, a[1].as_f64()? as f32, a[2].as_f64()? as f32))
}

fn cam(v: &Value) -> Option<Cam> {
    Some(Cam { position: vec3(&v["position"])?, look_at: vec3(&v["look_at"])?, vfov_deg: v["vfov_deg"].as_f64().unwrap_or(50.0) as f32 })
}

fn read_json(p: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(p).with_context(|| format!("reading {}", p.display()))?).with_context(|| format!("parsing {}", p.display()))
}

/// The package dirs of a Voyager scene root (`splats: [{id, path}]`), in its order.
pub fn root_packages(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let v = read_json(&root.join("scene.json"))?;
    let list = v["splats"].as_array().context("scene.json has no splats list")?;
    Ok(list.iter().filter_map(|s| Some((s["id"].as_str()?.to_string(), root.join(s["path"].as_str()?)))).collect())
}

pub fn is_root(dir: &Path) -> bool {
    dir.is_dir() && read_json(&dir.join("scene.json")).is_ok_and(|v| v["splats"].is_array())
}

/// Resolve a scene argument to a package dir (or a bare splat file).
pub fn resolve(input: &Path, pick: Pick) -> Result<PathBuf> {
    if !is_root(input) {
        return Ok(input.to_path_buf());
    }
    let v = read_json(&input.join("scene.json"))?;
    let id = match pick {
        Pick::Id(id) => id.to_string(),
        Pick::Active => v["active_splat"].as_str().or(v["render_splat"].as_str()).context("scene.json names no active_splat")?.to_string(),
        Pick::Render => v["render_splat"].as_str().or(v["active_splat"].as_str()).context("scene.json names no render_splat")?.to_string(),
    };
    root_packages(input)?.into_iter().find(|(i, _)| *i == id).map(|(_, p)| p).with_context(|| format!("scene.json has no splat {id:?}"))
}

pub fn open(input: &Path, pick: Pick) -> Result<Scene> {
    let path = resolve(input, pick)?;
    let (g, package, meta) = if path.is_dir() {
        ensure!(path.join("scene.json").is_file(), "{} has no scene.json (not a web scene package)", path.display());
        let meta = read_json(&path.join("scene.json"))?;
        (splat_webexport::spz::read_package_splats(&path)?, Some(path.clone()), meta)
    } else {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let g = match ext.as_str() {
            "ply" => splat_webexport::ply::read_ply(&path)?,
            "spz" => splat_webexport::spz::read_spz(&path)?,
            _ => bail!("{}: expected a scene package dir, a .ply or a .spz", path.display()),
        };
        // A package's own splat file: use its scene.json for cameras and frame.
        let side = path.with_file_name("scene.json");
        let meta = if ext == "spz" && side.is_file() { read_json(&side)? } else { Value::Null };
        (g, None, meta)
    };
    ensure!(!g.is_empty(), "{} holds no splats", path.display());
    Ok(frame(g, package, meta))
}

fn frame(g: Gaussians, package: Option<PathBuf>, meta: Value) -> Scene {
    let up = vec3(&meta["up"]).map(|u| u.normalize_or(Vec3::Y)).unwrap_or(Vec3::Y);
    // Robust centre / size of a bare splat: the median position and 80th-percentile distance.
    let (median, spread) = {
        let mut axes: [Vec<f32>; 3] = Default::default();
        for p in g.pos.iter().step_by((g.len() / 20000).max(1)) {
            (0..3).for_each(|k| axes[k].push(p[k]));
        }
        let m = axes.map(|mut a| {
            a.sort_by(f32::total_cmp);
            a[a.len() / 2]
        });
        let m = Vec3::from(m);
        let mut d: Vec<f32> = g.pos.iter().step_by((g.len() / 20000).max(1)).map(|p| Vec3::from(*p).distance(m)).collect();
        d.sort_by(f32::total_cmp);
        (m, d[(d.len() * 8 / 10).min(d.len() - 1)].max(1e-3))
    };
    let focus = vec3(&meta["focus"]).unwrap_or(median);
    let radius = meta["radius"].as_f64().map(|r| r as f32).unwrap_or(spread);
    let mut cams: Vec<Cam> = Vec::new();
    cams.extend(cam(&meta["start_camera"]));
    cams.extend(meta["capture_cameras"].as_array().into_iter().flatten().filter_map(cam));
    if cams.is_empty() {
        // No package cameras: look at the centre from 2.2 radii out, slightly above.
        let side = up.any_orthonormal_vector();
        cams.push(Cam { position: focus + (side * 2.2 + up * 0.6) * radius, look_at: focus, vfov_deg: 50.0 });
    }
    Scene { g, package, meta, up, focus, radius, cams }
}

impl Scene {
    /// `auto`: the start camera plus the capture cameras farthest from those chosen (n total);
    /// `orbit:N`: N cameras on a ring around the focus at the start camera's distance and height.
    pub fn pick_cams(&self, spec: &str) -> Result<Vec<Cam>> {
        let spec = spec.trim();
        let (kind, n) = match spec.split_once(':') {
            Some((k, n)) => (k, n.parse::<usize>().with_context(|| format!("--cams {spec}: bad count"))?),
            None if spec.parse::<usize>().is_ok() => ("auto", spec.parse()?),
            None => (spec, 3),
        };
        ensure!(n >= 1 && n <= 16, "--cams: 1 to 16 cameras");
        match kind {
            "auto" => {
                let mut out = vec![self.cams[0]];
                while out.len() < n.min(self.cams.len()) {
                    let next = self.cams.iter().max_by(|a, b| {
                        let d = |c: &Cam| out.iter().map(|o| o.position.distance(c.position) + o.look_at.distance(c.look_at)).fold(f32::MAX, f32::min);
                        d(a).total_cmp(&d(b))
                    });
                    out.push(*next.unwrap());
                }
                while out.len() < n {
                    // Fewer package cameras than asked: fill with orbit views.
                    out.push(self.orbit(out.len() as f32 / n as f32 * 360.0));
                }
                Ok(out)
            }
            "orbit" => Ok((0..n).map(|i| self.orbit(i as f32 / n as f32 * 360.0)).collect()),
            "start" | "fixed" => Ok(vec![self.cams[0]; n.min(1)]),
            _ => bail!("--cams {spec}: expected N, auto, auto:N or orbit:N"),
        }
    }

    /// The start camera turned `deg` degrees about the up axis through the point it looks at.
    pub fn orbit(&self, deg: f32) -> Cam {
        let start = self.cams[0];
        let target = start.look_at;
        let rot = glam::Quat::from_axis_angle(self.up, deg.to_radians());
        Cam { position: target + rot * (start.position - target), look_at: target, vfov_deg: start.vfov_deg }
    }
}
