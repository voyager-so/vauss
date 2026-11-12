//! Render adapter for the B15 analytic-surface fixtures (`research/b15/analytic_surfaces.py`):
//! one request JSON in, one NPZ out. Renders 2DGS surfels through the native geometry pass and
//! reports RGB, alpha, expected/median camera-z depth and camera-space normals, plus the
//! fixed-cotangent objective when the request names one. Debug tooling for the surfel port.

use std::io::{Read, Seek, Write};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use brush_render::camera::{Camera, focal_to_fov};
use brush_render::gaussian_splats::{RasterPass, SplatRenderMode, Splats};
use brush_render::kernels::camera_model::CameraModel;
use brush_render::{TextureMode, render_splats, render_splats_geometry};
use burn::module::Module;
use burn::tensor::{Tensor, TensorData};
use glam::{Mat3, Quat, UVec2, Vec2, Vec3};

use crate::engine;

struct Array {
    shape: Vec<usize>,
    data: Vec<f64>,
}

fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let (_, rest) = header.split_once(&format!("'{key}'")).with_context(|| format!("NPY header missing {key}"))?;
    Ok(rest.trim_start().strip_prefix(':').context("NPY header colon")?.trim_start())
}

fn read_npy(mut r: impl Read) -> Result<Array> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    ensure!(&magic[..6] == b"\x93NUMPY", "not an NPY array");
    let len = if magic[6] == 1 {
        let mut b = [0; 2];
        r.read_exact(&mut b)?;
        u16::from_le_bytes(b) as usize
    } else {
        let mut b = [0; 4];
        r.read_exact(&mut b)?;
        u32::from_le_bytes(b) as usize
    };
    let mut header = vec![0; len];
    r.read_exact(&mut header)?;
    let header = String::from_utf8(header)?;
    ensure!(header_value(&header, "fortran_order")?.starts_with("False"), "C-order arrays required");
    let descr = header_value(&header, "descr")?.trim_start_matches('\'');
    let descr = descr.split('\'').next().context("dtype")?;
    let shape: Vec<usize> = header_value(&header, "shape")?
        .trim_start_matches('(')
        .split(')')
        .next()
        .context("shape")?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    let data = match descr {
        "<f4" => bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4")) as f64).collect(),
        "<f8" => bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().expect("8"))).collect(),
        "|b1" | "|u1" => bytes.iter().map(|&b| b as f64).collect(),
        other => bail!("unsupported NPY dtype {other}"),
    };
    Ok(Array { shape, data })
}

fn read_npz(path: &Path, name: &str) -> Result<Array> {
    let mut zip = zip::ZipArchive::new(std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?)?;
    read_member(&mut zip, name).with_context(|| format!("{}:{name}", path.display()))
}

fn read_member<R: Read + Seek>(zip: &mut zip::ZipArchive<R>, name: &str) -> Result<Array> {
    read_npy(zip.by_name(&format!("{name}.npy"))?)
}

fn f32s(a: &Array) -> Vec<f32> {
    a.data.iter().map(|&v| v as f32).collect()
}

/// `(name, shape, values, as_f64)`; arrays are stored as `<f4` unless `as_f64`.
fn write_npz(path: &Path, arrays: &[(&str, Vec<usize>, Vec<f64>, bool)]) -> Result<()> {
    let tmp = path.with_extension("npz.tmp");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&tmp)?);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, shape, data, as_f64) in arrays {
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        let shape_str = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
        let descr = if *as_f64 { "<f8" } else { "<f4" };
        let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
        // Magic(6) + version(2) + len(2) + header + '\n', padded to 64 bytes.
        let pad = (64 - (10 + header.len() + 1) % 64) % 64;
        header.push_str(&" ".repeat(pad));
        header.push('\n');
        zip.start_file(format!("{name}.npy"), opts)?;
        zip.write_all(b"\x93NUMPY\x01\x00")?;
        zip.write_all(&(header.len() as u16).to_le_bytes())?;
        zip.write_all(header.as_bytes())?;
        for &v in data {
            if *as_f64 {
                zip.write_all(&v.to_le_bytes())?;
            } else {
                zip.write_all(&(v as f32).to_le_bytes())?;
            }
        }
    }
    zip.finish()?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// Pinhole camera from an OpenCV-style `K` and camera-to-world pose.
fn fixture_camera(k: &[f64], c2w: &[f64], size: UVec2) -> Camera {
    let (fx, fy, cx, cy) = (k[0], k[4], k[2], k[5]);
    let col = |j: usize| Vec3::new(c2w[j] as f32, c2w[4 + j] as f32, c2w[8 + j] as f32);
    let rotation = Quat::from_mat3(&Mat3::from_cols(col(0), col(1), col(2))).normalize();
    let position = col(3);
    Camera::new(
        position,
        rotation,
        focal_to_fov(fx, size.x, &CameraModel::Pinhole),
        focal_to_fov(fy, size.y, &CameraModel::Pinhole),
        Vec2::new((cx / size.x as f64) as f32, (cy / size.y as f64) as f32),
        CameraModel::Pinhole,
    )
}

/// Fixture parameter groups in B15's layout (`log_scales` holds the two tangent axes).
#[derive(Clone)]
struct HostParams {
    groups: [Vec<f32>; 5],
    n: usize,
}

const GROUPS: [&str; 5] = ["means", "quats", "log_scales", "raw_opacities", "sh0"];

impl HostParams {
    fn load(params: &Path) -> Result<Self> {
        let mut groups: [Vec<f32>; 5] = Default::default();
        for (g, name) in groups.iter_mut().zip(GROUPS) {
            *g = f32s(&read_npz(params, name)?);
        }
        let n = groups[3].len();
        ensure!(
            groups[0].len() == 3 * n && groups[1].len() == 4 * n && groups[2].len() == 2 * n && groups[4].len() == 3 * n,
            "param shapes disagree"
        );
        Ok(Self { groups, n })
    }

    /// Splats with a tiny third log-scale, which the surfel path ignores.
    fn splats(&self, mode: SplatRenderMode, device: &burn::tensor::Device) -> Splats {
        let n = self.n;
        let [means, quats, ls, raw_opac, sh0] = self.groups.clone();
        let log_scales: Vec<f32> = ls.chunks_exact(2).flat_map(|s| [s[0], s[1], -12.0]).collect();
        Splats::from_tensor_data(
            Tensor::from_data(TensorData::new(means, [n, 3]), device),
            Tensor::from_data(TensorData::new(quats, [n, 4]), device),
            Tensor::from_data(TensorData::new(log_scales, [n, 3]), device),
            Tensor::from_data(TensorData::new(sh0, [n, 1, 3]), device),
            Tensor::from_data(TensorData::new(raw_opac, [n]), device),
            mode,
        )
    }

    fn shape(&self, g: usize) -> Vec<usize> {
        match g {
            0 | 4 => vec![self.n, 3],
            1 => vec![self.n, 4],
            2 => vec![self.n, 2],
            _ => vec![self.n],
        }
    }
}

fn fixture_splats(params: &Path, mode: SplatRenderMode, device: &burn::tensor::Device) -> Result<Splats> {
    Ok(HostParams::load(params)?.splats(mode, device))
}

/// Fixed upstream weights for the audited scalar
/// `sum(rgb * w_rgb) + sum(expected_depth * w_depth)`.
struct Cotangents {
    rgb: Vec<f32>,
    depth: Vec<f32>,
}

/// Deterministic standard normals (xorshift64* + Box-Muller), so audits are reproducible
/// without extra dependencies.
struct Normals(u64);

impl Normals {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// Forward-only objective through the geometry pass (expected depth = depth sum / alpha,
/// zero where alpha <= 1e-3, matching the adapter output).
async fn objective_forward(
    p: &HostParams,
    mode: SplatRenderMode,
    camera: &Camera,
    size: UVec2,
    cot: &Cotangents,
    device: &burn::tensor::Device,
) -> Result<f64> {
    let geo = render_splats_geometry(p.splats(mode, device), camera, size, Vec3::ZERO).await;
    let rgba = geo.img.into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let geom = geo.geom.into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let ch = brush_render::kernels::helpers::GEOM_OUT_CHANNELS as usize;
    let mut obj = 0.0f64;
    for (i, (px, g)) in rgba.chunks_exact(4).zip(geom.chunks_exact(ch)).enumerate() {
        for c in 0..3 {
            obj += px[c] as f64 * cot.rgb[3 * i + c] as f64;
        }
        // Gaussians have no geometry backward: their audit (a harness control) is RGB only.
        if px[3] > 1e-3 && mode == SplatRenderMode::Surfel {
            obj += (g[0] / px[3]) as f64 * cot.depth[i] as f64;
        }
    }
    Ok(obj)
}

/// The same objective through the differentiable surfel/gaussian render, with its gradient
/// for every parameter group (B15 layout).
async fn objective_with_grads(
    p: &HostParams,
    mode: SplatRenderMode,
    camera: &Camera,
    size: UVec2,
    cot: &Cotangents,
    device: &burn::tensor::Device,
    pass: RasterPass,
    with_grads: bool,
) -> Result<(f64, [Vec<f32>; 5])> {
    let splats = p.splats(mode, device).train();
    let diff_device = splats.device();
    let out = brush_render::bwd::render_splats_with_pass(splats.clone(), camera, size, Vec3::ZERO, pass, false).await;
    let (h, w) = (size.y as usize, size.x as usize);
    let rgb = out.img.clone().slice([0..h, 0..w, 0..3]);
    let w_rgb = Tensor::<3>::from_data(TensorData::new(cot.rgb.clone(), [h, w, 3]), &diff_device);
    let mut obj = (rgb * w_rgb).sum();
    if let Some(geom) = out.geom {
        let alpha = out.img.slice([0..h, 0..w, 3..4]);
        let depth_sum = geom.slice([0..h, 0..w, 0..1]);
        let mask = alpha.clone().detach().greater_elem(1e-3).float();
        let expected = depth_sum / alpha.clamp_min(1e-3) * mask;
        let w_depth = Tensor::<3>::from_data(TensorData::new(cot.depth.clone(), [h, w, 1]), &diff_device);
        obj = obj + (expected * w_depth).sum();
    }
    let value = obj.clone().into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?[0] as f64;
    if !with_grads {
        return Ok((value, Default::default()));
    }
    let grads = obj.backward();
    let host = |t: Option<Tensor<2>>| async move {
        match t {
            Some(t) => t.into_data_async().await.map_err(|e| anyhow::anyhow!("{e:?}"))?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}")),
            None => anyhow::Ok(vec![]),
        }
    };
    let n = p.n;
    let gt = host(splats.transforms.grad(&grads)).await?;
    let gt = if gt.is_empty() { vec![0.0; 10 * n] } else { gt };
    let sh = match splats.sh_coeffs.grad(&grads) {
        Some(t) => t.into_data_async().await.map_err(|e| anyhow::anyhow!("{e:?}"))?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?,
        None => vec![0.0; 3 * n],
    };
    let op = match splats.raw_opacities.grad(&grads) {
        Some(t) => t.into_data_async().await.map_err(|e| anyhow::anyhow!("{e:?}"))?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?,
        None => vec![0.0; n],
    };
    let rows = |a: usize, b: usize| -> Vec<f32> { gt.chunks_exact(10).flat_map(|r| r[a..b].to_vec()).collect() };
    Ok((value, [rows(0, 3), rows(3, 7), rows(7, 9), op, sh]))
}

/// `splat surfel-fixture REQUEST.json` (B15 render-adapter/v1).
pub fn run(request: &Path) -> Result<()> {
    let req: serde_json::Value = serde_json::from_slice(&std::fs::read(request)?)?;
    let field = |k: &str| req[k].as_str().map(Path::new).with_context(|| format!("request missing {k}"));
    let (gt, params, output) = (field("gt_camera")?, field("params")?, field("output")?);
    // Optional `render_mode: "gaussian"` renders the same params as flat 3D gaussians.
    let mode = match req["render_mode"].as_str().unwrap_or("surfel") {
        "surfel" => SplatRenderMode::Surfel,
        "gaussian" => SplatRenderMode::Default,
        other => bail!("unknown render_mode {other}"),
    };

    let k = read_npz(gt, "K")?.data;
    let c2w = read_npz(gt, "c2w")?.data;
    let rgb_gt = read_npz(gt, "rgb")?;
    let (h, w) = (rgb_gt.shape[0], rgb_gt.shape[1]);
    let size = UVec2::new(w as u32, h as u32);
    let camera = fixture_camera(&k, &c2w, size);

    let rt = engine::runtime()?;
    let mask = read_npz(gt, "mask")?.data;
    let cot = match req["cotangents"].as_str().map(Path::new) {
        Some(c) => Cotangents { rgb: f32s(&read_npz(c, "rgb")?), depth: f32s(&read_npz(c, "expected_depth")?) },
        // B15's recipe: normals scaled by 1/size (RGB) and by the GT mask / its area (depth).
        None => {
            let mut rng = Normals(req["seed"].as_u64().unwrap_or(42) | 1);
            let fg = mask.iter().filter(|&&m| m > 0.0).count().max(1) as f64;
            let rgb = (0..3 * w * h).map(|_| (rng.next() / (3 * w * h) as f64) as f32).collect();
            let depth = mask.iter().map(|&m| (rng.next() * m / fg) as f32).collect();
            Cotangents { rgb, depth }
        }
    };

    // `adjoint: {eps: [...]}` audits the analytic gradient against central differences of
    // actual renders along one random unit direction per parameter group.
    if let Some(adj) = req.get("adjoint") {
        let eps: Vec<f64> = adj["eps"].as_array().map(|a| a.iter().filter_map(|e| e.as_f64()).collect()).unwrap_or(vec![1e-3, 3e-4, 1e-4]);
        let host = HostParams::load(params)?;
        let report = rt.block_on(async {
            let device = engine::init_device().await;
            // The C^1 smoothstep around the 1/255 cutoff, on both sides of the comparison, and
            // directions supported on a few random splats, so cutoff steps and depth-order
            // swaps between overlapping splats don't swamp the differences.
            let pass = RasterPass::BackwardSmoothCutoff;
            let (analytic_obj, grads) =
                objective_with_grads(&host, mode, &camera, size, &cot, &device, pass, true).await?;
            let base_obj = objective_forward(&host, mode, &camera, size, &cot, &device).await?;
            let mut rng = Normals(adj["seed"].as_u64().unwrap_or(7) | 1);
            let support = adj["splats"].as_u64().unwrap_or(8) as usize;
            let chosen: Vec<usize> = (0..support).map(|_| (rng.uniform() * host.n as f64) as usize % host.n).collect();
            let mut rows = vec![];
            for g in 0..5 {
                let len = host.groups[g].len();
                let width = len / host.n;
                let mut dir = vec![0.0f64; len];
                for &s in &chosen {
                    for c in 0..width {
                        dir[s * width + c] = rng.next();
                    }
                }
                let norm = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
                dir.iter_mut().for_each(|d| *d /= norm);
                let analytic: f64 = grads[g].iter().zip(&dir).map(|(&a, &d)| a as f64 * d).sum();
                for &e in &eps {
                    let mut plus = host.clone();
                    let mut minus = host.clone();
                    for (i, d) in dir.iter().enumerate() {
                        plus.groups[g][i] += (e * d) as f32;
                        minus.groups[g][i] -= (e * d) as f32;
                    }
                    let hi = objective_with_grads(&plus, mode, &camera, size, &cot, &device, pass, false).await?.0;
                    let lo = objective_with_grads(&minus, mode, &camera, size, &cot, &device, pass, false).await?.0;
                    let numerical = (hi - lo) / (2.0 * e);
                    rows.push(serde_json::json!({
                        "parameter": GROUPS[g], "epsilon": e, "analytic_directional": analytic,
                        "numerical_directional": numerical, "absolute_error": (analytic - numerical).abs(),
                        "relative_error": (analytic - numerical).abs() / analytic.abs().max(numerical.abs()).max(1e-8),
                    }));
                }
            }
            anyhow::Ok(serde_json::json!({
                "schema": "g1-surfel-adjoint/v1", "render_mode": format!("{mode:?}"),
                "objective_autodiff": analytic_obj, "objective_forward": base_obj, "support_splats": chosen, "rows": rows,
            }))
        })?;
        std::fs::write(output, serde_json::to_string_pretty(&report)?)?;
        println!("{}", serde_json::to_string(&report)?);
        return Ok(());
    }

    // B15 adapter `gradients: true`: objective plus per-parameter gradients.
    if req["gradients"].as_bool().unwrap_or(false) {
        let host = HostParams::load(params)?;
        let (obj, grads) = rt.block_on(async {
            let device = engine::init_device().await;
            objective_with_grads(&host, mode, &camera, size, &cot, &device, RasterPass::Backward, true).await
        })?;
        let mut arrays = vec![("objective", vec![], vec![obj], true)];
        let names = ["grad_means", "grad_quats", "grad_log_scales", "grad_raw_opacities", "grad_sh0"];
        for (g, name) in names.iter().enumerate() {
            arrays.push((name, host.shape(g), grads[g].iter().map(|&v| v as f64).collect(), false));
        }
        write_npz(output, &arrays)?;
        println!("{}", serde_json::json!({"output": output, "objective": obj}));
        return Ok(());
    }

    // `rgb_only: true` renders through the production float RGB path (no geometry pass) and
    // prints a hash of the raw image, for byte-level comparisons between builds.
    if req["rgb_only"].as_bool().unwrap_or(false) {
        let rgba = rt.block_on(async {
            let device = engine::init_device().await;
            let splats = fixture_splats(params, mode, &device)?;
            let (img, _) = render_splats(splats, &camera, size, Vec3::ZERO, None, TextureMode::Float).await;
            img.into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))
        })?;
        use sha2::Digest;
        let digest = sha2::Sha256::digest(rgba.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        let rgba64: Vec<f64> = rgba.iter().map(|&v| v as f64).collect();
        write_npz(output, &[("rgba", vec![h, w, 4], rgba64, false)])?;
        println!("{}", serde_json::json!({"output": output, "rgba_sha256": hex}));
        return Ok(());
    }
    let (rgba, geom) = rt.block_on(async {
        let device = engine::init_device().await;
        let splats = fixture_splats(params, mode, &device)?;
        let geo = render_splats_geometry(splats, &camera, size, Vec3::ZERO).await;
        let rgba = geo.img.into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let geom = geo.geom.into_data_async().await?.try_into_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        anyhow::Ok((rgba, geom))
    })?;

    let channels = brush_render::kernels::helpers::GEOM_OUT_CHANNELS as usize;
    let n = w * h;
    let mut rgb = Vec::with_capacity(3 * n);
    let mut alpha = Vec::with_capacity(n);
    let mut expected = Vec::with_capacity(n);
    let mut median = Vec::with_capacity(n);
    let mut normal = Vec::with_capacity(3 * n);
    for (px, g) in rgba.chunks_exact(4).zip(geom.chunks_exact(channels)) {
        rgb.extend_from_slice(&px[..3]);
        alpha.push(px[3]);
        let hit = px[3] > 1e-3;
        expected.push(if hit { g[0] / px[3] } else { 0.0 });
        median.push(if hit { g[1] } else { 0.0 });
        let nv = Vec3::new(g[2], g[3], g[4]).normalize_or_zero();
        normal.extend_from_slice(&nv.to_array());
    }

    let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let mut arrays = vec![
        ("rgb", vec![h, w, 3], f(&rgb), false),
        ("alpha", vec![h, w], f(&alpha), false),
        ("expected_depth", vec![h, w], f(&expected), false),
        ("median_depth", vec![h, w], f(&median), false),
        ("normal", vec![h, w, 3], f(&normal), false),
    ];
    if let Some(cot) = req["cotangents"].as_str().map(Path::new) {
        let c_rgb = read_npz(cot, "rgb")?.data;
        let c_depth = read_npz(cot, "expected_depth")?.data;
        let objective: f64 = rgb.iter().zip(&c_rgb).map(|(&a, &b)| a as f64 * b).sum::<f64>()
            + expected.iter().zip(&c_depth).map(|(&a, &b)| a as f64 * b).sum::<f64>();
        arrays.push(("objective", vec![], vec![objective], true));
    }
    write_npz(output, &arrays)?;
    println!("{}", serde_json::json!({"output": output, "width": w, "height": h}));
    Ok(())
}
