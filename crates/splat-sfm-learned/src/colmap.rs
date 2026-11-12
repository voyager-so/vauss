//! COLMAP binary sparse-model reader and the camera graph's weakest cut, for the guard
//! ([`crate::guard_fallback`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, bail, ensure};

const MIN_SIDE: usize = 3;

/// The weakest camera-graph cut.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Cut {
    pub conductance: f64,
    /// Shared-point pair weight crossing the cut.
    pub cross_weight: f64,
    /// Cameras on the smaller side, sorted.
    pub names: Vec<String>,
    /// Smallest eigenvalues of the normalised Laplacian (up to five).
    pub eigenvalues: Vec<f64>,
}

/// Minimum-conductance sweep cut, or `None` for an unsupported graph.
pub(crate) fn weakest_cut(m: &Model) -> Option<Cut> {
    let ids: Vec<u32> = m.images.keys().copied().collect();
    let n = ids.len();
    let index: BTreeMap<u32, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut w = vec![0.0; n * n];
    for track in m.tracks.values() {
        let cams: Vec<usize> = track.iter().map(|id| index[id]).collect::<BTreeSet<_>>().into_iter().collect();
        for (a, &i) in cams.iter().enumerate() {
            for &j in &cams[a + 1..] {
                w[i * n + j] += 1.0;
                w[j * n + i] += 1.0;
            }
        }
    }
    let d: Vec<f64> = (0..n).map(|i| w[i * n..(i + 1) * n].iter().sum()).collect();
    if n < 6 || d.iter().any(|&x| x <= 0.0) {
        return None;
    }
    let laplacian = (0..n * n)
        .map(|k| {
            let (i, j) = (k / n, k % n);
            f64::from(u8::from(i == j)) - w[k] / (d[i] * d[j]).sqrt()
        })
        .collect();
    let (values, vectors) = symmetric_eigen(laplacian, n);
    let mut by_value: Vec<usize> = (0..n).collect();
    by_value.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let fiedler = by_value[1];
    let key: Vec<f64> = (0..n).map(|i| vectors[i * n + fiedler] / d[i].sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| key[a].total_cmp(&key[b]));

    // Sweep: side A = order[..k]. Weights are integers, so the running sums are exact.
    let mut in_a = vec![false; n];
    let (mut cross, mut deg_a) = (0.0, 0.0);
    let deg_total: f64 = d.iter().sum();
    let mut best: Option<(f64, f64, usize)> = None;
    for k in 1..=n - MIN_SIDE {
        let v = order[k - 1];
        let (to_a, to_b) = (0..n).filter(|&j| j != v).fold((0.0, 0.0), |(a, b), j| {
            if in_a[j] { (a + w[v * n + j], b) } else { (a, b + w[v * n + j]) }
        });
        in_a[v] = true;
        cross += to_b - to_a;
        deg_a += d[v];
        if k >= MIN_SIDE {
            let conductance = cross / deg_a.min(deg_total - deg_a);
            // First minimum wins, as Python's min().
            if best.is_none_or(|(c, _, _)| conductance < c) {
                best = Some((conductance, cross, k));
            }
        }
    }
    let (conductance, cross_weight, k) = best?;
    // Python: the smaller side is A only if strictly smaller.
    let small = if k < n - k { &order[..k] } else { &order[k..] };
    let mut names: Vec<String> = small.iter().map(|&i| m.images[&ids[i]].name.clone()).collect();
    names.sort();
    let eigenvalues = by_value.iter().take(5).map(|&i| values[i]).collect();
    Some(Cut { conductance, cross_weight, names, eigenvalues })
}

/// Cyclic Jacobi eigen-decomposition of a symmetric row-major `n x n` matrix: (eigenvalues,
/// eigenvectors as the columns of a row-major matrix).
fn symmetric_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v: Vec<f64> = (0..n * n).map(|k| f64::from(u8::from(k / n == k % n))).collect();
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|p| (p + 1..n).map(move |q| (p, q))).map(|(p, q)| a[p * n + q].powi(2)).sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for k in 0..n {
                    let (x, y) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * x - s * y;
                    a[k * n + q] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * x - s * y;
                    a[q * n + k] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * x - s * y;
                    v[k * n + q] = s * x + c * y;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// Unit (w, x, y, z) quaternion to a rotation matrix.
fn quat_to_rotation(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let [w, x, y, z] = q.map(|v| v / n);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

pub(crate) struct Camera {
    model: i32,
    /// In COLMAP order for `model` (SIMPLE_RADIAL: f, cx, cy, k).
    params: Vec<f64>,
}

impl Camera {
    /// Pixel of a camera-frame point (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV).
    pub(crate) fn project(&self, c: [f64; 3]) -> [f64; 2] {
        let (u, v) = (c[0] / c[2], c[1] / c[2]);
        let p = &self.params[..];
        match self.model {
            SIMPLE_PINHOLE => [p[0] * u + p[1], p[0] * v + p[2]],
            PINHOLE => [p[0] * u + p[2], p[1] * v + p[3]],
            SIMPLE_RADIAL => {
                let radial = p[3] * (u * u + v * v);
                [p[0] * (u + u * radial) + p[1], p[0] * (v + v * radial) + p[2]]
            }
            RADIAL => {
                let r2 = u * u + v * v;
                let radial = p[3] * r2 + p[4] * r2 * r2;
                [p[0] * (u + u * radial) + p[1], p[0] * (v + v * radial) + p[2]]
            }
            OPENCV => {
                let (k1, k2, p1, p2) = (p[4], p[5], p[6], p[7]);
                let (uu, vv, uv, r2) = (u * u, v * v, u * v, u * u + v * v);
                let radial = k1 * r2 + k2 * r2 * r2;
                let du = u * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu);
                let dv = v * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv);
                [p[0] * (u + du) + p[2], p[1] * (v + dv) + p[3]]
            }
            _ => unreachable!("camera model checked on read"),
        }
    }
}

pub(crate) struct Image {
    /// World-to-camera rotation (w, x, y, z) and translation.
    pub(crate) q: [f64; 4],
    pub(crate) t: [f64; 3],
    pub(crate) camera_id: u32,
    pub(crate) name: String,
    /// Observations of 3D points: (point2D index, pixel, point3D id).
    pub(crate) points2d: Vec<(u32, [f64; 2], u64)>,
}

/// A COLMAP binary sparse model with SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL or OPENCV
/// cameras.
pub(crate) struct Model {
    pub(crate) cameras: BTreeMap<u32, Camera>,
    pub(crate) images: BTreeMap<u32, Image>,
    pub(crate) points: BTreeMap<u64, [f64; 3]>,
    /// Point id -> observing image ids.
    tracks: BTreeMap<u64, Vec<u32>>,
}

impl Model {
    /// Squared reprojection error of every observation of a 3D point, keyed by (image name,
    /// point2D index): the same keypoint in any model reconstructed from the same database.
    pub(crate) fn squared_errors_by_name(&self) -> BTreeMap<(&str, u32), f64> {
        let mut errors = BTreeMap::new();
        for im in self.images.values() {
            let cam = &self.cameras[&im.camera_id];
            let r = quat_to_rotation(im.q);
            for &(index, uv, point) in &im.points2d {
                let Some(x) = self.points.get(&point) else { continue };
                let c: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| r[i][j] * x[j]).sum::<f64>() + im.t[i]);
                let [px, py] = cam.project(c);
                errors.insert((im.name.as_str(), index), (px - uv[0]).powi(2) + (py - uv[1]).powi(2));
            }
        }
        errors
    }
}

const SIMPLE_PINHOLE: i32 = 0;
const PINHOLE: i32 = 1;
const SIMPLE_RADIAL: i32 = 2;
const RADIAL: i32 = 3;
const OPENCV: i32 = 4;
const INVALID_POINT: u64 = u64::MAX;

/// Read `cameras.bin`, `images.bin` and `points3D.bin` from `dir`.
pub(crate) fn read_model(dir: &Path) -> anyhow::Result<Model> {
    let read = |file: &str| {
        let path = dir.join(file);
        std::fs::read(&path).with_context(|| format!("reading {}", path.display())).map(|b| Reader(b, 0))
    };
    let mut r = read("cameras.bin")?;
    let mut cameras = BTreeMap::new();
    for _ in 0..r.u64()? {
        let (id, model) = (r.u32()?, r.i32()?);
        r.skip(8 + 8)?; // width, height
        let n_params = match model {
            SIMPLE_PINHOLE => 3,
            PINHOLE | SIMPLE_RADIAL => 4,
            RADIAL => 5,
            OPENCV => 8,
            _ => bail!("camera {id}: unsupported camera model {model}"),
        };
        let params = (0..n_params).map(|_| r.f64()).collect::<anyhow::Result<_>>()?;
        cameras.insert(id, Camera { model, params });
    }
    let mut r = read("images.bin")?;
    let mut images = BTreeMap::new();
    for _ in 0..r.u64()? {
        let id = r.u32()?;
        let q = [r.f64()?, r.f64()?, r.f64()?, r.f64()?];
        let t = [r.f64()?, r.f64()?, r.f64()?];
        let camera_id = r.u32()?;
        ensure!(cameras.contains_key(&camera_id), "image {id}: unknown camera {camera_id}");
        let name = r.name()?;
        let mut points2d = Vec::new();
        for index in 0..u32::try_from(r.u64()?)? {
            let (uv, point) = ([r.f64()?, r.f64()?], r.u64()?);
            if point != INVALID_POINT {
                points2d.push((index, uv, point));
            }
        }
        images.insert(id, Image { q, t, camera_id, name, points2d });
    }
    let mut r = read("points3D.bin")?;
    let (mut points, mut tracks) = (BTreeMap::new(), BTreeMap::new());
    for _ in 0..r.u64()? {
        let id = r.u64()?;
        points.insert(id, [r.f64()?, r.f64()?, r.f64()?]);
        r.skip(3 + 8)?; // rgb, error
        let track = (0..r.u64()?)
            .map(|_| -> anyhow::Result<u32> {
                let image = r.u32()?;
                r.skip(4)?; // point2D index
                ensure!(images.contains_key(&image), "point {id}: unknown image {image}");
                Ok(image)
            })
            .collect::<anyhow::Result<_>>()?;
        tracks.insert(id, track);
    }
    Ok(Model { cameras, images, points, tracks })
}

/// Little-endian cursor over a COLMAP binary file.
struct Reader(Vec<u8>, usize);

impl Reader {
    fn take<const N: usize>(&mut self) -> anyhow::Result<[u8; N]> {
        let Some(bytes) = self.0.get(self.1..self.1 + N) else { bail!("truncated COLMAP binary model") };
        self.1 += N;
        Ok(bytes.try_into()?)
    }
    fn skip(&mut self, n: usize) -> anyhow::Result<()> {
        ensure!(self.1 + n <= self.0.len(), "truncated COLMAP binary model");
        self.1 += n;
        Ok(())
    }
    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn i32(&mut self) -> anyhow::Result<i32> {
        Ok(i32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn f64(&mut self) -> anyhow::Result<f64> {
        Ok(f64::from_le_bytes(self.take()?))
    }
    fn name(&mut self) -> anyhow::Result<String> {
        let end = self.0[self.1..].iter().position(|&b| b == 0).context("unterminated image name")?;
        let name = String::from_utf8(self.0[self.1..self.1 + end].to_vec())?;
        self.1 += end + 1;
        Ok(name)
    }
}