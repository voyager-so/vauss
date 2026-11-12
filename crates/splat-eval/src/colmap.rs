//! COLMAP sparse model reader (cameras/images/points3D, .bin or .txt).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

/// COLMAP camera models by id: (name, number of params).
pub const CAMERA_MODELS: [(&str, usize); 12] = [
    ("SIMPLE_PINHOLE", 3),
    ("PINHOLE", 4),
    ("SIMPLE_RADIAL", 4),
    ("RADIAL", 5),
    ("OPENCV", 8),
    ("OPENCV_FISHEYE", 8),
    ("FULL_OPENCV", 12),
    ("FOV", 5),
    ("SIMPLE_RADIAL_FISHEYE", 4),
    ("RADIAL_FISHEYE", 5),
    ("THIN_PRISM_FISHEYE", 12),
    ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
];

#[derive(Clone, Debug, Serialize)]
pub struct Camera {
    pub id: u32,
    pub model: String, // "PINHOLE", "OPENCV", ...
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Image {
    pub id: u32,
    pub name: String,
    /// World-to-camera rotation as (w, x, y, z) and translation.
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub camera_id: u32,
    pub num_points2d: usize,
    /// Observations that have a 3D point (point3D_id != -1): pixel xy and point id.
    pub obs_xy: Vec<[f64; 2]>,
    pub obs_point_id: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct Point3D {
    pub id: u64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    /// COLMAP's stored per-point mean reprojection error (Spirula writes 0 for every point).
    pub error: f64,
    pub track_len: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub cameras: BTreeMap<u32, Camera>,
    pub images: BTreeMap<u32, Image>,
    pub points: Vec<Point3D>,
}

/// Read `<dir>/{cameras,images,points3D}.bin`, falling back to the .txt files.
pub fn read_model(sparse_dir: &Path) -> Result<Model> {
    let model = if sparse_dir.join("images.bin").exists() {
        Model {
            cameras: parse_file(sparse_dir, "cameras.bin", |b| cameras_bin(&mut Bytes(b)))?,
            images: parse_file(sparse_dir, "images.bin", |b| images_bin(&mut Bytes(b)))?,
            points: parse_file(sparse_dir, "points3D.bin", |b| points_bin(&mut Bytes(b)))?,
        }
    } else {
        Model {
            cameras: parse_file(sparse_dir, "cameras.txt", |b| cameras_txt(utf8(b)?))?,
            images: parse_file(sparse_dir, "images.txt", |b| images_txt(utf8(b)?))?,
            points: parse_file(sparse_dir, "points3D.txt", |b| points_txt(utf8(b)?))?,
        }
    };
    Ok(model)
}

impl Model {
    /// Registered image names, sorted.
    pub fn registered_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.images.values().map(|im| im.name.clone()).collect();
        names.sort();
        names
    }

    /// World-to-camera rotation matrix of an image.
    pub fn rotation(q: [f64; 4]) -> [[f64; 3]; 3] {
        let [w, x, y, z] = q;
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * w * z, 2.0 * x * z + 2.0 * w * y],
            [2.0 * x * y + 2.0 * w * z, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * w * x],
            [2.0 * x * z - 2.0 * w * y, 2.0 * y * z + 2.0 * w * x, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ]
    }

    /// Camera centre in world coordinates: -R^T t.
    pub fn center(image: &Image) -> [f64; 3] {
        let r = Self::rotation(image.qvec);
        let t = image.tvec;
        std::array::from_fn(|j| -(r[0][j] * t[0] + r[1][j] * t[1] + r[2][j] * t[2]))
    }
}

fn parse_file<T>(dir: &Path, file: &str, parse: impl FnOnce(&[u8]) -> Result<T>) -> Result<T> {
    let path = dir.join(file);
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    parse(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn utf8(b: &[u8]) -> Result<&str> {
    Ok(std::str::from_utf8(b)?)
}

/// Little-endian cursor over a whole binary file (the files are tens of MB at most).
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let Some((head, rest)) = self.0.split_first_chunk::<N>() else {
            bail!("unexpected end of file");
        };
        self.0 = rest;
        Ok(*head)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take::<1>()?[0])
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take()?))
    }
    fn f64s<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut out = [0.0; N];
        for v in &mut out {
            *v = self.f64()?;
        }
        Ok(out)
    }
    fn id(&mut self) -> Result<u32> {
        let v = self.i32()?;
        u32::try_from(v).with_context(|| format!("negative id {v}"))
    }
    fn count(&mut self) -> Result<usize> {
        Ok(usize::try_from(self.u64()?)?)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        if n > self.0.len() {
            bail!("unexpected end of file");
        }
        self.0 = &self.0[n..];
        Ok(())
    }
    fn cstr(&mut self) -> Result<String> {
        let end = self.0.iter().position(|&c| c == 0).context("unterminated image name")?;
        let s = String::from_utf8(self.0[..end].to_vec()).context("image name is not UTF-8")?;
        self.0 = &self.0[end + 1..];
        Ok(s)
    }
}

fn cameras_bin(b: &mut Bytes) -> Result<BTreeMap<u32, Camera>> {
    let mut cams = BTreeMap::new();
    for _ in 0..b.count()? {
        let id = b.id()?;
        let model_id = b.i32()?;
        let &(model, n) = usize::try_from(model_id)
            .ok()
            .and_then(|m| CAMERA_MODELS.get(m))
            .with_context(|| format!("camera {id}: unknown model id {model_id}"))?;
        let (width, height) = (b.u64()?, b.u64()?);
        let params = (0..n).map(|_| b.f64()).collect::<Result<_>>()?;
        cams.insert(id, Camera { id, model: model.into(), width, height, params });
    }
    Ok(cams)
}

fn images_bin(b: &mut Bytes) -> Result<BTreeMap<u32, Image>> {
    let mut imgs = BTreeMap::new();
    for _ in 0..b.count()? {
        let id = b.id()?;
        let qvec = b.f64s()?;
        let tvec = b.f64s()?;
        let camera_id = b.id()?;
        let name = b.cstr()?;
        let num_points2d = b.count()?;
        let cap = num_points2d.min(b.0.len() / 24);
        let (mut obs_xy, mut obs_point_id) = (Vec::with_capacity(cap), Vec::with_capacity(cap));
        for _ in 0..num_points2d {
            let xy = b.f64s()?;
            let pid = b.i64()?;
            if pid >= 0 {
                obs_xy.push(xy);
                obs_point_id.push(pid as u64);
            }
        }
        imgs.insert(id, Image { id, name, qvec, tvec, camera_id, num_points2d, obs_xy, obs_point_id });
    }
    Ok(imgs)
}

/// One images.bin record with every 2D point (`Image` keeps only the triangulated ones).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageRecord {
    pub id: u32,
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub camera_id: u32,
    pub name: String,
    /// (x, y, point3D_id); -1 = untriangulated.
    pub points2d: Vec<([f64; 2], i64)>,
}

/// Every record of a binary images.bin, in file order.
pub fn read_images_bin(path: &Path) -> Result<Vec<ImageRecord>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let b = &mut Bytes(&bytes);
    let n = b.count()?;
    let mut out = Vec::with_capacity(n.min(bytes.len() / 64));
    for _ in 0..n {
        let id = b.id()?;
        let qvec = b.f64s()?;
        let tvec = b.f64s()?;
        let camera_id = b.id()?;
        let name = b.cstr()?;
        let count = b.count()?;
        let mut points2d = Vec::with_capacity(count.min(b.0.len() / 24));
        for _ in 0..count {
            points2d.push((b.f64s()?, b.i64()?));
        }
        out.push(ImageRecord { id, qvec, tvec, camera_id, name, points2d });
    }
    ensure!(b.0.is_empty(), "{}: trailing bytes", path.display());
    Ok(out)
}

/// COLMAP's binary images.bin layout (also what 4.2.1 writes): u64 count; per image i32 id,
/// 4 f64 qvec, 3 f64 tvec, i32 camera id, NUL-terminated name, u64 point count, then
/// (f64 x, f64 y, i64 point3D id) per point.
pub fn write_images_bin(path: &Path, images: &[ImageRecord]) -> Result<()> {
    let mut out = Vec::new();
    out.extend((images.len() as u64).to_le_bytes());
    for im in images {
        ensure!(!im.name.as_bytes().contains(&0), "image name contains NUL: {:?}", im.name);
        out.extend(i32::try_from(im.id)?.to_le_bytes());
        im.qvec.iter().chain(&im.tvec).for_each(|v| out.extend(v.to_le_bytes()));
        out.extend(i32::try_from(im.camera_id)?.to_le_bytes());
        out.extend(im.name.as_bytes());
        out.push(0);
        out.extend((im.points2d.len() as u64).to_le_bytes());
        for ([x, y], id) in &im.points2d {
            out.extend(x.to_le_bytes());
            out.extend(y.to_le_bytes());
            out.extend(id.to_le_bytes());
        }
    }
    write_atomic(path, &out)
}

/// COLMAP's binary cameras.bin layout: u64 count; per camera i32 id, i32 model id, u64 width,
/// u64 height, then the model's parameters as f64.
pub fn write_cameras_bin(path: &Path, cameras: &BTreeMap<u32, Camera>) -> Result<()> {
    let mut out = Vec::new();
    out.extend((cameras.len() as u64).to_le_bytes());
    for cam in cameras.values() {
        let model = CAMERA_MODELS.iter().position(|(name, _)| *name == cam.model).with_context(|| format!("camera {}: unknown model {}", cam.id, cam.model))?;
        ensure!(cam.params.len() == CAMERA_MODELS[model].1, "camera {}: {} has {} params, not {}", cam.id, cam.model, cam.params.len(), CAMERA_MODELS[model].1);
        out.extend(i32::try_from(cam.id)?.to_le_bytes());
        out.extend((model as i32).to_le_bytes());
        out.extend(cam.width.to_le_bytes());
        out.extend(cam.height.to_le_bytes());
        cam.params.iter().for_each(|v| out.extend(v.to_le_bytes()));
    }
    write_atomic(path, &out)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
}

/// Models whose first parameter is one focal length for both axes (f, cx, cy, ...); all others start fx, fy, cx, cy.
const SINGLE_FOCAL: [&str; 5] = ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"];

/// The same camera for images resized to `width` x `height`: focal length(s) and principal point scale per
/// axis (one focal length takes the x scale); distortion terms are dimensionless and stay.
pub fn scale_camera(cam: &Camera, width: u64, height: u64) -> Result<Camera> {
    ensure!(cam.width > 0 && cam.height > 0 && width > 0 && height > 0, "camera {}: zero size", cam.id);
    let (sx, sy) = (width as f64 / cam.width as f64, height as f64 / cam.height as f64);
    let mut params = cam.params.clone();
    let scales: &[f64] = if SINGLE_FOCAL.contains(&cam.model.as_str()) { &[sx, sx, sy] } else { &[sx, sy, sx, sy] };
    ensure!(params.len() >= scales.len(), "camera {}: {} has too few params", cam.id, cam.model);
    params.iter_mut().zip(scales).for_each(|(p, s)| *p *= s);
    Ok(Camera { width, height, params, ..cam.clone() })
}

/// Rewrite a binary model (`cameras.bin`, `images.bin`; points3D.bin has no pixel data) for images
/// resized to `sizes[camera id]`. COLMAP pixel coordinates start at the image corner, so scaling them
/// is exact. Cameras absent from `sizes` or already that size are unchanged. Returns the cameras rescaled.
pub fn rescale_model_bin(sparse_dir: &Path, sizes: &BTreeMap<u32, (u64, u64)>) -> Result<usize> {
    let cams = parse_file(sparse_dir, "cameras.bin", |b| cameras_bin(&mut Bytes(b)))?;
    let mut scaled = BTreeMap::new();
    let mut factors = BTreeMap::new();
    for (id, cam) in &cams {
        match sizes.get(id) {
            Some(&(w, h)) if (w, h) != (cam.width, cam.height) => {
                factors.insert(*id, (w as f64 / cam.width as f64, h as f64 / cam.height as f64));
                scaled.insert(*id, scale_camera(cam, w, h)?);
            }
            _ => {
                scaled.insert(*id, cam.clone());
            }
        }
    }
    if factors.is_empty() {
        return Ok(0);
    }
    let images_path = sparse_dir.join("images.bin");
    let mut images = read_images_bin(&images_path)?;
    for im in &mut images {
        if let Some(&(sx, sy)) = factors.get(&im.camera_id) {
            for (xy, _) in &mut im.points2d {
                xy[0] *= sx;
                xy[1] *= sy;
            }
        }
    }
    write_images_bin(&images_path, &images)?;
    write_cameras_bin(&sparse_dir.join("cameras.bin"), &scaled)?;
    Ok(factors.len())
}

fn points_bin(b: &mut Bytes) -> Result<Vec<Point3D>> {
    let n = b.count()?;
    let mut pts = Vec::with_capacity(n.min(b.0.len() / 51));
    for _ in 0..n {
        let id = b.u64()?;
        let xyz = b.f64s()?;
        let rgb = [b.u8()?, b.u8()?, b.u8()?];
        let error = b.f64()?;
        let track_len = b.count()?;
        b.skip(track_len.checked_mul(8).context("track length overflow")?)?;
        pts.push(Point3D { id, xyz, rgb, error, track_len });
    }
    Ok(pts)
}

/// Non-comment lines with their 1-based line numbers. Blank lines are kept: in images.txt an
/// image without observations has an empty second line.
fn data_lines(s: &str) -> impl Iterator<Item = (usize, &str)> {
    s.lines().enumerate().map(|(i, l)| (i + 1, l)).filter(|(_, l)| !l.starts_with('#'))
}

fn parse<T: std::str::FromStr>(tok: Option<&str>, what: &str, line: usize) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let tok = tok.with_context(|| format!("line {line}: missing {what}"))?;
    tok.parse().with_context(|| format!("line {line}: bad {what} {tok:?}"))
}

fn cameras_txt(s: &str) -> Result<BTreeMap<u32, Camera>> {
    let mut cams = BTreeMap::new();
    for (ln, l) in data_lines(s).filter(|(_, l)| !l.trim().is_empty()) {
        let mut t = l.split_whitespace();
        let id = parse(t.next(), "camera id", ln)?;
        let model = t.next().with_context(|| format!("line {ln}: missing model"))?.to_string();
        let width = parse(t.next(), "width", ln)?;
        let height = parse(t.next(), "height", ln)?;
        let params = t.map(|v| parse(Some(v), "param", ln)).collect::<Result<_>>()?;
        cams.insert(id, Camera { id, model, width, height, params });
    }
    Ok(cams)
}

fn images_txt(s: &str) -> Result<BTreeMap<u32, Image>> {
    let mut imgs = BTreeMap::new();
    let mut lines = data_lines(s);
    while let Some((ln, l)) = lines.next() {
        if l.trim().is_empty() {
            continue;
        }
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() < 10 {
            bail!("line {ln}: expected IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME");
        }
        let f = |i: usize| parse::<f64>(Some(t[i]), "pose value", ln);
        let (id, camera_id) = (parse(Some(t[0]), "image id", ln)?, parse(Some(t[8]), "camera id", ln)?);
        let qvec = [f(1)?, f(2)?, f(3)?, f(4)?];
        let tvec = [f(5)?, f(6)?, f(7)?];
        let (pln, points) = lines.next().unwrap_or((ln + 1, ""));
        let v: Vec<&str> = points.split_whitespace().collect();
        if !v.len().is_multiple_of(3) {
            bail!("line {pln}: POINTS2D is not a list of (X, Y, POINT3D_ID)");
        }
        let (mut obs_xy, mut obs_point_id) = (Vec::new(), Vec::new());
        for c in v.chunks_exact(3) {
            let pid: i64 = parse(Some(c[2]), "point3D id", pln)?;
            if pid >= 0 {
                obs_xy.push([parse(Some(c[0]), "x", pln)?, parse(Some(c[1]), "y", pln)?]);
                obs_point_id.push(pid as u64);
            }
        }
        let name = t[9..].join(" ");
        let num_points2d = v.len() / 3;
        imgs.insert(id, Image { id, name, qvec, tvec, camera_id, num_points2d, obs_xy, obs_point_id });
    }
    Ok(imgs)
}

fn points_txt(s: &str) -> Result<Vec<Point3D>> {
    let mut pts = Vec::new();
    for (ln, l) in data_lines(s).filter(|(_, l)| !l.trim().is_empty()) {
        let t: Vec<&str> = l.split_whitespace().collect();
        if t.len() < 8 {
            bail!("line {ln}: expected POINT3D_ID X Y Z R G B ERROR TRACK[]");
        }
        let f = |i: usize| parse::<f64>(Some(t[i]), "coordinate", ln);
        let u = |i: usize| parse::<u8>(Some(t[i]), "colour", ln);
        pts.push(Point3D {
            id: parse(Some(t[0]), "point id", ln)?,
            xyz: [f(1)?, f(2)?, f(3)?],
            rgb: [u(4)?, u(5)?, u(6)?],
            error: parse(Some(t[7]), "error", ln)?,
            track_len: (t.len() - 8) / 2,
        });
    }
    Ok(pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    }

    #[test]
    fn rotation_identity_and_axis() {
        assert_eq!(Model::rotation([1.0, 0.0, 0.0, 0.0]), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        // 90 degrees about z: x -> y.
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let r = Model::rotation([h, 0.0, 0.0, h]);
        let want = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((r[i][j] - want[i][j]).abs() < 1e-12, "{r:?}");
            }
        }
    }

    #[test]
    fn rotation_is_orthonormal() {
        let n = (0.3f64 * 0.3 + 0.1 * 0.1 + 0.5 * 0.5 + 0.8 * 0.8).sqrt();
        let r = Model::rotation([0.8 / n, 0.3 / n, -0.1 / n, 0.5 / n]);
        let rt = std::array::from_fn(|i| std::array::from_fn(|j| r[j][i]));
        let p = matmul(r, rt);
        for (i, row) in p.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!((v - f64::from(u8::from(i == j))).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn center_is_minus_rt_t() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let im = Image {
            id: 1,
            name: "a.png".into(),
            qvec: [h, 0.0, 0.0, h],
            tvec: [1.0, 2.0, 3.0],
            camera_id: 1,
            num_points2d: 0,
            obs_xy: vec![],
            obs_point_id: vec![],
        };
        // R^T t with R = rot_z(90): (2, -1, 3).
        let c = Model::center(&im);
        for (a, b) in c.iter().zip([-2.0, 1.0, -3.0]) {
            assert!((a - b).abs() < 1e-12, "{c:?}");
        }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("splat-eval-colmap-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fixture() -> (BTreeMap<u32, Camera>, Vec<ImageRecord>) {
        let cams = BTreeMap::from([
            (1, Camera { id: 1, model: "PINHOLE".into(), width: 2400, height: 1600, params: vec![1800.0, 1795.0, 1201.5, 799.25] }),
            (3, Camera { id: 3, model: "SIMPLE_RADIAL".into(), width: 1500, height: 2400, params: vec![2000.0, 750.0, 1200.0, -0.05] }),
        ]);
        let images = vec![
            ImageRecord { id: 2, qvec: [0.9, 0.1, -0.2, 0.3], tvec: [1.0, -2.0, 0.5], camera_id: 1, name: "frame_000002.jpg".into(),
                points2d: vec![([12.5, 1599.0], 7), ([2399.75, 0.25], -1)] },
            ImageRecord { id: 5, qvec: [1.0, 0.0, 0.0, 0.0], tvec: [0.0; 3], camera_id: 3, name: "sub dir/frame_000005.jpg".into(), points2d: vec![] },
            ImageRecord { id: 9, qvec: [0.5; 4], tvec: [3.0, 2.0, 1.0], camera_id: 3, name: "frame_000009.jpg".into(), points2d: vec![([750.0, 1200.0], 7)] },
        ];
        (cams, images)
    }

    #[test]
    fn binary_writer_round_trips() {
        let dir = scratch("roundtrip");
        let (cams, images) = fixture();
        write_cameras_bin(&dir.join("cameras.bin"), &cams).unwrap();
        write_images_bin(&dir.join("images.bin"), &images).unwrap();
        // Empty points3D.bin so read_model accepts the dir.
        std::fs::write(dir.join("points3D.bin"), 0u64.to_le_bytes()).unwrap();
        assert_eq!(read_images_bin(&dir.join("images.bin")).unwrap(), images);
        let model = read_model(&dir).unwrap();
        for (id, cam) in &cams {
            let got = &model.cameras[id];
            assert_eq!((got.model.as_str(), got.width, got.height, &got.params), (cam.model.as_str(), cam.width, cam.height, &cam.params));
        }
        assert_eq!(model.images[&2].obs_xy, vec![[12.5, 1599.0]]);
        assert_eq!(model.images[&2].num_points2d, 2);
        assert_eq!(model.images[&5].name, "sub dir/frame_000005.jpg");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rescale_scales_intrinsics_and_keypoints_per_axis() {
        let dir = scratch("rescale");
        let (cams, images) = fixture();
        write_cameras_bin(&dir.join("cameras.bin"), &cams).unwrap();
        write_images_bin(&dir.join("images.bin"), &images).unwrap();
        // Camera 1: 2400x1600 -> 1600x1067 (sx 2/3, sy 1067/1600); camera 3 already at its size.
        let n = rescale_model_bin(&dir, &BTreeMap::from([(1, (1600, 1067)), (3, (1500, 2400))])).unwrap();
        assert_eq!(n, 1);
        let (sx, sy) = (1600.0 / 2400.0, 1067.0 / 1600.0);
        let cams2 = parse_file(&dir, "cameras.bin", |b| cameras_bin(&mut Bytes(b))).unwrap();
        assert_eq!((cams2[&1].width, cams2[&1].height), (1600, 1067));
        let want = [1800.0 * sx, 1795.0 * sy, 1201.5 * sx, 799.25 * sy];
        for (a, b) in cams2[&1].params.iter().zip(want) {
            assert!((a - b).abs() < 1e-9, "{:?}", cams2[&1].params);
        }
        assert_eq!(cams2[&3].params, cams[&3].params);
        let im = read_images_bin(&dir.join("images.bin")).unwrap();
        assert_eq!(im[0].points2d, vec![([12.5 * sx, 1599.0 * sy], 7), ([2399.75 * sx, 0.25 * sy], -1)]);
        assert_eq!(im[2], images[2]);
        // A point projected through the old camera lands at the scaled pixel of the new one.
        let project = |c: &Camera, p: [f64; 3]| [c.params[0] * p[0] / p[2] + c.params[2], c.params[1] * p[1] / p[2] + c.params[3]];
        let (a, b) = (project(&cams[&1], [0.3, -0.2, 2.0]), project(&cams2[&1], [0.3, -0.2, 2.0]));
        assert!((b[0] - a[0] * sx).abs() < 1e-9 && (b[1] - a[1] * sy).abs() < 1e-9);
        // Single focal length: x scale for f, per-axis principal point, distortion unchanged.
        let s = scale_camera(&cams[&3], 1000, 1600).unwrap();
        assert_eq!(s.params, vec![2000.0 * (1000.0 / 1500.0), 750.0 * (1000.0 / 1500.0), 1200.0 * (1600.0 / 2400.0), -0.05]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_images_with_empty_points_line() {
        let s = "# comment\n1 1 0 0 0 0 0 0 1 a b.png\n\n2 1 0 0 0 0 0 1 1 c.png\n1.5 2.5 7 3 4 -1\n";
        let imgs = images_txt(s).unwrap();
        assert_eq!(imgs[&1].name, "a b.png");
        assert_eq!(imgs[&1].num_points2d, 0);
        assert_eq!(imgs[&2].num_points2d, 2);
        assert_eq!(imgs[&2].obs_point_id, vec![7]);
        assert_eq!(imgs[&2].obs_xy, vec![[1.5, 2.5]]);
    }
}
