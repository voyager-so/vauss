//! Fuse analytically rendered scenes and check the surface against the ground truth.

use glam::{Mat4, Vec3};
use splat_mesh::{Camera, Intrinsics, Mesh, MeshOpts, View, decimate, mesh_from_renders};

/// A shape we can ray-cast: returns the hit distance along a unit ray, and the colour there.
trait Shape: Sync {
    fn hit(&self, o: Vec3, d: Vec3) -> Option<f32>;
    fn color(&self, p: Vec3) -> [u8; 3];
    /// Unsigned distance from `p` to the surface.
    fn dist(&self, p: Vec3) -> f32;
}

struct Sphere {
    c: Vec3,
    r: f32,
}

impl Shape for Sphere {
    fn hit(&self, o: Vec3, d: Vec3) -> Option<f32> {
        let oc = o - self.c;
        let b = oc.dot(d);
        let disc = b * b - (oc.length_squared() - self.r * self.r);
        (disc >= 0.0).then(|| -b - disc.sqrt()).filter(|&t| t > 0.0)
    }
    fn color(&self, p: Vec3) -> [u8; 3] {
        let n = (p - self.c).normalize();
        [(n.x * 0.5 + 0.5) * 255.0, (n.y * 0.5 + 0.5) * 255.0, (n.z * 0.5 + 0.5) * 255.0].map(|v| v.round() as u8)
    }
    fn dist(&self, p: Vec3) -> f32 {
        ((p - self.c).length() - self.r).abs()
    }
}

struct Cube {
    h: f32,
}

impl Shape for Cube {
    fn hit(&self, o: Vec3, d: Vec3) -> Option<f32> {
        let inv = d.recip();
        let t0 = (Vec3::splat(-self.h) - o) * inv;
        let t1 = (Vec3::splat(self.h) - o) * inv;
        let tmin = t0.min(t1).max_element();
        let tmax = t0.max(t1).min_element();
        (tmax >= tmin && tmin > 0.0).then_some(tmin)
    }
    fn color(&self, _: Vec3) -> [u8; 3] {
        [200, 120, 40]
    }
    fn dist(&self, p: Vec3) -> f32 {
        let q = p.abs() - Vec3::splat(self.h);
        (q.max(Vec3::ZERO).length() + q.max_element().min(0.0)).abs()
    }
}

fn look_at(eye: Vec3, target: Vec3, k: Intrinsics) -> Camera {
    let fwd = (target - eye).normalize();
    let up = if fwd.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let right = fwd.cross(-up).normalize();
    let down = fwd.cross(right);
    let c2w = Mat4::from_cols(right.extend(0.0), down.extend(0.0), fwd.extend(0.0), eye.extend(1.0));
    Camera { intrinsics: k, world_to_cam: c2w.inverse() }
}

fn ring(n: usize, radius: f32, k: Intrinsics) -> Vec<Camera> {
    let mut cams = vec![];
    for (elev, count) in [(-0.6f32, n / 3), (0.0, n / 3), (0.7, n - 2 * (n / 3))] {
        for i in 0..count {
            let a = i as f32 / count as f32 * std::f32::consts::TAU + elev;
            let eye = Vec3::new(a.cos() * elev.cos(), elev.sin(), a.sin() * elev.cos()) * radius;
            cams.push(look_at(eye, Vec3::ZERO, k));
        }
    }
    cams
}

fn render(shape: &dyn Shape, cam: &Camera) -> View {
    let k = cam.intrinsics;
    let c2w = cam.cam_to_world();
    let o = c2w.w_axis.truncate();
    let n = (k.width * k.height) as usize;
    let (mut rgb, mut alpha, mut depth) = (vec![0u8; 3 * n], vec![0f32; n], vec![0f32; n]);
    for y in 0..k.height {
        for x in 0..k.width {
            let dc = Vec3::new((x as f32 + 0.5 - k.cx) / k.fx, (y as f32 + 0.5 - k.cy) / k.fy, 1.0);
            let d = c2w.transform_vector3(dc);
            if let Some(t) = shape.hit(o, d.normalize()) {
                let i = (y * k.width + x) as usize;
                let p = o + d.normalize() * t;
                depth[i] = t / d.length(); // ray length → view-space z
                alpha[i] = 1.0;
                rgb[3 * i..3 * i + 3].copy_from_slice(&shape.color(p));
            }
        }
    }
    View { camera: *cam, rgb, alpha, depth }
}

fn intr() -> Intrinsics {
    Intrinsics { fx: 220.0, fy: 220.0, cx: 120.0, cy: 90.0, width: 240, height: 180 }
}

fn errors(mesh: &Mesh, shape: &dyn Shape) -> (f32, f32) {
    let d: Vec<f32> = mesh.positions.iter().map(|&p| shape.dist(p)).collect();
    (d.iter().sum::<f32>() / d.len() as f32, d.iter().copied().fold(0.0, f32::max))
}

fn area(mesh: &Mesh) -> f32 {
    mesh.triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
            (b - a).cross(c - a).length() * 0.5
        })
        .sum()
}

fn opts() -> MeshOpts {
    // Perfect synthetic depth: check the fusion itself, without the noise-oriented defaults.
    MeshOpts { resolution: 96, max_triangles: None, trunc_voxels: 4.0, smooth_iterations: 0, ..Default::default() }
}

#[test]
fn sphere_surface_is_accurate() {
    let shape = Sphere { c: Vec3::ZERO, r: 1.0 };
    let cams = ring(30, 3.0, intr());
    let (mesh, stats) = mesh_from_renders(&cams, |c| Ok(render(&shape, c)), &opts()).unwrap();
    let vs = stats.voxel_size;
    let (mean, max) = errors(&mesh, &shape);
    eprintln!("{}", stats.to_json());
    eprintln!("sphere: voxel {vs:.4} mean err {:.3} vox, max {:.3} vox, area {:.3} (4π = {:.3})", mean / vs, max / vs, area(&mesh), 4.0 * std::f32::consts::PI);
    assert!(mean < 0.25 * vs, "mean error {mean} vs voxel {vs}");
    assert!(max < 0.75 * vs, "max error {max} vs voxel {vs}");
    let a = area(&mesh) / (4.0 * std::f32::consts::PI);
    assert!((0.97..1.03).contains(&a), "area ratio {a}");
    // Closed surface: every edge shared by exactly two triangles.
    let mut edges = std::collections::HashMap::new();
    for t in &mesh.triangles {
        for j in 0..3 {
            let (a, b) = (t[j].min(t[(j + 1) % 3]), t[j].max(t[(j + 1) % 3]));
            *edges.entry((a, b)).or_insert(0) += 1;
        }
    }
    let open = edges.values().filter(|&&c| c != 2).count();
    for ((a, b), c) in edges.iter().filter(|(_, c)| **c != 2).take(12) {
        let (pa, pb) = (mesh.positions[*a as usize] / vs, mesh.positions[*b as usize] / vs);
        eprintln!("open edge count {c}: {pa:.2} {pb:.2} r={:.3}", (pa * vs).length());
    }
    assert_eq!(open, 0, "{open} non-manifold/border edges");
    // Outward winding.
    let outward = mesh
        .triangles
        .iter()
        .filter(|t| {
            let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
            (b - a).cross(c - a).dot((a + b + c) / 3.0) > 0.0
        })
        .count();
    assert!(outward as f32 > 0.99 * mesh.triangles.len() as f32, "{outward}/{} outward", mesh.triangles.len());
    // Colours follow the normal map.
    let col_err: f32 = mesh
        .positions
        .iter()
        .zip(&mesh.colors)
        .map(|(&p, c)| {
            let e = shape.color(p);
            (0..3).map(|i| (c[i] as f32 - e[i] as f32).abs()).sum::<f32>() / 3.0
        })
        .sum::<f32>()
        / mesh.positions.len() as f32;
    assert!(col_err < 8.0, "mean colour error {col_err}");
    // Auto bounds hug the sphere.
    let b = stats.bounds.unwrap();
    assert!(b.min.max_element() < -0.95 && b.max.min_element() > 0.95 && b.extent().max_element() < 2.5, "{b:?}");
}

#[test]
fn cube_surface_is_accurate() {
    let shape = Cube { h: 0.7 };
    let cams = ring(30, 3.0, intr());
    let (mesh, stats) = mesh_from_renders(&cams, |c| Ok(render(&shape, c)), &opts()).unwrap();
    let vs = stats.voxel_size;
    let (mean, max) = errors(&mesh, &shape);
    eprintln!("cube: mean err {:.3} vox, max {:.3} vox, area {:.3} (expect {:.3})", mean / vs, max / vs, area(&mesh), 6.0 * 1.4 * 1.4);
    assert!(mean < 0.2 * vs, "mean error {mean} vs voxel {vs}");
    assert!(max < 1.5 * vs, "max error {max} vs voxel {vs}");
    let a = area(&mesh) / (6.0 * 1.4 * 1.4);
    assert!((0.95..1.05).contains(&a), "area ratio {a}");
}

#[test]
fn floater_seen_by_one_view_is_removed() {
    let shape = Sphere { c: Vec3::ZERO, r: 1.0 };
    let cams = ring(30, 3.0, intr());
    // The first view also sees a small blob floating between it and the sphere.
    let first = cams[0];
    let blob = Sphere { c: first.position() * 0.55, r: 0.08 };
    let render_with_blob = |c: &Camera| {
        let mut v = render(&shape, c);
        if *c == first || c.intrinsics != intr() && c.world_to_cam == first.world_to_cam {
            let b = render(&blob, c);
            for i in 0..v.depth.len() {
                if b.depth[i] > 0.0 && (v.depth[i] == 0.0 || b.depth[i] < v.depth[i]) {
                    v.depth[i] = b.depth[i];
                    v.alpha[i] = 1.0;
                }
            }
        }
        Ok(v)
    };
    let o = MeshOpts { bounds: Some(splat_mesh::Aabb { min: Vec3::splat(-1.5), max: Vec3::splat(1.5) }), ..opts() };
    let (mesh, _) = mesh_from_renders(&cams, render_with_blob, &o).unwrap();
    let near_blob = mesh.positions.iter().filter(|p| (**p - blob.c).length() < 0.2).count();
    assert_eq!(near_blob, 0, "floater survived ({near_blob} vertices)");
}

#[test]
fn decimation_keeps_shape() {
    let shape = Sphere { c: Vec3::ZERO, r: 1.0 };
    let cams = ring(30, 3.0, intr());
    let (mut mesh, stats) = mesh_from_renders(&cams, |c| Ok(render(&shape, c)), &opts()).unwrap();
    let before = mesh.triangles.len();
    let t = std::time::Instant::now();
    decimate(&mut mesh, before / 10);
    eprintln!("decimate {before} -> {} in {:?}", mesh.triangles.len(), t.elapsed());
    assert!(mesh.triangles.len() <= before / 10, "{} triangles left", mesh.triangles.len());
    assert!(mesh.triangles.len() > before / 20, "over-decimated to {}", mesh.triangles.len());
    let (mean, _) = errors(&mesh, &shape);
    assert!(mean < 0.5 * stats.voxel_size, "mean error after decimation {mean}");
    let a = area(&mesh) / (4.0 * std::f32::consts::PI);
    assert!((0.95..1.02).contains(&a), "area ratio {a}");
}

#[test]
fn glb_and_ply_round_trip_through_gltf() {
    let shape = Sphere { c: Vec3::ZERO, r: 1.0 };
    let cams = ring(12, 3.0, intr());
    let (mesh, _) = mesh_from_renders(&cams, |c| Ok(render(&shape, c)), &MeshOpts { resolution: 32, ..opts() }).unwrap();
    let dir = std::env::temp_dir().join(format!("splat-mesh-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    splat_mesh::write_glb(&mesh, &dir.join("m.glb")).unwrap();
    splat_mesh::write_ply(&mesh, &dir.join("m.ply")).unwrap();
    let glb = gltf::Gltf::from_slice(&std::fs::read(dir.join("m.glb")).unwrap()).unwrap();
    let blob = glb.blob.clone().unwrap();
    let prim = glb.document.meshes().next().unwrap().primitives().next().unwrap();
    let reader = prim.reader(|_| Some(&blob[..]));
    assert_eq!(reader.read_positions().unwrap().count(), mesh.positions.len());
    assert_eq!(reader.read_indices().unwrap().into_u32().count(), mesh.triangles.len() * 3);
    assert!(reader.read_colors(0).is_some());
    let ply = std::fs::read(dir.join("m.ply")).unwrap();
    assert!(ply.starts_with(b"ply\nformat binary_little_endian"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn preview_draws_the_mesh() {
    let shape = Sphere { c: Vec3::ZERO, r: 1.0 };
    let cams = ring(12, 3.0, intr());
    let (mesh, _) = mesh_from_renders(&cams, |c| Ok(render(&shape, c)), &MeshOpts { resolution: 48, ..opts() }).unwrap();
    let img = splat_mesh::preview::render(&mesh, &cams[0], splat_mesh::preview::Shade::Clay, [0, 0, 0]);
    let truth = render(&shape, &cams[0]);
    let mut agree = 0;
    for (i, px) in img.pixels().enumerate() {
        if (px.0[0] > 0) == (truth.alpha[i] > 0.0) {
            agree += 1;
        }
    }
    let frac = agree as f32 / truth.alpha.len() as f32;
    assert!(frac > 0.99, "silhouette agreement {frac}");
}
