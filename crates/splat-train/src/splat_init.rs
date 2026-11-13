use ball_tree::BallTree;
use brush_render::{
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats, inverse_sigmoid},
    kernels::camera_model::CameraModel,
};
use brush_serde::SplatData;
use burn::{config::Config, tensor::Device};
use glam::Vec3;
use rand::{Rng, RngExt, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::trace_span;

#[derive(Config, Debug)]
pub struct RandomSplatsConfig {
    #[config(default = 10000)]
    pub init_count: usize,
}

/// Estimate scene scale from camera positions.
///
/// Uses the average nearest-neighbor distance between cameras,
/// with a minimum of 1.0 (1 meter baseline).
fn estimate_scene_scale(cameras: &[Camera]) -> f32 {
    if cameras.len() < 2 {
        return 1.0;
    }

    let mut total_nn_dist = 0.0f32;
    for (i, cam) in cameras.iter().enumerate() {
        let mut min_dist = f32::INFINITY;
        for (j, other) in cameras.iter().enumerate() {
            if i != j {
                let d = cam.position.distance(other.position);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        total_nn_dist += min_dist;
    }

    let avg_nn = total_nn_dist / cameras.len() as f32;
    // Scene depth is roughly a few multiples of the camera spacing.
    // Use 3x the average spacing, with 1m floor.
    (avg_nn * 3.0).max(1.0)
}

/// Create initial splats by sampling random points inside camera frustums.
///
/// For each splat, a random camera is chosen, then a random ray direction
/// within its field of view is sampled, and a random depth along that ray
/// is picked.
pub fn create_random_splats(
    config: &RandomSplatsConfig,
    cameras: &[Camera],
    scene_scale_override: Option<f32>,
    rng: &mut impl Rng,
    mode: SplatRenderMode,
    device: &Device,
) -> Splats {
    let num_points = config.init_count;
    let scene_scale = scene_scale_override.unwrap_or_else(|| estimate_scene_scale(cameras));

    let positions = sample_random_positions(num_points, cameras, scene_scale, rng);

    // Random colors
    let sh_coeffs: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            [
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
            ]
        })
        .collect();

    // Random rotations (normalized quaternions)
    let rotations: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            let x: f32 = rng.random_range(-1.0..1.0);
            let y: f32 = rng.random_range(-1.0..1.0);
            let z: f32 = rng.random_range(-1.0..1.0);
            let w: f32 = rng.random_range(-1.0..1.0);
            let len = (x * x + y * y + z * z + w * w).sqrt().max(1e-6);
            [x / len, y / len, z / len, w / len]
        })
        .collect();

    // Random opacities
    let opacities: Vec<f32> = (0..num_points)
        .map(|_| rng.random_range(inverse_sigmoid(0.1)..inverse_sigmoid(0.25)))
        .collect();

    // Scale based on scene scale and point density
    let default_scale = (scene_scale / (num_points as f32).cbrt()).ln();
    let mut log_scales: Vec<f32> = vec![default_scale; num_points * 3];
    surfel_normal_scale(&mut log_scales, mode);

    Splats::from_raw(
        positions, rotations, log_scales, sh_coeffs, opacities, mode, device,
    )
}

/// CPU-only position sampling, shared by initialization and frustum regression tests.
fn sample_random_positions(
    num_points: usize,
    cameras: &[Camera],
    scene_scale: f32,
    rng: &mut impl Rng,
) -> Vec<f32> {
    let near = scene_scale * 0.05;
    let far = scene_scale;
    let ln_near = near.ln();
    let ln_far = far.ln();

    (0..num_points)
        .flat_map(|_| {
            let cam = &cameras[rng.random_range(0..cameras.len())];
            let local_to_world = cam.local_to_world();

            // Random direction within the camera's FOV
            let half_fov_x = (cam.fov_x * 0.5) as f32;
            let half_fov_y = (cam.fov_y * 0.5) as f32;
            let angle_range = |half_fov: f32, center: f32| {
                // Pinhole FOV encodes focal length, not symmetric image bounds
                // when the principal point is off-center. Preserve the existing
                // angular distribution (and exact centered-camera RNG stream).
                if cam.camera_model == CameraModel::Pinhole && center != 0.5 {
                    let span = 2.0 * half_fov.tan();
                    (-center * span).atan()..((1.0 - center) * span).atan()
                } else {
                    // Distorted lenses retain the existing FOV approximation.
                    -half_fov..half_fov
                }
            };
            let dx = rng
                .random_range(angle_range(half_fov_x, cam.center_uv.x))
                .tan();
            let dy = rng
                .random_range(angle_range(half_fov_y, cam.center_uv.y))
                .tan();

            // Log-uniform depth so we don't over-pack near the camera
            let depth = (rng.random_range(ln_near..ln_far)).exp();

            // Camera poses and the renderer use OpenCV camera space: +Z forward.
            let local_point = Vec3::new(dx * depth, dy * depth, depth);
            let world_point = local_to_world.transform_point3(local_point);

            [world_point.x, world_point.y, world_point.z]
        })
        .collect()
}

/// Surfels ignore the third (normal) log-scale; pin it to the smaller tangent
/// one so "max scale" heuristics only ever see the disk's real extent.
fn surfel_normal_scale(log_scales: &mut [f32], mode: SplatRenderMode) {
    if mode == SplatRenderMode::Surfel {
        for s in log_scales.chunks_exact_mut(3) {
            s[2] = s[0].min(s[1]);
        }
    }
}

pub fn bounds_from_pos(percentile: f32, means: &[f32]) -> BoundingBox {
    let (mut x_vals, mut y_vals, mut z_vals): (Vec<f32>, Vec<f32>, Vec<f32>) = means
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect();
    x_vals.retain(|x| x.is_finite());
    y_vals.retain(|y| y.is_finite());
    z_vals.retain(|z| z.is_finite());

    // If any axis is entirely non-finite, fall back to a unit box rather
    // than panicking on the percentile index.
    if x_vals.is_empty() || y_vals.is_empty() || z_vals.is_empty() {
        return BoundingBox::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
    }

    // Two order statistics per axis: O(n) selection instead of a full sort
    // (this runs on every refine, over every splat).
    let pick = |vals: &mut [f32]| -> (f32, f32) {
        let n = vals.len();
        let lo = ((1.0 - percentile) / 2.0 * n as f32) as usize;
        let hi = (n - 1).min(((1.0 + percentile) / 2.0 * n as f32) as usize);
        let lo_v = *vals.select_nth_unstable_by(lo, |a, b| a.total_cmp(b)).1;
        let hi_v = *vals.select_nth_unstable_by(hi, |a, b| a.total_cmp(b)).1;
        (lo_v, hi_v)
    };

    let (xmin, xmax) = pick(&mut x_vals);
    let (ymin, ymax) = pick(&mut y_vals);
    let (zmin, zmax) = pick(&mut z_vals);
    BoundingBox::from_min_max(Vec3::new(xmin, ymin, zmin), Vec3::new(xmax, ymax, zmax))
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct BallPoint(glam::Vec3A);

impl ball_tree::Point for BallPoint {
    fn distance(&self, other: &Self) -> f64 {
        self.0.distance(other.0) as f64
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {
        Self(self.0.lerp(other.0, d as f32 / self.0.distance(other.0)))
    }

    fn midpoint(a: &Self, b: &Self) -> Self {
        Self((a.0 + b.0) / 2.0)
    }
}

/// Compute scales using KNN based on point density.
fn compute_knn_scales(pos_data: &[f32]) -> Vec<f32> {
    let _ = trace_span!("compute_knn_scales").entered();

    let n_splats = pos_data.len() / 3;

    if n_splats < 3 {
        return vec![0.0; n_splats * 3];
    }

    let bounding_box = trace_span!("Bounds from pose").in_scope(|| bounds_from_pos(0.75, pos_data));
    let median_size = bounding_box.median_size().max(0.01);

    trace_span!("Splats KNN scale init").in_scope(|| {
        let tree_points: Vec<BallPoint> = pos_data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| BallPoint(glam::Vec3A::new(v[0], v[1], v[2])))
            .collect();

        let empty = vec![(); tree_points.len()];
        let tree = BallTree::new(tree_points.clone(), empty);

        tree_points
            .par_iter()
            .map_with(tree.query(), |query, p| {
                // Get half of the average of 2 nearest distances.
                let mut q = query.nn(p).skip(1);
                let a1 = q.next().unwrap().1 as f32;
                let a2 = q.next().unwrap().1 as f32;
                let dist = (a1 + a2) / 4.0;
                dist.clamp(1e-3, median_size * 0.1).ln()
            })
            .flat_map(|p| [p, p, p])
            .collect()
    })
}

/// True when every point has the same DC colour (to within 1e-3).
fn single_colour(sh_coeffs: &[f32], n_splats: usize) -> bool {
    if n_splats < 2 || sh_coeffs.len() < n_splats * 3 {
        return false;
    }
    let stride = sh_coeffs.len() / n_splats;
    let first = &sh_coeffs[0..3];
    sh_coeffs
        .chunks_exact(stride)
        .all(|c| c[..3].iter().zip(first).all(|(a, b)| (a - b).abs() < 1e-3))
}

pub fn to_init_splats(data: SplatData, mode: SplatRenderMode, device: &Device) -> Splats {
    let n_splats = data.num_splats();

    // Use KNN for scales if not provided
    let mut log_scales = data
        .log_scales
        .unwrap_or_else(|| compute_knn_scales(&data.means));
    surfel_normal_scale(&mut log_scales, mode);

    // Default rotation = identity quaternion [1, 0, 0, 0]
    let rotations = data
        .rotations
        .unwrap_or_else(|| [1.0, 0.0, 0.0, 0.0].repeat(n_splats));

    // Default opacity = inverse_sigmoid(0.5)
    let opacities = data
        .raw_opacities
        .unwrap_or_else(|| vec![inverse_sigmoid(0.5); n_splats]);

    // A seed without colour, or with one colour for every point (e.g. an SfM
    // export that dropped colours), gets random colours like random init, so
    // the splats don't start as one flat colour.
    let sh_coeffs = match data.sh_coeffs {
        Some(c) if !single_colour(&c, n_splats) => c,
        _ => {
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            (0..n_splats * 3)
                .map(|_| rng.random_range(0.0..1.0))
                .collect()
        }
    };

    Splats::from_raw(
        data.means, rotations, log_scales, sh_coeffs, opacities, mode, device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinhole_camera(position: Vec3, rotation: glam::Quat, center: glam::Vec2) -> Camera {
        let image_size = glam::uvec2(1280, 720);
        Camera::new(
            position,
            rotation,
            brush_render::camera::focal_to_fov(900.0, image_size.x, &CameraModel::Pinhole),
            brush_render::camera::focal_to_fov(650.0, image_size.y, &CameraModel::Pinhole),
            center,
            CameraModel::Pinhole,
        )
    }

    fn assert_samples_in_frustum(camera: Camera) {
        let scene_scale = 4.0;
        let count = 4096;
        let positions = sample_random_positions(
            count,
            &[camera],
            scene_scale,
            &mut rand::rngs::StdRng::seed_from_u64(42),
        );
        assert_eq!(positions.len(), count * 3);
        let image_size = glam::uvec2(1280, 720);
        let params = camera.build_pinhole_params(image_size);
        let mut min_uv = glam::Vec2::ONE;
        let mut max_uv = glam::Vec2::ZERO;
        let mut min_depth = f32::INFINITY;
        let mut max_depth = 0.0f32;
        for point in positions.chunks_exact(3) {
            let local = camera
                .world_to_local()
                .transform_point3(Vec3::from_slice(point));
            assert!(local.is_finite());
            // In front of the renderer near plane and within the configured depth range.
            assert!(local.z >= 0.01, "point behind camera: {local:?}");
            assert!((scene_scale * 0.05 - 1e-5..=scene_scale + 1e-5).contains(&local.z));
            // Same pinhole equations as project_pinhole, using independently specified fx/fy.
            let uv = glam::vec2(
                (params.fx * local.x / local.z + params.cx) / image_size.x as f32,
                (params.fy * local.y / local.z + params.cy) / image_size.y as f32,
            );
            assert!((-1e-5..=1.0 + 1e-5).contains(&uv.x), "u: {}", uv.x);
            assert!((-1e-5..=1.0 + 1e-5).contains(&uv.y), "v: {}", uv.y);
            min_uv = min_uv.min(uv);
            max_uv = max_uv.max(uv);
            min_depth = min_depth.min(local.z);
            max_depth = max_depth.max(local.z);
        }
        // Catch a degenerate sampler that puts all points on the optical axis or one plane.
        assert!(min_uv.max_element() < 0.02);
        assert!(max_uv.min_element() > 0.98);
        assert!(min_depth < scene_scale * 0.06);
        assert!(max_depth > scene_scale * 0.95);
    }

    #[test]
    fn random_positions_are_in_front_of_identity_and_transformed_cameras() {
        for (position, rotation) in [
            (Vec3::ZERO, glam::Quat::IDENTITY),
            (Vec3::new(3.0, -2.0, 5.0), glam::Quat::IDENTITY),
            (
                Vec3::ZERO,
                glam::Quat::from_rotation_y(std::f32::consts::PI),
            ),
            (
                Vec3::new(-4.0, 2.0, 3.0),
                glam::Quat::from_euler(glam::EulerRot::YXZ, 1.2, -0.4, 0.3),
            ),
        ] {
            assert_samples_in_frustum(pinhole_camera(position, rotation, glam::Vec2::splat(0.5)));
        }
    }

    #[test]
    fn random_positions_respect_off_center_principal_points() {
        for center in [
            glam::vec2(0.2, 0.8),
            glam::vec2(0.75, 0.3),
            glam::vec2(1.1, -0.1),
        ] {
            assert_samples_in_frustum(pinhole_camera(
                Vec3::new(2.0, -3.0, 1.0),
                glam::Quat::from_euler(glam::EulerRot::YXZ, -0.9, 0.2, -0.5),
                center,
            ));
        }
    }

    #[test]
    fn random_positions_preserve_seeded_sampling_and_rng_consumption() {
        let cameras = [
            pinhole_camera(Vec3::ZERO, glam::Quat::IDENTITY, glam::Vec2::splat(0.5)),
            pinhole_camera(
                Vec3::new(4.0, -2.0, 3.0),
                glam::Quat::from_rotation_y(0.7),
                glam::Vec2::splat(0.5),
            ),
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut reference_rng = rand::rngs::StdRng::seed_from_u64(123);
        let count = 128;
        let positions = sample_random_positions(count, &cameras, 4.0, &mut rng);
        for point in positions.chunks_exact(3) {
            // Original centered-camera draw order and distribution, with corrected +Z.
            let camera = &cameras[reference_rng.random_range(0..cameras.len())];
            let half_x = (camera.fov_x * 0.5) as f32;
            let half_y = (camera.fov_y * 0.5) as f32;
            let dx = reference_rng.random_range(-half_x..half_x).tan();
            let dy = reference_rng.random_range(-half_y..half_y).tan();
            let depth = reference_rng.random_range(0.2f32.ln()..4.0f32.ln()).exp();
            let expected =
                camera
                    .local_to_world()
                    .transform_point3(Vec3::new(dx * depth, dy * depth, depth));
            assert_eq!(point, expected.to_array());
        }
        // Later color/rotation/opacity draws must remain reproducible too.
        assert_eq!(rng.random::<u64>(), reference_rng.random::<u64>());
    }

    #[test]
    fn bounds_from_pos_all_nan_does_not_panic() {
        let means = vec![f32::NAN; 30];
        let bb = bounds_from_pos(0.8, &means);
        // We expect a finite fallback — no NaN leak, no panic.
        assert!(bb.center.is_finite(), "center: {:?}", bb.center);
        assert!(bb.extent.is_finite(), "extent: {:?}", bb.extent);
    }

    #[test]
    fn bounds_from_pos_empty_does_not_panic() {
        let bb = bounds_from_pos(0.8, &[]);
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
    }

    #[test]
    fn bounds_from_pos_mixed_nan_and_finite() {
        // Half NaN, half finite. The finite half should determine the bounds.
        let mut means = Vec::new();
        for i in 0..100 {
            if i % 2 == 0 {
                means.extend_from_slice(&[f32::NAN, f32::NAN, f32::NAN]);
            } else {
                means.extend_from_slice(&[i as f32, i as f32, i as f32]);
            }
        }
        let bb = bounds_from_pos(0.8, &means);
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
        // Extent should be reasonable (the finite values span 1..99).
        assert!(bb.extent.x > 0.0 && bb.extent.x < 100.0);
    }

    #[test]
    fn bounds_from_pos_one_axis_all_nan() {
        // x and z are OK, y is all NaN — we must not panic indexing into y.
        let mut means = Vec::new();
        for i in 0..50 {
            means.extend_from_slice(&[i as f32, f32::NAN, i as f32]);
        }
        let bb = bounds_from_pos(0.8, &means);
        // y axis collapses to the fallback, other axes should still be
        // reasonable.
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
    }
}
