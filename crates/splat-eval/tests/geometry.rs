use splat_eval::colmap::{Image, Point3D};
use splat_eval::geometry::{GeometryEvidence, assess_geometry, geometry_health};
use splat_eval::{Model, SfmHealthOptions, sfm_health};

#[test]
fn real_failed179_rejects_and_good241_passes_at_every_world_scale() {
    for (fixture, expected) in [
        (include_str!("fixtures/failed179.json"), false),
        (include_str!("fixtures/good241.json"), true),
    ] {
        let evidence: GeometryEvidence = serde_json::from_str(fixture).unwrap();
        for scale in [1e-9, 1e-3, 1.0, 3173.0, 1e9] {
            let mut e = evidence.clone();
            for v in &mut e.views {
                let [x, y, z] = v.center;
                // Rotation + translation + scale, not merely scaling a cached verdict.
                v.center = [scale * (z + 20.0), scale * (y - 4.0), scale * (-x + 7.0)];
                v.median_depth = v.median_depth.map(|d| d * scale);
            }
            for video in [false, true] {
                let health = assess_geometry(&e, video);
                assert_eq!(
                    health.pass, expected,
                    "scale={scale} video={video}: {health:?}"
                );
                assert_eq!(health.depth_scale_discontinuity, !expected);
            }
        }
    }
}

fn model() -> Model {
    let mut model = Model::default();
    for id in 0..32 {
        model.points.push(Point3D {
            id,
            xyz: [id as f64 * 0.01, 0.0, 10.0],
            rgb: [0; 3],
            error: 0.5,
            track_len: 30,
        });
    }
    for i in 0..30 {
        model.images.insert(
            i,
            Image {
                id: i,
                name: format!("frame_{i:06}.jpg"),
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [if i < 15 { 0.0 } else { (i - 15) as f64 * 0.1 }, 0.0, 0.0],
                camera_id: 1,
                num_points2d: 32,
                obs_xy: vec![],
                obs_point_id: (0..32).collect(),
            },
        );
    }
    model
}

#[test]
fn stationary_segment_and_entire_stationary_capture_are_not_collapses() {
    let mut m = model();
    for stationary in [false, true] {
        if stationary {
            for im in m.images.values_mut() {
                im.tvec = [0.0; 3];
            }
        }
        let h = sfm_health(
            &m,
            std::path::Path::new("fixture"),
            &SfmHealthOptions {
                video: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(h.geometry.pass, "{:?}", h.geometry);
        assert_eq!(h.gates["pose_continuity"], true);
        assert!(h.pose_continuity.unwrap().flipped_frames.is_empty());
    }
}

#[test]
fn translation_impulse_is_rejected_without_rotation_jump_but_sparse_photos_are_not() {
    let mut m = model();
    m.images.get_mut(&22).unwrap().tvec[0] = 160.0;
    for scale in [1e-6, 1.0, 1e6] {
        let mut scaled = m.clone();
        for p in &mut scaled.points {
            p.xyz = p.xyz.map(|v| v * scale);
        }
        for im in scaled.images.values_mut() {
            im.tvec = im.tvec.map(|v| v * scale);
        }
        let health = geometry_health(&scaled, true);
        assert!(!health.pass);
        assert_eq!(health.translation_jumps.len(), 1);
        assert_eq!(health.translation_jumps[0].image, "frame_000022.jpg");
        assert!(geometry_health(&scaled, false).pass);
    }
    // An unregistered interval isn't evidence of an instantaneous jump.
    m.images.get_mut(&22).unwrap().name = "another_sequence_000022.jpg".into();
    assert!(geometry_health(&m, true).translation_jumps.is_empty());
}

#[test]
fn disconnected_observation_components_and_invalid_depths_are_reported() {
    let mut m = model();
    let extra: Vec<_> = m
        .points
        .iter()
        .cloned()
        .map(|mut p| {
            p.id += 100;
            p
        })
        .collect();
    m.points.extend(extra);
    for im in m.images.values_mut().filter(|im| im.id >= 15) {
        im.obs_point_id.iter_mut().for_each(|id| *id += 100);
    }
    let h = geometry_health(&m, false);
    assert_eq!(h.component_sizes, [15, 15]);
    assert!(h.disconnected);
    assert!(!h.pass);
    // A real common track bridges the two groups.
    m.images.get_mut(&20).unwrap().obs_point_id.push(0);
    assert!(geometry_health(&m, false).pass);
    m.images.get_mut(&20).unwrap().tvec[2] = -20.0;
    let h = geometry_health(&m, false);
    assert!(!h.pass);
    assert_eq!(h.invalid_depth_frames, ["frame_000020.jpg"]);
}

#[test]
fn unknown_points_are_not_depth_or_connectivity_evidence() {
    let mut m = model();
    m.points.clear();
    let h = geometry_health(&m, true);
    assert_eq!(h.status, "insufficient");
    assert_eq!(h.insufficient_frames.len(), 30);
    assert!(h.depth_p90_over_p10.is_none());
    assert!(h.component_sizes.is_empty());
}
