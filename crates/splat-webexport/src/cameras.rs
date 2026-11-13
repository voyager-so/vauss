//! Exact capture calibration and rigid poses, independent of a viewer's orbit camera.
use glam::DMat4;
use serde_json::{Value, json};
use splat_shots::scene::SceneFrame;

pub const BLENDER_IMPORT_SCRIPT: &str = include_str!("import_cameras.py");

pub fn convention() -> Value {
    json!({
        "matrix_layout": "column_major",
        "pose": "camera_to_world",
        "camera_axes": "OpenCV: x right, y down, z forward",
        "pixel_coordinates": "image corner origin; pixel centres at (x + 0.5, y + 0.5)",
        "source_frame": "trained PLY coordinates (after scene_transform.json, if present)",
        "export_frame": "scene.json transform_from_source applied to source frame",
        "intrinsics": "native image dimensions; scale fx/cx by output_width/width and fy/cy by output_height/height",
    })
}

/// Preserve every capture camera, its full lens calibration, roll and principal point. The export
/// similarity scales translations, never the camera basis: c2w remains a rigid pose.
pub fn capture_cameras(frame: &SceneFrame, transform: &DMat4) -> Vec<Value> {
    let scale = transform.x_axis.truncate().length();
    frame
        .cameras
        .iter()
        .map(|c| {
            let mut c2w = *transform * c.c2w;
            for axis in [&mut c2w.x_axis, &mut c2w.y_axis, &mut c2w.z_axis] {
                *axis /= scale;
            }
            let pos = c2w.w_axis.truncate();
            let look = pos + c2w.z_axis.truncate();
            json!({
                "image": c.name, "width": c.width, "height": c.height,
                "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
                "model": c.model, "params": c.params,
                "K": [[c.fx, 0.0, c.cx], [0.0, c.fy, c.cy], [0.0, 0.0, 1.0]],
                "c2w": c2w.to_cols_array(), "source_c2w": c.c2w.to_cols_array(),
                // Existing interactive viewers can continue to use an approximate look-at camera.
                "position": pos.to_array(), "look_at": look.to_array(), "vfov_deg": c.vfov_deg(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DQuat, DVec3};
    use splat_shots::scene::CaptureCamera;

    #[test]
    fn export_crop_uses_horizontal_focal_and_offcentre_principal_point() {
        let c = CaptureCamera {
            name: "edge.jpg".into(),
            c2w: DMat4::IDENTITY,
            width: 1000,
            height: 800,
            fx: 400.0,
            fy: 2000.0,
            cx: 800.0,
            cy: 300.0,
            model: "PINHOLE".into(),
            params: vec![400.0, 2000.0, 800.0, 300.0],
        };
        let mut frame = SceneFrame {
            cameras: vec![c],
            points: vec![],
            up: DVec3::Y,
            target: DVec3::Z,
            radius: 1.0,
            ground: -1.0,
            image_dir: None,
        };
        let g = crate::ply::Gaussians {
            pos: vec![[-3.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 0.45, 2.0]],
            scale: vec![[-10.0; 3]; 3],
            ..Default::default()
        };
        // Pixel (200,300), outside at (1200,300), pixel (800,750).
        assert_eq!(
            crate::visible_from_capture(&g, &frame),
            vec![true, false, true]
        );
        frame.cameras[0].model = "OPENCV_FISHEYE".into();
        assert_eq!(crate::visible_from_capture(&g, &frame), vec![true; 3]);
    }

    #[test]
    fn colmap_loader_retains_exact_calibration_including_fov_and_training_transform() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("cameras.txt"), "1 PINHOLE 1180 784 1658.415340332738 1205.6835281040944 571.25 406.75\n2 FOV 640 480 700 510 310 230 0.2\n3 SIMPLE_PINHOLE 640 480 600 320 240\n").unwrap();
        std::fs::write(dir.path().join("images.txt"), "1 1 0 0 0 1 2 3 1 frame_000038.jpg\n\n2 1 0 0 0 2 3 4 2 fov.jpg\n\n3 1 0 0 0 3 4 5 3 simple.jpg\n\n").unwrap();
        std::fs::write(dir.path().join("points3D.txt"), "1 0 0 4 255 0 0 0\n").unwrap();
        let train = dir.path().join("scene_transform.json");
        std::fs::write(
            &train,
            r#"{"train_from_world":{"matrix_4x4":[[2,0,0,10],[0,2,0,20],[0,0,2,30],[0,0,0,1]]}}"#,
        )
        .unwrap();
        let frame = splat_shots::scene::load_scene(dir.path(), Some(&train)).unwrap();
        let cameras = capture_cameras(&frame, &DMat4::IDENTITY);
        let pin = cameras
            .iter()
            .find(|c| c["image"] == "frame_000038.jpg")
            .unwrap();
        assert_eq!(pin["fx"], 1658.415340332738);
        assert_eq!(pin["fy"], 1205.6835281040944);
        assert_eq!(pin["cx"], 571.25);
        assert_eq!(pin["cy"], 406.75);
        let pose: [f64; 16] = serde_json::from_value(pin["source_c2w"].clone()).unwrap();
        assert_eq!(&pose[12..15], &[8.0, 16.0, 24.0]);
        assert_eq!(pose[0], 1.0);
        let fov = cameras.iter().find(|c| c["image"] == "fov.jpg").unwrap();
        assert_eq!(fov["fy"], 510.0);
        assert_eq!(fov["params"][4], 0.2);
        let simple = cameras.iter().find(|c| c["image"] == "simple.jpg").unwrap();
        assert_eq!(simple["fx"], simple["fy"]);
        assert_eq!(simple["cx"], 320.0);
    }

    #[test]
    fn exact_camera_projection_survives_export_similarity_and_serialization() {
        // Demo's unequal focal lengths plus off-centre principal point and roll. A square-pixel
        // or look-at reconstruction would move these asymmetric samples by many pixels.
        let pose = DMat4::from_rotation_translation(
            DQuat::from_euler(glam::EulerRot::XYZ, 0.3, -0.7, 0.4),
            DVec3::new(1.2, 0.4, -1.7),
        );
        let c = CaptureCamera {
            name: "frame_000038.jpg".into(),
            c2w: pose,
            width: 1180,
            height: 784,
            fx: 1658.415340332738,
            fy: 1205.6835281040944,
            cx: 571.25,
            cy: 406.75,
            model: "PINHOLE".into(),
            params: vec![1658.415340332738, 1205.6835281040944, 571.25, 406.75],
        };
        let frame = SceneFrame {
            cameras: vec![c.clone(); 241],
            points: vec![],
            up: DVec3::Y,
            target: DVec3::ZERO,
            radius: 4.0,
            ground: -1.0,
            image_dir: None,
        };
        let transform = DMat4::from_scale_rotation_translation(
            DVec3::splat(3173.2133),
            DQuat::from_rotation_y(0.6),
            DVec3::new(9000.0, -100.0, 7000.0),
        );
        let encoded = serde_json::to_vec(&capture_cameras(&frame, &transform)).unwrap();
        let cameras: Vec<Value> = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(cameras.len(), 241); // no camera subsampling
        let exported = DMat4::from_cols_array(
            &serde_json::from_value::<[f64; 16]>(cameras[38]["c2w"].clone()).unwrap(),
        );
        assert!((exported.determinant() - 1.0).abs() < 1e-12);
        for p in [
            DVec3::new(-0.6, 0.3, 4.0),
            DVec3::new(0.4, -0.5, 3.0),
            DVec3::new(0.7, 0.8, 5.0),
        ] {
            let src = pose.transform_point3(p);
            let out = exported
                .inverse()
                .transform_point3(transform.transform_point3(src));
            let native = [c.fx * p.x / p.z + c.cx, c.fy * p.y / p.z + c.cy];
            let k = &cameras[38];
            let projected = [
                k["fx"].as_f64().unwrap() * out.x / out.z + k["cx"].as_f64().unwrap(),
                k["fy"].as_f64().unwrap() * out.y / out.z + k["cy"].as_f64().unwrap(),
            ];
            for axis in 0..2 {
                assert!((native[axis] - projected[axis]).abs() < 1e-9);
            }
            // Blender's local camera looks along -Z, +Y up: flip local Y/Z, not world X.
            let blender_pose = exported * DMat4::from_scale(DVec3::new(1.0, -1.0, -1.0));
            let b = blender_pose
                .inverse()
                .transform_point3(transform.transform_point3(src));
            assert!((c.fx * b.x / -b.z + c.cx - native[0]).abs() < 1e-9);
            assert!((c.fy * -b.y / -b.z + c.cy - native[1]).abs() < 1e-9);
        }
    }
}
