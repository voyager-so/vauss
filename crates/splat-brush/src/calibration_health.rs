//! Extreme focal-aspect drift gate. Supplied calibration is authoritative; square pixels are
//! only a weak prior for video without calibration. Unknown photo optics are not guessed.
use crate::{capture::Manifest, util::read_json};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use splat_eval::Model;
use std::{collections::BTreeMap, path::Path};

const MAX_ASPECT_DRIFT: f64 = 2.0;

pub(crate) fn check(model: &Model, capture: Option<&Manifest>, video: bool) -> Value {
    let prior: BTreeMap<_, _> = capture
        .into_iter()
        .flat_map(|m| &m.frames)
        .map(|f| (f.name.as_str(), f))
        .collect();
    let mut checked = vec![];
    let mut failed = vec![];
    let mut unknown = 0;
    for im in model.images.values() {
        let Some(c) = model.cameras.get(&im.camera_id) else {
            failed
                .push(json!({"image":im.name,"camera_id":im.camera_id,"reason":"camera missing"}));
            continue;
        };
        let p = prior.get(im.name.as_str());
        if p.is_none() && !video {
            unknown += 1;
            continue;
        }
        let fx = c.params.first().copied().unwrap_or(f64::NAN);
        let single = matches!(
            c.model.as_str(),
            "SIMPLE_PINHOLE"
                | "SIMPLE_RADIAL"
                | "RADIAL"
                | "SIMPLE_RADIAL_FISHEYE"
                | "RADIAL_FISHEYE"
        );
        let fy = if single {
            fx
        } else {
            c.params.get(1).copied().unwrap_or(f64::NAN)
        };
        let solved = (fx / c.width as f64) / (fy / c.height as f64);
        let expected = p.map_or(c.height as f64 / c.width as f64, |p| {
            (p.params[0] / p.width as f64) / (p.params[1] / p.height as f64)
        });
        let drift = (solved / expected).max(expected / solved);
        let valid = fx.is_finite()
            && fy.is_finite()
            && fx > 0.0
            && fy > 0.0
            && c.width > 0
            && c.height > 0
            && expected.is_finite()
            && expected > 0.0
            && drift.is_finite()
            && drift <= MAX_ASPECT_DRIFT * (1.0 + 8.0 * f64::EPSILON);
        let row = json!({"image":im.name,"camera_id":c.id,"model":c.model,
            "width":c.width,"height":c.height,"fx":fx,"fy":fy,"params":c.params,
            "prior_source":if p.is_some() { "capture_calibration" } else { "video_square_pixels" },
            "prior_params":p.map(|p| &p.params), "prior_width":p.map(|p| p.width),"prior_height":p.map(|p| p.height),
            "solved_normalized_focal_aspect":solved,"prior_normalized_focal_aspect":expected,
            "aspect_drift_factor":drift,"pass":valid});
        if !valid {
            failed.push(row.clone());
        }
        checked.push(row);
    }
    json!({"status":if !failed.is_empty() { "failed" } else if checked.is_empty() { "unmeasured" } else { "ok" },
        "pass":failed.is_empty(),"max_aspect_drift_factor":MAX_ASPECT_DRIFT,
        "checked":checked,"failures":failed,"unmeasured_views":unknown})
}

pub(crate) fn dataset(model: &Model, path: &Path, video: bool) -> Result<Value> {
    let manifest = path
        .join("capture.json")
        .is_file()
        .then(|| read_json::<Manifest>(&path.join("capture.json")))
        .transpose()?;
    Ok(check(model, manifest.as_ref(), video))
}

pub(crate) fn enforce(health: &Value) -> Result<()> {
    ensure!(
        health["pass"] == true,
        "calibration health failed: focal-aspect drift exceeds {MAX_ASPECT_DRIFT}x prior (or invalid calibration): {}",
        health["failures"]
            .as_array()
            .context("calibration health has no failures")?
            .iter()
            .map(|r| format!(
                "camera {} / {}: fx={}, fy={}, drift={}",
                r["camera_id"], r["image"], r["fx"], r["fy"], r["aspect_drift_factor"]
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extreme_solved_aspect_respects_calibrated_prior_and_video_provenance() {
        let solved: Value = serde_json::from_str(include_str!(
            "../fixtures/trusted-lane-c/solved-cameras.json"
        ))
        .unwrap();
        let bad = &solved["cameras"][0];
        let mut model = Model::default();
        model.cameras.insert(
            1,
            splat_eval::colmap::Camera {
                id: 1,
                model: "PINHOLE".into(),
                width: 1024,
                height: 576,
                params: serde_json::from_value(bad["params"].clone()).unwrap(),
            },
        );
        model.images.insert(
            1,
            splat_eval::colmap::Image {
                id: 1,
                camera_id: 1,
                name: "frame_000000.jpg".into(),
                qvec: [1., 0., 0., 0.],
                tvec: [0.; 3],
                num_points2d: 0,
                obs_xy: vec![],
                obs_point_id: vec![],
            },
        );
        let mut prior: Manifest = serde_json::from_value(json!({"schema_version":1,"format":"nerfstudio",
            "camera_convention":"opencv_camera_to_world","units":"source","frames":[{
                "name":"frame_000000.jpg","source_index":0,"source_name":"00000.png",
                "width":1024,"height":576,"model":"PINHOLE","params":[1092.371500965251,1084.2191087372448,512,288],
                "c2w":glam::DMat4::IDENTITY,"has_depth":false}]})).unwrap();
        let bad = check(&model, Some(&prior), false);
        assert_eq!(bad["pass"], false);
        assert!(
            enforce(&bad)
                .unwrap_err()
                .to_string()
                .contains("frame_000000.jpg")
        );
        assert_eq!(check(&model, None, true)["pass"], false);
        assert_eq!(check(&model, None, false)["status"], "unmeasured");
        // Genuine supplied anisotropy must survive, including different image resize scales.
        prior.frames[0].params = vec![
            model.cameras[&1].params[0] * 2.,
            model.cameras[&1].params[1] * 3.,
            1024.,
            864.,
        ];
        prior.frames[0].width = 2048;
        prior.frames[0].height = 1728;
        assert_eq!(check(&model, Some(&prior), true)["pass"], true);
        model.cameras.get_mut(&1).unwrap().params =
            vec![1658.415340332738, 1205.6835281040944, 512., 288.];
        assert_eq!(check(&model, None, true)["pass"], true);
        // Boundary is inclusive, and inversion cannot hide drift in the other axis.
        model.cameras.get_mut(&1).unwrap().params[0] = 1205.6835281040944 * 2.;
        assert_eq!(check(&model, None, true)["pass"], true);
        model.cameras.get_mut(&1).unwrap().params[0] = 1205.6835281040944 * 6.;
        assert_eq!(check(&model, None, true)["pass"], false);
        model.cameras.get_mut(&1).unwrap().params[0] = 1205.6835281040944 / 6.;
        assert_eq!(check(&model, None, true)["pass"], false);
        model.cameras.get_mut(&1).unwrap().params[0] = 1205.6835281040944 / 2.;
        assert_eq!(check(&model, None, true)["pass"], true);
        model.cameras.get_mut(&1).unwrap().params[0] = 1205.6835281040944;
        assert_eq!(check(&model, None, true)["pass"], true); // square pixels, non-square RGB
    }
}
