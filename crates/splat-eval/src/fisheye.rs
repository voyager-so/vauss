//! Kannala-Brandt fisheye cameras: COLMAP's OPENCV_FISHEYE, RADIAL_FISHEYE and SIMPLE_RADIAL_FISHEYE
//! are all the equidistant KB4 model `r = f * theta_d(theta)`, theta_d = theta (1 + k1 theta^2 +
//! k2 theta^4 + k3 theta^6 + k4 theta^8), with the latter two using a single focal and fewer terms.
//! The rasterizer trains these natively (splat-train-render's kannala_brandt_4); this is the CPU
//! side: reprojection, unprojection (depth seeds), undistortion and image-circle masks.

use crate::colmap::Camera;

/// COLMAP model names that are KB4 fisheyes (THIN_PRISM_FISHEYE adds tangential and prism terms
/// and is not handled here).
pub const KB4_MODELS: [&str; 3] = ["OPENCV_FISHEYE", "RADIAL_FISHEYE", "SIMPLE_RADIAL_FISHEYE"];

/// Any COLMAP fisheye model (KB4 or thin-prism), for checks that only need to know the lens is
/// not perspective.
pub fn is_fisheye_model(model: &str) -> bool {
    model.ends_with("_FISHEYE")
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kb4 {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k: [f64; 4],
}

impl Kb4 {
    /// The KB4 form of a COLMAP camera; None for other models or too few parameters.
    pub fn of(cam: &Camera) -> Option<Kb4> {
        Self::from_params(&cam.model, &cam.params)
    }

    pub fn from_params(model: &str, p: &[f64]) -> Option<Kb4> {
        Some(match model {
            "OPENCV_FISHEYE" if p.len() >= 8 => Kb4 { fx: p[0], fy: p[1], cx: p[2], cy: p[3], k: [p[4], p[5], p[6], p[7]] },
            "RADIAL_FISHEYE" if p.len() >= 5 => Kb4 { fx: p[0], fy: p[0], cx: p[1], cy: p[2], k: [p[3], p[4], 0.0, 0.0] },
            "SIMPLE_RADIAL_FISHEYE" if p.len() >= 4 => Kb4 { fx: p[0], fy: p[0], cx: p[1], cy: p[2], k: [p[3], 0.0, 0.0, 0.0] },
            _ => return None,
        })
    }

    /// theta_d(theta): the normalised image radius of a ray `theta` off the optical axis.
    pub fn theta_d(&self, theta: f64) -> f64 {
        let t2 = theta * theta;
        let [k1, k2, k3, k4] = self.k;
        theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    }

    fn dtheta_d(&self, theta: f64) -> f64 {
        let t2 = theta * theta;
        let [k1, k2, k3, k4] = self.k;
        1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
    }

    /// The largest angle off the axis (at most pi) up to which theta_d still increases: past it the
    /// polynomial folds back and two rays would land on one pixel.
    pub fn max_theta(&self) -> f64 {
        const STEPS: usize = 2048;
        let step = std::f64::consts::PI / STEPS as f64;
        (1..=STEPS).map(|i| i as f64 * step).find(|&t| self.dtheta_d(t) <= 0.0).map_or(std::f64::consts::PI, |t| t - step)
    }

    /// Inverse of [`Self::theta_d`] on [0, max_theta]; None when `theta_d` lies beyond the fold.
    pub fn theta(&self, theta_d: f64) -> Option<f64> {
        if !theta_d.is_finite() || theta_d < 0.0 {
            return None;
        }
        let max = self.max_theta();
        if theta_d > self.theta_d(max) + 1e-12 {
            return None;
        }
        // Newton from theta_d (exact for k = 0), bisection-guarded on [0, max].
        let (mut lo, mut hi) = (0.0, max);
        let mut t = theta_d.min(max);
        for _ in 0..60 {
            let f = self.theta_d(t) - theta_d;
            if f.abs() < 1e-13 {
                break;
            }
            if f > 0.0 { hi = t } else { lo = t }
            let d = self.dtheta_d(t);
            let next = if d > 0.0 { t - f / d } else { f64::NAN };
            t = if next.is_finite() && next > lo && next < hi { next } else { 0.5 * (lo + hi) };
        }
        Some(t)
    }

    /// Pixel of a camera-frame point (OpenCV axes); None past the fold angle. Unlike a pinhole this
    /// is defined behind the image plane (theta > 90 degrees) where the lens sees it.
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        let r = p[0].hypot(p[1]);
        let theta = r.atan2(p[2]);
        if !theta.is_finite() || theta > self.max_theta() {
            return None;
        }
        let (dx, dy) = if r > 1e-12 {
            let s = self.theta_d(theta) / r;
            (p[0] * s, p[1] * s)
        } else {
            // On the axis theta_d ~ theta ~ r / z.
            (p[0] / p[2], p[1] / p[2])
        };
        Some([self.fx * dx + self.cx, self.fy * dy + self.cy])
    }

    /// Unit ray (OpenCV camera axes) through pixel `uv` (pixel-centre convention of the caller);
    /// None past the fold.
    pub fn unproject(&self, uv: [f64; 2]) -> Option<[f64; 3]> {
        let (xd, yd) = ((uv[0] - self.cx) / self.fx, (uv[1] - self.cy) / self.fy);
        let theta_d = xd.hypot(yd);
        if theta_d < 1e-12 {
            return Some([0.0, 0.0, 1.0]);
        }
        let theta = self.theta(theta_d)?;
        let s = theta.sin() / theta_d;
        Some([xd * s, yd * s, theta.cos()])
    }

    /// Normalised pinhole coordinates (x/z, y/z) -> normalised fisheye coordinates.
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r = x.hypot(y);
        if r < 1e-12 {
            return (x, y);
        }
        let s = self.theta_d(r.atan()) / r;
        (x * s, y * s)
    }

    /// Normalised fisheye coordinates -> normalised pinhole coordinates; None at or beyond 90
    /// degrees off the axis (no pinhole image of those rays) or past the fold.
    pub fn undistort(&self, xd: f64, yd: f64) -> Option<(f64, f64)> {
        let theta_d = xd.hypot(yd);
        if theta_d < 1e-12 {
            return Some((xd, yd));
        }
        let theta = self.theta(theta_d)?;
        if theta >= std::f64::consts::FRAC_PI_2 - 1e-9 {
            return None;
        }
        let s = theta.tan() / theta_d;
        Some((xd * s, yd * s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens() -> Kb4 {
        // A GoPro-like OPENCV_FISHEYE at 1920x1080.
        Kb4 { fx: 880.0, fy: 878.0, cx: 960.5, cy: 540.25, k: [0.04, -0.012, 0.003, -0.0004] }
    }

    #[test]
    fn colmap_models_map_to_kb4() {
        let p = [500.0, 320.0, 240.0, 0.1, 0.02];
        assert_eq!(Kb4::from_params("RADIAL_FISHEYE", &p).unwrap().k, [0.1, 0.02, 0.0, 0.0]);
        assert_eq!(Kb4::from_params("SIMPLE_RADIAL_FISHEYE", &p[..4]).unwrap().fy, 500.0);
        let o = Kb4::from_params("OPENCV_FISHEYE", &[1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
        assert_eq!((o.fx, o.fy, o.cx, o.cy, o.k), (1., 2., 3., 4., [5., 6., 7., 8.]));
        assert!(Kb4::from_params("OPENCV", &[1., 2., 3., 4., 5., 6., 7., 8.]).is_none());
        assert!(Kb4::from_params("OPENCV_FISHEYE", &[1., 2., 3.]).is_none());
        assert!(is_fisheye_model("THIN_PRISM_FISHEYE") && !is_fisheye_model("OPENCV"));
    }

    #[test]
    fn equidistant_projection_matches_closed_form() {
        let l = Kb4 { fx: 300.0, fy: 300.0, cx: 0.0, cy: 0.0, k: [0.0; 4] };
        // 60 degrees off axis along +x: r = f * pi/3.
        let p = l.project([60f64.to_radians().sin(), 0.0, 60f64.to_radians().cos()]).unwrap();
        assert!((p[0] - 300.0 * std::f64::consts::FRAC_PI_3).abs() < 1e-9 && p[1].abs() < 1e-12);
        // 100 degrees: behind the image plane, still on the sensor.
        let p = l.project([100f64.to_radians().sin(), 0.0, 100f64.to_radians().cos()]).unwrap();
        assert!((p[0] - 300.0 * 100f64.to_radians()).abs() < 1e-9);
        assert_eq!(l.max_theta(), std::f64::consts::PI);
    }

    #[test]
    fn unproject_inverts_project() {
        let l = lens();
        for &(u, v) in &[(960.5, 540.25), (100.0, 80.0), (1900.0, 1000.0), (1500.0, 300.0)] {
            let ray = l.unproject([u, v]).unwrap();
            assert!((ray.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12);
            let back = l.project(ray.map(|x| x * 3.7)).unwrap();
            assert!((back[0] - u).abs() < 1e-6 && (back[1] - v).abs() < 1e-6, "{u},{v} -> {back:?}");
        }
    }

    #[test]
    fn undistort_inverts_distort_and_refuses_90_degrees() {
        let l = lens();
        for &(x, y) in &[(0.0, 0.0), (0.3, -0.2), (1.5, 0.9), (-4.0, 2.0)] {
            let (xd, yd) = l.distort(x, y);
            let (ux, uy) = l.undistort(xd, yd).unwrap();
            assert!((ux - x).abs() < 1e-9 && (uy - y).abs() < 1e-9, "{x},{y} -> {ux},{uy}");
        }
        let at_90 = l.theta_d(std::f64::consts::FRAC_PI_2);
        assert!(l.undistort(at_90 * 1.01, 0.0).is_none());
    }

    #[test]
    fn folding_polynomial_caps_the_angle() {
        // k1 < 0 large enough that theta_d peaks at theta = 1 rad (d/dtheta = 1 + 3 k1 theta^2).
        let l = Kb4 { fx: 100.0, fy: 100.0, cx: 0.0, cy: 0.0, k: [-1.0 / 3.0, 0.0, 0.0, 0.0] };
        assert!((l.max_theta() - 1.0).abs() < 2e-3);
        assert!(l.project([2f64.sin(), 0.0, 2f64.cos()]).is_none());
        assert!(l.theta(l.theta_d(1.0) * 1.1).is_none());
        assert!((l.theta(l.theta_d(0.5)).unwrap() - 0.5).abs() < 1e-9);
    }
}
