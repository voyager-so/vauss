//! Golden vectors: the cross-evaluator reference (`crates/splat-effects/golden.json`).
//!
//! 200 hashed splats x 6 graphs (4 nodes each, every node and mask type; the physical nodes with a floor) x 5 times
//! (0, 25, 50, 75, 100% of the duration). Per case and time it stores μ', linear α', the exact
//! covariance Σ' (xx xy xz yy yz zz, engine mode), the fast-mode q' (w,x,y,z, w >= 0) and linear
//! s', the DC coefficient and the max node weight. Compare Σ' or the fast-mode q'/s', never an
//! exact-mode q' (the eigen basis is not unique). Positions should match to 1e-4.

use glam::{Mat3, Quat, Vec3};
use serde_json::{Value, json};

use crate::hash::{pcg_hash, unit};
use crate::{CovMode, EvalOpts, Frame, Graph, Splats, evaluate};

pub const N: usize = 200;
pub const OBJECT_KEY: &str = "golden_selection";

/// The hashed rest splats (sh_degree 0).
pub fn rest() -> Splats {
    let mut g = Splats::default();
    for i in 0..N {
        let mut h = pcg_hash(i as u32 ^ 0x5EED_0001);
        let mut u = || {
            h = pcg_hash(h);
            unit(h)
        };
        g.pos.push([(u() - 0.5) * 4.0, (u() - 0.5) * 4.0 + 1.0, (u() - 0.5) * 4.0]);
        let q = Quat::from_xyzw(u() * 2.0 - 1.0, u() * 2.0 - 1.0, u() * 2.0 - 1.0, u() * 2.0 - 1.0 + 0.1).normalize();
        g.rot.push([q.w, q.x, q.y, q.z]);
        g.scale.push([-4.6 + 3.0 * u(), -4.6 + 3.0 * u(), -4.6 + 3.0 * u()]);
        g.opacity.push((u() - 0.5) * 8.0);
        g.dc.push([(u() - 0.5) * 2.0, (u() - 0.5) * 2.0, (u() - 0.5) * 2.0]);
    }
    g
}

/// Per-splat object weights for the object-mask case: a third 0, a third 1, a third fractional.
pub fn object_weights() -> Vec<f32> {
    (0..N as u32)
        .map(|i| match i % 3 {
            0 => 0.0,
            1 => 1.0,
            _ => unit(pcg_hash(i ^ 0x0B1E_C700)),
        })
        .collect()
}

pub fn graphs() -> Vec<Graph> {
    let src = [
        r#"{ "version": 1, "effect_id": "golden_vortex_room", "duration_s": 4, "seed": 42, "jacobian_step": 0.003,
          "nodes": [
            { "id": "twist", "type": "vortex", "mask": { "kind": "height", "min": 0.2, "max": 99, "falloff": 0.5 },
              "window": { "start_s": 0, "end_s": 4, "curve": "pulse" },
              "params": { "center": [0, 1.2, 0], "angle": 3.0, "angle_per_unit": 0.65, "lift": 0.8, "expand": 0.3 } },
            { "id": "ripple", "type": "wave", "mask": { "kind": "box", "min": [-1, -1, -1], "max": [1, 2, 1], "falloff": 0.4 },
              "window": { "start_s": 0.5, "end_s": 3.5, "curve": "linear" },
              "params": { "direction": [1, 0, 0.5], "wavelength": 1.5, "speed": 0.7, "amplitude": 0.2, "displace_axis": [0, 1, 0] } },
            { "id": "fade", "type": "opacity", "mask": { "kind": "sphere", "center": [0.5, 1, 0], "radius": 1.0, "falloff": 0.5 },
              "window": { "start_s": 0, "end_s": 4, "curve": "smooth" }, "params": { "to": 0.2 } },
            { "id": "gold", "type": "tint", "mask": { "kind": "invert", "mask": { "kind": "sphere", "center": [0, 1, 0], "radius": 1.5, "falloff": 0.3 } },
              "window": { "start_s": 0, "end_s": 4, "curve": "hold" }, "params": { "color": [1.0, 0.8, 0.3], "amount": 0.7 } }
          ] }"#,
        r#"{ "version": 1, "effect_id": "golden_explode_dissolve", "duration_s": 4, "seed": 7,
          "nodes": [
            { "id": "burst", "type": "explode", "window": { "start_s": 0, "end_s": 4, "curve": "pulse" },
              "params": { "center": [0, 1, 0], "distance": 1.5, "spin": 3.0, "jitter": 0.5 } },
            { "id": "melt", "type": "dissolve",
              "mask": { "kind": "mul", "masks": [ { "kind": "height", "axis": [0, 1, 0], "min": 0.5, "max": 99, "falloff": 0.5 },
                                                  { "kind": "sphere", "center": [0, 1, 0], "radius": 2.0, "falloff": 1.0 } ] },
              "window": { "start_s": 1, "end_s": 4, "curve": "smooth" },
              "params": { "direction": [0, 1, 0], "drift": 0.4, "softness": 0.15, "shrink": 0.5 } },
            { "id": "swirl", "type": "vortex", "window": { "start_s": 0, "end_s": 4, "curve": "out" },
              "params": { "center": [0.2, 0, -0.1], "axis": [0.3, 1, 0], "angle": 0.8 } },
            { "id": "cool", "type": "tint", "params": { "color": [0.6, 0.8, 1.2], "amount": 0.5 } }
          ] }"#,
        r#"{ "version": 1, "effect_id": "golden_morphs", "duration_s": 4, "seed": 3, "loop": true, "jacobian_step": 0.004,
          "nodes": [
            { "id": "ball", "type": "morph", "mask": { "kind": "box", "min": [-2, -1, -2], "max": [0, 3, 2], "falloff": 0.5 },
              "window": { "start_s": 0, "end_s": 4, "curve": "smooth" }, "params": { "target": "sphere", "center": [-1, 1, 0], "radius": 0.8 } },
            { "id": "tube", "type": "morph", "mask": { "kind": "height", "min": 1.0, "max": 2.5, "falloff": 0.3 },
              "window": { "start_s": 0, "end_s": 4, "curve": "pulse" }, "params": { "target": "cylinder", "center": [1, 0, 0], "radius": 0.6, "axis": [0, 1, 0.2] } },
            { "id": "flat", "type": "morph", "mask": { "kind": "sphere", "center": [1, 0, 1], "radius": 1.0, "falloff": 0.5 },
              "window": { "start_s": 0, "end_s": 4, "curve": "hold" }, "params": { "target": "plane", "center": [0, 0, 0], "axis": [0, 1, 0] } },
            { "id": "sway", "type": "wave", "params": { "direction": [0, 0, 1], "wavelength": 2.0, "speed": 0.5, "amplitude": 0.1, "displace_axis": [1, 0, 0] } }
          ] }"#,
        r#"{ "version": 1, "effect_id": "golden_object", "duration_s": 4, "seed": 1234,
          "nodes": [
            { "id": "vanish", "type": "dissolve", "mask": { "kind": "object", "selection": "golden_selection" },
              "window": { "start_s": 0, "end_s": 4, "curve": "linear" }, "params": { "drift": 0.3, "softness": 0.0, "shrink": 1.0 } },
            { "id": "scatter", "type": "explode", "mask": { "kind": "invert", "mask": { "kind": "object", "selection": "golden_selection" } },
              "window": { "start_s": 0, "end_s": 4, "curve": "smooth" }, "params": { "center": [0, 0, 0], "distance": 0.5, "spin": -2.0 } },
            { "id": "ghost", "type": "opacity", "mask": { "kind": "object", "selection": "golden_selection" },
              "window": { "start_s": 2, "end_s": 2, "curve": "linear" }, "params": { "to": 0.5 } },
            { "id": "twirl", "type": "vortex",
              "mask": { "kind": "mul", "masks": [ { "kind": "object", "selection": "golden_selection" }, { "kind": "box", "min": [-1, 0, -1], "max": [1, 2, 1] } ] },
              "window": { "start_s": 0, "end_s": 4, "curve": "pulse" }, "params": { "angle": 1.5, "angle_per_unit": 0.5, "lift": 0.2 } }
          ] }"#,
        r#"{ "version": 1, "effect_id": "golden_crumble_shatter", "duration_s": 4, "seed": 99, "floor": { "normal": [0, 1, 0], "offset": -1.2 },
          "nodes": [
            { "id": "dust", "type": "crumble", "mask": { "kind": "object", "selection": "golden_selection" },
              "params": { "start_s": 0.5, "sweep_s": 1.5, "burst": 0.6, "grain": 0.004, "restitution": 0.3, "friction": 0.5, "settle_s": 0.8, "pile_h": 0.2,
                          "front": "side", "jitter": 0.05, "wind": [0.2, 0, 0.1], "turbulence": 0.05, "haze": 0.2, "shadow": 0.5 } },
            { "id": "shards", "type": "shatter", "mask": { "kind": "box", "min": [-2, -1, -2], "max": [0, 3, 2] },
              "params": { "seeds": 12, "start_s": 0.3, "burst": 1.2, "spin": 5.0, "restitution": 0.4, "friction": 0.5 } },
            { "id": "back", "type": "crumble", "mask": { "kind": "sphere", "center": [1, 1, 1], "radius": 0.8 },
              "params": { "start_s": 0.2, "sweep_s": 1.0, "reverse": true } },
            { "id": "dim", "type": "tint", "mask": { "kind": "height", "min": -9, "max": 0.5 }, "params": { "color": [0.9, 0.9, 1.0], "amount": 1.0 } }
          ] }"#,
        r#"{ "version": 1, "effect_id": "golden_ripple_float_melt", "duration_s": 4, "seed": 5, "floor": { "normal": [0, 1, 0.1], "offset": -1.0 }, "aniso_max": 2.0,
          "nodes": [
            { "id": "wave", "type": "ripple", "mask": { "kind": "sphere", "center": [0, 0, 0], "radius": 2.0, "falloff": 1.0 },
              "params": { "source": [0.3, -1, -0.2], "start_s": 0.2, "speed": 1.2, "wavelength": 0.7, "amplitude": 0.15, "decay": 0.3, "packet_width": 1.5, "light_dir": [0.3, 0.9, 0.3], "shade": 1.2 } },
            { "id": "lift", "type": "float", "mask": { "kind": "box", "min": [0, 0, -2], "max": [2, 3, 0] },
              "params": { "start_s": 0.5, "rise": 0.6, "rise_s": 1.5, "spin": 0.8, "bob": 0.05, "bob_hz": 0.7 } },
            { "id": "goo", "type": "melt", "mask": { "kind": "box", "min": [-2, -1, 0], "max": [0, 3, 2] },
              "params": { "start_s": 0.4, "duration_s": 2.5, "spread": 0.5, "sheen": 0.2 } },
            { "id": "swirl", "type": "vortex", "mask": { "kind": "sphere", "center": [1, 0, 1], "radius": 0.7, "falloff": 0.3 },
              "window": { "start_s": 0, "end_s": 4, "curve": "pulse" }, "params": { "center": [1, 0, 1], "angle": 1.2 } }
          ] }"#,
        r#"{ "version": 1, "effect_id": "golden_ripple_rigid_float_tilt", "duration_s": 4, "seed": 17, "floor": { "normal": [0, 1, 0], "offset": -1.0 },
          "nodes": [
            { "id": "lawn", "type": "ripple", "mask": { "kind": "mul", "masks": [ { "kind": "height", "min": -9, "max": -0.5, "falloff": 0.3 },
                                                                                   { "kind": "invert", "mask": { "kind": "object", "selection": "golden_selection" } } ] },
              "params": { "source": [-0.5, -1, 0.4], "start_s": 0.1, "speed": 1.5, "wavelength": 0.9, "amplitude": 0.12, "decay": 0.1, "packet_width": 2.0, "shade": 1.0 } },
            { "id": "rock", "type": "ripple", "mask": { "kind": "object", "selection": "golden_selection" },
              "params": { "source": [-0.5, -1, 0.4], "start_s": 0.1, "speed": 1.5, "wavelength": 0.9, "amplitude": 0.12, "decay": 0.1, "packet_width": 2.0, "shade": 1.0, "rigid": true } },
            { "id": "hover", "type": "float", "mask": { "kind": "box", "min": [-2, 0, -2], "max": [0, 3, 0] },
              "params": { "start_s": 0.3, "rise": 0.5, "rise_s": 1.0, "spin": 0.4, "bob": 0.04, "bob_hz": 0.5, "down_s": 2.2, "tilt": 0.25, "tilt_dir": [1, 0.5, -0.5], "turn": 6.2831853, "pivot": [0.6, 0.2, 0.3], "land_hard": 1.0, "shadow": 0.6, "shadow_soft": 0.4 } }
          ] }"#,
    ];
    src.iter().map(|s| Graph::from_json(s).expect("golden graph is valid")).collect()
}

fn alpha(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

fn cov6(rot: [f32; 4], log_scale: [f32; 3]) -> [f32; 6] {
    let [w, x, y, z] = rot;
    let r = Mat3::from_quat(Quat::from_xyzw(x, y, z, w));
    let m = r * Mat3::from_diagonal(Vec3::from(log_scale).exp());
    let c = m * m.transpose();
    [c.x_axis.x, c.y_axis.x, c.z_axis.x, c.y_axis.y, c.z_axis.y, c.z_axis.z]
}

fn exp3(s: [f32; 3]) -> [f32; 3] {
    Vec3::from(s).exp().into()
}

/// The full golden document from the CPU reference evaluator.
pub fn generate() -> Value {
    generate_with(&|rest, g, t, opts| evaluate(rest, g, t, opts).expect("eval"))
}

/// The golden document with any evaluator (the GPU parity check passes the WGSL one).
pub fn generate_with(eval: &dyn Fn(&Splats, &Graph, f32, &EvalOpts) -> Frame) -> Value {
    let rest = rest();
    let mut opts = EvalOpts::default();
    opts.objects.insert(OBJECT_KEY.into(), object_weights());
    let mut cases = Vec::new();
    for g in graphs() {
        let h = crate::Prepared::new(&rest, &g, &opts).expect("prepare").jacobian_step();
        let mut frames = Vec::new();
        for k in 0..5 {
            let t = g.duration_s * k as f32 / 4.0;
            let ex: Frame = eval(&rest, &g, t, &opts);
            let fast_opts = EvalOpts { cov: CovMode::Fast, ..opts.clone() };
            let fa: Frame = eval(&rest, &g, t, &fast_opts);
            frames.push(json!({
                "t": t,
                "invalid": ex.invalid,
                "selected": ex.selected,
                "pos": ex.pos,
                "alpha": ex.opacity.iter().map(|l| alpha(*l)).collect::<Vec<_>>(),
                "cov": ex.rot.iter().zip(&ex.scale).map(|(r, s)| cov6(*r, *s)).collect::<Vec<_>>(),
                "fast_rot": fa.rot,
                "fast_scale": fa.scale.iter().map(|s| exp3(*s)).collect::<Vec<_>>(),
                "dc": ex.dc,
                "weight": ex.weight,
            }));
        }
        // The resolved graph (floor, gravity, aniso_max, node boxes filled in): what viewers read.
        let resolved = crate::eval::resolve(&rest, &g, &opts).expect("resolve");
        cases.push(json!({ "graph": serde_json::to_value(&resolved).expect("graph"), "jacobian_step": h, "frames": frames }));
    }
    json!({
        "about": "splat-effects golden vectors (effect graph v1). Regenerate: cargo run -p splat-effects --bin splat-effects-golden -- --write. \
                  rest.rot and fast_rot are w,x,y,z unit quaternions; scale/fast_scale linear; alpha linear; cov = exact-mode Σ' as xx xy xz yy yz zz; \
                  dc = SH DC coefficient (base colour 0.5 + 0.2820948·dc). Hash floats are (h >> 8)·2^-24. SH view direction: unchanged.",
        "n": N,
        "rest": {
            "pos": rest.pos,
            "rot": rest.rot,
            "log_scale": rest.scale,
            "scale": rest.scale.iter().map(|s| exp3(*s)).collect::<Vec<_>>(),
            "opacity_logit": rest.opacity,
            "alpha": rest.opacity.iter().map(|l| alpha(*l)).collect::<Vec<_>>(),
            "dc": rest.dc,
        },
        "objects": { OBJECT_KEY: object_weights() },
        "cases": cases,
    })
}

/// Every compared field. Fast-mode q' is ill-defined where J is singular (planar morphs), so
/// GPU checks gate on [`EXACT_FIELDS`] and report `fast_rot` separately.
pub const ALL_FIELDS: &[&str] = &["pos", "alpha", "cov", "fast_rot", "fast_scale", "dc", "weight"];
pub const EXACT_FIELDS: &[&str] = &["pos", "alpha", "cov", "fast_scale", "dc", "weight"];

/// Compare two golden documents over `fields`. Returns the max error per field, or a mismatch
/// report (positions/alpha/dc/rot absolute; Σ' relative to the splat's largest entry; s' relative).
pub fn compare(expected: &Value, got: &Value, tol: f64, fields: &[&str]) -> Result<Vec<(String, f64)>, String> {
    let mut worst: std::collections::BTreeMap<String, f64> = Default::default();
    let mut bad = Vec::new();
    let (ec, gc) = (expected["cases"].as_array().ok_or("no cases")?, got["cases"].as_array().ok_or("no cases")?);
    if ec.len() != gc.len() {
        return Err(format!("{} cases expected, got {}", ec.len(), gc.len()));
    }
    for (ci, (e, g)) in ec.iter().zip(gc).enumerate() {
        let id = e["graph"]["effect_id"].as_str().unwrap_or("?").to_owned();
        for (fi, (ef, gf)) in e["frames"].as_array().ok_or("frames")?.iter().zip(g["frames"].as_array().ok_or("frames")?).enumerate() {
            for &field in fields {
                let (mut a, mut b) = (flat(&ef[field]), flat(&gf[field]));
                // Σ' and s' are compared relative to the splat's own size (Σ' entries are ~1e-6).
                if field == "cov" && a.len() == b.len() {
                    for (ea, eb) in a.chunks_mut(6).zip(b.chunks_mut(6)) {
                        let m = ea.iter().fold(1e-30f64, |m, x| m.max(x.abs()));
                        ea.iter_mut().chain(eb.iter_mut()).for_each(|x| *x /= m);
                    }
                }
                if field == "fast_scale" && a.len() == b.len() {
                    for (x, y) in a.iter_mut().zip(b.iter_mut()) {
                        let m = x.abs().max(1e-6);
                        *x /= m;
                        *y /= m;
                    }
                }
                if a.len() != b.len() {
                    return Err(format!("case {ci} ({id}) frame {fi} {field}: {} values expected, got {}", a.len(), b.len()));
                }
                for (k, (x, y)) in a.iter().zip(&b).enumerate() {
                    let d = (x - y).abs();
                    if !(d <= tol) {
                        bad.push(format!("case {ci} ({id}) frame {fi} {field}[{k}]: expected {x}, got {y} (|d| {d:.3e} > {tol:.0e})"));
                    }
                    let w = worst.entry(field.to_owned()).or_insert(0.0);
                    *w = w.max(if d.is_nan() { f64::INFINITY } else { d });
                }
            }
        }
    }
    if !bad.is_empty() {
        let n = bad.len();
        bad.truncate(12);
        return Err(format!("{n} values out of tolerance; worst per field {worst:?}; first:\n  {}", bad.join("\n  ")));
    }
    Ok(worst.into_iter().collect())
}

fn flat(v: &Value) -> Vec<f64> {
    let mut out = Vec::new();
    fn go(v: &Value, out: &mut Vec<f64>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| go(x, out)),
            Value::Number(n) => out.push(n.as_f64().unwrap_or(f64::NAN)),
            _ => out.push(f64::NAN),
        }
    }
    go(v, &mut out);
    out
}
