//! Effect graph v1: `*.effect.json` types, parsing and validation.
//!
//! The types below are the contract. Parsing validates, so a `Graph`
//! in hand is always well formed; errors carry a JSON path (`nodes[0] (twist).params.axis: ...`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type V3 = [f32; 3];

#[derive(Debug, thiserror::Error)]
#[error("{path}: {msg}")]
pub struct GraphError {
    pub path: String,
    pub msg: String,
}

fn err(path: impl Into<String>, msg: impl Into<String>) -> GraphError {
    GraphError { path: path.into(), msg: msg.into() }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub version: u32,
    pub effect_id: String,
    pub duration_s: f32,
    #[serde(default = "default_fps")]
    pub fps: f32,
    #[serde(default, rename = "loop")]
    pub looping: bool,
    #[serde(default)]
    pub seed: u32,
    /// Central-difference step for the Jacobian, in scene units. Absent: 1e-3 x scene radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jacobian_step: Option<f32>,
    /// Floor plane `normal·p = offset` for the physical nodes. Absent: the package's
    /// collision/floor.json (`EvalOpts::floor`), else the up axis through the origin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor: Option<Floor>,
    /// Gravity in scene units/s². Absent: 9.81 × `EvalOpts::units_per_meter`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity: Option<f32>,
    /// Anisotropy clamp on every deformed splat: max/min axis ≤ max(rest ratio, 1) × aniso_max.
    /// Absent: [`DEFAULT_ANISO_MAX`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aniso_max: Option<f32>,
    pub nodes: Vec<Node>,
}

pub const DEFAULT_ANISO_MAX: f32 = 1.5;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Floor {
    pub normal: V3,
    pub offset: f32,
}

/// A resolved axis-aligned box (the rest positions of a node's mask members).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bbox {
    pub min: V3,
    pub max: V3,
}

fn default_fps() -> f32 {
    30.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawNode", into = "RawNode")]
pub struct Node {
    pub id: String,
    pub mask: Mask,
    /// Absent: the node is fully on at every t (curve weight 1).
    pub window: Option<Window>,
    pub op: Op,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNode {
    id: String,
    #[serde(rename = "type")]
    ty: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mask: Option<Mask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    window: Option<Window>,
    #[serde(default)]
    params: Value,
}

impl TryFrom<RawNode> for Node {
    type Error = String;
    fn try_from(r: RawNode) -> Result<Self, String> {
        let params = if r.params.is_null() { Value::Object(Default::default()) } else { r.params };
        let op: Op = serde_json::from_value(serde_json::json!({ "type": r.ty, "params": params }))
            .map_err(|e| format!("node {:?} ({}): {e}", r.id, r.ty))?;
        Ok(Self { id: r.id, mask: r.mask.unwrap_or(Mask::All), window: r.window, op })
    }
}

impl From<Node> for RawNode {
    fn from(n: Node) -> Self {
        let v = serde_json::to_value(&n.op).expect("op serializes");
        let ty = v["type"].as_str().expect("op type").to_owned();
        let mask = if n.mask == Mask::All { None } else { Some(n.mask) };
        Self { id: n.id, ty, mask, window: n.window, params: v["params"].clone() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Window {
    pub start_s: f32,
    pub end_s: f32,
    #[serde(default)]
    pub curve: Curve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Curve {
    #[default]
    Linear,
    Smooth,
    Pulse,
    Hold,
    Out,
}

/// Region weight in [0, 1], evaluated on the rest position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Mask {
    All,
    Box {
        min: V3,
        max: V3,
        #[serde(default)]
        falloff: f32,
    },
    Sphere {
        center: V3,
        radius: f32,
        #[serde(default)]
        falloff: f32,
    },
    Height {
        /// Absent: the scene's up axis.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        axis: Option<V3>,
        #[serde(default = "neg_big")]
        min: f32,
        #[serde(default = "big")]
        max: f32,
        #[serde(default)]
        falloff: f32,
    },
    Object(ObjectMask),
    Invert {
        mask: Box<Mask>,
    },
    Mul {
        masks: Vec<Mask>,
    },
}

fn big() -> f32 {
    1e30
}
fn neg_big() -> f32 {
    -1e30
}

/// Select-by-name selection (`selected_ids.npy` / `object_ids.npy`, splat-core edit.rs). The
/// caller resolves it to per-splat weights and passes them in [`crate::EvalOpts::objects`] under
/// [`ObjectMask::key`]. `box` is the exported box-with-falloff stand-in for viewers without
/// per-splat ids; the engine uses it only when no weights were supplied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectMask {
    pub selection: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_ids: Option<Vec<u32>>,
    #[serde(default, rename = "box", skip_serializing_if = "Option::is_none")]
    pub bbox: Option<BoxRegion>,
    /// More selections merged in (per-splat max), resolved by the caller.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub union: Vec<String>,
    /// Grow the selection by this distance (scene units, 3D) around its splats.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dilate: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxRegion {
    pub min: V3,
    pub max: V3,
    #[serde(default)]
    pub falloff: f32,
}

impl ObjectMask {
    /// Key for [`crate::EvalOpts::objects`]: `selection`, or `selection#id1,id2` with object ids.
    pub fn key(&self) -> String {
        let mut k = match &self.object_ids {
            None => self.selection.clone(),
            Some(ids) => {
                let ids: Vec<String> = ids.iter().map(u32::to_string).collect();
                format!("{}#{}", self.selection, ids.join(","))
            }
        };
        for u in &self.union {
            k.push('+');
            k.push_str(u);
        }
        if let Some(d) = self.dilate {
            k.push_str(&format!("~{d}"));
        }
        k
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "params", rename_all = "snake_case")]
pub enum Op {
    Vortex(Vortex),
    Wave(Wave),
    Explode(Explode),
    Dissolve(Dissolve),
    Morph(Morph),
    Opacity(Opacity),
    Tint(Tint),
    Crumble(Crumble),
    Shatter(Shatter),
    Ripple(Ripple),
    Float(Float),
    Melt(Melt),
}

// ---- physical nodes (effect graph v1). Closed-form in (rest, graph, t); they
// take `start_s` instead of a `window`. `bbox` is the members' rest box (mask weight >= 0.5),
// filled from the rest splats when absent (`eval::resolve`), so viewers never scan splats. ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Crumble {
    #[serde(default = "d_start")]
    pub start_s: f32,
    #[serde(default = "d_sweep")]
    pub sweep_s: f32,
    /// Outward horizontal launch speed (units/s).
    #[serde(default = "d_burst")]
    pub burst: f32,
    /// Upward launch speed (units/s), × (0.5 + r).
    #[serde(default = "d_up_kick")]
    pub up_kick: f32,
    /// Grain size: every scale axis → min(s, grain) over the first 0.2 s of flight.
    #[serde(default = "d_grain")]
    pub grain: f32,
    #[serde(default = "d_restitution")]
    pub restitution: f32,
    #[serde(default = "d_friction")]
    pub friction: f32,
    /// Time after landing over which a grain rises onto the heap.
    #[serde(default = "d_settle")]
    pub settle_s: f32,
    #[serde(default = "d_pile_h")]
    pub pile_h: f32,
    /// Heap radius. Absent: 0.75 × the bbox's horizontal half-diagonal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pile_r: Option<f32>,
    /// Colour factor once landed (reads as dust).
    #[serde(default = "d_dust")]
    pub dust: f32,
    /// Desaturation (0..1) once landed, toward the grain's own luminance.
    #[serde(default = "d_dust_desat")]
    pub dust_desat: f32,
    /// Play t → duration − t (reassembly).
    #[serde(default)]
    pub reverse: bool,
    /// Where the disintegration front starts: `top` (sweeps down), `bottom` or `side`.
    #[serde(default)]
    pub front: FrontKind,
    /// Front travel direction (overrides `front`). `side` without it: the horizontal of +x.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub front_dir: Option<V3>,
    /// Release jitter as a fraction of `sweep_s`: small = a crisp front.
    #[serde(default = "d_jitter")]
    pub jitter: f32,
    /// Grain size range as factors of `grain` (log-uniform per splat): flakes to fine dust.
    #[serde(default = "d_size_min")]
    pub size_min: f32,
    #[serde(default = "d_size_max")]
    pub size_max: f32,
    /// Constant wind velocity (units/s) while airborne.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub wind: V3,
    /// Horizontal turbulence amplitude (units) and spatial frequency (1/units).
    #[serde(default = "d_turbulence")]
    pub turbulence: f32,
    #[serde(default = "d_turb_freq")]
    pub turb_freq: f32,
    /// Fraction of released splats that become haze puffs, their size, peak alpha factor,
    /// lifetime and rise speed.
    #[serde(default = "d_haze")]
    pub haze: f32,
    #[serde(default = "d_haze_size")]
    pub haze_size: f32,
    #[serde(default = "d_haze_alpha")]
    pub haze_alpha: f32,
    #[serde(default = "d_haze_s")]
    pub haze_s: f32,
    #[serde(default = "d_haze_rise")]
    pub haze_rise: f32,
    /// Front irregularity: release-height noise amplitude (fraction of the sweep range) and its
    /// spatial frequency (1/units).
    #[serde(default = "d_front_noise")]
    pub front_noise: f32,
    #[serde(default = "d_front_freq")]
    pub front_freq: f32,
    /// Fraction of grains that are chunky fragments (CHUNK_SCALE × size).
    #[serde(default = "d_chunk_frac")]
    pub chunk_frac: f32,
    /// Released grains are clamped to ±palette_tol of this colour. Absent: resolved to the
    /// members' mean base colour.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palette: Option<V3>,
    #[serde(default = "d_palette_tol")]
    pub palette_tol: f32,
    /// Grains shrink by this fraction as they settle: big flecks in the air, a fine crisp heap.
    #[serde(default = "d_land_shrink")]
    pub land_shrink: f32,
    /// Per-grain brightness jitter (± fraction).
    #[serde(default = "d_grain_jitter")]
    pub grain_jitter: f32,
    /// Minimum opacity of released flakes (0..1): faint interior splats become solid grains.
    #[serde(default = "d_grain_opacity")]
    pub grain_opacity: f32,
    /// Mound self-shadowing: the lowest grains darken by this factor (0..1).
    #[serde(default = "d_heap_ao")]
    pub heap_ao: f32,
    /// How far landed grains pull from their landing spot toward the footprint centre (0..1).
    #[serde(default = "d_gather")]
    pub gather: f32,
    /// Contact shadow: floor splats under the pile darken by up to this factor.
    #[serde(default = "d_shadow")]
    pub shadow: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Bbox>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontKind {
    #[default]
    Top,
    Bottom,
    Side,
}

fn is_zero3(v: &V3) -> bool {
    *v == [0.0; 3]
}
fn d_jitter() -> f32 { 0.02 }
fn d_size_min() -> f32 { 0.3 }
fn d_size_max() -> f32 { 3.0 }
fn d_turbulence() -> f32 { 0.03 }
fn d_turb_freq() -> f32 { 3.0 }
fn d_haze() -> f32 { 0.0 }
fn d_gather() -> f32 { 0.6 }
fn d_heap_ao() -> f32 { 0.55 }
fn d_grain_opacity() -> f32 { 0.9 }
fn d_front_noise() -> f32 { 0.04 }
fn d_front_freq() -> f32 { 9.0 }
fn d_chunk_frac() -> f32 { 0.03 }
fn d_palette_tol() -> f32 { 0.15 }
fn d_grain_jitter() -> f32 { 0.1 }
fn d_land_shrink() -> f32 { 0.5 }
fn d_haze_size() -> f32 { 0.03 }
fn d_haze_alpha() -> f32 { 0.25 }
fn d_haze_s() -> f32 { 2.5 }
fn d_haze_rise() -> f32 { 0.12 }
fn d_shadow() -> f32 { 0.45 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shatter {
    /// Voronoi seeds K (chunks).
    #[serde(default = "d_seeds")]
    pub seeds: u32,
    #[serde(default = "d_start")]
    pub start_s: f32,
    #[serde(default = "d_shatter_burst")]
    pub burst: f32,
    #[serde(default = "d_up_kick")]
    pub up_kick: f32,
    /// Spin rate (rad/s) at flight, × (0.5 + r) per chunk.
    #[serde(default = "d_spin")]
    pub spin: f32,
    #[serde(default = "d_shatter_restitution")]
    pub restitution: f32,
    #[serde(default = "d_friction")]
    pub friction: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Bbox>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ripple {
    pub source: V3,
    #[serde(default = "d_start")]
    pub start_s: f32,
    #[serde(default = "d_speed")]
    pub speed: f32,
    #[serde(default = "d_wavelength")]
    pub wavelength: f32,
    #[serde(default = "d_amplitude")]
    pub amplitude: f32,
    #[serde(default = "d_decay")]
    pub decay: f32,
    #[serde(default = "one")]
    pub packet_width: f32,
    #[serde(default = "d_light")]
    pub light_dir: V3,
    #[serde(default = "d_shade")]
    pub shade: f32,
    /// Rigid object mode: the members (mask >= 0.5) lift and tilt as ONE body, by the wave
    /// height and slope sampled at the bbox's floor footprint centre. Use one rigid ripple node
    /// per object and exclude the objects from the floor ripple's mask.
    #[serde(default)]
    pub rigid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Bbox>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Float {
    #[serde(default = "d_start")]
    pub start_s: f32,
    /// Rise height (units).
    #[serde(default = "d_rise")]
    pub rise: f32,
    /// Time to reach full rise.
    #[serde(default = "d_rise_s")]
    pub rise_s: f32,
    /// Spin rate (rad/s) about the vertical through the bbox centre.
    #[serde(default = "d_float_spin")]
    pub spin: f32,
    /// Bob amplitude (units) and frequency (Hz).
    #[serde(default = "d_bob")]
    pub bob: f32,
    #[serde(default = "d_bob_hz")]
    pub bob_hz: f32,
    /// Time the object starts settling back down (over `rise_s`); its spin stops once down.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub down_s: Option<f32>,
    /// Lean (rad) toward the horizontal part of `tilt_dir` (in the object's own frame, so it
    /// turns with it), eased in and out with the lift.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub tilt: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tilt_dir: Option<V3>,
    /// Flight: total yaw (rad) about the vertical through bbox centre + horiz(`pivot`), eased
    /// from mid-rise to mid-descent (to start + 2.5·rise_s without `down_s`). With `pivot` set
    /// and turn = 2π the object flies a circle and comes back to land where it started.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub turn: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pivot: Option<V3>,
    /// Descent profile in [0, 1]: 0 eases down (smoothstep), 1 lands hard (accelerates into
    /// the touchdown, x²).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub land_hard: f32,
    /// Ground shadow strength in [0, 1]: non-member splats under the object darken in a soft
    /// ellipse (the bbox footprint) that follows it, growing and fading with height.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub shadow: f32,
    /// Shadow blur growth per unit of height above the receiving splat.
    #[serde(default = "d_shadow_soft")]
    pub shadow_soft: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Bbox>,
}

fn d_shadow_soft() -> f32 {
    0.5
}

fn is_zero(v: &f32) -> bool {
    *v == 0.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Melt {
    #[serde(default = "d_start")]
    pub start_s: f32,
    #[serde(default = "d_melt_dur")]
    pub duration_s: f32,
    /// Horizontal spread factor at full melt.
    #[serde(default = "d_spread")]
    pub spread: f32,
    /// Colour gain at full melt (sheen).
    #[serde(default = "d_sheen")]
    pub sheen: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Bbox>,
}

fn d_start() -> f32 { 1.0 }
fn d_sweep() -> f32 { 1.5 }
fn d_burst() -> f32 { 0.4 }
fn d_up_kick() -> f32 { 0.6 }
fn d_grain() -> f32 { 0.005 }
fn d_restitution() -> f32 { 0.3 }
fn d_friction() -> f32 { 0.6 }
fn d_settle() -> f32 { 1.0 }
fn d_pile_h() -> f32 { 0.1 }
fn d_dust() -> f32 { 0.85 }
fn d_dust_desat() -> f32 { 0.6 }
fn d_seeds() -> u32 { 24 }
fn d_shatter_burst() -> f32 { 1.0 }
fn d_spin() -> f32 { 4.0 }
fn d_shatter_restitution() -> f32 { 0.35 }
fn d_speed() -> f32 { 1.5 }
fn d_wavelength() -> f32 { 0.8 }
fn d_amplitude() -> f32 { 0.05 }
fn d_decay() -> f32 { 0.2 }
fn d_light() -> V3 { [0.3, 0.9, 0.3] }
fn d_shade() -> f32 { 0.6 }
fn d_rise() -> f32 { 0.5 }
fn d_rise_s() -> f32 { 2.0 }
fn d_float_spin() -> f32 { 0.3 }
fn d_bob() -> f32 { 0.03 }
fn d_bob_hz() -> f32 { 0.4 }
fn d_melt_dur() -> f32 { 3.0 }
fn d_spread() -> f32 { 0.6 }
fn d_sheen() -> f32 { 0.15 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vortex {
    #[serde(default)]
    pub center: V3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<V3>,
    /// Radians at full weight.
    #[serde(default)]
    pub angle: f32,
    /// Extra radians per scene unit of height along the axis.
    #[serde(default)]
    pub angle_per_unit: f32,
    #[serde(default)]
    pub lift: f32,
    #[serde(default)]
    pub expand: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wave {
    /// Used as given (not normalised): its length scales the spatial frequency.
    pub direction: V3,
    pub wavelength: f32,
    #[serde(default)]
    pub speed: f32,
    pub amplitude: f32,
    /// Used as given. Absent: the scene's up axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub displace_axis: Option<V3>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Explode {
    #[serde(default)]
    pub center: V3,
    pub distance: f32,
    /// Radians of rigid per-splat spin at full weight.
    #[serde(default)]
    pub spin: f32,
    #[serde(default)]
    pub jitter: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dissolve {
    /// Absent: the scene's up axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<V3>,
    #[serde(default)]
    pub drift: f32,
    #[serde(default = "default_softness")]
    pub softness: f32,
    #[serde(default)]
    pub shrink: f32,
}

fn default_softness() -> f32 {
    0.1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphTarget {
    Sphere,
    Cylinder,
    Plane,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Morph {
    pub target: MorphTarget,
    #[serde(default)]
    pub center: V3,
    #[serde(default = "one")]
    pub radius: f32,
    /// Cylinder axis or plane normal. Absent: the scene's up axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<V3>,
}

fn one() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Opacity {
    pub to: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tint {
    pub color: V3,
    #[serde(default = "one")]
    pub amount: f32,
}

impl Op {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Vortex(_) => "vortex",
            Self::Wave(_) => "wave",
            Self::Explode(_) => "explode",
            Self::Dissolve(_) => "dissolve",
            Self::Morph(_) => "morph",
            Self::Opacity(_) => "opacity",
            Self::Tint(_) => "tint",
            Self::Crumble(_) => "crumble",
            Self::Shatter(_) => "shatter",
            Self::Ripple(_) => "ripple",
            Self::Float(_) => "float",
            Self::Melt(_) => "melt",
        }
    }

    /// Physical nodes: closed-form motion from `start_s`, no `window`.
    pub fn is_physical(&self) -> bool {
        matches!(self, Self::Crumble(_) | Self::Shatter(_) | Self::Ripple(_) | Self::Float(_) | Self::Melt(_))
    }
}

impl Graph {
    /// Parse and validate an `*.effect.json`.
    pub fn from_json(s: &str) -> Result<Self, GraphError> {
        let g: Self = serde_json::from_str(s).map_err(|e| err("effect.json", e.to_string()))?;
        g.validate()?;
        Ok(g)
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("graph serializes")
    }

    /// Every object mask in the graph (nested ones included), for the caller to resolve.
    pub fn object_masks(&self) -> Vec<&ObjectMask> {
        fn walk<'a>(m: &'a Mask, out: &mut Vec<&'a ObjectMask>) {
            match m {
                Mask::Object(o) => out.push(o),
                Mask::Invert { mask } => walk(mask, out),
                Mask::Mul { masks } => masks.iter().for_each(|m| walk(m, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for n in &self.nodes {
            walk(&n.mask, &mut out);
        }
        out
    }

    /// Check every field the evaluators rely on. `from_json` already calls this.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.version != 1 {
            return Err(err("version", format!("unsupported version {} (expected 1)", self.version)));
        }
        if self.effect_id.trim().is_empty() {
            return Err(err("effect_id", "must be non-empty"));
        }
        pos("duration_s", self.duration_s)?;
        pos("fps", self.fps)?;
        if let Some(h) = self.jacobian_step {
            pos("jacobian_step", h)?;
        }
        if let Some(f) = &self.floor {
            nonzero3("floor.normal", f.normal)?;
            fin("floor.offset", f.offset)?;
        }
        if let Some(g) = self.gravity {
            pos("gravity", g)?;
        }
        if let Some(a) = self.aniso_max {
            if !(a.is_finite() && a >= 1.0) {
                return Err(err("aniso_max", format!("must be >= 1, got {a}")));
            }
        }
        let mut ids = std::collections::HashSet::new();
        for (k, n) in self.nodes.iter().enumerate() {
            let p = format!("nodes[{k}] ({})", n.id);
            if n.id.trim().is_empty() {
                return Err(err(format!("nodes[{k}].id"), "must be non-empty"));
            }
            if !ids.insert(n.id.as_str()) {
                return Err(err(format!("{p}.id"), "duplicate node id"));
            }
            validate_mask(&n.mask, &format!("{p}.mask"))?;
            if n.op.is_physical() && n.window.is_some() {
                return Err(err(format!("{p}.window"), format!("{} takes params.start_s, not a window", n.op.type_name())));
            }
            if let Some(w) = &n.window {
                fin(&format!("{p}.window.start_s"), w.start_s)?;
                fin(&format!("{p}.window.end_s"), w.end_s)?;
                if w.end_s < w.start_s {
                    return Err(err(format!("{p}.window"), format!("end_s {} is before start_s {}", w.end_s, w.start_s)));
                }
            }
            validate_op(&n.op, &format!("{p}.params"))?;
        }
        Ok(())
    }
}

fn fin(path: &str, v: f32) -> Result<(), GraphError> {
    if v.is_finite() { Ok(()) } else { Err(err(path, format!("must be finite, got {v}"))) }
}
fn pos(path: &str, v: f32) -> Result<(), GraphError> {
    if v.is_finite() && v > 0.0 { Ok(()) } else { Err(err(path, format!("must be > 0, got {v}"))) }
}
fn nonneg(path: &str, v: f32) -> Result<(), GraphError> {
    if v.is_finite() && v >= 0.0 { Ok(()) } else { Err(err(path, format!("must be >= 0, got {v}"))) }
}
fn fin3(path: &str, v: V3) -> Result<(), GraphError> {
    if v.iter().all(|x| x.is_finite()) { Ok(()) } else { Err(err(path, format!("must be finite, got {v:?}"))) }
}
fn nonzero3(path: &str, v: V3) -> Result<(), GraphError> {
    fin3(path, v)?;
    if v.iter().map(|x| x * x).sum::<f32>() > 1e-12 { Ok(()) } else { Err(err(path, "must be non-zero")) }
}
fn opt_axis(path: &str, v: Option<V3>) -> Result<(), GraphError> {
    v.map_or(Ok(()), |v| nonzero3(path, v))
}

fn validate_mask(m: &Mask, p: &str) -> Result<(), GraphError> {
    match m {
        Mask::All => Ok(()),
        Mask::Box { min, max, falloff } => validate_box(p, *min, *max, *falloff),
        Mask::Sphere { center, radius, falloff } => {
            fin3(&format!("{p}.center"), *center)?;
            nonneg(&format!("{p}.radius"), *radius)?;
            nonneg(&format!("{p}.falloff"), *falloff)
        }
        Mask::Height { axis, min, max, falloff } => {
            opt_axis(&format!("{p}.axis"), *axis)?;
            fin(&format!("{p}.min"), *min)?;
            fin(&format!("{p}.max"), *max)?;
            if max < min {
                return Err(err(p, format!("max {max} is below min {min}")));
            }
            nonneg(&format!("{p}.falloff"), *falloff)
        }
        Mask::Object(o) => {
            if o.selection.trim().is_empty() {
                return Err(err(format!("{p}.selection"), "must name a selection artifact"));
            }
            if let Some(b) = &o.bbox {
                validate_box(&format!("{p}.box"), b.min, b.max, b.falloff)?;
            }
            if let Some(d) = o.dilate {
                nonneg(&format!("{p}.dilate"), d)?;
            }
            Ok(())
        }
        Mask::Invert { mask } => validate_mask(mask, &format!("{p}.mask")),
        Mask::Mul { masks } => {
            if masks.is_empty() {
                return Err(err(format!("{p}.masks"), "must hold at least one mask"));
            }
            for (k, m) in masks.iter().enumerate() {
                validate_mask(m, &format!("{p}.masks[{k}]"))?;
            }
            Ok(())
        }
    }
}

fn validate_box(p: &str, min: V3, max: V3, falloff: f32) -> Result<(), GraphError> {
    fin3(&format!("{p}.min"), min)?;
    fin3(&format!("{p}.max"), max)?;
    if (0..3).any(|k| max[k] < min[k]) {
        return Err(err(p, format!("max {max:?} is below min {min:?} on some axis")));
    }
    nonneg(&format!("{p}.falloff"), falloff)
}

fn validate_op(op: &Op, p: &str) -> Result<(), GraphError> {
    let f = |k: &str| format!("{p}.{k}");
    match op {
        Op::Vortex(v) => {
            fin3(&f("center"), v.center)?;
            opt_axis(&f("axis"), v.axis)?;
            for (k, x) in [("angle", v.angle), ("angle_per_unit", v.angle_per_unit), ("lift", v.lift), ("expand", v.expand)] {
                fin(&f(k), x)?;
            }
        }
        Op::Wave(w) => {
            nonzero3(&f("direction"), w.direction)?;
            pos(&f("wavelength"), w.wavelength)?;
            fin(&f("speed"), w.speed)?;
            fin(&f("amplitude"), w.amplitude)?;
            opt_axis(&f("displace_axis"), w.displace_axis)?;
        }
        Op::Explode(e) => {
            fin3(&f("center"), e.center)?;
            fin(&f("distance"), e.distance)?;
            fin(&f("spin"), e.spin)?;
            nonneg(&f("jitter"), e.jitter)?;
        }
        Op::Dissolve(d) => {
            opt_axis(&f("direction"), d.direction)?;
            fin(&f("drift"), d.drift)?;
            if !(0.0..=0.5).contains(&d.softness) {
                return Err(err(f("softness"), format!("must be in [0, 0.5], got {}", d.softness)));
            }
            if !(0.0..=1.0).contains(&d.shrink) {
                return Err(err(f("shrink"), format!("must be in [0, 1], got {}", d.shrink)));
            }
        }
        Op::Morph(m) => {
            fin3(&f("center"), m.center)?;
            nonneg(&f("radius"), m.radius)?;
            opt_axis(&f("axis"), m.axis)?;
        }
        Op::Opacity(o) => nonneg(&f("to"), o.to)?,
        Op::Tint(t) => {
            fin3(&f("color"), t.color)?;
            if t.color.iter().any(|c| *c < 0.0) {
                return Err(err(f("color"), "components must be >= 0"));
            }
            fin(&f("amount"), t.amount)?;
        }
        Op::Crumble(c) => {
            fin(&f("start_s"), c.start_s)?;
            nonneg(&f("sweep_s"), c.sweep_s)?;
            fin(&f("burst"), c.burst)?;
            fin(&f("up_kick"), c.up_kick)?;
            pos(&f("grain"), c.grain)?;
            unit_range(&f("restitution"), c.restitution)?;
            pos(&f("friction"), c.friction)?;
            pos(&f("settle_s"), c.settle_s)?;
            nonneg(&f("pile_h"), c.pile_h)?;
            if let Some(r) = c.pile_r {
                pos(&f("pile_r"), r)?;
            }
            nonneg(&f("dust"), c.dust)?;
            unit_range(&f("dust_desat"), c.dust_desat)?;
            if let Some(d) = c.front_dir {
                nonzero3(&f("front_dir"), d)?;
            }
            unit_range(&f("jitter"), c.jitter)?;
            pos(&f("size_min"), c.size_min)?;
            pos(&f("size_max"), c.size_max)?;
            fin3(&f("wind"), c.wind)?;
            nonneg(&f("turbulence"), c.turbulence)?;
            nonneg(&f("turb_freq"), c.turb_freq)?;
            unit_range(&f("haze"), c.haze)?;
            pos(&f("haze_size"), c.haze_size)?;
            unit_range(&f("haze_alpha"), c.haze_alpha)?;
            pos(&f("haze_s"), c.haze_s)?;
            fin(&f("haze_rise"), c.haze_rise)?;
            unit_range(&f("shadow"), c.shadow)?;
            unit_range(&f("gather"), c.gather)?;
            unit_range(&f("heap_ao"), c.heap_ao)?;
            unit_range(&f("grain_opacity"), c.grain_opacity)?;
            nonneg(&f("front_noise"), c.front_noise)?;
            nonneg(&f("front_freq"), c.front_freq)?;
            unit_range(&f("chunk_frac"), c.chunk_frac)?;
            if let Some(pl) = c.palette {
                fin3(&f("palette"), pl)?;
            }
            nonneg(&f("palette_tol"), c.palette_tol)?;
            unit_range(&f("grain_jitter"), c.grain_jitter)?;
            unit_range(&f("land_shrink"), c.land_shrink)?;
            opt_bbox(&f("bbox"), c.bbox)?;
        }
        Op::Shatter(s) => {
            if !(1..=64).contains(&s.seeds) {
                return Err(err(f("seeds"), format!("must be in [1, 64], got {}", s.seeds)));
            }
            fin(&f("start_s"), s.start_s)?;
            fin(&f("burst"), s.burst)?;
            fin(&f("up_kick"), s.up_kick)?;
            fin(&f("spin"), s.spin)?;
            unit_range(&f("restitution"), s.restitution)?;
            pos(&f("friction"), s.friction)?;
            opt_bbox(&f("bbox"), s.bbox)?;
        }
        Op::Ripple(r) => {
            fin3(&f("source"), r.source)?;
            fin(&f("start_s"), r.start_s)?;
            fin(&f("speed"), r.speed)?;
            pos(&f("wavelength"), r.wavelength)?;
            fin(&f("amplitude"), r.amplitude)?;
            nonneg(&f("decay"), r.decay)?;
            pos(&f("packet_width"), r.packet_width)?;
            nonzero3(&f("light_dir"), r.light_dir)?;
            fin(&f("shade"), r.shade)?;
            opt_bbox(&f("bbox"), r.bbox)?;
        }
        Op::Float(x) => {
            fin(&f("start_s"), x.start_s)?;
            fin(&f("rise"), x.rise)?;
            pos(&f("rise_s"), x.rise_s)?;
            fin(&f("spin"), x.spin)?;
            fin(&f("bob"), x.bob)?;
            nonneg(&f("bob_hz"), x.bob_hz)?;
            if let Some(d) = x.down_s {
                if !(d.is_finite() && d >= x.start_s) {
                    return Err(err(f("down_s"), format!("must be >= start_s, got {d}")));
                }
            }
            fin(&f("tilt"), x.tilt)?;
            if let Some(d) = x.tilt_dir {
                fin3(&f("tilt_dir"), d)?;
            }
            fin(&f("turn"), x.turn)?;
            unit_range(&f("land_hard"), x.land_hard)?;
            unit_range(&f("shadow"), x.shadow)?;
            nonneg(&f("shadow_soft"), x.shadow_soft)?;
            if let Some(v) = x.pivot {
                fin3(&f("pivot"), v)?;
            }
            opt_bbox(&f("bbox"), x.bbox)?;
        }
        Op::Melt(m) => {
            fin(&f("start_s"), m.start_s)?;
            pos(&f("duration_s"), m.duration_s)?;
            fin(&f("spread"), m.spread)?;
            fin(&f("sheen"), m.sheen)?;
            opt_bbox(&f("bbox"), m.bbox)?;
        }
    }
    Ok(())
}

fn unit_range(path: &str, v: f32) -> Result<(), GraphError> {
    if (0.0..=1.0).contains(&v) { Ok(()) } else { Err(err(path, format!("must be in [0, 1], got {v}"))) }
}

fn opt_bbox(path: &str, b: Option<Bbox>) -> Result<(), GraphError> {
    b.map_or(Ok(()), |b| validate_box(path, b.min, b.max, 0.0))
}

