//! Shared contracts between crates. Change only with a note to the coordinator.
//!
//! Every tool is `(run, request) -> ToolResult`. The same request types generate the CLI
//! (`vauss tool <name>`), the MCP server schemas and the agent's tool list via schemars.
//! Paths the model sees are artifact IDs; the run allocates real paths under the run root.

use std::collections::BTreeMap;

use schemars::{JsonSchema, schema::RootSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod dev;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Source,
    Frames,
    Masks,
    Sfm,
    Split,
    Candidate,
    Eval,
    Images,
    Mesh,
    Export,
    Report,
    Shot,
    Video,
    /// Per-Gaussian object selection on a candidate (select_objects).
    Selection,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    /// e.g. "frames_01", "sfm_02", "cand_c01"
    pub id: String,
    pub kind: ArtifactKind,
    /// Relative to the run root.
    pub path: String,
    #[serde(default)]
    pub meta: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Running,
    Succeeded,
    Failed,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ToolError {
    /// e.g. "oom", "no_video_track", "insufficient_eval_views", "unsupported_capability"
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ToolResult {
    pub schema_version: u32,
    pub tool: String,
    pub status: Status,
    /// Set when status == running (long jobs: sfm, train, mesh, generate).
    pub job_id: Option<String>,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    /// Missing values are absent or null with a reason, never 0.
    #[serde(default)]
    pub metrics: BTreeMap<String, Value>,
    /// Run-relative image paths the driver should look at.
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub error: Option<ToolError>,
    pub wall_s: Option<f64>,
}

/// Terminal outcome of a run, set by the driver's finish call and checked by code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Succeeded,
    /// Usable output below quality gates.
    Partial,
    /// Missing info only the user can give (e.g. a metric reference).
    NeedsInput,
    /// Capture cannot support 3D (e.g. pure rotation); no long train spent.
    Refused,
    Failed,
}

// ---- tool requests. Bounds are generous; the agent may ask for less, never more. ----
// Bounds are enforced by `validate()` on each request (schemars carries them into the schema).

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Result metrics include orientation_suspect, static_frac, black_frac, sharpness and exposure stats.
pub struct InspectInput {
    #[serde(default = "d_12")]
    #[schemars(range(min = 1, max = 48))]
    pub sample_count: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameStrategy {
    Uniform,
    SharpnessWindow,
}

/// Where a video's frames go along the clip.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FrameSpacing {
    /// About 2 frames per second, denser in turns and sparser when still (within 4x of the base).
    #[default]
    Rate,
    /// Even steps of measured camera motion, however uneven in time: for a few keyframes that
    /// must each overlap the next (the quick first-view SfM). Photo folders ignore it.
    Motion,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExtractFrames {
    #[serde(default = "d_sharpness")]
    pub strategy: FrameStrategy,
    /// Maximum frames kept. Video keeps about 2 frames per second (denser in turns, sparser when
    /// still), at least 150 (or every frame of a shorter clip), up to this cap; photo folders keep
    /// up to this many images, evenly spaced.
    #[serde(default = "d_600")]
    #[schemars(range(min = 20, max = 1500))]
    pub target_count: u32,
    #[serde(default = "d_1600")]
    #[schemars(range(min = 480, max = 4096))]
    pub max_edge: u32,
    pub time_range_s: Option<(f64, f64)>,
    /// Pixels stored rotated without metadata: 0, 90, 180 or 270.
    #[serde(default)]
    pub rotate_deg: u32,
    #[serde(default = "d_true")]
    pub drop_duplicates: bool,
    #[serde(default = "d_true")]
    pub drop_black: bool,
    #[serde(default)]
    pub spacing: FrameSpacing,
    /// Video: plan the frames inside the clip prefix whose camera motion stays within this many
    /// image widths (at least 5 s), for the quick first-view SfM whose few frames must each overlap
    /// the next. None: the whole clip (or time range).
    #[serde(default)]
    pub max_motion_widths: Option<f64>,
    /// Photos larger than max_edge also get larger copies for SfM (frames/sfm_images/). Off for
    /// small preview extractions.
    #[serde(default = "d_true")]
    pub sfm_copies: bool,
    /// Quick first-view frames that a full extraction replaces: max_edge may go below the
    /// 960 px camera-solving floor (down to 480 px) that ordinary extractions are raised to.
    #[serde(default)]
    pub preview: bool,
    /// Video full extractions: give the camera budget of sustained low-detail (blurred) stretches
    /// to clear time, keeping a thinner sample inside them. Off by default: it cut one room clip's 0-10 s
    /// opening off the rest of the room at any thin density (local product SfM placed 49/69/51% at
    /// 2/3/4 views/s, 50% at 1; unthinned 95%, the 2 fps plan 96%).
    #[serde(default)]
    pub thin_low_detail: bool,
    /// Video full extractions: base frames per second (None: SPLAT_FRAME_FPS, else 5). The SfM
    /// placement retry asks for 2, the pre-326e03c sampling (that room clip: 2 fps 98% placed, 5 fps 43-51%).
    #[serde(default)]
    pub base_fps: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MaskPurpose {
    SfmFeatures,
    TrainingIgnore,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MaskDistractors {
    pub frames_id: String,
    #[serde(default = "d_mask_prompt")]
    pub prompt: String,
    #[serde(default = "d_both")]
    pub purpose: MaskPurpose,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum CameraModel {
    Auto,
    Pinhole,
    Opencv,
    OpencvFisheye,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Pairs {
    Auto,
    Sequential,
    Exhaustive,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReconstructSfm {
    pub frames_id: String,
    #[serde(default = "d_medium")]
    pub quality: Quality,
    #[serde(default = "d_auto_cam")]
    pub camera_model: CameraModel,
    #[serde(default = "d_auto_pairs")]
    pub pairs: Pairs,
    pub mask_id: Option<String>,
    /// Optional feature-only segmentation inside the SfM worker (when the backend supports it).
    #[serde(default)]
    pub mask_prompt: Option<String>,
    /// A quick first-view solve that a full SfM replaces: it passes with 40% of its frames in one
    /// model, where a full SfM needs half of them registered (brush backend).
    #[serde(default)]
    pub preview: bool,
    /// Cloud quick solves (brush backend): train this first-view preview in the same remote job
    /// right after the solve, on the default interval-8 split. One upload and one queue instead
    /// of two, and no result round trip in between. Best effort: the result's metrics name the
    /// preview's candidate (`preview_candidate_id`) or why there is none (`preview_chain_error`).
    #[serde(default)]
    pub preview_train: Option<PreviewTrain>,
    /// COLMAP mapper seed (None: SPLAT_SFM_MAPPER_SEED, else 0). The placement retry re-solves with
    /// another seed, since one unlucky mapper run can leave a section out.
    #[serde(default)]
    pub mapper_seed: Option<u64>,
}

/// The first-view preview a quick solve trains in its own job ([`ReconstructSfm::preview_train`]).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreviewTrain {
    #[schemars(range(min = 200, max = 60000))]
    pub total_steps: u32,
    #[serde(default = "d_1m")]
    #[schemars(range(min = 10000, max = 10000000))]
    pub cap_max: u32,
    #[serde(default = "d_1")]
    #[schemars(range(min = 1, max = 8))]
    pub resolution_divisor: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DefineSplit {
    pub sfm_id: String,
    #[serde(default = "d_8")]
    #[schemars(range(min = 2, max = 50))]
    pub interval: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    Off,
    Mild,
    Strong,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TrainSplats {
    pub sfm_id: String,
    pub split_id: String,
    /// Fast first-view candidate: honor the requested recipe even in cloud MAX runs.
    #[serde(default)]
    pub preview: bool,
    #[serde(default = "d_7000")]
    #[schemars(range(min = 200, max = 60000))]
    pub total_steps: u32,
    /// Keep an explicitly requested step count instead of the cloud preset's minimum.
    #[serde(default)]
    pub total_steps_explicit: bool,
    /// Opt into the full-quality MAX recipe on local or cloud GPUs.
    #[serde(default)]
    pub max_quality: bool,
    #[serde(default = "d_1m")]
    #[schemars(range(min = 10000, max = 10000000))]
    pub cap_max: u32,
    /// Keep an explicitly requested cap instead of the cloud preset's splat budget.
    #[serde(default)]
    pub cap_max_explicit: bool,
    #[serde(default = "d_1")]
    #[schemars(range(min = 1, max = 8))]
    pub resolution_divisor: u32,
    #[serde(default = "d_off")]
    pub floater_suppression: Strength,
    #[serde(default = "d_off")]
    pub distraction_robustness: Strength,
    /// Warm start from another candidate's splats.
    pub init_candidate_id: Option<String>,
    /// True resume from a full checkpoint.
    pub resume_candidate_id: Option<String>,
    /// Cloud video MAX: also write the model two thirds in (or at its early stop), when above this many
    /// splats, for an interactive refine forked beside this training (`fork_of_candidate_id`).
    #[serde(default)]
    pub fork_min_splats: Option<u32>,
    /// Cloud: warm start from the fork model that the (still running) training
    /// of this candidate writes for it (`fork_min_splats`); fails if that training writes none.
    #[serde(default)]
    pub fork_of_candidate_id: Option<String>,
    /// Training RNG seed (initialization, view order, GPU sampling). Default 42.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Score the held-out views every this many steps. Default: total_steps / 5, clamped to 500-5000
    /// (a preview scores only at its end).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub eval_every: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Evaluate {
    pub candidate_id: String,
    pub split_id: String,
    #[serde(default = "d_true")]
    pub lpips: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ViewSet {
    Heldout,
    TrainingSample,
    Orbit,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenderViews {
    pub candidate_id: String,
    #[serde(default = "d_heldout")]
    pub views: ViewSet,
    #[serde(default = "d_8")]
    #[schemars(range(min = 1, max = 48))]
    pub max_views: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeshFormat {
    Ply,
    Obj,
    Glb,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeshColor {
    None,
    Vertex,
    Texture,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExtractMesh {
    pub candidate_id: String,
    #[serde(default = "d_glb")]
    pub format: MeshFormat,
    #[serde(default = "d_texture")]
    pub color: MeshColor,
    #[serde(default = "d_2048")]
    #[schemars(range(min = 256, max = 8192))]
    pub texture_size: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShotMove {
    Orbit,
    PushIn,
    PullOut,
    CraneUp,
    CraneDown,
    DollyLeft,
    DollyRight,
    FigureEight,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub enum Aspect {
    #[serde(rename = "16:9")]
    Wide,
    #[serde(rename = "9:16")]
    Tall,
    #[serde(rename = "1:1")]
    Square,
}

/// Camera move from a trained candidate -> artifact kind "shot" (path.json keyframes + risk.json).
/// metrics: risk low|medium|high, novelty_max (scene radii from nearest similarly-facing capture camera).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlanShot {
    pub candidate_id: String,
    /// e.g. "slow orbit at dusk, 5s"; parsed when `move` is absent.
    #[serde(default)]
    pub prompt: String,
    #[serde(rename = "move")]
    pub movement: Option<ShotMove>,
    #[serde(default = "d_5f")]
    pub duration_s: f64,
    /// Orbit: degrees; other moves: scene radii travelled.
    pub amount: Option<f64>,
    /// Scene radii along up, -2..2.
    #[serde(default)]
    pub height_offset: f64,
    #[serde(default = "d_wide")]
    pub aspect: Aspect,
    /// Capture image name to start from.
    pub start_camera: Option<String>,
}

/// Render a shot -> artifact kind "video" (rgb/depth/alpha mp4 + frames); images = first/mid/last.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenderShot {
    pub shot_id: String,
}

/// Long job: rough render -> hosted video model(s) via fal.ai -> one "video" artifact per model.
/// Model keys are defined in splat-shots (e.g. "luma_ray32_v2v", "wan_vace_depth", "ltx23_render_to_real").
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerateShot {
    pub shot_id: String,
    #[serde(default = "d_models")]
    pub models: Vec<String>,
    /// What the scene is ("a stone pegasus statue in a park").
    #[serde(default)]
    pub scene_text: String,
    /// "at dusk, golden light"
    #[serde(default)]
    pub style: String,
}

/// -> artifact kind "export": bundle with scene.blend, mesh.glb, splat.ply, camera_path.json.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportBlender {
    pub shot_id: String,
    pub mesh_id: Option<String>,
    #[serde(default = "d_true")]
    pub build: bool,
}

/// Long job: find the Gaussians of the objects a short noun prompt names, on a trained candidate ->
/// artifact kind "selection": selected_ids.npy (u32 indices into the candidate's splat.ply),
/// selection.json (per-object stats), masks/ (per-view object masks, 255 = object) and before/after
/// strips (GT+mask | full | removed | object only) on held-out views.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SelectObjects {
    pub candidate_id: String,
    /// Detector query: short object nouns separated by semicolons ("bench", "red car"), never a sentence.
    pub prompt: String,
    /// Training views segmented and voted on, evenly spaced over the capture.
    #[serde(default = "d_24")]
    #[schemars(range(min = 4, max = 64))]
    pub max_views: u32,
    /// Fraction of a Gaussian's rendered weight that must fall inside the object masks.
    #[serde(default = "d_half")]
    #[schemars(range(min = 0.1, max = 0.95))]
    pub threshold: f64,
    /// "all": every matching object (each a separate object_id); "largest": only the object with the
    /// most votes (the prompt names one thing: "the chair").
    #[serde(default = "d_all")]
    pub instances: Instances,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Instances {
    All,
    Largest,
}

/// Delete selected Gaussians -> a new candidate (splat.ply) whose parent is `candidate_id`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoveObjects {
    pub candidate_id: String,
    pub selection_id: String,
    /// Objects of the selection to remove (selection.json objects[].object_id); default all.
    #[serde(default)]
    pub object_ids: Vec<u32>,
}

/// Long job: fill the hole a removal left (masked fine-tune of the hole from inpainted views) -> a new
/// candidate.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InpaintRemoved {
    /// The edited candidate remove_objects made.
    pub candidate_id: String,
    pub selection_id: String,
    #[serde(default = "d_1500")]
    #[schemars(range(min = 200, max = 10000))]
    pub steps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobStatus {
    pub job_id: String,
    /// Block up to this long (0..600 s) for completion.
    #[serde(default)]
    pub wait_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StopJob {
    pub job_id: String,
}

/// The driver's terminal call; code checks it against artifacts before accepting.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Finish {
    pub status: RunStatus,
    pub explanation: String,
    pub candidate_id: Option<String>,
    /// Known problems with the delivered result, one per entry.
    #[serde(default)]
    pub remaining_defects: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecordDecision {
    pub explanation: String,
    #[serde(default)]
    pub evidence: Vec<String>,
}

/// A long operation handed to the core JobRunner by a backend.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    pub tool: String,
    pub argv: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub cwd: Option<String>,
    /// Name of the completion handler registered by the backend; called as
    /// handler(run, job_dir, exit_code) -> ToolResult. Must be idempotent (may re-run after a crash).
    pub on_complete: String,
    #[serde(default)]
    pub meta: BTreeMap<String, Value>,
}

/// Optional overrides of the run's resolved CLI settings. Omitted fields keep those settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunDefaultPipeline {
    pub steps: Option<u32>,
    pub target_count: Option<u32>,
    pub max_edge: Option<u32>,
    pub cap_max: Option<u32>,
    pub resolution_divisor: Option<u32>,
    pub no_lpips: Option<bool>,
}

/// Tool name -> JSON schema of its request. Backend tools first, then shot tools, then built-ins.
pub fn tool_schemas() -> Vec<(&'static str, &'static str, RootSchema)> {
    vec![
        ("run_default_pipeline", "Call FIRST for normal reconstruction goals. Runs the same optimized pipeline as the fixed driver: quick preview, full training, web export, held-out evaluation and interactive refinement. Inherits CLI/cloud settings; pass only explicitly requested overrides. Blocks until complete, returns candidates and comparison images. Does not finish or perform mesh/object edits: add only goal-requested extras, then finish.", schema_for!(RunDefaultPipeline)),
        ("inspect_input", "Probe the source video/images: codec, rotation, sharpness, exposure, static/black frames, sample frames as images. For manual recovery after run_default_pipeline.", schema_for!(InspectInput)),
        ("extract_frames", "Select frames from the source into a frames artifact (sharpest per window by default). Returns frames_id and a contact sheet.", schema_for!(ExtractFrames)),
        ("mask_distractors", "Segment moving distractors (people, cars...) from a short noun prompt into masks that SfM features and/or the training loss ignore. Returns overlay images; check the subject is not masked.", schema_for!(MaskDistractors)),
        ("reconstruct_sfm", "Structure-from-motion on a frames artifact. Long job: returns job_id; poll job_status. Result has SfM health metrics and gate verdicts. Settle masks and any SfM retry before define_split.", schema_for!(ReconstructSfm)),
        ("define_split", "Freeze the held-out evaluation split (every Nth registered frame) on an SfM artifact. Do this once, before any training; never change it to hide failures.", schema_for!(DefineSplit)),
        ("train_splats", "Train a Gaussian-splat candidate on an SfM + split. Long job: returns job_id; poll job_status. init_candidate_id = warm start, resume_candidate_id = true resume.", schema_for!(TrainSplats)),
        ("evaluate", "Score a trained candidate on the frozen held-out views (PSNR/SSIM/LPIPS, per-view and worst views) and return gt|render comparison images; look at them. Long job: returns job_id; poll job_status (it may wait for the GPU and run CPU LPIPS).", schema_for!(Evaluate)),
        ("render_views", "Render a candidate from held-out cameras, a training sample, or an orbit, returned as images for visual review.", schema_for!(RenderViews)),
        ("extract_mesh", "Extract a mesh (glb/ply/obj) from a candidate. Long job: returns job_id; poll job_status. Result includes geometry previews; look at them.", schema_for!(ExtractMesh)),
        ("plan_shot", "Plan a camera move (orbit, push_in, crane_up...) through a trained candidate. Returns a shot_id plus risk (low/medium/high) and novelty_max: how far the path strays from captured viewpoints, where splats break down.", schema_for!(PlanShot)),
        ("render_shot", "Render a planned shot to video (rgb/depth/alpha). Returns first/mid/last frames as images; look at them.", schema_for!(RenderShot)),
        ("generate_shot", "Long job, costs money: send a rendered shot through hosted video model(s) to restyle/clean it. Use only when the goal asks for a generated/stylised video. Third-party hosted models on fal.ai (the user's FAL_KEY); outputs are governed by each provider's terms.", schema_for!(GenerateShot)),
        ("export_blender", "Bundle a shot for Blender: scene.blend, scene.usdc, mesh.glb, splat.ply, camera_path.json.", schema_for!(ExportBlender)),
        ("select_objects", "Long job: select the Gaussians of the object(s) a short noun prompt names (\"bench\") on a trained candidate. Returns a selection_id, per-object stats and held-out before/after strips (GT+mask | full | removed | object only); look at them before removing.", schema_for!(SelectObjects)),
        ("remove_objects", "Delete a selection's Gaussians from a candidate, making a new edited candidate (its own splat.ply; the parent is kept). Render it with render_views to check the result.", schema_for!(RemoveObjects)),
        ("inpaint_removed", "Long job: fill the hole remove_objects left, from inpainted views, making a new candidate. Returns unsupported while unavailable.", schema_for!(InpaintRemoved)),
        ("job_status", "Status of a long job. wait_s blocks up to that many seconds for it to finish; on completion returns the job's full result. One call every few minutes is plenty.", schema_for!(JobStatus)),
        ("stop_job", "Stop a running job gracefully (trainers save a resumable checkpoint).", schema_for!(StopJob)),
        ("record_decision", "Record a decision and its evidence in the run journal. Use at every stage choice, retry or remedy: what you saw, what you changed, what you expect.", schema_for!(RecordDecision)),
        ("finish", "End the run with an honest verdict (status, explanation, best candidate, remaining defects). Call exactly once, last. Then stop.", schema_for!(Finish)),
    ]
}

pub const BACKEND_TOOLS: &[&str] = &[
    "inspect_input", "extract_frames", "mask_distractors", "reconstruct_sfm", "define_split",
    "train_splats", "evaluate", "render_views", "extract_mesh",
];
pub const SHOT_TOOLS: &[&str] = &["plan_shot", "render_shot", "generate_shot", "export_blender"];
/// Edits of a trained candidate. select_objects and inpaint_removed are backend jobs; remove_objects is core.
pub const EDIT_TOOLS: &[&str] = &["select_objects", "remove_objects", "inpaint_removed"];

fn d_12() -> u32 { 12 }
fn d_24() -> u32 { 24 }
fn d_1500() -> u32 { 1500 }
fn d_half() -> f64 { 0.5 }
fn d_all() -> Instances { Instances::All }
fn d_600() -> u32 { 600 }
fn d_1600() -> u32 { 1600 }
fn d_8() -> u32 { 8 }
fn d_7000() -> u32 { 7000 }
fn d_1m() -> u32 { 1_000_000 }
fn d_1() -> u32 { 1 }
fn d_2048() -> u32 { 2048 }
fn d_5f() -> f64 { 5.0 }
fn d_true() -> bool { true }
fn d_sharpness() -> FrameStrategy { FrameStrategy::SharpnessWindow }
fn d_mask_prompt() -> String { "person".into() }
fn d_both() -> MaskPurpose { MaskPurpose::Both }
fn d_medium() -> Quality { Quality::Medium }
fn d_auto_cam() -> CameraModel { CameraModel::Auto }
fn d_auto_pairs() -> Pairs { Pairs::Auto }
fn d_off() -> Strength { Strength::Off }
fn d_heldout() -> ViewSet { ViewSet::Heldout }
fn d_glb() -> MeshFormat { MeshFormat::Glb }
fn d_texture() -> MeshColor { MeshColor::Texture }
fn d_wide() -> Aspect { Aspect::Wide }
fn d_models() -> Vec<String> { vec!["luma_ray32_v2v".into()] }

#[cfg(test)]
mod tests {
    #[test]
    fn schemas_build() {
        assert_eq!(super::tool_schemas().len(), 21);
    }
}
