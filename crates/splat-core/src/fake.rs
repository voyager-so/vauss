use crate::{
    Backend, RunCtx, atomic_json,
    jobs::{failure, result},
    run::read_json,
};
use anyhow::Result;
use serde::Serialize;
use serde_json::{Value, json};
use splat_contracts::*;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// CPU-only backend for exercising orchestration. Every artifact and metric is labelled synthetic.
#[derive(Default)]
pub struct FakeBackend {
    executable: Option<PathBuf>,
}
impl FakeBackend {
    pub fn with_executable(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: Some(executable.into()),
        }
    }
    fn artifact(
        &self,
        run: &dyn RunCtx,
        tool: &str,
        kind: ArtifactKind,
        req: &impl Serialize,
    ) -> Result<ToolResult> {
        let (id, path) = run.new_artifact_dir(kind)?;
        let mut meta: BTreeMap<String, Value> = serde_json::from_value(serde_json::to_value(req)?)?;
        meta.insert("synthetic".into(), json!(true));
        atomic_json(path.join("manifest.json"), &meta)?;
        let mut r = result(tool, Status::Succeeded);
        r.artifacts.push(Artifact {
            id,
            kind,
            path: path
                .strip_prefix(run.root())?
                .to_string_lossy()
                .into_owned(),
            meta,
        });
        r.warnings.push("Synthetic fixture output".into());
        Ok(r)
    }
    fn job(
        &self,
        run: &dyn RunCtx,
        tool: &str,
        kind: ArtifactKind,
        req: &impl Serialize,
    ) -> Result<ToolResult> {
        let mut r = self.artifact(run, tool, kind, req)?;
        let a = r.artifacts[0].clone();
        run.register(a.clone())?;
        let executable = self
            .executable
            .clone()
            .map(Ok)
            .unwrap_or_else(std::env::current_exe)?;
        let spec = JobSpec {
            tool: tool.into(),
            argv: vec![
                executable.to_string_lossy().into_owned(),
                "__splat_fake".into(),
                "150".into(),
            ],
            env: Default::default(),
            cwd: None,
            on_complete: "fake:complete".into(),
            meta: BTreeMap::from([("artifact".into(), json!(a))]),
        };
        r.status = Status::Running;
        r.job_id = Some(run.start_job(spec)?);
        Ok(r)
    }
}
impl Backend for FakeBackend {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn capabilities(&self) -> Value {
        json!({"backend":"fake","synthetic":true,"tools":BACKEND_TOOLS})
    }
    fn inspect_input(&self, run: &dyn RunCtx, _req: InspectInput) -> Result<ToolResult> {
        let mut r = result("inspect_input", Status::Succeeded);
        r.metrics.insert("synthetic".into(), json!(true));
        r.metrics
            .insert("source_exists".into(), json!(run.source().exists()));
        r.warnings
            .push("Fake backend does not measure reconstruction quality".into());
        Ok(r)
    }
    fn extract_frames(&self, run: &dyn RunCtx, req: ExtractFrames) -> Result<ToolResult> {
        self.artifact(run, "extract_frames", ArtifactKind::Frames, &req)
    }
    fn mask_distractors(&self, run: &dyn RunCtx, req: MaskDistractors) -> Result<ToolResult> {
        run.path(&req.frames_id)?;
        self.artifact(run, "mask_distractors", ArtifactKind::Masks, &req)
    }
    fn reconstruct_sfm(&self, run: &dyn RunCtx, req: ReconstructSfm) -> Result<ToolResult> {
        run.path(&req.frames_id)?;
        self.job(run, "reconstruct_sfm", ArtifactKind::Sfm, &req)
    }
    fn define_split(&self, run: &dyn RunCtx, req: DefineSplit) -> Result<ToolResult> {
        run.path(&req.sfm_id)?;
        self.artifact(run, "define_split", ArtifactKind::Split, &req)
    }
    fn train_splats(&self, run: &dyn RunCtx, req: TrainSplats) -> Result<ToolResult> {
        run.path(&req.sfm_id)?;
        run.path(&req.split_id)?;
        self.job(run, "train_splats", ArtifactKind::Candidate, &req)
    }
    fn evaluate(&self, run: &dyn RunCtx, req: Evaluate) -> Result<ToolResult> {
        run.path(&req.candidate_id)?;
        run.path(&req.split_id)?;
        let mut r = self.artifact(run, "evaluate", ArtifactKind::Eval, &req)?;
        r.metrics = BTreeMap::from([
            ("synthetic".into(), json!(true)),
            ("psnr".into(), json!(30.)),
            ("ssim".into(), json!(0.95)),
            ("num_eval_images".into(), json!(8)),
        ]);
        atomic_json(
            run.root().join(&r.artifacts[0].path).join("metrics.json"),
            &r.metrics,
        )?;
        Ok(r)
    }
    fn render_views(&self, run: &dyn RunCtx, req: RenderViews) -> Result<ToolResult> {
        run.path(&req.candidate_id)?;
        self.artifact(run, "render_views", ArtifactKind::Images, &req)
    }
    fn extract_mesh(&self, run: &dyn RunCtx, req: ExtractMesh) -> Result<ToolResult> {
        run.path(&req.candidate_id)?;
        self.job(run, "extract_mesh", ArtifactKind::Mesh, &req)
    }
    fn on_job_complete(
        &self,
        run: &dyn RunCtx,
        _handler: &str,
        dir: &Path,
        exit_code: i32,
    ) -> Result<ToolResult> {
        let spec: JobSpec = read_json(dir.join("request.json"))?;
        let mut r = if exit_code == 0 {
            result(&spec.tool, Status::Succeeded)
        } else {
            failure(&spec.tool, "job_exit", format!("Exit {exit_code}"))
        };
        if let Some(a) = spec.meta.get("artifact") {
            let a: Artifact = serde_json::from_value(a.clone())?;
            if exit_code == 0 {
                write_fixture(&run.root().join(&a.path), a.kind)?;
            }
            r.artifacts.push(a);
        }
        r.warnings.push("Synthetic fixture output".into());
        Ok(r)
    }
}

/// Tiny real outputs for a finished job, so a fake run's chosen candidate goes through the real web
/// export (a finished run must end with a viewable package): an SfM job writes a COLMAP text model
/// (eight 64x48 cameras on a unit ring looking at the origin, a floor one unit below them), a train
/// job a 27-gaussian splat.ply around the origin. Deterministic and CPU-only.
fn write_fixture(dir: &Path, kind: ArtifactKind) -> Result<()> {
    match kind {
        ArtifactKind::Sfm => {
            let sparse = dir.join("sparse/0");
            std::fs::create_dir_all(&sparse)?;
            std::fs::write(sparse.join("cameras.txt"), "1 PINHOLE 64 48 50 50 32 24\n")?;
            // Camera k: camera-to-world is a turn of phi about +y (world down, as in OpenCV), the
            // centre one unit back along its forward axis, so every world-to-camera t is (0, 0, 1).
            let images: String = (0..8)
                .map(|k| {
                    let half = k as f64 * std::f64::consts::PI / 8.0;
                    format!("{} {} 0 {} 0 0 0 1 1 fake_{k:03}.png\n\n", k + 1, half.cos(), -half.sin())
                })
                .collect();
            std::fs::write(sparse.join("images.txt"), images)?;
            let points: String = (0..49)
                .map(|i| {
                    let (x, z) = ((i % 7) as f64 / 3.0 - 1.0, (i / 7) as f64 / 3.0 - 1.0);
                    format!("{} {x} 1 {z} 128 128 128 0.5\n", i + 1)
                })
                .collect();
            std::fs::write(sparse.join("points3D.txt"), points)?;
        }
        ArtifactKind::Candidate => {
            let mut ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 27\n".to_vec();
            for p in ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"] {
                ply.extend(format!("property float {p}\n").bytes());
            }
            ply.extend(b"end_header\n");
            for i in 0..27 {
                let at = |d: usize| ((i / 3usize.pow(d as u32)) % 3) as f32 * 0.1 - 0.1;
                let scale = (0.03f32).ln();
                for v in [at(0), at(1), at(2), scale, scale, scale, 1., 0., 0., 0., 2., at(0) * 10., at(1) * 10., at(2) * 10.] {
                    ply.extend(v.to_le_bytes());
                }
            }
            std::fs::write(dir.join("splat.ply"), ply)?;
        }
        _ => {}
    }
    Ok(())
}
