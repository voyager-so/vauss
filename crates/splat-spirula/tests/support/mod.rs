#![allow(dead_code)]
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use splat_contracts::*;
use splat_core::{Backend, RunCtx};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};

pub struct LocalRun {
    pub root: PathBuf,
    pub source: PathBuf,
    pub artifacts: Mutex<BTreeMap<String, Artifact>>,
    pub jobs: Mutex<BTreeMap<String, JobSpec>>,
}
impl LocalRun {
    pub fn new(root: &Path, source: &Path) -> Self {
        fs::create_dir_all(root).unwrap();
        Self {
            root: root.canonicalize().unwrap(),
            source: source.to_path_buf(),
            artifacts: Mutex::new(BTreeMap::new()),
            jobs: Mutex::new(BTreeMap::new()),
        }
    }
    pub fn finish(&self, backend: &dyn Backend, jid: &str) -> Result<ToolResult> {
        let spec = self.jobs.lock().unwrap()[jid].clone();
        let job = self.root.join("jobs").join(jid);
        let stdout = fs::File::create(job.join("stdout.log"))?;
        let stderr = fs::File::create(job.join("stderr.log"))?;
        let start = Instant::now();
        let mut child = Command::new(&spec.argv[0])
            .args(&spec.argv[1..])
            .envs(&spec.env)
            .current_dir(spec.cwd.as_ref().unwrap())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()?;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            backend.job_progress(self, &job);
            std::thread::sleep(Duration::from_millis(250));
        };
        let code = status.code().unwrap_or(130);
        let mut r = backend.on_job_complete(self, &spec.on_complete, &job, code)?;
        r.wall_s = Some(start.elapsed().as_secs_f64());
        fs::write(job.join("result.json"), serde_json::to_vec_pretty(&r)?)?;
        Ok(r)
    }
    pub fn add(&self, kind: ArtifactKind, meta: Value) -> (Artifact, PathBuf) {
        let (id, path) = self.new_artifact_dir(kind).unwrap();
        let a = Artifact {
            id,
            kind,
            path: path
                .strip_prefix(&self.root)
                .unwrap()
                .to_str()
                .unwrap()
                .into(),
            meta: meta.as_object().unwrap().clone().into_iter().collect(),
        };
        self.register(a.clone()).unwrap();
        (a, path)
    }
}
impl RunCtx for LocalRun {
    fn root(&self) -> &Path {
        &self.root
    }
    fn run_id(&self) -> &str {
        "spirula-local-test"
    }
    fn source(&self) -> &Path {
        &self.source
    }
    fn goal(&self) -> &str {
        "Verify Spirula backend via its trait"
    }
    fn new_artifact_dir(&self, kind: ArtifactKind) -> Result<(String, PathBuf)> {
        let kind = serde_json::to_value(kind)?;
        let base = self.root.join("artifacts");
        fs::create_dir_all(&base)?;
        for n in 1..10000 {
            let id = format!("{}_{n:03}", kind.as_str().unwrap());
            let path = base.join(&id);
            match fs::create_dir(&path) {
                Ok(()) => return Ok((id, path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e.into()),
            }
        }
        anyhow::bail!("No available artifact ID")
    }
    fn register(&self, a: Artifact) -> Result<()> {
        let mut all = self.artifacts.lock().unwrap();
        if let Some(prior) = all.get(&a.id) {
            ensure!(
                serde_json::to_value(prior)? == serde_json::to_value(&a)?,
                "Immutable artifact changed"
            );
        }
        all.insert(a.id.clone(), a);
        fs::write(
            self.root.join("registry.json"),
            serde_json::to_vec_pretty(&*all)?,
        )?;
        Ok(())
    }
    fn artifact(&self, id: &str) -> Result<Artifact> {
        self.artifacts
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .context("Unknown artifact")
    }
    fn path(&self, id: &str) -> Result<PathBuf> {
        Ok(self.root.join(self.artifact(id)?.path))
    }
    fn journal(&self, _event: &str, _data: Value) -> Result<()> {
        Ok(())
    }
    fn start_job(&self, spec: JobSpec) -> Result<String> {
        let mut jobs = self.jobs.lock().unwrap();
        let id = format!("job_{:03}", jobs.len() + 1);
        let path = self.root.join("jobs").join(&id);
        fs::create_dir_all(&path)?;
        fs::write(path.join("request.json"), serde_json::to_vec_pretty(&spec)?)?;
        jobs.insert(id.clone(), spec);
        Ok(id)
    }
}
pub fn model(run: &LocalRun, backend: &dyn Backend) -> (Artifact, PathBuf, Artifact) {
    let (frames, frame_dir) = run.add(ArtifactKind::Frames, json!({}));
    fs::create_dir_all(frame_dir.join("images")).unwrap();
    let (sfm, path) = run.add(ArtifactKind::Sfm, json!({"frames_id":frames.id,"sparse_dir":"sparse/0"}));
    fs::create_dir_all(path.join("images")).unwrap();
    fs::create_dir_all(path.join("sparse/0")).unwrap();
    for i in 0..10 {
        let image = image::RgbImage::from_fn(64, 64, |x, y| {
            image::Rgb([
                ((x * 3 + i * 23) % 256) as u8,
                ((y * 5 + i * 37) % 256) as u8,
                ((x + y * 2 + i * 47) % 256) as u8,
            ])
        });
        image.save(path.join(format!("images/{i:03}.png"))).unwrap();
        fs::hard_link(path.join(format!("images/{i:03}.png")), frame_dir.join(format!("images/{i:03}.png"))).unwrap();
    }
    fs::write(
        path.join("sparse/0/cameras.txt"),
        "1 PINHOLE 64 64 50 50 32 32\n",
    )
    .unwrap();
    fs::write(
        path.join("sparse/0/images.txt"),
        (0..10)
            .map(|i| format!("{} 1 0 0 0 0 0 0 1 {i:03}.png\n1 1 -1\n", i + 1))
            .collect::<String>(),
    )
    .unwrap();
    fs::write(path.join("sparse/0/points3D.txt"), "").unwrap();
    let split = backend
        .define_split(
            run,
            DefineSplit {
                sfm_id: sfm.id.clone(),
                interval: 2,
            },
        )
        .unwrap();
    assert_eq!(split.status, Status::Succeeded, "{split:?}");
    (sfm, path, split.artifacts[0].clone())
}
pub fn import_split_inputs(source: &dyn RunCtx, run: &LocalRun, sfm_id: &str, split_id: &str) -> Result<()> {
    let frozen = splat_core::split::load(source, split_id)?;
    let ids = std::collections::BTreeSet::from([
        sfm_id.to_owned(), split_id.to_owned(), frozen.sfm_id,
        frozen.frames_id.context("Frozen split has no source frames")?,
    ]);
    for id in ids {
        let mut artifact = source.artifact(&id)?;
        artifact.path = source.path(&id)?.to_string_lossy().into_owned();
        run.register(artifact)?;
    }
    Ok(())
}
pub fn request<T: serde::de::DeserializeOwned>(v: Value) -> T {
    serde_json::from_value(v).unwrap()
}
