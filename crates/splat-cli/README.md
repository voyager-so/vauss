# splat

The CLI and MCP server share `splat_core::dispatch`. Run folders retain the source
fingerprint, journal, state snapshot, artifacts, jobs, reports, and resolved user
configuration. Subprocess workers survive the calling CLI process.

```sh
splat tools --json
vauss doctor --json
vauss init capture.mp4 --goal 'Reconstruct the statue' --backend spirula --run-dir runs/statue
vauss tool inspect_input --run-dir runs/statue --input '{"sample_count":12}'
vauss tool extract_frames --run-dir runs/statue --input @frames-request.json
vauss run capture.mp4 --goal 'Reconstruct the statue' --driver fixed --backend spirula --run-dir runs/statue-full
vauss status runs/statue-full --json
splat logs --run-dir runs/statue-full --follow
vauss stop runs/statue-full
vauss resume --run-dir runs/statue-full
vauss fetch ffmpeg
splat shot TRAINED_OUTPUT --out runs/orbit --prompt 'slow orbit, 5s'
```

`--input -` reads JSON from stdin. `--json` emits a single JSON line on stdout;
progress and diagnostics use stderr. Tool failures exit 5 with a structured result.
`--driver fixed` never calls an LLM. `--time-budget 10m` requests a graceful stop at
the deadline. `stop` sends one interrupt and waits for checkpointing; it does not
escalate to killing the backend.

`run` defaults to the Brush backend and Claude driver. Use `--driver fixed` for
the deterministic pipeline and `--no-mesh` when a splat-only result is wanted.
The final stderr summary names the selected splat, mesh, and reports; `report.md`
links those files directly. Missing deliverables are marked explicitly.

## What a Brush run produces

- **Splat** (`splat.ply`) with held-out PSNR/SSIM/LPIPS. Training scores the held-out
  views itself; `evaluate` reuses those renders, so it takes about a second.
- **Two meshes.** The main deliverable is the subject the cameras look at (an orbit's
  centre, e.g. the statue and its plinth), with vertex colours. The whole scene is a
  second mesh artifact, listed after it.
- **SfM** always runs splat-sfm (COLMAP underneath). It checks its own health and
  reports problems in plain words:
  - *"Only N of M frames could be placed"*: the camera moved too fast or the frames
    overlap too little. Film more slowly with more overlap, or extract more frames.
  - *"The frames formed K separate groups (…); the largest has only N of M"*: the input
    mixes different scenes or has gaps in coverage. Use one continuous capture, or add
    frames that bridge the groups.
  - *"The camera barely moved, so there's no depth to build a 3D scene from…"*: replaces
    either of the above, or a failed solve, when the clip moved under 1 image width or
    90%+ of matched pairs show no viewpoint change (also *"turned in place"* and, for photos,
    *"taken from nearly the same spot"*). The measured motion goes in the error's evidence
    and the log. Re-shoot while walking around the subject or stepping sideways.
  All of these stop the run before training rather than training on a handful of cameras.
  Smaller problems (a few unplaced frames, a stray piece) are warnings.
- **Masks.** Sky/person masks (and any `mask_distractors` result) keep moving or empty
  regions out of SfM matching; masks with purpose `training_ignore`/`both` are also
  ignored by the training loss. Static background (sky, buildings, ground, walls…) is
  never dropped from training, whatever the prompt asks. Photos with transparency use their alpha as a mask.
- **Dark footage.** Very dark frames get a brightened copy used only to find SfM
  features; training and the final splat use the original frames.

The fixed pipeline accepts `--steps`, `--target-count`, `--max-edge`, `--cap-max`,
`--resolution-divisor`, `--no-mesh`, and `--no-lpips`. It saves these settings and
reuses them on resume; explicit resume flags override individual settings. A
candidate resume request is sent after interruption or a retryable training
failure. Spirula requires a full optimizer checkpoint; Brush resumes splats and
the learning-rate schedule without restoring optimizer state.

Pass `--config toolchain.json`, or set `SPLAT_CONFIG`, to configure helper paths:

```json
{
  "ffmpeg": "/path/to/ffmpeg",
  "ffprobe": "/path/to/ffprobe",
  "spirula": "/path/to/spirula",
  "colmap": "/path/to/colmap",
  "voyager_ffmpeg_tree": null,
  "cache_dir": null,
  "models_dir": null,
  "onnxruntime": null,
  "blender": null
}
```

Environment overrides: `SPLAT_FFMPEG`, `SPLAT_FFPROBE`, `SPLAT_AGENT_SPIRULA`,
`SPLAT_VOYAGER_FFMPEG_TREE`, `SPLAT_CACHE_DIR`, `SPLAT_COLMAP`, `SPLAT_MODELS_DIR`,
`SPLAT_ONNXRUNTIME` (or `ORT_DYLIB_PATH`), and `SPLAT_BLENDER`.
FFmpeg execution resolves an explicit path, then Voyager's staged tree, then the
pinned LGPL download. `doctor` inventories offline and never downloads. Optional
Spirula is a separate GPL executable, selected with `--backend spirula`.
`SPLAT_GITHUB_TOKEN` is passed explicitly to the fetcher for private pinned model
assets; it is never saved in the run configuration.

COLMAP is fetched when Brush first needs reconstruction and no local executable
is configured. SAM weights are assembled when Spirula first masks distractions.
LPIPS models and ONNX Runtime are resolved when evaluation requests LPIPS. If
LPIPS cannot be downloaded or loaded, evaluation retains PSNR/SSIM and records
`lpips_unavailable` with the reason. Configured model directories are respected;
missing files there are reported instead of overwriting user-managed models.
Status, stop, and job reconciliation never download dependencies.

Shot tools use the same registry through CLI or MCP. `splat shot` plans and renders
a move from a trained output; `scene`, `generate`, and `review` expose the related
raw-input commands. FFmpeg paths are passed explicitly to renderers and workers.

`--backend fake` is a CPU-only orchestration fixture. Its artifacts and metrics
are marked synthetic and cannot be used to assess reconstruction quality.

Dependencies added directly by this crate: clap, ctrlc, fs2, uuid, serde,
serde_json, sha2, and anyhow (all MIT OR Apache-2.0); tempfile (tests, same license).
Backend and download licenses are recorded by their respective crates.
