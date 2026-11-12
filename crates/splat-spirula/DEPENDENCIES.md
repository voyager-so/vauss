# Dependencies and external programs

The Rust backend is Apache-2.0 and links no GPL code. It invokes a separately located,
unmodified Spirula release binary (GPL-3.0) as a subprocess. Spirula is never vendored.
FFmpeg/ffprobe are also configured external executables; temporary trims use FFV1,
available in LGPL FFmpeg builds, without requiring libx264.

Direct crates added/used here:

| Crate | License | Use |
|---|---|---|
| anyhow | MIT OR Apache-2.0 | Error context at filesystem/process boundaries |
| serde, serde_json | MIT OR Apache-2.0 | Artifacts, requests and results |
| image | MIT OR Apache-2.0 | Decode/rotate/resize JPEG, PNG, WebP, TIFF and BMP; previews |
| regex | MIT OR Apache-2.0 | Spirula stdout records |
| gltf (via splat-mesh, not a direct dependency) | MIT OR Apache-2.0 | GLB geometry for CPU mesh previews; import/native decoding disabled |
| sha2 | MIT OR Apache-2.0 | Immutable split and model checksums |
| tempfile (tests) | MIT OR Apache-2.0 | Isolated test fixtures |
| splat-core, splat-contracts, splat-eval | Apache-2.0 | Workspace interfaces and metrics |

SAM 2.1 Hiera Small GGML and Grounding DINO Tiny weights are Apache-2.0. `models.json`
records exact URLs, SHA-256 hashes, and original checkpoint provenance. SAM's original
PyTorch checkpoint cannot be loaded by the Spirula release binary; use the GGML
conversion. Grounding DINO also needs its BERT `vocab.txt` beside the detector weights.
`SpirulaConfig` accepts every program/model path for doctor/download integration; defaults
respect environment overrides and the existing development-host installation.
