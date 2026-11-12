# Core dependency licenses

All linked dependencies are permissive. Workspace versions live in the root manifest.

| Dependency | License | Purpose |
| --- | --- | --- |
| serde, serde_json, anyhow, thiserror | MIT OR Apache-2.0 | Typed storage and errors |
| fs2 | MIT OR Apache-2.0 | Cross-process file locks and disk space |
| parking_lot | MIT OR Apache-2.0 | Reentrant in-process run lock |
| sha2 | MIT OR Apache-2.0 | Source fingerprints and request identities |
| uuid | MIT OR Apache-2.0 | Run, artifact, and job identifiers |
| chrono | MIT OR Apache-2.0 | UTC journal timestamps |
| libc | MIT OR Apache-2.0 | Unix process groups and identities |
| windows-sys | MIT OR Apache-2.0 | Windows process identities and Job Objects |
| tempfile (tests) | MIT OR Apache-2.0 | Isolated runtime fixtures |

No reconstruction engine is linked by this crate. External engines are subprocesses.
