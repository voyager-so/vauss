# Linux COLMAP helper

The CPU recipe is `scripts/colmap/build.sh`, run by `.github/workflows/colmap.yml`
on PRs and version tags. CI uploads an archive, checksum, audit report, and fetch
pin candidate. It never publishes a release. Native execution and audit must pass
before a candidate is promoted into `crates/splat-fetch/assets.json`.

The build uses Ubuntu 22.04 (glibc 2.35), four compilation jobs, and exact SHA-256
source pins for COLMAP 4.2.1, Ceres 2.2.0, OpenImageIO 2.5.19.0, TIFF 4.7.0,
PoseLib and Faiss. Distribution package versions are recorded in provenance;
binary reproducibility across changing Ubuntu package snapshots is not claimed.
The build host uses a SHA-256-pinned CMake 3.31.6 because Faiss needs CMake 3.24+;
that tool is not included in the runtime archive.
No Python is required to run the resulting helper. Python is a build/audit tool.

## CPU configuration and behavior

- `colmap` is the default CPU artifact, with a flat `bin/colmap`, `lib/`, `share/`
  layout. The binary and shared-library RPATHs are relative to their locations.
- SIFT extraction and matching use the CPU. CUDA, HIP, GUI, OpenGL, CGAL, LSD,
  dense MVS, ONNX inference, and COLMAP's own network downloads are disabled.
  Learned ALIKED/LightGlue and cuDSS are not provided by this helper.
- Ceres uses Eigen sparse solvers with SuiteSparse disabled. COLMAP 4.2.1 itself
  also directly requires CHOLMOD's GPL supernodal implementation; turning off
  SuiteSparse in Ceres alone is insufficient. The recorded `no-cholmod.patch`
  replaces that solver with Eigen's simplicial LLT, retains the existing LDLT
  fallback, and removes the CHOLMOD dependency. This can change solve time and
  numerical behavior; the upstream sparse-solver tests run against the patch.
- OpenImageIO is built with a narrow codec/dependency set. TIFF is built without
  JBIG, since Ubuntu's TIFF dependency can otherwise introduce GPL libjbig.
- The bundle audit rejects unknown runtime libraries, GPL components without a
  runtime exception, unresolved libraries, and references to build directories.
  Shared compiler runtimes retain their GCC runtime-exception notices. If
  OpenBLAS brings LGPL libquadmath, its matching source package and full notice
  are included. The ELF loader and glibc remain host dependencies.

Sources for the audit: [COLMAP build options](https://github.com/colmap/colmap/blob/4.2.1/CMakeLists.txt),
[COLMAP sparse solver](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/optim/sparse_cholesky.h),
and [CHOLMOD module licenses](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.12.0/CHOLMOD/Doc/License.txt).
The audit describes the selected build and does not relicense upstream code.

## CUDA gate

The reserved ID is `colmap-cuda`. Linux CUDA packaging is blocked: COLMAP 4.2.1's
bundled [SiftGPU license](https://github.com/colmap/colmap/blob/4.2.1/src/thirdparty/SiftGPU/LICENSE)
grants educational, research and non-profit use. There is no verified general
commercial redistribution grant in this recipe. `build.sh ... cuda` fails with
that reason rather than producing a misleadingly cleared artifact. A reviewed
license or replacement implementation is needed before enabling the CUDA CI job.
CUDA runtime redistribution terms and driver requirements must also be recorded.

## Building it yourself

The recipe runs in a stock Ubuntu 22.04 container. Give Docker explicit CPU and memory limits:

```sh
docker run --rm --init --cpus 4 --memory 16g --memory-swap 16g \
  -v "$PWD/scripts/colmap:/recipe:ro" -v "$PWD/colmap-build:/build" -v "$PWD/colmap-out:/out" \
  ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401 \
  bash -c '/recipe/deps.sh && /recipe/build.sh /build /out cpu'
```

After relocation, test `colmap -h`, CPU extraction and matching on real images, and a small reconstruction
before promoting a new pin. The archive's SHA-256 and byte count must match its sidecar and the hosted release
asset.

## Verified release (2026-10-08)

`colmap-4.2.1-splat.1-linux-x64-cpu.tar.gz`, 123,112,017 bytes, SHA-256
`6cb71b40a523e17be5f31d7092e5ea1a6eafd68f7d74bc384ca286f339e5fa44`, built by CI, hosted on the
[`colmap-linux-v1`](https://github.com/voyager-so/vauss/releases/tag/colmap-linux-v1) release (the hosted
digest matches). It is pinned in `assets.json` (`colmap`,
`x86_64-unknown-linux-gnu`, entrypoint `bin/colmap`) and bundled in Linux release
archives at `tools/colmap/`; a source build downloads it on first use.

- Relocation: extracted outside the build tree, `env -u LD_LIBRARY_PATH bin/colmap -h`
  reports 4.2.1 "without GPU support"; `ldd` resolves everything except glibc
  (libc, libm, libpthread, ld-linux) from `lib/`.
- Real run through the product path (`vauss init` → `extract_frames` →
  `reconstruct_sfm`, quality low, Brush's default flags) on 20 Truck frames at
  1024 px: 20/20 registered, 6,252 points, 0.40 px mean reprojection error, all
  `sfm-gates-v2` gates pass; extract 7 s, match 88 s, map 30 s. The CPU-only build
  accepts COLMAP's default `use_gpu` settings.
- CPU portability: all AVX-512 code is inside Faiss's runtime-dispatched SIMD
  kernels. A full extract/match/map run under `qemu-x86_64 -cpu EPYC-Milan` (Zen 3,
  AVX2 and no AVX-512) registers 6/6 images, while a stock AVX-512 conda build dies
  with Illegal instruction under the same emulation.
- `splat-fetch` download, hash check, extraction and offline cache re-validation all pass.

Proof files (`proof-*.json/txt/log`), the audit and the pin candidate are attached
to the release.
