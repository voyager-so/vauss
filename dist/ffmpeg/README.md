# splat's FFmpeg build

`splat-fetch` downloads a static LGPL FFmpeg (`ffmpeg` + `ffprobe`) as the `ffmpeg-public` asset
(`crates/splat-fetch/assets.json`), and release archives bundle it (`scripts/bundle-tools.sh`). This
folder builds those binaries, so every one we ship has complete corresponding source.

| File | What |
|---|---|
| `sources.txt` | Every source tarball, pinned by SHA-256, with its licence and licence files. |
| `config/common.flags`, `config/<target>.flags` | FFmpeg's configure flags, per target. |
| `build.sh TARGET OUT` | Fetches and checks the tarballs, builds the libraries static, configures and builds FFmpeg, refuses GPL/nonfree/version-3, packs `OUT/ffmpeg-<version>-<TARGET>.tar.gz` (`.zip` on Windows) and writes the configure output to `OUT/build-<TARGET>/`. |
| `smoke.sh BINDIR` | Licence check, bit-exact decode of 7 public clips (H.264, HEVC 10-bit, Dolby Vision HEVC, VP9, AV1, a GoPro mp4, a rotated phone mov), the encoders splat calls, and drawtext. |
| `release.sh`, `notices.sh`, `pack.py` | Assemble the release, the combined notices, and reproducible archives. |

`.github/workflows/ffmpeg.yml` runs it: manually (Actions > ffmpeg > Run workflow), and on a tag
`ffmpeg-public-v<n>` (the `release_tag` in build.sh), which publishes a prerelease (never Latest) with the binaries, every source tarball, FFmpeg's signature, this folder as a build kit,
each target's configure line, `config.log` and toolchain, and the licence notices.

## Targets

| splat target | Built on | Notes |
|---|---|---|
| `x86_64-unknown-linux-gnu` | Ubuntu 22.04 | glibc stays dynamic (runs on glibc >= 2.35); everything else static. |
| `aarch64-apple-darwin` | macOS 14, arm64 | macOS 11+. VideoToolbox is the OS framework. |
| `x86_64-pc-windows-msvc` | Ubuntu 24.04, cross with mingw-w64 | Fully static `.exe`s, no DLLs (the standard way static FFmpeg for Windows is built). Checked on a Windows runner. |

## What is in it, and why

LGPL-2.1-or-later only: `--disable-gpl --disable-nonfree --disable-version3 --disable-autodetect`.
External libraries, all permissive: zlib, FreeType (FTL), HarfBuzz, libopus, libvpx, dav1d; plus
the Khronos Vulkan headers on Linux (FFmpeg loads `libvulkan` at run time) and the mingw-w64 runtime
and libgcc on Windows. Portions of this software are copyright © The FreeType Project
(www.freetype.org). All rights reserved.

What splat needs from FFmpeg (`crates/splat-eval`, `crates/splat-spirula`, `crates/splat-shots`):

- **Decode** phone, GoPro and camera video: H.264, HEVC 8/10-bit (incl. Dolby Vision), VP9, ProRes,
  MJPEG (FFmpeg's own decoders) and AV1 (dav1d), plus PNG, JPEG, WebP, TIFF, EXR stills.
- **Encode** (`crates/splat-shots/src/encode.rs`): H.264 when an LGPL-compatible encoder works, else
  VP9 (libvpx) + Opus (libopus) in WebM for people and MPEG-4 part 2 for model uploads; AAC, JPEG
  and PNG from FFmpeg's own encoders.
- **drawtext** for `splat ad`'s title cards (`crates/splat-shots/src/ad.rs`, a font file, no
  fontconfig): FreeType + HarfBuzz. The previous build had no drawtext, so `splat ad` failed there.
- **H.264 encode:** libx264 is GPL, so it is never built. Each OS's own encoder is used instead:
  VideoToolbox (`h264_videotoolbox`) on macOS, Media Foundation (`h264_mf`) on Windows, Vulkan Video
  (`h264_vulkan`, the GPU's fixed-function encoder) on Linux. OpenH264 is not built: only Cisco's
  own binaries carry its patent licence, and they would need a separate run-time download.

Compared with the previous third-party build (AtlasYang ffmpeg-static-builds 8.0.3 build 4), libaom,
libvorbis and libwebp are gone: splat never encodes AV1, Vorbis or WebP with FFmpeg, and FFmpeg's
own decoders read Vorbis and WebP (dav1d decodes AV1).

## Rebuild

```sh
dist/ffmpeg/build.sh x86_64-unknown-linux-gnu out   # needs build tools: see the header of build.sh
dist/ffmpeg/smoke.sh out/../unpacked                # after unpacking the archive there
```

Sources come from `OUT/sources/` when they are there (SHA-256 checked), else from the URLs in
`sources.txt`. To rebuild without downloading anything, put the release's six tarballs,
`ffmpeg-8.0.3.tar.xz.asc` and `ffmpeg-devel.asc` there first. The compilers and build tools (apt or
Homebrew packages, versions recorded in `toolchain-*.txt`) are installed separately and not archived;
nor are the mingw-w64 runtime and libgcc sources (permissive and GCC runtime exception; their
notices ship in `LICENSES/mingw-w64/`).

To bump a component: change its line in `sources.txt` (version, file, SHA-256, URL), run the
workflow, bump `release_tag` in build.sh, push that tag to publish, and re-pin `assets.json`.
