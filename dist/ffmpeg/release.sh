#!/usr/bin/env bash
# Assembles the ffmpeg-public-v<n> prerelease from the build jobs' artifacts:
#   dist/ffmpeg/release.sh ARTIFACTS OUT TAG COMMIT
# OUT/files gets the three binary archives, every source tarball (and FFmpeg's signature), the build
# kit (dist/ffmpeg/ at COMMIT), each target's configure output, the licence notices and SHA256SUMS;
# OUT/NOTES.md the release text.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
art="${1:?}" out="${3:+$2}" tag="${3:?usage: release.sh ARTIFACTS OUT TAG COMMIT}" commit="${4:?}"
files="$out/files"
want="$(sed -n 's/^release_tag=//p' "$here/build.sh")"
[[ "$tag" == "$want" ]] || { echo "release.sh: tag $tag, but build.sh builds for $want" >&2; exit 1; }
rm -rf "${out:?}" && mkdir -p "$files" "$out/licenses"
targets=(x86_64-unknown-linux-gnu aarch64-apple-darwin x86_64-pc-windows-msvc)
for t in "${targets[@]}"; do
  a="$art/ffmpeg-$t"
  cp "$a"/ffmpeg-*-"$t".* "$files/"
  cp "$a/build-$t/configure-line.txt" "$files/configure-$t.txt"
  cp "$a/build-$t/config.log" "$files/config-$t.log"
  cp "$a/build-$t/toolchain.txt" "$files/toolchain-$t.txt"
  mkdir -p "$out/licenses/$t"
  case "$(ls "$files"/ffmpeg-*-"$t".*)" in
    *.zip) python3 -m zipfile -e "$files"/ffmpeg-*-"$t".zip "$out/licenses/$t" ;;
    *) tar -xzf "$files"/ffmpeg-*-"$t".tar.gz -C "$out/licenses/$t" ;;
  esac
done
# Source tarballs: exactly the pinned files, checked again.
grep -v '^#' "$here/sources.txt" | while read -r name version file sha rest; do
  [[ -n "$name" ]] || continue
  f="$(ls "$art"/ffmpeg-*/sources/"$file" | head -1)"
  [[ "$(sha256sum "$f" | cut -c1-64)" == "$sha" ]] || { echo "SHA-256 mismatch: $f" >&2; exit 1; }
  cp "$f" "$files/"
done
cp "$(ls "$art"/ffmpeg-*/sources/ffmpeg-*.tar.xz.asc | head -1)" "$(ls "$art"/ffmpeg-*/sources/ffmpeg-devel.asc | head -1)" "$files/"
python3 "$here/pack.py" "$here" "$files/ffmpeg-build-kit-$tag.tar.gz" >/dev/null
"$here/notices.sh" "$out"/licenses/*/LICENSES >"$files/FFMPEG-THIRD-PARTY.txt"
cp "$out/licenses/x86_64-unknown-linux-gnu/LICENSES/ffmpeg/COPYING.LGPLv2.1" "$files/"
(cd "$files" && sha256sum * >SHA256SUMS)
version="$(awk '$1 == "ffmpeg" {print $2}' "$here/sources.txt")"
{
  echo "splat's own static LGPL build of FFmpeg $version (ffmpeg + ffprobe), which \`splat-fetch\` pins as \`ffmpeg-public\`. Built by [.github/workflows/ffmpeg.yml](https://github.com/voyager-so/vauss/blob/$commit/.github/workflows/ffmpeg.yml) from [dist/ffmpeg/](https://github.com/voyager-so/vauss/tree/$commit/dist/ffmpeg) at \`$commit\`."
  echo
  echo "- **Licence:** LGPL-2.1-or-later. \`--disable-gpl --disable-nonfree --disable-version3\`; no libx264/libx265. Statically linked: zlib, FreeType, HarfBuzz, libopus, libvpx, dav1d (+ Vulkan headers on Linux, the mingw-w64 runtime and libgcc on Windows). Portions of this software are copyright © The FreeType Project (www.freetype.org). All rights reserved. Every licence text is in \`FFMPEG-THIRD-PARTY.txt\` and in each archive's \`LICENSES/\`."
  echo "- **Complete corresponding source:** every source tarball the binaries were built from (pinned by SHA-256 in \`sources.txt\`), FFmpeg's release signature and signing key, the build kit (\`build.sh\`, \`sources.txt\`, per-target configure flags), and each target's configure line, \`config.log\` and toolchain versions. Rebuild: unpack the build kit, then \`./build.sh TARGET OUT\` (put the tarballs, the \`.asc\` signature and \`ffmpeg-devel.asc\` in \`OUT/sources/\` and nothing is downloaded). Not archived: the compilers and build tools (apt/Homebrew packages, versions in \`toolchain-*.txt\`) and the mingw-w64/libgcc runtime sources (permissive / GCC runtime exception; notices included)."
  echo "- **H.264 encode:** no libx264 (GPL). macOS has VideoToolbox, Windows Media Foundation, Linux Vulkan Video (GPU). Without one, splat falls back to VP9 (libvpx) for people and MPEG-4 part 2 for model uploads."
  echo "- **Runs on:** Linux x86-64 with glibc >= 2.35, macOS 11+ on Apple silicon, Windows 10+ x64."
  echo
  echo "| File | Bytes | SHA-256 |"
  echo "|---|---:|---|"
  for f in "$files"/ffmpeg-"$version"-*; do
    echo "| \`$(basename "$f")\` | $(stat -c%s "$f") | \`$(sha256sum "$f" | cut -c1-64)\` |"
  done
} >"$out/NOTES.md"
ls -l "$files"
