#!/usr/bin/env bash
# Stage pinned runtime helpers at TOOLS_DIR/<asset-id>/ so an unpacked archive needs no network or token.
# Every file comes from splat-fetch's SHA-256-verified install of the assets.json pin.
set -euo pipefail
cd "$(dirname "$0")/.."
target=${1:?usage: bundle-tools.sh TARGET TOOLS_DIR}
tools=${2:?usage: bundle-tools.sh TARGET TOOLS_DIR}
# Linux has no upstream COLMAP binary; bundle our audited CPU build once it is pinned. macOS and
# Windows use the official upstream COLMAP release, fetched on first use rather than redistributed.
ids=(ffmpeg-public onnxruntime lpips-alex lpips-vgg gltfpack)
if [[ $target == x86_64-unknown-linux-gnu ]] &&
  jq -e --arg t "$target" 'any(.[]; .id == "colmap" and .target == $t)' crates/splat-fetch/assets.json >/dev/null; then
  ids+=(colmap)
fi
# Public release assets work anonymously. An explicitly supplied token remains optional.
if [[ -n ${SPLAT_GITHUB_TOKEN:-${GITHUB_TOKEN:-}} ]]; then
  export SPLAT_GITHUB_TOKEN=${SPLAT_GITHUB_TOKEN:-$GITHUB_TOKEN}
fi
export SPLAT_CACHE_DIR=${SPLAT_BUNDLE_CACHE:-target/bundle-cache}
cargo build --locked --release -p splat-fetch --example fetch
mkdir -p "$tools"
for id in "${ids[@]}"; do
  target/release/examples/fetch "$id" --target "$target" --stage "$tools"
done
# FFmpeg is redistributed here. Its archive already carries SOURCE.txt (this target's components) and
# LICENSES/; add the LGPL text, the release pointer and every component's notices at the top too.
cp dist/licenses/FFMPEG-COPYING.LGPLv2.1 "$tools/ffmpeg-public/COPYING.LGPLv2.1"
cp dist/licenses/FFMPEG-SOURCE.txt "$tools/ffmpeg-public/SOURCE-RELEASE.txt"
cp dist/licenses/FFMPEG-THIRD-PARTY.txt "$tools/ffmpeg-public/THIRD-PARTY.txt"
cp dist/licenses/GLTFPACK-NOTICES.txt "$tools/gltfpack/GLTFPACK-NOTICES.txt"
