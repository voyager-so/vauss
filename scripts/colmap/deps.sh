#!/usr/bin/env bash
# Build-host setup only: run in an ephemeral Ubuntu 22.04 container/CI runner.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git curl ca-certificates patch pkg-config \
  python3 jq patchelf file unzip \
  libboost-graph-dev libboost-program-options-dev libboost-system-dev \
  libboost-filesystem-dev libboost-thread-dev libeigen3-dev libmetis-dev \
  libsqlite3-dev libglew-dev libgl1-mesa-dev libgoogle-glog-dev libgflags-dev \
  libjpeg-dev libpng-dev libopenexr-dev libopenblas-pthread-dev \
  libfmt-dev robin-map-dev libpugixml-dev libgtest-dev libgmock-dev \
  zlib1g-dev libdeflate-dev libzstd-dev liblzma-dev

# Faiss 1.14.1 requires CMake >=3.24; Ubuntu 22.04 supplies 3.22.
if [[ ! -x /opt/splat-cmake/bin/cmake ]]; then
  archive=$(mktemp)
  trap 'rm -f "$archive"' EXIT
  curl --fail --location --proto '=https' --proto-redir '=https' \
    https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz -o "$archive"
  echo "5a1133ff103c71eb5120e2cc3de922733e7d8a26a98ae716397e8676adb367bf  $archive" | sha256sum -c -
  mkdir -p /opt/splat-cmake
  tar -xzf "$archive" --strip-components=1 -C /opt/splat-cmake
fi
for tool in cmake ctest cpack; do ln -sf "/opt/splat-cmake/bin/$tool" "/usr/local/bin/$tool"; done
