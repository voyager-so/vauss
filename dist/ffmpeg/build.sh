#!/usr/bin/env bash
# Builds the static LGPL FFmpeg (ffmpeg + ffprobe) that splat downloads as `ffmpeg-public`, from the
# pinned tarballs in sources.txt and the configure flags in config/. Every input is in this folder,
# so the published release carries complete corresponding source: the tarballs, this script, the
# flags, and each target's configure log.
#
#   dist/ffmpeg/build.sh TARGET OUT
#
# TARGET: x86_64-unknown-linux-gnu (on Ubuntu 22.04), aarch64-apple-darwin (on macOS arm64), or
# x86_64-pc-windows-msvc (cross-compiled with mingw-w64 on Ubuntu 24.04; the triple is the splat
# target the binary serves). OUT gets ffmpeg-<version>-<TARGET>.tar.gz (.zip on Windows) and a
# build-<TARGET>/ folder with the configure output.
#
# Needs: curl, tar, xz, make, a C compiler, pkg-config, meson, ninja, python3; nasm on x86-64;
# mingw-w64 for Windows. SOURCES (default OUT/sources) caches the verified tarballs; JOBS sets
# make's parallelism.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# The prerelease these binaries are published as (SOURCE.txt points there). Bump it for every build
# that changes the binaries; release.sh refuses any other tag.
release_tag=ffmpeg-public-v3
target=${1:?usage: build.sh TARGET OUT}
mkdir -p "${2:?usage: build.sh TARGET OUT}"
out="$(cd "$2" && pwd)"
sources="${SOURCES:-$out/sources}"
work="${WORK:-$out/work-$target}"
prefix="$work/prefix"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}"
die() { echo "build.sh: $*" >&2; exit 1; }

case "$target" in
  x86_64-unknown-linux-gnu) os=linux; components=(zlib freetype harfbuzz opus libvpx dav1d vulkan-headers) ;;
  aarch64-apple-darwin) os=macos; components=(zlib freetype harfbuzz opus libvpx dav1d)
    export MACOSX_DEPLOYMENT_TARGET=11.0 ;;
  x86_64-pc-windows-msvc) os=windows; components=(zlib freetype harfbuzz opus libvpx dav1d)
    cross=x86_64-w64-mingw32 ;;
  *) die "unknown target $target" ;;
esac

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi; }

# row NAME: the sources.txt line for NAME, as an array in $src.
row() {
  local line
  line="$(grep -E "^$1 " "$here/sources.txt")" || die "$1 is not in sources.txt"
  read -r -a src <<<"$line"
}

# fetch NAME: downloads NAME's tarball into $sources (once), trying each URL in turn, and checks it.
fetch() {
  row "$1"
  local file="$sources/${src[2]}" url
  mkdir -p "$sources"
  if [[ ! -f "$file" ]] || [[ "$(sha256 "$file")" != "${src[3]}" ]]; then
    for url in "${src[@]:6}"; do
      curl -fsSL --retry 3 --connect-timeout 20 -o "$file.part" "$url" && mv "$file.part" "$file" && break
      echo "build.sh: could not download $url" >&2
    done
  fi
  [[ -f "$file" ]] || die "could not download ${src[2]}"
  [[ "$(sha256 "$file")" == "${src[3]}" ]] || die "SHA-256 mismatch for ${src[2]}"
}

# topdir FILE: the top-level folder of a source tarball (sed reads it all: no SIGPIPE under pipefail).
topdir() { tar -tf "$1" | sed -n '1s|/.*||p'; }

# unpack NAME: a fresh copy of NAME's source tree in $work; prints its path.
unpack() {
  fetch "$1"
  local top
  top="$(topdir "$sources/${src[2]}")"
  rm -rf "${work:?}/$top"
  tar -xf "$sources/${src[2]}" -C "$work"
  echo "$work/$top"
}

mkdir -p "$work" "$prefix/lib/pkgconfig" "$prefix/include"
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig" PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig"

# FFmpeg's release signature, when gpg is there (the SHA-256 pin is what the build trusts).
fetch ffmpeg
if command -v gpg >/dev/null; then
  export GNUPGHOME="$work/gnupg"
  rm -rf "$GNUPGHOME" && mkdir -m 700 "$GNUPGHOME"
  # Downloaded once into $sources (the release carries both), so a rebuild from them is offline.
  [[ -s "$sources/ffmpeg-devel.asc" ]] || curl -fsSL --retry 3 -o "$sources/ffmpeg-devel.asc" https://ffmpeg.org/ffmpeg-devel.asc
  [[ -s "$sources/${src[2]}.asc" ]] || curl -fsSL --retry 3 -o "$sources/${src[2]}.asc" "${src[6]}.asc"
  gpg --batch --quiet --import "$sources/ffmpeg-devel.asc" 2>/dev/null
  gpg --batch --status-fd 1 --verify "$sources/${src[2]}.asc" "$sources/${src[2]}" 2>/dev/null |
    grep -q 'VALIDSIG FCF986EA15E6E293A5644F10B4322F04D67658D8' || die "FFmpeg's release signature did not verify"
fi

cc=cc; host_flags=(); meson_cross=()
if [[ $os == windows ]]; then
  cc=$cross-gcc
  host_flags=(--host=$cross)
  printf '%s\n' "[binaries]" "c = '$cross-gcc'" "cpp = '$cross-g++'" "ar = '$cross-ar'" "strip = '$cross-strip'" \
    "windres = '$cross-windres'" "pkg-config = 'pkg-config'" "[host_machine]" "system = 'windows'" "cpu_family = 'x86_64'" "cpu = 'x86_64'" \
    "endian = 'little'" >"$work/mingw.meson"
  meson_cross=(--cross-file "$work/mingw.meson")
fi
export CFLAGS="-O2 -fPIC${MACOSX_DEPLOYMENT_TARGET:+ -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET}"
[[ $os == windows ]] && export CFLAGS="-O2"
export CXXFLAGS="$CFLAGS"
meson_common=(--prefix="$prefix" --libdir=lib --buildtype=release --default-library=static)

build_zlib() {
  local s; s="$(unpack zlib)"
  if [[ $os == windows ]]; then
    make -C "$s" -f win32/Makefile.gcc PREFIX=$cross- -j"$jobs" libz.a
    cp "$s/libz.a" "$prefix/lib/" && cp "$s/zlib.h" "$s/zconf.h" "$prefix/include/"
  else
    (cd "$s" && CC=$cc ./configure --prefix="$prefix" --static && make -j"$jobs" && make install)
  fi
}

build_freetype() {
  local s; s="$(unpack freetype)"
  meson setup "$s/build" "$s" ${meson_cross[@]+"${meson_cross[@]}"} "${meson_common[@]}" \
    -Dbrotli=disabled -Dbzip2=disabled -Dharfbuzz=disabled -Dpng=disabled -Dzlib=none -Dtests=disabled
  ninja -C "$s/build" -j"$jobs" install
}

build_harfbuzz() {
  local s; s="$(unpack harfbuzz)"
  meson setup "$s/build" "$s" ${meson_cross[@]+"${meson_cross[@]}"} "${meson_common[@]}" -Dfreetype=enabled \
    -Dglib=disabled -Dgobject=disabled -Dcairo=disabled -Dchafa=disabled -Dicu=disabled -Dgraphite2=disabled \
    -Dpng=disabled -Dzlib=disabled -Dfontations=disabled -Dharfrust=disabled -Dkbts=disabled -Dwasm=disabled \
    -Dgdi=disabled -Ddirectwrite=disabled -Dcoretext=disabled -Draster=disabled -Dvector=disabled -Dgpu=disabled \
    -Dsubset=disabled -Dtests=disabled -Dintrospection=disabled -Ddocs=disabled -Dutilities=disabled -Dbenchmark=disabled
  ninja -C "$s/build" -j"$jobs" install
}

build_opus() {
  local s; s="$(unpack opus)"
  (cd "$s" && ./configure --prefix="$prefix" ${host_flags[@]+"${host_flags[@]}"} --disable-shared --enable-static \
    --disable-doc --disable-extra-programs --disable-stack-protector && make -j"$jobs" && make install)
}

build_libvpx() {
  local s vpx_target
  s="$(unpack libvpx)"
  case "$target" in
    x86_64-unknown-linux-gnu) vpx_target=x86_64-linux-gcc ;;
    aarch64-apple-darwin) vpx_target=arm64-darwin20-gcc ;;
    x86_64-pc-windows-msvc) vpx_target=x86_64-win64-gcc ;;
  esac
  (cd "$s" && CROSS=${cross:+$cross-} ./configure --prefix="$prefix" --target=$vpx_target --enable-static \
    --disable-shared --enable-pic --disable-examples --disable-tools --disable-docs --disable-unit-tests \
    --enable-vp9-highbitdepth --enable-runtime-cpu-detect && make -j"$jobs" && make install)
}

build_dav1d() {
  local s; s="$(unpack dav1d)"
  meson setup "$s/build" "$s" ${meson_cross[@]+"${meson_cross[@]}"} "${meson_common[@]}" \
    -Denable_tools=false -Denable_tests=false
  ninja -C "$s/build" -j"$jobs" install
}

build_vulkan-headers() {
  local s; s="$(unpack vulkan-headers)"
  cp -R "$s/include/." "$prefix/include/"
}

for c in "${components[@]}"; do "build_$c"; done

# FFmpeg itself.
s="$(unpack ffmpeg)"
flags=()
for f in "$here/config/common.flags" "$here/config/$target.flags"; do
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] || flags+=("$line")
  done <"$f"
done
for f in "${flags[@]}"; do
  case "$f" in --enable-gpl | --enable-nonfree | --enable-version3) die "refusing $f: this build is LGPL-2.1-or-later only" ;; esac
done
row ffmpeg
version="${src[1]}"
(cd "$s" && ./configure --prefix="$prefix" --extra-version=splat "--extra-cflags=-I$prefix/include" \
  "--extra-ldflags=-L$prefix/lib" "${flags[@]}") || { tail -50 "$s/ffbuild/config.log"; die "configure failed"; }
grep -qx '#define CONFIG_GPL 0' "$s/config.h" && grep -qx '#define CONFIG_NONFREE 0' "$s/config.h" &&
  grep -qx '#define CONFIG_VERSION3 0' "$s/config.h" || die "configure enabled a GPL, nonfree or version-3 component"
make -C "$s" -j"$jobs"

# The package: the two programs, every component's licence, and where the source is.
exe=""; [[ $os == windows ]] && exe=.exe
pkg="$work/pkg"
rm -rf "$pkg" && mkdir -p "$pkg/LICENSES"
cp "$s/ffmpeg$exe" "$s/ffprobe$exe" "$pkg/"
strip_tool=strip; [[ $os == windows ]] && strip_tool=$cross-strip
"$strip_tool" "$pkg/ffmpeg$exe" "$pkg/ffprobe$exe"
{
  echo "FFmpeg $version for $target, built by splat's dist/ffmpeg/build.sh: LGPL-2.1-or-later, static,"
  echo "no GPL or nonfree components (no libx264/libx265). Statically linked components:"
  echo
} >"$pkg/SOURCE.txt"
for c in ffmpeg "${components[@]}"; do
  row "$c"
  mkdir -p "$pkg/LICENSES/$c"
  IFS=, read -r -a lic <<<"${src[5]}"
  for l in "${lic[@]}"; do cp "$work/$(topdir "$sources/${src[2]}")/$l" "$pkg/LICENSES/$c/"; done
  printf '  %-15s %-10s %-18s %s\n                 sha256 %s\n' "$c" "${src[1]}" "${src[4]//_/ }" "${src[6]}" "${src[3]}" >>"$pkg/SOURCE.txt"
done
if [[ $os == windows ]]; then
  # The mingw-w64 C runtime and GCC's libgcc are linked in statically.
  mkdir -p "$pkg/LICENSES/mingw-w64"
  cp /usr/share/doc/mingw-w64-common/copyright "$pkg/LICENSES/mingw-w64/COPYING" 2>/dev/null ||
    cp /usr/share/doc/mingw-w64/copyright "$pkg/LICENSES/mingw-w64/COPYING"
  cp /usr/share/doc/gcc-mingw-w64-base/copyright "$pkg/LICENSES/mingw-w64/GCC-COPYING" 2>/dev/null || true
  echo "  mingw-w64 runtime and libgcc (toolchain; notices in LICENSES/mingw-w64)" >>"$pkg/SOURCE.txt"
fi
cat >>"$pkg/SOURCE.txt" <<EOF

Every tarball above, the build script, the configure flags and this target's configure output are
published with the binaries at https://github.com/voyager-so/vauss/releases/tag/$release_tag
(the build kit is also in the splat repository under dist/ffmpeg/). Licence texts: LICENSES/.
EOF

# The build record: the configure line FFmpeg reports, its full log, and the toolchain.
rec="$out/build-$target"
rm -rf "$rec" && mkdir -p "$rec"
sed -n 's/^#define FFMPEG_CONFIGURATION "\(.*\)"$/\1/p' "$s/config.h" | sed "s|$prefix|PREFIX|g" >"$rec/configure-line.txt"
sed "s|$prefix|PREFIX|g" "$s/ffbuild/config.log" >"$rec/config.log"
cp "$s/config.h" "$rec/config.h"
{
  echo "target: $target"
  uname -a
  "${cross:+$cross-}gcc" --version 2>/dev/null | head -1 || cc --version | head -1
  [[ $os == macos ]] && { sw_vers; xcrun --show-sdk-version; }
  command -v nasm >/dev/null && nasm -v
  meson --version | sed 's/^/meson /'
  command -v dpkg-query >/dev/null && dpkg-query -W libc6 gcc binutils mingw-w64-common gcc-mingw-w64-x86-64-win32 2>/dev/null || true
} >"$rec/toolchain.txt" 2>&1
cp "$rec/configure-line.txt" "$pkg/LICENSES/ffmpeg/CONFIGURATION"

name="ffmpeg-$version-$target"
python3 "$here/pack.py" "$pkg" "$out/$name.$([[ $os == windows ]] && echo zip || echo tar.gz)"
ls -l "$out"
