#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
work=${1:?usage: build.sh WORK_DIR OUTPUT_DIR [cpu|cuda]}
output=${2:?usage: build.sh WORK_DIR OUTPUT_DIR [cpu|cuda]}
variant=${3:-cpu}
case "$variant" in
  cpu) ;;
  cuda)
    echo 'CUDA bundle blocked: bundled SiftGPU has an educational/research/non-profit license grant. A reviewed replacement or license is required.' >&2
    exit 1 ;;
  *) echo "Unknown variant: $variant" >&2; exit 1 ;;
esac
jobs=${CMAKE_BUILD_PARALLEL_LEVEL:-4}
[[ "$jobs" =~ ^[1-4]$ ]] || { echo 'Use 1-4 build jobs' >&2; exit 1; }
mkdir -p "$work" "$output"
work=$(cd "$work" && pwd)
output=$(cd "$output" && pwd)
sources=$work/sources
prefix=$work/prefix
installed=$work/installed
mkdir -p "$sources" "$prefix/share/licenses" "$prefix/share/build"

# All source bytes, including COLMAP's FetchContent dependencies, are pinned here.
python3 - "$recipe/sources.json" "$work" <<'PY'
import hashlib, json, pathlib, subprocess, sys, urllib.request, zipfile
manifest, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for source in json.loads(manifest.read_text()):
    archive = work / (source['name'] + ('.zip' if source['url'].endswith('.zip') else '.tar.gz'))
    if not archive.exists():
        with urllib.request.urlopen(source['url'], timeout=120) as remote, archive.open('wb') as local:
            import shutil
            shutil.copyfileobj(remote, local)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == source['sha256'], archive
    target = work / 'sources' / source['name']
    if target.exists():
        continue
    target.mkdir()
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        root, = target.iterdir()
        for child in root.iterdir():
            child.rename(target / child.name)
        root.rmdir()
    else:
        subprocess.run(['tar', '-xzf', str(archive), '--strip-components=1', '-C', str(target)], check=True)
PY
if [[ ! -f "$sources/colmap/.splat-patched" ]]; then
  patch --directory "$sources/colmap" -p1 < "$recipe/no-cholmod.patch"
  touch "$sources/colmap/.splat-patched"
fi

common=(-GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix"
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_PREFIX_PATH="$prefix"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON)
build_install() {
  local name=$1
  shift
  cmake -S "$sources/$name" -B "$work/build-$name" "${common[@]}" "$@"
  cmake --build "$work/build-$name" --parallel "$jobs"
  cmake --install "$work/build-$name"
}
# Ubuntu's TIFF can pull GPL libjbig into the closure. Build only the codecs needed.
build_install tiff -Djbig=OFF -Dwebp=OFF -Dlzma=OFF -Dtiff-tools=OFF -Dtiff-tests=OFF \
  -Dtiff-contrib=OFF -Dtiff-docs=OFF
build_install oiio -DUSE_PYTHON=OFF -DUSE_QT=OFF -DOIIO_BUILD_TOOLS=OFF \
  -DOIIO_BUILD_TESTS=OFF -DOIIO_BUILD_DOCS=OFF -DBUILD_MISSING_DEPS=OFF \
  -DBUILD_MISSING_FMT=OFF -DBUILD_MISSING_ROBINMAP=OFF \
  -DUSE_FFmpeg=OFF -DUSE_LibRaw=OFF -DUSE_Libheif=OFF -DUSE_OpenCV=OFF \
  -DUSE_OpenVDB=OFF -DUSE_OpenColorIO=OFF -DUSE_Freetype=OFF -DUSE_DCMTK=OFF \
  -DUSE_OpenJPEG=OFF -DUSE_Ptex=OFF -DUSE_WebP=OFF -DUSE_GIF=OFF \
  -DUSE_TBB=OFF -DUSE_BZip2=OFF -DUSE_EXTERNAL_PUGIXML=ON
build_install ceres -DSUITESPARSE=OFF -DEIGENSPARSE=ON -DEIGENMETIS=OFF \
  -DUSE_CUDA=OFF -DLAPACK=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_DOCUMENTATION=OFF -DCMAKE_CXX_FLAGS=-DEIGEN_MPL2_ONLY

cmake -S "$sources/colmap" -B "$work/build-colmap" "${common[@]}" \
  -DCMAKE_INSTALL_PREFIX="$installed" -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_FLAGS=-DEIGEN_MPL2_ONLY -DCOLMAP_HASH_MAP_BACKEND=STD \
  -DCUDA_ENABLED=OFF -DHIP_ENABLED=OFF -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF -DONNX_ENABLED=OFF -DDOWNLOAD_ENABLED=OFF \
  -DMVS_ENABLED=OFF -DIPO_ENABLED=OFF -DTESTS_ENABLED="${SPLAT_RUN_SOLVER_TESTS:-OFF}" \
  -DFETCHCONTENT_SOURCE_DIR_POSELIB="$sources/poselib" \
  -DFETCHCONTENT_SOURCE_DIR_FAISS="$sources/faiss"
cmake --build "$work/build-colmap" --parallel "$jobs" --target colmap_main
if [[ ${SPLAT_RUN_SOLVER_TESTS:-0} == 1 ]]; then
  cmake --build "$work/build-colmap" --parallel "$jobs" \
    --target colmap_optim_sparse_cholesky_test colmap_optim_least_absolute_deviations_test
  ctest --test-dir "$work/build-colmap" --output-on-failure \
    -R 'sparse_cholesky|least_absolute_deviations'
fi
install -D -m755 "$work/build-colmap/src/colmap/exe/colmap" "$installed/bin/colmap"

python3 - "$sources" "$prefix" "$recipe" <<'PY'
import hashlib, json, pathlib, shutil, subprocess, sys
sources, prefix, recipe = map(pathlib.Path, sys.argv[1:])
for name in ('colmap','oiio','ceres','tiff','poselib','faiss'):
    dest = prefix/'share/licenses'/name
    dest.mkdir(parents=True, exist_ok=True)
    for path in (sources/name).iterdir():
        if path.is_file() and path.name.upper().startswith(('LICENSE','COPYING','NOTICE','THIRD-PARTY')):
            shutil.copy2(path, dest/path.name)
for name in ('VLFeat','PoissonRecon'):
    dest = prefix/'share/licenses'/name
    dest.mkdir(exist_ok=True)
    shutil.copy2(sources/'colmap/src/thirdparty'/name/'LICENSE', dest/'LICENSE')
for name in ('libeigen3-dev','libmetis-dev','libboost1.74-dev','libfmt-dev','robin-map-dev'):
    dest = prefix/'share/licenses'/name
    dest.mkdir(exist_ok=True)
    shutil.copy2(pathlib.Path('/usr/share/doc')/name/'copyright', dest/'copyright')
record = {'sources':json.loads((recipe/'sources.json').read_text()),
          'colmap_patch_sha256':hashlib.sha256((recipe/'no-cholmod.patch').read_bytes()).hexdigest(),
          'system_packages':subprocess.check_output(['dpkg-query','-W','-f=${binary:Package} ${Version}\n'],text=True),
          'variant':'cpu','solver':'Eigen SimplicialLLT with LDLT fallback; Ceres Eigen sparse',
          'disabled':['CGAL','LSD','CHOLMOD','SuiteSparse','SiftGPU','CUDA','ONNX','GUI','MVS'],
          'glibc_minimum':'2.35','version':'4.2.1-splat.1'}
(prefix/'share/build/provenance.json').write_text(json.dumps(record,indent=2)+'\n')
PY
# Ubuntu OpenBLAS's Fortran runtime can depend on LGPL libquadmath. Ship the
# matching Debian-format source package alongside the unmodified shared library.
quadmath_source=$(dpkg-query -W -f='${source:Package}' libquadmath0)
quadmath_version=$(dpkg-query -W -f='${source:Version}' libquadmath0)
source_dir=$prefix/share/sources/$quadmath_source
if ! compgen -G "$source_dir/*.dsc" >/dev/null; then
  mkdir -p "$source_dir"
  sed -n '/^deb /s/^deb /deb-src /p' /etc/apt/sources.list > /etc/apt/sources.list.d/splat-source.list
  apt-get update
  (cd "$source_dir" && apt-get source --download-only "$quadmath_source=$quadmath_version")
fi
python3 "$recipe/bundle.py" --prefix "$prefix" --install "$installed" \
  --sources "$sources" --output "$output" --variant "$variant" \
  --version 4.2.1-splat.1 --recipe-dir "$recipe"
