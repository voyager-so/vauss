#!/usr/bin/env python3
"""Audit and package a trusted COLMAP build; Python is not a runtime dependency.

Input contract: build.sh stages notices at PREFIX/share/licenses/<component>,
source/build provenance at PREFIX/share/build/provenance.json, and any required
LGPL source packages at PREFIX/share/sources/<source-package>. SOURCES contains
the unpacked source directories named in SOURCE_COMPONENTS. No downloads occur.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass


class AuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class Rule:
    license: str
    component: str
    packages: tuple[str, ...] = ()
    source_required: bool = False


GCC_EXCEPTION = "GPL-3.0-or-later WITH GCC-exception-3.1"
HOST_LIBRARIES = frozenset(
    {
        "ld-linux-x86-64.so.2", "libc.so.6", "libm.so.6", "libpthread.so.0",
        "libdl.so.2", "librt.so.1", "libresolv.so.2", "libutil.so.1", "libanl.so.1",
    }
)
SOURCE_COMPONENTS = ("colmap", "ceres", "oiio", "tiff", "faiss", "poselib")

# Exact SONAMEs, not a prefix-based approval. A changed ABI requires review here.
# System library approval also checks the owning Debian/Ubuntu binary package.
RULES = {
    "libcolmap.so.4": Rule("BSD-3-Clause", "colmap"),
    "libceres.so.4": Rule("BSD-3-Clause", "ceres"),
    "libOpenImageIO.so.2.5": Rule("Apache-2.0 AND BSD-3-Clause", "oiio"),
    "libOpenImageIO_Util.so.2.5": Rule("Apache-2.0 AND BSD-3-Clause", "oiio"),
    "libtiff.so.6": Rule("libtiff", "tiff"),
    "libtiffxx.so.6": Rule("libtiff", "tiff"),
    "libfaiss.so": Rule("MIT", "faiss"),
    "libPoseLib.so": Rule("BSD-3-Clause", "poselib"),
    "libgcc_s.so.1": Rule(GCC_EXCEPTION, "gcc-runtime", ("libgcc-s1",)),
    "libstdc++.so.6": Rule(GCC_EXCEPTION, "gcc-runtime", ("libstdc++6",)),
    "libgomp.so.1": Rule(GCC_EXCEPTION, "gcc-runtime", ("libgomp1",)),
    "libgfortran.so.5": Rule(GCC_EXCEPTION, "gcc-runtime", ("libgfortran5",)),
    "libquadmath.so.0": Rule("LGPL-2.1-or-later", "gcc-runtime", ("libquadmath0",), True),
    "libglog.so.0": Rule("BSD-3-Clause", "glog", ("libgoogle-glog0v5",)),
    "libgflags.so.2.2": Rule("BSD-3-Clause", "gflags", ("libgflags2.2",)),
    "libgflags_nothreads.so.2.2": Rule("BSD-3-Clause", "gflags", ("libgflags2.2",)),
    "libunwind.so.8": Rule("MIT", "libunwind", ("libunwind8",)),
    "libsqlite3.so.0": Rule("blessing", "sqlite", ("libsqlite3-0",)),
    "libopenblas.so.0": Rule("BSD-3-Clause", "openblas", ("libopenblas0-pthread", "libopenblas0-openmp", "libopenblas0-serial")),
    "libblas.so.3": Rule("BSD-3-Clause", "blas", ("libblas3", "libopenblas0-pthread", "libopenblas0-openmp", "libopenblas0-serial")),
    "liblapack.so.3": Rule("BSD-3-Clause", "lapack", ("liblapack3", "libopenblas0-pthread", "libopenblas0-openmp", "libopenblas0-serial")),
    "libflann.so.1.9": Rule("BSD-3-Clause", "flann", ("libflann1.9",)),
    "libflann_cpp.so.1.9": Rule("BSD-3-Clause", "flann", ("libflann1.9",)),
    "liblz4.so.1": Rule("BSD-2-Clause", "lz4", ("liblz4-1",)),
    # XZ 5.2.5 COPYING assigns liblzma to the public domain. The GPL scripts
    # and optional LGPL getopt belong to other components, not this library.
    "liblzma.so.5": Rule("LicenseRef-Public-Domain", "xz-utils", ("liblzma5",)),
    "libz.so.1": Rule("Zlib", "zlib", ("zlib1g",)),
    "libzstd.so.1": Rule("BSD-3-Clause", "zstd", ("libzstd1",)),
    "libjpeg.so.8": Rule("BSD-3-Clause AND IJG AND Zlib", "libjpeg-turbo", ("libjpeg-turbo8",)),
    "libpng16.so.16": Rule("Libpng", "libpng", ("libpng16-16",)),
    "libdeflate.so.0": Rule("MIT", "libdeflate", ("libdeflate0",)),
    "libfmt.so.8": Rule("MIT", "fmt", ("libfmt8",)),
    "libmetis.so.5": Rule("Apache-2.0", "metis", ("libmetis5",)),
    "libpugixml.so.1": Rule("MIT", "pugixml", ("libpugixml1v5",)),
    "libImath-3_1.so.29": Rule("BSD-3-Clause", "imath", ("libimath-3-1-29",)),
}
for _openexr in ("OpenEXR", "OpenEXRCore", "OpenEXRUtil", "Iex", "IlmThread"):
    RULES[f"lib{_openexr}-3_1.so.30"] = Rule(
        "BSD-3-Clause", "openexr", ("libopenexr-3-1-30",)
    )
for _openexr in ("IlmImf", "IlmImfUtil"):
    RULES[f"lib{_openexr}-2_5.so.25"] = Rule(
        "BSD-3-Clause", "openexr", ("libopenexr25",)
    )
for _ilmbase in ("Imath", "Half", "Iex", "IexMath", "IlmThread"):
    RULES[f"lib{_ilmbase}-2_5.so.25"] = Rule(
        "BSD-3-Clause", "ilmbase", ("libilmbase25",)
    )
for _boost in ("program_options", "filesystem", "system", "graph", "regex", "serialization", "iostreams", "thread", "atomic", "chrono", "date_time"):
    RULES[f"libboost_{_boost}.so.1.74.0"] = Rule(
        "BSL-1.0", "boost", (f"libboost-{_boost.replace('_', '-')}1.74.0",)
    )

FORBIDDEN = re.compile(
    r"lib(?:cholmod|suitesparse|amd|camd|ccolamd|colamd|spqr|umfpack|klu|"
    r"cxsparse|gmp|mpfr|CGAL|jbig|freeimage|cuda|nvidia|cudart|cublas|cufft|"
    r"curand|cusolver|cusparse|nvrtc|cudnn|python)", re.IGNORECASE,
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def command(args: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env, check=False)
    if result.returncode:
        raise AuditError(f"{args[0]} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def parse_ldd(output: str) -> dict[str, Path]:
    closure = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("linux-vdso.so.1 "):
            continue
        if "not found" in line:
            raise AuditError(f"unresolved shared library: {line}")
        match = re.fullmatch(r"(\S+) => (/.*?) \(0x[0-9a-fA-F]+\)", line)
        if match:
            name, value = match.groups()
        else:
            match = re.fullmatch(r"(/.*?) \(0x[0-9a-fA-F]+\)", line)
            if not match:
                raise AuditError(f"unrecognized ldd output: {line}")
            value = match.group(1)
            name = Path(value).name
        path = Path(value)
        if name in closure and closure[name] != path:
            raise AuditError(f"ambiguous library resolution: {name}")
        closure[name] = path
    if not closure:
        raise AuditError("expected a dynamically linked Linux ELF executable")
    return closure


def library_rule(name: str) -> Rule | None:
    if name in HOST_LIBRARIES:
        return None
    if FORBIDDEN.match(name):
        raise AuditError(f"forbidden dependency {name}: GPL, GPU, or unreviewed optional runtime")
    if name not in RULES:
        raise AuditError(f"unreviewed SONAME {name}; add an explicit audited rule before packaging")
    return RULES[name]


def dynamic_info(path: Path) -> tuple[list[str], str | None]:
    headers = command(["readelf", "-h", str(path)])
    if not re.search(r"Class:\s+ELF64", headers) or not re.search(
        r"Machine:\s+Advanced Micro Devices X86-64", headers
    ):
        raise AuditError(f"not a Linux x86-64 ELF file: {path}")
    output = command(["readelf", "-d", str(path)])
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", output)
    sonames = re.findall(r"\(SONAME\).*?\[(.*?)\]", output)
    if len(sonames) > 1:
        raise AuditError(f"multiple SONAMEs: {path}")
    for name in needed + sonames:
        if "/" in name or name in ("", ".", ".."):
            raise AuditError(f"nonrelocatable ELF dependency name {name!r} in {path}")
    return needed, sonames[0] if sonames else None


def glibc_requirement(version_info: str) -> tuple[int, ...]:
    # Defined versions in libstdc++ do not describe what it needs from glibc.
    needs = version_info.partition("Version needs section")[2]
    versions = [tuple(map(int, text.split("."))) for text in
                re.findall(r"Name: GLIBC_(\d+(?:\.\d+)+)\b", needs)]
    return max(versions, default=(0, 0))


def check_glibc(path: Path) -> str:
    required = glibc_requirement(command(["readelf", "--version-info", str(path)]))
    if required > (2, 35):
        raise AuditError(f"{path.name} requires GLIBC_{'.'.join(map(str, required))}; target floor is 2.35")
    return ".".join(map(str, required))


def system_package(path: Path, rule: Rule) -> dict[str, str]:
    real = path.resolve(strict=True)
    candidates = [str(real), str(path)]
    if str(real).startswith("/usr/lib/"):
        candidates.append(str(real)[4:])
    elif str(real).startswith("/lib/"):
        candidates.append("/usr" + str(real))
    owners = set()
    for candidate in dict.fromkeys(candidates):
        result = subprocess.run(["dpkg-query", "-S", candidate], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if ": " in line:
                    owners.add(line.split(": ", 1)[0])
    valid = sorted(owner for owner in owners if owner.split(":", 1)[0] in rule.packages)
    if len(valid) != 1:
        raise AuditError(f"{path}: expected owner in {rule.packages}, found {sorted(owners)}")
    owner = valid[0]
    output = command(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}", owner])
    fields = output.split("\t")
    if len(fields) != 4 or not all(fields):
        raise AuditError(f"incomplete source package metadata for {owner}")
    return dict(zip(("package", "version", "source_package", "source_version"), fields))


def copy_regular_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise AuditError(f"missing real input directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in (".git", "__pycache__") for part in relative.parts):
            continue
        target = destination / relative
        if path.is_symlink():
            raise AuditError(f"metadata/source input contains a symlink: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            target.chmod(0o644)
        else:
            raise AuditError(f"metadata/source input is not a regular file: {path}")


def require_notices(root: Path, component: str) -> None:
    directory = root / component
    if not directory.is_dir() or not any(path.is_file() and path.stat().st_size for path in directory.rglob("*")):
        raise AuditError(f"missing staged license notices for {component}: {directory}")


def parse_dsc(text: str) -> dict[str, str]:
    fields = {}
    current = None
    for line in text.splitlines():
        if line.startswith("-----BEGIN PGP SIGNATURE-----"):
            break
        if line.startswith((" ", "\t")) and current:
            fields[current] += "\n" + line.strip()
        elif re.match(r"[A-Za-z][A-Za-z0-9-]*:", line):
            current, value = line.split(":", 1)
            fields[current] = value.strip()
        else:
            current = None
    return fields


def verify_source_package(directory: Path, package: str, version: str) -> list[Path]:
    manifests = sorted(directory.glob("*.dsc"))
    for manifest in manifests:
        fields = parse_dsc(manifest.read_text())
        if fields.get("Source") != package or fields.get("Version") != version:
            continue
        files = [manifest]
        checksums = fields.get("Checksums-Sha256", "").strip().splitlines()
        if not checksums:
            raise AuditError(f"source package has no SHA256 file list: {manifest}")
        for line in checksums:
            parts = line.split()
            if len(parts) != 3 or not re.fullmatch(r"[a-f0-9]{64}", parts[0]) or not parts[1].isdigit():
                raise AuditError(f"invalid source package checksum line in {manifest}: {line}")
            expected, size, name = parts
            if Path(name).name != name or name in (".", ".."):
                raise AuditError(f"unsafe source package filename: {name}")
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != int(size) or digest(path) != expected:
                raise AuditError(f"source package checksum mismatch or missing file: {path}")
            files.append(path)
        return files
    raise AuditError(f"LGPL source package required: {directory} must contain {package} {version} .dsc and referenced sources")


def copy_system_notices(info: dict[str, str], rule: Rule, prefix: Path, bundle: Path) -> None:
    package = info["package"].split(":", 1)[0]
    source = Path("/usr/share/doc") / package / "copyright"
    if not source.is_file() or not source.stat().st_size:
        raise AuditError(f"missing system library copyright: {source}")
    destination = bundle / "share/licenses/system" / package
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / "copyright")
    info["source_locator"] = (
        f"https://launchpad.net/ubuntu/+source/{info['source_package']}/{info['source_version']}"
    )
    # Debian copyright files refer to shared license texts by absolute filename.
    common = Path("/usr/share/common-licenses")
    common_dest = bundle / "share/licenses/common-licenses"
    common_dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(common.iterdir()):
        if path.is_file():
            shutil.copyfile(path, common_dest / path.name)
    if rule.source_required:
        directory = prefix / "share/sources" / info["source_package"]
        files = verify_source_package(directory, info["source_package"], info["source_version"])
        destination = bundle / "share/sources" / info["source_package"]
        destination.mkdir(parents=True, exist_ok=True)
        for path in files:
            shutil.copyfile(path, destination / path.name)
        info["included_source"] = str(destination.relative_to(bundle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def deterministic_archive(root: Path, archive: Path, epoch: int) -> None:
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch, compresslevel=9) as compressed:
            with tarfile.open(mode="w|", fileobj=compressed, format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(root.rglob("*")):
                    if not path.is_dir() and not path.is_file() and not path.is_symlink():
                        raise AuditError(f"unsupported archive member: {path}")
                    info = tar.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = epoch
                    info.pax_headers = {}
                    info.mode = 0o755 if path.is_dir() or os.access(path, os.X_OK) else 0o644
                    if path.is_file() and not path.is_symlink():
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
                    else:
                        tar.addfile(info)


def bundle(args: argparse.Namespace) -> dict[str, object]:
    if args.variant != "cpu":
        raise AuditError("CUDA packaging is blocked: SiftGPU's bundled noncommercial grant has not been cleared")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        raise AuditError("version must be a safe archive filename component")
    prefix = args.prefix.resolve(strict=True)
    install = args.install.resolve(strict=True)
    sources = args.sources.resolve(strict=True)
    recipe = args.recipe_dir.resolve(strict=True)
    executable = install / "bin/colmap"
    if not executable.is_file() or executable.is_symlink():
        raise AuditError(f"missing regular COLMAP executable: {executable}")
    provenance_path = prefix / "share/build/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if not isinstance(provenance, dict) or not provenance:
        raise AuditError("build provenance must be a nonempty JSON object")
    for component in SOURCE_COMPONENTS:
        if not (sources / component).is_dir():
            raise AuditError(f"missing pinned source directory: {sources / component}")
        require_notices(prefix / "share/licenses", component)
    environment = os.environ.copy()
    for variable in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_LIBRARY_PATH"):
        environment.pop(variable, None)
    environment["LC_ALL"] = "C"
    environment["LD_LIBRARY_PATH"] = ":".join(
        str(root / sub) for root in (install, prefix)
        for sub in ("lib", "lib64", "lib/x86_64-linux-gnu") if (root / sub).is_dir()
    )
    closure = parse_ldd(command(["ldd", str(executable)], env=environment))
    for name in closure:
        library_rule(name)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"colmap-{args.version}-linux-x64-{args.variant}"
    archive = args.output / f"{stem}.tar.gz"
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    if not 0 <= epoch <= 0xFFFFFFFF:
        raise AuditError("SOURCE_DATE_EPOCH must fit an unsigned 32-bit timestamp")
    with tempfile.TemporaryDirectory(prefix=f".{stem}-", dir=args.output) as working:
        root = Path(working) / "bundle"
        (root / "bin").mkdir(parents=True)
        (root / "lib").mkdir()
        copy_regular_tree(prefix / "share/licenses", root / "share/licenses")
        copy_regular_tree(prefix / "share/build", root / "share/build")
        copy_regular_tree(recipe, root / "share/audit/recipe")
        records = []
        for name, source in [("colmap", executable), *sorted(closure.items())]:
            host = name in HOST_LIBRARIES
            rule = None if name == "colmap" else library_rule(name)
            needed, soname = dynamic_info(source)
            if soname and name != soname:
                raise AuditError(f"resolved SONAME mismatch for {name}: {soname}")
            if any(dependency not in closure for dependency in needed):
                raise AuditError(f"incomplete dynamic closure for {name}: {needed}")
            record = {"name": name, "needed": sorted(needed), "soname": soname,
                      "host_only": host, "original_sha256": digest(source)}
            if host:
                record["license"] = "LGPL-2.1-or-later (host system; not redistributed)"
                records.append(record)
                continue
            record["glibc_required"] = check_glibc(source)
            if rule:
                record.update(license=rule.license, component=rule.component)
                resolved = source.resolve(strict=True)
                if resolved.is_relative_to(prefix) or resolved.is_relative_to(install):
                    if rule.packages:
                        raise AuditError(f"{name} must come from its reviewed system package, not an unrecorded local build")
                    require_notices(prefix / "share/licenses", rule.component)
                else:
                    if not rule.packages:
                        raise AuditError(f"{name} must come from the pinned source build")
                    info = system_package(resolved, rule)
                    copy_system_notices(info, rule, prefix, root)
                    record["system_package"] = info
            else:
                record.update(license="BSD-3-Clause", component="colmap")
            destination = root / ("bin" if name == "colmap" else "lib") / name
            shutil.copyfile(source, destination)
            destination.chmod(0o755)
            # DT_RPATH makes the lib directory transitive, including any dlopen by
            # dependencies. Every object is patched so moved bundles stay local.
            rpath = "$ORIGIN/../lib" if name == "colmap" else "$ORIGIN"
            command(["patchelf", "--force-rpath", "--set-rpath", rpath, str(destination)])
            if command(["patchelf", "--print-rpath", str(destination)]).strip() != rpath:
                raise AuditError(f"failed to set relocatable rpath: {destination}")
            record.update(path=destination.relative_to(root).as_posix(), sha256=digest(destination))
            records.append(record)
        verify_environment = environment.copy()
        verify_environment.pop("LD_LIBRARY_PATH", None)
        relocated = parse_ldd(command(["ldd", str(root / "bin/colmap")], env=verify_environment))
        if set(relocated) != set(closure):
            raise AuditError("relocated dependency closure changed")
        for name, path in relocated.items():
            if name not in HOST_LIBRARIES and not path.resolve().is_relative_to(root / "lib"):
                raise AuditError(f"relocated library escaped bundle: {name} => {path}")
        audit = {
            "schema_version": 1, "variant": args.variant, "version": args.version,
            "target": "x86_64-unknown-linux-gnu", "glibc_floor": "2.35",
            "runtime": "Linux x86-64, glibc >= 2.35; no Python, GPU driver, CUDA, GUI, or external image plugins",
            "license_policy": "Exact SONAME and package allowlist; GCC runtime exceptions permitted; LGPL libraries require complete matching source packages",
            "libraries": records,
        }
        write_json(root / "share/audit/dynamic-closure.json", audit)
        (root / "share/licenses/README.txt").write_text(
            "COLMAP helper: separate executable, not linked into splat.\n"
            "See share/audit/dynamic-closure.json for the exact library inventory.\n"
            "Notices for statically built components are in this directory.\n"
            "System copyright files referring to /usr/share/common-licenses use the\n"
            "copies in share/licenses/common-licenses.\n"
            "GCC runtime libraries use GPL-3.0-or-later WITH GCC-exception-3.1;\n"
            "the exception applies to qualifying independently compiled programs.\n"
            "libquadmath, when present, is LGPL-2.1-or-later and does not use that\n"
            "exception. Its matching source package is in share/sources.\n"
            "You may replace or relink LGPL shared libraries and reverse engineer\n"
            "the executable to debug your modifications to those libraries.\n"
            "No vendor GPU driver or Python runtime is redistributed.\n"
        )
        write_json(root / "share/audit/files.json", {
            path.relative_to(root).as_posix(): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()
        })
        temporary_archive = Path(working) / archive.name
        deterministic_archive(root, temporary_archive, epoch)
        candidate = {
            "id": "colmap", "target": "x86_64-unknown-linux-gnu", "version": args.version,
            "url": None, "github_api_url": None, "sha256": digest(temporary_archive),
            "bytes": temporary_archive.stat().st_size, "format": "tar_gz",
            "entrypoint": "bin/colmap", "executable": True,
            "license": "BSD-3-Clause AND other reviewed licenses; see share/licenses",
            "license_url": "https://github.com/colmap/colmap/blob/4.2.1/COPYING.txt",
            "source_url": "https://github.com/colmap/colmap/tree/4.2.1",
            "requires_gpl_consent": False,
            "note": "UNHOSTED PIN CANDIDATE; requires immutable hosting before adding to splat-fetch. CPU only. Linux glibc >= 2.35. Preserve bundled library notices and sources.",
        }
        os.replace(temporary_archive, archive)
        (args.output / f"{archive.name}.sha256").write_text(f"{candidate['sha256']}  {archive.name}\n")
        write_json(args.output / f"{stem}.pin-candidate.json", candidate)
        write_json(args.output / f"{stem}.audit.json", audit)
        return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prefix", "install", "sources", "output", "recipe-dir"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--version", default="4.2.1-splat.1")
    args = parser.parse_args()
    try:
        print(json.dumps(bundle(args), indent=2, sort_keys=True))
    except (AuditError, OSError, ValueError) as error:
        print(f"bundle audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
