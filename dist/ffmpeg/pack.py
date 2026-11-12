"""Packs a folder into a reproducible .tar.gz or .zip: sorted names, fixed times, no owners.

    python3 pack.py DIR OUT.tar.gz|OUT.zip

The archive holds DIR's contents at its root (ffmpeg and ffprobe sit at the top, as splat-fetch expects).
"""
import gzip
import os
import sys
import tarfile
import zipfile

src, out = sys.argv[1], sys.argv[2]
files = sorted(
    os.path.relpath(os.path.join(root, name), src).replace(os.sep, "/")
    for root, _, names in os.walk(src)
    for name in names
)


def mode(rel):
    return 0o755 if os.access(os.path.join(src, rel), os.X_OK) else 0o644


if out.endswith(".zip"):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in files:
            info = zipfile.ZipInfo(rel, (1980, 1, 1, 0, 0, 0))
            info.external_attr = mode(rel) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(os.path.join(src, rel), "rb") as f:
                z.writestr(info, f.read(), compresslevel=9)
else:
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for rel in files:
                info = tar.gettarinfo(os.path.join(src, rel), arcname=rel)
                info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                info.mode = mode(rel)
                with open(os.path.join(src, rel), "rb") as f:
                    tar.addfile(info, f)
print(out, os.path.getsize(out))
