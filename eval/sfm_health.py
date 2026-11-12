#!/usr/bin/env python3
"""SfM health report for a COLMAP sparse model (binary or text). Pure numpy, no pycolmap.

Usage:
    python eval/sfm_health.py SPARSE_DIR [--images IMAGES_DIR | --num-input N] [--video]
                                         [--holdout-every 8] [--json out.json]

SPARSE_DIR holds cameras/images/points3D as .bin or .txt (e.g. dataset/sparse/0).
Prints a one-screen summary and a list of gate results; exits 1 if any gate fails.
Gates are provisional.
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

# model_id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".exr"}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_bin(d: Path):
    cams, imgs = {}, {}
    with open(d / "cameras.bin", "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cid, mid, w, h = _read(f, "iiQQ")
            name, n = CAMERA_MODELS[mid]
            cams[cid] = {"model": name, "width": w, "height": h, "params": list(_read(f, "d" * n))}
    with open(d / "images.bin", "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            iid, *qt, cid = _read(f, "i7di")
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            n2d = _read(f, "Q")[0]
            pts = np.frombuffer(f.read(24 * n2d), dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            obs = pts[pts["id"] >= 0]
            imgs[iid] = {"name": name.decode(), "q": qt[:4], "t": qt[4:], "camera_id": cid,
                         "n2d": int(n2d), "n3d": len(obs),
                         "obs_xy": np.stack([obs["x"], obs["y"]], 1), "obs_id": obs["id"].astype(np.int64)}
    errs, tracks, xyz, pids = [], [], [], []
    with open(d / "points3D.bin", "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            pid, x, y, z, _r, _g, _b, err, tl = _read(f, "Q3d3BdQ")
            f.seek(8 * tl, 1)
            errs.append(err); tracks.append(tl); xyz.append((x, y, z)); pids.append(pid)
    return cams, imgs, np.array(errs), np.array(tracks), np.array(xyz).reshape(-1, 3), np.array(pids, np.int64)


def read_txt(d: Path):
    def lines(p):
        return [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    cams, imgs = {}, {}
    for l in lines(d / "cameras.txt"):
        t = l.split()
        cams[int(t[0])] = {"model": t[1], "width": int(t[2]), "height": int(t[3]), "params": [float(v) for v in t[4:]]}
    il = lines(d / "images.txt")
    for a, b in zip(il[0::2], il[1::2]):
        t = a.split()
        v = np.array(b.split(), dtype=np.float64).reshape(-1, 3) if b.strip() else np.zeros((0, 3))
        keep = v[:, 2] >= 0
        imgs[int(t[0])] = {"name": " ".join(t[9:]), "q": [float(x) for x in t[1:5]], "t": [float(x) for x in t[5:8]],
                           "camera_id": int(t[8]), "n2d": len(v), "n3d": int(keep.sum()),
                           "obs_xy": v[keep, :2], "obs_id": v[keep, 2].astype(np.int64)}
    errs, tracks, xyz, pids = [], [], [], []
    for l in lines(d / "points3D.txt"):
        t = l.split()
        pids.append(int(t[0])); xyz.append([float(v) for v in t[1:4]]); errs.append(float(t[7])); tracks.append((len(t) - 8) // 2)
    return cams, imgs, np.array(errs), np.array(tracks), np.array(xyz).reshape(-1, 3), np.array(pids, np.int64)


def project(cam, pc):
    """Project camera-frame points (N,3) to pixels for common COLMAP models; None if unsupported."""
    m, p = cam["model"], cam["params"]
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    if m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
        k1 = p[3] if len(p) > 3 else 0.0; k2 = p[4] if len(p) > 4 else 0.0
        r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
        xd, yd = x * rad, y * rad
    elif m in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[:4]
        k1, k2, p1, p2 = p[4:8] if m == "OPENCV" else (0.0, 0.0, 0.0, 0.0)
        r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    else:
        return None
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


def reprojection_errors(cams, imgs, xyz, pids):
    """Per-observation reprojection error (px), recomputed from poses; None if a camera model is unsupported."""
    lut = {int(p): i for i, p in enumerate(pids)}
    out = []
    for im in imgs.values():
        if not len(im["obs_id"]):
            continue
        idx = np.array([lut.get(int(i), -1) for i in im["obs_id"]])
        ok = idx >= 0
        pc = xyz[idx[ok]] @ qvec_to_R(im["q"]).T + np.array(im["t"])
        uv = project(cams[im["camera_id"]], pc)
        if uv is None:
            return None
        out.append(np.linalg.norm(uv - im["obs_xy"][ok], axis=1))
    return np.concatenate(out) if out else None


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def frame_index(name: str):
    m = re.findall(r"\d+", Path(name).stem)
    return int(m[-1]) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sparse_dir", type=Path)
    ap.add_argument("--images", type=Path, help="input image folder, to count unregistered frames")
    ap.add_argument("--num-input", type=int, help="number of input frames, if --images is not given")
    ap.add_argument("--video", action="store_true", help="looser reprojection gate (2.0px instead of 1.5px) for video frames")
    ap.add_argument("--holdout-every", type=int, default=8, help="held-out view stride for coverage check")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    d = a.sparse_dir
    cams, imgs, errs, tracks, xyz, pids = read_bin(d) if (d / "images.bin").exists() else read_txt(d)
    reproj_source = "points3D"
    if len(errs) and not errs.any():
        # Some writers (e.g. Spirula's SfM) leave every point's error at 0: recompute from the poses.
        obs_err = reprojection_errors(cams, imgs, xyz, pids)
        errs, reproj_source = (obs_err, "recomputed") if obs_err is not None else (np.zeros(0), "not recorded")

    input_names = None
    if a.images:
        input_names = sorted(str(p.relative_to(a.images)) for p in a.images.rglob("*") if p.suffix.lower() in IMG_EXTS)
    n_input = len(input_names) if input_names is not None else (a.num_input or len(imgs))
    reg_names = {v["name"] for v in imgs.values()}
    n_reg = len(imgs)

    centers = np.array([-qvec_to_R(v["q"]).T @ np.array(v["t"]) for v in imgs.values()]).reshape(-1, 3)
    obs = np.array([v["n3d"] for v in imgs.values()])
    cam_extent = float(np.linalg.norm(centers - np.median(centers, 0), axis=1).max()) if n_reg else 0.0
    pts_med_dist = float(np.median(np.linalg.norm(xyz - np.median(centers, 0), axis=1))) if len(xyz) else 0.0

    # Unregistered runs (video): consecutive input frames missing from the model = a tracking break.
    gaps, held_unreg = [], []
    if input_names is not None:
        missing = [i for i, n in enumerate(input_names) if n not in reg_names]
        run = []
        for i in missing:
            if run and i != run[-1] + 1:
                gaps.append((input_names[run[0]], len(run))); run = []
            run.append(i)
        if run:
            gaps.append((input_names[run[0]], len(run)))
        held_unreg = [input_names[i] for i in missing if i % a.holdout_every == 0]

    r = {
        "sparse_dir": str(d),
        "cameras": {k: {"model": c["model"], "size": [c["width"], c["height"]], "f": c["params"][:2]} for k, c in cams.items()},
        "n_input": n_input, "n_registered": n_reg, "registered_frac": n_reg / max(n_input, 1),
        "n_points": int(len(xyz)),
        "mean_reproj_err_px": float(errs.mean()) if len(errs) else None,
        "median_reproj_err_px": float(np.median(errs)) if len(errs) else None,
        "mean_track_len": float(tracks.mean()) if len(tracks) else None,
        "median_track_len": float(np.median(tracks)) if len(tracks) else None,
        "n_points_multiview": int((tracks >= 3).sum()) if len(tracks) else 0,
        "points_per_image_mean": float(obs.mean()) if n_reg else 0.0,
        "points_per_image_min": int(obs.min()) if n_reg else 0,
        "images_under_100_points": int((obs < 100).sum()),
        "camera_extent": cam_extent, "points_median_dist_from_cameras": pts_med_dist,
        "unregistered_runs": [{"first": f, "length": n} for f, n in gaps],
        "unregistered_holdouts": held_unreg,
    }
    r["reproj_source"] = reproj_source
    reproj_max = 2.0 if a.video else 1.5
    gates = {
        "registered_frac>=0.95": r["registered_frac"] >= 0.95,
        "points_multiview>=1000": r["n_points_multiview"] >= 1000,
        "points_per_image_mean>=300": r["points_per_image_mean"] >= 300,
        "images_under_100_points<=2%": r["images_under_100_points"] <= 0.02 * max(n_reg, 1),
        "no_unregistered_run>5": all(n <= 5 for _, n in gaps),
        "points_not_degenerate": len(xyz) >= 1000,
    }
    if r["mean_reproj_err_px"] is not None:
        gates[f"mean_reproj_err<={reproj_max}px"] = r["mean_reproj_err_px"] <= reproj_max
    r["gates"] = gates
    r["pass"] = all(gates.values())
    r["gate_policy"] = "sfm-gates-v2"  # v1 gated median_track_len>=3; v2 gates points_multiview>=1000

    print(f"{d}: {n_reg}/{n_input} registered ({100 * r['registered_frac']:.1f}%), {len(xyz)} points, "
          f"{len(cams)} camera(s) {sorted({c['model'] for c in cams.values()})}")
    if len(tracks):
        rep = (f"reproj err mean {r['mean_reproj_err_px']:.3f}px median {r['median_reproj_err_px']:.3f}px ({reproj_source})"
               if r["mean_reproj_err_px"] is not None else "reproj err not recorded")
        print(f"  {rep} | "
              f"track len mean {r['mean_track_len']:.2f} median {r['median_track_len']:.0f} | "
              f"pts/img mean {r['points_per_image_mean']:.0f} min {r['points_per_image_min']}")
    if gaps:
        print(f"  unregistered runs: {len(gaps)}, longest {max(n for _, n in gaps)} (from {max(gaps, key=lambda g: g[1])[0]})")
    for k, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    if a.json:
        a.json.write_text(json.dumps(r, indent=2))
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
