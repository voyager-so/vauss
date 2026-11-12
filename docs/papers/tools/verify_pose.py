#!/usr/bin/env python3
"""Publications-owned verifier for POSE (relative-pose AUC) and MESH (DTU Chamfer) claims.

Companion to verify.py (image metrics). Same rule: recompute every headline number from the raw
per-camera / per-scan outputs with publications' own code, never from a lane's summary, and fail
closed when coverage does not match. stdlib + numpy only; CPU; read-only on all inputs.

Subcommands
-----------
pose-rows   Per-camera pose rows (B17 export_pose_rows.py format): one `<run>.jsonl` per run with
            one row per EXPECTED camera {scene, view_id, repeat, arm, registered, predicted_w2c,
            reference_w2c, aligned_center_error_m} plus `<run>-summary.json` {n_expected,
            n_scored, n_registered, views_hash, scorer_sha, auc, ate_m}.
              verify_pose.py pose-rows --rows DIR [--manifest M.json] [--baseline ARM]
                  [--pair B=A ...] --json OUT
pose-scenes Per-scene AUC tables only (P-MAP heldout evidence); no per-camera rows exist, so this
            recomputes the scene/seed aggregation and cross-checks coverage, and says so.
              verify_pose.py pose-scenes --seed-dir S=DIR ... --json OUT
mesh        DTU per-scan mesh metrics.json (B15): recompute Chamfer = (accuracy+completeness)/2
            per scan and the unweighted scan means; check evaluator commit, protocol and GT point
            counts are identical within each scan across arms.
              verify_pose.py mesh --arm NAME=GLOB_WITH_{scan} ... --scans 24,37,... --json OUT

Pose metric (independent re-implementation of frozen posecmp_v2, sha256 f6f07624...): for every
unordered pair (a, b) of EXPECTED cameras (expected-list order, a before b) with a non-zero GT
relative baseline, the relative pose is R_ab = R_b R_a^T, t_ab = t_b - R_ab t_a on w2c matrices.
error = max(angle(R_ab_pred R_ab_gt^T), angle(t_ab_pred, t_ab_gt)) in degrees (180 when the
predicted baseline is zero); a pair with either camera unregistered has error = inf. AUC@T is the
exact step-CDF integral mean(max(0, 1 - error/T)) over ALL expected pairs, so missing cameras
contribute zero. Thresholds 3, 5, 10, 20, 30 are reported (v2 froze 3/5/30; v3 added 10/20).

Fail-closed gates (pose-rows): duplicate or missing view_ids, n_scored/n_expected/n_registered
or views_hash disagreeing with the rows, registered flag inconsistent with predicted_w2c,
invalid rotations, per-scene expected lists or GT poses differing between arms of the same
(scene, repeat), an arm missing a (scene, repeat) cell, and, when --manifest is given, rows whose
expected list or GT poses differ from the manifest's expected.json / gt/images.txt.
The lane's per-run AUC and ATE are compared and reported but never used.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

THRESHOLDS = (3, 5, 10, 20, 30)
PACIFIC = ZoneInfo("America/Los_Angeles")
EPS = 1e-9


class VerifyError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except OSError:
        return None


def sample_sd(values):
    v = np.asarray(values, float)
    return float(v.std(ddof=1)) if len(v) > 1 else None


def stat(values):
    return {"values": [float(x) for x in values], "mean": float(np.mean(values)),
            "sample_sd": sample_sd(values)}


# ---------------------------------------------------------------- pose metric

def check_pose(name, m):
    m = np.asarray(m, float)
    if m.shape != (3, 4) or not np.isfinite(m).all():
        raise VerifyError(f"invalid pose matrix for {name}")
    r = m[:, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
        raise VerifyError(f"invalid rotation for {name}")
    return m


def angle_deg(cosine):
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def pair_errors(names, gt, pred):
    """Errors for all unordered expected pairs (inf where a camera is unregistered)."""
    n = len(names)
    G = np.stack([gt[k] for k in names])
    have = np.array([k in pred for k in names])
    P = np.stack([pred[k] if k in pred else np.eye(3, 4) for k in names])
    ia, ib = np.triu_indices(n, k=1)
    Rg = G[ib, :, :3] @ np.transpose(G[ia, :, :3], (0, 2, 1))
    tg = G[ib, :, 3] - np.einsum("kij,kj->ki", Rg, G[ia, :, 3])
    Rp = P[ib, :, :3] @ np.transpose(P[ia, :, :3], (0, 2, 1))
    tp = P[ib, :, 3] - np.einsum("kij,kj->ki", Rp, P[ia, :, 3])
    keep = np.linalg.norm(tg, axis=1) > EPS
    dR = Rp @ np.transpose(Rg, (0, 2, 1))
    rot = angle_deg((np.trace(dR, axis1=1, axis2=2) - 1) / 2)
    ntp, ntg = np.linalg.norm(tp, axis=1), np.linalg.norm(tg, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tra = angle_deg(np.einsum("ki,ki->k", tp, tg) / (ntp * ntg))
    tra = np.where(ntp <= EPS, 180.0, tra)
    err = np.maximum(rot, tra)
    err = np.where(have[ia] & have[ib], err, np.inf)
    return err[keep], int((~keep).sum())


def auc(errors, thresholds=THRESHOLDS):
    e = np.asarray(errors, float)
    return {str(t): float(np.maximum(0.0, 1.0 - e / t).mean()) for t in thresholds}


def sim3_center_errors(names, gt, pred):
    """Umeyama Sim(3) of predicted camera centres onto GT centres over registered cameras."""
    common = [k for k in names if k in pred]
    if len(common) < 3:
        return {}, None
    c = lambda m: -m[:, :3].T @ m[:, 3]
    X = np.array([c(pred[k]) for k in common])
    Y = np.array([c(gt[k]) for k in common])
    mx, my = X.mean(0), Y.mean(0)
    x, y = X - mx, Y - my
    U, D, Vt = np.linalg.svd(y.T @ x / len(x))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt)) or 1.0])
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (x ** 2).sum(1).mean()
    fit = s * x @ R.T + my
    d = np.linalg.norm(fit - Y, axis=1)
    return dict(zip(common, d.tolist())), float(np.sqrt((d ** 2).mean()))


# ---------------------------------------------------------------- independent raw sources

def read_images_txt(path):
    """COLMAP text images.txt -> {name: 3x4 w2c}. Independent of pycolmap."""
    out = {}
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    # COLMAP writes two lines per image; the POINTS2D line may be empty, so pair by position.
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) % 2:
        lines.append("")
    for line in lines[0::2]:
        p = line.split()
        qw, qx, qy, qz = map(float, p[1:5])
        t = np.array(list(map(float, p[5:8])))
        q = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        name = p[9]
        if name in out:
            raise VerifyError(f"duplicate image {name} in {path}")
        out[name] = np.hstack([R, t[:, None]])
    return out


def read_images_bin(path):
    """COLMAP binary images.bin -> {name: 3x4 w2c}. Independent of pycolmap."""
    import struct
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    (n,), off = struct.unpack_from("<Q", data, 0), 8
    for _ in range(n):
        vals = struct.unpack_from("<I7dI", data, off)
        off += struct.calcsize("<I7dI")
        end = data.index(b"\x00", off)
        name = data[off:end].decode()
        off = end + 1
        (npts,) = struct.unpack_from("<Q", data, off)
        off += 8 + npts * 24
        qw, qx, qy, qz, tx, ty, tz = vals[1:8]
        q = np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        if name in out:
            raise VerifyError(f"duplicate image {name} in {path}")
        out[name] = np.hstack([R, np.array([[tx], [ty], [tz]])])
    if off != len(data):
        raise VerifyError(f"{path}: trailing bytes; not a classic images.bin layout")
    return out


def read_model(path):
    if os.path.isdir(path):
        b = os.path.join(path, "images.bin")
        return read_images_bin(b) if os.path.exists(b) else read_images_txt(os.path.join(path, "images.txt"))
    with np.load(path, allow_pickle=False) as d:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in d["names"]]
        return {k: np.asarray(m, float)[:3, :4] for k, m in zip(names, d["ext"])}


# ---------------------------------------------------------------- pose-rows

def load_run(jsonl):
    stem = jsonl[:-len(".jsonl")]
    objs = [json.loads(l) for l in open(jsonl) if l.strip()]
    rows = [o for o in objs if "view_id" in o]
    inline = [o for o in objs if "view_id" not in o]
    if len(inline) > 1:
        raise VerifyError(f"{jsonl}: {len(inline)} non-row objects (expected at most one inline summary)")
    sidecar = None
    for cand in (stem + "-summary.json", stem + ".summary.json", os.path.join(os.path.dirname(jsonl), "summary.json")):
        if os.path.exists(cand) and cand.endswith(("-summary.json", ".summary.json")):
            sidecar = json.load(open(cand))
            break
    if inline and sidecar is not None:
        for k in ("views_hash", "n_expected", "n_scored", "n_registered"):
            if k in inline[0] and k in sidecar and inline[0][k] != sidecar[k]:
                raise VerifyError(f"{jsonl}: inline summary {k}={inline[0][k]} != sidecar {sidecar[k]}")
    summary = inline[0] if inline else sidecar
    if summary is None:
        raise VerifyError(f"{jsonl}: no inline summary object and no sidecar summary")
    if not rows:
        raise VerifyError(f"{jsonl}: no per-camera rows")
    names = [r["view_id"] for r in rows]
    if len(set(names)) != len(names):
        raise VerifyError(f"{jsonl}: duplicate view_id")
    keys = {(r["scene"], r.get("repeat", r.get("seed")), r["arm"]) for r in rows}
    if len(keys) != 1:
        raise VerifyError(f"{jsonl}: rows span several (scene, repeat, arm): {sorted(keys)}")
    scene, repeat, arm = keys.pop()
    gt, pred = {}, {}
    for r in rows:
        gt[r["view_id"]] = check_pose(r["view_id"], r["reference_w2c"])
        if bool(r["registered"]) != (r["predicted_w2c"] is not None):
            raise VerifyError(f"{jsonl}: registered flag inconsistent for {r['view_id']}")
        if r["registered"]:
            pred[r["view_id"]] = check_pose(r["view_id"], r["predicted_w2c"])
    hash_json = hashlib.sha256(json.dumps(sorted(names), separators=(",", ":")).encode()).hexdigest()
    hash_nl = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    if summary.get("views_hash") not in (hash_json, hash_nl):
        raise VerifyError(f"{jsonl}: views_hash does not match rows")
    for k, v in (("n_expected", len(names)), ("n_scored", len(names)), ("n_registered", len(pred))):
        if summary.get(k) != v:
            raise VerifyError(f"{jsonl}: summary {k}={summary.get(k)} but rows give {v}")
    return dict(path=jsonl, summary=summary, scene=scene, repeat=repeat, arm=arm, names=names,
                gt=gt, pred=pred, rows=rows, rows_sha256=sha256_file(jsonl), views_hash=hash_json)


def cmd_pose_rows(a):
    files = sorted(glob.glob(os.path.join(a.rows, "*.jsonl")))
    if not files:
        raise VerifyError(f"no *.jsonl under {a.rows}")
    runs = [load_run(f) for f in files]
    manifest = {}
    if a.manifest:
        for it in json.load(open(a.manifest)):
            manifest[(it["scene"], it["repeat"], it["arm"])] = it
        if set(manifest) != {(r["scene"], r["repeat"], r["arm"]) for r in runs}:
            raise VerifyError("manifest (scene, repeat, arm) set differs from the exported rows")

    arms = sorted({r["arm"] for r in runs})
    cells = sorted({(r["scene"], r["repeat"]) for r in runs})
    by = {(r["scene"], r["repeat"], r["arm"]): r for r in runs}
    gates = []
    for cell in cells:
        present = [arm for arm in arms if (*cell, arm) in by]
        if len(present) != len(arms):
            raise VerifyError(f"coverage: cell {cell} lacks arms {sorted(set(arms) - set(present))}")
        ref = by[(*cell, arms[0])]
        for arm in arms[1:]:
            r = by[(*cell, arm)]
            if r["names"] != ref["names"]:
                raise VerifyError(f"coverage: expected list differs for {cell} {arm} vs {arms[0]}")
            for k in ref["names"]:
                if not np.allclose(r["gt"][k], ref["gt"][k], atol=1e-12):
                    raise VerifyError(f"GT pose differs for {cell} {k} between arms")
        gates.append(f"{cell[0]} r{cell[1]}: {len(arms)} arms share {len(ref['names'])} expected names "
                     f"and GT poses (views_hash {ref['views_hash'][:12]})")
    # scene expected list identical across repeats
    for scene in sorted({c[0] for c in cells}):
        hs = {by[(scene, rep, arms[0])]["views_hash"] for (s, rep) in cells if s == scene}
        if len(hs) != 1:
            raise VerifyError(f"coverage: {scene} expected list changes across repeats")

    manifest_checks = 0
    if manifest:
        gt_cache = {}
        for r in runs:
            it = manifest[(r["scene"], r["repeat"], r["arm"])]
            exp = json.load(open(it["expected"]))
            if exp != r["names"]:
                raise VerifyError(f"{r['path']}: rows' view order/list differs from {it['expected']}")
            gtp = os.path.join(it["gt"], "images.txt")
            if gtp not in gt_cache:
                gt_cache[gtp] = read_images_txt(gtp)
            for k in r["names"]:
                g = gt_cache[gtp].get(k)
                if g is None or not np.allclose(g, r["gt"][k], atol=1e-6):
                    raise VerifyError(f"{r['path']}: reference_w2c for {k} differs from {gtp}")
            model = read_model(it["model"])
            if set(model) - set(r["names"]):
                raise VerifyError(f"{it['model']}: registers images outside the expected list")
            if set(model) != set(r["pred"]):
                raise VerifyError(f"{r['path']}: registered set differs from raw model {it['model']}")
            for k, m in model.items():
                if not np.allclose(m, r["pred"][k], atol=1e-6):
                    raise VerifyError(f"{r['path']}: predicted_w2c for {k} differs from {it['model']}")
            manifest_checks += 1

    per_run = []
    max_lane_diff = {str(t): 0.0 for t in THRESHOLDS}
    max_ate_diff, max_centre_diff = 0.0, 0.0
    for r in runs:
        errs, skipped = pair_errors(r["names"], r["gt"], r["pred"])
        n = len(r["names"])
        if len(errs) + skipped != n * (n - 1) // 2:
            raise VerifyError("pair count mismatch")
        A = auc(errs)
        centres, ate = sim3_center_errors(r["names"], r["gt"], r["pred"])
        for row in r["rows"]:
            lane = row.get("aligned_center_error_m")
            mine = centres.get(row["view_id"])
            if (lane is None) != (mine is None):
                raise VerifyError(f"{r['path']}: aligned centre error presence mismatch {row['view_id']}")
            if lane is not None:
                max_centre_diff = max(max_centre_diff, abs(lane - mine))
        lane_auc = r["summary"].get("auc", {})
        for t in THRESHOLDS:
            if str(t) in lane_auc:
                max_lane_diff[str(t)] = max(max_lane_diff[str(t)], abs(lane_auc[str(t)] - A[str(t)]))
        if ate is not None and r["summary"].get("ate_m") is not None:
            max_ate_diff = max(max_ate_diff, abs(ate - r["summary"]["ate_m"]))
        if manifest and "expected_auc3" in manifest[(r["scene"], r["repeat"], r["arm"])]:
            max_lane_diff["3"] = max(max_lane_diff["3"], abs(manifest[(r["scene"], r["repeat"], r["arm"])]["expected_auc3"] - A["3"]))
        per_run.append(dict(scene=r["scene"], repeat=r["repeat"], arm=r["arm"], n_expected=n,
                            n_registered=len(r["pred"]), n_pairs=int(len(errs)),
                            zero_baseline_pairs_excluded=skipped, auc=A, ate_m=ate,
                            lane_auc=lane_auc, rows_sha256=r["rows_sha256"],
                            views_hash=r["views_hash"], scorer_sha=r["summary"].get("scorer_sha")))

    scenes = sorted({c[0] for c in cells})
    repeats = sorted({c[1] for c in cells})
    look = {(p["scene"], p["repeat"], p["arm"]): p for p in per_run}
    arms_out = {}
    for arm in arms:
        out = {}
        for t in THRESHOLDS:
            rep_means = [float(np.mean([look[(s, rep, arm)]["auc"][str(t)] for s in scenes])) for rep in repeats]
            out[str(t)] = stat(rep_means)
        out["per_scene_auc3"] = {s: stat([look[(s, rep, arm)]["auc"]["3"] for rep in repeats]) for s in scenes}
        out["registered"] = {s: [look[(s, rep, arm)]["n_registered"] for rep in repeats] for s in scenes}
        arms_out[arm] = out
    pairs_out = {}
    for spec in a.pair or []:
        b, base = spec.split("=")
        out = {}
        for t in THRESHOLDS:
            d = [float(np.mean([look[(s, rep, b)]["auc"][str(t)] - look[(s, rep, base)]["auc"][str(t)]
                                for s in scenes])) for rep in repeats]
            out[str(t)] = stat(d)
        out["scene_wins_auc3"] = int(sum(
            np.mean([look[(s, rep, b)]["auc"]["3"] for rep in repeats]) >
            np.mean([look[(s, rep, base)]["auc"]["3"] for rep in repeats]) for s in scenes))
        pairs_out[f"{b} - {base}"] = out
    scorer = sorted({p["scorer_sha"] for p in per_run if p["scorer_sha"]})
    return dict(kind="pose-rows", rows_dir=os.path.abspath(a.rows),
                manifest=os.path.abspath(a.manifest) if a.manifest else None,
                manifest_sha256=sha256_file(a.manifest) if a.manifest else None,
                scenes=scenes, repeats=repeats, arms=arms, n_runs=len(runs),
                lane_scorer_sha256=scorer,
                statistic="per repeat: unweighted mean over scenes; then mean and sample SD over repeats",
                gates=gates + [f"manifest cross-check (expected.json order + gt/images.txt poses) passed for {manifest_checks} runs"],
                arms_summary=arms_out, paired=pairs_out,
                max_abs_diff_vs_lane_run_auc=max_lane_diff, max_abs_diff_vs_lane_ate_m=max_ate_diff,
                max_abs_diff_vs_lane_aligned_center_error_m=max_centre_diff,
                per_run=per_run)


# ---------------------------------------------------------------- pose-scenes (no per-camera rows)

def cmd_pose_scenes(a):
    seeds = {}
    for spec in a.seed_dir:
        seed, d = spec.split("=", 1)
        seeds[seed] = d
    baseline, completed, gates, inputs = {}, {}, [], {}
    scene_sets = {}
    for seed, d in seeds.items():
        man = json.load(open(os.path.join(d, a.manifest_name.format(seed=seed))))
        met = json.load(open(os.path.join(d, a.metrics_name.format(seed=seed))))
        com = json.load(open(os.path.join(d, a.completion_name.format(seed=seed))))
        for f in (a.manifest_name, a.metrics_name, a.completion_name):
            p = os.path.join(d, f.format(seed=seed))
            inputs[p] = sha256_file(p)
        scenes = list(man["scenes"])
        if len(set(scenes)) != len(scenes):
            raise VerifyError(f"seed {seed}: duplicate scenes in manifest")
        cscenes = [r["scene"] for r in com]
        if sorted(cscenes) != sorted(scenes) or len(set(cscenes)) != len(cscenes):
            raise VerifyError(f"seed {seed}: completion scenes {sorted(cscenes)} != manifest {sorted(scenes)}")
        base_rows = {r["scene"]: r for r in met if r["arm"] == a.baseline_arm}
        if sorted(base_rows) != sorted(scenes):
            raise VerifyError(f"seed {seed}: baseline arm rows do not cover the manifest scenes")
        for r in com:
            s = r["scene"]
            n_exp = len(man["inputs"][s]["expected"])
            m = base_rows[s]
            if m["expected"] != n_exp or m["score"]["n_expected"] != n_exp:
                raise VerifyError(f"seed {seed} {s}: expected count mismatch")
            npairs = n_exp * (n_exp - 1) // 2 - m["score"]["reference_zero_baseline_pairs_excluded"]
            if m["score"]["n_expected_pairs"] != npairs:
                raise VerifyError(f"seed {seed} {s}: AUC denominator is not all expected pairs")
            if r["registered_before"] != m["registered"]:
                raise VerifyError(f"seed {seed} {s}: completion registered_before != metrics registered")
            if r["registered_after"] > n_exp or r["registered_after"] < r["registered_before"]:
                raise VerifyError(f"seed {seed} {s}: impossible registered_after")
            for t in THRESHOLDS:
                if abs(r["baseline_auc"][str(t)] - m["auc"][str(t)]) > 1e-12:
                    raise VerifyError(f"seed {seed} {s}: completion baseline_auc != metrics auc @{t}")
                if abs(m["score"]["all_expected_reference_pairs"].get(str(t), {"step_cdf_auc": m["auc"][str(t)]})["step_cdf_auc"] - m["auc"][str(t)]) > 1e-12:
                    raise VerifyError(f"seed {seed} {s}: auc != score.all_expected_reference_pairs @{t}")
            if r["registered_after"] == r["registered_before"] and any(
                    abs(r["completed_auc"][str(t)] - r["baseline_auc"][str(t)]) > 1e-12 for t in THRESHOLDS):
                raise VerifyError(f"seed {seed} {s}: no camera added but AUC changed")
            baseline[(seed, s)] = r["baseline_auc"]
            completed[(seed, s)] = r["completed_auc"]
        scene_sets[seed] = sorted(scenes)
        gates.append(f"seed {seed}: {len(scenes)} scenes; completion/metrics/manifest scene sets, expected counts, "
                     f"all-pairs denominators and baseline AUC agree")
    if len({tuple(v) for v in scene_sets.values()}) != 1:
        raise VerifyError("scene set differs across seeds")
    scenes = next(iter(scene_sets.values()))
    out = {}
    for t in THRESHOLDS:
        b = [float(np.mean([baseline[(sd, s)][str(t)] for s in scenes])) for sd in seeds]
        c = [float(np.mean([completed[(sd, s)][str(t)] for s in scenes])) for sd in seeds]
        out[str(t)] = dict(baseline=stat(b), completion=stat(c),
                           paired_delta=stat([y - x for x, y in zip(b, c)]))
    per_scene = {s: dict(baseline_auc3=[baseline[(sd, s)]["3"] for sd in seeds],
                         completion_auc3=[completed[(sd, s)]["3"] for sd in seeds]) for s in scenes}
    lane = None
    if a.lane_summary:
        lane = json.load(open(a.lane_summary)).get("heldout")
        inputs[a.lane_summary] = sha256_file(a.lane_summary)
    diffs = {}
    if lane:
        for t in THRESHOLDS:
            if str(t) in lane:
                diffs[str(t)] = {k: abs(lane[str(t)][k]["mean"] - out[str(t)][k]["mean"])
                                 for k in ("baseline", "completion", "paired_delta")}
    return dict(kind="pose-scenes", per_camera_rows=False,
                limitation=("No per-camera pose rows or pose models survive for this claim: the COLMAP "
                            "sparse models and completion.npz artifacts referenced by the evidence are "
                            "absent from disk. Only the per-scene AUC aggregation, coverage and "
                            "cross-file consistency are recomputed; per-scene AUCs are taken as reported."),
                seeds=list(seeds), scenes=scenes, baseline_arm=a.baseline_arm,
                statistic="per seed: unweighted mean over scenes; then mean and sample SD over seeds",
                gates=gates, results=out, per_scene=per_scene,
                max_abs_diff_vs_lane_summary=diffs, inputs_sha256=inputs)


# ---------------------------------------------------------------- mesh

def cmd_mesh(a):
    scans = [int(s) for s in a.scans.split(",")]
    arms = []
    for spec in a.arm:
        name, pat = spec.split("=", 1)
        arms.append((name, pat))
    rows, gates, inputs = {}, [], {}
    for scan in scans:
        rec = {}
        for name, pat in arms:
            hits = sorted(glob.glob(pat.format(scan=scan)))
            if len(hits) != 1:
                raise VerifyError(f"scan {scan} arm {name}: {len(hits)} files match {pat}")
            m = json.load(open(hits[0]))
            inputs[hits[0]] = sha256_file(hits[0])
            if m.get("scan") != scan or m.get("mode") != "mesh":
                raise VerifyError(f"{hits[0]}: scan/mode mismatch")
            acc, comp = float(m["accuracy_mm"]), float(m["completeness_mm"])
            ch = (acc + comp) / 2
            ev = os.path.join(os.path.dirname(hits[0]), "evaluation", "metrics.json")
            if os.path.exists(ev):
                e = json.load(open(ev))
                inputs[ev] = sha256_file(ev)
                for k in ("accuracy_mm", "completeness_mm", "chamfer_mm", "mesh_sha256",
                          "completeness_points", "accuracy_points", "evaluator_commit"):
                    if e.get(k) != m.get(k):
                        raise VerifyError(f"{hits[0]}: {k} differs from evaluator output {ev}")
            rec[name] = dict(path=hits[0], accuracy_mm=acc, completeness_mm=comp,
                             chamfer_recomputed_mm=ch, chamfer_lane_mm=m.get("chamfer_mm"),
                             evaluator_commit=m.get("evaluator_commit"), protocol=m.get("protocol"),
                             completeness_points=m.get("completeness_points"),
                             accuracy_points=m.get("accuracy_points"), mesh_sha256=m.get("mesh_sha256"),
                             seed=m.get("seed"), arm_label=m.get("arm"))
        for k in ("evaluator_commit", "protocol", "completeness_points"):
            vals = {json.dumps(r[k], sort_keys=True) for r in rec.values()}
            if len(vals) != 1:
                raise VerifyError(f"scan {scan}: {k} differs across arms: {vals}")
        g = next(iter(rec.values()))
        gates.append(f"scan {scan}: {len(rec)} arms share evaluator {g['evaluator_commit'][:8]}, "
                     f"protocol and GT point count {g['completeness_points']}")
        rows[scan] = rec
    means = {name: dict(chamfer_mm=float(np.mean([rows[s][name]["chamfer_recomputed_mm"] for s in scans])),
                        accuracy_mm=float(np.mean([rows[s][name]["accuracy_mm"] for s in scans])),
                        completeness_mm=float(np.mean([rows[s][name]["completeness_mm"] for s in scans])))
             for name, _ in arms}
    max_scan_diff = max(abs(r["chamfer_recomputed_mm"] - r["chamfer_lane_mm"])
                        for rec in rows.values() for r in rec.values())
    lane_means, diffs = None, {}
    if a.lane_summary:
        L = json.load(open(a.lane_summary))
        inputs[a.lane_summary] = sha256_file(a.lane_summary)
        lane_means = L.get("means")
        for spec in a.lane_key or []:
            name, key = spec.split("=")
            diffs[name] = dict(lane=lane_means[key], recomputed=means[name]["chamfer_mm"],
                               abs_diff=abs(lane_means[key] - means[name]["chamfer_mm"]))
        lane_rows = {r["scan"]: r for r in L["rows"]}
        if sorted(lane_rows) != scans:
            raise VerifyError("lane summary scan list differs from --scans")
        for spec in a.lane_key or []:
            name, key = spec.split("=")
            diffs[name]["max_abs_diff_per_scan"] = max(
                abs(lane_rows[s][key] - rows[s][name]["chamfer_recomputed_mm"]) for s in scans)
            for s in scans:
                emb = lane_rows[s].get(name)
                if isinstance(emb, dict) and emb.get("mesh_sha256") != rows[s][name]["mesh_sha256"]:
                    raise VerifyError(f"scan {s} {name}: lane summary mesh_sha256 differs from metrics.json")
    return dict(kind="mesh", scans=scans, arms=[n for n, _ in arms],
                statistic="Chamfer per scan = (accuracy + completeness)/2; unweighted mean over scans",
                per_scan_point_distances=False,
                limitation=("metrics.json carries per-scan accuracy/completeness means, not per-point "
                            "distances; the meshes and GT point clouds are not re-evaluated here."),
                gates=gates, means=means, vs_lane=diffs,
                max_abs_diff_scan_chamfer_vs_lane_mm=max_scan_diff,
                per_scan={str(s): rows[s] for s in scans}, inputs_sha256=inputs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pose-rows")
    q.add_argument("--rows", required=True)
    q.add_argument("--manifest")
    q.add_argument("--pair", action="append", help="B=A paired delta (B - A)")
    q.add_argument("--json")
    q = sub.add_parser("pose-scenes")
    q.add_argument("--seed-dir", action="append", required=True, help="SEED=DIR")
    q.add_argument("--manifest-name", default="heldout-s{seed}-manifest.json")
    q.add_argument("--metrics-name", default="heldout-s{seed}-metrics.json")
    q.add_argument("--completion-name", default="heldout-s{seed}-completion.json")
    q.add_argument("--baseline-arm", default="incremental")
    q.add_argument("--lane-summary")
    q.add_argument("--json")
    q = sub.add_parser("mesh")
    q.add_argument("--arm", action="append", required=True, help="NAME=PATH_PATTERN with {scan}")
    q.add_argument("--scans", required=True)
    q.add_argument("--lane-summary")
    q.add_argument("--lane-key", action="append", help="ARM=KEY in lane summary 'means'")
    q.add_argument("--json")
    a = p.parse_args(argv)
    try:
        res = {"pose-rows": cmd_pose_rows, "pose-scenes": cmd_pose_scenes, "mesh": cmd_mesh}[a.cmd](a)
    except VerifyError as e:
        print(f"FAIL (closed): {e}", file=sys.stderr)
        return 2
    res["verifier"] = dict(path="docs/papers/tools/verify_pose.py", sha256=sha256_file(__file__),
                           repo_head=git_head(), argv=sys.argv[1:],
                           run_at=datetime.now(PACIFIC).isoformat(timespec="seconds"))
    text = json.dumps(res, indent=1, allow_nan=False, default=float) + "\n"
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        open(a.json, "w").write(text)
    brief = {k: v for k, v in res.items() if k not in ("per_run", "per_scan", "inputs_sha256", "per_scene")}
    print(json.dumps(brief, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
