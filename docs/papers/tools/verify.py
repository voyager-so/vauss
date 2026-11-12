#!/usr/bin/env python3
"""Publications-owned verifier: recompute paired deltas from raw per-view rows.

Owned by the publications lead (agentic-research-swarms recommendations #1, #4, #5, #6).
It reads per-view metric rows straight from run artifacts with its own code, never a lane's
summary, and fails closed when the arms were not scored on the same views.

Usage
-----
  verify.py --arm NAME=SPEC --arm NAME=SPEC [--arm NAME=SPEC ...]
            [--metric psnr ...] [--common-views] [--expected-views FILE]
            [--raw-names] [--reps 10000] [--rng-seed 0] [--json OUT.json]

The FIRST arm named is the baseline A; every later arm B is reported as B - A. A single arm
gives absolute per-scene means and their mean over scenes, with the same coverage gate.
Each scene also gets a W/=/L call: W only when every B run beats every A run on that metric
(perf's scoreboard range rule), L for the reverse, = otherwise; the tally counts calls.
Repeating a NAME unions several specs into one arm.

SPEC = GLOB[::KEY=VALUE[,KEY=VALUE...]]
  GLOB   a path or glob (recursive ** allowed), relative to the cwd.
  KEY    a dotted field of the run record (scene, seed, technique, engine.kind,
         engine.label, pair_id, ts, role, ...); VALUE is an fnmatch pattern on str(value).
         `a|b|c` matches any alternative. KEY!=VALUE excludes. The reserved key `rows` picks the per-view list by dotted path
         (e.g. rows=sharp_reference.como_secondary.rows for the B1a CoMo PNG column).

Per-view formats read (auto-detected per file):
  * per-view row schema v1 (docs/papers/claims/README.md): one JSONL per run, rows with
    view_id/scene/split/seed/arm/metrics plus one summary object (in the file or a sibling
    summary.json); views_hash and n_scored in
    the summary are recomputed and must match the rows, or the run is refused.
  * splat-perf Row (schema splat-perf/row/v1): `per_image[]` with `image` + metrics. One row
    per JSONL line (runs/perf/results.jsonl) or one row per .json file (B13 confirmation).
    Seed = `seed`, else `replicate_label`, else `run_id`/`exp_id`, else `--seed` in engine.args, else
    `pair_id`, `fal.request_id`, `ts` (unseeded rival runs).
  * bare per-view list (`[{"name"|"image": ..., "psnr": ...}]`, X5 per_image.json and
    kitchen-v1/*-views.json); scene/seed/technique/n_test come from a sibling `result.json`
    or `<stem minus -views>.json`.
  * B1a gsplat result.json: `sharp_reference.rows[]` (image, psnr, ssim, lpips_alex).
  * dicts of such rows keyed by experiment id.
Artifacts that carry only means (X5 comparison.json, B13 results.jsonl, B13 replication
metrics.json, B1a A1-summary.json, ...) are rejected with an explicit "no per-view rows" error.

Statistics: per (scene, run) the metric is the mean over that run's analysed views; per scene
the arm value is the mean over its runs and delta = B - A (equal to the mean paired per-seed
delta when both arms have the same seeds). The headline is the unweighted mean across scenes,
with a scene-clustered bootstrap 95% CI (resample scenes). With a single scene the CI falls
back to resampling runs (paired by seed when the seeds match) and says so.

Hashes: a run's `views_hash` is sha256 of its newline-joined sorted raw view ids (as in schema
v1); `common_view_hash` is the same recipe on the normalised names actually analysed.
View names are normalised (basename, extension and a leading `eval_` stripped) unless
--raw-names. Duplicate views inside one run and duplicate (scene, run) keys inside one arm fail.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

METRICS = ("psnr", "ssim", "lpips_vgg", "lpips_alex")
ALIASES = {"lpips_vgg": ("lpips_vgg", "lpips_vgg_norm"), "psnr": ("psnr",), "ssim": ("ssim",),
           "lpips_alex": ("lpips_alex",)}
LOWER_IS_BETTER = {"lpips_vgg", "lpips_alex"}
PACIFIC = ZoneInfo("America/Los_Angeles")


class VerifyError(Exception):
    pass


# ---------------------------------------------------------------- loading

def norm_view(name: str, raw: bool) -> str:
    if raw:
        return name
    base = os.path.basename(str(name))
    base = re.sub(r"\.(png|jpe?g|exr|tiff?)$", "", base, flags=re.I)
    return base[5:] if base.startswith("eval_") else base


def dig(obj, dotted):
    for part in dotted.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def is_view_list(v) -> bool:
    return (isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
            and any(k in v[0] for k in ("image", "name", "view", "camera"))
            and any(k in v[0] for a in ALIASES.values() for k in a))


def view_metrics(row: dict) -> dict:
    out = {}
    for m, keys in ALIASES.items():
        for k in keys:
            if isinstance(row.get(k), (int, float)):
                out[m] = float(row[k])
                break
    return out


def perf_seed(rec: dict):
    for k in ("seed", "replicate_label"):
        if rec.get(k) is not None:
            return rec[k] if k == "seed" else f"rep{rec[k]}"
    for k in ("run_id", "exp_id"):
        if rec.get(k):
            return rec[k]
    args = dig(rec, "engine.args") or []
    for i, a in enumerate(args):
        if a == "--seed" and i + 1 < len(args):
            return int(args[i + 1]) if args[i + 1].isdigit() else args[i + 1]
        if isinstance(a, str) and a.startswith("--seed="):
            return a.split("=", 1)[1]
    for k in ("pair_id", "fal.request_id", "ts"):
        if dig(rec, k):
            return dig(rec, k)
    return None


def sibling_meta(path: str) -> dict:
    d, stem = os.path.dirname(path), os.path.basename(path)
    cands = []
    if stem.endswith("-views.json"):
        cands.append(os.path.join(d, stem[: -len("-views.json")] + ".json"))
    cands.append(os.path.join(d, "result.json"))
    for c in cands:
        if os.path.exists(c) and os.path.abspath(c) != os.path.abspath(path):
            with open(c) as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                meta = {k: v for k, v in meta.items() if k != "environment"}
                meta["_meta_file"] = c
                return meta
    return {}


def make_run(rec, views_list, path, fmt, raw):
    views, raw_names, dups = {}, [], []
    for v in views_list:
        name = v.get("view_id") or v.get("image") or v.get("name") or v.get("view") or v.get("camera")
        key = norm_view(name, raw)
        if key in views:
            dups.append(key)
        views[key] = view_metrics(v)
        raw_names.append(str(name))
    if dups:
        raise VerifyError(f"{path}: duplicate view(s) {sorted(set(dups))} in one run "
                          f"(scene {rec.get('scene')}, seed {perf_seed(rec)}); refusing to score")
    n_exp = None
    for k in ("n_expected", "n_test"):
        if isinstance(rec.get(k), int):
            n_exp = rec[k]
            break
    return {"scene": rec.get("scene") or os.path.basename(os.path.dirname(path)),
            "run": perf_seed(rec), "views": views, "raw_hash": sha_views(raw_names),
            "n_expected": n_exp, "source": path,
            "format": fmt, "meta": rec}


def records_from_file(path: str, rows_path: str | None, raw: bool, nopv: list):
    """Yield candidate runs (unfiltered). Means-only records go to `nopv`."""
    out = []
    if path.endswith(".jsonl"):
        with open(path) as f:
            recs = [(i, json.loads(line)) for i, line in enumerate(f, 1) if line.strip()]
        if any(isinstance(r, dict) and "view_id" in r for _, r in recs):
            return per_view_v1(path, [r for _, r in recs], raw)
        for i, rec in recs:
            out += records_from_obj(rec, f"{path}:{i}", rows_path, raw, nopv)
        return out
    with open(path) as f:
        obj = json.load(f)
    if is_view_list(obj):
        meta = sibling_meta(path)
        if not meta:
            meta = {"scene": os.path.basename(os.path.dirname(path))}
        meta.setdefault("technique", os.path.basename(os.path.dirname(path)))
        return [make_run(meta, obj, path, "per-view list + sibling meta", raw)]
    return records_from_obj(obj, path, rows_path, raw, nopv)


def per_view_v1(path, recs, raw):
    """Per-view row schema v1 (docs/papers/claims/README.md): one JSONL per run, rows with
    view_id plus one summary object. Fails closed on any inconsistency with the summary."""
    rows = [r for r in recs if "view_id" in r]
    summaries = [r for r in recs if "view_id" not in r]
    side = os.path.join(os.path.dirname(path), "summary.json")
    if not summaries and os.path.exists(side):
        with open(side) as f:
            summaries = [json.load(f)]
    if len(summaries) != 1:
        raise VerifyError(f"{path}: per-view v1 needs exactly one summary object (in the file or a "
                          f"sibling summary.json), found {len(summaries)}")
    summ = summaries[0]
    other = [r for r in rows if r.get("split", "test") != "test"]
    rows = [r for r in rows if r.get("split", "test") == "test"]
    groups = {(r.get("scene"), r.get("seed"), r.get("arm")) for r in rows}
    if len(groups) != 1:
        raise VerifyError(f"{path}: per-view v1 file holds {len(groups)} (scene, seed, arm) groups; one run per file")
    scene, seed, arm = groups.pop()
    ids = [r["view_id"] for r in rows]
    got = sha_views(ids)
    errs = []
    if summ.get("views_hash") and summ["views_hash"] != got:
        errs.append(f"views_hash {summ['views_hash']} != recomputed {got}")
    if summ.get("n_scored") is not None and summ["n_scored"] != len(ids):
        errs.append(f"n_scored {summ['n_scored']} != {len(ids)} test rows")
    if errs:
        raise VerifyError(f"{path}: summary disagrees with rows: " + "; ".join(errs))
    meta = dict(summ, scene=scene, seed=seed, arm=arm, split="test")
    fmt = "per-view v1" + (f" ({len(other)} non-test rows ignored)" if other else "")
    return [make_run(meta, rows, path, fmt, raw)]


def records_from_obj(rec, where, rows_path, raw, nopv):
    if not isinstance(rec, dict):
        nopv.append((where, rec, "not a JSON object"))
        return []
    if rows_path:
        v = dig(rec, rows_path)
        if is_view_list(v):
            parent = dig(rec, rows_path.rsplit(".", 1)[0]) if "." in rows_path else None
            if isinstance(parent, dict) and not any(isinstance(rec.get(k), int) for k in ("n_expected", "n_test")):
                for k in ("n_expected", "n_test"):
                    if isinstance(parent.get(k), int):
                        rec = dict(rec, n_expected=parent[k], n_expected_source=f"{rows_path.rsplit('.', 1)[0]}.{k}")
                        break
            return [make_run(rec, v, where, f"rows={rows_path}", raw)]
    elif is_view_list(rec.get("per_image")):
        return [make_run(rec, rec["per_image"], where, "splat-perf Row per_image", raw)]
    elif is_view_list(dig(rec, "sharp_reference.rows")):
        return [make_run(rec, rec["sharp_reference"]["rows"], where,
                         "B1a result.json sharp_reference.rows", raw)]
    # dict keyed by experiment id whose values are rows
    subs = [(k, v) for k, v in rec.items() if isinstance(v, dict) and k.startswith("exp_")]
    if subs and len(subs) == len(rec):
        out = []
        for k, v in subs:
            out += records_from_obj(v, f"{where}#{k}", rows_path, raw, nopv)
        return out
    why = "means only, no per-view rows"
    if "arms" in rec and "delta_psnr" in rec:
        why += " (X5 comparison.json: per-view rows live in <run>/<technique>/per_image.json)"
    nopv.append((where, rec, why))
    return []


def match(rec: dict, filters) -> bool:
    for key, neg, pat in filters:
        val = dig(rec, key)
        ok = val is not None and any(fnmatch.fnmatchcase(str(val), p) for p in pat.split("|"))
        if ok == neg:
            return False
    return True


def parse_spec(spec: str):
    if "=" not in spec:
        raise VerifyError(f"--arm {spec!r}: expected NAME=SPEC")
    name, rest = spec.split("=", 1)
    pattern, _, flt = rest.partition("::")
    filters, rows_path = [], None
    for item in filter(None, flt.split(",")):
        neg = "!=" in item
        k, v = item.split("!=" if neg else "=", 1)
        if k == "rows":
            rows_path = v
        else:
            filters.append((k, neg, v))
    return name, pattern, filters, rows_path


def load_arm(name, specs, raw):
    runs, files, nopv_all = [], [], []
    for pattern, filters, rows_path in specs:
        paths = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
        if not paths:
            raise VerifyError(f"arm {name}: no files match {pattern!r}")
        for p in paths:
            files.append(p)
            nopv = []
            cand = records_from_file(p, rows_path, raw, nopv)
            runs += [r for r in cand if match(r["meta"], filters)]
            nopv_all += [(w, why) for (w, rec, why) in nopv if match(rec, filters)]
    if not runs:
        msg = f"arm {name}: no per-view rows after filters"
        if nopv_all:
            msg += f"; {len(nopv_all)} matching record(s) carry {nopv_all[0][1]} (e.g. {nopv_all[0][0]})"
        raise VerifyError(msg)
    seen = {}
    for r in runs:
        key = (r["scene"], str(r["run"]))
        if key in seen:
            raise VerifyError(f"arm {name}: two runs for scene {key[0]} seed/run {key[1]}: "
                              f"{seen[key]} and {r['source']} (add a filter)")
        seen[key] = r["source"]
    return runs, sorted(set(files)), nopv_all


# ---------------------------------------------------------------- stats

def sha_views(names) -> str:
    """views_hash of per-view schema v1: sha256 of the newline-joined sorted view ids."""
    return hashlib.sha256("\n".join(sorted(str(n) for n in names)).encode()).hexdigest()


def view_hash(scene, views) -> str:
    """Hash of a normalised view set (same recipe as sha_views, on normalised names)."""
    return sha_views(views)


def engine_sha(rec):
    """sha256 of the engine binary a run record ran: `engine_sha256`, else the content-addressed dir of
    `engine.binary` (fal runners: .../bin/<sha256>/<name>). None when the record has neither."""
    sha = rec.get("engine_sha256") if isinstance(rec, dict) else None
    if sha:
        return sha
    binary = dig(rec, "engine.binary") if isinstance(rec, dict) else None
    for part in str(binary or "").split("/"):
        if len(part) == 64 and all(c in "0123456789abcdef" for c in part):
            return part
    return None


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_expected(path, raw):
    with open(path) as f:
        text = f.read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return {s: {norm_view(v, raw) for v in vs} for s, vs in obj.items()}
    if isinstance(obj, list):
        return {"*": {norm_view(v, raw) for v in obj}}
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        scene, view = (parts[0], parts[1]) if len(parts) > 1 else ("*", parts[0])
        out.setdefault(scene, set()).add(norm_view(view, raw))
    return out


def bootstrap_mean(values, reps, rng):
    values = np.asarray(values, float)
    if len(values) < 2:
        return None
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    m = values[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def bootstrap_two(a, b, reps, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) + len(b) < 3:
        return None
    ma = a[rng.integers(0, len(a), size=(reps, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), size=(reps, len(b)))].mean(axis=1)
    d = mb - ma
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def analyse(arms, args):
    names = list(arms)
    base = names[0]
    fails, notes = [], []
    scene_sets = {n: {r["scene"] for r in arms[n]["runs"]} for n in names}
    scenes = sorted(scene_sets[base])
    for n in names[1:]:
        if scene_sets[n] != scene_sets[base]:
            fails.append(f"scene sets differ: {base} {sorted(scene_sets[base])} vs {n} {sorted(scene_sets[n])}")
    if fails:
        return None, fails, notes
    expected = load_expected(args.expected_views, args.raw_names) if args.expected_views else None

    per_scene, skipped = {}, {}
    for s in scenes:
        runs = {n: sorted([r for r in arms[n]["runs"] if r["scene"] == s], key=lambda r: str(r["run"]))
                for n in names}
        all_runs = [r for n in names for r in runs[n]]
        sets = [set(r["views"]) for r in all_runs]
        n_exp = {r["n_expected"] for r in all_runs if r["n_expected"] is not None}
        exp_set = None
        if expected is not None:
            exp_set = expected.get(s, expected.get("*"))
            if exp_set is None:
                fails.append(f"{s}: --expected-views has no entry for this scene")
                continue
        n_expected = len(exp_set) if exp_set is not None else (max(n_exp) if n_exp else None)
        if len(n_exp) > 1:
            notes.append(f"{s}: runs disagree on n_expected {sorted(n_exp)}; using max")
        common = set.intersection(*sets)
        identical = all(x == sets[0] for x in sets)
        problems = []
        if not identical:
            problems.append("view sets differ across runs/arms")
        if exp_set is not None:
            extra = set.union(*sets) - exp_set
            if extra:
                fails.append(f"{s}: views outside the expected set: {sorted(extra)[:8]}")
            if not exp_set <= common:
                problems.append(f"{len(exp_set - common)} expected view(s) not scored by every run")
            common &= exp_set
        elif n_expected is not None and any(len(x) < n_expected for x in sets):
            problems.append(f"some runs scored fewer than n_expected={n_expected} views")
        if problems and not args.common_views:
            detail = "; ".join(f"{n}/{r['run']}: n={len(r['views'])} hash={r['raw_hash'][:16]}"
                               for n in names for r in runs[n])
            fails.append(f"{s}: {'; '.join(problems)} [{detail}] (use --common-views to restrict to the intersection)")
            continue
        if not common:
            fails.append(f"{s}: empty common view set")
            continue
        metrics = [m for m in args.metric
                   if all(m in r["views"][v] for r in all_runs for v in common)]
        for m in args.metric:
            if m not in metrics:
                skipped.setdefault(m, []).append(s)
        arm_out = {}
        for n in names:
            run_out = []
            for r in runs[n]:
                dropped = sorted(set(r["views"]) - common)
                run_out.append({
                    "run": r["run"], "source": r["source"], "format": r["format"],
                    "n_scored": len(r["views"]), "views_hash": r["raw_hash"],
                    "normalised_views_hash": view_hash(s, r["views"]),
                    "dropped_for_common": dropped,
                    "mean": {m: float(np.mean([r["views"][v][m] for v in sorted(common)])) for m in metrics},
                })
            arm_out[n] = {"n_runs": len(run_out), "runs": run_out,
                          "mean": {m: float(np.mean([x["mean"][m] for x in run_out])) for m in metrics}}
        per_scene[s] = {"n_expected": n_expected, "n_common": len(common),
                        "common_view_hash": view_hash(s, common), "identical_view_sets": identical,
                        "common_views": sorted(common), "metrics": metrics, "arms": arm_out}
    for m, ss in skipped.items():
        notes.append(f"metric {m} absent from some views in {len(ss)} scene(s) ({', '.join(ss[:3])}"
                     f"{', ...' if len(ss) > 3 else ''}); not reported for those")
    return per_scene, fails, notes


def range_call(a_vals, b_vals, m):
    """W/=/L for B against A: W only when every B run beats every A run (perf's scoreboard rule)."""
    hi = m not in LOWER_IS_BETTER
    if (min(b_vals) > max(a_vals)) if hi else (max(b_vals) < min(a_vals)):
        return "W"
    if (min(a_vals) > max(b_vals)) if hi else (max(a_vals) < min(b_vals)):
        return "L"
    return "="


def absolute(per_scene, names, args):
    out = {}
    scenes = sorted(per_scene)
    for n in names:
        res = {}
        for m in args.metric:
            if not all(m in per_scene[s]["metrics"] for s in scenes):
                continue
            vals = {s: per_scene[s]["arms"][n]["mean"][m] for s in scenes}
            res[m] = {"mean_over_scenes": float(np.mean(list(vals.values()))), "per_scene": vals}
        out[n] = res
    return out


def summarise(per_scene, names, args):
    rng = np.random.default_rng(args.rng_seed)
    base = names[0]
    scenes = sorted(per_scene)
    comps = {}
    for b in names[1:]:
        comp = {"a": base, "b": b, "metrics": {}}
        metrics = [m for m in args.metric if all(m in per_scene[s]["metrics"] for s in scenes)]
        for m in metrics:
            scene_d, scene_rows = [], []
            for s in scenes:
                A, B = per_scene[s]["arms"][base], per_scene[s]["arms"][b]
                d = B["mean"][m] - A["mean"][m]
                seeds_a = {str(r["run"]): r["mean"][m] for r in A["runs"]}
                seeds_b = {str(r["run"]): r["mean"][m] for r in B["runs"]}
                paired = sorted(seeds_a) == sorted(seeds_b)
                seed_d = {k: seeds_b[k] - seeds_a[k] for k in sorted(seeds_a)} if paired else None
                scene_d.append(d)
                call = range_call(list(seeds_a.values()), list(seeds_b.values()), m)
                scene_rows.append({"scene": s, "delta": d, "a": A["mean"][m], "b": B["mean"][m], "call": call,
                                   "n_runs_a": A["n_runs"], "n_runs_b": B["n_runs"],
                                   "n_views": per_scene[s]["n_common"],
                                   "paired_by_seed": paired, "per_seed_delta": seed_d,
                                   "seed_sd": (float(np.std(list(seed_d.values()), ddof=1))
                                               if seed_d and len(seed_d) > 1 else None)})
            mean = float(np.mean(scene_d))
            if len(scenes) >= 2:
                ci, unit = bootstrap_mean(scene_d, args.reps, rng), "scene-clustered bootstrap (resample scenes)"
            else:
                r0 = scene_rows[0]
                if r0["paired_by_seed"]:
                    ci = bootstrap_mean(list(r0["per_seed_delta"].values()), args.reps, rng)
                    unit = "single scene: paired seed bootstrap (resample seeds)"
                else:
                    A, B = per_scene[scenes[0]]["arms"][base], per_scene[scenes[0]]["arms"][b]
                    ci = bootstrap_two([r["mean"][m] for r in A["runs"]], [r["mean"][m] for r in B["runs"]],
                                       args.reps, rng)
                    unit = "single scene: unpaired run bootstrap (resample each arm's runs)"
            tally = {k: sum(1 for r in scene_rows if r["call"] == k) for k in ("W", "=", "L")}
            comp["metrics"][m] = {"mean_delta": mean, "ci95": ci, "ci_unit": unit, "tally": tally,
                                  "n_scenes": len(scenes),
                                  "positive_scenes": sum(1 for d in scene_d if (d < 0 if m in LOWER_IS_BETTER else d > 0)),
                                  "scenes": scene_rows}
        comps[b] = comp
    return comps


def fmt(x, nd=4):
    return "-" if x is None else f"{x:+.{nd}f}"


def print_report(per_scene, comps, arms, names, args, notes):
    print(f"arms: A={names[0]}  " + "  ".join(f"B={n}" for n in names[1:]))
    for n in names:
        print(f"  {n}: {len(arms[n]['runs'])} run(s) from {len(arms[n]['files'])} file(s); "
              f"formats {sorted({r['format'] for r in arms[n]['runs']})}")
    print("\nper scene (view coverage):")
    for s, ps in sorted(per_scene.items()):
        cov = "  ".join(f"{n}[" + ",".join(f"{r['run']}:{r['n_scored']}" for r in ps["arms"][n]["runs"]) + "]"
                        for n in names)
        print(f"  {s}: n_expected={ps['n_expected']} n_common={ps['n_common']} hash={ps['common_view_hash'][:16]}"
              f"{'' if ps['identical_view_sets'] else ' (RESTRICTED to common views)'}  {cov}")
    for b, comp in comps.items():
        print(f"\n{b} - {names[0]}:")
        for m, r in comp["metrics"].items():
            ci = r["ci95"]
            ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "n/a"
            t = r["tally"]
            print(f"  {m:<10} mean {fmt(r['mean_delta'])}  95% CI {ci_s}  "
                  f"({r['n_scenes']} scene(s), {r['positive_scenes']} improved; {r['ci_unit']})  "
                  f"tally W{t['W']} ={t['=']} L{t['L']}")
            for row in r["scenes"]:
                sd = f" seed SD {row['seed_sd']:.4f}" if row["seed_sd"] is not None else ""
                ps = (" seeds " + " ".join(f"{k}:{v:+.4f}" for k, v in row["per_seed_delta"].items())
                      if row["per_seed_delta"] else "")
                print(f"      {row['scene']:<26} {fmt(row['delta'])} {row['call']}  (A {row['a']:.4f} n={row['n_runs_a']}, "
                      f"B {row['b']:.4f} n={row['n_runs_b']}, views {row['n_views']}){sd}{ps}")
    for n in notes:
        print(f"note: {n}")


def git_head():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=SPEC")
    ap.add_argument("--metric", action="append", choices=METRICS)
    ap.add_argument("--common-views", action="store_true",
                    help="restrict each scene to the views scored by every run of every arm")
    ap.add_argument("--expected-views", metavar="FILE",
                    help="JSON {scene: [views]} / JSON list / text 'scene view' or 'view' lines")
    ap.add_argument("--raw-names", action="store_true", help="compare view names verbatim")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--json", metavar="OUT")
    ap.add_argument("--label", help="free-text label stored in the JSON")
    args = ap.parse_args(argv)
    args.metric = args.metric or list(METRICS)

    out = {"schema": "pubs-verify/v1", "label": args.label, "argv": sys.argv[1:], "cwd": os.getcwd(),
           "generated_at": datetime.now(PACIFIC).isoformat(timespec="seconds"),
           "verifier_sha256": file_sha(os.path.abspath(__file__)), "verifier_git_head": git_head(),
           "options": {"common_views": args.common_views, "expected_views": args.expected_views,
                       "raw_names": args.raw_names, "reps": args.reps, "rng_seed": args.rng_seed},
           "status": "FAIL"}
    code = 0
    try:
        specs = {}
        for a in args.arm:
            name, pattern, filters, rows_path = parse_spec(a)
            specs.setdefault(name, []).append((pattern, filters, rows_path))
        if not specs:
            raise VerifyError("need at least one arm")
        arms = {}
        for name, sp in specs.items():
            runs, files, nopv = load_arm(name, sp, args.raw_names)
            arms[name] = {"runs": runs, "files": files, "nopv": nopv}
        names = list(arms)
        out["inputs"] = {n: {"specs": [{"glob": p, "filters": [f"{k}{'!=' if ng else '='}{v}" for k, ng, v in f],
                                        "rows": r} for p, f, r in specs[n]],
                             "files": [{"path": os.path.abspath(f), "sha256": file_sha(f)} for f in arms[n]["files"]],
                             # External comparisons: one pinned engine sha per arm (lane-brief).
                             "engine_sha256": sorted({s for r in arms[n]["runs"] if (s := engine_sha(r["meta"]))}),
                             "engine_commit": sorted({str(c) for r in arms[n]["runs"]
                                                      if (c := (r["meta"] or {}).get("engine_commit"))}),
                             "engine_sha_unrecorded": sum(1 for r in arms[n]["runs"] if not engine_sha(r["meta"]))}
                         for n in names}
        per_scene, fails, notes = analyse(arms, args)
        out["notes"] = notes
        if fails:
            out["failures"] = fails
            raise VerifyError("FAIL CLOSED:\n  " + "\n  ".join(fails))
        comps = summarise(per_scene, names, args)
        absol = absolute(per_scene, names, args)
        out.update(status="OK", per_scene=per_scene, absolute=absol, comparisons=comps)
        print_report(per_scene, comps, arms, names, args, notes)
        print("\nabsolute (mean over scenes of per-scene arm means):")
        for n, res in absol.items():
            print(f"  {n}: " + "  ".join(f"{m} {v['mean_over_scenes']:.4f}" for m, v in res.items())
                  + f"  ({len(per_scene)} scene(s))")
    except VerifyError as e:
        out.setdefault("failures", [str(e)])
        print(str(e), file=sys.stderr)
        code = 2
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
            f.write("\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
