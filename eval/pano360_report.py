#!/usr/bin/env python3
"""360 acceptance report (scripts/accept/pano360.sh): the gate table and PASS/FAIL.

Usage:
  pano360_report.py OUT_DIR                 print report.md (exit 1 when a blocking gate fails)
  pano360_report.py --views-request RUN     print a render_views request for the up/down/stitch screenshots

Reads, per run under OUT_DIR/runs/: state.json (the last successful evaluate), the evaluation's
metrics.json (per-image PSNR/LPIPS), the SfM artifact's reconstruction.json (panorama block, stage
times) and the split manifest. Held-out scores use views v00-v12 of each held-out panorama; the
nadir view (v13, mostly masked) is reported separately.
"""

import json
import re
import statistics
import sys
from pathlib import Path

VIEW = re.compile(r"^v(\d\d)_(.+)$")
ROWS = {"level": range(0, 4), "up": range(4, 8), "down": range(8, 12), "zenith": [12], "nadir": [13]}
RIG_ANGLE_DEG, RIG_CENTRE_REL = 0.01, 1e-4
# The A vs B comparison needs this many held-out panoramas scored by both arms.
MIN_SHARED_PANORAMAS = 4
DETECTED = "360 footage detected (equirectangular)"


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def view_of(name):
    m = VIEW.match(Path(name).stem if name.endswith((".png", ".jpg")) else name)
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def last_evaluation(run):
    """(candidate_id, eval artifact dir, split_id) of the newest finished evaluation."""
    state = load(run / "state.json") or {}
    best = None
    for art in state.get("artifacts", {}).values():
        if art.get("kind") != "eval":
            continue
        metrics = run / art["path"] / "metrics.json"
        if metrics.is_file() and (best is None or metrics.stat().st_mtime > best[0]):
            meta = art.get("meta", {})
            best = (metrics.stat().st_mtime, meta.get("candidate_id"), run / art["path"], meta.get("split_id"))
    return best[1:] if best else (None, None, None)


def panorama_sfm(run):
    """The newest SfM artifact whose reconstruction.json carries a panorama block."""
    found = []
    for rec_path in (run / "artifacts").glob("sfm_*/reconstruction.json"):
        rec = load(rec_path)
        if rec and rec.get("panorama"):
            found.append((rec_path.stat().st_mtime, rec_path.parent, rec))
    return max(found, key=lambda f: f[0])[1:] if found else (None, None)


def per_image(eval_dir):
    """{image stem: (view, psnr, lpips)} from the evaluation's per-image rows."""
    metrics = load(eval_dir / "metrics.json") if eval_dir else None
    out = {}
    for row in (metrics or {}).get("per_image", []):
        image = str(row.get("image", ""))
        view, _ = view_of(image)
        if view is None:
            continue
        lpips = next((v for k, v in row.items() if k.startswith("lpips") and isinstance(v, (int, float))), None)
        out[image] = (view, row.get("psnr"), lpips)
    return out


def scores(rows, only=None):
    """{view: [(psnr, lpips)]}, optionally restricted to the image stems in `only`."""
    out = {}
    for image, (view, psnr, lpips) in rows.items():
        if only is None or image in only:
            out.setdefault(view, []).append((psnr, lpips))
    return out


def gate_means(rows, only=None):
    """Mean PSNR and LPIPS over views v00-v12."""
    s = scores(rows, only)
    views = [x for v in range(13) for x in s.get(v, [])]
    return mean(x[0] for x in views), mean(x[1] for x in views)


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(values) if values else None


def summarize(run):
    run = Path(run)
    r = {"run": run.name, "exists": run.is_dir()}
    sfm_dir, rec = panorama_sfm(run)
    if rec:
        p = rec["panorama"]
        r.update(panoramas_registered=p.get("panoramas_registered"), panoramas_input=p.get("panoramas_input"),
                 rig=p.get("rig"), rig_check=p.get("rig_check"), components=len(rec.get("component_sizes") or [1]),
                 views_registered=(rec.get("health") or {}).get("n_registered"), views_input=(rec.get("health") or {}).get("n_input"),
                 sfm_seconds=rec.get("wall_seconds"), stages={s["name"]: round(s["seconds"], 1) for s in rec.get("stages", [])})
    candidate, eval_dir, split_id = last_evaluation(run)
    r["candidate_id"] = candidate
    r["per_image"] = per_image(eval_dir)
    s = scores(r["per_image"])
    r["psnr"], r["lpips"] = gate_means(r["per_image"])
    r["rows"] = {name: mean(x[0] for v in views for x in s.get(v, [])) for name, views in ROWS.items()}
    if split_id:
        manifest = load(run / "artifacts" / split_id / "manifest.json") or {}
        group = lambda n: view_of(n)[1] or n
        held = {group(n) for n in manifest.get("heldout_names", [])}
        train = {group(n) for n in manifest.get("train_names", [])}
        r["split_ok"] = bool(held) and not (held & train)
        r["heldout_panoramas"] = len(held)
    return r


def fmt(v, digits=2):
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def report(out):
    out = Path(out)
    scenes = (out / "scenes.txt").read_text().split()
    runs = {}
    for line in (out / "runs.tsv").read_text().splitlines():
        name, code, secs = line.split("\t")
        runs[name] = (int(code), int(secs))
    gates, lines = [], []

    def gate(name, ok, detail):
        gates.append(ok)
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    summary = {}
    for scene in scenes:
        a, b = summarize(out / "runs" / f"{scene}-rig"), summarize(out / "runs" / f"{scene}-norig")
        summary[scene] = {"rig": a, "norig": b}
        reg_a = (a.get("panoramas_registered") or 0) / max(a.get("panoramas_input") or 1, 1)
        reg_b = (b.get("panoramas_registered") or 0) / max(b.get("panoramas_input") or 1, 1)
        b_exit = runs.get(f"{scene}-norig", (None,))[0]
        views_b = (b.get("views_registered") or 0) / max(b.get("views_input") or 1, 1)
        # B solved: its run finished and it registered the same 95% of its views A must.
        b_solved = b_exit == 0 and b.get("psnr") is not None and views_b >= 0.95
        gate(f"{scene}: panoramas registered (A)", reg_a >= 0.95 and a.get("components") == 1 and reg_a >= reg_b,
             f"A {a.get('panoramas_registered')}/{a.get('panoramas_input')} ({reg_a:.0%}), {a.get('components')} component(s); "
             f"views {a.get('views_registered')}/{a.get('views_input')}; B {b.get('panoramas_registered')}/{b.get('panoramas_input')} "
             f"panoramas, views {b.get('views_registered')}/{b.get('views_input')}")
        rc = a.get("rig_check") or {}
        gate(f"{scene}: rig check", bool(rc) and rc.get("worst_angle_deg", 1) <= RIG_ANGLE_DEG and rc.get("worst_centre_rel", 1) <= RIG_CENTRE_REL,
             f"worst {fmt(rc.get('worst_angle_deg'), 6)} deg, centres {fmt(rc.get('worst_centre_rel'), 8)} of a frame step")
        if b_solved:
            # Paired: only the held-out views both arms registered and scored.
            common = set(a["per_image"]) & set(b["per_image"])
            shared = len({view_of(n)[1] for n in common})
            pa, la = gate_means(a["per_image"], common)
            pb, lb = gate_means(b["per_image"], common)
            ok = shared >= MIN_SHARED_PANORAMAS and pa is not None and pb is not None and pa >= pb - 0.2 \
                and (la is None or lb is None or la <= lb + 0.01)
            detail = f"on {len(common)} shared views ({shared} panoramas, need {MIN_SHARED_PANORAMAS}): A {fmt(pa)} dB / {fmt(la, 3)} vs B {fmt(pb)} dB / {fmt(lb, 3)}"
        else:
            # Spec: if B fails to solve, this gate passes. "Fails to solve" means its run ended without
            # an evaluation, not that it never ran.
            ok = b_exit is not None and views_b < 0.95
            detail = (f"B did not solve (exit {b_exit}, {b.get('views_registered')}/{b.get('views_input')} views registered, "
                      f"{b.get('components')} components; its own held-out score {fmt(b.get('psnr'))} dB / {fmt(b.get('lpips'), 3)}); "
                      f"A {fmt(a.get('psnr'))} dB / {fmt(a.get('lpips'), 3)}")
        gate(f"{scene}: held-out v00-v12, A vs B", ok, detail)
        gate(f"{scene}: held-out PSNR (A) >= 20 dB", (a.get("psnr") or 0) >= 20.0, f"{fmt(a.get('psnr'))} dB over {a.get('heldout_panoramas')} held-out panoramas")
        gate(f"{scene}: split keeps panoramas whole", bool(a.get("split_ok")), f"{a.get('heldout_panoramas')} held-out panoramas")
        sfm_s, run_s = a.get("sfm_seconds"), runs.get(f"{scene}-rig", (1, 10**9))[1]
        gate(f"{scene}: time on the lease (A)", sfm_s is not None and sfm_s <= 1800 and run_s <= 3600,
             f"camera solve {fmt((sfm_s or 0) / 60, 1)} min, whole run {run_s / 60:.1f} min")
        exports = {k: runs.get(f"{scene}-export-{k}", (None,))[0] for k in ("viewer", "web", "blender", "game")}
        gate(f"{scene}: exports", all(c == 0 for c in exports.values()), ", ".join(f"{k} exit {c}" for k, c in exports.items()))
    first = scenes[0]
    stderr = (out / "logs" / f"{first}-detect.stderr")
    detected = stderr.is_file() and DETECTED in stderr.read_text(errors="replace")
    # The detection run stops after the split: its panorama count is the split manifest's frames.
    split_frames = lambda run: max((len((load(p) or {}).get("frames", [])) for p in (run / "artifacts").glob("frames_*/panorama.json")), default=None)
    d_count, a_count = split_frames(out / "runs" / f"{first}-detect"), split_frames(out / "runs" / f"{first}-rig")
    same = d_count is not None and d_count == a_count
    gate(f"{first}: detection without --camera-model", detected and same,
         f"message {'printed' if detected else 'missing'}; {d_count} panoramas split vs A {a_count}")
    gate("viewer check (scripts/check-view.sh)", runs.get("check-view", (1,))[0] == 0, f"exit {runs.get('check-view', ('?',))[0]}")
    ok = all(gates)
    print(f"# Vauss 360 acceptance: {'PASS' if ok else 'FAIL'}\n")
    print("Data: OmniPhotos (CC BY 4.0, Bertel et al. 2020, doi:10.15125/BATH-00948), scenes " + ", ".join(scenes) + ".\n")
    print("| Gate | Result | Measured |\n|---|---|---|")
    print("\n".join(lines))
    print("\n## Reported, not blocking\n")
    print("| Scene | Arm | level | up | down | zenith | nadir (v13) | SfM stages (s) |\n|---|---|---|---|---|---|---|---|")
    for scene, arms in summary.items():
        for arm, r in arms.items():
            rows = r.get("rows", {})
            print(f"| {scene} | {arm} | " + " | ".join(fmt(rows.get(k)) for k in ROWS) + f" | {r.get('stages', {})} |")
    for arms in summary.values():
        for r in arms.values():
            r.pop("per_image", None)
    (out / "report.json").write_text(json.dumps({"pass": ok, "scenes": summary, "runs": runs}, indent=2))
    return 0 if ok else 1


def views_request(run):
    """render_views for one held-out panorama: looking up (v12), down (v13) and across the stitch line (v02)."""
    run = Path(run)
    candidate, _, split_id = last_evaluation(run)
    manifest = load(run / "artifacts" / split_id / "manifest.json") if split_id else None
    if not candidate or not manifest:
        return 1
    held = sorted(manifest["heldout_names"])
    frame = view_of(held[0])[1]
    names = [n for n in held if view_of(n)[1] == frame and view_of(n)[0] in (2, 12, 13)]
    print(json.dumps({"candidate_id": candidate, "image_names": names, "max_views": len(names)}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--views-request":
        sys.exit(views_request(sys.argv[2]))
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(report(sys.argv[1]))
