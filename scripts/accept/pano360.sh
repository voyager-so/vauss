#!/usr/bin/env bash
# 360 equirect acceptance run (GPU): OmniPhotos (CC BY 4.0, Bertel et al. 2020, doi:10.15125/BATH-00948).
# Usage: pano360.sh DATASET_DIR OUT_DIR
#   DATASET_DIR: the unzipped Preprocessed/ folder (one folder per scene, stitched frames in <scene>/Input/).
#   OUT_DIR: runs/, logs/, exports checks and report.md. VAUSS=<binary> picks the vauss binary.
# Per scene (Temple3 and Ballintoy when present, else the first two by name), seed 42, default steps:
#   arm A (rig)    vauss run <scene> --camera-model equirect
#   arm B (no rig) the same with SPLAT_PANO_RIG=0 and --no-mesh
# plus one run on the first scene without --camera-model (detection: stopped once the frames are split,
# since only the detection message and the panorama count are checked) and the exports and viewer check
# on arm A. The B arms and the detection run go alongside the A arms (B matches far fewer pairs: no rig
# expansion), so the whole run fits a 3 h lease.
# eval/pano360_report.py then prints the gate table and PASS/FAIL.
set -uo pipefail
data=${1:?usage: pano360.sh DATASET_DIR OUT_DIR}
out=${2:?usage: pano360.sh DATASET_DIR OUT_DIR}
vauss=${VAUSS:-vauss}
here=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out/runs" "$out/logs"
scenes=()
for s in Temple3 Ballintoy; do [[ -d $data/$s/Input ]] && scenes+=("$s"); done
if (( ${#scenes[@]} < 2 )); then
  scenes=()
  while IFS= read -r d; do scenes+=("$(basename "$d")"); done < <(find "$data" -mindepth 1 -maxdepth 1 -type d -exec test -d '{}/Input' ';' -print | sort | head -2)
fi
(( ${#scenes[@]} == 2 )) || { echo "need two scenes with Input/ under $data" >&2; exit 2; }
printf '%s\n' "${scenes[@]}" > "$out/scenes.txt"
: > "$out/runs.tsv"

# run NAME ARGS... : one command (logs/NAME.*), its exit code and wall seconds recorded in runs.tsv.
run() {
  local name=$1; shift
  local start=$SECONDS
  "$@" > "$out/logs/$name.stdout" 2> "$out/logs/$name.stderr"
  local code=$?
  printf '%s\t%s\t%s\n' "$name" "$code" "$((SECONDS - start))" >> "$out/runs.tsv"
  echo "$name: exit $code in $((SECONDS - start)) s" >&2
  return 0
}

# detect SCENE: vauss run without --camera-model, stopped once its frames are split (journal panorama_split).
detect() {
  local r=$out/runs/$1-detect start=$SECONDS
  "$vauss" run "$data/$1/Input" --input-kind video --driver fixed --non-interactive --goal "omniphotos $1" --seed 42 \
    --no-mesh --run-dir "$r" > "$out/logs/$1-detect.stdout" 2> "$out/logs/$1-detect.stderr" &
  local pid=$!
  until grep -q '"panorama_split"' "$r/journal.jsonl" 2>/dev/null || ! kill -0 "$pid" 2>/dev/null; do sleep 5; done
  local split=0
  grep -q '"panorama_split"' "$r/journal.jsonl" 2>/dev/null && { split=1; "$vauss" stop "$r" > /dev/null 2>&1; }
  wait "$pid"
  printf '%s\t%s\t%s\n' "$1-detect" "$((1 - split))" "$((SECONDS - start))" >> "$out/runs.tsv"
}

(
  for s in "${scenes[@]}"; do
    run "$s-norig" env SPLAT_PANO_RIG=0 "$vauss" run "$data/$s/Input" --camera-model equirect --input-kind video --driver fixed \
      --non-interactive --goal "omniphotos $s" --seed 42 --no-mesh --run-dir "$out/runs/$s-norig"
  done
  detect "${scenes[0]}"
) &
side=$!
for s in "${scenes[@]}"; do
  run "$s-rig" "$vauss" run "$data/$s/Input" --camera-model equirect --input-kind video --driver fixed --non-interactive \
    --goal "omniphotos $s" --seed 42 --run-dir "$out/runs/$s-rig"
done
wait "$side"

# Exports on arm A (vauss run already writes the viewer, web and blender packs; rerun each command explicitly).
for s in "${scenes[@]}"; do
  r=$out/runs/$s-rig
  [[ -d $r ]] || continue
  run "$s-export-viewer" "$vauss" export-viewer "$r"
  run "$s-export-web" "$vauss" export-web "$r" --out "$out/exports/$s-web"
  run "$s-export-blender" "$vauss" export-blender "$r"
  run "$s-export-game" "$vauss" export-game "$r"
done
run check-view bash "$here/scripts/check-view.sh" "$vauss" "$out/check-view"
# Screenshots (reported): one held-out panorama looking up (v12), down (v13) and across the stitch line (v02).
mkdir -p "$out/screenshots"
for s in "${scenes[@]}"; do
  r=$out/runs/$s-rig
  req=$(python3 "$here/eval/pano360_report.py" --views-request "$r") || continue
  "$vauss" --json tool render_views --run-dir "$r" --input "$req" > "$out/logs/$s-views.json" 2> "$out/logs/$s-views.stderr" || continue
  python3 -c 'import json,sys,shutil,pathlib
r=json.load(open(sys.argv[1])); run=pathlib.Path(sys.argv[2]); dst=pathlib.Path(sys.argv[3])
for i in r.get("result", r).get("images", []): shutil.copy(run / i, dst / (sys.argv[4] + "-" + pathlib.Path(i).name))' \
    "$out/logs/$s-views.json" "$r" "$out/screenshots" "$s" || true
done

python3 "$here/eval/pano360_report.py" "$out" > "$out/report.md"
code=$?
cat "$out/report.md"
exit $code
