#!/usr/bin/env bash
# SPLAT_BIN selects an existing CLI; --data preserves solved scale/orientation.
set -euo pipefail
if (( $# < 2 )); then echo 'usage: package.sh SPLAT.ply OUT [--data COLMAP_DIR | --web EXISTING_WEB] [--collider VIEWER] [--title TITLE] [--provenance PRIVATE.json]' >&2; exit 2; fi
ply=$(realpath "$1"); out=$(realpath -m "$2"); shift 2
script=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${SPLAT_BIN:-splat}
data=; web=; collider=; provenance=; title='Free walk room'
while (( $# )); do case "$1" in
 --data) data=$2;; --web) web=$2;; --collider) collider=$2;; --title) title=$2;; --provenance) provenance=$2;; *) echo "unknown option: $1" >&2;exit 2;; esac;shift 2;done
[[ -f $ply ]] || { echo "not found: $ply" >&2;exit 2; }
[[ ! -e $out ]] || { echo "output exists: $out (choose a new directory)" >&2;exit 2; }
if [[ -n $provenance ]];then provenance=$(realpath -m "$provenance"); [[ $provenance != "$out"/* ]] || { echo "--provenance must be outside the served package $out" >&2;exit 2; };fi
mkdir -p "$(dirname "$out")"
stage=$(mktemp -d "$(dirname "$out")/.free-roam-package.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
if [[ -z $web ]];then
 web="$stage/web"
 if [[ -n $data ]];then
  "$bin" export-web "$ply" --out "$web" --data "$data" --no-collision --lod inline
 else
  mkdir -p "$web"
  "$bin" export-web "$ply" --raw --out "$web/scene.spz"
  python3 "$script/raw-scene.py" "$ply" "$web/scene.json"
 fi
fi
args=(--no-collide)
"$bin" export-viewer "$web" --out "$stage/viewer" --title "$title" "${args[@]}"
if [[ -z $collider ]];then python3 "$script/collider.py" "$ply" "$web/scene.json" "$stage/viewer";fi
patch=(--no-backup)  # a fresh package has no earlier page to keep
[[ -z $collider ]] || patch+=(--collision-from "$collider")
python3 "$script/../patch-viewer-roam.py" "$stage/viewer" "${patch[@]}"
python3 - "$stage/viewer" <<'PY'
import json,re,sys
from pathlib import Path
p=Path(sys.argv[1]); cfg=json.loads(re.search(r'^const CFG = (.*);$',(p/'index.html').read_text(),re.M)[1])
if not cfg.get('collide'): raise SystemExit('export has no collision grid; supply --collider with a package in the same frame')
PY
# Served metadata carries a content hash, never local paths; --provenance keeps them privately (outside OUT).
python3 "$script/provenance.py" "$stage/viewer" "$ply" --web "$web" ${provenance:+--private "$provenance"}
# Bake imports locally too: localhost packages must not need a CDN at viewing time.
python3 "$script/vendor.py" "$stage/viewer"
mv "$stage/viewer" "$out"
echo "Viewer: $out/index.html"
