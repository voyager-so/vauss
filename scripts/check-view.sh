#!/usr/bin/env bash
# Smoke-test the embedded web app: `vauss view` serves the picker, answers /api/health, and exits 0 on /api/quit.
set -euo pipefail
splat=${1:?usage: check-view.sh SPLAT_BINARY OUT_DIR}
out=${2:?usage: check-view.sh SPLAT_BINARY OUT_DIR}
mkdir -p "$out"
# stdin stays open (the server exits when it closes) until /api/quit stops it.
"$splat" --json view --no-open < <(sleep 300) > "$out/view.stdout" 2> "$out/view.stderr" &
pid=$!
for _ in $(seq 1 120); do
  [[ -s $out/view.stdout ]] && break
  kill -0 "$pid" 2>/dev/null || { cat "$out/view.stderr" >&2; echo 'vauss view exited before listening' >&2; exit 1; }
  sleep 0.5
done
url=$(head -n 1 "$out/view.stdout" | jq -r .url)
url=${url%/}
[[ $url == http://* ]] || { echo "no url in: $(head -n 1 "$out/view.stdout")" >&2; kill "$pid"; exit 1; }
curl -fsS "$url/api/health" | tee "$out/health.json"
echo
curl -fsS -X POST -H 'content-type: application/json' -d '{}' "$url/api/quit" > /dev/null
code=0
wait "$pid" || code=$?
[[ $code -eq 0 ]] || { echo "vauss view exited $code after /api/quit" >&2; exit 1; }
echo "vauss view smoke passed: $url"
