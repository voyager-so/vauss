#!/usr/bin/env bash
# Assert an unpacked archive's `vauss doctor` JSON is ready and resolves every bundled helper from tools/.
set -euo pipefail
target=${1:?usage: check-bundled-doctor.sh TARGET DOCTOR_JSON}
report=${2:?usage: check-bundled-doctor.sh TARGET DOCTOR_JSON}
checks=('.readiness.ready == true' '.readiness.required.ffmpeg.source == "bundled"'
  '.readiness.optional.onnxruntime.source == "bundled"'
  '.readiness.optional["lpips-alex"].source == "bundled"' '.readiness.optional["lpips-vgg"].source == "bundled"')
if [[ $target == x86_64-unknown-linux-gnu ]] &&
  jq -e --arg t "$target" 'any(.[]; .id == "colmap" and .target == $t)' "$(dirname "$0")/../crates/splat-fetch/assets.json" >/dev/null; then
  checks+=('.readiness.required.colmap.source == "bundled"')
fi
jq '.readiness' "$report"
for check in "${checks[@]}"; do
  jq -e "$check" "$report" >/dev/null || { echo "bundled doctor check failed: $check" >&2; exit 1; }
done
echo "bundled doctor checks passed (${#checks[@]})"
