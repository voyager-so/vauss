#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
target=${1:?usage: package.sh TARGET [OUTPUT_DIR]}
case "$target" in
  x86_64-unknown-linux-gnu|aarch64-apple-darwin) ;;
  *) echo "Unsupported Unix target: $target" >&2; exit 1 ;;
esac
output=${2:-target/dist}
mkdir -p "$output"
output=$(cd "$output" && pwd)
version=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.name == "splat-contracts") | .version')
name="vauss-v${version}-${target}"
binary=${SPLAT_BINARY:-target/$target/release/vauss}
if [[ -z ${SPLAT_BINARY:-} ]]; then cargo build --locked --release -p splat-cli --bin vauss --target "$target"; fi
[[ -f "$binary" ]] || { echo "Missing binary: $binary" >&2; exit 1; }
# SPLAT_LICENSES supports reusing an already generated, reviewed inventory locally.
licenses=${SPLAT_LICENSES:-$output/THIRD_PARTY_LICENSES.txt}
if [[ -z ${SPLAT_LICENSES:-} ]]; then scripts/licenses.sh "$licenses"; fi
[[ -s "$licenses" ]] || { echo 'Missing third-party license inventory' >&2; exit 1; }
stage=$(mktemp -d "$output/.stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/$name"
scripts/bundle-tools.sh "$target" "$stage/$name/tools"
cp "$binary" "$stage/$name/vauss"
chmod 755 "$stage/$name/vauss"
ln -s vauss "$stage/$name/splat"
ln -s vauss "$stage/$name/voyager-splat"
cp "$licenses" "$stage/$name/THIRD_PARTY_LICENSES.txt"
cp dist/LICENSE.txt "$stage/$name/LICENSE.txt"
# Apache-2.0 4(d): ship the top-level NOTICE plus every NOTICE it points to.
for notice in NOTICE crates/splat-train/NOTICE crates/splat-eval/NOTICE crates/splat-sfm-learned/NOTICE crates/splat-sfm-learned/LICENSE-IJG.txt vendor/cubecl-common/NOTICE.md vendor/cubecl-wgpu/NOTICE.md; do
  printf '==> %s <==\n' "$notice"; cat "$notice"; printf '\n'
done > "$stage/$name/NOTICE.txt"
cp docs/distribution.md "$stage/$name/distribution.md"
cp crates/splat-fetch/assets.json "$stage/$name/assets.json"
tar -czf "$output/$name.tar.gz" -C "$stage" "$name"
cd "$output"
if command -v sha256sum >/dev/null; then sha256sum "$name.tar.gz" > "$name.tar.gz.sha256";
else shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256"; fi
printf '%s\n' "$output/$name.tar.gz"
