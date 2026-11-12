#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
output=${1:-target/dist/THIRD_PARTY_LICENSES.txt}
mkdir -p "$(dirname "$output")"
# Failure is fatal: never ship an archive with an incomplete license inventory.
cargo about generate --locked --fail --manifest-path crates/splat-cli/Cargo.toml --config dist/about.toml dist/about.hbs --output-file "$output"
