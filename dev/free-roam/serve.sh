#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then echo 'usage: serve.sh PACKAGE PORT [UNIT]' >&2; exit 2; fi
root=$(realpath "$1")
port=$2
unit=${3:-free-roam-$port}
[[ $port =~ ^[0-9]+$ && $unit =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
exec systemd-run --user --unit="$unit" --property=Restart=on-failure --property=MemoryMax=1G python3 -m http.server "$port" --bind 127.0.0.1 --directory "$root"
