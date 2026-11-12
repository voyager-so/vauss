#!/usr/bin/env bash
# Prints one notices file from the LICENSES/ folders of built packages (the union over targets):
#   dist/ffmpeg/notices.sh PKG_LICENSES_DIR... > FFMPEG-THIRD-PARTY.txt
set -euo pipefail
echo "Third-party notices for splat's FFmpeg build (ffmpeg and ffprobe, all targets)."
echo "FFmpeg is LGPL-2.1-or-later; the libraries statically linked into it are listed below with"
echo "their licence texts. Source for every component: SOURCE.txt beside the binaries."
for c in $(for d in "$@"; do ls "$d"; done | sort -u); do
  for f in $(for d in "$@"; do [[ -d "$d/$c" ]] && (cd "$d/$c" && ls); done | grep -vx CONFIGURATION | sort -u); do
    for d in "$@"; do
      if [[ -f "$d/$c/$f" ]]; then
        printf '\n\n==== %s: %s ====\n\n' "$c" "$f"
        cat "$d/$c/$f"
        break
      fi
    done
  done
done
