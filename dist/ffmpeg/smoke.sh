#!/usr/bin/env bash
# Checks a built ffmpeg/ffprobe pair: LGPL licence and configuration, decode of public clips
# (bit-exact against frame hashes taken with the previous ffmpeg-public build and Ubuntu's FFmpeg),
# and the encoders splat uses (crates/splat-shots/src/encode.rs).
#
#   dist/ffmpeg/smoke.sh BINDIR [SAMPLES]
#
# SAMPLES caches the clips (default BINDIR/../samples). The clips come from FFmpeg's FATE test suite
# (https://fate-suite.ffmpeg.org/) and are used for testing only, never redistributed.
set -euo pipefail
bin="$(cd "${1:?usage: smoke.sh BINDIR [SAMPLES]}" && pwd)"
samples="${2:-$bin/../samples}"
mkdir -p "$samples"
exe=""; [[ -f "$bin/ffmpeg.exe" ]] && exe=.exe
ff="$bin/ffmpeg$exe" fp="$bin/ffprobe$exe"
fail=0
bad() { echo "FAIL: $*"; fail=1; }
md5() { if command -v md5sum >/dev/null; then md5sum | cut -c1-32; else command md5 -q; fi; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi; }

"$ff" -hide_banner -version | head -1
lic="$("$ff" -hide_banner -L)"
grep -q 'GNU Lesser General Public' <<<"$lic" || bad "ffmpeg -L is not the LGPL"
conf="$("$ff" -hide_banner -buildconf)"
for f in --enable-gpl --enable-nonfree --enable-version3 --enable-libx264 --enable-libx265; do
  grep -q -- "$f" <<<"$conf" && bad "built with $f"
done
"$fp" -hide_banner -version >/dev/null || bad "ffprobe does not run"

# clip  sha256  codec  frames  md5 of the framemd5 lines (decoded video, after autorotation)
while read -r path hash codec frames want; do
  f="$samples/$(basename "$path")"
  [[ -f "$f" && "$(sha256 "$f")" == "$hash" ]] ||
    curl -fsSL --retry 5 --retry-all-errors -o "$f" "https://fate-suite.ffmpeg.org/$path"
  [[ "$(sha256 "$f")" == "$hash" ]] || { bad "$path: download or SHA-256 mismatch"; continue; }
  got_codec="$("$fp" -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$f" | tr -d '\r' | head -1)"
  lines="$("$ff" -v quiet -i "$f" -map 0:v:0 -f framemd5 - | tr -d '\r' | grep -v '^#' || true)"
  n="$(grep -c . <<<"$lines" || true)"
  got="$(md5 <<<"$lines")"
  if [[ "$got_codec" == "$codec" && "$n" == "$frames" && "$got" == "$want" ]]; then
    echo "ok   decode $codec $(basename "$path"): $n frames, bit-exact"
  else
    bad "decode $(basename "$path"): codec $got_codec ($codec), $n frames ($frames), $got ($want)"
  fi
done <<'CLIPS'
h264-conformance/BA_MW_D.264 47c59fbe8de6edad04457b8b412579d10cf6ecf87393f252cb2493f9c20dca32 h264 100 d3a86871082f9ffbae39538932a33154
mov/displaymatrix.mov 02ad3fcdc28a1c8d9d81384829f60c5c28cd23a5a8570cbb56d0f90647ba92b4 h264 33 94007abe4f9ff73946977f466ba6a70b
mov/fake-gp-media-with-real-gpmf.mp4 4e9d19e7fd98e57468c5cbc572445682e4ee917da59d765c258cb72ca9e83fb1 h264 250 f971af6e0c72c3220c3dc6aa2e4177fb
hevc-conformance/WPP_A_ericsson_MAIN10_2.bit 3b0d7323bf81d64d3c1443c9cebb8df6a42e0719ed422e64125e03971999b956 hevc 48 59862fc4f1697ab6ea427f34a7658cc9
mov/dovi-p81.mp4 3313278f3dc49b0b3680ec7c3d72622ec0a770faff9721c5842aaf899d5471c5 hevc 10 2ca34418b64e381eeed9b6a9b1e20a3e
vp9-test-vectors/vp90-2-00-quantizer-00.webm 92f26f690d76b46bab8b0807e95b593b7cb159472cfecc35ddf545c889825c6d vp9 2 87fa73dfa39fe0ea2de586bb29e7d623
av1/film_grain.ivf 40470898ac832513f2e0cce3d1d4e1c864b8a05ba79b5466167f2a65b03cd149 av1 10 add49b71c003c3062b56db76f8aa1d4b
CLIPS

# Encoders splat calls. Required: compiled in and working everywhere.
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
src=(-f lavfi -i testsrc=s=256x144:r=8:d=1 -f lavfi -i sine=d=1)
enc() { # NAME EXT ARGS...: encode the test clip, then decode it back.
  local name="$1" ext="$2"; shift 2
  if "$ff" -hide_banner -v error -y "${src[@]}" -vf format=yuv420p "$@" "$tmp/$name.$ext" &&
     "$ff" -hide_banner -v error -i "$tmp/$name.$ext" -f null - ; then
    echo "ok   encode $name"; return 0
  fi
  return 1
}
enc vp9+opus webm -c:v libvpx-vp9 -crf 24 -b:v 0 -deadline good -cpu-used 4 -row-mt 1 -c:a libopus || bad "encode libvpx-vp9 + libopus"
enc mpeg4+aac mp4 -c:v mpeg4 -q:v 2 -c:a aac || bad "encode mpeg4 + aac"
enc mjpeg jpg -frames:v 1 -an -c:v mjpeg -q:v 2 -update 1 || bad "encode jpeg"
"$ff" -hide_banner -v error -y -f lavfi -i testsrc=s=64x64:d=1 -frames:v 1 "$tmp/p.png" && echo "ok   encode png" || bad "encode png"
# drawtext, as splat ad's title cards use it (crates/splat-shots/src/ad.rs): the card must differ from a
# blank one. The font path is relative: Windows drive colons break filter arguments.
cp "$(dirname "$0")/../../crates/splat-shots/src/BebasNeue-Regular.ttf" "$tmp/font.ttf"
card() { (cd "$tmp" && "$ff" -hide_banner -v error -f lavfi -i color=c=black:s=640x360:d=0.1 "$@" -frames:v 1 -f framemd5 - | grep -v '^#'); }
text="$(card -vf "drawtext=fontfile=font.ttf:text='Splat':fontsize=150:fontcolor=white:x=(w-tw)/2:y=(h-th)/2" || true)"
if [[ -n "$text" && "$text" != "$(card)" ]]; then echo "ok   drawtext"; else bad "drawtext"; fi

# H.264: whichever OS/GPU encoder this target has must be compiled in; whether it runs depends on
# the machine (CI runners have no GPU), so a failed trial is reported, not fatal.
encoders="$("$ff" -hide_banner -encoders)"
case "$(uname -s)" in
  Darwin) want=(h264_videotoolbox hevc_videotoolbox) ;;
  MINGW* | MSYS* | CYGWIN*) want=(h264_mf hevc_mf) ;;
  *) want=(h264_vulkan hevc_vulkan) ;;
esac
for e in "${want[@]}"; do
  grep -qw "$e" <<<"$encoders" || bad "$e is not compiled in"
done
h=${want[0]}
pre=(); vf=format=yuv420p; args=(-b:v 12M)
[[ $h == h264_vulkan ]] && { pre=(-init_hw_device vulkan=vk -filter_hw_device vk); vf=format=nv12,hwupload; }
[[ $h == h264_videotoolbox ]] && args+=(-allow_sw 1)
if "$ff" -hide_banner -v error -y ${pre[@]+"${pre[@]}"} -f lavfi -i testsrc=s=256x144:r=8:d=0.5 -vf "$vf" -c:v "$h" "${args[@]}" "$tmp/h.mp4" 2>"$tmp/h.err"; then
  echo "ok   encode $h (runs on this machine)"
else
  echo "note encode $h is compiled in but does not run on this machine: $(head -c 300 "$tmp/h.err")"
fi
grep -qw libx264 <<<"$encoders" && bad "libx264 is present"
exit $fail
