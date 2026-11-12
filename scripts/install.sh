#!/bin/sh
set -eu
version=${VAUSS_VERSION:-${SPLAT_VERSION:-${1:-}}}
[ -n "$version" ] || { echo 'Usage: install.sh VERSION (for example 0.0.1)' >&2; exit 1; }
version=${version#v}
case "$version" in *[!0-9A-Za-z.+-]*|'') echo 'Invalid version' >&2; exit 1;; esac
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) target=x86_64-unknown-linux-gnu ;;
  Darwin-arm64) target=aarch64-apple-darwin ;;
  *) echo 'Supported: Linux x64 and macOS arm64. Windows: use install.ps1.' >&2; exit 1 ;;
esac
name="vauss-v$version-$target"
archive="$name.tar.gz"
base="${VAUSS_RELEASE_BASE_URL:-${SPLAT_RELEASE_BASE_URL:-https://github.com/voyager-so/vauss/releases/download}}/vauss-v$version"
install_dir=${VAUSS_INSTALL_DIR:-${SPLAT_INSTALL_DIR:-"$HOME/.local/bin"}}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT HUP INT TERM
curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --retry 3 "$base/$archive" -o "$stage/$archive"
curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --retry 3 "$base/$archive.sha256" -o "$stage/checksum"
expected=$(awk 'NR == 1 {print $1}' "$stage/checksum")
case "$expected" in *[!0-9a-fA-F]*|'') echo 'Invalid checksum' >&2; exit 1;; esac
[ "${#expected}" -eq 64 ] || { echo 'Invalid checksum length' >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then actual=$(sha256sum "$stage/$archive" | awk '{print $1}');
else actual=$(shasum -a 256 "$stage/$archive" | awk '{print $1}'); fi
[ "$actual" = "$expected" ] || { echo 'Checksum mismatch; nothing installed' >&2; exit 1; }
# Release archives contain one known directory. Verify names before extracting.
tar -tzf "$stage/$archive" | while IFS= read -r entry; do
  case "$entry" in
    *..*|/*) echo "Unsafe archive member: $entry" >&2; exit 1;;
    "$name/"|"$name/vauss"|"$name/splat"|"$name/voyager-splat"|"$name/LICENSE.txt"|"$name/NOTICE.txt"|"$name/THIRD_PARTY_LICENSES.txt"|"$name/distribution.md"|"$name/assets.json"|"$name/tools/"*) ;;
    *) echo "Unexpected archive member: $entry" >&2; exit 1;;
  esac
done
tar -xzf "$stage/$archive" -C "$stage"
# Keep helpers and notices beside the resolved executable.
vauss_home=${VAUSS_HOME:-${SPLAT_HOME:-"$install_dir/../share/vauss"}}
mkdir -p "$vauss_home" "$install_dir"
rm -rf "${vauss_home:?}/.${version:?}-new-$$"
mv "$stage/$name" "$vauss_home/.$version-new-$$"
rm -rf "${vauss_home:?}/${version:?}"
mv "$vauss_home/.$version-new-$$" "$vauss_home/$version"
chmod 755 "$vauss_home/$version/vauss"
for command in vauss splat voyager-splat; do
  ln -sfn "$(cd "$vauss_home/$version" && pwd)/vauss" "$install_dir/.$command-new-$$"
  mv -f "$install_dir/.$command-new-$$" "$install_dir/$command"
done
printf 'Installed %s -> %s\nAdd %s to PATH, then run: vauss doctor\n' "$install_dir/vauss" "$vauss_home/$version/vauss" "$install_dir"
