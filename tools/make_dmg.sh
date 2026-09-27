#!/bin/sh
# Package an installed FROST.app into a distributable .dmg.
#
#   sh tools/make_dmg.sh [app-path] [out-dir]
#     app-path  default ~/Applications/FROST.app   (read only; never modified)
#     out-dir   default <repo>/dist
#
# The app is copied to a temp staging folder first; all edits (icon injection, re-sign) happen
# on the copy. The installed app is never touched, so it is safe to run while FROST is running.
#
# Size note: the models are NOT bundled. They live in ~/Library/Application Support/FROST and are
# fetched by bootstrap.sh, so the dmg is roughly the size of the bundle itself: a few tens of MB
# of binaries and dylibs plus the 129 MB mlx.metallib (UDZO zlib compression takes some off).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$HOME/Applications/FROST.app}"
OUT="${2:-$ROOT/dist}"
ICNS="$ROOT/native/icon/FROST.icns"

[ -d "$APP/Contents" ] || { echo "not an app bundle: $APP" >&2; exit 1; }
[ -f "$ICNS" ] || { echo "missing $ICNS; run: sh tools/make_icon.sh" >&2; exit 1; }
for t in cp plutil codesign hdiutil; do command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 1; }; done

STAGE=""; MNT=""
cleanup() {
  [ -n "$MNT" ] && [ -d "$MNT" ] && hdiutil detach "$MNT" -quiet 2>/dev/null || true
  [ -n "$MNT" ] && rmdir "$MNT" 2>/dev/null || true
  [ -n "$STAGE" ] && rm -rf "$STAGE"
}
trap cleanup EXIT

# 1. Stage a copy. BSD cp -R copies symlinks as symlinks (Frameworks/mlx.metallib must stay one:
#    MLX looks for the metallib next to libmlx.dylib, codesign refuses data under Frameworks/).
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/frost-dmg.XXXXXX")"
cp -R "$APP" "$STAGE/FROST.app"
C="$STAGE/FROST.app/Contents"
[ -L "$C/Frameworks/mlx.metallib" ] || { echo "staging lost the mlx.metallib symlink" >&2; exit 1; }

# 2. Inject the icon if the installed bundle predates it, then re-sign ad-hoc inside out so the
#    bundle seal covers the new Resources/ entry and Info.plist.
if [ ! -f "$C/Resources/FROST.icns" ]; then
  echo "injecting icon into staged copy"
  cp "$ICNS" "$C/Resources/FROST.icns"
  plutil -replace CFBundleIconFile -string FROST "$C/Info.plist"
  for f in "$C"/Frameworks/*.dylib "$C/Helpers/frost"; do codesign --force --sign - "$f" 2>/dev/null; done
  codesign --force --sign - "$STAGE/FROST.app" 2>/dev/null
fi
codesign --verify --deep --strict "$STAGE/FROST.app"

# 3. Drag-to-install target and version.
ln -s /Applications "$STAGE/Applications"
VERSION="$(plutil -extract CFBundleShortVersionString raw -o - "$C/Info.plist")"
DMG="$OUT/FROST-$VERSION.dmg"
mkdir -p "$OUT"

# 4. Build, verify checksum, then mount read-only and check the payload as a user would see it.
hdiutil create -volname FROST -srcfolder "$STAGE" -ov -format UDZO -fs HFS+ "$DMG" >/dev/null
hdiutil verify "$DMG" >/dev/null
MNT="$(mktemp -d "${TMPDIR:-/tmp}/frost-mnt.XXXXXX")"
hdiutil attach -readonly -nobrowse -mountpoint "$MNT" "$DMG" >/dev/null
codesign --verify --deep --strict "$MNT/FROST.app"
[ -L "$MNT/Applications" ] || { echo "Applications symlink missing in dmg" >&2; exit 1; }
[ -f "$MNT/FROST.app/Contents/Resources/FROST.icns" ] || { echo "FROST.icns missing in dmg" >&2; exit 1; }
[ -L "$MNT/FROST.app/Contents/Frameworks/mlx.metallib" ] || { echo "mlx.metallib symlink lost in dmg" >&2; exit 1; }
hdiutil detach "$MNT" -quiet; rmdir "$MNT"; MNT=""

echo "$DMG"
echo "$(stat -f %z "$DMG") bytes"
