#!/bin/sh
# Render the FROST icon and build native/icon/FROST.icns.
# Usage: sh tools/make_icon.sh
# Needs only Xcode command line tools: swift (or swiftc), sips, iconutil.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/native/icon/frost_icon.swift"
ICNS="$ROOT/native/icon/FROST.icns"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/frost-icon.XXXXXX")"; trap 'rm -rf "$WORK"' EXIT
PNG="$WORK/icon_1024.png"
SET="$WORK/FROST.iconset"

# 1. Render the 1024 px master. `swift file.swift` interprets; fall back to swiftc if
#    scripting is unavailable (some CLT-only installs).
if ! swift "$SRC" "$PNG" 2>"$WORK/swift.err"; then
  echo "swift scripting failed, compiling with swiftc" >&2
  swiftc -O "$SRC" -o "$WORK/frost_icon"
  "$WORK/frost_icon" "$PNG"
fi

# 2. Downsample into the iconset Apple expects (each @2x is the next size up).
mkdir "$SET"
for s in 16 32 128 256 512; do
  d=$((s * 2))
  sips -z "$s" "$s" "$PNG" --out "$SET/icon_${s}x${s}.png" >/dev/null
  sips -z "$d" "$d" "$PNG" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
done

# 3. Pack.
iconutil -c icns "$SET" -o "$ICNS"

# 4. Verify by round-tripping the icns back to an iconset and listing every size.
iconutil -c iconset "$ICNS" -o "$WORK/check.iconset"
n=0
for f in "$WORK"/check.iconset/*.png; do
  w=$(sips -g pixelWidth "$f" | awk '/pixelWidth/{print $2}')
  echo "  $(basename "$f") ${w}px"
  n=$((n + 1))
done
[ "$n" -eq 10 ] || { echo "expected 10 iconset entries, got $n" >&2; exit 1; }
echo "wrote $ICNS ($(stat -f %z "$ICNS") bytes)"
