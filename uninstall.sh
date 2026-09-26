#!/bin/sh
# FROST uninstall: removes ONLY files recorded in the install manifest.
# Preserves downloaded models and user data unless --purge is given.
set -eu
APP_SUPPORT="$HOME/Library/Application Support/FROST"
MANIFEST="$APP_SUPPORT/install-manifest.txt"
say(){ printf '\033[36m[frost]\033[0m %s\n' "$1"; }
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST — nothing to uninstall"; exit 0; }
while IFS= read -r p; do
  [ -n "$p" ] || continue
  if [ -e "$p" ] || [ -L "$p" ]; then say "removing $p"; rm -rf "$p"; fi
done < "$MANIFEST"
rm -f "$MANIFEST"
if [ "${1:-}" = "--purge" ]; then
  say "purging models + data at $APP_SUPPORT"; rm -rf "$APP_SUPPORT"
else
  say "kept models + data at $APP_SUPPORT (use --purge to remove)"
fi
say "uninstalled FROST-owned files only"
