#!/bin/sh
# Remove FROST. Only paths recorded in the install manifest are touched; models, the chat
# database and logs are kept unless --purge is given. Never touches anything outside FROST's own
# locations.  Usage: sh uninstall.sh [--purge]
set -eu
APP_SUPPORT="$HOME/Library/Application Support/FROST"
MANIFEST="$APP_SUPPORT/install-manifest.json"
LEGACY="$APP_SUPPORT/install-manifest.txt"
say() { printf '\033[36m[frost]\033[0m %s\n' "$1"; }
allowed() { case "$1" in
  "$HOME/Applications/FROST.app"|"$HOME/.local/bin/frost"|"$APP_SUPPORT/bin/frost"|"$APP_SUPPORT/bin/frost_mojo_helper") return 0;; *) return 1;; esac; }
remove() { if allowed "$1"; then if [ -e "$1" ] || [ -L "$1" ]; then say "removing $1"; rm -rf "$1"; fi; else say "ignoring unrecognized manifest path: $1"; fi; }

if [ -f "$MANIFEST" ]; then
  # paths array: "..." entries on the "paths" line
  grep -o '"paths": \[[^]]*\]' "$MANIFEST" | grep -o '"[^"]*"' | tr -d '"' | grep -v '^paths$' | while IFS= read -r p; do remove "$p"; done
  rm -f "$MANIFEST"
elif [ -f "$LEGACY" ]; then
  while IFS= read -r p; do [ -n "$p" ] && remove "$p"; done < "$LEGACY"; rm -f "$LEGACY"
else
  say "no manifest at $MANIFEST — nothing to uninstall"; exit 0
fi
rm -rf "$APP_SUPPORT/backup" "$APP_SUPPORT/bin" 2>/dev/null || true
if [ "${1:-}" = "--purge" ]; then
  say "purging models, chat database, indexes and logs under $APP_SUPPORT"
  rm -rf "$APP_SUPPORT"
else
  say "kept models, chat database and logs in $APP_SUPPORT (use --purge to remove)"
fi
say "done"
