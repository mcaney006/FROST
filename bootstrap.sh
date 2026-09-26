#!/bin/sh
# FROST bootstrap: idempotent build → model → package → install → verify.
# Safe to re-run. Installs only under the user's home; writes an uninstall manifest.
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
HELPER_SRC="$ROOT/native/kernels/frost_mojo_helper.mojo"
APP_SUPPORT="$HOME/Library/Application Support/FROST"
MODEL_DIR="$APP_SUPPORT/models/nomic-embed-text-v1.5"
BIN_DIR="$APP_SUPPORT/bin"
APP="$HOME/Applications/FROST.app"
CLI_LINK="$HOME/.local/bin/frost"
MANIFEST="$APP_SUPPORT/install-manifest.txt"
REV="e9b6763023c676ca8431644204f50c2b100d9aab"
BASE="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/$REV"

say() { printf '\033[36m[frost]\033[0m %s\n' "$1"; }
die() { printf '\033[31m[frost] %s\033[0m\n' "$1" >&2; exit 1; }

say "1/8 checking toolchains"
for t in cargo zig mojo clang; do command -v "$t" >/dev/null || die "missing tool: $t"; done
[ -f /opt/homebrew/lib/libmlxc.dylib ] || die "mlx-c not found (brew install mlx-c)"

say "2/8 building Mojo kernel helper"
mkdir -p "$BIN_DIR"
MOJO_LOG="${TMPDIR:-/tmp}/frost-mojo-build.$$.log"
if mojo build "$HELPER_SRC" -o "$BIN_DIR/frost_mojo_helper" >"$MOJO_LOG" 2>&1; then
  grep -vi crashpad "$MOJO_LOG" || true
else
  grep -vi crashpad "$MOJO_LOG" >&2 || true
  rm -f "$MOJO_LOG"
  die "mojo helper build failed"
fi
rm -f "$MOJO_LOG"
[ -x "$BIN_DIR/frost_mojo_helper" ] || die "mojo helper build failed"

say "3/8 building release binaries (Rust + Zig via cargo)"
( cd "$ROOT" && cargo build --release -p frost-cli -p frost-desktop >/dev/null 2>&1 ) || die "cargo build failed"

say "4/8 ensuring model weights (resumable, pinned rev $REV)"
mkdir -p "$MODEL_DIR"
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt; do
  [ -s "$MODEL_DIR/$f" ] || curl -fsSL -o "$MODEL_DIR/$f" "$BASE/$f"
done
if [ ! -s "$MODEL_DIR/model.safetensors" ]; then
  say "    downloading model.safetensors (~522 MiB)…"
  curl -fSL -C - -o "$MODEL_DIR/model.safetensors.part" "$BASE/model.safetensors"
  mv "$MODEL_DIR/model.safetensors.part" "$MODEL_DIR/model.safetensors"
fi
# record a locally-computed digest (provenance; not an independent authenticity check)
shasum -a 256 "$MODEL_DIR/model.safetensors" > "$MODEL_DIR/model.safetensors.sha256" || true

say "5/8 bounded tests (release-backed where possible)"
( cd "$ROOT" && cargo test --release -p frost-core -p frost-index -p frost-platform >/dev/null 2>&1 ) || die "core tests failed"

say "6/8 assembling FROST.app"
CT="$APP/Contents"
rm -rf "$APP"
mkdir -p "$CT/MacOS" "$CT/Resources"
cp "$ROOT/target/release/frost-desktop" "$CT/MacOS/FROST"
cp "$BIN_DIR/frost_mojo_helper" "$CT/Resources/frost_mojo_helper"
cat > "$CT/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>FROST</string>
  <key>CFBundleDisplayName</key><string>FROST</string>
  <key>CFBundleIdentifier</key><string>com.frost.decision</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>FROST</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
# ad-hoc sign (local only; NOT Developer ID / notarized)
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || say "    (ad-hoc signing skipped)"

say "7/8 installing CLI → $CLI_LINK (copy, project-independent)"
cp "$ROOT/target/release/frost" "$BIN_DIR/frost"
mkdir -p "$(dirname "$CLI_LINK")"
ln -sf "$BIN_DIR/frost" "$CLI_LINK"

say "8/8 writing manifest + verifying"
{ echo "$APP"; echo "$CLI_LINK"; echo "$BIN_DIR/frost"; echo "$BIN_DIR/frost_mojo_helper"; } > "$MANIFEST"
FROST_MOJO_HELPER="$BIN_DIR/frost_mojo_helper" "$ROOT/target/release/frost" diagnose || die "diagnose failed"

cat <<DONE

\033[32m[frost] install complete\033[0m
  App : $APP   (open with: open "$APP")
  CLI : $CLI_LINK   (ensure ~/.local/bin is on PATH)
  Data: $APP_SUPPORT
  Uninstall: sh "$ROOT/uninstall.sh"
DONE
