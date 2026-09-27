#!/bin/sh
# FROST bootstrap: verify prerequisites (installing only what is missing and permitted), fetch the
# pinned models, build, run the mandatory tests, stage a self-contained FROST.app, verify the
# staged bundle, then install it with backup/rollback under an ownership manifest.
#
#   sh bootstrap.sh [--skip-fetch] [--skip-tests] [--launch] [--force]
#
# Never uses Python. Never touches system settings, SIP, Gatekeeper, fan/thermal/power controls.
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP_SUPPORT="$HOME/Library/Application Support/FROST"
MODELS="$APP_SUPPORT/models"
APP_DIR="$HOME/Applications"
APP="$APP_DIR/FROST.app"
CLI_LINK="$HOME/.local/bin/frost"
MANIFEST="$APP_SUPPORT/install-manifest.json"
LEGACY_MANIFEST="$APP_SUPPORT/install-manifest.txt"
LOG_DIR="$APP_SUPPORT/logs"; mkdir -p "$LOG_DIR" "$APP_SUPPORT/backup"
LOG="$LOG_DIR/bootstrap-$(date +%Y%m%d-%H%M%S).log"
MOJO_DEFAULT="$HOME/.local/share/max-codex/max-nightly-env/lib/python3.12/site-packages/modular/bin/mojo"
export FROST_MOJO_BIN="${FROST_MOJO_BIN:-$MOJO_DEFAULT}"

SKIP_FETCH=0; SKIP_TESTS=0; LAUNCH=0; FORCE=0
for a in "$@"; do case "$a" in
  --skip-fetch) SKIP_FETCH=1;; --skip-tests) SKIP_TESTS=1;; --launch) LAUNCH=1;; --force) FORCE=1;;
  *) echo "unknown flag $a"; exit 2;; esac; done

say() { printf '\033[36m[frost]\033[0m %s\n' "$1" | tee -a "$LOG"; }
die() { printf '\033[31m[frost] FAIL:\033[0m %s\n' "$1" | tee -a "$LOG"; exit 1; }
run() { "$@" >>"$LOG" 2>&1 || die "$* (see $LOG)"; }

# ---------------------------------------------------------------------------- 1. prerequisites
say "1/9 prerequisites (log: $LOG)"
[ "$(uname -m)" = "arm64" ] || die "Apple silicon required"
need_brew() { command -v brew >/dev/null || die "Homebrew is required to install $1 (https://brew.sh); not installing it silently"; }
install_if_missing() { # formula, command-or-file to test
  if [ -e "$2" ] || command -v "$2" >/dev/null 2>&1; then return 0; fi
  need_brew "$1"; say "    installing missing dependency: brew install $1"; run brew install "$1"
}
xcode-select -p >/dev/null 2>&1 || die "Command Line Tools missing: run xcode-select --install"
install_if_missing rust cargo
install_if_missing zig zig
install_if_missing mlx /opt/homebrew/opt/mlx/lib/libmlx.dylib
install_if_missing mlx-c /opt/homebrew/opt/mlx-c/lib/libmlxc.dylib
for t in cargo zig clang curl shasum codesign install_name_tool otool; do command -v "$t" >/dev/null || die "missing tool: $t"; done
say "    rust $(rustc --version | cut -d' ' -f2) · zig $(zig version) · mlx $(brew list --versions mlx 2>/dev/null | cut -d' ' -f2) · mlx-c $(brew list --versions mlx-c 2>/dev/null | cut -d' ' -f2) · clang $(clang --version | head -1 | cut -d' ' -f4)"
MOJO_OK=0
if [ -x "$FROST_MOJO_BIN" ] && file "$FROST_MOJO_BIN" | grep -q Mach-O; then MOJO_OK=1; say "    mojo native compiler: $("$FROST_MOJO_BIN" --version 2>/dev/null | head -1) (Python-free invocation; see verification/mojo_audit.md)"
else say "    mojo native compiler not found at $FROST_MOJO_BIN → sampling kernels will use the Rust reference (reported as such; not installing Python to get Mojo)"; fi

# ---------------------------------------------------------------------------- 2. models
if [ $SKIP_FETCH -eq 0 ]; then
  say "2/9 models (pinned revisions, resumable, digest-verified)"
  run sh "$ROOT/tools/fetch_ministral.sh"
  run sh "$ROOT/tools/fetch_nomic.sh"
else say "2/9 models: skipped (--skip-fetch)"; fi
[ -f "$MODELS/Ministral-3-8B-Instruct-2512-4bit/MANIFEST.json" ] || die "generator checkpoint missing"
[ -f "$MODELS/nomic-embed-text-v1.5/model.safetensors" ] || die "retrieval encoder missing"

# ---------------------------------------------------------------------------- 3. build
say "3/9 building (cargo, release; Zig via build.rs; Mojo dylib via the native compiler)"
run cargo build --release --manifest-path "$ROOT/Cargo.toml" -p frost-cli -p frost-desktop
KERNELS_DYLIB=$(ls -t "$ROOT"/target/release/build/frost-kernels-*/out/libfrost_kernels.dylib 2>/dev/null | head -1 || true)
MODULAR_LIB=""
if [ -n "$KERNELS_DYLIB" ]; then
  # The Mojo driver bakes an absolute rpath to the SDK lib/ dir into the dylib it built; vendor the runtime
  # dylibs from there (not from $HOME-relative guesses, which break under another HOME or a moved SDK).
  MODULAR_LIB="$(otool -l "$KERNELS_DYLIB" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}' | grep -v '^@' | head -1)"
  if [ -z "$MODULAR_LIB" ] || [ ! -f "$MODULAR_LIB/libKGENCompilerRTShared.dylib" ]; then
    say "    libfrost_kernels.dylib exists but its Modular runtime dir is missing ('$MODULAR_LIB'); not bundling the Mojo kernels"
    KERNELS_DYLIB=""
  fi
fi

# ---------------------------------------------------------------------------- 4. tests
if [ $SKIP_TESTS -eq 0 ]; then
  say "4/9 tests: unit suites"
  run cargo test --release --manifest-path "$ROOT/Cargo.toml" --workspace
  say "    mandatory acceptance (real models; serialized; FAIL if assets are missing)"
  run cargo test --release --manifest-path "$ROOT/Cargo.toml" -p frost-gen -p frost-model -p frost-chat -p frost-repo -- --ignored --test-threads=1
  if [ $MOJO_OK -eq 1 ]; then FROST_REQUIRE_MOJO=1 run cargo test --release --manifest-path "$ROOT/Cargo.toml" -p frost-kernels; fi
else say "4/9 tests: skipped (--skip-tests)"; fi

# ---------------------------------------------------------------------------- 5. stage the bundle
say "5/9 staging FROST.app"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/frost-stage.XXXXXX")"; trap 'rm -rf "$STAGE"' EXIT
S="$STAGE/FROST.app/Contents"; mkdir -p "$S/MacOS" "$S/Helpers" "$S/Frameworks" "$S/Resources"
cp "$ROOT/target/release/frost-desktop" "$S/MacOS/FROST"
cp "$ROOT/target/release/frost" "$S/Helpers/frost"  # Helpers/, not MacOS/: APFS is case-insensitive, so MacOS/frost would overwrite MacOS/FROST
cp /opt/homebrew/opt/mlx/lib/libmlx.dylib /opt/homebrew/opt/mlx-c/lib/libmlxc.dylib "$S/Frameworks/"
# The metallib is data, not code: codesign rejects it under Frameworks/. Keep it in Resources/ and
# leave a symlink next to libmlx.dylib, which is where MLX looks for it (colocated lookup).
cp /opt/homebrew/opt/mlx/lib/mlx.metallib "$S/Resources/mlx.metallib"; chmod u+w "$S/Resources/mlx.metallib"
ln -s ../Resources/mlx.metallib "$S/Frameworks/mlx.metallib"
if [ -n "$KERNELS_DYLIB" ]; then
  cp "$KERNELS_DYLIB" "$S/Frameworks/libfrost_kernels.dylib"
  for l in libKGENCompilerRTShared libAsyncRTMojoBindings libMSupportGlobals libAsyncRTRuntimeGlobals; do cp "$MODULAR_LIB/$l.dylib" "$S/Frameworks/"; done
fi
chmod -R u+w "$S/Frameworks"
cp "$ROOT/native/icon/FROST.icns" "$S/Resources/FROST.icns"  # app icon; regenerate with tools/make_icon.sh
VERSION="$(grep -m1 '^version' "$ROOT/Cargo.toml" | sed 's/.*"\(.*\)".*/\1/')"
cat > "$S/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>FROST</string>
  <key>CFBundleDisplayName</key><string>FROST</string>
  <key>CFBundleIdentifier</key><string>dev.frost.chat</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>FROST</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>FROST</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticTermination</key><false/>
  <key>NSHumanReadableCopyright</key><string>Local-only. Models: Mistral AI (Apache-2.0), Nomic AI (Apache-2.0).</string>
</dict></plist>
PLIST
# Rewrite install names so nothing resolves through /opt/homebrew or a Python environment.
for b in "$S/MacOS/FROST" "$S/Helpers/frost"; do
  install_name_tool -change /opt/homebrew/opt/mlx-c/lib/libmlxc.dylib @rpath/libmlxc.dylib \
                    -change /opt/homebrew/opt/mlx/lib/libmlx.dylib  @rpath/libmlx.dylib \
                    -add_rpath @executable_path/../Frameworks "$b" >>"$LOG" 2>&1
done
install_name_tool -id @rpath/libmlxc.dylib -change /opt/homebrew/opt/mlx/lib/libmlx.dylib @rpath/libmlx.dylib "$S/Frameworks/libmlxc.dylib" >>"$LOG" 2>&1
install_name_tool -id @rpath/libmlx.dylib "$S/Frameworks/libmlx.dylib" >>"$LOG" 2>&1
if [ -f "$S/Frameworks/libfrost_kernels.dylib" ]; then
  install_name_tool -id @rpath/libfrost_kernels.dylib "$S/Frameworks/libfrost_kernels.dylib" >>"$LOG" 2>&1
  for rp in $(otool -l "$S/Frameworks/libfrost_kernels.dylib" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}'); do
    case "$rp" in @loader_path*) ;; *) install_name_tool -delete_rpath "$rp" "$S/Frameworks/libfrost_kernels.dylib" >>"$LOG" 2>&1 || true;; esac
  done
  for l in "$S"/Frameworks/libKGENCompilerRTShared.dylib "$S"/Frameworks/libAsyncRTMojoBindings.dylib "$S"/Frameworks/libMSupportGlobals.dylib "$S"/Frameworks/libAsyncRTRuntimeGlobals.dylib; do
    otool -l "$l" | grep -q 'path @loader_path' || install_name_tool -add_rpath @loader_path "$l" >>"$LOG" 2>&1
  done
fi
# Ad-hoc signing, inside-out (local only: NOT Developer ID, NOT notarized; Gatekeeper untouched).
for f in "$S"/Frameworks/*.dylib "$S/Helpers/frost"; do run codesign --force --sign - "$f"; done  # MacOS/FROST is signed with the bundle below
run codesign --force --sign - "$STAGE/FROST.app"

# ---------------------------------------------------------------------------- 6. verify the staged bundle
say "6/9 verifying the staged bundle"
if otool -L "$S/MacOS/FROST" "$S/Helpers/frost" "$S"/Frameworks/*.dylib | grep -q '/opt/homebrew\|site-packages'; then die "staged bundle still references /opt/homebrew or a Python environment"; fi
run codesign --verify --deep --strict "$STAGE/FROST.app"
loaded=$(DYLD_PRINT_LIBRARIES=1 "$S/Helpers/frost" models 2>&1 | grep -c 'FROST.app/Contents/Frameworks/libmlx' || true)
[ "$loaded" -ge 1 ] || die "the staged CLI did not load the bundled libmlx (loaded=$loaded)"
run "$S/Helpers/frost" diagnose
say "    staged bundle loads bundled MLX; diagnose PASS"

# ---------------------------------------------------------------------------- 7. install with ownership check + backup/rollback
say "7/9 installing → $APP"
owned() { # path listed in the JSON or legacy manifest?
  { [ -f "$MANIFEST" ] && grep -Fq "\"$1\"" "$MANIFEST"; } || { [ -f "$LEGACY_MANIFEST" ] && grep -Fxq "$1" "$LEGACY_MANIFEST"; }
}
mkdir -p "$APP_DIR" "$(dirname "$CLI_LINK")"
BACKUP=""
if [ -e "$APP" ]; then
  if owned "$APP" || [ $FORCE -eq 1 ]; then
    BACKUP="$APP_SUPPORT/backup/FROST.app.$(date +%Y%m%d-%H%M%S)"; mv "$APP" "$BACKUP"; say "    previous app backed up to $BACKUP"
  else die "$APP exists but was not installed by this bootstrap (not in the manifest). Remove it yourself or pass --force."; fi
fi
if [ -L "$CLI_LINK" ] || [ -e "$CLI_LINK" ]; then
  tgt="$(readlink "$CLI_LINK" 2>/dev/null || true)"
  case "$tgt" in *FROST.app/Contents/Helpers/frost|*FROST.app/Contents/MacOS/frost|*"Application Support/FROST/bin/frost") ;; *)
    if ! owned "$CLI_LINK" && [ $FORCE -eq 0 ]; then die "$CLI_LINK exists and is not FROST's symlink; refusing to replace it"; fi;; esac
fi
rollback() { say "    ROLLBACK: restoring previous installation"; rm -rf "$APP"; [ -n "$BACKUP" ] && mv "$BACKUP" "$APP"; }
if ! mv "$STAGE/FROST.app" "$APP"; then rollback; die "could not move the app into place"; fi
if ! "$APP/Contents/Helpers/frost" diagnose >>"$LOG" 2>&1; then rollback; die "installed CLI failed diagnose; rolled back"; fi
ln -sfn "$APP/Contents/Helpers/frost" "$CLI_LINK"
# legacy files from the previous product (owned by the old manifest) are retired
for legacy in "$APP_SUPPORT/bin/frost" "$APP_SUPPORT/bin/frost_mojo_helper"; do
  if [ -e "$legacy" ] && owned "$legacy"; then rm -f "$legacy"; fi
done
rmdir "$APP_SUPPORT/bin" 2>/dev/null || true
# keep only the newest backup
ls -1dt "$APP_SUPPORT"/backup/FROST.app.* 2>/dev/null | tail -n +2 | xargs -I{} rm -rf {} 2>/dev/null || true

# ---------------------------------------------------------------------------- 8. manifest
say "8/9 writing manifest"
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
{
  printf '{\n  "version": "%s",\n  "installed_at": "%s",\n  "installer": "bootstrap.sh",\n' "$VERSION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "paths": ["%s", "%s"],\n' "$APP" "$CLI_LINK"
  printf '  "binaries": {"FROST": "%s", "frost": "%s"},\n' "$(sha "$APP/Contents/MacOS/FROST")" "$(sha "$APP/Contents/Helpers/frost")"
  printf '  "frameworks": ['; first=1; for f in "$APP"/Contents/Frameworks/*; do [ $first -eq 1 ] || printf ', '; first=0; printf '"%s"' "$(basename "$f")"; done; printf '],\n'
  printf '  "models": ["%s", "%s"],\n' "$MODELS/Ministral-3-8B-Instruct-2512-4bit" "$MODELS/nomic-embed-text-v1.5"
  printf '  "data": ["%s"],\n' "$APP_SUPPORT/frost.sqlite"
  printf '  "toolchain": {"rustc": "%s", "zig": "%s", "mlx": "%s", "mlx-c": "%s", "mojo": "%s"},\n' \
    "$(rustc --version | cut -d' ' -f2)" "$(zig version)" "$(brew list --versions mlx 2>/dev/null | cut -d' ' -f2)" "$(brew list --versions mlx-c 2>/dev/null | cut -d' ' -f2)" \
    "$([ $MOJO_OK -eq 1 ] && "$FROST_MOJO_BIN" --version 2>/dev/null | head -1 || echo unavailable)"
  printf '  "signing": "ad-hoc (local only; not Developer ID; not notarized)"\n}\n'
} > "$MANIFEST"
rm -f "$LEGACY_MANIFEST"

# ---------------------------------------------------------------------------- 9. verify the installed product
say "9/9 verifying the installed product"
run "$APP/Contents/Helpers/frost" diagnose --json
"$APP/Contents/MacOS/FROST" --headless-check >"$LOG_DIR/headless-check.json" 2>>"$LOG" || die "installed app failed --headless-check (see $LOG)"
say "    app headless check: $(grep -o '"state":"[a-z]*"' "$LOG_DIR/headless-check.json" | head -1)"
"$APP/Contents/Helpers/frost" mode quiet --if-unset >>"$LOG" 2>&1 || true
if [ $LAUNCH -eq 1 ]; then open "$APP"; say "    launched $APP (Quiet mode)"; fi

cat <<DONE

$(printf '\033[32m[frost] install complete\033[0m')
  App      : $APP            (open "$APP")
  CLI      : $CLI_LINK  → $APP/Contents/Helpers/frost
  Data     : $APP_SUPPORT   (frost.sqlite, models/, logs/, backup/)
  Manifest : $MANIFEST
  Uninstall: sh "$ROOT/uninstall.sh" [--purge]
DONE
