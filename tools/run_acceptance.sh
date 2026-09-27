#!/bin/sh
# Release-backed acceptance: real models, real installed binaries, one model process at a time.
# Writes verification/results.json (machine-readable) and verification/acceptance.log.
# Usage: tools/run_acceptance.sh [--skip-eval]   (evals take ~5-15 min per fixture)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Applications/FROST.app"
CLI="$APP/Contents/Helpers/frost"
OUT="$ROOT/verification"; mkdir -p "$OUT/coding_eval"
LOG="$OUT/acceptance.log"; : > "$LOG"
RES="$OUT/results.json"
SKIP_EVAL=0; [ "${1:-}" = "--skip-eval" ] && SKIP_EVAL=1
results=""
add() { # name status detail
  d=$(printf '%s' "$3" | tr '\n' ' ' | tr -d '\000-\037' | sed 's/\\/\\\\/g; s/"/\\"/g' | cut -c1-400)  # JSON-safe: no control chars (ANSI codes), escaped backslashes and quotes
  results="$results{\"check\": \"$1\", \"status\": \"$2\", \"detail\": \"$d\"},"
  printf '%-6s %-40s %s\n' "$2" "$1" "$d" | tee -a "$LOG"
}
run_check() { # name command...   (PASS on exit 0, FAIL otherwise; last 2 lines as detail)
  name="$1"; shift
  out=$("$@" 2>&1); rc=$?
  printf '### %s (rc=%s)\n%s\n' "$name" "$rc" "$out" >> "$LOG"
  if [ $rc -eq 0 ]; then add "$name" PASS "$(printf '%s' "$out" | tail -2)"; else add "$name" FAIL "$(printf '%s' "$out" | tail -3)"; fi
}
no_model_running() { if pgrep -f 'FROST.app/Contents/MacOS/FROST|frost-desktop' >/dev/null; then add "$1" BLOCKED "a FROST app process is running; one model process at a time"; return 1; fi; return 0; }

cd "$ROOT" || exit 1
echo "FROST acceptance — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "machine: $(sysctl -n hw.model) $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB, macOS $(sw_vers -productVersion)" | tee -a "$LOG"

# 1. unit suites (no model)
run_check "unit tests (workspace)" cargo test --release --workspace --quiet
# 2. mandatory model-backed suites, serialized
if no_model_running "acceptance: generator/encoder/engine"; then
  run_check "acceptance: generator/encoder/engine/repo" cargo test --release -p frost-gen -p frost-model -p frost-chat -p frost-repo --quiet -- --ignored --test-threads=1
fi
# 3. native kernels: Mojo required, parity vs reference
FROST_REQUIRE_MOJO=1 run_check "mojo kernels (required, parity)" cargo test --release -p frost-kernels --quiet
run_check "zig index + q4 validation" cargo test --release -p frost-index --quiet
run_check "tools/runner/patch/fixtures" cargo test --release -p frost-tools --quiet
# 4. audits
run_check "python audit" sh tools/python_audit.sh
run_check "offline audit (no network crates; 0 sockets during generation)" sh tools/offline_audit.sh "$CLI"
# 5. installed product
if [ -x "$CLI" ]; then
  run_check "installed CLI diagnose" "$CLI" diagnose --json
  if otool -L "$APP"/Contents/MacOS/* "$APP"/Contents/Frameworks/*.dylib | grep -q '/opt/homebrew\|site-packages'; then add "installed bundle self-contained" FAIL "references /opt/homebrew or a Python env"; else add "installed bundle self-contained" PASS "all Mach-O deps via @rpath into Contents/Frameworks"; fi
  run_check "installed bundle signature" codesign --verify --deep --strict "$APP"
  if no_model_running "installed app headless check"; then run_check "installed app headless check" "$APP/Contents/MacOS/FROST" --headless-check; fi
  if no_model_running "installed CLI generation"; then run_check "installed CLI generation (--load)" "$CLI" diagnose --load; fi
else
  add "installed CLI diagnose" FAIL "not installed at $CLI (run bootstrap.sh)"
fi
# 6. coding evaluation (model quality is recorded, not asserted)
if [ $SKIP_EVAL -eq 0 ] && [ -x "$CLI" ]; then
  for fx in rust-slugify zig-rle c-parse-kv; do
    if no_model_running "coding eval: $fx"; then
      out=$("$CLI" eval --fixture "$ROOT/fixtures/$fx" --out "$OUT/coding_eval" 2>&1); rc=$?
      printf '### eval %s (rc=%s)\n%s\n' "$fx" "$rc" "$out" >> "$LOG"
      line=$(printf '%s' "$out" | grep "^$fx:" | tail -1)
      case "$line" in *"plumbing PASS"*) add "coding eval plumbing: $fx" PASS "$line";; *) add "coding eval plumbing: $fx" FAIL "$(printf '%s' "$out" | tail -2)";; esac
      case "$line" in *"held-out PASS"*) add "coding eval quality: $fx" PASS "$line";; *) add "coding eval quality: $fx" FAIL "$line";; esac
    fi
  done
else add "coding eval" NOT_TESTED "skipped (--skip-eval or no installed CLI)"; fi
# 7. isolated uninstall/reinstall (throwaway HOME, models symlinked, no re-download)
# The Mojo SDK default path is HOME-relative; resolve it from the real HOME so the isolated install keeps the same toolchain.
MOJO_REAL="$HOME/.local/share/max-codex/max-nightly-env/lib/python3.12/site-packages/modular/bin/mojo"
[ -x "${FROST_MOJO_BIN:-$MOJO_REAL}" ] && export FROST_MOJO_BIN="${FROST_MOJO_BIN:-$MOJO_REAL}"
ISO="$(mktemp -d "${TMPDIR:-/tmp}/frost-iso.XXXXXX")"
mkdir -p "$ISO/Library/Application Support/FROST" "$ISO/Applications" "$ISO/.local/bin"
ln -s "$HOME/Library/Application Support/FROST/models" "$ISO/Library/Application Support/FROST/models"
if no_model_running "isolated install/uninstall/reinstall"; then
  out=$(HOME="$ISO" sh bootstrap.sh --skip-fetch --skip-tests 2>&1 && HOME="$ISO" sh uninstall.sh 2>&1 && [ ! -e "$ISO/Applications/FROST.app" ] && [ -e "$ISO/Library/Application Support/FROST/models" ] && HOME="$ISO" sh bootstrap.sh --skip-fetch --skip-tests 2>&1 && [ -x "$ISO/Applications/FROST.app/Contents/Helpers/frost" ]); rc=$?
  printf '### isolated install (rc=%s)\n%s\n' "$rc" "$out" >> "$LOG"
  if [ $rc -eq 0 ]; then add "isolated install/uninstall/reinstall" PASS "install → uninstall keeps models → reinstall ok (HOME=$ISO)"; else add "isolated install/uninstall/reinstall" FAIL "$(printf '%s' "$out" | tail -3)"; fi
fi
rm -rf "$ISO"

printf '{\n  "run_at": "%s",\n  "machine": "%s",\n  "checks": [%s]\n}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(sysctl -n hw.model) $(sysctl -n machdep.cpu.brand_string) macOS $(sw_vers -productVersion)" "${results%,}" > "$RES"
echo "results: $RES ; log: $LOG"
n=$(grep -o '"status": "FAIL"' "$RES" | wc -l | tr -d ' ')
if [ "$n" = "0" ]; then echo "ACCEPTANCE: no FAIL"; else echo "ACCEPTANCE: $n FAIL"; exit 1; fi
