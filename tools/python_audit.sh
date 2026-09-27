#!/bin/sh
# Python audit for the FROST workflow, phase by phase. Prints PASS/FAIL lines and exits non-zero
# on any FAIL. Usage: tools/python_audit.sh [--with-build]   (--with-build re-runs a cargo build
# while sampling the process table for python — process-execution evidence, ~1-3 min).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${FROST_APP:-$HOME/Applications/FROST.app}"
fail=0
ok()  { printf 'PASS %-34s %s\n' "$1" "$2"; }
bad() { printf 'FAIL %-34s %s\n' "$1" "$2"; fail=1; }

# 1. Dependency graph: no Python bindings or interpreters in any Rust dependency.
if (cd "$ROOT" && cargo tree --workspace -e normal,build 2>/dev/null | grep -Eiq 'pyo3|cpython|python3-sys|inline-python|rustpython'); then
  bad "cargo dependency graph" "python-related crate present"
else ok "cargo dependency graph" "no pyo3/cpython/python crates (normal + build deps)"; fi

# 2. Scripts and build.rs never invoke python/pip/uv/conda/pixi. Command position only (start of
#    line or after ; & | ( or a backtick), so prose that mentions Python does not trip it.
hits=$(grep -nE '(^|[;&|`(]) *(python[0-9.]*|pip3?|uv|conda|pixi|virtualenv) ' "$ROOT"/bootstrap.sh "$ROOT"/uninstall.sh "$ROOT"/tools/*.sh 2>/dev/null | grep -v '^[^:]*:[0-9]*: *#' || true)
hits2=$(grep -nE 'Command::new\("(python[0-9.]*|pip3?|uv|conda|pixi)"' "$ROOT"/crates/*/build.rs 2>/dev/null || true)
if [ -n "$hits$hits2" ]; then bad "scripts/build.rs invocations" "$hits $hits2"; else ok "scripts/build.rs invocations" "no python/pip/uv/conda/pixi commands in scripts or build.rs"; fi

# 3. The Mojo compiler the build invokes is the native Mach-O binary (not the venv's Python script)
#    and links no libpython. (Its path lives inside a venv; the first otool line is its own path.)
MOJO="${FROST_MOJO_BIN:-$HOME/.local/share/max-codex/max-nightly-env/lib/python3.12/site-packages/modular/bin/mojo}"
if [ -x "$MOJO" ]; then
  if file "$MOJO" | grep -q 'Mach-O'; then
    if otool -L "$MOJO" | tail -n +2 | grep -qi libpython; then bad "mojo compiler binary" "links libpython"; else ok "mojo compiler binary" "native Mach-O, links no libpython"; fi
  else bad "mojo compiler binary" "not a native binary: $(file "$MOJO" | cut -c1-80)"; fi
else ok "mojo compiler binary" "absent: kernels fall back to the Rust reference (reported, not hidden)"; fi

# 4. Runtime linkage: every Mach-O we ship links no libpython.
found=""
for f in "$ROOT"/target/release/frost "$ROOT"/target/release/frost-desktop "$APP"/Contents/MacOS/* "$APP"/Contents/Frameworks/*.dylib; do
  [ -f "$f" ] || continue
  if otool -L "$f" 2>/dev/null | tail -n +2 | grep -qi libpython; then found="$found $f"; fi
done
if [ -n "$found" ]; then bad "runtime linkage" "libpython linked by:$found"; else ok "runtime linkage" "no libpython in CLI, app, or bundled dylibs"; fi

# 5. Bundled paths: nothing in the app resolves through a Python environment.
if [ -d "$APP" ]; then
  if otool -l "$APP"/Contents/MacOS/* "$APP"/Contents/Frameworks/*.dylib 2>/dev/null | grep -A2 LC_RPATH | grep -q 'site-packages\|max-nightly-env'; then
    bad "bundled rpaths" "an rpath points into a Python environment"
  else ok "bundled rpaths" "no rpath into site-packages / venv"; fi
else ok "bundled rpaths" "app not installed yet (skipped)"; fi

# 6. Optional: process-execution evidence during a build.
if [ "${1:-}" = "--with-build" ]; then
  log=$(mktemp)
  ( while :; do pgrep -lf '[p]ython' >> "$log" 2>/dev/null; sleep 0.05; done ) & sampler=$!
  (cd "$ROOT" && touch native/kernels/frost_sampling.mojo && cargo build --release -p frost-kernels -p frost-cli >/dev/null 2>&1); rc=$?
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  mine=$(grep -c 'frost\|mojo\|cargo' "$log" 2>/dev/null || echo 0)
  if [ "$rc" -ne 0 ]; then bad "build-time process sampling" "build failed"
  elif [ "$mine" -gt 0 ]; then bad "build-time process sampling" "python process related to the build observed"
  else ok "build-time process sampling" "no python process spawned by the build (50 ms sampler; unrelated python processes ignored)"; fi
  rm -f "$log"
fi

if [ $fail -eq 0 ]; then echo "PYTHON AUDIT: PASS"; else echo "PYTHON AUDIT: FAIL"; exit 1; fi
