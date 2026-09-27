#!/bin/sh
# Offline audit: (1) no network-client crate in the dependency graph, (2) no sockets opened by a
# real generation. Usage: tools/offline_audit.sh [path-to-frost-cli]  (defaults to the installed CLI)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLI="${1:-$HOME/Applications/FROST.app/Contents/Helpers/frost}"
fail=0
ok()  { printf 'PASS %-30s %s\n' "$1" "$2"; }
bad() { printf 'FAIL %-30s %s\n' "$1" "$2"; fail=1; }

if (cd "$ROOT" && cargo tree --workspace -e normal 2>/dev/null | grep -Eiq '\b(reqwest|hyper|ureq|curl|hf-hub|tokio|isahc|attohttpc|rustls|native-tls|openssl)\b'); then
  bad "network crates" "an HTTP/TLS client crate is linked into the runtime graph"
else ok "network crates" "no reqwest/hyper/ureq/curl/hf-hub/tls crates in runtime deps"; fi

if [ -x "$CLI" ]; then
  out=$(mktemp)
  "$CLI" gen --max 24 --prompt "Say hello." >"$out" 2>&1 & pid=$!
  socks=0
  while kill -0 "$pid" 2>/dev/null; do
    n=$(lsof -a -p "$pid" -i 2>/dev/null | grep -vc COMMAND); [ "$n" -gt "$socks" ] && socks=$n
    sleep 0.2
  done
  wait "$pid"; rc=$?
  if [ "$rc" -ne 0 ]; then bad "generation offline" "CLI exited $rc: $(tail -2 "$out" | tr '\n' ' ')"
  elif [ "$socks" -gt 0 ]; then bad "generation offline" "$socks internet socket(s) observed during generation"
  else ok "generation offline" "0 internet sockets during a real generation ($(grep -o '[0-9.]* tok/s' "$out" | tail -1))"; fi
  rm -f "$out"
else ok "generation offline" "CLI not found at $CLI (skipped)"; fi

[ $fail -eq 0 ] && echo "OFFLINE AUDIT: PASS" || { echo "OFFLINE AUDIT: FAIL"; exit 1; }
