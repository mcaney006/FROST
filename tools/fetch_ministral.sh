#!/bin/sh
# Pinned, resumable, digest-verified download of the FROST generator checkpoint.
# Native tools only (curl, shasum). No Python. Safe to re-run: verified files are skipped.
set -eu

REPO="mlx-community/Ministral-3-8B-Instruct-2512-4bit"
REV="182f003f01daa75f9de0f2c4d379722fd0bc1c61"            # pinned HF revision (2025-12-06)
DEST="${FROST_MODELS_DIR:-$HOME/Library/Application Support/FROST/models}/Ministral-3-8B-Instruct-2512-4bit"
BASE="https://huggingface.co/$REPO/resolve/$REV"

# name  bytes  sha256 ("-" = upstream publishes no LFS digest; the local hash is recorded as LOCAL)
# Deliberately NOT fetched: tekken.json (duplicate tokenizer format), processor_config.json (vision),
# model.safetensors.index.json (stale: lists FP8 tensor names / 10.4 GB that do not match the shards).
FILES='model-00001-of-00002.safetensors 5294694324 0b7c0de8f4da647f054095add6d93c1cfdd7f7f38128ecb666ebace634169aa6
model-00002-of-00002.safetensors 301990235 aa49bdcf4394ab16f0cbd45b750313978073f514da8b56e802ee3c7568b3a865
tokenizer.json 17077402 286acad9b0e27fce778ac429763536accf618ccb6ed72963b6f94685e531c5c7
config.json 1986 -
generation_config.json 131 -
tokenizer_config.json 21188 -
chat_template.jinja 7759 -
params.json 1185 -'

say() { printf '[fetch] %s\n' "$1"; }
size_of() { stat -f %z "$1"; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

mkdir -p "$DEST"
MANIFEST="$DEST/MANIFEST.json"
TMP_MANIFEST="$MANIFEST.tmp"
printf '{\n  "repo": "%s",\n  "revision": "%s",\n  "files": [\n' "$REPO" "$REV" > "$TMP_MANIFEST"
first=1

printf '%s\n' "$FILES" | while read -r name bytes sha; do
  [ -n "$name" ] || continue
  final="$DEST/$name"; part="$final.part"
  if [ -f "$final" ] && [ "$(size_of "$final")" = "$bytes" ] && [ -f "$final.sha256" ]; then
    got=$(cat "$final.sha256")
    if [ "$sha" = "-" ] || [ "$got" = "$sha" ]; then say "ok      $name"; else say "REVERIFY $name"; rm -f "$final.sha256"; fi
  fi
  if [ ! -f "$final.sha256" ]; then
    say "fetch   $name ($bytes bytes)"
    # -C - resumes a partial .part; --fail surfaces HTTP errors; retries cover flaky links.
    curl -L --fail --retry 5 --retry-delay 3 -C - -o "$part" "$BASE/$name" || { rm -f "$part"; say "FAILED  $name"; exit 1; }
    got_bytes=$(size_of "$part")
    [ "$got_bytes" = "$bytes" ] || { say "SIZE MISMATCH $name: $got_bytes != $bytes"; rm -f "$part"; exit 1; }
    got=$(sha_of "$part")
    if [ "$sha" != "-" ] && [ "$got" != "$sha" ]; then say "DIGEST MISMATCH $name"; rm -f "$part"; exit 1; fi
    mv -f "$part" "$final"
    printf '%s\n' "$got" > "$final.sha256"
    say "verified $name"
  fi
  got=$(cat "$final.sha256")
  if [ "$sha" = "-" ]; then src="LOCAL"; else src="UPSTREAM_LFS"; fi
  [ $first -eq 1 ] || printf ',\n' >> "$TMP_MANIFEST"
  first=0
  printf '    {"name": "%s", "bytes": %s, "sha256": "%s", "digest_source": "%s"}' "$name" "$bytes" "$got" "$src" >> "$TMP_MANIFEST"
done
# `while` ran in a subshell; re-derive completion from the marker files rather than shell state.
missing=0
printf '%s\n' "$FILES" | while read -r name bytes sha; do [ -f "$DEST/$name.sha256" ] || exit 1; done || missing=1
[ $missing -eq 0 ] || { rm -f "$TMP_MANIFEST"; say "incomplete"; exit 1; }
printf '\n  ]\n}\n' >> "$TMP_MANIFEST"
mv -f "$TMP_MANIFEST" "$MANIFEST"
say "complete: $DEST"
