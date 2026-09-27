#!/bin/sh
# Pinned, resumable, digest-verified download of the retrieval encoder (nomic-embed-text-v1.5).
# Native tools only (curl, shasum). Safe to re-run: verified files are skipped.
set -eu
REPO="nomic-ai/nomic-embed-text-v1.5"
REV="e9b6763023c676ca8431644204f50c2b100d9aab"
DEST="${FROST_MODELS_DIR:-$HOME/Library/Application Support/FROST/models}/nomic-embed-text-v1.5"
BASE="https://huggingface.co/$REPO/resolve/$REV"
# name bytes sha256 ("-" = no upstream LFS digest; local hash recorded as LOCAL)
FILES='model.safetensors 546938168 9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c
config.json 2538 -
tokenizer.json 711396 -
tokenizer_config.json 1191 -
special_tokens_map.json 695 -
vocab.txt 231508 -'

say() { printf '[fetch-nomic] %s\n' "$1"; }
mkdir -p "$DEST"
printf '%s\n' "$FILES" | while read -r name bytes sha; do
  [ -n "$name" ] || continue
  final="$DEST/$name"; part="$final.part"
  if [ -f "$final" ] && [ "$(stat -f %z "$final")" = "$bytes" ]; then
    got=$(shasum -a 256 "$final" | cut -d' ' -f1)   # always rehash: a cached marker cannot be trusted
    if [ "$sha" = "-" ] || [ "$got" = "$sha" ]; then printf '%s\n' "$got" > "$final.sha256"; say "ok      $name"; continue; fi
    say "DIGEST MISMATCH $name — refetching"; rm -f "$final" "$final.sha256"
  fi
  say "fetch   $name ($bytes bytes)"
  curl -L --fail --retry 5 --retry-delay 3 -C - -o "$part" "$BASE/$name" || { rm -f "$part"; say "FAILED $name"; exit 1; }
  [ "$(stat -f %z "$part")" = "$bytes" ] || { say "SIZE MISMATCH $name"; rm -f "$part"; exit 1; }
  got=$(shasum -a 256 "$part" | cut -d' ' -f1)
  if [ "$sha" != "-" ] && [ "$got" != "$sha" ]; then say "DIGEST MISMATCH $name"; rm -f "$part"; exit 1; fi
  mv -f "$part" "$final"; printf '%s\n' "$got" > "$final.sha256"; say "verified $name"
done
printf '{"repo":"%s","revision":"%s"}\n' "$REPO" "$REV" > "$DEST/MANIFEST.json"
rm -f "$DEST/.dl_done"
say "complete: $DEST"
