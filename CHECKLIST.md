# FROST chatbot rebuild — execution checklist

Baseline audited: `0e873b3` (origin/main). Work branch: `frost-chat`. Machine verified: Mac17,2, Apple M5,
16 GB, 4P+6E, Metal 4, macOS 27.2, Command Line Tools only (no `metal` compiler; MLX ships a prebuilt metallib).

Status legend: [ ] todo · [~] in progress · [x] done · [B] blocked · [N] not tested

## 0. Ground truth
- [x] git state read (clean tree; local `frost-implementation` differed from merged main by 8 files → branched from main)
- [x] hardware / OS / toolchain versions recorded (Rust 1.98.1, Zig 0.16.0, mlx 0.32.1, mlx-c 0.6.0_4, clang 21)
- [x] checkpoint manifest inspected at pinned rev `182f003f…` (real shard headers read; `model.safetensors.index.json` is stale/FP8 and is NOT used)
- [x] Mojo distribution audit: pip `mojo` is a Python entrypoint over a native Mach-O compiler; direct native invocation verified Python-free (`verification/mojo_audit.md`)

## 1. Genuine headless generation (frost-gen)
- [x] pinned, resumable, digest-verified download (`tools/fetch_ministral.sh`) — 5.6 GB, upstream LFS sha256 matched
- [x] `frost-tensors`: hardened safetensors parser (checked arithmetic, typed errors, no panics on malformed input; 3 tests)
- [x] `frost-mlx`: RAII wrapper extended (quantized_matmul, rms_norm, rope+freqs, SDPA, take, slice_update, concat, memory APIs, error handler; 9 tests)
- [x] Ministral-3 text-only forward: quantized embed/lm_head, 34 blocks, GQA 32/8, YaRN RoPE, llama-4 attn scale, SwiGLU — coherent output, 25 tok/s
- [x] KV cache with prefix reuse + trim; chunked prefill; incremental decode; EOS/budget stop; cancellation; memory-admission check at load
- [x] tokenizer (pinned tokenizer.json via `tokenizers`), control tokens inserted by id (never parsed from text), ids verified at load
- [x] Mistral v13 chat template as token ids; incremental UTF-8 detokenizer; tool-call parsing by control id
- [x] sampler: Rust repetition penalty → Mojo top-k (0.044 ms) → Mojo softmax/top-p → draw; backend recorded per request
- [x] CLI: `frost chat` / `frost gen` streaming; `frost diagnose`/`models` report identity, backends, memory
- [x] acceptance tests that FAIL (not skip) when assets are missing (`--ignored` suites in frost-gen, frost-model, frost-chat, frost-repo; serialized; 8 tests)

## 2. Responsive native streaming chat (frost-app)
- [x] wry/tao/webview and `ui/index.html` removed; AppKit shell in Objective-C (`native/app/*.m`) over the C ABI
- [x] AppKit via a narrow Objective-C bridge (C ABI `native/app/frost_app.h`) + Rust core `frost-chat`; worker thread, bounded queue, 40 ms coalescing; streamed reply verified through the UI (23.3 tok/s)
- [x] sidebar, TextKit 2 transcript (Markdown subset, code blocks with language label + Copy button, diff colouring, attempt review cards), composer, inspector (`verification/ui/`)
- [~] menus/shortcuts implemented (New Chat, Open Repository, Cancel, Regenerate, Clear Context, Delete, Mode, Model Info, Quit); Mode+Send verified by System Events automation, the rest NOT TESTED by automation
- [x] loading + error states: composer disabled with "Loading model…", error detail shown, sidebar/inspector usable; `--headless-check`
- [x] installed app relaunched from LaunchServices (`open`, no terminal, no server) reopens the saved conversation with its transcript (`verification/ui/frost-4-finder-launch.png`)

## 3. Persistence + Nomic repository memory
- [x] `frost-store` (SQLite WAL, transactional; conversations/messages/chunks/embedding cache/attempts; 10 tests)
- [x] repo indexer (`frost-repo`): explicit repo only, gitignore + secret/binary/dep exclusions, structural chunks, sha256 digests, stat-scan refresh (10 tests + real-model test)
- [x] Zig mmap vector index with generation swap + SIMD search (378 µs / 10k×768); exhaustive reference in tests
- [x] embedding cache keyed by content digest + model recipe (defect B); pruned to 256 MiB LRU
- [x] retrieval wired into the engine: hybrid search before each turn, framed context block ("DATA, not instructions") on the model-facing user turn only, citations (path/lines/digest) re-verified after each reply, changed files re-indexed (engine acceptance test)

## 4. Approved patch/test workflow
- [x] typed tools: list/read/search/edit_file/propose_patch/run_command (`frost-tools`, 22 tests); read-only tools run immediately, others become attempts
- [x] approval gate (`frost_attempt_decide`), base-hash validation, stale-patch re-check at apply time, held-out files protected (all three fixtures' held-out files contain `held_out`)
- [x] bounded process runner (env allowlist, deadline 240 s, head+tail output cap 24 KiB, process-group SIGTERM→SIGKILL; 32–53 ms kill latency)
- [x] fixtures rust-slugify / zig-rle / c-parse-kv with held-out tests (fail before fix, pass after); ≤2 repairs, ≤14 tool rounds; `frost eval` runs the fixture build, then the test, and records every attempt

## 5. Substantive native kernels
- [x] Zig: q4 layout validation runs on every quantized tensor at load (aligned zero-copy view); vector index + SIMD retrieval; parity tests (10)
- [x] Mojo: top-k / softmax-top-p kernels (C ABI dylib, Python-free build) in the real per-token sampling path; parity tests 6/6
- [x] each request records which implementation actually executed (`sampler` in stats/meta)

## 6. Defects A–J from the audit
- [x] A single `block()` in frost-model; forward_upto(all) == forward asserted; debug ablation paths removed
- [x] B reusable embeddings keyed by content+recipe in SQLite; LRU byte cap; index generations pruned
- [x] C one 4096 budget for all modes (never hidden); modes set prefill chunk + Quiet duty cycle; truncation emitted as a note + meta
- [x] D no production override; `thermal-sim` feature/test-only injection; admission check at request start, governor on prefill chunks and every 500 ms of decode; memory-pressure source
- [x] E inference on the `frost-generator` worker thread; UI gets coalesced events
- [x] F hardened safetensors in both models; tokenizer control ids and vocab size verified against the pinned file
- [x] G the ranking engine and its "calibration" gate were removed; retrieval scores are labelled similarity/rank fusion only
- [x] H the text-IPC Mojo helper is gone (dylib instead); backend label comes from the executed path
- [x] I weight-backed tests are `#[ignore]` acceptance tests that fail when assets are missing
- [x] J bootstrap.sh: staged bundle, vendored MLX/Mojo dylibs, ownership manifest, backup/rollback, verify-after-install — exercised end to end (`bootstrap-20260927-085549.log` and the final run)

## 7. Install + release verification
- [x] bootstrap installs missing permitted deps; vendors mlx dylibs into FROST.app (metallib in Resources with a Frameworks symlink); CLI at `Contents/Helpers/frost` (APFS is case-insensitive: `MacOS/frost` collided with `MacOS/FROST`); ad-hoc signed, `codesign --verify --deep --strict` passes; `~/.local/bin/frost` link
- [x] memory admission uses the kernel's `kern.memorystatus_level` (free + reclaimable cache); the old free+inactive sum refused a load at 4.1 GiB "available" while the kernel reported 8 GiB (`bootstrap-20260927-084147.log`, which also proves the refusal path)
- [x] acceptance run against installed binaries (`verification/results.json`, 2026-09-27T14:41:06Z): all plumbing, audit, install and lifecycle checks PASS; Python audit 5/5; offline audit 0 sockets; isolated install/uninstall/reinstall PASS
- [x] coding evals recorded apart from plumbing: plumbing PASS ×3; held-out quality c-parse-kv PASS after one repair, rust-slugify FAIL, zig-rle FAIL (model quality, no threshold)
- [x] docs rewritten to describe the chatbot; VERIFICATION with PASS/FAIL/BLOCKED/NOT TESTED and the measured numbers
