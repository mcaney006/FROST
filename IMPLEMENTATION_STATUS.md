# FROST implementation status

State of the `frost-chat` branch (uncommitted work on top of `0e873b3`). The earlier product, a
state-to-action ranking engine, has been removed; this file describes only the chatbot.

Labels: **PASS** verified on this machine (Mac17,2, Apple M5, 16 GB, macOS 27.2) during
development, by a test in the tree or a measured run; **FAIL**; **BLOCKED** (outside this
project's control); **NOT TESTED** (built, not exercised); **NOT IMPLEMENTED**. Final acceptance
results live in [VERIFICATION.md](VERIFICATION.md).

## Features

| Area | Feature | Status | Evidence / note |
|---|---|---|---|
| Generator | Ministral-3 text decoder on MLX (4-bit, 34 layers, GQA, YaRN, SwiGLU) | PASS | `frost-gen` acceptance tests: coherent answers, 25 tok/s |
| Generator | Pinned revision, architecture, quantization and vocab checks at load | PASS | `Generator::load`, `Config::from_json` |
| Generator | Zig q4 layout validation of all 240 quantized tensors at load | PASS | `layout_check` in reply meta |
| Generator | Memory admission check before load | PASS (refusal path) | `bootstrap-20260927-084147.log` (in `~/Library/Application Support/FROST/logs/`): "5322 MiB needed (weights + working set), 4095 MiB available"; engine status `error`, sends refused, tests failed loudly. That figure came from the old free + inactive + purgeable + speculative formula while the kernel reported 50% free; available memory now reads `kern.memorystatus_level` |
| Generator | KV cache in 256-token slabs, prefix reuse, trim on regenerate | PASS | 89 of 99 tokens reused on turn two |
| Generator | Chunked prefill, cancellation, refusal over the 4096 budget | PASS | `frost-gen` acceptance tests |
| Tokenizer | Pinned `tokenizer.json`, 17 control ids verified, special tokens never parsed from text | PASS | `Tokenizer::load`; unit tests |
| Tokenizer | Mistral v13 template to ids; tool calls parsed by control id | PASS | `template.rs` |
| Sampling | Mojo top-k and softmax/top-p in the per-token path, Rust reference fallback | PASS | `verification/mojo_audit.md` (parity tests) |
| Sampling | Backend label from the path that actually ran | PASS | `sampler: mojo-dylib` in reply meta |
| Engine | One worker thread, one-slot queue, busy refusal, 40 ms delta coalescing | PASS | `frost-chat` engine acceptance test |
| Engine | Regenerate, Clear Context, persistence of meta | PASS | engine acceptance test |
| Engine | Context truncation note and `context_truncated_messages` | NOT TESTED | Implemented; no test fills the context |
| Engine | Thermal admission at request start (Serious/Critical defers, `deferred_before_start`) | PASS (simulated) | `thermal-sim` in the engine test: the state is set before the send, so the reply is deferred before it starts |
| Engine | Thermal stop during a reply, at a prefill-chunk or token boundary | NOT TESTED | Implemented (governor on every `PrefillChunk`, then every 500 ms of decode); no test changes the state mid-reply; a real Serious state NOT TESTED |
| Engine | Memory pressure: critical defers at request start; warn clears MLX cache, critical stops and unloads | NOT TESTED | Implemented; nothing injects pressure |
| Engine | Modes: prefill chunk 256/512/1024, Quiet decode rest | NOT TESTED | Implemented; effect on heat or speed not measured |
| Engine | Single-instance lock on the data directory | PASS | unit test `engine_without_model_reports_error_status_and_refuses_sends`. Through the C ABI a failed `Engine::new` is reported via status `error` and `frost_last_error` (no test drives that path) |
| Store | SQLite WAL, migrations, cascades, attempts, embedding LRU | PASS | 10 unit tests in `frost-store` |
| App | AppKit shell: Mode menu and Send, streamed reply | PASS | macOS System Events automation switched Mode to Performance and sent a prompt; reply streamed, 66 new tokens at 23.3 tok/s, sampler `mojo-dylib`, finish `eos` (`verification/ui/`: `ui-automation-run.log`, `frost-1-ready.png`; `frost-3-done.png` shows an earlier Quiet-mode reply mid-stream with a second Send refused as busy) |
| App | Other menu items (New Chat, Open Repository, Cancel, Regenerate, Clear Context, Delete, Model Info, toggles), review cards, Copy | NOT TESTED | Not driven by automation |
| App | Loading and error states before the model is ready | NOT TESTED | Read-only composer while loading; error banner in transcript |
| App | C ABI and `--headless-check` | PASS | Bootstrap step 9 in `bootstrap-20260927-085549.log` and the acceptance run: state `ready`, `mlx_active_bytes` 4,775,780,608 |
| App | Relaunch from Finder with saved conversations | PASS | `open ~/Applications/FROST.app` after the acceptance run: sidebar listed the saved conversation, selecting it showed the full transcript (`verification/ui/frost-4-finder-launch.png`) |
| Agent | Read-only `list_files`, `read_file`, `search` inside the attached repo | PASS | `agent.rs` round-trip test on `rust-slugify` |
| Agent | `edit_file` / `propose_patch` / `run_command` as attempts awaiting approval; deny path | PASS / NOT TESTED | Approve path tested; `edit_file` becomes a unified diff (`diff_for_replacement`) and shares the patch path; deny path has no test |
| Agent | Base-sha256 validation, stale-patch refusal at apply, `held_out` paths protected | PASS | `agent.rs` test, `frost-tools` tests (22); every fixture's held-out file has `held_out` in its path |
| Agent | Bounded runner: env allowlist, deadline, head+tail cap, process-group kill | PASS | `frost-tools` tests; kill 32 to 53 ms after the deadline |
| Agent | Fixtures `rust-slugify`, `zig-rle`, `c-parse-kv` with live held-out oracles | PASS | `fixture_*_oracle_is_live` tests (fail before, pass after a correct fix) |
| Agent | `frost eval` end to end (runs EXPECTED `build`, then `test`) | PASS plumbing; held-out 1 PASS, 2 FAIL | Acceptance run (`verification/coding_eval/*.json`): all three fixtures applied patches and ran commands through the approval loop with held-out files unchanged; c-parse-kv passed its held-out test after one repair, rust-slugify and zig-rle failed on model-written code; 0 of 3 passed on the first attempt (VERIFICATION item 5) |
| Retrieval | Repository indexer, hybrid search, citation verification (`frost-repo`) wired into chat | PASS | `retrieve()` in `frost-chat` before every reply with a repository attached; engine test `repository_grounded_answers_cite_verified_spans_and_track_file_changes` passed in `bootstrap-20260927-084613.log` and `bootstrap-20260927-085549.log`; 10 unit tests in `frost-repo` |
| Retrieval | Zig FROSTIDX1 mmap index, generation swap, SIMD exact search | PASS | `frost-index` tests; 378 µs for 10k x 768 |
| Retrieval | nomic encoder on MLX, scalar parity cos 1.000000 | PASS | `frost-model` acceptance tests |
| Install | Staged bundle, vendored dylibs, rpath rewrite, ad-hoc signing, manifest, backup | PASS | `sh bootstrap.sh` with no flags completed all nine steps twice (`bootstrap-20260927-085549.log`, then `bootstrap-20260927-094338.log` from the finished tree); the installed app comes from the second run. Isolated install/uninstall/reinstall under a throwaway HOME PASS in the acceptance run. Rollback not exercised |
| Install | Python audit (5 checks) | PASS | `tools/python_audit.sh` |
| Install | Offline audit (no network crates, 0 sockets during a generation) | PASS | `tools/offline_audit.sh` in the acceptance run: no network crates, 0 internet sockets during a real `frost gen` |
| Build | Mojo compiler invoked Python-free | PASS | `verification/mojo_audit.md` |
| Build | Installing the Mojo compiler without Python | BLOCKED | No official channel does it (`mojo_audit.md` 1c) |
| Other | Live file watcher | NOT IMPLEMENTED | Chat re-stats the repository's files before each reply instead |
| Other | OS sandbox for approved commands | NOT IMPLEMENTED | Documented as not a sandbox |
| Other | Energy measurement | NOT IMPLEMENTED | No method used |
| Other | `frost train` projection head | Experiment | Refuses unless thermal state is Nominal; not used by chat |

## Defects A to J from the audit

| | Defect | Resolution | State |
|---|---|---|---|
| A | Duplicate encoder block code and debug ablation paths | One `block()` in `frost-model`; test asserts `forward_upto(all layers) == forward` | Resolved |
| B | Embeddings recomputed; unbounded caches | `embedding_cache` keyed by (recipe, content digest) with byte-bounded LRU pruning (256 MiB in `frost-repo`); a recipe change forces re-index | Resolved; reachable from chat through `retrieve()` |
| C | Hidden per-mode token budgets | One 4096 budget in every mode, 1024 reserved; modes change prefill chunk and Quiet pacing only; truncation is a note plus meta | Resolved. `frost_core::Mode::input_token_budget` (512/1024/2048) survives as unused code |
| D | Thermal override in production | No override; `thermal-sim` feature for tests only; OS state checked at request start, on every prefill chunk and every 500 ms during decode; libdispatch memory-pressure source | Resolved |
| E | Inference on the UI thread | All generation on the `frost-generator` worker; UI gets coalesced events | Resolved |
| F | Unchecked weight and tokenizer parsing | `frost-tensors` hardened reader for both models; control ids and vocab size verified | Resolved |
| G | Ranking engine and "calibration" gate | `frost-service` deleted; retrieval scores labelled similarity or rank fusion | Resolved. Ranking-era types (`Candidate`, `Decision`, `Calibration`) remain unused in `frost-core` |
| H | Text-IPC Mojo helper | Replaced by a C-ABI dylib; label comes from the executed path | Resolved |
| I | Weight-backed tests silently skipped | `#[ignore]` acceptance tests that panic when assets are missing | Resolved. `frost-repo`'s ignored test is in the `--ignored` lists of both `bootstrap.sh` and `tools/run_acceptance.sh` |
| J | Installer | Staged bundle, vendored dylibs, ownership manifest, backup and rollback, verify after install | Resolved; exercised end to end in `bootstrap-20260927-085549.log` (rollback path not triggered) |

Exercising J on 2026-09-27 took four bootstrap runs. Three failed and were fixed in this order:
the memory admission formula undercounted available memory (see the Generator row above); the CLI
was staged as `Contents/MacOS/frost`, which on case-insensitive APFS overwrote the app binary
`Contents/MacOS/FROST` (the CLI now lives in `Contents/Helpers/frost`); and codesign rejected
`mlx.metallib` under `Contents/Frameworks` (it now lives in `Contents/Resources` with a symlink
in `Frameworks`).

## Known gaps and defects found in the code

1. Quiet mode sleeps 0.4 times each step's compute time (capped at 60 ms), which is about 29% of
   wall time, not 40%.
2. `frost-tools` can plan diagnostics commands, but the model is offered only six tools; there
   is no `diagnostics` tool.
3. Leftovers: two `sha2` versions in the lock file (0.10.9 for `frost-repo`, 0.11.0 for
   `frost-tools`); the ranking-era types in `frost-core` (defects C and G); the `frost-train`
   experiment.
4. No upstream golden-vector parity for the generator (needs Python offline); energy not
   measured; app menu items other than Mode and Send not driven by automation.
5. The engine test's thermal simulation sets Serious before sending, so it exercises the
   request-start deferral only; no test stops a reply mid-prefill or mid-decode.

Resolved since the previous revision of this list: `frost-repo` is wired into chat;
`frost_engine_new` reports the real error instead of building a temp-directory engine or
panicking; the C fixture's held-out test is `held_out_test_kv.c`, so the `held_out` guard covers
all three fixtures; `frost eval` runs `build` before `test`; thermal and memory admission is
checked at request start and on every prefill chunk; `frost_app.h` comments match the ABI; the
always-passing "python-free process" diagnose check is gone and `frost help` has no stray line;
bootstrap step 9 uses `frost mode quiet --if-unset` and no longer uses brace expansion; the
ranking-era files under `verification/` and `native/proofs/mlx_gpu_test.c` are deleted.
