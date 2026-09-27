# FROST

FROST is a local coding chatbot for Apple silicon Macs. It chats with
Ministral-3-8B-Instruct-2512 (4-bit, MLX) on the GPU, stores conversations in SQLite, and can
work on one repository you attach to a conversation: it retrieves relevant excerpts on every turn
and reads files itself, but every patch and every command it wants to run waits for your
approval. Nothing is sent over the network after the models are downloaded.

It ships as a native AppKit app (`FROST.app`) and a CLI (`frost`) that share one Rust engine.

This README describes the `frost-chat` branch as it is in the tree. Per-feature status lives in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), evidence in [VERIFICATION.md](VERIFICATION.md).
One thing to know up front: UI automation has driven only the Mode menu and Send; the other menu
items have NOT been tested by automation.

## Requirements

| Need | Detail |
|---|---|
| Hardware | Apple silicon (bootstrap refuses anything but `arm64`). Developed and measured on a Mac17,2 (Apple M5, 16 GB). |
| macOS | Bundle declares macOS 15.0 minimum; only macOS 27.2 has been exercised. |
| Free memory | The loader refuses to start unless available memory (plus MLX's own cache) covers the 4.45 GiB of weights plus 768 MiB headroom. Available memory is the kernel's `kern.memorystatus_level` (the percentage `memory_pressure` prints as "System-wide memory free percentage") times physical memory; if that sysctl fails, free + inactive + purgeable + speculative pages. |
| Disk | About 6 GB for models (5.6 GB generator, 0.55 GB encoder) plus the build tree. |
| Toolchain | Command Line Tools (full Xcode not needed), Rust 1.98, Zig 0.16, Homebrew `mlx` 0.32.1 and `mlx-c` 0.6.0. `bootstrap.sh` installs missing Rust/Zig/MLX/mlx-c with Homebrew and stops if Homebrew itself is missing. |
| Mojo (optional) | The native Mojo compiler at `$FROST_MOJO_BIN`. Without it the sampler uses the Rust reference path and reports `rust-reference`. Bootstrap never installs Python to get Mojo. |
| Network | Only for the pinned model downloads (`tools/fetch_*.sh`, curl). |

## Install

```sh
sh bootstrap.sh              # full run: prerequisites, models, build, tests, stage, install
sh bootstrap.sh --skip-fetch --skip-tests   # rebuild and reinstall only
sh bootstrap.sh --launch     # open the app when done
sh bootstrap.sh --force      # replace an app or CLI link that the manifest does not own
```

The script runs nine steps and logs to `~/Library/Application Support/FROST/logs/`:

1. Checks prerequisites (installs only what is missing and permitted).
2. Downloads both models at pinned revisions, resumable, size- and sha256-checked.
3. `cargo build --release` of `frost-cli` and `frost-desktop` (Zig and Mojo are built by `build.rs`).
4. Runs the workspace tests, then the model-backed acceptance tests (`--ignored`, serialized,
   they FAIL when a model is missing), then, if the Mojo compiler is present, the kernel tests
   with `FROST_REQUIRE_MOJO=1`.
5. Stages `FROST.app` in a temp dir: the app binary as `Contents/MacOS/FROST`, the CLI as
   `Contents/Helpers/frost` (APFS is case-insensitive, so `MacOS/frost` would overwrite
   `MacOS/FROST`); copies `libmlx`, `libmlxc`, `libfrost_kernels.dylib` and four Modular runtime
   dylibs into `Contents/Frameworks`; puts `mlx.metallib` in `Contents/Resources` with a symlink
   to it in `Contents/Frameworks` (codesign rejects the metallib under Frameworks); rewrites
   install names to `@rpath`, strips the venv rpath, signs each dylib and the CLI ad-hoc, then
   the bundle.
6. Verifies the staged bundle: no `/opt/homebrew` or `site-packages` references, strict
   `codesign --verify`, the staged CLI actually loads the bundled `libmlx`, `frost diagnose` passes.
7. Installs to `~/Applications/FROST.app` with a backup of the previous app and rollback if the
   installed CLI fails `diagnose`; links `~/.local/bin/frost` to `Contents/Helpers/frost`.
8. Writes `install-manifest.json` (owned paths, binary digests, toolchain versions).
9. Runs `frost diagnose --json` and `FROST --headless-check` against the installed copy, and
   sets the mode to Quiet only if no mode is stored yet (`frost mode quiet --if-unset`).

Signing is ad-hoc only: not Developer ID, not notarized, Gatekeeper untouched.

Uninstall: `sh uninstall.sh` removes only the paths listed in the manifest and keeps models,
the chat database and logs; `sh uninstall.sh --purge` removes all of
`~/Library/Application Support/FROST`.

## Using the app

The window has a conversation sidebar, a transcript, a composer and an inspector (model identity,
context budget and last-reply stats, attached repository, notes). While the model loads the
composer is read-only; if loading fails a banner at the top of the transcript shows the error and
sending is disabled.

| Menu item | Shortcut |
|---|---|
| New Chat / Open Repository… | ⌘N / ⌘O |
| Send (Return also sends, Shift+Return is a newline) | ⌘↩ |
| Cancel / Regenerate / Clear Context | ⌘. / ⌘R / ⌘K |
| Delete Conversation (asks first) | ⌘⌫ |
| Mode: Quiet, Balanced, Performance | Mode menu |
| Toggle Sidebar / Toggle Inspector | ⌥⌘S / ⌥⌘I |
| Wrap Code Blocks, Model Info… | View menu, Model menu |

Code blocks render with a language label and a Copy button, and diffs are coloured. Clear
Context hides earlier turns from the model without deleting them from the transcript.

Coding work: attach a repository with Open Repository. Before every reply FROST re-checks the
repository for changed files, re-indexes them with the nomic encoder (paused, with a note, under
thermal or memory pressure), and prepends the six best excerpts (hybrid vector and keyword
search) to the model's copy of your message, framed as data. Each reply's meta lists the cited
path, line range and digest, re-verified against the file after the reply. The model can also
call `list_files`, `read_file` and `search` (literal substring search), which run at once,
read-only, inside that directory. `edit_file`, `propose_patch` and `run_command` appear in the
transcript as review cards with Approve and Deny. An approved patch is re-checked against the
file hashes it was written for and refused if the file changed; an approved command runs with a minimal environment, a deadline and
an output cap. Approved commands are NOT sandboxed: a build script runs with your privileges.

## Using the CLI

```sh
frost diagnose [--json] [--load]     # component checks; --load also generates one reply
frost models                         # generator and encoder identity, no weights loaded
frost gen --prompt "..." [--max N] [--temp T] [--system TEXT] [--json]
frost chat [--max N] [--temp T]      # interactive; /reset, /stats, /quit
frost eval --fixture fixtures/rust-slugify [--out DIR] [--max-rounds N] [--budget-s S]
frost mode quiet|balanced|performance [--if-unset]
frost train [--epochs N]             # optional experiment, not used by chat
```

`gen` and `chat` drive the generator directly. `eval` runs a fixture task through the same
engine and approval loop the app uses, auto-approving every proposal, then judges the result with
the fixture's build command (if any) followed by its held-out test, and writes `verification/coding_eval/<fixture>.json` (relative to
the current directory unless `--out` is given).

Only one engine may own a data directory at a time (a `flock` on `frost.lock`). `eval` uses a
throwaway data directory with the models symlinked in, so it does not take the app's lock, but it
loads its own model copy, as do `gen`, `chat` and `diagnose --load`. Quit the app first: 16 GB
does not hold two copies, and the memory admission check will refuse the second load.

## What runs where

| Work | Where |
|---|---|
| Transformer forward pass (34 layers, 4-bit matmuls, attention, KV cache) | MLX on the Metal GPU, through mlx-c |
| Repository retrieval: nomic encoder forward pass | MLX on the Metal GPU |
| Repository retrieval: exact vector search; BM25-lite and rank fusion | Zig index (SIMD), CPU; Rust, CPU |
| Top-k and softmax/top-p over the logits | Mojo kernels in `libfrost_kernels.dylib` on the CPU (Rust reference if absent) |
| Repetition penalty (off by default), greedy argmax, the categorical draw | Rust, CPU |
| Checkpoint layout validation at load | Zig (`frost_q4_validate`), CPU |
| Tokenizer, chat template, engine, persistence, tools | Rust, CPU |
| Window, menus, transcript | AppKit (Objective-C), main thread |
| Thermal state, memory pressure, free memory | macOS APIs via Objective-C |

Generation runs on one background thread; the UI receives coalesced text deltas at most every 40 ms.

## Limits

- Context: 4096 tokens per request in every mode, of which 1024 are reserved for the reply.
  Older turns are dropped to fit and a note says so.
- One generation at a time; a second send while busy is refused, not queued.
- Measured decode speed is about 25 tokens/s on the M5 (temperature 0.15); Quiet mode is slower
  on purpose.
- Serious or Critical thermal state, or Critical memory pressure, when a reply would start
  finishes it at once with a note. During a reply, Serious or Critical thermal state stops it at
  the next prefill chunk or token boundary; Critical memory pressure stops it, drops the KV cache
  and unloads the model until your next message.
- No OS sandbox for approved commands. No live file watching: the repository is re-stat'ed
  before each reply instead.
- Vision input is not supported (the checkpoint's vision tower is never loaded).

## Documents

[ARCHITECTURE.md](ARCHITECTURE.md) · [MODEL_CARD.md](MODEL_CARD.md) ·
[DEPENDENCIES.md](DEPENDENCIES.md) · [INVENTION_NOTES.md](INVENTION_NOTES.md) ·
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) · [VERIFICATION.md](VERIFICATION.md)
