# FROST

An offline, thermally-aware **state → action decision engine** for Apple Silicon.
Given a state and a set of allowed candidate actions, FROST ranks the candidates
with a real, model-backed encoder, returns an inspectable decision (or an
abstention), and spends extra computation only when justified.

It is **not** a chatbot, an autonomous shell executor, or a claim to beat any
reasoning model. Action selection here only *ranks* — it never executes shell
commands, sends messages, or mutates external systems.

## What it actually is

- A Cargo workspace in **Rust** (engine, scheduling, FFI, CLI, persistence).
- A **Zig** action-index library (binary-sketch Hamming, INT8 and FP32 dot) with
  a tested C ABI, linked from Rust and parity-checked against a scalar reference.
- A **Mojo** numerical kernel (fused L2-normalize + cosine rerank) compiled to a
  native binary and driven as a persistent coprocessor over bounded IPC.
- An **MLX** (Apple GPU) backend that runs the full-depth
  `nomic-embed-text-v1.5` encoder via the `mlx-c` C API from Rust — real
  downloaded weights, real tokenizer, verified against an independent scalar
  reference forward.
- A **Tauri-style desktop app** (`FROST.app`) built on the system webview (wry),
  whose backend is the exact same engine as the CLI.

The full-depth pretrained embedding ranker is the **reference/fallback** ranker.
A separately trained projection head (contrastive, gradient-checked) is included
and validated but is **not** enabled by default (it did not beat the reference on
held-out routing). Progressive early-exit heads are **not implemented** — see the
capability matrix in `IMPLEMENTATION_STATUS.md`.

## Install (one command)

```sh
sh bootstrap.sh
```

This is idempotent. It verifies toolchains (Rust, Zig, Mojo, MLX), builds the
Mojo kernel and the release binaries, downloads the pinned model weights
(~522 MiB, resumable) if absent, runs bounded tests, assembles and installs
`~/Applications/FROST.app`, links the CLI at `~/.local/bin/frost`, and verifies.

Prerequisites (Homebrew): `rust`, `zig`, `mlx`, `mlx-c`, and a Mojo/MAX nightly
providing `mojo`. Full Xcode is **not** required (MLX ships a precompiled
metallib; the GPU path works with Command Line Tools only).

## Use

CLI:

```sh
frost diagnose                 # component + self-test check
frost models                   # model provenance + fingerprint
frost rank --state "the user wants to listen to music" \
  --action "play=start playing a song" \
  --action "delete=erase all backup files" \
  --action "brightness=increase screen brightness"
frost bench --n 20             # encode-latency micro-benchmark
frost train --epochs 150       # train + validate a projection head (checkpoint)
```

Example (`frost rank`, measured on this machine):

```
pick: play
  1. play         0.6736
  2. brightness   0.5488
  3. delete       0.4999
trace: exit=full width=768 device=gpu cache_hit=false scorer=mojo-ipc
  encode: ~35 ms (cold)   score: ~0.7 ms
```

Desktop:

```sh
open ~/Applications/FROST.app
```

## Uninstall

```sh
sh uninstall.sh            # removes only FROST-owned files; keeps models
sh uninstall.sh --purge    # also removes models + data
```

## Layout

```
crates/frost-core       request/decision types, policies, calibration, cache keys
crates/frost-index      Rust wrapper over the Zig index + scalar parity
crates/frost-model      MLX encoder, tokenizer, safetensors, scalar reference
crates/frost-platform   macOS thermal/power/memory via NSProcessInfo
crates/frost-service    the shared Engine + persistent Mojo kernel
crates/frost-train      contrastive head training (gradient-checked)
crates/frost-cli        the `frost` CLI
crates/frost-desktop    FROST.app (wry webview) — same engine
native/index            frost_index.zig (C ABI)
native/kernels          frost_mojo_helper.mojo (compiled coprocessor)
```

See `ARCHITECTURE.md`, `MODEL_CARD.md`, `VERIFICATION.md`, `DEPENDENCIES.md`,
`INVENTION_NOTES.md`, and `IMPLEMENTATION_STATUS.md`.

## License

Dual MIT OR Apache-2.0 for FROST's own code. The model weights and tokenizer are
`nomic-ai/nomic-embed-text-v1.5` under their upstream license (Apache-2.0);
see `DEPENDENCIES.md`.
