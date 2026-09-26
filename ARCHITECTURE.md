# FROST — Architecture

Actual implemented boundaries (not aspirational). Every crate below exists,
compiles, and is exercised by tests.

## Process & data flow

```
                       ┌────────────────────────── FROST.app (wry webview) ─┐
 CLI  ── frost ─┐      │  index.html  ──ipc──► handle()                     │
                │      └───────────────────────────────┬────────────────────┘
                ▼                                       ▼
        ┌──────────────────────  frost_service::Engine  ──────────────────────┐
        │  exact-decision cache (fingerprint keyed)                           │
        │  thermal policy (frost-platform: NSProcessInfo)                     │
        │  calibration margin-gate (frost-core)                              │
        └───────┬───────────────────────────┬───────────────────────────────┘
                │ encode                     │ score
                ▼                            ▼
   frost_model::Embedder            frost_service::MojoKernel  (persistent)
   ├─ tokenizer (BERT WordPiece)      └─ frost_mojo_helper (compiled Mojo,
   ├─ nomic-bert forward on MLX-C        line-framed pipe IPC, NORM/SCORE)
   │  (GPU, Apple metallib)          fallback: frost_index (Zig C-ABI dot)
   └─ safetensors (mmap)
```

The same `Engine` instance backs the CLI, the desktop app, and the tests. There
is one model-owning process per surface; the Mojo helper is a single long-lived
child, never spawned per request.

## Crates and ownership

- **frost-core** — pure logic, no I/O. `Request`/`Candidate`/`Decision`/`Scored`/
  `Trace`, `Mode`, `ThermalState`, `Exit`, `Width`, `Calibration` (margin-gate
  abstention), `Fingerprint`, FNV-1a hashing and order-independent id-set hashing
  (the cache-safety invariant). Unit-tested.

- **frost-index** — safe Rust over `native/index/frost_index.zig` (compiled by
  `build.rs` via `zig build-lib`). Cascade primitives: binary-sketch Hamming
  (XOR+popcount), INT8 dot (explicit caller scale), FP32 exact dot. A scalar Rust
  reference backs the parity tests.

- **frost-model** — the encoder.
  - `mlx.rs`: a minimal RAII wrapper over `mlx-c` (matmul, add, mul, sigmoid/silu,
    softmax, `fast_layer_norm`, `fast_rope`, transpose, reshape, mean). Verified
    against hand math.
  - `model.rs`: nomic-bert forward — Rust-side embedding gather + token-type, MLX
    LayerNorm, 12 postnorm blocks (fused QKV, RoPE base 1000 non-interleaved,
    manual bidirectional attention, SwiGLU `fc11 * silu(fc12)`), mean pool.
    Weights mmap'd from safetensors, projection weights pre-transposed.
  - `tokenizer.rs`: BERT-uncased WordPiece over the real `vocab.txt`.
  - `reference.rs`: independent f64 scalar forward for parity.

- **frost-platform** — macOS thermal/power/memory via an Objective-C shim over
  `NSProcessInfo` (`thermalState`, `physicalMemory`, `isLowPowerModeEnabled`).
  Thermal *state*, explicitly not a temperature.

- **frost-service** — `Engine` (encode → score → margin-gate → cache, with the
  thermal-deferral policy and a thermal-simulation override) and `MojoKernel`
  (the persistent coprocessor + install-aware helper discovery).

- **frost-train** — matrices, seeded init, symmetric InfoNCE loss with
  finite-difference-verified analytic gradients, SGD, JSON checkpoints, and the
  synthetic-dataset train→save→reload→rank pipeline.

- **frost-cli** — `diagnose | models | rank | bench | train`.

- **frost-desktop** — wry/tao shell; embeds `ui/index.html`; IPC → the same
  `Engine`; `--selftest` runs the backend headlessly.

## FFI boundaries (all bounded, checked)

- **Rust ↔ Zig**: static lib, C ABI, slices with checked equal lengths; values
  returned by value; parity-tested against a scalar reference.
- **Rust ↔ MLX (mlx-c)**: `mlx_array` handles are RAII-freed; ops return status
  ints that are asserted; data copied out only after `eval`. Weights are copied
  into MLX arrays (alignment-safe) from the mmap.
- **Rust ↔ Mojo**: no direct dylib (this Mojo nightly's Pointer/origin
  unification blocks a clean buffer C-ABI export; see IMPLEMENTATION_STATUS).
  Instead a persistent process over a bounded, line-framed text protocol
  (NORM / SCORE / PING). One process, single-flight, RSS recorded.

## Decision semantics

- Candidates filtered by the optional allowed-id set.
- Empty bank → `Deferred{EmptyBank}`.
- Thermal Serious/Critical → `Deferred{ThermalCritical}` for uncached work; cache
  hits still serve (no new neural work). A thermal limit never turns an uncertain
  answer confident — it defers.
- Ranking sorted by raw cosine, deterministic id tie-break.
- Pick is returned only if top-1 clears the calibrated margin; else abstain on the
  pick while still returning the full ranking.
- Trace records exit, width, device, cache-hit, thermal, per-stage timings, model
  fingerprint, and notes (scorer used, deferral reasons).

## Persistence & install

- Model + tokenizer: `~/Library/Application Support/FROST/models/…`.
- Mojo helper + CLI copy: `…/FROST/bin/`. Trained head: `…/FROST/heads/`.
- App: `~/Applications/FROST.app` (ad-hoc signed — not Developer ID / notarized).
- CLI: `~/.local/bin/frost`. Install manifest + `uninstall.sh` remove only owned
  files.
