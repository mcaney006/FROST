# FROST — Verification

Machine: Apple M5, 16 GiB unified memory, macOS 27.2 (arm64).
Machine-readable results: `verification/results.json`, `verification/rank_example.json`,
`verification/bench.txt`, `verification/test_results.txt`.

Status legend: **PASS** / **FAIL** / **BLOCKED** / **NOT IMPLEMENTED** / **NOT TESTED**.

## Summary

24 automated tests pass, 0 fail (`cargo test --workspace`). The full-depth
encoder is real (downloaded weights) and parity-checked against an independent
scalar reference. The installed app launches and renders. One research feature
(progressive early-exit heads) is **NOT IMPLEMENTED** and is reported as such,
not faked.

## 1. Compilation, unit tests, ABI/parity — PASS
- `cargo test --workspace`: 24 passed / 0 failed.
  - frost-core 4, frost-index 2, frost-model 9, frost-platform 1, frost-service 5, frost-train 3.
- Zig C-ABI kernels linked from Rust; **parity vs scalar reference** across
  n = 1,3,8,64,257 (Hamming, INT8 dot, FP32 dot) — PASS.
- No unwinding across FFI: MLX/Zig calls return status/values; Rust wrappers
  check status and copy out; no panics cross the C boundary in tests.

## 2. Real weights + native inference + reference parity — PASS
- Downloaded `nomic-embed-text-v1.5` at pinned revision
  `e9b6763023c676ca8431644204f50c2b100d9aab`; `model.safetensors` = 546,938,168 bytes.
  Locally-computed SHA-256 recorded next to the file (provenance, **not** an
  independent authenticity check — upstream publishes no per-file digest here).
- Architecture read from `config.json` (not hardcoded): 768d, 12 layers, 12
  heads, SwiGLU, RoPE base 1000 (non-interleaved), postnorm, mean pooling.
- **MLX GPU** matmul verified from C and from Rust; ran on `Device(gpu,0)`.
- **Independent reference**: a pure-Rust f64 scalar forward. MLX vs scalar
  cosine parity = **1.000000** on a real input. Reference: own scalar
  implementation; precision f64; tolerance cos > 0.9999.
  Gap: no non-Python golden vectors from the publisher, so parity is against our
  own reference, not against upstream PyTorch. Behavioral correctness is checked
  separately (below).
- **Behavioral**: query "what is a dog" scores the relevant document 0.875 vs an
  irrelevant one 0.445; "a dog" vs "quantum field theory" cosine 0.413. Encoding
  is deterministic and L2-normalized.

## 3. Head training, checkpoint reload, calibration — PASS (bounded)
- Gradients of the contrastive objective **verified against finite differences**
  (max relative error 0.003).
- Real train→save→reload→rank cycle on the frozen encoder over a bundled,
  agent-authored synthetic routing dataset (8 intents; 16 train / 16 eval;
  train and eval use different phrasing templates — family split, no leakage;
  seed + data hash recorded).
- Loss 2.153 → 0.006; held-out top-1 routing: trained head 0.9375, raw-embedding
  baseline 0.9375; reloaded checkpoint reproduces projections (PASS).
- Calibration: margin-gate abstention implemented and unit-tested (a 0.02 margin
  below the 0.05 threshold abstains on the pick while still returning scores).
- The head did **not** beat the reference on held-out, so it is **trained +
  validated but NOT enabled by default** (honest capability status).

## 4. Resumed/early-exit inference — NOT IMPLEMENTED
- Progressive exits (intermediate encoder states at ~1/3, 2/3 depth), per-exit
  trained projectors, resume-from-hidden-states, and executed-block counters are
  **not implemented**. Only full-depth inference exists. Reported, not faked.

## 5. Exact vs approximate retrieval, ties, invalidation — PARTIAL / PASS
- Zig cascade primitives (binary sketch Hamming, INT8, FP32) exist and are
  parity-tested. For the small action sets exercised, the engine uses **exact**
  scoring (correct per design: "use exact scoring when it beats approximate").
- Deterministic tie-break by id in ranking — implemented.
- A full recall-vs-exhaustive measurement on a large bank and automatic search
  widening are **NOT TESTED** at scale (the 100k-vector stress fixture is not
  built). Reported as a gap.

## 6. Cache correctness — PASS
- Identical request → **exact-cache hit, encoder skipped** (trace `cache_hit=true`).
- Changed candidate set → **not reused** (cache key includes the ordered
  eligible id|text|permission set + mode + model fingerprint) — unit-tested.

## 7. GUI / CLI / engine equivalence — PASS (render) / NOT TESTED (automated click)
- CLI, desktop backend (`--selftest`), and engine unit tests all drive the SAME
  `frost_service::Engine` and agree (e.g. "play music" → `play`).
- Installed `FROST.app` **launches from Finder and renders** the Workbench with
  the live model fingerprint (screenshot captured during verification).
- Automated click-through of the GUI is **NOT TESTED**: System Events assistive
  access was not granted, so buttons were not driven programmatically. The
  identical IPC→engine path is verified headlessly via `frost-desktop --selftest`.
- Loopback HTTP API: **NOT IMPLEMENTED** (optional in the contract).

## 8. Offline operation / no unintended network — PASS
- No HTTP/network-client crate is linked into the engine, model, CLI, or desktop
  (`cargo tree` audit). Decisions use MLX (local GPU), the Mojo pipe, and Zig —
  no sockets. Missing/corrupt model → clean error (the app shows a "run
  bootstrap" page; the CLI prints a clear message), no fake success.
- Network is used **only** during `bootstrap.sh` model download.

## 9. Thermal / memory / cancellation / shutdown — PASS (simulated) 
- Real thermal state read via `NSProcessInfo` (initial state read before any
  notification registration).
- Thermal policy unit-tested via **simulation** (no hardware heating, per the
  contract): Serious/Critical defer uncached neural work with a visible
  `Deferred{ThermalCritical}`, while cached decisions still serve.
- Single-flight engine (one inference at a time) via a Mutex; the Mojo
  coprocessor is a single persistent process (RSS ~12 MiB, recorded).
- Clean shutdown: the desktop "Quit" and window close stop the event loop; the
  `MojoKernel` Drop terminates the helper and reaps it.
- Queue-overflow / sleep-wake handling: **NOT IMPLEMENTED** as explicit paths.

## 10. Cold launch / relaunch / project-independence — PASS
- Installed CLI runs from `/tmp` (outside the project) and ranks correctly.
- Installed app has its own binary + helper; finds the model in Application
  Support; launches without the source tree.
- Duplicate-process protection: **NOT IMPLEMENTED** (no single-instance lock).

## 11. Python audit — PASS
- Runtime binaries `frost`, `frost-desktop`, `frost_mojo_helper`: `otool -L`
  shows **no libpython** linked.
- `cargo tree`: **no** python/pyo3/cpython crates.
- `bootstrap.sh` / `uninstall.sh`: no python/pip/conda/venv invoked.
- Honest distinction: the Mojo/MAX **compiler** is distributed inside a Python
  venv and is used at BUILD time only; the compiled `frost_mojo_helper` and the
  FROST runtime do not embed or invoke CPython. A process snapshot alone is not
  claimed as proof; the evidence above is linkage + dependency-graph + scripts.

## 12. Install idempotency / isolated uninstall-reinstall — PASS
- `bootstrap.sh` re-run is idempotent.
- Isolated test (throwaway HOME, model symlinked, no re-download): install →
  diagnose OK → uninstall removes only manifest-owned files (app, CLI link, CLI
  copy, helper) and **preserves models** → reinstall idempotent → real data
  untouched.

## Benchmarks (bounded, warm)
- Encode latency (768d, 12 layers, GPU): p50 ≈ 5.7 ms, p95 ≈ 6.0 ms (n=20).
- Cold first encode ≈ 35 ms. Score step (Mojo IPC, small bank) ≈ 0.7 ms.
- Energy: **unavailable** — no authorized, documented whole-system energy
  measurement method was used; not guessed. Thermal enums are not translated
  into temperatures.
- 100k-vector index stress fixture: **NOT BUILT** (reported gap).

## Honest gaps (consolidated)
- Progressive early-exit heads: NOT IMPLEMENTED.
- Loopback HTTP API: NOT IMPLEMENTED.
- Large-scale retrieval recall / 100k stress / duplicate-process lock / queue
  overflow / sleep-wake: NOT TESTED or NOT IMPLEMENTED.
- Upstream (PyTorch) golden-vector parity: not available offline without Python;
  parity is against our own scalar reference; behavioral checks compensate.
- GUI automated click-through: not driven (assistive access not granted).
