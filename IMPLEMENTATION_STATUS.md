# FROST — Implementation Status (living handoff)

Machine: Apple M5, 16 GiB unified, macOS 27.2 (arm64), 539 GiB free.
Repo: ~/Developer/FROST (github mcaney006/FROST, was empty except README).

## Toolchain (pinned, verified present)
| Tool | Version | Notes |
|---|---|---|
| Rust/Cargo | 1.98.1 (Homebrew) | app core, FFI |
| Zig | 0.16.0 (Homebrew) | action index C ABI |
| Mojo/MAX | 1.1.0.dev2026082005 | nightly, `std.` stdlib namespace, `fn` removed→`def` |
| MLX | 0.32.1 (Homebrew) | prebuilt metallib — GPU works w/o Metal compiler |
| mlx-c | 0.6.0_4 (Homebrew) | C API + libmlxc.dylib |
| Xcode | CommandLineTools only | full Xcode / `metal` compiler ABSENT (see blockers) |
| clang | Apple | FFI glue |

## Boundary proofs (all PASS) — native/proofs, native/kernels, native/index
- **MLX GPU** [PASS]: mlx-c matmul on Device(gpu,0), exact result, python OFF PATH.
  `native/proofs/mlx_gpu_test.c`.
- **Zig C-ABI** [PASS]: hamming/dot_i8/dot_f32 via static lib linked from C.
  hamming=8, dot_i8=32, dot_f32=32.0. `native/index/frost_index.zig`.
- **Mojo kernel** [PASS via sanctioned IPC fallback]: persistent line-framed helper,
  real compiled kernel; PING→PONG, NORM[3,4]→[0.6,0.8], SCORE→[1,0].
  `native/kernels/frost_mojo_helper.mojo` (+ compiled binary).

## Python audit (honest)
- Mojo/MAX ships INSIDE a Python venv (`max-nightly-env`, has pyvenv.cfg, cpython .so).
  The `mojo` COMPILER binary links NO libpython (`otool -L`). Compiler is a build-time
  tool; the FROST runtime (Rust + Zig + MLX + compiled Mojo helper) does not embed or
  invoke CPython. No pip/CPython/PyTorch installed FOR this project.
- Distinction reported per contract: inert venv-delivered compiler ≠ runtime Python dep.

## Blockers (honest)
- **Mojo buffer C-ABI export [BLOCKED on this nightly]**: Pointer/UnsafePointer origin
  unification makes any pointer param parametric; `@export` rejects parametric fns, and
  `MutableAnyOrigin` / from-address ctor are not reachable offline. Mitigated by the
  contract-sanctioned persistent-helper-over-IPC route (implemented, PASS). Upgrade path:
  when the toolchain re-exposes a concrete-origin raw pointer, switch helper→direct dylib.
- **Metal source compiler ABSENT**: only CommandLineTools installed. MLX's OWN ops ship a
  precompiled metallib (GPU works). Authoring NEW custom .metal kernels would need full
  Xcode — not required; MLX ops cover the encoder. Custom Metal = NOT ATTEMPTED.

## Capability matrix (implemented / trained / validated / enabled)
| Component | impl | trained | validated | enabled |
|---|---|---|---|---|
| MLX GPU compute path | yes | n/a | yes | yes |
| Zig index kernels | yes | n/a | yes | yes |
| Mojo IPC kernel | yes | n/a | yes | yes |
| Rust workspace | in progress | - | - | - |
| Encoder (nomic-embed) | pending | n/a | pending | pending |
| Progressive heads | pending | pending | pending | disabled |

## Next
Rust workspace → frost-index(Zig FFI) → frost-service(Mojo IPC) → frost-model(MLX+weights)
→ frost-core/train → frost-cli → frost-desktop(Tauri) → package/install → verify.

---

## FINAL STATE (end-to-end build complete)

Workspace tests: **24 passed / 0 failed** (`cargo test --workspace`).
Installed: `~/Applications/FROST.app`, CLI `~/.local/bin/frost`, data in
`~/Library/Application Support/FROST`. App launches from Finder and renders
(screenshot-verified); CLI works from outside the project tree.

### Capability matrix (final)
| Component | impl | trained | validated | enabled |
|---|---|---|---|---|
| Full-depth encoder (MLX GPU) | yes | n/a | yes (scalar parity + behavioral) | yes |
| Zig retrieval-index kernels | yes | n/a | yes (parity) | yes |
| Mojo IPC scoring kernel | yes | n/a | yes (round-trip) | yes |
| frost-platform thermal | yes | n/a | yes | yes |
| Engine (encode/score/gate/cache/thermal) | yes | n/a | yes | yes |
| Calibration margin-gate abstention | yes | n/a | yes | yes |
| Full-depth projection head | yes | yes | yes | **no** (didn't beat baseline) |
| Progressive early-exit heads | **no** | no | no | no |
| Loopback HTTP API | **no** | - | - | no |
| 100k index stress / large-scale recall | **no** | - | - | - |
| Duplicate-process lock / queue overflow | **no** | - | - | - |

### PASS / BLOCKED / NOT IMPLEMENTED
- **PASS**: native boundaries (MLX/Zig/Mojo), real weights + inference + scalar
  parity, tokenizer, engine decisions, cache invalidation, thermal deferral
  (simulated), head training with finite-diff-verified gradients + train/save/
  reload/rank, Python audit, offline audit, install idempotency, isolated
  uninstall/reinstall, GUI render + backend selftest, CLI, packaging.
- **BLOCKED (upstream toolchain)**: direct Mojo buffer C-ABI export (Pointer/
  origin unification in this nightly) — mitigated by the sanctioned persistent
  IPC helper. Full-Xcode-only Metal source compiler absent — not needed (MLX
  prebuilt metallib covers the GPU path); custom .metal kernels not attempted.
- **NOT IMPLEMENTED (honest)**: progressive early-exit heads; loopback HTTP API;
  100k stress fixture / large-scale recall measurement; duplicate-process lock;
  explicit queue-overflow and sleep/wake paths; upstream PyTorch golden-vector
  parity (no non-Python reference offline).

### The one substantive encoder bug found & fixed
The full-depth forward initially collapsed every input to cos≈1.0. Localized via
the scalar reference (MLX==scalar, so architecture not composition) to the
SwiGLU gate order: nomic is `fc11 * silu(fc12)`; the first cut applied silu to
`fc11`. After the fix, "dog" vs "quantum field theory" cos = 0.413 and relevant
docs rank correctly. Verified against nomic's published modeling source.
