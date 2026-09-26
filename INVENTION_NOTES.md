# FROST — Invention Notes

Local engineering notes distinguishing inherited techniques, implementation
choices, testable hypotheses, and measured differences. **No** patentability or
ownership conclusions are drawn here; everything stays local.

## Inherited techniques (not novel)
- Contrastive text embeddings, InfoNCE, Matryoshka nested widths, rotary position
  embeddings, SwiGLU MLPs, BERT WordPiece tokenization, mean pooling — all prior
  art. FROST uses them; it does not claim them.
- The nomic-bert architecture and weights are third-party (Apache-2.0).
- Early-exit / progressive networks and calibrated selective prediction are
  established research areas (referenced as the design target, not implemented).

## Implementation choices (this project's engineering)
1. **Three-language native stack with checked boundaries**: Rust owns the engine;
   Zig provides the index kernels over a tested C ABI; Mojo provides a compiled
   numerical kernel. Each boundary is parity- or round-trip-tested.
2. **Mojo-over-IPC coprocessor**: the pinned Mojo nightly unified `Pointer` with
   origin tracking, which makes any pointer-typed parameter parametric and blocks
   `@export` of a buffer-taking C-ABI function; the from-address constructor and
   `MutableAnyOrigin` were not reachable offline. Rather than fake a dylib, FROST
   uses a persistent Mojo process over a bounded line-framed protocol — the
   contract's sanctioned fallback. Upgrade path: switch to a direct dylib when the
   toolchain re-exposes a concrete-origin raw pointer.
3. **MLX encoder from Rust via mlx-c**, using `fast_layer_norm`/`fast_rope` plus
   manual attention. Embedding gather done in Rust (batch=1) to sidestep gather
   FFI entirely.
4. **Independent scalar reference** for parity — a deliberate second
   implementation in f64 so MLX correctness is checked without Python.
5. **Cache-safety fingerprint**: order-independent id-set hash + model fingerprint
   + mode, so a decision is never reused across different allowed-candidate sets,
   permissions, or model versions.
6. **Thermal as a scheduling policy**, read from `NSProcessInfo` state (never a
   temperature), with a simulation override so the deferral logic is testable
   without heating hardware.

## Testable hypotheses (stated, not asserted as fact)
- *H1*: a small trained projection head over frozen nomic embeddings improves
  held-out routing over raw cosine. **Measured: not supported here** — head 0.9375
  vs baseline 0.9375 on the bundled synthetic eval. Larger/harder data might
  change this; untested.
- *H2*: progressive early exits can preserve routing accuracy at lower depth for
  easy states. **Untested — not implemented.**
- *H3*: the Zig binary-sketch cascade beats exact scoring above some bank size.
  **Untested at scale** (no 100k fixture); at small sizes exact scoring is chosen.

## Measured differences / facts (this machine)
- MLX vs scalar-reference parity: cosine 1.000000.
- Contrastive-head gradient vs finite differences: max rel. error 0.003.
- Encode latency: p50 ≈ 5.7 ms, p95 ≈ 6.0 ms warm; ~35 ms cold (768d/12L, GPU).
- Mojo coprocessor RSS ≈ 12 MiB; score step ≈ 0.7 ms for small banks.
- Runtime binaries link no libpython; no network client in the dependency graph.

## Explicitly NOT claimed
- No benchmark inheritance from CLM/nomic or any model.
- No performance guarantees, no energy or zero-heat claims (energy unavailable,
  not guessed).
- No general-intelligence claims from the synthetic demonstration dataset.
