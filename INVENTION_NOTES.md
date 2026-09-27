# FROST implementation notes

What FROST does differently from the usual way of running a local model, where each idea comes
from, and what was measured. Nothing here is claimed as a new algorithm; the value is in the
combination and in checking each piece.

## Choices and their prior art

| Choice in FROST | Prior art it follows | What FROST adds or changes |
|---|---|---|
| Ministral-3 forward pass written in Rust over mlx-c, no Python | mlx-lm (Python) implements the same architecture on MLX; llama.cpp does native inference in C/C++ | A single-process native app with no interpreter; the YaRN frequency construction mirrors mlx-lm's `YarnRoPE`, and a unit test re-derives it independently |
| Chat template rendered straight to token ids; control tokens never parsed from text | mistral-common builds prompts as token sequences in the same way | Control ids are checked against the pinned vocabulary at load, and tool calls are parsed only from real `[TOOL_CALLS]`/`[ARGS]` ids, so text in a file or a message cannot forge one |
| KV cache reused across turns by longest common prefix, trimmed on regenerate | llama.cpp prompt cache, vLLM prefix caching | Applied per conversation in a desktop app; reuse is reported in every reply's meta |
| Checkpoint validated by a Zig kernel before any tensor reaches the GPU | safetensors' own header checks; loaders that trust the file | Every one of the 240 quantized tensors has its packing, sizes and scale/bias finiteness checked, and the result is reported rather than assumed |
| Bounded-heap top-k over 131072 logits in Mojo, softmax over the 64 survivors | Partial selection in llama.cpp and other samplers; heap selection is textbook | Kept in the real per-token path behind a C ABI with an exact-parity Rust reference, and each reply records which one ran |
| Memory admission before load; stop and unload under critical pressure | macOS apps reacting to `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` | Refuses a load that would swap instead of letting the OS page 4.5 GB of weights |
| Generation does not start, and stops, on Serious/Critical thermal state, no override | Apple's guidance to reduce work on `NSProcessInfo.thermalState` changes | Checked when a request would start, on every prefill chunk and at token boundaries, with a visible note; only a test-only feature can simulate it |
| Model proposes, user approves, compiler output is the evidence | Coding agents with permission prompts (for example Aider, Claude Code); SWE-bench-style held-out test judging | Patches are pinned to the sha256 of each file and re-validated at apply time, like `git apply --check` or an HTTP `If-Match`; the repair budget (2) and round budget (14) are enforced in code, not left to the prompt |
| Bounded process runner | `timeout(1)`, process-group kill in CI runners | Scrubbed env, no shell, head+tail capture, group kill; explicitly documented as not a sandbox |
| Hybrid retrieval: exact vector top-k fused with BM25-lite by reciprocal-rank fusion, run before every reply with a repository attached | RRF (Cormack et al., 2009); hybrid search in common search engines | An exhaustive scalar check reports the Zig index's recall against brute force, and citations are re-verified against current file bytes after each reply |
| mmap exact vector index with atomic generation swap | Flat indexes (for example FAISS `IndexFlatIP`); write-temp-then-rename | Header hash, version and exact-length checks on open, so a truncated or edited file is refused |
| Mojo built without running Python | The official `mojo` launcher is a Python script that execs a native compiler | Calls the native compiler directly with a scrubbed environment; see `verification/mojo_audit.md` |

## Measured facts (Mac17,2, Apple M5, 16 GB, macOS 27.2)

| Measurement | Value | Source |
|---|---|---|
| Generator weights in MLX | 4.45 GiB | `weight_bytes`, `frost models` |
| Load time | about 7 to 9 s | CLI load log |
| Decode | 25 tokens/s, 57 to 136 new tokens, temperature 0.15 | `frost-gen` acceptance runs |
| Prefill | 88 tokens in about 520 ms warm; first call adds about 800 ms Metal warm-up | same |
| Decode through the app | 23.3 tokens/s, 66 new tokens, Performance mode; prefill 1230 ms for 133 prompt tokens | `verification/ui/ui-automation-run.log` |
| Prefix reuse | 89 of 99 tokens on turn two | engine acceptance test |
| Mojo top-k, n = 131072, k = 64 | 0.044 ms (was 5.9 ms with a full sort; the Rust full-sort reference takes 8.0 to 8.7 ms) | `cargo run --release -p frost-kernels --example bench` |
| Mojo softmax/top-p over 64 candidates | about 1 µs | same |
| Zig exact search, 10,000 x 768, k = 10 | 378 µs p50 | `frost-index` test `search_latency_10k_x_768` |
| Encoder parity, MLX vs scalar f64 | cosine 1.000000 | `frost-model` acceptance test |
| Runner kill after deadline | 32 to 53 ms | `frost-tools` runner tests |
| Final acceptance run | Every check PASS except held-out quality for rust-slugify and zig-rle; 95 unit and 8 model-backed tests passed; installed `diagnose --load` 22.7 tokens/s greedy | `verification/results.json` (2026-09-27T14:41:06Z) |

Perspective on the sampler numbers: at 25 tokens/s a decode step takes about 40 ms, so the Mojo
top-k plus softmax cost about 0.1% of a step. The Rust reference is a full sort; at roughly 8 ms
it would cost about a fifth of a step. The comparison is against that reference, not against an
optimized Rust heap, which would likely be close to the Mojo kernel.

## Explicitly not claimed

- No novel algorithm. Every technique above has the prior art listed next to it.
- No claim that Mojo is faster than well-written Rust; only that the Mojo kernels are in the
  real path, agree with the reference, and are cheap.
- No model-quality claim on coding tasks; fixture results (1 of 3 held-out tests passed, after
  one repair) are reported apart from plumbing results.
- No security boundary around approved commands, and no protection against a repository that
  tries to instruct the model beyond telling the model to treat repository text as data.
- No upstream golden-vector parity for the generator, no energy or temperature figures, no
  behaviour beyond the 4096-token budget.
- No claim that retrieval improves answer quality. The engine test checks that citations verify
  against the file bytes and follow a changed file, not that answers get better.
