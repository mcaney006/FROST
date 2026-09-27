# FROST model card

FROST uses two models. They are loaded, reported and labelled separately; the retrieval encoder
never produces chat text.

## 1. Generator: Ministral-3-8B-Instruct-2512, 4-bit MLX

| Field | Value |
|---|---|
| Checkpoint | `mlx-community/Ministral-3-8B-Instruct-2512-4bit` |
| Revision | `182f003f01daa75f9de0f2c4d379722fd0bc1c61` (pinned in `tools/fetch_ministral.sh` and `frost_gen::REVISION`) |
| Upstream model | Mistral AI, Ministral 3 8B Instruct (2512); MLX conversion by mlx-community |
| License | Apache-2.0 per the upstream model cards. No LICENSE file is fetched, so this is not verifiable from the tree. |
| Architecture | `Mistral3ForConditionalGeneration`, `text_config.model_type = ministral3` |
| Text model | 34 layers, hidden 4096, 32 query heads / 8 KV heads (GQA), head_dim 128, MLP 14336 (SwiGLU), vocab 131072, RMSNorm eps 1e-5, untied embeddings |
| Positions | YaRN RoPE, theta 1e6, factor 16 over an original 16384 window; llama-4 attention scale with beta 0.1, which is exactly 1.0 below position 16384 and therefore a no-op inside FROST's 4096-token budget |
| Quantization | MLX affine 4-bit, group size 64, packed U32 weights with bf16 scales and biases; embedding and `lm_head` are quantized too; activations bf16, logits read as f32 |
| Text-only | The vision tower (Pixtral config, about 0.8 GB in shard 1) is never loaded; image input is not supported |
| Weights in memory | 4.45 GiB copied into MLX (`weight_bytes` 4,775,780,352, reported by `frost models` and in every reply's meta) |

Integrity. `tools/fetch_ministral.sh` downloads by revision with curl, resumes partial files,
checks byte sizes, and checks sha256 against the upstream LFS digests for the two shards and
`tokenizer.json`:

```
model-00001-of-00002.safetensors  5294694324  0b7c0de8f4da647f054095add6d93c1cfdd7f7f38128ecb666ebace634169aa6
model-00002-of-00002.safetensors   301990235  aa49bdcf4394ab16f0cbd45b750313978073f514da8b56e802ee3c7568b3a865
tokenizer.json                      17077402  286acad9b0e27fce778ac429763536accf618ccb6ed72963b6f94685e531c5c7
```

The small JSON/Jinja files have no upstream digest; their local sha256 is recorded as `LOCAL` in
`MANIFEST.json`. `model.safetensors.index.json` is deliberately not fetched: at this revision it
lists FP8 tensor names that do not match the shards. The loader refuses a `MANIFEST.json` whose
revision differs from the pinned one, refuses any architecture, quantization mode, bit width or
RoPE type other than the ones above, and refuses a tokenizer whose vocabulary size differs from
the model's.

Load-time checks. Every quantized tensor (240 of them) passes through the Zig validator
(`frost_q4_validate`) before it is copied to MLX: shape and packing agree, sizes do not overflow,
and every scale and bias is finite. The result is reported as `layout_check`, for example
`zig q4_validate: 240 tensors, 0 non-finite scales/biases`.

Tokenizer. The pinned `tokenizer.json` through the Rust `tokenizers` crate (0.23, Oniguruma
regex backend). Special-token matching is disabled for content, so text such as `[INST]` stays
text. The 17 named control tokens (`<s>`=1, `</s>`=2, `[INST]`=3, `[/INST]`=4, `[TOOL_CALLS]`=9,
`[SYSTEM_PROMPT]`=17, `[ARGS]`=32 and the rest) are checked against the vocabulary at load, and
ids below 1000 are treated as control tokens. Output is detokenized incrementally from byte-level
BPE bytes, emitting only complete UTF-8 characters.

Chat template. The Mistral v13 template from the checkpoint's `chat_template.jinja`, rendered
directly to ids in Rust:

```
<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]([AVAILABLE_TOOLS]{json}[/AVAILABLE_TOOLS])?
([INST]{user}[/INST]{assistant}</s> | [TOOL_RESULTS]{result}[/TOOL_RESULTS])*
```

FROST always supplies its own system prompt, so the template's built-in default is never used.
Consecutive same-role messages are merged with a blank line (the upstream template would raise).
Tool calls are parsed only from real `[TOOL_CALLS]` and `[ARGS]` ids, never from text.

Sampling defaults (`SampleParams::default`, used by the app and engine):

| Parameter | Default | Note |
|---|---|---|
| temperature | 0.15 | Mistral's recommendation for this family is a low temperature; 0 means greedy argmax |
| top_k | 64 | Mojo bounded-heap top-k over all 131072 logits |
| top_p | 1.0 | Off; the Mojo softmax/top-p kernel still normalizes the 64 candidates |
| repetition penalty | 1.0 | Off by default; implemented in Rust over the last 64 generated ids |
| seed | 0x5eed | Seeded xorshift, so runs are reproducible |

NaN or degenerate logits fall back to argmax instead of sampling. Each reply records the backend
that ran: `mojo-dylib`, `rust-reference` (dylib not loaded) or `rust-argmax` (greedy).

Context. 4096 tokens per request in every mode, 1024 reserved for the reply. The checkpoint
supports far more (262144 positions); 4096 is FROST's budget until larger contexts are measured
for memory. A prompt that would not fit is truncated from the oldest turn, with a visible note; a
single message that alone exceeds 3072 tokens is refused.

Measured on this Mac (Mac17,2, Apple M5, 16 GB, macOS 27.2):

| Metric | Value |
|---|---|
| Load (weights into MLX) | about 7 to 9 s |
| Decode | 25 tokens/s (57 to 136 new tokens, temperature 0.15) |
| Prefill | 88 tokens in about 520 ms warm; the first call adds about 800 ms of Metal warm-up |
| Prefix reuse | 89 of 99 prompt tokens reused on the second turn of the engine acceptance test |
| Decode through the app | 23.3 tokens/s (66 new tokens, Performance mode, `verification/ui/ui-automation-run.log`) |
| Final acceptance run | Installed `diagnose --load`: 22.7 tokens/s, greedy (`rust-argmax`), one-word reply; installed app `--headless-check`: ready with `mlx_active_bytes` 4,775,780,608 (`tools/run_acceptance.sh`, `verification/results.json`, 2026-09-27T14:41:06Z) |

## 2. Retrieval encoder: nomic-embed-text-v1.5

| Field | Value |
|---|---|
| Checkpoint | `nomic-ai/nomic-embed-text-v1.5` @ `e9b6763023c676ca8431644204f50c2b100d9aab` |
| License | Apache-2.0 per the upstream model card (not verifiable from the tree) |
| Weights | `model.safetensors`, 546,938,168 bytes, sha256 `9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c` (upstream LFS digest, checked by `tools/fetch_nomic.sh`) |
| Architecture | nomic-bert: 768 hidden, 12 layers, 12 heads, SwiGLU, non-interleaved RoPE base 1000, post-norm, masked mean pooling, L2-normalized output |
| Precision / backend | Unquantized weights on MLX (GPU) |
| Task prefixes | `search_query: ` and `search_document: ` prepended to the text, per the model card |
| Verification | An independent f64 scalar Rust forward agrees with the MLX forward at cosine 1.000000; a behavioural test requires a relevant document to outscore an irrelevant one by 0.2 |
| Used by | `frost-repo` (repository indexing and hybrid search), which chat calls before every reply in a conversation with an attached repository; the encoder is loaded on first use. Also the `frost train` experiment, which chat does not use. |

## What is NOT claimed

- No quality claim for the generator on coding tasks. The coding fixtures record results: in the
  final acceptance run it solved 1 of 3 held-out tasks (c-parse-kv, after one repair) and 0 of 3
  on the first attempt.
- No golden-vector parity against upstream PyTorch for the generator: producing reference
  logits offline would need Python. Correctness rests on behavioural acceptance tests (correct
  answers to fixed questions, coherent multi-turn text, prefix reuse) and on unit tests that
  re-derive YaRN frequencies and the llama-4 scale independently.
- The encoder's parity is against FROST's own scalar reference, not upstream vectors.
- No claim of long-context behaviour beyond 4096 tokens, of vision, or of multilingual quality.
- Thermal state is an OS signal, not a temperature; no energy or power figures are claimed.
- The optional `frost-train` projection head did not beat the raw encoder on its held-out split
  and is not used anywhere in chat.
