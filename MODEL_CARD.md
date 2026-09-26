# FROST — Model Card

## Backbone (bootstrap encoder)

- **Model**: `nomic-ai/nomic-embed-text-v1.5`, revision
  `e9b6763023c676ca8431644204f50c2b100d9aab` (pinned).
- **Why this model**: a pragmatic, non-Qwen, pretrained text encoder used as the
  bootstrap. It is **not** the CLM/Contrastive-LM checkpoint, **not** a custom SSM
  encoder, and it does **not** inherit any other model's published benchmark
  results. FROST makes no accuracy claims about general intelligence.
- **Architecture** (read from `config.json`, not hardcoded): nomic-bert, hidden
  768, 12 layers, 12 heads, head dim 64, intermediate 3072, SwiGLU MLP
  (`fc11 * silu(fc12)`), rotary position embeddings (base 1000, non-interleaved,
  full head dim; no learned position embeddings), postnorm LayerNorm (eps 1e-12),
  no qkv/fc1 bias, token-type embeddings, vocab 30528.
- **Implementation**: Rust + `mlx-c` (Apple GPU). Embedding gather in Rust;
  LayerNorm/RoPE via MLX fast ops; attention computed manually (bidirectional,
  batch=1, scale 1/√64); mean pooling; L2 normalization. Task prefixes
  `search_query:` / `search_document:` per the model card.
- **Tokenizer**: BERT-uncased WordPiece over the model's real `vocab.txt`
  (from-scratch, no HF `tokenizers` dependency). Accent stripping omitted —
  an approximation vs the upstream normalizer for non-ASCII input.

### Verification of the backbone
- Independent **f64 scalar reference forward** (pure Rust). MLX vs scalar cosine
  parity **1.000000** on a real input.
- Behavioral: relevant vs irrelevant documents separate clearly (0.875 vs 0.445;
  "dog" vs "quantum field theory" 0.413). Deterministic, L2-normalized outputs.
- **Verification gap**: no publisher-provided golden vectors and no non-Python
  reference available offline, so parity is against our own reference rather than
  upstream PyTorch. Shape checks alone are not treated as correctness.

## Trained head (FROST's own)

- **What**: two projection heads (state, action) 768 → 256 into a shared cosine
  space, trained with a symmetric InfoNCE contrastive objective over the frozen
  backbone embeddings. This is a genuinely trained head — analytic gradients are
  **finite-difference-verified** (max rel. error 0.003) — not a random init
  described as trained.
- **Data**: a bounded, **agent-authored synthetic routing dataset** (8 intents:
  music, files, email, brightness, alarm, web, screenshot, lock). 16 train / 16
  eval states. Train and eval use **different phrasing templates per intent**
  (family split — no paraphrase leakage). This is a synthetic demonstration
  dataset, **not** evidence of general intelligence. Seed and data hash recorded.
- **Result**: loss 2.153 → 0.006; held-out top-1 routing 0.9375 (trained head)
  vs 0.9375 (raw-embedding baseline). Checkpoint save/reload reproduces
  projections. The head matched but did not beat the baseline.
- **Enabled?** No. Per the promotion rule, a head is enabled by default only
  after it meets an explicit recorded criterion beating the reference on held-out
  data. It did not, so the **full-depth reference ranker stays enabled** and the
  head ships as a validated-but-disabled checkpoint.

## Progressive early-exit heads

- **NOT IMPLEMENTED.** Intermediate encoder exits (~1/3, 2/3 depth), per-exit
  trained projectors, nested widths 64/128/256 with per-exit calibration,
  resume-from-hidden-states, and executed-block counters are not built. Only
  full-depth inference exists. This is stated plainly and not simulated.

## Capability matrix

| Component                     | implemented | trained | validated | enabled |
|-------------------------------|:-----------:|:-------:|:---------:|:-------:|
| Full-depth encoder (MLX)      | yes         | n/a     | yes¹      | yes     |
| Zig retrieval-index kernels   | yes         | n/a     | yes       | yes     |
| Mojo IPC scoring kernel       | yes         | n/a     | yes       | yes     |
| Calibration margin-gate       | yes         | n/a     | yes       | yes     |
| Full-depth projection head    | yes         | yes     | yes       | no²     |
| Progressive early-exit heads  | no          | no      | no        | no      |

¹ against our scalar reference + behavioral checks, not upstream golden vectors.
² validated but did not beat the reference on held-out routing.

## Intended use / limits

Ranking allowed candidate actions for a described state, offline, on-device.
It does not execute actions. It makes no guarantees of correctness, no energy or
zero-heat claims, and no benchmark inheritance. Confidence is calibrated only
where the margin gate applies; otherwise it abstains on the pick.
