# FROST — Dependencies & Notices

## Toolchains (pinned; verified present at build)

| Tool   | Version                     | Role                              | Source            |
|--------|-----------------------------|-----------------------------------|-------------------|
| Rust   | 1.98.1 (Homebrew)           | app core, FFI, CLI, desktop       | Homebrew          |
| Zig    | 0.16.0                      | action-index C-ABI library        | Homebrew          |
| Mojo   | 1.1.0.dev2026082005         | compiled numerical kernel (build) | MAX nightly       |
| MLX    | 0.32.1                      | Apple-GPU tensor ops (runtime)    | Homebrew (`mlx`)  |
| mlx-c  | 0.6.0_4                     | C API for MLX                     | Homebrew (`mlx-c`)|
| clang  | Apple (Command Line Tools)  | ObjC shim, FFI glue               | Xcode CLT         |

Full Xcode is **not** required. MLX ships a precompiled metallib, so the GPU path
runs with Command Line Tools only (`metal`/`actool` absent — see IMPLEMENTATION_STATUS).

## Rust crates (direct)

| Crate        | Version | Used by            | Why (ladder: reuse over rewrite)         |
|--------------|---------|--------------------|------------------------------------------|
| serde        | 1       | most crates        | (de)serialization of types/checkpoints   |
| serde_json   | 1       | model/train/cli    | config, JSON output, checkpoints         |
| anyhow       | 1       | model/service/…    | error propagation                        |
| thiserror    | 2       | core/index/service | typed errors                             |
| memmap2      | 0.9     | frost-model        | mmap safetensors (avoid 522 MiB copy)    |
| half         | 2       | frost-model        | f16 support (declared; fp32 weights used)|
| wry          | 0.52    | frost-desktop      | system webview shell                     |
| tao          | 0.35    | frost-desktop      | window/event loop for wry                |
| cc           | 1 (build)| frost-platform    | compile the ObjC NSProcessInfo shim      |

No Python, PyO3, CPython, PyTorch, or ONNX-runtime crates. No HTTP/network-client
crate (reqwest/hyper/ureq/curl) anywhere in the graph — the engine is offline by
construction; `curl` is used only by `bootstrap.sh` to download weights.

## Third-party model & assets

- **nomic-ai/nomic-embed-text-v1.5** — weights, `config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`. License:
  Apache-2.0 (upstream). Downloaded at pinned revision
  `e9b6763023c676ca8431644204f50c2b100d9aab` into Application Support. FROST does
  not redistribute the weights; `bootstrap.sh` fetches them from Hugging Face.
- The nomic-bert architecture and RoPE/SwiGLU conventions were read from the
  upstream `config.json` and the public `modeling_hf_nomic_bert.py` (consulted for
  block structure only; no upstream code is vendored).

## Notices

- MLX and mlx-c are © Apple, MIT-licensed. Linked dynamically from Homebrew.
- Zig standard library © the Zig contributors, MIT.
- Mojo/MAX © Modular; used as a build-time compiler only. The compiler ships
  inside a Python virtual environment (build-time tool); the FROST runtime links
  no libpython (see VERIFICATION §11).
- Rust crates retain their own MIT/Apache-2.0 licenses; see each crate.

FROST's own source is dual MIT OR Apache-2.0.
