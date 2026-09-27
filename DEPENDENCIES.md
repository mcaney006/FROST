# FROST dependencies

Everything FROST needs at build time, at run time, and inside `FROST.app`, with the reason each
piece is there. Versions come from `Cargo.lock`, the crate manifests, and the machine the project
was built on (Mac17,2, macOS 27.2, Command Line Tools only).

## Toolchains

| Tool | Version | Used for | Required? |
|---|---|---|---|
| Rust / Cargo | 1.98.1 (`rust-version = "1.98"`, `rust-toolchain.toml`: stable) | All crates | Yes |
| Command Line Tools, clang | clang 21 (Apple) | `cc` builds of the Objective-C shell, `platform_mac.m`, vendored Oniguruma and SQLite; linking | Yes. Full Xcode and the `metal` compiler are not needed: MLX ships a prebuilt `mlx.metallib` |
| Zig | 0.16.0 | `zig build-lib` of `native/index/frost_index.zig` from `frost-index/build.rs` | Yes |
| MLX | 0.32.1 (Homebrew `mlx`) | GPU tensor runtime; `libmlx.dylib` + `mlx.metallib` | Yes |
| mlx-c | 0.6.0_4 (Homebrew `mlx-c`) | C API over MLX; `libmlxc.dylib` | Yes |
| Mojo compiler | 1.1.0.dev2026082005 (nightly), native Mach-O at `$FROST_MOJO_BIN` | Builds `libfrost_kernels.dylib` | Optional: without it the Rust reference sampler is used and labelled |
| Homebrew | any | Installing missing Rust, Zig, MLX, mlx-c | Only if one of those is missing |
| curl, shasum, codesign, install_name_tool, otool | system | Model fetch, digests, bundle staging and verification | Yes (bootstrap checks) |

The only Mojo compiler on this machine lives inside a Python virtual environment
(`~/.local/share/max-codex/max-nightly-env/.../modular/bin/mojo`). The build runs that native
binary directly with a scrubbed environment and never executes Python; see
`verification/mojo_audit.md`. No official Modular channel installs the compiler without a Python
interpreter, so a Python-free install of the compiler itself is BLOCKED upstream. Python is never
installed by FROST's scripts.

## Rust crates (direct dependencies)

| Crate | Locked version | License | Used by | Why |
|---|---|---|---|---|
| `tokenizers` (no default features, `onig`) | 0.23.2 | Apache-2.0 | frost-gen | Exact byte-level BPE of the pinned `tokenizer.json`; no hub or network features |
| `onig` / `onig_sys` | 6.5.3 / 69.9.3 | MIT (vendored Oniguruma C: BSD-style, `oniguruma/COPYING`) | via tokenizers | Regex engine for the pre-tokenizer |
| `rusqlite` (`bundled`) | 0.40.2 (libsqlite3-sys 0.38.2, SQLite 3.53.2) | MIT (SQLite: public domain) | frost-store | Conversations, attempts, chunks, embedding cache; bundled so no system SQLite is assumed |
| `memmap2` | 0.9.11 | MIT OR Apache-2.0 | frost-tensors | Zero-copy views of multi-GB safetensors shards |
| `serde`, `serde_json` | 1.0.229, 1.0.151 | MIT OR Apache-2.0 | most crates | Config, ABI JSON, events, stored meta |
| `thiserror` | 2.0.21 | MIT OR Apache-2.0 | most crates | Typed errors |
| `anyhow` | 1.0.104 | MIT OR Apache-2.0 | frost-cli, frost-model, frost-train, frost-gen | Error context in binaries and the encoder |
| `sha2` | 0.11.0 (frost-tools), 0.10.9 (frost-repo) | MIT OR Apache-2.0 | frost-tools, frost-repo | File base hashes for patches; chunk digests. Two versions are in the lock file |
| `ignore` | 0.4.33 | Unlicense OR MIT | frost-repo | `.gitignore`-aware walk that does not follow symlinks |
| `cc` (build) | 1.5.1 | MIT OR Apache-2.0 | frost-desktop, frost-platform | Compile Objective-C sources |

`Cargo.lock` holds 137 packages, workspace crates and transitive dependencies included. `tools/python_audit.sh` checks the graph
for Python bindings (pyo3, cpython, python3-sys, inline-python, rustpython) and
`tools/offline_audit.sh` checks it for HTTP/TLS clients (reqwest, hyper, ureq, curl, hf-hub,
tokio, rustls, native-tls, openssl). Both audits passed in the final acceptance run: no
Python crates in normal or build dependencies (python audit 5/5), and no network crates in the
runtime dependencies with 0 internet sockets during a real generation (offline audit).

## Models

| Model | Revision | Size | License | Fetched by |
|---|---|---|---|---|
| `mlx-community/Ministral-3-8B-Instruct-2512-4bit` | `182f003f01daa75f9de0f2c4d379722fd0bc1c61` | 5.6 GB (2 shards + tokenizer + config) | Apache-2.0 (upstream model card) | `tools/fetch_ministral.sh` |
| `nomic-ai/nomic-embed-text-v1.5` | `e9b6763023c676ca8431644204f50c2b100d9aab` | 547 MB | Apache-2.0 (upstream model card) | `tools/fetch_nomic.sh` |

Both are stored under `~/Library/Application Support/FROST/models/` with a `.sha256` marker per
file and a `MANIFEST.json`. They are not bundled into the app and are kept by `uninstall.sh`
unless `--purge` is given. `FROST_MODELS_DIR` overrides the download location for the fetch
scripts only; the app and CLI always read the Application Support path.

## Vendored into FROST.app/Contents/Frameworks

| File | Source | Install name in the bundle |
|---|---|---|
| `libmlx.dylib` | `/opt/homebrew/opt/mlx/lib` | `@rpath/libmlx.dylib` |
| `mlx.metallib` (symlink to `../Resources/mlx.metallib`) | created by bootstrap | not code; MLX finds the metallib next to `libmlx.dylib` |
| `libmlxc.dylib` | `/opt/homebrew/opt/mlx-c/lib` | `@rpath/libmlxc.dylib` |
| `libfrost_kernels.dylib` | built from `native/kernels/frost_sampling.mojo` (about 39 KB) | `@rpath/libfrost_kernels.dylib`; non-`@loader_path` rpaths deleted |
| `libKGENCompilerRTShared.dylib`, `libAsyncRTMojoBindings.dylib`, `libMSupportGlobals.dylib`, `libAsyncRTRuntimeGlobals.dylib` | Modular SDK `lib/` next to the compiler | resolved through `@loader_path` |

The metallib itself (129 MB, from `/opt/homebrew/opt/mlx/lib`) is copied to
`Contents/Resources/mlx.metallib`: codesign treats everything under `Frameworks` as nested code
and rejects a metallib there. The app binary is `Contents/MacOS/FROST` and the CLI is
`Contents/Helpers/frost` (on case-insensitive APFS, `MacOS/frost` would be the same file as
`MacOS/FROST`). Both executables get `@executable_path/../Frameworks` as an rpath, and their
only non-system entries in `otool -L` are `@rpath/libmlxc.dylib` and `@rpath/libmlx.dylib`.
Bootstrap refuses to install a bundle in which `otool -L` still shows `/opt/homebrew` or
`site-packages`. Each dylib and `Helpers/frost` is signed ad-hoc, then the bundle, and
`codesign --verify --deep --strict` passes. The Mojo dylibs are copied only when the Mojo build succeeded.

## Notices

- MLX: MIT License, Copyright 2023 Apple Inc. mlx-c: MIT License, Copyright 2023 ml-explore.
  Both are redistributed inside the app bundle.
- Modular runtime dylibs: the `mojo-compiler` package declares
  `LicenseRef-MAX-Platform-Software-License`. Redistributing these four dylibs inside FROST.app
  has not been reviewed against those terms; do that before distributing a build to anyone else.
- Model weights keep their upstream licenses; FROST does not modify or redistribute them.
- The app is signed ad-hoc for local use only.
- Rust crate licenses are listed above; full texts are in each crate's source in the Cargo registry.
