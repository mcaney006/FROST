# Mojo integration audit — Python-free compile path and C-ABI buffer kernels

Date: 2026-09-26/27. Host: macOS 27.2 (Darwin 27.2.0), Apple M5, Command Line Tools only.
Branch: `frost-chat`. Scope of edits: `native/kernels/frost_sampling.mojo`, `crates/frost-kernels/**`, this file.

## Verdict

**Mojo integration Python-free: YES** — for *building and running* kernels.
**Installing** Mojo without a Python interpreter: **NO** (no official route).

Reasoning: the `mojo` on PATH is a Python launcher script, but it only `execve`s a native
Mach-O compiler (`<sdk>/modular/bin/mojo`) after setting four environment variables. Invoked
directly with those variables, that binary compiles Python-free Mojo source with **no libpython
loaded, no python process spawned, no python on PATH, HOME unwritable** (evidence below). The
resulting dylib depends only on four Modular runtime dylibs plus libSystem. The compiler and
runtime are therefore Python-free at build and run time. Every official *distribution* channel
(pip wheel, conda `mojo` / `mojo-compiler`) declares a `python` dependency, so a Python
interpreter is installed alongside the SDK — it is a packaging dependency only, never executed
by this project (build.rs runs the native binary with `env_clear()`).

| Item | Result |
|---|---|
| 1a. Locate native compiler + env contract | PASS |
| 1b. Direct native compile is Python-free | PASS |
| 1c. Official install route without Python interpreter | FAIL (none exists) |
| 1c. Native compiler needs Python for non-interop source | PASS (it does not) |
| 2. C-ABI buffer export from Mojo → callable from C | PASS (all 5 spellings) |
| 3. `frost_sampling.mojo` + `frost-kernels` crate + tests | PASS (5/5, `FROST_REQUIRE_MOJO=1`) |
| 3. Reference-only path when compiler missing | PASS (builds; tests FAIL under `FROST_REQUIRE_MOJO=1`) |

## 1a. Entrypoint and native binary

```
$ readlink -f ~/.local/bin/mojo
/Users/michael.jr/.local/share/max-codex/max-nightly-env/bin/mojo        # python script (#!.../bin/python)
```
`site-packages/mojo/_entrypoints.py::exec_mojo` → `os.execve(env["MODULAR_MOJO_MAX_DRIVER_PATH"], sys.argv, env)`
with `env = _sdk_default_env() | os.environ`, where `_sdk_default_env()` (`mojo/run.py`) sets:
```
MODULAR_MAX_PACKAGE_ROOT       = <root>
MODULAR_MOJO_MAX_PACKAGE_ROOT  = <root>
MODULAR_MOJO_MAX_DRIVER_PATH   = <root>/bin/mojo
MODULAR_MOJO_MAX_IMPORT_PATH   = <root>/lib/mojo
<root> = site-packages/modular   (mojo/_package_root.py, "wheel root")
```
```
$ file <root>/bin/mojo
Mach-O 64-bit executable arm64      (119.6 MB; sha256 b58344ba4efca3825266fd34925f8a6f38e49c5ab8db4bd91428f8ffbba4c96e)
$ <root>/bin/mojo --version
Mojo 1.1.0.dev2026082005 (c72288dd)
$ otool -L <root>/bin/mojo        # no libpython; only system frameworks + two @rpath SDK libs
  /usr/lib/libc++.1.dylib  Foundation  CoreFoundation  libobjc  IOKit  CoreGraphics  Metal
  SystemConfiguration  libSystem  libbsm  ApplicationServices  CoreText  Security
  @rpath/libMSupportGlobals.dylib  @rpath/libAsyncRTRuntimeGlobals.dylib
$ otool -l <root>/bin/mojo | grep -A2 LC_RPATH | grep path
  @loader_path/../lib   (+ bazel runfiles paths)
$ strings <root>/bin/mojo | grep -iE 'libpython|Python\.framework|python3\.'      → no output
```
The two rpath libs (and `libKGENCompilerRTShared`, `libAsyncRTMojoBindings` used by compiled
output) link only against each other and system libs (`otool -L` on each: no libpython).

## 1b. Direct native compile, Python-free (evidence)

Setup: `PATH` = a temp dir containing only symlinks `cc c++ clang clang++ ld xcrun ar strip dsymutil codesign`
(`command -v python python3 python3.12` → none), `HOME=/nonexistent`, `MOJO_PYTHON_LIBRARY=/nonexistent`.
Sampler: `ps -axo pid=,comm=` filtered to executables named `python*`, every 50 ms, diffed against baseline.

```
$ env -i PATH=$BIN HOME=/nonexistent TMPDIR=$T MODULAR_HOME=$T/modhome \
    MODULAR_MOJO_MAX_PACKAGE_ROOT=$ROOT MODULAR_MOJO_MAX_IMPORT_PATH=$ROOT/lib/mojo \
    MOJO_PYTHON_LIBRARY=/nonexistent DYLD_PRINT_LIBRARIES=1 \
    $ROOT/bin/mojo build hello.mojo -o $T/hello
build exit=0
dyld log: 682 lines; grep -i python (excluding the SDK's own site-packages path string) → NONE
non-system libraries loaded: <root>/bin/mojo, <root>/lib/libMSupportGlobals.dylib, <root>/lib/libAsyncRTRuntimeGlobals.dylib
sampler: 55 samples @50 ms; new python interpreter PIDs vs baseline → none
$ env -i $T/hello
hello from mojo
$ otool -L $T/hello → @rpath/libKGENCompilerRTShared.dylib, @rpath/libAsyncRTMojoBindings.dylib, libSystem
```
Stderr only: `Failed to initialize Crashpad ... unable to locate crashpad handler executable` (harmless,
because `HOME` is unwritable) and `ld: warning: object file ... built for newer 'macOS' version (27.2) than being linked (27.0)`.

### Exactly which env vars the native binary needs (probed by elimination)

| Setting | Required? | Failure without it |
|---|---|---|
| `MODULAR_MOJO_MAX_IMPORT_PATH=<root>/lib/mojo` | yes | `failed to parse the provided Mojo source module` (no stdlib) |
| `MODULAR_MOJO_MAX_PACKAGE_ROOT=<root>` | yes | `error: unable to locate Mojo CompilerRT library` |
| writable cache: `MODULAR_HOME=<dir>` **or** writable `HOME` | yes (one of) | `<HOME>/.cache/modular could not be created: Read-only file system` |
| a C compiler named `cc` on `PATH` (Xcode CLT `/usr/bin/cc`) | yes | `error: unable to find suitable c compiler for linking` (clang/ld/xcrun alone are not enough) |
| `MODULAR_MAX_PACKAGE_ROOT`, `MODULAR_MOJO_MAX_DRIVER_PATH` | no | (set by the Python launcher; unused by `build`) |
| `MOJO_PYTHON_LIBRARY` | no | set to `/nonexistent` as a hard guarantee; no effect on non-interop source |

`PACKAGE_ROOT` alone does not imply the import path; both must be set. Verified: `MINIMAL OK` with
exactly `MODULAR_MOJO_MAX_PACKAGE_ROOT + MODULAR_MOJO_MAX_IMPORT_PATH + MODULAR_HOME`.

## 1c. Official distribution

Docs fetched (Markdown renditions; append `.md` to a mojolang.org URL):
- `https://mojolang.org/install.md`: "Mojo installs exactly like a Python or Conda package on macOS and Linux".
  Only listed routes: `pixi add mojo` / `pixi init ... -c https://conda.modular.com/max-nightly/ -c conda-forge`,
  `uv pip install mojo` / `uv add mojo`. (docs.modular.com/mojo/manual/install meta description:
  "You can install Mojo using pixi, uv, conda, pip, or other Python/Conda package managers.")
- `https://mojolang.org/docs/requirements.md` — Software: "A C compiler (such as cc, gcc, or clang) on Linux — used as a linker.
  Xcode or Xcode Command Line Tools 16 or later on macOS." Python is **not** listed as a runtime requirement.
- GitHub `modular/modular` `mojo/docs/manual/*.md{,x}` → 404 (docs source not at that path); the rendered `.md` above was used instead.

Conda repodata (`https://conda.modular.com/max-nightly/osx-arm64/repodata.json`, 4.0 MB, 2026-09-26; noarch 2.4 MB):
```
mojo-1.2.0.dev2026092605-release.conda            depends: python >=3.10 | mojo-compiler ==1.2.0.dev2026092605 | mblack ==26.7.0.dev2026092605 | jupyter_client >=8.6.2,<8.7
mojo-compiler-1.2.0.dev2026092605-release.conda   depends: mojo-python ==1.2.0.dev2026092605
mojo-python-1.2.0.dev2026092605-release.conda     depends: python >=3.10        (noarch)
mojo-compiler-1.1.0.dev2026082005 (installed ver) depends: mojo-python ==1.1.0.dev2026082005
max-core-26.7.0.dev2026092605                     depends: mojo-compiler ==26.7.0.dev2026092605-era
```
Every `mojo-compiler` build since `0.25.6.0` depends on `mojo-python`, which depends on `python`; the 16
oldest builds (25.5.0 .. 25.6.0.dev2025081205) had no depends at all. `mojo-dev` is stale (25.5.0).
Channel package names (osx-arm64): `max max-core max-python mojo mojo-compiler mojo-dev`; noarch: `max max-all max-benchmark max-pipelines max-serve mblack modular mojo-jupyter mojo-python`.

pip METADATA in the venv (never ran pip):
```
mojo 1.1.0.dev2026082005                  Requires-Dist: mojo-compiler==…, mblack==26.6.0.dev2026082005, mojo-lldb-libs==…
mojo-compiler 1.1.0.dev2026082005         Requires-Dist: mojo-compiler-mojo-libs==…
mojo-compiler-mojo-libs / mojo-lldb-libs  (no Requires-Dist)
max 26.6.0.dev2026082005                  Requires-Python: >=3.10,<3.15; Requires-Dist: max-core, click, msgspec, numpy, psutil, rich, …
max-core 26.6.0.dev2026082005             Requires-Dist: mojo-compiler==…, max-mojo-libs==…
modular 26.6.0.dev2026082005              Requires-Dist: max[benchmark], max[serve], mojo
venv interpreter: uv-managed CPython 3.12.13 (~/.local/share/uv/python/cpython-3.12-macos-aarch64-none)
```
Plain statement: **no official route installs the compiler without also installing a Python interpreter**
(pip wheels are Python packages by construction; the conda `mojo-compiler` package depends on `python` via
`mojo-python`). **The native compiler binary itself does not need Python** for a non-Python-interop source
(1b). Compiled artifacts do not need Python either (1b, 2).

## 2. C-ABI buffer export feasibility

Stdlib packages ship as `MPKG` binaries (`std.mojoc`; `strings` finds no identifiers), so API spellings were
taken from the `.mojo` sources shipped in `site-packages/max/**` and confirmed by compiling. Findings for this nightly:

- `@export` decorator with `abi("C")` **after the signature**: `@export\ndef f(...) abi("C") -> Int32`.
  `@export("name")` also works. The form noted earlier in this repo, `@export("name", abi("C"))`, is **rejected**:
  `error: @export requires a string specifying the name of the exported symbol`.
- `UnsafePointer` is deprecated (`warning: 'UnsafePointer' is deprecated, use 'Pointer' instead`). The idiom is
  `Pointer[Float32, MutAnyOrigin](unsafe_from_address=addr)`; index with `p[unsafe_offset=i]` (positional
  `__getitem__` is deprecated). Origins: `MutAnyOrigin`, `ImmutAnyOrigin`, `MutUntrackedOrigin`.
- `OpaquePointer[MutUntrackedOrigin]` + `.unsafe_bitcast[Float32]()` (`bitcast` deprecated).
- `sort` is a builtin: `sort(list, cmp_fn)` (comparator as a runtime argument; `std.algorithm` has no `sort`,
  `List` has no `.sort`). `from std.math import exp, isnan, isinf`, `from std.memory import Pointer, OpaquePointer`.

All five parameter spellings build as `--emit shared-lib`, export the symbol (`nm -gU`), and pass a C test that
`dlopen`s each dylib and scales a real 8-float buffer by 2.5 (`clang -Wall -Wextra ctest.c`):
```
./liba_int_addr.dylib  probe_a_scale (addr: Int → Pointer[..](unsafe_from_address=))     rc=1 buf=2.5 5 7.5 10 → PASS
./libb_opaque.dylib    probe_b_scale (op: OpaquePointer[MutUntrackedOrigin] + bitcast)   → PASS
./libc_unsafeptr.dylib probe_c_scale (p: UnsafePointer[Float32, MutAnyOrigin], deprecated) → PASS
./libd_pointer.dylib   probe_d_scale (p: Pointer[Float32, MutAnyOrigin])                  → PASS
./libe_untracked.dylib probe_e_scale (p: Pointer[Float32, MutUntrackedOrigin])            → PASS
```
The prior failure ("parametric over origin") does not reproduce once the origin is spelled explicitly; the
`Int`-address form (A) was chosen for the real kernel because it keeps the exported signature free of Mojo types.
`Float32` scalars pass by value in the C ABI (verified: `k: Float32` = 2.5).

Compiled-dylib runtime dependencies (`otool -L` + Python-free `DYLD_PRINT_LIBRARIES` dlopen trace):
```
@rpath/libKGENCompilerRTShared.dylib  @rpath/libAsyncRTMojoBindings.dylib  /usr/lib/libSystem.B.dylib
→ transitively libMSupportGlobals.dylib, libAsyncRTRuntimeGlobals.dylib  (all from <root>/lib; none is Python)
LC_RPATH: <root>/lib (baked by the driver) [+ @loader_path when passed -Xlinker -rpath -Xlinker @loader_path]
```
`mojo build` flags used: `--emit shared-lib -O3 -Xlinker -rpath -Xlinker @loader_path` (`-O` default is 3; range 0–3).

## 3. Implementation

`native/kernels/frost_sampling.mojo` — exported C signatures (all return `Int32`: 0 = ok, negative = error):
```
int32_t frost_kernels_abi_version(void);                                              // returns 1
int32_t frost_topk_f32(int64_t logits_addr, int64_t n, int64_t k,
                       int64_t out_idx_addr /*uint32_t[k]*/, int64_t out_val_addr /*float[k]*/);
int32_t frost_softmax_topp_f32(int64_t logits_addr, int64_t n, float temperature, float top_p,
                       int64_t out_prob_addr /*float[n]*/, int64_t out_idx_addr /*uint32_t[n]*/,
                       int64_t out_n_addr /*int64_t*/);
codes: -1 n==0  -2 k==0  -3 k>n  -4 NaN logit  -5 temperature NaN or <=0  -6 top_p NaN or ∉(0,1]
       -7 degenerate (max scaled logit not finite: any +inf, or all -inf)
```
top-k: descending, ties → lower index; `-inf` allowed. Implemented as a bounded min-heap of size k over the
input scan (O(n log k), the input is never sorted): the heap root is the element to evict next — smallest value,
equal values with the higher index first — and because indices are scanned ascending only a strictly larger
value displaces the root, so later equal values never replace earlier ones. The k survivors are then sorted.
No large-k fallback: at k = n it degenerates to heapify + sort, same order as the old full sort (6.9 ms).

Performance (`cargo run --release -p frost-kernels --example bench`, n = 131072, best of 20; C harness for other k):
```
                          before (full sort)   after (k-heap)      Rust reference
topk k=64                 5.937 ms             0.044 ms            8.0 – 8.7 ms
topk k=1 / 1024 / 4096    –                    0.046 / 0.214 / 1.148 ms
topk k=16384 / 131072     –                    3.07 / 6.94 ms
softmax_topp n=131072     6.67 ms              unchanged (6.7–8.6 ms run-to-run under concurrent builds)
softmax_topp n=64         0.001 ms             unchanged
```
softmax left as a full sort on purpose: on the bench data (t=0.7, p=0.95) the kept prefix is 13712 of 131072
entries, so a heap pre-selection of "candidates that could matter" is no cheaper than the sort, and the hot
path (softmax over the 64 top-k survivors) is already 1 µs. A tiered heap select (top-M then fall back to the
full sort when `cum < top_p`) would keep results bit-identical (same elements in the same order → same f64
prefix sums) if a peaked-distribution case ever shows up in profiling.

softmax: `z = x * (1/t)` in Float64, max-subtracted
`exp`, Float64 sum, sort desc (ties → lower index), keep the smallest prefix with `cum >= top_p` (all `n` if
rounding never reaches it), renormalize by the kept mass. Direct C check of the real dylib (Python-free env):
```
abi_version=1 | topk [1,+inf,3,-inf,3] → idx 1 2 4 0 3 | codes -1 -2 -3 -4 | softmax [10,0,0,0] p=0.1 → count 1 idx 0 prob 1
softmax p=1.0 → count 4 sum 1.000000 order 0 1 2 3 | codes t0=-5 topp0=-6 topp1.5=-6 +inf=-7 all-inf=-7
```

`crates/frost-kernels/build.rs`: runs `$FROST_MOJO_BIN` (default
`$HOME/.local/share/max-codex/max-nightly-env/lib/python3.12/site-packages/modular/bin/mojo`) with `env_clear()` and
exactly `PATH=/usr/bin:/bin TMPDIR=$OUT_DIR MODULAR_HOME=$OUT_DIR/modular_home MODULAR_MOJO_MAX_PACKAGE_ROOT=<root>
MODULAR_MOJO_MAX_IMPORT_PATH=<root>/lib/mojo MOJO_PYTHON_LIBRARY=/nonexistent`, emits
`cargo:rustc-env=FROST_KERNELS_DYLIB=$OUT_DIR/libfrost_kernels.dylib`; on any failure emits
`cargo:rustc-cfg=mojo_unavailable` + `cargo:warning` and the crate still builds.

`crates/frost-kernels/src/lib.rs`: `backend()` → `"mojo-dylib" | "unavailable"`, `load_error()`, `topk`,
`softmax_topp`, `reference` module, `KernelError` (thiserror, `PartialEq`). Loads via std-only
`extern "C" { dlopen; dlsym; dlerror }` (`RTLD_NOW`), checks `frost_kernels_abi_version() == 1`. Search order:
`$FROST_KERNELS_DYLIB` (runtime override) → build-time path → `<exe dir>/libfrost_kernels.dylib` →
`<exe dir>/../Frameworks/libfrost_kernels.dylib`.

### Tests (`cargo test -p frost-kernels`)
```
FROST_REQUIRE_MOJO=1 cargo test -p frost-kernels
test tests::backend_reports_and_honours_requirement ... ok
test tests::topk_edge_cases ... ok
test tests::softmax_topp_edge_cases ... ok
test tests::topk_parity_with_reference ... ok            # n ∈ {1,7,64,131072} × k ∈ {1,5,50}, exact Result equality
test tests::softmax_topp_parity_with_reference ... ok    # same n × (t,p) ∈ {(1,1),(0.7,0.1),(2,0.5)}; idx exact, prob ≤1e-6 rel
test tests::topk_heavy_ties_and_k_up_to_n ... ok         # 1000 values with 5 distinct levels, k ∈ {1,37,200,999,1000}; k = n-1, n
test result: ok. 6 passed; 0 failed
```
Edge cases covered: empty, NaN, k>n, k=0, +inf/-inf (topk keeps them; softmax `+inf`/all-`-inf` → `Degenerate`,
lone `-inf` → zero mass), temperature 0/negative → `InvalidTemperature`, top_p 0/1.5 → `InvalidTopP`, top_p 1.0
(all mass) and 0.1 (minimal prefix). `FROST_REQUIRE_MOJO=1` asserts `backend()=="mojo-dylib"` in every test.
Forced cargo rebuild with the python-interpreter sampler (65 samples @50 ms): new python PIDs → none.
```
$ FROST_MOJO_BIN=/nonexistent/mojo cargo test -p frost-kernels
warning: frost-kernels@0.1.0: frost-kernels: Mojo backend unavailable: compiler not found at /nonexistent/mojo (set FROST_MOJO_BIN)
test result: ok. 5 passed; 0 failed            # reference backend exercised
$ FROST_MOJO_BIN=/nonexistent/mojo FROST_REQUIRE_MOJO=1 cargo test -p frost-kernels
test tests::backend_reports_and_honours_requirement ... FAILED   (all 5 FAILED, none skipped)
assertion `left == right` failed: Mojo backend required but unavailable: Some("dylib was not built: Mojo compiler unavailable at build time (see cargo warnings)")
$ FROST_REQUIRE_MOJO=1 cargo test -p frost-kernels          # restored
test result: ok. 5 passed; 0 failed
```

## Integrator notes

- Runtime: `libfrost_kernels.dylib` must sit next to the four SDK runtime dylibs, or reach them through its rpath
  (`<root>/lib` is baked in; `@loader_path` is added). For `FROST.app`, copy `libfrost_kernels.dylib` **and**
  `libKGENCompilerRTShared.dylib libAsyncRTMojoBindings.dylib libMSupportGlobals.dylib libAsyncRTRuntimeGlobals.dylib`
  from `<root>/lib` into `Contents/Frameworks/`; the loader finds `<exe>/../Frameworks/libfrost_kernels.dylib`.
  Or point `FROST_KERNELS_DYLIB` at any copy. Dylib size ≈ 39 KB.
- `bootstrap.sh` still runs `mojo build` = the Python launcher (lines 21–35, `frost_mojo_helper`). Out of scope
  here; switch it to `<root>/bin/mojo` with the env in 1b, or drop the helper in favour of this crate.
- The old repo note `@export("name", abi("C"))` is wrong for this nightly; use `@export` + `abi("C")` suffix.
- `mojo` nightlies move fast (1.1.0.dev2026082005 installed vs 1.2.0.dev2026092605 published). Pin via `FROST_MOJO_BIN`.
- Deprecation warnings avoided: no `UnsafePointer`, no positional pointer indexing, no `bitcast`.
