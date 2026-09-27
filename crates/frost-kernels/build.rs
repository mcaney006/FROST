// Compiles native/kernels/frost_sampling.mojo into OUT_DIR/libfrost_kernels.dylib with the
// *native* Mojo driver. The `mojo` on PATH is a Python entrypoint script (execve's this same
// binary), so we never touch it: we run <sdk>/modular/bin/mojo directly with a scrubbed env.
// Evidence for the env contract: verification/mojo_audit.md.
//
// Compiler lookup: $FROST_MOJO_BIN, else DEFAULT_MOJO_BIN. If the compiler is missing or the
// build fails we emit `mojo_unavailable` and the crate still builds (reference backend only).
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_MOJO_BIN: &str =
    ".local/share/max-codex/max-nightly-env/lib/python3.12/site-packages/modular/bin/mojo";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(mojo_unavailable)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=FROST_MOJO_BIN");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("../../native/kernels/frost_sampling.mojo");
    println!("cargo:rerun-if-changed={}", src.display());

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dylib = out.join("libfrost_kernels.dylib");
    println!("cargo:rustc-env=FROST_KERNELS_DYLIB={}", dylib.display());

    let mojo = std::env::var_os("FROST_MOJO_BIN").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(DEFAULT_MOJO_BIN)
    });
    match build(&mojo, &src, &dylib, &out) {
        Ok(()) => {}
        Err(msg) => {
            println!("cargo:warning=frost-kernels: Mojo backend unavailable: {msg}");
            // A dylib left over from an earlier build must not outlive this decision: the crate now
            // reports the backend unavailable, and bootstrap must not bundle a stale kernel library.
            let _ = std::fs::remove_file(&dylib);
            println!("cargo:rustc-cfg=mojo_unavailable");
        }
    }
}

fn build(mojo: &Path, src: &Path, dylib: &Path, out: &Path) -> Result<(), String> {
    if !mojo.is_file() {
        return Err(format!("compiler not found at {} (set FROST_MOJO_BIN)", mojo.display()));
    }
    // <root>/bin/mojo -> <root>; the driver needs the stdlib import path and a writable cache.
    let root = mojo.parent().and_then(Path::parent).ok_or("bad compiler path")?;
    let modular_home = out.join("modular_home");
    std::fs::create_dir_all(&modular_home).map_err(|e| e.to_string())?;

    let output = Command::new(mojo)
        .env_clear()
        .env("PATH", "/usr/bin:/bin") // cc/ld for the link step; nothing else
        .env("TMPDIR", out)
        .env("MODULAR_HOME", &modular_home)
        .env("MODULAR_MOJO_MAX_PACKAGE_ROOT", root)
        .env("MODULAR_MOJO_MAX_IMPORT_PATH", root.join("lib/mojo"))
        .env("MOJO_PYTHON_LIBRARY", "/nonexistent") // hard guarantee: no libpython, ever
        .args(["build", "--emit", "shared-lib", "-O3"])
        // The driver bakes an absolute rpath to <root>/lib; add @loader_path so a bundled copy
        // finds the four runtime dylibs (KGENCompilerRTShared, AsyncRTMojoBindings,
        // MSupportGlobals, AsyncRTRuntimeGlobals) placed beside it in Contents/Frameworks.
        .args(["-Xlinker", "-rpath", "-Xlinker", "@loader_path"])
        .arg(src)
        .arg("-o")
        .arg(dylib)
        .output()
        .map_err(|e| format!("failed to spawn {}: {e}", mojo.display()))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let err: Vec<&str> = err.lines().filter(|l| !l.contains("Crashpad")).collect();
        return Err(format!("mojo build failed ({}):\n{}", output.status, err.join("\n")));
    }
    if !dylib.is_file() {
        return Err("mojo build reported success but produced no dylib".into());
    }
    Ok(())
}
