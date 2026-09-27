// Compiles the Zig action-index kernels into a static lib and links them.
// Real Rust->Zig boundary: the .a is produced by `zig build-lib` at build time.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // workspace_root/native/index/frost_index.zig
    let zig_src = manifest
        .parent().unwrap()      // crates/
        .parent().unwrap()      // workspace root
        .join("native/index/frost_index.zig");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let lib = out.join("libfrost_index.a");

    println!("cargo:rerun-if-changed={}", zig_src.display());

    let status = Command::new("zig")
        .args([
            "build-lib",
            zig_src.to_str().unwrap(),
            "-O", "ReleaseFast",
            "-lc", // std.c (open/mmap/fsync/rename) + c_allocator for the index handle
        ])
        .arg(format!("-femit-bin={}", lib.display()))
        .status()
        .expect("failed to run zig — is it on PATH? (brew install zig)");
    assert!(status.success(), "zig build-lib failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=frost_index");
}
