//! Compiles the AppKit shell (native/app/*.m, ARC) into a static lib linked into frost-desktop.
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../native/app");
    println!("cargo:rerun-if-changed={}", dir.display()); // new files in the directory too
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("native/app")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("m" | "h")))
        .collect();
    files.sort();
    let mut build = cc::Build::new();
    for p in &files {
        println!("cargo:rerun-if-changed={}", p.display());
        if p.extension().and_then(|e| e.to_str()) == Some("m") {
            build.file(p);
        }
    }
    build
        .flag("-fobjc-arc")
        .flag("-Wall")
        .flag("-Wextra")
        .flag("-Werror=return-type")
        .define("FROST_VERSION", format!("\"{}\"", env!("CARGO_PKG_VERSION")).as_str())
        .compile("frost_ui");
    println!("cargo:rustc-link-lib=framework=AppKit");
    println!("cargo:rustc-link-lib=framework=Foundation");
}
