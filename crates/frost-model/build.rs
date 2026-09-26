// Link the Homebrew mlx-c and mlx dynamic libraries.
fn main() {
    let prefix = "/opt/homebrew";
    println!("cargo:rustc-link-search=native={prefix}/lib");
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");
    // ensure the dylibs resolve at runtime without DYLD_* env
    println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");
    println!("cargo:rerun-if-changed=build.rs");
}
