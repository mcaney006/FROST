// Compile the Objective-C shim that reads NSProcessInfo thermal + memory state.
fn main() {
    cc::Build::new()
        .file("src/platform_mac.m")
        .flag("-fobjc-arc")
        .compile("frost_platform_mac");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rerun-if-changed=src/platform_mac.m");
}
