fn main() {
    // Required by tango-bench: export symbols for dynamic linking from benchmarks
    println!("cargo:rustc-link-arg-benches=-rdynamic");
    println!("cargo:rerun-if-changed=build.rs");
}
