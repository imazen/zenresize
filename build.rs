fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Required by tango-bench: export symbols for dynamic linking from benchmarks.
    // Only emit when the benches/ directory is present (i.e. not in a published package
    // where benches/ is excluded from the tarball — cargo errors on this directive
    // when no benchmark targets exist in the package).
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let benches_dir = std::path::Path::new(&manifest_dir).join("benches");
    if benches_dir.exists() {
        println!("cargo:rustc-link-arg-benches=-rdynamic");
    }
}
