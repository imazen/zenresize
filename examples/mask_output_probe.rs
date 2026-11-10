//! Diagnostic version of the asserting mask-output integration test.
#[path = "../tests/support/mask_outputs.rs"]
mod mask_outputs;

fn main() {
    mask_outputs::run(|case, components, bytes| {
        println!("{case}: premul_wrong_components={components} over_wrong_bytes={bytes}");
    });
}
