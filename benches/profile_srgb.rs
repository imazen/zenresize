//! Minimal profiling binary for callgrind / perf.
//! Usage: cargo build --release --bench profile_srgb
//!        valgrind --tool=callgrind target/release/deps/profile_srgb-*

use std::path::Path;

fn main() {
    let corpus = Path::new("/home/lilith/work/codec-corpus");
    let img = image::open(corpus.join(
        "clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png",
    ))
    .unwrap()
    .to_rgba8();
    let (w, h) = img.dimensions();
    let rgba = img.into_raw();

    let config = zenresize::ResizeConfig::builder(w, h, w / 2, h / 2)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
        .srgb()
        .build();

    // Run 10 iterations to get stable counts
    for _ in 0..10 {
        let result = zenresize::resize(&config, &rgba);
        std::hint::black_box(&result);
    }
}
