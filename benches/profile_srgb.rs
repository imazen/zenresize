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
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgbx))
        .srgb()
        .build();

    // Profile with Resizer (cached weights) for pure resize cost
    let mut resizer = zenresize::Resizer::new(&config);
    for _ in 0..10 {
        let result = resizer.resize(&rgba);
        std::hint::black_box(&result);
    }
}
