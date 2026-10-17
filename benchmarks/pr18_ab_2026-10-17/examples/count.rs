//! Fixed-work loop for `perf stat -e instructions`: retired instructions per
//! call are a noise-free complement to the timed A/B.
//! Usage: count <main|pr18> <srgb_i16|linear_i16|linear_f32|f32_io> <w> <h> <ow> <oh> <iters> [new-only]
use std::hint::black_box;

macro_rules! run {
    ($krate:ident, $path:expr, $w:expr, $h:expr, $ow:expr, $oh:expr, $iters:expr, $new_only:expr) => {{
        use $krate::{Filter, PixelDescriptor, ResizeConfig, Resizer};
        let b = ResizeConfig::builder($w, $h, $ow, $oh).filter(Filter::Lanczos);
        let cfg = match $path {
            "srgb_i16" => b.format(PixelDescriptor::RGBA8_SRGB).srgb(),
            "linear_i16" => b.format(PixelDescriptor::RGBX8_SRGB).linear(),
            "linear_f32" => b.format(PixelDescriptor::RGBA8_SRGB).linear(),
            _ => b.format(PixelDescriptor::RGBAF32_LINEAR),
        }
        .build();
        let bytes: Vec<u8> = (0..$w as usize * $h as usize * 4)
            .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
            .collect();
        let floats: Vec<f32> = bytes.iter().map(|&v| f32::from(v) / 255.0).collect();
        for _ in 0..$iters {
            if $new_only {
                black_box(Resizer::new(black_box(&cfg)));
            } else if $path == "f32_io" {
                black_box(Resizer::new(&cfg).resize_f32(black_box(&floats)));
            } else {
                black_box(Resizer::new(&cfg).resize(black_box(&bytes)));
            }
        }
    }};
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: Vec<u32> = a[3..8].iter().map(|s| s.parse().unwrap()).collect();
    let path = a[2].as_str();
    let new_only = a.get(8).is_some_and(|s| s == "new-only");
    match a[1].as_str() {
        "main" => run!(old, path, n[0], n[1], n[2], n[3], n[4], new_only),
        "pr18" => run!(new, path, n[0], n[1], n[2], n[3], n[4], new_only),
        _ => panic!("arm"),
    }
}
