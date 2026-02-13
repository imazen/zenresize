//! Stress test: resize at various sizes (up to 8K) and ratios.
//! Verifies no panics and prints timing for each combination.
//!
//! Usage:
//!   cargo test --release --test size_ratio_stress -- --nocapture
//!   cargo test --release --test size_ratio_stress --features unsafe_kernels -- --nocapture

use std::time::Instant;

fn make_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) as usize) * 4;
            rgba[i] = (x % 256) as u8;
            rgba[i + 1] = (y % 256) as u8;
            rgba[i + 2] = ((x + y) % 256) as u8;
            rgba[i + 3] = 255;
        }
    }
    rgba
}

fn resize_srgb(rgba: &[u8], w: u32, h: u32, out_w: u32, out_h: u32) -> (Vec<u8>, f64) {
    let config = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
        .srgb()
        .build();
    let start = Instant::now();
    let result = zenresize::resize(&config, rgba);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    (result, elapsed)
}

fn resize_linear(rgba: &[u8], w: u32, h: u32, out_w: u32, out_h: u32) -> (Vec<u8>, f64) {
    let config = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
        .linear()
        .build();
    let start = Instant::now();
    let result = zenresize::resize(&config, rgba);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    (result, elapsed)
}

#[test]
fn stress_sizes_and_ratios() {
    let sizes: &[(u32, u32, &str)] = &[
        (1920, 1080, "1080p"),
        (2048, 2048, "2K_sq"),
        (3840, 2160, "4K"),
        (4096, 4096, "4K_sq"),
        (7680, 4320, "8K"),
    ];

    // (numerator, denominator) for ratio
    let ratios: &[(u32, u32, &str)] = &[
        (1, 10, "10%"),
        (1, 4, "25%"),
        (1, 3, "33%"),
        (1, 2, "50%"),
        (3, 4, "75%"),
        (3, 2, "150%"),
        (2, 1, "200%"),
        (3, 1, "300%"),
    ];

    let feature = if cfg!(feature = "unsafe_kernels") {
        "unsafe_kernels"
    } else {
        "safe (default)"
    };
    println!("\n=== zenresize size/ratio stress test ({}) ===\n", feature);
    println!(
        "{:<8} {:>10} {:>6} {:>10} {:>10} {:>10}",
        "Size", "Input", "Ratio", "Output", "sRGB ms", "Linear ms"
    );
    println!("{:-<8} {:->10} {:->6} {:->10} {:->10} {:->10}", "", "", "", "", "", "");

    for &(w, h, name) in sizes {
        let rgba = make_gradient(w, h);

        for &(num, den, ratio_name) in ratios {
            let out_w = (w * num) / den;
            let out_h = (h * num) / den;

            // Skip upscales on 8K (too slow for a test)
            if w >= 7680 && num > den {
                continue;
            }
            // Skip 300% on 4K+ (would be huge)
            if w >= 3840 && num >= 3 {
                continue;
            }

            let (result_srgb, ms_srgb) = resize_srgb(&rgba, w, h, out_w, out_h);
            assert_eq!(
                result_srgb.len(),
                (out_w as usize) * (out_h as usize) * 4,
                "sRGB output size mismatch for {}@{}", name, ratio_name
            );

            // Only run linear on smaller cases (it's ~10x slower)
            let ms_linear = if w <= 2048 {
                let (result_linear, ms) = resize_linear(&rgba, w, h, out_w, out_h);
                assert_eq!(
                    result_linear.len(),
                    (out_w as usize) * (out_h as usize) * 4,
                    "Linear output size mismatch for {}@{}", name, ratio_name
                );
                format!("{:.1}", ms)
            } else {
                "skip".to_string()
            };

            println!(
                "{:<8} {:>4}x{:<5} {:>6} {:>4}x{:<5} {:>10.1} {:>10}",
                name, w, h, ratio_name, out_w, out_h, ms_srgb, ms_linear
            );
        }
    }
    println!("\nAll sizes and ratios passed.");
}

/// Test odd/prime dimensions that might trigger edge cases in chunking
#[test]
fn stress_odd_dimensions() {
    let feature = if cfg!(feature = "unsafe_kernels") {
        "unsafe_kernels"
    } else {
        "safe (default)"
    };
    println!("\n=== Odd dimension stress test ({}) ===\n", feature);

    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 1, 1, 1),
        (1, 1, 100, 100),
        (100, 100, 1, 1),
        (7, 7, 3, 3),
        (13, 17, 5, 7),
        (127, 127, 63, 63),
        (255, 255, 128, 128),
        (997, 997, 499, 499),    // prime dimensions
        (1023, 1023, 512, 512),  // just under power of 2
        (1025, 1025, 513, 513),  // just over power of 2
        (3, 3, 1000, 1000),      // extreme upscale
        (1000, 1000, 3, 3),      // extreme downscale
        (4096, 1, 2048, 1),      // single-row wide
        (1, 4096, 1, 2048),      // single-column tall
        (3840, 2160, 1920, 1080), // 4K to 1080p
        (3840, 2160, 1280, 720),  // 4K to 720p
        (3840, 2160, 640, 360),   // 4K to 360p
    ];

    for &(in_w, in_h, out_w, out_h) in cases {
        let rgba = make_gradient(in_w, in_h);
        let config = zenresize::ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            })
            .srgb()
            .build();
        let result = zenresize::resize(&config, &rgba);
        assert_eq!(
            result.len(),
            (out_w as usize) * (out_h as usize) * 4,
            "Output size mismatch for {}x{} -> {}x{}",
            in_w, in_h, out_w, out_h
        );
        println!("  OK: {}x{} -> {}x{}", in_w, in_h, out_w, out_h);
    }
    println!("\nAll odd dimensions passed.");
}
