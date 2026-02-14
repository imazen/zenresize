//! Performance sweep across sizes and ratios.
//! Outputs CSV to stdout for charting.
//!
//! Usage:
//!   cargo bench --bench sweep_bench 2>/dev/null > /tmp/sweep_safe.csv
//!   cargo bench --bench sweep_bench --features unsafe_kernels 2>/dev/null > /tmp/sweep_unsafe.csv

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

fn bench_srgb(rgba: &[u8], w: u32, h: u32, out_w: u32, out_h: u32, iters: usize) -> (f64, f64) {
    let config = zenresize::ResizeConfig::builder(w, h, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgba))
        .srgb()
        .build();

    // Warmup
    for _ in 0..3 {
        std::hint::black_box(zenresize::resize(&config, rgba));
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let result = zenresize::resize(&config, rgba);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&result);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Trim 20% outliers (top and bottom 10%)
    let trim = times.len() / 10;
    let trimmed = &times[trim..times.len() - trim];
    let mean = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
    let stddev = if trimmed.len() > 1 {
        (trimmed.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (trimmed.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    (mean, stddev)
}

fn main() {
    let feature = if cfg!(feature = "unsafe_kernels") {
        "unsafe_kernels"
    } else {
        "safe"
    };

    // Square sizes: fine-grained sweep
    let sizes: Vec<u32> = vec![
        64, 128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 1920, 2048, 2560, 3072, 3840,
        4096, 5120, 6144, 7680,
    ];

    // Ratios as (numerator, denominator)
    let ratios: &[(u32, u32, &str)] = &[
        (1, 8, "12.5%"),
        (1, 4, "25%"),
        (1, 3, "33%"),
        (1, 2, "50%"),
        (2, 3, "67%"),
        (3, 4, "75%"),
        (1, 1, "100%"),
        (3, 2, "150%"),
        (2, 1, "200%"),
        (3, 1, "300%"),
    ];

    // Header
    println!(
        "feature,in_size,in_pixels,ratio,out_w,out_h,out_pixels,mean_ms,stddev_ms,in_mpps,out_mpps"
    );

    for &size in &sizes {
        let rgba = make_gradient(size, size);
        let in_pixels = (size as f64) * (size as f64);

        for &(num, den, ratio_label) in ratios {
            let out_w = (size * num) / den;
            let out_h = (size * num) / den;
            if out_w == 0 || out_h == 0 {
                continue;
            }

            // Skip cases that would take too long or use too much memory
            let out_pixels_total = (out_w as u64) * (out_h as u64);
            if out_pixels_total > 100_000_000 {
                continue; // >100MP output
            }
            if in_pixels > 50_000_000.0 && num > den {
                continue; // skip upscale on huge inputs
            }

            let iters = if in_pixels > 10_000_000.0 {
                10
            } else if in_pixels > 1_000_000.0 {
                20
            } else {
                40
            };

            let (mean, stddev) = bench_srgb(&rgba, size, size, out_w, out_h, iters);
            let out_pixels = (out_w as f64) * (out_h as f64);
            let in_mpps = in_pixels / (mean / 1000.0) / 1e6;
            let out_mpps = out_pixels / (mean / 1000.0) / 1e6;

            println!(
                "{},{},{},{},{},{},{},{:.4},{:.4},{:.2},{:.2}",
                feature,
                size,
                in_pixels as u64,
                ratio_label,
                out_w,
                out_h,
                out_pixels as u64,
                mean,
                stddev,
                in_mpps,
                out_mpps,
            );

            // Flush each line immediately for progress monitoring
            use std::io::Write;
            std::io::stdout().flush().ok();
        }

        // Print progress to stderr
        eprint!("  done: {}x{}\r", size, size);
    }
    eprintln!();
}
