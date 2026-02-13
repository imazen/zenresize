//! Compare synthetic gradient vs real photo performance.
//! Tests whether pixel entropy affects SIMD kernel performance.
//!
//! cargo test --release --test real_vs_synth -- --nocapture

use std::path::Path;
use std::time::Instant;

struct TestImage {
    name: &'static str,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn make_gradient(w: u32, h: u32) -> TestImage {
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
    TestImage {
        name: "synth_gradient",
        width: w,
        height: h,
        rgba,
    }
}

fn make_uniform(w: u32, h: u32) -> TestImage {
    let rgba = vec![128u8; (w as usize) * (h as usize) * 4];
    TestImage {
        name: "synth_uniform",
        width: w,
        height: h,
        rgba,
    }
}

fn load_image(path: &Path, name: &'static str) -> Option<TestImage> {
    let img = image::open(path).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some(TestImage {
        name,
        width: w,
        height: h,
        rgba: img.into_raw(),
    })
}

fn bench_resize(img: &TestImage, out_w: u32, out_h: u32, iterations: usize) -> (f64, f64) {
    let config = zenresize::ResizeConfig::builder(img.width, img.height, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
        .srgb()
        .build();

    // Warmup
    for _ in 0..3 {
        std::hint::black_box(zenresize::resize(&config, &img.rgba));
    }

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let result = zenresize::resize(&config, &img.rgba);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&result);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Trim top/bottom 10%
    let trim = times.len() / 10;
    let trimmed = &times[trim..times.len() - trim];
    let mean = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
    let stddev = (trimmed.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (trimmed.len() - 1) as f64)
        .sqrt();
    (mean, stddev)
}

#[test]
fn real_vs_synth_comparison() {
    let corpus = Path::new("/home/lilith/work/codec-corpus");
    let iterations = 30;

    let feature = if cfg!(feature = "unsafe_kernels") {
        "unsafe_kernels"
    } else {
        "safe (default)"
    };

    println!(
        "\n=== Real vs Synthetic Image Performance ({}) ===\n",
        feature
    );

    // Test at 2048x2048 - compare synth vs real CLIC image
    {
        let clic = load_image(
            &corpus.join("clic2025/final-test/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png"),
            "clic_2048sq",
        );

        let gradient = make_gradient(2048, 2048);
        let uniform = make_uniform(2048, 2048);

        println!("--- 2048x2048, 50% downscale ---");
        println!("{:<20} {:>10} {:>10}", "Image", "Mean ms", "Stddev");
        println!("{:-<20} {:->10} {:->10}", "", "", "");

        let (m, s) = bench_resize(&gradient, 1024, 1024, iterations);
        println!("{:<20} {:>10.2} {:>10.3}", gradient.name, m, s);

        let (m, s) = bench_resize(&uniform, 1024, 1024, iterations);
        println!("{:<20} {:>10.2} {:>10.3}", uniform.name, m, s);

        if let Some(ref img) = clic {
            let (m, s) = bench_resize(img, 1024, 1024, iterations);
            println!("{:<20} {:>10.2} {:>10.3}", img.name, m, s);
        }

        println!();
        println!("--- 2048x2048, 25% downscale ---");
        println!("{:<20} {:>10} {:>10}", "Image", "Mean ms", "Stddev");
        println!("{:-<20} {:->10} {:->10}", "", "", "");

        let (m, s) = bench_resize(&gradient, 512, 512, iterations);
        println!("{:<20} {:>10.2} {:>10.3}", gradient.name, m, s);

        let (m, s) = bench_resize(&uniform, 512, 512, iterations);
        println!("{:<20} {:>10.2} {:>10.3}", uniform.name, m, s);

        if let Some(ref img) = clic {
            let (m, s) = bench_resize(img, 512, 512, iterations);
            println!("{:<20} {:>10.2} {:>10.3}", img.name, m, s);
        }
    }

    // Test at ~6K - compare synth vs real photo
    {
        let real_6k = load_image(
            &corpus.join("imageflow/test_inputs/5760_x_4320.jpg"),
            "photo_5760x4320",
        );

        println!();
        println!("--- 5760x4320 (real) vs 5760x4320 (synth), 50% downscale ---");
        println!("{:<20} {:>10} {:>10}", "Image", "Mean ms", "Stddev");
        println!("{:-<20} {:->10} {:->10}", "", "", "");

        let gradient_6k = make_gradient(5760, 4320);
        let (m, s) = bench_resize(&gradient_6k, 2880, 2160, iterations);
        println!("{:<20} {:>10.2} {:>10.3}", gradient_6k.name, m, s);

        if let Some(ref img) = real_6k {
            let (m, s) = bench_resize(img, 2880, 2160, iterations);
            println!("{:<20} {:>10.2} {:>10.3}", img.name, m, s);
        } else {
            println!("(5760x4320.jpg not found, skipping real image)");
        }

        println!();
        println!("--- 5760x4320, 25% downscale ---");
        println!("{:<20} {:>10} {:>10}", "Image", "Mean ms", "Stddev");
        println!("{:-<20} {:->10} {:->10}", "", "", "");

        let (m, s) = bench_resize(&gradient_6k, 1440, 1080, iterations);
        println!("{:<20} {:>10.2} {:>10.3}", gradient_6k.name, m, s);

        if let Some(ref img) = real_6k {
            let (m, s) = bench_resize(img, 1440, 1080, iterations);
            println!("{:<20} {:>10.2} {:>10.3}", img.name, m, s);
        }
    }

    println!("\nDone.");
}
