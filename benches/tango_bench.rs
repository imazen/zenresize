//! Tango paired benchmarks for regression detection.
//!
//! Usage:
//!   # Save baseline:
//!   cargo export target/benchmarks -- bench --bench tango_bench
//!
//!   # Compare after changes:
//!   cargo bench --bench tango_bench -- compare target/benchmarks/tango_bench
//!
//!   # Quick 1-second comparison:
//!   cargo bench --bench tango_bench -- compare target/benchmarks/tango_bench -t 1
//!
//!   # With pic-scale SIMD (for cross-library comparison):
//!   cargo export target/benchmarks -- bench --bench tango_bench --features bench-simd-competitors
//!   cargo bench --bench tango_bench --features bench-simd-competitors -- compare target/benchmarks/tango_bench

use std::hint::black_box;
use tango_bench::{IntoBenchmarks, benchmark_fn, tango_benchmarks, tango_main};

// ---------------------------------------------------------------------------
// Test image
// ---------------------------------------------------------------------------

struct TestImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn test_image(w: u32, h: u32) -> TestImage {
    let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * 4;
            rgba[i] = (x % 256) as u8;
            rgba[i + 1] = (y % 256) as u8;
            rgba[i + 2] = ((x + y) % 256) as u8;
            rgba[i + 3] = 255;
        }
    }
    TestImage {
        width: w,
        height: h,
        rgba,
    }
}

// ---------------------------------------------------------------------------
// Zenresize benchmarks
// ---------------------------------------------------------------------------

fn zen_benchmarks() -> impl IntoBenchmarks {
    [
        // Primary: sRGB downscale 50%, 1024→512
        benchmark_fn("zen/srgb/down50/1024", |b| {
            let img = test_image(1024, 1024);
            b.iter(move || {
                let config = zenresize::ResizeConfig::builder(img.width, img.height, 512, 512)
                    .filter(zenresize::Filter::Lanczos)
                    .format(zenresize::PixelDescriptor::RGBX8_SRGB)
                    .srgb()
                    .build();
                black_box(zenresize::Resizer::new(&config).resize(&img.rgba))
            })
        }),
        // sRGB downscale 50%, 576→288
        benchmark_fn("zen/srgb/down50/576", |b| {
            let img = test_image(576, 576);
            b.iter(move || {
                let config = zenresize::ResizeConfig::builder(img.width, img.height, 288, 288)
                    .filter(zenresize::Filter::Lanczos)
                    .format(zenresize::PixelDescriptor::RGBX8_SRGB)
                    .srgb()
                    .build();
                black_box(zenresize::Resizer::new(&config).resize(&img.rgba))
            })
        }),
        // Linear downscale 50% (alpha-premultiply path)
        benchmark_fn("zen/linear/down50/1024", |b| {
            let img = test_image(1024, 1024);
            b.iter(move || {
                let config = zenresize::ResizeConfig::builder(img.width, img.height, 512, 512)
                    .filter(zenresize::Filter::Lanczos)
                    .format(zenresize::PixelDescriptor::RGBA8_SRGB)
                    .linear()
                    .build();
                black_box(zenresize::Resizer::new(&config).resize(&img.rgba))
            })
        }),
        // sRGB upscale 200%, 256→512
        benchmark_fn("zen/srgb/up200/256", |b| {
            let img = test_image(256, 256);
            b.iter(move || {
                let config = zenresize::ResizeConfig::builder(img.width, img.height, 512, 512)
                    .filter(zenresize::Filter::Lanczos)
                    .format(zenresize::PixelDescriptor::RGBX8_SRGB)
                    .srgb()
                    .build();
                black_box(zenresize::Resizer::new(&config).resize(&img.rgba))
            })
        }),
        // Cached-weights Resizer path
        benchmark_fn("zen/resizer/srgb/down50/1024", |b| {
            let img = test_image(1024, 1024);
            let config = zenresize::ResizeConfig::builder(img.width, img.height, 512, 512)
                .filter(zenresize::Filter::Lanczos)
                .format(zenresize::PixelDescriptor::RGBX8_SRGB)
                .srgb()
                .build();
            let mut resizer = zenresize::Resizer::new(&config);
            b.iter(move || black_box(resizer.resize(&img.rgba)))
        }),
    ]
}

// ---------------------------------------------------------------------------
// Competitor benchmarks (for cross-library paired comparison)
// ---------------------------------------------------------------------------

fn competitor_benchmarks() -> impl IntoBenchmarks {
    let simd_label = if cfg!(feature = "bench-simd-competitors") {
        "simd"
    } else {
        "scalar"
    };

    vec![
        benchmark_fn(format!("picscale_{simd_label}/srgb/down50/1024"), |b| {
            let img = test_image(1024, 1024);
            b.iter(move || {
                use pic_scale::*;
                let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
                scaler.set_threading_policy(ThreadingPolicy::Single);
                let src_size = ImageSize::new(img.width as usize, img.height as usize);
                let dst_size = ImageSize::new(512, 512);
                let store = ImageStore::<u8, 4>::from_slice(
                    &img.rgba,
                    img.width as usize,
                    img.height as usize,
                )
                .unwrap();
                let mut dst = ImageStoreMut::<u8, 4>::alloc(512, 512);
                let plan = scaler.plan_rgba_resampling(src_size, dst_size, true).unwrap();
                let _ = plan.resample(&store, &mut dst);
                black_box(dst.as_bytes().to_vec())
            })
        }),
        benchmark_fn("picscale_safe/srgb/down50/1024", |b| {
            let img = test_image(1024, 1024);
            b.iter(move || {
                use pic_scale_safe::*;
                let src_size = ImageSize::new(img.width as usize, img.height as usize);
                let dst_size = ImageSize::new(512, 512);
                let mut src = img.rgba.clone();
                image_to_linear::<4>(&mut src, TransferFunction::Srgb);
                let mut result =
                    resize_rgba8(&src, src_size, dst_size, ResamplingFunction::Lanczos3).unwrap();
                linear_to_gamma_image::<4>(&mut result, TransferFunction::Srgb);
                black_box(result)
            })
        }),
        benchmark_fn("fir/srgb/down50/1024", |b| {
            let img = test_image(1024, 1024);
            b.iter(move || {
                use fast_image_resize as fir;
                use fir::images::{Image, ImageRef};
                use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
                let src = ImageRef::new(img.width, img.height, &img.rgba, PixelType::U8x4).unwrap();
                let mut dst = Image::new(512, 512, PixelType::U8x4);
                let mut resizer = Resizer::new();
                let opts =
                    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
                resizer.resize(&src, &mut dst, &opts).unwrap();
                black_box(dst.into_vec())
            })
        }),
    ]
}

tango_benchmarks!(zen_benchmarks(), competitor_benchmarks());
tango_main!();
