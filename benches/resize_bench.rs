//! Benchmark zenresize against competitors: pic-scale, fast_image_resize, resize.
//!
//! All benchmarks use Lanczos3 filter, single-threaded, RGBA u8 pixels.
//! Each library is tested in sRGB mode (fast) and linear-light mode where available.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

// ---------------------------------------------------------------------------
// Synthetic image generation (real vs synth is <1% difference for SIMD kernels)
// ---------------------------------------------------------------------------

struct TestImage {
    name: &'static str,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn make_gradient(w: u32, h: u32, name: &'static str) -> TestImage {
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
        name,
        width: w,
        height: h,
        rgba,
    }
}

fn test_images() -> Vec<TestImage> {
    vec![
        make_gradient(1024, 1024, "1024sq"),
        make_gradient(576, 576, "576sq"),
    ]
}

// ---------------------------------------------------------------------------
// zenresize
// ---------------------------------------------------------------------------

fn bench_zenresize_srgb(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(img.width, img.height, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgba))
        .srgb()
        .build();
    zenresize::resize(&config, &img.rgba)
}

fn bench_zenresize_linear(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(img.width, img.height, out_w, out_h)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::Srgb8(zenresize::PixelLayout::Rgba))
        .linear()
        .build();
    zenresize::resize(&config, &img.rgba)
}

// ---------------------------------------------------------------------------
// pic-scale
// ---------------------------------------------------------------------------

fn bench_picscale_srgb(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use pic_scale::*;
    let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    scaler.set_threading_policy(ThreadingPolicy::Single);
    let store = ImageStore::<u8, 4>::from_slice(&img.rgba, img.width as usize, img.height as usize)
        .unwrap();
    let mut dst = ImageStoreMut::<u8, 4>::alloc(out_w as usize, out_h as usize);
    scaler.resize_rgba(&store, &mut dst, true);
    dst.as_bytes().to_vec()
}

fn bench_picscale_linear(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use pic_scale::*;
    let mut scaler = LinearScaler::new(ResamplingFunction::Lanczos3);
    scaler.set_threading_policy(ThreadingPolicy::Single);
    let store = ImageStore::<u8, 4>::from_slice(&img.rgba, img.width as usize, img.height as usize)
        .unwrap();
    let mut dst = ImageStoreMut::<u8, 4>::alloc(out_w as usize, out_h as usize);
    scaler.resize_rgba(&store, &mut dst, true);
    dst.as_bytes().to_vec()
}

// ---------------------------------------------------------------------------
// fast_image_resize
// ---------------------------------------------------------------------------

fn bench_fir_srgb(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use fast_image_resize as fir;
    use fir::images::{Image, ImageRef};
    use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

    let src = ImageRef::new(img.width, img.height, &img.rgba, PixelType::U8x4).unwrap();
    let mut dst = Image::new(out_w, out_h, PixelType::U8x4);
    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer.resize(&src, &mut dst, &opts).unwrap();
    dst.into_vec()
}

fn bench_fir_linear(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use fast_image_resize as fir;
    use fir::images::{Image, ImageRef};
    use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

    let mapper = fir::create_srgb_mapper();
    let src = ImageRef::new(img.width, img.height, &img.rgba, PixelType::U8x4).unwrap();

    // sRGB u8 -> linear u16
    let mut linear_src = Image::new(img.width, img.height, PixelType::U16x4);
    mapper.forward_map(&src, &mut linear_src).unwrap();

    // resize in linear
    let mut linear_dst = Image::new(out_w, out_h, PixelType::U16x4);
    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer.resize(&linear_src, &mut linear_dst, &opts).unwrap();

    // linear u16 -> sRGB u8
    let mut dst = Image::new(out_w, out_h, PixelType::U8x4);
    mapper.backward_map(&linear_dst, &mut dst).unwrap();
    dst.into_vec()
}

// ---------------------------------------------------------------------------
// resize crate
// ---------------------------------------------------------------------------

fn bench_resize_crate_srgb(img: &TestImage, out_w: u32, out_h: u32) -> Vec<u8> {
    use rgb::FromSlice;

    let mut dst = vec![0u8; out_w as usize * out_h as usize * 4];
    let mut resizer = resize::new(
        img.width as usize,
        img.height as usize,
        out_w as usize,
        out_h as usize,
        resize::Pixel::RGBA8P,
        resize::Type::Lanczos3,
    )
    .unwrap();
    resizer
        .resize(img.rgba.as_rgba(), dst.as_rgba_mut())
        .unwrap();
    dst
}

// ---------------------------------------------------------------------------
// Criterion benchmarks
// ---------------------------------------------------------------------------

fn downscale_50(c: &mut Criterion) {
    let images = test_images();

    let mut group = c.benchmark_group("downscale_50pct");
    group.sample_size(20);

    for img in &images {
        let out_w = img.width / 2;
        let out_h = img.height / 2;
        let megapixels = (img.width as f64 * img.height as f64) / 1e6;
        let param = format!("{}_{}", img.name, img.width);

        group.throughput(criterion::Throughput::Elements(
            (img.width as u64) * (img.height as u64),
        ));

        group.bench_with_input(
            BenchmarkId::new("zenresize_srgb", &param),
            &megapixels,
            |b, _| b.iter(|| bench_zenresize_srgb(img, out_w, out_h)),
        );
        group.bench_with_input(
            BenchmarkId::new("zenresize_linear", &param),
            &megapixels,
            |b, _| b.iter(|| bench_zenresize_linear(img, out_w, out_h)),
        );
        group.bench_with_input(
            BenchmarkId::new("pic_scale_srgb", &param),
            &megapixels,
            |b, _| b.iter(|| bench_picscale_srgb(img, out_w, out_h)),
        );
        group.bench_with_input(
            BenchmarkId::new("pic_scale_linear", &param),
            &megapixels,
            |b, _| b.iter(|| bench_picscale_linear(img, out_w, out_h)),
        );
        group.bench_with_input(BenchmarkId::new("fir_srgb", &param), &megapixels, |b, _| {
            b.iter(|| bench_fir_srgb(img, out_w, out_h))
        });
        group.bench_with_input(
            BenchmarkId::new("fir_linear", &param),
            &megapixels,
            |b, _| b.iter(|| bench_fir_linear(img, out_w, out_h)),
        );
        group.bench_with_input(
            BenchmarkId::new("resize_crate_srgb", &param),
            &megapixels,
            |b, _| b.iter(|| bench_resize_crate_srgb(img, out_w, out_h)),
        );
    }
    group.finish();
}

fn downscale_25(c: &mut Criterion) {
    let images = test_images();

    let mut group = c.benchmark_group("downscale_25pct");
    group.sample_size(20);

    for img in &images {
        let out_w = img.width / 4;
        let out_h = img.height / 4;
        let param = format!("{}_{}", img.name, img.width);

        group.throughput(criterion::Throughput::Elements(
            (img.width as u64) * (img.height as u64),
        ));

        group.bench_with_input(BenchmarkId::new("zenresize_srgb", &param), img, |b, img| {
            b.iter(|| bench_zenresize_srgb(img, out_w, out_h))
        });
        group.bench_with_input(
            BenchmarkId::new("zenresize_linear", &param),
            img,
            |b, img| b.iter(|| bench_zenresize_linear(img, out_w, out_h)),
        );
        group.bench_with_input(BenchmarkId::new("pic_scale_srgb", &param), img, |b, img| {
            b.iter(|| bench_picscale_srgb(img, out_w, out_h))
        });
        group.bench_with_input(
            BenchmarkId::new("pic_scale_linear", &param),
            img,
            |b, img| b.iter(|| bench_picscale_linear(img, out_w, out_h)),
        );
        group.bench_with_input(BenchmarkId::new("fir_srgb", &param), img, |b, img| {
            b.iter(|| bench_fir_srgb(img, out_w, out_h))
        });
        group.bench_with_input(BenchmarkId::new("fir_linear", &param), img, |b, img| {
            b.iter(|| bench_fir_linear(img, out_w, out_h))
        });
        group.bench_with_input(
            BenchmarkId::new("resize_crate_srgb", &param),
            img,
            |b, img| b.iter(|| bench_resize_crate_srgb(img, out_w, out_h)),
        );
    }
    group.finish();
}

fn upscale_200(c: &mut Criterion) {
    let images = test_images();

    let mut group = c.benchmark_group("upscale_200pct");
    group.sample_size(10);

    // Use the smaller image for upscaling to keep bench time reasonable
    let img = images.iter().min_by_key(|i| i.width * i.height).unwrap();
    let out_w = img.width * 2;
    let out_h = img.height * 2;
    let param = format!("{}_{}", img.name, img.width);

    group.throughput(criterion::Throughput::Elements(
        (out_w as u64) * (out_h as u64),
    ));

    group.bench_with_input(BenchmarkId::new("zenresize_srgb", &param), img, |b, img| {
        b.iter(|| bench_zenresize_srgb(img, out_w, out_h))
    });
    group.bench_with_input(
        BenchmarkId::new("zenresize_linear", &param),
        img,
        |b, img| b.iter(|| bench_zenresize_linear(img, out_w, out_h)),
    );
    group.bench_with_input(BenchmarkId::new("pic_scale_srgb", &param), img, |b, img| {
        b.iter(|| bench_picscale_srgb(img, out_w, out_h))
    });
    group.bench_with_input(
        BenchmarkId::new("pic_scale_linear", &param),
        img,
        |b, img| b.iter(|| bench_picscale_linear(img, out_w, out_h)),
    );
    group.bench_with_input(BenchmarkId::new("fir_srgb", &param), img, |b, img| {
        b.iter(|| bench_fir_srgb(img, out_w, out_h))
    });
    group.bench_with_input(BenchmarkId::new("fir_linear", &param), img, |b, img| {
        b.iter(|| bench_fir_linear(img, out_w, out_h))
    });
    group.bench_with_input(
        BenchmarkId::new("resize_crate_srgb", &param),
        img,
        |b, img| b.iter(|| bench_resize_crate_srgb(img, out_w, out_h)),
    );

    group.finish();
}

criterion_group!(benches, downscale_50, downscale_25, upscale_200);
criterion_main!(benches);
