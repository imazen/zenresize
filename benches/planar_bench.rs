//! Benchmark: interleaved 4ch vs planar resize strategies.
//!
//! Compares four approaches at 0.5, 1, 5, 12, 24 megapixels (50% downscale):
//!   1. Interleaved Rgbx — 4ch i16 sRGB fast path, no premul (pure filtering cost)
//!   2. Interleaved Rgba — 4ch i16 sRGB fast path, with premul/unpremul
//!   3. 3 full-size planes via PlaneResizer (i16 1ch, simulates 4:4:4)
//!   4. 1 full + 2 half-size planes via PlaneResizer (i16 1ch, simulates 4:2:0)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use zenresize::{Filter, PixelFormat, PixelLayout, PlaneResizer, ResizeConfig, Resizer};

// ---------------------------------------------------------------------------
// Test dimensions: (width, height, label)
// ---------------------------------------------------------------------------

const SIZES: &[(u32, u32, &str)] = &[
    (816, 613, "0.5MP"),  // 500_208
    (1154, 866, "1MP"),   // 999_364
    (2582, 1936, "5MP"),  // 4_998_752
    (4000, 3000, "12MP"), // 12_000_000
    (6000, 4000, "24MP"), // 24_000_000
];

const FILTER: Filter = Filter::Lanczos;

// ---------------------------------------------------------------------------
// Benchmark group
// ---------------------------------------------------------------------------

fn planar_vs_interleaved(c: &mut Criterion) {
    let mut group = c.benchmark_group("planar_vs_interleaved");
    group.sample_size(10);

    for &(w, h, label) in SIZES {
        let out_w = w / 2;
        let out_h = h / 2;
        let pixels = w as u64 * h as u64;

        group.throughput(criterion::Throughput::Elements(pixels));

        // -- Rgbx interleaved (no premul) --
        {
            let input = vec![128u8; (w as usize) * (h as usize) * 4];
            let mut output = vec![0u8; (out_w as usize) * (out_h as usize) * 4];
            let config = ResizeConfig::builder(w, h, out_w, out_h)
                .filter(FILTER)
                .format(PixelFormat::Srgb8(PixelLayout::Rgbx))
                .srgb()
                .build();
            let mut resizer = Resizer::new(&config);

            group.bench_with_input(BenchmarkId::new("rgbx_interleaved", label), &(), |b, _| {
                b.iter(|| resizer.resize_into(&input, &mut output));
            });
        }

        // -- Rgba interleaved (with premul/unpremul) --
        {
            let input = vec![128u8; (w as usize) * (h as usize) * 4];
            let mut output = vec![0u8; (out_w as usize) * (out_h as usize) * 4];
            let config = ResizeConfig::builder(w, h, out_w, out_h)
                .filter(FILTER)
                .format(PixelFormat::Srgb8(PixelLayout::Rgba))
                .srgb()
                .build();
            let mut resizer = Resizer::new(&config);

            group.bench_with_input(BenchmarkId::new("rgba_interleaved", label), &(), |b, _| {
                b.iter(|| resizer.resize_into(&input, &mut output));
            });
        }

        // -- 3 full planes (4:4:4) --
        {
            let plane = vec![2048i16; (w as usize) * (h as usize)];
            let planes = [plane.clone(), plane.clone(), plane.clone()];
            let pout = vec![0i16; (out_w as usize) * (out_h as usize)];
            let mut outputs = [pout.clone(), pout.clone(), pout.clone()];
            let mut resizer = PlaneResizer::new(FILTER, w, h, out_w, out_h);

            group.bench_with_input(BenchmarkId::new("3_full_planes_444", label), &(), |b, _| {
                b.iter(|| {
                    for (p, o) in planes.iter().zip(outputs.iter_mut()) {
                        resizer.resize_plane(p, w as usize, o, out_w as usize);
                    }
                });
            });
        }

        // -- 1 full + 2 half planes (4:2:0) --
        {
            let half_w = w / 2;
            let half_h = h / 2;
            let out_half_w = half_w / 2;
            let out_half_h = half_h / 2;

            let y_plane = vec![2048i16; (w as usize) * (h as usize)];
            let cb_plane = vec![2048i16; (half_w as usize) * (half_h as usize)];
            let cr_plane = vec![2048i16; (half_w as usize) * (half_h as usize)];
            let mut y_out = vec![0i16; (out_w as usize) * (out_h as usize)];
            let mut cb_out = vec![0i16; (out_half_w as usize) * (out_half_h as usize)];
            let mut cr_out = vec![0i16; (out_half_w as usize) * (out_half_h as usize)];

            let mut full_resizer = PlaneResizer::new(FILTER, w, h, out_w, out_h);
            let mut half_resizer =
                PlaneResizer::new(FILTER, half_w, half_h, out_half_w, out_half_h);

            group.bench_with_input(BenchmarkId::new("1full_2half_420", label), &(), |b, _| {
                b.iter(|| {
                    full_resizer.resize_plane(&y_plane, w as usize, &mut y_out, out_w as usize);
                    half_resizer.resize_plane(
                        &cb_plane,
                        half_w as usize,
                        &mut cb_out,
                        out_half_w as usize,
                    );
                    half_resizer.resize_plane(
                        &cr_plane,
                        half_w as usize,
                        &mut cr_out,
                        out_half_w as usize,
                    );
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// f32 benchmark group
// ---------------------------------------------------------------------------

fn f32_planar_vs_interleaved(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_planar_vs_interleaved");
    group.sample_size(10);

    for &(w, h, label) in SIZES {
        let out_w = w / 2;
        let out_h = h / 2;
        let pixels = w as u64 * h as u64;

        group.throughput(criterion::Throughput::Elements(pixels));

        // -- f32 Rgbx interleaved (4ch, no premul) --
        {
            let input = vec![0.5f32; (w as usize) * (h as usize) * 4];
            let mut output = vec![0.0f32; (out_w as usize) * (out_h as usize) * 4];
            let config = ResizeConfig::builder(w, h, out_w, out_h)
                .filter(FILTER)
                .format(PixelFormat::LinearF32(PixelLayout::Rgbx))
                .build();
            let mut resizer = Resizer::new(&config);

            group.bench_with_input(BenchmarkId::new("f32_rgbx_4ch", label), &(), |b, _| {
                b.iter(|| resizer.resize_f32_into(&input, &mut output));
            });
        }

        // -- f32 Gray interleaved (1ch) × 3 --
        {
            let plane = vec![0.5f32; (w as usize) * (h as usize)];
            let planes = [plane.clone(), plane.clone(), plane.clone()];
            let pout = vec![0.0f32; (out_w as usize) * (out_h as usize)];
            let mut outputs = [pout.clone(), pout.clone(), pout.clone()];
            let config = ResizeConfig::builder(w, h, out_w, out_h)
                .filter(FILTER)
                .format(PixelFormat::LinearF32(PixelLayout::Gray))
                .build();
            let mut resizer = Resizer::new(&config);

            group.bench_with_input(BenchmarkId::new("f32_gray_3planes", label), &(), |b, _| {
                b.iter(|| {
                    for (p, o) in planes.iter().zip(outputs.iter_mut()) {
                        resizer.resize_f32_into(p, o);
                    }
                });
            });
        }

        // -- f32 Gray 1 full + 2 half (4:2:0) --
        {
            let half_w = w / 2;
            let half_h = h / 2;
            let out_half_w = half_w / 2;
            let out_half_h = half_h / 2;

            let y_plane = vec![0.5f32; (w as usize) * (h as usize)];
            let cb_plane = vec![0.5f32; (half_w as usize) * (half_h as usize)];
            let cr_plane = vec![0.5f32; (half_w as usize) * (half_h as usize)];
            let mut y_out = vec![0.0f32; (out_w as usize) * (out_h as usize)];
            let mut cb_out = vec![0.0f32; (out_half_w as usize) * (out_half_h as usize)];
            let mut cr_out = vec![0.0f32; (out_half_w as usize) * (out_half_h as usize)];

            let full_config = ResizeConfig::builder(w, h, out_w, out_h)
                .filter(FILTER)
                .format(PixelFormat::LinearF32(PixelLayout::Gray))
                .build();
            let half_config = ResizeConfig::builder(half_w, half_h, out_half_w, out_half_h)
                .filter(FILTER)
                .format(PixelFormat::LinearF32(PixelLayout::Gray))
                .build();
            let mut full_resizer = Resizer::new(&full_config);
            let mut half_resizer = Resizer::new(&half_config);

            group.bench_with_input(BenchmarkId::new("f32_gray_420", label), &(), |b, _| {
                b.iter(|| {
                    full_resizer.resize_f32_into(&y_plane, &mut y_out);
                    half_resizer.resize_f32_into(&cb_plane, &mut cb_out);
                    half_resizer.resize_f32_into(&cr_plane, &mut cr_out);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, planar_vs_interleaved, f32_planar_vs_interleaved);
criterion_main!(benches);
