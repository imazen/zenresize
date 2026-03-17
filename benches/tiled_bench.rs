//! Benchmark: tiled vs non-tiled fullframe V-filter.
//!
//! Tests the V-filter in isolation on the intermediate buffer at various
//! image sizes and tile widths. The tiled variant processes column strips
//! to keep shared input rows hot in L1 cache across consecutive output rows.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use zenresize::filter::{Filter, InterpolationDetails};
use zenresize::weights::I16WeightTable;

struct VFilterSetup {
    name: &'static str,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    intermediate: Vec<u8>,
    v_weights: I16WeightTable,
    h_row_len: usize,
}

impl VFilterSetup {
    fn new(name: &'static str, in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> Self {
        let channels = 4usize;
        let h_row_len = out_w as usize * channels;
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let v_weights = I16WeightTable::new(in_h, out_h, &filter);

        // Synthetic intermediate buffer (H-filtered data).
        let intermediate_len = h_row_len * in_h as usize;
        let mut intermediate = vec![0u8; intermediate_len];
        for (i, v) in intermediate.iter_mut().enumerate() {
            *v = (i % 256) as u8;
        }

        Self {
            name,
            in_w,
            in_h,
            out_w,
            out_h,
            intermediate,
            v_weights,
            h_row_len,
        }
    }
}

fn bench_v_filter(c: &mut Criterion) {
    let setups = vec![
        // Small: 1024→512 (h_row_len=2048, ~12KB per tap set, fits L1)
        VFilterSetup::new("1024sq_2x", 1024, 1024, 512, 512),
        // Medium: 2048→1024 (h_row_len=4096, ~24KB per tap set, L1 boundary)
        VFilterSetup::new("2048sq_2x", 2048, 2048, 1024, 1024),
        // Large: 4000→800 (h_row_len=3200, ~19KB per tap set, many taps for 5× down)
        VFilterSetup::new("4000x3000_5x", 4000, 3000, 800, 600),
        // Wide: 3840→1920 (h_row_len=7680, ~46KB per tap set, exceeds L1)
        VFilterSetup::new("4k_2x", 3840, 2160, 1920, 1080),
        // Very wide: 7680→3840 (h_row_len=15360, ~92KB per tap set)
        VFilterSetup::new("8k_2x", 7680, 4320, 3840, 2160),
        // Heavy downscale: 4000→400 (large tap count, ~10×)
        VFilterSetup::new("4000x3000_10x", 4000, 3000, 400, 300),
    ];

    let tile_sizes: &[usize] = &[64, 128, 256, 512];

    let mut group = c.benchmark_group("v_filter");
    group.sample_size(30);

    for setup in &setups {
        let out_len = setup.h_row_len * setup.out_h as usize;
        let taps = setup.v_weights.max_taps;
        let h_row_len = setup.h_row_len;

        // Baseline: non-tiled
        group.bench_function(
            BenchmarkId::new("baseline", format!("{}_t{}", setup.name, taps)),
            |b| {
                let mut output = vec![0u8; out_len];
                b.iter(|| {
                    zenresize::simd::filter_v_all_u8_i16(
                        &setup.intermediate,
                        &mut output,
                        h_row_len,
                        setup.in_h as usize,
                        setup.out_h as usize,
                        &setup.v_weights,
                    );
                });
            },
        );

        // Tiled variants at different tile sizes
        for &tile_chunks in tile_sizes {
            let tile_bytes = tile_chunks * 16;
            let working_set_kb = taps * tile_bytes / 1024;
            group.bench_function(
                BenchmarkId::new(
                    format!("tiled_{}c_{}KB", tile_chunks, working_set_kb),
                    format!("{}_t{}", setup.name, taps),
                ),
                |b| {
                    let mut output = vec![0u8; out_len];
                    b.iter(|| {
                        zenresize::simd::filter_v_all_u8_i16_tiled(
                            &setup.intermediate,
                            &mut output,
                            h_row_len,
                            setup.in_h as usize,
                            setup.out_h as usize,
                            &setup.v_weights,
                            tile_chunks,
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

/// Correctness check: verify tiled output matches baseline exactly.
fn bench_correctness(c: &mut Criterion) {
    let setup = VFilterSetup::new("correctness_4k", 3840, 2160, 1920, 1080);
    let out_len = setup.h_row_len * setup.out_h as usize;

    let mut baseline = vec![0u8; out_len];
    zenresize::simd::filter_v_all_u8_i16(
        &setup.intermediate,
        &mut baseline,
        setup.h_row_len,
        setup.in_h as usize,
        setup.out_h as usize,
        &setup.v_weights,
    );

    for &tile_chunks in &[64, 128, 256, 512] {
        let mut tiled = vec![0u8; out_len];
        zenresize::simd::filter_v_all_u8_i16_tiled(
            &setup.intermediate,
            &mut tiled,
            setup.h_row_len,
            setup.in_h as usize,
            setup.out_h as usize,
            &setup.v_weights,
            tile_chunks,
        );
        assert_eq!(baseline, tiled, "Tiled output (tile_chunks={}) differs from baseline!", tile_chunks);
    }

    c.bench_function("correctness_verified", |b| {
        b.iter(|| std::hint::black_box(42));
    });
}

criterion_group!(benches, bench_v_filter, bench_correctness);
criterion_main!(benches);
