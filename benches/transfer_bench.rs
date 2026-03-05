//! Benchmark comparing transfer function approaches:
//! - powf baseline: libm powf() scalar loops (auto-vectorized by LLVM)
//! - zenresize: rational polynomial / fast_powf via AVX2 SIMD batch
//! - colorutils-rs: erydanos-based pow_fast (for sRGB/BT.709/PQ/HLG)
//!
//! All batch benchmarks process 1000 RGBA pixels (4000 f32 values).

use criterion::{Criterion, criterion_group, criterion_main};
use zenresize::{Bt709, Hlg, Pq, Srgb, TransferCurve};

const ROW_LEN: usize = 4000; // 1000 RGBA pixels

/// Generate test data: f32 values in [0, 1] (encoded/signal space)
fn test_row() -> Vec<f32> {
    (0..ROW_LEN)
        .map(|i| i as f32 / ROW_LEN as f32)
        .collect()
}

/// Generate test data: linear f32 values for from_linear benchmarks
fn linear_row() -> Vec<f32> {
    (0..ROW_LEN)
        .map(|i| (i as f32 / ROW_LEN as f32).powi(2)) // squared for more realistic linear distribution
        .collect()
}

/// Generate test data: u8 RGBA image (1000 pixels)
fn test_u8_rgba() -> Vec<u8> {
    (0..ROW_LEN)
        .map(|i| (i % 256) as u8)
        .collect()
}

// ============================================================================
// Scalar powf baseline — simulates the old code path
// ============================================================================

fn srgb_to_linear_powf(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_from_linear_powf(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

const BT709_ALPHA: f32 = 0.09929682680944;
const BT709_BETA: f32 = 0.018053968510807;

fn bt709_to_linear_powf(v: f32) -> f32 {
    if v < 4.5 * BT709_BETA {
        v / 4.5
    } else {
        ((v + BT709_ALPHA) / (1.0 + BT709_ALPHA)).powf(1.0 / 0.45)
    }
}

fn bt709_from_linear_powf(v: f32) -> f32 {
    if v < BT709_BETA {
        4.5 * v
    } else {
        (1.0 + BT709_ALPHA) * v.powf(0.45) - BT709_ALPHA
    }
}

const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;

fn pq_to_linear_powf(v: f32) -> f32 {
    if v <= 0.0 { return 0.0; }
    let vp = v.powf(1.0 / PQ_M2);
    let num = (vp - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * vp;
    if den <= 0.0 { return 1.0; }
    (num / den).powf(1.0 / PQ_M1)
}

fn pq_from_linear_powf(v: f32) -> f32 {
    if v <= 0.0 { return 0.0; }
    let vp = v.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * vp;
    let den = 1.0 + PQ_C3 * vp;
    (num / den).powf(PQ_M2)
}

const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.55991073;

fn hlg_to_linear_powf(v: f32) -> f32 {
    if v <= 0.0 { 0.0 }
    else if v <= 0.5 { (v * v) / 3.0 }
    else { (((v - HLG_C) / HLG_A).exp() + HLG_B) / 12.0 }
}

fn hlg_from_linear_powf(v: f32) -> f32 {
    if v <= 0.0 { 0.0 }
    else if v <= 1.0 / 12.0 { (3.0 * v).sqrt() }
    else { HLG_A * (12.0 * v - HLG_B).ln() + HLG_C }
}

// ============================================================================
// Batch f32→f32 to_linear: powf vs zenresize SIMD vs colorutils-rs scalar
// ============================================================================

fn bench_batch_to_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_to_linear_f32");
    let encoded_row = test_row();

    // --- sRGB ---
    group.bench_function("srgb/powf_loop", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] { *v = srgb_to_linear_powf(*v); }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("srgb/zenresize", |b| {
        let srgb = Srgb;
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            srgb.f32_to_linear_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.bench_function("srgb/colorutils", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = colorutils_rs::srgb_to_linear(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    // --- BT.709 ---
    group.bench_function("bt709/powf_loop", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] { *v = bt709_to_linear_powf(*v); }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("bt709/zenresize", |b| {
        let bt709 = Bt709;
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            bt709.f32_to_linear_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.bench_function("bt709/colorutils", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = colorutils_rs::rec709_to_linear(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    // --- PQ ---
    group.bench_function("pq/powf_loop", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] { *v = pq_to_linear_powf(*v); }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("pq/zenresize", |b| {
        let pq = Pq;
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            pq.f32_to_linear_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.bench_function("pq/colorutils", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = colorutils_rs::pq_to_linear(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    // --- HLG ---
    group.bench_function("hlg/powf_loop", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] { *v = hlg_to_linear_powf(*v); }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("hlg/zenresize", |b| {
        let hlg = Hlg;
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            hlg.f32_to_linear_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.bench_function("hlg/colorutils", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = colorutils_rs::hlg_to_linear(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.finish();
}

// ============================================================================
// Batch f32→f32 from_linear: powf vs zenresize SIMD vs colorutils-rs scalar
// ============================================================================

fn bench_batch_from_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_from_linear_f32");
    let linear_data = linear_row();

    // --- sRGB ---
    group.bench_function("srgb/powf_loop", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] { *v = srgb_from_linear_powf(*v); }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("srgb/zenresize", |b| {
        let srgb = Srgb;
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            srgb.linear_to_f32_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.bench_function("srgb/colorutils", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = colorutils_rs::srgb_from_linear(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    // --- BT.709 ---
    group.bench_function("bt709/powf_loop", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] { *v = bt709_from_linear_powf(*v); }
            }
            std::hint::black_box(&row);
        });
        });

    group.bench_function("bt709/zenresize", |b| {
        let bt709 = Bt709;
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            bt709.linear_to_f32_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.bench_function("bt709/colorutils", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = colorutils_rs::rec709_from_linear(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    // --- PQ ---
    group.bench_function("pq/powf_loop", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] { *v = pq_from_linear_powf(*v); }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("pq/zenresize", |b| {
        let pq = Pq;
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            pq.linear_to_f32_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.bench_function("pq/colorutils", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = colorutils_rs::pq_from_linear(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    // --- HLG ---
    group.bench_function("hlg/powf_loop", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] { *v = hlg_from_linear_powf(*v); }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("hlg/zenresize", |b| {
        let hlg = Hlg;
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            hlg.linear_to_f32_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.bench_function("hlg/colorutils", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = colorutils_rs::hlg_from_linear(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.finish();
}

// ============================================================================
// Batch u8→f32: colorutils-rs image API vs zenresize
// ============================================================================

fn bench_u8_to_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("u8_to_linear_f32");
    let input = test_u8_rgba();

    group.bench_function("srgb/zenresize", |b| {
        let srgb = Srgb;
        let luts = srgb.build_luts();
        let mut dst = vec![0.0f32; ROW_LEN];
        b.iter(|| {
            srgb.u8_to_linear_f32(&input, &mut dst, &luts, 4, true, false);
            std::hint::black_box(&dst);
        });
    });

    group.bench_function("srgb/colorutils", |b| {
        let mut dst = vec![0.0f32; ROW_LEN];
        let src_stride = 1000 * 4; // 1000 RGBA pixels * 4 bytes/pixel
        let dst_stride = 1000 * 4 * 4; // 1000 RGBA pixels * 4 channels * 4 bytes/f32
        b.iter(|| {
            colorutils_rs::rgba_to_linear(
                &input, src_stride,
                &mut dst, dst_stride,
                1000, 1,
                colorutils_rs::TransferFunction::Srgb,
            );
            std::hint::black_box(&dst);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_batch_to_linear, bench_batch_from_linear, bench_u8_to_linear);
criterion_main!(benches);
