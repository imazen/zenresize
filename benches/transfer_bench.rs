//! Benchmark comparing transfer function approaches:
//! - Baseline: libm powf() (current code)
//! - Fast: rational polynomial / fast_powf (fastmath.rs)
//! - colorutils-rs style: erydanos pow_fast (via generic pow_fast scalar)
//!
//! Tests both scalar loops and batch (row-at-a-time) throughput.

use criterion::{Criterion, criterion_group, criterion_main};
use zenresize::{Bt709, Hlg, Pq, Srgb, TransferCurve};

const ROW_LEN: usize = 4000; // 1000 RGBA pixels

/// Generate test data: f32 values in [0, 1]
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

// ============================================================================
// Scalar powf baseline — simulates the CURRENT code path
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
// Benchmark functions
// ============================================================================

fn bench_scalar_loop(
    c: &mut Criterion,
    name: &str,
    powf_fn: fn(f32) -> f32,
    fast_fn: fn(f32) -> f32,
    data: &[f32],
) {
    let mut group = c.benchmark_group(name);

    group.bench_function("powf", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &v in data {
                sum += std::hint::black_box(powf_fn(v));
            }
            std::hint::black_box(sum)
        });
    });

    group.bench_function("fastmath", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &v in data {
                sum += std::hint::black_box(fast_fn(v));
            }
            std::hint::black_box(sum)
        });
    });

    group.finish();
}

fn bench_batch_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_f32_to_linear_inplace");
    let encoded_row = test_row();

    // sRGB
    group.bench_function("srgb_powf", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = srgb_to_linear_powf(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("srgb_fast", |b| {
        let srgb = Srgb;
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            srgb.f32_to_linear_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    // BT.709
    group.bench_function("bt709_powf", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = bt709_to_linear_powf(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("bt709_fast", |b| {
        let bt709 = Bt709;
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            bt709.f32_to_linear_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    // PQ
    group.bench_function("pq_powf", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = pq_to_linear_powf(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("pq_fast", |b| {
        let pq = Pq;
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            pq.f32_to_linear_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    // HLG
    group.bench_function("hlg_powf", |b| {
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = hlg_to_linear_powf(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("hlg_fast", |b| {
        let hlg = Hlg;
        let mut row = encoded_row.clone();
        b.iter(|| {
            row.copy_from_slice(&encoded_row);
            hlg.f32_to_linear_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.finish();
}

fn bench_batch_from_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_linear_to_f32_inplace");
    let linear_data = linear_row();

    // sRGB
    group.bench_function("srgb_powf", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = srgb_from_linear_powf(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("srgb_fast", |b| {
        let srgb = Srgb;
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            srgb.linear_to_f32_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    // PQ
    group.bench_function("pq_powf", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = pq_from_linear_powf(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("pq_fast", |b| {
        let pq = Pq;
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            pq.linear_to_f32_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    // HLG
    group.bench_function("hlg_powf", |b| {
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            for pixel in row.chunks_exact_mut(4) {
                for v in &mut pixel[..3] {
                    *v = hlg_from_linear_powf(*v);
                }
            }
            std::hint::black_box(&row);
        });
    });

    group.bench_function("hlg_fast", |b| {
        let hlg = Hlg;
        let mut row = linear_data.clone();
        b.iter(|| {
            row.copy_from_slice(&linear_data);
            hlg.linear_to_f32_inplace(&mut row, 4, true, false);
            std::hint::black_box(&row);
        });
    });

    group.finish();
}

fn bench_scalars(c: &mut Criterion) {
    use zenresize::fastmath;

    let encoded = test_row();
    let linear = linear_row();

    bench_scalar_loop(c, "srgb_to_linear", srgb_to_linear_powf, fastmath::srgb_to_linear, &encoded);
    bench_scalar_loop(c, "srgb_from_linear", srgb_from_linear_powf, fastmath::srgb_from_linear, &linear);
    bench_scalar_loop(c, "bt709_to_linear", bt709_to_linear_powf, fastmath::bt709_to_linear, &encoded);
    bench_scalar_loop(c, "bt709_from_linear", bt709_from_linear_powf, fastmath::bt709_from_linear, &linear);
    bench_scalar_loop(c, "pq_to_linear", pq_to_linear_powf, fastmath::pq_to_linear, &encoded);
    bench_scalar_loop(c, "pq_from_linear", pq_from_linear_powf, fastmath::pq_from_linear, &linear);
    bench_scalar_loop(c, "hlg_to_linear", hlg_to_linear_powf, fastmath::hlg_to_linear, &encoded);
    bench_scalar_loop(c, "hlg_from_linear", hlg_from_linear_powf, fastmath::hlg_from_linear, &linear);
}

criterion_group!(benches, bench_scalars, bench_batch_inplace, bench_batch_from_linear);
criterion_main!(benches);
