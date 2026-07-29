//! `filter_h_row_f32_to_f16` must be bit-identical to the scalar formulation
//! it replaced.
//!
//! That kernel's convolution used to run fully scalar — a `/` and a `%` per
//! output element — while its f32 sibling `filter_h_4ch` vectorized over the
//! RGBA pixel. The per-kernel tier bench caught the consequence: NEON was
//! 5.2-7.0% SLOWER than its own scalar tier (CI excluding zero), where the
//! f32 sibling was 6.34x faster.
//!
//! Vectorizing it is only safe if the accumulator is bit-identical. It should
//! be: for `channels == 4` a flat chunk of 4 IS one output pixel, and lane `c`
//! performs the same multiply-add sequence in the same order that the scalar
//! closure performed for element `i + c`. This test asserts that rather than
//! assuming it — a 1-ULP f32 drift usually vanishes under f16 rounding, which
//! is exactly what would make such a regression escape a tolerance-based test.
//!
//! The reference below is the ORIGINAL scalar body, transcribed verbatim, so
//! the comparison is against the pre-change behaviour and not against the new
//! code restated.

use zenresize::filter::{Filter, InterpolationDetails};
use zenresize::simd::__bench_kernels as k;
use zenresize::weights::F32WeightTable;

/// Half-precision encode matching `scalar::f32_to_f16_soft` via the crate's
/// own round-trip: we compare bit patterns produced by the kernel, so the
/// reference must produce them the same way. Rather than duplicate the soft
/// encoder, the reference accumulates in scalar f32 and defers to the crate's
/// own f32->f16 row kernel for the conversion.
fn reference(input: &[f32], out_width: usize, weights: &F32WeightTable, channels: usize) -> Vec<u16> {
    let total = out_width * channels;
    // Verbatim from the original implementation.
    let acc_at = |i: usize| -> f32 {
        let out_x = i / channels;
        let c = i % channels;
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let mut acc = 0.0f32;
        for (t, &weight) in w.iter().enumerate() {
            acc += input[(left + t) * channels + c] * weight;
        }
        acc
    };
    let accs: Vec<f32> = (0..total).map(acc_at).collect();
    let mut out = vec![0u16; total];
    k::f32_to_f16_row(&accs, &mut out);
    out
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f32_unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
}

/// Sweep real resize ratios (down, up, near-identity) x every filter whose
/// tap count differs, asserting exact u16 equality on every output element.
#[test]
fn f16_h_filter_bit_identical_to_scalar_formulation() {
    let mut rng = Rng(0xF16_0BEEF);
    let ratios: &[(u32, u32)] = &[
        (1920, 960),  // 2:1 down — the common web case
        (1920, 1919), // near-identity, max taps
        (1024, 256),  // 4:1 down
        (256, 1024),  // 4:1 up
        (100, 33),    // non-power-of-2 down
        (33, 100),    // non-power-of-2 up
        (17, 16),     // tiny, odd
    ];
    let filters = [
        Filter::Lanczos,
        Filter::LanczosSharp,
        Filter::Triangle,
        Filter::Mitchell,
        Filter::CatmullRom,
    ];

    let mut cases = 0usize;
    for &(in_w, out_w) in ratios {
        for &f in &filters {
            let det = InterpolationDetails::create(f);
            let w = F32WeightTable::new(in_w, out_w, &det);
            // Pad the row: the 4ch kernel reads to `max_taps`.
            let n = in_w as usize * 4;
            let input: Vec<f32> = (0..n + 512)
                .map(|i| if i < n { rng.f32_unit() * 4.0 - 1.0 } else { 0.0 })
                .collect();

            let expect = reference(&input, out_w as usize, &w, 4);
            let mut got = vec![0u16; out_w as usize * 4];
            k::filter_h_row_f32_to_f16(&input, &mut got, &w, 4);

            assert_eq!(
                got, expect,
                "f16 H filter diverged: {in_w}->{out_w}, filter {f:?}"
            );
            cases += 1;
        }
    }
    assert!(cases >= 35, "expected a real sweep, ran {cases} cases");
}

/// The non-RGBA channel counts still take the original scalar path; pin them
/// too so a future edit cannot silently change them either.
#[test]
fn f16_h_filter_non_rgba_channels_unchanged() {
    let mut rng = Rng(0x0DD_C0DE);
    let det = InterpolationDetails::create(Filter::Lanczos);
    for channels in [1usize, 2, 3] {
        for &(in_w, out_w) in &[(640u32, 320u32), (200, 601)] {
            let w = F32WeightTable::new(in_w, out_w, &det);
            let n = in_w as usize * channels;
            let input: Vec<f32> = (0..n + 512)
                .map(|i| if i < n { rng.f32_unit() } else { 0.0 })
                .collect();
            let expect = reference(&input, out_w as usize, &w, channels);
            let mut got = vec![0u16; out_w as usize * channels];
            k::filter_h_row_f32_to_f16(&input, &mut got, &w, channels);
            assert_eq!(got, expect, "ch={channels} {in_w}->{out_w}");
        }
    }
}
