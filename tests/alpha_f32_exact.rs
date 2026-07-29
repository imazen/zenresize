//! The f32 premultiply/unpremultiply row kernels must be bit-identical to the
//! scalar bodies they replaced.
//!
//! Both were plain scalar loops in every dispatch tier — no SIMD anywhere — so
//! the NEON arm paid the `#[arcane]` boundary for nothing. The per-kernel bench
//! measured premultiply_alpha_row at 0.92x against its own forced-scalar tier
//! (CI [-4.3%, -2.8%]), i.e. an actual regression, and unpremultiply at 1.00x.
//!
//! The hand-written `vld4q_f32` replacements are exact by construction:
//! premultiply performs the same single `c * a` multiply, and unpremultiply the
//! same `1.0 / a` followed by `c * inv_a` — the same two roundings in the same
//! order, with the threshold branch expressed as a select over the ORIGINAL
//! lane so a below-threshold pixel is returned untouched bit-for-bit.
//!
//! "Exact by construction" is the claim this file exists to check, over values
//! chosen to break it: denormals, the exact threshold, zero of both signs,
//! infinities, and alpha outside [0, 1].

use zenresize::simd::__bench_kernels as k;

/// Verbatim transcription of the original scalar bodies.
mod reference {
    pub fn premultiply(row: &mut [f32]) {
        for pixel in row.chunks_exact_mut(4) {
            let a = pixel[3];
            pixel[0] *= a;
            pixel[1] *= a;
            pixel[2] *= a;
        }
    }
    pub fn unpremultiply(row: &mut [f32]) {
        for pixel in row.chunks_exact_mut(4) {
            let a = pixel[3];
            if a > 1.0 / 1024.0 {
                let inv_a = 1.0 / a;
                pixel[0] *= inv_a;
                pixel[1] *= inv_a;
                pixel[2] *= inv_a;
            }
        }
    }
}

/// Alpha values that sit on every boundary the kernels care about.
fn alphas() -> Vec<f32> {
    let t = 1.0f32 / 1024.0;
    vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,          // smallest normal
        f32::from_bits(1),          // smallest denormal
        t,                          // exactly the threshold — must NOT be live
        f32::from_bits(t.to_bits() - 1), // just below
        f32::from_bits(t.to_bits() + 1), // just above — must be live
        0.001,
        0.5,
        1.0,
        2.0,      // alpha > 1 (out of range but must not be special-cased)
        -1.0,     // negative alpha
        1e-20,
        1e20,
        f32::INFINITY,
    ]
}

fn channels() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        255.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        1e-30,
        1e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.123_456_79,
    ]
}

/// Compare bit patterns, not values: `-0.0 == 0.0` and `NaN != NaN` under `==`,
/// and this is precisely a test about bit-level identity.
fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: element {i} (pixel {}, ch {}) got {g:?} want {w:?}",
            i / 4,
            i % 4
        );
    }
}

fn build_row() -> Vec<f32> {
    let mut row = Vec::new();
    for &a in &alphas() {
        for c in channels().chunks(3) {
            let mut px = [0.0f32; 4];
            for (j, &v) in c.iter().enumerate() {
                px[j] = v;
            }
            px[3] = a;
            row.extend_from_slice(&px);
        }
    }
    row
}

#[test]
fn premultiply_alpha_row_bit_identical() {
    let row = build_row();
    let mut want = row.clone();
    reference::premultiply(&mut want);
    let mut got = row.clone();
    k::premultiply_alpha_row(&mut got);
    assert_bits_eq(&got, &want, "premultiply");
}

#[test]
fn unpremultiply_alpha_row_bit_identical() {
    let row = build_row();
    let mut want = row.clone();
    reference::unpremultiply(&mut want);
    let mut got = row.clone();
    k::unpremultiply_alpha_row(&mut got);
    assert_bits_eq(&got, &want, "unpremultiply");
}

/// The NEON kernels process 4 pixels per `vld4q_f32` and hand the remainder to
/// the portable body; cover every remainder length so the seam is exercised.
#[test]
fn alpha_row_tail_lengths_bit_identical() {
    let base = build_row();
    for px in 0..=13usize {
        let row: Vec<f32> = base[..px * 4].to_vec();

        let mut want = row.clone();
        reference::premultiply(&mut want);
        let mut got = row.clone();
        k::premultiply_alpha_row(&mut got);
        assert_bits_eq(&got, &want, &format!("premultiply tail px={px}"));

        let mut want = row.clone();
        reference::unpremultiply(&mut want);
        let mut got = row.clone();
        k::unpremultiply_alpha_row(&mut got);
        assert_bits_eq(&got, &want, &format!("unpremultiply tail px={px}"));
    }
}
